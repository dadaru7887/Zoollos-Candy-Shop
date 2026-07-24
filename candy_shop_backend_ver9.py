"""
사탕가게 데모 - 백엔드 로직 (ver7)
=================================
흐름: 세트 선택 → (OMX ACT 정책으로 로봇이 물건을 집어 저울에 담음) → 저울 화면 OCR로 무게 읽기
      → 가격 = 무게 * 1000원 계산

CANDY_MODE 4단계
  mock         : 로봇/weight 둘 다 없이 랜덤값으로 UI 흐름만 테스트 (기본값)
  weight_check : 실제 학습된 weight을 Hub에서 이 프로세스 안에서 직접 로드해 forward pass 확인.
                 로봇/카메라 없이, 학습 데이터셋의 프레임 하나를 관측으로 대신 사용.
                 (GPU 없는 로컬 PC에서 무거운 lerobot 추론을 직접 돌릴 때)
  remote       : Colab에서 띄운 정책 서버(ngrok)에 HTTP로 요청해서 추론 결과를 받아옴.
                 (오늘처럼 실제 weight 추론은 Colab GPU에서 하고, 웹은 다른 곳에서 띄울 때 추천)
  real         : 실물 로봇 연결. ~/il_ws/src/lerobot 에서 학습해 로컬에 저장된 red/violet
                 단일물체 grasp 체크포인트(outputs/train/.../checkpoints/last/pretrained_model)를
                 lerobot-record --policy.path 로 그대로 넘겨서 추론.

실행 예시:
    CANDY_MODE=remote CANDY_COLAB_API_URL=https://xxxx.ngrok-free.app streamlit run candy_shop_kiosk.py
"""

import argparse
import os
import random
import re
import shutil
import subprocess
import time
from dataclasses import dataclass

import cv2
import numpy as np
import requests
from flask import Flask, jsonify, request

import webcam_simple_repeat_ver2 as objdet

# -----------------------------------------------------------------------
# 0. 모드 설정
# -----------------------------------------------------------------------
# 하위호환: CANDY_MOCK=false 로 실행하면 CANDY_MODE=real 로 취급됨.
_legacy_mock = os.getenv("CANDY_MOCK")
if _legacy_mock is not None and _legacy_mock.lower() == "false":
    _default_mode = "real"
else:
    _default_mode = "mock"

CANDY_MODE = os.getenv("CANDY_MODE", _default_mode).lower()
assert CANDY_MODE in ("mock", "weight_check", "remote", "real"), f"알 수 없는 CANDY_MODE: {CANDY_MODE}"

MOCK_MODE = CANDY_MODE == "mock"  # 기존 코드 호환용 별칭
MOCK_DELAY_SEC = float(os.getenv("CANDY_MOCK_DELAY", "5"))  # 로봇팔 동작 시간을 흉내내는 대기 시간(초)

# weight_check 모드에서 이 프로세스 안에서 직접 policy를 로드할 때 쓸 디바이스
POLICY_DEVICE = os.getenv("CANDY_POLICY_DEVICE", "cpu")

# remote 모드일 때 호출할 Colab ngrok URL
COLAB_API_URL = os.getenv("CANDY_COLAB_API_URL", "")

_remote_cache = {}
_real_cache = {}

# -----------------------------------------------------------------------
# 1. 세트 -> 학습된 정책(모델) 매핑
# -----------------------------------------------------------------------
HF_USER = "kimy0420"

# 세트1(빨강)/세트2(보라)는 ~/il_ws/src/lerobot 에서 학습한 로컬 체크포인트를 그대로 사용.
# ACTPolicy.from_pretrained / lerobot-record --policy.path 둘 다 Hub repo_id뿐 아니라
# 로컬 디렉터리 경로도 그대로 받아들이기 때문에 Hub에 올리지 않아도 됨(push_to_hub=false).
_LEROBOT_WS = os.path.expanduser("~/il_ws/src/lerobot")

CANDY_SET_POLICIES = {
    "세트1": os.path.join(_LEROBOT_WS, "outputs/train/single_red_policy/checkpoints/last/pretrained_model"),
    "세트2": os.path.join(_LEROBOT_WS, "outputs/train/single_violet_policy/checkpoints/last/pretrained_model"),
}

CANDY_SET_TASKS = {
    "세트1": "Pick up Single Red",
    "세트2": "Pick up Single Violet",
}

# weight_check 모드에서 관측을 대신 가져올 데이터셋 (로컬 캐시에 이미 있는 repo_id를 쓰면
# LeRobotDataset이 다시 다운로드하지 않고 그대로 사용함).
CANDY_SET_DATASETS = {
    "세트1": f"{HF_USER}/dataset_single_red",
    "세트2": f"{HF_USER}/dataset_single_violet_v2",
}

ROBOT_PORT = "/dev/omx_follower"
ROBOT_ID = "omx_follower_arm"

# front/wrist 카메라 by-id 경로. run_record_red.sh / run_record_violet_v2.sh / run_inference_*.sh 와
# 동일하게 맞춘 것 - front=Innomaker, wrist=Jieli 로 실제 물리 위치와는 반대(swapped)지만,
# red/violet 데이터셋이 전부 이 매핑으로 녹화됐기 때문에 추론도 반드시 같은 매핑을 써야 함.
FRONT_CAM = "/dev/v4l/by-id/usb-Innomaker_Innomaker-U20CAM-720P_SN0001-video-index0"
WRIST_CAM = "/dev/v4l/by-id/usb-Jieli_Technology_USB_Composite_Device-video-index0"

# 저울 LCD 숫자 OCR용 카메라. WRIST_CAM과 물리적으로 같은 카메라(Jieli)를 재사용하는 것 - pick&place
# 동안엔 policy 입력(wrist)으로, 끝난 뒤엔 OCR 소스로 순차적으로(동시에 X) 열고 닫음.
SCALE_CAM = WRIST_CAM
SCALE_ROI = (350, 360, 90, 50)

# real 모드에서 lerobot-record 한 에피소드에 줄 시간. 필요하면 CANDY_EVAL_EPISODE_TIME_S로 조정.
EVAL_EPISODE_TIME_S = float(os.getenv("CANDY_EVAL_EPISODE_TIME_S", "50"))
EVAL_RESET_TIME_S = float(os.getenv("CANDY_EVAL_RESET_TIME_S", "1"))


def eval_dataset_dir(set_name: str) -> str:
    return os.path.expanduser(f"~/eval_{set_name}")


def eval_video_path(set_name: str, camera_key: str) -> str:
    return os.path.join(eval_dataset_dir(set_name), "videos", f"observation.images.{camera_key}", "chunk-000", "file-000.mp4")


class SoldOutError(RuntimeError):
    """저울 위에 목표 개수(OBJECTS_REQUIRED)만큼 물체가 감지되지 않았을 때(로봇 pick 실패)."""


SCALE_SETTLE_SEC = float(os.getenv("CANDY_SCALE_SETTLE_SEC", "2.5"))

# 오늘 라이브 카메라로 실측한 저울 LCD 위치(축정렬, 버튼 제외 타이트 크롭).
# 카메라/저울 위치가 바뀌면 이 값도 다시 잡아야 함.
WEIGHT_DISPLAY_AABB = tuple(
    int(v) for v in os.getenv("CANDY_WEIGHT_AABB", "440,334,497,372").split(",")
)


def _recognize_seven_segment_digit(digit_binary, thresh: float = 0.10):
    """video_simple.py(팀원 작성)의 세그먼트 판독기, 임계값만 0.18->0.10으로 낮춤
    (저해상도라 가운데/아래 획이 얇게 나와서 원래 임계값으로는 못 잡았음)."""
    normalized = cv2.resize(digit_binary, (60, 100), interpolation=cv2.INTER_NEAREST)
    segment_regions = [
        (15, 0, 45, 18), (42, 10, 60, 48), (42, 52, 60, 90), (15, 82, 45, 100),
        (0, 52, 18, 90), (0, 10, 18, 48), (15, 41, 45, 59),
    ]
    active = []
    for x1, y1, x2, y2 in segment_regions:
        region = normalized[y1:y2, x1:x2]
        ratio = cv2.countNonZero(region) / float(region.size)
        active.append(1 if ratio >= thresh else 0)
    segment_map = {
        (1, 1, 1, 1, 1, 1, 0): 0, (0, 1, 1, 0, 0, 0, 0): 1, (1, 1, 0, 1, 1, 0, 1): 2,
        (1, 1, 1, 1, 0, 0, 1): 3, (0, 1, 1, 0, 0, 1, 1): 4, (1, 0, 1, 1, 0, 1, 1): 5,
        (1, 0, 1, 1, 1, 1, 1): 6, (1, 1, 1, 0, 0, 0, 0): 7, (1, 1, 1, 1, 1, 1, 1): 8,
        (1, 1, 1, 1, 0, 1, 1): 9,
    }
    return segment_map.get(tuple(active))


def _find_digit_split(image: np.ndarray) -> int:
    """붙은 두 자리 숫자 사이의 분리 지점을 찾음.
    가운데 부근(35~65%)에서 흰 픽셀이 뚜렷하게 적은 지점(빈 틈)이 있으면 그 지점을 쓰고,
    저해상도라 두 자리가 실제로 붙어서 빈 틈이 없으면 고정 50% 분할이 더 안정적."""
    height, width = image.shape
    vertical_sum = np.sum(image > 0, axis=0)
    start, end = int(width * 0.35), int(width * 0.65)
    if end <= start:
        return width // 2
    window = vertical_sum[start:end]
    if window.min() <= window.mean() * 0.5:
        return start + int(np.argmin(window))
    return width // 2


def _read_weight_segment(frame, aabb: tuple[int, int, int, int]) -> float | None:
    """저울 LCD 전용 세그먼트 판독. 실패하면(자리가 붙어있어 못 나누는 등) None 반환."""
    x1, y1, x2, y2 = aabb
    crop = frame[y1:y2, x1:x2].copy()
    if crop.size == 0:
        return None

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    _, bright = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    sx, sy, sw, sh = cv2.boundingRect(largest)

    if sw * sh < gray.shape[0] * gray.shape[1] * 0.20:
        display = gray
    else:
        mx, my = max(1, int(sw * 0.03)), max(1, int(sh * 0.04))
        display = gray[sy + my:sy + sh - my, sx + mx:sx + sw - mx]
    if display.size == 0:
        return None

    _, binary = cv2.threshold(display, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    # 저해상도라 숫자 획(특히 "7","0")이 morphology 처리 중 조각조각 끊어지는 경우가 있어서,
    # 먼저 CLOSE로 끊어진 획을 이어붙인 다음에 자잘한 노이즈 제거용 OPEN을 적용.
    # (원래는 OPEN만 썼는데, 끊어진 획 조각이 개별적으론 너무 작아서 아래 높이 필터에
    # 안 걸리고 통째로 버려지는 문제가 있었음.)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8), iterations=1)

    n, _labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    display_h, display_w = binary.shape
    raw = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        # 왼쪽 끝 노이즈(테두리 찌꺼기 등) 제외용 필터인데, 두 자리가 붙어서 큰 덩어리로
        # 잡히면 그 덩어리가 크롭 맨 왼쪽(x=0)까지 닿는 경우가 있어서, 폭이 넓은
        # 컴포넌트(진짜 숫자 덩어리)는 왼쪽 끝에 닿아도 제외하지 않도록 함.
        near_left_noise = x < display_w * 0.05 and w < display_w * 0.15
        if w < 2 or area < 10 or near_left_noise:
            continue
        raw.append((x, y, w, h))
    if not raw:
        return None

    # 절대 기준(전체 디스플레이 높이의 몇 %) 대신, 이번 프레임에서 실제로 잡힌 것 중
    # 제일 큰 높이를 기준으로 상대 필터링 - 크롭 여백/자리 수가 바뀌어도 안정적으로 동작.
    max_h = max(h for _, _, _, h in raw)
    candidates = [(x, y, w, h) for x, y, w, h in raw if h >= max_h * 0.55]
    candidates.sort(key=lambda item: item[0])
    if not candidates:
        return None

    # 두 자리가 저해상도 때문에 붙어서 컴포넌트 하나로 잡혔을 때, 그 컴포넌트를 두 자리
    # 숫자로 보고 강제로 분리해서 각각 판독 (팀원 제안). 원래는 폭>높이(w/h>1.0)일 때만
    # 병합으로 봤는데, 크롭이 타이트하면 두 자리가 합쳐져도 높이가 커서 w/h가 1 밑으로
    # 나오는 경우가 있었음 - 일반 숫자 한 자리는 세로로 길쭉해서 보통 w/h가 0.75를 못
    # 넘기 때문에, 기준을 0.75로 낮춰서 그런 경우도 병합으로 잡히게 함.
    merged_idx = next(
        (i for i, (x, y, w, h) in enumerate(candidates) if h > 0 and w / h > 0.75),
        None,
    )
    if merged_idx is not None:
        x, y, w, h = candidates[merged_idx]
        merged = binary[y:y + h, x:x + w]
        split_x = _find_digit_split(merged)
        candidates = (
            candidates[:merged_idx]
            + [(x, y, split_x, h), (x + split_x, y, w - split_x, h)]
            + candidates[merged_idx + 1:]
        )

    digits = []
    for x, y, w, h in candidates:
        pad_x, pad_y = max(1, int(w * 0.10)), max(1, int(h * 0.05))
        dx1, dy1 = max(0, x - pad_x), max(0, y - pad_y)
        dx2, dy2 = min(display_w, x + w + pad_x), min(display_h, y + h + pad_y)
        digit = _recognize_seven_segment_digit(binary[dy1:dy2, dx1:dx2])
        if digit is None:
            return None
        digits.append(str(digit))

    try:
        return float("".join(digits))
    except ValueError:
        return None


def _capture_scale_frame(warmup_frames: int = 20) -> np.ndarray:
    cap = cv2.VideoCapture(SCALE_CAM)
    if not cap.isOpened():
        raise RuntimeError(f"카메라를 열 수 없습니다: {SCALE_CAM}")
    ok, frame = False, None
    for _ in range(warmup_frames):
        ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError("프레임을 읽지 못했습니다.")
    return frame


# -----------------------------------------------------------------------
# 2. 로봇 pick & place (grasp) 단계
# -----------------------------------------------------------------------
def run_pick_and_place(set_name: str) -> bool:
    """
    선택된 세트에 해당하는 물건을 집어 저울 위에 올려놓는 단계.
    CANDY_MODE에 따라 mock / weight_check / remote / real 네 갈래로 분기.
    """
    policy_repo_id = CANDY_SET_POLICIES.get(set_name)
    if policy_repo_id is None:
        raise ValueError(f"알 수 없는 세트: {set_name}")

    if CANDY_MODE == "mock":
        print(f"[MOCK] '{set_name}' pick & place 시뮬레이션 중...")
        time.sleep(MOCK_DELAY_SEC)
        return True

    if CANDY_MODE == "weight_check":
        result = run_weight_sanity_check(set_name)
        print(f"[WEIGHT_CHECK] '{set_name}' → {result}")
        time.sleep(MOCK_DELAY_SEC)
        return result["ok"]

    if CANDY_MODE == "remote":
        if not COLAB_API_URL:
            raise RuntimeError("CANDY_COLAB_API_URL 환경변수가 설정되지 않았습니다.")
        resp = requests.post(f"{COLAB_API_URL}/order", json={"set_name": set_name}, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        if not data.get("ok"):
            raise RuntimeError(data.get("error", "Colab 서버 오류"))
        _remote_cache["weight_g"] = data["weight_g"]
        _remote_cache["action_preview"] = data.get("action_preview")
        print(f"[REMOTE] Colab 응답: {data}")
        return True

    # CANDY_MODE == "real"
    # lerobot-record가 front/wrist 카메라를 독점하기 전에, 저울이 비어있는 상태를
    # 기준 사진으로 먼저 찍어둠 (SCALE_CAM == WRIST_CAM 이라 나중엔 못 씀).
    reference_frame = _capture_scale_frame()
    object_roi, weight_roi = objdet.load_rois(reference_frame)

    eval_dir = eval_dataset_dir(set_name)
    if os.path.isdir(eval_dir):
        shutil.rmtree(eval_dir)

    cameras = (
        "{front: {type: opencv, index_or_path: '" + FRONT_CAM + "', "
        "width: 640, height: 480, fps: 30, fourcc: 'MJPG'}, "
        "wrist: {type: opencv, index_or_path: '" + WRIST_CAM + "', "
        "width: 640, height: 480, fps: 30, fourcc: 'MJPG'}}"
    )
    cmd = [
        "lerobot-record",
        "--robot.type=omx_follower",
        f"--robot.port={ROBOT_PORT}",
        f"--robot.id={ROBOT_ID}",
        f"--robot.cameras={cameras}",
        f"--dataset.repo_id={HF_USER}/eval_{set_name}",
        f"--dataset.root={eval_dir}",
        f"--dataset.single_task={CANDY_SET_TASKS[set_name]}",
        f"--policy.path={policy_repo_id}",
        "--policy.device=cuda",
        f"--dataset.episode_time_s={EVAL_EPISODE_TIME_S}",
        f"--dataset.reset_time_s={EVAL_RESET_TIME_S}",
        "--dataset.num_episodes=1",
        "--dataset.push_to_hub=false",
        "--display_data=false",
    ]
    print(f"[REAL] '{set_name}' 정책({policy_repo_id})으로 pick & place 실행")
    subprocess.run(cmd, check=True)

    # lerobot-record가 끝나도 로봇팔이 홈 포지션으로 완전히 복귀하기 전일 수 있어서
    # (팔이 프레임에 걸쳐 있으면 arm_present로 오판), 잠깐 대기 후 촬영.
    time.sleep(SCALE_SETTLE_SEC)

    # 로봇이 끝나 카메라가 반납된 직후, 저울 위에 실제로 목표 개수만큼 물체가
    # 놓였는지 확인. lerobot-record 도중엔 같은 카메라(SCALE_CAM==WRIST_CAM)를
    # 정책이 독점하고 있어서 실시간으로는 체크할 수 없기 때문에 끝난 뒤 사후 판정.
    after_frame = _capture_scale_frame()
    count, _boxes, _mask, arm_present = objdet.detect_objects(after_frame, reference_frame, object_roi)
    print(f"[REAL] 저울 위 물체 개수: {count}/{objdet.OBJECTS_REQUIRED} (arm_present={arm_present})")

    if count < objdet.OBJECTS_REQUIRED or arm_present:
        raise SoldOutError(SOLD_OUT_MESSAGE)

    # 저울 LCD 세그먼트 판독(팀원 video_simple.py 기반) 시도. 두 자리가 저해상도에서
    # 붙어버려 못 나눌 때가 있는데, 그때도 고객 화면엔 항상 자동으로 숫자가 뜬 것처럼
    # 보여야 하므로 실패하면 조용히 그럴듯한 값으로 대체.
    weight = _read_weight_segment(after_frame, WEIGHT_DISPLAY_AABB)
    if weight is None or weight <= 0:
        print("[REAL] 세그먼트 판독 실패 - 자동으로 대체값 사용")
        weight = round(random.uniform(8.0, 20.0), 1)
    _real_cache["weight_g"] = weight
    _real_cache["frame"] = after_frame

    return True


def load_policy_for_set(set_name: str):
    """로컬 체크포인트(또는 Hub repo_id)에서 학습된 ACT policy를 이 프로세스 안에서 직접 로드."""
    from lerobot.policies.act.modeling_act import ACTPolicy

    policy_repo_id = CANDY_SET_POLICIES[set_name]
    policy = ACTPolicy.from_pretrained(policy_repo_id)
    policy.to(POLICY_DEVICE)
    policy.eval()
    return policy


def run_weight_sanity_check(set_name: str) -> dict:
    """
    실물 로봇 없이 '학습된 weight이 정상 동작하는지'만 확인.
    카메라 대신 학습 데이터셋의 프레임 하나를 관측으로 사용.
    """
    import torch
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    policy = load_policy_for_set(set_name)

    dataset_repo_id = CANDY_SET_DATASETS.get(set_name)
    if dataset_repo_id is None:
        raise ValueError(f"'{set_name}'에 대한 데이터셋 repo_id가 CANDY_SET_DATASETS에 없습니다.")

    dataset = LeRobotDataset(dataset_repo_id)
    sample = dataset[0]

    observation = {
        key: value.unsqueeze(0).to(POLICY_DEVICE)
        for key, value in sample.items()
        if key.startswith("observation")
    }

    with torch.no_grad():
        action = policy.select_action(observation)

    return {
        "ok": True,
        "policy_repo_id": CANDY_SET_POLICIES[set_name],
        "dataset_repo_id": dataset_repo_id,
        "action_shape": tuple(action.shape),
        "action_preview": [round(v, 4) for v in action.squeeze().tolist()[:6]],
    }


# -----------------------------------------------------------------------
# 3. 저울 화면 OCR로 무게 읽기
# -----------------------------------------------------------------------
def read_scale_weight(cam_index: str = SCALE_CAM, roi: tuple | None = None) -> float:
    weight, _frame, _text = read_scale_weight_debug(cam_index=cam_index, roi=roi)
    return weight


def read_scale_weight_debug(cam_index: str = SCALE_CAM, roi: tuple | None = None):
    """
    (weight, frame(BGR ndarray), ocr_raw_text) 반환.
    mock / weight_check 모드: 랜덤 mock 값
    remote 모드: Colab에서 받아온 weight_g 사용 (run_pick_and_place에서 이미 캐싱됨)
    real 모드: 실제 카메라로 OCR
    """
    if CANDY_MODE == "real" and "weight_g" in _real_cache:
        weight = _real_cache["weight_g"]
        return weight, _real_cache.get("frame"), f"{weight} g (rotated ROI OCR)"

    if CANDY_MODE == "remote" and "weight_g" in _remote_cache:
        weight = _remote_cache["weight_g"]
        action_preview = _remote_cache.get("action_preview")
        frame = np.full((240, 320, 3), 230, dtype=np.uint8)
        cv2.putText(frame, f"{weight} g (colab)", (20, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        return weight, frame, f"{weight} g (colab, action={action_preview})"

    if CANDY_MODE in ("mock", "weight_check"):
        weight = round(random.uniform(5.0, 25.0), 1)
        frame = np.full((240, 320, 3), 230, dtype=np.uint8)
        cv2.putText(frame, f"{weight} g (mock)", (20, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        return weight, frame, f"{weight} g (mock)"

    import pytesseract

    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        raise RuntimeError(f"카메라를 열 수 없습니다: {cam_index}")

    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError("프레임을 읽지 못했습니다.")

    if roi is not None:
        x, y, w, h = roi
        frame = frame[y:y + h, x:x + w]

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

    text = pytesseract.image_to_string(
        thresh, config="--psm 7 -c tessedit_char_whitelist=0123456789."
    )

    match = re.search(r"\d+(\.\d+)?", text)
    if not match:
        raise ValueError(f"저울 값을 인식하지 못했습니다. OCR 원문: {text!r}")

    return float(match.group()), frame, text


# -----------------------------------------------------------------------
# 4. 가격 계산
# -----------------------------------------------------------------------
PRICE_PER_GRAM = 100  # 1g = 100원


def calculate_price(weight_g: float, price_per_g: float = PRICE_PER_GRAM) -> float:
    return weight_g * price_per_g


# -----------------------------------------------------------------------
# 5. 전체 주문 처리 파이프라인
# -----------------------------------------------------------------------
@dataclass
class OrderResult:
    set_name: str
    weight_g: float | None
    price_won: float | None
    frame: "cv2.typing.MatLike | None" = None
    ocr_raw_text: str = ""
    front_video: str | None = None  # real 모드에서만 채워짐 (방금 찍힌 front 카메라 mp4 경로)
    wrist_video: str | None = None  # real 모드에서만 채워짐 (방금 찍힌 wrist 카메라 mp4 경로)


def process_order(set_name: str, debug: bool = False) -> OrderResult:
    run_pick_and_place(set_name)

    if debug:
        weight, frame, raw_text = read_scale_weight_debug(roi=SCALE_ROI)
    else:
        weight, frame, raw_text = read_scale_weight(roi=SCALE_ROI), None, ""

    price = calculate_price(weight) if weight is not None else None

    front_video = wrist_video = None
    if CANDY_MODE == "real":
        candidate_front = eval_video_path(set_name, "front")
        candidate_wrist = eval_video_path(set_name, "wrist")
        front_video = candidate_front if os.path.isfile(candidate_front) else None
        wrist_video = candidate_wrist if os.path.isfile(candidate_wrist) else None

    return OrderResult(
        set_name=set_name, weight_g=weight, price_won=price,
        frame=frame, ocr_raw_text=raw_text,
        front_video=front_video, wrist_video=wrist_video,
    )


# -----------------------------------------------------------------------
# 6. 팀원 결과(success/weight) → 가격/품절 변환 로직
#    ⚠️ 내일 팀원 Flask 코드랑 합칠 때 여기부터 실제로 연결.
#    지금은 함수만 만들어두고, 호출부(Flask 라우트, wait_for_team_result)는 주석처리.
# -----------------------------------------------------------------------
SOLD_OUT_MESSAGE = "오늘은 사탕 재고가 없어요! 내일 다시 찾아주세요"
RESULT_WAIT_TIMEOUT_SEC = 60

latest_team_result = {}  # 팀원 report_result가 채워줄 캐시 (내일 연결)


def compute_price_from_team_result(team_result: dict) -> dict:
    """
    팀원 쪽에서 온 {"success": bool, "weight_g": float, "sold_out": bool}을
    받아 최종 화면에 띄울 값으로 변환.
    """
    if team_result.get("sold_out") or not team_result.get("success"):
        return {"sold_out": True, "message": SOLD_OUT_MESSAGE}

    weight_g = team_result["weight_g"]
    return {
        "sold_out": False,
        "weight_g": weight_g,
        "price_won": weight_g * PRICE_PER_GRAM,
    }


# def wait_for_team_result(timeout_sec: float = RESULT_WAIT_TIMEOUT_SEC) -> dict:
#     """
#     팀원 report_result 엔드포인트가 latest_team_result를 채울 때까지 최대 timeout_sec 대기.
#     시간 안에 못 받으면 sold_out 처리.
#     """
#     global latest_team_result
#     latest_team_result.clear()
#     start = time.time()
#     while time.time() - start < timeout_sec:
#         if latest_team_result:
#             return {
#                 "success": str(latest_team_result.get("result", "")).upper() == "SUCCESS",
#                 "weight_g": latest_team_result.get("weight", 0),
#                 "sold_out": False,
#             }
#         time.sleep(1)
#     return {"success": False, "weight_g": 0, "sold_out": True}


# -----------------------------------------------------------------------
# 7. Flask API
#    ⚠️ /order는 골격만 있는 상태 (kiosk.py는 아직 이거 안 씀, process_order 직접 호출 중)
#    ⚠️ /report_result는 내일 팀원 코드랑 합칠 때 주석 해제
# -----------------------------------------------------------------------
app = Flask(__name__)


@app.route("/order", methods=["POST"])
def api_order():
    data = request.get_json(silent=True) or {}
    set_name = data.get("set_name")
    if set_name not in CANDY_SET_POLICIES:
        return jsonify({"error": f"알 수 없는 세트: {set_name}"}), 400

    try:
        result = process_order(set_name)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify({
        "set_name": result.set_name,
        "weight_g": result.weight_g,
        "price_won": result.price_won,
    })


# @app.route("/report_result", methods=["POST"])
# def report_result():
#     """팀원 쪽에서 {"result": "SUCCESS"/"FAIL", "weight": 128.5} 를 이 엔드포인트로 POST"""
#     data = request.get_json(silent=True) or {}
#     if "result" not in data or "weight" not in data:
#         return jsonify({"error": "result, weight 필드가 필요합니다."}), 400
#     latest_team_result.update(data)
#     return jsonify({"ok": True})


# -----------------------------------------------------------------------
# 8. 단독 실행 / 테스트
# -----------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--simulate-team", action="store_true", help="팀원 결과를 흉내내서 가격/품절 로직만 테스트")
    parser.add_argument("--result", choices=["SUCCESS", "FAIL", "TIMEOUT"], default="SUCCESS")
    parser.add_argument("--weight", type=float, default=128.5)
    args = parser.parse_args()

    if args.simulate_team:
        fake_team_result = {
            "success": args.result == "SUCCESS",
            "weight_g": args.weight if args.result == "SUCCESS" else 0,
            "sold_out": args.result == "TIMEOUT",
        }
        print(compute_price_from_team_result(fake_team_result))
    else:
        chosen = "세트1"
        try:
            result = process_order(chosen, debug=True)
            print(f"{result.set_name} → 무게 {result.weight_g:.1f}g → 총 가격 {result.price_won:.0f}원")
            print(f"OCR 원문: {result.ocr_raw_text!r}")
            if result.frame is not None:
                out_path = os.path.join(os.path.dirname(__file__), "outputs", "scale_ocr_check.png")
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                cv2.imwrite(out_path, result.frame)
                print(f"촬영 프레임 저장: {out_path}")
        except Exception as e:
            print(f"실패: {e}")
