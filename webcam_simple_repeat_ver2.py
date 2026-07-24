"""
webcam_simple_repeat.py의 회전 ROI 지원 버전.
====================================================
저울이 카메라와 수직으로 안 맞고 비스듬히 놓여있을 때, cv2.selectROI(축 정렬 사각형만
가능)로는 정확한 영역을 못 잡아서, 4개 꼭짓점을 직접 클릭해 기울어진 사각형(ROI)을
지정하고 warpPerspective로 똑바로 펴서 비교하는 방식으로 바꿈.

원본(webcam_simple_repeat.py)은 건드리지 않음. candy_shop_backend_ver8.py가 쓰는
load_rois() / detect_objects() 두 함수는 이름과 반환값 형태(개수/박스/마스크/arm_present)를
그대로 유지해서, backend 쪽은 import 한 줄만 바꾸면 됨 (호출부 수정 불필요).

roi_config.json(원본, 축 정렬 좌표)과는 별개로 roi_config_v2.json(회전 4점 좌표)을 씀.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import cv2
import numpy as np
import pytesseract

ROI_FILE = Path("roi_config_v2.json")
SUCCESS_DIR = Path("success_images")
OBJECTS_REQUIRED = 2
HOLD_SECONDS = 2.0
TIMEOUT_SECONDS = 60.0
MIN_AREA = 350
MAX_OBJECT_AREA = 10000
ROBOT_ARM_MIN_AREA = 12000
CLEAR_HOLD_SECONDS = 1.0
DIFF_THRESHOLD = 35


# -----------------------------------------------------------------------
# 회전 ROI: 4점 클릭 선택 + 정렬 + 투시변환(warp)
# -----------------------------------------------------------------------
def order_points(points) -> np.ndarray:
    """4점을 좌상->우상->우하->좌하 순서로 정렬 (getPerspectiveTransform용)."""
    pts = np.array(points, dtype=np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).flatten()
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def select_roi(frame, title) -> np.ndarray:
    """4개 꼭짓점을 순서 상관없이 클릭 -> 자동 정렬된 (4,2) 배열 반환."""
    points: list[tuple[int, int]] = []

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append((x, y))

    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    for _ in range(10):
        cv2.imshow(title, frame)
        cv2.waitKey(30)  # QT 백엔드가 창을 실제로 띄울 시간을 넉넉히 줌
    cv2.setMouseCallback(title, on_mouse)

    while True:
        vis = frame.copy()
        for i, p in enumerate(points):
            cv2.circle(vis, p, 5, (0, 0, 255), -1)
            if i > 0:
                cv2.line(vis, points[i - 1], p, (0, 255, 0), 2)
        if len(points) == 4:
            cv2.line(vis, points[3], points[0], (0, 255, 0), 2)
        cv2.putText(
            vis, f"{title}: {len(points)}/4 점 클릭  (r=리셋, Enter/Space=확정, c=취소)",
            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2,
        )
        cv2.imshow(title, vis)
        key = cv2.waitKey(20) & 0xFF

        if key == ord("r"):
            points.clear()
        elif key == ord("c"):
            cv2.destroyWindow(title)
            raise RuntimeError(f"ROI 선택이 취소되었습니다: {title}")
        elif key in (13, 32) and len(points) == 4:  # Enter / Space
            break

    cv2.destroyWindow(title)
    return order_points(points)


def load_rois(frame):
    if ROI_FILE.exists():
        data = json.loads(ROI_FILE.read_text(encoding="utf-8"))
        object_roi = np.array(data["object_roi"], dtype=np.float32)
        weight_roi = np.array(data["weight_roi"], dtype=np.float32)
        return object_roi, weight_roi

    object_roi = select_roi(frame, "1. object area (4 points)")
    weight_roi = select_roi(frame, "2. scale display (4 points)")
    ROI_FILE.write_text(
        json.dumps({"object_roi": object_roi.tolist(), "weight_roi": weight_roi.tolist()}, indent=2),
        encoding="utf-8",
    )
    return object_roi, weight_roi


def warp_roi(frame, quad_points: np.ndarray):
    """4점으로 정의된 기울어진 영역을 직사각형으로 펴서 잘라냄."""
    tl, tr, br, bl = quad_points
    width = int(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)))
    height = int(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr)))
    width, height = max(width, 1), max(height, 1)
    dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(quad_points, dst)
    return cv2.warpPerspective(frame, matrix, (width, height))


# -----------------------------------------------------------------------
# 물체 감지 (회전 ROI 버전) - 반환 형태는 원본과 동일: (count, boxes, mask, arm_present)
# -----------------------------------------------------------------------
def _match_brightness(ref: np.ndarray, cur: np.ndarray) -> np.ndarray:
    """reference/after 프레임이 카메라를 매번 새로 열어서 찍은 거라(자동노출이
    두 번 독립적으로 수렴), 물체가 안 움직여도 전체적으로 밝기가 달라질 수 있음.
    그러면 diff가 화면 전체에 옅게 깔려서 물체 개수를 잘못 세게 되므로,
    cur을 ref 평균 밝기에 맞춰 스케일링해서 순수 구조적 차이만 남김."""
    ref_mean = float(cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY).mean())
    cur_mean = float(cv2.cvtColor(cur, cv2.COLOR_BGR2GRAY).mean())
    if cur_mean < 1.0:
        return cur
    gain = float(np.clip(ref_mean / cur_mean, 0.5, 2.0))
    return cv2.convertScaleAbs(cur, alpha=gain, beta=0)


def detect_objects(frame, reference, roi: np.ndarray):
    cur = warp_roi(frame, roi)
    ref = warp_roi(reference, roi)
    cur = _match_brightness(ref, cur)

    diff = cv2.absdiff(cv2.GaussianBlur(ref, (7, 7), 0), cv2.GaussianBlur(cur, (7, 7), 0))
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)
    k = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)
    mask = cv2.erode(mask, k, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    arm_present = False

    for c in contours:
        area = cv2.contourArea(c)

        if area >= ROBOT_ARM_MIN_AREA:
            arm_present = True
            continue

        if area < MIN_AREA or area > MAX_OBJECT_AREA:
            continue

        x, y, bw, bh = cv2.boundingRect(c)
        boxes.append((x, y, bw, bh))

    return len(boxes), boxes, mask, arm_present


def read_weight(frame, roi: np.ndarray):
    crop = warp_roi(frame, roi)
    if crop.size == 0:
        return 0.0
    img = cv2.resize(crop, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    text = pytesseract.image_to_string(binary, config="--psm 7 -c tessedit_char_whitelist=0123456789.-").strip()
    m = re.search(r"-?\d+(?:[.,]\d+)?", text)
    if not m:
        return 0.0
    try:
        return float(m.group().replace(",", "."))
    except ValueError:
        return 0.0


def save_success(frame):
    SUCCESS_DIR.mkdir(exist_ok=True)
    path = SUCCESS_DIR / f"success_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
    cv2.imwrite(str(path), frame)
    return str(path)


def send_result(result, weight):
    response = {"result": result, "weight": weight}
    print(response)
    # 여기에 서버 전송 또는 ROS publish 연결
