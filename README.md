# 🍬 Zoollos Candy Shop

로봇팔(OMX)이 모방학습(ACT) 정책으로 사탕을 저울에 올리고, 카메라로 저울 LCD를 읽어 무게 → 가격을 계산해 보여주는 키오스크 데모입니다.

A robot-arm (OMX) kiosk demo that uses imitation-learning (ACT) policies to place candy on a scale, reads the scale's LCD display via camera, and computes/displays the price from the detected weight.

**흐름 / Flow**
세트 선택(빨강/보라) → 로봇이 ACT 정책으로 해당 사탕을 집어 저울에 올림 → 카메라로 저울 개수/LCD 숫자 판독 → 가격 계산 → 결과 표시
Select a set (red/violet) → robot picks the candy via its ACT policy and places it on the scale → camera detects object count / reads the LCD digits → price is computed → result is shown.

---

## 시스템 구성 / Components

| 파일 / File | 역할 / Role |
|---|---|
| `candy_shop_backend_ver9.py` | 핵심 로직: 로봇 제어, 물체 감지 판정, 무게 인식, 가격 계산 <br> Core logic: robot control, object-count detection, weight recognition, price calculation |
| `webcam_simple_repeat_ver2.py` | 물체 개수 감지 (회전 ROI + 배경 차분) <br> Object-count detection (rotated ROI + background subtraction) |
| `live_view_server.py` | 브라우저(localhost:5001)에서 드래그로 관심영역(물체/무게) 재지정 <br> Browser-based tool to re-draw the object/weight ROIs by dragging |
| `show_check_regions.py` | 현재 설정된 ROI를 라이브 프레임에 그려 이미지로 저장 (확인용) <br> Draws the currently configured ROIs on a live frame and saves it, for verification |
| `redo_regions.py` | ROI 재설정 보조 스크립트 <br> Helper script for redefining ROIs |
| `roi_config_v2.json` | 물체/무게 ROI 좌표 설정 <br> ROI coordinates for object/weight detection |

**CANDY_MODE 4단계 / 4 operating modes**: `mock`(랜덤값/random values) · `weight_check`(정책 forward pass만 확인/policy forward-pass check only) · `remote`(Colab 서버 호출/calls a remote Colab server) · `real`(실물 로봇+카메라/real robot + camera).

### 물체 개수 판정 / Object-count detection
`webcam_simple_repeat_ver2.detect_objects()` — 로봇 이동 전/후 프레임을 4점 ROI로 `warpPerspective` 후 GaussianBlur → absdiff → threshold → morphology → contour 개수로 판정.
Warps before/after frames through the 4-point ROI, then GaussianBlur → absdiff → threshold → morphology → contour counting to decide how many objects are on the scale.

### 무게 인식 / Weight recognition
OCR(pytesseract) 대신 직접 만든 규칙 기반 7-세그먼트 디코더 사용 (저해상도 LCD에서 tesseract 인식률이 낮아 세그먼트 방식으로 전환).
Uses a hand-built rule-based 7-segment decoder instead of OCR (pytesseract), since tesseract accuracy was poor on the low-resolution LCD.

### 가격 계산 / Price calculation
```
PRICE_PER_GRAM = 100  # 1g = 100원 / 100 KRW
price = weight_g * PRICE_PER_GRAM
```

---

## 학습된 정책 / Trained policies

| 폴더 / Folder | 내용 / Contents |
|---|---|
| `single_red_policy/` | 빨강 세트 grasp 정책 (ACT) <br> Red-set grasp policy (ACT) |
| `single_violet_policy/` | 보라 세트 grasp 정책 (ACT) <br> Violet-set grasp policy (ACT) |

공통 하이퍼파라미터 / Shared hyperparameters: `policy.type=act`, `batch_size=8`, `steps=50000`, `vision_backbone=resnet18`, `chunk_size=100`, `n_action_steps=100`, `n_encoder_layers=4`, `n_decoder_layers=1`, `n_heads=8`, `optimizer_lr=1e-5`.

가중치 파일이 커서(수백 MB) 이 리포에는 포함하지 않고 구글 드라이브로 배포합니다. 다운로드 후 리포 루트에 압축을 풀어 `single_red_policy/`, `single_violet_policy/` 폴더로 놓으면 됩니다.
Weight files are large (hundreds of MB) so they're distributed via Google Drive instead of this repo. Download and unzip into the repo root as `single_red_policy/` and `single_violet_policy/`.

- `single_red_policy.zip` — TODO: 드라이브 링크 추가 / add Drive link
- `single_violet_policy.zip` — TODO: 드라이브 링크 추가 / add Drive link

---

## 실행 스크립트 / Run scripts

| 스크립트 / Script | 용도 / Purpose |
|---|---|
| `run_teleop.sh` | 리더-팔로워 원격조작 / Leader-follower teleoperation |
| `run_record*.sh` | 시연 데이터 녹화 (`lerobot-record`) / Record demonstration episodes |
| `run_train*.sh` | ACT 정책 학습 (`lerobot-train`) / Train the ACT policy |
| `run_inference_*.sh` | 학습된 정책으로 평가 실행 / Run policy evaluation/inference |

각 스크립트는 `lerobot` 워크스페이스(`~/il_ws/src/lerobot`)와 Hugging Face 로그인(`hf auth whoami`)을 전제로 합니다.
Each script assumes a `lerobot` workspace at `~/il_ws/src/lerobot` and an active Hugging Face login (`hf auth whoami`).

---

## 참고 문서 / Further docs

- [`review.md`](review.md) — 프로젝트 리뷰 및 트러블슈팅 기록 (국문) / Project review & troubleshooting log (Korean)
- [`candy_shop.txt`](candy_shop.txt) — 물체 감지·무게 인식 로직 상세 (국문) / Detailed detection/recognition logic (Korean)
- [`Zoollos-Candy-Shop.pdf`](Zoollos-Candy-Shop.pdf) — 발표 자료 / Presentation slides
- `inference.MOVE` (데모 영상 / demo video, 84MB) — 리포에는 없음, 구글 드라이브 참고: TODO 링크 추가 / not in this repo, see Google Drive: TODO add link
