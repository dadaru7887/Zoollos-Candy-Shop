# 사탕가게 키오스크 프로젝트 리뷰

## 1. 프로젝트 개요

OMX 로봇팔 + ACT(모방학습) 정책으로 사탕을 저울 위에 올려놓고, 저울 LCD 화면을 카메라로 읽어 무게 → 가격을 계산해 보여주는 키오스크 데모.

**흐름**: 세트 선택(빨강/보라) → 로봇이 ACT 정책으로 해당 사탕을 집어 저울에 올림 → 카메라로 저울 개수/LCD 숫자 판독 → 가격 계산 → Streamlit 화면에 결과 표시.

## 2. 시스템 구성

| 파일 | 역할 |
|---|---|
| `candy_shop_backend_ver9.py` | 핵심 로직 — 로봇 제어, 물체감지 판정, 무게 OCR, 가격계산 |
| `candy_shop_kiosk_ver6.py` | Streamlit UI (세트 선택 → 결과 카드 표시) |
| `webcam_simple_repeat_ver2.py` | 물체 개수 감지 (회전 ROI + 배경 차분) |
| `live_view_server.py` | 브라우저(localhost:5001)에서 드래그로 영역(물체/무게) 재지정 |
| `show_check_regions.py` | 지금 설정된 영역을 라이브 프레임에 그려서 이미지로 저장 (확인용) |

**CANDY_MODE 4단계**: `mock`(랜덤값) / `weight_check`(정책 forward pass만 확인) / `remote`(Colab 서버 호출) / `real`(실물 로봇+카메라).

### 물체 개수 판정 (`webcam_simple_repeat_ver2.detect_objects`)
- 로봇이 움직이기 전(reference)/후(after) 프레임을 4점 회전 ROI로 `warpPerspective`, GaussianBlur → absdiff → threshold → morphology → contour 개수 세기.
- `OBJECTS_REQUIRED = 2`(스낵박스 1개 + 사탕 1개가 저울 위에 있어야 함), 팔이 화면에 걸치면(`ROBOT_ARM_MIN_AREA` 이상) `arm_present=True`로 재판정.

### 무게 인식 (`_read_weight_segment`)
- OCR 라이브러리(pytesseract) 대신 **직접 만든 7-세그먼트 판독기** 사용 (실제 real 모드 경로에서 pytesseract 코드는 호출되지 않음, 레거시로만 남아있음).
- 고정 좌표(`WEIGHT_DISPLAY_AABB`)로 LCD 부분을 크롭 → Otsu 이진화 → 연결성분 분석 → 세그먼트 on/off 패턴을 숫자로 매핑.

### 가격 계산
```
PRICE_PER_GRAM = 100  # 1g = 100원
price = weight_g * PRICE_PER_GRAM
```

## 3. 학습 하이퍼파라미터

red/violet 단일 물체 grasp 정책 모두 동일 세팅.

| 항목 | 값 |
|---|---|
| policy.type | ACT |
| batch_size | 8 |
| steps | 50,000 |
| log_freq | 200 |
| save_freq | red: 10,000 / violet: 1,000 |
| vision_backbone | resnet18 |
| chunk_size / n_action_steps | 100 / 100 |
| n_encoder_layers / n_decoder_layers | 4 / 1 |
| n_heads | 8 |
| optimizer_lr | 1e-5 |

## 4. 오늘 트러블슈팅 기록

### 4.1 `lerobot-record: No such file or directory`
- **원인**: Streamlit을 venv 활성화 없이 `python3 -m streamlit run ...`으로 띄워서 PATH에 `venv/il/bin`이 안 잡힘.
- **해결**: `source venv/il/bin/activate` 후 `streamlit run`으로 재실행.

### 4.2 에피소드 타임아웃
- 2-pick 작업 때 20초로 끊기던 문제 → 60초로 고정했었는데, 오늘 다시 50초 → 40초로 낮춰 응답 속도 개선 시도.
- **참고**: `lerobot-record`는 정해진 시간만큼 무조건 도는 블로킹 호출이라, "성공하면 즉시 응답" 같은 조기 종료는 구조적으로 어려움 — 실제 소요시간에 맞춰 시간 값을 줄이는 것만 현실적인 대안.

### 4.3 무게 숫자 판독 오류 (여러 건)
- **자릿수 하나가 통째로 사라짐** ("37"→"3", "30"→"3"): 저해상도에서 `MORPH_OPEN`이 획을 끊어놔서, 조각난 파편이 높이 필터(절대 기준)에 걸려 탈락. → `MORPH_CLOSE`로 먼저 이어붙이고, 높이 필터를 "이번 프레임에서 제일 큰 높이의 55%" 상대 기준으로 변경.
- **왼쪽 끝 노이즈 필터가 진짜 숫자까지 제거**: 두 자리가 붙어서 큰 덩어리로 잡히면 크롭 왼쪽 끝(x=0)에 닿는 경우가 있는데, "왼쪽 끝은 노이즈"로 보고 무조건 제외하던 필터가 이 덩어리까지 지워버림 → 필터에 "폭이 좁을 때만" 조건 추가.
- **병합 자리 분리 기준(w/h>1.0)이 타이트한 크롭에선 안 걸림**: 크롭이 타이트하면 두 자리가 붙어도 높이가 커서 w/h가 1 밑으로 나옴 → 기준을 0.75로 완화.
- **자동 LCD 화면 탐지(옵션 B) 시도 후 롤백**: 버튼(원형)과 LCD 화면(사각형)을 자동 구분하는 로직을 추가했다가, 다른 케이스에서 오히려 이상하게 나와서 시도 전 백업본으로 원복.
- **최종 원인은 좌표 자체가 살짝 낮게 잡힘**: 여러 차례 "카메라가 흔들린다/로봇 자세가 다르다"고 오판했었는데, 실제로는 카메라도 저울도 고정이었고 — 마지막에 다시 잡은 좌표의 세로 범위(y)가 실제 숫자 위치보다 13px 정도 아래로 치우쳐 있었던 게 진짜 원인. 실패했던 실제 프레임으로 좌표를 다시 정밀 측정해서 고정.

### 4.4 물체 개수 오판정 ("품절" 오표시)
- 실제로는 물체 2개가 맞게 놓여있는데 배경 차분 결과가 1개/5개로 잘못 세는 문제.
- **추정 원인**: reference/after 프레임을 카메라를 매번 새로 열어서 찍다 보니 자동노출이 독립적으로 수렴해 전체 밝기가 미세하게 달라짐.
- **조치**: 워밍업 프레임 5→20장, `detect_objects`에 밝기 보정(after 프레임을 before 평균 밝기에 맞춰 스케일링) 추가.

## 5. 남은 과제 / 참고사항
- 물체 개수 판정은 밝기 보정 후 재현 테스트에서는 정상 동작했지만, 라이브 환경에서 추가 검증 필요.
- 무게 인식 좌표(`WEIGHT_DISPLAY_AABB`)는 저울/카메라를 물리적으로 건드리면 다시 잡아야 함 — `live_view_server.py`(브라우저 드래그) 또는 `show_check_regions.py`(현재 좌표 확인용)로 재조정 가능.
- `single_red_policy` 체크포인트가 불완전함 — `model.safetensors`(실제 학습 가중치)가 없고 `training_state/optimizer_state.safetensors`만 있음. 학습 머신/Colab 등 원본 위치에서 찾아 보완 필요. (violet은 반대로 `model.safetensors`는 있고 `optimizer_state.safetensors`가 없는데, 이건 추론엔 문제없고 학습 재개만 안 되는 정상 상태.)
