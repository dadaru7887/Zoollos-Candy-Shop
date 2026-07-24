"""
물체 감지 영역 + 무게 읽는 영역을 드래그 방식(cv2.selectROI, 검증된 방법)으로 다시 지정.
- 물체 감지 영역 -> roi_config_v2.json의 object_roi에 저장 (weight_roi는 이제 안 씀)
- 무게 읽는 영역 -> candy_shop_backend_ver9.py의 WEIGHT_DISPLAY_AABB 기본값을 자동으로 수정

로봇이 동작 중이 아닐 때(카메라가 비어있을 때)만 실행해줘.

사용법:
  /home/newuser/venv/cv_debug/bin/python3 redo_regions.py
"""
import json
import re
from pathlib import Path

import cv2

from candy_shop_backend_ver9 import SCALE_CAM

ROI_FILE = Path("roi_config_v2.json")
BACKEND_FILE = Path("candy_shop_backend_ver9.py")

cap = cv2.VideoCapture(SCALE_CAM)
if not cap.isOpened():
    raise SystemExit(f"카메라를 열 수 없습니다: {SCALE_CAM}")

frame = None
for _ in range(5):
    ok, frame = cap.read()
cap.release()
if not ok:
    raise SystemExit("프레임을 읽지 못했습니다.")

print("1) 물체(사탕)가 놓일 저울 위 영역을 드래그해서 잡아줘. 끝나면 Enter/Space.")
ox, oy, ow, oh = map(int, cv2.selectROI("1. 물체 감지 영역", frame, False, False))
cv2.destroyAllWindows()
if ow <= 0 or oh <= 0:
    raise SystemExit("물체 감지 영역 선택 실패")

print("2) 저울 LCD 숫자 화면 영역을 드래그해서 잡아줘 (숫자만 딱 타이트하게). 끝나면 Enter/Space.")
wx, wy, ww, wh = map(int, cv2.selectROI("2. 무게 읽는 영역", frame, False, False))
cv2.destroyAllWindows()
if ww <= 0 or wh <= 0:
    raise SystemExit("무게 읽는 영역 선택 실패")

object_roi = [[ox, oy], [ox + ow, oy], [ox + ow, oy + oh], [ox, oy + oh]]
ROI_FILE.write_text(json.dumps({"object_roi": object_roi}, indent=2), encoding="utf-8")
print(f"object_roi 저장 완료 -> {ROI_FILE.resolve()}")
print(f"  object_roi = {object_roi}")

weight_aabb = f"{wx},{wy},{wx + ww},{wy + wh}"
text = BACKEND_FILE.read_text(encoding="utf-8")
new_text, n = re.subn(
    r'CANDY_WEIGHT_AABB", "[^"]*"',
    f'CANDY_WEIGHT_AABB", "{weight_aabb}"',
    text,
)
if n == 0:
    print("경고: WEIGHT_DISPLAY_AABB 기본값을 못 찾아서 자동 수정 실패. 직접 알려줄게:")
else:
    BACKEND_FILE.write_text(new_text, encoding="utf-8")
    print(f"{BACKEND_FILE} 자동 수정 완료")
print(f"  weight_aabb = ({wx}, {wy}, {wx + ww}, {wy + wh})")
