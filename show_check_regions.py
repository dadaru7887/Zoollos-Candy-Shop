"""
지금 이 순간 카메라로 한 장 찍어서, 실제로 코드가 체크하고 있는 영역을
박스로 그려서 저장하는 스크립트 (GUI 창 없이 이미지 파일로 확인).

사용법:
  /home/newuser/venv/il/bin/python3 show_check_regions.py
"""
import cv2
import numpy as np

from candy_shop_backend_ver9 import SCALE_CAM, WEIGHT_DISPLAY_AABB
import webcam_simple_repeat_ver2 as objdet

cap = cv2.VideoCapture(SCALE_CAM)
if not cap.isOpened():
    raise SystemExit(f"카메라를 열 수 없습니다: {SCALE_CAM}")

frame = None
for _ in range(5):
    ok, frame = cap.read()
cap.release()
if not ok:
    raise SystemExit("프레임을 읽지 못했습니다.")

vis = frame.copy()

# 물체 감지 영역(회전 4점) - 초록색
if objdet.ROI_FILE.exists():
    import json
    data = json.loads(objdet.ROI_FILE.read_text(encoding="utf-8"))
    object_roi = np.array(data["object_roi"], dtype=np.int32)
    cv2.polylines(vis, [object_roi], isClosed=True, color=(0, 255, 0), thickness=2)
    cv2.putText(vis, "object_roi", tuple(object_roi[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# 무게 읽는 영역(축정렬) - 노란색
x1, y1, x2, y2 = WEIGHT_DISPLAY_AABB
cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 255), 2)
cv2.putText(vis, "weight_aabb", (x1, max(15, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

out_path = "dbg_live_check.png"
cv2.imwrite(out_path, vis)
print(f"저장: {out_path}")
