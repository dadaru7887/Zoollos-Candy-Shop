"""
저울 카메라(SCALE_CAM) 실시간 화면을, 물체 감지 범위(초록)와 무게 읽는 범위(노랑)
박스를 겹쳐서 창으로 띄우는 스크립트.
로봇이 동작 중일 땐 이 카메라를 lerobot-record가 독점하고 있어서 이 스크립트를
동시에 못 씀 - 로봇이 멈춰있을 때(주문 사이)만 실행해줘.

사용법:
  /home/newuser/venv/cv_debug/bin/python3 live_webcam_view.py
  (종료: 창에서 q 키)
"""
import json

import cv2
import numpy as np

from candy_shop_backend_ver9 import SCALE_CAM, WEIGHT_DISPLAY_AABB
import webcam_simple_repeat_ver2 as objdet

object_roi = None
if objdet.ROI_FILE.exists():
    data = json.loads(objdet.ROI_FILE.read_text(encoding="utf-8"))
    object_roi = np.array(data["object_roi"], dtype=np.int32)

cap = cv2.VideoCapture(SCALE_CAM)
if not cap.isOpened():
    raise SystemExit(f"카메라를 열 수 없습니다: {SCALE_CAM}")

print("실시간 화면 창이 뜹니다. 종료하려면 그 창에서 q 키를 눌러주세요.")

while True:
    ok, frame = cap.read()
    if not ok:
        print("프레임을 읽지 못했습니다.")
        break

    if object_roi is not None:
        cv2.polylines(frame, [object_roi], isClosed=True, color=(0, 255, 0), thickness=2)
        cv2.putText(frame, "object_roi", tuple(object_roi[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    x1, y1, x2, y2 = WEIGHT_DISPLAY_AABB
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
    cv2.putText(frame, "weight_aabb", (x1, max(15, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    cv2.imshow("Scale Camera (q to quit)", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
