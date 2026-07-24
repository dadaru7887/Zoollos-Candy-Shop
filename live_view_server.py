"""
저울 카메라 실시간 화면을 브라우저에서 보고, 마우스로 드래그해서 영역을 저장하는 서버
(OpenCV Qt 창 문제를 완전히 우회 - 브라우저만 있으면 됨).

사용법:
  /home/newuser/venv/il/bin/python3 live_view_server.py
  그다음 브라우저에서 http://localhost:5001 접속

로봇이 동작 중일 땐 카메라를 lerobot-record가 독점하고 있어서 못 씀 -
로봇이 멈춰있을 때(주문 사이)만 켜줘. 종료: 터미널에서 Ctrl+C.

주의: 저장하면 파일(roi_config_v2.json / candy_shop_backend_ver9.py)만 바뀜.
이미 떠있는 streamlit은 값을 메모리에 캐싱하고 있어서, 저장 후 streamlit을
껐다가 다시 켜야 반영됨.
"""
import json
import re
import time
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, Response, jsonify, request

from candy_shop_backend_ver9 import SCALE_CAM, WEIGHT_DISPLAY_AABB
import webcam_simple_repeat_ver2 as objdet

ROI_FILE = Path("roi_config_v2.json")
BACKEND_FILE = Path("candy_shop_backend_ver9.py")

app = Flask(__name__)


def gen_frames():
    cap = cv2.VideoCapture(SCALE_CAM)
    if not cap.isOpened():
        raise RuntimeError(f"카메라를 열 수 없습니다: {SCALE_CAM}")

    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.1)
            continue

        if ROI_FILE.exists():
            data = json.loads(ROI_FILE.read_text(encoding="utf-8"))
            object_roi = np.array(data["object_roi"], dtype=np.int32)
            cv2.polylines(frame, [object_roi], isClosed=True, color=(0, 255, 0), thickness=2)

        x1, y1, x2, y2 = WEIGHT_DISPLAY_AABB
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

        ok, buf = cv2.imencode(".jpg", frame)
        if not ok:
            continue
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")


@app.route("/")
def index():
    return """
    <html><body style="margin:0; background:#111; text-align:center; font-family:sans-serif;">
      <h3 style="color:#eee;">저울 카메라 (초록=물체 감지 영역, 노랑=무게 읽는 영역)</h3>
      <p style="color:#ccc;">마우스로 드래그해서 박스를 그린 다음, 저장할 버튼을 눌러줘</p>
      <div style="position:relative; display:inline-block;">
        <img id="feed" src="/stream" width="640" height="480" />
        <canvas id="overlay" width="640" height="480"
          style="position:absolute; top:0; left:0; cursor:crosshair;"></canvas>
      </div>
      <div style="margin-top:12px;">
        <button onclick="saveRoi('object')" style="font-size:16px; padding:8px 16px;">✅ 물체 감지 영역으로 저장</button>
        <button onclick="saveRoi('weight')" style="font-size:16px; padding:8px 16px;">✅ 무게 읽는 영역으로 저장</button>
      </div>
      <p id="status" style="color:#0f0; font-size:14px;"></p>
      <script>
        const canvas = document.getElementById('overlay');
        const ctx = canvas.getContext('2d');
        let dragging = false, x1=0, y1=0, x2=0, y2=0;

        canvas.addEventListener('mousedown', e => {
          const r = canvas.getBoundingClientRect();
          x1 = e.clientX - r.left; y1 = e.clientY - r.top;
          x2 = x1; y2 = y1; dragging = true;
        });
        canvas.addEventListener('mousemove', e => {
          if (!dragging) return;
          const r = canvas.getBoundingClientRect();
          x2 = e.clientX - r.left; y2 = e.clientY - r.top;
          ctx.clearRect(0,0,canvas.width,canvas.height);
          ctx.strokeStyle = 'red'; ctx.lineWidth = 2;
          ctx.strokeRect(Math.min(x1,x2), Math.min(y1,y2), Math.abs(x2-x1), Math.abs(y2-y1));
        });
        window.addEventListener('mouseup', () => { dragging = false; });

        function saveRoi(kind) {
          const box = {
            x1: Math.round(Math.min(x1,x2)), y1: Math.round(Math.min(y1,y2)),
            x2: Math.round(Math.max(x1,x2)), y2: Math.round(Math.max(y1,y2)),
          };
          if (box.x2 - box.x1 < 3 || box.y2 - box.y1 < 3) {
            document.getElementById('status').textContent = '박스를 먼저 드래그해줘!';
            return;
          }
          fetch('/save_roi', {
            method: 'POST', headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({kind, ...box}),
          }).then(r => r.json()).then(data => {
            document.getElementById('status').textContent = data.message;
          });
        }
      </script>
    </body></html>
    """


@app.route("/stream")
def stream():
    return Response(gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/save_roi", methods=["POST"])
def save_roi():
    data = request.get_json()
    kind = data["kind"]
    x1, y1, x2, y2 = data["x1"], data["y1"], data["x2"], data["y2"]

    if kind == "object":
        object_roi = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
        existing = json.loads(ROI_FILE.read_text(encoding="utf-8")) if ROI_FILE.exists() else {}
        existing["object_roi"] = object_roi
        ROI_FILE.write_text(json.dumps(existing, indent=2), encoding="utf-8")
        return jsonify({"message": f"object_roi 저장 완료: {object_roi} (streamlit 재시작 필요)"})

    if kind == "weight":
        aabb = f"{x1},{y1},{x2},{y2}"
        text = BACKEND_FILE.read_text(encoding="utf-8")
        new_text, n = re.subn(r'CANDY_WEIGHT_AABB", "[^"]*"', f'CANDY_WEIGHT_AABB", "{aabb}"', text)
        if n == 0:
            return jsonify({"message": "실패: WEIGHT_DISPLAY_AABB 기본값을 못 찾음"})
        BACKEND_FILE.write_text(new_text, encoding="utf-8")
        return jsonify({"message": f"weight_aabb 저장 완료: ({x1},{y1},{x2},{y2}) (streamlit 재시작 필요)"})

    return jsonify({"message": "알 수 없는 종류"})


if __name__ == "__main__":
    print("http://localhost:5001 접속해서 확인해줘 (로봇 멈춰있을 때만)")
    app.run(host="0.0.0.0", port=5001, threaded=True)
