import base64
import threading
import time

from gevent import monkey
monkey.patch_all()

import cv2
import mediapipe as mp
from flask import Flask
from flask_socketio import SocketIO


app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="gevent")


class CameraStreamer:
    def __init__(self) -> None:
        self._thread = None
        self._lock = threading.Lock()
        self._running = False

    def start(self) -> None:
        with self._lock:
            if self._running:
                return
            self._running = True
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()

    def _run(self) -> None:
        mp_face = mp.solutions.face_mesh
        face_mesh = mp_face.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
        )

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            self._running = False
            return

        try:
            while self._running:
                success, frame = cap.read()
                if not success:
                    break

                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = face_mesh.process(rgb)

                direction = None
                if result.multi_face_landmarks:
                    face_landmarks = result.multi_face_landmarks[0]
                    nose = face_landmarks.landmark[1]
                    direction = self._nose_direction(nose)
                    self._draw_nose_direction(frame, nose, direction)

                ok, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ok:
                    payload = base64.b64encode(buffer).decode("ascii")
                    socketio.emit("frame", {"image": payload})
                    if direction:
                        socketio.emit("direction", {"direction": direction})

                time.sleep(0.03)
        finally:
            cap.release()
            self._running = False

    def _nose_direction(self, nose):
        dx = nose.x - 0.5
        dy = nose.y - 0.5
        threshold = 0.08
        if abs(dx) < threshold and abs(dy) < threshold:
            return None
        if abs(dx) > abs(dy):
            return "RIGHT" if dx > 0 else "LEFT"
        return "DOWN" if dy > 0 else "UP"

    def _draw_nose_direction(self, frame, nose, direction) -> None:
        cx = int(nose.x * frame.shape[1])
        cy = int(nose.y * frame.shape[0])
        cv2.circle(frame, (cx, cy), 6, (126, 240, 193), -1)
        if not direction:
            return
        arrow_len = 140
        if direction == "RIGHT":
            end = (cx + arrow_len, cy)
        elif direction == "LEFT":
            end = (cx - arrow_len, cy)
        elif direction == "UP":
            end = (cx, cy - arrow_len)
        else:
            end = (cx, cy + arrow_len)
        cv2.arrowedLine(
            frame,
            (cx, cy),
            end,
            (0, 255, 255),
            6,
            tipLength=0.3,
        )
        cv2.putText(
            frame,
            direction,
            (cx - 50, cy + 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.1,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )


streamer = CameraStreamer()


@app.get("/")
def health_check():
    return {"status": "ok"}


@socketio.on("connect")
def handle_connect():
    streamer.start()


if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=5000)
