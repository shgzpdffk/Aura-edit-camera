import sys
import time

import cv2
from PySide6.QtCore import QThread, Signal

from engine import MotionDetector


class CameraThread(QThread):
    frame_ready = Signal(object, object, float)
    failed = Signal(str)
    opened = Signal()

    def __init__(self, index=0, source=None, parent=None):
        super().__init__(parent)
        self.index = index
        self.source = source
        self.pending = False

    def run(self):
        capture = None
        try:
            detector = MotionDetector()
            if self.source:
                capture = cv2.VideoCapture(str(self.source))
            else:
                backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
                capture = cv2.VideoCapture(self.index, backend)
                if not capture.isOpened() and sys.platform == "win32":
                    capture.release()
                    capture = cv2.VideoCapture(self.index, cv2.CAP_MSMF)
                capture.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
                capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
                capture.set(cv2.CAP_PROP_FPS, 30)
            if not capture.isOpened():
                raise RuntimeError("Камера недоступна. Закрой приложения, которые её используют, "
                                   "проверь доступ к камере в параметрах Windows или выбери другую камеру.")
            self.opened.emit()
            failures = 0
            while not self.isInterruptionRequested():
                start = time.monotonic()
                ok, frame = capture.read()
                if not ok:
                    if self.source:
                        capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    failures += 1
                    if failures > 40:
                        raise RuntimeError("Камера перестала отдавать кадры. Проверь подключение и нажми «Старт».")
                    self.msleep(40)
                    continue
                failures = 0
                frame = cv2.flip(frame, 1) if not self.source else frame
                frame = fit_16_9(frame)
                observation = detector.update(frame, start)
                # At most one frame queued for the UI. Slow computers do not
                # accumulate old frames, memory or several seconds of latency.
                if not self.pending:
                    self.pending = True
                    self.frame_ready.emit(frame, observation, start)
                remaining = 1 / 30 - (time.monotonic() - start)
                if remaining > 0:
                    self.msleep(int(remaining * 1000))
        except Exception as exc:
            self.failed.emit(str(exc))
        finally:
            if capture is not None:
                capture.release()


def fit_16_9(frame):
    h, w = frame.shape[:2]
    if w / h > 16 / 9:
        crop = int(h * 16 / 9)
        x = (w - crop) // 2
        frame = frame[:, x:x + crop]
    else:
        crop = int(w * 9 / 16)
        y = (h - crop) // 2
        frame = frame[y:y + crop]
    return cv2.resize(frame, (960, 540))
