"""Camera analysis and bounded, in-memory replay. No network or microphone."""
from __future__ import annotations

import bisect
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

cv2.setNumThreads(2)


@dataclass
class Observation:
    face: tuple[int, int, int, int] | None
    motion: float
    present: bool


class MotionDetector:
    """Face tracking + image changes: head motion, expressions and body gestures.

    This is deliberately not an emotion/attractiveness classifier. Face-region
    differences pick up expressions; whole-frame changes also allow hand gestures.
    A recent face is required so an empty room cannot keep triggering edits.
    """

    def __init__(self):
        model = Path(__file__).resolve().parent / "assets" / "models" / "face_detection_yunet_2023mar.onnx"
        self.detector = cv2.FaceDetectorYN.create(str(model), "", (320, 192), 0.65, 0.3, 5000)
        self.previous = None
        self.previous_face = None
        self.face = None
        self.last_seen = -100.0
        self.last_scan = -100.0

    @staticmethod
    def difference(a, b):
        if a is None or b is None or a.shape != b.shape:
            return 0.0
        # Remove global brightness changes from camera auto-exposure.
        a = a.astype(np.float32)
        b = b.astype(np.float32)
        delta = np.abs((a - np.median(a)) - (b - np.median(b)))
        return float(np.mean(np.maximum(delta - 5.0, 0.0)))

    def update(self, frame, now, detected_face="auto"):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (320, 180))
        small = cv2.GaussianBlur(small, (5, 5), 0)
        if now - self.last_scan >= 0.16:
            self.last_scan = now
            if isinstance(detected_face, str):
                # Pad to a multiple of 32 for the ONNX detector; motion coordinates
                # still refer to the unpadded 320x180 image.
                detector_frame = cv2.resize(frame, (320, 180))
                detector_frame = cv2.copyMakeBorder(detector_frame, 0, 12, 0, 0, cv2.BORDER_CONSTANT)
                _, faces = self.detector.detect(detector_frame)
                box = max(faces, key=lambda b: b[2] * b[3])[:4] if faces is not None else None
            else:
                box = detected_face  # Explicit injectable observation for tests.
            if box is not None:
                self.face = tuple(int(v) for v in box)
                self.last_seen = now
        present = now - self.last_seen < 1.4
        if not present:
            self.face = None
        face_now = None
        if self.face:
            x, y, w, h = self.face
            roi = small[max(0, y):min(180, y + h), max(0, x):min(320, x + w)]
            if roi.size:
                face_now = cv2.resize(roi, (80, 80))
        expression = self.difference(face_now, self.previous_face)
        movement = self.difference(small, self.previous)
        score = min(100.0, max(expression * 5.0, movement * 4.0)) if present else 0.0
        self.previous = small
        self.previous_face = face_now
        scaled = None
        if self.face:
            sx, sy = frame.shape[1] / 320, frame.shape[0] / 180
            x, y, w, h = self.face
            scaled = (int(x * sx), int(y * sy), int(w * sx), int(h * sy))
        return Observation(scaled, score, present)


class TriggerGate:
    """Debounce motion and never queue a trigger during an edit or its cooldown."""

    def __init__(self, sensitivity=85, cooldown=2.5):
        self.sensitivity = sensitivity
        self.cooldown = cooldown
        self.ready_at = 0.0
        self.active_since = None

    @property
    def threshold(self):
        return 2.0 + (100 - self.sensitivity) * 0.18

    def reset(self, now, delay=1.0):
        self.ready_at = now + delay
        self.active_since = None

    def check(self, observation, now, busy=False, buffered_seconds=0.0):
        if busy or now < self.ready_at or buffered_seconds < 1.0:
            self.active_since = None
            return False
        if not observation.present or observation.motion < self.threshold:
            self.active_since = None
            return False
        if self.active_since is None:
            self.active_since = now
        if now - self.active_since < 0.05:
            return False
        self.reset(now, self.cooldown)
        return True


class ReplayBuffer:
    def __init__(self, seconds=6.0, fps=15):
        self.seconds = seconds
        self.fps = fps
        self.frames = deque(maxlen=int(seconds * fps) + 2)
        self.last = -100.0

    def push(self, frame, now):
        if now - self.last < 1.0 / self.fps:
            return
        self.last = now
        picture = cv2.resize(frame, (640, 360))
        ok, data = cv2.imencode(".jpg", picture, [cv2.IMWRITE_JPEG_QUALITY, 88])
        if ok:
            self.frames.append((now, data.tobytes()))
        while self.frames and now - self.frames[0][0] > self.seconds:
            self.frames.popleft()

    @property
    def duration(self):
        return self.frames[-1][0] - self.frames[0][0] if len(self.frames) > 1 else 0.0

    def snapshot(self):
        return list(self.frames)


class SelfEdit:
    """A six-second montage from real, timestamped webcam frames.

    Rhythm cuts, slow motion, punch-ins, freeze, ghosting and split frames.
    Used for both live playback and MP4 export, so the export matches playback.
    """

    duration = 6.0
    fps = 30

    def __init__(self, snapshot, variant=0):
        if len(snapshot) < 2:
            raise ValueError("Для эдита нужна хотя бы секунда записи с камеры.")
        self.variant = variant
        self.times = [x[0] for x in snapshot]
        self.frames = [cv2.imdecode(np.frombuffer(x[1], np.uint8), cv2.IMREAD_COLOR) for x in snapshot]
        if any(x is None for x in self.frames):
            raise ValueError("Не удалось прочитать кадр вебки.")
        self.span = max(0.01, self.times[-1] - self.times[0])

    def frame_at(self, t):
        t = min(max(t, 0.0), self.duration - 0.001)
        beat = int(t / 0.4)
        phase = (t % 0.4) / 0.4
        if t < 1.2:
            pos = 0.48 + t / 1.2 * 0.44
        elif t < 2.0:
            pos = 0.93  # Beat-drop freeze.
        elif t < 4.4:
            pos = (0.20 + (t - 2.0) * 0.32) % 1.0
        else:
            pos = 0.58 + (t - 4.4) * 0.23
        target = self.times[0] + min(1.0, pos) * self.span
        i = min(len(self.frames) - 1, bisect.bisect_left(self.times, target))
        frame = self.frames[i].copy()
        h, w = frame.shape[:2]
        punch = max(0.0, 1.0 - phase * 5)
        zoom = 1.08 + 0.10 * punch + (0.12 if 1.2 <= t < 2.0 else 0.0)
        cw, ch = int(w / zoom), int(h / zoom)
        dx = int(np.sin(t * 58) * punch * 9)
        dy = int(np.cos(t * 47) * punch * 6)
        x = max(0, min(w - cw, (w - cw) // 2 + dx))
        y = max(0, min(h - ch, (h - ch) // 2 + dy))
        frame = cv2.resize(frame[y:y + ch, x:x + cw], (w, h))
        frame = np.clip(frame.astype(np.float32) * 1.18 - 14, 0, 255).astype(np.uint8)
        if self.variant % 3 == 1:
            mono = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.cvtColor(mono, cv2.COLOR_GRAY2BGR)
        else:
            frame[:, :, 0] = cv2.add(frame[:, :, 0], 12)
        if punch > 0.60:
            frame[:, :, 2] = np.roll(frame[:, :, 2], 4, axis=1)
            frame[:, :, 0] = np.roll(frame[:, :, 0], -3, axis=1)
            frame = cv2.convertScaleAbs(frame, alpha=1, beta=18 * punch)
        if 3.2 <= t < 4.4:
            ghost = np.roll(frame, 28, axis=1)
            frame = cv2.addWeighted(frame, 0.78, ghost, 0.22, 0)
        if 4.4 <= t < 5.2:
            side = cv2.resize(frame, (w // 3, h))
            frame = np.concatenate([side, cv2.flip(side, 1), side], axis=1)
            frame = cv2.resize(frame, (w, h))
        if t >= 1.2:
            text = ["+999 AURA", "LOCKED IN", "ABSOLUTE CINEMA"][self.variant % 3]
            scale = 0.95 if len(text) < 15 else 0.78
            size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, scale, 2)
            p = ((w - size[0]) // 2, h - 33)
            cv2.putText(frame, text, p, cv2.FONT_HERSHEY_DUPLEX, scale, (8, 8, 8), 6, cv2.LINE_AA)
            color = (210, 255, 150) if self.variant % 3 != 1 else (235, 235, 235)
            cv2.putText(frame, text, p, cv2.FONT_HERSHEY_DUPLEX, scale, color, 2, cv2.LINE_AA)
        return frame
