"""AuraCam — local webcam confidence booster. Run: python main.py."""
from __future__ import annotations

import argparse
import logging
import math
import os
from pathlib import Path
import sys
import time

import cv2
from PySide6.QtCore import QPointF, QRectF, Qt, QTimer, QUrl, QSettings
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen, QPolygonF, QShortcut, QKeySequence
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer, QVideoSink
from PySide6.QtWidgets import (
    QApplication, QComboBox, QFileDialog, QHBoxLayout, QLabel, QMainWindow,
    QMessageBox, QPushButton, QSlider, QVBoxLayout, QWidget,
)

from camera import CameraThread
from engine import ReplayBuffer, SelfEdit, TriggerGate
from media import ASSETS, ExportThread, clips, soundtrack

GREEN = QColor("#adffc2")


def data_directory():
    override = os.environ.get("AURACAM_DATA_DIR")
    if override:
        root = Path(override)
    elif sys.platform == "win32":
        root = Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "AuraCam"
    else:
        root = Path.home() / ".auracam"
    root.mkdir(parents=True, exist_ok=True)
    return root


def qimage(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QImage.Format_RGB888).copy()


class Monitor(QWidget):
    def __init__(self):
        super().__init__()
        self.setMinimumSize(640, 360)
        self.image = QImage()
        self.overlay = QImage()
        self.face = None
        self.strength = 0.0
        self.state = "CAMERA OFF"
        self.caption = "Нажми «Старт», чтобы включить вебку"
        self.started = time.monotonic()
        self.active = False
        self.self_mode = False
        self.progress = 0.0
        self.edit_count = 0

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(), QColor("#090e0c"))
        width = min(self.width() - 20, (self.height() - 20) * 16 / 9)
        height = width * 9 / 16
        view = QRectF((self.width() - width) / 2, (self.height() - height) / 2, width, height)
        if not self.image.isNull():
            p.drawImage(view, self.image)
        else:
            p.setPen(QPen(QColor("#17281f"), 1))
            for x in range(int(view.left()), int(view.right()), 40):
                p.drawLine(x, int(view.top()), x, int(view.bottom()))
            for y in range(int(view.top()), int(view.bottom()), 40):
                p.drawLine(int(view.left()), y, int(view.right()), y)
            p.setFont(QFont("Consolas", 23, QFont.Bold))
            p.setPen(GREEN)
            p.drawText(view.adjusted(20, -24, -20, -24), Qt.AlignCenter, "AURA / MONITOR")
            p.setFont(QFont("Segoe UI", 11))
            p.setPen(QColor("#93a59b"))
            p.drawText(view.adjusted(20, 45, -20, 45), Qt.AlignCenter, self.caption)

        p.setPen(QPen(QColor(173, 255, 194, 150), 1))
        p.drawRect(view)
        # Corner marks reproduce the thin green monitor HUD from the reference.
        for x, y, dx, dy in [(view.left(), view.top(), 1, 1), (view.right(), view.top(), -1, 1),
                              (view.left(), view.bottom(), 1, -1), (view.right(), view.bottom(), -1, -1)]:
            p.setPen(QPen(GREEN, 2))
            p.drawLine(QPointF(x, y), QPointF(x + 13 * dx, y))
            p.drawLine(QPointF(x, y), QPointF(x, y + 13 * dy))
        sx, sy = view.width() / 960, view.height() / 540
        if self.face and self.active:
            x, y, w, h = self.face
            box = QRectF(view.left() + x * sx, view.top() + y * sy, w * sx, h * sy)
            back = box.translated(8 * sx, -8 * sy)
            p.setPen(QPen(GREEN, 1))
            p.drawRect(box)
            p.drawRect(back)
            for a, b in [(box.topLeft(), back.topLeft()), (box.topRight(), back.topRight()),
                          (box.bottomLeft(), back.bottomLeft()), (box.bottomRight(), back.bottomRight())]:
                p.drawLine(a, b)
            self.label(p, box.left(), box.bottom() + 14, f"SUBJECT 01  /  MOTION {int(self.strength):02d}", 8)

        pad = 14 * sx
        if not self.overlay.isNull():
            if self.self_mode or self.overlay.width() > self.overlay.height():
                ow = view.width() * 0.61
            else:
                ow = view.width() * 0.30
            oh = ow * self.overlay.height() / max(1, self.overlay.width())
            if oh > view.height() * 0.78:
                oh = view.height() * 0.78
                ow = oh * self.overlay.width() / self.overlay.height()
            area = QRectF(view.right() - ow - pad, view.top() + 23 * sy, ow, oh)
            p.fillRect(area.adjusted(-1, -1, 1, 1), QColor("#061009"))
            p.drawImage(area, self.overlay)
            p.setPen(QPen(GREEN, 1))
            p.drawRect(area)
            if self.self_mode:
                self.label(p, area.left() + 7, area.top() + 16, "YOUR AURA / REPLAY", 8)
            p.fillRect(QRectF(area.left(), area.bottom() - 2, area.width() * self.progress, 2), GREEN)
        else:
            area = QRectF(view.right() - view.width() * 0.245 - pad, view.top() + 25 * sy,
                          view.width() * 0.245, view.height() * 0.30)
            p.fillRect(area, QColor(3, 13, 10, 200))
            p.setPen(QPen(QColor(173, 255, 194, 140), 1))
            p.drawRect(area)
            c = area.center()
            radius = area.height() * 0.26
            theta = time.monotonic() * 0.28
            vertices = [QPointF(c.x() + math.cos(theta + i * math.pi / 3) * radius,
                               c.y() + math.sin(theta + i * math.pi / 3) * radius) for i in range(6)]
            p.drawPolygon(QPolygonF(vertices))
            p.drawLine(QPointF(c.x() - radius * 1.5, c.y()), QPointF(c.x() + radius * 1.5, c.y()))
            p.drawLine(QPointF(c.x(), c.y() - radius * 1.5), QPointF(c.x(), c.y() + radius * 1.5))
            p.setFont(QFont("Consolas", 8, QFont.Bold))
            p.setPen(GREEN)
            p.drawText(area, Qt.AlignCenter, self.state)
        self.label(p, view.left() + 10, view.top() + 19, "● LIVE" if self.active else "○ OFFLINE", 9)
        self.label(p, view.right() - 63, view.top() + 17, "CAM 01", 8)
        seconds = max(0, int(time.monotonic() - self.started)) if self.active else 0
        clock = f"{seconds // 3600:02}:{seconds // 60 % 60:02}:{seconds % 60:02}"
        self.label(p, view.left() + 10, view.bottom() - 11, f"REC  {clock}", 9)
        self.label(p, view.right() - 110, view.bottom() - 11, f"EDITS  {self.edit_count:03}", 9)
        p.end()

    @staticmethod
    def label(p, x, y, text, size):
        p.setFont(QFont("Consolas", size, QFont.Bold))
        p.setPen(QColor(0, 0, 0, 190))
        p.drawText(QPointF(x + 1, y + 1), text)
        p.setPen(GREEN)
        p.drawText(QPointF(x, y), text)


class MainWindow(QMainWindow):
    def __init__(self, no_camera=False, test_source=None):
        super().__init__()
        self.setWindowTitle("AuraCam — confidence booster")
        self.resize(1120, 755)
        self.setMinimumSize(760, 555)
        self.worker = None
        self.exporter = None
        self.closing = False
        self.test_source = test_source
        self.buffer = ReplayBuffer()
        self.gate = TriggerGate()
        self.busy = False
        self.active_edit = None
        self.last_edit = None
        self.edit_started = 0.0
        self.action_count = 0
        self.clip_index = 0
        self.settings = QSettings(str(data_directory() / "settings.ini"), QSettings.IniFormat)
        self.monitor = Monitor()
        self.audio_output = QAudioOutput(self)
        self.player = QMediaPlayer(self)
        self.player.setAudioOutput(self.audio_output)
        self.sink = QVideoSink(self)
        self.player.setVideoSink(self.sink)
        self.sink.videoFrameChanged.connect(self.video_frame)
        self.player.mediaStatusChanged.connect(self.media_status)
        self.player.errorOccurred.connect(self.media_error)

        root = QWidget()
        layout = QVBoxLayout(root)
        layout.setContentsMargins(18, 14, 18, 12)
        layout.setSpacing(10)
        header = QHBoxLayout()
        title = QLabel("AURA / MONITOR")
        title.setObjectName("title")
        header.addWidget(title)
        header.addStretch()
        tagline = QLabel("my confidence booster app")
        tagline.setObjectName("subtle")
        header.addWidget(tagline)
        layout.addLayout(header)
        layout.addWidget(self.monitor, 1)

        controls = QHBoxLayout()
        self.start_button = QPushButton("Старт")
        self.start_button.setObjectName("primary")
        self.start_button.clicked.connect(self.toggle_camera)
        controls.addWidget(self.start_button)
        self.camera_choice = QComboBox()
        self.camera_choice.addItems([f"Камера {i + 1}" for i in range(4)])
        self.camera_choice.setCurrentIndex(int(self.settings.value("camera", 0)))
        controls.addWidget(self.camera_choice)
        self.mode = QComboBox()
        self.mode.addItems(["Микс: я + эдиты", "Только мой эдит", "Только видеоэдиты"])
        self.mode.setCurrentIndex(int(self.settings.value("mode", 0)))
        controls.addWidget(self.mode)
        controls.addStretch()
        self.now_button = QPushButton("Эдит сейчас")
        self.now_button.clicked.connect(self.manual_edit)
        self.now_button.setEnabled(False)
        controls.addWidget(self.now_button)
        self.save_button = QPushButton("Сохранить мой эдит")
        self.save_button.clicked.connect(self.save_edit)
        self.save_button.setEnabled(False)
        controls.addWidget(self.save_button)
        layout.addLayout(controls)

        tuning = QHBoxLayout()
        tuning.addWidget(QLabel("Чувствительность"))
        self.sensitivity = QSlider(Qt.Horizontal)
        self.sensitivity.setRange(10, 100)
        self.sensitivity.setValue(int(self.settings.value("sensitivity", 85)))
        self.sensitivity.setMaximumWidth(170)
        self.sensitivity.valueChanged.connect(lambda v: setattr(self.gate, "sensitivity", v))
        self.gate.sensitivity = self.sensitivity.value()
        tuning.addWidget(self.sensitivity)
        tuning.addSpacing(14)
        tuning.addWidget(QLabel("Звук"))
        self.volume = QSlider(Qt.Horizontal)
        self.volume.setRange(0, 100)
        self.volume.setValue(int(self.settings.value("volume", 65)))
        self.volume.setMaximumWidth(120)
        self.volume.valueChanged.connect(lambda v: self.audio_output.setVolume(v / 100))
        self.audio_output.setVolume(self.volume.value() / 100)
        tuning.addWidget(self.volume)
        tuning.addStretch()
        self.full_button = QPushButton("На весь экран · F11")
        self.full_button.clicked.connect(self.toggle_fullscreen)
        tuning.addWidget(self.full_button)
        layout.addLayout(tuning)
        self.status = QLabel("Поверни голову, улыбнись или покажи жест — и поймай свой эдит.")
        self.status.setWordWrap(True)
        self.status.setObjectName("subtle")
        self.status.setMinimumHeight(30)
        layout.addWidget(self.status)
        self.setCentralWidget(root)
        self.setStyleSheet("""
            QMainWindow, QWidget { background: #111713; color: #d9e7dd; font-family: 'Segoe UI'; font-size: 12px; }
            QLabel#title { color: #adffc2; font-family: 'Consolas'; font-size: 16px; font-weight: bold; }
            QLabel#subtle { color: #8eaa98; }
            QPushButton, QComboBox { background: #1c2920; border: 1px solid #354b3c;
                border-radius: 4px; padding: 8px 11px; min-height: 18px; }
            QPushButton:hover { background: #2a3f30; border-color: #8deba5; }
            QPushButton:disabled { color: #5c6c60; border-color: #25382a; }
            QPushButton#primary { background: #b3f8c6; color: #102016; font-weight: bold; min-width: 62px; }
            QComboBox::drop-down { border: none; width: 20px; }
            QSlider::groove:horizontal { height: 3px; background: #334a3b; }
            QSlider::handle:horizontal { background: #adffc2; width: 12px; margin: -5px 0; border-radius: 6px; }
            QSlider::sub-page:horizontal { background: #78bc8b; }
        """)
        self.tick_timer = QTimer(self)
        self.tick_timer.timeout.connect(self.tick)
        self.tick_timer.start(33)
        QShortcut(QKeySequence("F11"), self, self.toggle_fullscreen)
        QShortcut(QKeySequence("Escape"), self, self.escape)
        QShortcut(QKeySequence("Space"), self, self.manual_edit)
        if not no_camera:
            QTimer.singleShot(250, self.start_camera)

    def toggle_fullscreen(self):
        self.showNormal() if self.isFullScreen() else self.showFullScreen()

    def escape(self):
        if self.isFullScreen():
            self.showNormal()
        elif self.busy:
            self.finish_edit()

    def toggle_camera(self):
        if self.worker is not None:
            self.stop_camera()
        else:
            self.start_camera()

    def start_camera(self):
        if self.worker is not None:
            return
        self.buffer = ReplayBuffer()
        self.gate.reset(time.monotonic(), 2.0)
        self.camera_choice.setEnabled(False)
        self.start_button.setText("Стоп")
        self.status.setText("Подключаю камеру…")
        self.worker = CameraThread(self.camera_choice.currentIndex(), self.test_source, self)
        self.worker.frame_ready.connect(self.on_frame)
        self.worker.failed.connect(self.camera_error)
        self.worker.finished.connect(self.camera_finished)
        self.worker.start()

    def stop_camera(self):
        self.finish_edit()
        self.monitor.active = False
        if self.worker is not None:
            self.monitor.caption = "Нажми «Старт», чтобы включить вебку"
            self.worker.requestInterruption()
            self.start_button.setEnabled(False)
            self.status.setText("Отключаю камеру…")

    def camera_finished(self):
        self.finish_edit()
        if self.worker is not None:
            self.worker.deleteLater()
            self.worker = None
        self.buffer = ReplayBuffer()
        self.monitor.image = QImage()
        self.monitor.face = None
        self.monitor.active = False
        self.monitor.state = "CAMERA OFF"
        self.start_button.setText("Старт")
        self.start_button.setEnabled(True)
        self.camera_choice.setEnabled(True)
        self.now_button.setEnabled(False)
        if self.status.text() == "Отключаю камеру…":
            self.status.setText("Камера выключена. Нажми «Старт», чтобы продолжить.")
        if self.closing:
            self.close()

    def camera_error(self, message):
        logging.error("Camera: %s", message)
        self.status.setText(message)
        self.monitor.caption = "Камера недоступна — проверь подключение и доступ в Windows"

    def on_frame(self, frame, observation, now):
        if self.worker:
            self.worker.pending = False
            if self.worker.isInterruptionRequested():
                return
        if not self.monitor.active:
            self.monitor.started = now
        self.monitor.active = True
        self.monitor.image = qimage(frame)
        self.monitor.face = observation.face
        self.monitor.strength = observation.motion
        self.buffer.push(frame, now)
        self.now_button.setEnabled(self.buffer.duration >= 1 and not self.busy)
        if not self.busy:
            remaining = max(0, self.gate.ready_at - now)
            if self.buffer.duration < 1:
                self.monitor.state = "BUFFERING…"
                self.status.setText("Собираю первый фрагмент вебки…")
            elif remaining > 0:
                self.monitor.state = "RECHARGING"
                self.status.setText(f"Следующий эдит через {remaining:.1f} с — продолжай двигаться.")
            elif not observation.present:
                self.monitor.state = "FINDING YOU"
                self.status.setText("Покажи лицо камере. При плохом свете или в профиль лицо может не определяться.")
            else:
                self.monitor.state = "WAITING…"
                self.status.setText("Вижу тебя. Двигай головой, меняй мимику или покажи жест.")
        if self.gate.check(observation, now, self.busy, self.buffer.duration):
            self.start_edit()

    def manual_edit(self):
        if not self.busy and self.monitor.active and self.buffer.duration >= 1:
            self.start_edit()

    def start_edit(self):
        available = clips()
        mode = self.mode.currentIndex()
        own = mode == 1 or not available or (mode == 0 and self.action_count % 2 == 0)
        self.action_count += 1
        self.monitor.edit_count = self.action_count
        self.busy = True
        self.now_button.setEnabled(False)
        self.monitor.overlay = QImage()
        self.monitor.self_mode = own
        self.edit_started = time.monotonic()
        self.player.stop()
        self.player.setSource(QUrl())
        if own:
            try:
                self.active_edit = SelfEdit(self.buffer.snapshot(), (self.action_count - 1) // 2)
            except ValueError as exc:
                self.finish_edit()
                self.status.setText(str(exc))
                return
            self.last_edit = self.active_edit
            self.save_button.setEnabled(self.exporter is None)
            track = soundtrack(self.active_edit.variant)
            if track:
                self.player.setSource(QUrl.fromLocalFile(str(track)))
                self.player.play()
            self.status.setText("ТВОЙ ЭДИТ — последние секунды вебки, музыка и +999 AURA.")
        else:
            self.active_edit = None
            path = available[self.clip_index % len(available)]
            self.clip_index += 1
            self.player.setSource(QUrl.fromLocalFile(str(path)))
            self.player.play()
            self.status.setText("AURA EDIT — фрагмент из твоего референса.")

    def video_frame(self, frame):
        if self.busy and not self.monitor.self_mode:
            image = frame.toImage()
            if not image.isNull():
                self.monitor.overlay = image

    def media_status(self, status):
        if status == QMediaPlayer.EndOfMedia and self.busy and not self.monitor.self_mode:
            self.finish_edit()

    def media_error(self, error, message):
        if error != QMediaPlayer.NoError:
            logging.error("Media: %s", message)
            if self.busy and not self.monitor.self_mode:
                self.finish_edit()
            self.status.setText("Не удалось воспроизвести видео/звук: " + message)

    def finish_edit(self):
        if not self.busy:
            return
        self.busy = False
        self.player.stop()
        self.monitor.overlay = QImage()
        self.active_edit = None
        self.gate.reset(time.monotonic(), self.gate.cooldown)

    def tick(self):
        if self.busy:
            elapsed = time.monotonic() - self.edit_started
            if self.active_edit:
                if elapsed >= self.active_edit.duration:
                    self.finish_edit()
                else:
                    self.monitor.overlay = qimage(self.active_edit.frame_at(elapsed))
                    self.monitor.progress = elapsed / self.active_edit.duration
            else:
                duration = self.player.duration()
                self.monitor.progress = self.player.position() / duration if duration else 0
                if elapsed >= 12:
                    self.finish_edit()
        self.monitor.update()

    def save_edit(self):
        if self.last_edit is None or self.exporter is not None:
            return
        name, _ = QFileDialog.getSaveFileName(self, "Сохранить твой эдит", "my-aura-edit.mp4", "MP4 (*.mp4)")
        if not name:
            return
        if not name.lower().endswith(".mp4"):
            name += ".mp4"
        self.exporter = ExportThread(self.last_edit, name, self)
        self.exporter.progress.connect(lambda v: self.save_button.setText(f"Сохранение {v}%"))
        self.exporter.completed.connect(self.export_done)
        self.exporter.failed.connect(self.export_failed)
        self.exporter.finished.connect(self.export_finished)
        self.save_button.setEnabled(False)
        self.exporter.start()

    def export_done(self, path):
        QMessageBox.information(self, "Эдит сохранён", f"Готово! Видео с музыкой:\n{path}")

    def export_failed(self, message):
        logging.error("Export: %s", message)
        if not self.closing:
            QMessageBox.warning(self, "Не удалось сохранить эдит", message)

    def export_finished(self):
        self.exporter.deleteLater()
        self.exporter = None
        self.save_button.setText("Сохранить мой эдит")
        self.save_button.setEnabled(self.last_edit is not None)
        if self.closing:
            self.close()

    def closeEvent(self, event):
        self.closing = True
        self.player.stop()
        for name, value in [("camera", self.camera_choice.currentIndex()), ("mode", self.mode.currentIndex()),
                            ("sensitivity", self.sensitivity.value()), ("volume", self.volume.value())]:
            self.settings.setValue(name, value)
        self.settings.sync()
        pending = False
        if self.worker is not None:
            self.worker.requestInterruption()
            pending = True
        if self.exporter is not None:
            self.exporter.requestInterruption()
            pending = True
        if pending:
            self.status.setText("Закрываю камеру и освобождаю память…")
            event.ignore()
        else:
            self.last_edit = None
            event.accept()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-camera", action="store_true", help="Open without accessing camera")
    parser.add_argument("--test-video", type=Path, help="Use a local video as an integration test source")
    args = parser.parse_args()
    logging.basicConfig(filename=data_directory() / "auracam.log", level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s", encoding="utf-8")
    app = QApplication(sys.argv)
    app.setApplicationName("AuraCam")
    app.setOrganizationName("AuraCam")
    window = MainWindow(args.no_camera, args.test_video)
    window.show()
    return app.exec()


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        import traceback
        details = traceback.format_exc()
        try:
            (data_directory() / "startup-error.txt").write_text(details, encoding="utf-8")
        except OSError:
            pass
        if sys.platform == "win32":
            import ctypes
            ctypes.windll.user32.MessageBoxW(None, "AuraCam не запустился. Подробности:\n" + details[-1800:],
                                            "AuraCam", 0x10)
        else:
            print(details, file=sys.stderr)
        sys.exit(1)
