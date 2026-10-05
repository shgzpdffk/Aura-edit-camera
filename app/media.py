"""Local media lookup and matching MP4 export."""
from pathlib import Path
import os
import subprocess
import sys

from PySide6.QtCore import QThread, Signal

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "assets"


def clips():
    return sorted((ASSETS / "edits").glob("*.mp4"))


def soundtrack(variant=0):
    tracks = sorted((ASSETS / "sounds").glob("*.wav"))
    return tracks[variant % len(tracks)] if tracks else None


def ffmpeg_path():
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


class ExportThread(QThread):
    progress = Signal(int)
    completed = Signal(str)
    failed = Signal(str)

    def __init__(self, edit, path, parent=None):
        super().__init__(parent)
        self.edit = edit
        self.path = Path(path)

    def run(self):
        # Write to a sibling temporary file first. A failed export never damages
        # a previous video that the user selected for replacement.
        temp = self.path.with_name(self.path.stem + ".auracam-part.mp4")
        process = None
        try:
            cmd = [ffmpeg_path(), "-hide_banner", "-loglevel", "error", "-y",
                   "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", "640x360",
                   "-r", "30", "-i", "pipe:0"]
            track = soundtrack(self.edit.variant)
            if track:
                cmd += ["-stream_loop", "-1", "-i", str(track)]
            cmd += ["-t", "6", "-c:v", "libx264", "-preset", "veryfast",
                    "-crf", "19", "-pix_fmt", "yuv420p"]
            if track:
                cmd += ["-c:a", "aac", "-b:a", "192k", "-af", "afade=t=out:st=5.7:d=0.3"]
            cmd += ["-movflags", "+faststart", str(temp)]
            kwargs = {"creationflags": subprocess.CREATE_NO_WINDOW} if sys.platform == "win32" else {}
            process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                                       stderr=subprocess.PIPE, **kwargs)
            for i in range(180):
                if self.isInterruptionRequested():
                    raise InterruptedError("Сохранение отменено.")
                process.stdin.write(self.edit.frame_at(i / 30).tobytes())
                if i % 15 == 0:
                    self.progress.emit(int(i / 180 * 100))
            process.stdin.close()
            process.stdin = None
            _, error = process.communicate(timeout=60)
            if process.returncode:
                raise RuntimeError(error.decode("utf-8", "replace")[-1500:])
            os.replace(temp, self.path)
            self.completed.emit(str(self.path))
        except Exception as exc:
            if process is not None and process.poll() is None:
                process.kill()
                process.communicate()
            temp.unlink(missing_ok=True)
            self.failed.emit(str(exc))
