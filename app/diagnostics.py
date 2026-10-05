"""Offline dependency diagnostic; does not open the camera."""
from pathlib import Path
import os
import platform
import sys
import traceback

lines = ["AuraCam diagnostics", platform.platform(), sys.version]
handles = []
if sys.platform == "win32":
    vendor = Path(sys.executable).parent / "Lib/site-packages"
    for name in ("PySide6", "shiboken6", "numpy.libs", "cv2"):
        if (vendor / name).is_dir():
            handles.append(os.add_dll_directory(str(vendor / name)))
try:
    import cv2
    import numpy
    from PySide6 import QtCore, QtGui, QtWidgets, QtMultimedia
    from engine import MotionDetector
    from media import clips, soundtrack, ffmpeg_path
    MotionDetector()
    lines.extend([f"OpenCV: {cv2.__version__}", f"NumPy: {numpy.__version__}",
                  f"Qt: {QtCore.qVersion()}", f"Videos: {len(clips())}",
                  f"Soundtrack: {soundtrack()}", f"FFmpeg: {ffmpeg_path()}", "IMPORTS OK"])
except Exception:
    lines.append(traceback.format_exc())
text = "\n".join(lines)
print(text)
target = Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "AuraCam"
target.mkdir(parents=True, exist_ok=True)
(target / "diagnostics.txt").write_text(text, encoding="utf-8")
print("Report:", target / "diagnostics.txt")
