"""Entry point that can show a useful error even if a DLL fails to import."""
from pathlib import Path
import os
import sys
import traceback

handles = []
try:
    if sys.platform == "win32":
        vendor = Path(sys.executable).parent / "Lib" / "site-packages"
        for name in ("PySide6", "shiboken6", "numpy.libs", "cv2"):
            directory = vendor / name
            if directory.is_dir():
                handles.append(os.add_dll_directory(str(directory)))
    from main import main
    code = main()
except Exception:
    details = traceback.format_exc()
    directory = Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "AuraCam"
    try:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "startup-error.txt").write_text(details, encoding="utf-8")
    except OSError:
        pass
    if sys.platform == "win32":
        import ctypes
        ctypes.windll.user32.MessageBoxW(None,
            "AuraCam не запустился. Пришли этот текст разработчику:\n\n" + details[-2000:],
            "AuraCam — ошибка запуска", 0x10)
    elif sys.stderr:
        sys.stderr.write(details)
    code = 1
sys.exit(code)
