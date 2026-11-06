"""AuNW Analysis as a desktop application.

Double-click the desktop shortcut (created by install_windows.bat), or run
    pythonw desktop.py        (Windows, no console)
    python desktop.py         (any system)

What happens:
1. The analysis server (Streamlit) is started invisibly on a free local port.
2. As soon as it answers, a native application window shows the interface:
   - pywebview (Windows: Edge WebView2) if installed, otherwise
   - Microsoft Edge or Google Chrome in "app mode" (a window without tabs or
     address bar), otherwise
   - the default web browser.
3. Closing the window stops the server. Nothing is reachable from other
   computers; everything stays on this machine.
"""
from __future__ import annotations

import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TITLE = "AuNW Analysis"
ICON = ROOT / "assets" / "aunw.ico" if os.name == "nt" else ROOT / "assets" / "aunw.png"
DATA_HOME = Path(os.environ.get("LOCALAPPDATA", Path.home() / ".local" / "share")) / "AuNWAnalysis"
NO_WINDOW = 0x08000000 if os.name == "nt" else 0       # CREATE_NO_WINDOW: no console pops up


def log(msg: str) -> None:
    DATA_HOME.mkdir(parents=True, exist_ok=True)
    with open(DATA_HOME / "desktop.log", "a", encoding="utf-8") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {msg}\n")


def error_box(msg: str) -> None:
    """Show an error even when started without a console (pythonw)."""
    log("ERROR " + msg)
    if os.name == "nt":
        import ctypes
        ctypes.windll.user32.MessageBoxW(0, msg, TITLE, 0x10)
    else:
        print(msg, file=sys.stderr)


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def start_server(port: int) -> subprocess.Popen:
    """Run the Streamlit server with the interpreter of this environment (python.exe)."""
    python = Path(sys.executable)
    if python.name.lower() == "pythonw.exe":               # the server itself may use a console-less python
        python = python.with_name("python.exe")
    DATA_HOME.mkdir(parents=True, exist_ok=True)
    out = open(DATA_HOME / "server.log", "w", encoding="utf-8")
    cmd = [str(python), "-m", "streamlit", "run", str(ROOT / "app.py"),
           "--server.port", str(port), "--server.address", "localhost",
           "--server.headless", "true", "--browser.gatherUsageStats", "false"]
    return subprocess.Popen(cmd, cwd=str(ROOT), stdout=out, stderr=subprocess.STDOUT,
                            creationflags=NO_WINDOW)


def wait_until_ready(url: str, server: subprocess.Popen, timeout: float = 120) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout:
        if server.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(url + "/_stcore/health", timeout=2) as r:
                if r.status == 200:
                    return True
        except OSError:
            time.sleep(0.5)
    return False


def open_webview(url: str) -> bool:
    """Native window via pywebview. Returns False if pywebview is unavailable."""
    try:
        import webview
    except ImportError:
        return False
    try:
        # pywebview ignores file downloads unless this is switched on; with it, the
        # download buttons save to the Downloads folder (or ask where to save).
        if hasattr(webview, "settings"):
            webview.settings["ALLOW_DOWNLOADS"] = True
        webview.create_window(TITLE, url, width=1480, height=940, min_size=(1000, 700))
        webview.start(icon=str(ICON)) if ICON.exists() else webview.start()
        return True
    except Exception as exc:                                # e.g. missing WebView2 runtime
        log(f"pywebview failed: {exc!r}")
        return False


def open_app_window(url: str) -> bool:
    """Edge/Chrome in app mode with a private profile, so we can wait for the window to close."""
    candidates = [shutil.which("msedge"), shutil.which("chrome"), shutil.which("google-chrome"),
                  shutil.which("chromium")]
    if os.name == "nt":
        for base in (os.environ.get("PROGRAMFILES(X86)", ""), os.environ.get("PROGRAMFILES", ""),
                     os.environ.get("LOCALAPPDATA", "")):
            candidates += [os.path.join(base, "Microsoft", "Edge", "Application", "msedge.exe"),
                           os.path.join(base, "Google", "Chrome", "Application", "chrome.exe")]
    exe = next((c for c in candidates if c and os.path.exists(c)), None)
    if exe is None:
        return False
    profile = DATA_HOME / "browser-profile"
    proc = subprocess.Popen([exe, f"--app={url}", f"--user-data-dir={profile}",
                             "--window-size=1480,940", "--no-first-run"])
    proc.wait()
    return True


def main() -> None:
    port = free_port()
    url = f"http://localhost:{port}"
    log(f"starting server on {url}")
    server = start_server(port)
    try:
        if not wait_until_ready(url, server):
            error_box("The analysis server did not start.\n\nDetails: "
                      f"{DATA_HOME / 'server.log'}\n\nWas install_windows.bat run successfully?")
            return
        if not (open_webview(url) or open_app_window(url)):
            import webbrowser
            webbrowser.open(url)
            if sys.stdin and sys.stdin.isatty():
                input(f"{TITLE} is running at {url}. Press Enter to quit.\n")
            else:
                server.wait()
    finally:
        server.terminate()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()
        log("stopped")


if __name__ == "__main__":
    main()
