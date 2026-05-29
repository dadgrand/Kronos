#!/usr/bin/env python3
"""Kronos Web UI startup script."""

import os
import subprocess
import sys
import time
import webbrowser


def check_dependencies():
    """Check whether Web UI dependencies are installed."""
    try:
        import flask  # noqa: F401
        import flask_cors  # noqa: F401
        import numpy  # noqa: F401
        import pandas  # noqa: F401
        import plotly  # noqa: F401

        print("All Web UI dependencies are installed")
        return True
    except ImportError as exc:
        print(f"Missing dependency: {exc}")
        print("Please run: pip install -r requirements.txt")
        return False


def install_dependencies():
    """Install Web UI dependencies."""
    print("Installing dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("Dependencies installation completed")
        return True
    except subprocess.CalledProcessError:
        print("Dependencies installation failed")
        return False


def main():
    """Start the Web UI with production-safe defaults."""
    print("Starting Kronos Web UI...")
    print("=" * 50)

    if not check_dependencies():
        print("\nAuto-install dependencies? (y/n): ", end="")
        if input().lower() == "y":
            if not install_dependencies():
                return
        else:
            print("Please manually install dependencies and retry")
            return

    try:
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from model import Kronos, KronosPredictor, KronosTokenizer  # noqa: F401

        print("Kronos model library available")
    except ImportError:
        print("Kronos model library is not available; model loading endpoint will report unavailable")

    print("\nStarting Web server...")
    os.environ["FLASK_APP"] = "app.py"
    os.environ.setdefault("FLASK_ENV", "production")

    try:
        from app import app

        host = os.environ.get("KRONOS_WEB_HOST", "127.0.0.1")
        port = int(os.environ.get("KRONOS_WEB_PORT", "7070"))
        debug = os.environ.get("FLASK_DEBUG", "0") == "1"
        url_host = "localhost" if host in ("127.0.0.1", "0.0.0.0") else host
        url = f"http://{url_host}:{port}"

        print("Web server configured successfully")
        print(f"Access URL: {url}")
        print("Tip: Press Ctrl+C to stop server")

        time.sleep(2)
        webbrowser.open(url)
        app.run(debug=debug, host=host, port=port)
    except Exception as exc:
        print(f"Startup failed: {exc}")
        print("Please check if the configured port is occupied")


if __name__ == "__main__":
    main()
