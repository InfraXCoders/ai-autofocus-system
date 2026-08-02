"""Web demo entry point — lets a REMOTE tester use THEIR OWN camera.

Unlike main.py (which opens a desktop window using YOUR webcam), this starts
a small web server. A browser — yours or a remote tester's — opens the page,
grants camera access, and streams frames to the AI pipeline over WebSocket;
annotated frames stream back in real time.

Local use:
    python web_main.py
    open http://localhost:8000

Share with a remote tester via ngrok:
    python web_main.py
    ngrok http 8000                 # in another terminal
    # share the printed https://....ngrok-free.app URL

Use a different port if 8000 is already taken on your machine:
    python web_main.py --port 8010
    ngrok http 8010

Each browser tab gets its own tracker/lens state, so multiple testers can
use the same URL without interfering with each other.
"""

import argparse

import uvicorn

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="AI Autofocus — web demo server")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--host", default="0.0.0.0")
    args = p.parse_args()
    uvicorn.run("web.server:app", host=args.host, port=args.port)
