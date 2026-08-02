"""WebSocket server exposing the autofocus pipeline to a browser.

This lets a REMOTE tester use their OWN webcam: their browser captures
frames via getUserMedia, sends each one over the WebSocket as a JPEG, the
server runs it through the exact same `AutofocusPipeline` used by the
desktop demo, draws the same overlay, and streams the annotated frame back.

Each connection gets its own `AutofocusPipeline` instance so multiple
simultaneous testers don't share tracker/lens state with each other.

Run it:
    python web_main.py
    ngrok http 8000        # then share the printed https URL

Privacy note: frames are processed in memory for the life of the connection
and are never written to disk or logged.
"""

import asyncio
import base64
import json
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

import config
from src.overlay import draw_hud, draw_overlay, handle_key
from src.pipeline import AutofocusPipeline

app = FastAPI()

_INDEX_HTML = (Path(__file__).parent / "static" / "index.html").read_text()


@app.get("/")
def index():
    return HTMLResponse(_INDEX_HTML)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    pipeline = AutofocusPipeline()
    loop = asyncio.get_event_loop()
    last_distance_m = None

    try:
        while True:
            raw = await ws.receive_text()
            msg = json.loads(raw)
            msg_type = msg.get("type")

            if msg_type == "frame":
                frame = _decode_frame(msg.get("image", ""))
                if frame is None:
                    continue
                frame = cv2.flip(frame, 1)  # mirror, matches the desktop demo

                # pipeline.step() runs MediaPipe inference (CPU-bound); offload
                # it so this connection's I/O doesn't block other testers.
                state = await loop.run_in_executor(None, pipeline.step, frame)
                last_distance_m = state["distance_m"]

                draw_overlay(frame, state)
                draw_hud(frame, state, fps=msg.get("fps", 0), draw_fps=True)

                await ws.send_text(json.dumps({
                    "type": "frame",
                    "image": _encode_frame(frame),
                    "state": _serialize_state(state),
                }))

            elif msg_type == "select":
                pipeline.select_at(int(msg["x"]), int(msg["y"]))

            elif msg_type == "key":
                handle_key(pipeline, msg.get("key", ""), last_distance_m,
                           config.CALIBRATION_STEP)

    except WebSocketDisconnect:
        pass
    finally:
        pipeline.close()


def _decode_frame(data_url: str):
    try:
        _, b64data = data_url.split(",", 1)
        buf = base64.b64decode(b64data)
        arr = np.frombuffer(buf, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except (ValueError, cv2.error):
        return None


def _encode_frame(frame) -> str:
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
    if not ok:
        return ""
    b64 = base64.b64encode(buf).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def _serialize_state(state: dict) -> dict:
    return {
        "lens": state["lens"],
        "lens_points": state["lens_points"],
        "focus_position": state["focus_position"],
        "distance_m": state["distance_m"],
        "lost": state["lost"],
        "calibrating": state["calibrating"],
    }
