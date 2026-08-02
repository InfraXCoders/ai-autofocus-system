# AI Cinema Autofocus System

An intelligent autofocus ecosystem for professional filmmaking. A **Vision Pod** mounts on top of the camera, uses **AI + LiDAR** to decide *what* to focus on and *how far* it is, and sends real-time focus commands to a wireless lens motor — so the operator never has to manually pull focus.

> Think of it as an "AI brain" upgrade to wireless follow-focus systems (Tilta Nucleus, DJI Focus, PDMovie).

---

## System Overview

```
  [ CAMERA + LiDAR ]  ->  [ AI BRAIN ]  ->  [ FOCUS MOTOR ]
      (the eyes)           (the mind)         (the hands)
```

### Data Pipeline

```
Camera frame
    |
1. AI Vision Engine    -> "There's a face/person at pixel (x,y)"
    |
2. Tracking Engine     -> "Keep following THIS specific subject"
    |
3. LiDAR Fusion Engine -> "That subject is 2.4m away" (AI + LiDAR)
    |
4. Focus Engine        -> "For this lens, 2.4m = focus position 37"
    ^
5. Lens Database       -> provides the lens focus curve
    |
6. Wireless Comm       -> sends "37" to the motor
    |
   Motor turns -> lens is in focus

7. Mobile App          -> operator taps a subject, picks lens, watches preview
```

---

## Project Status

| Phase | Module | Status |
|-------|--------|--------|
| **1** | AI Vision Engine (face / eye / body detection) | ✅ **Working** |
| **2** | Tracking Engine (Kalman + re-ID) | ✅ **Working** |
| **3** | LiDAR Fusion Engine (multi-cue distance) | ✅ **Working** (pre-LiDAR) |
| **4** | Focus Engine (+ manual calibration mode) | ✅ **Working** |
| **5** | Lens Database (curves + JSON persistence) | ✅ **Working** |
| 6 | Wireless Communication | 🔲 Stubbed (virtual motor) |
| 7 | Mobile App (Flutter) | 🔲 Not started |

---

## Quick Start

```bash
# 1. Create & activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the live vision demo (uses your webcam)
python main.py
```

### Options

```bash
python main.py --source 0            # webcam index (default 0)
python main.py --source video.mp4    # run on a video file
python main.py --no-mesh             # faster: skip detailed eye mesh
```

### Controls

| Key | Action |
|-----|--------|
| click | tap-to-track — lock focus onto a subject |
| `l` | cycle the active lens |
| `c` | toggle calibration mode (pauses auto-focus) |
| `i` / `k` | while calibrating: rack focus in / out by hand |
| `a` | while calibrating: save a calibration point at the subject's current distance |
| `s` | save the active lens's calibration to disk |
| `q` | quit |

### Automated tests (no webcam needed)

```bash
.venv/bin/python -m tests.test_tracking          # Phase 1-2: detection + tracking
.venv/bin/python -m tests.test_focus_and_lens    # Phase 3: distance fusion + lens DB
```

---

## Remote testing (share with someone else via ngrok)

`main.py` opens a window using *your* camera — there's nothing to share. To let
someone else test with **their own camera** over the internet, use the web
demo instead: their browser captures their webcam, streams frames to your
machine over WebSocket, and gets back the same AI-annotated video in real time.

```bash
# 1. Install ngrok (one-time)
brew install ngrok/ngrok/ngrok
ngrok config add-authtoken <your-token>   # free account at ngrok.com

# 2. Start the web server
python web_main.py                 # serves on http://localhost:8000
# if 8000 is taken: python web_main.py --port 8010

# 3. In another terminal, open a tunnel
ngrok http 8000                    # (match the port from step 2)
```

`ngrok` prints an `https://....ngrok-free.app` URL — send that to your tester.
When they open it and allow camera access, they'll see the same live overlay
(boxes, distance, focus HUD) running on their own face, and can use the same
tap-to-track + calibration controls as the desktop app.

Each browser tab gets an independent tracker/lens session, so multiple people
can open the link at once without interfering with each other. Video is
processed in memory only — never written to disk.

---

## Repository Structure

```
ai-autofocus-system/
├── main.py                  # Desktop entry point — runs the pipeline on YOUR webcam/video
├── web_main.py              # Web entry point — lets a remote tester use THEIR camera
├── config.py                # Central settings
├── requirements.txt
├── src/
│   ├── pipeline.py          # Wires all modules together
│   ├── geometry.py          # Shared IoU helper
│   ├── overlay.py           # Shared HUD/box drawing (desktop + web)
│   ├── vision/              # Module 1: AI Vision Engine        ✅
│   ├── tracking/            # Module 2: Tracking Engine         ✅
│   ├── lidar_fusion/        # Module 3: LiDAR Fusion            ✅ (pre-LiDAR)
│   ├── focus/               # Module 4: Focus Engine            ✅
│   ├── lens_db/             # Module 5: Lens Database           ✅
│   └── comms/               # Module 6: Wireless Comm           (stub)
├── web/                     # FastAPI + browser front-end for remote testing (ngrok)
├── tests/                   # Automated tests (no webcam needed)
├── data/                    # Local lens calibration (gitignored)
└── docs/
    └── architecture.md      # Full architecture + roadmap
```

---

## Roadmap (100-day MVP)

- **Phase 0** (Days 1–10): Foundation — repo, webcam capture ✅
- **Phase 1** (Days 10–35): AI Vision Engine — face/eye/body detection ✅
- **Phase 2** (Days 35–55): Tracking Engine — Kalman motion prediction + re-ID ✅
- **Phase 3** (Days 55–75): Distance + Focus logic — multi-cue fusion + lens calibration ✅
- **Phase 4** (Days 75–90): First hardware — ESP32 + motor over wireless
- **Phase 5** (Days 90–100): Flutter mobile app — preview + tap-to-track

See [docs/architecture.md](docs/architecture.md) for detail.

---

## Tech Stack

Python · OpenCV · MediaPipe · ONNX Runtime · Flutter (app) · ESP32 / STM32 (hardware)
