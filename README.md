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
| 3 | LiDAR Fusion Engine | 🔲 Stubbed |
| 4 | Focus Engine | 🔲 Stubbed |
| 5 | Lens Database | 🔲 Stubbed |
| 6 | Wireless Communication | 🔲 Stubbed |
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

Press **`q`** to quit the preview window.

### Options

```bash
python main.py --source 0            # webcam index (default 0)
python main.py --source video.mp4    # run on a video file
python main.py --no-mesh             # faster: skip detailed eye mesh
```

---

## Repository Structure

```
ai-autofocus-system/
├── main.py                  # Entry point — runs the pipeline on webcam/video
├── config.py                # Central settings
├── requirements.txt
├── src/
│   ├── pipeline.py          # Wires all modules together
│   ├── vision/              # Module 1: AI Vision Engine  ✅
│   ├── tracking/            # Module 2: Tracking Engine    (stub)
│   ├── lidar_fusion/        # Module 3: LiDAR Fusion       (stub)
│   ├── focus/               # Module 4: Focus Engine       (stub)
│   ├── lens_db/             # Module 5: Lens Database      (stub)
│   └── comms/               # Module 6: Wireless Comm      (stub)
└── docs/
    └── architecture.md      # Full architecture + roadmap
```

---

## Roadmap (100-day MVP)

- **Phase 0** (Days 1–10): Foundation — repo, webcam capture ✅
- **Phase 1** (Days 10–35): AI Vision Engine — face/eye/body detection ✅
- **Phase 2** (Days 35–55): Tracking Engine — Kalman motion prediction + re-ID ✅
- **Phase 3** (Days 55–75): Distance + Focus logic (AI depth first, LiDAR later)
- **Phase 4** (Days 75–90): First hardware — ESP32 + motor over wireless
- **Phase 5** (Days 90–100): Flutter mobile app — preview + tap-to-track

See [docs/architecture.md](docs/architecture.md) for detail.

---

## Tech Stack

Python · OpenCV · MediaPipe · ONNX Runtime · Flutter (app) · ESP32 / STM32 (hardware)
