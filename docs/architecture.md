# Architecture & Roadmap

## 1. Design principle: one data type, decoupled modules

Every module communicates through the `Detection` object (`src/vision/types.py`).
The Vision Engine *produces* Detections; Tracking *selects* one; LiDAR Fusion
*annotates* it with a distance; the Focus Engine *converts* that to a focus
position. Because the contract between modules is fixed, any module's internals
can be rewritten (better AI model, real LiDAR, real motor) without breaking the
others.

```
frame
  │
  ▼
┌────────────────────┐
│ 1. Vision Engine   │  faces / eyes / bodies  → List[Detection]
└─────────┬──────────┘
          ▼
┌────────────────────┐
│ 2. Tracking Engine │  pick & follow ONE subject → Detection (with track_id)
└─────────┬──────────┘
          ▼
┌────────────────────┐   ┌──────────────────┐
│ 3. LiDAR Fusion    │◄──│  LiDAR sensor     │ (Phase 4 hardware)
│    Engine          │   └──────────────────┘
└─────────┬──────────┘  → distance_m
          ▼
┌────────────────────┐   ┌──────────────────┐
│ 4. Focus Engine    │◄──│ 5. Lens Database  │  focus curve per lens
└─────────┬──────────┘   └──────────────────┘
          ▼  focus position 0..1
┌────────────────────┐   ┌──────────────────┐
│ 6. Wireless Comm   │──▶│  Focus Motor      │ (Phase 4 hardware)
└────────────────────┘   └──────────────────┘

  7. Mobile App (Flutter) talks to the pipeline: preview + tap-to-track + settings
```

## 2. Latency budget (why this must stay fast)

For focus to feel "cinematic," the loop from *frame captured* to *motor moved*
should complete in **under ~100 ms**. Rough per-stage budget:

| Stage | Target |
|-------|--------|
| Vision detection | < 40 ms |
| Tracking + fusion + focus math | < 10 ms |
| Wireless send | < 20 ms |
| Motor movement | mechanical |

This is why heavy modules eventually move from Python to **C++ / ONNX Runtime**
and the link uses **ESP-NOW / Nordic** rather than plain Wi-Fi.

## 3. Safety behavior (non-negotiable)

When the subject is lost, the Focus Engine **HOLDS** the last position — it must
never "hunt" (rack focus in and out searching). A held frame is recoverable; a
hunting lens ruins the shot. See `FocusEngine.update()`.

## 4. 100-day MVP roadmap

| Phase | Days | Deliverable | Stack |
|-------|------|-------------|-------|
| 0 | 1–10 | Repo, webcam capture, folder structure | Python, OpenCV |
| **1** | 10–35 | **AI Vision Engine** (face/eye/body) ✅ | MediaPipe |
| 2 | 35–55 | Tracking: lock, motion prediction, re-ID | Kalman + embeddings |
| 3 | 55–75 | Distance (AI depth) + Focus + Lens DB | ONNX depth model |
| 4 | 75–90 | First hardware: ESP32 + motor over wireless | ESP32, pyserial/BLE |
| 5 | 90–100 | Flutter app: preview + tap-to-track | Flutter |

## 5. What is real vs simulated today

| Module | Today | Upgrade path |
|--------|-------|--------------|
| Vision | ✅ real (MediaPipe) | custom ONNX models for objects |
| Tracking | ⚠️ nearest-neighbor placeholder | Kalman filter + appearance re-ID |
| LiDAR Fusion | ⚠️ face-size distance estimate | real LiDAR + AI depth fusion |
| Focus | ✅ real smoothing logic | tune curves per lens |
| Lens DB | ✅ interpolation, in-memory | JSON save/load, real calibration |
| Wireless | ⚠️ virtual motor (no-op) | ESP-NOW / BLE transport |

## 6. Team & effort (honest estimate)

- **100-day MVP** (software brain + basic motor demo): realistic with 1–2 focused people.
- **Full professional product** (robust, all lenses, polished app, certified hardware): **1.5–2 years**, ideally:
  - 1 AI / Computer-Vision engineer (core)
  - 1 Embedded / firmware engineer (Phase 4+)
  - 1 Mobile developer (Flutter)
  - Part-time cinematographer (Gurpreet) for real-world testing & focus feel
