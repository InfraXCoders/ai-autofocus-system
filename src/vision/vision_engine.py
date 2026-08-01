"""Module 1: AI Vision Engine — the working core of Phase 1.

Uses Google MediaPipe to detect:
  * Faces        (bounding box + eye keypoints)   -> FaceDetection
  * Eyes         (precise landmarks)              -> FaceMesh (optional)
  * Human bodies (person present + pose)          -> Pose (optional)

Output is a list of `Detection` objects that downstream modules consume.
"""

import cv2
import mediapipe as mp

from .types import Detection, Point

# MediaPipe eye landmark indices (from the 468-point FaceMesh) used to draw
# a tight box around each eye.
_LEFT_EYE = [33, 133, 159, 145, 153, 154, 155, 246]
_RIGHT_EYE = [362, 263, 386, 374, 380, 381, 382, 466]


class VisionEngine:
    """Runs subject detection on individual camera frames."""

    def __init__(
        self,
        face_confidence: float = 0.5,
        model_selection: int = 1,
        enable_face_mesh: bool = True,
        enable_pose: bool = True,
    ):
        self.enable_face_mesh = enable_face_mesh
        self.enable_pose = enable_pose

        self._face = mp.solutions.face_detection.FaceDetection(
            model_selection=model_selection,
            min_detection_confidence=face_confidence,
        )

        self._mesh = None
        if enable_face_mesh:
            self._mesh = mp.solutions.face_mesh.FaceMesh(
                max_num_faces=3,
                refine_landmarks=True,
                min_detection_confidence=face_confidence,
                min_tracking_confidence=0.5,
            )

        self._pose = None
        if enable_pose:
            self._pose = mp.solutions.pose.Pose(
                model_complexity=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )

    def process(self, frame_bgr) -> list[Detection]:
        """Detect all subjects in one BGR frame. Returns a list of Detections."""
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False  # perf: MediaPipe can read-only

        detections: list[Detection] = []
        detections += self._detect_faces(rgb, w, h)
        if self._mesh is not None:
            detections += self._detect_eyes(rgb, w, h)
        if self._pose is not None:
            detections += self._detect_body(rgb, w, h)

        return detections

    # ---- Faces (+ eye keypoints) ----
    def _detect_faces(self, rgb, w, h) -> list[Detection]:
        out: list[Detection] = []
        result = self._face.process(rgb)
        if not result.detections:
            return out

        for det in result.detections:
            box = det.location_data.relative_bounding_box
            x = max(0, int(box.xmin * w))
            y = max(0, int(box.ymin * h))
            bw = int(box.width * w)
            bh = int(box.height * h)
            score = det.score[0] if det.score else 0.0

            keypoints: dict[str, Point] = {}
            # MediaPipe FaceDetection exposes 6 keypoints; 0 & 1 are the eyes.
            kp = det.location_data.relative_keypoints
            if len(kp) >= 2:
                for name, idx in (("right_eye", 0), ("left_eye", 1)):
                    p = kp[idx]
                    keypoints[name] = Point(p.x, p.y, int(p.x * w), int(p.y * h))

            out.append(
                Detection(
                    kind="face",
                    confidence=float(score),
                    x=x, y=y, w=bw, h=bh,
                    keypoints=keypoints,
                )
            )
        return out

    # ---- Eyes (precise, via FaceMesh) ----
    def _detect_eyes(self, rgb, w, h) -> list[Detection]:
        out: list[Detection] = []
        result = self._mesh.process(rgb)
        if not result.multi_face_landmarks:
            return out

        for landmarks in result.multi_face_landmarks:
            for name, idxs in (("left_eye", _LEFT_EYE), ("right_eye", _RIGHT_EYE)):
                xs = [landmarks.landmark[i].x for i in idxs]
                ys = [landmarks.landmark[i].y for i in idxs]
                x0, x1 = int(min(xs) * w), int(max(xs) * w)
                y0, y1 = int(min(ys) * h), int(max(ys) * h)
                pad = 4
                out.append(
                    Detection(
                        kind="eye",
                        confidence=1.0,
                        x=max(0, x0 - pad), y=max(0, y0 - pad),
                        w=(x1 - x0) + 2 * pad, h=(y1 - y0) + 2 * pad,
                        keypoints={"which": Point(0, 0)},  # placeholder tag
                    )
                )
        return out

    # ---- Body / person (via Pose) ----
    def _detect_body(self, rgb, w, h) -> list[Detection]:
        out: list[Detection] = []
        result = self._pose.process(rgb)
        if not result.pose_landmarks:
            return out

        xs = [lm.x for lm in result.pose_landmarks.landmark]
        ys = [lm.y for lm in result.pose_landmarks.landmark]
        vis = [lm.visibility for lm in result.pose_landmarks.landmark]
        if max(vis) < 0.5:
            return out

        x0, x1 = int(min(xs) * w), int(max(xs) * w)
        y0, y1 = int(min(ys) * h), int(max(ys) * h)
        out.append(
            Detection(
                kind="body",
                confidence=float(sum(vis) / len(vis)),
                x=max(0, x0), y=max(0, y0),
                w=max(1, x1 - x0), h=max(1, y1 - y0),
            )
        )
        return out

    def close(self):
        self._face.close()
        if self._mesh is not None:
            self._mesh.close()
        if self._pose is not None:
            self._pose.close()
