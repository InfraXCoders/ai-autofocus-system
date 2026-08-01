"""A compact constant-velocity Kalman filter for bounding-box tracking.

State vector (6):   [cx, cy, w, h, vx, vy]
Measurement (4):    [cx, cy, w, h]

The filter predicts where a subject will be on the *next* frame (motion
prediction) and smooths noisy detections. When a detection is briefly missing,
`predict()` alone keeps the estimate moving so focus doesn't jump.

Implemented with plain NumPy so there is no extra dependency (no filterpy).
"""

import numpy as np


class KalmanBoxTracker:
    def __init__(self, cxcywh):
        cx, cy, w, h = cxcywh
        # State: position + size + velocity of the center.
        self.x = np.array([cx, cy, w, h, 0.0, 0.0], dtype=float)

        # State transition: cx += vx, cy += vy (constant velocity).
        self.F = np.eye(6)
        self.F[0, 4] = 1.0
        self.F[1, 5] = 1.0

        # Measurement matrix: we observe [cx, cy, w, h].
        self.H = np.zeros((4, 6))
        self.H[0, 0] = self.H[1, 1] = self.H[2, 2] = self.H[3, 3] = 1.0

        # Covariance: start uncertain about velocity.
        self.P = np.eye(6) * 10.0
        self.P[4, 4] = self.P[5, 5] = 1000.0

        # Process noise: allow the model some drift, more on velocity.
        self.Q = np.diag([1.0, 1.0, 1.0, 1.0, 10.0, 10.0])

        # Measurement noise: centers are precise, sizes noisier.
        self.R = np.diag([1.0, 1.0, 10.0, 10.0])

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        # Keep width/height positive.
        self.x[2] = max(1.0, self.x[2])
        self.x[3] = max(1.0, self.x[3])
        return self.cxcywh

    def update(self, cxcywh):
        z = np.asarray(cxcywh, dtype=float)
        y = z - self.H @ self.x                      # innovation
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)     # Kalman gain
        self.x = self.x + K @ y
        self.P = (np.eye(6) - K @ self.H) @ self.P
        return self.cxcywh

    @property
    def cxcywh(self):
        return self.x[0], self.x[1], self.x[2], self.x[3]

    @property
    def velocity(self):
        return self.x[4], self.x[5]
