"""
vision_sensor.py — CLIP visual sensor for Unitree Go1 → VLA pipeline

pip install open-clip-torch opencv-python-headless

Model: ViT-B/32 (CLIP, OpenAI pretrained)
  - ~340MB one-time download
  - Runs on CPU at ~8-12Hz on Go1's Intel NUC
  - Outputs: 512-dim L2-normalized embedding per frame
"""

import time
import threading
from typing import Optional

import numpy as np
import torch
import open_clip
from PIL import Image


class VisualSensor:
    """
    Wraps CLIP ViT-B/32 as a streaming visual sensor.
    Designed to run as a background thread on the Go1,
    producing embeddings that feed directly into a VLA.

    Usage:
        sensor = VisualSensor()
        sensor.start()

        # in your VLA loop:
        embedding = sensor.latest_embedding   # (512,) numpy float32
    """

    MODEL_NAME = "ViT-B-32"
    PRETRAINED  = "openai"
    EMBED_DIM   = 512

    def __init__(
        self,
        camera_index: int = 0,
        target_hz: float = 10.0,
        device: str = "cpu",
    ):
        self.camera_index = camera_index
        self.target_hz    = target_hz
        self.device       = torch.device(device)

        # Load model + preprocessor (downloads weights on first run, ~340MB)
        print(f"[VisualSensor] Loading {self.MODEL_NAME} on {device}...")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            self.MODEL_NAME, pretrained=self.PRETRAINED
        )
        self.model.eval().to(self.device)
        print("[VisualSensor] Model ready.")

        # Shared state (thread-safe via lock)
        self._lock            = threading.Lock()
        self._latest_embedding: Optional[np.ndarray] = None
        self._latest_timestamp: float = 0.0
        self._running         = False
        self._thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start background capture + encoding thread."""
        self._running = True
        self._thread  = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        print(f"[VisualSensor] Streaming at {self.target_hz}Hz from camera {self.camera_index}")

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)

    @property
    def latest_embedding(self) -> Optional[np.ndarray]:
        """Latest (512,) L2-normalized float32 embedding. None until first frame."""
        with self._lock:
            return self._latest_embedding.copy() if self._latest_embedding is not None else None

    @property
    def latest_timestamp(self) -> float:
        """Unix timestamp of the latest embedding."""
        with self._lock:
            return self._latest_timestamp

    def encode_frame(self, frame_bgr: np.ndarray) -> np.ndarray:
        """
        Encode a single BGR frame (e.g. from cv2.VideoCapture) → (512,) embedding.
        Useful for one-shot encoding without the background thread.
        """
        pil_img = Image.fromarray(frame_bgr[..., ::-1])  # BGR → RGB
        tensor  = self.preprocess(pil_img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            emb = self.model.encode_image(tensor)
            emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb.squeeze(0).cpu().numpy().astype(np.float32)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _loop(self) -> None:
        try:
            import cv2
        except ImportError:
            raise ImportError("pip install opencv-python-headless")

        cap      = cv2.VideoCapture(self.camera_index)
        interval = 1.0 / self.target_hz

        if not cap.isOpened():
            print(f"[VisualSensor] ERROR: cannot open camera {self.camera_index}")
            return

        while self._running:
            t0 = time.monotonic()

            ret, frame = cap.read()
            if not ret:
                continue

            emb = self.encode_frame(frame)

            with self._lock:
                self._latest_embedding  = emb
                self._latest_timestamp  = time.time()

            elapsed = time.monotonic() - t0
            sleep   = max(0.0, interval - elapsed)
            time.sleep(sleep)

        cap.release()


# ------------------------------------------------------------------
# VLA integration shim
# ------------------------------------------------------------------

class VLASensorBridge:
    """
    Thin adapter: pulls the latest visual embedding and packages it
    with the proprioceptive state vector for VLA consumption.

    Expected VLA input shape: [visual_embedding | proprio_state]
    = [512 + proprio_dim] float32 vector
    """

    def __init__(self, sensor: VisualSensor, proprio_dim: int = 12):
        self.sensor     = sensor
        self.proprio_dim = proprio_dim

    def get_observation(self, proprio_state: np.ndarray) -> Optional[np.ndarray]:
        """
        Returns concatenated [visual (512) | proprio (N)] observation vector.
        Returns None if no frame has been received yet.
        """
        emb = self.sensor.latest_embedding
        if emb is None:
            return None

        assert proprio_state.shape == (self.proprio_dim,), \
            f"Expected proprio_state shape ({self.proprio_dim},), got {proprio_state.shape}"

        return np.concatenate([emb, proprio_state.astype(np.float32)])


# ------------------------------------------------------------------
# Smoke test (run on Go1 directly)
# ------------------------------------------------------------------

if __name__ == "__main__":
    sensor = VisualSensor(camera_index=0, target_hz=10.0)
    sensor.start()

    bridge = VLASensorBridge(sensor, proprio_dim=12)

    print("Reading embeddings for 5 seconds...")
    t_end = time.time() + 5.0
    while time.time() < t_end:
        dummy_proprio = np.zeros(12, dtype=np.float32)  # replace with real joint states
        obs = bridge.get_observation(dummy_proprio)
        if obs is not None:
            print(f"  obs shape={obs.shape}  |  visual norm={np.linalg.norm(obs[:512]):.4f}")
        time.sleep(0.1)

    sensor.stop()
    print("Done.")