from typing import Optional

import numpy as np
import torch
import open_clip
from PIL import Image

from cadenza.stack.modalities.base import Modality, ModalityResult


class VisualSensor(Modality):
    """CLIP ViT-B/32 image embedder plugged into cadenza's modality pipeline.

    Each tick, cadenza hands us an Observation whose .camera field is a
    HxWx3 uint8 RGB array rendered from the robot's forward camera. We
    encode it with CLIP and expose a 512-D L2-normalized embedding under
    the "visual_embedding" key for the VLA to consume.
    """

    name = "visual_clip"
    description = "CLIP ViT-B/32 image embedding from the robot's forward camera"

    MODEL_NAME = "ViT-B-32"
    PRETRAINED = "openai"
    EMBED_DIM = 512

    def __init__(self, device: str = "cpu"):
        self.device = torch.device(device)
        self.model = None
        self.preprocess = None

    def setup(self) -> None:
        print(f"[VisualSensor] Loading {self.MODEL_NAME} on {self.device}...")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            self.MODEL_NAME, pretrained=self.PRETRAINED
        )
        self.model.eval().to(self.device)
        print("[VisualSensor] Model ready.")

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    def compute(self, observation) -> ModalityResult:
        frame: Optional[np.ndarray] = getattr(observation, "camera", None)
        if frame is None:
            return ModalityResult(keys={}, summary="no camera frame")

        pil_img = Image.fromarray(frame)
        tensor = self.preprocess(pil_img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            emb = self.model.encode_image(tensor)
            emb = emb / emb.norm(dim=-1, keepdim=True)
        emb_np = emb.squeeze(0).cpu().numpy().astype(np.float32)

        return ModalityResult(
            keys={"visual_embedding": emb_np},
            summary=f"clip emb head={emb_np[:3].round(2).tolist()}",
        )
