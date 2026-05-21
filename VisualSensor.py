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
    description = "CLIP ViT-B/32 zero-shot scene detector (spheres + obstacles)"

    MODEL_NAME = "ViT-B-32"
    PRETRAINED = "openai"
    EMBED_DIM = 512

    # Zero-shot queries. Each query maps to (positive prompts, negative prompts).
    # CLIP softmax across all prompts → probability mass on positives = score.
    QUERIES: dict[str, tuple[list[str], list[str]]] = {
        "sphere_visible": (
            [
                "a brightly colored ball on the ground",
                "a red, yellow, green, or blue sphere",
                "a colorful round ball in the scene",
            ],
            [
                "ground with no balls or spheres",
                "an empty area without any colored sphere",
                "a robot looking at terrain without balls",
            ],
        ),
        "obstacle_close": (
            [
                "a large block or wall directly in front of the camera",
                "an obstacle blocking the path ahead at close range",
                "a steep slope filling the view",
            ],
            [
                "an open clear path ahead",
                "empty space in front of the robot",
                "a wide view of distant terrain with no near obstacle",
            ],
        ),
    }

    def __init__(self, device: str = "cpu"):
        self.device = torch.device(device)
        self.model = None
        self.preprocess = None
        self.tokenizer = None
        # name → (stacked_text_embeddings, n_positive_prompts)
        self._query_text_emb: dict[str, tuple] = {}

    def setup(self) -> None:
        print(f"[VisualSensor] Loading {self.MODEL_NAME} on {self.device}...", flush=True)
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            self.MODEL_NAME, pretrained=self.PRETRAINED
        )
        self.model.eval().to(self.device)
        self.tokenizer = open_clip.get_tokenizer(self.MODEL_NAME)

        # Pre-encode all query prompts once. Stack pos+neg per query so we can
        # softmax over them at inference; probability mass on the positive
        # subset is the detection score.
        with torch.no_grad():
            for name, (pos_prompts, neg_prompts) in self.QUERIES.items():
                all_prompts = pos_prompts + neg_prompts
                tokens = self.tokenizer(all_prompts).to(self.device)
                text_emb = self.model.encode_text(tokens)
                text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
                self._query_text_emb[name] = (text_emb, len(pos_prompts))

        print(
            f"[VisualSensor] Model ready. Zero-shot queries: "
            f"{list(self.QUERIES.keys())}",
            flush=True,
        )

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    def compute(self, observation) -> ModalityResult:
        frame: Optional[np.ndarray] = getattr(observation, "camera", None)
        if frame is None:
            print("[VisualSensor] WARNING: observation.camera is None — no scene image to encode.", flush=True)
            return ModalityResult(keys={}, summary="no camera frame")

        pil_img = Image.fromarray(frame)
        tensor = self.preprocess(pil_img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            img_emb = self.model.encode_image(tensor)
            img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)

            scores: dict[str, float] = {}
            for name, (text_emb, n_pos) in self._query_text_emb.items():
                # CLIP-style: logits = 100 * img·text^T, softmax across prompts,
                # sum probability mass on the positive subset.
                logits = 100.0 * (img_emb @ text_emb.T)
                probs = logits.softmax(dim=-1).squeeze(0)
                scores[name] = float(probs[:n_pos].sum().item())

        emb_np = img_emb.squeeze(0).cpu().numpy().astype(np.float32)

        keys: dict = {"visual_embedding": emb_np}
        keys.update(scores)

        summary = "  ".join(f"{k}={v:.2f}" for k, v in scores.items())
        return ModalityResult(keys=keys, summary=summary)
