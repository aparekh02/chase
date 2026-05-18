"""SmolVLA → cadenza WorldModelAdapter.

SmolVLA's lerobot/smolvla_base checkpoint was trained on 6-DoF manipulator
data (state(6) + 3 RGB views → action(6)), NOT on quadruped locomotion.
We feed it cadenza's observation reshaped into that schema, then heuristically
map the predicted 6-D action vector to a named go1 action (walk/turn/stand).

The semantics are nonsense — SmolVLA has never seen a Go1 — but this keeps
the cadenza stack shape intact and the pipeline end-to-end runnable. Swap in
a locomotion-trained VLA for behavior that actually makes sense.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.factory import make_pre_post_processors
from cadenza.stack.adapters.base import (
    WorldModelAdapter,
    AdapterReply,
    ProposedAction,
)


class VLA(WorldModelAdapter):
    name = "smolvla_go1"
    description = "SmolVLA wrapped as a cadenza adapter (heuristic action mapping)"

    THRESHOLD = 0.05  # below this magnitude, command treated as 'stand'

    def __init__(self, model_id: str = "lerobot/smolvla_base", **kwargs):
        super().__init__(checkpoint=model_id, **kwargs)
        self.device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
        self.policy: SmolVLAPolicy | None = None
        self.preprocess = None
        self.postprocess = None

    @classmethod
    def detect(cls, root: Path) -> Path | None:
        return None  # always passed in as an instance — no FS detection

    def _load_impl(self) -> None:
        print(f"[VLA] Loading {self.checkpoint} on {self.device}...")
        self.policy = (
            SmolVLAPolicy.from_pretrained(str(self.checkpoint))
            .to(self.device)
            .eval()
        )
        self.preprocess, self.postprocess = make_pre_post_processors(
            self.policy.config,
            str(self.checkpoint),
            preprocessor_overrides={"device_processor": {"device": str(self.device)}},
        )
        print("[VLA] Ready.")

    def _build_batch(self, observation: dict, goal: str) -> dict:
        cam = observation.get("camera")
        if cam is None:
            raise RuntimeError(
                "VLA adapter requires observation.camera. Make sure the cadenza "
                "stack is rendering (render_camera=True, the default)."
            )

        pil = Image.fromarray(cam).resize((256, 256))
        img = torch.from_numpy(np.asarray(pil)).float().permute(2, 0, 1) / 255.0

        pos = np.asarray(observation.get("pos", [0.0, 0.0, 0.0]), dtype=np.float32)
        rpy = np.asarray(observation.get("rpy", [0.0, 0.0, 0.0]), dtype=np.float32)
        state = torch.from_numpy(np.concatenate([pos, rpy]))

        return {
            "observation.images.camera1": img,
            "observation.images.camera2": img,
            "observation.images.camera3": img,
            "observation.state": state,
            "task": goal,
        }

    def _map_to_named_action(self, raw_action: torch.Tensor, vocabulary) -> ProposedAction:
        v = raw_action.detach().cpu().numpy().reshape(-1)
        fwd = float(v[0]) if v.size > 0 else 0.0
        yaw = float(v[-1]) if v.size > 1 else 0.0
        rationale = f"smolvla v[:3]={v[:3].round(2).tolist()}, v[-1]={yaw:.2f}"

        if abs(fwd) < self.THRESHOLD and abs(yaw) < self.THRESHOLD:
            return self._pick(vocabulary, "stand", {}, rationale)

        if abs(yaw) > abs(fwd):
            magnitude = min(abs(yaw), math.pi / 2)
            if yaw > 0:
                return self._pick(vocabulary, "turn_left", {"rotation_rad": magnitude}, rationale)
            return self._pick(vocabulary, "turn_right", {"rotation_rad": magnitude}, rationale)

        distance = min(abs(fwd) * 2.0, 1.0)
        if fwd > 0:
            return self._pick(vocabulary, "walk_forward", {"distance_m": distance}, rationale)
        return self._pick(vocabulary, "walk_backward", {"distance_m": distance}, rationale)

    @staticmethod
    def _pick(vocabulary, name: str, params: dict, rationale: str) -> ProposedAction:
        if name in vocabulary:
            return ProposedAction(name=name, params=params, rationale=rationale)
        return ProposedAction(name="stand", params={}, rationale=f"vocab missing {name!r}, fell back")

    def propose_actions(self, observation, goal, vocabulary, history=None) -> AdapterReply:
        self.load()
        raw = self._build_batch(observation, goal)
        batch = self.preprocess(raw)
        with torch.inference_mode():
            action = self.policy.select_action(batch)
            action = self.postprocess(action)
        proposed = self._map_to_named_action(action, vocabulary)
        return AdapterReply(actions=[proposed], done=False, note=proposed.rationale)
