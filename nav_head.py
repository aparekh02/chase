"""Tiny navigation policy head on top of frozen SmolVLA — chase-mission.

cadenza-lab's LoRAActionDecoder is a *positional* goal->action-sequence decoder:
action 0 is read from goal-token slot 0, which (we verified) is a constant BOS
embedding — so it cannot map a whole observation sentence to a single action.
For closed-loop, per-tick observation->action control we need a policy head that
reads a POOLED representation of the observation, which is exactly the standard
VLA action-head design.

So: SmolVLA (frozen) encodes the live observation prompt; this small MLP head —
the only thing trained — maps SmolVLA's mean-pooled hidden state to the next
navigation action. That makes the dog's every move the VLA head's output
conditioned on its live coordinates, not a replayed script.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn

# The navigation action set the head chooses from. Params are fixed per action
# (small increments for fine closed-loop steering); the head only picks WHICH.
ACTIONS = ["walk_forward", "turn_left", "turn_right", "side_step_left", "side_step_right"]
ACTION_PARAMS = {
    "walk_forward": {"distance_m": 0.5},
    "turn_left": {"rotation_rad": 0.5},
    "turn_right": {"rotation_rad": 0.5},
    "side_step_left": {"distance_m": 0.4},
    "side_step_right": {"distance_m": 0.4},
}

HEAD_FILENAME = "nav_head.pt"
HEAD_META_FILENAME = "nav_head.json"


class NavHead(nn.Module):
    def __init__(self, in_dim: int = 960, hidden: int = 128, n_actions: int = len(ACTIONS)):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, n_actions)
        )

    def forward(self, x):
        return self.net(x)


@torch.no_grad()
def embed(encoder, prompt: str) -> torch.Tensor:
    """Mean-pooled SmolVLA text-tower hidden state for one observation prompt."""
    h = encoder._encode_text(prompt)        # [T, 960], frozen
    return h.mean(dim=0).float()            # [960]


def save(head: NavHead, project_dir: Path, in_dim: int) -> Path:
    d = Path(project_dir) / "lora"
    d.mkdir(parents=True, exist_ok=True)
    path = d / HEAD_FILENAME
    torch.save(head.state_dict(), path)
    (d / HEAD_META_FILENAME).write_text(json.dumps(
        {"in_dim": in_dim, "actions": ACTIONS}, indent=2))
    return path


def load(project_dir: Path) -> NavHead:
    d = Path(project_dir) / "lora"
    meta = json.loads((d / HEAD_META_FILENAME).read_text())
    head = NavHead(in_dim=int(meta["in_dim"]))
    head.load_state_dict(torch.load(d / HEAD_FILENAME, weights_only=True))
    head.eval()
    return head


@torch.no_grad()
def decide(head: NavHead, encoder, prompt: str):
    """Observation prompt -> (action_name, params, confidence) via SmolVLA+head."""
    logits = head(embed(encoder, prompt).unsqueeze(0))[0]
    probs = torch.softmax(logits, dim=-1)
    i = int(probs.argmax())
    name = ACTIONS[i]
    return name, dict(ACTION_PARAMS[name]), float(probs[i])
