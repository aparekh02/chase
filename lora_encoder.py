"""Project encoder hook for chase-mission.

cadenza-api's LoRA pipeline (``env lora finetune`` / ``env run --policy lora``)
looks for a ``lora_encoder.py`` at the project root exposing
``build_encoder(max_seq_len) -> nn.Module``. That's how chase-mission plugs its
OWN base/VLA — SmolVLA — into cadenza-lab's LoRA head, WITHOUT cadenza-api ever
importing lerobot. lerobot lives here, in chase-mission, alone.
"""

from __future__ import annotations


def build_encoder(max_seq_len: int = 16):
    """Return chase-mission's frozen SmolVLA encoder for the LoRA head."""
    from smolvla_encoder import SmolVLAEncoder
    return SmolVLAEncoder(max_seq_len=max_seq_len)
