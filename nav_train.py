"""Train chase-mission's navigation policy head on frozen SmolVLA.

REPLACES the old hardcoded per-phase plans (lora_smolvla.py's PHASE_PLANS).
The head is trained to map a live OBSERVATION prompt — built from the robot's
coordinates by nav_policy.build_prompt — to the next navigation action. Labels
come from nav_policy.expert_action (a geometric reactive controller) over the
full discrete observation table: imitation learning, so at run time the dog's
every move is the VLA head's output conditioned on its live observation.

Run:
    PYTHONPATH=/Users/akshparekh/Documents/cadenza-cli \
      /Users/akshparekh/Documents/cadenza-projects/rescue-dog/.venv/bin/python \
      chase-mission/nav_train.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from smolvla_encoder import SmolVLAEncoder
import nav_policy as nav
import nav_head as nh


def build_table():
    """Every (target, obstacle) bucket -> (prompt, expert action index)."""
    rows = []
    for tgt, obs in nav.all_buckets():
        prompt = nav.build_prompt(tgt, obs)
        name, _ = nav.expert_action(tgt, obs)
        rows.append((prompt, name, nh.ACTIONS.index(name)))
    return rows


def main() -> None:
    print("chase-mission nav policy — SmolVLA(frozen) + trained policy head\n")
    table = build_table()
    print(f"imitation table: {len(table)} observation->action labels")
    for prompt, name, _ in table:
        print(f"  {prompt[55:]:<40} -> {name}")

    print("\nloading frozen SmolVLA base (lerobot/smolvla_base, CPU)…")
    encoder = SmolVLAEncoder(max_seq_len=16)

    # Pre-embed every observation prompt through frozen SmolVLA once.
    X = torch.stack([nh.embed(encoder, p) for p, _, _ in table])      # [N, 960]
    y = torch.tensor([idx for _, _, idx in table])                    # [N]
    in_dim = X.shape[1]

    head = nh.NavHead(in_dim=in_dim)
    opt = torch.optim.Adam(head.parameters(), lr=1e-3)
    lossf = torch.nn.CrossEntropyLoss()
    print(f"\ntraining policy head on SmolVLA embeddings (in_dim={in_dim})…")
    for epoch in range(400):
        opt.zero_grad()
        loss = lossf(head(X), y)
        loss.backward()
        opt.step()
    with torch.no_grad():
        acc = (head(X).argmax(1) == y).float().mean().item()
    print(f"  final loss {loss.item():.4f}  |  train accuracy {acc*100:.0f}%")

    path = nh.save(head, HERE, in_dim)
    print(f"  head saved -> {path}")

    # Show the learned closed-loop mapping (what the VLA will do per observation).
    print("\nlearned policy (observation -> action picked by SmolVLA+head):")
    for tgt, obs in nav.all_buckets():
        p = nav.build_prompt(tgt, obs)
        name, _, conf = nh.decide(head, encoder, p)
        exp, _ = nav.expert_action(tgt, obs)
        mark = "ok " if name == exp else "XX "
        print(f"  {mark}target={tgt:<6} obstacle={obs:<6} -> {name:<12} (conf {conf:.2f})")


if __name__ == "__main__":
    main()
