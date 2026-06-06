"""chase-mission: fine-tune cadenza-lab's LoRA head ON TOP OF frozen SmolVLA.

The real architecture you asked for:

    SmolVLA (lerobot/smolvla_base, FROZEN)
        -> VLM text-tower hidden states [B, T, 960]
        -> cadenza-lab LoRAActionHead  (trainable low-rank A/B only)
        -> go1 ActionCall sequence

SmolVLA is the base; the ONLY thing trained is cadenza-lab's LoRA. lerobot lives
here in chase-mission; the fine-tuning itself is the VLA-agnostic cadenza-api
pipeline, handed our SmolVLA encoder through its ``encoder=`` hook.

Run (with an interpreter that has lerobot + cadenza-lab, e.g. the cadenza venv,
plus cadenza-api on PYTHONPATH)::

    PYTHONPATH=/path/to/cadenza-api \
      /path/to/cadenza/.venv/bin/python chase-mission/lora_smolvla.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))  # for smolvla_encoder

from cadenza_api import customenv as ce
from cadenza_api import lora_data as ld
from cadenza_api import lora_finetune as lf
from smolvla_encoder import SmolVLAEncoder

# Per-phase go1 library-action plan — the project-specific training signal.
PHASE_PLANS = {
    "descend_in":    [{"action": "walk_forward", "distance_m": 1.5},
                      {"action": "walk_forward", "distance_m": 1.0}],
    "scan_debris":   [{"action": "turn_left", "rotation_rad": 0.8},
                      {"action": "turn_right", "rotation_rad": 1.6},
                      {"action": "turn_left", "rotation_rad": 0.8}],
    "signal_sphere": [{"action": "stand"}, {"action": "shake_hand"}],
    "ascend_out":    [{"action": "climb_step"},
                      {"action": "walk_forward", "distance_m": 2.0}],
}

G, C, D, B, R = "\033[32m", "\033[36m", "\033[2m", "\033[1m", "\033[0m"


def main() -> None:
    cfg = ce.load_config(HERE / "env.json")
    print(f"{B}chase-mission{R} — robot={cfg.robot}, {len(cfg.phases)} phases")

    print(f"\n{B}1. loading frozen SmolVLA base{R} {D}(lerobot/smolvla_base, CPU)…{R}")
    encoder = SmolVLAEncoder(max_seq_len=16)
    print(f"   SmolVLA text-tower hidden_dim = {encoder.hidden_dim}")

    print(f"\n{B}2. authoring project patterns (goal → go1 actions){R}")
    for phase in cfg.phases:
        plan = PHASE_PLANS.get(phase.name)
        if plan:
            ld.add_pattern(HERE, phase.goal_prompt, plan)
            print(f"   {phase.name:14} → {' → '.join(a['action'] for a in plan)}")

    ds = ld.build_dataset(HERE)
    print(f"\n{B}3. fine-tuning cadenza-lab LoRA on SmolVLA{R} {D}({len(ds)} patterns){R}")
    lcfg = lf.LoRAConfig(robot=cfg.robot, hidden_dim=encoder.hidden_dim,
                         max_seq_len=16, lora_rank=8)
    res = lf.train_project_lora(HERE, lcfg, ds, epochs=300, lr=0.02,
                                encoder=encoder)
    print(f"   loss {res.initial_loss:.3f} → {G}{res.final_loss:.4f}{R} over "
          f"{res.epochs} epochs {D}({res.trainable_params} trainable A/B params; "
          f"SmolVLA frozen){R}")
    print(f"   LoRA head saved → {res.adapter_path.relative_to(HERE.parent)}")

    print(f"\n{B}4. decode mission goals through SmolVLA→LoRA{R}")
    decoder, _ = lf.load_project_lora(HERE, encoder=encoder)
    all_ok = True
    for phase in cfg.phases:
        plan = PHASE_PLANS.get(phase.name)
        if not plan:
            continue
        report = decoder.decode_report(phase.goal_prompt)
        got = [c.action_name for c in report.calls]
        want = [a["action"] for a in plan]
        ok = got == want
        all_ok = all_ok and ok
        print(f"\n   {(G+'✓'+R) if ok else '✗'} {C}{phase.name}{R}: "
              f"{D}{phase.goal_prompt[:58]}…{R}")
        for i, c in enumerate(report.calls, 1):
            params = " ".join(f"{p}={getattr(c, p):.2f}"
                              for p in ("distance_m", "rotation_rad")
                              if abs(getattr(c, p, 0.0)) > 1e-6)
            print(f"        {i}. {c.action_name} {params} "
                  f"{D}(conf {report.action_confidence[i-1]:.2f}){R}")

    print(f"\n{G if all_ok else ''}{B}done — SmolVLA(frozen) + cadenza-lab LoRA: "
          f"{'all goals decoded to their plans' if all_ok else 'see mismatches'}{R}")


if __name__ == "__main__":
    main()
