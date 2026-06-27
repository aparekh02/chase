"""Closed-loop VLA navigation for chase-mission — the proof.

Every tick:
  1. read the robot's LIVE coordinates (x, y, yaw) from the running CustomEnv,
  2. work out, from those coordinates, where the current target sphere is and
     whether rubble is in the way (nav_policy),
  3. build the natural-language observation prompt and let SmolVLA -> the trained
     policy head pick the next action (nav_head.decide) — this is the ONLY thing
     deciding the move; nothing is scripted,
  4. step the env with that action and print the coordinate trace.

When the target sphere comes within reach AND into view, the dog fires the
control-panel signal (a symbolic, non-locomotion action) and locks onto the next
sphere. Run headless; the printed trace is the evidence that the VLA — not a
hardcoded plan — steers around the rubble and finds all three spheres.

Run:
    PYTHONPATH=/Users/akshparekh/Documents/cadenza-cli \
      /Users/akshparekh/Documents/cadenza-projects/rescue-dog/.venv/bin/python \
      chase-mission/vla_navigate.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from cadenza_cli import customenv as ce
from smolvla_encoder import SmolVLAEncoder
import nav_policy as nav
import nav_head as nh

SIGNAL_RANGE_M = 1.5   # must get THIS close (around obstacles) before it counts as found
MAX_TICKS = 200


def main() -> None:
    cfg = ce.load_config(HERE / "nav_env.json")
    fov, max_d = cfg.vision.fov_deg, cfg.vision.max_distance_m
    rubble = [(o.position[0], o.position[1]) for o in cfg.objects if o.tag == "rubble"]
    spheres = [(o.position[0], o.position[1]) for o in cfg.objects if o.tag == "sphere"]

    view = "--view" in sys.argv  # open the live MuJoCo viewer (needs mjpython on macOS)

    print("loading frozen SmolVLA + trained nav head…")
    encoder = SmolVLAEncoder(max_seq_len=16)
    head = nh.load(HERE)

    if view:
        print("opening MuJoCo viewer — watch the dog strafe around the rubble…")
    env = ce.CustomEnv(cfg, headless=not view)
    obs, info = env.reset()
    pos = info["pos"]
    yaw = env._current_yaw()

    # Visit spheres nearest-first; lock onto one until it is found.
    remaining = sorted(range(len(spheres)),
                       key=lambda i: nav.distance(pos, spheres[i]))
    found: list[int] = []

    print(f"\nclosed-loop VLA navigation — {len(spheres)} spheres, "
          f"{len(rubble)} rubble piles, FOV {fov:.0f}°/{max_d:.0f}m\n")
    print(f"{'tick':>4} {'x':>7} {'y':>7} {'yaw°':>6}  {'target':>6} "
          f"{'brng°':>6} {'dist':>5}  {'obstacle':>9}  {'decided by VLA':>16}")

    tick = 0
    flipped = False
    while remaining and tick < MAX_TICKS:
        tgt_i = remaining[0]
        tgt = spheres[tgt_i]
        brng = nav.bearing_deg(pos, yaw, tgt)
        dist = nav.distance(pos, tgt)
        visible = nav.is_visible(pos, yaw, tgt, fov, max_d)

        if visible and dist <= SIGNAL_RANGE_M:
            action, who = "signal_sphere", "SIGNAL (sphere in view)"
        else:
            ob = nav.nearest_obstacle(pos, yaw, rubble)
            tb, obk = nav.target_bucket(brng), nav.obstacle_bucket(ob)
            prompt = nav.build_prompt(tb, obk)
            name, params, conf = nh.decide(head, encoder, prompt)
            action = name if not params else _call(name, params)
            who = f"VLA:{name}({conf:.2f})"

        obs, reward, term, trunc, info = env.step(action)
        tick += 1
        pos = info["pos"]
        yaw = env._current_yaw()

        ob = nav.nearest_obstacle(pos, yaw, rubble)
        obk = nav.obstacle_bucket(ob)
        obtxt = f"{obk}" + (f"@{ob[0]:.1f}m" if ob else "")
        print(f"{tick:>4} {pos[0]:>7.2f} {pos[1]:>7.2f} {math.degrees(yaw):>6.0f}  "
              f"s{tgt_i+1:>5} {brng:>6.0f} {dist:>5.2f}  {obtxt:>9}  {who:>16}")

        if action == "signal_sphere" or (action and getattr(action, "action_name", "") == "signal_sphere"):
            found.append(tgt_i)
            remaining.pop(0)
            remaining.sort(key=lambda i: nav.distance(pos, spheres[i]))
            print(f"     >>> FOUND sphere s{tgt_i+1} at {spheres[tgt_i]} — "
                  f"{len(found)}/{len(spheres)} found <<<")

        if term and not env.mission_success:
            flipped = True
            print("     !!! robot flipped — terminating !!!")
            break

    env.close()
    sig = env.signal_summary
    print(f"\nresult: {len(found)}/{len(spheres)} spheres found in {tick} ticks"
          + ("  (FLIPPED)" if flipped else ""))
    print(f"control panel: {sig['correct']} correct / {sig['false_positive']} "
          f"false-positive / {sig['missed']} missed")
    print("order found:", [f"s{i+1}" for i in found])


def _call(name: str, params: dict):
    """Wrap a name+params into a cadenza ActionCall the env can execute."""
    try:
        from cadenza.actions.library import ActionCall
        return ActionCall(action_name=name, **params)
    except Exception:
        return name


if __name__ == "__main__":
    main()
