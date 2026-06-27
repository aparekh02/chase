"""Shared navigation observation model for chase-mission's closed-loop VLA driver.

This module is the ONE place that turns live coordinates into (a) the
natural-language observation prompt the VLA is fed, and (b) — for TRAINING ONLY
— the geometric expert action that prompt should map to. At run time the expert
is never called: the SmolVLA->LoRA head produces the action from the prompt.

So the split is:
  * `build_prompt(...)`   — used at BOTH train and run time (same obs → same text)
  * `expert_action(...)`  — used at TRAIN time only, to label the imitation data

Geometry (calibrated empirically against the go1 in cadenza-lab):
  * cadenza forward is -X; robot yaw 0 means facing -X.
  * turn_left increases yaw (CCW); turn_right decreases it.
  * world heading angle = yaw + 180deg, so a target's bearing in the robot frame
    is wrap(atan2(dy,dx) - yaw - 180deg). bearing > 0 is to the LEFT (turn_left
    closes it); bearing < 0 is to the RIGHT (turn_right closes it).
"""

from __future__ import annotations

import math

# Action params kept deliberately small so the closed loop steers in fine
# increments (large steps overshoot into rubble and bang the gait around).
STEP_M = 0.5
TURN_RAD = 0.5    # ~28 deg
STRAFE_M = 0.4

# Obstacle gate: a rubble box counts as "in the way" only if it is within this
# range AND inside the forward hemisphere (so debris behind us is ignored).
OBSTACLE_RANGE_M = 1.15
OBSTACLE_FRONT_DEG = 75.0

# Observation buckets ---------------------------------------------------------
TARGET_BUCKETS = ("ahead", "left", "right", "behind")
# Obstacles are bucketed purely by which SIDE they are on (sign of bearing), so
# the dodge can always strafe away from the correct side — even for a near-dead-
# ahead box, where the sign still says which way to peel.
OBSTACLE_BUCKETS = ("none", "left", "right")

TARGET_PHRASE = {
    "ahead":  "the survivor sphere is straight ahead of me",
    "left":   "the survivor sphere is to my left",
    "right":  "the survivor sphere is to my right",
    "behind": "the survivor sphere is behind me",
}
OBSTACLE_PHRASE = {
    "none":  "the path is clear",
    "left":  "a rubble pile is close on my left",
    "right": "a rubble pile is close on my right",
}


def wrap_deg(a: float) -> float:
    """Wrap an angle to (-180, 180]."""
    return (a + 180.0) % 360.0 - 180.0


def bearing_deg(robot_xy, yaw_rad: float, target_xy) -> float:
    """Bearing of target in the robot frame, degrees. >0 left, <0 right, 0 ahead."""
    dx = target_xy[0] - robot_xy[0]
    dy = target_xy[1] - robot_xy[1]
    world = math.degrees(math.atan2(dy, dx))
    return wrap_deg(world - math.degrees(yaw_rad) - 180.0)


def distance(a, b) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def target_bucket(bearing: float) -> str:
    b = abs(bearing)
    if b <= 22.0:
        return "ahead"
    if b >= 110.0:
        return "behind"
    return "left" if bearing > 0 else "right"


def nearest_obstacle(robot_xy, yaw_rad: float, rubble_xys):
    """Return (distance, bearing) of the closest in-front rubble, or None."""
    best = None
    for r in rubble_xys:
        d = distance(robot_xy, r)
        if d > OBSTACLE_RANGE_M:
            continue
        b = bearing_deg(robot_xy, yaw_rad, r)
        if abs(b) > OBSTACLE_FRONT_DEG:
            continue
        if best is None or d < best[0]:
            best = (d, b)
    return best


def obstacle_bucket(obs) -> str:
    if obs is None:
        return "none"
    _, b = obs
    return "left" if b > 0 else "right"


def build_prompt(tgt_bucket: str, obs_bucket: str) -> str:
    """The natural-language observation the VLA is conditioned on (train & run)."""
    return (
        "navigate to the survivor sphere. "
        f"{TARGET_PHRASE[tgt_bucket]}; {OBSTACLE_PHRASE[obs_bucket]}."
    )


def expert_action(tgt_bucket: str, obs_bucket: str):
    """TRAIN-TIME label only: reactive geometric controller → (name, params).

    Avoid first (steer away from / around rubble), then seek the sphere. Signal
    is NOT produced here — it is a non-locomotion control-panel action handled by
    the driver when the sphere is in view, kept symbolic on purpose.
    """
    # Avoid FIRST — strafe sideways away from the obstacle while keeping the
    # heading toward the goal (turning in place doesn't translate past it; that
    # is the reactive-nav livelock). side_step_right moves +Y, side_step_left -Y.
    if obs_bucket == "left":      # obstacle on my left (-Y) → strafe +Y
        return "side_step_right", {"distance_m": STRAFE_M}
    if obs_bucket == "right":     # obstacle on my right (+Y) → strafe -Y
        return "side_step_left", {"distance_m": STRAFE_M}
    # Front clear → seek the sphere.
    if tgt_bucket == "ahead":
        return "walk_forward", {"distance_m": STEP_M}
    if tgt_bucket == "left":
        return "turn_left", {"rotation_rad": TURN_RAD}
    if tgt_bucket == "right":
        return "turn_right", {"rotation_rad": TURN_RAD}
    return "turn_left", {"rotation_rad": TURN_RAD}  # behind → spin to acquire


def all_buckets():
    """Every (target, obstacle) combo — the full imitation training table."""
    for t in TARGET_BUCKETS:
        for o in OBSTACLE_BUCKETS:
            yield t, o


def is_visible(robot_xy, yaw_rad: float, target_xy, fov_deg: float, max_d: float) -> bool:
    """Is the target sphere inside the robot's forward FOV cone right now?"""
    if distance(robot_xy, target_xy) > max_d:
        return False
    return abs(bearing_deg(robot_xy, yaw_rad, target_xy)) <= fov_deg / 2.0
