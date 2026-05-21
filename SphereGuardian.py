"""SphereGuardian — VLAGuardian + CLIP sphere spotting (rescue-dog behavior).

Subclasses cadenza's built-in VLAGuardian so we keep its raycast-based
obstacle detection and avoidance planning. On top of that, every poll we
also run a CLIP zero-shot check for a brightly colored ball in the camera
frame. When one is spotted (and the dog hasn't just saluted), the guardian
reports an "interruption" with position="sphere"; Sequential then pauses
the current step, runs the sit → stand_up salute, and resumes the original
traversal step from where it left off.
"""

from __future__ import annotations

import numpy as np
import open_clip
import torch
from PIL import Image

from cadenza.go1 import Step
from cadenza.vla.guardian import ObstacleResult, VLAGuardian


class SphereGuardian(VLAGuardian):
    """Obstacle avoidance + brightly-colored-sphere salute."""

    CLIP_MODEL_NAME = "ViT-B-32"
    CLIP_PRETRAINED = "openai"

    # Score thresholds. CLIP softmax across pos+neg prompts; we sum
    # probability mass on the positive subset.
    SPHERE_TRIGGER = 0.65   # above this → salute (if armed)
    SPHERE_REARM = 0.40     # below this → re-arm so the next sphere counts

    # Real-obstacle avoidance tuning — tight caps so we don't drift.
    BACK_OFF_DIST = 0.25           # walk_backward when too close
    BACK_OFF_TRIGGER = 0.45        # distance below which we back off first
    LATERAL_MIN = 0.35             # min side-step
    LATERAL_MAX = 0.55             # HARD cap: never drift more than this per avoidance
    LATERAL_MARGIN = 0.20          # extra clearance beyond measured extent
    FORWARD_MIN = 0.45             # min walk-past
    FORWARD_MAX = 0.85             # cap so we don't sprint past
    FORWARD_MARGIN = 0.15          # extra clearance beyond measured depth

    # Tight-corridor escape (both sides blocked).
    TRAPPED_CLEAR_THRESHOLD = 0.20   # clearance below this on both sides = trapped
    TRAPPED_BACKOFF = 0.50           # how far to retreat when trapped
    TRAPPED_TURN_RAD = 0.45          # small reorienting turn (~26°)

    # Low-obstacle "crawl over" detection.
    LOW_OBSTACLE_HEIGHT = 0.10       # below this, crawl_forward instead of around
    CRAWL_OVER_DIST_MAX = 0.80       # cap on how far we'll commit to crawling

    # Periodic crouch peek — crawl_forward (not sit) lowers body while
    # moving so we can look under/around debris for low spheres.
    CROUCH_PEEK_EVERY = 3
    CROUCH_PEEK_DIST = 0.25
    CROUCH_PEEK_SPEED = 0.5

    POS_PROMPTS = [
        "a brightly colored ball on the ground",
        "a red, yellow, green, or blue sphere",
        "a colorful round ball in front of the robot",
    ]
    NEG_PROMPTS = [
        "ground with no balls or spheres",
        "an empty area without any colored sphere",
        "terrain without any ball visible",
    ]

    def __init__(self, robot: str = "go1", **kwargs):
        # Forward only the kwargs that were actually supplied so VLAGuardian's
        # own defaults (e.g. model_id="HuggingFaceTB/SmolVLM-256M-Instruct")
        # apply when we don't override them.
        super().__init__(robot=robot, **kwargs)
        self._clip_model = None
        self._clip_preprocess = None
        self._sphere_text_emb: torch.Tensor | None = None
        self._n_pos = len(self.POS_PROMPTS)
        # Latch: only salute once per sphere encounter; re-arm when we no
        # longer see one (score drops below SPHERE_REARM).
        self._armed = True
        # Track real-obstacle avoidance count for periodic crouch-peeking.
        self._avoidance_count = 0
        # Anti-drift: remember last side we escaped to and force a flip if
        # the next obstacle would push us further the same way.
        self._last_side: str | None = None  # "left" | "right" | None

    # ── Lifecycle ───────────────────────────────────────────────────────────

    def load(self):
        super().load()  # base VLAGuardian (SmolVLM) for obstacle planning
        print("  SphereGuardian: loading CLIP ViT-B-32 for sphere spotting...")
        self._clip_model, _, self._clip_preprocess = open_clip.create_model_and_transforms(
            self.CLIP_MODEL_NAME, pretrained=self.CLIP_PRETRAINED
        )
        self._clip_model.eval()
        tokenizer = open_clip.get_tokenizer(self.CLIP_MODEL_NAME)
        with torch.no_grad():
            tokens = tokenizer(self.POS_PROMPTS + self.NEG_PROMPTS)
            text_emb = self._clip_model.encode_text(tokens)
            text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
            self._sphere_text_emb = text_emb
        print("  SphereGuardian: CLIP ready")

    # ── Sphere spotting ─────────────────────────────────────────────────────

    def _sphere_score(self, mj_model, mj_data) -> float:
        """Render the robot camera and run CLIP zero-shot. Returns p(sphere)."""
        frame = self._render_camera(mj_model, mj_data)
        if frame is None:
            return 0.0
        pil = Image.fromarray(frame)
        tensor = self._clip_preprocess(pil).unsqueeze(0)
        with torch.no_grad():
            img_emb = self._clip_model.encode_image(tensor)
            img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
            logits = 100.0 * (img_emb @ self._sphere_text_emb.T)
            probs = logits.softmax(dim=-1).squeeze(0)
            return float(probs[: self._n_pos].sum().item())

    # ── Guardian interface ──────────────────────────────────────────────────

    def check_raycast_only(self, mj_model, mj_data, verbose: bool = True):
        """Called every ~0.3s during gait. Returns (detected, dist, position)."""
        # 1. Real obstacle check (raycast) — VLAGuardian's job.
        detected, dist, position = super().check_raycast_only(
            mj_model, mj_data, verbose=verbose,
        )
        if detected:
            # Obstacle override always wins. Re-arm sphere latch — after the
            # avoidance maneuver, the robot may face a sphere again.
            self._armed = True
            return True, dist, position

        # 2. No obstacle; check for a sphere.
        score = self._sphere_score(mj_model, mj_data)
        if score >= self.SPHERE_TRIGGER and self._armed:
            if verbose:
                print(f"       >>> SPHERE SPOTTED (clip_score={score:.2f}) — saluting <<<",
                      flush=True)
            self._armed = False  # don't re-trigger until we lose sight of this sphere
            return True, 0.0, "sphere"

        if score < self.SPHERE_REARM:
            # No sphere in view → rearm so the next encounter counts.
            self._armed = True

        return False, float("inf"), "none"

    def plan_avoidance(self, mj_model, mj_data, position: str, dist: float) -> ObstacleResult:
        if position == "sphere":
            # The "avoidance" maneuver for a sphere is just a salute.
            return ObstacleResult(
                detected=True,
                distance=0.0,
                position="sphere",
                size="sphere",
                avoidance_actions=[
                    Step(name="sit"),
                    Step(name="stand_up"),
                ],
                raw_response="rescue dog spotted a sphere — sitting to acknowledge",
            )
        # Real obstacle: let the base class measure (raycast + optional VLM
        # size hint). We discard its U-shape avoidance_actions list and
        # build our own compact one in get_avoidance_steps.
        return super().plan_avoidance(mj_model, mj_data, position, dist)

    def get_avoidance_steps(self, result: ObstacleResult) -> list:
        if result.position == "sphere":
            return [Step("sit"), Step("stand_up")]
        return self._compact_obstacle_steps(result)

    def _compact_obstacle_steps(self, result: ObstacleResult) -> list:
        """Bounded, debris-friendly avoidance.

        Three regimes, chosen from the raycast measurement:

          A. LOW obstacle (height < LOW_OBSTACLE_HEIGHT)
             → crawl_forward over it (body lowers in the same gait).
          B. TRAPPED (clear_left and clear_right both < TRAPPED_CLEAR_THRESHOLD)
             → walk_backward + small turn to find a new heading; the
               scripted step resumes from a more open vantage.
          C. NORMAL → backoff (if very close) + side_step + walk_forward +
               side_step_back. Lateral capped at LATERAL_MAX so drift per
               avoidance is bounded; anti-drift latch flips the chosen
               side when the last avoidance already escaped that way.

        Every CROUCH_PEEK_EVERY-th avoidance appends a brief crawl_forward
        so the dog crouches to scan under/around debris.
        """
        meas = result.measurement
        if meas is None:
            return super().get_avoidance_steps(result)

        # ── Regime A: low obstacle → crawl over ─────────────────────────
        if meas.height < self.LOW_OBSTACLE_HEIGHT:
            crawl_dist = min(
                self.CRAWL_OVER_DIST_MAX,
                max(0.50, meas.distance + meas.depth + 0.20),
            )
            print(f"       [SphereGuardian] LOW obstacle (h={meas.height:.2f}m) "
                  f"→ crawl_forward {crawl_dist:.2f}m", flush=True)
            self._avoidance_count += 1
            self._last_side = None  # crawling doesn't bias sides
            return [Step("crawl_forward", speed=0.5, distance_m=crawl_dist)]

        # ── Regime B: both sides blocked (close quarters) ────────────────
        if (meas.clear_left < self.TRAPPED_CLEAR_THRESHOLD
                and meas.clear_right < self.TRAPPED_CLEAR_THRESHOLD):
            # Pick the slightly-more-open side to turn toward.
            turn_right = meas.clear_right >= meas.clear_left
            turn_name = "turn_right" if turn_right else "turn_left"
            print(f"       [SphereGuardian] TRAPPED (clear_L={meas.clear_left:.2f} "
                  f"clear_R={meas.clear_right:.2f}) → back off + small {turn_name}",
                  flush=True)
            self._avoidance_count += 1
            self._last_side = None  # turning resets the lateral bias
            return [
                Step("walk_backward", speed=1.0, distance_m=self.TRAPPED_BACKOFF),
                Step(turn_name, speed=1.0, rotation_rad=self.TRAPPED_TURN_RAD),
            ]

        # ── Regime C: standard compact side-step ─────────────────────────
        # Initial side selection from raycast bias / free-space measurement.
        if result.position == "left":
            go_right = True
        elif result.position == "right":
            go_right = False
        else:
            go_right = meas.clear_right >= meas.clear_left

        # Anti-drift: if the LAST avoidance already escaped the same side
        # AND the opposite side has at least minimum clearance, flip. This
        # turns repeated avoidances into a zigzag instead of a drift.
        if self._last_side == "right" and go_right and meas.clear_left > self.LATERAL_MIN:
            go_right = False
            print("       [SphereGuardian] anti-drift: flipping RIGHT→LEFT",
                  flush=True)
        elif self._last_side == "left" and not go_right and meas.clear_right > self.LATERAL_MIN:
            go_right = True
            print("       [SphereGuardian] anti-drift: flipping LEFT→RIGHT",
                  flush=True)

        side_out = "side_step_right" if go_right else "side_step_left"
        side_back = "side_step_left" if go_right else "side_step_right"

        lateral_extent = (
            abs(meas.lateral_extent_right) if go_right
            else abs(meas.lateral_extent_left)
        )
        lateral = min(
            self.LATERAL_MAX,
            max(self.LATERAL_MIN, lateral_extent + self.LATERAL_MARGIN),
        )
        forward = min(
            self.FORWARD_MAX,
            max(self.FORWARD_MIN, meas.distance + meas.depth + self.FORWARD_MARGIN),
        )

        steps: list = []
        if result.distance < self.BACK_OFF_TRIGGER:
            steps.append(Step("walk_backward", speed=1.0, distance_m=self.BACK_OFF_DIST))

        steps.append(Step(side_out, speed=1.0, distance_m=lateral))
        steps.append(Step("walk_forward", speed=1.0, distance_m=forward))
        steps.append(Step(side_back, speed=1.0, distance_m=lateral))

        self._avoidance_count += 1
        self._last_side = "right" if go_right else "left"

        if self._avoidance_count % self.CROUCH_PEEK_EVERY == 0:
            steps.append(Step("crawl_forward",
                              speed=self.CROUCH_PEEK_SPEED,
                              distance_m=self.CROUCH_PEEK_DIST))
            print(f"       [SphereGuardian] crouch-peek (crawl_forward) "
                  f"appended after avoidance #{self._avoidance_count}",
                  flush=True)

        side = "RIGHT" if go_right else "LEFT"
        backoff = "yes" if result.distance < self.BACK_OFF_TRIGGER else "no"
        print(f"       [SphereGuardian] compact avoidance: {side} "
              f"lateral={lateral:.2f}m forward={forward:.2f}m backoff={backoff}",
              flush=True)
        return steps
