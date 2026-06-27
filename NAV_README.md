# chase-mission — closed-loop VLA navigation

Proof that the dog navigates **on the VLA**, reading its live coordinates, and
genuinely steers **around** obstacles to find the spheres — nothing scripted at
run time.

## What changed vs the old setup

The previous `lora_smolvla.py` hard-coded a per-phase action list (`PHASE_PLANS`)
and trained the LoRA to replay it open-loop. That is a memorised script, not
navigation. It is left in the repo for reference; the navigation system below
replaces it.

cadenza-lab's `LoRAActionDecoder` is a *positional* goal→action-sequence decoder
(action 0 is read from goal-token slot 0, which is a constant BOS embedding), so
it cannot map a whole observation sentence to a single action — verified
empirically (slot-0 cosine = 1.0000 across all prompts). For per-tick
observation→action control we use the standard VLA action-head design instead: a
small trainable head on top of frozen SmolVLA's pooled hidden state.

## Files

| file | role |
|------|------|
| `nav_env.json` | flat scene: 3 spheres on the corridor, 1 rubble box per leg offset onto the path |
| `nav_policy.py` | turns live `(x, y, yaw)` into the observation prompt (train + run) and the geometric expert action (train labels only) |
| `nav_head.py` | the trained policy head on frozen SmolVLA + `decide()` |
| `nav_train.py` | imitation-trains the head on the discrete observation→action table |
| `vla_navigate.py` | the closed-loop runner + coordinate-trace printout |

## How the loop works (every tick)

1. read the robot's **live coordinates** `(x, y, yaw)` from the running `CustomEnv`;
2. from those coordinates compute where the current sphere is (bearing/distance)
   and which side the nearest rubble is on — `nav_policy`;
3. build the natural-language observation prompt and let **SmolVLA → the trained
   head** pick the action (`nav_head.decide`). This is the only thing choosing the
   move;
4. step the env and print the trace.

The head is **imitation-trained** (behaviour cloning) on a geometric expert over
the full discrete observation table — so the expert is the training signal, but
at run time every move is the VLA head's output conditioned on live coordinates.
Avoidance uses `side_step` (strafe around the obstacle while keeping heading on
the goal); seeking uses `walk_forward`/`turn`. Signalling a found sphere is a
symbolic control-panel action (kept out of the learned head on purpose).

## Run it

```bash
VENV=/Users/akshparekh/Documents/cadenza-projects/rescue-dog/.venv/bin/python
REPO=/Users/akshparekh/Documents/cadenza-cli      # for cadenza_cli on PYTHONPATH

# 1. train the policy head on frozen SmolVLA (one-time; ~30s on CPU)
PYTHONPATH=$REPO $VENV chase-mission/nav_train.py

# 2. run the closed-loop navigator and watch the coordinate trace
PYTHONPATH=$REPO $VENV chase-mission/vla_navigate.py
```

Needs the rescue-dog venv (lerobot + cadenza-lab 1.5.1) with the repo on
`PYTHONPATH` for `cadenza_cli`.

## Last verified result

```
closed-loop VLA navigation — 3 spheres, 3 rubble piles, FOV 90°/4m
 t1  walk_forward toward s1 → rubble on the right → VLA side_step_left (strafes around)
 ... → reaches s1 → SIGNAL → s2 (strafe around -Y rubble) → SIGNAL → s3 → SIGNAL
result: 3/3 spheres found in 38 ticks
control panel: 3 correct / 0 false-positive / 0 missed
```

## Honest scope

- The head is behaviour-cloned from a geometric reactive controller; it is the VLA
  that runs the policy at inference, but the *competence ceiling* is that expert.
- The observation is fed to SmolVLA as text built from coordinates (plus the
  optional camera token the encoder supports); it is not learned end-to-end from
  pixels.
- The cadenza-lab go1 gait flips on contact and turning barely translates, so the
  scene keeps real but clearable berth; tight maze-threading is not reliable here.
