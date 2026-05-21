import cadenza
from cadenza.inference import ChainOfThought
from cadenza._assets import ensure_robot_assets

MODEL_XML = ensure_robot_assets("go1") / "scene.xml"

#Main physical AI/sensor models
from VisualSensor import VisualSensor as img
from LLMAdapter import LLMAdapter as VLA

#basic/fundamental libraries
import random
from dotenv import load_dotenv
load_dotenv()

ANGLE_DEG = 20

slope1 = cadenza.Slope.from_ground(
    near_x=-4.5, hx=1.0, hy=1.2, angle_deg=20,
)

slope2 = cadenza.Slope.from_ground(
    near_x=-9.0, hx=1.0, hy=1.2, angle_deg=160,
)


scene = (
    cadenza.Scene()
    .add(slope1)
    .add(slope2)
    .add_sphere(position=(-6.75, 0.0, 0.0), radius=0.15, rgba=(0.22, 0.90, 0.52, 1.0))
    .add_sphere(position=(-5.25, 0.4, 0.0), radius=0.15, rgba=(0.92, 0.90, 0.22, 1.0))
    .add_sphere(position=(-8.75, -0.7, 0.0), radius=0.15, rgba=(0.62, 0.30, 0.82, 1.0))
    .snake_on_slope(slope1, count=8, snake_y=0.80)
    .snake_on_slope(slope2, count=8, snake_y=-0.80)
)

random.seed(7)
placed: list[tuple[float, float, float]] = []  
for _ in range(30):
    for _attempt in range(20):                   
        x = random.uniform(-9.0, -4.0)           
        y = random.uniform(-2.0, 2.0)
        s = random.uniform(0.08, 0.16)
        if abs(x) < 0.7 and abs(y) < 0.35:       
            continue
        if any((x - px) ** 2 + (y - py) ** 2 < (s + ps + 0.30) ** 2
               for px, py, ps in placed):
            continue
        placed.append((x, y, s))
        h = random.uniform(0.12, 0.22)
        scene.add_box(
            position=(x, y, h),
            size=(s, s, h),
            rgba=(0.72, 0.30, 0.22, 1.0),
        )
        break

# cadenza.view(robot="go1", scene=scene)
xml_path = scene.compile(MODEL_XML, "disaster_scene.xml")

GOAL = (
    "You drive a Unitree Go1 quadruped one action per tick. Pick exactly ONE "
    "action per tick using ONLY the current observation — you have NO memory "
    "of previous ticks. Forward direction is -x (walking forward decreases "
    "pos.x).\n\n"
    "TRUST THE SCORES, NOT YOUR OWN VISUAL JUDGMENT. The observation "
    "includes sphere_visible_score and obstacle_close_score, produced by a "
    "dedicated zero-shot CLIP detector. Those numbers are reliable; do NOT "
    "second-guess them by looking at the image yourself.\n\n"
    "DECISION RULES (check in order; first match wins):\n"
    "1. If body_height < 0.30 m: call stand_up (robot is recovering from "
    "a sit pose).\n"
    "2. If sphere_visible_score > 0.65: call sit (a colored ball is in "
    "view; acknowledge it; rule 1 will stand back up next tick).\n"
    "3. If obstacle_close_score > 0.65: call walk_backward with "
    "distance_m=0.4 (something is blocking the path).\n"
    "4. If pos.x > -2.0: call walk_forward with distance_m=0.4 (still "
    "near spawn, advance into the scene).\n"
    "5. Otherwise: call side_step_right with distance_m=0.4 (default "
    "exploration motion — strafe right in a wide loop).\n\n"
    "FORBIDDEN ACTIONS: never call turn_left, turn_right, "
    "precision_turn_left, precision_turn_right, jump, bound_forward, "
    "rear_kick, shake_hand, or lie_down. The robot must never spin in "
    "place or turn around."
)

# ChainOfThought owns the model + modalities + goal. go1.run() is given a
# trigger step; CoT takes over from the first tick and drives every action
# from there. (The goal-mode path go1.run(goal=...) would bypass CoT
# entirely, which is why the visual sensor never got set up before.)
go1 = cadenza.go1(
    xml_path="disaster_scene.xml",
    inference=ChainOfThought(
        model=VLA(),
        sense=[img()],
        goal=GOAL,
        logging="sequential_run.jsonl",
    ),
)

go1.run([go1.walk_forward()])