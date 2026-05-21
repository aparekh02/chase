import cadenza
from cadenza.inference import Sequential
from cadenza._assets import ensure_robot_assets
from SphereGuardian import SphereGuardian

MODEL_XML = ensure_robot_assets("go1") / "scene.xml"

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

go1 = cadenza.go1(
    xml_path="disaster_scene.xml",
    inference=Sequential(
        guardian=SphereGuardian,
        retries=2,                      # cap avoidances per step → bounded drift
        logging="sequential_run.jsonl",
    ),
)

# Scripted traversal: walk into the scene, then strafe right across it in
# segments. Each step is guardian-protected; obstacles encountered along
# the way are auto-avoided.
go1.run([
    # Staircase traversal: each forward leg is followed by a shorter
    # side-step. Short legs keep individual gait drift small AND let the
    # guardian re-scan the scene more often. Mix of motion types means
    # the run isn't dominated by any single direction.
    go1.walk_forward(distance_m=2.5),
    go1.side_step_right(distance_m=1.5),
    go1.walk_forward(distance_m=2.0),
    go1.side_step_right(distance_m=1.5),
    go1.walk_forward(distance_m=2.0),
    go1.side_step_right(distance_m=1.5),
    go1.walk_backward(distance_m=1.0),
    go1.side_step_left(distance_m=1.0),
])