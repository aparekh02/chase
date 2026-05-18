import cadenza
from cadenza.inference import ChainOfThought
from cadenza._assets import ensure_robot_assets

MODEL_XML = ensure_robot_assets("go1") / "scene.xml"

#Main physical AI/sensor models
from VisualSensor import VisualSensor as img
from SmolVLA import VLA

#basic/fundamental libraries
import random

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
    inference=ChainOfThought(
        logging="sequential_run.jsonl",
    ),
)

go1.setup(
    model=VLA(),
    sense=[img()],
)

go1.run(
    goal="move towards the the group of slopes and boxes in front of you. then navigate the perimeter of the entire system, continuously going right around the entire perimeter. YOUR GOAL IS TO FIND THE SPHERES WITHIN THE SCENE, which are seperated around the group of stuff. When you see one, sit down. then stand up and continue looking for more.",
)