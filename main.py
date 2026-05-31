"""Chase rescue-dog — basic Go1 + Groq vision LLM demo.

A minimal example of using cadenza-lab to drive a quadruped through a MuJoCo
disaster scene with a real AI brain. Every tick the Groq vision LLM sees the
forward camera + robot state and picks ONE action from the Cadenza action
library via tool-use.

Run:
    mjpython main.py        # macOS — with the MuJoCo viewer
    python   main.py        # no viewer

Requires GROQ_API_KEY in a .env file or in the environment.
"""

import os
import random
from pathlib import Path

from dotenv import load_dotenv

import cadenza
from cadenza import ChainOfThought
from cadenza._assets import ensure_robot_assets

from LLMAdapter import LLMAdapter
from VisualSensor import VisualSensor


load_dotenv()

HERE = Path(__file__).resolve().parent
MODEL_XML = ensure_robot_assets("go1") / "scene.xml"
LOG = HERE / "sequential_run.jsonl"

GOAL = (
    "Descend into the debris field, look around to find a brightly colored "
    "sphere, then climb out the far side. Keep your balance. If you stop "
    "making progress, change tactics — do not repeat the same action."
)
TARGET = (-9.0, 0.0)  # cadenza forward is -X; the robot heads into the valley


def build_scene():
    """Two slopes (entry + exit), three colored spheres, ~30 random debris boxes."""
    slope1 = cadenza.Slope.from_ground(near_x=-4.5, hx=1.0, hy=1.2, angle_deg=20)
    slope2 = cadenza.Slope.from_ground(near_x=-9.0, hx=1.0, hy=1.2, angle_deg=160)
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
            scene.add_box(position=(x, y, h), size=(s, s, h),
                          rgba=(0.72, 0.30, 0.22, 1.0))
            break
    return scene


def main() -> None:
    if not os.environ.get("GROQ_API_KEY"):
        raise SystemExit(
            "GROQ_API_KEY not set. Put it in a .env file:\n"
            "    GROQ_API_KEY=gsk_..."
        )

    scene = build_scene()
    xml_path = scene.compile(MODEL_XML, str(HERE / "disaster_scene.xml"))

    go1 = cadenza.go1(
        xml_path=str(xml_path),
        inference=ChainOfThought(
            model=LLMAdapter(),               # the AI brain (Groq vision LLM)
            sense=[VisualSensor()],           # CLIP perception modality
            goal=GOAL,
            target=TARGET,
            max_steps=80,
            logging=str(LOG),
        ),
    )
    go1.run([go1.walk_forward()])             # seed action; the LLM drives from here
    print(f"\ndecision log -> {LOG}")


if __name__ == "__main__":
    main()
