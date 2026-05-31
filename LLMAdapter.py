"""Groq LLM → cadenza WorldModelAdapter.

Each tick we hand a vision-capable Groq model the camera frame, robot pose,
and the goal, then let it pick exactly one named action from the cadenza
vocabulary via tool use. Output → ProposedAction.
"""

from __future__ import annotations

import base64
import io
import json
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from groq import Groq

from cadenza.stack.adapters.base import (
    WorldModelAdapter,
    AdapterReply,
    ProposedAction,
)


SYSTEM_PROMPT = (
    "You are the navigation brain for a Unitree Go1 quadruped robot in a "
    "MuJoCo simulation. On each tick you receive: a forward camera frame "
    "(LOOK AT IT — it is your primary input), the robot's pose, and a "
    "natural-language goal.\n\n"
    "Pick EXACTLY ONE tool call per tick. The robot does not move unless "
    "you call a tool, but it also only changes direction when you pick a "
    "DIFFERENT tool. So:\n"
    "  - To strafe right → call side_step_right (NOT walk_forward).\n"
    "  - To stop and pose → call sit, then on the next tick call stand_up.\n"
    "  - To back off → call walk_backward.\n"
    "  - Only call walk_forward when the goal genuinely calls for moving "
    "    in the heading direction.\n\n"
    "Read the goal carefully and pick the action that matches what it "
    "asks for RIGHT NOW. If the goal says 'strafe right', do not keep "
    "calling walk_forward — pick side_step_right. If it says 'sit when "
    "you see a sphere' and you see a colored sphere in the camera frame, "
    "pick sit. Do not narrate — just call the tool with sensible scalar "
    "params (distance_m around 0.4, rotation_rad around 0.35)."
)


class LLMAdapter(WorldModelAdapter):
    name = "groq_llm_go1"
    description = "Groq vision LLM reasoner over the cadenza action vocabulary"

    DEFAULT_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        max_tokens: int = 512,
        jpeg_quality: int = 80,
        stuck_window: int = 3,
        stuck_disp_m: float = 0.08,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model_id = model or self.DEFAULT_MODEL
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        if not self.api_key:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Export it in your shell:\n"
                "    export GROQ_API_KEY=gsk_..."
            )
        self.max_tokens = int(max_tokens)
        self.jpeg_quality = int(jpeg_quality)
        self.client: Groq | None = None
        self._tools_cache: list[dict] | None = None
        self._diag_logged: bool = False

        # Self-contained progress tracking: surface a "you are stuck" alert in
        # the prompt so the model doesn't repeat a useless action forever. This
        # is a feature of THIS adapter — no external state, no cadenza-api.
        self.stuck_window = int(stuck_window)
        self.stuck_disp_m = float(stuck_disp_m)
        self._pos_history: list[tuple[float, float]] = []
        self._last_action: str | None = None
        self._repeat_count: int = 0

    def _progress_alert(self, observation: dict) -> str:
        """Build a 'you are stuck' alert from recent displacement + action repeats."""
        pos = observation.get("pos") or [0.0, 0.0, 0.0]
        xy = (float(pos[0]), float(pos[1]))
        self._pos_history.append(xy)
        self._pos_history = self._pos_history[-(self.stuck_window + 1):]
        if len(self._pos_history) <= self.stuck_window:
            return ""
        first = self._pos_history[0]
        disp = ((xy[0] - first[0]) ** 2 + (xy[1] - first[1]) ** 2) ** 0.5
        if disp < self.stuck_disp_m:
            return (
                f"\n\n⚠ NO-PROGRESS ALERT: you have moved only {disp:.2f} m over "
                f"the last {self.stuck_window} ticks"
                + (f" while repeating '{self._last_action}' {self._repeat_count}× in a row"
                   if self._repeat_count >= 2 else "")
                + ". Whatever you just did is NOT working. Do NOT repeat it. Pick a "
                "DIFFERENT action: turn_left/turn_right to change heading, "
                "crawl_forward to get through tight/low debris, or walk_backward "
                "to back out and try another route."
            )
        return ""

    @classmethod
    def detect(cls, root: Path) -> Path | None:
        return None  # adapter always supplied as an instance

    def _load_impl(self) -> None:
        self.client = Groq(api_key=self.api_key)
        print(f"[LLMAdapter] Ready (groq model={self.model_id})")

    def _encode_image(self, frame: np.ndarray) -> str:
        pil = Image.fromarray(frame)
        buf = io.BytesIO()
        pil.save(buf, format="JPEG", quality=self.jpeg_quality)
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{b64}"

    def _tools(self, vocabulary) -> list[dict]:
        if self._tools_cache is None:
            self._tools_cache = [
                {
                    "type": "function",
                    "function": {
                        "name": e["name"],
                        "description": e["description"],
                        "parameters": e["input_schema"],
                    },
                }
                for e in vocabulary.to_tool_schema()
            ]
        return self._tools_cache

    def _user_content(self, observation: dict, goal: str) -> list[dict]:
        pos = observation.get("pos") or [0.0, 0.0, 0.0]
        rpy = observation.get("rpy") or [0.0, 0.0, 0.0]
        body_h = observation.get("body_height", 0.0)
        contacts = observation.get("foot_contacts", [])
        target = observation.get("target_xy")

        sphere_score = observation.get("sphere_visible")
        obstacle_score = observation.get("obstacle_close")

        lines = [
            f"Goal: {goal}",
            "",
            "Robot state:",
            f"  pos (x,y,z): ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}) m",
            f"  rpy (rad):   ({rpy[0]:.2f}, {rpy[1]:.2f}, {rpy[2]:.2f})",
            f"  body_height: {float(body_h):.2f} m",
            f"  foot_contacts: {list(contacts)}",
        ]
        if sphere_score is not None:
            lines.append(
                f"  sphere_visible_score: {float(sphere_score):.2f} "
                f"(0.0=no colored ball in view, 1.0=clear colored ball)"
            )
        if obstacle_score is not None:
            lines.append(
                f"  obstacle_close_score:  {float(obstacle_score):.2f} "
                f"(0.0=clear path, 1.0=obstacle/wall close in front)"
            )
        if target is not None:
            lines.append(f"  target_xy: ({target[0]:.2f}, {target[1]:.2f}) m")
        lines.append("")
        lines.append("Call exactly one tool to decide the next action.")
        text = "\n".join(lines)
        # Self-contained no-progress alert — keeps the model from looping on a
        # useless action. Purely internal to this adapter.
        text += self._progress_alert(observation)

        content: list[dict] = [{"type": "text", "text": text}]
        cam = observation.get("camera")
        if cam is not None:
            content.append({
                "type": "image_url",
                "image_url": {"url": self._encode_image(cam)},
            })
        return content

    def _fallback(self, note: str) -> AdapterReply:
        return AdapterReply(
            actions=[ProposedAction(name="stand", params={}, rationale=note)],
            done=False,
            note=note,
        )

    @staticmethod
    def _unwrap_params(params: dict) -> dict:
        """Llama-4 tool calls often wrap scalars in {"value": x} or {<name>: x}.
        Flatten so cadenza receives plain numbers/ints/bools.
        """
        flat: dict[str, Any] = {}
        for key, val in (params or {}).items():
            if isinstance(val, dict) and len(val) == 1:
                inner_key, inner_val = next(iter(val.items()))
                if inner_key in ("value", key) and not isinstance(inner_val, dict):
                    flat[key] = inner_val
                    continue
            flat[key] = val
        return flat

    def _extract_failed_generation(self, exc: Exception) -> str | None:
        """Pull the raw model output Groq attaches to tool_use_failed 400s."""
        body = getattr(exc, "body", None)
        if isinstance(body, dict):
            err = body.get("error") or {}
            gen = err.get("failed_generation")
            if isinstance(gen, str):
                return gen
        # Fallback: scrape it out of the stringified error.
        m = re.search(r"failed_generation['\"]:\s*['\"](.+?)['\"]\s*\}", str(exc), re.DOTALL)
        if m:
            try:
                return m.group(1).encode("utf-8").decode("unicode_escape")
            except UnicodeDecodeError:
                return m.group(1)
        return None

    def _rescue_from_error(self, exc: Exception, vocabulary) -> AdapterReply | None:
        raw = self._extract_failed_generation(exc)
        if not raw:
            return None
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return None
        if isinstance(payload, dict):
            payload = [payload]
        if not (isinstance(payload, list) and payload):
            return None
        first = payload[0]
        name = first.get("name")
        params = first.get("parameters") or first.get("arguments") or {}
        if isinstance(params, str):
            try:
                params = json.loads(params)
            except json.JSONDecodeError:
                params = {}
        params = self._unwrap_params(params)
        if not name or name not in vocabulary:
            return None
        return AdapterReply(
            actions=[ProposedAction(
                name=name,
                params=params,
                rationale=f"rescued from tool_use_failed: {name}({params})",
            )],
            done=False,
            note=f"rescued: {name}({params})",
        )

    def propose_actions(self, observation, goal, vocabulary, history=None) -> AdapterReply:
        self.load()

        if not self._diag_logged:
            self._diag_logged = True
            cam = observation.get("camera")
            emb = observation.get("visual_embedding")
            print(
                "[LLMAdapter] first tick obs keys: "
                f"{sorted(observation.keys())}", flush=True,
            )
            print(
                "[LLMAdapter]   camera: "
                f"{'shape=' + str(cam.shape) + ' dtype=' + str(cam.dtype) if cam is not None else 'MISSING (LLM has no vision)'}",
                flush=True,
            )
            print(
                "[LLMAdapter]   visual_embedding (from VisualSensor): "
                f"{'shape=' + str(emb.shape) if emb is not None else 'MISSING (sensor not contributing)'}",
                flush=True,
            )

        try:
            resp = self.client.chat.completions.create(
                model=self.model_id,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": self._user_content(observation, goal)},
                ],
                tools=self._tools(vocabulary),
                tool_choice="required",
                max_tokens=self.max_tokens,
                temperature=0.2,
            )
        except Exception as e:
            rescued = self._rescue_from_error(e, vocabulary)
            if rescued is not None:
                self._track(rescued.actions[0].name if rescued.actions else None)
                return rescued
            return self._fallback(f"groq error: {e}")

        msg = resp.choices[0].message
        tool_calls = getattr(msg, "tool_calls", None) or []
        if not tool_calls:
            return self._fallback(f"no tool call; content={msg.content!r}")

        call = tool_calls[0]
        name = call.function.name
        try:
            params = json.loads(call.function.arguments) if call.function.arguments else {}
        except json.JSONDecodeError:
            params = {}

        if name not in vocabulary:
            return self._fallback(f"model picked unknown action {name!r}")

        self._track(name)
        return AdapterReply(
            actions=[ProposedAction(
                name=name,
                params=params,
                rationale=f"groq: {name}({params})",
            )],
            done=False,
            note=f"{name}({params})",
        )

    def _track(self, name: str | None) -> None:
        """Update the repeated-action counter feeding the no-progress alert."""
        if name and name == self._last_action:
            self._repeat_count += 1
        else:
            self._repeat_count = 1
        self._last_action = name
