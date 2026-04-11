import html
import importlib
import json
import math
import os
import textwrap
from typing import Any, Dict, List, Optional, Sequence, Tuple

import gradio as gr
import pandas as pd
from openai import OpenAI

try:
    from models import RedlineAction
    from server.redline_env_environment import RedlineEnvironment
except ModuleNotFoundError:
    from redline_env.models import RedlineAction
    from redline_env.server.redline_env_environment import RedlineEnvironment

POLLUTANT_OPTIONS = ["NO2", "PM25", "SO2", "CO"]
DEFAULT_TASK = "medium"
MAP_CACHE: Optional[Dict[str, Any]] = None
_BASELINE_CACHE: Any = None


def _baseline_module() -> Any:
    global _BASELINE_CACHE
    if _BASELINE_CACHE is not None:
        return _BASELINE_CACHE

    placeholder_env = {
        "API_BASE_URL": "https://api.openai.com/v1",
        "MODEL_NAME": "gpt-4.1-mini",
        "API_KEY": "gradio-placeholder-key",
    }
    original_env = {key: os.environ.get(key) for key in placeholder_env}
    try:
        for key, value in placeholder_env.items():
            if not os.environ.get(key):
                os.environ[key] = value
        try:
            _BASELINE_CACHE = importlib.import_module("inference")
        except ModuleNotFoundError:
            _BASELINE_CACHE = importlib.import_module("redline_env.inference")
        return _BASELINE_CACHE
    finally:
        for key, value in original_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _env_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_map_cache() -> Dict[str, Any]:
    global MAP_CACHE
    if MAP_CACHE is None:
        csv_path = os.path.join(_env_root(), "Panaji_ULTRA.csv")
        frame = pd.read_csv(csv_path, usecols=["latitude", "longitude"])
        latitudes = [float(value) for value in frame["latitude"].drop_duplicates().tolist()]
        longitudes = [float(value) for value in frame["longitude"].drop_duplicates().tolist()]
        MAP_CACHE = {
            "latitudes": latitudes,
            "longitudes": longitudes,
            "center": [
                sum(latitudes) / len(latitudes),
                sum(longitudes) / len(longitudes),
            ],
        }
    return MAP_CACHE


def _to_latlon(point: Tuple[int, int]) -> List[float]:
    lookup = _load_map_cache()
    x = min(max(int(point[0]), 0), len(lookup["longitudes"]) - 1)
    y = min(max(int(point[1]), 0), len(lookup["latitudes"]) - 1)
    return [lookup["latitudes"][y], lookup["longitudes"][x]]


def _build_local_candidates(
    current: Tuple[int, int],
    goal: Tuple[int, int],
    sensors: Dict[str, float],
) -> Dict[str, Dict[str, float]]:
    baseline = _baseline_module()
    current_distance = math.dist(current, goal)
    candidates: Dict[str, Dict[str, float]] = {}
    for direction, (dx, dy) in baseline.ALL_DELTAS.items():
        next_pos = (current[0] + dx, current[1] + dy)
        candidates[direction] = {
            "next_pos": next_pos,
            "toxicity": float(sensors.get(direction, 1.0)),
            "delta_to_goal": current_distance - math.dist(next_pos, goal),
        }
    return candidates


def _summarize_candidates(candidates: Dict[str, Dict[str, float]]) -> str:
    valid = [(direction, meta) for direction, meta in candidates.items() if meta["toxicity"] < 1.0]
    ranked = sorted(
        valid,
        key=lambda item: (
            item[1]["delta_to_goal"] <= 0,
            item[1]["toxicity"],
            -item[1]["delta_to_goal"],
        ),
    )
    lines: List[str] = []
    for direction, meta in ranked:
        progress_tag = "closer" if meta["delta_to_goal"] > 0 else "farther"
        lines.append(
            f"- {direction}: next={meta['next_pos']}, tox={meta['toxicity']:.3f}, delta={meta['delta_to_goal']:+.3f}, {progress_tag}"
        )
    return "\n".join(lines)


def _recent_visit_count(
    position: Tuple[int, int],
    recent_positions: List[Tuple[int, int]],
    lookback: int,
) -> int:
    return sum(1 for past in recent_positions[-lookback:] if past == position)


def _is_local_cycle(
    recent_positions: List[Tuple[int, int]],
    window: int = 12,
    unique_cap: int = 5,
) -> bool:
    return len(recent_positions) >= window and len(set(recent_positions[-window:])) <= unique_cap


def _choose_non_looping_direction(
    task: str,
    candidates: Dict[str, Dict[str, float]],
    preferred_direction: str,
    recent_positions: List[Tuple[int, int]],
) -> str:
    if task == "medium":
        visit_lookback = 12
        preferred_next = candidates[preferred_direction]["next_pos"]
        previous_pos = recent_positions[-2] if len(recent_positions) >= 2 else None
        preferred_visits = _recent_visit_count(preferred_next, recent_positions, visit_lookback)
        if preferred_next != previous_pos and preferred_visits <= 1:
            return preferred_direction

        valid_moves = [
            (direction, meta)
            for direction, meta in candidates.items()
            if meta["toxicity"] < 1.0
        ]
        if not valid_moves:
            return preferred_direction

        best = min(
            valid_moves,
            key=lambda item: (
                item[1]["next_pos"] == previous_pos,
                _recent_visit_count(item[1]["next_pos"], recent_positions, visit_lookback),
                item[1]["delta_to_goal"] <= 0,
                -item[1]["delta_to_goal"],
                item[1]["toxicity"],
            ),
        )
        return best[0]

    lookback = 10 if task == "medium" else 3
    recent_set = set(recent_positions[-lookback:])
    preferred_next = candidates[preferred_direction]["next_pos"]
    if preferred_next not in recent_set:
        return preferred_direction

    valid_non_looping = [
        (direction, meta)
        for direction, meta in candidates.items()
        if meta["toxicity"] < 1.0 and meta["next_pos"] not in recent_set
    ]
    if not valid_non_looping:
        return preferred_direction

    progressive_non_looping = [
        (direction, meta)
        for direction, meta in valid_non_looping
        if meta["delta_to_goal"] > 0
    ]
    ranked_pool = progressive_non_looping or valid_non_looping
    best = min(
        ranked_pool,
        key=lambda item: (
            item[1]["toxicity"],
            item[1]["delta_to_goal"] <= 0,
            -item[1]["delta_to_goal"],
        ),
    )
    return best[0]


def _fallback_direction(
    task: str,
    current: Tuple[int, int],
    goal: Tuple[int, int],
    sensors: Dict[str, float],
) -> str:
    candidates = _build_local_candidates(current, goal, sensors)
    valid = [direction for direction, meta in candidates.items() if meta["toxicity"] < 1.0]
    if not valid:
        return "N"

    if task == "medium":
        ranked = sorted(valid, key=lambda direction: (candidates[direction]["toxicity"], -candidates[direction]["delta_to_goal"]))
    else:
        ranked = sorted(
            valid,
            key=lambda direction: (
                candidates[direction]["delta_to_goal"] <= 0,
                candidates[direction]["toxicity"],
                -candidates[direction]["delta_to_goal"],
            ),
        )
    return ranked[0]


def _build_user_prompt(
    task: str,
    step: int,
    current: Tuple[int, int],
    goal: Tuple[int, int],
    sensors: Dict[str, float],
    planner_move: Optional[str],
    planner_preview: List[Tuple[int, int]],
    planner_note: str,
    recent_positions: List[Tuple[int, int]],
) -> str:
    baseline = _baseline_module()
    goal_dx = goal[0] - current[0]
    goal_dy = goal[1] - current[1]
    candidates = _build_local_candidates(current, goal, sensors)
    planner_tox = float(sensors.get(planner_move, 1.0)) if planner_move else 1.0
    task_note = {
        "easy": "Easy: short trip to the goal with as little exposure as practical.",
        "medium": "Medium: minimize exposure first, even if the route is longer.",
        "hard": "Hard: pollution changes over time and by location, so favor moves that are clean now while preserving options for the next few steps.",
    }.get(task, "Reach the goal safely.")

    planner_block = textwrap.dedent(
        f"""
        A* planner summary:
        - Search mode: 4-direction road-safe shortest-path A*
        - Pollution role in A*: none
        - Diagonal use: only allowed as corner smoothing on open-road curves
        - Recommended next move: {planner_move}
        - Planner move toxicity: {planner_tox:.3f}
        - Planner note: {planner_note}
        - Planned preview: {planner_preview}
        """
    ).strip()
    if task == "medium":
        planner_block = textwrap.dedent(
            f"""
            A* planner summary:
            - Search mode: 4-direction road-safe shortest-path A*
            - Pollution role in A*: none
            - Planner next move: {planner_move}
            - Planner heading only: {baseline.planner_heading_note(planner_preview)}
            - IMPORTANT: for medium, shortest path is weak guidance only. Clean air matters more.
            """
        ).strip()
    elif task == "hard":
        planner_block = textwrap.dedent(
            f"""
            A* planner summary:
            - Search mode: 4-direction road-safe shortest-path A*
            - Pollution role in A*: none
            - Planner heading only: {baseline.planner_heading_note(planner_preview)}
            - IMPORTANT: for hard, local sensor changes matter more than the planner. Treat the planner as a loose compass only.
            """
        ).strip()

    return textwrap.dedent(
        f"""
        {task_note}
        Step: {step}
        Current: {current}
        Goal: {goal}
        Goal delta: dx={goal_dx}, dy={goal_dy}
        Coordinate reminder: x increases East and y increases South.
        Sensors: {sensors}

        {planner_block}

        Valid local move table:
        {_summarize_candidates(candidates)}

        STRICT INSTRUCTIONS:
        - Use the planner as destination guidance only.
        - Pollution avoidance is your task, based on local sensors.
        - Recent positions: {recent_positions[-4:]}
        - Avoid stepping back into any of the last 3 positions when another valid move exists.
        - If any valid move has delta_to_goal > 0 and toxicity at least 0.03 lower than the planner move, DO NOT choose the planner move.
        - If the planner move is among the dirtiest progressive options, reject it.
        - Only choose a non-progressive move when all progressive moves are blocked or much dirtier.
        - Do not say you are considering a cleaner alternative and then still choose the dirtier planner move.
        - MEDIUM MODE OVERRIDE: keep moving through cleaner air, but do not camp in one local pocket; if you start circling, immediately take the cleanest progressive move into less-visited territory.
        - HARD MODE OVERRIDE: local sensor changes outrank planner advice. Prefer cleaner progressive reroutes over planner alignment.
        """
    ).strip()


def _resolve_direction(
    task: str,
    current: Tuple[int, int],
    goal: Tuple[int, int],
    sensors: Dict[str, float],
    planner_move: Optional[str],
    model_move: Optional[str],
    recent_positions: List[Tuple[int, int]],
    no_progress_streak: int,
    medium_detour_credit: int,
) -> Tuple[str, bool]:
    candidates = _build_local_candidates(current, goal, sensors)
    fallback = _fallback_direction(task, current, goal, sensors)
    planner_choice = planner_move if planner_move and float(sensors.get(planner_move, 1.0)) < 1.0 else fallback
    planned_meta = candidates[planner_choice]

    if task == "medium":
        visit_lookback = 12
        valid_moves = [
            (direction, meta)
            for direction, meta in candidates.items()
            if meta["toxicity"] < 1.0
        ]
        progressive_moves = [
            (direction, meta)
            for direction, meta in valid_moves
            if meta["delta_to_goal"] > 0
        ]
        recent_medium = set(recent_positions[-10:])
        frontier_moves = [
            (direction, meta)
            for direction, meta in valid_moves
            if meta["next_pos"] not in recent_medium
        ]
        frontier_progressive_moves = [
            (direction, meta)
            for direction, meta in progressive_moves
            if meta["next_pos"] not in recent_medium
        ]
        medium_pool = frontier_progressive_moves or progressive_moves or frontier_moves or valid_moves
        medium_choice = min(
            medium_pool,
            key=lambda item: (
                _recent_visit_count(item[1]["next_pos"], recent_positions, visit_lookback),
                item[1]["delta_to_goal"] <= 0,
                -item[1]["delta_to_goal"],
                item[1]["toxicity"],
            ),
        )[0]
        if _is_local_cycle(recent_positions) or no_progress_streak >= 5:
            return _choose_non_looping_direction(task, candidates, medium_choice, recent_positions), False

        if model_move in candidates and candidates[model_move]["toxicity"] < 1.0:
            model_meta = candidates[model_move]
            model_visits = _recent_visit_count(model_meta["next_pos"], recent_positions, visit_lookback)
            if (
                medium_detour_credit > 0
                and model_visits == 0
                and model_meta["toxicity"] <= planned_meta["toxicity"] - 0.05
                and model_meta["delta_to_goal"] >= -0.35
            ):
                return _choose_non_looping_direction(task, candidates, model_move, recent_positions), True
            if model_meta["delta_to_goal"] > 0 and model_visits <= 2:
                return _choose_non_looping_direction(task, candidates, model_move, recent_positions), False
            if (
                model_visits == 0
                and model_meta["toxicity"] <= planned_meta["toxicity"] - 0.08
                and model_meta["delta_to_goal"] >= -0.25
            ):
                return _choose_non_looping_direction(task, candidates, model_move, recent_positions), False

        cleaner_progressive = [
            (direction, meta)
            for direction, meta in progressive_moves
            if meta["toxicity"] <= planned_meta["toxicity"] - 0.03
        ]
        if cleaner_progressive:
            best_cleaner = min(
                cleaner_progressive,
                key=lambda item: (
                    _recent_visit_count(item[1]["next_pos"], recent_positions, visit_lookback),
                    item[1]["toxicity"],
                    -item[1]["delta_to_goal"],
                ),
            )[0]
            return _choose_non_looping_direction(task, candidates, best_cleaner, recent_positions), False
        return _choose_non_looping_direction(task, candidates, planner_choice, recent_positions), False

    if task == "hard":
        valid_moves = [
            (direction, meta)
            for direction, meta in candidates.items()
            if meta["toxicity"] < 1.0
        ]
        progressive_moves = [
            (direction, meta)
            for direction, meta in valid_moves
            if meta["delta_to_goal"] > 0
        ]

        if model_move in candidates and candidates[model_move]["toxicity"] < 1.0:
            model_meta = candidates[model_move]
            if model_meta["delta_to_goal"] > 0:
                return _choose_non_looping_direction(task, candidates, model_move, recent_positions), False

        cleaner_progressive = [
            (direction, meta)
            for direction, meta in progressive_moves
            if meta["toxicity"] <= planned_meta["toxicity"] - 0.01
        ]
        if cleaner_progressive:
            best_reroute = min(
                cleaner_progressive,
                key=lambda item: (item[1]["toxicity"], -item[1]["delta_to_goal"]),
            )[0]
            return _choose_non_looping_direction(task, candidates, best_reroute, recent_positions), False

        if model_move in candidates and candidates[model_move]["toxicity"] < 1.0:
            model_meta = candidates[model_move]
            if model_meta["toxicity"] <= planned_meta["toxicity"] - 0.08 and model_meta["delta_to_goal"] >= -0.25:
                return _choose_non_looping_direction(task, candidates, model_move, recent_positions), False

    progressive = [
        (direction, meta)
        for direction, meta in candidates.items()
        if meta["toxicity"] < 1.0 and meta["delta_to_goal"] > 0
    ]
    cleaner_progressive = [
        (direction, meta)
        for direction, meta in progressive
        if meta["toxicity"] <= planned_meta["toxicity"] - 0.03
    ]
    if cleaner_progressive and (model_move is None or model_move == planner_choice):
        best_cleaner = min(
            cleaner_progressive,
            key=lambda item: (item[1]["toxicity"], -item[1]["delta_to_goal"]),
        )
        return _choose_non_looping_direction(task, candidates, best_cleaner[0], recent_positions), False

    if model_move is None or model_move not in candidates:
        return _choose_non_looping_direction(task, candidates, planner_choice, recent_positions), False
    if float(sensors.get(model_move, 1.0)) >= 1.0:
        return _choose_non_looping_direction(task, candidates, planner_choice, recent_positions), False

    model_meta = candidates[model_move]
    if model_meta["delta_to_goal"] > 0:
        return _choose_non_looping_direction(task, candidates, model_move, recent_positions), False
    if model_meta["toxicity"] + 0.18 < planned_meta["toxicity"] and model_meta["delta_to_goal"] >= planned_meta["delta_to_goal"] - 1.0:
        return _choose_non_looping_direction(task, candidates, model_move, recent_positions), False
    return _choose_non_looping_direction(task, candidates, planner_choice, recent_positions), False


def _request_model_direction(client: OpenAI, user_prompt: str) -> Optional[str]:
    baseline = _baseline_module()
    completion = client.chat.completions.create(
        model=os.getenv("MODEL_NAME") or baseline.MODEL_NAME,
        messages=[
            {"role": "system", "content": baseline.SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
        max_tokens=300,
        stream=False,
    )
    raw_response = (completion.choices[0].message.content or "").strip()
    return baseline.parse_direction(raw_response)


def _make_model_client() -> Tuple[Optional[OpenAI], str]:
    baseline = _baseline_module()
    api_key = os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY") or os.getenv("HF_TOKEN")
    base_url = os.getenv("API_BASE_URL") or os.getenv("OPENAI_BASE_URL") or baseline.API_BASE_URL
    model_name = os.getenv("MODEL_NAME") or baseline.MODEL_NAME
    if not api_key:
        return None, f"live model disabled: missing API_KEY, OPENAI_API_KEY, or HF_TOKEN for {model_name}"
    return OpenAI(base_url=base_url, api_key=api_key), f"live model enabled: {model_name}"


def _score_episode(success: bool, initial_distance: float, final_distance: float) -> float:
    if success or initial_distance == 0:
        return 1.0
    return max(0.0, min(1.0, (initial_distance - final_distance) / initial_distance))


def _run_episode(
    policy: str,
    task: str,
    pollutants: Sequence[str],
    origin: Tuple[int, int],
    goal: Tuple[int, int],
    client: Optional[OpenAI],
) -> Dict[str, Any]:
    baseline = _baseline_module()
    env = RedlineEnvironment()
    max_steps = int(os.getenv("REDLINE_ENV_V4_MAX_STEPS", str(baseline.DEFAULT_MAX_STEPS.get(task, 120))))
    rewards: List[float] = []
    exposures: List[float] = []
    directions: List[str] = []
    recent_positions: List[Tuple[int, int]] = []
    best_distance = float("inf")
    no_progress_streak = 0
    medium_detour_credit = 2
    runtime_note = ""

    try:
        obs = env.reset(origin=origin, destination=goal, pollutant=list(pollutants), task=task)
        path: List[Tuple[int, int]] = [tuple(obs.current_position)]
        initial_distance = math.dist(tuple(obs.current_position), tuple(obs.goal_position))
        success = tuple(obs.current_position) == tuple(obs.goal_position)

        for step in range(1, max_steps + 1):
            if obs.done or success:
                break

            current = tuple(obs.current_position)
            goal_position = tuple(obs.goal_position)
            sensors = dict(obs.get_actions)
            recent_positions.append(current)

            current_distance = math.dist(current, goal_position)
            if current_distance + 1e-6 < best_distance:
                best_distance = current_distance
                no_progress_streak = 0
                if task == "medium":
                    medium_detour_credit = 2
            else:
                no_progress_streak += 1

            planned_path = baseline.astar_path(env.active_obstacle_mask, current, goal_position)
            planner_move, planner_preview, planner_note = baseline.planner_next_move(env.active_obstacle_mask, planned_path)

            if policy == "a_star":
                final_move = planner_move or _fallback_direction(task, current, goal_position, sensors)
            else:
                model_move = None
                if client is not None:
                    try:
                        user_prompt = _build_user_prompt(
                            task,
                            step,
                            current,
                            goal_position,
                            sensors,
                            planner_move,
                            planner_preview,
                            planner_note,
                            recent_positions,
                        )
                        model_move = _request_model_direction(client, user_prompt)
                    except Exception as exc:
                        if not runtime_note:
                            runtime_note = f"live model fallback after step {step}: {exc}"

                final_move, detour_used = _resolve_direction(
                    task,
                    current,
                    goal_position,
                    sensors,
                    planner_move,
                    model_move,
                    recent_positions,
                    no_progress_streak,
                    medium_detour_credit,
                )
                if task == "medium" and detour_used:
                    medium_detour_credit = 0

            if not final_move:
                final_move = _fallback_direction(task, current, goal_position, sensors)

            directions.append(final_move)
            obs = env.step(RedlineAction(direction=final_move))
            current_position = tuple(obs.current_position)
            path.append(current_position)
            rewards.append(float(obs.reward or 0.0))
            exposures.append(float(env._get_cell_toxicity(int(current_position[0]), int(current_position[1]))))

            if obs.done:
                success = current_position == tuple(obs.goal_position)
                break

        final_position = tuple(obs.current_position)
        final_distance = math.dist(final_position, tuple(obs.goal_position))
        return {
            "policy": policy,
            "success": success,
            "steps": len(rewards),
            "score": _score_episode(success, initial_distance, final_distance),
            "total_reward": float(sum(rewards)),
            "cumulative_exposure": float(sum(exposures)),
            "avg_exposure": float(sum(exposures) / len(exposures)) if exposures else 0.0,
            "final_distance": float(final_distance),
            "step_cap": max_steps,
            "path": path,
            "directions": directions,
            "runtime_note": runtime_note,
            "done": bool(obs.done),
            "origin": tuple(origin),
            "goal": tuple(goal),
        }
    finally:
        try:
            env.close()
        except Exception:
            pass


def _metrics_frame(results: Sequence[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for item in results:
        rows.append(
            {
                "route": "A*" if item["policy"] == "a_star" else "Model",
                "success": "yes" if item["success"] else "no",
                "steps": item["steps"],
                "score": round(item["score"], 3),
                "total_reward": round(item["total_reward"], 3),
                "cumulative_exposure": round(item["cumulative_exposure"], 3),
                "avg_exposure": round(item["avg_exposure"], 3),
                "final_distance": round(item["final_distance"], 3),
            }
        )
    return pd.DataFrame(rows)


def _build_status_markdown(
    task: str,
    pollutants: Sequence[str],
    origin: Tuple[int, int],
    goal: Tuple[int, int],
    model_status: str,
    results: Sequence[Dict[str, Any]],
) -> str:
    lines = [
        "### Panaji comparison",
        f"- task: `{task}`",
        f"- pollutants: `{', '.join(pollutants)}`",
        f"- origin: `{origin}`",
        f"- goal: `{goal}`",
        f"- step cap: `{results[0]['step_cap']}`",
        f"- model status: {model_status}",
    ]
    runtime_note = next((item["runtime_note"] for item in results if item["policy"] == "model" and item["runtime_note"]), "")
    if runtime_note:
        lines.append(f"- runtime note: {runtime_note}")
    return "\n".join(lines)


def _build_map_html(
    origin: Tuple[int, int],
    goal: Tuple[int, int],
    a_star_path: Sequence[Tuple[int, int]],
    model_path: Sequence[Tuple[int, int]],
) -> str:
    payload = {
        "origin": _to_latlon(origin),
        "goal": _to_latlon(goal),
        "a_star": [_to_latlon(point) for point in a_star_path],
        "model": [_to_latlon(point) for point in model_path],
    }
    srcdoc = f"""<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <link
      rel="stylesheet"
      href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"
      integrity="sha256-p4NxAoJBhIIN+hmNHrzRCf9tD/miZyoHS5obTRR9BMY="
      crossorigin=""
    />
    <style>
      html, body, #map {{
        margin: 0;
        width: 100%;
        height: 100%;
      }}
      body {{
        background: linear-gradient(135deg, #f4f0e8 0%, #eef6ff 100%);
        font-family: Georgia, serif;
      }}
      #map {{
        border-radius: 16px;
      }}
      .leaflet-container {{
        background: #dfeaf3;
      }}
      .map-chip {{
        background: #123c69;
        color: white;
        border-radius: 999px;
        padding: 4px 8px;
        font-size: 12px;
        font-weight: 700;
      }}
    </style>
  </head>
  <body>
    <div id="map"></div>
    <script
      src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"
      integrity="sha256-20nQCchB9co0qIjJZRGuk2/Z9VM+kNiyxNV1lvTlZBo="
      crossorigin=""
    ></script>
    <script>
      const data = {json.dumps(payload)};
      const map = L.map("map", {{ zoomControl: true, preferCanvas: true }});
      L.tileLayer("https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png", {{
        maxZoom: 19,
        attribution: "&copy; OpenStreetMap contributors"
      }}).addTo(map);

      const startIcon = L.divIcon({{
        className: "",
        html: '<div class="map-chip">start</div>',
        iconSize: [48, 24],
        iconAnchor: [24, 12]
      }});
      const goalIcon = L.divIcon({{
        className: "",
        html: '<div class="map-chip" style="background:#6a040f;">goal</div>',
        iconSize: [44, 24],
        iconAnchor: [22, 12]
      }});

      const layers = [];
      const aStar = L.polyline(data.a_star, {{
        color: "#14746f",
        weight: 6,
        opacity: 0.88,
        lineJoin: "round"
      }}).addTo(map);
      layers.push(aStar);

      const model = L.polyline(data.model, {{
        color: "#bc4749",
        weight: 5,
        opacity: 0.88,
        lineJoin: "round"
      }}).addTo(map);
      layers.push(model);

      layers.push(L.marker(data.origin, {{ icon: startIcon }}).addTo(map));
      layers.push(L.marker(data.goal, {{ icon: goalIcon }}).addTo(map));

      const group = L.featureGroup(layers);
      const bounds = group.getBounds();
      if (bounds.isValid()) {{
        map.fitBounds(bounds.pad(0.35));
      }} else {{
        map.setView(data.origin, 14);
      }}
    </script>
  </body>
</html>"""
    return f"""
<div style="background:linear-gradient(135deg,#f4f0e8 0%,#eef6ff 100%);border:1px solid #d8d2c4;border-radius:20px;padding:14px;">
  <div style="display:flex;gap:10px;flex-wrap:wrap;margin-bottom:12px;font-family:Georgia,serif;">
    <span style="background:#123c69;color:#fff;padding:6px 10px;border-radius:999px;">shared start and goal</span>
    <span style="background:#14746f;color:#fff;padding:6px 10px;border-radius:999px;">a* route</span>
    <span style="background:#bc4749;color:#fff;padding:6px 10px;border-radius:999px;">model route</span>
  </div>
  <iframe
    srcdoc="{html.escape(srcdoc, quote=True)}"
    style="width:100%;height:560px;border:0;border-radius:16px;overflow:hidden;background:#eef6ff;"
    loading="lazy"
    referrerpolicy="no-referrer"
  ></iframe>
</div>
"""


def _run_comparison(task: str, pollutants: Sequence[str]):
    selected_task = task or DEFAULT_TASK
    selected_pollutants = list(pollutants) if pollutants else ["NO2"]
    seed_env = RedlineEnvironment()
    try:
        seed_obs = seed_env.reset(pollutant=selected_pollutants, task=selected_task)
        origin = tuple(seed_obs.current_position)
        goal = tuple(seed_obs.goal_position)
    finally:
        try:
            seed_env.close()
        except Exception:
            pass

    client, model_status = _make_model_client()
    a_star_result = _run_episode("a_star", selected_task, selected_pollutants, origin, goal, client=None)
    model_result = _run_episode("model", selected_task, selected_pollutants, origin, goal, client=client)

    report_payload = {
        "task": selected_task,
        "pollutants": selected_pollutants,
        "origin": origin,
        "goal": goal,
        "a_star": a_star_result,
        "model": model_result,
    }
    return (
        _build_status_markdown(selected_task, selected_pollutants, origin, goal, model_status, [a_star_result, model_result]),
        _metrics_frame([a_star_result, model_result]),
        _build_map_html(origin, goal, a_star_result["path"], model_result["path"]),
        json.dumps(report_payload, indent=2),
    )


def build_redline_gradio(
    web_manager: Any,
    action_fields: List[Dict[str, Any]],
    metadata: Any,
    is_chat_env: bool,
    title: str,
    quick_start_md: str,
) -> gr.Blocks:
    with gr.Blocks(title="Redline Command Center") as demo:
        gr.HTML(
            """
            <div style="background:linear-gradient(135deg,#1f3b4d 0%,#2a6f97 60%,#f4d35e 100%);border-radius:24px;padding:22px 24px;color:#fdfdfd;">
              <div style="font-family:Georgia,serif;font-size:32px;font-weight:700;line-height:1.1;">Redline command center</div>
              <div style="margin-top:10px;font-size:15px;max-width:900px;">
                Compare the raw A* path against the model-guided baseline on the final Panaji map, with the same origin and destination for both runs.
              </div>
            </div>
            """
        )
        with gr.Row():
            task = gr.Dropdown(
                choices=["easy", "medium", "hard"],
                value=DEFAULT_TASK,
                label="Task",
            )
            pollutants = gr.CheckboxGroup(
                choices=POLLUTANT_OPTIONS,
                value=["NO2"],
                label="Pollutants",
            )
            run_button = gr.Button("Run comparison", variant="primary")
        status = gr.Markdown(
            "Choose a task and pollutant set, then run a comparison on Panaji.",
        )
        metrics = gr.Dataframe(
            headers=[
                "route",
                "success",
                "steps",
                "score",
                "total_reward",
                "cumulative_exposure",
                "avg_exposure",
                "final_distance",
            ],
            interactive=False,
            wrap=True,
            label="Comparison metrics",
        )
        map_html = gr.HTML(
            "<div style='padding:22px;border:1px dashed #c9c5bc;border-radius:18px;'>Map preview will appear here after the first run.</div>",
        )
        report = gr.Code(
            label="Comparison report",
            language="json",
            interactive=False,
        )

        run_button.click(
            fn=_run_comparison,
            inputs=[task, pollutants],
            outputs=[status, metrics, map_html, report],
        )

    return demo
