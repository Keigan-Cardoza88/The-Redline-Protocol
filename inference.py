import asyncio
import heapq
import math
import os
import re
import textwrap
import time
from itertools import count
from typing import Dict, List, Optional, Tuple

import numpy as np
from openai import OpenAI
try:
    from client import RedlineEnv
    from models import RedlineAction
except ModuleNotFoundError:
    from .client import RedlineEnv
    from .models import RedlineAction

IMAGE_NAME = os.getenv("IMAGE_NAME") or "redline_env:latest"
API_BASE_URL = os.getenv("API_BASE_URL")
MODEL_NAME = os.getenv("MODEL_NAME")
API_KEY = os.getenv("API_KEY")

TASK_NAME = os.getenv("MY_ENV_V4_TASK", "echo")
BENCHMARK = os.getenv("MY_ENV_V4_BENCHMARK", "my_env_v4")
REGION = "Panaji"
DEFAULT_MAX_STEPS = {
    "easy": 100,
    "medium": 150,
    "hard": 200,
}


def _safe_int(value: Optional[str], default: int) -> int:
    try:
        return int(value) if value is not None else default
    except Exception:
        return default


MAX_STEPS = _safe_int(os.getenv("REDLINE_ENV_V4_MAX_STEPS"), DEFAULT_MAX_STEPS.get(TASK_NAME, 120))

CARDINAL_DELTAS: Dict[str, Tuple[int, int]] = {
    "N": (0, -1),
    "E": (1, 0),
    "S": (0, 1),
    "W": (-1, 0),
}
ALL_DELTAS: Dict[str, Tuple[int, int]] = {
    "N": (0, -1),
    "NE": (1, -1),
    "E": (1, 0),
    "SE": (1, 1),
    "S": (0, 1),
    "SW": (-1, 1),
    "W": (-1, 0),
    "NW": (-1, -1),
}
VALID_DIRECTIONS = set(ALL_DELTAS.keys())
PLANNER_DATA: Optional[np.ndarray] = None

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are an autonomous navigation AI in a grid city.
    Coordinate rule: x increases to the East, y increases to the South.
    That means North decreases y, and South increases y.

    An A* planner already computed a road-safe shortest path for you using only 4-direction search.
    It does NOT optimize for pollution. It only tells you how to stay oriented toward the destination
    without cutting through walls/buildings.

    Your job is to use that path as destination guidance while still using the local toxicity sensors
    to avoid pollution when you can do so sensibly.

    Diagonal motion is only used as a corner-smoothing shortcut when the next two A* steps form an open-road curve.

    STRICT DECISION RULES:
    1. Never choose a move with toxicity 1.0.
    2. The planner move is only orientation help, not an order.
    3. If another valid move is cleaner and still gets closer to the goal, prefer the cleaner move.
    4. Do not blindly repeat the planner move when its toxicity is clearly worse than another progressive move.
    5. Only accept a dirtier move when it gives clearly better destination progress.

    OUTPUT FORMAT RULES:
    - Explain your choice in at most 2 short sentences.
    - The final line MUST be exactly one of: [N], [NE], [E], [SE], [S], [SW], [W], [NW]
    - Do not put any other bracketed text anywhere.
    - If you are running out of space, end immediately with the final bracketed line.
    """
).strip()


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}", flush=True)


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


def load_planner_data() -> np.ndarray:
    global PLANNER_DATA
    if PLANNER_DATA is not None:
        return PLANNER_DATA

    root_dir = os.path.dirname(os.path.abspath(__file__))
    npy_path = os.path.join(root_dir, f"{REGION}_obstacles.npy")
    PLANNER_DATA = np.load(npy_path).astype(bool)
    return PLANNER_DATA


def heuristic(current: Tuple[int, int], goal: Tuple[int, int]) -> float:
    return abs(goal[0] - current[0]) + abs(goal[1] - current[1])


def is_road(obstacle_mask: np.ndarray, x: int, y: int) -> bool:
    return 0 <= y < obstacle_mask.shape[0] and 0 <= x < obstacle_mask.shape[1] and bool(obstacle_mask[y, x])


def reconstruct_path(
    came_from: Dict[Tuple[int, int], Tuple[int, int]],
    current: Tuple[int, int],
) -> List[Tuple[int, int]]:
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path


def astar_path(
    obstacle_mask: np.ndarray,
    start: Tuple[int, int],
    goal: Tuple[int, int],
) -> List[Tuple[int, int]]:
    if start == goal:
        return [start]

    frontier: List[Tuple[float, int, Tuple[int, int]]] = []
    tie_breaker = count()
    heapq.heappush(frontier, (heuristic(start, goal), next(tie_breaker), start))

    came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
    g_score: Dict[Tuple[int, int], float] = {start: 0.0}
    visited: set[Tuple[int, int]] = set()

    while frontier:
        _, _, current = heapq.heappop(frontier)
        if current in visited:
            continue
        if current == goal:
            return reconstruct_path(came_from, current)
        visited.add(current)

        for dx, dy in CARDINAL_DELTAS.values():
            nx, ny = current[0] + dx, current[1] + dy
            if not is_road(obstacle_mask, nx, ny):
                continue

            neighbor = (nx, ny)
            tentative_g = g_score[current] + 1.0

            if tentative_g < g_score.get(neighbor, float("inf")):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score = tentative_g + heuristic(neighbor, goal)
                heapq.heappush(frontier, (f_score, next(tie_breaker), neighbor))

    return []


def direction_from_delta(dx: int, dy: int) -> Optional[str]:
    for direction, (mx, my) in ALL_DELTAS.items():
        if (dx, dy) == (mx, my):
            return direction
    return None


def direction_between(a: Tuple[int, int], b: Tuple[int, int]) -> Optional[str]:
    return direction_from_delta(b[0] - a[0], b[1] - a[1])


def can_smooth_corner(obstacle_mask: np.ndarray, path: List[Tuple[int, int]]) -> bool:
    if len(path) < 3:
        return False
    start, first, second = path[0], path[1], path[2]
    dx = second[0] - start[0]
    dy = second[1] - start[1]
    if abs(dx) != 1 or abs(dy) != 1:
        return False
    if first[0] != start[0] and first[1] != start[1]:
        return False

    corner_a = (start[0] + dx, start[1])
    corner_b = (start[0], start[1] + dy)
    return is_road(obstacle_mask, corner_a[0], corner_a[1]) and is_road(obstacle_mask, corner_b[0], corner_b[1]) and is_road(obstacle_mask, second[0], second[1])


def planner_next_move(obstacle_mask: np.ndarray, path: List[Tuple[int, int]]) -> Tuple[Optional[str], List[Tuple[int, int]], str]:
    if len(path) < 2:
        return None, path[:1], "no path"
    if can_smooth_corner(obstacle_mask, path):
        diagonal = direction_between(path[0], path[2])
        if diagonal is not None:
            return diagonal, path[:6], f"curve-smoothed {diagonal}"
    return direction_between(path[0], path[1]), path[:6], "4-dir A* move"


def planner_heading_note(path: List[Tuple[int, int]]) -> str:
    if len(path) < 2:
        return "no heading"
    target = path[min(len(path) - 1, 6)]
    dx = target[0] - path[0][0]
    dy = target[1] - path[0][1]
    horizontal = "east" if dx > 0 else "west" if dx < 0 else "hold x"
    vertical = "south" if dy > 0 else "north" if dy < 0 else "hold y"
    return f"general heading: {horizontal} and {vertical}"


def build_local_candidates(current: Tuple[int, int], goal: Tuple[int, int], sensors: Dict[str, float]) -> Dict[str, Dict[str, float]]:
    current_distance = math.dist(current, goal)
    candidates: Dict[str, Dict[str, float]] = {}
    for direction, (dx, dy) in ALL_DELTAS.items():
        next_pos = (current[0] + dx, current[1] + dy)
        candidates[direction] = {
            "next_pos": next_pos,
            "toxicity": float(sensors.get(direction, 1.0)),
            "delta_to_goal": current_distance - math.dist(next_pos, goal),
        }
    return candidates


def summarize_candidates(candidates: Dict[str, Dict[str, float]]) -> str:
    valid = [(direction, meta) for direction, meta in candidates.items() if meta["toxicity"] < 1.0]
    ranked = sorted(valid, key=lambda item: (item[1]["delta_to_goal"] <= 0, item[1]["toxicity"], -item[1]["delta_to_goal"]))
    lines: List[str] = []
    for direction, meta in ranked:
        progress_tag = "closer" if meta["delta_to_goal"] > 0 else "farther"
        lines.append(
            f"- {direction}: next={meta['next_pos']}, tox={meta['toxicity']:.3f}, delta={meta['delta_to_goal']:+.3f}, {progress_tag}"
        )
    return "\n".join(lines)


def recent_visit_count(
    position: Tuple[int, int],
    recent_positions: List[Tuple[int, int]],
    lookback: int,
) -> int:
    return sum(1 for past in recent_positions[-lookback:] if past == position)


def is_local_cycle(recent_positions: List[Tuple[int, int]], window: int = 12, unique_cap: int = 5) -> bool:
    if len(recent_positions) < window:
        return False
    return len(set(recent_positions[-window:])) <= unique_cap


def choose_non_looping_direction(
    candidates: Dict[str, Dict[str, float]],
    preferred_direction: str,
    recent_positions: List[Tuple[int, int]],
) -> str:
    if TASK_NAME == "medium":
        visit_lookback = 12
        preferred_next = candidates[preferred_direction]["next_pos"]
        previous_pos = recent_positions[-2] if len(recent_positions) >= 2 else None
        preferred_visits = recent_visit_count(preferred_next, recent_positions, visit_lookback)
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
                recent_visit_count(item[1]["next_pos"], recent_positions, visit_lookback),
                item[1]["delta_to_goal"] <= 0,
                -item[1]["delta_to_goal"],
                item[1]["toxicity"],
            ),
        )
        return best[0]

    lookback = 10 if TASK_NAME == "medium" else 3
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


def fallback_direction(current: Tuple[int, int], goal: Tuple[int, int], sensors: Dict[str, float]) -> str:
    candidates = build_local_candidates(current, goal, sensors)
    valid = [direction for direction, meta in candidates.items() if meta["toxicity"] < 1.0]
    if not valid:
        return "N"

    if TASK_NAME == "medium":
        ranked = sorted(valid, key=lambda d: (candidates[d]["toxicity"], -candidates[d]["delta_to_goal"]))
    else:
        ranked = sorted(valid, key=lambda d: (candidates[d]["delta_to_goal"] <= 0, candidates[d]["toxicity"], -candidates[d]["delta_to_goal"]))
    return ranked[0]


def build_user_prompt(
    step: int,
    current: Tuple[int, int],
    goal: Tuple[int, int],
    sensors: Dict[str, float],
    planner_move: Optional[str],
    planner_preview: List[Tuple[int, int]],
    planner_note: str,
    recent_positions: List[Tuple[int, int]],
) -> str:
    goal_dx = goal[0] - current[0]
    goal_dy = goal[1] - current[1]
    candidates = build_local_candidates(current, goal, sensors)
    planner_tox = float(sensors.get(planner_move, 1.0)) if planner_move else 1.0
    task_note = {
        "easy": "Easy: short trip to the goal with as little exposure as practical.",
        "medium": "Medium: minimize exposure first, even if the route is longer.",
        "hard": "Hard: pollution changes over time and by location, so favor moves that are clean now while preserving options for the next few steps.",
    }.get(TASK_NAME, "Reach the goal safely.")

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
    if TASK_NAME == "medium":
        planner_block = textwrap.dedent(
            f"""
            A* planner summary:
            - Search mode: 4-direction road-safe shortest-path A*
            - Pollution role in A*: none
            - Planner next move: {planner_move}
            - Planner heading only: {planner_heading_note(planner_preview)}
            - IMPORTANT: for medium, shortest path is weak guidance only. Clean air matters more.
            """
        ).strip()
    elif TASK_NAME == "hard":
        planner_block = textwrap.dedent(
            f"""
            A* planner summary:
            - Search mode: 4-direction road-safe shortest-path A*
            - Pollution role in A*: none
            - Planner heading only: {planner_heading_note(planner_preview)}
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
        Coordinate reminder: positive dy means the goal is South.
        Sensors: {sensors}

        {planner_block}

        Valid local move table:
        {summarize_candidates(candidates)}

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


def parse_direction(text: str) -> Optional[str]:
    raw = (text or "").strip().upper()
    if not raw:
        return None

    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    for line in reversed(lines[-5:]):
        exact_bracket = re.fullmatch(r"\[(N|NE|E|SE|S|SW|W|NW)\]", line)
        if exact_bracket:
            return exact_bracket.group(1)
        exact_token = re.fullmatch(r"(N|NE|E|SE|S|SW|W|NW)", line)
        if exact_token:
            return exact_token.group(1)

    bracketed = re.findall(r"\[(N|NE|E|SE|S|SW|W|NW)\]", raw)
    if bracketed:
        return bracketed[-1]

    return None


def request_model_direction(client: OpenAI, user_prompt: str, step: int) -> Optional[str]:
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            max_tokens=300,
            stream=False,
        )
        raw_response = (completion.choices[0].message.content or "").strip()
        print(f"\n--- LLM Thinking (Step {step}) ---")
        print(raw_response)
        print("------------------------------")
        return parse_direction(raw_response)
    except Exception as exc:
        print(f"[DEBUG] model request failed at step {step}: {type(exc).__name__}: {exc}", flush=True)
        return None


def resolve_direction(
    current: Tuple[int, int],
    goal: Tuple[int, int],
    sensors: Dict[str, float],
    planner_move: Optional[str],
    model_move: Optional[str],
    recent_positions: List[Tuple[int, int]],
    no_progress_streak: int,
    medium_detour_credit: int,
) -> Tuple[str, bool]:
    candidates = build_local_candidates(current, goal, sensors)
    fallback = fallback_direction(current, goal, sensors)
    planner_choice = planner_move if planner_move and float(sensors.get(planner_move, 1.0)) < 1.0 else fallback
    planned_meta = candidates[planner_choice]
    if TASK_NAME == "medium":
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
                recent_visit_count(item[1]["next_pos"], recent_positions, visit_lookback),
                item[1]["delta_to_goal"] <= 0,
                -item[1]["delta_to_goal"],
                item[1]["toxicity"],
            ),
        )[0]
        if is_local_cycle(recent_positions) or no_progress_streak >= 5:
            return choose_non_looping_direction(candidates, medium_choice, recent_positions), False

        if model_move in candidates and candidates[model_move]["toxicity"] < 1.0:
            model_meta = candidates[model_move]
            model_visits = recent_visit_count(model_meta["next_pos"], recent_positions, visit_lookback)
            if (
                medium_detour_credit > 0
                and model_visits == 0
                and model_meta["toxicity"] <= planned_meta["toxicity"] - 0.05
                and model_meta["delta_to_goal"] >= -0.35
            ):
                return choose_non_looping_direction(candidates, model_move, recent_positions), True
            if model_meta["delta_to_goal"] > 0 and model_visits <= 2:
                return choose_non_looping_direction(candidates, model_move, recent_positions), False
            if (
                model_visits == 0
                and model_meta["toxicity"] <= planned_meta["toxicity"] - 0.08
                and model_meta["delta_to_goal"] >= -0.25
            ):
                return choose_non_looping_direction(candidates, model_move, recent_positions), False

        cleaner_progressive = [
            (direction, meta)
            for direction, meta in progressive_moves
            if meta["toxicity"] <= planned_meta["toxicity"] - 0.03
        ]
        if cleaner_progressive:
            best_cleaner = min(
                cleaner_progressive,
                key=lambda item: (
                    recent_visit_count(item[1]["next_pos"], recent_positions, visit_lookback),
                    item[1]["toxicity"],
                    -item[1]["delta_to_goal"],
                ),
            )[0]
            return choose_non_looping_direction(candidates, best_cleaner, recent_positions), False
        return choose_non_looping_direction(candidates, planner_choice, recent_positions), False

    if TASK_NAME == "hard":
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
                return choose_non_looping_direction(candidates, model_move, recent_positions), False

        cleaner_progressive = [
            (direction, meta)
            for direction, meta in progressive_moves
            if meta["toxicity"] <= planned_meta["toxicity"] - 0.01
        ]
        if cleaner_progressive:
            best_hard_reroute = min(
                cleaner_progressive,
                key=lambda item: (item[1]["toxicity"], -item[1]["delta_to_goal"]),
            )[0]
            return choose_non_looping_direction(candidates, best_hard_reroute, recent_positions), False

        if model_move in candidates and candidates[model_move]["toxicity"] < 1.0:
            model_meta = candidates[model_move]
            if model_meta["toxicity"] <= planned_meta["toxicity"] - 0.08 and model_meta["delta_to_goal"] >= -0.25:
                return choose_non_looping_direction(candidates, model_move, recent_positions), False

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
        best_cleaner_progressive = min(
            cleaner_progressive,
            key=lambda item: (item[1]["toxicity"], -item[1]["delta_to_goal"]),
        )
        return choose_non_looping_direction(candidates, best_cleaner_progressive[0], recent_positions), False

    if model_move is None or model_move not in candidates:
        return choose_non_looping_direction(candidates, planner_choice, recent_positions), False
    if float(sensors.get(model_move, 1.0)) >= 1.0:
        return choose_non_looping_direction(candidates, planner_choice, recent_positions), False

    model_meta = candidates[model_move]
    if model_meta["delta_to_goal"] > 0:
        return choose_non_looping_direction(candidates, model_move, recent_positions), False
    if model_meta["toxicity"] + 0.18 < planned_meta["toxicity"] and model_meta["delta_to_goal"] >= planned_meta["delta_to_goal"] - 1.0:
        return choose_non_looping_direction(candidates, model_move, recent_positions), False
    return choose_non_looping_direction(candidates, planner_choice, recent_positions), False


def _run_sync(coro):
    return asyncio.run(coro)


def main() -> None:
    env: Optional[RedlineEnv] = None
    obs = None
    rewards: List[float] = []
    recent_positions: List[Tuple[int, int]] = []
    steps_taken = 0
    score = 0.0
    success = False
    best_distance = float("inf")
    no_progress_streak = 0
    medium_detour_credit = 2
    initial_distance = 0.0
    error_message: Optional[str] = None
    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME or "unknown")

    if not API_BASE_URL or not API_KEY:
        print("❌ ENV DEBUG:")
        print("API_BASE_URL:", API_BASE_URL)
        print("MODEL_NAME:", MODEL_NAME)
        print("API_KEY exists:", bool(API_KEY))
        error_message = "RuntimeError: Missing required environment variables"
        log_step(step=0, action="ERROR", reward=0.0, done=True, error=error_message)
        log_end(success=False, steps=0, score=0.0, rewards=[])
        return

    try:
        obstacle_mask = load_planner_data()
        client = OpenAI(
            base_url=API_BASE_URL,
            api_key=API_KEY
        )
        env = _run_sync(RedlineEnv.from_docker_image(IMAGE_NAME))
        result = _run_sync(env.reset())
        obs = result.observation

        for step in range(1, MAX_STEPS + 1):
            time.sleep(0.5)
            if result.done:
                break

            current = obs.current_position
            goal = obs.goal_position
            sensors = obs.get_actions
            recent_positions.append(current)

            if step == 1:
                initial_distance = math.dist(current, goal)
            current_distance = math.dist(current, goal)
            if current_distance + 1e-6 < best_distance:
                best_distance = current_distance
                no_progress_streak = 0
                if TASK_NAME == "medium":
                    medium_detour_credit = 2
            else:
                no_progress_streak += 1

            planned_path = astar_path(obstacle_mask, current, goal)
            planner_move, planner_preview, planner_note = planner_next_move(obstacle_mask, planned_path)
            user_prompt = build_user_prompt(step, current, goal, sensors, planner_move, planner_preview, planner_note, recent_positions)
            model_move = request_model_direction(client, user_prompt, step)
            if model_move is None:
                print("[SAFE FALLBACK] Using planner move")
                model_move = planner_move
            final_move, detour_used = resolve_direction(
                current,
                goal,
                sensors,
                planner_move,
                model_move,
                recent_positions,
                no_progress_streak,
                medium_detour_credit,
            )
            if TASK_NAME == "medium" and detour_used:
                medium_detour_credit = 0

            result = _run_sync(env.step(RedlineAction(direction=final_move)))
            obs = result.observation
            reward = result.reward or 0.0
            done = result.done
            rewards.append(reward)
            steps_taken = step

            log_step(step=step, action=final_move, reward=reward, done=done, error=None)

            if done:
                if obs.current_position == obs.goal_position:
                    success = True
                break

        if success:
            score = 1.0
        else:
            final_distance = math.dist(obs.current_position, obs.goal_position)
            if initial_distance == 0:
                score = 1.0
            else:
                score = max(0.0, min(1.0, (initial_distance - final_distance) / initial_distance))
    except Exception as exc:
        error_message = f"{type(exc).__name__}: {exc}"
        print(f"[DEBUG] main error: {error_message}", flush=True)
        if obs is not None:
            try:
                final_distance = math.dist(obs.current_position, obs.goal_position)
                if initial_distance == 0:
                    score = 1.0
                else:
                    score = max(0.0, min(1.0, (initial_distance - final_distance) / initial_distance))
            except Exception as score_exc:
                print(f"[DEBUG] score recovery error: {score_exc}", flush=True)
        if steps_taken == 0:
            log_step(step=0, action="ERROR", reward=0.0, done=True, error=error_message)

    finally:
        try:
            if env is not None:
                _run_sync(env.close())
        except Exception as exc:
            print(f"[DEBUG] env.close() error: {exc}", flush=True)
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        print(f"[DEBUG] fatal error escaped main: {type(exc).__name__}: {exc}", flush=True)
        try:
            log_step(step=0, action="ERROR", reward=0.0, done=True, error=f"{type(exc).__name__}: {exc}")
        except Exception:
            pass
        try:
            log_end(success=False, steps=0, score=0.0, rewards=[])
        except Exception:
            pass
