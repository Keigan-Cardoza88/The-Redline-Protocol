import math
import os
from collections import deque
import numpy as np
import pandas as pd
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import RedlineAction, RedlineObservation
except ImportError:
    from models import RedlineAction, RedlineObservation

class RedlineEnvironment(Environment):
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self.city_dict = {}
        self.origin = (0.0, 0.0)
        self.destination = (0.0, 0.0)
        self.task_difficulty = 0
        self.region = "Panaji"
        self.hard_phase_grid = None
        self.hard_amplitude_grid = None
        self.hard_axis_grid = None
        self.hard_cross_grid = None

    def _build_toxicity_grid(self, selected_map, pollutant=None):
        pollutant_list = pollutant if pollutant is not None else ["NO2"]
        mti_grid = np.zeros((selected_map["height"], selected_map["width"]))
        pollutant_weight_dict = {"PM25": 3.0, "NO2": 2.0, "SO2": 1.5, "CO": 1.0}

        for poll in pollutant_list:
            if poll in selected_map and poll in pollutant_weight_dict:
                mti_grid += selected_map[poll] * pollutant_weight_dict[poll]

        m_min, m_max = np.min(mti_grid), np.max(mti_grid)
        normalized = (mti_grid - m_min) / (m_max - m_min) if m_max > m_min else np.zeros_like(mti_grid)
        return pollutant_list, normalized

    def _prepare_hard_dynamics(self):
        y_idx, x_idx = np.indices(self.active_mti_grid.shape, dtype=float)
        self.hard_phase_grid = (0.17 * x_idx) + (0.23 * y_idx) + (self.active_mti_grid * math.pi * 2.10)
        self.hard_amplitude_grid = 0.22 + (self.active_mti_grid * 0.32)
        self.hard_axis_grid = (0.90 * x_idx) + (1.10 * y_idx)
        self.hard_cross_grid = x_idx - y_idx

    def _compute_road_distances(self, start):
        distance_grid = np.full((self.height, self.width), -1, dtype=np.int32)
        sx, sy = int(start[0]), int(start[1])

        if sx < 0 or sx >= self.width or sy < 0 or sy >= self.height:
            return distance_grid
        if not self.active_obstacle_mask[sy, sx]:
            return distance_grid

        queue = deque([(sx, sy)])
        distance_grid[sy, sx] = 0

        while queue:
            cx, cy = queue.popleft()
            next_distance = distance_grid[cy, cx] + 1

            for dx, dy in ((0, -1), (1, 0), (0, 1), (-1, 0)):
                nx, ny = cx + dx, cy + dy
                if nx < 0 or nx >= self.width or ny < 0 or ny >= self.height:
                    continue
                if not self.active_obstacle_mask[ny, nx]:
                    continue
                if distance_grid[ny, nx] != -1:
                    continue

                distance_grid[ny, nx] = next_distance
                queue.append((nx, ny))

        return distance_grid

    def _get_cell_toxicity(self, x, y):
        base_toxicity = float(self.active_mti_grid[y, x])
        if getattr(self, "task_level", "") != "hard":
            return base_toxicity

        step = float(self._state.step_count)
        phase = float(self.hard_phase_grid[y, x])
        amplitude = float(self.hard_amplitude_grid[y, x])
        axis_value = float(self.hard_axis_grid[y, x])
        cross_value = float(self.hard_cross_grid[y, x])

        primary_wave = math.sin((step / 4.5) + phase)
        secondary_wave = math.cos((step / 8.0) - (phase * 0.55))
        tertiary_wave = math.sin((step / 2.8) + (phase * 1.35))
        band_center = math.fmod(step * 2.35, max(1.0, float(self.width + self.height)))
        moving_band = math.exp(-(((axis_value - band_center) ** 2) / 85.0))
        cross_center = math.sin(step / 6.0) * 18.0
        cross_band = math.exp(-(((cross_value - cross_center) ** 2) / 140.0))

        dynamic_toxicity = (
            base_toxicity
            + (base_toxicity * amplitude * primary_wave)
            + (0.08 * secondary_wave)
            + (0.05 * tertiary_wave)
            + (0.16 * moving_band)
            + (0.08 * cross_band)
        )
        return min(1.0, max(0.0, dynamic_toxicity))

    def _choose_destination(self, valid_roads):
        ox, oy = self.origin
        origin_arr = np.array([oy, ox], dtype=float)
        road_points = valid_roads.astype(float)
        deltas = road_points - origin_arr
        distances = np.sqrt((deltas[:, 0] ** 2) + (deltas[:, 1] ** 2))
        toxicities = self.active_mti_grid[valid_roads[:, 0], valid_roads[:, 1]]

        valid_mask = distances > 0
        if not np.any(valid_mask):
            return self.origin

        if self.task_level == "easy":
            candidate_mask = valid_mask & (distances >= 6) & (distances <= 18) & (toxicities < 0.55)
            if not np.any(candidate_mask):
                candidate_mask = valid_mask & (distances >= 4) & (distances <= 24)
            scores = distances + (toxicities * 10.0)
        elif self.task_level == "medium":
            candidate_mask = valid_mask & (distances >= 45) & (toxicities < 0.45)
            if not np.any(candidate_mask):
                candidate_mask = valid_mask & (distances >= 32) & (toxicities < 0.60)
            if not np.any(candidate_mask):
                candidate_mask = valid_mask & (distances >= 24)
            scores = (toxicities * 120.0) + (distances * 0.05)
        else:
            graph_distances = self._compute_road_distances(self.origin)[valid_roads[:, 0], valid_roads[:, 1]].astype(float)
            volatility = self.hard_amplitude_grid[valid_roads[:, 0], valid_roads[:, 1]]
            reachable_mask = graph_distances > 0
            hard_target_distance = 160.0
            candidate_mask = reachable_mask & (graph_distances >= 145) & (graph_distances <= 185) & (toxicities < 0.72)
            if not np.any(candidate_mask):
                candidate_mask = reachable_mask & (graph_distances >= 130) & (graph_distances <= 205) & (toxicities < 0.82)
            if not np.any(candidate_mask):
                candidate_mask = reachable_mask & (graph_distances >= 115)
            scores = np.abs(graph_distances - hard_target_distance) + (toxicities * 26.0) - (volatility * 10.0)

        candidate_indices = np.where(candidate_mask)[0]
        if len(candidate_indices) == 0:
            candidate_indices = np.where(valid_mask)[0]

        best_index = int(candidate_indices[np.argmin(scores[candidate_indices])])
        gy, gx = valid_roads[best_index]
        return (int(gx), int(gy))

    def load_city_data(self, region):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = os.path.dirname(current_dir)

        csv_file = os.path.join(root_dir, f"{region}_ULTRA.csv")
        if not os.path.exists(csv_file):
            csv_file = os.path.join(current_dir, f"{region}_ULTRA.csv")
            
        npy_file = os.path.join(root_dir, f"{region}_obstacles.npy")
        if not os.path.exists(npy_file):
            npy_file = os.path.join(current_dir, f"{region}_obstacles.npy")

        try:
            print(f"[LOADING] Fetching {region} 30m data into memory...")
            df = pd.read_csv(csv_file)

            height = df['latitude'].nunique()
            width = df['longitude'].nunique()
            grid_area = height * width
            
            obstacle_mask = np.load(npy_file).flatten()[:grid_area].reshape(height, width)
            
            self.city_dict[region] = {
                "height": height,
                "width": width,
                "NO2": df["NO2_norm"].values[:grid_area].reshape(height, width),
                "SO2": df["SO2_norm"].values[:grid_area].reshape(height, width),
                "PM25": df["PM25_norm"].values[:grid_area].reshape(height, width),
                "CO": df["CO_norm"].values[:grid_area].reshape(height, width),
                "is_road": obstacle_mask
            }
            print(f"[SUCCESS] {region} loaded ({height}x{width} grid).")
            
        except Exception as e:
            print(f"\n[CRITICAL ERROR] Failed to load {region}!")
            print(f"Error details: {str(e)}\n")
            raise e 

    def reset(self, origin=None, destination=None, region=None, pollutant=None, task=None) -> RedlineObservation:
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self.reward = 0.0
        self.region = "Panaji"
        
        if self.region not in self.city_dict:
            self.load_city_data(self.region)

        selected_map = self.city_dict[self.region]
        self.active_obstacle_mask = selected_map["is_road"]
        self.height = selected_map["height"]
        self.width = selected_map["width"]
        self.task_level = str(task or os.getenv("REDLINE_ENV_V4_TASK", "medium")).lower()
        self.pollutant_list, self.active_mti_grid = self._build_toxicity_grid(selected_map, pollutant)
        self._prepare_hard_dynamics()

        if origin is None or destination is None:
            valid_roads = np.argwhere(self.active_obstacle_mask == True)
            
            if len(valid_roads) == 0:
                print("[ERROR] No road found in mask!")
                self.origin, self.destination = (0, 0), (self.width-1, self.height-1)
            else:
                if origin is None:
                    y, x = valid_roads[len(valid_roads) // 10]
                    self.origin = (int(x), int(y))
                else: self.origin = origin

                if destination is None:
                    self.destination = self._choose_destination(valid_roads)
        else:
            self.origin, self.destination = origin, destination

        return RedlineObservation(
            current_position=self.origin,
            goal_position=self.destination,
            get_actions=self._get_actions(self.origin),
            task_description=f"Navigate Panaji safely. Mode: {self.task_level}"
        )

    def step(self, action: RedlineAction) -> RedlineObservation:
        self._state.step_count += 1
        x, y = int(self.origin[0]), int(self.origin[1])
        dist_before = math.sqrt((x - self.destination[0])**2 + (y - self.destination[1])**2)
        direction_moved = action.direction.upper()
        
        direction_dict = {
            "N": (0, -1), "NE": (1, -1), "E": (1, 0), "SE": (1, 1),
            "S": (0, 1), "SW": (-1, 1), "W": (-1, 0), "NW": (-1, -1),
        }
        dx, dy = direction_dict.get(direction_moved, (0, 0))
        new_x = x + dx
        new_y = y + dy
        
        done = False
        hit_wall = False

        # wall and corner squeeze logic
        def is_wall(cx, cy):
            if cx < 0 or cx >= self.width or cy < 0 or cy >= self.height:
                return True
            return not self.active_obstacle_mask[cy, cx]

        if is_wall(new_x, new_y):
            hit_wall = True
        else:
            is_diagonal = (dx != 0 and dy != 0)
            if is_diagonal:
                if is_wall(new_x, y) and is_wall(x, new_y):
                    hit_wall = True

        if hit_wall:
            step_reward = -20.0
        else:
            self.origin = (new_x, new_y)
            base_toxicity = self._get_cell_toxicity(new_x, new_y)
            dist_after = math.sqrt((new_x - self.destination[0])**2 + (new_y - self.destination[1])**2)
            progress_reward = 0.8 if dist_after < dist_before else -0.1
            
            if self.task_level == "easy":
                wandering_penalty = -0.1
                toxicity_penalty = base_toxicity * 2.0
            elif self.task_level == "medium":
                wandering_penalty = -0.01 
                toxicity_penalty = base_toxicity * 15.0 
            elif self.task_level == "hard":
                wandering_penalty = -0.3
                toxicity_penalty = base_toxicity * 9.5
            else:
                wandering_penalty = -0.1
                toxicity_penalty = base_toxicity

            step_reward = wandering_penalty - toxicity_penalty + progress_reward
            
        if self.origin == self.destination:
            done = True
            step_reward += 100
        self.reward += step_reward

        return RedlineObservation(
            current_position = self.origin,
            goal_position = self.destination,
            get_actions = self._get_actions(self.origin), 
            task_description = f"Navigate Panaji safely. Mode: {self.task_level}",
            done = done,
            reward = step_reward
        )

    def _get_actions(self, origin):
        x, y = int(origin[0]), int(origin[1])

        def is_wall(cx, cy):
            if cx < 0 or cx >= self.width or cy < 0 or cy >= self.height:
                return True
            return not self.active_obstacle_mask[cy, cx]

        def safe_get(dy, dx):
            sy = y + dy
            sx = x + dx
            
            if is_wall(sx, sy):
                return 1.0 
            
            if dx != 0 and dy != 0:
                if is_wall(sx, y) and is_wall(x, sy):
                    return 1.0  

            return self._get_cell_toxicity(sx, sy)

        self.actions_dict = {
            "N":  safe_get(-1, 0), "NE": safe_get(-1, 1), "E":  safe_get(0, 1), 
            "SE": safe_get(1, 1),  "S":  safe_get(1, 0),  "SW": safe_get(1, -1),
            "W":  safe_get(0, -1), "NW": safe_get(-1, -1),
        }
        return self.actions_dict

    @property
    def state(self) -> State:
        return self._state
