from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

try:
    from models import RedlineAction, RedlineObservation
except ModuleNotFoundError:
    from .models import RedlineAction, RedlineObservation


class RedlineEnv(
    EnvClient[RedlineAction, RedlineObservation, State]
):
    def _step_payload(self, action: RedlineAction) -> Dict:
        return {
            "direction": action.direction,
        }

    def _parse_result(self, payload: Dict) -> StepResult[RedlineObservation]:
        obs_data = payload.get("observation", {})
        observation = RedlineObservation(
            current_position = tuple(obs_data.get("current_position", (0, 0))),
            goal_position = tuple(obs_data.get("goal_position", (0, 0))),
            get_actions = dict(obs_data.get("get_actions", {})),
            task_description = str(obs_data.get("task_description", "Navigate to the destination safely.")),
            done = payload.get("done", False),
            reward = payload.get("reward", 0.0),
        )

        return StepResult(
            observation = observation,
            reward = payload.get("reward", 0.0),
            done = payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
