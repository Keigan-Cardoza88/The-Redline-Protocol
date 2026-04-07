from openenv.core.env_server.types import Action, Observation, State
from pydantic import Field, BaseModel
from typing import Tuple, Dict, Literal


class RedlineAction(Action):
    direction: Literal["N", "NE", "E", "SE", "S", "SW", "W", "NW"] = Field(
        ..., description="The direction in which the llm moves."
    )

class RedlineObservation(Observation):
    current_position : Tuple[int, int]
    goal_position : Tuple[int, int]
    get_actions : dict[str, float]
    task_description : str
    done: bool = False
    reward: float = 0.0

