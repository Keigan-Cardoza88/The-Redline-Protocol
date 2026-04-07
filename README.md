---
title: Redline Env
emoji: 🚦
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# Redline Env

`redline_env` is an OpenEnv environment for pollution-aware urban navigation.

The agent operates on a real road mask derived from Panaji map data and must move from an origin to a destination while balancing route efficiency against air-quality exposure. This is meant to model a real-world decision problem: planning navigation under environmental risk, not solving a toy maze.

## Why This Environment Exists

Many navigation systems optimize only for shortest path or travel time. In the real world, that is often not enough. A safer route may be slightly longer but meaningfully cleaner. `redline_env` is designed to evaluate whether an agent can make those tradeoffs under different task objectives.

The environment currently uses a single final map:

- `Panaji`

## Tasks

This submission uses one environment with three task modes. That is intentional.

The task is selected through configuration, not by editing code. Each mode changes the objective and reward tradeoff while keeping the same action and observation interface.

### `easy`

Reach the goal on a relatively short route while keeping pollution exposure low when practical.

- Short path matters
- Low exposure matters
- Good first benchmark for basic competence

### `medium`

Reach the goal with the least exposure possible, even if a significantly longer route is required.

- Clean air matters most
- Long detours are acceptable
- Designed to reward strong pollution avoidance

### `hard`

Reach the goal while local pollution changes over time and by location, forcing the agent to react to moving high-exposure bands instead of following a static route.

- Progress still matters
- Low exposure still matters
- Local toxicity can change by both time and position

## Action Space

The agent emits one of 8 typed movement directions:

- `N`
- `NE`
- `E`
- `SE`
- `S`
- `SW`
- `W`
- `NW`

The action model is implemented in [models.py](./models.py).

## Observation Space

Each observation contains:

- `current_position`: current `(x, y)` grid coordinate
- `goal_position`: target `(x, y)` grid coordinate
- `get_actions`: local toxicity lookup for all 8 directions
- `task_description`: task text for the current mode
- `done`: whether the episode has terminated
- `reward`: reward from the latest transition

The observation model is implemented in [models.py](./models.py).

## Reward Design

The reward is shaped across the trajectory instead of being purely terminal.

Common elements:

- progress reward for moving closer to the goal
- penalty for wandering
- toxicity penalty based on local pollution
- large penalty for invalid wall hits
- large success bonus on reaching the destination

Task-specific behavior:

- `easy`: balances short route and pollution exposure
- `medium`: strongly penalizes exposure and tolerates longer routes
- `hard`: combines path pressure with time-varying toxicity

## Environment API

The server implements the standard OpenEnv lifecycle:

- `reset()`
- `step(action)`
- `state`

The server implementation lives in [server/redline_env_environment.py](./server/redline_env_environment.py).

## Supported Selectors

Judges and users should not need to edit source files to switch tasks.

### Task selector

The baseline inference script selects the task with the environment variable:

```powershell
$env:REDLINE_ENV_V4_TASK = "easy"
```

Valid values:

- `easy`
- `medium`
- `hard`

It is best to set this explicitly every run, even if you are using defaults elsewhere.

For direct server or client calls, `reset(...)` also accepts `task="easy" | "medium" | "hard"`.

### Pollutant selector

Pollutants are selected through `reset(...)`, not by editing code.

Current supported pollutant channels:

- `NO2`
- `PM25`
- `SO2`
- `CO`

Pass pollutants as a list, for example:

```python
obs = env.reset(pollutant=["NO2"])
```

or

```python
obs = env.reset(pollutant=["NO2", "PM25"])
```

Use a list, not a single string.

If no pollutant list is provided, the current default behavior uses `["NO2"]`.

## Web UI

When the container runs, the Gradio web interface is enabled at `/web`.

The default OpenEnv Playground tab remains available. A second custom tab adds a judge-facing comparison surface for the final Panaji map with:

- task selector for `easy`, `medium`, `hard`
- pollutant selector
- side-by-side route comparison for raw `A*` and the model-guided baseline
- OpenStreetMap route overlay
- comparison metrics even when a run does not finish

Current custom metrics:

- success or failure
- steps
- score
- total reward
- cumulative exposure
- average exposure
- final distance to goal

If `HF_TOKEN` or `API_KEY` is not available at runtime, the custom tab still renders and falls back to planner-guided behavior for the model lane instead of crashing.

## Baseline Inference

The baseline agent is implemented in [inference.py](./inference.py).

It uses:

- an LLM through the OpenAI client
- local toxicity readings from the environment
- an A* planner for road-safe destination guidance
- deterministic post-processing to avoid obviously bad or looping moves

The script emits structured logs in the required format:

- `[START]`
- `[STEP]`
- `[END]`

This is intended to support reproducible evaluation across `easy`, `medium`, and `hard`.

### Latest local baseline

Latest local runs on Panaji with the current baseline:

- `easy`: `success=true`, about `92` steps, score `1.000`
- `medium`: `success=true`, about `95` steps, score `1.000`
- `hard`: `success=true`, about `93` steps, score `1.000`

These should be treated as baseline reference points for the current repo state, not as theoretical maxima.

## Required Environment Variables

Set these before running the baseline:

```powershell
$env:HF_TOKEN = "<your-token>"
$env:API_BASE_URL = "https://router.huggingface.co/v1"
$env:MODEL_NAME = "deepseek-ai/DeepSeek-V3-0324"
$env:REDLINE_ENV_V4_TASK = "easy"
```

Optional:

```powershell
$env:IMAGE_NAME = "redline_env:latest"
$env:REDLINE_ENV_V4_MAX_STEPS = "120"
```

Notes:

- `HF_TOKEN` is read by the inference script as the API key input.
- `REDLINE_ENV_V4_TASK` should be set explicitly for every run.
- `IMAGE_NAME` can be used to override the Docker image used by `inference.py`.
- `REDLINE_ENV_V4_MAX_STEPS` can be used to override the default step budget for a task.

## Local Development

Install the environment:

```powershell
pip install .
```

Run the server locally:

```powershell
$env:ENABLE_WEB_INTERFACE = "true"
python -m server.app
```

or:

```powershell
$env:ENABLE_WEB_INTERFACE = "true"
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

Then open:

```text
http://localhost:8000/web
```

## Docker

Build the image from this directory:

```powershell
docker build -t redline_env:latest .
```

Run it:

```powershell
docker run --rm -p 8000:8000 redline_env:latest
```

The image now enables the Gradio interface by default, so `/web` is available without extra flags.

## Example Usage

### Connect to a running server

```python
from client import RedlineEnv
from models import RedlineAction

env = RedlineEnv(base_url="http://localhost:8000")
result = env.reset(pollutant=["NO2"])
result = env.step(RedlineAction(direction="E"))
print(result.observation.current_position)
```

### Run the baseline with a selected task

```powershell
$env:HF_TOKEN = "<your-token>"
$env:API_BASE_URL = "https://router.huggingface.co/v1"
$env:MODEL_NAME = "deepseek-ai/DeepSeek-V3-0324"
$env:REDLINE_ENV_V4_TASK = "medium"
python inference.py
```

## Hugging Face Space Deployment

Push from this directory with the OpenEnv CLI:

```powershell
openenv push --repo-id <hf-username>/<space-name> --enable-interface
```

Recommended Space secrets:

- `HF_TOKEN`
- `API_BASE_URL`
- `MODEL_NAME`

Recommended values:

- `API_BASE_URL=https://router.huggingface.co/v1`
- `MODEL_NAME=deepseek-ai/DeepSeek-V3-0324`

After deployment, verify:

1. `<space-url>/reset` returns HTTP `200`
2. `<space-url>/web` loads the Playground and Custom tabs
3. the Custom tab completes an `A*` vs model comparison run

## OpenEnv Manifest

The environment manifest is defined in [openenv.yaml](./openenv.yaml).

Current runtime configuration:

- runtime: `fastapi`
- app: `server.app:app`
- port: `8000`

## Project Structure

```text
redline_env/
|- __init__.py
|- client.py
|- inference.py
|- models.py
|- openenv.yaml
|- pyproject.toml
|- README.md
|- Dockerfile
|- Panaji_ULTRA.csv
|- Panaji_obstacles.npy
`- server/
   |- __init__.py
   |- app.py
   |- gradio_builder.py
   `- redline_env_environment.py
```

## Submission Notes

For Round 1, this project should be understood as:

- one real-world OpenEnv environment
- three task modes selected through supported configuration
- typed models and standard environment API
- reproducible baseline inference script
- deterministic partial-credit scoring in `[0, 1]`

The intended evaluation story is:

- `easy`: basic competence under route and pollution tradeoffs
- `medium`: stronger pollution-first reasoning
- `hard`: dynamic conditions that are harder to exploit with static heuristics
