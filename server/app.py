try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:
    raise ImportError(
        "openenv is required for the web interface. Install dependencies with '\n    uv sync\n'"
    ) from e

try:
    from models import RedlineAction, RedlineObservation
    from server.gradio_builder import build_redline_gradio
    from server.redline_env_environment import RedlineEnvironment
except ModuleNotFoundError:
    from redline_env.models import RedlineAction, RedlineObservation
    from redline_env.server.gradio_builder import build_redline_gradio
    from redline_env.server.redline_env_environment import RedlineEnvironment


app = create_app(
    RedlineEnvironment,
    RedlineAction,
    RedlineObservation,
    env_name="redline_env",
    max_concurrent_envs=3,
    gradio_builder=build_redline_gradio,
)


def main(host: str = "0.0.0.0", port: int = 8000):
    import uvicorn
    import os

    uvicorn.run(
        app,
        host=host,
        port=port,
        ws_ping_interval=float(os.getenv("REDLINE_SERVER_WS_PING_INTERVAL_S", "20")),
        ws_ping_timeout=float(os.getenv("REDLINE_SERVER_WS_PING_TIMEOUT_S", "120")),
        timeout_keep_alive=int(os.getenv("REDLINE_SERVER_KEEPALIVE_S", "120")),
    )


if __name__ == '__main__':
    main()
