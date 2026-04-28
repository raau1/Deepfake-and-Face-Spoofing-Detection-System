"""
Launcher for the FastAPI deepfake-detection web server.

Reads host/port from config/config.yaml (api.host, api.port), then hands off
to uvicorn. Equivalent to running:

    uvicorn src.api.main:app --host <host> --port <port>

but with the advantages that (a) config.yaml is the single source of truth
for deployment settings, and (b) this file can be run directly via
"python scripts/run_api.py" without remembering the uvicorn incantation.
"""

import argparse
from pathlib import Path

import uvicorn
import yaml


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    config_path = project_root / "config" / "config.yaml"

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    api_cfg = config.get("api", {})
    default_host = api_cfg.get("host", "127.0.0.1")
    default_port = int(api_cfg.get("port", 8000))

    parser = argparse.ArgumentParser(description="Run the deepfake detection API.")
    parser.add_argument("--host", default=default_host,
                        help="Host interface to bind (default from config.yaml)")
    parser.add_argument("--port", type=int, default=default_port,
                        help="TCP port to listen on (default from config.yaml)")
    parser.add_argument("--reload", action="store_true",
                        help="Auto-reload on source changes (development only)")
    args = parser.parse_args()

    print(f"Starting Deepfake Detection API on http://{args.host}:{args.port}")
    print(f"Interactive docs: http://{args.host}:{args.port}/docs")

    uvicorn.run(
        "src.api.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
