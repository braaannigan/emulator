from __future__ import annotations

import argparse
import json

from src.skydiscovery.unet_search import run_openrouter_smoke_test


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke test SkyDiscover against OpenRouter.")
    parser.add_argument(
        "--env-var",
        default="OPENROUTER_GEMMA3_27B_IT_API_KEY",
        help="Environment variable containing the OpenRouter API key.",
    )
    parser.add_argument(
        "--model",
        default="google/gemma-3-27b-it",
        help="OpenRouter model name.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run_openrouter_smoke_test(args.env_var, args.model)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
