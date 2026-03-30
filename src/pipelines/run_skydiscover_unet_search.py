from __future__ import annotations

import argparse
import json

from src.skydiscovery.unet_search import run_unet_discovery


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a bounded SkyDiscover search over U-Net emulator overrides.")
    parser.add_argument(
        "--base-config",
        default="config/emulator/unet_thickness_shifting_wind_huv_tau_current_window250_deep_spinup100_convnext.yaml",
        help="Base U-Net config to mutate.",
    )
    parser.add_argument(
        "--source-experiment-id",
        default=None,
        help="Optional override for the generator dataset.",
    )
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
    parser.add_argument(
        "--discovery-epochs",
        type=int,
        default=5,
        help="Training epochs per evaluated candidate.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=4,
        help="SkyDiscover iterations.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional SkyDiscover output directory.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run_unet_discovery(
        env_var_name=args.env_var,
        model_name=args.model,
        base_config_path=args.base_config,
        source_experiment_id=args.source_experiment_id,
        discovery_epochs=args.discovery_epochs,
        iterations=args.iterations,
        output_dir=args.output_dir,
    )
    payload = {
        "best_score": result.best_score,
        "initial_score": result.initial_score,
        "metrics": result.metrics,
        "output_dir": result.output_dir,
    }
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

