from __future__ import annotations

import argparse
import json

from src.pipelines.autonomous_unet_experiment_loop import run_autonomous_unet_experiment_loop


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a policy-driven autonomous loop for config-only U-Net experiments.")
    parser.add_argument(
        "--policy",
        default="config/autoloop/default.yaml",
        help="Path to the autonomous loop policy YAML.",
    )
    parser.add_argument(
        "--batch-id",
        default=None,
        help="Optional explicit batch id. Defaults to a timestamp.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate proposals and configs but do not launch training.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = run_autonomous_unet_experiment_loop(
        policy_path=args.policy,
        dry_run=args.dry_run,
        batch_id=args.batch_id,
    )
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
