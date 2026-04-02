#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
TOPK_ROOT = REPO_ROOT / "outputs" / "topk"
INTERIM_EMULATOR_ROOT = REPO_ROOT / "data" / "interim" / "emulator"
RAW_ROOT = REPO_ROOT / "data" / "raw"


@dataclass(frozen=True)
class SkyIterationRecord:
    iteration: int
    program_id: str
    eval_mse_mean: float
    eval_mse_last: float
    train_loss: float
    metrics_path: str
    rollout_path: str
    checkpoint_path: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Show status for recent SkyDiscover and training runs.")
    parser.add_argument("--sky-log", help="Optional explicit SkyDiscover log path.")
    parser.add_argument("--training-id", help="Optional explicit training experiment id.")
    parser.add_argument(
        "--only",
        choices=("sky", "training", "all"),
        default="all",
        help="Restrict output to one kind of status.",
    )
    return parser.parse_args()


def latest_sky_log(topk_root: Path = TOPK_ROOT) -> Path | None:
    logs = sorted(topk_root.glob("**/logs/topk_*.log"), key=lambda path: path.stat().st_mtime, reverse=True)
    return logs[0] if logs else None


def parse_sky_iterations(log_text: str) -> list[SkyIterationRecord]:
    pattern = re.compile(
        r"Iteration (?P<iteration>\d+): Program (?P<program>[0-9a-f-]+).*?"
        r"Metrics: .*?eval_mse_mean=(?P<mean>[0-9.]+), "
        r"eval_mse_last=(?P<last>[0-9.]+), train_loss=(?P<loss>[0-9.]+), metrics_path=(?P<metrics>[^,]+), "
        r"rollout_path=(?P<rollout>[^,]+), checkpoint_path=(?P<checkpoint>[^\n]+)",
        re.DOTALL,
    )
    records: list[SkyIterationRecord] = []
    for match in pattern.finditer(log_text):
        records.append(
            SkyIterationRecord(
                iteration=int(match.group("iteration")),
                program_id=match.group("program"),
                eval_mse_mean=float(match.group("mean")),
                eval_mse_last=float(match.group("last")),
                train_loss=float(match.group("loss")),
                metrics_path=match.group("metrics"),
                rollout_path=match.group("rollout"),
                checkpoint_path=match.group("checkpoint"),
            )
        )
    return records


def sky_status(log_path: Path) -> str:
    log_text = log_path.read_text(encoding="utf-8")
    run_dir = log_path.parent.parent
    best_info_path = run_dir / "best" / "best_program_info.json"
    iterations = parse_sky_iterations(log_text)
    completed = "✅ Discovery process completed" in log_text

    lines = ["SkyDiscover"]
    lines.append(f"  run_dir: {run_dir}")
    lines.append(f"  log: {log_path}")
    lines.append(f"  status: {'completed' if completed else 'running'}")
    if iterations:
        last = max(iterations, key=lambda record: record.iteration)
        best = min(iterations, key=lambda record: record.eval_mse_mean)
        lines.append(
            f"  latest_completed: iter {last.iteration} mean={last.eval_mse_mean:.4f} last={last.eval_mse_last:.4f}"
        )
        lines.append(
            f"  best_seen: iter {best.iteration} mean={best.eval_mse_mean:.4f} last={best.eval_mse_last:.4f}"
        )
    else:
        lines.append("  latest_completed: none")

    evaluating_matches = re.findall(r"Evaluating program ([0-9a-f-]+)", log_text)
    if not completed and evaluating_matches:
        lines.append(f"  current_program: {evaluating_matches[-1]}")

    if best_info_path.exists():
        payload = json.loads(best_info_path.read_text(encoding="utf-8"))
        metrics = payload.get("metrics", {})
        lines.append(f"  best_program_id: {payload.get('id')}")
        lines.append(
            f"  best_metrics: mean={float(metrics.get('eval_mse_mean', 0.0)):.4f} "
            f"last={float(metrics.get('eval_mse_last', 0.0)):.4f}"
        )
        lines.append(f"  best_metrics_path: {metrics.get('metrics_path')}")
    return "\n".join(lines)


def latest_training_record(interim_root: Path = INTERIM_EMULATOR_ROOT, raw_root: Path = RAW_ROOT) -> tuple[Path, Path | None] | None:
    candidates: list[tuple[float, Path, Path | None]] = []
    for history_path in interim_root.glob("**/training_history.json"):
        experiment_dir = history_path.parent
        experiment_id = experiment_dir.name
        metrics_matches = sorted(raw_root.glob(f"*/emulator/*/{experiment_id}/metrics.json"))
        metrics_path = metrics_matches[0] if metrics_matches else None
        updated_at = max(
            path.stat().st_mtime
            for path in (history_path, metrics_path)
            if path is not None and path.exists()
        )
        candidates.append((updated_at, history_path, metrics_path))
    if not candidates:
        return None
    _, history_path, metrics_path = max(candidates, key=lambda item: item[0])
    return history_path, metrics_path


def find_training_record(experiment_id: str, interim_root: Path = INTERIM_EMULATOR_ROOT, raw_root: Path = RAW_ROOT) -> tuple[Path, Path | None] | None:
    matches = list(interim_root.glob(f"*/{experiment_id}/training_history.json"))
    if not matches:
        return None
    history_path = matches[0]
    metrics_matches = sorted(raw_root.glob(f"*/emulator/*/{experiment_id}/metrics.json"))
    return history_path, (metrics_matches[0] if metrics_matches else None)


def training_status(history_path: Path, metrics_path: Path | None) -> str:
    history = json.loads(history_path.read_text(encoding="utf-8"))
    lines = ["Training"]
    lines.append(f"  experiment_id: {history.get('experiment_id', history_path.parent.name)}")
    lines.append(f"  history: {history_path}")
    lines.append(f"  status: {history.get('status', 'unknown')}")
    lines.append(
        f"  epochs: {int(history.get('epochs_completed', 0))} / {int(history.get('epochs_total', 0))}"
    )
    losses = history.get("epoch_train_losses", [])
    if losses:
        lines.append(f"  latest_epoch_loss: {float(losses[-1]):.6f}")
    if metrics_path is not None and metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        lines.append(f"  metrics: {metrics_path}")
        lines.append(
            f"  eval: mean={float(metrics.get('eval_mse_mean', 0.0)):.4f} "
            f"last={float(metrics.get('eval_mse_max', 0.0)):.4f}"
        )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    sections: list[str] = []

    if args.only in {"sky", "all"}:
        log_path = Path(args.sky_log) if args.sky_log else latest_sky_log()
        if log_path is None or not log_path.exists():
            sections.append("SkyDiscover\n  no runs found")
        else:
            sections.append(sky_status(log_path))

    if args.only in {"training", "all"}:
        training_record = (
            find_training_record(args.training_id) if args.training_id else latest_training_record()
        )
        if training_record is None:
            sections.append("Training\n  no runs found")
        else:
            history_path, metrics_path = training_record
            sections.append(training_status(history_path, metrics_path))

    print("\n\n".join(sections))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
