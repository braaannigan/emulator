from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


AUTOLOOP_ROOT = Path("data/interim/autoloop/unet")


@dataclass(frozen=True)
class AutonomousBatchRecord:
    batch_id: str
    phase: str
    updated_at: str
    llm_calls_used: int
    patch_calls_used: int
    candidate_count: int
    completed_count: int
    screened_out_count: int
    best_eval_mse_mean: float | None
    failure_reason: str | None
    ledger_path: Path


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected object in {path}")
    return payload


def list_autonomous_batches(root: Path = AUTOLOOP_ROOT) -> list[AutonomousBatchRecord]:
    if not root.exists():
        return []
    records: list[AutonomousBatchRecord] = []
    for batch_dir in sorted((path for path in root.iterdir() if path.is_dir() and path.name != "memory"), reverse=True):
        ledger_path = batch_dir / "ledger.json"
        if not ledger_path.exists():
            continue
        payload = _load_json(ledger_path)
        candidates = payload.get("candidates", [])
        if not isinstance(candidates, list):
            candidates = []
        completed_metrics = [
            float(candidate["eval_mse_mean"])
            for candidate in candidates
            if isinstance(candidate, dict) and candidate.get("eval_mse_mean") is not None
        ]
        records.append(
            AutonomousBatchRecord(
                batch_id=str(payload.get("batch_id", batch_dir.name)),
                phase=str(payload.get("phase", "unknown")),
                updated_at=str(payload.get("updated_at", "")),
                llm_calls_used=int(payload.get("llm_calls_used", 0)),
                patch_calls_used=int(payload.get("patch_calls_used", 0)),
                candidate_count=len(candidates),
                completed_count=sum(
                    1 for candidate in candidates if isinstance(candidate, dict) and candidate.get("status") == "completed"
                ),
                screened_out_count=len(payload.get("screened_out", [])) if isinstance(payload.get("screened_out", []), list) else 0,
                best_eval_mse_mean=min(completed_metrics) if completed_metrics else None,
                failure_reason=None if payload.get("failure_reason") is None else str(payload.get("failure_reason")),
                ledger_path=ledger_path,
            )
        )
    return records


def load_autonomous_batch_details(ledger_path: Path) -> dict[str, Any]:
    return _load_json(ledger_path)
