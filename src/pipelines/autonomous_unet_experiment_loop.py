from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import re
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import yaml

from openai import OpenAI

from src.generator.config import timestamp_experiment_id
from src.models.unet_thickness.config import load_unet_thickness_config
from src.skydiscovery.unet_search import (
    OPENROUTER_API_BASE,
    configure_openrouter,
    validate_candidate_overrides,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _slugify(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return normalized or "candidate"


@dataclass(frozen=True)
class LLMPolicy:
    env_var_name: str
    model_name: str
    api_base: str = OPENROUTER_API_BASE
    temperature: float = 0.3
    max_tokens: int = 4000
    timeout_seconds: int = 180
    max_total_calls: int = 2


@dataclass(frozen=True)
class BatchPolicy:
    max_hypotheses: int
    max_total_train_runs: int
    max_total_llm_calls: int
    max_total_patch_calls: int
    sequential: bool
    allow_code_patches: bool
    stop_after_generation: bool = False


@dataclass(frozen=True)
class PatchingPolicy:
    max_patch_attempts_per_hypothesis: int
    max_repair_attempts_per_patch: int


@dataclass(frozen=True)
class TrainingPolicy:
    require_benchmark_path: bool
    require_preflight_tests: bool
    preflight_test_paths: tuple[str, ...]


@dataclass(frozen=True)
class PromotionPolicy:
    primary_metric: str
    maximize: bool
    must_beat_incumbent_by: float


@dataclass(frozen=True)
class EvaluationPolicy:
    competitive_within_ratio: float = 0.2
    max_total_calls: int = 2
    final_step_heatmap_filename: str = "final_step_heatmap.png"
    evaluator_payload_filename: str = "evaluator_payload.json"
    artifact_guidance: str = (
        "Prioritize suppression of boundary reflections and other numerical artifacts. "
        "A candidate that is within the competitive band can be preferred even if its eval MSE is modestly worse, "
        "provided the final-step rollout heatmap shows meaningfully cleaner dynamics."
    )


@dataclass(frozen=True)
class SearchPolicy:
    round_modes: tuple[str, ...] = ("explore", "exploit", "artifact_hardening", "exploit")
    explore_max_per_family: int = 2
    promising_within_ratio: float = 0.35
    max_optimizer_only_proposals_per_explore_round: int = 1


@dataclass(frozen=True)
class AutonomousLoopPolicy:
    base_config_path: Path
    experiment_family: str
    experiment_log_path: Path
    experiment_compact_path: Path
    legacy_experiment_markdown_path: Path | None
    python_executable: str
    output_root: Path
    llm: LLMPolicy
    batch: BatchPolicy
    patching: PatchingPolicy
    training: TrainingPolicy
    promotion: PromotionPolicy
    evaluation: EvaluationPolicy
    search: SearchPolicy


@dataclass(frozen=True)
class HypothesisProposal:
    name: str
    hypothesis: str
    implementation_type: str = "config"
    search_mode: str = "exploit"
    hypothesis_family: str = "optimizer_tuning"
    overrides: dict[str, Any] = field(default_factory=dict)
    patch_targets: list[str] = field(default_factory=list)
    patch_plan: str | None = None


@dataclass
class CandidateRunRecord:
    name: str
    hypothesis: str
    implementation_type: str
    search_mode: str
    hypothesis_family: str
    config_path: str
    experiment_id: str
    overrides: dict[str, Any]
    status: str
    patch_targets: list[str] = field(default_factory=list)
    patch_plan: str | None = None
    workspace_path: str | None = None
    patch_path: str | None = None
    branch_name: str | None = None
    branch_commit: str | None = None
    llm_calls_used: int = 0
    patch_calls_used: int = 0
    ranking_score: float | None = None
    started_at: str | None = None
    finished_at: str | None = None
    return_code: int | None = None
    duration_seconds: float | None = None
    metrics_path: str | None = None
    training_history_path: str | None = None
    eval_mse_mean: float | None = None
    stop_reason: str | None = None
    result_class: str | None = None
    is_competitive: bool = False
    final_step_heatmap_path: str | None = None
    evaluator_payload_path: str | None = None
    evaluator_summary_path: str | None = None
    artifact_severity: int | None = None
    artifact_tags: list[str] = field(default_factory=list)
    evaluator_notes: str | None = None
    accept_tradeoff_for_cleaner_rollout: bool | None = None
    stdout_tail: str | None = None
    stderr_tail: str | None = None


@dataclass
class BatchLedger:
    batch_id: str
    created_at: str
    updated_at: str
    policy_path: str
    base_config_path: str
    experiment_family: str
    phase: str
    llm_calls_used: int = 0
    evaluator_calls_used: int = 0
    patch_calls_used: int = 0
    proposals: list[dict[str, Any]] = field(default_factory=list)
    screened_out: list[dict[str, Any]] = field(default_factory=list)
    candidates: list[CandidateRunRecord] = field(default_factory=list)
    stage_events: list[dict[str, Any]] = field(default_factory=list)
    preflight: dict[str, Any] = field(default_factory=dict)
    summary_path: str | None = None
    failure_reason: str | None = None
    failure_details: str | None = None


@dataclass
class FamilyMemory:
    experiment_family: str
    updated_at: str
    tried_dedupe_keys: list[list[list[str]]] = field(default_factory=list)
    banned_override_pairs: list[list[str]] = field(default_factory=list)
    completed_runs: list[dict[str, Any]] = field(default_factory=list)


def append_stage_event(
    ledger: BatchLedger,
    *,
    category: str,
    message: str,
    experiment_id: str | None = None,
    phase: str | None = None,
    details: dict[str, Any] | None = None,
) -> None:
    ledger.stage_events.append(
        {
            "timestamp": _utcnow(),
            "category": category,
            "phase": ledger.phase if phase is None else phase,
            "experiment_id": experiment_id,
            "message": message,
            "details": {} if details is None else details,
        }
    )


def load_autonomous_unet_policy(path: str | Path) -> AutonomousLoopPolicy:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Autoloop policy must contain a top-level mapping.")

    def _resolve(value: str | Path) -> Path:
        raw = Path(value)
        return raw if raw.is_absolute() else (REPO_ROOT / raw)

    python_executable = str(_resolve(payload.get("python_executable", ".venv/bin/python")))

    return AutonomousLoopPolicy(
        base_config_path=_resolve(payload["base_config_path"]),
        experiment_family=str(payload["experiment_family"]),
        experiment_log_path=_resolve(payload["experiment_log_path"]),
        experiment_compact_path=_resolve(payload["experiment_compact_path"]),
        legacy_experiment_markdown_path=(
            None
            if payload.get("legacy_experiment_markdown_path") is None
            else _resolve(payload["legacy_experiment_markdown_path"])
        ),
        python_executable=python_executable,
        output_root=_resolve(payload.get("output_root", "data/interim/autoloop/unet")),
        llm=LLMPolicy(**payload["llm"]),
        batch=BatchPolicy(**payload["batch"]),
        patching=PatchingPolicy(**payload["patching"]),
        training=TrainingPolicy(
            require_benchmark_path=bool(payload["training"]["require_benchmark_path"]),
            require_preflight_tests=bool(payload["training"]["require_preflight_tests"]),
            preflight_test_paths=tuple(str(value) for value in payload["training"].get("preflight_test_paths", [])),
        ),
        promotion=PromotionPolicy(**payload["promotion"]),
        evaluation=EvaluationPolicy(**payload.get("evaluation", {})),
        search=SearchPolicy(
            **{
                **payload.get("search", {}),
                "round_modes": tuple(payload.get("search", {}).get("round_modes", ("explore", "exploit", "artifact_hardening", "exploit"))),
            }
        ),
    )


def _load_base_config_payload(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Base config at {path} must contain a top-level mapping.")
    return payload


def _load_metrics_payload(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Metrics file at {path} must contain a top-level object.")
    return payload


def _read_text_if_exists(path: Path | None) -> str | None:
    if path is None or not path.exists():
        return None
    return path.read_text(encoding="utf-8")


def load_jsonl_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        payload = json.loads(stripped)
        if isinstance(payload, dict):
            records.append(payload)
    return records


def _jsonl_known_experiment_ids(path: Path) -> set[str]:
    return {
        str(record["experiment_id"])
        for record in load_jsonl_records(path)
        if isinstance(record.get("experiment_id"), str)
    }


def _proposal_dedupe_key(proposal: HypothesisProposal) -> tuple[tuple[str, str], ...]:
    if proposal.implementation_type == "code":
        key_items = [("implementation_type", "code")]
        key_items.extend(("patch_target", target) for target in sorted(proposal.patch_targets))
        if proposal.patch_plan is not None:
            key_items.append(("patch_plan", proposal.patch_plan))
        return tuple(key_items)
    return tuple(sorted((key, json.dumps(value, sort_keys=True)) for key, value in proposal.overrides.items()))


def _serialize_dedupe_key(key: tuple[tuple[str, str], ...]) -> list[list[str]]:
    return [[item[0], item[1]] for item in key]


def _deserialize_dedupe_key(payload: list[list[str]]) -> tuple[tuple[str, str], ...]:
    return tuple((str(item[0]), str(item[1])) for item in payload)


def dedupe_and_limit_proposals(
    proposals: list[HypothesisProposal],
    *,
    max_hypotheses: int,
) -> list[HypothesisProposal]:
    seen: set[tuple[tuple[str, str], ...]] = set()
    retained: list[HypothesisProposal] = []
    for proposal in proposals:
        key = _proposal_dedupe_key(proposal)
        if key in seen:
            continue
        seen.add(key)
        retained.append(proposal)
        if len(retained) >= max_hypotheses:
            break
    return retained


def memory_path_for_policy(policy: AutonomousLoopPolicy) -> Path:
    return policy.output_root / "memory" / f"{policy.experiment_family}.json"


def load_family_memory(path: Path, *, experiment_family: str) -> FamilyMemory:
    if not path.exists():
        return FamilyMemory(experiment_family=experiment_family, updated_at=_utcnow())
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Family memory at {path} must contain a top-level object.")
    return FamilyMemory(
        experiment_family=str(payload.get("experiment_family", experiment_family)),
        updated_at=str(payload.get("updated_at", _utcnow())),
        tried_dedupe_keys=[list(item) for item in payload.get("tried_dedupe_keys", [])],
        banned_override_pairs=[list(item) for item in payload.get("banned_override_pairs", [])],
        completed_runs=list(payload.get("completed_runs", [])),
    )


def write_family_memory(path: Path, memory: FamilyMemory) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    memory.updated_at = _utcnow()
    path.write_text(json.dumps(asdict(memory), indent=2), encoding="utf-8")


def screen_proposals_against_memory(
    proposals: list[HypothesisProposal],
    *,
    memory: FamilyMemory,
    max_hypotheses: int,
    allow_code_patches: bool,
) -> tuple[list[HypothesisProposal], list[dict[str, Any]]]:
    retained: list[HypothesisProposal] = []
    rejected: list[dict[str, Any]] = []
    tried_keys = {_deserialize_dedupe_key(item) for item in memory.tried_dedupe_keys}
    banned_pairs = {(str(item[0]), str(item[1])) for item in memory.banned_override_pairs}
    seen: set[tuple[tuple[str, str], ...]] = set()

    for proposal in proposals:
        dedupe_key = _proposal_dedupe_key(proposal)
        if dedupe_key in seen:
            rejected.append({"name": proposal.name, "reason": "duplicate_in_batch", "overrides": proposal.overrides})
            continue
        seen.add(dedupe_key)
        if dedupe_key in tried_keys:
            rejected.append({"name": proposal.name, "reason": "duplicate_in_memory", "overrides": proposal.overrides})
            continue
        if proposal.implementation_type == "code" and not allow_code_patches:
            rejected.append(
                {
                    "name": proposal.name,
                    "reason": "requires_code_patch",
                    "patch_targets": proposal.patch_targets,
                    "patch_plan": proposal.patch_plan,
                }
            )
            continue
        banned_match = next(
            (
                [key, json.dumps(value, sort_keys=True)]
                for key, value in sorted(proposal.overrides.items())
                if (key, json.dumps(value, sort_keys=True)) in banned_pairs
            ),
            None,
        )
        if banned_match is not None:
            rejected.append(
                {
                    "name": proposal.name,
                    "reason": "banned_override_pair",
                    "matched_pair": banned_match,
                    "overrides": proposal.overrides,
                }
            )
            continue
        retained.append(proposal)
        if len(retained) >= max_hypotheses:
            break
    return retained, rejected


def _is_memory_run_strong(run: dict[str, Any], incumbent_metrics: dict[str, Any] | None, promotion: PromotionPolicy) -> bool:
    if run.get("result_class") == "promoted" or bool(run.get("is_competitive")):
        return True
    if incumbent_metrics is None or not isinstance(run.get("eval_mse_mean"), (int, float)):
        return False
    incumbent_value = incumbent_metrics.get(promotion.primary_metric)
    if not isinstance(incumbent_value, (int, float)):
        return False
    candidate_value = float(run["eval_mse_mean"])
    incumbent_value = float(incumbent_value)
    return candidate_value <= incumbent_value * 1.2 if not promotion.maximize else candidate_value >= incumbent_value * 0.8


def score_proposal_against_memory(
    proposal: HypothesisProposal,
    *,
    memory: FamilyMemory,
    incumbent_metrics: dict[str, Any] | None,
    promotion: PromotionPolicy,
) -> tuple[float, list[str]]:
    score = 0.0
    reasons: list[str] = []
    if proposal.implementation_type == "code":
        score -= 1.0
        reasons.append("code hypotheses are more expensive than config mutations")
    override_pairs = [(key, json.dumps(value, sort_keys=True)) for key, value in sorted(proposal.overrides.items())]
    for run in memory.completed_runs:
        overrides = run.get("overrides", {})
        if not isinstance(overrides, dict):
            continue
        overlap = sum(1 for key, value in override_pairs if overrides.get(key) == json.loads(value))
        if overlap == 0:
            continue
        if _is_memory_run_strong(run, incumbent_metrics, promotion):
            score += 2.5 * overlap
            reasons.append(f"overlaps with strong run {run.get('experiment_id')}")
        elif run.get("result_class") == "negative_result":
            penalty = 2.0 * overlap
            if isinstance(run.get("artifact_severity"), int) and int(run["artifact_severity"]) >= 4:
                penalty += 1.0 * overlap
            score -= penalty
            reasons.append(f"overlaps with weak run {run.get('experiment_id')}")
    score -= 0.15 * len(proposal.overrides)
    if len(proposal.overrides) <= 2:
        score += 0.5
        reasons.append("small local mutation")
    return score, reasons


def rank_proposals_against_memory(
    proposals: list[HypothesisProposal],
    *,
    memory: FamilyMemory,
    incumbent_metrics: dict[str, Any] | None,
    promotion: PromotionPolicy,
    max_hypotheses: int,
) -> tuple[list[HypothesisProposal], list[dict[str, Any]]]:
    ranked: list[tuple[float, int, HypothesisProposal, list[str]]] = []
    for index, proposal in enumerate(proposals):
        score, reasons = score_proposal_against_memory(
            proposal,
            memory=memory,
            incumbent_metrics=incumbent_metrics,
            promotion=promotion,
        )
        ranked.append((score, index, proposal, reasons))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    retained = [item[2] for item in ranked[:max_hypotheses]]
    annotations = [
        {
            "name": proposal.name,
            "ranking_score": score,
            "ranking_reasons": reasons,
        }
        for score, _index, proposal, reasons in ranked
    ]
    return retained, annotations


def classify_candidate_result(
    candidate: CandidateRunRecord,
    *,
    incumbent_metrics: dict[str, Any] | None,
    promotion: PromotionPolicy,
) -> str:
    if candidate.status != "completed":
        return "infra_failed"
    if candidate.eval_mse_mean is None:
        return "missing_metrics"
    incumbent_value = None if incumbent_metrics is None else incumbent_metrics.get(promotion.primary_metric)
    if incumbent_value is None:
        return "completed_unscored"
    candidate_value = float(candidate.eval_mse_mean)
    threshold = float(incumbent_value) - promotion.must_beat_incumbent_by
    if promotion.maximize:
        return "promoted" if candidate_value >= threshold else "negative_result"
    return "promoted" if candidate_value <= threshold else "negative_result"


def is_candidate_competitive(
    candidate: CandidateRunRecord,
    *,
    incumbent_metrics: dict[str, Any] | None,
    promotion: PromotionPolicy,
    evaluation: EvaluationPolicy,
) -> bool:
    if candidate.eval_mse_mean is None or incumbent_metrics is None:
        return False
    incumbent_value = incumbent_metrics.get(promotion.primary_metric)
    if incumbent_value is None:
        return False
    candidate_value = float(candidate.eval_mse_mean)
    incumbent_value = float(incumbent_value)
    ratio = float(evaluation.competitive_within_ratio)
    if promotion.maximize:
        return candidate_value >= incumbent_value * (1.0 - ratio)
    return candidate_value <= incumbent_value * (1.0 + ratio)


def _final_spatial_slice(data_array: xr.DataArray) -> np.ndarray:
    selected = data_array.isel(time_days=-1)
    while selected.ndim > 2:
        selected = selected.isel({selected.dims[0]: 0})
    values = np.asarray(selected.values, dtype=float)
    if values.ndim != 2:
        raise ValueError(f"Expected 2D final spatial slice, found shape {values.shape}.")
    return values


def write_final_step_heatmap(rollout_path: Path, output_path: Path) -> Path:
    dataset = xr.open_dataset(rollout_path)
    try:
        truth = _final_spatial_slice(dataset["truth_layer_thickness"])
        rollout = _final_spatial_slice(dataset["rollout_layer_thickness"])
        error = rollout - truth
    finally:
        dataset.close()

    shared_min = float(min(np.min(truth), np.min(rollout)))
    shared_max = float(max(np.max(truth), np.max(rollout)))
    error_abs = float(np.max(np.abs(error)))

    figure, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
    truth_image = axes[0].imshow(truth, origin="lower", cmap="viridis", vmin=shared_min, vmax=shared_max, aspect="auto")
    axes[0].set_title("Truth Final Step")
    axes[1].imshow(rollout, origin="lower", cmap="viridis", vmin=shared_min, vmax=shared_max, aspect="auto")
    axes[1].set_title("Rollout Final Step")
    error_image = axes[2].imshow(
        error,
        origin="lower",
        cmap="RdBu_r",
        vmin=-error_abs if error_abs > 0.0 else -1.0,
        vmax=error_abs if error_abs > 0.0 else 1.0,
        aspect="auto",
    )
    axes[2].set_title("Rollout - Truth")
    for axis in axes:
        axis.set_xlabel("x")
        axis.set_ylabel("y")
    figure.colorbar(truth_image, ax=axes[:2], shrink=0.85)
    figure.colorbar(error_image, ax=axes[2], shrink=0.85)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    return output_path


def maybe_write_evaluator_artifacts(
    policy: AutonomousLoopPolicy,
    *,
    candidate: CandidateRunRecord,
    incumbent_metrics: dict[str, Any] | None,
) -> CandidateRunRecord:
    config = load_unet_thickness_config(candidate.config_path).with_overrides(experiment_id=candidate.experiment_id)
    is_competitive = is_candidate_competitive(
        candidate,
        incumbent_metrics=incumbent_metrics,
        promotion=policy.promotion,
        evaluation=policy.evaluation,
    )
    if not is_competitive or not config.rollout_path.exists():
        candidate.is_competitive = is_competitive
        return candidate

    heatmap_path = config.raw_experiment_dir / policy.evaluation.final_step_heatmap_filename
    write_final_step_heatmap(config.rollout_path, heatmap_path)
    payload_path = config.raw_experiment_dir / policy.evaluation.evaluator_payload_filename
    incumbent_value = None if incumbent_metrics is None else incumbent_metrics.get(policy.promotion.primary_metric)
    payload = {
        "experiment_id": candidate.experiment_id,
        "hypothesis": candidate.hypothesis,
        "metrics_path": None if candidate.metrics_path is None else str(Path(candidate.metrics_path).resolve()),
        "rollout_path": str(config.rollout_path.resolve()),
        "final_step_heatmap_path": str(heatmap_path.resolve()),
        "primary_metric": policy.promotion.primary_metric,
        "candidate_primary_metric_value": candidate.eval_mse_mean,
        "incumbent_primary_metric_value": incumbent_value,
        "competitive_within_ratio": policy.evaluation.competitive_within_ratio,
        "guidance": policy.evaluation.artifact_guidance,
        "artifact_focus": [
            "boundary reflections strengthening over time",
            "artifact propagation inward from boundaries",
            "numerical ringing or nonphysical edge amplification",
        ],
    }
    payload_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    candidate.is_competitive = True
    candidate.final_step_heatmap_path = str(heatmap_path.resolve())
    candidate.evaluator_payload_path = str(payload_path.resolve())
    return candidate


def build_evaluator_prompt(
    *,
    candidate: CandidateRunRecord,
    evaluator_payload: dict[str, Any],
    compact_summary: str | None,
) -> tuple[str, str]:
    system = (
        "You evaluate autonomous emulator experiment results. "
        "Return JSON only with keys: artifact_severity, artifact_tags, accept_tradeoff_for_cleaner_rollout, "
        "notes_for_next_hypothesis_batch, banned_patterns. "
        "artifact_severity must be an integer from 0 to 5 where 5 is severe boundary-driven numerical artifact."
    )
    user_payload = {
        "candidate": {
            "experiment_id": candidate.experiment_id,
            "hypothesis": candidate.hypothesis,
            "eval_mse_mean": candidate.eval_mse_mean,
            "result_class": candidate.result_class,
        },
        "review_payload": evaluator_payload,
        "compact_summary": compact_summary,
    }
    return system, json.dumps(user_payload, indent=2, sort_keys=True)


def evaluate_competitive_candidate_via_openrouter(
    policy: AutonomousLoopPolicy,
    *,
    candidate: CandidateRunRecord,
) -> dict[str, Any]:
    if candidate.evaluator_payload_path is None:
        raise ValueError("Competitive evaluator requires evaluator_payload_path.")
    evaluator_payload = json.loads(Path(candidate.evaluator_payload_path).read_text(encoding="utf-8"))
    compact_summary = _read_text_if_exists(policy.experiment_compact_path)
    api_key = configure_openrouter(policy.llm.env_var_name)
    client = OpenAI(base_url=policy.llm.api_base, api_key=api_key)
    system_prompt, user_prompt = build_evaluator_prompt(
        candidate=candidate,
        evaluator_payload=evaluator_payload,
        compact_summary=compact_summary,
    )
    response = client.chat.completions.create(
        model=policy.llm.model_name,
        temperature=0.1,
        max_tokens=policy.llm.max_tokens,
        timeout=policy.llm.timeout_seconds,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    content = response.choices[0].message.content
    if content is None:
        raise ValueError("Evaluator returned empty content.")
    payload = json.loads(content)
    if not isinstance(payload, dict):
        raise ValueError("Evaluator payload must be an object.")
    severity = int(payload.get("artifact_severity", 0))
    if severity < 0 or severity > 5:
        raise ValueError(f"artifact_severity out of range: {severity}")
    artifact_tags = payload.get("artifact_tags", [])
    banned_patterns = payload.get("banned_patterns", [])
    return {
        "artifact_severity": severity,
        "artifact_tags": [str(tag) for tag in artifact_tags] if isinstance(artifact_tags, list) else [],
        "accept_tradeoff_for_cleaner_rollout": bool(payload.get("accept_tradeoff_for_cleaner_rollout", False)),
        "notes_for_next_hypothesis_batch": str(payload.get("notes_for_next_hypothesis_batch", "")),
        "banned_patterns": [str(pattern) for pattern in banned_patterns] if isinstance(banned_patterns, list) else [],
    }


def maybe_run_competitive_evaluator(
    policy: AutonomousLoopPolicy,
    *,
    candidate: CandidateRunRecord,
    batch_dir: Path,
) -> CandidateRunRecord:
    if not candidate.is_competitive or candidate.evaluator_payload_path is None:
        return candidate
    evaluation = evaluate_competitive_candidate_via_openrouter(policy, candidate=candidate)
    summary_path = batch_dir / "evaluations" / f"{candidate.experiment_id}.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(evaluation, indent=2), encoding="utf-8")
    candidate.evaluator_summary_path = str(summary_path.resolve())
    candidate.artifact_severity = int(evaluation["artifact_severity"])
    candidate.artifact_tags = list(evaluation["artifact_tags"])
    candidate.evaluator_notes = evaluation["notes_for_next_hypothesis_batch"]
    candidate.accept_tradeoff_for_cleaner_rollout = bool(evaluation["accept_tradeoff_for_cleaner_rollout"])
    return candidate


def derive_banned_override_pairs(memory: FamilyMemory) -> list[list[str]]:
    negative_counts: dict[tuple[str, str], int] = {}
    promoted_pairs: set[tuple[str, str]] = set()
    for run in memory.completed_runs:
        overrides = run.get("overrides", {})
        if not isinstance(overrides, dict):
            continue
        result_class = str(run.get("result_class", ""))
        for key, value in sorted(overrides.items()):
            pair = (str(key), json.dumps(value, sort_keys=True))
            if result_class == "promoted":
                promoted_pairs.add(pair)
            elif result_class == "negative_result":
                negative_counts[pair] = negative_counts.get(pair, 0) + 1
    retained = []
    for pair, count in sorted(negative_counts.items()):
        if count >= 2 and pair not in promoted_pairs:
            retained.append([pair[0], pair[1]])
    return retained


def _parse_hypothesis_response(content: str) -> tuple[list[HypothesisProposal], list[dict[str, Any]]]:
    payload = json.loads(content)
    hypotheses = payload.get("hypotheses")
    if not isinstance(hypotheses, list):
        raise ValueError("LLM response must include a top-level 'hypotheses' list.")
    proposals: list[HypothesisProposal] = []
    rejected: list[dict[str, Any]] = []
    for entry in hypotheses:
        if not isinstance(entry, dict):
            raise ValueError("Each hypothesis entry must be an object.")
        name = str(entry["name"])
        hypothesis = str(entry["hypothesis"])
        implementation_type = str(entry.get("implementation_type", "config"))
        search_mode = str(entry.get("search_mode", "exploit"))
        hypothesis_family = str(entry.get("hypothesis_family", "optimizer_tuning"))
        if search_mode not in {"explore", "exploit", "artifact_hardening"}:
            rejected.append(
                {
                    "name": name,
                    "hypothesis": hypothesis,
                    "reason": "invalid_search_mode",
                    "details": f"Unsupported search_mode: {search_mode}",
                }
            )
            continue
        if implementation_type not in {"config", "code"}:
            rejected.append(
                {
                    "name": name,
                    "hypothesis": hypothesis,
                    "reason": "invalid_implementation_type",
                    "details": f"Unsupported implementation_type: {implementation_type}",
                }
            )
            continue
        if implementation_type == "code":
            raw_targets = entry.get("patch_targets", [])
            if not isinstance(raw_targets, list) or not all(isinstance(value, str) for value in raw_targets):
                rejected.append(
                    {
                        "name": name,
                        "hypothesis": hypothesis,
                        "reason": "invalid_patch_targets",
                        "details": "Code hypotheses must include patch_targets as a list of strings.",
                    }
                )
                continue
            patch_plan = entry.get("patch_plan")
            if patch_plan is not None and not isinstance(patch_plan, str):
                rejected.append(
                    {
                        "name": name,
                        "hypothesis": hypothesis,
                        "reason": "invalid_patch_plan",
                        "details": "patch_plan must be a string when provided.",
                    }
                )
                continue
            proposals.append(
                HypothesisProposal(
                    name=name,
                    hypothesis=hypothesis,
                    implementation_type="code",
                    search_mode=search_mode,
                    hypothesis_family=hypothesis_family,
                    patch_targets=list(raw_targets),
                    patch_plan=patch_plan,
                )
            )
            continue

        raw_overrides = entry.get("overrides")
        if not isinstance(raw_overrides, dict):
            rejected.append(
                {
                    "name": name,
                    "hypothesis": hypothesis,
                    "reason": "invalid_overrides_payload",
                    "details": "Config hypotheses must include overrides as an object.",
                }
            )
            continue
        try:
            overrides = validate_candidate_overrides(dict(raw_overrides))
        except ValueError as exc:
            rejected.append(
                {
                    "name": name,
                    "hypothesis": hypothesis,
                    "reason": "invalid_llm_proposal",
                    "details": str(exc),
                    "overrides": raw_overrides,
                }
            )
            continue
        proposals.append(
            HypothesisProposal(
                name=name,
                hypothesis=hypothesis,
                implementation_type="config",
                search_mode=search_mode,
                hypothesis_family=hypothesis_family,
                overrides=overrides,
            )
        )
    return proposals, rejected


def build_hypothesis_prompt(
    *,
    base_config_payload: dict[str, Any],
    incumbent_metrics: dict[str, Any] | None,
    memory: FamilyMemory,
    policy: AutonomousLoopPolicy,
) -> tuple[str, str]:
    benchmark_summary = "No benchmark metrics available."
    if incumbent_metrics is not None:
        benchmark_summary = json.dumps(
            {
                "eval_mse_mean": incumbent_metrics.get("eval_mse_mean"),
                "eval_mse_max": incumbent_metrics.get("eval_mse_max"),
                "train_loss": incumbent_metrics.get("train_loss"),
                "epochs_completed": incumbent_metrics.get("epochs_completed"),
                "stopped_early": incumbent_metrics.get("stopped_early"),
            },
            indent=2,
            sort_keys=True,
        )
    compact_summary = _read_text_if_exists(policy.experiment_compact_path)
    compact_summary_block = compact_summary if compact_summary is not None else "No compact experiment summary available yet."
    recent_evaluator_feedback = [
        {
            "experiment_id": run.get("experiment_id"),
            "artifact_severity": run.get("artifact_severity"),
            "artifact_tags": run.get("artifact_tags"),
            "accept_tradeoff_for_cleaner_rollout": run.get("accept_tradeoff_for_cleaner_rollout"),
            "evaluator_notes": run.get("evaluator_notes"),
        }
        for run in memory.completed_runs[-5:]
        if run.get("evaluator_notes") is not None or run.get("artifact_severity") is not None
    ]
    system = (
        "You are proposing bounded U-Net emulator experiment hypotheses for an ocean modeling repo. "
        "Return JSON only with top-level key 'hypotheses'. "
        "Each hypothesis must contain keys: name, hypothesis, implementation_type. "
        "implementation_type must be either 'config' or 'code'. "
        "If implementation_type is 'config', include an 'overrides' object and do not change more than 4 override keys. "
        "If implementation_type is 'code', include 'patch_targets' as a list of file paths and 'patch_plan' as a short implementation note. "
        "For config hypotheses, each override must stay inside the validated search space already used by the repo."
    )
    user = (
        f"Base config:\n{json.dumps(base_config_payload, indent=2, sort_keys=True)}\n\n"
        f"Current incumbent metrics:\n{benchmark_summary}\n\n"
        f"Policy:\n"
        f"- max_hypotheses: {policy.batch.max_hypotheses}\n"
        f"- executable path today: config hypotheses\n"
        f"- allow_code_patches policy flag: {policy.batch.allow_code_patches}\n"
        f"- primary metric: {policy.promotion.primary_metric}\n"
        f"- maximize: {policy.promotion.maximize}\n\n"
        "Additional objective:\n"
        "- Boundary reflections that strengthen over time and propagate inward are a first-class failure mode.\n"
        "- Do not optimize blindly for eval_mse_mean if a proposal is likely to worsen numerical artifacts.\n"
        "- Competitive runs may later be judged with final-step heatmaps, and slightly worse MSE can be acceptable if artifacts are materially reduced.\n\n"
        f"Compacted experiment summary for review:\n{compact_summary_block}\n\n"
        f"Structured memory:\n{json.dumps({'banned_override_pairs': memory.banned_override_pairs, 'recent_completed_runs': memory.completed_runs[-5:], 'recent_evaluator_feedback': recent_evaluator_feedback}, indent=2, sort_keys=True)}\n\n"
        "Generate a diverse but disciplined batch of hypotheses. "
        "Prefer executable config hypotheses when possible, but you may propose code-backed hypotheses if the idea truly requires code changes. "
        "Prefer meaningful structural or optimizer changes over tiny nudges. "
        "Avoid near-duplicates."
    )
    return system, user


def propose_hypotheses_via_openrouter(
    policy: AutonomousLoopPolicy,
    *,
    base_config_payload: dict[str, Any],
    incumbent_metrics: dict[str, Any] | None,
    memory: FamilyMemory,
    max_hypotheses: int,
) -> tuple[list[HypothesisProposal], list[dict[str, Any]]]:
    api_key = configure_openrouter(policy.llm.env_var_name)
    client = OpenAI(base_url=policy.llm.api_base, api_key=api_key)
    system_prompt, user_prompt = build_hypothesis_prompt(
        base_config_payload=base_config_payload,
        incumbent_metrics=incumbent_metrics,
        memory=memory,
        policy=policy,
    )
    response = client.chat.completions.create(
        model=policy.llm.model_name,
        temperature=policy.llm.temperature,
        max_tokens=policy.llm.max_tokens,
        timeout=policy.llm.timeout_seconds,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    content = response.choices[0].message.content
    if content is None:
        raise ValueError("LLM returned empty content.")
    proposals, rejected = _parse_hypothesis_response(content)
    if not proposals:
        raise ValueError(f"LLM returned no valid proposals. Rejections: {json.dumps(rejected, sort_keys=True)}")
    return dedupe_and_limit_proposals(proposals, max_hypotheses=max_hypotheses), rejected


def build_code_patch_prompt(
    *,
    proposal: HypothesisProposal,
    file_payloads: dict[str, str],
    previous_error: str | None = None,
) -> tuple[str, str]:
    system = (
        "You are generating a unified diff patch for a bounded code experiment in a Python repository. "
        "Return only a git-compatible unified diff. "
        "Touch only the listed patch_targets. "
        "Do not include markdown fences or explanations."
    )
    payload = {
        "hypothesis": proposal.hypothesis,
        "patch_targets": proposal.patch_targets,
        "patch_plan": proposal.patch_plan,
        "previous_error": previous_error,
        "files": file_payloads,
    }
    user = json.dumps(payload, indent=2, sort_keys=True)
    return system, user


def generate_code_patch_via_openrouter(
    policy: AutonomousLoopPolicy,
    *,
    proposal: HypothesisProposal,
    previous_error: str | None = None,
) -> str:
    api_key = configure_openrouter(policy.llm.env_var_name)
    client = OpenAI(base_url=policy.llm.api_base, api_key=api_key)
    file_payloads: dict[str, str] = {}
    for target in proposal.patch_targets:
        path = REPO_ROOT / target
        if not path.exists():
            raise ValueError(f"Patch target does not exist: {target}")
        file_payloads[target] = path.read_text(encoding="utf-8")
    system_prompt, user_prompt = build_code_patch_prompt(
        proposal=proposal,
        file_payloads=file_payloads,
        previous_error=previous_error,
    )
    response = client.chat.completions.create(
        model=policy.llm.model_name,
        temperature=0.1,
        max_tokens=policy.llm.max_tokens,
        timeout=policy.llm.timeout_seconds,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    content = response.choices[0].message.content
    if content is None:
        raise ValueError("LLM returned empty patch content.")
    return content.strip()


def branch_name_for_candidate(record: CandidateRunRecord) -> str:
    return f"autoloop/{record.experiment_id}"


def prepare_candidate_workspace(record: CandidateRunRecord) -> tuple[Path, str]:
    worktree_root = Path(tempfile.gettempdir()) / "autoloop-worktrees" / record.experiment_id
    branch_name = branch_name_for_candidate(record)
    if worktree_root.exists():
        subprocess.run(["git", "worktree", "remove", "--force", str(worktree_root)], cwd=REPO_ROOT, check=False)
    subprocess.run(["git", "branch", "-D", branch_name], cwd=REPO_ROOT, check=False, capture_output=True, text=True)
    worktree_root.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "worktree", "add", "-b", branch_name, str(worktree_root), "HEAD"], cwd=REPO_ROOT, check=True)
    return worktree_root, branch_name


def cleanup_candidate_workspace(worktree_root: Path) -> None:
    subprocess.run(["git", "worktree", "remove", "--force", str(worktree_root)], cwd=REPO_ROOT, check=False)


def apply_patch_in_workspace(worktree_root: Path, patch_text: str, patch_path: Path) -> None:
    patch_path.parent.mkdir(parents=True, exist_ok=True)
    patch_path.write_text(patch_text + "\n", encoding="utf-8")
    subprocess.run(["git", "apply", "--check", str(patch_path)], cwd=worktree_root, check=True, capture_output=True, text=True)
    subprocess.run(["git", "apply", str(patch_path)], cwd=worktree_root, check=True, capture_output=True, text=True)


def sync_candidate_config_into_workspace(record: CandidateRunRecord, worktree_root: Path) -> Path:
    config_path = Path(record.config_path)
    relative = config_path.relative_to(REPO_ROOT)
    workspace_config_path = worktree_root / relative
    workspace_config_path.parent.mkdir(parents=True, exist_ok=True)
    workspace_config_path.write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")
    return workspace_config_path


def commit_candidate_branch_state(
    *,
    record: CandidateRunRecord,
    worktree_root: Path,
    workspace_config_path: Path,
) -> str:
    paths_to_add = [str(workspace_config_path.relative_to(worktree_root))]
    for target in record.patch_targets:
        paths_to_add.append(target)
    subprocess.run(["git", "add", "-f", *paths_to_add], cwd=worktree_root, check=True)
    commit_message = f"Autoloop candidate {record.experiment_id}"
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Autoloop",
            "-c",
            "user.email=autoloop@example.com",
            "commit",
            "-m",
            commit_message,
        ],
        cwd=worktree_root,
        check=True,
        capture_output=True,
        text=True,
    )
    completed = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=worktree_root, check=True, capture_output=True, text=True)
    return completed.stdout.strip()


def run_preflight_checks(policy: AutonomousLoopPolicy) -> dict[str, Any]:
    config = load_unet_thickness_config(policy.base_config_path)
    benchmark_path = config.early_stopping_best_metrics_path
    checks: dict[str, Any] = {
        "base_config_path": str(policy.base_config_path),
        "base_config_exists": policy.base_config_path.exists(),
        "experiment_log_path": str(policy.experiment_log_path),
        "experiment_log_exists": policy.experiment_log_path.exists(),
        "experiment_compact_path": str(policy.experiment_compact_path),
        "experiment_compact_exists": policy.experiment_compact_path.exists(),
        "benchmark_metrics_path": None if benchmark_path is None else str(benchmark_path),
        "benchmark_metrics_exists": False if benchmark_path is None else benchmark_path.exists(),
    }
    if policy.training.require_benchmark_path and not checks["benchmark_metrics_exists"]:
        raise ValueError("Benchmark metrics path is required but missing.")
    if policy.training.require_preflight_tests and policy.training.preflight_test_paths:
        command = [policy.python_executable, "-m", "pytest", "-q", *policy.training.preflight_test_paths]
        completed = subprocess.run(command, cwd=Path.cwd(), capture_output=True, text=True, check=False)
        checks["preflight_tests"] = {
            "return_code": completed.returncode,
            "command": command,
            "stdout_tail": completed.stdout[-2000:],
            "stderr_tail": completed.stderr[-2000:],
        }
        if completed.returncode != 0:
            raise RuntimeError("Preflight tests failed.")
    return checks


def materialize_candidate_configs(
    policy: AutonomousLoopPolicy,
    *,
    proposals: list[HypothesisProposal],
    batch_dir: Path,
    start_index: int = 1,
) -> list[CandidateRunRecord]:
    base_payload = _load_base_config_payload(policy.base_config_path)
    config_dir = batch_dir / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    batch_id = batch_dir.name
    records: list[CandidateRunRecord] = []
    for index, proposal in enumerate(proposals, start=start_index):
        payload = dict(base_payload)
        payload["hypothesis"] = proposal.hypothesis
        payload.update(proposal.overrides)
        for path_key in ("source_data_root", "raw_output_root", "interim_output_root", "early_stopping_best_metrics_path"):
            if payload.get(path_key) is not None:
                resolved = Path(payload[path_key])
                payload[path_key] = str(resolved if resolved.is_absolute() else (REPO_ROOT / resolved))
        suffix = _slugify(proposal.name)
        experiment_id = f"{batch_id}-{index:02d}-{suffix}"
        config_path = config_dir / f"{index:02d}-{suffix}.yaml"
        config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        records.append(
            CandidateRunRecord(
                name=proposal.name,
                hypothesis=proposal.hypothesis,
                implementation_type=proposal.implementation_type,
                search_mode=proposal.search_mode,
                hypothesis_family=proposal.hypothesis_family,
                config_path=str(config_path.resolve()),
                experiment_id=experiment_id,
                overrides=proposal.overrides,
                status="materialized",
                patch_targets=proposal.patch_targets,
                patch_plan=proposal.patch_plan,
            )
        )
    return records


def _load_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else None


def run_candidate_training(
    policy: AutonomousLoopPolicy,
    *,
    record: CandidateRunRecord,
) -> CandidateRunRecord:
    started_at = _utcnow()
    started_perf = time.perf_counter()
    cwd = REPO_ROOT
    patch_path: Path | None = None
    worktree_root: Path | None = None
    workspace_config_path: Path | None = None
    completed: subprocess.CompletedProcess[str] | None = None
    failure_text: str | None = None
    patch_attempts_used = 0
    branch_name: str | None = None
    branch_commit: str | None = None
    try:
        worktree_root, branch_name = prepare_candidate_workspace(record)
        cwd = worktree_root
        workspace_config_path = sync_candidate_config_into_workspace(record, worktree_root)
        if record.implementation_type == "code":
            patch_path = REPO_ROOT / "data" / "interim" / "autoloop" / "unet" / record.experiment_id / "patch.diff"
            previous_error: str | None = None
            applied = False
            for _attempt in range(policy.patching.max_patch_attempts_per_hypothesis):
                patch_attempts_used += 1
                patch_text = generate_code_patch_via_openrouter(
                    policy,
                    proposal=HypothesisProposal(
                        name=record.name,
                        hypothesis=record.hypothesis,
                        implementation_type=record.implementation_type,
                        overrides=record.overrides,
                        patch_targets=record.patch_targets,
                        patch_plan=record.patch_plan,
                    ),
                    previous_error=previous_error,
                )
                try:
                    apply_patch_in_workspace(worktree_root, patch_text, patch_path)
                    applied = True
                    break
                except subprocess.CalledProcessError as exc:
                    previous_error = exc.stderr[-2000:] if exc.stderr else str(exc)
            if not applied:
                raise RuntimeError(f"Unable to apply generated patch for {record.experiment_id}: {previous_error}")

            if policy.training.require_preflight_tests and policy.training.preflight_test_paths:
                test_command = [policy.python_executable, "-m", "pytest", "-q", *policy.training.preflight_test_paths]
                test_completed = subprocess.run(test_command, cwd=worktree_root, capture_output=True, text=True, check=False)
                if test_completed.returncode != 0:
                    raise RuntimeError(f"Patched workspace preflight failed: {test_completed.stderr[-2000:] or test_completed.stdout[-2000:]}")

        branch_commit = commit_candidate_branch_state(
            record=record,
            worktree_root=worktree_root,
            workspace_config_path=workspace_config_path,
        )

        command = [
            policy.python_executable,
            "-m",
            "src.pipelines.train_unet_thickness",
            "--config",
            str(workspace_config_path),
            "--experiment-id",
            record.experiment_id,
        ]
        completed = subprocess.run(command, cwd=cwd, capture_output=True, text=True, check=False)
    except BaseException as exc:
        failure_text = str(exc)
        completed = subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr=failure_text)
    finally:
        duration_seconds = time.perf_counter() - started_perf
        finished_at = _utcnow()
        if worktree_root is not None:
            cleanup_candidate_workspace(worktree_root)

    config = load_unet_thickness_config(record.config_path).with_overrides(experiment_id=record.experiment_id)
    history = _load_json_if_exists(config.training_history_path)
    metrics = _load_json_if_exists(config.metrics_path)

    updated = CandidateRunRecord(
        name=record.name,
        hypothesis=record.hypothesis,
        implementation_type=record.implementation_type,
        search_mode=record.search_mode,
        hypothesis_family=record.hypothesis_family,
        config_path=record.config_path,
        experiment_id=record.experiment_id,
        overrides=record.overrides,
        status="completed" if completed.returncode == 0 else "failed",
        patch_targets=record.patch_targets,
        patch_plan=record.patch_plan,
        workspace_path=None if worktree_root is None else str(worktree_root),
        patch_path=None if patch_path is None else str(patch_path),
        branch_name=branch_name,
        branch_commit=branch_commit,
        llm_calls_used=record.llm_calls_used,
        patch_calls_used=patch_attempts_used,
        ranking_score=record.ranking_score,
        started_at=started_at,
        finished_at=finished_at,
        return_code=completed.returncode,
        duration_seconds=duration_seconds,
        metrics_path=None if metrics is None else str(config.metrics_path),
        training_history_path=None if history is None else str(config.training_history_path),
        eval_mse_mean=None if metrics is None else metrics.get("eval_mse_mean"),
        stop_reason=(None if history is None else history.get("stop_reason")) or failure_text,
        result_class=record.result_class,
        is_competitive=record.is_competitive,
        final_step_heatmap_path=record.final_step_heatmap_path,
        evaluator_payload_path=record.evaluator_payload_path,
        evaluator_summary_path=record.evaluator_summary_path,
        artifact_severity=record.artifact_severity,
        artifact_tags=list(record.artifact_tags),
        evaluator_notes=record.evaluator_notes,
        accept_tradeoff_for_cleaner_rollout=record.accept_tradeoff_for_cleaner_rollout,
        stdout_tail=completed.stdout[-2000:],
        stderr_tail=completed.stderr[-2000:],
    )
    return updated


def reconcile_candidate_from_artifacts(record: CandidateRunRecord) -> CandidateRunRecord:
    config = load_unet_thickness_config(record.config_path).with_overrides(experiment_id=record.experiment_id)
    history = _load_json_if_exists(config.training_history_path)
    metrics = _load_json_if_exists(config.metrics_path)
    if history is None and metrics is None:
        return record

    status = record.status
    if record.return_code is None and metrics is not None:
        status = "completed"

    stop_reason = record.stop_reason
    if stop_reason is None:
        if history is not None:
            stop_reason = history.get("stop_reason")
        elif metrics is not None:
            stop_reason = metrics.get("stop_reason")

    eval_mse_mean = record.eval_mse_mean
    if eval_mse_mean is None and metrics is not None:
        eval_mse_mean = metrics.get("eval_mse_mean")

    finished_at = record.finished_at
    if finished_at is None:
        if history is not None and isinstance(history.get("updated_at"), str):
            finished_at = history["updated_at"]
        elif metrics is not None and isinstance(metrics.get("updated_at"), str):
            finished_at = metrics["updated_at"]

    return CandidateRunRecord(
        name=record.name,
        hypothesis=record.hypothesis,
        implementation_type=record.implementation_type,
        search_mode=record.search_mode,
        hypothesis_family=record.hypothesis_family,
        config_path=record.config_path,
        experiment_id=record.experiment_id,
        overrides=record.overrides,
        status=status,
        patch_targets=record.patch_targets,
        patch_plan=record.patch_plan,
        workspace_path=record.workspace_path,
        patch_path=record.patch_path,
        branch_name=record.branch_name,
        branch_commit=record.branch_commit,
        llm_calls_used=record.llm_calls_used,
        patch_calls_used=record.patch_calls_used,
        ranking_score=record.ranking_score,
        started_at=record.started_at,
        finished_at=finished_at,
        return_code=record.return_code,
        duration_seconds=record.duration_seconds,
        metrics_path=None if metrics is None else str(config.metrics_path),
        training_history_path=None if history is None else str(config.training_history_path),
        eval_mse_mean=eval_mse_mean,
        stop_reason=stop_reason,
        result_class=record.result_class,
        is_competitive=record.is_competitive,
        final_step_heatmap_path=record.final_step_heatmap_path,
        evaluator_payload_path=record.evaluator_payload_path,
        evaluator_summary_path=record.evaluator_summary_path,
        artifact_severity=record.artifact_severity,
        artifact_tags=list(record.artifact_tags),
        evaluator_notes=record.evaluator_notes,
        accept_tradeoff_for_cleaner_rollout=record.accept_tradeoff_for_cleaner_rollout,
        stdout_tail=record.stdout_tail,
        stderr_tail=record.stderr_tail,
    )


def current_git_branch() -> str:
    completed = subprocess.run(
        ["git", "branch", "--show-current"],
        cwd=Path.cwd(),
        capture_output=True,
        text=True,
        check=False,
    )
    value = completed.stdout.strip()
    return value or "unknown"


def current_git_commit() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=Path.cwd(),
        capture_output=True,
        text=True,
        check=False,
    )
    value = completed.stdout.strip()
    return value or "unknown"


def render_candidate_outcome_markdown(candidate: CandidateRunRecord) -> str:
    config_label = Path(candidate.config_path).name
    branch_text = ""
    if candidate.branch_name is not None:
        branch_text = f" Branch: `{candidate.branch_name}`."
    if candidate.branch_commit is not None:
        branch_text += f" Commit: `{candidate.branch_commit}`."
    if candidate.status == "skipped_budget":
        return (
            "Not run due to autonomous batch budget limit. "
            f"Config: [`{config_label}`]({Path(candidate.config_path).resolve()}).{branch_text}"
        )
    if candidate.status != "completed":
        reason = candidate.stop_reason or "training subprocess failed"
        return (
            "Failed during autonomous execution. "
            f"Reason: `{reason}`. Config: [`{config_label}`]({Path(candidate.config_path).resolve()}).{branch_text}"
        )
    metrics_label = None if candidate.metrics_path is None else Path(candidate.metrics_path).resolve()
    metric_text = "metrics unavailable"
    if candidate.eval_mse_mean is not None:
        metric_text = f"`eval_mse_mean = {float(candidate.eval_mse_mean):.4f}`"
    stop_reason_text = "" if not candidate.stop_reason else f" The controller recorded stop reason: `{candidate.stop_reason}`."
    competitive_text = ""
    if candidate.is_competitive and candidate.final_step_heatmap_path is not None:
        competitive_text = (
            f" Competitive run review artifact: [`final_step_heatmap.png`]({Path(candidate.final_step_heatmap_path).resolve()})."
        )
    evaluator_text = ""
    if candidate.artifact_severity is not None:
        evaluator_text = (
            f" Evaluator artifact severity: `{candidate.artifact_severity}`."
        )
        if candidate.accept_tradeoff_for_cleaner_rollout is not None:
            evaluator_text += (
                f" Tradeoff accepted for cleaner rollout: `{candidate.accept_tradeoff_for_cleaner_rollout}`."
            )
    if metrics_label is not None:
        return (
            "Autonomous batch result. "
            f"The run wrote [`metrics.json`]({metrics_label}) with {metric_text}.{stop_reason_text}{competitive_text}{evaluator_text} "
            f"Config: [`{config_label}`]({Path(candidate.config_path).resolve()}).{branch_text}"
        )
    return f"Autonomous batch result with {metric_text}.{stop_reason_text}{branch_text}"


def _extract_markdown_link_target(text: str, suffix: str) -> str | None:
    match = re.search(rf"\]\(([^)]+{re.escape(suffix)})\)", text)
    return None if match is None else match.group(1)


def _extract_eval_mse_mean(text: str) -> float | None:
    match = re.search(r"eval_mse_mean\s*=\s*([0-9]+(?:\.[0-9]+)?)", text)
    return None if match is None else float(match.group(1))


def backfill_experiment_log_jsonl_from_markdown(jsonl_path: Path, markdown_path: Path) -> int:
    if not markdown_path.exists():
        return 0
    known_ids = _jsonl_known_experiment_ids(jsonl_path)
    appended = 0
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("a", encoding="utf-8") as handle:
        for line in markdown_path.read_text(encoding="utf-8").splitlines():
            if not line.startswith("| `"):
                continue
            parts = [part.strip() for part in line.strip().strip("|").split("|")]
            if len(parts) != 5:
                continue
            experiment_id = parts[0].strip("`")
            if experiment_id in known_ids:
                continue
            outcome_text = parts[4]
            record = {
                "logged_at": _utcnow(),
                "log_source": "markdown_backfill",
                "experiment_id": experiment_id,
                "branch_name": parts[1].strip("`"),
                "branch_commit": parts[2].strip("`"),
                "hypothesis": parts[3],
                "status": (
                    "not_run"
                    if "Not run." in outcome_text
                    else "failed"
                    if "Failed during autonomous execution" in outcome_text
                    else "completed"
                ),
                "result_class": (
                    "invalid"
                    if "Invalid benchmark" in outcome_text
                    else "negative_result"
                    if "negative" in outcome_text.lower()
                    else None
                ),
                "eval_mse_mean": _extract_eval_mse_mean(outcome_text),
                "metrics_path": _extract_markdown_link_target(outcome_text, "metrics.json"),
                "config_path": _extract_markdown_link_target(outcome_text, ".yaml"),
                "outcome_markdown": outcome_text,
            }
            handle.write(json.dumps(record, sort_keys=True) + "\n")
            known_ids.add(experiment_id)
            appended += 1
    return appended


def append_experiment_log_record(
    log_path: Path,
    *,
    batch_id: str,
    experiment_family: str,
    candidate: CandidateRunRecord,
) -> None:
    known_ids = _jsonl_known_experiment_ids(log_path)
    if candidate.experiment_id in known_ids:
        return
    payload = {
        "logged_at": _utcnow(),
        "log_source": "autonomous_loop",
        "batch_id": batch_id,
        "experiment_family": experiment_family,
        "experiment_id": candidate.experiment_id,
        "name": candidate.name,
        "hypothesis": candidate.hypothesis,
        "implementation_type": candidate.implementation_type,
        "status": candidate.status,
        "result_class": candidate.result_class,
        "is_competitive": candidate.is_competitive,
        "branch_name": candidate.branch_name,
        "branch_commit": candidate.branch_commit,
        "config_path": candidate.config_path,
        "metrics_path": candidate.metrics_path,
        "training_history_path": candidate.training_history_path,
        "final_step_heatmap_path": candidate.final_step_heatmap_path,
        "evaluator_payload_path": candidate.evaluator_payload_path,
        "eval_mse_mean": candidate.eval_mse_mean,
        "stop_reason": candidate.stop_reason,
        "started_at": candidate.started_at,
        "finished_at": candidate.finished_at,
        "duration_seconds": candidate.duration_seconds,
        "patch_targets": candidate.patch_targets,
        "patch_plan": candidate.patch_plan,
        "patch_calls_used": candidate.patch_calls_used,
        "llm_calls_used": candidate.llm_calls_used,
        "ranking_score": candidate.ranking_score,
        "overrides": candidate.overrides,
        "evaluator_summary_path": candidate.evaluator_summary_path,
        "artifact_severity": candidate.artifact_severity,
        "artifact_tags": candidate.artifact_tags,
        "evaluator_notes": candidate.evaluator_notes,
        "accept_tradeoff_for_cleaner_rollout": candidate.accept_tradeoff_for_cleaner_rollout,
    }
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def render_experiment_compact_markdown(
    *,
    experiment_family: str,
    records: list[dict[str, Any]],
) -> str:
    family_records = [record for record in records if record.get("experiment_family", experiment_family) == experiment_family]
    completed_with_metric = [
        record
        for record in family_records
        if record.get("status") == "completed" and isinstance(record.get("eval_mse_mean"), (int, float))
    ]
    completed_with_metric.sort(key=lambda record: float(record["eval_mse_mean"]))
    best = completed_with_metric[0] if completed_with_metric else None
    best_value = None if best is None else float(best["eval_mse_mean"])
    competitive = [
        record
        for record in completed_with_metric
        if bool(record.get("is_competitive"))
        or (
            best_value is not None
            and float(record["eval_mse_mean"]) <= best_value * 1.2
        )
    ][:5]
    early_stops = [
        record
        for record in family_records
        if (
            isinstance(record.get("stop_reason"), str)
            and "early-stopping threshold" in str(record.get("stop_reason"))
        )
        or (
            isinstance(record.get("outcome_markdown"), str)
            and "early-stopping check" in str(record.get("outcome_markdown")).lower()
        )
        or (
            isinstance(record.get("outcome_markdown"), str)
            and "stopped at epoch" in str(record.get("outcome_markdown")).lower()
        )
    ]

    good_override_counts: dict[str, int] = {}
    bad_override_counts: dict[str, int] = {}
    for record in family_records:
        overrides = record.get("overrides")
        if not isinstance(overrides, dict):
            continue
        target = good_override_counts if bool(record.get("is_competitive")) or record.get("result_class") == "promoted" else bad_override_counts
        for key, value in sorted(overrides.items()):
            token = f"{key}={json.dumps(value, sort_keys=True)}"
            target[token] = target.get(token, 0) + 1

    lines = [
        f"# Experiment Compact: `{experiment_family}`",
        "",
        f"- Total logged records: `{len(family_records)}`",
        f"- Completed runs with metrics: `{len(completed_with_metric)}`",
        f"- Early-stopped runs: `{len(early_stops)}`",
    ]
    if best is not None:
        lines.append(
            f"- Current best: `{best['experiment_id']}` with `eval_mse_mean = {float(best['eval_mse_mean']):.4f}`"
        )
    lines.extend(["", "## What Worked"])
    if competitive:
        for record in competitive:
            lines.append(
                f"- `{record['experiment_id']}` reached `eval_mse_mean = {float(record['eval_mse_mean']):.4f}`. "
                f"Hypothesis: {record.get('hypothesis', 'n/a')}"
            )
    else:
        lines.append("- No competitive runs recorded yet.")
    if good_override_counts:
        lines.append("")
        lines.append("Signals seen in stronger runs:")
        for token, count in sorted(good_override_counts.items(), key=lambda item: (-item[1], item[0]))[:8]:
            lines.append(f"- `{token}` appeared `{count}` time(s) in competitive or promoted runs.")
    lines.extend(["", "## What Didn't"])
    if early_stops:
        for record in early_stops[:8]:
            lines.append(
                f"- `{record['experiment_id']}` stopped early with `eval_mse_mean = {float(record.get('eval_mse_mean')):.4f}`. "
                f"Reason: `{record.get('stop_reason')}`"
            )
    else:
        lines.append("- No early-stop failures recorded yet.")
    if bad_override_counts:
        lines.append("")
        lines.append("Repeated weak motifs:")
        for token, count in sorted(bad_override_counts.items(), key=lambda item: (-item[1], item[0]))[:8]:
            lines.append(f"- `{token}` appeared `{count}` time(s) in non-competitive runs.")
    lines.extend(
        [
            "",
            "## Guidance For Hypothesis Generation",
            "- Review the strongest competitive runs first before proposing new variants.",
            "- Treat growing boundary reflections and inward-propagating edge artifacts as a primary failure mode, not a cosmetic issue.",
            "- Slightly worse eval MSE can be acceptable if a competitive run materially reduces numerical artifacts.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_experiment_compact_markdown(
    *,
    log_path: Path,
    compact_path: Path,
    experiment_family: str,
) -> None:
    records = load_jsonl_records(log_path)
    compact_path.parent.mkdir(parents=True, exist_ok=True)
    compact_path.write_text(
        render_experiment_compact_markdown(experiment_family=experiment_family, records=records),
        encoding="utf-8",
    )


def write_ledger(path: Path, ledger: BatchLedger) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ledger.updated_at = _utcnow()
    payload = asdict(ledger)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def persist_batch_state(ledger_path: Path, summary_path: Path, ledger: BatchLedger) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    write_ledger(ledger_path, ledger)
    summary_path.write_text(render_batch_summary(ledger), encoding="utf-8")


def _record_candidate_result(
    *,
    ledger: BatchLedger,
    candidate: CandidateRunRecord,
    proposal: HypothesisProposal,
    incumbent_metrics: dict[str, Any] | None,
    promotion: PromotionPolicy,
    memory: FamilyMemory,
    memory_path: Path,
) -> None:
    result_class = classify_candidate_result(
        candidate,
        incumbent_metrics=incumbent_metrics,
        promotion=promotion,
    )
    candidate.result_class = result_class
    memory.tried_dedupe_keys.append(_serialize_dedupe_key(_proposal_dedupe_key(proposal)))
    memory.completed_runs.append(
        {
            "experiment_id": candidate.experiment_id,
            "hypothesis": candidate.hypothesis,
            "overrides": candidate.overrides,
            "result_class": result_class,
            "eval_mse_mean": candidate.eval_mse_mean,
            "stop_reason": candidate.stop_reason,
            "is_competitive": candidate.is_competitive,
            "final_step_heatmap_path": candidate.final_step_heatmap_path,
            "evaluator_payload_path": candidate.evaluator_payload_path,
            "evaluator_summary_path": candidate.evaluator_summary_path,
            "artifact_severity": candidate.artifact_severity,
            "artifact_tags": candidate.artifact_tags,
            "evaluator_notes": candidate.evaluator_notes,
            "accept_tradeoff_for_cleaner_rollout": candidate.accept_tradeoff_for_cleaner_rollout,
        }
    )
    memory.banned_override_pairs = derive_banned_override_pairs(memory)
    write_family_memory(memory_path, memory)


def render_batch_summary(ledger: BatchLedger) -> str:
    lines = [
        f"# Autonomous Batch `{ledger.batch_id}`",
        "",
        f"- Created: `{ledger.created_at}`",
        f"- Updated: `{ledger.updated_at}`",
        f"- Phase: `{ledger.phase}`",
        f"- Base config: `{ledger.base_config_path}`",
        f"- Experiment family: `{ledger.experiment_family}`",
        f"- LLM calls used: `{ledger.llm_calls_used}`",
        f"- Evaluator calls used: `{ledger.evaluator_calls_used}`",
        f"- Patch calls used: `{ledger.patch_calls_used}`",
    ]
    if ledger.failure_reason is not None:
        lines.append(f"- Failure reason: `{ledger.failure_reason}`")
    if ledger.failure_details is not None:
        lines.append(f"- Failure details: `{ledger.failure_details}`")
    lines.extend(["", "## Candidates"])
    if not ledger.candidates:
        lines.append("- No candidates materialized.")
    else:
        for candidate in ledger.candidates:
            metric = "n/a" if candidate.eval_mse_mean is None else f"{candidate.eval_mse_mean:.4f}"
            lines.append(
                f"- `{candidate.experiment_id}`: status=`{candidate.status}`, eval_mse_mean=`{metric}`, "
                f"ranking_score=`{candidate.ranking_score}`, result_class=`{candidate.result_class}`, "
                f"competitive=`{candidate.is_competitive}`, artifact_severity=`{candidate.artifact_severity}`, "
                f"stop_reason=`{candidate.stop_reason}`"
            )
    return "\n".join(lines) + "\n"


def run_autonomous_unet_experiment_loop(
    *,
    policy_path: str | Path,
    dry_run: bool = False,
    batch_id: str | None = None,
) -> dict[str, Any]:
    policy = load_autonomous_unet_policy(policy_path)
    resolved_batch_id = batch_id or timestamp_experiment_id()
    batch_dir = policy.output_root / resolved_batch_id
    ledger_path = batch_dir / "ledger.json"
    summary_path = batch_dir / "summary.md"
    memory_path = memory_path_for_policy(policy)
    base_config = load_unet_thickness_config(policy.base_config_path)
    incumbent_metrics = _load_metrics_payload(base_config.early_stopping_best_metrics_path)
    memory = load_family_memory(memory_path, experiment_family=policy.experiment_family)

    ledger = BatchLedger(
        batch_id=resolved_batch_id,
        created_at=_utcnow(),
        updated_at=_utcnow(),
        policy_path=str(policy_path),
        base_config_path=str(policy.base_config_path),
        experiment_family=policy.experiment_family,
        phase="preflight",
        summary_path=str(summary_path),
        failure_reason=None,
        failure_details=None,
    )
    append_stage_event(ledger, category="batch", message="Batch created.", phase="preflight")
    persist_batch_state(ledger_path, summary_path, ledger)

    ledger.preflight = run_preflight_checks(policy)
    append_stage_event(ledger, category="phase", message="Preflight checks completed.", phase="preflight", details=ledger.preflight)
    if policy.legacy_experiment_markdown_path is not None:
        backfilled = backfill_experiment_log_jsonl_from_markdown(
            policy.experiment_log_path,
            policy.legacy_experiment_markdown_path,
        )
        if backfilled:
            append_stage_event(
                ledger,
                category="log",
                message="Backfilled legacy markdown experiments into JSONL log.",
                phase="preflight",
                details={"backfilled_count": backfilled},
            )
            write_experiment_compact_markdown(
                log_path=policy.experiment_log_path,
                compact_path=policy.experiment_compact_path,
                experiment_family=policy.experiment_family,
            )
    persist_batch_state(ledger_path, summary_path, ledger)

    try:
        base_payload = _load_base_config_payload(policy.base_config_path)
        while True:
            remaining_train_runs = policy.batch.max_total_train_runs - len(ledger.candidates)
            if remaining_train_runs <= 0:
                break
            if ledger.llm_calls_used >= policy.batch.max_total_llm_calls:
                break

            ledger.phase = "proposal"
            append_stage_event(ledger, category="phase", message="Entering proposal generation.", phase="proposal")
            persist_batch_state(ledger_path, summary_path, ledger)
            proposals, llm_rejected = propose_hypotheses_via_openrouter(
                policy,
                base_config_payload=base_payload,
                incumbent_metrics=incumbent_metrics,
                memory=memory,
                max_hypotheses=min(policy.batch.max_hypotheses, remaining_train_runs),
            )
            ledger.llm_calls_used += 1
            append_stage_event(
                ledger,
                category="proposal",
                message="Received proposal batch from hypothesis model.",
                phase="proposal",
                details={"proposal_count": len(proposals), "rejected_count": len(llm_rejected)},
            )
            proposals, rejected = screen_proposals_against_memory(
                proposals,
                memory=memory,
                max_hypotheses=min(policy.batch.max_hypotheses, remaining_train_runs),
                allow_code_patches=policy.batch.allow_code_patches,
            )
            proposals, ranking_annotations = rank_proposals_against_memory(
                proposals,
                memory=memory,
                incumbent_metrics=incumbent_metrics,
                promotion=policy.promotion,
                max_hypotheses=min(policy.batch.max_hypotheses, remaining_train_runs),
            )
            append_stage_event(
                ledger,
                category="ranking",
                message="Ranked screened proposals.",
                phase="proposal",
                details={"retained_count": len(proposals), "screened_out_count": len(rejected)},
            )
            annotation_by_name = {
                str(entry["name"]): entry for entry in ranking_annotations
            }
            ledger.proposals.extend(
                {
                    **asdict(proposal),
                    "ranking_score": annotation_by_name.get(proposal.name, {}).get("ranking_score"),
                    "ranking_reasons": annotation_by_name.get(proposal.name, {}).get("ranking_reasons", []),
                }
                for proposal in proposals
            )
            ledger.screened_out.extend([*llm_rejected, *rejected])
            if ledger.llm_calls_used > policy.batch.max_total_llm_calls:
                raise RuntimeError("LLM call budget exceeded during proposal generation.")
            persist_batch_state(ledger_path, summary_path, ledger)

            if not proposals:
                continue

            ledger.phase = "materialize"
            append_stage_event(ledger, category="phase", message="Materializing candidate configs.", phase="materialize")
            new_candidates = materialize_candidate_configs(
                policy,
                proposals=proposals,
                batch_dir=batch_dir,
                start_index=len(ledger.candidates) + 1,
            )
            for record in new_candidates:
                ranking_entry = annotation_by_name.get(record.name)
                if ranking_entry is not None and isinstance(ranking_entry.get("ranking_score"), (int, float)):
                    record.ranking_score = float(ranking_entry["ranking_score"])
            ledger.candidates.extend(new_candidates)
            for record in new_candidates:
                append_stage_event(
                    ledger,
                    category="candidate",
                    message="Candidate materialized.",
                    phase="materialize",
                    experiment_id=record.experiment_id,
                    details={"name": record.name, "ranking_score": record.ranking_score},
                )
            persist_batch_state(ledger_path, summary_path, ledger)

            if dry_run or policy.batch.stop_after_generation:
                continue

            ledger.phase = "training"
            append_stage_event(ledger, category="phase", message="Entering training.", phase="training")
            persist_batch_state(ledger_path, summary_path, ledger)
            start_index = len(ledger.candidates) - len(new_candidates)
            for offset, proposal in enumerate(proposals):
                candidate_index = start_index + offset
                candidate = ledger.candidates[candidate_index]
                if candidate.implementation_type == "code" and ledger.patch_calls_used >= policy.batch.max_total_patch_calls:
                    candidate.status = "skipped_budget"
                    candidate.stop_reason = "Patch call budget exhausted."
                    ledger.candidates[candidate_index] = candidate
                    append_stage_event(
                        ledger,
                        category="candidate",
                        message="Candidate skipped due to patch budget.",
                        phase="training",
                        experiment_id=candidate.experiment_id,
                    )
                    persist_batch_state(ledger_path, summary_path, ledger)
                    continue

                candidate.status = "running"
                candidate.started_at = _utcnow()
                ledger.candidates[candidate_index] = candidate
                append_stage_event(
                    ledger,
                    category="candidate",
                    message="Candidate training started.",
                    phase="training",
                    experiment_id=candidate.experiment_id,
                    details={"ranking_score": candidate.ranking_score},
                )
                persist_batch_state(ledger_path, summary_path, ledger)

                ledger.candidates[candidate_index] = run_candidate_training(policy, record=candidate)
                ledger.patch_calls_used += ledger.candidates[candidate_index].patch_calls_used
                ledger.candidates[candidate_index] = reconcile_candidate_from_artifacts(ledger.candidates[candidate_index])
                append_stage_event(
                    ledger,
                    category="candidate",
                    message="Candidate training finished.",
                    phase="training",
                    experiment_id=ledger.candidates[candidate_index].experiment_id,
                    details={
                        "status": ledger.candidates[candidate_index].status,
                        "eval_mse_mean": ledger.candidates[candidate_index].eval_mse_mean,
                    },
                )
                ledger.candidates[candidate_index] = maybe_write_evaluator_artifacts(
                    policy,
                    candidate=ledger.candidates[candidate_index],
                    incumbent_metrics=incumbent_metrics,
                )
                if ledger.candidates[candidate_index].is_competitive:
                    append_stage_event(
                        ledger,
                        category="evaluator",
                        message="Competitive candidate produced evaluator artifacts.",
                        phase="training",
                        experiment_id=ledger.candidates[candidate_index].experiment_id,
                        details={"evaluator_payload_path": ledger.candidates[candidate_index].evaluator_payload_path},
                    )
                if (
                    ledger.candidates[candidate_index].is_competitive
                    and ledger.evaluator_calls_used < policy.evaluation.max_total_calls
                    and ledger.llm_calls_used < policy.batch.max_total_llm_calls
                ):
                    try:
                        ledger.candidates[candidate_index] = maybe_run_competitive_evaluator(
                            policy,
                            candidate=ledger.candidates[candidate_index],
                            batch_dir=batch_dir,
                        )
                        ledger.evaluator_calls_used += 1
                        ledger.llm_calls_used += 1
                        append_stage_event(
                            ledger,
                            category="evaluator",
                            message="Evaluator review completed.",
                            phase="training",
                            experiment_id=ledger.candidates[candidate_index].experiment_id,
                            details={
                                "artifact_severity": ledger.candidates[candidate_index].artifact_severity,
                                "accept_tradeoff_for_cleaner_rollout": ledger.candidates[candidate_index].accept_tradeoff_for_cleaner_rollout,
                            },
                        )
                    except Exception as exc:
                        ledger.candidates[candidate_index].evaluator_notes = f"Evaluator failed: {exc}"
                        append_stage_event(
                            ledger,
                            category="evaluator",
                            message="Evaluator review failed.",
                            phase="training",
                            experiment_id=ledger.candidates[candidate_index].experiment_id,
                            details={"error": str(exc)},
                        )
                _record_candidate_result(
                    ledger=ledger,
                    candidate=ledger.candidates[candidate_index],
                    proposal=proposal,
                    incumbent_metrics=incumbent_metrics,
                    promotion=policy.promotion,
                    memory=memory,
                    memory_path=memory_path,
                )
                append_experiment_log_record(
                    policy.experiment_log_path,
                    batch_id=resolved_batch_id,
                    experiment_family=policy.experiment_family,
                    candidate=ledger.candidates[candidate_index],
                )
                write_experiment_compact_markdown(
                    log_path=policy.experiment_log_path,
                    compact_path=policy.experiment_compact_path,
                    experiment_family=policy.experiment_family,
                )
                persist_batch_state(ledger_path, summary_path, ledger)

        ledger.phase = "dry_run_complete" if (dry_run or policy.batch.stop_after_generation) else "complete"
        append_stage_event(ledger, category="batch", message="Batch completed.", phase=ledger.phase)
    except BaseException as exc:
        ledger.phase = "aborted"
        ledger.failure_reason = type(exc).__name__
        ledger.failure_details = str(exc)
        append_stage_event(
            ledger,
            category="batch",
            message="Batch aborted.",
            phase="aborted",
            details={"failure_reason": ledger.failure_reason, "failure_details": ledger.failure_details},
        )
        raise
    finally:
        for index, candidate in enumerate(ledger.candidates):
            reconciled = reconcile_candidate_from_artifacts(candidate)
            if reconciled.status == "running" and reconciled.return_code is None and reconciled.metrics_path is None:
                reconciled.status = "aborted"
                if reconciled.finished_at is None:
                    reconciled.finished_at = _utcnow()
            ledger.candidates[index] = reconciled
        persist_batch_state(ledger_path, summary_path, ledger)
        if policy.experiment_log_path.exists():
            write_experiment_compact_markdown(
                log_path=policy.experiment_log_path,
                compact_path=policy.experiment_compact_path,
                experiment_family=policy.experiment_family,
            )
            persist_batch_state(ledger_path, summary_path, ledger)
    return {"batch_id": resolved_batch_id, "ledger_path": str(ledger_path), "summary_path": str(summary_path)}
