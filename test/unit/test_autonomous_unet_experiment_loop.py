from __future__ import annotations

import json
from pathlib import Path

import xarray as xr

from src.models.unet_thickness.config import load_unet_thickness_config
from src.pipelines.autonomous_unet_experiment_loop import (
    AutonomousLoopPolicy,
    BatchLedger,
    FamilyMemory,
    HypothesisProposal,
    _parse_hypothesis_response,
    append_experiment_log_record,
    backfill_experiment_log_jsonl_from_markdown,
    dedupe_and_limit_proposals,
    derive_banned_override_pairs,
    load_autonomous_unet_policy,
    load_jsonl_records,
    memory_path_for_policy,
    materialize_candidate_configs,
    maybe_write_evaluator_artifacts,
    persist_batch_state,
    rank_proposals_against_memory,
    reconcile_candidate_from_artifacts,
    render_experiment_compact_markdown,
    render_batch_summary,
    run_autonomous_unet_experiment_loop,
    screen_proposals_against_memory,
    write_experiment_compact_markdown,
)


def test_load_autonomous_unet_policy_reads_defaults():
    policy = load_autonomous_unet_policy("config/autoloop/default.yaml")

    assert isinstance(policy, AutonomousLoopPolicy)
    assert policy.batch.max_hypotheses == 8
    assert policy.batch.allow_code_patches is True
    assert policy.llm.max_total_calls == 8
    assert policy.llm.model_name == "openai/gpt-5.3-codex"
    assert policy.evaluation.competitive_within_ratio == 0.2
    assert policy.experiment_log_path.name.endswith(".jsonl")
    assert policy.experiment_compact_path.name.endswith(".md")
    assert memory_path_for_policy(policy).name == "double_gyre_shifting_wind_2layer.json"


def test_dedupe_and_limit_proposals_filters_duplicate_override_sets():
    proposals = [
        HypothesisProposal(name="a", hypothesis="h1", overrides={"hidden_channels": 24}),
        HypothesisProposal(name="b", hypothesis="h2", overrides={"hidden_channels": 24}),
        HypothesisProposal(name="c", hypothesis="h3", overrides={"stage_depth": 2}),
    ]

    retained = dedupe_and_limit_proposals(proposals, max_hypotheses=2)

    assert [proposal.name for proposal in retained] == ["a", "c"]


def test_materialize_candidate_configs_writes_hypothesis_and_overrides(tmp_path: Path):
    policy = load_autonomous_unet_policy("config/autoloop/default.yaml")
    batch_dir = tmp_path / "20260410T120000"
    proposals = [
        HypothesisProposal(
            name="Gated Skip",
            hypothesis="Test gated skip fusion.",
            overrides={"skip_fusion_mode": "gated"},
        )
    ]

    records = materialize_candidate_configs(policy, proposals=proposals, batch_dir=batch_dir)

    assert len(records) == 1
    config_path = Path(records[0].config_path)
    payload = json.loads(json.dumps(__import__("yaml").safe_load(config_path.read_text(encoding="utf-8"))))
    assert payload["hypothesis"] == "Test gated skip fusion."
    assert payload["skip_fusion_mode"] == "gated"
    assert records[0].experiment_id.endswith("gated-skip")
    assert records[0].search_mode == "exploit"
    assert records[0].hypothesis_family == "optimizer_tuning"


def test_materialize_candidate_configs_respects_start_index(tmp_path: Path):
    policy = load_autonomous_unet_policy("config/autoloop/default.yaml")
    proposals = [
        HypothesisProposal(name="A", hypothesis="h1", overrides={"hidden_channels": 24}),
        HypothesisProposal(name="B", hypothesis="h2", overrides={"hidden_channels": 32}),
    ]

    records = materialize_candidate_configs(
        policy,
        proposals=proposals,
        batch_dir=tmp_path / "20260410T120000",
        start_index=3,
    )

    assert records[0].experiment_id.startswith("20260410T120000-03-")
    assert records[1].experiment_id.startswith("20260410T120000-04-")


def test_render_batch_summary_lists_candidates():
    ledger = BatchLedger(
        batch_id="20260410T120000",
        created_at="2026-04-10T12:00:00+00:00",
        updated_at="2026-04-10T12:10:00+00:00",
        policy_path="config/autoloop/default.yaml",
        base_config_path="config/base.yaml",
        experiment_family="double_gyre_shifting_wind_2layer",
        phase="complete",
    )
    ledger.candidates = []

    summary = render_batch_summary(ledger)

    assert "Autonomous Batch" in summary
    assert "No candidates materialized." in summary


def test_screen_proposals_against_memory_rejects_duplicates_and_banned_pairs():
    memory = FamilyMemory(
        experiment_family="double_gyre_shifting_wind_2layer",
        updated_at="2026-04-10T12:00:00+00:00",
        tried_dedupe_keys=[[["hidden_channels", "24"]]],
        banned_override_pairs=[["stage_depth", "3"]],
    )
    proposals = [
        HypothesisProposal(name="repeat", hypothesis="repeat", overrides={"hidden_channels": 24}),
        HypothesisProposal(name="banned", hypothesis="banned", overrides={"stage_depth": 3}),
        HypothesisProposal(name="keep", hypothesis="keep", overrides={"skip_fusion_mode": "gated"}),
    ]

    retained, rejected = screen_proposals_against_memory(
        proposals,
        memory=memory,
        max_hypotheses=3,
        allow_code_patches=False,
    )
    assert [proposal.name for proposal in retained] == ["keep"]
    assert [entry["reason"] for entry in rejected] == ["duplicate_in_memory", "banned_override_pair"]


def test_screen_proposals_against_memory_marks_code_hypotheses_when_patching_disabled():
    memory = FamilyMemory(
        experiment_family="double_gyre_shifting_wind_2layer",
        updated_at="2026-04-10T12:00:00+00:00",
    )
    proposals = [
        HypothesisProposal(
            name="code-change",
            hypothesis="Adjust training loop logic.",
            implementation_type="code",
            patch_targets=["src/models/unet_thickness/training.py"],
            patch_plan="Add an alternate curriculum scheduler.",
        )
    ]

    retained, rejected = screen_proposals_against_memory(
        proposals,
        memory=memory,
        max_hypotheses=3,
        allow_code_patches=False,
    )

    assert retained == []
    assert rejected[0]["reason"] == "requires_code_patch"


def test_rank_proposals_against_memory_prefers_strong_motifs_and_penalizes_weak_ones():
    memory = FamilyMemory(
        experiment_family="double_gyre_shifting_wind_2layer",
        updated_at="2026-04-10T12:00:00+00:00",
        completed_runs=[
            {
                "experiment_id": "good-1",
                "overrides": {"num_levels": 3},
                "result_class": "promoted",
                "eval_mse_mean": 46.4,
            },
            {
                "experiment_id": "bad-1",
                "overrides": {"stage_depth": 3},
                "result_class": "negative_result",
                "eval_mse_mean": 221.5,
                "artifact_severity": 5,
            },
        ],
    )
    proposals = [
        HypothesisProposal(name="bad", hypothesis="bad", overrides={"stage_depth": 3}),
        HypothesisProposal(name="good", hypothesis="good", overrides={"num_levels": 3}),
    ]

    retained, annotations = rank_proposals_against_memory(
        proposals,
        memory=memory,
        incumbent_metrics={"eval_mse_mean": 50.0},
        promotion=load_autonomous_unet_policy("config/autoloop/default.yaml").promotion,
        max_hypotheses=2,
    )

    assert [proposal.name for proposal in retained] == ["good", "bad"]
    assert annotations[0]["ranking_score"] >= annotations[1]["ranking_score"]


def test_derive_banned_override_pairs_promotes_repeat_negative_motifs_only():
    memory = FamilyMemory(
        experiment_family="double_gyre_shifting_wind_2layer",
        updated_at="2026-04-10T12:00:00+00:00",
        completed_runs=[
            {"overrides": {"stage_depth": 3}, "result_class": "negative_result"},
            {"overrides": {"stage_depth": 3}, "result_class": "negative_result"},
            {"overrides": {"skip_fusion_mode": "gated"}, "result_class": "promoted"},
            {"overrides": {"skip_fusion_mode": "gated"}, "result_class": "negative_result"},
        ],
    )

    banned = derive_banned_override_pairs(memory)

    assert banned == [["stage_depth", "3"]]


def test_backfill_experiment_log_jsonl_from_markdown_and_compact(tmp_path: Path):
    markdown_path = tmp_path / "experiments.md"
    markdown_path.write_text(
        "# Experiments\n\n"
        "| Timestamp | Branch | Commit | Hypothesis | Outcome |\n"
        "| --- | --- | --- | --- | --- |\n"
        "| `batch-01-a` | `main` | `old` | old hypothesis | Autonomous batch result. The run wrote [`metrics.json`](/tmp/metrics.json) with `eval_mse_mean = 12.5000`. Config: [`cfg.yaml`](/tmp/cfg.yaml). |\n",
        encoding="utf-8",
    )
    log_path = tmp_path / "experiments.jsonl"

    appended = backfill_experiment_log_jsonl_from_markdown(log_path, markdown_path)

    assert appended == 1
    records = load_jsonl_records(log_path)
    assert records[0]["experiment_id"] == "batch-01-a"
    compact = render_experiment_compact_markdown(
        experiment_family="double_gyre_shifting_wind_2layer",
        records=[{**records[0], "experiment_family": "double_gyre_shifting_wind_2layer"}],
    )
    assert "Current best" in compact


def test_append_experiment_log_record_writes_jsonl_entry(tmp_path: Path):
    candidate = materialize_candidate_configs(
        load_autonomous_unet_policy("config/autoloop/default.yaml"),
        proposals=[HypothesisProposal(name="A", hypothesis="New hypothesis", overrides={"hidden_channels": 24})],
        batch_dir=tmp_path / "batch-01",
    )[0]
    candidate.status = "completed"
    candidate.branch_name = "autoloop/batch-01-a"
    candidate.branch_commit = "abc123"
    candidate.eval_mse_mean = 12.5
    candidate.result_class = "promoted"
    candidate.is_competitive = True
    candidate.metrics_path = str(tmp_path / "metrics.json")
    Path(candidate.metrics_path).write_text("{}", encoding="utf-8")
    log_path = tmp_path / "experiments.jsonl"

    append_experiment_log_record(
        log_path,
        batch_id="batch-01",
        experiment_family="double_gyre_shifting_wind_2layer",
        candidate=candidate,
    )

    records = load_jsonl_records(log_path)
    assert records[0]["branch_name"] == "autoloop/batch-01-a"
    assert records[0]["branch_commit"] == "abc123"
    assert records[0]["hypothesis"] == "New hypothesis"
    assert records[0]["is_competitive"] is True


def test_parse_hypothesis_response_rejects_invalid_proposals_without_failing_batch():
    payload = {
        "hypotheses": [
            {
                "name": "invalid",
                "hypothesis": "Uses unsupported curriculum keys.",
                "overrides": {"curriculum_rollout_steps": [1, 2, 3]},
            },
            {
                "name": "valid",
                "hypothesis": "Uses a supported skip fusion change.",
                "implementation_type": "config",
                "search_mode": "explore",
                "hypothesis_family": "skip_connection_design",
                "overrides": {"skip_fusion_mode": "gated"},
            },
            {
                "name": "code_valid",
                "hypothesis": "Needs a training-loop change.",
                "implementation_type": "code",
                "search_mode": "artifact_hardening",
                "hypothesis_family": "rollout_stabilization",
                "patch_targets": ["src/models/unet_thickness/training.py"],
                "patch_plan": "Add an alternate residual scheduler.",
            },
        ]
    }

    proposals, rejected = _parse_hypothesis_response(json.dumps(payload))

    assert [proposal.name for proposal in proposals] == ["valid", "code_valid"]
    assert proposals[1].implementation_type == "code"
    assert proposals[0].search_mode == "explore"
    assert proposals[0].hypothesis_family == "skip_connection_design"
    assert proposals[1].search_mode == "artifact_hardening"
    assert proposals[1].hypothesis_family == "rollout_stabilization"
    assert len(rejected) == 1
    assert rejected[0]["reason"] == "invalid_llm_proposal"
    assert "Unsupported override keys" in rejected[0]["details"]


def test_materialize_candidate_configs_preserves_research_mode_fields(tmp_path: Path):
    policy = load_autonomous_unet_policy("config/autoloop/default.yaml")
    proposals = [
        HypothesisProposal(
            name="Boundary Idea",
            hypothesis="Test a boundary-focused idea.",
            search_mode="artifact_hardening",
            hypothesis_family="boundary_condition_handling",
            overrides={"skip_fusion_mode": "gated"},
        )
    ]

    record = materialize_candidate_configs(
        policy,
        proposals=proposals,
        batch_dir=tmp_path / "20260410T120000",
    )[0]

    assert record.search_mode == "artifact_hardening"
    assert record.hypothesis_family == "boundary_condition_handling"


def test_reconcile_candidate_from_artifacts_marks_completed_from_metrics(tmp_path: Path):
    policy = load_autonomous_unet_policy("config/autoloop/default.yaml")
    candidate = materialize_candidate_configs(
        policy,
        proposals=[HypothesisProposal(name="A", hypothesis="hyp", overrides={"hidden_channels": 24})],
        batch_dir=tmp_path / "batch",
    )[0]
    config = load_unet_thickness_config(candidate.config_path).with_overrides(experiment_id=candidate.experiment_id)
    config.metrics_path.parent.mkdir(parents=True, exist_ok=True)
    config.metrics_path.write_text(
        json.dumps(
            {
                "eval_mse_mean": 12.5,
                "stop_reason": "Stopped by threshold.",
                "updated_at": "2026-04-10T12:34:56+00:00",
            }
        ),
        encoding="utf-8",
    )

    candidate.status = "running"
    reconciled = reconcile_candidate_from_artifacts(candidate)

    assert reconciled.status == "completed"
    assert reconciled.eval_mse_mean == 12.5
    assert reconciled.stop_reason == "Stopped by threshold."
    assert reconciled.metrics_path is not None
    assert reconciled.finished_at == "2026-04-10T12:34:56+00:00"


def test_persist_batch_state_writes_summary_and_ledger(tmp_path: Path):
    ledger = BatchLedger(
        batch_id="20260410T120000",
        created_at="2026-04-10T12:00:00+00:00",
        updated_at="2026-04-10T12:00:00+00:00",
        policy_path="config/autoloop/default.yaml",
        base_config_path="config/base.yaml",
        experiment_family="double_gyre_shifting_wind_2layer",
        phase="training",
    )
    ledger_path = tmp_path / "ledger.json"
    summary_path = tmp_path / "summary.md"

    # Reuse a real candidate record so summary rendering matches runtime behavior.
    policy = load_autonomous_unet_policy("config/autoloop/default.yaml")
    ledger.candidates = materialize_candidate_configs(
        policy,
        proposals=[HypothesisProposal(name="A", hypothesis="hyp", overrides={"hidden_channels": 24})],
        batch_dir=tmp_path / "batch",
    )

    persist_batch_state(ledger_path, summary_path, ledger)

    assert ledger_path.exists()
    assert summary_path.exists()
    assert "Autonomous Batch" in summary_path.read_text(encoding="utf-8")


def test_maybe_write_evaluator_artifacts_for_competitive_run(tmp_path: Path):
    policy_path = tmp_path / "policy.yaml"
    policy_path.write_text(
        "\n".join(
            [
                "base_config_path: config/emulator/unet_thickness_shifting_wind_2layer_window250_convnext_multistep2_dilated_bilinear_addskip_lr3p8e4_wd1p8e5_latecurr_residual_skydiscover.yaml",
                "experiment_family: double_gyre_shifting_wind_2layer",
                f"experiment_log_path: {tmp_path / 'experiments.jsonl'}",
                f"experiment_compact_path: {tmp_path / 'experiments_compact.md'}",
                "python_executable: .venv/bin/python",
                f"output_root: {tmp_path / 'autoloop'}",
                "llm:",
                "  env_var_name: UNUSED",
                "  model_name: openai/gpt-5.3-codex",
                "  api_base: https://openrouter.ai/api/v1",
                "  temperature: 0.3",
                "  max_tokens: 1000",
                "  timeout_seconds: 30",
                "  max_total_calls: 1",
                "batch:",
                "  max_hypotheses: 1",
                "  max_total_train_runs: 1",
                "  max_total_llm_calls: 1",
                "  max_total_patch_calls: 0",
                "  sequential: true",
                "  allow_code_patches: false",
                "  stop_after_generation: false",
                "patching:",
                "  max_patch_attempts_per_hypothesis: 0",
                "  max_repair_attempts_per_patch: 0",
                "training:",
                "  require_benchmark_path: false",
                "  require_preflight_tests: false",
                "  preflight_test_paths: []",
                "promotion:",
                "  primary_metric: eval_mse_mean",
                "  maximize: false",
                "  must_beat_incumbent_by: 0.0",
                "evaluation:",
                "  competitive_within_ratio: 0.2",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    policy = load_autonomous_unet_policy(policy_path)
    candidate = materialize_candidate_configs(
        policy,
        proposals=[HypothesisProposal(name="A", hypothesis="hyp", overrides={"hidden_channels": 24})],
        batch_dir=tmp_path / "batch",
    )[0]
    config = load_unet_thickness_config(candidate.config_path).with_overrides(experiment_id=candidate.experiment_id)
    config.raw_experiment_dir.mkdir(parents=True, exist_ok=True)
    xr.Dataset(
        data_vars={
            "truth_layer_thickness": (("time_days", "y", "x"), [[[1.0, 2.0], [3.0, 4.0]]]),
            "rollout_layer_thickness": (("time_days", "y", "x"), [[[1.1, 1.9], [3.2, 3.8]]]),
        },
        coords={"time_days": [1.0], "y": [0.0, 1.0], "x": [0.0, 1.0]},
    ).to_netcdf(config.rollout_path)
    config.metrics_path.write_text(json.dumps({"eval_mse_mean": 54.0}), encoding="utf-8")

    candidate.status = "completed"
    candidate.eval_mse_mean = 54.0
    candidate.metrics_path = str(config.metrics_path)

    updated = maybe_write_evaluator_artifacts(
        policy,
        candidate=candidate,
        incumbent_metrics={"eval_mse_mean": 50.0},
    )

    assert updated.is_competitive is True
    assert updated.final_step_heatmap_path is not None
    assert Path(updated.final_step_heatmap_path).exists()
    assert updated.evaluator_payload_path is not None
    payload = json.loads(Path(updated.evaluator_payload_path).read_text(encoding="utf-8"))
    assert payload["candidate_primary_metric_value"] == 54.0
    assert "boundary reflections" in payload["guidance"]


def test_run_autonomous_loop_continues_generating_until_train_budget_filled(tmp_path: Path, monkeypatch):
    policy_path = tmp_path / "policy.yaml"
    policy_path.write_text(
        "\n".join(
            [
                "base_config_path: config/emulator/unet_thickness_shifting_wind_2layer_window250_convnext_multistep2_dilated_bilinear_addskip_lr3p8e4_wd1p8e5_latecurr_residual_skydiscover.yaml",
                "experiment_family: double_gyre_shifting_wind_2layer",
                f"experiment_log_path: {tmp_path / 'experiments.jsonl'}",
                f"experiment_compact_path: {tmp_path / 'experiments_compact.md'}",
                "python_executable: .venv/bin/python",
                f"output_root: {tmp_path / 'autoloop'}",
                "llm:",
                "  env_var_name: UNUSED",
                "  model_name: openai/gpt-5.3-codex",
                "  api_base: https://openrouter.ai/api/v1",
                "  temperature: 0.3",
                "  max_tokens: 1000",
                "  timeout_seconds: 30",
                "  max_total_calls: 3",
                "batch:",
                "  max_hypotheses: 2",
                "  max_total_train_runs: 2",
                "  max_total_llm_calls: 3",
                "  max_total_patch_calls: 0",
                "  sequential: true",
                "  allow_code_patches: false",
                "  stop_after_generation: false",
                "patching:",
                "  max_patch_attempts_per_hypothesis: 0",
                "  max_repair_attempts_per_patch: 0",
                "training:",
                "  require_benchmark_path: true",
                "  require_preflight_tests: false",
                "  preflight_test_paths: []",
                "promotion:",
                "  primary_metric: eval_mse_mean",
                "  maximize: false",
                "  must_beat_incumbent_by: 0.0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "experiments.jsonl").write_text("", encoding="utf-8")

    proposal_rounds = iter(
        [
            ([HypothesisProposal(name="Round1", hypothesis="h1", overrides={"hidden_channels": 24})], []),
            ([HypothesisProposal(name="Round2", hypothesis="h2", overrides={"hidden_channels": 32})], []),
        ]
    )

    monkeypatch.setattr(
        "src.pipelines.autonomous_unet_experiment_loop.propose_hypotheses_via_openrouter",
        lambda *args, **kwargs: next(proposal_rounds),
    )
    monkeypatch.setattr(
        "src.pipelines.autonomous_unet_experiment_loop.run_preflight_checks",
        lambda policy: {"ok": True},
    )
    monkeypatch.setattr(
        "src.pipelines.autonomous_unet_experiment_loop.write_experiment_compact_markdown",
        lambda *args, **kwargs: None,
    )

    def fake_run_candidate_training(policy, *, record):
        return type(record)(
            **{
                **record.__dict__,
                "status": "completed",
                "return_code": 0,
                "finished_at": "2026-04-10T12:00:00+00:00",
                "duration_seconds": 1.0,
                "eval_mse_mean": 100.0,
                "stop_reason": "negative",
                "metrics_path": str(tmp_path / f"{record.experiment_id}_metrics.json"),
            }
        )

    monkeypatch.setattr(
        "src.pipelines.autonomous_unet_experiment_loop.run_candidate_training",
        fake_run_candidate_training,
    )
    monkeypatch.setattr(
        "src.pipelines.autonomous_unet_experiment_loop.reconcile_candidate_from_artifacts",
        lambda record: record,
    )
    monkeypatch.setattr(
        "src.pipelines.autonomous_unet_experiment_loop.maybe_write_evaluator_artifacts",
        lambda policy, candidate, incumbent_metrics: candidate,
    )

    payload = run_autonomous_unet_experiment_loop(policy_path=policy_path, batch_id="batch-01")
    ledger = json.loads(Path(payload["ledger_path"]).read_text(encoding="utf-8"))

    assert ledger["phase"] == "complete"
    assert ledger["llm_calls_used"] == 2
    assert len(ledger["candidates"]) == 2
    assert ledger["candidates"][0]["experiment_id"].startswith("batch-01-01-")
    assert ledger["candidates"][1]["experiment_id"].startswith("batch-01-02-")


def test_run_autonomous_loop_records_evaluator_feedback_for_competitive_runs(tmp_path: Path, monkeypatch):
    policy_path = tmp_path / "policy.yaml"
    policy_path.write_text(
        "\n".join(
            [
                "base_config_path: config/emulator/unet_thickness_shifting_wind_2layer_window250_convnext_multistep2_dilated_bilinear_addskip_lr3p8e4_wd1p8e5_latecurr_residual_skydiscover.yaml",
                "experiment_family: double_gyre_shifting_wind_2layer",
                f"experiment_log_path: {tmp_path / 'experiments.jsonl'}",
                f"experiment_compact_path: {tmp_path / 'experiments_compact.md'}",
                "python_executable: .venv/bin/python",
                f"output_root: {tmp_path / 'autoloop'}",
                "llm:",
                "  env_var_name: UNUSED",
                "  model_name: openai/gpt-5.3-codex",
                "  api_base: https://openrouter.ai/api/v1",
                "  temperature: 0.3",
                "  max_tokens: 1000",
                "  timeout_seconds: 30",
                "  max_total_calls: 3",
                "batch:",
                "  max_hypotheses: 1",
                "  max_total_train_runs: 1",
                "  max_total_llm_calls: 3",
                "  max_total_patch_calls: 0",
                "  sequential: true",
                "  allow_code_patches: false",
                "  stop_after_generation: false",
                "patching:",
                "  max_patch_attempts_per_hypothesis: 0",
                "  max_repair_attempts_per_patch: 0",
                "training:",
                "  require_benchmark_path: true",
                "  require_preflight_tests: false",
                "  preflight_test_paths: []",
                "promotion:",
                "  primary_metric: eval_mse_mean",
                "  maximize: false",
                "  must_beat_incumbent_by: 0.0",
                "evaluation:",
                "  competitive_within_ratio: 0.2",
                "  max_total_calls: 1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "experiments.jsonl").write_text("", encoding="utf-8")
    monkeypatch.setattr(
        "src.pipelines.autonomous_unet_experiment_loop.run_preflight_checks",
        lambda policy: {"ok": True},
    )
    monkeypatch.setattr(
        "src.pipelines.autonomous_unet_experiment_loop.propose_hypotheses_via_openrouter",
        lambda *args, **kwargs: ([HypothesisProposal(name="Round1", hypothesis="h1", overrides={"hidden_channels": 24})], []),
    )
    monkeypatch.setattr(
        "src.pipelines.autonomous_unet_experiment_loop.run_candidate_training",
        lambda policy, *, record: type(record)(
            **{
                **record.__dict__,
                "status": "completed",
                "return_code": 0,
                "finished_at": "2026-04-10T12:00:00+00:00",
                "duration_seconds": 1.0,
                "eval_mse_mean": 54.0,
                "stop_reason": "done",
                "metrics_path": str(tmp_path / f"{record.experiment_id}_metrics.json"),
            }
        ),
    )
    monkeypatch.setattr(
        "src.pipelines.autonomous_unet_experiment_loop.reconcile_candidate_from_artifacts",
        lambda record: record,
    )
    monkeypatch.setattr(
        "src.pipelines.autonomous_unet_experiment_loop.maybe_write_evaluator_artifacts",
        lambda policy, candidate, incumbent_metrics: type(candidate)(
            **{
                **candidate.__dict__,
                "is_competitive": True,
                "evaluator_payload_path": str(tmp_path / "payload.json"),
            }
        ),
    )
    monkeypatch.setattr(
        "src.pipelines.autonomous_unet_experiment_loop.maybe_run_competitive_evaluator",
        lambda policy, candidate, batch_dir: type(candidate)(
            **{
                **candidate.__dict__,
                "artifact_severity": 2,
                "artifact_tags": ["boundary_reflection_growth"],
                "evaluator_notes": "Prefer cleaner boundaries.",
                "accept_tradeoff_for_cleaner_rollout": True,
                "evaluator_summary_path": str(tmp_path / "evaluation.json"),
            }
        ),
    )

    payload = run_autonomous_unet_experiment_loop(policy_path=policy_path, batch_id="batch-eval")
    ledger = json.loads(Path(payload["ledger_path"]).read_text(encoding="utf-8"))

    assert ledger["evaluator_calls_used"] == 1
    candidate = ledger["candidates"][0]
    assert candidate["artifact_severity"] == 2
    assert candidate["accept_tradeoff_for_cleaner_rollout"] is True


def test_run_autonomous_loop_preserves_research_fields_and_stage_events(tmp_path: Path, monkeypatch):
    policy_path = tmp_path / "policy.yaml"
    policy_path.write_text(
        "\n".join(
            [
                "base_config_path: config/emulator/unet_thickness_shifting_wind_2layer_window250_convnext_multistep2_dilated_bilinear_addskip_lr3p8e4_wd1p8e5_latecurr_residual_skydiscover.yaml",
                "experiment_family: double_gyre_shifting_wind_2layer",
                f"experiment_log_path: {tmp_path / 'experiments.jsonl'}",
                f"experiment_compact_path: {tmp_path / 'experiments_compact.md'}",
                "python_executable: .venv/bin/python",
                f"output_root: {tmp_path / 'autoloop'}",
                "llm:",
                "  env_var_name: UNUSED",
                "  model_name: openai/gpt-5.3-codex",
                "  api_base: https://openrouter.ai/api/v1",
                "  temperature: 0.3",
                "  max_tokens: 1000",
                "  timeout_seconds: 30",
                "  max_total_calls: 3",
                "batch:",
                "  max_hypotheses: 1",
                "  max_total_train_runs: 1",
                "  max_total_llm_calls: 3",
                "  max_total_patch_calls: 0",
                "  sequential: true",
                "  allow_code_patches: false",
                "  stop_after_generation: false",
                "patching:",
                "  max_patch_attempts_per_hypothesis: 0",
                "  max_repair_attempts_per_patch: 0",
                "training:",
                "  require_benchmark_path: true",
                "  require_preflight_tests: false",
                "  preflight_test_paths: []",
                "promotion:",
                "  primary_metric: eval_mse_mean",
                "  maximize: false",
                "  must_beat_incumbent_by: 0.0",
                "evaluation:",
                "  competitive_within_ratio: 0.2",
                "  max_total_calls: 0",
                "search:",
                "  round_modes: [explore]",
                "  explore_max_per_family: 2",
                "  promising_within_ratio: 0.35",
                "  max_optimizer_only_proposals_per_explore_round: 1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "experiments.jsonl").write_text("", encoding="utf-8")
    monkeypatch.setattr(
        "src.pipelines.autonomous_unet_experiment_loop.run_preflight_checks",
        lambda policy: {"ok": True},
    )
    monkeypatch.setattr(
        "src.pipelines.autonomous_unet_experiment_loop.propose_hypotheses_via_openrouter",
        lambda *args, **kwargs: (
            [
                HypothesisProposal(
                    name="Boundary Idea",
                    hypothesis="Test a more exploratory boundary idea.",
                    search_mode="explore",
                    hypothesis_family="boundary_condition_handling",
                    overrides={"skip_fusion_mode": "gated"},
                )
            ],
            [],
        ),
    )
    monkeypatch.setattr(
        "src.pipelines.autonomous_unet_experiment_loop.run_candidate_training",
        lambda policy, *, record: type(record)(
            **{
                **record.__dict__,
                "status": "completed",
                "return_code": 0,
                "finished_at": "2026-04-10T12:00:00+00:00",
                "duration_seconds": 1.0,
                "eval_mse_mean": 80.0,
                "stop_reason": "done",
                "metrics_path": str(tmp_path / f"{record.experiment_id}_metrics.json"),
            }
        ),
    )
    monkeypatch.setattr(
        "src.pipelines.autonomous_unet_experiment_loop.reconcile_candidate_from_artifacts",
        lambda record: record,
    )
    monkeypatch.setattr(
        "src.pipelines.autonomous_unet_experiment_loop.maybe_write_evaluator_artifacts",
        lambda policy, candidate, incumbent_metrics: candidate,
    )

    payload = run_autonomous_unet_experiment_loop(policy_path=policy_path, batch_id="batch-research")
    ledger = json.loads(Path(payload["ledger_path"]).read_text(encoding="utf-8"))

    candidate = ledger["candidates"][0]
    assert candidate["search_mode"] == "explore"
    assert candidate["hypothesis_family"] == "boundary_condition_handling"
    categories = [event["category"] for event in ledger["stage_events"]]
    assert "proposal" in categories
    assert "ranking" in categories
    assert "candidate" in categories


def test_run_autonomous_loop_records_failure_details_on_abort(tmp_path: Path, monkeypatch):
    policy_path = tmp_path / "policy.yaml"
    policy_path.write_text(
        "\n".join(
            [
                "base_config_path: config/emulator/unet_thickness_shifting_wind_2layer_window250_convnext_multistep2_dilated_bilinear_addskip_lr3p8e4_wd1p8e5_latecurr_residual_skydiscover.yaml",
                "experiment_family: double_gyre_shifting_wind_2layer",
                f"experiment_log_path: {tmp_path / 'experiments.jsonl'}",
                f"experiment_compact_path: {tmp_path / 'experiments_compact.md'}",
                "python_executable: .venv/bin/python",
                f"output_root: {tmp_path / 'autoloop'}",
                "llm:",
                "  env_var_name: UNUSED",
                "  model_name: openai/gpt-5.3-codex",
                "  api_base: https://openrouter.ai/api/v1",
                "  temperature: 0.3",
                "  max_tokens: 1000",
                "  timeout_seconds: 30",
                "  max_total_calls: 1",
                "batch:",
                "  max_hypotheses: 1",
                "  max_total_train_runs: 1",
                "  max_total_llm_calls: 1",
                "  max_total_patch_calls: 0",
                "  sequential: true",
                "  allow_code_patches: false",
                "  stop_after_generation: false",
                "patching:",
                "  max_patch_attempts_per_hypothesis: 0",
                "  max_repair_attempts_per_patch: 0",
                "training:",
                "  require_benchmark_path: true",
                "  require_preflight_tests: false",
                "  preflight_test_paths: []",
                "promotion:",
                "  primary_metric: eval_mse_mean",
                "  maximize: false",
                "  must_beat_incumbent_by: 0.0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "experiments.jsonl").write_text("", encoding="utf-8")
    monkeypatch.setattr(
        "src.pipelines.autonomous_unet_experiment_loop.run_preflight_checks",
        lambda policy: {"ok": True},
    )
    monkeypatch.setattr(
        "src.pipelines.autonomous_unet_experiment_loop.propose_hypotheses_via_openrouter",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("proposal failed")),
    )

    try:
        run_autonomous_unet_experiment_loop(policy_path=policy_path, batch_id="batch-fail")
    except RuntimeError:
        pass

    ledger = json.loads((tmp_path / "autoloop" / "batch-fail" / "ledger.json").read_text(encoding="utf-8"))
    assert ledger["phase"] == "aborted"
    assert ledger["failure_reason"] == "RuntimeError"
    assert ledger["failure_details"] == "proposal failed"
