# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the run-history store + progressive budget schedule."""

import pytest

from tfm_demo import runs


@pytest.fixture(autouse=True)
def _isolated_store(tmp_path, monkeypatch):
    monkeypatch.setattr(runs, "_PATH", tmp_path / ".runs_history.json")


def test_budget_schedule_monotonic_and_plateaus():
    budgets = [runs.budget_for(i) for i in range(1, 10)]
    embed = [b["embed_max"] for b in budgets]
    scale = [b["xgb_scale"] for b in budgets]
    assert embed == sorted(embed), "embed budget must never shrink"
    assert scale == sorted(scale), "boosting-round scale must never shrink"
    # Beyond the schedule the last tier repeats.
    assert budgets[5] == budgets[4] == runs.budget_for(99)
    assert budgets[-1]["xgb_scale"] == 1.0


def test_budget_respects_env_ceiling(monkeypatch):
    monkeypatch.setattr(runs, "_EMBED_CEILING", 6000)
    assert runs.budget_for(1)["embed_max"] == 4000
    assert runs.budget_for(5)["embed_max"] == 6000


def test_append_assigns_index_and_registry_defaults():
    rec = runs.append_run({"models": [], "lift": {}})
    assert rec["run"] == 1
    assert rec["run_id"].startswith("r1-")
    assert rec["registry"]["registered"] is False
    rec2 = runs.append_run({"models": [], "lift": {}})
    assert rec2["run"] == 2
    assert [r["run"] for r in runs.history()] == [1, 2]
    assert runs.next_run_index() == 3


def test_update_registry_patches_matching_run():
    rec = runs.append_run({})
    updated = runs.update_registry(rec["run_id"], {"registered": True,
                                                   "model_version": 3})
    assert updated["registry"]["registered"] is True
    assert updated["registry"]["model_version"] == 3
    assert runs.history()[0]["registry"]["model_version"] == 3
    assert runs.update_registry("r99-nope", {"registered": True}) is None


def test_reset_roundtrip():
    runs.append_run({})
    runs.reset()
    assert runs.history() == []
    assert runs.next_run_index() == 1
    assert runs.next_budget() == runs.budget_for(1)


def test_corrupt_file_tolerated():
    runs._PATH.write_text("{not json")
    assert runs.history() == []
    rec = runs.append_run({})
    assert rec["run"] == 1
