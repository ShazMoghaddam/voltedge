"""
Tests for GitHub Actions workflow files — validate YAML structure and
required jobs/steps without spinning up a real GitHub runner.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml


WORKFLOWS_DIR = Path(__file__).parent.parent.parent / ".github" / "workflows"
CI_YAML  = WORKFLOWS_DIR / "ci.yml"
CD_YAML  = WORKFLOWS_DIR / "cd.yml"


@pytest.fixture(scope="module")
def ci():
    return yaml.safe_load(CI_YAML.read_text())


@pytest.fixture(scope="module")
def cd():
    return yaml.safe_load(CD_YAML.read_text())


# ── CI file structure ─────────────────────────────────────────────────────────

def test_ci_file_exists():
    assert CI_YAML.exists()


def test_ci_is_valid_yaml(ci):
    assert isinstance(ci, dict)


def test_ci_has_name(ci):
    assert "name" in ci


def test_ci_triggers_on_push_and_pr(ci):
    on = ci.get(True, {})
    assert "push" in on
    assert "pull_request" in on


def test_ci_triggers_on_main_and_develop(ci):
    branches = ci[True]["push"]["branches"]
    assert "main" in branches
    assert "develop" in branches


def test_ci_has_required_jobs(ci):
    jobs = ci.get("jobs", {})
    for required in ("lint", "test", "build"):
        assert required in jobs, f"Missing CI job: {required}"


def test_ci_test_job_has_matrix(ci):
    test_job = ci["jobs"]["test"]
    assert "matrix" in test_job.get("strategy", {})


def test_ci_test_matrix_covers_311_and_312(ci):
    versions = ci["jobs"]["test"]["strategy"]["matrix"]["python-version"]
    assert "3.11" in versions
    assert "3.12" in versions


def test_ci_test_job_runs_pytest(ci):
    steps = ci["jobs"]["test"]["steps"]
    pytest_steps = [s for s in steps if "pytest" in str(s.get("run", ""))]
    assert len(pytest_steps) > 0


def test_ci_test_job_has_coverage(ci):
    steps = ci["jobs"]["test"]["steps"]
    cov_steps = [s for s in steps if "--cov" in str(s.get("run", ""))]
    assert len(cov_steps) > 0


def test_ci_build_depends_on_lint_and_test(ci):
    needs = ci["jobs"]["build"].get("needs", [])
    assert "lint" in needs
    assert "test" in needs


def test_ci_has_security_job(ci):
    assert "security" in ci["jobs"]


def test_ci_security_uses_bandit(ci):
    steps = ci["jobs"]["security"]["steps"]
    bandit_steps = [s for s in steps if "bandit" in str(s.get("run", "")).lower()
                    or "bandit" in str(s.get("name", "")).lower()]
    assert len(bandit_steps) > 0


def test_ci_has_docker_job(ci):
    assert "docker" in ci["jobs"]


def test_ci_docker_job_smoke_tests_health(ci):
    steps = ci["jobs"]["docker"]["steps"]
    health_steps = [s for s in steps if "/health" in str(s.get("run", ""))]
    assert len(health_steps) > 0


def test_ci_lstm_job_only_runs_on_main_push(ci):
    lstm_job = ci["jobs"].get("test-lstm", {})
    condition = lstm_job.get("if", "")
    assert "main" in condition


# ── CD file structure ─────────────────────────────────────────────────────────

def test_cd_file_exists():
    assert CD_YAML.exists()


def test_cd_is_valid_yaml(cd):
    assert isinstance(cd, dict)


def test_cd_triggers_on_version_tags(cd):
    on = cd.get(True, {})
    assert "push" in on
    tags = on["push"].get("tags", [])
    assert any("v*" in t for t in tags)


def test_cd_has_required_jobs(cd):
    jobs = cd.get("jobs", {})
    for required in ("context", "push-image", "deploy", "notify"):
        assert required in jobs, f"Missing CD job: {required}"


def test_cd_deploy_depends_on_push_image(cd):
    needs = cd["jobs"]["deploy"].get("needs", [])
    assert "push-image" in needs


def test_cd_deploy_has_health_check(cd):
    steps = cd["jobs"]["deploy"]["steps"]
    health_steps = [s for s in steps if "/health" in str(s.get("run", ""))]
    assert len(health_steps) > 0


def test_cd_notify_runs_always(cd):
    notify = cd["jobs"]["notify"]
    assert notify.get("if", "").strip() == "always()"


def test_cd_deploy_uses_environment_context(cd):
    deploy = cd["jobs"]["deploy"]
    env = deploy.get("environment", "")
    assert "context" in str(env) or "environment" in str(env)


def test_cd_uses_ecr_for_docker(cd):
    steps = cd["jobs"]["push-image"]["steps"]
    ecr_steps = [s for s in steps if "ecr" in str(s).lower()]
    assert len(ecr_steps) > 0
