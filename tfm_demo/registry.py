# SPDX-License-Identifier: Apache-2.0
"""Cloudera Model Registry + CML Model endpoint integration.

This module owns ALL cmlapi/MLflow knowledge (the same way nexus.py owns the
NEXUS SDK). Everything heavier than stdlib is imported lazily inside functions,
so importing this module off-CML (Docker demo, laptop) costs nothing and the
API can always report *why* the registry is unavailable instead of erroring.

Registration path (most robust across CML versions): log the trained combined
head + preprocessor + PCA as an MLflow pyfunc with
`registered_model_name=MODEL_NAME` — on CML the tracking server IS the Model
Registry, so that single call creates the registered model on first use and a
new version on every later one. cmlapi is then used for what MLflow can't do:
resolving the workspace registry view and building/deploying the CML Model
endpoint. Every cmlapi surface that has drifted across CML releases
(create_model_build's registered_model_version_id, list_registered_models) is
feature-detected with a fallback, ending at a plain file-based model build
using tfm_demo/registry_predict.py.

Scope note (also in governance/model-cards/tfm_fraud_heads.md): the registered
bundle is the trained XGBoost combined head with its preprocessor and PCA. It
scores rows that already carry the 64 PCA components — the foundation-model
embedding stage runs upstream (it needs the FM checkpoint and a GPU) and is not
part of the registered asset.
"""

from __future__ import annotations

import json
import os
import time
from typing import Callable, Dict, List, Optional, Tuple

from .config import ARTIFACTS

MODEL_NAME = "tfm-fraud-combined"
EXPERIMENT_NAME = "tfm-fraud-demo"

_REQUIRED_ENV = ("CDSW_API_URL", "CDSW_APIV2_KEY", "CDSW_PROJECT_ID")
_ARTIFACT_FILES = ("xgb_combined.joblib", "preprocessor.joblib", "pca.joblib",
                   "summary.json")

Progress = Callable[[str], None]
StageHook = Callable[[str], None]

_client_cache = None


def available() -> Tuple[bool, Optional[str]]:
    """(True, None) when the registry integration can run here, else
    (False, human-readable reason) — the UI shows the reason verbatim."""
    missing = [v for v in _REQUIRED_ENV if not os.environ.get(v)]
    if missing:
        if "CDSW_API_URL" in missing or "CDSW_PROJECT_ID" in missing:
            return False, "not running on Cloudera AI (no CML workload environment)"
        return False, "APIv2 key not injected into this workload (CDSW_APIV2_KEY)"
    try:
        import cmlapi  # noqa: F401
    except ImportError:
        return False, "cmlapi is not importable in this runtime"
    return True, None


def _client():
    global _client_cache
    if _client_cache is None:
        import cmlapi
        _client_cache = cmlapi.default_client()
    return _client_cache


def _project_id() -> str:
    return os.environ["CDSW_PROJECT_ID"]


def artifacts_ready() -> Tuple[bool, Optional[str]]:
    missing = [f for f in _ARTIFACT_FILES if not (ARTIFACTS / f).exists()]
    if missing:
        return False, f"missing artifacts: {', '.join(missing)} — run an export first"
    return True, None


# --------------------------------------------------------------------------- #
# status (cheap — polled by the UI)
# --------------------------------------------------------------------------- #
def status() -> Dict:
    """Availability + registered versions + CML Model deployment state."""
    ok, reason = available()
    out: Dict = {"available": ok, "reason": reason, "model_name": MODEL_NAME,
                 "versions": [], "model": None, "deployment": None}
    ready, art_reason = artifacts_ready()
    out["artifacts_ready"] = ready
    out["artifacts_reason"] = art_reason
    if not ok:
        return out
    try:
        out["versions"] = _list_versions()
        model = _find_model()
        if model is not None:
            out["model"] = {"id": model.id, "name": model.name}
            out["deployment"] = _deployment_status(model.id)
    except Exception as exc:                                       # noqa: BLE001
        out["reason"] = f"registry query failed: {exc}"
    return out


def _list_versions() -> List[Dict]:
    """Registered versions of MODEL_NAME, newest first. Tries cmlapi's registry
    surface, falls back to the MLflow registry client."""
    client = _client()
    if hasattr(client, "list_registered_models"):
        try:
            resp = client.list_registered_models(search_filter=json.dumps(
                {"model_name": MODEL_NAME}))
            for m in getattr(resp, "models", None) or []:
                if m.name != MODEL_NAME:
                    continue
                versions = []
                for v in getattr(m, "model_versions", None) or []:
                    versions.append({
                        "version": getattr(v, "version", None),
                        "version_id": getattr(v, "model_version_id", None),
                        "created_at": str(getattr(v, "created_at", "") or ""),
                    })
                versions.sort(key=lambda v: v["version"] or 0, reverse=True)
                return versions
        except Exception:                                          # noqa: BLE001
            pass
    try:
        from mlflow.tracking import MlflowClient
        return [{
            "version": int(v.version), "version_id": None,
            "created_at": str(v.creation_timestamp),
        } for v in sorted(MlflowClient().search_model_versions(
            f"name='{MODEL_NAME}'"), key=lambda v: int(v.version), reverse=True)]
    except Exception:                                              # noqa: BLE001
        return []


def _find_model():
    """The CML Model named MODEL_NAME in this project, or None."""
    client = _client()
    resp = client.list_models(_project_id(),
                              search_filter=json.dumps({"name": MODEL_NAME}))
    for m in getattr(resp, "models", None) or []:
        if m.name == MODEL_NAME:
            return m
    return None


def _deployment_status(model_id: str) -> Optional[Dict]:
    client = _client()
    try:
        builds = client.list_model_builds(_project_id(), model_id)
        build = max(getattr(builds, "model_builds", None) or [],
                    key=lambda b: str(b.created_at), default=None)
        if build is None:
            return None
        out = {"build_status": build.status, "status": None, "url": None}
        deps = client.list_model_deployments(_project_id(), model_id, build.id)
        dep = max(getattr(deps, "model_deployments", None) or [],
                  key=lambda d: str(d.created_at), default=None)
        if dep is not None:
            out["status"] = dep.status
        base = os.environ.get("CDSW_API_URL", "")
        access_key = getattr(_get_model(model_id), "access_key", None)
        if base and access_key:
            # .../api/v1 -> workspace root; the model endpoint lives under it.
            out["url"] = f"{base.split('/api/')[0]}/model?accessKey={access_key}"
        return out
    except Exception:                                              # noqa: BLE001
        return None


def _get_model(model_id: str):
    return _client().get_model(_project_id(), model_id)


# --------------------------------------------------------------------------- #
# register
# --------------------------------------------------------------------------- #
def _pyfunc_model():
    """The mlflow.pyfunc wrapper for the registered bundle, built lazily so
    importing this module never pulls mlflow. Input: a pandas frame with the
    raw feature columns PLUS pca_0..pca_{k-1} embedding components (the FM
    embedding + PCA stage runs upstream). Output: fraud probability."""
    import mlflow.pyfunc

    class CombinedHeadModel(mlflow.pyfunc.PythonModel):
        def load_context(self, context):
            import joblib
            self.preproc = joblib.load(context.artifacts["preprocessor"])
            self.pca = joblib.load(context.artifacts["pca"])
            self.clf = joblib.load(context.artifacts["xgb_combined"])

        def predict(self, context, model_input):
            import numpy as np
            pca_cols = [c for c in model_input.columns if c.startswith("pca_")]
            raw = model_input[[c for c in model_input.columns
                               if c not in pca_cols]]
            X = np.hstack([self.preproc.transform(raw),
                           model_input[pca_cols].to_numpy()])
            return self.clf.predict_proba(X)[:, 1]

    return CombinedHeadModel()


def register_latest(progress: Progress, on_stage: StageHook) -> Dict:
    """Register the current demo_artifacts bundle as a new version of
    MODEL_NAME in the workspace Model Registry. Returns the version info."""
    ok, reason = available()
    if not ok:
        raise RuntimeError(f"Model Registry unavailable: {reason}")
    ready, art_reason = artifacts_ready()
    if not ready:
        raise RuntimeError(art_reason)

    from . import runs

    latest_run = runs.latest()
    summary = json.loads((ARTIFACTS / "summary.json").read_text())
    metrics = {m["key"]: m for m in summary.get("models", [])}

    on_stage("log")
    progress("Logging model to the workspace MLflow registry ...")
    import mlflow
    import mlflow.pyfunc

    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(run_name=(latest_run or {}).get("run_id", "manual")):
        if latest_run:
            mlflow.log_params({
                "run_index": latest_run["run"],
                "embed_max_per_split": latest_run["budget"]["embed_max"],
                "xgb_scale": latest_run["budget"]["xgb_scale"],
            })
        for key in ("raw", "embed", "combined"):
            if key in metrics:
                mlflow.log_metric(f"{key}_test_auc", metrics[key]["test_auc"])
                mlflow.log_metric(f"{key}_test_ap", metrics[key]["test_ap"])
        info = mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=_pyfunc_model(),
            artifacts={
                "xgb_combined": str(ARTIFACTS / "xgb_combined.joblib"),
                "preprocessor": str(ARTIFACTS / "preprocessor.joblib"),
                "pca": str(ARTIFACTS / "pca.joblib"),
            },
            registered_model_name=MODEL_NAME,
        )
    progress(f"Logged model ({info.model_uri})")

    on_stage("version")
    progress("Resolving the new registry version ...")
    version = version_id = None
    for _ in range(10):                    # registration is async on some versions
        versions = _list_versions()
        if versions:
            version = versions[0]["version"]
            version_id = versions[0]["version_id"]
            break
        time.sleep(2)
    progress(f"Registered {MODEL_NAME} v{version}"
             if version is not None else
             "Registered (version number not yet visible in the registry)")

    if latest_run:
        runs.update_registry(latest_run["run_id"], {
            "registered": True, "model_version": version,
            "version_id": version_id, "registered_at": runs.now_iso(),
        })
    on_stage("done")
    return {"model_name": MODEL_NAME, "version": version,
            "version_id": version_id, "run_id": (latest_run or {}).get("run_id")}


# --------------------------------------------------------------------------- #
# deploy
# --------------------------------------------------------------------------- #
def _pick_runtime() -> str:
    """A Python Standard workbench runtime identifier for the model build."""
    client = _client()
    resp = client.list_runtimes(search_filter=json.dumps(
        {"kernel": "Python 3.12", "edition": "Standard"}))
    candidates = getattr(resp, "runtimes", None) or []
    if not candidates:
        resp = client.list_runtimes(search_filter=json.dumps(
            {"edition": "Standard"}))
        candidates = getattr(resp, "runtimes", None) or []
    if not candidates:
        raise RuntimeError("no Standard runtime available for the model build")
    return sorted(candidates, key=lambda r: r.image_identifier)[-1].image_identifier


def deploy_latest(progress: Progress, on_stage: StageHook) -> Dict:
    """Build + deploy the most recently registered version as a CML Model
    endpoint (create-or-reuse the Model, new build, new deployment)."""
    ok, reason = available()
    if not ok:
        raise RuntimeError(f"Model Registry unavailable: {reason}")
    versions = _list_versions()
    if not versions:
        raise RuntimeError("no registered version to deploy — register first")
    target = versions[0]

    import cmlapi
    client = _client()
    project_id = _project_id()

    on_stage("model")
    model = _find_model()
    if model is None:
        progress(f"Creating CML Model {MODEL_NAME!r} ...")
        model = client.create_model(cmlapi.CreateModelRequest(
            project_id=project_id, name=MODEL_NAME,
            description="TFM fraud demo — combined XGBoost head "
                        "(registered via the Model Lifecycle dashboard)",
            disable_authentication=True,
        ), project_id)
    else:
        progress(f"Reusing CML Model {MODEL_NAME!r} ({model.id})")

    on_stage("build")
    runtime = _pick_runtime()
    progress(f"Starting model build (runtime {runtime}) ...")
    build = None
    if target["version_id"]:
        try:
            build = client.create_model_build(cmlapi.CreateModelBuildRequest(
                project_id=project_id, model_id=model.id,
                registered_model_version_id=target["version_id"],
                runtime_identifier=runtime,
            ), project_id, model.id)
            progress(f"Building from registry version v{target['version']}")
        except Exception as exc:                                   # noqa: BLE001
            progress(f"(registry-version build not supported here: {exc}; "
                     "falling back to a file-based build)")
    if build is None:
        # File-based fallback: serve demo_artifacts directly via the scoring
        # shim (works on CML versions without registry-backed builds).
        build = client.create_model_build(cmlapi.CreateModelBuildRequest(
            project_id=project_id, model_id=model.id,
            file_path="tfm_demo/registry_predict.py", function_name="predict",
            kernel="python3", runtime_identifier=runtime,
        ), project_id, model.id)
        progress("Building from tfm_demo/registry_predict.py")

    while build.status not in ("built", "build failed"):
        time.sleep(10)
        build = client.get_model_build(project_id, model.id, build.id)
        progress(f"  build: {build.status}")
    if build.status != "built":
        raise RuntimeError(f"model build failed (build {build.id}) — see the "
                           "Model Builds page in the CML project")

    on_stage("deploy")
    progress("Deploying the model endpoint (2 vCPU / 4 GB) ...")
    dep = client.create_model_deployment(cmlapi.CreateModelDeploymentRequest(
        project_id=project_id, model_id=model.id, build_id=build.id,
        cpu=2, memory=4,
    ), project_id, model.id, build.id)
    while dep.status not in ("deployed", "stopped", "failed"):
        time.sleep(10)
        dep = client.get_model_deployment(project_id, model.id, build.id, dep.id)
        progress(f"  deployment: {dep.status}")
    if dep.status != "deployed":
        raise RuntimeError(f"model deployment ended in state {dep.status!r}")

    from . import runs
    latest_run = runs.latest()
    if latest_run:
        runs.update_registry(latest_run["run_id"],
                             {"deployed": True, "deployed_at": runs.now_iso()})
    on_stage("done")
    progress(f"Deployed {MODEL_NAME} v{target['version']}")
    return {"model_name": MODEL_NAME, "version": target["version"],
            "deployment_status": dep.status}
