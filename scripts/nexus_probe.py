# SPDX-License-Identifier: Apache-2.0
"""Day-one NEXUS verification harness — run this the day Fundamental access
lands, before wiring anything else (see docs/nexus-ltm-design.md).

Each step isolates one unknown from the design doc's checklist, so a failure
names the exact thing to raise with Fundamental support:

    1. SDK import + version          (confirms the pip package name)
    2. AWS identity (STS)            (credential chain works from this box)
    3. SageMaker describe-endpoint   (endpoint exists + InService)
    4. S3 staging round-trip         (put/head/delete under NEXUS_S3_PREFIX)
    5. fit/adapt, 50 synthetic rows  (timed — the fit-semantics probe)
    6. single-row predict ×5         (p50/p95 wall clock -> NEXUS_SCORE_TIMEOUT_S)
    7. dtype round-trip              (all 13 RAW_FEATURE_COLS incl. nulls)

Steps 5-7 need the SDK transport (tfm_demo/nexus.py `_fit`/`_predict_frame`)
implemented; until then they print SKIP with the blocking reason. With
NEXUS_MODE=stub only steps 1-4 are meaningful (1 fails by design, 2-4 need AWS
env); the stub path itself is exercised by the app, not this probe.

Run from a CML session or the demo box:

    NEXUS_MODE=live NEXUS_ENDPOINT_NAME=... NEXUS_S3_BUCKET=... \
        python scripts/nexus_probe.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tfm_demo import nexus                                        # noqa: E402
from tfm_demo.config import RAW_FEATURE_COLS                      # noqa: E402

RESULTS: list[tuple[str, str, str]] = []


def step(label: str, fn):
    try:
        detail = fn() or ""
        RESULTS.append((label, "PASS", str(detail)))
    except nexus.NexusUnavailable as exc:
        RESULTS.append((label, "SKIP", str(exc)))
    except Exception as exc:                                       # noqa: BLE001
        RESULTS.append((label, "FAIL", str(exc).split("\n", 1)[0][:200]))


def _synthetic_frame(n: int):
    """n rows over all 13 RAW_FEATURE_COLS, mixing dtypes the way TabFormer
    does — including a null Zip and a huge hashed Merchant Name."""
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        "User": rng.integers(0, 100, n),
        "Card": rng.integers(0, 4, n),
        "Year": 2019,
        "Month": rng.integers(1, 13, n),
        "Day": rng.integers(1, 29, n),
        "Hour": rng.integers(0, 24, n),
        "Amount": rng.uniform(1, 900, n).round(2),
        "Use Chip": rng.choice(
            ["Chip Transaction", "Online Transaction", "Swipe Transaction"], n),
        "Merchant Name": rng.integers(-(2**62), 2**62, n).astype("int64"),
        "Merchant City": rng.choice(["AUSTIN", "ONLINE", "DALLAS"], n),
        "Merchant State": rng.choice(["TX", "ONLINE", "CA"], n),
        "Zip": rng.choice([78758.0, 75201.0, float("nan")], n),
        "MCC": rng.choice([5411, 5541, 5942], n),
    })
    return df[RAW_FEATURE_COLS]


def main() -> None:
    print(f"\nNEXUS probe · mode={nexus.mode()} · {nexus.target()}\n")

    step("1. SDK import + version", lambda: nexus._sdk() and "ok")

    def _sts():
        import boto3
        ident = boto3.client("sts").get_caller_identity()
        return f"account {ident['Account']} · {ident['Arn'].rsplit('/', 1)[-1]}"
    step("2. AWS identity (STS)", _sts)

    def _endpoint():
        c = nexus.check()
        if not c["ok"]:
            raise RuntimeError(c["error"])
        return f"InService · {c['latency_ms']} ms"
    step("3. SageMaker describe-endpoint + staging bucket", _endpoint)

    def _staging():
        import boto3
        bucket = os.environ.get("NEXUS_S3_BUCKET", "")
        if not bucket:
            raise RuntimeError("NEXUS_S3_BUCKET not set")
        key = f"{os.environ.get('NEXUS_S3_PREFIX', 'nexus-staging').strip('/')}/.probe"
        s3 = boto3.client("s3")
        s3.put_object(Bucket=bucket, Key=key, Body=b"probe")
        s3.head_object(Bucket=bucket, Key=key)
        s3.delete_object(Bucket=bucket, Key=key)
        return "put/head/delete ok"
    step("4. S3 staging round-trip", _staging)

    def _fit():
        df = _synthetic_frame(50)
        y = (df["Amount"] > 450).astype(int).to_numpy()
        t0 = time.perf_counter()
        ref = nexus._fit(df, y, df.iloc[:10], y[:10], print)
        return f"{time.perf_counter() - t0:.1f} s · ref keys {sorted(ref)}"
    step("5. fit/adapt on 50 synthetic rows (timed)", _fit)

    def _latency():
        df = _synthetic_frame(1)
        times = []
        for _ in range(5):
            t0 = time.perf_counter()
            p = nexus._predict_frame(df, None, timeout_s=60)
            times.append(time.perf_counter() - t0)
            float(p[0])                    # shape/orientation sanity
        times.sort()
        return (f"p50 {times[2]:.2f} s · p95 ~{times[-1]:.2f} s — set "
                f"NEXUS_SCORE_TIMEOUT_S accordingly (current "
                f"{nexus.score_timeout_s()})")
    step("6. single-row predict ×5 (p50/p95)", _latency)

    def _dtypes():
        df = _synthetic_frame(8)           # includes NaN Zip + int64 merchants
        p = nexus._predict_frame(df, None, timeout_s=60)
        if len(p) != len(df):
            raise RuntimeError(f"got {len(p)} probs for {len(df)} rows")
        if not all(0.0 <= float(x) <= 1.0 for x in p):
            raise RuntimeError(f"probabilities out of [0,1]: {list(p)[:3]} ...")
        return "13-column mixed-dtype frame accepted, probs in [0,1]"
    step("7. dtype round-trip (nulls, int64 merchant hash)", _dtypes)

    width = max(len(r[0]) for r in RESULTS)
    fails = skips = 0
    for label, verdict, detail in RESULTS:
        print(f"  {label:<{width}}  {verdict}  {detail}")
        fails += verdict == "FAIL"
        skips += verdict == "SKIP"
    print()
    if fails:
        print(f"{fails} step(s) FAILED — send the failing step + message to "
              "Fundamental support / your AWS admin.")
    elif skips:
        print(f"{skips} step(s) SKIPPED — implement the transport in "
              "tfm_demo/nexus.py (_sdk/_fit/_predict_frame) and re-run.")
    else:
        print("All steps pass — flip NEXUS_MODE=live and run an export.")


if __name__ == "__main__":
    main()
