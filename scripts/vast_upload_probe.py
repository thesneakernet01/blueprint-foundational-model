# SPDX-License-Identifier: Apache-2.0
"""Pinpoint which upload wire format the VAST endpoint/gateway rejects.

scripts/vast_probe.py proved the metadata calls work, but real uploads die
with "Connection was closed before we received a valid response" / 502 while
its ZERO-BYTE PutObject passed. The app's uploads differ from that probe on
three axes, each a known S3-compatibility landmine:

  * body size        — a gateway body-size limit kills anything over N bytes
  * Expect header    — botocore adds `Expect: 100-continue` to FILE-LIKE
                       bodies only (bytes bodies skip it); proxies mishandle it
  * trailing checksums — botocore >= 1.36 frames uploads as aws-chunked with a
                       trailing CRC32 unless told not to; older stores can't
                       parse the framing

This script PUTs a matrix of (size × body kind × checksum mode) objects and
prints PASS/FAIL per cell, escalating sizes until a variant first fails —
so one run names the exact offending combination. Run from a CML session:

    python scripts/vast_upload_probe.py

Credentials/endpoint come from env if set (VAST_ACCESS_KEY / VAST_SECRET_KEY /
VAST_ENDPOINT / VAST_BUCKET / VAST_PATH), otherwise it prompts. Probe objects
are written under <path>/upload_probe/ and deleted afterwards.
"""

from __future__ import annotations

import getpass
import io
import os
import sys

try:
    import boto3
    import botocore
    from botocore.config import Config
except ImportError:
    sys.exit("boto3 not installed. Run:  pip install boto3")

ENDPOINT = os.environ.get("VAST_ENDPOINT", "https://s3.previewhub.dev")
REGION = os.environ.get("VAST_REGION", "vast")
BUCKET = os.environ.get("VAST_BUCKET", "mschuler-cloudera")
PATH = os.environ.get("VAST_PATH", "mschuler-bucket").strip("/")

SIZES = [1 << 10, 100 << 10, 512 << 10, 1 << 20, 2 << 20, 4 << 20, 8 << 20]

# (label, config kwargs) — "legacy" turns the >=1.36 trailing checksums off.
CHECKSUM_MODES = [
    ("checksums=legacy ", {"request_checksum_calculation": "when_required",
                           "response_checksum_validation": "when_required"}),
    ("checksums=default", {}),
]
# (label, wrap) — file-like bodies get botocore's Expect: 100-continue header.
BODY_KINDS = [
    ("bytes (no Expect)   ", lambda b: b),
    ("file-like (Expect)  ", io.BytesIO),
]


def client_for(extra: dict):
    access = os.environ.get("VAST_ACCESS_KEY") or input("Access key: ").strip()
    secret = os.environ.get("VAST_SECRET_KEY") or getpass.getpass("Secret key (hidden): ").strip()
    os.environ["VAST_ACCESS_KEY"], os.environ["VAST_SECRET_KEY"] = access, secret
    try:
        cfg = Config(s3={"addressing_style": "path"}, signature_version="s3v4",
                     retries={"max_attempts": 1}, connect_timeout=20,
                     read_timeout=120, **extra)
    except TypeError:                      # botocore < 1.36: already legacy
        cfg = Config(s3={"addressing_style": "path"}, signature_version="s3v4",
                     retries={"max_attempts": 1}, connect_timeout=20,
                     read_timeout=120)
    return boto3.client("s3", endpoint_url=ENDPOINT, region_name=REGION,
                        aws_access_key_id=access, aws_secret_access_key=secret,
                        config=cfg, verify=False)


def human(n: int) -> str:
    return f"{n >> 10} KB" if n < (1 << 20) else f"{n >> 20} MB"


def main() -> None:
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    print(f"boto3 {boto3.__version__} · botocore {botocore.__version__} · "
          f"endpoint {ENDPOINT} · bucket {BUCKET}\n")

    keys = []
    for ck_label, ck_extra in CHECKSUM_MODES:
        s3 = client_for(ck_extra)
        for body_label, wrap in BODY_KINDS:
            first_fail = None
            for size in SIZES:
                key = f"{PATH}/upload_probe/{ck_label.strip()}_{len(keys)}_{size}"
                data = os.urandom(size)
                try:
                    s3.put_object(Bucket=BUCKET, Key=key, Body=wrap(data))
                    keys.append(key)
                    verdict = "PASS"
                except Exception as exc:                           # noqa: BLE001
                    verdict = f"FAIL  {str(exc).splitlines()[0][:120]}"
                    first_fail = size
                print(f"  {ck_label} · {body_label} · {human(size):>7}  {verdict}",
                      flush=True)
                if first_fail:
                    print(f"  {ck_label} · {body_label} · (skipping larger sizes)")
                    break
            print()

    # cleanup
    s3 = client_for(dict(CHECKSUM_MODES[0][1]))
    for key in keys:
        try:
            s3.delete_object(Bucket=BUCKET, Key=key)
        except Exception:                                          # noqa: BLE001
            pass
    print("Probe objects cleaned up.")
    print("\nReading the matrix: a size where only 'checksums=default' fails -> "
          "trailing-checksum framing; only 'file-like' fails -> the Expect "
          "100-continue header; everything above one size fails -> a gateway "
          "body-size limit at that boundary.")


if __name__ == "__main__":
    main()
