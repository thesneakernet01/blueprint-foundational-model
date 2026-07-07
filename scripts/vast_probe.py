# SPDX-License-Identifier: Apache-2.0
"""Replay Hadoop S3A's getFileStatus/mkdirs call sequence against VAST.

The Impala warehouse fails `CREATE DATABASE ... LOCATION 's3a://...'` with a
bare "400 null" from VAST, and the s3a error doesn't say WHICH S3 API was
rejected. Hadoop's getFileStatus on `s3a://<bucket>/<path>` issues, in order:

    1. HeadObject  <path>            (is it a file?)
    2. HeadObject  <path>/           (is it a directory marker?)
    3. ListObjectsV2 prefix=<path>/  (does anything exist under it?)

and creating the database directory then does:

    4. PutObject   <path>/<db>/     (zero-byte directory marker)

This script replays each call individually with boto3 (plus ListObjects V1 as
a comparison for #3) and reports PASS/FAIL per call, so the offending API is
identified instead of guessed. Run it from a CML session:

    python scripts/vast_probe.py

Credentials/endpoint come from env if set (VAST_ACCESS_KEY / VAST_SECRET_KEY /
VAST_ENDPOINT / VAST_BUCKET / VAST_PATH), otherwise it prompts.
"""

from __future__ import annotations

import getpass
import os
import sys

try:
    import boto3
    from botocore.config import Config
    from botocore.exceptions import ClientError
except ImportError:
    sys.exit("boto3 not installed. Run:  pip install boto3")

ENDPOINT = os.environ.get("VAST_ENDPOINT", "https://s3.previewhub.dev")
REGION = os.environ.get("VAST_REGION", "vast")
BUCKET = os.environ.get("VAST_BUCKET", "mschuler-cloudera")
PATH = os.environ.get("VAST_PATH", "mschuler-bucket").strip("/")


def main() -> None:
    access = os.environ.get("VAST_ACCESS_KEY") or input("Access key: ").strip()
    secret = os.environ.get("VAST_SECRET_KEY") or getpass.getpass("Secret key (hidden): ").strip()

    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    s3 = boto3.client(
        "s3",
        endpoint_url=ENDPOINT,
        aws_access_key_id=access,
        aws_secret_access_key=secret,
        region_name=REGION,
        config=Config(s3={"addressing_style": "path"}, signature_version="s3v4"),
        verify=False,
    )

    results = []

    def step(label, fn, ok_codes=()):
        """Run one S3 call; NotFound-style codes count as PASS (s3a expects
        them on nonexistent paths) — only protocol rejections are failures."""
        try:
            fn()
            results.append((label, "PASS", ""))
        except ClientError as e:
            err = e.response.get("Error", {})
            code = err.get("Code", "?")
            status = e.response.get("ResponseMetadata", {}).get("HTTPStatusCode", "?")
            rid = e.response.get("ResponseMetadata", {}).get("RequestId", "?")
            if code in ok_codes or status == 404:
                results.append((label, "PASS", f"(expected {code})"))
            else:
                results.append((label, "FAIL", f"HTTP {status} code={code} reqid={rid} msg={err.get('Message')}"))
        except Exception as e:                                     # noqa: BLE001
            results.append((label, "FAIL", str(e)))

    marker = f"{PATH}/fsi_demo_probe/"

    step("1. HeadObject  <path>      (file probe)",
         lambda: s3.head_object(Bucket=BUCKET, Key=PATH), ok_codes=("404", "NoSuchKey"))
    step("2. HeadObject  <path>/     (dir-marker probe)",
         lambda: s3.head_object(Bucket=BUCKET, Key=PATH + "/"), ok_codes=("404", "NoSuchKey"))
    step("3a. ListObjectsV2 prefix=<path>/ max-keys=2",
         lambda: s3.list_objects_v2(Bucket=BUCKET, Prefix=PATH + "/", MaxKeys=2))
    step("3b. ListObjectsV2 + delimiter (s3a dir listing shape)",
         lambda: s3.list_objects_v2(Bucket=BUCKET, Prefix=PATH + "/", MaxKeys=2,
                                    Delimiter="/", FetchOwner=False))
    step("3c. ListObjects V1 prefix=<path>/ (fallback API)",
         lambda: s3.list_objects(Bucket=BUCKET, Prefix=PATH + "/", MaxKeys=2))
    step("4. PutObject   <path>/fsi_demo_probe/ (zero-byte dir marker)",
         lambda: s3.put_object(Bucket=BUCKET, Key=marker, Body=b""))
    step("5. HeadObject  the marker just written",
         lambda: s3.head_object(Bucket=BUCKET, Key=marker))
    step("6. DeleteObject the marker (cleanup)",
         lambda: s3.delete_object(Bucket=BUCKET, Key=marker))
    step("7. HeadBucket  (bucket existence probe)",
         lambda: s3.head_bucket(Bucket=BUCKET))

    print(f"\nEndpoint {ENDPOINT} · bucket {BUCKET} · path /{PATH}\n")
    width = max(len(r[0]) for r in results)
    failures = 0
    for label, verdict, detail in results:
        print(f"  {label:<{width}}  {verdict}  {detail}")
        failures += verdict == "FAIL"
    print()
    if failures:
        print(f"{failures} call(s) rejected — that's the API the warehouse is tripping on.")
        print("Send the failing call name + reqid to the VAST/previewhub admin.")
    else:
        print("Every s3a-shaped call works via boto3 -> the 400 comes from a header")
        print("the Java AWS SDK adds that boto3 doesn't. Next step: VAST-side audit")
        print("log for the warehouse's request ID (it names the rejected header).")


if __name__ == "__main__":
    main()
