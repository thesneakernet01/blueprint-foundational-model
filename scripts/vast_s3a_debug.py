# SPDX-License-Identifier: Apache-2.0
"""Round 2 of the s3a->VAST debug: write-path test + raw HEAD headers.

Round 1 findings (see git history for the first version): with the warehouse's
five fs.s3a.bucket.* properties, the Java SDK's LIST works against VAST (200)
— only HeadObject draws the bodiless 400, and s3a's getFileStatus recovers
from it via the LIST. list.version / change.detection are irrelevant.

This round answers the two questions that remain:
  1. Does the WRITE path work? (mkdir -> touchz -> ls -> rm — the same ops
     CREATE DATABASE/TABLE need). If yes, only the HEAD quirk stands between
     us and a working warehouse.
  2. What EXACTLY does the failing HEAD look like on the wire? Round 1's wire
     log stayed silent because Hadoop bundles the AWS SDK with a SHADED Apache
     HTTP client — its logger lives under software.amazon.awssdk.thirdparty.*.
     This version enables the right loggers and prints the failing HEAD's raw
     request headers + raw response line, which is what the VAST/previewhub
     admin needs.

Variants: base config, then with the s3a auditor off (it adds a long Referer
header boto3 never sends — one of the few remaining request differences).

Run from a CML session (downloads from round 1 are reused):

    python scripts/vast_s3a_debug.py
"""

from __future__ import annotations

import getpass
import os
import shutil
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path

ENDPOINT = os.environ.get("VAST_ENDPOINT", "https://s3.previewhub.dev")
REGION = os.environ.get("VAST_REGION", "vast")
BUCKET = os.environ.get("VAST_BUCKET", "mschuler-cloudera")
PATH = os.environ.get("VAST_PATH", "mschuler-bucket").strip("/")

WORK = Path(os.environ.get("VAST_DEBUG_DIR", "/tmp/vast_s3a_debug"))
HADOOP_URL = "https://dlcdn.apache.org/hadoop/common/hadoop-3.4.1/hadoop-3.4.1.tar.gz"
JRE_URL = ("https://api.adoptium.net/v3/binary/latest/17/ga/linux/x64/"
           "jre/hotspot/normal/eclipse")

CORE_SITE = """<?xml version="1.0"?>
<configuration>
  <property><name>fs.s3a.bucket.{b}.endpoint</name><value>{endpoint}</value></property>
  <property><name>fs.s3a.bucket.{b}.path.style.access</name><value>true</value></property>
  <property><name>fs.s3a.bucket.{b}.endpoint.region</name><value>{region}</value></property>
  <property><name>fs.s3a.bucket.{b}.access.key</name><value>{access}</value></property>
  <property><name>fs.s3a.bucket.{b}.secret.key</name><value>{secret}</value></property>
</configuration>
"""

# The SDK bundle shades Apache HttpClient — the wire/header loggers live under
# software.amazon.awssdk.thirdparty.org.apache.http, NOT org.apache.http.
LOG4J = """log4j.rootLogger=INFO,console
log4j.appender.console=org.apache.log4j.ConsoleAppender
log4j.appender.console.target=System.err
log4j.appender.console.layout=org.apache.log4j.PatternLayout
log4j.appender.console.layout.ConversionPattern=%d{ISO8601} %-5p %c{2}: %m%n
log4j.logger.software.amazon.awssdk.request=DEBUG
log4j.logger.software.amazon.awssdk.thirdparty.org.apache.http.headers=DEBUG
log4j.logger.software.amazon.awssdk.thirdparty.org.apache.http.wire=DEBUG
log4j.logger.org.apache.hadoop.fs.s3a=DEBUG
"""

VARIANTS = [
    ("base", []),
    ("no-audit-referer", ["-Dfs.s3a.audit.enabled=false"]),
]

# The op sequence CREATE DATABASE/TABLE + INSERT need, as fs shell steps.
def _steps(target: str):
    d = target + "s3a_probe_dir"
    return [
        ("mkdir",  ["-mkdir", "-p", d]),
        ("touchz", ["-touchz", d + "/probe.txt"]),
        ("ls",     ["-ls", d]),
        ("rm",     ["-rm", "-r", "-skipTrash", d]),
    ]


def _download(url: str, dest: Path, label: str) -> None:
    if dest.exists():
        print(f"{label}: already downloaded ({dest})")
        return
    print(f"{label}: downloading {url} ...")
    tmp = dest.with_suffix(".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(dest)
    print(f"{label}: done.")


def _extract(tgz: Path, marker: str) -> Path:
    for child in WORK.iterdir():
        if child.is_dir() and (child / marker).exists():
            return child
    print(f"extracting {tgz.name} ...")
    with tarfile.open(tgz, "r:gz") as tar:
        tar.extractall(WORK)
    for child in WORK.iterdir():
        if child.is_dir() and (child / marker).exists():
            return child
    sys.exit(f"could not find {marker} under {WORK}")


def _java_home() -> str:
    java = shutil.which("java")
    if java:
        return str(Path(java).resolve().parent.parent)
    _download(JRE_URL, WORK / "temurin17-jre.tar.gz", "JRE 17")
    return str(_extract(WORK / "temurin17-jre.tar.gz", "bin/java"))


def _head_excerpt(log: Path) -> str:
    """Raw header lines (shaded-httpclient '>> ' / '<< ' output) around the
    first 400 response; falls back to the awssdk.request summaries."""
    lines = log.read_text(errors="replace").splitlines()
    idx = next((i for i, l in enumerate(lines)
                if ('<< "HTTP/1.1 400' in l) or ("<< HTTP/1.1 400" in l)), None)
    if idx is not None:
        start = idx
        for j in range(idx, max(idx - 300, -1), -1):
            if ">> " in lines[j] and any(m in lines[j] for m in
                                         ("HEAD ", "GET ", "PUT ", "POST ", "DELETE ")):
                start = j
                break
        window = lines[start:idx + 20]
        hdrs = [l for l in window if (">> " in l or "<< " in l)]
        return "\n".join(hdrs or window)
    summaries = [l for l in lines if "awssdk.request" in l or "Status Code: 400" in l]
    return "\n".join(summaries[-15:]) if summaries else "\n".join(lines[-25:])


def main() -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    access = os.environ.get("VAST_ACCESS_KEY") or input("Access key: ").strip()
    secret = os.environ.get("VAST_SECRET_KEY") or getpass.getpass("Secret key (hidden): ").strip()

    java_home = _java_home()
    _download(HADOOP_URL, WORK / "hadoop-3.4.1.tar.gz", "Hadoop 3.4.1 (~1 GB)")
    hadoop_home = _extract(WORK / "hadoop-3.4.1.tar.gz", "bin/hadoop")

    conf = WORK / "conf"
    conf.mkdir(exist_ok=True)
    (conf / "core-site.xml").write_text(CORE_SITE.format(
        b=BUCKET, endpoint=ENDPOINT, region=REGION, access=access, secret=secret))
    (conf / "log4j.properties").write_text(LOG4J)

    env = {
        **os.environ,
        "JAVA_HOME": java_home,
        "HADOOP_HOME": str(hadoop_home),
        "HADOOP_CONF_DIR": str(conf),
        "HADOOP_OPTIONAL_TOOLS": "hadoop-aws",
        "HADOOP_HEAPSIZE_MAX": "1g",
    }
    target = f"s3a://{BUCKET}/{PATH}/"

    # Run EVERY variant even when the round trip passes: the question is not
    # only "does it work" but "does any variant make the HEAD stop 400ing" —
    # a zero-400 variant is a direct warehouse fix for Impala's validation.
    verdicts = []
    for name, extra in VARIANTS:
        print(f"\n=== variant: {name} ===")
        all_ok = True
        n400 = 0
        for step, args in _steps(target):
            logf = WORK / f"wire-{name}-{step}.log"
            with open(logf, "w") as out:
                rc = subprocess.run(
                    [str(hadoop_home / "bin" / "hadoop"), "fs", *extra, *args],
                    env=env, stdout=out, stderr=subprocess.STDOUT, text=True,
                ).returncode
            hits = logf.read_text(errors="replace").count("Status Code: 400")
            n400 += hits
            print(f"  {step:<7} {'OK' if rc == 0 else f'FAILED (exit {rc})'}"
                  f"{f'  ({hits} recovered 400s)' if hits else ''}")
            if rc != 0:
                all_ok = False
                print(f"  --- raw HTTP around the 400 ({logf}) ---")
                print(_head_excerpt(logf))
                print("  ---")
                break
        verdicts.append((name, extra, all_ok, n400))
        print(f"  round trip: {'OK' if all_ok else 'FAILED'} · "
              f"HTTP 400s seen: {n400}")

    clean = next((v for v in verdicts if v[2] and v[3] == 0), None)
    print()
    if clean:
        name, extra, *_ = clean
        print(f"CLEAN VARIANT FOUND: '{name}' — round trip OK with ZERO 400s.")
        if extra:
            prop = extra[0][2:].split("=")
            print(f"Warehouse fix: fs.s3a.bucket.{BUCKET}.{prop[0][7:]} = {prop[1]}")
            print("Add it to coordinator/executor/catalogd hadoop-core-site "
                  "(and the Database Catalog metastore), then retry "
                  "CREATE DATABASE ... LOCATION in Hue.")
    else:
        print("No variant eliminated the 400s (round trip may still pass — "
              "Hadoop recovers, Impala's validation doesn't).")
        print("Raw failing HEAD for the VAST admin — run:")
        print(f"  grep -a -B2 -A12 'HTTP/1.1 400' {WORK}/wire-base-mkdir.log | head -60")
        print("(HEAD on a missing key should return 404; VAST returns a "
              "bodiless 400 for this request shape.)")


if __name__ == "__main__":
    main()
