# SPDX-License-Identifier: Apache-2.0
"""Reproduce the warehouse's s3a->VAST "400 null" with full wire logging.

boto3 replay (scripts/vast_probe.py) passed every S3 call, so the 400 the
Impala warehouse gets from `s3a://mschuler-cloudera/...` must come from the
Java AWS SDK v2's HTTP specifics (headers/encoding), or from a proxy in front
of VAST rejecting them. This script runs the SAME client stack the warehouse
uses — Hadoop S3A on AWS Java SDK v2 — from a CML session, with the HTTP wire
log turned on, so the exact failing request and raw 400 response are captured.

It is self-contained and needs no root:
  * downloads a Temurin JRE 17 (skipped if `java` is already present),
  * downloads Apache Hadoop 3.4.1 (bundles hadoop-aws + AWS SDK v2),
  * writes a core-site.xml with the same five fs.s3a.bucket.* properties the
    warehouse uses, and a log4j config with SDK request + wire logging,
  * runs `hadoop fs -ls s3a://<bucket>/<path>/` in three variants:
       base, +list.version=1, +change.detection.mode=none
    stopping at the first success.

Run from a CML session (>= 4 GB memory profile, ~2 GB scratch disk):

    python scripts/vast_s3a_debug.py

Full logs land in $VAST_DEBUG_DIR (default /tmp/vast_s3a_debug); the script
prints the wire excerpt around the first 400 for each failing variant — that
excerpt is the thing to share/compare (it names every header the SDK sent).
Credentials/endpoint via VAST_ACCESS_KEY / VAST_SECRET_KEY / VAST_ENDPOINT /
VAST_BUCKET / VAST_PATH env vars, prompted if unset.
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

LOG4J = """log4j.rootLogger=INFO,console
log4j.appender.console=org.apache.log4j.ConsoleAppender
log4j.appender.console.target=System.err
log4j.appender.console.layout=org.apache.log4j.PatternLayout
log4j.appender.console.layout.ConversionPattern=%d{ISO8601} %-5p %c{2}: %m%n
# The two loggers that reveal the exact HTTP conversation:
log4j.logger.software.amazon.awssdk.request=DEBUG
log4j.logger.org.apache.http.wire=DEBUG
log4j.logger.org.apache.hadoop.fs.s3a=DEBUG
"""

VARIANTS = [
    ("base", []),
    ("list-v1", ["-Dfs.s3a.list.version=1"]),
    ("no-change-detection", ["-Dfs.s3a.change.detection.mode=none"]),
]


def _download(url: str, dest: Path, label: str) -> None:
    if dest.exists():
        print(f"{label}: already downloaded ({dest})")
        return
    print(f"{label}: downloading {url} ...")
    tmp = dest.with_suffix(".part")

    def hook(n, bs, total):
        done = n * bs
        if total > 0 and n % 200 == 0:
            print(f"  {done/1e6:,.0f}/{total/1e6:,.0f} MB", end="\r", flush=True)

    urllib.request.urlretrieve(url, tmp, reporthook=hook)
    tmp.replace(dest)
    print(f"\n{label}: done.")


def _extract(tgz: Path, marker: str) -> Path:
    """Extract tgz into WORK (once) and return the top-level dir containing
    `marker` (e.g. bin/java or bin/hadoop)."""
    for child in WORK.iterdir():
        if child.is_dir() and (child / marker).exists():
            return child
    print(f"extracting {tgz.name} ...")
    with tarfile.open(tgz, "r:gz") as tar:
        tar.extractall(WORK)
    for child in WORK.iterdir():
        if child.is_dir() and (child / marker).exists():
            return child
    sys.exit(f"could not find {marker} under {WORK} after extracting {tgz.name}")


def _java_home() -> str:
    java = shutil.which("java")
    if java:
        print(f"using system java: {java}")
        return str(Path(java).resolve().parent.parent)
    _download(JRE_URL, WORK / "temurin17-jre.tar.gz", "JRE 17")
    home = _extract(WORK / "temurin17-jre.tar.gz", "bin/java")
    print(f"using downloaded JRE: {home}")
    return str(home)


def _excerpt(log: Path) -> str:
    """The wire-log window around the first 400 response (or the tail)."""
    lines = log.read_text(errors="replace").splitlines()
    idx = next((i for i, l in enumerate(lines) if '<< "HTTP/1.1 400' in l or
                "Status Code: 400" in l), None)
    if idx is None:
        return "\n".join(lines[-40:])
    start = idx
    for j in range(idx, max(idx - 400, -1), -1):   # back to the request line
        if '>> "' in lines[j] and any(m in lines[j] for m in
                                      ('HEAD ', 'GET ', 'PUT ', 'POST ', 'DELETE ')):
            start = j
            break
    return "\n".join(lines[start:idx + 30])


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
        "HADOOP_OPTIONAL_TOOLS": "hadoop-aws",   # puts hadoop-aws + SDK on the classpath
        "HADOOP_HEAPSIZE_MAX": "1g",
    }
    target = f"s3a://{BUCKET}/{PATH}/"

    for name, extra in VARIANTS:
        logf = WORK / f"wire-{name}.log"
        print(f"\n=== variant: {name}  ({' '.join(extra) or 'no extra options'}) ===")
        with open(logf, "w") as out:
            rc = subprocess.run(
                [str(hadoop_home / "bin" / "hadoop"), "fs", *extra, "-ls", target],
                env=env, stdout=out, stderr=subprocess.STDOUT, text=True,
            ).returncode
        if rc == 0:
            print(f"SUCCESS — `hadoop fs -ls {target}` worked with variant '{name}'!")
            if extra:
                prop = extra[0][2:].split("=")
                print(f"Fix for the warehouse: add "
                      f"fs.s3a.bucket.{BUCKET}.{prop[0][7:]} = {prop[1]} "
                      "(coordinator, executor, catalogd -> hadoop-core-site).")
            else:
                print("The base config works on Hadoop 3.4.1 — the warehouse's "
                      "Hadoop/SDK build differs; share the warehouse version with "
                      "the platform/VAST folks.")
            return
        print(f"FAILED (exit {rc}) — wire excerpt around the 400 "
              f"(full log: {logf}):\n")
        print(_excerpt(logf))

    print("\nAll variants failed — the excerpts above show every header the Java "
          "SDK sent and the raw 400 response (its Server: header reveals whether "
          "VAST or a proxy answered). Share them with the previewhub/VAST admin.")


if __name__ == "__main__":
    main()
