# SPDX-License-Identifier: Apache-2.0
"""Lightweight CPU / RAM / GPU utilization sampling for the build monitor.

`sample()` is called on every GET /api/export/status poll so the Build
artifacts dialog can show live meters while the export runs. Everything is
best-effort and stdlib-only: any probe that fails reports None and the UI
hides that meter (e.g. a laptop with no GPU or no /proc).

Container-aware on purpose: the demo runs inside a CML container whose /proc
shows the whole node, so CPU is measured against the cgroup quota and RAM
against the cgroup limit (v2 then v1), falling back to host-wide /proc
readings outside a container. GPU comes from nvidia-smi (driver-level, no
Python GPU imports — safe to call from the status endpoint at any time).
Samples are cached ~1 s so a burst of polls costs one probe.
"""

from __future__ import annotations

import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

_CACHE_SEC = 1.0
_lock = threading.Lock()
_cached: Dict = {}
_cached_at = 0.0
# Previous CPU reading: ("cgroup", monotonic_ts, usage_sec) or
# ("proc", total_jiffies, idle_jiffies) — deltas give a utilization %.
_cpu_prev: Optional[Tuple] = None


def _read(path: str) -> Optional[str]:
    try:
        return Path(path).read_text().strip()
    except Exception:                                              # noqa: BLE001
        return None


def _cpu_quota() -> float:
    """CPUs this container may use (cgroup quota), else the host count."""
    raw = _read("/sys/fs/cgroup/cpu.max")                          # v2: "quota period"
    if raw:
        quota, _, period = raw.partition(" ")
        if quota != "max":
            try:
                return max(int(quota) / int(period or 100000), 0.01)
            except ValueError:
                pass
    q = _read("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")               # v1
    p = _read("/sys/fs/cgroup/cpu/cpu.cfs_period_us")
    try:
        if q and p and int(q) > 0:
            return max(int(q) / int(p), 0.01)
    except ValueError:
        pass
    import os
    return float(os.cpu_count() or 1)


def _cpu_pct() -> Optional[float]:
    """Container CPU use as % of its quota; host-wide % outside a container.
    Needs two samples — the first call after startup returns None."""
    global _cpu_prev
    now = time.monotonic()

    usage = None                                                   # cgroup CPU-seconds
    raw = _read("/sys/fs/cgroup/cpu.stat")                         # v2
    if raw:
        for line in raw.splitlines():
            if line.startswith("usage_usec"):
                usage = int(line.split()[1]) / 1e6
                break
    if usage is None:
        raw = _read("/sys/fs/cgroup/cpuacct/cpuacct.usage")        # v1 (ns)
        if raw:
            try:
                usage = int(raw) / 1e9
            except ValueError:
                pass
    if usage is not None:
        prev, _cpu_prev = _cpu_prev, ("cgroup", now, usage)
        if prev and prev[0] == "cgroup" and now > prev[1]:
            pct = (usage - prev[2]) / (now - prev[1]) / _cpu_quota() * 100
            return round(min(max(pct, 0.0), 100.0), 1)
        return None

    raw = _read("/proc/stat")                                      # host-wide fallback
    if raw and raw.startswith("cpu "):
        parts = [int(x) for x in raw.splitlines()[0].split()[1:]]
        idle = parts[3] + (parts[4] if len(parts) > 4 else 0)
        total = sum(parts)
        prev, _cpu_prev = _cpu_prev, ("proc", total, idle)
        if prev and prev[0] == "proc" and total > prev[1]:
            dt, didle = total - prev[1], idle - prev[2]
            return round(min(max((1 - didle / dt) * 100, 0.0), 100.0), 1)
    return None


def _ram() -> Tuple[Optional[float], Optional[float]]:
    """(used_gb, total_gb) — cgroup current/limit, falling back to /proc/meminfo."""
    used = total = None
    cur = _read("/sys/fs/cgroup/memory.current")                   # v2
    if cur is not None:
        used = int(cur)
        mx = _read("/sys/fs/cgroup/memory.max")
        if mx and mx != "max":
            total = int(mx)
    else:
        cur = _read("/sys/fs/cgroup/memory/memory.usage_in_bytes")  # v1
        if cur is not None:
            used = int(cur)
            mx = _read("/sys/fs/cgroup/memory/memory.limit_in_bytes")
            if mx and int(mx) < 1 << 60:                           # "unlimited" sentinel
                total = int(mx)
    if total is None:
        raw = _read("/proc/meminfo")
        if raw:
            mem = {}
            for line in raw.splitlines()[:4]:
                key, _, val = line.partition(":")
                mem[key] = int(val.split()[0]) * 1024
            if "MemTotal" in mem:
                total = mem["MemTotal"]
                if used is None and "MemAvailable" in mem:
                    used = total - mem["MemAvailable"]
    to_gb = lambda b: round(b / 1e9, 2) if b is not None else None  # noqa: E731
    return to_gb(used), to_gb(total)


def _gpu() -> Dict:
    """Name, utilization % and memory from nvidia-smi (None-filled if absent)."""
    out = {"gpu_name": None, "gpu_util_pct": None,
           "gpu_mem_used_gb": None, "gpu_mem_total_gb": None}
    smi = shutil.which("nvidia-smi")
    if not smi:
        return out
    try:
        r = subprocess.run(
            [smi, "--query-gpu=name,utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=3,
        )
        name, util, used, total = [s.strip() for s in
                                   r.stdout.strip().splitlines()[0].split(",")]
        out.update(
            gpu_name=name,
            gpu_util_pct=float(util),
            gpu_mem_used_gb=round(float(used) * 2**20 / 1e9, 2),   # MiB -> GB
            gpu_mem_total_gb=round(float(total) * 2**20 / 1e9, 2),
        )
    except Exception:                                              # noqa: BLE001
        pass
    return out


def sample() -> Dict:
    """One resource snapshot, ~1 s cached. Keys are always present; values are
    None where the probe is unavailable."""
    global _cached, _cached_at
    with _lock:
        now = time.monotonic()
        if _cached and now - _cached_at < _CACHE_SEC:
            return _cached
        ram_used, ram_total = _ram()
        snap = {"cpu_pct": _cpu_pct(),
                "ram_used_gb": ram_used, "ram_total_gb": ram_total}
        snap.update(_gpu())
        _cached, _cached_at = snap, now
        return snap
