"""Run one command on a GPU with enough free memory (the machine is shared), retrying after CUDA out-of-memory.

usage: gpu_run.py --mem MiB [--max-per-gpu N] [--retries R] [--gpus 0 1 2 3] --log FILE -- cmd ...
The GPU is picked under a file lock: free = total - used - (memory announced by this launcher's jobs started in the
last 120 s) - 300 MiB margin; the GPU with the most free memory among those with free >= MiB and fewer than N of this
launcher's jobs is used. Slots: <R5>/gpu_slots/gpu<g>_<pid>.json. Exit code = the command's last exit code.
Never touches processes it did not start.
"""
import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path

SLOTS = Path.home() / "vins_gonogo_20260925" / "r5" / "gpu_slots"
OOM = ("CUDA out of memory", "OutOfMemoryError", "CUDA error: out of memory", "CUBLAS_STATUS_ALLOC_FAILED",
       "CUDNN_STATUS_NOT_INITIALIZED", "CUDA error: an illegal memory access")


def gpu_state():
    out = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.total,memory.used",
                                   "--format=csv,noheader,nounits"], text=True)
    return {int(a): (int(b), int(c)) for a, b, c in (line.split(",") for line in out.strip().splitlines())}


def my_slots():
    res = {}
    for f in SLOTS.glob("gpu*_*.json"):
        try:
            d = json.loads(f.read_text())
            os.kill(d["pid"], 0)
        except (ProcessLookupError, json.JSONDecodeError, FileNotFoundError, KeyError):
            f.unlink(missing_ok=True)
            continue
        except PermissionError:
            pass
        res.setdefault(d["gpu"], []).append(d)
    return res


def pick(mem, max_per, gpus):
    st, slots, now = gpu_state(), my_slots(), time.time()
    best = None
    for g in gpus:
        total, used = st[g]
        mine = slots.get(g, [])
        if len(mine) >= max_per:
            continue
        pending = sum(d["mem"] for d in mine if now - d["start"] < 120)
        free = total - used - pending - 300
        if free >= mem and (best is None or free > best[1]):
            best = (g, free)
    return best


def main():
    argv = sys.argv[1:]
    if "--" not in argv:
        raise SystemExit("usage: gpu_run.py [options] --log FILE -- cmd ...")
    cut = argv.index("--")
    parser = argparse.ArgumentParser()
    parser.add_argument("--mem", type=int, required=True)
    parser.add_argument("--max-per-gpu", type=int, default=3)
    parser.add_argument("--retries", type=int, default=8)
    parser.add_argument("--gpus", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--log", required=True)
    opts = parser.parse_args(argv[:cut])
    cmd = argv[cut + 1:]
    SLOTS.mkdir(parents=True, exist_ok=True)
    log = Path(opts.log)
    log.parent.mkdir(parents=True, exist_ok=True)
    rc = 1
    for attempt in range(opts.retries + 1):
        while True:
            with open(SLOTS / "pick.lock", "w") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                choice = pick(opts.mem, opts.max_per_gpu, opts.gpus)
                if choice is not None:
                    g = choice[0]
                    env = os.environ.copy()
                    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
                    env["CUDA_VISIBLE_DEVICES"] = str(g)
                    offset = log.stat().st_size if log.exists() else 0
                    handle = open(log, "a")
                    handle.write(f"[gpu_run] attempt {attempt} on GPU {g} at {time.strftime('%H:%M:%S')}\n")
                    handle.flush()
                    proc = subprocess.Popen(cmd, env=env, stdout=handle, stderr=subprocess.STDOUT)
                    slot = SLOTS / f"gpu{g}_{proc.pid}.json"
                    slot.write_text(json.dumps({"pid": proc.pid, "gpu": g, "mem": opts.mem, "start": time.time(),
                                                "cmd": " ".join(cmd)[:300]}))
                    break
            time.sleep(20)
        rc = proc.wait()
        handle.close()
        slot.unlink(missing_ok=True)
        if rc == 0:
            return 0
        with open(log, errors="replace") as fh:
            fh.seek(offset)
            text = fh.read()
        if not any(m in text for m in OOM):
            return rc
        with open(log, "a") as fh:
            fh.write(f"[gpu_run] out of memory on GPU {g}; retry in 60 s\n")
        time.sleep(60)
    return rc


if __name__ == "__main__":
    sys.exit(main())
