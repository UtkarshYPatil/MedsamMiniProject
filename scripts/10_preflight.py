"""
Check everything `docker compose up` needs, before spending ten minutes on it.

    python scripts/10_preflight.py
    python scripts/10_preflight.py --gpu

Every failure this catches is one we hit for real: a missing checkpoint that
only surfaced on the first job, a COPY source that did not exist, a port
already taken by a previous run, a GPU override on a host with no driver.

Exit code 0 means go. 1 means something will fail.
"""
import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

OK, WARN, FAIL = "ok", "warn", "fail"
_MARK = {OK: "  ok  ", WARN: " warn ", FAIL: " FAIL "}
_results = []
_quiet = False          # --json emits only JSON, so it can be piped


def check(name, state, detail=""):
    _results.append((name, state, detail))
    if not _quiet:
        print(f"[{_MARK[state]}] {name}" + (f"\n          {detail}" if detail else ""))


def run(cmd, timeout=60):
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return p.returncode, (p.stdout or "") + (p.stderr or "")
    except FileNotFoundError:
        return 127, "command not found"
    except subprocess.TimeoutExpired:
        return 124, "timed out"


# --------------------------------------------------------------------- checks

def check_docker():
    if not shutil.which("docker"):
        check("docker installed", FAIL, "install Docker Desktop and reopen this shell")
        return False
    code, out = run(["docker", "info", "--format", "{{.ServerVersion}}"], timeout=90)
    if code != 0:
        detail = "start Docker Desktop and wait for the whale to stop animating."
        first = out.strip().splitlines()[0][:120] if out.strip() else ""
        if first:
            detail += "\n          " + first
        check("docker daemon running", FAIL, detail)
        return False
    check("docker daemon running", OK, f"server {out.strip()}")

    code, out = run(["docker", "compose", "version"])
    if code != 0:
        check("docker compose v2", FAIL, "the bundled `docker compose` is missing")
        return False
    check("docker compose v2", OK, out.strip().splitlines()[0] if out.strip() else "")
    return True


def check_compose_files(gpu):
    files = ["docker-compose.yml"] + (["docker-compose.gpu.yml"] if gpu else [])
    for f in files:
        if not (ROOT / f).exists():
            check(f"{f} present", FAIL, "expected in the repository root")
            return
    args = []
    for f in files:
        args += ["-f", str(ROOT / f)]
    code, out = run(["docker", "compose"] + args + ["config", "-q"])
    if code != 0:
        check("compose files valid", FAIL, out.strip()[:400])
    else:
        check("compose files valid", OK, " + ".join(files))


def check_build_context():
    """Every COPY source must exist, or the build dies partway through."""
    import re
    missing = []
    for df in sorted((ROOT / "docker").glob("*.Dockerfile")):
        text = df.read_text()
        for m in re.finditer(r"^COPY\s+(?!--from)(.+)$", text, re.M):
            parts = [p for p in m.group(1).split() if not p.startswith("--")]
            for src in parts[:-1]:
                if any(c in src for c in "*?["):
                    if not list(ROOT.glob(src)):
                        missing.append(f"{df.name}: {src}")
                elif not (ROOT / src).exists():
                    missing.append(f"{df.name}: {src}")
    if missing:
        check("Dockerfile COPY sources exist", FAIL, "\n          ".join(missing))
    else:
        check("Dockerfile COPY sources exist", OK)


def check_checkpoint():
    dry = os.environ.get("DRY_RUN", "").strip().lower() in ("1", "true", "yes", "on")
    env_file = ROOT / ".env"
    if not dry and env_file.exists():
        for line in env_file.read_text().splitlines():
            if line.strip().startswith("DRY_RUN="):
                dry = line.split("=", 1)[1].strip().lower() in ("1", "true", "yes", "on")

    ckpt = ROOT / "checkpoints" / "MedSAM2_CTLesion.pt"
    if ckpt.exists():
        mb = ckpt.stat().st_size / 1e6
        if mb < 10:
            check("model checkpoint", FAIL,
                  f"{ckpt.name} is only {mb:.1f} MB - the download was truncated")
        else:
            check("model checkpoint", OK, f"{ckpt.name}, {mb:.0f} MB")
    elif dry:
        check("model checkpoint", WARN,
              "absent, but DRY_RUN is set: the dummy segmenter will run and "
              "every number it reports is made up")
    else:
        check("model checkpoint", FAIL,
              "checkpoints/MedSAM2_CTLesion.pt is missing, so the worker will "
              "refuse to start.\n          Run: python scripts/download_checkpoint.py"
              "\n          Or set DRY_RUN=1 to demonstrate the stack without it.")


def check_ports():
    for name, var, default in (("UI", "UI_PORT", 8501), ("API", "API_PORT", 8000)):
        port = int(os.environ.get(var, default))
        with socket.socket() as s:
            s.settimeout(1.0)
            taken = s.connect_ex(("127.0.0.1", port)) == 0
        if taken:
            check(f"port {port} free ({name})", WARN,
                  f"something is already listening. If it is an older run: "
                  f"docker compose down. Otherwise set {var} to another port.")
        else:
            check(f"port {port} free ({name})", OK)


def check_disk():
    try:
        free = shutil.disk_usage(ROOT).free / 1e9
    except OSError:
        return
    # The worker image carries torch; the GPU one carries CUDA as well.
    if free < 12:
        check("disk space", FAIL, f"{free:.1f} GB free; the images need roughly 12 GB")
    elif free < 25:
        check("disk space", WARN, f"{free:.1f} GB free; tight but probably enough")
    else:
        check("disk space", OK, f"{free:.0f} GB free")


def check_gpu():
    code, out = run(["docker", "run", "--rm", "--gpus", "all",
                     "nvidia/cuda:12.8.0-base-ubuntu24.04", "nvidia-smi"], timeout=180)
    if code == 0 and "Driver Version" in out:
        line = next((l for l in out.splitlines() if "Driver Version" in l), "")
        check("GPU visible to docker", OK, line.strip())
    else:
        check("GPU visible to docker", FAIL,
              "the GPU overlay will start a container that cannot see a card.\n"
              "          Install the NVIDIA Container Toolkit, or drop "
              "-f docker-compose.gpu.yml and run on CPU.")


# ----------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", action="store_true",
                    help="also check the GPU overlay and the host driver")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    a = ap.parse_args()

    global _quiet
    _quiet = a.json
    if not _quiet:
        print(f"preflight for {ROOT}\n")

    if check_docker():
        check_compose_files(a.gpu)
        check_ports()
        if a.gpu:
            check_gpu()
    check_build_context()
    check_checkpoint()
    check_disk()

    fails = [r for r in _results if r[1] == FAIL]
    warns = [r for r in _results if r[1] == WARN]

    if a.json:
        print(json.dumps([{"check": n, "state": s, "detail": d}
                          for n, s, d in _results], indent=2))
        return 1 if fails else 0

    print("\n" + "=" * 64)
    if fails:
        print(f"{len(fails)} blocking problem(s). Fix these first:")
        for n, _, _ in fails:
            print(f"  - {n}")
        return 1

    cmd = ("docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build"
           if a.gpu else "docker compose up --build")
    print(f"Ready{f' ({len(warns)} warning(s))' if warns else ''}. Start it with:\n")
    print(f"    {cmd}\n")
    print("    UI   http://localhost:8501")
    print("    API  http://localhost:8000/docs")
    print("    health http://localhost:8000/healthz")
    return 0


if __name__ == "__main__":
    sys.exit(main())
