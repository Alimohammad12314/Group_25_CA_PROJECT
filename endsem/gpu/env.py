"""Record the machine (GPU.md section 0) as results/env.json and results/env.txt.

    python -m gpu.env          (inside gpurun, so the GPU is visible to torch)
"""
import json
import os
import platform
import resource
import shutil
import subprocess

from .common import RESULTS


def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60).stdout.strip()
    except Exception as exc:          # a missing tool must not stop the record
        return f"unavailable: {exc}"


def collect():
    import torch
    env = dict(
        gpu=sh("nvidia-smi --query-gpu=index,name,memory.total,power.limit,clocks.max.sm,pci.bus_id,"
               "pcie.link.gen.max,pcie.link.gen.current,pcie.link.width.max,pcie.link.width.current "
               "--format=csv"),
        driver=sh("nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1"),
        topo=sh("nvidia-smi topo -m"),
        cpu=sh("lscpu | grep -i 'model name\\|socket\\|numa'"),
        mem=sh("free -g"),
        memlock_soft_hard=list(resource.getrlimit(resource.RLIMIT_MEMLOCK)),
        disk=sh("df -h ."),
        python=platform.python_version(),
        torch=torch.__version__, torch_cuda=torch.version.cuda,
        cuda_device=torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        cuda_visible=os.environ.get("CUDA_VISIBLE_DEVICES"),
        numactl=sh("numactl --show"),
        clocks_locked=False,
        rapl_readable=any(os.access(f, os.R_OK) for f in
                          __import__("glob").glob("/sys/class/powercap/intel-rapl:*/energy_uj")),
    )
    try:
        import transformers
        env["transformers"] = transformers.__version__
    except ImportError:
        pass
    env["has_numactl"] = shutil.which("numactl") is not None
    return env


def main():
    os.makedirs(RESULTS, exist_ok=True)
    env = collect()
    with open(os.path.join(RESULTS, "env.json"), "w") as f:
        json.dump(env, f, indent=1)
    with open(os.path.join(RESULTS, "env.txt"), "w") as f:
        for k, v in env.items():
            f.write(f"## {k}\n{v}\n\n")
    print(open(os.path.join(RESULTS, "env.txt")).read())


if __name__ == "__main__":
    main()
