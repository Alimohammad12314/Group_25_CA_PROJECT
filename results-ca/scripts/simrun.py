"""Shared helpers: run one LLMSimulator configuration in an isolated directory.

Both simulator binaries read ./config.yaml (and the SSD-offloading copy re-reads it
inside every linear layer), plus ./dram_config_HBM3*.yaml, from the current working
directory. Each run therefore gets its own temporary directory, which lets us run
many configurations in parallel and keep the exact config next to the result.
"""
import copy
import glob
import os
import shutil
import subprocess
import tempfile
from multiprocessing import Pool

import pandas as pd
import yaml

RESULTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # results-ca/
REPO = os.path.dirname(RESULTS)                                         # repository root


def _first_built(*candidates):
    """First simulator root that has a built ./build/run (falls back to the first candidate)."""
    for c in candidates:
        if c and os.path.exists(os.path.join(c, "build", "run")):
            return c
    return candidates[0]


# original LLMSimulator (Duplex, Logic-PIM): submodule external/LLMSimulator, set up by setup.sh
LLMSIM = os.environ.get("LLMSIM_ROOT") or _first_built(
    os.path.join(REPO, "external", "LLMSimulator"), os.path.expanduser("~/LLMSimulator"))
# SSD-offloading copy (GPU-only + expert offloading): submodule LLMSimulator
SSDSIM = os.environ.get("SSDSIM_ROOT") or os.path.join(REPO, "LLMSimulator")
# exact base configs used for the published results (all relevant keys are overridden per run)
LLMSIM_BASE_CFG = os.path.join(RESULTS, "configs", "llmsim_base_config.yaml")
SSD_BASE_CFG = os.path.join(RESULTS, "configs", "ssd_base_config.yaml")


def deep_update(d, u):
    for k, v in u.items():
        if isinstance(v, dict):
            d[k] = deep_update(d.get(k, {}) or {}, v)
        else:
            d[k] = v
    return d


def run_config(args):
    """args = (sim_root, base_cfg, overrides, archive_dir). Returns (overrides, DataFrame|None, log)."""
    sim_root, base, over, archive = args
    cfg = deep_update(copy.deepcopy(base), over)
    cfg["log"]["output_directory"] = "./out"
    work = tempfile.mkdtemp(prefix="sim_")
    try:
        for f in glob.glob(os.path.join(sim_root, "dram_config_HBM3*.yaml")):
            shutil.copy(f, work)
        os.makedirs(os.path.join(work, "out"))
        with open(os.path.join(work, "config.yaml"), "w") as fh:
            yaml.safe_dump(cfg, fh, sort_keys=False)
        p = subprocess.run([os.path.join(sim_root, "build", "run")], cwd=work,
                           capture_output=True, text=True)
        csvs = glob.glob(os.path.join(work, "out", "*.csv"))
        df = pd.read_csv(csvs[0]) if (p.returncode == 0 and csvs) else None
        if archive:
            os.makedirs(archive, exist_ok=True)
            shutil.copy(os.path.join(work, "config.yaml"), os.path.join(archive, "config.yaml"))
            with open(os.path.join(archive, "run.log"), "w") as fh:
                fh.write(p.stdout[-20000:] + p.stderr[-5000:])
            if csvs:
                shutil.copy(csvs[0], os.path.join(archive, "output.csv"))
            for extra in glob.glob(os.path.join(work, "*.csv")):   # routing logs
                shutil.copy(extra, archive)
        return over, df, p.stdout[-2000:] + p.stderr[-2000:]
    finally:
        shutil.rmtree(work, ignore_errors=True)


def run_many(jobs, procs=28):
    with Pool(procs) as pool:
        return pool.map(run_config, jobs)


def load_yaml(path):
    with open(path) as fh:
        return yaml.safe_load(fh)
