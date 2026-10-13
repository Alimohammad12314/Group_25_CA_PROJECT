#!/usr/bin/env python3
import glob, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

d = sorted(glob.glob("results/part2_3/*/"))[-1]
s = pd.read_csv(d + "summary.csv")
out = d + "plots"; os.makedirs(out, exist_ok=True)
g5 = s[s.nvlink_gen == 5].sort_values("batch")
g4 = s[s.nvlink_gen == 4].sort_values("batch")
EXPERT_GB = 3 * 4096 * 14336 * 2 * 8 * 32 / 1e9   # Mixtral, FP16, all experts

def done(name, xl, yl, title):
    plt.xscale("log", base=2); plt.xlabel(xl); plt.ylabel(yl); plt.title(title)
    plt.grid(alpha=.3); plt.legend(); plt.tight_layout()
    plt.savefig(f"{out}/{name}.png", dpi=150); plt.close()

plt.figure(figsize=(6, 4))
plt.plot(g4.batch, g4.load_ms, "o-", label="load, 450 GB/s")
plt.plot(g5.batch, g5.load_ms, "o-", label="load, 900 GB/s")
plt.plot(g5.batch, g5.exec_offloadmode_ms, "s--", label="exec, offload mode")
plt.plot(g5.batch, g5.exec_nooffload_ms, "s--", label="exec, no offload")
plt.yscale("log")
done("1_load_vs_exec", "batch size", "expert time per step (ms, log)", "Expert loading vs execution")

plt.figure(figsize=(6, 4))
b = g5.batch
plt.plot(b, g5.implied_GB_loaded, "o-", label="simulator (900 GB/s)")
plt.plot(b, EXPERT_GB * (1 - 0.75 ** b), "k--", label="uniform-random top-2 expectation")
plt.axhline(EXPERT_GB, color="gray", ls=":", label=f"all experts ({EXPERT_GB:.1f} GB)")
done("2_gb_loaded", "batch size", "expert weights loaded per step (GB)", "Loaded bytes vs batch")

plt.figure(figsize=(6, 4))
plt.plot(g5.batch, g5.base_step_ms, "o-", label="ideal, all in HBM")
plt.plot(g5.batch, g5.framework_offload_step_ms, "o-", label="framework (perfect overlap)")
plt.plot(g5.batch, g5.serial_ondemand_ms, "o-", label="on-demand serial (load, then compute)")
done("3_step_latency", "batch size", "step latency (ms)", "Step latency definitions (900 GB/s)")

plt.figure(figsize=(6, 4))
plt.plot(g5.batch, g5.energy_base_J_tok, "o-", label="no offload")
plt.plot(g5.batch, g5.energy_offload_J_tok, "o-", label="offloaded (900 GB/s)")
plt.yscale("log")
done("4_energy", "batch size", "energy per token (J, log)", "Energy per token")
print("plots in", out)
