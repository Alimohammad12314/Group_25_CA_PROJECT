"""Diagnostic: one short decode per configuration, full result dicts (not part of the results)."""
import json
import sys

from engine.timeline import SimConfig

from .common import CALIBRATION, MODEL_DIR, doc_tokens, gpu_workload, sim_overrides, stream_for
from .runtime import Runtime
import os

if __name__ == "__main__":
    batch, ctx, tokens = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    configs = sys.argv[4].split(",")
    ratio = float(sys.argv[5]) if len(sys.argv) > 5 else 0.25
    ids, _ = doc_tokens()
    rt = Runtime(MODEL_DIR, slab_slots=128, verbose=False)
    wl = gpu_workload(batch, ctx, groups=1, tokens=tokens)
    needed = wl.needed()
    for name in configs:
        pred, kw = sim_overrides(name, ratio)
        st = stream_for(wl, pred, f"ctx{ctx}") if pred else None
        cal = CALIBRATION if os.path.exists(CALIBRATION) else None
        res, live, p = rt.run(wl, needed, st, SimConfig(gpu="H100", link="PCIe5", calibration=cal, context=ctx, **kw),
                              ids, ctx, progress=False)
        res.pop("layer_hit_rate")
        print(name, json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in res.items()}), flush=True)
    rt.close()
