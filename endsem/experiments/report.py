"""Headline tables from the sweep results (run after e4 and e5).

    python -m experiments.report

Writes results/headline.csv: per model, batch, link and cache setting, the token latency of
on-demand loading, the best policy without prediction, prediction for eviction only, the full
design and the oracle predictor, plus the full design's traffic, prefetch quality and energy.
"""
import os

import pandas as pd

from .common import RESULTS

BASELINES = ["LRU", "LRU + overlap", "LFU + overlap"]


def main():
    df = pd.read_csv(os.path.join(RESULTS, "e4_sweeps", "sweeps.csv"))
    if "error" in df:
        df = df[df["error"].isna()]
    df = df[df.sweep == "main"]
    key = ["model", "batch", "link"]
    ond = df[df.policy == "on-demand"].set_index(key)
    rows = []
    for (m, b, link, cache), g in df[df.cache != "0%"].groupby(key + ["cache"]):
        g = g.set_index("policy")
        base = g.loc[[p for p in BASELINES if p in g.index]]
        best_name = base.token_latency_ms.idxmin()
        full = g.loc["full design"]
        o = ond.loc[(m, b, link)]
        rows.append(dict(model=m, batch=b, link=link, cache=cache, cache_ratio=full.cache_ratio,
                         ondemand_ms=o.token_latency_ms,
                         best_no_prediction=best_name, best_no_prediction_ms=base.token_latency_ms.min(),
                         predict_evict_only_ms=g.loc["predict-evict only"].token_latency_ms,
                         full_ms=full.token_latency_ms,
                         oracle_ms=g.loc["oracle"].token_latency_ms,
                         full_vs_ondemand=o.token_latency_ms / full.token_latency_ms,
                         full_vs_best_baseline=base.token_latency_ms.min() / full.token_latency_ms,
                         oracle_vs_best_baseline=base.token_latency_ms.min() / g.loc["oracle"].token_latency_ms,
                         full_hit_rate=full.hit_rate, full_stall_fraction=full.stall_fraction,
                         full_prefetch_accuracy=full.prefetch_accuracy,
                         full_prefetch_timely=full.prefetch_timely_frac,
                         full_GB_per_token=full.GB_per_token,
                         full_wasted_GB_per_token=full.wasted_GB_per_token,
                         best_baseline_GB_per_token=base.loc[best_name].GB_per_token,
                         full_J_per_token=full.energy_J_per_token,
                         best_baseline_J_per_token=base.loc[best_name].energy_J_per_token,
                         ondemand_J_per_token=o.energy_J_per_token))
    H = pd.DataFrame(rows)
    H.to_csv(os.path.join(RESULTS, "headline.csv"), index=False)
    pd.set_option("display.width", 250)
    show = ["model", "batch", "link", "cache", "ondemand_ms", "best_no_prediction", "best_no_prediction_ms",
            "predict_evict_only_ms", "full_ms", "oracle_ms", "full_vs_best_baseline", "oracle_vs_best_baseline"]
    print(H[H.cache.isin(["physical", "25%"])][show].round(2).to_string(index=False))
    print("\nfull design vs best no-prediction baseline, all settings:")
    print(H.groupby("model").full_vs_best_baseline.describe()[["mean", "min", "max"]].round(3).to_string())
    print("\nshare of settings where the full design is at least as fast as the best baseline:",
          round(float((H.full_vs_best_baseline >= 0.995).mean()), 3))


if __name__ == "__main__":
    main()
