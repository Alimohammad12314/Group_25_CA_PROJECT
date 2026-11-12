"""One trace format shared by every model and every source.

A trace stores, for each request, the experts the router picked at every MoE layer:

    decode[r, t, l, :]   top-k expert ids of decode token t at MoE layer l   (int16, -1 = padding)
    warm[r, p, l, :]     the same for the request's last prompt (prefill) tokens

Prompt routing is kept separately because a real server sees it before decoding
starts: it is used only to warm up the predictors, never scored.
"""
import json
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Trace:
    model: str            # LLMSimulator model name: mixtral, grok1, llama4_maverick, deepseekR1
    source: str           # where the routing came from (dataset id or "synthetic:...")
    num_experts: int
    top_k: int
    decode: np.ndarray    # [R, T, L, k] int16
    dec_len: np.ndarray   # [R] valid decode tokens per request
    warm: np.ndarray      # [R, P, L, k] int16
    warm_len: np.ndarray  # [R] valid prompt tokens per request (right-aligned: the last warm_len rows)
    tags: list = field(default_factory=list)   # per-request label, e.g. "aime_2024/10"

    @property
    def num_layers(self):
        return self.decode.shape[2]

    @property
    def num_requests(self):
        return self.decode.shape[0]

    def save(self, path):
        meta = dict(model=self.model, source=self.source, num_experts=self.num_experts,
                    top_k=self.top_k, tags=self.tags)
        np.savez_compressed(path, decode=self.decode, dec_len=self.dec_len, warm=self.warm,
                            warm_len=self.warm_len, meta=np.array(json.dumps(meta)))

    @classmethod
    def load(cls, path):
        z = np.load(path, allow_pickle=False)
        meta = json.loads(str(z["meta"]))
        return cls(model=meta["model"], source=meta["source"], num_experts=meta["num_experts"],
                   top_k=meta["top_k"], decode=z["decode"], dec_len=z["dec_len"], warm=z["warm"],
                   warm_len=z["warm_len"], tags=meta.get("tags", []))

    def describe(self):
        return (f"{self.model}: {self.num_requests} requests, {self.num_layers} MoE layers, "
                f"{self.num_experts} experts, top-{self.top_k}, decode tokens median "
                f"{int(np.median(self.dec_len))}, source {self.source}")
