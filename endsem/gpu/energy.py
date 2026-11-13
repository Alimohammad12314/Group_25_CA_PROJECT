"""Energy counters around a run: NVML total energy (GPU) and RAPL (host) when readable.

GPU   pynvml.nvmlDeviceGetTotalEnergyConsumption, in mJ since the driver loaded. Use the counter,
      not sampled power: it integrates every millisecond, including short stalls.
Host  /sys/class/powercap/intel-rapl:*/energy_uj (package and, on Intel, DRAM). On this machine
      (AMD EPYC) the files exist but are root-only, so host energy is reported as unavailable.
"""
import glob
import os
import time


def _gpu_index():
    vis = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
    return int(vis) if vis.isdigit() else 0


class GPUEnergy:
    def __init__(self, index=None):
        import pynvml
        pynvml.nvmlInit()
        self.nv = pynvml
        self.h = pynvml.nvmlDeviceGetHandleByIndex(_gpu_index() if index is None else index)

    def read(self):
        """Joules since driver load."""
        return self.nv.nvmlDeviceGetTotalEnergyConsumption(self.h) * 1e-3

    def power(self):
        return self.nv.nvmlDeviceGetPowerUsage(self.h) * 1e-3

    def sample_power(self, seconds, fn=None):
        """Mean power over `seconds` from the energy counter (fn runs in a loop meanwhile)."""
        e0, t0 = self.read(), time.time()
        while time.time() - t0 < seconds:
            if fn is not None:
                fn()
            else:
                time.sleep(0.05)
        e1, t1 = self.read(), time.time()
        return (e1 - e0) / (t1 - t0)


class HostEnergy:
    """Sum of readable RAPL domains, in joules; available=False if none is readable."""

    def __init__(self):
        self.files = []
        for d in sorted(glob.glob("/sys/class/powercap/intel-rapl:*")):
            f = os.path.join(d, "energy_uj")
            try:
                with open(f) as fh:
                    fh.read()
                self.files.append(f)
            except OSError:
                pass
        self.available = bool(self.files)

    def read(self):
        if not self.available:
            return float("nan")
        tot = 0.0
        for f in self.files:
            with open(f) as fh:
                tot += int(fh.read()) * 1e-6
        return tot
