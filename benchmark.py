"""Measure SSM-Inception size, FLOPs, latency, throughput, and process RSS."""

from __future__ import annotations

import argparse
import io
import json
import statistics
import threading
import time

import numpy as np
import psutil
import torch
from thop import profile

from ssm_inception import SSMInception


class RSSSampler:
    def __init__(self, interval_seconds: float = 0.001):
        self.interval = interval_seconds
        self.process = psutil.Process()
        self.baseline = self.process.memory_info().rss
        self.peak = self.baseline
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self):
        while not self.stop_event.is_set():
            self.peak = max(self.peak, self.process.memory_info().rss)
            time.sleep(self.interval)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop_event.set()
        self.thread.join()
        self.peak = max(self.peak, self.process.memory_info().rss)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=1000)
    args = parser.parse_args()

    if args.device == "cpu":
        torch.set_num_threads(args.threads)
        torch.set_num_interop_threads(1)
    device = torch.device(args.device)
    torch.manual_seed(9527)
    model = SSMInception().to(device).eval()
    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    sample = torch.randn(1, 45, 125, device=device)

    thop_flops, _ = profile(model, inputs=(sample,), verbose=False)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)

    times_ms = []
    with torch.inference_mode():
        for _ in range(args.warmup):
            model(sample)
        synchronize(device)
        with RSSSampler() as memory:
            for _ in range(args.iterations):
                synchronize(device)
                start = time.perf_counter_ns()
                model(sample)
                synchronize(device)
                times_ms.append((time.perf_counter_ns() - start) / 1.0e6)

    result = {
        "device": str(device),
        "input_shape": [1, 45, 125],
        "dtype": "float32",
        "warmup_iterations": args.warmup,
        "measured_iterations": args.iterations,
        "torch_cpu_threads": args.threads if device.type == "cpu" else None,
        "parameters": parameters,
        "serialized_state_dict_kib": len(buffer.getvalue()) / 1024.0,
        "flops_m_thop": thop_flops / 1.0e6,
        "latency_mean_ms": statistics.mean(times_ms),
        "latency_median_ms": statistics.median(times_ms),
        "latency_p95_ms": float(np.percentile(times_ms, 95)),
        "throughput_samples_per_second": 1000.0 / statistics.mean(times_ms),
        "process_rss_baseline_mib": memory.baseline / 1024.0**2,
        "process_rss_peak_mib": memory.peak / 1024.0**2,
        "process_rss_increment_mib": (memory.peak - memory.baseline) / 1024.0**2,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
