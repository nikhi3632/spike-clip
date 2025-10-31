# metrics.py
import torch
import psutil
import numpy as np
import pynvml

def compute_latency(latencies):
    latencies = np.array(latencies)
    return latencies.mean(), latencies.std()

def compute_throughput(num_samples, total_time):
    return num_samples / total_time if total_time > 0 else 0

def compute_memory_usage():
    if torch.cuda.is_available():
        mem = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB
        torch.cuda.reset_peak_memory_stats()
    else:
        mem = psutil.virtual_memory().used / (1024 ** 2)
    return mem

def compute_power_usage(gpu_index=0):
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
        power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # mW → W
        pynvml.nvmlShutdown()
        return power
    except Exception:
        return 0.0
