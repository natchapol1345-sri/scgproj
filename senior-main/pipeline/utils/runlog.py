"""Per-run metrics logger — appends one CSV row per pipeline invocation."""
import csv
import datetime
import os
import platform
import sys
import threading
import time

import torch

try:
    import psutil
except ImportError:
    psutil = None


LOG_COLUMNS = [
    "timestamp", "command", "input", "output", "device", "gpu_name", "platform",
    "total_time_s", "inference_time_s", "cpu_count", "cpu_load_pct",
    "ram_used_gb", "ram_peak_gb", "vram_used_gb", "vram_peak_gb",
    "obj_vol_cm3", "box_vol_cm3",
]


def _fmt(v, spec):
    return format(v, spec) if isinstance(v, (int, float)) else ""


class RunLogger:
    """Sample CPU/RAM in background, capture VRAM peak via torch counters, write row on stop."""

    def __init__(self, log_path, device, sample_interval=0.5):
        self.log_path = log_path
        self.device = str(device)
        self.dkind = self.device.split(":")[0]
        self.t0 = time.time()
        self.sample_interval = sample_interval
        self.proc = psutil.Process() if psutil else None
        self.ram_peak_bytes = 0
        self.cpu_samples = []
        self._stop = threading.Event()
        self._thread = None
        if self.dkind == "cuda" and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def start(self):
        if self.proc is None:
            return
        # Prime cpu_percent so subsequent calls return deltas.
        self.proc.cpu_percent(interval=None)
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def _sample_loop(self):
        while not self._stop.is_set():
            try:
                rss = self.proc.memory_info().rss
                if rss > self.ram_peak_bytes:
                    self.ram_peak_bytes = rss
                pct = self.proc.cpu_percent(interval=None)
                self.cpu_samples.append(pct)
            except Exception:
                pass
            self._stop.wait(self.sample_interval)

    def _gpu_name(self):
        if self.dkind == "cuda" and torch.cuda.is_available():
            try:
                return torch.cuda.get_device_name(0)
            except Exception:
                return "cuda"
        if self.dkind == "mps":
            return "Apple MPS"
        return ""

    def _vram(self):
        if self.dkind == "cuda" and torch.cuda.is_available():
            used = torch.cuda.memory_allocated() / (1024 ** 3)
            peak = torch.cuda.max_memory_allocated() / (1024 ** 3)
            return used, peak
        if self.dkind == "mps":
            try:
                used = torch.mps.current_allocated_memory() / (1024 ** 3)
                peak = torch.mps.driver_allocated_memory() / (1024 ** 3)
                return used, peak
            except Exception:
                return None, None
        return None, None

    def stop_and_write(self, input_path, output_path, inference_time,
                       obj_vol_cm3=None, box_vol_cm3=None):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

        total_time = time.time() - self.t0

        ram_used = ram_peak = cpu_load = None
        if self.proc is not None:
            try:
                rss_now = self.proc.memory_info().rss
                ram_used = rss_now / (1024 ** 3)
                ram_peak = max(self.ram_peak_bytes, rss_now) / (1024 ** 3)
            except Exception:
                pass
            if self.cpu_samples:
                cpu_load = sum(self.cpu_samples) / len(self.cpu_samples)

        vram_used, vram_peak = self._vram()

        row = {
            "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
            "command": " ".join(sys.argv),
            "input": input_path,
            "output": output_path,
            "device": self.device,
            "gpu_name": self._gpu_name(),
            "platform": platform.platform(),
            "total_time_s": _fmt(total_time, ".2f"),
            "inference_time_s": _fmt(inference_time, ".2f") if inference_time is not None else "",
            "cpu_count": os.cpu_count() or "",
            "cpu_load_pct": _fmt(cpu_load, ".1f"),
            "ram_used_gb": _fmt(ram_used, ".2f"),
            "ram_peak_gb": _fmt(ram_peak, ".2f"),
            "vram_used_gb": _fmt(vram_used, ".2f"),
            "vram_peak_gb": _fmt(vram_peak, ".2f"),
            "obj_vol_cm3": _fmt(obj_vol_cm3, ".4f") if obj_vol_cm3 is not None else "",
            "box_vol_cm3": _fmt(box_vol_cm3, ".4f") if box_vol_cm3 is not None else "",
        }

        new_file = not os.path.exists(self.log_path)
        os.makedirs(os.path.dirname(self.log_path) or ".", exist_ok=True)
        with open(self.log_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=LOG_COLUMNS)
            if new_file:
                writer.writeheader()
            writer.writerow(row)
        print(f"  Logged run → {self.log_path}")
