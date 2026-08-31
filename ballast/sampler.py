import psutil
import threading
import time
import csv
import os
import logging
from pathlib import Path

log = logging.getLogger("ballast")


# Poll process memory and CPU at a configured ms interval
# Uses two different modes: 
# "snapshot" collects samples internally and returns aggregates (peak_ram, avg_ram, cpu_pct)
# "sampled" writes every sample for a given millisecond interval to a CSV for time-series analysis

class ResourceSampler:

    def __init__(self, pid=None, interval_ms=100, mode="snapshot", csv_path=None, tag=None):

        self.pid = pid if pid is not None else os.getpid()
        self.interval_s = interval_ms / 1000.0
        self.mode = mode
        self.csv_path = Path(csv_path) if csv_path else None
        self.tag = tag

        # list of (timestamp_ns, rss_bytes, cpu_pct)
        self._samples = []
        self._thread = None
        self._stop = threading.Event()
        self._proc = None
        self._csv_writer = None
        self._csv_file = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop()

    def start(self):
        self._proc = psutil.Process(self.pid)
        self._proc.cpu_percent(interval=None)

        if self.mode == "sampled":
            if self.csv_path is None:
                log.warning("sampled mode without csv_path, samples will be discarded.")
            else:
                self._csv_file = open(self.csv_path, "a", newline="")
                self._csv_writer = csv.writer(self._csv_file)
                if self.csv_path.stat().st_size == 0:
                    self._csv_writer.writerow(["timestamp_ns", "tag", "rss_mb", "cpu_pct"])

        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=self.interval_s * 5)
        
        if self._csv_file:
            self._csv_file.close()
            self._csv_file = None

    def _run(self):
        while not self._stop.is_set():
            try:
                rss = self._proc.memory_info().rss
                cpu = self._proc.cpu_percent(interval=None)
                ts = time.perf_counter_ns()
            except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
                log.warning(f"ResourceSampler: process gone or access denied ({e})")
                break

            if self.mode == "snapshot":
                self._samples.append((ts, rss, cpu))
            else:
                self._csv_writer.writerow([ts, self.tag or "", round(rss / (1024**2), 2), cpu])

            self._stop.wait(self.interval_s)

    def aggregate(self):

        if not self._samples:
            log.warning("ResourceSampler: no samples collected")
            return {"peak_ram_mb": None, "avg_ram_mb": None, "cpu_pct": None, "sample_count": 0}

        rss_values = [s[1] for s in self._samples]
        cpu_values = [s[2] for s in self._samples if s[2] > 0]  # drop priming zeros

        return {
            "peak_ram_mb": round(max(rss_values) / (1024**2), 2),
            "avg_ram_mb": round((sum(rss_values) / len(rss_values)) / (1024**2), 2),
            "cpu_pct": round(sum(cpu_values) / len(cpu_values), 2) if cpu_values else 0,
            "sample_count": len(self._samples)
        }
