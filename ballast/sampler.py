from ballast.schema import SamplingMode
import psutil
import threading
import time
import csv
import os
import logging
from pathlib import Path

# Poll process memory and CPU at a configured ms interval
# Uses two different modes: 
# "snapshot" collects samples internally and returns aggregates (peak_ram, avg_ram, cpu_pct)
# "sampled" writes every sample for a given millisecond interval to a CSV for time-series analysis

log = logging.getLogger("ballast")

class ResourceSampler:

    def __init__(self, pid: int | None = None, interval_ms: int = 100, mode: SamplingMode = SamplingMode.SNAPSHOT, csv_path: Path | None = None, tag: str | None = None, run_id: str = "", run_timestamp: str = "", engine_name: str = "", model_name: str = "", prompt: str = "", repeat: int = 0, phase: str = ""):

        self.pid: int = pid if pid is not None else os.getpid()
        self.interval_s: float = interval_ms / 1000.0
        self.mode: SamplingMode = mode
        self.csv_path: Path | None = Path(csv_path) if csv_path else None
        self.tag: str | None = tag

        # list of (timestamp_ns, rss_bytes, cpu_pct)
        self._samples: list[tuple[int, int, float]] = []
        self._thread: threading.Thread | None = None
        self._stop: threading.Event = threading.Event()
        self._proc: psutil.Process | None = None
        self._csv_writer = None
        self._csv_file = None
        self.run_id = run_id
        self.run_timestamp = run_timestamp
        self.engine_name = engine_name
        self.model_name = model_name
        self.prompt = prompt
        self.repeat = repeat
        self.phase = phase

    def __enter__(self) -> "ResourceSampler":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()

    def start(self) -> None:
        self._proc = psutil.Process(self.pid)
        self._proc.cpu_percent(interval=None)

        if self.mode is SamplingMode.SAMPLED:
            if self.csv_path is None:
                raise ValueError("sampled mode requires csv_path")
            self._csv_file = open(self.csv_path, "a", newline="")
            self._csv_writer = csv.writer(self._csv_file)

        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    
    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=self.interval_s * 5)
        
        if self._csv_file:
            self._csv_file.close()
            self._csv_file = None

    def _run(self) -> None:

        assert self._proc is not None
        assert self.mode is not SamplingMode.SAMPLED or self._csv_writer is not None

        while not self._stop.is_set():
            try:
                rss = self._proc.memory_info().rss
                cpu = self._proc.cpu_percent(interval=None)
                ts = time.perf_counter_ns()
            except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
                log.warning(f"ResourceSampler: process gone or access denied ({e})")
                break

            if self.mode is SamplingMode.SNAPSHOT:
                self._samples.append((ts, rss, cpu))
            else:
                self._csv_writer.writerow([
                    self.run_id, self.run_timestamp, ts,
                    self.engine_name, self.model_name, self.prompt, self.repeat, self.phase,
                    round(rss / (1024**2), 2), cpu
                ])

            self._stop.wait(self.interval_s)

    def aggregate(self) -> dict:

        if not self._samples:
            log.warning("ResourceSampler: no samples collected")
            return {"peak_ram_mb": None, "avg_ram_mb": None, "cpu_pct": None}

        rss_values = [s[1] for s in self._samples]
        cpu_values = [s[2] for s in self._samples if s[2] > 0]  # drop priming zeros

        return {
            "peak_ram_mb": round(max(rss_values) / (1024**2), 2),
            "avg_ram_mb": round((sum(rss_values) / len(rss_values)) / (1024**2), 2),
            "cpu_pct": round(sum(cpu_values) / len(cpu_values), 2) if cpu_values else 0
        }
