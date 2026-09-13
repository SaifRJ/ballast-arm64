from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum

class SamplingMode(str, Enum):
    SNAPSHOT = "snapshot"
    SAMPLED = "sampled"


class Task(str, Enum):
    BENCHMARK = "benchmark"
    OPTIMISE = "optimise"
    COMPARE = "compare"


class LogLevel(str, Enum):
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class CacheType(str, Enum):
    F32 = "f32"
    F16 = "f16"
    BF16 = "bf16"
    Q8_0 = "q8_0"
    Q5_0 = "q5_0"
    Q5_1 = "q5_1"
    Q4_0 = "q4_0"
    Q4_1 = "q4_1"
    IQ4_NL = "iq4_nl"


@dataclass
class EngineConfig:
    name: str
    source: str | None = None
    tag: str | None = None
    cmake_flags: dict = field(default_factory=dict)
    path: str | None = None
    bin_dir: Path | None = None


@dataclass
class ModelConfig:
    name: str
    source: str
    context_size: int
    cache_type_k: CacheType | None
    cache_type_v: CacheType | None
    fa: int
    batch_size: int
    generated_tokens: int
    swa_full: bool
    local_path: Path | None = None


@dataclass
class CorpusConfig:
    name: str
    source: str
    chunks: int | str
    local_path: Path | None = None


@dataclass
class RuntimeConfig:
    n_threads: int | str
    n_threads_batch: int | str


@dataclass
class PipelineConfig:
    mode: SamplingMode
    task: Task
    auto_repair: bool
    sample_interval_ms: int
    repeats: int
    warmup: int
    prompts: list[str]
    thread_scaling: list[int] | str | bool
    thread_scaling_prompt_tokens: int
    parallel_builds: int | str
    output_dir: str
    log_level: LogLevel


class Metric(str, Enum):
    PREFILL = "prefill"
    GENERATION = "generation"
    RAM_CPU = "ram_cpu"
    KV_CACHE = "kv_cache"
    PERPLEXITY = "perplexity"
    THREAD_SCALING = "thread_scaling"
    POWER_DRAW = "power_draw"
    HARDWARE_ID = "hardware_id"


ALL_METRICS: frozenset[Metric] = frozenset(Metric)


@dataclass
class MetricSet:
    enabled: frozenset[Metric]

    def __contains__(self, metric: Metric) -> bool:
        return metric in self.enabled

@dataclass
class BallastConfig:
    engines: list[EngineConfig]
    models: list[ModelConfig]
    corpora: list[CorpusConfig]
    runtime: RuntimeConfig
    pipeline: PipelineConfig
    metrics: MetricSet


@dataclass
class PrefillMetrics:
    prefill_tps: float | None
    prefill_ms: float | None
    prefill_tps_stddev: float | None = None


@dataclass
class GenerationMetrics:
    gen_tps: float | None
    ttft_ms: float | None
    gen_tps_stddev: float | None = None


@dataclass
class ResourceMetrics:
    peak_ram_mb: float | None
    avg_ram_mb: float | None
    cpu_pct: float | None
    sample_count: int


@dataclass
class KVUsage:
    kv_used_mb: float | None
    kv_utilisation: float | None