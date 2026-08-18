<img src="assets/ballast-logo.svg" alt="Ballast for Arm64" style="width: 600px;" />

**Ballast** is a benchmarking and optimisation pipeline for LLM inference on Arm64 hardware. It evaluates model performance across the principal factors that affect inference throughput and resource usage, including memory footprint, prompt-processing throughput, token-generation rate, KV-cache behaviour, and thread scaling. It also provides a consistent framework for comparing inference engine builds, quantisation strategies, and configuration parameters under controlled conditions.

Rather than relying on isolated measurements or nominal performance figures, Ballast provides a repeatable and configurable benchmarking process that produces structured results suitable for direct comparison and further analysis.

## What it does

Given a `ballast.yaml` defining the models, engine builds, and measurement parameters, Ballast:

- **Provisions the environment**: clones and builds the specified `llama.cpp` configurations, downloads models from Hugging Face or accepts local GGUF files, and fetches any required evaluation content.
- **Runs the benchmark suite**: executes every combination of engine, model, prompt tier, and repeat, recording per-run metrics to timestamped CSV files.
- **Records reproducibility metadata**: captures the `llama.cpp` git SHA, CMake configuration flags, host CPU details, and other relevant information alongside each run.

The entire benchmark is configured through a single YAML file. Users specify what to benchmark, Ballast then manages the execution and measurement process.

## Features

- **Runs on Arm64 hardware:** Runs on hardware supported by `llama.cpp`, including Graviton, Ampere, Grace, Raspberry Pi, and Apple Silicon.
- **Supports GGUF models:** models hosted on Hugging Face as well as local GGUF files specified in `ballast.yaml`.
- **Multiple engine builds:** Compares baseline and optimised engine configurations in a single run, with independent CMake flags for each build.
- **Configurable measurements:** Context size, generation length, thread-count sweeps, repeat counts, and perplexity chunk counts are configured directly from YAML.
- **Reproducibility:** Records the `llama.cpp` commit SHA, build flags, and hardware fingerprint for every benchmark result.
- **Idempotent installation:** Re-running an unchanged configuration requires no additional work. Changes to models or engine configurations only trigger the necessary additional steps.

## Benchmarking Metrics

Ballast can record the following metrics for each benchmark run:

- **Peak RAM** — maximum resident set size (RSS) observed during the run.
- **Average RAM** — RSS sampled at 100 ms intervals and aggregated over the run.
- **CPU utilisation** — proportion of available CPU capacity consumed during inference.
- **Prompt-processing speed** — prefill throughput, measured in tokens per second.
- **Token generation speed** — decode throughput, measured in tokens per second.
- **Time-to-first-token** — latency between prompt submission and generation of the first output token.
- **Perplexity** — model performance measured against an evaluation corpus.
- **Thread scaling** — inference throughput as a function of thread count.
- **KV-cache** — allocated and used cache size, together with utilisation relative to the configured context window.

Model metadata, including parameter count, layer count, KV-head configuration, and quantisation types, is also captured automatically for each run.

## Run Modes and Tasks

Ballast supports multiple sampling modes and task types, configured through `ballast.yaml`.

**Sampling modes** define how measurements are collected during a run:

- **`standard`:** Performs a single measurement per model. Intended for rapid iteration and preliminary results.
- **`time-series`:** Performs extended benchmarking with continuous time-series sampling. This can identify thermal throttling, memory leaks, and KV-cache growth that may not be apparent in shorter runs. Expect long run times.

**Task types** define the purpose of the run:

- **`benchmark`:** Runs the specified engine × model × prompt matrix and records the configured measurements.
- **`optimise`:** Sweeps runtime configurations, including flags, thread counts, and cache quantisation, to identify the best-performing configuration for a given model and hardware platform.
- **`compare`:** Compares results against previous runs to identify statistically significant performance changes.

Ballast can also be integrated into CI pipelines to detect performance regressions when changes are made to the inference stack.

## Requirements

- Arm64 Linux (Ubuntu/Debian tested)
- Python 3.10 or newer
- Approximately 20GB of free disk space for a typical multi-model, multi-engine run

macOS support is planned.

## Setup

Clone the repository: 
~~~
git clone https://github.com/<owner>/ballast-arm64.git
~~~

Run the `bootstrap.sh` script:
~~~
cd ballast-arm64
./bootstrap.sh
~~~

Activate the environment:
~~~
source .venv/bin/activate
~~~

Finally, edit `ballast.yaml` to define the benchmark run, including the engines, models, and measurement parameters. A working default configuration is included.

Run the pipeline:
~~~
python3 -m ballast
~~~

On a first run, Ballast installs or locates the LLM models specified, builds the llama.cpp configurations entries, and fetches any evaluation content needed (e.g. perplexity corpus) before benchmarking. Subsequent runs will skip straight to benchmarking.

All results and outputs are saved to `results/Benchmark_<timestamp>/` - one folder is generated per each run of the pipeline.

## Configuration

`ballast.yaml` is the single source of truth for a benchmark run. The main configuration sections are:

- **`engines:`** One entry for each `llama.cpp` build to benchmark. Each entry specifies a source repository, git tag, and CMake flags. Pre-built `llama.cpp` directories can be referenced using `path:` instead.
- **`models:`** One entry for each model, specified by a GGUF URL or local file path.
- **`run_settings:`** Measurement parameters including context size, generation token count, repeat count, thread-scaling behaviour, and prompt/token settings.
- **`metrics:`** Specifies which measurements to collect. Model metadata is recorded for every run regardless of the selected metrics.

The default `ballast.yaml` included in the repository provides a working example that can be copied and modified.

## Project scope

Ballast enables users to compare configurations and identify which optimisation strategies provide measurable benefits on their specific hardware. Reported performance differences reflect the combined effect of engine build, quantisation, and runtime configuration. The separation of these contributions is the user's job, and Ballast is designed to make that separation possible and easy.

## Limitations

- Power draw measurement is not currently supported.
- Cross-platform support is not yet supported but is planned as part of a broader refactor toward direct `llama.cpp` integration.
- KV cache accounting assumes standard-attention transformer architectures. Models with exotic attention mechanisms (MLA, recurrent) report N/A for cache metrics.
- The TTFT metric is currently an approximation. Direct integration with `llama.cpp` is planned to provide more accurate measurement.

## License

This project falls under MIT Licensing. Please see LICENSE.md for more details.