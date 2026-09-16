<img src="assets/ballast-logo.svg" alt="Ballast for Arm64" style="width: 600px;" />

**Ballast** is a benchmarking and optimisation pipeline for LLM inference on Arm64 hardware. It evaluates model performance across the principal factors that affect inference throughput and resource usage, including memory footprint, prompt-processing throughput, token-generation rate, KV-cache behaviour, and thread scaling. It also provides a consistent framework for comparing inference engine builds, quantisation strategies, and configuration parameters under controlled conditions.

> Designed for engineering and research, Ballast provides a repeatable and configurable benchmarking process with an emphasis on reproducibility, comparison, and analysis. If you experience any bugs working with Ballast or would like to suggest a new feature, feel free to contact me at saif.alhmoud.dev@gmail.com (or continue building Ballast yourself, it is under an MIT License!)

## What does Ballast do?

Ballast automates the setup and evaluation of `llama.cpp` on Arm64 hardware end to end. Using a user-configured `ballast.yaml` file, it clones and builds llama.cpp under multiple CMake configurations, downloads the models and evaluation corpora you specify, runs a benchmark suite across the resulting matrix, and writes structured CSV outputs alongside a full audit trail of the engine builds, host, and configuration used.

- **Runs on Arm64 hardware.** Any target `llama.cpp` supports, including Graviton, Ampere, Grace, Neoverse-class servers, Raspberry Pi, and Apple Silicon.
- **Multiple engine builds in a single run.** Benchmark performance on baseline and optimised engine configurations, with independent CMake flags per build (e.g. KleidiAI on vs. off). Manifests capture the git SHA, resolved flags, and build date for every engine.
- **Bring your own llama.cpp, or let Ballast build it.** Point at an existing installation via a local path, or hand Ballast a source URL and tag and it will clone and build it from scratch.
- **GGUF models from anywhere.** URLs (Hugging Face and elsewhere) or local files, configured per model in YAML.
- **Comprehensive metric list.** Prefill throughput, decode throughput, TTFT, RAM, CPU utilisation, perplexity, thread scaling, KV cache utilisation. 
- **Supports different sampling methodologies.** Snapshot mode takes one measurement and can aggregate over each measurement window; sampled mode records measurements at a configurable interval (100 ms default) throughout the inference process.
- **Full reproducibility.** Every run folder contains engine manifests, model metadata, host hardware details, and benchmark results in CSV format. A run can be reproduced from the manifests alone.

## What metrics does it capture?

**Ballast** can record the following metrics for each benchmark run.

### OS Resources & Hardware

| Metric | Description |
|---|---|
| **Peak RAM** | Maximum resident set size (RSS) observed during the run. |
| **Average RAM** | RSS sampled at ms intervals (ms interval is set by the user) and aggregated over the run. |
| **CPU utilisation** | Proportion of available CPU capacity consumed during inference. |
| **Power draw** | System power consumed during inference, measured in watt-hours. |
| **CPU thermal temperatures** | CPU temperatures exposed by system sensors, measured in °C. |
| **Hardware details** | Hardware information captured from the host system. |

### LLM Performance

| Metric | Description |
|---|---|
| **Prompt-processing speed** | Prefill throughput, measured in tokens per second. |
| **Token-generation speed** | Decode throughput, measured in tokens per second. |
| **Time-to-first-token (TTFT)** | Latency between prompt submission and generation of the first output token. |
| **Perplexity** | Measure of how well the model predicts tokens against an evaluation corpus. |
| **Thread scaling** | Inference throughput as a function of thread count. |
| **KV-cache** | Allocated and used cache size relative to the configured context window. |

In addition to performance metrics, **Ballast** also captures engine manifests and model metadata, which includes parameter count, layer count, KV-head configuration, and quantisation types, for each run.

## Configuration

Ballast is configured via the `ballast.yaml` file. Users can edit this file to control the pipeline. 

Configuration is divided into six main sections: 

| Section | Description |
|---|---|
| `engines` | Specify the engine source, version, build configuration, and CMake flags |
| `models` | Specify the GGUF models, context size, batch size, KV-cache configuration, and generation settings |
| `corpora` | Specify which evaluation corpus to use and how many chunks to measure perplexity. |
| `runtime` | Used to configure threads used at inference |
| `pipeline` | Sampling modes, benchmarking/optimisation tasks, user prompts, prompt repetition, warmups, thread scaling settings, parallel builds, and output directory |
| `metrics` | Defines what system and inference measurements to collect. |

**Sampling modes** define how measurements are collected during a run:

- `snapshot` Performs a single 'snapshot' measurement per model. Intended for rapid iteration and preliminary results.
- `sampled` Performs extended benchmarking with continuous time-series sampling. This can identify thermal throttling, memory leaks, and KV-cache growth that may not be apparent in shorter runs. Expect long run times.

**Task types** define the purpose of the run:

- `benchmark` Runs the specified engine × model × prompt matrix and records the configured measurements.
- `optimise` Sweeps runtime configurations, including flags, thread counts, and cache quantisation, to identify the best-performing configuration for a given model and hardware platform.

## Setup Procedure

1. Clone the repository: 
~~~
git clone https://github.com/SaifRJ/ballast-arm64.git
~~~

2. Navigate into the cloned directory:
~~~
cd ballast-arm64
~~~

3. Run the `bootstrap.sh` script:
~~~
./bootstrap.sh
~~~

4. Activate the environment:
~~~
source .venv/bin/activate
~~~

5. Finally, edit `ballast.yaml` to define the benchmark run, including the engines, models, and measurement parameters. 
   A working default configuration is included.

6. Run the pipeline:
~~~
python3 -m ballast
~~~

> On the first run, Ballast installs or locates the LLM models specified, installs and builds the llama.cpp configurations entries, and fetches any evaluation content needed (e.g. a corpus text if perplexity is being measured) before benchmarking. Subsequent runs will skip straight to benchmarking/optimising.

All results and outputs are saved to `results/Benchmark_<timestamp>/` - one folder is generated per each run of the pipeline.

## Project scope

Ballast enables users to compare configurations and identify which optimisation strategies provide measurable benefits on their specific hardware. Reported performance differences reflect the combined effect of engine build, quantisation, and runtime configuration. The separation of these contributions is the user's job, and Ballast is designed to make that separation possible and easy.

## Known Limitations

- Power draw measurement is currently in the works.
- Models with exotic attention mechanisms (MLA, recurrent) report N/A for KV-cache metrics.
- Ballast cannot be integrated into CI pipelines yet, such as to detect performance regressions when changes are made to the inference stack. Support for this is planned and in the works.

## License

This project falls under MIT Licensing. You can fork and edit this code as much as you like. Please see LICENSE.md for more details.