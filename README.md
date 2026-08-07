# Project Thesis

This is a solo-developer project which benchmarks the RAM utilisation of various LLMs on Arm64 Graviton, with the goal of evaluating and improving performance metrics using KV-cache quantisation and Arm-specific build flags in llama.cpp. 

This project aims to have three core delivarables:

1. A standardised benchmarking pipeline (benchmark.py) that tests, measures, and records LLM performance on Arm64 chips using custom and built-in llama.cpp functions. This tool is used to measure how effective the implemented optimisations are on each model.  
2. KV-Cache quantisation to halve the amount of RAM used whilst minimally affecting output quality.
3. Arm-specific build flags to optimise llama.cpp functions on Arm64 chips

All deliverables are designed as a 'harness' meant to be dropped directly into an instance with llama.cpp and executed for ease of use.

Development Environment Specs:
Developed and tested on AWS Graviton3 (c7g EC2 Instance), Arm64 Ubuntu 24.04, 8 vCPU Cores, 60GB of Storage, 16GB RAM


## Benchmarking Methodology

A select group of four LLMs were asked the same 5 prompts of varying length (prompts equivalent to 10, 200, 1000, 4000, 8000 tokens) to observe how fuller context windows affect specific performance metrics. 

The metrics measured and recorded were:
- Peak RAM utilisation (mean & median across repeats, from /usr/bin/time)
- RAM utilisation (mean & median across repeats, sampled RSS during the run)
- Prompt-processing speed (prefill, from llama-bench)
- Time-to-first-token (ms, from llama-bench)
- Token generation speed (tokens/sec, from llama-bench)
- Perplexity (one value per config, from llama-perplexity)
- Thread-scaling (how much work is done on varying thread counts)

LLM models tested:
- Gemma 3 4B (Google)
- Qwen3 4B (Alibaba)
- Llama 3.2 3B (Meta)
- Phi-4-mini 3.8B (Microsoft)

**Note: any LLMs that are llama.cpp compatible and in a GGUF file format can be used!**

Prompts used:
- Found in 'eval/prompts' folder, contains prompts ranging from just ~7 tokens to ~8000 tokens to observe how varying context window sizes affect specific metrics

## Optimisation Methodology

## Optimisation Achieved

## Setup & Reproducing Results

## Findings & Limitations

Findings:
- 

Limitations:
- Due to the nature of this project mostly being on a Cloud VM, measuring power-draw efficiency and optimisation was unfortunately not plausible.
- Optimising model size on disk was not accounted for, and is not the goal of this project.

## Additional Resources
