import benchmark as bm

# main.py: orchestration for the Arm64 LLM benchmarking pipeline
# Manages code orchestration and the core benchmarking loop
# All functions live in benchmark.py, main.py only decides what runs and in what order

START_TIME = bm.run_time()
config = bm.load_config()

# Run config
run = config["run_settings"]
CONTEXT_SIZE = run["context_size"]
GENERATED_TOKENS = run["generated_tokens"]
REPEATS = run["repeats"]
THREAD_CAP = run["thread_cap"]
PERPLEXITY_CHUNKS = run["perplexity_chunks"]
PROMPTS = run["prompts"]
models = [(m["name"], m["source"]) for m in config["models"]]

# CSV output config setup
RESULTS_FIELDS = [
    # Run / workload
    "timestamp",
    "engine",
    "model",
    "prompt",
    "ctx",
    "threads",
    "repeat",

    # Model / configuration
    "model_size_bytes",
    "model_n_params",
    "n_layer",
    "n_head_kv",
    "type_k",
    "type_v",
    "key_length",
    "value_length",

    # Prefill performance
    "prefill_tps",
    "prefill_ms",
    "prefill_tps_stddev",

    # Generation performance
    "gen_tokens",
    "gen_tps",
    "gen_tps_stddev",
    "ttft_ms",
    "perplexity",
    "perplexity_chunks",

    # CPU/RAM
    "cpu_pct",
    "avg_ram_mb",
    "peak_ram_mb",

    # KV cache
    "kv_alloc_mb",
    "kv_used_mb",
    "kv_utilisation",
]

THREAD_FIELDS = ["model", "engine", "prompt", 
                 "prompt_tokens", "threads", "prefill_tps"]

run_id = (bm.run_time()).strftime("%Y-%m-%d_%H-%M")

def main():

    # Download models or use existing models if specified 
    available_models = bm.download_models(models)

    # Check model availability 
    if not available_models:
        print("> No models available to benchmark. Exiting pipeline.")
        return

    # Check/install the perplexity corpus is available for perplexity measurements
    bm.install_perplexity_corpus() 

    # Detect available CPU cores
    thread_count = bm.get_thread_count()

    thread_list = bm.get_thread_list(THREAD_CAP)

    for engine in config["engines"]:

        engine_name = engine["name"]

        # Prepare CSV output
        results_csv = bm.ensure_csv(run_id, RESULTS_FIELDS, f"{engine_name}_results.csv")
        thread_scaling_csv = bm.ensure_csv(run_id, THREAD_FIELDS, f"{engine_name}_thread_scaling.csv")

        for model_name, model_path in available_models:

            # Measure model perplexity
            perplexity = bm.measure_perplexity(model_path, PERPLEXITY_CHUNKS, engine_name)

            for prompt in PROMPTS:

                # Locate prompt file
                prompt_file = bm.prompts_dir / f"{prompt}.txt"

                if not prompt_file.exists():
                    print(f"\n> ERROR: Unable to read prompt '{prompt}' (file missing: {prompt_file}). Prompt will be skipped.")
                    print("-> Please ensure the prompt file exists and matches the name of the main.py list")
                    continue

                # Returns the number of tokens in the prompt file based on approx. 0.75 word/token ratio
                prompt_tokens = bm.count_tokens(prompt_file)

                # Measure thread throughput per model per prompt
                scaling = bm.measure_thread_scaling(model_path, prompt_tokens, thread_list, engine_name)
                    
                for threads, tps in scaling:
                    bm.append_row(thread_scaling_csv, THREAD_FIELDS, {
                    "engine": engine_name,
                    "model": model_name,
                    "prompt": prompt, 
                    "prompt_tokens": prompt_tokens,
                    "threads": threads, 
                    "prefill_tps": tps})

                for repeat_number in range(1, REPEATS + 1):

                    print(f"\n> {model_name} / {prompt}: Repeat {repeat_number}/{REPEATS}")

                    # Returns Peak RAM and CPU% for this model/prompt/repeat combination
                    peak_ram_mb, cpu_percent, avg_ram_mb = bm.measure_ram_cpu(model_path, prompt_file, CONTEXT_SIZE, GENERATED_TOKENS, thread_count, engine_name)

                    # Returns prompt-processing speed and token generation speed
                    metrics = bm.measure_bench_metrics(model_path, prompt_tokens, GENERATED_TOKENS, thread_count, engine_name)

                    kv = bm.compute_kv_cache(metrics, CONTEXT_SIZE, prompt_tokens, GENERATED_TOKENS)

                    # Assemble CSV row and append values
                    bm.append_row(results_csv, RESULTS_FIELDS, {
                        "timestamp": (bm.run_time()).strftime("%Y-%m-%dT%H:%M:%S"),
                        "engine": engine_name,
                        "model": model_name,
                        "prompt": prompt,
                        "ctx": CONTEXT_SIZE,
                        "threads": thread_count,
                        "gen_tokens": GENERATED_TOKENS,
                        "repeat": repeat_number,
                        "peak_ram_mb": peak_ram_mb if peak_ram_mb is not None else "NA",
                        "avg_ram_mb": avg_ram_mb if avg_ram_mb is not None else "NA",
                        "cpu_pct": cpu_percent if cpu_percent is not None else "NA",
                        "prefill_tps": metrics.get("prefill_tps", "NA"),
                        "prefill_ms": metrics.get("prefill_ms", "NA"),
                        "prefill_tps_stddev": metrics.get("prefill_tps_stddev", "NA"),
                        "gen_tps": metrics.get("gen_tps", "NA"),
                        "gen_tps_stddev": metrics.get("gen_tps_stddev", "NA"),
                        "ttft_ms": metrics.get("ttft_ms", "NA"),
                        "perplexity": perplexity if perplexity is not None else "NA",
                        "perplexity_chunks": PERPLEXITY_CHUNKS,
                        "model_size_bytes": metrics.get("model_size_bytes", "NA"),
                        "model_n_params": metrics.get("model_n_params", "NA"),
                        "type_k": metrics.get("type_k", "NA"),
                        "type_v": metrics.get("type_v", "NA"),
                        "n_layer": metrics.get("n_layer", "NA"),
                        "n_head_kv": metrics.get("n_head_kv", "NA"),
                        "key_length": metrics.get("key_length", "NA"),
                        "value_length": metrics.get("value_length", "NA"),
                        "kv_alloc_mb": kv.get("kv_alloc_mb", "NA"),
                        "kv_used_mb": kv.get("kv_used_mb", "NA"),
                        "kv_utilisation": kv.get("kv_utilisation", "NA"),
                    })

    # group by (config, model, prompt), mean/median across repeats
    # bm.compute_summary(run_id)

    # charts / html from summary
    # bm.generate_report(run_id)

    print(f"\n> PROCESS COMPLETE. \n> Results in results/Benchmark_{run_id}/ ")
    print(f"\n> Run time: {str(bm.run_time() - START_TIME).split('.')[0]}")

if __name__ == "__main__":
    main()