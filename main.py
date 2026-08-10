import benchmark as bm

# main.py: orchestration for the Arm64 LLM benchmarking pipeline
# Manages code orchestration and the core benchmarking loop
# All functions live in benchmark.py, main.py only decides what runs and in what order

# Run config
PROMPTS = ["short", "medium", "long", "very_long"]
REPEATS = 1
CONTEXT_SIZE = 8192
GENERATED_TOKENS = 100
CONFIG_TAG = "baseline"
INSTANCE = "c7g.2xlarge"
START_TIME = bm.run_time()

# CSV output config setup
CSV_FIELDS = [
    "timestamp", "config", "instance", "model", "prompt-tier",
    "ctx", "threads", "gen_tokens", "repeat", "peak_ram_mb", 
    "cpu_pct", "prefill_tps", "prefill_tps_stddev", "gen_tps",
    "gen_tps_stddev", "ttft_ms"]
run_id = (bm.run_time()).strftime("%Y-%m-%d_%H-%M")

def main():

    # Read the list of model entries in models.txt 
    models = bm.load_models()

    # Download models or use existing models if specified 
    available_models = bm.download_models(models)
    
    # Detect available CPU cores for thread count measurements
    thread_count = bm.get_thread_count()

    # Create results.csv folder to store benchmark outputs
    results_csv = bm.ensure_results_csv(run_id, CSV_FIELDS)

    # Main loop returns performance metrics of every baseline/optimised model x every prompt x repeated N times to ensure statistical accuracy
    for model_name, model_source in available_models:

        for prompt in PROMPTS:

            # Locate prompt file
            prompt_file = bm.prompts_dir / f"{prompt}.txt"

            if not prompt_file.exists():
                print(f"\n> ERROR: Unable to read prompt '{prompt}' (file missing: {prompt_file}). Prompt will be skipped.")
                print("-> Please ensure the prompt file exists and matches the name of the main.py list")
                continue

            # Returns the number of tokens in the prompt file based on approx. 0.75 word/token ratio
            prompt_tokens = bm.count_tokens(prompt_file)
 
            for repeat_number in range(1, REPEATS + 1):

                print(f"\n> {model_name} / {prompt}: repeat {repeat_number}/{REPEATS}")
 
                # Returns Peak RAM and CPU% for this model/prompt/repeat combination
                peak_ram_mb, cpu_percent = bm.measure_ram_cpu(model_source, prompt_file, CONTEXT_SIZE, GENERATED_TOKENS, thread_count)

                # Returns prompt-processing speed and token generation speed
                metrics = bm.measure_bench_metrics(model_source, prompt_tokens, GENERATED_TOKENS, thread_count)
 
                # Assemble CSV row and append values
                bm.append_row(results_csv, CSV_FIELDS, {
                    "timestamp": (bm.run_time()).strftime("%Y-%m-%dT%H:%M:%S"),
                    "config": CONFIG_TAG,
                    "instance": INSTANCE,
                    "model": model_name,
                    "prompt-tier": prompt,
                    "ctx": CONTEXT_SIZE,
                    "threads": thread_count,
                    "gen_tokens": GENERATED_TOKENS,
                    "repeat": repeat_number,
                    "peak_ram_mb": peak_ram_mb if peak_ram_mb is not None else "NA",
                    "cpu_pct": cpu_percent if cpu_percent is not None else "NA",
                    "prefill_tps": metrics.get("prefill_tps", "NA"),
                    "prefill_tps_stddev": metrics.get("prefill_tps_stddev", "NA"),
                    "gen_tps": metrics.get("gen_tps", "NA"),
                    "gen_tps_stddev": metrics.get("gen_tps_stddev", "NA"),
                    "ttft_ms": metrics.get("ttft_ms", "NA")
                })

    # group by (config, model, prompt), mean/median across repeats
    # bm.compute_summary(run_id)

    # charts / html from summary
    # bm.generate_report(run_id)

    print(f"\n> PROCESS COMPLETE. \n> Results in results/benchmark_{run_id}/ ")
    print(f"\n> Run time: {str(bm.run_time() - START_TIME).split('.')[0]}")

if __name__ == "__main__":
    main()