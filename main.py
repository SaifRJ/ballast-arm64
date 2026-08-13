import benchmark as bm

# main.py: orchestration for the Arm64 LLM benchmarking pipeline
# Manages code orchestration and the core benchmarking loop
# All functions live in benchmark.py, main.py only decides what runs and in what order

# Run config
START_TIME = bm.run_time()
PROMPTS = ["short", "medium", "long", "very_long"]
REPEATS = 1
CONTEXT_SIZE = 8192
GENERATED_TOKENS = 100
# CONFIG_TAG = ["baseline", "optimised"]
CONFIG_TAG = "baseline"
PERPLEXITY_CHUNKS = 40
THREAD_CAP = 8

# CSV output config setup
RESULTS_FIELDS = [
    "timestamp", "config", "model", "prompt-tier",
    "ctx", "threads", "gen_tokens", "repeat", "peak_ram_mb", "avg_ram_mb",
    "cpu_pct", "prefill_tps", "prefill_tps_stddev", "gen_tps",
    "gen_tps_stddev", "ttft_ms", "perplexity", "perplexity_chunks"]

THREAD_FIELDS = ["model", "config", "prompt-tier", 
                 "prompt_tokens", "threads", "prefill_tps"]

run_id = (bm.run_time()).strftime("%Y-%m-%d_%H-%M")

def main():

    # Read the list of model entries in models.txt 
    models = bm.load_models()

    # Download models or use existing models if specified 
    available_models = bm.download_models(models)

    # Check model availability 
    if not available_models:
        print("> No models available to benchmark. Exiting pipeline.")
        return

    # Check prompt availability
    if not [p for p in PROMPTS if (bm.prompts_dir / f"{p}.txt").exists()]:
        print("> No prompts found in eval/prompts. Exiting pipeline.")
        return

    # Check/install the perplexity corpus is available for perplexity measurements
    bm.install_perplexity_corpus() 

    # Detect available CPU cores
    thread_count = bm.get_thread_count()

    # Return a list of even numbers of threads capped at 8
    thread_pairs = bm.get_thread_list(THREAD_CAP)

    # Create results.csv folder to store baseline/optimised benchmark outputs
    results_csv = bm.ensure_csv(run_id, RESULTS_FIELDS, f"{CONFIG_TAG}_results.csv")

    # Create a thread_scaling.csv file to store baseline/optimised thread scaling measurements 
    thread_scaling_csv = bm.ensure_csv(run_id, THREAD_FIELDS, f"{CONFIG_TAG}_thread_scaling.csv")

    # Main loop returns performance metrics of every baseline/optimised model x every prompt
    for model_name, model_path in available_models:

        # Measure model perplexity
        perplexity = bm.measure_perplexity(model_path, PERPLEXITY_CHUNKS)

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
            scaling = bm.measure_thread_scaling(model_path, prompt_tokens, thread_pairs)
                
            for threads, tps in scaling:
                bm.append_row(thread_scaling_csv, THREAD_FIELDS, {
                "model": model_name, "config": CONFIG_TAG,
                "prompt-tier": prompt, "prompt_tokens": prompt_tokens,
                "threads": threads, "prefill_tps": tps})
 
            for repeat_number in range(1, REPEATS + 1):

                print(f"\n> {model_name} / {prompt}: repeat {repeat_number}/{REPEATS}")
 
                # Returns Peak RAM and CPU% for this model/prompt/repeat combination
                peak_ram_mb, cpu_percent, avg_ram_mb = bm.measure_ram_cpu(model_path, prompt_file, CONTEXT_SIZE, GENERATED_TOKENS, thread_count)

                # Returns prompt-processing speed and token generation speed
                metrics = bm.measure_bench_metrics(model_path, prompt_tokens, GENERATED_TOKENS, thread_count)
 
                # Assemble CSV row and append values
                bm.append_row(results_csv, RESULTS_FIELDS, {
                    "timestamp": (bm.run_time()).strftime("%Y-%m-%dT%H:%M:%S"),
                    "config": CONFIG_TAG,
                    "model": model_name,
                    "prompt-tier": prompt,
                    "ctx": CONTEXT_SIZE,
                    "threads": thread_count,
                    "gen_tokens": GENERATED_TOKENS,
                    "repeat": repeat_number,
                    "peak_ram_mb": peak_ram_mb if peak_ram_mb is not None else "NA",
                    "avg_ram_mb": avg_ram_mb if avg_ram_mb is not None else "NA",
                    "cpu_pct": cpu_percent if cpu_percent is not None else "NA",
                    "prefill_tps": metrics.get("prefill_tps", "NA"),
                    "prefill_tps_stddev": metrics.get("prefill_tps_stddev", "NA"),
                    "gen_tps": metrics.get("gen_tps", "NA"),
                    "gen_tps_stddev": metrics.get("gen_tps_stddev", "NA"),
                    "ttft_ms": metrics.get("ttft_ms", "NA"),
                    "perplexity": perplexity if perplexity is not None else "NA",
                    "perplexity_chunks": PERPLEXITY_CHUNKS
                })

    # group by (config, model, prompt), mean/median across repeats
    # bm.compute_summary(run_id)

    # charts / html from summary
    # bm.generate_report(run_id)

    print(f"\n> PROCESS COMPLETE. \n> Results in results/Benchmark_{run_id}/ ")
    print(f"\n> Run time: {str(bm.run_time() - START_TIME).split('.')[0]}")

if __name__ == "__main__":
    main()