import benchmark as bm

# main.py: orchestration for the Arm64 LLM benchmarking pipeline
# Manages code orchestration and the core benchmarking loop
# All functions live in benchmark.py, main.py only decides what runs and in what order

# Run config
PROMPTS = ["short", "medium", "long", "very_long", "extra_long"]
REPEATS = 10
CONTEXT_SIZE = 8192
GENERATED_TOKENS = 100
CONFIG_TAG = "baseline"
INSTANCE = "c7g.2xlarge"

def main():

    # Load the list of models we want to benchmark from models.txt
    models = bm.load_models()

    # Download the models from Hugging Face and cache them locally; returns a list of (model_name, model_repo) tuples for successful downloads
    print(f"\n> Retrieving available models from Hugging Face and caching them locally...")
    available_models = bm.download_models(models)
    
    # Detect available CPU cores for thread count measurements
    thread_count = bm.get_thread_count()

    # CSV output setup
    CSV_FIELDS = [
        "timestamp", "config", "instance", "model", "prompt-tier",
        "ctx", "threads", "gen_tokens", "repeat",
        "peak_ram_mb", "cpu_pct"]
    run_id = bm.new_run_id()
    results_csv = bm.ensure_results_csv(run_id, CSV_FIELDS)

    # Core loop: return performance metrics of every baseline/optimised model x every prompt length x repeated N times for statistical accuracy
    for model_name, model_repo in available_models:

        for prompt in PROMPTS:

            # Resolve the prompt file for this tier; skip cleanly if missing.
            prompt_file = bm.prompts_dir / f"{prompt}.txt"
            if not prompt_file.exists():
                print(f"\n> SKIP prompt '{prompt}' (missing {prompt_file})")
                continue
 
            for repeat_number in range(1, REPEATS + 1):
                print(f"\n> [{model_name} / {prompt}] repeat {repeat_number}/{REPEATS}")
 
                # RAM + CPU from one /usr/bin/time-wrapped llama-cli run
                peak_ram_mb, cpu_percent = bm.measure_ram_cpu(
                    model_repo,
                    prompt_file,
                    CONTEXT_SIZE,
                    GENERATED_TOKENS,
                    thread_count,
                )
 
                # assemble one row and append it
                bm.append_row(results_csv, CSV_FIELDS, {
                    "timestamp": bm.timestamp_now(),
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
                })

    # group by (config, model, prompt), mean/median across repeats
    # bm.compute_summary(run_id)

    # charts / html from summary
    # bm.generate_report(run_id)

    print(f"\n> PROCESS COMPLETE. \n> Results in results/benchmark_{run_id}/ ")

if __name__ == "__main__":
    main()