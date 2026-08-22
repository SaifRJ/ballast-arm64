from ballast.config import load_config, prompts_dir
import ballast.benchmark as bm
import ballast.install as inst
import uuid

# main.py: orchestration for the Arm64 LLM benchmarking pipeline
# Manages code orchestration and the core benchmarking loop
# All functions live in benchmark.py, main.py only decides what runs and in what order

start_time = bm.run_time()
run_timestamp = (bm.run_time()).strftime("%Y-%m-%d_%H-%M")
run_id = str(uuid.uuid4())
config = load_config()

# Run config
run = config["run_settings"]
CONTEXT_SIZE = run["context_size"]
GENERATED_TOKENS = run["generated_tokens"]
REPEATS = run["repeats"]
THREAD_SCALING = run["thread_scaling"]
PROMPTS = run["prompts"]

def main():

    # Validate engine entries in ballast.yaml
    inst.validate_engine_entries(config["engines"])

    # Install valid engine entries
    inst.install_engines(config["engines"])

    # Return list of successfully installed engines
    engines = inst.get_available_engines(config["engines"])

    # Validate model entry format in ballast.yaml
    inst.validate_model_entries(config["models"])

    # Install valid model entries
    inst.install_models(config["models"])

    # Return list of successfully installed models
    models = inst.get_available_models(config["models"])

    # Validate corpus entry format in ballast.yaml
    inst.validate_corpus_entries(config["corpora"])

    # Install valid corpus entries
    inst.install_corpora(config["corpora"])

    # Return list of successfully installed corpora
    corpora = inst.get_available_corpora(config["corpora"])

    # Detect available CPU cores
    thread_count = bm.get_thread_count()

    # Return thread sweep configuration from ballast.yaml
    thread_list = bm.get_thread_list(THREAD_SCALING)

    # Copy generated engine manifests
    bm.snapshot_manifests(engines, run_timestamp)

    for engine in engines:

        engine_name = engine["name"]

        # Create all output CSVs for this engine
        outputs = bm.create_run_outputs(run_timestamp, engine_name)

        for model in models:

            # Measure model perplexity per corpus provided
            for corpus in corpora:

                # Measure perplexity per model per corpus
                perplexity = bm.measure_perplexity(model["local_path"], corpus["local_path"], corpus["chunks"], engine_name)

                # Append ppl values to perplexity_{engine_name}.csv file output
                bm.record_perplexity(outputs["perplexity"], engine_name, model, corpus, perplexity, run_id)

            # Append model info and architecture detail to model_info_{engine_name}.csv file output 
            # fix todo: model metrics aren't defined until later in the loop, but model info only needs writing once per model 
            # bm.record_model_info(outputs["model_info"], engine_name, model, metrics, kv)

            for prompt in PROMPTS:

                # Locate prompt file
                prompt_file = prompts_dir / f"{prompt}.txt"

                if not prompt_file.exists():
                    print(f"\n> ERROR: Unable to read prompt '{prompt}' (file missing: {prompt_file}). Prompt will be skipped.")
                    print("-> Please ensure the prompt file exists and matches the name of the main.py list")
                    continue

                # Returns the number of tokens in the prompt file based on approx. 0.75 word/token ratio
                prompt_tokens = bm.count_tokens(prompt_file)

                if thread_list:
                    # Measure thread throughput per model per prompt
                    scaling = bm.measure_thread_scaling(model["local_path"], prompt_tokens, thread_list, engine_name)

                    # Append thread scaling values to CSV file
                    bm.record_thread_scaling(outputs["threads"], engine_name, model, prompt, prompt_tokens, scaling, run_id)

                for repeat_number in range(1, REPEATS + 1):

                    print(f"\n> {model["name"]} / {prompt}: Repeat {repeat_number}/{REPEATS}")

                    # Returns Peak RAM and CPU% for this model/prompt/repeat combination
                    ram_cpu = bm.measure_ram_cpu(model["local_path"], prompt_file, CONTEXT_SIZE, GENERATED_TOKENS, thread_count, engine_name)

                    # Returns prompt-processing speed and token generation speed
                    metrics = bm.measure_bench_metrics(model["local_path"], prompt_tokens, GENERATED_TOKENS, thread_count, engine_name)

                    kv = bm.compute_kv_cache(metrics, CONTEXT_SIZE, prompt_tokens, GENERATED_TOKENS)

                    # Append performance metric values to performance_{engine_name}.csv file output
                    bm.record_performance(outputs["performance"], engine_name, model, prompt, repeat_number, CONTEXT_SIZE, thread_count, GENERATED_TOKENS, metrics, ram_cpu, kv, run_id)
                    
    # group by (config, model, prompt), mean/median across repeats
    # bm.compute_summary(run_timestamp)

    # charts / html from summary
    # bm.generate_report(run_timestamp)

    print(f"\n> PROCESS COMPLETE. \n> Results in results/Benchmark_{run_timestamp}/ ")
    print(f"\n> Run time: {str(bm.run_time() - start_time).split('.')[0]}")

if __name__ == "__main__":
    main()