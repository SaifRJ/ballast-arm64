from ballast.config import load_config, init_run, run_id, run_timestamp, prompts_dir
import ballast.benchmark as bm
import ballast.install as inst

# main.py: orchestration for the Arm64 LLM benchmarking pipeline
# Manages code orchestration and the core benchmarking loop
# All functions live in benchmark.py, main.py only decides what runs and in what order

start_time = bm.run_time()
config = load_config()
log = init_run()

# Run config
# todo: collapse run_settings and runtime_flags into one section
run = config["run_settings"]
runtime_flags = config["runtime_flags"]
CONTEXT_SIZE = run["context_size"]
GENERATED_TOKENS = run["generated_tokens"]
REPEATS = run["repeats"]
THREAD_SCALING = run["thread_scaling"]
PROMPTS = run["prompts"]
CACHE_TYPE_K = runtime_flags.get("cache_type_k")
CACHE_TYPE_V = runtime_flags.get("cache_type_v")

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

            # Returns a Llama object instance the caller owns for the lifetime of the model's benchmark run
            llm = bm.load_engine(engine_name, model["local_path"], CONTEXT_SIZE, thread_count, CACHE_TYPE_K, CACHE_TYPE_V)

            # Run a small inference to avoid first-call cost from affecting measurements
            bm.warmup_engine(llm) 

            # Retrieve a dict containing model metadata
            model_info = bm.get_model_info(llm)

            # Return KV-cache allocation per model
            kv_alloc = bm.compute_kv_alloc(model_info, CONTEXT_SIZE, CACHE_TYPE_K, CACHE_TYPE_V)

            # Append model info and architecture detail to model_info_{engine_name}.csv file output 
            bm.record_model_info(outputs["model_info"], engine_name, model, model_info, kv_alloc, run_id, run_timestamp)

            for corpus in corpora:

                # Measure perplexity per model per corpus
                perplexity = bm.measure_perplexity(model["local_path"], corpus["local_path"], corpus["chunks"], engine_name)

                # Append ppl values to perplexity_{engine_name}.csv file output
                bm.record_perplexity(outputs["perplexity"], engine_name, model, corpus, perplexity, CONTEXT_SIZE, run_id, run_timestamp)

            for prompt in PROMPTS:

                # Locate prompt file
                prompt_file = prompts_dir / f"{prompt}.txt"

                if not prompt_file.exists():
                    print(f"\n> ERROR: Unable to read prompt '{prompt}' (file missing: {prompt_file}). Prompt will be skipped.")
                    print("-> Please ensure the prompt file exists and matches the name of the main.py list")
                    continue

                # Read prompt file for tokenization
                prompt_text = bm.read_prompt_file(prompt_file)

                # Return prompt contents as tokens
                prompt_token_ids = bm.tokenize_prompt(llm, prompt_text)

                # Compute number of tokens
                n_prompt = len(prompt_token_ids)

                # Measure thread throughput per model per prompt
                scaling = bm.measure_thread_scaling(model["local_path"], n_prompt, thread_list, engine_name)

                # Append thread scaling values to CSV file
                bm.record_thread_scaling(outputs["threads"], engine_name, model, prompt, prompt_token_ids, scaling, run_id, run_timestamp)

                for repeat_number in range(1, REPEATS + 1):

                    print(f"\n> {model["name"]} / {prompt}: Repeat {repeat_number}/{REPEATS}")

                    # Returns Peak RAM and CPU% for this model/prompt/repeat combination
                    ram_cpu = bm.measure_ram_cpu(model["local_path"], prompt_file, CONTEXT_SIZE, GENERATED_TOKENS, thread_count, engine_name)

                    # Measure prefill/s
                    prefill_metrics = bm.measure_prefill(llm, prompt_token_ids)

                    # Measure token generation/s
                    generation_metrics = bm.measure_generation(llm, prompt_token_ids, GENERATED_TOKENS)

                    # Read how full the kv-cache is
                    kv_usage = bm.read_kv_usage(llm, kv_alloc, CONTEXT_SIZE)

                    # Append performance metric values to performance_{engine_name}.csv file output
                    bm.record_performance(outputs["performance"], engine_name, model, prompt, repeat_number, CONTEXT_SIZE, thread_count, 
                                          GENERATED_TOKENS, prefill_metrics, generation_metrics, ram_cpu, kv_usage, run_id, run_timestamp, 
                                          type_k=CACHE_TYPE_K or "f16", type_v=CACHE_TYPE_V or "f16")

            # Delete the llm object at the end of each model's loop to ensure a clean run per model
            del llm
                    
    # group by (config, model, prompt), mean/median across repeats
    # bm.compute_summary(run_timestamp)

    # charts / html from summary
    # bm.generate_report(run_timestamp)

    print(f"\n> PROCESS COMPLETE. \n> Results in results/Benchmark_{run_timestamp}/ ")
    print(f"\n> Run time: {str(bm.run_time() - start_time).split('.')[0]}")

if __name__ == "__main__":
    main()