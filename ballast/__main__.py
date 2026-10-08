from ballast.config import load_config, init_run, run_id, run_timestamp
from ballast.schema import Metric
from ballast.sampler import ResourceSampler
import ballast.benchmark as bm
import ballast.install as inst

# This module controls top-level code orchestration for the entire pipeline. 
# No functions are defined here.

def main():

    start_time = bm.run_time()
    config = load_config()
    log = init_run()
    engines_yaml = config.engines
    models_yaml = config.models
    corpora_yaml = config.corpora
    pipeline_yaml = config.pipeline
    runtime_yaml = config.runtime
    metrics_yaml = config.metrics

    # Validate engine entries in ballast.yaml
    inst.validate_engine_entries(engines_yaml)

    # Install valid engine entries
    inst.install_engines(engines_yaml)

    # Return list of successfully installed engines
    engines = inst.get_available_engines(engines_yaml)

    # Validate model entry format in ballast.yaml
    inst.validate_model_entries(models_yaml)

    # Install valid model entries
    inst.install_models(models_yaml)

    # Return list of successfully installed models
    models = inst.get_available_models(models_yaml)

    # Validate corpus entry format in ballast.yaml
    inst.validate_corpus_entries(corpora_yaml)

    # Install valid corpus entries
    inst.install_corpora(corpora_yaml)

    # Return list of successfully installed corpora
    corpora = inst.get_available_corpora(corpora_yaml)

    # Return thread sweep configuration from ballast.yaml
    thread_list = bm.get_thread_list(pipeline_yaml.thread_scaling)

    # Copy generated engine manifests
    bm.snapshot_manifests(engines)

    # snapshot the host hardware details
    # hardware_info = bm.read_hardware_fingerprint()
    # bm.record_hardware_info(outputs["hardware_info"], hardware_info, run_id, run_timestamp)

    for engine in engines:

        # Create all output CSVs for this engine
        outputs = bm.create_run_outputs(engine.name, pipeline_yaml.mode)

        # engine router placeholder
        # bm.activate_engine(engine)

        for model in models:

            # Returns a Llama object instance the caller owns for the lifetime of the model's benchmark run
            llm = bm.load_engine(model, runtime_yaml)

            # Run a small inference to avoid first-call cost from affecting measurements
            bm.warmup_engine(llm) 

            # Retrieve a dict containing model metadata
            model_info = bm.get_model_info(llm)

            if Metric.KV_CACHE in metrics_yaml.enabled:

                # Return KV-cache allocation per model
                kv_alloc = bm.compute_kv_alloc(model_info, model)
            else: kv_alloc = None

            if Metric.THREAD_SCALING in metrics_yaml.enabled:
                
                # Measure thread throughput per model per prompt
                thread_scaling = bm.measure_thread_scaling(model, runtime_yaml, thread_list, pipeline_yaml.thread_scaling_prompt_tokens)

                # Append thread scaling values to CSV file
                bm.record_thread_scaling(outputs["threads"], engine, model, pipeline_yaml.thread_scaling_prompt_tokens, thread_scaling, run_id, run_timestamp)

            if Metric.PERPLEXITY in metrics_yaml.enabled:

                for corpus in corpora:

                    # Measure perplexity per model per corpus
                    perplexity = bm.measure_perplexity(model, corpus, engine.name)

                    # Append ppl values to perplexity_{engine_name}.csv file output
                    bm.record_perplexity(outputs["perplexity"], engine, model, corpus, perplexity, run_id, run_timestamp)

            for prompt in pipeline_yaml.prompts:

                # Read prompt file for tokenization
                prompt_text = bm.read_prompt_file(prompt)

                # Return prompt contents as tokens
                if not prompt_text:
                    continue

                prompt_token_ids = bm.tokenize_prompt(llm, prompt_text)

                for repeat_number in range(1, pipeline_yaml.repeats + 1):

                    prefill_metrics = None
                    generation_metrics = None
                    ram_cpu = None
                    kv_usage = None

                    log.info(f"{model.name} / {prompt}: Repeat {repeat_number}/{pipeline_yaml.repeats}")

                    with ResourceSampler(interval_ms=pipeline_yaml.sample_interval_ms, mode=pipeline_yaml.mode, csv_path=outputs.get("samples"), tag=f"{model.name}/{prompt}/rep{repeat_number}", run_id=run_id, run_timestamp=run_timestamp, engine_name=engine.name, model_name=model.name, prompt=prompt, repeat=repeat_number, phase="combined") as sampler:

                        if Metric.PREFILL in metrics_yaml.enabled:

                            # Measure prefill/s
                            prefill_metrics = bm.measure_prefill(llm, prompt_token_ids) 

                        if Metric.GENERATION in metrics_yaml.enabled:

                            # Measure token generation/s
                            generation_metrics = bm.measure_generation(llm, prompt_token_ids, model.generated_tokens)

                        # if Metric.MEMORY_BANDWIDTH in metrics_yaml.enabled:
                        #     mbu = bm.measure_mbu(llm, prompt_token_ids)

                        # if Metric.CPU_TEMP in metrics_yaml.enabled:
                        #     cpu_temp = bm.read_cpu_temp()

                        # if Metric.POWER_DRAW in metrics_yaml.enabled:
                        #     power = bm.measure_power(llm, prompt_token_ids, model.generated_tokens)

                    if Metric.RAM_CPU in metrics_yaml.enabled:

                        # Measure RAM/CPU usage via PID sampler
                        ram_cpu = sampler.aggregate()

                    if Metric.KV_CACHE in metrics_yaml.enabled:

                        # Read how full the KV cache is
                        kv_usage = bm.read_kv_usage(llm, kv_alloc, model)

                    # Append performance metric values to performance_{engine_name}.csv file output
                    bm.record_performance(outputs["performance"], engine, model, runtime_yaml, prompt, repeat_number, prefill_metrics, generation_metrics, ram_cpu, kv_usage, run_id, run_timestamp)

            # Append model info and architecture detail to model_info_{engine_name}.csv file output 
            bm.record_model_info(outputs["model_info"], engine, model, model_info, kv_alloc, run_id, run_timestamp)

            # Delete the llm object at the end of each model's loop to ensure a clean run per model
            del llm
                    
    # group by (config, model, prompt), mean/median across repeats
    # bm.compute_summary(run_timestamp)

    # charts / html from summary
    # bm.generate_report(run_timestamp)

    log.info(f"PROCESS COMPLETE.")
    log.info(f"Results in results/Benchmark_{run_timestamp}/")
    log.info(f"Run time: {str(bm.run_time() - start_time).split('.')[0]}")

if __name__ == "__main__":
    main()