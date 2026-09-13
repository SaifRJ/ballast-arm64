from ballast.config import load_config, init_run, run_id, run_timestamp
from ballast.schema import Metric
from ballast.sampler import ResourceSampler
import ballast.benchmark as bm
import ballast.install as inst

# This module controls top-level code orchestration for the entire pipeline the pipeline. 
# No functions are defined here.

start_time = bm.run_time()
config = load_config()
log = init_run()
engines_yaml = config.engines
models_yaml = config.models
corpora_yaml = config.corpora
pipeline_yaml = config.pipeline
runtime_yaml = config.runtime
metrics_yaml = config.metrics

def main():

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
    bm.snapshot_manifests(engines, run_timestamp)

    for engine in engines:

        # Create all output CSVs for this engine
        outputs = bm.create_run_outputs(run_timestamp, engine.name, pipeline_yaml.mode)

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

            # Append model info and architecture detail to model_info_{engine_name}.csv file output 
            bm.record_model_info(outputs["model_info"], engine.name, model, model_info, kv_alloc, run_id, run_timestamp)

            if Metric.THREAD_SCALING in metrics_yaml.enabled:
                # Measure thread throughput per model per prompt
                scaling = bm.measure_thread_scaling(model, runtime_yaml, thread_list, pipeline_yaml.thread_scaling_prompt_tokens)

                # Append thread scaling values to CSV file
                bm.record_thread_scaling(outputs["threads"], engine.name, model, pipeline_yaml.thread_scaling_prompt_tokens, scaling, run_id, run_timestamp)

            if Metric.PERPLEXITY in metrics_yaml.enabled:
                for corpus in corpora:
                    # Measure perplexity per model per corpus
                    perplexity = bm.measure_perplexity(model, corpus, engine.name)

                    # Append ppl values to perplexity_{engine_name}.csv file output
                    bm.record_perplexity(outputs["perplexity"], engine.name, model, corpus, perplexity, run_id, run_timestamp)

            for prompt in pipeline_yaml.prompts:

                # Read prompt file for tokenization
                prompt_text = bm.read_prompt_file(prompt)

                # Return prompt contents as tokens
                if prompt_text:
                    prompt_token_ids = bm.tokenize_prompt(llm, prompt_text)

                for repeat_number in range(1, pipeline_yaml.repeats + 1):

                    log.info(f"\n> {model.name} / {prompt}: Repeat {repeat_number}/{pipeline_yaml.repeats}")

                    with ResourceSampler(interval_ms=100, mode="snapshot") as sampler:

                        if Metric.PREFILL in metrics_yaml.enabled:
                            # Measure prefill/s
                            prefill_metrics = bm.measure_prefill(llm, prompt_token_ids) 

                        if Metric.GENERATION in metrics_yaml.enabled:
                            # Measure token generation/s
                            generation_metrics = bm.measure_generation(llm, prompt_token_ids, model.generated_tokens)

                    # Measure RAM/CPU usage via PID sampler
                    ram_cpu = sampler.aggregate()
                    
                    # Read how full the kv-cache is
                    kv_usage = bm.read_kv_usage(llm, kv_alloc, model.context_size) #ignore

                    # Append performance metric values to performance_{engine_name}.csv file output
                    bm.record_performance(outputs["performance"], engine.name, model, prompt, repeat_number, prefill_metrics, generation_metrics, ram_cpu, kv_usage, run_id, run_timestamp)

            # Delete the llm object at the end of each model's loop to ensure a clean run per model
            del llm
                    
    # group by (config, model, prompt), mean/median across repeats
    # bm.compute_summary(run_timestamp)

    # charts / html from summary
    # bm.generate_report(run_timestamp)

    log.info(f"\n> PROCESS COMPLETE. \n> Results in results/Benchmark_{run_timestamp}/ ")
    log.info(f"\n> Run time: {str(bm.run_time() - start_time).split('.')[0]}")

if __name__ == "__main__":
    main()