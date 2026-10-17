import os
import sys
import subprocess
from ballast.schema import EngineConfig
from ballast.config import engines_dir, run_id, run_timestamp
from ballast.config import load_config, init_run, run_id, run_timestamp
from ballast.schema import Metric, SamplingMode
from ballast.sampler import ResourceSampler
import ballast.benchmark as bm
import ballast.install as inst

def spawn_engine_subprocess(engine: EngineConfig) -> None:

    # Spawn a fresh Python process to benchmark one engine.
    
    engine_bin = engines_dir / engine.name / "build" / "bin"
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = f"{engine_bin}:{env.get('LD_LIBRARY_PATH', '')}"
    env["LLAMA_CPP_LIB"] = str(engine_bin / "libllama.so")
    env["BALLAST_ENGINE_NAME"] = engine.name
    env["BALLAST_RUN_ID"] = run_id
    env["BALLAST_RUN_TIMESTAMP"] = run_timestamp
    subprocess.run([sys.executable, "-m", "ballast"], env=env, check=True)


def run_engine_benchmark(engine_name: str) -> None:

    # Entry point when running inside a per-engine subprocess

    config = load_config()
    log = init_run()

    engines = inst.get_available_engines(config.engines)
    engine = next(e for e in engines if e.name == engine_name)
    models = inst.get_available_models(config.models)
    corpora = inst.get_available_corpora(config.corpora)
    thread_list = bm.get_thread_list(config.pipeline.thread_scaling)
    pipeline_yaml = config.pipeline
    runtime_yaml = config.runtime
    metrics_yaml = config.metrics

    for model in models:

        # Returns a Llama object instance the caller owns for the lifetime of the model's benchmark run
        llm, model_load_time = bm.load_engine(model, runtime_yaml)

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
            bm.record_thread_scaling(engine, model, pipeline_yaml.thread_scaling_prompt_tokens, thread_scaling, run_id, run_timestamp)

        if Metric.PERPLEXITY in metrics_yaml.enabled:

            for corpus in corpora:

                # Measure perplexity per model per corpus
                perplexity = bm.measure_perplexity(model, corpus, engine.name)

                # Append ppl values to perplexity_{engine_name}.csv file output
                bm.record_perplexity(engine, model, corpus, perplexity, run_id, run_timestamp)

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

                sampler_csv_path = bm.ensure_csv(bm.SAMPLES_FIELDS, f"samples_{engine.name}.csv") if pipeline_yaml.mode is SamplingMode.SAMPLED else None
                with ResourceSampler(
                    interval_ms=pipeline_yaml.sample_interval_ms,
                    mode=pipeline_yaml.mode,
                    csv_path=sampler_csv_path,
                    tag=f"{model.name}/{prompt}/rep{repeat_number}",
                    run_id=run_id,
                    run_timestamp=run_timestamp,
                    engine_name=engine.name,
                    model_name=model.name,
                    prompt=prompt,
                    repeat=repeat_number,
                    phase="combined") as sampler:

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
                bm.record_performance(engine, model, runtime_yaml, prompt, repeat_number, prefill_metrics, generation_metrics, ram_cpu, kv_usage, model_load_time, prompt_token_ids, run_id, run_timestamp)

        # Append model info and architecture detail to model_info_{engine_name}.csv file output 
        bm.record_model_info(engine, model, model_info, kv_alloc, run_id, run_timestamp)

        # Delete the llm object at the end of each model's loop to ensure a clean run per model
        del llm