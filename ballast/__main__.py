import os

from ballast.config import load_config, init_run, run_timestamp
import ballast.benchmark as bm
import ballast.install as inst
from ballast.runner import spawn_engine_subprocess

# This module controls top-level code orchestration for the entire pipeline. 
# No functions are defined here.

def main():

    start_time = bm.run_time()
    config = load_config()
    log = init_run()
    engines_yaml = config.engines
    models_yaml = config.models
    corpora_yaml = config.corpora

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

    # Validate corpus entry format in ballast.yaml
    inst.validate_corpus_entries(corpora_yaml)

    # Install valid corpus entries
    inst.install_corpora(corpora_yaml)

    # Copy generated engine manifests
    bm.snapshot_manifests(engines)

    # snapshot the host hardware details
    # hardware_info = bm.read_hardware_fingerprint()
    # bm.record_hardware_info(outputs["hardware_info"], hardware_info, run_id, run_timestamp)

    for engine in engines:

        spawn_engine_subprocess(engine)
                    
    # group by (config, model, prompt), mean/median across repeats
    # bm.compute_summary(run_timestamp)

    # charts / html from summary
    # bm.generate_report(run_timestamp)

    log.info(f"PROCESS COMPLETE.")
    log.info(f"Results in results/Benchmark_{run_timestamp}/")
    log.info(f"Run time: {str(bm.run_time() - start_time).split('.')[0]}")

if __name__ == "__main__":
    engine_name = os.environ.get("BALLAST_ENGINE_NAME")
    
    if engine_name:
        from ballast.runner import run_engine_benchmark
        run_engine_benchmark(engine_name)
    else:
        main()