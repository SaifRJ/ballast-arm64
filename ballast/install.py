from ballast.config import engines_dir, models_dir, perplexity_dir, REQUIRED_BINARIES, run_time
from ballast.schema import EngineConfig, ModelConfig, CorpusConfig
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError
from pathlib import Path
import subprocess
import shutil
import zipfile
import json
import os
import logging

log = logging.getLogger("ballast")


def validate_engine_entries(engines: list[EngineConfig]) -> None:

    if not engines:
        raise ValueError(
            "\n> No engines defined in ballast.yaml."
            "\n-> Add at least one entry under 'engines:' before running install."
        )

    seen_names: set[str] = set()

    for index, engine in enumerate(engines, 1):
        name = engine.name.strip()

        if not name:
            raise ValueError(
                f"\n> Engine at position {index} has an empty name."
            )

        if name in seen_names:
            raise ValueError(
                f"\n> Duplicate engine name: '{name}'."
                f"\n-> Engine names must be unique within ballast.yaml."
            )
        seen_names.add(name)

        if "/" in name or "\\" in name or name.startswith("."):
            raise ValueError(
                f"\n> Engine name '{name}' contains invalid characters."
                f"\n-> Names must not contain '/', '\\', or start with '.'"
            )

        has_source = engine.source is not None or engine.tag is not None or bool(engine.cmake_flags)
        has_path = engine.path is not None

        if has_source and has_path:
            raise ValueError(
                f"\n> Engine '{name}' has both 'path' and 'source'/'tag'/'cmake_flags'."
                f"\n-> Choose one: 'path' for a pre-built engine, or 'source'+'tag' for Ballast to build."
            )

        if not has_source and not has_path:
            raise ValueError(
                f"\n> Engine '{name}' has no build instructions."
                f"\n-> Provide either 'source' + 'tag' or 'path'."
            )

        if has_source:
            if not engine.source:
                raise ValueError(f"\n> Engine '{name}' is missing 'source'.")
            if not engine.tag:
                raise ValueError(f"\n> Engine '{name}' is missing 'tag'.")

            _validate_engine_source(name, engine.source, engine.tag)

        if has_path:
            path_obj = Path(engine.path).expanduser().resolve()
            if not path_obj.exists():
                raise ValueError(
                    f"\n> Engine '{name}' path does not exist: {path_obj}"
                )
            if not path_obj.is_dir():
                raise ValueError(
                    f"\n> Engine '{name}' path is not a directory: {path_obj}"
                )

    log.info(f"{len(engines)} engine spec(s) validated.")


def _validate_engine_source(engine_name: str, source: str, tag: str) -> None:

    try:
        result = subprocess.run(
            ["git", "ls-remote", source],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )

    except subprocess.TimeoutExpired:
        raise ValueError(
            f"\n> Engine '{engine_name}': git remote timed out after 15s"
            f"\n-> Source: {source}"
            f"\n-> Check your network, or verify the URL is correct."
        )
    
    except FileNotFoundError:
        raise ValueError(
            f"\n> 'git' command not found on PATH."
            f"\n-> Install git before running Ballast install."
        )

    if result.returncode != 0:
        raise ValueError(
            f"\n> Engine '{engine_name}': cannot reach git source."
            f"\n-> Source: {source}"
            f"\n-> git error: {result.stderr.strip()}"
        )

    refs = result.stdout
    tag_patterns = [
        f"refs/tags/{tag}",
        f"refs/heads/{tag}",
    ]

    if any(pat in refs for pat in tag_patterns):
        return

    if len(tag) >= 7 and all(c in "0123456789abcdef" for c in tag.lower()):
        if any(line.startswith(tag.lower()) for line in refs.lower().splitlines()):
            return

    raise ValueError(
        f"\n> Engine '{engine_name}': tag '{tag}' not found in {source}"
        f"\n-> Verify the tag exists (check the repo's releases/tags page)."
    )

def get_available_engines(engines: list[EngineConfig]) -> list[EngineConfig]:

    available = []
    for engine in engines:
        bin_dir = engines_dir / engine.name / "build" / "bin"

        # engine is available only if all required binaries are present
        if all((bin_dir / b).exists() for b in REQUIRED_BINARIES):
            engine.bin_dir = bin_dir
            available.append(engine)

    return available


def install_engines(engines: list[EngineConfig]) -> None:

    engines_dir.mkdir(parents=True, exist_ok=True)
    log.info("Installing engines...")

    for engine in engines:

        if engine.path is not None:
            _link_prebuilt_engine(engine)
            continue

        if _needs_rebuild(engine):
            log.info(f"[{engine.name}] building...")
            build_engine(engine)
        else:
            log.info(f"[{engine.name}] already built, manifest matches, skipping.")


def build_engine(engine: EngineConfig) -> None:

    assert engine.source is not None, "build_engine requires source"
    assert engine.tag is not None, "build_engine requires tag"

    engine_dir = engines_dir / engine.name
    source_dir = engine_dir / "source"
    build_dir = engine_dir / "build"
    logs_dir = engine_dir / "logs"
    build_log = logs_dir / "build.log"

    engine_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    assert engine.source is not None, "build_engine called on path-based engine"

    if not source_dir.exists():
        _run_logged(["git", "clone", engine.source, str(source_dir)], build_log, f"clone {engine.source}")
    else:
        _run_logged(["git", "-C", str(source_dir), "fetch", "--tags", "--quiet"], build_log, f"fetch {engine.name}")

    assert engine.tag is not None
    _run_logged(["git", "-C", str(source_dir), "checkout", "--quiet", engine.tag], build_log, f"checkout {engine.tag}")

    result = subprocess.run(["git", "-C", str(source_dir), "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
    resolved_sha = result.stdout.strip()

    configure_cmd = ["cmake", "-B", str(build_dir), "-S", str(source_dir), "-DCMAKE_BUILD_TYPE=Release"]
    for flag_name, flag_value in engine.cmake_flags.items():
        configure_cmd.append(f"-D{flag_name}={flag_value}")

    _run_logged(configure_cmd, build_log, f"cmake configure {engine.name}")

    _run_logged(["cmake", "--build", str(build_dir), "--config", "Release", "-j", str(os.cpu_count() or 1)], build_log, f"cmake build {engine.name}")

    _write_manifest(engine_dir, engine, resolved_sha)
    log.info(f"[{engine.name}] built successfully.")


def _link_prebuilt_engine(engine: EngineConfig) -> None:

    assert engine.path is not None, "_link_prebuilt_engine requires engine.path"
    user_path = Path(engine.path).expanduser().resolve()
    engine_dir = engines_dir / engine.name
    build_dir = engine_dir / "build"

    engine_dir.mkdir(parents=True, exist_ok=True)

    # if build already links to the right place, nothing to do
    if build_dir.is_symlink() and build_dir.resolve() == user_path:
        log.info(f"[{engine.name}] already linked to {user_path}, skipping.")
        return

    # replace whatever's there
    if build_dir.exists() or build_dir.is_symlink():
        if build_dir.is_symlink():
            build_dir.unlink()
        else:
            shutil.rmtree(build_dir)

    build_dir.symlink_to(user_path)
    _write_manifest(engine_dir, engine, resolved_sha=None)
    log.info(f"[{engine.name}] linked to {user_path}")


def _needs_rebuild(engine: EngineConfig) -> bool:

    manifest_path = engines_dir / engine.name / "manifest.json"

    if not manifest_path.exists():
        return True

    try:
        with open(manifest_path) as f:
            manifest = json.load(f)
    except (json.JSONDecodeError, OSError):
        return True

    if manifest.get("source") != engine.source: return True
    if manifest.get("tag") != engine.tag: return True
    if manifest.get("cmake_flags", {}) != engine.cmake_flags: return True

    bin_dir = engines_dir / engine.name / "build" / "bin"
    if not all((bin_dir / b).exists() for b in REQUIRED_BINARIES):
        return True

    return False


def _write_manifest(engine_dir: Path, engine: EngineConfig, resolved_sha: str | None) -> None:
    manifest = {
        "name": engine.name,
        "source": engine.source,
        "tag": engine.tag,
        "resolved_git_sha": resolved_sha,
        "cmake_flags": engine.cmake_flags,
        "path": engine.path,
        "user_supplied": engine.path is not None,
        "build_date": run_time().isoformat(),
    }

    with open(engine_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)


def _run_logged(cmd: list[str], log_path: Path, description: str) -> None:

    with open(log_path, "a") as f:
        f.write(f"\n=== {description} ===\n")
        f.write(f"$ {' '.join(cmd)}\n\n")
        f.flush()

        result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=False)

    if result.returncode != 0:
        log.error(f"\nFAILED: {description}")

        with open(log_path) as f:
            lines = f.readlines()

        for line in lines[-40:]:
            log.error(f"  {line.rstrip()}")

        raise RuntimeError(f"> Build step failed: {description}")
    

def validate_model_entries(models: list[ModelConfig]) -> None:

    if not models:
        raise ValueError(
            "\n> No models defined in ballast.yaml."
            "\n-> Add at least one entry under 'models:' before running."
        )

    seen_names: set[str] = set()

    for index, model in enumerate(models, 1):
        name = model.name.strip()

        if not name:
            raise ValueError(f"\n> Model at position {index} has an empty name.")

        if name in seen_names:
            raise ValueError(
                f"\n> Duplicate model name: '{name}'."
                f"\n-> Model names must be unique within ballast.yaml."
            )
        
        seen_names.add(name)

        if "/" in name or "\\" in name or name.startswith("."):
            raise ValueError(
                f"\n> Model name '{name}' contains invalid characters."
                f"\n-> Names must not contain '/', '\\', or start with '.'"
            )

        source = model.source.strip()

        if source.lower().startswith("http"):
            _validate_model_url(name, source)
        else:
            _validate_model_local_path(name, source)

    log.info(f"{len(models)} model spec(s) validated.")


def _validate_model_url(model_name: str, source: str) -> None:
    
    # ensure GGUF URL is reachable before installation
    
    if not source.lower().endswith(".gguf"):
        raise ValueError(
            f"\n> Model '{model_name}': source URL does not end in '.gguf'"
            f"\n-> Source: {source}"
            f"\n-> The URL must point directly at a .gguf file (HuggingFace: use /resolve/ not /blob/)."
        )

    try:
        req = Request(source, method="HEAD")
        with urlopen(req, timeout=15) as response:
            if response.status >= 400:
                raise ValueError(
                    f"\n> Model '{model_name}': URL returned HTTP {response.status}"
                    f"\n-> Source: {source}"
                )
            
    except HTTPError as e:
        raise ValueError(
            f"\n> Model '{model_name}': URL not reachable (HTTP {e.code})"
            f"\n-> Source: {source}"
        )
    
    except URLError as e:
        raise ValueError(
            f"\n> Model '{model_name}': URL not reachable"
            f"\n-> Source: {source}"
            f"\n-> Error: {e.reason}"
        )


def _validate_model_local_path(model_name: str, source: str) -> None:

    # Validate local path exists and is a GGUF file
    path = Path(source).expanduser().resolve()

    if not path.exists():
        raise ValueError(
            f"\n> Model '{model_name}' points to a local file that does not exist: {path}"
            f"\n-> Verify the path in ballast.yaml, or provide a download URL."
        )
    
    if not path.is_file():
        raise ValueError(
            f"\n> Model '{model_name}' path is not a file: {path}"
            f"\n-> 'source' should be a .gguf file, not a directory."
        )
    
    if path.suffix.lower() != ".gguf":
        raise ValueError(
            f"\n> Model '{model_name}' local file is not a .gguf: {path}"
            f"\n-> Ballast can only benchmark GGUF models."
        )


def install_models(models: list[ModelConfig]) -> None:

    models_dir.mkdir(parents=True, exist_ok=True)
    log.info("Installing models...")

    for model in models:
        local_path = models_dir / f"{model.name}.gguf"

        if local_path.exists() or local_path.is_symlink():
            log.info(f"[{model.name}] already installed, skipping.")
            continue

        if not model.source.lower().startswith("http"):
            src_path = Path(model.source).expanduser().resolve()
            local_path.symlink_to(src_path)
            log.info(f"[{model.name}] symlinked from {src_path}")
            continue

        log.info(f"[{model.name}] downloading from {model.source}")
        command = ["wget", "-q", "--show-progress", "-O", str(local_path), model.source]

        try:
            subprocess.run(command, check=True)
            log.info(f"[{model.name}] installed to {local_path.name}")

        except subprocess.CalledProcessError:
            if local_path.exists():
                local_path.unlink()
            log.error(f"[{model.name}] FAILED to download from {model.source}")


def get_available_models(models: list[ModelConfig]) -> list[ModelConfig]:

    available = []
    for model in models:
        
        local_path = models_dir / f"{model.name}.gguf"

        if local_path.exists() or local_path.is_symlink():
            model.local_path = local_path
            available.append(model)

    return available


def validate_corpus_entries(corpora: list[CorpusConfig]) -> None:

    if not corpora:
        raise ValueError(
            "\n> No corpora defined in ballast.yaml."
            "\n-> Add at least one entry under 'corpora:' before running."
        )

    seen_names: set[str] = set()

    for index, corpus in enumerate(corpora, 1):
        name = corpus.name.strip()

        if not name:
            raise ValueError(
                f"\n> Corpus at position {index} has an empty name."
            )

        if name in seen_names:
            raise ValueError(
                f"\n> Duplicate corpus name: '{name}'."
                f"\n-> Corpus names must be unique within ballast.yaml."
            )
        seen_names.add(name)

        if "/" in name or "\\" in name or name.startswith("."):
            raise ValueError(
                f"\n> Corpus name '{name}' contains invalid characters."
                f"\n-> Names must not contain '/', '\\', or start with '.'"
            )

        source = corpus.source.strip()

        if corpus.chunks != "all" and (not isinstance(corpus.chunks, int) or corpus.chunks < 1):
            raise ValueError(
                f"\n> Corpus '{name}' has invalid 'chunks': {corpus.chunks!r}"
                f"\n-> 'chunks' must be a positive integer or 'all'."
            )

        if source.lower().startswith("http"):
            _validate_corpus_url(name, source)
        else:
            _validate_corpus_local_path(name, source)

    log.info(f"{len(corpora)} corpus spec(s) validated.")


def _validate_corpus_url(corpus_name: str, source: str) -> None:

    try:
        req = Request(source, method="HEAD")
        with urlopen(req, timeout=15) as response:
            if response.status >= 400:
                raise ValueError(
                    f"\n> Corpus '{corpus_name}': URL returned HTTP {response.status}"
                    f"\n-> Source: {source}"
                )
            
    except HTTPError as e:
        raise ValueError(
            f"\n> Corpus '{corpus_name}': URL not reachable (HTTP {e.code})"
            f"\n-> Source: {source}"
        )
    
    except URLError as e:
        raise ValueError(
            f"\n> Corpus '{corpus_name}': URL not reachable"
            f"\n-> Source: {source}"
            f"\n-> Error: {e.reason}"
        )


def _validate_corpus_local_path(corpus_name: str, source: str) -> None:

    path = Path(source).expanduser().resolve()

    if not path.exists():
        raise ValueError(
            f"\n> Corpus '{corpus_name}' points to a local file that does not exist: {path}"
        )
    
    if not path.is_file():
        raise ValueError(
            f"\n> Corpus '{corpus_name}' path is not a file: {path}"
        )


def install_corpora(corpora: list[CorpusConfig]) -> None:

    perplexity_dir.mkdir(parents=True, exist_ok=True)
    log.info("Installing corpora...")

    for corpus in corpora:
        local_path = perplexity_dir / f"{corpus.name}.txt"

        if local_path.exists() or local_path.is_symlink():
            log.info(f"[{corpus.name}] already installed, skipping.")
            continue

        # local file
        if not corpus.source.lower().startswith("http"):

            src_path = Path(corpus.source).expanduser().resolve()

            if src_path.suffix.lower() == ".zip":
                _extract_zip_to(src_path, local_path, corpus.name)
            else:
                local_path.symlink_to(src_path)
                log.info(f"[{corpus.name}] symlinked from {src_path}")
            continue

        # URL
        is_zip = corpus.source.lower().endswith(".zip")
        download_target = perplexity_dir / (f"{corpus.name}.zip" if is_zip else f"{corpus.name}.txt")

        log.info(f"[{corpus.name}] downloading from {corpus.source}")

        try:
            subprocess.run(["wget", "-q", "--show-progress", "-O", str(download_target), corpus.source],check=True)

        except subprocess.CalledProcessError:
            if download_target.exists():
                download_target.unlink()

            log.error(f"[{corpus.name}] FAILED to download from {corpus.source}")
            continue

        if is_zip:
            _extract_zip_to(download_target, local_path, corpus.name)
            download_target.unlink()

        else:
            log.info(f"[{corpus.name}] installed to {local_path.name}")


def _extract_zip_to(zip_path: Path, target_path: Path, corpus_name: str) -> None:

    try:
        with zipfile.ZipFile(zip_path) as zf:
            candidates = [n for n in zf.namelist() if n.endswith((".raw", ".txt"))]

            if not candidates:
                log.error(f"[{corpus_name}] FAILED: no .raw or .txt file found in zip")
                return
            
            member = candidates[0]

            with zf.open(member) as src, open(target_path, "wb") as dst:
                dst.write(src.read())

        log.info(f"[{corpus_name}] extracted {member} to {target_path.name}")

    except (zipfile.BadZipFile, OSError) as e:

        if target_path.exists():
            target_path.unlink()
            
        log.error(f"[{corpus_name}] FAILED to extract zip: {e}")


def get_available_corpora(corpora: list[CorpusConfig]) -> list[CorpusConfig]:

    available = []
    for corpus in corpora:

        local_path = perplexity_dir / f"{corpus.name}.txt"

        if local_path.exists() or local_path.is_symlink():
            corpus.local_path = local_path
            available.append(corpus)

    return available