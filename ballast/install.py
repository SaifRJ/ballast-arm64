from ballast.config import engines_dir, models_dir, perplexity_dir, REQUIRED_BINARIES, run_time
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

def validate_engine_entries(engines):

    # Validate every engine entry from ballast.yaml before install runs
    # Checks structural shape and that URLs / tags / paths resolve

    if not engines:
        raise ValueError(
            "\n> No engines defined in ballast.yaml."
            "\n-> Add at least one entry under 'engines:' before running install."
        )

    seen_names = set()

    for index, engine in enumerate(engines, 1):

        # verify name
        name = engine.get("name")
        if not name or not isinstance(name, str) or not name.strip():
            raise ValueError(
                f"\n> Engine at position {index} is missing 'name' or has an empty name."
                f"\n-> Every engine entry needs a non-empty 'name' field."
            )
        name = name.strip()

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

        # verify if source-based or path-based
        has_source = "source" in engine or "tag" in engine or "cmake_flags" in engine
        has_path = "path" in engine

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

        # if source-based, verify if URL is reachable and if tag exists in remote
        if has_source:
            source = engine.get("source")
            tag = engine.get("tag")

            if not source or not isinstance(source, str):
                raise ValueError(
                    f"\n> Engine '{name}' is missing 'source' or 'source' is not a string."
                    f"\n-> 'source' should be a git URL, e.g. https://github.com/ggml-org/llama.cpp"
                )
            if not tag or not isinstance(tag, str):
                raise ValueError(
                    f"\n> Engine '{name}' is missing 'tag' or 'tag' is not a string."
                    f"\n-> 'tag' should be a git tag or commit SHA, e.g. b10327"
                )

            cmake_flags = engine.get("cmake_flags", {})
            if not isinstance(cmake_flags, dict):
                raise ValueError(
                    f"\n> Engine '{name}' has 'cmake_flags' that is not a mapping."
                    f"\n-> Format: {{ GGML_NATIVE: ON, GGML_CPU_KLEIDIAI: ON }}"
                )

            _validate_engine_source(name, source, tag)

        # if path-based, directory must exist on disk right now
        if has_path:
            path = engine.get("path")
            if not path or not isinstance(path, str):
                raise ValueError(
                    f"\n> Engine '{name}' has 'path' that is missing or not a string."
                    f"\n-> 'path' should point to an existing llama.cpp build directory."
                )

            path_obj = Path(path).expanduser().resolve()
            if not path_obj.exists():
                raise ValueError(
                    f"\n> Engine '{name}' path does not exist: {path_obj}"
                    f"\n-> Verify the path in ballast.yaml points to a built llama.cpp directory."
                )
            if not path_obj.is_dir():
                raise ValueError(
                    f"\n> Engine '{name}' path is not a directory: {path_obj}"
                    f"\n-> 'path' should be a directory, not a file."
                )

    print(f"> {len(engines)} engine spec(s) validated.")
    return engines


def _validate_engine_source(engine_name, source, tag):

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

def get_available_engines(engines):

    available = []
    for engine in engines:
        name = engine["name"]
        bin_dir = engines_dir / name / "build" / "bin"

        # engine is available only if all required binaries are present
        if all((bin_dir / b).exists() for b in REQUIRED_BINARIES):
            engine["bin_dir"] = bin_dir
            available.append(engine)

    return available

def install_engines(engines):

    engines_dir.mkdir(parents=True, exist_ok=True)
    print("\n> Installing engines...")

    for engine in engines:
        name = engine["name"]

        if "path" in engine:
            _link_prebuilt_engine(engine)
            continue

        if _needs_rebuild(engine):
            print(f"-> [{name}] building...")
            build_engine(engine)
        else:
            print(f"-> [{name}] already built, manifest matches, skipping.")


def build_engine(engine):
    name = engine["name"]
    source = engine["source"]
    tag = engine["tag"]
    cmake_flags = engine.get("cmake_flags", {})

    engine_dir = engines_dir / name
    source_dir = engine_dir / "source"
    build_dir = engine_dir / "build"
    logs_dir = engine_dir / "logs"
    build_log = logs_dir / "build.log"

    engine_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    # clone if not present
    if not source_dir.exists():
        _run_logged(["git", "clone", source, str(source_dir)], build_log, f"clone {source}")
    else:
        _run_logged(["git", "-C", str(source_dir), "fetch", "--tags", "--quiet"], build_log, f"fetch {name}")

    _run_logged(["git", "-C", str(source_dir), "checkout", "--quiet", tag], build_log, f"checkout {tag}")

    # resolved SHA
    result = subprocess.run(
        ["git", "-C", str(source_dir), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True,
    )
    resolved_sha = result.stdout.strip()

    # cmake configure with user-specified flags
    configure_cmd = ["cmake", "-B", str(build_dir), "-S", str(source_dir), "-DCMAKE_BUILD_TYPE=Release"]

    for flag_name, flag_value in cmake_flags.items():
        configure_cmd.append(f"-D{flag_name}={flag_value}")

    _run_logged(configure_cmd, build_log, f"cmake configure {name}")

    # cmake build with all cores
    _run_logged(
        ["cmake", "--build", str(build_dir), "--config", "Release", "-j", str(os.cpu_count() or 1)],
        build_log, f"cmake build {name}",
    )

    _write_manifest(engine_dir, engine, resolved_sha)
    print(f"-> [{name}] built successfully.")


def _link_prebuilt_engine(engine):
    name = engine["name"]
    user_path = Path(engine["path"]).expanduser().resolve()
    engine_dir = engines_dir / name
    build_dir = engine_dir / "build"

    engine_dir.mkdir(parents=True, exist_ok=True)

    # if build already links to the right place, nothing to do
    if build_dir.is_symlink() and build_dir.resolve() == user_path:
        print(f"-> [{name}] already linked to {user_path}, skipping.")
        return

    # replace whatever's there
    if build_dir.exists() or build_dir.is_symlink():
        if build_dir.is_symlink():
            build_dir.unlink()
        else:
            shutil.rmtree(build_dir)

    build_dir.symlink_to(user_path)
    _write_manifest(engine_dir, engine, resolved_sha=None)
    print(f"-> [{name}] linked to {user_path}")


def _needs_rebuild(engine):
    name = engine["name"]
    manifest_path = engines_dir / name / "manifest.json"

    if not manifest_path.exists():
        return True

    try:
        with open(manifest_path) as f:
            manifest = json.load(f)
    except (json.JSONDecodeError, OSError):
        return True

    if manifest.get("source") != engine.get("source"): return True
    if manifest.get("tag") != engine.get("tag"): return True
    if manifest.get("cmake_flags", {}) != engine.get("cmake_flags", {}): return True

    bin_dir = engines_dir / name / "build" / "bin"
    if not all((bin_dir / b).exists() for b in REQUIRED_BINARIES):
        return True

    return False


def _write_manifest(engine_dir, engine, resolved_sha):
    manifest = {
        "name": engine["name"],
        "source": engine.get("source"),
        "tag": engine.get("tag"),
        "resolved_git_sha": resolved_sha,
        "cmake_flags": engine.get("cmake_flags", {}),
        "path": engine.get("path"),
        "user_supplied": "path" in engine,
        "build_date": run_time().isoformat(),
    }

    with open(engine_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)


def _run_logged(cmd, log_path, description):
    with open(log_path, "a") as log:
        log.write(f"\n=== {description} ===\n")
        log.write(f"$ {' '.join(cmd)}\n\n")
        log.flush()

        result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=False)

    if result.returncode != 0:
        # dump tail of log so user sees why without opening the file
        print(f"\n> FAILED: {description}")
        print(f"> Last 40 lines of {log_path}:")
        with open(log_path) as log:
            lines = log.readlines()
        for line in lines[-40:]:
            print(f"  {line.rstrip()}")
        raise RuntimeError(f"Build step failed: {description}")
    

def validate_model_entries(models):

    # Validate each model to have:
    # 'name' (unique, filesystem-safe)
    # 'source' (URL ending in .gguf, or a local .gguf filepath)

    if not models:
        raise ValueError(
            "\n> No models defined in ballast.yaml."
            "\n-> Add at least one entry under 'models:' before running."
        )

    seen_names = set()

    for index, model in enumerate(models, 1):

        name = model.get("name")
        if not name or not isinstance(name, str) or not name.strip():
            raise ValueError(
                f"\n> Model at position {index} is missing 'name' or has an empty name."
                f"\n-> Every model entry needs a non-empty 'name' field."
            )
        name = name.strip()

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

        source = model.get("source")
        if not source or not isinstance(source, str):
            raise ValueError(
                f"\n> Model '{name}' is missing 'source' or 'source' is not a string."
                f"\n-> 'source' should be a .gguf URL, or a local filepath to a .gguf file."
            )
        source = source.strip()

        if source.lower().startswith("http"):
            _validate_model_url(name, source)
        else:
            _validate_model_local_path(name, source)

    print(f"> {len(models)} model spec(s) validated.")
    return models


def _validate_model_url(model_name, source):
    
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


def _validate_model_local_path(model_name, source):

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
            f"\n-> Ballast benchmarks GGUF models specifically."
        )


def install_models(models):

    models_dir.mkdir(parents=True, exist_ok=True)
    print("\n> Installing models...")

    for model in models:
        name = model["name"]
        source = model["source"]
        local_path = models_dir / f"{name}.gguf"

        if local_path.exists() or local_path.is_symlink():
            print(f"-> [{name}] already installed, skipping.")
            continue

        if not source.lower().startswith("http"):
            src_path = Path(source).expanduser().resolve()
            local_path.symlink_to(src_path)
            print(f"-> [{name}] symlinked from {src_path}")
            continue

        print(f"-> [{name}] downloading from {source}")
        command = ["wget", "-q", "--show-progress", "-O", str(local_path), source]

        try:
            subprocess.run(command, check=True)
            print(f"-> [{name}] installed to {local_path.name}")

        except subprocess.CalledProcessError:

            if local_path.exists():
                local_path.unlink()
            print(f"-> [{name}] FAILED to download from {source}")


def get_available_models(models):

    available = []
    for model in models:
        name = model["name"]
        local_path = models_dir / f"{name}.gguf"

        if local_path.exists() or local_path.is_symlink():
            model["local_path"] = local_path
            available.append(model)

    return available


def validate_corpus_entries(corpora):

    if not corpora:
        raise ValueError(
            "\n> No corpora defined in ballast.yaml."
            "\n-> Add at least one entry under 'corpora:' before running."
        )

    seen_names = set()

    for index, corpus in enumerate(corpora, 1):

        name = corpus.get("name")
        if not name or not isinstance(name, str) or not name.strip():
            raise ValueError(
                f"\n> Corpus at position {index} is missing 'name' or has an empty name."
                f"\n-> Every corpus entry needs a non-empty 'name' field."
            )
        name = name.strip()

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

        source = corpus.get("source")
        if not source or not isinstance(source, str):
            raise ValueError(
                f"\n> Corpus '{name}' is missing 'source' or 'source' is not a string."
                f"\n-> 'source' should be a URL or local filepath (.txt, .raw, or .zip)."
            )
        source = source.strip()

        chunks = corpus.get("chunks")
        if chunks != "all" and (not isinstance(chunks, int) or chunks < 1):
            raise ValueError(
                f"\n> Corpus '{name}' has invalid 'chunks': {chunks!r}"
                f"\n-> 'chunks' must be a positive integer or 'all'."
            )

        if source.lower().startswith("http"):
            _validate_corpus_url(name, source)
        else:
            _validate_corpus_local_path(name, source)

    print(f"> {len(corpora)} corpus spec(s) validated.")
    return corpora


def _validate_corpus_url(corpus_name, source):

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


def _validate_corpus_local_path(corpus_name, source):

    path = Path(source).expanduser().resolve()

    if not path.exists():
        raise ValueError(
            f"\n> Corpus '{corpus_name}' points to a local file that does not exist: {path}"
        )
    if not path.is_file():
        raise ValueError(
            f"\n> Corpus '{corpus_name}' path is not a file: {path}"
        )


def install_corpora(corpora):

    perplexity_dir.mkdir(parents=True, exist_ok=True)
    print("\n> Installing corpora...")

    for corpus in corpora:
        name = corpus["name"]
        source = corpus["source"]
        local_path = perplexity_dir / f"{name}.txt"

        if local_path.exists() or local_path.is_symlink():
            print(f"-> [{name}] already installed, skipping.")
            continue

        # local file
        if not source.lower().startswith("http"):
            src_path = Path(source).expanduser().resolve()
            if src_path.suffix.lower() == ".zip":
                _extract_zip_to(src_path, local_path, name)
            else:
                local_path.symlink_to(src_path)
                print(f"-> [{name}] symlinked from {src_path}")
            continue

        # URL
        is_zip = source.lower().endswith(".zip")
        download_target = perplexity_dir / (f"{name}.zip" if is_zip else f"{name}.txt")

        print(f"-> [{name}] downloading from {source}")
        try:
            subprocess.run(
                ["wget", "-q", "--show-progress", "-O", str(download_target), source],
                check=True,
            )
        except subprocess.CalledProcessError:
            if download_target.exists():
                download_target.unlink()
            print(f"-> [{name}] FAILED to download from {source}")
            continue

        if is_zip:
            _extract_zip_to(download_target, local_path, name)
            download_target.unlink()
        else:
            print(f"-> [{name}] installed to {local_path.name}")


def _extract_zip_to(zip_path, target_path, corpus_name):

    try:
        with zipfile.ZipFile(zip_path) as zf:
            candidates = [n for n in zf.namelist() if n.endswith((".raw", ".txt"))]
            if not candidates:
                print(f"-> [{corpus_name}] FAILED: no .raw or .txt file found in zip")
                return
            member = candidates[0]
            with zf.open(member) as src, open(target_path, "wb") as dst:
                dst.write(src.read())
        print(f"-> [{corpus_name}] extracted {member} to {target_path.name}")
    except (zipfile.BadZipFile, OSError) as e:
        if target_path.exists():
            target_path.unlink()
        print(f"-> [{corpus_name}] FAILED to extract zip: {e}")


def get_available_corpora(corpora):

    available = []
    for corpus in corpora:
        name = corpus["name"]
        local_path = perplexity_dir / f"{name}.txt"

        if local_path.exists() or local_path.is_symlink():
            corpus["local_path"] = local_path
            available.append(corpus)

    return available