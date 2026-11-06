import os
import subprocess
import sys
import platform
import logging

log = logging.getLogger("ballast")

IS_LINUX = sys.platform.startswith("linux")
IS_MACOS = sys.platform == "darwin"
# IS_ARM = platform.machine() in ("aarch64", "arm64")


def platform_name() -> str:
        
    if IS_LINUX: return "linux"
    if IS_MACOS: return "macos"
    
    return sys.platform


def require_supported_platform() -> None:

    if not (IS_LINUX or IS_MACOS):

        raise RuntimeError(
            f"\n> Unsupported OS platform: {sys.platform}"
            f"\n-> Ballast currently only supports Linux and macOS"
        )


def shared_oslib_extension() -> str:
    return ".dylib" if IS_MACOS else ".so"


def library_path_env_var() -> str:
    # On macOS, SIP strips DYLD_* variables for system binaries, so run Ballast from a venv or Homebrew Python
    return "DYLD_LIBRARY_PATH" if IS_MACOS else "LD_LIBRARY_PATH"


def libllama_name() -> str:
    return f"libllama{shared_oslib_extension()}"


def sysctl(name: str) -> str | None:
    
    # macOS exposes hardware details through sysctl rather than /proc and /sys
    if not IS_MACOS:
        return None
    try:
        result = subprocess.run(["sysctl", "-n", name], capture_output=True, text=True, check=True)
        return result.stdout.strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


def performance_core_count() -> int:
    
    if IS_MACOS:
        value = sysctl("hw.perflevel0.physicalcpu")

        if value:
            return int(value)
        
    return os.cpu_count() or 1


def mac_ggml_metal_flag(cmake_flags: dict) -> None:
    
    # llama.cpp enables GGML_METAL by default on macOS, which offloads to the GPU and disables CPU inference
    if IS_MACOS and str(cmake_flags.get("GGML_METAL", "ON")).upper() not in ("OFF", "0", "FALSE"):
        log.warning("detected macOS platform, GGML_METAL is enabled by default by llama.cpp. Set GGML_METAL=OFF in your ballast.yaml file to force CPU inference")