
from ballast.config import repo_root
from ballast.platform_adapter import IS_LINUX
import hashlib
import json
import os
import platform
import subprocess
import psutil
from pathlib import Path
import logging

log = logging.getLogger("ballast")

# MIDR_EL1 implementer and part numbers

# https://support.arm.com/documentation/ddi0406/b/System-Level-Architecture/Protected-Memory-System-Architecture--PMSA-/CP15-registers-for-a-PMSA-implementation/c0--Main-ID-Register--MIDR-?lang=en
CPU_IMPLEMENTERS = {
    0x41: "Arm", 
    0x61: "Apple", 
    0xC0: "Ampere",
    0x4E: "NVIDIA", 
    0x51: "Qualcomm", 
    0x56: "Marvell",
    0x42: "Broadcom", 
    0x43: "Cavium", 
    0x46: "Fujitsu"
}

# ReferenceS:
# https://kib.kiev.ua/x86docs/ARM/XML/2026-09/SysReg_xml_A_profile_2026-03_96-2026-03_rel/xhtml/AArch64-regindex.html
# https://support.arm.com/documentation/ddi0434/b/Debug/Management-registers/Processor-ID-Registers?lang=en
CPU_PARTS = {
    (0x41, 0xd08): "Cortex-A72", # Raspberry Pi 4
    (0x41, 0xd0b): "Cortex-A76", # Raspberry Pi 5
    (0x41, 0xd0c): "Neoverse N1", # Graviton2, Ampere Altra
    (0x41, 0xd40): "Neoverse V1", # Graviton3
    (0x41, 0xd49): "Neoverse N2",
    (0x41, 0xd4f): "Neoverse V2", # Graviton4, NVIDIA Grace
}

# Fields that identify the hardware. Software and host identity are excluded.
HARDWARE_ID_FIELDS = [
    "cpu_model", "cpu_midr_el1", "cpu_features", "cpu_sve_vector_length_bits",
    "cpu_cores_logical", "cpu_cores_physical", "l1d_cache_kb", "l1i_cache_kb",
    "l2_cache_kb", "l3_cache_kb", "total_ram_mb", "system_model", "numa_nodes",
]


def _read(path: str | Path) -> str | None:
    try:
        return Path(path).read_text().strip().strip("\x00")
    
    except OSError:
        return None


def _cpuinfo_first_block() -> dict:

    text = _read("/proc/cpuinfo") or ""
    info = {}

    for line in text.split("\n\n")[0].splitlines():

        if ":" in line:
            key, value = line.split(":", 1)
            info[key.strip()] = value.strip()

    return info


def _decode_midr(midr_hex: str | None) -> dict:

    if not midr_hex:
        return {}
    
    midr = int(midr_hex, 16)
    implementer = (midr >> 24) & 0xFF
    part = (midr >> 4) & 0xFFF

    return {
        "implementer": CPU_IMPLEMENTERS.get(implementer, hex(implementer)),
        "part": CPU_PARTS.get((implementer, part), hex(part)),
        "variant": (midr >> 20) & 0xF,
        "revision": midr & 0xF
    }


def _caches_kb() -> dict:

    caches = {
        "l1d_cache_kb": None, 
        "l1i_cache_kb": None, 
        "l2_cache_kb": None, 
        "l3_cache_kb": None
        }

    for index in sorted(Path("/sys/devices/system/cpu/cpu0/cache").glob("index*")):
        
        level, kind, size = _read(index / "level"), _read(index / "type"), _read(index / "size")
        
        if not (level and kind and size):
            continue

        kb = int(size.rstrip("K")) if size.endswith("K") else int(size.rstrip("M")) * 1024

        if level == "1":
            caches["l1d_cache_kb" if kind == "Data" else "l1i_cache_kb"] = kb

        elif level in ("2", "3"):
            caches[f"l{level}_cache_kb"] = kb

    return caches


def _system_model() -> str | None:
    return _read("/sys/devices/virtual/dmi/id/product_name") or _read("/proc/device-tree/model")


def _ballast_git_sha() -> str | None:
    try:
        result = subprocess.run(["git", "-C", str(repo_root), "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def collect_hardware_manifest() -> dict:

    if not IS_LINUX:
        raise NotImplementedError("Hardware fingerprinting is Linux-only for now (see platform_adapter).")

    cpuinfo = _cpuinfo_first_block()
    midr_hex = _read("/sys/devices/system/cpu/cpu0/regs/identification/midr_el1")
    midr = _decode_midr(midr_hex)
    sve_bytes = _read("/proc/sys/abi/sve_default_vector_length")

    try:
        import llama_cpp  # type: ignore
        llama_cpp_python_version = llama_cpp.__version__

    except (ImportError, AttributeError):
        llama_cpp_python_version = None

    hardware_manifest = {
        # Hardware
        "cpu_model": midr.get("part") or cpuinfo.get("model name"),
        "cpu_vendor": midr.get("implementer"),
        "cpu_arch": platform.machine(),
        "cpu_midr_el1": midr_hex,
        "cpu_variant": midr.get("variant"),
        "cpu_revision": midr.get("revision"),
        "cpu_features": sorted((cpuinfo.get("Features") or cpuinfo.get("flags") or "").split()),
        "cpu_sve_vector_length_bits": int(sve_bytes) * 8 if sve_bytes else None,
        "cpu_cores_logical": os.cpu_count(),
        "cpu_cores_physical": psutil.cpu_count(logical=False),
        **_caches_kb(),
        "total_ram_mb": round(psutil.virtual_memory().total / (1024**2)),
        "total_swap_mb": round(psutil.swap_memory().total / (1024**2)),
        "numa_nodes": len(list(Path("/sys/devices/system/node").glob("node[0-9]*"))) or None,
        "system_model": _system_model(),
        "cpu_governor": _read("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"),
        "cpu_max_freq_khz": _read("/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq"),

        # Software and host
        "hostname": platform.node(),
        "os_name": platform.freedesktop_os_release().get("PRETTY_NAME"),
        "kernel_version": platform.release(),
        "python_version": platform.python_version(),
        "llama_cpp_python_version": llama_cpp_python_version,
        "ballast_git_sha": _ballast_git_sha()
    }

    identity = json.dumps({k: hardware_manifest[k] for k in HARDWARE_ID_FIELDS}, sort_keys=True)
    hardware_manifest["hardware_id"] = hashlib.sha256(identity.encode()).hexdigest()[:12]

    return hardware_manifest


def write_hardware_manifest(path: Path) -> dict:
    
    hardware_manifest = collect_hardware_manifest()
    log.info(f"Hardware details: {hardware_manifest['cpu_model']} ({hardware_manifest['system_model']}), id {hardware_manifest['hardware_id']}")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(hardware_manifest, indent=2))

    return hardware_manifest