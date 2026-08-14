#!/usr/bin/env bash
set -euo pipefail

LLAMA_TAG="b10327"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLAMA_DIR="${REPO_ROOT}/llama"

echo "=============================================="
echo " Ballast setup"
echo " llama.cpp pinned to ${LLAMA_TAG}"
echo "=============================================="

# System dependencies
echo ""
echo "> Installing system dependencies..."
sudo apt-get update -qq
sudo apt-get install -y \
    build-essential cmake git wget time \
    python3 python3-pip python3-dev python3-venv \
    libcurl4-openssl-dev

# Python version check
echo ""
echo "> Checking Python version..."
PY_OK=$(python3 -c 'import sys; print(1 if sys.version_info >= (3, 10) else 0)')
if [ "${PY_OK}" -ne 1 ]; then
    echo "> ERROR: Python 3.10+ required. Found: $(python3 --version)"
    exit 1
fi
echo "> Python $(python3 --version | awk '{print $2}') OK"

# Python virtual environment and dependencies
echo ""
echo "> Creating virtual environment..."
python3 -m venv "${REPO_ROOT}/.venv"
source "${REPO_ROOT}/.venv/bin/activate"

echo "> Installing Python dependencies..."
pip install --upgrade pip
pip install -r "${REPO_ROOT}/requirements.txt"

# Fetch llama.cpp source
echo ""
echo "> Fetching llama.cpp source @ ${LLAMA_TAG}..."
mkdir -p "${LLAMA_DIR}"
SRC="${LLAMA_DIR}/llama.cpp"
if [ ! -d "${SRC}" ]; then
    git clone https://github.com/ggml-org/llama.cpp.git "${SRC}"
fi
cd "${SRC}"
git fetch --tags --quiet
git checkout --quiet "${LLAMA_TAG}"

# Baseline build (generic ARMv8-A and Arm SIMD extensions disabled)
echo ""
echo "> Building BASELINE engine (generic ARMv8-A, no Arm SIMD)..."
cmake -B build-baseline -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF -DGGML_CPU_ARM_ARCH=armv8-a -DGGML_CPU_KLEIDIAI=OFF
cmake --build build-baseline --config Release -j"$(nproc)"

# Optimised build (native tuning and KleidiAI microkernels)
# GGML_NATIVE=ON auto-enables the host core's dotprod/i8mm/SVE
# GGML_CPU_KLEIDIAI=ON adds Arm's hand-optimised microkernels on top
echo ""
echo "> Building OPTIMISED engine (native + KleidiAI)..."
cmake -B build-optimised -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON -DGGML_CPU_KLEIDIAI=ON
cmake --build build-optimised --config Release -j"$(nproc)"

# Build llama instances under names the pipeline expects
echo ""
echo "> Linking build outputs..."
ln -sfn "${SRC}/build-baseline" "${LLAMA_DIR}/baseline_llama"
ln -sfn "${SRC}/build-optimised" "${LLAMA_DIR}/optimised_llama"

# Verify both builds produced the binaries the pipeline needs
echo ""
echo "> Verifying binaries..."
MISSING=0
for build in baseline_llama optimised_llama; do
    for bin in llama-cli llama-bench llama-perplexity; do
        if [ ! -f "${LLAMA_DIR}/${build}/bin/${bin}" ]; then
            echo "  ERROR: ${bin} missing from ${build}"
            MISSING=1
        fi
    done
done

if [ "${MISSING}" -ne 0 ]; then
echo ""
echo "> SETUP FAILED: one or more binaries did not build. Please see errors above."
exit 1
fi

echo ""
echo " Setup complete."
echo ""
echo " To run the pipeline:"
echo " -> source .venv/bin/activate"
echo " -> python3 main.py"
echo "=============================================="