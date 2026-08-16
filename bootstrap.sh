#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=============================================="
echo " Ballast Bootstrap"
echo "=============================================="

# Detect OS and set package manager commands
echo ""
echo "> Detecting platform..."
case "$(uname -s)" in
    Linux*)
        OS="linux"
        UPDATE="sudo apt-get update -qq"
        INSTALL="sudo apt-get install -y"
        PACKAGES="build-essential cmake git wget time python3 python3-pip python3-dev python3-venv libcurl4-openssl-dev"
        ;;
    Darwin*)
        OS="macos"
        if ! command -v brew >/dev/null 2>&1; then
            echo "> ERROR: Homebrew not found. Install from https://brew.sh first."
            exit 1
        fi
        UPDATE="brew update"
        INSTALL="brew install"
        PACKAGES="cmake git wget python@3.11 curl"
        ;;
    *)
        echo "> ERROR: Unsupported OS: $(uname -s)"
        echo "-> Ballast currently supports Linux (Debian/Ubuntu) and macOS."
        exit 1
        ;;
esac
echo "> Platform: ${OS}"

# System dependencies
echo ""
echo "> Installing system dependencies..."
$UPDATE
$INSTALL $PACKAGES

# Python version check
echo ""
echo "> Checking Python version..."
PY_OK=$(python3 -c 'import sys; print(1 if sys.version_info >= (3, 10) else 0)')
if [ "${PY_OK}" -ne 1 ]; then
    echo "> ERROR: Python 3.10+ required. Found: $(python3 --version)"
    exit 1
fi
echo "> Python $(python3 --version | awk '{print $2}') OK"

# Virtual environment
echo ""
echo "> Creating virtual environment..."
python3 -m venv "${REPO_ROOT}/.venv"
source "${REPO_ROOT}/.venv/bin/activate"

# Python dependencies
echo ""
echo "> Installing Python dependencies..."
pip install --upgrade pip
pip install -r "${REPO_ROOT}/requirements.txt"

echo ""
echo " Bootstrap complete."
echo ""
echo " Next steps:"
echo " -> source .venv/bin/activate"
echo " -> python3 install.py       # build engines from ballast.yaml"
echo " -> python3 main.py          # run benchmarks"
echo "=============================================="