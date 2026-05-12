#!/bin/bash
# GCP VM setup script for the uncertainty benchmark.
# Run ONCE after first SSH into the uncertainty-benchmark VM.
# The DLVM image already has: Python 3.10, PyTorch 2.9, CUDA 12.9, nvidia-smi.
#
# Usage:
#   chmod +x setup_vm.sh && ./setup_vm.sh

set -e
echo "=== Uncertainty Benchmark VM Setup ==="

# ── 1. System packages ────────────────────────────────────────────────────
echo "[1/6] Installing system packages..."
sudo apt-get update -qq
sudo apt-get install -y git python3-pip python3-venv tmux htop nvtop

# ── 2. Clone repo ─────────────────────────────────────────────────────────
echo "[2/6] Cloning repository..."
REPO_DIR="$HOME/final_project"
if [ -d "$REPO_DIR" ]; then
    echo "  Repo already exists — pulling latest..."
    cd "$REPO_DIR" && git pull
else
    git clone https://github.com/dagmaros27/Llm-uncertainity-benchmark.git "$REPO_DIR"
    cd "$REPO_DIR"
    git checkout uncertainty-benchmark
fi
cd "$REPO_DIR"
echo "  Branch: $(git branch --show-current)"

# ── 3. Python environment ─────────────────────────────────────────────────
echo "[3/6] Setting up Python environment..."
python3 -m venv "$REPO_DIR/.venv"
source "$REPO_DIR/.venv/bin/activate"

pip install --upgrade pip -q
pip install -r uncertainty_benchmark/requirements.txt -q
echo "  Packages installed."

# ── 4. PYTHONPATH ─────────────────────────────────────────────────────────
echo "[4/6] Configuring PYTHONPATH..."
BASHRC="$HOME/.bashrc"
EXPORT_LINE="export PYTHONPATH=\"$REPO_DIR:\$PYTHONPATH\""
if ! grep -q "final_project" "$BASHRC"; then
    echo "" >> "$BASHRC"
    echo "# Uncertainty benchmark" >> "$BASHRC"
    echo "$EXPORT_LINE" >> "$BASHRC"
    echo "source $REPO_DIR/.venv/bin/activate" >> "$BASHRC"
fi
export PYTHONPATH="$REPO_DIR:$PYTHONPATH"

# ── 5. .env file ─────────────────────────────────────────────────────────
echo "[5/6] Checking .env file..."
ENV_PATH="$REPO_DIR/uncertainty_benchmark/.env"
if [ -f "$ENV_PATH" ]; then
    echo "  .env found at $ENV_PATH"
    # Quick key check (don't print values)
    for KEY in VERTEX_API_KEY HF_TOKEN WANDB_API_KEY; do
        if grep -q "^${KEY}=" "$ENV_PATH"; then
            echo "  [OK] $KEY is set"
        else
            echo "  [MISSING] $KEY — add it to $ENV_PATH"
        fi
    done
else
    echo "  .env NOT found — creating template at $ENV_PATH"
    cat > "$ENV_PATH" << 'EOF'
VERTEX_API_KEY=your_gemini_api_key_here
HF_TOKEN=your_huggingface_token_here
WANDB_API_KEY=your_wandb_api_key_here
EOF
    echo "  Fill in $ENV_PATH before running experiments."
fi

# ── 6. Smoke tests ────────────────────────────────────────────────────────
echo "[6/6] Running smoke tests..."

echo "  CUDA:"
python3 -c "import torch; print(f'    torch={torch.__version__} cuda={torch.cuda.is_available()} device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"none\"}')"

echo "  nvidia-smi:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null | sed 's/^/    /'

echo "  Parsing module:"
python3 -c "
import sys; sys.path.insert(0, '$REPO_DIR')
from uncertainty_benchmark.src.parsing import parse_with_schema
r = parse_with_schema('{\"ask_cq\": true, \"confidence\": 80}', ['ask_cq', 'confidence'])
assert r is not None
print('    OK')
"

echo "  WandB login (API key from .env):"
python3 -c "
import sys, os; sys.path.insert(0, '$REPO_DIR')
from uncertainty_benchmark.src.utils import load_dotenv
from pathlib import Path
load_dotenv(Path('$ENV_PATH'))
key = os.environ.get('WANDB_API_KEY', '')
if key and key != 'your_wandb_api_key_here':
    import wandb
    wandb.login(key=key, relogin=True)
    print('    WandB login OK')
else:
    print('    WANDB_API_KEY not set — skip')
" 2>&1 | tail -2

echo ""
echo "=== Setup complete ==="
echo "Next: source ~/.bashrc  (or open a new shell)"
echo "Then: python3 uncertainty_benchmark/scripts/smoke_test_providers.py"
