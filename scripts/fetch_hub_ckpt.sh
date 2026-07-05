#!/bin/bash
# Download a Hub checkpoint into node-local storage, not the home HF cache.
set -euo pipefail

if [ "${1:-}" = "--help" ] || [ "${1:-}" = "-h" ]; then
  cat <<'EOF'
Usage: scripts/fetch_hub_ckpt.sh <hub_repo_id> <dest_dir>

Downloads a model snapshot with Hugging Face Hub local_dir semantics and prints
the resolved local path. Use a node-local dest_dir such as ${ALAB_NODE_TMP}/sft_init.
EOF
  exit 0
fi

if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <hub_repo_id> <dest_dir>" >&2
  exit 2
fi

HUB_REPO_ID="$1"
DEST_DIR="$2"
mkdir -p "${DEST_DIR}"

python3 - "$HUB_REPO_ID" "$DEST_DIR" <<'PY'
import os
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

repo_id = sys.argv[1]
dest = Path(sys.argv[2]).resolve()
dest.mkdir(parents=True, exist_ok=True)

token = os.environ.get("HF_TOKEN")
if token is None:
    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    token_path = hf_home / "token"
    if token_path.exists():
        token = token_path.read_text(encoding="utf-8").strip()

path = snapshot_download(
    repo_id=repo_id,
    repo_type="model",
    local_dir=str(dest),
    cache_dir=str(dest / ".hf_cache"),
    token=token,
)
print(str(Path(path).resolve()), flush=True)
PY
