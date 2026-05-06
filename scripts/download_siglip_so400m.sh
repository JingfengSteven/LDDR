#!/usr/bin/env bash
set -euo pipefail

MODEL_ID="${MODEL_ID:-google/siglip-so400m-patch14-384}"
HF_HOME_DEFAULT="/path/to/your/workspace/cache_center/huggingface"
HF_HOME="${HF_HOME:-$HF_HOME_DEFAULT}"
CACHE_DIR="${CACHE_DIR:-$HF_HOME}"
LOCAL_DIR="${LOCAL_DIR:-}"
PYTHON_BIN="${PYTHON_BIN:-python}"

echo "MODEL_ID=${MODEL_ID}"
echo "HF_HOME=${HF_HOME}"
echo "CACHE_DIR=${CACHE_DIR}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
  else
    echo "python/python3 not found"
    exit 1
  fi
fi

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "HF_TOKEN is not set; proceeding without token"
fi

export HF_HOME

"${PYTHON_BIN}" - <<'PY'
import os
from huggingface_hub import snapshot_download

model_id = os.environ.get("MODEL_ID", "google/siglip-so400m-patch14-384")
cache_dir = os.environ.get("CACHE_DIR")
local_dir = os.environ.get("LOCAL_DIR") or None
token = os.environ.get("HF_TOKEN") or None

kwargs = {
    "repo_id": model_id,
    "cache_dir": cache_dir,
    "token": token,
    "resume_download": True,
}

if local_dir:
    kwargs["local_dir"] = local_dir
    kwargs["local_dir_use_symlinks"] = False

path = snapshot_download(**kwargs)
print(f"downloaded_to={path}")
PY

cat <<EOF

Download finished.

Recommended runtime env:
  export HF_HOME="${HF_HOME}"

Then run lmms-eval as usual.
EOF
