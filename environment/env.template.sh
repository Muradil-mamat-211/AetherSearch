#!/usr/bin/env bash
set -euo pipefail

# Set this to the root of your local Search-R1 workspace clone/cache.
export WORK=${WORK:-/path/to/search-r1-workspace}

export HF_HOME=${HF_HOME:-$WORK/hf_cache}
export HF_HUB_CACHE=${HF_HUB_CACHE:-$WORK/hf_cache/hub}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-$WORK/hf_cache/datasets}
export HF_XET_HIGH_PERFORMANCE=${HF_XET_HIGH_PERFORMANCE:-1}
export HF_HUB_ETAG_TIMEOUT=${HF_HUB_ETAG_TIMEOUT:-60}
export HF_HUB_DOWNLOAD_TIMEOUT=${HF_HUB_DOWNLOAD_TIMEOUT:-600}

# Required only when downloading/uploading private HuggingFace assets.
# export HF_TOKEN=hf_xxx
