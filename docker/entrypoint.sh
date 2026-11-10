#!/bin/sh
# Optional checkpoint fetch, then the container command.
# MODEL_URL can be an HTTPS URL or a pre-signed S3 URL. No AWS credentials
# are baked into the image. A private object needs a URL that already grants
# access, or the file can be mounted at MODEL_PATH instead.

set -eu

dest="${MODEL_PATH:-/models/best.pt}"

if [ -n "${MODEL_URL:-}" ] && [ ! -f "$dest" ]; then
  mkdir -p "$(dirname "$dest")"
  echo "Downloading checkpoint to ${dest}"
  MODEL_PATH="$dest" python -c 'import os, urllib.request; urllib.request.urlretrieve(os.environ["MODEL_URL"], os.environ["MODEL_PATH"])'
fi

exec "$@"
