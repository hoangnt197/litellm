#!/bin/sh
set -eu

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
repo_dir=$(CDPATH= cd -- "$script_dir/.." && pwd)
revision=${LITELLM_REVISION:-$(git -C "$repo_dir" rev-parse HEAD)}
repository=${LITELLM_IMAGE_REPOSITORY:-ghcr.io/hoangnt197/litellm}
image="$repository:sha-$revision"

docker build \
  --target runtime \
  --build-arg "VCS_REF=$revision" \
  --build-arg "SOURCE_REPOSITORY=https://github.com/hoangnt197/litellm" \
  --tag "$image" \
  "$repo_dir"

docker run --rm --network none --entrypoint python \
  --volume "$repo_dir/deployments/production/test_reasoning_policy.py:/app/test_reasoning_policy.py:ro" \
  "$image" /app/test_reasoning_policy.py

printf '%s\n' "$image"
