#!/usr/bin/env bash

# Scripts/configs/manifests stay in the repository. High-volume raw telemetry
# and logs default to the user-requested runtime tmpfs and are referenced by
# symlinks from each repository run directory.
RIVF26_BULK_ROOT=${RIVF26_BULK_ROOT:-/run/user/1009/ducct/rivf26}
export RIVF26_BULK_ROOT
