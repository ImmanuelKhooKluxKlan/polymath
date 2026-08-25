#!/bin/sh
set -eu

if [ "$(id -u)" = "0" ]; then
  data_dir="${POLYMATH_DATA_DIR:-/tmp/polymath-data}"
  mkdir -p "$data_dir"
  chown -R node:node "$data_dir"
  exec gosu node "$@"
fi

exec "$@"
