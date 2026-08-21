#!/usr/bin/env bash
set -euo pipefail

exec 9>/tmp/polymath-deploy.lock
flock -n 9 || exit 0

REPOSITORY_DIR=/home/ubuntu/polymath
DEPLOY_STATE_DIR=/home/ubuntu/.local/state/polymath
DEPLOY_STATE_FILE=$DEPLOY_STATE_DIR/deployed-revision

mkdir -p $DEPLOY_STATE_DIR
cd $REPOSITORY_DIR

git fetch --quiet origin main
REMOTE_REVISION=$(git rev-parse origin/main)

if grep -qxF $REMOTE_REVISION $DEPLOY_STATE_FILE 2>/dev/null; then
  exit 0
fi

git merge --ff-only origin/main
sudo -n docker compose up -d --build --remove-orphans --wait --wait-timeout 180
curl --fail --silent --show-error --retry 12 --retry-delay 5 https://polymathmusician67.com/api/health

printf '%s\n' $REMOTE_REVISION > $DEPLOY_STATE_FILE
