#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Public images only. An explicit empty Hub entry prevents Docker from
# auto-selecting osxkeychain, which may block unattended builds on macOS.
# Respect an explicitly supplied configuration for authenticated registries.
if [[ -z "${DOCKER_CONFIG:-}" ]]; then
  export DOCKER_CONFIG="$root/tmp/docker-cli"
  mkdir -p "$DOCKER_CONFIG"
  python3 - "$DOCKER_CONFIG" <<'PY'
import json
from pathlib import Path
import sys
directory = Path(sys.argv[1])
config = directory/'config.json'
if not config.exists():
    config.write_text(json.dumps({'auths': {'https://index.docker.io/v1/': {}},
        'cliPluginsExtraDirs': [str(Path.home()/'.docker/cli-plugins')]}))
contexts = directory/'contexts'
source = Path.home()/'.docker/contexts'
if source.exists() and not contexts.exists():
    contexts.symlink_to(source, target_is_directory=True)
PY
fi
if [[ -z "${DOCKER_CONTEXT:-}" && -z "${DOCKER_HOST:-}" && -S "$HOME/.orbstack/run/docker.sock" ]]; then
  export DOCKER_HOST="unix://$HOME/.orbstack/run/docker.sock"
fi
exec docker "$@"
