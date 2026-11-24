#!/usr/bin/env bash
# Environment helper for myvideoai under a restricted sandbox:
#   - torch / torchvision are vendored (extracted wheels) under .vendored/torch_pkgs
#   - pure-python deps (basicsr, realesrgan, cv2, ...) under .vendored/py_pkgs
#   - sitecustomize.py in py_pkgs auto-shims removed torchvision/torch APIs
#
# The project root is derived from this script's own location, so the helper
# works from any checkout location.
#
# Usage:  ./run_env.sh -m myvideoai -i in.mp4 -o out.mp4
#     or  source run_env.sh   # then plain `python ...` works in this shell
set -euo pipefail

# project root = parent directory of the myvideoai/ package folder
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Windows Python expects ';' as the path separator, POSIX Python expects ':'.
if [ "${OS:-}" = "Windows_NT" ]; then
  PY_SEP=";"
else
  PY_SEP=":"
fi
export PYTHONPATH="${ROOT}/.vendored/torch_pkgs${PY_SEP}${ROOT}/.vendored/py_pkgs${PYTHONPATH:+${PY_SEP}$PYTHONPATH}"
export KMP_DUPLICATE_LIB_OK=TRUE
# cd to the project ROOT (parent of this package dir) so `python -m myvideoai` resolves
cd "$ROOT"

if [ "$#" -gt 0 ]; then
  exec python "$@"
fi
echo "[run_env] PYTHONPATH=$PYTHONPATH"
echo "[run_env] ready. try:  python -m myvideoai --help"
