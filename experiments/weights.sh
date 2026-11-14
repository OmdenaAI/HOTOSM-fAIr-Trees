#!/usr/bin/env bash
# Model files under weights/ are tracked by DVC. Git holds the .dvc receipts (md5 + size) and
# weights/SHA256SUMS; the files live in the public Hugging Face dataset repo
# OmarHelmy/fair-trees-weights, which DVC reads as a plain HTTP remote.
#
#   experiments/weights.sh pull    download every tracked file, then check its sha256
#   experiments/weights.sh check   check the sha256 of the files already on disk
#   experiments/weights.sh push    upload new files (needs `hf auth login` with write access)
#
# Adding a file: dvc add weights/<path>, add its line to weights/SHA256SUMS, then push.
set -euo pipefail

HF_REPO="OmarHelmy/fair-trees-weights"
cd "$(git rev-parse --show-toplevel)"

# pathspec 1.x removed a name dvc 3.63 imports, so it is held below 1.
dvc() { uvx -q --from "dvc==3.63.0" --with "pathspec<1" dvc "$@"; }
check() { (cd weights && sha256sum -c SHA256SUMS); }

case "${1:-}" in
  pull)
    dvc pull
    check
    ;;
  check)
    check
    ;;
  push)
    check
    # HF takes no plain HTTP uploads, so DVC writes its cache layout to a local folder that hf uploads.
    staging="$(mktemp -d)"
    trap 'rm -rf "$staging"' EXIT
    dvc remote add --local --force staging "$staging"
    dvc push -r staging
    dvc remote remove --local staging
    hf upload "$HF_REPO" "$staging" . --repo-type dataset --commit-message "Add DVC cache files"
    ;;
  *)
    echo "usage: $0 pull|check|push" >&2
    exit 2
    ;;
esac
