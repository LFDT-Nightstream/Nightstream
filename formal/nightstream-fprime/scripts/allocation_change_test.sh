#!/usr/bin/env bash
# Run the fixed extra-cell probe in an existing clean scratch worktree.
# The caller owns scratch creation/cache preparation. No production source moves.
# A failed full build is diagnostic evidence; classify its named failures before
# deciding whether an unrelated phase proof depends on the changed allocation.
set -euo pipefail

if (( $# != 2 )); then
  echo "usage: allocation_change_test.sh <clean-scratch-worktree> <new-evidence-directory>" >&2
  exit 2
fi
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
owner="$(git -C "$script_dir" rev-parse --show-toplevel)"
scratch="$(cd "$1" && pwd)"
if [[ "$scratch" == "$owner" || ! -f "$scratch/.git" ]]; then
  echo "The probe requires a separate Git worktree" >&2
  exit 2
fi
if [[ -n "$(git -C "$scratch" status --porcelain)" ]]; then
  echo "The scratch worktree must be clean" >&2
  exit 2
fi
if [[ -e "$2" ]]; then
  echo "The evidence directory must be new" >&2
  exit 2
fi
mkdir -p -- "$2"
evidence="$(cd "$2" && pwd)"
patch="$script_dir/fixtures/signed_split_extra_cell.patch"
project="$scratch/formal/nightstream-fprime"
git -C "$scratch" rev-parse HEAD > "$evidence/source-commit.txt"
git -C "$scratch" apply --check "$patch"
git -C "$scratch" apply "$patch"
trap 'git -C "$scratch" apply -R "$patch"' EXIT
git -C "$scratch" diff --binary > "$evidence/perturbation.patch"

run_target() {
  local label="$1" target="$2" rc=0
  (cd "$project" && timeout --signal=KILL 1500s bash scripts/validate.sh build "$target") \
    > "$evidence/$label.log" 2>&1 || rc=$?
  printf '%s\t%s\t%s\n' "$label" "$target" "$rc" >> "$evidence/results.tsv"
  echo "[allocation] $label exit=$rc"
  return "$rc"
}

# A failed leaf is an invalid experiment, not evidence about parent interfaces.
run_target leaf NightstreamFPrime.Layout.PiDEC.v1_1.Leaves.SignedSplitScalar
control_failed=0
run_target piccs NightstreamFPrime.Layout.PiCCS.v1_1.Lowering || control_failed=1
run_target pirlc NightstreamFPrime.Layout.PiRLC.v1_1.Lowering || control_failed=1
run_target pirlc_input NightstreamFPrime.Layout.Stage1.PiRLCInputBounds || control_failed=1
full_failed=0
# Distinguish interface/consumer failures from value-only default-profile checks.
run_target pidec_ranges NightstreamFPrime.Layout.Stage1.PiDECSourceSupportData || full_failed=1
run_target nifs_consumer NightstreamFPrime.Export.Stage1.PiDECCommitmentMatrixEntry || full_failed=1
run_target full NightstreamFPrime || full_failed=1

python3 -B - "$evidence" <<'PY'
import json
from pathlib import Path
import re
import sys

root = Path(sys.argv[1])
results = []
for row in (root / 'results.tsv').read_text().splitlines():
    label, target, status = row.split('\t')
    log = (root / (label + '.log')).read_text()
    errors = re.findall(r'^error: (NightstreamFPrime/[^\n]+)', log, re.MULTILINE)
    results.append({'label': label, 'target': target, 'exit_code': int(status), 'errors': errors})
(root / 'report.json').write_text(json.dumps({
    'source_commit': (root / 'source-commit.txt').read_text().strip(),
    'change': 'One extra sign-hint cell per scalar; 270 scalar invocations; unchanged constraints/specification',
    'scope': 'Compiler results. Review parent-size failures separately from unrelated phase failures.',
    'results': results}, indent=2) + '\n')
PY

if (( control_failed || full_failed )); then
  echo "[allocation] inspect report.json; the original source is restored on exit"
  exit 1
fi
echo "[allocation] all selected targets passed; the original source is restored on exit"
