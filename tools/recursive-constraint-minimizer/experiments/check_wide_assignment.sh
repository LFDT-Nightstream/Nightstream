#!/usr/bin/env bash
# Check the full sealed transport against the Lean wide-sampler base fixture.
set -euo pipefail
if [[ $# -ne 1 ]]; then
  echo "usage: $0 /path/to/lean-wide-physical-package.json" >&2
  exit 2
fi
package=$(realpath "$1")
cd "$(dirname "$0")/../../.."
# The project requires a 300-second cap for each non-Lean test invocation.
printf '%s\n%s\n' "$package.sealed.json" "$package.base.json" | timeout --signal=KILL 300 cargo test -p nightstream-fprime --release --lib \
  package::sealed::wide_assignment_tests::wide_sealed_package_constructs_a_complete_assignment \
  -- --exact --ignored --nocapture
