#!/usr/bin/env bash
# Mechanical enforcement of the architecture boundaries in
# FPRIME_LEAN_ARCHITECTURE_SPEC.md. Every check is a hard failure.
#   check-boundaries.sh [package-dir]
# With no argument, it checks F′. Another package, such as
# ../compression-lean, gets only the package-wide checks. The layer,
# matrix and consumer rules belong to F′.
set -euo pipefail

# Associative arrays require Bash 4. macOS ships Bash 3.2; use Homebrew when present.
if (( BASH_VERSINFO[0] < 4 )); then
  for boundary_shell in /opt/homebrew/bin/bash /usr/local/bin/bash; do
    if [[ -x "$boundary_shell" ]] && "$boundary_shell" -c '(( BASH_VERSINFO[0] >= 4 ))'; then
      exec "$boundary_shell" "${BASH_SOURCE[0]}" "$@"
    fi
  done
  echo "[boundary] Bash 4 or newer is required; run this script with a newer bash." >&2
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
# Resolve the argument from the caller's directory and compare physical
# paths, so that every spelling of the F′ path selects the F′ checks.
package="$(cd "${1:-$ROOT}" && pwd -P)"
cd "$package"
status=0
fail() { echo "[boundary] $1" >&2; status=1; }

lean_files() {
  find . -path ./.lake -prune -o -name '*.lean' -print | sed 's#^\./##'
}

if lean_files | xargs grep -nE '^import Nightstream\.' 2>/dev/null; then
  fail "import from formal/nightstream-lean is prohibited"
fi

if lean_files | xargs grep -nE 'native_decide|ofReduceBool|\+native\b|\bnative\s*:=\s*true\b' 2>/dev/null; then
  fail "native_decide / decide +native / ofReduceBool is prohibited"
fi
# `set_option … in`, a doc comment, attributes and modifiers can come before
# `axiom` or `unsafe` on its line. A comment that only names the word passes.
if lean_files | xargs grep -nE '^\s*(set_option\s+\S+\s+\S+\s+in\s+)*(/--.*-/\s*)?(@\[.*\]\s*)?((private|protected|public|noncomputable|unsafe)\s+)*(axiom|unsafe)\b|\bsorry\b|\badmit\b' 2>/dev/null; then
  fail "sorry / admit / axiom / unsafe is prohibited"
fi

# Read all of find's output. A `grep -q` reader can exit first; under
# pipefail, the writer's SIGPIPE then turns a match into a pass.
if [[ -n "$(find . -path ./.lake -prune -o -type d -iname generated -print)" ]]; then
  fail "Generated/ directories are prohibited inside the proof library"
fi
if lean_files | xargs grep -ln 'artifactSha256\|GENERATED FILE\|do not edit' 2>/dev/null; then
  fail "generated-artifact markers found in the proof library"
fi
if lean_files | xargs awk 'length($0) > 2000 { print FILENAME ":" FNR; exit 1 }' 2>/dev/null; then :; else
  fail "embedded data (line > 2000 chars) found"
fi

if lean_files | xargs grep -nE 'set_option (maxRecDepth|maxHeartbeats|synthInstance\.maxHeartbeats)' 2>/dev/null \
   | grep -vE 'set_option (maxRecDepth|maxHeartbeats) [0-9]+ in -- fixed-size:'; then
  fail "recursion/heartbeat override without scoped fixed-size justification"
fi

if lean_files | xargs wc -l | awk '$2 != "total" && $1 >= 1500 { print; found=1 } END { exit found ? 1 : 0 }'; then :; else
  fail "file at or above 1500 lines"
fi

if grep -n 'globs' lakefile.toml; then
  fail "lakefile must use explicit roots, not globs"
fi

if lean_files | xargs grep -niE 'radix.?four|radix4|k_?rho.*14|kRho.*14|\bb := 4\b' 2>/dev/null; then
  fail "alternate-profile surface found"
fi

# 9. Every source module must be reachable from one declared library or
# executable root. An unimported file is not checked by `lake build` and
# cannot provide assurance evidence.
declare -A module_path=()
declare -A reachable=()
queue=()
while IFS= read -r file; do
  module="${file%.lean}"
  module="${module//\//.}"
  module_path["$module"]="$file"
done < <(lean_files)

# The modules that a file imports: the `import` tokens of its header, up to the
# first other token. Comments are removed first, and block comments nest, so a
# comment that holds an `import` line imports nothing.
imports_of() {
  awk '
    {
      rest = $0; code = ""
      while (rest != "") {
        o = index(rest, "/-"); c = index(rest, "-/")
        if (depth > 0) {
          if (c && (!o || c < o)) { depth--; rest = substr(rest, c + 2) }
          else if (o) { depth++; rest = substr(rest, o + 2) }
          else rest = ""
        } else {
          d = index(rest, "--")
          if (d && (!o || d < o)) { code = code substr(rest, 1, d - 1); rest = "" }
          else if (o) { code = code substr(rest, 1, o - 1) " "; depth++; rest = substr(rest, o + 2) }
          else { code = code rest; rest = "" }
        }
      }
      n = split(code, word)
      for (i = 1; i <= n; i++) {
        if (name) { print word[i]; name = 0 }
        else if (word[i] == "import") name = 1
        else exit
      }
    }' "$1"
}

while IFS= read -r module; do
  queue+=("$module")
done < <(awk '
  function names(line) {
    while (match(line, /"[A-Za-z0-9_.]+"/)) {
      print substr(line, RSTART + 1, RLENGTH - 2)
      line = substr(line, RSTART + RLENGTH)
    }
  }
  # A `roots` array may continue on later lines until its closing bracket.
  /^(root|roots)[[:space:]]*=/ { names($0); open = ($0 ~ /\[/ && $0 !~ /\]/); next }
  open { names($0); if ($0 ~ /\]/) open = 0 }
' lakefile.toml)

queue_index=0
while (( queue_index < ${#queue[@]} )); do
  module="${queue[$queue_index]}"
  ((queue_index += 1))
  [[ -z "${reachable[$module]+present}" ]] || continue
  file="${module_path[$module]:-}"
  [[ -n "$file" ]] || continue
  reachable["$module"]=1
  while IFS= read -r imported; do
    [[ -n "${module_path[$imported]:-}" ]] || continue
    [[ -n "${reachable[$imported]+present}" ]] || queue+=("$imported")
  done < <(imports_of "$file")
done

for module in "${!module_path[@]}"; do
  if [[ -z "${reachable[$module]+present}" ]]; then
    fail "unreachable Lean module: ${module_path[$module]}"
  fi
done

if [[ "$package" != "$ROOT" ]]; then
  if (( status == 0 )); then echo "[boundary] $package: all package-wide checks passed"; fi
  exit $status
fi

declare -A rank=([Spec]=0 [Circuit]=1 [Gadgets]=2 [Lifecycle]=3 [Layout]=4 [Export]=5)
while IFS= read -r f; do
  layer="$(echo "$f" | sed -E 's#^NightstreamFPrime/([A-Za-z]+).*#\1#')"
  [[ -n "${rank[$layer]:-}" ]] || continue
  while IFS= read -r imp; do
    target="$(echo "$imp" | sed -E 's#^import NightstreamFPrime\.([A-Za-z]+).*#\1#')"
    [[ -n "${rank[$target]:-}" ]] || continue
    if (( rank[$target] > rank[$layer] )); then
      fail "$f imports upward from $target"
    fi
  done < <(grep -E '^import NightstreamFPrime\.' "$f" || true)
done < <(find NightstreamFPrime -name '*.lean' -path 'NightstreamFPrime/*/*')

if ! timeout --signal=KILL 300s python3 -B scripts/check_matrix_codecs.py; then
  fail "physical declaration returned to the MatrixProgram codec owner"
fi

if ! timeout --signal=KILL 300s python3 -B - <<'PY'
from pathlib import Path
import sys
sys.path.insert(0, 'scripts')
from rebuild_radius import graph, dependents

prefix = 'NightstreamFPrime.Layout.Stage1.'
interfaces = {prefix + name for name in
              ('Spartan', 'PiDECSourceSupportData', 'RunningTransitionSourceSupportData',
               'Lowering', 'CompactPullback')}
restricted_owners = ('SpartanRows', 'LoweringRows', 'RunningTransitionLowering',
                     'PilotPiCCSPiRLCPiDECRunningTransition',
                     'AssemblerApplicationCompleteness', 'SpartanValues',
                     'RunningTransitionValues')
edges, _ = graph(Path('.'))
failures = []
for name in restricted_owners:
    owner = prefix + name
    for interface in sorted(interfaces.intersection(dependents(edges, owner))):
        failures.append(f'{interface} imports restricted owner {owner}')
if failures:
    for failure in failures:
        print('[layout-interfaces] ' + failure, file=sys.stderr)
    raise SystemExit(1)
print('[layout-interfaces] data interfaces exclude row plans and default-value proofs')
PY
then
  fail "row plans or default-value proofs crossed an allocation data interface"
fi

for retired in crates/neo-fold-legacy crates/neo-wasm crates/wip-spartan tools/recursive-constraint-minimizer/bridge; do
  if [[ -f "$ROOT/../../$retired/Cargo.toml" ]]; then
    fail "retired Rust consumer remains supported: $retired"
  fi
done
if grep -nE 'neo-fold-legacy|neo-wasm|wip-spartan|legacy-adapter' "$ROOT/../../Cargo.toml" "$ROOT/../../crates/"*/Cargo.toml; then
  fail "retired lifecycle dependency or feature remains supported"
fi

if (( status == 0 )); then echo "[boundary] all checks passed"; fi
exit $status
