#!/usr/bin/env bash
# Mechanical enforcement of the architecture boundaries in
# FPRIME_LEAN_ARCHITECTURE_SPEC.md. Every check is a hard failure.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
status=0
fail() { echo "[boundary] $1" >&2; status=1; }

lean_files() {
  find NightstreamFPrime tests -name '*.lean' -print
  printf '%s\n' NightstreamFPrime.lean
}

# 1. No import from the frozen package.
if lean_files | xargs grep -nE '^import Nightstream\.' 2>/dev/null; then
  fail "import from formal/nightstream-lean is prohibited"
fi

# 2. No native_decide anywhere; no ofReduceBool; no sorry/admit/axiom/unsafe.
if lean_files | xargs grep -n 'native_decide\|ofReduceBool' 2>/dev/null; then
  fail "native_decide / ofReduceBool is prohibited"
fi
if lean_files | xargs grep -nE '^\s*(axiom|unsafe)\b|\bsorry\b|\badmit\b' 2>/dev/null; then
  fail "sorry / admit / axiom / unsafe is prohibited"
fi

# 3. No generated modules and no embedded artifact data in the proof library.
if find NightstreamFPrime -type d -iname 'Generated' | grep -q .; then
  fail "Generated/ directories are prohibited inside the proof library"
fi
if lean_files | xargs grep -ln 'artifactSha256\|GENERATED FILE\|do not edit' 2>/dev/null; then
  fail "generated-artifact markers found in the proof library"
fi
# Any single source line longer than 2000 characters is treated as embedded data.
if lean_files | xargs awk 'length($0) > 2000 { print FILENAME ":" FNR; exit 1 }' 2>/dev/null; then :; else
  fail "embedded data (line > 2000 chars) found"
fi

# 4. No recursion/heartbeat overrides: they mark artifact-sized elaboration.
# A maxRecDepth override is admitted only as a scoped `... in` with the
# fixed-size marker; anything else (and every maxHeartbeats) is a failure.
if lean_files | xargs grep -nE 'set_option (maxRecDepth|maxHeartbeats|synthInstance\.maxHeartbeats)' 2>/dev/null \
   | grep -vE 'set_option (maxRecDepth|maxHeartbeats) [0-9]+ in -- fixed-size:'; then
  fail "recursion/heartbeat override without scoped fixed-size justification"
fi

# 5. Files below 1500 lines.
if lean_files | xargs wc -l | awk '$2 != "total" && $1 >= 1500 { print; found=1 } END { exit found ? 1 : 0 }'; then :; else
  fail "file at or above 1500 lines"
fi

# 6. Explicit roots only: no glob in lakefile.
if grep -n 'globs' lakefile.toml; then
  fail "lakefile must use explicit roots, not globs"
fi

# 7. One profile. No radix-four / k_rho 14 / b = 4 surface.
if lean_files | xargs grep -niE 'radix.?four|radix4|k_?rho.*14|kRho.*14|\bb := 4\b' 2>/dev/null; then
  fail "alternate-profile surface found"
fi

# 8. Layer direction: Spec ← Circuit ← Gadgets ← Lifecycle ← Layout ← Export.
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

while IFS= read -r module; do
  queue+=("$module")
done < <(awk '
  /^(root|roots)[[:space:]]*=/ {
    line = $0
    while (match(line, /"[A-Za-z0-9_.]+"/)) {
      print substr(line, RSTART + 1, RLENGTH - 2)
      line = substr(line, RSTART + RLENGTH)
    }
  }
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
  done < <(sed -nE 's/^import[[:space:]]+([A-Za-z0-9_.]+).*$/\1/p' "$file")
done

for module in "${!module_path[@]}"; do
  if [[ -z "${reachable[$module]+present}" ]]; then
    fail "unreachable Lean module: ${module_path[$module]}"
  fi
done

# 10. The superseded Rust PiCCS/F' circuit is private diagnostic code. Once
# PiCCS rows reached the sealed package, no public constructor or registered
# integration target may reopen that native relation or an unapproved backend.
RUST_ROOT="$ROOT/../../crates/neo-fold-clean"
while IFS=: read -r file pattern; do
  if grep -nE "$pattern" "$RUST_ROOT/$file"; then
    fail "public superseded Rust F' authority in $file"
  fi
done <<'EOF'
src/lifecycle/mod.rs:^[[:space:]]*pub fn nifs_v_circuit_config\b
src/paper/reductions/pi_ccs_circuit/verifier.rs:^[[:space:]]*pub fn enforce_pi_ccs(_with_matrix_digest_wires)?\b
src/paper/nifs/circuit/mod.rs:^[[:space:]]*pub fn enforce_nifs_v_circuit_with_transcript
src/paper/f_prime/r1cs.rs:^[[:space:]]*pub fn enforce_f_prime_recursive_step_circuit
EOF

require_test_only_module() {
  local file="$1"
  local declaration="$2"
  if ! awk -v declaration="$declaration" '
    previous == "#[cfg(test)]" && $0 == declaration { found = 1 }
    { previous = $0 }
    END { exit found ? 0 : 1 }
  ' "$RUST_ROOT/$file"; then
    fail "$declaration is not test-only in $file"
  fi
}

require_test_only_module src/engine/mod.rs "pub mod decider;"
require_test_only_module src/frontends/r1cs_f_prime/mod.rs "pub mod native_ccs;"
require_test_only_module src/frontends/r1cs_f_prime/mod.rs "pub mod nebula_combined_ccs;"
require_test_only_module src/frontends/r1cs_f_prime/terminal_r1cs/mod.rs "mod lifecycle;"

if grep -nE '^[[:space:]]*pub mod f_prime;' \
  "$RUST_ROOT/src/frontends/nebula/mod.rs"; then
  fail "unfinished Stage 2 F' lifecycle is public"
fi
if ! grep -qxF 'pub(crate) mod f_prime;' \
  "$RUST_ROOT/src/frontends/nebula/mod.rs"; then
  fail "unfinished Stage 2 F' lifecycle is not crate-private"
fi

if ! grep -qxF 'autotests = false' "$RUST_ROOT/Cargo.toml"; then
  fail "Stage 2 F' auto-discovered integration tests are not disabled"
fi

if grep -nF 'terminal_link_rows_export' \
  "$RUST_ROOT/tests/system/formal_conformance.rs"; then
  fail "retired terminal-link row exporter remains in the aggregate test"
fi

while IFS= read -r retired_test; do
  if grep -nF "path = \"$retired_test\"" "$RUST_ROOT/Cargo.toml"; then
    fail "retired private-authority test remains registered: $retired_test"
  fi
done <<'EOF'
tests/gadgets/ce_continuity.rs
tests/gadgets/f_prime_base_state_lean_artifact.rs
tests/gadgets/f_prime_state_link_lean_artifact.rs
tests/gadgets/f_prime_terminal_link_lean_artifact.rs
tests/gadgets/nebula_recursive_arm_lean_artifact.rs
tests/nebula/f_prime.rs
tests/nebula/streaming_claim_replay.rs
tests/nebula/streaming_claim_replay_linked_overlay.rs
tests/nebula/streaming_lifecycle_profile.rs
tests/nebula/streaming_lifecycle_semantic_link.rs
tests/nebula/streaming_pi_ccs_round_relation.rs
tests/nebula/streaming_pi_rlc_family_b2_k16.rs
tests/nebula/streaming_pi_rlc_family_public.rs
tests/nebula/streaming_pi_rlc_family_relation.rs
tests/nebula/streaming_pi_rlc_family_replay.rs
tests/nebula/streaming_pi_rlc_phase_envelope.rs
tests/nebula/streaming_pi_rlc_poseidon_call_layout.rs
tests/nebula/streaming_pi_rlc_poseidon_provenance.rs
tests/nebula/streaming_prelude_relation.rs
tests/nebula/streaming_prior_state_replay_artifact.rs
tests/nebula/streaming_prior_state_replay_relation.rs
tests/nebula/streaming_program.rs
tests/nebula/streaming_state_envelope.rs
tests/nebula/streaming_terminal_profile.rs
tests/system/formal_conformance/terminal_link_rows_export.rs
EOF

if grep -nE '^name = "(gadgets_f_prime_recursive_manifest|gadgets_nifs_compiler_conformance|reductions_nifs_v_transcript|f_prime_r1cs|system_decider_r1cs|system_lean_native_ccs_manifest|perf_fibonacci_bits|perf_lowered_nifs_v)"$' \
  "$RUST_ROOT/Cargo.toml"; then
  fail "retired native F' integration target remains registered"
fi

if (( status == 0 )); then echo "[boundary] all checks passed"; fi
exit $status
