# Generated Lean files

Generated Lean modules are committed mirrors of Rust output. They are review
evidence. They are not protocol authority and they do not prove the meaning of
the rows that they contain.

## Ownership

Generated modules live below these roots:

```text
Nightstream/Implementation/R1CS/Artifacts/<owner>/Generated/
Nightstream/Implementation/Rust/CanonicalConformance/<owner>/Generated/
```

Handwritten facade and correspondence modules can import generated data.
Protocol and security modules must not import generated shards directly. The
static quarantine gate checks this rule.

The Rust test that contains the artifact path is the path owner. A generated
file without a live Rust path owner is stale and must be removed.

## Current drift owners

| Evidence family | Rust drift target |
|---|---|
| Goldilocks Poseidon2 constants | `cargo test -p neo-ccs --release --test poseidon2_round_constants` |
| Phi81 bar matrix | `cargo test -p neo-math --release --test phi81_bar_lean_artifact` |
| Padded-row one-joint PiCCS layout, gamma slots, transcript tags, and codec | `cargo test -p neo-reductions --release --test padded_row_identity_lean_artifact` |
| Canonical u64, increment, and addition | `gadgets_lean_artifact`, `gadgets_u64_increment_lean_artifact`, `gadgets_u64_add_lean_artifact` |
| Seeded Phi81 and shifted ternary | `gadgets_seeded_phi81_lean_artifact`, `gadgets_shifted_ternary_lean_artifact` |
| Poseidon2 permutation | `gadgets_poseidon2_lean_artifact` |
| F-prime counter, encoding, state links, terminal link, base state, chunk digest, and base program | The matching `gadgets_f_prime_*_lean_artifact` targets |
| PiRLC packed-Mod-5 and aggregate-acceptance leaves | `gadgets_packed_mod5_lean_artifact`, `gadgets_aggregate_acceptance_lean_artifact` |
| PiRLC projection boundary | `gadgets_pi_rlc_projection_boundary` |
| Recursive verifier manifest, transcript layout, source roles, and output-authority S-box census | `gadgets_f_prime_recursive_manifest` |
| Isolated SumCheck compiler row and both one-joint production call sites | `gadgets_nifs_compiler_conformance` |
| Selective 270-coordinate carrier and selector coverage | `f_prime_selective_snapshot` |
| Strict PiDEC source rows | `f_prime_pi_dec_source_lean_artifact` |
| Canonical PiDEC X rows | `f_prime_pi_dec_canonical_x_lean_artifact` |
| Native step, terminal link, and one-slot conformance records | `system_formal_conformance` |

The selected full recursive circuit is checked by live Rust synthesis tests.
There is no committed full-matrix Lean dump. Exact runtime synthesis is the
Rust-conformant gate; the absence of a complete Lean matrix artifact remains a
declared assurance boundary.

## Required focused commands

Run these commands from the repository root. Every Rust command is subject to
the five-minute non-Lean limit.

```bash
cargo test -p neo-reductions --release \
  --test padded_row_identity_lean_artifact

cargo test -p neo-fold-clean --release \
  --test gadgets_nifs_compiler_conformance \
  isolated_sumcheck_round_artifact_matches -- --exact

cargo test -p neo-fold-clean --release \
  --test gadgets_nifs_compiler_conformance \
  full_history_sumcheck_call_sites_match_isolated_compiler -- --exact

cargo test -p neo-fold-clean --release \
  --test f_prime_selective_snapshot \
  selective_carrier_270_lean_artifact_matches_compiler -- --exact

cargo test -p neo-fold-clean --release \
  --test f_prime_selective_snapshot \
  selective_snapshot_selector_gate_coverage_matches_final_matrices -- --exact
```

## Review workflow

1. Run the owning Rust target.
2. If it writes a sibling `*.expected` file, inspect the complete diff and its
   source metadata. A changed digest is not approval.
3. Promote the reviewed bytes deliberately.
4. Remove the `*.expected` file.
5. Run the same Rust target again.
6. Run `./scripts/validate.sh static` from `formal/nightstream-lean`.
7. Run the focused Lean build and axiom target that consumes the artifact.

Do not commit:

- `*.expected` review output;
- `.lake/`, `target/`, or profiling output;
- a generated shard without a live Rust drift owner;
- a generated artifact that names a removed protocol variant.
