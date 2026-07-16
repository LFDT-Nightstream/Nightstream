# Generated files

Generated Lean modules are committed evidence mirrors of production Rust output.
They are reviewed artifacts, not handwritten proof code and not authority by
themselves.

## Location and ownership

Generated modules live below:

```text
Nightstream/Implementation/R1CS/Artifacts/<owner>/Generated/
```

Small generated entrypoints have stable facades under `R1CS/Artifacts`.
Handwritten assemblies in `R1CS/Ownership` expose complete ordered multi-shard
artifacts. Handwritten theorems in `R1CS/Correspondence` assign semantics to
those rows. Consumers outside the R1CS implementation import those stable
modules instead of individual generated shards.

The generator families are:

| Generated owner | Rust drift/regeneration target |
|---|---|
| Phi81 runtime bar matrix | `cargo test -p neo-math --release --test phi81_bar_lean_artifact` |
| SplitNc packed-carrier counterexample | `cargo test -p neo-reductions --release --test pi_ccs_nc_carrier_lean_artifact` |
| Fixed F' carrier-fixture NIFS/F' counterexample | `cargo test -p neo-fold-clean --release --test f_prime_fixed_carrier_nifs_lean_artifact` |
| Canonical-u64 | `gadgets_lean_artifact` |
| Seeded Phi81 | `gadgets_seeded_phi81_lean_artifact` |
| Shifted ternary source and schema-3 isolated shared-slot lowering (20 residual pairs, one tail, 82 products) | `gadgets_shifted_ternary_lean_artifact` |
| U64 increment/addition | `gadgets_u64_increment_lean_artifact`, `gadgets_u64_add_lean_artifact` |
| F' counter, encoding, links, base pins | `gadgets_f_prime_counter_lean_artifact`, `gadgets_f_prime_encoding_lean_artifact`, `gadgets_f_prime_terminal_link_lean_artifact`, `gadgets_f_prime_state_link_lean_artifact`, `gadgets_f_prime_base_state_lean_artifact` |
| Poseidon2, F' digest, base program, CE continuity | `gadgets_poseidon2_lean_artifact`, `gadgets_f_prime_chunk_digest_lean_artifact`, `gadgets_f_prime_base_program_lean_artifact`, `gadgets_f_prime_ce_continuity_lean_artifact` |
| Steady-recursive manifest | `gadgets_f_prime_recursive_manifest` |
| Fixed F' base/recursive source-role census and compact ordinary source-loop placement metadata | `cargo test -p neo-fold-clean --release --test f_prime_full_relation -- --nocapture` |
| Output-authority Poseidon2 S-box call manifest | `gadgets_f_prime_recursive_manifest output_authority_sbox_lean::output_authority_sbox_lean_manifest_matches_audited_production -- --exact` |
| NIFS/SumCheck compiler artifact | `gadgets_nifs_compiler_conformance` |
| PiRLC projection boundary | `gadgets_pi_rlc_projection_boundary` |
| Full-history M4 manifest and owner shards | `system_decider_r1cs` targeted tests listed below |

The Rust test source is the authoritative path registry. When a generated Lean
file moves, update the matching Rust path constant in the same change; otherwise
the drift gate is disconnected.

## Regeneration workflow

Each generator renders the expected bytes, compares them with the committed
file, and fails on drift. Most generators write a sibling `*.expected` file on
failure.

1. Run the relevant target from the repository root, in release mode, under the
   hard five-minute non-Lean cap.
2. Inspect the generated `*.expected` diff and the associated row/profile
   metadata. A changed digest alone is not approval.
3. Replace the committed artifact deliberately, remove the `*.expected` file,
   and rerun the same Rust target.
4. From `formal/nightstream-lean`, run `./scripts/validate.sh all`.
5. Update the property specification and assurance ledger if the artifact
   identity, supported profile, or theorem scope changed.

For a single gadget:

```bash
timeout 300s cargo test -p neo-fold-clean --release \
  --test gadgets_lean_artifact
```

Replace the final target with the target from the table. Full-history outputs
are regenerated and drift-checked with these bounded targeted invocations:

```bash
timeout 300s cargo test -p neo-fold-clean --release \
  --test system_decider_r1cs \
  m4_manifest::full_history_m4_manifest_matches_exact_composed_rows -- --exact

timeout 300s cargo test -p neo-fold-clean --release \
  --test system_decider_r1cs \
  m4_manifest::terminal_parent_and_accumulator_artifacts_match_exact_rows -- --exact
```

`timeout 300s` documents the process limit for a normal shell. Agent tool calls
must additionally set their tool timeout to 300,000 ms as required by the root
instructions.

## Review gate

Do not commit:

- `*.expected` review output;
- `.lake/`, `target/`, or profiling output;
- a generated shard without its stable facade and drift target;
- a manifest whose Rust generator or source anchors are absent from the same
  branch.
