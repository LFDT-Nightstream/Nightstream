# CIR-POSEIDON2

```text
property_id: CIR-POSEIDON2
claim:
  The exact 600 production Goldilocks width-8 Poseidon2 permutation rows form
  a deterministic SSA program from the constant-one lane and eight input
  lanes to eight output lanes. Every satisfying assignment agrees with the
  executable extracted program; equal inputs force equal outputs; and every
  canonical input has a satisfying witness constructed by that program.
assumptions:
  - Canonical Goldilocks representatives and constant-one column zero.
  - The Rust-to-Lean renderer is a translation boundary. The Rust drift gate
    hashes the exact sparse triplets and rejects any unreviewed row change.
non_goals:
  - Collision resistance or permutation security.
  - Equality with the native Rust `PERM` function; that is RUST-REFINE/M5.
  - Claiming one fixed permutation artifact covers different widths,
    parameter sets, or round schedules.
paper_sources:
  - none; this is an implementation-level circuit functionality property.
rust_surfaces:
  - crates/neo-fold-clean/src/engine/r1cs_circuit/poseidon2.rs
  - crates/neo-fold-clean/tests/gadgets/poseidon2_lean_artifact.rs
circuit_or_encoding_artifacts:
  - Poseidon2PermutationArtifact.lean, schema 1, 600 rows, 609 columns.
failure_class:
  The circuit permits two different output states for one input, rejects a
  valid deterministic execution, or silently changes its sparse row program.
counterexample_or_witness:
  Rust retains an honest witness and mutations of all eight output lanes; the
  exact artifact drift test covers every row and column.
lean_theorems:
  - Nightstream.Implementation.R1CS.Poseidon2PermutationSound.poseidon2Permutation_sound
  - Nightstream.Implementation.R1CS.Poseidon2PermutationSound.poseidon2Permutation_outputs_unique
  - Nightstream.Implementation.R1CS.Poseidon2PermutationSound.poseidon2Permutation_complete
axiom_report:
  Uses [propext, Quot.sound], guarded fail-closed in tests/Axioms.lean. The
  600-entry structural certificate uses kernel `decide`, not native_decide.
proof_hash:
  Full row artifact 85b252a27c9d203cbd7e0daaa967e531eb10dc394efa0b73e51adf32d55ed664.
  Witness artifact d6113649596ceb307ca4b580b11e435ecebdc652f89e157a21075a809a8e4e4a.
conformance_status:
  Artifact-checked for exact row functionality, output uniqueness, and
  completeness. Native Rust permutation refinement remains M5.
retest_commands:
  - cd formal/nightstream-lean && lake build && lake exe check
  - cargo test -p neo-fold-clean --release --test gadgets_poseidon2_lean_artifact
```
