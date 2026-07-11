# CIR-U64ADD — no-wrap u64 addition artifact soundness

This property covers the production arithmetic primitive used for
`step_count_out = step_count_in + rows_in_chunk` in F'.

```text
property_id: CIR-U64ADD
claim:
  Any canonical-residue assignment satisfying the exact 319 exported rows
  of three `alloc_u64_bits` calls followed by `enforce_u64_add` has its output
  word equal the integer sum of the two input words. The top-bit equation has
  no carry-out wire, so overflow cannot satisfy the artifact.
assumptions:
  - EuclidPrime goldilocksP, passed as a typed hypothesis, to derive Boolean
    wire values from the emitted `enforce_bit` rows.
  - Assignment values are canonical Goldilocks residues and column 0 is one.
  - The Rust-to-Lean row emitter is trusted until independently verified.
non_goals:
  - Binding the second word to the runtime `rows_in_chunk` constant.
  - Canonical field decomposition; that is property CIR-U64CANON.
  - The surrounding F' state, transcript, NIFS, and hash constraints.
paper_sources:
  - none; this is an implementation-level counter-integrity property
rust_surfaces:
  - crates/neo-fold-clean/src/engine/r1cs_circuit/u64.rs
    (`alloc_u64_bits`, `enforce_u64_add`)
  - crates/neo-fold-clean/src/engine/r1cs_circuit/boolean.rs (`enforce_bit`)
  - crates/neo-fold-clean/src/engine/r1cs_circuit/builder.rs (row emission)
circuit_or_encoding_artifacts:
  - Nightstream/Implementation/R1CS/U64AddArtifact.lean
    (sha256:65c37fec97e5ea8b0da4f4b4e9c522f70705bc3aa28ef2741361d2da69395a4d)
failure_class:
  An under-constrained carry chain or accepted wraparound lets a prover lower
  the post-state `step_count` while claiming a valid F' transition.
counterexample_or_witness:
  `overflowWitness` encodes `u64::MAX + 1 = 0`. It passes rows 0-317 and fails
  exactly row 318 in Rust and Lean.
lean_theorems:
  - Nightstream.Implementation.R1CS.u64AddArtifactRows_eq
  - Nightstream.Implementation.R1CS.u64Add_sound
axiom_report:
  [propext, Quot.sound], guarded fail-closed in tests/Axioms.lean.
proof_hash:
  sha256:abeb3d7bb0af40ed59b5c63080c6d6aed87db9c7e25bd7c5aef18404eea00796
conformance_status:
  artifact-checked for this addition slice; exact row drift and the honest
  and overflow witnesses are checked independently in Rust and Lean. Native
  `advance_state` uses checked addition and the lifecycle regression rejects
  `step_count = u64::MAX` rather than wrapping.
retest_commands:
  - cargo test -p neo-fold-clean --release --test gadgets_u64_add_lean_artifact
  - cargo test -p neo-fold-clean --release --test system_lifecycle_finalization
  - cd formal/nightstream-lean && lake build && lake exe check
```
