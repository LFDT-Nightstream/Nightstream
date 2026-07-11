# CIR-U64INC — no-wrap u64 increment artifact soundness

This property covers the production arithmetic primitive used for
`chunk_count_out = chunk_count_in + 1` in F'.

```text
property_id: CIR-U64INC
claim:
  Any canonical-residue assignment satisfying the exact 255 exported rows
  of `alloc_u64_bits(input)`, `alloc_u64_bits(output)`, and
  `enforce_u64_increment(input, output)` has output = input + 1 over the
  integers. The last equation has no carry-out wire, so overflow cannot
  satisfy the artifact.
assumptions:
  - EuclidPrime goldilocksP, passed as a typed hypothesis, to derive Boolean
    wire values from the emitted `enforce_bit` rows.
  - Assignment values are canonical Goldilocks residues and column 0 is one.
  - The Rust-to-Lean row emitter is trusted until independently verified.
non_goals:
  - Canonical field decomposition; that is property CIR-U64CANON.
  - `step_count + rows_in_chunk`; that requires the u64-add artifact.
  - The surrounding F' state, transcript, NIFS, and hash constraints.
paper_sources:
  - none; this is an implementation-level counter-integrity property
rust_surfaces:
  - crates/neo-fold-clean/src/engine/r1cs_circuit/u64.rs
    (`alloc_u64_bits`, `enforce_u64_increment`)
  - crates/neo-fold-clean/src/engine/r1cs_circuit/boolean.rs (`enforce_bit`)
  - crates/neo-fold-clean/src/engine/r1cs_circuit/builder.rs (row emission)
circuit_or_encoding_artifacts:
  - Nightstream/Implementation/R1CS/U64IncrementArtifact.lean
    (sha256:9dcbe8e37068eebf875aaf690475f993072bf55554e68810dfb39350dbd1b303)
failure_class:
  Counter wraparound or an under-constrained carry chain lets a prover claim
  a smaller post-state counter after reaching the u64 boundary.
counterexample_or_witness:
  `overflowWitness` encodes `u64::MAX + 1 = 0`. It passes rows 0-253 and fails
  exactly row 254 in Rust and Lean.
lean_theorems:
  - Nightstream.Implementation.R1CS.artifactRows_eq
  - Nightstream.Implementation.R1CS.u64Increment_sound
axiom_report:
  [propext, Quot.sound], guarded fail-closed in tests/Axioms.lean.
proof_hash:
  sha256:f1b39169973fda94e27616045c265b232d071b4868644940e613189827d91796
conformance_status:
  artifact-checked for this increment slice; exact row drift and the honest
  and overflow witnesses are checked independently in Rust and Lean. Native
  `advance_state` uses checked addition and the lifecycle regression rejects
  `chunk_count = u64::MAX` rather than wrapping.
retest_commands:
  - cargo test -p neo-fold-clean --release --test gadgets_u64_increment_lean_artifact
  - cargo test -p neo-fold-clean --release --test system_lifecycle_finalization
  - cd formal/nightstream-lean && lake build && lake exe check
```
