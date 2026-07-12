# CIR-U64CANON — canonical-u64 decomposition artifact soundness

The tracer bullet for `CIR-SOUND`: the first theorem stated over an exact
generated R1CS artifact rather than a handwritten surrogate.

```text
property_id: CIR-U64CANON
claim:
  Any assignment (canonical residues, constant-one wire at column 0)
  satisfying the exact 69 exported rows of `decompose_var_to_u64_bits`
  has its 64 bit columns recompose, over the integers, to the canonical
  value of the decomposed field element, with the recomposed value below
  the Goldilocks modulus. In particular the non-canonical `x + p` second
  representation is impossible.
assumptions:
  - EuclidPrime goldilocksP, passed as a typed hypothesis: the Euclid
    divisor property of 2^64 - 2^32 + 1, a consequence of its primality,
    not yet reconstructed locally (spec §9 mathematical boundary).
  - Lean kernel; Rust-to-Lean artifact emitter (the ~60-line
    `emit_lean` in the exporter test) trusted until itself verified.
non_goals:
  - Completeness beyond the exported honest witness.
  - Commitment binding, witness extraction, surrounding F' rows.
  - The committed-representation lowering of these rows.
paper_sources:
  - none (pure implementation property; supports FPR counter transitions)
rust_surfaces:
  - crates/neo-fold-clean/src/engine/r1cs_circuit/u64.rs
    (`decompose_var_to_u64_bits`, `alloc_u64_bits`)
  - crates/neo-fold-clean/src/engine/r1cs_circuit/boolean.rs (`enforce_bit`)
  - crates/neo-fold-clean/src/engine/r1cs_circuit/builder.rs (row emission)
circuit_or_encoding_artifacts:
  - Nightstream/Implementation/R1CS/Artifacts/CanonicalU64/Generated/CanonicalU64Artifact.lean (generated;
    sha256:ede705cfce2629faa01db47136ca76277920debd9597b25dfe206294f9149497)
failure_class:
  A witness re-encoding a small field value x through the bits of x + p,
  forging a second valid 64-bit representation (counter/range malleability
  in the F' counter path).
counterexample_or_witness:
  `forgedWitness` (5 encoded as the bits of 5 + p): passes rows 0-67,
  fails exactly row 68 (the canonicity gate) in both Rust
  (`first_unsatisfied_row() == Some(68)`) and Lean
  (`tests/R1csArtifact.lean`).
lean_theorems:
  - Nightstream.Implementation.R1CS.canonicalU64_sound
  - Nightstream.Implementation.R1CS.bitRow_le_one
axiom_report:
  [propext, Quot.sound] — guarded fail-closed in tests/Axioms.lean.
proof_hash:
  sha256:bb2fb8213a1a328c73cf42af89666b836ea9912c2ef4ae36e9a62828c5f848cf
conformance_status:
  artifact-checked for this gadget slice. Row-content drift gate and twin
  witness checks live in Rust; `lake exe check` recomputes both witness
  verdicts over the exported rows.
retest_commands:
  - cargo test -p neo-fold-clean --release --test gadgets_lean_artifact
  - cd formal/nightstream-lean && lake build && lake exe check
```
