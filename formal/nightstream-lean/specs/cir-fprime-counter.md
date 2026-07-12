# CIR-FPR-COUNTER — production F' counter-block artifact soundness

```text
property_id: CIR-FPR-COUNTER
claim:
  Every canonical-residue assignment satisfying the exact schema-v1 660-row
  block emitted by the production-used F' input-binding and recursive-counter
  helpers binds both incoming field counters to canonical source-image words,
  fixes rows_in_chunk to 7, and enforces the integer no-wrap equations
  chunk_out = chunk_in + 1 and step_out = step_in + 7.
assumptions:
  - EuclidPrime goldilocksP, passed as a typed hypothesis.
  - Assignment values are canonical residues and column zero is one.
  - The deterministic Rust-to-Lean sparse-row emitter is trusted.
non_goals:
  - NIFS.V, transcript, accumulator, application, and x_out constraints around
    this block.
  - Completeness of the entire F' witness generator.
  - Cryptographic Ajtai or Poseidon2 security.
paper_sources:
  - HyperNova Construction 2 counter advance.
rust_surfaces:
  - crates/neo-fold-clean/src/paper/f_prime/r1cs.rs
    (enforce_f_prime_counter_input_binding,
     enforce_f_prime_recursive_counter_transition)
  - crates/neo-fold-clean/src/paper/f_prime/source_image_circuit.rs
  - crates/neo-fold-clean/src/engine/r1cs_circuit/u64.rs
circuit_or_encoding_artifacts:
  - Nightstream/Implementation/R1CS/Artifacts/FPrime/Generated/FPrimeCounterArtifact.lean
    (schema 1, payload sha256:e49966c230a36a76ff2f98ca4b4d52de7ccdb97d947aa6b32797176dab4e1ad7)
failure_class:
  A source-image disconnect, forged claimed counter, forged batch cardinality,
  noncanonical field encoding, or wrapping carry chain is accepted.
counterexample_or_witness:
  - wrongSourceWitness fails first at row 132.
  - wrongStepWitness fails first at row 139.
  - wrongRowsWitness fails first at row 400.
  Rust and Lean independently pin all three first-failure positions.
lean_theorems:
  - Nightstream.Implementation.R1CS.satisfies_pull_of_rowsIncluded
  - Nightstream.Implementation.R1CS.FPrimeCounterSound.fPrimeCounter_sound
axiom_report:
  [propext, Quot.sound], guarded fail-closed in tests/Axioms.lean.
proof_hash:
  sha256:66799d4cbdfb0b055a16de1a59f96b62b28cccccf662d317126f946725d3d204
conformance_status:
  artifact-checked. Production calls the same two helpers as the exporter;
  byte drift, layout drift, honest behavior, and three rejection paths fail
  the Rust test until the checked Lean artifact is deliberately regenerated.
retest_commands:
  - cargo test -p neo-fold-clean --release --test gadgets_f_prime_counter_lean_artifact
  - cd formal/nightstream-lean && lake build && lake exe check
```
