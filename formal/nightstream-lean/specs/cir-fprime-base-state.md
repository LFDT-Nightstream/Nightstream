# CIR-FPR-BASE-PINS

```text
property_id: CIR-FPR-BASE-PINS
claim:
  Every canonical-residue assignment satisfying the exact plain seeded
  base-state row family carries preprocessing-derived constants in all 31
  authority coordinates: verifier-key and PiCCS-header lanes, zero counters,
  initial/current boundary, pc=1, semantic seed, empty accumulator, and public
  trace seed.
assumptions:
  - Canonical assignment representatives and constant-one column zero.
  - The seeded direct-CCS fixture is the schema-1 artifact profile; every
    other preprocessing profile requires its own drift hash even though the
    universal pin-row theorem is value-parametric.
non_goals:
  - Internal correctness of the base F' step.
  - The four dummy x_out lanes allocated by the isolation wrapper.
  - Nebula base-lane constants, which require a separate artifact.
paper_sources:
  - HyperNova Construction 2 base state and verifier-owned parameters.
rust_surfaces:
  - crates/neo-fold-clean/src/engine/decider.rs
    (enforce_base_state_constants)
  - crates/neo-fold-clean/src/engine/decider_test_isolation.rs
    (enforce_base_state_constants_against)
circuit_or_encoding_artifacts:
  - FPrimeBaseStateArtifact.lean, schema 1, 31 rows, 36 columns.
failure_class:
  The prover selects any base authority coordinate and consistently rehashes
  descendants without encountering a verifier-owned constant row.
counterexample_or_witness:
  The Rust gate mutates one representative in every authority family and
  rejects all ten forgeries. Lean proves the conclusion for every satisfying
  assignment, not just the committed honest vector.
lean_theorems:
  - Nightstream.Implementation.R1CS.FPrimeBaseStateSound.fPrimeBaseState_sound
axiom_report:
  Uses [propext, Quot.sound], guarded fail-closed in tests/Axioms.lean.
proof_hash:
  Row artifact 62cd8b38ed65e890ddac462cb6c66a1de22a30093865fa9169e420d77be9f605.
  Witness artifact 4a7c6f6e7d43fae90beded4d4646d6982494dfcf4e5a15161cc0b59ced98bda4.
conformance_status:
  Artifact-checked for the seeded plain base-state profile. CIR-SOUND remains
  open until all producer-local and composed row owners are closed.
retest_commands:
  - cd formal/nightstream-lean && lake build && lake exe check
  - cargo test -p neo-fold-clean --release --test gadgets_f_prime_base_state_lean_artifact
```
