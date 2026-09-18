# CIR-FPR-BASE-PROGRAM

```text
property_id: CIR-FPR-BASE-PROGRAM
claim:
  The complete 12,498-row production plain F' base-step artifact is exactly a
  checked program with 10,900 deterministic SSA definitions and 1,598 retained
  verifier assertions. Every satisfying canonical assignment agrees with the
  executable interpreter and makes every assertion true; equal program inputs
  force equal x_out lanes; and any canonical input satisfying the assertions
  yields a satisfying exact-row witness.
assumptions:
  - Canonical Goldilocks representatives and constant-one column zero.
  - Schema-1 fixed profile: plain/stateless F', rows_in_chunk=3, m_in=257,
    seeded preprocessing fixture 42, and the current production parameters.
  - Native-decide validates the 12,498-entry structural classification.
non_goals:
  - Claiming the extracted assertion predicate already refines every field of
    Step.BaseLocalHolds. That decoding theorem remains part of CIR-SOUND.
  - Base authority pins owned by the full-history builder; those are covered
    separately by CIR-FPR-BASE-PINS.
  - Nebula, stateful application suffixes, or other fixed profiles.
paper_sources:
  - HyperNova Construction 2 base branch.
rust_surfaces:
  - crates/neo-fold-clean/src/paper/f_prime/r1cs.rs
    (enforce_f_prime_base_step_circuit)
  - crates/neo-fold-clean/tests/gadgets/f_prime_base_program_lean_artifact.rs
circuit_or_encoding_artifacts:
  - FPrimeBaseProgramArtifact.lean plus eleven generated instruction shards,
    schema 1, 12,498 rows, 12,041 columns.
failure_class:
  A row is omitted, an assertion is incorrectly treated as a derived value,
  one input admits two x_out values, or valid checked execution cannot produce
  a satisfying witness.
counterexample_or_witness:
  Rust checks the honest full witness and rejects both a noninitial input
  counter and a self-consistently rebuilt forged chunk digest.
lean_theorems:
  - Nightstream.Implementation.R1CS.CheckedProgram.sound
  - Nightstream.Implementation.R1CS.CheckedProgram.complete
  - Nightstream.Implementation.R1CS.FPrimeBaseProgramSound.fPrimeBaseProgram_sound
  - Nightstream.Implementation.R1CS.FPrimeBaseProgramSound.fPrimeBaseProgram_xOut_unique
  - Nightstream.Implementation.R1CS.FPrimeBaseProgramSound.fPrimeBaseProgram_complete
axiom_report:
  Uses [propext, Lean.ofReduceBool, Lean.trustCompiler, Quot.sound], guarded
  fail-closed in tests/Axioms.lean. The compiler axioms enter only through the
  generated structural certificate, not the generic checked-program theorem.
proof_hash:
  Full row artifact a88ddfea5c2e8c806dc00bef350e806548fa658a76a6829bc0a99281325ffe01.
  Witness artifact fd957da44004143a118fb005ed366fd05d6ce85c47607f6f2b42c56c77a0dcb1.
conformance_status:
  Artifact-checked for exact checked-program functionality, x_out uniqueness,
  and completeness in this fixed profile. High-level BaseLocalHolds decoding
  and supported-profile generalization remain open CIR-SOUND work.
retest_commands:
  - cd formal/nightstream-lean && lake build && lake exe check
  - cargo test -p neo-fold-clean --release --test gadgets_f_prime_base_program_lean_artifact
  - cargo test -p neo-fold-clean --release --test f_prime_r1cs
```
