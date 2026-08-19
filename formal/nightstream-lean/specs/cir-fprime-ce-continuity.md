# CIR-FPR-CE-CONTINUITY

```text
property_id: CIR-FPR-CE-CONTINUITY
claim:
  For the schema-1 one-claim profile, every canonical assignment satisfying
  the exact continuity rows has equality of all 1,297 coordinates in the
  prior PiDEC child and next PiCCS running views. The full artifact also
  contains six verifier-owned metadata pins.
assumptions:
  - Canonical assignment representatives and constant-one column zero.
  - The one-claim production fixture geometry recorded by the artifact.
non_goals:
  - Correctness of PiDEC or PiCCS internally.
  - Multi-claim and Nebula-adv layouts, which require separately hashed
    manifests and pointwise application of the same theorem pattern.
paper_sources:
  - HyperNova/SuperNeo accumulator continuity between consecutive NIFS folds.
rust_surfaces:
  - crates/neo-fold-clean/src/engine/decider.rs
    (enforce_children_equal_running)
  - crates/neo-fold-clean/src/engine/decider_test_isolation.rs
    (enforce_ce_continuity_against_self)
circuit_or_encoding_artifacts:
  - FPrimeCeContinuityArtifact.lean, schema 1, 1,303 full rows, 2,596
    columns, and 1,297 theorem-owned continuity rows.
failure_class:
  A prior child and next running accumulator agree only in a compact digest,
  while an omitted commitment, projection, evaluation, range, sidecar, or
  fold-digest coordinate changes.
counterexample_or_witness:
  Rust mutates 15 independent coordinate families. Lean proves all 1,297
  equalities for every satisfying assignment.
lean_theorems:
  - Nightstream.Implementation.R1CS.FPrimeCeContinuitySound.fPrimeCeContinuity_sound
axiom_report:
  Uses [propext, Quot.sound], guarded fail-closed in tests/Axioms.lean.
proof_hash:
  Full row artifact 28f0dab3afbac6183eb0804330ff1c23ee2d897dfe1a68fe4715a7e7bf16435b.
  Witness artifact 11964e484703fe0029d1fa15cd83cc1b5c1af00c62f1b91ba69ea74ad8898f77.
conformance_status:
  Artifact-checked for the one-claim plain profile. Whole CIR-SOUND remains
  open on internal step/NIFS/hash/application correspondence and other fixed
  layouts.
retest_commands:
  - cd formal/nightstream-lean && lake build && lake exe check
  - cargo test -p neo-fold-clean --release --test gadgets_f_prime_ce_continuity_lean_artifact
  - cargo test -p neo-fold-clean --release --test system_decider_r1cs
```
