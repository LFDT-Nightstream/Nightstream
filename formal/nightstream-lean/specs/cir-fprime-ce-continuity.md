# CIR-FPR-CE-CONTINUITY

```text
property_id: CIR-FPR-CE-CONTINUITY
claim:
  For the schema-1 one-claim profile, every canonical assignment satisfying
  the exact continuity rows has equality of all 1,169 retained CE-core
  coordinates in the prior PiDEC child and next PiCCS running views. The
  artifact's no-read relation omits child/running y_zcol. That theorem proves
  only what the current continuity rows read; it does not prove that the
  omission is a sound authority boundary. Parent and terminal CE validation
  retain y_zcol, and the delayed old-point projection bridge remains open.
  The full artifact also contains six verifier-owned metadata pins.
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
    (enforce_child_core_equal_running)
  - crates/neo-fold-clean/src/engine/decider_test_isolation.rs
    (enforce_ce_continuity_against_self)
circuit_or_encoding_artifacts:
  - FPrimeCeContinuityArtifact.lean, schema 1, 1,175 full rows, 2,340
    columns, and 1,169 theorem-owned continuity rows.
failure_class:
  A prior child and next running accumulator agree only in a compact digest,
  while an omitted commitment, projection, evaluation, range, sidecar, or
  fold-digest coordinate changes.
counterexample_or_witness:
  Rust differentially mutates child/running y_zcol and proves identical rows
  and witnesses. Lean proves all 1,169 retained equalities for every
  satisfying assignment.
lean_theorems:
  - Nightstream.Implementation.R1CS.FPrimeCeContinuitySound.fPrimeCeContinuity_sound
axiom_report:
  Uses [propext, Quot.sound], guarded fail-closed in tests/Axioms.lean.
proof_hash:
  Full row artifact 9395371940047222a2661ac0d5b948a27ca0b743151a01d8e74901c2c59c4506.
  Witness artifact 17f8eabc453b8543511ce5ca708226298def3c236c1d30d528bdc3acc2f4a470.
conformance_status:
  Artifact-checked for the one-claim plain profile. Whole CIR-SOUND remains
  open on internal step/NIFS/hash/application correspondence and other fixed
  layouts.
retest_commands:
  - cd formal/nightstream-lean && lake build && lake exe check
  - cargo test -p neo-fold-clean --release --test gadgets_f_prime_ce_continuity_lean_artifact
  - cargo test -p neo-fold-clean --release --test system_decider_r1cs
```
