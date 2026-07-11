# CIR-FPR-STATE-LINK

```text
property_id: CIR-FPR-STATE-LINK
claim:
  Every canonical-residue assignment satisfying the exact plain-layout state
  continuity rows has direct wire equality for all 31 adjacent F' state
  coordinates: vk/header digests, counters, z0/zi, pc, semantic and accumulator
  digests, and public trace. Digest equality is not used as authority; each
  lane is equated independently.
assumptions:
  - Canonical assignment representatives and constant-one column zero.
  - Rust-to-Lean row extraction through the existing decider isolation helper.
non_goals:
  - Nebula-enabled state layouts, which require a separate artifact.
  - Correctness of either adjacent step internally.
  - CE-claim continuity, owned by a separate decider row family.
paper_sources:
  - HyperNova Construction 2 repeated F' state threading.
rust_surfaces:
  - crates/neo-fold-clean/src/engine/decider.rs (enforce_state_link)
  - crates/neo-fold-clean/src/engine/decider_test_isolation.rs
    (enforce_state_link_against_self)
circuit_or_encoding_artifacts:
  - FPrimeStateLinkArtifact.lean, schema 1, 31 rows, 63 columns.
failure_class:
  Any adjacent state coordinate is disconnected while the rest of the trace
  remains self-consistent.
counterexample_or_witness:
  Rust probes mutate every coordinate family. The Lean twin mutates only the
  next step-count wire and first fails row 9.
lean_theorems:
  - Nightstream.Implementation.R1CS.FPrimeStateLinkSound.fPrimeStateLink_sound
axiom_report:
  Uses [propext, Quot.sound], guarded fail-closed in tests/Axioms.lean.
proof_hash:
  Recorded in assurance/evidence-ledger.jsonl after final gates.
conformance_status:
  Artifact-checked over the exact plain row program. The dedicated Rust
  drift target passes and checks all ten coordinate families.
retest_commands:
  - cd formal/nightstream-lean && lake build && lake exe check
  - cargo test -p neo-fold-clean --release --test gadgets_f_prime_state_link_lean_artifact
```
