# REL-CCS / REL-CE / REL-CONCRETE — concrete relation semantics

```text
property_ids: REL-CCS, REL-CE, REL-CONCRETE
claim:
  The generic Definition-12 CCS and Definition-13 CE predicates instantiate
  with canonical Goldilocks residues, the quadratic extension X^2-7, the
  cyclotomic quotient X^54+X^27+1, centered norm, column-major ring packing,
  an explicit Ajtai matrix action, prefix projection, row-wise CCS polynomial
  evaluation, and coefficient-packed multilinear evaluation. Membership
  expands exactly to commitment, projection, norm, structure/point shape, and
  relation truth. Canonical statements satisfy the relation; mutations of
  commitment, public input, point dimension, or evaluations are rejected.
assumptions:
  - The production modulus is represented by Fin q; no local primality theorem
    is required by these addition/multiplication-only relation definitions.
  - The Ajtai key and CCS structure are verifier-owned configuration.
non_goals:
  - Cryptographic binding of the Ajtai action.
  - Refinement to optimized Rust ring, sparse-matrix, MLE, or serialization
    code; that evidence is required before rust-conformant status.
  - Sumcheck, folding, extraction, or Fiat-Shamir soundness.
paper_sources:
  - SuperNeo Definitions 11, 12, 13, and 14.
rust_surfaces:
  - crates/neo-fold-clean/src/paper/relations/instance.rs
  - crates/neo-ccs/src/relations.rs
  - crates/neo-math/src/ring.rs
circuit_or_encoding_artifacts:
  - none; this milestone is model-proved, not artifact-checked.
failure_class:
  A relation accepts a self-consistent claim without an authoritative opening,
  uses the wrong public prefix or norm, omits CCS truth, accepts a malformed CE
  point domain, or accepts forged matrix evaluations.
counterexample_or_witness:
  tests/ConcreteRelations.lean contains wrong commitment, wrong public input,
  invalid point length with matching computed outputs, and wrong evaluation
  array witnesses. The invalid-point witness exposed and closed a missing
  obligation in the generic CE predicate.
lean_theorems:
  - Nightstream.SuperNeo.Concrete.ccsMembership_iff
  - Nightstream.SuperNeo.Concrete.ceMembership_iff
  - Nightstream.SuperNeo.Concrete.canonicalCCS_holds
  - Nightstream.SuperNeo.Concrete.canonicalCE_holds
  - Nightstream.SuperNeo.Concrete.ccs_rejects_wrong_commitment
  - Nightstream.SuperNeo.Concrete.ce_rejects_invalid_point
axiom_report:
  The four membership/constructor theorems depend only on [propext], guarded
  fail-closed in tests/Axioms.lean.
proof_hash:
  relation sha256:f774a7488382643c3faf512ef7aaeb2b09fd385caec85e8730a3103a431521e6
  algebra sha256:bf77975952c8151fba4103aa1b623496f11a5d722934ac66183d7055cf123926
conformance_status:
  model-proved. No Rust refinement claim is made for the algebra or evaluator.
retest_commands:
  - cd formal/nightstream-lean && lake build && lake exe check
```
