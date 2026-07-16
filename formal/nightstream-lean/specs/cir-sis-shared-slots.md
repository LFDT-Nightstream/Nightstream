# CIR-SIS-SHARED-SLOTS — reduced production opening correspondence

```text
property_id: CIR-SIS-SHARED-SLOTS
claim:
  For the exact isolated one-field gadget-native fixture, production
  acceptance consists of 20 residual-pair rows, one ordinary centered tail,
  and the actual 82 retained single-product rows read from the CCS matrices.
  These 103 physical rows encode 123 logical retained obligations. Under the
  explicit ProjectiveSevenNonresidue premise and verifier-fixed constant-one
  convention, they accept exactly when the decoded structural shared-slot
  assignment satisfies all 124 canonical shifted-ternary source rows.

  The generated artifact pins the production polynomial arity and its exact
  `(d^3-d)^2 - 7(e^3-e)^2`, `d^3-d`, and `A*B-C`
  specializations, the ordered LEFT/RIGHT roles, source and target columns,
  field coefficients, retained and omitted row indices, centered pair/tail
  schedule, pre-CSC expanded rows, and actual post-CSC retained product rows.

production_tree:
  shifted_ternary:
    physical_rows:
      centered_residual_pairs: 20
      centered_ordinary_tail: 1
      retained_products: 82
      total: 103
    logical_retained_obligations:
      centered_unit: 41
      negative_definition: 41
      borrow_transition: 41
      total: 123
    omitted_as_proved_consequences:
      negative_bitness: 41
      internal_borrow_bitness: 40
      negative_support: 41
      reconstruction: 1

authority_boundary:
  The field has no independent target witness. It is decoded from the same
  41 digit slots used by the centered gates. This structural alias, the
  exact source-row validator, and the exact retained production rows are the
  authority; no digest or self-consistent witness is accepted as authority.

assumptions:
  - The generated fixture is the exact isolated one-field harness only.
  - Goldilocks primality and canonical decoded residues.
  - The verifier fixes the encoded constant-one column to one.
  - `ProjectiveSevenNonresidue`: over the Nightstream Goldilocks residue
    carrier, `a^2 - 7*b^2 = 0` implies `a = 0` and `b = 0`.

non_goals:
  - Full fixed-F' conformance or a manifest of every production opening.
  - Honest shared-slot witness materialization for arbitrary callers.
  - SIS commitment binding or a cryptographic security reduction.
  - A proved refinement from SuperNeo's extension-field nonresidue theorem to
    Nightstream's `Fin goldilocksP` carrier. Until that bridge is imported,
    the physical-row theorem remains conditional.

paper_sources:
  - docs/superneo-paper/07-7-neo-s-folding-scheme-for-ccs.md (the committed
    witness relation that this encoding must preserve; the paper does not
    specify this residual-pair lowering)

rust_surfaces:
  - frontends/f_prime/gadget_native/balanced_ternary.rs
  - frontends/f_prime/gadget_native/shared_slots.rs
  - frontends/f_prime/gadget_native.rs
  - tests/gadgets/shifted_ternary_lean_artifact.rs
  - tests/f_prime/low_norm_r1cs.rs

circuit_or_encoding_artifacts:
  - Nightstream/Implementation/R1CS/Artifacts/ShiftedTernary/Generated/
    ShiftedTernarySharedSlotsArtifact.lean (schema 3; exact roles, polynomial,
    20-pair/one-tail schedule, and 82 product rows)

failure_class:
  A swapped LEFT/RIGHT matrix role, cross-family pairing, reordered/duplicated
  coordinate, missing odd tail, or use of the pair equation without the
  projective-seven premise can make the physical row claim diverge from the
  41 centered obligations.

counterexample_or_witness:
  - The focused Rust test swaps matrix roles 46 and 47 while retaining the
    row's coordinate audit; `balanced_ternary_rows` must reject it.
  - Pair-side and odd-tail value mutations make the materialized CCS
    assignment unsatisfied.
  - Generator assertions reject duplicate/missing coordinates and schedule
    drift; the Lean census proves pair/tail/product row-ID disjointness, while
    the live Rust reader rejects incompatible family/role replay.

lean_theorems:
  - ShiftedTernarySharedSlots.production_gate_polynomial
  - ShiftedTernarySharedSlots.production_census
  - ShiftedTernarySharedSlots.artifactGateAccepts_iff_productionAccepts
  - ShiftedTernarySharedSlots.artifactGateAccepts_iff_canonicalRows
  - ShiftedTernarySharedSlots.production_decoded_sharedAlias
  - ShiftedTernarySharedSlots.productionAccepts_iff_canonicalRows
  - ShiftedTernarySharedSlots.production_complete
  - ShiftedTernaryCenteredZero.centered_zero_unique

axiom_report:
  `artifactGateAccepts_iff_canonicalRows` depends on [propext,
  Classical.choice, Lean.trustCompiler, Quot.sound], guarded fail-closed in
  tests/Axioms/Implementation.lean. `ProjectiveSevenNonresidue` is an explicit
  theorem argument, not an axiom or typeclass instance.

proof_hash:
  pending final focused artifact-drift and Rust tamper gates

artifact:
  Nightstream/Implementation/R1CS/Artifacts/ShiftedTernary/Generated/
    ShiftedTernarySharedSlotsArtifact.lean

evidence_state: artifact-checked for the exact isolated fixture
qualification: the physical-row equivalence is conditional on
  ProjectiveSevenNonresidue until the Nightstream carrier bridge is proved
route: theorem-first, then generated production conformance
conformance_status:
  Artifact schema, exact matrix-role replay, and Lean physical-row theorem are
  mapped. Final schema-3 evidence remains pending until the focused release
  Rust drift/tamper gates pass; this property must not be called
  rust-conformant while the projective-seven carrier bridge is open.
retest_commands:
  - timeout 300s cargo test -p neo-fold-clean --release --test
    gadgets_shifted_ternary_lean_artifact -- --nocapture
  - timeout 300s cargo test -p neo-fold-clean --release --test
    f_prime_low_norm_r1cs coordinate_gates -- --nocapture
  - cd formal/nightstream-lean && timeout 900s lake build
    Nightstream.Implementation.R1CS.Correspondence.ShiftedTernary.SharedSlots
  - cd formal/nightstream-lean && timeout 900s lake env lean
    tests/ShiftedTernary.lean
  - cd formal/nightstream-lean && timeout 900s lake env lean
    tests/Axioms/Implementation.lean
```
