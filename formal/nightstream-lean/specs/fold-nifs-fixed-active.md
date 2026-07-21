# FOLD-NIFS — fixed-active paper profile

```text
property_id: FOLD-NIFS

claim:
  The production-selected SuperNeo edge has K = 1, k = 14, and b = 2. Its
  public input is one fresh CCS statement plus fourteen running CE statements;
  PiCCS produces fifteen internal CE statements, PiRLC computes one internal
  combined parent, and the public output is fourteen CE children accepted by
  the exact PiDEC recomposition verifier.

  Deterministic split_b children are an honest-completeness constructor, not
  the accepted-output relation. The concrete Phi81 realization refines the
  parameterized relation. The existing richer ResultTransition equals a paper
  realization plus polynomial-input binding, incoming-parent authority,
  outgoing-parent materialization, and a separate canonical-child
  strengthening.

assumptions:
  - The parameterized theorem receives RelationSemantics and PiRLC/PiDEC
    algebras satisfying their published laws.
  - Conditional honest completeness receives source openings, a common
    structure and prior point, one valid new point, and strong-set challenges.
  - The concrete theorem uses the independently defined Phi81 relation,
    source binding, and Phi81 PiRLC/PiDEC algebras.
  - Relating target openings to the source-derived combined opening uses the
    explicit PiDEC parent-opening binding-collision alternative.

non_goals:
  - Fiat-Shamir or sampler provenance and probability bounds.
  - Split-NC executable soundness, extraction, or the disputed single-Q
    message-flow equivalence.
  - HyperNova lifecycle, Rust, R1CS, costs, necessity, or row removal.
  - Claiming deterministic private child splitting is required for soundness.

paper_sources:
  - SuperNeo Sections 7.3, 7.4, and 7.5.
  - HyperNova Definition 12 and Construction 2.

rust_surfaces:
  - none; this checkpoint is model-level.

circuit_or_encoding_artifacts:
  - none.

failure_class:
  A semantic target copies the honest prover's non-unique private radix split
  into verifier acceptance, thereby retaining unnecessary canonical-child
  constraints or excluding a sound accepted decomposition.

counterexample_or_witness:
  PiDEC.Necessity.ProductionChildSubstitution.rightAccepted_but_notCanonical
  gives two distinct valid strict-2 fourteen-child families with the identical
  recomposed assignment and parent; strict public PiDEC accepts both.

lean_theorems:
  - Nifs.PaperProfile.complete
  - Nifs.PaperProfile.Realization.parentOpening
  - Nifs.PaperProfile.Realization.outputAccepted
  - Nifs.PaperProfile.Realization.parentOpening_eq_recompose_or_bindingCollision
  - Nifs.ConcretePhi81.FixedActive.PaperProfile.Realization.toGeneric
  - Nifs.ConcretePhi81.FixedActive.PaperProfile.complete
  - Nifs.ConcretePhi81.FixedActive.resultTransition_iff_exists_paperDecomposition
  - Nifs.ConcretePhi81.FixedActive.ResultTransition.toPaperProfile

axiom_report:
  The profile counts use no axioms. The guarded theorem union is exactly
  [propext, Classical.choice, Quot.sound]. No compiler-trusted decision
  procedure or project-added axiom is used.

proof_hash:
  Recorded after the final focused gates in assurance/evidence-ledger.jsonl.

conformance_status:
  model-proved for the fixed-active relation profile, concrete Phi81
  instantiation, and rich semantic ownership decomposition. Transcript,
  security, Rust/R1CS, cost, necessity, and row-removal refinement remain open.

retest_commands:
  - cd formal/nightstream-lean && LEAN_BUILD_TARGET=Nightstream.SuperNeo.Folding.Nifs.PaperProfile ./scripts/validate.sh build
  - cd formal/nightstream-lean && LEAN_BUILD_TARGET=Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.FixedActive ./scripts/validate.sh build
  - cd formal/nightstream-lean && LEAN_BUILD_TARGET=tests.NifsConcretePhi81PaperProfile ./scripts/validate.sh build
  - cd formal/nightstream-lean && LEAN_BUILD_TARGET=tests.Axioms.NifsConcretePhi81PaperProfile ./scripts/validate.sh build
```
