# FOLD-PICCS / FOLD-PIRLC / FOLD-PIDEC / FOLD-COMPOSE

```text
property_ids: FOLD-PICCS, FOLD-PIRLC, FOLD-PIDEC, FOLD-COMPOSE
claim:
  PiCCS maps the structurally exact product CCS(b)^K x CE(b)^k to K+k fresh
  CE(b) openings and preserves the commitment projection phi. Given an ambient
  CE(q/2) output witness, its independent SumCheck arithmetization recovers the
  original fresh relation or exposes a bounded bad challenge.

  PiRLC applies one shared verifier challenge family to commitment, public
  input, witness, and every evaluation claim. Honest K+k fresh openings combine
  to CE(B). Its security interface is intentionally weak: extraction returns
  CE(q/2) openings for the input tuple or an actual sampling-failure proof, and
  two successful extractions at the same commitment projection are equal or
  produce a literal Definition-4 (2B,C)-relaxed-binding collision. There is no
  standalone PiRLC knowledge theorem.

  PiDEC splits CE(B) into exactly k fresh CE(b) children, proves exact witness,
  commitment, public-input, and evaluation recomposition, and reconstructs a
  valid parent opening from valid children and accepted public equations.

  The composed theorem starts from valid final PiDEC children, reconstructs the
  PiRLC parent, runs two weak extractions, uses their shared-phi uniqueness, and
  returns witnesses for every original PiCCS source or one named event:
  SumCheck challenge collision, Appendix-D.5 sampling failure, or literal
  relaxed-binding collision. The bad-event proposition is not vacuously
  inhabited; tests prove it false in a lawful concrete toy model and force the
  knowledge branch.
assumptions:
  - PiRLC.Algebra supplies commitment/projection/evaluation homomorphism and the
    Definition-14 norm-growth law under the verifier-owned arity cap.
  - PiDEC.Algebra supplies exact split/recompose and its homomorphism/norm laws.
  - WeakExtractor is the explicit Appendix-D.5 rewinding boundary. Its failed
    branch carries evidence of a concrete SamplingBoundary.Failure proposition.
  - UniquenessBridge is the explicit D.5 construction from distinct ambient
    extractions at one phi to a literal (2B,C)-relaxed-binding collision.
  - rewindArithmetization connects equal rewound witnesses to the independent
    SumCheck truth path and fresh-norm/source-payload claim.
non_goals:
  - Mechanization of expected-polynomial-time rewinding, probability measures,
    strong-sampling-set bounds, Module-SIS hardness, or the union bound.
  - Concrete proofs that the production optimized ring mixers instantiate all
    PiRLC/PiDEC algebra laws; those are the next refinement layer.
  - Fiat-Shamir, transcript, sidecar/digest, circuit, serialization, terminal
    decider, or Rust control-flow refinement.
paper_sources:
  - SuperNeo Definitions 9 and 10, Theorem 6, Lemmas 3 and 4, Theorem 7.
  - SuperNeo sections 7.3-7.5 and Appendices D.4-D.6.
rust_surfaces:
  - crates/neo-fold-clean/src/paper/reductions/pi_ccs.rs
  - crates/neo-fold-clean/src/paper/reductions/pi_rlc.rs
  - crates/neo-fold-clean/src/paper/reductions/pi_dec.rs
  - crates/neo-fold-clean/src/paper/nifs/verifier.rs
circuit_or_encoding_artifacts:
  - none; these properties are model-proved, not artifact-checked.
failure_class:
  A fold changes authority-bearing coordinates under different coefficients,
  loses the fresh/combined/ambient norm distinction, treats weak PiRLC as a
  standalone knowledge reduction, skips exact decomposition recomposition, or
  hides a cryptographic failure inside accepted-implies-valid.
counterexample_or_witness:
  tests/Folding.lean includes a false CCS payload with an accepted colliding
  SumCheck, rejects PiRLC and PiDEC stage mutations, exercises exact product
  arity, and instantiates the full theorem in a lawful model where every named
  bad event is proved false and the knowledge branch is therefore mandatory.
lean_theorems:
  - Nightstream.SuperNeo.Folding.PiCCS.product_complete
  - Nightstream.SuperNeo.Folding.PiCCS.strong_extract_or_bad_challenge
  - Nightstream.SuperNeo.Folding.PiCCS.repeated_outputs_same_phi
  - Nightstream.SuperNeo.Folding.PiRLC.complete
  - Nightstream.SuperNeo.Folding.PiRLC.same_phi_extractions_unique_or_collision
  - Nightstream.SuperNeo.Folding.PiDEC.complete
  - Nightstream.SuperNeo.Folding.PiDEC.split_recompose_exact
  - Nightstream.SuperNeo.Folding.PiDEC.reduce_knowledge
  - Nightstream.SuperNeo.Folding.Composition.shared_phi
  - Nightstream.SuperNeo.Folding.Composition.fold_knowledge_or_bad_event
axiom_report:
  Product completeness and PiDEC completeness use no axioms. PiRLC completeness
  and PiDEC knowledge reduction use only Quot.sound. The extraction/composition
  theorems use [propext, Classical.choice, Quot.sound]. All central reports are
  guarded fail-closed in tests/Axioms.lean; no sorryAx is present.
proof_hash:
  PiCCS sha256:88eb58d0a3d46001cb046a8f77effef05851f27830809c781b3465971b5ec4e6
  PiRLC sha256:a1b4379132a11aecb74b869970be336b5d62dd63245e865b7e9a417b1a96b190
  PiDEC sha256:deccb031dd2fde05fce2a7bcf0d1d6068285739da5482259beeb8ff6c801dc35
  composition sha256:3ec7cd7e20c75c53fc7f630c4b5ecfef6bf0fd5820523d958999c57bd73ee255
conformance_status:
  model-proved. Rust symbol anchors cover the live PiCCS -> PiRLC -> PiDEC
  verifier order and key shape guards, but no rust-conformant claim is made.
retest_commands:
  - cd formal/nightstream-lean && lake build && lake exe check
```
