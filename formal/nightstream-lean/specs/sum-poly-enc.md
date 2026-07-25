# SUM-POLY-ENC

```text
property_id: SUM-POLY-ENC
claim:
  For the paper-joint Pi_CCS verifier, every successfully decoded raw round
  message has exactly the verifier-owned paper degree width. Any repository
  SumCheckFailure identifies one causal round whose claimed polynomial was
  fixed before that round's challenge, differs from the independently derived
  semantic polynomial, and collides at the sampled challenge. Finite root
  counting and the recursive Cartesian challenge-word law bound the union of
  all such rounds by

      cubeVariables * sumcheckWidth / alphabet.cardinality.

  Averaging over alpha, gamma, prover tapes, and the irrelevant post-prefix
  target seed does not multiply that loss. Therefore the existing
  SumCheckSoundnessContract is constructed for the literal operational
  sumCheckBadChallengeEvent and is not retained as a theorem premise.
assumptions:
  - PaperDegreeWidthExact context.
  - The paper field operations satisfy InterpolationEvaluationLaws.
  - Multiplication has no zero divisors.
  - The finite challenge support is nonempty and duplicate-free, as enforced
    by Support, and context.challengeSetSize equals its cardinality.
  - For the extraction corollary only: the separate alpha/gamma mixing
    probability contract, success floor, and raw repeated-witness mismatch
    bound required by Appendix D.4.
non_goals:
  - Fiat-Shamir or any random-oracle theorem.
  - The separate alpha/gamma Schwartz-Zippel bound.
  - A concrete Goldilocks-extension instantiation of the field laws.
  - Rust, transcript, Poseidon2, Ajtai, R1CS, generated-artifact, IR, cost, or
    minimality refinement.
  - Production two-SumCheck FE/NC refinement.
paper_sources:
  - docs/superneo-paper/04-4-preliminaries.md:85, Definition 6 and its
    ell*d/|K| SumCheck soundness bound.
  - docs/superneo-paper/07-7-neo-s-folding-scheme-for-ccs.md:58-80, verifier
    challenge order and Pi_CCS SumCheck.
  - docs/superneo-paper/13-d-deferred-theorems-and-proofs.md:80-175 and
    256, Appendix D.4 public-coin ordering, degree ceiling, and union bound.
rust_surfaces:
  - none. This property is paper/model-level.
circuit_or_encoding_artifacts:
  - none. No row artifact or encoding is selected or certified here.
failure_class:
  A decoded fixed-width prover message contains a nonzero coefficient above
  the paper degree, or a current message depends on its current/future
  challenge, allowing a false SumCheck statement to collide with probability
  above the paper budget.
counterexample_or_witness:
  Necessity.SumCheckSoundnessContract retains the degree-six polynomial that
  vanishes on all six sampled challenges. SUM-DEGREE-WIDTH proves that its
  loose width-six context is not PaperDegreeWidthExact, so it cannot enter the
  positive theorem. The positive theorem then counts roots of every eligible
  fixed-width difference polynomial and transports the exact submitted raw
  certificate rather than substituting a new certificate.
lean_theorems:
  - FiniteRootCounting.roots_count_le_degree
  - FiniteRootCounting.collisions_count_le_degree
  - CausalSumCheckBound.detectsFrom_count_le
  - CausalSumCheckBound.probability_detects_le_ratio
  - StrongExecution.execute_history_eq_replayPrefix
  - FixedPhase.badChallenge_implies_causal_decomposition
  - SumCheckSoundness.sumCheckFailure_implies_detects
  - SumCheckSoundness.verifierDetects_probability_le
  - SecurityContracts.sumCheckBadChallengeEvent_eq_true_iff
  - SumCheckSoundness.sumCheckBadChallenge_probability_le
  - SumCheckSoundness.sumCheckSoundnessContract_of_rootCounting
  - SumCheckSoundness.extraction_after_first_success_of_rootCounting
axiom_report:
  Every listed headline theorem is guarded fail-closed in
  tests/Axioms/PiCcsPaperJointSumCheckSoundness.lean. The recorded dependency
  set is exactly [propext, Classical.choice, Quot.sound]; there is no sorryAx,
  new axiom, unsafe declaration, or Lean.trustCompiler dependency.
proof_hash:
  Filled from the final task-owned Lean sources in the evidence ledger entry.
conformance_status:
  model-proved. The theorem constructs the repository's finite operational
  contract, but is not a concrete field, Fiat-Shamir, Rust, or R1CS
  refinement.
retest_commands:
  - cd formal/nightstream-lean &&
      ./scripts/validate.sh bounded lake build
        Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution.SumCheckSoundness
        tests.PiCcsPaperJointSumCheckSoundness
        tests.Axioms.PiCcsPaperJointSumCheckSoundness
  - cd formal/nightstream-lean && LEAN_TIMEOUT_SECONDS=900
      ./scripts/validate.sh static
```
