# FOLD-PICCS-SPLIT-RESIDUAL-ALIGNMENT

```text
property_id: FOLD-PICCS-SPLIT-RESIDUAL-ALIGNMENT
claim:
  The original slot-for-slot target is rejected by a kernel-checked
  obstruction. SuperNeo Section 7.3 owns one row-domain polynomial:

      CCS exponents:       0 .. K-1
      norm exponents:      K .. 2K+k-1
      carried exponents:   2K+k+I(running,matrix,coefficient).

  Production instead owns two product-domain SumChecks. FE serializes
  row-then-lane and represents the carried coefficient coordinate on the
  lane axis. NC serializes block-then-lane and uses the paper's relative
  norm weights inside its independent zero claim.

  Consequently, literal Boolean-coordinate and gamma-slot identity is not
  the refinement theorem. The soundness-relevant theorem is semantic:

      split uncompressed residuals vanish
        iff the Section-7.3 CCS/norm/carried relation holds,

  and accepted production verification yields that paper relation or an
  exact FeFailure/NcFailure.
specification_change:
  The earlier positive literal-slot target is removed under the charter rule
  permitting a specification correction after a proved obstruction.
  not_literalResidualSlotAlignment is the cited obstruction.
exact_obstruction:
  At K=1, k=1, t=2 and the production six-bit Phi81 lane domain:

  - production FE has 1+6=7 coordinates; the paper row cube has 1;
  - one production carried gamma group has total exponent 5;
  - paper coefficient-zero and coefficient-one slots have exponents 4 and 6;
  - both paper coefficient slots are owned by that one production group and
    distinguished by the lane coordinate instead;
  - the first production-relative NC exponent is 0 while its absolute
    paper joint-Q norm exponent is 1.
positive_correspondence:
  - ncRelativeExponent_eq_paperLocal proves exact relative NC alignment.
  - ncJointExponent_eq_paperNormSlot proves the explicit outer K shift.
  - semanticResidualsZero_iff_paperHolds proves soundness and completeness
    of the uncompressed split residual family for the paper relation.
  - accepted_implies_paper_or_residual_failure preserves every exact FE/NC
    algebraic event rather than assuming slot identity.
assumptions:
  - NormRange.BaseFieldNoZeroDivisors for the norm-residual soundness
    direction, owned for concrete discharge by ARITH-GOLDILOCKS-FIELD.
non_goals:
  - Identity between the production FE/NC messages and the displayed paper
    one-joint SumCheck.
  - Eliminating or probability-bounding FE/NC bad events.
  - Fiat-Shamir, Poseidon2, concrete field certificates, Rust, R1CS, IR,
    encoding, physical rows, columns, or costs.
paper_sources:
  - docs/superneo-paper/07-7-neo-s-folding-scheme-for-ccs.md, Section 7.3.
  - docs/superneo-paper/13-d-deferred-theorems-and-proofs.md, Appendix D.4.
lean_theorems:
  - ProductionRefinement.ResidualAlignment.not_literalResidualSlotAlignment
  - ProductionRefinement.ResidualAlignment.carriedCoefficientAxis_is_not_gammaAxis
  - ProductionRefinement.ResidualAlignment.ncRelativeExponent_eq_paperLocal
  - ProductionRefinement.ResidualAlignment.ncJointExponent_eq_paperNormSlot
  - ProductionRefinement.ResidualAlignment.semanticResidualsZero_iff_paperHolds
  - ProductionRefinement.ResidualAlignment.accepted_implies_paper_or_residual_failure
axiom_report:
  The obstruction uses [propext, Quot.sound]. The accepted-production
  headline uses [propext, Classical.choice, Quot.sound]. No guard admits
  sorryAx or Lean.trustCompiler.
conformance_status:
  model-proved registered-deviation refinement. Literal slot identity is
  precisely obstructed; semantic reduction to the paper relation is proved.
retest_commands:
  - cd formal/nightstream-lean && LEAN_TIMEOUT_SECONDS=900
      LEAN_BUILD_TARGET=tests.PiCcsSplitNcResidualAlignment
      ./scripts/validate.sh build
  - cd formal/nightstream-lean && LEAN_TIMEOUT_SECONDS=900
      LEAN_BUILD_TARGET=tests.Axioms.PiCcsSplitNcResidualAlignment
      ./scripts/validate.sh build
  - cd formal/nightstream-lean && LEAN_TIMEOUT_SECONDS=900
      LEAN_BUILD_TARGET=tests.Axioms.FPrimeFrozenProductionDeviations
      ./scripts/validate.sh build
```
