# FOLD-PICCS-SPLIT-OUTPUT-AUTHORITY

```text
property_id: FOLD-PICCS-SPLIT-OUTPUT-AUTHORITY
claim:
  The production Split-NC prefix owns its terminal claims and downstream
  outputs. For the canonical materialization:

  - yRing is the authoritative FE evaluation;
  - packed yZcol is bound at the exact transcript-derived NC block;
  - the FE message terminal equals the FE semantic polynomial at the replayed
    row-then-lane point;
  - the ordinary or delayed NC message terminal equals the selected NC
    semantic polynomial at the replayed block-then-lane point; and
  - the complete PiRLC output product equals PiCCS.honestOutputs in
    authoritative source order.

  Accepted production verification therefore implies

      (paper relation and terminal/output authority)
        or FeFailure
        or NcFailure
        or RegisteredDeviationObligation.

  The theorem derives this partition. It does not take exclusion of an
  algebraic failure as a caller premise.
assumptions:
  - NormRange.BaseFieldNoZeroDivisors, owned for concrete discharge by
    ARITH-GOLDILOCKS-FIELD.
  - The paper field and ring laws already carried by the production semantics.
non_goals:
  - Eliminating or probability-bounding FeFailure or NcFailure.
  - Alpha/gamma mixing bounds, SumCheck root-counting bounds, Fiat-Shamir,
    Poseidon2, or random-oracle refinement.
  - Commitment binding or extraction.
  - Rust, R1CS, IR, encoding, physical rows, columns, or costs.
  - Message identity between the production split and the paper's displayed
    one-joint SumCheck.
paper_sources:
  - docs/superneo-paper/07-7-neo-s-folding-scheme-for-ccs.md, Section 7.3
    verifier terminal checks, output construction, and PiRLC handoff.
registered_deviations:
  - delayed packed y_zcol
  - block/lane combined NC
failure_class:
  An accepted certificate reaches the paper branch while its verifier-computed
  FE or NC terminal, yRing, packed yZcol, or PiRLC output product is not the
  authoritative semantic value.
lean_theorems:
  - ProductionRefinement.OutputAuthority.Holds
  - ProductionRefinement.OutputAuthority.of_paper
  - ProductionRefinement.OutputAuthority.accepted_implies_paper_and_authority_or_named_failure
axiom_report:
  The focused and frozen headline guards permit exactly propext,
  Classical.choice, and Quot.sound. There is no sorryAx, new axiom,
  Lean.trustCompiler, or unsafe declaration on the guarded path.
conformance_status:
  model-proved. This closes terminal/output authority only; residual-slot
  alignment, concrete field certificates, mixing probabilities, and
  Fiat-Shamir remain separate.
retest_commands:
  - cd formal/nightstream-lean && LEAN_TIMEOUT_SECONDS=900
      LEAN_BUILD_TARGET=tests.PiCcsSplitNcOutputAuthority
      ./scripts/validate.sh build
  - cd formal/nightstream-lean && LEAN_TIMEOUT_SECONDS=900
      LEAN_BUILD_TARGET=tests.Axioms.PiCcsSplitNcOutputAuthority
      ./scripts/validate.sh build
  - cd formal/nightstream-lean && LEAN_TIMEOUT_SECONDS=900
      LEAN_BUILD_TARGET=tests.Axioms.FPrimeFrozenProductionDeviations
      ./scripts/validate.sh build
```
