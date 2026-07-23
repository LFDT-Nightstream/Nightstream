# F-prime paper deviations

This register records protocol behavior that is not literally part of the
SuperNeo or HyperNova constructions cited by the frozen F-prime specification.
A deviation is part of the specified protocol only after a kernel-checked
paper-layer theorem reduces its complete verifier transition to the cited
paper relation. Semantic rearrangement, implementation correspondence, or a
matching digest is not sufficient.

## FPR-DEV-PIRLC-AMBIENT-STRICT-HALF

- Paper boundary: SuperNeo Definition 12 and Appendix D.5.
- Literal conflict: Definition 12 uses the strict predicate
  `||z||_infinity < bound`, while Appendix D.5 claims every field element is
  covered at `bound = q / 2`. For odd `q`, the midpoint residues have centered
  magnitude exactly `floor(q / 2)` and are excluded.
- Frozen correction: retain the strict relation and use
  `floor(q / 2) + 1` for the ambient extraction relation.
- Kernel evidence:
  `PiRLC.PaperCorrections.midpointResidue_not_literalAmbientBounded` and
  `PiRLC.PaperCorrections.all_centeredMagnitude_lt_correctedAmbientBound`.
- Classification: paper-level correction. The corrected bound must be used by
  the paper-authoritative Pi_RLC weak reduction and later implementation
  refinement.

## FPR-DEV-PIRLC-COORDINATE-FORK-LOSS

- Paper boundary: SuperNeo Appendix C, Theorem 10, and Appendix D.5.
- Literal conflict: the local Theorem 10 text renders the coordinate-wise
  extraction loss as `ell / |C^ell|`, while Appendix D.5 applies the theorem
  with loss `(ell + 1) / |C|` for `ell = K + k`.
- Kernel obstruction: with `|C| = 3` and `ell = 2`,
  `RenderedBoundObstruction.rendered_denominator_bound_counterexample` proves
  that an adversary can accept exactly `3 / 9` challenges while no accepted
  coordinate fork exists. The rendered loss `2 / 9` would therefore leave a
  strictly positive claimed lower bound, disproving that denominator.
- Selected conservative correction: use Appendix D.5's numerator and
  denominator, `(ell + 1) / |C|`. The cited coordinate-fork lemma actually
  gives the sharper `ell / |C|` loss for the two-special case, so the D.5
  expression is conservative rather than tight. The model-level definition
  `PiRLC.samplingErrorNumerator ell = ell + 1` records only the numerator; it
  is not evidence for the probability bound.
- Classification: paper-level correction pending an operational
  finite-uniform coordinate-fork theorem that proves the sharper success loss,
  derives the selected D.5 loss, and proves the expected-query bound. Until
  that theorem exists, this entry does not establish the quantitative weak
  reduction or its security-reduced error.

## FPR-DEV-PICCS-FIRST-SUCCESS-CONDITIONING

- Paper boundary: SuperNeo Definition 10 and Appendix D.4, extractor Items
  2--8.
- Quantitative conflict: Appendix D.4 rejection-samples the first ambient
  success before drawing a fresh second execution. If ambient success has
  probability `p`, a raw two-run witness-disagreement probability `delta`
  becomes `delta / p` in the first-success-conditioned extractor experiment.
  The qualitative paper argument remains valid because `p` is non-negligible,
  but an exact theorem cannot subtract the unchanged raw `delta`.
- Kernel obstruction:
  `StrongConditioningObstruction.unchanged_raw_uniqueness_budget_counterexample`
  exhibits four uniform executions with ambient success `2/4`, raw
  disagreement `2/16`, and conditioned disagreement `2/8`.
- Required correction: the operational `Pi_CCS` strong game must either charge
  a conditioning-adjusted uniqueness budget using an explicit success lower
  bound, bound the first-success-conditioned disagreement event directly, or
  use a proved negligible-function calculus. It must fix the first witness
  before the fresh alpha/gamma used for SumCheck and Schwartz--Zippel.
- Classification: paper-level quantitative correction. No straight-line or
  adaptive-witness mixing-root bound may be used to discharge the strong
  reduction.

## FPR-DEV-BLOCK-LANE-COMBINED-NC

- Added behavior: the production-oriented Pi_CCS model separates FE from a
  block/lane NC representation and later combines NC obligations in its own
  verifier/transcript flow.
- Paper baseline: SuperNeo Section 7.3 uses one joint polynomial `Q` and one
  SumCheck.
- Classification: model-proved registered protocol deviation.
- Kernel evidence: `DelayedCombinedNc.expectedRound_quartic` and
  `expectedRound_has_five_coefficients` prove the exact combined round-degree
  contract; `DelayedCombinedNc.Acceptance.accepted_implies_truth_and_parentProjection_or_badEvent`
  reduces raw fixed-phase acceptance to ordinary NC truth plus the delayed
  projection identity or one named selector, residual, or SumCheck event.
  `BlockLaneCombinedNc.ProductionPiCcs.ncAccepted_implies_truth_or_badEvent`
  instantiates that result for the fixed F-prime schedule, while
  `accepted_implies_paper_and_yRingBound_or_yRingUnbound_or_badEvent` reduces
  the successful current-step result to the Section 7.3 relation with
  `y_ring` kept separate. `ProductionProjection.authoritativeRunningProjection_eq_projectedRawRecomposition`
  proves that the 64-lane block representation contains exactly the 54 active
  Phi81 lanes and ten derived-zero padding lanes.
- Executable boundary: the protocol-owned `piCcsMessageCheck` is exact to the
  public combined-NC message acceptance predicate. Raw-assignment authority is
  supplied by the delayed lifecycle theorem below, never by a child
  `y_zcol` sidecar or digest.
- Required implementation closure: differential Rust agreement, concrete
  transcript/dataflow refinement, and generated-row realization remain open.
- Excluded claims: this model proof is not Rust/R1CS conformance, generated-row
  authority, a cost result, or optimization permission.

## FPR-DEV-DELAYED-PACKED-YZCOL

- Added behavior: production carries packed `y_zcol` authority across a
  one-fold delay and closes the predecessor output in a later recursive or
  terminal step.
- Paper baseline: HyperNova Construction 2 performs one selected NIFS fold in
  each recursive `F'_j` invocation and its terminal verifier performs no final
  fold. SuperNeo's CE relation binds its current evaluation claims directly.
- Classification: model-proved registered protocol deviation.
- Executable step evidence: `DelayedPackedYZcol.Checker.check_eq_true_iff_accepted`
  proves exactness of the typed Boolean step checker. The checker replays the
  public combined-NC message, sampler, paper-output, and canonical-parent
  opening obligations; `baseCheck_eq_true_iff_accepted` additionally checks
  that the base carries no pending predecessor.
- Transcript evidence: `ChallengeAuthority.holds` fixes the statement,
  ordered parent/children, pending value, and polynomial bound before sampling;
  gives distinct producer-beta and batch-weight domains; and records their
  statement-to-core-to-producer-to-batch order. It makes no claim about the
  Poseidon2 permutation.
- Lifecycle evidence: `Trace.closedTrace_implies_baseAndAllClosed_or_failure`
  covers the explicit base, every recursive edge, and the terminal raw-child
  opening. Its successful branch proves the base pending value is absent and
  every output has both `PackedYZcolBoundAtBlock` and the independent frozen
  paper-profile transition. Its failure branch contains only located
  `yRingUnbound`, selector/root, SumCheck, Pi_RLC mixing, parent-opening, or
  accumulator-binding events. There is no generic `outputUnbound`,
  `refinementFailure`, source-projection premise, output-column-binding
  premise, or child `CeClaim.y_zcol` authority.
- One-fold boundary: a successor's accepted combined-NC proof closes exactly
  its predecessor; the final predecessor is closed by an opening over the
  fourteen ordered raw child assignments. Binding is recomputed from the full
  typed parent/child/pending payload, so a digest is compression rather than
  authority.
- Required implementation closure: differential Rust agreement and concrete
  Rust/R1CS refinement remain open. Until then this is model-proved
  composition over protocol data, not Rust-conformant production authority.
- Excluded claims: no generated-row authority, cost result, or row-removal
  permission follows from this theorem.
