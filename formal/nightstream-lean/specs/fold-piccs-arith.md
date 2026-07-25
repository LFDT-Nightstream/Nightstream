# FOLD-PICCS-ARITH — constructed PiCCS arithmetization

```text
property_id: FOLD-PICCS-ARITH
claim:
  The PiCCS.Arithmetization witness is constructed rather than assumed. It is a
  proof-side semantic witness relative to the extracted assignments, built from
  authoritative statements, the typed production certificate, derived
  challenges, and those extracted assignments. Its FE and NC truth paths are
  established independently from the FE and NC semantic polynomials; the
  PaperJoint layer supplies only shared source ordering, challenge-coordinate
  alignment, and the refinement of the independently established obligations
  into the paper joint relation.
assumptions:
  - Ideal-interactive challenges. Fiat-Shamir derivation is out of scope.
  - Extracted assignments are supplied by the rewind, not by the verifier.
non_goals:
  - Physical rows, columns, or generated artifacts (CIR-* under M4).
  - Fiat-Shamir or Poseidon2 transcript refinement (FOLD-NIFS-FS).
  - Any probability bound, including mixing-root and round-collision
    probabilities (BAD-BOUND).
  - The separate slot-for-slot residual theorem
    (FOLD-PICCS-SPLIT-RESIDUAL-ALIGNMENT).
  - Closed Goldilocks and extension-field arithmetic certificates
    (ARITH-GOLDILOCKS-FIELD).
paper_sources:
  - docs/superneo-paper/07-7-neo-s-folding-scheme-for-ccs.md:76-95
  - docs/superneo-paper/13-d-deferred-theorems-and-proofs.md:80-150 (D.4
    Lemma 7, which separates its three obligations by linear independence of
    powers of the *indeterminate* C, not at a sampled gamma)
```

## Exact target

`Nightstream.SuperNeo.Folding.PiCCS.Arithmetization` (`PiCCS.lean:306-328`) is a
four-field `Prop`, parameterized by `(assignments : Fin arity.total → Assignment)`:

```lean
feTruthPath             : SumCheck.TruthPath ops attempt.fe
ncTruthPath             : SumCheck.TruthPath ops attempt.nc
feClaimTrue_of_payloads : PayloadsHold semantics attempt assignments →
                            SumCheck.Claim.True attempt.fe
ncClaimTrue_of_norms    : NormsHold semantics params assignments →
                            SumCheck.Claim.True attempt.nc
```

Because its statement mentions `assignments`, it is **not** executable verifier
data and cannot be constructed from public verifier data alone. It is a
proof-side object relative to what the extractor produced.

## The soundness constraint that shapes the proof

**Do not derive `feTruthPath` or `ncTruthPath` from acceptance of, or equality
for, the single sampled joint `Q`.**

At a fixed sampled `gamma`, cancellation between the CCS, norm, and carried
blocks is *exactly* the mixing-root event. Deriving the component truth paths
from joint truth would silently eliminate `MixingRoot`, which the design names
as a distinct bad event — `PiCCS.BadChallenge`'s own comment records that it
"excludes FE/NC mixing roots, which occur before the corresponding SumCheck
polynomial is formed." D.4's Lemma 7 obtains its three-way separation by
treating `gamma` as an indeterminate `C`, not by evaluating at a sample.

Correct direction:

```text
authoritative sources + extracted assignments
        |-> canonical FE semantic polynomial -> feTruthPath
        |-> canonical NC semantic polynomial -> ncTruthPath

FE truth + NC truth  ->  paper joint obligations

sampled joint equality  -/->  separate FE/NC truth
```

## Exact generic obstruction

The obstruction is now kernel-checked:

```lean
theorem
  ArithmetizationObstruction
    .accepted_payloads_norms_ambient_without_arithmetization :
  ∃ attempt assignments,
    PiCCS.Accepted ops attempt ∧
    PiCCS.PayloadsHold semantics attempt assignments ∧
    PiCCS.NormsHold semantics params assignments ∧
    PiCCS.AmbientOutputsHold semantics params attempt assignments ∧
    ¬ PiCCS.Arithmetization semantics params ops attempt assignments
```

The countermodel closes the claimed FE chain while choosing an incompatible
`trueInitial`. It reaches the exact accepted/payload/norm/ambient interface.
It proves only that arithmetization is not automatic for every arbitrary
ghost-bearing `PiCCS.Attempt`; it does not obstruct a semantic interpretation
of the concrete production transcript.

## Implemented semantic adapter

The smallest-change proof-only adapter is implemented instead of redesigning
`PiCCS.Attempt`:

1. `FixedPhase.SemanticView.Wire` contains only the claimed initial value,
   verifier-computed terminal, challenge list, fixed-width messages, and
   support cardinality. It has no `q`, `trueInitial`, or expected-round field.
   `semanticInstance` recomputes those semantic fields after one explicit
   polynomial is fixed.
2. `SumCheck.SemanticAdapter` constructs separate FE and NC semantic views over
   the fixed wire certificates. FE uses the FE polynomial; NC uses the
   block/lane NC polynomial. Each view proves claimed-chain acceptance,
   terminal binding, its own truth path, and its own initial-claim implication.
   Neither component is derived from equality of the sampled paper joint
   polynomial.
3. `ProductTruth.SourceBridge`, `NormBridge`, and `Carried` transport extracted
   assignment facts to those two independent semantic statements. In
   particular, `carriedTruth_of_payloads` reads every public running claim
   verbatim and proves its equation; it never replaces an adversarial claim by
   an honestly recomputed one.
4. The coordinate carrier is the literal verifier order. FE uses
   `row.coordinates ++ lane.coordinates`; block/lane NC uses
   `block.coordinates ++ lane.coordinates`. The semantic instances consume the
   exact transcript-derived coordinate lists, and the terminal theorems decode
   those same lists into the corresponding protocol points.

## Production base/delayed construction

`ProductionRefinement.SemanticAttempt` reuses the FE view but selects the NC
polynomial from the transcript-owned pending-state tag:

- `pending = none`: the ordinary block/lane NC polynomial;
- `pending = some`: the actual delayed combined-NC polynomial, including the
  full pending-vector projection.

Its `arithmetization` theorem constructs the four fields of
`PiCCS.Arithmetization` for the source-aligned assignment product. The final
composition theorem does not assume its `PendingBound`: if the delayed state is
not bound, it returns the existing
`RegisteredDeviationObligation.delayedPackedYZcol` branch.

The shared physical `betaA` and `gamma` schedule is retained, but FE and NC
remain different semantic polynomials and different truth equations.
Consequently `FeFailure` and `NcFailure`, including `MixingRoot`, remain live
named branches.

## Removing `rewindArithmetization`

The generic composition theorem originally asks the caller for
`rewindArithmetization` at both extractor outputs. The concrete theorem avoids
that callback through one checked reference opening:

1. accepted production verification yields paper truth or exact FE/NC failure;
2. source and materialized-output binding prove that the authoritative source
   assignment product opens the exact PiRLC input statements;
3. the independent FE/NC construction proves arithmetization for that same
   assignment product;
4. commitment uniqueness compares the extractor's ambient opening with the
   checked reference opening;
5. equality transports arithmetization to the extracted assignments, while
   disagreement is the existing relaxed-binding collision.

Thus `CanonicalOpening.SourceInput.Carrier.data` is not assumed to be a
malicious-soundness witness merely because it is canonical. It becomes the
reference only after its source, public-input, commitment, output, and
evaluation equations have been proved. No generic `refinementFailure`,
`outputUnbound`, or replacement escape event is introduced.

The fixed-active headline is:

```lean
ProductionRefinement.SemanticComposition
  .fold_extraction_or_named_failure
```

It concludes extracted PiCCS validity or the exact union of the pre-existing
generic composition event, `FeFailure`, `NcFailure`, and
`RegisteredDeviationObligation`. It has no `rewindArithmetization`,
`SumCheckSoundnessContract`, generic mixing-soundness contract, semantic ghost,
paper-truth, or delayed-state-bound premise. The typed
`BaseFieldNoZeroDivisors` input remains deliberately owned by
`ARITH-GOLDILOCKS-FIELD`.

## Success criteria

- Concrete fixed-active composition exports no arithmetization hypothesis.
- No `SumCheckSoundnessContract`, generic mixing contract, or caller-supplied
  ghost survives in the exported statement.
- `MixingRoot` remains a live named event; nothing in this property may make it
  unreachable.
- The arbitrary-attempt obstruction, semantic views, component adapters,
  generic reference composition, production arithmetization, final theorem,
  and frozen facade export all have fail-closed axiom guards.

```text
conformance_status:
  model-proved (2026-07-24). This is an ideal-interactive, concrete fixed-active
  semantic refinement. It is not a probability, Fiat-Shamir, Rust, R1CS, IR,
  encoding, row, or concrete-field-certificate result.
retest_commands:
  - cd formal/nightstream-lean &&
      LEAN_TIMEOUT_SECONDS=900
      LEAN_BUILD_TARGET=tests.PiCcsSplitNcSemanticComposition
      ./scripts/validate.sh build
  - cd formal/nightstream-lean &&
      LEAN_TIMEOUT_SECONDS=900
      LEAN_BUILD_TARGET=tests.Axioms.PiCcsSplitNcSemanticComposition
      ./scripts/validate.sh build
  - cd formal/nightstream-lean &&
      LEAN_TIMEOUT_SECONDS=900
      LEAN_BUILD_TARGET=tests.Axioms.FPrimeFrozenProductionDeviations
      ./scripts/validate.sh build
  - cd formal/nightstream-lean && ./scripts/validate.sh axioms
```

## Remaining successor

`FPR-NIFS-BRIDGE` may now consume the fixed-active composition theorem instead
of supplying `rewindArithmetization`. The separate M2 properties for concrete
field certificates, residual slot alignment, Split-NC output authority, and
Fiat--Shamir retain their own scopes.
