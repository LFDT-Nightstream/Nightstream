# SuperNeo/HyperNova Lean handoff

This is a standalone technical briefing for an AI that can read the public
repository and development branch but cannot see the local working tree. It
contains the current semantic status, unresolved proof boundaries, and the
critical newer Lean interfaces. The SuperNeo and HyperNova paper sections
listed below must be supplied separately.

No headline result below is claimed to be Rust-conformant or security-reduced.
Unless stated otherwise, a proved Lean result is **model-level**. Artifact
replay is labeled **artifact-checked** and is not a substitute for semantic
refinement.

## Repository access

- Public repository:
  [LFDT-Nightstream/Nightstream](https://github.com/LFDT-Nightstream/Nightstream).
- Development branch:
  [`nico/f-prime-constraints-cuda-formal`](https://github.com/LFDT-Nightstream/Nightstream/tree/nico/f-prime-constraints-cuda-formal).
- Public branch baseline used by this handoff:
  [`1668e929e89e6868bcb44ac41f0d20fc15a71284`](https://github.com/LFDT-Nightstream/Nightstream/commit/1668e929e89e6868bcb44ac41f0d20fc15a71284).

The branch provides the complete dependency graph and preceding proofs. Some
interfaces reproduced below are work after that public baseline; until a newer
commit is pushed, this document is the authority for that small delta.

The Lean project is under `formal/nightstream-lean`. The primary public-branch
modules are:

- `Nightstream/SuperNeo/InteractiveReduction/Paper.lean`;
- `Nightstream/SuperNeo/InteractiveReduction/StrongWeakComposition.lean`;
- `Nightstream/SuperNeo/InteractiveReduction/KnowledgeComposition.lean`;
- `Nightstream/SuperNeo/Folding/PiCCS/PaperJoint/StrongExecution/FinitePaperStrong.lean`;
- `Nightstream/SuperNeo/Folding/PiRLC/PaperWeakFiniteUniform.lean`;
- `Nightstream/SuperNeo/Folding/Nifs/PaperNonInteractive/**`;
- `Nightstream/HyperNova/Construction2/Paper.lean`;
- `Nightstream/Protocol/FPrime/CanonicalVerifier.lean`;
- `Nightstream/Protocol/FPrime/Frozen/Obligations.lean`.

The newer `FiatShamirContract`, `RandomOracleBoundary`, linked-composition
repair, and oracle-obstruction interfaces are reproduced in Section 7 even if
they are not yet visible at the public baseline.

## 1. Mission

The required chain is:

```text
SuperNeo paper relations
  Π_CCS: strong reduction
    ↓ same ordered commitment projection φ
  Π_RLC: weak reduction
    ↓
  Π_DEC: reduction of knowledge
    ↓
  Π_SuperNeo := Π_DEC ∘ Π_RLC ∘ Π_CCS
    ↓ deterministic one-message NIFS.V
HyperNova Construction 2
  exact base branch + exact recursive branch + explicit terminal verifier
    ↓
compact typed executable checker
    ↓ differential Lean/Rust execution
    ↓ obligation minimality
    ↓ typed IR, selected encoding, Rust/R1CS refinement
```

The immediate state is:

- The deterministic paper relation, NIFS graph, Construction-2 transition,
  terminal relation, and compact executable checker exist at model level.
- The finite operational PiCCS/PiRLC/PiDEC composition exists at model level.
- The frozen composition interface was repaired so the final game is computed
  from explicit operational couplings instead of supplied by the caller.
- Full noninteractive knowledge soundness is not closed. A typed
  full-public-input transcript boundary and a generic six-event random-oracle
  probability contract now exist, but the concrete oracle experiment, event
  bounds, and multi-forking theorem do not. The asymptotic first-success
  sampler is also still open.
- The charter-mandated Lean/Rust differential test over identical typed inputs
  has not been established. Later minimality and encoding work therefore
  cannot yet count as completion of the charter, even where useful model or
  artifact slices already exist.

## 2. Exclusive paper authority

Give the receiving AI this document and these paper sections:

### SuperNeo

- Section 4, Preliminaries.
- Section 5, Embedding Products with Evaluation Homomorphism.
- Section 6, Strong and Weak Interactive Reductions, especially Definitions 9
  and 10 and Theorem 6.
- Section 7, Neo's Folding Scheme for CCS, especially PiCCS, PiRLC, PiDEC,
  Lemmas 3–4, and Theorem 7.
- Appendix Sections D.3–D.6 only.

### HyperNova

- Section 3, Multi-folding Schemes.
- Section 4, A Multi-folding Scheme for CCS.
- Section 6.2, NIVC-Compatible Multi-folding Schemes.
- Section 6.3, A Compiler from NIVC-Compatible Folding Schemes to NIVC,
  especially Construction 2 and `F'_j`.
- Appendix H.1.
- Appendix H.3.

Rust, R1CS artifacts, generated rows, historical measurements, and existing
dimensions are never semantic authority. Existing Lean theorems may be reused
only after their statements are compared with the cited paper text.

Permitted assumptions are limited to:

- the paper's field and ring laws;
- commitment binding and homomorphism contracts;
- strong-sampling-set properties;
- the SumCheck contract;
- Fiat-Shamir under an explicit random-oracle contract.

Poseidon2 and Ajtai internals are out of scope. Their exact inputs, ordering,
domain separation, and coordinate alignment are in scope. Concrete
instantiation, transcript collision, binding, and extraction failures must
remain separately named events.

## 3. Non-negotiable proof discipline

- No `sorry`, `admit`, `postulate`, new `axiom`, `unsafe`, or `sorryAx`.
- Every headline theorem needs a fail-closed `#audit_axioms` guard.
- Do not add a generic `refinementFailure`, `outputUnbound`, or equivalent
  escape branch.
- Digests compress data; they are not authority. Recompute or replay them from
  authoritative typed inputs.
- If a proof starts enumerating Rust fields or row layouts, stop and rederive
  the theorem from the paper.
- Do not change frozen semantics to fit Rust. A semantic change requires a
  paper citation or a kernel-checked obstruction.
- Do not certify row-level artifacts before the IR cost model has selected the
  encoding.

Required sequencing:

```text
paper semantics
→ obligations 1–7
→ differential Lean/Rust execution
→ obligation 8
→ IR and encoding selection
→ obligations 9–10
→ Rust/R1CS refinement
→ obligation 11
```

## 4. Obligation status

| # | Current evidence | Tier | What is still required |
|---|---|---|---|
| 1. PiCCS strong | `AsymptoticPaperStrong.paperStrong` constructs the unbounded first-success/fresh-second game, derives almost-sure termination and EPT, and `Frozen.PiCcsFirstSuccessBridge.piCcsStrong_of_unboundedFirstSuccess` reaches the exact frozen target. `SUM-DEGREE-WIDTH` requires exact syntax-derived width and the loose-width probability-one countermodel certifies necessity. `SUM-POLY-ENC` now constructs the existing finite operational `SumCheckSoundnessContract` from root counting, causal successive-coordinate sampling, exact event transport, and the explicit multi-round union bound. | model-level | Instantiate the separate alpha/gamma Schwartz--Zippel contract and the concrete field/oracle boundary. Fiat--Shamir remains separate. |
| 2. PiRLC weak | `PaperWeakFiniteUniform.paperWeak` proves finite-uniform weakness with loss `(ell + 1) / |C|`, ambient extraction using the corrected bound, query complexity, and relaxed binding as the only witness-uniqueness premise. | model-level | Instantiate the concrete strong-sampling set, relaxed Ajtai binding, and the Fiat-Shamir-derived challenge distribution. |
| 3. PiDEC knowledge reduction | `Frozen.SuperNeo.piDec_reductionOfKnowledge` exports the exact zero-loss straight-line paper reduction. | model-level | Later connect the abstract commitments/evaluations and child alignment to production data. |
| 4. composition | The generic strong–weak theorem, sequential knowledge-composition theorem, frozen linked game, and concrete finite `finiteReductionOfKnowledge` exist. The interactive error is fork sampling + SumCheck + Schwartz–Zippel + one conditioned binding loss + PiDEC zero. `nonInteractiveTotal` adds the explicit six-event Fiat-Shamir budget. | model-level | Lift the finite result through the missing sampler and instantiate the random-oracle contract. No caller-selected final game is allowed. |
| 5. NIFS | `PaperNonInteractive.verify` is deterministic. `verify_sound` returns the independent transition or exactly one of five algebraic/extraction events; `verify_complete` proves graph completeness. A separate typed boundary absorbs the complete running/fresh public pair, replays PiCCS, absorbs its output, and derives every coordinate-indexed PiRLC response. A generic six-event oracle contract proves union accounting only. | model-level | Define the actual oracle experiment and prove each event bound, including multi-fork programming; then connect those bounds to NIFS extraction and concrete primitive assumptions. |
| 6. HyperNova `F'` | `Construction2.Paper.Transition`, `holds_iff_transition`, and terminal theorems cover exactly base/recursive transitions and a terminal verifier with no final fold. The two production deviations have separate reduction theorems. | model-level | Carry noninteractive NIFS security through recursive extraction and later prove production refinement. |
| 7. executable checker | `CanonicalVerifier.eval` and terminal `eval` are extensionally equal to the frozen relations. Fixed-one specializations also exist. | model-level | Create a concrete typed Rust-to-frozen input/output map and run both implementations on the same honest and independently mutated cases. |
| 8. obligation minimality | Fixed-one step/terminal and several macro-plan inclusion-minimality theorems exist. | model-level | They are local finite-carrier results, not yet charter completion. First finish the required differential gate, then prove necessity for every obligation retained by the actual canonical checker or derive and remove it. |
| 9. typed IR | `Implementation.Lowering.Typed` has typed primitives, independent `exec`/`Holds`, soundness/completeness, explicit schemas, four-way costs, and total structural receipts. Fixed-one step/terminal programs exist. | model-level | Under the mandated sequence, re-audit this only after obligation 8. Finish any primitive-specific contracts used by the selected full checker. |
| 10. encoding | A generic Goldilocks receipt compiler, finite normal-form selector, and several direct-call recipes exist. | model-level / partial artifact-checked slices | Select and certify one complete encoding of the checked program. Prove unique row/column ownership, no emissions outside receipts, program-derived cost, and minimum in one explicit finite normal form or rewrite class. Current partial recipes and generated-row slices do not establish this globally. |
| 11. Rust/R1CS refinement | Many isolated generated-row and ownership facts exist. Native control-flow receipts are replayed in Lean. | artifact-checked, not Rust-conformant | Prove the entire requested chain: Rust acceptance = checker; R1CS soundness; honest satisfying assignments; Rust typed program = Lean IR program; physical rows = compiler output; extraction or a named event. |

## 5. Paper corrections and registered deviations

These are not optional implementation choices.

### Corrected PiRLC ambient bound

SuperNeo's strict relation uses `||z||∞ < bound`, but Appendix D.5 informally
uses `q/2` as a universal ambient bound. For odd `q`, midpoint residues have
magnitude `floor(q/2)` and fail the strict inequality.

The frozen correction is:

```text
ambient bound = floor(q / 2) + 1
```

Evidence:

- `PiRLC.PaperCorrections.midpointResidue_not_literalAmbientBounded`;
- `PiRLC.PaperCorrections.all_centeredMagnitude_lt_correctedAmbientBound`.

Tier: model-level paper obstruction and correction.

### Corrected PiRLC coordinate-fork loss

The rendered local denominator `ell / |C^ell|` is false. A finite
counterexample exists at `|C| = 3`, `ell = 2`. The frozen conservative loss is
the Appendix D.5 expression:

```text
(ell + 1) / |C|
```

`PaperWeakFiniteUniform.paperWeak` derives this from the sharper coordinate
fork inequality. Tier: model-level.

### PiCCS first-success conditioning

If ambient success is `p` and raw repeated-witness disagreement is `delta`,
conditioning the first run on success changes the usable disagreement budget
to `delta / p`. The raw `delta` cannot simply be subtracted unchanged.

Evidence:

- `StrongConditioningObstruction.unchanged_raw_uniqueness_budget_counterexample`;
- `FinitePaperStrong.finitePaperStrong`.

Tier: model-level. The finite theorem is complete; the unbounded sampler bridge
remains open.

### PiCCS target exponent and norm displays

Section 7.3's displayed `Q` shifts carried-evaluation terms by `2K+k`, while
the displayed target uses local exponents. Lean proves these conventions
differ whenever a carried coordinate exists. The frozen convention is the
coherent absolute exponent.

The displayed norm-product bounds are also malformed. The frozen roots are
derived from the authoritative strict centered predicate `|z| < b`; at `b=2`
they are `-1, 0, 1`.

Evidence:

- `PiCCS.PaperCorrections.literalTargetExponent_ne_frozen`;
- `PiCCS.PaperCorrections.literalSection73NormIndices_ne_strictCentered_at_two`.

Tier: model-level proved obstruction and frozen correction.

### Repaired composition linkage

The old frozen facade allowed a caller to provide an arbitrary final
knowledge game, so component proofs did not imply anything about that game.
`CompositionLinkageObstruction.unlinked_fields_countermodel` proves the
defect. The current `SuperNeoGames` stores exact strong–weak and PiDEC
couplings; both composed games are definitionally computed.

Tier: model-level specification repair.

### Block/lane combined-NC

Production splits FE from a block/lane NC flow, whereas SuperNeo Section 7.3
uses one joint polynomial and one SumCheck. The deviation is model-level only:

- its combined round degree and width are proved;
- accepted combined-NC reduces to ordinary NC truth and the paper relation or
  named selector/root/SumCheck events;
- the 64-lane representation is proved to contain 54 active lanes and ten
  derived-zero lanes.

Rust transcript/dataflow and generated-row refinement remain open.

### Delayed packed `y_zcol`

Production closes a predecessor's packed `y_zcol` one fold later. The
model-level lifecycle theorem covers base, every recursive edge, and terminal
closure. Successful closure gives the frozen paper transition plus the packed
bound; failures are only located algebraic, extraction, or binding events.

No child `y_zcol` sidecar or digest is authority. Rust/R1CS refinement remains
open.

## 6. The three immediate blockers

### A. Concrete PiCCS algebraic contracts

The unbounded first-success sampler is closed at model level. The remaining
PiCCS security boundary is concrete instantiation of the two charter-permitted
fixed-witness algebraic analyses. The exact paper/NIFS degree is now
verifier-owned and kernel-proved below Appendix D.4's ceiling; finite root
counting, causal successive-coordinate sampling, exact event transport, and
the explicit round union bound construct the existing
`SumCheckSoundnessContract`. A loose-width context is retained only as a
necessity counterexample and cannot inhabit the paper family or key.

Still required for a concrete instantiation:

1. bind the production extension field and challenge set to the finite
   support interface, including no zero divisors and exact cardinality;
2. instantiate the separate `MixingRootProbabilityContract`;
3. retain SumCheck and Schwartz--Zippel as separate, unconditioned losses.

### B. Concrete Fiat-Shamir instantiation

The frozen paper target is now closed under the charter-permitted explicit
random-oracle contract. `fullOracleMixtureNifsNonInteractiveSound` concludes
the literal `NifsNonInteractiveSound` proposition for the complete correlated
prefix/post-prefix experiment, while `nifsSoundAndCompleteModulo` supplies the
exact deterministic soundness/completeness core. A constant oracle still
typechecks, so deterministic replay alone remains insufficient.

Already established at model level:

- `ExplicitRandomOracleContract` has one bound field for each of six named
  event predicates and no generic failure escape;
- `anyFailure_probability_le_total` proves only the exact union bookkeeping;
- the complete running/fresh public pair is absorbed before PiCCS;
- the full PiCCS replay and output absorption are definitionally linked;
- every PiRLC response uses the actually reached post-output state and its
  literal finite coordinate.
- the selected support has zero challenge-sampling failure;
- the multi-fork programming loss is proved internally as
  `(ell + 1) / |C|`;
- the frozen quantitative theorem retains exactly one target-witness event,
  four interactive events, and four transcript-collision premises.

Still required:

1. discharge the accepted target-witness, interactive, and four collision
   contracts for the selected concrete instantiation;
2. connect the bounded PiRLC sampler to the reached post-PiCCS state, including
   explicit shortfall and bias/termination analysis;
3. separately refine typed encodings, numeric tags, ordering, coordinate
   layout, and state transitions to production Poseidon2.

Do not put Poseidon2 permutation internals in the paper layer.

### C. True Lean/Rust differential execution

Current native receipts replay Rust control flow with primitive outcomes
supplied by the receipt. Current one-slot canonical cases use hand-authored
`rustAccepted` bits. Neither proves both sides received the same protocol
input.

Required closure:

1. define a typed map from one production Rust input into the frozen
   `CanonicalVerifier` or `CanonicalTerminalVerifier` input;
2. define the corresponding output map;
3. have Rust generate honest cases and independent mutations;
4. run the Lean executable checker on those exact mapped values;
5. fail closed on every disagreement;
6. keep primitive-receipt truth separate from later full Rust refinement.

Only after this gate should obligation 8 or encoding certification continue.

## 7. Critical Lean interfaces

Binder boilerplate is omitted only where the surrounding text says so. The
displayed definitions and proof bodies are otherwise unchanged. These blocks
are included so the receiving AI can see the local post-baseline interfaces
that may not yet exist on the public branch.

### 7.1 Quantitative targets

Current quantitative vocabulary:

```lean
structure ProbabilityScale (Weight : Type uWeight) where
  zero : Weight
  one : Weight
  add : Weight -> Weight -> Weight
  subtract : Weight -> Weight -> Weight
  le : Weight -> Weight -> Prop
  le_refl : forall weight, le weight weight
  le_trans : forall {left middle right},
    le left middle -> le middle right -> le left right
  subtract_zero : forall weight, subtract weight zero = weight

structure ProbabilityExperiment
    (scale : ProbabilityScale Weight)
    (Outcome : Type uOutcome) where
  probability : (Outcome -> Prop) -> Weight
  monotone : forall {left right : Outcome -> Prop},
    (forall outcome, left outcome -> right outcome) ->
      scale.le (probability left) (probability right)

structure KnowledgeGame
    (Weight : Type uWeight)
    (Adversary : Type uAdversary)
    (Extractor : Type uExtractor) where
  perfectComplete : Prop
  publicCoin : Prop
  adversaryExpectedPolynomialTime : Adversary -> Prop
  extractorExpectedPolynomialTime : Adversary -> Extractor -> Prop
  extractionEligible : Adversary -> Prop
  adversarySuccess : Adversary -> Weight
  sourceWitnessExtracted : Adversary -> Extractor -> Weight

def ReductionOfKnowledge
    {Weight : Type uWeight}
    {Adversary : Type uAdversary}
    {Extractor : Type uExtractor}
    (scale : ProbabilityScale Weight)
    (game : KnowledgeGame Weight Adversary Extractor)
    (error : Weight) : Prop :=
  game.perfectComplete /\
  game.publicCoin /\
  forall adversary,
    game.adversaryExpectedPolynomialTime adversary ->
    game.extractionEligible adversary ->
    exists extractor,
      game.extractorExpectedPolynomialTime adversary extractor /\
      scale.le
        (scale.subtract (game.adversarySuccess adversary) error)
        (game.sourceWitnessExtracted adversary extractor)

structure StrongGame
    (Weight : Type uWeight)
    (Adversary : Type uAdversary)
    (Extractor : Type uExtractor) where
  perfectComplete : Prop
  publicCoin : Prop
  adversaryExpectedPolynomialTime : Adversary -> Prop
  extractorExpectedPolynomialTime : Adversary -> Extractor -> Prop
  extractionEligible : Adversary -> Prop
  repeatedOutputPhiMismatch : Adversary -> Weight
  ambientOutputSuccess : Adversary -> Weight
  repeatedOutputWitnessDisagreement : Adversary -> Weight
  sourceWitnessExtracted : Adversary -> Extractor -> Weight

def RejectionAdjustedStrong
    {Weight : Type uWeight}
    {Adversary : Type uAdversary}
    {Extractor : Type uExtractor}
    (scale : ProbabilityScale Weight)
    (adjust : Weight -> Weight -> Weight)
    (game : StrongGame Weight Adversary Extractor)
    (successFloor intrinsicExtractionError
      rawOutputUniquenessError : Weight) : Prop :=
  game.perfectComplete /\
  game.publicCoin /\
  (forall adversary,
    game.adversaryExpectedPolynomialTime adversary ->
      game.repeatedOutputPhiMismatch adversary = scale.zero) /\
  forall adversary,
    game.adversaryExpectedPolynomialTime adversary ->
    game.extractionEligible adversary ->
    scale.le successFloor (game.ambientOutputSuccess adversary) /\
    (scale.le (game.repeatedOutputWitnessDisagreement adversary)
        rawOutputUniquenessError ->
      exists extractor,
        game.extractorExpectedPolynomialTime adversary extractor /\
        scale.le
          (scale.subtract (game.ambientOutputSuccess adversary)
            (scale.add intrinsicExtractionError
              (adjust rawOutputUniquenessError successFloor)))
          (game.sourceWitnessExtracted adversary extractor))

structure WeakGame
    (Weight : Type uWeight)
    (Adversary : Type uAdversary)
    (PairedAdversary : Type uPairedAdversary)
    (Extractor : Type uExtractor) where
  perfectComplete : Prop
  publicCoin : Prop
  adversaryExpectedPolynomialTime : Adversary -> Prop
  pairedAdversaryExpectedPolynomialTime : PairedAdversary -> Prop
  extractorExpectedPolynomialTime : Adversary -> Extractor -> Prop
  extractionEligible : Adversary -> Prop
  adversarySuccess : Adversary -> Weight
  ambientSourceWitnessExtracted : Adversary -> Extractor -> Weight
  left : PairedAdversary -> Adversary
  right : PairedAdversary -> Adversary
  samePhiInputsAlways : PairedAdversary -> Prop
  pairedWitnessDisagreement :
    PairedAdversary -> Extractor -> Extractor -> Weight

def Weak
    {Weight : Type uWeight}
    {Adversary : Type uAdversary}
    {PairedAdversary : Type uPairedAdversary}
    {Extractor : Type uExtractor}
    (scale : ProbabilityScale Weight)
    (game : WeakGame Weight Adversary PairedAdversary Extractor)
    (extractionError witnessUniquenessError : Weight) : Prop :=
  game.perfectComplete /\
  game.publicCoin /\
  exists chooseExtractor : Adversary -> Extractor,
    (forall adversary,
      game.adversaryExpectedPolynomialTime adversary ->
      game.extractionEligible adversary ->
        game.extractorExpectedPolynomialTime adversary
          (chooseExtractor adversary) /\
        scale.le
          (scale.subtract (game.adversarySuccess adversary) extractionError)
          (game.ambientSourceWitnessExtracted adversary
            (chooseExtractor adversary))) /\
    (forall paired,
      game.pairedAdversaryExpectedPolynomialTime paired ->
      game.samePhiInputsAlways paired ->
        scale.le
          (game.pairedWitnessDisagreement paired
            (chooseExtractor (game.left paired))
            (chooseExtractor (game.right paired)))
          witnessUniquenessError)
```

The exact interactive and Fiat-Shamir exits are:

```lean
inductive InteractiveSecurityEvent (sourceCount : Nat) where
  | piCcsMixingRoot
  | piCcsSumCheckBadChallenge
  | piRlcForkSamplingFailure
  | piRlcRelaxedBindingCollision (source : Fin sourceCount)
deriving Repr, DecidableEq

inductive FiatShamirSecurityEvent where
  | publicInputBindingCollision
  | transcriptReplayCollision
  | transcriptStateCollision
  | outputAbsorptionCollision
  | challengeSamplingFailure
  | multiForkProgrammingFailure
deriving Repr, DecidableEq

structure FiatShamirErrorBudget (Weight : Type uWeight) where
  publicInputBindingCollision : Weight
  transcriptReplayCollision : Weight
  transcriptStateCollision : Weight
  outputAbsorptionCollision : Weight
  challengeSamplingFailure : Weight
  multiForkProgrammingFailure : Weight

def FiatShamirErrorBudget.total
    (scale : ProbabilityScale Weight)
    (budget : FiatShamirErrorBudget Weight) : Weight :=
  scale.add budget.publicInputBindingCollision
    (scale.add budget.transcriptReplayCollision
      (scale.add budget.transcriptStateCollision
        (scale.add budget.outputAbsorptionCollision
          (scale.add budget.challengeSamplingFailure
            budget.multiForkProgrammingFailure))))

structure InteractiveErrorBudget (Weight : Type uWeight) where
  piCcsSumCheck : Weight
  piCcsSchwartzZippel : Weight
  piRlcForkSampling : Weight
  piCcsSuccessFloor : Weight
  relaxedBindingRaw : Weight
  adjustUniqueness : Weight -> Weight -> Weight

def InteractiveErrorBudget.adjustedRelaxedBinding
    (budget : InteractiveErrorBudget Weight) : Weight :=
  budget.adjustUniqueness budget.relaxedBindingRaw budget.piCcsSuccessFloor

def InteractiveErrorBudget.strongWeakTotal
    (scale : ProbabilityScale Weight)
    (budget : InteractiveErrorBudget Weight) : Weight :=
  scale.add budget.piRlcForkSampling
    (scale.add
      (scale.add budget.piCcsSumCheck budget.piCcsSchwartzZippel)
      budget.adjustedRelaxedBinding)

def InteractiveErrorBudget.total
    (scale : ProbabilityScale Weight)
    (budget : InteractiveErrorBudget Weight) : Weight :=
  scale.add scale.zero (budget.strongWeakTotal scale)

def nonInteractiveTotal
    (scale : ProbabilityScale Weight)
    (interactive : InteractiveErrorBudget Weight)
    (fiatShamir : FiatShamirErrorBudget Weight) : Weight :=
  scale.add (interactive.total scale) (fiatShamir.total scale)
```

### 7.2 Frozen linked composition

Current frozen target:

```lean
structure SuperNeoGames where
  Weight : Type uWeight
  scale : ProbabilityScale Weight

  PiCcsAdversary : Type uPiCcsAdversary
  PiCcsExtractor : Type uPiCcsExtractor
  piCcs : StrongGame Weight PiCcsAdversary PiCcsExtractor

  PiRlcAdversary : Type uPiRlcAdversary
  PiRlcPairedAdversary : Type uPiRlcPairedAdversary
  PiRlcExtractor : Type uPiRlcExtractor
  piRlc : WeakGame Weight PiRlcAdversary PiRlcPairedAdversary PiRlcExtractor

  StrongWeakAdversary : Type uStrongWeakAdversary
  strongWeakCoupling :
    Nightstream.SuperNeo.InteractiveReduction.StrongWeakComposition.Coupling
      scale piCcs piRlc StrongWeakAdversary

  PiDecAdversary : Type uPiDecAdversary
  PiDecExtractor : Type uPiDecExtractor
  piDec : KnowledgeGame Weight PiDecAdversary PiDecExtractor

  ComposedAdversary : Type uComposedAdversary
  piDecCoupling :
    Nightstream.SuperNeo.InteractiveReduction.KnowledgeComposition.Coupling
      scale
      (Nightstream.SuperNeo.InteractiveReduction.StrongWeakComposition.knowledgeGame
        scale piCcs piRlc strongWeakCoupling)
      piDec ComposedAdversary

  IntermediateInstance : Type uIntermediate
  Projection : Type uProjection
  piCcsProjection : IntermediateInstance -> Projection
  piRlcProjection : IntermediateInstance -> Projection

  errorBudget : InteractiveErrorBudget Weight
  scaleLaws :
    Nightstream.SuperNeo.InteractiveReduction.StrongWeakComposition.ScaleLaws
      scale

def strongWeakKnowledgeGame (games : SuperNeoGames) :=
  Nightstream.SuperNeo.InteractiveReduction.StrongWeakComposition.knowledgeGame
    games.scale games.piCcs games.piRlc games.strongWeakCoupling

def superNeoCompositionGame (games : SuperNeoGames) :=
  Nightstream.SuperNeo.InteractiveReduction.KnowledgeComposition.knowledgeGame
    games.scale (strongWeakKnowledgeGame games) games.piDec
    games.piDecCoupling

def PiCcsStrong (games : SuperNeoGames) : Prop :=
  RejectionAdjustedStrong games.scale
    games.errorBudget.adjustUniqueness games.piCcs
    games.errorBudget.piCcsSuccessFloor
    (games.scale.add games.errorBudget.piCcsSumCheck
      games.errorBudget.piCcsSchwartzZippel)
    games.errorBudget.relaxedBindingRaw

def PiRlcWeak (games : SuperNeoGames) : Prop :=
  Weak games.scale games.piRlc
    games.errorBudget.piRlcForkSampling
    games.errorBudget.relaxedBindingRaw

def SharedCommitmentProjection (games : SuperNeoGames) : Prop :=
  games.piCcsProjection = games.piRlcProjection

def PiDecReductionOfKnowledge (games : SuperNeoGames) : Prop :=
  ReductionOfKnowledge games.scale games.piDec games.scale.zero

def SuperNeoCompositionReductionOfKnowledge (games : SuperNeoGames) : Prop :=
  ReductionOfKnowledge games.scale (superNeoCompositionGame games)
    (games.errorBudget.total games.scale)

theorem superNeoCompositionReductionOfKnowledge
    (games : SuperNeoGames)
    (piCcs : PiCcsStrong games)
    (piRlc : PiRlcWeak games)
    (piDec : PiDecReductionOfKnowledge games) :
    SuperNeoCompositionReductionOfKnowledge games := by
  apply
    Nightstream.SuperNeo.InteractiveReduction.KnowledgeComposition.reductionOfKnowledge
      games.scale games.scaleLaws
      (strongWeakKnowledgeGame games) games.piDec games.piDecCoupling
      (games.errorBudget.strongWeakTotal games.scale) games.scale.zero
  · exact
      Nightstream.SuperNeo.InteractiveReduction.StrongWeakComposition.reductionOfKnowledge
        games.scale games.scaleLaws games.piCcs games.piRlc
        games.strongWeakCoupling games.errorBudget.adjustUniqueness
        games.errorBudget.piCcsSuccessFloor
        (games.scale.add games.errorBudget.piCcsSumCheck
          games.errorBudget.piCcsSchwartzZippel)
        games.errorBudget.piRlcForkSampling
        games.errorBudget.relaxedBindingRaw piCcs piRlc
  · exact piDec

def SuperNeoPaperObligations (games : SuperNeoGames) : Prop :=
  PiCcsStrong games /\
  PiRlcWeak games /\
  SharedCommitmentProjection games /\
  PiDecReductionOfKnowledge games /\
  SuperNeoCompositionReductionOfKnowledge games

theorem superNeoPaperObligations_of_components
    (games : SuperNeoGames)
    (piCcs : PiCcsStrong games)
    (piRlc : PiRlcWeak games)
    (sharedProjection : SharedCommitmentProjection games)
    (piDec : PiDecReductionOfKnowledge games) :
    SuperNeoPaperObligations games :=
  ⟨piCcs, piRlc, sharedProjection, piDec,
    superNeoCompositionReductionOfKnowledge games piCcs piRlc piDec⟩
```

The operational coupling is not merely projection equality. Its critical
fields are:

```lean
structure StrongWeakComposition.ScaleLaws
    (scale : ProbabilityScale Weight) : Prop where
  subtract_mono_left : forall {left right error},
    scale.le left right ->
      scale.le (scale.subtract left error) (scale.subtract right error)
  subtract_subtract : forall probability first second,
    scale.subtract (scale.subtract probability first) second =
      scale.subtract probability (scale.add first second)

structure StrongWeakComposition.Coupling
    (scale : ProbabilityScale Weight)
    (strongGame : StrongGame Weight StrongAdversary StrongExtractor)
    (weakGame : WeakGame Weight WeakAdversary PairedAdversary WeakExtractor)
    (ComposedAdversary : Type uComposedAdversary) where
  toWeak : ComposedAdversary -> WeakAdversary
  toStrong : ComposedAdversary -> WeakExtractor -> StrongAdversary
  paired : ComposedAdversary -> WeakExtractor -> PairedAdversary
  pairedLeft : forall adversary extractor,
    weakGame.left (paired adversary extractor) = toWeak adversary
  pairedRight : forall adversary extractor,
    weakGame.right (paired adversary extractor) = toWeak adversary
  pairedExpectedPolynomialTime : forall adversary extractor,
    weakGame.adversaryExpectedPolynomialTime (toWeak adversary) ->
    weakGame.extractorExpectedPolynomialTime (toWeak adversary) extractor ->
    weakGame.pairedAdversaryExpectedPolynomialTime
      (paired adversary extractor)
  pairedSamePhi : forall adversary extractor,
    weakGame.samePhiInputsAlways (paired adversary extractor)
  intermediateProbability : forall adversary extractor,
    strongGame.ambientOutputSuccess (toStrong adversary extractor) =
      weakGame.ambientSourceWitnessExtracted (toWeak adversary) extractor
  repeatedWitnessProbability : forall adversary extractor,
    strongGame.repeatedOutputWitnessDisagreement
        (toStrong adversary extractor) =
      weakGame.pairedWitnessDisagreement (paired adversary extractor)
        extractor extractor

structure KnowledgeComposition.Coupling
    (scale : ProbabilityScale Weight)
    (firstGame : KnowledgeGame Weight FirstAdversary FirstExtractor)
    (secondGame : KnowledgeGame Weight SecondAdversary SecondExtractor)
    (ComposedAdversary : Type uComposedAdversary) where
  toSecond : ComposedAdversary -> SecondAdversary
  toFirst : ComposedAdversary -> SecondExtractor -> FirstAdversary
  intermediateProbability : forall adversary extractor,
    firstGame.adversarySuccess (toFirst adversary extractor) =
      secondGame.sourceWitnessExtracted (toSecond adversary) extractor
```

The `StrongWeakComposition` and `KnowledgeComposition` prefixes above denote
the two Lean namespaces. Universe and implicit type binders are omitted.

### 7.3 Finite component theorem boundaries

The PiCCS result:

```lean
theorem finitePaperStrong
    (context : Context Extension Commitment PublicInput shape
      columns blockCount)
    (alphabet : Support Extension)
    (adversaryExpectedPolynomialTime :
      Adversary context ProverSeed TargetSeed ProverTape -> Prop)
    (successFloor rawMismatchBudget mixingBudget sumCheckBudget : Rat)
    (ambientAdmissible : context.params.b <=
      Nightstream.SuperNeo.Folding.PiRLC.PaperCorrections.correctedAmbientBoundFor
        context.params)
    (contracts : NamedSecurityContracts context alphabet
      adversaryExpectedPolynomialTime mixingBudget sumCheckBudget) :
    RejectionAdjustedStrong
      Nightstream.SuperNeo.InteractiveReduction.FiniteUniform.scale
      (fun raw floor => raw / floor)
      (finiteStrongGame context alphabet adversaryExpectedPolynomialTime
        successFloor)
      successFloor (mixingBudget + sumCheckBudget) rawMismatchBudget := by
  refine ⟨perfectComplete context ambientAdmissible,
    publicCoin Extension shape ProverTape, ?_, ?_⟩
  · intro adversary _adversaryEpt
    exact outputPhiMismatchProbability_eq_zero context alphabet adversary
  · intro adversary adversaryEpt eligible
    refine ⟨eligible.2, ?_⟩
    intro rawMismatchBound
    refine ⟨.firstSuccessFreshSecond, ?_, ?_⟩
    · exact ⟨rfl,
        uniformTruncatedWorkBound_of_eligible context alphabet adversary
          successFloor eligible⟩
    · change
        (experiment context alphabet adversary).probabilityBool
              (success context) -
            ((mixingBudget + sumCheckBudget) +
              rawMismatchBudget / successFloor) <=
          sourceExtractionProbability context alphabet adversary successFloor
      rw [sourceExtractionProbability_eq_of_eligible context alphabet
        adversary successFloor eligible]
      exact extraction_after_first_success_of_securityContracts
        context alphabet adversary successFloor rawMismatchBudget
        mixingBudget sumCheckBudget eligible.1 eligible.2 rawMismatchBound
        (contracts.mixing adversary adversaryEpt)
        (contracts.sumCheck adversary adversaryEpt)
```

The PiRLC result:

```lean
theorem paperWeak
    (laws : ExtractionAlgebra context.semantics context.params context.algebra)
    (strongSet : StrongSetUnits laws.ring context.algebra.challengeValid)
    (ops : RelaxedBindingOps Assignment Commitment Scalar)
    (bindingLaws : RelaxedBindingLaws context.semantics context.params
      context.algebra laws ops)
    (verifier : VerifierData context)
    (relaxedBindingError : Rat)
    (binding : RelaxedBindingSecurity laws strongSet ops verifier
      relaxedBindingError) :
    Weak scale
      (weakGame laws strongSet ops verifier relaxedBindingError binding)
      (ratio (context.arity.total + 1) verifier.alphabet.cardinality)
      relaxedBindingError := by
  simpa [weakGame, correctedLoss] using
    (PaperWeakReduction.paperWeak scale ratio context laws strongSet ops
      bindingLaws relaxedBindingError
      (operationalGame laws strongSet ops verifier relaxedBindingError binding))
```

These are generic excerpts with implicit type parameters omitted from the
display. The omitted binders are ordinary universe, carrier, and shape
parameters; they add no semantic premises.

### 7.4 NIFS transition, event boundary, and checker

The independent transition:

```lean
def Transition
    (key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound)
    (running : Running Extension Commitment PublicInput shape)
    (fresh : Fresh Commitment PublicInput shape)
    (result : Running Extension Commitment PublicInput shape) : Prop :=
  exists proof : Proof Extension Commitment shape degreeBound,
  exists sourceWitness : OutputWitness shape columns,
  exists childAssignments : Fin key.params.k -> Assignment F columns,
    Realization key running fresh result proof sourceWitness childAssignments
```

The five allowed NIFS failures:

```lean
inductive BadEvent
    (key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound)
    (running : Running Extension Commitment PublicInput shape)
    (fresh : Fresh Commitment PublicInput shape)
    (proof : Proof Extension Commitment shape degreeBound)
    (_result : Running Extension Commitment PublicInput shape) : Prop where
  | piRlcCoordinateForkExtraction
      (failure : PiRlcCoordinateForkExtractionFailure key running fresh proof)
  | piDecChildExtraction
      (failure : PiDecChildExtractionFailure key running fresh proof)
  | piCcsMixingRoot
      (sourceWitness : OutputWitness shape columns)
      (root : PiCcsMixingRoot key running fresh proof sourceWitness)
  | piCcsSumCheckCollision
      (sourceWitness : OutputWitness shape columns)
      (collision : PiCcsSumCheckCollision key running fresh proof sourceWitness)
  | parentBindingCollision
      (collision : Nonempty (PiDEC.ParentOpeningBindingCollision
        key.semantics key.params
        (key.parent running fresh proof).commitment))
```

The executable graph:

```lean
def verify
    (key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound)
    (running : Running Extension Commitment PublicInput shape)
    (fresh : Fresh Commitment PublicInput shape)
    (proof : Proof Extension Commitment shape degreeBound) :
    Option (Running Extension Commitment PublicInput shape) :=
  if piCcsCheck key running fresh proof && piDecCheck key running fresh proof then
    some (key.output running fresh proof)
  else
    none
```

Soundness and graph completeness:

```lean
theorem verify_sound
    (key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound)
    (running : Running Extension Commitment PublicInput shape)
    (fresh : Fresh Commitment PublicInput shape)
    (proof : Proof Extension Commitment shape degreeBound)
    (result : Running Extension Commitment PublicInput shape)
    (accepted : verify key running fresh proof = some result) :
    Transition key running fresh result ∨
      BadEvent key running fresh proof result

theorem verify_complete
    (key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound)
    (running : Running Extension Commitment PublicInput shape)
    (fresh : Fresh Commitment PublicInput shape)
    (result : Running Extension Commitment PublicInput shape)
    (transition : Transition key running fresh result) :
    exists proof, verify key running fresh proof = some result
```

These last two blocks reproduce the exact theorem statements. Their proof
bodies are omitted; no caller-supplied correctness premise is hidden.

### 7.5 HyperNova transition and executable `F'_j`

The deterministic NIFS surface and Construction-2 carriers are:

```lean
structure Verifier
    (Key : Type uKey)
    (Running : Type uRunning)
    (Fresh : Type uFresh)
    (Proof : Type uProof) where
  verify : Key -> Running -> Fresh -> Proof -> Option Running

def Accepts
    (verifier : Verifier Key Running Fresh Proof)
    (key : Key)
    (running : Running)
    (fresh : Fresh)
    (proof : Proof)
    (output : Running) : Prop :=
  verifier.verify key running fresh proof = some output

structure HashPreimage
    (Key : Type uKey)
    (State : Type uState)
    (Running : Type uRunning)
    (slotCount : Nat) where
  verifierKeys : Fin slotCount -> Key
  iteration : Nat
  z0 : State
  current : State
  running : Fin slotCount -> Running
  pc : Nat

structure Setup
    (Key : Type uKey)
    (Running : Type uRunning)
    (Fresh : Type uFresh)
    (Proof : Type uProof)
    (slotCount : Nat) where
  verifierKeys : Fin slotCount -> Key
  nifs : Verifier Key Running Fresh Proof
  defaultRunning : Running

structure Machine
    (Key : Type uKey)
    (Digest : Type uDigest)
    (State : Type uState)
    (Witness : Type uWitness)
    (Running : Type uRunning)
    (Fresh : Type uFresh)
    (Encoded : Type uEncoded)
    (slotCount : Nat) where
  control : State -> Witness -> Fin slotCount
  step : Fin slotCount -> State -> Witness -> State
  freshPublic : Fresh -> Encoded
  encodeInstance : Digest -> Encoded
  hash : HashPreimage Key State Running slotCount -> Digest

structure Input
    (Key : Type uKey)
    (State : Type uState)
    (Witness : Type uWitness)
    (Running : Type uRunning)
    (Fresh : Type uFresh)
    (Proof : Type uProof)
    (slotCount : Nat) where
  iteration : Nat
  z0 : State
  zi : State
  running : Fin slotCount -> Running
  fresh : Fresh
  priorPc : Nat
  witness : Witness
  nifsProof : Proof

structure Output
    (Digest : Type uDigest)
    (State : Type uState)
    (Running : Type uRunning)
    (slotCount : Nat) where
  zNext : State
  runningNext : Fin slotCount -> Running
  pcNext : Fin slotCount
  x : Digest
```

The independent Construction-2 transition:

```lean
def Transition
    (setup : Setup Key Running Fresh Proof slotCount)
    (machine : Machine Key Digest State Witness Running Fresh Encoded slotCount)
    (functionIndex : Fin slotCount)
    (input : Input Key State Witness Running Fresh Proof slotCount)
    (output : Output Digest State Running slotCount) : Prop :=
  machine.control input.zi input.witness = functionIndex /\
  output.pcNext = functionIndex /\
  output.zNext = machine.step functionIndex input.zi input.witness /\
  output.x = machine.hash (nextHashPreimage setup input output) /\
  ((input.iteration = 0 /\
      input.z0 = input.zi /\
      output.runningNext = fun _ => setup.defaultRunning) \/
    exists priorPcValid : InRange slotCount input.priorPc,
      0 < input.iteration /\
      machine.freshPublic input.fresh =
        machine.encodeInstance (machine.hash (priorHashPreimage setup input)) /\
      Accepts setup.nifs
        (setup.verifierKeys (selectedIndex priorPcValid))
        (input.running (selectedIndex priorPcValid)) input.fresh input.nifsProof
        (output.runningNext (selectedIndex priorPcValid)) /\
      forall slot, slot ≠ selectedIndex priorPcValid ->
        output.runningNext slot = input.running slot)
```

The compact evaluator:

```lean
def eval
    [DecidableEq State]
    [DecidableEq Encoded]
    (setup : Setup Key Running Fresh Proof slotCount)
    (machine : Machine Key Digest State Witness Running Fresh Encoded slotCount)
    (functionIndex : Fin slotCount)
    (input : Input Key State Witness Running Fresh Proof slotCount) :
    Option (Output Digest State Running slotCount) :=
  letI : Decidable (InRange slotCount input.priorPc) := by
    unfold InRange
    infer_instance
  if machine.control input.zi input.witness = functionIndex then
    if input.iteration = 0 then
      if input.z0 = input.zi then
        some (outputFor setup machine functionIndex input
          (fun _ => setup.defaultRunning))
      else
        none
    else if priorPcValid : InRange slotCount input.priorPc then
      let selected := selectedIndex priorPcValid
      if machine.freshPublic input.fresh =
          machine.encodeInstance (machine.hash (priorHashPreimage setup input)) then
        match setup.nifs.verify (setup.verifierKeys selected)
            (input.running selected) input.fresh input.nifsProof with
        | none => none
        | some folded =>
            some (outputFor setup machine functionIndex input
              (replaceSelected input.running selected folded))
      else
        none
    else
      none
  else
    none
```

Extensional equality:

```lean
theorem accepts_iff_transition
    [DecidableEq State]
    [DecidableEq Encoded]
    (setup : Setup Key Running Fresh Proof slotCount)
    (machine : Machine Key Digest State Witness Running Fresh Encoded slotCount)
    (functionIndex : Fin slotCount)
    (input : Input Key State Witness Running Fresh Proof slotCount)
    (output : Output Digest State Running slotCount) :
    Accepts setup machine functionIndex input output <->
      Transition setup machine functionIndex input output
```

Terminal semantics are separate. The base terminal accepts only
`iteration = 0 /\ zi = z0`. The recursive terminal checks the prior public
hash, every running relation, and the selected fresh relation. It never calls
`NIFS.V`.

### 7.6 Why typed Fiat-Shamir scheduling is insufficient

The complete executable content of the current obstruction module, excluding
only its module comment, is:

```lean
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.FiatShamir

set_option autoImplicit false

namespace Nightstream.Protocol.FPrime.Frozen.NonInteractiveOracleObstruction

open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.FiatShamir

def shape : Shape where
  cubeVariables := 1
  freshCount := 0
  runningCount := 0
  matrixCount := 0
  coefficientCount := 0

def constantOracle : Oracle Bool Bool Unit shape where
  initialState := fun _context => ()
  absorbRound := fun _state _round _message => ()
  squeeze := fun _state _label => (false, ())

def constantPublicInputAbsorber : Unit -> Bool -> Bool -> Unit :=
  fun _state _running _fresh => ()

def certificate : Certificate Bool shape where
  rounds := fun _round => { coefficients := [false] }

theorem distinct_contexts_same_derived :
    (true : Bool) ≠ false /\
      derive constantOracle true certificate =
        derive constantOracle false certificate := by
  constructor
  · decide
  · rfl

theorem distinct_labels_same_squeeze :
    let alpha : ChallengeLabel shape := .alpha ⟨0, by decide⟩
    alpha ≠ .gamma /\
      constantOracle.squeeze () alpha =
      constantOracle.squeeze () .gamma := by
  decide

theorem distinct_public_inputs_same_bound_state :
    ((true, true) : Bool × Bool) ≠ (false, false) /\
      constantPublicInputAbsorber () true true =
        constantPublicInputAbsorber () false false := by
  decide

end Nightstream.Protocol.FPrime.Frozen.NonInteractiveOracleObstruction
```

Any claimed noninteractive soundness theorem that assumes only this `Oracle`
interface is false.

The current generic probability boundary is:

```lean
structure EventPredicates (Outcome : Type uOutcome) where
  publicInputBindingCollision : Outcome -> Prop
  transcriptReplayCollision : Outcome -> Prop
  transcriptStateCollision : Outcome -> Prop
  outputAbsorptionCollision : Outcome -> Prop
  challengeSamplingFailure : Outcome -> Prop
  multiForkProgrammingFailure : Outcome -> Prop

def AnyFailure
    (events : EventPredicates Outcome)
    (outcome : Outcome) : Prop :=
  events.publicInputBindingCollision outcome \/
    events.transcriptReplayCollision outcome \/
    events.transcriptStateCollision outcome \/
    events.outputAbsorptionCollision outcome \/
    events.challengeSamplingFailure outcome \/
    events.multiForkProgrammingFailure outcome

structure ExplicitRandomOracleContract
    (experiment : ProbabilityExperiment scale Outcome)
    (events : EventPredicates Outcome)
    (budget : FiatShamirErrorBudget Weight) : Prop where
  publicInputBindingCollision :
    scale.le
      (experiment.probability events.publicInputBindingCollision)
      budget.publicInputBindingCollision
  transcriptReplayCollision :
    scale.le
      (experiment.probability events.transcriptReplayCollision)
      budget.transcriptReplayCollision
  transcriptStateCollision :
    scale.le
      (experiment.probability events.transcriptStateCollision)
      budget.transcriptStateCollision
  outputAbsorptionCollision :
    scale.le
      (experiment.probability events.outputAbsorptionCollision)
      budget.outputAbsorptionCollision
  challengeSamplingFailure :
    scale.le
      (experiment.probability events.challengeSamplingFailure)
      budget.challengeSamplingFailure
  multiForkProgrammingFailure :
    scale.le
      (experiment.probability events.multiForkProgrammingFailure)
      budget.multiForkProgrammingFailure

theorem anyFailure_probability_le_total
    (scaleLaws : ProbabilityCalculus.ScaleLaws scale)
    (experiment : ProbabilityExperiment scale Outcome)
    (unionLaw : ProbabilityCalculus.UnionBound experiment)
    (events : EventPredicates Outcome)
    (budget : FiatShamirErrorBudget Weight)
    (contract : ExplicitRandomOracleContract experiment events budget) :
    scale.le
      (experiment.probability (AnyFailure events))
      (budget.total scale)
```

This theorem is only union-bound bookkeeping. It proves none of the six
contract fields.

The typed NIFS handoff now satisfies these exact equations:

```lean
def Key.publicInputState (key : Key ...) (running : Running ...)
    (fresh : Fresh ...) : State :=
  key.absorbPublicInput key.initialTranscriptState running fresh

theorem piCcsExecution_coins_eq_replayInput
    (key : Key ...) (running : Running ...) (fresh : Fresh ...)
    (proof : Proof ...) :
    (key.piCcsExecution running fresh proof).coins =
      (piCcsReplayInput key running fresh proof).derive key.oracle := by
  rfl

theorem piCcsExecution_outgoingState_eq_postOutput
    (key : Key ...) (running : Running ...) (fresh : Fresh ...)
    (proof : Proof ...) :
    (key.piCcsExecution running fresh proof).outgoingState =
      key.oracle.absorbOutput
        ((piCcsReplayInput key running fresh proof).derive key.oracle).finalState
        (key.piCcsCertificate running fresh proof).output := by
  rfl

theorem piRlcChallenge_eq_response_after_piCcsOutput
    (key : Key ...) (running : Running ...) (fresh : Fresh ...)
    (proof : Proof ...) (coordinate : Fin key.arity.total) :
    key.piRlcChallenges running fresh proof coordinate =
      key.piRlcResponse
        (key.oracle.absorbOutput
          ((piCcsReplayInput key running fresh proof).derive key.oracle).finalState
          (key.piCcsCertificate running fresh proof).output)
        coordinate := by
  rfl
```

The ellipses abbreviate ordinary carrier and shape parameters. They do not
hide semantic premises.

### 7.7 Typed IR ownership and cost boundary

The selected cost order is exact and lexicographic:

```lean
inductive Ownership where
  | committedColumn
  | publicColumn
  | auxiliaryColumn

structure Cost where
  recurringRows : Nat
  committedColumns : Nat
  publicColumns : Nat
  auxiliaryColumns : Nat

def Cost.LexLe (left right : Cost) : Prop :=
  left.recurringRows < right.recurringRows ∨
  (left.recurringRows = right.recurringRows ∧
    (left.committedColumns < right.committedColumns ∨
      (left.committedColumns = right.committedColumns ∧
        (left.publicColumns < right.publicColumns ∨
          (left.publicColumns = right.publicColumns ∧
            left.auxiliaryColumns ≤ right.auxiliaryColumns)))))
```

Every typed primitive has deterministic execution and an independently stated
relation:

```lean
inductive Primitive (signature : Signature) :
    Schema signature.types -> Schema signature.types -> Type where
  | literal ...
  | linear ...
  | product ...
  | invoke ...
  | assertTrue ...

theorem Primitive.exec_eq_some_iff_holds
    (primitive : Primitive signature input output)
    (source : Schema.Values signature.types input)
    (result : Schema.Values signature.types output) :
    primitive.exec source = some result ↔ primitive.Holds source result
```

The constructor ellipses in this last display are intentional; the complete
constructors are in `Implementation/Lowering/Typed/Program.lean`. The important
boundary is that `invoke` receives separately typed references and no
caller-supplied acceptance proposition.

Receipts are total and program cost is derived from them:

```lean
def Program.receipt (program : Program signature input output) :
    ProgramReceipt program where
  inputs := programInputReceipt signature input
  body := program.body.receiptAt .root

def Program.emissions (program : Program signature input output) :
    List (OwnedEvent signature) :=
  program.receipt.events

def Program.cost (program : Program signature input output) : Cost :=
  program.receipt.cost

theorem Program.flattened_conservation
    (program : Program signature input output) :
    program.emissions =
      (programInputReceipt signature input).events ++
        (program.body.receiptAt .root).events :=
  rfl

theorem Program.cost_eq_receipt_event_cost
    (program : Program signature input output) :
    program.cost = eventsCost program.emissions :=
  rfl
```

These are obligation-9 foundations, not proof that the production circuit is
their compiler output.

## 8. Fail-closed trust boundary

The work-in-progress interfaces use guards with these intended outcomes:

- `superNeoCompositionReductionOfKnowledge`: no axioms;
- `superNeoPaperObligations_of_components`: no axioms;
- `finitePaperStrong`: only `propext`, `Classical.choice`, `Quot.sound`;
- `paperWeak`: only `propext`, `Classical.choice`, `Quot.sound`;
- ordered-commitment alignment: at most `propext`;
- `distinct_contexts_same_derived`: at most `propext`;
- `distinct_labels_same_squeeze`: no axioms;
- `distinct_public_inputs_same_bound_state`: no axioms;
- the PiCCS replay/output and PiRLC coordinate equations: only `propext` and
  `Quot.sound`;
- `anyFailure_iff_exists_event`: no axioms;
- `anyFailure_probability_le_total`: no axioms;
- transcript-event classification: only `propext` and `Quot.sound`.

These are the required audit results, not permission to add assumptions.

## 9. Recommended next-agent work order

1. Read the paper sections in Section 2 before proposing a semantic change.
2. Use the public branch for dependencies and the code in this handoff for the
   newer composition/random-oracle interfaces.
3. Audit the repaired frozen composition against SuperNeo Theorem 6 and D.3.
4. Close one remaining paper-semantics blocker:
   - concrete instantiation of the PiCCS algebraic contracts, or
   - instantiation of the random-oracle contract plus multi-forking.
5. State the target theorem before implementing it. It must consume operational
   games and contracts, not a premise equivalent to its conclusion.
6. Add a kernel obstruction before weakening or changing any frozen target.
7. Once obligations 1–7 are genuinely closed, build the shared typed
   Lean/Rust differential corpus.
8. Do not count later minimality, IR, encoding, or artifact work until the
   sequencing gate is satisfied.

## 10. One-sentence handoff prompt

> Continue the paper-authoritative SuperNeo/HyperNova Lean proof from this
> document and the public `nico/f-prime-constraints-cuda-formal` branch:
> preserve the frozen semantics and named-event boundary, audit the linked
> composition and typed random-oracle boundary reproduced here, then close
> either the concrete PiCCS algebraic contracts or the
> random-oracle/multi-forking instantiation; do not use Rust, digests, generated
> artifacts, or caller-supplied semantic conclusions as authority, and do not
> advance minimality/encoding certification before the genuine shared-input
> Lean/Rust differential gate.
