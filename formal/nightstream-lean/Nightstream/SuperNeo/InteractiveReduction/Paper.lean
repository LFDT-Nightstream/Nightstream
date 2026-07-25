/-!
Paper-owned quantitative security vocabulary for interactive reductions.

Source: SuperNeo Definitions 5, 9, and 10 and Theorem 6.

Owns: the exact probability inequalities required of reductions of knowledge,
weak reductions, and strong reductions.

Does not own: a concrete protocol game, an adversary language, a probability
implementation, SumCheck, commitment security, Fiat--Shamir, Rust, R1CS, or
costs.

Emits constraints: no.

The concrete game owner must define every probability below from its actual
algorithms and transcript distribution.  Supplying one of these contracts as
a premise is not a proof that Pi_CCS, Pi_RLC, or Pi_DEC satisfies it.
-/

namespace Nightstream.SuperNeo.InteractiveReduction.Paper

universe uWeight uAdversary uPairedAdversary uExtractor

/-- Minimal arithmetic used to state security inequalities without selecting
a probability library. -/
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

universe uOutcome

/-- Probability of predicates in one concrete protocol experiment.  The sole
law needed by straight-line extraction is monotonicity under implication. -/
structure ProbabilityExperiment
    {Weight : Type uWeight}
    (scale : ProbabilityScale Weight)
    (Outcome : Type uOutcome) where
  probability : (Outcome -> Prop) -> Weight
  monotone : forall {left right : Outcome -> Prop},
    (forall outcome, left outcome -> right outcome) ->
      scale.le (probability left) (probability right)

/-- Complete experiment summary for Definition 5.  Its fields are computed by
an independently defined protocol game. -/
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

/-- SuperNeo Definition 5, with the negligible loss named by `error`. -/
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

/-- Complete experiment summary for Definition 10. -/
structure StrongGame
    (Weight : Type uWeight)
    (Adversary : Type uAdversary)
    (Extractor : Type uExtractor) where
  perfectComplete : Prop
  publicCoin : Prop
  adversaryExpectedPolynomialTime : Adversary -> Prop
  extractorExpectedPolynomialTime : Adversary -> Extractor -> Prop
  extractionEligible : Adversary -> Prop
  /-- Probability that two accepting repetitions disagree after applying the
  declared output projection `phi`. Definition 10 requires this to be zero. -/
  repeatedOutputPhiMismatch : Adversary -> Weight
  /-- Probability that an accepting output belongs to the ambient target
  relation `R'_2`. -/
  ambientOutputSuccess : Adversary -> Weight
  /-- Probability that two accepting repetitions expose distinct target
  witnesses. Definition 10 conditions extraction on this being negligible. -/
  repeatedOutputWitnessDisagreement : Adversary -> Weight
  sourceWitnessExtracted : Adversary -> Extractor -> Weight

/-- SuperNeo Definition 10 with an already-conditioned witness-uniqueness
budget. `intrinsicExtractionError` is the reduction's own extraction loss.
`outputUniquenessError` must bound the disagreement distribution actually
seen by the extractor; a raw two-run bound cannot be reused here after
rejection sampling without a conditioning adjustment. Concrete `Pi_CCS`
uses `RejectionAdjustedStrong` below instead. -/
def Strong
    {Weight : Type uWeight}
    {Adversary : Type uAdversary}
    {Extractor : Type uExtractor}
    (scale : ProbabilityScale Weight)
    (game : StrongGame Weight Adversary Extractor)
    (intrinsicExtractionError outputUniquenessError : Weight) : Prop :=
  game.perfectComplete /\
  game.publicCoin /\
  (forall adversary,
    game.adversaryExpectedPolynomialTime adversary ->
      game.repeatedOutputPhiMismatch adversary = scale.zero) /\
  forall adversary,
    game.adversaryExpectedPolynomialTime adversary ->
    game.extractionEligible adversary ->
    scale.le (game.repeatedOutputWitnessDisagreement adversary)
      outputUniquenessError ->
    exists extractor,
      game.extractorExpectedPolynomialTime adversary extractor /\
      scale.le
        (scale.subtract (game.ambientOutputSuccess adversary)
          (scale.add intrinsicExtractionError outputUniquenessError))
        (game.sourceWitnessExtracted adversary extractor)

/-- Quantitative strong reduction with the rejection-sampling adjustment from
Appendix D.4 exposed explicitly.

`rawOutputUniquenessError` bounds the literal Definition-10 two-run witness
disagreement event. `successFloor` is a concrete lower bound on relaxed
success, derived from `extractionEligible`. The final loss charges
`adjust rawOutputUniquenessError successFloor`, never the raw error itself.
For ordinary probabilities, `adjust delta mu` is the conditioning loss
`delta / mu`. -/
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

/-- Complete experiment summary for Definition 9.  A paired adversary is the
paper's `(B,B')` experiment producing two same-`phi` inputs. -/
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
  /-- Probability that the two selected extractors return different non-bottom
  witnesses in the paper's same-`phi` experiment. -/
  pairedWitnessDisagreement :
    PairedAdversary -> Extractor -> Extractor -> Weight

/-- SuperNeo Definition 9.  The chosen extractor for each adversary is exposed
so the same algorithm is used in the paired witness-uniqueness experiment. -/
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

/-- Named interactive security exits in the SuperNeo composition. -/
inductive InteractiveSecurityEvent (sourceCount : Nat) where
  | piCcsMixingRoot
  | piCcsSumCheckBadChallenge
  | piRlcForkSamplingFailure
  | piRlcRelaxedBindingCollision (source : Fin sourceCount)
deriving Repr, DecidableEq

/-- Fiat--Shamir adds only explicitly named random-oracle exits. -/
inductive FiatShamirSecurityEvent where
  | publicInputBindingCollision
  | transcriptReplayCollision
  | transcriptStateCollision
  | outputAbsorptionCollision
  | challengeSamplingFailure
  | multiForkProgrammingFailure
deriving Repr, DecidableEq

/-- Symbolic ownership of the six Fiat--Shamir/random-oracle error terms.
Every field must be instantiated by the corresponding exact event in a
concrete oracle experiment. -/
structure FiatShamirErrorBudget (Weight : Type uWeight) where
  publicInputBindingCollision : Weight
  transcriptReplayCollision : Weight
  transcriptStateCollision : Weight
  outputAbsorptionCollision : Weight
  challengeSamplingFailure : Weight
  multiForkProgrammingFailure : Weight

/-- The exact random-oracle loss in transcript schedule order.  No
commutativity of `scale.add` is assumed. -/
def FiatShamirErrorBudget.total
    {Weight : Type uWeight}
    (scale : ProbabilityScale Weight)
    (budget : FiatShamirErrorBudget Weight) : Weight :=
  scale.add budget.publicInputBindingCollision
    (scale.add budget.transcriptReplayCollision
      (scale.add budget.transcriptStateCollision
        (scale.add budget.outputAbsorptionCollision
          (scale.add budget.challengeSamplingFailure
            budget.multiForkProgrammingFailure))))

/-- Symbolic ownership of the four interactive error terms from Appendix D. -/
structure InteractiveErrorBudget (Weight : Type uWeight) where
  piCcsSumCheck : Weight
  piCcsSchwartzZippel : Weight
  piRlcForkSampling : Weight
  /-- Non-negligible lower bound `mu` required by the `Pi_CCS` rejection
  sampler. This is not itself an error term. -/
  piCcsSuccessFloor : Weight
  /-- Raw Definition-10 / relaxed-binding disagreement bound `delta`. -/
  relaxedBindingRaw : Weight
  /-- Conditioning adjustment; for concrete probabilities this is
  `delta / mu`. -/
  adjustUniqueness : Weight -> Weight -> Weight

/-- The one binding loss actually charged after rejection conditioning. -/
def InteractiveErrorBudget.adjustedRelaxedBinding
    {Weight : Type uWeight}
    (budget : InteractiveErrorBudget Weight) : Weight :=
  budget.adjustUniqueness budget.relaxedBindingRaw budget.piCcsSuccessFloor

/-- The exact loss of the strong--weak composition in the syntactic order
produced by Theorem 6: weak fork sampling, then the two intrinsic `Pi_CCS`
losses, then the once-adjusted relaxed-binding loss. -/
def InteractiveErrorBudget.strongWeakTotal
    {Weight : Type uWeight}
    (scale : ProbabilityScale Weight)
    (budget : InteractiveErrorBudget Weight) : Weight :=
  scale.add budget.piRlcForkSampling
    (scale.add
      (scale.add budget.piCcsSumCheck budget.piCcsSchwartzZippel)
      budget.adjustedRelaxedBinding)

/-- The exact additive loss of
`Pi_DEC ∘ Pi_RLC ∘ Pi_CCS`; `Pi_DEC` contributes the explicit zero loss from
Theorem 7.  No unnamed final `negl` term is available. -/
def InteractiveErrorBudget.total
    {Weight : Type uWeight}
    (scale : ProbabilityScale Weight)
    (budget : InteractiveErrorBudget Weight) : Weight :=
  scale.add scale.zero (budget.strongWeakTotal scale)

/-- The one extraction event introduced when an accepted public NIFS
transcript is related to Definition 5's witness-carrying target relation.

This is not an intrinsic loss of `Pi_DEC`: Theorem 7 remains zero-loss once
its target child witnesses exist. Public verifier acceptance alone does not
construct those witnesses. -/
structure NifsExtractionErrorBudget (Weight : Type uWeight) where
  piDecTargetWitnessFailure : Weight

/-- NIFS residual loss in event order: first accepted NIFS execution without
target witnesses, then the four nonzero interactive composition terms.
Rejected transcripts do not consume this extraction budget. Theorem 7's
intrinsic zero is intentionally not relabeled as this bridge event. -/
def nifsInteractiveTotal
    {Weight : Type uWeight}
    (scale : ProbabilityScale Weight)
    (extraction : NifsExtractionErrorBudget Weight)
    (interactive : InteractiveErrorBudget Weight) : Weight :=
  scale.add extraction.piDecTargetWitnessFailure
    (interactive.strongWeakTotal scale)

/-- Full non-interactive SuperNeo loss: first the exact NIFS extraction and
interactive losses, then the explicitly enumerated Fiat--Shamir/random-oracle
loss. -/
def nonInteractiveTotal
    {Weight : Type uWeight}
    (scale : ProbabilityScale Weight)
    (extraction : NifsExtractionErrorBudget Weight)
    (interactive : InteractiveErrorBudget Weight)
    (fiatShamir : FiatShamirErrorBudget Weight) : Weight :=
  scale.add (nifsInteractiveTotal scale extraction interactive)
    (fiatShamir.total scale)

end Nightstream.SuperNeo.InteractiveReduction.Paper
