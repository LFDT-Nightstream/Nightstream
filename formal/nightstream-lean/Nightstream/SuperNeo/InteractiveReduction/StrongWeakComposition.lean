import Nightstream.SuperNeo.InteractiveReduction.Paper

/-!
Paper strong--weak composition with an explicit operational coupling.

Source: SuperNeo Theorem 6 and Appendix D.3.

Owns: the generic extractor order `weak` then `strong`, the one-time use of
the weak witness-uniqueness bound as the strong reduction's output-
uniqueness bound, and exact additive loss accounting.

Does not own: concrete Pi_CCS or Pi_RLC experiments, a probability
implementation, either component reduction, Pi_DEC, Fiat--Shamir, Rust,
R1CS, or costs.

The coupling below is not a semantic escape hatch.  A concrete owner must
derive its fields by mapping one composed operational execution to the two
component experiments.  In particular, it contains no extraction inequality
and no proposition equivalent to the composition conclusion.

Emits constraints: no.
-/

namespace Nightstream.SuperNeo.InteractiveReduction.StrongWeakComposition

open Nightstream.SuperNeo.InteractiveReduction.Paper

universe uWeight uStrongAdversary uStrongExtractor
  uWeakAdversary uPairedAdversary uWeakExtractor uComposedAdversary

/-- Probability arithmetic needed to compose two subtractive extraction
inequalities.  These are ordinary laws of truncated probability subtraction;
they do not mention a protocol event. -/
structure ScaleLaws
    {Weight : Type uWeight}
    (scale : ProbabilityScale Weight) : Prop where
  subtract_mono_left : forall {left right error},
    scale.le left right ->
      scale.le (scale.subtract left error) (scale.subtract right error)
  subtract_subtract : forall probability first second,
    scale.subtract (scale.subtract probability first) second =
      scale.subtract probability (scale.add first second)

/-- The extractor returned by the composition first stores Pi_RLC's weak
extractor and then Pi_CCS's strong extractor. -/
structure Extractor
    (WeakExtractor : Type uWeakExtractor)
    (StrongExtractor : Type uStrongExtractor) where
  weak : WeakExtractor
  strong : StrongExtractor

/-- Exact maps between one composed execution and the component games used
in Appendix D.3.

`toWeak` runs the first-stage prover and exposes its output as the input of
the weak reduction.  `toStrong` runs that first stage followed by the chosen
weak extractor.  `paired` is the two-repetition experiment needed to feed the
weak witness-uniqueness theorem into the strong reduction.

The two probability equalities identify predicates computed by those exact
maps.  They are the obligations that the concrete operational experiment
must prove; no arbitrary composed game or caller-provided acceptance
predicate is present. -/
structure Coupling
    {Weight : Type uWeight}
    (scale : ProbabilityScale Weight)
    {StrongAdversary : Type uStrongAdversary}
    {StrongExtractor : Type uStrongExtractor}
    {WeakAdversary : Type uWeakAdversary}
    {PairedAdversary : Type uPairedAdversary}
    {WeakExtractor : Type uWeakExtractor}
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

/-- The knowledge game computed from an exact strong--weak coupling.

The adversary is eligible precisely when the induced weak adversary is
eligible and every EPT weak extractor induces an eligible strong adversary.
The composed success and source-extraction probabilities are component-game
probabilities, not new caller fields. -/
def knowledgeGame
    {Weight : Type uWeight}
    (scale : ProbabilityScale Weight)
    {StrongAdversary : Type uStrongAdversary}
    {StrongExtractor : Type uStrongExtractor}
    {WeakAdversary : Type uWeakAdversary}
    {PairedAdversary : Type uPairedAdversary}
    {WeakExtractor : Type uWeakExtractor}
    {ComposedAdversary : Type uComposedAdversary}
    (strongGame : StrongGame Weight StrongAdversary StrongExtractor)
    (weakGame : WeakGame Weight WeakAdversary PairedAdversary WeakExtractor)
    (coupling : Coupling scale strongGame weakGame ComposedAdversary) :
    KnowledgeGame Weight ComposedAdversary
      (Extractor WeakExtractor StrongExtractor) where
  perfectComplete := strongGame.perfectComplete /\ weakGame.perfectComplete
  publicCoin := strongGame.publicCoin /\ weakGame.publicCoin
  adversaryExpectedPolynomialTime := fun adversary =>
    weakGame.adversaryExpectedPolynomialTime (coupling.toWeak adversary) /\
    forall extractor,
      weakGame.extractorExpectedPolynomialTime
          (coupling.toWeak adversary) extractor ->
        strongGame.adversaryExpectedPolynomialTime
          (coupling.toStrong adversary extractor)
  extractorExpectedPolynomialTime := fun adversary extractor =>
    weakGame.extractorExpectedPolynomialTime
        (coupling.toWeak adversary) extractor.weak /\
      strongGame.extractorExpectedPolynomialTime
        (coupling.toStrong adversary extractor.weak) extractor.strong
  extractionEligible := fun adversary =>
    weakGame.extractionEligible (coupling.toWeak adversary) /\
    forall extractor,
      weakGame.extractorExpectedPolynomialTime
          (coupling.toWeak adversary) extractor ->
        strongGame.extractionEligible
          (coupling.toStrong adversary extractor)
  adversarySuccess := fun adversary =>
    weakGame.adversarySuccess (coupling.toWeak adversary)
  sourceWitnessExtracted := fun adversary extractor =>
    strongGame.sourceWitnessExtracted
      (coupling.toStrong adversary extractor.weak) extractor.strong

/-- SuperNeo Theorem 6 with Appendix D.4's success-gated disagreement loss
made explicit. The weak reduction supplies the raw witness-disagreement bound
`delta`; the strong reduction charges its declared root envelope exactly once.
No pointwise success floor is present. -/
theorem reductionOfKnowledge
    {Weight : Type uWeight}
    (scale : ProbabilityScale Weight)
    (scaleLaws : ScaleLaws scale)
    {StrongAdversary : Type uStrongAdversary}
    {StrongExtractor : Type uStrongExtractor}
    {WeakAdversary : Type uWeakAdversary}
    {PairedAdversary : Type uPairedAdversary}
    {WeakExtractor : Type uWeakExtractor}
    {ComposedAdversary : Type uComposedAdversary}
    (strongGame : StrongGame Weight StrongAdversary StrongExtractor)
    (weakGame : WeakGame Weight WeakAdversary PairedAdversary WeakExtractor)
    (coupling : Coupling scale strongGame weakGame ComposedAdversary)
    (strongIntrinsicError weakExtractionError
      witnessUniquenessRaw witnessUniquenessRoot : Weight)
    (strong : SuccessGatedStrong scale strongGame
      strongIntrinsicError witnessUniquenessRaw witnessUniquenessRoot)
    (weak : Weak scale weakGame weakExtractionError
      witnessUniquenessRaw) :
    ReductionOfKnowledge scale
      (knowledgeGame scale strongGame weakGame coupling)
      (scale.add weakExtractionError
        (scale.add strongIntrinsicError
          witnessUniquenessRoot)) := by
  rcases strong with
    ⟨strongComplete, strongPublicCoin, _strongPhiRestricted,
      strongExtraction⟩
  rcases weak with
    ⟨weakComplete, weakPublicCoin, chooseWeakExtractor,
      weakExtraction, weakUniqueness⟩
  refine ⟨⟨strongComplete, weakComplete⟩,
    ⟨strongPublicCoin, weakPublicCoin⟩, ?_⟩
  intro adversary adversaryExpected eligible
  let weakAdversary := coupling.toWeak adversary
  let weakExtractor := chooseWeakExtractor weakAdversary
  have weakResult := weakExtraction weakAdversary adversaryExpected.1 eligible.1
  have weakExtractorExpected :
      weakGame.extractorExpectedPolynomialTime weakAdversary weakExtractor :=
    weakResult.1
  have pairedExpected :
      weakGame.pairedAdversaryExpectedPolynomialTime
        (coupling.paired adversary weakExtractor) :=
    coupling.pairedExpectedPolynomialTime adversary weakExtractor
      adversaryExpected.1 weakExtractorExpected
  have pairedUnique := weakUniqueness
    (coupling.paired adversary weakExtractor) pairedExpected
    (coupling.pairedSamePhi adversary weakExtractor)
  have weakExtractorAtLeft :
      chooseWeakExtractor
          (weakGame.left (coupling.paired adversary weakExtractor)) =
        weakExtractor := by
    rw [coupling.pairedLeft adversary weakExtractor]
  have weakExtractorAtRight :
      chooseWeakExtractor
          (weakGame.right (coupling.paired adversary weakExtractor)) =
        weakExtractor := by
    rw [coupling.pairedRight adversary weakExtractor]
  rw [weakExtractorAtLeft, weakExtractorAtRight] at pairedUnique
  have strongWitnessUnique :
      scale.le
        (strongGame.repeatedOutputWitnessDisagreement
          (coupling.toStrong adversary weakExtractor))
        witnessUniquenessRaw := by
    rw [coupling.repeatedWitnessProbability adversary weakExtractor]
    exact pairedUnique
  have strongAdversaryExpected :
      strongGame.adversaryExpectedPolynomialTime
        (coupling.toStrong adversary weakExtractor) :=
    adversaryExpected.2 weakExtractor weakExtractorExpected
  have strongEligible :
      strongGame.extractionEligible
        (coupling.toStrong adversary weakExtractor) :=
    eligible.2 weakExtractor weakExtractorExpected
  rcases strongExtraction (coupling.toStrong adversary weakExtractor)
      strongAdversaryExpected strongEligible strongWitnessUnique with
    ⟨strongExtractor, strongExtractorExpected, strongResult⟩
  refine ⟨⟨weakExtractor, strongExtractor⟩,
    ⟨weakExtractorExpected, strongExtractorExpected⟩, ?_⟩
  have weakResult' :
      scale.le
        (scale.subtract
          (weakGame.adversarySuccess weakAdversary) weakExtractionError)
        (strongGame.ambientOutputSuccess
          (coupling.toStrong adversary weakExtractor)) := by
    rw [coupling.intermediateProbability adversary weakExtractor]
    exact weakResult.2
  have nested := scaleLaws.subtract_mono_left weakResult'
      (error := scale.add strongIntrinsicError
        witnessUniquenessRoot)
  have composedNested := scale.le_trans nested strongResult
  rw [scaleLaws.subtract_subtract] at composedNested
  exact composedNested

end Nightstream.SuperNeo.InteractiveReduction.StrongWeakComposition
