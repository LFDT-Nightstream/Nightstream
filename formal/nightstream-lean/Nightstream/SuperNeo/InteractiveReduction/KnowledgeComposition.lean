import Nightstream.SuperNeo.InteractiveReduction.StrongWeakComposition

/-!
Sequential composition of two explicitly linked reductions of knowledge.

Source: SuperNeo Definition 5 and the final Pi_DEC composition used after
Theorem 6 in Sections 7 and D.6.

Owns: reverse-order extraction for two operationally linked knowledge games
and exact additive loss accounting.

Does not own: either component reduction, the operational maps for SuperNeo,
probability arithmetic beyond the named scale laws, Fiat--Shamir, Rust,
R1CS, or costs.

Emits constraints: no.
-/

namespace Nightstream.SuperNeo.InteractiveReduction.KnowledgeComposition

open Nightstream.SuperNeo.InteractiveReduction.Paper

universe uWeight uFirstAdversary uFirstExtractor
  uSecondAdversary uSecondExtractor uComposedAdversary

/-- The extractor for a sequential composition first runs the second-stage
extractor, then feeds its opening to the first-stage extractor. -/
structure Extractor
    (SecondExtractor : Type uSecondExtractor)
    (FirstExtractor : Type uFirstExtractor) where
  second : SecondExtractor
  first : FirstExtractor

/-- Exact operational maps for one two-stage execution.

The probability equality identifies the second extractor's concrete output
event with the first reduction's concrete success event.  It is the sole
intermediate-link obligation and must be proved from an actual execution map;
the structure contains no extraction inequality or final conclusion. -/
structure Coupling
    {Weight : Type uWeight}
    (scale : ProbabilityScale Weight)
    {FirstAdversary : Type uFirstAdversary}
    {FirstExtractor : Type uFirstExtractor}
    {SecondAdversary : Type uSecondAdversary}
    {SecondExtractor : Type uSecondExtractor}
    (firstGame : KnowledgeGame Weight FirstAdversary FirstExtractor)
    (secondGame : KnowledgeGame Weight SecondAdversary SecondExtractor)
    (ComposedAdversary : Type uComposedAdversary) where
  toSecond : ComposedAdversary -> SecondAdversary
  toFirst : ComposedAdversary -> SecondExtractor -> FirstAdversary
  intermediateProbability : forall adversary extractor,
    firstGame.adversarySuccess (toFirst adversary extractor) =
      secondGame.sourceWitnessExtracted (toSecond adversary) extractor

/-- Knowledge game computed from the exact two-stage maps. -/
def knowledgeGame
    {Weight : Type uWeight}
    (scale : ProbabilityScale Weight)
    {FirstAdversary : Type uFirstAdversary}
    {FirstExtractor : Type uFirstExtractor}
    {SecondAdversary : Type uSecondAdversary}
    {SecondExtractor : Type uSecondExtractor}
    {ComposedAdversary : Type uComposedAdversary}
    (firstGame : KnowledgeGame Weight FirstAdversary FirstExtractor)
    (secondGame : KnowledgeGame Weight SecondAdversary SecondExtractor)
    (coupling : Coupling scale firstGame secondGame ComposedAdversary) :
    KnowledgeGame Weight ComposedAdversary
      (Extractor SecondExtractor FirstExtractor) where
  perfectComplete := firstGame.perfectComplete /\ secondGame.perfectComplete
  publicCoin := firstGame.publicCoin /\ secondGame.publicCoin
  adversaryExpectedPolynomialTime := fun adversary =>
    secondGame.adversaryExpectedPolynomialTime
        (coupling.toSecond adversary) /\
      forall extractor,
        secondGame.extractorExpectedPolynomialTime
            (coupling.toSecond adversary) extractor ->
          firstGame.adversaryExpectedPolynomialTime
            (coupling.toFirst adversary extractor)
  extractorExpectedPolynomialTime := fun adversary extractor =>
    secondGame.extractorExpectedPolynomialTime
        (coupling.toSecond adversary) extractor.second /\
      firstGame.extractorExpectedPolynomialTime
        (coupling.toFirst adversary extractor.second) extractor.first
  extractionEligible := fun adversary =>
    secondGame.extractionEligible (coupling.toSecond adversary) /\
      forall extractor,
        secondGame.extractorExpectedPolynomialTime
            (coupling.toSecond adversary) extractor ->
          firstGame.extractionEligible
            (coupling.toFirst adversary extractor)
  adversarySuccess := fun adversary =>
    secondGame.adversarySuccess (coupling.toSecond adversary)
  sourceWitnessExtracted := fun adversary extractor =>
    firstGame.sourceWitnessExtracted
      (coupling.toFirst adversary extractor.second) extractor.first

/-- Sequential reductions compose with the sum of their exact losses. -/
theorem reductionOfKnowledge
    {Weight : Type uWeight}
    (scale : ProbabilityScale Weight)
    (scaleLaws :
      Nightstream.SuperNeo.InteractiveReduction.StrongWeakComposition.ScaleLaws
        scale)
    {FirstAdversary : Type uFirstAdversary}
    {FirstExtractor : Type uFirstExtractor}
    {SecondAdversary : Type uSecondAdversary}
    {SecondExtractor : Type uSecondExtractor}
    {ComposedAdversary : Type uComposedAdversary}
    (firstGame : KnowledgeGame Weight FirstAdversary FirstExtractor)
    (secondGame : KnowledgeGame Weight SecondAdversary SecondExtractor)
    (coupling : Coupling scale firstGame secondGame ComposedAdversary)
    (firstError secondError : Weight)
    (firstReduction : ReductionOfKnowledge scale firstGame firstError)
    (secondReduction : ReductionOfKnowledge scale secondGame secondError) :
    ReductionOfKnowledge scale
      (knowledgeGame scale firstGame secondGame coupling)
      (scale.add secondError firstError) := by
  rcases firstReduction with
    ⟨firstComplete, firstPublicCoin, firstExtraction⟩
  rcases secondReduction with
    ⟨secondComplete, secondPublicCoin, secondExtraction⟩
  refine ⟨⟨firstComplete, secondComplete⟩,
    ⟨firstPublicCoin, secondPublicCoin⟩, ?_⟩
  intro adversary adversaryExpected eligible
  rcases secondExtraction (coupling.toSecond adversary)
      adversaryExpected.1 eligible.1 with
    ⟨secondExtractor, secondExtractorExpected, secondResult⟩
  have firstAdversaryExpected :
      firstGame.adversaryExpectedPolynomialTime
        (coupling.toFirst adversary secondExtractor) :=
    adversaryExpected.2 secondExtractor secondExtractorExpected
  have firstEligible :
      firstGame.extractionEligible
        (coupling.toFirst adversary secondExtractor) :=
    eligible.2 secondExtractor secondExtractorExpected
  rcases firstExtraction (coupling.toFirst adversary secondExtractor)
      firstAdversaryExpected firstEligible with
    ⟨firstExtractor, firstExtractorExpected, firstResult⟩
  refine ⟨⟨secondExtractor, firstExtractor⟩,
    ⟨secondExtractorExpected, firstExtractorExpected⟩, ?_⟩
  have secondResult' :
      scale.le
        (scale.subtract
          (secondGame.adversarySuccess (coupling.toSecond adversary))
          secondError)
        (firstGame.adversarySuccess
          (coupling.toFirst adversary secondExtractor)) := by
    rw [coupling.intermediateProbability adversary secondExtractor]
    exact secondResult
  have nested := scaleLaws.subtract_mono_left secondResult'
    (error := firstError)
  have composedNested := scale.le_trans nested firstResult
  rw [scaleLaws.subtract_subtract] at composedNested
  change scale.le
    (scale.subtract
      (secondGame.adversarySuccess (coupling.toSecond adversary))
      (scale.add secondError firstError))
    (firstGame.sourceWitnessExtracted
      (coupling.toFirst adversary secondExtractor) firstExtractor)
  exact composedNested

end Nightstream.SuperNeo.InteractiveReduction.KnowledgeComposition
