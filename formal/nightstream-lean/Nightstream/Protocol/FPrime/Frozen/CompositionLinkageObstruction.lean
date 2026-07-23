import Nightstream.Protocol.FPrime.Frozen.Obligations

/-!
Countermodel to treating the frozen SuperNeo game fields as a composition.

Owns: one finite kernel-checked instantiation in which the Pi_CCS, Pi_RLC,
shared-projection, and Pi_DEC targets hold while the independently supplied
composed game is not complete.

Does not own: the repaired operational linkage, a concrete protocol game,
probability, SuperNeo security, Rust, R1CS, artifacts, or costs.

Emits constraints: no.
-/

namespace Nightstream.Protocol.FPrime.Frozen.CompositionLinkageObstruction

open Nightstream.SuperNeo.InteractiveReduction.Paper
open Nightstream.Protocol.FPrime.Frozen.Obligations

/-- Degenerate exact arithmetic is sufficient because the obstruction is the
absence of any connection between the component and composed games. -/
def unitScale : ProbabilityScale Unit where
  zero := ()
  one := ()
  add := fun _ _ => ()
  subtract := fun weight _ => weight
  le := fun _ _ => True
  le_refl := fun _ => True.intro
  le_trans := fun _ _ => True.intro
  subtract_zero := fun _ => rfl

def strongGame : StrongGame Unit Unit Unit where
  perfectComplete := True
  publicCoin := True
  adversaryExpectedPolynomialTime := fun _ => True
  extractorExpectedPolynomialTime := fun _ _ => True
  extractionEligible := fun _ => True
  repeatedOutputPhiMismatch := fun _ => ()
  ambientOutputSuccess := fun _ => ()
  repeatedOutputWitnessDisagreement := fun _ => ()
  sourceWitnessExtracted := fun _ _ => ()

def weakGame : WeakGame Unit Unit Unit Unit where
  perfectComplete := True
  publicCoin := True
  adversaryExpectedPolynomialTime := fun _ => True
  pairedAdversaryExpectedPolynomialTime := fun _ => True
  extractorExpectedPolynomialTime := fun _ _ => True
  extractionEligible := fun _ => True
  adversarySuccess := fun _ => ()
  ambientSourceWitnessExtracted := fun _ _ => ()
  left := fun _ => ()
  right := fun _ => ()
  samePhiInputsAlways := fun _ => True
  pairedWitnessDisagreement := fun _ _ _ => ()

def piDecGame : KnowledgeGame Unit Unit Unit where
  perfectComplete := True
  publicCoin := True
  adversaryExpectedPolynomialTime := fun _ => True
  extractorExpectedPolynomialTime := fun _ _ => True
  extractionEligible := fun _ => True
  adversarySuccess := fun _ => ()
  sourceWitnessExtracted := fun _ _ => ()

/-- This field may currently be chosen independently of every component. -/
def falseComposedGame : KnowledgeGame Unit Unit Unit where
  perfectComplete := False
  publicCoin := True
  adversaryExpectedPolynomialTime := fun _ => True
  extractorExpectedPolynomialTime := fun _ _ => True
  extractionEligible := fun _ => True
  adversarySuccess := fun _ => ()
  sourceWitnessExtracted := fun _ _ => ()

def games : SuperNeoGames where
  Weight := Unit
  scale := unitScale
  PiCcsAdversary := Unit
  PiCcsExtractor := Unit
  piCcs := strongGame
  PiRlcAdversary := Unit
  PiRlcPairedAdversary := Unit
  PiRlcExtractor := Unit
  piRlc := weakGame
  PiDecAdversary := Unit
  PiDecExtractor := Unit
  piDec := piDecGame
  ComposedAdversary := Unit
  ComposedExtractor := Unit
  composed := falseComposedGame
  IntermediateInstance := Unit
  Projection := Unit
  piCcsProjection := fun _ => ()
  piRlcProjection := fun _ => ()
  errorBudget := {
    piCcsSumCheck := ()
    piCcsSchwartzZippel := ()
    piRlcForkSampling := ()
    piCcsSuccessFloor := ()
    relaxedBindingRaw := ()
    adjustUniqueness := fun _ _ => ()
  }

theorem piCcsStrong : PiCcsStrong games := by
  refine ⟨True.intro, True.intro, ?_, ?_⟩
  · intro _ _
    rfl
  · intro _ _ _
    refine ⟨True.intro, ?_⟩
    intro _
    exact ⟨(), True.intro, True.intro⟩

theorem piRlcWeak : PiRlcWeak games := by
  refine ⟨True.intro, True.intro, ⟨fun _ => (), ?_, ?_⟩⟩
  · intro _ _ _
    exact ⟨True.intro, True.intro⟩
  · intro _ _ _
    exact True.intro

theorem sharedCommitmentProjection : SharedCommitmentProjection games := by
  rfl

theorem piDecReductionOfKnowledge : PiDecReductionOfKnowledge games := by
  refine ⟨True.intro, True.intro, ?_⟩
  intro _ _ _
  exact ⟨(), True.intro, True.intro⟩

theorem not_superNeoCompositionReductionOfKnowledge :
    ¬ SuperNeoCompositionReductionOfKnowledge games := by
  intro composed
  exact composed.1

/-- The component propositions and projection equality do not entail the
composition proposition for the current unconstrained `SuperNeoGames`. -/
theorem unlinked_fields_countermodel :
    PiCcsStrong games /\
    PiRlcWeak games /\
    SharedCommitmentProjection games /\
    PiDecReductionOfKnowledge games /\
    ¬ SuperNeoCompositionReductionOfKnowledge games := by
  exact ⟨piCcsStrong, piRlcWeak, sharedCommitmentProjection,
    piDecReductionOfKnowledge, not_superNeoCompositionReductionOfKnowledge⟩

end Nightstream.Protocol.FPrime.Frozen.CompositionLinkageObstruction
