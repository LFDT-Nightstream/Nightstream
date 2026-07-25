import Nightstream.Protocol.FPrime.Frozen.Obligations

/-!
Countermodel that justified removing the free final-game field from the frozen
SuperNeo specification.

Owns: one finite kernel-checked model of the prior unlinked shape in which the
Pi_CCS, Pi_RLC, shared-projection, and Pi_DEC targets hold while an
independently supplied composed game is not complete.

Does not own: a concrete protocol game, probability, SuperNeo security, Rust,
R1CS, artifacts, or costs.  The repaired frozen interface is owned by
`Frozen.Obligations.SuperNeoGames`, whose final game is now definitionally
computed from explicit operational couplings.

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

/-- In the obsolete shape, this field could be chosen independently of every
component. -/
def falseComposedGame : KnowledgeGame Unit Unit Unit where
  perfectComplete := False
  publicCoin := True
  adversaryExpectedPolynomialTime := fun _ => True
  extractorExpectedPolynomialTime := fun _ _ => True
  extractionEligible := fun _ => True
  adversarySuccess := fun _ => ()
  sourceWitnessExtracted := fun _ _ => ()

/-- The obsolete shape kept only inside this countermodel.  Its final game is
unconstrained by its component games. -/
structure UnlinkedGames where
  piCcs : StrongGame Unit Unit Unit
  piRlc : WeakGame Unit Unit Unit Unit
  piDec : KnowledgeGame Unit Unit Unit
  composed : KnowledgeGame Unit Unit Unit
  piCcsProjection : Unit -> Unit
  piRlcProjection : Unit -> Unit

def games : UnlinkedGames where
  piCcs := strongGame
  piRlc := weakGame
  piDec := piDecGame
  composed := falseComposedGame
  piCcsProjection := fun _ => ()
  piRlcProjection := fun _ => ()

theorem piCcsStrong :
    RejectionAdjustedStrong unitScale (fun _ _ => ()) games.piCcs
      () () () := by
  refine ⟨True.intro, True.intro, ?_, ?_⟩
  · intro _ _
    rfl
  · intro _ _ _
    refine ⟨True.intro, ?_⟩
    intro _
    exact ⟨(), True.intro, True.intro⟩

theorem piRlcWeak : Weak unitScale games.piRlc () () := by
  refine ⟨True.intro, True.intro, ⟨fun _ => (), ?_, ?_⟩⟩
  · intro _ _ _
    exact ⟨True.intro, True.intro⟩
  · intro _ _ _
    exact True.intro

theorem sharedCommitmentProjection :
    games.piCcsProjection = games.piRlcProjection := by
  rfl

theorem piDecReductionOfKnowledge :
    ReductionOfKnowledge unitScale games.piDec () := by
  refine ⟨True.intro, True.intro, ?_⟩
  intro _ _ _
  exact ⟨(), True.intro, True.intro⟩

theorem not_superNeoCompositionReductionOfKnowledge :
    ¬ ReductionOfKnowledge unitScale games.composed () := by
  intro composed
  exact composed.1

/-- The component propositions and projection equality did not entail the
composition proposition when the final game was a free field. -/
theorem unlinked_fields_countermodel :
    RejectionAdjustedStrong unitScale (fun _ _ => ()) games.piCcs
        () () () /\
    Weak unitScale games.piRlc () () /\
    games.piCcsProjection = games.piRlcProjection /\
    ReductionOfKnowledge unitScale games.piDec () /\
    ¬ ReductionOfKnowledge unitScale games.composed () := by
  exact ⟨piCcsStrong, piRlcWeak, sharedCommitmentProjection,
    piDecReductionOfKnowledge, not_superNeoCompositionReductionOfKnowledge⟩

end Nightstream.Protocol.FPrime.Frozen.CompositionLinkageObstruction
