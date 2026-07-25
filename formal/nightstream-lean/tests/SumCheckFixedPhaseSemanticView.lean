import Nightstream.SuperNeo.SumCheck.FixedPhase.SemanticView

/-! Focused ghost-free semantic-view regression. -/

namespace NightstreamTests.SumCheckFixedPhaseSemanticView

open Nightstream.SuperNeo.SumCheck
open Nightstream.SuperNeo.SumCheck.Finite
open Nightstream.SuperNeo.SumCheck.Finite.FixedPhase

def ops : Ops Nat where
  zero := 0
  one := 1
  add := Nat.add
  mul := Nat.mul

def q : List Nat -> Nat := fun _ => 0

def wire : SemanticView.Wire Nat 0 where
  initial := 0
  terminal := 0
  challenges := []
  certificate := { rounds := [] }
  challengeSetSize := 1

example : SemanticView.Accepted ops wire := by
  rfl

example :
    Nightstream.SuperNeo.SumCheck.Accepted ops.toSymbolic
      (SemanticView.semanticInstance ops q wire) := by
  exact SemanticView.accepted_implies_symbolicAccepted ops q wire
    (by rfl) (by rfl)

example :
    Nightstream.SuperNeo.SumCheck.TruthPath ops.toSymbolic
      (SemanticView.semanticInstance ops q wire) := by
  exact SemanticView.accepted_implies_truthPath ops q wire
    (by rfl) (by rfl)

end NightstreamTests.SumCheckFixedPhaseSemanticView
