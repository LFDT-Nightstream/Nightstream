import Nightstream.SuperNeo.Folding.PiDEC

/-!
Adversarial model regression for the Π_DEC parent-opening binding gate.

The toy commitment deliberately maps distinct Boolean assignments to the same
value. Accepted public recomposition and valid openings therefore must expose
the collision branch; they may not manufacture witness equality.
-/

namespace NightstreamTests.PiDECBinding

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Folding

def params : GlobalParams where
  q := 97
  b := 2
  k := 1
  maxFresh := 0
  expansionT := 1
  rlc_bound := by decide

def semantics : RelationSemantics Unit Bool Unit Unit Unit Unit where
  commit := fun _ => ()
  projectPublicInput := fun _ => ()
  normBounded := fun _ _ => True
  ccsSatisfied := fun _ _ => True
  evaluationPointValid := fun _ _ => True
  evaluations := fun _ _ _ => #[()]

def claim (stage : NormStage) : CE.Instance Unit Unit Unit Unit Unit where
  constraintSystem := ()
  commitment := ()
  publicInput := ()
  point := ()
  evaluations := #[()]
  stage := stage

def algebra : PiDEC.Algebra Unit Bool Unit Unit Unit Unit semantics params where
  splitAssignment := fun assignment _ => assignment
  recomposeAssignment := fun assignments => assignments ⟨0, by decide⟩
  recomposeCommitment := fun _ => ()
  recomposePublicInput := fun _ => ()
  recomposeEvaluations := fun _ => #[()]
  split_recompose := by intro assignment; rfl
  split_norm := by intros; trivial
  recompose_norm := by intros; trivial
  commit_hom := by intros; rfl
  publicInput_hom := by intros; rfl
  evaluations_hom := by intros; rfl

def attempt : PiDEC.Attempt Unit Unit Unit Unit Unit params where
  parent := claim .combined
  children := fun _ => claim .fresh

theorem accepted : PiDEC.Accepted algebra attempt := by
  exact {
    parentCombined := rfl
    childFresh := fun _ => rfl
    sameStructure := fun _ => rfl
    samePoint := fun _ => rfl
    commitmentEquation := rfl
    publicInputEquation := rfl
    evaluationEquation := rfl
  }

theorem parentValid : CE.Holds semantics params attempt.parent false := by
  exact ⟨⟨rfl, rfl, trivial⟩, trivial, rfl⟩

def childAssignments : Fin params.k → Bool := fun _ => true

theorem childrenValid : ∀ i,
    CE.Holds semantics params (attempt.children i) (childAssignments i) := by
  intro i
  exact ⟨⟨rfl, rfl, trivial⟩, trivial, rfl⟩

/-- Distinct valid parent and recomposed openings force the named collision. -/
example : Nonempty
    (PiDEC.ParentOpeningBindingCollision semantics params
      attempt.parent.commitment) := by
  rcases PiDEC.accepted_parent_eq_recompose_or_bindingCollision
      semantics params algebra attempt false childAssignments accepted
      parentValid childrenValid with equal | collision
  · change false = true at equal
    cases equal
  · exact collision

end NightstreamTests.PiDECBinding
