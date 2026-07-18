import Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270.PaddedIdentityEvaluation
import Nightstream.SuperNeo.Concrete.Phi81Relation.PiDECAlgebra.Algebra
import Nightstream.SuperNeo.Folding.PiDEC.CanonicalParent

/-!
Exact production-Phi81 child-substitution counterexample for parent-only
authority.

Assurance tier: model-level.

Owns: two distinct fourteen-child families of valid strict-`b` Phi81 CE
openings, their identical recomposed assignment and public PiDEC parent, full
PiDEC acceptance of both families, and the resulting failure of any handle
that depends only on that parent to bind the child vector.

Does not own: Rust/R1CS refinement, a concrete Poseidon2 implementation,
exploitation of a lifecycle, probability, costs, or row removal.

Emits constraints: no.

Authority boundary: this is not a hash collision. Both child families have
the exact same parent statement. The alias is the production radix identity
`1 + 2*0 = -1 + 2*1`, lifted coordinatewise to the complete 270-coordinate
Phi81 carrier. Therefore strict PiDEC recomposition can validate a parent but
cannot make a parent-only digest authoritative for its children.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `pi_dec.necessity.child_substitution.assignments` | two distinct strict-2 child vectors recompose identically | counterexample | `recomposedAssignments_eq`, `assignments_ne` |
| `pi_dec.necessity.child_substitution.openings` | every child is a valid fresh Phi81 CE opening | derived | `leftChildrenValid`, `rightChildrenValid` |
| `pi_dec.necessity.child_substitution.parent` | both vectors compute the exact same combined parent statement | derived | `parents_eq` |
| `pi_dec.necessity.child_substitution.acceptance` | strict production PiDEC accepts both vectors under that parent | counterexample | `leftAccepted`, `rightAccepted` |
| `pi_dec.necessity.child_substitution.handle` | no function of only the parent binds the accepted child vector | impossibility theorem | `no_parentOnlyHandle_binds` |
-/

namespace Nightstream.SuperNeo.Folding.PiDEC.Necessity.ProductionChildSubstitution

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Concrete.Phi81Relation.PiRLCAlgebra.Norm
open Nightstream.SuperNeo.Folding
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint

namespace Fixture

abbrev Shape := FPrimeCarrier270.PaddedIdentityEvaluation.shape
abbrev Child := PiDECAlgebra.Radix.ChildIndex
abbrev Assignment := Phi81Relation.Assignment Shape
abbrev Statement :=
  Phi81Relation.CEStatement Shape
    (PiRLCAlgebra.Commitment.Value 0)

def kPositive : 0 < productionGlobalParams.k := by decide

def childZero : Child := ⟨0, by decide⟩

def childOne : Child := ⟨1, by decide⟩

def firstPublicColumn : Fin (Phi81Relation.Shape.publicWidth Shape) :=
  ⟨0, by decide⟩

/-- Canonical `(1,0,0,...)` signed child digits. -/
def leftDigit (child : Child) : F :=
  if child.val = 0 then 1 else 0

/-- Equally short `(-1,1,0,...)` signed child digits. -/
def rightDigit (child : Child) : F :=
  if child.val = 0 then -1 else if child.val = 1 then 1 else 0

def leftAssignments (child : Child) : Assignment :=
  fun _ => leftDigit child

def rightAssignments (child : Child) : Assignment :=
  fun _ => rightDigit child

@[simp] theorem centeredMagnitude_one :
    centeredMagnitude (1 : F) = 1 := by
  decide

theorem leftDigitsShort (child : Child) :
    centeredMagnitude (leftDigit child) < productionGlobalParams.b := by
  by_cases zero : child.val = 0 <;>
    simp [leftDigit, zero, productionGlobalParams,
      Centered.centeredMagnitude_zero, centeredMagnitude_one]

theorem rightDigitsShort (child : Child) :
    centeredMagnitude (rightDigit child) < productionGlobalParams.b := by
  by_cases zero : child.val = 0
  · simp [rightDigit, zero, productionGlobalParams,
      Centered.centeredMagnitude_neg, centeredMagnitude_one]
  · by_cases one : child.val = 1 <;>
      simp [rightDigit, zero, one, productionGlobalParams,
        Centered.centeredMagnitude_zero, centeredMagnitude_one]

theorem leftAssignmentsShort (child : Child) :
    assignmentNormBounded productionGlobalParams.b
      (leftAssignments child) := by
  intro _column
  exact leftDigitsShort child

theorem rightAssignmentsShort (child : Child) :
    assignmentNormBounded productionGlobalParams.b
      (rightAssignments child) := by
  intro _column
  exact rightDigitsShort child

/-- Exact production-field radix alias at one coefficient. -/
theorem recomposedScalar_eq :
    PiDECAlgebra.Radix.recomposeScalar leftDigit =
      PiDECAlgebra.Radix.recomposeScalar rightDigit := by
  decide

/-- The scalar alias holds at every one of the 270 typed coordinates. -/
theorem recomposedAssignments_eq :
    PiDECAlgebra.Radix.recomposeAssignment leftAssignments =
      PiDECAlgebra.Radix.recomposeAssignment rightAssignments := by
  funext column
  simp only [PiDECAlgebra.Radix.recomposeAssignment_apply]
  exact recomposedScalar_eq

/-- The child assignment families themselves are visibly different. -/
theorem assignments_ne : leftAssignments ≠ rightAssignments := by
  intro equal
  have atChild := congrFun equal childZero
  have atColumn := congrFun atChild ⟨0, by decide⟩
  have impossible : (1 : F) = -1 := by
    simpa [leftAssignments, rightAssignments, leftDigit, rightDigit,
      childZero] using atColumn
  exact (by decide : (1 : F) ≠ -1) impossible

/-- Empty-row typed Ajtai key. It is a model fixture, not a binding claim. -/
def key : PiRLCAlgebra.Commitment.Key Shape 0 :=
  fun row => Fin.elim0 row

def commit := PiRLCAlgebra.Commitment.commit key

def algebra := PiDECAlgebra.Algebra.concrete key

def point : Phi81Relation.Point Shape where
  coordinates := List.replicate
    (Phi81Relation.Shape.rowVariables Shape) K.zero
  dimension := by simp

def childStatement
    (assignments : Child -> Assignment)
    (child : Child) : Statement :=
  canonicalCEStatement commit
    FPrimeCarrier270.PaddedIdentityEvaluation.system .fresh point
    (assignments child)

def leftChildren : Child -> Statement := childStatement leftAssignments

def rightChildren : Child -> Statement := childStatement rightAssignments

theorem leftChildrenValid (child : Child) :
    CE.Holds (relationSemantics commit) productionGlobalParams
      (leftChildren child) (leftAssignments child) := by
  exact canonicalCE_holds commit productionGlobalParams
    FPrimeCarrier270.PaddedIdentityEvaluation.system .fresh point
    (leftAssignments child) (leftAssignmentsShort child)

theorem rightChildrenValid (child : Child) :
    CE.Holds (relationSemantics commit) productionGlobalParams
      (rightChildren child) (rightAssignments child) := by
  exact canonicalCE_holds commit productionGlobalParams
    FPrimeCarrier270.PaddedIdentityEvaluation.system .fresh point
    (rightAssignments child) (rightAssignmentsShort child)

theorem leftCompatible :
    CanonicalParent.Compatible kPositive leftChildren := by
  exact {
    childFresh := fun _ => rfl
    sameStructure := fun _ => rfl
    samePoint := fun _ => rfl
  }

theorem rightCompatible :
    CanonicalParent.Compatible kPositive rightChildren := by
  exact {
    childFresh := fun _ => rfl
    sameStructure := fun _ => rfl
    samePoint := fun _ => rfl
  }

def leftParent : Statement :=
  CanonicalParent.parent algebra kPositive leftChildren

def rightParent : Statement :=
  CanonicalParent.parent algebra kPositive rightChildren

theorem leftParentValid :
    CE.Holds (relationSemantics commit) productionGlobalParams leftParent
      (PiDECAlgebra.Radix.recomposeAssignment leftAssignments) := by
  exact CanonicalParent.holds_of_children
    (relationSemantics commit) productionGlobalParams algebra kPositive
    leftChildren leftCompatible leftAssignments leftChildrenValid

theorem rightParentValid :
    CE.Holds (relationSemantics commit) productionGlobalParams rightParent
      (PiDECAlgebra.Radix.recomposeAssignment rightAssignments) := by
  exact CanonicalParent.holds_of_children
    (relationSemantics commit) productionGlobalParams algebra kPositive
    rightChildren rightCompatible rightAssignments rightChildrenValid

private theorem statements_eq_of_same_opening
    {left right : Statement}
    {assignment : Assignment}
    (leftValid : CE.Holds (relationSemantics commit) productionGlobalParams
      left assignment)
    (rightValid : CE.Holds (relationSemantics commit) productionGlobalParams
      right assignment)
    (sameStructure : left.constraintSystem = right.constraintSystem)
    (samePoint : left.point = right.point)
    (sameStage : left.stage = right.stage) :
    left = right := by
  have sameCommitment : left.commitment = right.commitment :=
    leftValid.1.1.symm.trans rightValid.1.1
  have samePublicInput : left.publicInput = right.publicInput :=
    leftValid.1.2.1.symm.trans rightValid.1.2.1
  have sameEvaluations : left.evaluations = right.evaluations := by
    calc
      left.evaluations =
          (relationSemantics commit).evaluations left.constraintSystem
            assignment left.point := leftValid.2.2.symm
      _ = (relationSemantics commit).evaluations right.constraintSystem
            assignment right.point := by rw [sameStructure, samePoint]
      _ = right.evaluations := rightValid.2.2
  rcases left with ⟨_, _, _, _, _, _⟩
  rcases right with ⟨_, _, _, _, _, _⟩
  simp_all

/-- Both distinct child vectors compute the exact same parent statement. -/
theorem parents_eq : leftParent = rightParent := by
  apply statements_eq_of_same_opening leftParentValid
  · rw [recomposedAssignments_eq]
    exact rightParentValid
  · change FPrimeCarrier270.PaddedIdentityEvaluation.system =
      FPrimeCarrier270.PaddedIdentityEvaluation.system
    rfl
  · change point = point
    rfl
  · change NormStage.combined = NormStage.combined
    rfl

/-- The public child statement families differ at their first public
coordinate, independently of the empty commitment fixture. -/
theorem children_ne : leftChildren ≠ rightChildren := by
  intro equal
  have atChild := congrFun equal childZero
  have atPublic := congrArg
    (fun statement => statement.publicInput firstPublicColumn) atChild
  have impossible : (1 : F) = -1 := by
    simpa [leftChildren, rightChildren, childStatement, leftAssignments,
      rightAssignments, leftDigit, rightDigit,
      Phi81Relation.projectPublicInput, Phi81Relation.Shape.publicColumn,
      childZero, firstPublicColumn] using atPublic
  exact (by decide : (1 : F) ≠ -1) impossible

def parent : Statement := leftParent

theorem leftAccepted :
    PiDEC.Accepted algebra {
      parent := parent
      children := leftChildren
    } := by
  exact CanonicalParent.accepted_of_compatible algebra kPositive leftChildren
    leftCompatible

theorem rightAccepted :
    PiDEC.Accepted algebra {
      parent := parent
      children := rightChildren
    } := by
  simpa [parent, parents_eq] using
    (CanonicalParent.accepted_of_compatible algebra kPositive rightChildren
      rightCompatible)

end Fixture

/-- Exact production-profile witness that one public parent authorizes two
distinct valid child vectors under strict PiDEC recomposition. -/
structure Witness where
  parent : Fixture.Statement
  leftChildren : Fixture.Child -> Fixture.Statement
  rightChildren : Fixture.Child -> Fixture.Statement
  different : leftChildren ≠ rightChildren
  leftAccepted : PiDEC.Accepted Fixture.algebra {
    parent := parent
    children := leftChildren
  }
  rightAccepted : PiDEC.Accepted Fixture.algebra {
    parent := parent
    children := rightChildren
  }
  leftAssignments : Fixture.Child -> Fixture.Assignment
  rightAssignments : Fixture.Child -> Fixture.Assignment
  leftValid : forall child,
    CE.Holds (relationSemantics Fixture.commit) productionGlobalParams
      (leftChildren child) (leftAssignments child)
  rightValid : forall child,
    CE.Holds (relationSemantics Fixture.commit) productionGlobalParams
      (rightChildren child) (rightAssignments child)

namespace Witness

/-- Any digest or handle computed from only the common parent is unchanged by
the exact child substitution. This is equality of inputs, not a hash
collision. -/
theorem sameHandle
    (witness : Witness)
    {Digest : Type}
    (handle : Fixture.Statement -> Digest) :
    handle witness.parent = handle witness.parent := rfl

end Witness

def witness : Witness := {
  parent := Fixture.parent
  leftChildren := Fixture.leftChildren
  rightChildren := Fixture.rightChildren
  different := Fixture.children_ne
  leftAccepted := Fixture.leftAccepted
  rightAccepted := Fixture.rightAccepted
  leftAssignments := Fixture.leftAssignments
  rightAssignments := Fixture.rightAssignments
  leftValid := Fixture.leftChildrenValid
  rightValid := Fixture.rightChildrenValid
}

/-- The authority claim made by a parent-only recursive handle: equal handles
for two strict PiDEC-accepted families under the same parent should force the
children to be equal. -/
def ParentOnlyHandleBinds
    {Digest : Type}
    (handle : Fixture.Statement -> Digest) : Prop :=
  forall
      (parent : Fixture.Statement)
      (leftChildren rightChildren : Fixture.Child -> Fixture.Statement),
    PiDEC.Accepted Fixture.algebra {
      parent := parent
      children := leftChildren
    } ->
    PiDEC.Accepted Fixture.algebra {
      parent := parent
      children := rightChildren
    } ->
    handle parent = handle parent ->
    leftChildren = rightChildren

/-- No digest or other handle computed from only the checked parent can bind
the production child vector. The refutation uses identical parent inputs, so
it assumes no hash collision and no cryptographic weakness. -/
theorem no_parentOnlyHandle_binds
    {Digest : Type}
    (handle : Fixture.Statement -> Digest) :
    ¬ParentOnlyHandleBinds handle := by
  intro binds
  exact witness.different
    (binds witness.parent witness.leftChildren witness.rightChildren
      witness.leftAccepted witness.rightAccepted rfl)

end Nightstream.SuperNeo.Folding.PiDEC.Necessity.ProductionChildSubstitution
