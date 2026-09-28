import NightstreamFPrime.Spec.Folding.Nifs.PaperWeakLaw
import NightstreamFPrime.Spec.Folding.PiRLC.PaperForkBinding

/-!
Compute the cross-difference used by SuperNeo v1.1 B.3 from two actual weak
endpoints. The program takes raw returned data, not a selected collision proof.
Its clock includes the seven primitive calls and the presence checks.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Folding.Nifs.BindingOutput

open PiRLC.PaperForkAlgebra PiRLC.PaperForkExtractionWork

variable {Scalar Assignment : Type*}

/-- Compute `deltaLeft * differenceRight - deltaRight * differenceLeft`.
No inverse, equality search, or commitment oracle is executed. -/
def crossDifference (program : Primitives Scalar Assignment)
    (leftBaseScalar leftForkScalar rightBaseScalar rightForkScalar : Scalar)
    (leftBase leftFork rightBase rightFork : Assignment) : Result Assignment :=
  let deltaLeft := program.scalarSub leftBaseScalar leftForkScalar
  let deltaRight := program.scalarSub rightBaseScalar rightForkScalar
  let differenceLeft := program.assignmentSub leftBase leftFork
  let differenceRight := program.assignmentSub rightBase rightFork
  let left := program.scalarAction deltaLeft.value differenceRight.value
  let right := program.scalarAction deltaRight.value differenceLeft.value
  let output := program.assignmentSub left.value right.value
  ⟨output.value, deltaLeft.work + deltaRight.work + differenceLeft.work +
    differenceRight.work + left.work + right.work + output.work + 1⟩

theorem crossDifference_value
    (ring : CommutativeRingOps Scalar) (module : ModuleOps Scalar Assignment)
    (program : Primitives Scalar Assignment) (correct : Correct ring module program)
    (leftBaseScalar leftForkScalar rightBaseScalar rightForkScalar : Scalar)
    (leftBase leftFork rightBase rightFork : Assignment) :
    (crossDifference program leftBaseScalar leftForkScalar rightBaseScalar rightForkScalar
      leftBase leftFork rightBase rightFork).value =
      module.sub
        (module.smul (ring.sub leftBaseScalar leftForkScalar) (module.sub rightBase rightFork))
        (module.smul (ring.sub rightBaseScalar rightForkScalar) (module.sub leftBase leftFork)) := by
  simp only [crossDifference, correct.assignmentSub, correct.scalarAction, correct.scalarSub]

/-- Two scalar differences, three assignment differences, two actions, and
one result return give this bound. It is derived from the executed calls. -/
def crossWork (bounds : PrimitiveBounds) : Nat :=
  2 * bounds.scalarSub + 3 * bounds.assignmentSub + 2 * bounds.scalarAction + 1

/-- Three coordinate budgets cover the seven cross-difference calls. -/
theorem crossWork_le_coordinateWork (bounds : PrimitiveBounds) :
    crossWork bounds ≤ 3 * bounds.coordinateWork := by
  unfold crossWork PrimitiveBounds.coordinateWork
  omega

theorem crossDifference_work_le
    (ring : CommutativeRingOps Scalar) (program : Primitives Scalar Assignment)
    (bounds : PrimitiveBounds) (bounded : Bounded ring program bounds)
    (leftBaseScalar leftForkScalar rightBaseScalar rightForkScalar : Scalar)
    (leftBase leftFork rightBase rightFork : Assignment) :
    (crossDifference program leftBaseScalar leftForkScalar rightBaseScalar rightForkScalar
      leftBase leftFork rightBase rightFork).work ≤ crossWork bounds := by
  have dl := bounded.scalarSub leftBaseScalar leftForkScalar
  have dr := bounded.scalarSub rightBaseScalar rightForkScalar
  have vl := bounded.assignmentSub leftBase leftFork
  have vr := bounded.assignmentSub rightBase rightFork
  have al := bounded.scalarAction (program.scalarSub leftBaseScalar leftForkScalar).value
    (program.assignmentSub rightBase rightFork).value
  have ar := bounded.scalarAction (program.scalarSub rightBaseScalar rightForkScalar).value
    (program.assignmentSub leftBase leftFork).value
  have output := bounded.assignmentSub
    (program.scalarAction (program.scalarSub leftBaseScalar leftForkScalar).value
      (program.assignmentSub rightBase rightFork).value).value
    (program.scalarAction (program.scalarSub rightBaseScalar rightForkScalar).value
      (program.assignmentSub leftBase leftFork).value).value
  dsimp only [crossDifference, crossWork]
  omega

variable {member : Scalar → Prop} {count : Nat}

/-- Read one coordinate from the literal endpoints. The driver charges four
endpoint/base-presence checks and, when both bases exist, two fork checks. -/
def candidate (program : Primitives Scalar Assignment)
    (left right : PaperWeakLaw.Endpoint (Fin count) {scalar // member scalar} Assignment)
    (coordinate : Fin count) : Result (Option Assignment) :=
  match left, right with
  | some (leftVector, some leftBase, leftOutputs),
      some (rightVector, some rightBase, rightOutputs) =>
      match (leftOutputs coordinate).2, (rightOutputs coordinate).2 with
      | some leftFork, some rightFork =>
          let result := crossDifference program
            (leftVector coordinate).val (leftOutputs coordinate).1.val
            (rightVector coordinate).val (rightOutputs coordinate).1.val
            leftBase leftFork rightBase rightFork
          ⟨some result.value, result.work + 6⟩
      | _, _ => ⟨none, 6⟩
  | _, _ => ⟨none, 4⟩

theorem candidate_work_le
    (ring : CommutativeRingOps Scalar) (program : Primitives Scalar Assignment)
    (bounds : PrimitiveBounds) (bounded : Bounded ring program bounds)
    (left right : PaperWeakLaw.Endpoint (Fin count) {scalar // member scalar} Assignment)
    (coordinate : Fin count) :
    (candidate program left right coordinate).work ≤ crossWork bounds + 6 := by
  unfold candidate
  split
  · split
    · exact Nat.add_le_add_right (crossDifference_work_le ring program bounds bounded _ _ _ _ _ _ _ _) 6
    · change 6 ≤ crossWork bounds + 6
      omega
  · change 4 ≤ crossWork bounds + 6
    omega

end NightstreamFPrime.Spec.Folding.Nifs.BindingOutput
