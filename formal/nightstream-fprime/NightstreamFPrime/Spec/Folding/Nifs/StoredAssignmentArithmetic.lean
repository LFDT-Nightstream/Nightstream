import NightstreamFPrime.Spec.Folding.PiRLC.PaperForkExtractionWork
import NightstreamFPrime.Spec.Phi81Relation.EvaluationHomomorphism.PiDEC
import Init.Data.Vector.OfFn

/-!
Stored assignment arithmetic for B.3 subtraction and B.4 recomposition.
The array builder charges its invoked coordinate program, loop branch and
push, initial capacity, and final return. Field operations have fixed-size
Goldilocks operands. Function-valued semantic assignments are only views.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Folding.Nifs.StoredAssignmentArithmetic

open NightstreamFPrime.Spec
open Phi81Relation.EvaluationHomomorphism
open _root_.NightstreamFPrime.Spec.Folding.PiRLC.PaperForkExtractionWork (Result)

abbrev StoredAssignment (width : Nat) := Vector F width

def view {width : Nat} (stored : StoredAssignment width) : Fin width → F := stored.get

private def coordinateAction {width : Nat} (coordinate : Fin width → Result F)
    (index : Fin width) : StateM Nat F := fun work =>
  let result := coordinate index
  (result.value, work + result.work + 2)

/-- A tail-recursive array traversal, with no intermediate list of indices. -/
def build {width : Nat} (coordinate : Fin width → Result F) : Result (StoredAssignment width) :=
  let result := (Vector.ofFnM (coordinateAction coordinate)).run (width + 1)
  ⟨result.1, result.2 + 1⟩

private theorem build_state_value {width : Nat} (coordinate : Fin width → Result F)
    (initial : Nat) :
    ((Vector.ofFnM (coordinateAction coordinate)).run initial).1 =
      Vector.ofFn (fun index => (coordinate index).value) := by
  induction width generalizing initial with
  | zero => rw [Vector.ofFnM_zero]; rfl
  | succ width ih =>
      rw [Vector.ofFnM_succ, Vector.ofFn_succ]
      simp only [StateT.run_bind, StateT.run_pure]
      change
        (((Vector.ofFnM (coordinateAction (fun index => coordinate index.castSucc))).run initial).1).push
          (coordinate (Fin.last width)).value = _
      rw [ih]
      rfl

private theorem build_state_work {width : Nat} (coordinate : Fin width → Result F)
    (bound : Nat) (bounded : ∀ index, (coordinate index).work ≤ bound) (initial : Nat) :
    ((Vector.ofFnM (coordinateAction coordinate)).run initial).2 ≤ initial + width * (bound + 2) := by
  induction width generalizing initial with
  | zero =>
      rw [Vector.ofFnM_zero]
      change initial ≤ initial + 0 * (bound + 2)
      simp only [Nat.zero_mul, Nat.add_zero, Nat.le_refl]
  | succ width ih =>
      rw [Vector.ofFnM_succ]
      simp only [StateT.run_bind, StateT.run_pure]
      change
        ((Vector.ofFnM (coordinateAction (fun index => coordinate index.castSucc))).run initial).2 +
          (coordinate (Fin.last width)).work + 2 ≤ _
      have previous := ih (fun index => coordinate index.castSucc)
        (fun index => bounded index.castSucc) initial
      have last := bounded (Fin.last width)
      rw [Nat.add_mul]
      omega

theorem build_value {width : Nat} (coordinate : Fin width → Result F) :
    view (build coordinate).value = fun index => (coordinate index).value := by
  funext index
  simp [view, build, build_state_value, Vector.get, Vector.ofFn]

theorem build_work_le {width : Nat} (coordinate : Fin width → Result F)
    (bound : Nat) (bounded : ∀ index, (coordinate index).work ≤ bound) :
    (build coordinate).work ≤ width + 1 + width * (bound + 2) + 1 := by
  exact Nat.add_le_add_right (build_state_work coordinate bound bounded (width + 1)) 1

/-- Two array reads, one field subtraction, and one result return. -/
def subtract {width : Nat} (left right : StoredAssignment width) : Result (StoredAssignment width) :=
  build fun column =>
    let leftRead : Result F := ⟨left.get column, 1⟩
    let rightRead : Result F := ⟨right.get column, 1⟩
    ⟨leftRead.value - rightRead.value, leftRead.work + rightRead.work + 1 + 1⟩

theorem subtract_value {width : Nat} (left right : StoredAssignment width) :
    view (subtract left right).value = fun column => view left column - view right column :=
  build_value _

theorem subtract_work_le {width : Nat} (left right : StoredAssignment width) :
    (subtract left right).work ≤ width + 1 + width * ((1 + 1 + 1 + 1) + 2) + 1 :=
  build_work_le _ _ (fun _ => Nat.le_refl _)

private def sumCoordinates : {count : Nat} → (Fin count → Result F) → Result F
  | 0, _ => ⟨0, 1⟩
  | _ + 1, term =>
      let head := term 0
      let tail := sumCoordinates (fun index => term index.succ)
      ⟨head.value + tail.value, head.work + tail.work + 1 + 1⟩

private theorem sumCoordinates_work_le (bound : Nat) : ∀ {count : Nat}
    (term : Fin count → Result F) (_bounded : ∀ index, (term index).work ≤ bound),
    (sumCoordinates term).work ≤ count * (bound + 2) + 1
  | 0, _, _ => by simp [sumCoordinates]
  | count + 1, term, bounded => by
      have head := bounded 0
      have tail := sumCoordinates_work_le bound (fun index => term index.succ)
        (fun index => bounded index.succ)
      change (term 0).work + (sumCoordinates (fun index => term index.succ)).work + 1 + 1 ≤ _
      rw [Nat.add_mul]
      omega

private theorem sumCoordinates_value {width : Nat} : ∀ {count : Nat}
    (weights : Fin count → F) (assignments : Fin count → Fin width → F)
    (term : Fin count → Result F) (column : Fin width)
    (_values : ∀ index, (term index).value = weights index * assignments index column),
    (sumCoordinates term).value = BaseLinear.Raw.combineAssignments weights assignments column
  | 0, _, _, _, _, _ => rfl
  | _ + 1, weights, assignments, term, column, values => by
      change (term 0).value + (sumCoordinates (fun index => term index.succ)).value = _
      rw [values 0, sumCoordinates_value (fun index => weights index.succ)
        (fun index => assignments index.succ) (fun index => term index.succ) column
        (fun index => values index.succ)]
      rfl

/-- Each product reads the stored weight, source row, and coefficient. The
source recursion has the protocol's child count; only the array builder
traverses assignment width. -/
def combine {width count : Nat} (weights : Vector F count)
    (assignments : Vector (StoredAssignment width) count) : Result (StoredAssignment width) :=
  build fun column => sumCoordinates fun source =>
    let weight : Result F := ⟨weights.get source, 1⟩
    let row : Result (StoredAssignment width) := ⟨assignments.get source, 1⟩
    let field : Result F := ⟨row.value.get column, 1⟩
    ⟨weight.value * field.value, weight.work + row.work + field.work + 1 + 1⟩

theorem combine_value {width count : Nat} (weights : Vector F count)
    (assignments : Vector (StoredAssignment width) count) :
    view (combine weights assignments).value =
      BaseLinear.Raw.combineAssignments weights.get (fun source => view (assignments.get source)) := by
  rw [combine, build_value]
  funext column
  exact sumCoordinates_value _ _ _ _ (fun _ => rfl)

theorem combine_work_le {width count : Nat} (weights : Vector F count)
    (assignments : Vector (StoredAssignment width) count) :
    (combine weights assignments).work ≤
      width + 1 + width * (count * ((1 + 1 + 1 + 1 + 1) + 2) + 1 + 2) + 1 := by
  apply build_work_le
  intro column
  exact sumCoordinates_work_le _ _ (fun _ => Nat.le_refl _)

private def binaryPower : Nat → Result Nat
  | 0 => ⟨1, 1⟩
  | exponent + 1 =>
      let previous := binaryPower exponent
      ⟨previous.value * 2, previous.work + 1 + 1⟩

private theorem binaryPower_value (exponent : Nat) :
    (binaryPower exponent).value = 2 ^ exponent := by
  induction exponent with
  | zero => rfl
  | succ exponent ih => simpa only [binaryPower, Nat.pow_succ] using congrArg (· * 2) ih

private theorem binaryPower_work (exponent : Nat) :
    (binaryPower exponent).work = exponent * 2 + 1 := by
  induction exponent with
  | zero => rfl
  | succ exponent ih =>
      change (binaryPower exponent).work + 1 + 1 = _
      rw [ih, Nat.add_mul]

/-- The only call constructs the selected 16 binary weights. Each exponent
is below 16, so the natural-number operands remain below 2^16. -/
private def radixPowers (_ : Unit) : Result (Vector F productionGlobalParams.k) :=
  build fun index =>
    let power := binaryPower index.val
    ⟨⟨power.value % goldilocksModulus, Nat.mod_lt _ (by decide)⟩, power.work + 1 + 1⟩

private theorem radixPowers_value :
    view (radixPowers ()).value = PiDEC.radixWeight := by
  rw [radixPowers, build_value]
  funext index
  apply Fin.ext
  simp only [binaryPower_value, PiDEC.radixWeight, productionGlobalParams]

private theorem radixPowers_work_le :
    (radixPowers ()).work ≤ productionGlobalParams.k + 1 +
      productionGlobalParams.k * ((productionGlobalParams.k * 2 + 1 + 1 + 1) + 2) + 1 := by
  apply build_work_le
  intro index
  change (binaryPower index.val).work + 1 + 1 ≤ _
  rw [binaryPower_work]
  have indexBound := index.isLt
  omega

/-- B.4 recomposition computes its fixed weights and the whole stored
parent. No semantic assignment function is evaluated as a primitive. -/
def recompose {width : Nat} (assignments : Vector (StoredAssignment width) productionGlobalParams.k) :
    Result (StoredAssignment width) :=
  let weights := radixPowers ()
  let parent := combine weights.value assignments
  ⟨parent.value, weights.work + parent.work + 1⟩

theorem recompose_value {width : Nat}
    (assignments : Vector (StoredAssignment width) productionGlobalParams.k) :
    view (recompose assignments).value =
      PiDEC.Raw.recomposeAssignment (fun source => view (assignments.get source)) := by
  change view (combine (radixPowers ()).value assignments).value = _
  rw [combine_value, show (radixPowers ()).value.get = PiDEC.radixWeight from radixPowers_value]
  rfl

theorem recompose_work_le {width : Nat}
    (assignments : Vector (StoredAssignment width) productionGlobalParams.k) :
    (recompose assignments).work ≤
      (productionGlobalParams.k + 1 +
        productionGlobalParams.k * ((productionGlobalParams.k * 2 + 1 + 1 + 1) + 2) + 1) +
      (width + 1 + width * (productionGlobalParams.k * ((1 + 1 + 1 + 1 + 1) + 2) + 1 + 2) + 1) + 1 := by
  exact Nat.add_le_add_right
    (Nat.add_le_add radixPowers_work_le (combine_work_le (radixPowers ()).value assignments)) 1

end NightstreamFPrime.Spec.Folding.Nifs.StoredAssignmentArithmetic
