import NightstreamFPrime.Spec.Folding.Nifs.StoredAssignmentArithmetic
import Mathlib.Tactic.SplitIfs

/-!
Materialized Phi81 arithmetic for extraction and dense commitment checks.
Every product constructs all 54 coefficients. Counts cover array reads,
field and index operations, branches, constructors, and returns. Semantic
coefficient functions are views of stored arrays, not executable primitives.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Phi81Relation.EvaluationHomomorphism.StoredRingArithmetic

open NightstreamFPrime.Spec
open _root_.NightstreamFPrime.Spec.Folding.Nifs.StoredAssignmentArithmetic
  (build build_value build_work_le)
open _root_.NightstreamFPrime.Spec.Folding.PiRLC.PaperForkExtractionWork (Result)

abbrev StoredRing := Vector F ringDegree

private def sum (term : Nat → Result F) : Nat → Result F
  | 0 => ⟨0, 2⟩
  | count + 1 =>
      let prior := sum term count
      let current := term count
      ⟨prior.value + current.value, prior.work + current.work + 4⟩

private theorem sum_work_le (term : Nat → Result F) (count bound : Nat)
    (bounded : ∀ index, index < count → (term index).work ≤ bound) :
    (sum term count).work ≤ count * (bound + 4) + 2 := by
  induction count with
  | zero => simp [sum]
  | succ count ih =>
      have prior := ih (fun index live => bounded index (Nat.lt_trans live (Nat.lt_succ_self count)))
      have current := bounded count (Nat.lt_succ_self count)
      simp only [sum, Nat.add_mul, Nat.one_mul]
      omega

/-- A live term charges three comparisons and branches, one subtraction,
two bounded-index constructors, two array projections and reads, one field
product, and one return. Each failed branch charges its executed prefix. -/
private def rawTerm (left right : StoredRing) (degree index : Nat) : Result F :=
  if live : index < ringDegree then
    if index ≤ degree then
      let remaining := degree - index
      if within : remaining < ringDegree then
        ⟨left.get ⟨index, live⟩ * right.get ⟨remaining, within⟩, 15⟩
      else ⟨0, 8⟩
    else ⟨0, 5⟩
  else ⟨0, 3⟩

private theorem rawTerm_value (left right : StoredRing) (degree index : Nat)
    (live : index < ringDegree) :
    (rawTerm left right degree index).value =
      if index ≤ degree ∧ degree - index < ringDegree then
        ringFCoeff left.get index * ringFCoeff right.get (degree - index)
      else 0 := by
  by_cases below : index ≤ degree <;> by_cases remaining : degree - index < ringDegree <;>
    simp [rawTerm, live, below, remaining, ringFCoeff]

private theorem rawTerm_work_le (left right : StoredRing) (degree index : Nat) :
    (rawTerm left right degree index).work ≤ 15 := by
  dsimp only [rawTerm]
  split_ifs <;> dsimp only <;> decide

private def raw (left right : StoredRing) (degree : Nat) : Result F :=
  let result := sum (rawTerm left right degree) ringDegree
  ⟨result.value, result.work + 2⟩

private theorem raw_prefix_value (left right : StoredRing) (degree count : Nat)
    (bounded : count ≤ ringDegree) :
    (sum (rawTerm left right degree) count).value =
      (List.range count).foldl (fun accumulated index =>
        if index ≤ degree ∧ degree - index < ringDegree then
          accumulated + ringFCoeff left.get index * ringFCoeff right.get (degree - index)
        else accumulated) 0 := by
  induction count with
  | zero => rfl
  | succ count ih =>
      have earlier := ih (Nat.le_trans (Nat.le_succ count) bounded)
      have live : count < ringDegree := Nat.lt_of_lt_of_le (Nat.lt_succ_self count) bounded
      simp only [sum, List.range_succ, List.foldl_append, List.foldl_cons, List.foldl_nil,
        earlier, rawTerm_value left right degree count live]
      split_ifs
      · rfl
      · exact _root_.NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ConcreteCarrier.baseLaws.add_zero _

private theorem raw_value (left right : StoredRing) (degree : Nat) :
    (raw left right degree).value = rawMulCoeffF left.get right.get degree :=
  raw_prefix_value left right degree ringDegree (Nat.le_refl _)

private def rawWork : Nat := ringDegree * (15 + 4) + 2 + 2

private theorem raw_work_le (left right : StoredRing) (degree : Nat) :
    (raw left right degree).work ≤ rawWork :=
  Nat.add_le_add_right
    (sum_work_le _ _ 15 (fun index _ => rawTerm_work_le left right degree index)) 2

private def coefficient (left right : StoredRing) (output : Fin ringDegree) : Result F :=
  let degree := output.val
  let low := raw left right degree
  let folded := if degree < ringMiddleDegree then
      let value := raw left right (degree + ringDegree)
      (⟨value.value, value.work + 4⟩ : Result F)
    else
      let value := raw left right (degree + ringMiddleDegree)
      ⟨value.value, value.work + 4⟩
  let twiceDegree := degree + 81
  let twice := if twiceDegree ≤ 106 then
      let value := raw left right twiceDegree
      (⟨value.value, value.work + 4⟩ : Result F)
    else ⟨0, 4⟩
  ⟨low.value - folded.value + twice.value, low.work + folded.work + twice.work + 4⟩

private theorem coefficient_value (left right : StoredRing) (output : Fin ringDegree) :
    (coefficient left right output).value = ringFMul left.get right.get output := by
  dsimp only [coefficient, ringFMul]
  split_ifs <;> simp only [raw_value]

private theorem coefficient_work_le (left right : StoredRing) (output : Fin ringDegree) :
    (coefficient left right output).work ≤ 3 * rawWork + 12 := by
  have low := raw_work_le left right output.val
  have lowFold := raw_work_le left right (output.val + ringDegree)
  have highFold := raw_work_le left right (output.val + ringMiddleDegree)
  have twice := raw_work_le left right (output.val + 81)
  dsimp only [coefficient]
  split_ifs <;> dsimp only <;> omega

/-- The array builder charges each coefficient, allocation, push, and return.
Two more operations account for the coordinate closure and outer return. -/
def multiply (left right : StoredRing) : Result StoredRing :=
  let result := build (coefficient left right)
  ⟨result.value, result.work + 2⟩

def multiplyWork : Nat := ringDegree + 1 + ringDegree * ((3 * rawWork + 12) + 2) + 1 + 2

theorem multiply_value (left right : StoredRing) :
    (multiply left right).value.get = ringFMul left.get right.get := by
  change _root_.NightstreamFPrime.Spec.Folding.Nifs.StoredAssignmentArithmetic.view
    (build (coefficient left right)).value = _
  rw [build_value]
  funext output
  exact coefficient_value left right output

theorem multiply_work_le (left right : StoredRing) :
    (multiply left right).work ≤ multiplyWork :=
  Nat.add_le_add_right
    (build_work_le _ _ (fun output => coefficient_work_le left right output)) 2

/-- Build the identity array; each coefficient reads and compares its index,
branches, and returns. The outer closure and return add two operations. -/
def one (_ : Unit) : Result StoredRing :=
  let result := build (fun index : Fin ringDegree =>
    (⟨if index.val = 0 then 1 else 0, 4⟩ : Result F))
  ⟨result.value, result.work + 2⟩

def oneWork : Nat := ringDegree + 1 + ringDegree * (4 + 2) + 1 + 2

theorem one_value : (one ()).value.get = ringFOne := by
  change _root_.NightstreamFPrime.Spec.Folding.Nifs.StoredAssignmentArithmetic.view (build _).value = _
  rw [build_value]
  rfl

theorem one_work_le : (one ()).work ≤ oneWork :=
  Nat.add_le_add_right (build_work_le _ _ (fun _ => Nat.le_refl 4)) 2

/-- Each coordinate reads both stored arrays and adds their field values.
The read count includes each vector's array projection. -/
def add (left right : StoredRing) : Result StoredRing :=
  let result := build (fun index => (⟨left.get index + right.get index, 6⟩ : Result F))
  ⟨result.value, result.work + 2⟩

def addWork : Nat := ringDegree + 1 + ringDegree * (6 + 2) + 1 + 2

theorem add_value (left right : StoredRing) :
    (add left right).value.get = ringFAdd left.get right.get := by
  change _root_.NightstreamFPrime.Spec.Folding.Nifs.StoredAssignmentArithmetic.view (build _).value = _
  rw [build_value]
  rfl

theorem add_work_le (left right : StoredRing) : (add left right).work ≤ addWork :=
  Nat.add_le_add_right (build_work_le _ _ (fun _ => Nat.le_refl 6)) 2

end NightstreamFPrime.Spec.Phi81Relation.EvaluationHomomorphism.StoredRingArithmetic
