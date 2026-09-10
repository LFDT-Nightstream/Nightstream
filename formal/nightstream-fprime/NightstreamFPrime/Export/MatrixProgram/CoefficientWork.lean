import NightstreamFPrime.Export.MatrixProgram.SparseWork
import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.StoredWitnessCheckEntries

/-!
Counted Phi81 coefficient expansion of one already constructed sparse row.
All 54 source lanes use duplicate-aware stored-entry scans. The logical-width
guard applies to each source lane, including when the queried assignment
column lies in the carrier completion suffix. Row generation is a separate
caller cost. No matrix function is executed by this program.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Export.MatrixProgram.CoefficientWork

open _root_.NightstreamFPrime.Spec
open _root_.NightstreamFPrime.Layout.ProductionRelation
open _root_.NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open ConcreteCarrier PaperLinearAlgebra MatrixCoefficientSource
open _root_.NightstreamFPrime.Spec.Folding.PiRLC.PaperForkExtractionWork (Result)
open StoredWitnessCheckEntries (kernelWeight kernelWork)

private def sum (term : Nat → Result F) : Nat → Result F
  | 0 => ⟨0, 3⟩
  | count + 1 =>
      let previous := sum term count
      let current := term count
      ⟨previous.value + current.value, previous.work + current.work + 8⟩

private theorem sum_value (term : Nat → Result F) (count : Nat) :
    (sum term count).value = sumRange baseOps count (fun index => (term index).value) := by
  induction count with
  | zero => rfl
  | succ count ih => simpa only [sum, sumRange, ih, baseOps]

private theorem sum_work_le (term : Nat → Result F) (count bound : Nat)
    (bounded : ∀ index, index < count → (term index).work ≤ bound) :
    (sum term count).work ≤ count * (bound + 8) + 3 := by
  induction count with
  | zero => simp [sum]
  | succ count ih =>
      have previous := ih (fun index live => bounded index (Nat.lt_trans live (Nat.lt_succ_self count)))
      have current := bounded count (Nat.lt_succ_self count)
      simp only [sum, Nat.add_mul, Nat.one_mul]
      omega

/-- Live terms charge both comparisons/branches, flat-index arithmetic,
two bounded-index constructors, two calls/value reads, multiplication and
Result construction (14). Out-of-range terms charge their executed prefix,
field-zero construction, and Result construction. -/
private def term {columns : Nat} (form : SparseForm columns)
    (output assignment : Fin ringDegree) (block lane : Nat) : Result F :=
  if live : lane < ringDegree then
    let flat := block * ringDegree + lane
    if within : flat < columns then
      let coefficient := SparseWork.coefficient form ⟨flat, within⟩
      let weight := kernelWeight output ⟨lane, live⟩ assignment
      ⟨coefficient.value * weight.value, coefficient.work + weight.work + 14⟩
    else ⟨0, 8⟩
  else ⟨0, 4⟩

private theorem term_work_le {columns : Nat} (form : SparseForm columns)
    (output assignment : Fin ringDegree) (block lane : Nat) :
    (term form output assignment block lane).work ≤
      10 * form.entries.length + 8 + kernelWork + 14 := by
  dsimp only [term]
  split
  · rename_i live
    split
    · rename_i within
      have coefficient := SparseWork.coefficient_work_le form ⟨_, within⟩
      have weight := StoredWitnessCheckEntries.kernelWeight_work_le output ⟨lane, live⟩ assignment
      dsimp only
      omega
    · dsimp only
      omega
  · dsimp only
    omega

/-- Decode the assignment column once and sum the fixed 54 source lanes.
The wrapper charges its column read, division/remainder, Fin construction,
term closure, sum call, result-value read and Result construction (8).
Each sum step charges dispatch/descent, two calls/value reads, addition and
Result construction (8). -/
def coefficient {logicalWidth : Nat} (form : SparseForm logicalWidth)
    (output : Fin ringDegree)
    (column : Fin (Phi81CarrierLayout.carrierWidth logicalWidth)) : Result F :=
  let flat := column.val
  let block := flat / ringDegree
  let assignment : Fin ringDegree := ⟨flat % ringDegree, Nat.mod_lt _ (by decide)⟩
  let result := sum (term form output assignment block) ringDegree
  ⟨result.value, result.work + 8⟩

def workBound (entries : Nat) : Nat :=
  ringDegree * (10 * entries + 8 + kernelWork + 14 + 8) + 3 + 8

theorem coefficient_work_le {logicalWidth : Nat} (form : SparseForm logicalWidth)
    (output : Fin ringDegree)
    (column : Fin (Phi81CarrierLayout.carrierWidth logicalWidth)) :
    (coefficient form output column).work ≤ workBound form.entries.length := by
  unfold coefficient workBound
  exact Nat.add_le_add_right
    (sum_work_le _ _ _ (fun lane _ => term_work_le form output _ _ lane)) 8

private theorem term_value {arity logicalWidth : Nat}
    (form : SparseForm logicalWidth) (matrix : BooleanMatrix F arity logicalWidth)
    (vertex : BooleanVertex arity)
    (same : ∀ column, form.coefficient column = matrix vertex column)
    (output : Fin ringDegree)
    (column : Fin (Phi81CarrierLayout.carrierWidth logicalWidth))
    (lane : Nat) (live : lane < ringDegree) :
    (term form output (Phi81ColumnLayout.decode column).2
      (column.val / ringDegree) lane).value =
      baseOps.mul
        (match Phi81ColumnLayout.encode? (Phi81ColumnLayout.decode column).1 ⟨lane, live⟩ with
        | none => baseOps.zero
        | some source => Phi81CarrierLayout.extendMatrix (0 : F) matrix vertex source)
        (Phi81CoefficientKernel.phi81Kernel.weight output ⟨lane, live⟩
          (Phi81ColumnLayout.decode column).2) := by
  have zeroMul (value : F) : (0 : F) * value = 0 :=
    (baseLaws.mul_comm 0 value).trans (baseLaws.mul_zero value)
  by_cases within : column.val / ringDegree * ringDegree + lane < logicalWidth
  · have carrierBound : column.val / ringDegree * ringDegree + lane <
        Phi81CarrierLayout.carrierWidth logicalWidth :=
      Nat.lt_of_lt_of_le within (Phi81CarrierLayout.logicalWidth_le_carrierWidth logicalWidth)
    simp [term, live, within, carrierBound, SparseWork.coefficient_value, same,
      StoredWitnessCheckEntries.kernelWeight_value, Phi81ColumnLayout.encode?,
      Phi81ColumnLayout.flatIndex, Phi81ColumnLayout.decode,
      Phi81CarrierLayout.extendMatrix, Phi81CarrierLayout.logicalColumn?, baseOps]
  · by_cases carrierBound : column.val / ringDegree * ringDegree + lane <
        Phi81CarrierLayout.carrierWidth logicalWidth <;>
      simp [term, live, within, carrierBound, Phi81ColumnLayout.encode?,
        Phi81ColumnLayout.flatIndex, Phi81ColumnLayout.decode,
        Phi81CarrierLayout.extendMatrix, Phi81CarrierLayout.logicalColumn?, baseOps, zeroMul]

/-- A semantic row equality is used only by the proof. The executable input
is the stored form, whose generation clock remains with its caller. -/
theorem coefficient_value (arity freshCount runningCount matrixCount logicalWidth : Nat)
    (matrices : Fin matrixCount → BooleanMatrix F arity logicalWidth)
    (polynomial : CCSResidualTable.ConstraintPolynomial F matrixCount)
    (matrix : Fin matrixCount) (vertex : BooleanVertex arity)
    (form : SparseForm logicalWidth)
    (same : ∀ column, form.coefficient column = matrices matrix vertex column)
    (output : Fin ringDegree)
    (column : Fin (Phi81CarrierLayout.carrierWidth logicalWidth)) :
    (coefficient form output column).value =
      (Phi81MatrixSource.source arity freshCount runningCount matrixCount logicalWidth matrices polynomial).coefficientMatrix
        baseOps matrix output vertex column := by
  simp only [coefficient, sum_value, MatrixSource.coefficientMatrix, MatrixSource.coefficientMatrixOf,
    Phi81MatrixSource.source, Phi81MatrixSource.phi81Shape, Phi81CarrierLayout.layout,
    Phi81ColumnLayout.layout, MatrixSource.paddedEntry]
  apply sumRange_congr baseOps ringDegree
  intro lane live
  simp only [dif_pos live]
  have termEq := term_value form (matrices matrix) vertex same output column lane live
  cases encoded : Phi81ColumnLayout.encode? (Phi81ColumnLayout.decode column).1 ⟨lane, live⟩ <;>
    simp only [encoded] at termEq ⊢ <;>
    simpa only [Phi81ColumnLayout.decode] using termEq

end NightstreamFPrime.Export.MatrixProgram.CoefficientWork
