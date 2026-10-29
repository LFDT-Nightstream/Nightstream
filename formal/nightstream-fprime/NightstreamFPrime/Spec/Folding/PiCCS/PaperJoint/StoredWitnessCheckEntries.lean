import NightstreamFPrime.Spec.Folding.PiCCS.CanonicalRowLayout
import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.Phi81MatrixSource
import NightstreamFPrime.Spec.Folding.PiRLC.PaperForkExtractionWork
import Mathlib.Tactic.SplitIfs

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.StoredWitnessCheckEntries

open NightstreamFPrime.Spec
open ConcreteCarrier MatrixCoefficientSource PaperLinearAlgebra UnifiedSources
open _root_.NightstreamFPrime.Spec.Folding.PiRLC.PaperForkExtractionWork (Result)

def nativeBarEntry (output input : Fin ringDegree) : Result F :=
  let out := output.val
  let lane := input.val
  if out = 0 then
    if lane = 0 then ⟨1, 7⟩ else ⟨0, 7⟩
  else if out < ringMiddleDegree then
    if lane = ringMiddleDegree - out then ⟨-1, 11⟩
    else if lane = ringDegree - out then ⟨-1, 14⟩ else ⟨0, 13⟩
  else if lane = ringDegree - out then ⟨-1, 11⟩ else ⟨0, 10⟩

theorem nativeBarEntry_value (output input : Fin ringDegree) :
    (nativeBarEntry output input).value = Phi81CoefficientKernel.nativeBarEntry output input := by
  dsimp only [nativeBarEntry, Phi81CoefficientKernel.nativeBarEntry]
  split_ifs <;> simp_all

theorem nativeBarEntry_work_le (output input : Fin ringDegree) :
    (nativeBarEntry output input).work ≤ 14 := by
  dsimp only [nativeBarEntry]
  split_ifs <;> dsimp only <;> omega

private def sum (term : Nat → Result F) : Nat → Result F
  | 0 => ⟨0, 2⟩
  | count + 1 =>
      let prior := sum term count
      let current := term count
      ⟨prior.value + current.value, prior.work + current.work + 4⟩

private theorem sum_value (term : Nat → Result F) (count : Nat) :
    (sum term count).value = sumRange baseOps count (fun index => (term index).value) := by
  induction count with
  | zero => rfl
  | succ count ih => simp only [sum, sumRange, ih, baseOps]

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

private def monomial (assignment : Fin ringDegree) (degree : Nat) : Result F :=
  ⟨if degree = assignment.val then 1 else 0, 4⟩

private def rawTerm (row assignment : Fin ringDegree) (degree index : Nat) : Result F :=
  if live : index < ringDegree then
    if index ≤ degree then
      let remaining := degree - index
      if remaining < ringDegree then
        let left := nativeBarEntry ⟨index, live⟩ row
        let right := monomial assignment remaining
        ⟨left.value * right.value, left.work + right.work + 10⟩
      else ⟨0, 8⟩
    else ⟨0, 5⟩
  else ⟨0, 3⟩

private theorem rawTerm_value (row assignment : Fin ringDegree) (degree index : Nat)
    (live : index < ringDegree) :
    (rawTerm row assignment degree index).value =
      if index ≤ degree ∧ degree - index < ringDegree then
        ringFCoeff (Phi81CoefficientKernel.barBasis row) index *
          ringFCoeff (ringFMonomial assignment.val 1) (degree - index)
      else 0 := by
  by_cases below : index ≤ degree <;> by_cases remaining : degree - index < ringDegree <;>
    simp [rawTerm, live, below, remaining, nativeBarEntry_value, monomial,
      ringFCoeff, Phi81CoefficientKernel.barBasis, ringFMonomial]

private theorem rawTerm_work_le (row assignment : Fin ringDegree) (degree index : Nat) :
    (rawTerm row assignment degree index).work ≤ 28 := by
  dsimp only [rawTerm, monomial]
  split
  · rename_i live
    have bar := nativeBarEntry_work_le ⟨index, live⟩ row
    split_ifs <;> dsimp only <;> omega
  · dsimp only
    omega

private def raw (row assignment : Fin ringDegree) (degree : Nat) : Result F :=
  let value := sum (rawTerm row assignment degree) ringDegree
  ⟨value.value, value.work + 2⟩

private theorem raw_prefix_value (row assignment : Fin ringDegree) (degree : Nat)
    (count : Nat) (bounded : count ≤ ringDegree) :
    (sum (rawTerm row assignment degree) count).value =
      (List.range count).foldl (fun accumulated index =>
        if index ≤ degree ∧ degree - index < ringDegree then
          accumulated + ringFCoeff (Phi81CoefficientKernel.barBasis row) index *
            ringFCoeff (ringFMonomial assignment.val 1) (degree - index)
        else accumulated) 0 := by
  induction count with
  | zero => rfl
  | succ count ih =>
      have earlier := ih (Nat.le_trans (Nat.le_succ count) bounded)
      have live : count < ringDegree := Nat.lt_of_lt_of_le (Nat.lt_succ_self count) bounded
      simp only [sum, List.range_succ, List.foldl_append, List.foldl_cons, List.foldl_nil,
        earlier, rawTerm_value row assignment degree count live]
      split_ifs
      · rfl
      · exact baseLaws.add_zero _

private theorem raw_value (row assignment : Fin ringDegree) (degree : Nat) :
    (raw row assignment degree).value =
      rawMulCoeffF (Phi81CoefficientKernel.barBasis row) (ringFMonomial assignment.val 1) degree :=
  raw_prefix_value row assignment degree ringDegree (Nat.le_refl _)

private def rawWork : Nat := ringDegree * (28 + 4) + 2 + 2

private theorem raw_work_le (row assignment : Fin ringDegree) (degree : Nat) :
    (raw row assignment degree).work ≤ rawWork :=
  Nat.add_le_add_right
    (sum_work_le _ _ 28 (fun index _ => rawTerm_work_le row assignment degree index)) 2

def kernelWeight (output row assignment : Fin ringDegree) : Result F :=
  let degree := output.val
  let low := raw row assignment degree
  let folded := if degree < ringMiddleDegree then
      let value := raw row assignment (degree + ringDegree)
      (⟨value.value, value.work + 4⟩ : Result F)
    else
      let value := raw row assignment (degree + ringMiddleDegree)
      ⟨value.value, value.work + 4⟩
  let twiceDegree := degree + 81
  let twice := if twiceDegree ≤ 106 then
      let value := raw row assignment twiceDegree
      (⟨value.value, value.work + 4⟩ : Result F)
    else ⟨0, 4⟩
  ⟨low.value - folded.value + twice.value, low.work + folded.work + twice.work + 4⟩

def kernelWork : Nat := 3 * rawWork + 12

theorem kernelWeight_value (output row assignment : Fin ringDegree) :
    (kernelWeight output row assignment).value =
      Phi81CoefficientKernel.phi81Kernel.weight output row assignment := by
  dsimp only [kernelWeight, Phi81CoefficientKernel.phi81Kernel, ringFMul]
  simp only [show ringDegree + ringMiddleDegree = 81 from rfl, show 2 * (ringDegree - 1) = 106 from rfl]
  split_ifs <;> simp only [raw_value]

theorem kernelWeight_work_le (output row assignment : Fin ringDegree) :
    (kernelWeight output row assignment).work ≤ kernelWork := by
  have low := raw_work_le row assignment output.val
  have lowFold := raw_work_le row assignment (output.val + ringDegree)
  have highFold := raw_work_le row assignment (output.val + ringMiddleDegree)
  have twice := raw_work_le row assignment (output.val + 81)
  dsimp only [kernelWeight]
  unfold kernelWork
  split_ifs <;> dsimp only <;> omega

private def vertexIndex : {arity : Nat} → BooleanVertex arity → Result Nat
  | 0, .nil => ⟨0, 2⟩
  | _ + 1, .cons bit tail =>
      let prior := vertexIndex tail
      ⟨(if bit then 1 else 0) + 2 * prior.value, prior.work + 7⟩

private theorem vertexIndex_value {arity : Nat} (vertex : BooleanVertex arity) :
    (vertexIndex vertex).value = NumericBooleanDomain.index vertex := by
  induction vertex with
  | nil => rfl
  | @cons arity bit tail ih =>
      cases bit <;> simp only [vertexIndex, NumericBooleanDomain.index, ih] <;> rfl

private def padTerm (columns block selected : Nat) (output assignment : Fin ringDegree)
    (lane : Nat) : Result F :=
  if live : lane < ringDegree then
    let flat := block * ringDegree + lane
    if flat < columns then
      if flat = selected then
        let value := kernelWeight output ⟨lane, live⟩ assignment
        ⟨value.value, value.work + 10⟩
      else ⟨0, 9⟩
    else ⟨0, 7⟩
  else ⟨0, 3⟩

def padEntry {arity columns : Nat} (output : Fin ringDegree)
    (vertex : BooleanVertex arity) (column : Fin columns) : Result F :=
  let selected := vertexIndex vertex
  let flatColumn := column.val
  let block := flatColumn / ringDegree
  let assignment : Fin ringDegree :=
    ⟨flatColumn % ringDegree, Nat.mod_lt _ (by decide)⟩
  let value := sum (padTerm columns block selected.value output assignment) ringDegree
  ⟨value.value, selected.work + value.work + 6⟩

private theorem padTerm_value {arity columns : Nat}
    (covered : columns ≤ 2 ^ arity) (output : Fin ringDegree)
    (vertex : BooleanVertex arity) (column : Fin columns)
    (lane : Nat) (live : lane < ringDegree) :
    (padTerm columns (column.val / ringDegree) (NumericBooleanDomain.index vertex)
      output (Phi81ColumnLayout.decode column).2 lane).value =
      baseOps.mul
        (match Phi81ColumnLayout.encode? (Phi81ColumnLayout.decode column).1 ⟨lane, live⟩ with
        | none => baseOps.zero
        | some flat => (CanonicalRowLayout.layout arity columns covered).paddedIdentityEntry
            baseOps.zero baseOps.one vertex flat)
        (Phi81CoefficientKernel.phi81Kernel.weight output ⟨lane, live⟩
          (Phi81ColumnLayout.decode column).2) := by
  have oneMul (value : F) : (1 : F) * value = value := baseLaws.one_mul value
  have zeroMul (value : F) : (0 : F) * value = 0 :=
    (baseLaws.mul_comm 0 value).trans (baseLaws.mul_zero value)
  by_cases within : column.val / ringDegree * ringDegree + lane < columns
  · by_cases same : column.val / ringDegree * ringDegree + lane = NumericBooleanDomain.index vertex
    · have rowLive : NumericBooleanDomain.index vertex < columns := by omega
      simp [padTerm, live, same, kernelWeight_value, Phi81ColumnLayout.encode?,
        Phi81ColumnLayout.flatIndex, Phi81ColumnLayout.decode, CanonicalRowLayout.layout,
        ColumnLayout.paddedIdentityEntry, rowLive, baseOps, oneMul]
    · by_cases rowLive : NumericBooleanDomain.index vertex < columns <;>
        simp [padTerm, live, within, same, Phi81ColumnLayout.encode?,
          Phi81ColumnLayout.flatIndex, Phi81ColumnLayout.decode, CanonicalRowLayout.layout,
          ColumnLayout.paddedIdentityEntry, rowLive, Fin.ext_iff, baseOps, zeroMul]
  · simp [padTerm, live, within, Phi81ColumnLayout.encode?, Phi81ColumnLayout.flatIndex,
      Phi81ColumnLayout.decode, baseOps, zeroMul]

theorem padEntry_value (arity freshCount runningCount matrixCount logicalWidth : Nat)
    (matrices : Fin matrixCount → BooleanMatrix F arity logicalWidth)
    (polynomial : CCSResidualTable.ConstraintPolynomial F matrixCount)
    (covered : Phi81CarrierLayout.carrierWidth logicalWidth ≤ 2 ^ arity)
    (output : Fin ringDegree) (vertex : BooleanVertex arity)
    (column : Fin (Phi81CarrierLayout.carrierWidth logicalWidth)) :
    (padEntry output vertex column).value =
      (Phi81MatrixSource.source arity freshCount runningCount matrixCount logicalWidth matrices polynomial).coefficientMatrixOf baseOps
          (fun row column =>
            (CanonicalRowLayout.layout arity (Phi81CarrierLayout.carrierWidth logicalWidth) covered).paddedIdentityEntry
              baseOps.zero baseOps.one row column)
          output vertex column := by
  simp only [padEntry, vertexIndex_value, sum_value, MatrixSource.coefficientMatrixOf,
    Phi81MatrixSource.source, Phi81MatrixSource.phi81Shape, Phi81CarrierLayout.layout,
    Phi81ColumnLayout.layout, MatrixSource.paddedEntry]
  apply sumRange_congr baseOps ringDegree
  intro lane live
  simp only [dite_eq_left live]
  have term := padTerm_value covered output vertex column lane live
  cases encoded : Phi81ColumnLayout.encode? (Phi81ColumnLayout.decode column).1 ⟨lane, live⟩ <;>
    simp only [encoded] at term ⊢ <;>
    simpa only [Phi81ColumnLayout.decode] using term

end NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.StoredWitnessCheckEntries
