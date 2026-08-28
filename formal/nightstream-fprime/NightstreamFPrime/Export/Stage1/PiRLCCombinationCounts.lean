import NightstreamFPrime.Export.Stage1.PiRLCCombinationInvocations

/-!
Owns structural row-count proofs for the compact PiRLC combination schedule.

The proofs keep the source, block, lane, and cell products symbolic. They do
not evaluate the full invocation list.
-/

namespace NightstreamFPrime.Export.Stage1.PiRLCCombinationInvocations

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Export.Package
open NightstreamFPrime.Layout
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Lifecycle.PiRLC.v1_1
open NightstreamFPrime.Layout.Stage1

abbrev laneFreshCount :=
  NightstreamFPrime.Layout.PiRLC.v1_1.CombinationStep.laneFreshCount

private theorem compactInvocationRowCountFor_invocation
    (templates : List CompactRowTemplate)
    (selection : ∀ source : Nat, ∀ lane : Fin ringDegree,
      templates[PiRLCCombinationTemplates.templateIndex source lane.val]? =
        some (PiRLCCombinationTemplates.template (firstSource source) lane))
    (logicalStart rowStart freshStart blockCount cellCount valueStride source
      block cell : Nat) (lane : Fin ringDegree)
    (valueSourceStart : Nat → Nat → Nat → Nat) :
    compactInvocationRowCountFor templates
        (invocation logicalStart rowStart freshStart blockCount cellCount
          valueStride source block lane.val cell valueSourceStart) =
      laneFreshCount lane + 1 := by
  unfold compactInvocationRowCountFor invocation
  dsimp only
  rw [selection source lane]
  exact PiRLCCombinationTemplates.template_rows_length _ _

private theorem listSumOfFn {count : Nat} (values : Fin count → Nat) :
    (List.ofFn values).sum = ∑ index, values index := by
  induction count with
  | zero => rfl
  | succ count inductionHypothesis =>
      simp [List.ofFn_succ, Fin.sum_univ_succ, inductionHypothesis]

private theorem indexedLaneRowCountSum
    (blockCount cellCount : Nat) [NeZero cellCount] :
    (List.ofFn fun index : Fin (CombinationStep.privateCount blockCount cellCount) =>
      laneFreshCount (CombinationStep.laneOf index) + 1).sum =
        blockCount * cellCount * 8154 := by
  rw [listSumOfFn]
  let outerEquiv :
      Fin (CombinationStep.privateCount blockCount cellCount) ≃
        Fin blockCount × Fin (ringDegree * cellCount) := by
    unfold CombinationStep.privateCount
    exact finProdFinEquiv.symm
  let innerEquiv :
      Fin (ringDegree * cellCount) ≃ Fin ringDegree × Fin cellCount :=
    finProdFinEquiv.symm
  calc
    ∑ index : Fin (CombinationStep.privateCount blockCount cellCount),
        (laneFreshCount (CombinationStep.laneOf index) + 1) =
      ∑ pair : Fin blockCount × Fin (ringDegree * cellCount),
        (laneFreshCount (innerEquiv pair.2).1 + 1) := by
          apply Fintype.sum_equiv outerEquiv
          intro index
          rfl
    _ = ∑ block : Fin blockCount,
        ∑ inner : Fin (ringDegree * cellCount),
          (laneFreshCount (innerEquiv inner).1 + 1) := by
            rw [Fintype.sum_prod_type]
    _ = ∑ block : Fin blockCount,
        ∑ pair : Fin ringDegree × Fin cellCount,
          (laneFreshCount pair.1 + 1) := by
            apply Finset.sum_congr rfl
            intro block _
            apply Fintype.sum_equiv innerEquiv
            intro inner
            rfl
    _ = ∑ block : Fin blockCount,
        ∑ lane : Fin ringDegree,
          ∑ cell : Fin cellCount,
            (laneFreshCount lane + 1) := by
              simp_rw [Fintype.sum_prod_type]
    _ = ∑ block : Fin blockCount,
        cellCount *
          (∑ lane : Fin ringDegree,
            (laneFreshCount lane + 1)) := by
              apply Finset.sum_congr rfl
              intro block _
              rw [Finset.mul_sum]
              apply Finset.sum_congr rfl
              intro lane _
              simp
    _ = blockCount * cellCount * 8154 := by
      have laneSum :
          ∑ lane : Fin ringDegree,
              (laneFreshCount lane + 1) = 8154 := by
        rw [← listSumOfFn]
        exact PiRLCCombinationTemplates.laneRowCount_sum
      rw [laneSum]
      simp [Nat.mul_assoc, Nat.mul_comm, Nat.mul_left_comm]

private theorem sourceCompactRowCountFor
    (templates : List CompactRowTemplate)
    (selection : ∀ source : Nat, ∀ lane : Fin ringDegree,
      templates[PiRLCCombinationTemplates.templateIndex source lane.val]? =
        some (PiRLCCombinationTemplates.template (firstSource source) lane))
    (logicalStart rowStart freshStart blockCount cellCount valueStride source : Nat)
    [NeZero cellCount] (valueSourceStart : Nat → Nat → Nat → Nat) :
    compactRowCountFor templates
        (List.ofFn fun index : Fin
            (CombinationStep.privateCount blockCount cellCount) =>
          let coordinates := CombinationStep.coordinates index
          invocation logicalStart rowStart freshStart blockCount cellCount
            valueStride source coordinates.1.val coordinates.2.1.val
              coordinates.2.2.val valueSourceStart) =
      blockCount * cellCount * 8154 := by
  unfold compactRowCountFor
  rw [List.map_ofFn]
  calc
    (List.ofFn fun index : Fin
        (CombinationStep.privateCount blockCount cellCount) =>
      compactInvocationRowCountFor templates
        (let coordinates := CombinationStep.coordinates index
         invocation logicalStart rowStart freshStart blockCount cellCount
           valueStride source coordinates.1.val coordinates.2.1.val
             coordinates.2.2.val valueSourceStart)).sum =
      (List.ofFn fun index : Fin
          (CombinationStep.privateCount blockCount cellCount) =>
        laneFreshCount (CombinationStep.laneOf index) + 1).sum := by
          apply congrArg List.sum
          apply congrArg List.ofFn
          funext index
          simpa [CombinationStep.laneOf] using
            compactInvocationRowCountFor_invocation templates selection
              logicalStart rowStart freshStart blockCount cellCount valueStride
              source (CombinationStep.coordinates index).1.val
              (CombinationStep.coordinates index).2.2.val
              (CombinationStep.coordinates index).2.1 valueSourceStart
    _ = blockCount * cellCount * 8154 :=
      indexedLaneRowCountSum blockCount cellCount

private theorem compactRowCountFor_flatMap {Index : Type}
    (templates : List CompactRowTemplate) (indices : List Index)
    (entries : Index → List CompactRowInvocation) :
    compactRowCountFor templates (indices.flatMap entries) =
      (indices.map fun index =>
        compactRowCountFor templates (entries index)).sum := by
  induction indices with
  | nil => rfl
  | cons index rest inductionHypothesis =>
      simp [compactRowCountFor_append, inductionHypothesis]

private theorem sum_map_const {Index : Type}
    (indices : List Index) (value : Nat) :
    (indices.map fun _ => value).sum = indices.length * value := by
  induction indices with
  | nil => simp
  | cons index rest inductionHypothesis =>
      simp only [List.map_cons, List.sum_cons, List.length_cons,
        inductionHypothesis, Nat.succ_mul]
      omega

theorem familyCompactRowCountFor
    (templates : List CompactRowTemplate)
    (selection : ∀ source : Nat, ∀ lane : Fin ringDegree,
      templates[PiRLCCombinationTemplates.templateIndex source lane.val]? =
        some (PiRLCCombinationTemplates.template (firstSource source) lane))
    (logicalStart rowStart freshStart blockCount cellCount valueStride : Nat)
    [NeZero cellCount] (valueSourceStart : Nat → Nat → Nat → Nat) :
    compactRowCountFor templates
        (familyInvocations logicalStart rowStart freshStart blockCount cellCount
          valueStride valueSourceStart) =
      sourceCount * blockCount * cellCount * 8154 := by
  unfold familyInvocations
  rw [compactRowCountFor_flatMap]
  simp_rw [sourceCompactRowCountFor templates selection logicalStart rowStart
    freshStart blockCount cellCount valueStride]
  rw [sum_map_const, List.length_range]
  ring

theorem invocationsCompactRowCountFor
    (templates : List CompactRowTemplate)
    (selection : ∀ source : Nat, ∀ lane : Fin ringDegree,
      templates[PiRLCCombinationTemplates.templateIndex source lane.val]? =
        some (PiRLCCombinationTemplates.template (firstSource source) lane)) :
    compactRowCountFor templates invocations = 7346754 := by
  rw [invocations, compactRowCountFor_append, compactRowCountFor_append,
    compactRowCountFor_append]
  rw [commitmentInvocations, familyCompactRowCountFor templates selection,
    publicInputInvocations, familyCompactRowCountFor templates selection,
    evalKInvocations, familyCompactRowCountFor templates selection,
    evalAInvocations, familyCompactRowCountFor templates selection]
  norm_num [sourceCount]

end NightstreamFPrime.Export.Stage1.PiRLCCombinationInvocations
