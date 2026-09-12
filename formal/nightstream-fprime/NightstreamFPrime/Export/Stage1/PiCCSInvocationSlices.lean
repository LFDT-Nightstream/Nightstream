import NightstreamFPrime.Export.Stage1.PiCCSCompletedReadout
import NightstreamFPrime.Export.Stage1.PiCCSTranscriptDirectSemantics

/-!
Owns the four indexed views of the existing physical C invocation list.
Each slice selects the same invocation in its semantic compiler trace.
Only list composition and the proved trace equalities are used.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.PiCCSInvocationSlices

open NightstreamFPrime.Export.Package
open NightstreamFPrime.Layout.ProductionRelation
open PiCCSPoseidonPreservation
open PiCCSInvocations
open PiCCSTranscriptDirectSemantics

private theorem statement_length :
    (statementTrace Data.logicalWidth Data.publicFits).invocations.length = statementCount :=
  statementInvocations_length Data.logicalWidth Data.publicFits

private theorem challenge_length :
    (challengeSemanticTrace Data.logicalWidth Data.publicFits).invocations.length = challengeCount := by
  have owned := challengeInvocations_length Data.logicalWidth Data.publicFits
  rw [challengeTrace_eq_semantic] at owned
  exact owned

private theorem round_length :
    (roundSemanticTrace Data.logicalWidth Data.publicFits).invocations.length = roundCount := by
  have owned := roundInvocations_length Data.logicalWidth Data.publicFits
  rw [roundTrace_eq_semantic] at owned
  exact owned

private theorem output_length :
    (outputSemanticTrace Data.logicalWidth Data.publicFits).invocations.length = outputCount := by
  have owned := outputInvocations_length Data.logicalWidth Data.publicFits
  rw [outputTrace_eq_semantic] at owned
  exact owned

private theorem semantic_list :
    PiCCSInvocations.invocations Data.logicalWidth Data.publicFits =
      (statementTrace Data.logicalWidth Data.publicFits).invocations ++
      (challengeSemanticTrace Data.logicalWidth Data.publicFits).invocations ++
      (roundSemanticTrace Data.logicalWidth Data.publicFits).invocations ++
      (outputSemanticTrace Data.logicalWidth Data.publicFits).invocations := by
  unfold PiCCSInvocations.invocations
  rw [challengeTrace_eq_semantic, roundTrace_eq_semantic, outputTrace_eq_semantic]

private theorem physical_get? (index : InvocationIndex) :
    some (physicalInvocation index) =
      (PiCCSInvocations.invocations Data.logicalWidth Data.publicFits)[index.val]? := by
  have packageBound : index.val < PoseidonRetainedBlock.basePackage.permutationInvocations.length := by
    rw [PoseidonRetainedBlock.basePackage_permutationInvocations_length]
    exact (laterIndex index).isLt
  have cBound : index.val < (PiCCSInvocations.invocations Data.logicalWidth Data.publicFits).length := by
    rw [PiCCSInvocations.invocations_length]
    simpa only [PiCCSPoseidonPlan.invocationCount_eq] using index.isLt
  calc
    some (physicalInvocation index) =
        PoseidonRetainedBlock.basePackage.permutationInvocations[index.val]? :=
      (List.getElem?_eq_getElem packageBound).symm
    _ = (Data.permutationInvocations ())[index.val]? :=
      congrArg (fun entries : List PermutationInvocation => entries[index.val]?)
        PoseidonRetainedBlock.basePackage_permutationInvocations_eq
    _ = (PiCCSInvocations.invocations Data.logicalWidth Data.publicFits)[index.val]? := by
      rw [Data.permutationInvocations_eq, List.getElem?_append_left cBound]

private theorem middle_get? {Value : Type*} (before selected after : List Value)
    (index : Nat) (bounded : index < selected.length) :
    ((before ++ selected) ++ after)[before.length + index]? = selected[index]? := by
  rw [List.getElem?_append_left (by simp only [List.length_append]; omega)]
  rw [List.getElem?_append_right (Nat.le_add_right _ _)]
  rw [Nat.add_sub_cancel_left]

/-- The statement slice selects its actual statement compiler invocation. -/
theorem statement_invocation (index : Fin statementCount) :
    physicalInvocation (PoseidonActionSemantics.sliceIndex statementOffset statementCount statementFits index) =
      (statementTrace Data.logicalWidth Data.publicFits).invocations.get
        ⟨index.val, by rw [statement_length]; exact index.isLt⟩ := by
  apply Option.some.inj
  have selected := physical_get?
    (PoseidonActionSemantics.sliceIndex statementOffset statementCount statementFits index)
  rw [semantic_list] at selected
  simp only [PoseidonActionSemantics.sliceIndex, statementOffset, Nat.zero_add] at selected
  have bounded : index.val < (statementTrace Data.logicalWidth Data.publicFits).invocations.length := by
    rw [statement_length]
    exact index.isLt
  rw [List.append_assoc, List.append_assoc,
    List.getElem?_append_left bounded] at selected
  exact selected.trans (List.getElem?_eq_getElem bounded)

/-- The challenge slice selects the semantic challenge compiler invocation. -/
theorem challenge_invocation (index : Fin challengeCount) :
    physicalInvocation (PoseidonActionSemantics.sliceIndex challengeOffset challengeCount challengeFits index) =
      (challengeSemanticTrace Data.logicalWidth Data.publicFits).invocations.get
        ⟨index.val, by rw [challenge_length]; exact index.isLt⟩ := by
  apply Option.some.inj
  have selected := physical_get?
    (PoseidonActionSemantics.sliceIndex challengeOffset challengeCount challengeFits index)
  rw [semantic_list] at selected
  have bounded : index.val < (challengeSemanticTrace Data.logicalWidth Data.publicFits).invocations.length := by
    rw [challenge_length]
    exact index.isLt
  have offset : challengeOffset =
      (statementTrace Data.logicalWidth Data.publicFits).invocations.length := by
    rw [statement_length]
    rfl
  simp only [PoseidonActionSemantics.sliceIndex] at selected
  rw [offset, List.append_assoc
    ((statementTrace Data.logicalWidth Data.publicFits).invocations ++
      (challengeSemanticTrace Data.logicalWidth Data.publicFits).invocations)
    (roundSemanticTrace Data.logicalWidth Data.publicFits).invocations
    (outputSemanticTrace Data.logicalWidth Data.publicFits).invocations,
    middle_get? _ _ _ index.val bounded] at selected
  exact selected.trans (List.getElem?_eq_getElem bounded)

/-- The round slice selects the semantic round compiler invocation. -/
theorem round_invocation (index : Fin roundCount) :
    physicalInvocation (PoseidonActionSemantics.sliceIndex roundOffset roundCount roundFits index) =
      (roundSemanticTrace Data.logicalWidth Data.publicFits).invocations.get
        ⟨index.val, by rw [round_length]; exact index.isLt⟩ := by
  apply Option.some.inj
  have selected := physical_get?
    (PoseidonActionSemantics.sliceIndex roundOffset roundCount roundFits index)
  rw [semantic_list] at selected
  have bounded : index.val < (roundSemanticTrace Data.logicalWidth Data.publicFits).invocations.length := by
    rw [round_length]
    exact index.isLt
  have offset : roundOffset =
      ((statementTrace Data.logicalWidth Data.publicFits).invocations ++
        (challengeSemanticTrace Data.logicalWidth Data.publicFits).invocations).length := by
    rw [List.length_append, statement_length, challenge_length]
    rfl
  simp only [PoseidonActionSemantics.sliceIndex] at selected
  rw [offset, middle_get? _ _ _ index.val bounded] at selected
  exact selected.trans (List.getElem?_eq_getElem bounded)

/-- The output slice selects the semantic output compiler invocation. -/
theorem output_invocation (index : Fin outputCount) :
    physicalInvocation (PoseidonActionSemantics.sliceIndex outputOffset outputCount outputFits index) =
      (outputSemanticTrace Data.logicalWidth Data.publicFits).invocations.get
        ⟨index.val, by rw [output_length]; exact index.isLt⟩ := by
  apply Option.some.inj
  have selected := physical_get?
    (PoseidonActionSemantics.sliceIndex outputOffset outputCount outputFits index)
  rw [semantic_list] at selected
  have bounded : index.val < (outputSemanticTrace Data.logicalWidth Data.publicFits).invocations.length := by
    rw [output_length]
    exact index.isLt
  have offset : outputOffset =
      (((statementTrace Data.logicalWidth Data.publicFits).invocations ++
        (challengeSemanticTrace Data.logicalWidth Data.publicFits).invocations) ++
        (roundSemanticTrace Data.logicalWidth Data.publicFits).invocations).length := by
    rw [List.length_append, List.length_append, statement_length, challenge_length, round_length]
    rfl
  simp only [PoseidonActionSemantics.sliceIndex] at selected
  rw [offset, List.getElem?_append_right (Nat.le_add_right _ _),
    Nat.add_sub_cancel_left] at selected
  exact selected.trans (List.getElem?_eq_getElem bounded)

end NightstreamFPrime.Export.Stage1.PiCCSInvocationSlices
