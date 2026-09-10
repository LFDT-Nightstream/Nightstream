import NightstreamFPrime.Export.Stage1.PiDECMatrixProgram

/-!
Proves exact source custody for the compact PiDEC ordinary-row program. Every
source column used by a canonical PiDEC row resolves through exactly one
Lean-authored retained range and equals the proof-oriented direct source map.
-/

namespace NightstreamFPrime.Export.Stage1.PiDECMatrixProgram

open NightstreamFPrime.Export.MatrixProgram
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.Stage1
open NightstreamFPrime.Layout.Stage1.PiDECSourceSupport
open PiDECRetainedBlocks
open PiDECRetainedGeometry

private theorem sourceStarts_local :
    Spartan.piCcsPhaseOffset ≤ PiDECSourceSupport.parentCommitmentStart ∧
    Spartan.piCcsPhaseOffset ≤ PiDECSourceSupport.parentPublicInputStart ∧
    Spartan.piCcsPhaseOffset ≤ PiDECSourceSupport.parentEvalKStart ∧
    Spartan.piCcsPhaseOffset ≤ PiDECSourceSupport.parentEvalAStart ∧
    Spartan.piCcsPhaseOffset ≤ PiDECInputs.proofInputStart ∧
    Spartan.piCcsPhaseOffset ≤ PiDECStarts.phaseLogicalStart ∧
    Spartan.piCcsPhaseOffset ≤ PiDECStarts.phaseFreshStart := by
  rcases PiDECSourceSupport.source_ranges_ordered with
    ⟨firstLocal, commitmentPublic, publicEvalK, evalKEvalA, evalAProof,
      proofLogical, logicalFresh⟩
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_⟩ <;> omega

private theorem mapped_range_le (start count next : Nat)
    (startLocal : Spartan.piCcsPhaseOffset ≤ start) (ordered : start + count ≤ next) :
    Spartan.sourceToSpartan start + count ≤ Spartan.sourceToSpartan next := by
  rw [← Spartan.sourceToSpartan_add_of_piCcsLocal start count startLocal]
  by_cases same : start + count = next
  · rw [same]
  · exact Nat.le_of_lt (Spartan.sourceToSpartan_lt_of_piCcsLocal
      (start + count) next (by omega) (by omega))

private theorem mapped_ranges_ordered :
    Spartan.sourceToSpartan PiDECSourceSupport.parentCommitmentStart +
        PiDECInputs.commitmentWordsPerChild ≤
      Spartan.sourceToSpartan PiDECSourceSupport.parentPublicInputStart ∧
    Spartan.sourceToSpartan PiDECSourceSupport.parentPublicInputStart +
        PiDECInputs.publicInputWordsPerChild ≤
      Spartan.sourceToSpartan PiDECSourceSupport.parentEvalKStart ∧
    Spartan.sourceToSpartan PiDECSourceSupport.parentEvalKStart +
        PiDECInputs.evalKWordsPerChild ≤
      Spartan.sourceToSpartan PiDECSourceSupport.parentEvalAStart ∧
    Spartan.sourceToSpartan PiDECSourceSupport.parentEvalAStart +
        PiDECInputs.evalAWordsPerChild ≤
      Spartan.sourceToSpartan PiDECInputs.proofInputStart ∧
    Spartan.sourceToSpartan PiDECInputs.proofInputStart +
        PiDECInputs.proofInputColumnCount =
      Spartan.sourceToSpartan PiDECStarts.phaseLogicalStart ∧
    Spartan.sourceToSpartan PiDECStarts.phaseLogicalStart + 270 =
      Spartan.sourceToSpartan PiDECStarts.phaseFreshStart := by
  rcases PiDECSourceSupport.source_ranges_ordered with
    ⟨_, commitmentPublic, publicEvalK, evalKEvalA, evalAProof,
      proofLogical, logicalFresh⟩
  rcases sourceStarts_local with
    ⟨commitmentLocal, publicLocal, evalKLocal, evalALocal, proofLocal,
      logicalLocal, _⟩
  refine ⟨mapped_range_le _ _ _ commitmentLocal commitmentPublic,
    mapped_range_le _ _ _ publicLocal publicEvalK,
    mapped_range_le _ _ _ evalKLocal evalKEvalA,
    mapped_range_le _ _ _ evalALocal evalAProof, ?_, ?_⟩
  · rw [← Spartan.sourceToSpartan_add_of_piCcsLocal _ _ proofLocal, proofLogical]
  · rw [← Spartan.sourceToSpartan_add_of_piCcsLocal _ _ logicalLocal, logicalFresh]

theorem parentCommitmentRange_form?
    {program : ApplicationProgram} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth)
    (index : Fin PiDECInputs.commitmentWordsPerChild) :
    (parentCommitmentRange program).form? logicalWidth
        (Spartan.sourceToSpartan
          (PiDECSourceSupport.parentCommitmentStart + index.val)) =
      some ((PiDECDirectPlan.Location.parentCommitment index).form geometry) := by
  rw [Spartan.sourceToSpartan_add_of_piCcsLocal _ _
    sourceStarts_local.1]
  simpa [parentCommitmentRange, PiDECDirectPlan.Location.form] using
    (SourceRange.form?_ofSemantic (parentCommitmentBlock program)
      (PiDECRetainedGeometry.parentCommitmentStart program)
      (Spartan.sourceToSpartan PiDECSourceSupport.parentCommitmentStart)
      PiDECInputs.commitmentWordsPerChild 0
      (parentCommitmentFits geometry) (by rfl) index)

theorem parentPublicInputRange_form?
    {program : ApplicationProgram} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth)
    (index : Fin PiDECInputs.publicInputWordsPerChild) :
    (parentPublicInputRange program).form? logicalWidth
        (Spartan.sourceToSpartan
          (PiDECSourceSupport.parentPublicInputStart + index.val)) =
      some ((PiDECDirectPlan.Location.parentPublicInput index).form geometry) := by
  rw [Spartan.sourceToSpartan_add_of_piCcsLocal _ _
    sourceStarts_local.2.1]
  simpa [parentPublicInputRange, PiDECDirectPlan.Location.form] using
    (SourceRange.form?_ofSemantic (parentPublicInputBlock program)
      (PiDECRetainedGeometry.parentPublicInputStart program)
      (Spartan.sourceToSpartan PiDECSourceSupport.parentPublicInputStart)
      PiDECInputs.publicInputWordsPerChild 0
      (parentPublicInputFits geometry) (by rfl) index)

theorem parentEvalKRange_form?
    {program : ApplicationProgram} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth)
    (index : Fin PiDECInputs.evalKWordsPerChild) :
    (parentEvalKRange program).form? logicalWidth
        (Spartan.sourceToSpartan
          (PiDECSourceSupport.parentEvalKStart + index.val)) =
      some ((PiDECDirectPlan.Location.parentEvalK index).form geometry) := by
  rw [Spartan.sourceToSpartan_add_of_piCcsLocal _ _
    sourceStarts_local.2.2.1]
  simpa [parentEvalKRange, PiDECDirectPlan.Location.form] using
    (SourceRange.form?_ofSemantic (parentEvalKBlock program)
      (PiDECRetainedGeometry.parentEvalKStart program)
      (Spartan.sourceToSpartan PiDECSourceSupport.parentEvalKStart)
      PiDECInputs.evalKWordsPerChild 0
      (parentEvalKFits geometry) (by rfl) index)

theorem parentEvalARange_form?
    {program : ApplicationProgram} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth)
    (index : Fin PiDECInputs.evalAWordsPerChild) :
    (parentEvalARange program).form? logicalWidth
        (Spartan.sourceToSpartan
          (PiDECSourceSupport.parentEvalAStart + index.val)) =
      some ((PiDECDirectPlan.Location.parentEvalA index).form geometry) := by
  rw [Spartan.sourceToSpartan_add_of_piCcsLocal _ _
    sourceStarts_local.2.2.2.1]
  simpa [parentEvalARange, PiDECDirectPlan.Location.form] using
    (SourceRange.form?_ofSemantic (parentEvalABlock program)
      (PiDECRetainedGeometry.parentEvalAStart program)
      (Spartan.sourceToSpartan PiDECSourceSupport.parentEvalAStart)
      PiDECInputs.evalAWordsPerChild 0
      (parentEvalAFits geometry) (by rfl) index)

theorem proofRange_form?
    {program : ApplicationProgram} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth)
    (index : Fin PiDECInputs.proofInputColumnCount) :
    (proofRange program).form? logicalWidth
        (Spartan.sourceToSpartan (PiDECInputs.proofInputStart + index.val)) =
      some ((PiDECDirectPlan.Location.proof index).form geometry) := by
  rw [Spartan.sourceToSpartan_add_of_piCcsLocal _ _
    sourceStarts_local.2.2.2.2.1]
  simpa [proofRange, PiDECDirectPlan.Location.form] using
    (SourceRange.form?_ofSemantic (proofBlock program) (proofStart program)
      (Spartan.sourceToSpartan PiDECInputs.proofInputStart)
      PiDECInputs.proofInputColumnCount 0 (proofFits geometry) (by rfl) index)

theorem logicalRange_form?
    {program : ApplicationProgram} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth) (index : Fin 270) :
    (logicalRange program).form? logicalWidth
        (Spartan.sourceToSpartan
          (PiDECStarts.phaseLogicalStart + index.val)) =
      some ((PiDECDirectPlan.Location.logical index).form geometry) := by
  rw [Spartan.sourceToSpartan_add_of_piCcsLocal _ _
    sourceStarts_local.2.2.2.2.2.1]
  simpa [logicalRange, PiDECDirectPlan.Location.form] using
    (SourceRange.form?_ofSemantic (logicalBlock program) (logicalStart program)
      (Spartan.sourceToSpartan PiDECStarts.phaseLogicalStart) 270 0
      (logicalFits geometry) (by rfl) index)

theorem freshRange_form?
    {program : ApplicationProgram} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth) (index : Fin freshCount) :
    (freshRange program).form? logicalWidth
        (Spartan.sourceToSpartan (PiDECStarts.phaseFreshStart + index.val)) =
      some ((PiDECDirectPlan.Location.fresh index).form geometry) := by
  rw [Spartan.sourceToSpartan_add_of_piCcsLocal _ _
    sourceStarts_local.2.2.2.2.2.2]
  simpa [freshRange, PiDECDirectPlan.Location.form] using
    (SourceRange.form?_ofSemantic (freshBlock program) (freshStart program)
      (Spartan.sourceToSpartan PiDECStarts.phaseFreshStart) freshCount 0
      (freshFits geometry) (by rfl) index)

private theorem order_through_range {first boundary count last : Nat}
    (firstBound : first ≤ boundary) (nextBound : boundary + count ≤ last) :
    first ≤ last := by omega

private theorem before_of_order {start count next index : Nat}
    (indexBound : index < count) (ordered : start + count ≤ next) :
    start + index < next := by omega

private theorem after_of_order {start count next index : Nat}
    (ordered : start + count ≤ next) : start + count ≤ next + index := by omega

/-- The compact substitution reconstructs every direct PiDEC source
location and rejects all overlapping interpretations. -/
theorem substitution_location_form?
    {program : ApplicationProgram} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth)
    (location : PiDECDirectPlan.Location) :
    (substitution program).form? logicalWidth
        (Spartan.sourceToSpartan location.sourceColumn) =
      some (location.form geometry) := by
  rcases sourceStarts_local with
    ⟨commitmentLocal, publicLocal, evalKLocal, evalALocal, proofLocal,
      logicalLocal, freshLocal⟩
  rcases mapped_ranges_ordered with
    ⟨commitmentPublic, publicEvalK, evalKEvalA, evalAProof,
      proofLogical, logicalFresh⟩
  cases location with
  | parentCommitment index =>
      have indexBound := index.isLt
      have target := Spartan.sourceToSpartan_add_of_piCcsLocal
        PiDECSourceSupport.parentCommitmentStart index.val commitmentLocal
      have selected := parentCommitmentRange_form? geometry index
      rw [target] at selected
      simp only [PiDECDirectPlan.Location.sourceColumn]
      rw [target]
      have parentPublicInputNone := SourceRange.form?_eq_none_of_before
        (parentPublicInputRange program) logicalWidth (Spartan.sourceToSpartan PiDECSourceSupport.parentCommitmentStart + index.val) (by
          dsimp only [parentPublicInputRange, SourceRange.ofSemantic]
          exact before_of_order indexBound commitmentPublic)
      have parentEvalKNone := SourceRange.form?_eq_none_of_before
        (parentEvalKRange program) logicalWidth (Spartan.sourceToSpartan PiDECSourceSupport.parentCommitmentStart + index.val) (by
          dsimp only [parentEvalKRange, SourceRange.ofSemantic]
          exact before_of_order indexBound (order_through_range commitmentPublic publicEvalK))
      have parentEvalANone := SourceRange.form?_eq_none_of_before
        (parentEvalARange program) logicalWidth (Spartan.sourceToSpartan PiDECSourceSupport.parentCommitmentStart + index.val) (by
          dsimp only [parentEvalARange, SourceRange.ofSemantic]
          exact before_of_order indexBound (order_through_range (order_through_range commitmentPublic publicEvalK) evalKEvalA))
      have proofNone := SourceRange.form?_eq_none_of_before
        (proofRange program) logicalWidth (Spartan.sourceToSpartan PiDECSourceSupport.parentCommitmentStart + index.val) (by
          dsimp only [proofRange, SourceRange.ofSemantic]
          exact before_of_order indexBound (order_through_range (order_through_range (order_through_range commitmentPublic publicEvalK) evalKEvalA) evalAProof))
      have logicalNone := SourceRange.form?_eq_none_of_before
        (logicalRange program) logicalWidth (Spartan.sourceToSpartan PiDECSourceSupport.parentCommitmentStart + index.val) (by
          dsimp only [logicalRange, SourceRange.ofSemantic]
          exact before_of_order indexBound (order_through_range (order_through_range (order_through_range (order_through_range commitmentPublic publicEvalK) evalKEvalA) evalAProof) (le_of_eq proofLogical)))
      have freshNone := SourceRange.form?_eq_none_of_before
        (freshRange program) logicalWidth (Spartan.sourceToSpartan PiDECSourceSupport.parentCommitmentStart + index.val) (by
          dsimp only [freshRange, SourceRange.ofSemantic]
          exact before_of_order indexBound (order_through_range (order_through_range (order_through_range (order_through_range (order_through_range commitmentPublic publicEvalK) evalKEvalA) evalAProof) (le_of_eq proofLogical)) (le_of_eq logicalFresh)))
      simp only [substitution, SourceSubstitution.form?, List.filterMap_cons,
        List.filterMap_nil, selected, parentPublicInputNone, parentEvalKNone,
        parentEvalANone, proofNone, logicalNone, freshNone, List.append_nil]
  | parentPublicInput index =>
      have indexBound := index.isLt
      have target := Spartan.sourceToSpartan_add_of_piCcsLocal
        PiDECSourceSupport.parentPublicInputStart index.val publicLocal
      have selected := parentPublicInputRange_form? geometry index
      rw [target] at selected
      simp only [PiDECDirectPlan.Location.sourceColumn]
      rw [target]
      have parentCommitmentNone := SourceRange.form?_eq_none_of_after
        (parentCommitmentRange program) logicalWidth (Spartan.sourceToSpartan PiDECSourceSupport.parentPublicInputStart + index.val) (by
          dsimp only [parentCommitmentRange, SourceRange.ofSemantic]
          exact after_of_order commitmentPublic)
      have parentEvalKNone := SourceRange.form?_eq_none_of_before
        (parentEvalKRange program) logicalWidth (Spartan.sourceToSpartan PiDECSourceSupport.parentPublicInputStart + index.val) (by
          dsimp only [parentEvalKRange, SourceRange.ofSemantic]
          exact before_of_order indexBound publicEvalK)
      have parentEvalANone := SourceRange.form?_eq_none_of_before
        (parentEvalARange program) logicalWidth (Spartan.sourceToSpartan PiDECSourceSupport.parentPublicInputStart + index.val) (by
          dsimp only [parentEvalARange, SourceRange.ofSemantic]
          exact before_of_order indexBound (order_through_range publicEvalK evalKEvalA))
      have proofNone := SourceRange.form?_eq_none_of_before
        (proofRange program) logicalWidth (Spartan.sourceToSpartan PiDECSourceSupport.parentPublicInputStart + index.val) (by
          dsimp only [proofRange, SourceRange.ofSemantic]
          exact before_of_order indexBound (order_through_range (order_through_range publicEvalK evalKEvalA) evalAProof))
      have logicalNone := SourceRange.form?_eq_none_of_before
        (logicalRange program) logicalWidth (Spartan.sourceToSpartan PiDECSourceSupport.parentPublicInputStart + index.val) (by
          dsimp only [logicalRange, SourceRange.ofSemantic]
          exact before_of_order indexBound (order_through_range (order_through_range (order_through_range publicEvalK evalKEvalA) evalAProof) (le_of_eq proofLogical)))
      have freshNone := SourceRange.form?_eq_none_of_before
        (freshRange program) logicalWidth (Spartan.sourceToSpartan PiDECSourceSupport.parentPublicInputStart + index.val) (by
          dsimp only [freshRange, SourceRange.ofSemantic]
          exact before_of_order indexBound (order_through_range (order_through_range (order_through_range (order_through_range publicEvalK evalKEvalA) evalAProof) (le_of_eq proofLogical)) (le_of_eq logicalFresh)))
      simp only [substitution, SourceSubstitution.form?, List.filterMap_cons,
        List.filterMap_nil, parentCommitmentNone, selected, parentEvalKNone,
        parentEvalANone, proofNone, logicalNone, freshNone, List.append_nil]
  | parentEvalK index =>
      have indexBound := index.isLt
      have target := Spartan.sourceToSpartan_add_of_piCcsLocal
        PiDECSourceSupport.parentEvalKStart index.val evalKLocal
      have selected := parentEvalKRange_form? geometry index
      rw [target] at selected
      simp only [PiDECDirectPlan.Location.sourceColumn]
      rw [target]
      have parentCommitmentNone := SourceRange.form?_eq_none_of_after
        (parentCommitmentRange program) logicalWidth (Spartan.sourceToSpartan PiDECSourceSupport.parentEvalKStart + index.val) (by
          dsimp only [parentCommitmentRange, SourceRange.ofSemantic]
          exact after_of_order (order_through_range commitmentPublic publicEvalK))
      have parentPublicInputNone := SourceRange.form?_eq_none_of_after
        (parentPublicInputRange program) logicalWidth (Spartan.sourceToSpartan PiDECSourceSupport.parentEvalKStart + index.val) (by
          dsimp only [parentPublicInputRange, SourceRange.ofSemantic]
          exact after_of_order publicEvalK)
      have parentEvalANone := SourceRange.form?_eq_none_of_before
        (parentEvalARange program) logicalWidth (Spartan.sourceToSpartan PiDECSourceSupport.parentEvalKStart + index.val) (by
          dsimp only [parentEvalARange, SourceRange.ofSemantic]
          exact before_of_order indexBound evalKEvalA)
      have proofNone := SourceRange.form?_eq_none_of_before
        (proofRange program) logicalWidth (Spartan.sourceToSpartan PiDECSourceSupport.parentEvalKStart + index.val) (by
          dsimp only [proofRange, SourceRange.ofSemantic]
          exact before_of_order indexBound (order_through_range evalKEvalA evalAProof))
      have logicalNone := SourceRange.form?_eq_none_of_before
        (logicalRange program) logicalWidth (Spartan.sourceToSpartan PiDECSourceSupport.parentEvalKStart + index.val) (by
          dsimp only [logicalRange, SourceRange.ofSemantic]
          exact before_of_order indexBound (order_through_range (order_through_range evalKEvalA evalAProof) (le_of_eq proofLogical)))
      have freshNone := SourceRange.form?_eq_none_of_before
        (freshRange program) logicalWidth (Spartan.sourceToSpartan PiDECSourceSupport.parentEvalKStart + index.val) (by
          dsimp only [freshRange, SourceRange.ofSemantic]
          exact before_of_order indexBound (order_through_range (order_through_range (order_through_range evalKEvalA evalAProof) (le_of_eq proofLogical)) (le_of_eq logicalFresh)))
      simp only [substitution, SourceSubstitution.form?, List.filterMap_cons,
        List.filterMap_nil, parentCommitmentNone, parentPublicInputNone, selected,
        parentEvalANone, proofNone, logicalNone, freshNone, List.append_nil]
  | parentEvalA index =>
      have indexBound := index.isLt
      have target := Spartan.sourceToSpartan_add_of_piCcsLocal
        PiDECSourceSupport.parentEvalAStart index.val evalALocal
      have selected := parentEvalARange_form? geometry index
      rw [target] at selected
      simp only [PiDECDirectPlan.Location.sourceColumn]
      rw [target]
      have parentCommitmentNone := SourceRange.form?_eq_none_of_after
        (parentCommitmentRange program) logicalWidth (Spartan.sourceToSpartan PiDECSourceSupport.parentEvalAStart + index.val) (by
          dsimp only [parentCommitmentRange, SourceRange.ofSemantic]
          exact after_of_order (order_through_range (order_through_range commitmentPublic publicEvalK) evalKEvalA))
      have parentPublicInputNone := SourceRange.form?_eq_none_of_after
        (parentPublicInputRange program) logicalWidth (Spartan.sourceToSpartan PiDECSourceSupport.parentEvalAStart + index.val) (by
          dsimp only [parentPublicInputRange, SourceRange.ofSemantic]
          exact after_of_order (order_through_range publicEvalK evalKEvalA))
      have parentEvalKNone := SourceRange.form?_eq_none_of_after
        (parentEvalKRange program) logicalWidth (Spartan.sourceToSpartan PiDECSourceSupport.parentEvalAStart + index.val) (by
          dsimp only [parentEvalKRange, SourceRange.ofSemantic]
          exact after_of_order evalKEvalA)
      have proofNone := SourceRange.form?_eq_none_of_before
        (proofRange program) logicalWidth (Spartan.sourceToSpartan PiDECSourceSupport.parentEvalAStart + index.val) (by
          dsimp only [proofRange, SourceRange.ofSemantic]
          exact before_of_order indexBound evalAProof)
      have logicalNone := SourceRange.form?_eq_none_of_before
        (logicalRange program) logicalWidth (Spartan.sourceToSpartan PiDECSourceSupport.parentEvalAStart + index.val) (by
          dsimp only [logicalRange, SourceRange.ofSemantic]
          exact before_of_order indexBound (order_through_range evalAProof (le_of_eq proofLogical)))
      have freshNone := SourceRange.form?_eq_none_of_before
        (freshRange program) logicalWidth (Spartan.sourceToSpartan PiDECSourceSupport.parentEvalAStart + index.val) (by
          dsimp only [freshRange, SourceRange.ofSemantic]
          exact before_of_order indexBound (order_through_range (order_through_range evalAProof (le_of_eq proofLogical)) (le_of_eq logicalFresh)))
      simp only [substitution, SourceSubstitution.form?, List.filterMap_cons,
        List.filterMap_nil, parentCommitmentNone, parentPublicInputNone, parentEvalKNone,
        selected, proofNone, logicalNone, freshNone, List.append_nil]
  | proof index =>
      have indexBound := index.isLt
      have target := Spartan.sourceToSpartan_add_of_piCcsLocal
        PiDECInputs.proofInputStart index.val proofLocal
      have selected := proofRange_form? geometry index
      rw [target] at selected
      simp only [PiDECDirectPlan.Location.sourceColumn]
      rw [target]
      have parentCommitmentNone := SourceRange.form?_eq_none_of_after
        (parentCommitmentRange program) logicalWidth (Spartan.sourceToSpartan PiDECInputs.proofInputStart + index.val) (by
          dsimp only [parentCommitmentRange, SourceRange.ofSemantic]
          exact after_of_order (order_through_range (order_through_range (order_through_range commitmentPublic publicEvalK) evalKEvalA) evalAProof))
      have parentPublicInputNone := SourceRange.form?_eq_none_of_after
        (parentPublicInputRange program) logicalWidth (Spartan.sourceToSpartan PiDECInputs.proofInputStart + index.val) (by
          dsimp only [parentPublicInputRange, SourceRange.ofSemantic]
          exact after_of_order (order_through_range (order_through_range publicEvalK evalKEvalA) evalAProof))
      have parentEvalKNone := SourceRange.form?_eq_none_of_after
        (parentEvalKRange program) logicalWidth (Spartan.sourceToSpartan PiDECInputs.proofInputStart + index.val) (by
          dsimp only [parentEvalKRange, SourceRange.ofSemantic]
          exact after_of_order (order_through_range evalKEvalA evalAProof))
      have parentEvalANone := SourceRange.form?_eq_none_of_after
        (parentEvalARange program) logicalWidth (Spartan.sourceToSpartan PiDECInputs.proofInputStart + index.val) (by
          dsimp only [parentEvalARange, SourceRange.ofSemantic]
          exact after_of_order evalAProof)
      have logicalNone := SourceRange.form?_eq_none_of_before
        (logicalRange program) logicalWidth (Spartan.sourceToSpartan PiDECInputs.proofInputStart + index.val) (by
          dsimp only [logicalRange, SourceRange.ofSemantic]
          exact before_of_order indexBound (le_of_eq proofLogical))
      have freshNone := SourceRange.form?_eq_none_of_before
        (freshRange program) logicalWidth (Spartan.sourceToSpartan PiDECInputs.proofInputStart + index.val) (by
          dsimp only [freshRange, SourceRange.ofSemantic]
          exact before_of_order indexBound (order_through_range (le_of_eq proofLogical) (le_of_eq logicalFresh)))
      simp only [substitution, SourceSubstitution.form?, List.filterMap_cons,
        List.filterMap_nil, parentCommitmentNone, parentPublicInputNone, parentEvalKNone,
        parentEvalANone, selected, logicalNone, freshNone, List.append_nil]
  | logical index =>
      have indexBound := index.isLt
      have target := Spartan.sourceToSpartan_add_of_piCcsLocal
        PiDECStarts.phaseLogicalStart index.val logicalLocal
      have selected := logicalRange_form? geometry index
      rw [target] at selected
      simp only [PiDECDirectPlan.Location.sourceColumn]
      rw [target]
      have parentCommitmentNone := SourceRange.form?_eq_none_of_after
        (parentCommitmentRange program) logicalWidth (Spartan.sourceToSpartan PiDECStarts.phaseLogicalStart + index.val) (by
          dsimp only [parentCommitmentRange, SourceRange.ofSemantic]
          exact after_of_order (order_through_range (order_through_range (order_through_range (order_through_range commitmentPublic publicEvalK) evalKEvalA) evalAProof) (le_of_eq proofLogical)))
      have parentPublicInputNone := SourceRange.form?_eq_none_of_after
        (parentPublicInputRange program) logicalWidth (Spartan.sourceToSpartan PiDECStarts.phaseLogicalStart + index.val) (by
          dsimp only [parentPublicInputRange, SourceRange.ofSemantic]
          exact after_of_order (order_through_range (order_through_range (order_through_range publicEvalK evalKEvalA) evalAProof) (le_of_eq proofLogical)))
      have parentEvalKNone := SourceRange.form?_eq_none_of_after
        (parentEvalKRange program) logicalWidth (Spartan.sourceToSpartan PiDECStarts.phaseLogicalStart + index.val) (by
          dsimp only [parentEvalKRange, SourceRange.ofSemantic]
          exact after_of_order (order_through_range (order_through_range evalKEvalA evalAProof) (le_of_eq proofLogical)))
      have parentEvalANone := SourceRange.form?_eq_none_of_after
        (parentEvalARange program) logicalWidth (Spartan.sourceToSpartan PiDECStarts.phaseLogicalStart + index.val) (by
          dsimp only [parentEvalARange, SourceRange.ofSemantic]
          exact after_of_order (order_through_range evalAProof (le_of_eq proofLogical)))
      have proofNone := SourceRange.form?_eq_none_of_after
        (proofRange program) logicalWidth (Spartan.sourceToSpartan PiDECStarts.phaseLogicalStart + index.val) (by
          dsimp only [proofRange, SourceRange.ofSemantic]
          exact after_of_order (le_of_eq proofLogical))
      have freshNone := SourceRange.form?_eq_none_of_before
        (freshRange program) logicalWidth (Spartan.sourceToSpartan PiDECStarts.phaseLogicalStart + index.val) (by
          dsimp only [freshRange, SourceRange.ofSemantic]
          exact before_of_order indexBound (le_of_eq logicalFresh))
      simp only [substitution, SourceSubstitution.form?, List.filterMap_cons,
        List.filterMap_nil, parentCommitmentNone, parentPublicInputNone, parentEvalKNone,
        parentEvalANone, proofNone, selected, freshNone, List.append_nil]
  | fresh index =>
      have target := Spartan.sourceToSpartan_add_of_piCcsLocal
        PiDECStarts.phaseFreshStart index.val freshLocal
      have selected := freshRange_form? geometry index
      rw [target] at selected
      simp only [PiDECDirectPlan.Location.sourceColumn]
      rw [target]
      have parentCommitmentNone := SourceRange.form?_eq_none_of_after
        (parentCommitmentRange program) logicalWidth (Spartan.sourceToSpartan PiDECStarts.phaseFreshStart + index.val) (by
          dsimp only [parentCommitmentRange, SourceRange.ofSemantic]
          exact after_of_order (order_through_range (order_through_range (order_through_range (order_through_range (order_through_range commitmentPublic publicEvalK) evalKEvalA) evalAProof) (le_of_eq proofLogical)) (le_of_eq logicalFresh)))
      have parentPublicInputNone := SourceRange.form?_eq_none_of_after
        (parentPublicInputRange program) logicalWidth (Spartan.sourceToSpartan PiDECStarts.phaseFreshStart + index.val) (by
          dsimp only [parentPublicInputRange, SourceRange.ofSemantic]
          exact after_of_order (order_through_range (order_through_range (order_through_range (order_through_range publicEvalK evalKEvalA) evalAProof) (le_of_eq proofLogical)) (le_of_eq logicalFresh)))
      have parentEvalKNone := SourceRange.form?_eq_none_of_after
        (parentEvalKRange program) logicalWidth (Spartan.sourceToSpartan PiDECStarts.phaseFreshStart + index.val) (by
          dsimp only [parentEvalKRange, SourceRange.ofSemantic]
          exact after_of_order (order_through_range (order_through_range (order_through_range evalKEvalA evalAProof) (le_of_eq proofLogical)) (le_of_eq logicalFresh)))
      have parentEvalANone := SourceRange.form?_eq_none_of_after
        (parentEvalARange program) logicalWidth (Spartan.sourceToSpartan PiDECStarts.phaseFreshStart + index.val) (by
          dsimp only [parentEvalARange, SourceRange.ofSemantic]
          exact after_of_order (order_through_range (order_through_range evalAProof (le_of_eq proofLogical)) (le_of_eq logicalFresh)))
      have proofNone := SourceRange.form?_eq_none_of_after
        (proofRange program) logicalWidth (Spartan.sourceToSpartan PiDECStarts.phaseFreshStart + index.val) (by
          dsimp only [proofRange, SourceRange.ofSemantic]
          exact after_of_order (order_through_range (le_of_eq proofLogical) (le_of_eq logicalFresh)))
      have logicalNone := SourceRange.form?_eq_none_of_after
        (logicalRange program) logicalWidth (Spartan.sourceToSpartan PiDECStarts.phaseFreshStart + index.val) (by
          dsimp only [logicalRange, SourceRange.ofSemantic]
          exact after_of_order (le_of_eq logicalFresh))
      simp only [substitution, SourceSubstitution.form?, List.filterMap_cons,
        List.filterMap_nil, parentCommitmentNone, parentPublicInputNone, parentEvalKNone,
        parentEvalANone, proofNone, logicalNone, selected, List.append_nil]

/-- On every source column used by a canonical PiDEC row, the package
substitution is exactly the direct Lean source map. -/
theorem substitution_agrees_on_target
    {program : ApplicationProgram} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth)
    (column : Fin Spartan.spartanColumnCount)
    (support : PiDECSourceSupport.Target column.val) :
    (substitution program).form? logicalWidth column.val =
      some ((PiDECDirectPlan.sourceMap geometry).form column) := by
  rcases PiDECDirectPlan.classifyTarget_complete support with
    ⟨decoded, found, mapped⟩
  change (substitution program).form? logicalWidth column.val =
    some (match PiDECDirectPlan.classifyTarget column.val with
      | none => .empty
      | some value => value.location.form geometry)
  rw [found]
  have target :
      Spartan.sourceToSpartan decoded.location.sourceColumn = column.val := by
    rw [decoded.owns, mapped]
  simpa only [target] using
    (substitution_location_form? geometry decoded.location)

end NightstreamFPrime.Export.Stage1.PiDECMatrixProgram
