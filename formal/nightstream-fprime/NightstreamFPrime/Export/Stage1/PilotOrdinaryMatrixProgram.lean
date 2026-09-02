import NightstreamFPrime.Export.MatrixProgram.Program
import NightstreamFPrime.Export.Stage1.PerApplicationSourceProjection
import NightstreamFPrime.Export.Stage1.PilotOrdinaryDirectPlan

/-!
Owns the compact package operands for the 1,330 non-Poseidon pilot rows.
The exact table preserves the canonical package order: witness instructions
first, then assertions. Six sparse source ranges reconstruct the existing
proof-oriented direct source map.

This module does not select pilot semantics or invent rows. Its public row
theorem requires the caller to load the identity-checked package row selected
by the Lean-authored table.
-/

namespace NightstreamFPrime.Export.Stage1.PilotOrdinaryMatrixProgram

open NightstreamFPrime.Export.MatrixProgram
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Lifecycle
open PilotOrdinaryRetainedBlocks
open PilotOrdinaryRetainedGeometry

abbrev Program := Lifecycle.Stage1.Application.Program

def instructionIndices : List Nat :=
  (PilotData.witnessInstructions ()).map fun instruction =>
    instruction.rowIndex

def assertionIndices : List Nat :=
  (PilotData.assertionRows ()).map fun row => row.rowIndex

/-- Exact physical package row indices in the same order as
`PilotOrdinaryDirectSource.sourceRows`. -/
def rowIndexReference : List Nat := instructionIndices ++ assertionIndices

theorem rowIndexReference_length : rowIndexReference.length = 1330 := by
  have sourceLength := PilotOrdinaryDirectSource.sourceRows_length
  unfold PilotOrdinaryDirectSource.sourceRows
    PilotOrdinaryDirectSource.instructionRows
    PilotOrdinaryDirectSource.assertionRows at sourceLength
  simpa [rowIndexReference, instructionIndices, assertionIndices] using
    sourceLength

/-- Random-access exact package order. A table is required because the
canonical package performs a stable witness/assertion partition. -/
def rowSchedule : IndexSchedule :=
  .indexTable rowIndexReference.toArray

@[simp] theorem rowSchedule_count : rowSchedule.count = 1330 := by
  change rowIndexReference.toArray.size = 1330
  simpa using rowIndexReference_length

theorem rowSchedule_indices : rowSchedule.indices = rowIndexReference := by
  simp [rowSchedule, IndexSchedule.indices]

theorem rowSchedule_index? (ordinal : Nat) :
    rowSchedule.index? ordinal = rowIndexReference[ordinal]? := by
  rw [IndexSchedule.index?_eq_getElem?, rowSchedule_indices]

def rowIndexAt (index : Fin 1330) : Nat :=
  rowIndexReference.get (Fin.cast rowIndexReference_length.symm index)

theorem rowSchedule_indexAt (index : Fin 1330) :
    rowSchedule.index? index.val = some (rowIndexAt index) := by
  rw [rowSchedule_index?]
  have bound : index.val < rowIndexReference.length := by
    rw [rowIndexReference_length]
    exact index.isLt
  rw [List.getElem?_eq_getElem bound]
  unfold rowIndexAt
  rfl

private theorem priorPublicTarget (index : Fin 270) :
    PilotSpartan.sourceToSpartan
        (PilotProduction.priorPublicInputStart + index.val) =
      14722239 + index.val := by
  have bound := index.isLt
  change PilotSpartan.sourceToSpartan (49393 + index.val) = _
  unfold PilotSpartan.sourceToSpartan
  rw [if_neg (by
    norm_num [PilotSpartan.priorPublicStart_value] at bound ⊢ <;> omega)]
  rw [if_pos (by
    norm_num [PilotSpartan.outputPreimageStart_value] at bound ⊢ <;> omega)]
  norm_num [PilotSpartan.firstPublicStart_value,
    PilotSpartan.priorPublicStart_value]

private theorem canonicalLocalTarget (index : Fin 264) :
    PilotSpartan.sourceToSpartan
        (PriorStateHash.hashEnd PilotProduction.priorInterface
          PilotProduction.witnessOffset + index.val) =
      7409986 + index.val := by
  have bound := index.isLt
  unfold PriorStateHash.hashEnd
  rw [PilotProduction.priorHashLogicalLength_eq,
    PilotProduction.witnessOffset_eq]
  unfold PilotSpartan.sourceToSpartan
  rw [if_neg (by
    norm_num [PilotSpartan.priorPublicStart_value] at bound ⊢ <;> omega)]
  rw [if_neg (by
    norm_num [PilotSpartan.outputPreimageStart_value] at bound ⊢ <;> omega)]
  rw [if_neg (by
    norm_num [PilotSpartan.outputDigestStart_value] at bound ⊢ <;> omega)]
  rw [if_neg (by
    norm_num [PilotSpartan.witnessStart_value] at bound ⊢ <;> omega)]
  norm_num [PilotSpartan.witnessPrivateStart_value,
    PilotSpartan.witnessStart_value] <;> omega

private theorem canonicalFreshTarget (index : Fin 788) :
    PilotSpartan.sourceToSpartan
        (PilotValues.logicalColumnCount + index.val) =
      14721450 + index.val := by
  have bound := index.isLt
  change PilotSpartan.sourceToSpartan (14721724 + index.val) = _
  unfold PilotSpartan.sourceToSpartan
  rw [if_neg (by
    norm_num [PilotSpartan.priorPublicStart_value] at bound ⊢ <;> omega)]
  rw [if_neg (by
    norm_num [PilotSpartan.outputPreimageStart_value] at bound ⊢ <;> omega)]
  rw [if_neg (by
    norm_num [PilotSpartan.outputDigestStart_value] at bound ⊢ <;> omega)]
  rw [if_neg (by
    norm_num [PilotSpartan.witnessStart_value] at bound ⊢ <;> omega)]
  norm_num [PilotSpartan.witnessPrivateStart_value,
    PilotSpartan.witnessStart_value] <;> omega

private theorem outputDigestTarget (index : Fin 4) :
    PilotSpartan.sourceToSpartan
        (PilotProduction.outputDigestStart + index.val) =
      14722509 + index.val := by
  have bound := index.isLt
  change PilotSpartan.sourceToSpartan (99056 + index.val) = _
  unfold PilotSpartan.sourceToSpartan
  rw [if_neg (by
    norm_num [PilotSpartan.priorPublicStart_value] at bound ⊢ <;> omega)]
  rw [if_neg (by
    norm_num [PilotSpartan.outputPreimageStart_value] at bound ⊢ <;> omega)]
  rw [if_neg (by
    norm_num [PilotSpartan.outputDigestStart_value] at bound ⊢ <;> omega)]
  rw [if_pos (by
    norm_num [PilotSpartan.witnessStart_value] at bound ⊢ <;> omega)]
  norm_num [PilotSpartan.secondPublicStart_value,
    PilotSpartan.outputDigestStart_value]

def priorDigestRange (program : Program) : SourceRange :=
  SourceRange.ofSemantic (PiCCSOrdinaryRetainedBlocks.priorLastBlock program)
    (PiCCSOrdinaryRetainedGeometry.priorLastStart program) 7409978 4 584

def canonicalLocalRange (program : Program) : SourceRange :=
  SourceRange.ofSemantic (canonicalLocalBlock program)
    (canonicalLocalStart program) 7409986 264 0

def outputStateRange (program : Program) : SourceRange :=
  SourceRange.ofSemantic (PiCCSOrdinaryRetainedBlocks.outputLastBlock program)
    (PiCCSOrdinaryRetainedGeometry.outputLastStart program) 14721442 4 584

def canonicalFreshRange (program : Program) : SourceRange :=
  SourceRange.ofSemantic (canonicalFreshBlock program)
    (canonicalFreshStart program) 14721450 788 0

def priorPublicRange (program : Program) : SourceRange :=
  SourceRange.ofSemantic
    (PiCCSOrdinaryRetainedBlocks.freshPublicInputBlock program)
    (PiCCSOrdinaryRetainedGeometry.freshPublicInputStart program)
    14722239 270 0

def outputDigestRange (program : Program) : SourceRange :=
  SourceRange.ofSemantic (outputDigestBlock program)
    (outputDigestStart program) 14722509 4 0

private theorem rangeValues (program : Program) :
    (priorDigestRange program).sourceStart = 7409978 ∧
    (priorDigestRange program).sourceCount = 4 ∧
    (canonicalLocalRange program).sourceStart = 7409986 ∧
    (canonicalLocalRange program).sourceCount = 264 ∧
    (outputStateRange program).sourceStart = 14721442 ∧
    (outputStateRange program).sourceCount = 4 ∧
    (canonicalFreshRange program).sourceStart = 14721450 ∧
    (canonicalFreshRange program).sourceCount = 788 ∧
    (priorPublicRange program).sourceStart = 14722239 ∧
    (priorPublicRange program).sourceCount = 270 ∧
    (outputDigestRange program).sourceStart = 14722509 ∧
    (outputDigestRange program).sourceCount = 4 := by
  norm_num [priorDigestRange, canonicalLocalRange, outputStateRange,
    canonicalFreshRange, priorPublicRange, outputDigestRange,
    SourceRange.ofSemantic]

theorem priorDigestRange_form?
    {program : Program} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth) (lane : Fin 4) :
    (priorDigestRange program).form? logicalWidth
        (PilotSpartan.sourceToSpartan
          (PilotProduction.priorDigestStart + lane.val)) =
      some ((PilotOrdinaryDirectPlan.Location.priorDigest lane).form
        geometry) := by
  rw [PilotOrdinaryDirectSource.priorDigest_targetColumn]
  simpa [priorDigestRange, PilotOrdinaryDirectPlan.Location.form,
    PilotOrdinaryDirectPlan.finalSlot] using
    (SourceRange.form?_ofSemantic
      (PiCCSOrdinaryRetainedBlocks.priorLastBlock program)
      (PiCCSOrdinaryRetainedGeometry.priorLastStart program)
      7409978 4 584
      (PiCCSOrdinaryRetainedGeometry.priorLastFits
        (PilotOrdinaryDirectPlan.piCcsGeometry geometry))
      (by
        change 588 ≤ 592
        norm_num) lane)

theorem canonicalLocalRange_form?
    {program : Program} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth) (index : Fin 264) :
    (canonicalLocalRange program).form? logicalWidth
        (PilotSpartan.sourceToSpartan
          (PriorStateHash.hashEnd PilotProduction.priorInterface
            PilotProduction.witnessOffset + index.val)) =
      some ((PilotOrdinaryDirectPlan.Location.canonicalLocal index).form
        geometry) := by
  rw [canonicalLocalTarget]
  simpa [canonicalLocalRange, PilotOrdinaryDirectPlan.Location.form] using
    (SourceRange.form?_ofSemantic (canonicalLocalBlock program)
      (canonicalLocalStart program) 7409986 264 0
      (canonicalLocalFits geometry) (by norm_num) index)

theorem outputStateRange_form?
    {program : Program} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth) (lane : Fin 4) :
    (outputStateRange program).form? logicalWidth
        (PilotSpartan.sourceToSpartan
          (PilotProduction.lifecycleOutputOffset +
            PilotValues.absorbCount * 592 + 584 + lane.val)) =
      some ((PilotOrdinaryDirectPlan.Location.outputState lane).form
        geometry) := by
  rw [PilotOrdinaryDirectSource.outputState_targetColumn]
  simpa [outputStateRange, PilotOrdinaryDirectPlan.Location.form,
    PilotOrdinaryDirectPlan.finalSlot] using
    (SourceRange.form?_ofSemantic
      (PiCCSOrdinaryRetainedBlocks.outputLastBlock program)
      (PiCCSOrdinaryRetainedGeometry.outputLastStart program)
      14721442 4 584
      (PiCCSOrdinaryRetainedGeometry.outputLastFits
        (PilotOrdinaryDirectPlan.piCcsGeometry geometry))
      (by
        change 588 ≤ 592
        norm_num) lane)

theorem canonicalFreshRange_form?
    {program : Program} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth) (index : Fin 788) :
    (canonicalFreshRange program).form? logicalWidth
        (PilotSpartan.sourceToSpartan
          (PilotValues.logicalColumnCount + index.val)) =
      some ((PilotOrdinaryDirectPlan.Location.canonicalFresh index).form
        geometry) := by
  rw [canonicalFreshTarget]
  simpa [canonicalFreshRange, PilotOrdinaryDirectPlan.Location.form] using
    (SourceRange.form?_ofSemantic (canonicalFreshBlock program)
      (canonicalFreshStart program) 14721450 788 0
      (canonicalFreshFits geometry) (by norm_num) index)

theorem priorPublicRange_form?
    {program : Program} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth) (index : Fin 270) :
    (priorPublicRange program).form? logicalWidth
        (PilotSpartan.sourceToSpartan
          (PilotProduction.priorPublicInputStart + index.val)) =
      some ((PilotOrdinaryDirectPlan.Location.priorPublic index).form
        geometry) := by
  rw [priorPublicTarget]
  simpa [priorPublicRange, PilotOrdinaryDirectPlan.Location.form] using
    (SourceRange.form?_ofSemantic
      (PiCCSOrdinaryRetainedBlocks.freshPublicInputBlock program)
      (PiCCSOrdinaryRetainedGeometry.freshPublicInputStart program)
      14722239 270 0
      (PiCCSOrdinaryRetainedGeometry.freshPublicInputFits
        (PilotOrdinaryDirectPlan.piCcsGeometry geometry))
      (by
        change 270 ≤ 270
        norm_num) index)

theorem outputDigestRange_form?
    {program : Program} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth) (index : Fin 4) :
    (outputDigestRange program).form? logicalWidth
        (PilotSpartan.sourceToSpartan
          (PilotProduction.outputDigestStart + index.val)) =
      some ((PilotOrdinaryDirectPlan.Location.outputDigest index).form
        geometry) := by
  rw [outputDigestTarget]
  simpa [outputDigestRange, PilotOrdinaryDirectPlan.Location.form] using
    (SourceRange.form?_ofSemantic (outputDigestBlock program)
      (outputDigestStart program) 14722509 4 0
      (outputDigestFits geometry) (by norm_num) index)

/-- Complete pilot ordinary substitution in increasing post-Spartan column
order. -/
def substitution (program : Program) : SourceSubstitution where
  ranges := [priorDigestRange program, canonicalLocalRange program,
    outputStateRange program, canonicalFreshRange program,
    priorPublicRange program, outputDigestRange program]

theorem substitution_priorDigest_form?
    {program : Program} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth) (lane : Fin 4) :
    (substitution program).form? logicalWidth
        (PilotSpartan.sourceToSpartan
          (PilotProduction.priorDigestStart + lane.val)) =
      some ((PilotOrdinaryDirectPlan.Location.priorDigest lane).form
        geometry) := by
  have target := PilotOrdinaryDirectSource.priorDigest_targetColumn lane
  have selected := priorDigestRange_form? geometry lane
  rw [target] at selected ⊢
  have bound := lane.isLt
  have values := rangeValues program
  have localNone := SourceRange.form?_eq_none_of_before
    (canonicalLocalRange program) logicalWidth (7409978 + lane.val) (by omega)
  have outputNone := SourceRange.form?_eq_none_of_before
    (outputStateRange program) logicalWidth (7409978 + lane.val) (by omega)
  have freshNone := SourceRange.form?_eq_none_of_before
    (canonicalFreshRange program) logicalWidth (7409978 + lane.val) (by omega)
  have publicNone := SourceRange.form?_eq_none_of_before
    (priorPublicRange program) logicalWidth (7409978 + lane.val) (by omega)
  have digestNone := SourceRange.form?_eq_none_of_before
    (outputDigestRange program) logicalWidth (7409978 + lane.val) (by omega)
  simp [substitution, SourceSubstitution.form?, selected, localNone,
    outputNone, freshNone, publicNone, digestNone]

theorem substitution_canonicalLocal_form?
    {program : Program} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth) (index : Fin 264) :
    (substitution program).form? logicalWidth
        (PilotSpartan.sourceToSpartan
          (PriorStateHash.hashEnd PilotProduction.priorInterface
            PilotProduction.witnessOffset + index.val)) =
      some ((PilotOrdinaryDirectPlan.Location.canonicalLocal index).form
        geometry) := by
  have target := canonicalLocalTarget index
  have selected := canonicalLocalRange_form? geometry index
  rw [target] at selected ⊢
  have bound := index.isLt
  have values := rangeValues program
  have priorNone := SourceRange.form?_eq_none_of_after
    (priorDigestRange program) logicalWidth (7409986 + index.val) (by omega)
  have outputNone := SourceRange.form?_eq_none_of_before
    (outputStateRange program) logicalWidth (7409986 + index.val) (by omega)
  have freshNone := SourceRange.form?_eq_none_of_before
    (canonicalFreshRange program) logicalWidth (7409986 + index.val) (by omega)
  have publicNone := SourceRange.form?_eq_none_of_before
    (priorPublicRange program) logicalWidth (7409986 + index.val) (by omega)
  have digestNone := SourceRange.form?_eq_none_of_before
    (outputDigestRange program) logicalWidth (7409986 + index.val) (by omega)
  simp [substitution, SourceSubstitution.form?, priorNone, selected,
    outputNone, freshNone, publicNone, digestNone]

theorem substitution_outputState_form?
    {program : Program} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth) (lane : Fin 4) :
    (substitution program).form? logicalWidth
        (PilotSpartan.sourceToSpartan
          (PilotProduction.lifecycleOutputOffset +
            PilotValues.absorbCount * 592 + 584 + lane.val)) =
      some ((PilotOrdinaryDirectPlan.Location.outputState lane).form
        geometry) := by
  have target := PilotOrdinaryDirectSource.outputState_targetColumn lane
  have selected := outputStateRange_form? geometry lane
  rw [target] at selected ⊢
  have bound := lane.isLt
  have values := rangeValues program
  have priorNone := SourceRange.form?_eq_none_of_after
    (priorDigestRange program) logicalWidth (14721442 + lane.val) (by omega)
  have localNone := SourceRange.form?_eq_none_of_after
    (canonicalLocalRange program) logicalWidth (14721442 + lane.val) (by omega)
  have freshNone := SourceRange.form?_eq_none_of_before
    (canonicalFreshRange program) logicalWidth (14721442 + lane.val) (by omega)
  have publicNone := SourceRange.form?_eq_none_of_before
    (priorPublicRange program) logicalWidth (14721442 + lane.val) (by omega)
  have digestNone := SourceRange.form?_eq_none_of_before
    (outputDigestRange program) logicalWidth (14721442 + lane.val) (by omega)
  simp [substitution, SourceSubstitution.form?, priorNone, localNone,
    selected, freshNone, publicNone, digestNone]

theorem substitution_canonicalFresh_form?
    {program : Program} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth) (index : Fin 788) :
    (substitution program).form? logicalWidth
        (PilotSpartan.sourceToSpartan
          (PilotValues.logicalColumnCount + index.val)) =
      some ((PilotOrdinaryDirectPlan.Location.canonicalFresh index).form
        geometry) := by
  have target := canonicalFreshTarget index
  have selected := canonicalFreshRange_form? geometry index
  rw [target] at selected ⊢
  have bound := index.isLt
  have values := rangeValues program
  have priorNone := SourceRange.form?_eq_none_of_after
    (priorDigestRange program) logicalWidth (14721450 + index.val) (by omega)
  have localNone := SourceRange.form?_eq_none_of_after
    (canonicalLocalRange program) logicalWidth (14721450 + index.val) (by omega)
  have outputNone := SourceRange.form?_eq_none_of_after
    (outputStateRange program) logicalWidth (14721450 + index.val) (by omega)
  have publicNone := SourceRange.form?_eq_none_of_before
    (priorPublicRange program) logicalWidth (14721450 + index.val) (by omega)
  have digestNone := SourceRange.form?_eq_none_of_before
    (outputDigestRange program) logicalWidth (14721450 + index.val) (by omega)
  simp [substitution, SourceSubstitution.form?, priorNone, localNone,
    outputNone, selected, publicNone, digestNone]

theorem substitution_priorPublic_form?
    {program : Program} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth) (index : Fin 270) :
    (substitution program).form? logicalWidth
        (PilotSpartan.sourceToSpartan
          (PilotProduction.priorPublicInputStart + index.val)) =
      some ((PilotOrdinaryDirectPlan.Location.priorPublic index).form
        geometry) := by
  have target := priorPublicTarget index
  have selected := priorPublicRange_form? geometry index
  rw [target] at selected ⊢
  have bound := index.isLt
  have values := rangeValues program
  have priorNone := SourceRange.form?_eq_none_of_after
    (priorDigestRange program) logicalWidth (14722239 + index.val) (by omega)
  have localNone := SourceRange.form?_eq_none_of_after
    (canonicalLocalRange program) logicalWidth (14722239 + index.val) (by omega)
  have outputNone := SourceRange.form?_eq_none_of_after
    (outputStateRange program) logicalWidth (14722239 + index.val) (by omega)
  have freshNone := SourceRange.form?_eq_none_of_after
    (canonicalFreshRange program) logicalWidth (14722239 + index.val) (by omega)
  have digestNone := SourceRange.form?_eq_none_of_before
    (outputDigestRange program) logicalWidth (14722239 + index.val) (by omega)
  simp [substitution, SourceSubstitution.form?, priorNone, localNone,
    outputNone, freshNone, selected, digestNone]

theorem substitution_outputDigest_form?
    {program : Program} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth) (index : Fin 4) :
    (substitution program).form? logicalWidth
        (PilotSpartan.sourceToSpartan
          (PilotProduction.outputDigestStart + index.val)) =
      some ((PilotOrdinaryDirectPlan.Location.outputDigest index).form
        geometry) := by
  have target := outputDigestTarget index
  have selected := outputDigestRange_form? geometry index
  rw [target] at selected ⊢
  have bound := index.isLt
  have values := rangeValues program
  have priorNone := SourceRange.form?_eq_none_of_after
    (priorDigestRange program) logicalWidth (14722509 + index.val) (by omega)
  have localNone := SourceRange.form?_eq_none_of_after
    (canonicalLocalRange program) logicalWidth (14722509 + index.val) (by omega)
  have outputNone := SourceRange.form?_eq_none_of_after
    (outputStateRange program) logicalWidth (14722509 + index.val) (by omega)
  have freshNone := SourceRange.form?_eq_none_of_after
    (canonicalFreshRange program) logicalWidth (14722509 + index.val) (by omega)
  have publicNone := SourceRange.form?_eq_none_of_after
    (priorPublicRange program) logicalWidth (14722509 + index.val) (by omega)
  simp [substitution, SourceSubstitution.form?, priorNone, localNone,
    outputNone, freshNone, publicNone, selected]

theorem substitution_location_form?
    {program : Program} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth)
    (location : PilotOrdinaryDirectPlan.Location) :
    (substitution program).form? logicalWidth
        (PilotSpartan.sourceToSpartan location.sourceColumn) =
      some (location.form geometry) := by
  cases location with
  | priorDigest lane => exact substitution_priorDigest_form? geometry lane
  | priorPublic index => exact substitution_priorPublic_form? geometry index
  | canonicalLocal index =>
      exact substitution_canonicalLocal_form? geometry index
  | outputState lane => exact substitution_outputState_form? geometry lane
  | canonicalFresh index =>
      exact substitution_canonicalFresh_form? geometry index
  | outputDigest index =>
      exact substitution_outputDigest_form? geometry index

theorem substitution_agrees_on_target
    {program : Program} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth)
    (column : Fin PilotSpartan.spartanColumnCount)
    (support : PilotOrdinaryDirectSource.Target column.val) :
    (substitution program).form? logicalWidth column.val =
      some ((PilotOrdinaryDirectPlan.sourceMap geometry).form column) := by
  rcases PilotOrdinaryDirectPlan.classifyTarget_complete support with
    ⟨decoded, found, mapped⟩
  change (substitution program).form? logicalWidth column.val =
    some (match PilotOrdinaryDirectPlan.classifyTarget column.val with
      | none => .empty
      | some value => value.location.form geometry)
  rw [found]
  have target :
      PilotSpartan.sourceToSpartan decoded.location.sourceColumn =
        column.val := by
    rw [decoded.owns, mapped]
  simpa only [target] using
    (substitution_location_form? geometry decoded.location)

private theorem programRow_support (index : Fin 1330) :
    (PilotOrdinaryDirectSource.programRow index).VarsSatisfy
      PilotOrdinaryDirectSource.Target := by
  exact PilotOrdinaryDirectSource.sourceRows_varsSatisfy _
    (List.get_mem _
      (Fin.cast PilotOrdinaryDirectSource.sourceRows_length.symm index))

theorem substitution_agrees_on_programRow
    {program : Program} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth) (index : Fin 1330) :
    let row := PilotOrdinaryDirectSource.programRow index
    Ordinary.AgreesOnTerms (substitution program)
        (PilotOrdinaryDirectPlan.sourceMap geometry) row.a.terms ∧
      Ordinary.AgreesOnTerms (substitution program)
        (PilotOrdinaryDirectPlan.sourceMap geometry) row.b.terms ∧
      Ordinary.AgreesOnTerms (substitution program)
        (PilotOrdinaryDirectPlan.sourceMap geometry) row.c.terms := by
  dsimp only
  have scope := programRow_support index
  refine ⟨?_, ?_, ?_⟩
  · intro term member bounded
    exact substitution_agrees_on_target geometry ⟨term.1, bounded⟩
      (scope.1 term member)
  · intro term member bounded
    exact substitution_agrees_on_target geometry ⟨term.1, bounded⟩
      (scope.2.1 term member)
  · intro term member bounded
    exact substitution_agrees_on_target geometry ⟨term.1, bounded⟩
      (scope.2.2 term member)

def block {program : Program} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth) : Ordinary.Block where
  rows := rowSchedule
  oneColumn := (oneColumn geometry).val
  substitution := substitution program
  projection := PerApplicationSourceProjection.pilot program

@[simp] theorem block_rowCount
    {program : Program} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth) :
    (block geometry).rowCount = 1330 := by
  exact rowSchedule_count

def matrixProgram {program : Program} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth) : MatrixProgram.Program where
  blocks := [.ordinary (block geometry)]

@[simp] theorem matrixProgram_rowCount
    {program : Program} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth) :
    (matrixProgram geometry).rowCount = 1330 := by
  rw [show matrixProgram geometry =
      MatrixProgram.Program.mk [.ordinary (block geometry)] by rfl]
  rw [MatrixProgram.Program.singleton_rowCount]
  exact block_rowCount geometry

end NightstreamFPrime.Export.Stage1.PilotOrdinaryMatrixProgram
