import NightstreamFPrime.Export.Stage1.PerApplicationSourceProjection
import NightstreamFPrime.Export.Stage1.PilotOrdinaryDirectPlan
import NightstreamFPrime.Layout.MatrixProgram.Program
import NightstreamFPrime.Export.Stage1.PilotOrdinaryDirectSource

namespace NightstreamFPrime.Export.Stage1.PilotOrdinaryMatrixProgram

open NightstreamFPrime.Layout.MatrixProgram
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

def rowIndexReference : List Nat := instructionIndices ++ assertionIndices

theorem rowIndexReference_length : rowIndexReference.length = 596 := by
  have sourceLength := PilotOrdinaryDirectSource.sourceRows_length
  unfold PilotOrdinaryDirectSource.sourceRows
    PilotOrdinaryDirectSource.instructionRows
    PilotOrdinaryDirectSource.assertionRows at sourceLength
  simpa [rowIndexReference, instructionIndices, assertionIndices] using
    sourceLength

def rowSchedule : IndexSchedule :=
  .indexTable rowIndexReference.toArray

@[simp] theorem rowSchedule_count : rowSchedule.count = 596 := by
  change rowIndexReference.toArray.size = 596
  simpa using rowIndexReference_length

theorem rowSchedule_indices : rowSchedule.indices = rowIndexReference := by
  simp [rowSchedule, IndexSchedule.indices]

theorem rowSchedule_index? (ordinal : Nat) :
    rowSchedule.index? ordinal = rowIndexReference[ordinal]? := by
  rw [IndexSchedule.index?_eq_getElem?, rowSchedule_indices]

def rowIndexAt (index : Fin 596) : Nat :=
  rowIndexReference.get (Fin.cast rowIndexReference_length.symm index)

theorem rowSchedule_indexAt (index : Fin 596) :
    rowSchedule.index? index.val = some (rowIndexAt index) := by
  rw [rowSchedule_index?]
  have bound : index.val < rowIndexReference.length := by
    rw [rowIndexReference_length]
    exact index.isLt
  rw [List.getElem?_eq_getElem bound]
  unfold rowIndexAt
  rfl

private theorem priorPublicTarget (index : Fin 320) :
    PilotPackageOrder.sourceToPackage
        (PilotProduction.priorPublicInputStart + index.val) =
      5343007 + index.val := by
  have bound := index.isLt
  change PilotPackageOrder.sourceToPackage (28913 + index.val) = _
  unfold PilotPackageOrder.sourceToPackage
  rw [ite_eq_right (by
    norm_num [PilotPackageOrder.priorPublicStart_value] at bound ⊢)]
  rw [ite_eq_left (by
    norm_num [PilotPackageOrder.outputPreimageStart_value] at bound ⊢; omega)]
  norm_num [PilotPackageOrder.firstPublicStart_value,
    PilotPackageOrder.priorPublicStart_value]

private theorem canonicalLocalTarget (index : Fin 268) :
    PilotPackageOrder.sourceToPackage
        (PriorStateHash.hashEnd PilotProduction.priorInterface
          PilotProduction.witnessOffset + index.val) =
      2700282 + index.val := by
  have bound := index.isLt
  unfold PriorStateHash.hashEnd
  rw [PilotProduction.priorHashLogicalLength_eq,
    PilotProduction.witnessOffset_eq]
  unfold PilotPackageOrder.sourceToPackage
  rw [ite_eq_right (by
    norm_num [PilotPackageOrder.priorPublicStart_value] at bound ⊢; omega)]
  rw [ite_eq_right (by
    norm_num [PilotPackageOrder.outputPreimageStart_value] at bound ⊢; omega)]
  rw [ite_eq_right (by
    norm_num [PilotPackageOrder.outputDigestStart_value] at bound ⊢; omega)]
  rw [ite_eq_right (by
    norm_num [PilotPackageOrder.witnessStart_value] at bound ⊢; omega)]
  norm_num [PilotPackageOrder.witnessPrivateStart_value,
    PilotPackageOrder.witnessStart_value]; omega

private theorem outputDigestTarget (index : Fin 4) :
    PilotPackageOrder.sourceToPackage
        (PilotProduction.outputDigestStart + index.val) =
      5343327 + index.val := by
  have bound := index.isLt
  change PilotPackageOrder.sourceToPackage (58146 + index.val) = _
  unfold PilotPackageOrder.sourceToPackage
  rw [ite_eq_right (by
    norm_num [PilotPackageOrder.priorPublicStart_value] at bound ⊢; omega)]
  rw [ite_eq_right (by
    norm_num [PilotPackageOrder.outputPreimageStart_value] at bound ⊢; omega)]
  rw [ite_eq_right (by
    norm_num [PilotPackageOrder.outputDigestStart_value] at bound ⊢)]
  rw [ite_eq_left (by
    norm_num [PilotPackageOrder.witnessStart_value] at bound ⊢; omega)]
  norm_num [PilotPackageOrder.secondPublicStart_value,
    PilotPackageOrder.outputDigestStart_value]

def priorDigestRange (program : Program) : SourceRange :=
  SourceRange.ofSemantic (PiCCSOrdinaryRetainedBlocks.priorLastBlock program)
    (PiCCSOrdinaryRetainedGeometry.priorLastStart program) 2700266 4 1080

def canonicalLocalRange (program : Program) : SourceRange :=
  SourceRange.ofSemantic (canonicalLocalBlock program)
    (canonicalLocalStart program) 2700282 268 0

def outputStateRange (program : Program) : SourceRange :=
  SourceRange.ofSemantic (PiCCSOrdinaryRetainedBlocks.outputLastBlock program)
    (PiCCSOrdinaryRetainedGeometry.outputLastStart program) 5342990 4 1080

def priorPublicRange (program : Program) : SourceRange :=
  SourceRange.ofSemantic
    (PiCCSOrdinaryRetainedBlocks.freshPublicInputBlock program)
    (PiCCSOrdinaryRetainedGeometry.freshPublicInputStart program)
    5343007 320 0

def outputDigestRange (program : Program) : SourceRange :=
  SourceRange.ofSemantic (outputDigestBlock program)
    (outputDigestStart program) 5343327 4 0

private theorem rangeValues (program : Program) :
    (priorDigestRange program).sourceStart = 2700266 ∧
    (priorDigestRange program).sourceCount = 4 ∧
    (canonicalLocalRange program).sourceStart = 2700282 ∧
    (canonicalLocalRange program).sourceCount = 268 ∧
    (outputStateRange program).sourceStart = 5342990 ∧
    (outputStateRange program).sourceCount = 4 ∧
    (priorPublicRange program).sourceStart = 5343007 ∧
    (priorPublicRange program).sourceCount = 320 ∧
    (outputDigestRange program).sourceStart = 5343327 ∧
    (outputDigestRange program).sourceCount = 4 := by
  norm_num [priorDigestRange, canonicalLocalRange, outputStateRange,
    priorPublicRange, outputDigestRange,
    SourceRange.ofSemantic]

theorem priorDigestRange_form?
    {program : Program} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth) (lane : Fin 4) :
    (priorDigestRange program).form? logicalWidth
        (PilotPackageOrder.sourceToPackage
          (PilotProduction.priorDigestStart + lane.val)) =
      some ((PilotOrdinaryDirectPlan.Location.priorDigest lane).form
        geometry) := by
  rw [PilotOrdinaryDirectSource.priorDigest_targetColumn]
  simpa [priorDigestRange, PilotOrdinaryDirectPlan.Location.form,
    PilotOrdinaryDirectPlan.finalSlot] using!
    (SourceRange.form?_ofSemantic
      (PiCCSOrdinaryRetainedBlocks.priorLastBlock program)
      (PiCCSOrdinaryRetainedGeometry.priorLastStart program)
      2700266 4 1080
      (PiCCSOrdinaryRetainedGeometry.priorLastFits
        (PilotOrdinaryDirectPlan.piCcsGeometry geometry))
      (by
        change 1084 ≤ 1096
        norm_num) lane)

theorem canonicalLocalRange_form?
    {program : Program} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth) (index : Fin 268) :
    (canonicalLocalRange program).form? logicalWidth
        (PilotPackageOrder.sourceToPackage
          (PriorStateHash.hashEnd PilotProduction.priorInterface
            PilotProduction.witnessOffset + index.val)) =
      some ((PilotOrdinaryDirectPlan.Location.canonicalLocal index).form
        geometry) := by
  rw [canonicalLocalTarget]
  simpa [canonicalLocalRange, PilotOrdinaryDirectPlan.Location.form] using!
    (SourceRange.form?_ofSemantic (canonicalLocalBlock program)
      (canonicalLocalStart program) 2700282 268 0
      (canonicalLocalFits geometry) (by norm_num) index)

theorem outputStateRange_form?
    {program : Program} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth) (lane : Fin 4) :
    (outputStateRange program).form? logicalWidth
        (PilotPackageOrder.sourceToPackage
          (PilotProduction.lifecycleOutputOffset +
            PilotValues.absorbCount * 1096 + 1080 + lane.val)) =
      some ((PilotOrdinaryDirectPlan.Location.outputState lane).form
        geometry) := by
  rw [PilotOrdinaryDirectSource.outputState_targetColumn]
  simpa [outputStateRange, PilotOrdinaryDirectPlan.Location.form,
    PilotOrdinaryDirectPlan.finalSlot] using!
    (SourceRange.form?_ofSemantic
      (PiCCSOrdinaryRetainedBlocks.outputLastBlock program)
      (PiCCSOrdinaryRetainedGeometry.outputLastStart program)
      5342990 4 1080
      (PiCCSOrdinaryRetainedGeometry.outputLastFits
        (PilotOrdinaryDirectPlan.piCcsGeometry geometry))
      (by
        change 1084 ≤ 1096
        norm_num) lane)

theorem priorPublicRange_form?
    {program : Program} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth) (index : Fin 320) :
    (priorPublicRange program).form? logicalWidth
        (PilotPackageOrder.sourceToPackage
          (PilotProduction.priorPublicInputStart + index.val)) =
      some ((PilotOrdinaryDirectPlan.Location.priorPublic index).form
        geometry) := by
  rw [priorPublicTarget]
  simpa [priorPublicRange, PilotOrdinaryDirectPlan.Location.form] using!
    (SourceRange.form?_ofSemantic
      (PiCCSOrdinaryRetainedBlocks.freshPublicInputBlock program)
      (PiCCSOrdinaryRetainedGeometry.freshPublicInputStart program)
      5343007 320 0
      (PiCCSOrdinaryRetainedGeometry.freshPublicInputFits
        (PilotOrdinaryDirectPlan.piCcsGeometry geometry))
      (by
        change 320 ≤ 320
        norm_num) index)

theorem outputDigestRange_form?
    {program : Program} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth) (index : Fin 4) :
    (outputDigestRange program).form? logicalWidth
        (PilotPackageOrder.sourceToPackage
          (PilotProduction.outputDigestStart + index.val)) =
      some ((PilotOrdinaryDirectPlan.Location.outputDigest index).form
        geometry) := by
  rw [outputDigestTarget]
  simpa [outputDigestRange, PilotOrdinaryDirectPlan.Location.form] using!
    (SourceRange.form?_ofSemantic (outputDigestBlock program)
      (outputDigestStart program) 5343327 4 0
      (outputDigestFits geometry) (by norm_num) index)

def substitution (program : Program) : SourceSubstitution where
  ranges := [priorDigestRange program, canonicalLocalRange program,
    outputStateRange program, priorPublicRange program,
    outputDigestRange program]

theorem substitution_priorDigest_form?
    {program : Program} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth) (lane : Fin 4) :
    (substitution program).form? logicalWidth
        (PilotPackageOrder.sourceToPackage
          (PilotProduction.priorDigestStart + lane.val)) =
      some ((PilotOrdinaryDirectPlan.Location.priorDigest lane).form
        geometry) := by
  have target := PilotOrdinaryDirectSource.priorDigest_targetColumn lane
  have selected := priorDigestRange_form? geometry lane
  rw [target] at selected ⊢
  have bound := lane.isLt
  have values := rangeValues program
  have localNone := SourceRange.form?_eq_none_of_before
    (canonicalLocalRange program) logicalWidth (2700266 + lane.val) (by omega)
  have outputNone := SourceRange.form?_eq_none_of_before
    (outputStateRange program) logicalWidth (2700266 + lane.val) (by omega)
  have publicNone := SourceRange.form?_eq_none_of_before
    (priorPublicRange program) logicalWidth (2700266 + lane.val) (by omega)
  have digestNone := SourceRange.form?_eq_none_of_before
    (outputDigestRange program) logicalWidth (2700266 + lane.val) (by omega)
  simp [substitution, SourceSubstitution.form?, selected, localNone,
    outputNone, publicNone, digestNone]

theorem substitution_canonicalLocal_form?
    {program : Program} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth) (index : Fin 268) :
    (substitution program).form? logicalWidth
        (PilotPackageOrder.sourceToPackage
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
    (priorDigestRange program) logicalWidth (2700282 + index.val) (by omega)
  have outputNone := SourceRange.form?_eq_none_of_before
    (outputStateRange program) logicalWidth (2700282 + index.val) (by omega)
  have publicNone := SourceRange.form?_eq_none_of_before
    (priorPublicRange program) logicalWidth (2700282 + index.val) (by omega)
  have digestNone := SourceRange.form?_eq_none_of_before
    (outputDigestRange program) logicalWidth (2700282 + index.val) (by omega)
  simp [substitution, SourceSubstitution.form?, priorNone, selected,
    outputNone, publicNone, digestNone]

theorem substitution_outputState_form?
    {program : Program} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth) (lane : Fin 4) :
    (substitution program).form? logicalWidth
        (PilotPackageOrder.sourceToPackage
          (PilotProduction.lifecycleOutputOffset +
            PilotValues.absorbCount * 1096 + 1080 + lane.val)) =
      some ((PilotOrdinaryDirectPlan.Location.outputState lane).form
        geometry) := by
  have target := PilotOrdinaryDirectSource.outputState_targetColumn lane
  have selected := outputStateRange_form? geometry lane
  rw [target] at selected ⊢
  have bound := lane.isLt
  have values := rangeValues program
  have priorNone := SourceRange.form?_eq_none_of_after
    (priorDigestRange program) logicalWidth (5342990 + lane.val) (by omega)
  have localNone := SourceRange.form?_eq_none_of_after
    (canonicalLocalRange program) logicalWidth (5342990 + lane.val) (by omega)
  have publicNone := SourceRange.form?_eq_none_of_before
    (priorPublicRange program) logicalWidth (5342990 + lane.val) (by omega)
  have digestNone := SourceRange.form?_eq_none_of_before
    (outputDigestRange program) logicalWidth (5342990 + lane.val) (by omega)
  simp [substitution, SourceSubstitution.form?, priorNone, localNone,
    selected, publicNone, digestNone]

theorem substitution_priorPublic_form?
    {program : Program} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth) (index : Fin 320) :
    (substitution program).form? logicalWidth
        (PilotPackageOrder.sourceToPackage
          (PilotProduction.priorPublicInputStart + index.val)) =
      some ((PilotOrdinaryDirectPlan.Location.priorPublic index).form
        geometry) := by
  have target := priorPublicTarget index
  have selected := priorPublicRange_form? geometry index
  rw [target] at selected ⊢
  have bound := index.isLt
  have values := rangeValues program
  have priorNone := SourceRange.form?_eq_none_of_after
    (priorDigestRange program) logicalWidth (5343007 + index.val) (by omega)
  have localNone := SourceRange.form?_eq_none_of_after
    (canonicalLocalRange program) logicalWidth (5343007 + index.val) (by omega)
  have outputNone := SourceRange.form?_eq_none_of_after
    (outputStateRange program) logicalWidth (5343007 + index.val) (by omega)
  have digestNone := SourceRange.form?_eq_none_of_before
    (outputDigestRange program) logicalWidth (5343007 + index.val) (by omega)
  simp [substitution, SourceSubstitution.form?, priorNone, localNone,
    outputNone, selected, digestNone]

theorem substitution_outputDigest_form?
    {program : Program} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth) (index : Fin 4) :
    (substitution program).form? logicalWidth
        (PilotPackageOrder.sourceToPackage
          (PilotProduction.outputDigestStart + index.val)) =
      some ((PilotOrdinaryDirectPlan.Location.outputDigest index).form
        geometry) := by
  have target := outputDigestTarget index
  have selected := outputDigestRange_form? geometry index
  rw [target] at selected ⊢
  have bound := index.isLt
  have values := rangeValues program
  have priorNone := SourceRange.form?_eq_none_of_after
    (priorDigestRange program) logicalWidth (5343327 + index.val) (by omega)
  have localNone := SourceRange.form?_eq_none_of_after
    (canonicalLocalRange program) logicalWidth (5343327 + index.val) (by omega)
  have outputNone := SourceRange.form?_eq_none_of_after
    (outputStateRange program) logicalWidth (5343327 + index.val) (by omega)
  have publicNone := SourceRange.form?_eq_none_of_after
    (priorPublicRange program) logicalWidth (5343327 + index.val) (by omega)
  simp [substitution, SourceSubstitution.form?, priorNone, localNone,
    outputNone, publicNone, selected]

theorem substitution_location_form?
    {program : Program} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth)
    (location : PilotOrdinaryDirectPlan.Location) :
    (substitution program).form? logicalWidth
        (PilotPackageOrder.sourceToPackage location.sourceColumn) =
      some (location.form geometry) := by
  cases location with
  | priorDigest lane => exact substitution_priorDigest_form? geometry lane
  | priorPublic index => exact substitution_priorPublic_form? geometry index
  | canonicalLocal index =>
      exact substitution_canonicalLocal_form? geometry index
  | outputState lane => exact substitution_outputState_form? geometry lane
  | outputDigest index =>
      exact substitution_outputDigest_form? geometry index

theorem substitution_agrees_on_target
    {program : Program} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth)
    (column : Fin PilotPackageOrder.packageColumnCount)
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
      PilotPackageOrder.sourceToPackage decoded.location.sourceColumn =
        column.val := by
    rw [decoded.owns, mapped]
  simpa only [target] using
    (substitution_location_form? geometry decoded.location)

private theorem programRow_support (index : Fin 596) :
    (PilotOrdinaryDirectSource.programRow index).VarsSatisfy
      PilotOrdinaryDirectSource.Target := by
  exact PilotOrdinaryDirectSource.sourceRows_varsSatisfy _
    (List.get_mem _
      (Fin.cast PilotOrdinaryDirectSource.sourceRows_length.symm index))

theorem substitution_agrees_on_programRow
    {program : Program} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth) (index : Fin 596) :
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
    (block geometry).rowCount = 596 := by
  exact rowSchedule_count

def matrixProgram {program : Program} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth) : MatrixProgram.Program where
  blocks := [.ordinary (block geometry)]

@[simp] theorem matrixProgram_rowCount
    {program : Program} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth) :
    (matrixProgram geometry).rowCount = 596 := by
  rw [show matrixProgram geometry =
      MatrixProgram.Program.mk [.ordinary (block geometry)] by rfl]
  rw [MatrixProgram.Program.singleton_rowCount]
  exact block_rowCount geometry

end NightstreamFPrime.Export.Stage1.PilotOrdinaryMatrixProgram
