import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.SourceDisposition.Pivots

/-!
Exact dependency classification of the physical compiler input boundary.

Owns: exclusive classification of every retained physical-compiler input as
an exact source input or rewrite-terminal pivot, and inclusion of all 68
rowless outputs in the exact source input boundary.

Does not own: selected-row satisfaction, protocol acceptance, transcript
authority, commitment binding, costs, or permission to remove rows.

Assurance tier: artifact-checked for the fixed generated production profile
once this leaf validates.
-/

/-!
Emits constraints: none; this module classifies existing compiler-input boundary rows.

| Stable stage path | Obligation | Authority class |
|---|---|---|
| `f_prime.pi_ccs_nc.delayed.combined.source_disposition.input_boundary` | Prove every compiler input belongs to the explicit verifier-owned boundary. | checked artifact |

-/

namespace Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.SourceDisposition

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Program
open Nightstream.Implementation.R1CS.CheckedProgram
open Nightstream.Implementation.R1CS.ProjectionIndexedRows
open Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc
open Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized

/-! ## Compiler input classification -/

structure InputColumnShape where
  column : Nat
  sourceInput : Bool
  terminalPivot : Bool
deriving DecidableEq, Repr

def SourceInputColumn (column : Nat) : Prop :=
  column ∈ Provenance.sourceColumns ∧
    column ∉ SourceExecution.definitionOutputs

instance (column : Nat) : Decidable (SourceInputColumn column) := by
  unfold SourceInputColumn
  infer_instance

def inputColumnShape (column : Nat) : InputColumnShape :=
  { column
    sourceInput := decide (SourceInputColumn column)
    terminalPivot := decide (column ∈ terminalPivotColumns) }

def inputColumnShapeCheck (values : List InputColumnShape) : Bool :=
  values.all fun shape =>
    shape.sourceInput || shape.terminalPivot

def sourceInputShapeCheck (values : List InputColumnShape) : Bool :=
  values.all InputColumnShape.sourceInput

private theorem classifiedColumns_of_shapeCheck
    {values : List Nat}
    (checked : inputColumnShapeCheck
      (values.map inputColumnShape) = true) :
    ∀ column ∈ values,
      column ∈ SourceExecution.inputColumns ∨
        column ∈ terminalPivotColumns := by
  intro column member
  have shapeMember : inputColumnShape column ∈
      values.map inputColumnShape := List.mem_map.mpr ⟨column, member, rfl⟩
  have shapeTrue := (List.all_eq_true.mp checked) _ shapeMember
  have covered :
      (inputColumnShape column).sourceInput = true ∨
        (inputColumnShape column).terminalPivot = true := by
    simpa only [Bool.or_eq_true] using shapeTrue
  rcases covered with sourceInput | pivot
  · left
    apply (SourceExecution.mem_inputColumns_iff column).mpr
    exact of_decide_eq_true (by
      simpa only [inputColumnShape, SourceInputColumn] using sourceInput)
  · right
    exact of_decide_eq_true (by
      simpa only [inputColumnShape] using pivot)

private theorem sourceInputColumns_of_shapeCheck
    {values : List Nat}
    (checked : sourceInputShapeCheck
      (values.map inputColumnShape) = true) :
    ∀ column ∈ values, column ∈ SourceExecution.inputColumns := by
  intro column member
  have shapeMember : inputColumnShape column ∈
      values.map inputColumnShape := List.mem_map.mpr ⟨column, member, rfl⟩
  have sourceInput := (List.all_eq_true.mp checked) _ shapeMember
  apply (SourceExecution.mem_inputColumns_iff column).mpr
  exact of_decide_eq_true (by
    simpa only [inputColumnShape, SourceInputColumn] using sourceInput)

def retainedSlotColumns (values : List RawSourceSlot) : List Nat :=
  values.map RawSourceSlot.column

def retainedSlotChunks : List (List RawSourceSlot) := [
  Provenance.RetainedSlots.Chunk0.values,
  Provenance.RetainedSlots.Chunk1.values,
  Provenance.RetainedSlots.Chunk2.values,
  Provenance.RetainedSlots.Chunk3.values,
  Provenance.RetainedSlots.Chunk4.values,
  Provenance.RetainedSlots.Chunk5.values,
  Provenance.RetainedSlots.Chunk6.values,
  Provenance.RetainedSlots.Chunk7.values,
  Provenance.RetainedSlots.Chunk8.values,
  Provenance.RetainedSlots.Chunk9.values,
  Provenance.RetainedSlots.Chunk10.values,
  Provenance.RetainedSlots.Chunk11.values,
  Provenance.RetainedSlots.Chunk12.values,
  Provenance.RetainedSlots.Chunk13.values,
  Provenance.RetainedSlots.Chunk14.values,
  Provenance.RetainedSlots.Chunk15.values,
  Provenance.RetainedSlots.Chunk16.values,
  Provenance.RetainedSlots.Chunk17.values,
  Provenance.RetainedSlots.Chunk18.values,
  Provenance.RetainedSlots.Chunk19.values,
  Provenance.RetainedSlots.Chunk20.values,
  Provenance.RetainedSlots.Chunk21.values,
  Provenance.RetainedSlots.Chunk22.values,
  Provenance.RetainedSlots.Chunk23.values,
  Provenance.RetainedSlots.Chunk24.values]

private theorem retainedSlotChunks_exact :
    retainedSlotChunks.flatten = Provenance.retainedSlots := by
  simp only [retainedSlotChunks, Provenance.retainedSlots,
    Provenance.RetainedSlots.values, List.flatten_cons, List.flatten_nil,
    List.append_nil, List.append_assoc]

/-! The 25 retained-slot subjects are compact proof-free `(Nat, Bool, Bool)`
records.  Direct chunks zero through twenty-three contain exactly 128
records; direct chunk twenty-four is the complete 99-record remainder. -/

private theorem retainedInput0 :
    inputColumnShapeCheck ((retainedSlotColumns
      Provenance.RetainedSlots.Chunk0.values).map inputColumnShape) = true := by native_decide
private theorem retainedInput1 :
    inputColumnShapeCheck ((retainedSlotColumns
      Provenance.RetainedSlots.Chunk1.values).map inputColumnShape) = true := by native_decide
private theorem retainedInput2 :
    inputColumnShapeCheck ((retainedSlotColumns
      Provenance.RetainedSlots.Chunk2.values).map inputColumnShape) = true := by native_decide
private theorem retainedInput3 :
    inputColumnShapeCheck ((retainedSlotColumns
      Provenance.RetainedSlots.Chunk3.values).map inputColumnShape) = true := by native_decide
private theorem retainedInput4 :
    inputColumnShapeCheck ((retainedSlotColumns
      Provenance.RetainedSlots.Chunk4.values).map inputColumnShape) = true := by native_decide
private theorem retainedInput5 :
    inputColumnShapeCheck ((retainedSlotColumns
      Provenance.RetainedSlots.Chunk5.values).map inputColumnShape) = true := by native_decide
private theorem retainedInput6 :
    inputColumnShapeCheck ((retainedSlotColumns
      Provenance.RetainedSlots.Chunk6.values).map inputColumnShape) = true := by native_decide
private theorem retainedInput7 :
    inputColumnShapeCheck ((retainedSlotColumns
      Provenance.RetainedSlots.Chunk7.values).map inputColumnShape) = true := by native_decide
private theorem retainedInput8 :
    inputColumnShapeCheck ((retainedSlotColumns
      Provenance.RetainedSlots.Chunk8.values).map inputColumnShape) = true := by native_decide
private theorem retainedInput9 :
    inputColumnShapeCheck ((retainedSlotColumns
      Provenance.RetainedSlots.Chunk9.values).map inputColumnShape) = true := by native_decide
private theorem retainedInput10 :
    inputColumnShapeCheck ((retainedSlotColumns
      Provenance.RetainedSlots.Chunk10.values).map inputColumnShape) = true := by native_decide
private theorem retainedInput11 :
    inputColumnShapeCheck ((retainedSlotColumns
      Provenance.RetainedSlots.Chunk11.values).map inputColumnShape) = true := by native_decide
private theorem retainedInput12 :
    inputColumnShapeCheck ((retainedSlotColumns
      Provenance.RetainedSlots.Chunk12.values).map inputColumnShape) = true := by native_decide
private theorem retainedInput13 :
    inputColumnShapeCheck ((retainedSlotColumns
      Provenance.RetainedSlots.Chunk13.values).map inputColumnShape) = true := by native_decide
private theorem retainedInput14 :
    inputColumnShapeCheck ((retainedSlotColumns
      Provenance.RetainedSlots.Chunk14.values).map inputColumnShape) = true := by native_decide
private theorem retainedInput15 :
    inputColumnShapeCheck ((retainedSlotColumns
      Provenance.RetainedSlots.Chunk15.values).map inputColumnShape) = true := by native_decide
private theorem retainedInput16 :
    inputColumnShapeCheck ((retainedSlotColumns
      Provenance.RetainedSlots.Chunk16.values).map inputColumnShape) = true := by native_decide
private theorem retainedInput17 :
    inputColumnShapeCheck ((retainedSlotColumns
      Provenance.RetainedSlots.Chunk17.values).map inputColumnShape) = true := by native_decide
private theorem retainedInput18 :
    inputColumnShapeCheck ((retainedSlotColumns
      Provenance.RetainedSlots.Chunk18.values).map inputColumnShape) = true := by native_decide
private theorem retainedInput19 :
    inputColumnShapeCheck ((retainedSlotColumns
      Provenance.RetainedSlots.Chunk19.values).map inputColumnShape) = true := by native_decide
private theorem retainedInput20 :
    inputColumnShapeCheck ((retainedSlotColumns
      Provenance.RetainedSlots.Chunk20.values).map inputColumnShape) = true := by native_decide
private theorem retainedInput21 :
    inputColumnShapeCheck ((retainedSlotColumns
      Provenance.RetainedSlots.Chunk21.values).map inputColumnShape) = true := by native_decide
private theorem retainedInput22 :
    inputColumnShapeCheck ((retainedSlotColumns
      Provenance.RetainedSlots.Chunk22.values).map inputColumnShape) = true := by native_decide
private theorem retainedInput23 :
    inputColumnShapeCheck ((retainedSlotColumns
      Provenance.RetainedSlots.Chunk23.values).map inputColumnShape) = true := by native_decide
private theorem retainedInput24 :
    inputColumnShapeCheck ((retainedSlotColumns
      Provenance.RetainedSlots.Chunk24.values).map inputColumnShape) = true := by native_decide

private theorem retainedSlotColumn_classified
    {slot : RawSourceSlot} (member : slot ∈ Provenance.retainedSlots) :
    slot.column ∈ SourceExecution.inputColumns ∨
      slot.column ∈ terminalPivotColumns := by
  rw [← retainedSlotChunks_exact] at member
  rcases List.mem_flatten.mp member with ⟨chunk, chunkMember, slotMember⟩
  simp only [retainedSlotChunks, List.mem_cons, List.not_mem_nil,
    or_false] at chunkMember
  rcases chunkMember with rfl | rfl | rfl | rfl | rfl | rfl | rfl |
      rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl |
      rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl
  · exact classifiedColumns_of_shapeCheck retainedInput0 _
      (List.mem_map.mpr ⟨slot, slotMember, rfl⟩)
  · exact classifiedColumns_of_shapeCheck retainedInput1 _
      (List.mem_map.mpr ⟨slot, slotMember, rfl⟩)
  · exact classifiedColumns_of_shapeCheck retainedInput2 _
      (List.mem_map.mpr ⟨slot, slotMember, rfl⟩)
  · exact classifiedColumns_of_shapeCheck retainedInput3 _
      (List.mem_map.mpr ⟨slot, slotMember, rfl⟩)
  · exact classifiedColumns_of_shapeCheck retainedInput4 _
      (List.mem_map.mpr ⟨slot, slotMember, rfl⟩)
  · exact classifiedColumns_of_shapeCheck retainedInput5 _
      (List.mem_map.mpr ⟨slot, slotMember, rfl⟩)
  · exact classifiedColumns_of_shapeCheck retainedInput6 _
      (List.mem_map.mpr ⟨slot, slotMember, rfl⟩)
  · exact classifiedColumns_of_shapeCheck retainedInput7 _
      (List.mem_map.mpr ⟨slot, slotMember, rfl⟩)
  · exact classifiedColumns_of_shapeCheck retainedInput8 _
      (List.mem_map.mpr ⟨slot, slotMember, rfl⟩)
  · exact classifiedColumns_of_shapeCheck retainedInput9 _
      (List.mem_map.mpr ⟨slot, slotMember, rfl⟩)
  · exact classifiedColumns_of_shapeCheck retainedInput10 _
      (List.mem_map.mpr ⟨slot, slotMember, rfl⟩)
  · exact classifiedColumns_of_shapeCheck retainedInput11 _
      (List.mem_map.mpr ⟨slot, slotMember, rfl⟩)
  · exact classifiedColumns_of_shapeCheck retainedInput12 _
      (List.mem_map.mpr ⟨slot, slotMember, rfl⟩)
  · exact classifiedColumns_of_shapeCheck retainedInput13 _
      (List.mem_map.mpr ⟨slot, slotMember, rfl⟩)
  · exact classifiedColumns_of_shapeCheck retainedInput14 _
      (List.mem_map.mpr ⟨slot, slotMember, rfl⟩)
  · exact classifiedColumns_of_shapeCheck retainedInput15 _
      (List.mem_map.mpr ⟨slot, slotMember, rfl⟩)
  · exact classifiedColumns_of_shapeCheck retainedInput16 _
      (List.mem_map.mpr ⟨slot, slotMember, rfl⟩)
  · exact classifiedColumns_of_shapeCheck retainedInput17 _
      (List.mem_map.mpr ⟨slot, slotMember, rfl⟩)
  · exact classifiedColumns_of_shapeCheck retainedInput18 _
      (List.mem_map.mpr ⟨slot, slotMember, rfl⟩)
  · exact classifiedColumns_of_shapeCheck retainedInput19 _
      (List.mem_map.mpr ⟨slot, slotMember, rfl⟩)
  · exact classifiedColumns_of_shapeCheck retainedInput20 _
      (List.mem_map.mpr ⟨slot, slotMember, rfl⟩)
  · exact classifiedColumns_of_shapeCheck retainedInput21 _
      (List.mem_map.mpr ⟨slot, slotMember, rfl⟩)
  · exact classifiedColumns_of_shapeCheck retainedInput22 _
      (List.mem_map.mpr ⟨slot, slotMember, rfl⟩)
  · exact classifiedColumns_of_shapeCheck retainedInput23 _
      (List.mem_map.mpr ⟨slot, slotMember, rfl⟩)
  · exact classifiedColumns_of_shapeCheck retainedInput24 _
      (List.mem_map.mpr ⟨slot, slotMember, rfl⟩)

/-- Exact source-input and terminal-pivot classes are disjoint: every pivot
is a certified source-definition output, while source inputs exclude all
definition outputs. -/
theorem sourceInput_disjoint_terminalPivot {column : Nat}
    (sourceInput : column ∈ SourceExecution.inputColumns) :
    column ∉ terminalPivotColumns := by
  intro pivot
  exact ((SourceExecution.mem_inputColumns_iff column).mp sourceInput).2
    (terminalPivotColumns_subset_sourceOutputs column pivot)

/-- Every compiler-retained column is either a literal source input or an
exact rewrite-terminal pivot, and never both. This is an occurrence-wise
classification; it does not assert global retained-slot uniqueness. -/
theorem retainedColumns_classified :
    ∀ column ∈ CompilerExecution.retainedColumns,
      (column ∈ SourceExecution.inputColumns ∨
        column ∈ terminalPivotColumns) ∧
      ¬(column ∈ SourceExecution.inputColumns ∧
        column ∈ terminalPivotColumns) := by
  intro column member
  simp only [CompilerExecution.retainedColumns, List.mem_cons,
    List.mem_map] at member
  rcases member with rfl | ⟨slot, slotMember, rfl⟩
  · refine ⟨Or.inl SourceExecution.constantOne_mem_inputColumns, ?_⟩
    intro both
    exact sourceInput_disjoint_terminalPivot both.1 both.2
  · have classified := retainedSlotColumn_classified slotMember
    refine ⟨classified, ?_⟩
    intro both
    exact sourceInput_disjoint_terminalPivot both.1 both.2

theorem retainedColumns_subset_inputOrPivots :
    ∀ column ∈ CompilerExecution.retainedColumns,
      column ∈ SourceExecution.inputColumns ∨
        column ∈ terminalPivotColumns := by
  intro column member
  exact (retainedColumns_classified column member).1

/-! This final compiler-seed certificate contains exactly 68 proof-free
`(Nat, Bool, Bool)` records, one for each rowless output. -/
private theorem rowlessOutputsInput :
    sourceInputShapeCheck
      ((CompilerExecution.rowlessDefinitions.map Definition.output).map
        inputColumnShape) = true := by
  native_decide

private theorem mem_knownAfter
    {column : Nat} {known : List Nat} {values : List Definition}
    (member : column ∈ knownAfter known values) :
    column ∈ known ∨
      ∃ definition ∈ values, column = definition.output := by
  induction values generalizing known with
  | nil => exact Or.inl member
  | cons head tail inductionHypothesis =>
      rcases inductionHypothesis member with knownMember | defined
      · simp only [List.mem_cons] at knownMember
        rcases knownMember with equal | knownMember
        · exact Or.inr ⟨head, by simp, equal⟩
        · exact Or.inl knownMember
      · rcases defined with ⟨definition, definitionMember, equal⟩
        exact Or.inr ⟨definition, by simp [definitionMember], equal⟩

/-- Every rowless compiler output is a literal source input. -/
theorem rowlessDefinitionOutputs_subset_sourceInput :
    ∀ column ∈ CompilerExecution.rowlessDefinitions.map Definition.output,
      column ∈ SourceExecution.inputColumns := by
  exact sourceInputColumns_of_shapeCheck rowlessOutputsInput

/-- The physical compiler phase reads only literal source inputs or exact
rewrite-terminal pivots. The latter dependency requires interleaving source
and compiler execution; it cannot be replaced by seed preservation. -/
theorem physicalInputColumns_subset_sourceInputOrPivots :
    ∀ column ∈ CompilerExecution.physicalInputColumns,
      column ∈ SourceExecution.inputColumns ∨
        column ∈ terminalPivotColumns := by
  intro column member
  rcases mem_knownAfter member with retained | ⟨definition, defined, rfl⟩
  · exact retainedColumns_subset_inputOrPivots column retained
  · exact Or.inl (rowlessDefinitionOutputs_subset_sourceInput definition.output
      (List.mem_map.mpr ⟨definition, defined, rfl⟩))

end Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.SourceDisposition
