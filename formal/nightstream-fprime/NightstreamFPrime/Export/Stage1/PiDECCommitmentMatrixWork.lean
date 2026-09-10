import NightstreamFPrime.Export.Stage1.PiDECOrdinarySourceWork
import NightstreamFPrime.Export.Stage1.PiDECMatrixProgramSemantics
import NightstreamFPrime.Export.Stage1.PerApplicationMatrixProgram
import NightstreamFPrime.Export.MatrixProgram.RetainedWork

/-!
Counted retained compilation of the selected 1188 PiDEC commitment rows.
The source row is generated once. Its exact A/B/C lists feed the two retained
blocks, with every zero constant, duplicate and entry order preserved.
SuperNeo v1.1 Section 7.5 and Appendix B.4 own the commitment equation.
The existing commitmentBlock_row? theorem connects these direct forms to
the package source and its projection; no package accessor runs here.

Clocks count dispatch, data reads, scalar literals/arithmetic, helper calls
and constructors. Calls include passing existing arguments; callee work is
added separately. Proof transport and clock instrumentation are erased from
this count. These are named operations, not machine instructions or wall time.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.PiDECCommitmentMatrixWork

open _root_.NightstreamFPrime.Spec
open _root_.NightstreamFPrime.Layout
open _root_.NightstreamFPrime.Layout.Stage1
open _root_.NightstreamFPrime.Layout.ProductionRelation
open _root_.NightstreamFPrime.Export.MatrixProgram
open _root_.NightstreamFPrime.Spec.Folding.PiRLC.PaperForkExtractionWork (Result)

attribute [local irreducible] PiDECOrdinarySourceWork.commitmentRow
  PiDECOrdinaryDirectSource.commitmentProgramRow PiDECInputCheck.relation

private theorem selectedWidth : 254260583 = PiDECInputCheck.logicalWidth :=
  Poseidon2HashChainV1Package.logicalWidth.symm

private abbrev selectedGeometry : PiDECRetainedGeometry.Geometry
    Poseidon2HashChainV1Package.application PiDECInputCheck.logicalWidth :=
  PerApplicationMatrixProgram.piDecGeometry Poseidon2HashChainV1Package.application

attribute [local irreducible] PerApplicationFixedPoint.logicalWidth
  PiDECSourceSupport.parentCommitmentStart

/-- Two kind constructors, four literals, two block records, pair and Result. -/
private def blocks (_ : Unit) : Result (RetainedBlock × RetainedBlock) :=
  ⟨(⟨.field, 1188, 188579034⟩, ⟨.field, 49248, 195244650⟩), 10⟩

private theorem blocks_value (application : Lifecycle.Stage1.Application.Program) :
    (blocks ()).value =
      (RetainedBlock.ofSemantic (PiDECRetainedBlocks.parentCommitmentBlock application)
        (PiDECRetainedGeometry.parentCommitmentStart application),
       RetainedBlock.ofSemantic (PiDECRetainedBlocks.proofBlock application)
        (PiDECRetainedGeometry.proofStart application)) := by
  have parentStart : PiDECRetainedGeometry.parentCommitmentStart application = 188579034 := by
    have total := PiRLCRetainedGeometry.prefixLogicalWidth_eq application
    rw [PiRLCRetainedGeometry.prefixLogicalWidth,
      PiRLCRetainedGeometry.productOutputBlock,
      LowNormBlock.Block.lift_coordinateCount,
      PiRLCProductSourceBlocks.outputBlock_coordinateCount] at total
    change PiRLCRetainedGeometry.productOutputStart application + 19008 * 41 = 188579034
    omega
  have proofStart : PiDECRetainedGeometry.proofStart application = 195244650 := by
    unfold PiDECRetainedGeometry.proofStart RunningTransitionRetainedGeometry.piDecStart
      RunningTransitionRetainedGeometry.roundC1Start RunningTransitionRetainedGeometry.roundC0Start
    rw [PiCCSActionPayloadBlock.logicalWidth_eq]
    change 195242354 + 28 * 41 + 28 * 41 = 195244650
    decide
  change ((⟨.field, 1188, 188579034⟩, ⟨.field, 49248, 195244650⟩) : RetainedBlock × RetainedBlock) =
    (⟨.field, 1188, PiDECRetainedGeometry.parentCommitmentStart application⟩,
     ⟨.field, 49248, PiDECRetainedGeometry.proofStart application⟩)
  rw [parentStart, proofStart]

private theorem oneColumn_eq {application : Lifecycle.Stage1.Application.Program}
    {columns : Nat} (geometry : PiDECRetainedGeometry.Geometry application columns)
    (positive : 0 < columns) :
    (⟨0, positive⟩ : Fin columns) = PiDECRetainedGeometry.oneColumn geometry := by
  apply Fin.ext
  rfl

private def Supported (column : Nat) : Prop :=
  (20347121 ≤ column ∧ column < 20347121 + 1188) ∨
    (28972970 ≤ column ∧ column < 28972970 + 49248)

private theorem parent_column (index : Nat) :
    Spartan.sourceToSpartan (PiDECSourceSupport.parentCommitmentStart + index) =
      20347121 + index := by
  rw [Spartan.sourceToSpartan_add_of_piCcsLocal _ _ (by
    rw [PiDECSourceSupport.parentCommitmentStart_eq]
    decide)]
  rw [PiDECSourceSupport.parentCommitmentStart_eq]
  norm_num [Spartan.sourceToSpartan, Spartan.pilotSourceColumnCount,
    Spartan.proofInputSourceStart, Spartan.piCcsPhaseOffset, Spartan.piCcsLocalStart]

private theorem proof_column (index : Nat) :
    Spartan.sourceToSpartan (PiDECInputs.proofInputStart + index) = 28972970 + index := by
  rw [Spartan.sourceToSpartan_add_of_piCcsLocal _ _ (by decide)]
  rfl

private theorem row_supported (index : Fin 1188) :
    (PiDECOrdinarySourceWork.commitmentRow index).value.VarsSatisfy Supported := by
  rw [PiDECOrdinarySourceWork.commitmentRow_data]
  refine ⟨?_, ?_, ?_⟩
  · intro term member
    rcases List.mem_ofFn.mp member with ⟨child, rfl⟩
    have source : PiDECInputs.childCommitmentStart child + index.val =
        PiDECInputs.proofInputStart + (child.val * 1188 + index.val) := by
      change 28973248 + child.val * 1188 + index.val =
        28973248 + (child.val * 1188 + index.val)
      omega
    change Supported (Spartan.sourceToSpartan (PiDECInputs.childCommitmentStart child + index.val))
    rw [source, proof_column]
    apply Or.inr
    have childBound := child.isLt
    have indexBound := index.isLt
    constructor <;> omega
  · intro term member
    simp only [List.not_mem_nil] at member
  · intro term member
    have parent : Supported
        (Spartan.sourceToSpartan (PiDECSourceSupport.parentCommitmentStart + index.val)) := by
      rw [parent_column]
      apply Or.inl
      have bound := index.isLt
      constructor <;> omega
    exact (congrArg (fun pair : Nat × F => Supported pair.1)
      (List.mem_singleton.mp member)).mpr parent

private theorem sourceMap_at_location
    {application : Lifecycle.Stage1.Application.Program} {columns : Nat}
    (geometry : PiDECRetainedGeometry.Geometry application columns)
    (column : Fin Spartan.spartanColumnCount) (location : PiDECDirectPlan.Location)
    (represented : column.val = Spartan.sourceToSpartan location.sourceColumn) :
    (PiDECDirectPlan.sourceMap geometry).form column = location.form geometry := by
  have supported : PiDECSourceSupport.Target column.val := by
    rw [represented]
    exact PiDECSourceSupport.source_target _ location.sourceSupport
  have agrees := PiDECMatrixProgram.substitution_agrees_on_target geometry column supported
  have located := PiDECMatrixProgram.substitution_location_form? geometry location
  rw [← represented] at located
  exact Option.some.inj (agrees.symm.trans located)

/-- Read/match/payload/Result cost four. The None branch also constructs Unit,
calls empty, reads its value and constructs Result, costing six plus empty. -/
private def unpackForm {columns : Nat} (opened : Result (Option (SparseForm columns))) :
    Result (SparseForm columns) :=
  match opened.value with
  | some form => ⟨form, opened.work + 4⟩
  | none =>
      let result := SparseWork.empty ()
      ⟨result.value, opened.work + result.work + 6⟩

private theorem unpackForm_value {columns : Nat}
    (opened : Result (Option (SparseForm columns))) (form : SparseForm columns)
    (returned : opened.value = some form) : (unpackForm opened).value = form := by
  simp only [unpackForm, returned]

private theorem unpackForm_length_le {columns : Nat}
    (opened : Result (Option (SparseForm columns)))
    (lengths : ∀ form, opened.value = some form → form.entries.length ≤ 41) :
    (unpackForm opened).value.entries.length ≤ 41 := by
  cases returned : opened.value with
  | none => simp only [unpackForm, returned, SparseWork.empty, List.length_nil]; omega
  | some form => simpa only [unpackForm, returned] using lengths form returned

private theorem unpackForm_work_le {columns : Nat}
    (opened : Result (Option (SparseForm columns))) :
    (unpackForm opened).work ≤ opened.work + 9 := by
  cases returned : opened.value <;>
    simp only [unpackForm, returned, SparseWork.empty_work] <;> omega

attribute [local irreducible] unpackForm

private theorem unpackedRetained_length_le (columns : Nat) (block : RetainedBlock)
    (kind : block.kind = .field) (slot : Nat) :
    (unpackForm (RetainedWork.form? block columns slot)).value.entries.length ≤ 41 := by
  apply unpackForm_length_le
  intro form returned
  have length := RetainedWork.form?_length block columns slot form returned
  simpa only [kind, LowNormSlot.Kind.width, BalancedTernary.width] using length.le

/-- The typed source-row theorem limits this lookup to two disjoint ranges.
The wrapper charges comparison literal/comparison/branch (3), selected start
literal/subtraction/retained call (3), unpack call/value read/Result (3).
RetainedWork checks the slot and complete block geometry. -/
private def sourceForm (columns : Nat) (parent proof : RetainedBlock) (column : Nat) :
    Result (SparseForm columns) :=
  let opened := if column < 28972970 then
      RetainedWork.form? parent columns (column - 20347121)
    else RetainedWork.form? proof columns (column - 28972970)
  let result := unpackForm opened
  ⟨result.value, result.work + 9⟩

private theorem unpackChoice_length_le (columns : Nat) (condition : Prop) [Decidable condition]
    (left right : Result (Option (SparseForm columns)))
    (leftLength : (unpackForm left).value.entries.length ≤ 41)
    (rightLength : (unpackForm right).value.entries.length ≤ 41) :
    (let opened := if condition then left else right
     let result := unpackForm opened
     (⟨result.value, result.work + 9⟩ : Result (SparseForm columns))).value.entries.length ≤ 41 := by
  by_cases chosen : condition
  · simpa only [if_pos chosen] using leftLength
  · simpa only [if_neg chosen] using rightLength

private theorem sourceForm_length_le (columns : Nat) (parent proof : RetainedBlock)
    (parentKind : parent.kind = .field) (proofKind : proof.kind = .field) (column : Nat) :
    (sourceForm columns parent proof column).value.entries.length ≤ 41 :=
  @unpackChoice_length_le columns (column < 28972970)
    (inferInstance : Decidable (column < 28972970))
    (RetainedWork.form? parent columns (column - 20347121))
    (RetainedWork.form? proof columns (column - 28972970))
    (unpackedRetained_length_le columns parent parentKind (column - 20347121))
    (unpackedRetained_length_le columns proof proofKind (column - 28972970))

private theorem sourceForm_work_le (columns : Nat) (parent proof : RetainedBlock)
    (parentKind : parent.kind = .field) (proofKind : proof.kind = .field) (column : Nat) :
    (sourceForm columns parent proof column).work ≤ 13819 := by
  have parentWork := RetainedWork.form?_work_le parent columns (column - 20347121)
  have proofWork := RetainedWork.form?_work_le proof columns (column - 28972970)
  simp only [parentKind, proofKind, LowNormSlot.Kind.width, BalancedTernary.width] at parentWork proofWork
  have parentUnpack := unpackForm_work_le (RetainedWork.form? parent columns (column - 20347121))
  have proofUnpack := unpackForm_work_le (RetainedWork.form? proof columns (column - 28972970))
  by_cases before : column < 28972970
  · simp only [sourceForm, if_pos before]
    change (unpackForm (RetainedWork.form? parent columns (column - 20347121))).work + 9 ≤ 13819
    omega
  · simp only [sourceForm, if_neg before]
    change (unpackForm (RetainedWork.form? proof columns (column - 28972970))).work + 9 ≤ 13819
    omega

private theorem parentForm_correct
    {application : Lifecycle.Stage1.Application.Program} {columns : Nat}
    (geometry : PiDECRetainedGeometry.Geometry application columns) (index : Fin 1188) :
    (sourceForm columns (blocks ()).value.1 (blocks ()).value.2 (20347121 + index.val)).value =
        (PiDECDirectPlan.Location.parentCommitment index).form geometry ∧
      (sourceForm columns (blocks ()).value.1 (blocks ()).value.2
        (20347121 + index.val)).value.entries.length = 41 := by
  let form := (PiDECDirectPlan.Location.parentCommitment index).form geometry
  have opened : (RetainedWork.form? (blocks ()).value.1 columns index.val).value = some form := by
    rw [RetainedWork.form?_value, blocks_value application]
    exact RetainedBlock.form?_ofSemantic (PiDECRetainedBlocks.parentCommitmentBlock application)
      (PiDECRetainedGeometry.parentCommitmentStart application)
      (PiDECRetainedGeometry.parentCommitmentFits geometry) index
  have value := unpackForm_value _ form opened
  have length := RetainedWork.form?_length (blocks ()).value.1 columns index.val form opened
  have before : 20347121 + index.val < 28972970 := by omega
  have offset : 20347121 + index.val - 20347121 = index.val := by omega
  have resultValue : (sourceForm columns (blocks ()).value.1 (blocks ()).value.2
      (20347121 + index.val)).value = form := by
    simpa only [sourceForm, if_pos before, offset] using value
  refine ⟨resultValue, ?_⟩
  rw [resultValue]
  simpa only [blocks, LowNormSlot.Kind.width, BalancedTernary.width] using length

private theorem proofForm_correct
    {application : Lifecycle.Stage1.Application.Program} {columns : Nat}
    (geometry : PiDECRetainedGeometry.Geometry application columns) (index : Fin 49248) :
    (sourceForm columns (blocks ()).value.1 (blocks ()).value.2 (28972970 + index.val)).value =
        (PiDECDirectPlan.Location.proof index).form geometry ∧
      (sourceForm columns (blocks ()).value.1 (blocks ()).value.2
        (28972970 + index.val)).value.entries.length = 41 := by
  let form := (PiDECDirectPlan.Location.proof index).form geometry
  have opened : (RetainedWork.form? (blocks ()).value.2 columns index.val).value = some form := by
    rw [RetainedWork.form?_value, blocks_value application]
    exact RetainedBlock.form?_ofSemantic (PiDECRetainedBlocks.proofBlock application)
      (PiDECRetainedGeometry.proofStart application) (PiDECRetainedGeometry.proofFits geometry) index
  have value := unpackForm_value _ form opened
  have length := RetainedWork.form?_length (blocks ()).value.2 columns index.val form opened
  have before : ¬ 28972970 + index.val < 28972970 := by omega
  have offset : 28972970 + index.val - 28972970 = index.val := by omega
  have resultValue : (sourceForm columns (blocks ()).value.1 (blocks ()).value.2
      (28972970 + index.val)).value = form := by
    simpa only [sourceForm, if_neg before, offset] using value
  refine ⟨resultValue, ?_⟩
  rw [resultValue]
  simpa only [blocks, LowNormSlot.Kind.width, BalancedTernary.width] using length

private theorem sourceForm_correct
    {application : Lifecycle.Stage1.Application.Program} {columns : Nat}
    (geometry : PiDECRetainedGeometry.Geometry application columns)
    (column : Fin Spartan.spartanColumnCount) (supported : Supported column.val) :
    (sourceForm columns (blocks ()).value.1 (blocks ()).value.2 column.val).value =
        (PiDECDirectPlan.sourceMap geometry).form column ∧
      (sourceForm columns (blocks ()).value.1 (blocks ()).value.2 column.val).value.entries.length = 41 := by
  rcases supported with parent | proof
  · let index : Fin 1188 := ⟨column.val - 20347121, by omega⟩
    have coordinate : column.val = 20347121 + index.val := by dsimp only [index]; omega
    have represented : column.val = Spartan.sourceToSpartan
        (PiDECDirectPlan.Location.parentCommitment index).sourceColumn := by
      simpa only [PiDECDirectPlan.Location.sourceColumn, parent_column] using coordinate
    have mapped := sourceMap_at_location geometry column _ represented
    have correct := parentForm_correct geometry index
    rw [← coordinate] at correct
    exact ⟨correct.1.trans mapped.symm, correct.2⟩
  · let index : Fin 49248 := ⟨column.val - 28972970, by omega⟩
    have coordinate : column.val = 28972970 + index.val := by dsimp only [index]; omega
    have represented : column.val = Spartan.sourceToSpartan
        (PiDECDirectPlan.Location.proof index).sourceColumn := by
      simpa only [PiDECDirectPlan.Location.sourceColumn, proof_column] using coordinate
    have mapped := sourceMap_at_location geometry column _ represented
    have correct := proofForm_correct geometry index
    rw [← coordinate] at correct
    exact ⟨correct.1.trans mapped.symm, correct.2⟩

/-- Empty: dispatch, Unit, call, value and Result (5). Cons: dispatch/head/tail
reads (3), pair fields (2), four calls, four value reads, and Result (14). -/
private def terms (columns : Nat) (parent proof : RetainedBlock) :
    List (Nat × F) → Result (SparseForm columns)
  | [] =>
      let result := SparseWork.empty ()
      ⟨result.value, result.work + 5⟩
  | term :: rest =>
      let head := sourceForm columns parent proof term.1
      let tail := terms columns parent proof rest
      let scaled := SparseWork.scale term.2 head.value
      let joined := SparseWork.add scaled.value tail.value
      ⟨joined.value, head.work + tail.work + scaled.work + joined.work + 14⟩

private theorem terms_correct
    {application : Lifecycle.Stage1.Application.Program} {columns : Nat}
    (geometry : PiDECRetainedGeometry.Geometry application columns) (entries : List (Nat × F))
    (bounded : ∀ term ∈ entries, term.1 < Spartan.spartanColumnCount)
    (supported : ∀ term ∈ entries, Supported term.1) :
    (terms columns (blocks ()).value.1 (blocks ()).value.2 entries).value =
        SourceCompiler.compileTerms (PiDECDirectPlan.sourceMap geometry) entries bounded ∧
      (terms columns (blocks ()).value.1 (blocks ()).value.2 entries).value.entries.length =
        41 * entries.length := by
  induction entries with
  | nil => exact ⟨rfl, rfl⟩
  | cons term rest ih =>
      have head := sourceForm_correct geometry ⟨term.1, bounded term (by simp)⟩
        (supported term (by simp))
      have tail := ih (fun candidate member => bounded candidate (by simp [member]))
        (fun candidate member => supported candidate (by simp [member]))
      constructor
      · simp only [terms, SparseWork.add_value, SparseWork.scale_value,
          SourceCompiler.compileTerms, head.1, tail.1]
      · simp only [terms, SparseWork.add_value, SparseWork.scale_value,
          SparseForm.add, SparseForm.scale, List.length_append, List.length_map,
          head.2, tail.2, List.length_cons]
        omega

private theorem terms_work_le (columns : Nat) (parent proof : RetainedBlock)
    (parentKind : parent.kind = .field) (proofKind : proof.kind = .field)
    (entries : List (Nat × F)) :
    (terms columns parent proof entries).work ≤ 14843 * entries.length + 8 := by
  induction entries with
  | nil => simp only [terms, SparseWork.empty_work, List.length_nil]; omega
  | cons term rest ih =>
      have headWork := sourceForm_work_le columns parent proof parentKind proofKind term.1
      have headLength := sourceForm_length_le columns parent proof parentKind proofKind term.1
      have scaledLength : (SparseWork.scale term.2
          (sourceForm columns parent proof term.1).value).value.entries.length ≤ 41 := by
        simpa only [SparseWork.scale_value, SparseForm.scale, List.length_map] using headLength
      simp only [terms, SparseWork.scale_work, SparseWork.add_work, List.length_cons]
      omega

/-- Two affine-field reads, three calls, three value reads and Result (9). -/
private def combination (columns : Nat) (parent proof : RetainedBlock) (one : Fin columns)
    (row : R1CS.LinearCombination) : Result (SparseForm columns) :=
  let selected := terms columns parent proof row.terms
  let constant := SparseWork.singleton one row.constant
  let joined := SparseWork.add constant.value selected.value
  ⟨joined.value, selected.work + constant.work + joined.work + 9⟩

private theorem combination_correct
    {application : Lifecycle.Stage1.Application.Program} {columns : Nat}
    (geometry : PiDECRetainedGeometry.Geometry application columns) (one : Fin columns)
    (row : R1CS.LinearCombination) (bounded : SourceCompiler.CombinationBounded Spartan.spartanColumnCount row)
    (supported : row.VarsSatisfy Supported) :
    (combination columns (blocks ()).value.1 (blocks ()).value.2 one row).value =
        SourceCompiler.compileCombination (PiDECDirectPlan.sourceMap geometry) one row bounded ∧
      (combination columns (blocks ()).value.1 (blocks ()).value.2 one row).value.entries.length =
        1 + 41 * row.terms.length := by
  have selected := terms_correct geometry row.terms bounded supported
  constructor
  · simp only [combination, SparseWork.add_value, SparseWork.singleton_value,
      SourceCompiler.compileCombination, selected.1]
  · simp only [combination, SparseWork.add_value, SparseWork.singleton_value,
      SparseForm.add, SparseForm.singleton, List.length_append,
      List.length_cons, List.length_nil, selected.2]

private theorem combination_work_le (columns : Nat) (parent proof : RetainedBlock)
    (parentKind : parent.kind = .field) (proofKind : proof.kind = .field)
    (one : Fin columns) (row : R1CS.LinearCombination) :
    (combination columns parent proof one row).work ≤ 14843 * row.terms.length + 45 := by
  have selected := terms_work_le columns parent proof parentKind proofKind row.terms
  simp only [combination, SparseWork.singleton_work, SparseWork.add_work,
    SparseWork.singleton_value, SparseForm.singleton, List.length_cons, List.length_nil]
  omega

/-- Three row-field reads, four calls, one field literal, four value reads,
the Forms record and Result (14). -/
private def compileRow (columns : Nat) (parent proof : RetainedBlock) (one : Fin columns)
    (row : R1CS.Row) : Result (OrdinaryRow.Forms columns) :=
  let a := combination columns parent proof one row.a
  let b := combination columns parent proof one row.b
  let c := combination columns parent proof one row.c
  let selector := SparseWork.singleton one 1
  ⟨⟨selector.value, a.value, b.value, c.value⟩, a.work + b.work + c.work + selector.work + 14⟩

private theorem compileRow_value
    {application : Lifecycle.Stage1.Application.Program} {columns : Nat}
    (geometry : PiDECRetainedGeometry.Geometry application columns) (one : Fin columns)
    (row : R1CS.Row) (bounded : SourceCompiler.RowBounded Spartan.spartanColumnCount row)
    (supported : row.VarsSatisfy Supported) :
    (compileRow columns (blocks ()).value.1 (blocks ()).value.2 one row).value =
      SourceCompiler.compileRow (PiDECDirectPlan.sourceMap geometry) one row bounded := by
  have a := (combination_correct geometry one row.a bounded.1 supported.1).1
  have b := (combination_correct geometry one row.b bounded.2.1 supported.2.1).1
  have c := (combination_correct geometry one row.c bounded.2.2 supported.2.2).1
  simp only [compileRow, SparseWork.singleton_value, SourceCompiler.compileRow, a, b, c]

private theorem compileRow_lengths
    {application : Lifecycle.Stage1.Application.Program} {columns : Nat}
    (geometry : PiDECRetainedGeometry.Geometry application columns) (one : Fin columns)
    (row : R1CS.Row) (bounded : SourceCompiler.RowBounded Spartan.spartanColumnCount row)
    (supported : row.VarsSatisfy Supported) :
    let forms := (compileRow columns (blocks ()).value.1 (blocks ()).value.2 one row).value
    forms.selector.entries.length = 1 ∧ forms.a.entries.length = 1 + 41 * row.a.terms.length ∧
      forms.b.entries.length = 1 + 41 * row.b.terms.length ∧
      forms.c.entries.length = 1 + 41 * row.c.terms.length := by
  exact ⟨rfl, (combination_correct geometry one row.a bounded.1 supported.1).2,
    (combination_correct geometry one row.b bounded.2.1 supported.2.1).2,
    (combination_correct geometry one row.c bounded.2.2 supported.2.2).2⟩

private theorem compileRow_work_le (columns : Nat) (parent proof : RetainedBlock)
    (parentKind : parent.kind = .field) (proofKind : proof.kind = .field)
    (one : Fin columns) (row : R1CS.Row) :
    (compileRow columns parent proof one row).work ≤
      14843 * (row.a.terms.length + row.b.terms.length + row.c.terms.length) + 154 := by
  have a := combination_work_le columns parent proof parentKind proofKind one row.a
  have b := combination_work_le columns parent proof parentKind proofKind one row.b
  have c := combination_work_le columns parent proof parentKind proofKind one row.c
  simp only [compileRow, SparseWork.singleton_work]
  omega

/-- Three calls, Unit, retained value and pair-field reads (3), zero/Fin
constructors (2), source/compiled value reads (2), and Result total 12. -/
private def run (columns : Nat) (positive : 0 < columns) (index : Fin 1188) :
    Result (OrdinaryRow.Forms columns) :=
  let source := PiDECOrdinarySourceWork.commitmentRow index
  let retained := blocks ()
  let data := retained.value
  let compiled := compileRow columns data.1 data.2 ⟨0, positive⟩ source.value
  ⟨compiled.value, source.work + retained.work + compiled.work + 12⟩

private theorem source_bounded (index : Fin 1188) :
    SourceCompiler.RowBounded Spartan.spartanColumnCount
      (PiDECOrdinarySourceWork.commitmentRow index).value := by
  rw [PiDECOrdinarySourceWork.commitmentRow_value]
  exact PiDECOrdinaryDirectSource.commitmentProgramRow_bounded PiDECInputCheck.relation index

private theorem run_value
    {application : Lifecycle.Stage1.Application.Program} {columns : Nat}
    (positive : 0 < columns) (geometry : PiDECRetainedGeometry.Geometry application columns)
    (index : Fin 1188) :
    (run columns positive index).value =
      PiDECMatrixProgram.commitmentDirectForms PiDECInputCheck.relation geometry index := by
  have compiled := compileRow_value geometry (⟨0, positive⟩ : Fin columns)
    (PiDECOrdinarySourceWork.commitmentRow index).value (source_bounded index) (row_supported index)
  change (compileRow columns (blocks ()).value.1 (blocks ()).value.2 ⟨0, positive⟩
    (PiDECOrdinarySourceWork.commitmentRow index).value).value = _
  rw [compiled, oneColumn_eq geometry positive]
  exact SourceCompiler.compileRow_eq_of_row (PiDECDirectPlan.sourceMap geometry)
    (PiDECRetainedGeometry.oneColumn geometry)
    (PiDECOrdinarySourceWork.commitmentRow_value index) (source_bounded index)
    (PiDECOrdinaryDirectSource.commitmentProgramRow_bounded PiDECInputCheck.relation index)

private theorem run_lengths
    {application : Lifecycle.Stage1.Application.Program} {columns : Nat}
    (positive : 0 < columns) (geometry : PiDECRetainedGeometry.Geometry application columns)
    (index : Fin 1188) :
    (run columns positive index).value.selector.entries.length = 1 ∧
      (run columns positive index).value.a.entries.length = 657 ∧
      (run columns positive index).value.b.entries.length = 1 ∧
      (run columns positive index).value.c.entries.length = 42 := by
  have lengths := compileRow_lengths geometry (⟨0, positive⟩ : Fin columns)
    (PiDECOrdinarySourceWork.commitmentRow index).value (source_bounded index) (row_supported index)
  rcases PiDECOrdinarySourceWork.commitmentRow_lengths index with ⟨a, b, c⟩
  simpa only [a, b, c] using lengths

/-- 17 terms include one retained lookup, scale, append and loop step each.
Three affine wrappers and the selector/row wrapper supply the remaining 154.
The source call, two block records, run wrapper and width wrapper are added. -/
def commitmentFormsWork : Nat :=
  PiDECOrdinarySourceWork.commitmentRowWork + 10 + 17 * 14843 + 154 + 12 + 4

private theorem run_work_le (columns : Nat) (positive : 0 < columns) (index : Fin 1188) :
    (run columns positive index).work + 4 ≤ commitmentFormsWork := by
  have sourceWork := PiDECOrdinarySourceWork.commitmentRow_work_le index
  have compiledWork := compileRow_work_le columns (blocks ()).value.1 (blocks ()).value.2
    rfl rfl (⟨0, positive⟩ : Fin columns) (PiDECOrdinarySourceWork.commitmentRow index).value
  rcases PiDECOrdinarySourceWork.commitmentRow_lengths index with ⟨a, b, c⟩
  simp only [a, b, c] at compiledWork
  simp only [run, commitmentFormsWork]
  change (PiDECOrdinarySourceWork.commitmentRow index).work + 10 +
    (compileRow columns (blocks ()).value.1 (blocks ()).value.2 ⟨0, positive⟩
      (PiDECOrdinarySourceWork.commitmentRow index).value).work + 12 + 4 ≤ _
  omega

/-- Execute with the proved selected width literal. Literal/call/value/Result
cost four; the equality transport changes only the erased type index. -/
def commitmentForms (index : Fin 1188) :
    Result (OrdinaryRow.Forms PiDECInputCheck.logicalWidth) :=
  let result := run 254260583 (by decide) index
  ⟨selectedWidth ▸ result.value, result.work + 4⟩

private theorem cast_run_value {application : Lifecycle.Stage1.Application.Program}
    {columns output : Nat} (equal : columns = output) (positive : 0 < columns)
    (geometry : PiDECRetainedGeometry.Geometry application output) (index : Fin 1188) :
    (equal ▸ (run columns positive index).value) =
      PiDECMatrixProgram.commitmentDirectForms PiDECInputCheck.relation geometry index := by
  cases equal
  exact run_value positive geometry index

theorem commitmentForms_value (index : Fin 1188) :
    (commitmentForms index).value =
      PiDECMatrixProgram.commitmentDirectForms PiDECInputCheck.relation
        (PerApplicationMatrixProgram.piDecGeometry Poseidon2HashChainV1Package.application) index :=
  @cast_run_value Poseidon2HashChainV1Package.application 254260583 PiDECInputCheck.logicalWidth
    selectedWidth (by decide) selectedGeometry index

private theorem cast_lengths {columns output : Nat} (equal : columns = output)
    (forms : OrdinaryRow.Forms columns)
    (lengths : forms.selector.entries.length = 1 ∧ forms.a.entries.length = 657 ∧
      forms.b.entries.length = 1 ∧ forms.c.entries.length = 42) :
    (equal ▸ forms).selector.entries.length = 1 ∧ (equal ▸ forms).a.entries.length = 657 ∧
      (equal ▸ forms).b.entries.length = 1 ∧ (equal ▸ forms).c.entries.length = 42 := by
  cases equal
  exact lengths

theorem commitmentForms_lengths (index : Fin 1188) :
    (commitmentForms index).value.selector.entries.length = 1 ∧
      (commitmentForms index).value.a.entries.length = 657 ∧
      (commitmentForms index).value.b.entries.length = 1 ∧
      (commitmentForms index).value.c.entries.length = 42 :=
  @cast_lengths 254260583 PiDECInputCheck.logicalWidth selectedWidth
    (run 254260583 (by decide) index).value
    (@run_lengths Poseidon2HashChainV1Package.application 254260583 (by decide)
      (selectedWidth.symm ▸ selectedGeometry) index)

theorem commitmentForms_work_le (index : Fin 1188) :
    (commitmentForms index).work ≤ commitmentFormsWork :=
  run_work_le 254260583 (by decide) index

end NightstreamFPrime.Export.Stage1.PiDECCommitmentMatrixWork
