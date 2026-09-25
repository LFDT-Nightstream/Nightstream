import NightstreamFPrime.Export.Stage1.RunningTransitionReducedMatrixSemantics
import NightstreamFPrime.Layout.Stage1.RunningTransitionWordIndexing

/-! Decode the four data regions and three headers of each running group.
The matrix forms read the existing PiDEC and output-preimage coordinates. -/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.RunningTransitionReducedGroups

open NightstreamFPrime.Spec NightstreamFPrime.Circuit NightstreamFPrime.Layout
open NightstreamFPrime.Layout.Stage1 NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Layout.MatrixProgram
open NightstreamFPrime.Layout.MatrixProgram.AffineGrid
open NightstreamFPrime.Lifecycle NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Lifecycle.Stage1
open NightstreamFPrime.Lifecycle.PiCCS.v1_1
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra
open RunningTransitionReducedMatrixProgram RunningTransitionReducedMatrixSemantics
open RunningTransitionWordIndexing

private inductive Field where
  | commitment | publicInput | evalK | evalA
deriving DecidableEq

private def Field.start : Field → Nat
  | .commitment => 1
  | .publicInput => 1190
  | .evalK => 1461
  | .evalA => 1569

private def Field.width : Field → Nat
  | .commitment => PiCCSInputs.runningCommitmentWords
  | .publicInput => PiCCSInputs.runningPublicWords
  | .evalK => PiDECInputs.evalKWordsPerChild
  | .evalA => PiDECInputs.evalAWordsPerChild

private def Field.piDecBase : Field → Nat
  | .commitment => 0
  | .publicInput => PiDECInputs.publicInputStart - PiDECInputs.proofInputStart
  | .evalK => PiDECInputs.evalKInputStart - PiDECInputs.proofInputStart
  | .evalA => PiDECInputs.evalAInputStart - PiDECInputs.proofInputStart

private def Field.selected (field : Field) : Region :=
  region productionShape.runningCount 1 field.start field.width

private theorem data_bound (field : Field) (index : Fin field.width) :
    field.start + index.val < PiCCSInputs.runningGroupWords := by
  cases field with
  | commitment =>
      have bound : index.val < 1188 := index.isLt
      change 1 + index.val < 3081
      omega
  | publicInput =>
      have bound : index.val < 270 := index.isLt
      change 1190 + index.val < 3081
      omega
  | evalK =>
      have bound : index.val < 108 := index.isLt
      change 1461 + index.val < 3081
      omega
  | evalA =>
      have bound : index.val < 1512 := index.isLt
      change 1569 + index.val < 3081
      omega

private theorem piDec_bases : Field.publicInput.piDecBase = 44928 ∧
    Field.evalK.piDecBase = 19008 ∧ Field.evalA.piDecBase = 20736 := by
  change ((PiDECInputs.proofInputStart + 19008) + 1728) + 24192 -
      PiDECInputs.proofInputStart = 44928 ∧
    PiDECInputs.proofInputStart + 19008 - PiDECInputs.proofInputStart = 19008 ∧
    (PiDECInputs.proofInputStart + 19008) + 1728 - PiDECInputs.proofInputStart = 20736
  omega

private def position (field : Field) (index : Fin field.width) :
    Fin PiCCSInputs.runningGroupWords :=
  ⟨field.start + index.val, data_bound field index⟩

private def piDecIndex (source : Fin productionShape.runningCount)
    (field : Field) (index : Fin field.width) :
    Fin RunningTransitionSourceSupport.piDecCount :=
  ⟨field.piDecBase + source.val * field.width + index.val, by
    have sourceBound : source.val < 16 := source.isLt
    cases field with
    | commitment =>
        have bound : index.val < 1188 := index.isLt
        change 0 + source.val * 1188 + index.val < 49248
        omega
    | publicInput =>
        have bound : index.val < 270 := index.isLt
        change Field.publicInput.piDecBase + source.val * 270 + index.val < 49248
        rw [piDec_bases.1]
        omega
    | evalK =>
        have bound : index.val < 108 := index.isLt
        change Field.evalK.piDecBase + source.val * 108 + index.val < 49248
        rw [piDec_bases.2.1]
        omega
    | evalA =>
        have bound : index.val < 1512 := index.isLt
        change Field.evalA.piDecBase + source.val * 1512 + index.val < 49248
        rw [piDec_bases.2.2]
        omega⟩

private def outputIndex (source : Fin productionShape.runningCount)
    (field : Field) (index : Fin field.width) :
    Fin RunningTransitionSourceSupport.outputCount :=
  ⟨PiCCSInputs.runningGroupStart source.val + field.start + index.val, by
    have sourceBound : source.val < 16 := source.isLt
    have bound := data_bound field index
    change field.start + index.val < 3081 at bound
    change 96 + source.val * 3081 + field.start + index.val < 49393
    omega⟩

private def inputForm {program : ApplicationProgram} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth) (source : Fin productionShape.runningCount)
    (field : Field) (index : Fin field.width) : SparseForm logicalWidth :=
  (RunningTransitionDirectPlan.Location.piDec (piDecIndex source field index)).form geometry

private def outputForm {program : ApplicationProgram} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth) (source : Fin productionShape.runningCount)
    (field : Field) (index : Fin field.width) : SparseForm logicalWidth :=
  (RunningTransitionDirectPlan.Location.output (outputIndex source field index)).form geometry

private theorem outside_rule (selected : Region) (wire : RetainedBlock)
    (slot majorStride middleStride minorStride : Nat) (coefficient : F)
    (logicalWidth oneColumn : Nat) (source minor : Nat)
    (outside : minor < selected.minorStart ∨
      selected.minorStart + selected.minorCount ≤ minor) :
    (retained selected wire slot majorStride middleStride minorStride coefficient).form?
      logicalWidth oneColumn ⟨source, 0, minor⟩ = some none := by
  apply Rule.form?_eq_some_none
  apply Region.offsets?_eq_none_of_outside
  intro inside
  dsimp only [retained] at inside
  rcases outside with before | after <;> omega

private theorem four_forms {logicalWidth : Nat} (rules : Fin 4 → Rule)
    (oneColumn : Nat) (coordinate : Coordinate)
    (results : Fin 4 → Option (SparseForm logicalWidth))
    (loaded : ∀ index, (rules index).form? logicalWidth oneColumn coordinate =
      some (results index)) :
    (AffineGrid.Program.mk [rules 0, rules 1, rules 2, rules 3]).form?
      logicalWidth oneColumn coordinate =
      some (combine [results 0, results 1, results 2, results 3]) := by
  exact AffineGrid.Program.form?_of_results _ oneColumn coordinate _
    (.cons (loaded 0) (.cons (loaded 1) (.cons (loaded 2) (.cons (loaded 3) .nil))))

private def payloadRule (wire : RetainedBlock) (base stride : Field → Nat)
    (field : Field) : Rule :=
  retained field.selected wire (base field) (stride field) 0 1 1

private theorem payload_selected {logicalWidth : Nat}
    (wire : RetainedBlock) (fits : wire.start + wire.coordinateCount ≤ logicalWidth)
    (base stride : Field → Nat) (field : Field)
    (source : Fin productionShape.runningCount) (index : Fin field.width)
    (oneColumn : Nat)
    (bound : base field + source.val * stride field + index.val < wire.slotCount) :
    (payloadRule wire base stride field).form? logicalWidth oneColumn
      ⟨source.val, 0, field.start + index.val⟩ =
      some (some (wireForm wire fits
        ⟨base field + source.val * stride field + index.val, bound⟩)) := by
  have loaded := retainedRule_form field.selected source ⟨0, by change 0 < 1; omega⟩ index
    wire fits oneColumn (base field) (stride field) 0 1 (1 : F) (by simpa using bound)
  simpa only [payloadRule, Field.selected, region, Nat.zero_add, Nat.mul_zero,
    Nat.add_zero, Nat.mul_one, applyCoefficient, if_pos rfl] using! loaded

private def fieldAt : Fin 4 → Field :=
  ![.commitment, .publicInput, .evalK, .evalA]

private theorem payload_program {logicalWidth : Nat}
    (wire : RetainedBlock) (fits : wire.start + wire.coordinateCount ≤ logicalWidth)
    (base stride : Field → Nat) (field : Field)
    (source : Fin productionShape.runningCount) (index : Fin field.width)
    (oneColumn : Nat)
    (bound : base field + source.val * stride field + index.val < wire.slotCount) :
    (AffineGrid.Program.mk ((List.ofFn fieldAt).map (payloadRule wire base stride))).form?
      logicalWidth oneColumn ⟨source.val, 0, field.start + index.val⟩ =
      some (wireForm wire fits
        ⟨base field + source.val * stride field + index.val, bound⟩) := by
  have active := payload_selected wire fits base stride field source index oneColumn bound
  have outside (other : Field) (different : other ≠ field) :
      (payloadRule wire base stride other).form? logicalWidth oneColumn
        ⟨source.val, 0, field.start + index.val⟩ = some none := by
    apply outside_rule
    have indexBound := index.isLt
    cases field <;> cases other <;> try contradiction
    all_goals
      change _ < _ ∨ _ ≤ _
      norm_num [Field.selected, region, Field.start, Field.width,
        PiCCSInputs.runningCommitmentWords, PiCCSInputs.runningPublicWords,
        PiDECInputs.evalKWordsPerChild, PiDECInputs.evalAWordsPerChild,
        PiDEC.v1_1.RingKRecomposition.coordinateCount,
        PiDEC.v1_1.EvalKRecomposition.blockCount,
        PiDEC.v1_1.EvalARecomposition.blockCount,
        PiDEC.v1_1.RingKRecomposition.cellCount, ringDegree, productionShape,
        Spec.Folding.PiCCS.PaperJoint.Phi81MatrixSource.phi81Shape] at indexBound ⊢ <;> omega
  let form := wireForm wire fits
    ⟨base field + source.val * stride field + index.val, bound⟩
  let results := fun i : Fin 4 => if fieldAt i = field then some form else none
  have loaded (i : Fin 4) :
      (payloadRule wire base stride (fieldAt i)).form? logicalWidth oneColumn
        ⟨source.val, 0, field.start + index.val⟩ = some (results i) := by
    by_cases equal : fieldAt i = field
    · simpa only [results, equal, ↓reduceIte] using active
    · simpa only [results, if_neg equal] using outside (fieldAt i) equal
  have all := four_forms (fun i => payloadRule wire base stride (fieldAt i))
    oneColumn ⟨source.val, 0, field.start + index.val⟩ results loaded
  cases field <;>
    simpa [fieldAt, results, form, List.ofFn_succ, combine, addSelected,
      SparseForm.add, SparseForm.empty] using all

private theorem payload_header {logicalWidth : Nat}
    (wire : RetainedBlock) (base stride : Field → Nat)
    (source minor : Nat) (header : minor = 0 ∨ minor = 1189 ∨ minor = 1460)
    (oneColumn : Nat) :
    (AffineGrid.Program.mk ((List.ofFn fieldAt).map (payloadRule wire base stride))).form?
      logicalWidth oneColumn ⟨source, 0, minor⟩ = some .empty := by
  have loaded (field : Field) :
      (payloadRule wire base stride field).form? logicalWidth oneColumn
        ⟨source, 0, minor⟩ = some none := by
    apply outside_rule
    rcases header with rfl | rfl | rfl <;> cases field <;> decide
  have all := four_forms (fun i => payloadRule wire base stride (fieldAt i))
    oneColumn ⟨source, 0, minor⟩ (fun _ => none) (fun i => loaded (fieldAt i))
  simpa [fieldAt, List.ofFn_succ, combine, addSelected] using all

private def dataForms {program : ApplicationProgram} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth) (oneColumn : Fin logicalWidth)
    (source : Fin productionShape.runningCount) (field : Field) (index : Fin field.width) :
    OrdinaryRow.Forms logicalWidth :=
  { selector := SparseForm.singleton oneColumn 1
    a := flagForm geometry
    b := inputForm geometry source field index
    c := outputForm geometry source field index }

private def headerForms {program : ApplicationProgram} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth) (oneColumn : Fin logicalWidth) :
    OrdinaryRow.Forms logicalWidth :=
  { selector := SparseForm.singleton oneColumn 1
    a := flagForm geometry
    b := .empty
    c := .empty }

private theorem groups_shape (program : ApplicationProgram) (oneColumn : Nat) :
    (groupsGrid program oneColumn).shape =
      ⟨productionShape.runningCount, 1, PiCCSInputs.runningGroupWords⟩ := by rfl

private theorem right_data {program : ApplicationProgram} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth) (oneColumn : Fin logicalWidth)
    (source : Fin productionShape.runningCount) (field : Field) (index : Fin field.width) :
    (groupsGrid program oneColumn.val).right.form? logicalWidth oneColumn.val
      ⟨source.val, 0, field.start + index.val⟩ =
      some (inputForm geometry source field index) := by
  have loaded := payload_program (piDecWire program)
    (RunningTransitionRetainedGeometry.piDecFits geometry)
    Field.piDecBase Field.width field source index oneColumn.val
    (piDecIndex source field index).isLt
  have programEq : (groupsGrid program oneColumn.val).right =
      AffineGrid.Program.mk ((List.ofFn fieldAt).map
        (payloadRule (piDecWire program) Field.piDecBase Field.width)) := by rfl
  rw [programEq]
  have formEq := wireForm_ofSemantic (RunningTransitionRetainedBlocks.piDecBlock program)
    (RunningTransitionRetainedGeometry.piDecStart program)
    (RunningTransitionRetainedGeometry.piDecFits geometry) (piDecIndex source field index)
  exact loaded.trans (congrArg some formEq)

private theorem output_data {program : ApplicationProgram} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth) (oneColumn : Fin logicalWidth)
    (source : Fin productionShape.runningCount) (field : Field) (index : Fin field.width) :
    (groupsGrid program oneColumn.val).output.form? logicalWidth oneColumn.val
      ⟨source.val, 0, field.start + index.val⟩ =
      some (outputForm geometry source field index) := by
  have bound : PiCCSInputs.runningGroupStart 0 + field.start +
      source.val * PiCCSInputs.runningGroupWords + index.val <
        (outputWire program).slotCount := by
    have inside := (outputIndex source field index).isLt
    change 96 + source.val * 3081 + field.start + index.val < 49393 at inside
    change 96 + field.start + source.val * 3081 + index.val < 49393
    omega
  have loaded := payload_program (outputWire program)
    (RunningTransitionRetainedGeometry.outputFits geometry)
    (fun field => PiCCSInputs.runningGroupStart 0 + field.start)
    (fun _ => PiCCSInputs.runningGroupWords) field source index oneColumn.val bound
  have same : (⟨PiCCSInputs.runningGroupStart 0 + field.start +
      source.val * PiCCSInputs.runningGroupWords + index.val, bound⟩ :
      Fin (outputWire program).slotCount) = outputIndex source field index := by
    apply Fin.ext
    change 96 + field.start + source.val * 3081 + index.val =
      96 + source.val * 3081 + field.start + index.val
    omega
  rw [same] at loaded
  simpa only [groupsGrid, fieldAt, List.ofFn_succ, List.map_cons, List.map_nil,
    payloadRule, Field.selected, Field.start, Field.width,
    outputForm, RunningTransitionDirectPlan.Location.form, wireForm, outputWire,
    RetainedBlock.ofSemantic, RetainedBlock.semantic] using! loaded

private theorem data_row {program : ApplicationProgram} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth) (oneColumn : Fin logicalWidth)
    (source : Fin productionShape.runningCount) (field : Field) (index : Fin field.width) :
    (groupsGrid program oneColumn.val).row? logicalWidth
        (source.val * PiCCSInputs.runningGroupWords + (position field index).val) =
      some (dataForms geometry oneColumn source field index) := by
  have left := flagProgram_form geometry (region productionShape.runningCount 1 0
    PiCCSInputs.runningGroupWords) source ⟨0, by change 0 < 1; omega⟩ (position field index) oneColumn.val
  have loaded := MultiplicationGrid.Block.row?_of_results
    (groupsGrid program oneColumn.val) oneColumn rfl source ⟨0, by change 0 < 1; omega⟩
    (position field index) _ _ _ (by simpa only [groupsGrid, region, Nat.zero_add] using left)
    (right_data geometry oneColumn source field index)
    (output_data geometry oneColumn source field index)
  simpa only [dataForms, Fin.encodeProd, Fin.coe_mkDivMod, groups_shape, Nat.mul_one, Nat.one_mul, Nat.zero_mul, Nat.mul_zero, Nat.zero_add, Nat.mul_comm]
    using loaded

private theorem header_row {program : ApplicationProgram} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth) (oneColumn : Fin logicalWidth)
    (source : Fin productionShape.runningCount) (minor : Fin PiCCSInputs.runningGroupWords)
    (header : minor.val = 0 ∨ minor.val = 1189 ∨ minor.val = 1460) :
    (groupsGrid program oneColumn.val).row? logicalWidth
      (source.val * PiCCSInputs.runningGroupWords + minor.val) =
        some (headerForms geometry oneColumn) := by
  have left := flagProgram_form geometry (region productionShape.runningCount 1 0
    PiCCSInputs.runningGroupWords) source ⟨0, by change 0 < 1; omega⟩ minor oneColumn.val
  have right := payload_header (logicalWidth := logicalWidth) (piDecWire program)
    Field.piDecBase Field.width source.val minor.val header oneColumn.val
  have output := payload_header (logicalWidth := logicalWidth) (outputWire program)
    (fun field => PiCCSInputs.runningGroupStart 0 + field.start)
    (fun _ => PiCCSInputs.runningGroupWords) source.val minor.val header oneColumn.val
  have loaded := MultiplicationGrid.Block.row?_of_results
    (groupsGrid program oneColumn.val) oneColumn rfl source ⟨0, by change 0 < 1; omega⟩ minor _ _ _
    (by simpa only [groupsGrid, region, Nat.zero_add] using left)
    (by simpa only [groupsGrid, fieldAt, List.ofFn_succ, List.map_cons, List.map_nil,
      payloadRule, Field.selected, Field.start, Field.width, Field.piDecBase] using! right)
    (by simpa only [groupsGrid, fieldAt, List.ofFn_succ, List.map_cons, List.map_nil,
      payloadRule, Field.selected, Field.start, Field.width] using! output)
  simpa only [headerForms, Fin.encodeProd, Fin.coe_mkDivMod, groups_shape, Nat.mul_one, Nat.one_mul, Nat.zero_mul, Nat.mul_zero, Nat.zero_add, Nat.mul_comm]
    using loaded

private theorem decode_val {left right : Nat} (index : Fin (left * right)) :
    (Fin.decodeProd index).1.val * right + (Fin.decodeProd index).2.val = index.val := by
  simpa only [Fin.encodeProd, Fin.coe_mkDivMod, Nat.mul_comm] using
    congrArg Fin.val (Fin.encodeProd_decodeProd index)

private theorem component_var (start : Nat) (part : Fin 2) :
    component ⟨Expr.var start, Expr.var (start + 1)⟩ part =
      Expr.var (start + part.val) := by
  fin_cases part <;> rfl

private theorem data_words {relationWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Spec.Folding.PiCCS.PaperJoint.Phi81CarrierLayout.carrierWidth relationWidth}
    (source : Fin productionShape.runningCount) (field : Field) (index : Fin field.width) :
    RunningTransition.runningWord
        (RunningTransitionInputs.recursiveRunningExpr relationWidth publicFits)
        (groupIndex source (position field index)) =
      Expr.var (RunningTransitionDirectPlan.Location.piDec (piDecIndex source field index)).sourceColumn ∧
    RunningTransition.runningWord
        (RunningTransitionInputs.outputRunningExpr relationWidth publicFits)
        (groupIndex source (position field index)) =
      Expr.var (RunningTransitionDirectPlan.Location.output (outputIndex source field index)).sourceColumn ∧
    RunningTransition.defaultWord (logicalWidth := relationWidth) (publicFits := publicFits)
        (groupIndex source (position field index)) = 0 := by
  cases field with
  | commitment =>
      let pair : Fin productionProfile.commitmentWidth × Fin ringDegree := Fin.decodeProd index
      have split : pair.1.val * 54 + pair.2.val = index.val := decode_val (left := productionProfile.commitmentWidth) (right := ringDegree) index
      have same : position .commitment index = commitmentPosition pair.1 pair.2 := by
        apply Fin.ext
        change 1 + index.val = 1 + pair.1.val * 54 + pair.2.val
        omega
      rw [same, runningWord_commitment, runningWord_commitment, defaultWord_commitment]
      refine ⟨?_, ?_, rfl⟩
      · change Expr.var ((PiDECInputs.proofInputStart + source.val * 1188) +
            pair.1.val * 54 + pair.2.val) =
          Expr.var (PiDECInputs.proofInputStart + (0 + source.val * 1188 + index.val))
        apply congrArg Expr.var
        omega
      · change Expr.var (PilotProduction.outputPreimageStart +
            (96 + source.val * 3081 + 1) + pair.1.val * 54 + pair.2.val) =
          Expr.var (PilotProduction.outputPreimageStart +
            (96 + source.val * 3081 + 1 + index.val))
        apply congrArg Expr.var
        omega
  | publicInput =>
      let column : Fin (FullShape relationWidth publicFits).publicWidth := index
      have same : position .publicInput index = publicPosition column := by rfl
      rw [same, runningWord_publicInput, runningWord_publicInput, defaultWord_publicInput]
      refine ⟨?_, ?_, rfl⟩
      · change Expr.var (PiDECInputs.publicInputStart + source.val * 270 + index.val) =
          Expr.var (PiDECInputs.proofInputStart +
            (PiDECInputs.publicInputStart - PiDECInputs.proofInputStart +
              source.val * 270 + index.val))
        have start : PiDECInputs.proofInputStart ≤ PiDECInputs.publicInputStart := by
          unfold PiDECInputs.publicInputStart PiDECInputs.evalAInputStart
            PiDECInputs.evalKInputStart PiDECInputs.commitmentInputStart
          omega
        apply congrArg Expr.var
        omega
      · change Expr.var (PilotProduction.outputPreimageStart +
            (96 + source.val * 3081 + 1190) + index.val) =
          Expr.var (PilotProduction.outputPreimageStart +
            (96 + source.val * 3081 + 1190 + index.val))
        apply congrArg Expr.var
        omega
  | evalK =>
      let pair : Fin productionShape.coefficientCount × Fin 2 := Fin.decodeProd index
      have split : pair.1.val * 2 + pair.2.val = index.val := decode_val (left := productionShape.coefficientCount) (right := 2) index
      have same : position .evalK index = evalKPosition pair.1 pair.2 := by
        apply Fin.ext
        change 1461 + index.val = 1461 + pair.1.val * 2 + pair.2.val
        omega
      rw [same, runningWord_evalK, runningWord_evalK, defaultWord_evalK]
      refine ⟨?_, ?_, rfl⟩
      · change component ⟨Expr.var (PiDECInputs.childEvalKStart
            (RunningTransitionInputs.childOfRunning source) + pair.1.val * 2),
          Expr.var (PiDECInputs.childEvalKStart
            (RunningTransitionInputs.childOfRunning source) + pair.1.val * 2 + 1)⟩ pair.2 = _
        rw [component_var]
        change Expr.var (PiDECInputs.evalKInputStart + source.val * 108 +
            pair.1.val * 2 + pair.2.val) =
          Expr.var (PiDECInputs.proofInputStart +
            (PiDECInputs.evalKInputStart - PiDECInputs.proofInputStart +
              source.val * 108 + index.val))
        have start : PiDECInputs.proofInputStart ≤ PiDECInputs.evalKInputStart := by
          unfold PiDECInputs.evalKInputStart PiDECInputs.commitmentInputStart
          omega
        apply congrArg Expr.var
        omega
      · change component ⟨Expr.var (RunningTransitionInputs.outputBase +
            (PiCCSInputs.runningEvaluationStart source.val + pair.1.val * 2)),
          Expr.var (RunningTransitionInputs.outputBase +
            (PiCCSInputs.runningEvaluationStart source.val + pair.1.val * 2) + 1)⟩ pair.2 = _
        rw [component_var]
        change Expr.var (PilotProduction.outputPreimageStart +
            (96 + source.val * 3081 + 1461 + pair.1.val * 2) + pair.2.val) =
          Expr.var (PilotProduction.outputPreimageStart +
            (96 + source.val * 3081 + 1461 + index.val))
        apply congrArg Expr.var
        omega
  | evalA =>
      let outer : Fin productionShape.matrixCount × Fin (productionShape.coefficientCount * 2) :=
        Fin.decodeProd index
      let pair : Fin productionShape.coefficientCount × Fin 2 := Fin.decodeProd outer.2
      have outerSplit : outer.1.val * 108 + outer.2.val = index.val := decode_val (left := productionShape.matrixCount) (right := productionShape.coefficientCount * 2) index
      have innerSplit : pair.1.val * 2 + pair.2.val = outer.2.val := decode_val (left := productionShape.coefficientCount) (right := 2) outer.2
      have same : position .evalA index = evalAPosition outer.1 pair.1 pair.2 := by
        apply Fin.ext
        change 1569 + index.val = 1461 + 108 + outer.1.val * 108 + pair.1.val * 2 + pair.2.val
        omega
      rw [same, runningWord_evalA, runningWord_evalA, defaultWord_evalA]
      refine ⟨?_, ?_, rfl⟩
      · change component ⟨Expr.var (PiDECInputs.childEvalAStart
            (RunningTransitionInputs.childOfRunning source) +
              outer.1.val * 108 + pair.1.val * 2),
          Expr.var (PiDECInputs.childEvalAStart
            (RunningTransitionInputs.childOfRunning source) +
              outer.1.val * 108 + pair.1.val * 2 + 1)⟩ pair.2 = _
        rw [component_var]
        change Expr.var (PiDECInputs.evalAInputStart + source.val * 1512 +
            outer.1.val * 108 + pair.1.val * 2 + pair.2.val) =
          Expr.var (PiDECInputs.proofInputStart +
            (PiDECInputs.evalAInputStart - PiDECInputs.proofInputStart +
              source.val * 1512 + index.val))
        have start : PiDECInputs.proofInputStart ≤ PiDECInputs.evalAInputStart := by
          unfold PiDECInputs.evalAInputStart PiDECInputs.evalKInputStart
            PiDECInputs.commitmentInputStart
          omega
        apply congrArg Expr.var
        omega
      · change component ⟨Expr.var (RunningTransitionInputs.outputBase +
            (PiCCSInputs.runningEvaluationStart source.val + 108 +
              outer.1.val * 108 + pair.1.val * 2)),
          Expr.var (RunningTransitionInputs.outputBase +
            (PiCCSInputs.runningEvaluationStart source.val + 108 +
              outer.1.val * 108 + pair.1.val * 2) + 1)⟩ pair.2 = _
        rw [component_var]
        change Expr.var (PilotProduction.outputPreimageStart +
            (96 + source.val * 3081 + 1461 + 108 + outer.1.val * 108 + pair.1.val * 2) +
              pair.2.val) =
          Expr.var (PilotProduction.outputPreimageStart +
            (96 + source.val * 3081 + 1569 + index.val))
        apply congrArg Expr.var
        omega

private theorem data_preserves {program : ApplicationProgram} {logicalWidth relationWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Spec.Folding.PiCCS.PaperJoint.Phi81CarrierLayout.carrierWidth relationWidth}
    (relation : ProductionKey.LogicalRelation relationWidth publicFits)
    (geometry : Geometry program logicalWidth) (oneColumn : Fin logicalWidth)
    (assignment : Assignment F logicalWidth) (one : assignment oneColumn = 1)
    (source : Fin productionShape.runningCount) (field : Field) (index : Fin field.width) :
    (dataForms geometry oneColumn source field index).Preserves assignment
      (decodedEnv geometry assignment)
      (RunningTransitionReducedRows.muxRow relationWidth publicFits
        (groupIndex source (position field index))) := by
  rcases RunningTransitionReducedRows.muxRow_values relation (decodedEnv geometry assignment)
    (groupIndex source (position field index)) with ⟨flag, recursive, output⟩
  rcases data_words (relationWidth := relationWidth) (publicFits := publicFits)
    source field index with ⟨recursiveWord, outputWord, defaultWord⟩
  rw [recursiveWord, defaultWord, Expr.eval_var, sub_zero] at recursive
  rw [outputWord, defaultWord, Expr.eval_var, sub_zero] at output
  refine ⟨by simp [dataForms, one], ?_, ?_, ?_⟩
  · exact (congrArg (fun form : SparseForm logicalWidth => form.eval assignment)
      (sourceForm_flag geometry)).symm.trans flag.symm
  · exact (congrArg (fun form : SparseForm logicalWidth => form.eval assignment)
      (sourceForm_piDec geometry (piDecIndex source field index))).symm.trans recursive.symm
  · exact (congrArg (fun form : SparseForm logicalWidth => form.eval assignment)
      (sourceForm_output geometry (outputIndex source field index))).symm.trans output.symm

private theorem header_word {relationWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Spec.Folding.PiCCS.PaperJoint.Phi81CarrierLayout.carrierWidth relationWidth}
    (running : StatementAbsorption.RunningExpr relationWidth publicFits)
    (source : Fin productionShape.runningCount) (minor : Fin PiCCSInputs.runningGroupWords)
    (header : minor.val = 0 ∨ minor.val = 1189 ∨ minor.val = 1460) :
    RunningTransition.runningWord running (groupIndex source minor) =
      Expr.const (RunningTransition.defaultWord
        (logicalWidth := relationWidth) (publicFits := publicFits) (groupIndex source minor)) := by
  rcases header with first | second | third
  · have same : minor = ⟨0, by decide⟩ := Fin.ext first
    rw [same, runningWord_commitmentHeader, defaultWord_commitmentHeader]
  · have same : minor = ⟨1189, by decide⟩ := Fin.ext second
    rw [same, runningWord_publicHeader, defaultWord_publicHeader]
  · have same : minor = ⟨1460, by decide⟩ := Fin.ext third
    rw [same, runningWord_evaluationHeader, defaultWord_evaluationHeader]

private theorem header_preserves {program : ApplicationProgram} {logicalWidth relationWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Spec.Folding.PiCCS.PaperJoint.Phi81CarrierLayout.carrierWidth relationWidth}
    (relation : ProductionKey.LogicalRelation relationWidth publicFits)
    (geometry : Geometry program logicalWidth) (oneColumn : Fin logicalWidth)
    (assignment : Assignment F logicalWidth) (one : assignment oneColumn = 1)
    (source : Fin productionShape.runningCount) (minor : Fin PiCCSInputs.runningGroupWords)
    (header : minor.val = 0 ∨ minor.val = 1189 ∨ minor.val = 1460) :
    (headerForms geometry oneColumn).Preserves assignment (decodedEnv geometry assignment)
      (RunningTransitionReducedRows.muxRow relationWidth publicFits (groupIndex source minor)) := by
  rcases RunningTransitionReducedRows.muxRow_values relation (decodedEnv geometry assignment)
    (groupIndex source minor) with ⟨flag, recursive, output⟩
  rw [header_word _ source minor header, Expr.eval_const, sub_self] at recursive output
  refine ⟨by simp [headerForms, one], ?_, ?_, ?_⟩
  · exact (congrArg (fun form : SparseForm logicalWidth => form.eval assignment)
      (sourceForm_flag geometry)).symm.trans flag.symm
  · simpa only [headerForms, SparseForm.empty_eval] using recursive.symm
  · simpa only [headerForms, SparseForm.empty_eval] using output.symm

/-- Every row of every group has an exact decoded form and preserves the
original mux row. The cases cover all 3081 positions without expanding them. -/
theorem row_preserves {program : ApplicationProgram} {logicalWidth relationWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Spec.Folding.PiCCS.PaperJoint.Phi81CarrierLayout.carrierWidth relationWidth}
    (relation : ProductionKey.LogicalRelation relationWidth publicFits)
    (geometry : Geometry program logicalWidth) (oneColumn : Fin logicalWidth)
    (assignment : Assignment F logicalWidth) (one : assignment oneColumn = 1)
    (source : Fin productionShape.runningCount) (minor : Fin PiCCSInputs.runningGroupWords) :
    ∃ forms,
      (groupsGrid program oneColumn.val).row? logicalWidth
        (source.val * PiCCSInputs.runningGroupWords + minor.val) = some forms ∧
      forms.Preserves assignment (decodedEnv geometry assignment)
        (RunningTransitionReducedRows.muxRow relationWidth publicFits
          (groupIndex source minor)) := by
  have data (field : Field) (index : Fin field.width) (same : position field index = minor) :
      ∃ forms,
        (groupsGrid program oneColumn.val).row? logicalWidth
          (source.val * PiCCSInputs.runningGroupWords + minor.val) = some forms ∧
        forms.Preserves assignment (decodedEnv geometry assignment)
          (RunningTransitionReducedRows.muxRow relationWidth publicFits
            (groupIndex source minor)) := by
    refine ⟨dataForms geometry oneColumn source field index, ?_, ?_⟩
    · simpa only [same] using data_row geometry oneColumn source field index
    · simpa only [same] using data_preserves relation geometry oneColumn assignment one
        source field index
  have header (found : minor.val = 0 ∨ minor.val = 1189 ∨ minor.val = 1460) :
      ∃ forms,
        (groupsGrid program oneColumn.val).row? logicalWidth
          (source.val * PiCCSInputs.runningGroupWords + minor.val) = some forms ∧
        forms.Preserves assignment (decodedEnv geometry assignment)
          (RunningTransitionReducedRows.muxRow relationWidth publicFits
            (groupIndex source minor)) :=
    ⟨headerForms geometry oneColumn, header_row geometry oneColumn source minor found,
      header_preserves relation geometry oneColumn assignment one source minor found⟩
  have bound : minor.val < 3081 := minor.isLt
  by_cases zero : minor.val = 0
  · exact header (Or.inl zero)
  by_cases commitment : minor.val < 1189
  · apply data .commitment ⟨minor.val - 1, by change minor.val - 1 < 1188; omega⟩
    apply Fin.ext
    change 1 + (minor.val - 1) = minor.val
    omega
  by_cases publicHeader : minor.val = 1189
  · exact header (Or.inr (Or.inl publicHeader))
  by_cases publicData : minor.val < 1460
  · apply data .publicInput ⟨minor.val - 1190, by change minor.val - 1190 < 270; omega⟩
    apply Fin.ext
    change 1190 + (minor.val - 1190) = minor.val
    omega
  by_cases evaluationHeader : minor.val = 1460
  · exact header (Or.inr (Or.inr evaluationHeader))
  by_cases evalK : minor.val < 1569
  · apply data .evalK ⟨minor.val - 1461, by change minor.val - 1461 < 108; omega⟩
    apply Fin.ext
    change 1461 + (minor.val - 1461) = minor.val
    omega
  · apply data .evalA ⟨minor.val - 1569, by change minor.val - 1569 < 1512; omega⟩
    apply Fin.ext
    change 1569 + (minor.val - 1569) = minor.val
    omega

end NightstreamFPrime.Export.Stage1.RunningTransitionReducedGroups
