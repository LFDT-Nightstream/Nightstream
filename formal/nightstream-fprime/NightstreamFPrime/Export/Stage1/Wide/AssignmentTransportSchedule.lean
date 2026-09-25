import NightstreamFPrime.Export.Stage1.Wide.AssignmentTransportTail
import NightstreamFPrime.Export.Stage1.Wide.PiRLCTransportValues
import NightstreamFPrime.Export.Stage1.Wide.CommonSchedule

/-! The exact emitted suffix has the same coordinate order and values as
the direct sampler and ring witness. Physical range values come from the
constructive completion program. -/

namespace NightstreamFPrime.Export.Stage1.Wide.AssignmentTransportSchedule

open NightstreamFPrime.Circuit NightstreamFPrime.Spec NightstreamFPrime.Layout
open NightstreamFPrime.Lifecycle NightstreamFPrime.Lifecycle.PaperAlgebra
open ProductionRelation CanonicalBlockAssignment PiRlcWideSampler
open Spec.Folding.PiCCS.PaperJoint
open AssignmentTransport AssignmentTransportExecution AssignmentTransportSemantics

theorem cons_correct {left right : BlockValue} {leftRest rightRest : Schedule}
    (head : left.coordinateCount = right.coordinateCount ∧ ∀ index, left.coordinateAt index = right.coordinateAt index)
    (rest : coordinateCount leftRest = coordinateCount rightRest ∧ ∀ index, coordinateAt leftRest index = coordinateAt rightRest index) :
    coordinateCount (left :: leftRest) = coordinateCount (right :: rightRest) ∧
      ∀ index, coordinateAt (left :: leftRest) index = coordinateAt (right :: rightRest) index := by
  constructor
  · simp only [CanonicalBlockAssignment.coordinateCount, head.1, rest.1]
  · intro index
    simp only [coordinateAt, head.1]
    split
    · exact head.2 index
    · exact rest.2 _

theorem append_correct {left right leftRest rightRest : Schedule}
    (head : coordinateCount left = coordinateCount right ∧ ∀ index, coordinateAt left index = coordinateAt right index)
    (rest : coordinateCount leftRest = coordinateCount rightRest ∧ ∀ index, coordinateAt leftRest index = coordinateAt rightRest index) :
    coordinateCount (left ++ leftRest) = coordinateCount (right ++ rightRest) ∧
      ∀ index, coordinateAt (left ++ leftRest) index = coordinateAt (right ++ rightRest) index := by
  constructor
  · simp only [coordinateCount_append, head.1, rest.1]
  · intro index
    by_cases inside : index < coordinateCount left
    · rw [AssignmentTransportCommon.lookup_before _ _ _ inside,
        AssignmentTransportCommon.lookup_before _ _ _ (by rwa [← head.1])]
      exact head.2 index
    · have position : index = coordinateCount left + (index - coordinateCount left) := by omega
      rw [position, coordinateAt_append_offset, head.1, coordinateAt_append_offset]
      exact rest.2 _

private theorem flatten_correct {count : Nat} (left right : Fin count → Schedule)
    (same : ∀ source, coordinateCount (left source) = coordinateCount (right source) ∧
      ∀ index, coordinateAt (left source) index = coordinateAt (right source) index) :
    coordinateCount (List.ofFn left).flatten = coordinateCount (List.ofFn right).flatten ∧
      ∀ index, coordinateAt (List.ofFn left).flatten index = coordinateAt (List.ofFn right).flatten index := by
  induction count with
  | zero => exact ⟨rfl, fun _ => rfl⟩
  | succ count ih =>
    rw [List.ofFn_succ, List.ofFn_succ, List.flatten_cons, List.flatten_cons]
    exact append_correct (same 0) (ih (fun source => left source.succ) (fun source => right source.succ)
      (fun source => same source.succ))

theorem block_correct (plan : AssignmentTransport.Plan) (width : Nat) (env : Env)
    (block : Values) (reference : BlockValue)
    (kinds : block.kind = reference.block.kind) (counts : block.count = reference.block.slotCount)
    (values : ∀ (slot : Nat) (left : slot < block.count) (right : slot < reference.block.slotCount),
      sourceValue plan width env (AffineRuns.sourceAt block.sources slot) =
        reference.source (reference.block.source ⟨slot, right⟩)) :
    (blockValue plan width env block).coordinateCount = reference.coordinateCount ∧
      ∀ index, (blockValue plan width env block).coordinateAt index = reference.coordinateAt index := by
  constructor
  · change block.count * block.kind.width = reference.block.slotCount * reference.block.kind.width
    rw [kinds, counts]
  · exact PerApplicationAssignmentTransportExecution.blockValue_coordinateAt_eq _ _ kinds counts values

section Physical

variable {width : Nat}
  {fits : ringDegree * publicRingColumns ≤ Phi81CarrierLayout.carrierWidth width}

private theorem initial_value (program : RetainedLayout.Program) (env : Env)
    (application : Fin (PerApplicationPackage.addedPrivateColumnCount program) → F)
    (relation : ProductionKey.LogicalRelation width fits)
    (ajtai : AjtaiKey (logicalWidth := width) (publicFits := fits)) (template : Proof 9)
    (assumptions : PiCCS.v1_1.Formal.Assumptions relation
      (Layout.Stage1.PiCCSInputs.interface width fits) Layout.Stage1.PiCCSInputs.phaseOffset env)
    (rows : Layout.PiCCS.v1_1.PhysicalHolds relation
      (Layout.Stage1.PiCCSInputs.interface width fits) Layout.Stage1.PiCCSInputs.phaseOffset env) :
    Gadgets.Poseidon2.Layer.evalState env
        ((SamplerSourceValues.stageInterface (width := width) (fits := fits)).initialState SamplerSourceValues.stageStart) =
      PiRLCWitness.initial (Stage1Plan.piRlcInterface program)
        (AssignmentProjection.seed program (SourceAssignment.raw program env application).assignment) := by
  funext lane
  exact (PiRLCSourceInputs.initial_value program env application relation ajtai template assumptions rows lane).symm

private theorem sampler_source_lt (program : RetainedLayout.Program) (index : Fin (34 * 86)) :
    AffineRuns.sourceAt samplerPoseidon.sources index.val < CommonSchedule.physicalWidth program := by
  rw [samplerPoseidon, Values.ofSource_source]
  let invocation := index.val / 86
  let start := if invocation % 2 = 0 then Layout.Stage1.Wide.PiRLCStarts.entryLogicalStart (invocation / 2)
    else Layout.Stage1.Wide.PiRLCStarts.advanceLogicalStart (invocation / 2)
  have invocationBound : invocation / 2 < 17 := by dsimp only [invocation]; omega
  have lower : 14751804 ≤ start := by
    dsimp only [start]
    split
    · change 14751804 ≤ 19513117 + invocation / 2 * 3205; omega
    · change 14751804 ≤ 19513117 + invocation / 2 * 3205 + 592 + 2021; omega
  have upper : start + 592 < Layout.Stage1.Wide.SourceOrder.sourceWidth := by
    dsimp only [start]
    split
    · change 19513117 + invocation / 2 * 3205 + 592 < 27859538; omega
    · change 19513117 + invocation / 2 * 3205 + 592 + 2021 + 592 < 27859538; omega
  change Layout.Stage1.Wide.SourceOrder.column start + (PoseidonRetainedSlots.localOutput _).val < _
  rw [Layout.Stage1.Wide.SourceOrder.column_late start lower (by omega), CommonSchedule.physicalWidth,
    Layout.Stage1.Wide.SourceOrder.totalColumns_eq]
  have localBound : (PoseidonRetainedSlots.localOutput
      ⟨index.val % 86, by rw [PoseidonRetainedSlots.rows_length]; omega⟩).val < 592 :=
    (PoseidonRetainedSlots.localOutput _).isLt
  rw [Layout.Stage1.Wide.SourceOrder.sourceWidth_eq] at upper
  omega

private theorem range_source_lt (program : RetainedLayout.Program) (source : Fin 17)
    (block : LowNormBlock.Block 2025) (index : Fin block.slotCount)
    (retained : 1408 ≤ (block.source index).val) :
    AffineRuns.sourceAt (rangeBlock source.val block).sources index.val < CommonSchedule.physicalWidth program := by
  rw [rangeBlock_physical_source source block index (by omega)]
  have sourceBound : source.val < 17 := source.isLt
  have cellBound : (block.source index).val < 2025 := (block.source index).isLt
  have lower : 14751804 ≤ Layout.Stage1.Wide.PiRLCStarts.rangeLogicalStart source.val + ((block.source index).val - 4) := by
    change 14751804 ≤ 19513117 + source.val * 3205 + 592 + ((block.source index).val - 4); omega
  have upper : Layout.Stage1.Wide.PiRLCStarts.rangeLogicalStart source.val + ((block.source index).val - 4) <
      Layout.Stage1.Wide.SourceOrder.sourceWidth := by
    change 19513117 + source.val * 3205 + 592 + ((block.source index).val - 4) < 27859538; omega
  rw [Layout.Stage1.Wide.SourceOrder.column_late _ lower upper, CommonSchedule.physicalWidth,
    Layout.Stage1.Wide.SourceOrder.totalColumns_eq]
  rw [Layout.Stage1.Wide.SourceOrder.sourceWidth_eq] at upper
  omega

theorem sampler_correct (program : RetainedLayout.Program) (env : Env)
    (application : Fin (PerApplicationPackage.addedPrivateColumnCount program) → F)
    (plan : AssignmentTransport.Plan) (relation : ProductionKey.LogicalRelation width fits)
    (ajtai : AjtaiKey (logicalWidth := width) (publicFits := fits)) (template : Proof 9)
    (cAssumptions : PiCCS.v1_1.Formal.Assumptions relation
      (Layout.Stage1.PiCCSInputs.interface width fits) Layout.Stage1.PiCCSInputs.phaseOffset env)
    (cRows : Layout.PiCCS.v1_1.PhysicalHolds relation
      (Layout.Stage1.PiCCSInputs.interface width fits) Layout.Stage1.PiCCSInputs.phaseOffset env)
    (rRows : Layout.PiRLC.Wide.PhysicalHolds relation
      (Layout.Stage1.Wide.PiRLCInputs.interface (logicalWidth := width) (publicFits := fits)) SamplerSourceValues.stageStart env)
    (completed : PiRLC.Wide.Formal.RangesCompleted
      (Layout.Stage1.Wide.PiRLCInputs.interface (logicalWidth := width) (publicFits := fits)) SamplerSourceValues.stageStart env) :
    let actual := ([samplerPoseidon] ++ samplerRanges).map
      (blockValue plan (CommonSchedule.physicalWidth program) (physicalValues program env application))
    let initial := PiRLCWitness.initial (Stage1Plan.piRlcInterface program)
      (AssignmentProjection.seed program (SourceAssignment.raw program env application).assignment)
    coordinateCount actual = coordinateCount (AssignmentTransportTail.sampler initial) ∧
      ∀ index, coordinateAt actual index = coordinateAt (AssignmentTransportTail.sampler initial) index := by
  let initial := PiRLCWitness.initial (Stage1Plan.piRlcInterface program)
    (AssignmentProjection.seed program (SourceAssignment.raw program env application).assignment)
  have initialEq := initial_value program env application relation ajtai template cAssumptions cRows
  have poseidon := block_correct plan (CommonSchedule.physicalWidth program) (physicalValues program env application)
    samplerPoseidon (ofBlock BatchPlan.poseidonBlock (Witness.sboxValues initial)) rfl rfl (by
      intro slot left right
      rw [sourceValue_physical _ _ _ _ (sampler_source_lt program ⟨slot, left⟩),
        SamplerSourceValues.physical_sbox program env application relation rRows ⟨slot, left⟩, initialEq]
      simp only [ofBlock, BatchPlan.poseidonBlock, id_eq, initial])
  have rangeEq (source : Fin 17) (block : LowNormBlock.Block 2025)
      (retained : ∀ slot, 1408 ≤ (block.source slot).val) :
      (blockValue plan (CommonSchedule.physicalWidth program) (physicalValues program env application)
        (rangeBlock source.val block)).coordinateCount =
          (ofBlock block (fun column => Witness.rangeSources initial source column.val)).coordinateCount ∧
      ∀ index, (blockValue plan (CommonSchedule.physicalWidth program) (physicalValues program env application)
        (rangeBlock source.val block)).coordinateAt index =
          (ofBlock block (fun column => Witness.rangeSources initial source column.val)).coordinateAt index := by
    apply block_correct _ _ _ _ _ rfl rfl
    intro slot left right
    rw [sourceValue_physical _ _ _ _ (range_source_lt program source block ⟨slot, left⟩ (retained _)),
      SamplerSourceValues.physical_range program env application relation rRows completed source block
        ⟨slot, left⟩ (retained _), initialEq]
    rfl
  have ranges (source : Fin 17) := cons_correct (rangeEq source Retained.canonicalBits (by
    intro slot; change 1408 ≤ 1408 + 66 * (slot.val / 64) + slot.val % 64; omega))
    (cons_correct (rangeEq source Retained.canonicalFields (by
      intro slot; change 1408 ≤ 1408 + 66 * (slot.val / 2) + 64 + slot.val % 2; omega))
      (cons_correct (rangeEq source Retained.resultBits (by intro slot; change 1408 ≤ 1672 + slot.val; omega))
        (show coordinateCount [] = coordinateCount [] ∧ ∀ index, coordinateAt [] index = coordinateAt [] index from
          ⟨rfl, fun _ => rfl⟩)))
  have lists : samplerRanges.map (blockValue plan (CommonSchedule.physicalWidth program) (physicalValues program env application)) =
      (List.ofFn fun source : Fin 17 =>
        [blockValue plan (CommonSchedule.physicalWidth program) (physicalValues program env application) (rangeBlock source.val Retained.canonicalBits),
         blockValue plan (CommonSchedule.physicalWidth program) (physicalValues program env application) (rangeBlock source.val Retained.canonicalFields),
         blockValue plan (CommonSchedule.physicalWidth program) (physicalValues program env application) (rangeBlock source.val Retained.resultBits)]).flatten := by
    rw [samplerRanges, List.map_flatMap]
    simp only [List.map_cons, List.map_nil]
    rw [List.ofFn_eq_map, ← List.map_coe_finRange_eq_range (n := 17), List.flatMap_map,
      List.flatMap_def]
  dsimp only
  simp only [List.map_append, List.map_cons, List.map_nil, List.singleton_append, AssignmentTransportTail.sampler, lists]
  exact cons_correct poseidon (flatten_correct _ _ ranges)

end Physical

private def fieldHalves (count : Nat) (values : Fin (2 * count) → F) : Schedule :=
  [ofBlock (FieldAssignment.block count) (fun slot => values ⟨slot.val, by omega⟩),
   ofBlock (FieldAssignment.block count) (fun slot => values ⟨count + slot.val, by omega⟩)]

private theorem fieldHalves_correct (count : Nat) (values : Fin (2 * count) → F) :
    coordinateCount (fieldHalves count values) = coordinateCount [ofBlock (FieldAssignment.block (2 * count)) values] ∧
      ∀ index, coordinateAt (fieldHalves count values) index =
        coordinateAt [ofBlock (FieldAssignment.block (2 * count)) values] index := by
  constructor
  · change count * 41 + (count * 41 + 0) = 2 * count * 41 + 0
    omega
  · intro index
    simp only [fieldHalves, coordinateAt, BlockValue.coordinateAt, ofBlock, FieldAssignment.block,
      BlockValue.coordinateCount, LowNormBlock.Block.coordinateCount, LowNormSlot.Kind.width, BalancedTernary.width, id_eq]
    by_cases first : index < count * 41
    · have all : index < 2 * count * 41 := by omega
      simp only [if_pos first, dif_pos first, if_pos all, dif_pos all]
    · by_cases second : index - count * 41 < count * 41
      · have all : index < 2 * count * 41 := by omega
        simp only [if_neg first, dif_neg first, if_pos second, dif_pos second, if_pos all, dif_pos all]
        apply congrArg₂ (LowNormSlot.coordinate .field)
        · apply congrArg values
          apply Fin.ext
          dsimp only
          omega
        · apply Fin.ext
          dsimp only
          omega
      · have all : ¬index < 2 * count * 41 := by omega
        simp only [if_neg first, dif_neg first, if_neg second, dif_neg second, if_neg all, dif_neg all]

variable {width : Nat}
  {fits : ringDegree * publicRingColumns ≤ Phi81CarrierLayout.carrierWidth width}

theorem fields_correct (program : RetainedLayout.Program) (env : Env)
    (application : Fin (PerApplicationPackage.addedPrivateColumnCount program) → F)
    (plan : AssignmentTransport.Plan)
    (emitted : AssignmentTransport.plan program (CommonSchedule.physicalWidth program) = .ok plan)
    (outputs : Values) (outputEmitted : commonBlock program .productOutput = .ok outputs)
    (relation : ProductionKey.LogicalRelation width fits)
    (ajtai : AjtaiKey (logicalWidth := width) (publicFits := fits)) (template : Proof 9)
    (cAssumptions : PiCCS.v1_1.Formal.Assumptions relation
      (Layout.Stage1.PiCCSInputs.interface width fits) Layout.Stage1.PiCCSInputs.phaseOffset env)
    (cRows : Layout.PiCCS.v1_1.PhysicalHolds relation
      (Layout.Stage1.PiCCSInputs.interface width fits) Layout.Stage1.PiCCSInputs.phaseOffset env)
    (rAssumptions : PiRLC.Wide.Formal.Assumptions relation
      (Layout.Stage1.Wide.PiRLCInputs.interface (logicalWidth := width) (publicFits := fits)) SamplerSourceValues.stageStart env)
    (rRows : Layout.PiRLC.Wide.PhysicalHolds relation
      (Layout.Stage1.Wide.PiRLCInputs.interface (logicalWidth := width) (publicFits := fits)) SamplerSourceValues.stageStart env) :
    let actual := [outputs, ⟨.field, 52326, [⟨CommonSchedule.physicalWidth program, 1, 52326⟩]⟩].map
      (blockValue plan (CommonSchedule.physicalWidth program) (physicalValues program env application))
    let initial := PiRLCWitness.initial (Stage1Plan.piRlcInterface program)
      (AssignmentProjection.seed program (SourceAssignment.raw program env application).assignment)
    let values := PiRLCWitness.inputValues (Stage1Plan.piRlcInterface program)
      (AssignmentProjection.seed program (SourceAssignment.raw program env application).assignment)
    coordinateCount actual = coordinateCount [ofBlock PiRLCGeometry.fieldBlock (PiRLCValues.fieldValue initial values)] ∧
      ∀ index, coordinateAt actual index = coordinateAt [ofBlock PiRLCGeometry.fieldBlock (PiRLCValues.fieldValue initial values)] index := by
  let initial := PiRLCWitness.initial (Stage1Plan.piRlcInterface program)
    (AssignmentProjection.seed program (SourceAssignment.raw program env application).assignment)
  let values := PiRLCWitness.inputValues (Stage1Plan.piRlcInterface program)
    (AssignmentProjection.seed program (SourceAssignment.raw program env application).assignment)
  let fields := PiRLCValues.fieldValue initial values
  have geometry := commonBlock_geometry program .productOutput outputs outputEmitted
  have outputKind : outputs.kind = .field := geometry.1
  have outputCount : outputs.count = 52326 := by
    rw [geometry.2, commonLimit, if_neg (by decide)]
    change (PiRLCRetainedGeometry.productOutputBlock program).slotCount = 52326
    rw [PiRLCRetainedGeometry.productOutputBlock, LowNormBlock.Block.lift_slotCount,
      PiRLCProductSourceBlocks.outputBlock_slotCount]
  have outputEq := block_correct plan (CommonSchedule.physicalWidth program) (physicalValues program env application)
    outputs (ofBlock (FieldAssignment.block 52326) (fun slot => fields ⟨slot.val, by change slot.val < 104652; omega⟩))
    outputKind outputCount (by
      intro slot left right
      have bounded := commonBlock_source_lt program .productOutput outputs outputEmitted ⟨slot, right⟩
      have mapped := commonBlock_source program .productOutput outputs outputEmitted ⟨slot, right⟩
      rw [sourceValue_physical _ _ _ _ (CommonSchedule.finalColumn_lt program _ _ bounded mapped),
        PiRLCTransportValues.output_value program env application outputs outputEmitted relation ajtai template
          cAssumptions cRows rAssumptions rRows ⟨slot, right⟩]
      change _ = PiRLCValues.fieldValue initial values ⟨slot, _⟩
      have inside : slot < PiRLCGeometry.outputCount := by
        rw [PiRLCGeometry.outputCount_eq]
        exact right
      rw [PiRLCValues.fieldValue, dif_pos inside])
  have quotientEq := block_correct plan (CommonSchedule.physicalWidth program) (physicalValues program env application)
    ⟨.field, 52326, [⟨CommonSchedule.physicalWidth program, 1, 52326⟩]⟩
    (ofBlock (FieldAssignment.block 52326) (fun slot => fields ⟨52326 + slot.val, by change 52326 + slot.val < 104652; omega⟩))
    rfl rfl (by
      intro slot left right
      rw [AffineRuns.sourceAt, if_pos left]
      simp only [Nat.one_mul]
      rw [sourceValue_quotient plan (CommonSchedule.physicalWidth program) (physicalValues program env application) ⟨slot, left⟩,
        PiRLCTransportValues.quotient_value program env application (CommonSchedule.physicalWidth program) plan emitted
          relation ajtai template cAssumptions cRows rRows ⟨slot, left⟩]
      change _ = PiRLCValues.fieldValue initial values ⟨52326 + slot, _⟩
      rw [PiRLCValues.fieldValue, dif_neg (by change ¬52326 + slot < 52326; omega)]
      simp only [PiRLCGeometry.outputCount_eq, Nat.add_sub_cancel_left, initial, values])
  have blocks := cons_correct outputEq (cons_correct quotientEq
    (show coordinateCount [] = coordinateCount [] ∧ ∀ index, coordinateAt [] index = coordinateAt [] index from
      ⟨rfl, fun _ => rfl⟩))
  have halves := fieldHalves_correct 52326 fields
  exact ⟨blocks.1.trans halves.1, fun index => (blocks.2 index).trans (halves.2 index)⟩

theorem schedule_correct (program : RetainedLayout.Program) (env : Env)
    (application : Fin (PerApplicationPackage.addedPrivateColumnCount program) → F)
    (plan : AssignmentTransport.Plan)
    (emitted : AssignmentTransport.plan program (CommonSchedule.physicalWidth program) = .ok plan)
    (relation : ProductionKey.LogicalRelation width fits)
    (ajtai : AjtaiKey (logicalWidth := width) (publicFits := fits)) (template : Proof 9)
    (cAssumptions : PiCCS.v1_1.Formal.Assumptions relation
      (Layout.Stage1.PiCCSInputs.interface width fits) Layout.Stage1.PiCCSInputs.phaseOffset env)
    (cRows : Layout.PiCCS.v1_1.PhysicalHolds relation
      (Layout.Stage1.PiCCSInputs.interface width fits) Layout.Stage1.PiCCSInputs.phaseOffset env)
    (rAssumptions : PiRLC.Wide.Formal.Assumptions relation
      (Layout.Stage1.Wide.PiRLCInputs.interface (logicalWidth := width) (publicFits := fits)) SamplerSourceValues.stageStart env)
    (rRows : Layout.PiRLC.Wide.PhysicalHolds relation
      (Layout.Stage1.Wide.PiRLCInputs.interface (logicalWidth := width) (publicFits := fits)) SamplerSourceValues.stageStart env)
    (completed : PiRLC.Wide.Formal.RangesCompleted
      (Layout.Stage1.Wide.PiRLCInputs.interface (logicalWidth := width) (publicFits := fits)) SamplerSourceValues.stageStart env) :
    let raw := SourceAssignment.raw program env application
    let reference := AssignmentTransportCommon.common raw ++
      AssignmentTransportTail.tail
        (PiRLCWitness.initial (Stage1Plan.piRlcInterface program) (AssignmentProjection.seed program raw.assignment))
        (PiRLCWitness.inputValues (Stage1Plan.piRlcInterface program) (AssignmentProjection.seed program raw.assignment))
    coordinateCount (schedule plan (CommonSchedule.physicalWidth program) (physicalValues program env application)) = coordinateCount reference ∧
      ∀ index, coordinateAt (schedule plan (CommonSchedule.physicalWidth program) (physicalValues program env application)) index = coordinateAt reference index := by
  obtain ⟨common, outputs, commonEmitted, outputEmitted, blocks, _, _, _, _⟩ :=
    emitted_parts program (CommonSchedule.physicalWidth program) plan emitted
  have commonEq := CommonSchedule.common_correct program env application plan common commonEmitted
  have samplerEq := sampler_correct program env application plan relation ajtai template cAssumptions cRows rRows completed
  have fieldsEq := fields_correct program env application plan emitted outputs outputEmitted relation ajtai template
    cAssumptions cRows rAssumptions rRows
  have combined := append_correct commonEq (append_correct samplerEq fieldsEq)
  dsimp only
  rw [schedule, blocks]
  simpa only [List.map_append, List.append_assoc, AssignmentTransportTail.tail] using combined

end NightstreamFPrime.Export.Stage1.Wide.AssignmentTransportSchedule
