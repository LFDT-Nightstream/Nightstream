import NightstreamFPrime.Circuit.StraightLineValues
import NightstreamFPrime.Export.Stage1.PermutationCompilerTransport
import NightstreamFPrime.Export.Stage1.Wide.AssignmentTransportSemantics
import NightstreamFPrime.Lifecycle.PiRLC.v1_1.TranscriptAbsorption.Witness

/-! The physical permutation and compact sampler read the same S-box
values. Causal arithmetic rows supply this correspondence; the separate
range completion proof supplies the hinted inverse values. -/

namespace NightstreamFPrime.Export.Stage1.Wide.SamplerSourceValues

open NightstreamFPrime.Circuit NightstreamFPrime.Spec NightstreamFPrime.Layout
open NightstreamFPrime.Gadgets.Poseidon2
open ProductionRelation
open Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Lifecycle.PaperAlgebra

private def permutationView (env : Env) (start : Nat) (initial : Fin 8 → F) : Env := fun column =>
  if before : column < 8 then initial ⟨column, before⟩ else env (start + (column - 8))

private theorem permutationView_input (env : Env) (start : Nat) (initial : Fin 8 → F) (lane : Fin 8) :
    permutationView env start initial lane.val = initial lane := by
  simp only [permutationView, dif_pos lane.isLt]

private theorem permutationView_local (env : Env) (start : Nat) (initial : Fin 8 → F) (index : Nat) :
    permutationView env start initial (8 + index) = env (start + index) := by
  simp only [permutationView, dif_neg (by omega : ¬8 + index < 8), Nat.add_sub_cancel_left]

/-- Any accepted permutation recipe has the constructor's exact retained
S-box values. The proof is structural in the recipe list. -/
theorem sbox_value (env : Env) (start : Nat) (input : Layer.EState)
    (rows : ConstraintsHold env (recipeConstraints start (Permutation.compile start input Permutation.schedule).recipes))
    (slot : Fin PoseidonRetainedSlots.rows.length) :
    env (start + (PoseidonRetainedSlots.localOutput slot).val) =
      PoseidonCompactWitness.retained (Layer.evalState env input) slot := by
  let initial := Layer.evalState env input
  let view := permutationView env start initial
  have viewInput : Layer.evalState view PoseidonScheduleTrace.canonicalState = initial := by
    funext lane
    exact permutationView_input env start initial lane
  have viewRows := PermutationCompilerTransport.compileConstraintsHold_of_transport view env
    PoseidonScheduleTrace.inputCount start PoseidonScheduleTrace.canonicalState input Permutation.schedule
    viewInput (fun index _ => permutationView_local env start initial index) rows
  have sourceRows := PoseidonCompactWitness.source_rows initial
  have causal := Permutation.compile_schedule_causal PoseidonScheduleTrace.inputCount
    PoseidonScheduleTrace.canonicalState (fun lane => lane.isLt)
  have same := recipeConstraints_values_unique view (PoseidonCompactWitness.source initial)
    PoseidonScheduleTrace.inputCount PoseidonCompactWitness.recipes causal viewRows sourceRows
    (by
      intro index below
      exact (permutationView_input env start initial ⟨index, below⟩).trans
        (PoseidonCompactWitness.source_input initial ⟨index, below⟩).symm)
    (PoseidonRetainedSlots.rows.get slot).step.output.val
    (by
      have bounded := (PoseidonRetainedSlots.rows.get slot).step.output.isLt
      change (PoseidonRetainedSlots.rows.get slot).step.output.val < 600 at bounded
      rw [PoseidonCompactWitness.recipes, Permutation.compile_schedule_recipe_count]
      exact bounded)
  have read : view (PoseidonRetainedSlots.rows.get slot).step.output.val =
      env (start + (PoseidonRetainedSlots.localOutput slot).val) := by
    change permutationView env start initial (PoseidonRetainedSlots.rows.get slot).step.output.val = _
    rw [PoseidonRetainedSlots.output_eq_input_add_local]
    exact permutationView_local env start initial _
  exact read.symm.trans same

theorem output_value (env : Env) (start : Nat) (input : Layer.EState)
    (rows : ConstraintsHold env (recipeConstraints start (Permutation.compile start input Permutation.schedule).recipes)) :
    Layer.evalState env (Permutation.scheduleOutput start) =
      PoseidonCompactWitness.output (Layer.evalState env input) := by
  have result := Permutation.compile_schedule_sound env start input rows
  rw [← Permutation.scheduleOutput_eq_compile] at result
  exact List.ofFn_inj.mp (result.trans (PoseidonCompactWitness.output_eq_permute _).symm)

open NightstreamFPrime.Lifecycle.PiRLC.Wide
open NightstreamFPrime.Lifecycle.PiRLC.v1_1

private theorem domain_even (before : PiRlcWideSampler.Witness.FieldState) (source : Nat) :
    PiRlcWideSampler.Witness.domainInput before (source * 2) =
      fun lane => before lane + PiRlcWideSampler.BatchPlan.entryWord source lane := by
  funext lane
  simp only [PiRlcWideSampler.Witness.domainInput, if_pos (Nat.mul_mod_left _ _),
    show source * 2 / 2 = source by omega]

private theorem domain_odd (before : PiRlcWideSampler.Witness.FieldState) (source : Nat) :
    PiRlcWideSampler.Witness.domainInput before (source * 2 + 1) = before := by
  funext lane
  simp only [PiRlcWideSampler.Witness.domainInput, if_neg (by omega : ¬(source * 2 + 1) % 2 = 0)]

theorem scalar_rows (interface : Batch.Interface) (offset : Nat) (env : Env)
    (rows : holdsFlat env (Batch.operations interface offset)) (source : Fin 17) :
    holdsFlat env (Scalar.operations (Batch.childInterface interface offset source.val)
      source.val (Batch.sourceOffset offset source.val)) := by
  exact Sequence.child_rows env (Batch.operations interface offset) rows (Batch.childName source.val)
    (Scalar.circuit (Batch.childInterface interface offset source.val) source.val) (Batch.sourceOffset offset source.val)
    (Batch.child_member interface offset Batch.sourceCount source.val source.isLt)

theorem entry_rows (interface : Batch.Interface) (offset : Nat) (env : Env)
    (rows : holdsFlat env (Batch.operations interface offset)) (source : Fin 17) :
    ConstraintsHold env (recipeConstraints (Batch.sourceOffset offset source.val)
      (Permutation.compile (Batch.sourceOffset offset source.val)
        (TranscriptAbsorption.permutationInput (Batch.childInterface interface offset source.val)
          source.val (Batch.sourceOffset offset source.val)) Permutation.schedule).recipes) := by
  have child := Sequence.child_rows env _ (scalar_rows interface offset env rows source)
    "pirlc.wide.enter_scalar" (Scalar.entry (Batch.childInterface interface offset source.val) source.val)
    (Batch.sourceOffset offset source.val) (by exact List.Mem.head _)
  change holdsFlat env (Circuit.ops (TranscriptAbsorption.circuit
    (Batch.childInterface interface offset source.val) source.val).main (Batch.sourceOffset offset source.val)) at child
  rw [holdsFlat, TranscriptAbsorption.witnessConstraints] at child
  exact child

theorem advance_rows (interface : Batch.Interface) (offset : Nat) (env : Env)
    (rows : holdsFlat env (Batch.operations interface offset)) (source : Fin 17) :
    ConstraintsHold env (recipeConstraints (Scalar.advanceOffset (Batch.sourceOffset offset source.val))
      (Permutation.compile (Scalar.advanceOffset (Batch.sourceOffset offset source.val))
        (Scalar.enteredState (Batch.childInterface interface offset source.val) source.val
          (Batch.sourceOffset offset source.val)) Permutation.schedule).recipes) := by
  have child := Sequence.child_rows env _ (scalar_rows interface offset env rows source)
    "pirlc.wide.advance"
    (Scalar.advance (Batch.childInterface interface offset source.val) source.val (Batch.sourceOffset offset source.val))
    (Scalar.advanceOffset (Batch.sourceOffset offset source.val)) (by exact List.Mem.tail _ (List.Mem.tail _ (List.Mem.head _)))
  change holdsFlat env (Permutation.Owned.operations _ _) at child
  rw [holdsFlat, Permutation.Owned.flatConstraints_operations] at child
  exact child

theorem entry_input (interface : Batch.Interface) (offset : Nat) (env : Env) (source : Nat) :
    Layer.evalState env (TranscriptAbsorption.permutationInput (Batch.childInterface interface offset source)
      source (Batch.sourceOffset offset source)) =
      fun lane => (Layer.evalState env (Batch.stateAtExpr interface offset source)) lane +
        PiRlcWideSampler.BatchPlan.entryWord source lane := by
  funext lane
  fin_cases lane <;> rfl

theorem entry_state (interface : Batch.Interface) (offset : Nat) (env : Env)
    (rows : holdsFlat env (Batch.operations interface offset)) (source : Fin 17) :
    Layer.evalState env (Scalar.enteredState (Batch.childInterface interface offset source.val) source.val
      (Batch.sourceOffset offset source.val)) =
      PoseidonCompactWitness.output (fun lane =>
        (Layer.evalState env (Batch.stateAtExpr interface offset source.val)) lane +
          PiRlcWideSampler.BatchPlan.entryWord source.val lane) := by
  rw [Scalar.enteredState, TranscriptAbsorption.output_eq_permutation]
  rw [output_value env _ _ (entry_rows interface offset env rows source), entry_input]

/-- Two physical permutations per scalar follow the compact constructor's
exact state sequence. The proof is by source count, not by emitted rows. -/
theorem boundary_state (interface : Batch.Interface) (offset : Nat) (env : Env)
    (rows : holdsFlat env (Batch.operations interface offset)) (count : Nat) (bounded : count ≤ 17) :
    Layer.evalState env (Batch.stateAtExpr interface offset count) =
      PiRlcWideSampler.Witness.stateAt (Layer.evalState env (interface.initialState offset)) (count * 2) := by
  induction count with
  | zero => rfl
  | succ count ih =>
    have within : count < 17 := by omega
    have entered := entry_state interface offset env rows ⟨count, within⟩
    rw [ih (by omega)] at entered
    have advanced := output_value env _ _ (advance_rows interface offset env rows ⟨count, within⟩)
    rw [entered] at advanced
    change Layer.evalState env (Scalar.outputState (Batch.childInterface interface offset count) count
      (Batch.sourceOffset offset count)) = _
    change Layer.evalState env (Permutation.scheduleOutput (Scalar.advanceOffset (Batch.sourceOffset offset count))) = _
    rw [advanced]
    rw [show (count + 1) * 2 = (count * 2 + 1) + 1 by omega,
      PiRlcWideSampler.Witness.stateAt, domain_odd, PiRlcWideSampler.Witness.stateAt, domain_even]

theorem draw_value (interface : Batch.Interface) (offset : Nat) (env : Env)
    (rows : holdsFlat env (Batch.operations interface offset)) (source : Fin 17) :
    (fun lane : Fin 4 => ((Scalar.rangeInterface (Batch.childInterface interface offset source.val) source.val
      (Batch.sourceOffset offset source.val)).source lane (Scalar.rangeOffset (Batch.sourceOffset offset source.val))).eval env) =
      PiRlcWideSampler.Witness.drawAt (Layer.evalState env (interface.initialState offset)) source := by
  have entered := entry_state interface offset env rows source
  rw [boundary_state interface offset env rows source.val source.isLt.le] at entered
  funext lane
  change (Layer.evalState env (Scalar.enteredState (Batch.childInterface interface offset source.val) source.val
    (Batch.sourceOffset offset source.val))) ⟨lane.val, _⟩ = _
  rw [entered]
  unfold PiRlcWideSampler.Witness.drawAt
  rw [PiRlcWideSampler.Witness.stateAt, domain_even]

theorem entry_sbox (interface : Batch.Interface) (offset : Nat) (env : Env)
    (rows : holdsFlat env (Batch.operations interface offset)) (source : Fin 17)
    (slot : Fin PoseidonRetainedSlots.rows.length) :
    env (Batch.sourceOffset offset source.val + (PoseidonRetainedSlots.localOutput slot).val) =
      PoseidonCompactWitness.retained
        (PiRlcWideSampler.Witness.inputAt (Layer.evalState env (interface.initialState offset)) (source.val * 2)) slot := by
  rw [sbox_value env _ _ (entry_rows interface offset env rows source), entry_input,
    boundary_state interface offset env rows source.val source.isLt.le]
  unfold PiRlcWideSampler.Witness.inputAt
  rw [domain_even]

theorem advance_sbox (interface : Batch.Interface) (offset : Nat) (env : Env)
    (rows : holdsFlat env (Batch.operations interface offset)) (source : Fin 17)
    (slot : Fin PoseidonRetainedSlots.rows.length) :
    env (Scalar.advanceOffset (Batch.sourceOffset offset source.val) + (PoseidonRetainedSlots.localOutput slot).val) =
      PoseidonCompactWitness.retained
        (PiRlcWideSampler.Witness.inputAt (Layer.evalState env (interface.initialState offset)) (source.val * 2 + 1)) slot := by
  rw [sbox_value env _ _ (advance_rows interface offset env rows source), entry_state interface offset env rows source,
    boundary_state interface offset env rows source.val source.isLt.le]
  unfold PiRlcWideSampler.Witness.inputAt
  rw [domain_odd, PiRlcWideSampler.Witness.stateAt, domain_even]

theorem range_value (interface : Batch.Interface) (offset : Nat) (env : Env)
    (inputs : Batch.Assumptions interface offset)
    (rows : holdsFlat env (Batch.operations interface offset))
    (completed : Batch.RangesCompleted interface offset Batch.sourceCount env)
    (source : Fin 17) (index : Nat) (bounded : index < Gadgets.Sampling.WideReduction.privateCount) :
    env (Gadgets.Sampling.WideReduction.Program.coreOffset (Scalar.rangeOffset (Batch.sourceOffset offset source.val)) + index) =
      PiRlcWideSampler.Witness.rangeSources (Layer.evalState env (interface.initialState offset)) source (1408 + index) := by
  have read := AssignmentTransportSemantics.completed_range_value (Batch.childInterface interface offset source.val)
    source.val (Batch.sourceOffset offset source.val) env (Batch.child_inputs interface offset source.val inputs)
    (completed source.val source.isLt) index bounded
  dsimp only at read
  rw [draw_value interface offset env rows source] at read
  exact read

variable {width : Nat}
  {fits : ringDegree * publicRingColumns ≤ Phi81CarrierLayout.carrierWidth width}

def stageInterface : Batch.Interface :=
  Lifecycle.PiRLC.Wide.Formal.samplerInterface (Lifecycle.PiRLC.Wide.Formal.atOffset
    (Layout.Stage1.Wide.PiRLCInputs.interface (logicalWidth := width) (publicFits := fits))
    Layout.Stage1.Wide.PiRLCInputs.phaseOffset)

abbrev stageStart := Layout.Stage1.Wide.PiRLCInputs.phaseOffset

theorem stage_rows (relation : Lifecycle.ProductionKey.LogicalRelation width fits) (env : Env)
    (rows : Layout.PiRLC.Wide.PhysicalHolds relation
      (Layout.Stage1.Wide.PiRLCInputs.interface (logicalWidth := width) (publicFits := fits)) stageStart env) :
    holdsFlat env (Batch.operations (stageInterface (width := width) (fits := fits)) stageStart) := by
  have phase := Layout.PiRLC.Wide.physical_implies_holdsFlat relation _ _ env rows
  rw [Lifecycle.PiRLC.Wide.Formal.main_ops] at phase
  have sampled := Sequence.child_rows env _ phase "pirlc.wide.sampler"
    (Lifecycle.PiRLC.Wide.Formal.samplerCircuit (Lifecycle.PiRLC.Wide.Formal.atOffset
      (Layout.Stage1.Wide.PiRLCInputs.interface (logicalWidth := width) (publicFits := fits)) stageStart))
    stageStart (by exact List.Mem.tail _ (List.Mem.head _))
  exact Sequence.child_rows env _ sampled "pirlc.wide.batch"
    (Batch.circuit (stageInterface (width := width) (fits := fits))) stageStart (by exact List.Mem.head _)

private theorem late_source (program : AssignmentTransport.Program) (env : Env)
    (application : Fin (PerApplicationPackage.addedPrivateColumnCount program) → F)
    (source : Nat) (lower : 14751804 ≤ source) (upper : source < Layout.Stage1.Wide.SourceOrder.sourceWidth) :
    AssignmentTransportSemantics.physicalValues program env application (Layout.Stage1.Wide.SourceOrder.column source) =
      env source := by
  apply AssignmentTransportSemantics.physicalValues_private_source program env application source upper
  rw [Layout.Stage1.Wide.SourceOrder.column_late source lower upper, Layout.Stage1.Wide.SourceOrder.privateColumns_eq]
  rw [Layout.Stage1.Wide.SourceOrder.sourceWidth_eq] at upper
  omega

theorem physical_sbox (program : AssignmentTransport.Program) (env : Env)
    (application : Fin (PerApplicationPackage.addedPrivateColumnCount program) → F)
    (relation : Lifecycle.ProductionKey.LogicalRelation width fits)
    (rows : Layout.PiRLC.Wide.PhysicalHolds relation
      (Layout.Stage1.Wide.PiRLCInputs.interface (logicalWidth := width) (publicFits := fits)) stageStart env)
    (index : Fin (34 * 86)) :
    AssignmentTransportSemantics.physicalValues program env application
        (AffineRuns.sourceAt AssignmentTransport.samplerPoseidon.sources index.val) =
      PiRlcWideSampler.Witness.sboxValues
        (Layer.evalState env ((stageInterface (width := width) (fits := fits)).initialState stageStart)) index := by
  let invocation := index.val / 86
  let source : Fin 17 := ⟨invocation / 2, by dsimp only [invocation]; omega⟩
  let slot : Fin PoseidonRetainedSlots.rows.length := ⟨index.val % 86, by rw [PoseidonRetainedSlots.rows_length]; omega⟩
  have sourceBound : source.val < 17 := source.isLt
  have localBound : (PoseidonRetainedSlots.localOutput slot).val < 592 := (PoseidonRetainedSlots.localOutput slot).isLt
  have batchRows := stage_rows relation env rows
  rw [AssignmentTransport.samplerPoseidon, AssignmentTransport.Values.ofSource_source]
  change AssignmentTransportSemantics.physicalValues program env application
      (Layout.Stage1.Wide.SourceOrder.column
        (if invocation % 2 = 0 then Layout.Stage1.Wide.PiRLCStarts.entryLogicalStart source.val
          else Layout.Stage1.Wide.PiRLCStarts.advanceLogicalStart source.val) +
        (PoseidonRetainedSlots.localOutput slot).val) =
    PoseidonCompactWitness.retained
      (PiRlcWideSampler.Witness.inputAt
        (Layer.evalState env ((stageInterface (width := width) (fits := fits)).initialState stageStart)) invocation) slot
  by_cases even : invocation % 2 = 0
  · rw [if_pos even]
    have lower : 14751804 ≤ Layout.Stage1.Wide.PiRLCStarts.entryLogicalStart source.val := by
      change 14751804 ≤ 19513117 + source.val * 3205; omega
    have upper : Layout.Stage1.Wide.PiRLCStarts.entryLogicalStart source.val +
        (PoseidonRetainedSlots.localOutput slot).val < Layout.Stage1.Wide.SourceOrder.sourceWidth := by
      change 19513117 + source.val * 3205 + (PoseidonRetainedSlots.localOutput slot).val < 27859538; omega
    rw [← Layout.Stage1.Wide.SourceOrder.column_late_add _ _ lower upper,
      late_source program env application _ (by omega) upper]
    have value := entry_sbox (stageInterface (width := width) (fits := fits)) stageStart env batchRows source slot
    have position : source.val * 2 = invocation := by dsimp only [source]; omega
    rw [position] at value
    exact value
  · rw [if_neg even]
    have lower : 14751804 ≤ Layout.Stage1.Wide.PiRLCStarts.advanceLogicalStart source.val := by
      change 14751804 ≤ 19513117 + source.val * 3205 + 2613; omega
    have upper : Layout.Stage1.Wide.PiRLCStarts.advanceLogicalStart source.val +
        (PoseidonRetainedSlots.localOutput slot).val < Layout.Stage1.Wide.SourceOrder.sourceWidth := by
      change 19513117 + source.val * 3205 + 2613 + (PoseidonRetainedSlots.localOutput slot).val < 27859538; omega
    rw [← Layout.Stage1.Wide.SourceOrder.column_late_add _ _ lower upper,
      late_source program env application _ (by omega) upper]
    have value := advance_sbox (stageInterface (width := width) (fits := fits)) stageStart env batchRows source slot
    have position : source.val * 2 + 1 = invocation := by dsimp only [source]; omega
    rw [position] at value
    exact value

theorem physical_range (program : AssignmentTransport.Program) (env : Env)
    (application : Fin (PerApplicationPackage.addedPrivateColumnCount program) → F)
    (relation : Lifecycle.ProductionKey.LogicalRelation width fits)
    (rows : Layout.PiRLC.Wide.PhysicalHolds relation
      (Layout.Stage1.Wide.PiRLCInputs.interface (logicalWidth := width) (publicFits := fits)) stageStart env)
    (completed : Lifecycle.PiRLC.Wide.Formal.RangesCompleted
      (Layout.Stage1.Wide.PiRLCInputs.interface (logicalWidth := width) (publicFits := fits)) stageStart env)
    (source : Fin 17) (block : LowNormBlock.Block 2025) (index : Fin block.slotCount)
    (retained : 1408 ≤ (block.source index).val) :
    AssignmentTransportSemantics.physicalValues program env application
        (AffineRuns.sourceAt (AssignmentTransport.rangeBlock source.val block).sources index.val) =
      PiRlcWideSampler.Witness.rangeSources
        (Layer.evalState env ((stageInterface (width := width) (fits := fits)).initialState stageStart)) source
        (block.source index).val := by
  have sourceBound : source.val < 17 := source.isLt
  have cellBound : (block.source index).val < 2025 := (block.source index).isLt
  rw [AssignmentTransport.rangeBlock_physical_source source block index (by omega)]
  have lower : 14751804 ≤ Layout.Stage1.Wide.PiRLCStarts.rangeLogicalStart source.val + ((block.source index).val - 4) := by
    change 14751804 ≤ 19513117 + source.val * 3205 + 592 + ((block.source index).val - 4); omega
  have upper : Layout.Stage1.Wide.PiRLCStarts.rangeLogicalStart source.val + ((block.source index).val - 4) <
      Layout.Stage1.Wide.SourceOrder.sourceWidth := by
    change 19513117 + source.val * 3205 + 592 + ((block.source index).val - 4) < 27859538; omega
  rw [late_source program env application _ lower upper]
  have value := range_value (stageInterface (width := width) (fits := fits)) stageStart env
    (Layout.Stage1.Wide.PiRLCInputBounds.assumptions relation env).sampler (stage_rows relation env rows)
    completed source ((block.source index).val - 1408) (by change _ < 617; omega)
  have location : Gadgets.Sampling.WideReduction.Program.coreOffset
      (Scalar.rangeOffset (Batch.sourceOffset stageStart source.val)) + ((block.source index).val - 1408) =
      Layout.Stage1.Wide.PiRLCStarts.rangeLogicalStart source.val + ((block.source index).val - 4) := by
    change 19513117 + source.val * 3205 + 592 + 1404 + ((block.source index).val - 1408) =
      19513117 + source.val * 3205 + 592 + ((block.source index).val - 4)
    omega
  rw [location, Nat.add_sub_of_le retained] at value
  exact value

theorem physical_digitBit (program : AssignmentTransport.Program) (env : Env)
    (application : Fin (PerApplicationPackage.addedPrivateColumnCount program) → F)
    (plan : AssignmentTransport.Plan) (sources : plan.challengeSources = AssignmentTransport.challenges)
    (source : Fin 17) (lane : Fin ringDegree) (bit : Fin 3) :
    AssignmentTransportExecution.digitBit plan (AssignmentTransportSemantics.physicalValues program env application) source lane bit =
      env (Gadgets.Sampling.WideReduction.digitStart (Gadgets.Sampling.WideReduction.Program.coreOffset
        (Layout.Stage1.Wide.PiRLCStarts.rangeLogicalStart source.val)) + 3 * lane.val + bit.val) := by
  rw [AssignmentTransportExecution.digitBit, sources, AssignmentTransportSemantics.challenge_source]
  apply late_source
  · change 14751804 ≤ 19513117 + source.val * 3205 + 592 + 1404 + 264 + 131 + 3 * lane.val + bit.val
    omega
  · have s : source.val < 17 := source.isLt
    have l : lane.val < 54 := lane.isLt
    have b : bit.val < 3 := bit.isLt
    change 19513117 + source.val * 3205 + 592 + 1404 + 264 + 131 + 3 * lane.val + bit.val < 27859538
    omega

theorem physical_digit (program : AssignmentTransport.Program) (env : Env)
    (application : Fin (PerApplicationPackage.addedPrivateColumnCount program) → F)
    (plan : AssignmentTransport.Plan) (sources : plan.challengeSources = AssignmentTransport.challenges)
    (source : Fin 17) (lane : Fin ringDegree) :
    AssignmentTransportExecution.digit plan (AssignmentTransportSemantics.physicalValues program env application) source lane =
      Gadgets.Sampling.WideReduction.digitValue env (Gadgets.Sampling.WideReduction.Program.coreOffset
        (Layout.Stage1.Wide.PiRLCStarts.rangeLogicalStart source.val)) lane.val := by
  simp only [AssignmentTransportExecution.digit, physical_digitBit program env application plan sources,
    Gadgets.Sampling.WideReduction.digitValue, Gadgets.Sampling.WideReduction.digitBit,
    Gadgets.Sampling.WideReduction.digitBitCount, Expr.eval_var, Nat.add_zero]

theorem digit_sampled (program : AssignmentTransport.Program) (env : Env)
    (application : Fin (PerApplicationPackage.addedPrivateColumnCount program) → F)
    (plan : AssignmentTransport.Plan) (sources : plan.challengeSources = AssignmentTransport.challenges)
    (relation : Lifecycle.ProductionKey.LogicalRelation width fits)
    (rows : Layout.PiRLC.Wide.PhysicalHolds relation
      (Layout.Stage1.Wide.PiRLCInputs.interface (logicalWidth := width) (publicFits := fits)) stageStart env)
    (source : Fin 17) (lane : Fin ringDegree) :
    AssignmentTransportExecution.digit plan (AssignmentTransportSemantics.physicalValues program env application) source lane =
      (Spec.Folding.Nifs.NonInteractive.PiRlcWideSampler.Transcript.scalarAt
        (List.ofFn (Layer.evalState env ((stageInterface (width := width) (fits := fits)).initialState stageStart))) source.val lane).val := by
  rw [physical_digit program env application plan sources]
  have specification := Layout.PiRLC.Wide.physical_implies_specHolds relation _ _ env
    (Layout.Stage1.Wide.PiRLCInputBounds.assumptions relation env) rows
  exact specification.sampler.toSpecHolds.digits source lane

theorem challenge_value (program : AssignmentTransport.Program) (env : Env)
    (application : Fin (PerApplicationPackage.addedPrivateColumnCount program) → F)
    (plan : AssignmentTransport.Plan) (sources : plan.challengeSources = AssignmentTransport.challenges)
    (relation : Lifecycle.ProductionKey.LogicalRelation width fits)
    (rows : Layout.PiRLC.Wide.PhysicalHolds relation
      (Layout.Stage1.Wide.PiRLCInputs.interface (logicalWidth := width) (publicFits := fits)) stageStart env)
    (source : Fin 17) :
    AssignmentTransportExecution.challengeRing plan (AssignmentTransportSemantics.physicalValues program env application) source =
      PiRLCValues.challenge (Layer.evalState env ((stageInterface (width := width) (fits := fits)).initialState stageStart)) source.val := by
  funext lane
  change Gadgets.Sampling.WideReduction.fieldOfNat (AssignmentTransportExecution.digit plan _ source lane) - 2 = _
  rw [digit_sampled program env application plan sources relation rows]
  exact Batch.centered_digit _

theorem completed_digitBit (program : AssignmentTransport.Program) (env : Env)
    (application : Fin (PerApplicationPackage.addedPrivateColumnCount program) → F)
    (plan : AssignmentTransport.Plan) (sources : plan.challengeSources = AssignmentTransport.challenges)
    (relation : Lifecycle.ProductionKey.LogicalRelation width fits)
    (rows : Layout.PiRLC.Wide.PhysicalHolds relation
      (Layout.Stage1.Wide.PiRLCInputs.interface (logicalWidth := width) (publicFits := fits)) stageStart env)
    (completed : Lifecycle.PiRLC.Wide.Formal.RangesCompleted
      (Layout.Stage1.Wide.PiRLCInputs.interface (logicalWidth := width) (publicFits := fits)) stageStart env)
    (source : Fin 17) (lane : Fin ringDegree) (bit : Fin 3) :
    AssignmentTransportExecution.digitBit plan (AssignmentTransportSemantics.physicalValues program env application) source lane bit =
      PiRlcWideSampler.Witness.rangeSources
        (Layer.evalState env ((stageInterface (width := width) (fits := fits)).initialState stageStart)) source
        (1803 + 3 * lane.val + bit.val) := by
  rw [physical_digitBit program env application plan sources]
  have l : lane.val < 54 := lane.isLt
  have b : bit.val < 3 := bit.isLt
  have read := range_value (stageInterface (width := width) (fits := fits)) stageStart env
    (Layout.Stage1.Wide.PiRLCInputBounds.assumptions relation env).sampler (stage_rows relation env rows)
    completed source (395 + 3 * lane.val + bit.val) (by change _ < 617; omega)
  have address : Gadgets.Sampling.WideReduction.digitStart (Gadgets.Sampling.WideReduction.Program.coreOffset
      (Layout.Stage1.Wide.PiRLCStarts.rangeLogicalStart source.val)) + 3 * lane.val + bit.val =
      Gadgets.Sampling.WideReduction.Program.coreOffset (Scalar.rangeOffset (Batch.sourceOffset stageStart source.val)) +
        (395 + 3 * lane.val + bit.val) := by
    change 19513117 + source.val * 3205 + 592 + 1404 + 264 + 131 + 3 * lane.val + bit.val =
      19513117 + source.val * 3205 + 592 + 1404 + (395 + 3 * lane.val + bit.val)
    omega
  rw [address, read]
  congr 1
  omega

/-- The canonical physical completion passes the executor's explicit digit checks. -/
theorem digits_valid (program : AssignmentTransport.Program) (env : Env)
    (application : Fin (PerApplicationPackage.addedPrivateColumnCount program) → F)
    (plan : AssignmentTransport.Plan) (sources : plan.challengeSources = AssignmentTransport.challenges)
    (relation : Lifecycle.ProductionKey.LogicalRelation width fits)
    (rows : Layout.PiRLC.Wide.PhysicalHolds relation
      (Layout.Stage1.Wide.PiRLCInputs.interface (logicalWidth := width) (publicFits := fits)) stageStart env)
    (completed : Lifecycle.PiRLC.Wide.Formal.RangesCompleted
      (Layout.Stage1.Wide.PiRLCInputs.interface (logicalWidth := width) (publicFits := fits)) stageStart env) :
    AssignmentTransportExecution.DigitsValid plan (AssignmentTransportSemantics.physicalValues program env application) := by
  constructor
  · intro source lane bit
    rw [completed_digitBit program env application plan sources relation rows completed source lane bit]
    exact AssignmentTransportSemantics.range_digit_bit _ lane bit
  · intro source lane
    rw [digit_sampled program env application plan sources relation rows]
    have bound : (Spec.Folding.Nifs.NonInteractive.PiRlcWideSampler.Transcript.scalarAt
        (List.ofFn (Layer.evalState env ((stageInterface (width := width) (fits := fits)).initialState stageStart)))
        source.val lane).val < 5 := (Spec.Folding.Nifs.NonInteractive.PiRlcWideSampler.Transcript.scalarAt _ _ _).isLt
    omega

end NightstreamFPrime.Export.Stage1.Wide.SamplerSourceValues
