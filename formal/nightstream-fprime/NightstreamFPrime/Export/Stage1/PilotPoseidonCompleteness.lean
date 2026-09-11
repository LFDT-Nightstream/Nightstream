import NightstreamFPrime.Export.Stage1.PerApplicationSourceAssignment
import NightstreamFPrime.Export.Stage1.PilotPoseidonPreservation
import NightstreamFPrime.Export.PermutationOutput
import NightstreamFPrime.Layout.ProductionRelation.PoseidonSboxSourceCompleteness

/-!
Owns the pilot hash-row consumer for the complete canonical source copy.
Actual prior/output hash rows determine each retained S-box source value and
its complete source permutation trace. The existing adjacent compiler theorem
then proves the direct pilot plan's rows. No hash-output or retained-equation
assumption replaces the physical hash rows.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.PilotPoseidonCompleteness

open NightstreamFPrime.Circuit
open NightstreamFPrime.Export.Package
open NightstreamFPrime.Gadgets.Poseidon2
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Layout.Stage1
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra
open PerApplicationCanonicalAssignment
open PerApplicationAssignmentTransportExecution

private def templateEnv (chain : HashChain) (invocation : Nat) (target : Env) : Env :=
  fun column => (PilotData.columnRef column).eval (fun reference =>
    (instantiateColumn (PilotData.circuitPackage ()) chain invocation reference).eval target)

private theorem templateEnv_input (chain : HashChain) (invocation : Nat)
    (target : Env) (lane : Fin 8) :
    templateEnv chain invocation target lane.val =
      (invocationInput (PilotData.circuitPackage ()) chain invocation lane.val).eval target := by
  unfold templateEnv
  rw [show PilotData.columnRef lane.val = .input lane.val by
    simp [PilotData.columnRef, lane.isLt]]
  rfl

private theorem templateEnv_local (chain : HashChain) (invocation : Nat)
    (target : Env) (index : Nat) :
    templateEnv chain invocation target (8 + index) =
      target (invocationLocalStart (PilotData.circuitPackage ()) chain invocation + index) := by
  unfold templateEnv
  rw [show PilotData.columnRef (8 + index) = .local index by simp [PilotData.columnRef]]
  simp [ColumnRef.eval, instantiateColumn]

private theorem template_constraints (chain : HashChain) (invocation : Nat)
    (target : Env)
    (rows : TemplateInvocationHolds (Data.circuitPackage ()) chain invocation target) :
    ConstraintsHold (templateEnv chain invocation target) (PilotData.canonicalConstraints ()) := by
  have pilotRows : TemplateInvocationHolds (PilotData.circuitPackage ()) chain invocation target := by
    intro row member
    apply rows row
    rw [Data.circuitPackage_permutation]
    exact member
  have physical := (Pilot.canonicalTemplateInvocation_iff chain invocation target).mp pilotRows
  exact R1CS.lowerConstraints_sound (templateEnv chain invocation target)
    (PilotData.canonicalConstraints ()) 600 physical

private theorem template_finalLayer (chain : HashChain) (invocation : Nat)
    (target : Env)
    (rows : TemplateInvocationHolds (Data.circuitPackage ()) chain invocation target) :
    (fun lane : Fin 8 => target
      (invocationLocalStart (PilotData.circuitPackage ()) chain invocation + 584 + lane.val)) =
      Layer.externalF (fun lane => target
        (invocationLocalStart (PilotData.circuitPackage ()) chain invocation +
          (PoseidonRetainedSlots.localOutput (PoseidonRetainedSlots.finalRow lane)).val)) := by
  have final := PermutationOutput.canonical_finalLayer
    (templateEnv chain invocation target) (template_constraints chain invocation target rows)
  have outputEq : Layer.evalState (templateEnv chain invocation target)
      (Permutation.scheduleOutput PoseidonScheduleTrace.inputCount) =
      fun lane : Fin 8 => target
        (invocationLocalStart (PilotData.circuitPackage ()) chain invocation + 584 + lane.val) := by
    funext lane
    change templateEnv chain invocation target (592 + lane.val) = _
    rw [show 592 + lane.val = 8 + (584 + lane.val) by omega, templateEnv_local]
    congr 1
    omega
  have boxesEq : (fun lane : Fin 8 => templateEnv chain invocation target
      (PoseidonRetainedSlots.rows.get (PoseidonRetainedSlots.finalRow lane)).step.output.val) =
      fun lane => target
        (invocationLocalStart (PilotData.circuitPackage ()) chain invocation +
          (PoseidonRetainedSlots.localOutput (PoseidonRetainedSlots.finalRow lane)).val) := by
    funext lane
    rw [PoseidonRetainedSlots.output_eq_input_add_local]
    exact templateEnv_local chain invocation target _
  rw [outputEq, boxesEq] at final
  exact final

private theorem retainedSource_private
    (application : Lifecycle.Stage1.Application.Program) (target : Env)
    (applicationPrivate : Fin (PerApplicationPackage.addedPrivateColumnCount application) → F)
    (raw : RawValues application)
    (baseEq : raw.base = PerApplicationSourceAssignment.ofCompleted application target applicationPrivate)
    (column : Fin (PiRLCRetainedGeometry.sourceWidth application))
    (privateColumn : column.val < PiRLCProductPlan.basePackage.layout.constantColumn) :
    raw.retainedSource column = target column.val := by
  let baseColumn : Fin (PiRLCProductPlan.baseSourceWidth application) :=
    ⟨column.val, lt_of_lt_of_le privateColumn (PiRLCProductPlan.basePackage_fits application)⟩
  have same : column = PiRLCRetainedPreservation.baseSourceColumn application baseColumn := by
    apply Fin.ext
    rfl
  have before : baseColumn.val < PerApplicationPackage.basePackage.layout.constantColumn := privateColumn
  calc
    raw.retainedSource column = raw.retainedSource
        (PiRLCRetainedPreservation.baseSourceColumn application baseColumn) :=
      congrArg raw.retainedSource same
    _ = raw.base baseColumn :=
      PiRLCRetainedPreservation.sourceAssignment_base application raw.base
        raw.groupValue raw.products baseColumn
    _ = PerApplicationSourceAssignment.ofCompleted application target
        applicationPrivate baseColumn :=
      congrArg (fun base : BaseValues application => base baseColumn) baseEq
    _ = target column.val := by
      simp only [PerApplicationSourceAssignment.ofCompleted, dif_pos before]
      rfl

private theorem priorSlot_source (application : Lifecycle.Stage1.Application.Program)
    (invocation : Fin PilotPoseidonPlan.invocationCount)
    (row : Fin PoseidonRetainedSlots.rows.length) :
    ((PilotPoseidonPlan.priorSchedule application).block.source
      (PoseidonRetainedFamily.slot (PilotPoseidonPlan.priorSchedule application) invocation row)).val =
      PoseidonRetainedBlock.priorWitnessStart invocation +
        (PoseidonRetainedSlots.localOutput row).val := by
  have selected : PoseidonRetainedFamily.slot (PilotPoseidonPlan.priorSchedule application)
      invocation row = (Fin.encodeProd (invocation, row) : Fin PoseidonRetainedBlock.priorBlock.slotCount) := by
    apply Fin.ext
    rfl
  rw [selected]
  simp only [PilotPoseidonPlan.priorSchedule,
    PiRLCRetainedGeometry.priorPoseidonBlock, LowNormBlock.Block.lift,
    PoseidonRetainedBlock.priorBlock,
    Layout.ProductionRelation.PoseidonRetainedBlock.block, Fin.decodeProd_encodeProd]

private theorem outputSlot_source (application : Lifecycle.Stage1.Application.Program)
    (invocation : Fin PilotPoseidonPlan.invocationCount)
    (row : Fin PoseidonRetainedSlots.rows.length) :
    ((PilotPoseidonPlan.outputSchedule application).block.source
      (PoseidonRetainedFamily.slot (PilotPoseidonPlan.outputSchedule application) invocation row)).val =
      PoseidonRetainedBlock.outputWitnessStart invocation +
        (PoseidonRetainedSlots.localOutput row).val := by
  have selected : PoseidonRetainedFamily.slot (PilotPoseidonPlan.outputSchedule application)
      invocation row = (Fin.encodeProd (invocation, row) : Fin PoseidonRetainedBlock.outputBlock.slotCount) := by
    apply Fin.ext
    rfl
  rw [selected]
  simp only [PilotPoseidonPlan.outputSchedule,
    PiRLCRetainedGeometry.outputPoseidonBlock, LowNormBlock.Block.lift,
    PoseidonRetainedBlock.outputBlock,
    Layout.ProductionRelation.PoseidonRetainedBlock.block, Fin.decodeProd_encodeProd]

section Copied

variable (application : Lifecycle.Stage1.Application.Program) (target : Env)
  (applicationPrivate : Fin (PerApplicationPackage.addedPrivateColumnCount application) → F)
  (raw : RawValues application)
  (baseEq : raw.base = PerApplicationSourceAssignment.ofCompleted application target applicationPrivate)

include baseEq

local notation "pilotGeometry" => DirectPrefixPlan.pilotGeometry
  (PerApplicationCanonicalEncodes.poseidonGeometry application)

private theorem form_eval_private
    (block : LowNormBlock.Block (PiRLCRetainedGeometry.sourceWidth application))
    (start : Nat)
    (fits : start + block.coordinateCount ≤ PerApplicationFixedPoint.logicalWidth application)
    (encodes : block.EncodesAt start fits raw.assignment raw.retainedSource)
    (slot : Fin block.slotCount)
    (privateColumn : (block.source slot).val <
      PiRLCProductPlan.basePackage.layout.constantColumn) :
    (block.form start fits slot).eval raw.assignment = target (block.source slot).val := by
  exact (LowNormBlock.Block.form_eval block start fits raw.assignment
    raw.retainedSource encodes slot).trans
      (retainedSource_private application target applicationPrivate raw baseEq
        (block.source slot) privateColumn)

private theorem prior_form
    (invocation : Fin PilotPoseidonPlan.invocationCount)
    (row : Fin PoseidonRetainedSlots.rows.length) :
    (PoseidonRetainedFamily.form (PilotPoseidonPlan.priorSchedule application)
      (PiRLCRetainedGeometry.priorPoseidonStart application)
      (PiRLCRetainedGeometry.priorPoseidonFits
        (PiRLCPoseidonGeometry.prefixGeometry pilotGeometry)) invocation row).eval raw.assignment =
      target (invocationLocalStart (PilotData.circuitPackage ()) Data.priorChain invocation.val +
        (PoseidonRetainedSlots.localOutput row).val) := by
  have encoding : (PilotPoseidonPlan.priorSchedule application).block.EncodesAt
      (PiRLCRetainedGeometry.priorPoseidonStart application)
      (PiRLCRetainedGeometry.priorPoseidonFits
        (PiRLCPoseidonGeometry.prefixGeometry pilotGeometry))
      raw.assignment raw.retainedSource :=
    (PerApplicationCanonicalEncodes.retainedEncodes raw).priorPoseidon
  have privateColumn : ((PilotPoseidonPlan.priorSchedule application).block.source
      (PoseidonRetainedFamily.slot (PilotPoseidonPlan.priorSchedule application)
        invocation row)).val < PiRLCProductPlan.basePackage.layout.constantColumn := by
    rw [priorSlot_source]
    exact Nat.lt_of_lt_of_le
      (Nat.add_lt_add_left (PoseidonRetainedSlots.localOutput row).isLt _)
      (PoseidonRetainedBlock.priorWitnessStart_bound invocation)
  calc
    _ = raw.retainedSource ((PilotPoseidonPlan.priorSchedule application).block.source
        (PoseidonRetainedFamily.slot (PilotPoseidonPlan.priorSchedule application)
          invocation row)) :=
      PoseidonRetainedFamily.form_eval (PilotPoseidonPlan.priorSchedule application)
        (PiRLCRetainedGeometry.priorPoseidonStart application)
        (PiRLCRetainedGeometry.priorPoseidonFits
          (PiRLCPoseidonGeometry.prefixGeometry pilotGeometry))
        raw.assignment raw.retainedSource encoding invocation row
    _ = target ((PilotPoseidonPlan.priorSchedule application).block.source
        (PoseidonRetainedFamily.slot (PilotPoseidonPlan.priorSchedule application)
          invocation row)).val :=
      retainedSource_private application target applicationPrivate raw baseEq _ privateColumn
    _ = target (PoseidonRetainedBlock.priorWitnessStart invocation +
        (PoseidonRetainedSlots.localOutput row).val) :=
      congrArg target (priorSlot_source application invocation row)
    _ = _ := rfl


private theorem output_form
    (invocation : Fin PilotPoseidonPlan.invocationCount)
    (row : Fin PoseidonRetainedSlots.rows.length) :
    (PoseidonRetainedFamily.form (PilotPoseidonPlan.outputSchedule application)
      (PiRLCRetainedGeometry.outputPoseidonStart application)
      (PiRLCRetainedGeometry.outputPoseidonFits
        (PiRLCPoseidonGeometry.prefixGeometry pilotGeometry)) invocation row).eval raw.assignment =
      target (invocationLocalStart (PilotData.circuitPackage ()) Data.outputChain invocation.val +
        (PoseidonRetainedSlots.localOutput row).val) := by
  have encoding : (PilotPoseidonPlan.outputSchedule application).block.EncodesAt
      (PiRLCRetainedGeometry.outputPoseidonStart application)
      (PiRLCRetainedGeometry.outputPoseidonFits
        (PiRLCPoseidonGeometry.prefixGeometry pilotGeometry))
      raw.assignment raw.retainedSource :=
    (PerApplicationCanonicalEncodes.retainedEncodes raw).outputPoseidon
  have privateColumn : ((PilotPoseidonPlan.outputSchedule application).block.source
      (PoseidonRetainedFamily.slot (PilotPoseidonPlan.outputSchedule application)
        invocation row)).val < PiRLCProductPlan.basePackage.layout.constantColumn := by
    rw [outputSlot_source]
    exact Nat.lt_of_lt_of_le
      (Nat.add_lt_add_left (PoseidonRetainedSlots.localOutput row).isLt _)
      (PoseidonRetainedBlock.outputWitnessStart_bound invocation)
  calc
    _ = raw.retainedSource ((PilotPoseidonPlan.outputSchedule application).block.source
        (PoseidonRetainedFamily.slot (PilotPoseidonPlan.outputSchedule application)
          invocation row)) :=
      PoseidonRetainedFamily.form_eval (PilotPoseidonPlan.outputSchedule application)
        (PiRLCRetainedGeometry.outputPoseidonStart application)
        (PiRLCRetainedGeometry.outputPoseidonFits
          (PiRLCPoseidonGeometry.prefixGeometry pilotGeometry))
        raw.assignment raw.retainedSource encoding invocation row
    _ = target ((PilotPoseidonPlan.outputSchedule application).block.source
        (PoseidonRetainedFamily.slot (PilotPoseidonPlan.outputSchedule application)
          invocation row)).val :=
      retainedSource_private application target applicationPrivate raw baseEq _ privateColumn
    _ = target (PoseidonRetainedBlock.outputWitnessStart invocation +
        (PoseidonRetainedSlots.localOutput row).val) :=
      congrArg target (outputSlot_source application invocation row)
    _ = _ := rfl


private theorem priorInput_form (slot : Fin Data.priorChain.inputLength) :
    ((PiRLCPoseidonGeometry.priorInputBlock application).form
      (PiRLCPoseidonGeometry.priorInputStart application)
      (PiRLCPoseidonGeometry.priorInputFits pilotGeometry) slot).eval raw.assignment =
      target (Data.priorChain.inputStart + slot.val) := by
  have encoding := (PerApplicationCanonicalEncodes.runningPrefixEncodes raw).prior.pilotPriorInput
  apply form_eval_private application target applicationPrivate raw baseEq
    (PiRLCPoseidonGeometry.priorInputBlock application) _ _ encoding slot
  exact Nat.lt_of_lt_of_le (Nat.add_lt_add_left slot.isLt _)
    PoseidonInputRetainedBlock.priorInputEnd

private theorem outputInput_form (slot : Fin Data.outputChain.inputLength) :
    ((PiRLCPoseidonGeometry.outputInputBlock application).form
      (PiRLCPoseidonGeometry.outputInputStart application)
      (PiRLCPoseidonGeometry.outputInputFits pilotGeometry) slot).eval raw.assignment =
      target (Data.outputChain.inputStart + slot.val) := by
  have encoding := (PerApplicationCanonicalEncodes.runningPrefixEncodes raw).prior.pilotOutputInput
  apply form_eval_private application target applicationPrivate raw baseEq
    (PiRLCPoseidonGeometry.outputInputBlock application) _ _ encoding slot
  exact Nat.lt_of_lt_of_le (Nat.add_lt_add_left slot.isLt _)
    PoseidonInputRetainedBlock.outputInputEnd

private theorem prior_output
    (hashRows : HashChainHolds (Data.circuitPackage ()) Data.priorChain target)
    (invocation : Fin PilotPoseidonPlan.invocationCount) :
    SparseLayer.evalState raw.assignment
      ((PilotPoseidonPlan.priorInterface pilotGeometry).output invocation) =
      fun lane : Fin 8 => target
        (invocationLocalStart (PilotData.circuitPackage ()) Data.priorChain invocation.val +
          584 + lane.val) := by
  have equations := template_finalLayer Data.priorChain invocation.val target
    (hashRows invocation.val (Nat.le_of_lt_succ invocation.isLt))
  rw [equations]
  change SparseLayer.evalState raw.assignment
    (SparseLayer.external fun lane =>
      PoseidonRetainedFamily.form (PilotPoseidonPlan.priorSchedule application)
        (PiRLCRetainedGeometry.priorPoseidonStart application)
        (PiRLCRetainedGeometry.priorPoseidonFits
          (PiRLCPoseidonGeometry.prefixGeometry pilotGeometry))
        invocation (PoseidonRetainedSlots.finalRow lane)) = _
  funext lane
  simp only [SparseLayer.evalState, SparseLayer.eval_external]
  apply congrFun (congrArg Layer.externalF _) lane
  funext current
  exact prior_form application target applicationPrivate raw baseEq invocation _

private theorem output_output
    (hashRows : HashChainHolds (Data.circuitPackage ()) Data.outputChain target)
    (invocation : Fin PilotPoseidonPlan.invocationCount) :
    SparseLayer.evalState raw.assignment
      ((PilotPoseidonPlan.outputInterface pilotGeometry).output invocation) =
      fun lane : Fin 8 => target
        (invocationLocalStart (PilotData.circuitPackage ()) Data.outputChain invocation.val +
          584 + lane.val) := by
  have equations := template_finalLayer Data.outputChain invocation.val target
    (hashRows invocation.val (Nat.le_of_lt_succ invocation.isLt))
  rw [equations]
  change SparseLayer.evalState raw.assignment
    (SparseLayer.external fun lane =>
      PoseidonRetainedFamily.form (PilotPoseidonPlan.outputSchedule application)
        (PiRLCRetainedGeometry.outputPoseidonStart application)
        (PiRLCRetainedGeometry.outputPoseidonFits
          (PiRLCPoseidonGeometry.prefixGeometry pilotGeometry))
        invocation (PoseidonRetainedSlots.finalRow lane)) = _
  funext lane
  simp only [SparseLayer.evalState, SparseLayer.eval_external]
  apply congrFun (congrArg Layer.externalF _) lane
  funext current
  exact output_form application target applicationPrivate raw baseEq invocation _

omit baseEq in
private theorem previous_eval {sourceWidth count logicalWidth : Nat}
    (schedule : PoseidonRetainedFamily.Schedule sourceWidth count)
    (start : Nat) (fits : start + schedule.block.coordinateCount ≤ logicalWidth)
    (assignment : Assignment F logicalWidth) (chain : HashChain) (sourceTarget : Env)
    (outputs : ∀ invocation : Fin count,
      SparseLayer.evalState assignment (PoseidonRetainedFamily.outputState schedule start fits invocation) =
        fun lane : Fin 8 => sourceTarget
          (invocationLocalStart (PilotData.circuitPackage ()) chain invocation.val + 584 + lane.val))
    (invocation : Fin count) (lane : Fin 8) :
    (PilotPoseidonPlan.previousOutput schedule start fits invocation lane).eval assignment =
      if invocation.val = 0 then 0 else sourceTarget
        (invocationLocalStart (PilotData.circuitPackage ()) chain (invocation.val - 1) +
          584 + lane.val) := by
  by_cases first : invocation.val = 0
  · simp [PilotPoseidonPlan.previousOutput, first]
  · simp only [PilotPoseidonPlan.previousOutput, dif_neg first, if_neg first]
    exact congrFun (outputs _) lane

private theorem prior_input
    (hashRows : HashChainHolds (Data.circuitPackage ()) Data.priorChain target)
    (invocation : Fin PilotPoseidonPlan.invocationCount) :
    SparseLayer.evalState raw.assignment (PilotPoseidonPlan.priorInputState pilotGeometry invocation) =
      Layer.evalState (templateEnv Data.priorChain invocation.val target)
        PoseidonScheduleTrace.canonicalState := by
  have one : raw.assignment (PiRLCPoseidonGeometry.oneColumn pilotGeometry) = 1 :=
    PerApplicationCanonicalAssignment.assignment_one raw
  funext lane
  have previous := previous_eval (PilotPoseidonPlan.priorSchedule application)
    (PiRLCRetainedGeometry.priorPoseidonStart application)
    (PiRLCRetainedGeometry.priorPoseidonFits (PiRLCPoseidonGeometry.prefixGeometry pilotGeometry))
    raw.assignment Data.priorChain target
    (prior_output application target applicationPrivate raw baseEq hashRows) invocation lane
  change (PilotPoseidonPlan.priorInputState pilotGeometry invocation lane).eval raw.assignment =
    templateEnv Data.priorChain invocation.val target lane.val
  rw [templateEnv_input]
  unfold PilotPoseidonPlan.priorInputState invocationInput
  change _ = (if invocation.val < Data.priorChain.absorbCount then
    if lane.val < Poseidon2.rate ∧ invocation.val * Poseidon2.rate + lane.val < Data.priorChain.inputLength then
      R1CS.LinearCombination.add
        (if invocation.val = 0 then R1CS.LinearCombination.zero else R1CS.LinearCombination.ofVar
          (invocationLocalStart (PilotData.circuitPackage ()) Data.priorChain (invocation.val - 1) + 584 + lane.val))
        (R1CS.LinearCombination.ofVar (Data.priorChain.inputStart + (invocation.val * Poseidon2.rate + lane.val)))
    else
      (if invocation.val = 0 then R1CS.LinearCombination.zero else R1CS.LinearCombination.ofVar
        (invocationLocalStart (PilotData.circuitPackage ()) Data.priorChain (invocation.val - 1) + 584 + lane.val))
    else if lane.val = 0 then R1CS.LinearCombination.add
      (if invocation.val = 0 then R1CS.LinearCombination.zero else R1CS.LinearCombination.ofVar
        (invocationLocalStart (PilotData.circuitPackage ()) Data.priorChain (invocation.val - 1) + 584 + lane.val))
      R1CS.LinearCombination.one
    else (if invocation.val = 0 then R1CS.LinearCombination.zero else R1CS.LinearCombination.ofVar
      (invocationLocalStart (PilotData.circuitPackage ()) Data.priorChain (invocation.val - 1) + 584 + lane.val))).eval target
  split_ifs <;> simp_all only [and_true, true_and, and_false, false_and, not_true_eq_false,
    ite_true, ite_false, dite_true, dite_false,
    SparseForm.add_eval, SparseForm.singleton_eval, one,
    mul_one, one_mul, previous,
    (priorInput_form application target applicationPrivate raw baseEq),
    R1CS.LinearCombination.eval_add, R1CS.LinearCombination.eval_ofVar,
    R1CS.LinearCombination.eval_zero, R1CS.LinearCombination.eval_one, Nat.add_assoc]

private theorem output_input
    (hashRows : HashChainHolds (Data.circuitPackage ()) Data.outputChain target)
    (invocation : Fin PilotPoseidonPlan.invocationCount) :
    SparseLayer.evalState raw.assignment (PilotPoseidonPlan.outputInputState pilotGeometry invocation) =
      Layer.evalState (templateEnv Data.outputChain invocation.val target)
        PoseidonScheduleTrace.canonicalState := by
  have one : raw.assignment (PiRLCPoseidonGeometry.oneColumn pilotGeometry) = 1 :=
    PerApplicationCanonicalAssignment.assignment_one raw
  funext lane
  have previous := previous_eval (PilotPoseidonPlan.outputSchedule application)
    (PiRLCRetainedGeometry.outputPoseidonStart application)
    (PiRLCRetainedGeometry.outputPoseidonFits (PiRLCPoseidonGeometry.prefixGeometry pilotGeometry))
    raw.assignment Data.outputChain target
    (output_output application target applicationPrivate raw baseEq hashRows) invocation lane
  change (PilotPoseidonPlan.outputInputState pilotGeometry invocation lane).eval raw.assignment =
    templateEnv Data.outputChain invocation.val target lane.val
  rw [templateEnv_input]
  unfold PilotPoseidonPlan.outputInputState invocationInput
  change _ = (if invocation.val < Data.outputChain.absorbCount then
    if lane.val < Poseidon2.rate ∧ invocation.val * Poseidon2.rate + lane.val < Data.outputChain.inputLength then
      R1CS.LinearCombination.add
        (if invocation.val = 0 then R1CS.LinearCombination.zero else R1CS.LinearCombination.ofVar
          (invocationLocalStart (PilotData.circuitPackage ()) Data.outputChain (invocation.val - 1) + 584 + lane.val))
        (R1CS.LinearCombination.ofVar (Data.outputChain.inputStart + (invocation.val * Poseidon2.rate + lane.val)))
    else
      (if invocation.val = 0 then R1CS.LinearCombination.zero else R1CS.LinearCombination.ofVar
        (invocationLocalStart (PilotData.circuitPackage ()) Data.outputChain (invocation.val - 1) + 584 + lane.val))
    else if lane.val = 0 then R1CS.LinearCombination.add
      (if invocation.val = 0 then R1CS.LinearCombination.zero else R1CS.LinearCombination.ofVar
        (invocationLocalStart (PilotData.circuitPackage ()) Data.outputChain (invocation.val - 1) + 584 + lane.val))
      R1CS.LinearCombination.one
    else (if invocation.val = 0 then R1CS.LinearCombination.zero else R1CS.LinearCombination.ofVar
      (invocationLocalStart (PilotData.circuitPackage ()) Data.outputChain (invocation.val - 1) + 584 + lane.val))).eval target
  split_ifs <;> simp_all only [and_true, true_and, and_false, false_and, not_true_eq_false,
    ite_true, ite_false, dite_true, dite_false,
    SparseForm.add_eval, SparseForm.singleton_eval, one,
    mul_one, one_mul, previous,
    (outputInput_form application target applicationPrivate raw baseEq),
    R1CS.LinearCombination.eval_add, R1CS.LinearCombination.eval_ofVar,
    R1CS.LinearCombination.eval_zero, R1CS.LinearCombination.eval_one, Nat.add_assoc]

/-- Actual prior and output hash-template rows imply every compact direct
pilot row on the exact canonical source copy. All retained S-box equations,
input absorption values, and previous-output links are derived from those
rows and the existing source projection. No hash-output equality is assumed. -/
private theorem rowsZero_of_hashRows
    (priorRows : HashChainHolds (Data.circuitPackage ()) Data.priorChain target)
    (outputRows : HashChainHolds (Data.circuitPackage ()) Data.outputChain target) :
    (PilotPoseidonPlan.plan pilotGeometry).RowsZero raw.assignment := by
  have one : raw.assignment (PiRLCPoseidonGeometry.oneColumn pilotGeometry) = 1 :=
    PerApplicationCanonicalAssignment.assignment_one raw
  apply PilotPoseidonPlan.equations_imply_rowsZero pilotGeometry raw.assignment one
  · intro invocation
    refine PoseidonSboxSourceCompleteness.equations_of_sourceRows
      (PoseidonSboxFamilyPlan.invocationInterface
        (PilotPoseidonPlan.priorInterface pilotGeometry) invocation)
      raw.assignment (templateEnv Data.priorChain invocation.val target) ?_ ?_ ?_ ?_
    · exact one
    · exact prior_input application target applicationPrivate raw baseEq priorRows invocation
    · intro row
      rw [PoseidonRetainedSlots.output_eq_input_add_local]
      simp only [PoseidonScheduleTrace.inputCount]
      rw [templateEnv_local]
      exact prior_form application target applicationPrivate raw baseEq invocation row
    · exact template_constraints Data.priorChain invocation.val target
        (priorRows invocation.val (Nat.le_of_lt_succ invocation.isLt))
  · intro invocation
    refine PoseidonSboxSourceCompleteness.equations_of_sourceRows
      (PoseidonSboxFamilyPlan.invocationInterface
        (PilotPoseidonPlan.outputInterface pilotGeometry) invocation)
      raw.assignment (templateEnv Data.outputChain invocation.val target) ?_ ?_ ?_ ?_
    · exact one
    · exact output_input application target applicationPrivate raw baseEq outputRows invocation
    · intro row
      rw [PoseidonRetainedSlots.output_eq_input_add_local]
      simp only [PoseidonScheduleTrace.inputCount]
      rw [templateEnv_local]
      exact output_form application target applicationPrivate raw baseEq invocation row
    · exact template_constraints Data.outputChain invocation.val target
        (outputRows invocation.val (Nat.le_of_lt_succ invocation.isLt))

end Copied

/-- Actual physical pilot hash rows give the exact compact pilot row family
on the canonical source copy. Source-slot and S-box equations are derived;
the caller supplies no equality between assignment environments. -/
theorem rowsZero_of_completed_hashRows
    (application : Lifecycle.Stage1.Application.Program) (target : Env)
    (applicationPrivate : Fin (PerApplicationPackage.addedPrivateColumnCount application) → F)
    (priorRows : HashChainHolds (Data.circuitPackage ()) Data.priorChain target)
    (outputRows : HashChainHolds (Data.circuitPackage ()) Data.outputChain target) :
    (PilotPoseidonPlan.plan (DirectPrefixPlan.pilotGeometry
      (PerApplicationCanonicalEncodes.poseidonGeometry application))).RowsZero
      (canonicalRawValues application
        (PerApplicationSourceAssignment.ofCompleted application target applicationPrivate)).assignment := by
  exact rowsZero_of_hashRows application target applicationPrivate
    (canonicalRawValues application
      (PerApplicationSourceAssignment.ofCompleted application target applicationPrivate))
    rfl priorRows outputRows

end NightstreamFPrime.Export.Stage1.PilotPoseidonCompleteness
