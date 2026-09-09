import NightstreamFPrime.Export.Stage1.PilotOrdinaryDirectPlan
import NightstreamFPrime.Export.Stage1.PilotPoseidonPlan

/-!
Owns the value-preservation bridge for the two direct pilot Poseidon2 chains.
It connects retained preimage forms to the exact lifted pilot environment and
then derives the canonical sponge recurrence from the direct permutation rows.

This module does not claim that the unused legacy permutation locals satisfy
the old expanded template rows.
-/

namespace NightstreamFPrime.Export.Stage1.PilotPoseidonPreservation

open NightstreamFPrime.Gadgets.Poseidon2
open NightstreamFPrime.Export.Package
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Layout.Stage1
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra

def sourceAssignment (program : Lifecycle.Stage1.Application.Program)
    (base : Fin (PiRLCProductPlan.baseSourceWidth program) → F)
    (groupValue : Fin PiRLCProductSchedule.invocationCount → Fin 33 → F)
    (products : Fin PiRLCFirst54DirectSchedule.candidateCount → F) :
    Fin (PiRLCRetainedGeometry.sourceWidth program) → F :=
  PiRLCRetainedPreservation.sourceAssignment program base groupValue products

structure Encoding {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat}
    (geometry : PiRLCPoseidonGeometry.Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth)
    (base : Fin (PiRLCProductPlan.baseSourceWidth program) → F)
    (groupValue : Fin PiRLCProductSchedule.invocationCount → Fin 33 → F)
    (products : Fin PiRLCFirst54DirectSchedule.candidateCount → F) : Prop where
  priorInput : (PiRLCPoseidonGeometry.priorInputBlock program).EncodesAt
    (PiRLCPoseidonGeometry.priorInputStart program)
    (PiRLCPoseidonGeometry.priorInputFits geometry) assignment
    (sourceAssignment program base groupValue products)
  outputInput : (PiRLCPoseidonGeometry.outputInputBlock program).EncodesAt
    (PiRLCPoseidonGeometry.outputInputStart program)
    (PiRLCPoseidonGeometry.outputInputFits geometry) assignment
    (sourceAssignment program base groupValue products)

private theorem constant_le_total :
    PiRLCProductPlan.basePackage.layout.constantColumn ≤
      PiRLCProductPlan.basePackage.layout.totalColumnCount := by
  have constant : PiRLCProductPlan.basePackage.layout.constantColumn =
      29336446 := Package.circuitPackage_layout_values.2.2.1
  have total : PiRLCProductPlan.basePackage.layout.totalColumnCount =
      29336725 := Package.circuitPackage_layout_values.2.2.2.2
  rw [constant, total]
  omega

/-- A retained source that names one private physical package column reads
the same final per-application package value as the transition environment. -/
private theorem sourceAssignment_privatePhysical
    (program : Lifecycle.Stage1.Application.Program)
    (base : Fin (PiRLCProductPlan.baseSourceWidth program) → F)
    (groupValue : Fin PiRLCProductSchedule.invocationCount → Fin 33 → F)
    (products : Fin PiRLCFirst54DirectSchedule.candidateCount → F)
    (source : Fin (PiRLCRetainedGeometry.sourceWidth program))
    (column : Nat)
    (sourceEq : source.val = column)
    (privateBound : column <
      PiRLCProductPlan.basePackage.layout.constantColumn)
    (beforeTranscript : column < PiCCSTranscriptReadout.phaseStart + 584) :
    sourceAssignment program base groupValue products source =
      RunningTransitionDirectPlan.transitionEnv program base column := by
  have totalBound : column <
      PiRLCProductPlan.basePackage.layout.totalColumnCount :=
    Nat.lt_of_lt_of_le privateBound constant_le_total
  let shifted := PiRLCProductPlan.shiftedPackageColumn program column totalBound
  have sourceColumn : source =
      PiRLCRetainedPreservation.baseSourceColumn program shifted := by
    apply Fin.ext
    rw [sourceEq]
    simp [shifted, PiRLCProductPlan.shiftedPackageColumn,
      PiRLCRetainedPreservation.baseSourceColumn,
      PiRLCFirst54DirectPlan.prefixColumn,
      ProductRetainedBlock.baseColumn, FieldSuffixBlock.baseColumn,
      PerApplicationPackage.shiftColumn_private program column privateBound]
  rw [sourceColumn]
  unfold sourceAssignment
  rw [PiRLCRetainedPreservation.sourceAssignment_base]
  have viewEq : RunningTransitionDirectPlan.transitionEnv program base column =
      RunningTransitionDirectPlan.packageEnv program base column := by
    apply PermutationOutput.Readout.env_of_decode_none
    unfold PermutationOutput.Readout.decode
    rw [dif_neg (Nat.not_le.mpr beforeTranscript)]
  rw [viewEq]
  exact (SourceCompiler.sourceEnv_at base shifted).symm

private theorem priorInputPrivate
    (index : Fin Data.priorChain.inputLength) :
    Data.priorChain.inputStart + index.val <
      PiRLCProductPlan.basePackage.layout.constantColumn := by
  have endBound := PoseidonInputRetainedBlock.priorInputEnd
  have indexBound := index.isLt
  exact Nat.lt_of_lt_of_le (Nat.add_lt_add_left indexBound _) endBound

private theorem outputInputPrivate
    (index : Fin Data.outputChain.inputLength) :
    Data.outputChain.inputStart + index.val <
      PiRLCProductPlan.basePackage.layout.constantColumn := by
  have endBound := PoseidonInputRetainedBlock.outputInputEnd
  have indexBound := index.isLt
  exact Nat.lt_of_lt_of_le (Nat.add_lt_add_left indexBound _) endBound

private theorem priorInputLift (index : Fin Data.priorChain.inputLength) :
    Data.priorChain.inputStart + index.val =
      Spartan.liftPilotColumn
        (PilotData.priorChain.inputStart + index.val) := by
  have indexBound := index.isLt
  change index.val < 49393 at indexBound
  have lifted := Spartan.liftPilotColumn_add_of_input
    PilotData.priorChain.inputStart index.val (by
      change 0 + index.val < 98786
      omega)
  unfold Data.priorChain Data.liftPilotChain
  exact lifted.symm

private theorem outputInputLift (index : Fin Data.outputChain.inputLength) :
    Data.outputChain.inputStart + index.val =
      Spartan.liftPilotColumn
        (PilotData.outputChain.inputStart + index.val) := by
  have indexBound := index.isLt
  change index.val < 49393 at indexBound
  have lifted := Spartan.liftPilotColumn_add_of_input
    PilotData.outputChain.inputStart index.val (by
      change 49393 + index.val < 98786
      omega)
  unfold Data.outputChain Data.liftPilotChain
  exact lifted.symm

/-- Every direct prior-chain input form is the exact canonical pilot word. -/
theorem priorInputForm_eval
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : PiRLCPoseidonGeometry.Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth)
    (base : Fin (PiRLCProductPlan.baseSourceWidth program) → F)
    (groupValue : Fin PiRLCProductSchedule.invocationCount → Fin 33 → F)
    (products : Fin PiRLCFirst54DirectSchedule.candidateCount → F)
    (encoding : Encoding geometry assignment base groupValue products)
    (index : Fin Data.priorChain.inputLength) :
    ((PiRLCPoseidonGeometry.priorInputBlock program).form
      (PiRLCPoseidonGeometry.priorInputStart program)
      (PiRLCPoseidonGeometry.priorInputFits geometry) index).eval assignment =
      PilotOrdinaryDirectPlan.pilotEnv program base
        (PilotData.priorChain.inputStart + index.val) := by
  rw [LowNormBlock.Block.form_eval _ _ _ assignment _ encoding.priorInput]
  rw [sourceAssignment_privatePhysical program base groupValue products
    _ (Data.priorChain.inputStart + index.val) (by rfl)
    (priorInputPrivate index) (by
      have bound : index.val < 49393 := index.isLt
      change 0 + index.val < PiCCSTranscriptReadout.phaseStart + 584
      rw [PiCCSTranscriptReadout.phaseStart_eq]
      omega)]
  unfold PilotOrdinaryDirectPlan.pilotEnv
  rw [priorInputLift index]

/-- Every direct output-chain input form is the exact canonical pilot word. -/
theorem outputInputForm_eval
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : PiRLCPoseidonGeometry.Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth)
    (base : Fin (PiRLCProductPlan.baseSourceWidth program) → F)
    (groupValue : Fin PiRLCProductSchedule.invocationCount → Fin 33 → F)
    (products : Fin PiRLCFirst54DirectSchedule.candidateCount → F)
    (encoding : Encoding geometry assignment base groupValue products)
    (index : Fin Data.outputChain.inputLength) :
    ((PiRLCPoseidonGeometry.outputInputBlock program).form
      (PiRLCPoseidonGeometry.outputInputStart program)
      (PiRLCPoseidonGeometry.outputInputFits geometry) index).eval assignment =
      PilotOrdinaryDirectPlan.pilotEnv program base
        (PilotData.outputChain.inputStart + index.val) := by
  rw [LowNormBlock.Block.form_eval _ _ _ assignment _ encoding.outputInput]
  rw [sourceAssignment_privatePhysical program base groupValue products
    _ (Data.outputChain.inputStart + index.val) (by rfl)
    (outputInputPrivate index) (by
      have bound : index.val < 49393 := index.isLt
      change 49393 + index.val < PiCCSTranscriptReadout.phaseStart + 584
      rw [PiCCSTranscriptReadout.phaseStart_eq]
      omega)]
  unfold PilotOrdinaryDirectPlan.pilotEnv
  rw [outputInputLift index]

def priorOutputValue {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat}
    (geometry : PiRLCPoseidonGeometry.Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth)
    (invocation : Fin PilotPoseidonPlan.invocationCount) : Layer.FState :=
  SparseLayer.evalState assignment
    ((PilotPoseidonPlan.priorInterface geometry).output invocation)

def outputOutputValue {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat}
    (geometry : PiRLCPoseidonGeometry.Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth)
    (invocation : Fin PilotPoseidonPlan.invocationCount) : Layer.FState :=
  SparseLayer.evalState assignment
    ((PilotPoseidonPlan.outputInterface geometry).output invocation)

def previousValue
    (output : Fin PilotPoseidonPlan.invocationCount → Layer.FState)
    (invocation : Fin PilotPoseidonPlan.invocationCount) : Layer.FState :=
  if first : invocation.val = 0 then fun _ => 0
  else output ⟨invocation.val - 1, by
    have invocationBound := invocation.isLt
    omega⟩

def canonicalInput (chain : HashChain) (env : Circuit.Env)
    (output : Fin PilotPoseidonPlan.invocationCount → Layer.FState)
    (invocation : Fin PilotPoseidonPlan.invocationCount) : Layer.FState :=
  let previous := previousValue output invocation
  if invocation.val < chain.absorbCount then
    fun lane =>
      if rateLane : lane.val < Spec.Poseidon2.rate then
        let offset := invocation.val * Spec.Poseidon2.rate + lane.val
        if offset < chain.inputLength then
          previous lane + env (chain.inputStart + offset)
        else
          previous lane
      else
        previous lane
  else
    fun lane => if lane.val = 0 then previous lane + 1 else previous lane

private theorem priorPreviousOutput_eval
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : PiRLCPoseidonGeometry.Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth)
    (invocation : Fin PilotPoseidonPlan.invocationCount) :
    SparseLayer.evalState assignment
        (PilotPoseidonPlan.previousOutput
          (PilotPoseidonPlan.priorSchedule program)
          (PiRLCRetainedGeometry.priorPoseidonStart program)
          (PiRLCRetainedGeometry.priorPoseidonFits
            (PiRLCPoseidonGeometry.prefixGeometry geometry)) invocation) =
      previousValue (priorOutputValue geometry assignment) invocation := by
  funext lane
  unfold PilotPoseidonPlan.previousOutput previousValue priorOutputValue
  split
  · exact SparseForm.empty_eval assignment
  · rfl

private theorem outputPreviousOutput_eval
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : PiRLCPoseidonGeometry.Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth)
    (invocation : Fin PilotPoseidonPlan.invocationCount) :
    SparseLayer.evalState assignment
        (PilotPoseidonPlan.previousOutput
          (PilotPoseidonPlan.outputSchedule program)
          (PiRLCRetainedGeometry.outputPoseidonStart program)
          (PiRLCRetainedGeometry.outputPoseidonFits
            (PiRLCPoseidonGeometry.prefixGeometry geometry)) invocation) =
      previousValue (outputOutputValue geometry assignment) invocation := by
  funext lane
  unfold PilotPoseidonPlan.previousOutput previousValue outputOutputValue
  split
  · exact SparseForm.empty_eval assignment
  · rfl

theorem priorInputState_eval
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : PiRLCPoseidonGeometry.Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth)
    (env : Circuit.Env)
    (one : assignment (PiRLCPoseidonGeometry.oneColumn geometry) = 1)
    (inputValues : ∀ index : Fin Data.priorChain.inputLength,
      ((PiRLCPoseidonGeometry.priorInputBlock program).form
        (PiRLCPoseidonGeometry.priorInputStart program)
        (PiRLCPoseidonGeometry.priorInputFits geometry) index).eval assignment =
          env (PilotData.priorChain.inputStart + index.val))
    (invocation : Fin PilotPoseidonPlan.invocationCount) :
    SparseLayer.evalState assignment
        (PilotPoseidonPlan.priorInputState geometry invocation) =
      canonicalInput PilotData.priorChain
        env
        (priorOutputValue geometry assignment) invocation := by
  funext lane
  change (PilotPoseidonPlan.priorInputState geometry invocation lane).eval
      assignment =
    canonicalInput PilotData.priorChain
      env
      (priorOutputValue geometry assignment) invocation lane
  unfold PilotPoseidonPlan.priorInputState canonicalInput
  by_cases absorbing : invocation.val < Data.priorChain.absorbCount
  · rw [if_pos absorbing]
    have pilotAbsorbing : invocation.val < PilotData.priorChain.absorbCount := by
      simpa [Data.priorChain, Data.liftPilotChain] using absorbing
    rw [if_pos pilotAbsorbing]
    by_cases rateLane : lane.val < Spec.Poseidon2.rate
    · rw [dif_pos rateLane, dif_pos rateLane]
      let offset := invocation.val * Spec.Poseidon2.rate + lane.val
      by_cases present : offset < Data.priorChain.inputLength
      · rw [dif_pos present]
        have pilotPresent : offset < PilotData.priorChain.inputLength := by
          simpa [Data.priorChain, Data.liftPilotChain] using present
        rw [if_pos pilotPresent]
        simp only [SparseLayer.evalState, SparseForm.add_eval]
        rw [show
            (PilotPoseidonPlan.previousOutput
              (PilotPoseidonPlan.priorSchedule program)
              (PiRLCRetainedGeometry.priorPoseidonStart program)
              (PiRLCRetainedGeometry.priorPoseidonFits
                (PiRLCPoseidonGeometry.prefixGeometry geometry)) invocation lane).eval
                assignment =
              previousValue (priorOutputValue geometry assignment)
                invocation lane by
          exact congrFun (priorPreviousOutput_eval geometry assignment invocation)
            lane]
        apply congrArg (fun value =>
          previousValue (priorOutputValue geometry assignment)
            invocation lane + value)
        simpa only [offset] using
          inputValues ⟨offset, present⟩
      · rw [dif_neg present]
        have pilotAbsent : ¬offset < PilotData.priorChain.inputLength := by
          simpa [Data.priorChain, Data.liftPilotChain] using present
        rw [if_neg pilotAbsent]
        exact congrFun (priorPreviousOutput_eval geometry assignment invocation)
          lane
    · rw [dif_neg rateLane, dif_neg rateLane]
      exact congrFun (priorPreviousOutput_eval geometry assignment invocation)
        lane
  · rw [if_neg absorbing]
    have pilotFinal : ¬invocation.val < PilotData.priorChain.absorbCount := by
      simpa [Data.priorChain, Data.liftPilotChain] using absorbing
    rw [if_neg pilotFinal]
    by_cases zeroLane : lane.val = 0
    · rw [if_pos zeroLane, if_pos zeroLane]
      simp only [SparseLayer.evalState, SparseForm.add_eval,
        SparseForm.singleton_eval, one, mul_one]
      rw [show
          (PilotPoseidonPlan.previousOutput
            (PilotPoseidonPlan.priorSchedule program)
            (PiRLCRetainedGeometry.priorPoseidonStart program)
            (PiRLCRetainedGeometry.priorPoseidonFits
              (PiRLCPoseidonGeometry.prefixGeometry geometry)) invocation lane).eval
              assignment =
            previousValue (priorOutputValue geometry assignment)
              invocation lane by
        exact congrFun (priorPreviousOutput_eval geometry assignment invocation)
          lane]
    · rw [if_neg zeroLane, if_neg zeroLane]
      exact congrFun (priorPreviousOutput_eval geometry assignment invocation)
        lane

theorem outputInputState_eval
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : PiRLCPoseidonGeometry.Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth)
    (env : Circuit.Env)
    (one : assignment (PiRLCPoseidonGeometry.oneColumn geometry) = 1)
    (inputValues : ∀ index : Fin Data.outputChain.inputLength,
      ((PiRLCPoseidonGeometry.outputInputBlock program).form
        (PiRLCPoseidonGeometry.outputInputStart program)
        (PiRLCPoseidonGeometry.outputInputFits geometry) index).eval assignment =
          env (PilotData.outputChain.inputStart + index.val))
    (invocation : Fin PilotPoseidonPlan.invocationCount) :
    SparseLayer.evalState assignment
        (PilotPoseidonPlan.outputInputState geometry invocation) =
      canonicalInput PilotData.outputChain
        env
        (outputOutputValue geometry assignment) invocation := by
  funext lane
  change (PilotPoseidonPlan.outputInputState geometry invocation lane).eval
      assignment =
    canonicalInput PilotData.outputChain
      env
      (outputOutputValue geometry assignment) invocation lane
  unfold PilotPoseidonPlan.outputInputState canonicalInput
  by_cases absorbing : invocation.val < Data.outputChain.absorbCount
  · rw [if_pos absorbing]
    have pilotAbsorbing : invocation.val < PilotData.outputChain.absorbCount := by
      simpa [Data.outputChain, Data.liftPilotChain] using absorbing
    rw [if_pos pilotAbsorbing]
    by_cases rateLane : lane.val < Spec.Poseidon2.rate
    · rw [dif_pos rateLane, dif_pos rateLane]
      let offset := invocation.val * Spec.Poseidon2.rate + lane.val
      by_cases present : offset < Data.outputChain.inputLength
      · rw [dif_pos present]
        have pilotPresent : offset < PilotData.outputChain.inputLength := by
          simpa [Data.outputChain, Data.liftPilotChain] using present
        rw [if_pos pilotPresent]
        simp only [SparseLayer.evalState, SparseForm.add_eval]
        rw [show
            (PilotPoseidonPlan.previousOutput
              (PilotPoseidonPlan.outputSchedule program)
              (PiRLCRetainedGeometry.outputPoseidonStart program)
              (PiRLCRetainedGeometry.outputPoseidonFits
                (PiRLCPoseidonGeometry.prefixGeometry geometry)) invocation lane).eval
                assignment =
              previousValue (outputOutputValue geometry assignment)
                invocation lane by
          exact congrFun (outputPreviousOutput_eval geometry assignment invocation)
            lane]
        apply congrArg (fun value =>
          previousValue (outputOutputValue geometry assignment)
            invocation lane + value)
        simpa only [offset] using
          inputValues ⟨offset, present⟩
      · rw [dif_neg present]
        have pilotAbsent : ¬offset < PilotData.outputChain.inputLength := by
          simpa [Data.outputChain, Data.liftPilotChain] using present
        rw [if_neg pilotAbsent]
        exact congrFun (outputPreviousOutput_eval geometry assignment invocation)
          lane
    · rw [dif_neg rateLane, dif_neg rateLane]
      exact congrFun (outputPreviousOutput_eval geometry assignment invocation)
        lane
  · rw [if_neg absorbing]
    have pilotFinal : ¬invocation.val < PilotData.outputChain.absorbCount := by
      simpa [Data.outputChain, Data.liftPilotChain] using absorbing
    rw [if_neg pilotFinal]
    by_cases zeroLane : lane.val = 0
    · rw [if_pos zeroLane, if_pos zeroLane]
      simp only [SparseLayer.evalState, SparseForm.add_eval,
        SparseForm.singleton_eval, one, mul_one]
      rw [show
          (PilotPoseidonPlan.previousOutput
            (PilotPoseidonPlan.outputSchedule program)
            (PiRLCRetainedGeometry.outputPoseidonStart program)
            (PiRLCRetainedGeometry.outputPoseidonFits
              (PiRLCPoseidonGeometry.prefixGeometry geometry)) invocation lane).eval
              assignment =
            previousValue (outputOutputValue geometry assignment)
              invocation lane by
        exact congrFun (outputPreviousOutput_eval geometry assignment invocation)
          lane]
    · rw [if_neg zeroLane, if_neg zeroLane]
      exact congrFun (outputPreviousOutput_eval geometry assignment invocation)
        lane

structure CanonicalSemantics (chain : HashChain) (env : Circuit.Env)
    (output : Fin PilotPoseidonPlan.invocationCount → Layer.FState) : Prop where
  invocation : ∀ current,
    List.ofFn (output current) =
      Spec.Poseidon2.permute
        (List.ofFn (canonicalInput chain env output current))

/-- Direct pilot permutation semantics use the exact canonical sponge inputs
for both hash chains. -/
theorem canonicalSemantics
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : PiRLCPoseidonGeometry.Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth)
    (env : Circuit.Env)
    (one : assignment (PiRLCPoseidonGeometry.oneColumn geometry) = 1)
    (priorValues : ∀ index : Fin Data.priorChain.inputLength,
      ((PiRLCPoseidonGeometry.priorInputBlock program).form
        (PiRLCPoseidonGeometry.priorInputStart program)
        (PiRLCPoseidonGeometry.priorInputFits geometry) index).eval assignment =
          env (PilotData.priorChain.inputStart + index.val))
    (outputValues : ∀ index : Fin Data.outputChain.inputLength,
      ((PiRLCPoseidonGeometry.outputInputBlock program).form
        (PiRLCPoseidonGeometry.outputInputStart program)
        (PiRLCPoseidonGeometry.outputInputFits geometry) index).eval assignment =
          env (PilotData.outputChain.inputStart + index.val))
    (semantics : PilotPoseidonPlan.Semantics geometry assignment) :
    CanonicalSemantics PilotData.priorChain
        env
        (priorOutputValue geometry assignment) ∧
      CanonicalSemantics PilotData.outputChain
        env
        (outputOutputValue geometry assignment) := by
  constructor
  · constructor
    intro current
    calc
      List.ofFn (priorOutputValue geometry assignment current) =
          Spec.Poseidon2.permute
            (List.ofFn (SparseLayer.evalState assignment
              ((PilotPoseidonPlan.priorInterface geometry).input current))) :=
        semantics.prior current
      _ = Spec.Poseidon2.permute
          (List.ofFn (canonicalInput PilotData.priorChain
            env
            (priorOutputValue geometry assignment) current)) := by
        apply congrArg Spec.Poseidon2.permute
        change List.ofFn (SparseLayer.evalState assignment
          (PilotPoseidonPlan.priorInputState geometry current)) = _
        exact congrArg List.ofFn
          (priorInputState_eval geometry assignment env one priorValues current)
  · constructor
    intro current
    calc
      List.ofFn (outputOutputValue geometry assignment current) =
          Spec.Poseidon2.permute
            (List.ofFn (SparseLayer.evalState assignment
              ((PilotPoseidonPlan.outputInterface geometry).input current))) :=
        semantics.output current
      _ = Spec.Poseidon2.permute
          (List.ofFn (canonicalInput PilotData.outputChain
            env
            (outputOutputValue geometry assignment) current)) := by
        apply congrArg Spec.Poseidon2.permute
        change List.ofFn (SparseLayer.evalState assignment
          (PilotPoseidonPlan.outputInputState geometry current)) = _
        exact congrArg List.ofFn
          (outputInputState_eval geometry assignment env one outputValues current)

theorem canonicalInput_absorb (chain : HashChain) (env : Circuit.Env)
    (output : Fin PilotPoseidonPlan.invocationCount → Layer.FState)
    (current : Fin PilotPoseidonPlan.invocationCount)
    (absorbing : current.val < chain.absorbCount) :
    canonicalInput chain env output current = fun lane =>
      previousValue output current lane +
        NightstreamFPrime.Export.Pilot.chainBlockState
          chain current.val env lane := by
  funext lane
  unfold canonicalInput NightstreamFPrime.Export.Pilot.chainBlockState
  rw [if_pos absorbing]
  by_cases rateLane : lane.val < Spec.Poseidon2.rate
  · rw [dif_pos rateLane]
    let offset := current.val * Spec.Poseidon2.rate + lane.val
    by_cases present : offset < chain.inputLength
    · rw [if_pos present]
      simp [PilotData.circuitPackage, PilotData.poseidonSchedule, rateLane,
        present, offset]
    · rw [if_neg present]
      simp [PilotData.circuitPackage, PilotData.poseidonSchedule, rateLane,
        present, offset]
  · rw [dif_neg rateLane]
    simp [PilotData.circuitPackage, PilotData.poseidonSchedule, rateLane]

theorem canonicalInput_final (chain : HashChain) (env : Circuit.Env)
    (output : Fin PilotPoseidonPlan.invocationCount → Layer.FState)
    (current : Fin PilotPoseidonPlan.invocationCount)
    (final : ¬current.val < chain.absorbCount) :
    canonicalInput chain env output current =
      Hash.padF (previousValue output current) := by
  funext lane
  unfold canonicalInput Hash.padF
  rw [if_neg final]

def invocationAt (chain : HashChain)
    (countEq : chain.absorbCount + 1 = PilotPoseidonPlan.invocationCount)
    (count : Nat) (bound : count ≤ chain.absorbCount) :
    Fin PilotPoseidonPlan.invocationCount :=
  ⟨count, by omega⟩

private theorem permute_eq_runF (state : Layer.FState) :
    Spec.Poseidon2.permute (List.ofFn state) =
      List.ofFn (Permutation.runF Permutation.schedule state) := by
  calc
    Spec.Poseidon2.permute (List.ofFn state) =
        Permutation.runReference Permutation.schedule (List.ofFn state) :=
      (Permutation.runReference_schedule _).symm
    _ = List.ofFn (Permutation.runF Permutation.schedule state) :=
      (Permutation.runF_eq_reference _ _).symm

/-- Before invocation `count`, the direct carried state is exactly the
canonical sponge state after `count` absorbed blocks. -/
theorem previousValue_eq_chainAbsorbed
    (chain : HashChain) (env : Circuit.Env)
    (output : Fin PilotPoseidonPlan.invocationCount → Layer.FState)
    (countEq : chain.absorbCount + 1 = PilotPoseidonPlan.invocationCount)
    (semantics : CanonicalSemantics chain env output) :
    ∀ count, ∀ bound : count ≤ chain.absorbCount,
      previousValue output (invocationAt chain countEq count bound) =
        NightstreamFPrime.Export.Pilot.chainAbsorbed chain env count := by
  intro count
  induction count with
  | zero =>
      intro bound
      funext lane
      simp [previousValue, invocationAt,
        NightstreamFPrime.Export.Pilot.chainAbsorbed, Hash.zeroF]
  | succ count inductionHypothesis =>
      intro bound
      have countLt : count < chain.absorbCount := by omega
      let current := invocationAt chain countEq count (Nat.le_of_lt countLt)
      have prior := inductionHypothesis (Nat.le_of_lt countLt)
      have inputEq := canonicalInput_absorb chain env output current (by
        simpa [current, invocationAt] using countLt)
      have step := semantics.invocation current
      have outputEq : output current =
          NightstreamFPrime.Export.Pilot.chainAbsorbed
            chain env (count + 1) := by
        apply List.ofFn_injective
        calc
          List.ofFn (output current) =
              Spec.Poseidon2.permute
                (List.ofFn (canonicalInput chain env output current)) := step
          _ = Spec.Poseidon2.permute (List.ofFn (fun lane =>
                NightstreamFPrime.Export.Pilot.chainAbsorbed
                    chain env count lane +
                  NightstreamFPrime.Export.Pilot.chainBlockState
                    chain count env lane)) := by
              rw [inputEq, prior]
              simp [current, invocationAt]
          _ = List.ofFn (Permutation.runF Permutation.schedule (fun lane =>
                NightstreamFPrime.Export.Pilot.chainAbsorbed
                    chain env count lane +
                  NightstreamFPrime.Export.Pilot.chainBlockState
                    chain count env lane)) := permute_eq_runF _
          _ = List.ofFn
              (NightstreamFPrime.Export.Pilot.chainAbsorbed
                chain env (count + 1)) := by rfl
      calc
        previousValue output
            (invocationAt chain countEq (count + 1) bound) =
            output current := by
          unfold previousValue
          rw [dif_neg (by simp [invocationAt])]
          apply congrArg output
          apply Fin.ext
          simp [current, invocationAt]
        _ = NightstreamFPrime.Export.Pilot.chainAbsorbed
            chain env (count + 1) := outputEq

def finalInvocation (chain : HashChain)
    (countEq : chain.absorbCount + 1 = PilotPoseidonPlan.invocationCount) :
    Fin PilotPoseidonPlan.invocationCount :=
  invocationAt chain countEq chain.absorbCount (Nat.le_refl _)

def directDigest (chain : HashChain)
    (countEq : chain.absorbCount + 1 = PilotPoseidonPlan.invocationCount)
    (output : Fin PilotPoseidonPlan.invocationCount → Layer.FState) :
    Fin 4 → F :=
  Hash.digestF (output (finalInvocation chain countEq))

/-- The final direct invocation is the exact padded permutation over the
complete absorbed state. -/
theorem finalOutput_eq_hashState
    (chain : HashChain) (env : Circuit.Env)
    (output : Fin PilotPoseidonPlan.invocationCount → Layer.FState)
    (countEq : chain.absorbCount + 1 = PilotPoseidonPlan.invocationCount)
    (semantics : CanonicalSemantics chain env output) :
    output (finalInvocation chain countEq) =
      Permutation.runF Permutation.schedule
        (Hash.padF (NightstreamFPrime.Export.Pilot.chainAbsorbed
          chain env chain.absorbCount)) := by
  let current := finalInvocation chain countEq
  have carried := previousValue_eq_chainAbsorbed chain env output countEq
    semantics chain.absorbCount (Nat.le_refl _)
  have carriedCurrent : previousValue output current =
      NightstreamFPrime.Export.Pilot.chainAbsorbed
        chain env chain.absorbCount := by
    simpa [current, finalInvocation] using carried
  have inputEq := canonicalInput_final chain env output current (by
    simp [current, finalInvocation, invocationAt])
  have step := semantics.invocation current
  apply List.ofFn_injective
  calc
    List.ofFn (output current) =
        Spec.Poseidon2.permute
          (List.ofFn (canonicalInput chain env output current)) := step
    _ = Spec.Poseidon2.permute
        (List.ofFn (Hash.padF
          (NightstreamFPrime.Export.Pilot.chainAbsorbed
            chain env chain.absorbCount))) := by
      rw [inputEq, carriedCurrent]
    _ = List.ofFn (Permutation.runF Permutation.schedule
        (Hash.padF (NightstreamFPrime.Export.Pilot.chainAbsorbed
          chain env chain.absorbCount))) := permute_eq_runF _

private theorem chainChunks_eq_inputChunks (chain : HashChain)
    (env : Circuit.Env)
    (countEq : chain.absorbCount =
      (chain.inputLength + Spec.Poseidon2.rate - 1) /
        Spec.Poseidon2.rate) :
    NightstreamFPrime.Export.Pilot.chainChunks chain env =
      Hash.inputChunks
        (NightstreamFPrime.Export.Pilot.chainInputValues chain env) := by
  unfold NightstreamFPrime.Export.Pilot.chainChunks Hash.inputChunks
    NightstreamFPrime.Export.Pilot.chainBlockList
  simp only [NightstreamFPrime.Export.Pilot.chainInputValues,
    List.length_ofFn]
  rw [countEq]
  rfl

private theorem chainAbsorbed_final_eq_absorbManyF (chain : HashChain)
    (env : Circuit.Env)
    (countEq : chain.absorbCount =
      (chain.inputLength + Spec.Poseidon2.rate - 1) /
        Spec.Poseidon2.rate) :
    NightstreamFPrime.Export.Pilot.chainAbsorbed
        chain env chain.absorbCount =
      Hash.absorbManyF Hash.zeroF
        (Hash.inputChunks
          (NightstreamFPrime.Export.Pilot.chainInputValues chain env)) := by
  have absorbed :=
    NightstreamFPrime.Export.Pilot.chainAbsorbed_eq_absorbManyF
      chain env chain.absorbCount (Nat.le_refl _)
  have chunksLength :
      (NightstreamFPrime.Export.Pilot.chainChunks chain env).length =
        chain.absorbCount := by
    simp [NightstreamFPrime.Export.Pilot.chainChunks]
  calc
    NightstreamFPrime.Export.Pilot.chainAbsorbed
        chain env chain.absorbCount =
        Hash.absorbManyF Hash.zeroF
          ((NightstreamFPrime.Export.Pilot.chainChunks chain env).take
            chain.absorbCount) := absorbed
    _ = Hash.absorbManyF Hash.zeroF
        (NightstreamFPrime.Export.Pilot.chainChunks chain env) := by
      rw [← chunksLength, List.take_length]
    _ = Hash.absorbManyF Hash.zeroF
        (Hash.inputChunks
          (NightstreamFPrime.Export.Pilot.chainInputValues chain env)) := by
      rw [chainChunks_eq_inputChunks chain env countEq]

/-- The first four lanes of the final direct state are the exact reference
Poseidon2 hash of the canonical chain input. -/
theorem directDigest_eq_hash
    (chain : HashChain) (env : Circuit.Env)
    (output : Fin PilotPoseidonPlan.invocationCount → Layer.FState)
    (invocationCountEq :
      chain.absorbCount + 1 = PilotPoseidonPlan.invocationCount)
    (chunkCountEq : chain.absorbCount =
      (chain.inputLength + Spec.Poseidon2.rate - 1) /
        Spec.Poseidon2.rate)
    (semantics : CanonicalSemantics chain env output) :
    List.ofFn (directDigest chain invocationCountEq output) =
      Spec.Poseidon2.hash
        (NightstreamFPrime.Export.Pilot.chainInputValues chain env) := by
  have finalState := finalOutput_eq_hashState chain env output
    invocationCountEq semantics
  have absorbed := chainAbsorbed_final_eq_absorbManyF chain env chunkCountEq
  calc
    List.ofFn (directDigest chain invocationCountEq output) =
        List.ofFn (Hash.digestF
          (Permutation.runF Permutation.schedule
            (Hash.padF (NightstreamFPrime.Export.Pilot.chainAbsorbed
              chain env chain.absorbCount)))) := by
      unfold directDigest
      rw [finalState]
    _ = List.ofFn (Hash.hashF
        (NightstreamFPrime.Export.Pilot.chainInputValues chain env)) := by
      unfold Hash.hashF
      rw [absorbed]
    _ = Spec.Poseidon2.hash
        (NightstreamFPrime.Export.Pilot.chainInputValues chain env) :=
      Hash.hashF_eq_reference _

theorem priorInvocationCount_eq :
    PilotData.priorChain.absorbCount + 1 =
      PilotPoseidonPlan.invocationCount := by
  change 12349 + 1 = 12350
  decide

theorem outputInvocationCount_eq :
    PilotData.outputChain.absorbCount + 1 =
      PilotPoseidonPlan.invocationCount := by
  change 12349 + 1 = 12350
  decide

private theorem priorChunkCount_eq :
    PilotData.priorChain.absorbCount =
      (PilotData.priorChain.inputLength + Spec.Poseidon2.rate - 1) /
        Spec.Poseidon2.rate := by
  change 12349 = (49393 + 4 - 1) / 4
  decide

private theorem outputChunkCount_eq :
    PilotData.outputChain.absorbCount =
      (PilotData.outputChain.inputLength + Spec.Poseidon2.rate - 1) /
        Spec.Poseidon2.rate := by
  change 12349 = (49393 + 4 - 1) / 4
  decide

structure HashFacts {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat}
    (geometry : PiRLCPoseidonGeometry.Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth)
    (env : Circuit.Env) : Prop where
  prior : List.ofFn (directDigest PilotData.priorChain
      priorInvocationCount_eq (priorOutputValue geometry assignment)) =
    Spec.Poseidon2.hash
      (NightstreamFPrime.Export.Pilot.chainInputValues PilotData.priorChain
        env)
  output : List.ofFn (directDigest PilotData.outputChain
      outputInvocationCount_eq (outputOutputValue geometry assignment)) =
    Spec.Poseidon2.hash
      (NightstreamFPrime.Export.Pilot.chainInputValues PilotData.outputChain
        env)

/-- Both fixed pilot chains compute their exact canonical Poseidon2 hashes. -/
theorem semantics_imply_hashFacts
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : PiRLCPoseidonGeometry.Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth)
    (env : Circuit.Env)
    (one : assignment (PiRLCPoseidonGeometry.oneColumn geometry) = 1)
    (priorValues : ∀ index : Fin Data.priorChain.inputLength,
      ((PiRLCPoseidonGeometry.priorInputBlock program).form
        (PiRLCPoseidonGeometry.priorInputStart program)
        (PiRLCPoseidonGeometry.priorInputFits geometry) index).eval assignment =
          env (PilotData.priorChain.inputStart + index.val))
    (outputValues : ∀ index : Fin Data.outputChain.inputLength,
      ((PiRLCPoseidonGeometry.outputInputBlock program).form
        (PiRLCPoseidonGeometry.outputInputStart program)
        (PiRLCPoseidonGeometry.outputInputFits geometry) index).eval assignment =
          env (PilotData.outputChain.inputStart + index.val))
    (semantics : PilotPoseidonPlan.Semantics geometry assignment) :
    HashFacts geometry assignment env := by
  have canonical := canonicalSemantics geometry assignment env one priorValues
    outputValues semantics
  constructor
  · exact directDigest_eq_hash PilotData.priorChain
      env
      (priorOutputValue geometry assignment) priorInvocationCount_eq
      priorChunkCount_eq canonical.1
  · exact directDigest_eq_hash PilotData.outputChain
      env
      (outputOutputValue geometry assignment) outputInvocationCount_eq
      outputChunkCount_eq canonical.2

end NightstreamFPrime.Export.Stage1.PilotPoseidonPreservation
