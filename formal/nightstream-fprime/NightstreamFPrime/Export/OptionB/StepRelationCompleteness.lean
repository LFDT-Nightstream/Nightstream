import NightstreamFPrime.Export.OptionB.StepRelation
import NightstreamFPrime.Export.OptionB.StepPackageCompleteness

/-! Completeness of the option-B step relations (CW-5.3), the converses of
`StepRelation.light_stepHolds` and `StepRelation.refresh_stepHolds`. An honest light or refresh
step gives an environment that carries the step and the public input `encHash x` below
`base` (`LightAssemblerCompleteness.completeWith`, `RefreshAssemblerCompleteness.completeWith`).
Its digest columns carry `x`, so `CircuitRelationCompleteness.rowsZero_of_holdsFlat` gives the
prover's assignment of the exported relation: its rows are zero, its public input encodes the
step's output hash, and it decodes to the honest environment on every live column below the
first fresh column. For the package's own slots (`StepPackage.slotRuns`) the cover holds when the
blocks pack (`light_packageComplete`, `refresh_packageComplete`). -/

namespace NightstreamFPrime.Export.OptionB.StepRelationCompleteness

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra
open NightstreamFPrime.Spec.HyperNova.Construction2
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PiCCS.v1_2
open NightstreamFPrime.Lifecycle.OptionB
open NightstreamFPrime.Lifecycle.OptionB.SlotWords
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.OptionB
open NightstreamFPrime.Export.Generic
open NightstreamFPrime.Export.OptionB.StepRelation

variable {logicalWidth : Nat}
  {publicFits : ringDegree * PaperAlgebra.publicRingColumns ≤
    Phi81CarrierLayout.carrierWidth logicalWidth}

/-- The public input `encHash digest` as the columns below `base`. -/
def publicFront (digest : Digest) : Env := fun column =>
  if inside : column < ringDegree * PaperAlgebra.publicRingColumns then
    encHash (publicFits := publicFits) digest ⟨column, inside⟩
  else 0

theorem publicInput_front {env : Env} (digest : Digest)
    (front : ∀ column, column < base → env column = publicFront (publicFits := publicFits) digest
      column) :
    LogicalPlan.publicInput (publicFits := publicFits) env = encHash digest := by
  funext column
  show env column.val = _
  have inside : column.val < ringDegree * PaperAlgebra.publicRingColumns := column.isLt
  rw [front column.val inside, publicFront, dite_eq_left inside]

theorem digest_words {digest : List F} {words : Fin 4 → Expr} {env : Env}
    (digestEq : digest = List.ofFn fun lane => (words lane).eval env) (word : Fin 4) :
    (words word).eval env = digest.getD word.val 0 := by
  rw [digestEq, List.getD_eq_getElem _ _ (by simp), List.getElem_ofFn]

theorem base_eq : base = 270 := by decide

theorem publicCount_le {firstFresh : Nat} (atLeast : base ≤ firstFresh) :
    LogicalPlan.publicCount ≤ firstFresh := by
  have : LogicalPlan.publicCount = 270 := by decide
  have := base_eq
  omega

section Light

variable (program : Stage1.Application.Program)

/-- The light circuit's constraints and decode constraints read below `firstFresh`. -/
theorem light_below (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : PaperAlgebra.AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
    (firstFresh : Nat)
    (footprint : ∀ expression ∈ flatConstraints (Circuit.ops (LightStepLayout.main program relation 0
        (LightAssembler.interface base program.witnessWordCount)) (LightAssembler.sharedEnd base
        program.witnessWordCount)),
      expression.VarsBelow (LightAssembler.sharedEnd base program.witnessWordCount +
        localLength (Circuit.ops (LightStepLayout.main program relation 0
        (LightAssembler.interface base program.witnessWordCount)) (LightAssembler.sharedEnd base
        program.witnessWordCount))))
    (firstFreshLe : LightAssembler.sharedEnd base program.witnessWordCount + localLength
      (Circuit.ops (LightStepLayout.main program relation 0
        (LightAssembler.interface base program.witnessWordCount)) (LightAssembler.sharedEnd base
        program.witnessWordCount)) ≤ firstFresh) :
    ∀ expression ∈ CircuitRelation.constraints
      (LightAssemblerCircuit.stepCircuit program base relation 0 ajtai)
      (LightAssembler.sharedEnd base program.witnessWordCount) (lightDigestColumn program),
      expression.VarsBelow firstFresh := by
  have endEq := LightAssembler.sharedEnd_eq base program.witnessWordCount
  have baseEq := base_eq
  intro expression member
  rcases List.mem_append.mp member with flat | decode
  · exact Expr.VarsBelow.mono _ (footprint expression flat) firstFreshLe
  · refine CircuitRelationCompleteness.decodeConstraints_below _ (fun word => ?_) (by omega)
      expression decode
    have := word.isLt
    simp only [lightDigestColumn, LightAssembler.digestsStart, LightAssembler.slotsStart,
      LightAssembler.witnessStart, LightAssembler.slotWordCount]
    omega

/-- Completeness of the light step relation, the converse of `StepRelation.light_stepHolds`: an
honest light step gives an assignment of the light circuit's exported relation whose rows are zero
and whose public input encodes the step's output hash. It decodes to an environment that carries
the step on every live column below the first fresh column. The fresh columns start at or after
the circuit's last local, and every live column of the relation has a slot. -/
theorem light_stepComplete (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : PaperAlgebra.AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
    (circuitKeys : Fin 2 → CircuitKey logicalWidth publicFits)
    (step : AppState → AppWitness → AppState) (control : AppState → AppWitness → Fin 2)
    (stepEq : program.step = step)
    (input : StepInput logicalWidth publicFits) (output : StepOutput logicalWidth publicFits)
    (proof : Spec.Folding.Light.Proof LightShape 8) (selected : Fin 2)
    (folded : SlotClaim logicalWidth publicFits)
    (holds : StepHolds Folding.Schedule.productionLightFoldCap ajtai circuitKeys step control 0
      input output)
    (valid : program.valid input.zi input.witness)
    (selectedPc : input.iteration ≠ 0 → input.priorPc = selected.val + 1)
    (freshPublic : input.fresh.publicInputs ⟨0, by decide⟩ =
      encHash (publicFits := publicFits) (stateHash (toState (Paper.priorHashPreimage
        ((setup Folding.Schedule.productionLightFoldCap ajtai circuitKeys).forCircuit 0)
        input))))
    (proofEq : input.iteration ≠ 0 → input.nifsProof = .light proof)
    (accepted : OptionBFolds.folds .light (keys (circuitKeys selected).relation ajtai)
      (toClaim ((circuitKeys selected).system ajtai) (input.running selected).1) input.fresh
      (.light proof) = some (toClaim ((circuitKeys selected).system ajtai) folded))
    (depthSmall : (input.running selected).2 < Folding.Schedule.productionLightFoldCap)
    (iterationBound : input.iteration < goldilocksModulus)
    (priorPcBound : input.priorPc < goldilocksModulus)
    (priorDepth : ∀ slot, (input.running slot).2 < goldilocksModulus)
    (nextDepth : ∀ slot, (output.runningNext slot).2 < goldilocksModulus)
    (keyLength : ∀ key, (circuitKeys key).digest.length = 4)
    (z0Length : input.z0.length = 4) (ziLength : input.zi.length = 4)
    (zNextLength : output.zNext.length = 4)
    (witnessLength : input.witness.length = program.witnessWordCount)
    (firstFresh width : Nat)
    (fits : CircuitRelation.Fits CircuitRelation.publicColumns (CircuitRelation.pieces
      (LightAssemblerCircuit.stepCircuit program base relation 0 ajtai)
      (LightAssembler.sharedEnd base program.witnessWordCount)
      (lightDigestColumn program)) firstFresh width)
    (slots : List (LowNormAssignment.Slot width))
    (covers : CircuitRelationCompleteness.Covers slots CircuitRelation.publicColumns
      (CircuitRelation.pieces (LightAssemblerCircuit.stepCircuit program base relation 0 ajtai)
        (LightAssembler.sharedEnd base program.witnessWordCount) (lightDigestColumn program))
      firstFresh)
    (firstFreshLe : LightAssembler.sharedEnd base program.witnessWordCount + localLength
      (Circuit.ops (LightStepLayout.main program relation 0
        (LightAssembler.interface base program.witnessWordCount)) (LightAssembler.sharedEnd base
        program.witnessWordCount)) ≤ firstFresh)
    (publicInside : LogicalPlan.publicCount ≤ width) :
    ∃ (env : Env) (assignment : Assignment F (ProductionAssignment.logicalWidth slots)),
      LightStepSoundness.StepRepresents program
        (LightAssembler.interface base program.witnessWordCount) ajtai circuitKeys 0 input output
        proof env ∧
      (CircuitRelation.plan _ _ firstFresh width fits (SlotLayout.layout slots)).RowsZero
        assignment ∧
      LogicalPlan.publicInput (publicFits := publicFits)
          (LogicalPlan.sourceEnv (SlotLayout.layout slots) assignment) = encHash output.x ∧
      ∀ column, column < firstFresh → column < width →
        CircuitRelationCompleteness.deadColumn CircuitRelation.publicColumns
          (CircuitRelation.pieces (LightAssemblerCircuit.stepCircuit program base relation 0 ajtai)
            (LightAssembler.sharedEnd base program.witnessWordCount) (lightDigestColumn program))
          column = false →
          LogicalPlan.sourceEnv (SlotLayout.layout slots) assignment column = env column := by
  obtain ⟨env, front, footprint, rows, represents⟩ := LightAssemblerCompleteness.completeWith
    program relation base ajtai circuitKeys step control 0 rfl stepEq input output proof selected
    folded holds valid selectedPc freshPublic proofEq accepted depthSmall iterationBound
    priorPcBound priorDepth nextDepth keyLength z0Length ziLength zNextLength witnessLength
    (publicFront (publicFits := publicFits) output.x)
  have endEq := LightAssembler.sharedEnd_eq base program.witnessWordCount
  obtain ⟨rowsZero, encoded, decode⟩ := CircuitRelationCompleteness.rowsZero_of_holdsFlat
    (publicFits := publicFits) (LightAssemblerCircuit.stepCircuit program base relation 0 ajtai)
    (LightAssembler.sharedEnd base program.witnessWordCount) firstFresh (lightDigestColumn program)
    fits slots covers (publicCount_le (by omega)) publicInside env output.x
    (publicInput_front (publicFits := publicFits) output.x front)
    (fun word => digest_words represents.outputDigest word)
    (light_below program relation ajtai firstFresh footprint firstFreshLe) rows
  exact ⟨env, _, represents, rowsZero, encoded, decode⟩

/-- `light_stepComplete` at the light package's own slots (`StepPackage.packageOf`), whose cover
holds when the blocks pack. -/
theorem light_packageComplete (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : PaperAlgebra.AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
    (circuitKeys : Fin 2 → CircuitKey logicalWidth publicFits)
    (step : AppState → AppWitness → AppState) (control : AppState → AppWitness → Fin 2)
    (stepEq : program.step = step)
    (input : StepInput logicalWidth publicFits) (output : StepOutput logicalWidth publicFits)
    (proof : Spec.Folding.Light.Proof LightShape 8) (selected : Fin 2)
    (folded : SlotClaim logicalWidth publicFits)
    (holds : StepHolds Folding.Schedule.productionLightFoldCap ajtai circuitKeys step control 0
      input output)
    (valid : program.valid input.zi input.witness)
    (selectedPc : input.iteration ≠ 0 → input.priorPc = selected.val + 1)
    (freshPublic : input.fresh.publicInputs ⟨0, by decide⟩ =
      encHash (publicFits := publicFits) (stateHash (toState (Paper.priorHashPreimage
        ((setup Folding.Schedule.productionLightFoldCap ajtai circuitKeys).forCircuit 0)
        input))))
    (proofEq : input.iteration ≠ 0 → input.nifsProof = .light proof)
    (accepted : OptionBFolds.folds .light (keys (circuitKeys selected).relation ajtai)
      (toClaim ((circuitKeys selected).system ajtai) (input.running selected).1) input.fresh
      (.light proof) = some (toClaim ((circuitKeys selected).system ajtai) folded))
    (depthSmall : (input.running selected).2 < Folding.Schedule.productionLightFoldCap)
    (iterationBound : input.iteration < goldilocksModulus)
    (priorPcBound : input.priorPc < goldilocksModulus)
    (priorDepth : ∀ slot, (input.running slot).2 < goldilocksModulus)
    (nextDepth : ∀ slot, (output.runningNext slot).2 < goldilocksModulus)
    (keyLength : ∀ key, (circuitKeys key).digest.length = 4)
    (z0Length : input.z0.length = 4) (ziLength : input.zi.length = 4)
    (zNextLength : output.zNext.length = 4)
    (witnessLength : input.witness.length = program.witnessWordCount)
    (firstFresh width : Nat) (batches : List WitnessBatch)
    (fits : CircuitRelation.Fits CircuitRelation.publicColumns (CircuitRelation.pieces
      (LightAssemblerCircuit.stepCircuit program base relation 0 ajtai)
      (LightAssembler.sharedEnd base program.witnessWordCount)
      (lightDigestColumn program)) firstFresh width)
    (packed : CircuitRelation.packable CircuitRelation.publicColumns (CircuitRelation.pieces
      (LightAssemblerCircuit.stepCircuit program base relation 0 ajtai)
      (LightAssembler.sharedEnd base program.witnessWordCount) (lightDigestColumn program)) = true)
    (firstFreshLe : LightAssembler.sharedEnd base program.witnessWordCount + localLength
      (Circuit.ops (LightStepLayout.main program relation 0
        (LightAssembler.interface base program.witnessWordCount)) (LightAssembler.sharedEnd base
        program.witnessWordCount)) ≤ firstFresh) :
    ∃ (env : Env) (assignment : Assignment F (ProductionAssignment.logicalWidth
        (SlotLayout.runSlots width (StepPackage.slotRuns CircuitRelation.publicColumns
          (CircuitRelation.pieces (LightAssemblerCircuit.stepCircuit program base relation 0 ajtai)
            (LightAssembler.sharedEnd base program.witnessWordCount) (lightDigestColumn program))
          firstFresh batches)))),
      LightStepSoundness.StepRepresents program
        (LightAssembler.interface base program.witnessWordCount) ajtai circuitKeys 0 input output
        proof env ∧
      (CircuitRelation.plan _ _ firstFresh width fits (SlotLayout.layout (SlotLayout.runSlots width
        (StepPackage.slotRuns CircuitRelation.publicColumns (CircuitRelation.pieces
          (LightAssemblerCircuit.stepCircuit program base relation 0 ajtai)
          (LightAssembler.sharedEnd base program.witnessWordCount) (lightDigestColumn program))
          firstFresh batches)))).RowsZero assignment ∧
      LogicalPlan.publicInput (publicFits := publicFits)
          (LogicalPlan.sourceEnv (SlotLayout.layout (SlotLayout.runSlots width
            (StepPackage.slotRuns CircuitRelation.publicColumns (CircuitRelation.pieces
              (LightAssemblerCircuit.stepCircuit program base relation 0 ajtai)
              (LightAssembler.sharedEnd base program.witnessWordCount) (lightDigestColumn program))
              firstFresh batches))) assignment) = encHash output.x := by
  have endEq := LightAssembler.sharedEnd_eq base program.witnessWordCount
  have publicInside := CircuitRelationCompleteness.publicInside_of_packed fits packed
    (publicCount_le (by omega))
  obtain ⟨env, assignment, represents, rowsZero, encoded, _⟩ := light_stepComplete program
    relation ajtai circuitKeys step control stepEq input output proof selected folded holds valid
    selectedPc freshPublic proofEq accepted depthSmall iterationBound priorPcBound priorDepth
    nextDepth keyLength z0Length ziLength zNextLength witnessLength firstFresh width fits _
    (StepPackageCompleteness.covers_packageOf _ firstFresh width batches packed
      (StepPackageCompleteness.size_le fits packed))
    firstFreshLe publicInside
  exact ⟨env, assignment, represents, rowsZero, encoded⟩

end Light

section Refresh

variable (program : Stage1.Application.Program)

/-- The refresh circuit's constraints and decode constraints read below `firstFresh`. -/
theorem refresh_below (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : PaperAlgebra.AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
    (firstFresh : Nat)
    (footprint : ∀ expression ∈ flatConstraints (Circuit.ops (RefreshStepLayout.main program relation 1
        (RefreshAssembler.interface base program.witnessWordCount)) (RefreshAssembler.sharedEnd base
        program.witnessWordCount)),
      expression.VarsBelow (RefreshAssembler.sharedEnd base program.witnessWordCount +
        localLength (Circuit.ops (RefreshStepLayout.main program relation 1
        (RefreshAssembler.interface base program.witnessWordCount)) (RefreshAssembler.sharedEnd base
        program.witnessWordCount))))
    (firstFreshLe : RefreshAssembler.sharedEnd base program.witnessWordCount + localLength
      (Circuit.ops (RefreshStepLayout.main program relation 1
        (RefreshAssembler.interface base program.witnessWordCount)) (RefreshAssembler.sharedEnd base
        program.witnessWordCount)) ≤ firstFresh) :
    ∀ expression ∈ CircuitRelation.constraints
      (RefreshAssemblerCircuit.stepCircuit program base relation 1 ajtai)
      (RefreshAssembler.sharedEnd base program.witnessWordCount) (refreshDigestColumn program),
      expression.VarsBelow firstFresh := by
  have endEq := RefreshAssembler.sharedEnd_eq base program.witnessWordCount
  have baseEq := base_eq
  intro expression member
  rcases List.mem_append.mp member with flat | decode
  · exact Expr.VarsBelow.mono _ (footprint expression flat) firstFreshLe
  · refine CircuitRelationCompleteness.decodeConstraints_below _ (fun word => ?_) (by omega)
      expression decode
    have := word.isLt
    simp only [refreshDigestColumn, LightAssembler.digestsStart, LightAssembler.slotsStart,
      LightAssembler.witnessStart, LightAssembler.slotWordCount]
    omega

/-- Completeness of the refresh step relation, the converse of
`StepRelation.refresh_stepHolds`. -/
theorem refresh_stepComplete (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : PaperAlgebra.AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
    (circuitKeys : Fin 2 → CircuitKey logicalWidth publicFits)
    (step : AppState → AppWitness → AppState) (control : AppState → AppWitness → Fin 2)
    (stepEq : program.step = step)
    (input : StepInput logicalWidth publicFits) (output : StepOutput logicalWidth publicFits)
    (proof : Spec.Folding.Refresh.Proof PaperAlgebra.Commitment productionShape 8)
    (selected : Fin 2) (folded : SlotClaim logicalWidth publicFits)
    (holds : StepHolds Folding.Schedule.productionLightFoldCap ajtai circuitKeys step control 1
      input output)
    (valid : program.valid input.zi input.witness)
    (selectedPc : input.iteration ≠ 0 → input.priorPc = selected.val + 1)
    (freshPublic : input.fresh.publicInputs ⟨0, by decide⟩ =
      encHash (publicFits := publicFits) (stateHash (toState (Paper.priorHashPreimage
        ((setup Folding.Schedule.productionLightFoldCap ajtai circuitKeys).forCircuit 1)
        input))))
    (proofEq : input.iteration ≠ 0 → input.nifsProof = .refresh proof)
    (accepted : OptionBFolds.folds .refresh (keys (circuitKeys selected).relation ajtai)
      (toClaim ((circuitKeys selected).system ajtai) (input.running selected).1) input.fresh
      (.refresh proof) = some (toClaim ((circuitKeys selected).system ajtai) folded))
    (iterationBound : input.iteration < goldilocksModulus)
    (priorPcBound : input.priorPc < goldilocksModulus)
    (priorDepth : ∀ slot, (input.running slot).2 < goldilocksModulus)
    (nextDepth : ∀ slot, (output.runningNext slot).2 < goldilocksModulus)
    (keyLength : ∀ key, (circuitKeys key).digest.length = 4)
    (z0Length : input.z0.length = 4) (ziLength : input.zi.length = 4)
    (zNextLength : output.zNext.length = 4)
    (witnessLength : input.witness.length = program.witnessWordCount)
    (firstFresh width : Nat)
    (fits : CircuitRelation.Fits CircuitRelation.publicColumns (CircuitRelation.pieces
      (RefreshAssemblerCircuit.stepCircuit program base relation 1 ajtai)
      (RefreshAssembler.sharedEnd base program.witnessWordCount)
      (refreshDigestColumn program)) firstFresh width)
    (slots : List (LowNormAssignment.Slot width))
    (covers : CircuitRelationCompleteness.Covers slots CircuitRelation.publicColumns
      (CircuitRelation.pieces (RefreshAssemblerCircuit.stepCircuit program base relation 1 ajtai)
        (RefreshAssembler.sharedEnd base program.witnessWordCount) (refreshDigestColumn program))
      firstFresh)
    (firstFreshLe : RefreshAssembler.sharedEnd base program.witnessWordCount + localLength
      (Circuit.ops (RefreshStepLayout.main program relation 1
        (RefreshAssembler.interface base program.witnessWordCount)) (RefreshAssembler.sharedEnd base
        program.witnessWordCount)) ≤ firstFresh)
    (publicInside : LogicalPlan.publicCount ≤ width) :
    ∃ (env : Env) (assignment : Assignment F (ProductionAssignment.logicalWidth slots)),
      RefreshStepSoundness.StepRepresents program
        (RefreshAssembler.interface base program.witnessWordCount) ajtai circuitKeys 1 input output
        proof env ∧
      (CircuitRelation.plan _ _ firstFresh width fits (SlotLayout.layout slots)).RowsZero
        assignment ∧
      LogicalPlan.publicInput (publicFits := publicFits)
          (LogicalPlan.sourceEnv (SlotLayout.layout slots) assignment) = encHash output.x ∧
      ∀ column, column < firstFresh → column < width →
        CircuitRelationCompleteness.deadColumn CircuitRelation.publicColumns
          (CircuitRelation.pieces (RefreshAssemblerCircuit.stepCircuit program base relation 1
            ajtai) (RefreshAssembler.sharedEnd base program.witnessWordCount)
            (refreshDigestColumn program)) column = false →
          LogicalPlan.sourceEnv (SlotLayout.layout slots) assignment column = env column := by
  obtain ⟨env, front, footprint, rows, represents⟩ := RefreshAssemblerCompleteness.completeWith
    program relation base ajtai circuitKeys step control 1 rfl stepEq input output proof selected
    folded holds valid selectedPc freshPublic proofEq accepted iterationBound priorPcBound
    priorDepth nextDepth keyLength z0Length ziLength zNextLength witnessLength
    (publicFront (publicFits := publicFits) output.x)
  have endEq := RefreshAssembler.sharedEnd_eq base program.witnessWordCount
  obtain ⟨rowsZero, encoded, decode⟩ := CircuitRelationCompleteness.rowsZero_of_holdsFlat
    (publicFits := publicFits) (RefreshAssemblerCircuit.stepCircuit program base relation 1 ajtai)
    (RefreshAssembler.sharedEnd base program.witnessWordCount) firstFresh
    (refreshDigestColumn program) fits slots covers (publicCount_le (by omega)) publicInside env
    output.x (publicInput_front (publicFits := publicFits) output.x front)
    (fun word => digest_words represents.outputDigest word)
    (refresh_below program relation ajtai firstFresh footprint firstFreshLe) rows
  exact ⟨env, _, represents, rowsZero, encoded, decode⟩

/-- `refresh_stepComplete` at the refresh package's own slots, whose cover holds when the blocks
pack. -/
theorem refresh_packageComplete (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : PaperAlgebra.AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
    (circuitKeys : Fin 2 → CircuitKey logicalWidth publicFits)
    (step : AppState → AppWitness → AppState) (control : AppState → AppWitness → Fin 2)
    (stepEq : program.step = step)
    (input : StepInput logicalWidth publicFits) (output : StepOutput logicalWidth publicFits)
    (proof : Spec.Folding.Refresh.Proof PaperAlgebra.Commitment productionShape 8)
    (selected : Fin 2) (folded : SlotClaim logicalWidth publicFits)
    (holds : StepHolds Folding.Schedule.productionLightFoldCap ajtai circuitKeys step control 1
      input output)
    (valid : program.valid input.zi input.witness)
    (selectedPc : input.iteration ≠ 0 → input.priorPc = selected.val + 1)
    (freshPublic : input.fresh.publicInputs ⟨0, by decide⟩ =
      encHash (publicFits := publicFits) (stateHash (toState (Paper.priorHashPreimage
        ((setup Folding.Schedule.productionLightFoldCap ajtai circuitKeys).forCircuit 1)
        input))))
    (proofEq : input.iteration ≠ 0 → input.nifsProof = .refresh proof)
    (accepted : OptionBFolds.folds .refresh (keys (circuitKeys selected).relation ajtai)
      (toClaim ((circuitKeys selected).system ajtai) (input.running selected).1) input.fresh
      (.refresh proof) = some (toClaim ((circuitKeys selected).system ajtai) folded))
    (iterationBound : input.iteration < goldilocksModulus)
    (priorPcBound : input.priorPc < goldilocksModulus)
    (priorDepth : ∀ slot, (input.running slot).2 < goldilocksModulus)
    (nextDepth : ∀ slot, (output.runningNext slot).2 < goldilocksModulus)
    (keyLength : ∀ key, (circuitKeys key).digest.length = 4)
    (z0Length : input.z0.length = 4) (ziLength : input.zi.length = 4)
    (zNextLength : output.zNext.length = 4)
    (witnessLength : input.witness.length = program.witnessWordCount)
    (firstFresh width : Nat) (batches : List WitnessBatch)
    (fits : CircuitRelation.Fits CircuitRelation.publicColumns (CircuitRelation.pieces
      (RefreshAssemblerCircuit.stepCircuit program base relation 1 ajtai)
      (RefreshAssembler.sharedEnd base program.witnessWordCount)
      (refreshDigestColumn program)) firstFresh width)
    (packed : CircuitRelation.packable CircuitRelation.publicColumns (CircuitRelation.pieces
      (RefreshAssemblerCircuit.stepCircuit program base relation 1 ajtai)
      (RefreshAssembler.sharedEnd base program.witnessWordCount) (refreshDigestColumn program)) =
        true)
    (firstFreshLe : RefreshAssembler.sharedEnd base program.witnessWordCount + localLength
      (Circuit.ops (RefreshStepLayout.main program relation 1
        (RefreshAssembler.interface base program.witnessWordCount)) (RefreshAssembler.sharedEnd base
        program.witnessWordCount)) ≤ firstFresh) :
    ∃ (env : Env) (assignment : Assignment F (ProductionAssignment.logicalWidth
        (SlotLayout.runSlots width (StepPackage.slotRuns CircuitRelation.publicColumns
          (CircuitRelation.pieces (RefreshAssemblerCircuit.stepCircuit program base relation 1
            ajtai) (RefreshAssembler.sharedEnd base program.witnessWordCount)
            (refreshDigestColumn program)) firstFresh batches)))),
      RefreshStepSoundness.StepRepresents program
        (RefreshAssembler.interface base program.witnessWordCount) ajtai circuitKeys 1 input output
        proof env ∧
      (CircuitRelation.plan _ _ firstFresh width fits (SlotLayout.layout (SlotLayout.runSlots width
        (StepPackage.slotRuns CircuitRelation.publicColumns (CircuitRelation.pieces
          (RefreshAssemblerCircuit.stepCircuit program base relation 1 ajtai)
          (RefreshAssembler.sharedEnd base program.witnessWordCount) (refreshDigestColumn program))
          firstFresh batches)))).RowsZero assignment ∧
      LogicalPlan.publicInput (publicFits := publicFits)
          (LogicalPlan.sourceEnv (SlotLayout.layout (SlotLayout.runSlots width
            (StepPackage.slotRuns CircuitRelation.publicColumns (CircuitRelation.pieces
              (RefreshAssemblerCircuit.stepCircuit program base relation 1 ajtai)
              (RefreshAssembler.sharedEnd base program.witnessWordCount)
              (refreshDigestColumn program)) firstFresh batches))) assignment) =
        encHash output.x := by
  have endEq := RefreshAssembler.sharedEnd_eq base program.witnessWordCount
  have publicInside := CircuitRelationCompleteness.publicInside_of_packed fits packed
    (publicCount_le (by omega))
  obtain ⟨env, assignment, represents, rowsZero, encoded, _⟩ := refresh_stepComplete program
    relation ajtai circuitKeys step control stepEq input output proof selected folded holds valid
    selectedPc freshPublic proofEq accepted iterationBound priorPcBound priorDepth nextDepth
    keyLength z0Length ziLength zNextLength witnessLength firstFresh width fits _
    (StepPackageCompleteness.covers_packageOf _ firstFresh width batches packed
      (StepPackageCompleteness.size_le fits packed))
    firstFreshLe publicInside
  exact ⟨env, assignment, represents, rowsZero, encoded⟩

end Refresh

end NightstreamFPrime.Export.OptionB.StepRelationCompleteness
