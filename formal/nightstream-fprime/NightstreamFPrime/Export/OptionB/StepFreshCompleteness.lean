import NightstreamFPrime.Export.OptionB.StepRelationCompleteness

/-! Completeness of the option-B step relations at the fresh instance (CW-5.3), the converses of
`StepRelation.light_freshHolds` and `StepRelation.refresh_freshHolds`. Π_CCS reads each fresh
witness coordinate in `{-1, 0, 1}`. For slots whose bit and centered slots read only public
columns, `light_normComplete` and `refresh_normComplete` add this norm bound to
`light_stepComplete` and `refresh_stepComplete`. The fresh instance is read at the key width:
`layoutAt` reads the slots there when their logical width is the key width, which
`CircuitRelationNorm.paddedSlots` reaches. With the norm bound, `freshHolds_of_rowsZero` gives a
witness that holds as a fresh instance (`Plan.rowsZero_implies_freshHolds`), so an honest step
gives one (`light_freshComplete`, `refresh_freshComplete`).

A package's slots (`StepPackage.slotRuns`) also have bit slots on the columns of `bit` hints, but
only on a column that a row forces to 0 or 1 (`StepPackageCompleteness.slotRuns_forced`), so each
bit slot holds 0 or 1 wherever the rows hold. At a package's slots padded to a key width
(`packageSlots`) the assignment is in the norm bound with no premise on the slots
(`packageSlots_norm`): an honest step gives the fresh instance (`light_packageFreshComplete`,
`refresh_packageFreshComplete`), and so does each environment that satisfies the circuit's rows
(`packageFresh_of_rows`, without the premise of `packageFresh_of_hints` on the hints' values). -/

namespace NightstreamFPrime.Export.OptionB.StepFreshCompleteness

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
open NightstreamFPrime.Export.Generic.Pieces
open NightstreamFPrime.Export.Generic.SlotLayout
open NightstreamFPrime.Export.OptionB.StepRelation
open NightstreamFPrime.Export.OptionB.StepRelationCompleteness

variable {logicalWidth : Nat}
  {publicFits : ringDegree * PaperAlgebra.publicRingColumns ≤
    Phi81CarrierLayout.carrierWidth logicalWidth}

/-! ### The fresh instance at the key width -/

/-- The slot layout at a key width equal to the slots' logical width. -/
def layoutAt {width : Nat} (slots : List (LowNormAssignment.Slot width))
    (widthEq : ProductionAssignment.logicalWidth slots = logicalWidth) :
    LogicalPlan.Layout width logicalWidth :=
  widthEq ▸ layout slots

/-- An assignment at the slots with zero rows, in the norm bound and with public input
`encHash digest` gives a witness at the key width that holds as a fresh instance, with its own
Ajtai commitment and the public input `encHash digest`. -/
theorem freshHolds_of_rowsZero {width : Nat} (slots : List (LowNormAssignment.Slot width))
    (widthEq : ProductionAssignment.logicalWidth slots = logicalWidth)
    (ajtai : PaperAlgebra.AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
    (cubeFits : Phi81CarrierLayout.carrierWidth logicalWidth ≤ 2 ^ cubeVariables)
    (publicColumns : Nat) (pieces : List Piece) (next : Nat)
    (fits : CircuitRelation.Fits publicColumns pieces next width)
    (publicInside : LogicalPlan.publicCount ≤ width)
    (assignment : Assignment F (ProductionAssignment.logicalWidth slots))
    (rowsZero : (CircuitRelation.plan publicColumns pieces next width fits (layout slots)).RowsZero
      assignment)
    (norm : ∀ column, centeredMagnitude (assignment column) < 2) (digest : Digest)
    (encoded : LogicalPlan.publicInput (publicFits := publicFits)
      (LogicalPlan.sourceEnv (layout slots) assignment) = encHash digest) :
    ∃ witness : Assignment F (Phi81CarrierLayout.carrierWidth logicalWidth),
      CCS.Holds (PaperAlgebra.semantics ajtai) productionGlobalParams
        (freshStatement ((CircuitRelation.plan publicColumns pieces next width fits
          (layoutAt slots widthEq)).logicalRelation (publicFits := publicFits) cubeFits)
          ⟨fun _ => Phi81Relation.PiRLCAlgebra.Commitment.commit ajtai witness,
            fun _ => encHash digest⟩) witness := by
  subst widthEq
  exact ⟨_, Layout.ProductionRelation.Plan.rowsZero_implies_freshHolds _ cubeFits ajtai assignment
    (encHash digest) rowsZero norm
    ((publicInput_eq_project (layout slots) assignment publicInside).symm.trans encoded)⟩

/-! ### Honest steps -/

/-- The public front of an honest step is valid for every kind of slot. -/
theorem publicFront_valid (digest : Digest) (kind : LowNormSlot.Kind) (column : Nat) :
    LowNormSlot.Valid kind (publicFront (publicFits := publicFits) digest column) := by
  unfold publicFront
  split
  · cases kind with
    | bit => exact CircuitRelationNorm.encHash_bit (publicFits := publicFits) digest _
    | centered => exact encHash_norm (publicFits := publicFits) digest _
    | field => trivial
  · cases kind with
    | bit => exact Or.inl rfl
    | centered => show centeredMagnitude (0 : F) < 2; decide
    | field => trivial

section Light

variable (program : Stage1.Application.Program)

/-- `light_stepComplete` with the norm bound of the assignment, for slots whose bit and
centered slots read only public columns, such as `CircuitRelationNorm.paddedSlots` of field
slots. -/
theorem light_normComplete (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
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
    (publicInside : LogicalPlan.publicCount ≤ width)
    (lowSlots : ∀ slot ∈ slots, slot.kind ≠ .field →
      slot.source.val < LogicalPlan.publicCount) :
    ∃ (env : Env) (assignment : Assignment F (ProductionAssignment.logicalWidth slots)),
      LightStepSoundness.StepRepresents program
        (LightAssembler.interface base program.witnessWordCount) ajtai circuitKeys 0 input output
        proof env ∧
      (CircuitRelation.plan _ _ firstFresh width fits (SlotLayout.layout slots)).RowsZero
        assignment ∧
      LogicalPlan.publicInput (publicFits := publicFits)
          (LogicalPlan.sourceEnv (SlotLayout.layout slots) assignment) = encHash output.x ∧
      (∀ column, centeredMagnitude (assignment column) < 2) ∧
      ∀ slot ∈ slots, slot.source.val < firstFresh →
        LogicalPlan.sourceEnv (SlotLayout.layout slots) assignment slot.source.val =
          env slot.source.val := by
  obtain ⟨env, front, footprint, rows, represents⟩ := LightAssemblerCompleteness.completeWith
    program relation base ajtai circuitKeys step control 0 rfl stepEq input output proof
    selected folded holds valid selectedPc freshPublic proofEq accepted depthSmall iterationBound
    priorPcBound priorDepth nextDepth keyLength z0Length ziLength zNextLength witnessLength
    (publicFront (publicFits := publicFits) output.x)
  have endEq := LightAssembler.sharedEnd_eq base program.witnessWordCount
  have publicBelow : LogicalPlan.publicCount ≤ firstFresh := publicCount_le (by omega)
  obtain ⟨rowsZero, encoded, decode⟩ := CircuitRelationCompleteness.rowsZero_of_holdsFlat
    (publicFits := publicFits) (LightAssemblerCircuit.stepCircuit program base relation 0 ajtai)
    (LightAssembler.sharedEnd base program.witnessWordCount) firstFresh
    (lightDigestColumn program) fits slots covers publicBelow publicInside env output.x
    (publicInput_front (publicFits := publicFits) output.x front)
    (fun word => digest_words represents.outputDigest word)
    (light_below program relation ajtai firstFresh footprint firstFreshLe) rows
  refine ⟨env, _, represents, rowsZero, encoded,
    CircuitRelationNorm.assignment_norm slots _ _ firstFresh env publicBelow ?_ ?_, decode⟩
  · intro column isPublic
    rw [front column isPublic]
    exact publicFront_valid (publicFits := publicFits) output.x .centered column
  · intro slot member nonField
    have isPublic := lowSlots slot member nonField
    refine ⟨lt_of_lt_of_le isPublic publicBelow, ?_⟩
    rw [front _ isPublic]
    exact publicFront_valid (publicFits := publicFits) output.x slot.kind _

/-- Completeness of the light step's fresh instance, the converse of
`StepRelation.light_freshHolds`: an honest light step gives a witness that holds as a fresh
instance of the light circuit's relation, with its own Ajtai commitment and the public input
`encHash` of the step's output hash. The relation reads the slots at the key width (`layoutAt`),
which `CircuitRelationNorm.paddedSlots` reaches. The bit and centered slots read only public
columns. -/
theorem light_freshComplete (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
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
    (publicInside : LogicalPlan.publicCount ≤ width)
    (lowSlots : ∀ slot ∈ slots, slot.kind ≠ .field →
      slot.source.val < LogicalPlan.publicCount)
    (widthEq : ProductionAssignment.logicalWidth slots = logicalWidth)
    (cubeFits : Phi81CarrierLayout.carrierWidth logicalWidth ≤ 2 ^ cubeVariables) :
    ∃ (env : Env) (witness : Assignment F (Phi81CarrierLayout.carrierWidth logicalWidth)),
      LightStepSoundness.StepRepresents program
        (LightAssembler.interface base program.witnessWordCount) ajtai circuitKeys 0 input output
        proof env ∧
      CCS.Holds (PaperAlgebra.semantics ajtai) productionGlobalParams
        (freshStatement ((CircuitRelation.plan _ _ firstFresh width fits
          (layoutAt slots widthEq)).logicalRelation (publicFits := publicFits) cubeFits)
          ⟨fun _ => Phi81Relation.PiRLCAlgebra.Commitment.commit ajtai witness,
            fun _ => encHash output.x⟩) witness := by
  obtain ⟨env, assignment, represents, rowsZero, encoded, norm, _⟩ :=
    light_normComplete program relation ajtai circuitKeys step control stepEq input output
    proof selected folded holds valid selectedPc freshPublic proofEq accepted depthSmall
    iterationBound priorPcBound priorDepth nextDepth keyLength z0Length ziLength zNextLength
    witnessLength firstFresh width fits slots covers firstFreshLe publicInside lowSlots
  obtain ⟨witness, fresh⟩ := freshHolds_of_rowsZero slots widthEq ajtai cubeFits _ _ firstFresh
    fits publicInside assignment rowsZero norm output.x encoded
  exact ⟨env, witness, represents, fresh⟩

end Light

section Refresh

variable (program : Stage1.Application.Program)

/-- `refresh_stepComplete` with the norm bound of the assignment, for slots whose bit and
centered slots read only public columns, such as `CircuitRelationNorm.paddedSlots` of field
slots. -/
theorem refresh_normComplete (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
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
    (publicInside : LogicalPlan.publicCount ≤ width)
    (lowSlots : ∀ slot ∈ slots, slot.kind ≠ .field →
      slot.source.val < LogicalPlan.publicCount) :
    ∃ (env : Env) (assignment : Assignment F (ProductionAssignment.logicalWidth slots)),
      RefreshStepSoundness.StepRepresents program
        (RefreshAssembler.interface base program.witnessWordCount) ajtai circuitKeys 1 input output
        proof env ∧
      (CircuitRelation.plan _ _ firstFresh width fits (SlotLayout.layout slots)).RowsZero
        assignment ∧
      LogicalPlan.publicInput (publicFits := publicFits)
          (LogicalPlan.sourceEnv (SlotLayout.layout slots) assignment) = encHash output.x ∧
      (∀ column, centeredMagnitude (assignment column) < 2) ∧
      ∀ slot ∈ slots, slot.source.val < firstFresh →
        LogicalPlan.sourceEnv (SlotLayout.layout slots) assignment slot.source.val =
          env slot.source.val := by
  obtain ⟨env, front, footprint, rows, represents⟩ := RefreshAssemblerCompleteness.completeWith
    program relation base ajtai circuitKeys step control 1 rfl stepEq input output proof
    selected folded holds valid selectedPc freshPublic proofEq accepted iterationBound priorPcBound
    priorDepth nextDepth keyLength z0Length ziLength zNextLength witnessLength
    (publicFront (publicFits := publicFits) output.x)
  have endEq := RefreshAssembler.sharedEnd_eq base program.witnessWordCount
  have publicBelow : LogicalPlan.publicCount ≤ firstFresh := publicCount_le (by omega)
  obtain ⟨rowsZero, encoded, decode⟩ := CircuitRelationCompleteness.rowsZero_of_holdsFlat
    (publicFits := publicFits) (RefreshAssemblerCircuit.stepCircuit program base relation 1 ajtai)
    (RefreshAssembler.sharedEnd base program.witnessWordCount) firstFresh
    (refreshDigestColumn program) fits slots covers publicBelow publicInside env output.x
    (publicInput_front (publicFits := publicFits) output.x front)
    (fun word => digest_words represents.outputDigest word)
    (refresh_below program relation ajtai firstFresh footprint firstFreshLe) rows
  refine ⟨env, _, represents, rowsZero, encoded,
    CircuitRelationNorm.assignment_norm slots _ _ firstFresh env publicBelow ?_ ?_, decode⟩
  · intro column isPublic
    rw [front column isPublic]
    exact publicFront_valid (publicFits := publicFits) output.x .centered column
  · intro slot member nonField
    have isPublic := lowSlots slot member nonField
    refine ⟨lt_of_lt_of_le isPublic publicBelow, ?_⟩
    rw [front _ isPublic]
    exact publicFront_valid (publicFits := publicFits) output.x slot.kind _

/-- Completeness of the refresh step's fresh instance, the converse of
`StepRelation.refresh_freshHolds`: an honest refresh step gives a witness that holds as a fresh
instance of the refresh circuit's relation, with its own Ajtai commitment and the public input
`encHash` of the step's output hash. The relation reads the slots at the key width (`layoutAt`),
which `CircuitRelationNorm.paddedSlots` reaches. The bit and centered slots read only public
columns. -/
theorem refresh_freshComplete (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
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
    (publicInside : LogicalPlan.publicCount ≤ width)
    (lowSlots : ∀ slot ∈ slots, slot.kind ≠ .field →
      slot.source.val < LogicalPlan.publicCount)
    (widthEq : ProductionAssignment.logicalWidth slots = logicalWidth)
    (cubeFits : Phi81CarrierLayout.carrierWidth logicalWidth ≤ 2 ^ cubeVariables) :
    ∃ (env : Env) (witness : Assignment F (Phi81CarrierLayout.carrierWidth logicalWidth)),
      RefreshStepSoundness.StepRepresents program
        (RefreshAssembler.interface base program.witnessWordCount) ajtai circuitKeys 1 input output
        proof env ∧
      CCS.Holds (PaperAlgebra.semantics ajtai) productionGlobalParams
        (freshStatement ((CircuitRelation.plan _ _ firstFresh width fits
          (layoutAt slots widthEq)).logicalRelation (publicFits := publicFits) cubeFits)
          ⟨fun _ => Phi81Relation.PiRLCAlgebra.Commitment.commit ajtai witness,
            fun _ => encHash output.x⟩) witness := by
  obtain ⟨env, assignment, represents, rowsZero, encoded, norm, _⟩ :=
    refresh_normComplete program relation ajtai circuitKeys step control stepEq input output
    proof selected folded holds valid selectedPc freshPublic proofEq accepted iterationBound
    priorPcBound priorDepth nextDepth keyLength z0Length ziLength zNextLength witnessLength
    firstFresh width fits slots covers firstFreshLe publicInside lowSlots
  obtain ⟨witness, fresh⟩ := freshHolds_of_rowsZero slots widthEq ajtai cubeFits _ _ firstFresh
    fits publicInside assignment rowsZero norm output.x encoded
  exact ⟨env, witness, represents, fresh⟩

end Refresh

/-! ### Packages -/

/-- The fresh instance of a step circuit's package (`StepPackage.packageOf`, with blocks that
pack) at its slots padded to the key width. It holds for an environment that satisfies the
circuit's rows, carries `encHash digest` in its public columns and `digest` in its digest columns,
and holds each `bit` hint's value at its column below the first fresh column. -/
theorem packageFresh_of_hints (circuit : FormalCircuit) (offset next : Nat)
    (digestColumn : Fin 4 → Nat) {width : Nat} (batches : List WitnessBatch)
    (fits : CircuitRelation.Fits CircuitRelation.publicColumns
      (CircuitRelation.pieces circuit offset digestColumn) next width)
    (packed : CircuitRelation.packable CircuitRelation.publicColumns
      (CircuitRelation.pieces circuit offset digestColumn) = true)
    (publicBelow : LogicalPlan.publicCount ≤ next) (inside : LogicalPlan.publicCount ≤ width)
    (count : Nat)
    (widthEq : ProductionAssignment.logicalWidth (CircuitRelationNorm.paddedSlots
      (runSlots width (StepPackage.slotRuns CircuitRelation.publicColumns
        (CircuitRelation.pieces circuit offset digestColumn) next batches)) inside count) =
      logicalWidth)
    (ajtai : PaperAlgebra.AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
    (cubeFits : Phi81CarrierLayout.carrierWidth logicalWidth ≤ 2 ^ cubeVariables)
    (env : Env) (digest : Digest)
    (encoded : LogicalPlan.publicInput (publicFits := publicFits) env = encHash digest)
    (words : ∀ word, env (digestColumn word) = digest.getD word.val 0)
    (below : ∀ expression ∈ CircuitRelation.constraints circuit offset digestColumn,
      expression.VarsBelow next)
    (rows : holdsFlat env (Circuit.ops circuit.main offset))
    (hints : ∀ batch ∈ batches, ∀ index source bit,
      batch.hints[index]? = some (Hint.bit source bit) →
        batch.start + batch.recipes.length + index < next ∧
          env (batch.start + batch.recipes.length + index) = Hint.eval env (.bit source bit)) :
    ∃ witness : Assignment F (Phi81CarrierLayout.carrierWidth logicalWidth),
      CCS.Holds (PaperAlgebra.semantics ajtai) productionGlobalParams
        (freshStatement ((CircuitRelation.plan _ _ next width fits
          (layoutAt _ widthEq)).logicalRelation (publicFits := publicFits) cubeFits)
          ⟨fun _ => Phi81Relation.PiRLCAlgebra.Commitment.commit ajtai witness,
            fun _ => encHash digest⟩) witness := by
  let slots := CircuitRelationNorm.paddedSlots (runSlots width (StepPackage.slotRuns
    CircuitRelation.publicColumns (CircuitRelation.pieces circuit offset digestColumn) next
    batches)) inside count
  have publicInside := CircuitRelationCompleteness.publicInside_of_packed fits packed publicBelow
  have covers : CircuitRelationCompleteness.Covers slots CircuitRelation.publicColumns
      (CircuitRelation.pieces circuit offset digestColumn) next :=
    CircuitRelationNorm.paddedSlots_covers (StepPackageCompleteness.covers_packageOf _ next width
      batches packed (StepPackageCompleteness.size_le fits packed)) inside count
  have publicEnv : ∀ column (isPublic : column < LogicalPlan.publicCount),
      env column = encHash (publicFits := publicFits) digest ⟨column, isPublic⟩ :=
    fun column isPublic => congrFun encoded ⟨column, isPublic⟩
  obtain ⟨rowsZero, publicEq, _⟩ := CircuitRelationCompleteness.rowsZero_of_holdsFlat
    (publicFits := publicFits) circuit offset next digestColumn fits slots covers publicBelow
    publicInside env digest encoded words below rows
  have valid : CircuitRelationNorm.SlotsValid slots next env :=
    CircuitRelationNorm.paddedSlots_valid (StepPackageCompleteness.slotRuns_valid _ _ next width
      batches env hints) inside count publicBelow (by
        rw [publicEnv (LogicalPlan.publicCount - 1) (by decide)]
        exact encHash_tail digest _ (by show 257 ≤ LogicalPlan.publicCount - 1; decide))
  have norm := CircuitRelationNorm.assignment_norm slots CircuitRelation.publicColumns
    (CircuitRelation.pieces circuit offset digestColumn) next env publicBelow
    (fun column isPublic => by
      rw [publicEnv column isPublic]
      exact encHash_norm digest _) valid
  exact freshHolds_of_rowsZero slots widthEq ajtai cubeFits _ _ next fits publicInside _ rowsZero
    norm digest publicEq

/-! ### The package's own slots -/

/-- A step circuit's package slots (`StepPackage.slotRuns`) padded with `count` zero slots, which
read the public column 269 (`CircuitRelationNorm.paddedSlots`). -/
def packageSlots (circuit : FormalCircuit) (offset next : Nat) (digestColumn : Fin 4 → Nat)
    {width : Nat} (batches : List WitnessBatch) (inside : LogicalPlan.publicCount ≤ width)
    (count : Nat) : List (LowNormAssignment.Slot width) :=
  CircuitRelationNorm.paddedSlots (runSlots width (StepPackage.slotRuns CircuitRelation.publicColumns
    (CircuitRelation.pieces circuit offset digestColumn) next batches)) inside count

/-- At a step circuit's padded package slots, the prover's assignment is in the norm bound once
its plan rows are zero and its public columns encode a digest: each bit slot of the package reads
a column that a row forces to 0 or 1 (`StepPackageCompleteness.slotRuns_forced`), and each zero
slot reads a public column. -/
theorem packageSlots_norm (circuit : FormalCircuit) (offset next : Nat)
    (digestColumn : Fin 4 → Nat) {width : Nat} (batches : List WitnessBatch)
    (fits : CircuitRelation.Fits CircuitRelation.publicColumns
      (CircuitRelation.pieces circuit offset digestColumn) next width)
    (packed : CircuitRelation.packable CircuitRelation.publicColumns
      (CircuitRelation.pieces circuit offset digestColumn) = true)
    (publicBelow : LogicalPlan.publicCount ≤ next) (inside : LogicalPlan.publicCount ≤ width)
    (count : Nat) (env : Env) (digest : Digest)
    (encoded : LogicalPlan.publicInput (publicFits := publicFits) env = encHash digest)
    (rowsZero : (CircuitRelation.plan _ _ next width fits (SlotLayout.layout
      (packageSlots circuit offset next digestColumn batches inside count))).RowsZero
      (CircuitRelationCompleteness.assignment
        (packageSlots circuit offset next digestColumn batches inside count)
        CircuitRelation.publicColumns (CircuitRelation.pieces circuit offset digestColumn) next env))
    (column : Fin (ProductionAssignment.logicalWidth
      (packageSlots circuit offset next digestColumn batches inside count))) :
    centeredMagnitude (CircuitRelationCompleteness.assignment
      (packageSlots circuit offset next digestColumn batches inside count)
      CircuitRelation.publicColumns (CircuitRelation.pieces circuit offset digestColumn) next env
      column) < 2 := by
  have positive : 0 < LogicalPlan.publicCount := by decide
  have low : ∀ column, column < next → CircuitRelationCompleteness.filled
      CircuitRelation.publicColumns (CircuitRelation.pieces circuit offset digestColumn) next env
      column = env column :=
    fun column below => CircuitRelationNorm.filled_below _ _ next env below
  apply CircuitRelationNorm.encode_norm_of_forced _ fits.rows _ _
    (CircuitRelationNorm.relationRows_of_rowsZero fits _ _ rowsZero)
  · rw [low 0 (by omega)]
    exact CircuitRelationCompleteness.one_of_encoded (publicFits := publicFits) env digest encoded
  · intro column isPublic kind
    rw [low column (by omega)]
    exact CircuitRelationNorm.encoded_valid (publicFits := publicFits) encoded isPublic kind
  · intro slot member nonField
    rcases List.mem_append.mp member with own | zero
    · obtain ⟨bit, row, rowMember, found⟩ := StepPackageCompleteness.slotRuns_forced _ _ next width
        batches own nonField
      refine Or.inr ⟨bit, row, ?_, found⟩
      unfold CircuitRelation.relationRows
      rw [ite_eq_left packed]
      exact List.mem_append_left _ rowMember
    · rw [List.eq_of_mem_replicate zero]
      exact Or.inl (show LogicalPlan.publicCount - 1 < LogicalPlan.publicCount by omega)

/-- The fresh instance of a step circuit's package at its slots padded to the key width holds for
an environment that satisfies the circuit's rows and carries `encHash digest` in its public columns
and `digest` in its digest columns. Unlike `packageFresh_of_hints`, it needs no premise on the
values of the `bit` hints. -/
theorem packageFresh_of_rows (circuit : FormalCircuit) (offset next : Nat)
    (digestColumn : Fin 4 → Nat) {width : Nat} (batches : List WitnessBatch)
    (fits : CircuitRelation.Fits CircuitRelation.publicColumns
      (CircuitRelation.pieces circuit offset digestColumn) next width)
    (packed : CircuitRelation.packable CircuitRelation.publicColumns
      (CircuitRelation.pieces circuit offset digestColumn) = true)
    (publicBelow : LogicalPlan.publicCount ≤ next) (inside : LogicalPlan.publicCount ≤ width)
    (count : Nat)
    (widthEq : ProductionAssignment.logicalWidth
      (packageSlots circuit offset next digestColumn batches inside count) = logicalWidth)
    (ajtai : PaperAlgebra.AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
    (cubeFits : Phi81CarrierLayout.carrierWidth logicalWidth ≤ 2 ^ cubeVariables)
    (env : Env) (digest : Digest)
    (encoded : LogicalPlan.publicInput (publicFits := publicFits) env = encHash digest)
    (words : ∀ word, env (digestColumn word) = digest.getD word.val 0)
    (below : ∀ expression ∈ CircuitRelation.constraints circuit offset digestColumn,
      expression.VarsBelow next)
    (rows : holdsFlat env (Circuit.ops circuit.main offset)) :
    ∃ witness : Assignment F (Phi81CarrierLayout.carrierWidth logicalWidth),
      CCS.Holds (PaperAlgebra.semantics ajtai) productionGlobalParams
        (freshStatement ((CircuitRelation.plan _ _ next width fits
          (layoutAt _ widthEq)).logicalRelation (publicFits := publicFits) cubeFits)
          ⟨fun _ => Phi81Relation.PiRLCAlgebra.Commitment.commit ajtai witness,
            fun _ => encHash digest⟩) witness := by
  have publicInside := CircuitRelationCompleteness.publicInside_of_packed fits packed publicBelow
  have covers := CircuitRelationNorm.paddedSlots_covers (StepPackageCompleteness.covers_packageOf _
    next width batches packed (StepPackageCompleteness.size_le fits packed)) inside count
  obtain ⟨rowsZero, publicEq, _⟩ := CircuitRelationCompleteness.rowsZero_of_holdsFlat
    (publicFits := publicFits) circuit offset next digestColumn fits
    (packageSlots circuit offset next digestColumn batches inside count) covers publicBelow
    publicInside env digest encoded words below rows
  exact freshHolds_of_rowsZero _ widthEq ajtai cubeFits _ _ next fits publicInside _ rowsZero
    (packageSlots_norm circuit offset next digestColumn batches fits packed publicBelow inside count
      env digest encoded rowsZero) digest publicEq

section LightPackage

variable (program : Stage1.Application.Program)

/-- `light_normComplete` at the light package's own slots padded with `count` zero slots, when the
blocks pack: no premise on the kinds of the slots. -/
theorem light_packageNormComplete (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
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
        program.witnessWordCount)) ≤ firstFresh)
    (inside : LogicalPlan.publicCount ≤ width) (count : Nat) :
    ∃ (env : Env) (assignment : Assignment F (ProductionAssignment.logicalWidth
        (packageSlots (LightAssemblerCircuit.stepCircuit program base relation 0 ajtai)
        (LightAssembler.sharedEnd base program.witnessWordCount) firstFresh
        (lightDigestColumn program) batches inside count))),
      LightStepSoundness.StepRepresents program
        (LightAssembler.interface base program.witnessWordCount) ajtai circuitKeys 0 input output
        proof env ∧
      (CircuitRelation.plan _ _ firstFresh width fits (SlotLayout.layout
        (packageSlots (LightAssemblerCircuit.stepCircuit program base relation 0 ajtai)
        (LightAssembler.sharedEnd base program.witnessWordCount) firstFresh
        (lightDigestColumn program) batches inside count))).RowsZero assignment ∧
      LogicalPlan.publicInput (publicFits := publicFits)
          (LogicalPlan.sourceEnv (SlotLayout.layout
            (packageSlots (LightAssemblerCircuit.stepCircuit program base relation 0 ajtai)
        (LightAssembler.sharedEnd base program.witnessWordCount) firstFresh
        (lightDigestColumn program) batches inside count)) assignment) = encHash output.x ∧
      ∀ column, centeredMagnitude (assignment column) < 2 := by
  obtain ⟨env, front, footprint, rows, represents⟩ := LightAssemblerCompleteness.completeWith
    program relation base ajtai circuitKeys step control 0 rfl stepEq input output proof
    selected folded holds valid selectedPc freshPublic proofEq accepted depthSmall iterationBound
    priorPcBound priorDepth nextDepth keyLength z0Length ziLength zNextLength witnessLength
    (publicFront (publicFits := publicFits) output.x)
  have endEq := LightAssembler.sharedEnd_eq base program.witnessWordCount
  have publicBelow : LogicalPlan.publicCount ≤ firstFresh := publicCount_le (by omega)
  have encodedEnv := publicInput_front (publicFits := publicFits) output.x front
  have covers := CircuitRelationNorm.paddedSlots_covers (StepPackageCompleteness.covers_packageOf _
    firstFresh width batches packed (StepPackageCompleteness.size_le fits packed)) inside count
  obtain ⟨rowsZero, encoded, _⟩ := CircuitRelationCompleteness.rowsZero_of_holdsFlat
    (publicFits := publicFits) (LightAssemblerCircuit.stepCircuit program base relation 0 ajtai)
    (LightAssembler.sharedEnd base program.witnessWordCount) firstFresh
    (lightDigestColumn program) fits _ covers publicBelow inside env output.x encodedEnv
    (fun word => digest_words represents.outputDigest word)
    (light_below program relation ajtai firstFresh footprint firstFreshLe) rows
  exact ⟨env, _, represents, rowsZero, encoded, packageSlots_norm _ _ firstFresh _ batches fits
    packed publicBelow inside count env output.x encodedEnv rowsZero⟩

/-- `light_freshComplete` at the light package's own slots padded to the key width, when the blocks
pack: an honest light step gives a witness that holds as a fresh instance of the light circuit's
relation, with its own Ajtai commitment and the public input `encHash` of the step's output hash,
with no premise on the kinds of the slots. -/
theorem light_packageFreshComplete (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
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
        program.witnessWordCount)) ≤ firstFresh)
    (inside : LogicalPlan.publicCount ≤ width) (count : Nat)
    (widthEq : ProductionAssignment.logicalWidth
      (packageSlots (LightAssemblerCircuit.stepCircuit program base relation 0 ajtai)
        (LightAssembler.sharedEnd base program.witnessWordCount) firstFresh
        (lightDigestColumn program) batches inside count) = logicalWidth)
    (cubeFits : Phi81CarrierLayout.carrierWidth logicalWidth ≤ 2 ^ cubeVariables) :
    ∃ (env : Env) (witness : Assignment F (Phi81CarrierLayout.carrierWidth logicalWidth)),
      LightStepSoundness.StepRepresents program
        (LightAssembler.interface base program.witnessWordCount) ajtai circuitKeys 0 input output
        proof env ∧
      CCS.Holds (PaperAlgebra.semantics ajtai) productionGlobalParams
        (freshStatement ((CircuitRelation.plan _ _ firstFresh width fits
          (layoutAt _ widthEq)).logicalRelation (publicFits := publicFits) cubeFits)
          ⟨fun _ => Phi81Relation.PiRLCAlgebra.Commitment.commit ajtai witness,
            fun _ => encHash output.x⟩) witness := by
  have endEq := LightAssembler.sharedEnd_eq base program.witnessWordCount
  have publicInside := CircuitRelationCompleteness.publicInside_of_packed fits packed
    (publicCount_le (by omega))
  obtain ⟨env, assignment, represents, rowsZero, encoded, norm⟩ :=
    light_packageNormComplete program relation ajtai circuitKeys step control stepEq input output
    proof selected folded holds valid selectedPc freshPublic proofEq accepted depthSmall
    iterationBound priorPcBound priorDepth nextDepth keyLength z0Length ziLength zNextLength
    witnessLength firstFresh width batches fits packed firstFreshLe inside count
  obtain ⟨witness, fresh⟩ := freshHolds_of_rowsZero _ widthEq ajtai cubeFits _ _ firstFresh
    fits publicInside assignment rowsZero norm output.x encoded
  exact ⟨env, witness, represents, fresh⟩

end LightPackage

section RefreshPackage

variable (program : Stage1.Application.Program)

/-- `refresh_normComplete` at the refresh package's own slots padded with `count` zero slots, when
the blocks pack: no premise on the kinds of the slots. -/
theorem refresh_packageNormComplete (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
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
        program.witnessWordCount)) ≤ firstFresh)
    (inside : LogicalPlan.publicCount ≤ width) (count : Nat) :
    ∃ (env : Env) (assignment : Assignment F (ProductionAssignment.logicalWidth
        (packageSlots (RefreshAssemblerCircuit.stepCircuit program base relation 1 ajtai)
        (RefreshAssembler.sharedEnd base program.witnessWordCount) firstFresh
        (refreshDigestColumn program) batches inside count))),
      RefreshStepSoundness.StepRepresents program
        (RefreshAssembler.interface base program.witnessWordCount) ajtai circuitKeys 1 input output
        proof env ∧
      (CircuitRelation.plan _ _ firstFresh width fits (SlotLayout.layout
        (packageSlots (RefreshAssemblerCircuit.stepCircuit program base relation 1 ajtai)
        (RefreshAssembler.sharedEnd base program.witnessWordCount) firstFresh
        (refreshDigestColumn program) batches inside count))).RowsZero assignment ∧
      LogicalPlan.publicInput (publicFits := publicFits)
          (LogicalPlan.sourceEnv (SlotLayout.layout
            (packageSlots (RefreshAssemblerCircuit.stepCircuit program base relation 1 ajtai)
        (RefreshAssembler.sharedEnd base program.witnessWordCount) firstFresh
        (refreshDigestColumn program) batches inside count)) assignment) = encHash output.x ∧
      ∀ column, centeredMagnitude (assignment column) < 2 := by
  obtain ⟨env, front, footprint, rows, represents⟩ := RefreshAssemblerCompleteness.completeWith
    program relation base ajtai circuitKeys step control 1 rfl stepEq input output proof
    selected folded holds valid selectedPc freshPublic proofEq accepted iterationBound priorPcBound
    priorDepth nextDepth keyLength z0Length ziLength zNextLength witnessLength
    (publicFront (publicFits := publicFits) output.x)
  have endEq := RefreshAssembler.sharedEnd_eq base program.witnessWordCount
  have publicBelow : LogicalPlan.publicCount ≤ firstFresh := publicCount_le (by omega)
  have encodedEnv := publicInput_front (publicFits := publicFits) output.x front
  have covers := CircuitRelationNorm.paddedSlots_covers (StepPackageCompleteness.covers_packageOf _
    firstFresh width batches packed (StepPackageCompleteness.size_le fits packed)) inside count
  obtain ⟨rowsZero, encoded, _⟩ := CircuitRelationCompleteness.rowsZero_of_holdsFlat
    (publicFits := publicFits) (RefreshAssemblerCircuit.stepCircuit program base relation 1 ajtai)
    (RefreshAssembler.sharedEnd base program.witnessWordCount) firstFresh
    (refreshDigestColumn program) fits _ covers publicBelow inside env output.x encodedEnv
    (fun word => digest_words represents.outputDigest word)
    (refresh_below program relation ajtai firstFresh footprint firstFreshLe) rows
  exact ⟨env, _, represents, rowsZero, encoded, packageSlots_norm _ _ firstFresh _ batches fits
    packed publicBelow inside count env output.x encodedEnv rowsZero⟩

/-- `refresh_freshComplete` at the refresh package's own slots padded to the key width, when the
blocks pack, with no premise on the kinds of the slots. -/
theorem refresh_packageFreshComplete (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
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
        program.witnessWordCount)) ≤ firstFresh)
    (inside : LogicalPlan.publicCount ≤ width) (count : Nat)
    (widthEq : ProductionAssignment.logicalWidth
      (packageSlots (RefreshAssemblerCircuit.stepCircuit program base relation 1 ajtai)
        (RefreshAssembler.sharedEnd base program.witnessWordCount) firstFresh
        (refreshDigestColumn program) batches inside count) = logicalWidth)
    (cubeFits : Phi81CarrierLayout.carrierWidth logicalWidth ≤ 2 ^ cubeVariables) :
    ∃ (env : Env) (witness : Assignment F (Phi81CarrierLayout.carrierWidth logicalWidth)),
      RefreshStepSoundness.StepRepresents program
        (RefreshAssembler.interface base program.witnessWordCount) ajtai circuitKeys 1 input output
        proof env ∧
      CCS.Holds (PaperAlgebra.semantics ajtai) productionGlobalParams
        (freshStatement ((CircuitRelation.plan _ _ firstFresh width fits
          (layoutAt _ widthEq)).logicalRelation (publicFits := publicFits) cubeFits)
          ⟨fun _ => Phi81Relation.PiRLCAlgebra.Commitment.commit ajtai witness,
            fun _ => encHash output.x⟩) witness := by
  have endEq := RefreshAssembler.sharedEnd_eq base program.witnessWordCount
  have publicInside := CircuitRelationCompleteness.publicInside_of_packed fits packed
    (publicCount_le (by omega))
  obtain ⟨env, assignment, represents, rowsZero, encoded, norm⟩ :=
    refresh_packageNormComplete program relation ajtai circuitKeys step control stepEq input output
    proof selected folded holds valid selectedPc freshPublic proofEq accepted iterationBound
    priorPcBound priorDepth nextDepth keyLength z0Length ziLength zNextLength witnessLength
    firstFresh width batches fits packed firstFreshLe inside count
  obtain ⟨witness, fresh⟩ := freshHolds_of_rowsZero _ widthEq ajtai cubeFits _ _ firstFresh
    fits publicInside assignment rowsZero norm output.x encoded
  exact ⟨env, witness, represents, fresh⟩

end RefreshPackage

end NightstreamFPrime.Export.OptionB.StepFreshCompleteness
