import NightstreamFPrime.Export.BoundedDepth.StepCompleteness
import NightstreamFPrime.Export.BoundedDepth.StepRelation

/-! # Claims: the two step circuits

Soundness and completeness of the light circuit and the refresh circuit, on the rows of the
relation that Lean exports (`SPEC.md` Algorithm 3, `SECURITY_MODEL.md` adversary 1).

- Assumptions: none. Each claim is deterministic.
- `light_sound` and `refresh_sound` start from the rows of the logical relation and the public
  input. They give the step relation and the output hash.
- Open: the claims are about the Lean package. Rust expands each package, and a test compares
  the rows with the checksum of Lean (`SECURITY_MODEL.md` trust item 5). -/

namespace NightstreamFPrime.Claims.Step

open Spec Spec.Folding.PiCCS.PaperJoint Lifecycle Lifecycle.BoundedDepth Layout.BoundedDepth Export.Generic Export.BoundedDepth

/-- A satisfying assignment of the light circuit's logical relation whose public input encodes
`digest` gives the light case of the bounded-depth step relation, with output hash the first four
words of `digest` (with zeros for missing words, so `digest` itself when it has four words), on the
step with the other slot of `running` when that slot's digest is its prior digest word. -/
theorem light_sound
    {logicalWidth : ℕ}
    {publicFits : ringDegree * PaperAlgebra.publicRingColumns ≤ RingCarrierLayout.carrierWidth logicalWidth}
    (program : Stage1.Application.Program)
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : PaperAlgebra.AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
    (circuitKeys : Fin 2 → CircuitKey logicalWidth publicFits)
    (step : AppState → AppWitness → AppState)
    (control : AppState → AppWitness → Fin 2)
    (stepEq : program.step = step)
    (validControl : ∀ (state : AppState) (witness : AppWitness),
        program.valid state witness → control state witness = 0)
    (firstFresh : ℕ)
    (width : ℕ)
    (fitsPackage : CircuitRelation.Fits CircuitRelation.publicColumns
        (SpecializedCircuit.pieces
          (LightAssemblerCircuit.stepCircuit program StepRelation.base relation 0 ajtai)
          (LightAssembler.sharedEnd StepRelation.base program.witnessWordCount)
          (StepRelation.lightDigestColumn program))
        firstFresh width)
    (layout : LogicalPlan.Layout width logicalWidth)
    (assignment : PaperLinearAlgebra.Assignment F logicalWidth)
    (digest : Digest)
    (encoded : Eq (α := PaperAlgebra.PublicInput (logicalWidth := logicalWidth) (publicFits := publicFits))
        (LogicalPlan.publicInput (LogicalPlan.sourceEnv layout assignment)) (encHash digest))
    (rowsZero : (CircuitRelation.plan CircuitRelation.publicColumns
            (SpecializedCircuit.pieces
              (LightAssemblerCircuit.stepCircuit program StepRelation.base relation 0 ajtai)
              (LightAssembler.sharedEnd StepRelation.base program.witnessWordCount)
              (StepRelation.lightDigestColumn program))
            firstFresh width fitsPackage layout).RowsZero
        assignment)
    (running : Fin 2 → SlotData logicalWidth publicFits)
    (others : StepRelation.OtherDigest program running
        (StepRelation.lightEnv program relation ajtai firstFresh layout assignment)) :
    have env := StepRelation.lightEnv program relation ajtai firstFresh layout assignment;
      BoundedDepth.StepHolds Folding.Schedule.productionLightFoldCap ajtai
          (StepRelation.decodedKeys circuitKeys
            (LightStepLayout.Interface.verifierKeys (logicalWidth := logicalWidth) (publicFits :=
              publicFits) (StepRelation.lightInterface program))
            env)
          step control 0 (StepRelation.lightInput program running env)
          (StepRelation.lightOutput program relation 0 running env) ∧
        (StepRelation.lightOutput program relation 0 running env).x =
          List.ofFn (n := 4) fun (word : Fin 4) => List.getD digest (↑word) 0 :=
  @Export.BoundedDepth.StepRelation.light_stepHolds logicalWidth publicFits program relation ajtai
    circuitKeys step control stepEq validControl firstFresh width fitsPackage layout assignment digest
    encoded rowsZero running others

/-- A satisfying assignment of the refresh circuit's logical relation whose public input encodes
`digest` gives the refresh case of the bounded-depth step relation, with output hash the first four
words of `digest` (with zeros for missing words, so `digest` itself when it has four words), on the
step with the other slot of `running` when that slot's digest is its prior digest word. -/
theorem refresh_sound
    {logicalWidth : ℕ}
    {publicFits : ringDegree * PaperAlgebra.publicRingColumns ≤ RingCarrierLayout.carrierWidth logicalWidth}
    (program : Stage1.Application.Program)
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : PaperAlgebra.AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
    (circuitKeys : Fin 2 → CircuitKey logicalWidth publicFits)
    (step : AppState → AppWitness → AppState)
    (control : AppState → AppWitness → Fin 2)
    (stepEq : program.step = step)
    (validControl : ∀ (state : AppState) (witness : AppWitness),
        program.valid state witness → control state witness = 1)
    (firstFresh : ℕ)
    (width : ℕ)
    (fitsPackage : CircuitRelation.Fits CircuitRelation.publicColumns
        (SpecializedCircuit.pieces
          (RefreshAssemblerCircuit.stepCircuit program StepRelation.base relation 1 ajtai)
          (RefreshAssembler.sharedEnd StepRelation.base program.witnessWordCount)
          (StepRelation.refreshDigestColumn program))
        firstFresh width)
    (layout : LogicalPlan.Layout width logicalWidth)
    (assignment : PaperLinearAlgebra.Assignment F logicalWidth)
    (digest : Digest)
    (encoded : Eq (α := PaperAlgebra.PublicInput (logicalWidth := logicalWidth) (publicFits := publicFits))
        (LogicalPlan.publicInput (LogicalPlan.sourceEnv layout assignment)) (encHash digest))
    (rowsZero : (CircuitRelation.plan CircuitRelation.publicColumns
            (SpecializedCircuit.pieces
              (RefreshAssemblerCircuit.stepCircuit program StepRelation.base relation 1 ajtai)
              (RefreshAssembler.sharedEnd StepRelation.base program.witnessWordCount)
              (StepRelation.refreshDigestColumn program))
            firstFresh width fitsPackage layout).RowsZero
        assignment)
    (running : Fin 2 → SlotData logicalWidth publicFits)
    (others : StepRelation.RefreshOtherDigest program running
        (StepRelation.refreshEnv program relation ajtai firstFresh layout assignment)) :
    have env := StepRelation.refreshEnv program relation ajtai firstFresh layout assignment;
      BoundedDepth.StepHolds Folding.Schedule.productionLightFoldCap ajtai
          (StepRelation.decodedKeys circuitKeys
            (RefreshStepLayout.Interface.verifierKeys (logicalWidth := logicalWidth) (publicFits :=
              publicFits) (StepRelation.refreshInterface program))
            env)
          step control 1 (StepRelation.refreshInput program running env)
          (StepRelation.refreshOutput program relation 1 running env) ∧
        (StepRelation.refreshOutput program relation 1 running env).x =
          List.ofFn (n := 4) fun (word : Fin 4) => List.getD digest (↑word) 0 :=
  @Export.BoundedDepth.StepRelation.refresh_stepHolds logicalWidth publicFits program relation ajtai
    circuitKeys step control stepEq validControl firstFresh width fitsPackage layout assignment digest
    encoded rowsZero running others

/-- An honest light step's assignment (`LightAssemblerCompleteness.complete`) gives a fresh
instance with public input `encHash` of the step's output hash that holds under the light key,
with a witness that is zero past the package's logical width. -/
theorem light_complete
    {logicalWidth : ℕ}
    {publicFits : ringDegree * PaperAlgebra.publicRingColumns ≤ RingCarrierLayout.carrierWidth logicalWidth}
    (program : Stage1.Application.Program)
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : PaperAlgebra.AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
    (circuitKeys : Fin 2 → CircuitKey logicalWidth publicFits)
    (input : StepInput logicalWidth publicFits)
    (output : StepOutput logicalWidth publicFits)
    (proof : Folding.Light.Proof LightShape 8)
    (env : Circuit.Env)
    (holds : Circuit.holdsFlat env
        ((LightAssemblerCircuit.stepCircuit program StepRelation.base relation 0 ajtai).main.ops
          (LightAssembler.sharedEnd StepRelation.base program.witnessWordCount)))
    (represents : LightStepSoundness.StepRepresents program (StepRelation.lightInterface program) ajtai
        circuitKeys 0 input output proof env)
    (publicEq : ∀ (column : ℕ) (inside : column < CircuitRelation.publicColumns),
        env column = encHash (publicFits := publicFits) output.x ⟨column, inside⟩)
    (firstFresh : ℕ)
    (width : ℕ)
    (fitsPackage : CircuitRelation.Fits CircuitRelation.publicColumns
        (StepCompleteness.lightPieces program relation ajtai) firstFresh width)
    (readsOwnColumns : (StepCompleteness.lightPieces program relation ajtai).all
          (PieceRowsCompleteness.pieceBelow firstFresh) =
        true)
    (publicBelow : CircuitRelation.publicColumns ≤ firstFresh)
    (batches : List Circuit.WitnessBatch)
    (fitsKeyWidth : Layout.ProductionAssignment.logicalWidth
          (SlotLayout.runSlots width
            (StepPackage.slotRuns CircuitRelation.publicColumns
              (StepCompleteness.lightPieces program relation ajtai) firstFresh batches)) ≤
        logicalWidth)
    (cubeFits : RingCarrierLayout.carrierWidth logicalWidth ≤ 2 ^ cubeVariables) :
    ∃ (assignment : PaperLinearAlgebra.Assignment F logicalWidth),
        CCS.Holds (PaperAlgebra.semantics ajtai) productionGlobalParams
            (freshStatement
              ((CircuitRelation.plan CircuitRelation.publicColumns
                    (StepCompleteness.lightPieces program relation ajtai) firstFresh width fitsPackage
                    (WideLayout.widen
                      (SlotLayout.layout
                        (SlotLayout.runSlots width
                          (StepPackage.slotRuns CircuitRelation.publicColumns
                            (StepCompleteness.lightPieces program relation ajtai) firstFresh
                            batches)))
                      fitsKeyWidth)).logicalRelation
                cubeFits)
              {
                commitments := fun (_ : Fin productionShape.freshCount) =>
                  RingRelation.PiRLCAlgebra.Commitment.commit ajtai
                    (RingCarrierLayout.extendAssignment 0 assignment),
                publicInputs := fun (_ : Fin productionShape.freshCount) => encHash output.x })
            (RingCarrierLayout.extendAssignment 0 assignment) ∧
          ∀ (column : Fin (RingCarrierLayout.carrierWidth logicalWidth)),
            Layout.ProductionAssignment.logicalWidth
                  (SlotLayout.runSlots width
                    (StepPackage.slotRuns CircuitRelation.publicColumns
                      (StepCompleteness.lightPieces program relation ajtai) firstFresh batches)) ≤
                ↑column →
              RingCarrierLayout.extendAssignment 0 assignment column = 0 :=
  @Export.BoundedDepth.StepCompleteness.light_complete logicalWidth publicFits program relation
    ajtai circuitKeys input output proof env holds represents publicEq firstFresh width fitsPackage
    readsOwnColumns publicBelow batches fitsKeyWidth cubeFits

/-- An honest refresh step's assignment (`RefreshAssemblerCompleteness.complete`) gives a fresh
instance with public input `encHash` of the step's output hash that holds under the refresh key,
with a witness that is zero past the package's logical width. -/
theorem refresh_complete
    {logicalWidth : ℕ}
    {publicFits : ringDegree * PaperAlgebra.publicRingColumns ≤ RingCarrierLayout.carrierWidth logicalWidth}
    (program : Stage1.Application.Program)
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : PaperAlgebra.AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
    (circuitKeys : Fin 2 → CircuitKey logicalWidth publicFits)
    (input : StepInput logicalWidth publicFits)
    (output : StepOutput logicalWidth publicFits)
    (proof : Folding.Refresh.Proof PaperAlgebra.Commitment productionShape 8)
    (env : Circuit.Env)
    (holds : Circuit.holdsFlat env
        ((RefreshAssemblerCircuit.stepCircuit program StepRelation.base relation 1 ajtai).main.ops
          (RefreshAssembler.sharedEnd StepRelation.base program.witnessWordCount)))
    (represents : RefreshStepSoundness.StepRepresents program (StepRelation.refreshInterface program) ajtai
        circuitKeys 1 input output proof env)
    (publicEq : ∀ (column : ℕ) (inside : column < CircuitRelation.publicColumns),
        env column = encHash (publicFits := publicFits) output.x ⟨column, inside⟩)
    (firstFresh : ℕ)
    (width : ℕ)
    (fitsPackage : CircuitRelation.Fits CircuitRelation.publicColumns
        (StepCompleteness.refreshPieces program relation ajtai) firstFresh width)
    (readsOwnColumns : (StepCompleteness.refreshPieces program relation ajtai).all
          (PieceRowsCompleteness.pieceBelow firstFresh) =
        true)
    (publicBelow : CircuitRelation.publicColumns ≤ firstFresh)
    (batches : List Circuit.WitnessBatch)
    (fitsKeyWidth : Layout.ProductionAssignment.logicalWidth
          (SlotLayout.runSlots width
            (StepPackage.slotRuns CircuitRelation.publicColumns
              (StepCompleteness.refreshPieces program relation ajtai) firstFresh batches)) ≤
        logicalWidth)
    (cubeFits : RingCarrierLayout.carrierWidth logicalWidth ≤ 2 ^ cubeVariables) :
    ∃ (assignment : PaperLinearAlgebra.Assignment F logicalWidth),
        CCS.Holds (PaperAlgebra.semantics ajtai) productionGlobalParams
            (freshStatement
              ((CircuitRelation.plan CircuitRelation.publicColumns
                    (StepCompleteness.refreshPieces program relation ajtai) firstFresh width fitsPackage
                    (WideLayout.widen
                      (SlotLayout.layout
                        (SlotLayout.runSlots width
                          (StepPackage.slotRuns CircuitRelation.publicColumns
                            (StepCompleteness.refreshPieces program relation ajtai) firstFresh
                            batches)))
                      fitsKeyWidth)).logicalRelation
                cubeFits)
              {
                commitments := fun (_ : Fin productionShape.freshCount) =>
                  RingRelation.PiRLCAlgebra.Commitment.commit ajtai
                    (RingCarrierLayout.extendAssignment 0 assignment),
                publicInputs := fun (_ : Fin productionShape.freshCount) => encHash output.x })
            (RingCarrierLayout.extendAssignment 0 assignment) ∧
          ∀ (column : Fin (RingCarrierLayout.carrierWidth logicalWidth)),
            Layout.ProductionAssignment.logicalWidth
                  (SlotLayout.runSlots width
                    (StepPackage.slotRuns CircuitRelation.publicColumns
                      (StepCompleteness.refreshPieces program relation ajtai) firstFresh
                      batches)) ≤
                ↑column →
              RingCarrierLayout.extendAssignment 0 assignment column = 0 :=
  @Export.BoundedDepth.StepCompleteness.refresh_complete logicalWidth publicFits program relation
    ajtai circuitKeys input output proof env holds represents publicEq firstFresh width fitsPackage
    readsOwnColumns publicBelow batches fitsKeyWidth cubeFits

end NightstreamFPrime.Claims.Step
