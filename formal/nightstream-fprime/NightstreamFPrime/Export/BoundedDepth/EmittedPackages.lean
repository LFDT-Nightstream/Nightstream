import NightstreamFPrime.Export.BoundedDepth.StepExtend

/-! The bundles `TerminalBridge.Exports` and `StepExtend.Complete` of the two packages that
`lake exe emitBoundedDepth` writes for a program (`EmitMain.emitPackage`). The emitter sets the
first fresh column of each package to its shared end plus the locals of its operations
(`lightFresh`, `refreshFresh`), the source width to `2^24` (`sourceWidth`, `EmitMain.width`), and
the witness batches to the batches of its operations (`lightBatches`, `refreshBatches`). The
slot layout of each package is `SlotLayout.layout` of its slot runs, widened to the key width.

Lean proves the other fields of the bundles from these choices: the slot layouts, the public
columns inside the source width, and the first fresh column after the public columns and at the
end of the circuit's locals. `Facts` holds each fact that the bundles need and that Lean does not
prove for the emitted packages: the row, column and block bounds (`CircuitRelation.Fits`), the key
width of the slot layout, and two checks that the emitter runs before it writes a package. -/

set_option autoImplicit false

namespace NightstreamFPrime.Export.BoundedDepth.EmittedPackages

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.BoundedDepth
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.BoundedDepth
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Export.Generic
open NightstreamFPrime.Export.BoundedDepth.StepCompleteness

variable {logicalWidth : Nat}
  {publicFits : ringDegree * PaperAlgebra.publicRingColumns ≤
    RingCarrierLayout.carrierWidth logicalWidth}
  (program : Stage1.Application.Program)
  (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
  (ajtai : PaperAlgebra.AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))

/-- The source width of each emitted package. -/
def sourceWidth : Nat := 2 ^ 24

/-- The first fresh column of the light package. -/
def lightFresh : Nat :=
  LightAssembler.sharedEnd StepRelation.base program.witnessWordCount +
    localLength (lightOps program relation)

/-- The first fresh column of the refresh package. -/
def refreshFresh : Nat :=
  RefreshAssembler.sharedEnd StepRelation.base program.witnessWordCount +
    localLength (refreshOps program relation)

/-- The witness batches of the light package. -/
def lightBatches : List WitnessBatch := witnesses (lightOps program relation)

/-- The witness batches of the refresh package. -/
def refreshBatches : List WitnessBatch := witnesses (refreshOps program relation)

theorem inside : ringDegree * PaperAlgebra.publicRingColumns ≤ sourceWidth := by
  decide

theorem publicColumns_le_lightFresh :
    CircuitRelation.publicColumns ≤ lightFresh program relation := by
  have shared : CircuitRelation.publicColumns ≤
      LightAssembler.sharedEnd StepRelation.base program.witnessWordCount := by
    have baseEq : StepRelation.base = CircuitRelation.publicColumns := rfl
    simp only [LightAssembler.sharedEnd, LightAssembler.digestSelectStart,
      LightAssembler.outputStart, LightAssembler.roundsStart, LightAssembler.freshStart,
      LightAssembler.digestsStart, LightAssembler.slotsStart, LightAssembler.witnessStart, baseEq]
    omega
  exact Nat.le_add_right_of_le shared

theorem publicColumns_le_refreshFresh :
    CircuitRelation.publicColumns ≤ refreshFresh program relation := by
  have shared : CircuitRelation.publicColumns ≤
      RefreshAssembler.sharedEnd StepRelation.base program.witnessWordCount := by
    have baseEq : StepRelation.base = CircuitRelation.publicColumns := rfl
    simp only [RefreshAssembler.sharedEnd, RefreshAssembler.digestSelectStart,
      RefreshAssembler.outputStart, RefreshAssembler.roundsStart, RefreshAssembler.digitsStart,
      RefreshAssembler.childEvaluationsStart, RefreshAssembler.childCommitmentsStart,
      LightAssembler.freshStart, LightAssembler.digestsStart, LightAssembler.slotsStart,
      LightAssembler.witnessStart, baseEq]
    omega
  exact Nat.le_add_right_of_le shared

section Bundle

/-! The bundle is stated over abstract first fresh columns and witness batches, so no definition
of the bundle unfolds the operations of a circuit. `Emitted` fixes them to the emitter's values. -/

variable (lightFresh refreshFresh : Nat) (lightBatches refreshBatches : List WitnessBatch)

/-- The facts of the two packages of `program` that Lean does not prove. The emitter checks the
two facts `BlocksPack` and `ReadsNoPublicColumn` of each package before it writes the package. It
prints the sizes of `FitsPackage` but does not check them. -/
structure Facts where
  /-- The rows, the columns and the Poseidon2 blocks of the light package fit its size. -/
  lightFitsPackage : CircuitRelation.Fits CircuitRelation.publicColumns
    (lightPieces program relation ajtai) lightFresh sourceWidth
  /-- The rows, the columns and the Poseidon2 blocks of the refresh package fit its size. -/
  refreshFitsPackage : CircuitRelation.Fits CircuitRelation.publicColumns
    (refreshPieces program relation ajtai) refreshFresh sourceWidth
  /-- The slot layout of the light package fits the key width. -/
  lightFitsKeyWidth : ProductionAssignment.logicalWidth (SlotLayout.runSlots sourceWidth
    (StepPackage.slotRuns CircuitRelation.publicColumns (lightPieces program relation ajtai)
      lightFresh lightBatches)) ≤ logicalWidth
  /-- The slot layout of the refresh package fits the key width. -/
  refreshFitsKeyWidth : ProductionAssignment.logicalWidth (SlotLayout.runSlots sourceWidth
    (StepPackage.slotRuns CircuitRelation.publicColumns (refreshPieces program relation ajtai)
      refreshFresh refreshBatches)) ≤ logicalWidth
  /-- The Poseidon2 blocks of the light package pack into its S-box columns. -/
  lightBlocksPack :
    CircuitRelation.packable CircuitRelation.publicColumns (lightPieces program relation ajtai) =
      true
  /-- The Poseidon2 blocks of the refresh package pack into its S-box columns. -/
  refreshBlocksPack :
    CircuitRelation.packable CircuitRelation.publicColumns (refreshPieces program relation ajtai) =
      true
  /-- No row of the light circuit reads a public column. -/
  lightReadsNoPublicColumn : publicFree (flatConstraints (Circuit.ops
    (LightAssemblerCircuit.stepCircuit program StepRelation.base relation 0 ajtai).main
    (LightAssembler.sharedEnd StepRelation.base program.witnessWordCount))) = true
  /-- No row of the refresh circuit reads a public column. -/
  refreshReadsNoPublicColumn : publicFree (flatConstraints (Circuit.ops
    (RefreshAssemblerCircuit.stepCircuit program StepRelation.base relation 1 ajtai).main
    (RefreshAssembler.sharedEnd StepRelation.base program.witnessWordCount))) = true

variable {program relation ajtai lightFresh refreshFresh lightBatches refreshBatches}

/-- The slot layout of the light package, widened to the key width. -/
def Facts.lightLayout
    (facts : Facts program relation ajtai lightFresh refreshFresh lightBatches refreshBatches) :
    LogicalPlan.Layout sourceWidth logicalWidth :=
  WideLayout.widen (SlotLayout.layout _) facts.lightFitsKeyWidth

/-- The slot layout of the refresh package, widened to the key width. -/
def Facts.refreshLayout
    (facts : Facts program relation ajtai lightFresh refreshFresh lightBatches refreshBatches) :
    LogicalPlan.Layout sourceWidth logicalWidth :=
  WideLayout.widen (SlotLayout.layout _) facts.refreshFitsKeyWidth

variable {circuitKeys : Fin 2 → CircuitKey logicalWidth publicFits}
  {cubeFits : RingCarrierLayout.carrierWidth logicalWidth ≤ 2 ^ cubeVariables}

/-- The bundle of the two packages. The keys must carry their relations. -/
def Facts.exports
    (facts : Facts program relation ajtai lightFresh refreshFresh lightBatches refreshBatches)
    (lightKey : (circuitKeys 0).relation =
      (CircuitRelation.plan _ _ lightFresh sourceWidth facts.lightFitsPackage
        facts.lightLayout).logicalRelation (publicFits := publicFits) cubeFits)
    (refreshKey : (circuitKeys 1).relation =
      (CircuitRelation.plan _ _ refreshFresh sourceWidth facts.refreshFitsPackage
        facts.refreshLayout).logicalRelation (publicFits := publicFits) cubeFits) :
    TerminalBridge.Exports relation ajtai circuitKeys program.step cubeFits where
  program := program
  programStep := rfl
  lightFresh := lightFresh
  refreshFresh := refreshFresh
  lightWidth := sourceWidth
  refreshWidth := sourceWidth
  lightFitsPackage := facts.lightFitsPackage
  refreshFitsPackage := facts.refreshFitsPackage
  lightLayout := facts.lightLayout
  refreshLayout := facts.refreshLayout
  lightInside := inside
  refreshInside := inside
  lightKey := lightKey
  refreshKey := refreshKey

/-- The completeness bundle of the two packages, when each first fresh column is after the
public columns and at or after the end of its circuit's locals. -/
def Facts.complete
    (facts : Facts program relation ajtai lightFresh refreshFresh lightBatches refreshBatches)
    (lightKey : (circuitKeys 0).relation =
      (CircuitRelation.plan _ _ lightFresh sourceWidth facts.lightFitsPackage
        facts.lightLayout).logicalRelation (publicFits := publicFits) cubeFits)
    (refreshKey : (circuitKeys 1).relation =
      (CircuitRelation.plan _ _ refreshFresh sourceWidth facts.refreshFitsPackage
        facts.refreshLayout).logicalRelation (publicFits := publicFits) cubeFits)
    (lightPublicBelow : CircuitRelation.publicColumns ≤ lightFresh)
    (refreshPublicBelow : CircuitRelation.publicColumns ≤ refreshFresh)
    (lightAfterLocals : LightAssembler.sharedEnd StepRelation.base program.witnessWordCount +
      localLength (lightOps program relation) ≤ lightFresh)
    (refreshAfterLocals : RefreshAssembler.sharedEnd StepRelation.base program.witnessWordCount +
      localLength (refreshOps program relation) ≤ refreshFresh) :
    StepExtend.Complete (facts.exports lightKey refreshKey) where
  lightBatches := lightBatches
  lightFitsKeyWidth := facts.lightFitsKeyWidth
  lightLayout := by dsimp only [Facts.exports]; rfl
  lightBlocksPack := facts.lightBlocksPack
  lightFreshAfterLocals := lightAfterLocals
  lightPublicBelow := lightPublicBelow
  lightReadsNoPublicColumn := facts.lightReadsNoPublicColumn
  refreshBatches := refreshBatches
  refreshFitsKeyWidth := facts.refreshFitsKeyWidth
  refreshLayout := by dsimp only [Facts.exports]; rfl
  refreshBlocksPack := facts.refreshBlocksPack
  refreshFreshAfterLocals := refreshAfterLocals
  refreshPublicBelow := refreshPublicBelow
  refreshReadsNoPublicColumn := facts.refreshReadsNoPublicColumn

end Bundle

/-- The facts of the two packages that the emitter writes for `program`. -/
abbrev Emitted := Facts program relation ajtai (lightFresh program relation)
  (refreshFresh program relation) (lightBatches program relation) (refreshBatches program relation)

variable {program relation ajtai}
  {circuitKeys : Fin 2 → CircuitKey logicalWidth publicFits}
  {cubeFits : RingCarrierLayout.carrierWidth logicalWidth ≤ 2 ^ cubeVariables}

/-- The completeness bundle of the two emitted packages. Their first fresh columns are after the
public columns (`publicColumns_le_lightFresh`, `publicColumns_le_refreshFresh`), and they are the
ends of the circuits' locals (`lightFresh`, `refreshFresh`). -/
def Emitted.complete (facts : Emitted program relation ajtai)
    (lightKey : (circuitKeys 0).relation =
      (CircuitRelation.plan _ _ (lightFresh program relation) sourceWidth facts.lightFitsPackage
        facts.lightLayout).logicalRelation (publicFits := publicFits) cubeFits)
    (refreshKey : (circuitKeys 1).relation =
      (CircuitRelation.plan _ _ (refreshFresh program relation) sourceWidth facts.refreshFitsPackage
        facts.refreshLayout).logicalRelation (publicFits := publicFits) cubeFits) :
    StepExtend.Complete (facts.exports lightKey refreshKey) :=
  Facts.complete facts lightKey refreshKey (publicColumns_le_lightFresh program relation)
    (publicColumns_le_refreshFresh program relation) (Nat.le_refl _) (Nat.le_refl _)

end NightstreamFPrime.Export.BoundedDepth.EmittedPackages
