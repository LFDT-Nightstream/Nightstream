import NightstreamFPrime.Export.BoundedDepth.StepExtend

/-! The bundles `TerminalBridge.Exports` and `StepExtend.Complete` of the two packages that
`lake exe emitBoundedDepth` writes for a program (`EmitMain.emitPackage`). The emitter sets the
first fresh column of each package to its shared end plus the locals of its operations
(`lightFresh`, `refreshFresh`), the source width to `2^24` (`sourceWidth`, `EmitMain.width`), and
the witness batches to the batches of its operations (`lightBatches`, `refreshBatches`). The
slot layout of each package is `SlotLayout.layout` of its slot runs, widened to the key width.

Lean proves the other fields of the bundles from these choices: the slot layouts, the public
columns inside the source width, the first fresh column after the public columns and at the end
of the circuit's locals, and the row, column and block bounds of each package
(`CircuitRelation.Fits`, by `Facts.lightFits` and `Facts.refreshFits`). `Facts` holds the
capacity of the two packages (SPEC.md section 4.4), which Lean does not prove for an application:
the S-box columns end inside the source width, the rows are at most `2^28`, and the slot layout
fits the key width. The emitter checks each of these numbers before it writes a package. -/

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

/-- The capacity of the two packages of `program` (SPEC.md section 4.4). Lean does not prove these
numbers for an application; the emitter checks each of them before it writes a package. -/
structure Facts where
  /-- The S-box columns of the light package end inside the source width. -/
  lightColumns : CircuitRelation.sboxStart (lightPieces program relation ajtai) lightFresh +
    (CircuitRelation.blocksOf (lightPieces program relation ajtai)).length * 150 ≤ sourceWidth
  /-- The rows of the light relation, with its S-box rows, are at most `2^28`. -/
  lightRows : (CircuitRelation.relationRows CircuitRelation.publicColumns
      (lightPieces program relation ajtai) lightFresh sourceWidth).length +
    (CircuitRelation.blocksOf (lightPieces program relation ajtai)).length * 150 ≤
      2 ^ cubeVariables
  /-- The S-box columns of the refresh package end inside the source width. -/
  refreshColumns : CircuitRelation.sboxStart (refreshPieces program relation ajtai) refreshFresh +
    (CircuitRelation.blocksOf (refreshPieces program relation ajtai)).length * 150 ≤ sourceWidth
  /-- The rows of the refresh relation, with its S-box rows, are at most `2^28`. -/
  refreshRows : (CircuitRelation.relationRows CircuitRelation.publicColumns
      (refreshPieces program relation ajtai) refreshFresh sourceWidth).length +
    (CircuitRelation.blocksOf (refreshPieces program relation ajtai)).length * 150 ≤
      2 ^ cubeVariables
  /-- The slot layout of the light package fits the key width. -/
  lightFitsKeyWidth : ProductionAssignment.logicalWidth (SlotLayout.runSlots sourceWidth
    (StepPackage.slotRuns CircuitRelation.publicColumns (lightPieces program relation ajtai)
      lightFresh lightBatches)) ≤ logicalWidth
  /-- The slot layout of the refresh package fits the key width. -/
  refreshFitsKeyWidth : ProductionAssignment.logicalWidth (SlotLayout.runSlots sourceWidth
    (StepPackage.slotRuns CircuitRelation.publicColumns (refreshPieces program relation ajtai)
      refreshFresh refreshBatches)) ≤ logicalWidth

variable {program relation ajtai lightFresh refreshFresh lightBatches refreshBatches}

/-- The light package fits its source width and the cube when its first fresh column is at or
after the end of the light circuit's locals. -/
theorem Facts.lightFits
    (facts : Facts program relation ajtai lightFresh refreshFresh lightBatches refreshBatches)
    (afterLocals : LightAssembler.sharedEnd StepRelation.base program.witnessWordCount +
      localLength (lightOps program relation) ≤ lightFresh) :
    CircuitRelation.Fits CircuitRelation.publicColumns (lightPieces program relation ajtai)
      lightFresh sourceWidth :=
  RelationFits.fits_of_capacity (lightPieces_packable program relation ajtai)
    (lightPieces_below program relation ajtai afterLocals) (by decide) facts.lightColumns
    facts.lightRows

/-- The refresh package fits its source width and the cube when its first fresh column is at or
after the end of the refresh circuit's locals. -/
theorem Facts.refreshFits
    (facts : Facts program relation ajtai lightFresh refreshFresh lightBatches refreshBatches)
    (afterLocals : RefreshAssembler.sharedEnd StepRelation.base program.witnessWordCount +
      localLength (refreshOps program relation) ≤ refreshFresh) :
    CircuitRelation.Fits CircuitRelation.publicColumns (refreshPieces program relation ajtai)
      refreshFresh sourceWidth :=
  RelationFits.fits_of_capacity (refreshPieces_packable program relation ajtai)
    (refreshPieces_below program relation ajtai afterLocals) (by decide) facts.refreshColumns
    facts.refreshRows

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

/-- The bundle of the two packages, when each first fresh column is at or after the end of its
circuit's locals. The keys must carry their relations. -/
def Facts.exports
    (facts : Facts program relation ajtai lightFresh refreshFresh lightBatches refreshBatches)
    (lightAfterLocals : LightAssembler.sharedEnd StepRelation.base program.witnessWordCount +
      localLength (lightOps program relation) ≤ lightFresh)
    (refreshAfterLocals : RefreshAssembler.sharedEnd StepRelation.base program.witnessWordCount +
      localLength (refreshOps program relation) ≤ refreshFresh)
    (lightKey : (circuitKeys 0).relation =
      (CircuitRelation.plan _ _ lightFresh sourceWidth (facts.lightFits lightAfterLocals)
        facts.lightLayout).logicalRelation (publicFits := publicFits) cubeFits)
    (refreshKey : (circuitKeys 1).relation =
      (CircuitRelation.plan _ _ refreshFresh sourceWidth (facts.refreshFits refreshAfterLocals)
        facts.refreshLayout).logicalRelation (publicFits := publicFits) cubeFits) :
    TerminalBridge.Exports relation ajtai circuitKeys program.step cubeFits where
  program := program
  programStep := rfl
  lightFresh := lightFresh
  refreshFresh := refreshFresh
  lightWidth := sourceWidth
  refreshWidth := sourceWidth
  lightFitsPackage := facts.lightFits lightAfterLocals
  refreshFitsPackage := facts.refreshFits refreshAfterLocals
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
    (lightAfterLocals : LightAssembler.sharedEnd StepRelation.base program.witnessWordCount +
      localLength (lightOps program relation) ≤ lightFresh)
    (refreshAfterLocals : RefreshAssembler.sharedEnd StepRelation.base program.witnessWordCount +
      localLength (refreshOps program relation) ≤ refreshFresh)
    (lightKey : (circuitKeys 0).relation =
      (CircuitRelation.plan _ _ lightFresh sourceWidth (facts.lightFits lightAfterLocals)
        facts.lightLayout).logicalRelation (publicFits := publicFits) cubeFits)
    (refreshKey : (circuitKeys 1).relation =
      (CircuitRelation.plan _ _ refreshFresh sourceWidth (facts.refreshFits refreshAfterLocals)
        facts.refreshLayout).logicalRelation (publicFits := publicFits) cubeFits)
    (lightPublicBelow : CircuitRelation.publicColumns ≤ lightFresh)
    (refreshPublicBelow : CircuitRelation.publicColumns ≤ refreshFresh) :
    StepExtend.Complete (facts.exports lightAfterLocals refreshAfterLocals lightKey refreshKey) where
  lightBatches := lightBatches
  lightFitsKeyWidth := facts.lightFitsKeyWidth
  lightLayout := by dsimp only [Facts.exports]; rfl
  lightFreshAfterLocals := lightAfterLocals
  lightPublicBelow := lightPublicBelow
  refreshBatches := refreshBatches
  refreshFitsKeyWidth := facts.refreshFitsKeyWidth
  refreshLayout := by dsimp only [Facts.exports]; rfl
  refreshFreshAfterLocals := refreshAfterLocals
  refreshPublicBelow := refreshPublicBelow

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
      (CircuitRelation.plan _ _ (lightFresh program relation) sourceWidth
        (facts.lightFits (Nat.le_refl _)) facts.lightLayout).logicalRelation
          (publicFits := publicFits) cubeFits)
    (refreshKey : (circuitKeys 1).relation =
      (CircuitRelation.plan _ _ (refreshFresh program relation) sourceWidth
        (facts.refreshFits (Nat.le_refl _)) facts.refreshLayout).logicalRelation
          (publicFits := publicFits) cubeFits) :
    StepExtend.Complete (facts.exports (Nat.le_refl _) (Nat.le_refl _) lightKey refreshKey) :=
  Facts.complete facts (Nat.le_refl _) (Nat.le_refl _) lightKey refreshKey
    (publicColumns_le_lightFresh program relation) (publicColumns_le_refreshFresh program relation)

end NightstreamFPrime.Export.BoundedDepth.EmittedPackages
