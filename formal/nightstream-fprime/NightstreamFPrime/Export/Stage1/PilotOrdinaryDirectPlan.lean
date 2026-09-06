import NightstreamFPrime.Export.Stage1.PilotOrdinaryRetainedGeometry
import NightstreamFPrime.Export.Stage1.RunningTransitionDirectPlan

/-!
Owns the executable source resolver and direct 14-matrix plan for the exact
1,330 non-Poseidon pilot rows. The resolver is fixed by the Lean source
support proof and the canonical lifted pilot environment.
-/

namespace NightstreamFPrime.Export.Stage1.PilotOrdinaryDirectPlan

open NightstreamFPrime.Circuit
open NightstreamFPrime.Export.Package
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Layout.Stage1
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra
open PilotOrdinaryDirectSource

def piCcsGeometry {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat}
    (geometry : PilotOrdinaryRetainedGeometry.Geometry program logicalWidth) :
    PiCCSOrdinaryRetainedGeometry.Geometry program logicalWidth :=
  PilotOrdinaryRetainedGeometry.prefixGeometry geometry

def oneColumn {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat}
    (geometry : PilotOrdinaryRetainedGeometry.Geometry program logicalWidth) :
    Fin logicalWidth :=
  PilotOrdinaryRetainedGeometry.oneColumn geometry

def finalSlot (lane : Fin 4) : Fin 592 :=
  ⟨584 + lane.val, by have bound := lane.isLt; omega⟩

inductive Location where
  | priorDigest (lane : Fin 4)
  | priorPublic (index : Fin 270)
  | canonicalLocal (index : Fin 264)
  | outputState (lane : Fin 4)
  | canonicalFresh (index : Fin 788)
  | outputDigest (lane : Fin 4)

namespace Location

def sourceColumn : Location → Nat
  | .priorDigest lane => PilotProduction.priorDigestStart + lane.val
  | .priorPublic index => PilotProduction.priorPublicInputStart + index.val
  | .canonicalLocal index =>
      PriorStateHash.hashEnd PilotProduction.priorInterface
        PilotProduction.witnessOffset + index.val
  | .outputState lane => PilotProduction.lifecycleOutputOffset +
      PilotValues.absorbCount * 592 + 584 + lane.val
  | .canonicalFresh index => PilotValues.logicalColumnCount + index.val
  | .outputDigest lane => PilotProduction.outputDigestStart + lane.val

theorem physicalSupport (location : Location) :
    PhysicalSource location.sourceColumn := by
  cases location with
  | priorDigest lane =>
      apply Or.inl
      apply LogicalSource.priorDigest
      change InRange PilotProduction.priorDigestStart 4
        (PilotProduction.priorDigestStart + lane.val)
      unfold InRange
      have bound := lane.isLt
      omega
  | priorPublic index =>
      apply Or.inl
      apply LogicalSource.priorPublic
      change InRange PilotProduction.priorPublicInputStart 270
        (PilotProduction.priorPublicInputStart + index.val)
      unfold InRange
      have bound := index.isLt
      omega
  | canonicalLocal index =>
      apply Or.inl
      apply LogicalSource.canonicalLocal
      change InRange
        (PriorStateHash.hashEnd PilotProduction.priorInterface
          PilotProduction.witnessOffset) 264
        (PriorStateHash.hashEnd PilotProduction.priorInterface
          PilotProduction.witnessOffset + index.val)
      unfold InRange
      have bound := index.isLt
      omega
  | outputState lane =>
      apply Or.inl
      apply LogicalSource.outputState
      change InRange
        (PilotProduction.lifecycleOutputOffset +
          PilotValues.absorbCount * 592 + 584) 4
        (PilotProduction.lifecycleOutputOffset +
          PilotValues.absorbCount * 592 + 584 + lane.val)
      unfold InRange
      have bound := lane.isLt
      omega
  | canonicalFresh index =>
      apply Or.inr
      constructor
      · change PilotValues.logicalColumnCount ≤
          PilotValues.logicalColumnCount + index.val
        omega
      · have bound := index.isLt
        change index.val < 788 at bound
        change PilotValues.logicalColumnCount + index.val <
          PilotValues.sourceColumnCount
        norm_num [PilotValues.logicalColumnCount,
          PilotValues.sourceColumnCount, PilotValues.externalColumnCount,
          PilotValues.outputDigestStart, PilotValues.outputPreimageStart,
          PilotValues.priorPublicInputStart, PilotValues.priorPreimageStart,
          PilotValues.stateHashWords, PilotValues.stateHashBaseWords,
          PilotValues.hashWitnessCount, PilotValues.absorbCount,
          PilotValues.permutationRecipeCount, Spec.Poseidon2.rate,
          PilotValues.priorCanonicalPrivateCount,
          PilotValues.priorCanonicalFreshCount] at bound ⊢
        omega
  | outputDigest lane =>
      apply Or.inl
      apply LogicalSource.outputDigest
      change InRange PilotProduction.outputDigestStart 4
        (PilotProduction.outputDigestStart + lane.val)
      unfold InRange
      have bound := lane.isLt
      omega

theorem sourceColumn_lt (location : Location) :
    location.sourceColumn < PilotSpartan.SourceColumnCount := by
  simpa [PilotSpartan.SourceColumnCount] using
    physicalSource_lt location.sourceColumn location.physicalSupport

theorem targetColumn_lt (location : Location) :
    PilotSpartan.sourceToSpartan location.sourceColumn <
      PilotSpartan.spartanColumnCount :=
  PilotSpartan.sourceToSpartan_lt location.sourceColumn location.sourceColumn_lt

theorem stage1Map (location : Location) :
    Spartan.sourceToSpartan location.sourceColumn =
      Spartan.liftPilotColumn
        (PilotSpartan.sourceToSpartan location.sourceColumn) := by
  unfold Spartan.sourceToSpartan
  rw [if_pos]
  have bound := location.sourceColumn_lt
  rw [PilotSpartan.sourceColumnCount_eq] at bound
  simpa [Spartan.pilotSourceColumnCount] using bound

def form {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat}
    (geometry : PilotOrdinaryRetainedGeometry.Geometry program logicalWidth) :
    Location → SparseForm logicalWidth
  | .priorDigest lane =>
      (PiCCSOrdinaryRetainedBlocks.priorLastBlock program).form
        (PiCCSOrdinaryRetainedGeometry.priorLastStart program)
        (PiCCSOrdinaryRetainedGeometry.priorLastFits (piCcsGeometry geometry))
        (finalSlot lane)
  | .priorPublic index =>
      (PiCCSOrdinaryRetainedBlocks.freshPublicInputBlock program).form
        (PiCCSOrdinaryRetainedGeometry.freshPublicInputStart program)
        (PiCCSOrdinaryRetainedGeometry.freshPublicInputFits
          (piCcsGeometry geometry)) index
  | .canonicalLocal index =>
      (PilotOrdinaryRetainedBlocks.canonicalLocalBlock program).form
        (PilotOrdinaryRetainedGeometry.canonicalLocalStart program)
        (PilotOrdinaryRetainedGeometry.canonicalLocalFits geometry) index
  | .outputState lane =>
      (PiCCSOrdinaryRetainedBlocks.outputLastBlock program).form
        (PiCCSOrdinaryRetainedGeometry.outputLastStart program)
        (PiCCSOrdinaryRetainedGeometry.outputLastFits (piCcsGeometry geometry))
        (finalSlot lane)
  | .canonicalFresh index =>
      (PilotOrdinaryRetainedBlocks.canonicalFreshBlock program).form
        (PilotOrdinaryRetainedGeometry.canonicalFreshStart program)
        (PilotOrdinaryRetainedGeometry.canonicalFreshFits geometry) index
  | .outputDigest lane =>
      (PilotOrdinaryRetainedBlocks.outputDigestBlock program).form
        (PilotOrdinaryRetainedGeometry.outputDigestStart program)
        (PilotOrdinaryRetainedGeometry.outputDigestFits geometry) lane

end Location

def rangeIndex {start count column : Nat} (inside : InRange start count column) :
    Fin count :=
  ⟨column - start, by unfold InRange at inside; omega⟩

@[simp] theorem rangeIndex_source {start count column : Nat}
    (inside : InRange start count column) :
    start + (rangeIndex inside).val = column := by
  change start + (column - start) = column
  unfold InRange at inside
  omega

local instance (start count column : Nat) : Decidable (InRange start count column) :=
  by unfold InRange; infer_instance

structure Located (source : Nat) where
  location : Location
  owns : location.sourceColumn = source

def classifySource (source : Nat) : Option (Located source) :=
  if priorDigest : InRange PilotProduction.priorDigestStart 4 source then
    some ⟨.priorDigest (rangeIndex priorDigest), by
      rw [Location.sourceColumn, rangeIndex_source priorDigest]⟩
  else if priorPublic : InRange PilotProduction.priorPublicInputStart 270 source then
    some ⟨.priorPublic (rangeIndex priorPublic), by
      rw [Location.sourceColumn, rangeIndex_source priorPublic]⟩
  else if canonicalLocal : InRange
      (PriorStateHash.hashEnd PilotProduction.priorInterface
        PilotProduction.witnessOffset) 264 source then
    some ⟨.canonicalLocal (rangeIndex canonicalLocal), by
      rw [Location.sourceColumn, rangeIndex_source canonicalLocal]⟩
  else if outputState : InRange
      (PilotProduction.lifecycleOutputOffset +
        PilotValues.absorbCount * 592 + 584) 4 source then
    some ⟨.outputState (rangeIndex outputState), by
      rw [Location.sourceColumn, rangeIndex_source outputState]⟩
  else if canonicalFresh : InRange PilotValues.logicalColumnCount 788 source then
    some ⟨.canonicalFresh (rangeIndex canonicalFresh), by
      rw [Location.sourceColumn, rangeIndex_source canonicalFresh]⟩
  else if outputDigest : InRange PilotProduction.outputDigestStart 4 source then
    some ⟨.outputDigest (rangeIndex outputDigest), by
      rw [Location.sourceColumn, rangeIndex_source outputDigest]⟩
  else none

private theorem fresh_inRange {source : Nat}
    (fresh : PilotValues.logicalColumnCount ≤ source ∧
      source < PilotValues.sourceColumnCount) :
    InRange PilotValues.logicalColumnCount 788 source := by
  unfold InRange
  have endEq : PilotValues.sourceColumnCount =
      PilotValues.logicalColumnCount + 788 := by rfl
  omega

theorem classifySource_complete {source : Nat}
    (support : PhysicalSource source) : (classifySource source).isSome := by
  unfold classifySource
  split
  · rfl
  split
  · rfl
  split
  · rfl
  split
  · rfl
  split
  · rfl
  split
  · rfl
  · rename_i noPriorDigest noPriorPublic noCanonicalLocal noOutputState
      noCanonicalFresh noOutputDigest
    exfalso
    rcases support with logical | fresh
    · rcases logical with priorDigest | priorPublic | canonicalLocal |
        outputState | outputDigest
      · exact noPriorDigest priorDigest
      · exact noPriorPublic priorPublic
      · exact noCanonicalLocal canonicalLocal
      · exact noOutputState outputState
      · exact noOutputDigest outputDigest
    · exact noCanonicalFresh (fresh_inRange fresh)

structure Decoded where
  source : Nat
  location : Location
  owns : location.sourceColumn = source

def classifyTarget (column : Nat) : Option Decoded :=
  match PilotSpartan.spartanToSource column with
  | none => none
  | some source =>
      match classifySource source with
      | none => none
      | some located => some ⟨source, located.location, located.owns⟩

theorem classifyTarget_complete {column : Nat} (support : Target column) :
    ∃ decoded, classifyTarget column = some decoded ∧
      PilotSpartan.sourceToSpartan decoded.source = column := by
  rcases support with ⟨source, sourceSupport, rfl⟩
  have bound := physicalSource_lt source sourceSupport
  have inverse := PilotSpartan.spartanToSource_sourceToSpartan source (by
    simpa [PilotSpartan.SourceColumnCount] using bound)
  have complete := classifySource_complete sourceSupport
  cases found : classifySource source with
  | none => simp [found] at complete
  | some located =>
      refine ⟨⟨source, located.location, located.owns⟩, ?_, rfl⟩
      simp [classifyTarget, inverse, found]

def sourceMap {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat}
    (geometry : PilotOrdinaryRetainedGeometry.Geometry program logicalWidth) :
    SourceCompiler.SourceMap PilotSpartan.spartanColumnCount logicalWidth where
  form := fun column =>
    match classifyTarget column.val with
    | none => .empty
    | some decoded => decoded.location.form geometry

def pilotEnv (program : Lifecycle.Stage1.Application.Program)
    (base : Fin (PiRLCProductPlan.baseSourceWidth program) → F) : Env :=
  fun column => RunningTransitionDirectPlan.transitionEnv program base
    (Spartan.liftPilotColumn column)

structure Encodes {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat}
    (geometry : PilotOrdinaryRetainedGeometry.Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth)
    (base : Fin (PiRLCProductPlan.baseSourceWidth program) → F)
    (groupValue : Fin PiRLCProductSchedule.invocationCount → Fin 33 → F)
    (products : Fin PiRLCFirst54DirectSchedule.candidateCount → F) : Prop where
  prior : PiCCSOrdinaryRetainedGeometry.Encodes (piCcsGeometry geometry)
    assignment (PiRLCRetainedPreservation.sourceAssignment
      program base groupValue products)
  added : PilotOrdinaryRetainedGeometry.Encodes geometry assignment
    (PiRLCRetainedPreservation.sourceAssignment
      program base groupValue products)

theorem Location.stage1SourceColumn_lt (location : Location) :
    location.sourceColumn < Spartan.SourceColumnCount := by
  have bound := location.sourceColumn_lt
  rw [PilotSpartan.sourceColumnCount_eq] at bound
  rw [Spartan.sourceColumnCount_eq]
  omega

theorem sourceAssignment_at
    {program : Lifecycle.Stage1.Application.Program}
    (base : Fin (PiRLCProductPlan.baseSourceWidth program) → F)
    (groupValue : Fin PiRLCProductSchedule.invocationCount → Fin 33 → F)
    (products : Fin PiRLCFirst54DirectSchedule.candidateCount → F)
    (location : Location) :
    PiRLCRetainedPreservation.sourceAssignment program base groupValue products
        (RunningTransitionRetainedBlocks.packageSourceColumn program
          location.sourceColumn location.stage1SourceColumn_lt) =
      pilotEnv program base
        (PilotSpartan.sourceToSpartan location.sourceColumn) := by
  rw [RunningTransitionDirectPlan.sourceAssignment_packageSource
    program base groupValue products location.sourceColumn
    location.stage1SourceColumn_lt]
  rw [← RunningTransitionDirectPlan.transitionEnv_of_outside program base
    location.sourceColumn location.stage1SourceColumn_lt (by
      left
      have bound := location.sourceColumn_lt
      rw [PilotSpartan.sourceColumnCount_eq] at bound
      rw [PiCCSInputs.phaseOffset_eq]
      omega)]
  unfold pilotEnv
  rw [location.stage1Map]

private theorem priorLastWitnessStart_eq :
    PoseidonRetainedBlock.priorWitnessStart
      PiCCSOrdinaryRetainedBlocks.priorLastInvocation = 7438682 := by rfl

private theorem outputLastWitnessStart_eq :
    PoseidonRetainedBlock.outputWitnessStart
      PiCCSOrdinaryRetainedBlocks.outputLastInvocation = 14750146 := by rfl

private theorem liftPriorDigestTarget (lane : Fin 4) :
    Spartan.liftPilotColumn (7409978 + lane.val) = 7439266 + lane.val := by
  have laneBound := lane.isLt
  unfold Spartan.liftPilotColumn
  rw [if_neg (by norm_num [Spartan.pilotInputPrivateColumnCount]; omega)]
  rw [if_pos (by norm_num [Spartan.pilotPrivateColumnCount]; omega)]
  norm_num [Spartan.proofInputColumnCount]
  omega

private theorem liftOutputStateTarget (lane : Fin 4) :
    Spartan.liftPilotColumn (14721442 + lane.val) = 14750730 + lane.val := by
  have laneBound := lane.isLt
  unfold Spartan.liftPilotColumn
  rw [if_neg (by norm_num [Spartan.pilotInputPrivateColumnCount]; omega)]
  rw [if_pos (by norm_num [Spartan.pilotPrivateColumnCount]; omega)]
  norm_num [Spartan.proofInputColumnCount]
  omega

private theorem priorFinalColumn_eq (lane : Fin 4) :
    PoseidonRetainedBlock.priorWitnessStart
          PiCCSOrdinaryRetainedBlocks.priorLastInvocation +
        (finalSlot lane).val =
      Spartan.sourceToSpartan (Location.priorDigest lane).sourceColumn := by
  have sourceBound : (Location.priorDigest lane).sourceColumn <
      Spartan.pilotSourceColumnCount := by
    have bound := (Location.priorDigest lane).sourceColumn_lt
    rw [PilotSpartan.sourceColumnCount_eq] at bound
    simpa [Spartan.pilotSourceColumnCount] using bound
  unfold Spartan.sourceToSpartan
  rw [if_pos sourceBound]
  change _ = Spartan.liftPilotColumn
    (PilotSpartan.sourceToSpartan
      (PilotProduction.priorDigestStart + lane.val))
  rw [PilotOrdinaryDirectSource.priorDigest_targetColumn,
    liftPriorDigestTarget, priorLastWitnessStart_eq]
  change 7438682 + (584 + lane.val) = 7439266 + lane.val
  omega

private theorem outputFinalColumn_eq (lane : Fin 4) :
    PoseidonRetainedBlock.outputWitnessStart
          PiCCSOrdinaryRetainedBlocks.outputLastInvocation +
        (finalSlot lane).val =
      Spartan.sourceToSpartan (Location.outputState lane).sourceColumn := by
  have sourceBound : (Location.outputState lane).sourceColumn <
      Spartan.pilotSourceColumnCount := by
    have bound := (Location.outputState lane).sourceColumn_lt
    rw [PilotSpartan.sourceColumnCount_eq] at bound
    simpa [Spartan.pilotSourceColumnCount] using bound
  unfold Spartan.sourceToSpartan
  rw [if_pos sourceBound]
  change _ = Spartan.liftPilotColumn
    (PilotSpartan.sourceToSpartan
      (PilotProduction.lifecycleOutputOffset +
        PilotValues.absorbCount * 592 + 584 + lane.val))
  rw [PilotOrdinaryDirectSource.outputState_targetColumn,
    liftOutputStateTarget, outputLastWitnessStart_eq]
  change 14750146 + (584 + lane.val) = 14750730 + lane.val
  omega

private theorem priorFinalColumn_bound (lane : Fin 4) :
    PoseidonRetainedBlock.priorWitnessStart
          PiCCSOrdinaryRetainedBlocks.priorLastInvocation +
        (finalSlot lane).val <
      PerApplicationPackage.basePackage.layout.totalColumnCount := by
  have laneBound := (finalSlot lane).isLt
  have total : PerApplicationPackage.basePackage.layout.totalColumnCount =
      29336725 := Package.circuitPackage_layout_values.2.2.2.2
  rw [priorLastWitnessStart_eq, total]
  change 7438682 + (584 + lane.val) < 29336725
  omega

private theorem outputFinalColumn_bound (lane : Fin 4) :
    PoseidonRetainedBlock.outputWitnessStart
          PiCCSOrdinaryRetainedBlocks.outputLastInvocation +
        (finalSlot lane).val <
      PerApplicationPackage.basePackage.layout.totalColumnCount := by
  have laneBound := (finalSlot lane).isLt
  have total : PerApplicationPackage.basePackage.layout.totalColumnCount =
      29336725 := Package.circuitPackage_layout_values.2.2.2.2
  rw [outputLastWitnessStart_eq, total]
  change 14750146 + (584 + lane.val) < 29336725
  omega

private theorem priorLastBlock_source
    (program : Lifecycle.Stage1.Application.Program) (lane : Fin 4) :
    (PiCCSOrdinaryRetainedBlocks.priorLastBlock program).source
        (finalSlot lane) =
      RunningTransitionRetainedBlocks.packageSourceColumn program
        (Location.priorDigest lane).sourceColumn
        (Location.priorDigest lane).stage1SourceColumn_lt := by
  unfold PiCCSOrdinaryRetainedBlocks.priorLastBlock
    PiCCSOrdinaryRetainedBlocks.packageFieldBlock
  change
    PiCCSOrdinaryRetainedBlocks.packageSourceColumn program
        (PoseidonRetainedBlock.priorWitnessStart
            PiCCSOrdinaryRetainedBlocks.priorLastInvocation +
          (finalSlot lane).val) (priorFinalColumn_bound lane) = _
  unfold PiCCSOrdinaryRetainedBlocks.packageSourceColumn
    RunningTransitionRetainedBlocks.packageSourceColumn
  apply congrArg (PiRLCRetainedPreservation.baseSourceColumn program)
  apply Fin.ext
  unfold PiRLCProductPlan.shiftedPackageColumn
  exact congrArg (PerApplicationPackage.shiftColumn program)
    (priorFinalColumn_eq lane)

private theorem outputLastBlock_source
    (program : Lifecycle.Stage1.Application.Program) (lane : Fin 4) :
    (PiCCSOrdinaryRetainedBlocks.outputLastBlock program).source
        (finalSlot lane) =
      RunningTransitionRetainedBlocks.packageSourceColumn program
        (Location.outputState lane).sourceColumn
        (Location.outputState lane).stage1SourceColumn_lt := by
  unfold PiCCSOrdinaryRetainedBlocks.outputLastBlock
    PiCCSOrdinaryRetainedBlocks.packageFieldBlock
  change
    PiCCSOrdinaryRetainedBlocks.packageSourceColumn program
        (PoseidonRetainedBlock.outputWitnessStart
            PiCCSOrdinaryRetainedBlocks.outputLastInvocation +
          (finalSlot lane).val) (outputFinalColumn_bound lane) = _
  unfold PiCCSOrdinaryRetainedBlocks.packageSourceColumn
    RunningTransitionRetainedBlocks.packageSourceColumn
  apply congrArg (PiRLCRetainedPreservation.baseSourceColumn program)
  apply Fin.ext
  unfold PiRLCProductPlan.shiftedPackageColumn
  exact congrArg (PerApplicationPackage.shiftColumn program)
    (outputFinalColumn_eq lane)

theorem Location.form_eval
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : PilotOrdinaryRetainedGeometry.Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth)
    (base : Fin (PiRLCProductPlan.baseSourceWidth program) → F)
    (groupValue : Fin PiRLCProductSchedule.invocationCount → Fin 33 → F)
    (products : Fin PiRLCFirst54DirectSchedule.candidateCount → F)
    (encodes : Encodes geometry assignment base groupValue products)
    (location : Location) :
    (location.form geometry).eval assignment =
      pilotEnv program base
        (PilotSpartan.sourceToSpartan location.sourceColumn) := by
  cases location with
  | priorDigest lane =>
      rw [form, LowNormBlock.Block.form_eval
        (PiCCSOrdinaryRetainedBlocks.priorLastBlock program)
        (PiCCSOrdinaryRetainedGeometry.priorLastStart program)
        (PiCCSOrdinaryRetainedGeometry.priorLastFits (piCcsGeometry geometry))
        assignment (PiRLCRetainedPreservation.sourceAssignment
          program base groupValue products) encodes.prior.priorLast
        (finalSlot lane)]
      rw [priorLastBlock_source program lane]
      change PiRLCRetainedPreservation.sourceAssignment program base groupValue products
          (RunningTransitionRetainedBlocks.packageSourceColumn program
            (Location.priorDigest lane).sourceColumn
            (Location.priorDigest lane).stage1SourceColumn_lt) = _
      exact sourceAssignment_at base groupValue products (.priorDigest lane)
  | priorPublic index =>
      rw [form, LowNormBlock.Block.form_eval
        (PiCCSOrdinaryRetainedBlocks.freshPublicInputBlock program)
        (PiCCSOrdinaryRetainedGeometry.freshPublicInputStart program)
        (PiCCSOrdinaryRetainedGeometry.freshPublicInputFits
          (piCcsGeometry geometry)) assignment
        (PiRLCRetainedPreservation.sourceAssignment
          program base groupValue products) encodes.prior.freshPublicInput index]
      change PiRLCRetainedPreservation.sourceAssignment
          program base groupValue products
          (RunningTransitionRetainedBlocks.packageSourceColumn program
            (Location.priorPublic index).sourceColumn
            (Location.priorPublic index).stage1SourceColumn_lt) = _
      exact sourceAssignment_at base groupValue products (.priorPublic index)
  | canonicalLocal index =>
      rw [form, LowNormBlock.Block.form_eval
        (PilotOrdinaryRetainedBlocks.canonicalLocalBlock program)
        (PilotOrdinaryRetainedGeometry.canonicalLocalStart program)
        (PilotOrdinaryRetainedGeometry.canonicalLocalFits geometry) assignment
        (PiRLCRetainedPreservation.sourceAssignment
          program base groupValue products) encodes.added.canonicalLocal index]
      change PiRLCRetainedPreservation.sourceAssignment
          program base groupValue products
          (RunningTransitionRetainedBlocks.packageSourceColumn program
            (Location.canonicalLocal index).sourceColumn
            (Location.canonicalLocal index).stage1SourceColumn_lt) = _
      exact sourceAssignment_at base groupValue products (.canonicalLocal index)
  | outputState lane =>
      rw [form, LowNormBlock.Block.form_eval
        (PiCCSOrdinaryRetainedBlocks.outputLastBlock program)
        (PiCCSOrdinaryRetainedGeometry.outputLastStart program)
        (PiCCSOrdinaryRetainedGeometry.outputLastFits (piCcsGeometry geometry))
        assignment (PiRLCRetainedPreservation.sourceAssignment
          program base groupValue products) encodes.prior.outputLast
        (finalSlot lane)]
      rw [outputLastBlock_source program lane]
      change PiRLCRetainedPreservation.sourceAssignment program base groupValue products
          (RunningTransitionRetainedBlocks.packageSourceColumn program
            (Location.outputState lane).sourceColumn
            (Location.outputState lane).stage1SourceColumn_lt) = _
      exact sourceAssignment_at base groupValue products (.outputState lane)
  | canonicalFresh index =>
      rw [form, LowNormBlock.Block.form_eval
        (PilotOrdinaryRetainedBlocks.canonicalFreshBlock program)
        (PilotOrdinaryRetainedGeometry.canonicalFreshStart program)
        (PilotOrdinaryRetainedGeometry.canonicalFreshFits geometry) assignment
        (PiRLCRetainedPreservation.sourceAssignment
          program base groupValue products) encodes.added.canonicalFresh index]
      change PiRLCRetainedPreservation.sourceAssignment
          program base groupValue products
          (RunningTransitionRetainedBlocks.packageSourceColumn program
            (Location.canonicalFresh index).sourceColumn
            (Location.canonicalFresh index).stage1SourceColumn_lt) = _
      exact sourceAssignment_at base groupValue products (.canonicalFresh index)
  | outputDigest lane =>
      rw [form, LowNormBlock.Block.form_eval
        (PilotOrdinaryRetainedBlocks.outputDigestBlock program)
        (PilotOrdinaryRetainedGeometry.outputDigestStart program)
        (PilotOrdinaryRetainedGeometry.outputDigestFits geometry) assignment
        (PiRLCRetainedPreservation.sourceAssignment
          program base groupValue products) encodes.added.outputDigest lane]
      change PiRLCRetainedPreservation.sourceAssignment
          program base groupValue products
          (RunningTransitionRetainedBlocks.packageSourceColumn program
            (Location.outputDigest lane).sourceColumn
            (Location.outputDigest lane).stage1SourceColumn_lt) = _
      exact sourceAssignment_at base groupValue products (.outputDigest lane)

theorem priorDigest_form_eval_chainOutput
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : PilotOrdinaryRetainedGeometry.Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth)
    (env : Env)
    (values : ∀ lane : Fin 4,
      ((Location.priorDigest lane).form geometry).eval assignment =
        env (PilotSpartan.sourceToSpartan (Location.priorDigest lane).sourceColumn))
    (lane : Fin 4) :
    ((Location.priorDigest lane).form geometry).eval assignment =
      NightstreamFPrime.Export.Pilot.chainOutputState PilotData.priorChain
        PilotData.priorChain.absorbCount env
        ⟨lane.val, Nat.lt_trans lane.isLt (by decide)⟩ := by
  rw [values lane]
  unfold NightstreamFPrime.Export.Pilot.chainOutputState
    NightstreamFPrime.Export.Package.invocationLocalStart
  change env
      (PilotSpartan.sourceToSpartan
        (PilotProduction.priorDigestStart + lane.val)) = _
  rw [PilotOrdinaryDirectSource.priorDigest_targetColumn]
  norm_num [PilotData.circuitPackage, PilotData.permutationTemplate,
    PilotData.priorChain, PilotData.priorWitnessStart,
    Spec.Poseidon2.rate, PilotValues.absorbCount,
    PilotValues.stateHashWords, PilotValues.stateHashBaseWords]

theorem outputState_form_eval_chainOutput
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : PilotOrdinaryRetainedGeometry.Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth)
    (env : Env)
    (values : ∀ lane : Fin 4,
      ((Location.outputState lane).form geometry).eval assignment =
        env (PilotSpartan.sourceToSpartan (Location.outputState lane).sourceColumn))
    (lane : Fin 4) :
    ((Location.outputState lane).form geometry).eval assignment =
      NightstreamFPrime.Export.Pilot.chainOutputState PilotData.outputChain
        PilotData.outputChain.absorbCount env
        ⟨lane.val, Nat.lt_trans lane.isLt (by decide)⟩ := by
  rw [values lane]
  unfold NightstreamFPrime.Export.Pilot.chainOutputState
    NightstreamFPrime.Export.Package.invocationLocalStart
  change env
      (PilotSpartan.sourceToSpartan
        (PilotProduction.lifecycleOutputOffset +
          PilotValues.absorbCount * 592 + 584 + lane.val)) = _
  rw [PilotOrdinaryDirectSource.outputState_targetColumn]
  norm_num [PilotData.circuitPackage, PilotData.permutationTemplate,
    PilotData.outputChain, PilotData.outputWitnessStart,
    Spec.Poseidon2.rate, PilotValues.absorbCount,
    PilotValues.stateHashWords, PilotValues.stateHashBaseWords,
    PilotValues.outputWitnessStart, PilotValues.witnessPrivateStart,
    PilotValues.hashWitnessCount, PilotValues.priorCanonicalPrivateCount]

theorem sourceMap_form_eval_of_target
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : PilotOrdinaryRetainedGeometry.Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth)
    (base : Fin (PiRLCProductPlan.baseSourceWidth program) → F)
    (groupValue : Fin PiRLCProductSchedule.invocationCount → Fin 33 → F)
    (products : Fin PiRLCFirst54DirectSchedule.candidateCount → F)
    (encodes : Encodes geometry assignment base groupValue products)
    (column : Fin PilotSpartan.spartanColumnCount)
    (support : Target column.val) :
    ((sourceMap geometry).form column).eval assignment =
      pilotEnv program base column.val := by
  rcases classifyTarget_complete support with ⟨decoded, found, mapped⟩
  change (match classifyTarget column.val with
    | none => SparseForm.empty
    | some value => value.location.form geometry).eval assignment = _
  rw [found]
  rw [Location.form_eval geometry assignment base groupValue products encodes
    decoded.location]
  have mappedLocation :
      PilotSpartan.sourceToSpartan decoded.location.sourceColumn =
        column.val := by
    rw [decoded.owns, mapped]
  rw [mappedLocation]

private theorem preservesCombination
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : PilotOrdinaryRetainedGeometry.Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth)
    (base : Fin (PiRLCProductPlan.baseSourceWidth program) → F)
    (groupValue : Fin PiRLCProductSchedule.invocationCount → Fin 33 → F)
    (products : Fin PiRLCFirst54DirectSchedule.candidateCount → F)
    (encodes : Encodes geometry assignment base groupValue products)
    (combination : R1CS.LinearCombination)
    (bounded : SourceCompiler.CombinationBounded
      PilotSpartan.spartanColumnCount combination)
    (scope : combination.VarsSatisfy Target) :
    OrdinarySourcePlan.SourceMap.PreservesCombination (sourceMap geometry)
      assignment (pilotEnv program base) combination bounded := by
  intro term member
  exact sourceMap_form_eval_of_target geometry assignment base groupValue
    products encodes ⟨term.1, bounded term member⟩ (scope term member)

private theorem programRow_support
    (index : Fin 1330) :
    (PilotOrdinaryDirectSource.programRow index).VarsSatisfy Target := by
  exact sourceRows_varsSatisfy _
    (List.get_mem _ (Fin.cast sourceRows_length.symm index))

def inputs
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : PilotOrdinaryRetainedGeometry.Geometry program logicalWidth) :
    PilotOrdinaryDirectSource.program.Inputs logicalWidth where
  oneColumn := oneColumn geometry
  sourceMap := fun _ => sourceMap geometry

theorem inputs_preserve
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : PilotOrdinaryRetainedGeometry.Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth)
    (base : Fin (PiRLCProductPlan.baseSourceWidth program) → F)
    (groupValue : Fin PiRLCProductSchedule.invocationCount → Fin 33 → F)
    (products : Fin PiRLCFirst54DirectSchedule.candidateCount → F)
    (encodes : Encodes geometry assignment base groupValue products) :
    ∀ index, OrdinarySourcePlan.SourceMap.PreservesRow
      ((inputs geometry).sourceMap index) assignment (pilotEnv program base)
      (PilotOrdinaryDirectSource.program.row index)
      (PilotOrdinaryDirectSource.program.bounded index) := by
  intro index
  have directScope := programRow_support index
  have scope :
      (PilotOrdinaryDirectSource.program.row index).VarsSatisfy Target := by
    simpa only [PilotOrdinaryDirectSource.program,
      PilotOrdinaryDirectSource.SupportedProgram.toProgram,
      PilotOrdinaryDirectSource.supportedProgram] using directScope
  exact ⟨
    preservesCombination geometry assignment base groupValue products encodes
      _ _ scope.1,
    preservesCombination geometry assignment base groupValue products encodes
      _ _ scope.2.1,
    preservesCombination geometry assignment base groupValue products encodes
      _ _ scope.2.2⟩

theorem programRow_preserve
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : PilotOrdinaryRetainedGeometry.Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth)
    (base : Fin (PiRLCProductPlan.baseSourceWidth program) → F)
    (groupValue : Fin PiRLCProductSchedule.invocationCount → Fin 33 → F)
    (products : Fin PiRLCFirst54DirectSchedule.candidateCount → F)
    (encodes : Encodes geometry assignment base groupValue products)
    (index : Fin 1330) :
    OrdinarySourcePlan.SourceMap.PreservesRow (sourceMap geometry) assignment
      (pilotEnv program base) (PilotOrdinaryDirectSource.programRow index)
      (PilotOrdinaryDirectSource.programRow_bounded index) := by
  have scope := programRow_support index
  exact ⟨
    preservesCombination geometry assignment base groupValue products encodes
      _ _ scope.1,
    preservesCombination geometry assignment base groupValue products encodes
      _ _ scope.2.1,
    preservesCombination geometry assignment base groupValue products encodes
      _ _ scope.2.2⟩

def rowForms
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : PilotOrdinaryRetainedGeometry.Geometry program logicalWidth)
    (index : Fin 1330) : OrdinaryRow.Forms logicalWidth :=
  SourceCompiler.compileRow (sourceMap geometry) (oneColumn geometry)
    (PilotOrdinaryDirectSource.programRow index)
    (PilotOrdinaryDirectSource.programRow_bounded index)

/-- Canonical direct 14-matrix rows for all 1,330 non-Poseidon pilot rows. -/
def plan
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : PilotOrdinaryRetainedGeometry.Geometry program logicalWidth) :
    ProductionRelation.Plan logicalWidth :=
  OrdinaryRow.planOfForms (by norm_num [Lifecycle.cubeVariables])
    (rowForms geometry)

@[simp] theorem plan_rowCount
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : PilotOrdinaryRetainedGeometry.Geometry program logicalWidth) :
    (plan geometry).rowCount = 1330 := by
  rfl

/-- Matrix acceptance is exactly the canonical Lean-lowered pilot ordinary
row relation. -/
theorem rowsZero_iff_rowsHold
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : PilotOrdinaryRetainedGeometry.Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth)
    (base : Fin (PiRLCProductPlan.baseSourceWidth program) → F)
    (groupValue : Fin PiRLCProductSchedule.invocationCount → Fin 33 → F)
    (products : Fin PiRLCFirst54DirectSchedule.candidateCount → F)
    (one : assignment (oneColumn geometry) = 1)
    (encodes : Encodes geometry assignment base groupValue products) :
    (plan geometry).RowsZero assignment ↔
      R1CS.RowsHold (pilotEnv program base)
        PilotOrdinaryDirectSource.sourceRows := by
  rw [← PilotOrdinaryDirectSource.programRows_hold_iff_rowsHold
    (pilotEnv program base)]
  constructor
  · intro rows index
    have preserves := OrdinarySourcePlan.compileRow_preserves_local
      (sourceMap geometry) (oneColumn geometry)
      (PilotOrdinaryDirectSource.programRow index)
      (PilotOrdinaryDirectSource.programRow_bounded index) assignment
      (pilotEnv program base) one
      (programRow_preserve geometry assignment base groupValue products encodes
        index)
    exact (OrdinaryRow.planOfForms_residual_zero_iff
      (by norm_num [Lifecycle.cubeVariables]) (rowForms geometry) assignment
      (pilotEnv program base) index (PilotOrdinaryDirectSource.programRow index)
      preserves).mp (rows index)
  · intro rows index
    have preserves := OrdinarySourcePlan.compileRow_preserves_local
      (sourceMap geometry) (oneColumn geometry)
      (PilotOrdinaryDirectSource.programRow index)
      (PilotOrdinaryDirectSource.programRow_bounded index) assignment
      (pilotEnv program base) one
      (programRow_preserve geometry assignment base groupValue products encodes
        index)
    exact (OrdinaryRow.planOfForms_residual_zero_iff
      (by norm_num [Lifecycle.cubeVariables]) (rowForms geometry) assignment
      (pilotEnv program base) index (PilotOrdinaryDirectSource.programRow index)
      preserves).mpr (rows index)

end NightstreamFPrime.Export.Stage1.PilotOrdinaryDirectPlan
