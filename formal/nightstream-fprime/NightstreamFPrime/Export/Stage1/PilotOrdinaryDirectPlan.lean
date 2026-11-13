import NightstreamFPrime.Export.Stage1.PilotOrdinaryRetainedGeometry
import NightstreamFPrime.Export.Stage1.RunningTransitionDirectPlan
import NightstreamFPrime.Layout.Stage1.PackageOrderValues
import NightstreamFPrime.Export.Stage1.PilotOrdinaryDirectSource

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

theorem piCcsGeometry {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat}
    (geometry : PilotOrdinaryRetainedGeometry.Geometry program logicalWidth) :
    PiCCSOrdinaryRetainedGeometry.Geometry program logicalWidth :=
  PilotOrdinaryRetainedGeometry.prefixGeometry geometry

def oneColumn {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat}
    (geometry : PilotOrdinaryRetainedGeometry.Geometry program logicalWidth) :
    Fin logicalWidth :=
  PilotOrdinaryRetainedGeometry.oneColumn geometry

def finalSlot (lane : Fin 4) : Fin 1096 :=
  ⟨1080 + lane.val, by have bound := lane.isLt; omega⟩

inductive Location where
  | priorDigest (lane : Fin 4)
  | priorPublic (index : Fin 320)
  | canonicalLocal (index : Fin 268)
  | outputState (lane : Fin 4)
  | outputDigest (lane : Fin 4)

namespace Location

def sourceColumn : Location → Nat
  | .priorDigest lane => PilotProduction.priorDigestStart + lane.val
  | .priorPublic index => PilotProduction.priorPublicInputStart + index.val
  | .canonicalLocal index =>
      PriorStateHash.hashEnd PilotProduction.priorInterface
        PilotProduction.witnessOffset + index.val
  | .outputState lane => PilotProduction.lifecycleOutputOffset +
      PilotValues.absorbCount * 1096 + 1080 + lane.val
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
      change InRange PilotProduction.priorPublicInputStart 320
        (PilotProduction.priorPublicInputStart + index.val)
      unfold InRange
      have bound := index.isLt
      omega
  | canonicalLocal index =>
      apply Or.inl
      apply LogicalSource.canonicalLocal
      change InRange
        (PriorStateHash.hashEnd PilotProduction.priorInterface
          PilotProduction.witnessOffset) 268
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
          PilotValues.absorbCount * 1096 + 1080) 4
        (PilotProduction.lifecycleOutputOffset +
          PilotValues.absorbCount * 1096 + 1080 + lane.val)
      unfold InRange
      have bound := lane.isLt
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
    location.sourceColumn < PilotPackageOrder.SourceColumnCount := by
  exact physicalSource_lt location.sourceColumn location.physicalSupport

theorem targetColumn_lt (location : Location) :
    PilotPackageOrder.sourceToPackage location.sourceColumn <
      PilotPackageOrder.packageColumnCount :=
  PilotPackageOrder.sourceToPackage_lt location.sourceColumn location.sourceColumn_lt

theorem stage1Map (location : Location) :
    PackageOrder.sourceToPackage location.sourceColumn =
      PackageOrder.liftPilotColumn
        (PilotPackageOrder.sourceToPackage location.sourceColumn) := by
  unfold PackageOrder.sourceToPackage
  rw [ite_eq_left]
  have bound := location.sourceColumn_lt
  rw [PilotPackageOrder.sourceColumnCount_eq] at bound
  simpa [PackageOrder.pilotSourceColumnCount] using bound

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
  else if priorPublic : InRange PilotProduction.priorPublicInputStart 320 source then
    some ⟨.priorPublic (rangeIndex priorPublic), by
      rw [Location.sourceColumn, rangeIndex_source priorPublic]⟩
  else if canonicalLocal : InRange
      (PriorStateHash.hashEnd PilotProduction.priorInterface
        PilotProduction.witnessOffset) 268 source then
    some ⟨.canonicalLocal (rangeIndex canonicalLocal), by
      rw [Location.sourceColumn, rangeIndex_source canonicalLocal]⟩
  else if outputState : InRange
      (PilotProduction.lifecycleOutputOffset +
        PilotValues.absorbCount * 1096 + 1080) 4 source then
    some ⟨.outputState (rangeIndex outputState), by
      rw [Location.sourceColumn, rangeIndex_source outputState]⟩
  else if outputDigest : InRange PilotProduction.outputDigestStart 4 source then
    some ⟨.outputDigest (rangeIndex outputDigest), by
      rw [Location.sourceColumn, rangeIndex_source outputDigest]⟩
  else none

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
  · rename_i noPriorDigest noPriorPublic noCanonicalLocal noOutputState
      noOutputDigest
    exfalso
    rcases support with logical | fresh
    · rcases logical with priorDigest | priorPublic | canonicalLocal |
        outputState | outputDigest
      · exact noPriorDigest priorDigest
      · exact noPriorPublic priorPublic
      · exact noCanonicalLocal canonicalLocal
      · exact noOutputState outputState
      · exact noOutputDigest outputDigest
    · have endEq : PilotValues.sourceColumnCount =
          PilotValues.logicalColumnCount := by rfl
      omega

structure Decoded where
  source : Nat
  location : Location
  owns : location.sourceColumn = source

def classifyTarget (column : Nat) : Option Decoded :=
  match PilotPackageOrder.packageToSource column with
  | none => none
  | some source =>
      match classifySource source with
      | none => none
      | some located => some ⟨source, located.location, located.owns⟩

theorem classifyTarget_complete {column : Nat} (support : Target column) :
    ∃ decoded, classifyTarget column = some decoded ∧
      PilotPackageOrder.sourceToPackage decoded.source = column := by
  rcases support with ⟨source, sourceSupport, rfl⟩
  have bound := physicalSource_lt source sourceSupport
  have inverse := PilotPackageOrder.packageToSource_sourceToPackage source bound
  have complete := classifySource_complete sourceSupport
  cases found : classifySource source with
  | none => simp [found] at complete
  | some located =>
      refine ⟨⟨source, located.location, located.owns⟩, ?_, rfl⟩
      simp [classifyTarget, inverse, found]

def sourceMap {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat}
    (geometry : PilotOrdinaryRetainedGeometry.Geometry program logicalWidth) :
    SourceCompiler.SourceMap PilotPackageOrder.packageColumnCount logicalWidth where
  form := fun column =>
    match classifyTarget column.val with
    | none => .empty
    | some decoded => decoded.location.form geometry

def pilotEnv (program : Lifecycle.Stage1.Application.Program)
    (base : Fin (PiRLCProductPlan.baseSourceWidth program) → F) : Env :=
  fun column => RunningTransitionDirectPlan.transitionEnv program base
    (PackageOrder.liftPilotColumn column)

structure Encodes {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat}
    (geometry : PilotOrdinaryRetainedGeometry.Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth)
    (base : Fin (PiRLCProductPlan.baseSourceWidth program) → F)
    (groupValue : Fin PiRLCProductSchedule.invocationCount → Fin 1 → F) : Prop where
  prior : PiCCSOrdinaryRetainedGeometry.Encodes (piCcsGeometry geometry)
    assignment (PiRLCRetainedPreservation.sourceAssignment
      program base groupValue)
  added : PilotOrdinaryRetainedGeometry.Encodes geometry assignment
    (PiRLCRetainedPreservation.sourceAssignment
      program base groupValue)

theorem Location.stage1SourceColumn_lt (location : Location) :
    location.sourceColumn < PackageOrder.SourceColumnCount := by
  have bound := location.sourceColumn_lt
  rw [PilotPackageOrder.sourceColumnCount_eq] at bound
  rw [PackageOrder.sourceColumnCount_eq]
  omega

theorem sourceAssignment_at
    {program : Lifecycle.Stage1.Application.Program}
    (base : Fin (PiRLCProductPlan.baseSourceWidth program) → F)
    (groupValue : Fin PiRLCProductSchedule.invocationCount → Fin 1 → F)
    (location : Location) :
    PiRLCRetainedPreservation.sourceAssignment program base groupValue
        (RunningTransitionRetainedBlocks.packageSourceColumn program
          location.sourceColumn location.stage1SourceColumn_lt) =
      pilotEnv program base
        (PilotPackageOrder.sourceToPackage location.sourceColumn) := by
  rw [RunningTransitionDirectPlan.sourceAssignment_packageSource
    program base groupValue location.sourceColumn
    location.stage1SourceColumn_lt]
  rw [← RunningTransitionDirectPlan.transitionEnv_of_outside program base
    location.sourceColumn location.stage1SourceColumn_lt (by
      left
      have bound := location.sourceColumn_lt
      rw [PilotPackageOrder.sourceColumnCount_eq] at bound
      rw [PiCCSStarts.statementWitnessStart_eq]
      omega)]
  unfold pilotEnv
  rw [location.stage1Map]

private theorem priorLastWitnessStart_eq :
    PoseidonRetainedBlock.priorWitnessStart
      PiCCSOrdinaryRetainedBlocks.priorLastInvocation = 2716842 := by rfl

private theorem outputLastWitnessStart_eq :
    PoseidonRetainedBlock.outputWitnessStart
      PiCCSOrdinaryRetainedBlocks.outputLastInvocation = 5359566 := by rfl

private theorem liftPriorDigestTarget (lane : Fin 4) :
    PackageOrder.liftPilotColumn (2700266 + lane.val) = 2717922 + lane.val := by
  have laneBound := lane.isLt
  unfold PackageOrder.liftPilotColumn
  rw [ite_eq_right (by norm_num [PackageOrder.pilotInputPrivateColumnCount]; omega)]
  rw [ite_eq_left (by norm_num [PackageOrder.pilotPrivateColumnCount]; omega)]
  norm_num [PackageOrder.proofInputColumnCount]
  omega

private theorem liftOutputStateTarget (lane : Fin 4) :
    PackageOrder.liftPilotColumn (5342990 + lane.val) = 5360646 + lane.val := by
  have laneBound := lane.isLt
  unfold PackageOrder.liftPilotColumn
  rw [ite_eq_right (by norm_num [PackageOrder.pilotInputPrivateColumnCount]; omega)]
  rw [ite_eq_left (by norm_num [PackageOrder.pilotPrivateColumnCount]; omega)]
  norm_num [PackageOrder.proofInputColumnCount]
  omega

private theorem priorFinalColumn_eq (lane : Fin 4) :
    PoseidonRetainedBlock.priorWitnessStart
          PiCCSOrdinaryRetainedBlocks.priorLastInvocation +
        (finalSlot lane).val =
      PackageOrder.sourceToPackage (Location.priorDigest lane).sourceColumn := by
  have sourceBound : (Location.priorDigest lane).sourceColumn <
      PackageOrder.pilotSourceColumnCount := by
    have bound := (Location.priorDigest lane).sourceColumn_lt
    rw [PilotPackageOrder.sourceColumnCount_eq] at bound
    simpa [PackageOrder.pilotSourceColumnCount] using bound
  unfold PackageOrder.sourceToPackage
  rw [ite_eq_left sourceBound]
  change _ = PackageOrder.liftPilotColumn
    (PilotPackageOrder.sourceToPackage
      (PilotProduction.priorDigestStart + lane.val))
  rw [PilotOrdinaryDirectSource.priorDigest_targetColumn,
    liftPriorDigestTarget, priorLastWitnessStart_eq]
  change 2716842 + (1080 + lane.val) = 2717922 + lane.val
  omega

private theorem outputFinalColumn_eq (lane : Fin 4) :
    PoseidonRetainedBlock.outputWitnessStart
          PiCCSOrdinaryRetainedBlocks.outputLastInvocation +
        (finalSlot lane).val =
      PackageOrder.sourceToPackage (Location.outputState lane).sourceColumn := by
  have sourceBound : (Location.outputState lane).sourceColumn <
      PackageOrder.pilotSourceColumnCount := by
    have bound := (Location.outputState lane).sourceColumn_lt
    rw [PilotPackageOrder.sourceColumnCount_eq] at bound
    simpa [PackageOrder.pilotSourceColumnCount] using bound
  unfold PackageOrder.sourceToPackage
  rw [ite_eq_left sourceBound]
  change _ = PackageOrder.liftPilotColumn
    (PilotPackageOrder.sourceToPackage
      (PilotProduction.lifecycleOutputOffset +
        PilotValues.absorbCount * 1096 + 1080 + lane.val))
  rw [PilotOrdinaryDirectSource.outputState_targetColumn,
    liftOutputStateTarget, outputLastWitnessStart_eq]
  change 5359566 + (1080 + lane.val) = 5360646 + lane.val
  omega

private theorem priorFinalColumn_bound (lane : Fin 4) :
    PoseidonRetainedBlock.priorWitnessStart
          PiCCSOrdinaryRetainedBlocks.priorLastInvocation +
        (finalSlot lane).val <
      PerApplicationPackage.basePackage.layout.totalColumnCount := by
  have laneBound := (finalSlot lane).isLt
  have total : PerApplicationPackage.basePackage.layout.totalColumnCount =
      11415253 := Package.circuitPackage_layout_values.2.2.2.2
  rw [priorLastWitnessStart_eq, total]
  change 2716842 + (1080 + lane.val) < 11415253
  omega

private theorem outputFinalColumn_bound (lane : Fin 4) :
    PoseidonRetainedBlock.outputWitnessStart
          PiCCSOrdinaryRetainedBlocks.outputLastInvocation +
        (finalSlot lane).val <
      PerApplicationPackage.basePackage.layout.totalColumnCount := by
  have laneBound := (finalSlot lane).isLt
  have total : PerApplicationPackage.basePackage.layout.totalColumnCount =
      11415253 := Package.circuitPackage_layout_values.2.2.2.2
  rw [outputLastWitnessStart_eq, total]
  change 5359566 + (1080 + lane.val) < 11415253
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
    (groupValue : Fin PiRLCProductSchedule.invocationCount → Fin 1 → F)
    (encodes : Encodes geometry assignment base groupValue)
    (location : Location) :
    (location.form geometry).eval assignment =
      pilotEnv program base
        (PilotPackageOrder.sourceToPackage location.sourceColumn) := by
  cases location with
  | priorDigest lane =>
      rw [form, LowNormBlock.Block.form_eval
        (PiCCSOrdinaryRetainedBlocks.priorLastBlock program)
        (PiCCSOrdinaryRetainedGeometry.priorLastStart program)
        (PiCCSOrdinaryRetainedGeometry.priorLastFits (piCcsGeometry geometry))
        assignment (PiRLCRetainedPreservation.sourceAssignment
          program base groupValue) encodes.prior.priorLast
        (finalSlot lane)]
      rw [priorLastBlock_source program lane]
      exact sourceAssignment_at base groupValue (.priorDigest lane)
  | priorPublic index =>
      rw [form, LowNormBlock.Block.form_eval
        (PiCCSOrdinaryRetainedBlocks.freshPublicInputBlock program)
        (PiCCSOrdinaryRetainedGeometry.freshPublicInputStart program)
        (PiCCSOrdinaryRetainedGeometry.freshPublicInputFits
          (piCcsGeometry geometry)) assignment
        (PiRLCRetainedPreservation.sourceAssignment
          program base groupValue) encodes.prior.freshPublicInput index]
      change PiRLCRetainedPreservation.sourceAssignment
          program base groupValue
          (RunningTransitionRetainedBlocks.packageSourceColumn program
            (Location.priorPublic index).sourceColumn
            (Location.priorPublic index).stage1SourceColumn_lt) = _
      exact sourceAssignment_at base groupValue (.priorPublic index)
  | canonicalLocal index =>
      rw [form, LowNormBlock.Block.form_eval
        (PilotOrdinaryRetainedBlocks.canonicalLocalBlock program)
        (PilotOrdinaryRetainedGeometry.canonicalLocalStart program)
        (PilotOrdinaryRetainedGeometry.canonicalLocalFits geometry) assignment
        (PiRLCRetainedPreservation.sourceAssignment
          program base groupValue) encodes.added.canonicalLocal index]
      change PiRLCRetainedPreservation.sourceAssignment
          program base groupValue
          (RunningTransitionRetainedBlocks.packageSourceColumn program
            (Location.canonicalLocal index).sourceColumn
            (Location.canonicalLocal index).stage1SourceColumn_lt) = _
      exact sourceAssignment_at base groupValue (.canonicalLocal index)
  | outputState lane =>
      rw [form, LowNormBlock.Block.form_eval
        (PiCCSOrdinaryRetainedBlocks.outputLastBlock program)
        (PiCCSOrdinaryRetainedGeometry.outputLastStart program)
        (PiCCSOrdinaryRetainedGeometry.outputLastFits (piCcsGeometry geometry))
        assignment (PiRLCRetainedPreservation.sourceAssignment
          program base groupValue) encodes.prior.outputLast
        (finalSlot lane)]
      rw [outputLastBlock_source program lane]
      exact sourceAssignment_at base groupValue (.outputState lane)
  | outputDigest lane =>
      rw [form, LowNormBlock.Block.form_eval
        (PilotOrdinaryRetainedBlocks.outputDigestBlock program)
        (PilotOrdinaryRetainedGeometry.outputDigestStart program)
        (PilotOrdinaryRetainedGeometry.outputDigestFits geometry) assignment
        (PiRLCRetainedPreservation.sourceAssignment
          program base groupValue) encodes.added.outputDigest lane]
      change PiRLCRetainedPreservation.sourceAssignment
          program base groupValue
          (RunningTransitionRetainedBlocks.packageSourceColumn program
            (Location.outputDigest lane).sourceColumn
            (Location.outputDigest lane).stage1SourceColumn_lt) = _
      exact sourceAssignment_at base groupValue (.outputDigest lane)

theorem priorDigest_form_eval_chainOutput
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : PilotOrdinaryRetainedGeometry.Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth)
    (env : Env)
    (values : ∀ lane : Fin 4,
      ((Location.priorDigest lane).form geometry).eval assignment =
        env (PilotPackageOrder.sourceToPackage (Location.priorDigest lane).sourceColumn))
    (lane : Fin 4) :
    ((Location.priorDigest lane).form geometry).eval assignment =
      NightstreamFPrime.Export.Pilot.chainOutputState PilotData.priorChain
        PilotData.priorChain.absorbCount env
        ⟨lane.val, Nat.lt_trans lane.isLt (by decide)⟩ := by
  rw [values lane]
  unfold NightstreamFPrime.Export.Pilot.chainOutputState
    NightstreamFPrime.Export.Package.invocationLocalStart
  change env
      (PilotPackageOrder.sourceToPackage
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
        env (PilotPackageOrder.sourceToPackage (Location.outputState lane).sourceColumn))
    (lane : Fin 4) :
    ((Location.outputState lane).form geometry).eval assignment =
      NightstreamFPrime.Export.Pilot.chainOutputState PilotData.outputChain
        PilotData.outputChain.absorbCount env
        ⟨lane.val, Nat.lt_trans lane.isLt (by decide)⟩ := by
  rw [values lane]
  unfold NightstreamFPrime.Export.Pilot.chainOutputState
    NightstreamFPrime.Export.Package.invocationLocalStart
  change env
      (PilotPackageOrder.sourceToPackage
        (PilotProduction.lifecycleOutputOffset +
          PilotValues.absorbCount * 1096 + 1080 + lane.val)) = _
  rw [PilotOrdinaryDirectSource.outputState_targetColumn]
  rfl

theorem sourceMap_form_eval_of_target
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : PilotOrdinaryRetainedGeometry.Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth)
    (base : Fin (PiRLCProductPlan.baseSourceWidth program) → F)
    (groupValue : Fin PiRLCProductSchedule.invocationCount → Fin 1 → F)
    (encodes : Encodes geometry assignment base groupValue)
    (column : Fin PilotPackageOrder.packageColumnCount)
    (support : Target column.val) :
    ((sourceMap geometry).form column).eval assignment =
      pilotEnv program base column.val := by
  rcases classifyTarget_complete support with ⟨decoded, found, mapped⟩
  change (match classifyTarget column.val with
    | none => SparseForm.empty
    | some value => value.location.form geometry).eval assignment = _
  rw [found]
  rw [Location.form_eval geometry assignment base groupValue encodes
    decoded.location]
  have mappedLocation :
      PilotPackageOrder.sourceToPackage decoded.location.sourceColumn =
        column.val := by
    rw [decoded.owns, mapped]
  rw [mappedLocation]

private theorem preservesCombination
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : PilotOrdinaryRetainedGeometry.Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth)
    (base : Fin (PiRLCProductPlan.baseSourceWidth program) → F)
    (groupValue : Fin PiRLCProductSchedule.invocationCount → Fin 1 → F)
    (encodes : Encodes geometry assignment base groupValue)
    (combination : R1CS.LinearCombination)
    (bounded : SourceCompiler.CombinationBounded
      PilotPackageOrder.packageColumnCount combination)
    (scope : combination.VarsSatisfy Target) :
    OrdinarySourcePlan.SourceMap.PreservesCombination (sourceMap geometry)
      assignment (pilotEnv program base) combination bounded := by
  intro term member
  exact sourceMap_form_eval_of_target geometry assignment base groupValue encodes ⟨term.1, bounded term member⟩ (scope term member)

private theorem programRow_support
    (index : Fin 596) :
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
    (groupValue : Fin PiRLCProductSchedule.invocationCount → Fin 1 → F)
    (encodes : Encodes geometry assignment base groupValue) :
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
    preservesCombination geometry assignment base groupValue encodes
      _ _ scope.1,
    preservesCombination geometry assignment base groupValue encodes
      _ _ scope.2.1,
    preservesCombination geometry assignment base groupValue encodes
      _ _ scope.2.2⟩

theorem programRow_preserve
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : PilotOrdinaryRetainedGeometry.Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth)
    (base : Fin (PiRLCProductPlan.baseSourceWidth program) → F)
    (groupValue : Fin PiRLCProductSchedule.invocationCount → Fin 1 → F)
    (encodes : Encodes geometry assignment base groupValue)
    (index : Fin 596) :
    OrdinarySourcePlan.SourceMap.PreservesRow (sourceMap geometry) assignment
      (pilotEnv program base) (PilotOrdinaryDirectSource.programRow index)
      (PilotOrdinaryDirectSource.programRow_bounded index) := by
  have scope := programRow_support index
  exact ⟨
    preservesCombination geometry assignment base groupValue encodes
      _ _ scope.1,
    preservesCombination geometry assignment base groupValue encodes
      _ _ scope.2.1,
    preservesCombination geometry assignment base groupValue encodes
      _ _ scope.2.2⟩

def rowForms
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : PilotOrdinaryRetainedGeometry.Geometry program logicalWidth)
    (index : Fin 596) : OrdinaryRow.Forms logicalWidth :=
  SourceCompiler.compileRow (sourceMap geometry) (oneColumn geometry)
    (PilotOrdinaryDirectSource.programRow index)
    (PilotOrdinaryDirectSource.programRow_bounded index)

def plan
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : PilotOrdinaryRetainedGeometry.Geometry program logicalWidth) :
    ProductionRelation.Plan logicalWidth :=
  OrdinaryRow.planOfForms (by norm_num [Lifecycle.cubeVariables])
    (rowForms geometry)

@[simp] theorem plan_rowCount
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : PilotOrdinaryRetainedGeometry.Geometry program logicalWidth) :
    (plan geometry).rowCount = 596 := by
  rfl

theorem rowsZero_iff_rowsHold
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : PilotOrdinaryRetainedGeometry.Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth)
    (base : Fin (PiRLCProductPlan.baseSourceWidth program) → F)
    (groupValue : Fin PiRLCProductSchedule.invocationCount → Fin 1 → F)
    (one : assignment (oneColumn geometry) = 1)
    (encodes : Encodes geometry assignment base groupValue) :
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
      (programRow_preserve geometry assignment base groupValue encodes
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
      (programRow_preserve geometry assignment base groupValue encodes
        index)
    exact (OrdinaryRow.planOfForms_residual_zero_iff
      (by norm_num [Lifecycle.cubeVariables]) (rowForms geometry) assignment
      (pilotEnv program base) index (PilotOrdinaryDirectSource.programRow index)
      preserves).mpr (rows index)

end NightstreamFPrime.Export.Stage1.PilotOrdinaryDirectPlan
