import NightstreamFPrime.Export.Stage1.RunningTransitionDirectSource
import NightstreamFPrime.Export.Stage1.RunningTransitionRetainedGeometry

/-!
Owns the executable source resolver and direct 14-matrix plan for the
running-instance transition. The resolver uses the established Spartan
partial inverse and only the six retained support blocks.

This module does not append the plan to other Stage 1 phases or construct a
final package identity.
-/

namespace NightstreamFPrime.Export.Stage1.RunningTransitionDirectPlan

open NightstreamFPrime.Circuit
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Layout.Stage1
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra
open RunningTransitionRetainedBlocks
open RunningTransitionRetainedGeometry

/-- One exact retained support location before the Spartan permutation. -/
inductive Location where
  | state (index : Fin RunningTransitionSourceSupport.stateCount)
  | output (index : Fin RunningTransitionSourceSupport.outputCount)
  | roundC0 (coordinate : Fin productionShape.cubeVariables)
  | roundC1 (coordinate : Fin productionShape.cubeVariables)
  | piDec (index : Fin RunningTransitionSourceSupport.piDecCount)
  | fresh (index : Fin freshCount)

namespace Location

def sourceColumn : Location → Nat
  | .state index => RunningTransitionSourceSupport.stateStart + index.val
  | .output index => RunningTransitionSourceSupport.outputStart + index.val
  | .roundC0 coordinate => PiCCSStarts.roundTranscriptWitnessStart +
      coordinate.val * RunningTransitionInputs.roundStride +
        RunningTransitionInputs.roundSampleC0Offset
  | .roundC1 coordinate => PiCCSStarts.roundTranscriptWitnessStart +
      coordinate.val * RunningTransitionInputs.roundStride +
        RunningTransitionInputs.roundSampleC1Offset
  | .piDec index => RunningTransitionSourceSupport.piDecStart + index.val
  | .fresh index => RunningTransitionInputs.phaseOffset + index.val

theorem sourceColumn_lt (location : Location) :
    location.sourceColumn < Spartan.SourceColumnCount := by
  cases location with
  | state index =>
      have bound := index.isLt
      change index.val < 11 at bound
      rw [sourceColumn, RunningTransitionSourceSupport.stateStart_eq,
        Spartan.sourceColumnCount_eq]
      omega
  | output index =>
      have bound := index.isLt
      change index.val < 49393 at bound
      rw [sourceColumn, RunningTransitionSourceSupport.outputStart_eq,
        Spartan.sourceColumnCount_eq]
      omega
  | roundC0 coordinate =>
      have bound := coordinate.isLt
      change coordinate.val < 28 at bound
      rw [sourceColumn, PiCCSStarts.roundTranscriptWitnessStart_eq,
        Spartan.sourceColumnCount_eq]
      norm_num [RunningTransitionInputs.roundStride,
        RunningTransitionInputs.roundSampleC0Offset]
      omega
  | roundC1 coordinate =>
      have bound := coordinate.isLt
      change coordinate.val < 28 at bound
      rw [sourceColumn, PiCCSStarts.roundTranscriptWitnessStart_eq,
        Spartan.sourceColumnCount_eq]
      norm_num [RunningTransitionInputs.roundStride,
        RunningTransitionInputs.roundSampleC1Offset]
      omega
  | piDec index =>
      have bound := index.isLt
      change index.val < 49248 at bound
      rw [sourceColumn, RunningTransitionSourceSupport.piDecStart_eq,
        Spartan.sourceColumnCount_eq]
      omega
  | fresh index =>
      have bound := index.isLt
      change index.val < 296138 at bound
      rw [sourceColumn, Spartan.sourceColumnCount_eq]
      norm_num [RunningTransitionInputs.phaseOffset]
      omega

def form {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat} (geometry : Geometry program logicalWidth) :
    Location → SparseForm logicalWidth
  | .state index => (stateBlock program).form
      (stateStart program) (stateFits geometry) index
  | .output index => (outputBlock program).form
      (outputStart program) (outputFits geometry) index
  | .roundC0 coordinate => (roundC0Block program).form
      (roundC0Start program) (roundC0Fits geometry) coordinate
  | .roundC1 coordinate => (roundC1Block program).form
      (roundC1Start program) (roundC1Fits geometry) coordinate
  | .piDec index => (piDecBlock program).form
      (piDecStart program) (piDecFits geometry) index
  | .fresh index => (freshBlock program).form
      (freshStart program) (freshFits geometry) index

/-- Every selected form reconstructs its exact nested package source. -/
theorem form_eval {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat} (geometry : Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth)
    (source : Fin (sourceWidth program) → F)
    (encodes : Encodes geometry assignment source) (location : Location) :
    (location.form geometry).eval assignment =
      source (packageSourceColumn program location.sourceColumn
        location.sourceColumn_lt) := by
  cases location with
  | state index =>
      rw [form, LowNormBlock.Block.form_eval _ _ _ assignment source
        encodes.state]
      apply congrArg source
      apply Fin.ext
      rfl
  | output index =>
      rw [form, LowNormBlock.Block.form_eval _ _ _ assignment source
        encodes.output]
      apply congrArg source
      apply Fin.ext
      rfl
  | roundC0 coordinate =>
      rw [form, LowNormBlock.Block.form_eval _ _ _ assignment source
        encodes.roundC0]
      apply congrArg source
      apply Fin.ext
      rfl
  | roundC1 coordinate =>
      rw [form, LowNormBlock.Block.form_eval _ _ _ assignment source
        encodes.roundC1]
      apply congrArg source
      apply Fin.ext
      rfl
  | piDec index =>
      rw [form, LowNormBlock.Block.form_eval _ _ _ assignment source
        encodes.piDec]
      apply congrArg source
      apply Fin.ext
      rfl
  | fresh index =>
      rw [form, LowNormBlock.Block.form_eval _ _ _ assignment source
        encodes.fresh]
      apply congrArg source
      apply Fin.ext
      rfl

end Location

def rangeIndex {start count column : Nat}
    (inside : RunningTransitionSourceSupport.InRange start count column) :
    Fin count :=
  ⟨column - start, by
    unfold RunningTransitionSourceSupport.InRange at inside
    omega⟩

@[simp] theorem rangeIndex_source {start count column : Nat}
    (inside : RunningTransitionSourceSupport.InRange start count column) :
    start + (rangeIndex inside).val = column := by
  change start + (column - start) = column
  unfold RunningTransitionSourceSupport.InRange at inside
  omega

local instance (start count column : Nat) : Decidable
    (RunningTransitionSourceSupport.InRange start count column) := by
  unfold RunningTransitionSourceSupport.InRange
  infer_instance

/-- A proof-carrying point-coordinate result. Proofs erase in executable
code; the decoder performs one subtraction, division, bound check, and exact
equality check. -/
structure PointLocation (offset column : Nat) where
  coordinate : Fin productionShape.cubeVariables
  owns : column = PiCCSStarts.roundTranscriptWitnessStart +
    coordinate.val * RunningTransitionInputs.roundStride + offset

def pointCoordinate? (offset column : Nat) : Option (PointLocation offset column) :=
  let candidate :=
    (column - PiCCSStarts.roundTranscriptWitnessStart) /
      RunningTransitionInputs.roundStride
  if bound : candidate < productionShape.cubeVariables then
    if owns : column = PiCCSStarts.roundTranscriptWitnessStart +
        candidate * RunningTransitionInputs.roundStride + offset then
      some ⟨⟨candidate, bound⟩, owns⟩
    else
      none
  else
    none

private theorem pointQuotient
    (coordinate : Fin productionShape.cubeVariables) (offset : Nat)
    (offsetBound : offset < RunningTransitionInputs.roundStride) :
    (PiCCSStarts.roundTranscriptWitnessStart +
          coordinate.val * RunningTransitionInputs.roundStride + offset -
        PiCCSStarts.roundTranscriptWitnessStart) /
      RunningTransitionInputs.roundStride = coordinate.val := by
  have subtraction :
      PiCCSStarts.roundTranscriptWitnessStart +
            coordinate.val * RunningTransitionInputs.roundStride + offset -
          PiCCSStarts.roundTranscriptWitnessStart =
        offset + RunningTransitionInputs.roundStride * coordinate.val := by
    calc
      PiCCSStarts.roundTranscriptWitnessStart +
              coordinate.val * RunningTransitionInputs.roundStride + offset -
            PiCCSStarts.roundTranscriptWitnessStart =
          coordinate.val * RunningTransitionInputs.roundStride + offset := by
        omega
      _ = offset + RunningTransitionInputs.roundStride * coordinate.val := by
        simp [Nat.mul_comm, Nat.add_comm]
  rw [subtraction,
    Nat.add_mul_div_left offset coordinate.val (by
      norm_num [RunningTransitionInputs.roundStride]),
    Nat.div_eq_of_lt offsetBound, Nat.zero_add]

theorem pointCoordinate?_canonical
    (coordinate : Fin productionShape.cubeVariables) (offset : Nat)
    (offsetBound : offset < RunningTransitionInputs.roundStride) :
    pointCoordinate? offset
        (PiCCSStarts.roundTranscriptWitnessStart +
          coordinate.val * RunningTransitionInputs.roundStride + offset) =
      some ⟨coordinate, rfl⟩ := by
  unfold pointCoordinate?
  rw [pointQuotient coordinate offset offsetBound]
  simp [coordinate.isLt]

/-- The source classifier returns its ownership equation with the location. -/
structure Located (column : Nat) where
  location : Location
  owns : location.sourceColumn = column

def classifySource (column : Nat) : Option (Located column) :=
  if state : RunningTransitionSourceSupport.InRange
      RunningTransitionSourceSupport.stateStart
      RunningTransitionSourceSupport.stateCount column then
    some ⟨.state (rangeIndex state), by
      rw [Location.sourceColumn, rangeIndex_source state]⟩
  else if output : RunningTransitionSourceSupport.InRange
      RunningTransitionSourceSupport.outputStart
      RunningTransitionSourceSupport.outputCount column then
    some ⟨.output (rangeIndex output), by
      rw [Location.sourceColumn, rangeIndex_source output]⟩
  else match pointCoordinate?
      RunningTransitionInputs.roundSampleC0Offset column with
    | some point => some ⟨.roundC0 point.coordinate, point.owns.symm⟩
    | none => match pointCoordinate?
        RunningTransitionInputs.roundSampleC1Offset column with
      | some point => some ⟨.roundC1 point.coordinate, point.owns.symm⟩
      | none =>
        if piDec : RunningTransitionSourceSupport.InRange
            RunningTransitionSourceSupport.piDecStart
            RunningTransitionSourceSupport.piDecCount column then
          some ⟨.piDec (rangeIndex piDec), by
            rw [Location.sourceColumn, rangeIndex_source piDec]⟩
        else if fresh : RunningTransitionSourceSupport.InRange
            RunningTransitionInputs.phaseOffset freshCount column then
          some ⟨.fresh (rangeIndex fresh), by
            rw [Location.sourceColumn, rangeIndex_source fresh]⟩
        else
          none

private theorem fresh_inRange {column : Nat}
    (inside : RunningTransitionInputs.phaseOffset ≤ column ∧
      column < RunningTransitionSourceSupport.physicalEnd) :
    RunningTransitionSourceSupport.InRange
      RunningTransitionInputs.phaseOffset freshCount column := by
  unfold RunningTransitionSourceSupport.InRange freshCount
  norm_num [RunningTransitionSourceSupport.physicalEnd,
    RunningTransitionInputs.phaseOffset] at inside ⊢
  omega

/-- Every source that occurs in a transition row is found by the executable
classifier. -/
theorem classifySource_complete {column : Nat}
    (support : RunningTransitionSourceSupport.Source column) :
    (classifySource column).isSome := by
  by_cases state : RunningTransitionSourceSupport.InRange
    RunningTransitionSourceSupport.stateStart
      RunningTransitionSourceSupport.stateCount column
  · unfold classifySource
    rw [dif_pos state]
    rfl
  by_cases output : RunningTransitionSourceSupport.InRange
    RunningTransitionSourceSupport.outputStart
      RunningTransitionSourceSupport.outputCount column
  · unfold classifySource
    rw [dif_neg state, dif_pos output]
    rfl
  cases c0 : pointCoordinate?
      RunningTransitionInputs.roundSampleC0Offset column with
  | some point =>
    unfold classifySource
    rw [dif_neg state, dif_neg output, c0]
    rfl
  | none =>
    cases c1 : pointCoordinate?
        RunningTransitionInputs.roundSampleC1Offset column with
    | some point =>
      unfold classifySource
      rw [dif_neg state, dif_neg output, c0, c1]
      rfl
    | none =>
      by_cases piDec : RunningTransitionSourceSupport.InRange
        RunningTransitionSourceSupport.piDecStart
          RunningTransitionSourceSupport.piDecCount column
      · unfold classifySource
        rw [dif_neg state, dif_neg output, c0, c1, dif_pos piDec]
        rfl
      by_cases fresh : RunningTransitionSourceSupport.InRange
        RunningTransitionInputs.phaseOffset freshCount column
      · unfold classifySource
        rw [dif_neg state, dif_neg output, c0, c1, dif_neg piDec,
          dif_pos fresh]
        rfl
      · exfalso
        rcases support with external | localSupport
        · rcases external with stateSupport | outputSupport |
              pointSupport | piDecSupport
          · exact state stateSupport
          · exact output outputSupport
          · rcases pointSupport with ⟨coordinate, pointC0 | pointC1⟩
            · subst column
              have canonical := pointCoordinate?_canonical coordinate
                RunningTransitionInputs.roundSampleC0Offset (by
                  norm_num [RunningTransitionInputs.roundSampleC0Offset,
                    RunningTransitionInputs.roundStride])
              rw [canonical] at c0
              contradiction
            · subst column
              have canonical := pointCoordinate?_canonical coordinate
                RunningTransitionInputs.roundSampleC1Offset (by
                  norm_num [RunningTransitionInputs.roundSampleC1Offset,
                    RunningTransitionInputs.roundStride])
              rw [canonical] at c1
              contradiction
          · exact piDec
              (RunningTransitionSourceSupport.piDecField_inRange piDecSupport)
        · exact fresh (fresh_inRange localSupport)

/-- Decoded target column. The source ownership proof is retained while the
Spartan target equality is proved only for supported row terms. -/
structure Decoded where
  source : Nat
  location : Location
  owns : location.sourceColumn = source

def classifyTarget (column : Nat) : Option Decoded :=
  match Spartan.spartanToSource column with
  | none => none
  | some source =>
      match classifySource source with
      | none => none
      | some located => some ⟨source, located.location, located.owns⟩

theorem classifyTarget_complete {column : Nat}
    (support : RunningTransitionSourceSupport.Target column) :
    ∃ decoded, classifyTarget column = some decoded ∧
      Spartan.sourceToSpartan decoded.source = column := by
  rcases support with ⟨source, sourceSupport, rfl⟩
  have bound :=
    RunningTransitionSourceSupport.source_lt_sourceColumnCount sourceSupport
  have inverse := Spartan.spartanToSource_sourceToSpartan source bound
  have complete := classifySource_complete sourceSupport
  cases found : classifySource source with
  | none => simp [found] at complete
  | some located =>
      refine ⟨⟨source, located.location, located.owns⟩, ?_, rfl⟩
      simp [classifyTarget, inverse, found]

def sourceMap {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat} (geometry : Geometry program logicalWidth) :
    SourceCompiler.SourceMap Spartan.spartanColumnCount logicalWidth where
  form := fun column =>
    match classifyTarget column.val with
    | none => .empty
    | some decoded => decoded.location.form geometry

/-- The old package-row environment reads each Spartan column through the
per-application shift into the complete package source prefix. -/
def transitionEnv (program : Lifecycle.Stage1.Application.Program)
    (base : Fin (PiRLCProductPlan.baseSourceWidth program) → F) : Env :=
  fun column =>
    if bound : column < PiRLCProductPlan.basePackage.layout.totalColumnCount then
      base (PiRLCProductPlan.shiftedPackageColumn program column bound)
    else
      0

private theorem mapped_lt_basePackage (source : Nat)
    (bound : source < Spartan.SourceColumnCount) :
    Spartan.sourceToSpartan source <
      PiRLCProductPlan.basePackage.layout.totalColumnCount := by
  have mapped := Spartan.sourceToSpartan_lt source bound
  have total : PiRLCProductPlan.basePackage.layout.totalColumnCount = 29336725 :=
    NightstreamFPrime.Export.Stage1.Package.circuitPackage_layout_values.2.2.2.2
  rw [total]
  simpa [Spartan.spartanColumnCount] using mapped

theorem sourceAssignment_packageSource
    (program : Lifecycle.Stage1.Application.Program)
    (base : Fin (PiRLCProductPlan.baseSourceWidth program) → F)
    (groupValue : Fin PiRLCProductSchedule.invocationCount → Fin 33 → F)
    (products : Fin PiRLCFirst54DirectSchedule.candidateCount → F)
    (source : Nat) (bound : source < Spartan.SourceColumnCount) :
    PiRLCRetainedPreservation.sourceAssignment program base groupValue products
        (packageSourceColumn program source bound) =
      transitionEnv program base (Spartan.sourceToSpartan source) := by
  rw [packageSourceColumn,
    PiRLCRetainedPreservation.sourceAssignment_base]
  unfold transitionEnv
  rw [dif_pos (mapped_lt_basePackage source bound)]

theorem sourceMap_form_eval_of_target
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth)
    (base : Fin (PiRLCProductPlan.baseSourceWidth program) → F)
    (groupValue : Fin PiRLCProductSchedule.invocationCount → Fin 33 → F)
    (products : Fin PiRLCFirst54DirectSchedule.candidateCount → F)
    (encodes : Encodes geometry assignment
      (PiRLCRetainedPreservation.sourceAssignment program base groupValue products))
    (column : Fin Spartan.spartanColumnCount)
    (support : RunningTransitionSourceSupport.Target column.val) :
    ((sourceMap geometry).form column).eval assignment =
      transitionEnv program base column.val := by
  rcases classifyTarget_complete support with ⟨decoded, found, mapped⟩
  change (match classifyTarget column.val with
    | none => SparseForm.empty
    | some value => value.location.form geometry).eval assignment = _
  rw [found]
  rw [Location.form_eval geometry assignment _ encodes decoded.location]
  rw [sourceAssignment_packageSource program base groupValue products
    decoded.location.sourceColumn decoded.location.sourceColumn_lt]
  have mappedLocation :
      Spartan.sourceToSpartan decoded.location.sourceColumn = column.val := by
    rw [decoded.owns, mapped]
  rw [mappedLocation]

private theorem preservesCombination
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth)
    (base : Fin (PiRLCProductPlan.baseSourceWidth program) → F)
    (groupValue : Fin PiRLCProductSchedule.invocationCount → Fin 33 → F)
    (products : Fin PiRLCFirst54DirectSchedule.candidateCount → F)
    (encodes : Encodes geometry assignment
      (PiRLCRetainedPreservation.sourceAssignment program base groupValue products))
    (combination : R1CS.LinearCombination)
    (bounded : SourceCompiler.CombinationBounded Spartan.spartanColumnCount
      combination)
    (scope : combination.VarsSatisfy RunningTransitionSourceSupport.Target) :
    OrdinarySourcePlan.SourceMap.PreservesCombination (sourceMap geometry)
      assignment (transitionEnv program base) combination bounded := by
  intro term member
  exact sourceMap_form_eval_of_target geometry assignment base groupValue
    products encodes ⟨term.1, bounded term member⟩ (scope term member)

private theorem programRow_support
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (index : Fin (RunningTransitionDirectSource.program relation).rowCount) :
    ((RunningTransitionDirectSource.program relation).row index).VarsSatisfy
      RunningTransitionSourceSupport.Target := by
  apply RunningTransitionSourceSupport.remappedRows_varsSatisfy relation
  exact List.get_mem _ index

def inputs
    {program : Lifecycle.Stage1.Application.Program}
    {sourceLogicalWidth targetLogicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth sourceLogicalWidth}
    (relation : ProductionKey.LogicalRelation sourceLogicalWidth publicFits)
    (geometry : Geometry program targetLogicalWidth) :
    (RunningTransitionDirectSource.program relation).Inputs
      targetLogicalWidth where
  oneColumn := oneColumn geometry
  sourceMap := fun _ => sourceMap geometry

theorem inputs_preserve
    {program : Lifecycle.Stage1.Application.Program}
    {sourceLogicalWidth targetLogicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth sourceLogicalWidth}
    (relation : ProductionKey.LogicalRelation sourceLogicalWidth publicFits)
    (geometry : Geometry program targetLogicalWidth)
    (assignment : Assignment F targetLogicalWidth)
    (base : Fin (PiRLCProductPlan.baseSourceWidth program) → F)
    (groupValue : Fin PiRLCProductSchedule.invocationCount → Fin 33 → F)
    (products : Fin PiRLCFirst54DirectSchedule.candidateCount → F)
    (encodes : Encodes geometry assignment
      (PiRLCRetainedPreservation.sourceAssignment program base groupValue products)) :
    ∀ index, OrdinarySourcePlan.SourceMap.PreservesRow
      ((inputs relation geometry).sourceMap index) assignment
      (transitionEnv program base)
      ((RunningTransitionDirectSource.program relation).row index)
      ((RunningTransitionDirectSource.program relation).bounded index) := by
  intro index
  have scope := programRow_support relation index
  exact ⟨
    preservesCombination geometry assignment base groupValue products encodes
      _ _ scope.1,
    preservesCombination geometry assignment base groupValue products encodes
      _ _ scope.2.1,
    preservesCombination geometry assignment base groupValue products encodes
      _ _ scope.2.2⟩

/-- Canonical direct 14-matrix rows for the running transition. -/
def plan
    {program : Lifecycle.Stage1.Application.Program}
    {sourceLogicalWidth targetLogicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth sourceLogicalWidth}
    (relation : ProductionKey.LogicalRelation sourceLogicalWidth publicFits)
    (geometry : Geometry program targetLogicalWidth) :
    ProductionRelation.Plan targetLogicalWidth :=
  (RunningTransitionDirectSource.program relation).compile
    (inputs relation geometry) |>.toPlan

@[simp] theorem plan_rowCount
    {program : Lifecycle.Stage1.Application.Program}
    {sourceLogicalWidth targetLogicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth sourceLogicalWidth}
    (relation : ProductionKey.LogicalRelation sourceLogicalWidth publicFits)
    (geometry : Geometry program targetLogicalWidth) :
    (plan relation geometry).rowCount = 345495 := by
  change (RunningTransitionDirectSource.program relation).rowCount = 345495
  exact RunningTransitionDirectSource.program_rowCount relation

/-- The compiled transition plan depends on relation shape only. -/
theorem plan_eq_of_same_shape
    {program : Lifecycle.Stage1.Application.Program}
    {sourceLogicalWidth targetLogicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth sourceLogicalWidth}
    (left right : ProductionKey.LogicalRelation sourceLogicalWidth publicFits)
    (geometry : Geometry program targetLogicalWidth) :
    plan left geometry = plan right geometry := by
  rfl

/-- Direct matrix acceptance is exactly the existing physical running
transition under the per-application package pullback. -/
theorem rowsZero_iff_physical
    {program : Lifecycle.Stage1.Application.Program}
    {sourceLogicalWidth targetLogicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth sourceLogicalWidth}
    (relation : ProductionKey.LogicalRelation sourceLogicalWidth publicFits)
    (geometry : Geometry program targetLogicalWidth)
    (assignment : Assignment F targetLogicalWidth)
    (base : Fin (PiRLCProductPlan.baseSourceWidth program) → F)
    (groupValue : Fin PiRLCProductSchedule.invocationCount → Fin 33 → F)
    (products : Fin PiRLCFirst54DirectSchedule.candidateCount → F)
    (one : assignment (oneColumn geometry) = 1)
    (encodes : Encodes geometry assignment
      (PiRLCRetainedPreservation.sourceAssignment program base groupValue products)) :
    (plan relation geometry).RowsZero assignment ↔
      RunningTransitionLayout.PhysicalHolds sourceLogicalWidth publicFits
        (Spartan.pullback (transitionEnv program base)) := by
  rw [plan]
  rw [OrdinarySourcePlan.Program.rowsZero_iff
    (RunningTransitionDirectSource.program relation)
    (inputs relation geometry) assignment (transitionEnv program base) one
    (inputs_preserve relation geometry assignment base groupValue products encodes)]
  exact RunningTransitionDirectSource.program_holds_iff_physical relation
    (transitionEnv program base)

end NightstreamFPrime.Export.Stage1.RunningTransitionDirectPlan
