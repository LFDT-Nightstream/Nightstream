import NightstreamFPrime.Export.Stage1.PiDECRetainedGeometry
import NightstreamFPrime.Export.Stage1.RunningTransitionDirectPlan
import NightstreamFPrime.Layout.PiDEC.v1_1.Preservation

/-!
Owns the executable retained-source resolver and direct 14-matrix plan for the
four nonempty canonical PiDEC row packets. It does not append the plan to the
final Stage 1 package or close PiDEC conformance.
-/

namespace NightstreamFPrime.Export.Stage1.PiDECDirectPlan

open NightstreamFPrime.Circuit
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Layout.Stage1
open NightstreamFPrime.Layout.Stage1.PiDECSourceSupport
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra
open PiDECRetainedBlocks
open PiDECRetainedGeometry

inductive Location where
  | parentCommitment (index : Fin PiDECInputs.commitmentWordsPerChild)
  | parentPublicInput (index : Fin PiDECInputs.publicInputWordsPerChild)
  | parentEvalK (index : Fin PiDECInputs.evalKWordsPerChild)
  | parentEvalA (index : Fin PiDECInputs.evalAWordsPerChild)
  | proof (index : Fin PiDECInputs.proofInputColumnCount)
  | logical (index : Fin 270)
  | fresh (index : Fin freshCount)

namespace Location

def sourceColumn : Location → Nat
  | .parentCommitment index =>
      PiDECSourceSupport.parentCommitmentStart + index.val
  | .parentPublicInput index =>
      PiDECSourceSupport.parentPublicInputStart + index.val
  | .parentEvalK index => PiDECSourceSupport.parentEvalKStart + index.val
  | .parentEvalA index => PiDECSourceSupport.parentEvalAStart + index.val
  | .proof index => PiDECInputs.proofInputStart + index.val
  | .logical index => PiDECStarts.phaseLogicalStart + index.val
  | .fresh index => PiDECStarts.phaseFreshStart + index.val

theorem sourceSupport (location : Location) : Source location.sourceColumn := by
  cases location with
  | parentCommitment index =>
      apply parent_source
      apply PiDECSourceSupport.parentCommitment
      constructor
      · change PiDECSourceSupport.parentCommitmentStart ≤
          PiDECSourceSupport.parentCommitmentStart + index.val
        omega
      · change PiDECSourceSupport.parentCommitmentStart + index.val <
          PiDECSourceSupport.parentCommitmentStart +
            PiDECInputs.commitmentWordsPerChild
        have bound := index.isLt
        omega
  | parentPublicInput index =>
      apply parent_source
      apply PiDECSourceSupport.parentPublicInput
      constructor
      · change PiDECSourceSupport.parentPublicInputStart ≤
          PiDECSourceSupport.parentPublicInputStart + index.val
        omega
      · change PiDECSourceSupport.parentPublicInputStart + index.val <
          PiDECSourceSupport.parentPublicInputStart +
            PiDECInputs.publicInputWordsPerChild
        have bound := index.isLt
        omega
  | parentEvalK index =>
      apply parent_source
      apply PiDECSourceSupport.parentEvalK
      constructor
      · change PiDECSourceSupport.parentEvalKStart ≤
          PiDECSourceSupport.parentEvalKStart + index.val
        omega
      · change PiDECSourceSupport.parentEvalKStart + index.val <
          PiDECSourceSupport.parentEvalKStart + PiDECInputs.evalKWordsPerChild
        have bound := index.isLt
        omega
  | parentEvalA index =>
      apply parent_source
      apply PiDECSourceSupport.parentEvalA
      constructor
      · change PiDECSourceSupport.parentEvalAStart ≤
          PiDECSourceSupport.parentEvalAStart + index.val
        omega
      · change PiDECSourceSupport.parentEvalAStart + index.val <
          PiDECSourceSupport.parentEvalAStart + PiDECInputs.evalAWordsPerChild
        have bound := index.isLt
        omega
  | proof index =>
      apply proof_source
      constructor
      · change PiDECInputs.proofInputStart ≤
          PiDECInputs.proofInputStart + index.val
        omega
      · change PiDECInputs.proofInputStart + index.val <
          PiDECInputs.proofInputStart + PiDECInputs.proofInputColumnCount
        have bound := index.isLt
        omega
  | logical index =>
      apply logical_source
      constructor
      · change PiDECStarts.phaseLogicalStart ≤
          PiDECStarts.phaseLogicalStart + index.val
        omega
      · change PiDECStarts.phaseLogicalStart + index.val <
          PiDECStarts.phaseLogicalStart + 270
        have bound := index.isLt
        omega
  | fresh index =>
      apply fresh_source
      constructor
      · change PiDECStarts.phaseFreshStart ≤
          PiDECStarts.phaseFreshStart + index.val
        omega
      · change PiDECStarts.phaseFreshStart + index.val <
          PiDECStarts.phaseFreshStart + freshCount
        have bound := index.isLt
        omega

theorem sourceColumn_lt (location : Location) :
    location.sourceColumn < Spartan.SourceColumnCount :=
  PiDECSourceSupport.source_lt_sourceColumnCount location.sourceSupport

def form {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat} (geometry : Geometry program logicalWidth) :
    Location → SparseForm logicalWidth
  | .parentCommitment index => (parentCommitmentBlock program).form
      (PiDECRetainedGeometry.parentCommitmentStart program)
      (parentCommitmentFits geometry) index
  | .parentPublicInput index => (parentPublicInputBlock program).form
      (PiDECRetainedGeometry.parentPublicInputStart program)
      (parentPublicInputFits geometry) index
  | .parentEvalK index => (parentEvalKBlock program).form
      (PiDECRetainedGeometry.parentEvalKStart program)
      (parentEvalKFits geometry) index
  | .parentEvalA index => (parentEvalABlock program).form
      (PiDECRetainedGeometry.parentEvalAStart program)
      (parentEvalAFits geometry) index
  | .proof index => (proofBlock program).form (proofStart program)
      (proofFits geometry) index
  | .logical index => (logicalBlock program).form (logicalStart program)
      (logicalFits geometry) index
  | .fresh index => (freshBlock program).form (freshStart program)
      (freshFits geometry) index

theorem form_eval {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat} (geometry : Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth)
    (source : Fin (sourceWidth program) → F)
    (encodes : Encodes geometry assignment source) (location : Location) :
    (location.form geometry).eval assignment =
      source (RunningTransitionRetainedBlocks.packageSourceColumn program
        location.sourceColumn location.sourceColumn_lt) := by
  cases location with
  | parentCommitment index =>
      rw [form, LowNormBlock.Block.form_eval _ _ _ assignment source
        encodes.parentCommitment]
      apply congrArg source
      apply Fin.ext
      rfl
  | parentPublicInput index =>
      rw [form, LowNormBlock.Block.form_eval _ _ _ assignment source
        encodes.parentPublicInput]
      apply congrArg source
      apply Fin.ext
      rfl
  | parentEvalK index =>
      rw [form, LowNormBlock.Block.form_eval _ _ _ assignment source
        encodes.parentEvalK]
      apply congrArg source
      apply Fin.ext
      rfl
  | parentEvalA index =>
      rw [form, LowNormBlock.Block.form_eval _ _ _ assignment source
        encodes.parentEvalA]
      apply congrArg source
      apply Fin.ext
      rfl
  | proof index =>
      rw [form, LowNormBlock.Block.form_eval _ _ _ assignment source encodes.proof]
      apply congrArg source
      apply Fin.ext
      rfl
  | logical index =>
      rw [form, LowNormBlock.Block.form_eval _ _ _ assignment source encodes.logical]
      apply congrArg source
      apply Fin.ext
      rfl
  | fresh index =>
      rw [form, LowNormBlock.Block.form_eval _ _ _ assignment source encodes.fresh]
      apply congrArg source
      apply Fin.ext
      rfl

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

structure Located (column : Nat) where
  location : Location
  owns : location.sourceColumn = column

def classifySource (column : Nat) : Option (Located column) :=
  if commitment : InRange PiDECSourceSupport.parentCommitmentStart
      PiDECInputs.commitmentWordsPerChild column then
    some ⟨.parentCommitment (rangeIndex commitment), by
      rw [Location.sourceColumn, rangeIndex_source commitment]⟩
  else if publicInput : InRange PiDECSourceSupport.parentPublicInputStart
      PiDECInputs.publicInputWordsPerChild column then
    some ⟨.parentPublicInput (rangeIndex publicInput), by
      rw [Location.sourceColumn, rangeIndex_source publicInput]⟩
  else if evalK : InRange PiDECSourceSupport.parentEvalKStart
      PiDECInputs.evalKWordsPerChild column then
    some ⟨.parentEvalK (rangeIndex evalK), by
      rw [Location.sourceColumn, rangeIndex_source evalK]⟩
  else if evalA : InRange PiDECSourceSupport.parentEvalAStart
      PiDECInputs.evalAWordsPerChild column then
    some ⟨.parentEvalA (rangeIndex evalA), by
      rw [Location.sourceColumn, rangeIndex_source evalA]⟩
  else if proof : InRange PiDECInputs.proofInputStart
      PiDECInputs.proofInputColumnCount column then
    some ⟨.proof (rangeIndex proof), by
      rw [Location.sourceColumn, rangeIndex_source proof]⟩
  else if logical : InRange PiDECStarts.phaseLogicalStart 270 column then
    some ⟨.logical (rangeIndex logical), by
      rw [Location.sourceColumn, rangeIndex_source logical]⟩
  else if fresh : InRange PiDECStarts.phaseFreshStart freshCount column then
    some ⟨.fresh (rangeIndex fresh), by
      rw [Location.sourceColumn, rangeIndex_source fresh]⟩
  else
    none

theorem classifySource_complete {column : Nat} (support : Source column) :
    (classifySource column).isSome := by
  by_cases commitment : InRange PiDECSourceSupport.parentCommitmentStart
      PiDECInputs.commitmentWordsPerChild column
  · unfold classifySource
    rw [dif_pos commitment]
    rfl
  by_cases publicInput : InRange PiDECSourceSupport.parentPublicInputStart
      PiDECInputs.publicInputWordsPerChild column
  · unfold classifySource
    rw [dif_neg commitment, dif_pos publicInput]
    rfl
  by_cases evalK : InRange PiDECSourceSupport.parentEvalKStart
      PiDECInputs.evalKWordsPerChild column
  · unfold classifySource
    rw [dif_neg commitment, dif_neg publicInput, dif_pos evalK]
    rfl
  by_cases evalA : InRange PiDECSourceSupport.parentEvalAStart
      PiDECInputs.evalAWordsPerChild column
  · unfold classifySource
    rw [dif_neg commitment, dif_neg publicInput, dif_neg evalK, dif_pos evalA]
    rfl
  by_cases proof : InRange PiDECInputs.proofInputStart
      PiDECInputs.proofInputColumnCount column
  · unfold classifySource
    rw [dif_neg commitment, dif_neg publicInput, dif_neg evalK, dif_neg evalA,
      dif_pos proof]
    rfl
  by_cases logical : InRange PiDECStarts.phaseLogicalStart 270 column
  · unfold classifySource
    rw [dif_neg commitment, dif_neg publicInput, dif_neg evalK, dif_neg evalA,
      dif_neg proof, dif_pos logical]
    rfl
  by_cases fresh : InRange PiDECStarts.phaseFreshStart freshCount column
  · unfold classifySource
    rw [dif_neg commitment, dif_neg publicInput, dif_neg evalK, dif_neg evalA,
      dif_neg proof, dif_neg logical, dif_pos fresh]
    rfl
  · exfalso
    rcases support with logicalSource | freshSupport
    · rcases logicalSource with external | logicalSupport
      · rcases external with parent | proofSupport
        · rcases parent with commitmentSupport | publicSupport | evalKSupport |
            evalASupport
          · exact commitment commitmentSupport
          · exact publicInput publicSupport
          · exact evalK evalKSupport
          · exact evalA evalASupport
        · exact proof proofSupport
      · exact logical logicalSupport
    · exact fresh freshSupport

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

theorem classifyTarget_complete {column : Nat} (support : Target column) :
    ∃ decoded, classifyTarget column = some decoded ∧
      Spartan.sourceToSpartan decoded.source = column := by
  rcases support with ⟨source, sourceSupport, rfl⟩
  have inverse := Spartan.spartanToSource_sourceToSpartan source
    (PiDECSourceSupport.source_lt_sourceColumnCount sourceSupport)
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

theorem sourceMap_form_eval_of_target
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth)
    (base : Fin (PiRLCProductPlan.baseSourceWidth program) → F)
    (groupValue : Fin PiRLCProductSchedule.invocationCount → Fin 33 → F)
    (products : Fin PiRLCFirst54DirectSchedule.candidateCount → F)
    (encodes : Encodes geometry assignment
      (PiRLCRetainedPreservation.sourceAssignment program base groupValue products))
    (column : Fin Spartan.spartanColumnCount) (support : Target column.val) :
    ((sourceMap geometry).form column).eval assignment =
      RunningTransitionDirectPlan.transitionEnv program base column.val := by
  rcases classifyTarget_complete support with ⟨decoded, found, mapped⟩
  change (match classifyTarget column.val with
    | none => SparseForm.empty
    | some value => value.location.form geometry).eval assignment = _
  rw [found]
  rw [Location.form_eval geometry assignment _ encodes decoded.location]
  rw [RunningTransitionDirectPlan.sourceAssignment_packageSource program base
    groupValue products decoded.location.sourceColumn
    decoded.location.sourceColumn_lt]
  have mappedLocation :
      Spartan.sourceToSpartan decoded.location.sourceColumn = column.val := by
    rw [decoded.owns, mapped]
  rw [mappedLocation]

private theorem preservesCombination
    {application : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : Geometry application logicalWidth)
    (assignment : Assignment F logicalWidth)
    (base : Fin (PiRLCProductPlan.baseSourceWidth application) → F)
    (groupValue : Fin PiRLCProductSchedule.invocationCount → Fin 33 → F)
    (products : Fin PiRLCFirst54DirectSchedule.candidateCount → F)
    (encodes : Encodes geometry assignment
      (PiRLCRetainedPreservation.sourceAssignment application base groupValue
        products))
    (combination : R1CS.LinearCombination)
    (bounded : SourceCompiler.CombinationBounded Spartan.spartanColumnCount
      combination) (scope : combination.VarsSatisfy Target) :
    OrdinarySourcePlan.SourceMap.PreservesCombination (sourceMap geometry)
      assignment (RunningTransitionDirectPlan.transitionEnv application base)
      combination bounded := by
  intro term member
  exact sourceMap_form_eval_of_target geometry assignment base groupValue products
    encodes ⟨term.1, bounded term member⟩ (scope term member)

def inputs {application : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat}
    (sourceProgram : OrdinarySourcePlan.Program Spartan.spartanColumnCount)
    (geometry : Geometry application logicalWidth) :
    sourceProgram.Inputs logicalWidth where
  oneColumn := oneColumn geometry
  sourceMap := fun _ => sourceMap geometry

theorem inputs_preserve
    {application : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (sourceProgram : OrdinarySourcePlan.Program Spartan.spartanColumnCount)
    (geometry : Geometry application logicalWidth)
    (assignment : Assignment F logicalWidth)
    (base : Fin (PiRLCProductPlan.baseSourceWidth application) → F)
    (groupValue : Fin PiRLCProductSchedule.invocationCount → Fin 33 → F)
    (products : Fin PiRLCFirst54DirectSchedule.candidateCount → F)
    (encodes : Encodes geometry assignment
      (PiRLCRetainedPreservation.sourceAssignment application base groupValue
        products))
    (scope : ∀ index, (sourceProgram.row index).VarsSatisfy Target) :
    ∀ index, OrdinarySourcePlan.SourceMap.PreservesRow
      ((inputs sourceProgram geometry).sourceMap index) assignment
      (RunningTransitionDirectPlan.transitionEnv application base)
      (sourceProgram.row index) (sourceProgram.bounded index) := by
  intro index
  have rowScope := scope index
  exact ⟨
    preservesCombination geometry assignment base groupValue products encodes
      _ _ rowScope.1,
    preservesCombination geometry assignment base groupValue products encodes
      _ _ rowScope.2.1,
    preservesCombination geometry assignment base groupValue products encodes
      _ _ rowScope.2.2⟩

structure SupportedProgram (rows : List R1CS.Row) where
  rowCount : Nat
  rowCount_le : rowCount ≤ 2 ^ Lifecycle.cubeVariables
  row : Fin rowCount → R1CS.Row
  exactRows : List.ofFn row = rows
  supported : ∀ index, (row index).VarsSatisfy Target

def SupportedProgram.toProgram {rows : List R1CS.Row}
    (source : SupportedProgram rows) :
    OrdinarySourcePlan.Program Spartan.spartanColumnCount where
  rowCount := source.rowCount
  rowCount_le := source.rowCount_le
  row := source.row
  bounded := fun index => (source.supported index).mono _
    (fun _ support => PiDECOrdinaryDirectSource.target_lt_spartanColumnCount support)

section

variable {relationLogicalWidth : Nat}
  {relationPublicFits : ringDegree * publicRingColumns ≤
    Phi81CarrierLayout.carrierWidth relationLogicalWidth}

def publicSource
    (relation : ProductionKey.LogicalRelation relationLogicalWidth
      relationPublicFits) :
    SupportedProgram (PiDECOrdinaryDirectSource.publicRows
      relationLogicalWidth relationPublicFits) where
  rowCount := 22680
  rowCount_le := by norm_num [Lifecycle.cubeVariables]
  row := PiDECOrdinaryDirectSource.publicProgramRow relation
  exactRows := PiDECOrdinaryDirectSource.publicProgramRows_eq relation
  supported := PiDECOrdinaryDirectSource.publicProgramRow_varsSatisfy relation

def commitmentSource
    (relation : ProductionKey.LogicalRelation relationLogicalWidth
      relationPublicFits) :
    SupportedProgram (PiDECOrdinaryDirectSource.commitmentRows
      relationLogicalWidth relationPublicFits) where
  rowCount := 1188
  rowCount_le := by norm_num [Lifecycle.cubeVariables]
  row := PiDECOrdinaryDirectSource.commitmentProgramRow relation
  exactRows := PiDECOrdinaryDirectSource.commitmentProgramRows_eq relation
  supported := PiDECOrdinaryDirectSource.commitmentProgramRow_varsSatisfy relation

def evalKSource
    (relation : ProductionKey.LogicalRelation relationLogicalWidth
      relationPublicFits) :
    SupportedProgram (PiDECOrdinaryDirectSource.evalKRows
      relationLogicalWidth relationPublicFits) where
  rowCount := 108
  rowCount_le := by norm_num [Lifecycle.cubeVariables]
  row := PiDECOrdinaryDirectSource.evalKProgramRow relation
  exactRows := PiDECOrdinaryDirectSource.evalKProgramRows_eq relation
  supported := PiDECOrdinaryDirectSource.evalKProgramRow_varsSatisfy relation

def evalASource
    (relation : ProductionKey.LogicalRelation relationLogicalWidth
      relationPublicFits) :
    SupportedProgram (PiDECOrdinaryDirectSource.evalARows
      relationLogicalWidth relationPublicFits) where
  rowCount := 1512
  rowCount_le := by norm_num [Lifecycle.cubeVariables]
  row := PiDECOrdinaryDirectSource.evalAProgramRow relation
  exactRows := PiDECOrdinaryDirectSource.evalAProgramRows_eq relation
  supported := PiDECOrdinaryDirectSource.evalAProgramRow_varsSatisfy relation

def publicPlan {application : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat}
    (relation : ProductionKey.LogicalRelation relationLogicalWidth
      relationPublicFits)
    (geometry : Geometry application logicalWidth) : ProductionRelation.Plan logicalWidth :=
  (publicSource relation).toProgram.compile
    (inputs (publicSource relation).toProgram geometry) |>.toPlan

def commitmentPlan {application : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat}
    (relation : ProductionKey.LogicalRelation relationLogicalWidth
      relationPublicFits)
    (geometry : Geometry application logicalWidth) : ProductionRelation.Plan logicalWidth :=
  (commitmentSource relation).toProgram.compile
    (inputs (commitmentSource relation).toProgram geometry) |>.toPlan

def evalKPlan {application : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat}
    (relation : ProductionKey.LogicalRelation relationLogicalWidth
      relationPublicFits)
    (geometry : Geometry application logicalWidth) : ProductionRelation.Plan logicalWidth :=
  (evalKSource relation).toProgram.compile
    (inputs (evalKSource relation).toProgram geometry) |>.toPlan

def evalAPlan {application : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat}
    (relation : ProductionKey.LogicalRelation relationLogicalWidth
      relationPublicFits)
    (geometry : Geometry application logicalWidth) : ProductionRelation.Plan logicalWidth :=
  (evalASource relation).toProgram.compile
    (inputs (evalASource relation).toProgram geometry) |>.toPlan

@[simp] theorem publicPlan_rowCount
    {application : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (relation : ProductionKey.LogicalRelation relationLogicalWidth
      relationPublicFits)
    (geometry : Geometry application logicalWidth) :
    (publicPlan relation geometry).rowCount = 22680 := by
  rfl

@[simp] theorem commitmentPlan_rowCount
    {application : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (relation : ProductionKey.LogicalRelation relationLogicalWidth
      relationPublicFits)
    (geometry : Geometry application logicalWidth) :
    (commitmentPlan relation geometry).rowCount = 1188 := by
  rfl

@[simp] theorem evalKPlan_rowCount
    {application : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (relation : ProductionKey.LogicalRelation relationLogicalWidth
      relationPublicFits)
    (geometry : Geometry application logicalWidth) :
    (evalKPlan relation geometry).rowCount = 108 := by
  rfl

@[simp] theorem evalAPlan_rowCount
    {application : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (relation : ProductionKey.LogicalRelation relationLogicalWidth
      relationPublicFits)
    (geometry : Geometry application logicalWidth) :
    (evalAPlan relation geometry).rowCount = 1512 := by
  rfl

private theorem evalPlans_fit
    {application : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (relation : ProductionKey.LogicalRelation relationLogicalWidth
      relationPublicFits)
    (geometry : Geometry application logicalWidth) :
    (evalKPlan relation geometry).rowCount + (evalAPlan relation geometry).rowCount ≤
      2 ^ Lifecycle.cubeVariables := by
  rw [evalKPlan_rowCount, evalAPlan_rowCount]
  norm_num [Lifecycle.cubeVariables]

def evaluationPlan {application : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat}
    (relation : ProductionKey.LogicalRelation relationLogicalWidth
      relationPublicFits)
    (geometry : Geometry application logicalWidth) : ProductionRelation.Plan logicalWidth :=
  ProductionRelation.Plan.append (evalKPlan relation geometry)
    (evalAPlan relation geometry) (evalPlans_fit relation geometry)

@[simp] theorem evaluationPlan_rowCount
    {application : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (relation : ProductionKey.LogicalRelation relationLogicalWidth
      relationPublicFits)
    (geometry : Geometry application logicalWidth) :
    (evaluationPlan relation geometry).rowCount = 1620 := by
  simp [evaluationPlan]

private theorem recompositionPlans_fit
    {application : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (relation : ProductionKey.LogicalRelation relationLogicalWidth
      relationPublicFits)
    (geometry : Geometry application logicalWidth) :
    (commitmentPlan relation geometry).rowCount +
        (evaluationPlan relation geometry).rowCount ≤
      2 ^ Lifecycle.cubeVariables := by
  rw [commitmentPlan_rowCount, evaluationPlan_rowCount]
  norm_num [Lifecycle.cubeVariables]

def recompositionPlan {application : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat}
    (relation : ProductionKey.LogicalRelation relationLogicalWidth
      relationPublicFits)
    (geometry : Geometry application logicalWidth) : ProductionRelation.Plan logicalWidth :=
  ProductionRelation.Plan.append (commitmentPlan relation geometry)
    (evaluationPlan relation geometry) (recompositionPlans_fit relation geometry)

@[simp] theorem recompositionPlan_rowCount
    {application : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (relation : ProductionKey.LogicalRelation relationLogicalWidth
      relationPublicFits)
    (geometry : Geometry application logicalWidth) :
    (recompositionPlan relation geometry).rowCount = 2808 := by
  simp [recompositionPlan]

private theorem allPlans_fit
    {application : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (relation : ProductionKey.LogicalRelation relationLogicalWidth
      relationPublicFits)
    (geometry : Geometry application logicalWidth) :
    (publicPlan relation geometry).rowCount +
        (recompositionPlan relation geometry).rowCount ≤
      2 ^ Lifecycle.cubeVariables := by
  rw [publicPlan_rowCount, recompositionPlan_rowCount]
  norm_num [Lifecycle.cubeVariables]

def plan {application : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (relation : ProductionKey.LogicalRelation relationLogicalWidth
      relationPublicFits)
    (geometry : Geometry application logicalWidth) : ProductionRelation.Plan logicalWidth :=
  ProductionRelation.Plan.append (publicPlan relation geometry)
    (recompositionPlan relation geometry) (allPlans_fit relation geometry)

@[simp] theorem plan_rowCount
    {application : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (relation : ProductionKey.LogicalRelation relationLogicalWidth
      relationPublicFits)
    (geometry : Geometry application logicalWidth) :
    (plan relation geometry).rowCount = 25488 := by
  simp [plan]

/-- The PiDEC matrix plan depends on relation shape only. The logical
relation value supplies proof certificates but does not select any row. -/
theorem plan_eq_of_same_shape
    {application : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    {relationLogicalWidth : Nat}
    {relationPublicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth relationLogicalWidth}
    (left right : ProductionKey.LogicalRelation relationLogicalWidth
      relationPublicFits)
    (geometry : Geometry application logicalWidth) :
    plan left geometry = plan right geometry := by
  rfl

private theorem holds_iff_rowsHold_ofFn {count : Nat}
    (rowAt : Fin count → R1CS.Row) (env : Env) :
    (∀ index, (rowAt index).Holds env) ↔
      R1CS.RowsHold env (List.ofFn rowAt) := by
  unfold R1CS.RowsHold
  exact List.forall_mem_ofFn_iff.symm

private theorem predicate_iff_of_eq {Alpha : Type} (predicate : Alpha → Prop)
    {left right : Alpha} (equal : left = right) :
    predicate left ↔ predicate right := by
  cases equal
  rfl

private theorem supportedHolds_iff_rowsHold {rows : List R1CS.Row}
    (source : SupportedProgram rows) (env : Env) :
    source.toProgram.Holds env ↔ R1CS.RowsHold env rows := by
  exact (holds_iff_rowsHold_ofFn source.row env).trans
    (predicate_iff_of_eq (R1CS.RowsHold env) source.exactRows)

private theorem publicHolds_iff_rowsHold
    (relation : ProductionKey.LogicalRelation relationLogicalWidth
      relationPublicFits)
    (env : Env) :
    (publicSource relation).toProgram.Holds env ↔
      R1CS.RowsHold env (PiDECOrdinaryDirectSource.publicRows
        relationLogicalWidth relationPublicFits) := by
  exact supportedHolds_iff_rowsHold (publicSource relation) env

private theorem commitmentHolds_iff_rowsHold
    (relation : ProductionKey.LogicalRelation relationLogicalWidth
      relationPublicFits)
    (env : Env) :
    (commitmentSource relation).toProgram.Holds env ↔
      R1CS.RowsHold env (PiDECOrdinaryDirectSource.commitmentRows
        relationLogicalWidth relationPublicFits) := by
  exact supportedHolds_iff_rowsHold (commitmentSource relation) env

private theorem evalKHolds_iff_rowsHold
    (relation : ProductionKey.LogicalRelation relationLogicalWidth
      relationPublicFits)
    (env : Env) :
    (evalKSource relation).toProgram.Holds env ↔
      R1CS.RowsHold env (PiDECOrdinaryDirectSource.evalKRows
        relationLogicalWidth relationPublicFits) := by
  exact supportedHolds_iff_rowsHold (evalKSource relation) env

private theorem evalAHolds_iff_rowsHold
    (relation : ProductionKey.LogicalRelation relationLogicalWidth
      relationPublicFits)
    (env : Env) :
    (evalASource relation).toProgram.Holds env ↔
      R1CS.RowsHold env (PiDECOrdinaryDirectSource.evalARows
        relationLogicalWidth relationPublicFits) := by
  exact supportedHolds_iff_rowsHold (evalASource relation) env

private theorem compiledRowsZero_iff
    {application : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    {rows : List R1CS.Row} (source : SupportedProgram rows)
    (geometry : Geometry application logicalWidth)
    (assignment : Assignment F logicalWidth)
    (base : Fin (PiRLCProductPlan.baseSourceWidth application) → F)
    (groupValue : Fin PiRLCProductSchedule.invocationCount → Fin 33 → F)
    (products : Fin PiRLCFirst54DirectSchedule.candidateCount → F)
    (one : assignment (oneColumn geometry) = 1)
    (encodes : Encodes geometry assignment
      (PiRLCRetainedPreservation.sourceAssignment application base groupValue
        products)) :
    (source.toProgram.compile (inputs source.toProgram geometry)).toPlan.RowsZero
        assignment ↔
      source.toProgram.Holds
        (RunningTransitionDirectPlan.transitionEnv application base) := by
  exact OrdinarySourcePlan.Program.rowsZero_iff source.toProgram
    (inputs source.toProgram geometry) assignment
    (RunningTransitionDirectPlan.transitionEnv application base) one
    (inputs_preserve source.toProgram geometry assignment base groupValue products
      encodes source.supported)

theorem publicRowsZero_iff_rowsHold
    {application : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (relation : ProductionKey.LogicalRelation relationLogicalWidth
      relationPublicFits)
    (geometry : Geometry application logicalWidth)
    (assignment : Assignment F logicalWidth)
    (base : Fin (PiRLCProductPlan.baseSourceWidth application) → F)
    (groupValue : Fin PiRLCProductSchedule.invocationCount → Fin 33 → F)
    (products : Fin PiRLCFirst54DirectSchedule.candidateCount → F)
    (one : assignment (oneColumn geometry) = 1)
    (encodes : Encodes geometry assignment
      (PiRLCRetainedPreservation.sourceAssignment application base groupValue
        products)) :
    (publicPlan relation geometry).RowsZero assignment ↔
      R1CS.RowsHold (RunningTransitionDirectPlan.transitionEnv application base)
        (PiDECOrdinaryDirectSource.publicRows relationLogicalWidth
          relationPublicFits) := by
  rw [publicPlan, compiledRowsZero_iff (publicSource relation) geometry assignment
    base groupValue products one encodes]
  exact publicHolds_iff_rowsHold relation _

theorem commitmentRowsZero_iff_rowsHold
    {application : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (relation : ProductionKey.LogicalRelation relationLogicalWidth
      relationPublicFits)
    (geometry : Geometry application logicalWidth)
    (assignment : Assignment F logicalWidth)
    (base : Fin (PiRLCProductPlan.baseSourceWidth application) → F)
    (groupValue : Fin PiRLCProductSchedule.invocationCount → Fin 33 → F)
    (products : Fin PiRLCFirst54DirectSchedule.candidateCount → F)
    (one : assignment (oneColumn geometry) = 1)
    (encodes : Encodes geometry assignment
      (PiRLCRetainedPreservation.sourceAssignment application base groupValue
        products)) :
    (commitmentPlan relation geometry).RowsZero assignment ↔
      R1CS.RowsHold (RunningTransitionDirectPlan.transitionEnv application base)
        (PiDECOrdinaryDirectSource.commitmentRows relationLogicalWidth
          relationPublicFits) := by
  rw [commitmentPlan, compiledRowsZero_iff (commitmentSource relation) geometry
    assignment base groupValue products one encodes]
  exact commitmentHolds_iff_rowsHold relation _

theorem evalKRowsZero_iff_rowsHold
    {application : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (relation : ProductionKey.LogicalRelation relationLogicalWidth
      relationPublicFits)
    (geometry : Geometry application logicalWidth)
    (assignment : Assignment F logicalWidth)
    (base : Fin (PiRLCProductPlan.baseSourceWidth application) → F)
    (groupValue : Fin PiRLCProductSchedule.invocationCount → Fin 33 → F)
    (products : Fin PiRLCFirst54DirectSchedule.candidateCount → F)
    (one : assignment (oneColumn geometry) = 1)
    (encodes : Encodes geometry assignment
      (PiRLCRetainedPreservation.sourceAssignment application base groupValue
        products)) :
    (evalKPlan relation geometry).RowsZero assignment ↔
      R1CS.RowsHold (RunningTransitionDirectPlan.transitionEnv application base)
        (PiDECOrdinaryDirectSource.evalKRows relationLogicalWidth
          relationPublicFits) := by
  rw [evalKPlan, compiledRowsZero_iff (evalKSource relation) geometry assignment
    base groupValue products one encodes]
  exact evalKHolds_iff_rowsHold relation _

theorem evalARowsZero_iff_rowsHold
    {application : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (relation : ProductionKey.LogicalRelation relationLogicalWidth
      relationPublicFits)
    (geometry : Geometry application logicalWidth)
    (assignment : Assignment F logicalWidth)
    (base : Fin (PiRLCProductPlan.baseSourceWidth application) → F)
    (groupValue : Fin PiRLCProductSchedule.invocationCount → Fin 33 → F)
    (products : Fin PiRLCFirst54DirectSchedule.candidateCount → F)
    (one : assignment (oneColumn geometry) = 1)
    (encodes : Encodes geometry assignment
      (PiRLCRetainedPreservation.sourceAssignment application base groupValue
        products)) :
    (evalAPlan relation geometry).RowsZero assignment ↔
      R1CS.RowsHold (RunningTransitionDirectPlan.transitionEnv application base)
        (PiDECOrdinaryDirectSource.evalARows relationLogicalWidth
          relationPublicFits) := by
  rw [evalAPlan, compiledRowsZero_iff (evalASource relation) geometry assignment
    base groupValue products one encodes]
  exact evalAHolds_iff_rowsHold relation _

/-- The assembled direct plan vanishes exactly when all canonical PiDEC source
rows hold in their original order. -/
theorem rowsZero_iff_rowsHold
    {application : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (relation : ProductionKey.LogicalRelation relationLogicalWidth
      relationPublicFits)
    (geometry : Geometry application logicalWidth)
    (assignment : Assignment F logicalWidth)
    (base : Fin (PiRLCProductPlan.baseSourceWidth application) → F)
    (groupValue : Fin PiRLCProductSchedule.invocationCount → Fin 33 → F)
    (products : Fin PiRLCFirst54DirectSchedule.candidateCount → F)
    (one : assignment (oneColumn geometry) = 1)
    (encodes : Encodes geometry assignment
      (PiRLCRetainedPreservation.sourceAssignment application base groupValue
        products)) :
    (plan relation geometry).RowsZero assignment ↔
      R1CS.RowsHold (RunningTransitionDirectPlan.transitionEnv application base)
        (PiDECOrdinaryDirectSource.sourceRows relationLogicalWidth
          relationPublicFits) := by
  rw [plan, ProductionRelation.Plan.append_rowsZero_iff]
  rw [recompositionPlan, ProductionRelation.Plan.append_rowsZero_iff]
  rw [evaluationPlan, ProductionRelation.Plan.append_rowsZero_iff]
  rw [publicRowsZero_iff_rowsHold relation geometry assignment base groupValue
    products one encodes]
  rw [commitmentRowsZero_iff_rowsHold relation geometry assignment base groupValue
    products one encodes]
  rw [evalKRowsZero_iff_rowsHold relation geometry assignment base groupValue
    products one encodes]
  rw [evalARowsZero_iff_rowsHold relation geometry assignment base groupValue
    products one encodes]
  simp only [PiDECOrdinaryDirectSource.sourceRows, R1CS.rowsHold_append]
  constructor
  · rintro ⟨publicHolds, commitmentHolds, evalKHolds, evalAHolds⟩
    exact ⟨⟨⟨publicHolds, commitmentHolds⟩, evalKHolds⟩, evalAHolds⟩
  · rintro ⟨⟨⟨publicHolds, commitmentHolds⟩, evalKHolds⟩, evalAHolds⟩
    exact ⟨publicHolds, commitmentHolds, evalKHolds, evalAHolds⟩

/-- The direct plan is exactly the canonical Lean-lowered PiDEC relation. -/
theorem rowsZero_iff_canonicalRowsHold
    {application : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (relation : ProductionKey.LogicalRelation relationLogicalWidth
      relationPublicFits)
    (geometry : Geometry application logicalWidth)
    (assignment : Assignment F logicalWidth)
    (base : Fin (PiRLCProductPlan.baseSourceWidth application) → F)
    (groupValue : Fin PiRLCProductSchedule.invocationCount → Fin 33 → F)
    (products : Fin PiRLCFirst54DirectSchedule.candidateCount → F)
    (one : assignment (oneColumn geometry) = 1)
    (encodes : Encodes geometry assignment
      (PiRLCRetainedPreservation.sourceAssignment application base groupValue
        products)) :
    (plan relation geometry).RowsZero assignment ↔
      R1CS.RowsHold (RunningTransitionDirectPlan.transitionEnv application base)
        ((PiDECArithmetic.canonicalPlan relationLogicalWidth
          relationPublicFits).rows.map
          Rows.CompiledRow.toR1CS) := by
  exact (rowsZero_iff_rowsHold relation geometry assignment base groupValue
    products one encodes).trans
      (predicate_iff_of_eq
        (R1CS.RowsHold (RunningTransitionDirectPlan.transitionEnv application base))
        PiDECOrdinaryDirectSource.sourceRows_eq_canonical)

/-- Acceptance by the direct 14-matrix plan implies the exact PiDEC phase
predicate under the existing formal input assumptions. -/
theorem rowsZero_implies_phaseHolds
    {application : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (relation : ProductionKey.LogicalRelation relationLogicalWidth
      relationPublicFits)
    (ajtai : AjtaiKey
      (logicalWidth := relationLogicalWidth)
      (publicFits := relationPublicFits))
    (geometry : Geometry application logicalWidth)
    (assignment : Assignment F logicalWidth)
    (base : Fin (PiRLCProductPlan.baseSourceWidth application) → F)
    (groupValue : Fin PiRLCProductSchedule.invocationCount → Fin 33 → F)
    (products : Fin PiRLCFirst54DirectSchedule.candidateCount → F)
    (one : assignment (oneColumn geometry) = 1)
    (encodes : Encodes geometry assignment
      (PiRLCRetainedPreservation.sourceAssignment application base groupValue
        products))
    (assumptions :
      Lifecycle.PiDEC.v1_1.Formal.Assumptions relation
        (PiDECArithmetic.phaseInterface relationLogicalWidth
          relationPublicFits)
        PiDECInputs.phaseOffset
        (Spartan.pullback
          (RunningTransitionDirectPlan.transitionEnv application base)))
    (accepted : (plan relation geometry).RowsZero assignment) :
    Lifecycle.PiDEC.v1_1.Semantics.PhaseHolds relation ajtai
      (PiDECArithmetic.phaseInterface relationLogicalWidth relationPublicFits)
      PiDECInputs.phaseOffset
        (Spartan.pullback
          (RunningTransitionDirectPlan.transitionEnv application base)) := by
  have canonicalRows :=
    (rowsZero_iff_canonicalRowsHold relation geometry assignment base groupValue
      products one encodes).mp accepted
  have exactRows := PiDECArithmetic.Plan.rows_to_layout
    (PiDECArithmetic.canonicalPlan relationLogicalWidth relationPublicFits)
    (PiDECArithmetic.canonicalLayoutPlan relation)
    (PiDECArithmetic.canonicalPlan_matches relation)
  have remappedRows : R1CS.RowsHold
      (RunningTransitionDirectPlan.transitionEnv application base)
      (Spartan.remapRows (PiDECArithmetic.canonicalLayoutPlan relation).rows) :=
    (predicate_iff_of_eq
      (R1CS.RowsHold (RunningTransitionDirectPlan.transitionEnv application base))
      exactRows).mp canonicalRows
  have physicalRows : R1CS.RowsHold
      (Spartan.pullback
        (RunningTransitionDirectPlan.transitionEnv application base))
      (PiDECArithmetic.canonicalLayoutPlan relation).rows :=
    (Spartan.remapRows_hold
      (RunningTransitionDirectPlan.transitionEnv application base)
      (PiDECArithmetic.canonicalLayoutPlan relation).rows).mp remappedRows
  exact Layout.PiDEC.v1_1.physical_implies_phaseHolds relation ajtai
    (PiDECArithmetic.phaseInterface relationLogicalWidth relationPublicFits)
    PiDECInputs.phaseOffset
    (Spartan.pullback
      (RunningTransitionDirectPlan.transitionEnv application base))
    assumptions physicalRows

end

end NightstreamFPrime.Export.Stage1.PiDECDirectPlan
