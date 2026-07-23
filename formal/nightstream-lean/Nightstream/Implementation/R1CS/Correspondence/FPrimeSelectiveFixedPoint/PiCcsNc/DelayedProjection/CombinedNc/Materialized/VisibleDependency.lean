import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.RewriteBlockSemantics
import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.SourceDisposition.Pivots

/-!
Bounded dependency certificate for the 1,689 visible combined-NC outputs.

Owns: strict backward references for the 748 physical compiler definitions
and 941 rewrite-terminal pivots, exact per-shard projected cardinalities, and
kernel eliminators exposing the source of every dependency.

Does not own: selected-row satisfaction, rewrite recurrence truth, assignment
agreement, source-program reconstruction, transcript authority, commitment
binding, costs, or permission to remove rows.

The executable subjects are compact proof-free summaries.  The seven physical
subjects contain exactly `60, 128, 128, 128, 128, 128, 48` records.  The 26
pivot subjects contain chain/step record pairs of exactly
`62/62, 64/64, 64/64, 64/64, 64/64, 64/64, 64/64, 64/64, 50/50,
7/43, 14/86, 7/43, 14/86, 7/43, 14/86, 7/43, 14/86, 7/43,
3/39, 21/57, 57/63, 64/64, 59/65, 64/64, 20/20, 2/2` records.
Thus the largest native subject has 128 projected records.

Those summaries account for 1,019 physical RHS references, 22,459 raw rewrite
reference occurrences, and 51 terminal-tail references.  After per-step
deduplication and restriction to visible outputs, the rewrite summaries cover
exactly 6,480 contribution edges plus 51 terminal-tail edges.  The 6,531
visible rewrite edges are never materialized as one list.

Assurance tier: artifact-checked for the fixed generated production profile
once this leaf validates.
-/

/-!
Emits constraints: none; this module proves the visible dependency schedule.

| Stable stage path | Obligation | Authority class |
|---|---|---|
| `f_prime.pi_ccs_nc.delayed.combined.visible_dependency` | Prove every rewrite input is fixed by the visible boundary or an earlier definition. | derived |

-/

namespace Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.VisibleDependency

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Program
open Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc
open Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized
open RewriteBatchIndex
open RewriteBlockSemantics
open SourceDisposition

/-! ## Shared dependency vocabulary -/

/-- A visible proof step may read a compiler-phase input (literal source input
or an exact rewrite pivot) or a physical compiler output. -/
def AllowedEarlier (target column : Nat) : Prop :=
  (column ∈ CompilerExecution.physicalInputColumns ∨
      column ∈ physicalDefinitionOutputs) ∧
    column < target

instance (target column : Nat) : Decidable (AllowedEarlier target column) := by
  unfold AllowedEarlier
  infer_instance

/-- Outputs whose agreement is established by the visible induction.  This is
used only to count the compact dependency edges; authority still comes from
`AllowedEarlier`, not membership in this list. -/
def visibleOutputColumns : List Nat :=
  physicalDefinitionOutputs ++ terminalPivotColumns

private theorem boolAnd_left {left right : Bool}
    (both : (left && right) = true) : left = true := by
  cases left <;> cases right <;> simp_all

private theorem boolAnd_right {left right : Bool}
    (both : (left && right) = true) : right = true := by
  cases left <;> cases right <;> simp_all

/-! ## Physical compiler definitions -/

def PhysicalReferencesEarlier (definition : Definition) : Prop :=
  ∀ column ∈ definition.rhs.refs, column < definition.output

instance (definition : Definition) :
    Decidable (PhysicalReferencesEarlier definition) := by
  unfold PhysicalReferencesEarlier
  infer_instance

structure PhysicalDependencyShape where
  output : Nat
  referenceCount : Nat
  referencesEarlier : Bool
deriving DecidableEq, Repr

def physicalDependencyShape
    (definition : Definition) : PhysicalDependencyShape :=
  { output := definition.output
    referenceCount := definition.rhs.refs.length
    referencesEarlier := decide (PhysicalReferencesEarlier definition) }

def physicalShapeCheck (values : List PhysicalDependencyShape) : Bool :=
  values.all PhysicalDependencyShape.referencesEarlier

def physicalReferenceCount (values : List PhysicalDependencyShape) : Nat :=
  (values.map PhysicalDependencyShape.referenceCount).sum

def physicalChunk0Shapes : List PhysicalDependencyShape :=
  CompilerExecution.physicalChunk0Definitions.map physicalDependencyShape

def physicalChunk1Shapes : List PhysicalDependencyShape :=
  CompilerExecution.physicalChunk1Definitions.map physicalDependencyShape

def physicalChunk2Shapes : List PhysicalDependencyShape :=
  CompilerExecution.physicalChunk2Definitions.map physicalDependencyShape

def physicalChunk3Shapes : List PhysicalDependencyShape :=
  CompilerExecution.physicalChunk3Definitions.map physicalDependencyShape

def physicalChunk4Shapes : List PhysicalDependencyShape :=
  CompilerExecution.physicalChunk4Definitions.map physicalDependencyShape

def physicalChunk5Shapes : List PhysicalDependencyShape :=
  CompilerExecution.physicalChunk5Definitions.map physicalDependencyShape

def physicalChunk6Shapes : List PhysicalDependencyShape :=
  CompilerExecution.physicalChunk6Definitions.map physicalDependencyShape

/-! Each certificate receives only the named proof-free shape list.  Exact
outer/reference cardinalities are `60/0, 128/0, 128/32, 128/256, 128/256,
128/328, 48/147`.  The last two counts include the separately stored nonzero
constant after it is materialized as a reference to column zero. -/

set_option maxRecDepth 100000 in
private theorem physicalChunk0Certificate :
    physicalShapeCheck physicalChunk0Shapes = true ∧
      physicalChunk0Shapes.length = 60 ∧
      physicalReferenceCount physicalChunk0Shapes = 0 := by
  native_decide

set_option maxRecDepth 100000 in
private theorem physicalChunk1Certificate :
    physicalShapeCheck physicalChunk1Shapes = true ∧
      physicalChunk1Shapes.length = 128 ∧
      physicalReferenceCount physicalChunk1Shapes = 0 := by
  native_decide

set_option maxRecDepth 100000 in
private theorem physicalChunk2Certificate :
    physicalShapeCheck physicalChunk2Shapes = true ∧
      physicalChunk2Shapes.length = 128 ∧
      physicalReferenceCount physicalChunk2Shapes = 32 := by
  native_decide

set_option maxRecDepth 100000 in
private theorem physicalChunk3Certificate :
    physicalShapeCheck physicalChunk3Shapes = true ∧
      physicalChunk3Shapes.length = 128 ∧
      physicalReferenceCount physicalChunk3Shapes = 256 := by
  native_decide

set_option maxRecDepth 100000 in
private theorem physicalChunk4Certificate :
    physicalShapeCheck physicalChunk4Shapes = true ∧
      physicalChunk4Shapes.length = 128 ∧
      physicalReferenceCount physicalChunk4Shapes = 256 := by
  native_decide

set_option maxRecDepth 100000 in
private theorem physicalChunk5Certificate :
    physicalShapeCheck physicalChunk5Shapes = true ∧
      physicalChunk5Shapes.length = 128 ∧
      physicalReferenceCount physicalChunk5Shapes = 328 := by
  native_decide

set_option maxRecDepth 100000 in
private theorem physicalChunk6Certificate :
    physicalShapeCheck physicalChunk6Shapes = true ∧
      physicalChunk6Shapes.length = 48 ∧
      physicalReferenceCount physicalChunk6Shapes = 147 := by
  native_decide

private theorem physicalReference_lt_of_check
    {definitions : List Definition}
    (checked : physicalShapeCheck
      (definitions.map physicalDependencyShape) = true)
    {definition : Definition} (member : definition ∈ definitions)
    {column : Nat} (reference : column ∈ definition.rhs.refs) :
    column < definition.output := by
  have shapeMember : physicalDependencyShape definition ∈
      definitions.map physicalDependencyShape :=
    List.mem_map.mpr ⟨definition, member, rfl⟩
  have shapeTrue := (List.all_eq_true.mp checked) _ shapeMember
  have earlier : PhysicalReferencesEarlier definition :=
    of_decide_eq_true (by
      simpa only [physicalDependencyShape] using shapeTrue)
  exact earlier column reference

private theorem physicalReference_lt
    {definition : Definition}
    (member : definition ∈ CompilerExecution.physicalDefinitions)
    {column : Nat} (reference : column ∈ definition.rhs.refs) :
    column < definition.output := by
  simp only [CompilerExecution.physicalDefinitions,
    List.mem_append] at member
  rcases member with member | member | member | member | member | member | member
  · exact physicalReference_lt_of_check
      (by simpa only [physicalChunk0Shapes] using
        physicalChunk0Certificate.1) member reference
  · exact physicalReference_lt_of_check
      (by simpa only [physicalChunk1Shapes] using
        physicalChunk1Certificate.1) member reference
  · exact physicalReference_lt_of_check
      (by simpa only [physicalChunk2Shapes] using
        physicalChunk2Certificate.1) member reference
  · exact physicalReference_lt_of_check
      (by simpa only [physicalChunk3Shapes] using
        physicalChunk3Certificate.1) member reference
  · exact physicalReference_lt_of_check
      (by simpa only [physicalChunk4Shapes] using
        physicalChunk4Certificate.1) member reference
  · exact physicalReference_lt_of_check
      (by simpa only [physicalChunk5Shapes] using
        physicalChunk5Certificate.1) member reference
  · exact physicalReference_lt_of_check
      (by simpa only [physicalChunk6Shapes] using
        physicalChunk6Certificate.1) member reference

/-- Every physical RHS reference is an exact compiler input and is strictly
smaller than the definition output.  The second disjunct of `AllowedEarlier`
is retained in the common interface for rewrite steps. -/
theorem physicalReference_allowedEarlier
    {definition : Definition}
    (member : definition ∈ CompilerExecution.physicalDefinitions)
    {column : Nat} (reference : column ∈ definition.rhs.refs) :
    AllowedEarlier definition.output column := by
  constructor
  · left
    exact CompilerExecution.physicalDefinitionsIndependentValid.referencesOnly
      definition member column reference
  · exact physicalReference_lt member reference

/-! ## Rewrite-terminal pivot chains -/

def StepDependencies (target : Nat) (step : RawRewriteStep) : Prop :=
  ∀ column ∈ rawContributionReferences step, AllowedEarlier target column

instance (target : Nat) (step : RawRewriteStep) :
    Decidable (StepDependencies target step) := by
  unfold StepDependencies
  infer_instance

def TailDependencies (target : Nat)
    (output : RawLinearCombination) : Prop :=
  ∀ term ∈ output.terms.drop 1, AllowedEarlier target term.column

instance (target : Nat) (output : RawLinearCombination) :
    Decidable (TailDependencies target output) := by
  unfold TailDependencies
  infer_instance

structure RewriteStepDependencyShape where
  referenceCount : Nat
  visibleReferenceCount : Nat
  allowedEarlier : Bool
deriving DecidableEq, Repr

def rewriteStepDependencyShape (target : Nat)
    (step : RawRewriteStep) : RewriteStepDependencyShape :=
  let references := rawContributionReferences step
  { referenceCount := references.length
    visibleReferenceCount :=
      (references.eraseDups.filter fun column =>
        decide (column ∈ visibleOutputColumns)).length
    allowedEarlier := decide (StepDependencies target step) }

structure PivotChainDependencyShape where
  target : Nat
  resolved : Bool
  steps : List RewriteStepDependencyShape
  tailReferenceCount : Nat
  tailAllowedEarlier : Bool
deriving DecidableEq, Repr

/-- Total proof-free projection.  Missing source output or target information
sets `resolved = false`; no failed lookup is silently dropped. -/
def pivotChainDependencyShape
    (chain : List RawRewriteStep) : PivotChainDependencyShape :=
  match rawSourceOutput? chain, rawChainTarget? chain with
  | some output, some target =>
      { target
        resolved := true
        steps := chain.map (rewriteStepDependencyShape target)
        tailReferenceCount := (output.terms.drop 1).length
        tailAllowedEarlier := decide (TailDependencies target output) }
  | _, _ =>
      { target := 0
        resolved := false
        steps := []
        tailReferenceCount := 0
        tailAllowedEarlier := false }

def pivotShapeCheck (values : List PivotChainDependencyShape) : Bool :=
  values.all fun shape =>
    shape.resolved &&
      (shape.steps.all RewriteStepDependencyShape.allowedEarlier &&
        (shape.tailAllowedEarlier &&
          decide (shape.tailReferenceCount ≤ 2)))

def pivotStepCount (values : List PivotChainDependencyShape) : Nat :=
  (values.map fun chain => chain.steps.length).sum

def pivotReferenceCount (values : List PivotChainDependencyShape) : Nat :=
  (values.map fun chain =>
    (chain.steps.map RewriteStepDependencyShape.referenceCount).sum).sum

def pivotVisibleReferenceCount
    (values : List PivotChainDependencyShape) : Nat :=
  (values.map fun chain =>
    (chain.steps.map
      RewriteStepDependencyShape.visibleReferenceCount).sum).sum

def pivotTailReferenceCount
    (values : List PivotChainDependencyShape) : Nat :=
  (values.map PivotChainDependencyShape.tailReferenceCount).sum

def batchChainsOrEmpty (batch : Batch) : List (List RawRewriteStep) :=
  (batchChains? batch).getD []

def pivotChainsForBatches
    (batches : List Batch) : List (List RawRewriteStep) :=
  batches.flatMap batchChainsOrEmpty

def pivotShapesForBatches
    (batches : List Batch) : List PivotChainDependencyShape :=
  (pivotChainsForBatches batches).map pivotChainDependencyShape

/-! ## Exact bounded pivot subjects -/

private def pivotChunk0 := pivotShapesForBatches result0.closed
private def pivotChunk1 := pivotShapesForBatches result1.closed
private def pivotChunk2 := pivotShapesForBatches result2.closed
private def pivotChunk3 := pivotShapesForBatches result3.closed
private def pivotChunk4 := pivotShapesForBatches result4.closed
private def pivotChunk5 := pivotShapesForBatches result5.closed
private def pivotChunk6 := pivotShapesForBatches result6.closed
private def pivotChunk7 := pivotShapesForBatches result7.closed
private def pivotChunk8 := pivotShapesForBatches result8.closed
private def pivotChunk9 := pivotShapesForBatches result9.closed
private def pivotChunk10 := pivotShapesForBatches result10.closed
private def pivotChunk11 := pivotShapesForBatches result11.closed
private def pivotChunk12 := pivotShapesForBatches result12.closed
private def pivotChunk13 := pivotShapesForBatches result13.closed
private def pivotChunk14 := pivotShapesForBatches result14.closed
private def pivotChunk15 := pivotShapesForBatches result15.closed
private def pivotChunk16 := pivotShapesForBatches result16.closed
private def pivotChunk17 := pivotShapesForBatches result17.closed
private def pivotChunk18Head :=
  pivotShapesForBatches (result18.closed.take 1)
private def pivotChunk18Tail :=
  pivotShapesForBatches (result18.closed.drop 1)
private def pivotChunk19 := pivotShapesForBatches result19.closed
private def pivotChunk20 := pivotShapesForBatches result20.closed
private def pivotChunk21 := pivotShapesForBatches result21.closed
private def pivotChunk22 := pivotShapesForBatches result22.closed
private def pivotChunk23 := pivotShapesForBatches result23.closed
private def pivotFinalCarry := pivotShapesForBatches [carry23]

/-- One exact native subject contract.  `chains + steps ≤ 128` is checked
outside this predicate by the literal cardinalities in each theorem below. -/
def PivotSubjectExact (values : List PivotChainDependencyShape)
    (chains steps references visibleReferences tails : Nat) : Prop :=
  pivotShapeCheck values = true ∧
    values.length = chains ∧
    pivotStepCount values = steps ∧
    pivotReferenceCount values = references ∧
    pivotVisibleReferenceCount values = visibleReferences ∧
    pivotTailReferenceCount values = tails

instance (values : List PivotChainDependencyShape)
    (chains steps references visibleReferences tails : Nat) :
    Decidable (PivotSubjectExact values chains steps references
      visibleReferences tails) := by
  unfold PivotSubjectExact
  infer_instance

set_option maxRecDepth 100000 in
private theorem pivotChunk0Certificate :
    PivotSubjectExact pivotChunk0 62 62 558 120 0 := by native_decide
set_option maxRecDepth 100000 in
private theorem pivotChunk1Certificate :
    PivotSubjectExact pivotChunk1 64 64 576 116 0 := by native_decide
set_option maxRecDepth 100000 in
private theorem pivotChunk2Certificate :
    PivotSubjectExact pivotChunk2 64 64 576 96 0 := by native_decide
set_option maxRecDepth 100000 in
private theorem pivotChunk3Certificate :
    PivotSubjectExact pivotChunk3 64 64 576 96 0 := by native_decide
set_option maxRecDepth 100000 in
private theorem pivotChunk4Certificate :
    PivotSubjectExact pivotChunk4 64 64 576 104 0 := by native_decide
set_option maxRecDepth 100000 in
private theorem pivotChunk5Certificate :
    PivotSubjectExact pivotChunk5 64 64 576 128 0 := by native_decide
set_option maxRecDepth 100000 in
private theorem pivotChunk6Certificate :
    PivotSubjectExact pivotChunk6 64 64 576 128 0 := by native_decide
set_option maxRecDepth 100000 in
private theorem pivotChunk7Certificate :
    PivotSubjectExact pivotChunk7 64 64 576 128 0 := by native_decide
set_option maxRecDepth 100000 in
private theorem pivotChunk8Certificate :
    PivotSubjectExact pivotChunk8 50 50 450 100 0 := by native_decide
set_option maxRecDepth 100000 in
private theorem pivotChunk9Certificate :
    PivotSubjectExact pivotChunk9 7 43 971 308 3 := by native_decide
set_option maxRecDepth 100000 in
private theorem pivotChunk10Certificate :
    PivotSubjectExact pivotChunk10 14 86 1942 616 6 := by native_decide
set_option maxRecDepth 100000 in
private theorem pivotChunk11Certificate :
    PivotSubjectExact pivotChunk11 7 43 971 308 3 := by native_decide
set_option maxRecDepth 100000 in
private theorem pivotChunk12Certificate :
    PivotSubjectExact pivotChunk12 14 86 1942 616 6 := by native_decide
set_option maxRecDepth 100000 in
private theorem pivotChunk13Certificate :
    PivotSubjectExact pivotChunk13 7 43 971 308 3 := by native_decide
set_option maxRecDepth 100000 in
private theorem pivotChunk14Certificate :
    PivotSubjectExact pivotChunk14 14 86 1942 616 6 := by native_decide
set_option maxRecDepth 100000 in
private theorem pivotChunk15Certificate :
    PivotSubjectExact pivotChunk15 7 43 971 308 3 := by native_decide
set_option maxRecDepth 100000 in
private theorem pivotChunk16Certificate :
    PivotSubjectExact pivotChunk16 14 86 1942 616 6 := by native_decide
set_option maxRecDepth 100000 in
private theorem pivotChunk17Certificate :
    PivotSubjectExact pivotChunk17 7 43 971 308 3 := by native_decide
set_option maxRecDepth 100000 in
private theorem pivotChunk18HeadCertificate :
    PivotSubjectExact pivotChunk18Head 3 39 935 296 3 := by native_decide
set_option maxRecDepth 100000 in
private theorem pivotChunk18TailCertificate :
    PivotSubjectExact pivotChunk18Tail 21 57 1097 340 3 := by native_decide
set_option maxRecDepth 100000 in
private theorem pivotChunk19Certificate :
    PivotSubjectExact pivotChunk19 57 63 705 220 3 := by native_decide
set_option maxRecDepth 100000 in
private theorem pivotChunk20Certificate :
    PivotSubjectExact pivotChunk20 64 64 576 136 0 := by native_decide
set_option maxRecDepth 100000 in
private theorem pivotChunk21Certificate :
    PivotSubjectExact pivotChunk21 59 65 709 272 3 := by native_decide
set_option maxRecDepth 100000 in
private theorem pivotChunk22Certificate :
    PivotSubjectExact pivotChunk22 64 64 576 132 0 := by native_decide
set_option maxRecDepth 100000 in
private theorem pivotChunk23Certificate :
    PivotSubjectExact pivotChunk23 20 20 180 56 0 := by native_decide
set_option maxRecDepth 100000 in
private theorem pivotFinalCarryCertificate :
    PivotSubjectExact pivotFinalCarry 2 2 18 8 0 := by native_decide

/-! ## Kernel eliminators -/

structure PivotChainDependencies (chain : List RawRewriteStep)
    (output : RawLinearCombination) (target : Nat) : Prop where
  outputExact : rawSourceOutput? chain = some output
  targetExact : rawChainTarget? chain = some target
  stepsEarlier : ∀ step ∈ chain, StepDependencies target step
  tailEarlier : TailDependencies target output
  tailLength : (output.terms.drop 1).length ≤ 2

private theorem pivotChainDependencies_of_check
    {chains : List (List RawRewriteStep)}
    (checked : pivotShapeCheck
      (chains.map pivotChainDependencyShape) = true)
    {chain : List RawRewriteStep} (member : chain ∈ chains) :
    ∃ output target, PivotChainDependencies chain output target := by
  have shapeMember : pivotChainDependencyShape chain ∈
      chains.map pivotChainDependencyShape :=
    List.mem_map.mpr ⟨chain, member, rfl⟩
  have shapeValid := (List.all_eq_true.mp checked) _ shapeMember
  cases outputResult : rawSourceOutput? chain with
  | none =>
      simp [pivotChainDependencyShape, outputResult] at shapeValid
  | some output =>
      cases targetResult : rawChainTarget? chain with
      | none =>
          simp [pivotChainDependencyShape, outputResult, targetResult]
            at shapeValid
      | some target =>
          have _resolved := boolAnd_left shapeValid
          have remainder := boolAnd_right shapeValid
          have stepsChecked := boolAnd_left remainder
          have tailFacts := boolAnd_right remainder
          have tailChecked := boolAnd_left tailFacts
          have tailBound := boolAnd_right tailFacts
          have normalizedSteps :
              (chain.map (rewriteStepDependencyShape target)).all
                RewriteStepDependencyShape.allowedEarlier = true := by
            simpa [pivotChainDependencyShape, outputResult,
              targetResult] using stepsChecked
          have normalizedTail :
              decide (TailDependencies target output) = true := by
            simpa [pivotChainDependencyShape, outputResult,
              targetResult] using tailChecked
          have normalizedTailBound :
              decide ((output.terms.drop 1).length ≤ 2) = true := by
            simpa [pivotChainDependencyShape, outputResult,
              targetResult] using tailBound
          refine ⟨output, target, ?_⟩
          refine
            { outputExact := outputResult
              targetExact := targetResult
              stepsEarlier := ?_
              tailEarlier := ?_
              tailLength := of_decide_eq_true normalizedTailBound }
          · intro step stepMember
            have stepShapeMember : rewriteStepDependencyShape target step ∈
                chain.map (rewriteStepDependencyShape target) :=
              List.mem_map.mpr ⟨step, stepMember, rfl⟩
            have stepTrue :=
              (List.all_eq_true.mp normalizedSteps) _ stepShapeMember
            exact of_decide_eq_true (by
              simpa [rewriteStepDependencyShape] using stepTrue)
          · exact of_decide_eq_true normalizedTail

private theorem chain_mem_pivotChainsForBatches
    {batches : List Batch} {batch : Batch}
    (batchMember : batch ∈ batches)
    {chains : List (List RawRewriteStep)}
    (chainsExact : batchChains? batch = some chains)
    {chain : List RawRewriteStep} (chainMember : chain ∈ chains) :
    chain ∈ pivotChainsForBatches batches := by
  unfold pivotChainsForBatches
  apply List.mem_flatMap.mpr
  refine ⟨batch, batchMember, ?_⟩
  simpa [batchChainsOrEmpty, chainsExact] using chainMember

private theorem pivotChainDependencies_of_subject
    {batches : List Batch}
    (checked : pivotShapeCheck (pivotShapesForBatches batches) = true)
    {batch : Batch} (batchMember : batch ∈ batches)
    {chains : List (List RawRewriteStep)}
    (chainsExact : batchChains? batch = some chains)
    {chain : List RawRewriteStep} (chainMember : chain ∈ chains) :
    ∃ output target, PivotChainDependencies chain output target := by
  apply pivotChainDependencies_of_check
    (by simpa only [pivotShapesForBatches] using checked)
  exact chain_mem_pivotChainsForBatches batchMember chainsExact chainMember

/-- Every generated rewrite chain has an exact terminal target.  Every raw
contribution reference and each of the at-most-two triangular tail references
is a compiler input or physical compiler output strictly below that target.
The batch/chain premises are structural lookup evidence, not semantic
acceptance or assignment agreement. -/
theorem generatedPivotChain_dependencies
    {batch : Batch} (batchMember : batch ∈ allBatches)
    {chains : List (List RawRewriteStep)}
    (chainsExact : batchChains? batch = some chains)
    {chain : List RawRewriteStep} (chainMember : chain ∈ chains) :
    ∃ output target, PivotChainDependencies chain output target := by
  have split : batch ∈ batchChunks.flatten ∨ batch ∈ [carry23] := by
    simpa [allBatches, batches] using batchMember
  rcases split with inChunks | inCarry
  · rcases List.mem_flatten.mp inChunks with
      ⟨chunk, chunkMember, localMember⟩
    simp only [batchChunks, List.mem_cons, List.not_mem_nil,
      or_false] at chunkMember
    rcases chunkMember with
      rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl |
      rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl |
      rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl
    · exact pivotChainDependencies_of_subject pivotChunk0Certificate.1
        localMember chainsExact chainMember
    · exact pivotChainDependencies_of_subject pivotChunk1Certificate.1
        localMember chainsExact chainMember
    · exact pivotChainDependencies_of_subject pivotChunk2Certificate.1
        localMember chainsExact chainMember
    · exact pivotChainDependencies_of_subject pivotChunk3Certificate.1
        localMember chainsExact chainMember
    · exact pivotChainDependencies_of_subject pivotChunk4Certificate.1
        localMember chainsExact chainMember
    · exact pivotChainDependencies_of_subject pivotChunk5Certificate.1
        localMember chainsExact chainMember
    · exact pivotChainDependencies_of_subject pivotChunk6Certificate.1
        localMember chainsExact chainMember
    · exact pivotChainDependencies_of_subject pivotChunk7Certificate.1
        localMember chainsExact chainMember
    · exact pivotChainDependencies_of_subject pivotChunk8Certificate.1
        localMember chainsExact chainMember
    · exact pivotChainDependencies_of_subject pivotChunk9Certificate.1
        localMember chainsExact chainMember
    · exact pivotChainDependencies_of_subject pivotChunk10Certificate.1
        localMember chainsExact chainMember
    · exact pivotChainDependencies_of_subject pivotChunk11Certificate.1
        localMember chainsExact chainMember
    · exact pivotChainDependencies_of_subject pivotChunk12Certificate.1
        localMember chainsExact chainMember
    · exact pivotChainDependencies_of_subject pivotChunk13Certificate.1
        localMember chainsExact chainMember
    · exact pivotChainDependencies_of_subject pivotChunk14Certificate.1
        localMember chainsExact chainMember
    · exact pivotChainDependencies_of_subject pivotChunk15Certificate.1
        localMember chainsExact chainMember
    · exact pivotChainDependencies_of_subject pivotChunk16Certificate.1
        localMember chainsExact chainMember
    · exact pivotChainDependencies_of_subject pivotChunk17Certificate.1
        localMember chainsExact chainMember
    · rw [← List.take_append_drop 1 result18.closed] at localMember
      rcases List.mem_append.mp localMember with headMember | tailMember
      · exact pivotChainDependencies_of_subject
          pivotChunk18HeadCertificate.1 headMember chainsExact chainMember
      · exact pivotChainDependencies_of_subject
          pivotChunk18TailCertificate.1 tailMember chainsExact chainMember
    · exact pivotChainDependencies_of_subject pivotChunk19Certificate.1
        localMember chainsExact chainMember
    · exact pivotChainDependencies_of_subject pivotChunk20Certificate.1
        localMember chainsExact chainMember
    · exact pivotChainDependencies_of_subject pivotChunk21Certificate.1
        localMember chainsExact chainMember
    · exact pivotChainDependencies_of_subject pivotChunk22Certificate.1
        localMember chainsExact chainMember
    · exact pivotChainDependencies_of_subject pivotChunk23Certificate.1
        localMember chainsExact chainMember
  · exact pivotChainDependencies_of_subject pivotFinalCarryCertificate.1
      inCarry chainsExact chainMember

end Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.VisibleDependency
