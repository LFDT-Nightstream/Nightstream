import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.CanonicalStepConstructionPlans
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.HonestAssignment

/-!
Contract: artifact-independent construction of the exact visible Step SSA
coordinates.

Owns:
- canonical input, branch-output, join, continuation, and result coordinates;
- preservation of both branch contexts and all verifier controls;
- the codec-domain premise for one honest Step execution.

Does not own: primitive temporary witnesses, row satisfaction, branch
activation or mux equations, production codecs, Rust behavior, or generated
artifacts.

Emits constraints: no.
-/

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding

open Nightstream.Implementation.Lowering.Typed
open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Vocabulary
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Step

namespace CanonicalStepCompleteness

abbrev StepInputFor (parameters : Parameters) :=
  Step.Canonical.Input
    parameters.State parameters.Witness parameters.Running
    parameters.Fresh parameters.NifsProof

abbrev StepOutputFor (parameters : Parameters) :=
  Nightstream.HyperNova.Construction2.Paper.Output
    parameters.Digest parameters.State parameters.Running 1

/-- The selected one-slot running result carried by a typed Step output. -/
def selectedRunning
    {parameters : Parameters}
    (output : StepOutputFor parameters) : parameters.Running :=
  output.runningNext Vocabulary.Step.selected

/-- Exact codec-domain premise for every value used by the recursive arm and
the final continuation.  The base literal's domain premise remains the
explicit canonical-plan parameter. -/
def AdmissibleExecution
    (parameters : Parameters)
    (profile : Profile parameters)
    (input : StepInputFor parameters)
    (runningNext : parameters.Running) : Prop :=
  HonestAssignment.Admissible (profile.family parameters)
      (afterNifsValues parameters input runningNext) ∧
    HonestAssignment.Admissible (profile.family parameters)
      (afterHashNextValues parameters input runningNext)

/-- Initial values for the verifier one coordinate and both Step branch
activations.  Every typed and temporary coordinate starts at zero. -/
def controlAssignment (selected : Bool) : ColumnId -> Field :=
  fun id =>
    if id = oneColumn then
      1
    else if id =
        activationColumn SourceOwners.stepBranchPath true then
      if selected then 1 else 0
    else if id =
        activationColumn SourceOwners.stepBranchPath false then
      if selected then 0 else 1
    else
      0

def ControlsExact
    (selected : Bool)
    (assignment : ColumnId -> Field) : Prop :=
  assignment oneColumn = 1 ∧
    assignment
        (activationColumn SourceOwners.stepBranchPath true) =
      (if selected then 1 else 0) ∧
    assignment
        (activationColumn SourceOwners.stepBranchPath false) =
      (if selected then 0 else 1)

theorem controlAssignment_exact (selected : Bool) :
    ControlsExact selected (controlAssignment selected) := by
  constructor
  · simp [controlAssignment, oneColumn]
  · constructor <;>
      simp [controlAssignment, oneColumn, activationColumn]

theorem ControlsExact.of_agrees
    {selected : Bool}
    {before after : ColumnId -> Field}
    (exact : ControlsExact selected before)
    (agrees :
      AgreesOn
        [oneColumn,
          activationColumn SourceOwners.stepBranchPath true,
          activationColumn SourceOwners.stepBranchPath false]
        before after) :
    ControlsExact selected after := by
  exact ⟨
    (agrees oneColumn (by simp)).trans exact.1,
    (agrees
      (activationColumn SourceOwners.stepBranchPath true)
      (by simp)).trans exact.2.1,
    (agrees
      (activationColumn SourceOwners.stepBranchPath false)
      (by simp)).trans exact.2.2⟩

private theorem input_ids_nodup
    (parameters : Parameters) :
    (CanonicalContexts.Step.input parameters
      ).toSchemaBundles.ids.Nodup := by
  simpa [CanonicalContexts.Step.input, SchemaBundles.ids,
    ColumnPlan.schemaColumnIds, Columns.toSchemaBundles_columns] using
    (ColumnPlan.allocateSchemaFrom_ids_nodup
      (fun slot => PhysicalOwner.typed (.input slot))
      0 (stepInputSchema parameters))

private theorem joined_ids_nodup
    (parameters : Parameters) :
    (CanonicalContexts.Step.joined parameters
      ).toSchemaBundles.ids.Nodup := by
  simpa [CanonicalContexts.Step.joined, branchJoinColumns,
    SchemaBundles.ids, ColumnPlan.schemaColumnIds,
    Columns.toSchemaBundles_columns] using
    (ColumnPlan.allocateSchemaFrom_ids_nodup
      (fun _ => PhysicalOwner.typed
        (.branch SourceOwners.stepBranchPath))
      0
      (Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.joinedSchema
        parameters))

private theorem input_disjoint_controls
    (parameters : Parameters) :
    IdsDisjoint
      (CanonicalContexts.Step.input parameters).toSchemaBundles.ids
      [oneColumn,
        activationColumn SourceOwners.stepBranchPath true,
        activationColumn SourceOwners.stepBranchPath false] := by
  intro id inputMember controlMember
  simp only [List.mem_cons, List.not_mem_nil, or_false] at controlMember
  rcases controlMember with oneMember | trueMember | falseMember
  · subst id
    exact CanonicalPrimitivePlan.inputDisjointOne
      (stepInputSchema parameters) oneColumn inputMember (by simp)
  · have disjoint :=
      CanonicalPrimitivePlan.inputDisjointActivations
        (stepInputSchema parameters) SourceOwners.stepBranchPath
    exact disjoint id inputMember (by
      simp [activationColumns, trueMember])
  · have disjoint :=
      CanonicalPrimitivePlan.inputDisjointActivations
        (stepInputSchema parameters) SourceOwners.stepBranchPath
    exact disjoint id inputMember (by
      simp [activationColumns, falseMember])

private theorem joined_disjoint_controls
    (parameters : Parameters) :
    IdsDisjoint
      (CanonicalContexts.Step.joined parameters).toSchemaBundles.ids
      [oneColumn,
        activationColumn SourceOwners.stepBranchPath true,
        activationColumn SourceOwners.stepBranchPath false] := by
  intro id joinedMember controlMember
  simp only [List.mem_cons, List.not_mem_nil, or_false] at controlMember
  rcases controlMember with oneMember | trueMember | falseMember
  · subst id
    have disjoint :=
      ColumnPlan.prelude_typed_ids_disjoint
        (types := typeSystem parameters)
        (fun _ => .branch SourceOwners.stepBranchPath)
        0
        (Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.joinedSchema
          parameters)
    exact disjoint oneColumn (by simp) (by
      simpa [CanonicalContexts.Step.joined, branchJoinColumns,
        SchemaBundles.ids, ColumnPlan.schemaColumnIds,
        Columns.toSchemaBundles_columns] using joinedMember)
  · have disjoint :=
      ColumnPlan.typed_activation_ids_disjoint
        (types := typeSystem parameters)
        (fun _ => .branch SourceOwners.stepBranchPath)
        0
        (Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.joinedSchema
          parameters)
        SourceOwners.stepBranchPath
    apply disjoint id
    · simpa [CanonicalContexts.Step.joined, branchJoinColumns,
        SchemaBundles.ids, ColumnPlan.schemaColumnIds,
        Columns.toSchemaBundles_columns] using joinedMember
    · simp [activationColumns, trueMember]
  · have disjoint :=
      ColumnPlan.typed_activation_ids_disjoint
        (types := typeSystem parameters)
        (fun _ => .branch SourceOwners.stepBranchPath)
        0
        (Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.joinedSchema
          parameters)
        SourceOwners.stepBranchPath
    apply disjoint id
    · simpa [CanonicalContexts.Step.joined, branchJoinColumns,
        SchemaBundles.ids, ColumnPlan.schemaColumnIds,
        Columns.toSchemaBundles_columns] using joinedMember
    · simp [activationColumns, falseMember]

private theorem input_disjoint_joined
    (parameters : Parameters) :
    IdsDisjoint
      (CanonicalContexts.Step.input parameters).toSchemaBundles.ids
      (CanonicalContexts.Step.joined parameters).toSchemaBundles.ids := by
  simpa [CanonicalContexts.Step.input, CanonicalContexts.Step.joined,
    SchemaBundles.ids, ColumnPlan.schemaColumnIds,
    Columns.toSchemaBundles_columns] using
    (ColumnPlan.input_branch_ids_disjoint
      SourceOwners.stepBranchPath
      (stepInputSchema parameters)
      (Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.joinedSchema
        parameters))

private theorem instruction_changes_preserve_controls
    {types : TypeSystem}
    (path : OwnerPath)
    (output : Schema types)
    (before after : ColumnId -> Field)
    (changes :
      ChangesOnly
        (instructionColumns path output).toSchemaBundles.ids
        before after) :
    AgreesOn
      [oneColumn,
        activationColumn SourceOwners.stepBranchPath true,
        activationColumn SourceOwners.stepBranchPath false]
      before after := by
  apply agreesOn_of_changesOnly
  · intro id outputMember controlMember
    simp only [List.mem_cons, List.not_mem_nil, or_false] at controlMember
    rcases controlMember with oneMember | trueMember | falseMember
    · subst id
      exact CanonicalPrimitivePlan.instructionOutputsDisjointOne
        path output oneColumn outputMember (by simp)
    · have disjoint :=
        CanonicalPrimitivePlan.instructionOutputsDisjointActivations
          path output SourceOwners.stepBranchPath
      exact disjoint id outputMember (by
        simp [activationColumns, trueMember])
    · have disjoint :=
        CanonicalPrimitivePlan.instructionOutputsDisjointActivations
          path output SourceOwners.stepBranchPath
      exact disjoint id outputMember (by
        simp [activationColumns, falseMember])
  · exact changes

private theorem instruction_changes_preserve_joined
    {parameters : Parameters}
    (path : OwnerPath)
    (output : Schema (typeSystem parameters))
    (before after : ColumnId -> Field)
    (changes :
      ChangesOnly
        (instructionColumns path output).toSchemaBundles.ids
        before after) :
    AgreesOn
      (CanonicalContexts.Step.joined parameters).toSchemaBundles.ids
      before after := by
  apply agreesOn_of_changesOnly
    (changed :=
      (instructionColumns path output).toSchemaBundles.ids)
  · simpa [CanonicalContexts.Step.joined, SchemaBundles.ids,
      ColumnPlan.schemaColumnIds, Columns.toSchemaBundles_columns] using
      (ColumnPlan.instruction_branch_ids_disjoint
        path SourceOwners.stepBranchPath output
        (Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.joinedSchema
          parameters))
  · exact changes

private theorem joined_encoded_preserved_by_instruction
    {parameters : Parameters}
    {profile : Profile parameters}
    {output : Schema (typeSystem parameters)}
    (path : OwnerPath)
    (before after : ColumnId -> Field)
    (runningNext : parameters.Running)
    (changes :
      ChangesOnly
        (instructionColumns path output).toSchemaBundles.ids
        before after)
    (encoded :
      Columns.Encodes (profile.family parameters)
        (CanonicalContexts.Step.joined parameters)
        before (joinedValues parameters runningNext)) :
    Columns.Encodes (profile.family parameters)
      (CanonicalContexts.Step.joined parameters)
      after (joinedValues parameters runningNext) := by
  apply
    (CanonicalContexts.Step.joined parameters
      ).toSchemaBundles.encodes_of_agrees
        (profile.family parameters) before after
        (joinedValues parameters runningNext)
  · exact instruction_changes_preserve_joined
      path output before after changes
  · exact encoded

private theorem encodes_preserved_by_instruction
    {parameters : Parameters}
    {profile : Profile parameters}
    {context output : Schema (typeSystem parameters)}
    (path : OwnerPath)
    (columns : Columns context)
    (values : Schema.Values (typeSystem parameters) context)
    (before after : ColumnId -> Field)
    (excludes :
      CanonicalPrimitivePlan.ContextExcludesOwner
        (.typed (.instruction path)) columns)
    (changes :
      ChangesOnly
        (instructionColumns path output).toSchemaBundles.ids
        before after)
    (encoded :
      Columns.Encodes (profile.family parameters)
        columns before values) :
    Columns.Encodes (profile.family parameters)
      columns after values := by
  apply columns.toSchemaBundles.encodes_of_agrees
    (profile.family parameters) before after values
  · exact agreesOn_of_changesOnly
      (CanonicalPrimitivePlan.ContextExcludesOwner.instructionOutputsDisjoint
        path columns excludes)
      changes
  · exact encoded

private theorem afterBaseLiteral_excludes
    (parameters : Parameters)
    (target : OwnerPath)
    (literalDifferent : SourceOwners.stepBaseDefaultPath ≠ target)
    (equalityDifferent : SourceOwners.stepBaseStateEqualPath ≠ target)
    (selectorDifferent : SourceOwners.stepSelectorPath ≠ target)
    (applyDifferent : SourceOwners.stepApplyPath ≠ target) :
    CanonicalPrimitivePlan.ContextExcludesOwner
      (.typed (.instruction target))
      (CanonicalContexts.Step.afterBaseLiteral parameters) :=
  CanonicalPrimitivePlan.ContextExcludesOwner.append
    (CanonicalPrimitivePlan.ContextExcludesOwner.instruction
      SourceOwners.stepBaseDefaultPath target
      [Ports.committedRunning parameters] literalDifferent)
    (CanonicalStepPlan.afterBaseEquality_excludes parameters target
      equalityDifferent selectorDifferent applyDifferent)

private theorem afterNifs_excludes
    (parameters : Parameters)
    (target : OwnerPath)
    (nifsDifferent : SourceOwners.stepRecursiveNifsPath ≠ target)
    (equalityDifferent :
      SourceOwners.stepRecursiveEncodedEqualPath ≠ target)
    (encodeDifferent : SourceOwners.stepRecursiveEncodePath ≠ target)
    (freshDifferent :
      SourceOwners.stepRecursiveFreshPublicPath ≠ target)
    (hashDifferent : SourceOwners.stepRecursiveHashPriorPath ≠ target)
    (selectorDifferent : SourceOwners.stepSelectorPath ≠ target)
    (applyDifferent : SourceOwners.stepApplyPath ≠ target) :
    CanonicalPrimitivePlan.ContextExcludesOwner
      (.typed (.instruction target))
      (CanonicalContexts.Step.afterNifs parameters) :=
  CanonicalPrimitivePlan.ContextExcludesOwner.append
    (CanonicalPrimitivePlan.ContextExcludesOwner.instruction
      SourceOwners.stepRecursiveNifsPath target
      [Ports.committedRunning parameters] nifsDifferent)
    (CanonicalStepPlan.afterEncodedEquality_excludes parameters target
      equalityDifferent encodeDifferent freshDifferent hashDifferent
      selectorDifferent applyDifferent)

/-- Exact visible contexts retained after the final continuation output is
written.  Temporary call witnesses are deliberately absent. -/
structure VisibleWitness
    (parameters : Parameters)
    (profile : Profile parameters)
    (input : StepInputFor parameters)
    (runningNext : parameters.Running) where
  assignment : ColumnId -> Field
  controls :
    ControlsExact (decide (input.iteration = 0)) assignment
  baseEncoded :
    Columns.Encodes (profile.family parameters)
      (CanonicalContexts.Step.afterBaseLiteral parameters)
      assignment
      (afterBaseLiteralValues parameters input)
  recursiveEncoded :
    Columns.Encodes (profile.family parameters)
      (CanonicalContexts.Step.afterNifs parameters)
      assignment
      (afterNifsValues parameters input runningNext)
  finalEncoded :
    Columns.Encodes (profile.family parameters)
      (CanonicalContexts.Step.afterHashNext parameters)
      assignment
      (afterHashNextValues parameters input runningNext)

theorem VisibleWitness.afterEncodedEqualityEncoded
    {parameters : Parameters}
    {profile : Profile parameters}
    {input : StepInputFor parameters}
    {runningNext : parameters.Running}
    (visible : VisibleWitness parameters profile input runningNext) :
    Columns.Encodes (profile.family parameters)
      (CanonicalContexts.Step.afterEncodedEquality parameters)
      visible.assignment
      (afterEncodedEqualityValues parameters input) := by
  exact Columns.right_encodes_of_append
    (profile.family parameters) visible.assignment
    (instructionColumns SourceOwners.stepRecursiveNifsPath
      [Ports.committedRunning parameters])
    (CanonicalContexts.Step.afterEncodedEquality parameters)
    (.cons runningNext .nil)
    (afterEncodedEqualityValues parameters input)
    visible.recursiveEncoded

theorem VisibleWitness.afterEncodeEncoded
    {parameters : Parameters}
    {profile : Profile parameters}
    {input : StepInputFor parameters}
    {runningNext : parameters.Running}
    (visible : VisibleWitness parameters profile input runningNext) :
    Columns.Encodes (profile.family parameters)
      (CanonicalContexts.Step.afterEncode parameters)
      visible.assignment
      (afterEncodeValues parameters input) := by
  exact Columns.right_encodes_of_append
    (profile.family parameters) visible.assignment
    (instructionColumns SourceOwners.stepRecursiveEncodedEqualPath
      [Ports.auxiliaryBit parameters])
    (CanonicalContexts.Step.afterEncode parameters)
    (.cons (priorLinkAccepted parameters input) .nil)
    (afterEncodeValues parameters input)
    visible.afterEncodedEqualityEncoded

theorem VisibleWitness.afterFreshPublicEncoded
    {parameters : Parameters}
    {profile : Profile parameters}
    {input : StepInputFor parameters}
    {runningNext : parameters.Running}
    (visible : VisibleWitness parameters profile input runningNext) :
    Columns.Encodes (profile.family parameters)
      (CanonicalContexts.Step.afterFreshPublic parameters)
      visible.assignment
      (afterFreshPublicValues parameters input) := by
  exact Columns.right_encodes_of_append
    (profile.family parameters) visible.assignment
    (instructionColumns SourceOwners.stepRecursiveEncodePath
      [Ports.auxiliaryEncoded parameters])
    (CanonicalContexts.Step.afterFreshPublic parameters)
    (.cons
      (parameters.machine.encodeInstance (priorDigest parameters input))
      .nil)
    (afterFreshPublicValues parameters input)
    visible.afterEncodeEncoded

theorem VisibleWitness.afterHashEncoded
    {parameters : Parameters}
    {profile : Profile parameters}
    {input : StepInputFor parameters}
    {runningNext : parameters.Running}
    (visible : VisibleWitness parameters profile input runningNext) :
    Columns.Encodes (profile.family parameters)
      (CanonicalContexts.Step.afterHash parameters)
      visible.assignment
      (afterHashValues parameters input) := by
  exact Columns.right_encodes_of_append
    (profile.family parameters) visible.assignment
    (instructionColumns SourceOwners.stepRecursiveFreshPublicPath
      [Ports.auxiliaryEncoded parameters])
    (CanonicalContexts.Step.afterHash parameters)
    (.cons (parameters.machine.freshPublic input.fresh) .nil)
    (afterHashValues parameters input)
    visible.afterFreshPublicEncoded

theorem VisibleWitness.commonEncoded
    {parameters : Parameters}
    {profile : Profile parameters}
    {input : StepInputFor parameters}
    {runningNext : parameters.Running}
    (visible : VisibleWitness parameters profile input runningNext) :
    Columns.Encodes (profile.family parameters)
      (CanonicalContexts.Step.common parameters)
      visible.assignment
      (commonValues parameters input) := by
  exact Columns.right_encodes_of_append
    (profile.family parameters) visible.assignment
    (instructionColumns SourceOwners.stepRecursiveHashPriorPath
      [Ports.auxiliaryDigest parameters])
    (CanonicalContexts.Step.common parameters)
    (.cons (priorDigest parameters input) .nil)
    (commonValues parameters input)
    visible.afterHashEncoded

theorem VisibleWitness.afterStepEncoded
    {parameters : Parameters}
    {profile : Profile parameters}
    {input : StepInputFor parameters}
    {runningNext : parameters.Running}
    (visible : VisibleWitness parameters profile input runningNext) :
    Columns.Encodes (profile.family parameters)
      (CanonicalContexts.Step.afterStep parameters)
      visible.assignment
      (afterStepValues parameters input) := by
  exact Columns.right_encodes_of_append
    (profile.family parameters) visible.assignment
    (instructionColumns SourceOwners.stepSelectorPath
      [Ports.auxiliaryBit parameters])
    (CanonicalContexts.Step.afterStep parameters)
    (.cons (decide (input.iteration = 0)) .nil)
    (afterStepValues parameters input)
    visible.commonEncoded

theorem VisibleWitness.inputEncoded
    {parameters : Parameters}
    {profile : Profile parameters}
    {input : StepInputFor parameters}
    {runningNext : parameters.Running}
    (visible : VisibleWitness parameters profile input runningNext) :
    Columns.Encodes (profile.family parameters)
      (CanonicalContexts.Step.input parameters)
      visible.assignment
      (stepInputValues parameters input) := by
  exact Columns.right_encodes_of_append
    (profile.family parameters) visible.assignment
    (instructionColumns SourceOwners.stepApplyPath
      [Ports.committedState parameters])
    (CanonicalContexts.Step.input parameters)
    (.cons
      (parameters.machine.step Vocabulary.Step.selected
        input.zi input.witness) .nil)
    (stepInputValues parameters input)
    visible.afterStepEncoded

theorem VisibleWitness.afterBaseEqualityEncoded
    {parameters : Parameters}
    {profile : Profile parameters}
    {input : StepInputFor parameters}
    {runningNext : parameters.Running}
    (visible : VisibleWitness parameters profile input runningNext) :
    Columns.Encodes (profile.family parameters)
      (CanonicalContexts.Step.afterBaseEquality parameters)
      visible.assignment
      (afterBaseEqualityValues parameters input) := by
  exact Columns.right_encodes_of_append
    (profile.family parameters) visible.assignment
    (instructionColumns SourceOwners.stepBaseDefaultPath
      [Ports.committedRunning parameters])
    (CanonicalContexts.Step.afterBaseEquality parameters)
    (.cons (defaultRunning parameters) .nil)
    (afterBaseEqualityValues parameters input)
    visible.baseEncoded

theorem VisibleWitness.continuationInputEncoded
    {parameters : Parameters}
    {profile : Profile parameters}
    {input : StepInputFor parameters}
    {runningNext : parameters.Running}
    (visible : VisibleWitness parameters profile input runningNext) :
    Columns.Encodes (profile.family parameters)
      (CanonicalContexts.Step.continuationInput parameters)
      visible.assignment
      (continuationInputValues parameters input runningNext) := by
  exact Columns.right_encodes_of_append
    (profile.family parameters) visible.assignment
    (instructionColumns SourceOwners.stepContinuationHashPath
      [Ports.publicDigest parameters])
    (CanonicalContexts.Step.continuationInput parameters)
    (.cons (nextDigest parameters input runningNext) .nil)
    (continuationInputValues parameters input runningNext)
    visible.finalEncoded

theorem VisibleWitness.joinedEncoded
    {parameters : Parameters}
    {profile : Profile parameters}
    {input : StepInputFor parameters}
    {runningNext : parameters.Running}
    (visible : VisibleWitness parameters profile input runningNext) :
    Columns.Encodes (profile.family parameters)
      (CanonicalContexts.Step.joined parameters)
      visible.assignment
      (joinedValues parameters runningNext) := by
  exact Columns.left_encodes_of_append
    (profile.family parameters) visible.assignment
    (CanonicalContexts.Step.joined parameters)
    (CanonicalContexts.Step.common parameters)
    (joinedValues parameters runningNext)
    (commonValues parameters input)
    visible.continuationInputEncoded

theorem VisibleWitness.resultEncoded
    {parameters : Parameters}
    {profile : Profile parameters}
    {input : StepInputFor parameters}
    {runningNext : parameters.Running}
    (visible : VisibleWitness parameters profile input runningNext) :
    Columns.Encodes (profile.family parameters)
      (CanonicalContexts.Step.result parameters)
      visible.assignment
      (resultValuesFor parameters input runningNext) := by
  have exported :=
    Columns.export_encodes
      (profile.family parameters) visible.assignment
      (CanonicalContexts.Step.resultExports parameters)
      (CanonicalContexts.Step.resultExportsCompatible parameters)
      (CanonicalContexts.Step.afterHashNext parameters)
      (afterHashNextValues parameters input runningNext)
      visible.finalEncoded
  simpa [CanonicalContexts.Step.result,
    CanonicalContexts.Step.resultExports] using exported

/-- Canonical codecs construct every visible Step coordinate, including both
private-arm outputs and the final joined result, before any temporary recipe
witness is completed. -/
theorem exists_visible
    (parameters : Parameters)
    (profile : Profile parameters)
    (recipes :
      CallRecipes (signature parameters) (profile.family parameters))
    (defaultAdmissible :
      (profile.family parameters).codecFor (.data .running) |>.Admissible
        (defaultRunning parameters))
    (input : StepInputFor parameters)
    (runningNext : parameters.Running)
    (admissible :
      AdmissibleExecution parameters profile input runningNext) :
    Nonempty (VisibleWitness parameters profile input runningNext) := by
  let selected := decide (input.iteration = 0)
  let family := profile.family parameters
  let inputValues := stepInputValues parameters input
  let afterStepValues' := afterStepValues parameters input
  let commonValues' := commonValues parameters input
  let baseEqualityValues' := afterBaseEqualityValues parameters input
  let baseLiteralValues' := afterBaseLiteralValues parameters input
  let afterHashValues' := afterHashValues parameters input
  let afterFreshPublicValues' := afterFreshPublicValues parameters input
  let afterEncodeValues' := afterEncodeValues parameters input
  let afterEqualityValues' := afterEncodedEqualityValues parameters input
  let afterNifsValues' := afterNifsValues parameters input runningNext
  let joinedValues' := joinedValues parameters runningNext
  let continuationValues' :=
    continuationInputValues parameters input runningNext
  let finalValues' := afterHashNextValues parameters input runningNext

  change
    HonestAssignment.Admissible family afterNifsValues' ∧
      HonestAssignment.Admissible family finalValues' at admissible
  have recursiveAdmissible :
      HonestAssignment.Admissible family afterNifsValues' :=
    admissible.1
  have finalAdmissible :
      HonestAssignment.Admissible family finalValues' :=
    admissible.2
  have nifsOutputAdmissible :
      HonestAssignment.Admissible
        (schema := [Ports.committedRunning parameters]) family
        (.cons runningNext .nil) :=
    ⟨recursiveAdmissible.1, True.intro⟩
  have afterEqualityAdmissible :
      HonestAssignment.Admissible family afterEqualityValues' :=
    recursiveAdmissible.2
  have equalityOutputAdmissible :
      HonestAssignment.Admissible
        (schema := [Ports.auxiliaryBit parameters]) family
        (.cons (priorLinkAccepted parameters input) .nil) :=
    ⟨afterEqualityAdmissible.1, True.intro⟩
  have afterEncodeAdmissible :
      HonestAssignment.Admissible family afterEncodeValues' :=
    afterEqualityAdmissible.2
  have encodeOutputAdmissible :
      HonestAssignment.Admissible
        (schema := [Ports.auxiliaryEncoded parameters]) family
        (.cons
          (parameters.machine.encodeInstance
            (priorDigest parameters input)) .nil) :=
    ⟨afterEncodeAdmissible.1, True.intro⟩
  have afterFreshPublicAdmissible :
      HonestAssignment.Admissible family afterFreshPublicValues' :=
    afterEncodeAdmissible.2
  have freshPublicOutputAdmissible :
      HonestAssignment.Admissible
        (schema := [Ports.auxiliaryEncoded parameters]) family
        (.cons (parameters.machine.freshPublic input.fresh) .nil) :=
    ⟨afterFreshPublicAdmissible.1, True.intro⟩
  have afterHashAdmissible :
      HonestAssignment.Admissible family afterHashValues' :=
    afterFreshPublicAdmissible.2
  have hashOutputAdmissible :
      HonestAssignment.Admissible
        (schema := [Ports.auxiliaryDigest parameters]) family
        (.cons (priorDigest parameters input) .nil) :=
    ⟨afterHashAdmissible.1, True.intro⟩
  have commonAdmissible :
      HonestAssignment.Admissible family commonValues' :=
    afterHashAdmissible.2
  have selectorOutputAdmissible :
      HonestAssignment.Admissible
        (schema := [Ports.auxiliaryBit parameters]) family
        (.cons selected .nil) :=
    ⟨commonAdmissible.1, True.intro⟩
  have afterStepAdmissible :
      HonestAssignment.Admissible family afterStepValues' :=
    commonAdmissible.2
  have applyOutputAdmissible :
      HonestAssignment.Admissible
        (schema := [Ports.committedState parameters]) family
        (.cons
          (parameters.machine.step Vocabulary.Step.selected
            input.zi input.witness) .nil) :=
    ⟨afterStepAdmissible.1, True.intro⟩
  have inputAdmissible :
      HonestAssignment.Admissible family inputValues :=
    afterStepAdmissible.2
  have baseEqualityOutputAdmissible :
      HonestAssignment.Admissible
        (schema := [Ports.auxiliaryBit parameters]) family
        (.cons (stateEqual parameters input.z0 input.zi) .nil) := by
    change True ∧ True
    exact ⟨True.intro, True.intro⟩
  have continuationAdmissible :
      HonestAssignment.Admissible family continuationValues' :=
    finalAdmissible.2
  have joinedAdmissible :
      HonestAssignment.Admissible family joinedValues' := by
    change HonestAssignment.Admissible family
      (joinedValues'.append commonValues') at continuationAdmissible
    exact HonestAssignment.Admissible.left_of_append
      family joinedValues' commonValues' continuationAdmissible
  have continuationOutputAdmissible :
      HonestAssignment.Admissible
        (schema := [Ports.publicDigest parameters]) family
        (.cons (nextDigest parameters input runningNext) .nil) :=
    ⟨finalAdmissible.1, True.intro⟩

  let initial := controlAssignment selected
  let joinedAssignment :=
    HonestAssignment.encodeInto family
      (CanonicalContexts.Step.joined parameters)
      joinedValues' initial
  have joinedEncoded :
      Columns.Encodes family
        (CanonicalContexts.Step.joined parameters)
        joinedAssignment joinedValues' := by
    exact HonestAssignment.encodeInto_encodes family
      (CanonicalContexts.Step.joined parameters)
      joinedValues' initial
      (CanonicalContexts.Step.joinedWidths parameters profile)
      joinedAdmissible (joined_ids_nodup parameters)
  have joinedControls :
      ControlsExact selected joinedAssignment := by
    apply (controlAssignment_exact selected).of_agrees
    exact agreesOn_of_changesOnly
      (joined_disjoint_controls parameters)
      (HonestAssignment.encodeInto_changesOnly family
        (CanonicalContexts.Step.joined parameters)
        joinedValues' initial)

  let inputAssignment :=
    HonestAssignment.encodeInto family
      (CanonicalContexts.Step.input parameters)
      inputValues joinedAssignment
  have inputEncoded :
      Columns.Encodes family
        (CanonicalContexts.Step.input parameters)
        inputAssignment inputValues := by
    exact HonestAssignment.encodeInto_encodes family
      (CanonicalContexts.Step.input parameters)
      inputValues joinedAssignment
      (CanonicalContexts.Step.inputWidths parameters profile)
      inputAdmissible (input_ids_nodup parameters)
  have inputControls :
      ControlsExact selected inputAssignment := by
    apply joinedControls.of_agrees
    exact agreesOn_of_changesOnly
      (input_disjoint_controls parameters)
      (HonestAssignment.encodeInto_changesOnly family
        (CanonicalContexts.Step.input parameters)
        inputValues joinedAssignment)
  have joinedAtInput :
      Columns.Encodes family
        (CanonicalContexts.Step.joined parameters)
        inputAssignment joinedValues' := by
    exact HonestAssignment.encodes_preserved_by_encodeInto
      family
      (CanonicalContexts.Step.joined parameters)
      (CanonicalContexts.Step.input parameters)
      joinedValues' inputValues joinedAssignment
      (input_disjoint_joined parameters) joinedEncoded

  let applyPlan :=
    CanonicalStepConstructionPlans.apply parameters profile recipes
  rcases applyPlan.extendEncoded inputAssignment inputValues
      (.cons
        (parameters.machine.step Vocabulary.Step.selected
          input.zi input.witness) .nil)
      inputEncoded applyOutputAdmissible with
    ⟨afterStep, afterStepEncoded, applyChanges, _⟩
  have afterStepControls :
      ControlsExact selected afterStep := by
    apply inputControls.of_agrees
    exact instruction_changes_preserve_controls
      SourceOwners.stepApplyPath
      [Ports.committedState parameters]
      inputAssignment afterStep applyChanges
  have joinedAtAfterStep :
      Columns.Encodes family
        (CanonicalContexts.Step.joined parameters)
        afterStep joinedValues' := by
    exact joined_encoded_preserved_by_instruction
      SourceOwners.stepApplyPath inputAssignment afterStep
      runningNext applyChanges joinedAtInput

  let selectorPlan :=
    CanonicalStepConstructionPlans.selector parameters profile recipes
  rcases selectorPlan.extendEncoded afterStep afterStepValues'
      (.cons selected .nil)
      afterStepEncoded selectorOutputAdmissible with
    ⟨common, commonEncoded, selectorChanges, _⟩
  have commonControls :
      ControlsExact selected common := by
    apply afterStepControls.of_agrees
    exact instruction_changes_preserve_controls
      SourceOwners.stepSelectorPath
      [Ports.auxiliaryBit parameters]
      afterStep common selectorChanges
  have joinedAtCommon :
      Columns.Encodes family
        (CanonicalContexts.Step.joined parameters)
        common joinedValues' := by
    exact joined_encoded_preserved_by_instruction
      SourceOwners.stepSelectorPath afterStep common
      runningNext selectorChanges joinedAtAfterStep

  let hashPlan :=
    CanonicalStepConstructionPlans.recursiveHash
      parameters profile recipes
  rcases hashPlan.extendEncoded common commonValues'
      (.cons (priorDigest parameters input) .nil)
      commonEncoded hashOutputAdmissible with
    ⟨afterHash, afterHashEncoded, hashChanges, _⟩
  have afterHashControls :
      ControlsExact selected afterHash := by
    apply commonControls.of_agrees
    exact instruction_changes_preserve_controls
      SourceOwners.stepRecursiveHashPriorPath
      [Ports.auxiliaryDigest parameters]
      common afterHash hashChanges
  have joinedAtHash :
      Columns.Encodes family
        (CanonicalContexts.Step.joined parameters)
        afterHash joinedValues' := by
    exact joined_encoded_preserved_by_instruction
      SourceOwners.stepRecursiveHashPriorPath common afterHash
      runningNext hashChanges joinedAtCommon
  have commonAtHash :
      Columns.Encodes family
        (CanonicalContexts.Step.common parameters)
        afterHash commonValues' := by
    exact encodes_preserved_by_instruction
      SourceOwners.stepRecursiveHashPriorPath
      (CanonicalContexts.Step.common parameters)
      commonValues' common afterHash
      (CanonicalStepPlan.common_excludes parameters
        SourceOwners.stepRecursiveHashPriorPath
        (by decide) (by decide))
      hashChanges commonEncoded

  let freshPlan :=
    CanonicalStepConstructionPlans.recursiveFreshPublic
      parameters profile recipes
  rcases freshPlan.extendEncoded afterHash afterHashValues'
      (.cons (parameters.machine.freshPublic input.fresh) .nil)
      afterHashEncoded freshPublicOutputAdmissible with
    ⟨afterFresh, afterFreshEncoded, freshChanges, _⟩
  have afterFreshControls :
      ControlsExact selected afterFresh := by
    apply afterHashControls.of_agrees
    exact instruction_changes_preserve_controls
      SourceOwners.stepRecursiveFreshPublicPath
      [Ports.auxiliaryEncoded parameters]
      afterHash afterFresh freshChanges
  have joinedAtFresh :
      Columns.Encodes family
        (CanonicalContexts.Step.joined parameters)
        afterFresh joinedValues' := by
    exact joined_encoded_preserved_by_instruction
      SourceOwners.stepRecursiveFreshPublicPath afterHash afterFresh
      runningNext freshChanges joinedAtHash
  have commonAtFresh :
      Columns.Encodes family
        (CanonicalContexts.Step.common parameters)
        afterFresh commonValues' := by
    exact encodes_preserved_by_instruction
      SourceOwners.stepRecursiveFreshPublicPath
      (CanonicalContexts.Step.common parameters)
      commonValues' afterHash afterFresh
      (CanonicalStepPlan.common_excludes parameters
        SourceOwners.stepRecursiveFreshPublicPath
        (by decide) (by decide))
      freshChanges commonAtHash

  let encodePlan :=
    CanonicalStepConstructionPlans.recursiveEncode
      parameters profile recipes
  rcases encodePlan.extendEncoded afterFresh afterFreshPublicValues'
      (.cons
        (parameters.machine.encodeInstance
          (priorDigest parameters input)) .nil)
      afterFreshEncoded encodeOutputAdmissible with
    ⟨afterEncode, afterEncodeEncoded, encodeChanges, _⟩
  have afterEncodeControls :
      ControlsExact selected afterEncode := by
    apply afterFreshControls.of_agrees
    exact instruction_changes_preserve_controls
      SourceOwners.stepRecursiveEncodePath
      [Ports.auxiliaryEncoded parameters]
      afterFresh afterEncode encodeChanges
  have joinedAtEncode :
      Columns.Encodes family
        (CanonicalContexts.Step.joined parameters)
        afterEncode joinedValues' := by
    exact joined_encoded_preserved_by_instruction
      SourceOwners.stepRecursiveEncodePath afterFresh afterEncode
      runningNext encodeChanges joinedAtFresh
  have commonAtEncode :
      Columns.Encodes family
        (CanonicalContexts.Step.common parameters)
        afterEncode commonValues' := by
    exact encodes_preserved_by_instruction
      SourceOwners.stepRecursiveEncodePath
      (CanonicalContexts.Step.common parameters)
      commonValues' afterFresh afterEncode
      (CanonicalStepPlan.common_excludes parameters
        SourceOwners.stepRecursiveEncodePath
        (by decide) (by decide))
      encodeChanges commonAtFresh

  let equalityPlan :=
    CanonicalStepConstructionPlans.recursiveEncodedEquality
      parameters profile recipes
  rcases equalityPlan.extendEncoded afterEncode afterEncodeValues'
      (.cons (priorLinkAccepted parameters input) .nil)
      afterEncodeEncoded equalityOutputAdmissible with
    ⟨afterEquality, afterEqualityEncoded, equalityChanges, _⟩
  have afterEqualityControls :
      ControlsExact selected afterEquality := by
    apply afterEncodeControls.of_agrees
    exact instruction_changes_preserve_controls
      SourceOwners.stepRecursiveEncodedEqualPath
      [Ports.auxiliaryBit parameters]
      afterEncode afterEquality equalityChanges
  have joinedAtEquality :
      Columns.Encodes family
        (CanonicalContexts.Step.joined parameters)
        afterEquality joinedValues' := by
    exact joined_encoded_preserved_by_instruction
      SourceOwners.stepRecursiveEncodedEqualPath afterEncode afterEquality
      runningNext equalityChanges joinedAtEncode
  have commonAtEquality :
      Columns.Encodes family
        (CanonicalContexts.Step.common parameters)
        afterEquality commonValues' := by
    exact encodes_preserved_by_instruction
      SourceOwners.stepRecursiveEncodedEqualPath
      (CanonicalContexts.Step.common parameters)
      commonValues' afterEncode afterEquality
      (CanonicalStepPlan.common_excludes parameters
        SourceOwners.stepRecursiveEncodedEqualPath
        (by decide) (by decide))
      equalityChanges commonAtEncode

  let nifsPlan :=
    CanonicalStepConstructionPlans.recursiveNifs
      parameters profile recipes
  rcases nifsPlan.extendEncoded afterEquality afterEqualityValues'
      (.cons runningNext .nil)
      afterEqualityEncoded nifsOutputAdmissible with
    ⟨afterNifs, afterNifsEncoded, nifsChanges, _⟩
  have afterNifsControls :
      ControlsExact selected afterNifs := by
    apply afterEqualityControls.of_agrees
    exact instruction_changes_preserve_controls
      SourceOwners.stepRecursiveNifsPath
      [Ports.committedRunning parameters]
      afterEquality afterNifs nifsChanges
  have joinedAtNifs :
      Columns.Encodes family
        (CanonicalContexts.Step.joined parameters)
        afterNifs joinedValues' := by
    exact joined_encoded_preserved_by_instruction
      SourceOwners.stepRecursiveNifsPath afterEquality afterNifs
      runningNext nifsChanges joinedAtEquality
  have commonAtNifs :
      Columns.Encodes family
        (CanonicalContexts.Step.common parameters)
        afterNifs commonValues' := by
    exact encodes_preserved_by_instruction
      SourceOwners.stepRecursiveNifsPath
      (CanonicalContexts.Step.common parameters)
      commonValues' afterEquality afterNifs
      (CanonicalStepPlan.common_excludes parameters
        SourceOwners.stepRecursiveNifsPath
        (by decide) (by decide))
      nifsChanges commonAtEquality

  let baseEqualityPlan :=
    CanonicalStepConstructionPlans.baseEquality
      parameters profile recipes
  rcases baseEqualityPlan.extendEncoded afterNifs commonValues'
      (.cons (stateEqual parameters input.z0 input.zi) .nil)
      commonAtNifs baseEqualityOutputAdmissible with
    ⟨afterBaseEquality, afterBaseEqualityEncoded,
      baseEqualityChanges, _⟩
  have afterBaseEqualityControls :
      ControlsExact selected afterBaseEquality := by
    apply afterNifsControls.of_agrees
    exact instruction_changes_preserve_controls
      SourceOwners.stepBaseStateEqualPath
      [Ports.auxiliaryBit parameters]
      afterNifs afterBaseEquality baseEqualityChanges
  have joinedAtBaseEquality :
      Columns.Encodes family
        (CanonicalContexts.Step.joined parameters)
        afterBaseEquality joinedValues' := by
    exact joined_encoded_preserved_by_instruction
      SourceOwners.stepBaseStateEqualPath afterNifs afterBaseEquality
      runningNext baseEqualityChanges joinedAtNifs
  have commonAtBaseEquality :
      Columns.Encodes family
        (CanonicalContexts.Step.common parameters)
        afterBaseEquality commonValues' := by
    exact encodes_preserved_by_instruction
      SourceOwners.stepBaseStateEqualPath
      (CanonicalContexts.Step.common parameters)
      commonValues' afterNifs afterBaseEquality
      (CanonicalStepPlan.common_excludes parameters
        SourceOwners.stepBaseStateEqualPath
        (by decide) (by decide))
      baseEqualityChanges commonAtNifs
  have recursiveAtBaseEquality :
      Columns.Encodes family
        (CanonicalContexts.Step.afterNifs parameters)
        afterBaseEquality afterNifsValues' := by
    exact encodes_preserved_by_instruction
      SourceOwners.stepBaseStateEqualPath
      (CanonicalContexts.Step.afterNifs parameters)
      afterNifsValues' afterNifs afterBaseEquality
      (afterNifs_excludes parameters
        SourceOwners.stepBaseStateEqualPath
        (by decide) (by decide) (by decide) (by decide)
        (by decide) (by decide) (by decide))
      baseEqualityChanges afterNifsEncoded

  let baseLiteralPlan :=
    CanonicalStepConstructionPlans.baseLiteral
      parameters profile defaultAdmissible
  rcases baseLiteralPlan.extendEncoded
      afterBaseEquality baseEqualityValues'
      afterBaseEqualityEncoded
      (CanonicalStepPlan.one_excludes_instruction
        SourceOwners.stepBaseDefaultPath)
      (CanonicalStepPlan.activation_excludes_instruction
        SourceOwners.stepBranchPath
        SourceOwners.stepBaseDefaultPath true)
      (CanonicalStepPlan.afterBaseEquality_excludes parameters
        SourceOwners.stepBaseDefaultPath
        (by decide) (by decide) (by decide)) with
    ⟨afterBaseLiteral, afterBaseLiteralEncoded,
      baseLiteralChanges, _⟩
  have afterBaseLiteralControls :
      ControlsExact selected afterBaseLiteral := by
    apply afterBaseEqualityControls.of_agrees
    exact instruction_changes_preserve_controls
      SourceOwners.stepBaseDefaultPath
      [Ports.committedRunning parameters]
      afterBaseEquality afterBaseLiteral baseLiteralChanges
  have joinedAtBaseLiteral :
      Columns.Encodes family
        (CanonicalContexts.Step.joined parameters)
        afterBaseLiteral joinedValues' := by
    exact joined_encoded_preserved_by_instruction
      SourceOwners.stepBaseDefaultPath afterBaseEquality afterBaseLiteral
      runningNext baseLiteralChanges joinedAtBaseEquality
  have commonAtBaseLiteral :
      Columns.Encodes family
        (CanonicalContexts.Step.common parameters)
        afterBaseLiteral commonValues' := by
    exact encodes_preserved_by_instruction
      SourceOwners.stepBaseDefaultPath
      (CanonicalContexts.Step.common parameters)
      commonValues' afterBaseEquality afterBaseLiteral
      (CanonicalStepPlan.common_excludes parameters
        SourceOwners.stepBaseDefaultPath
        (by decide) (by decide))
      baseLiteralChanges commonAtBaseEquality
  have recursiveAtBaseLiteral :
      Columns.Encodes family
        (CanonicalContexts.Step.afterNifs parameters)
        afterBaseLiteral afterNifsValues' := by
    exact encodes_preserved_by_instruction
      SourceOwners.stepBaseDefaultPath
      (CanonicalContexts.Step.afterNifs parameters)
      afterNifsValues' afterBaseEquality afterBaseLiteral
      (afterNifs_excludes parameters
        SourceOwners.stepBaseDefaultPath
        (by decide) (by decide) (by decide) (by decide)
        (by decide) (by decide) (by decide))
      baseLiteralChanges recursiveAtBaseEquality

  have continuationEncoded :
      Columns.Encodes family
        (CanonicalContexts.Step.continuationInput parameters)
        afterBaseLiteral continuationValues' := by
    exact Columns.append_encodes family afterBaseLiteral
      (CanonicalContexts.Step.joined parameters)
      (CanonicalContexts.Step.common parameters)
      joinedValues' commonValues'
      joinedAtBaseLiteral commonAtBaseLiteral

  let continuationPlan :=
    CanonicalStepConstructionPlans.continuationHash
      parameters profile recipes
  rcases continuationPlan.extendEncoded
      afterBaseLiteral continuationValues'
      (.cons (nextDigest parameters input runningNext) .nil)
      continuationEncoded continuationOutputAdmissible with
    ⟨finalAssignment, finalEncoded, continuationChanges, _⟩
  have finalControls :
      ControlsExact selected finalAssignment := by
    apply afterBaseLiteralControls.of_agrees
    exact instruction_changes_preserve_controls
      SourceOwners.stepContinuationHashPath
      [Ports.publicDigest parameters]
      afterBaseLiteral finalAssignment continuationChanges
  have baseAtFinal :
      Columns.Encodes family
        (CanonicalContexts.Step.afterBaseLiteral parameters)
        finalAssignment baseLiteralValues' := by
    exact encodes_preserved_by_instruction
      SourceOwners.stepContinuationHashPath
      (CanonicalContexts.Step.afterBaseLiteral parameters)
      baseLiteralValues' afterBaseLiteral finalAssignment
      (afterBaseLiteral_excludes parameters
        SourceOwners.stepContinuationHashPath
        (by decide) (by decide) (by decide) (by decide))
      continuationChanges afterBaseLiteralEncoded
  have recursiveAtFinal :
      Columns.Encodes family
        (CanonicalContexts.Step.afterNifs parameters)
        finalAssignment afterNifsValues' := by
    exact encodes_preserved_by_instruction
      SourceOwners.stepContinuationHashPath
      (CanonicalContexts.Step.afterNifs parameters)
      afterNifsValues' afterBaseLiteral finalAssignment
      (afterNifs_excludes parameters
        SourceOwners.stepContinuationHashPath
        (by decide) (by decide) (by decide) (by decide)
        (by decide) (by decide) (by decide))
      continuationChanges recursiveAtBaseLiteral
  exact ⟨⟨finalAssignment, finalControls,
    baseAtFinal, recursiveAtFinal, finalEncoded⟩⟩

end CanonicalStepCompleteness

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding
