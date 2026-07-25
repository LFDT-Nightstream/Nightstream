import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.CanonicalTerminalAlwaysSeparation
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.CanonicalTerminalSoundness
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.HonestAssignment
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ThreeGroupCompletion
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.VisibleCompletion
import Nightstream.Implementation.Lowering.Goldilocks.ReceiptSatisfaction

/-!
Contract: artifact-independent honest assignment construction for the
canonical fixed-one Terminal encoding.

Owns:
- canonical visible coordinates for one accepted typed Terminal execution;
- exact completion of every selected and inactive primitive receipt;
- construction of a satisfying physical assignment from semantic acceptance.

Does not own: production codecs or recipes, Rust behavior, numeric R1CS
matrices, generated artifacts, or extraction.

Emits constraints: no new constraints; the witness satisfies exactly the
selected canonical receipt program.
-/

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding

open Nightstream.Implementation.Lowering.Typed
open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Vocabulary
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Terminal

namespace CanonicalTerminalCompleteness

abbrev TerminalStatementFor (parameters : Parameters) :=
  Nightstream.HyperNova.Construction2.Paper.TerminalStatement
    parameters.State

abbrev TerminalProofFor (parameters : Parameters) :=
  FixedOneTerminal.Proof parameters

/-- Exact codec-domain premise for every semantic value used by the longest
Terminal SSA context.  Boolean admissibility is automatic; the substantive
parts are the selected bounded-natural and production data codecs. -/
def AdmissibleExecution
    (parameters : Parameters)
    (profile : Profile parameters)
    (statement : TerminalStatementFor parameters)
    (proof : TerminalProofFor parameters) : Prop :=
  HonestAssignment.Admissible (profile.family parameters)
    (afterFreshCheckValues parameters
      (decide (statement.iteration = 0)) statement proof)

/-- Initial values for the verifier one coordinate and both Terminal branch
activations.  All typed and temporary coordinates start at zero. -/
def controlAssignment (selected : Bool) : ColumnId -> Field :=
  fun id =>
    if id = oneColumn then
      1
    else if id =
        activationColumn SourceOwners.terminalBranchPath true then
      if selected then 1 else 0
    else if id =
        activationColumn SourceOwners.terminalBranchPath false then
      if selected then 0 else 1
    else
      0

def ControlsExact
    (selected : Bool)
    (assignment : ColumnId -> Field) : Prop :=
  assignment oneColumn = 1 ∧
    assignment
        (activationColumn SourceOwners.terminalBranchPath true) =
      (if selected then 1 else 0) ∧
    assignment
        (activationColumn SourceOwners.terminalBranchPath false) =
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
          activationColumn SourceOwners.terminalBranchPath true,
          activationColumn SourceOwners.terminalBranchPath false]
        before after) :
    ControlsExact selected after := by
  exact ⟨
    (agrees oneColumn (by simp)).trans exact.1,
    (agrees
      (activationColumn SourceOwners.terminalBranchPath true)
      (by simp)).trans exact.2.1,
    (agrees
      (activationColumn SourceOwners.terminalBranchPath false)
      (by simp)).trans exact.2.2⟩

private theorem input_ids_nodup
    (parameters : Parameters) :
    (CanonicalContexts.Terminal.input parameters
      ).toSchemaBundles.ids.Nodup := by
  simpa [CanonicalContexts.Terminal.input, SchemaBundles.ids,
    ColumnPlan.schemaColumnIds, Columns.toSchemaBundles_columns] using
    (ColumnPlan.allocateSchemaFrom_ids_nodup
      (fun slot => PhysicalOwner.typed (.input slot))
      0 (terminalInputSchema parameters))

private theorem input_disjoint_controls
    (parameters : Parameters) :
    IdsDisjoint
      (CanonicalContexts.Terminal.input parameters
        ).toSchemaBundles.ids
      [oneColumn,
        activationColumn SourceOwners.terminalBranchPath true,
        activationColumn SourceOwners.terminalBranchPath false] := by
  intro id inputMember controlMember
  simp only [List.mem_cons, List.not_mem_nil, or_false] at controlMember
  rcases controlMember with oneMember | trueMember | falseMember
  · subst id
    exact CanonicalPrimitivePlan.inputDisjointOne
      (terminalInputSchema parameters) oneColumn inputMember (by simp)
  · have disjoint :=
      CanonicalPrimitivePlan.inputDisjointActivations
        (terminalInputSchema parameters) SourceOwners.terminalBranchPath
    exact disjoint id inputMember (by
      simp [activationColumns, trueMember])
  · have disjoint :=
      CanonicalPrimitivePlan.inputDisjointActivations
        (terminalInputSchema parameters) SourceOwners.terminalBranchPath
    exact disjoint id inputMember (by
      simp [activationColumns, falseMember])

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
        activationColumn SourceOwners.terminalBranchPath true,
        activationColumn SourceOwners.terminalBranchPath false]
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
          path output SourceOwners.terminalBranchPath
      exact disjoint id outputMember (by
        simp [activationColumns, trueMember])
    · have disjoint :=
        CanonicalPrimitivePlan.instructionOutputsDisjointActivations
          path output SourceOwners.terminalBranchPath
      exact disjoint id outputMember (by
        simp [activationColumns, falseMember])
  · exact changes

/-- Visible typed contexts needed by all Terminal primitive occurrences.
Temporary witnesses are deliberately absent. -/
structure VisibleWitness
    (parameters : Parameters)
    (profile : Profile parameters)
    (statement : TerminalStatementFor parameters)
    (proof : TerminalProofFor parameters) where
  assignment : ColumnId -> Field
  controls :
    ControlsExact (decide (statement.iteration = 0)) assignment
  baseEncoded :
    Columns.Encodes (profile.family parameters)
      (CanonicalContexts.Terminal.afterBaseEquality parameters)
      assignment
      (afterBaseEqualityValues parameters
        (decide (statement.iteration = 0)) statement proof)
  recursiveEncoded :
    Columns.Encodes (profile.family parameters)
      (CanonicalContexts.Terminal.afterFreshCheck parameters)
      assignment
      (afterFreshCheckValues parameters
        (decide (statement.iteration = 0)) statement proof)

private theorem VisibleWitness.afterRunningEncoded
    {parameters : Parameters}
    {profile : Profile parameters}
    {statement : TerminalStatementFor parameters}
    {proof : TerminalProofFor parameters}
    (visible : VisibleWitness parameters profile statement proof) :
    Columns.Encodes (profile.family parameters)
      (CanonicalContexts.Terminal.afterRunningCheck parameters)
      visible.assignment
      (afterRunningCheckValues parameters
        (decide (statement.iteration = 0)) statement proof) := by
  exact Columns.right_encodes_of_append
    (profile.family parameters) visible.assignment
    (instructionColumns SourceOwners.terminalRecursiveFreshCheckPath
      [Ports.auxiliaryBit parameters])
    (CanonicalContexts.Terminal.afterRunningCheck parameters)
    (.cons (freshAcceptedValue parameters proof) .nil)
    (afterRunningCheckValues parameters
      (decide (statement.iteration = 0)) statement proof)
    visible.recursiveEncoded

private theorem VisibleWitness.afterEqualityEncoded
    {parameters : Parameters}
    {profile : Profile parameters}
    {statement : TerminalStatementFor parameters}
    {proof : TerminalProofFor parameters}
    (visible : VisibleWitness parameters profile statement proof) :
    Columns.Encodes (profile.family parameters)
      (CanonicalContexts.Terminal.afterEncodedEquality parameters)
      visible.assignment
      (afterEncodedEqualityValues parameters
        (decide (statement.iteration = 0)) statement proof) := by
  exact Columns.right_encodes_of_append
    (profile.family parameters) visible.assignment
    (instructionColumns SourceOwners.terminalRecursiveRunningCheckPath
      [Ports.auxiliaryBit parameters])
    (CanonicalContexts.Terminal.afterEncodedEquality parameters)
    (.cons (runningAcceptedValue parameters proof) .nil)
    (afterEncodedEqualityValues parameters
      (decide (statement.iteration = 0)) statement proof)
    visible.afterRunningEncoded

private theorem VisibleWitness.afterEncodeEncoded
    {parameters : Parameters}
    {profile : Profile parameters}
    {statement : TerminalStatementFor parameters}
    {proof : TerminalProofFor parameters}
    (visible : VisibleWitness parameters profile statement proof) :
    Columns.Encodes (profile.family parameters)
      (CanonicalContexts.Terminal.afterEncode parameters)
      visible.assignment
      (afterEncodeValues parameters
        (decide (statement.iteration = 0)) statement proof) := by
  exact Columns.right_encodes_of_append
    (profile.family parameters) visible.assignment
    (instructionColumns SourceOwners.terminalRecursiveEncodedEqualPath
      [Ports.auxiliaryBit parameters])
    (CanonicalContexts.Terminal.afterEncode parameters)
    (.cons (priorLinkAccepted parameters statement proof) .nil)
    (afterEncodeValues parameters
      (decide (statement.iteration = 0)) statement proof)
    visible.afterEqualityEncoded

private theorem VisibleWitness.afterFreshPublicEncoded
    {parameters : Parameters}
    {profile : Profile parameters}
    {statement : TerminalStatementFor parameters}
    {proof : TerminalProofFor parameters}
    (visible : VisibleWitness parameters profile statement proof) :
    Columns.Encodes (profile.family parameters)
      (CanonicalContexts.Terminal.afterFreshPublic parameters)
      visible.assignment
      (afterFreshPublicValues parameters
        (decide (statement.iteration = 0)) statement proof) := by
  exact Columns.right_encodes_of_append
    (profile.family parameters) visible.assignment
    (instructionColumns SourceOwners.terminalRecursiveEncodePath
      [Ports.auxiliaryEncoded parameters])
    (CanonicalContexts.Terminal.afterFreshPublic parameters)
    (.cons
      (parameters.machine.encodeInstance
        (priorDigest parameters statement proof)) .nil)
    (afterFreshPublicValues parameters
      (decide (statement.iteration = 0)) statement proof)
    visible.afterEncodeEncoded

private theorem VisibleWitness.afterHashEncoded
    {parameters : Parameters}
    {profile : Profile parameters}
    {statement : TerminalStatementFor parameters}
    {proof : TerminalProofFor parameters}
    (visible : VisibleWitness parameters profile statement proof) :
    Columns.Encodes (profile.family parameters)
      (CanonicalContexts.Terminal.afterHash parameters)
      visible.assignment
      (afterHashValues parameters
        (decide (statement.iteration = 0)) statement proof) := by
  exact Columns.right_encodes_of_append
    (profile.family parameters) visible.assignment
    (instructionColumns SourceOwners.terminalRecursiveFreshPublicPath
      [Ports.auxiliaryEncoded parameters])
    (CanonicalContexts.Terminal.afterHash parameters)
    (.cons (parameters.machine.freshPublic proof.fresh) .nil)
    (afterHashValues parameters
      (decide (statement.iteration = 0)) statement proof)
    visible.afterFreshPublicEncoded

private theorem VisibleWitness.branchEncoded
    {parameters : Parameters}
    {profile : Profile parameters}
    {statement : TerminalStatementFor parameters}
    {proof : TerminalProofFor parameters}
    (visible : VisibleWitness parameters profile statement proof) :
    Columns.Encodes (profile.family parameters)
      (CanonicalContexts.Terminal.branchInput parameters)
      visible.assignment
      (branchValues parameters
        (decide (statement.iteration = 0)) statement proof) := by
  exact Columns.right_encodes_of_append
    (profile.family parameters) visible.assignment
    (instructionColumns SourceOwners.terminalRecursiveHashPriorPath
      [Ports.auxiliaryDigest parameters])
    (CanonicalContexts.Terminal.branchInput parameters)
    (.cons (priorDigest parameters statement proof) .nil)
    (branchValues parameters
      (decide (statement.iteration = 0)) statement proof)
    visible.afterHashEncoded

private theorem VisibleWitness.inputEncoded
    {parameters : Parameters}
    {profile : Profile parameters}
    {statement : TerminalStatementFor parameters}
    {proof : TerminalProofFor parameters}
    (visible : VisibleWitness parameters profile statement proof) :
    Columns.Encodes (profile.family parameters)
      (CanonicalContexts.Terminal.input parameters)
      visible.assignment
      (terminalInputValues parameters statement proof) := by
  exact Columns.right_encodes_of_append
    (profile.family parameters) visible.assignment
    (instructionColumns SourceOwners.terminalSelectorPath
      [Ports.auxiliaryBit parameters])
    (CanonicalContexts.Terminal.input parameters)
    (.cons (decide (statement.iteration = 0)) .nil)
    (terminalInputValues parameters statement proof)
    visible.branchEncoded

/-- Canonical codecs construct all visible Terminal SSA coordinates while
preserving the verifier one and both branch activations. -/
theorem exists_visible
    (parameters : Parameters)
    (profile : Profile parameters)
    (recipes :
      CallRecipes (signature parameters) (profile.family parameters))
    (statement : TerminalStatementFor parameters)
    (proof : TerminalProofFor parameters)
    (admissible :
      AdmissibleExecution parameters profile statement proof) :
    Nonempty (VisibleWitness parameters profile statement proof) := by
  let selected := decide (statement.iteration = 0)
  let family := profile.family parameters
  let inputValues := terminalInputValues parameters statement proof
  let branchValues' := branchValues parameters selected statement proof
  let afterHashValues' :=
    afterHashValues parameters selected statement proof
  let afterFreshPublicValues' :=
    afterFreshPublicValues parameters selected statement proof
  let afterEncodeValues' :=
    afterEncodeValues parameters selected statement proof
  let afterEqualityValues' :=
    afterEncodedEqualityValues parameters selected statement proof
  let afterRunningValues' :=
    afterRunningCheckValues parameters selected statement proof
  let afterFreshValues' :=
    afterFreshCheckValues parameters selected statement proof

  change HonestAssignment.Admissible family afterFreshValues' at admissible
  have freshOutputAdmissible :
      HonestAssignment.Admissible
        (schema := [Ports.auxiliaryBit parameters]) family
        (.cons (freshAcceptedValue parameters proof) .nil) :=
    ⟨admissible.1, True.intro⟩
  have afterRunningAdmissible :
      HonestAssignment.Admissible family afterRunningValues' :=
    admissible.2
  have runningOutputAdmissible :
      HonestAssignment.Admissible
        (schema := [Ports.auxiliaryBit parameters]) family
        (.cons (runningAcceptedValue parameters proof) .nil) :=
    ⟨afterRunningAdmissible.1, True.intro⟩
  have afterEqualityAdmissible :
      HonestAssignment.Admissible family afterEqualityValues' :=
    afterRunningAdmissible.2
  have equalityOutputAdmissible :
      HonestAssignment.Admissible
        (schema := [Ports.auxiliaryBit parameters]) family
        (.cons (priorLinkAccepted parameters statement proof) .nil) :=
    ⟨afterEqualityAdmissible.1, True.intro⟩
  have afterEncodeAdmissible :
      HonestAssignment.Admissible family afterEncodeValues' :=
    afterEqualityAdmissible.2
  have encodeOutputAdmissible :
      HonestAssignment.Admissible
        (schema := [Ports.auxiliaryEncoded parameters]) family
        (.cons
          (parameters.machine.encodeInstance
            (priorDigest parameters statement proof)) .nil) :=
    ⟨afterEncodeAdmissible.1, True.intro⟩
  have afterFreshPublicAdmissible :
      HonestAssignment.Admissible family afterFreshPublicValues' :=
    afterEncodeAdmissible.2
  have freshPublicOutputAdmissible :
      HonestAssignment.Admissible
        (schema := [Ports.auxiliaryEncoded parameters]) family
        (.cons (parameters.machine.freshPublic proof.fresh) .nil) :=
    ⟨afterFreshPublicAdmissible.1, True.intro⟩
  have afterHashAdmissible :
      HonestAssignment.Admissible family afterHashValues' :=
    afterFreshPublicAdmissible.2
  have hashOutputAdmissible :
      HonestAssignment.Admissible
        (schema := [Ports.auxiliaryDigest parameters]) family
        (.cons (priorDigest parameters statement proof) .nil) :=
    ⟨afterHashAdmissible.1, True.intro⟩
  have branchAdmissible :
      HonestAssignment.Admissible family branchValues' :=
    afterHashAdmissible.2
  have selectorOutputAdmissible :
      HonestAssignment.Admissible
        (schema := [Ports.auxiliaryBit parameters]) family
        (.cons selected .nil) :=
    ⟨branchAdmissible.1, True.intro⟩
  have inputAdmissible :
      HonestAssignment.Admissible family inputValues :=
    branchAdmissible.2
  have baseOutputAdmissible :
      HonestAssignment.Admissible
        (schema := [Ports.auxiliaryBit parameters]) family
        (.cons (stateEqual parameters statement.zi statement.z0) .nil) := by
    change True ∧ True
    exact ⟨True.intro, True.intro⟩

  let initial := controlAssignment selected
  let inputAssignment :=
    HonestAssignment.encodeInto family
      (CanonicalContexts.Terminal.input parameters)
      inputValues initial
  have inputEncoded :
      Columns.Encodes family
        (CanonicalContexts.Terminal.input parameters)
        inputAssignment inputValues := by
    exact HonestAssignment.encodeInto_encodes family
      (CanonicalContexts.Terminal.input parameters)
      inputValues initial
      (CanonicalContexts.Terminal.inputWidths parameters profile)
      inputAdmissible (input_ids_nodup parameters)
  have inputControls :
      ControlsExact selected inputAssignment := by
    apply (controlAssignment_exact selected).of_agrees
    exact agreesOn_of_changesOnly
      (input_disjoint_controls parameters)
      (HonestAssignment.encodeInto_changesOnly family
        (CanonicalContexts.Terminal.input parameters)
        inputValues initial)

  let selectorPlan :=
    CanonicalTerminalPlan.selectorInvokePlan parameters profile recipes
  rcases selectorPlan.extendEncoded inputAssignment inputValues
      (.cons selected .nil) inputEncoded selectorOutputAdmissible with
    ⟨afterSelector, branchEncoded, selectorChanges, _⟩
  have afterSelectorControls :
      ControlsExact selected afterSelector := by
    apply inputControls.of_agrees
    exact instruction_changes_preserve_controls
      SourceOwners.terminalSelectorPath
      [Ports.auxiliaryBit parameters]
      inputAssignment afterSelector selectorChanges

  let hashPlan :=
    CanonicalTerminalPlan.recursiveHashInvokePlan parameters profile recipes
  rcases hashPlan.extendEncoded afterSelector branchValues'
      (.cons (priorDigest parameters statement proof) .nil)
      branchEncoded hashOutputAdmissible with
    ⟨afterHash, afterHashEncoded, hashChanges, _⟩
  have afterHashControls :
      ControlsExact selected afterHash := by
    apply afterSelectorControls.of_agrees
    exact instruction_changes_preserve_controls
      SourceOwners.terminalRecursiveHashPriorPath
      [Ports.auxiliaryDigest parameters]
      afterSelector afterHash hashChanges

  let freshPublicPlan :=
    CanonicalTerminalPlan.recursiveFreshPublicInvokePlan
      parameters profile recipes
  rcases freshPublicPlan.extendEncoded afterHash afterHashValues'
      (.cons (parameters.machine.freshPublic proof.fresh) .nil)
      afterHashEncoded freshPublicOutputAdmissible with
    ⟨afterFreshPublic, afterFreshPublicEncoded,
      freshPublicChanges, _⟩
  have afterFreshPublicControls :
      ControlsExact selected afterFreshPublic := by
    apply afterHashControls.of_agrees
    exact instruction_changes_preserve_controls
      SourceOwners.terminalRecursiveFreshPublicPath
      [Ports.auxiliaryEncoded parameters]
      afterHash afterFreshPublic freshPublicChanges

  let encodePlan :=
    CanonicalTerminalPlan.recursiveEncodeInvokePlan parameters profile recipes
  rcases encodePlan.extendEncoded
      afterFreshPublic afterFreshPublicValues'
      (.cons
        (parameters.machine.encodeInstance
          (priorDigest parameters statement proof)) .nil)
      afterFreshPublicEncoded encodeOutputAdmissible with
    ⟨afterEncode, afterEncodeEncoded, encodeChanges, _⟩
  have afterEncodeControls :
      ControlsExact selected afterEncode := by
    apply afterFreshPublicControls.of_agrees
    exact instruction_changes_preserve_controls
      SourceOwners.terminalRecursiveEncodePath
      [Ports.auxiliaryEncoded parameters]
      afterFreshPublic afterEncode encodeChanges

  let equalityPlan :=
    CanonicalTerminalPlan.recursiveEncodedEqualityInvokePlan
      parameters profile recipes
  rcases equalityPlan.extendEncoded afterEncode afterEncodeValues'
      (.cons (priorLinkAccepted parameters statement proof) .nil)
      afterEncodeEncoded equalityOutputAdmissible with
    ⟨afterEquality, afterEqualityEncoded, equalityChanges, _⟩
  have afterEqualityControls :
      ControlsExact selected afterEquality := by
    apply afterEncodeControls.of_agrees
    exact instruction_changes_preserve_controls
      SourceOwners.terminalRecursiveEncodedEqualPath
      [Ports.auxiliaryBit parameters]
      afterEncode afterEquality equalityChanges

  let runningPlan :=
    CanonicalTerminalPlan.recursiveRunningCheckInvokePlan
      parameters profile recipes
  rcases runningPlan.extendEncoded afterEquality afterEqualityValues'
      (.cons (runningAcceptedValue parameters proof) .nil)
      afterEqualityEncoded runningOutputAdmissible with
    ⟨afterRunning, afterRunningEncoded, runningChanges, _⟩
  have afterRunningControls :
      ControlsExact selected afterRunning := by
    apply afterEqualityControls.of_agrees
    exact instruction_changes_preserve_controls
      SourceOwners.terminalRecursiveRunningCheckPath
      [Ports.auxiliaryBit parameters]
      afterEquality afterRunning runningChanges

  let freshPlan :=
    CanonicalTerminalPlan.recursiveFreshCheckInvokePlan
      parameters profile recipes
  rcases freshPlan.extendEncoded afterRunning afterRunningValues'
      (.cons (freshAcceptedValue parameters proof) .nil)
      afterRunningEncoded freshOutputAdmissible with
    ⟨afterFresh, afterFreshEncoded, freshChanges, _⟩
  have afterFreshControls :
      ControlsExact selected afterFresh := by
    apply afterRunningControls.of_agrees
    exact instruction_changes_preserve_controls
      SourceOwners.terminalRecursiveFreshCheckPath
      [Ports.auxiliaryBit parameters]
      afterRunning afterFresh freshChanges

  have afterRunningAtFresh :
      Columns.Encodes family
        (CanonicalContexts.Terminal.afterRunningCheck parameters)
        afterFresh afterRunningValues' := by
    exact Columns.right_encodes_of_append family afterFresh
      (instructionColumns SourceOwners.terminalRecursiveFreshCheckPath
        [Ports.auxiliaryBit parameters])
      (CanonicalContexts.Terminal.afterRunningCheck parameters)
      (.cons (freshAcceptedValue parameters proof) .nil)
      afterRunningValues' afterFreshEncoded
  have afterEqualityAtFresh :
      Columns.Encodes family
        (CanonicalContexts.Terminal.afterEncodedEquality parameters)
        afterFresh afterEqualityValues' := by
    exact Columns.right_encodes_of_append family afterFresh
      (instructionColumns SourceOwners.terminalRecursiveRunningCheckPath
        [Ports.auxiliaryBit parameters])
      (CanonicalContexts.Terminal.afterEncodedEquality parameters)
      (.cons (runningAcceptedValue parameters proof) .nil)
      afterEqualityValues' afterRunningAtFresh
  have afterEncodeAtFresh :
      Columns.Encodes family
        (CanonicalContexts.Terminal.afterEncode parameters)
        afterFresh afterEncodeValues' := by
    exact Columns.right_encodes_of_append family afterFresh
      (instructionColumns SourceOwners.terminalRecursiveEncodedEqualPath
        [Ports.auxiliaryBit parameters])
      (CanonicalContexts.Terminal.afterEncode parameters)
      (.cons (priorLinkAccepted parameters statement proof) .nil)
      afterEncodeValues' afterEqualityAtFresh
  have afterFreshPublicAtFresh :
      Columns.Encodes family
        (CanonicalContexts.Terminal.afterFreshPublic parameters)
        afterFresh afterFreshPublicValues' := by
    exact Columns.right_encodes_of_append family afterFresh
      (instructionColumns SourceOwners.terminalRecursiveEncodePath
        [Ports.auxiliaryEncoded parameters])
      (CanonicalContexts.Terminal.afterFreshPublic parameters)
      (.cons
        (parameters.machine.encodeInstance
          (priorDigest parameters statement proof)) .nil)
      afterFreshPublicValues' afterEncodeAtFresh
  have afterHashAtFresh :
      Columns.Encodes family
        (CanonicalContexts.Terminal.afterHash parameters)
        afterFresh afterHashValues' := by
    exact Columns.right_encodes_of_append family afterFresh
      (instructionColumns SourceOwners.terminalRecursiveFreshPublicPath
        [Ports.auxiliaryEncoded parameters])
      (CanonicalContexts.Terminal.afterHash parameters)
      (.cons (parameters.machine.freshPublic proof.fresh) .nil)
      afterHashValues' afterFreshPublicAtFresh
  have branchAtFresh :
      Columns.Encodes family
        (CanonicalContexts.Terminal.branchInput parameters)
        afterFresh branchValues' := by
    exact Columns.right_encodes_of_append family afterFresh
      (instructionColumns SourceOwners.terminalRecursiveHashPriorPath
        [Ports.auxiliaryDigest parameters])
      (CanonicalContexts.Terminal.branchInput parameters)
      (.cons (priorDigest parameters statement proof) .nil)
      branchValues' afterHashAtFresh

  let basePlan :=
    CanonicalTerminalPlan.baseEqualityInvokePlan parameters profile recipes
  rcases basePlan.extendEncoded afterFresh branchValues'
      (.cons (stateEqual parameters statement.zi statement.z0) .nil)
      branchAtFresh baseOutputAdmissible with
    ⟨visible, baseEncoded, baseChanges, _⟩
  have visibleControls :
      ControlsExact selected visible := by
    apply afterFreshControls.of_agrees
    exact instruction_changes_preserve_controls
      SourceOwners.terminalBaseStateEqualPath
      [Ports.auxiliaryBit parameters]
      afterFresh visible baseChanges
  have recursivePreserved :
      Columns.Encodes family
        (CanonicalContexts.Terminal.afterFreshCheck parameters)
        visible afterFreshValues' := by
    apply
      (CanonicalContexts.Terminal.afterFreshCheck parameters
        ).toSchemaBundles.encodes_of_agrees
          family afterFresh visible afterFreshValues'
    · exact agreesOn_of_changesOnly
        (CanonicalPrimitivePlan.ContextExcludesOwner.instructionOutputsDisjoint
          SourceOwners.terminalBaseStateEqualPath
          (CanonicalContexts.Terminal.afterFreshCheck parameters)
          (CanonicalCompletionPlans.Terminal.afterFreshCheck_excludes_baseEquality
            parameters))
        baseChanges
    · exact afterFreshEncoded
  exact ⟨⟨visible, visibleControls, baseEncoded, recursivePreserved⟩⟩

theorem baseAccepted
    (parameters : Parameters)
    (statement : TerminalStatementFor parameters)
    (proof : TerminalProofFor parameters)
    (accepted : Accepts parameters statement proof)
    (iterationZero : statement.iteration = 0) :
    stateEqual parameters statement.zi statement.z0 = true := by
  have executed :
      (program parameters).exec
          (terminalInputValues parameters statement proof) =
        some .nil :=
    ((program parameters).exec_eq_some_iff_holds
      (terminalInputValues parameters statement proof) .nil).2 accepted
  rw [program_exec_eq_reference] at executed
  cases endpoint : stateEqual parameters statement.zi statement.z0 <;>
    simp [referenceExec, branchReferenceExec, baseReferenceExec,
      iterationZero, endpoint] at executed ⊢

theorem recursiveAccepted
    (parameters : Parameters)
    (statement : TerminalStatementFor parameters)
    (proof : TerminalProofFor parameters)
    (accepted : Accepts parameters statement proof)
    (iterationNonzero : ¬ statement.iteration = 0) :
    priorLinkAccepted parameters statement proof = true ∧
      runningAcceptedValue parameters proof = true ∧
      freshAcceptedValue parameters proof = true := by
  have executed :
      (program parameters).exec
          (terminalInputValues parameters statement proof) =
        some .nil :=
    ((program parameters).exec_eq_some_iff_holds
      (terminalInputValues parameters statement proof) .nil).2 accepted
  rw [program_exec_eq_reference] at executed
  cases prior : priorLinkAccepted parameters statement proof <;>
    cases running : runningAcceptedValue parameters proof <;>
      cases fresh : freshAcceptedValue parameters proof <;>
        simp [referenceExec, branchReferenceExec, recursiveReferenceExec,
          iterationNonzero, prior, running, fresh] at executed ⊢

theorem alwaysHonest
    (parameters : Parameters)
    (profile : Profile parameters)
    (recipes :
      CallRecipes (signature parameters) (profile.family parameters))
    (statement : TerminalStatementFor parameters)
    (proof : TerminalProofFor parameters)
    (visible : VisibleWitness parameters profile statement proof) :
    (CanonicalCompletionPlans.Terminal.always
      parameters profile recipes).HonestActive visible.assignment := by
  let selector :=
    CanonicalTerminalPlan.selectorPlan parameters profile recipes
  have semantic :
      (iterationZeroCall parameters).Holds
        (terminalInputValues parameters statement proof)
        (branchValues parameters
          (decide (statement.iteration = 0)) statement proof) :=
    ((iterationZeroCall parameters).exec_eq_some_iff_holds
      (terminalInputValues parameters statement proof)
      (branchValues parameters
        (decide (statement.iteration = 0)) statement proof)).1
      (iterationZeroCall_exec parameters statement proof)
  have resultEncoded :
      Columns.Encodes (profile.family parameters)
        selector.resultColumns visible.assignment
        (branchValues parameters
          (decide (statement.iteration = 0)) statement proof) := by
    simpa [selector, CanonicalTerminalPlan.selectorPlan,
      PrimitivePlan.resultColumns,
      CanonicalContexts.Terminal.branchInput] using
      visible.branchEncoded
  change ArmPlan.HonestActiveOccurrences visible.assignment
    [selector.occurrence]
  exact ⟨
    selector.honestActive visible.assignment
      (terminalInputValues parameters statement proof)
      (branchValues parameters
        (decide (statement.iteration = 0)) statement proof)
      visible.inputEncoded resultEncoded semantic,
    True.intro⟩

theorem baseHonestActive
    (parameters : Parameters)
    (profile : Profile parameters)
    (recipes :
      CallRecipes (signature parameters) (profile.family parameters))
    (statement : TerminalStatementFor parameters)
    (proof : TerminalProofFor parameters)
    (visible : VisibleWitness parameters profile statement proof)
    (condition :
      stateEqual parameters statement.zi statement.z0 = true) :
    (CanonicalCompletionPlans.Terminal.onTrue
      parameters profile recipes).HonestActive visible.assignment := by
  let equality :=
    CanonicalTerminalPlan.baseEqualityPlan parameters profile recipes
  let assertion :=
    CanonicalTerminalPlan.baseAssertionPlan parameters profile
  have equalitySemantic :
      (baseStateEqualCall parameters).Holds
        (branchValues parameters
          (decide (statement.iteration = 0)) statement proof)
        (afterBaseEqualityValues parameters
          (decide (statement.iteration = 0)) statement proof) :=
    ((baseStateEqualCall parameters).exec_eq_some_iff_holds
      (branchValues parameters
        (decide (statement.iteration = 0)) statement proof)
      (afterBaseEqualityValues parameters
        (decide (statement.iteration = 0)) statement proof)).1
      (baseStateEqualCall_exec parameters
        (decide (statement.iteration = 0)) statement proof)
  have equalityResultEncoded :
      Columns.Encodes (profile.family parameters)
        equality.resultColumns visible.assignment
        (afterBaseEqualityValues parameters
          (decide (statement.iteration = 0)) statement proof) := by
    simpa [equality, CanonicalTerminalPlan.baseEqualityPlan,
      PrimitivePlan.resultColumns,
      CanonicalContexts.Terminal.afterBaseEquality] using
      visible.baseEncoded
  have equalityHonest :=
    equality.honestActive visible.assignment
      (branchValues parameters
        (decide (statement.iteration = 0)) statement proof)
      (afterBaseEqualityValues parameters
        (decide (statement.iteration = 0)) statement proof)
      visible.branchEncoded equalityResultEncoded equalitySemantic
  have assertionSemantic :
      (Primitive.assertTrue (signature := signature parameters)
        (.here (Ports.auxiliaryBit parameters))).Holds
          (afterBaseEqualityValues parameters
            (decide (statement.iteration = 0)) statement proof)
          (afterBaseEqualityValues parameters
            (decide (statement.iteration = 0)) statement proof) := by
    exact ⟨condition, rfl⟩
  have assertionResultEncoded :
      Columns.Encodes (profile.family parameters)
        assertion.resultColumns visible.assignment
        (afterBaseEqualityValues parameters
          (decide (statement.iteration = 0)) statement proof) := by
    simpa [assertion, CanonicalTerminalPlan.baseAssertionPlan,
      PrimitivePlan.resultColumns] using visible.baseEncoded
  have assertionHonest :=
    assertion.honestActive visible.assignment
      (afterBaseEqualityValues parameters
        (decide (statement.iteration = 0)) statement proof)
      (afterBaseEqualityValues parameters
        (decide (statement.iteration = 0)) statement proof)
      visible.baseEncoded assertionResultEncoded assertionSemantic
  change ArmPlan.HonestActiveOccurrences visible.assignment
    [equality.occurrence, assertion.occurrence]
  exact ⟨equalityHonest, assertionHonest, True.intro⟩

theorem recursiveHonestActive
    (parameters : Parameters)
    (profile : Profile parameters)
    (recipes :
      CallRecipes (signature parameters) (profile.family parameters))
    (statement : TerminalStatementFor parameters)
    (proof : TerminalProofFor parameters)
    (visible : VisibleWitness parameters profile statement proof)
    (conditions :
      priorLinkAccepted parameters statement proof = true ∧
        runningAcceptedValue parameters proof = true ∧
        freshAcceptedValue parameters proof = true) :
    (CanonicalCompletionPlans.Terminal.onFalse
      parameters profile recipes).HonestActive visible.assignment := by
  let hash :=
    CanonicalTerminalPlan.recursiveHashPlan parameters profile recipes
  let freshPublic :=
    CanonicalTerminalPlan.recursiveFreshPublicPlan
      parameters profile recipes
  let encode :=
    CanonicalTerminalPlan.recursiveEncodePlan parameters profile recipes
  let equality :=
    CanonicalTerminalPlan.recursiveEncodedEqualityPlan
      parameters profile recipes
  let priorAssertion :=
    CanonicalTerminalPlan.recursivePriorAssertionPlan parameters profile
  let running :=
    CanonicalTerminalPlan.recursiveRunningCheckPlan
      parameters profile recipes
  let runningAssertion :=
    CanonicalTerminalPlan.recursiveRunningAssertionPlan parameters profile
  let fresh :=
    CanonicalTerminalPlan.recursiveFreshCheckPlan
      parameters profile recipes
  let freshAssertion :=
    CanonicalTerminalPlan.recursiveFreshAssertionPlan parameters profile

  have hashResultEncoded :
      Columns.Encodes (profile.family parameters)
        hash.resultColumns visible.assignment
        (afterHashValues parameters
          (decide (statement.iteration = 0)) statement proof) := by
    simpa [hash, CanonicalTerminalPlan.recursiveHashPlan,
      PrimitivePlan.resultColumns,
      CanonicalContexts.Terminal.afterHash] using visible.afterHashEncoded
  have hashHonest :=
    hash.honestActive visible.assignment
      (branchValues parameters
        (decide (statement.iteration = 0)) statement proof)
      (afterHashValues parameters
        (decide (statement.iteration = 0)) statement proof)
      visible.branchEncoded hashResultEncoded
      (((hashPriorCall parameters).exec_eq_some_iff_holds _ _).1
        (hashPriorCall_exec parameters
          (decide (statement.iteration = 0)) statement proof))

  have freshPublicResultEncoded :
      Columns.Encodes (profile.family parameters)
        freshPublic.resultColumns visible.assignment
        (afterFreshPublicValues parameters
          (decide (statement.iteration = 0)) statement proof) := by
    simpa [freshPublic, CanonicalTerminalPlan.recursiveFreshPublicPlan,
      PrimitivePlan.resultColumns,
      CanonicalContexts.Terminal.afterFreshPublic] using
      visible.afterFreshPublicEncoded
  have freshPublicHonest :=
    freshPublic.honestActive visible.assignment
      (afterHashValues parameters
        (decide (statement.iteration = 0)) statement proof)
      (afterFreshPublicValues parameters
        (decide (statement.iteration = 0)) statement proof)
      visible.afterHashEncoded freshPublicResultEncoded
      (((freshPublicCall parameters).exec_eq_some_iff_holds _ _).1
        (freshPublicCall_exec parameters
          (decide (statement.iteration = 0)) statement proof))

  have encodeResultEncoded :
      Columns.Encodes (profile.family parameters)
        encode.resultColumns visible.assignment
        (afterEncodeValues parameters
          (decide (statement.iteration = 0)) statement proof) := by
    simpa [encode, CanonicalTerminalPlan.recursiveEncodePlan,
      PrimitivePlan.resultColumns,
      CanonicalContexts.Terminal.afterEncode] using
      visible.afterEncodeEncoded
  have encodeHonest :=
    encode.honestActive visible.assignment
      (afterFreshPublicValues parameters
        (decide (statement.iteration = 0)) statement proof)
      (afterEncodeValues parameters
        (decide (statement.iteration = 0)) statement proof)
      visible.afterFreshPublicEncoded encodeResultEncoded
      (((encodeInstanceCall parameters).exec_eq_some_iff_holds _ _).1
        (encodeInstanceCall_exec parameters
          (decide (statement.iteration = 0)) statement proof))

  have equalityResultEncoded :
      Columns.Encodes (profile.family parameters)
        equality.resultColumns visible.assignment
        (afterEncodedEqualityValues parameters
          (decide (statement.iteration = 0)) statement proof) := by
    simpa [equality,
      CanonicalTerminalPlan.recursiveEncodedEqualityPlan,
      PrimitivePlan.resultColumns,
      CanonicalContexts.Terminal.afterEncodedEquality] using
      visible.afterEqualityEncoded
  have equalityHonest :=
    equality.honestActive visible.assignment
      (afterEncodeValues parameters
        (decide (statement.iteration = 0)) statement proof)
      (afterEncodedEqualityValues parameters
        (decide (statement.iteration = 0)) statement proof)
      visible.afterEncodeEncoded equalityResultEncoded
      (((encodedEqualCall parameters).exec_eq_some_iff_holds _ _).1
        (encodedEqualCall_exec parameters
          (decide (statement.iteration = 0)) statement proof))

  have priorAssertionResultEncoded :
      Columns.Encodes (profile.family parameters)
        priorAssertion.resultColumns visible.assignment
        (afterEncodedEqualityValues parameters
          (decide (statement.iteration = 0)) statement proof) := by
    simpa [priorAssertion,
      CanonicalTerminalPlan.recursivePriorAssertionPlan,
      PrimitivePlan.resultColumns] using visible.afterEqualityEncoded
  have priorAssertionHonest :=
    priorAssertion.honestActive visible.assignment
      (afterEncodedEqualityValues parameters
        (decide (statement.iteration = 0)) statement proof)
      (afterEncodedEqualityValues parameters
        (decide (statement.iteration = 0)) statement proof)
      visible.afterEqualityEncoded priorAssertionResultEncoded
      (by exact ⟨conditions.1, rfl⟩)

  have runningResultEncoded :
      Columns.Encodes (profile.family parameters)
        running.resultColumns visible.assignment
        (afterRunningCheckValues parameters
          (decide (statement.iteration = 0)) statement proof) := by
    simpa [running, CanonicalTerminalPlan.recursiveRunningCheckPlan,
      PrimitivePlan.resultColumns,
      CanonicalContexts.Terminal.afterRunningCheck] using
      visible.afterRunningEncoded
  have runningHonest :=
    running.honestActive visible.assignment
      (afterEncodedEqualityValues parameters
        (decide (statement.iteration = 0)) statement proof)
      (afterRunningCheckValues parameters
        (decide (statement.iteration = 0)) statement proof)
      visible.afterEqualityEncoded runningResultEncoded
      (((runningCheckCall parameters).exec_eq_some_iff_holds _ _).1
        (runningCheckCall_exec parameters
          (decide (statement.iteration = 0)) statement proof))

  have runningAssertionResultEncoded :
      Columns.Encodes (profile.family parameters)
        runningAssertion.resultColumns visible.assignment
        (afterRunningCheckValues parameters
          (decide (statement.iteration = 0)) statement proof) := by
    simpa [runningAssertion,
      CanonicalTerminalPlan.recursiveRunningAssertionPlan,
      PrimitivePlan.resultColumns] using visible.afterRunningEncoded
  have runningAssertionHonest :=
    runningAssertion.honestActive visible.assignment
      (afterRunningCheckValues parameters
        (decide (statement.iteration = 0)) statement proof)
      (afterRunningCheckValues parameters
        (decide (statement.iteration = 0)) statement proof)
      visible.afterRunningEncoded runningAssertionResultEncoded
      (by exact ⟨conditions.2.1, rfl⟩)

  have freshResultEncoded :
      Columns.Encodes (profile.family parameters)
        fresh.resultColumns visible.assignment
        (afterFreshCheckValues parameters
          (decide (statement.iteration = 0)) statement proof) := by
    simpa [fresh, CanonicalTerminalPlan.recursiveFreshCheckPlan,
      PrimitivePlan.resultColumns,
      CanonicalContexts.Terminal.afterFreshCheck] using
      visible.recursiveEncoded
  have freshHonest :=
    fresh.honestActive visible.assignment
      (afterRunningCheckValues parameters
        (decide (statement.iteration = 0)) statement proof)
      (afterFreshCheckValues parameters
        (decide (statement.iteration = 0)) statement proof)
      visible.afterRunningEncoded freshResultEncoded
      (((freshCheckCall parameters).exec_eq_some_iff_holds _ _).1
        (freshCheckCall_exec parameters
          (decide (statement.iteration = 0)) statement proof))

  have freshAssertionResultEncoded :
      Columns.Encodes (profile.family parameters)
        freshAssertion.resultColumns visible.assignment
        (afterFreshCheckValues parameters
          (decide (statement.iteration = 0)) statement proof) := by
    simpa [freshAssertion,
      CanonicalTerminalPlan.recursiveFreshAssertionPlan,
      PrimitivePlan.resultColumns] using visible.recursiveEncoded
  have freshAssertionHonest :=
    freshAssertion.honestActive visible.assignment
      (afterFreshCheckValues parameters
        (decide (statement.iteration = 0)) statement proof)
      (afterFreshCheckValues parameters
        (decide (statement.iteration = 0)) statement proof)
      visible.recursiveEncoded freshAssertionResultEncoded
      (by exact ⟨conditions.2.2, rfl⟩)

  change ArmPlan.HonestActiveOccurrences visible.assignment
    [hash.occurrence, freshPublic.occurrence, encode.occurrence,
      equality.occurrence, priorAssertion.occurrence, running.occurrence,
      runningAssertion.occurrence, fresh.occurrence,
      freshAssertion.occurrence]
  exact ⟨hashHonest, freshPublicHonest, encodeHonest, equalityHonest,
    priorAssertionHonest, runningHonest, runningAssertionHonest,
    freshHonest, freshAssertionHonest, True.intro⟩

theorem baseHonestInactive
    (parameters : Parameters)
    (profile : Profile parameters)
    (recipes :
      CallRecipes (signature parameters) (profile.family parameters))
    (assignment : ColumnId -> Field) :
    (CanonicalCompletionPlans.Terminal.onTrue
      parameters profile recipes).HonestInactive assignment := by
  let equality :=
    CanonicalTerminalPlan.baseEqualityPlan parameters profile recipes
  let assertion :=
    CanonicalTerminalPlan.baseAssertionPlan parameters profile
  change ArmPlan.HonestInactiveOccurrences assignment
    [equality.occurrence, assertion.occurrence]
  exact ⟨equality.honestInactive assignment True.intro,
    assertion.honestInactive assignment True.intro, True.intro⟩

theorem recursiveHonestInactive
    (parameters : Parameters)
    (profile : Profile parameters)
    (recipes :
      CallRecipes (signature parameters) (profile.family parameters))
    (assignment : ColumnId -> Field) :
    (CanonicalCompletionPlans.Terminal.onFalse
      parameters profile recipes).HonestInactive assignment := by
  let hash :=
    CanonicalTerminalPlan.recursiveHashPlan parameters profile recipes
  let freshPublic :=
    CanonicalTerminalPlan.recursiveFreshPublicPlan
      parameters profile recipes
  let encode :=
    CanonicalTerminalPlan.recursiveEncodePlan parameters profile recipes
  let equality :=
    CanonicalTerminalPlan.recursiveEncodedEqualityPlan
      parameters profile recipes
  let priorAssertion :=
    CanonicalTerminalPlan.recursivePriorAssertionPlan parameters profile
  let running :=
    CanonicalTerminalPlan.recursiveRunningCheckPlan
      parameters profile recipes
  let runningAssertion :=
    CanonicalTerminalPlan.recursiveRunningAssertionPlan parameters profile
  let fresh :=
    CanonicalTerminalPlan.recursiveFreshCheckPlan
      parameters profile recipes
  let freshAssertion :=
    CanonicalTerminalPlan.recursiveFreshAssertionPlan parameters profile
  change ArmPlan.HonestInactiveOccurrences assignment
    [hash.occurrence, freshPublic.occurrence, encode.occurrence,
      equality.occurrence, priorAssertion.occurrence, running.occurrence,
      runningAssertion.occurrence, fresh.occurrence,
      freshAssertion.occurrence]
  exact ⟨
    hash.honestInactive assignment True.intro,
    freshPublic.honestInactive assignment True.intro,
    encode.honestInactive assignment True.intro,
    equality.honestInactive assignment True.intro,
    priorAssertion.honestInactive assignment True.intro,
    running.honestInactive assignment True.intro,
    runningAssertion.honestInactive assignment True.intro,
    fresh.honestInactive assignment True.intro,
    freshAssertion.honestInactive assignment True.intro,
    True.intro⟩

/-- The completed occurrence groups, unchanged visible coordinates, and the
two canonical activation equations are exactly the physical receipt rows. -/
theorem physicalOfCompletedGroups
    (parameters : Parameters)
    (profile : Profile parameters)
    (recipes :
      CallRecipes (signature parameters) (profile.family parameters))
    (statement : TerminalStatementFor parameters)
    (proof : TerminalProofFor parameters)
    (visible : VisibleWitness parameters profile statement proof)
    (assignment : ColumnId -> Field)
    (alwaysAgrees :
      AgreesOn
        (CanonicalCompletionPlans.Terminal.always
          parameters profile recipes).visibleIds
        visible.assignment assignment)
    (baseAgrees :
      AgreesOn
        (CanonicalCompletionPlans.Terminal.onTrue
          parameters profile recipes).visibleIds
        visible.assignment assignment)
    (recursiveAgrees :
      AgreesOn
        (CanonicalCompletionPlans.Terminal.onFalse
          parameters profile recipes).visibleIds
        visible.assignment assignment)
    (alwaysRows :
      Satisfies
        (CanonicalCompletionPlans.Terminal.always
          parameters profile recipes).rows assignment)
    (baseRows :
      Satisfies
        (CanonicalCompletionPlans.Terminal.onTrue
          parameters profile recipes).rows assignment)
    (recursiveRows :
      Satisfies
        (CanonicalCompletionPlans.Terminal.onFalse
          parameters profile recipes).rows assignment) :
    (CanonicalTerminalSoundness.encoding
      parameters profile recipes).PhysicalSatisfies assignment ∧
      Columns.Encodes (profile.family parameters)
        (CanonicalContexts.Terminal.input parameters) assignment
        (terminalInputValues parameters statement proof) := by
  let always :=
    CanonicalCompletionPlans.Terminal.always parameters profile recipes
  let base :=
    CanonicalCompletionPlans.Terminal.onTrue parameters profile recipes
  let recursive :=
    CanonicalCompletionPlans.Terminal.onFalse parameters profile recipes
  have oneInBase : oneColumn ∈ base.visibleIds := by
    simp only [base, CanonicalCompletionPlans.Terminal.onTrue,
      ArmPlan.visibleIds, List.flatMap_cons, List.flatMap_nil,
      List.append_nil, List.mem_append]
    left
    change oneColumn ∈
      (PrimitivePlan.invoke
        (CanonicalTerminalPlan.baseEqualityInvokePlan
          parameters profile recipes)).occurrence.visibleIds
    exact
      (CanonicalTerminalPlan.baseEqualityInvokePlan
        parameters profile recipes).occurrenceOneMemVisible
  have trueInBase :
      activationColumn SourceOwners.terminalBranchPath true ∈
        base.visibleIds := by
    simp only [base, CanonicalCompletionPlans.Terminal.onTrue,
      ArmPlan.visibleIds, List.flatMap_cons, List.flatMap_nil,
      List.append_nil, List.mem_append]
    left
    change activationColumn SourceOwners.terminalBranchPath true ∈
      (PrimitivePlan.invoke
        (CanonicalTerminalPlan.baseEqualityInvokePlan
          parameters profile recipes)).occurrence.visibleIds
    exact
      (CanonicalTerminalPlan.baseEqualityInvokePlan
        parameters profile recipes).occurrenceActiveMemVisible
  have falseInRecursive :
      activationColumn SourceOwners.terminalBranchPath false ∈
        recursive.visibleIds := by
    simp only [recursive, CanonicalCompletionPlans.Terminal.onFalse,
      ArmPlan.visibleIds, List.flatMap_cons, List.flatMap_nil,
      List.append_nil, List.mem_append]
    left
    change activationColumn SourceOwners.terminalBranchPath false ∈
      (PrimitivePlan.invoke
        (CanonicalTerminalPlan.recursiveHashInvokePlan
          parameters profile recipes)).occurrence.visibleIds
    exact
      (CanonicalTerminalPlan.recursiveHashInvokePlan
        parameters profile recipes).occurrenceActiveMemVisible
  have controlsAgree :
      AgreesOn
        [oneColumn,
          activationColumn SourceOwners.terminalBranchPath true,
          activationColumn SourceOwners.terminalBranchPath false]
        visible.assignment assignment := by
    intro id member
    simp only [List.mem_cons, List.not_mem_nil, or_false] at member
    rcases member with oneMember | trueMember | falseMember
    · subst id
      exact baseAgrees oneColumn oneInBase
    · subst id
      exact baseAgrees _ trueInBase
    · subst id
      exact recursiveAgrees _ falseInRecursive
  have controls := visible.controls.of_agrees controlsAgree
  have inputAgrees :
      AgreesOn
        (CanonicalContexts.Terminal.input parameters
          ).toSchemaBundles.ids
        visible.assignment assignment := by
    apply agreesOn_of_subset _ alwaysAgrees
    intro id member
    simpa [always, CanonicalCompletionPlans.Terminal.always,
      ArmPlan.visibleIds] using
      (CanonicalTerminalPlan.selectorInvokePlan
        parameters profile recipes).occurrenceInputIdsSubsetVisible
          id member
  have inputEncoded :
      Columns.Encodes (profile.family parameters)
        (CanonicalContexts.Terminal.input parameters) assignment
        (terminalInputValues parameters statement proof) := by
    apply
      (CanonicalContexts.Terminal.input parameters
        ).toSchemaBundles.encodes_of_agrees
          (profile.family parameters) visible.assignment assignment
          (terminalInputValues parameters statement proof)
    · exact inputAgrees
    · exact visible.inputEncoded
  have branchDecoded :
      Columns.Decodes (profile.family parameters)
        (CanonicalContexts.Terminal.branchInput parameters)
        visible.assignment
        (branchValues parameters
          (decide (statement.iteration = 0)) statement proof) :=
    (CanonicalContexts.Terminal.branchInput parameters
      ).toSchemaBundles.decodes_of_encodes
        (profile.family parameters) visible.assignment
        (branchValues parameters
          (decide (statement.iteration = 0)) statement proof)
        visible.branchEncoded
  have selectorDecodedAtVisible :=
    CanonicalTerminalSoundness.decodedBitReference
      parameters profile visible.assignment
      (CanonicalContexts.Terminal.branchInput parameters)
      (branchValues parameters
        (decide (statement.iteration = 0)) statement proof)
      (BranchRef.iterationZero parameters)
      (CanonicalContexts.Terminal.branchInputWidths parameters profile)
      branchDecoded
  have selectorInBase :
      CanonicalContexts.Terminal.selector parameters profile ∈
        base.visibleIds := by
    simp only [base, CanonicalCompletionPlans.Terminal.onTrue,
      ArmPlan.visibleIds, List.flatMap_cons, List.flatMap_nil,
      List.append_nil, List.mem_append]
    left
    change CanonicalContexts.Terminal.selector parameters profile ∈
      (PrimitivePlan.invoke
        (CanonicalTerminalPlan.baseEqualityInvokePlan
          parameters profile recipes)).occurrence.visibleIds
    apply
      (CanonicalTerminalPlan.baseEqualityInvokePlan
        parameters profile recipes).occurrenceInputIdsSubsetVisible
    exact CanonicalPrimitivePlan.bitCoordinate_mem profile
      (BranchRef.iterationZero parameters)
      (CanonicalContexts.Terminal.branchInput parameters)
      (CanonicalContexts.Terminal.branchInputWidths parameters profile)
  have selectorDecoded :
      boolCodec.decode
          [assignment
            (CanonicalContexts.Terminal.selector parameters profile)] =
        some (decide (statement.iteration = 0)) := by
    rw [baseAgrees _ selectorInBase]
    exact selectorDecodedAtVisible
  let activation :=
    CanonicalBranchPlan.activationRecipe
      SourceOwners.terminalBranchPath oneColumn oneColumn
      (CanonicalContexts.Terminal.selector parameters profile)
  have activationRows : Satisfies activation.rows assignment := by
    by_cases iterationZero : statement.iteration = 0
    · apply activation.selected_true_complete assignment controls.1
      · simpa [iterationZero] using selectorDecoded
      · have trueOne :
            assignment
                (activationColumn SourceOwners.terminalBranchPath true) =
              1 := by
          simpa [iterationZero] using controls.2.1
        have falseZero :
            assignment
                (activationColumn SourceOwners.terminalBranchPath false) =
              0 := by
          simpa [iterationZero] using controls.2.2
        exact ⟨by simpa [activation] using trueOne.trans controls.1.symm,
          by simpa [activation] using falseZero⟩
    · apply activation.selected_false_complete assignment controls.1
      · simpa [iterationZero] using selectorDecoded
      · have trueZero :
            assignment
                (activationColumn SourceOwners.terminalBranchPath true) =
              0 := by
          simpa [iterationZero] using controls.2.1
        have falseOne :
            assignment
                (activationColumn SourceOwners.terminalBranchPath false) =
              1 := by
          simpa [iterationZero] using controls.2.2
        exact ⟨by simpa [activation] using trueZero,
          by simpa [activation] using falseOne.trans controls.1.symm⟩
  have bodyRows :
      Satisfies
        (always.rows ++
          (activation.rows ++ (base.rows ++ recursive.rows)))
        assignment := by
    apply (satisfies_append_iff always.rows _ assignment).2
    refine ⟨alwaysRows, ?_⟩
    apply (satisfies_append_iff activation.rows _ assignment).2
    refine ⟨activationRows, ?_⟩
    exact (satisfies_append_iff base.rows recursive.rows assignment).2
      ⟨baseRows, recursiveRows⟩
  refine ⟨?_, inputEncoded⟩
  constructor
  · rw [(CanonicalTerminalSoundness.encoding
      parameters profile recipes).oneExact]
    exact controls.1
  · change Satisfies
      ((CanonicalTerminalPlan.receipts parameters profile recipes
        ).flatMap fun receipt => receipt.rows) assignment
    simpa [CanonicalTerminalPlan.receipts,
      CanonicalTerminalPlan.bodyReceipts,
      always, base, recursive,
      CanonicalCompletionPlans.Terminal.always,
      CanonicalCompletionPlans.Terminal.onTrue,
      CanonicalCompletionPlans.Terminal.onFalse,
      ArmPlan.rows, CanonicalBranchPlan.activation_rows_conserved,
      CanonicalBranchPlan.emptyJoinReceipt, InputReceipts.rows_empty] using
      bodyRows

end CanonicalTerminalCompleteness

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding
