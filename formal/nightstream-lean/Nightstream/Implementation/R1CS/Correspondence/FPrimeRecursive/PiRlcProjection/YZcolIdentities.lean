import Nightstream.Implementation.R1CS.Artifacts.FPrimeRecursive.PiRlcProjection.YZcolIdentities
import Nightstream.Implementation.R1CS.Correspondence.FPrimeRecursive.PiRlcProjection.BetaLadder
import Nightstream.Implementation.R1CS.Correspondence.FPrimeRecursive.PiRlcProjection.RhoEvaluations
import Nightstream.Implementation.R1CS.Correspondence.FPrimeRecursive.YZcolProjection
import Nightstream.Implementation.R1CS.Correspondence.Projection.ProjectionBatchSound

/-!
Conditional source-R1CS soundness for both complete active PiRLC `y_zcol`
projection identities.

Owns: composition of separately owned beta, rho, output-evaluation, and new
identity-local rows into two complete `ProjectionTrace.evaluation_sound`
proofs, batch acceptance, and the deterministic exact-or-bad-root partition.

Does not own: transcript-derived beta/rho authority, semantic meaning of input,
output, or quotient columns, parent-opening authority, full-row embedding,
bad-root probability, normal-form reduction, encoded lowering, padding, or
permission to remove rows.

Emits constraints: no.

Assurance tier: conditional model-level semantics over artifact-checked active
fixed-profile source rows. Closed fixed-artifact facts proved with
`native_decide` inherit the native compiler trust boundary and have focused
trust-surface guards; the abstract projection implication remains ordinary
theorem composition. This is not whole-verifier Rust conformance or a
security reduction.

| Protocol → phase | Exact row premise | Deterministic result |
|---|---:|---|
| `projection_shared.beta_ladder` | 272 | ladder definitions hold |
| `projection_shared.rho_evaluations` | 1,620 | 15 rho evaluator definitions hold |
| `identities.y_zcol.evaluations.output` | 216 | both output evaluator definitions hold |
| remaining `identities.y_zcol` leaves | 3,616 | local definitions and final checks hold |
| complete two-identity batch | all four premises | `BatchExact ∨ BatchBadRoot` |
-/

namespace Nightstream.Implementation.R1CS.FPrimeRecursivePiRlcProjection.YZcolIdentities.Refinement

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.FPrimeRecursivePiRlcProjection.YZcolIdentities
open Nightstream.SuperNeo.ProjectionCheck

set_option maxRecDepth 100000

def betaSourceRows : List Row :=
  FPrimeRecursivePiRlcProjection.BetaLadder.ownedSourceRows

def rhoSourceRows : List Row :=
  FPrimeRecursivePiRlcProjection.RhoEvaluations.ownedSourceRows

def outputSourceRows : List Row :=
  FPrimeRecursiveYZcolProjection.ownedSourceRows

def newLocalSourceRowsOnly : List Row :=
  newLocalSourceRows.map Prod.snd

def definitionsOf (definitions : List (Nat × Program.Definition)) :
    List Program.Definition :=
  definitions.map Prod.snd

def rowsOf (rows : List (Nat × Row)) : List Row :=
  rows.map Prod.snd

private theorem definitionsHold_of_matched_source
    {rows : List (Nat × Row)}
    {definitions : List (Nat × Program.Definition)}
    {assignment : Nat → Nat}
    (assignmentCanonical : ∀ column, assignment column < goldilocksP)
    (constantOne : assignment 0 = 1)
    (definitionsCanonical :
      ∀ definition ∈ definitionsOf definitions, definition.Canonical)
    (rowsMatch : RowsMatchDefinitions rows definitions)
    (sourceSatisfies : Satisfies (rowsOf rows) assignment) :
    ProjectionProgram.DefinitionsHold assignment
      (definitionsOf definitions) := by
  have builderRows : Satisfies
      (definitions.map fun entry => entry.2.builderRow) assignment := by
    exact ActiveIndexedRows.builderRows_satisfied_of_indexedRowsMatch
      rows definitions rowsMatch sourceSatisfies
  have normalizedBuilderRows : Satisfies
      ((definitionsOf definitions).map Program.Definition.builderRow)
      assignment := by
    simpa [definitionsOf, List.map_map] using builderRows
  exact Program.builderDefinitions_sound assignmentCanonical constantOne
    definitionsCanonical normalizedBuilderRows

private theorem satisfies_of_subset
    {small large : List Row} {assignment : Nat → Nat}
    (subset : ∀ row ∈ small, row ∈ large)
    (largeSatisfies : Satisfies large assignment) :
    Satisfies small assignment := by
  intro row member
  exact largeSatisfies row (subset row member)

def limb0InputDefinitions0 : List Program.Definition :=
  definitionsOf limb0InputRowDefinitions0
def limb0InputDefinitions1 : List Program.Definition :=
  definitionsOf limb0InputRowDefinitions1
def limb0InputDefinitions2 : List Program.Definition :=
  definitionsOf limb0InputRowDefinitions2
def limb1InputDefinitions0 : List Program.Definition :=
  definitionsOf limb1InputRowDefinitions0
def limb1InputDefinitions1 : List Program.Definition :=
  definitionsOf limb1InputRowDefinitions1
def limb1InputDefinitions2 : List Program.Definition :=
  definitionsOf limb1InputRowDefinitions2
def limb0TailDefinitions : List Program.Definition :=
  definitionsOf limb0TailDefinitionRowDefinitions
def limb1TailDefinitions : List Program.Definition :=
  definitionsOf limb1TailDefinitionRowDefinitions

def limb0NewLocalDefinitions : List Program.Definition :=
  limb0InputDefinitions0 ++ limb0InputDefinitions1 ++
    limb0InputDefinitions2 ++ limb0TailDefinitions

def limb1NewLocalDefinitions : List Program.Definition :=
  limb1InputDefinitions0 ++ limb1InputDefinitions1 ++
    limb1InputDefinitions2 ++ limb1TailDefinitions

private theorem limb0Input0_canonical :
    ∀ definition ∈ limb0InputDefinitions0, definition.Canonical := by
  native_decide
private theorem limb0Input1_canonical :
    ∀ definition ∈ limb0InputDefinitions1, definition.Canonical := by
  native_decide
private theorem limb0Input2_canonical :
    ∀ definition ∈ limb0InputDefinitions2, definition.Canonical := by
  native_decide
private theorem limb1Input0_canonical :
    ∀ definition ∈ limb1InputDefinitions0, definition.Canonical := by
  native_decide
private theorem limb1Input1_canonical :
    ∀ definition ∈ limb1InputDefinitions1, definition.Canonical := by
  native_decide
private theorem limb1Input2_canonical :
    ∀ definition ∈ limb1InputDefinitions2, definition.Canonical := by
  native_decide
private theorem limb0Tail_canonical :
    ∀ definition ∈ limb0TailDefinitions, definition.Canonical := by
  native_decide
private theorem limb1Tail_canonical :
    ∀ definition ∈ limb1TailDefinitions, definition.Canonical := by
  native_decide

private theorem componentSatisfies
    {component : List (Nat × Row)} {assignment : Nat → Nat}
    (componentSubset :
      ∀ entry ∈ component, entry ∈ newLocalSourceRows)
    (localSatisfies : Satisfies newLocalSourceRowsOnly assignment) :
    Satisfies (rowsOf component) assignment := by
  apply satisfies_of_subset _ localSatisfies
  intro row member
  rcases List.mem_map.mp member with ⟨entry, entryMember, rowEquality⟩
  exact List.mem_map.mpr
    ⟨entry, componentSubset entry entryMember, rowEquality⟩

private theorem limb0Input0_satisfies
    {assignment : Nat → Nat}
    (localSatisfies : Satisfies newLocalSourceRowsOnly assignment) :
    Satisfies (rowsOf limb0InputSourceRows0) assignment := by
  apply componentSatisfies _ localSatisfies
  intro entry member
  simp [newLocalSourceRows,
    limb0NewLocalSourceRows, member]

private theorem limb0Input1_satisfies
    {assignment : Nat → Nat}
    (localSatisfies : Satisfies newLocalSourceRowsOnly assignment) :
    Satisfies (rowsOf limb0InputSourceRows1) assignment := by
  apply componentSatisfies _ localSatisfies
  intro entry member
  simp [newLocalSourceRows,
    limb0NewLocalSourceRows, member]

private theorem limb0Input2_satisfies
    {assignment : Nat → Nat}
    (localSatisfies : Satisfies newLocalSourceRowsOnly assignment) :
    Satisfies (rowsOf limb0InputSourceRows2) assignment := by
  apply componentSatisfies _ localSatisfies
  intro entry member
  simp [newLocalSourceRows,
    limb0NewLocalSourceRows, member]

private theorem limb1Input0_satisfies
    {assignment : Nat → Nat}
    (localSatisfies : Satisfies newLocalSourceRowsOnly assignment) :
    Satisfies (rowsOf limb1InputSourceRows0) assignment := by
  apply componentSatisfies _ localSatisfies
  intro entry member
  simp [newLocalSourceRows,
    limb1NewLocalSourceRows, member]

private theorem limb1Input1_satisfies
    {assignment : Nat → Nat}
    (localSatisfies : Satisfies newLocalSourceRowsOnly assignment) :
    Satisfies (rowsOf limb1InputSourceRows1) assignment := by
  apply componentSatisfies _ localSatisfies
  intro entry member
  simp [newLocalSourceRows,
    limb1NewLocalSourceRows, member]

private theorem limb1Input2_satisfies
    {assignment : Nat → Nat}
    (localSatisfies : Satisfies newLocalSourceRowsOnly assignment) :
    Satisfies (rowsOf limb1InputSourceRows2) assignment := by
  apply componentSatisfies _ localSatisfies
  intro entry member
  simp [newLocalSourceRows,
    limb1NewLocalSourceRows, member]

private theorem limb0Tail_satisfies
    {assignment : Nat → Nat}
    (localSatisfies : Satisfies newLocalSourceRowsOnly assignment) :
    Satisfies (rowsOf limb0TailDefinitionSourceRows) assignment := by
  apply componentSatisfies _ localSatisfies
  intro entry member
  simp [newLocalSourceRows,
    limb0NewLocalSourceRows, member]

private theorem limb1Tail_satisfies
    {assignment : Nat → Nat}
    (localSatisfies : Satisfies newLocalSourceRowsOnly assignment) :
    Satisfies (rowsOf limb1TailDefinitionSourceRows) assignment := by
  apply componentSatisfies _ localSatisfies
  intro entry member
  simp [newLocalSourceRows,
    limb1NewLocalSourceRows, member]

theorem newLocalSourceRows_definitionsHold
    {assignment : Nat → Nat}
    (assignmentCanonical : ∀ column, assignment column < goldilocksP)
    (constantOne : assignment 0 = 1)
    (localSatisfies : Satisfies newLocalSourceRowsOnly assignment) :
    ProjectionProgram.DefinitionsHold assignment
        limb0NewLocalDefinitions ∧
      ProjectionProgram.DefinitionsHold assignment
        limb1NewLocalDefinitions := by
  have l00 := definitionsHold_of_matched_source assignmentCanonical
    constantOne limb0Input0_canonical limb0_input_rows_match.1
    (limb0Input0_satisfies localSatisfies)
  have l01 := definitionsHold_of_matched_source assignmentCanonical
    constantOne limb0Input1_canonical limb0_input_rows_match.2.1
    (limb0Input1_satisfies localSatisfies)
  have l02 := definitionsHold_of_matched_source assignmentCanonical
    constantOne limb0Input2_canonical limb0_input_rows_match.2.2
    (limb0Input2_satisfies localSatisfies)
  have l0t := definitionsHold_of_matched_source assignmentCanonical
    constantOne limb0Tail_canonical tail_rows_match.1
    (limb0Tail_satisfies localSatisfies)
  have l10 := definitionsHold_of_matched_source assignmentCanonical
    constantOne limb1Input0_canonical limb1_input_rows_match.1
    (limb1Input0_satisfies localSatisfies)
  have l11 := definitionsHold_of_matched_source assignmentCanonical
    constantOne limb1Input1_canonical limb1_input_rows_match.2.1
    (limb1Input1_satisfies localSatisfies)
  have l12 := definitionsHold_of_matched_source assignmentCanonical
    constantOne limb1Input2_canonical limb1_input_rows_match.2.2
    (limb1Input2_satisfies localSatisfies)
  have l1t := definitionsHold_of_matched_source assignmentCanonical
    constantOne limb1Tail_canonical tail_rows_match.2
    (limb1Tail_satisfies localSatisfies)
  constructor
  · intro definition member
    simp only [limb0NewLocalDefinitions, List.mem_append] at member
    rcases member with rest | memberTail
    · rcases rest with rest | member2
      · rcases rest with member0 | member1
        · exact l00 definition member0
        · exact l01 definition member1
      · exact l02 definition member2
    · exact l0t definition memberTail
  · intro definition member
    simp only [limb1NewLocalDefinitions, List.mem_append] at member
    rcases member with rest | memberTail
    · rcases rest with rest | member2
      · rcases rest with member0 | member1
        · exact l10 definition member0
        · exact l11 definition member1
      · exact l12 definition member2
    · exact l1t definition memberTail

def SharedDefinitions (trace : ProjectionProgram.ProjectionTrace) :
    List Program.Definition :=
  trace.ladder.definitions ++
    trace.pairs.flatMap fun pair => pair.rhoEvaluation.definitions

def LocalDefinitions (trace : ProjectionProgram.ProjectionTrace) :
    List Program.Definition :=
  trace.pairs.flatMap (fun pair =>
      pair.inputEvaluation.definitions ++ pair.product.definitions) ++
    trace.outputEvaluation.definitions ++
    trace.quotientEvaluation.definitions ++
    trace.quotientPhiProduct.definitions

def ArtifactSharedDefinitions : List Program.Definition :=
  FPrimeRecursivePiRlcProjection.BetaLadder.Refinement.ownedDefinitions ++
    FPrimeRecursivePiRlcProjection.RhoEvaluations.Refinement.ownedDefinitions

def limb0ArtifactLocalDefinitions : List Program.Definition :=
  limb0NewLocalDefinitions ++ outputLimb0Owner.evalTrace.definitions

def limb1ArtifactLocalDefinitions : List Program.Definition :=
  limb1NewLocalDefinitions ++ outputLimb1Owner.evalTrace.definitions

theorem limb0_sharedDefinitions_exact :
    SharedDefinitions limb0Trace = ArtifactSharedDefinitions := by
  native_decide

theorem limb1_sharedDefinitions_exact :
    SharedDefinitions limb1Trace = ArtifactSharedDefinitions := by
  native_decide

theorem limb0_localDefinitions_permutation :
    (LocalDefinitions limb0Trace).Perm
      limb0ArtifactLocalDefinitions := by
  native_decide

theorem limb1_localDefinitions_permutation :
    (LocalDefinitions limb1Trace).Perm
      limb1ArtifactLocalDefinitions := by
  native_decide

theorem definitions_eq_shared_append_local
    (trace : ProjectionProgram.ProjectionTrace) :
    trace.definitions = SharedDefinitions trace ++ LocalDefinitions trace := by
  simp [ProjectionProgram.ProjectionTrace.definitions, SharedDefinitions,
    LocalDefinitions, List.append_assoc]

def SharedDefinitionsHold (assignment : Nat → Nat)
    (trace : ProjectionProgram.ProjectionTrace) : Prop :=
  ProjectionProgram.DefinitionsHold assignment (SharedDefinitions trace)

def LocalDefinitionsHold (assignment : Nat → Nat)
    (trace : ProjectionProgram.ProjectionTrace) : Prop :=
  ProjectionProgram.DefinitionsHold assignment (LocalDefinitions trace)

theorem sourceRows_sharedDefinitionsHold
    {assignment : Nat → Nat}
    (assignmentCanonical : ∀ column, assignment column < goldilocksP)
    (constantOne : assignment 0 = 1)
    (betaSatisfies : Satisfies betaSourceRows assignment)
    (rhoSatisfies : Satisfies rhoSourceRows assignment) :
    SharedDefinitionsHold assignment limb0Trace ∧
      SharedDefinitionsHold assignment limb1Trace := by
  have betaDefinitions :=
    FPrimeRecursivePiRlcProjection.BetaLadder.Refinement.ownedSourceRows_definitionsHold
      assignmentCanonical constantOne betaSatisfies
  have rhoDefinitions :=
    FPrimeRecursivePiRlcProjection.RhoEvaluations.Refinement.ownedSourceRows_definitionsHold
      assignmentCanonical constantOne rhoSatisfies
  have artifact : ProjectionProgram.DefinitionsHold assignment
      ArtifactSharedDefinitions := by
    intro definition member
    simp only [ArtifactSharedDefinitions, List.mem_append] at member
    rcases member with member | member
    · exact betaDefinitions definition member
    · exact rhoDefinitions definition member
  constructor
  · unfold SharedDefinitionsHold
    rw [limb0_sharedDefinitions_exact]
    exact artifact
  · unfold SharedDefinitionsHold
    rw [limb1_sharedDefinitions_exact]
    exact artifact

theorem sourceRows_localDefinitionsHold
    {assignment : Nat → Nat}
    (assignmentCanonical : ∀ column, assignment column < goldilocksP)
    (constantOne : assignment 0 = 1)
    (outputSatisfies : Satisfies outputSourceRows assignment)
    (localSatisfies : Satisfies newLocalSourceRowsOnly assignment) :
    LocalDefinitionsHold assignment limb0Trace ∧
      LocalDefinitionsHold assignment limb1Trace := by
  have newLocal := newLocalSourceRows_definitionsHold
    assignmentCanonical constantOne localSatisfies
  have outputDefinitions :=
    FPrimeRecursiveYZcolProjection.Refinement.ownedSourceRows_definitionsHold
      assignmentCanonical constantOne outputSatisfies
  have output0 : ProjectionProgram.DefinitionsHold assignment
      outputLimb0Owner.evalTrace.definitions := by
    intro definition member
    exact outputDefinitions definition (by
      apply List.mem_append_left
      exact member)
  have output1 : ProjectionProgram.DefinitionsHold assignment
      outputLimb1Owner.evalTrace.definitions := by
    intro definition member
    exact outputDefinitions definition (by
      apply List.mem_append_right
      exact member)
  have artifact0 : ProjectionProgram.DefinitionsHold assignment
      limb0ArtifactLocalDefinitions := by
    intro definition member
    simp only [limb0ArtifactLocalDefinitions, List.mem_append] at member
    rcases member with member | member
    · exact newLocal.1 definition member
    · exact output0 definition member
  have artifact1 : ProjectionProgram.DefinitionsHold assignment
      limb1ArtifactLocalDefinitions := by
    intro definition member
    simp only [limb1ArtifactLocalDefinitions, List.mem_append] at member
    rcases member with member | member
    · exact newLocal.2 definition member
    · exact output1 definition member
  constructor
  · intro definition member
    apply artifact0 definition
    rw [← limb0_localDefinitions_permutation.mem_iff]
    exact member
  · intro definition member
    apply artifact1 definition
    rw [← limb1_localDefinitions_permutation.mem_iff]
    exact member

private theorem traceDefinitionsHold_of_shared_local
    {assignment : Nat → Nat}
    {trace : ProjectionProgram.ProjectionTrace}
    (shared : SharedDefinitionsHold assignment trace)
    (localDefinitions : LocalDefinitionsHold assignment trace) :
    ProjectionProgram.DefinitionsHold assignment trace.definitions := by
  intro definition member
  rw [definitions_eq_shared_append_local] at member
  rcases List.mem_append.mp member with member | member
  · exact shared definition member
  · exact localDefinitions definition member

private theorem limb0CheckSource_satisfies
    {assignment : Nat → Nat}
    (localSatisfies : Satisfies newLocalSourceRowsOnly assignment) :
    Satisfies (rowsOf limb0CheckSourceRows) assignment := by
  apply componentSatisfies _ localSatisfies
  intro entry member
  simp [newLocalSourceRows,
    limb0NewLocalSourceRows, member]

private theorem limb1CheckSource_satisfies
    {assignment : Nat → Nat}
    (localSatisfies : Satisfies newLocalSourceRowsOnly assignment) :
    Satisfies (rowsOf limb1CheckSourceRows) assignment := by
  apply componentSatisfies _ localSatisfies
  intro entry member
  simp [newLocalSourceRows,
    limb1NewLocalSourceRows, member]

theorem limb0_checkRows_exact :
    limb0CheckRows.map Prod.snd = limb0Trace.checks := by
  native_decide

theorem limb1_checkRows_exact :
    limb1CheckRows.map Prod.snd = limb1Trace.checks := by
  native_decide

theorem sourceRows_checksHold
    {assignment : Nat → Nat}
    (localSatisfies : Satisfies newLocalSourceRowsOnly assignment) :
    Satisfies limb0Trace.checks assignment ∧
      Satisfies limb1Trace.checks assignment := by
  have check0 : Satisfies (limb0CheckRows.map Prod.snd) assignment := by
    apply ActiveIndexedRows.rows_satisfied_of_indexedRowsMatchRows
      limb0CheckSourceRows limb0CheckRows check_rows_match.1
    exact limb0CheckSource_satisfies localSatisfies
  have check1 : Satisfies (limb1CheckRows.map Prod.snd) assignment := by
    apply ActiveIndexedRows.rows_satisfied_of_indexedRowsMatchRows
      limb1CheckSourceRows limb1CheckRows check_rows_match.2
    exact limb1CheckSource_satisfies localSatisfies
  constructor
  · rwa [limb0_checkRows_exact] at check0
  · rwa [limb1_checkRows_exact] at check1

theorem completeSourceRows_evaluations_sound
    {assignment : Nat → Nat}
    (assignmentCanonical : ∀ column, assignment column < goldilocksP)
    (constantOne : assignment 0 = 1)
    (betaSatisfies : Satisfies betaSourceRows assignment)
    (rhoSatisfies : Satisfies rhoSourceRows assignment)
    (outputSatisfies : Satisfies outputSourceRows assignment)
    (localSatisfies : Satisfies newLocalSourceRowsOnly assignment) :
    eval ProjectionProgram.K.ops
        (limb0Trace.identity assignment).lhs
        (limb0Trace.identity assignment).beta =
      eval ProjectionProgram.K.ops
        (limb0Trace.identity assignment).rhs
        (limb0Trace.identity assignment).beta ∧
    eval ProjectionProgram.K.ops
        (limb1Trace.identity assignment).lhs
        (limb1Trace.identity assignment).beta =
      eval ProjectionProgram.K.ops
        (limb1Trace.identity assignment).rhs
        (limb1Trace.identity assignment).beta := by
  have shared := sourceRows_sharedDefinitionsHold assignmentCanonical
    constantOne betaSatisfies rhoSatisfies
  have localProof := sourceRows_localDefinitionsHold assignmentCanonical
    constantOne outputSatisfies localSatisfies
  have checks := sourceRows_checksHold localSatisfies
  constructor
  · exact limb0Trace.evaluation_sound assignment constantOne limb0_layout
      (traceDefinitionsHold_of_shared_local shared.1 localProof.1) checks.1
  · exact limb1Trace.evaluation_sound assignment constantOne limb1_layout
      (traceDefinitionsHold_of_shared_local shared.2 localProof.2) checks.2

theorem limb0_pairs_nonempty : limb0Trace.pairs ≠ [] := by native_decide
theorem limb1_pairs_nonempty : limb1Trace.pairs ≠ [] := by native_decide

theorem limb0_pair_widths : ∀ pair ∈ limb0Trace.pairs,
    pair.rhoColumns.length = 54 ∧ pair.inputColumns.length = 54 := by
  native_decide

theorem limb1_pair_widths : ∀ pair ∈ limb1Trace.pairs,
    pair.rhoColumns.length = 54 ∧ pair.inputColumns.length = 54 := by
  native_decide

theorem completeSourceRows_batchAccepted
    {assignment : Nat → Nat}
    (assignmentCanonical : ∀ column, assignment column < goldilocksP)
    (constantOne : assignment 0 = 1)
    (betaSatisfies : Satisfies betaSourceRows assignment)
    (rhoSatisfies : Satisfies rhoSourceRows assignment)
    (outputSatisfies : Satisfies outputSourceRows assignment)
    (localSatisfies : Satisfies newLocalSourceRowsOnly assignment) :
    BatchAccepted ProjectionProgram.K.ops
      (ProjectionProgram.BatchIdentity traces assignment) := by
  have evaluations := completeSourceRows_evaluations_sound
    assignmentCanonical constantOne betaSatisfies rhoSatisfies
    outputSatisfies localSatisfies
  intro identity member
  simp only [ProjectionProgram.BatchIdentity, traces, List.map_cons,
    List.map_nil, List.mem_cons, List.not_mem_nil, or_false] at member
  rcases member with rfl | rfl
  · exact ⟨limb0Trace.identity_wellFormed_of_widths assignment
      limb0_layout limb0_pairs_nonempty limb0_pair_widths, evaluations.1⟩
  · exact ⟨limb1Trace.identity_wellFormed_of_widths assignment
      limb1_layout limb1_pairs_nonempty limb1_pair_widths, evaluations.2⟩

/-- Exact active source-row satisfaction gives the full deterministic PiRLC
projection guarantee. The right branch remains a named bad-root event; this
theorem deliberately assigns it no probability bound. -/
theorem completeSourceRows_batchExact_or_badRoot
    {assignment : Nat → Nat}
    (assignmentCanonical : ∀ column, assignment column < goldilocksP)
    (constantOne : assignment 0 = 1)
    (betaSatisfies : Satisfies betaSourceRows assignment)
    (rhoSatisfies : Satisfies rhoSourceRows assignment)
    (outputSatisfies : Satisfies outputSourceRows assignment)
    (localSatisfies : Satisfies newLocalSourceRowsOnly assignment) :
    BatchExact (ProjectionProgram.BatchIdentity traces assignment) ∨
      BatchBadRoot ProjectionProgram.K.ops
        (ProjectionProgram.BatchIdentity traces assignment) := by
  exact batchAccepted_implies_exact_or_badRoot ProjectionProgram.K.ops
    (ProjectionProgram.BatchIdentity traces assignment)
    (completeSourceRows_batchAccepted assignmentCanonical constantOne
      betaSatisfies rhoSatisfies outputSatisfies localSatisfies)

end Nightstream.Implementation.R1CS.FPrimeRecursivePiRlcProjection.YZcolIdentities.Refinement
