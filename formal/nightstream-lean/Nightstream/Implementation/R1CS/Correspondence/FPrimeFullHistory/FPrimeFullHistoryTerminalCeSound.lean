import Nightstream.Implementation.R1CS.Ownership.FPrimeFullHistory.FPrimeFullHistoryTerminalCeArtifact
import Nightstream.Implementation.R1CS.Correspondence.Gadgets.TerminalCeSound

/-!
Artifact-level semantic soundness for every direct terminal-CE child in the
supported full-history profile.

The generated artifact certifies exact row identity, phase boundaries, and
column ownership.  This module derives `TerminalCE.ClaimHolds` from those rows;
no generated field or certificate carries that conclusion.
-/

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalCeSound

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Program
open Nightstream.Implementation.R1CS.CheckedProgram
open Nightstream.Implementation.R1CS.TerminalCeCompiler
open Nightstream.Implementation.R1CS.TerminalCeSound
open Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalCe

set_option maxRecDepth 524288
set_option maxHeartbeats 4000000

private theorem semantic_known
    {column : Nat}
    (member : column ∈ TerminalCeCompiler.semanticColumns layout) :
    column ∈ knownAfter inputColumns (definitions instructions) :=
  semantic_columns_known column member

/-- Exact CIR-SOUND theorem for the canonical 21,542-row direct-CE program. -/
theorem canonical_sound
    (prime : EuclidPrime goldilocksP)
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfies : Satisfies rows assignment) :
    TerminalCeCompiler.ClaimHolds program assignment := by
  let final := program.final assignment
  have checked : SoundResult inputColumns instructions assignment assignment :=
    CheckedProgram.sound definitions_wellFormed definitions_canonical
      checks_reference (fun _ _ => rfl) canonical one satisfies
  have finalCanonical : ∀ column, final column < goldilocksP := by
    exact run_canonical canonical
  have zeroInput : 0 ∈ inputColumns := by native_decide
  have finalOne : final 0 = 1 := by
    exact (run_preserves_known definitions_wellFormed assignment 0 zeroInput).trans one

  have commitmentSlice :
      Satisfies (checks program.commitmentInstructions) final :=
    checks_slice_satisfy checked.checksHold schedule.commitmentStart
      schedule.commitmentEnd
  have commitmentChecksSatisfy :
      Satisfies (LinearOutputs.rows program.commitmentChecks) final := by
    rw [commitment_checks_match]
    exact commitmentSlice
  have commitmentEqualities := LinearOutputs.rows_sound finalCanonical finalOne
    linear_checks_canonical.1 commitmentChecksSatisfy
  have commitmentValues :
      valuesAt assignment layout.commitmentCols =
        program.expectedFields assignment Program.commitmentChecks := by
    apply valuesAt_outputs_eq_expected commitment_check_outputs
    · intro output member
      apply semantic_known
      simp [TerminalCeCompiler.semanticColumns, member]
    · exact checked.agreement
    · exact commitmentEqualities
  have commitmentHolds :
      program.expectedCommitment assignment =
        decodeCommitment layout assignment := by
    exact commitmentValues.symm

  have publicSlice :
      Satisfies (CheckedProgram.rows program.publicInstructions) assignment :=
    rows_slice_satisfy satisfies schedule.publicInputStart
      schedule.publicInputEnd
  have publicRowsSatisfy :
      Satisfies
        (LinearOutputs.rows (TerminalCeCompiler.projectionChecks layout))
        assignment := by
    rw [← public_program_match]
    exact publicSlice
  have publicHolds :
      decodePublicInput layout assignment = projectedPublic layout assignment :=
    projection_sound canonical one publicRowsSatisfy

  have normSlice :
      Satisfies (CheckedProgram.rows program.normInstructionsSlice) assignment :=
    rows_slice_satisfy satisfies schedule.normStart schedule.normEnd
  have normRowsSatisfy :
      Satisfies
        (CheckedProgram.rows (TerminalCeCompiler.normInstructions layout))
        assignment := by
    rw [← norm_program_match]
    exact normSlice
  have normHolds : NormHolds layout assignment :=
    normInstructions_sound prime canonical one normRowsSatisfy

  have evaluationSlice :
      Satisfies (checks program.evaluationInstructions) final :=
    checks_slice_satisfy checked.checksHold schedule.evaluationsStart
      schedule.evaluationsEnd
  have evaluationChecksSatisfy :
      Satisfies (LinearOutputs.rows program.evaluationChecks) final := by
    rw [evaluation_checks_match]
    exact evaluationSlice
  have evaluationEqualities := LinearOutputs.rows_sound finalCanonical finalOne
    linear_checks_canonical.2.1 evaluationChecksSatisfy
  have evaluationValues :
      valuesAt assignment layout.evaluationCols.flatten =
        program.expectedFields assignment Program.evaluationChecks := by
    apply valuesAt_outputs_eq_expected evaluation_check_outputs
    · intro output member
      apply semantic_known
      simp [TerminalCeCompiler.semanticColumns, member]
    · exact checked.agreement
    · exact evaluationEqualities
  have evaluationsHold :
      decodeEvaluations layout assignment =
        program.expectedEvaluations assignment := by
    unfold decodeEvaluations Program.expectedEvaluations
    change (layout.evaluationCols.map fun row => pairs (valuesAt assignment row)) =
      (Program.splitByLengths (layout.evaluationCols.map List.length)
        (program.expectedFields assignment Program.evaluationChecks)).map pairs
    rw [← evaluationValues]
    unfold valuesAt
    rw [splitByLengths_map_flatten]
    simp [List.map_map, Function.comp_def]

  have constantTermSlice :
      Satisfies (CheckedProgram.rows program.constantTermInstructions)
        assignment :=
    rows_slice_satisfy satisfies schedule.constantTermStart
      schedule.constantTermEnd
  have constantTermRowsSatisfy :
      Satisfies
        (LinearOutputs.rows (TerminalCeCompiler.constantTermChecks layout))
        assignment := by
    rw [← constant_term_program_match]
    exact constantTermSlice
  have constantTermsHold :
      (decodeEvaluations layout assignment).map
          (fun evaluation => evaluation.headD ProjectionProgram.K.zero) =
        decodeConstantTerms layout assignment :=
    constantTermChecks_sound layout_shape canonical one constantTermRowsSatisfy

  have ncSlice : Satisfies (checks program.ncInstructions) final :=
    checks_slice_satisfy checked.checksHold schedule.ncChannelStart
      schedule.ncChannelEnd
  have ncChecksSatisfy : Satisfies (LinearOutputs.rows program.ncChecks) final := by
    rw [nc_checks_match]
    exact ncSlice
  have ncEqualities := LinearOutputs.rows_sound finalCanonical finalOne
    linear_checks_canonical.2.2 ncChecksSatisfy
  have ncValues :
      valuesAt assignment layout.ncEvaluationCols =
        program.expectedFields assignment Program.ncChecks := by
    apply valuesAt_outputs_eq_expected nc_check_outputs
    · intro output member
      apply semantic_known
      simp [TerminalCeCompiler.semanticColumns, member]
    · exact checked.agreement
    · exact ncEqualities
  have ncHolds :
      (decodeSidecar layout assignment).evaluations =
        program.expectedNcEvaluations assignment := by
    unfold decodeSidecar Program.expectedNcEvaluations
    rw [← ncValues]

  unfold TerminalCeCompiler.ClaimHolds
  unfold Nightstream.Protocol.TerminalCE.ClaimHolds
  refine ⟨?_, commitmentHolds, ?_, ?_, ?_, ?_, constantTermsHold, ?_⟩
  · rfl
  · simpa [Program.semantics, TerminalCeCompiler.context,
      TerminalCeCompiler.claim] using publicHolds.symm
  · simpa [Program.semantics, TerminalCeCompiler.context,
      checkNorm_eq_true_iff] using normHolds
  · simp [Program.semantics, TerminalCeCompiler.context,
      TerminalCeCompiler.claim, decodePoint, kValuesAt]
  · simp [Program.semantics, TerminalCeCompiler.context,
      TerminalCeCompiler.claim, decodePoint, kValuesAt]
    simpa using evaluationsHold.symm
  · simp only [Program.semantics, TerminalCeCompiler.context,
      TerminalCeCompiler.claim, Bool.and_eq_true, decide_eq_true_eq,
      true_and]
    constructor
    · simp [decodeSidecar, kValuesAt]
    · simpa [decodeSidecar] using ncHolds

/-- Exact CIR-COMPLETE theorem for one canonical direct-CE claim. Semantic
validity of the independently decoded claim constructs a satisfying assignment
for every emitted row through the checked-program interpreter. -/
theorem canonical_complete
    {state : Nat → Nat}
    (stateCanonical : ∀ column, state column < goldilocksP)
    (one : state 0 = 1)
    (holds : TerminalCeCompiler.ClaimHolds program state) :
    Satisfies rows (program.final state) := by
  let final := program.final state
  have preserves : AgreeOn final state inputColumns :=
    run_preserves_known definitions_wellFormed state
  have finalCanonical : ∀ column, final column < goldilocksP :=
    run_canonical stateCanonical
  have zeroInput : 0 ∈ inputColumns := by native_decide
  have finalOne : final 0 = 1 := (preserves 0 zeroInput).trans one
  have definitionsHold := run_definitions_hold definitions_wellFormed state

  unfold TerminalCeCompiler.ClaimHolds at holds
  unfold Nightstream.Protocol.TerminalCE.ClaimHolds at holds
  rcases holds with ⟨_, commitmentAccepted, publicAccepted, normAccepted,
    _, evaluationsAccepted, constantTermsAccepted, sidecarAccepted⟩

  have commitmentSemantic : program.expectedCommitment state =
      decodeCommitment layout state := by
    simpa [Program.semantics, TerminalCeCompiler.claim] using
      commitmentAccepted
  have publicSemantic : projectedPublic layout state =
      decodePublicInput layout state := by
    simpa [Program.semantics, TerminalCeCompiler.context,
      TerminalCeCompiler.claim] using publicAccepted
  have normState : NormHolds layout state := by
    apply (checkNorm_eq_true_iff layout state).1
    simpa [Program.semantics, TerminalCeCompiler.context,
      TerminalCeCompiler.claim] using normAccepted
  have evaluationsSemantic : program.expectedEvaluations state =
      decodeEvaluations layout state := by
    simpa [Program.semantics, TerminalCeCompiler.context,
      TerminalCeCompiler.claim, decodePoint, kValuesAt] using
      evaluationsAccepted
  have constantTermsSemantic :
      (decodeEvaluations layout state).map
          (fun evaluation => evaluation.headD ProjectionProgram.K.zero) =
        decodeConstantTerms layout state := by
    simpa [Program.semantics, TerminalCeCompiler.claim] using
      constantTermsAccepted
  have ncSemantic : program.expectedNcEvaluations state =
      (decodeSidecar layout state).evaluations := by
    symm
    simpa [Program.semantics, TerminalCeCompiler.context,
      TerminalCeCompiler.claim, decodeSidecar, kValuesAt] using
      sidecarAccepted

  have commitmentValues : valuesAt state layout.commitmentCols =
      program.commitmentChecks.map fun check =>
        ProjectionProgram.residue (check.expected final) := by
    simpa [decodeCommitment, Program.expectedCommitment,
      Program.expectedFields, final] using commitmentSemantic.symm
  have commitmentEqualities := equalities_of_valuesAt_eq_expected
    commitment_check_outputs (known := inputColumns)
    (state := state) (final := final) (by
      intro output member
      apply semantic_columns_input output
      simp [TerminalCeCompiler.semanticColumns, member])
    preserves stateCanonical commitmentValues
  have commitmentRows :
      Satisfies (LinearOutputs.rows program.commitmentChecks) final :=
    LinearOutputs.rows_complete finalCanonical finalOne
      linear_checks_canonical.1 commitmentEqualities
  have commitmentChecks :
      Satisfies (checks program.commitmentInstructions) final := by
    rw [← commitment_checks_match]
    exact commitmentRows

  have publicRowsState :
      Satisfies (LinearOutputs.rows (projectionChecks layout)) state :=
    projection_complete stateCanonical one publicSemantic.symm
  have publicExactState :
      Satisfies (CheckedProgram.rows program.publicInstructions) state := by
    rw [public_program_match]
    exact publicRowsState
  have publicExactFinal :
      Satisfies (CheckedProgram.rows program.publicInstructions) final := by
    intro row member
    apply (rowHolds_agree preserves row
      (public_rows_reference_input row member)).mpr
    exact publicExactState row member
  have publicChecks : Satisfies (checks program.publicInstructions) final :=
    checksSatisfy_of_satisfies publicExactFinal

  have normFinal : NormHolds layout final := by
    intro column member
    rw [preserves column (semantic_columns_input column (by
      simp [TerminalCeCompiler.semanticColumns, member]))]
    exact normState column member
  have normDefinitions : ∀ definition ∈
      definitions (TerminalCeCompiler.normInstructions layout),
      Definition.Holds final definition := by
    intro definition member
    exact definitionsHold definition (norm_definitions_in_program definition member)
  have normChecks : Satisfies (checks program.normInstructionsSlice) final := by
    rw [norm_program_match]
    exact normChecks_complete finalOne normFinal normDefinitions

  have evaluationFields : valuesAt state layout.evaluationCols.flatten =
      program.expectedFields state Program.evaluationChecks := by
    exact (evaluationFields_eq_of_decoded layout_shape
      evaluation_check_outputs evaluationsSemantic).symm
  have evaluationEqualities := equalities_of_valuesAt_eq_expected
    evaluation_check_outputs (known := inputColumns)
    (state := state) (final := final) (by
      intro output member
      apply semantic_columns_input output
      simp [TerminalCeCompiler.semanticColumns, member])
    preserves stateCanonical evaluationFields
  have evaluationRows :
      Satisfies (LinearOutputs.rows program.evaluationChecks) final :=
    LinearOutputs.rows_complete finalCanonical finalOne
      linear_checks_canonical.2.1 evaluationEqualities
  have evaluationChecks :
      Satisfies (checks program.evaluationInstructions) final := by
    rw [← evaluation_checks_match]
    exact evaluationRows

  have constantTermRowsState : Satisfies
      (LinearOutputs.rows (constantTermChecks layout)) state :=
    constantTermChecks_complete layout_shape stateCanonical one
      constantTermsSemantic
  have constantTermExactState :
      Satisfies (CheckedProgram.rows program.constantTermInstructions) state := by
    rw [constant_term_program_match]
    exact constantTermRowsState
  have constantTermExactFinal :
      Satisfies (CheckedProgram.rows program.constantTermInstructions) final := by
    intro row member
    apply (rowHolds_agree preserves row
      (constant_term_rows_reference_input row member)).mpr
    exact constantTermExactState row member
  have constantTermChecks :
      Satisfies (checks program.constantTermInstructions) final :=
    checksSatisfy_of_satisfies constantTermExactFinal

  have ncFields : valuesAt state layout.ncEvaluationCols =
      program.expectedFields state Program.ncChecks := by
    exact (ncFields_eq_of_decoded layout_shape nc_check_outputs
      ncSemantic).symm
  have ncEqualities := equalities_of_valuesAt_eq_expected nc_check_outputs
    (known := inputColumns) (state := state) (final := final) (by
      intro output member
      apply semantic_columns_input output
      simp [TerminalCeCompiler.semanticColumns, member])
    preserves stateCanonical ncFields
  have ncRows : Satisfies (LinearOutputs.rows program.ncChecks) final :=
    LinearOutputs.rows_complete finalCanonical finalOne
      linear_checks_canonical.2.2 ncEqualities
  have ncChecks : Satisfies (checks program.ncInstructions) final := by
    rw [← nc_checks_match]
    exact ncRows

  have allChecks : ChecksHold state instructions := by
    change Satisfies (checks instructions) final
    rw [phase_partition]
    simpa [checks] using satisfies_append commitmentChecks
      (satisfies_append publicChecks
        (satisfies_append normChecks
          (satisfies_append evaluationChecks
            (satisfies_append constantTermChecks ncChecks))))
  exact CheckedProgram.complete definitions_wellFormed definitions_canonical
    stateCanonical zeroInput one allChecks

private theorem mapped_sound
    (prime : EuclidPrime goldilocksP)
    (columnMap : List Nat)
    (mapsOne : Relabel.column columnMap 0 = 0)
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfies : Satisfies (rows.map (Relabel.row columnMap)) assignment) :
    TerminalCeCompiler.ClaimHolds program
      (Relabel.assignment columnMap assignment) := by
  apply canonical_sound prime (Relabel.canonical canonical)
    (Relabel.constantOne mapsOne one)
  exact (Relabel.satisfies_mapped_iff rows columnMap assignment).mp satisfies

/-- Every one of the 14 direct terminal-CE children satisfies the independent
claim predicate after decoding through its exact injective column map. -/
def AllClaimsHold (assignment : Nat → Nat) : Prop :=
  ∀ columnMap ∈ columnMaps,
    TerminalCeCompiler.ClaimHolds program
      (Relabel.assignment columnMap assignment)

theorem all_claims_sound
    (prime : EuclidPrime goldilocksP)
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfies : Satisfies terminalCeRows assignment) :
    AllClaimsHold assignment := by
  intro columnMap columnMapMember
  apply mapped_sound prime columnMap
    (column_maps_one columnMap columnMapMember) canonical one
  intro row rowMember
  apply satisfies row
  unfold terminalCeRows
  apply List.mem_flatten.mpr
  refine ⟨rows.map (Relabel.row columnMap), ?_, rowMember⟩
  unfold claimRows
  exact List.mem_map.mpr ⟨columnMap, columnMapMember, rfl⟩

/-- Independent compiler execution for one mapped terminal-CE child.  The
source assignment contains the semantic claim inputs.  `program.final` is the
deterministic checked-program interpreter, and `output` identifies that
interpreter result with the mapped slice of the full-history assignment.  No
R1CS satisfaction proposition occurs in this witness. -/
structure ClaimExecution
    (columnMap : List Nat) (assignment : Nat → Nat) where
  source : Nat → Nat
  sourceCanonical : ∀ column, source column < goldilocksP
  sourceOne : source 0 = 1
  claim : TerminalCeCompiler.ClaimHolds program source
  output : program.final source = Relabel.assignment columnMap assignment

/-- One native/compiler execution constructs every row of its exact relabeled
terminal-CE child. -/
theorem ClaimExecution.compiles
    {columnMap : List Nat} {assignment : Nat → Nat}
    (execution : ClaimExecution columnMap assignment) :
    Satisfies (rows.map (Relabel.row columnMap)) assignment := by
  apply (Relabel.satisfies_mapped_iff rows columnMap assignment).mpr
  rw [← execution.output]
  exact canonical_complete execution.sourceCanonical execution.sourceOne
    execution.claim

/-- Native witness-generation contract for all fourteen direct terminal-CE
children in production order. -/
def CompilerWitness (assignment : Nat → Nat) : Type :=
  ∀ columnMap ∈ columnMaps, ClaimExecution columnMap assignment

/-- Exact CIR-COMPLETE theorem for the complete 301,588-row terminal-CE
family.  Each child is generated by the same canonical compiler from an
independently valid decoded claim. -/
theorem all_claims_complete
    {assignment : Nat → Nat}
    (witness : CompilerWitness assignment) :
    Satisfies terminalCeRows assignment := by
  intro row rowMember
  unfold terminalCeRows at rowMember
  rcases List.mem_flatten.mp rowMember with ⟨mappedRows, mappedRowsMember,
    rowMember⟩
  unfold claimRows at mappedRowsMember
  rcases List.mem_map.mp mappedRowsMember with ⟨columnMap, columnMapMember,
    rfl⟩
  exact (witness columnMap columnMapMember).compiles row rowMember

end Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalCeSound
