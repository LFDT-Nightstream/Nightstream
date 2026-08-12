import Nightstream.Implementation.R1CS.Correspondence.Gadgets.TerminalCeSound
import Nightstream.Implementation.R1CS.Canonical.GoldilocksField

/-!
Contract: whole-program soundness for the direct terminal-CE compiler.

The structural certificate below contains only schedule, row-membership,
canonical-coefficient, and column-availability facts. It contains no terminal
acceptance result and no component of `TerminalCE.ClaimHolds`.

Exact row satisfaction derives every terminal claim obligation: public width,
commitment output, public projection, strict norm, point shape, complete
evaluation output, constant terms, and the NC sidecar.

This module does not prove that one generated Rust program has this structural
certificate, or that `Program.expectedEvaluations` refines the Phi81 product
relation. Those are separate artifact and algorithm-refinement obligations.

Assurance tier: model-level compiler soundness.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.TerminalCeProgramSound

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Program
open Nightstream.Implementation.R1CS.CheckedProgram
open Nightstream.Implementation.R1CS.TerminalCeCompiler
open Nightstream.Implementation.R1CS.TerminalCeSound
open Nightstream.Implementation.R1CS.ProjectionProgram

/-- Structural obligations for one emitted direct terminal-CE program.

Every row condition is an actual membership fact about the emitted program.
The certificate cannot state that the terminal claim is valid. -/
structure Structural (program : TerminalCeCompiler.Program) : Prop where
  shape : ShapeValid program.layout
  definitionsWellFormed :
    WellFormed program.inputColumns program.definitions
  definitionsCanonical :
    ∀ definition ∈ program.definitions, definition.Canonical
  checksReference :
    ChecksReference
      (knownAfter program.inputColumns program.definitions)
      program.instructions
  constantOneInput : 0 ∈ program.inputColumns
  commitmentOutputs :
    program.commitmentChecks.map LinearOutputs.Check.output =
      program.layout.commitmentCols
  evaluationOutputs :
    program.evaluationChecks.map LinearOutputs.Check.output =
      program.layout.evaluationCols.flatten
  ncOutputs :
    program.ncChecks.map LinearOutputs.Check.output =
      program.layout.ncEvaluationCols
  commitmentCanonical : LinearOutputs.Canonical program.commitmentChecks
  evaluationCanonical : LinearOutputs.Canonical program.evaluationChecks
  ncCanonical : LinearOutputs.Canonical program.ncChecks
  commitmentRowsChecked : ∀ row,
    row ∈ LinearOutputs.rows program.commitmentChecks →
      row ∈ checks program.instructions
  projectionRowsIncluded : ∀ row,
    row ∈ LinearOutputs.rows (projectionChecks program.layout) →
      row ∈ CheckedProgram.rows program.instructions
  normRowsIncluded : ∀ row,
    row ∈ CheckedProgram.rows
        (TerminalCeCompiler.normInstructions program.layout) →
      row ∈ CheckedProgram.rows program.instructions
  evaluationRowsChecked : ∀ row,
    row ∈ LinearOutputs.rows program.evaluationChecks →
      row ∈ checks program.instructions
  constantTermRowsIncluded : ∀ row,
    row ∈ LinearOutputs.rows (constantTermChecks program.layout) →
      row ∈ CheckedProgram.rows program.instructions
  ncRowsChecked : ∀ row,
    row ∈ LinearOutputs.rows program.ncChecks →
      row ∈ checks program.instructions
  commitmentOutputsKnown : ∀ column,
    column ∈ program.layout.commitmentCols →
      column ∈ knownAfter program.inputColumns program.definitions
  evaluationOutputsKnown : ∀ column,
    column ∈ program.layout.evaluationCols.flatten →
      column ∈ knownAfter program.inputColumns program.definitions
  ncOutputsKnown : ∀ column,
    column ∈ program.layout.ncEvaluationCols →
      column ∈ knownAfter program.inputColumns program.definitions

private theorem satisfies_subrows
    {small large : List Row} {assignment : Nat → Nat}
    (included : ∀ row, row ∈ small → row ∈ large)
    (satisfied : Satisfies large assignment) :
    Satisfies small assignment := by
  intro row member
  exact satisfied row (included row member)

private theorem knownAfter_of_input
    {inputColumns : List Nat} {definitions : List Definition}
    {column : Nat} (member : column ∈ inputColumns) :
    column ∈ knownAfter inputColumns definitions := by
  induction definitions generalizing inputColumns with
  | nil => exact member
  | cons definition definitions inductionHypothesis =>
      exact inductionHypothesis (by simp [member])

/-- Field-level output equality reconstructs the exact nested evaluation
array. Even row lengths are not needed in this direction because no pairing
operation must be inverted. -/
theorem decodedEvaluations_eq_expected_of_fields
    {program : TerminalCeCompiler.Program} {assignment : Nat → Nat}
    (equal : valuesAt assignment program.layout.evaluationCols.flatten =
      program.expectedFields assignment Program.evaluationChecks) :
    decodeEvaluations program.layout assignment =
      program.expectedEvaluations assignment := by
  unfold decodeEvaluations Program.expectedEvaluations
  have split := splitByLengths_map_flatten
    program.layout.evaluationCols (fieldAt assignment)
  have fields :
      program.layout.evaluationCols.flatten.map (fieldAt assignment) =
        program.expectedFields assignment Program.evaluationChecks := by
    simpa [valuesAt] using equal
  rw [← fields]
  simpa [valuesAt, List.map_map, Function.comp_def] using
    congrArg (List.map pairs) split.symm

/-- Field-level output equality reconstructs the exact NC evaluation list. -/
theorem decodedNc_eq_expected_of_fields
    {program : TerminalCeCompiler.Program} {assignment : Nat → Nat}
    (equal : valuesAt assignment program.layout.ncEvaluationCols =
      program.expectedFields assignment Program.ncChecks) :
    (decodeSidecar program.layout assignment).evaluations =
      program.expectedNcEvaluations assignment := by
  simpa [decodeSidecar, Program.expectedNcEvaluations] using
    congrArg pairs equal

/-- Exact direct-terminal rows derive the complete independent terminal-CE
claim predicate. No acceptance Boolean or semantic validity premise occurs in
the theorem. -/
theorem rows_sound
    {program : TerminalCeCompiler.Program} {assignment : Nat → Nat}
    (structural : Structural program)
    (assignmentCanonical : ∀ column, assignment column < goldilocksP)
    (constantOne : assignment 0 = 1)
    (satisfied : Satisfies
      (CheckedProgram.rows program.instructions) assignment) :
    TerminalCeCompiler.ClaimHolds program assignment := by
  let final := program.final assignment
  have compilerSound : CheckedProgram.SoundResult program.inputColumns
      program.instructions assignment assignment := by
    apply CheckedProgram.sound structural.definitionsWellFormed
      structural.definitionsCanonical structural.checksReference
      (fun _ _ => rfl) assignmentCanonical constantOne satisfied
  have finalCanonical : ∀ column, final column < goldilocksP := by
    exact Program.run_canonical assignmentCanonical
  have oneKnown : 0 ∈
      knownAfter program.inputColumns program.definitions :=
    knownAfter_of_input structural.constantOneInput
  have finalOne : final 0 = 1 := by
    exact (compilerSound.agreement 0 oneKnown).trans constantOne

  have commitmentSatisfied :
      Satisfies (LinearOutputs.rows program.commitmentChecks) final :=
    satisfies_subrows structural.commitmentRowsChecked
      compilerSound.checksHold
  have commitmentEqualities := LinearOutputs.rows_sound finalCanonical
    finalOne structural.commitmentCanonical commitmentSatisfied
  have commitmentFields :
      valuesAt assignment program.layout.commitmentCols =
        program.expectedFields assignment Program.commitmentChecks := by
    exact valuesAt_outputs_eq_expected structural.commitmentOutputs
      structural.commitmentOutputsKnown compilerSound.agreement
      commitmentEqualities
  have commitmentExact : program.expectedCommitment assignment =
      decodeCommitment program.layout assignment := by
    simpa [Program.expectedCommitment, decodeCommitment] using
      commitmentFields.symm

  have projectionSatisfied :
      Satisfies (LinearOutputs.rows (projectionChecks program.layout))
        assignment :=
    satisfies_subrows structural.projectionRowsIncluded satisfied
  have projectionExact :
      decodePublicInput program.layout assignment =
        projectedPublic program.layout assignment :=
    projection_sound assignmentCanonical constantOne projectionSatisfied

  have normSatisfied : Satisfies
      (CheckedProgram.rows
        (TerminalCeCompiler.normInstructions program.layout)) assignment :=
    satisfies_subrows structural.normRowsIncluded satisfied
  have normExact : NormHolds program.layout assignment :=
    normInstructions_sound
      Nightstream.Implementation.R1CS.Canonical.GoldilocksField.goldilocks_euclidPrime
      assignmentCanonical constantOne normSatisfied
  have normAccepted : checkNorm program.layout assignment = true :=
    (checkNorm_eq_true_iff program.layout assignment).2 normExact

  have evaluationSatisfied :
      Satisfies (LinearOutputs.rows program.evaluationChecks) final :=
    satisfies_subrows structural.evaluationRowsChecked
      compilerSound.checksHold
  have evaluationEqualities := LinearOutputs.rows_sound finalCanonical
    finalOne structural.evaluationCanonical evaluationSatisfied
  have evaluationFields :
      valuesAt assignment program.layout.evaluationCols.flatten =
        program.expectedFields assignment Program.evaluationChecks := by
    exact valuesAt_outputs_eq_expected structural.evaluationOutputs
      structural.evaluationOutputsKnown compilerSound.agreement
      evaluationEqualities
  have evaluationsExact :
      program.expectedEvaluations assignment =
        decodeEvaluations program.layout assignment := by
    exact (decodedEvaluations_eq_expected_of_fields evaluationFields).symm

  have constantTermSatisfied : Satisfies
      (LinearOutputs.rows (constantTermChecks program.layout)) assignment :=
    satisfies_subrows structural.constantTermRowsIncluded satisfied
  have constantTermsExact :
      (decodeEvaluations program.layout assignment).map
          (fun evaluation => evaluation.headD K.zero) =
        decodeConstantTerms program.layout assignment :=
    constantTermChecks_sound structural.shape assignmentCanonical constantOne
      constantTermSatisfied

  have ncSatisfied : Satisfies
      (LinearOutputs.rows program.ncChecks) final :=
    satisfies_subrows structural.ncRowsChecked compilerSound.checksHold
  have ncEqualities := LinearOutputs.rows_sound finalCanonical finalOne
    structural.ncCanonical ncSatisfied
  have ncFields : valuesAt assignment program.layout.ncEvaluationCols =
      program.expectedFields assignment Program.ncChecks := by
    exact valuesAt_outputs_eq_expected structural.ncOutputs
      structural.ncOutputsKnown compilerSound.agreement ncEqualities
  have ncExact : program.expectedNcEvaluations assignment =
      (decodeSidecar program.layout assignment).evaluations := by
    exact (decodedNc_eq_expected_of_fields ncFields).symm

  unfold TerminalCeCompiler.ClaimHolds
  unfold Nightstream.Protocol.TerminalCE.ClaimHolds
  refine ⟨?_, commitmentExact, ?_, ?_, ?_, ?_, constantTermsExact, ?_⟩
  · cases expected : program.layout.expectedPublicWidth with
    | none =>
        simp [TerminalCeCompiler.context,
          Nightstream.Protocol.TerminalCE.PublicWidthHolds,
          expected]
    | some width =>
        simpa [TerminalCeCompiler.context,
          Nightstream.Protocol.TerminalCE.PublicWidthHolds,
          TerminalCeCompiler.claim, expected] using
          structural.shape.publicWidthPinned
  · simp only [Program.semantics, TerminalCeCompiler.claim,
      TerminalCeCompiler.context]
    simpa using congrArg some projectionExact.symm
  · simp [Program.semantics, TerminalCeCompiler.context, normAccepted]
  · simp [Program.semantics, TerminalCeCompiler.context,
      TerminalCeCompiler.claim, decodePoint, kValuesAt]
  · simp only [Program.semantics, TerminalCeCompiler.context,
      TerminalCeCompiler.claim]
    have pointLength :
        (decodePoint program.layout assignment).length =
          program.layout.pointCols.length := by
      simp [decodePoint, kValuesAt]
    simp [pointLength, evaluationsExact]
  · simp only [Program.semantics, TerminalCeCompiler.context,
      TerminalCeCompiler.claim]
    simp [decodeSidecar, kValuesAt, ncExact]

end Nightstream.Implementation.R1CS.TerminalCeProgramSound
