import Nightstream.Implementation.NebulaV2.NIFS.Running.Codec
import Nightstream.Implementation.NebulaV2.Production.FPrime.Terminal.FoldManifestFor
import Nightstream.Implementation.NebulaV2.Production.FPrime.Recursive.RecursiveCoreManifestFor
import Nightstream.Implementation.R1CS.Core.Semantics
import Nightstream.Protocol.NebulaV2.ProductionProfileCandidates

/-!
Contract: one dimension authority for a generated field-native augmented
relation, its SuperNeo NIFS key, and its terminal verifier.

HyperNova encodes each augmented function `F'` as the fresh relation that the
next invocation folds. Therefore the generated augmented rows, the source
compiler, and the NIFS cube must use one exact row exponent. A separate
terminal circuit may use another exponent, but it must open the relation with
the original exact exponent.

This file rejects the current accidental reuse of the 25-variable bit-serial
reference shape by every field-native candidate. The fixed-25 lower bound
proves only that the selected exponent exceeds 25. A separate field requires
the exponent-indexed core at the selected exponent to occur in the generated
row list. This file does not select a final exponent. Selection requires the
complete generated row list and fit proof.

The digest fields below are identifiers only. They do not prove the row or
shape equalities.

Assurance tier: generated-artifact release gate.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.NebulaV2.ProductionRelationDimensions

open Nightstream.Implementation.NebulaV2
open Nightstream.Implementation.R1CS
open Nightstream.Protocol.NebulaV2
open Nightstream.Protocol.NebulaV2.ProductionProfileCandidates

/-! ## Finite terminal column scope -/

/-- Every column mentioned by one terminal linear combination lies in the
finite terminal assignment. -/
def TerminalTermsBelow (columns : Nat) (terms : List (Nat × Nat)) : Prop :=
  forall term, term ∈ terms -> term.1 < columns

/-- All three linear combinations of one terminal row use only allocated
terminal columns. -/
def TerminalRowBelow (columns : Nat) (row : Row) : Prop :=
  TerminalTermsBelow columns row.a /\
    TerminalTermsBelow columns row.b /\
    TerminalTermsBelow columns row.c

/-- Complete finite-column scope for a generated terminal row list. -/
def TerminalRowsBelow (columns : Nat) (rows : List Row) : Prop :=
  forall row, row ∈ rows -> TerminalRowBelow columns row

/-- Exact mandatory-core count at the historical fixed-25 exponent. This is
computed from the same row census as the selected-exponent manifest. It is a
lower bound for a complete generated relation, not a production census. -/
def referenceCoreLowerBound (candidate : Id) : Nat :=
  ProductionRecursiveCoreGeometryFor.knownCoreRows candidate 25

theorem referenceCoreLowerBound_exceeds_25 (candidate : Id) :
    2 ^ 25 < referenceCoreLowerBound candidate := by
  cases candidate with
  | e1 => rw [referenceCoreLowerBound,
      ProductionRecursiveCoreGeometryFor.knownCoreRows_25_table.1]; decide
  | e4 => rw [referenceCoreLowerBound,
      ProductionRecursiveCoreGeometryFor.knownCoreRows_25_table.2.1]; decide
  | e8 => rw [referenceCoreLowerBound,
      ProductionRecursiveCoreGeometryFor.knownCoreRows_25_table.2.2.1]; decide
  | e16 => rw [referenceCoreLowerBound,
      ProductionRecursiveCoreGeometryFor.knownCoreRows_25_table.2.2.2]; decide

/-- Exact dimensions that one generated field-native verifier key must own.
No field states a soundness, acceptance, extraction, or execution result. -/
structure Artifact (candidate : Id) where
  profile : Profile.Identity
  profileExact : profile = ProductionProfileCandidates.identity candidate
  verifierKeyDigest : Digest.Value
  relationManifestDigest : Digest.Value
  terminalManifestDigest : Digest.Value
  relationRowVariables : Nat
  recursiveRows : List Row
  referenceCoreLowerBoundIncluded :
    referenceCoreLowerBound candidate <= recursiveRows.length
  coreProgram :
    ProductionRecursiveCoreManifestFor.Program candidate relationRowVariables
  exponentIndexedCoreIncluded : coreProgram.RowsIncluded recursiveRows
  recursiveRowsFit : recursiveRows.length <= 2 ^ relationRowVariables
  sourceCompilerRowVariables : Nat
  sourceCompilerExact : sourceCompilerRowVariables = relationRowVariables
  nifsShape :
    Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.Shape
  nifsShapeExact : nifsShape = ProductNifsCodec.shapeFor relationRowVariables
  runningFieldCoordinates : Nat
  runningFieldCoordinatesExact :
    runningFieldCoordinates =
      ProductionProfileCandidates.runningFieldCoordinatesFor
        relationRowVariables
  fieldNativeEnvelopeCoordinates : Nat
  fieldNativeEnvelopeCoordinatesExact :
    fieldNativeEnvelopeCoordinates =
      ProductionProfileCandidates.fieldNativeEnvelopeCoordinatesFor candidate
        relationRowVariables
  nifsPublicFrameFields : Nat
  nifsPublicFrameFieldsExact :
    nifsPublicFrameFields = 17 + runningFieldCoordinates + 3888 + 540
  terminalOpenedRelationRowVariables : Nat
  terminalOpensExactRelation :
    terminalOpenedRelationRowVariables = relationRowVariables
  terminalStatementLayout : ProductionPaperTerminalStatementRowsFor.Layout
  terminalCircuitRowVariables : Nat
  terminalAssignmentWidth : Nat
  terminalRows : List Row
  terminalColumnsBelow : TerminalRowsBelow terminalAssignmentWidth terminalRows
  terminalProgramRowsIncluded :
    ({ fold := coreProgram.fold
       statementLayout := terminalStatementLayout } :
      ProductionPaperTerminalFoldManifestFor.Program candidate
        relationRowVariables).RowsIncluded terminalRows
  terminalRowsFit : terminalRows.length <= 2 ^ terminalCircuitRowVariables
  terminalCarrierFits :
    terminalAssignmentWidth <= 2 ^ terminalCircuitRowVariables

namespace Artifact

/-- The terminal verifier uses the exact common fold owned by the recursive
relation artifact. It cannot select a second fold program. -/
def terminalProgram
    {candidate : Id} (artifact : Artifact candidate) :
    ProductionPaperTerminalFoldManifestFor.Program candidate
      artifact.relationRowVariables where
  fold := artifact.coreProgram.fold
  statementLayout := artifact.terminalStatementLayout

theorem terminalProgram_fold
    {candidate : Id} (artifact : Artifact candidate) :
    artifact.terminalProgram.fold = artifact.coreProgram.fold := rfl

/-- The known field-native recursive core alone rules out every relation
cube with at most 25 variables. -/
theorem relationRowVariables_exceed_25
    {candidate : Id} (artifact : Artifact candidate) :
    25 < artifact.relationRowVariables := by
  by_contra notGreater
  have exponentBound : artifact.relationRowVariables <= 25 :=
    Nat.le_of_not_gt notGreater
  have cubeBound : 2 ^ artifact.relationRowVariables <= 2 ^ 25 :=
    Nat.pow_le_pow_right (by decide) exponentBound
  have coreFits :
      referenceCoreLowerBound candidate <= 2 ^ artifact.relationRowVariables :=
    artifact.referenceCoreLowerBoundIncluded.trans artifact.recursiveRowsFit
  have coreTooLarge := referenceCoreLowerBound_exceeds_25 candidate
  omega

/-- Every current field-native candidate needs at least 26 relation row
variables before any omitted generated family is added. -/
theorem relationRowVariables_minimum
    {candidate : Id} (artifact : Artifact candidate) :
    26 <= artifact.relationRowVariables := by
  have exceeds := artifact.relationRowVariables_exceed_25
  omega

/-- The generated manifest contains the mandatory recursive core at the same
exponent used by the relation, compiler, NIFS shape, and terminal opening. -/
theorem selected_exponent_core_included
    {candidate : Id} (artifact : Artifact candidate) :
    artifact.coreProgram.RowsIncluded artifact.recursiveRows :=
  artifact.exponentIndexedCoreIncluded

/-- Actual ordered core-row inclusion implies the independent numeric census.
The converse is false. -/
theorem selected_exponent_core_count_fits
    {candidate : Id} (artifact : Artifact candidate) :
    ProductionRecursiveCoreGeometryFor.knownCoreRows candidate
        artifact.relationRowVariables <=
      artifact.recursiveRows.length :=
  artifact.coreProgram.length_le_of_rowsIncluded
    artifact.exponentIndexedCoreIncluded

/-- Satisfying the generated relation satisfies the exact mandatory core.
This is the semantic consequence that a row-count test could not provide. -/
theorem selected_exponent_core_satisfied
    {candidate : Id} (artifact : Artifact candidate)
    {assignment : Nat -> Nat}
    (satisfied : Satisfies artifact.recursiveRows assignment) :
    Satisfies artifact.coreProgram.rows assignment :=
  artifact.coreProgram.satisfies_of_rowsIncluded
    artifact.exponentIndexedCoreIncluded satisfied

/-- The generated terminal manifest contains the exact terminal program,
including the same common fold that the recursive relation owns. -/
theorem terminal_program_included
    {candidate : Id} (artifact : Artifact candidate) :
    artifact.terminalProgram.RowsIncluded artifact.terminalRows :=
  artifact.terminalProgramRowsIncluded

/-- Ordered terminal-program containment gives the independent terminal row
count lower bound. A terminal capacity check alone does not give this fact. -/
theorem terminal_program_count_fits
    {candidate : Id} (artifact : Artifact candidate) :
    ProductionPaperTerminalFoldManifestFor.rowCount candidate
        artifact.relationRowVariables <= artifact.terminalRows.length := by
  rw [← artifact.terminalProgram.rows_length_exact]
  exact artifact.terminalProgramRowsIncluded.length_le

/-- Satisfaction of the generated terminal relation implies satisfaction of
the exact terminal program that shares the recursive relation's fold. -/
theorem terminal_program_satisfied
    {candidate : Id} (artifact : Artifact candidate)
    {assignment : Nat -> Nat}
    (satisfied : Satisfies artifact.terminalRows assignment) :
    Satisfies artifact.terminalProgram.rows assignment :=
  artifact.terminalProgram.satisfies_of_rowsIncluded
    artifact.terminalProgramRowsIncluded satisfied

/-- The generated terminal verifier has one finite Boolean cube that covers
both every terminal row and every referenced assignment column. This is the
terminal analogue of the padded-identity rectangular capacity condition. -/
theorem terminal_rows_and_columns_fit
    {candidate : Id} (artifact : Artifact candidate) :
    artifact.terminalRows.length <=
        2 ^ artifact.terminalCircuitRowVariables /\
      artifact.terminalAssignmentWidth <=
        2 ^ artifact.terminalCircuitRowVariables :=
  ⟨artifact.terminalRowsFit, artifact.terminalCarrierFits⟩

/-- Every generated terminal row refers only to its finite assignment. -/
theorem terminal_columns_scoped
    {candidate : Id} (artifact : Artifact candidate) :
    TerminalRowsBelow artifact.terminalAssignmentWidth
      artifact.terminalRows :=
  artifact.terminalColumnsBelow

/-- The source compiler and NIFS verifier use the same exponent derived from
the complete generated relation. -/
theorem source_and_nifs_exact
    {candidate : Id} (artifact : Artifact candidate) :
    artifact.sourceCompilerRowVariables = artifact.relationRowVariables /\
      artifact.nifsShape.cubeVariables = artifact.relationRowVariables := by
  constructor
  · exact artifact.sourceCompilerExact
  · rw [artifact.nifsShapeExact]
    rfl

/-- The terminal circuit may have its own capacity, but the relation that it
opens has the exact augmented-relation exponent. -/
theorem terminal_opening_exact
    {candidate : Id} (artifact : Artifact candidate) :
    artifact.terminalOpenedRelationRowVariables =
      artifact.nifsShape.cubeVariables := by
  rw [artifact.terminalOpensExactRelation, artifact.nifsShapeExact]
  rfl

/-- No current field-native generated artifact can reuse the fixed
25-variable reference NIFS shape. -/
theorem nifsShape_ne_reference25
    {candidate : Id} (artifact : Artifact candidate) :
    artifact.nifsShape ≠ ProductNifsCodec.shape := by
  intro equal
  have exponentEqual : artifact.relationRowVariables = 25 := by
    have shapeEqual :
        ProductNifsCodec.shapeFor artifact.relationRowVariables =
          ProductNifsCodec.shape :=
      artifact.nifsShapeExact.symm.trans equal
    exact congrArg
      Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.Shape.cubeVariables
      shapeEqual
  have minimum := artifact.relationRowVariables_minimum
  omega

/-- The production running carrier cannot retain the fixed-25 count of
83,210 fields. -/
theorem runningFieldCoordinates_exceed_reference
    {candidate : Id} (artifact : Artifact candidate) :
    83210 < artifact.runningFieldCoordinates := by
  rw [artifact.runningFieldCoordinatesExact,
    ProductionProfileCandidates.runningFieldCoordinatesFor]
  have minimum := artifact.relationRowVariables_minimum
  omega

/-- The production NIFS public frame cannot retain the fixed-25 count of
87,655 fields. -/
theorem nifsPublicFrameFields_exceed_reference
    {candidate : Id} (artifact : Artifact candidate) :
    87655 < artifact.nifsPublicFrameFields := by
  rw [artifact.nifsPublicFrameFieldsExact]
  have running := artifact.runningFieldCoordinates_exceed_reference
  omega

end Artifact

/-! ## Necessity countermodel -/

/-- The old terminal gate compared only row counts. It did not establish that
the verifier checked the required terminal equations. -/
def TerminalLengthOnly (required finalRows : List Row) : Prop :=
  required.length <= finalRows.length

/-- A length-only terminal gate accepts replacement of a mandatory rejecting
row by a zero row. Ordered terminal-program containment rejects it. -/
theorem terminalLengthOnly_accepts_zero_row_substitution :
    TerminalLengthOnly
        [ProductionRecursiveCoreManifestFor.rejectingConstantRow]
        [ProductionRecursiveCoreManifestFor.zeroRow] /\
      ¬ [ProductionRecursiveCoreManifestFor.rejectingConstantRow].Sublist
        [ProductionRecursiveCoreManifestFor.zeroRow] := by
  simp [TerminalLengthOnly,
    ProductionRecursiveCoreManifestFor.rejectingConstantRow,
    ProductionRecursiveCoreManifestFor.zeroRow]

/-- An unsafe split check that lets the recursive circuit and folded relation
use unrelated exponents. -/
def SplitExponentFit
    (recursiveRows recursiveCircuitRowVariables nifsRowVariables : Nat) : Prop :=
  recursiveRows <= 2 ^ recursiveCircuitRowVariables /\
    nifsRowVariables = 25

/-- The fixed-25 E=1 lower-bound core fits a 26-variable recursive circuit
while the unsafe split check still selects a 25-variable NIFS relation. This
is the exact mismatch that one shared dimension authority prevents. -/
theorem splitExponentFit_accepts_incompatible_e1 :
    SplitExponentFit (referenceCoreLowerBound .e1) 26 25 := by
  rw [referenceCoreLowerBound,
    ProductionRecursiveCoreGeometryFor.knownCoreRows_25_table.1]
  norm_num [SplitExponentFit]

/-- A row-only terminal capacity check omits the assignment-width bound. -/
def TerminalRowCapacityOnly
    (rowCount rowVariables : Nat) : Prop :=
  rowCount <= 2 ^ rowVariables

/-- Complete numeric terminal capacity check. -/
def TerminalRectangularCapacity
    (rowCount assignmentWidth rowVariables : Nat) : Prop :=
  rowCount <= 2 ^ rowVariables /\
    assignmentWidth <= 2 ^ rowVariables

/-- One terminal row fits a one-variable row cube while a three-column
assignment does not. Row capacity alone cannot certify a finite terminal
artifact. -/
theorem terminalRowCapacityOnly_does_not_imply_rectangularCapacity :
    TerminalRowCapacityOnly 1 1 /\
      ¬ TerminalRectangularCapacity 1 3 1 := by
  norm_num [TerminalRowCapacityOnly, TerminalRectangularCapacity]

end Nightstream.Implementation.NebulaV2.ProductionRelationDimensions
