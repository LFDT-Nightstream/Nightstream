import Nightstream.Implementation.Nebula.Production.FPrime.Lifetime.FoldCoreManifestFor
import Nightstream.Implementation.Nebula.Production.FPrime.Terminal.OpeningRowsFor
import Nightstream.Implementation.Nebula.Production.FPrime.Terminal.PublicRowsFor
import Nightstream.Implementation.Nebula.Production.FPrime.Terminal.StatementRowsFor

/-!
Contract: exact numeric row manifest for the terminal paper fold and close.

The numeric manifest contains the common fold core, one unconditional close
row, all 177 public-statement recomposition rows, and all 178 terminal
public-result link rows. A separate typed-fold bridge embeds those exact rows
into the same assignment used by all fourteen terminal opening and CE-core
row families. This manifest verifies and consumes the trailing complete fresh
claim, checks
its delayed memory batch, computes the complete paper-NIFS output, requires
the final memory carry to be closed, and binds the closed result to the exact
verifier-owned public bit image. It has no continuation or successor rows.

`Result.exactInvocation` accepts the typed terminal row family, not a
`ProductOpening`. The opening and public-result equality are both derived.

Assurance tier: exponent-indexed terminal-fold row implementation.

Does not own byte-to-bit parsing, a compact backend, generated-artifact
containment, Rust, or cryptography.
-/

set_option autoImplicit false
set_option maxRecDepth 100000
set_option maxHeartbeats 5000000

namespace Nightstream.Implementation.Nebula.ProductionPaperTerminalFoldManifestFor

open Nightstream.Implementation.Nebula
open Nightstream.Implementation.R1CS
open Nightstream.Protocol.Nebula
open Nightstream.Protocol.Nebula.FPrime
open Nightstream.Protocol.Nebula.ProductionProfileCandidates
open Nightstream.Protocol.Nebula.WasmStatement
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint

abbrev FullShape
    (rowVariables logicalWidth : Nat)
    (publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth) :=
  ProductPaperAlgebraFor.FullShape rowVariables logicalWidth publicFits

structure Program (candidate : Id) (rowVariables : Nat) where
  fold : ProductionPaperFoldCoreManifestFor.Program candidate rowVariables
  statementLayout : ProductionPaperTerminalStatementRowsFor.Layout

def Program.closingLayout
    {candidate : Id} {rowVariables : Nat}
    (program : Program candidate rowVariables) :
    TerminalClosedCarryRows.Layout :=
  ProductionPaperTerminalInvocationRowsSoundFor.closingLayout candidate
    rowVariables program.fold.priorLayout

def Program.rows
    {candidate : Id} {rowVariables : Nat}
    (program : Program candidate rowVariables) : List Row :=
  program.fold.rows ++
    TerminalClosedCarryRows.rows program.closingLayout ++
      ProductionPaperTerminalStatementRowsFor.rows program.statementLayout ++
        ProductionPaperTerminalPublicRowsFor.rows candidate
          program.fold.priorLayout program.statementLayout.statement

def rowCount (candidate : Id) (rowVariables : Nat) : Nat :=
  ProductionPaperFoldCoreManifestFor.rowCount candidate rowVariables + 1 +
    177 + 178

theorem Program.rows_length_exact
    {candidate : Id} {rowVariables : Nat}
    (program : Program candidate rowVariables) :
    program.rows.length = rowCount candidate rowVariables := by
  simp [Program.rows, rowCount,
    ProductionPaperFoldCoreManifestFor.Program.rows_length_exact,
    TerminalClosedCarryRows.rows_length,
    ProductionPaperTerminalStatementRowsFor.rows_length_exact,
    ProductionPaperTerminalPublicRowsFor.rows_length_exact]

theorem Program.fold_satisfied
    {candidate : Id} {rowVariables : Nat}
    {program : Program candidate rowVariables} {assignment : Nat -> Nat}
    (satisfied : Satisfies program.rows assignment) :
    Satisfies program.fold.rows assignment := by
  intro row member
  exact satisfied row (by simp [Program.rows, member])

theorem Program.closing_satisfied
    {candidate : Id} {rowVariables : Nat}
    {program : Program candidate rowVariables} {assignment : Nat -> Nat}
    (satisfied : Satisfies program.rows assignment) :
    Satisfies (TerminalClosedCarryRows.rows program.closingLayout)
      assignment := by
  intro row member
  exact satisfied row (by simp [Program.rows, member])

theorem Program.public_satisfied
    {candidate : Id} {rowVariables : Nat}
    {program : Program candidate rowVariables} {assignment : Nat -> Nat}
    (satisfied : Satisfies program.rows assignment) :
    Satisfies
      (ProductionPaperTerminalPublicRowsFor.rows candidate
        program.fold.priorLayout program.statementLayout.statement)
      assignment := by
  intro row member
  exact satisfied row (by simp [Program.rows, member])

theorem Program.statement_satisfied
    {candidate : Id} {rowVariables : Nat}
    {program : Program candidate rowVariables} {assignment : Nat -> Nat}
    (satisfied : Satisfies program.rows assignment) :
    Satisfies
      (ProductionPaperTerminalStatementRowsFor.rows program.statementLayout)
      assignment := by
  intro row member
  exact satisfied row (by simp [Program.rows, member])

structure Result
    {ProgramType : Type} (candidate : Id)
    {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    (statementId : ProductConcreteNifsFor.StatementId)
    (config : ProductPaperAlgebraFor.Config rowVariables logicalWidth publicFits
      operationsShape snapshotShape)
    (artifact : ProductConcreteNifsFor.RelationArtifact rowVariables
      logicalWidth publicFits)
    (program : Program candidate rowVariables)
    (assignment : Nat -> Nat) (headers : ChainHeaders Digest.Value)
    (priorPrefix : ProductionPaperPriorStateAuthorityRowsFor.Prefix candidate
      (FullShape rowVariables logicalWidth publicFits))
    (value : ProductionFieldNativeFullClaim.Value candidate
      (FullShape rowVariables logicalWidth publicFits))
    (proof : ProductionProductPiCcsTypedBridgeFor.ExactProof rowVariables)
    (recursive : ProductionPaperRecursiveRelationRowsSoundFor.Result candidate
      statementId config artifact program.fold.priorLayout assignment headers
      priorPrefix value proof)
    (statement : ProductionStatement ProgramType) :
    Prop where
  assignmentCanonical : forall column, assignment column < goldilocksP
  one : assignment 0 = 1
  finalPhase :
    (ProductionPaperTerminalInvocationRowsSoundFor.finalWire
      recursive.memoryResult).phase = .closed
  finalSemantic : recursive.memoryResult.semantic
      (Fin.last (ProductionMemoryCheckedBatchRows.StepCount candidate)) =
    .closed (ProductionPaperTerminalInvocationRowsSoundFor.finalClosed
      recursive.memoryResult)
  publicResult :
    ProductionPaperTerminalInvocationRowsSoundFor.PublicChecks candidate
      statement recursive.priorState
      (ProductionPaperTerminalInvocationRowsSoundFor.finalClosed
        recursive.memoryResult)

theorem Program.rows_imply_result
    {ProgramType : Type} (candidate : Id)
    {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    (statementId : ProductConcreteNifsFor.StatementId)
    (config : ProductPaperAlgebraFor.Config rowVariables logicalWidth publicFits
      operationsShape snapshotShape)
    (artifact : ProductConcreteNifsFor.RelationArtifact rowVariables
      logicalWidth publicFits)
    (program : Program candidate rowVariables)
    (headers : ChainHeaders Digest.Value)
    (priorPrefix : ProductionPaperPriorStateAuthorityRowsFor.Prefix candidate
      (FullShape rowVariables logicalWidth publicFits))
    (assignment : Nat -> Nat)
    (priorPrefixPlaced : ProductionPaperPriorStateAuthorityRowsFor.PrefixPlaced
      (ProductPaperAlgebraFor.fullShapeContract rowVariables logicalWidth
        publicFits) program.fold.priorLayout assignment priorPrefix)
    (statement : ProductionStatement ProgramType)
    (value : ProductionFieldNativeFullClaim.Value candidate
      (FullShape rowVariables logicalWidth publicFits))
    (valueCanonical : value.Canonical)
    (proof : ProductionProductPiCcsTypedBridgeFor.ExactProof rowVariables)
    (baseWires : ProductionProductPiCcsTypedBridgeFor.Wires rowVariables)
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (prefixCanonical : ProductionFullClaimNifsPublicCarrierFor.PrefixCanonical
      candidate (FullShape rowVariables logicalWidth publicFits) 9)
    (headersPlaced : ProductionMemoryCheckedBatchRows.HeadersPlaced
      program.fold.priorLayout.ccs.core.batch.frame.memory assignment headers)
    (carryHeadersPlaced : MemoryCarryRows.HeadersPlaced
      program.fold.priorLayout.carry.carry assignment headers)
    (placement : ProductionPaperRecursiveRelationRowsSoundFor.Placement
      candidate statementId config artifact program.fold.priorLayout value
      proof baseWires program.fold.samplerBase
      program.fold.piRlcAlgebraLayout program.fold.piDecLayout assignment
      canonical)
    (matched : program.fold.MatchesRows statementId config artifact value
      (ProductionPaperRecursiveRelationRowsSoundFor.boundWires
        (logicalWidth := logicalWidth) (publicFits := publicFits) candidate
        rowVariables program.fold.priorLayout.ccs.carrier baseWires)
      program.fold.samplerBase program.fold.piRlcAlgebraLayout
      program.fold.piDecLayout program.fold.nifsOutputLayout
      program.fold.priorLayout)
    (publicImage : WasmPublicStatementEncoding.PublicImage)
    (decoded : publicImage.DecodesFor
      (ProductionProfileCandidates.identity candidate) statement)
    (publicBitsPlaced : ProductionPaperTerminalStatementRowsFor.BitsPlaced
      program.statementLayout assignment publicImage)
    (satisfied : Satisfies program.rows assignment) :
    exists recursive : ProductionPaperRecursiveRelationRowsSoundFor.Result
        candidate statementId config artifact program.fold.priorLayout
        assignment headers priorPrefix value proof,
      recursive.nifsOutputLayout = program.fold.nifsOutputLayout /\
        recursive.compactManifest = program.fold.seedManifest /\
        Result candidate statementId config artifact program assignment headers
          priorPrefix value proof recursive statement := by
  have foldRows := program.fold_satisfied satisfied
  have rowsHold := program.fold.rows_imply_recursive_rowsHold
    (proof := proof) matched foldRows
  rcases
      ProductionPaperRecursiveRelationRowsSoundFor.rows_imply_verified_exact_claim_and_memory_transition
        candidate statementId config artifact program.fold.priorLayout
        program.fold.priorValid headers priorPrefix assignment
        priorPrefixPlaced statement value valueCanonical proof baseWires
        program.fold.samplerBase program.fold.piRlcAlgebraLayout
        program.fold.piDecLayout program.fold.nifsOutputLayout
        program.fold.seedManifest program.fold.compactLayout canonical one
        prefixCanonical headersPlaced carryHeadersPlaced rowsHold placement with
    ⟨recursive, _nifsOutputLayoutExact, compactManifestExact,
      _samplerSucceeded, _piCcsAccepted, _piDecAccepted, _verifierOutputExact⟩
  have closingRows := program.closing_satisfied satisfied
  have statementRows := program.statement_satisfied satisfied
  have publicRows := program.public_satisfied satisfied
  have phaseClosed :
      (ProductionPaperTerminalInvocationRowsSoundFor.finalWire
        recursive.memoryResult).phase = .closed := by
    exact TerminalClosedCarryRows.parsed_phase_closed canonical one
      (recursive.memoryResult.boundaryParsed
        (Fin.last (ProductionMemoryCheckedBatchRows.StepCount candidate)))
      closingRows
  have semanticClosed : recursive.memoryResult.semantic
      (Fin.last (ProductionMemoryCheckedBatchRows.StepCount candidate)) =
      .closed (ProductionPaperTerminalInvocationRowsSoundFor.finalClosed
        recursive.memoryResult) := by
    rw [recursive.memoryResult.semanticExact]
    have rawPhase :
        (recursive.memoryResult.boundary
          (Fin.last (ProductionMemoryCheckedBatchRows.StepCount candidate))).phase =
          .closed := by
      simpa [ProductionPaperTerminalInvocationRowsSoundFor.finalWire] using
        phaseClosed
    unfold MemoryCarryParser.semanticCarry
    rw [rawPhase]
    rfl
  have statementPlaced :=
    ProductionPaperTerminalStatementRowsFor.rows_imply_statementPlaced
      decoded canonical one publicBitsPlaced statementRows
  have publicResult :=
    ProductionPaperTerminalPublicRowsFor.rows_imply_publicChecks candidate
      recursive statement decoded program.statementLayout.statement
      statementPlaced canonical one publicRows
  exact ⟨recursive, _nifsOutputLayoutExact, compactManifestExact,
    { assignmentCanonical := canonical
      one := one
      finalPhase := phaseClosed
      finalSemantic := semanticClosed
      publicResult := publicResult }⟩

namespace Result

/-- Add the same-assignment terminal opening and CE-core rows. The opening is
derived from the verified trailing NIFS output carrier; it is not supplied as
a premise. The exact terminal invocation contains no successor or fresh-claim
producer. -/
theorem exactInvocation
    {ProgramType : Type}
    {candidate : Id} {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    {statementId : ProductConcreteNifsFor.StatementId}
    {config : ProductPaperAlgebraFor.Config rowVariables logicalWidth publicFits
      operationsShape snapshotShape}
    {artifact : ProductConcreteNifsFor.RelationArtifact rowVariables
      logicalWidth publicFits}
    {manifest : SeedSchedule.Manifest}
    {program : Program candidate rowVariables}
    {assignment :
      Nightstream.Implementation.Lowering.Goldilocks.ColumnId -> F}
    {headers : ChainHeaders Digest.Value}
    {priorPrefix : ProductionPaperPriorStateAuthorityRowsFor.Prefix candidate
      (FullShape rowVariables logicalWidth publicFits)}
    {value : ProductionFieldNativeFullClaim.Value candidate
      (FullShape rowVariables logicalWidth publicFits)}
    {proof : ProductionProductPiCcsTypedBridgeFor.ExactProof rowVariables}
    {recursive : ProductionPaperRecursiveRelationRowsSoundFor.Result candidate
      statementId config artifact program.fold.priorLayout
      (ProductionPaperTerminalOpeningRowsFor.pulledAssignment assignment)
      headers priorPrefix value proof}
    (statement : ProductionStatement ProgramType)
    (result : Result candidate statementId config artifact program
      (ProductionPaperTerminalOpeningRowsFor.pulledAssignment assignment)
      headers priorPrefix value proof recursive statement)
    (family : ProductionPaperTerminalOpeningRowsFor.Family
      (manifest := manifest) statementId config artifact
      program.fold.priorLayout headers priorPrefix value proof assignment
      recursive) :
    ProductionPaperTerminalInvocationRowsSoundFor.ExactInvocation candidate
      statementId config artifact program.fold.priorLayout
      (ProductionPaperTerminalOpeningRowsFor.pulledAssignment assignment)
      headers priorPrefix value proof recursive
      (family.opening result.one) statement := by
  have transitionClosed : ProductionBatchedFPrime.Transition
      (ProductionPaperRecursiveRelationRowsSoundFor.paperVerifier candidate
        statementId config artifact)
      MemoryProductBalanceRows.ConcreteBalanced
      (recursive.memoryResult.semantic 0) recursive.verified
      (.closed (ProductionPaperTerminalInvocationRowsSoundFor.finalClosed
        recursive.memoryResult)) := by
    rw [← result.finalSemantic]
    exact recursive.transition
  exact
    { assignmentCanonical := result.assignmentCanonical
      one := result.one
      trailingVerified := recursive.verified.accepted
      trailingClaimExact := recursive.claimExact
      trailingProofExact := recursive.proofExact
      finalFoldOutput := rfl
      finalPhase := result.finalPhase
      finalSemantic := result.finalSemantic
      consumesTrailing := transitionClosed
      childrenHold := family.holds result.one
      publicResult := result.publicResult }

end Result

def Program.RowsIncluded
    {candidate : Id} {rowVariables : Nat}
    (program : Program candidate rowVariables) (finalRows : List Row) : Prop :=
  program.rows.Sublist finalRows

theorem Program.satisfies_of_rowsIncluded
    {candidate : Id} {rowVariables : Nat}
    {program : Program candidate rowVariables} {finalRows : List Row}
    {assignment : Nat -> Nat}
    (included : program.RowsIncluded finalRows)
    (satisfied : Satisfies finalRows assignment) :
    Satisfies program.rows assignment := by
  intro row member
  exact satisfied row (included.subset member)

end Nightstream.Implementation.Nebula.ProductionPaperTerminalFoldManifestFor
