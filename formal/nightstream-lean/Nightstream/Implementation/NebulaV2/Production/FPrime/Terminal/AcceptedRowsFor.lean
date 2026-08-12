import Nightstream.Implementation.NebulaV2.Production.FPrime.Terminal.CompleteProgramFor

/-!
Contract: row-accepted terminal package for the production paper F-prime
lifetime.

The package contains the complete numeric terminal-fold manifest inputs and
one typed Goldilocks assignment. `Rows.recursive` and `Rows.result` are chosen
only from `Program.rows_imply_result`. `Accepted` also fixes one complete typed
terminal program. Its exact rows derive all fourteen terminal openings from
the same assignment.

No `ProductOpening`, recursive result, public check, closed carry, or exact
terminal invocation is an input. This file does not own generated-artifact
containment, byte parsing, a compact terminal backend, Rust refinement, or
cryptographic reductions.

Assurance tier: exponent-indexed terminal-row composition.

Emits constraints: no.
-/

set_option autoImplicit false
set_option maxRecDepth 100000

namespace Nightstream.Implementation.NebulaV2.ProductionPaperTerminalAcceptedRowsFor

open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.Implementation.NebulaV2
open Nightstream.Implementation.R1CS
open Nightstream.Protocol.NebulaV2
open Nightstream.Protocol.NebulaV2.FPrime
open Nightstream.Protocol.NebulaV2.ProductionProfileCandidates
open Nightstream.Protocol.NebulaV2.WasmStatement
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint

abbrev FullShape
    (rowVariables logicalWidth : Nat)
    (publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth) :=
  ProductPaperAlgebraFor.FullShape rowVariables logicalWidth publicFits

/-- All premises of the complete terminal-manifest soundness theorem. The
typed assignment is the sole source of numeric terminal columns. -/
structure Rows
    {ProgramType : Type} (candidate : Id)
    {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    (statementId : ProductConcreteNifsFor.StatementId)
    (config : ProductPaperAlgebraFor.Config rowVariables logicalWidth publicFits
      operationsShape snapshotShape)
    (artifact : ProductConcreteNifsFor.RelationArtifact rowVariables
      logicalWidth publicFits)
    (headers : ChainHeaders Digest.Value)
    (statement : ProductionStatement ProgramType)
    (value : ProductionFieldNativeFullClaim.Value candidate
      (FullShape rowVariables logicalWidth publicFits))
    (proof : ProductionProductPiCcsTypedBridgeFor.ExactProof rowVariables)
    (foldProgram : ProductionPaperTerminalFoldManifestFor.Program candidate
      rowVariables) where
  assignment : ColumnId -> F
  priorPrefix : ProductionPaperPriorStateAuthorityRowsFor.Prefix candidate
    (FullShape rowVariables logicalWidth publicFits)
  baseWires : ProductionProductPiCcsTypedBridgeFor.Wires rowVariables
  priorPrefixPlaced :
    ProductionPaperPriorStateAuthorityRowsFor.PrefixPlaced
      (ProductPaperAlgebraFor.fullShapeContract rowVariables logicalWidth
        publicFits)
      foldProgram.fold.priorLayout
      (ProductionPaperTerminalOpeningRowsFor.pulledAssignment assignment)
      priorPrefix
  valueCanonical : value.Canonical
  one : ProductionPaperTerminalOpeningRowsFor.pulledAssignment assignment 0 = 1
  prefixCanonical :
    ProductionFullClaimNifsPublicCarrierFor.PrefixCanonical candidate
      (FullShape rowVariables logicalWidth publicFits) 9
  headersPlaced : ProductionMemoryCheckedBatchRows.HeadersPlaced
    foldProgram.fold.priorLayout.ccs.core.batch.frame.memory
    (ProductionPaperTerminalOpeningRowsFor.pulledAssignment assignment) headers
  carryHeadersPlaced : MemoryCarryRows.HeadersPlaced
    foldProgram.fold.priorLayout.carry.carry
    (ProductionPaperTerminalOpeningRowsFor.pulledAssignment assignment) headers
  placement : ProductionPaperRecursiveRelationRowsSoundFor.Placement candidate
    statementId config artifact foldProgram.fold.priorLayout value proof
    baseWires foldProgram.fold.samplerBase foldProgram.fold.piRlcAlgebraLayout
    foldProgram.fold.piDecLayout
    (ProductionPaperTerminalOpeningRowsFor.pulledAssignment assignment)
    (ProductionPaperTerminalOpeningRowsFor.pulledAssignment_canonical assignment)
  matched : foldProgram.fold.MatchesRows statementId config artifact value
    (ProductionPaperRecursiveRelationRowsSoundFor.boundWires
      (logicalWidth := logicalWidth) (publicFits := publicFits) candidate
      rowVariables foldProgram.fold.priorLayout.ccs.carrier baseWires)
    foldProgram.fold.samplerBase foldProgram.fold.piRlcAlgebraLayout
    foldProgram.fold.piDecLayout foldProgram.fold.nifsOutputLayout
    foldProgram.fold.priorLayout
  publicImage : WasmPublicStatementEncoding.PublicImage
  decoded : publicImage.DecodesFor
    (ProductionProfileCandidates.identity candidate) statement
  publicBitsPlaced : ProductionPaperTerminalStatementRowsFor.BitsPlaced
    foldProgram.statementLayout
    (ProductionPaperTerminalOpeningRowsFor.pulledAssignment assignment)
    publicImage
  satisfied : Satisfies foldProgram.rows
    (ProductionPaperTerminalOpeningRowsFor.pulledAssignment assignment)

namespace Rows

/-- The terminal numeric rows produce a recursive result and all close/public
facts. The result is existential because the lower row theorem owns its
construction. -/
theorem existsResult
    {ProgramType : Type} {candidate : Id}
    {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    {statementId : ProductConcreteNifsFor.StatementId}
    {config : ProductPaperAlgebraFor.Config rowVariables logicalWidth publicFits
      operationsShape snapshotShape}
    {artifact : ProductConcreteNifsFor.RelationArtifact rowVariables
      logicalWidth publicFits}
    {headers : ChainHeaders Digest.Value}
    {statement : ProductionStatement ProgramType}
    {value : ProductionFieldNativeFullClaim.Value candidate
      (FullShape rowVariables logicalWidth publicFits)}
    {proof : ProductionProductPiCcsTypedBridgeFor.ExactProof rowVariables}
    {foldProgram : ProductionPaperTerminalFoldManifestFor.Program candidate
      rowVariables}
    (rows : Rows candidate statementId config artifact headers statement value
      proof foldProgram) :
    exists recursive : ProductionPaperRecursiveRelationRowsSoundFor.Result
        candidate statementId config artifact foldProgram.fold.priorLayout
        (ProductionPaperTerminalOpeningRowsFor.pulledAssignment rows.assignment)
        headers rows.priorPrefix value proof,
      recursive.nifsOutputLayout = foldProgram.fold.nifsOutputLayout /\
        recursive.compactManifest = foldProgram.fold.seedManifest /\
        ProductionPaperTerminalFoldManifestFor.Result candidate statementId
          config artifact foldProgram
          (ProductionPaperTerminalOpeningRowsFor.pulledAssignment rows.assignment)
          headers rows.priorPrefix value proof recursive statement := by
  exact foldProgram.rows_imply_result candidate statementId config artifact
    headers rows.priorPrefix
    (ProductionPaperTerminalOpeningRowsFor.pulledAssignment rows.assignment)
    rows.priorPrefixPlaced statement value rows.valueCanonical proof
    rows.baseWires
    (ProductionPaperTerminalOpeningRowsFor.pulledAssignment_canonical
      rows.assignment)
    rows.one rows.prefixCanonical rows.headersPlaced rows.carryHeadersPlaced
    rows.placement rows.matched rows.publicImage rows.decoded
    rows.publicBitsPlaced rows.satisfied

/-- The exact recursive result selected from the row theorem. It is not a
field that a caller can manufacture. -/
noncomputable def recursive
    {ProgramType : Type} {candidate : Id}
    {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    {statementId : ProductConcreteNifsFor.StatementId}
    {config : ProductPaperAlgebraFor.Config rowVariables logicalWidth publicFits
      operationsShape snapshotShape}
    {artifact : ProductConcreteNifsFor.RelationArtifact rowVariables
      logicalWidth publicFits}
    {headers : ChainHeaders Digest.Value}
    {statement : ProductionStatement ProgramType}
    {value : ProductionFieldNativeFullClaim.Value candidate
      (FullShape rowVariables logicalWidth publicFits)}
    {proof : ProductionProductPiCcsTypedBridgeFor.ExactProof rowVariables}
    {foldProgram : ProductionPaperTerminalFoldManifestFor.Program candidate
      rowVariables}
    (rows : Rows candidate statementId config artifact headers statement value
      proof foldProgram) :
    ProductionPaperRecursiveRelationRowsSoundFor.Result candidate statementId
      config artifact foldProgram.fold.priorLayout
      (ProductionPaperTerminalOpeningRowsFor.pulledAssignment rows.assignment)
      headers rows.priorPrefix value proof :=
  Classical.choose rows.existsResult

/-- Close and public facts for the selected recursive result. -/
theorem result
    {ProgramType : Type} {candidate : Id}
    {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    {statementId : ProductConcreteNifsFor.StatementId}
    {config : ProductPaperAlgebraFor.Config rowVariables logicalWidth publicFits
      operationsShape snapshotShape}
    {artifact : ProductConcreteNifsFor.RelationArtifact rowVariables
      logicalWidth publicFits}
    {headers : ChainHeaders Digest.Value}
    {statement : ProductionStatement ProgramType}
    {value : ProductionFieldNativeFullClaim.Value candidate
      (FullShape rowVariables logicalWidth publicFits)}
    {proof : ProductionProductPiCcsTypedBridgeFor.ExactProof rowVariables}
    {foldProgram : ProductionPaperTerminalFoldManifestFor.Program candidate
      rowVariables}
    (rows : Rows candidate statementId config artifact headers statement value
      proof foldProgram) :
    ProductionPaperTerminalFoldManifestFor.Result candidate statementId config
    artifact foldProgram
      (ProductionPaperTerminalOpeningRowsFor.pulledAssignment rows.assignment)
      headers rows.priorPrefix value proof rows.recursive statement :=
  (Classical.choose_spec rows.existsResult).2.2

/-- The recursive result uses the exact final-running carrier layout fixed by
the terminal fold program. This equality is retained from the row theorem; it
is not a terminal-opening premise. -/
theorem recursiveNifsOutputLayoutExact
    {ProgramType : Type} {candidate : Id}
    {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    {statementId : ProductConcreteNifsFor.StatementId}
    {config : ProductPaperAlgebraFor.Config rowVariables logicalWidth publicFits
      operationsShape snapshotShape}
    {artifact : ProductConcreteNifsFor.RelationArtifact rowVariables
      logicalWidth publicFits}
    {headers : ChainHeaders Digest.Value}
    {statement : ProductionStatement ProgramType}
    {value : ProductionFieldNativeFullClaim.Value candidate
      (FullShape rowVariables logicalWidth publicFits)}
    {proof : ProductionProductPiCcsTypedBridgeFor.ExactProof rowVariables}
    {foldProgram : ProductionPaperTerminalFoldManifestFor.Program candidate
      rowVariables}
    (rows : Rows candidate statementId config artifact headers statement value
      proof foldProgram) :
    rows.recursive.nifsOutputLayout =
      foldProgram.fold.nifsOutputLayout :=
  (Classical.choose_spec rows.existsResult).1

/-- The exact terminal fold rows select their own compact-chain seed
manifest. A terminal node cannot supply this equality. -/
theorem recursiveCompactManifestExact
    {ProgramType : Type} {candidate : Id}
    {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    {statementId : ProductConcreteNifsFor.StatementId}
    {config : ProductPaperAlgebraFor.Config rowVariables logicalWidth publicFits
      operationsShape snapshotShape}
    {artifact : ProductConcreteNifsFor.RelationArtifact rowVariables
      logicalWidth publicFits}
    {headers : ChainHeaders Digest.Value}
    {statement : ProductionStatement ProgramType}
    {value : ProductionFieldNativeFullClaim.Value candidate
      (FullShape rowVariables logicalWidth publicFits)}
    {proof : ProductionProductPiCcsTypedBridgeFor.ExactProof rowVariables}
    {foldProgram : ProductionPaperTerminalFoldManifestFor.Program candidate
      rowVariables}
    (rows : Rows candidate statementId config artifact headers statement value
      proof foldProgram) :
    rows.recursive.compactManifest = foldProgram.fold.seedManifest :=
  (Classical.choose_spec rows.existsResult).2.1

end Rows

/-- Complete terminal acceptance from the numeric fold manifest and the
same-assignment typed opening/core rows. -/
structure Accepted
    {ProgramType : Type} (candidate : Id)
    {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    (statementId : ProductConcreteNifsFor.StatementId)
    (config : ProductPaperAlgebraFor.Config rowVariables logicalWidth publicFits
      operationsShape snapshotShape)
    (artifact : ProductConcreteNifsFor.RelationArtifact rowVariables
      logicalWidth publicFits)
    (headers : ChainHeaders Digest.Value)
    (statement : ProductionStatement ProgramType)
    (value : ProductionFieldNativeFullClaim.Value candidate
      (FullShape rowVariables logicalWidth publicFits))
    (proof : ProductionProductPiCcsTypedBridgeFor.ExactProof rowVariables)
    (foldProgram : ProductionPaperTerminalFoldManifestFor.Program candidate
      rowVariables)
    (program : ProductionPaperTerminalCompleteProgramFor.Program candidate config
      foldProgram) where
  rows : Rows candidate statementId config artifact headers statement value proof
    foldProgram
  programRows : program.RowsSatisfied artifact.system rows.assignment

namespace Accepted

/-- The complete verifier-selected typed program derives the opening family.
The family is not an acceptance input. -/
noncomputable def family
    {ProgramType : Type} {candidate : Id}
    {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    {statementId : ProductConcreteNifsFor.StatementId}
    {config : ProductPaperAlgebraFor.Config rowVariables logicalWidth publicFits
      operationsShape snapshotShape}
    {artifact : ProductConcreteNifsFor.RelationArtifact rowVariables
      logicalWidth publicFits}
    {headers : ChainHeaders Digest.Value}
    {statement : ProductionStatement ProgramType}
    {value : ProductionFieldNativeFullClaim.Value candidate
      (FullShape rowVariables logicalWidth publicFits)}
    {proof : ProductionProductPiCcsTypedBridgeFor.ExactProof rowVariables}
    {foldProgram : ProductionPaperTerminalFoldManifestFor.Program candidate
      rowVariables}
    {program : ProductionPaperTerminalCompleteProgramFor.Program candidate config
      foldProgram}
    (accepted : Accepted candidate statementId config artifact headers statement
      value proof foldProgram program) :
    ProductionPaperTerminalOpeningRowsFor.Family
      (manifest := foldProgram.fold.seedManifest)
      statementId config artifact foldProgram.fold.priorLayout
      headers accepted.rows.priorPrefix value proof accepted.rows.assignment
      accepted.rows.recursive := by
  exact program.family accepted.rows.recursiveNifsOutputLayoutExact
    accepted.programRows

noncomputable def opening
    {ProgramType : Type} {candidate : Id}
    {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    {statementId : ProductConcreteNifsFor.StatementId}
    {config : ProductPaperAlgebraFor.Config rowVariables logicalWidth publicFits
      operationsShape snapshotShape}
    {artifact : ProductConcreteNifsFor.RelationArtifact rowVariables
      logicalWidth publicFits}
    {headers : ChainHeaders Digest.Value}
    {statement : ProductionStatement ProgramType}
    {value : ProductionFieldNativeFullClaim.Value candidate
      (FullShape rowVariables logicalWidth publicFits)}
    {proof : ProductionProductPiCcsTypedBridgeFor.ExactProof rowVariables}
    {foldProgram : ProductionPaperTerminalFoldManifestFor.Program candidate
      rowVariables}
    {program : ProductionPaperTerminalCompleteProgramFor.Program candidate config
      foldProgram}
    (accepted : Accepted candidate statementId config artifact headers statement
      value proof foldProgram program) :
    ProductionPaperTerminalInvocationRowsSoundFor.ProductOpening candidate
      statementId config artifact value proof :=
  accepted.family.opening accepted.rows.result.one

/-- Exact trailing-claim consumption, close, public checks, and all fourteen
same-assignment child openings, derived from rows. -/
theorem exactInvocation
    {ProgramType : Type} {candidate : Id}
    {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    {statementId : ProductConcreteNifsFor.StatementId}
    {config : ProductPaperAlgebraFor.Config rowVariables logicalWidth publicFits
      operationsShape snapshotShape}
    {artifact : ProductConcreteNifsFor.RelationArtifact rowVariables
      logicalWidth publicFits}
    {headers : ChainHeaders Digest.Value}
    {statement : ProductionStatement ProgramType}
    {value : ProductionFieldNativeFullClaim.Value candidate
      (FullShape rowVariables logicalWidth publicFits)}
    {proof : ProductionProductPiCcsTypedBridgeFor.ExactProof rowVariables}
    {foldProgram : ProductionPaperTerminalFoldManifestFor.Program candidate
      rowVariables}
    {program : ProductionPaperTerminalCompleteProgramFor.Program candidate config
      foldProgram}
    (accepted : Accepted candidate statementId config artifact headers statement
      value proof foldProgram program) :
    ProductionPaperTerminalInvocationRowsSoundFor.ExactInvocation candidate
      statementId config artifact foldProgram.fold.priorLayout
      (ProductionPaperTerminalOpeningRowsFor.pulledAssignment
        accepted.rows.assignment)
      headers accepted.rows.priorPrefix value proof accepted.rows.recursive
      accepted.opening statement :=
  accepted.rows.result.exactInvocation statement accepted.family

end Accepted

end Nightstream.Implementation.NebulaV2.ProductionPaperTerminalAcceptedRowsFor
