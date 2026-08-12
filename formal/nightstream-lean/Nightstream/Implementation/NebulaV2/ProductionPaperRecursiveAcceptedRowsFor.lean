import Nightstream.Implementation.NebulaV2.ProductionRecursiveCoreManifestFor
import Nightstream.Implementation.NebulaV2.ProductionPaperRecursiveInvocationRowsSoundFor

/-!
Contract: row-derived core for one production recursive F-prime invocation.

One satisfying recursive manifest derives the exact prior-claim verification,
delayed memory transition, NIFS output, successor, and 28-field memory
challenge authority.  The challenge authority is not an input.

Application lowering and fresh-claim construction remain explicit compiler
boundaries. The current producer memory batch is derived from rows in the
same fixed recursive manifest. This module does not claim that the remaining
generated rows exist.

Assurance tier: exponent-indexed recursive-row composition.

Emits constraints: no new rows.
-/

set_option autoImplicit false
set_option maxRecDepth 500000
set_option maxHeartbeats 5000000

namespace Nightstream.Implementation.NebulaV2.ProductionPaperRecursiveAcceptedRowsFor

open Nightstream.Implementation.NebulaV2
open Nightstream.Implementation.R1CS
open Nightstream.Protocol.NebulaV2
open Nightstream.Protocol.NebulaV2.ApplicationBatch
open Nightstream.Protocol.NebulaV2.FPrime
open Nightstream.Protocol.NebulaV2.ProductionProfileCandidates
open Nightstream.Protocol.NebulaV2.WasmState
open Nightstream.Protocol.NebulaV2.WasmStatement
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint

abbrev FullShape
    (rowVariables logicalWidth : Nat)
    (publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth) :=
  ProductPaperAlgebraFor.FullShape rowVariables logicalWidth publicFits

/-- Inputs and placements for one complete recursive core manifest. Numeric
row authority comes from `assignment` and `satisfied`. `statementCanonical`
is the explicit external parser boundary; it is not inferred from the fresh
claim. -/
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
    (proof : ProductionProductPiCcsTypedBridgeFor.ExactProof rowVariables) where
  program : ProductionRecursiveCoreManifestFor.Program candidate rowVariables
  assignment : Nat -> Nat
  priorPrefix : ProductionPaperPriorStateAuthorityRowsFor.Prefix candidate
    (FullShape rowVariables logicalWidth publicFits)
  baseWires : ProductionProductPiCcsTypedBridgeFor.Wires rowVariables
  assignmentCanonical : forall column, assignment column < goldilocksP
  one : assignment 0 = 1
  priorPrefixPlaced :
    ProductionPaperPriorStateAuthorityRowsFor.PrefixPlaced
      (ProductPaperAlgebraFor.fullShapeContract rowVariables logicalWidth
        publicFits) program.fold.priorLayout assignment priorPrefix
  valueCanonical : value.Canonical
  statementCanonical :
    (WasmPublicStatementEncoding.PublicImage.ofStatement statement).DecodesFor
      (identity candidate) statement
  /-- The five static memory-challenge digests in the generated recursive
  program belong to this exact decoded statement. Equality of only the
  verifier-key record does not bind the application relation, program, or
  memory plan. -/
  statementIdentityExact :
    program.statementIdentity = statement.base.identity
  prefixCanonical :
    ProductionFullClaimNifsPublicCarrierFor.PrefixCanonical candidate
      (FullShape rowVariables logicalWidth publicFits) 9
  headersPlaced : ProductionMemoryCheckedBatchRows.HeadersPlaced
    program.fold.priorLayout.ccs.core.batch.frame.memory assignment headers
  carryHeadersPlaced : MemoryCarryRows.HeadersPlaced
    program.fold.priorLayout.carry.carry assignment headers
  placement : ProductionPaperRecursiveRelationRowsSoundFor.Placement candidate
    statementId config artifact program.fold.priorLayout value proof baseWires
    program.fold.samplerBase program.fold.piRlcAlgebraLayout
    program.fold.piDecLayout assignment assignmentCanonical
  matched : program.MatchesRecursiveRows statementId config artifact value
    (ProductionPaperRecursiveRelationRowsSoundFor.boundWires
      (logicalWidth := logicalWidth) (publicFits := publicFits) candidate
      rowVariables program.fold.priorLayout.ccs.carrier baseWires)
    program.fold.samplerBase program.fold.piRlcAlgebraLayout
    program.fold.piDecLayout program.fold.nifsOutputLayout
    program.fold.priorLayout
  satisfied : Satisfies program.rows assignment

namespace Rows

/-- The static memory-challenge authority used by the recursive rows is the
identity of the exact decoded public statement. -/
theorem challengeStatementIdentityExact
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
    (rows : Rows candidate statementId config artifact headers statement value
      proof) :
    rows.program.statementIdentity = statement.base.identity :=
  rows.statementIdentityExact

theorem existsRecursive
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
    (rows : Rows candidate statementId config artifact headers statement value
      proof) :
    exists recursive : ProductionPaperRecursiveRelationRowsSoundFor.Result
        candidate statementId config artifact rows.program.fold.priorLayout
        rows.assignment headers rows.priorPrefix value proof,
      recursive.nifsOutputLayout = rows.program.fold.nifsOutputLayout /\
        recursive.compactManifest = rows.program.fold.seedManifest := by
  have rowBundle := rows.program.rows_imply_recursive_rowsHold (proof := proof)
    rows.matched rows.satisfied
  rcases
      ProductionPaperRecursiveRelationRowsSoundFor.rows_imply_verified_exact_claim_and_memory_transition
        candidate statementId config artifact rows.program.fold.priorLayout
        rows.program.fold.priorValid headers rows.priorPrefix rows.assignment
        rows.priorPrefixPlaced statement value rows.valueCanonical proof
        rows.baseWires rows.program.fold.samplerBase
        rows.program.fold.piRlcAlgebraLayout rows.program.fold.piDecLayout
        rows.program.fold.nifsOutputLayout rows.program.fold.seedManifest
        rows.program.fold.compactLayout rows.assignmentCanonical rows.one
        rows.prefixCanonical rows.headersPlaced rows.carryHeadersPlaced
        rowBundle rows.placement with
    ⟨recursive, nifsOutputLayoutExact, compactManifestExact, _⟩
  exact ⟨recursive, nifsOutputLayoutExact, compactManifestExact⟩

/-- The recursive result selected by the row theorem.  A caller cannot provide
this value as a field. -/
@[irreducible] noncomputable def recursive
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
    (rows : Rows candidate statementId config artifact headers statement value
      proof) :
    ProductionPaperRecursiveRelationRowsSoundFor.Result candidate statementId
      config artifact rows.program.fold.priorLayout rows.assignment headers
      rows.priorPrefix value proof :=
  Exists.choose rows.existsRecursive

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
    (rows : Rows candidate statementId config artifact headers statement value
      proof) :
    rows.recursive.nifsOutputLayout = rows.program.fold.nifsOutputLayout :=
  by
    rw [recursive]
    exact (Exists.choose_spec rows.existsRecursive).1

/-- The compact-chain manifest is selected by the exact recursive row
program. It is not a caller-supplied equality. -/
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
    (rows : Rows candidate statementId config artifact headers statement value
      proof) :
    rows.recursive.compactManifest = rows.program.fold.seedManifest := by
  rw [recursive]
  exact (Exists.choose_spec rows.existsRecursive).2

/-- The recursive manifest and the semantic verifier use one statement ID.
This is static manifest equality, not transcript or verifier acceptance. -/
theorem statementIdExact
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
    (rows : Rows candidate statementId config artifact headers statement value
      proof) :
    rows.program.fold.statementId = statementId :=
  rows.matched.foldMatches.statementIdExact

/-- The recursive successor reads the exact NIFS output columns that the
verified prior-claim fold produced.  This is a manifest fact and needs no
application witness. -/
theorem nifsOutputAlias
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
    (rows : Rows candidate statementId config artifact headers statement value
      proof)
    (index : Fin (ProductNifsCodec.runningFieldCountFor rowVariables)) :
    rows.program.successorLayout.nifsOutputColumn index =
      rows.recursive.nifsOutputLayout.carrierColumn index := by
  rw [rows.recursiveNifsOutputLayoutExact]
  exact rows.program.successorNifsOutputAlias index

/-- Every current-batch boundary uses the chain headers already placed in the
verified prior carry. Header placement is not a producer witness. -/
theorem currentMemoryHeadersPlaced
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
    (rows : Rows candidate statementId config artifact headers statement value
      proof) :
    ProductionMemoryCheckedBatchRows.HeadersPlaced
      rows.program.currentMemoryLayout rows.assignment headers := by
  intro index role lane
  rw [rows.program.currentMemoryHeadersFromPrior index role lane]
  exact rows.carryHeadersPlaced role lane

/-- The next fresh claim's complete memory batch is selected by the fixed
recursive rows. A caller cannot replace it with a separate typed result. -/
@[irreducible] noncomputable def currentMemory
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
    (rows : Rows candidate statementId config artifact headers statement value
      proof) :
    ProductionMemoryCheckedBatchRows.Result
      rows.program.currentMemoryLayout rows.assignment headers :=
  ProductionMemoryCheckedBatchRows.derive rows.program.currentMemoryValid
    headers rows.assignmentCanonical rows.one rows.currentMemoryHeadersPlaced
    (rows.program.currentMemory_satisfied rows.satisfied)

/-- The first row-derived boundary is the outgoing continuation carry. -/
theorem currentMemoryStartParsed
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
    (rows : Rows candidate statementId config artifact headers statement value
      proof) :
    MemoryCarryPublicRows.ParsedColumnsMatch
      rows.program.continuationLayout.outgoing.reference rows.assignment headers
      (rows.currentMemory.boundary 0) := by
  rw [← rows.program.currentMemoryStartsAt]
  exact rows.currentMemory.boundaryParsed 0

end Rows

/-- Application witness for the row-derived recursive result. The outgoing
carry comes from the current checked-memory rows. No challenge authority
occurs in this structure. -/
structure Application
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
    (rows : Rows candidate statementId config artifact headers statement value
      proof)
    (machine : Machine ProgramType) (programValue : ProgramType) where
  applicationAfter : AppStateVector
  batch : Batch candidate machine programValue
    (WasmStateEncoding.decode rows.recursive.priorState.applicationState)
    applicationAfter
  applicationPlaced :
    ProductionRecursiveSuccessorRowsFor.ApplicationProducerPlaced
      rows.program.successorLayout rows.assignment batch
  priorCanonical : rows.recursive.priorState.Canonical headers

namespace Application

/-- The application successor consumes the exact start boundary of the
row-derived current memory batch. -/
noncomputable def outgoing
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
    {rows : Rows candidate statementId config artifact headers statement value
      proof}
    {machine : Machine ProgramType} {programValue : ProgramType}
    (application : Application rows machine programValue) :
    MemoryCarryCodec.Value :=
  rows.currentMemory.boundary 0

theorem outgoingParsed
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
    {rows : Rows candidate statementId config artifact headers statement value
      proof}
    {machine : Machine ProgramType} {programValue : ProgramType}
    (application : Application rows machine programValue) :
    MemoryCarryPublicRows.ParsedColumnsMatch
      rows.program.continuationLayout.outgoing.reference rows.assignment headers
      application.outgoing := by
  exact rows.currentMemoryStartParsed

noncomputable def coreEvidence
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
    {rows : Rows candidate statementId config artifact headers statement value
      proof}
    {machine : Machine ProgramType} {programValue : ProgramType}
    (application : Application rows machine programValue) :
    ProductionRecursiveSuccessorFor.CoreEvidence candidate statementId config
      artifact rows.program.fold.priorLayout rows.assignment headers
      rows.priorPrefix value proof rows.recursive machine programValue where
  applicationAfter := application.applicationAfter
  batch := application.batch
  continuation := rows.program.continuationLayout
  continuationValid := rows.program.continuationValid
  continuationIntermediate := rows.program.continuationIntermediate
  outgoing := application.outgoing
  outgoingParsed := application.outgoingParsed
  continuationRows := rows.program.continuation_satisfied rows.satisfied
  assignmentCanonical := rows.assignmentCanonical
  one := rows.one
  priorCanonical := application.priorCanonical

@[irreducible] noncomputable def successor
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
    {rows : Rows candidate statementId config artifact headers statement value
      proof}
    {machine : Machine ProgramType} {programValue : ProgramType}
    (application : Application rows machine programValue) :=
  ProductionRecursiveSuccessorFor.value candidate statementId config artifact
    value proof rows.recursive.priorState application.batch application.outgoing

end Application

end Nightstream.Implementation.NebulaV2.ProductionPaperRecursiveAcceptedRowsFor
