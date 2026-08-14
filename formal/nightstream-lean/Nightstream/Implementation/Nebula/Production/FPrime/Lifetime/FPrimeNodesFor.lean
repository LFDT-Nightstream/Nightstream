import Nightstream.Implementation.Nebula.Production.FPrime.Base.AcceptedRowsFor
import Nightstream.Implementation.Nebula.Production.FPrime.Recursive.AcceptedApplicationFor
import Nightstream.Implementation.Nebula.Production.FPrime.Recursive.ProducerInvocationFor
import Nightstream.Implementation.Nebula.Production.FPrime.Terminal.AcceptedRowsFor

/-!
Contract: exponent-indexed concrete witnesses for the three HyperNova
Construction-2 invocation roles.

`BaseNode` has no prior claim. `RecursiveNode` verifies and consumes one exact
prior complete claim and produces one next complete claim. `TerminalNode`
verifies and consumes one exact trailing claim and has no producer fields.

This module owns only local invocation witnesses. Global ordering is proved in
`ProductionPaperExactFPrimeLifetimeFor`.

Assurance tier: exponent-indexed invocation model.

Emits constraints: no.
-/

set_option autoImplicit false
set_option maxRecDepth 30000

namespace Nightstream.Implementation.Nebula.ProductionPaperExactFPrimeLifetimeFor

open Nightstream.Implementation.Nebula
open Nightstream.Protocol.Nebula
open Nightstream.Protocol.Nebula.ApplicationBatch
open Nightstream.Protocol.Nebula.FPrime
open Nightstream.Protocol.Nebula.ProductionProfileCandidates
open Nightstream.Protocol.Nebula.WasmState
open Nightstream.Protocol.Nebula.WasmStatement
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint

structure Context (Program : Type) where
  candidate : Id
  rowVariables : Nat
  logicalWidth : Nat
  publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth
  operationsShape : Phi81Relation.Shape
  snapshotShape : Phi81Relation.Shape
  statementId : ProductConcreteNifsFor.StatementId
  config : ProductPaperAlgebraFor.Config rowVariables logicalWidth publicFits
    operationsShape snapshotShape
  artifact : ProductConcreteNifsFor.RelationArtifact rowVariables logicalWidth
    publicFits
  relationAuthority : ProductionFreshClaimProducerFor.RelationAuthority
    publicFits artifact
  seedManifest : SeedSchedule.Manifest
  seedManifestProfile : seedManifest.profile = identity candidate
  headers : ChainHeaders Digest.Value
  statement : ProductionStatement Program
  machine : Machine Program
  baseWidths : FullClaimEnvelope.CompilerWidths
  baseArtifact : BaseManifestSchema.Artifact baseWidths
  baseMemoryAuthority : ProductionBaseCurrentMemoryRowsFor.Authority candidate
    baseArtifact
  baseChallengeProgram : ProductionBaseChallengeAuthorityRowsFor.Program
    candidate rowVariables
  baseChallengeRowsMatched : baseChallengeProgram.MatchesArtifact baseArtifact
  baseChallengeStatementIdExact : baseChallengeProgram.statementId = statementId
  baseChallengeStatementIdentityExact :
    baseChallengeProgram.statementIdentity = statement.base.identity
  recursiveProgram : ProductionRecursiveCoreManifestFor.Program candidate
    rowVariables
  terminalStatementLayout : ProductionPaperTerminalStatementRowsFor.Layout
  terminalTypedProgram : ProductionPaperTerminalCompleteProgramFor.Program
    candidate config
      { fold := recursiveProgram.fold
        statementLayout := terminalStatementLayout }
  baseArtifactProfileExact : baseArtifact.profile = identity candidate
  baseArtifactRowVariablesExact : baseArtifact.rowVariableCount = rowVariables
  baseArtifactSeedExact : baseArtifact.seedManifest = seedManifest
  recursiveSeedManifestExact :
    recursiveProgram.fold.seedManifest = seedManifest
  baseArmRowsExact : relationAuthority.fPrimeProgram.baseRows =
    baseArtifact.programRows
  recursiveProgramIncluded : recursiveProgram.RowsIncluded
    relationAuthority.fPrimeProgram.recursiveRows
  baseIterationColumnExact :
    relationAuthority.fPrimeProgram.layout.iterationZero.iterationColumn =
      baseArtifact.layouts.baseIteration.iterationColumn
  recursiveIterationColumnExact :
    relationAuthority.fPrimeProgram.layout.iterationZero.iterationColumn =
      recursiveProgram.fold.priorLayout.state.invocationColumn
  recursiveStatementIdExact :
    recursiveProgram.fold.statementId = statementId
  recursiveStatementIdentityExact :
    recursiveProgram.statementIdentity = statement.base.identity

namespace Context

/-- The terminal program shares the recursive fold core by construction. The
verifier context cannot supply an unrelated fold plus an equality proof. -/
def terminalProgram
    {Program : Type} (context : Context Program) :
    ProductionPaperTerminalFoldManifestFor.Program context.candidate
      context.rowVariables where
  fold := context.recursiveProgram.fold
  statementLayout := context.terminalStatementLayout

@[simp] theorem terminalProgram_fold
    {Program : Type} (context : Context Program) :
    context.terminalProgram.fold = context.recursiveProgram.fold := by
  rfl

abbrev FullShape {Program : Type} (context : Context Program) :=
  ProductPaperAlgebraFor.FullShape context.rowVariables context.logicalWidth
    context.publicFits

abbrev Claim {Program : Type} (context : Context Program) :=
  ProductionFieldNativeFullClaim.Value context.candidate context.FullShape

abbrev Successor {Program : Type} (context : Context Program) :=
  ProductionSuccessorStateBinding.Value context.candidate context.FullShape

abbrev Proof {Program : Type} (context : Context Program) :=
  ProductionProductPiCcsTypedBridgeFor.ExactProof context.rowVariables

abbrev FreshAssignment {Program : Type} (context : Context Program) :=
  ProductPaperAlgebraFor.Assignment context.rowVariables context.logicalWidth
    context.publicFits

abbrev Schema {Program : Type} (context : Context Program) :=
  ProductionPaperRecursiveRelationRowsSoundFor.ProtocolSchema
    context.rowVariables context.logicalWidth context.publicFits

abbrev Verifier {Program : Type} (context : Context Program) :=
  ProductionPaperRecursiveRelationRowsSoundFor.paperVerifier context.candidate
    context.statementId context.config context.artifact

abbrev Receipt {Program : Type} (context : Context Program) :=
  ProductionBatchedFPrime.Verified context.candidate context.Schema
    Digest.Value (ProductState.Challenges K) (ProductState.State K)
    context.Verifier

abbrev ProtocolClaim {Program : Type} (context : Context Program) :=
  ProductionBatchedFPrime.Claim context.candidate context.Schema Digest.Value
    (ProductState.Challenges K) (ProductState.State K)

end Context

structure BaseNode
    {Program : Type} (context : Context Program) where
  baseRows : ProductionPaperBaseAcceptedRowsFor.Accepted context.baseWidths
  baseArtifactExact : baseRows.artifact = context.baseArtifact
  after : AppStateVector
  batch : Batch context.candidate context.machine
    context.statement.base.program
    context.statement.base.initialApplicationState after
  freshAssignment : context.FreshAssignment
  supplement : ProductionPaperBaseAcceptedRowsFor.Supplement context.candidate
    context.statementId context.config context.artifact
    context.relationAuthority context.headers
    context.statement baseRows context.machine after batch
    (baseArtifactExact.symm ▸ context.baseMemoryAuthority)
    context.baseChallengeProgram freshAssignment

namespace BaseNode

/-- The selected verifier context fixes the exact profile used by the base
segment-opening rows. -/
theorem baseArtifactProfileExact
    {Program : Type} {context : Context Program} (node : BaseNode context) :
    node.baseRows.artifact.profile = identity context.candidate := by
  calc
    node.baseRows.artifact.profile = context.baseArtifact.profile :=
      congrArg BaseManifestSchema.Artifact.profile node.baseArtifactExact
    _ = identity context.candidate := context.baseArtifactProfileExact

/-- The base seed manifest is fixed by the selected base artifact. -/
theorem seedManifestExact
    {Program : Type} {context : Context Program} (node : BaseNode context) :
    node.baseRows.artifact.seedManifest = context.seedManifest := by
  calc
    node.baseRows.artifact.seedManifest = context.baseArtifact.seedManifest :=
      congrArg BaseManifestSchema.Artifact.seedManifest node.baseArtifactExact
    _ = context.seedManifest := context.baseArtifactSeedExact

/-- The base memory layout is fixed by the selected base artifact. -/
noncomputable def memoryAuthority
    {Program : Type} {context : Context Program} (node : BaseNode context) :
    ProductionBaseCurrentMemoryRowsFor.Authority context.candidate
      node.baseRows.artifact :=
  node.baseArtifactExact.symm ▸ context.baseMemoryAuthority

/-- Claim-zero memory data is derived from the satisfying base assignment. -/
noncomputable def memoryResult
    {Program : Type} {context : Context Program} (node : BaseNode context) :
    ProductionMemoryCheckedBatchRows.Result node.memoryAuthority.layout
      node.baseRows.assignment context.headers :=
  node.supplement.memoryResult

noncomputable def opening
    {Program : Type} {context : Context Program} (node : BaseNode context) :
    ProductionPaperBaseInvocationFor.Opening :=
  node.baseRows.opening

noncomputable def producer
    {Program : Type} {context : Context Program} (node : BaseNode context) :
    context.Successor :=
  ProductionPaperBaseInvocationFor.state context.candidate context.headers
    context.statement node.opening node.batch

noncomputable def claim
    {Program : Type} {context : Context Program} (node : BaseNode context) :
    context.Claim :=
  ProductionPaperBaseInvocationFor.claim context.candidate context.statementId
    context.config context.headers context.statement node.opening node.batch
    node.memoryResult.suffixBatch node.freshAssignment

/-- Base exactness is derived from the independent evidence. It is not a node
input. -/
theorem exact
    {Program : Type} {context : Context Program} (node : BaseNode context) :
    ProductionPaperBaseInvocationFor.ExactInvocation context.candidate
      context.statementId context.config context.artifact
      context.relationAuthority context.headers
      context.statement node.opening context.machine
      node.after node.batch node.memoryResult node.freshAssignment
      (node.supplement.evidence node.baseArtifactProfileExact) :=
  ProductionPaperBaseInvocationFor.exact context.candidate
    context.statementId context.config context.artifact
    context.relationAuthority context.headers
    context.statement node.opening context.machine
    node.after node.batch node.memoryResult node.freshAssignment
    (node.supplement.evidence node.baseArtifactProfileExact)

theorem invocationIndex_is_one
    {Program : Type} {context : Context Program} (node : BaseNode context) :
    node.producer.augmentedInvocationIndex = 1 :=
  node.exact.baseInvocationIndex

/-- The base memory challenge uses the unique verifier-derived base
authority.  It cannot use an authority selected independently of the
statement, canonical base input, or base successor prefix. -/
theorem challengeAuthorityExact
    {Program : Type} {context : Context Program} (node : BaseNode context) :
    node.opening.authority =
      ProductionPaperBaseInvocationFor.challengeAuthority
        (rowVariables := context.rowVariables)
        (logicalWidth := context.logicalWidth)
        (publicFits := context.publicFits) context.candidate
        context.statementId context.headers context.statement node.opening
        node.batch :=
  node.exact.challengeAuthorityExact

/-- The base relation uses the row-forced zero iteration and therefore selects
the one verifier-owned base manifest. -/
theorem freshSelectsFixedBase
    {Program : Type} {context : Context Program} (node : BaseNode context) :
    node.baseRows.assignment
          context.relationAuthority.fPrimeProgram.layout.iterationZero.iterationColumn =
        0 /\
      R1CS.Satisfies context.baseArtifact.programRows
        node.baseRows.assignment := by
  have baseZero := node.baseRows.inputIterationZero
  rw [node.baseArtifactExact] at baseZero
  have iterationZero :
      node.baseRows.assignment
          context.relationAuthority.fPrimeProgram.layout.iterationZero.iterationColumn =
        0 := by
    rw [context.baseIterationColumnExact]
    exact baseZero
  have selected := node.supplement.freshRelation.selectedBranch
  rcases selected with base | recursive
  · refine ⟨iterationZero, ?_⟩
    exact Eq.mp
      (congrArg
        (fun rows => R1CS.Satisfies rows node.baseRows.assignment)
        context.baseArmRowsExact)
      base.2
  · omega

end BaseNode

structure RecursiveNode
    {Program : Type} (context : Context Program) (previous : context.Claim) where
  proof : context.Proof
  rows : ProductionPaperRecursiveAcceptedRowsFor.Rows context.candidate
    context.statementId context.config context.artifact context.headers
    context.statement previous proof
  programExact : rows.program = context.recursiveProgram
  application : ProductionPaperRecursiveAcceptedRowsFor.Application rows
    context.machine context.statement.base.program
  freshAssignment : context.FreshAssignment
  evidence : ProductionPaperRecursiveProducerInvocationFor.Evidence
    context.candidate context.statementId context.config context.artifact
    context.relationAuthority context.statement rows.program.fold.priorLayout
    rows.assignment context.headers rows.priorPrefix previous proof
    rows.recursive context.machine context.statement.base.program
    rows.program.successorLayout application.supplement rows.currentMemory
    freshAssignment

namespace RecursiveNode

abbrev program
    {Program : Type} {context : Context Program} {previous : context.Claim}
    (node : RecursiveNode context previous) := node.rows.program

abbrev priorAuthority
    {Program : Type} {context : Context Program} {previous : context.Claim}
    (node : RecursiveNode context previous) := node.rows.program.fold.priorLayout

abbrev assignment
    {Program : Type} {context : Context Program} {previous : context.Claim}
    (node : RecursiveNode context previous) := node.rows.assignment

theorem programSatisfied
    {Program : Type} {context : Context Program} {previous : context.Claim}
    (node : RecursiveNode context previous) :
    R1CS.Satisfies node.program.rows node.assignment := node.rows.satisfied

abbrev priorPrefix
    {Program : Type} {context : Context Program} {previous : context.Claim}
    (node : RecursiveNode context previous) := node.rows.priorPrefix

noncomputable abbrev recursive
    {Program : Type} {context : Context Program} {previous : context.Claim}
    (node : RecursiveNode context previous) := node.rows.recursive

abbrev successorLayout
    {Program : Type} {context : Context Program} {previous : context.Claim}
    (node : RecursiveNode context previous) := node.rows.program.successorLayout

noncomputable abbrev supplement
    {Program : Type} {context : Context Program} {previous : context.Claim}
    (node : RecursiveNode context previous) := node.application.supplement

noncomputable abbrev currentMemory
    {Program : Type} {context : Context Program} {previous : context.Claim}
    (node : RecursiveNode context previous) := node.rows.currentMemory

@[simp] theorem priorAuthorityExact
    {Program : Type} {context : Context Program} {previous : context.Claim}
    (node : RecursiveNode context previous) :
    node.priorAuthority = node.program.fold.priorLayout := rfl

@[simp] theorem successorLayoutExact
    {Program : Type} {context : Context Program} {previous : context.Claim}
    (node : RecursiveNode context previous) :
    node.successorLayout = node.program.successorLayout := rfl

/-- The recursive row program fixes the compact-chain seed manifest. -/
theorem seedManifestExact
    {Program : Type} {context : Context Program} {previous : context.Claim}
    (node : RecursiveNode context previous) :
    node.recursive.compactManifest = context.seedManifest := by
  calc
    node.recursive.compactManifest = node.program.fold.seedManifest := by
      exact node.rows.recursiveCompactManifestExact
    _ = context.recursiveProgram.fold.seedManifest :=
      congrArg (fun program => program.fold.seedManifest) node.programExact
    _ = context.seedManifest := context.recursiveSeedManifestExact

/-- Construct the local F-prime node from one satisfying recursive manifest.
The recursive verifier result, NIFS output, successor, challenge authority,
and successor-row supplement are all selected or derived from `rows`.

`producerEvidence` contains only application lowering and fresh-claim
compiler boundaries. The current memory batch comes from `rows`; it is not a
constructor input. -/
noncomputable def ofAcceptedRows
    {Program : Type} {context : Context Program} {previous : context.Claim}
    {proof : context.Proof}
    (rows : ProductionPaperRecursiveAcceptedRowsFor.Rows context.candidate
      context.statementId context.config context.artifact context.headers
      context.statement previous proof)
    (application : ProductionPaperRecursiveAcceptedRowsFor.Application rows
      context.machine context.statement.base.program)
    (freshAssignment : context.FreshAssignment)
    (producerEvidence : ProductionPaperRecursiveProducerInvocationFor.Evidence
      context.candidate context.statementId context.config context.artifact
      context.relationAuthority context.statement rows.program.fold.priorLayout
      rows.assignment
      context.headers rows.priorPrefix previous proof rows.recursive
      context.machine context.statement.base.program
      rows.program.successorLayout application.supplement rows.currentMemory
      freshAssignment)
    (programExact : rows.program = context.recursiveProgram) :
    RecursiveNode context previous where
  proof := proof
  rows := rows
  programExact := programExact
  application := application
  freshAssignment := freshAssignment
  evidence := producerEvidence

noncomputable def nextClaim
    {Program : Type} {context : Context Program} {previous : context.Claim}
    (node : RecursiveNode context previous) : context.Claim :=
  ProductionPaperRecursiveProducerInvocationFor.claim context.candidate
    context.statementId context.config node.supplement
    node.currentMemory.suffixBatch node.freshAssignment

/-- Recursive consume-before-produce exactness is derived from the recursive
rows and the explicit application/fresh-producer evidence. It is not a node
input. -/
theorem exact
    {Program : Type} {context : Context Program} {previous : context.Claim}
    (node : RecursiveNode context previous) :
    ProductionPaperRecursiveProducerInvocationFor.ExactInvocation
      context.candidate context.statementId context.config context.artifact
      context.relationAuthority context.statement node.priorAuthority
      node.assignment context.headers
      node.priorPrefix previous node.proof node.recursive context.machine
      context.statement.base.program node.successorLayout node.supplement
      node.currentMemory node.freshAssignment node.evidence :=
  ProductionPaperRecursiveProducerInvocationFor.exact node.evidence

theorem consumes_previous
    {Program : Type} {context : Context Program} {previous : context.Claim}
    (node : RecursiveNode context previous) :
    node.recursive.verified.claim = previous.toProtocolClaim :=
  node.exact.previousConsumed.verifiedClaimExact

theorem proof_is_exact
    {Program : Type} {context : Context Program} {previous : context.Claim}
    (node : RecursiveNode context previous) :
    node.recursive.verified.proof = node.proof :=
  node.exact.previousConsumed.verifiedProofExact

theorem accepted
    {Program : Type} {context : Context Program} {previous : context.Claim}
    (node : RecursiveNode context previous) :
    context.Verifier node.recursive.verified.proof
      node.recursive.verified.claim :=
  node.exact.previousConsumed.verifierAccepted

/-- The branch input is the invocation index of the exact prior state parsed
by the recursive verifier rows. -/
theorem inputIterationExact
    {Program : Type} {context : Context Program} {previous : context.Claim}
    (node : RecursiveNode context previous) :
    node.assignment
          context.relationAuthority.fPrimeProgram.layout.iterationZero.iterationColumn =
      node.recursive.priorState.augmentedInvocationIndex := by
  calc
    node.assignment
          context.relationAuthority.fPrimeProgram.layout.iterationZero.iterationColumn =
        node.assignment
          context.recursiveProgram.fold.priorLayout.state.invocationColumn := by
      rw [context.recursiveIterationColumnExact]
    _ = node.assignment
          node.program.fold.priorLayout.state.invocationColumn := by
      exact congrArg
        (fun program =>
          node.assignment program.fold.priorLayout.state.invocationColumn)
        node.programExact.symm
    _ = node.assignment node.priorAuthority.state.invocationColumn := by
      rw [node.priorAuthorityExact]
    _ = node.recursive.priorState.augmentedInvocationIndex :=
      node.recursive.priorAuthorityResult.priorPlaced.invocation

/-- A reachable recursive invocation has a positive prior invocation index.
For such a node, the same committed witness selects and satisfies the one
verifier-owned recursive manifest. -/
theorem freshSelectsFixedRecursive
    {Program : Type} {context : Context Program} {previous : context.Claim}
    (node : RecursiveNode context previous)
    (priorPositive :
      0 < node.recursive.priorState.augmentedInvocationIndex) :
    0 < node.assignment
          context.relationAuthority.fPrimeProgram.layout.iterationZero.iterationColumn /\
      R1CS.Satisfies context.recursiveProgram.rows node.assignment := by
  have iterationPositive :
      0 < node.assignment
          context.relationAuthority.fPrimeProgram.layout.iterationZero.iterationColumn := by
    rw [node.inputIterationExact]
    exact priorPositive
  have selected := node.evidence.freshRelation.selectedBranch
  rcases selected with base | recursive
  · have baseZero :
        node.assignment
            context.relationAuthority.fPrimeProgram.layout.iterationZero.iterationColumn =
          0 := by
      exact base.1
    omega
  · refine ⟨iterationPositive, ?_⟩
    exact context.recursiveProgram.satisfies_of_rowsIncluded
      context.recursiveProgramIncluded recursive.2

end RecursiveNode

structure TerminalNode
    {Program : Type} (context : Context Program) (previous : context.Claim) where
  proof : context.Proof
  rowAccepted : ProductionPaperTerminalAcceptedRowsFor.Accepted
    context.candidate context.statementId context.config context.artifact
    context.headers context.statement previous proof context.terminalProgram
    context.terminalTypedProgram

namespace TerminalNode

def priorAuthority
    {Program : Type} {context : Context Program} {previous : context.Claim}
    (node : TerminalNode context previous) :
    ProductionPaperPriorStateAuthorityRowsFor.Layout context.candidate
      context.rowVariables :=
  context.terminalProgram.fold.priorLayout

def assignment
    {Program : Type} {context : Context Program} {previous : context.Claim}
    (node : TerminalNode context previous) : Nat -> Nat :=
  ProductionPaperTerminalOpeningRowsFor.pulledAssignment
    node.rowAccepted.rows.assignment

/-- The accepted terminal assignment satisfies the one terminal fold program
selected by the verifier context. -/
theorem fixedProgramSatisfied
    {Program : Type} {context : Context Program} {previous : context.Claim}
    (node : TerminalNode context previous) :
    R1CS.Satisfies context.terminalProgram.rows node.assignment := by
  exact node.rowAccepted.rows.satisfied

/-- The same typed assignment satisfies the verifier-owned complete terminal
program. This includes all fourteen common-witness openings and CE checks. -/
theorem fixedTypedProgramSatisfied
    {Program : Type} {context : Context Program} {previous : context.Claim}
    (node : TerminalNode context previous) :
    context.terminalTypedProgram.RowsSatisfied context.artifact.system
      node.rowAccepted.rows.assignment := by
  exact node.rowAccepted.programRows

/-- The terminal verifier uses the same common paper-fold core as every
recursive invocation. -/
theorem commonFoldExact
    {Program : Type} {context : Context Program} {previous : context.Claim}
    (node : TerminalNode context previous) :
    context.terminalProgram.fold = context.recursiveProgram.fold := rfl

/-- The terminal row program fixes the compact-chain seed manifest. -/
theorem compactManifestExact
    {Program : Type} {context : Context Program} {previous : context.Claim}
    (node : TerminalNode context previous) :
    node.rowAccepted.rows.recursive.compactManifest = context.seedManifest := by
  calc
    node.rowAccepted.rows.recursive.compactManifest =
        context.terminalProgram.fold.seedManifest :=
      node.rowAccepted.rows.recursiveCompactManifestExact
    _ = context.recursiveProgram.fold.seedManifest := rfl
    _ = context.seedManifest := context.recursiveSeedManifestExact

def priorPrefix
    {Program : Type} {context : Context Program} {previous : context.Claim}
    (node : TerminalNode context previous) :
    ProductionPaperPriorStateAuthorityRowsFor.Prefix context.candidate
      context.FullShape :=
  node.rowAccepted.rows.priorPrefix

noncomputable def recursive
    {Program : Type} {context : Context Program} {previous : context.Claim}
    (node : TerminalNode context previous) :
    ProductionPaperRecursiveRelationRowsSoundFor.Result context.candidate
      context.statementId context.config context.artifact node.priorAuthority
      node.assignment context.headers node.priorPrefix previous node.proof :=
  node.rowAccepted.rows.recursive

noncomputable def opening
    {Program : Type} {context : Context Program} {previous : context.Claim}
    (node : TerminalNode context previous) :
    ProductionPaperTerminalInvocationRowsSoundFor.ProductOpening
      context.candidate context.statementId context.config context.artifact
      previous node.proof :=
  node.rowAccepted.opening

/-- Terminal exactness is derived from one row-accepted terminal assignment.
It is not a node input. -/
theorem exact
    {Program : Type} {context : Context Program} {previous : context.Claim}
    (node : TerminalNode context previous) :
    ProductionPaperTerminalInvocationRowsSoundFor.ExactInvocation
      context.candidate context.statementId context.config context.artifact
      node.priorAuthority node.assignment context.headers node.priorPrefix
      previous node.proof node.recursive node.opening context.statement :=
  node.rowAccepted.exactInvocation

theorem consumes_trailing
    {Program : Type} {context : Context Program} {previous : context.Claim}
    (node : TerminalNode context previous) :
    node.recursive.verified.claim = previous.toProtocolClaim :=
  node.exact.trailingClaimExact

theorem proof_is_exact
    {Program : Type} {context : Context Program} {previous : context.Claim}
    (node : TerminalNode context previous) :
    node.recursive.verified.proof = node.proof :=
  node.exact.trailingProofExact

theorem accepted
    {Program : Type} {context : Context Program} {previous : context.Claim}
    (node : TerminalNode context previous) :
    context.Verifier node.recursive.verified.proof
      node.recursive.verified.claim :=
  node.exact.trailingVerified

end TerminalNode

end Nightstream.Implementation.Nebula.ProductionPaperExactFPrimeLifetimeFor
