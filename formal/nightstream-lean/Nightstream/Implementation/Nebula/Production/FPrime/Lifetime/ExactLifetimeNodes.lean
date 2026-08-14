import Mathlib.Algebra.Field.TransferInstance
import Nightstream.Implementation.Nebula.Core.ConcreteField
import Nightstream.Implementation.Nebula.Production.FPrime.Lifetime.ExactFPrimeLifetimeFor
import Nightstream.Implementation.Nebula.Production.Memory.RowTrace
import Nightstream.Implementation.Nebula.Production.Memory.RowSegments
import Nightstream.Implementation.Nebula.Production.FPrime.Base.AcceptedRowsFor
import Nightstream.Implementation.Nebula.Production.FPrime.Base.InvocationFor
import Nightstream.Implementation.Nebula.Production.FPrime.Recursive.ProducerInvocationFor
import Nightstream.Implementation.Nebula.Production.FPrime.Lifetime.StateContinuityFor
import Nightstream.Implementation.Nebula.Production.FPrime.Terminal.InvocationRowsSoundFor
import Nightstream.Implementation.Nebula.Production.FPrime.Terminal.AcceptedRowsFor
import Nightstream.Implementation.Nebula.Production.Artifact.VerifierArtifactFor
import Nightstream.Implementation.Nebula.Production.Artifact.SemanticAuthority
import Nightstream.Protocol.Nebula.ApplicationBatchCompletion
import Nightstream.Protocol.Nebula.ProductionBatchedAugmentedLifecycle
import Nightstream.Protocol.Nebula.ProductionBatchedDelayedReverse
import Nightstream.Protocol.Nebula.WasmPublicStatementEncoding

/-!
Contract: fixed verifier context and exact local nodes for the production
Nebula-on-SuperNeo delayed-consumption adaptation of HyperNova Construction 2.

This module owns the verifier-selected base, recursive, and terminal row
programs. It also owns local producer, consumer, same-claim, and terminal
evidence. It does not construct or extract the global delayed lifetime.

Does not own cryptographic probability bounds, generated-row extraction,
external-byte parsing, recursive-size closure, Rust refinement, or a
deployed terminal verifier.

Emits constraints: no.
-/

set_option autoImplicit false
set_option maxRecDepth 100000
set_option maxHeartbeats 5000000

namespace Nightstream.Implementation.Nebula.ProductionPaperExactLifetime

open Nightstream.Implementation.Nebula
open Nightstream.Protocol.Nebula
open Nightstream.Protocol.Nebula.ApplicationBatch
open Nightstream.Protocol.Nebula.AugmentedLifecycle
open Nightstream.Protocol.Nebula.FPrime
open Nightstream.Protocol.Nebula.ProductionBatchedFPrime
open Nightstream.Protocol.Nebula.ProductionProfileCandidates
open Nightstream.Protocol.Nebula.WasmState
open Nightstream.Protocol.Nebula.WasmStatement
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint

/-- The executable coefficient pair uses the proved field equivalence. -/
noncomputable local instance concreteKField : Field K :=
  ConcreteField.superNeoEquiv.field

/-- The legacy bit-serial public decoder and any field-native candidate
decoder cannot both describe one statement. This is why the production
F-prime context must use `DecodesFor (identity candidate)`. -/
theorem legacyDecoder_conflicts_with_candidate
    {Program : Type}
    {image : WasmPublicStatementEncoding.PublicImage}
    {statement : ProductionStatement Program}
    (candidate : Id)
    (legacy : image.Decodes statement)
    (selected : image.DecodesFor (identity candidate) statement) : False := by
  apply identity_ne_v2 candidate
  exact selected.exactProfile.symm.trans legacy.exactProfile

/-! ## Fixed verifier context -/

/-- All data fixed for one relation and verifier-key identity. None of these
fields is proof-carried acceptance evidence. -/
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
  publicImage : WasmPublicStatementEncoding.PublicImage
  publicDecoded : publicImage.DecodesFor (identity candidate) statement
  semanticAuthority : ProductionSemanticAuthority.Artifact Program
  semanticAuthorityMatches :
    ProductionSemanticAuthority.MatchesStatement semanticAuthority statement
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

/-- The application machine is selected by the verifier-owned semantic
artifact. It is not an independent theorem parameter. -/
abbrev machine {Program : Type} (context : Context Program) : Machine Program :=
  context.semanticAuthority.machine

/-- The snapshot-root function is selected by the same semantic artifact as
the application machine and memory-plan identity. -/
abbrev snapshotRoot {Program : Type} (context : Context Program) :
    Snapshot -> Digest.Value :=
  context.semanticAuthority.snapshotRoot

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

abbrev Collision {Program : Type} (context : Context Program) :=
  ProductionPaperStateContinuityFor.Collision context.candidate
    context.FullShape context.statementId

/-- Verifier-recomputed root of the complete public initial memory image. -/
def authoritativeInitialMemoryRoot
    {Program : Type} (context : Context Program) : Digest.Value :=
  context.snapshotRoot (Snapshot.ofImage context.statement.base.initialImage)

/-- Claim-lifecycle view of the same verifier-owned context. The conversion
drops only semantic-extraction data; it changes no protocol parameter. -/
def claimLifecycle
    {Program : Type} (context : Context Program) :
    ProductionPaperExactFPrimeLifetimeFor.Context Program where
  candidate := context.candidate
  rowVariables := context.rowVariables
  logicalWidth := context.logicalWidth
  publicFits := context.publicFits
  operationsShape := context.operationsShape
  snapshotShape := context.snapshotShape
  statementId := context.statementId
  config := context.config
  artifact := context.artifact
  relationAuthority := context.relationAuthority
  seedManifest := context.seedManifest
  seedManifestProfile := context.seedManifestProfile
  headers := context.headers
  statement := context.statement
  machine := context.machine
  baseWidths := context.baseWidths
  baseArtifact := context.baseArtifact
  baseMemoryAuthority := context.baseMemoryAuthority
  baseChallengeProgram := context.baseChallengeProgram
  baseChallengeRowsMatched := context.baseChallengeRowsMatched
  baseChallengeStatementIdExact := context.baseChallengeStatementIdExact
  baseChallengeStatementIdentityExact :=
    context.baseChallengeStatementIdentityExact
  recursiveProgram := context.recursiveProgram
  terminalStatementLayout := context.terminalStatementLayout
  terminalTypedProgram := context.terminalTypedProgram
  baseArtifactProfileExact := context.baseArtifactProfileExact
  baseArtifactRowVariablesExact := context.baseArtifactRowVariablesExact
  baseArtifactSeedExact := context.baseArtifactSeedExact
  recursiveSeedManifestExact := context.recursiveSeedManifestExact
  baseArmRowsExact := context.baseArmRowsExact
  recursiveProgramIncluded := context.recursiveProgramIncluded
  baseIterationColumnExact := context.baseIterationColumnExact
  recursiveIterationColumnExact := context.recursiveIterationColumnExact
  recursiveStatementIdExact := context.recursiveStatementIdExact
  recursiveStatementIdentityExact := context.recursiveStatementIdentityExact

end Context

/-! ## Generated verifier context -/

/-- Inputs that are outside the generated relation artifact. The relation,
compiler, NIFS, base program, recursive program, and terminal program are not
repeated here. They all come from `verifierArtifact`. -/
structure GeneratedContext (Program : Type) where
  candidate : Id
  baseWidths : FullClaimEnvelope.CompilerWidths
  verifierArtifact : ProductionVerifierArtifactFor.Artifact candidate
    baseWidths
  headers : ChainHeaders Digest.Value
  statement : ProductionStatement Program
  publicImage : WasmPublicStatementEncoding.PublicImage
  publicDecoded : publicImage.DecodesFor (identity candidate) statement
  /-- The decoded statement and recursive row program use one complete
  statement identity. This includes the application relation, program, and
  memory plan; verifier-key equality alone is insufficient. -/
  statementIdentityExact :
    statement.base.identity =
      verifierArtifact.dimensions.coreProgram.statementIdentity
  semanticAuthority : ProductionSemanticAuthority.Artifact Program
  semanticAuthorityMatches :
    ProductionSemanticAuthority.MatchesStatement semanticAuthority statement

namespace GeneratedContext

abbrev machine {Program : Type} (generated : GeneratedContext Program) :
    Machine Program :=
  generated.semanticAuthority.machine

abbrev snapshotRoot {Program : Type} (generated : GeneratedContext Program) :
    Snapshot -> Digest.Value :=
  generated.semanticAuthority.snapshotRoot

/-- The external statement and generated verifier artifact use one complete
verifier-key identity, including every independently framed manifest digest. -/
theorem statementVerifierKeySelected
    {Program : Type} (generated : GeneratedContext Program) :
    generated.statement.base.identity.verifierKey =
      generated.verifierArtifact.verifierKeyIdentity := by
  calc
    generated.statement.base.identity.verifierKey =
        generated.verifierArtifact.dimensions.coreProgram.statementIdentity.verifierKey :=
      congrArg Soundness.StatementIdentity.verifierKey
        generated.statementIdentityExact
    _ = generated.verifierArtifact.verifierKeyIdentity :=
      generated.verifierArtifact.recursiveStatementVerifierKeyExact

/-- The full statement identity, not only its verifier-key subrecord, is
selected by the generated recursive program. -/
theorem statementIdentitySelected
    {Program : Type} (generated : GeneratedContext Program) :
    generated.verifierArtifact.dimensions.coreProgram.statementIdentity =
      generated.statement.base.identity :=
  generated.statementIdentityExact.symm

/-- The verifier-selected base challenge program uses the same complete
statement identity as the recursive program and decoded public statement. -/
theorem baseChallengeStatementIdentitySelected
    {Program : Type} (generated : GeneratedContext Program) :
    generated.verifierArtifact.baseChallengeProgram.statementIdentity =
      generated.statement.base.identity := by
  exact generated.verifierArtifact.baseChallengeStatementIdentityExact.trans
    generated.statementIdentitySelected

/-- The verifier-selected base challenge program and recursive fold use the
same statement-domain identifier. -/
theorem baseChallengeStatementIdSelected
    {Program : Type} (generated : GeneratedContext Program) :
    generated.verifierArtifact.baseChallengeProgram.statementId =
      generated.verifierArtifact.statementId :=
  generated.verifierArtifact.baseChallengeStatementIdExact

/-- Fixed lifetime context derived from one verifier-owned artifact. All row
and relation identities are definitions or structural fields of that artifact.
-/
def context
    {Program : Type} (generated : GeneratedContext Program) : Context Program where
  candidate := generated.candidate
  rowVariables := generated.verifierArtifact.dimensions.relationRowVariables
  logicalWidth := generated.verifierArtifact.LogicalWidth
  publicFits := generated.verifierArtifact.PublicFits
  operationsShape := generated.verifierArtifact.operationsShape
  snapshotShape := generated.verifierArtifact.snapshotShape
  statementId := generated.verifierArtifact.statementId
  config := generated.verifierArtifact.config
  artifact := generated.verifierArtifact.relationArtifact
  relationAuthority := generated.verifierArtifact.relationAuthority
  seedManifest := generated.verifierArtifact.base.seedManifest
  seedManifestProfile :=
    generated.verifierArtifact.base.seedManifestProfile.trans
      generated.verifierArtifact.baseProfileSelected
  headers := generated.headers
  statement := generated.statement
  publicImage := generated.publicImage
  publicDecoded := generated.publicDecoded
  semanticAuthority := generated.semanticAuthority
  semanticAuthorityMatches := generated.semanticAuthorityMatches
  baseWidths := generated.baseWidths
  baseArtifact := generated.verifierArtifact.base
  baseMemoryAuthority := generated.verifierArtifact.baseMemory
  baseChallengeProgram := generated.verifierArtifact.baseChallengeProgram
  baseChallengeRowsMatched :=
    generated.verifierArtifact.baseChallengeRowsMatched
  baseChallengeStatementIdExact :=
    generated.verifierArtifact.baseChallengeStatementIdExact
  baseChallengeStatementIdentityExact := by
    exact generated.baseChallengeStatementIdentitySelected
  recursiveProgram := generated.verifierArtifact.dimensions.coreProgram
  terminalStatementLayout :=
    generated.verifierArtifact.dimensions.terminalStatementLayout
  terminalTypedProgram := generated.verifierArtifact.terminalTypedProgram
  baseArtifactProfileExact :=
    generated.verifierArtifact.baseProfileSelected
  baseArtifactRowVariablesExact :=
    generated.verifierArtifact.baseRowVariablesExact
  baseArtifactSeedExact := rfl
  recursiveSeedManifestExact :=
    generated.verifierArtifact.seedManifestExact
  baseArmRowsExact := rfl
  recursiveProgramIncluded :=
    generated.verifierArtifact.dimensions.exponentIndexedCoreIncluded
  baseIterationColumnExact :=
    generated.verifierArtifact.baseIterationColumnExact
  recursiveIterationColumnExact :=
    generated.verifierArtifact.recursiveIterationColumnExact
  recursiveStatementIdExact :=
    generated.verifierArtifact.recursiveStatementIdExact
  recursiveStatementIdentityExact :=
    generated.statementIdentitySelected

@[simp] theorem context_relationAuthority
    {Program : Type} (generated : GeneratedContext Program) :
    generated.context.relationAuthority =
      generated.verifierArtifact.relationAuthority := rfl

@[simp] theorem context_fPrimeProgram
    {Program : Type} (generated : GeneratedContext Program) :
    generated.context.relationAuthority.fPrimeProgram =
      generated.verifierArtifact.fPrimeProgram := rfl

@[simp] theorem context_terminalProgram
    {Program : Type} (generated : GeneratedContext Program) :
    generated.context.terminalProgram =
      generated.verifierArtifact.dimensions.terminalProgram := rfl

@[simp] theorem context_config
    {Program : Type} (generated : GeneratedContext Program) :
    generated.context.config = generated.verifierArtifact.config := rfl

@[simp] theorem context_relationArtifact
    {Program : Type} (generated : GeneratedContext Program) :
    generated.context.artifact =
      generated.verifierArtifact.relationArtifact := rfl

end GeneratedContext

/-! ## Exact producer authority -/

/-- Facts emitted by an exact base or recursive producer. The application
state and full CCS image are both authority-bearing. -/
structure ProducerAuthority
    {Program : Type} (context : Context Program)
    (producer : context.Successor) (claim : context.Claim)
    (applicationAfter : AppStateVector) : Prop where
  canonical : producer.Canonical context.headers
  applicationExact : producer.applicationState =
    WasmStateEncoding.encode applicationAfter
  fullMatches : ProductionMemoryBoundCcsPublic.FullMatches claim.ccsPublic
    (ProductionSuccessorStateBinding.outputDigest context.statementId producer)
    claim.memory

/-- The row-derived memory batch produced with one complete fresh claim.
This is producer evidence. It is not the delayed consumer replay that checks
the same claim in the next invocation. -/
structure ProducedBatch
    {Program : Type} (context : Context Program)
    (producer : context.Successor) (claim : context.Claim) where
  layout : ProductionMemoryCheckedBatchRows.Layout context.candidate
  assignment : Nat -> Nat
  result : ProductionMemoryCheckedBatchRows.Result layout assignment
    context.headers
  producerCanonical : producer.Canonical context.headers
  beforeExact : result.semantic 0 =
    MemoryCarryParser.semanticCarry producer.memoryCarry
      producerCanonical.memoryCarry.stepIndex
  memoryExact : result.suffixBatch = claim.memory

namespace ProducedBatch

/-- Pair producer rows with the later verifier receipt for that exact full
claim. The receipt supplies acceptance. The producer supplies all semantic
memory records. -/
def rowEvidence
    {Program : Type} {context : Context Program}
    {producer : context.Successor} {claim : context.Claim}
    (produced : ProducedBatch context producer claim)
    (receipt : context.Receipt)
    (claimExact : receipt.claim = claim.toProtocolClaim
      (NifsProof := ProductionProductPiCcsTypedBridgeFor.ExactProof
        context.rowVariables)) :
    ProductionMemoryRowTrace.BatchEvidence context.candidate context.Schema
      context.Verifier context.headers where
  receipt := receipt
  layout := produced.layout
  assignment := produced.assignment
  result := produced.result
  memoryExact := by
    rw [claimExact]
    exact produced.memoryExact

end ProducedBatch

/-- Semantic carry represented by the complete authenticated successor. -/
def producerCarry
    {Program : Type} {context : Context Program}
    {producer : context.Successor} {claim : context.Claim}
    {applicationAfter : AppStateVector}
    (authority : ProducerAuthority context producer claim applicationAfter) :
    Carry Digest.Value (ProductState.Challenges K) (ProductState.State K) :=
  MemoryCarryParser.semanticCarry producer.memoryCarry
    authority.canonical.memoryCarry.stepIndex

namespace ProducedBatch

/-- The producer's authenticated carry is exactly the start boundary of its
own row-derived batch. -/
theorem before_eq_producerCarry
    {Program : Type} {context : Context Program}
    {producer : context.Successor} {claim : context.Claim}
    {applicationAfter : AppStateVector}
    (produced : ProducedBatch context producer claim)
    (authority : ProducerAuthority context producer claim applicationAfter) :
    produced.result.semantic 0 = producerCarry authority := by
  unfold producerCarry
  simpa using produced.beforeExact

end ProducedBatch

/-! ## Concrete recursive and terminal nodes -/

/-- One exact nonterminal consumer-producer invocation. Every dependent item
is tied to the same previous complete claim. -/
structure RecursiveNode
    {Program : Type} (context : Context Program) (previous : context.Claim) where
  proof : ProductionProductPiCcsTypedBridgeFor.ExactProof context.rowVariables
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
    node.recursive.compactManifest = node.program.fold.seedManifest :=
      node.rows.recursiveCompactManifestExact
    _ = context.recursiveProgram.fold.seedManifest :=
      congrArg (fun program => program.fold.seedManifest) node.programExact
    _ = context.seedManifest := context.recursiveSeedManifestExact

/-- Row-owned constructor for the semantic lifetime node. The accepted
recursive manifest fixes the prior-claim verifier result, the NIFS output,
the successor, the challenge authority, and the current memory batch. The
separate producer evidence ends at application lowering and fresh-claim
relation satisfaction. -/
noncomputable def ofAcceptedRows
    {Program : Type} {context : Context Program} {previous : context.Claim}
    {proof : ProductionProductPiCcsTypedBridgeFor.ExactProof
      context.rowVariables}
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

/-- Local recursive exactness is derived from evidence. -/
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

noncomputable def nextProducer
    {Program : Type} {context : Context Program} {previous : context.Claim}
    (node : RecursiveNode context previous) : context.Successor :=
  node.supplement.successor

noncomputable def nextClaim
    {Program : Type} {context : Context Program} {previous : context.Claim}
    (node : RecursiveNode context previous) : context.Claim :=
  ProductionPaperRecursiveProducerInvocationFor.claim context.candidate
    context.statementId context.config node.supplement
    node.currentMemory.suffixBatch node.freshAssignment

/-- The current producer rows belong to the next claim, not to the claim
consumed by this recursive invocation. -/
noncomputable def currentProduced
    {Program : Type} {context : Context Program} {previous : context.Claim}
    (node : RecursiveNode context previous) :
    ProducedBatch context node.nextProducer node.nextClaim where
  layout := node.program.currentMemoryLayout
  assignment := node.assignment
  result := node.currentMemory
  producerCanonical := node.exact.previousConsumed.successorCanonical
  beforeExact := by
    simpa [nextProducer,
      ProductionPaperRecursiveInvocationRowsSoundFor.Supplement.successor,
      ProductionRecursiveSuccessorFor.value] using
      node.exact.currentMemoryStartsAfterContinuation
  memoryExact := by
    exact node.exact.nextMemoryExact.symm

noncomputable def nextApplication
    {Program : Type} {context : Context Program} {previous : context.Claim}
    (node : RecursiveNode context previous) : AppStateVector :=
  node.supplement.evidence.applicationAfter

noncomputable def applicationRows
    {Program : Type} {context : Context Program} {previous : context.Claim}
    (node : RecursiveNode context previous) : List ApplicationTrace.ApplicationRow :=
  node.supplement.evidence.batch.rows

/-- The next producer authority is derived from the exact generated producer;
it is not supplied by the lifetime. -/
theorem nextAuthority
    {Program : Type} {context : Context Program} {previous : context.Claim}
    (node : RecursiveNode context previous) :
    ProducerAuthority context node.nextProducer node.nextClaim
      node.nextApplication := by
  refine
    { canonical := node.exact.previousConsumed.successorCanonical
      applicationExact := ?_
      fullMatches := node.exact.nextPublicExact }
  rfl

/-- The claim-order view retains every relation witness. -/
noncomputable def toClaimNode
    {Program : Type} {context : Context Program} {previous : context.Claim}
    (node : RecursiveNode context previous) :
    ProductionPaperExactFPrimeLifetimeFor.RecursiveNode
      context.claimLifecycle previous where
  proof := node.proof
  rows := node.rows
  programExact := node.programExact
  application := node.application
  freshAssignment := node.freshAssignment
  evidence := node.evidence

@[simp] theorem toClaimNode_priorInvocationIndex
    {Program : Type} {context : Context Program} {previous : context.Claim}
    (node : RecursiveNode context previous) :
    node.toClaimNode.recursive.priorState.augmentedInvocationIndex =
      node.recursive.priorState.augmentedInvocationIndex := by
  rfl

end RecursiveNode

/-- One row-accepted terminal invocation. Its recursive result, close checks,
public checks, and fourteen same-assignment child openings are derived from
the complete terminal manifest package. It has no successor or fresh claim. -/
structure TerminalNode
    {Program : Type} (context : Context Program) (previous : context.Claim) where
  proof : ProductionProductPiCcsTypedBridgeFor.ExactProof context.rowVariables
  accepted : ProductionPaperTerminalAcceptedRowsFor.Accepted
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
    node.accepted.rows.assignment

/-- The terminal assignment satisfies the exact terminal program selected by
the lifetime context, not only the program stored inside the accepted node. -/
theorem fixedProgramSatisfied
    {Program : Type} {context : Context Program} {previous : context.Claim}
    (node : TerminalNode context previous) :
    R1CS.Satisfies context.terminalProgram.rows node.assignment := by
  exact node.accepted.rows.satisfied

/-- The same assignment satisfies the complete verifier-owned typed terminal
program, including every opening and CE row family. -/
theorem fixedTypedProgramSatisfied
    {Program : Type} {context : Context Program} {previous : context.Claim}
    (node : TerminalNode context previous) :
    context.terminalTypedProgram.RowsSatisfied context.artifact.system
      node.accepted.rows.assignment := by
  exact node.accepted.programRows

/-- The terminal relation and every recursive invocation use one common
paper-fold core. -/
theorem commonFoldExact
    {Program : Type} {context : Context Program} {previous : context.Claim}
    (node : TerminalNode context previous) :
    context.terminalProgram.fold = context.recursiveProgram.fold := rfl

/-- The terminal row program fixes the compact-chain seed manifest. -/
theorem compactManifestExact
    {Program : Type} {context : Context Program} {previous : context.Claim}
    (node : TerminalNode context previous) :
    node.accepted.rows.recursive.compactManifest = context.seedManifest := by
  calc
    node.accepted.rows.recursive.compactManifest =
        context.terminalProgram.fold.seedManifest :=
      node.accepted.rows.recursiveCompactManifestExact
    _ = context.recursiveProgram.fold.seedManifest := rfl
    _ = context.seedManifest := context.recursiveSeedManifestExact

def priorPrefix
    {Program : Type} {context : Context Program} {previous : context.Claim}
    (node : TerminalNode context previous) :
    ProductionPaperPriorStateAuthorityRowsFor.Prefix context.candidate
      context.FullShape :=
  node.accepted.rows.priorPrefix

noncomputable def recursive
    {Program : Type} {context : Context Program} {previous : context.Claim}
    (node : TerminalNode context previous) :
    ProductionPaperRecursiveRelationRowsSoundFor.Result context.candidate
      context.statementId context.config context.artifact node.priorAuthority
      node.assignment context.headers node.priorPrefix previous node.proof :=
  node.accepted.rows.recursive

noncomputable def opening
    {Program : Type} {context : Context Program} {previous : context.Claim}
    (node : TerminalNode context previous) :
    ProductionPaperTerminalInvocationRowsSoundFor.ProductOpening
      context.candidate context.statementId context.config context.artifact
      previous node.proof :=
  node.accepted.opening

theorem exact
    {Program : Type} {context : Context Program} {previous : context.Claim}
    (node : TerminalNode context previous) :
    ProductionPaperTerminalInvocationRowsSoundFor.ExactInvocation
      context.candidate context.statementId context.config context.artifact
      node.priorAuthority node.assignment context.headers node.priorPrefix
      previous node.proof node.recursive node.opening context.statement :=
  node.accepted.exactInvocation

noncomputable def toClaimNode
    {Program : Type} {context : Context Program} {previous : context.Claim}
    (node : TerminalNode context previous) :
    ProductionPaperExactFPrimeLifetimeFor.TerminalNode
      context.claimLifecycle previous where
  proof := node.proof
  rowAccepted := node.accepted

@[simp] theorem toClaimNode_priorInvocationIndex
    {Program : Type} {context : Context Program} {previous : context.Claim}
    (node : TerminalNode context previous) :
    node.toClaimNode.recursive.priorState.augmentedInvocationIndex =
      node.recursive.priorState.augmentedInvocationIndex := by
  rfl

end TerminalNode

end Nightstream.Implementation.Nebula.ProductionPaperExactLifetime
