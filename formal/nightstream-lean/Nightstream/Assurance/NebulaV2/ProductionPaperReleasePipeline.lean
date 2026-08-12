import Nightstream.Implementation.NebulaV2.ProductionPaperExactLifetimeSoundness
import Nightstream.Implementation.NebulaV2.RecursiveSizeClosure
import Nightstream.Protocol.NebulaV2.WasmPublicStatementEncoding

/-!
Contract: staged acceptance bridge for the exact production-paper
Nebula-on-SuperNeo F-prime lifetime.

This module keeps five external release obligations separate: canonical byte
decoding, terminal-backend extraction, fold-knowledge extraction, generated
relation refinement, and application-port refinement. Each stage can fail
only through its named event. The final stage produces exact local base,
recursive, and terminal nodes. The model theorem then derives the completed
application and memory execution from those nodes.

No stage can return the final `HasSoundExecution` conclusion. The local node
refinement stage contains `ApplicationBatch.Runs` proofs because generated
application-row refinement is not yet available. It does not contain a
completed global execution, global memory execution, global balance fact, or
the exported soundness conclusion.

Cryptographic state collisions and row-derived memory failures remain exact
typed failure witnesses. This module does not assign probabilities to them.

Assurance tier: staged implementation-refinement boundaries and deterministic
composition theorem. It is not a deployed soundness proof by itself.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Assurance.NebulaV2.ProductionPaperReleasePipeline

open Nightstream.Implementation.NebulaV2.ProductionPaperExactLifetime
open Nightstream.Implementation.NebulaV2.ProductionPaperExactLifetimeSoundness
open Nightstream.Protocol.NebulaV2.Soundness
open Nightstream.Protocol.NebulaV2.WasmPublicStatementEncoding
open Nightstream.Protocol.NebulaV2.WasmState

/-- One complete exact production lifetime. The dependent tail can consume
only the claim and producer batch emitted by its base node. -/
structure ExtractedLifetime
    {Program : Type} (context : Context Program) where
  base : BaseNode context
  tail : Tail context base.producer base.claim base.after base.produced

/-- Acceptance of canonical bytes by one selected terminal verifier. -/
def Accepts
    {Bytes Parsed : Type}
    (decode : Bytes → Option Parsed)
    (terminalAccepts : Parsed → Prop)
    (proof : Bytes) : Prop :=
  ∃ parsed, decode proof = some parsed ∧ terminalAccepts parsed

/-- Exact proof-system failures allowed before generated rows are recovered.
This family prevents one generic fold-extraction event from hiding setup,
bundle, commitment, transcript, or sampler failures. -/
inductive FoldBoundaryFailure (occurs : BadEvent → Prop) : Prop where
  | bundlePropagation (failure : occurs .bundlePropagation) :
      FoldBoundaryFailure occurs
  | commitmentBinding (failure : occurs .commitmentBinding) :
      FoldBoundaryFailure occurs
  | compactTokenBinding (failure : occurs .compactTokenBinding) :
      FoldBoundaryFailure occurs
  | seededSetup (failure : occurs .seededSetup) :
      FoldBoundaryFailure occurs
  | poseidonOrTranscript (failure : occurs .poseidonOrTranscript) :
      FoldBoundaryFailure occurs
  | piRlcSampler (failure : occurs .piRlcSampler) :
      FoldBoundaryFailure occurs
  | foldExtraction (failure : occurs .foldExtraction) :
      FoldBoundaryFailure occurs

/-- Exact failures allowed while refining an extracted fold witness to the
one generated F-prime relation. -/
inductive RelationBoundaryFailure (occurs : BadEvent → Prop) : Prop where
  | fPrimeLifecycle (failure : occurs .fPrimeLifecycle) :
      RelationBoundaryFailure occurs
  | recursiveSizeClosure (failure : occurs .recursiveSizeClosure) :
      RelationBoundaryFailure occurs
  | circuitRefinement (failure : occurs .circuitRefinement) :
      RelationBoundaryFailure occurs

/-- External stages that must be discharged by distinct deployed artifacts.

The intermediate types and relations are deliberately visible. A release
review must inspect a concrete inhabitant of every field. Merely inhabiting
this structure is not implementation evidence; the countermodel module proves
that each abstract stage can otherwise ignore its input. -/
structure StagedExtraction
    {Bytes Parsed Program : Type}
    (decode : Bytes → Option Parsed)
    (terminalAccepts : Parsed → Prop)
    (occurs : BadEvent → Prop)
    (context : Context Program) where
  CanonicalProof : Type
  TerminalWitness : Type
  FoldWitness : Type
  GeneratedRows : Type
  decodedCanonical : Bytes → Parsed → CanonicalProof → Prop
  terminalExtracted : Parsed → CanonicalProof → TerminalWitness → Prop
  foldExtracted : TerminalWitness → FoldWitness → Prop
  rowsRefined : FoldWitness → GeneratedRows → Prop
  lifetimeRefined : GeneratedRows → ExtractedLifetime context → Prop
  decodeRefinement : ∀ proof parsed,
    decode proof = some parsed →
      occurs .decode ∨
        ∃ canonical, decodedCanonical proof parsed canonical
  terminalBackendExtraction : ∀ proof parsed canonical,
    decodedCanonical proof parsed canonical →
      terminalAccepts parsed →
        occurs .terminalBackend ∨
          ∃ terminal, terminalExtracted parsed canonical terminal
  foldKnowledgeExtraction : ∀ parsed canonical terminal,
    terminalExtracted parsed canonical terminal →
      FoldBoundaryFailure occurs ∨
        ∃ fold, foldExtracted terminal fold
  generatedRelationRefinement : ∀ terminal fold,
    foldExtracted terminal fold →
      RelationBoundaryFailure occurs ∨
        ∃ rows, rowsRefined fold rows
  applicationPortRefinement : ∀ fold rows,
    rowsRefined fold rows →
      occurs .applicationPortCoverage ∨
        ∃ lifetime, lifetimeRefined rows lifetime

/-- Retained evidence from all five external stages for one accepted proof.
The relations keep each intermediate artifact tied to its predecessor. -/
structure ExtractionTrace
    {Bytes Parsed Program : Type}
    {decode : Bytes → Option Parsed}
    {terminalAccepts : Parsed → Prop}
    {occurs : BadEvent → Prop}
    {context : Context Program}
    (stages : StagedExtraction decode terminalAccepts occurs context)
    (proof : Bytes) (parsed : Parsed) (lifetime : ExtractedLifetime context) : Type where
  canonical : stages.CanonicalProof
  decodedCanonical : stages.decodedCanonical proof parsed canonical
  terminal : stages.TerminalWitness
  terminalExtracted : stages.terminalExtracted parsed canonical terminal
  fold : stages.FoldWitness
  foldExtracted : stages.foldExtracted terminal fold
  rows : stages.GeneratedRows
  rowsRefined : stages.rowsRefined fold rows
  lifetimeRefined : stages.lifetimeRefined rows lifetime

/-- Exact release-boundary failures. A stage cannot use an unrelated event to
avoid its extraction obligation. -/
inductive ExtractionFailure (occurs : BadEvent → Prop) : Prop where
  | decode (failure : occurs .decode) : ExtractionFailure occurs
  | terminalBackend (failure : occurs .terminalBackend) : ExtractionFailure occurs
  | fold (failure : FoldBoundaryFailure occurs) : ExtractionFailure occurs
  | relation (failure : RelationBoundaryFailure occurs) :
      ExtractionFailure occurs
  | applicationPortCoverage (failure : occurs .applicationPortCoverage) :
      ExtractionFailure occurs

/-- Failures retained by the direct production path. Staged extraction
failures are separate from exact semantic or cryptographic obstructions
produced by the lifetime theorem. -/
inductive Failure
    {Program : Type}
    (occurs : BadEvent → Prop)
    (context : Context Program) : Prop where
  | extraction (failure : ExtractionFailure occurs) : Failure occurs context
  | lifetime
      (failure :
        Nightstream.Implementation.NebulaV2.ProductionPaperExactLifetimeSoundness.Failure
          context) :
      Failure occurs context

/-! ## Exact failure accounting -/

namespace FoldBoundaryFailure

/-- Each proof-system boundary failure exposes its exact registered event. -/
theorem impliesAnyBad
    {occurs : BadEvent → Prop} (failure : FoldBoundaryFailure occurs) :
    AnyBad occurs := by
  cases failure with
  | bundlePropagation evidence => exact ⟨.bundlePropagation, evidence⟩
  | commitmentBinding evidence => exact ⟨.commitmentBinding, evidence⟩
  | compactTokenBinding evidence => exact ⟨.compactTokenBinding, evidence⟩
  | seededSetup evidence => exact ⟨.seededSetup, evidence⟩
  | poseidonOrTranscript evidence => exact ⟨.poseidonOrTranscript, evidence⟩
  | piRlcSampler evidence => exact ⟨.piRlcSampler, evidence⟩
  | foldExtraction evidence => exact ⟨.foldExtraction, evidence⟩

end FoldBoundaryFailure

namespace RelationBoundaryFailure

/-- Each generated-relation boundary failure exposes its exact registered
event. -/
theorem impliesAnyBad
    {occurs : BadEvent → Prop} (failure : RelationBoundaryFailure occurs) :
    AnyBad occurs := by
  cases failure with
  | fPrimeLifecycle evidence => exact ⟨.fPrimeLifecycle, evidence⟩
  | recursiveSizeClosure evidence => exact ⟨.recursiveSizeClosure, evidence⟩
  | circuitRefinement evidence => exact ⟨.circuitRefinement, evidence⟩

end RelationBoundaryFailure

namespace ExtractionFailure

/-- Each staged extraction constructor yields its exact public event. -/
theorem impliesAnyBad
    {occurs : BadEvent → Prop} (failure : ExtractionFailure occurs) :
    AnyBad occurs := by
  cases failure with
  | decode evidence => exact ⟨.decode, evidence⟩
  | terminalBackend evidence => exact ⟨.terminalBackend, evidence⟩
  | fold evidence => exact evidence.impliesAnyBad
  | relation evidence => exact evidence.impliesAnyBad
  | applicationPortCoverage evidence =>
      exact ⟨.applicationPortCoverage, evidence⟩

end ExtractionFailure

/-- External accounting for the exact obstructions produced by the lifetime
theorem. This premise does not contain an execution witness or the desired
soundness conclusion. A concrete release must prove it from its collision,
fingerprint, and generated-row security games. -/
structure LifetimeFailureAccounting
    {Program : Type}
    (occurs : BadEvent → Prop)
    (context : Context Program) : Prop where
  stateCollision : ∀ _collision : context.Collision,
    occurs .poseidonOrTranscript
  fingerprint : ∀ {before :
      Nightstream.Protocol.NebulaV2.FPrime.ClosedCarry
        Nightstream.Protocol.NebulaV2.Digest.Value}
      (run :
        Nightstream.Implementation.NebulaV2.ProductionMemoryRowSegments.SegmentRun
          context.candidate context.Schema context.Verifier context.headers
          before),
    Nightstream.Protocol.NebulaV2.IdealFingerprint.EvaluationFailure
        (Nightstream.Implementation.NebulaV2.ProductionMemorySegmentSoundness.SegmentRun.fingerprintCheck
          run) →
      occurs .memoryFingerprint
  initialRoot : ∀ {before :
      Nightstream.Protocol.NebulaV2.FPrime.ClosedCarry
        Nightstream.Protocol.NebulaV2.Digest.Value}
      (run :
        Nightstream.Implementation.NebulaV2.ProductionMemoryRowSegments.SegmentRun
          context.candidate context.Schema context.Verifier context.headers
          before),
    context.snapshotRoot
          (Nightstream.Implementation.NebulaV2.ProductionMemorySnapshotCoverage.SegmentRun.snapshot
            run .initialSnapshot) ≠ before.memoryRoot →
      occurs .circuitRefinement
  finalRoot : ∀ {before :
      Nightstream.Protocol.NebulaV2.FPrime.ClosedCarry
        Nightstream.Protocol.NebulaV2.Digest.Value}
      (run :
        Nightstream.Implementation.NebulaV2.ProductionMemoryRowSegments.SegmentRun
          context.candidate context.Schema context.Verifier context.headers
          before),
    context.snapshotRoot
          (Nightstream.Implementation.NebulaV2.ProductionMemorySnapshotCoverage.SegmentRun.snapshot
            run .finalSnapshot) ≠ run.after.memoryRoot →
      occurs .circuitRefinement
  snapshotCollision :
    Nightstream.Protocol.NebulaV2.IdealAcceptance.SnapshotRootCollision
        context.snapshotRoot →
      occurs .poseidonOrTranscript

namespace Failure

/-- Every retained failure implies an exact named event once the concrete
lifetime-failure accounting obligation is discharged. -/
theorem impliesAnyBad
    {Program : Type} {occurs : BadEvent → Prop} {context : Context Program}
    (accounting : LifetimeFailureAccounting occurs context)
    (failure : Failure occurs context) :
    AnyBad occurs := by
  cases failure with
  | extraction extractionFailure => exact extractionFailure.impliesAnyBad
  | lifetime lifetimeFailure =>
      cases lifetimeFailure with
      | stateCollision collision =>
          exact ⟨.poseidonOrTranscript, accounting.stateCollision collision⟩
      | memory memoryFailure =>
          cases memoryFailure with
          | fingerprint run evaluationFailure =>
              exact ⟨.memoryFingerprint,
                accounting.fingerprint run evaluationFailure⟩
          | initialRoot run mismatch =>
              exact ⟨.circuitRefinement, accounting.initialRoot run mismatch⟩
          | finalRoot run mismatch =>
              exact ⟨.circuitRefinement, accounting.finalRoot run mismatch⟩
          | snapshotCollision collision =>
              exact ⟨.poseidonOrTranscript,
                accounting.snapshotCollision collision⟩

end Failure

/-- Successful acceptance retains the parsed value, the exact extracted F'
lifetime, and the certificate derived from that same lifetime. -/
structure Certificate
    {Bytes Parsed Program : Type}
    {decode : Bytes → Option Parsed}
    {terminalAccepts : Parsed → Prop}
    {occurs : BadEvent → Prop}
    {context : Context Program}
    (boundary : StagedExtraction decode terminalAccepts occurs context)
    (proof : Bytes) : Type where
  parsed : Parsed
  decoded : decode proof = some parsed
  lifetime : ExtractedLifetime context
  trace : ExtractionTrace boundary proof parsed lifetime
  semantic :
    Nightstream.Implementation.NebulaV2.ProductionPaperExactLifetimeSoundness.CertifiedExecution
      lifetime.base

namespace Certificate

/-- The release certificate exposes the exact base memory-challenge
authority.  A deployed extractor must therefore recover a base node whose
authority binds the verifier-owned identity, canonical base input, and base
successor prefix. -/
theorem baseChallengeAuthorityExact
    {Bytes Parsed Program : Type}
    {decode : Bytes → Option Parsed}
    {terminalAccepts : Parsed → Prop}
    {occurs : BadEvent → Prop}
    {context : Context Program}
    {boundary : StagedExtraction decode terminalAccepts occurs context}
    {proof : Bytes}
    (certificate : Certificate boundary proof) :
    certificate.lifetime.base.opening.authority =
      Nightstream.Implementation.NebulaV2.ProductionPaperBaseInvocationFor.challengeAuthority
        (rowVariables := context.rowVariables)
        (logicalWidth := context.logicalWidth)
        (publicFits := context.publicFits) context.candidate
        context.statementId context.headers context.statement
        certificate.lifetime.base.opening certificate.lifetime.base.batch :=
  certificate.semantic.baseChallengeAuthorityExact

/-- The certificate exposes the complete delayed claim schedule, including
the separate trailing terminal consumer. -/
theorem exactClaimSchedule
    {Bytes Parsed Program : Type}
    {decode : Bytes → Option Parsed}
    {terminalAccepts : Parsed → Prop}
    {occurs : BadEvent → Prop}
    {context : Context Program}
    {boundary : StagedExtraction decode terminalAccepts occurs context}
    {proof : Bytes}
    (certificate : Certificate boundary proof) :
    Nightstream.Implementation.NebulaV2.ProductionPaperExactFPrimeLifetimeFor.Lifetime.ExactSchedule
      certificate.semantic.extraction.claimLifetime :=
  certificate.semantic.exactClaimSchedule

/-- The extracted artifact also retains row-derived selection of the fixed
base arm and every fixed recursive arm. -/
theorem fixedBranchSchedule
    {Bytes Parsed Program : Type}
    {decode : Bytes → Option Parsed}
    {terminalAccepts : Parsed → Prop}
    {occurs : BadEvent → Prop}
    {context : Context Program}
    {boundary : StagedExtraction decode terminalAccepts occurs context}
    {proof : Bytes}
    (certificate : Certificate boundary proof) :
    Nightstream.Implementation.NebulaV2.ProductionPaperExactFPrimeLifetimeFor.Lifetime.FixedBranchSchedule
      certificate.semantic.extraction.claimLifetime :=
  certificate.semantic.fixedBranchSchedule

/-- The recursive and terminal consumers occur at the exact invocation
indexes `1, ..., T`. This is stronger than equality of the produced and
consumed claim counts. -/
theorem consumerInvocationIndicesExact
    {Bytes Parsed Program : Type}
    {decode : Bytes → Option Parsed}
    {terminalAccepts : Parsed → Prop}
    {occurs : BadEvent → Prop}
    {context : Context Program}
    {boundary : StagedExtraction decode terminalAccepts occurs context}
    {proof : Bytes}
    (certificate : Certificate boundary proof) :
    certificate.semantic.extraction.claimLifetime.schedule.consumerInvocationIndices =
      List.range' 1 certificate.semantic.extraction.receipts.length :=
  certificate.semantic.consumerInvocationIndicesExact

/-- Every recursive or terminal consumer receives the complete predecessor
state emitted by the preceding producer. Equality is over the full typed
state, not only its invocation index or digest. -/
theorem fullStateContinuityExact
    {Bytes Parsed Program : Type}
    {decode : Bytes → Option Parsed}
    {terminalAccepts : Parsed → Prop}
    {occurs : BadEvent → Prop}
    {context : Context Program}
    {boundary : StagedExtraction decode terminalAccepts occurs context}
    {proof : Bytes}
    (certificate : Certificate boundary proof) :
    Nightstream.Implementation.NebulaV2.ProductionPaperExactFPrimeLifetimeFor.Schedule.FullStateContinuous
      certificate.lifetime.base.producer
      certificate.semantic.extraction.claimLifetime.schedule :=
  certificate.semantic.fullStateContinuityExact

/-- The memory trace uses the exact receipt sequence consumed by the F-prime
schedule. -/
theorem receiptsExact
    {Bytes Parsed Program : Type}
    {decode : Bytes → Option Parsed}
    {terminalAccepts : Parsed → Prop}
    {occurs : BadEvent → Prop}
    {context : Context Program}
    {boundary : StagedExtraction decode terminalAccepts occurs context}
    {proof : Bytes}
    (certificate : Certificate boundary proof) :
    certificate.semantic.extraction.receipts =
      certificate.semantic.extraction.claimLifetime.consumedReceipts :=
  certificate.semantic.receiptsExact

/-- The deployed-path certificate retains the row-derived same-witness CCS
opening for every produced claim and consumed receipt. -/
theorem exactClaimOpenings
    {Bytes Parsed Program : Type}
    {decode : Bytes → Option Parsed}
    {terminalAccepts : Parsed → Prop}
    {occurs : BadEvent → Prop}
    {context : Context Program}
    {boundary : StagedExtraction decode terminalAccepts occurs context}
    {proof : Bytes}
    (certificate : Certificate boundary proof) :
    Nightstream.Implementation.NebulaV2.ProductionPaperClaimOpeningLifetimeFor.ClaimLifetime.ExactOpenings
      certificate.semantic.extraction.claimLifetime :=
  certificate.semantic.exactClaimOpenings

/-- Every commitment bundle consumed by the deployed-path certificate opens
from one row-derived assignment.  Use `everyConsumedClaimHolds` for the
simultaneous relation and public-input statement. -/
theorem everyConsumedBundleOpens
    {Bytes Parsed Program : Type}
    {decode : Bytes → Option Parsed}
    {terminalAccepts : Parsed → Prop}
    {occurs : BadEvent → Prop}
    {context : Context Program}
    {boundary : StagedExtraction decode terminalAccepts occurs context}
    {proof : Bytes}
    (certificate : Certificate boundary proof) :
    forall receipt,
      receipt ∈ certificate.semantic.extraction.claimLifetime.consumedReceipts ->
        exists assignment : context.claimLifecycle.FreshAssignment,
          Nightstream.Implementation.NebulaV2.ProductCommitmentAlgebra.commit
              context.config assignment =
            Nightstream.Implementation.NebulaV2.ProductNifsCodec.codecBundle
              receipt.claim.commitmentBundle :=
  certificate.semantic.everyConsumedBundleOpens

/-- Every consumed receipt retains one complete same-witness `CCS.Holds`
fact.  The product opening, public input, norm bound, and selected relation
cannot use detached assignments. -/
theorem everyConsumedClaimHolds
    {Bytes Parsed Program : Type}
    {decode : Bytes → Option Parsed}
    {terminalAccepts : Parsed → Prop}
    {occurs : BadEvent → Prop}
    {context : Context Program}
    {boundary : StagedExtraction decode terminalAccepts occurs context}
    {proof : Bytes}
    (certificate : Certificate boundary proof) :
    forall receipt,
      receipt ∈ certificate.semantic.extraction.claimLifetime.consumedReceipts ->
        exists assignment : context.claimLifecycle.FreshAssignment,
          Nightstream.SuperNeo.CCS.Holds
            (Nightstream.Implementation.NebulaV2.ProductPaperAlgebraFor.semantics
              context.config)
            Nightstream.SuperNeo.Concrete.productionGlobalParams
            (Nightstream.Implementation.NebulaV2.ProductionPaperClaimOpeningLifetimeFor.ClaimLifetime.statementOfProtocolClaim
              context.claimLifecycle receipt.claim) assignment :=
  certificate.semantic.everyConsumedClaimHolds

/-- For a generated verifier context, each consumed claim selects the exact
generated base branch or satisfies the exact mandatory recursive core. The
claim opening and branch extraction use one assignment. -/
theorem everyConsumedClaimSelectsGeneratedCore
    {Bytes Parsed Program : Type}
    {decode : Bytes → Option Parsed}
    {terminalAccepts : Parsed → Prop}
    {occurs : BadEvent → Prop}
    {generated : GeneratedContext Program}
    {boundary : StagedExtraction decode terminalAccepts occurs
      generated.context}
    {proof : Bytes}
    (certificate : Certificate boundary proof) :
    forall receipt,
      receipt ∈ certificate.semantic.extraction.claimLifetime.consumedReceipts ->
        exists assignment : generated.verifierArtifact.Assignment,
          Nightstream.SuperNeo.CCS.Holds
              (Nightstream.Implementation.NebulaV2.ProductPaperAlgebraFor.semantics
                generated.verifierArtifact.config)
              Nightstream.SuperNeo.Concrete.productionGlobalParams
              (Nightstream.Implementation.NebulaV2.ProductionPaperClaimOpeningLifetimeFor.ClaimLifetime.statementOfProtocolClaim
                generated.context.claimLifecycle receipt.claim) assignment /\
            generated.verifierArtifact.ExactGeneratedCoreBranch assignment := by
  intro receipt member
  have opened := certificate.semantic.exactClaimOpenings.consumed receipt member
  rcases
      Nightstream.Implementation.NebulaV2.ProductionPaperClaimOpeningLifetimeFor.ClaimLifetime.ReceiptOpening.exactDecodedBranch
        opened with
    ⟨assignment, holds, branch⟩
  refine ⟨assignment, holds, ?_⟩
  apply generated.verifierArtifact.generatedBranch_implies_coreBranch
  exact
    (generated.verifierArtifact.exactDecodedBranch_iff_generated
      assignment).mp branch

end Certificate

/-- Release-stage certificate for one generated verifier artifact. In
addition to the exact extracted lifetime, it contains finite source and
carrier capacity for the same artifact. It does not claim complete
Definition-12 recursive-size closure until generated-row refinement proves
that the artifact contains every required F-prime operation. -/
structure GeneratedCertificate
    {Bytes Parsed Program : Type}
    {decode : Bytes → Option Parsed}
    {terminalAccepts : Parsed → Prop}
    {occurs : BadEvent → Prop}
    (generated : GeneratedContext Program)
    (boundary : StagedExtraction decode terminalAccepts occurs
      generated.context)
    (proof : Bytes) : Type where
  semantic : Certificate boundary proof
  finiteCapacity :
    Nightstream.Implementation.NebulaV2.RecursiveSizeClosure.FiniteArtifactCapacity
      generated.verifierArtifact

namespace GeneratedCertificate

/-- The generated certificate uses a canonical fixed-width codec for the
exact complete source assignment, not a caller-selected payload type. -/
theorem recursivePayloadCanonical
    {Bytes Parsed Program : Type}
    {decode : Bytes → Option Parsed}
    {terminalAccepts : Parsed → Prop}
    {occurs : BadEvent → Prop}
    {generated : GeneratedContext Program}
    {boundary : StagedExtraction decode terminalAccepts occurs
      generated.context}
    {proof : Bytes}
    (certificate : GeneratedCertificate generated boundary proof) :
    (Nightstream.Implementation.NebulaV2.RecursiveSizeClosure.payloadCodec
      generated.verifierArtifact).Canonical :=
  certificate.finiteCapacity.payloadCanonical

/-- The same artifact fixes the complete F-prime row cube and expanded
low-norm carrier bound. -/
theorem rowsAndCarrierFit
    {Bytes Parsed Program : Type}
    {decode : Bytes → Option Parsed}
    {terminalAccepts : Parsed → Prop}
    {occurs : BadEvent → Prop}
    {generated : GeneratedContext Program}
    {boundary : StagedExtraction decode terminalAccepts occurs
      generated.context}
    {proof : Bytes}
    (certificate : GeneratedCertificate generated boundary proof) :
    generated.verifierArtifact.fPrimeProgram.rows.length <=
        2 ^ generated.verifierArtifact.dimensions.relationRowVariables /\
      Nightstream.Implementation.NebulaV2.ProductionFreshLowNormEncoding.logicalWidth
          generated.verifierArtifact.privateWidth <=
        2 ^ generated.verifierArtifact.dimensions.relationRowVariables :=
  ⟨certificate.finiteCapacity.exactRowDomain.1,
    certificate.finiteCapacity.carrierFits⟩

/-- The same generated verifier artifact also fixes a finite terminal
assignment and proves that both its terminal rows and all terminal columns fit
the selected terminal Boolean cube. -/
theorem terminalRowsAndCarrierFit
    {Bytes Parsed Program : Type}
    {decode : Bytes → Option Parsed}
    {terminalAccepts : Parsed → Prop}
    {occurs : BadEvent → Prop}
    {generated : GeneratedContext Program}
    {boundary : StagedExtraction decode terminalAccepts occurs
      generated.context}
    {proof : Bytes}
    (_certificate : GeneratedCertificate generated boundary proof) :
    generated.verifierArtifact.dimensions.terminalRows.length <=
        2 ^ generated.verifierArtifact.dimensions.terminalCircuitRowVariables /\
      generated.verifierArtifact.dimensions.terminalAssignmentWidth <=
        2 ^ generated.verifierArtifact.dimensions.terminalCircuitRowVariables :=
  generated.verifierArtifact.dimensions.terminal_rows_and_columns_fit

/-- The accepted public statement and the generated relation use one full
verifier-key identity. Equality covers all seven manifest digests; aggregate
digest equality alone is not used as authority. -/
theorem verifierKeyIdentityExact
    {Bytes Parsed Program : Type}
    {decode : Bytes → Option Parsed}
    {terminalAccepts : Parsed → Prop}
    {occurs : BadEvent → Prop}
    {generated : GeneratedContext Program}
    {boundary : StagedExtraction decode terminalAccepts occurs
      generated.context}
    {proof : Bytes}
    (_certificate : GeneratedCertificate generated boundary proof) :
    generated.statement.base.identity.verifierKey =
      generated.verifierArtifact.verifierKeyIdentity :=
  generated.statementVerifierKeySelected

end GeneratedCertificate

/-- Conditional elimination theorem for the exact production-paper lifetime.
The proof composes five named release obligations before it invokes the exact
local F-prime lifetime theorem. -/
theorem acceptance_under_staged_refinement_implies_certificate_or_failure
    {Bytes Parsed Program : Type}
    {decode : Bytes → Option Parsed}
    {terminalAccepts : Parsed → Prop}
    {occurs : BadEvent → Prop}
    {context : Context Program}
    (boundary : StagedExtraction decode terminalAccepts occurs context)
    {proof : Bytes}
    (accepted : Accepts decode terminalAccepts proof) :
    Failure occurs context \/ Nonempty (Certificate boundary proof) := by
  rcases accepted with ⟨parsed, decoded, terminalAccepted⟩
  rcases boundary.decodeRefinement proof parsed decoded with
    decodeFailure | ⟨canonical, decodedCanonical⟩
  · exact Or.inl (.extraction (.decode decodeFailure))
  · rcases boundary.terminalBackendExtraction proof parsed canonical
        decodedCanonical terminalAccepted with
      terminalFailure | ⟨terminal, terminalExtracted⟩
    · exact Or.inl (.extraction (.terminalBackend terminalFailure))
    · rcases boundary.foldKnowledgeExtraction parsed canonical terminal
          terminalExtracted with
        foldFailure | ⟨fold, foldExtracted⟩
      · exact Or.inl (.extraction (.fold foldFailure))
      · rcases boundary.generatedRelationRefinement terminal fold foldExtracted with
          relationFailure | ⟨rows, rowsRefined⟩
        · exact Or.inl (.extraction (.relation relationFailure))
        · rcases boundary.applicationPortRefinement fold rows rowsRefined with
            applicationFailure | ⟨lifetime, lifetimeRefined⟩
          · exact Or.inl
              (.extraction (.applicationPortCoverage applicationFailure))
          · rcases
              exact_lifetime_implies_certificate_or_failure lifetime.base
                lifetime.tail with
              lifetimeFailure | semantic
            · exact Or.inl (.lifetime lifetimeFailure)
            · rcases semantic with ⟨semantic⟩
              exact Or.inr ⟨
                { parsed := parsed
                  decoded := decoded
                  lifetime := lifetime
                  trace :=
                    { canonical := canonical
                      decodedCanonical := decodedCanonical
                      terminal := terminal
                      terminalExtracted := terminalExtracted
                      fold := fold
                      foldExtracted := foldExtracted
                      rows := rows
                      rowsRefined := rowsRefined
                      lifetimeRefined := lifetimeRefined }
                  semantic := semantic }⟩

/-- Semantic projection for callers that do not need the retained parsed
artifact and exact receipt schedule. -/
theorem acceptance_under_staged_refinement_implies_execution_or_failure
    {Bytes Parsed Program : Type}
    {decode : Bytes → Option Parsed}
    {terminalAccepts : Parsed → Prop}
    {occurs : BadEvent → Prop}
    {context : Context Program}
    (boundary : StagedExtraction decode terminalAccepts occurs context)
    {proof : Bytes}
    (accepted : Accepts decode terminalAccepts proof) :
    Failure occurs context \/
      HasSoundExecution context.machine.semantics context.statement.base
        context.snapshotRoot := by
  rcases acceptance_under_staged_refinement_implies_certificate_or_failure
      boundary accepted
    with failure | certificate
  · exact Or.inl failure
  · rcases certificate with ⟨certificate⟩
    exact Or.inr certificate.semantic.execution

/-- Release form with one public bad-event disjunction. Exact model failures
enter this disjunction only through their constructor-specific accounting
map. -/
theorem acceptance_under_staged_refinement_implies_any_bad_or_execution
    {Bytes Parsed Program : Type}
    {decode : Bytes → Option Parsed}
    {terminalAccepts : Parsed → Prop}
    {occurs : BadEvent → Prop}
    {context : Context Program}
    (boundary : StagedExtraction decode terminalAccepts occurs context)
    (accounting : LifetimeFailureAccounting occurs context)
    {proof : Bytes}
    (accepted : Accepts decode terminalAccepts proof) :
    Or (AnyBad occurs)
      (HasSoundExecution context.machine.semantics context.statement.base
        context.snapshotRoot) := by
  rcases acceptance_under_staged_refinement_implies_execution_or_failure
      boundary accepted with failure | execution
  · exact Or.inl (failure.impliesAnyBad accounting)
  · exact Or.inr execution

/-- If no registered release event occurs, accepted bytes have one completed
application execution and one matching global memory execution. -/
theorem acceptance_under_staged_refinement_and_no_bad_implies_execution
    {Bytes Parsed Program : Type}
    {decode : Bytes → Option Parsed}
    {terminalAccepts : Parsed → Prop}
    {occurs : BadEvent → Prop}
    {context : Context Program}
    (boundary : StagedExtraction decode terminalAccepts occurs context)
    (accounting : LifetimeFailureAccounting occurs context)
    (noBad : ∀ event, ¬ occurs event)
    {proof : Bytes}
    (accepted : Accepts decode terminalAccepts proof) :
    HasSoundExecution context.machine.semantics context.statement.base
      context.snapshotRoot := by
  rcases acceptance_under_staged_refinement_implies_any_bad_or_execution
      boundary accounting accepted with ⟨event, eventOccurs⟩ | execution
  · exact False.elim (noBad event eventOccurs)
  · exact execution

/-- The fixed WASM machine also derives the authenticated terminal result.
The exact public image is recovered from the decoder evidence stored in the
verifier context. -/
theorem acceptance_under_staged_refinement_implies_wasm_result_or_failure
    {Bytes Parsed Program : Type}
    {decode : Bytes → Option Parsed}
    {terminalAccepts : Parsed → Prop}
    {occurs : BadEvent → Prop}
    {context : Context Program}
    (boundary : StagedExtraction decode terminalAccepts occurs context)
    {proof : Bytes}
    (accepted : Accepts decode terminalAccepts proof) :
    Failure occurs context \/
      (HasSoundExecution context.machine.semantics context.statement.base
          context.snapshotRoot /\
        context.statement.base.expectedResult.finalApplicationState.Terminal
          context.statement.base.expectedResult.outcome /\
        context.publicImage = PublicImage.ofStatement context.statement) := by
  rcases acceptance_under_staged_refinement_implies_execution_or_failure
      boundary accepted
    with failure | execution
  · exact Or.inl failure
  · right
    have terminal :
        context.statement.base.expectedResult.finalApplicationState.Terminal
          context.statement.base.expectedResult.outcome := by
      rcases execution with
        ⟨_finalSnapshot, _segmentAccesses, applicationExecution,
          _memoryExecution, _segmentCount, _coverage, _finalRoot⟩
      exact context.machine.completedExecution_final_terminal
        applicationExecution
    exact ⟨execution, terminal, context.publicDecoded.exactImage⟩

/-! ## Generated-artifact release surface -/

/-- Release-facing staged theorem. Unlike the generic composition theorem,
this statement cannot use independently supplied base, recursive, compiler,
NIFS, or terminal programs. They come from one generated verifier artifact.
The deployed extraction boundary remains explicit. -/
theorem generated_acceptance_under_staged_refinement_implies_certificate_or_failure
    {Bytes Parsed Program : Type}
    {decode : Bytes → Option Parsed}
    {terminalAccepts : Parsed → Prop}
    {occurs : BadEvent → Prop}
    {generated : GeneratedContext Program}
    (boundary : StagedExtraction decode terminalAccepts occurs
      generated.context)
    {proof : Bytes}
    (accepted : Accepts decode terminalAccepts proof) :
    Failure occurs generated.context \/
      Nonempty (GeneratedCertificate generated boundary proof) := by
  rcases acceptance_under_staged_refinement_implies_certificate_or_failure
      boundary accepted
    with failure | certificate
  · exact Or.inl failure
  · rcases certificate with ⟨certificate⟩
    exact Or.inr ⟨
      { semantic := certificate
        finiteCapacity :=
          Nightstream.Implementation.NebulaV2.RecursiveSizeClosure.finiteArtifactCapacity
            generated.verifierArtifact }⟩

/-- Semantic projection of the generated-artifact release theorem. -/
theorem generated_acceptance_under_staged_refinement_implies_execution_or_failure
    {Bytes Parsed Program : Type}
    {decode : Bytes → Option Parsed}
    {terminalAccepts : Parsed → Prop}
    {occurs : BadEvent → Prop}
    {generated : GeneratedContext Program}
    (boundary : StagedExtraction decode terminalAccepts occurs
      generated.context)
    {proof : Bytes}
    (accepted : Accepts decode terminalAccepts proof) :
    Failure occurs generated.context \/
      HasSoundExecution generated.machine.semantics generated.statement.base
        generated.snapshotRoot := by
  rcases
      generated_acceptance_under_staged_refinement_implies_certificate_or_failure
        boundary accepted
    with failure | certificate
  · exact Or.inl failure
  · rcases certificate with ⟨certificate⟩
    exact Or.inr certificate.semantic.semantic.execution

/-- Generated-artifact release form with exact bad-event accounting. -/
theorem generated_acceptance_under_staged_refinement_implies_any_bad_or_execution
    {Bytes Parsed Program : Type}
    {decode : Bytes → Option Parsed}
    {terminalAccepts : Parsed → Prop}
    {occurs : BadEvent → Prop}
    {generated : GeneratedContext Program}
    (boundary : StagedExtraction decode terminalAccepts occurs
      generated.context)
    (accounting : LifetimeFailureAccounting occurs generated.context)
    {proof : Bytes}
    (accepted : Accepts decode terminalAccepts proof) :
    Or (AnyBad occurs)
      (HasSoundExecution generated.machine.semantics generated.statement.base
        generated.snapshotRoot) :=
  acceptance_under_staged_refinement_implies_any_bad_or_execution
    boundary accounting accepted

/-- Generated-artifact acceptance implies semantic execution when no exact
registered event occurs. -/
theorem generated_acceptance_under_staged_refinement_and_no_bad_implies_execution
    {Bytes Parsed Program : Type}
    {decode : Bytes → Option Parsed}
    {terminalAccepts : Parsed → Prop}
    {occurs : BadEvent → Prop}
    {generated : GeneratedContext Program}
    (boundary : StagedExtraction decode terminalAccepts occurs
      generated.context)
    (accounting : LifetimeFailureAccounting occurs generated.context)
    (noBad : ∀ event, ¬ occurs event)
    {proof : Bytes}
    (accepted : Accepts decode terminalAccepts proof) :
    HasSoundExecution generated.machine.semantics generated.statement.base
      generated.snapshotRoot :=
  acceptance_under_staged_refinement_and_no_bad_implies_execution
    boundary accounting noBad accepted

/-- Full WASM-result projection for one generated verifier artifact. -/
theorem generated_acceptance_under_staged_refinement_implies_wasm_result_or_failure
    {Bytes Parsed Program : Type}
    {decode : Bytes → Option Parsed}
    {terminalAccepts : Parsed → Prop}
    {occurs : BadEvent → Prop}
    {generated : GeneratedContext Program}
    (boundary : StagedExtraction decode terminalAccepts occurs
      generated.context)
    {proof : Bytes}
    (accepted : Accepts decode terminalAccepts proof) :
    Failure occurs generated.context \/
      (HasSoundExecution generated.machine.semantics generated.statement.base
          generated.snapshotRoot /\
        generated.statement.base.expectedResult.finalApplicationState.Terminal
          generated.statement.base.expectedResult.outcome /\
        generated.publicImage = PublicImage.ofStatement generated.statement) := by
  rcases
      generated_acceptance_under_staged_refinement_implies_certificate_or_failure
        boundary accepted
    with failure | certificate
  · exact Or.inl failure
  · rcases certificate with ⟨certificate⟩
    right
    have execution := certificate.semantic.semantic.execution
    have terminal :
        generated.statement.base.expectedResult.finalApplicationState.Terminal
          generated.statement.base.expectedResult.outcome := by
      rcases execution with
        ⟨_finalSnapshot, _segmentAccesses, applicationExecution,
          _memoryExecution, _segmentCount, _coverage, _finalRoot⟩
      exact generated.machine.completedExecution_final_terminal
        applicationExecution
    exact ⟨certificate.semantic.semantic.execution, terminal,
      generated.publicDecoded.exactImage⟩

end Nightstream.Assurance.NebulaV2.ProductionPaperReleasePipeline
