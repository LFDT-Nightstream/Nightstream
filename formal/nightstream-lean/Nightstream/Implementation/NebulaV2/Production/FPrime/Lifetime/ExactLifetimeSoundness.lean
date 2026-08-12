import Nightstream.Implementation.NebulaV2.Production.Memory.ChainSoundness
import Nightstream.Implementation.NebulaV2.Production.FPrime.Lifetime.ExactLifetime
import Nightstream.Implementation.NebulaV2.Production.FPrime.Lifetime.ClaimOpeningLifetimeFor
import Nightstream.Protocol.NebulaV2.Soundness

/-!
Contract: model-level soundness of one exponent-indexed, paper-exact
Nebula-on-SuperNeo F-prime lifetime.

The theorem starts from an exact semantic base node, an exact delayed tail,
and the producer-to-application port link. Several NIFS, memory, state, and
terminal facts inside those nodes are row-derived. The generated application
relation and the complete base-row composition are separate boundaries; this
theorem does not derive its nodes from a single generated relation manifest.

From those inputs, it derives one completed application execution and one
global memory execution over the same ordered accesses. The retained
extraction also contains the exact base, recursive, and trailing-terminal
claim schedule. Semantic extraction and claim verification use the same
ordered receipt list.

No premise supplies a completed execution, memory execution, global access
list, multiset balance, or the desired soundness conclusion. Every remaining
model failure is an explicit state-transcript collision, fingerprint
evaluation failure, memory-root mismatch, or snapshot-root collision.

This file does not own cryptographic probability bounds, generated-row
extraction, external-byte parsing, recursive-size closure, Rust refinement,
or a deployed terminal verifier.

Assurance tier: exponent-indexed semantic-model soundness bridge with named
row-derived components. It is not an artifact-checked relation theorem.

Emits constraints: no.
-/

set_option autoImplicit false
set_option maxRecDepth 100000

namespace Nightstream.Implementation.NebulaV2.ProductionPaperExactLifetimeSoundness

open Nightstream.Implementation.NebulaV2
open Nightstream.Implementation.NebulaV2.ProductionPaperExactLifetime
open Nightstream.Protocol.NebulaV2
open Nightstream.Protocol.NebulaV2.ApplicationTrace
open Nightstream.Protocol.NebulaV2.FPrime
open Nightstream.Protocol.NebulaV2.Soundness

/-- Memory-specific failures exposed by the exact row-derived lifetime. -/
abbrev MemoryFailure
    {Program : Type} (context : Context Program) :=
  ProductionMemoryChainSoundness.Failure context.Verifier
    context.headers context.snapshotRoot

/-- The two failure classes left at this model boundary. -/
inductive Failure
    {Program : Type} (context : Context Program) : Prop where
  | stateCollision (collision : context.Collision) : Failure context
  | memory (failure : MemoryFailure context) : Failure context

/-- Successful extraction retains both operational semantics and the exact
F-prime receipt schedule from the same tail. -/
structure CertifiedExecution
    {Program : Type} {context : Context Program}
    (base : BaseNode context) where
  extraction : LifetimeExtraction base
  execution : HasSoundExecution context.machine.semantics
    context.statement.base context.snapshotRoot

namespace CertifiedExecution

/-- The semantic certificate retains the exact verifier-derived authority
used to open the first memory segment.  This excludes a prover-selected base
challenge at the successful model boundary. -/
theorem baseChallengeAuthorityExact
    {Program : Type} {context : Context Program}
    {base : BaseNode context}
    (_certificate : CertifiedExecution base) :
    base.opening.authority =
      ProductionPaperBaseInvocationFor.challengeAuthority
        (rowVariables := context.rowVariables)
        (logicalWidth := context.logicalWidth)
        (publicFits := context.publicFits) context.candidate
        context.statementId context.headers context.statement base.opening
        base.batch :=
  base.challengeAuthorityExact

/-- The certificate retains the complete claim schedule, including the
separate trailing terminal consumer. -/
theorem exactClaimSchedule
    {Program : Type} {context : Context Program}
    {base : BaseNode context}
    (certificate : CertifiedExecution base) :
    ProductionPaperExactFPrimeLifetimeFor.Lifetime.ExactSchedule
      certificate.extraction.claimLifetime :=
  certificate.extraction.exactClaimSchedule

/-- The same certificate retains row-derived selection of the one fixed base
arm and every fixed recursive arm. -/
theorem fixedBranchSchedule
    {Program : Type} {context : Context Program}
    {base : BaseNode context}
    (certificate : CertifiedExecution base) :
    ProductionPaperExactFPrimeLifetimeFor.Lifetime.FixedBranchSchedule
      certificate.extraction.claimLifetime :=
  certificate.extraction.fixedBranchSchedule

/-- The receipt list used for memory semantics is the receipt list verified
by the retained F-prime schedule. -/
theorem receiptsExact
    {Program : Type} {context : Context Program}
    {base : BaseNode context}
    (certificate : CertifiedExecution base) :
    certificate.extraction.receipts =
      certificate.extraction.claimLifetime.consumedReceipts :=
  certificate.extraction.receipts_eq_consumedReceipts

/-- The certificate retains the row-derived invocation index of every actual
consumer node. The indexes are exactly `1, ..., T`, including the terminal
consumer at `T`. -/
theorem consumerInvocationIndicesExact
    {Program : Type} {context : Context Program}
    {base : BaseNode context}
    (certificate : CertifiedExecution base) :
    certificate.extraction.claimLifetime.schedule.consumerInvocationIndices =
      List.range' 1 certificate.extraction.receipts.length :=
  certificate.extraction.consumerInvocationIndices_exact

/-- The certificate retains equality of every complete produced state with
the complete prior state consumed next, including the terminal consumer. -/
theorem fullStateContinuityExact
    {Program : Type} {context : Context Program}
    {base : BaseNode context}
    (certificate : CertifiedExecution base) :
    certificate.extraction.claimLifetime.schedule.FullStateContinuous
      base.producer :=
  certificate.extraction.fullStateContinuityExact

/-- Every produced claim and every consumed receipt in the certified lifetime
has a CCS opening and relation witness in one exact assignment. -/
theorem exactClaimOpenings
    {Program : Type} {context : Context Program}
    {base : BaseNode context}
    (certificate : CertifiedExecution base) :
    ProductionPaperClaimOpeningLifetimeFor.ClaimLifetime.ExactOpenings
      certificate.extraction.claimLifetime :=
  ProductionPaperClaimOpeningLifetimeFor.ClaimLifetime.Lifetime.exactOpenings
    certificate.extraction.claimLifetime

/-- Each verified receipt bundle used by the certificate opens from one
row-derived assignment.  Use `everyConsumedClaimHolds` for the simultaneous
relation and public-input statement. -/
theorem everyConsumedBundleOpens
    {Program : Type} {context : Context Program}
    {base : BaseNode context}
    (certificate : CertifiedExecution base) :
    forall receipt,
      receipt ∈ certificate.extraction.claimLifetime.consumedReceipts ->
        exists assignment : context.claimLifecycle.FreshAssignment,
          ProductCommitmentAlgebra.commit context.config assignment =
            ProductNifsCodec.codecBundle
              receipt.claim.commitmentBundle :=
  ProductionPaperClaimOpeningLifetimeFor.ClaimLifetime.Lifetime.everyConsumedBundleOpens
    certificate.extraction.claimLifetime

/-- Each verified receipt used by the certificate has one assignment that
simultaneously opens the complete bundle, supplies the public input, meets the
fresh norm bound, and satisfies the selected relation. -/
theorem everyConsumedClaimHolds
    {Program : Type} {context : Context Program}
    {base : BaseNode context}
    (certificate : CertifiedExecution base) :
    forall receipt,
      receipt ∈ certificate.extraction.claimLifetime.consumedReceipts ->
        exists assignment : context.claimLifecycle.FreshAssignment,
          Nightstream.SuperNeo.CCS.Holds
            (ProductPaperAlgebraFor.semantics context.config)
            Nightstream.SuperNeo.Concrete.productionGlobalParams
            (ProductionPaperClaimOpeningLifetimeFor.ClaimLifetime.statementOfProtocolClaim
              context.claimLifecycle receipt.claim) assignment :=
  ProductionPaperClaimOpeningLifetimeFor.ClaimLifetime.Lifetime.everyConsumedClaimHolds
    certificate.extraction.claimLifetime

end CertifiedExecution

/-- Exact exponent-indexed semantic producer and consumer nodes imply the
complete public execution unless one named model or cryptographic-boundary
event occurs. The theorem derives, rather than assumes, both executions. It
does not derive the nodes from one complete generated row manifest. -/
theorem exact_lifetime_implies_certificate_or_failure
    {Program : Type} {context : Context Program}
    (base : BaseNode context)
    (tail : Tail context base.producer base.claim base.after base.produced) :
    Failure context \/ Nonempty (CertifiedExecution base) := by
  rcases base.extract_or_collision tail with
    ⟨⟨extracted⟩⟩ | stateCollision
  · rcases extracted.rowSegmentChain with
      ⟨allBatches, _receiptsExact, rowChain, _batchesExact, portsExact⟩
    rcases ProductionMemoryChainSoundness.Chain.executesOrFailure
        context.snapshotRoot rowChain
        context.publicDecoded.segmentCountPositive with
      memoryFailure | ⟨⟨memoryExecution⟩⟩
    · exact Or.inl (.memory memoryFailure)
    · rcases extracted.exactCompletedRun with ⟨completed⟩
      let canonicalInitial := Snapshot.ofImage
        context.statement.base.initialImage
      have equalInitialRoot :
          context.snapshotRoot memoryExecution.initialSnapshot =
            context.snapshotRoot canonicalInitial := by
        calc
          context.snapshotRoot memoryExecution.initialSnapshot =
              (ProductionPaperBaseInvocationFor.initialClosed
                context.authoritativeInitialMemoryRoot).memoryRoot :=
            memoryExecution.initialRoot
          _ = context.authoritativeInitialMemoryRoot := rfl
          _ = context.snapshotRoot canonicalInitial := rfl
      rcases IdealAcceptance.snapshot_eq_or_collision equalInitialRoot with
        initialExact | initialCollision
      · have accessListsExact :
            ProductionMemoryRowSegments.accesses allBatches =
              completed.execution.segmentAccesses.flatten := by
          calc
            ProductionMemoryRowSegments.accesses allBatches =
                ApplicationBatch.accesses extracted.applicationRows :=
              portsExact.symm
            _ = completed.execution.accesses := completed.accessesExact
            _ = completed.execution.segmentAccesses.flatten :=
              completed.execution.segmentAccesses_flatten.symm
        have memoryExecutes : Memory.Executes
            canonicalInitial.tuples 0
            completed.execution.segmentAccesses.flatten
            memoryExecution.finalSnapshot.tuples
            context.statement.base.finalGlobalTimestamp := by
          have exact := memoryExecution.executes
          rw [initialExact] at exact
          rw [accessListsExact, extracted.finalTimestamp] at exact
          simpa [canonicalInitial,
            ProductionPaperBaseInvocationFor.initialClosed] using exact
        have finalRootExact :
            context.statement.base.expectedResult.finalMemoryRoot =
              context.snapshotRoot memoryExecution.finalSnapshot := by
          calc
            context.statement.base.expectedResult.finalMemoryRoot =
                extracted.finalMemory.memoryRoot :=
              extracted.finalMemoryRoot.symm
            _ = context.snapshotRoot memoryExecution.finalSnapshot :=
              memoryExecution.finalRoot.symm
        exact Or.inr ⟨
          { extraction := extracted
            execution :=
              ⟨memoryExecution.finalSnapshot,
                completed.execution.segmentAccesses,
                completed.execution,
                memoryExecutes,
                completed.execution.segmentAccesses_length,
                rfl,
                finalRootExact⟩ }⟩
      · exact Or.inl (.memory (.snapshotCollision initialCollision))
  · exact Or.inl (.stateCollision stateCollision)

/-- Semantic projection for callers that do not need the retained claim and
receipt schedule. -/
theorem exact_lifetime_implies_execution_or_failure
    {Program : Type} {context : Context Program}
    (base : BaseNode context)
    (tail : Tail context base.producer base.claim base.after base.produced) :
    Failure context \/
      HasSoundExecution context.machine.semantics context.statement.base
        context.snapshotRoot := by
  rcases exact_lifetime_implies_certificate_or_failure base tail with
      failure | certificate
  · exact Or.inl failure
  · rcases certificate with ⟨certificate⟩
    exact Or.inr certificate.execution

end Nightstream.Implementation.NebulaV2.ProductionPaperExactLifetimeSoundness
