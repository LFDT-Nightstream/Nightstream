import Nightstream.Implementation.Nebula.Production.FPrime.Lifetime.ExactLifetimeNodes

/-!
Contract: global delayed-lifetime extraction for the production
Nebula-on-SuperNeo adaptation of the HyperNova Construction-2 relation.

The base invocation produces claim zero. Each recursive invocation consumes
the exact prior claim and produces one next claim. The terminal invocation
consumes the trailing claim and produces no successor. This module recovers
cross-invocation continuity, the exact application run, and the complete
memory schedule from the fixed local nodes.

The trailing consumer is an added V2 relation. The HyperNova paper terminal
checks the trailing fresh relation but performs no final NIFS fold.

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
noncomputable local instance concreteKFieldExtraction : Field K :=
  ConcreteField.superNeoEquiv.field

/-! ## Exact delayed lifetime syntax -/

/-- The lifetime contains only exact row-derived invocations. The producer to
consumer state link is deliberately absent and is recovered by the theorem. -/
inductive Tail
    {Program : Type} (context : Context Program) :
    (producer : context.Successor) -> (previous : context.Claim) ->
      AppStateVector -> ProducedBatch context producer previous -> Type
  | terminal
      {producer : context.Successor} {previous : context.Claim}
      {applicationAfter : AppStateVector}
      {produced : ProducedBatch context producer previous}
      (node : TerminalNode context previous) :
      Tail context producer previous applicationAfter produced
  | recursive
      {producer : context.Successor} {previous : context.Claim}
      {applicationAfter : AppStateVector}
      {produced : ProducedBatch context producer previous}
      (node : RecursiveNode context previous)
      (rest : Tail context node.nextProducer node.nextClaim
        node.nextApplication node.currentProduced) :
      Tail context producer previous applicationAfter produced

namespace Tail

/-- Erase semantic extraction data while retaining the exact base-independent
claim schedule. The terminal constructor remains mandatory. -/
noncomputable def claimSchedule
    {Program : Type} {context : Context Program}
    {producer : context.Successor} {previous : context.Claim}
    {applicationAfter : AppStateVector}
    {produced : ProducedBatch context producer previous}
    (tail : Tail context producer previous applicationAfter produced) :
    ProductionPaperExactFPrimeLifetimeFor.Schedule
      context.claimLifecycle previous := by
  induction tail with
  | terminal node =>
      exact ProductionPaperExactFPrimeLifetimeFor.TerminalNode.finish
        node.toClaimNode
  | recursive node rest inductionHypothesis =>
      exact ProductionPaperExactFPrimeLifetimeFor.RecursiveNode.prepend
        node.toClaimNode inductionHypothesis

/-- Receipts used by semantic extraction are the receipts of the exact
claim schedule. This definition prevents the semantic memory trace from
selecting a second, unrelated verifier transcript. -/
noncomputable def receipts
    {Program : Type} {context : Context Program}
    {producer : context.Successor} {previous : context.Claim}
    {applicationAfter : AppStateVector}
    {produced : ProducedBatch context producer previous}
    (tail : Tail context producer previous applicationAfter produced) :
    List context.Receipt :=
  ExactDelayedSchedule.Schedule.receipts tail.claimSchedule

/-- Invocation indexes parsed from the exact prior state consumed by each
post-base node. The list includes the terminal consumer. It is not a synthetic
index list derived only from the number of claims. -/
noncomputable def consumerInvocationIndices
    {Program : Type} {context : Context Program}
    {producer : context.Successor} {previous : context.Claim}
    {applicationAfter : AppStateVector}
    {produced : ProducedBatch context producer previous} :
    Tail context producer previous applicationAfter produced -> List Nat
  | .terminal node =>
      [node.recursive.priorState.augmentedInvocationIndex]
  | .recursive node rest =>
      node.recursive.priorState.augmentedInvocationIndex ::
        consumerInvocationIndices rest

@[simp] theorem receipts_terminal
    {Program : Type} {context : Context Program}
    {producer : context.Successor} {previous : context.Claim}
    {applicationAfter : AppStateVector}
    {produced : ProducedBatch context producer previous}
    (node : TerminalNode context previous) :
    receipts (Tail.terminal (producer := producer)
      (applicationAfter := applicationAfter) (produced := produced) node) =
        [node.recursive.verified] := by
  rfl

@[simp] theorem receipts_recursive
    {Program : Type} {context : Context Program}
    {producer : context.Successor} {previous : context.Claim}
    {applicationAfter : AppStateVector}
    {produced : ProducedBatch context producer previous}
    (node : RecursiveNode context previous)
    (rest : Tail context node.nextProducer node.nextClaim
      node.nextApplication node.currentProduced) :
    receipts (Tail.recursive (producer := producer)
      (applicationAfter := applicationAfter) (produced := produced) node
      rest) = node.recursive.verified :: receipts rest := by
  rfl

@[simp] theorem consumerInvocationIndices_terminal
    {Program : Type} {context : Context Program}
    {producer : context.Successor} {previous : context.Claim}
    {applicationAfter : AppStateVector}
    {produced : ProducedBatch context producer previous}
    (node : TerminalNode context previous) :
    consumerInvocationIndices (Tail.terminal (producer := producer)
      (applicationAfter := applicationAfter) (produced := produced) node) =
        [node.recursive.priorState.augmentedInvocationIndex] := by
  rfl

@[simp] theorem consumerInvocationIndices_recursive
    {Program : Type} {context : Context Program}
    {producer : context.Successor} {previous : context.Claim}
    {applicationAfter : AppStateVector}
    {produced : ProducedBatch context producer previous}
    (node : RecursiveNode context previous)
    (rest : Tail context node.nextProducer node.nextClaim
      node.nextApplication node.currentProduced) :
    consumerInvocationIndices (Tail.recursive (producer := producer)
      (applicationAfter := applicationAfter) (produced := produced) node
      rest) = node.recursive.priorState.augmentedInvocationIndex ::
        consumerInvocationIndices rest := by
  rfl

/-- There is one parsed consumer index for each verified complete claim,
including the trailing claim consumed by the terminal node. -/
theorem consumerInvocationIndices_length
    {Program : Type} {context : Context Program}
    {producer : context.Successor} {previous : context.Claim}
    {applicationAfter : AppStateVector}
    {produced : ProducedBatch context producer previous}
    (tail : Tail context producer previous applicationAfter produced) :
    tail.consumerInvocationIndices.length = tail.receipts.length := by
  induction tail with
  | terminal node => rfl
  | recursive node rest inductionHypothesis =>
      simp only [consumerInvocationIndices_recursive, receipts_recursive,
        List.length_cons, inductionHypothesis]

/-- Erasing semantic extraction data preserves the actual prior-state index
read by every recursive and terminal consumer. -/
theorem claimSchedule_consumerInvocationIndices
    {Program : Type} {context : Context Program}
    {producer : context.Successor} {previous : context.Claim}
    {applicationAfter : AppStateVector}
    {produced : ProducedBatch context producer previous}
    (tail : Tail context producer previous applicationAfter produced) :
    tail.claimSchedule.consumerInvocationIndices =
      tail.consumerInvocationIndices := by
  induction tail with
  | terminal node => rfl
  | recursive node rest inductionHypothesis =>
      change
        node.toClaimNode.recursive.priorState.augmentedInvocationIndex ::
            rest.claimSchedule.consumerInvocationIndices =
          node.recursive.priorState.augmentedInvocationIndex ::
            rest.consumerInvocationIndices
      rw [RecursiveNode.toClaimNode_priorInvocationIndex,
        inductionHypothesis]

end Tail

/-! ## Extracted global result -/

/-- Result obtained after every exact cross-invocation link is recovered.
Its application relation starts after the indexed producer. -/
structure Extraction
    {Program : Type} {context : Context Program}
    {producer : context.Successor} {previous : context.Claim}
    {applicationAfter : AppStateVector}
    (authority : ProducerAuthority context producer previous applicationAfter)
    (produced : ProducedBatch context producer previous)
    (scheduledReceipts : List context.Receipt) where
  applicationRows : List ApplicationTrace.ApplicationRow
  receipts : List context.Receipt
  receiptsExact : receipts = scheduledReceipts
  finalMemory : ClosedCarry Digest.Value
  application : Runs context.machine context.statement.base.program
    applicationAfter applicationRows
    context.statement.base.expectedResult.finalApplicationState
    (realRowCount applicationRows)
  rowBefore : ProductionMemoryStepSemantics.ConcreteCarry
  rowDelayed : ProductionMemoryRowTrace.DelayedRun context.Verifier
    context.headers rowBefore receipts finalMemory
  rowBeforeExact : rowBefore = producerCarry authority
  portsExact :
    ProductionApplicationBatchBridge.memoryAccesses produced.result ++
        ApplicationBatch.accesses applicationRows =
      rowDelayed.accesses
  rowLengthAccounting : applicationRows.length +
      ApplicationBatch.rowsPerFreshClaim context.candidate =
    receipts.length * ApplicationBatch.rowsPerFreshClaim context.candidate
  realRowsAccounting : producer.realApplicationRowCount +
    realRowCount applicationRows =
      context.statement.base.expectedResult.realApplicationRowCount
  claimCountAccounting : producer.augmentedInvocationIndex + receipts.length =
    context.statement.base.segmentCount * claimsPerSegment context.candidate + 1
  finalSegment : finalMemory.segmentIndex = context.statement.base.segmentCount
  finalTimestamp : finalMemory.globalTimestamp =
    context.statement.base.finalGlobalTimestamp
  finalMemoryRoot : finalMemory.memoryRoot =
    context.statement.base.expectedResult.finalMemoryRoot

/-! ## Completion row boundary -/

/-- Exact typed row shape decoded from the generated application rows. This
does not assume an operational run or a completed execution. -/
structure CompletionRows
    {Program : Type} (context : Context Program)
    (rows : List ApplicationTrace.ApplicationRow) where
  activeRows : List Ports.NormalizedRow
  terminalRow : Ports.NormalizedRow
  rowsExact : rows = activeRows.map ApplicationTrace.ApplicationRow.active ++
    [ApplicationTrace.terminalApplicationRow terminalRow
      context.statement.base.expectedResult.outcome] ++
    List.replicate
      (Completion.segmentCapacity context.statement.base.segmentCount -
        context.statement.base.expectedResult.realApplicationRowCount)
      .padding
  realRowCountExact : context.statement.base.expectedResult.realApplicationRowCount =
    activeRows.length + 1
  segmentCountPositive : 0 < context.statement.base.segmentCount
  segmentCountBound : context.statement.base.segmentCount <=
    Lifecycle.maximumSegments
  realRowCountBound :
    context.statement.base.expectedResult.realApplicationRowCount <
      Completion.realApplicationRowLimit
  fitsDeclaredSegments :
    context.statement.base.expectedResult.realApplicationRowCount <=
      Completion.segmentCapacity context.statement.base.segmentCount
  smallestSegmentCount : context.statement.base.segmentCount =
    Completion.minimumSegmentCount
      context.statement.base.expectedResult.realApplicationRowCount

/-! ## Local cross-invocation reductions -/

/-- Multiplication transported through the proved field equivalence is the
same value as the executable SuperNeo multiplication. -/
theorem transferred_mul_eq_concrete (left right : K) :
    left * right = K.mul left right := by
  apply ConcreteField.superNeoEquiv.injective
  change ConcreteField.superNeoEquiv left *
      ConcreteField.superNeoEquiv right =
    ConcreteField.superNeoEquiv (K.mul left right)
  exact (ConcreteField.superNeoEquiv_mul left right).symm

/-- The transferred field multiplication is the concrete `K.mul` operation,
so row-derived close balance is the protocol balance predicate. -/
theorem concreteBalanced_implies_balanced
    {products : ProductState.State K}
    (balanced : MemoryProductBalanceRows.ConcreteBalanced products) :
    ProductState.Balanced products := by
  intro repetition
  change (products repetition).initialSnapshot *
      (products repetition).writes =
    (products repetition).reads *
      (products repetition).finalSnapshot
  rw [transferred_mul_eq_concrete, transferred_mul_eq_concrete]
  exact balanced repetition

/-- Producer rows and consumer replay rows for the same accepted claim have
the same final carry once their start boundary is equal. Semantic execution
therefore uses producer records without changing the delayed F-prime state. -/
theorem ProducedBatch.after_eq_consumer
    {Program : Type} {context : Context Program}
    {producer : context.Successor} {claim : context.Claim}
    (produced : ProducedBatch context producer claim)
    (receipt : context.Receipt)
    (claimExact : receipt.claim = claim.toProtocolClaim
      (NifsProof := ProductionProductPiCcsTypedBridgeFor.ExactProof
        context.rowVariables))
    {consumerBefore consumerAfter :
      Carry Digest.Value (ProductState.Challenges K) (ProductState.State K)}
    (consumerTransition : ProductionBatchedFPrime.Transition
      context.Verifier MemoryProductBalanceRows.ConcreteBalanced
      consumerBefore receipt consumerAfter)
    (beforeExact : produced.result.semantic 0 = consumerBefore) :
    produced.result.semantic
        (Fin.last (ProductionMemoryCheckedBatchRows.StepCount
          context.candidate)) = consumerAfter := by
  let producerEvidence := produced.rowEvidence receipt claimExact
  have producerTransition := producerEvidence.transition
  have producerConsumes := producerTransition.consumes
  change ProductionBatchedFPrime.ConsumesList ProductState.Balanced
      (produced.result.semantic 0) receipt.claim.memory.suffixes
      (produced.result.semantic
        (Fin.last (ProductionMemoryCheckedBatchRows.StepCount
          context.candidate))) at producerConsumes
  have consumerTransitionBalanced : ProductionBatchedFPrime.Transition
      context.Verifier ProductState.Balanced consumerBefore receipt
      consumerAfter :=
    ⟨consumerTransition.consumes.mono
      (fun products balanced => concreteBalanced_implies_balanced balanced)⟩
  rw [beforeExact] at producerConsumes
  exact ProductionBatchedFPrime.ConsumesList.after_unique
    producerConsumes consumerTransitionBalanced.consumes

/-- Semantic carry decoding is independent of the proof object used for the
step-index bound. -/
theorem semanticCarry_congr
    {left right : MemoryCarryCodec.Value}
    (equal : left = right)
    (leftBound : left.stepIndex < Lifecycle.claimsPerSegment)
    (rightBound : right.stepIndex < Lifecycle.claimsPerSegment) :
    MemoryCarryParser.semanticCarry left leftBound =
      MemoryCarryParser.semanticCarry right rightBound := by
  subst right
  rfl

/-- Equal complete successor states make the authenticated producer carry the
exact boundary-zero carry consumed by the next row-derived memory batch. -/
theorem producerCarry_eq_consumerStart
    {Program : Type} {context : Context Program}
    {producer : context.Successor} {previous : context.Claim}
    {applicationAfter : AppStateVector}
    (authority : ProducerAuthority context producer previous applicationAfter)
    {priorAuthority : ProductionPaperPriorStateAuthorityRowsFor.Layout
      context.candidate context.rowVariables}
    {assignment : Nat -> Nat}
    {priorPrefix : ProductionPaperPriorStateAuthorityRowsFor.Prefix
      context.candidate context.FullShape}
    {proof : ProductionProductPiCcsTypedBridgeFor.ExactProof
      context.rowVariables}
    (recursive : ProductionPaperRecursiveRelationRowsSoundFor.Result
      context.candidate context.statementId context.config context.artifact
      priorAuthority assignment context.headers priorPrefix previous proof)
    (stateExact : producer = recursive.priorState) :
    producerCarry authority = recursive.memoryResult.semantic 0 := by
  rw [recursive.memoryResult.semanticExact]
  unfold producerCarry
  have carryExact :=
    recursive.priorAuthorityResult.prior_memoryCarry_eq_memory_start
  apply semanticCarry_congr
  exact (congrArg ProductionSuccessorStateBinding.Value.memoryCarry
    stateExact).trans carryExact

/-- One recursive node either consumes the state produced by the preceding
claim exactly or identifies the named state-transcript collision. -/
theorem RecursiveNode.state_equal_or_collision
    {Program : Type} {context : Context Program}
    {producer : context.Successor} {previous : context.Claim}
    {applicationAfter : AppStateVector}
    (authority : ProducerAuthority context producer previous applicationAfter)
    (node : RecursiveNode context previous) :
    producer = node.recursive.priorState \/ context.Collision := by
  exact ProductionPaperStateContinuityFor.state_equal_or_collision
    authority.canonical node.recursive.priorAuthorityResult.priorPlaced
    node.supplement.evidence.assignmentCanonical node.supplement.evidence.one
    authority.fullMatches node.recursive.ccsFullMatches

/-- The same reduction applies to the trailing terminal consumer. -/
theorem TerminalNode.state_equal_or_collision
    {Program : Type} {context : Context Program}
    {producer : context.Successor} {previous : context.Claim}
    {applicationAfter : AppStateVector}
    (authority : ProducerAuthority context producer previous applicationAfter)
    (node : TerminalNode context previous) :
    producer = node.recursive.priorState \/ context.Collision := by
  exact ProductionPaperStateContinuityFor.state_equal_or_collision
    authority.canonical node.recursive.priorAuthorityResult.priorPlaced
    node.exact.assignmentCanonical node.exact.one authority.fullMatches
    node.recursive.ccsFullMatches

/-- The indexes read from the actual row-derived prior states are consecutive
from the authenticated producer index. This is the missing link between the
claim-order schedule and the invocation counters inside the recursive rows.
If a consumer names a different prior state, the result is the named state
transcript collision, not a synthetic schedule with repaired indexes. -/
theorem Tail.consumerInvocationIndices_or_collision
    {Program : Type} {context : Context Program}
    {producer : context.Successor} {previous : context.Claim}
    {applicationAfter : AppStateVector}
    {produced : ProducedBatch context producer previous}
    (tail : Tail context producer previous applicationAfter produced)
    (authority : ProducerAuthority context producer previous applicationAfter) :
    tail.consumerInvocationIndices =
        List.range' producer.augmentedInvocationIndex tail.receipts.length \/
      context.Collision := by
  induction tail with
  | @terminal producer previous applicationAfter produced node =>
      rcases node.state_equal_or_collision authority with stateExact | collision
      · left
        have indexExact :
            node.recursive.priorState.augmentedInvocationIndex =
              producer.augmentedInvocationIndex := by
          exact congrArg
            ProductionSuccessorStateBinding.Value.augmentedInvocationIndex
            stateExact.symm
        rw [Tail.consumerInvocationIndices_terminal, Tail.receipts_terminal,
          indexExact]
        rfl
      · exact Or.inr collision
  | @recursive producer previous applicationAfter produced node rest
      inductionHypothesis =>
      rcases node.state_equal_or_collision authority with stateExact | collision
      · rcases inductionHypothesis node.nextAuthority with
          restExact | laterCollision
        · left
          have currentIndex :
              node.recursive.priorState.augmentedInvocationIndex =
                producer.augmentedInvocationIndex := by
            exact congrArg
              ProductionSuccessorStateBinding.Value.augmentedInvocationIndex
              stateExact.symm
          have nextIndex :
              node.nextProducer.augmentedInvocationIndex =
                producer.augmentedInvocationIndex + 1 := by
            simp [RecursiveNode.nextProducer,
              ProductionPaperRecursiveInvocationRowsSoundFor.Supplement.successor,
              ProductionRecursiveSuccessorFor.value, stateExact]
          rw [Tail.consumerInvocationIndices_recursive,
            Tail.receipts_recursive]
          simp only [List.length_cons, List.range'_succ]
          rw [currentIndex, restExact, nextIndex]
        · exact Or.inr laterCollision
      · exact Or.inr collision

/-- The exact claim schedule retains equality of every complete producer
state with the complete prior state read by its recursive or terminal
consumer. A mismatch is not repaired by equal counters; it is the named
state-transcript collision. -/
theorem Tail.fullStateContinuity_or_collision
    {Program : Type} {context : Context Program}
    {producer : context.Successor} {previous : context.Claim}
    {applicationAfter : AppStateVector}
    {produced : ProducedBatch context producer previous}
    (tail : Tail context producer previous applicationAfter produced)
    (authority : ProducerAuthority context producer previous applicationAfter) :
    tail.claimSchedule.FullStateContinuous producer \/ context.Collision := by
  induction tail with
  | @terminal producer previous applicationAfter produced node =>
      rcases node.state_equal_or_collision authority with stateExact | collision
      · exact Or.inl stateExact
      · exact Or.inr collision
  | @recursive producer previous applicationAfter produced node rest
      inductionHypothesis =>
      rcases node.state_equal_or_collision authority with stateExact | collision
      · rcases inductionHypothesis node.nextAuthority with
          restExact | laterCollision
        · exact Or.inl ⟨stateExact, restExact⟩
        · exact Or.inr laterCollision
      · exact Or.inr collision

/-- Every reachable nonterminal invocation selects the recursive arm of the
one verifier-owned F-prime relation. A failed state link is reported as the
named state-transcript collision; it is never treated as a recursive branch
certificate. -/
theorem Tail.fixedRecursiveBranches_or_collision
    {Program : Type} {context : Context Program}
    {producer : context.Successor} {previous : context.Claim}
    {applicationAfter : AppStateVector}
    {produced : ProducedBatch context producer previous}
    (tail : Tail context producer previous applicationAfter produced)
    (authority : ProducerAuthority context producer previous applicationAfter)
    (producerPositive : 0 < producer.augmentedInvocationIndex) :
    tail.claimSchedule.FixedRecursiveBranches \/ context.Collision := by
  induction tail with
  | @terminal producer previous applicationAfter produced node =>
      exact Or.inl trivial
  | @recursive producer previous applicationAfter produced node rest
      inductionHypothesis =>
      rcases node.state_equal_or_collision authority with stateExact | collision
      · have priorPositive :
            0 < node.recursive.priorState.augmentedInvocationIndex := by
          rw [← stateExact]
          exact producerPositive
        have current :=
          ProductionPaperExactFPrimeLifetimeFor.RecursiveNode.freshSelectsFixedRecursive
            node.toClaimNode priorPositive
        have nextPositive :
            0 < node.nextProducer.augmentedInvocationIndex := by
          simp [RecursiveNode.nextProducer,
            ProductionPaperRecursiveInvocationRowsSoundFor.Supplement.successor,
            ProductionRecursiveSuccessorFor.value]
        rcases inductionHypothesis node.nextAuthority nextPositive with
          restExact | laterCollision
        · exact Or.inl ⟨current.1, current.2, restExact⟩
        · exact Or.inr laterCollision
      · exact Or.inr collision

/-! ## Terminal extraction -/

private theorem terminal_application_exact
    {Program : Type} {context : Context Program}
    {producer : context.Successor} {previous : context.Claim}
    {applicationAfter : AppStateVector}
    (authority : ProducerAuthority context producer previous applicationAfter)
    (node : TerminalNode context previous)
    (stateExact : producer = node.recursive.priorState) :
    applicationAfter =
      context.statement.base.expectedResult.finalApplicationState := by
  apply WasmStateEncoding.encode_injective
  calc
    WasmStateEncoding.encode applicationAfter = producer.applicationState :=
      authority.applicationExact.symm
    _ = node.recursive.priorState.applicationState := by
      exact congrArg ProductionSuccessorStateBinding.Value.applicationState
        stateExact
    _ = context.statement.resultImage.finalApplicationState :=
      node.exact.publicResult.finalApplication
    _ = WasmStateEncoding.encode
        context.statement.base.expectedResult.finalApplicationState := by
      exact congrArg ResultImage.finalApplicationState
        context.statement.resultDecoded.exactImage

private theorem terminal_real_rows_exact
    {Program : Type} {context : Context Program}
    {producer : context.Successor} {previous : context.Claim}
    {applicationAfter : AppStateVector}
    (node : TerminalNode context previous)
    (stateExact : producer = node.recursive.priorState) :
    producer.realApplicationRowCount =
      context.statement.base.expectedResult.realApplicationRowCount := by
  calc
    producer.realApplicationRowCount =
        node.recursive.priorState.realApplicationRowCount := by
      exact congrArg
        ProductionSuccessorStateBinding.Value.realApplicationRowCount stateExact
    _ = context.statement.resultImage.realApplicationRowCount :=
      node.exact.publicResult.realApplicationRows
    _ = context.statement.base.expectedResult.realApplicationRowCount := by
      exact congrArg ResultImage.realApplicationRowCount
        context.statement.resultDecoded.exactImage

private theorem terminal_invocation_exact
    {Program : Type} {context : Context Program}
    {producer : context.Successor} {previous : context.Claim}
    {applicationAfter : AppStateVector}
    (node : TerminalNode context previous)
    (stateExact : producer = node.recursive.priorState) :
    producer.augmentedInvocationIndex =
      context.statement.base.segmentCount *
        claimsPerSegment context.candidate := by
  calc
    producer.augmentedInvocationIndex =
        node.recursive.priorState.augmentedInvocationIndex := by
      exact congrArg
        ProductionSuccessorStateBinding.Value.augmentedInvocationIndex
        stateExact
    _ = context.statement.base.segmentCount *
        claimsPerSegment context.candidate :=
      node.exact.publicResult.invocationIndex

/-- The trailing consumer either identifies a state collision or gives the
complete terminal extraction. No terminal execution is assumed. -/
theorem terminal_extract_or_collision
    {Program : Type} {context : Context Program}
    {producer : context.Successor} {previous : context.Claim}
    {applicationAfter : AppStateVector}
    (authority : ProducerAuthority context producer previous applicationAfter)
    (produced : ProducedBatch context producer previous)
    (node : TerminalNode context previous) :
    Nonempty (Extraction authority produced [node.recursive.verified]) \/
      context.Collision := by
  rcases node.state_equal_or_collision authority with stateExact | collision
  · have applicationExact := terminal_application_exact authority node stateExact
    have consumerStart :=
      producerCarry_eq_consumerStart authority node.recursive stateExact
    have producerStart := produced.before_eq_producerCarry authority
    have startExact : produced.result.semantic 0 =
        node.recursive.memoryResult.semantic 0 :=
      producerStart.trans consumerStart
    have afterExact := produced.after_eq_consumer node.recursive.verified
      node.recursive.claimExact node.recursive.transition startExact
    let finalMemory :=
      ProductionPaperTerminalInvocationRowsSoundFor.finalClosed
        node.recursive.memoryResult
    let memoryBatch : ProductionMemoryRowTrace.BatchEvidence
        context.candidate context.Schema context.Verifier context.headers :=
      produced.rowEvidence node.recursive.verified node.recursive.claimExact
    have closedExact : memoryBatch.after = .closed finalMemory :=
      afterExact.trans node.exact.finalSemantic
    refine Or.inl ⟨
      { applicationRows := []
        receipts := [node.recursive.verified]
        receiptsExact := rfl
        finalMemory := finalMemory
        application := ?_
        rowBefore := memoryBatch.before
        rowDelayed := .terminal memoryBatch finalMemory closedExact
        rowBeforeExact := producerStart
        portsExact := by
          simp only [ApplicationBatch.accesses, List.flatMap_nil,
            List.append_nil]
          calc
            ProductionApplicationBatchBridge.memoryAccesses produced.result =
                ProductionMemoryRowTrace.BatchEvidence.accesses memoryBatch :=
              rfl
            _ = (ProductionMemoryRowTrace.DelayedRun.terminal memoryBatch
                  finalMemory closedExact).accesses :=
              (ProductionMemoryRowTrace.DelayedRun.accesses_terminal
                memoryBatch finalMemory closedExact).symm
        rowLengthAccounting := by simp
        realRowsAccounting := ?_
        claimCountAccounting := ?_
        finalSegment := ?_
        finalTimestamp := ?_
        finalMemoryRoot := ?_ }⟩
    · simpa [applicationExact] using
        (Runs.nil applicationAfter)
    · simpa [ApplicationBatch.realRowCount] using
        terminal_real_rows_exact (applicationAfter := applicationAfter) node
          stateExact
    · have invocationExact :=
        terminal_invocation_exact (applicationAfter := applicationAfter) node
          stateExact
      simpa [invocationExact]
    · exact node.exact.publicResult.finalSegment
    · exact node.exact.publicResult.finalTimestamp
    · calc
        finalMemory.memoryRoot =
            context.statement.resultImage.finalMemoryRoot :=
          node.exact.publicResult.finalMemoryRoot
        _ = context.statement.base.expectedResult.finalMemoryRoot := by
          exact congrArg ResultImage.finalMemoryRoot
            context.statement.resultDecoded.exactImage
  · exact Or.inr collision

/-! ## Recursive lifetime extraction -/

/-- Every exact delayed tail either reconstructs its complete application and
memory lifetime or identifies the one named cross-invocation state collision.
The induction includes the trailing terminal claim. -/
theorem Tail.extract_or_collision
    {Program : Type} {context : Context Program}
    {producer : context.Successor} {previous : context.Claim}
    {applicationAfter : AppStateVector}
    {produced : ProducedBatch context producer previous}
    (tail : Tail context producer previous applicationAfter produced)
    (authority : ProducerAuthority context producer previous applicationAfter) :
    Nonempty (Extraction authority produced tail.receipts) \/
      context.Collision := by
  induction tail with
  | @terminal producer previous applicationAfter produced node =>
      simpa using terminal_extract_or_collision authority produced node
  | @recursive producer previous applicationAfter produced node rest
      inductionHypothesis =>
      rcases node.state_equal_or_collision authority with stateExact | collision
      · have applicationStart :
          WasmStateEncoding.decode node.recursive.priorState.applicationState =
            applicationAfter := by
          calc
            WasmStateEncoding.decode
                node.recursive.priorState.applicationState =
                WasmStateEncoding.decode producer.applicationState := by
              exact congrArg
                (fun state => WasmStateEncoding.decode
                  state.applicationState) stateExact.symm
            _ = WasmStateEncoding.decode
                (WasmStateEncoding.encode applicationAfter) := by
              exact congrArg WasmStateEncoding.decode
                authority.applicationExact
            _ = applicationAfter := WasmStateEncoding.decode_encode _
        have currentApplication : Runs context.machine
            context.statement.base.program applicationAfter
            node.applicationRows node.nextApplication
            (realRowCount node.applicationRows) := by
          simpa [RecursiveNode.applicationRows,
            RecursiveNode.nextApplication, applicationStart] using
            node.exact.previousConsumed.applicationRun
        have consumerStart :=
          producerCarry_eq_consumerStart authority node.recursive stateExact
        have producerStart := produced.before_eq_producerCarry authority
        have startExact : produced.result.semantic 0 =
            node.recursive.memoryResult.semantic 0 :=
          producerStart.trans consumerStart
        have afterExact := produced.after_eq_consumer
          node.recursive.verified node.recursive.claimExact
            node.recursive.transition startExact
        have continues :=
          node.supplement.evidence.memory_continues_from_rows
        have outgoingExact :
            MemoryCarryParser.semanticCarry node.supplement.evidence.outgoing
                node.supplement.evidence.outgoingParsed.parserCanonical.stepIndex =
              producerCarry (node.nextAuthority) := by
          unfold producerCarry
          apply semanticCarry_congr
          rfl
        have continuesToNext : ProductionMemoryRowTrace.BoundContinuation
            context.candidate
            context.headers
            (produced.result.semantic
              (Fin.last (ProductionMemoryCheckedBatchRows.StepCount
                context.candidate)))
            (producerCarry (node.nextAuthority)) :=
          { authority := node.supplement.evidence.authority
            exact := by
              simpa [outgoingExact, afterExact] using continues }
        rcases inductionHypothesis node.nextAuthority with
          ⟨⟨extracted⟩⟩ | laterCollision
        · let rows := node.applicationRows ++ extracted.applicationRows
          let receipts := node.recursive.verified :: extracted.receipts
          have nextPortsExact :
              ApplicationBatch.accesses node.applicationRows =
                ProductionApplicationBatchBridge.memoryAccesses
                  node.currentProduced.result := by
            simpa [RecursiveNode.applicationRows,
              RecursiveNode.currentProduced] using
              node.exact.currentPortsExact
          have applicationRun : Runs context.machine
              context.statement.base.program applicationAfter rows
              context.statement.base.expectedResult.finalApplicationState
              (realRowCount rows) := by
            have appended := currentApplication.append extracted.application
            simpa [rows, ApplicationBatchCompletion.realRowCount_append] using
              appended
          let memoryBatch : ProductionMemoryRowTrace.BatchEvidence
              context.candidate context.Schema context.Verifier context.headers :=
            produced.rowEvidence node.recursive.verified
              node.recursive.claimExact
          have continuesToRest : ProductionMemoryRowTrace.BoundContinuation
              context.candidate
              context.headers
              (produced.result.semantic
                (Fin.last (ProductionMemoryCheckedBatchRows.StepCount
                  context.candidate))) extracted.rowBefore :=
            { authority := continuesToNext.authority
              exact := by
                rw [extracted.rowBeforeExact]
                exact continuesToNext.exact }
          let rowRecursive : ProductionMemoryRowTrace.DelayedRun context.Verifier
              context.headers memoryBatch.before receipts
              extracted.finalMemory :=
            .recursive memoryBatch continuesToRest extracted.rowDelayed
          have nextRows : node.nextProducer.realApplicationRowCount =
              producer.realApplicationRowCount +
                realRowCount node.applicationRows := by
            simp [RecursiveNode.nextProducer,
              ProductionPaperRecursiveInvocationRowsSoundFor.Supplement.successor,
              ProductionRecursiveSuccessorFor.value,
              RecursiveNode.applicationRows, stateExact]
          have nextInvocation : node.nextProducer.augmentedInvocationIndex =
              producer.augmentedInvocationIndex + 1 := by
            simp [RecursiveNode.nextProducer,
              ProductionPaperRecursiveInvocationRowsSoundFor.Supplement.successor,
              ProductionRecursiveSuccessorFor.value, stateExact]
          refine Or.inl ⟨
            { applicationRows := rows
              receipts := receipts
              receiptsExact := by
                simp only [receipts]
                rw [extracted.receiptsExact]
                rfl
              finalMemory := extracted.finalMemory
              application := applicationRun
              rowBefore := memoryBatch.before
              rowDelayed := rowRecursive
              rowBeforeExact := producerStart
              portsExact := ?_
              rowLengthAccounting := ?_
              realRowsAccounting := ?_
              claimCountAccounting := ?_
              finalSegment := extracted.finalSegment
              finalTimestamp := extracted.finalTimestamp
              finalMemoryRoot := extracted.finalMemoryRoot }⟩
          · rw [show ApplicationBatch.accesses rows =
                  ApplicationBatch.accesses node.applicationRows ++
                    ApplicationBatch.accesses extracted.applicationRows by
                exact ApplicationBatch.accesses_append _ _]
            rw [show rowRecursive.accesses =
                ProductionMemoryRowTrace.BatchEvidence.accesses memoryBatch ++
                  extracted.rowDelayed.accesses by
              exact ProductionMemoryRowTrace.DelayedRun.accesses_recursive
                memoryBatch continuesToRest extracted.rowDelayed]
            change
              ProductionApplicationBatchBridge.memoryAccesses produced.result ++
                    (ApplicationBatch.accesses node.applicationRows ++
                      ApplicationBatch.accesses extracted.applicationRows) =
                ProductionMemoryRowTrace.BatchEvidence.accesses memoryBatch ++
                  extracted.rowDelayed.accesses
            rw [nextPortsExact, extracted.portsExact]
            rfl
          · simp only [rows, receipts, List.length_append, List.length_cons]
            have nodeRowsExact : node.applicationRows.length =
                ApplicationBatch.rowsPerFreshClaim context.candidate := by
              simpa [RecursiveNode.applicationRows] using
                node.supplement.evidence.batch.rowsExact
            rw [nodeRowsExact]
            rw [Nat.add_assoc, extracted.rowLengthAccounting]
            rw [Nat.add_mul, Nat.one_mul]
            exact Nat.add_comm _ _
          · simp only [rows,
              ApplicationBatchCompletion.realRowCount_append]
            rw [← Nat.add_assoc, ← nextRows]
            exact extracted.realRowsAccounting
          · simp only [receipts, List.length_cons]
            have restClaimCount := extracted.claimCountAccounting
            omega
        · exact Or.inr laterCollision
      · exact Or.inr collision

/-! ## Canonical base and complete lifetime -/

/-- The exact Construction-2 base invocation. Its type has no prior claim or
prior verifier proof. -/
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
  initialMemoryRootExact :
    baseRows.call.initialMemoryRoot = context.authoritativeInitialMemoryRoot

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
    {Program : Type} {context : Context Program}
    (node : BaseNode context) :
    ProductionBaseCurrentMemoryRowsFor.Authority context.candidate
      node.baseRows.artifact :=
  node.baseArtifactExact.symm ▸ context.baseMemoryAuthority

/-- Claim-zero memory data is derived from the satisfying base assignment. -/
noncomputable def memoryResult
    {Program : Type} {context : Context Program}
    (node : BaseNode context) :
    ProductionMemoryCheckedBatchRows.Result node.memoryAuthority.layout
      node.baseRows.assignment context.headers :=
  node.supplement.memoryResult

noncomputable def opening
    {Program : Type} {context : Context Program}
    (node : BaseNode context) : ProductionPaperBaseInvocationFor.Opening :=
  node.baseRows.opening

/-- Local base exactness is derived from evidence. -/
theorem exact
    {Program : Type} {context : Context Program}
    (node : BaseNode context) :
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

/-- The exact lifetime base node uses the verifier-derived challenge
authority.  In particular, its two dynamic digest fields come from the
canonical base input and the challenge-independent base successor prefix. -/
theorem challengeAuthorityExact
    {Program : Type} {context : Context Program}
    (node : BaseNode context) :
    node.opening.authority =
      ProductionPaperBaseInvocationFor.challengeAuthority
        (rowVariables := context.rowVariables)
        (logicalWidth := context.logicalWidth)
        (publicFits := context.publicFits) context.candidate
        context.statementId context.headers context.statement node.opening
        node.batch :=
  node.exact.challengeAuthorityExact

noncomputable def producer
    {Program : Type} {context : Context Program}
    (node : BaseNode context) : context.Successor :=
  ProductionPaperBaseInvocationFor.state context.candidate context.headers
    context.statement node.opening node.batch

noncomputable def claim
    {Program : Type} {context : Context Program}
    (node : BaseNode context) : context.Claim :=
  ProductionPaperBaseInvocationFor.claim context.candidate context.statementId
    context.config context.headers context.statement node.opening node.batch
    node.memoryResult.suffixBatch node.freshAssignment

/-- The base producer's memory rows are attached to claim zero before any
recursive consumer exists. -/
noncomputable def produced
    {Program : Type} {context : Context Program}
    (node : BaseNode context) :
    ProducedBatch context node.producer node.claim where
  layout := node.memoryAuthority.layout
  assignment := node.baseRows.assignment
  result := node.memoryResult
  producerCanonical := node.exact.stateCanonical
  beforeExact := by
    simpa [producer, ProductionPaperBaseInvocationFor.state,
      MemoryCarryParser.semanticCarry, CarryEncoding.encodeActive] using
      node.exact.delayedCurrentMemory
  memoryExact := node.exact.claimMemoryExact.symm

/-- Exact authority for the first produced complete claim. -/
theorem authority
    {Program : Type} {context : Context Program}
    (node : BaseNode context) :
    ProducerAuthority context node.producer node.claim node.after := by
  refine
    { canonical := node.exact.stateCanonical
      applicationExact := ?_
      fullMatches := ?_ }
  · rfl
  · exact ProductionFreshClaimProducerFor.value_ccs_fullMatches
      context.candidate context.statementId context.config
      node.producer node.memoryResult.suffixBatch node.freshAssignment

/-- The first authenticated producer carry is the exact verifier-authoritative
base opening. -/
theorem producerCarry_eq_active
    {Program : Type} {context : Context Program}
    (node : BaseNode context) :
    producerCarry node.authority =
      .active (node.opening.activeFor context.candidate context.headers) := by
  unfold producerCarry producer
  simp [ProductionPaperBaseInvocationFor.state,
    MemoryCarryParser.semanticCarry, CarryEncoding.encodeActive]

noncomputable def toClaimNode
    {Program : Type} {context : Context Program}
    (node : BaseNode context) :
    ProductionPaperExactFPrimeLifetimeFor.BaseNode
      context.claimLifecycle where
  baseRows := node.baseRows
  baseArtifactExact := node.baseArtifactExact
  after := node.after
  batch := node.batch
  freshAssignment := node.freshAssignment
  supplement := node.supplement

/-- The semantic tail and its base node determine the claim-order lifetime;
no separate claim schedule is supplied by the caller. -/
noncomputable def claimLifetime
    {Program : Type} {context : Context Program}
    (node : BaseNode context)
    {produced : ProducedBatch context node.producer node.claim}
    (tail : Tail context node.producer node.claim node.after produced) :
    ProductionPaperExactFPrimeLifetimeFor.Lifetime
      context.claimLifecycle where
  base := node.toClaimNode
  firstClaim := node.claim
  firstClaimExact := rfl
  schedule := tail.claimSchedule

theorem exactClaimSchedule
    {Program : Type} {context : Context Program}
    (node : BaseNode context)
    {produced : ProducedBatch context node.producer node.claim}
    (tail : Tail context node.producer node.claim node.after produced) :
    ProductionPaperExactFPrimeLifetimeFor.Lifetime.ExactSchedule
      (node.claimLifetime tail) :=
  ProductionPaperExactFPrimeLifetimeFor.Lifetime.exact_schedule
    (node.claimLifetime tail)

/-- Exact row-level invocation indexes for one complete specialized F-prime
lifetime. The base assignment reads zero. The actual prior-state assignments
read `1, ..., T`, where `T` is the number of consumed complete claims and the
last value belongs to the terminal consumer. -/
structure RowInvocationIndexSchedule
    {Program : Type} {context : Context Program}
    (base : BaseNode context)
    {produced : ProducedBatch context base.producer base.claim}
    (tail : Tail context base.producer base.claim base.after produced) : Prop where
  baseInputZero :
    base.baseRows.assignment
        context.relationAuthority.fPrimeProgram.layout.iterationZero.iterationColumn =
      0
  consumers : tail.consumerInvocationIndices =
    List.range' 1 tail.receipts.length

/-- The exact row-level index schedule is derived from the base row selector,
the authenticated successor chain, and every parsed consumer state. An index
substitution can only enter through the named state-transcript collision. -/
theorem invocationIndexSchedule_or_collision
    {Program : Type} {context : Context Program}
    (base : BaseNode context)
    {produced : ProducedBatch context base.producer base.claim}
    (tail : Tail context base.producer base.claim base.after produced) :
    RowInvocationIndexSchedule base tail \/ context.Collision := by
  rcases tail.consumerInvocationIndices_or_collision base.authority with
    indices | collision
  · left
    have baseBranch :=
      ProductionPaperExactFPrimeLifetimeFor.BaseNode.freshSelectsFixedBase
        base.toClaimNode
    refine { baseInputZero := baseBranch.1, consumers := ?_ }
    have baseIndex : base.producer.augmentedInvocationIndex = 1 :=
      base.exact.baseInvocationIndex
    rw [baseIndex] at indices
    exact indices
  · exact Or.inr collision

end BaseNode

/-- Kernel-extracted base-to-terminal F-prime lifetime. The result includes
the base application rows, every verified claim receipt, and the separate
trailing terminal consumer. -/
structure LifetimeExtraction
    {Program : Type} {context : Context Program}
    (base : BaseNode context) where
  applicationRows : List ApplicationTrace.ApplicationRow
  receipts : List context.Receipt
  claimSchedule : ProductionPaperExactFPrimeLifetimeFor.Schedule
    context.claimLifecycle base.claim
  claimReceiptsExact : receipts =
    ExactDelayedSchedule.Schedule.receipts claimSchedule
  consumerInvocationIndexSchedule :
    claimSchedule.consumerInvocationIndices =
      List.range' 1 receipts.length
  fullStateContinuity : claimSchedule.FullStateContinuous base.producer
  baseBranch :
    base.baseRows.assignment
          context.relationAuthority.fPrimeProgram.layout.iterationZero.iterationColumn =
        0 /\
      R1CS.Satisfies context.baseArtifact.programRows base.baseRows.assignment
  recursiveBranches : claimSchedule.FixedRecursiveBranches
  finalMemory : ClosedCarry Digest.Value
  baseOpen : openSegment
      (fun closed precommit activeAccessCount =>
        MemoryOpenSegment.deriveFor (identity context.candidate)
          base.opening.authority closed precommit activeAccessCount)
      context.headers base.opening.precommit base.opening.activeAccessCount
      (ProductionPaperBaseInvocationFor.initialClosed
        base.opening.initialMemoryRoot)
      (ProductionPaperBaseInvocationFor.initialClosed_canOpen
        base.opening.initialMemoryRoot)
      base.opening.activeCountInRange
      base.opening.initialEndTimestampInRange =
    .active (base.opening.activeFor context.candidate context.headers)
  application : Runs context.machine context.statement.base.program
    context.statement.base.initialApplicationState applicationRows
    context.statement.base.expectedResult.finalApplicationState
    context.statement.base.expectedResult.realApplicationRowCount
  rowBefore : ProductionMemoryStepSemantics.ConcreteCarry
  rowDelayed : ProductionMemoryRowTrace.DelayedRun context.Verifier
    context.headers rowBefore receipts finalMemory
  rowBeforeExact : rowBefore =
    .active (base.opening.activeFor context.candidate context.headers)
  portsExact : ApplicationBatch.accesses applicationRows = rowDelayed.accesses
  applicationRowsLength : applicationRows.length =
    Completion.segmentCapacity context.statement.base.segmentCount
  realRowsExact : realRowCount applicationRows =
    context.statement.base.expectedResult.realApplicationRowCount
  freshClaimCount : receipts.length = context.statement.base.segmentCount *
    claimsPerSegment context.candidate
  augmentedInvocationCount : 1 + receipts.length =
    context.statement.base.segmentCount * claimsPerSegment context.candidate + 1
  finalSegment : finalMemory.segmentIndex = context.statement.base.segmentCount
  finalTimestamp : finalMemory.globalTimestamp =
    context.statement.base.finalGlobalTimestamp
  finalMemoryRoot : finalMemory.memoryRoot =
    context.statement.base.expectedResult.finalMemoryRoot

/-- Claim-level lifetime retained by the semantic extraction. Its schedule is
the same schedule whose receipts drive the row-derived memory trace. -/
noncomputable def LifetimeExtraction.claimLifetime
    {Program : Type} {context : Context Program}
    {base : BaseNode context}
    (lifetime : LifetimeExtraction base) :
    ProductionPaperExactFPrimeLifetimeFor.Lifetime
      context.claimLifecycle where
  base := base.toClaimNode
  firstClaim := base.claim
  firstClaimExact := rfl
  schedule := lifetime.claimSchedule

/-- The retained claim lifetime has the complete base, recursive, and
terminal F-prime schedule. -/
theorem LifetimeExtraction.exactClaimSchedule
    {Program : Type} {context : Context Program}
    {base : BaseNode context}
    (lifetime : LifetimeExtraction base) :
    ProductionPaperExactFPrimeLifetimeFor.Lifetime.ExactSchedule
      lifetime.claimLifetime :=
  ProductionPaperExactFPrimeLifetimeFor.Lifetime.exact_schedule
    lifetime.claimLifetime

/-- The extracted lifetime uses only the fixed base and recursive arms of the
one verifier-owned F-prime relation. This statement is derived from the same
row assignments retained by the lifetime. -/
theorem LifetimeExtraction.fixedBranchSchedule
    {Program : Type} {context : Context Program}
    {base : BaseNode context}
    (lifetime : LifetimeExtraction base) :
    ProductionPaperExactFPrimeLifetimeFor.Lifetime.FixedBranchSchedule
      lifetime.claimLifetime :=
  { baseIterationZero := lifetime.baseBranch.1
    baseRows := lifetime.baseBranch.2
    recursiveRows := lifetime.recursiveBranches
    terminalRows := lifetime.claimSchedule.terminalNode.2.fixedProgramSatisfied
    terminalTypedRows :=
      lifetime.claimSchedule.terminalNode.2.fixedTypedProgramSatisfied }

/-- Semantic memory extraction and exact F-prime verification use one
ordered receipt list. -/
theorem LifetimeExtraction.receipts_eq_consumedReceipts
    {Program : Type} {context : Context Program}
    {base : BaseNode context}
    (lifetime : LifetimeExtraction base) :
    lifetime.receipts = lifetime.claimLifetime.consumedReceipts := by
  exact lifetime.claimReceiptsExact

/-- The actual prior-state index stored in every retained recursive or
terminal consumer is exactly `1, ..., T`. The final value belongs to the
terminal consumer. -/
theorem LifetimeExtraction.consumerInvocationIndices_exact
    {Program : Type} {context : Context Program}
    {base : BaseNode context}
    (lifetime : LifetimeExtraction base) :
    lifetime.claimLifetime.schedule.consumerInvocationIndices =
      List.range' 1 lifetime.receipts.length :=
  lifetime.consumerInvocationIndexSchedule

/-- Every retained recursive and terminal consumer reads the complete state
produced by its predecessor. This is the actual state chain, not only the
derived invocation-index sequence. -/
theorem LifetimeExtraction.fullStateContinuityExact
    {Program : Type} {context : Context Program}
    {base : BaseNode context}
    (lifetime : LifetimeExtraction base) :
    lifetime.claimLifetime.schedule.FullStateContinuous base.producer :=
  lifetime.fullStateContinuity

/-- The extracted row-derived delayed schedule has one exact segment
partition. Every batch in the result is the row result attached to the same
verified complete receipt. The segment count is derived, not supplied. -/
theorem LifetimeExtraction.rowSegmentChain
    {Program : Type} {context : Context Program}
    {base : BaseNode context}
    (lifetime : LifetimeExtraction base) :
    exists allBatches,
      ProductionMemoryRowSegments.receipts allBatches = lifetime.receipts /\
      ProductionMemoryRowSegments.Chain context.candidate context.Schema
        context.Verifier
        context.headers
        (ProductionPaperBaseInvocationFor.initialClosed
          context.authoritativeInitialMemoryRoot)
        allBatches lifetime.finalMemory context.statement.base.segmentCount /\
      allBatches = lifetime.rowDelayed.batches /\
      ApplicationBatch.accesses lifetime.applicationRows =
        ProductionMemoryRowSegments.accesses allBatches := by
  rcases
      ProductionMemoryRowSegments.delayedRun_to_rowSegmentChain
        base.opening.authority base.opening.precommit
        base.opening.activeAccessCount
        (ProductionPaperBaseInvocationFor.initialClosed_canOpen
          base.opening.initialMemoryRoot)
        base.opening.activeCountInRange base.opening.initialEndTimestampInRange
        lifetime.baseOpen lifetime.rowBeforeExact lifetime.rowDelayed with
    ⟨segmentCount, allBatches, receiptsExact, chain, _positive,
      batchesExact⟩
  have chainWithAuthoritativeInitial :
      ProductionMemoryRowSegments.Chain context.candidate context.Schema
        context.Verifier
        context.headers
        (ProductionPaperBaseInvocationFor.initialClosed
          context.authoritativeInitialMemoryRoot)
        allBatches lifetime.finalMemory segmentCount := by
    have rootExact : base.opening.initialMemoryRoot =
        context.authoritativeInitialMemoryRoot := base.initialMemoryRootExact
    simpa [rootExact] using chain
  have batchReceiptCount : allBatches.length = lifetime.receipts.length := by
    have exact := congrArg List.length receiptsExact
    simpa [ProductionMemoryRowSegments.receipts] using exact
  have chainCount := chain.exactBatchCount
  have productCountExact :
      segmentCount * claimsPerSegment context.candidate =
        context.statement.base.segmentCount *
          claimsPerSegment context.candidate := by
    calc
      segmentCount * claimsPerSegment context.candidate = allBatches.length :=
        chainCount.symm
      _ = lifetime.receipts.length := batchReceiptCount
      _ = context.statement.base.segmentCount *
          claimsPerSegment context.candidate := lifetime.freshClaimCount
  have claimsPositive :
      0 < claimsPerSegment context.candidate := by
    cases context.candidate <;> decide
  have segmentCountExact :
      segmentCount = context.statement.base.segmentCount := by
    exact Nat.eq_of_mul_eq_mul_right claimsPositive productCountExact
  subst segmentCount
  have accessesExact : ApplicationBatch.accesses lifetime.applicationRows =
      ProductionMemoryRowSegments.accesses allBatches := by
    calc
      ApplicationBatch.accesses lifetime.applicationRows =
          lifetime.rowDelayed.accesses := lifetime.portsExact
      _ = ProductionMemoryRowSegments.accesses
          lifetime.rowDelayed.batches := rfl
      _ = ProductionMemoryRowSegments.accesses allBatches := by
        rw [batchesExact]
  exact ⟨allBatches, receiptsExact, chainWithAuthoritativeInitial,
    batchesExact, accessesExact⟩

/-- Exact base plus exact tail reconstructs the complete lifetime, unless one
named cross-invocation state collision occurs. -/
theorem BaseNode.extract_or_collision
    {Program : Type} {context : Context Program}
    (base : BaseNode context)
    (tail : Tail context base.producer base.claim base.after base.produced) :
    Nonempty (LifetimeExtraction base) \/ context.Collision := by
  classical
  have basePortsExact : ApplicationBatch.accesses base.batch.rows =
      ProductionApplicationBatchBridge.memoryAccesses base.produced.result :=
    base.supplement.applicationMatched.accesses_exact
  have baseBranch :=
    ProductionPaperExactFPrimeLifetimeFor.BaseNode.freshSelectsFixedBase
      base.toClaimNode
  have baseIndex : base.producer.augmentedInvocationIndex = 1 :=
    base.exact.baseInvocationIndex
  have basePositive : 0 < base.producer.augmentedInvocationIndex := by
    omega
  by_cases collisionOccurs : context.Collision
  · exact Or.inr collisionOccurs
  have fullStateContinuity :
      tail.claimSchedule.FullStateContinuous base.producer := by
    rcases tail.fullStateContinuity_or_collision base.authority with
      exactContinuity | collision
    · exact exactContinuity
    · exact False.elim (collisionOccurs collision)
  have indexSchedule : RowInvocationIndexSchedule base tail := by
    rcases base.invocationIndexSchedule_or_collision tail with
      exactSchedule | collision
    · exact exactSchedule
    · exact False.elim (collisionOccurs collision)
  rcases tail.fixedRecursiveBranches_or_collision base.authority basePositive with
    branches | branchCollision
  · rcases tail.extract_or_collision base.authority with
      ⟨⟨extracted⟩⟩ | collision
    · let rows := base.batch.rows ++ extracted.applicationRows
      have rowsRun : Runs context.machine context.statement.base.program
          context.statement.base.initialApplicationState rows
          context.statement.base.expectedResult.finalApplicationState
          (realRowCount rows) := by
        have appended := base.exact.applicationRun.append extracted.application
        simpa [rows, ApplicationBatchCompletion.realRowCount_append] using
          appended
      have producerRows : base.producer.realApplicationRowCount =
          realRowCount base.batch.rows := by
        rfl
      have realRowsExact : realRowCount rows =
          context.statement.base.expectedResult.realApplicationRowCount := by
        simp only [rows, ApplicationBatchCompletion.realRowCount_append]
        rw [← producerRows]
        exact extracted.realRowsAccounting
      have applicationRun : Runs context.machine
          context.statement.base.program
          context.statement.base.initialApplicationState rows
          context.statement.base.expectedResult.finalApplicationState
          context.statement.base.expectedResult.realApplicationRowCount := by
        simpa [realRowsExact] using rowsRun
      have receiptCount : extracted.receipts.length =
          context.statement.base.segmentCount *
            claimsPerSegment context.candidate := by
        have allClaims := extracted.claimCountAccounting
        omega
      refine Or.inl ⟨
        { applicationRows := rows
          receipts := extracted.receipts
          claimSchedule := tail.claimSchedule
          claimReceiptsExact := by
            simpa [Tail.receipts] using extracted.receiptsExact
          consumerInvocationIndexSchedule := by
            calc
              tail.claimSchedule.consumerInvocationIndices =
                  tail.consumerInvocationIndices :=
                tail.claimSchedule_consumerInvocationIndices
              _ = List.range' 1 tail.receipts.length := indexSchedule.consumers
              _ = List.range' 1 extracted.receipts.length := by
                rw [extracted.receiptsExact]
          fullStateContinuity := fullStateContinuity
          baseBranch := baseBranch
          recursiveBranches := branches
          finalMemory := extracted.finalMemory
          baseOpen := base.opening.open_exact_for context.candidate
            context.headers
          application := applicationRun
          rowBefore := extracted.rowBefore
          rowDelayed := extracted.rowDelayed
          rowBeforeExact := extracted.rowBeforeExact.trans
            base.producerCarry_eq_active
          portsExact := ?_
          applicationRowsLength := ?_
          realRowsExact := realRowsExact
          freshClaimCount := receiptCount
          augmentedInvocationCount := ?_
          finalSegment := extracted.finalSegment
          finalTimestamp := extracted.finalTimestamp
          finalMemoryRoot := extracted.finalMemoryRoot }⟩
      · rw [show ApplicationBatch.accesses rows =
              ApplicationBatch.accesses base.batch.rows ++
                ApplicationBatch.accesses extracted.applicationRows by
            exact ApplicationBatch.accesses_append _ _]
        rw [basePortsExact]
        exact extracted.portsExact
      · calc
          rows.length = base.batch.rows.length +
              extracted.applicationRows.length := by
            simp only [rows, List.length_append]
          _ = ApplicationBatch.rowsPerFreshClaim context.candidate +
              extracted.applicationRows.length := by
            rw [base.batch.rowsExact]
          _ = extracted.applicationRows.length +
              ApplicationBatch.rowsPerFreshClaim context.candidate :=
            Nat.add_comm _ _
          _ = extracted.receipts.length *
              ApplicationBatch.rowsPerFreshClaim context.candidate :=
            extracted.rowLengthAccounting
          _ = (context.statement.base.segmentCount *
                claimsPerSegment context.candidate) *
              ApplicationBatch.rowsPerFreshClaim context.candidate := by
            rw [receiptCount]
          _ = context.statement.base.segmentCount *
              (claimsPerSegment context.candidate *
                ApplicationBatch.rowsPerFreshClaim context.candidate) := by
            rw [Nat.mul_assoc]
          _ = context.statement.base.segmentCount *
              Completion.applicationRowsPerSegment := by
            rw [ApplicationBatch.claims_rows_partition_segment]
          _ = Completion.segmentCapacity
              context.statement.base.segmentCount := rfl
      · omega
    · exact Or.inr collision
  · exact Or.inr branchCollision

/-- Exact completed-row syntax plus the extracted operational run derives a
completed execution. A completed execution is not an input assumption. -/
theorem LifetimeExtraction.completedExecution
    {Program : Type} {context : Context Program}
    {base : BaseNode context}
    (lifetime : LifetimeExtraction base)
    (completion : CompletionRows context lifetime.applicationRows) :
    Nonempty (ApplicationTrace.CompletedExecution context.machine.semantics
      context.statement.base.program
      context.statement.base.initialApplicationState
      context.statement.base.expectedResult
      context.statement.base.segmentCount) := by
  have exactRun := lifetime.application
  rw [completion.rowsExact] at exactRun
  exact ApplicationBatchCompletion.completedExecution_of_exact_rows exactRun
    completion.realRowCountExact completion.segmentCountPositive
    completion.segmentCountBound completion.realRowCountBound
    completion.fitsDeclaredSegments completion.smallestSegmentCount

/-- The exact operational application run, public typed terminal state, exact
fresh-claim row accounting, and verifier-checked statement bounds derive the
completed execution together with equality to the extracted lifecycle rows.
No completion-row shape is supplied by the caller. -/
theorem LifetimeExtraction.exactCompletedRun
    {Program : Type} {context : Context Program}
    {base : BaseNode context}
    (lifetime : LifetimeExtraction base) :
    Nonempty (ApplicationBatchCompletion.ExactCompletedRun context.machine
      context.statement.base.program
      context.statement.base.initialApplicationState
      context.statement.base.expectedResult
      context.statement.base.segmentCount lifetime.applicationRows) := by
  exact ApplicationBatchCompletion.exactCompletedRun_of_terminal_run
    lifetime.application context.statement.resultDecoded.terminal
    lifetime.applicationRowsLength
    context.publicDecoded.segmentCountPositive
    context.publicDecoded.segmentCountBound
    context.statement.resultDecoded.realRowCountPositive
    context.statement.resultDecoded.realRowCountBound
    context.publicDecoded.realRowsFitDeclaredSegments
    context.publicDecoded.smallestSegmentCount

/-- Erasing only the retained lifecycle-row equality gives the standard
completed-execution witness. -/
theorem LifetimeExtraction.completedExecutionDerived
    {Program : Type} {context : Context Program}
    {base : BaseNode context}
    (lifetime : LifetimeExtraction base) :
    Nonempty (ApplicationTrace.CompletedExecution context.machine.semantics
      context.statement.base.program
      context.statement.base.initialApplicationState
      context.statement.base.expectedResult
      context.statement.base.segmentCount) := by
  rcases lifetime.exactCompletedRun with ⟨completed⟩
  exact ⟨completed.execution⟩

end Nightstream.Implementation.Nebula.ProductionPaperExactLifetime
