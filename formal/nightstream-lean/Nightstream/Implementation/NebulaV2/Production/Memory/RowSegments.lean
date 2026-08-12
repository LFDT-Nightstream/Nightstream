import Mathlib.Algebra.Field.TransferInstance
import Nightstream.Implementation.NebulaV2.Core.ConcreteField
import Nightstream.Implementation.NebulaV2.Production.Memory.RowTrace
import Nightstream.Protocol.NebulaV2.ProductionBatchedScanSchedule

/-!
Contract: reverse an annotated delayed F-prime row trace into exact
row-derived memory segments.

Each segment keeps the production batch results that supplied its verified
claims. `BatchRun` proves that these batches share exact carry boundaries.
`SegmentRun` starts at the canonical open carry and ends only after the close
step. `Chain` preserves the exact boundary reopen between segments.

The reverse theorem does not accept a segment partition, a step list, or a
claim list as an input. It derives all three from the annotated delayed trace.

Does not own snapshot reconstruction, fingerprint probability, commitment
binding, application-row alignment, or deployed-verifier extraction.

Assurance tier: implementation-to-protocol bridge.

Emits constraints: no.
-/

set_option autoImplicit false
set_option maxRecDepth 100000

namespace Nightstream.Implementation.NebulaV2.ProductionMemoryRowSegments

open Nightstream.Implementation.NebulaV2.ProductionMemoryRowTrace
open Nightstream.Implementation.NebulaV2.ProductionMemoryStepSemantics
open Nightstream.Protocol.NebulaV2
open Nightstream.Protocol.NebulaV2.AugmentedLifecycle
open Nightstream.Protocol.NebulaV2.FPrime
open Nightstream.Protocol.NebulaV2.ProductionBatchedFPrime
open Nightstream.Protocol.NebulaV2.ProductionBatchedGlobalFPrime
open Nightstream.Protocol.NebulaV2.ProductionProfileCandidates
open Nightstream.SuperNeo.Concrete

/-- The concrete extension-field carrier uses the proved field equivalence. -/
noncomputable local instance concreteKField : Field K :=
  ConcreteField.superNeoEquiv.field

abbrev Evidence
    (candidate : Id) (schema : ProductionBatchedFPrime.Schema)
    (verify : BatchVerifier candidate schema Digest.Value K)
    (headers : ChainHeaders Digest.Value) :=
  ProductionMemoryRowTrace.BatchEvidence candidate schema verify headers

/-- Receipts in the exact order of their row-derived batches. -/
def receipts
    {candidate : Id} {schema : ProductionBatchedFPrime.Schema}
    {verify : BatchVerifier candidate schema Digest.Value K}
    {headers : ChainHeaders Digest.Value}
    (batches : List (Evidence candidate schema verify headers)) :
    List (Receipt candidate schema Digest.Value K verify) :=
  batches.map BatchEvidence.receipt

/-- Checked semantic steps in exact batch and inner-step order. -/
def steps
    {candidate : Id} {schema : ProductionBatchedFPrime.Schema}
    {verify : BatchVerifier candidate schema Digest.Value K}
    {headers : ChainHeaders Digest.Value}
    (batches : List (Evidence candidate schema verify headers)) :
    List ProductionMemoryStepSemantics.Step :=
  batches.flatMap BatchEvidence.steps

/-- Ordered application accesses in exact producer-batch order. -/
def accesses
    {candidate : Id} {schema : ProductionBatchedFPrime.Schema}
    {verify : BatchVerifier candidate schema Digest.Value K}
    {headers : ChainHeaders Digest.Value}
    (batches : List (Evidence candidate schema verify headers)) :
    List Access :=
  batches.flatMap BatchEvidence.accesses

@[simp] theorem accesses_append
    {candidate : Id} {schema : ProductionBatchedFPrime.Schema}
    {verify : BatchVerifier candidate schema Digest.Value K}
    {headers : ChainHeaders Digest.Value}
    (left right : List (Evidence candidate schema verify headers)) :
    accesses (left ++ right) = accesses left ++ accesses right := by
  simp [accesses]

/-- Exact carry chaining for a list of row-derived production batches. -/
inductive BatchRun
    {candidate : Id} {schema : ProductionBatchedFPrime.Schema}
    (verify : BatchVerifier candidate schema Digest.Value K)
    (headers : ChainHeaders Digest.Value) :
    ProductionMemoryStepSemantics.ConcreteCarry ->
      List (Evidence candidate schema verify headers) ->
      ProductionMemoryStepSemantics.ConcreteCarry -> Prop
  | nil (state : ProductionMemoryStepSemantics.ConcreteCarry) :
      BatchRun verify headers state [] state
  | cons
      {tail : List (Evidence candidate schema verify headers)}
      {final : ProductionMemoryStepSemantics.ConcreteCarry}
      (head : Evidence candidate schema verify headers)
      (rest : BatchRun verify headers head.after tail final) :
      BatchRun verify headers head.before (head :: tail) final

namespace BatchRun

variable {candidate : Id} {schema : ProductionBatchedFPrime.Schema}
variable {verify : BatchVerifier candidate schema Digest.Value K}
variable {headers : ChainHeaders Digest.Value}

/-- Row-derived batch runs compose without changing batch order. -/
theorem append
    {before middle after : ProductionMemoryStepSemantics.ConcreteCarry}
    {left right : List (Evidence candidate schema verify headers)}
    (first : BatchRun verify headers before left middle)
    (second : BatchRun verify headers middle right after) :
    BatchRun verify headers before (left ++ right) after := by
  induction first with
  | nil => exact second
  | cons head rest inductionHypothesis =>
      exact .cons head (inductionHypothesis second)

/-- Forget batch boundaries and keep the exact checked-step row run. -/
theorem toStepRun
    {before after : ProductionMemoryStepSemantics.ConcreteCarry}
    {batches : List (Evidence candidate schema verify headers)}
    (run : BatchRun verify headers before batches after) :
    ProductionMemoryStepSemantics.Run before (steps batches) after := by
  induction run with
  | nil => exact .nil _
  | cons head rest inductionHypothesis =>
      simpa [steps] using head.stepRun.append inductionHypothesis

/-- Forget row annotations and keep the exact verified batch transition. -/
theorem toVerifiedRun
    {before after : ProductionMemoryStepSemantics.ConcreteCarry}
    {batches : List (Evidence candidate schema verify headers)}
    (run : BatchRun verify headers before batches after) :
    ProductionBatchedFPrime.VerifiedRun verify ProductState.Balanced before
      (receipts batches) after := by
  induction run with
  | nil => exact .nil _
  | cons head rest inductionHypothesis =>
      simpa [receipts] using
        (ProductionBatchedFPrime.VerifiedRun.cons head.transition
          inductionHypothesis)

/-- The checked-step claims are exactly the suffixes of the same complete
verified receipts. -/
theorem claimsExact
    {before after : ProductionMemoryStepSemantics.ConcreteCarry}
    {batches : List (Evidence candidate schema verify headers)}
    (run : BatchRun verify headers before batches after) :
    ProductionMemoryStepSemantics.Run.claims (steps batches) =
      (receipts batches).flatMap fun receipt =>
        receipt.claim.memory.suffixes := by
  induction run with
  | nil => simp [steps, receipts, ProductionMemoryStepSemantics.Run.claims]
  | cons head rest inductionHypothesis =>
      simp only [steps, List.flatMap_cons,
        ProductionMemoryStepSemantics.Run.claims, List.map_append, receipts,
        List.map_cons, List.flatMap_cons]
      change
        ProductionMemoryStepSemantics.Run.claims head.steps ++
            ProductionMemoryStepSemantics.Run.claims (steps _) =
          head.receipt.claim.memory.suffixes ++
            (receipts _).flatMap fun receipt =>
              receipt.claim.memory.suffixes
      rw [head.claimsExact, inductionHypothesis]

/-- Batch and checked-step views retain one exact access order. -/
theorem accessesExact
    {before after : ProductionMemoryStepSemantics.ConcreteCarry}
    {batches : List (Evidence candidate schema verify headers)}
    (run : BatchRun verify headers before batches after) :
    ProductionMemoryStepSemantics.Run.accesses (steps batches) =
      accesses batches := by
  induction run with
  | nil => rfl
  | cons head rest inductionHypothesis =>
      simp only [steps, accesses, List.flatMap_cons,
        ProductionMemoryStepSemantics.Run.accesses, List.flatMap_append]
      change
        ProductionMemoryStepSemantics.Run.accesses head.steps ++
            ProductionMemoryStepSemantics.Run.accesses (steps _) =
          head.accesses ++ accesses _
      rw [← head.accesses_eq_steps, inductionHypothesis]

end BatchRun

/-! ## Exact row-derived segments -/

/-- One complete segment reconstructed from row-derived production batches. -/
structure SegmentRun
    (candidate : Id) (schema : ProductionBatchedFPrime.Schema)
    (verify : BatchVerifier candidate schema Digest.Value K)
    (headers : ChainHeaders Digest.Value)
    (before : ClosedCarry Digest.Value) where
  authority : MemoryOpenSegment.Authority
  precommit : Roots Digest.Value
  activeAccessCount : Nat
  canOpen : before.CanOpen
  activeCountInRange : activeAccessCount < operationCountLimit
  endTimestampInRange :
    before.globalTimestamp + activeAccessCount < timestampLimit
  active : ActiveCarry Digest.Value (ProductState.Challenges K)
    (ProductState.State K)
  after : ClosedCarry Digest.Value
  batches : List (Evidence candidate schema verify headers)
  opened :
    openSegment
        (fun closed roots count =>
          MemoryOpenSegment.deriveFor (identity candidate) authority closed
            roots count)
        headers precommit activeAccessCount before canOpen activeCountInRange
        endTimestampInRange = .active active
  consumed : BatchRun verify headers (.active active) batches (.closed after)

namespace SegmentRun

variable {candidate : Id} {schema : ProductionBatchedFPrime.Schema}
variable {verify : BatchVerifier candidate schema Digest.Value K}
variable {headers : ChainHeaders Digest.Value}
variable {before : ClosedCarry Digest.Value}

/-- Erasing row annotations gives the independent production segment. -/
def toProtocol
    (run : SegmentRun candidate schema verify headers before) :
    ProductionBatchedGlobalFPrime.SegmentRun candidate schema Digest.Value
      verify
        (fun closed roots count =>
          MemoryOpenSegment.deriveFor (identity candidate) run.authority
            closed roots count)
        headers before :=
  { precommit := run.precommit
    activeAccessCount := run.activeAccessCount
    canOpen := run.canOpen
    activeCountInRange := run.activeCountInRange
    endTimestampInRange := run.endTimestampInRange
    active := run.active
    after := run.after
    claims := receipts run.batches
    opened := run.opened
    consumed := run.consumed.toVerifiedRun }

/-- A reconstructed segment starts at checked step zero. -/
theorem startsAtStepZero
    (run : SegmentRun candidate schema verify headers before) :
    run.active.stepIndex.val = 0 :=
  run.toProtocol.startsAtStepZero

/-- A reconstructed segment contains the exact profile-specific number of
row-derived production batches. -/
theorem exactBatchCount
    (run : SegmentRun candidate schema verify headers before) :
    run.batches.length = claimsPerSegment candidate := by
  have exact := run.toProtocol.exactClaimCount
  simpa [toProtocol, receipts] using exact

/-- A reconstructed segment contains exactly 1,088 checked steps. -/
theorem exactStepCount
    (run : SegmentRun candidate schema verify headers before) :
    (steps run.batches).length = Lifecycle.claimsPerSegment := by
  have suffixCount :=
    ProductionBatchedScanSchedule.SegmentRun.suffixes_length_exact
      run.toProtocol
  simp only [ProductionBatchedScanSchedule.SegmentRun.suffixes, toProtocol]
    at suffixCount
  rw [← run.consumed.claimsExact] at suffixCount
  simpa [ProductionMemoryStepSemantics.Run.claims] using suffixCount

/-- Every row-derived checked step has the exact global position within its
segment. -/
theorem stepIndexAt
    (run : SegmentRun candidate schema verify headers before)
    (index : Fin (steps run.batches).length) :
    ((steps run.batches).get index).claim.stepIndex.val = index.val := by
  let claimIndex : Fin
      (ProductionMemoryStepSemantics.Run.claims (steps run.batches)).length :=
    index.cast (by simp [ProductionMemoryStepSemantics.Run.claims])
  have indexed :=
    ProductionBatchedScanSchedule.ConsumesList.claim_step_at
      run.consumed.toStepRun.toConsumesList run.active rfl claimIndex
  rw [run.startsAtStepZero] at indexed
  simpa [claimIndex, ProductionMemoryStepSemantics.Run.claims] using indexed

/-- Every row-derived checked step uses the segment bounds fixed by the
opening carry. -/
theorem segmentBoundsAt
    (run : SegmentRun candidate schema verify headers before)
    (index : Fin (steps run.batches).length) :
    ((steps run.batches).get index).claim.segmentStartTimestamp =
        run.active.segmentStartTimestamp /\
      ((steps run.batches).get index).claim.segmentEndTimestamp =
        run.active.segmentEndTimestamp := by
  let claimIndex : Fin
      (ProductionMemoryStepSemantics.Run.claims (steps run.batches)).length :=
    index.cast (by simp [ProductionMemoryStepSemantics.Run.claims])
  have bounds :=
    ProductionBatchedScanSchedule.ConsumesList.claim_segment_bounds_at
      run.consumed.toStepRun.toConsumesList run.active rfl claimIndex
  simpa [claimIndex, ProductionMemoryStepSemantics.Run.claims] using bounds

end SegmentRun

/-! ## Exact cross-segment row chain -/

/-- Exact boundary-contiguous chain of row-derived memory segments. -/
inductive Chain
    (candidate : Id) (schema : ProductionBatchedFPrime.Schema)
    (verify : BatchVerifier candidate schema Digest.Value K)
    (headers : ChainHeaders Digest.Value) :
    ClosedCarry Digest.Value ->
      List (Evidence candidate schema verify headers) ->
      ClosedCarry Digest.Value -> Nat -> Prop
  | nil (state : ClosedCarry Digest.Value) :
      Chain candidate schema verify headers state [] state 0
  | cons
      {before final : ClosedCarry Digest.Value}
      {tailBatches : List (Evidence candidate schema verify headers)}
      {tailSegments : Nat}
      (head : SegmentRun candidate schema verify headers before)
      (tail : Chain candidate schema verify headers head.after
        tailBatches final tailSegments) :
      Chain candidate schema verify headers before
        (head.batches ++ tailBatches) final (tailSegments + 1)

namespace Chain

variable {candidate : Id} {schema : ProductionBatchedFPrime.Schema}
variable {verify : BatchVerifier candidate schema Digest.Value K}
variable {headers : ChainHeaders Digest.Value}

/-- The complete row-derived batch list has the exact lifetime count. -/
theorem exactBatchCount
    {initial final : ClosedCarry Digest.Value}
    {batches : List (Evidence candidate schema verify headers)}
    {segmentCount : Nat}
    (chain : Chain candidate schema verify headers initial batches final
      segmentCount) :
    batches.length = segmentCount * claimsPerSegment candidate := by
  induction chain with
  | nil => simp
  | cons head tail inductionHypothesis =>
      simp [head.exactBatchCount, inductionHypothesis, Nat.add_mul,
        Nat.add_comm]

end Chain

/-! ## Reverse the annotated delayed schedule -/

/-- Equality-preserving view of a continuation. This avoids dependent
elimination on a concrete row-result projection. -/
private inductive ContinuationCase
    (candidate : Id)
    {headers : ChainHeaders Digest.Value}
    (intermediate outgoing : ProductionMemoryStepSemantics.ConcreteCarry) : Prop
  | interior
      (active : ActiveCarry Digest.Value (ProductState.Challenges K)
        (ProductState.State K))
      (intermediateExact : intermediate = .active active)
      (outgoingExact : outgoing = .active active) :
      ContinuationCase candidate intermediate outgoing
  | boundary
      (authority : MemoryOpenSegment.Authority)
      (closed : ClosedCarry Digest.Value)
      (precommit : Roots Digest.Value)
      (activeAccessCount : Nat)
      (canOpen : closed.CanOpen)
      (activeCountInRange : activeAccessCount < operationCountLimit)
      (endTimestampInRange :
        closed.globalTimestamp + activeAccessCount < timestampLimit)
      (intermediateExact : intermediate = .closed closed)
      (outgoingExact : outgoing =
        openSegment
          (fun boundary roots count =>
            MemoryOpenSegment.deriveFor (identity candidate) authority
              boundary roots count)
          headers precommit activeAccessCount closed canOpen activeCountInRange
          endTimestampInRange) :
      ContinuationCase candidate intermediate outgoing

/-- Every exact continuation has one of the two explicit cases. -/
private theorem continuationCase_of_continues
    {candidate : Id}
    {headers : ChainHeaders Digest.Value}
    {intermediate outgoing : ProductionMemoryStepSemantics.ConcreteCarry}
    (continues : ProductionMemoryRowTrace.BoundContinuation candidate headers
      intermediate outgoing) :
    @ContinuationCase candidate headers intermediate outgoing := by
  rcases continues with ⟨authority, exact⟩
  cases exact with
  | interior active => exact .interior active rfl rfl
  | boundary closed precommit activeAccessCount canOpen activeRange endRange =>
      exact .boundary authority closed precommit activeAccessCount
        canOpen activeRange endRange rfl rfl

/-- Complete the current row-derived segment and all later segments. -/
private theorem completeFromPrefix
    {candidate : Id} {schema : ProductionBatchedFPrime.Schema}
    {verify : BatchVerifier candidate schema Digest.Value K}
    {headers : ChainHeaders Digest.Value}
    {segmentBefore final : ClosedCarry Digest.Value}
    {active : ActiveCarry Digest.Value (ProductState.Challenges K)
      (ProductState.State K)}
    {current : ProductionMemoryStepSemantics.ConcreteCarry}
    {prefixReceipts tailReceipts : List
      (Receipt candidate schema Digest.Value K verify)}
    (segmentAuthority : MemoryOpenSegment.Authority)
    (precommit : Roots Digest.Value)
    (activeAccessCount : Nat)
    (canOpen : segmentBefore.CanOpen)
    (activeCountInRange : activeAccessCount < operationCountLimit)
    (endTimestampInRange :
      segmentBefore.globalTimestamp + activeAccessCount < timestampLimit)
    (opened : openSegment
        (fun closed roots count =>
          MemoryOpenSegment.deriveFor (identity candidate) segmentAuthority
            closed roots count)
        headers precommit activeAccessCount segmentBefore canOpen
        activeCountInRange endTimestampInRange = .active active)
    (prefixBatches : List (Evidence candidate schema verify headers))
    (prefixReceiptsExact : receipts prefixBatches = prefixReceipts)
    (prefixRun : BatchRun verify headers (.active active) prefixBatches current)
    (delayed : ProductionMemoryRowTrace.DelayedRun verify headers
      current tailReceipts final) :
    exists segmentCount allBatches,
      receipts allBatches = prefixReceipts ++ tailReceipts /\
        Chain candidate schema verify headers segmentBefore allBatches
          final segmentCount /\
        0 < segmentCount /\
        allBatches = prefixBatches ++ delayed.batches := by
  induction delayed generalizing segmentBefore active prefixReceipts
      prefixBatches segmentAuthority precommit activeAccessCount canOpen
      activeCountInRange endTimestampInRange with
  | @terminal batch terminalFinal closedExact =>
      let completeBatches := prefixBatches ++ [batch]
      have completeRun : BatchRun verify headers (.active active)
          completeBatches (.closed terminalFinal) := by
        have appended := prefixRun.append
          (BatchRun.cons batch (BatchRun.nil batch.after))
        simpa [completeBatches, closedExact] using appended
      let head : SegmentRun candidate schema verify headers
          segmentBefore :=
        { authority := segmentAuthority
          precommit := precommit
          activeAccessCount := activeAccessCount
          canOpen := canOpen
          activeCountInRange := activeCountInRange
          endTimestampInRange := endTimestampInRange
          active := active
          after := terminalFinal
          batches := completeBatches
          opened := opened
          consumed := completeRun }
      refine ⟨1, completeBatches, ?_, ?_, by omega, rfl⟩
      · change List.map BatchEvidence.receipt prefixBatches =
          prefixReceipts at prefixReceiptsExact
        simp [completeBatches, receipts, prefixReceiptsExact]
      · simpa [head] using
          (Chain.cons head (Chain.nil terminalFinal))
  | @recursive outgoing tail terminalFinal batch continues rest
      inductionHypothesis =>
      let extendedBatches := prefixBatches ++ [batch]
      have extendedRun : BatchRun verify headers (.active active)
          extendedBatches batch.after := by
        exact prefixRun.append
          (BatchRun.cons batch (BatchRun.nil batch.after))
      have extendedReceipts :
          receipts extendedBatches = prefixReceipts ++ [batch.receipt] := by
        change List.map BatchEvidence.receipt prefixBatches =
          prefixReceipts at prefixReceiptsExact
        simp [extendedBatches, receipts, prefixReceiptsExact]
      cases continuationCase_of_continues continues with
      | interior middleActive intermediateExact outgoingExact =>
          have extendedActive : BatchRun verify headers (.active active)
              extendedBatches (.active middleActive) := by
            simpa [intermediateExact] using extendedRun
          have outgoingActive : outgoing = .active middleActive :=
            outgoingExact
          rcases inductionHypothesis segmentAuthority precommit activeAccessCount canOpen
              activeCountInRange endTimestampInRange opened extendedBatches
              extendedReceipts (by simpa [outgoingActive] using extendedActive) with
            ⟨segmentCount, allBatches, receiptsExact, chain, positive,
              batchesExact⟩
          refine ⟨segmentCount, allBatches, ?_, chain, positive, ?_⟩
          simpa [List.append_assoc] using receiptsExact
          simpa [extendedBatches, List.append_assoc] using batchesExact
      | boundary nextAuthority closed nextPrecommit nextActiveAccessCount nextCanOpen
          nextActiveCountInRange nextEndTimestampInRange intermediateExact
          outgoingExact =>
          let nextActive : ActiveCarry Digest.Value
              (ProductState.Challenges K) (ProductState.State K) :=
            { segmentIndex := closed.segmentIndex
              stepIndex := ⟨0, by decide⟩
              globalTimestamp := closed.globalTimestamp
              segmentStartTimestamp := closed.globalTimestamp
              segmentActiveAccessCount := nextActiveAccessCount
              segmentEndTimestamp :=
                closed.globalTimestamp + nextActiveAccessCount
              challenge := MemoryOpenSegment.deriveFor (identity candidate)
                nextAuthority closed nextPrecommit nextActiveAccessCount
              products := ProductState.one
              dPre := nextPrecommit
              dSeen := headers.roots
              memoryRoot := closed.memoryRoot }
          have nextOpened : openSegment
              (fun boundary roots count =>
                MemoryOpenSegment.deriveFor (identity candidate) nextAuthority
                  boundary roots count)
              headers nextPrecommit nextActiveAccessCount closed nextCanOpen
              nextActiveCountInRange nextEndTimestampInRange =
                .active nextActive := rfl
          have nextPrefix : BatchRun verify headers (.active nextActive) []
              (.active nextActive) := BatchRun.nil _
          have outgoingActive : outgoing = .active nextActive := by
            simpa [nextActive] using outgoingExact
          have nextPrefixOutgoing : BatchRun verify headers
              (.active nextActive) [] outgoing := by
            simpa [outgoingActive] using nextPrefix
          rcases inductionHypothesis nextAuthority nextPrecommit nextActiveAccessCount
              nextCanOpen nextActiveCountInRange nextEndTimestampInRange
              nextOpened [] rfl nextPrefixOutgoing with
            ⟨tailSegments, tailBatches, tailReceiptsExact, tailChain,
              tailPositive, tailBatchesExact⟩
          have closedRun : BatchRun verify headers (.active active)
              extendedBatches (.closed closed) := by
            simpa [intermediateExact] using extendedRun
          let head : SegmentRun candidate schema verify headers
              segmentBefore :=
            { authority := segmentAuthority
              precommit := precommit
              activeAccessCount := activeAccessCount
              canOpen := canOpen
              activeCountInRange := activeCountInRange
              endTimestampInRange := endTimestampInRange
              active := active
              after := closed
              batches := extendedBatches
              opened := opened
              consumed := closedRun }
          let allBatches := extendedBatches ++ tailBatches
          refine ⟨tailSegments + 1, allBatches, ?_, ?_, by omega, ?_⟩
          · have extendedReceiptsMap :
                List.map BatchEvidence.receipt extendedBatches =
                  prefixReceipts ++ [batch.receipt] := by
              exact extendedReceipts
            have tailReceiptsMap :
                List.map BatchEvidence.receipt tailBatches = tail := by
              simpa [receipts] using tailReceiptsExact
            simp only [allBatches, receipts, List.map_append]
            rw [extendedReceiptsMap, tailReceiptsMap]
            simp [List.append_assoc]
          · simpa [head, allBatches] using Chain.cons head tailChain
          · simpa [allBatches, extendedBatches, List.append_assoc] using
              tailBatchesExact

/-- Reverse compiler theorem for the base-opened annotated row trace. No
segment partition or row-derived batch list is supplied by the caller. -/
theorem delayedRun_to_rowSegmentChain
    {candidate : Id} {schema : ProductionBatchedFPrime.Schema}
    {verify : BatchVerifier candidate schema Digest.Value K}
    {headers : ChainHeaders Digest.Value}
    {initial final : ClosedCarry Digest.Value}
    {active : ActiveCarry Digest.Value (ProductState.Challenges K)
      (ProductState.State K)}
    {rowBefore : ProductionMemoryStepSemantics.ConcreteCarry}
    {claimReceipts : List
      (Receipt candidate schema Digest.Value K verify)}
    (authority : MemoryOpenSegment.Authority)
    (precommit : Roots Digest.Value)
    (activeAccessCount : Nat)
    (canOpen : initial.CanOpen)
    (activeCountInRange : activeAccessCount < operationCountLimit)
    (endTimestampInRange :
      initial.globalTimestamp + activeAccessCount < timestampLimit)
    (opened : openSegment
        (fun closed roots count =>
          MemoryOpenSegment.deriveFor (identity candidate) authority closed
            roots count)
        headers precommit activeAccessCount initial canOpen activeCountInRange
        endTimestampInRange = .active active)
    (rowBeforeExact : rowBefore = .active active)
    (delayed : ProductionMemoryRowTrace.DelayedRun verify headers
      rowBefore claimReceipts final) :
    exists segmentCount allBatches,
      receipts allBatches = claimReceipts /\
        Chain candidate schema verify headers initial allBatches final
          segmentCount /\
        0 < segmentCount /\
        allBatches = delayed.batches := by
  subst rowBefore
  simpa using completeFromPrefix authority precommit activeAccessCount canOpen
    activeCountInRange endTimestampInRange opened [] rfl
      (BatchRun.nil (.active active)) delayed

end Nightstream.Implementation.NebulaV2.ProductionMemoryRowSegments
