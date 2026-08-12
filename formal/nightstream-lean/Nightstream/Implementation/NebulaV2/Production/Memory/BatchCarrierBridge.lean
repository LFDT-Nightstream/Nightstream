import Nightstream.Implementation.NebulaV2.Production.Carrier.FieldNativeFullClaim
import Nightstream.Implementation.NebulaV2.Production.Memory.CheckedBatchRows

/-!
Contract: physical carrier bridge from one production full claim to the
field-native checked memory batch.

`Placement` identifies the mixed claim coordinates that an independently
decoded full-claim carrier owns. It states exact counter-word and native-field
placement only. It does not assume equality of typed claims, a memory
transition, NIFS acceptance, or the final soundness conclusion.

Satisfying batch rows plus this physical placement recover the exact ordered
memory batch inside the full claim and consume that same batch.

Does not own the full-claim parser rows, absolute generated columns, NIFS
verification, state hashing, terminal verification, or Rust refinement.

Emits constraints: no; it proves the meaning of shared physical columns.
-/

set_option autoImplicit false
set_option maxRecDepth 100000

namespace Nightstream.Implementation.NebulaV2.ProductionMemoryBatchCarrierBridge

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.NebulaV2.ProductionMemoryCheckedBatchRows
open Nightstream.Implementation.NebulaV2.ProductionMemorySuffixCarrier
open Nightstream.Protocol.NebulaV2
open Nightstream.Protocol.NebulaV2.FPrime
open Nightstream.Protocol.NebulaV2.ProductState
open Nightstream.Protocol.NebulaV2.ProductionBatchedFPrime
open Nightstream.Protocol.NebulaV2.ProductionProfileCandidates
open Nightstream.Protocol.NebulaV2.WasmStatement
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation

abbrev Batch (candidate : Id) :=
  ProductionMemoryBatchPoseidonBinding.Batch candidate

/-- The indexed claim selected by the exact-length batch list. -/
def claimAt {candidate : Id} (batch : Batch candidate)
    (index : Fin (checkedStepsPerFreshClaim candidate)) :
    MemoryClaimCodec.Claim :=
  batch.suffixes.get (Fin.cast batch.length_exact.symm index)

theorem claimAt_mem
    {candidate : Id} (batch : Batch candidate)
    (index : Fin (checkedStepsPerFreshClaim candidate)) :
    claimAt batch index ∈ batch.suffixes := by
  exact List.get_mem _ _

theorem ofFn_claimAt
    {candidate : Id} (batch : Batch candidate) :
    List.ofFn (claimAt batch) = batch.suffixes := by
  have reindex := List.ofFn_congr batch.length_exact
    (List.get batch.suffixes)
  rw [List.ofFn_get] at reindex
  exact reindex.symm

/-- Generated physical placement of the full claim's memory subsection.

The counter side is a complete bounded word, not only its decoded integer.
The native side covers every one of the 76 non-counter field slots. A future
full-claim parser-row theorem must derive this placement from canonical input
columns. -/
structure Placement
    {candidate : Id} (layout : Layout candidate)
    (assignment : Nat -> Nat) (batch : Batch candidate) : Prop where
  counterWord : forall index counter,
    ((layout.steps index).claim.counters.word counter).digits assignment =
      WasmStateCodec.encodeWord counter.width
        (counter.claimValue (claimAt batch index))
  nativeField : forall index slot,
    assignment ((layout.steps index).claim.nativeFieldColumn slot) =
      (claimAt batch index).fieldValue slot.tag

private theorem counter_value_eq
    {candidate : Id} {layout : Layout candidate}
    {assignment : Nat -> Nat} {headers : ChainHeaders Digest.Value}
    (result : Result layout assignment headers)
    (batch : Batch candidate)
    (placement : Placement layout assignment batch)
    (batchCanonical : forall claim, claim ∈ batch.suffixes ->
      MemoryClaimCodec.Claim.Canonical claim)
    (index : Fin (StepCount candidate))
    (counter : MemoryClaimCounterRows.Counter) :
    counter.claimValue (result.claim index) =
      counter.claimValue (claimAt batch index) := by
  apply WasmStateCodec.encodeWord_injective_of_bound
    (width := counter.width)
  · rw [counter.claimValue_eq_tag, counter.width_eq_tag]
    exact MemoryClaimCodec.Claim.fieldValue_lt_width
      (result.claimParsed index).canonical counter.tag
  · rw [counter.claimValue_eq_tag, counter.width_eq_tag]
    exact MemoryClaimCodec.Claim.fieldValue_lt_width
      (batchCanonical _ (claimAt_mem batch index)) counter.tag
  · exact (result.counterWord index counter).symm.trans
      (placement.counterWord index counter)

private theorem native_value_eq
    {candidate : Id} {layout : Layout candidate}
    {assignment : Nat -> Nat} {headers : ChainHeaders Digest.Value}
    (result : Result layout assignment headers)
    (batch : Batch candidate)
    (placement : Placement layout assignment batch)
    (index : Fin (StepCount candidate))
    (slot : MemoryClaimFieldRows.Slot) :
    (result.claim index).fieldValue slot.tag =
      (claimAt batch index).fieldValue slot.tag := by
  have parsed := (result.claimParsed index).fields slot
  change assignment ((layout.steps index).claim.nativeFieldColumn slot) =
    (result.claim index).fieldValue slot.tag at parsed
  exact parsed.symm.trans (placement.nativeField index slot)

/-- Every row-derived suffix is the exact suffix at the same full-claim
carrier position. -/
theorem claim_eq_claimAt
    {candidate : Id} {layout : Layout candidate}
    {assignment : Nat -> Nat} {headers : ChainHeaders Digest.Value}
    (result : Result layout assignment headers)
    (batch : Batch candidate)
    (placement : Placement layout assignment batch)
    (batchCanonical : forall claim, claim ∈ batch.suffixes ->
      MemoryClaimCodec.Claim.Canonical claim)
    (index : Fin (StepCount candidate)) :
    result.claim index = claimAt batch index := by
  apply MemoryClaimCodec.Claim.fieldValue_injective
  funext tag
  cases tag with
  | segmentIndex =>
      simpa [MemoryClaimCounterRows.Counter.claimValue,
        MemoryClaimCodec.Claim.fieldValue] using
        (counter_value_eq result batch placement batchCanonical index
          .segmentIndex)
  | stepIndex =>
      simpa [MemoryClaimCounterRows.Counter.claimValue,
        MemoryClaimCodec.Claim.fieldValue] using
        (counter_value_eq result batch placement batchCanonical index
          .stepIndex)
  | timestampIn =>
      simpa [MemoryClaimCounterRows.Counter.claimValue,
        MemoryClaimCodec.Claim.fieldValue] using
        (counter_value_eq result batch placement batchCanonical index
          .timestampIn)
  | timestampOut =>
      simpa [MemoryClaimCounterRows.Counter.claimValue,
        MemoryClaimCodec.Claim.fieldValue] using
        (counter_value_eq result batch placement batchCanonical index
          .timestampOut)
  | segmentStartTimestamp =>
      simpa [MemoryClaimCounterRows.Counter.claimValue,
        MemoryClaimCodec.Claim.fieldValue] using
        (counter_value_eq result batch placement batchCanonical index
          .segmentStartTimestamp)
  | segmentEndTimestamp =>
      simpa [MemoryClaimCounterRows.Counter.claimValue,
        MemoryClaimCodec.Claim.fieldValue] using
        (counter_value_eq result batch placement batchCanonical index
          .segmentEndTimestamp)
  | activeAccessCount =>
      simpa [MemoryClaimCounterRows.Counter.claimValue,
        MemoryClaimCodec.Claim.fieldValue] using
        (counter_value_eq result batch placement batchCanonical index
          .activeAccessCount)
  | challenge repetition coordinate limb =>
      exact native_value_eq result batch placement index
        (.challenge repetition coordinate limb)
  | product side repetition role limb =>
      exact native_value_eq result batch placement index
        (.product side repetition role limb)
  | root stage role lane =>
      exact native_value_eq result batch placement index
        (.root stage role lane)

/-- The complete row-derived suffix batch is the exact full-claim memory
subsection. -/
theorem suffixBatch_eq
    {candidate : Id} {layout : Layout candidate}
    {assignment : Nat -> Nat} {headers : ChainHeaders Digest.Value}
    (result : Result layout assignment headers)
    (batch : Batch candidate)
    (placement : Placement layout assignment batch)
    (batchCanonical : forall claim, claim ∈ batch.suffixes ->
      MemoryClaimCodec.Claim.Canonical claim) :
    result.suffixBatch = batch := by
  apply SuffixBatch.ext
  calc
    List.ofFn result.claim = List.ofFn (claimAt batch) := by
      apply congrArg List.ofFn
      funext index
      exact claim_eq_claimAt result batch placement batchCanonical index
    _ = batch.suffixes := ofFn_claimAt batch

/-- Central no-split-authority theorem for the production memory block.
There is no caller-supplied batch result or transition premise. -/
theorem rows_bind_and_consume_full_claim_memory
    {candidate : Id}
    {fullShape : Phi81Relation.Shape}
    {layout : Layout candidate} (valid : layout.Valid)
    {assignment : Nat -> Nat}
    (headers : ChainHeaders Digest.Value)
    (canonicalAssignment : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (headersPlaced : HeadersPlaced layout assignment headers)
    (satisfied : Satisfies (rows layout) assignment)
    (value : ProductionFieldNativeFullClaim.Value candidate fullShape)
    (valueCanonical : value.Canonical)
    (placement : Placement layout assignment value.memory) :
    let result := derive valid headers canonicalAssignment one
      headersPlaced satisfied
    result.suffixBatch = value.memory /\
      ConsumesList MemoryProductBalanceRows.ConcreteBalanced
        (result.semantic 0) value.memory.suffixes
        (result.semantic (Fin.last (StepCount candidate))) := by
  let result := derive valid headers canonicalAssignment one
    headersPlaced satisfied
  have batchEqual := suffixBatch_eq result value.memory placement
    valueCanonical.memoryCanonical
  constructor
  · exact batchEqual
  · rw [← batchEqual]
    exact result.consumes_suffixBatch

end Nightstream.Implementation.NebulaV2.ProductionMemoryBatchCarrierBridge
