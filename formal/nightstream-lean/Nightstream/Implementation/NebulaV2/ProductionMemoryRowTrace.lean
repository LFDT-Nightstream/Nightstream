import Mathlib.Algebra.Field.TransferInstance
import Nightstream.Implementation.NebulaV2.ConcreteField
import Nightstream.Implementation.NebulaV2.MemoryOpenSegment
import Nightstream.Implementation.NebulaV2.ProductionApplicationBatchBridge
import Nightstream.Implementation.NebulaV2.ProductionMemoryStepSemantics
import Nightstream.Protocol.NebulaV2.ProductionBatchedAugmentedLifecycle

/-!
Contract: an exact delayed F-prime trace annotated by row-derived memory
batches.

`BatchEvidence` binds one verified complete production claim to the exact
checked-step row result whose suffix batch it contains. `DelayedRun` keeps
that row evidence at every delayed consumer, including the trailing terminal
claim. Erasing the annotations gives the independent production delayed run.

No receipt, suffix batch, transition, or segment continuation can be replaced
independently inside this trace.

Does not own construction of recursive invocations, segment partitioning,
snapshot coverage, challenge security, or deployed-verifier extraction.

Assurance tier: implementation-to-protocol bridge.

Emits constraints: no.
-/

set_option autoImplicit false
set_option maxRecDepth 100000

namespace Nightstream.Implementation.NebulaV2.ProductionMemoryRowTrace

open Nightstream.Implementation.NebulaV2.MemoryClaimProductUpdate
open Nightstream.Implementation.NebulaV2.MemoryProductBalanceRows
open Nightstream.Implementation.NebulaV2.ProductionMemoryStepSemantics
open Nightstream.Protocol.NebulaV2
open Nightstream.Protocol.NebulaV2.AugmentedLifecycle
open Nightstream.Protocol.NebulaV2.FPrime
open Nightstream.Protocol.NebulaV2.ProductState
open Nightstream.Protocol.NebulaV2.ProductionBatchedFPrime
open Nightstream.Protocol.NebulaV2.ProductionBatchedGlobalFPrime
open Nightstream.Protocol.NebulaV2.ProductionProfileCandidates
open Nightstream.SuperNeo.Concrete

/-- The executable coefficient pair uses the proved field equivalence. -/
noncomputable local instance concreteKField : Field K :=
  ConcreteField.superNeoEquiv.field

/-- Multiplication transported through the field equivalence is the concrete
SuperNeo multiplication used by the row balance checker. -/
theorem transferred_mul_eq_concrete (left right : K) :
    left * right = K.mul left right := by
  apply ConcreteField.superNeoEquiv.injective
  change ConcreteField.superNeoEquiv left *
      ConcreteField.superNeoEquiv right =
    ConcreteField.superNeoEquiv (K.mul left right)
  exact (ConcreteField.superNeoEquiv_mul left right).symm

/-- A concrete closing equation gives the independent product-state balance
predicate. -/
theorem concreteBalanced_implies_balanced
    {products : ProductState.State K}
    (balanced : ConcreteBalanced products) :
    ProductState.Balanced products := by
  intro repetition
  change (products repetition).initialSnapshot *
      (products repetition).writes =
    (products repetition).reads *
      (products repetition).finalSnapshot
  rw [transferred_mul_eq_concrete, transferred_mul_eq_concrete]
  exact balanced repetition

/-- One verified production receipt and the exact row result that supplied
its complete memory suffix batch. -/
structure BatchEvidence
    (candidate : Id) (schema : ProductionBatchedFPrime.Schema)
    (verify : BatchVerifier candidate schema Digest.Value K)
    (headers : ChainHeaders Digest.Value) where
  receipt : Receipt candidate schema Digest.Value K verify
  layout : ProductionMemoryCheckedBatchRows.Layout candidate
  assignment : Nat -> Nat
  result : ProductionMemoryCheckedBatchRows.Result layout assignment headers
  memoryExact : result.suffixBatch = receipt.claim.memory

namespace BatchEvidence

variable {candidate : Id} {schema : ProductionBatchedFPrime.Schema}
variable {verify : BatchVerifier candidate schema Digest.Value K}
variable {headers : ChainHeaders Digest.Value}

def before (batch : BatchEvidence candidate schema verify headers) :
    ConcreteCarry :=
  batch.result.semantic 0

def after (batch : BatchEvidence candidate schema verify headers) :
    ConcreteCarry :=
  batch.result.semantic
    (Fin.last (ProductionMemoryCheckedBatchRows.StepCount candidate))

def steps (batch : BatchEvidence candidate schema verify headers) :
    List ProductionMemoryStepSemantics.Step :=
  List.ofFn fun index =>
    ProductionMemoryStepSemantics.Step.ofResult batch.result index

/-- Exact ordered application accesses decoded from this producer batch. -/
def accesses (batch : BatchEvidence candidate schema verify headers) :
    List Access :=
  ProductionApplicationBatchBridge.memoryAccesses batch.result

private theorem flatMap_ofFn
    {Alpha Beta : Type} {count : Nat}
    (values : Fin count -> Alpha) (function : Alpha -> List Beta) :
    List.flatMap function (List.ofFn values) =
      (List.ofFn fun index => function (values index)).flatten := by
  induction count with
  | zero => rfl
  | succ count inductionHypothesis =>
      rw [List.ofFn_succ, List.ofFn_succ]
      simp only [List.flatMap_cons, List.flatten_cons]
      rw [inductionHypothesis (fun index => values index.succ)]

/-- The application-port view and the proof-independent step view preserve
the same physical-order access list. -/
theorem accesses_eq_steps
    (batch : BatchEvidence candidate schema verify headers) :
    batch.accesses = ProductionMemoryStepSemantics.Run.accesses batch.steps := by
  simp [accesses, steps, ProductionApplicationBatchBridge.memoryAccesses,
    ProductionApplicationBatchBridge.stepAccesses,
    ProductionMemoryStepSemantics.Run.accesses,
    ProductionMemoryStepSemantics.Step.ofResult]
  simpa [ProductionMemoryStepSemantics.Step.ofResult] using
    (flatMap_ofFn
      (fun index => ProductionMemoryStepSemantics.Step.ofResult
        batch.result index)
      ProductionMemoryStepSemantics.Step.accesses).symm

/-- All `E` semantic steps are derived from this row result. -/
theorem stepRun (batch : BatchEvidence candidate schema verify headers) :
    ProductionMemoryStepSemantics.Run batch.before batch.steps batch.after := by
  exact ProductionMemoryStepSemantics.Run.ofBatch batch.result

/-- The stripped semantic-step claims are the exact suffix list inside the
verified complete claim. -/
theorem claimsExact (batch : BatchEvidence candidate schema verify headers) :
    ProductionMemoryStepSemantics.Run.claims batch.steps =
      batch.receipt.claim.memory.suffixes := by
  rw [ProductionMemoryStepSemantics.Run.claims, steps, List.map_ofFn]
  change
    (List.ofFn fun index =>
      batch.result.claim index) =
      batch.receipt.claim.memory.suffixes
  have exact := congrArg
    (fun memory : SuffixBatch candidate Digest.Value (Challenges K) (State K) =>
      memory.suffixes) batch.memoryExact
  simpa [ProductionMemoryCheckedBatchRows.Result.suffixBatch] using exact

/-- The exact row transition consumes the suffix batch of the same verified
complete claim. -/
theorem transition (batch : BatchEvidence candidate schema verify headers) :
    ProductionBatchedFPrime.Transition verify ProductState.Balanced
      batch.before batch.receipt batch.after := by
  refine ⟨?_⟩
  change ConsumesList ProductState.Balanced batch.before
    batch.receipt.claim.memory.suffixes batch.after
  rw [← batch.claimsExact]
  exact batch.stepRun.toConsumesList.mono
    (fun products balanced => concreteBalanced_implies_balanced balanced)

end BatchEvidence

/-- One continuation with the exact challenge authority read from the same
recursive assignment. The authority is local because its prior-state and
running-accumulator digests change across F-prime invocations. -/
structure BoundContinuation
    (candidate : Id)
    (headers : ChainHeaders Digest.Value)
    (intermediate outgoing : ConcreteCarry) where
  authority : MemoryOpenSegment.Authority
  exact : Continues
    (fun closed precommit activeAccessCount =>
      MemoryOpenSegment.deriveFor (identity candidate) authority closed
        precommit activeAccessCount)
    headers intermediate outgoing

/-- Exact delayed schedule with one row-derived batch attached to every
verified receipt. The terminal constructor consumes the trailing batch and
does not create a successor. -/
inductive DelayedRun
    {candidate : Id} {schema : ProductionBatchedFPrime.Schema}
    (verify : BatchVerifier candidate schema Digest.Value K)
    (headers : ChainHeaders Digest.Value) :
    ConcreteCarry ->
      List (Receipt candidate schema Digest.Value K verify) ->
      ClosedCarry Digest.Value -> Type
  | terminal
      (batch : BatchEvidence candidate schema verify headers)
      (final : ClosedCarry Digest.Value)
      (closedExact : batch.after = .closed final) :
      DelayedRun verify headers batch.before [batch.receipt] final
  | recursive
      {outgoing : ConcreteCarry}
      {tail : List (Receipt candidate schema Digest.Value K verify)}
      {final : ClosedCarry Digest.Value}
      (batch : BatchEvidence candidate schema verify headers)
      (continues : BoundContinuation candidate headers batch.after outgoing)
      (rest : DelayedRun verify headers outgoing tail final) :
      DelayedRun verify headers batch.before
        (batch.receipt :: tail) final

namespace DelayedRun

variable {candidate : Id} {schema : ProductionBatchedFPrime.Schema}
variable {verify : BatchVerifier candidate schema Digest.Value K}
variable {headers : ChainHeaders Digest.Value}

/-- Exact producer batches retained by the delayed schedule. -/
def batches
    {before : ConcreteCarry}
    {receipts : List (Receipt candidate schema Digest.Value K verify)}
    {final : ClosedCarry Digest.Value}
    (run : DelayedRun verify headers before receipts final) :
    List (BatchEvidence candidate schema verify headers) :=
  match run with
  | .terminal batch _ _ => [batch]
  | .recursive batch _ rest => batch :: rest.batches

/-- Exact ordered producer accesses retained by the delayed schedule. -/
def accesses
    {before : ConcreteCarry}
    {receipts : List (Receipt candidate schema Digest.Value K verify)}
    {final : ClosedCarry Digest.Value}
    (run : DelayedRun verify headers before receipts final) :
    List Access :=
  run.batches.flatMap BatchEvidence.accesses

@[simp] theorem batches_terminal
    (batch : BatchEvidence candidate schema verify headers)
    (final : ClosedCarry Digest.Value)
    (closedExact : batch.after = .closed final) :
    (@DelayedRun.terminal candidate schema verify headers batch final
      closedExact).batches = [batch] := by
  rfl

@[simp] theorem batches_recursive
    {outgoing : ConcreteCarry}
    {tail : List (Receipt candidate schema Digest.Value K verify)}
    {final : ClosedCarry Digest.Value}
    (batch : BatchEvidence candidate schema verify headers)
    (continues : BoundContinuation candidate headers batch.after outgoing)
    (rest : DelayedRun verify headers outgoing tail final) :
    (@DelayedRun.recursive candidate schema verify headers outgoing tail
      final batch continues rest).batches = batch :: rest.batches := by
  rfl

@[simp] theorem accesses_terminal
    (batch : BatchEvidence candidate schema verify headers)
    (final : ClosedCarry Digest.Value)
    (closedExact : batch.after = .closed final) :
    (@DelayedRun.terminal candidate schema verify headers batch final
      closedExact).accesses = batch.accesses := by
  simp [accesses]

@[simp] theorem accesses_recursive
    {outgoing : ConcreteCarry}
    {tail : List (Receipt candidate schema Digest.Value K verify)}
    {final : ClosedCarry Digest.Value}
    (batch : BatchEvidence candidate schema verify headers)
    (continues : BoundContinuation candidate headers batch.after outgoing)
    (rest : DelayedRun verify headers outgoing tail final) :
    (@DelayedRun.recursive candidate schema verify headers outgoing tail
      final batch continues rest).accesses =
      batch.accesses ++ rest.accesses := by
  simp [accesses]

/-- Every annotated receipt has an accepted complete claim. -/
theorem everyReceiptAccepted
    {before : ConcreteCarry}
    {receipts : List (Receipt candidate schema Digest.Value K verify)}
    {final : ClosedCarry Digest.Value}
    (run : DelayedRun verify headers before receipts final) :
    forall receipt, receipt ∈ receipts ->
      verify receipt.proof receipt.claim :=
  by
    induction run with
    | terminal batch final closedExact =>
        intro receipt member
        simp only [List.mem_singleton] at member
        subst receipt
        exact batch.receipt.accepted
    | recursive batch continues rest inductionHypothesis =>
        intro receipt member
        rcases List.mem_cons.mp member with rfl | later
        · exact batch.receipt.accepted
        · exact inductionHypothesis receipt later

end DelayedRun

end Nightstream.Implementation.NebulaV2.ProductionMemoryRowTrace
