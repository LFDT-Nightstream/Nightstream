import Mathlib.Algebra.Field.TransferInstance
import Nightstream.Implementation.Nebula.Application.Ports.Refinement
import Nightstream.Implementation.Nebula.Production.Memory.CheckedBatchRows

/-!
Contract: proof-independent semantic steps extracted from production memory
rows.

Each `Step` contains the exact operation list, snapshot records, fingerprint
chunk, product update, and F-prime transition derived from one checked-step
row block. `Run` composes these values in exact row order. A production batch
derives one run of exactly `E` steps.

No access list, record list, product endpoint, transition, or timestamp order
is an input to `Step.ofResult` or `Run.ofBatch`.

Does not own application-control row kinds, cross-batch continuation,
full-segment scan coverage, challenge security, or Rust refinement.

Assurance tier: implementation-to-protocol bridge.

Emits constraints: no. It gives independent meaning to existing checked rows.
-/

set_option autoImplicit false
set_option maxRecDepth 100000

namespace Nightstream.Implementation.Nebula.ProductionMemoryStepSemantics

open Nightstream.Implementation.Nebula.ApplicationPortRefinement
open Nightstream.Implementation.Nebula.MemoryClaimProductUpdate
open Nightstream.Implementation.Nebula.MemoryProductClaimBridge
open Nightstream.Implementation.Nebula.MemoryProductBalanceRows
open Nightstream.Implementation.Nebula.MemoryProductSemanticBridge
open Nightstream.Implementation.Nebula.MemoryProductUpdateRows
open Nightstream.Protocol.Nebula
open Nightstream.Protocol.Nebula.ConcreteLaneGeometry
open Nightstream.Protocol.Nebula.FPrime
open Nightstream.Protocol.Nebula.Fingerprint
open Nightstream.Protocol.Nebula.ProductState
open Nightstream.Protocol.Nebula.ProductionProfileCandidates
open Nightstream.Protocol.Nebula.SnapshotSlot
open Nightstream.SuperNeo.Concrete

abbrev ConcreteCarry :=
  Carry Digest.Value (Challenges K) (State K)

/-- Semantic value of one satisfying production checked-step block. -/
structure Step where
  before : ConcreteCarry
  after : ConcreteCarry
  claim : MemoryClaimCodec.Claim
  records : CheckedStepRecords
  accesses : List Access
  consumes : Consumes ConcreteBalanced before claim after
  ordered : Ordered claim.timestampIn accesses claim.timestampOut
  productUpdate :
    mapState claim.productsAfter =
      ProductState.update ConcreteField.encode
        (mapChallenges claim.challenge)
        (mapState claim.productsBefore) records.chunk
  readsExact : records.chunk.reads =
    (Memory.readTuples accesses : Multiset MemTuple)
  writesExact : records.chunk.writes =
    (Memory.writeTuples accesses : Multiset MemTuple)
  snapshotGlobalIndex : forall role slot,
    ((records.snapshot role slot).1).globalIndex =
      SnapshotSlot.globalIndex claim.stepIndex.val slot
  snapshotValid : forall role slot,
    let entry := (records.snapshot role slot).1
    entry.value < valueLimit /\
      entry.timestamp <= SnapshotSlotRows.boundaryValue claim role

namespace Step

/-- One row-derived result index produces one complete independent step. -/
def ofResult
    {candidate : Id}
    {layout : ProductionMemoryCheckedBatchRows.Layout candidate}
    {assignment : Nat -> Nat}
    {headers : ChainHeaders Digest.Value}
    (result : ProductionMemoryCheckedBatchRows.Result layout assignment headers)
    (index : Fin (ProductionMemoryCheckedBatchRows.StepCount candidate)) :
    Step := by
  let source := result.source index
  let records := source.records
  let accesses := ApplicationPortRefinement.accesses source.operation
  refine
    { before := result.semantic index.castSucc
      after := result.semantic index.succ
      claim := result.claim index
      records := records
      accesses := accesses
      consumes := result.consumesAt index
      ordered := ApplicationPortRefinement.ordered source.operation
        (result.consumesAt index).timestampAdvance
      productUpdate := result.productUpdate index
      readsExact := ?_
      writesExact := ?_
      snapshotGlobalIndex := ?_
      snapshotValid := ?_ }
  · simpa [records, accesses, MemorySourceRows.Sound.records,
      CheckedStepRecords.chunk] using
      ApplicationPortRefinement.readRecordMultiset_eq source.operation
  · simpa [records, accesses, MemorySourceRows.Sound.records,
      CheckedStepRecords.chunk] using
      ApplicationPortRefinement.writeRecordMultiset_eq source.operation
  · intro role slot
    cases role <;> rfl
  · intro role slot
    have valid := source.snapshot.valid role slot
    have cellValid := valid.cell_valid
    cases role <;>
      simpa [records, MemorySourceRows.Sound.records,
        CheckedStepRecords.snapshot, SnapshotChunkRows.Sound.records,
        SnapshotSlot.ValidAt.boundedTuple, SnapshotSlot.Value.tuple,
        SnapshotSlot.ValidAt.cellState] using cellValid

/-- The snapshot side of a step has exactly 64 records and no holes. -/
def snapshotList (step : Step) (role : SnapshotRole) : List MemTuple :=
  List.ofFn fun slot : Fin scanSlots => (step.records.snapshot role slot).1

private theorem activeRecords_snapshotRecords
    (records : Fin scanSlots -> BoundedTuple) :
    activeRecords (snapshotRecords records) = List.ofFn records := by
  rw [show snapshotRecords records =
      (List.ofFn records).map some by
    rw [List.map_ofFn]
    rfl]
  simp [activeRecords]

/-- The snapshot list is exactly the matching product chunk multiset. -/
theorem snapshotList_coe_eq_chunk
    (step : Step) (role : SnapshotRole) :
    (step.snapshotList role : Multiset MemTuple) =
      match role with
      | .initialSnapshot => step.records.chunk.initialSnapshot
      | .finalSnapshot => step.records.chunk.finalSnapshot := by
  cases role <;>
    simp only [snapshotList, CheckedStepRecords.chunk]
  all_goals
    simp only [activeRecordMultiset]
    rw [activeRecords_snapshotRecords]
    rfl

end Step

/-! ## Exact semantic runs -/

/-- Exact carry chaining for proof-independent row-derived steps. -/
inductive Run : ConcreteCarry -> List Step -> ConcreteCarry -> Prop
  | nil (state : ConcreteCarry) : Run state [] state
  | cons
      {tail : List Step} {final : ConcreteCarry}
      (head : Step)
      (rest : Run head.after tail final) :
      Run head.before (head :: tail) final

namespace Run

def claims (steps : List Step) : List MemoryClaimCodec.Claim :=
  steps.map Step.claim

def accesses (steps : List Step) : List Access :=
  steps.flatMap Step.accesses

def chunks (steps : List Step) : List ProductState.Chunk :=
  steps.map fun step => step.records.chunk

def snapshotLists (steps : List Step) (role : SnapshotRole) :
    List (List MemTuple) :=
  steps.map fun step => step.snapshotList role

/-- Semantic runs compose without changing checked-step order. -/
theorem append
    {before middle after : ConcreteCarry}
    {left right : List Step}
    (first : Run before left middle)
    (second : Run middle right after) :
    Run before (left ++ right) after := by
  induction first with
  | nil => exact second
  | cons head rest inductionHypothesis =>
      exact .cons head (inductionHypothesis second)

/-- Forgetting row sources gives the exact independent suffix transition
chain. -/
theorem toConsumesList
    {before after : ConcreteCarry} {steps : List Step}
    (run : Run before steps after) :
    ProductionBatchedFPrime.ConsumesList ConcreteBalanced before
      (claims steps) after := by
  induction run with
  | nil => exact .nil _
  | cons head rest inductionHypothesis =>
      exact .cons head.consumes inductionHypothesis

theorem fromClosedIsEmpty
    {closed : ClosedCarry Digest.Value} {after : ConcreteCarry}
    {steps : List Step}
    (run : Run (.closed closed) steps after) :
    steps = [] /\ after = .closed closed := by
  have exact := run.toConsumesList.from_closed_is_empty
  exact ⟨by simpa [claims] using exact.1, exact.2⟩

/-- Local row orders and exact carry chaining give one global integer
timestamp order for the complete step run. -/
theorem ordered
    {before after : ConcreteCarry} {steps : List Step}
    (run : Run before steps after) :
    Ordered (carryTimestamp before) (accesses steps)
      (carryTimestamp after) := by
  induction run with
  | nil => exact .nil _
  | cons head rest inductionHypothesis =>
      have current := head.ordered
      rw [head.consumes.timestampIn_eq_before,
        head.consumes.timestampOut_eq_after] at current
      simpa [accesses] using current.append inductionHypothesis

/-- Every read chunk is exactly the multiset of row-derived application
reads. -/
theorem readsCover (steps : List Step) :
    ((chunks steps).map ProductState.Chunk.reads).sum =
      (Memory.readTuples (accesses steps) : Multiset MemTuple) := by
  induction steps with
  | nil => simp [chunks, accesses, Memory.readTuples]
  | cons head tail inductionHypothesis =>
      simp only [chunks, List.map_cons, List.sum_cons, accesses,
        List.flatMap_cons]
      rw [head.readsExact]
      have tailExact :
          (List.map ProductState.Chunk.reads
            (List.map (fun step : Step => step.records.chunk) tail)).sum =
            (Memory.readTuples (List.flatMap Step.accesses tail) :
              Multiset MemTuple) := by
        simpa [chunks, accesses] using inductionHypothesis
      rw [tailExact]
      simp [Memory.readTuples]

/-- Every write chunk is exactly the multiset of row-derived application
writes. -/
theorem writesCover (steps : List Step) :
    ((chunks steps).map ProductState.Chunk.writes).sum =
      (Memory.writeTuples (accesses steps) : Multiset MemTuple) := by
  induction steps with
  | nil => simp [chunks, accesses, Memory.writeTuples]
  | cons head tail inductionHypothesis =>
      simp only [chunks, List.map_cons, List.sum_cons, accesses,
        List.flatMap_cons]
      rw [head.writesExact]
      have tailExact :
          (List.map ProductState.Chunk.writes
            (List.map (fun step : Step => step.records.chunk) tail)).sum =
            (Memory.writeTuples (List.flatMap Step.accesses tail) :
              Multiset MemTuple) := by
        simpa [chunks, accesses] using inductionHypothesis
      rw [tailExact]
      simp [Memory.writeTuples]

private theorem ofIndexed
    {count : Nat}
    (states : Fin (count + 1) -> ConcreteCarry)
    (steps : Fin count -> Step)
    (beforeExact : forall index,
      (steps index).before = states index.castSucc)
    (afterExact : forall index,
      (steps index).after = states index.succ) :
    Run (states 0) (List.ofFn steps) (states (Fin.last count)) := by
  induction count with
  | zero =>
      simpa using (Run.nil (states 0))
  | succ count inductionHypothesis =>
      rw [List.ofFn_succ]
      have headBefore := beforeExact 0
      have headAfter := afterExact 0
      let tailStates : Fin (count + 1) -> ConcreteCarry :=
        fun index => states index.succ
      let tailSteps : Fin count -> Step :=
        fun index => steps index.succ
      have tailBefore : forall index,
          (tailSteps index).before = tailStates index.castSucc := by
        intro index
        exact beforeExact index.succ
      have tailAfter : forall index,
          (tailSteps index).after = tailStates index.succ := by
        intro index
        exact afterExact index.succ
      have tailRun := inductionHypothesis tailStates tailSteps
        tailBefore tailAfter
      have tailRunExact : Run (steps 0).after
          (List.ofFn fun index => steps index.succ)
          (states (Fin.last (count + 1))) := by
        simpa [tailStates, tailSteps, headAfter] using tailRun
      have complete := Run.cons (steps 0) tailRunExact
      simpa [headBefore] using complete

/-- One candidate-specific row result gives exactly its `E` checked semantic
steps, with all internal carry boundaries shared. -/
theorem ofBatch
    {candidate : Id}
    {layout : ProductionMemoryCheckedBatchRows.Layout candidate}
    {assignment : Nat -> Nat}
    {headers : ChainHeaders Digest.Value}
    (result : ProductionMemoryCheckedBatchRows.Result layout assignment headers) :
    Run (result.semantic 0)
      (List.ofFn fun index => Step.ofResult result index)
      (result.semantic
        (Fin.last (ProductionMemoryCheckedBatchRows.StepCount candidate))) := by
  exact ofIndexed result.semantic (fun index => Step.ofResult result index)
    (fun _ => rfl) (fun _ => rfl)

end Run

end Nightstream.Implementation.Nebula.ProductionMemoryStepSemantics
