import Nightstream.Implementation.Nebula.Application.Ports.Refinement
import Nightstream.Implementation.Nebula.Production.Memory.CheckedBatchRows
import Nightstream.Protocol.Nebula.ApplicationBatch

/-!
Contract: exact application-to-memory bridge for one production batch.

The memory rows derive 63 physical operation slots for each checked step.
This file reshapes those same slots into `E * 3` normalized application rows
and proves exact ordered access equality. `Matches` is the narrow generated
application-row boundary: it requires equality of the complete normalized
row lists, not only equality of counts or digests.

Does not own generated application-transition rows, lifecycle-tag columns,
the WASM row compiler, NIFS, state hashing, Rust, or cryptography.

Emits constraints: no. It gives application meaning to row-derived memory
ports and states the exact equality that generated application rows must
establish.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.Nebula.ProductionApplicationBatchBridge

open Nightstream.Protocol.Nebula
open Nightstream.Protocol.Nebula.ApplicationBatch
open Nightstream.Protocol.Nebula.Ports
open Nightstream.Protocol.Nebula.ProductionProfileCandidates

def stepNormalizedRows
    {candidate : Id}
    {layout : ProductionMemoryCheckedBatchRows.Layout candidate}
    {assignment : Nat -> Nat}
    {headers : FPrime.ChainHeaders Digest.Value}
    (result : ProductionMemoryCheckedBatchRows.Result layout assignment headers)
    (kinds : Fin (ProductionMemoryCheckedBatchRows.StepCount candidate) ->
      ApplicationRowIndex -> NormalizedRowKind)
    (index : Fin (ProductionMemoryCheckedBatchRows.StepCount candidate)) :
    List NormalizedRow :=
  ApplicationPortRefinement.rows (result.source index).operation (kinds index)

def memoryNormalizedRows
    {candidate : Id}
    {layout : ProductionMemoryCheckedBatchRows.Layout candidate}
    {assignment : Nat -> Nat}
    {headers : FPrime.ChainHeaders Digest.Value}
    (result : ProductionMemoryCheckedBatchRows.Result layout assignment headers)
    (kinds : Fin (ProductionMemoryCheckedBatchRows.StepCount candidate) ->
      ApplicationRowIndex -> NormalizedRowKind) : List NormalizedRow :=
  (List.ofFn fun index => stepNormalizedRows result kinds index).flatten

def stepAccesses
    {candidate : Id}
    {layout : ProductionMemoryCheckedBatchRows.Layout candidate}
    {assignment : Nat -> Nat}
    {headers : FPrime.ChainHeaders Digest.Value}
    (result : ProductionMemoryCheckedBatchRows.Result layout assignment headers)
    (index : Fin (ProductionMemoryCheckedBatchRows.StepCount candidate)) :
    List Access :=
  ApplicationPortRefinement.accesses (result.source index).operation

def memoryAccesses
    {candidate : Id}
    {layout : ProductionMemoryCheckedBatchRows.Layout candidate}
    {assignment : Nat -> Nat}
    {headers : FPrime.ChainHeaders Digest.Value}
    (result : ProductionMemoryCheckedBatchRows.Result layout assignment headers) :
    List Access :=
  (List.ofFn fun index => stepAccesses result index).flatten

private theorem flatten_ofFn_length
    {alpha : Type} {count width : Nat} (blocks : Fin count -> List alpha)
    (each : forall index, (blocks index).length = width) :
    (List.ofFn blocks).flatten.length = count * width := by
  rw [List.length_flatten]
  have constant : ∀ value, value ∈ (List.ofFn blocks).map List.length →
      value = width := by
    intro value member
    rcases List.mem_map.mp member with ⟨block, blockMember, rfl⟩
    rcases List.mem_ofFn.mp blockMember with ⟨index, rfl⟩
    exact each index
  rw [List.sum_eq_card_nsmul _ width constant]
  simp

theorem memoryNormalizedRows_length
    {candidate : Id}
    {layout : ProductionMemoryCheckedBatchRows.Layout candidate}
    {assignment : Nat -> Nat}
    {headers : FPrime.ChainHeaders Digest.Value}
    (result : ProductionMemoryCheckedBatchRows.Result layout assignment headers)
    (kinds : Fin (ProductionMemoryCheckedBatchRows.StepCount candidate) ->
      ApplicationRowIndex -> NormalizedRowKind) :
    (memoryNormalizedRows result kinds).length =
      ApplicationBatch.rowsPerFreshClaim candidate := by
  unfold memoryNormalizedRows ApplicationBatch.rowsPerFreshClaim
  exact flatten_ofFn_length _ fun index =>
    ApplicationPortRefinement.rows_length
      (result.source index).operation (kinds index)

private theorem flatMap_flatten
    {alpha beta : Type} (function : alpha -> List beta)
    (blocks : List (List alpha)) :
    blocks.flatten.flatMap function =
      (blocks.map fun block => block.flatMap function).flatten := by
  induction blocks with
  | nil => rfl
  | cons head tail inductionHypothesis =>
      simp [inductionHypothesis]

/-- The row-major application view has exactly the ordered access list from
the physical memory slots, including active ports after holes. -/
theorem memoryNormalizedRows_accesses
    {candidate : Id}
    {layout : ProductionMemoryCheckedBatchRows.Layout candidate}
    {assignment : Nat -> Nat}
    {headers : FPrime.ChainHeaders Digest.Value}
    (result : ProductionMemoryCheckedBatchRows.Result layout assignment headers)
    (kinds : Fin (ProductionMemoryCheckedBatchRows.StepCount candidate) ->
      ApplicationRowIndex -> NormalizedRowKind) :
    (memoryNormalizedRows result kinds).flatMap NormalizedRow.accesses =
      memoryAccesses result := by
  rw [memoryNormalizedRows, flatMap_flatten]
  rw [List.map_ofFn]
  apply congrArg List.flatten
  apply List.ofFn_inj.mpr
  funext index
  exact ApplicationPortRefinement.rows_flatMap_accesses
    (result.source index).operation (kinds index)

/-- Each checked-step access list has the active count decoded from the same
memory claim. -/
theorem stepAccesses_length
    {candidate : Id}
    {layout : ProductionMemoryCheckedBatchRows.Layout candidate}
    {assignment : Nat -> Nat}
    {headers : FPrime.ChainHeaders Digest.Value}
    (result : ProductionMemoryCheckedBatchRows.Result layout assignment headers)
    (index : Fin (ProductionMemoryCheckedBatchRows.StepCount candidate)) :
    (stepAccesses result index).length =
      (result.claim index).activeAccessCount := by
  exact ApplicationPortRefinement.accesses_length_eq_claimActiveCount
    (result.source index).operation

/-- Exact generated-row boundary. A deployment must derive this equality
from shared physical columns and application-transition rows. -/
def Matches
    {Program : Type} {candidate : Id}
    {machine : WasmState.Machine Program} {program : Program}
    {before after : WasmState.AppStateVector}
    {layout : ProductionMemoryCheckedBatchRows.Layout candidate}
    {assignment : Nat -> Nat}
    {headers : FPrime.ChainHeaders Digest.Value}
    (result : ProductionMemoryCheckedBatchRows.Result layout assignment headers)
    (batch : ApplicationBatch.Batch candidate machine program before after) :
    Prop :=
  ∃ kinds : Fin (ProductionMemoryCheckedBatchRows.StepCount candidate) →
      ApplicationRowIndex → NormalizedRowKind,
    ApplicationBatch.normalizedRows batch.rows =
      memoryNormalizedRows result kinds

namespace Matches

/-- A matched application transition has no memory effect outside the exact
row-derived physical operation ports. -/
theorem accesses_exact
    {Program : Type} {candidate : Id}
    {machine : WasmState.Machine Program} {program : Program}
    {before after : WasmState.AppStateVector}
    {layout : ProductionMemoryCheckedBatchRows.Layout candidate}
    {assignment : Nat -> Nat}
    {headers : FPrime.ChainHeaders Digest.Value}
    {result : ProductionMemoryCheckedBatchRows.Result layout assignment headers}
    {batch : ApplicationBatch.Batch candidate machine program before after}
    (matched : Matches result batch) :
    ApplicationBatch.accesses batch.rows = memoryAccesses result := by
  rcases matched with ⟨kinds, normalizedExact⟩
  calc
    ApplicationBatch.accesses batch.rows =
        (ApplicationBatch.normalizedRows batch.rows).flatMap
          NormalizedRow.accesses :=
      (ApplicationBatch.normalizedRows_flatMap_accesses batch.rows).symm
    _ = (memoryNormalizedRows result kinds).flatMap
          NormalizedRow.accesses := by rw [normalizedExact]
    _ = memoryAccesses result :=
      memoryNormalizedRows_accesses result kinds

end Matches

end Nightstream.Implementation.Nebula.ProductionApplicationBatchBridge
