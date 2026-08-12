import Nightstream.Protocol.NebulaV2.ApplicationBatchCompletion
import Nightstream.Protocol.NebulaV2.ProductionBatchedAugmentedLifecycle

/-!
Contract: completed-WASM extraction from one exact batch-aware F-prime
lifetime.

Assurance tier: model-level.

Owns the final reverse bridge from the application batches paired with the
same delayed memory claims to `CompletedExecution`. The extra premise is an
exact decoded row value, including the typed terminal payload. It is not an
assumed execution relation.

Does not own generated row decoding, claim extraction, fingerprint security,
commitment binding, recursive verification, terminal proof verification,
Rust refinement, or external bytes.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Protocol.NebulaV2.ProductionBatchedCompletion

open Nightstream.Protocol.NebulaV2.ApplicationBatch
open Nightstream.Protocol.NebulaV2.ApplicationBatchCompletion
open Nightstream.Protocol.NebulaV2.ApplicationTrace
open Nightstream.Protocol.NebulaV2.AugmentedLifecycle
open Nightstream.Protocol.NebulaV2.Completion
open Nightstream.Protocol.NebulaV2.FPrime
open Nightstream.Protocol.NebulaV2.Lifecycle
open Nightstream.Protocol.NebulaV2.Ports
open Nightstream.Protocol.NebulaV2.ProductionBatchedAugmentedLifecycle
open Nightstream.Protocol.NebulaV2.ProductionBatchedFPrime
open Nightstream.Protocol.NebulaV2.ProductionProfileCandidates
open Nightstream.Protocol.NebulaV2.WasmState

/-- Public completion fields that are not consequences of the operational
row relation. `rowsExact` fixes the complete typed terminal row and canonical
padding, not only their row-kind tags. -/
structure ExactCompletedRows
    {ChallengeField : Type} [Field ChallengeField]
    {candidate : Id} {schema : Schema} {Digest Program : Type}
    {verify : ProductionBatchedAugmentedLifecycle.BatchVerifier candidate
      schema Digest ChallengeField}
    {derive : ClosedCarry Digest -> Roots Digest -> Nat ->
      ProductState.Challenges ChallengeField}
    {headers : ChainHeaders Digest}
    {machine : Machine Program} {program : Program}
    {links : ClaimBatchLink candidate schema Digest Program verify machine
      program}
    {initialApplication : AppStateVector}
    {initialMemory finalMemory : ClosedCarry Digest}
    {result : ExecutionResult AppStateVector Digest}
    {segmentCount : Nat}
    (run : CompleteRun candidate schema Digest Program verify derive headers
      machine program links initialApplication initialMemory
      result.finalApplicationState finalMemory segmentCount) where
  activeRows : List NormalizedRow
  terminalRow : NormalizedRow
  rowsExact : run.applicationRows =
    activeRows.map ApplicationRow.active ++
      [terminalApplicationRow terminalRow result.outcome] ++
      List.replicate
        (segmentCapacity segmentCount - result.realApplicationRowCount)
        .padding
  realRowCountExact :
    result.realApplicationRowCount = activeRows.length + 1
  segmentCountBound : segmentCount <= Lifecycle.maximumSegments
  realRowCountBound : result.realApplicationRowCount < realApplicationRowLimit
  fitsDeclaredSegments :
    result.realApplicationRowCount <= segmentCapacity segmentCount
  smallestSegmentCount :
    segmentCount = minimumSegmentCount result.realApplicationRowCount

namespace ExactCompletedRows

/-- Exact typed rows and the lifetime application run reconstruct a complete
execution. The operational semantics comes from `run.application`; it is not
a field of `ExactCompletedRows`. -/
theorem completedExecution
    {ChallengeField : Type} [Field ChallengeField]
    {candidate : Id} {schema : Schema} {Digest Program : Type}
    {verify : ProductionBatchedAugmentedLifecycle.BatchVerifier candidate
      schema Digest ChallengeField}
    {derive : ClosedCarry Digest -> Roots Digest -> Nat ->
      ProductState.Challenges ChallengeField}
    {headers : ChainHeaders Digest}
    {machine : Machine Program} {program : Program}
    {links : ClaimBatchLink candidate schema Digest Program verify machine
      program}
    {initialApplication : AppStateVector}
    {initialMemory finalMemory : ClosedCarry Digest}
    {result : ExecutionResult AppStateVector Digest}
    {segmentCount : Nat}
    {run : CompleteRun candidate schema Digest Program verify derive headers
      machine program links initialApplication initialMemory
      result.finalApplicationState finalMemory segmentCount}
    (completed : ExactCompletedRows run) :
    Nonempty
      (CompletedExecution machine.semantics program initialApplication result
        segmentCount) := by
  rcases run.application_executes with ⟨count, applicationRun⟩
  rw [completed.rowsExact] at applicationRun
  have countFromRows := applicationRun.count_eq_realRowCount
  have countExact : count = result.realApplicationRowCount := by
    rw [countFromRows, completed.realRowCountExact]
    rw [realRowCount_append, realRowCount_append]
    simp
  have exactRun : ApplicationBatch.Runs machine program initialApplication
      (completed.activeRows.map ApplicationRow.active ++
        [terminalApplicationRow completed.terminalRow result.outcome] ++
        List.replicate
          (segmentCapacity segmentCount - result.realApplicationRowCount)
          .padding)
      result.finalApplicationState result.realApplicationRowCount := by
    simpa [countExact] using applicationRun
  exact completedExecution_of_exact_rows exactRun
    completed.realRowCountExact run.positiveSegments
    completed.segmentCountBound completed.realRowCountBound
    completed.fitsDeclaredSegments completed.smallestSegmentCount

/-- The exact lifetime still exposes the delayed base, recursive, and
trailing-terminal schedule used by the same completed execution. -/
theorem completedExecution_and_delayedSchedule
    {ChallengeField : Type} [Field ChallengeField]
    {candidate : Id} {schema : Schema} {Digest Program : Type}
    {verify : ProductionBatchedAugmentedLifecycle.BatchVerifier candidate
      schema Digest ChallengeField}
    {derive : ClosedCarry Digest -> Roots Digest -> Nat ->
      ProductState.Challenges ChallengeField}
    {headers : ChainHeaders Digest}
    {machine : Machine Program} {program : Program}
    {links : ClaimBatchLink candidate schema Digest Program verify machine
      program}
    {initialApplication : AppStateVector}
    {initialMemory finalMemory : ClosedCarry Digest}
    {result : ExecutionResult AppStateVector Digest}
    {segmentCount : Nat}
    {run : CompleteRun candidate schema Digest Program verify derive headers
      machine program links initialApplication initialMemory
      result.finalApplicationState finalMemory segmentCount}
    (completed : ExactCompletedRows run) :
    Nonempty
        (CompletedExecution machine.semantics program initialApplication result
          segmentCount) /\
      CompleteSchedule run.claims.length :=
  ⟨completed.completedExecution, run.complete_schedule⟩

end ExactCompletedRows

end Nightstream.Protocol.NebulaV2.ProductionBatchedCompletion
