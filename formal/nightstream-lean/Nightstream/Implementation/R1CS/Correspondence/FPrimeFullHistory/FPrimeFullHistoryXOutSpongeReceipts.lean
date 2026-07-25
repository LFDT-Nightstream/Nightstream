import Nightstream.Implementation.R1CS.Core.Poseidon2SpongeReceipt
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryBasePoseidonHashes
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryPriorLinkPoseidonHashes
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryRecursiveOutputPoseidonHashes
import Nightstream.Implementation.Rust.CanonicalConformance.NativeStep.StateXOutProgramRefinement

/-!
Contract: exact physical receipts for the three supported plain-state XOut
Poseidon2 sponge cores.

Assurance tier: artifact-checked.

Owns:
- derivation of the 23-field preimage length from the typed Rust source
  program;
- the resulting definitional 4,225-row/column sponge cost;
- exact absorb schedules for the base producer, recursive prior consumer, and
  recursive output producer;
- equality of each trace's actual owner-row slice with its reconstructed
  sponge rows;
- contiguous, duplicate-free row and fresh-column receipts for all three
  cores.

Does not own: the totalized optional-digest semantics of `paperHash`, its
alignment checks, the output presence coordinate, a `hashPrior` or `hashNext`
`CallRecipe`, whole-owner conservation, current whole-program generation,
compiled-Rust semantics, Poseidon2 native parity, or collision resistance.

Emits constraints: no. These theorems certify rows already emitted by the
three generated owners.
-/

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryXOutSpongeReceipts

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Poseidon2Sponge
open Nightstream.Implementation.Rust.CanonicalConformance.NativeStep

set_option maxRecDepth 524288
set_option maxHeartbeats 5000000

/-- Paper-shaped plain/stateless source program selected before inspecting
physical rows. -/
def sourceProgram : StateXOutProgram.Program :=
  StateXOutProgram.canonical false false

/-- Input-field cost computed by the selected source program. -/
def inputFields : Nat :=
  StateXOutProgram.cost sourceProgram

/-- Exact physical cost computed from `inputFields`, not measured from an
artifact after emission. -/
def physicalCost : Nat :=
  emissionCost inputFields

theorem sourceProgram_eq_generated :
    sourceProgram =
      StateXOutProgramRefinement.GeneratedProgram.select false false := by
  exact
    (StateXOutProgramRefinement.generated_eq_canonical false false).symm

theorem inputFields_eq : inputFields = 23 :=
  StateXOutProgram.statelessPlain_cost

theorem physicalCost_eq : physicalCost = 4225 := by
  native_decide

def baseTrace : Trace :=
  FPrimeFullHistoryBasePoseidonHashes.xOutTrace

def priorTrace : Trace :=
  FPrimeFullHistoryPriorLinkPoseidonHashes.priorXOutTrace

def recursiveOutputTrace : Trace :=
  FPrimeFullHistoryRecursiveOutputPoseidonHashes.xOutTrace

/-- Exact rate-four absorb and terminal-padding schedule induced by 23 input
fields. -/
def sourceSchedule : List ValueSchedule :=
  [.absorb 4, .absorb 4, .absorb 4, .absorb 4, .absorb 4, .absorb 3, .pad]

theorem baseSchedule_exact :
    valueSchedules baseTrace.rounds = sourceSchedule := by
  native_decide

theorem priorSchedule_exact :
    valueSchedules priorTrace.rounds = sourceSchedule := by
  native_decide

theorem recursiveOutputSchedule_exact :
    valueSchedules recursiveOutputTrace.rounds = sourceSchedule := by
  native_decide

/-- The exact base-state output sponge emission is the contiguous owner-row
slice `[6533, 10758)` and the contiguous fresh-column interval
`[6344, 10569)`. -/
theorem baseReceipt :
    EmissionReceipt baseTrace FPrimeFullHistoryBase.rows
      inputFields 6533 6344 := by
  constructor <;> native_decide

/-- The exact recursive prior-state consumer sponge emission is the
contiguous owner-row slice `[218, 4443)` and fresh-column interval
`[868073, 872298)`. -/
theorem priorReceipt :
    EmissionReceipt priorTrace FPrimeFullHistoryPriorLink.rows
      inputFields 218 868073 := by
  constructor <;> native_decide

/-- The exact recursive next-state output sponge emission is the contiguous
owner-row slice `[11, 4236)` and fresh-column interval
`[1127811, 1132036)`. -/
theorem recursiveOutputReceipt :
    EmissionReceipt recursiveOutputTrace FPrimeFullHistoryRecursiveOutput.rows
      inputFields 11 1127811 := by
  constructor <;> native_decide

theorem baseRows_exact_cost :
    baseTrace.rows.length = physicalCost := by
  exact baseReceipt.traceRows_length

theorem priorRows_exact_cost :
    priorTrace.rows.length = physicalCost := by
  exact priorReceipt.traceRows_length

theorem recursiveOutputRows_exact_cost :
    recursiveOutputTrace.rows.length = physicalCost := by
  exact recursiveOutputReceipt.traceRows_length

theorem base_conservation :
    baseTrace.rowIndices.length = baseTrace.allocatedColumns.length :=
  baseReceipt.row_column_conservation

theorem prior_conservation :
    priorTrace.rowIndices.length = priorTrace.allocatedColumns.length :=
  priorReceipt.row_column_conservation

theorem recursiveOutput_conservation :
    recursiveOutputTrace.rowIndices.length =
      recursiveOutputTrace.allocatedColumns.length :=
  recursiveOutputReceipt.row_column_conservation

/-- Column identities are erased before evaluation: all three generated cores
compute one identical pure sponge function on an equal 23-field vector. -/
theorem pureExecutions_equal
    (values : List Nat)
    (initialState : Nat → Nat) :
    runValueRounds baseTrace.rounds values initialState =
        runValueRounds priorTrace.rounds values initialState ∧
      runValueRounds baseTrace.rounds values initialState =
        runValueRounds recursiveOutputTrace.rounds values initialState := by
  constructor
  · exact runValueRounds_eq_of_schedules
      (baseSchedule_exact.trans priorSchedule_exact.symm) values initialState
  · exact runValueRounds_eq_of_schedules
      (baseSchedule_exact.trans recursiveOutputSchedule_exact.symm)
      values initialState

end Nightstream.Implementation.R1CS.FPrimeFullHistoryXOutSpongeReceipts
