import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryTerminalAccumulatorSegment0Instructions0
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryTerminalAccumulatorSegment0Inputs0
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryTerminalAccumulatorPoseidonHashes

/-!
Exact checked rows for the terminal post-fold accumulator owner.

| Branch | Mathematical obligation | Emits constraints |
|---|---|---|
| prefix | Pin the supported-profile constants and inactive-X zero | yes |
| digest | Poseidon2 over the 1,682-field accumulator-v1 projection | yes |

Owns: the direct accumulator digest computation.
Does not own: PiRLC parent authority or omitted y_zcol validation.
-/

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalAccumulator

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Program
open Nightstream.Implementation.R1CS.CheckedProgram

set_option maxRecDepth 1048576

def accumulatorClaimSourceColumns : List Nat :=
  FPrimeFullHistoryTerminalAccumulatorPoseidonHashes.accumulatorDigestTrace.inputColumns

def accumulatorDigestColumns : List Nat := [3489705, 3489706, 3489707, 3489708]
def rowStart : Nat := 3418586
def rowEnd : Nat := 3673497
def rowCount : Nat := 254911
def definitionCount : Nat := 27
def checkCount : Nat := 0

def segment0RowStart : Nat := 0
def segment0RowEnd : Nat := 27
def segment0InputColumns : List Nat :=
    Generated.segment0Inputs0
def segment0Instructions : List Instruction :=
    Generated.segment0Instructions0
def segment0Rows : List Row := CheckedProgram.rows segment0Instructions

theorem segment0_instructions_length :
    segment0Instructions.length = segment0RowEnd - segment0RowStart := by native_decide

theorem segment0_rows_length :
    segment0Rows.length = segment0RowEnd - segment0RowStart := by
  simpa [segment0Rows, CheckedProgram.rows] using segment0_instructions_length

theorem segment0_definitions_canonical :
    ∀ definition ∈ definitions segment0Instructions, definition.Canonical := by native_decide

theorem segment0_definitions_wellFormed :
    WellFormed segment0InputColumns (definitions segment0Instructions) := by native_decide

theorem segment0_checks_reference :
    ChecksReference
      (knownAfter segment0InputColumns (definitions segment0Instructions))
      segment0Instructions := by native_decide

def rowPieces : List (List Row) :=
  [segment0Rows, FPrimeFullHistoryTerminalAccumulatorPoseidonHashes.rows]

def rows : List Row := rowPieces.flatten

theorem rows_length : rows.length = rowCount := by
  simp only [rows, rowPieces, List.flatten_cons, List.flatten_nil,
    List.length_append, List.length_nil, segment0_rows_length,
    FPrimeFullHistoryTerminalAccumulatorPoseidonHashes.rows_length]
  native_decide

end Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalAccumulator
