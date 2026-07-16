import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryTerminalContinuityShard0
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryTerminalContinuityShard1
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryTerminalContinuityShard2
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryTerminalContinuityShard3
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryTerminalContinuityShard4
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryTerminalContinuityShard5
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryTerminalContinuityShard6
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryTerminalContinuityShard7
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryTerminalContinuityShard8
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryTerminalContinuityShard9
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryTerminalContinuityShard10
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryTerminalContinuityShard11
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryTerminalContinuityShard12
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryTerminalContinuityShard13

/-! Generated aggregate for the exact 14-child terminal continuity owner. -/

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalContinuity

open Nightstream.Implementation.R1CS

def rangeSha256 : String := "0b4d10fbf1323ab44f604d3236cbdd717c3e230c903d5fc9648ebc86c65ddd3a"
def rowStart : Nat := 3673497
def rowEnd : Nat := 3887263
def rowCount : Nat := 213766

def pairs : List (Nat × Nat) :=
  Generated0.pairs ++
  Generated1.pairs ++
  Generated2.pairs ++
  Generated3.pairs ++
  Generated4.pairs ++
  Generated5.pairs ++
  Generated6.pairs ++
  Generated7.pairs ++
  Generated8.pairs ++
  Generated9.pairs ++
  Generated10.pairs ++
  Generated11.pairs ++
  Generated12.pairs ++
  Generated13.pairs

def rows : List Row := EqualityPins.rows pairs

theorem shard_ranges_partition :
    Generated0.rowStart = rowStart ∧
    Generated0.rowEnd = Generated1.rowStart ∧
    Generated1.rowEnd = Generated2.rowStart ∧
    Generated2.rowEnd = Generated3.rowStart ∧
    Generated3.rowEnd = Generated4.rowStart ∧
    Generated4.rowEnd = Generated5.rowStart ∧
    Generated5.rowEnd = Generated6.rowStart ∧
    Generated6.rowEnd = Generated7.rowStart ∧
    Generated7.rowEnd = Generated8.rowStart ∧
    Generated8.rowEnd = Generated9.rowStart ∧
    Generated9.rowEnd = Generated10.rowStart ∧
    Generated10.rowEnd = Generated11.rowStart ∧
    Generated11.rowEnd = Generated12.rowStart ∧
    Generated12.rowEnd = Generated13.rowStart ∧
    Generated13.rowEnd = rowEnd := by native_decide

theorem pairs_length : pairs.length = rowCount := by
  simp only [pairs, List.length_append,
    Generated0.pairs_length,
    Generated1.pairs_length,
    Generated2.pairs_length,
    Generated3.pairs_length,
    Generated4.pairs_length,
    Generated5.pairs_length,
    Generated6.pairs_length,
    Generated7.pairs_length,
    Generated8.pairs_length,
    Generated9.pairs_length,
    Generated10.pairs_length,
    Generated11.pairs_length,
    Generated12.pairs_length,
    Generated13.pairs_length]
  decide

theorem rows_length : rows.length = rowCount := by
  simpa [rows, EqualityPins.rows] using pairs_length

theorem sound {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfies : Satisfies rows assignment) :
    ∀ pair ∈ pairs, assignment pair.1 = assignment pair.2 := by
  exact EqualityPins.rows_sound canonical one satisfies

theorem complete {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (equalities : ∀ pair ∈ pairs, assignment pair.1 = assignment pair.2) :
    Satisfies rows assignment := by
  exact EqualityPins.rows_complete canonical one equalities

end Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalContinuity
