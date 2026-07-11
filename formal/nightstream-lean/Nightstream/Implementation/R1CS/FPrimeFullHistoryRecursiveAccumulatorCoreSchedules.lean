import Nightstream.Implementation.R1CS.FPrimeFullHistoryRecursiveAccumulatorCoreSchedule0
import Nightstream.Implementation.R1CS.FPrimeFullHistoryRecursiveAccumulatorCoreSchedule1
import Nightstream.Implementation.R1CS.FPrimeFullHistoryRecursiveAccumulatorCoreSchedule2
import Nightstream.Implementation.R1CS.FPrimeFullHistoryRecursiveAccumulatorCoreSchedule3
import Nightstream.Implementation.R1CS.FPrimeFullHistoryRecursiveAccumulatorCoreSchedule4
import Nightstream.Implementation.R1CS.FPrimeFullHistoryRecursiveAccumulatorCoreSchedule5
import Nightstream.Implementation.R1CS.FPrimeFullHistoryRecursiveAccumulatorCoreSchedule6
import Nightstream.Implementation.R1CS.FPrimeFullHistoryRecursiveAccumulatorCoreSchedule7
import Nightstream.Implementation.R1CS.FPrimeFullHistoryRecursiveAccumulatorCoreSchedule8
import Nightstream.Implementation.R1CS.FPrimeFullHistoryRecursiveAccumulatorCoreSchedule9
import Nightstream.Implementation.R1CS.FPrimeFullHistoryRecursiveAccumulatorCoreSchedule10
import Nightstream.Implementation.R1CS.FPrimeFullHistoryRecursiveAccumulatorCoreSchedule11

/-! Aggregate certificate for every exact recursive-accumulator-core shifted-ternary schedule. -/

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryRecursiveAccumulatorCoreSchedule

open Nightstream.Implementation.R1CS

theorem rows_schedule (index : Nat) (indexLt : index < 192) :
    let map := FPrimeFullHistoryRecursiveAccumulatorCore.shiftedTernaryMaps.getD index default
    ((FPrimeFullHistoryRecursiveAccumulatorCoreCheckCoverage.shiftedOwnerRows map).drop
        (FPrimeFullHistoryRecursiveAccumulatorCoreCheckCoverage.shiftedLocalRowStart map)).take 124 =
    ShiftedTernaryCompiler.canonicalRows.map
      (Relabel.row
        (FPrimeFullHistoryRecursiveAccumulatorCoreCheckCoverage.columnMap map)) := by
  have shardCases : index < 16 ∨ (16 ≤ index ∧ index < 32) ∨ (32 ≤ index ∧ index < 48) ∨ (48 ≤ index ∧ index < 64) ∨ (64 ≤ index ∧ index < 80) ∨ (80 ≤ index ∧ index < 96) ∨ (96 ≤ index ∧ index < 112) ∨ (112 ≤ index ∧ index < 128) ∨ (128 ≤ index ∧ index < 144) ∨ (144 ≤ index ∧ index < 160) ∨ (160 ≤ index ∧ index < 176) ∨ (176 ≤ index ∧ index < 192) := by omega
  rcases shardCases with first | shard1 | shard2 | shard3 | shard4 | shard5 | shard6 | shard7 | shard8 | shard9 | shard10 | shard11
  · exact shard0_rows_schedule index (by omega) first
  · exact shard1_rows_schedule index shard1.1 shard1.2
  · exact shard2_rows_schedule index shard2.1 shard2.2
  · exact shard3_rows_schedule index shard3.1 shard3.2
  · exact shard4_rows_schedule index shard4.1 shard4.2
  · exact shard5_rows_schedule index shard5.1 shard5.2
  · exact shard6_rows_schedule index shard6.1 shard6.2
  · exact shard7_rows_schedule index shard7.1 shard7.2
  · exact shard8_rows_schedule index shard8.1 shard8.2
  · exact shard9_rows_schedule index shard9.1 shard9.2
  · exact shard10_rows_schedule index shard10.1 shard10.2
  · exact shard11_rows_schedule index shard11.1 shard11.2

end Nightstream.Implementation.R1CS.FPrimeFullHistoryRecursiveAccumulatorCoreSchedule
