import Nightstream.Implementation.R1CS.Artifacts.FPrimeRecursive.PiRlcProjection.BetaLadder
import tests.Axioms.Support

/-! Fail-closed dependencies for the active beta-ladder artifact. -/

open Nightstream.Implementation.R1CS.FPrimeRecursivePiRlcProjection.BetaLadder

/-- info: 'Nightstream.Implementation.R1CS.FPrimeRecursivePiRlcProjection.BetaLadder.data_check' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms data_check

/-- info: 'Nightstream.Implementation.R1CS.FPrimeRecursivePiRlcProjection.BetaLadder.owned_row_count' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms owned_row_count

/-- info: 'Nightstream.Implementation.R1CS.FPrimeRecursivePiRlcProjection.BetaLadder.source_rows_match' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms source_rows_match
