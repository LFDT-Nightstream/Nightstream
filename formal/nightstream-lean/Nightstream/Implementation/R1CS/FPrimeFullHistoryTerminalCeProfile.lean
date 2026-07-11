import Nightstream.Implementation.R1CS.FPrimeRecursiveManifestSchema

/-! Generated exact-shape manifest for the direct terminal-CE claim compiler. -/

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalCeProfile

def claimCount : Nat := 14
def rowCount : Nat := 21542
def definitionCount : Nat := 5904
def checkCount : Nat := 15638
def canonicalColumnCount : Nat := 7956
def claimRanges : List (Nat × Nat × Nat) := [(3775026, 3796568, 3212217), (3796568, 3818110, 3218391), (3818110, 3839652, 3224565), (3839652, 3861194, 3230739), (3861194, 3882736, 3236913), (3882736, 3904278, 3243087), (3904278, 3925820, 3249261), (3925820, 3947362, 3255435), (3947362, 3968904, 3261609), (3968904, 3990446, 3267783), (3990446, 4011988, 3273957), (4011988, 4033530, 3280131), (4033530, 4055072, 3286305), (4055072, 4076614, 3292479)]
def phaseRanges : List (String × Nat × Nat) := [("terminal_ce.claim.commitment", 0, 972), ("terminal_ce.claim.public_input", 972, 14850), ("terminal_ce.claim.norm", 14850, 15390), ("terminal_ce.claim.evaluations", 15390, 15784), ("terminal_ce.claim.constant_term", 15784, 15790), ("terminal_ce.claim.nc_channel", 15790, 21542)]

def phaseRange (phase : String × Nat × Nat) :
FPrimeRecursiveManifest.RowRange where
name := phase.1
rowStart := phase.2.1
rowEnd := phase.2.2
nonzeroEntries := 0
sha256 := ""

theorem claim_count : claimRanges.length = claimCount := by native_decide
theorem phase_schedule : phaseRanges.map Prod.fst =
["terminal_ce.claim.commitment", "terminal_ce.claim.public_input",
"terminal_ce.claim.norm", "terminal_ce.claim.evaluations",
"terminal_ce.claim.constant_term", "terminal_ce.claim.nc_channel"] := by
native_decide
theorem phase_coverage :
let ranges := phaseRanges.map phaseRange
FPrimeRecursiveManifest.covers 0 rowCount ranges = true := by
native_decide

end Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalCeProfile
