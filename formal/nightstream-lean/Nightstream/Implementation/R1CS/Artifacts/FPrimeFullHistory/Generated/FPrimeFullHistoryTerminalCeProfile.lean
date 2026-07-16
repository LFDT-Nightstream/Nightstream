import Nightstream.Implementation.R1CS.Ownership.FPrimeRecursive.FPrimeRecursiveManifestSchema

/-! Generated exact-shape manifest for the direct terminal-CE claim compiler. -/

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalCeProfile

def claimCount : Nat := 14
def rowCount : Nat := 21542
def definitionCount : Nat := 5904
def checkCount : Nat := 15638
def canonicalColumnCount : Nat := 7956
def claimRanges : List (Nat × Nat × Nat) := [(3891546, 3913088, 3495737), (3913088, 3934630, 3501911), (3934630, 3956172, 3508085), (3956172, 3977714, 3514259), (3977714, 3999256, 3520433), (3999256, 4020798, 3526607), (4020798, 4042340, 3532781), (4042340, 4063882, 3538955), (4063882, 4085424, 3545129), (4085424, 4106966, 3551303), (4106966, 4128508, 3557477), (4128508, 4150050, 3563651), (4150050, 4171592, 3569825), (4171592, 4193134, 3575999)]
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
