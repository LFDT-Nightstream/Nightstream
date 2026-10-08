import NightstreamFPrime.Lifecycle.Nebula.ProgramSoundness

/-! Owns the plan of the second memory-application test package, for the spec
§14 cases that the first plan cannot reach. ROM has four words and RAM eight,
so `r < μ` and rows O6 exist. A segment has one step (`N = 1`), there are at
most four segments, and timestamps have 4 bits. The ROM holds
`load 0; store 1; load 1; store 0`, so every step uses both ports and four
segments end at the largest reachable timestamp `S_max · N · B_ops = 8`. RAM
starts with `7` at address 0. This is a test geometry, not a production plan. -/

namespace NightstreamFPrime.Lifecycle.Nebula.SecondPlan

open NightstreamFPrime.Spec.Nebula

/-- The ROM words `op + 4 · arg` of `Machine`: `load 0`, `store 1`, `load 1`,
`store 0`. -/
def romImage : List ℕ := [1, 6, 5, 2]

def ramImage : List ℕ := [7]

def plan : Plan where
  r := 2
  μ := 3
  wTs := 4
  bOps := 2
  bScan := 12
  n := 1
  sMax := 4
  rom := fun a => romImage.getD a 0
  ram := fun a => ramImage.getD a 0

theorem two : plan.bOps = 2 := rfl

theorem valid : plan.Valid where
  exactCover := by decide
  timestampRange := by decide
  fieldEncoding := by decide
  laneBits := by decide
  addressWidth := by decide
  belowModulus := by decide
  positive := by decide
  romWords := by
    intro a below
    change a < 4 at below
    interval_cases a <;> decide
  ramWords := by
    intro a below
    change a < 8 at below
    interval_cases a <;> decide

theorem secure : plan.Secure := by
  unfold Plan.Secure
  decide

/-- The second memory application as a closed Stage 1 program. -/
def program : Stage1.Application.Program := MemoryApp.program plan two

end NightstreamFPrime.Lifecycle.Nebula.SecondPlan
