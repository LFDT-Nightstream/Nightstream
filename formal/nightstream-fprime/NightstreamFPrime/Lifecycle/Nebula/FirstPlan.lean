import NightstreamFPrime.Lifecycle.Nebula.ProgramSoundness

/-! Owns the plan of the first memory-application package (spec §4): ROM and
RAM of four words each, two ports, segments of two steps with four scan slots
each, at most four segments, and 5-bit timestamps. The ROM holds the machine
program `loadi 5; store 0; load 0; halt`, and the RAM starts at zero. This is
a test geometry for the first end-to-end run; a production plan selects its
own. -/

namespace NightstreamFPrime.Lifecycle.Nebula.FirstPlan

open NightstreamFPrime.Spec.Nebula

/-- The ROM words `op + 4 · arg` of `Machine`: `loadi 5`, `store 0`, `load 0`,
`halt`. -/
def romImage : List ℕ := [23, 2, 1, 0]

def plan : Plan where
  r := 2
  μ := 2
  wTs := 5
  bOps := 2
  bScan := 4
  n := 2
  sMax := 4
  rom := fun a => romImage.getD a 0
  ram := fun _ => 0

theorem two : plan.bOps = 2 := rfl

theorem valid : plan.Valid where
  exactCover := by decide
  timestampRange := by decide
  fieldEncoding := by decide
  laneBits := by decide
  addressWidth := le_rfl
  belowModulus := by decide
  positive := by decide
  romWords := by
    intro a below
    change a < 4 at below
    interval_cases a <;> decide
  ramWords := fun _ _ => by simp [plan]

theorem secure : plan.Secure := by
  unfold Plan.Secure
  decide

/-- The first memory application as a closed Stage 1 program. -/
def program : Stage1.Application.Program := MemoryApp.program plan two

end NightstreamFPrime.Lifecycle.Nebula.FirstPlan
