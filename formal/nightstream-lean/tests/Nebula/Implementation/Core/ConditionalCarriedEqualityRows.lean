import Nightstream.Implementation.Nebula.Core.ConditionalCarriedEqualityRows

/-! Focused phase-gating checks for extension-field equality rows. -/

set_option autoImplicit false

namespace tests.NebulaConditionalCarriedEqualityRows

open Nightstream.Implementation.Nebula.ConditionalCarriedEqualityRows
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical.KHorner
open Nightstream.Implementation.R1CS.Canonical.KMul
open Nightstream.Implementation.R1CS.Canonical.LinCombNormal

def left : Carried :=
  { low := [(2, 1)]
    high := [(3, 1)] }

def right : Carried :=
  { low := [(4, 1)]
    high := [(5, 1)] }

theorem exact_row_count : (rows 1 left right).length = 2 := rfl

/-- Active steps satisfy the close-only equality rows without assuming the
two products are equal. -/
theorem active_phase_does_not_require_balance
    (assignment : Nat → Nat)
    (one : assignment 0 = 1)
    (active : assignment 1 = 1) :
    Satisfies (rows 1 left right) assignment :=
  rows_complete_active one active

/-- Closed steps extract both extension-field coordinates. -/
theorem closed_phase_requires_exact_balance
    (assignment : Nat → Nat)
    (one : assignment 0 = 1)
    (closed : assignment 1 = 0)
    (holds : Satisfies (rows 1 left right) assignment) :
    carriedValue assignment left = carriedValue assignment right := by
  apply rows_sound_closed one closed
  · simp [right, Program.CanonicalTerms, goldilocksP]
  · simp [right, Program.CanonicalTerms, goldilocksP]
  · exact holds

end tests.NebulaConditionalCarriedEqualityRows
