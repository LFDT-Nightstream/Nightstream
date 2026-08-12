import Nightstream.Implementation.NebulaV2.Core.ConditionalEqualityRows

set_option autoImplicit false

namespace tests.NebulaV2ConditionalEqualityRows

open Nightstream.Implementation.NebulaV2.ConditionalEqualityRows
open Nightstream.Implementation.R1CS

def pairs : List (Nat × Nat) := [(2, 3), (4, 5)]

/-- In the closed phase, a changed inactive field cannot satisfy its gate. -/
theorem changed_closed_field_is_rejected
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (closed : assignment 1 = 0)
    (changed : assignment 2 ≠ assignment 3) :
    ¬ Satisfies (rows 1 pairs) assignment := by
  intro holds
  have equal := rows_sound_closed canonical one closed holds (2, 3)
    (by simp [pairs])
  exact changed equal

end tests.NebulaV2ConditionalEqualityRows
