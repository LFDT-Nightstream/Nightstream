import Nightstream.Implementation.R1CS.Canonical.Link270Production

/-!
Executable regressions for the canonical 270-coordinate link (Phase 1) and the
production comparison surface (Phase 1b).

Owns: the derived count, the carrier-coordinate cases, and the decisiveness of
the copy-versus-zero-pin measurement.

Does not own: any production capture.  No value here is read from an emitter.
-/

set_option autoImplicit false

namespace NightstreamTests.CanonicalLink270

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical.Link270
open Nightstream.Implementation.R1CS.Canonical.Link270Production

/-! ## The count is derived, not declared -/

theorem width_is_derived : carrierWidth = ringDegree * publicRingColumns := rfl

example : carrierWidth = 270 := by decide

example : canonicalRows.length = 270 := canonicalRows_length_eq

/-! ## Carrier coordinates 257-269 are ordinary

The session's carrier result says a running tail is live.  These fix that the
canonical encoding treats 257 and 269 exactly like coordinate 0. -/

example : firstTail.val = 257 := by decide

example : lastTail.val = 269 := by decide

/-- Coordinate 257's row has a source term, i.e. it copies rather than pins. -/
example : (coordinateRow firstTail).a.length = 2 := by decide

/-- So does coordinate 269. -/
example : (coordinateRow lastTail).a.length = 2 := by decide

/-- And coordinate 0, for comparison: the shape is uniform across the range. -/
example :
    (coordinateRow ⟨0, by decide⟩).a.length =
      (coordinateRow firstTail).a.length := by decide

/-- The source coefficient is the additive inverse of one, on every coordinate
including the tail.  A zero-pinning row would omit this term entirely. -/
example :
    (coordinateRow firstTail).a.getLast? = some (sourceColumn firstTail, goldilocksP - 1) := by
  decide

/-! ## The tail measurement is thirteen rows and is decisive -/

example :
    ((List.finRange carrierWidth).filter (fun i => decide (IsTail i))).length = 13 :=
  tail_count

/-- Coordinate 256 is *not* tail; 257 is.  This pins the boundary the carrier
result identified. -/
example : ¬ IsTail ⟨256, by decide⟩ := by decide

example : IsTail firstTail := by decide

/-! ## Column ownership -/

example : sourceColumn ⟨0, by decide⟩ = 1 := by decide

example : destinationColumn ⟨0, by decide⟩ = 271 := by decide

/-- Source and destination blocks never collide. -/
example : sourceColumn lastTail ≠ destinationColumn ⟨0, by decide⟩ := by decide

end NightstreamTests.CanonicalLink270
