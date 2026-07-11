import Nightstream.Implementation.R1CS.FPrimeBaseStateArtifact

/-!
Contract: universal soundness of the exact F' base-state authority pins.

Any canonical assignment satisfying the production row family carries the
verifier-derived value at every base-state coordinate. No digest is accepted
as authority merely because it is self-consistent: all four lanes of every
digest and all scalar coordinates are pinned independently.
-/

namespace Nightstream.Implementation.R1CS.FPrimeBaseStateSound

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.FPrimeBaseState

/-- Every verifier-owned `(column, value)` pair is present in the satisfying
assignment. -/
def Holds (z : Nat → Nat) : Prop :=
  ∀ pin ∈ pins, z pin.1 = pin.2

private theorem pinRow_mem {pin : Nat × Nat} (member : pin ∈ pins) :
    pinRow pin ∈ rows :=
  List.mem_map.mpr ⟨pin, member, rfl⟩

private theorem pin_of_row {z : Nat → Nat} (hcanon : ∀ column, z column < goldilocksP)
    (hone : z 0 = 1) {pin : Nat × Nat} (valueLt : pin.2 < goldilocksP)
    (holds : RowHolds z (pinRow pin)) : z pin.1 = pin.2 := by
  rcases pin with ⟨column, value⟩
  change value < goldilocksP at valueLt
  change RowHolds z (pinRow (column, value)) at holds
  change z column = value
  have columnLt := hcanon column
  by_cases valueZero : value = 0
  · subst value
    simp only [pinRow, ↓reduceIte, RowHolds, lcEval, List.foldl, hone,
      goldilocksP] at holds columnLt
    omega
  · simp only [pinRow, valueZero, ↓reduceIte, RowHolds, lcEval, List.foldl,
      hone, goldilocksP] at holds valueLt columnLt
    omega

/-- `CIR-FPR-BASE-PINS`: all 31 base authority coordinates equal their
preprocessing-derived constants for every satisfying assignment. -/
theorem fPrimeBaseState_sound {z : Nat → Nat}
    (hcanon : ∀ column, z column < goldilocksP)
    (hone : z 0 = 1) (hsat : Satisfies rows z) : Holds z := by
  intro pin member
  exact pin_of_row hcanon hone (pins_canonical pin member)
    (hsat _ (pinRow_mem member))

end Nightstream.Implementation.R1CS.FPrimeBaseStateSound
