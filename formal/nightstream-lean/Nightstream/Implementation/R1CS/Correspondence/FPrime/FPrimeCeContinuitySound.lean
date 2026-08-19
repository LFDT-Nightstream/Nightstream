import Nightstream.Implementation.R1CS.Ownership.FPrime.FPrimeCeContinuityArtifact

/-!
Contract: universal soundness of the exact one-claim CE continuity rows.

Every authority coordinate is equated directly between the prior PiDEC child
view and the next PiCCS running view: commitments, public projection, shape,
evaluation points, range point, ring evaluations, constant terms, z-column
evaluations, and fold digest. Compact digests are not substituted for these
wire equalities.
-/

set_option maxRecDepth 32768

namespace Nightstream.Implementation.R1CS.FPrimeCeContinuitySound

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.FPrimeCeContinuity

def Holds (z : Nat → Nat) : Prop :=
  ∀ pair ∈ columnPairs, z pair.1 = z pair.2

private theorem equalityRow_mem {pair : Nat × Nat} (member : pair ∈ columnPairs) :
    equalityRow pair ∈ continuityRows :=
  List.mem_map.mpr ⟨pair, member, rfl⟩

private theorem equality_of_row {z : Nat → Nat}
    (hcanon : ∀ column, z column < goldilocksP)
    (hone : z 0 = 1) {left right : Nat}
    (holds : RowHolds z (equalityRow (left, right))) : z left = z right := by
  have leftLt := hcanon left
  have rightLt := hcanon right
  simp only [equalityRow, RowHolds, lcEval, List.foldl, hone,
    goldilocksP] at holds leftLt rightLt
  omega

/-- `CIR-FPR-CE-CONTINUITY`: all 1,297 carried CE coordinates agree for
every satisfying one-claim continuity assignment. -/
theorem fPrimeCeContinuity_sound {z : Nat → Nat}
    (hcanon : ∀ column, z column < goldilocksP)
    (hone : z 0 = 1) (hsat : Satisfies continuityRows z) : Holds z := by
  intro pair member
  rcases pair with ⟨left, right⟩
  exact equality_of_row hcanon hone (hsat _ (equalityRow_mem member))

end Nightstream.Implementation.R1CS.FPrimeCeContinuitySound
