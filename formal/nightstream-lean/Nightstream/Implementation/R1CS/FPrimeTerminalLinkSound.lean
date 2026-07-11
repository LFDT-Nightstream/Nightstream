import Nightstream.Implementation.R1CS.FPrimeTerminalLinkArtifact

/-!
Contract: universal soundness of the exact trailing-batch delayed-link rows.

Every canonical-residue assignment satisfying the production terminal-link
artifact fixes the fresh affine-one slot and equates all 256 fresh public bits
to the last F' step's `x_out` bits. Host-side nonempty and length rejection is
tested by the Rust exporter and remains explicit in the property contract.
-/

set_option maxRecDepth 32768

namespace Nightstream.Implementation.R1CS.FPrimeTerminalLinkSound

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.FPrimeTerminalLink

structure Holds (z : Nat → Nat) : Prop where
  affineOne : z freshOneCol = 1
  linked : ∀ bit, bit < 256 →
    z (freshBitCol bit) = z (lastXOutBitCol bit)

private theorem oneRow_mem : oneRow ∈ rows := by simp [rows]

private theorem linkRow_mem {bit : Nat} (bitLt : bit < 256) :
    linkRow bit ∈ rows := by
  simp only [rows, List.mem_cons]
  right
  exact List.mem_map.mpr ⟨bit, List.mem_range.mpr bitLt, rfl⟩

private theorem equality_of_link_row {z : Nat → Nat}
    (hcanon : ∀ column, z column < goldilocksP)
    (hone : z 0 = 1) {left right : Nat}
    (holds : RowHolds z
      ⟨[(left, 1), (right, goldilocksP - 1)], [(0, 1)], []⟩) :
    z left = z right := by
  have leftLt := hcanon left
  have rightLt := hcanon right
  simp only [RowHolds, lcEval, List.foldl, hone, goldilocksP] at holds leftLt rightLt
  omega

/-- Exact terminal-link rows close the delayed public-input obligation. -/
theorem fPrimeTerminalLink_sound {z : Nat → Nat}
    (hcanon : ∀ column, z column < goldilocksP)
    (hone : z 0 = 1) (hsat : Satisfies rows z) :
    Holds z := by
  refine {
    affineOne := ?_
    linked := ?_
  }
  · have row := hsat oneRow oneRow_mem
    have freshLt := hcanon freshOneCol
    simp only [oneRow, RowHolds, lcEval, List.foldl, hone, goldilocksP] at row freshLt
    omega
  · intro bit bitLt
    exact equality_of_link_row hcanon hone
      (hsat (linkRow bit) (linkRow_mem bitLt))

end Nightstream.Implementation.R1CS.FPrimeTerminalLinkSound
