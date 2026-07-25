import Nightstream.Implementation.R1CS.Correspondence.FPrime.FPrimeEncodingSound
import Nightstream.Implementation.Rust.CanonicalConformance.NativeStep.CanonicalPlainCarrierLink

/-!
Contract: recover the exact canonical `enc_inst` bit at every output of the
532-row F-prime encoding owner.

Owns:
- a physical 64-bit vector reconstructed from each lane's Boolean public-bit
  columns;
- equality of that vector's integer value with the encoding owner's digest
  column;
- equality of every physical bit with the independently defined typed
  canonical digest encoder.

Does not own: placement of this isolated owner inside a larger artifact,
terminal-link rows, Rust-source refinement, or Poseidon2 digest semantics.

Emits constraints: no; it interprets the exact encoding ownership artifact.
-/

namespace Nightstream.Implementation.R1CS.FPrimeEncodingCanonicalBits

open Nightstream.Implementation.Encoding.FPrime
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.FPrimeEncoding
open Nightstream.Implementation.Rust.CanonicalConformance.NativeStep

set_option maxRecDepth 32768

/-- Reconstruct the low `width` values of an arbitrary little-endian bit
source. Each new value is prepended as the next most-significant bit. -/
def bitVectorPrefix
    (value : Nat → Nat) : (width : Nat) → BitVec width
  | 0 => 0
  | width + 1 =>
      BitVec.cons
        (decide (value width = 1))
        (bitVectorPrefix value width)

/-- Reconstruct the low `width` physical public-bit columns as a bitvector,
preserving the production's little-endian `publicBitCol lane bit` indexing. -/
def physicalLanePrefix
    (z : Nat → Nat) (lane : Nat) (width : Nat) : BitVec width :=
  bitVectorPrefix
    (fun bit => z (publicBitCol lane bit)) width

def physicalLane (z : Nat → Nat) (lane : Fin 4) : BitVec 64 :=
  physicalLanePrefix z lane.val 64

private def lanePrefixValue
    (z : Nat → Nat) (lane width : Nat) : Nat :=
  (List.range width).foldl
    (fun total bit =>
      total + 2 ^ bit * z (publicBitCol lane bit)) 0

theorem bitVectorPrefix_getLsbD
    (value : Nat → Nat) (width bit : Nat)
    (bitLt : bit < width) :
    (bitVectorPrefix value width).getLsbD bit =
      decide (value bit = 1) := by
  induction width with
  | zero => omega
  | succ width inductionHypothesis =>
      rw [bitVectorPrefix, BitVec.getLsbD_cons]
      by_cases top : bit = width
      · simp [top]
      · rw [if_neg top]
        apply inductionHypothesis
        omega

private theorem physicalLanePrefix_getLsbD
    (z : Nat → Nat) (lane width bit : Nat)
    (bitLt : bit < width) :
    (physicalLanePrefix z lane width).getLsbD bit =
      decide (z (publicBitCol lane bit) = 1) := by
  exact bitVectorPrefix_getLsbD _ width bit bitLt

private theorem booleanValue
    {value : Nat} (valueLe : value ≤ 1) :
    (decide (value = 1)).toNat = value := by
  by_cases one : value = 1
  · simp [one]
  · have zero : value = 0 := by omega
    simp [zero]

private theorem physicalLanePrefix_toNat
    {z : Nat → Nat}
    (holds : FPrimeEncodingSound.Holds z)
    {lane width : Nat}
    (laneLt : lane < 4)
    (widthLe : width ≤ 64) :
    (physicalLanePrefix z lane width).toNat =
      lanePrefixValue z lane width := by
  induction width with
  | zero =>
      simp [physicalLanePrefix, bitVectorPrefix, lanePrefixValue]
  | succ width inductionHypothesis =>
      have widthLt : width < 64 := by omega
      have bitLe :=
        holds.publicBoolean lane laneLt width widthLt
      rw [physicalLanePrefix, bitVectorPrefix, BitVec.toNat_cons',
        booleanValue bitLe, lanePrefixValue,
        List.range_succ, List.foldl_append]
      simp only [List.foldl_cons, List.foldl_nil]
      rw [
        Nat.shiftLeft_eq,
        Nat.mul_comm
          (z (publicBitCol lane width)) (2 ^ width)
      ]
      have prefixEqual :
          (bitVectorPrefix
            (fun bit => z (publicBitCol lane bit)) width).toNat =
            lanePrefixValue z lane width := by
        simpa [physicalLanePrefix] using
          inductionHypothesis (by omega)
      rw [prefixEqual]
      exact Nat.add_comm _ _

private theorem physicalLane_toNat
    {z : Nat → Nat}
    (holds : FPrimeEncodingSound.Holds z)
    (lane : Fin 4) :
    (physicalLane z lane).toNat =
      FPrimeEncodingSound.laneBitsValue z lane.val := by
  exact physicalLanePrefix_toNat holds lane.isLt (by decide)

/-- Canonical typed digest reconstructed from the four physical digest
columns; no carried digest is accepted as authority. -/
def digestOfAssignment
    (z : Nat → Nat)
    (canonical : ∀ column, z column < goldilocksP) :
    Digest :=
  fun lane => ⟨z (digestCol lane.val), canonical (digestCol lane.val)⟩

private theorem physicalLane_eq_encodedLane
    {z : Nat → Nat}
    (canonical : ∀ column, z column < goldilocksP)
    (holds : FPrimeEncodingSound.Holds z)
    (lane : Fin 4) :
    physicalLane z lane =
      encodeEncInst (digestOfAssignment z canonical) lane := by
  apply BitVec.eq_of_toNat_eq
  rw [physicalLane_toNat holds lane,
    ← (holds.laneCanonical lane.val lane.isLt).1]
  simp only [encodeEncInst, encodeLane, BitVec.toNat_ofNat,
    digestOfAssignment]
  apply (Nat.mod_eq_of_lt _).symm
  have digestLt := canonical (digestCol lane.val)
  have modulusLt : goldilocksP < 2 ^ 64 := by decide
  omega

/-- Every public-bit output of the exact encoding owner equals the
independently defined canonical bit of the digest reconstructed from its four
physical digest columns. -/
theorem publicBit_eq_encodedBit
    {z : Nat → Nat}
    (canonical : ∀ column, z column < goldilocksP)
    (holds : FPrimeEncodingSound.Holds z)
    (lane : Fin 4) (bit : Fin 64) :
    z (publicBitCol lane.val bit.val) =
      CanonicalPlainCarrierLink.encodedBit
        (digestOfAssignment z canonical) lane bit := by
  have physicalBit :=
    physicalLanePrefix_getLsbD
      z lane.val 64 bit.val bit.isLt
  change
    (physicalLane z lane).getLsbD bit.val =
      decide (z (publicBitCol lane.val bit.val) = 1) at physicalBit
  rw [physicalLane_eq_encodedLane canonical holds lane] at physicalBit
  have bitLe :=
    holds.publicBoolean lane.val lane.isLt bit.val bit.isLt
  unfold CanonicalPlainCarrierLink.encodedBit
  rw [physicalBit]
  by_cases one : z (publicBitCol lane.val bit.val) = 1
  · simp [one]
  · have zero : z (publicBitCol lane.val bit.val) = 0 := by
      omega
    simp [zero]

end Nightstream.Implementation.R1CS.FPrimeEncodingCanonicalBits
