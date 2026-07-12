import Nightstream.Implementation.Encoding.FPrime
import Nightstream.Implementation.R1CS.Correspondence.CanonicalU64.CanonicalU64Sound
import Nightstream.Implementation.R1CS.Ownership.FPrime.FPrimeEncodingArtifact

/-!
Contract: `ENC-CANON` soundness for the exact production F' encoding rows.

Every canonical-residue assignment satisfying the 532-row artifact binds the
256 public bits to four canonical Goldilocks lanes. The result is quantified
over all satisfying assignments, not only the exported honest witness. Equal
accepted public bits imply equal digest lanes.

Assumptions: canonical assignment representatives, the constant-one wire, and
the Euclid divisor property of the Goldilocks modulus. These are the same
typed arithmetic boundaries as the canonical-u64 gadget theorem.
-/

set_option maxRecDepth 32768
set_option maxHeartbeats 4000000

namespace Nightstream.Implementation.R1CS.FPrimeEncodingSound

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.FPrimeEncoding

def laneBitsValue (z : Nat → Nat) (lane : Nat) : Nat :=
  (List.range 64).foldl
    (fun total bit => total + 2 ^ bit * z (publicBitCol lane bit)) 0

def decodedDigest (z : Nat → Nat) : Fin 4 → Nat :=
  fun lane => z (digestCol lane)

private theorem laneRows_satisfied {z : Nat → Nat}
    (hsat : Satisfies rows z) {lane : Nat} (laneLt : lane < 4) :
    Satisfies (laneRows lane) z := by
  intro row rowMem
  apply hsat row
  apply List.mem_flatMap.mpr
  exact ⟨lane, List.mem_range.mpr laneLt, rowMem⟩

private theorem canonicalRows_satisfied {z : Nat → Nat}
    (hsat : Satisfies rows z) {lane : Nat} (laneLt : lane < 4) :
    Satisfies CanonicalU64.rows (pullAssignment z (canonicalMap lane)) := by
  intro row rowMem
  apply (rowHolds_pull_iff z (canonicalMap lane) row).mpr
  apply laneRows_satisfied hsat laneLt
  apply List.mem_append_left
  exact List.mem_map.mpr ⟨row, rowMem, rfl⟩

private theorem pulledCanonical {z : Nat → Nat}
    (hcanon : ∀ column, z column < goldilocksP) (lane : Nat) :
    ∀ column, pullAssignment z (canonicalMap lane) column < goldilocksP :=
  fun column => hcanon (canonicalMap lane column)

private theorem pulledOne {z : Nat → Nat} (hone : z 0 = 1) (lane : Nat) :
    pullAssignment z (canonicalMap lane) 0 = 1 := by
  simpa [pullAssignment, canonicalMap] using hone

private theorem canonical_lane
    (hq : EuclidPrime goldilocksP) {z : Nat → Nat}
    (hcanon : ∀ column, z column < goldilocksP)
    (hone : z 0 = 1) (hsat : Satisfies rows z)
    {lane : Nat} (laneLt : lane < 4) :
    z (digestCol lane) = bitsValue (pullAssignment z (canonicalMap lane)) ∧
      bitsValue (pullAssignment z (canonicalMap lane)) < goldilocksP := by
  have sound := canonicalU64_sound hq (pulledCanonical hcanon lane)
    (pulledOne hone lane) (canonicalRows_satisfied hsat laneLt)
  simpa [pullAssignment, canonicalMap, CanonicalU64.varCol] using sound

private theorem equalityRow_mem {lane bit : Nat}
    (laneLt : lane < 4) (bitLt : bit < 64) :
    equalityRow lane bit ∈ rows := by
  apply List.mem_flatMap.mpr
  refine ⟨lane, List.mem_range.mpr laneLt, ?_⟩
  apply List.mem_append_right
  exact List.mem_map.mpr ⟨bit, List.mem_range.mpr bitLt, rfl⟩

private theorem canonical_bit_rows_prefix :
    (List.range 64).map (fun bit => bitRow (CanonicalU64.bitCol bit)) =
      CanonicalU64.rows.take 64 := by
  decide

private theorem canonical_bitRow_mem {bit : Nat} (bitLt : bit < 64) :
    bitRow (CanonicalU64.bitCol bit) ∈ CanonicalU64.rows := by
  have inGenerated : bitRow (CanonicalU64.bitCol bit) ∈
      (List.range 64).map (fun index =>
        bitRow (CanonicalU64.bitCol index)) := by
    exact List.mem_map.mpr ⟨bit, List.mem_range.mpr bitLt, rfl⟩
  rw [canonical_bit_rows_prefix] at inGenerated
  exact List.mem_of_mem_take inGenerated

private theorem public_bit_eq
    (hq : EuclidPrime goldilocksP) {z : Nat → Nat}
    (hcanon : ∀ column, z column < goldilocksP)
    (hone : z 0 = 1) (hsat : Satisfies rows z)
    {lane bit : Nat} (laneLt : lane < 4) (bitLt : bit < 64) :
    z (publicBitCol lane bit) =
      pullAssignment z (canonicalMap lane) (CanonicalU64.bitCol bit) := by
  let pulled := pullAssignment z (canonicalMap lane)
  have canonicalSat := canonicalRows_satisfied hsat laneLt
  have bitRowSat : RowHolds pulled (bitRow (CanonicalU64.bitCol bit)) := by
    apply canonicalSat
    exact canonical_bitRow_mem bitLt
  have bitLe : pulled (CanonicalU64.bitCol bit) ≤ 1 :=
    bitRow_le_one hq (pulledCanonical hcanon lane _) (pulledOne hone lane)
      bitRowSat
  have equalRowSat := hsat (equalityRow lane bit)
    (equalityRow_mem laneLt bitLt)
  have publicLt := hcanon (publicBitCol lane bit)
  have canonicalLt := pulledCanonical hcanon lane (CanonicalU64.bitCol bit)
  simp only [equalityRow, RowHolds, lcEval, List.foldl, hone,
    goldilocksP, pullAssignment] at equalRowSat bitLe publicLt canonicalLt ⊢
  omega

private theorem foldl_bits_congr (xs : List Nat) (left right : Nat → Nat)
    (equal : ∀ bit ∈ xs, left bit = right bit) (initial : Nat) :
    xs.foldl (fun total bit => total + 2 ^ bit * left bit) initial =
      xs.foldl (fun total bit => total + 2 ^ bit * right bit) initial := by
  induction xs generalizing initial with
  | nil => rfl
  | cons head tail ih =>
      simp only [List.foldl_cons]
      rw [equal head (by simp)]
      apply ih
      intro bit bitMem
      exact equal bit (by simp [bitMem])

private theorem laneBitsValue_eq_canonical
    (hq : EuclidPrime goldilocksP) {z : Nat → Nat}
    (hcanon : ∀ column, z column < goldilocksP)
    (hone : z 0 = 1) (hsat : Satisfies rows z)
    {lane : Nat} (laneLt : lane < 4) :
    laneBitsValue z lane =
      bitsValue (pullAssignment z (canonicalMap lane)) := by
  unfold laneBitsValue bitsValue
  apply foldl_bits_congr
  intro bit bitMem
  exact public_bit_eq hq hcanon hone hsat laneLt
    (List.mem_range.mp bitMem)

/-- Exact artifact-level encoding contract for one satisfying assignment. -/
structure Holds (z : Nat → Nat) : Prop where
  laneCanonical : ∀ lane, lane < 4 →
    z (digestCol lane) = laneBitsValue z lane ∧
      laneBitsValue z lane < goldilocksP
  publicBoolean : ∀ lane, lane < 4 → ∀ bit, bit < 64 →
    z (publicBitCol lane bit) ≤ 1

/-- `ENC-CANON`: the exact 532 generated rows force canonical, injective F'
public encoding for every satisfying assignment. -/
theorem fPrimeEncoding_sound
    (hq : EuclidPrime goldilocksP) {z : Nat → Nat}
    (hcanon : ∀ column, z column < goldilocksP)
    (hone : z 0 = 1) (hsat : Satisfies rows z) :
    Holds z := by
  refine {
    laneCanonical := ?_
    publicBoolean := ?_
  }
  · intro lane laneLt
    have canonical := canonical_lane hq hcanon hone hsat laneLt
    have bitsEq := laneBitsValue_eq_canonical hq hcanon hone hsat laneLt
    exact ⟨by simpa [bitsEq] using canonical.1,
      by simpa [bitsEq] using canonical.2⟩
  · intro lane laneLt bit bitLt
    rw [public_bit_eq hq hcanon hone hsat laneLt bitLt]
    apply bitRow_le_one hq (pulledCanonical hcanon lane _)
      (pulledOne hone lane)
    apply canonicalRows_satisfied hsat laneLt
    exact canonical_bitRow_mem bitLt

/-- Two satisfying assignments with the same accepted public bits encode the
same four digest lanes. -/
theorem accepted_public_bits_injective
    (hq : EuclidPrime goldilocksP) {left right : Nat → Nat}
    (leftCanonical : ∀ column, left column < goldilocksP)
    (rightCanonical : ∀ column, right column < goldilocksP)
    (leftOne : left 0 = 1) (rightOne : right 0 = 1)
    (leftSat : Satisfies rows left) (rightSat : Satisfies rows right)
    (sameBits : ∀ lane, lane < 4 → ∀ bit, bit < 64 →
      left (publicBitCol lane bit) = right (publicBitCol lane bit)) :
    decodedDigest left = decodedDigest right := by
  funext lane
  have leftSound := fPrimeEncoding_sound hq leftCanonical leftOne leftSat
  have rightSound := fPrimeEncoding_sound hq rightCanonical rightOne rightSat
  change left (digestCol lane) = right (digestCol lane)
  rw [(leftSound.laneCanonical lane lane.isLt).1,
    (rightSound.laneCanonical lane lane.isLt).1]
  unfold laneBitsValue
  apply foldl_bits_congr
  intro bit bitMem
  exact sameBits lane lane.isLt bit (List.mem_range.mp bitMem)

end Nightstream.Implementation.R1CS.FPrimeEncodingSound
