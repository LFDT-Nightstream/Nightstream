import Mathlib.Data.List.GetD
import Nightstream.Implementation.R1CS.Correspondence.ShiftedTernary.CanonicalWord
import Nightstream.Protocol.NebulaV2.CompactCommit

/-!
Contract: exact bridge from the production shifted-ternary rows to the
`ShiftedTernary41V1` word used by the Nebula V2 compact commitment.

Assurance tier: implementation model.

Owns the pointwise equality between `Nat.digitsAppend` and the production
quotient digit, the equality of both centered-field digit maps, and the
row-derived equality of every production digit column to the protocol word.

Does not own seeded Ajtai rows, compact-token outputs, Poseidon2 chains,
transcript derivation, Rust conformance, or cryptographic binding.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.NebulaV2.ShiftedTernaryEncodingBridge

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.ShiftedTernaryCompiler
open Nightstream.Implementation.R1CS.ShiftedTernarySound
open Nightstream.Implementation.R1CS.ShiftedTernaryCanonicalWord
open Nightstream.Protocol.NebulaV2
open Nightstream.Protocol.NebulaV2.CompactCommit
open Nightstream.Protocol.NebulaV2.ShiftedTernary41V1

/-- A fixed-width protocol trit is the same quotient digit used by the
production witness generator. The proof covers both the native digit-list
prefix and its canonical zero padding. -/
theorem trits_getD_eq_quotient
    (value : CanonicalGoldilocks) (index : Nat)
    (indexLt : index < ShiftedTernary41V1.digitCount) :
    (trits value).getD index 0 =
      target value / 3 ^ index % 3 := by
  have digitsLengthLe :
      (Nat.digits 3 (target value)).length ≤
        ShiftedTernary41V1.digitCount := by
    exact (Nat.digits_length_le_iff
      (b := 3) (k := ShiftedTernary41V1.digitCount)
      (by decide) (target value)).2 (target_lt_wordCapacity value)
  by_cases inNativeDigits : index < (Nat.digits 3 (target value)).length
  · unfold trits Nat.digitsAppend
    rw [List.getD_append _ _ _ _ inNativeDigits]
    exact Nat.getD_digits (target value) index (by decide)
  · have inPadding :
        index - (Nat.digits 3 (target value)).length <
          ShiftedTernary41V1.digitCount -
            (Nat.digits 3 (target value)).length := by
      omega
    unfold trits Nat.digitsAppend
    rw [List.getD_append_right _ _ _ _ (Nat.le_of_not_gt inNativeDigits)]
    rw [List.getD_replicate (x := 0) (y := 0) inPadding]
    rw [← Nat.getD_digits (target value) index (by decide)]
    simp [List.getD_eq_getElem?_getD, inNativeDigits]

/-- Indexed protocol form of `trits_getD_eq_quotient`. -/
theorem tritAt_eq_quotient
    (value : CanonicalGoldilocks)
    (index : Fin ShiftedTernary41V1.digitCount) :
    tritAt value index = target value / 3 ^ index.val % 3 := by
  have equal := trits_getD_eq_quotient value index.val index.isLt
  have inWord : index.val < (trits value).length := by
    rw [trits_length]
    exact index.isLt
  rw [List.getD_eq_getElem
    (l := trits value) (d := 0) inWord] at equal
  unfold tritAt
  exact equal

/-- The implementation and protocol centered-field digits are identical for
every canonical field value and every V2 digit position. -/
theorem canonicalDigit_eq_fieldDigit_tritAt
    (value : CanonicalGoldilocks)
    (index : Fin ShiftedTernary41V1.digitCount) :
    canonicalDigit value.val index.val = fieldDigit (tritAt value index) := by
  unfold canonicalDigit
  rw [show
    (value.val + ShiftedTernaryCompiler.shift) % goldilocksP =
      target value by rfl]
  rw [tritAt_eq_quotient]
  generalize digitEq : target value / 3 ^ index.val % 3 = digit
  have digitLt : digit < 3 := by
    rw [← digitEq]
    exact Nat.mod_lt _ (by decide)
  have alternatives : digit = 0 ∨ digit = 1 ∨ digit = 2 := by omega
  rcases alternatives with equal | equal | equal <;>
    rw [equal] <;>
    rfl

/-- Exact row-to-protocol endpoint for one production digit column. No
prover-carried token or opening equality is a premise. -/
theorem productionDigit_eq_protocolDigit
    {assignment : Nat → Nat} {fieldColumn digitStart : Nat}
    (value : CanonicalGoldilocks)
    (sourceExact : assignment fieldColumn = value.val)
    (opening : CanonicalOpening
      (localAssignment assignment fieldColumn digitStart))
    (index : Fin ShiftedTernary41V1.digitCount) :
    assignment (digitStart + index.val) =
      fieldDigit (tritAt value index) := by
  rw [productionDigit_eq_canonicalDigit opening index.val index.isLt]
  rw [sourceExact]
  exact canonicalDigit_eq_fieldDigit_tritAt value index

end Nightstream.Implementation.NebulaV2.ShiftedTernaryEncodingBridge
