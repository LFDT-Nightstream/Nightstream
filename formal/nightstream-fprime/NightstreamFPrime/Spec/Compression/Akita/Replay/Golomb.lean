import Mathlib.Data.Nat.Size
import Mathlib.Algebra.BigOperators.Group.Finset.Basic

/-! The Golomb–Rice decoder of the response `z` of the terminal of the pinned Akita verifier
(Akita revision `d50f700`: `akita-params/src/golomb_rice.rs`, `golomb_rice_decode_vec`;
`akita-types/src/proof/tail_segments.rs`, `decode_terminal_z_golomb_payload`), and its squared
norm (`akita-params/src/sis/norm_bound.rs`, `checked_centered_l2_sq`).

The decoder is a DEFINITION of the byte encoding of the concrete instance: Akita has no
specification of it, and the evidence is the equality with the decoded `z` of the vectors.

- The bits are read least significant first in each byte. A value is a run of 1 bits, the
  quotient, at most `maxQuotient`; a 0 stop bit; and `rice` remainder bits, least significant
  first. `u = quotient·2^rice + remainder`, and the value is the zigzag decoding
  `(u >>> 1) ^ −(u &&& 1)`, which must be in `[−2^{w−1}, 2^{w−1})` for the zigzag width `w`.
- A value must be at most the cap `z_linf_cap` in absolute value when it has one, and a value of
  `i16`. After `count` values, the bits to the end of the byte are zero and no byte is left.
- `Params.width` is `golomb_rice_zigzag_width` of the bound (`i16::MAX` with no cap) and
  `Params.maxQuotient` is `golomb_rice_max_quotient_for_cap`. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita.Replay

namespace Golomb

/-- The numbers of the payload of `z` (`TailSegmentGroupLayout`): `z_coords`, `z_rice_low_bits`,
`z_linf_cap` and `z_payload_bytes`. -/
structure Params where
  count : ℕ
  rice : ℕ
  cap : Option ℕ
  maxBytes : ℕ
deriving DecidableEq, Repr

/-- `zigzag_encode` of a value: `2·n` for `n ≥ 0` and `2·|n| − 1` below. -/
def zigzag (n : ℤ) : ℕ := if 0 ≤ n then 2 * n.toNat else 2 * (-n).toNat - 1

namespace Params

variable (P : Params)

/-- The bound of the zigzag width: the cap, or `i16::MAX`. -/
def bound : ℕ := P.cap.getD 32767

/-- `golomb_rice_zigzag_width`: one more than the bit length of the bound. -/
def width : ℕ := if P.bound = 0 then 1 else Nat.size P.bound + 1

/-- `golomb_rice_max_quotient_for_cap`: the quotient of the bound. -/
def maxQuotient : ℕ := zigzag P.bound / 2 ^ P.rice

end Params

/-- Bit `i` of the bytes, the least significant bit of each byte first. -/
def bit (bytes : ByteArray) (i : ℕ) : Bool := (bytes.get! (i / 8)).toNat.testBit (i % 8)

/-- The run of 1 bits from `pos`: its length and the position after its stop bit, or `none` when
it is longer than `limit` or the bytes end before the stop bit. -/
def unary (bytes : ByteArray) (limit : ℕ) : ℕ → ℕ → ℕ → Option (ℕ × ℕ)
  | 0, _, _ => none
  | fuel + 1, pos, run =>
      if 8 * bytes.size ≤ pos then none
      else if bit bytes pos then
        if limit < run + 1 then none else unary bytes limit fuel (pos + 1) (run + 1)
      else some (run, pos + 1)

/-- `count` bits from `pos`, least significant first, or `none` when the bytes end. -/
def bits (bytes : ByteArray) (pos count : ℕ) : Option ℕ :=
  if 8 * bytes.size < pos + count then none
  else some ((List.range count).foldr (fun j acc => 2 * acc + (bit bytes (pos + j)).toNat) 0)

/-- `zigzag_decode` with the width check. -/
def unzigzag (u width : ℕ) : Option ℤ :=
  let n : ℤ := if u % 2 = 0 then (u / 2 : ℕ) else -((u / 2 : ℕ) : ℤ) - 1
  if -(2 ^ (width - 1) : ℤ) ≤ n ∧ n < 2 ^ (width - 1) then some n else none

/-- One value from `pos`, and the position after it (`golomb_rice_decode_one_from` and the
conversion of `decode_terminal_z_golomb_payload`). -/
def one (P : Params) (bytes : ByteArray) (pos : ℕ) : Option (ℤ × ℕ) := do
  let (quotient, pos) ← unary bytes P.maxQuotient (P.maxQuotient + 1) pos 0
  let remainder ← bits bytes pos P.rice
  let n ← unzigzag (quotient * 2 ^ P.rice + remainder) P.width
  if (P.cap.map fun cap => decide (cap < n.natAbs)).getD false then none
  else if n < -32768 ∨ 32767 < n then none
  else some (n, pos + P.rice)

/-- `count` values from `pos`. -/
def many (P : Params) (bytes : ByteArray) : ℕ → ℕ → Array ℤ → Option (Array ℤ × ℕ)
  | 0, pos, out => some (out, pos)
  | count + 1, pos, out => do
      let (n, pos) ← one P bytes pos
      many P bytes count pos (out.push n)

/-- `decode_terminal_z_golomb_payload`: the `count` values of the payload, with zero bits to the
end of the last byte and no byte after it. -/
def decode (P : Params) (bytes : ByteArray) : Option (Array ℤ) := do
  if P.maxBytes < bytes.size then none
  let (values, pos) ← many P bytes P.count 0 (Array.emptyWithCapacity P.count)
  let padded := (pos + 7) / 8 * 8
  if padded = 8 * bytes.size ∧ (List.range (padded - pos)).all fun j => !bit bytes (pos + j)
  then some values else none

end Golomb

/-- `checked_centered_l2_sq`: the sum of the squares of the values. The `u128` of the code does
not overflow for values of `i16` and fewer than `2^97` of them. -/
def centeredNormSq (z : Array ℤ) : ℕ := z.foldl (fun sum n => sum + n.natAbs ^ 2) 0

end NightstreamFPrime.Spec.Compression.Akita.Replay
