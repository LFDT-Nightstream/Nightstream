import SuperNeo.Norm

/-!
Base-2 decomposition primitives for SuperNeo.

This file provides the compact theorem-native core for decomposition (P6):
- scalar split into base-2 digits,
- scalar recomposition expression,
- terminal-quotient condition used by protocol constraints,
- row-wise lift to coefficient vectors,
- per-digit/per-entry norm bounds.

Scope note:
- this file stays intentionally small and constructive;
- it does not include high-level protocol wrappers.
-/

namespace SuperNeo

open F

/-- Bit extractor at position `i` from a natural number. -/
def bitAt (n i : Nat) : Nat :=
  (n / (2 ^ i)) % 2

/-- Every extracted bit is in `{0,1}` (strict form). -/
theorem bitAt_lt_two (n i : Nat) : bitAt n i < 2 := by
  unfold bitAt
  exact Nat.mod_lt _ (by decide : 0 < 2)

/-- Every extracted bit is in `{0,1}` (non-strict form). -/
theorem bitAt_le_one (n i : Nat) : bitAt n i ≤ 1 := by
  exact Nat.le_of_lt_succ (bitAt_lt_two n i)

-- Internal helper: small bits are always below the field modulus.
private theorem bitAt_lt_q (n i : Nat) : bitAt n i < Goldilocks.q := by
  have h2 : bitAt n i < 2 := bitAt_lt_two n i
  exact Nat.lt_of_lt_of_le h2 (Nat.succ_le_of_lt Goldilocks.q_gt_one)

-- Internal helper: `F.ofNat` does not change `{0,1}` values modulo `q`.
private theorem ofNat_bitAt_val (n i : Nat) :
    (F.ofNat (bitAt n i)).val = bitAt n i := by
  simp [F.ofNat, Nat.mod_eq_of_lt (bitAt_lt_q n i)]

/-! ### Scalar Decomposition -/

/-- Split one field element into `k` base-2 digits (least significant first). -/
def splitBase2Scalar (a : F) (k : Nat) : Array F :=
  Array.ofFn (fun i : Fin k => F.ofNat (bitAt a.val i.1))

/-- Natural-number recomposition before reduction mod `q`. -/
def recomposeBase2ScalarNat (digits : Array F) : Nat :=
  (List.range digits.size).foldl
    (fun acc i => acc + digits[i]!.val * (2 ^ i))
    0

/-- Recompose one field element from base-2 digits (least significant first). -/
def recomposeBase2Scalar (digits : Array F) : F :=
  F.ofNat (recomposeBase2ScalarNat digits)

/-- Scalar terminal quotient after consuming `k` base-2 digits. -/
def splitBase2TerminalQuot (a : F) (k : Nat) : Nat :=
  a.val / (2 ^ k)

/--
Scalar low-part remainder after consuming `k` base-2 digits.

This is the canonical arithmetic remainder in the Euclidean decomposition
`a.val = (a.val % 2^k) + 2^k * (a.val / 2^k)`.
-/
def splitBase2LowPartNat (a : F) (k : Nat) : Nat :=
  a.val % (2 ^ k)

/-- Predicate form used by protocol obligations: terminal quotient is zero. -/
def splitBase2TerminalZeroProp (a : F) (k : Nat) : Prop :=
  splitBase2TerminalQuot a k = 0

/--
Constructive scalar decomposition identity for base-2 split.

This theorem is definition-derived and does not rely on any check wrappers.
-/
theorem splitBase2DecompositionNat
    (a : F) (k : Nat) :
    splitBase2LowPartNat a k + (2 ^ k) * splitBase2TerminalQuot a k = a.val := by
  unfold splitBase2LowPartNat splitBase2TerminalQuot
  exact Nat.mod_add_div a.val (2 ^ k)

/--
If the terminal quotient is zero, the low part recovers the canonical value.
-/
theorem splitBase2LowPart_eq_val_of_terminal_zero
    (a : F) (k : Nat)
    (hZero : splitBase2TerminalZeroProp a k) :
    splitBase2LowPartNat a k = a.val := by
  have hDec : splitBase2LowPartNat a k + (2 ^ k) * splitBase2TerminalQuot a k = a.val :=
    splitBase2DecompositionNat a k
  have hZero' : a.val / (2 ^ k) = 0 := by
    simpa [splitBase2TerminalZeroProp, splitBase2TerminalQuot] using hZero
  have hTerm : (2 ^ k) * splitBase2TerminalQuot a k = 0 := by
    unfold splitBase2TerminalQuot
    simp [hZero']
  calc
    splitBase2LowPartNat a k = splitBase2LowPartNat a k + (2 ^ k) * splitBase2TerminalQuot a k := by
      simp [hTerm]
    _ = a.val := hDec

/-- Per-digit norm bound predicate for base-2 decomposition. -/
def splitBase2DigitsWithinBoundProp (a : F) (k : Nat) : Prop :=
  ∀ i : Fin k, normInfF (splitBase2Scalar a k)[i.1]! ≤ 1

@[simp] theorem splitBase2Scalar_size (a : F) (k : Nat) :
    (splitBase2Scalar a k).size = k := by
  simp [splitBase2Scalar]

theorem splitBase2Scalar_digit_val_eq
    (a : F) (k : Nat) (i : Fin k) :
    (splitBase2Scalar a k)[i.1]! = F.ofNat (bitAt a.val i.1) := by
  simp [splitBase2Scalar]

theorem splitBase2Scalar_digit_le_one
    (a : F) (k : Nat) (i : Fin k) :
    ((splitBase2Scalar a k)[i.1]!).val ≤ 1 := by
  have hEq : (splitBase2Scalar a k)[i.1]! = F.ofNat (bitAt a.val i.1) :=
    splitBase2Scalar_digit_val_eq a k i
  rw [hEq]
  simpa [ofNat_bitAt_val] using bitAt_le_one a.val i.1

@[simp] theorem recomposeBase2Scalar_eq_ofNat
    (digits : Array F) :
    recomposeBase2Scalar digits = F.ofNat (recomposeBase2ScalarNat digits) := by
  rfl

theorem recomposeBase2Scalar_split_formula
    (a : F) (k : Nat) :
    recomposeBase2Scalar (splitBase2Scalar a k)
      = F.ofNat (recomposeBase2ScalarNat (splitBase2Scalar a k)) := by
  rfl

/--
If the canonical scalar value is below `2^k`, the terminal quotient after
consuming `k` base-2 digits is zero.
-/
theorem splitBase2TerminalZeroProp_of_val_lt_pow
    (a : F) (k : Nat)
    (h : a.val < 2 ^ k) :
    splitBase2TerminalZeroProp a k := by
  unfold splitBase2TerminalZeroProp splitBase2TerminalQuot
  exact Nat.div_eq_of_lt h

theorem splitBase2Scalar_digit_norm_le_one
    (a : F) (k : Nat) (i : Fin k) :
    normInfF (splitBase2Scalar a k)[i.1]! ≤ 1 := by
  have hVal : ((splitBase2Scalar a k)[i.1]!).val ≤ 1 :=
    splitBase2Scalar_digit_le_one a k i
  have hHalf : ((splitBase2Scalar a k)[i.1]!).val ≤ Goldilocks.halfQ :=
    Nat.le_trans hVal Goldilocks.one_le_halfQ
  have hRep :
      F.centeredRep ((splitBase2Scalar a k)[i.1]!)
        = Int.ofNat ((splitBase2Scalar a k)[i.1]!).val :=
    F.centeredRep_eq_of_le_halfQ hHalf
  unfold normInfF F.centeredAbs
  rw [hRep]
  simpa using hVal

/-- All scalar split digits satisfy the expected norm bound `≤ 1`. -/
theorem splitBase2DigitsWithinBound
    (a : F) (k : Nat) :
    splitBase2DigitsWithinBoundProp a k := by
  intro i
  exact splitBase2Scalar_digit_norm_le_one a k i

/-! ### Vector Lift -/

/-- Row-wise base-2 split of a coefficient vector into `k` digit rows. -/
def splitBase2Coeffs (z : Coeffs) (k : Nat) : Array Coeffs :=
  Array.ofFn (fun i : Fin k => z.map (fun a => F.ofNat (bitAt a.val i.1)))

/-- Recompose a coefficient vector from base-2 rows. -/
def recomposeBase2Coeffs (rows : Array Coeffs) : Coeffs :=
  if _h : rows.size = 0 then
    #[]
  else
    let n := rows[0]!.size
    Array.ofFn (fun j : Fin n =>
      recomposeBase2Scalar (rows.map (fun row => row[j]!)))

/-- Per-entry norm bound predicate for row-wise base-2 decomposition. -/
def splitBase2RowsWithinBoundProp (z : Coeffs) (k : Nat) : Prop :=
  ∀ i : Fin k, ∀ j : Fin z.size, normInfF ((splitBase2Coeffs z k)[i.1]![j.1]!) ≤ 1

/-- Vector lift of the scalar decomposition identity. -/
theorem splitBase2CoeffsDecompositionNat
    (z : Coeffs) (k : Nat) :
    ∀ j : Fin z.size,
      splitBase2LowPartNat z[j.1] k + (2 ^ k) * splitBase2TerminalQuot z[j.1] k = z[j.1].val := by
  intro j
  exact splitBase2DecompositionNat z[j.1] k

@[simp] theorem splitBase2Coeffs_size (z : Coeffs) (k : Nat) :
    (splitBase2Coeffs z k).size = k := by
  simp [splitBase2Coeffs]

theorem splitBase2Coeffs_row_size
    (z : Coeffs) (k : Nat) (i : Fin k) :
    ((splitBase2Coeffs z k)[i.1]!).size = z.size := by
  simp [splitBase2Coeffs]

theorem splitBase2Coeffs_digit_le_one
    (z : Coeffs) (k : Nat)
    (i : Fin k) (j : Fin z.size) :
    (((splitBase2Coeffs z k)[i.1]![j.1]!).val) ≤ 1 := by
  have hEq :
      (splitBase2Coeffs z k)[i.1]!
        = z.map (fun a => F.ofNat (bitAt a.val i.1)) := by
    simp [splitBase2Coeffs]
  rw [hEq]
  have hVal : (F.ofNat (bitAt z[j.1].val i.1)).val ≤ 1 := by
    simpa [ofNat_bitAt_val] using bitAt_le_one z[j.1].val i.1
  simpa [Array.getElem_map, j.2] using hVal

theorem splitBase2Coeffs_digit_norm_le_one
    (z : Coeffs) (k : Nat)
    (i : Fin k) (j : Fin z.size) :
    normInfF ((splitBase2Coeffs z k)[i.1]![j.1]!) ≤ 1 := by
  have hVal : (((splitBase2Coeffs z k)[i.1]![j.1]!).val) ≤ 1 :=
    splitBase2Coeffs_digit_le_one z k i j
  have hHalf : (((splitBase2Coeffs z k)[i.1]![j.1]!).val) ≤ Goldilocks.halfQ :=
    Nat.le_trans hVal Goldilocks.one_le_halfQ
  have hRep :
      F.centeredRep ((splitBase2Coeffs z k)[i.1]![j.1]!)
        = Int.ofNat (((splitBase2Coeffs z k)[i.1]![j.1]!).val) :=
    F.centeredRep_eq_of_le_halfQ hHalf
  unfold normInfF F.centeredAbs
  rw [hRep]
  simpa using hVal

/-- All entries in all decomposition rows satisfy the expected norm bound `≤ 1`. -/
theorem splitBase2RowsWithinBound
    (z : Coeffs) (k : Nat) :
    splitBase2RowsWithinBoundProp z k := by
  intro i j
  exact splitBase2Coeffs_digit_norm_le_one z k i j

end SuperNeo
