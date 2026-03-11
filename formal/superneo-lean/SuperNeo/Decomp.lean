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

private theorem foldl_range_add_eq_of_pointwise
    (k : Nat) (f g : Nat → Nat)
    (hfg : ∀ i, i < k → f i = g i) :
    (List.range k).foldl (fun acc i => acc + f i) 0 =
      (List.range k).foldl (fun acc i => acc + g i) 0 := by
  induction k with
  | zero =>
      simp
  | succ k ih =>
      simp [List.range_succ, List.foldl_append]
      have hPrefix : ∀ i, i < k → f i = g i := by
        intro i hi
        exact hfg i (Nat.lt_trans hi (Nat.lt_succ_self k))
      have hRec := ih hPrefix
      simpa [hfg k (Nat.lt_succ_self k)] using congrArg (fun t => t + f k) hRec

private def base2PrefixSum (n k : Nat) : Nat :=
  (List.range k).foldl (fun acc i => acc + bitAt n i * (2 ^ i)) 0

private theorem base2PrefixSum_succ (n k : Nat) :
    base2PrefixSum n (k + 1) = base2PrefixSum n k + bitAt n k * (2 ^ k) := by
  unfold base2PrefixSum
  simp [List.range_succ, List.foldl_append]

private theorem base2PrefixSum_eq_lowPart (n k : Nat) :
    base2PrefixSum n k = n % (2 ^ k) := by
  induction k with
  | zero =>
      simp [base2PrefixSum, Nat.mod_one]
  | succ k ih =>
      rw [base2PrefixSum_succ, ih, bitAt]
      have hPow :
          n % (2 ^ (k + 1)) = n % (2 ^ k) + (2 ^ k) * (n / (2 ^ k) % 2) := by
        simpa [Nat.pow_succ] using (Nat.mod_pow_succ (x := n) (b := 2) (k := k))
      have hPow' :
          n % (2 ^ (k + 1)) = n % (2 ^ k) + (n / (2 ^ k) % 2) * (2 ^ k) := by
        simpa [Nat.mul_comm] using hPow
      exact hPow'.symm

/--
Constructive scalar recomposition for base-2 split:
recomposing the first `k` extracted bits equals the canonical low-part modulo `2^k`.
-/
theorem recomposeBase2ScalarNat_splitBase2Scalar_eq_lowPartNat
    (a : F) (k : Nat) :
    recomposeBase2ScalarNat (splitBase2Scalar a k) = splitBase2LowPartNat a k := by
  have hPrefix :
      recomposeBase2ScalarNat (splitBase2Scalar a k) = base2PrefixSum a.val k := by
    unfold recomposeBase2ScalarNat base2PrefixSum
    simpa [splitBase2Scalar_size] using
      (foldl_range_add_eq_of_pointwise k
        (fun i => ((splitBase2Scalar a k)[i]!).val * (2 ^ i))
        (fun i => bitAt a.val i * (2 ^ i))
        (by
          intro i hi
          have hDigit : ((splitBase2Scalar a k)[i]!).val = bitAt a.val i := by
            simp [splitBase2Scalar, hi, ofNat_bitAt_val]
          simp [hDigit]))
  have hLowPart : base2PrefixSum a.val k = a.val % (2 ^ k) :=
    base2PrefixSum_eq_lowPart a.val k
  unfold splitBase2LowPartNat
  exact hPrefix.trans hLowPart

/--
If `a.val < 2^k`, recomposing the base-2 split recovers the canonical representative.
-/
theorem recomposeBase2ScalarNat_splitBase2Scalar_eq_val_of_lt_pow
    (a : F) (k : Nat)
    (h : a.val < 2 ^ k) :
    recomposeBase2ScalarNat (splitBase2Scalar a k) = a.val := by
  rw [recomposeBase2ScalarNat_splitBase2Scalar_eq_lowPartNat]
  unfold splitBase2LowPartNat
  exact Nat.mod_eq_of_lt h

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

/--
Characterization of terminal-quotient zero for base-2 split:
`a.val / 2^k = 0` iff `a.val < 2^k`.
-/
theorem splitBase2TerminalZeroProp_iff_val_lt_pow
    (a : F) (k : Nat) :
    splitBase2TerminalZeroProp a k ↔ a.val < 2 ^ k := by
  constructor
  · intro hZero
    unfold splitBase2TerminalZeroProp splitBase2TerminalQuot at hZero
    rcases (Nat.div_eq_zero_iff).1 hZero with hDenZero | hLt
    · exact False.elim ((Nat.ne_of_gt (Nat.two_pow_pos k)) hDenZero)
    · exact hLt
  · intro hLt
    exact splitBase2TerminalZeroProp_of_val_lt_pow a k hLt

/--
If `a.val < 2^k`, the low-part remainder equals the canonical representative.
-/
theorem splitBase2LowPart_eq_val_of_val_lt_pow
    (a : F) (k : Nat)
    (h : a.val < 2 ^ k) :
    splitBase2LowPartNat a k = a.val := by
  exact splitBase2LowPart_eq_val_of_terminal_zero a k
    (splitBase2TerminalZeroProp_of_val_lt_pow a k h)

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

/--
Column-wise base-2 recomposition is constructive: each recomposed entry equals
the scalar low-part modulo `2^k` of the source coefficient.
-/
theorem recomposeBase2Coeffs_entry_eq_lowPartNat
    (z : Coeffs) (k : Nat) (j : Fin z.size) :
    recomposeBase2ScalarNat ((splitBase2Coeffs z k).map (fun row => row[j.1]!))
      = splitBase2LowPartNat z[j.1] k := by
  have hColEq :
      ((splitBase2Coeffs z k).map (fun row => row[j.1]!))
        = splitBase2Scalar z[j.1] k := by
    apply Array.ext
    · simp [splitBase2Coeffs, splitBase2Scalar]
    · intro i hi₁ hi₂
      simp [splitBase2Coeffs, splitBase2Scalar, j.2]
  calc
    recomposeBase2ScalarNat ((splitBase2Coeffs z k).map (fun row => row[j.1]!))
        = recomposeBase2ScalarNat (splitBase2Scalar z[j.1] k) := by
          simp [hColEq]
    _ = splitBase2LowPartNat z[j.1] k :=
          recomposeBase2ScalarNat_splitBase2Scalar_eq_lowPartNat z[j.1] k

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

/-! ### P6 Compatibility Surface (Balanced Split API) -/

/-!
This section provides the compact compatibility API still used by arithmetic-bundle/protocol-math-target style
layers (`splitBalancedVec`, `recomposeSplitDigits`, `digitsWithinBase`,
`splitRoundTrip`).

The implementation is intentionally theorem-facing:
- executable and deterministic,
- lightweight sound/complete lemmas for proposition-level lifting.
-/

/-- Convert a small signed integer digit to `F` via canonical representatives. -/
private def fieldOfSignedDigit (x : Int) : F :=
  F.ofNat (Int.toNat (x % Int.ofNat Goldilocks.q))

private def qInt : Int := Int.ofNat Goldilocks.q

private theorem qInt_pos : 0 < qInt := by
  simpa [qInt] using (Int.natCast_pos.mpr Goldilocks.q_pos)

private theorem qInt_ne_zero : qInt ≠ 0 :=
  Int.ofNat_ne_zero.mpr Goldilocks.q_ne_zero

private theorem residue_nat_lt_q (x : Int) : Int.toNat (x % qInt) < Goldilocks.q := by
  have hNonneg : 0 ≤ x % qInt := Int.emod_nonneg x qInt_ne_zero
  have hLt : x % qInt < qInt := Int.emod_lt_of_pos x qInt_pos
  exact (Int.toNat_lt hNonneg).2 (by simpa [qInt] using hLt)

@[simp] private theorem fieldOfSignedDigit_val (x : Int) :
    (fieldOfSignedDigit x).val = Int.toNat (x % qInt) := by
  unfold fieldOfSignedDigit F.ofNat
  exact Nat.mod_eq_of_lt (residue_nat_lt_q x)

@[simp] private theorem fieldOfSignedDigit_ofNat (n : Nat) :
    fieldOfSignedDigit (Int.ofNat n) = F.ofNat n := by
  apply Fin.ext
  rw [fieldOfSignedDigit_val]
  unfold qInt
  have hnonneg : 0 ≤ Int.ofNat n % Int.ofNat Goldilocks.q := by
    exact Int.emod_nonneg (Int.ofNat n) (Int.ofNat_ne_zero.mpr Goldilocks.q_ne_zero)
  have hcast1 :
      Int.ofNat ((Int.ofNat n % Int.ofNat Goldilocks.q).toNat) =
        Int.ofNat n % Int.ofNat Goldilocks.q := by
    exact Int.toNat_of_nonneg hnonneg
  have hcast2 :
      Int.ofNat n % Int.ofNat Goldilocks.q = Int.ofNat (n % Goldilocks.q) := by
    simpa using (Int.ofNat_mod_ofNat n Goldilocks.q)
  have hnat : (Int.ofNat n % Int.ofNat Goldilocks.q).toNat = n % Goldilocks.q := by
    exact (Int.ofNat_inj).1 (hcast1.trans hcast2)
  simpa using hnat

@[simp] private theorem fieldOfSignedDigit_zero :
    fieldOfSignedDigit 0 = 0 := by
  simpa using (fieldOfSignedDigit_ofNat 0)

@[simp] private theorem fieldOfSignedDigit_one :
    fieldOfSignedDigit 1 = (1 : F) := by
  simpa [F.one] using (fieldOfSignedDigit_ofNat 1)

private theorem fieldOfSignedDigit_add (x y : Int) :
    fieldOfSignedDigit (x + y) = fieldOfSignedDigit x + fieldOfSignedDigit y := by
  apply Fin.ext
  rw [fieldOfSignedDigit_val]
  change Int.toNat ((x + y) % qInt) =
    ((fieldOfSignedDigit x).val + (fieldOfSignedDigit y).val) % Goldilocks.q
  rw [fieldOfSignedDigit_val, fieldOfSignedDigit_val]
  apply Int.ofNat_inj.mp
  have hxyNonneg : 0 ≤ (x + y) % qInt := Int.emod_nonneg (x + y) qInt_ne_zero
  have hxNonneg : 0 ≤ x % qInt := Int.emod_nonneg x qInt_ne_zero
  have hyNonneg : 0 ≤ y % qInt := Int.emod_nonneg y qInt_ne_zero
  calc
    Int.ofNat (Int.toNat ((x + y) % qInt))
        = (x + y) % qInt := Int.toNat_of_nonneg hxyNonneg
    _ = (x % qInt + y % qInt) % qInt := by
          simpa using (Int.add_emod x y qInt)
    _ = (Int.ofNat (Int.toNat (x % qInt)) + Int.ofNat (Int.toNat (y % qInt))) % qInt := by
          simp [Int.toNat_of_nonneg hxNonneg, Int.toNat_of_nonneg hyNonneg]
    _ = Int.ofNat (Int.toNat (x % qInt) + Int.toNat (y % qInt)) % qInt := by
          simp
    _ = Int.ofNat ((Int.toNat (x % qInt) + Int.toNat (y % qInt)) % Goldilocks.q) := by
          simpa [qInt] using
            (Int.ofNat_mod_ofNat (Int.toNat (x % qInt) + Int.toNat (y % qInt)) Goldilocks.q)

private theorem fieldOfSignedDigit_mul (x y : Int) :
    fieldOfSignedDigit (x * y) = fieldOfSignedDigit x * fieldOfSignedDigit y := by
  apply Fin.ext
  rw [fieldOfSignedDigit_val]
  change Int.toNat ((x * y) % qInt) =
    ((fieldOfSignedDigit x).val * (fieldOfSignedDigit y).val) % Goldilocks.q
  rw [fieldOfSignedDigit_val, fieldOfSignedDigit_val]
  apply Int.ofNat_inj.mp
  have hxyNonneg : 0 ≤ (x * y) % qInt := Int.emod_nonneg (x * y) qInt_ne_zero
  have hxNonneg : 0 ≤ x % qInt := Int.emod_nonneg x qInt_ne_zero
  have hyNonneg : 0 ≤ y % qInt := Int.emod_nonneg y qInt_ne_zero
  calc
    Int.ofNat (Int.toNat ((x * y) % qInt))
        = (x * y) % qInt := Int.toNat_of_nonneg hxyNonneg
    _ = (x % qInt * (y % qInt)) % qInt := by
          simpa using (Int.mul_emod x y qInt)
    _ = (Int.ofNat (Int.toNat (x % qInt)) * Int.ofNat (Int.toNat (y % qInt))) % qInt := by
          simp [Int.toNat_of_nonneg hxNonneg, Int.toNat_of_nonneg hyNonneg]
    _ = Int.ofNat (Int.toNat (x % qInt) * Int.toNat (y % qInt)) % qInt := by
          simp
    _ = Int.ofNat ((Int.toNat (x % qInt) * Int.toNat (y % qInt)) % Goldilocks.q) := by
          simpa [qInt] using
            (Int.ofNat_mod_ofNat (Int.toNat (x % qInt) * Int.toNat (y % qInt)) Goldilocks.q)

private theorem fieldOfSignedDigit_centeredRep (a : F) :
    fieldOfSignedDigit (F.centeredRep a) = a := by
  by_cases hHalf : a.val ≤ Goldilocks.halfQ
  · rw [F.centeredRep_eq_of_le_halfQ hHalf]
    rw [fieldOfSignedDigit_ofNat]
    simpa using (F.ofNat_val a)
  · have hGt : Goldilocks.halfQ < a.val := Nat.lt_of_not_ge hHalf
    rw [F.centeredRep_eq_sub_q_of_halfQ_lt hGt]
    apply Fin.ext
    calc
      (fieldOfSignedDigit (Int.ofNat a.val - Int.ofNat Goldilocks.q)).val
          = Int.toNat ((Int.ofNat a.val - Int.ofNat Goldilocks.q) % Int.ofNat Goldilocks.q) := by
              simpa [qInt] using
                (fieldOfSignedDigit_val (Int.ofNat a.val - Int.ofNat Goldilocks.q))
      _ = Int.toNat (Int.ofNat a.val % Int.ofNat Goldilocks.q) := by
            simp [Int.sub_emod_right]
      _ = Int.toNat (Int.ofNat a.val) := by
            have hlt : Int.ofNat a.val < Int.ofNat Goldilocks.q := by
              exact Int.ofNat_lt.mpr a.isLt
            simpa using congrArg Int.toNat
              (Int.emod_eq_of_lt (Int.natCast_nonneg a.val) hlt)
      _ = a.val := by simp

/-- One balanced digit extraction step from an integer state in base `b`. -/
private def balancedDigitOfState (state : Int) (b : Nat) : Int :=
  let bInt : Int := Int.ofNat b
  let r : Nat := Int.toNat (state.emod bInt)
  if r < b / 2 then
    Int.ofNat r
  else if r = b / 2 then
    if state < 0 then Int.ofNat r - bInt else Int.ofNat r
  else
    Int.ofNat r - bInt

/-- Balanced step-state transition. -/
private def balancedNextState (state : Int) (b : Nat) : Int :=
  (state - balancedDigitOfState state b) / Int.ofNat b

/--
One balanced step satisfies a constructive decomposition whenever the chosen
digit is congruent to the state modulo `b`.
-/
private theorem balancedStep_decomposition_of_congruence
    (state : Int) (b : Nat)
    (hModEq : state % Int.ofNat b = balancedDigitOfState state b % Int.ofNat b) :
    state =
      balancedDigitOfState state b +
        Int.ofNat b * balancedNextState state b := by
  have hSubModZero : (state - balancedDigitOfState state b) % Int.ofNat b = 0 :=
    (Int.emod_eq_emod_iff_emod_sub_eq_zero).1 hModEq
  have hDiv : Int.ofNat b ∣ (state - balancedDigitOfState state b) :=
    (Int.dvd_iff_emod_eq_zero).2 hSubModZero
  have hMul :
      (state - balancedDigitOfState state b) =
        Int.ofNat b * ((state - balancedDigitOfState state b) / Int.ofNat b) := by
    calc
      state - balancedDigitOfState state b
          = ((state - balancedDigitOfState state b) / Int.ofNat b) * Int.ofNat b := by
              simpa using (Int.ediv_mul_cancel hDiv).symm
      _ = Int.ofNat b * ((state - balancedDigitOfState state b) / Int.ofNat b) := by
              simp [Int.mul_comm]
  calc
    state = (state - balancedDigitOfState state b) + balancedDigitOfState state b := by
      simpa [Int.add_comm] using (Int.sub_add_cancel state (balancedDigitOfState state b)).symm
    _ = balancedDigitOfState state b + (state - balancedDigitOfState state b) := by
      simp [Int.add_comm]
    _ = balancedDigitOfState state b +
          Int.ofNat b * ((state - balancedDigitOfState state b) / Int.ofNat b) := by
      exact congrArg (fun t => balancedDigitOfState state b + t) hMul
    _ = balancedDigitOfState state b + Int.ofNat b * balancedNextState state b := by
      rfl

/--
For `b ≥ 2`, the balanced digit chosen at one step is congruent to the current
state modulo `b`.
-/
private theorem balancedDigit_mod_eq_state_mod
    (state : Int) (b : Nat) (hb : 2 ≤ b) :
    balancedDigitOfState state b % Int.ofNat b = state % Int.ofNat b := by
  let bInt : Int := Int.ofNat b
  have hbPosNat : 0 < b := Nat.lt_of_lt_of_le (by decide : 0 < 2) hb
  have hbPosInt : 0 < bInt := by simpa [bInt] using (Int.natCast_pos.mpr hbPosNat)
  have hbNe : bInt ≠ 0 := by
    exact Int.ofNat_ne_zero.mpr (Nat.ne_of_gt hbPosNat)
  have hResNonneg : 0 ≤ state % bInt := Int.emod_nonneg state hbNe
  have hResLt : state % bInt < bInt := by simpa [bInt] using (Int.emod_lt_of_pos state hbPosInt)
  have hResidueNatCast : Int.ofNat (Int.toNat (state % bInt)) = state % bInt := by
    simpa using (Int.toNat_of_nonneg hResNonneg)
  have hResidueMod : (Int.ofNat (Int.toNat (state % bInt))) % bInt = state % bInt := by
    rw [hResidueNatCast]
    exact Int.emod_eq_of_lt hResNonneg hResLt
  have hResidueMod' :
      (Int.ofNat (Int.toNat (state.emod (Int.ofNat b)))) % Int.ofNat b = state % Int.ofNat b := by
    simpa [bInt] using hResidueMod
  by_cases hlt : Int.toNat (state % bInt) < b / 2
  ·
    have hlt' : Int.toNat (state.emod (Int.ofNat b)) < b / 2 := by
      simpa [bInt] using hlt
    have hDigitEq :
        balancedDigitOfState state b = Int.ofNat (Int.toNat (state.emod (Int.ofNat b))) := by
      unfold balancedDigitOfState
      rw [if_pos hlt']
    calc
      balancedDigitOfState state b % Int.ofNat b
          = (Int.ofNat (Int.toNat (state.emod (Int.ofNat b)))) % Int.ofNat b := by
              rw [hDigitEq]
      _ = state % Int.ofNat b := hResidueMod'
  · by_cases heq : Int.toNat (state % bInt) = b / 2
    · by_cases hneg : state < 0
      ·
        have hlt' : ¬ Int.toNat (state.emod (Int.ofNat b)) < b / 2 := by
          simpa [bInt] using hlt
        have heq' : Int.toNat (state.emod (Int.ofNat b)) = b / 2 := by
          simpa [bInt] using heq
        have hDigitEq :
            balancedDigitOfState state b
              = Int.ofNat (Int.toNat (state.emod (Int.ofNat b))) - Int.ofNat b := by
          unfold balancedDigitOfState
          rw [if_neg hlt', if_pos heq', if_pos hneg]
        calc
          balancedDigitOfState state b % Int.ofNat b
              = (Int.ofNat (Int.toNat (state.emod (Int.ofNat b))) - Int.ofNat b) % Int.ofNat b := by
                  rw [hDigitEq]
          _ = (Int.ofNat (Int.toNat (state.emod (Int.ofNat b)))) % Int.ofNat b :=
                Int.sub_emod_right (Int.ofNat (Int.toNat (state.emod (Int.ofNat b)))) (Int.ofNat b)
          _ = state % Int.ofNat b := hResidueMod'
      ·
        have hlt' : ¬ Int.toNat (state.emod (Int.ofNat b)) < b / 2 := by
          simpa [bInt] using hlt
        have heq' : Int.toNat (state.emod (Int.ofNat b)) = b / 2 := by
          simpa [bInt] using heq
        have hDigitEq :
            balancedDigitOfState state b = Int.ofNat (Int.toNat (state.emod (Int.ofNat b))) := by
          unfold balancedDigitOfState
          rw [if_neg hlt', if_pos heq', if_neg hneg]
        calc
          balancedDigitOfState state b % Int.ofNat b
              = (Int.ofNat (Int.toNat (state.emod (Int.ofNat b)))) % Int.ofNat b := by
                  rw [hDigitEq]
          _ = state % Int.ofNat b := hResidueMod'
    ·
      have hlt' : ¬ Int.toNat (state.emod (Int.ofNat b)) < b / 2 := by
        simpa [bInt] using hlt
      have heq' : ¬ Int.toNat (state.emod (Int.ofNat b)) = b / 2 := by
        simpa [bInt] using heq
      have hDigitEq :
          balancedDigitOfState state b
            = Int.ofNat (Int.toNat (state.emod (Int.ofNat b))) - Int.ofNat b := by
        unfold balancedDigitOfState
        rw [if_neg hlt', if_neg heq']
      calc
        balancedDigitOfState state b % Int.ofNat b
            = (Int.ofNat (Int.toNat (state.emod (Int.ofNat b))) - Int.ofNat b) % Int.ofNat b := by
                rw [hDigitEq]
        _ = (Int.ofNat (Int.toNat (state.emod (Int.ofNat b)))) % Int.ofNat b :=
              Int.sub_emod_right (Int.ofNat (Int.toNat (state.emod (Int.ofNat b)))) (Int.ofNat b)
        _ = state % Int.ofNat b := hResidueMod'

/-- Symmetric form of `balancedDigit_mod_eq_state_mod`. -/
private theorem balancedState_mod_eq_digit_mod
    (state : Int) (b : Nat) (hb : 2 ≤ b) :
    state % Int.ofNat b = balancedDigitOfState state b % Int.ofNat b := by
  exact (balancedDigit_mod_eq_state_mod state b hb).symm

/--
Per-step constructive decomposition for the concrete balanced digit rule
(`b ≥ 2` case).
-/
private theorem balancedStep_decomposition
    (state : Int) (b : Nat) (hb : 2 ≤ b) :
    state =
      balancedDigitOfState state b +
        Int.ofNat b * balancedNextState state b := by
  exact balancedStep_decomposition_of_congruence (state := state) (b := b)
    (hModEq := balancedState_mod_eq_digit_mod state b hb)

/-- Balanced state after `k` extraction steps. -/
private def balancedStateAfter (state : Int) (b : Nat) : Nat → Int
  | 0 => state
  | k + 1 => balancedNextState (balancedStateAfter state b k) b

/-- Weighted prefix sum of balanced digits across `k` steps. -/
private def balancedDigitPrefixSum (state : Int) (b : Nat) : Nat → Int
  | 0 => 0
  | k + 1 =>
      balancedDigitPrefixSum state b k +
        balancedDigitOfState (balancedStateAfter state b k) b * (Int.ofNat b ^ k)

/--
Constructive telescoping decomposition for balanced extraction over `k` steps:
the initial state equals weighted extracted digits plus terminal residue.
-/
private theorem balancedState_telescope_of_step
    (state : Int) (b : Nat)
    (hStep : ∀ s : Int,
      s = balancedDigitOfState s b + Int.ofNat b * balancedNextState s b) :
    ∀ k,
      state =
        balancedDigitPrefixSum state b k +
          (Int.ofNat b ^ k) * balancedStateAfter state b k
  | 0 => by
      simp [balancedDigitPrefixSum, balancedStateAfter]
  | k + 1 => by
      have ih := balancedState_telescope_of_step state b hStep k
      have hStepK :
          balancedStateAfter state b k =
            balancedDigitOfState (balancedStateAfter state b k) b +
              Int.ofNat b * balancedStateAfter state b (k + 1) := by
        simpa [balancedStateAfter, balancedNextState] using
          hStep (balancedStateAfter state b k)
      have hScaled :
          (Int.ofNat b ^ k) * balancedStateAfter state b k =
            (Int.ofNat b ^ k) *
              (balancedDigitOfState (balancedStateAfter state b k) b +
                Int.ofNat b * balancedStateAfter state b (k + 1)) :=
        congrArg (fun t => (Int.ofNat b ^ k) * t) hStepK
      calc
        state
            = balancedDigitPrefixSum state b k +
                (Int.ofNat b ^ k) * balancedStateAfter state b k := ih
        _ = balancedDigitPrefixSum state b k +
              (Int.ofNat b ^ k) *
                (balancedDigitOfState (balancedStateAfter state b k) b +
                  Int.ofNat b * balancedStateAfter state b (k + 1)) := by
              exact congrArg (fun t => balancedDigitPrefixSum state b k + t) hScaled
        _ = (balancedDigitPrefixSum state b k +
              balancedDigitOfState (balancedStateAfter state b k) b * (Int.ofNat b ^ k)) +
              ((Int.ofNat b ^ k) * Int.ofNat b) * balancedStateAfter state b (k + 1) := by
              simp [Int.mul_add, Int.mul_assoc, Int.mul_comm, Int.mul_left_comm, Int.add_assoc]
        _ = balancedDigitPrefixSum state b (k + 1) +
              (Int.ofNat b ^ (k + 1)) * balancedStateAfter state b (k + 1) := by
              simp [balancedDigitPrefixSum, Int.pow_succ, Int.mul_assoc, Int.mul_comm]

/-- Public wrapper: balanced integer state after `k` extraction steps. -/
def splitBalancedStateAfterInt (state : Int) (b k : Nat) : Int :=
  balancedStateAfter state b k

/-- Public wrapper: weighted balanced-digit prefix sum up to `k` steps. -/
def splitBalancedDigitPrefixSumInt (state : Int) (b k : Nat) : Int :=
  balancedDigitPrefixSum state b k

/--
Canonical fold view of the balanced digit-prefix sum.
-/
private theorem splitBalancedDigitPrefixSumInt_eq_fold
    (state : Int) (b k : Nat) :
    splitBalancedDigitPrefixSumInt state b k =
      (List.range k).foldl
        (fun acc i =>
          acc + balancedDigitOfState (splitBalancedStateAfterInt state b i) b * (Int.ofNat b ^ i))
        0 := by
  induction k with
  | zero =>
      rfl
  | succ k ih =>
      calc
        splitBalancedDigitPrefixSumInt state b (k + 1)
            = splitBalancedDigitPrefixSumInt state b k +
                balancedDigitOfState (splitBalancedStateAfterInt state b k) b * (Int.ofNat b ^ k) := by
                  rfl
        _ = (List.range k).foldl
              (fun acc i =>
                acc + balancedDigitOfState (splitBalancedStateAfterInt state b i) b * (Int.ofNat b ^ i))
              0 +
              balancedDigitOfState (splitBalancedStateAfterInt state b k) b * (Int.ofNat b ^ k) := by
                simp [ih]
        _ = (List.range (k + 1)).foldl
              (fun acc i =>
                acc + balancedDigitOfState (splitBalancedStateAfterInt state b i) b * (Int.ofNat b ^ i))
              0 := by
                simp [List.range_succ, List.foldl_append]

/-- Terminal integer state for balanced split of one scalar. -/
def splitBalancedTerminalState (a : F) (b k : Nat) : Int :=
  if b < 2 then
    F.centeredRep a
  else
    splitBalancedStateAfterInt (F.centeredRep a) b k

/-- Predicate form: balanced terminal state is zero after `k` steps. -/
def splitBalancedTerminalZeroProp (a : F) (b k : Nat) : Prop :=
  splitBalancedTerminalState a b k = 0

/--
Constructive balanced decomposition theorem (integer-state form), parameterized
by a one-step relation.
-/
theorem splitBalancedDecompositionInt_of_step
    (a : F) (b k : Nat)
    (hStep : ∀ s : Int,
      s = balancedDigitOfState s b + Int.ofNat b * balancedNextState s b) :
    F.centeredRep a =
      splitBalancedDigitPrefixSumInt (F.centeredRep a) b k +
        (Int.ofNat b ^ k) * splitBalancedStateAfterInt (F.centeredRep a) b k := by
  simpa [splitBalancedDigitPrefixSumInt, splitBalancedStateAfterInt] using
    (balancedState_telescope_of_step (state := F.centeredRep a) (b := b) hStep k)

/--
Terminal-zero corollary for the constructive balanced decomposition theorem.
-/
theorem splitBalancedDecompositionInt_of_terminal_zero_of_step
    (a : F) (b k : Nat)
    (hb : b ≥ 2)
    (hStep : ∀ s : Int,
      s = balancedDigitOfState s b + Int.ofNat b * balancedNextState s b)
    (hTerm : splitBalancedTerminalZeroProp a b k) :
    F.centeredRep a =
      splitBalancedDigitPrefixSumInt (F.centeredRep a) b k := by
  have hDec :=
    splitBalancedDecompositionInt_of_step (a := a) (b := b) (k := k) hStep
  have hbNotLt : ¬ b < 2 := Nat.not_lt.mpr hb
  have hTerm' : splitBalancedStateAfterInt (F.centeredRep a) b k = 0 := by
    unfold splitBalancedTerminalZeroProp splitBalancedTerminalState at hTerm
    simpa [hbNotLt, splitBalancedStateAfterInt] using hTerm
  simpa [hTerm', Int.mul_zero, Int.add_zero] using hDec

/--
Canonical constructive balanced decomposition theorem (`b ≥ 2`).
-/
theorem splitBalancedDecompositionInt
    (a : F) (b k : Nat) (hb : b ≥ 2) :
    F.centeredRep a =
      splitBalancedDigitPrefixSumInt (F.centeredRep a) b k +
        (Int.ofNat b ^ k) * splitBalancedStateAfterInt (F.centeredRep a) b k := by
  refine splitBalancedDecompositionInt_of_step (a := a) (b := b) (k := k) ?_
  intro s
  exact balancedStep_decomposition (state := s) (b := b) hb

/--
Equivalent fold form of `splitBalancedDecompositionInt`, exposing the exact
digit stream indexed by extraction step.
-/
theorem splitBalancedDecompositionInt_fold
    (a : F) (b k : Nat) (hb : b ≥ 2) :
    F.centeredRep a =
      (List.range k).foldl
        (fun acc i =>
          acc + balancedDigitOfState
            (splitBalancedStateAfterInt (F.centeredRep a) b i) b * (Int.ofNat b ^ i))
        0 +
      (Int.ofNat b ^ k) * splitBalancedStateAfterInt (F.centeredRep a) b k := by
  simpa [splitBalancedDigitPrefixSumInt_eq_fold] using
    (splitBalancedDecompositionInt a b k hb)

/--
Terminal-zero corollary of `splitBalancedDecompositionInt`.
-/
theorem splitBalancedDecompositionInt_of_terminal_zero
    (a : F) (b k : Nat) (hb : b ≥ 2)
    (hTerm : splitBalancedTerminalZeroProp a b k) :
    F.centeredRep a =
      splitBalancedDigitPrefixSumInt (F.centeredRep a) b k := by
  refine splitBalancedDecompositionInt_of_terminal_zero_of_step
      (a := a) (b := b) (k := k) hb ?_ hTerm
  intro s
  exact balancedStep_decomposition (state := s) (b := b) hb

/--
If balanced digits are congruent to states modulo `b` at each step, the
constructive integer decomposition theorem follows.
-/
theorem splitBalancedDecompositionInt_of_congruence
    (a : F) (b k : Nat)
    (hCongr : ∀ s : Int,
      s % Int.ofNat b = balancedDigitOfState s b % Int.ofNat b) :
    F.centeredRep a =
      splitBalancedDigitPrefixSumInt (F.centeredRep a) b k +
        (Int.ofNat b ^ k) * splitBalancedStateAfterInt (F.centeredRep a) b k := by
  refine splitBalancedDecompositionInt_of_step (a := a) (b := b) (k := k) ?_
  intro s
  exact balancedStep_decomposition_of_congruence (state := s) (b := b) (hModEq := hCongr s)

/-- Integer digit extracted at step `i` for one scalar. -/
private def splitBalancedScalarDigitInt (a : F) (b : Nat) (i : Nat) : Int :=
  balancedDigitOfState (splitBalancedStateAfterInt (F.centeredRep a) b i) b

/--
Balanced base-`b` split of one scalar into `k` digits.

For `b < 2`, returns the zero digit array of length `k`.
-/
def splitBalancedScalar (a : F) (b k : Nat) : Array F :=
  if b < 2 then
    Array.replicate k 0
  else
    Array.ofFn (fun i : Fin k => fieldOfSignedDigit (splitBalancedScalarDigitInt a b i.1))

@[simp] theorem splitBalancedScalar_size (a : F) (b k : Nat) :
    (splitBalancedScalar a b k).size = k := by
  by_cases hb : b < 2
  · simp [splitBalancedScalar, hb]
  · simp [splitBalancedScalar, hb]

/-- Degenerate-base behavior (`b < 2`) for balanced scalar split. -/
theorem splitBalancedScalar_eq_replicate_zero_of_base_lt_two
    {a : F} {b k : Nat}
    (hb : b < 2) :
    splitBalancedScalar a b k = Array.replicate k 0 := by
  simp [splitBalancedScalar, hb]

/-- Step-indexed view of balanced scalar digits for `b ≥ 2`. -/
theorem splitBalancedScalar_get_of_base_ge_two
    (a : F) (b k : Nat) (hb : b ≥ 2) (i : Fin k) :
    (splitBalancedScalar a b k)[i.1]! =
      fieldOfSignedDigit
        (balancedDigitOfState
          (splitBalancedStateAfterInt (F.centeredRep a) b i.1) b) := by
  have hbNotLt : ¬ b < 2 := Nat.not_lt.mpr hb
  simp [splitBalancedScalar, hbNotLt, splitBalancedScalarDigitInt]

/-- Balanced base-`b` split of a vector into `k` digit rows. -/
def splitBalancedVec (z : Array F) (b k : Nat) : Array (Array F) :=
  Array.ofFn (fun i : Fin k => z.map (fun a => (splitBalancedScalar a b k)[i.1]!))

@[simp] theorem splitBalancedVec_size (z : Array F) (b k : Nat) :
    (splitBalancedVec z b k).size = k := by
  simp [splitBalancedVec]

/-- Each balanced digit row has the same width as the source vector. -/
theorem splitBalancedVec_row_size
    (z : Array F) (b k : Nat) (i : Fin k) :
    ((splitBalancedVec z b k)[i.1]!).size = z.size := by
  simp [splitBalancedVec]

private def powF (x : F) : Nat → F
  | 0 => 1
  | n + 1 => powF x n * x

private theorem fieldOfSignedDigit_pow_ofNat (b i : Nat) :
    fieldOfSignedDigit (Int.ofNat b ^ i) = powF (F.ofNat b) i := by
  induction i with
  | zero =>
      simp [powF, fieldOfSignedDigit_one]
  | succ i ih =>
      calc
        fieldOfSignedDigit (Int.ofNat b ^ (i + 1))
            = fieldOfSignedDigit ((Int.ofNat b ^ i) * Int.ofNat b) := by
                simp [Int.pow_succ]
        _ = fieldOfSignedDigit (Int.ofNat b ^ i) * fieldOfSignedDigit (Int.ofNat b) := by
              rw [fieldOfSignedDigit_mul]
        _ = powF (F.ofNat b) i * fieldOfSignedDigit (Int.ofNat b) := by
              exact congrArg (fun t => t * fieldOfSignedDigit (Int.ofNat b)) ih
        _ = powF (F.ofNat b) i * F.ofNat b := by
              rw [fieldOfSignedDigit_ofNat b]
        _ = powF (F.ofNat b) (i + 1) := by
              simp [powF]

private theorem foldl_fieldOfSignedDigit_hom
    (b : Nat) (f : Nat → Int) :
    ∀ (l : List Nat) (accInt : Int),
      (l.foldl
        (fun acc i => acc + powF (F.ofNat b) i * fieldOfSignedDigit (f i))
        (fieldOfSignedDigit accInt))
      =
      fieldOfSignedDigit
        (l.foldl
          (fun acc i => acc + (Int.ofNat b ^ i) * f i)
          accInt) := by
  intro l
  induction l with
  | nil =>
      intro accInt
      simp
  | cons i tl ih =>
      intro accInt
      have hStep :
          fieldOfSignedDigit accInt + powF (F.ofNat b) i * fieldOfSignedDigit (f i) =
            fieldOfSignedDigit (accInt + (Int.ofNat b ^ i) * f i) := by
        calc
          fieldOfSignedDigit accInt + powF (F.ofNat b) i * fieldOfSignedDigit (f i)
              = fieldOfSignedDigit accInt +
                  fieldOfSignedDigit (Int.ofNat b ^ i) * fieldOfSignedDigit (f i) := by
                    rw [← fieldOfSignedDigit_pow_ofNat]
          _ = fieldOfSignedDigit accInt +
                fieldOfSignedDigit ((Int.ofNat b ^ i) * f i) := by
                  rw [← fieldOfSignedDigit_mul]
          _ = fieldOfSignedDigit (accInt + (Int.ofNat b ^ i) * f i) := by
                  rw [← fieldOfSignedDigit_add]
      simp [List.foldl, hStep]
      exact ih (accInt + (Int.ofNat b ^ i) * f i)

private theorem foldl_congr_on_list
    {α β : Type}
    (l : List α) (init : β)
    (f g : β → α → β)
    (hfg : ∀ acc x, x ∈ l → f acc x = g acc x) :
    l.foldl f init = l.foldl g init := by
  induction l generalizing init with
  | nil =>
      rfl
  | cons x xs ih =>
      have hHead : f init x = g init x := hfg init x (by simp)
      calc
        (x :: xs).foldl f init = xs.foldl f (f init x) := by rfl
        _ = xs.foldl f (g init x) := by rw [hHead]
        _ = xs.foldl g (g init x) := by
              apply ih
              intro acc y hy
              exact hfg acc y (by simp [hy])
        _ = (x :: xs).foldl g init := by rfl

/-- Scalar recomposition from balanced split digits. -/
def recomposeSplitBalancedScalar (a : F) (b k : Nat) : F :=
  (List.range k).foldl
    (fun acc i => acc + powF (F.ofNat b) i * (splitBalancedScalar a b k)[i]!)
    0

/--
Scalar field-lift boundary:
the field recomposition of split digits matches the integer digit-prefix sum embedding.
-/
def splitBalancedScalarFieldLiftProp (a : F) (b k : Nat) : Prop :=
  recomposeSplitBalancedScalar a b k =
    fieldOfSignedDigit (splitBalancedDigitPrefixSumInt (F.centeredRep a) b k)

/--
Canonical closure of scalar field-lift for balanced decomposition when `b ≥ 2`.
-/
theorem splitBalancedScalarFieldLiftProp_holds_of_base_ge_two
    (a : F) (b k : Nat)
    (hb : b ≥ 2) :
    splitBalancedScalarFieldLiftProp a b k := by
  unfold splitBalancedScalarFieldLiftProp recomposeSplitBalancedScalar
  have hDigits :
      ∀ i, i ∈ List.range k →
        (splitBalancedScalar a b k)[i]! =
          fieldOfSignedDigit
            (balancedDigitOfState
              (splitBalancedStateAfterInt (F.centeredRep a) b i) b) := by
    intro i hiMem
    exact splitBalancedScalar_get_of_base_ge_two a b k hb ⟨i, List.mem_range.mp hiMem⟩
  have hFoldRewrite :
      (List.range k).foldl
        (fun acc i => acc + powF (F.ofNat b) i * (splitBalancedScalar a b k)[i]!)
        0
      =
      (List.range k).foldl
        (fun acc i =>
          acc + powF (F.ofNat b) i *
            fieldOfSignedDigit
              (balancedDigitOfState
                (splitBalancedStateAfterInt (F.centeredRep a) b i) b))
        0 := by
    exact foldl_congr_on_list
      (l := List.range k)
      (init := 0)
      (f := fun acc i => acc + powF (F.ofNat b) i * (splitBalancedScalar a b k)[i]!)
      (g := fun acc i =>
        acc + powF (F.ofNat b) i *
          fieldOfSignedDigit
            (balancedDigitOfState
              (splitBalancedStateAfterInt (F.centeredRep a) b i) b))
      (by
        intro acc i hiMem
        simp [hDigits i hiMem])
  rw [hFoldRewrite]
  have hFoldIntComm :
      (List.range k).foldl
        (fun acc i =>
          acc + (Int.ofNat b ^ i) *
            balancedDigitOfState
              (splitBalancedStateAfterInt (F.centeredRep a) b i) b)
        0
      =
      (List.range k).foldl
        (fun acc i =>
          acc + balancedDigitOfState
            (splitBalancedStateAfterInt (F.centeredRep a) b i) b * (Int.ofNat b ^ i))
        0 := by
    exact foldl_congr_on_list
      (l := List.range k)
      (init := 0)
      (f := fun acc i =>
        acc + (Int.ofNat b ^ i) *
          balancedDigitOfState
            (splitBalancedStateAfterInt (F.centeredRep a) b i) b)
      (g := fun acc i =>
        acc + balancedDigitOfState
          (splitBalancedStateAfterInt (F.centeredRep a) b i) b * (Int.ofNat b ^ i))
      (by
        intro acc i hiMem
        simp [Int.mul_comm])
  have hHom :
      (List.range k).foldl
        (fun acc i =>
          acc + powF (F.ofNat b) i *
            fieldOfSignedDigit
              (balancedDigitOfState
                (splitBalancedStateAfterInt (F.centeredRep a) b i) b))
        0
      =
      fieldOfSignedDigit
        ((List.range k).foldl
          (fun acc i =>
            acc + (Int.ofNat b ^ i) *
              balancedDigitOfState
                (splitBalancedStateAfterInt (F.centeredRep a) b i) b)
          0) := by
    have h := foldl_fieldOfSignedDigit_hom
      (b := b)
      (f := fun i =>
        balancedDigitOfState
          (splitBalancedStateAfterInt (F.centeredRep a) b i) b)
      (l := List.range k)
      (accInt := 0)
    simpa [fieldOfSignedDigit_zero] using h
  rw [hHom, hFoldIntComm]
  simpa [splitBalancedDigitPrefixSumInt_eq_fold]

/-- Scalar digit-bound boundary for balanced decomposition. -/
def splitBalancedScalarDigitBoundProp (a : F) (b k : Nat) : Prop :=
  ∀ i : Fin k, normInfF ((splitBalancedScalar a b k)[i.1]!) < b

/--
Scalar constructive bridge:
if terminal state is zero and scalar field-lift is available, recomposition recovers
the original scalar.
-/
theorem splitBalancedScalar_recompose_eq_of_terminal_zero_of_field_lift
    (a : F) (b k : Nat)
    (hb : b ≥ 2)
    (hTerm : splitBalancedTerminalZeroProp a b k)
    (hLift : splitBalancedScalarFieldLiftProp a b k) :
    recomposeSplitBalancedScalar a b k = a := by
  have hDec :
      F.centeredRep a =
        splitBalancedDigitPrefixSumInt (F.centeredRep a) b k :=
    splitBalancedDecompositionInt_of_terminal_zero a b k hb hTerm
  calc
    recomposeSplitBalancedScalar a b k
        = fieldOfSignedDigit (splitBalancedDigitPrefixSumInt (F.centeredRep a) b k) := hLift
    _ = fieldOfSignedDigit (F.centeredRep a) := by
          exact (congrArg fieldOfSignedDigit hDec).symm
    _ = a := fieldOfSignedDigit_centeredRep a

/-- Recompose `z = Σ_i b^i * z_i` from split digits. -/
def recomposeSplitDigits (digits : Array (Array F)) (b : Nat) : Array F :=
  if digits.isEmpty then
    #[]
  else
    let width := (digits[0]!).size
    Array.ofFn (fun j : Fin width =>
      (List.range digits.size).foldl
        (fun acc i => acc + powF (F.ofNat b) i * (digits[i]![j.1]!))
        0)

/-- Vector terminal-zero boundary (per coordinate scalar terminal state). -/
def splitBalancedVecTerminalZeroProp (z : Array F) (b k : Nat) : Prop :=
  ∀ j : Fin z.size, splitBalancedTerminalZeroProp z[j.1]! b k

/-- Vector scalar-lift boundary (per coordinate scalar field-lift). -/
def splitBalancedVecFieldLiftProp (z : Array F) (b k : Nat) : Prop :=
  ∀ j : Fin z.size, splitBalancedScalarFieldLiftProp z[j.1]! b k

/-- Vector-level closure of scalar field-lift for `b ≥ 2`. -/
theorem splitBalancedVecFieldLiftProp_holds_of_base_ge_two
    (z : Array F) (b k : Nat)
    (hb : b ≥ 2) :
    splitBalancedVecFieldLiftProp z b k := by
  intro j
  exact splitBalancedScalarFieldLiftProp_holds_of_base_ge_two z[j.1]! b k hb

/-- Vector scalar digit-bound boundary (per coordinate scalar digit bounds). -/
def splitBalancedVecDigitBoundProp (z : Array F) (b k : Nat) : Prop :=
  ∀ j : Fin z.size, splitBalancedScalarDigitBoundProp z[j.1]! b k

/--
Entrywise identity: recomposing `splitBalancedVec` at column `j` equals
scalar recomposition of the `j`-th source scalar.
-/
theorem recomposeSplitDigits_splitBalancedVec_entry
    (z : Array F) (b k : Nat) (hk : 0 < k) (j : Fin z.size) :
    (recomposeSplitDigits (splitBalancedVec z b k) b)[j.1]! =
      recomposeSplitBalancedScalar z[j.1]! b k := by
  have hkNe : k ≠ 0 := Nat.ne_of_gt hk
  have hInner :
      ∀ i : Nat,
        (Array.ofFn (fun i : Fin k => z.map (fun a => (splitBalancedScalar a b k)[i.1])))[i]![j.1]! =
          (splitBalancedScalar z[j.1]! b k)[i]! := by
    intro i
    by_cases hi : i < k
    · simp [hi, j.2]
    · have hOut : ¬ i < (splitBalancedScalar z[j.1]! b k).size := by
        simpa [splitBalancedScalar_size] using hi
      have hDefaultDomLe : (default : Array F).size ≤ j.1 := by
        change 0 ≤ j.1
        exact Nat.zero_le j.1
      have hDefaultDom : ¬ j.1 < (default : Array F).size :=
        Nat.not_lt.mpr hDefaultDomLe
      have hDefault :
          (default : Array F)[j.1]! = (default : F) := by
        simpa using (getElem!_neg (c := (default : Array F)) (i := j.1) (h := hDefaultDom))
      simpa [hi, hOut] using hDefault
  have hBody :
      (fun acc i =>
        acc + powF (F.ofNat b) i *
          (Array.ofFn (fun i : Fin k => z.map (fun a => (splitBalancedScalar a b k)[i.1])))[i]![j.1]!) =
        (fun acc i => acc + powF (F.ofNat b) i * (splitBalancedScalar z[j.1]! b k)[i]!) := by
    funext acc i
    simp [hInner i]
  unfold recomposeSplitDigits recomposeSplitBalancedScalar
  simp [hkNe, splitBalancedVec]
  calc
    (Array.ofFn
      (fun j =>
        List.foldl
          (fun acc i =>
            acc + powF (F.ofNat b) i *
              (Array.ofFn (fun i : Fin k => z.map (fun a => (splitBalancedScalar a b k)[i.1])))[i]![j.1]!)
          0 (List.range k)))[j.1]!
        =
      List.foldl
        (fun acc i =>
          acc + powF (F.ofNat b) i *
            (Array.ofFn (fun i : Fin k => z.map (fun a => (splitBalancedScalar a b k)[i.1])))[i]![j.1]!)
        0 (List.range k) := by
          have hWidth :
              (Array.ofFn (fun i : Fin k => z.map (fun a => (splitBalancedScalar a b k)[i.1])))[0]!.size
                = z.size := by
            have h0 : 0 < k := hk
            simp [h0]
          simp [hWidth, j.2]
    _ =
      List.foldl
        (fun acc i => acc + powF (F.ofNat b) i * (splitBalancedScalar z[j.1]! b k)[i]!)
        0 (List.range k) := by
          simp [hBody]
    _ =
      List.foldl
        (fun acc i => acc + powF (F.ofNat b) i * (splitBalancedScalar z[j.1] b k)[i]!)
        0 (List.range k) := by
          simp [j.2]

/--
Vector constructive bridge from per-coordinate terminal-zero + scalar field-lift:
recomposition of `splitBalancedVec` recovers the input vector.
-/
theorem recomposeSplitDigits_splitBalancedVec_eq_of_terminal_zero_of_field_lift
    (z : Array F) (b k : Nat)
    (hb : b ≥ 2) (hk : 0 < k)
    (hTerm : splitBalancedVecTerminalZeroProp z b k)
    (hLift : splitBalancedVecFieldLiftProp z b k) :
    recomposeSplitDigits (splitBalancedVec z b k) b = z := by
  apply Array.ext
  · have hkNe : k ≠ 0 := Nat.ne_of_gt hk
    have hNotEmpty : ¬ (splitBalancedVec z b k).isEmpty := by
      simp [Array.isEmpty, splitBalancedVec_size, hkNe]
    have hRow0Size : ((splitBalancedVec z b k)[0]!).size = z.size := by
      exact splitBalancedVec_row_size z b k ⟨0, hk⟩
    simp [recomposeSplitDigits, hNotEmpty, hRow0Size]
  · intro i hi₁ hi₂
    let j : Fin z.size := ⟨i, hi₂⟩
    have hMain :
        (recomposeSplitDigits (splitBalancedVec z b k) b)[j.1]! = z[j.1]! := by
      calc
        (recomposeSplitDigits (splitBalancedVec z b k) b)[j.1]!
            = recomposeSplitBalancedScalar z[j.1]! b k := by
                exact recomposeSplitDigits_splitBalancedVec_entry z b k hk j
        _ = z[j.1]! := by
              exact splitBalancedScalar_recompose_eq_of_terminal_zero_of_field_lift
                (a := z[j.1]!) (b := b) (k := k) hb (hTerm j) (hLift j)
    simpa [j, hi₁, hi₂] using hMain

/-- Check that all digits lie in the expected centered norm window `< b`. -/
def digitsWithinBase (digits : Array (Array F)) (b : Nat) : Bool :=
  digits.all (fun row => row.all (fun x => normInfF x < b))

/--
Proposition-level surface corresponding to `digitsWithinBase`.

This is intentionally theorem-facing (`Prop`) so downstream layers can avoid
carrying raw `Bool` equalities as assumptions.
-/
def digitsWithinBaseProp (digits : Array (Array F)) (b : Nat) : Prop :=
  ∀ row, row ∈ digits → ∀ x, x ∈ row → normInfF x < b

/--
Vector digit-window predicate from per-coordinate scalar digit bounds.
-/
theorem digitsWithinBaseProp_of_scalar_digit_bounds
    (z : Array F) (b k : Nat)
    (hBounds : splitBalancedVecDigitBoundProp z b k) :
    digitsWithinBaseProp (splitBalancedVec z b k) b := by
  intro row hRow x hx
  rcases (Array.mem_iff_getElem).1 hRow with ⟨i, hi, hRowEq⟩
  rcases (Array.mem_iff_getElem).1 hx with ⟨j, hj, hXEq⟩
  subst hRowEq
  subst hXEq
  have hiK : i < k := by simpa [splitBalancedVec_size] using hi
  have hRowSize : ((splitBalancedVec z b k)[i]).size = z.size := by
    simp [splitBalancedVec]
  have hjZ : j < z.size := by
    simpa [hRowSize] using hj
  have hScalar : splitBalancedScalarDigitBoundProp z[j]! b k := hBounds ⟨j, hjZ⟩
  have hDigit : normInfF ((splitBalancedScalar z[j]! b k)[i]!) < b := hScalar ⟨i, hiK⟩
  simpa [splitBalancedVec, hiK, hjZ, splitBalancedScalar_size] using hDigit

/--
Converse lift for the balanced split shape:
if all entries of `splitBalancedVec z b k` satisfy the proposition-level digit window,
then we recover the per-coordinate scalar digit-bound boundary.
-/
theorem splitBalancedVecDigitBoundProp_of_digitsWithinBaseProp
    (z : Array F) (b k : Nat)
    (hDigits : digitsWithinBaseProp (splitBalancedVec z b k) b) :
    splitBalancedVecDigitBoundProp z b k := by
  intro j i
  let row : Array F := (splitBalancedVec z b k)[i.1]!
  have hRowMem : row ∈ splitBalancedVec z b k := by
    apply (Array.mem_iff_getElem).2
    refine ⟨i.1, ?_, ?_⟩
    · simpa [splitBalancedVec_size] using i.2
    · simp [row]
  have hRowSize : row.size = z.size := by
    simpa [row] using splitBalancedVec_row_size z b k i
  have hjRow : j.1 < row.size := by
    simpa [hRowSize] using j.2
  have hXMem : row[j.1] ∈ row := by
    apply (Array.mem_iff_getElem).2
    exact ⟨j.1, hjRow, rfl⟩
  have hBound : normInfF (row[j.1]) < b :=
    hDigits row hRowMem (row[j.1]) hXMem
  simpa [row, splitBalancedVec, i.2, j.2, hjRow] using hBound

/--
Canonical iff between per-coordinate scalar digit-bound obligations and the
row-wise proposition-level digit window on `splitBalancedVec`.
-/
theorem splitBalancedVecDigitBoundProp_iff_digitsWithinBaseProp
    (z : Array F) (b k : Nat) :
    splitBalancedVecDigitBoundProp z b k ↔
      digitsWithinBaseProp (splitBalancedVec z b k) b := by
  constructor
  · intro hBounds
    exact digitsWithinBaseProp_of_scalar_digit_bounds z b k hBounds
  · intro hDigits
    exact splitBalancedVecDigitBoundProp_of_digitsWithinBaseProp z b k hDigits

/-- Canonical bridge between executable digit bound checks and proposition-level obligations. -/
theorem digitsWithinBase_eq_true_iff_prop
  {digits : Array (Array F)} {b : Nat} :
  digitsWithinBase digits b = true ↔ digitsWithinBaseProp digits b := by
  unfold digitsWithinBase digitsWithinBaseProp
  constructor
  · intro h row hRow x hx
    have hRowOk : row.all (fun y => normInfF y < b) = true :=
      (Array.all_eq_true'.1 h) row hRow
    exact decide_eq_true_eq.mp ((Array.all_eq_true'.1 hRowOk) x hx)
  · intro h
    refine Array.all_eq_true'.2 ?_
    intro row hRow
    refine Array.all_eq_true'.2 ?_
    intro x hx
    simpa using (h row hRow x hx)

/--
Proposition-level round-trip boundary for balanced split.

This packages the decomposition obligations without exposing `Bool` directly.
-/
def splitBalancedRoundTripProp (z : Array F) (b k : Nat) : Prop :=
  b ≥ 2 ∧
    let digits := splitBalancedVec z b k
    recomposeSplitDigits digits b = z ∧ digitsWithinBaseProp digits b

/--
Constructive P6 closure (boundary-facing):
if per-coordinate terminal-zero, field-lift, and digit-window boundaries hold,
then the proposition-level balanced round-trip property holds.
-/
theorem splitBalancedRoundTripProp_of_constructive_boundaries
    (z : Array F) (b k : Nat)
    (hb : b ≥ 2) (hk : 0 < k)
    (hTerm : splitBalancedVecTerminalZeroProp z b k)
    (hLift : splitBalancedVecFieldLiftProp z b k)
    (hBounds : splitBalancedVecDigitBoundProp z b k) :
    splitBalancedRoundTripProp z b k := by
  have hRecompose :
      recomposeSplitDigits (splitBalancedVec z b k) b = z :=
    recomposeSplitDigits_splitBalancedVec_eq_of_terminal_zero_of_field_lift
      z b k hb hk hTerm hLift
  have hDigits :
      digitsWithinBaseProp (splitBalancedVec z b k) b :=
    digitsWithinBaseProp_of_scalar_digit_bounds z b k hBounds
  refine ⟨hb, ?_⟩
  exact ⟨hRecompose, hDigits⟩

/--
Constructive P6 closure without requiring an external field-lift premise:
for `b ≥ 2`, scalar field-lift is closed internally.
-/
theorem splitBalancedRoundTripProp_of_constructive_boundaries_no_field_lift
    (z : Array F) (b k : Nat)
    (hb : b ≥ 2) (hk : 0 < k)
    (hTerm : splitBalancedVecTerminalZeroProp z b k)
    (hBounds : splitBalancedVecDigitBoundProp z b k) :
    splitBalancedRoundTripProp z b k := by
  exact splitBalancedRoundTripProp_of_constructive_boundaries
    z b k hb hk hTerm
    (splitBalancedVecFieldLiftProp_holds_of_base_ge_two z b k hb)
    hBounds

/--
Executable split/recompose consistency check used by legacy split/arithmetic-bundle call sites.

Returns `true` only when:
- `b >= 2`,
- recomposition matches input,
- digits satisfy the norm bound check.
-/
def splitRoundTrip (z : Array F) (b k : Nat) : Bool :=
  if b < 2 then
    false
  else
    let digits := splitBalancedVec z b k
    decide (recomposeSplitDigits digits b = z) && digitsWithinBase digits b

/-- Soundness: a successful split round-trip check implies proposition-level obligations. -/
theorem splitRoundTrip_sound
  {z : Array F} {b k : Nat}
  (hOk : splitRoundTrip z b k = true) :
  b ≥ 2 ∧
    let digits := splitBalancedVec z b k
    recomposeSplitDigits digits b = z ∧ digitsWithinBase digits b = true := by
  unfold splitRoundTrip at hOk
  by_cases hb : b < 2
  · simp [hb] at hOk
  · have hbGe : b ≥ 2 := Nat.le_of_not_lt hb
    have hAnd :
        decide (recomposeSplitDigits (splitBalancedVec z b k) b = z) = true ∧
          digitsWithinBase (splitBalancedVec z b k) b = true := by
      simpa [hb, Bool.and_eq_true] using hOk
    refine ⟨hbGe, ?_⟩
    exact ⟨decide_eq_true_eq.mp hAnd.1, hAnd.2⟩

/-- Completeness: proposition-level split obligations imply successful check. -/
theorem splitRoundTrip_complete
  {z : Array F} {b k : Nat}
  (hProp : b ≥ 2 ∧
    let digits := splitBalancedVec z b k
    recomposeSplitDigits digits b = z ∧ digitsWithinBase digits b = true) :
  splitRoundTrip z b k = true := by
  rcases hProp with ⟨hbGe, hRest⟩
  rcases hRest with ⟨hRec, hDigits⟩
  have hbNotLt : ¬ b < 2 := Nat.not_lt.mpr hbGe
  unfold splitRoundTrip
  simp [hbNotLt, hRec, hDigits]

/-- Canonical iff surface for the balanced split check. -/
theorem splitRoundTrip_eq_true_iff
  {z : Array F} {b k : Nat} :
  splitRoundTrip z b k = true ↔
    b ≥ 2 ∧
      let digits := splitBalancedVec z b k
      recomposeSplitDigits digits b = z ∧ digitsWithinBase digits b = true := by
  constructor
  · exact splitRoundTrip_sound
  · exact splitRoundTrip_complete

/-- Soundness bridge from executable split check to proposition-level round-trip boundary. -/
theorem splitRoundTrip_sound_prop
  {z : Array F} {b k : Nat}
  (hOk : splitRoundTrip z b k = true) :
  splitBalancedRoundTripProp z b k := by
  rcases splitRoundTrip_sound hOk with ⟨hbGe, hRest⟩
  rcases hRest with ⟨hRec, hDigitsBool⟩
  refine ⟨hbGe, ?_⟩
  refine ⟨hRec, ?_⟩
  exact (digitsWithinBase_eq_true_iff_prop).1 hDigitsBool

/-- Completeness bridge from proposition-level round-trip boundary to executable split check. -/
theorem splitRoundTrip_complete_prop
  {z : Array F} {b k : Nat}
  (hProp : splitBalancedRoundTripProp z b k) :
  splitRoundTrip z b k = true := by
  rcases hProp with ⟨hbGe, hRest⟩
  rcases hRest with ⟨hRec, hDigitsProp⟩
  have hDigitsBool : digitsWithinBase (splitBalancedVec z b k) b = true :=
    (digitsWithinBase_eq_true_iff_prop).2 hDigitsProp
  exact splitRoundTrip_complete ⟨hbGe, hRec, hDigitsBool⟩

/-- Canonical iff between the executable split check and proposition-level round-trip boundary. -/
theorem splitRoundTrip_eq_true_iff_prop
  {z : Array F} {b k : Nat} :
  splitRoundTrip z b k = true ↔ splitBalancedRoundTripProp z b k := by
  constructor
  · exact splitRoundTrip_sound_prop
  · intro hProp
    exact splitRoundTrip_complete_prop hProp

/-- Extract base guard from proposition-level balanced round-trip obligations. -/
theorem splitBalancedRoundTripProp_base_ge_two
  {z : Array F} {b k : Nat}
  (hProp : splitBalancedRoundTripProp z b k) :
  b ≥ 2 :=
  hProp.1

/-- Extract recomposition equality from proposition-level balanced round-trip obligations. -/
theorem splitBalancedRoundTripProp_recompose_eq
  {z : Array F} {b k : Nat}
  (hProp : splitBalancedRoundTripProp z b k) :
  recomposeSplitDigits (splitBalancedVec z b k) b = z := by
  exact hProp.2.1

/-- Extract digit-window predicate from proposition-level balanced round-trip obligations. -/
theorem splitBalancedRoundTripProp_digitsWithinBaseProp
  {z : Array F} {b k : Nat}
  (hProp : splitBalancedRoundTripProp z b k) :
  digitsWithinBaseProp (splitBalancedVec z b k) b := by
  exact hProp.2.2

/-- Base guard: when `b < 2`, the round-trip check must fail. -/
theorem splitRoundTrip_eq_false_of_base_lt_two
  {z : Array F} {b k : Nat}
  (hb : b < 2) :
  splitRoundTrip z b k = false := by
  unfold splitRoundTrip
  simp [hb]

/-- Derived recomposition equation from a successful check. -/
theorem splitRoundTrip_recompose_eq
  {z : Array F} {b k : Nat}
  (hOk : splitRoundTrip z b k = true) :
  recomposeSplitDigits (splitBalancedVec z b k) b = z := by
  exact (splitRoundTrip_sound hOk).2.1

/-- Derived digit-window check from a successful check. -/
theorem splitRoundTrip_digitsWithinBase_eq_true
  {z : Array F} {b k : Nat}
  (hOk : splitRoundTrip z b k = true) :
  digitsWithinBase (splitBalancedVec z b k) b = true := by
  exact (splitRoundTrip_sound hOk).2.2

/-- Successful round-trip check implies the base guard `b >= 2`. -/
theorem splitRoundTrip_base_ge_two
  {z : Array F} {b k : Nat}
  (hOk : splitRoundTrip z b k = true) :
  b ≥ 2 := by
  exact (splitRoundTrip_sound hOk).1

/-- Successful round-trip check implies the proposition-level digit-window predicate. -/
theorem splitRoundTrip_digitsWithinBase_prop
  {z : Array F} {b k : Nat}
  (hOk : splitRoundTrip z b k = true) :
  digitsWithinBaseProp (splitBalancedVec z b k) b := by
  exact splitBalancedRoundTripProp_digitsWithinBaseProp (splitRoundTrip_sound_prop hOk)

/--
Successful executable round-trip check implies the per-coordinate scalar
digit-bound boundary for the canonical `splitBalancedVec` shape.
-/
theorem splitRoundTrip_splitBalancedVecDigitBoundProp
  {z : Array F} {b k : Nat}
  (hOk : splitRoundTrip z b k = true) :
  splitBalancedVecDigitBoundProp z b k := by
  exact splitBalancedVecDigitBoundProp_of_digitsWithinBaseProp z b k
    (splitRoundTrip_digitsWithinBase_prop hOk)

/--
Direct proposition-level digit-window extraction from per-coordinate scalar
digit-bound obligations for the canonical split shape.
-/
theorem splitBalancedVecDigitBoundProp_digitsWithinBaseProp
  {z : Array F} {b k : Nat}
  (hBounds : splitBalancedVecDigitBoundProp z b k) :
  digitsWithinBaseProp (splitBalancedVec z b k) b := by
  exact (splitBalancedVecDigitBoundProp_iff_digitsWithinBaseProp z b k).1 hBounds

/--
From proposition-level per-coordinate scalar digit bounds, the executable
digit-window check succeeds on `splitBalancedVec`.
-/
theorem digitsWithinBase_eq_true_of_splitBalancedVecDigitBoundProp
  {z : Array F} {b k : Nat}
  (hBounds : splitBalancedVecDigitBoundProp z b k) :
  digitsWithinBase (splitBalancedVec z b k) b = true := by
  exact (digitsWithinBase_eq_true_iff_prop).2
    ((splitBalancedVecDigitBoundProp_iff_digitsWithinBaseProp z b k).1 hBounds)

end SuperNeo
