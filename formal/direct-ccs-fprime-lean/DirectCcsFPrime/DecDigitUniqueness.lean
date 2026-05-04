/-!
Digit-level uniqueness checks for DEC authorization.

This module deliberately separates two facts:
- signed low-norm base-2 digits are not unique;
- canonical binary digits are unique in the local two-digit model.

The first theorem is the important guardrail for the implementation: proving
`Pi_DEC` recomposition plus `||child.z|| < 2` is not enough to justify a unique
private child vector.
-/

namespace DirectCcsFPrime

namespace DecDigitUniqueness

/-! ## Local two-digit counterexample -/

/-- Two-digit base-2 recomposition over integers. -/
def recompose2 (digits : Int × Int) : Int :=
  digits.1 + 2 * digits.2

/-- Signed low-norm digit window corresponding to `|digit| < 2`. -/
def signedLowNorm2 (digits : Int × Int) : Prop :=
  -1 ≤ digits.1 ∧ digits.1 ≤ 1 ∧ -1 ≤ digits.2 ∧ digits.2 ≤ 1

/-- Canonical binary digit window. -/
def binaryDigits2 (digits : Int × Int) : Prop :=
  (digits.1 = 0 ∨ digits.1 = 1) ∧ (digits.2 = 0 ∨ digits.2 = 1)

/--
Signed low-norm base-2 decompositions are not unique.

The same parent value `1` has both decompositions:

```text
1 = 1 + 2*0
1 = -1 + 2*1
```

Both digit pairs satisfy `|digit| < 2`. Therefore low-norm checks alone do not
force canonical DEC children.
-/
theorem signed_low_norm_base2_not_unique :
    recompose2 (1, 0) = recompose2 (-1, 1) ∧
    signedLowNorm2 (1, 0) ∧
    signedLowNorm2 (-1, 1) ∧
    (1, 0) ≠ (-1, 1) := by
  constructor
  · decide
  constructor
  · unfold signedLowNorm2
    omega
  constructor
  · unfold signedLowNorm2
    omega
  · intro h
    injection h with hFirst _hSecond
    omega

/-- Two canonical binary decompositions of value `1` must agree. -/
theorem binary_recompose_one_unique
    (digits : Int × Int)
    (hBin : binaryDigits2 digits)
    (hRec : recompose2 digits = 1) :
    digits = (1, 0) := by
  rcases hBin with ⟨h0 | h0, h1 | h1⟩
  · cases digits with
    | mk d0 d1 =>
        simp only at h0 h1 hRec
        subst d0
        subst d1
        unfold recompose2 at hRec
        omega
  · cases digits with
    | mk d0 d1 =>
        simp only at h0 h1 hRec
        subst d0
        subst d1
        unfold recompose2 at hRec
        omega
  · cases digits with
    | mk d0 d1 =>
        simp only at h0 h1
        subst d0
        subst d1
        rfl
  · cases digits with
    | mk d0 d1 =>
        simp only at h0 h1 hRec
        subst d0
        subst d1
        unfold recompose2 at hRec
        omega

/-! ## General canonical binary uniqueness -/

/-- Recompose least-significant-first base-2 natural digits. -/
def recomposeNatDigits : List Nat → Nat
  | [] => 0
  | d :: rest => d + 2 * recomposeNatDigits rest

/-- Canonical binary digit predicate for natural digits. -/
def binaryNatDigits (digits : List Nat) : Prop :=
  ∀ d, d ∈ digits → d < 2

private theorem head_eq_of_binary_recompose_eq
    {a b : Nat}
    {as bs : List Nat}
    (ha : a < 2)
    (hb : b < 2)
    (h :
      recomposeNatDigits (a :: as) =
      recomposeNatDigits (b :: bs)) :
    a = b := by
  have hMod := congrArg (fun n : Nat => n % 2) h
  simp [recomposeNatDigits, Nat.mod_eq_of_lt ha, Nat.mod_eq_of_lt hb] at hMod
  exact hMod

private theorem tail_recompose_eq_of_binary_recompose_eq
    {a : Nat}
    {as bs : List Nat}
    (h :
      recomposeNatDigits (a :: as) =
      recomposeNatDigits (a :: bs)) :
    recomposeNatDigits as = recomposeNatDigits bs := by
  simp [recomposeNatDigits] at h
  omega

theorem binary_nat_recompose_unique
    {as bs : List Nat}
    (hLen : as.length = bs.length)
    (hAs : binaryNatDigits as)
    (hBs : binaryNatDigits bs)
    (hRec : recomposeNatDigits as = recomposeNatDigits bs) :
    as = bs := by
  induction as generalizing bs with
  | nil =>
      cases bs with
      | nil => rfl
      | cons b bs =>
          simp at hLen
  | cons a as ih =>
      cases bs with
      | nil =>
          simp at hLen
      | cons b bs =>
          have ha : a < 2 := hAs a (by simp)
          have hb : b < 2 := hBs b (by simp)
          have hHead : a = b :=
            head_eq_of_binary_recompose_eq (as := as) (bs := bs) ha hb hRec
          subst b
          have hTailLen : as.length = bs.length := by
            simpa using hLen
          have hAsTail : binaryNatDigits as := by
            intro d hd
            exact hAs d (by simp [hd])
          have hBsTail : binaryNatDigits bs := by
            intro d hd
            exact hBs d (by simp [hd])
          have hTailRec : recomposeNatDigits as = recomposeNatDigits bs :=
            tail_recompose_eq_of_binary_recompose_eq hRec
          have hTail : as = bs :=
            ih hTailLen hAsTail hBsTail hTailRec
          subst hTail
          rfl

/--
Canonical binary decompositions are unique: no different same-length binary
digit list can recompose to the same value.
-/
theorem no_different_binary_recomposition
    {as bs : List Nat}
    (hLen : as.length = bs.length)
    (hAs : binaryNatDigits as)
    (hBs : binaryNatDigits bs)
    (hDiff : as ≠ bs) :
    recomposeNatDigits as ≠ recomposeNatDigits bs := by
  intro hRec
  exact hDiff (binary_nat_recompose_unique hLen hAs hBs hRec)

/--
Binary digits are still not enough without a fixed length: leading zeroes create
different decompositions of the same value.

This is the second guardrail for the reduced-handle implementation. A proof
that authorizes private children must prove exactly `k` child rows, not merely
that the supplied digits are binary and recompose correctly.
-/
theorem binary_recomposition_not_unique_without_length :
    recomposeNatDigits [] = recomposeNatDigits [0] ∧
    binaryNatDigits [] ∧
    binaryNatDigits [0] ∧
    ([] : List Nat) ≠ [0] := by
  constructor
  · rfl
  constructor
  · intro d hd
    cases hd
  constructor
  · intro d hd
    simp at hd
    omega
  · intro h
    cases h

/--
Even fixed-length binary digits are not unique if recomposition is checked only
modulo a small modulus without proving the recomposed integers are below the
modulus.

Both lists have length two and binary digits, but they recompose to `0` and `2`,
which are equal modulo `2`.
-/
theorem fixed_length_binary_mod_recomposition_not_unique_without_range :
    ([0, 0] : List Nat).length = ([0, 1] : List Nat).length ∧
    binaryNatDigits [0, 0] ∧
    binaryNatDigits [0, 1] ∧
    recomposeNatDigits [0, 0] % 2 = recomposeNatDigits [0, 1] % 2 ∧
    ([0, 0] : List Nat) ≠ [0, 1] := by
  constructor
  · rfl
  constructor
  · intro d hd
    simp at hd
    omega
  constructor
  · intro d hd
    simp at hd
    omega
  constructor
  · rfl
  · decide

/--
Binary recomposition is range-bounded by the digit length.

This is the bridge needed when an implementation checks field equality modulo
`q`: if `2^k < q` and the digit table has length `k`, binary recomposition is
below `q`, so modular equality can be lifted back to integer equality.
-/
theorem recomposeNatDigits_lt_two_pow_length
    (digits : List Nat)
    (hBin : binaryNatDigits digits) :
    recomposeNatDigits digits < 2 ^ digits.length := by
  induction digits with
  | nil =>
      simp [recomposeNatDigits]
  | cons d rest ih =>
      have hdLt : d < 2 := hBin d (by simp)
      have hdLe : d ≤ 1 := Nat.le_of_lt_succ hdLt
      have hRestBin : binaryNatDigits rest := by
        intro x hx
        exact hBin x (by simp [hx])
      have hRestLt : recomposeNatDigits rest < 2 ^ rest.length :=
        ih hRestBin
      have hRestLe : recomposeNatDigits rest ≤ 2 ^ rest.length - 1 :=
        Nat.le_pred_of_lt hRestLt
      have hPowPos : 0 < 2 ^ rest.length := Nat.two_pow_pos rest.length
      simp [recomposeNatDigits, Nat.pow_succ, Nat.mul_comm]
      omega

theorem binary_nat_recompose_unique_of_mod_eq_of_lt
    {as bs : List Nat}
    {modulus : Nat}
    (hLen : as.length = bs.length)
    (hAs : binaryNatDigits as)
    (hBs : binaryNatDigits bs)
    (hAsLt : recomposeNatDigits as < modulus)
    (hBsLt : recomposeNatDigits bs < modulus)
    (hMod :
      recomposeNatDigits as % modulus =
      recomposeNatDigits bs % modulus) :
    as = bs := by
  have hRec : recomposeNatDigits as = recomposeNatDigits bs := by
    simpa [Nat.mod_eq_of_lt hAsLt, Nat.mod_eq_of_lt hBsLt] using hMod
  exact binary_nat_recompose_unique hLen hAs hBs hRec

/-! ## Coefficient-column uniqueness -/

/--
Column-oriented digit table.

`cols j` is the least-significant-first digit list for coefficient column `j`.
This avoids hiding the per-coefficient nature of DEC recomposition behind row
packing.
-/
abbrev ColumnDigits (n : Nat) :=
  Fin n → List Nat

def binaryColumnDigits {n : Nat} (cols : ColumnDigits n) : Prop :=
  ∀ j, binaryNatDigits (cols j)

def sameColumnLengths {n : Nat} (a b : ColumnDigits n) : Prop :=
  ∀ j, (a j).length = (b j).length

def recomposeColumns {n : Nat} (cols : ColumnDigits n) : Fin n → Nat :=
  fun j => recomposeNatDigits (cols j)

theorem binary_column_recompose_unique
    {n : Nat}
    {a b : ColumnDigits n}
    (hLen : sameColumnLengths a b)
    (hA : binaryColumnDigits a)
    (hB : binaryColumnDigits b)
    (hRec : recomposeColumns a = recomposeColumns b) :
    a = b := by
  funext j
  have hRecJ :
      recomposeNatDigits (a j) = recomposeNatDigits (b j) := by
    simpa [recomposeColumns] using congrFun hRec j
  exact binary_nat_recompose_unique (hLen j) (hA j) (hB j) hRecJ

/--
If two same-shape binary digit tables both recompose to the same parent
coefficient vector, then the tables are equal.
-/
theorem binary_column_authorization_unique
    {n : Nat}
    {parent : Fin n → Nat}
    {a b : ColumnDigits n}
    (hLen : sameColumnLengths a b)
    (hA : binaryColumnDigits a)
    (hB : binaryColumnDigits b)
    (hRecA : recomposeColumns a = parent)
    (hRecB : recomposeColumns b = parent) :
    a = b := by
  apply binary_column_recompose_unique hLen hA hB
  exact hRecA.trans hRecB.symm

theorem no_different_binary_column_authorization
    {n : Nat}
    {parent : Fin n → Nat}
    {a b : ColumnDigits n}
    (hLen : sameColumnLengths a b)
    (hA : binaryColumnDigits a)
    (hB : binaryColumnDigits b)
    (hDiff : a ≠ b) :
    ¬ (recomposeColumns a = parent ∧ recomposeColumns b = parent) := by
  intro hRec
  exact hDiff
    (binary_column_authorization_unique hLen hA hB hRec.1 hRec.2)

theorem binary_column_recompose_unique_of_mod_eq_of_lt
    {n : Nat}
    {a b : ColumnDigits n}
    {modulus : Nat}
    (hLen : sameColumnLengths a b)
    (hA : binaryColumnDigits a)
    (hB : binaryColumnDigits b)
    (hALt : ∀ j, recomposeNatDigits (a j) < modulus)
    (hBLt : ∀ j, recomposeNatDigits (b j) < modulus)
    (hMod :
      ∀ j,
        recomposeNatDigits (a j) % modulus =
        recomposeNatDigits (b j) % modulus) :
    a = b := by
  funext j
  exact
    binary_nat_recompose_unique_of_mod_eq_of_lt
      (hLen j)
      (hA j)
      (hB j)
      (hALt j)
      (hBLt j)
      (hMod j)

end DecDigitUniqueness

end DirectCcsFPrime
