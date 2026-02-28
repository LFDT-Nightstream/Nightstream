import SuperNeo.Decomp

/-! Theorem-native arithmetic bridges for P6 decomposition. -/

namespace SuperNeo

private theorem F_ext {a b : F} (h : a.val = b.val) : a = b := by
  cases a
  cases b
  cases h
  rfl

private theorem q_int_ne_zero : (q : Int) ≠ 0 := by
  exact_mod_cast (Nat.ne_of_gt q_pos)

private theorem intEmod_toNat_lt_q (z : Int) : ((z % (q : Int)).toNat) < q := by
  have hNonneg : 0 ≤ z % (q : Int) := Int.emod_nonneg z q_int_ne_zero
  have hLtInt : z % (q : Int) < (q : Int) := by
    have hLtAbs : z % (q : Int) < ((q : Int).natAbs : Int) := Int.emod_lt z q_int_ne_zero
    simpa using hLtAbs
  exact (Int.toNat_lt hNonneg).2 hLtInt

private theorem intEmod_toNat_mod_eq (z : Int) :
    ((z % (q : Int)).toNat % q) = ((z % (q : Int)).toNat) := by
  exact Nat.mod_eq_of_lt (intEmod_toNat_lt_q z)

private theorem intNatCast_emod_toNat_eq_mod (n : Nat) :
    (((n : Int) % (q : Int)).toNat) = n % q := by
  have h : ((n : Int) % (q : Int)) = ((n % q : Nat) : Int) := by
    simp
  rw [h]
  simpa using (Int.toNat_natCast (n % q))

private theorem intEmod_toNat_add_mod (x y : Int) :
    (((x % (q : Int) + y % (q : Int)) % (q : Int)).toNat) =
      (((x % (q : Int)).toNat + (y % (q : Int)).toNat) % q) := by
  let ax : Nat := (x % (q : Int)).toNat
  let ay : Nat := (y % (q : Int)).toNat
  have hxNonneg : 0 ≤ x % (q : Int) := Int.emod_nonneg x q_int_ne_zero
  have hyNonneg : 0 ≤ y % (q : Int) := Int.emod_nonneg y q_int_ne_zero
  have hxEq : (x % (q : Int)) = (ax : Int) := by
    simpa [ax] using (Int.toNat_of_nonneg hxNonneg).symm
  have hyEq : (y % (q : Int)) = (ay : Int) := by
    simpa [ay] using (Int.toNat_of_nonneg hyNonneg).symm
  rw [hxEq, hyEq]
  have hAddCast : ((ax : Int) + (ay : Int)) = ((ax + ay : Nat) : Int) := by
    simp
  rw [hAddCast]
  simpa using intNatCast_emod_toNat_eq_mod (ax + ay)

private theorem intEmod_toNat_mul_mod (x y : Int) :
    ((((x % (q : Int)) * (y % (q : Int))) % (q : Int)).toNat) =
      (((x % (q : Int)).toNat * (y % (q : Int)).toNat) % q) := by
  let ax : Nat := (x % (q : Int)).toNat
  let ay : Nat := (y % (q : Int)).toNat
  have hxNonneg : 0 ≤ x % (q : Int) := Int.emod_nonneg x q_int_ne_zero
  have hyNonneg : 0 ≤ y % (q : Int) := Int.emod_nonneg y q_int_ne_zero
  have hxEq : (x % (q : Int)) = (ax : Int) := by
    simpa [ax] using (Int.toNat_of_nonneg hxNonneg).symm
  have hyEq : (y % (q : Int)) = (ay : Int) := by
    simpa [ay] using (Int.toNat_of_nonneg hyNonneg).symm
  rw [hxEq, hyEq]
  have hMulCast : ((ax : Int) * (ay : Int)) = ((ax * ay : Nat) : Int) := by
    simp
  rw [hMulCast]
  simpa using intNatCast_emod_toNat_eq_mod (ax * ay)

theorem p6OfIntSemiringAssumption_theorem : p6OfIntSemiringAssumption := by
  refine ⟨?_, ?_, ?_⟩
  · intro x y
    apply F_ext
    simp [F.ofInt, F.ofNat]
    change ((x + y) % (q : Int)).toNat % q = ((x % (q : Int)).toNat % q + (y % (q : Int)).toNat % q) % q
    rw [intEmod_toNat_mod_eq (z := x + y)]
    rw [intEmod_toNat_mod_eq (z := x)]
    rw [intEmod_toNat_mod_eq (z := y)]
    have hxy : ((x + y) % (q : Int)) = ((x % (q : Int) + y % (q : Int)) % (q : Int)) := by
      simpa using (Int.add_emod x y (q : Int))
    rw [hxy]
    exact intEmod_toNat_add_mod x y
  · intro x y
    apply F_ext
    simp [F.ofInt, F.ofNat]
    have hMulVal :
        ((({ val := (x % (q : Int)).toNat % q } : F) *
          ({ val := (y % (q : Int)).toNat % q } : F)).val) =
          ((x % (q : Int)).toNat % q * ((y % (q : Int)).toNat % q)) % q := by
      rfl
    rw [hMulVal]
    rw [intEmod_toNat_mod_eq (z := x * y)]
    rw [intEmod_toNat_mod_eq (z := x)]
    rw [intEmod_toNat_mod_eq (z := y)]
    have hxy : ((x * y) % (q : Int)) = (((x % (q : Int)) * (y % (q : Int))) % (q : Int)) := by
      simpa [Int.mul_assoc] using (Int.mul_emod x y (q : Int))
    rw [hxy]
    exact intEmod_toNat_mul_mod x y
  · intro n
    apply F_ext
    simp [F.ofInt, F.ofNat]
    rw [intEmod_toNat_mod_eq (z := (n : Int))]
    exact intNatCast_emod_toNat_eq_mod n

private theorem ofInt_nat (n : Nat) : F.ofInt (Int.ofNat n) = F.ofNat n :=
  (p6OfIntSemiringAssumption_theorem.2.2 n)

private theorem ofInt_sub_q_nat (n : Nat) :
    F.ofInt (Int.ofNat n - Int.ofNat q) = F.ofNat n := by
  apply F_ext
  simp [F.ofInt, F.ofNat]
  rw [intEmod_toNat_mod_eq (z := (n : Int))]
  exact intNatCast_emod_toNat_eq_mod n

theorem ofInt_centeredInt_eq_of_canonical
  {x : F}
  (hx : F.Canonical x) :
  F.ofInt (centeredInt x) = x := by
  cases x with
  | mk v =>
      unfold F.Canonical at hx
      unfold centeredInt
      by_cases hHalf : v ≤ halfQ
      · rw [if_pos hHalf]
        calc
          F.ofInt (Int.ofNat v) = F.ofNat v := ofInt_nat v
          _ = F.mk v := by
                simpa using (F.ofNat_val_eq_of_canonical (a := F.mk v) hx)
      · rw [if_neg hHalf]
        calc
          F.ofInt (Int.ofNat v - Int.ofNat q) = F.ofNat v := ofInt_sub_q_nat v
          _ = F.mk v := by
                simpa using (F.ofNat_val_eq_of_canonical (a := F.mk v) hx)

/-- Canonical-input version of the P6 reconstruction endpoint. -/
theorem recomposeSplitDigits_splitBalancedVec_eq_of_base_ge_two_of_state_zero_of_allCanonical
  {z : Array F} {b k : Nat}
  (hb : 2 ≤ b)
  (hk : 0 < k)
  (hZero : splitScalarTerminalZeroProp z b k)
  (hCanon : z.all (fun x => decide (F.Canonical x)) = true) :
  recomposeSplitDigits (splitBalancedVec z b k) b = z := by
  have hOfInt : p6OfIntSemiringAssumption := p6OfIntSemiringAssumption_theorem
  have hNE : (splitBalancedVec z b k).isEmpty = false :=
    splitBalancedVec_isEmpty_eq_false_of_k_pos (z := z) (b := b) (k := k) hk
  have h0 : 0 < (splitBalancedVec z b k).size := by
    simpa [splitBalancedVec_size (z := z) (b := b) (k := k)] using hk
  have hRow0 : ((splitBalancedVec z b k)[0]!).size = z.size := by
    simpa [h0] using
      (splitBalancedVec_row_size (z := z) (b := b) (k := k) (i := 0) h0)
  apply Array.ext
  · simpa [recomposeSplitDigits, hNE, hRow0]
  · intro j hjL hjR
    have hEntryOfInt :
        (recomposeSplitDigits (splitBalancedVec z b k) b)[j]'hjL = F.ofInt (centeredInt z[j]!) :=
      recomposeSplitDigits_splitBalancedVec_get_eq_ofInt_centeredInt_of_base_ge_two_of_state_zero
        (z := z) (b := b) (k := k) (j := j) hb hk hjR (hZero j hjR) hOfInt
    have hCanonJ : F.Canonical (z[j]!) :=
      F.canonical_getElem!_of_all hCanon j
    have hCenteredJ : F.ofInt (centeredInt z[j]!) = z[j]! :=
      ofInt_centeredInt_eq_of_canonical hCanonJ
    have hEntryBang :
        (recomposeSplitDigits (splitBalancedVec z b k) b)[j]'hjL
          = z[j]! := by
      calc
        (recomposeSplitDigits (splitBalancedVec z b k) b)[j]'hjL
            = F.ofInt (centeredInt z[j]!) := hEntryOfInt
        _ = z[j]! := hCenteredJ
    simpa [hjR] using hEntryBang

/-- Base-2 native `splitRoundTrip` closure from terminal-state-zero + canonicality. -/
theorem splitRoundTrip_true_of_base_two_of_state_zero_of_allCanonical
  {z : Array F} {k : Nat}
  (hk : 0 < k)
  (hZero : splitScalarTerminalZeroProp z 2 k)
  (hCanon : z.all (fun x => decide (F.Canonical x)) = true) :
  splitRoundTrip z 2 k = true := by
  apply splitRoundTrip_complete_prop
  refine ⟨by decide, ?_⟩
  refine ⟨?_, ?_⟩
  · exact recomposeSplitDigits_splitBalancedVec_eq_of_base_ge_two_of_state_zero_of_allCanonical
      (z := z) (b := 2) (k := k) (hb := by decide) hk hZero hCanon
  · exact splitBalancedVec_digitsWithinBaseProp_of_base_two (z := z) (k := k)

theorem splitScalarTerminalZeroProp_of_centeredInt_eq_splitScalarResidueFoldInt
  {z : Array F} {b k : Nat}
  (hb : 2 ≤ b)
  (hEq : ∀ j (_hj : j < z.size), centeredInt z[j]! = splitScalarResidueFoldInt z[j]! b k) :
  splitScalarTerminalZeroProp z b k := by
  intro j hj
  exact splitScalarState_fst_zero_of_centeredInt_eq_splitScalarResidueFoldInt
    (a := z[j]!) hb (hEq j hj)

theorem splitScalarTerminalZeroProp_iff_centeredInt_eq_splitScalarResidueFoldInt
  {z : Array F} {b k : Nat}
  (hb : 2 ≤ b) :
  splitScalarTerminalZeroProp z b k ↔
    (∀ j (_hj : j < z.size), centeredInt z[j]! = splitScalarResidueFoldInt z[j]! b k) := by
  constructor
  · intro hZero j hj
    exact centeredInt_eq_splitScalarResidueFoldInt_of_splitScalarState_fst_zero
      (a := z[j]!) hb (hZero j hj)
  · intro hEq
    exact splitScalarTerminalZeroProp_of_centeredInt_eq_splitScalarResidueFoldInt
      (z := z) (b := b) (k := k) hb hEq

theorem splitScalarTerminalZeroProp_iff_centeredInt_eq_splitScalarResidueFoldInt_base2
  {z : Array F} {k : Nat} :
  splitScalarTerminalZeroProp z 2 k ↔
    (∀ j (_hj : j < z.size), centeredInt z[j]! = splitScalarResidueFoldInt z[j]! 2 k) := by
  simpa using
    (splitScalarTerminalZeroProp_iff_centeredInt_eq_splitScalarResidueFoldInt
      (z := z) (b := 2) (k := k) (by decide : 2 ≤ 2))

theorem splitScalarTerminalZeroProp_of_residue_fold_eq_centeredInt_base2
  {z : Array F} {k : Nat}
  (hEq : ∀ j (_hj : j < z.size), centeredInt z[j]! = splitScalarResidueFoldInt z[j]! 2 k) :
  splitScalarTerminalZeroProp z 2 k := by
  exact (splitScalarTerminalZeroProp_iff_centeredInt_eq_splitScalarResidueFoldInt_base2
    (z := z) (k := k)).2 hEq

theorem recomposeSplitDigits_splitBalancedVec_eq_of_base_ge_two_of_residue_fold_eq_centeredInt_of_allCanonical
  {z : Array F} {b k : Nat}
  (hb : 2 ≤ b)
  (hk : 0 < k)
  (hEq : ∀ j (_hj : j < z.size), centeredInt z[j]! = splitScalarResidueFoldInt z[j]! b k)
  (hCanon : z.all (fun x => decide (F.Canonical x)) = true) :
  recomposeSplitDigits (splitBalancedVec z b k) b = z := by
  exact recomposeSplitDigits_splitBalancedVec_eq_of_base_ge_two_of_state_zero_of_allCanonical
    (z := z) (b := b) (k := k) hb hk
    (splitScalarTerminalZeroProp_of_centeredInt_eq_splitScalarResidueFoldInt
      (z := z) (b := b) (k := k) hb hEq)
    hCanon

theorem splitRoundTrip_true_of_base_two_of_residue_fold_eq_centeredInt_of_allCanonical
  {z : Array F} {k : Nat}
  (hk : 0 < k)
  (hEq : ∀ j (_hj : j < z.size), centeredInt z[j]! = splitScalarResidueFoldInt z[j]! 2 k)
  (hCanon : z.all (fun x => decide (F.Canonical x)) = true) :
  splitRoundTrip z 2 k = true := by
  exact splitRoundTrip_true_of_base_two_of_state_zero_of_allCanonical
    (z := z) (k := k) hk
    (splitScalarTerminalZeroProp_of_centeredInt_eq_splitScalarResidueFoldInt
      (z := z) (b := 2) (k := k) (hb := by decide) hEq)
    hCanon

/-- Challenge coefficients are canonical in the current `F` model. -/
theorem canonical_of_isChallengeCoeff
  {x : F}
  (hx : IsChallengeCoeff x) :
  F.Canonical x := by
  rcases hx with hNeg2 | hNeg1 | h0 | h1 | h2
  · subst hNeg2
    native_decide
  · subst hNeg1
    native_decide
  · subst h0
    native_decide
  · subst h1
    native_decide
  · subst h2
    native_decide

/--
Concrete base-2/k=8 residue-fold closure for challenge coefficients.
-/
theorem centeredInt_eq_splitScalarResidueFoldInt_of_isChallengeCoeff_base2_k8
  {x : F}
  (hx : IsChallengeCoeff x) :
  centeredInt x = splitScalarResidueFoldInt x 2 8 := by
  rcases hx with hNeg2 | hNeg1 | h0 | h1 | h2
  · subst hNeg2
    native_decide
  · subst hNeg1
    native_decide
  · subst h0
    native_decide
  · subst h1
    native_decide
  · subst h2
    native_decide

/--
Array-level challenge-coefficient condition implies base-2/k=8 residue-fold
closure on every entry.
-/
theorem centeredInt_eq_splitScalarResidueFoldInt_base2_k8_of_allChallenge
  {z : Array F}
  (hChallenge : ∀ j (hj : j < z.size), IsChallengeCoeff (z[j]'hj)) :
  ∀ j (_hj : j < z.size), centeredInt z[j]! = splitScalarResidueFoldInt z[j]! 2 8 := by
  intro j hj
  have hEq :
      centeredInt (z[j]'hj) = splitScalarResidueFoldInt (z[j]'hj) 2 8 :=
    centeredInt_eq_splitScalarResidueFoldInt_of_isChallengeCoeff_base2_k8
      (hChallenge j hj)
  simpa [hj] using hEq

/-- Array-level challenge-coefficient condition implies canonicality check pass. -/
theorem allCanonical_of_allChallenge
  {z : Array F}
  (hChallenge : ∀ j (hj : j < z.size), IsChallengeCoeff (z[j]'hj)) :
  z.all (fun x => decide (F.Canonical x)) = true := by
  apply (Array.all_eq_true).2
  intro j hj
  exact decide_eq_true
    (canonical_of_isChallengeCoeff (hChallenge j hj))

/-- Array-level challenge-coefficient condition implies terminal-state-zero for base-2/k=8. -/
theorem splitScalarTerminalZeroProp_of_allChallenge_base2_k8
  {z : Array F}
  (hChallenge : ∀ j (hj : j < z.size), IsChallengeCoeff (z[j]'hj)) :
  splitScalarTerminalZeroProp z 2 8 := by
  exact splitScalarTerminalZeroProp_of_residue_fold_eq_centeredInt_base2
    (z := z) (k := 8)
    (centeredInt_eq_splitScalarResidueFoldInt_base2_k8_of_allChallenge hChallenge)

/--
Concrete native base-2/k=8 P6 endpoint from challenge coefficients.

This is the first assumption-light closure path that avoids explicitly threading
`hZero` and `hCanon`.
-/
theorem splitRoundTrip_true_of_base_two_k8_of_allChallenge
  {z : Array F}
  (hChallenge : ∀ j (hj : j < z.size), IsChallengeCoeff (z[j]'hj)) :
  splitRoundTrip z 2 8 = true := by
  exact splitRoundTrip_true_of_base_two_of_state_zero_of_allCanonical
    (z := z) (k := 8) (by decide)
    (splitScalarTerminalZeroProp_of_allChallenge_base2_k8 hChallenge)
    (allCanonical_of_allChallenge hChallenge)

/--
Scalar challenge-coefficient closure for base-2 decomposition with any
`k ≥ 8`, bootstrapped from the concrete `k = 8` closure.
-/
theorem centeredInt_eq_splitScalarResidueFoldInt_of_isChallengeCoeff_base2_of_ge8
  {x : F} {k : Nat}
  (hk : 8 ≤ k)
  (hx : IsChallengeCoeff x) :
  centeredInt x = splitScalarResidueFoldInt x 2 k := by
  rcases Nat.exists_eq_add_of_le hk with ⟨n, rfl⟩
  have hBase : centeredInt x = splitScalarResidueFoldInt x 2 8 := by
    simpa using centeredInt_eq_splitScalarResidueFoldInt_of_isChallengeCoeff_base2_k8 hx
  have hStep :
      ∀ t : Nat,
        centeredInt x = splitScalarResidueFoldInt x 2 (8 + t) →
          centeredInt x = splitScalarResidueFoldInt x 2 (8 + (t + 1)) := by
    intro t hEq
    simpa [Nat.add_assoc] using
      (centeredInt_eq_splitScalarResidueFoldInt_base2_succ_of_eq (a := x) hEq)
  exact Nat.rec hBase (fun t ih => hStep t ih) n

/--
Array-level challenge-coefficient condition implies base-2 residue-fold closure
for all `k ≥ 8`.
-/
theorem centeredInt_eq_splitScalarResidueFoldInt_base2_of_allChallenge_of_ge8
  {z : Array F} {k : Nat}
  (hk : 8 ≤ k)
  (hChallenge : ∀ j (hj : j < z.size), IsChallengeCoeff (z[j]'hj)) :
  ∀ j (_hj : j < z.size), centeredInt z[j]! = splitScalarResidueFoldInt z[j]! 2 k := by
  intro j hj
  have hEq :
      centeredInt (z[j]'hj) = splitScalarResidueFoldInt (z[j]'hj) 2 k :=
    centeredInt_eq_splitScalarResidueFoldInt_of_isChallengeCoeff_base2_of_ge8
      hk (hChallenge j hj)
  simpa [hj] using hEq

/-- Array-level challenge-coefficient condition implies terminal-state-zero for base-2 and any `k ≥ 8`. -/
theorem splitScalarTerminalZeroProp_of_allChallenge_base2_of_ge8
  {z : Array F} {k : Nat}
  (hk : 8 ≤ k)
  (hChallenge : ∀ j (hj : j < z.size), IsChallengeCoeff (z[j]'hj)) :
  splitScalarTerminalZeroProp z 2 k := by
  exact splitScalarTerminalZeroProp_of_residue_fold_eq_centeredInt_base2
    (z := z) (k := k)
    (centeredInt_eq_splitScalarResidueFoldInt_base2_of_allChallenge_of_ge8 hk hChallenge)

/--
Concrete native base-2 P6 endpoint from challenge coefficients for any
`k ≥ 8`.
-/
theorem splitRoundTrip_true_of_base_two_of_allChallenge_of_ge8
  {z : Array F} {k : Nat}
  (hk : 8 ≤ k)
  (hChallenge : ∀ j (hj : j < z.size), IsChallengeCoeff (z[j]'hj)) :
  splitRoundTrip z 2 k = true := by
  exact splitRoundTrip_true_of_base_two_of_state_zero_of_allCanonical
    (z := z) (k := k) (Nat.lt_of_lt_of_le (by decide : 0 < 8) hk)
    (splitScalarTerminalZeroProp_of_allChallenge_base2_of_ge8 hk hChallenge)
    (allCanonical_of_allChallenge hChallenge)

end SuperNeo
