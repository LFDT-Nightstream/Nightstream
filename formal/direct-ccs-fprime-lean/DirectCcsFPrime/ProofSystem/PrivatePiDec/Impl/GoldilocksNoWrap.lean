import DirectCcsFPrime.Audit.Counterexamples.DecDigitUniqueness
import SuperNeo.Primitives.Goldilocks

/-!
Goldilocks no-wrap facts for fixed-length binary DEC authorization.

This module connects the abstract no-wrap theorem in `DecDigitUniqueness` to
the concrete SuperNeo Goldilocks parameter used by the direct CCS/F' model:
`k_dec = 14`.
-/

namespace DirectCcsFPrime

namespace GoldilocksNoWrap

open DecDigitUniqueness

/-- The SuperNeo Goldilocks modulus is larger than `2^14`. -/
theorem two_pow_14_lt_goldilocks_q :
    2 ^ 14 < SuperNeo.Goldilocks.q := by
  native_decide

/--
A binary digit list of exact length `14` recomposes below the Goldilocks
modulus. This discharges the no-wrap side condition for a single coefficient.
-/
theorem binary_length14_recompose_lt_goldilocks_q
    (digits : List Nat)
    (hBin : binaryNatDigits digits)
    (hLen : digits.length = 14) :
    recomposeNatDigits digits < SuperNeo.Goldilocks.q := by
  have hPow : recomposeNatDigits digits < 2 ^ digits.length :=
    recomposeNatDigits_lt_two_pow_length digits hBin
  have hPow14 : 2 ^ digits.length = 2 ^ 14 := by
    rw [hLen]
  exact Nat.lt_trans (by simpa [hPow14] using hPow) two_pow_14_lt_goldilocks_q

/--
Column-wise version: every binary digit column of exact length `14` is below
the Goldilocks modulus after recomposition.
-/
theorem binary_column_length14_recompose_lt_goldilocks_q
    {n : Nat}
    (cols : ColumnDigits n)
    (hBin : binaryColumnDigits cols)
    (hLen : ∀ j, (cols j).length = 14) :
    ∀ j, recomposeNatDigits (cols j) < SuperNeo.Goldilocks.q := by
  intro j
  exact binary_length14_recompose_lt_goldilocks_q (cols j) (hBin j) (hLen j)

/--
If two same-shape binary child tables of exact length `14` are equal only after
Goldilocks modular recomposition, they are still the same table. The length and
binary-digit checks give no-wrap, so modular equality lifts to integer equality.
-/
theorem binary_column_length14_unique_of_goldilocks_mod_eq
    {n : Nat}
    {a b : ColumnDigits n}
    (hLenA : ∀ j, (a j).length = 14)
    (hLenB : ∀ j, (b j).length = 14)
    (hA : binaryColumnDigits a)
    (hB : binaryColumnDigits b)
    (hMod :
      ∀ j,
        recomposeNatDigits (a j) % SuperNeo.Goldilocks.q =
        recomposeNatDigits (b j) % SuperNeo.Goldilocks.q) :
    a = b := by
  apply binary_column_recompose_unique_of_mod_eq_of_lt
  · intro j
    exact (hLenA j).trans (hLenB j).symm
  · exact hA
  · exact hB
  · exact binary_column_length14_recompose_lt_goldilocks_q a hA hLenA
  · exact binary_column_length14_recompose_lt_goldilocks_q b hB hLenB
  · exact hMod

end GoldilocksNoWrap

end DirectCcsFPrime
