import SuperNeo.Decomp

namespace SuperNeo

/--
Public-entry form of base-2 recomposition on coefficient rows:
when at least one digit row is present, recomposing the `j`-th column of
`splitBase2Coeffs z k` returns the canonical low part of `z[j]`.
-/
theorem recomposeBase2Coeffs_splitBase2Coeffs_entry
    (z : Coeffs) (k : Nat) (hk : 0 < k) (j : Fin z.size) :
    (recomposeBase2Coeffs (splitBase2Coeffs z k))[j.1]! =
      F.ofNat (splitBase2LowPartNat z[j.1] k) := by
  have hkNe : k ≠ 0 := Nat.ne_of_gt hk
  have hRow0Size : ((splitBase2Coeffs z k)[0]!).size = z.size := by
    exact splitBase2Coeffs_row_size z k ⟨0, hk⟩
  have hExpr :
      (recomposeBase2Coeffs (splitBase2Coeffs z k))[j.1]! =
        recomposeBase2Scalar ((splitBase2Coeffs z k).map (fun row => row[j.1]!)) := by
    unfold recomposeBase2Coeffs
    simp [splitBase2Coeffs_size, hkNe, hRow0Size, j.2]
  calc
    (recomposeBase2Coeffs (splitBase2Coeffs z k))[j.1]!
        = recomposeBase2Scalar ((splitBase2Coeffs z k).map (fun row => row[j.1]!)) := hExpr
    _ = F.ofNat (recomposeBase2ScalarNat ((splitBase2Coeffs z k).map (fun row => row[j.1]!))) := by
          rw [recomposeBase2Scalar_eq_ofNat]
    _ = F.ofNat (splitBase2LowPartNat z[j.1] k) := by
          rw [recomposeBase2Coeffs_entry_eq_lowPartNat z k j]

/--
If every coefficient has zero terminal quotient after consuming `k` base-2 digits,
then recomposing `splitBase2Coeffs z k` recovers `z`.

The `0 < k` guard is required because `recomposeBase2Coeffs` returns `#[]` on an
empty row array.
-/
theorem recomposeBase2Coeffs_splitBase2Coeffs_eq_of_terminal_zero
    (z : Coeffs) (k : Nat) (hk : 0 < k)
    (hTerm : ∀ j : Fin z.size, splitBase2TerminalZeroProp z[j.1] k) :
    recomposeBase2Coeffs (splitBase2Coeffs z k) = z := by
  apply Array.ext
  · have hkNe : k ≠ 0 := Nat.ne_of_gt hk
    have hRow0Size : ((splitBase2Coeffs z k)[0]!).size = z.size := by
      exact splitBase2Coeffs_row_size z k ⟨0, hk⟩
    unfold recomposeBase2Coeffs
    simp [splitBase2Coeffs_size, hkNe, hRow0Size]
  · intro i hi1 hi2
    let j : Fin z.size := ⟨i, hi2⟩
    have hLow : splitBase2LowPartNat z[j.1] k = z[j.1].val :=
      splitBase2LowPart_eq_val_of_terminal_zero z[j.1] k (hTerm j)
    have hMain :
        (recomposeBase2Coeffs (splitBase2Coeffs z k))[j.1]! = z[j.1]! := by
      calc
        (recomposeBase2Coeffs (splitBase2Coeffs z k))[j.1]!
            = F.ofNat (splitBase2LowPartNat z[j.1] k) :=
              recomposeBase2Coeffs_splitBase2Coeffs_entry z k hk j
        _ = F.ofNat z[j.1].val := by rw [hLow]
        _ = z[j.1]! := by simpa using (F.ofNat_val z[j.1]!)
    simpa [j, hi1, hi2] using hMain

/--
Range-bounded base-2 recomposition corollary:
if every canonical coefficient representative is below `2^k`, then the public
row-wise split/recompose round-trip is exact.
-/
theorem recomposeBase2Coeffs_splitBase2Coeffs_eq_of_val_lt_pow
    (z : Coeffs) (k : Nat) (hk : 0 < k)
    (hLt : ∀ j : Fin z.size, z[j.1].val < 2 ^ k) :
    recomposeBase2Coeffs (splitBase2Coeffs z k) = z := by
  refine recomposeBase2Coeffs_splitBase2Coeffs_eq_of_terminal_zero z k hk ?_
  intro j
  exact splitBase2TerminalZeroProp_of_val_lt_pow z[j.1] k (hLt j)

end SuperNeo
