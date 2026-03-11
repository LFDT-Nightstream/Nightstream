import SuperNeo.Goldilocks
import Mathlib.NumberTheory.LucasPrimality

namespace SuperNeo

namespace Goldilocks

/--
`q - 1` prime-factor decomposition used by the Lucas primality witness:
`q - 1 = 2^32 * 3 * 5 * 17 * 257 * 65537`.
-/
private theorem q_sub_one_factorization :
    q - 1 = (2 ^ 32) * (3 * 5 * 17 * 257 * 65537) := by
  native_decide

private theorem prime_dvd_q_sub_one_cases
    {p : Nat}
    (hp : p.Prime)
    (hpd : p ∣ q - 1) :
    p = 2 ∨ p = 3 ∨ p = 5 ∨ p = 17 ∨ p = 257 ∨ p = 65537 := by
  have hpd' : p ∣ (2 ^ 32) * (3 * 5 * 17 * 257 * 65537) := by
    simpa [q_sub_one_factorization] using hpd
  rcases hp.dvd_mul.mp hpd' with hPow | hRest
  · have hTwo : p ∣ 2 := hp.dvd_of_dvd_pow hPow
    exact Or.inl ((Nat.prime_dvd_prime_iff_eq hp Nat.prime_two).1 hTwo)
  · have hRest' : p ∣ 3 * (5 * 17 * 257 * 65537) := by
      simpa [Nat.mul_assoc] using hRest
    rcases hp.dvd_mul.mp hRest' with hThree | hRestTail
    · exact Or.inr (Or.inl ((Nat.prime_dvd_prime_iff_eq hp (by native_decide : Nat.Prime 3)).1 hThree))
    · have hTail' : p ∣ 5 * (17 * 257 * 65537) := by
        simpa [Nat.mul_assoc] using hRestTail
      rcases hp.dvd_mul.mp hTail' with hFive | hRestTail2
      · exact Or.inr (Or.inr (Or.inl
          ((Nat.prime_dvd_prime_iff_eq hp (by native_decide : Nat.Prime 5)).1 hFive)))
      · have hTail2' : p ∣ 17 * (257 * 65537) := by
          simpa [Nat.mul_assoc] using hRestTail2
        rcases hp.dvd_mul.mp hTail2' with hSeventeen | hRestTail3
        · exact Or.inr (Or.inr (Or.inr (Or.inl
            ((Nat.prime_dvd_prime_iff_eq hp (by native_decide : Nat.Prime 17)).1 hSeventeen))))
        · have hTail3' : p ∣ 257 * 65537 := by
            simpa [Nat.mul_assoc] using hRestTail3
          rcases hp.dvd_mul.mp hTail3' with h257 | h65537
          · exact Or.inr (Or.inr (Or.inr (Or.inr (Or.inl
              ((Nat.prime_dvd_prime_iff_eq hp (by native_decide : Nat.Prime 257)).1 h257)))))
          · exact Or.inr (Or.inr (Or.inr (Or.inr (Or.inr
              ((Nat.prime_dvd_prime_iff_eq hp (by native_decide : Nat.Prime 65537)).1 h65537)))))

/-- Constructive primality theorem for the Goldilocks modulus. -/
theorem q_prime : Nat.Prime q := by
  let a : ZMod q := 7
  refine lucas_primality q a ?_ ?_
  · native_decide
  · intro p hp hpd
    rcases prime_dvd_q_sub_one_cases hp hpd with rfl | rfl | rfl | rfl | rfl | rfl
    · native_decide
    · native_decide
    · native_decide
    · native_decide
    · native_decide
    · native_decide

instance fact_q_prime : Fact (Nat.Prime q) := ⟨q_prime⟩

end Goldilocks

end SuperNeo
