import SuperNeo.GoldilocksPrime

/-!
Contract interface for `SuperNeo.GoldilocksPrime`.

Spec: `specs/GoldilocksPrime.spec.md`

Paper anchors:
- Definition 1 (Fields, Rings, Dimensions), Section 4, lines 275-282: field modulus `q` is prime.
- Appendix B.2, lines 709-727: concrete Goldilocks modulus.
-/

namespace SuperNeo

namespace GoldilocksPrimeInterface

/-! ## Core Surfaces -/

/-- [Role: Theorem-Target] Curated theorem surface `Goldilocks.q_prime`. -/
abbrev Goldilocks_q_prime := SuperNeo.Goldilocks.q_prime

/-- [Role: Theorem-Target] Curated typeclass witness `Fact (Nat.Prime Goldilocks.q)`. -/
theorem Goldilocks_fact_q_prime : Fact (Nat.Prime SuperNeo.Goldilocks.q) := by
  infer_instance

end GoldilocksPrimeInterface

end SuperNeo
