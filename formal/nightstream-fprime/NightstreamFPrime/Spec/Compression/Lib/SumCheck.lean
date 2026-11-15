import NightstreamFPrime.Spec.Compression.Lib.Roots
import NightstreamFPrime.Spec.Compression.Lib.Cube
import Mathlib.Algebra.Polynomial.BigOperators

/-! The facts of one sum-check that its game needs: the polynomial of a round's coefficients
(Rust `Round` and `KeyRound`, `crates/ns-compress/src/sumcheck.rs`), the round check, and the
true round polynomials of a sum (`RoundTruth`). -/

namespace NightstreamFPrime.Spec.Compression.Lib

open Polynomial

variable {K : Type} [Field K]

/-- The polynomial with the coefficients `message`, from degree 0 up. -/
noncomputable def coeffPoly {m : ℕ} (message : Fin m → K) : K[X] :=
  ∑ i : Fin m, C (message i) * X ^ (i : ℕ)

theorem coeffPoly_natDegree {m : ℕ} (message : Fin (m + 1) → K) :
    (coeffPoly message).natDegree ≤ m := by
  unfold coeffPoly
  refine natDegree_sum_le_of_forall_le _ _ fun i _ => ?_
  exact (natDegree_C_mul_X_pow_le _ _).trans (Nat.lt_succ_iff.mp i.isLt)

theorem coeffPoly_eval {m : ℕ} (message : Fin m → K) (c : K) :
    (coeffPoly message).eval c = ∑ i : Fin m, message i * c ^ (i : ℕ) := by
  simp [coeffPoly, eval_finsetSum]

/-- A polynomial of degree at most `m` is the polynomial of its first `m + 1` coefficients. -/
theorem coeffPoly_coeff {m : ℕ} (p : K[X]) (degree : p.natDegree ≤ m) :
    coeffPoly (fun i : Fin (m + 1) => p.coeff i) = p := by
  unfold coeffPoly
  rw [Fin.sum_univ_eq_sum_range (fun i => C (p.coeff i) * X ^ i) (m + 1)]
  exact (p.as_sum_range_C_mul_X_pow' (Nat.lt_succ_of_le degree)).symm

/-- The round check of Rust `sumcheck::verify`: `c_0 + Σ_i c_i` is the value at 0 plus the value
at 1. -/
theorem coeffPoly_check {m : ℕ} (message : Fin (m + 1) → K) :
    (coeffPoly message).eval 0 + (coeffPoly message).eval 1 = message 0 + ∑ i, message i := by
  rw [coeffPoly_eval, coeffPoly_eval, Fin.sum_univ_succ]
  simp

/-- `round` gives the true round polynomials of the sum `truth`, of degree at most `d`, for the
first `bound` rounds. -/
structure RoundTruth (d : ℕ) (bound : ℕ) (truth : List K → K) (round : List K → K[X]) : Prop where
  degree : ∀ point : List K, point.length < bound → (round point).natDegree ≤ d
  halves : ∀ point : List K, point.length < bound → (round point).eval 0 + (round point).eval 1 = truth point
  binds : ∀ point : List K, ∀ c : K, point.length < bound → (round point).eval c = truth (point ++ [c])

end NightstreamFPrime.Spec.Compression.Lib
