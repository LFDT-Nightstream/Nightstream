import CompressionLean.Lib.Roots
import CompressionLean.Lib.Cube
import Mathlib.LinearAlgebra.Lagrange

namespace CompressionLean.Lib

open Polynomial

variable {K : Type} [Field K]

def node (d : ℕ) (i : Fin (d + 2)) : K := ((i : ℕ) : K)

def roundValues (d : ℕ) (claim : K) (message : Fin (d + 1) → K) (i : Fin (d + 2)) : K :=
  if hi : (i : ℕ) = 0 then message 0
  else if (i : ℕ) = 1 then claim - message 0
  else message ⟨(i : ℕ) - 1, by omega⟩

noncomputable def roundPoly (d : ℕ) (claim : K) (message : Fin (d + 1) → K) : K[X] :=
  Lagrange.interpolate Finset.univ (node d) (roundValues d claim message)

def NodesDistinct (K : Type) [Field K] (d : ℕ) : Prop :=
  Function.Injective (node (K := K) d)

section RoundPoly

variable {d : ℕ}

theorem roundPoly_eval_node (distinct : NodesDistinct K d) (claim : K) (message : Fin (d + 1) → K)
    (i : Fin (d + 2)) : (roundPoly d claim message).eval (node d i) = roundValues d claim message i :=
  Lagrange.eval_interpolate_at_node _ (distinct.injOn) (Finset.mem_univ i)

theorem roundPoly_sum (distinct : NodesDistinct K d) (claim : K) (message : Fin (d + 1) → K) :
    (roundPoly d claim message).eval 0 + (roundPoly d claim message).eval 1 = claim := by
  have zero := roundPoly_eval_node distinct claim message 0
  have one := roundPoly_eval_node distinct claim message 1
  simp only [node, Fin.val_zero, Nat.cast_zero, roundValues, ↓reduceDIte] at zero
  simp only [node, Fin.val_one, Nat.cast_one, roundValues, show ((1 : ℕ) = 0) = False by simp,
    dite_eq_right, ite_eq_left, not_false_eq_true] at one
  rw [zero, one]
  ring

theorem roundPoly_natDegree (distinct : NodesDistinct K d) (claim : K) (message : Fin (d + 1) → K) :
    (roundPoly d claim message).natDegree ≤ d + 1 := by
  have lt : (roundPoly d claim message).degree < (Finset.univ : Finset (Fin (d + 2))).card :=
    Lagrange.degree_interpolate_lt (roundValues d claim message) distinct.injOn
  rw [Finset.card_univ, Fintype.card_fin] at lt
  by_cases zero : roundPoly d claim message = 0
  · rw [zero, natDegree_zero]
    omega
  · rw [degree_eq_natDegree zero] at lt
    exact Nat.lt_succ_iff.mp (by exact_mod_cast lt)

end RoundPoly

variable [Fintype K] [DecidableEq K]

def Path (extra : List K → Finset K) : List K → List K → Prop
  | _, [] => True
  | point, c :: rest => c ∉ extra point ∧ Path extra (point ++ [c]) rest

structure RoundTruth (d : ℕ) (bound : ℕ) (truth : List K → K) (round : List K → K[X]) : Prop where
  degree : ∀ point : List K, point.length < bound → (round point).natDegree ≤ d + 1
  halves : ∀ point : List K, point.length < bound → (round point).eval 0 + (round point).eval 1 = truth point
  binds : ∀ point : List K, ∀ c : K, point.length < bound → (round point).eval c = truth (point ++ [c])

end CompressionLean.Lib
