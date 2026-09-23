import NightstreamFPrime.Layout.PiRlcWideSampler
import NightstreamFPrime.Layout.R1CS

/-! Direct rank-one rows for the wide sampler. Affine zero checks and
products of affine forms need no multiplication scratch. The canonical
high-word flag uses its existing equation with the product isolated.
Unsupported rows fail closed. No production compiler is changed here. -/

namespace NightstreamFPrime.Layout.PiRlcWideSampler.Rows

open NightstreamFPrime.Spec NightstreamFPrime.Circuit
open NightstreamFPrime.Gadgets.Sampling.WideReduction
open NightstreamFPrime.Gadgets.Range
open R1CS

def product (left right : Expr) (a : AffineResult left) (b : AffineResult right) :
    DirectConstraintResult (left * right) where
  row := ⟨a.combination, b.combination, LinearCombination.zero⟩
  sound := by
    intro env rows
    change a.combination.eval env * b.combination.eval env = 0 at rows
    simpa only [a.sound env, b.sound env, Expr.eval_hmul] using rows
  complete := by
    intro env rows
    change a.combination.eval env * b.combination.eval env = 0
    simpa only [a.sound env, b.sound env, Expr.eval_hmul] using rows

def flagExpression (offset : Nat) : Expr :=
  Expr.var (offset + CanonicalU64.bitCount + 1) - CanonicalU64.flagRecipe offset

def flag (offset : Nat) (a : AffineResult (CanonicalU64.highDifferenceExpr offset))
    (b : AffineResult (CanonicalU64.inverseExpr offset)) :
    DirectConstraintResult (flagExpression offset) where
  row := ⟨a.combination, b.combination,
    LinearCombination.add LinearCombination.one
      (LinearCombination.scale (-1) (LinearCombination.ofVar (offset + CanonicalU64.bitCount + 1)))⟩
  sound := by
    intro env rows
    change a.combination.eval env * b.combination.eval env = _ at rows
    simp only [a.sound env, b.sound env, LinearCombination.eval_add, LinearCombination.eval_one,
      LinearCombination.eval_scale, LinearCombination.eval_ofVar, neg_one_mul, ← sub_eq_add_neg] at rows
    change (Expr.var _ - (1 - CanonicalU64.highDifferenceExpr offset * CanonicalU64.inverseExpr offset)).eval env = 0
    simp only [Expr.eval_sub, Expr.eval_var, Expr.eval_hmul]
    apply sub_eq_zero.mpr
    apply eq_sub_iff_add_eq.mpr
    rw [add_comm]
    exact eq_sub_iff_add_eq.mp rows
  complete := by
    intro env rows
    change (Expr.var _ - (1 - CanonicalU64.highDifferenceExpr offset * CanonicalU64.inverseExpr offset)).eval env = 0 at rows
    simp only [Expr.eval_sub, Expr.eval_var, Expr.eval_hmul, sub_eq_zero] at rows
    change a.combination.eval env * b.combination.eval env = _
    simp only [a.sound env, b.sound env, LinearCombination.eval_add, LinearCombination.eval_one,
      LinearCombination.eval_scale, LinearCombination.eval_ofVar, neg_one_mul, ← sub_eq_add_neg]
    apply eq_sub_iff_add_eq.mpr
    rw [add_comm]
    exact eq_sub_iff_add_eq.mp rows

def basic? : (expression : Expr) → Option (DirectConstraintResult expression)
  | .mul left right => do
      let a ← lowerAffine left
      let b ← lowerAffine right
      pure (product left right a b)
  | expression => affineConstraint expression

def flag? (offset : Nat) : Option (DirectConstraintResult (flagExpression offset)) := do
  let a ← lowerAffine (CanonicalU64.highDifferenceExpr offset)
  let b ← lowerAffine (CanonicalU64.inverseExpr offset)
  pure (flag offset a b)

def flagAt? (offset : Nat) (expression : Expr) (lane : Fin fieldCount) :
    Option (DirectConstraintResult expression) :=
  if same : flagExpression (childOffset offset lane) = expression then
    (flag? (childOffset offset lane)).map fun compiled => same ▸ compiled
  else none

def constraint? (offset : Nat) (expression : Expr) : Option (DirectConstraintResult expression) :=
  match basic? expression with
  | some compiled => some compiled
  | none => (List.finRange fieldCount).findSome? (flagAt? offset expression)

def compile? (offset : Nat) : List Expr → Option (List Row)
  | [] => some []
  | expression :: rest =>
      match constraint? offset expression, compile? offset rest with
      | some head, some tail => some (head.row :: tail)
      | _, _ => none

theorem compile?_length (offset : Nat) (expressions : List Expr) (rows : List Row)
    (compiled : compile? offset expressions = some rows) : rows.length = expressions.length := by
  induction expressions generalizing rows with
  | nil =>
      have same := Option.some.inj compiled
      subst rows
      rfl
  | cons expression rest ih =>
      simp only [compile?] at compiled
      cases head : constraint? offset expression <;> rw [head] at compiled
      · cases compiled
      · cases tail : compile? offset rest <;> rw [tail] at compiled
        · cases compiled
        · cases Option.some.inj compiled
          simpa using ih _ tail

theorem compile?_correct (offset : Nat) (expressions : List Expr) (rows : List Row)
    (compiled : compile? offset expressions = some rows) (env : Env) :
    RowsHold env rows ↔ ConstraintsHold env expressions := by
  induction expressions generalizing rows with
  | nil =>
      have same := Option.some.inj compiled
      subst rows
      simp [RowsHold, ConstraintsHold]
  | cons expression rest ih =>
      simp only [compile?] at compiled
      cases head : constraint? offset expression with
      | none => rw [head] at compiled; cases compiled
      | some direct =>
          rw [head] at compiled
          cases tail : compile? offset rest with
          | none => rw [tail] at compiled; cases compiled
          | some remaining =>
              rw [tail] at compiled
              cases Option.some.inj compiled
              have restCorrect := ih remaining tail
              constructor
              · intro holds row member
                rcases List.mem_cons.mp member with rfl | member
                · exact direct.sound env (holds direct.row (by simp))
                · exact restCorrect.mp (fun r hr => holds r (by simp [hr])) row member
              · intro holds row member
                rcases List.mem_cons.mp member with rfl | member
                · exact direct.complete env (holds expression (by simp))
                · exact restCorrect.mpr (fun r hr => holds r (by simp [hr])) row member

end NightstreamFPrime.Layout.PiRlcWideSampler.Rows
