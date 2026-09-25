import NightstreamFPrime.Circuit.StraightLine

/-!
Owns the symbolic two-cell representation of the production quadratic
extension `K = F[X]/(X² - 7)`. The cell order is exactly the protocol
serializer order `c0, c1`.
-/

namespace NightstreamFPrime.Circuit.Quadratic

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit

structure KExpr where
  c0 : Expr
  c1 : Expr

namespace KExpr

def eval (env : Env) (value : KExpr) : K :=
  ⟨value.c0.eval env, value.c1.eval env⟩

def zero : KExpr := ⟨0, 0⟩
def one : KExpr := ⟨1, 0⟩

def add (left right : KExpr) : KExpr :=
  ⟨left.c0 + right.c0, left.c1 + right.c1⟩

def sub (left right : KExpr) : KExpr :=
  ⟨left.c0 - right.c0, left.c1 - right.c1⟩

/-- Multiplication in `F[X]/(X² - 7)`. -/
def mul (left right : KExpr) : KExpr :=
  ⟨left.c0 * right.c0 + 7 * left.c1 * right.c1,
    left.c0 * right.c1 + left.c1 * right.c0⟩

@[simp] theorem eval_zero (env : Env) : zero.eval env = K.zero := by
  rfl

@[simp] theorem eval_one (env : Env) : one.eval env = K.one := by
  rfl

@[simp] theorem eval_add (env : Env) (left right : KExpr) :
    (add left right).eval env = K.add (left.eval env) (right.eval env) := by
  rfl

@[simp] theorem eval_sub (env : Env) (left right : KExpr) :
    (sub left right).eval env = K.sub (left.eval env) (right.eval env) := by
  cases left
  cases right
  simp [sub, eval, K.sub]

@[simp] theorem eval_mul (env : Env) (left right : KExpr) :
    (mul left right).eval env = K.mul (left.eval env) (right.eval env) := by
  cases left
  cases right
  rfl

def equalities (left right : KExpr) : List Expr :=
  [left.c0 - right.c0, left.c1 - right.c1]

def VarsBelow (value : KExpr) (bound : Nat) : Prop :=
  value.c0.VarsBelow bound ∧ value.c1.VarsBelow bound

theorem add_varsBelow (left right : KExpr) (bound : Nat)
    (leftBelow : left.VarsBelow bound)
    (rightBelow : right.VarsBelow bound) :
    (add left right).VarsBelow bound :=
  ⟨Expr.VarsBelow.add left.c0 right.c0 bound leftBelow.1 rightBelow.1,
    Expr.VarsBelow.add left.c1 right.c1 bound leftBelow.2 rightBelow.2⟩

theorem mul_varsBelow (left right : KExpr) (bound : Nat)
    (leftBelow : left.VarsBelow bound)
    (rightBelow : right.VarsBelow bound) :
    (mul left right).VarsBelow bound := by
  constructor
  · exact Expr.VarsBelow.add _ _ bound
      (Expr.VarsBelow.mul _ _ bound leftBelow.1 rightBelow.1)
      (Expr.VarsBelow.mul _ _ bound
        (Expr.VarsBelow.mul _ _ bound trivial leftBelow.2) rightBelow.2)
  · exact Expr.VarsBelow.add _ _ bound
      (Expr.VarsBelow.mul _ _ bound leftBelow.1 rightBelow.2)
      (Expr.VarsBelow.mul _ _ bound leftBelow.2 rightBelow.1)

theorem equalities_varsBelow (left right : KExpr) (bound : Nat)
    (leftBelow : left.VarsBelow bound)
    (rightBelow : right.VarsBelow bound) :
    ∀ expression ∈ equalities left right,
      expression.VarsBelow bound := by
  intro expression member
  simp only [equalities, List.mem_cons, List.not_mem_nil, or_false] at member
  rcases member with rfl | rfl
  · exact Expr.VarsBelow.sub left.c0 right.c0 bound
      leftBelow.1 rightBelow.1
  · exact Expr.VarsBelow.sub left.c1 right.c1 bound
      leftBelow.2 rightBelow.2

theorem eval_eq_of_agree_below (value : KExpr) (bound : Nat)
    (left right : Env) (below : value.VarsBelow bound)
    (agrees : ∀ index, index < bound → left index = right index) :
    value.eval left = value.eval right := by
  exact congrArg₂ K.mk
    (value.c0.eval_eq_of_agree_below bound left right below.1 agrees)
    (value.c1.eval_eq_of_agree_below bound left right below.2 agrees)

theorem varsBelow_mono (value : KExpr) {lower upper : Nat}
    (below : value.VarsBelow lower) (le : lower ≤ upper) :
    value.VarsBelow upper :=
  ⟨Expr.VarsBelow.mono _ below.1 le, Expr.VarsBelow.mono _ below.2 le⟩

theorem equalities_hold_iff (env : Env) (left right : KExpr) :
    ConstraintsHold env (equalities left right) ↔
      left.eval env = right.eval env := by
  constructor
  · intro holds
    have c0Row : (left.c0 - right.c0).eval env = 0 :=
      holds (left.c0 - right.c0) (by simp [equalities])
    have c1Row : (left.c1 - right.c1).eval env = 0 :=
      holds (left.c1 - right.c1) (by simp [equalities])
    exact congrArg₂ K.mk
      (sub_eq_zero.mp (by simpa using c0Row))
      (sub_eq_zero.mp (by simpa using c1Row))
  · intro equals expression member
    have c0 : left.c0.eval env = right.c0.eval env :=
      congrArg K.c0 equals
    have c1 : left.c1.eval env = right.c1.eval env :=
      congrArg K.c1 equals
    simp only [equalities, List.mem_cons, List.not_mem_nil, or_false] at member
    rcases member with rfl | rfl
    · simpa using sub_eq_zero.mpr c0
    · simpa using sub_eq_zero.mpr c1

end KExpr

end NightstreamFPrime.Circuit.Quadratic
