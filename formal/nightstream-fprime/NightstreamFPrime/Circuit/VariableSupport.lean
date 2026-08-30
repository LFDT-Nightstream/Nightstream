import NightstreamFPrime.Circuit.StraightLine

/-!
Owns a generic predicate for the variable indices read by one circuit
expression. It does not select a protocol support set.
-/

namespace NightstreamFPrime.Circuit.Expr

/-- Every variable read by an expression satisfies one caller-selected
predicate. -/
def VarsSatisfy (allowed : Nat → Prop) : Expr → Prop
  | .var index => allowed index
  | .const _ => True
  | .add left right => left.VarsSatisfy allowed ∧ right.VarsSatisfy allowed
  | .mul left right => left.VarsSatisfy allowed ∧ right.VarsSatisfy allowed

namespace VarsSatisfy

theorem add (left right : Expr) (allowed : Nat → Prop)
    (leftSupported : left.VarsSatisfy allowed)
    (rightSupported : right.VarsSatisfy allowed) :
    (left + right).VarsSatisfy allowed :=
  ⟨leftSupported, rightSupported⟩

theorem mul (left right : Expr) (allowed : Nat → Prop)
    (leftSupported : left.VarsSatisfy allowed)
    (rightSupported : right.VarsSatisfy allowed) :
    (left * right).VarsSatisfy allowed :=
  ⟨leftSupported, rightSupported⟩

theorem neg (expression : Expr) (allowed : Nat → Prop)
    (supported : expression.VarsSatisfy allowed) :
    (-expression).VarsSatisfy allowed :=
  ⟨trivial, supported⟩

theorem sub (left right : Expr) (allowed : Nat → Prop)
    (leftSupported : left.VarsSatisfy allowed)
    (rightSupported : right.VarsSatisfy allowed) :
    (left - right).VarsSatisfy allowed :=
  ⟨leftSupported, ⟨trivial, rightSupported⟩⟩

theorem mono {allowed larger : Nat → Prop} (expression : Expr)
    (scope : expression.VarsSatisfy allowed)
    (includes : ∀ index, allowed index → larger index) :
    expression.VarsSatisfy larger := by
  induction expression with
  | var index => exact includes index scope
  | const value => trivial
  | add left right leftIH rightIH =>
      exact ⟨leftIH scope.1, rightIH scope.2⟩
  | mul left right leftIH rightIH =>
      exact ⟨leftIH scope.1, rightIH scope.2⟩

end VarsSatisfy

theorem varsSatisfy_lt_iff_varsBelow (expression : Expr) (bound : Nat) :
    expression.VarsSatisfy (fun index => index < bound) ↔
      expression.VarsBelow bound := by
  induction expression with
  | var index => rfl
  | const value => rfl
  | add left right leftIH rightIH =>
      simp only [VarsSatisfy, VarsBelow, leftIH, rightIH]
  | mul left right leftIH rightIH =>
      simp only [VarsSatisfy, VarsBelow, leftIH, rightIH]

end NightstreamFPrime.Circuit.Expr
