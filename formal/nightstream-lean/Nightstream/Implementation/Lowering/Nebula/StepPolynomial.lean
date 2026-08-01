import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ConcreteCarrier.Algebra

/-!
Lean-owned sparse CCS polynomial for one Nebula memory step.

Assurance tier: model-level.

Owns: the fifteen matrix-image roles, their numeric order, the exact eleven
monomials of the selected `S_mem` relation, the strict degree bound five, and
the reduction of each disjoint row family to its intended equation.

Does not own: matrix coefficients, witness columns, row schedules, the WASM
port map, transcript challenges, folding, R1CS lowering, Rust data, or cost.

The polynomial degree is four. The strict `ConstraintPolynomial.degreeBound`
is therefore five. The equality-gated PiCCS SumCheck ceiling is also five and
is derived from the monomial syntax.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.Lowering.Nebula.StepPolynomial

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.CCSResidualTable
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ConcreteCarrier

/-- Exact matrix arity of the selected Nebula step relation. -/
def matrixCount : Nat := 15

/-- Semantic name of each matrix image used by the Nebula step polynomial. -/
inductive Role where
  | bit
  | productLeft
  | productRight
  | linearLeft
  | linearRight
  | output
  | extensionA
  | extensionB
  | pad
  | active
  | fingerprintA
  | fingerprintB
  | valueA
  | valueB
  | value
deriving DecidableEq, Repr

/-- Numeric matrix position for each role. The order is part of the manifest
ABI and has no caller-selected permutation. -/
@[simp] def Role.index : Role -> Fin matrixCount
  | .bit => ⟨0, by decide⟩
  | .productLeft => ⟨1, by decide⟩
  | .productRight => ⟨2, by decide⟩
  | .linearLeft => ⟨3, by decide⟩
  | .linearRight => ⟨4, by decide⟩
  | .output => ⟨5, by decide⟩
  | .extensionA => ⟨6, by decide⟩
  | .extensionB => ⟨7, by decide⟩
  | .pad => ⟨8, by decide⟩
  | .active => ⟨9, by decide⟩
  | .fingerprintA => ⟨10, by decide⟩
  | .fingerprintB => ⟨11, by decide⟩
  | .valueA => ⟨12, by decide⟩
  | .valueB => ⟨13, by decide⟩
  | .value => ⟨14, by decide⟩

theorem Role.index_injective : Function.Injective Role.index := by
  intro left right equal
  cases left <;> cases right <;> simp_all

/-- Inverse semantic role for every physical matrix position. -/
@[simp] def Role.ofIndex : Fin matrixCount -> Role
  | ⟨0, _⟩ => .bit
  | ⟨1, _⟩ => .productLeft
  | ⟨2, _⟩ => .productRight
  | ⟨3, _⟩ => .linearLeft
  | ⟨4, _⟩ => .linearRight
  | ⟨5, _⟩ => .output
  | ⟨6, _⟩ => .extensionA
  | ⟨7, _⟩ => .extensionB
  | ⟨8, _⟩ => .pad
  | ⟨9, _⟩ => .active
  | ⟨10, _⟩ => .fingerprintA
  | ⟨11, _⟩ => .fingerprintB
  | ⟨12, _⟩ => .valueA
  | ⟨13, _⟩ => .valueB
  | ⟨14, _⟩ => .value

@[simp] theorem Role.ofIndex_index (role : Role) :
    Role.ofIndex role.index = role := by
  cases role <;> rfl

@[simp] theorem Role.index_ofIndex :
    forall index : Fin matrixCount, (Role.ofIndex index).index = index
  | ⟨0, _⟩ => rfl
  | ⟨1, _⟩ => rfl
  | ⟨2, _⟩ => rfl
  | ⟨3, _⟩ => rfl
  | ⟨4, _⟩ => rfl
  | ⟨5, _⟩ => rfl
  | ⟨6, _⟩ => rfl
  | ⟨7, _⟩ => rfl
  | ⟨8, _⟩ => rfl
  | ⟨9, _⟩ => rfl
  | ⟨10, _⟩ => rfl
  | ⟨11, _⟩ => rfl
  | ⟨12, _⟩ => rfl
  | ⟨13, _⟩ => rfl
  | ⟨14, _⟩ => rfl

/-- Construct one exponent vector. Every exported term lists each role at
most once. -/
def exponentVector (powers : List (Role × Nat)) : Fin matrixCount -> Nat :=
  fun index => powers.foldl
    (fun current power => if index = power.1.index then power.2 else current) 0

def monomial (coefficient : F) (powers : List (Role × Nat)) :
    Monomial F matrixCount where
  coefficient := coefficient
  exponents := exponentVector powers

/-- Exact ordered sparse syntax of the selected Nebula memory relation.

```text
B^2 - B + PL*PR + LL - LR - O
  + A*P + A*Q*FA - A*Q*GA*V
  + B*Q*FB - B*Q*GB*V
```
-/
def terms : List (Monomial F matrixCount) := [
  monomial 1 [(.bit, 2)],
  monomial (-1) [(.bit, 1)],
  monomial 1 [(.productLeft, 1), (.productRight, 1)],
  monomial 1 [(.linearLeft, 1)],
  monomial (-1) [(.linearRight, 1)],
  monomial (-1) [(.output, 1)],
  monomial 1 [(.extensionA, 1), (.pad, 1)],
  monomial 1 [(.extensionA, 1), (.active, 1), (.fingerprintA, 1)],
  monomial (-1) [(.extensionA, 1), (.active, 1), (.valueA, 1), (.value, 1)],
  monomial 1 [(.extensionB, 1), (.active, 1), (.fingerprintB, 1)],
  monomial (-1) [(.extensionB, 1), (.active, 1), (.valueB, 1), (.value, 1)]
]

theorem term_count_exact : terms.length = 11 := by
  rfl

private theorem every_term_degree_checked :
    terms.all (fun term => decide (term.totalDegree < 5)) = true := by
  decide

/-- The strict degree bound is derived from the explicit degree-four syntax. -/
def polynomial : ConstraintPolynomial F matrixCount where
  degreeBound := 5
  terms := terms
  termsBelowDegree := by
    intro term member
    exact of_decide_eq_true
      ((List.all_eq_true.mp every_term_degree_checked) term member)

set_option maxRecDepth 10000 in
/-- The equality-gated SumCheck ceiling is exactly five. -/
theorem canonicalEqualityGatedDegreeBound_exact :
    polynomial.canonicalEqualityGatedDegreeBound = 5 := by
  decide

/-- Direct evaluation of the exact sparse syntax. -/
def evaluate (point : Fin matrixCount -> F) : F :=
  evaluatePolynomial baseOps polynomial point

/-- Human-auditable form of the same sparse polynomial. Its addend order is
the monomial order above. -/
def residual (point : Fin matrixCount -> F) : F :=
  point Role.bit.index * point Role.bit.index +
    -(point Role.bit.index) +
    point Role.productLeft.index * point Role.productRight.index +
    point Role.linearLeft.index +
    -(point Role.linearRight.index) +
    -(point Role.output.index) +
    point Role.extensionA.index * point Role.pad.index +
    point Role.extensionA.index * point Role.active.index *
      point Role.fingerprintA.index +
    -(point Role.extensionA.index * point Role.active.index *
      point Role.valueA.index * point Role.value.index) +
    point Role.extensionB.index * point Role.active.index *
      point Role.fingerprintB.index +
    -(point Role.extensionB.index * point Role.active.index *
      point Role.valueB.index * point Role.value.index)

set_option maxRecDepth 10000 in
/-- The named equation is exactly the eleven-term sparse evaluator. -/
theorem evaluate_eq_residual (point : Fin matrixCount -> F) :
    evaluate point = residual point := by
  simp [evaluate, polynomial, terms, monomial, exponentVector, matrixCount,
    evaluatePolynomial, evaluateMonomial, pow, canonicalFinIndices, baseOps,
    residual, Role.index, Fin.one_mul, Fin.mul_one, Fin.zero_add,
    Lean.Grind.Fin.neg_mul]

/-- Sparse matrix-image constructor for the closed row families below. -/
def sparsePoint (entries : List (Role × F)) : Fin matrixCount -> F :=
  fun role => entries.foldl
    (fun current entry => if role = entry.1.index then entry.2 else current) 0

def bitPoint (bit : F) : Fin matrixCount -> F :=
  sparsePoint [(.bit, bit)]

def productPoint (left right : F) : Fin matrixCount -> F :=
  sparsePoint [(.productLeft, left), (.productRight, right)]

/-- Product equality uses `linearRight`, not the extension-output port. -/
def productEqualityPoint (left right output : F) : Fin matrixCount -> F :=
  sparsePoint [(.productLeft, left), (.productRight, right),
    (.linearRight, output)]

def linearPoint (left right : F) : Fin matrixCount -> F :=
  sparsePoint [(.linearLeft, left), (.linearRight, right)]

/-- One base-field component of the quadratic-extension product update. -/
def extensionUpdatePoint
    (output a b pad active fingerprintA fingerprintB valueA valueB value : F) :
    Fin matrixCount -> F :=
  sparsePoint [(.output, output), (.extensionA, a), (.extensionB, b),
    (.pad, pad), (.active, active), (.fingerprintA, fingerprintA),
    (.fingerprintB, fingerprintB), (.valueA, valueA), (.valueB, valueB),
    (.value, value)]

theorem evaluate_bitPoint (bit : F) :
    evaluate (bitPoint bit) = bit * bit + -bit := by
  rw [evaluate_eq_residual]
  simp [residual, bitPoint, sparsePoint, Role.index, matrixCount,
    Fin.mul_zero, Fin.add_zero, Lean.Grind.AddCommGroup.neg_zero]

theorem evaluate_productPoint (left right : F) :
    evaluate (productPoint left right) = left * right := by
  rw [evaluate_eq_residual]
  simp [residual, productPoint, sparsePoint, Role.index, matrixCount,
    Fin.mul_zero, Fin.zero_add, Fin.add_zero, Lean.Grind.AddCommGroup.neg_zero]

theorem evaluate_productEqualityPoint (left right output : F) :
    evaluate (productEqualityPoint left right output) =
      left * right + -output := by
  rw [evaluate_eq_residual]
  simp [residual, productEqualityPoint, sparsePoint, Role.index, matrixCount,
    Fin.mul_zero, Fin.zero_add, Fin.add_zero, Lean.Grind.AddCommGroup.neg_zero]

theorem evaluate_linearPoint (left right : F) :
    evaluate (linearPoint left right) = left + -right := by
  rw [evaluate_eq_residual]
  simp [residual, linearPoint, sparsePoint, Role.index, matrixCount,
    Fin.mul_zero, Fin.zero_add, Fin.add_zero, Lean.Grind.AddCommGroup.neg_zero]

theorem evaluate_extensionUpdatePoint
    (output a b pad active fingerprintA fingerprintB valueA valueB value : F) :
    evaluate (extensionUpdatePoint output a b pad active fingerprintA
      fingerprintB valueA valueB value) =
      -output + a * pad + a * active * fingerprintA +
        -(a * active * valueA * value) +
        b * active * fingerprintB +
        -(b * active * valueB * value) := by
  rw [evaluate_eq_residual]
  simp [residual, extensionUpdatePoint, sparsePoint, Role.index, matrixCount,
    Fin.mul_zero, Fin.zero_add, Fin.add_zero, Lean.Grind.AddCommGroup.neg_zero]

end Nightstream.Implementation.Lowering.Nebula.StepPolynomial
