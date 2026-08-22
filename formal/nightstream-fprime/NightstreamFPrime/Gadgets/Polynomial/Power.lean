import NightstreamFPrime.Gadgets.Polynomial.Horner
import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.TargetPolynomial

/-!
Owns one fixed-exponent power circuit over the production quadratic extension.

The circuit reuses the causal owned Horner child with the constant-first
coefficient list `[0, ..., 0, 1]`. It allocates `2 * exponent` base-field
variables and does not use exponent-sized kernel evaluation.
-/

namespace NightstreamFPrime.Gadgets.Polynomial.Power

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Circuit.Quadratic
open NightstreamFPrime.Gadgets.Polynomial
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ConcreteCarrier

structure Interface where
  point : Nat → KExpr

def coefficientExprs (exponent : Nat) : List KExpr :=
  List.replicate exponent KExpr.zero ++ [KExpr.one]

def hornerInterface (exponent : Nat) (interface : Interface) :
    Horner.Owned.Interface where
  point := interface.point
  coefficients := fun _ => coefficientExprs exponent

/-- The child-owned symbolic power. -/
def output (exponent : Nat) (interface : Interface) (offset : Nat) : KExpr :=
  Horner.Owned.output (hornerInterface exponent interface) offset

abbrev Assumptions (exponent : Nat) (interface : Interface)
    (offset : Nat) (env : Env) : Prop :=
  Horner.Owned.Assumptions (hornerInterface exponent interface) offset env

abbrev SpecHolds (exponent : Nat) (interface : Interface)
    (offset : Nat) (env : Env) : Prop :=
  Horner.Owned.SpecHolds (hornerInterface exponent interface) offset env

theorem assumptions_of_point_varsBelow (exponent : Nat)
    (interface : Interface) (offset : Nat) (env : Env)
    (pointBelow : (interface.point offset).VarsBelow offset) :
    Assumptions exponent interface offset env := by
  refine ⟨pointBelow, ?_⟩
  intro coefficient member
  change coefficient ∈ coefficientExprs exponent at member
  simp only [coefficientExprs, List.mem_append, List.mem_replicate,
    List.mem_singleton] at member
  rcases member with ⟨_, rfl⟩ | rfl <;> exact ⟨trivial, trivial⟩

def circuit (exponent : Nat) (interface : Interface) : FormalCircuit :=
  Horner.Owned.circuit (hornerInterface exponent interface)

theorem soundness (exponent : Nat) (interface : Interface)
    (env : Env) (offset : Nat)
    (assumptions : Assumptions exponent interface offset env)
    (rows : holds env
      (Circuit.ops (circuit exponent interface).main offset)) :
    SpecHolds exponent interface offset env :=
  Horner.Owned.soundness (hornerInterface exponent interface) env offset
    assumptions rows

/-- Honest execution constructs the owned power with no semantic premise. -/
theorem build (exponent : Nat) (interface : Interface)
    (env : Env) (offset : Nat)
    (assumptions : Assumptions exponent interface offset env) :
    ∃ completed,
      AgreesOutside env completed offset
        (localLength
          (Circuit.ops (circuit exponent interface).main offset)) ∧
      holdsFlat completed
        (Circuit.ops (circuit exponent interface).main offset) :=
  Horner.Owned.build (hornerInterface exponent interface) env offset assumptions

theorem completeness (exponent : Nat) (interface : Interface)
    (env : Env) (offset : Nat)
    (assumptions : Assumptions exponent interface offset env)
    (_specification : SpecHolds exponent interface offset env) :
    ∃ completed,
      AgreesOutside env completed offset
        (localLength
          (Circuit.ops (circuit exponent interface).main offset)) ∧
      holdsFlat completed
        (Circuit.ops (circuit exponent interface).main offset) :=
  build exponent interface env offset assumptions

theorem coefficientExprs_length (exponent : Nat) :
    (coefficientExprs exponent).length = exponent + 1 := by
  simp [coefficientExprs]

theorem localLength_eq (exponent : Nat) (interface : Interface)
    (offset : Nat) :
    localLength (Circuit.ops (circuit exponent interface).main offset) =
      2 * exponent := by
  change localLength
    (Circuit.ops (Horner.Owned.circuit
      (hornerInterface exponent interface)).main offset) = _
  rw [Horner.Owned.localLength_eq]
  change 2 * ((coefficientExprs exponent).length - 1) = 2 * exponent
  rw [coefficientExprs_length]
  omega

theorem operations_length (exponent : Nat) (interface : Interface)
    (offset : Nat) :
    (Circuit.ops (circuit exponent interface).main offset).length = 1 :=
  Horner.Owned.operations_length (hornerInterface exponent interface) offset

theorem flatConstraints_length (exponent : Nat) (interface : Interface)
    (offset : Nat) :
    (flatConstraints
      (Circuit.ops (circuit exponent interface).main offset)).length =
      2 * exponent := by
  change (flatConstraints (Circuit.ops (Horner.Owned.circuit
    (hornerInterface exponent interface)).main offset)).length = _
  rw [Horner.Owned.flatConstraints_length]
  change 2 * ((coefficientExprs exponent).length - 1) = 2 * exponent
  rw [coefficientExprs_length]
  omega

theorem flatConstraints_varsBelow_exact (exponent : Nat)
    (interface : Interface) (offset : Nat) (env : Env)
    (assumptions : Assumptions exponent interface offset env) :
    ∀ expression ∈ flatConstraints
      (Circuit.ops (circuit exponent interface).main offset),
      expression.VarsBelow (offset + 2 * exponent) := by
  have scope := Horner.Owned.flatConstraints_varsBelow
    (hornerInterface exponent interface) offset env assumptions
  change ∀ expression ∈ flatConstraints
      (Circuit.ops (circuit exponent interface).main offset),
    expression.VarsBelow
      (offset + localLength
        (Circuit.ops (circuit exponent interface).main offset)) at scope
  rw [localLength_eq exponent interface offset] at scope
  exact scope

/-- The owned fixed power lies inside its exact symbolic interval. -/
theorem output_varsBelow (exponent : Nat) (interface : Interface)
    (offset : Nat) (env : Env)
    (assumptions : Assumptions exponent interface offset env) :
    (output exponent interface offset).VarsBelow
      (offset + 2 * exponent) := by
  have below := Horner.Owned.output_varsBelow
    (hornerInterface exponent interface) offset env assumptions
  change (output exponent interface offset).VarsBelow
    (offset + localLength
      (Circuit.ops (circuit exponent interface).main offset)) at below
  rw [localLength_eq exponent interface offset] at below
  exact below

theorem evaluateCoefficients_eq_power (point : K) : ∀ exponent,
    SumCheck.Finite.Message.evaluateCoefficients extensionOps.toOps point
        (List.replicate exponent K.zero ++ [K.one]) =
      TargetPolynomial.power extensionOps.toOps point exponent
  | 0 => by
      simp [SumCheck.Finite.Message.evaluateCoefficients,
        TargetPolynomial.power, extensionLaws.mul_zero,
        extensionLaws.add_zero]
      rfl
  | exponent + 1 => by
      rw [List.replicate_succ, List.cons_append]
      simp only [SumCheck.Finite.Message.evaluateCoefficients,
        TargetPolynomial.power]
      rw [evaluateCoefficients_eq_power point exponent]
      change extensionOps.add extensionOps.zero
        (extensionOps.mul point
          (TargetPolynomial.power extensionOps.toOps point exponent)) = _
      exact extensionLaws.zero_add _

theorem coefficientExprs_eval (env : Env) (exponent : Nat) :
    (coefficientExprs exponent).map (KExpr.eval env) =
      List.replicate exponent K.zero ++ [K.one] := by
  simp [coefficientExprs]

/-- The child specification is exactly fixed exponentiation. -/
theorem spec_implies_power (exponent : Nat) (interface : Interface)
    (offset : Nat) (env : Env)
    (specification : SpecHolds exponent interface offset env) :
    (output exponent interface offset).eval env =
      TargetPolynomial.power extensionOps.toOps
        ((interface.point offset).eval env) exponent := by
  unfold SpecHolds Horner.Owned.SpecHolds hornerInterface at specification
  rw [coefficientExprs_eval] at specification
  exact specification.trans
    (evaluateCoefficients_eq_power ((interface.point offset).eval env) exponent)

theorem specHolds_iff_power (exponent : Nat) (interface : Interface)
    (offset : Nat) (env : Env) :
    SpecHolds exponent interface offset env ↔
      (output exponent interface offset).eval env =
        TargetPolynomial.power extensionOps.toOps
          ((interface.point offset).eval env) exponent := by
  constructor
  · exact spec_implies_power exponent interface offset env
  · intro specification
    unfold SpecHolds Horner.Owned.SpecHolds hornerInterface
    rw [coefficientExprs_eval, evaluateCoefficients_eq_power]
    exact specification

end NightstreamFPrime.Gadgets.Polynomial.Power
