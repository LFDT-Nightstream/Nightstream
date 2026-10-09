import NightstreamFPrime.Layout.Polynomial.Horner
import NightstreamFPrime.Lifecycle.PiCCS.v1_2.Completeness

/-!
Paper authority: SuperNeo v1.2, section 7.3, PiCCS input statement.
Obligation: Share the prior point and separate Eval_K / Eval_A input families.

Inputs:
- the parent PiCCS symbolic interface.

Outputs:
- the exact physical footprint of the parent-facing Statement-binding child.

Constraint groups:
- the twelve domain-chunk words of each state;
- four prior-context and four output-context equalities to the expected
  verifier-owned public value;
- for each of the 90 packed prior words: three lanes of one sign row and
  sixteen rank-one digit rows over one hinted sign column, then one affine
  packing row.

Parent coverage:
- `Formal.opsAt`, child `piccs.v1_2.statement_binding`.
-/

namespace NightstreamFPrime.Layout.PiCCS.v1_2.Leaves.StatementBinding

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Lifecycle.PiCCS.v1_2
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint

variable {logicalWidth degreeBound : Nat}
  {publicFits : ringDegree * publicRingColumns ≤
    Phi81CarrierLayout.carrierWidth logicalWidth}

structure InputsAffine
    (interface : Formal.Interface logicalWidth degreeBound publicFits)
    (offset : Nat) : Prop where
  priorState : ∀ index,
    R1CS.IsAffine (interface.priorState offset index)
  outputState : ∀ index,
    R1CS.IsAffine (interface.outputState offset index)
  expectedContext : ∀ lane,
    R1CS.IsAffine (interface.expectedContext offset lane)
  priorDigit : ∀ word lane child,
    R1CS.IsAffine ((interface.running offset).publicInput
        (Fin.cast runningCount_eq_radixChildCount.symm child) (packedColumn word lane)) ∧
      Layout.Polynomial.Horner.Nonconstant ((interface.running offset).publicInput
        (Fin.cast runningCount_eq_radixChildCount.symm child) (packedColumn word lane))

private theorem sub_affine {left right : Expr}
    (leftAffine : R1CS.IsAffine left)
    (rightAffine : R1CS.IsAffine right) :
    R1CS.IsAffine (left - right) := by
  exact R1CS.IsAffine.add leftAffine
    (R1CS.IsAffine.const_mul (-1) rightAffine)

private theorem stateAssertions_affine (state : Nat → Expr)
    (stateAffine : ∀ index, R1CS.IsAffine (state index)) :
    ∀ expression ∈ StateBinding.stateAssertions state,
      R1CS.IsAffine expression := by
  intro expression member
  rw [StateBinding.stateAssertions, List.mem_map] at member
  rcases member with ⟨word, _wordMember, rfl⟩
  exact sub_affine (stateAffine word.index) (R1CS.isAffine_const _)

private theorem contextAssertions_affine (state : Nat → Expr)
    (expected : Fin 4 → Expr)
    (stateAffine : ∀ index, R1CS.IsAffine (state index))
    (expectedAffine : ∀ lane, R1CS.IsAffine (expected lane)) :
    ∀ expression ∈ StateBinding.contextAssertions state expected,
      R1CS.IsAffine expression := by
  intro expression member
  rw [StateBinding.contextAssertions, List.mem_map] at member
  rcases member with ⟨lane, _laneMember, rfl⟩
  exact sub_affine (stateAffine _) (expectedAffine _)

private theorem constraintFreshCount_eq_zero_of_affine (expression : Expr)
    (affine : R1CS.IsAffine expression) :
    R1CS.constraintFreshCount expression = 0 := by
  have notNone := R1CS.directConstraint_ne_none_of_affine expression affine
  unfold R1CS.constraintFreshCount
  cases equal : R1CS.directConstraint expression with
  | none => exact False.elim (notNone equal)
  | some direct => rfl

private theorem constraintRowCount_eq_one_of_affine (expression : Expr)
    (affine : R1CS.IsAffine expression) :
    R1CS.constraintRowCount expression = 1 := by
  have notNone := R1CS.directConstraint_ne_none_of_affine expression affine
  unfold R1CS.constraintRowCount
  cases equal : R1CS.directConstraint expression with
  | none => exact False.elim (notNone equal)
  | some direct => rfl

/-- Every state-binding row lowers to one direct row with no fresh column:
the state rows are affine, and each sign or digit row is one rank-one
product of nonconstant affine factors. -/
private theorem constraint_direct
    (interface : Formal.Interface logicalWidth degreeBound publicFits)
    (offset : Nat) (inputs : InputsAffine interface offset) :
    ∀ expression ∈ flatConstraints (Circuit.ops
      (Formal.statementBindingCircuit interface).main offset),
      R1CS.constraintFreshCount expression = 0 ∧
        R1CS.constraintRowCount expression = 1 := by
  have ofAffine : ∀ expression, R1CS.IsAffine expression →
      R1CS.constraintFreshCount expression = 0 ∧
        R1CS.constraintRowCount expression = 1 := fun expression affine =>
    ⟨constraintFreshCount_eq_zero_of_affine expression affine,
      constraintRowCount_eq_one_of_affine expression affine⟩
  have ofProduct : ∀ left right, R1CS.IsAffine left → R1CS.IsAffine right →
      Layout.Polynomial.Horner.Nonconstant left →
      Layout.Polynomial.Horner.Nonconstant right →
      R1CS.constraintFreshCount (left * right) = 0 ∧
        R1CS.constraintRowCount (left * right) = 1 :=
    fun _ _ leftAffine rightAffine leftNonconstant rightNonconstant =>
      ⟨Layout.Polynomial.Horner.constraintFreshCount_mul leftAffine rightAffine
          leftNonconstant rightNonconstant,
        Layout.Polynomial.Horner.constraintRowCount_mul leftAffine rightAffine
          leftNonconstant rightNonconstant⟩
  have subNonconstant : ∀ left right : Expr,
      Layout.Polynomial.Horner.Nonconstant (left - right) := by
    intro _ _ value equal
    cases equal
  intro expression member
  unfold Formal.statementBindingCircuit at member
  rw [FormalCircuit.withConstantFootprint_main,
    StatementBinding.flatConstraints_eq_stateAssertions] at member
  rw [StateBinding.assertions, List.mem_append] at member
  rcases member with member | childMember
  · apply ofAffine
    rw [StateBinding.stateWordAssertions, List.mem_append] at member
    rcases member with priorMember | remainingMember
    · exact stateAssertions_affine _ inputs.priorState expression priorMember
    · rw [List.mem_append] at remainingMember
      rcases remainingMember with middleMember | outputContextMember
      · rw [List.mem_append] at middleMember
        rcases middleMember with outputMember | priorContextMember
        · exact stateAssertions_affine _ inputs.outputState expression
            outputMember
        · exact contextAssertions_affine _ _ inputs.priorState
            inputs.expectedContext expression priorContextMember
      · exact contextAssertions_affine _ _ inputs.outputState
          inputs.expectedContext expression outputContextMember
  · rcases StateBinding.childRow_cases childMember with
      ⟨word, ⟨lane, signRow | ⟨child, digitRow⟩⟩ | packedRow⟩
    · subst expression
      have sign : R1CS.IsAffine (StateBinding.signBit offset word lane) :=
        R1CS.isAffine_var _
      exact ofProduct _ _ sign (sub_affine sign (R1CS.isAffine_const _))
        (fun _ equal => by cases equal) (subNonconstant _ _)
    · subst expression
      have digit := inputs.priorDigit word lane child
      have sign : R1CS.IsAffine (StateBinding.signBit offset word lane) :=
        R1CS.isAffine_var _
      exact ofProduct _ _ digit.1
        (sub_affine digit.1 (sub_affine (R1CS.isAffine_const _)
          (R1CS.IsAffine.const_mul _ sign)))
        digit.2 (subNonconstant _ _)
    · subst expression
      apply ofAffine
      have recomposed (lane : Fin 3) :
          R1CS.IsAffine (PiDEC.v1_2.SignedSplitScalar.recomposeDigits fun child =>
            (interface.running offset).publicInput
              (Fin.cast runningCount_eq_radixChildCount.symm child)
              (packedColumn word lane)) :=
        PiDEC.v1_2.SignedSplitScalar.recomposeDigits_closed R1CS.IsAffine
          R1CS.isAffine_const (fun _ _ => R1CS.IsAffine.add)
          (fun weight _ => R1CS.IsAffine.const_mul weight)
          _ fun child => (inputs.priorDigit word lane child).1
      exact sub_affine (inputs.priorState _)
        (StateBinding.packWordExpr_closed R1CS.IsAffine (fun _ _ => R1CS.IsAffine.add)
          (fun weight _ => R1CS.IsAffine.const_mul weight)
          (recomposed 0) (recomposed 1) (recomposed 2))

def footprint
    (interface : Formal.Interface logicalWidth degreeBound publicFits)
    (inputs : ∀ offset, InputsAffine interface offset) :
    R1CS.CircuitFootprint (Formal.statementBindingCircuit interface) where
  freshColumnCount := fun _ => 0
  physicalRowCount := fun _ => 4712
  freshColumnCount_eq := by
    intro offset
    apply R1CS.totalFreshCount_eq_zero_of_noFresh
    intro expression member
    exact (constraint_direct interface offset (inputs offset) expression member).1
  physicalRowCount_eq := by
    intro offset
    rw [R1CS.totalRowCount_eq_length_of_rowsOne]
    · unfold Formal.statementBindingCircuit
      rw [FormalCircuit.withConstantFootprint_main]
      exact StatementBinding.flatConstraints_length
        (Formal.statementBindingInterface interface) offset
    · intro expression member
      exact (constraint_direct interface offset (inputs offset) expression member).2

theorem freshColumnCount_eq
    (interface : Formal.Interface logicalWidth degreeBound publicFits)
    (inputs : ∀ offset, InputsAffine interface offset)
    (offset : Nat) :
    R1CS.totalFreshCount (flatConstraints (Circuit.ops
      (Formal.statementBindingCircuit interface).main offset)) = 0 :=
  (footprint interface inputs).freshColumnCount_eq offset

theorem physicalRowCount_eq
    (interface : Formal.Interface logicalWidth degreeBound publicFits)
    (inputs : ∀ offset, InputsAffine interface offset)
    (offset : Nat) :
    R1CS.totalRowCount (flatConstraints (Circuit.ops
      (Formal.statementBindingCircuit interface).main offset)) = 4712 :=
  (footprint interface inputs).physicalRowCount_eq offset

end NightstreamFPrime.Layout.PiCCS.v1_2.Leaves.StatementBinding
