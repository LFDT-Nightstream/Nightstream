import NightstreamFPrime.Layout.R1CS
import NightstreamFPrime.Lifecycle.PiCCS.v1_1.Completeness

/-!
Paper authority: SuperNeo v1_1, section 7.3, PiCCS input statement.
Obligation: Share the prior point and separate Eval_K / Eval_A input families.

Inputs:
- the parent PiCCS symbolic interface.

Outputs:
- the exact physical footprint of the parent-facing Statement-binding child.

Constraint groups:
- canonical tag, block-length, and program-counter words;
- four prior-context and four output-context equalities to the expected
  verifier-owned public value.

Parent coverage:
- `Formal.opsAt`, child `piccs.v1_1.statement_binding`.
-/

namespace NightstreamFPrime.Layout.PiCCS.v1_1.Leaves.StatementBinding

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Lifecycle.PiCCS.v1_1
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

private theorem constraints_affine
    (interface : Formal.Interface logicalWidth degreeBound publicFits)
    (offset : Nat) (inputs : InputsAffine interface offset) :
    ∀ expression ∈ flatConstraints (Circuit.ops
      (Formal.statementBindingCircuit interface).main offset),
      R1CS.IsAffine expression := by
  intro expression member
  unfold Formal.statementBindingCircuit at member
  rw [FormalCircuit.withConstantFootprint_main,
    StatementBinding.flatConstraints_eq_stateAssertions] at member
  rw [StateBinding.assertions, List.mem_append] at member
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

private theorem directConstraint_ne_none_of_affine (expression : Expr)
    (affine : R1CS.IsAffine expression) :
    R1CS.directConstraint expression ≠ none := by
  rcases affine with ⟨lowered, loweredEq⟩
  cases expression with
  | var index =>
      simp [R1CS.directConstraint, R1CS.affineConstraint, loweredEq]
  | const value =>
      simp [R1CS.directConstraint, R1CS.affineConstraint, loweredEq]
  | mul left right =>
      simp [R1CS.directConstraint, R1CS.affineConstraint, loweredEq]
  | add left right =>
      cases left with
      | var output =>
          cases right with
          | mul factor recipe =>
              cases factor with
              | const coefficient =>
                  by_cases coefficientEq : coefficient = -1
                  · cases recipeEq : R1CS.directRecipeRow output recipe <;>
                      simp [R1CS.directConstraint, coefficientEq, recipeEq,
                        R1CS.affineConstraint, loweredEq]
                  · simp [R1CS.directConstraint, coefficientEq,
                      R1CS.affineConstraint, loweredEq]
              | var index =>
                  simp [R1CS.directConstraint, R1CS.affineConstraint,
                    loweredEq]
              | add first second =>
                  simp [R1CS.directConstraint, R1CS.affineConstraint,
                    loweredEq]
              | mul first second =>
                  simp [R1CS.directConstraint, R1CS.affineConstraint,
                    loweredEq]
          | var index =>
              simp [R1CS.directConstraint, R1CS.affineConstraint, loweredEq]
          | const value =>
              simp [R1CS.directConstraint, R1CS.affineConstraint, loweredEq]
          | add first second =>
              simp [R1CS.directConstraint, R1CS.affineConstraint, loweredEq]
      | const value =>
          simp [R1CS.directConstraint, R1CS.affineConstraint, loweredEq]
      | add first second =>
          simp [R1CS.directConstraint, R1CS.affineConstraint, loweredEq]
      | mul first second =>
          simp [R1CS.directConstraint, R1CS.affineConstraint, loweredEq]

private theorem constraintFreshCount_eq_zero_of_affine (expression : Expr)
    (affine : R1CS.IsAffine expression) :
    R1CS.constraintFreshCount expression = 0 := by
  have notNone := directConstraint_ne_none_of_affine expression affine
  unfold R1CS.constraintFreshCount
  cases equal : R1CS.directConstraint expression with
  | none => exact False.elim (notNone equal)
  | some direct => rfl

private theorem constraintRowCount_eq_one_of_affine (expression : Expr)
    (affine : R1CS.IsAffine expression) :
    R1CS.constraintRowCount expression = 1 := by
  have notNone := directConstraint_ne_none_of_affine expression affine
  unfold R1CS.constraintRowCount
  cases equal : R1CS.directConstraint expression with
  | none => exact False.elim (notNone equal)
  | some direct => rfl

def footprint
    (interface : Formal.Interface logicalWidth degreeBound publicFits)
    (inputs : ∀ offset, InputsAffine interface offset) :
    R1CS.CircuitFootprint (Formal.statementBindingCircuit interface) where
  freshColumnCount := fun _ => 0
  physicalRowCount := fun _ => 160
  freshColumnCount_eq := by
    intro offset
    apply R1CS.totalFreshCount_eq_zero_of_noFresh
    intro expression member
    exact constraintFreshCount_eq_zero_of_affine expression
      (constraints_affine interface offset (inputs offset) expression member)
  physicalRowCount_eq := by
    intro offset
    rw [R1CS.totalRowCount_eq_length_of_rowsOne]
    · unfold Formal.statementBindingCircuit
      rw [FormalCircuit.withConstantFootprint_main]
      exact StatementBinding.flatConstraints_length
        (Formal.statementBindingInterface interface) offset
    · intro expression member
      exact constraintRowCount_eq_one_of_affine expression
        (constraints_affine interface offset (inputs offset) expression member)

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
      (Formal.statementBindingCircuit interface).main offset)) = 160 :=
  (footprint interface inputs).physicalRowCount_eq offset

end NightstreamFPrime.Layout.PiCCS.v1_1.Leaves.StatementBinding
