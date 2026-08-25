import NightstreamFPrime.Layout.Polynomial.Horner
import NightstreamFPrime.Lifecycle.PiCCS.v1_1.Completeness

/-!
Paper authority: SuperNeo v1_1, section 7.3, claimed SumCheck total `T`.
Obligation: Evaluate the separate `Eval_K`, then `Eval_A`, coefficient sequence
at `γ` and expose the verifier-owned initial SumCheck claim.

Inputs:
- verifier-derived `γ`;
- 864 `Eval_K` coefficients;
- 12,096 `Eval_A` coefficients.

Outputs:
- the child-owned initial SumCheck claim.

Constraint groups:
- one reusable quadratic-extension Horner chain over 12,960 coefficients;
- no expected-output or boundary-copy row.

Parent coverage:
- `Formal.opsAt`, child `piccs.v1_1.initial_claim`.
-/

namespace NightstreamFPrime.Layout.PiCCS.v1_1.Leaves.InitialClaim

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Circuit.Quadratic
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Lifecycle.PiCCS.v1_1
open NightstreamFPrime.Lifecycle.PiCCS.v1_1.InitialClaim
open NightstreamFPrime.Layout.Polynomial.Horner
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint

variable {logicalWidth degreeBound : Nat}
  {publicFits : ringDegree * publicRingColumns ≤
    Phi81CarrierLayout.carrierWidth logicalWidth}

/-- Stable physical wire shape for the point and both separate v1_1
coefficient families. -/
structure InputsLinear
    (interface :
      NightstreamFPrime.Lifecycle.PiCCS.v1_1.InitialClaim.Interface)
    (offset : Nat) : Prop where
  gamma : KExprLinear (interface.gamma offset)
  eval_K : ∀ coordinate,
    KExprLinear (interface.eval_K offset coordinate)
  eval_A : ∀ coordinate,
    KExprLinear (interface.eval_A offset coordinate)

theorem coefficientExprs_linear
    (interface :
      NightstreamFPrime.Lifecycle.PiCCS.v1_1.InitialClaim.Interface)
    (offset : Nat) (inputs : InputsLinear interface offset) :
    ∀ coefficient ∈ coefficientExprs interface offset,
      KExprLinear coefficient := by
  intro coefficient member
  rw [coefficientExprs, List.mem_append] at member
  rcases member with member | member
  · rw [List.mem_map] at member
    rcases member with ⟨coordinate, _, rfl⟩
    exact inputs.eval_K coordinate
  · rw [List.mem_map] at member
    rcases member with ⟨coordinate, _, rfl⟩
    exact inputs.eval_A coordinate

/-- The nonempty Horner child exports one materialized linear claim. -/
theorem output_linear
    (interface :
      NightstreamFPrime.Lifecycle.PiCCS.v1_1.InitialClaim.Interface)
    (offset : Nat) (inputs : InputsLinear interface offset) :
    KExprLinear
      (NightstreamFPrime.Lifecycle.PiCCS.v1_1.InitialClaim.output
        interface offset) := by
  unfold NightstreamFPrime.Lifecycle.PiCCS.v1_1.InitialClaim.output
    NightstreamFPrime.Gadgets.Polynomial.Horner.Owned.output
    NightstreamFPrime.Gadgets.Polynomial.Horner.Owned.program
    NightstreamFPrime.Lifecycle.PiCCS.v1_1.InitialClaim.ownedInterface
  apply compile_output_linear
  · intro empty
    have coefficientExprsEmpty : coefficientExprs interface offset = [] := by
      simpa [NightstreamFPrime.Lifecycle.PiCCS.v1_1.InitialClaim.ownedInterface]
        using empty
    have length := coefficientExprs_length interface offset
    rw [coefficientExprsEmpty] at length
    simp at length
  · exact coefficientExprs_linear interface offset inputs

/-- The owned initial claim lies below the canonical SumCheck child start. -/
theorem output_varsBelow_sumcheck
    (interface : Formal.Interface logicalWidth degreeBound publicFits)
    (parentOffset : Nat) (env : Env)
    (assumptions :
      (Formal.initialClaimCircuit (Formal.atOffset interface parentOffset)
        ).assumptions (Formal.initialClaimOffset interface parentOffset) env) :
    (Formal.initialClaimOutput (Formal.atOffset interface parentOffset)
      (Formal.sumcheckOffset interface parentOffset)).VarsBelow
        (Formal.sumcheckOffset interface parentOffset) := by
  let frozen := Formal.atOffset interface parentOffset
  have childAssumptions : InitialClaim.Assumptions
      (Formal.initialClaimInterface frozen) (Formal.initialClaimStart frozen)
      env := by
    rw [Formal.initialClaimStart_atOffset interface parentOffset]
    exact assumptions
  have below := NightstreamFPrime.Gadgets.Polynomial.Horner.Owned.output_varsBelow
    (InitialClaim.ownedInterface (Formal.initialClaimInterface frozen))
    (Formal.initialClaimStart frozen) env childAssumptions
  have outputEq :
      Formal.initialClaimOutput frozen
          (Formal.sumcheckOffset interface parentOffset) =
        InitialClaim.output (Formal.initialClaimInterface frozen)
          (Formal.initialClaimStart frozen) := by
    rfl
  have boundEq : Formal.sumcheckOffset interface parentOffset =
      Formal.initialClaimStart frozen +
        localLength (Circuit.ops
          (InitialClaim.circuit (Formal.initialClaimInterface frozen)).main
          (Formal.initialClaimStart frozen)) := by
    calc
      Formal.sumcheckOffset interface parentOffset =
          Formal.sumcheckStart frozen :=
        (Formal.sumcheckStart_atOffset interface parentOffset).symm
      _ = _ := by
        unfold Formal.sumcheckStart
        rw [InitialClaim.localLength_eq]
        rfl
  rw [outputEq, boundEq]
  exact below

private theorem flatConstraints_eq_recipeConstraints
    (interface :
      NightstreamFPrime.Lifecycle.PiCCS.v1_1.InitialClaim.Interface)
    (offset : Nat) :
    flatConstraints (Circuit.ops (circuit interface).main offset) =
      recipeConstraints offset (program interface offset).recipes := by
  unfold circuit
  rw [NightstreamFPrime.Gadgets.Polynomial.Horner.Owned.circuit_ops,
    NightstreamFPrime.Gadgets.Polynomial.Horner.Owned.flatConstraints_opsAt]
  rfl

private theorem program_totalFreshCount
    (interface :
      NightstreamFPrime.Lifecycle.PiCCS.v1_1.InitialClaim.Interface)
    (offset : Nat) (inputs : InputsLinear interface offset) :
    R1CS.totalFreshCount
      (recipeConstraints offset (program interface offset).recipes) =
        90713 := by
  calc
    _ = 7 * ((coefficientExprs interface offset).length - 1) := by
      simpa only [program, ownedInterface,
        NightstreamFPrime.Gadgets.Polynomial.Horner.Owned.program] using
        compile_totalFreshCount offset (interface.gamma offset)
          (coefficientExprs interface offset) inputs.gamma
          (coefficientExprs_linear interface offset inputs)
    _ = 90713 := by
      rw [coefficientExprs_length]

private theorem program_totalRowCount
    (interface :
      NightstreamFPrime.Lifecycle.PiCCS.v1_1.InitialClaim.Interface)
    (offset : Nat) (inputs : InputsLinear interface offset) :
    R1CS.totalRowCount
      (recipeConstraints offset (program interface offset).recipes) =
        116631 := by
  calc
    _ = 9 * ((coefficientExprs interface offset).length - 1) := by
      simpa only [program, ownedInterface,
        NightstreamFPrime.Gadgets.Polynomial.Horner.Owned.program] using
        compile_totalRowCount offset (interface.gamma offset)
          (coefficientExprs interface offset) inputs.gamma
          (coefficientExprs_linear interface offset inputs)
    _ = 116631 := by
      rw [coefficientExprs_length]

/-- Exact parent-facing physical footprint for the complete 12,960-term
initial-claim Horner chain. -/
def footprint
    (interface : Formal.Interface logicalWidth degreeBound publicFits)
    (inputs : ∀ offset,
      InputsLinear (Formal.initialClaimInterface interface) offset) :
    R1CS.CircuitFootprint (Formal.initialClaimCircuit interface) where
  freshColumnCount := fun _ => 90713
  physicalRowCount := fun _ => 116631
  freshColumnCount_eq := by
    intro offset
    unfold Formal.initialClaimCircuit
    rw [FormalCircuit.withConstantFootprint_main]
    rw [flatConstraints_eq_recipeConstraints]
    exact program_totalFreshCount _ offset (inputs offset)
  physicalRowCount_eq := by
    intro offset
    unfold Formal.initialClaimCircuit
    rw [FormalCircuit.withConstantFootprint_main]
    rw [flatConstraints_eq_recipeConstraints]
    exact program_totalRowCount _ offset (inputs offset)

theorem freshColumnCount_eq
    (interface : Formal.Interface logicalWidth degreeBound publicFits)
    (inputs : ∀ offset,
      InputsLinear (Formal.initialClaimInterface interface) offset)
    (offset : Nat) :
    R1CS.totalFreshCount (flatConstraints (Circuit.ops
      (Formal.initialClaimCircuit interface).main offset)) = 90713 :=
  (footprint interface inputs).freshColumnCount_eq offset

theorem physicalRowCount_eq
    (interface : Formal.Interface logicalWidth degreeBound publicFits)
    (inputs : ∀ offset,
      InputsLinear (Formal.initialClaimInterface interface) offset)
    (offset : Nat) :
    R1CS.totalRowCount (flatConstraints (Circuit.ops
      (Formal.initialClaimCircuit interface).main offset)) = 116631 :=
  (footprint interface inputs).physicalRowCount_eq offset

theorem physicalPrivateColumnCount_eq
    (interface : Formal.Interface logicalWidth degreeBound publicFits)
    (inputs : ∀ offset,
      InputsLinear (Formal.initialClaimInterface interface) offset)
    (offset : Nat) :
    localLength (Circuit.ops (Formal.initialClaimCircuit interface).main
        offset) +
      R1CS.totalFreshCount (flatConstraints (Circuit.ops
        (Formal.initialClaimCircuit interface).main offset)) = 116631 := by
  have logicalColumns :
      localLength (Circuit.ops (Formal.initialClaimCircuit interface).main
        offset) = 25918 := by
    unfold Formal.initialClaimCircuit
    rw [FormalCircuit.withConstantFootprint_main]
    exact NightstreamFPrime.Lifecycle.PiCCS.v1_1.InitialClaim.localLength_eq
      (Formal.initialClaimInterface interface) offset
  rw [logicalColumns, freshColumnCount_eq interface inputs offset]

end NightstreamFPrime.Layout.PiCCS.v1_1.Leaves.InitialClaim
