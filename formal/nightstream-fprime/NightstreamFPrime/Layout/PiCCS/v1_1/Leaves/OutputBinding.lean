import NightstreamFPrime.Layout.Poseidon2.Duplex
import NightstreamFPrime.Layout.R1CS.Completeness
import NightstreamFPrime.Lifecycle.PiCCS.v1_1.Completeness

/-!
Paper authority: SuperNeo v1_1, section 7.3, Steps 3 and 5, and the
Fiat--Shamir handoff to PiRLC.
Obligation: Absorb the complete 17-source `y'` family in source, Pad, matrix,
coefficient order and expose the verifier-owned outgoing transcript state.

Inputs:
- the incoming constrained Poseidon2 state;
- 17 separate `Eval_K` Pad families;
- 17 by 14 separate `Eval_A` matrix families.

Outputs:
- the outgoing constrained transcript state;
- zero-copy reduced claim views owned by the logical leaf.

Constraint groups:
- one direct R1CS row for each Duplex compiler recipe;
- no final-state or reduced-claim copy row.

Parent coverage:
- `Formal.opsAt`, child `piccs.v1_1.output_binding`.
-/

namespace NightstreamFPrime.Layout.PiCCS.v1_1.Leaves.OutputBinding

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Circuit.Quadratic
open NightstreamFPrime.Gadgets.Poseidon2
open NightstreamFPrime.Gadgets.Poseidon2.Duplex
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Lifecycle.PiCCS.v1_1
open NightstreamFPrime.Layout.Poseidon2
open NightstreamFPrime.Layout.Poseidon2.Duplex
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint

variable {logicalWidth degreeBound : Nat}
  {publicFits : ringDegree * publicRingColumns ≤
    Phi81CarrierLayout.carrierWidth logicalWidth}

/-- Stable physical wire shape for every caller-owned expression read by the
output serializer. -/
structure InputsAffine
    (interface :
      NightstreamFPrime.Lifecycle.PiCCS.v1_1.OutputBinding.Interface)
    (offset : Nat) : Prop where
  initialState : StateAffine (interface.initialState offset)
  padCoordinate : ∀ source coefficient,
    KExprAffine
      ((interface.output offset).padCoordinate source coefficient)
  matrixCoordinate : ∀ source matrix coefficient,
    KExprAffine
      ((interface.output offset).matrixCoordinate source matrix coefficient)

private theorem serializeKExpr_affine (value : KExpr)
    (affine : KExprAffine value) :
    ListAffine
      (NightstreamFPrime.Lifecycle.PiCCS.v1_1.StatementAbsorption.serializeKExpr
        value) := by
  intro expression member
  simp only [
    NightstreamFPrime.Lifecycle.PiCCS.v1_1.StatementAbsorption.serializeKExpr,
    List.mem_cons, List.not_mem_nil, or_false] at member
  rcases member with rfl | rfl
  · exact affine.1
  · exact affine.2

private theorem padWords_affine
    (interface :
      NightstreamFPrime.Lifecycle.PiCCS.v1_1.OutputBinding.Interface)
    (offset : Nat) (inputs : InputsAffine interface offset)
    (source : Fin productionShape.sourceCount) :
    ListAffine
      (NightstreamFPrime.Lifecycle.PiCCS.v1_1.OutputBinding.padWords
        (interface.output offset) source) := by
  intro expression member
  rw [NightstreamFPrime.Lifecycle.PiCCS.v1_1.OutputBinding.padWords,
    List.mem_flatMap] at member
  rcases member with ⟨coefficient, _, member⟩
  exact serializeKExpr_affine _ (inputs.padCoordinate source coefficient)
    expression member

private theorem matrixWords_affine
    (interface :
      NightstreamFPrime.Lifecycle.PiCCS.v1_1.OutputBinding.Interface)
    (offset : Nat) (inputs : InputsAffine interface offset)
    (source : Fin productionShape.sourceCount) :
    ListAffine
      (NightstreamFPrime.Lifecycle.PiCCS.v1_1.OutputBinding.matrixWords
        (interface.output offset) source) := by
  intro expression member
  rw [NightstreamFPrime.Lifecycle.PiCCS.v1_1.OutputBinding.matrixWords,
    List.mem_flatMap] at member
  rcases member with ⟨matrix, _, member⟩
  rw [List.mem_flatMap] at member
  rcases member with ⟨coefficient, _, member⟩
  exact serializeKExpr_affine _
    (inputs.matrixCoordinate source matrix coefficient) expression member

private theorem sourceWords_affine
    (interface :
      NightstreamFPrime.Lifecycle.PiCCS.v1_1.OutputBinding.Interface)
    (offset : Nat) (inputs : InputsAffine interface offset)
    (source : Fin productionShape.sourceCount) :
    ListAffine
      (NightstreamFPrime.Lifecycle.PiCCS.v1_1.OutputBinding.sourceWords
        (interface.output offset) source) := by
  intro expression member
  rw [NightstreamFPrime.Lifecycle.PiCCS.v1_1.OutputBinding.sourceWords,
    List.mem_append] at member
  rcases member with member | member
  · exact padWords_affine interface offset inputs source expression member
  · exact matrixWords_affine interface offset inputs source expression member

private theorem outputWords_affine
    (interface :
      NightstreamFPrime.Lifecycle.PiCCS.v1_1.OutputBinding.Interface)
    (offset : Nat) (inputs : InputsAffine interface offset) :
    ListAffine
      (NightstreamFPrime.Lifecycle.PiCCS.v1_1.OutputBinding.outputWords
        interface offset) := by
  intro expression member
  rw [NightstreamFPrime.Lifecycle.PiCCS.v1_1.OutputBinding.outputWords,
    List.mem_flatMap] at member
  rcases member with ⟨source, _, member⟩
  exact sourceWords_affine interface offset inputs source expression member

private theorem blockExpr_affine (words : List Expr)
    (affine : ListAffine words) :
    ListAffine
      (NightstreamFPrime.Lifecycle.PiCCS.v1_1.StatementAbsorption.blockExpr
        words) := by
  intro expression member
  simp only [
    NightstreamFPrime.Lifecycle.PiCCS.v1_1.StatementAbsorption.blockExpr,
    List.mem_cons] at member
  rcases member with rfl | member
  · exact R1CS.isAffine_const _
  · exact affine expression member

theorem actions_affine
    (interface :
      NightstreamFPrime.Lifecycle.PiCCS.v1_1.OutputBinding.Interface)
    (offset : Nat) (inputs : InputsAffine interface offset) :
    ActionsAffine
      (NightstreamFPrime.Lifecycle.PiCCS.v1_1.OutputBinding.actions
        interface offset) := by
  intro action member
  simp only [NightstreamFPrime.Lifecycle.PiCCS.v1_1.OutputBinding.actions,
    List.mem_singleton] at member
  subst action
  exact blockExpr_affine _ (outputWords_affine interface offset inputs)

private theorem core_totalFreshCount
    (interface :
      NightstreamFPrime.Lifecycle.PiCCS.v1_1.OutputBinding.Interface)
    (offset : Nat) (inputs : InputsAffine interface offset) :
    R1CS.totalFreshCount (flatConstraints (Circuit.ops
      (NightstreamFPrime.Lifecycle.PiCCS.v1_1.OutputBinding.circuit interface
        ).main offset)) = 0 := by
  change R1CS.totalFreshCount (flatConstraints
    (Formal.Owned.opsAt
      (NightstreamFPrime.Lifecycle.PiCCS.v1_1.OutputBinding.duplexInterface
        interface) offset)) = 0
  rw [Formal.Owned.flatConstraints_opsAt]
  unfold Formal.Owned.program
  rw [NightstreamFPrime.Lifecycle.PiCCS.v1_1.OutputBinding.noAssertions]
  simp only [List.append_nil]
  apply R1CS.recipeConstraints_totalFreshCount
  exact compile_recipes_direct offset (interface.initialState offset)
    (NightstreamFPrime.Lifecycle.PiCCS.v1_1.OutputBinding.actions
      interface offset) inputs.initialState
    (actions_affine interface offset inputs)

private theorem core_totalRowCount
    (interface :
      NightstreamFPrime.Lifecycle.PiCCS.v1_1.OutputBinding.Interface)
    (offset : Nat) (inputs : InputsAffine interface offset) :
    R1CS.totalRowCount (flatConstraints (Circuit.ops
      (NightstreamFPrime.Lifecycle.PiCCS.v1_1.OutputBinding.circuit interface
        ).main offset)) = 4076512 := by
  change R1CS.totalRowCount (flatConstraints
    (Formal.Owned.opsAt
      (NightstreamFPrime.Lifecycle.PiCCS.v1_1.OutputBinding.duplexInterface
        interface) offset)) = 4076512
  rw [Formal.Owned.flatConstraints_opsAt]
  unfold Formal.Owned.program
  rw [NightstreamFPrime.Lifecycle.PiCCS.v1_1.OutputBinding.noAssertions]
  simp only [List.append_nil]
  rw [R1CS.recipeConstraints_totalRowCount]
  · rw [Formal.compile_recipes_length]
    exact
      NightstreamFPrime.Lifecycle.PiCCS.v1_1.OutputBinding.recipeCount_eq
        interface offset
  · exact compile_recipes_direct offset (interface.initialState offset)
      (NightstreamFPrime.Lifecycle.PiCCS.v1_1.OutputBinding.actions
        interface offset) inputs.initialState
      (actions_affine interface offset inputs)

def footprint
    (interface : Formal.Interface logicalWidth degreeBound publicFits)
    (inputs : ∀ offset,
      InputsAffine (Formal.outputBindingInterface interface) offset) :
    R1CS.CircuitFootprint (Formal.outputBindingCircuit interface) where
  freshColumnCount := fun _ => 0
  physicalRowCount := fun _ => 4076512
  freshColumnCount_eq := by
    intro offset
    unfold Formal.outputBindingCircuit
    rw [FormalCircuit.withConstantFootprint_main]
    exact core_totalFreshCount _ offset (inputs offset)
  physicalRowCount_eq := by
    intro offset
    unfold Formal.outputBindingCircuit
    rw [FormalCircuit.withConstantFootprint_main]
    exact core_totalRowCount _ offset (inputs offset)

theorem freshColumnCount_eq
    (interface : Formal.Interface logicalWidth degreeBound publicFits)
    (inputs : ∀ offset,
      InputsAffine (Formal.outputBindingInterface interface) offset)
    (offset : Nat) :
    R1CS.totalFreshCount (flatConstraints (Circuit.ops
      (Formal.outputBindingCircuit interface).main offset)) = 0 :=
  (footprint interface inputs).freshColumnCount_eq offset

theorem physicalRowCount_eq
    (interface : Formal.Interface logicalWidth degreeBound publicFits)
    (inputs : ∀ offset,
      InputsAffine (Formal.outputBindingInterface interface) offset)
    (offset : Nat) :
    R1CS.totalRowCount (flatConstraints (Circuit.ops
      (Formal.outputBindingCircuit interface).main offset)) = 4076512 :=
  (footprint interface inputs).physicalRowCount_eq offset

theorem physicalPrivateColumnCount_eq
    (interface : Formal.Interface logicalWidth degreeBound publicFits)
    (inputs : ∀ offset,
      InputsAffine (Formal.outputBindingInterface interface) offset)
    (offset : Nat) :
    localLength (Circuit.ops (Formal.outputBindingCircuit interface).main
        offset) +
      R1CS.totalFreshCount (flatConstraints (Circuit.ops
        (Formal.outputBindingCircuit interface).main offset)) = 4076512 := by
  have logicalColumns : localLength (Circuit.ops
      (Formal.outputBindingCircuit interface).main offset) = 4076512 :=
    (Formal.outputBindingCircuit interface).privateCount_eq offset
  rw [logicalColumns, freshColumnCount_eq interface inputs offset]

def logicalConstraints
    (interface :
      NightstreamFPrime.Lifecycle.PiCCS.v1_1.OutputBinding.Interface)
    (offset : Nat) : List Expr :=
  flatConstraints (Circuit.ops
    (NightstreamFPrime.Lifecycle.PiCCS.v1_1.OutputBinding.circuit interface
      ).main offset)

def physicalRows
    (interface :
      NightstreamFPrime.Lifecycle.PiCCS.v1_1.OutputBinding.Interface)
    (offset : Nat) : List R1CS.Row :=
  (R1CS.lowerConstraints (logicalConstraints interface offset)
    (offset + 4076512)).rows

def PhysicalHolds
    (interface :
      NightstreamFPrime.Lifecycle.PiCCS.v1_1.OutputBinding.Interface)
    (offset : Nat) (env : Env) : Prop :=
  R1CS.RowsHold env (physicalRows interface offset)

private theorem logicalConstraints_varsBelow
    (interface :
      NightstreamFPrime.Lifecycle.PiCCS.v1_1.OutputBinding.Interface)
    (offset : Nat) (env : Env)
    (assumptions :
      NightstreamFPrime.Lifecycle.PiCCS.v1_1.OutputBinding.Assumptions
        interface offset env) :
    ∀ expression ∈ logicalConstraints interface offset,
      expression.VarsBelow (offset + 4076512) := by
  have scope :=
    NightstreamFPrime.Lifecycle.PiCCS.v1_1.OutputBinding.flatConstraints_varsBelow
      interface offset env assumptions
  rw [NightstreamFPrime.Lifecycle.PiCCS.v1_1.OutputBinding.localLength_eq]
    at scope
  exact scope

theorem physical_implies_spec
    (interface :
      NightstreamFPrime.Lifecycle.PiCCS.v1_1.OutputBinding.Interface)
    (offset : Nat) (env : Env)
    (assumptions :
      NightstreamFPrime.Lifecycle.PiCCS.v1_1.OutputBinding.Assumptions
        interface offset env)
    (physical : PhysicalHolds interface offset env) :
    NightstreamFPrime.Lifecycle.PiCCS.v1_1.OutputBinding.SpecHolds
      interface offset env := by
  apply NightstreamFPrime.Lifecycle.PiCCS.v1_1.OutputBinding.soundness
    interface env offset assumptions
  apply holdsFlat_implies_holds
  unfold PhysicalHolds physicalRows at physical
  exact R1CS.lowerConstraints_sound env
    (logicalConstraints interface offset) (offset + 4076512) physical

theorem physical_complete
    (interface :
      NightstreamFPrime.Lifecycle.PiCCS.v1_1.OutputBinding.Interface)
    (offset : Nat) (env : Env)
    (assumptions :
      NightstreamFPrime.Lifecycle.PiCCS.v1_1.OutputBinding.Assumptions
        interface offset env)
    (specification :
      NightstreamFPrime.Lifecycle.PiCCS.v1_1.OutputBinding.SpecHolds
        interface offset env) :
    ∃ completed,
      AgreesOutside env completed offset
          (4076512 + R1CS.totalFreshCount
            (logicalConstraints interface offset)) ∧
        PhysicalHolds interface offset completed := by
  rcases NightstreamFPrime.Lifecycle.PiCCS.v1_1.OutputBinding.completeness
      interface env offset assumptions specification with
    ⟨logicalEnv, logicalAgrees, logicalRows⟩
  have lengthEq : localLength (Circuit.ops
      (NightstreamFPrime.Lifecycle.PiCCS.v1_1.OutputBinding.circuit interface
        ).main offset) = 4076512 :=
    NightstreamFPrime.Lifecycle.PiCCS.v1_1.OutputBinding.localLength_eq
      interface offset
  have logicalAgreesFixed :
      AgreesOutside env logicalEnv offset 4076512 := by
    rw [lengthEq] at logicalAgrees
    exact logicalAgrees
  have logicalAssumptions :
      NightstreamFPrime.Lifecycle.PiCCS.v1_1.OutputBinding.Assumptions
        interface offset logicalEnv := assumptions
  have scope := logicalConstraints_varsBelow interface offset logicalEnv
    logicalAssumptions
  have logicalHolds :
      ConstraintsHold logicalEnv (logicalConstraints interface offset) :=
    logicalRows
  rcases R1CS.lowerConstraints_complete logicalEnv
      (logicalConstraints interface offset) (offset + 4076512) scope logicalHolds
      with ⟨completed, physicalAgrees, physicalRowsHold⟩
  refine ⟨completed, logicalAgreesFixed.append physicalAgrees, ?_⟩
  exact physicalRowsHold

end NightstreamFPrime.Layout.PiCCS.v1_1.Leaves.OutputBinding
