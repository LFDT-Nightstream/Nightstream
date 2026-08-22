import NightstreamFPrime.Layout.Poseidon2.Duplex
import NightstreamFPrime.Lifecycle.PiCCS.v1_1.Completeness

/-!
Paper authority: SuperNeo v1_1, section 7.3, PiCCS transcript initialization.
Obligation: Absorb the exact public statement and verifier-owned v1_1 claims.

Inputs:
- the parent PiCCS running and fresh claims;
- separate `Eval_K` and `Eval_A` evaluation families.

Outputs:
- the child-owned Poseidon2 transcript state;
- the exact physical footprint of the parent-facing child.

Constraint groups:
- one direct R1CS row for each Poseidon2 compiler recipe;
- no assertion rows and no fresh lowering columns.

Parent coverage:
- `Formal.opsAt`, child `piccs.v1_1.statement_absorption`.
-/

namespace NightstreamFPrime.Layout.PiCCS.v1_1.Leaves.StatementAbsorption

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Circuit.Quadratic
open NightstreamFPrime.Gadgets.Poseidon2
open NightstreamFPrime.Gadgets.Poseidon2.Duplex
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Lifecycle.PiCCS.v1_1
open NightstreamFPrime.Lifecycle.PiCCS.v1_1.StatementAbsorption
open NightstreamFPrime.Layout.Poseidon2
open NightstreamFPrime.Layout.Poseidon2.Duplex
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint

variable {logicalWidth degreeBound : Nat}
  {publicFits : ringDegree * publicRingColumns ≤
    Phi81CarrierLayout.carrierWidth logicalWidth}

/-- Affine physical-input premise for the seven caller-owned expression
families read by the Statement-absorption serializer. -/
structure InputsAffine
    (interface :
      NightstreamFPrime.Lifecycle.PiCCS.v1_1.StatementAbsorption.Interface
        logicalWidth publicFits)
    (offset : Nat) : Prop where
  runningPoint : ∀ coordinate,
    KExprAffine ((interface.running offset).point coordinate)
  runningCommitment : ∀ source row coefficient,
    R1CS.IsAffine
      ((interface.running offset).commitment source row coefficient)
  runningPublicInput : ∀ source column,
    R1CS.IsAffine ((interface.running offset).publicInput source column)
  runningEval_K : ∀ source coefficient,
    KExprAffine
      (((interface.running offset).evaluation source).eval_K coefficient)
  runningEval_A : ∀ source matrix coefficient,
    KExprAffine
      (((interface.running offset).evaluation source).eval_A matrix coefficient)
  freshCommitment : ∀ source row coefficient,
    R1CS.IsAffine
      ((interface.fresh offset).commitment source row coefficient)
  freshPublicInput : ∀ source column,
    R1CS.IsAffine ((interface.fresh offset).publicInput source column)

private theorem serializeKExpr_affine (value : KExpr)
    (affine : KExprAffine value) :
    ListAffine (serializeKExpr value) := by
  intro expression member
  simp only [serializeKExpr, List.mem_cons, List.not_mem_nil,
    or_false] at member
  rcases member with rfl | rfl
  · exact affine.1
  · exact affine.2

private theorem serializePointExpr_affine
    (point : Fin productionShape.cubeVariables → KExpr)
    (affine : ∀ coordinate, KExprAffine (point coordinate)) :
    ListAffine (serializePointExpr point) := by
  intro expression member
  rw [serializePointExpr, List.mem_flatMap] at member
  rcases member with ⟨coordinate, _, member⟩
  exact serializeKExpr_affine (point coordinate) (affine coordinate)
    expression member

private theorem serializeCommitmentExpr_affine
    (commitment : Fin productionProfile.commitmentWidth →
      Fin ringDegree → Expr)
    (affine : ∀ row coefficient, R1CS.IsAffine (commitment row coefficient)) :
    ListAffine (serializeCommitmentExpr commitment) := by
  intro expression member
  rw [serializeCommitmentExpr, List.mem_flatMap] at member
  rcases member with ⟨row, _, member⟩
  rw [List.mem_map] at member
  rcases member with ⟨coefficient, _, rfl⟩
  exact affine row coefficient

private theorem serializePublicInputExpr_affine
    (input : Fin (FullShape logicalWidth publicFits).publicWidth → Expr)
    (affine : ∀ column, R1CS.IsAffine (input column)) :
    ListAffine (serializePublicInputExpr input) := by
  intro expression member
  rw [serializePublicInputExpr, List.mem_map] at member
  rcases member with ⟨column, _, rfl⟩
  exact affine column

private theorem serializeEvaluationExpr_affine
    (evaluation : EvaluationExpr)
    (padAffine : ∀ coefficient,
      KExprAffine (evaluation.eval_K coefficient))
    (matrixAffine : ∀ matrix coefficient,
      KExprAffine (evaluation.eval_A matrix coefficient)) :
    ListAffine (serializeEvaluationExpr evaluation) := by
  intro expression member
  rw [serializeEvaluationExpr, List.mem_append] at member
  rcases member with member | member
  · rw [List.mem_flatMap] at member
    rcases member with ⟨coefficient, _, member⟩
    exact serializeKExpr_affine _ (padAffine coefficient) expression member
  · rw [List.mem_flatMap] at member
    rcases member with ⟨matrix, _, member⟩
    rw [List.mem_flatMap] at member
    rcases member with ⟨coefficient, _, member⟩
    exact serializeKExpr_affine _ (matrixAffine matrix coefficient)
      expression member

private theorem constantWords_affine (words : List F) :
    ListAffine (constantWords words) := by
  intro expression member
  rw [constantWords, List.mem_map] at member
  rcases member with ⟨word, _, rfl⟩
  exact R1CS.isAffine_const word

private theorem blockExpr_affine (words : List Expr)
    (affine : ListAffine words) : ListAffine (blockExpr words) := by
  intro expression member
  simp only [blockExpr, List.mem_cons] at member
  rcases member with rfl | member
  · exact R1CS.isAffine_const _
  · exact affine expression member

private theorem BlocksAffine.append {first second : List (List Expr)}
    (firstAffine : BlocksAffine first)
    (secondAffine : BlocksAffine second) : BlocksAffine (first ++ second) := by
  intro block member
  rcases List.mem_append.mp member with member | member
  · exact firstAffine block member
  · exact secondAffine block member

private theorem BlocksAffine.flatMap
    {Index : Type} (indices : List Index)
    (blocks : Index → List (List Expr))
    (affine : ∀ index ∈ indices, BlocksAffine (blocks index)) :
    BlocksAffine (indices.flatMap blocks) := by
  intro block member
  rw [List.mem_flatMap] at member
  rcases member with ⟨index, indexMember, blockMember⟩
  exact affine index indexMember block blockMember

private theorem publicInputBlocks_affine
    (interface :
      NightstreamFPrime.Lifecycle.PiCCS.v1_1.StatementAbsorption.Interface
        logicalWidth publicFits)
    (offset : Nat) (inputs : InputsAffine interface offset) :
    BlocksAffine (publicInputBlocks interface offset) := by
  let running := interface.running offset
  let fresh := interface.fresh offset
  apply BlocksAffine.append
  · apply BlocksAffine.append
    · intro block member
      simp only [List.mem_singleton] at member
      subst block
      exact serializePointExpr_affine running.point inputs.runningPoint
    · apply BlocksAffine.flatMap
      intro index _ block member
      simp only [List.mem_cons, List.not_mem_nil, or_false] at member
      rcases member with rfl | rfl | rfl
      · exact serializeCommitmentExpr_affine (running.commitment index)
          (inputs.runningCommitment index)
      · exact serializePublicInputExpr_affine (running.publicInput index)
          (inputs.runningPublicInput index)
      · exact serializeEvaluationExpr_affine (running.evaluation index)
          (inputs.runningEval_K index) (inputs.runningEval_A index)
  · apply BlocksAffine.flatMap
    intro index _ block member
    simp only [List.mem_cons, List.not_mem_nil, or_false] at member
    rcases member with rfl | rfl
    · exact serializeCommitmentExpr_affine (fresh.commitment index)
        (inputs.freshCommitment index)
    · exact serializePublicInputExpr_affine (fresh.publicInput index)
        (inputs.freshPublicInput index)

private theorem verifierClaimWords_affine
    (interface :
      NightstreamFPrime.Lifecycle.PiCCS.v1_1.StatementAbsorption.Interface
        logicalWidth publicFits)
    (offset : Nat) (inputs : InputsAffine interface offset) :
    ListAffine (verifierClaimWords interface offset) := by
  intro expression member
  rw [verifierClaimWords, List.mem_append] at member
  rcases member with member | member
  · rw [List.mem_flatMap] at member
    rcases member with ⟨coordinate, _, member⟩
    exact serializeKExpr_affine _
      (inputs.runningEval_K coordinate.running coordinate.coefficient)
      expression member
  · rw [List.mem_flatMap] at member
    rcases member with ⟨coordinate, _, member⟩
    exact serializeKExpr_affine _
      (inputs.runningEval_A coordinate.running coordinate.matrix
        coordinate.coefficient) expression member

private theorem verifierInputBlocks_affine
    (interface :
      NightstreamFPrime.Lifecycle.PiCCS.v1_1.StatementAbsorption.Interface
        logicalWidth publicFits)
    (offset : Nat) (inputs : InputsAffine interface offset) :
    BlocksAffine (verifierInputBlocks interface offset) := by
  intro block member
  simp only [verifierInputBlocks, List.mem_cons, List.not_mem_nil,
    or_false] at member
  rcases member with rfl | rfl
  · exact serializePointExpr_affine _ inputs.runningPoint
  · exact verifierClaimWords_affine interface offset inputs

private theorem mappedBlocks_affine (blocks : List (List Expr))
    (blocksAffine : BlocksAffine blocks) :
    ActionsAffine (blocks.map absorbBlock) := by
  intro action member
  rw [List.mem_map] at member
  rcases member with ⟨block, blockMember, rfl⟩
  exact blockExpr_affine block (blocksAffine block blockMember)

theorem actions_affine
    (interface :
      NightstreamFPrime.Lifecycle.PiCCS.v1_1.StatementAbsorption.Interface
        logicalWidth publicFits)
    (offset : Nat) (inputs : InputsAffine interface offset) :
    ActionsAffine (actions interface offset) := by
  unfold actions publicInputActions verifierInputActions
  apply ActionsAffine.append
  · apply ActionsAffine.append
    · apply ActionsAffine.cons
      · exact constantWords_affine _
      · intro action member
        simp at member
    · exact mappedBlocks_affine _
        (publicInputBlocks_affine interface offset inputs)
  · exact mappedBlocks_affine _
      (verifierInputBlocks_affine interface offset inputs)

/-- Exact parent-facing physical footprint, conditional only on the declared
affine form of caller-owned symbolic inputs. -/
def footprint
    (interface : Formal.Interface logicalWidth degreeBound publicFits)
    (inputs : ∀ offset,
      InputsAffine (Formal.statementAbsorptionInterface interface) offset) :
    R1CS.CircuitFootprint (Formal.statementAbsorptionCircuit interface) where
  freshColumnCount := fun _ => 0
  physicalRowCount := fun _ => 10298432
  freshColumnCount_eq := by
    intro offset
    unfold Formal.statementAbsorptionCircuit
    rw [FormalCircuit.withConstantFootprint_main]
    change R1CS.totalFreshCount (flatConstraints
      (opsAt (Formal.statementAbsorptionInterface interface) offset)) = 0
    rw [NightstreamFPrime.Lifecycle.PiCCS.v1_1.StatementAbsorption.flatConstraints_opsAt]
    apply R1CS.recipeConstraints_totalFreshCount
    exact compile_recipes_direct offset Hash.zeroE
      (actions (Formal.statementAbsorptionInterface interface) offset)
      zeroE_affine
      (actions_affine _ offset (inputs offset))
  physicalRowCount_eq := by
    intro offset
    unfold Formal.statementAbsorptionCircuit
    rw [FormalCircuit.withConstantFootprint_main]
    change R1CS.totalRowCount (flatConstraints
      (opsAt (Formal.statementAbsorptionInterface interface) offset)) =
        10298432
    rw [NightstreamFPrime.Lifecycle.PiCCS.v1_1.StatementAbsorption.flatConstraints_opsAt]
    rw [R1CS.recipeConstraints_totalRowCount]
    exact NightstreamFPrime.Lifecycle.PiCCS.v1_1.StatementAbsorption.program_recipes_length
      (Formal.statementAbsorptionInterface interface) offset
    exact compile_recipes_direct offset Hash.zeroE
      (actions (Formal.statementAbsorptionInterface interface) offset)
      zeroE_affine
      (actions_affine _ offset (inputs offset))

theorem freshColumnCount_eq
    (interface : Formal.Interface logicalWidth degreeBound publicFits)
    (inputs : ∀ offset,
      InputsAffine (Formal.statementAbsorptionInterface interface) offset)
    (offset : Nat) :
    R1CS.totalFreshCount (flatConstraints (Circuit.ops
      (Formal.statementAbsorptionCircuit interface).main offset)) = 0 :=
  (footprint interface inputs).freshColumnCount_eq offset

theorem physicalRowCount_eq
    (interface : Formal.Interface logicalWidth degreeBound publicFits)
    (inputs : ∀ offset,
      InputsAffine (Formal.statementAbsorptionInterface interface) offset)
    (offset : Nat) :
    R1CS.totalRowCount (flatConstraints (Circuit.ops
      (Formal.statementAbsorptionCircuit interface).main offset)) =
        10298432 :=
  (footprint interface inputs).physicalRowCount_eq offset

end NightstreamFPrime.Layout.PiCCS.v1_1.Leaves.StatementAbsorption
