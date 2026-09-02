import NightstreamFPrime.Layout.Poseidon2.Duplex
import NightstreamFPrime.Lifecycle.PiCCS.v1_1.Completeness

/-!
Paper authority: SuperNeo v1_1, section 7.3, PiCCS transcript initialization.
Obligation: Absorb the pilot-bound prior digest and exact fresh claim.

Inputs:
- the prior digest projected from the fresh public input;
- the fresh commitment and public input.

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

/-- Affine physical-input premise for the two fresh expression families read
by the digest-only statement serializer. -/
structure InputsAffine
    (interface :
      NightstreamFPrime.Lifecycle.PiCCS.v1_1.StatementAbsorption.Interface
        logicalWidth publicFits)
    (offset : Nat) : Prop where
  freshCommitment : ∀ source row coefficient,
    R1CS.IsAffine
      ((interface.fresh offset).commitment source row coefficient)
  freshPublicInput : ∀ source column,
    R1CS.IsAffine ((interface.fresh offset).publicInput source column)

/-- Range premise for the same two caller-owned expression families. -/
structure InputsBelow
    (interface :
      NightstreamFPrime.Lifecycle.PiCCS.v1_1.StatementAbsorption.Interface
        logicalWidth publicFits)
    (offset : Nat) : Prop where
  freshCommitment : ∀ source row coefficient,
    ((interface.fresh offset).commitment source row coefficient).VarsBelow
      offset
  freshPublicInput : ∀ source column,
    ((interface.fresh offset).publicInput source column).VarsBelow offset

/-- The first domain-tag absorb establishes one permutation-owned state block,
which every later statement absorb preserves. -/
theorem finalState_fresh
    (interface :
      NightstreamFPrime.Lifecycle.PiCCS.v1_1.StatementAbsorption.Interface
        logicalWidth publicFits)
    (offset : Nat) :
    StateFresh
      (NightstreamFPrime.Lifecycle.PiCCS.v1_1.StatementAbsorption.finalState
        interface offset) := by
  unfold NightstreamFPrime.Lifecycle.PiCCS.v1_1.StatementAbsorption.finalState
    NightstreamFPrime.Lifecycle.PiCCS.v1_1.StatementAbsorption.program
    NightstreamFPrime.Lifecycle.PiCCS.v1_1.StatementAbsorption.actions
    NightstreamFPrime.Lifecycle.PiCCS.v1_1.StatementAbsorption.publicInputActions
  change StateFresh
    (Formal.compile offset Hash.zeroE
      (.absorb (constantWords
        NightstreamFPrime.Lifecycle.Transcript.piCcsDigestDomainTag) ::
        (publicInputBlocks interface offset).map absorbBlock)).output
  apply compile_output_fresh_of_head_absorb
  intro empty
  have lengthZero := congrArg List.length empty
  simp [Hash.inputChunks, constantWords,
    NightstreamFPrime.Lifecycle.Transcript.piCcsDigestDomainTag_length,
    Spec.Poseidon2.rate] at lengthZero

theorem finalState_affine
    (interface :
      NightstreamFPrime.Lifecycle.PiCCS.v1_1.StatementAbsorption.Interface
        logicalWidth publicFits)
    (offset : Nat) :
    StateAffine
      (NightstreamFPrime.Lifecycle.PiCCS.v1_1.StatementAbsorption.finalState
        interface offset) :=
  (finalState_fresh interface offset).affine

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

private theorem foldl_weighted_affine (indices : List Nat)
    (coefficient : Nat → F) (bit : Nat → Expr)
    (bitsAffine : ∀ index, R1CS.IsAffine (bit index))
    (initial : Expr) (initialAffine : R1CS.IsAffine initial) :
    R1CS.IsAffine (indices.foldl (fun value index =>
      value + Expr.const (coefficient index) * bit index) initial) := by
  induction indices generalizing initial with
  | nil => exact initialAffine
  | cons index rest inductionHypothesis =>
      apply inductionHypothesis
      exact R1CS.IsAffine.add initialAffine
        (R1CS.IsAffine.const_mul _ (bitsAffine index))

private theorem decodeHashWordExpr_affine
    (interface :
      NightstreamFPrime.Lifecycle.PiCCS.v1_1.StatementAbsorption.Interface
        logicalWidth publicFits)
    (offset : Nat) (inputs : InputsAffine interface offset) (word : Fin 4) :
    R1CS.IsAffine (decodeHashWordExpr
      ((interface.fresh offset).publicInput ⟨0, by decide⟩) word) := by
  unfold decodeHashWordExpr
  exact foldl_weighted_affine _ _ _
    (fun bit => inputs.freshPublicInput ⟨0, by decide⟩
      (NightstreamFPrime.Lifecycle.digestBitIndexNat
        (logicalWidth := logicalWidth) word bit))
    0 (R1CS.isAffine_const _)

private theorem priorDigestExpr_affine
    (interface :
      NightstreamFPrime.Lifecycle.PiCCS.v1_1.StatementAbsorption.Interface
        logicalWidth publicFits)
    (offset : Nat) (inputs : InputsAffine interface offset) :
    ListAffine (priorDigestExpr interface offset) := by
  intro expression member
  rw [priorDigestExpr, List.mem_ofFn'] at member
  rcases member with ⟨lane, rfl⟩
  exact decodeHashWordExpr_affine interface offset inputs lane

private theorem publicInputBlocks_affine
    (interface :
      NightstreamFPrime.Lifecycle.PiCCS.v1_1.StatementAbsorption.Interface
        logicalWidth publicFits)
    (offset : Nat) (inputs : InputsAffine interface offset) :
    BlocksAffine (publicInputBlocks interface offset) := by
  let fresh := interface.fresh offset
  apply BlocksAffine.append
  · intro block member
    simp only [List.mem_singleton] at member
    subst block
    exact priorDigestExpr_affine interface offset inputs
  · apply BlocksAffine.flatMap
    intro index _ block member
    simp only [List.mem_cons, List.not_mem_nil, or_false] at member
    rcases member with rfl | rfl
    · exact serializeCommitmentExpr_affine (fresh.commitment index)
        (inputs.freshCommitment index)
    · exact serializePublicInputExpr_affine (fresh.publicInput index)
        (inputs.freshPublicInput index)

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
  unfold actions publicInputActions
  apply ActionsAffine.append
  · apply ActionsAffine.cons
    · exact constantWords_affine _
    · intro action member
      simp at member
  · exact mappedBlocks_affine _
      (publicInputBlocks_affine interface offset inputs)

private def ListBelow (bound : Nat) (values : List Expr) : Prop :=
  ∀ expression ∈ values, expression.VarsBelow bound

private def BlocksBelow (bound : Nat) (blocks : List (List Expr)) : Prop :=
  ∀ block ∈ blocks, ListBelow bound block

private theorem serializeKExpr_below (bound : Nat) (value : KExpr)
    (below : value.VarsBelow bound) : ListBelow bound (serializeKExpr value) := by
  intro expression member
  simp only [serializeKExpr, List.mem_cons, List.not_mem_nil,
    or_false] at member
  rcases member with rfl | rfl
  · exact below.1
  · exact below.2

private theorem serializePointExpr_below (bound : Nat)
    (point : Fin productionShape.cubeVariables → KExpr)
    (below : ∀ coordinate, (point coordinate).VarsBelow bound) :
    ListBelow bound (serializePointExpr point) := by
  intro expression member
  rw [serializePointExpr, List.mem_flatMap] at member
  rcases member with ⟨coordinate, _, expressionMember⟩
  exact serializeKExpr_below bound (point coordinate) (below coordinate)
    expression expressionMember

private theorem serializeCommitmentExpr_below (bound : Nat)
    (commitment : Fin productionProfile.commitmentWidth →
      Fin ringDegree → Expr)
    (below : ∀ row coefficient, (commitment row coefficient).VarsBelow bound) :
    ListBelow bound (serializeCommitmentExpr commitment) := by
  intro expression member
  rw [serializeCommitmentExpr, List.mem_flatMap] at member
  rcases member with ⟨row, _, member⟩
  rw [List.mem_map] at member
  rcases member with ⟨coefficient, _, rfl⟩
  exact below row coefficient

private theorem serializePublicInputExpr_below (bound : Nat)
    (input : Fin (FullShape logicalWidth publicFits).publicWidth → Expr)
    (below : ∀ column, (input column).VarsBelow bound) :
    ListBelow bound (serializePublicInputExpr input) := by
  intro expression member
  rw [serializePublicInputExpr, List.mem_map] at member
  rcases member with ⟨column, _, rfl⟩
  exact below column

private theorem serializeEvaluationExpr_below (bound : Nat)
    (evaluation : EvaluationExpr)
    (padBelow : ∀ coefficient,
      (evaluation.eval_K coefficient).VarsBelow bound)
    (matrixBelow : ∀ matrix coefficient,
      (evaluation.eval_A matrix coefficient).VarsBelow bound) :
    ListBelow bound (serializeEvaluationExpr evaluation) := by
  intro expression member
  rw [serializeEvaluationExpr, List.mem_append] at member
  rcases member with member | member
  · rw [List.mem_flatMap] at member
    rcases member with ⟨coefficient, _, expressionMember⟩
    exact serializeKExpr_below bound _ (padBelow coefficient) expression
      expressionMember
  · rw [List.mem_flatMap] at member
    rcases member with ⟨matrix, _, member⟩
    rw [List.mem_flatMap] at member
    rcases member with ⟨coefficient, _, expressionMember⟩
    exact serializeKExpr_below bound _ (matrixBelow matrix coefficient)
      expression expressionMember

private theorem constantWords_below (bound : Nat) (words : List F) :
    ListBelow bound (constantWords words) := by
  intro expression member
  rw [constantWords, List.mem_map] at member
  rcases member with ⟨word, _, rfl⟩
  trivial

private theorem blockExpr_below (bound : Nat) (words : List Expr)
    (below : ListBelow bound words) : ListBelow bound (blockExpr words) := by
  intro expression member
  simp only [blockExpr, List.mem_cons] at member
  rcases member with rfl | member
  · trivial
  · exact below expression member

private theorem BlocksBelow.append {bound : Nat}
    {first second : List (List Expr)}
    (firstBelow : BlocksBelow bound first)
    (secondBelow : BlocksBelow bound second) :
    BlocksBelow bound (first ++ second) := by
  intro block member
  rcases List.mem_append.mp member with member | member
  · exact firstBelow block member
  · exact secondBelow block member

private theorem BlocksBelow.flatMap {bound : Nat}
    {Index : Type} (indices : List Index)
    (blocks : Index → List (List Expr))
    (below : ∀ index ∈ indices, BlocksBelow bound (blocks index)) :
    BlocksBelow bound (indices.flatMap blocks) := by
  intro block member
  rw [List.mem_flatMap] at member
  rcases member with ⟨index, indexMember, blockMember⟩
  exact below index indexMember block blockMember

private theorem foldl_weighted_below (bound : Nat) (indices : List Nat)
    (coefficient : Nat → F) (bit : Nat → Expr)
    (bitsBelow : ∀ index, (bit index).VarsBelow bound)
    (initial : Expr) (initialBelow : initial.VarsBelow bound) :
    (indices.foldl (fun value index =>
      value + Expr.const (coefficient index) * bit index) initial).VarsBelow
        bound := by
  induction indices generalizing initial with
  | nil => exact initialBelow
  | cons index rest inductionHypothesis =>
      apply inductionHypothesis
      exact Expr.VarsBelow.add _ _ _ initialBelow
        (Expr.VarsBelow.mul _ _ _ (by simp [Expr.VarsBelow])
          (bitsBelow index))

private theorem decodeHashWordExpr_below
    (interface :
      NightstreamFPrime.Lifecycle.PiCCS.v1_1.StatementAbsorption.Interface
        logicalWidth publicFits)
    (offset : Nat) (inputs : InputsBelow interface offset) (word : Fin 4) :
    (decodeHashWordExpr
      ((interface.fresh offset).publicInput ⟨0, by decide⟩) word).VarsBelow
        offset := by
  unfold decodeHashWordExpr
  exact foldl_weighted_below offset _ _ _
    (fun bit => inputs.freshPublicInput ⟨0, by decide⟩
      (NightstreamFPrime.Lifecycle.digestBitIndexNat
        (logicalWidth := logicalWidth) word bit))
    0 (by simp [Expr.VarsBelow])

private theorem priorDigestExpr_below
    (interface :
      NightstreamFPrime.Lifecycle.PiCCS.v1_1.StatementAbsorption.Interface
        logicalWidth publicFits)
    (offset : Nat) (inputs : InputsBelow interface offset) :
    ListBelow offset (priorDigestExpr interface offset) := by
  intro expression member
  rw [priorDigestExpr, List.mem_ofFn'] at member
  rcases member with ⟨lane, rfl⟩
  exact decodeHashWordExpr_below interface offset inputs lane

private theorem publicInputBlocks_below
    (interface :
      NightstreamFPrime.Lifecycle.PiCCS.v1_1.StatementAbsorption.Interface
        logicalWidth publicFits)
    (offset : Nat) (inputs : InputsBelow interface offset) :
    BlocksBelow offset (publicInputBlocks interface offset) := by
  let fresh := interface.fresh offset
  apply BlocksBelow.append
  · intro block member
    simp only [List.mem_singleton] at member
    subst block
    exact priorDigestExpr_below interface offset inputs
  · apply BlocksBelow.flatMap
    intro index _ block member
    simp only [List.mem_cons, List.not_mem_nil, or_false] at member
    rcases member with rfl | rfl
    · exact serializeCommitmentExpr_below offset (fresh.commitment index)
        (inputs.freshCommitment index)
    · exact serializePublicInputExpr_below offset (fresh.publicInput index)
        (inputs.freshPublicInput index)

private theorem mappedBlocks_below (bound : Nat) (blocks : List (List Expr))
    (blocksBelow : BlocksBelow bound blocks) :
    Formal.ActionsBelow bound (blocks.map absorbBlock) := by
  intro action member
  rw [List.mem_map] at member
  rcases member with ⟨block, blockMember, rfl⟩
  exact blockExpr_below bound block (blocksBelow block blockMember)

/-- The fixed statement serializer supplies the exact causal assumption of
the statement-absorption child. -/
theorem assumptions_of_inputsBelow
    (interface :
      NightstreamFPrime.Lifecycle.PiCCS.v1_1.StatementAbsorption.Interface
        logicalWidth publicFits)
    (offset : Nat) (inputs : InputsBelow interface offset) (env : Env) :
    NightstreamFPrime.Lifecycle.PiCCS.v1_1.StatementAbsorption.Assumptions
      interface offset env := by
  unfold NightstreamFPrime.Lifecycle.PiCCS.v1_1.StatementAbsorption.Assumptions
    actions publicInputActions
  intro action member
  rw [List.mem_append] at member
  rcases member with member | member
  · simp only [List.mem_singleton] at member
    subst action
    exact constantWords_below offset _
  · exact mappedBlocks_below offset _
      (publicInputBlocks_below interface offset inputs) action member

/-- The compiler-owned statement state lies below the next child boundary. -/
theorem finalState_varsBelow
    (interface :
      NightstreamFPrime.Lifecycle.PiCCS.v1_1.StatementAbsorption.Interface
        logicalWidth publicFits)
    (offset : Nat) (inputs : InputsBelow interface offset) :
    ∀ lane,
      (NightstreamFPrime.Lifecycle.PiCCS.v1_1.StatementAbsorption.finalState
        interface offset lane).VarsBelow
        (offset +
          (NightstreamFPrime.Lifecycle.PiCCS.v1_1.StatementAbsorption.program
            interface offset).recipes.length) := by
  have scope := (Formal.compile_scope offset Hash.zeroE
    (actions interface offset) (by
      intro lane
      simp [Hash.zeroE, Expr.VarsBelow])
    (assumptions_of_inputsBelow interface offset inputs (fun _ => 0))).1
  exact scope

/-- Exact parent-facing physical footprint, conditional only on the declared
affine form of caller-owned symbolic inputs. -/
def footprint
    (interface : Formal.Interface logicalWidth degreeBound publicFits)
    (inputs : ∀ offset,
      InputsAffine (Formal.statementAbsorptionInterface interface) offset) :
    R1CS.CircuitFootprint (Formal.statementAbsorptionCircuit interface) where
  freshColumnCount := fun _ => 0
  physicalRowCount := fun _ => 224368
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
        224368
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
        224368 :=
  (footprint interface inputs).physicalRowCount_eq offset

end NightstreamFPrime.Layout.PiCCS.v1_1.Leaves.StatementAbsorption
