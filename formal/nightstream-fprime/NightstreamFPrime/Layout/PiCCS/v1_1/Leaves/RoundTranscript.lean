import NightstreamFPrime.Layout.Poseidon2.Duplex
import NightstreamFPrime.Lifecycle.PiCCS.v1_1.Completeness

/-!
Paper authority: SuperNeo v1_1, section 7.3, indexed PiCCS SumCheck rounds.
Obligation: Absorb each prover polynomial, absorb its round label, and derive
the corresponding verifier challenge in exact round order.

Inputs:
- the prior child-owned transcript state;
- 24 prover polynomial messages of degree at most `degreeBound`.

Outputs:
- 24 verifier-derived round challenges;
- the child-owned outgoing transcript state.

Constraint groups:
- one generic message-absorption action group;
- one generic labelled squeeze action group;
- indexed composition over the fixed 24-round chain.

Parent coverage:
- `Formal.opsAt`, child `piccs.v1_1.round_transcript`.
-/

namespace NightstreamFPrime.Layout.PiCCS.v1_1.Leaves.RoundTranscript

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Circuit.Quadratic
open NightstreamFPrime.Gadgets.Poseidon2
open NightstreamFPrime.Gadgets.Poseidon2.Duplex
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Lifecycle.PiCCS.v1_1
open NightstreamFPrime.Lifecycle.PiCCS.v1_1.RoundTranscript
open NightstreamFPrime.Layout.Poseidon2
open NightstreamFPrime.Layout.Poseidon2.Duplex
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint

variable {logicalWidth degreeBound : Nat}
  {publicFits : ringDegree * publicRingColumns ≤
    Phi81CarrierLayout.carrierWidth logicalWidth}

/-- Affine physical-input premise for the incoming state and all prover round
message coefficients. Challenges are excluded because the compiler owns them. -/
structure InputsAffine
    (interface :
      NightstreamFPrime.Lifecycle.PiCCS.v1_1.RoundTranscript.Interface
        degreeBound)
    (offset : Nat) : Prop where
  initialState : StateAffine (interface.initialState offset)
  roundCoefficient : ∀ roundIndex coefficient,
    KExprAffine
      ((interface.round offset roundIndex).coefficient coefficient)

private theorem serializeKExpr_affine (value : KExpr)
    (affine : KExprAffine value) :
    ListAffine (serializeKExpr value) := by
  intro expression member
  simp only [serializeKExpr, List.mem_cons, List.not_mem_nil,
    or_false] at member
  rcases member with rfl | rfl
  · exact affine.1
  · exact affine.2

private theorem serializeKExprs_affine (values : List KExpr)
    (affine : ∀ value ∈ values, KExprAffine value) :
    ListAffine (serializeKExprs values) := by
  intro expression member
  rw [serializeKExprs, List.mem_flatMap] at member
  rcases member with ⟨value, valueMember, expressionMember⟩
  exact serializeKExpr_affine value (affine value valueMember)
    expression expressionMember

private theorem serializeRoundExpr_affine
    (message : Message degreeBound)
    (affine : ∀ coefficient,
      KExprAffine (message.coefficient coefficient)) :
    ListAffine (serializeRoundExpr message) := by
  unfold serializeRoundExpr
  apply serializeKExprs_affine
  intro value member
  rw [List.mem_ofFn'] at member
  rcases member with ⟨coefficient, rfl⟩
  exact affine coefficient

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

private theorem zero_affine : KExprAffine KExpr.zero := by
  exact ⟨R1CS.isAffine_const 0, R1CS.isAffine_const 0⟩

private theorem roundActionsWithExpected_affine
    (interface :
      NightstreamFPrime.Lifecycle.PiCCS.v1_1.RoundTranscript.Interface
        degreeBound)
    (offset : Nat) (roundIndex : Fin productionShape.cubeVariables)
    (expected : KExpr) (inputs : InputsAffine interface offset)
    (expectedAffine : KExprAffine expected) :
    ActionsAffine
      (roundActionsWithExpected interface offset roundIndex expected) := by
  let message := interface.round offset roundIndex
  have payloadAffine : ListAffine
      (Expr.const (NightstreamFPrime.Lifecycle.natWord roundIndex.val) ::
        serializeRoundExpr message) := by
    intro expression member
    rcases List.mem_cons.mp member with rfl | member
    · exact R1CS.isAffine_const _
    · exact serializeRoundExpr_affine message
        (inputs.roundCoefficient roundIndex) expression member
  unfold roundActionsWithExpected
  dsimp only
  apply ActionsAffine.cons
  · exact blockExpr_affine _ payloadAffine
  · apply ActionsAffine.cons
    · exact constantWords_affine _
    · apply ActionsAffine.cons
      · exact expectedAffine
      · intro action member
        simp at member

private theorem ActionsAffine.flatMap
    {Index : Type} (indices : List Index)
    (group : Index → List Formal.Action)
    (affine : ∀ index ∈ indices, ActionsAffine (group index)) :
    ActionsAffine (indices.flatMap group) := by
  intro action member
  rw [List.mem_flatMap] at member
  rcases member with ⟨index, indexMember, actionMember⟩
  exact affine index indexMember action actionMember

private theorem layoutActions_affine
    (interface :
      NightstreamFPrime.Lifecycle.PiCCS.v1_1.RoundTranscript.Interface
        degreeBound)
    (offset : Nat) (inputs : InputsAffine interface offset) :
    ActionsAffine (layoutActions interface offset) := by
  unfold layoutActions
  apply ActionsAffine.flatMap
  intro roundIndex _
  exact roundActionsWithExpected_affine interface offset roundIndex
    KExpr.zero inputs zero_affine

private theorem layoutProgram_samples_affine
    (interface :
      NightstreamFPrime.Lifecycle.PiCCS.v1_1.RoundTranscript.Interface
        degreeBound)
    (offset : Nat) (inputs : InputsAffine interface offset) :
    ∀ sample ∈ (layoutProgram interface offset).samples,
      KExprAffine sample := by
  exact compile_samples_affine offset (interface.initialState offset)
    (layoutActions interface offset) inputs.initialState
    (layoutActions_affine interface offset inputs)

private theorem challenge_affine
    (interface :
      NightstreamFPrime.Lifecycle.PiCCS.v1_1.RoundTranscript.Interface
        degreeBound)
    (offset : Nat) (inputs : InputsAffine interface offset)
    (roundIndex : Fin productionShape.cubeVariables) :
    KExprAffine (challenge interface offset roundIndex) := by
  exact layoutProgram_samples_affine interface offset inputs _
    (List.get_mem _ _)

theorem actions_affine
    (interface :
      NightstreamFPrime.Lifecycle.PiCCS.v1_1.RoundTranscript.Interface
        degreeBound)
    (offset : Nat) (inputs : InputsAffine interface offset) :
    ActionsAffine (actions interface offset) := by
  unfold actions roundActions
  apply ActionsAffine.flatMap
  intro roundIndex _
  exact roundActionsWithExpected_affine interface offset roundIndex
    (challenge interface offset roundIndex) inputs
    (challenge_affine interface offset inputs roundIndex)

/-- Exact parent-facing physical footprint for the indexed 24-round transcript
chain. -/
def footprint
    (interface : Formal.Interface logicalWidth degreeBound publicFits)
    (inputs : ∀ offset,
      InputsAffine (Formal.roundTranscriptInterface interface) offset) :
    R1CS.CircuitFootprint (Formal.roundTranscriptCircuit interface) where
  freshColumnCount := fun _ => 0
  physicalRowCount := fun _ =>
    24 *
      NightstreamFPrime.Lifecycle.PiCCS.v1_1.RoundTranscript.perRoundRecipeCount
        degreeBound
  freshColumnCount_eq := by
    intro offset
    unfold Formal.roundTranscriptCircuit
    rw [FormalCircuit.withConstantFootprint_main]
    change R1CS.totalFreshCount (flatConstraints
      (opsAt (Formal.roundTranscriptInterface interface) offset)) = 0
    rw [NightstreamFPrime.Lifecycle.PiCCS.v1_1.RoundTranscript.flatConstraints_opsAt]
    apply R1CS.recipeConstraints_totalFreshCount
    exact compile_recipes_direct offset
      ((Formal.roundTranscriptInterface interface).initialState offset)
      (actions (Formal.roundTranscriptInterface interface) offset)
      (inputs offset).initialState
      (actions_affine _ offset (inputs offset))
  physicalRowCount_eq := by
    intro offset
    unfold Formal.roundTranscriptCircuit
    rw [FormalCircuit.withConstantFootprint_main]
    change R1CS.totalRowCount (flatConstraints
      (opsAt (Formal.roundTranscriptInterface interface) offset)) =
        24 *
          NightstreamFPrime.Lifecycle.PiCCS.v1_1.RoundTranscript.perRoundRecipeCount
            degreeBound
    rw [NightstreamFPrime.Lifecycle.PiCCS.v1_1.RoundTranscript.flatConstraints_opsAt]
    rw [R1CS.recipeConstraints_totalRowCount]
    exact NightstreamFPrime.Lifecycle.PiCCS.v1_1.RoundTranscript.program_recipes_length
      (Formal.roundTranscriptInterface interface) offset
    exact compile_recipes_direct offset
      ((Formal.roundTranscriptInterface interface).initialState offset)
      (actions (Formal.roundTranscriptInterface interface) offset)
      (inputs offset).initialState
      (actions_affine _ offset (inputs offset))

theorem freshColumnCount_eq
    (interface : Formal.Interface logicalWidth degreeBound publicFits)
    (inputs : ∀ offset,
      InputsAffine (Formal.roundTranscriptInterface interface) offset)
    (offset : Nat) :
    R1CS.totalFreshCount (flatConstraints (Circuit.ops
      (Formal.roundTranscriptCircuit interface).main offset)) = 0 :=
  (footprint interface inputs).freshColumnCount_eq offset

theorem physicalRowCount_eq
    (interface : Formal.Interface logicalWidth degreeBound publicFits)
    (inputs : ∀ offset,
      InputsAffine (Formal.roundTranscriptInterface interface) offset)
    (offset : Nat) :
    R1CS.totalRowCount (flatConstraints (Circuit.ops
      (Formal.roundTranscriptCircuit interface).main offset)) =
        24 *
          NightstreamFPrime.Lifecycle.PiCCS.v1_1.RoundTranscript.perRoundRecipeCount
            degreeBound :=
  (footprint interface inputs).physicalRowCount_eq offset

theorem physicalRowCount_eq_of_degreeBound_eq_four
    (interface : Formal.Interface logicalWidth degreeBound publicFits)
    (inputs : ∀ offset,
      InputsAffine (Formal.roundTranscriptInterface interface) offset)
    (offset : Nat) (degreeBound_eq : degreeBound = 4) :
    R1CS.totalRowCount (flatConstraints (Circuit.ops
      (Formal.roundTranscriptCircuit interface).main offset)) = 85248 := by
  rw [physicalRowCount_eq interface inputs offset, degreeBound_eq]
  rfl

end NightstreamFPrime.Layout.PiCCS.v1_1.Leaves.RoundTranscript
