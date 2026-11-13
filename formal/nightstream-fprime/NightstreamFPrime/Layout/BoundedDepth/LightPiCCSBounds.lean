import NightstreamFPrime.Layout.PiCCS.v1_2.Leaves.NormTerminal
import NightstreamFPrime.Layout.PiCCS.v1_2.Leaves.StatementAbsorption
import NightstreamFPrime.Lifecycle.BoundedDepth.LightPiCCSLayout

/-! Input scoping of the light Π_CCS layout (assembler A2a): when the external words lie below
the parent offset, every leaf's inputs lie below the leaf's offset, so
`LightPiCCSLayout.Assumptions` holds (`assumptions`). Each leaf reads the external words or outputs of earlier leaves, and each
earlier output lies below the next leaf's offset. -/

namespace NightstreamFPrime.Layout.BoundedDepth.LightPiCCSBounds

open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.BoundedDepth

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Circuit.Quadratic
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.UnifiedSources
open NightstreamFPrime.Gadgets.Poseidon2
open NightstreamFPrime.Gadgets.Multilinear
open NightstreamFPrime.Gadgets.Polynomial
open NightstreamFPrime.Lifecycle.PiCCS.v1_2
open NightstreamFPrime.Lifecycle.BoundedDepth.LightPiCCSLayout

variable {logicalWidth degreeBound : Nat}
  {publicFits : ringDegree * PaperAlgebra.publicRingColumns ≤
    RingCarrierLayout.carrierWidth logicalWidth}

/-- The external words of the light Π_CCS layout lie below the parent offset. -/
structure InputsBelow (interface : LightPiCCSLayout.Interface logicalWidth degreeBound publicFits)
    (offset : Nat) : Prop where
  runningPoint : ∀ coordinate, (interface.runningPoint offset coordinate).VarsBelow offset
  runningPad : ∀ coefficient,
    ((interface.runningEvaluation offset).eval_K coefficient).VarsBelow offset
  runningMatrix : ∀ matrix coefficient,
    ((interface.runningEvaluation offset).eval_A matrix coefficient).VarsBelow offset
  freshCommitment : ∀ source row coefficient,
    ((interface.fresh offset).commitment source row coefficient).VarsBelow offset
  freshPublicInput : ∀ source column,
    ((interface.fresh offset).publicInput source column).VarsBelow offset
  round : ∀ roundIndex coefficient,
    ((interface.round offset roundIndex).coefficient coefficient).VarsBelow offset
  outputPad : ∀ source coefficient,
    ((interface.output offset).padCoordinate source coefficient).VarsBelow offset
  outputMatrix : ∀ source matrix coefficient,
    ((interface.output offset).matrixCoordinate source matrix coefficient).VarsBelow offset

theorem le_nextOffset (child : FormalCircuit) (offset : Nat) :
    offset ≤ Formal.nextOffset child offset :=
  Nat.le_add_right _ _

variable (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
  (interface : LightPiCCSLayout.Interface logicalWidth degreeBound publicFits) (offset : Nat)

theorem challenge_le : offset ≤ challengeOffset interface offset := le_nextOffset _ _
theorem rounds_le : challengeOffset interface offset ≤ roundsOffset interface offset :=
  le_nextOffset _ _
theorem initial_le : roundsOffset interface offset ≤ initialOffset interface offset :=
  le_nextOffset _ _
theorem sumcheck_le : initialOffset interface offset ≤ sumcheckOffset interface offset :=
  le_nextOffset _ _
theorem evalK_le : sumcheckOffset interface offset ≤ evalKOffset interface offset :=
  le_nextOffset _ _
theorem evalA_le : evalKOffset interface offset ≤ evalAOffset interface offset :=
  le_nextOffset _ _
theorem ccs_le : evalAOffset interface offset ≤ ccsOffset interface offset := le_nextOffset _ _
theorem norm_le : ccsOffset interface offset ≤ normOffset relation interface offset :=
  le_nextOffset _ _
theorem finalIdentity_le :
    normOffset relation interface offset ≤ finalIdentityOffset relation interface offset :=
  le_nextOffset _ _
theorem output_le :
    finalIdentityOffset relation interface offset ≤ outputOffset relation interface offset :=
  le_nextOffset _ _

theorem roundsOffset_eq :
    roundsOffset interface offset = challengeOffset interface offset + 4384 := by
  unfold roundsOffset Formal.nextOffset Formal.childLength challengeCircuit
  rw [ChallengeDerivation.localLength_eq]

theorem initial_offset_le : offset ≤ initialOffset interface offset :=
  Nat.le_trans (challenge_le interface offset)
    (Nat.le_trans (rounds_le interface offset) (initial_le interface offset))

theorem evalK_offset_le : offset ≤ evalKOffset interface offset :=
  Nat.le_trans (initial_offset_le interface offset)
    (Nat.le_trans (sumcheck_le interface offset) (evalK_le interface offset))

theorem ccs_offset_le : offset ≤ ccsOffset interface offset :=
  Nat.le_trans (evalK_offset_le interface offset)
    (Nat.le_trans (evalA_le interface offset) (ccs_le interface offset))

theorem rounds_le_norm : roundsOffset interface offset ≤ normOffset relation interface offset :=
  Nat.le_trans (initial_le interface offset) (Nat.le_trans (sumcheck_le interface offset)
    (Nat.le_trans (evalK_le interface offset) (Nat.le_trans (evalA_le interface offset)
      (Nat.le_trans (ccs_le interface offset) (norm_le relation interface offset)))))

variable (below : InputsBelow interface offset) (env : Env)

include below

theorem statement :
    (statementCircuit interface offset).assumptions offset env := by
  have production := Layout.PiCCS.v1_2.Leaves.StatementAbsorption.assumptions_of_inputsBelow
    (statementInterface interface offset) offset
    ⟨below.freshCommitment, below.freshPublicInput⟩ env
  change StatementAbsorption.Assumptions _ _ _ at production
  unfold StatementAbsorption.Assumptions StatementAbsorption.actions at production
  rw [StatementAbsorption.publicInputActions_eq_domain] at production
  refine ⟨fun lane => by simp [statementDuplex, Hash.zeroE, Expr.VarsBelow], ?_⟩
  intro action member
  change action ∈ StatementAbsorption.domainActions lightFoldDomainChunk _ offset at member
  simp only [StatementAbsorption.domainActions, List.mem_cons, List.not_mem_nil,
    or_false] at member
  rcases member with rfl | rfl
  · intro expression inside
    simp only [StatementAbsorption.constantWords, List.mem_map] at inside
    obtain ⟨_, _, rfl⟩ := inside
    trivial
  · exact production _ (by simp [StatementAbsorption.domainActions])

theorem challenge :
    (challengeCircuit interface offset).assumptions (challengeOffset interface offset) env :=
  fun lane => Duplex.Formal.Owned.output_varsBelow (statementDuplex interface offset) offset env
    (statement interface offset below env) lane

theorem rounds :
    (roundsCircuit interface offset).assumptions (roundsOffset interface offset) env := by
  refine ⟨fun lane => ?_, fun roundIndex coefficient => ?_⟩
  · have stateBelow := ChallengeDerivation.finalState_varsBelow (env := env)
      (challengeInterface interface offset) (challengeOffset interface offset)
      (challenge interface offset below env) lane
    rw [ChallengeDerivation.program_recipes_length] at stateBelow
    rw [roundsOffset_eq]
    exact stateBelow
  · exact KExpr.varsBelow_mono _ (below.round roundIndex coefficient)
      (Nat.le_trans (challenge_le interface offset) (rounds_le interface offset))

theorem gamma_below : (gamma interface offset).VarsBelow (roundsOffset interface offset) := by
  have gammaBelow := ChallengeDerivation.gamma_varsBelow (env := fun _ => 0)
    (challengeInterface interface offset) (challengeOffset interface offset)
    (challenge interface offset below (fun _ => 0))
  rw [ChallengeDerivation.program_recipes_length] at gammaBelow
  rw [roundsOffset_eq]
  exact gammaBelow

theorem roundPoint_below (coordinate : Fin productionShape.cubeVariables) :
    (roundPoint interface offset coordinate).VarsBelow (initialOffset interface offset) :=
  RoundTranscript.challenge_varsBelow (roundsInterface interface offset)
    (roundsOffset interface offset) (fun _ => 0) (rounds interface offset below (fun _ => 0))
    coordinate

theorem initial :
    (initialCircuit interface offset).assumptions (initialOffset interface offset) env := by
  change InitialClaim.Assumptions (initialInterface interface offset)
    (initialOffset interface offset) env
  refine ⟨KExpr.varsBelow_mono _ (gamma_below interface offset below)
    (initial_le interface offset), fun coefficient member => ?_⟩
  change coefficient ∈ InitialClaim.coefficientExprs (initialInterface interface offset)
    (initialOffset interface offset) at member
  rw [InitialClaim.coefficientExprs, List.mem_append] at member
  rcases member with member | member
  · rw [List.mem_map] at member
    obtain ⟨coordinate, _, rfl⟩ := member
    exact KExpr.varsBelow_mono _ (below.runningPad coordinate.coefficient)
      (initial_offset_le interface offset)
  · rw [List.mem_map] at member
    obtain ⟨coordinate, _, rfl⟩ := member
    exact KExpr.varsBelow_mono _ (below.runningMatrix coordinate.matrix coordinate.coefficient)
      (initial_offset_le interface offset)

theorem sumcheck :
    (sumcheckCircuit interface offset).assumptions (sumcheckOffset interface offset) env := by
  change SumcheckChain.Assumptions (sumcheckInterface interface offset)
    (sumcheckOffset interface offset) env
  refine ⟨Horner.Owned.output_varsBelow (InitialClaim.ownedInterface
      (initialInterface interface offset)) (initialOffset interface offset) env
      (initial interface offset below env),
    fun roundIndex => ⟨fun coefficient => ?_, ?_⟩⟩
  · exact KExpr.varsBelow_mono _ (below.round roundIndex coefficient)
      (Nat.le_trans (initial_offset_le interface offset) (sumcheck_le interface offset))
  · exact KExpr.varsBelow_mono _ (roundPoint_below interface offset below roundIndex)
      (sumcheck_le interface offset)

theorem evalK :
    (evalKCircuit interface offset).assumptions (evalKOffset interface offset) env := by
  have initialLe : initialOffset interface offset ≤ evalKOffset interface offset :=
    Nat.le_trans (sumcheck_le interface offset) (evalK_le interface offset)
  refine ⟨fun coordinate => ⟨?_, ?_⟩, ⟨?_, fun coefficient member => ?_⟩⟩
  · exact KExpr.varsBelow_mono _ (roundPoint_below interface offset below coordinate)
      initialLe
  · exact KExpr.varsBelow_mono _ (below.runningPoint coordinate)
      (evalK_offset_le interface offset)
  · exact KExpr.varsBelow_mono _ (gamma_below interface offset below)
      (Nat.le_trans (initial_le interface offset) initialLe)
  · change coefficient ∈ EvalKTerminal.coefficientExprs (evalKInterface interface offset)
      (evalKOffset interface offset) at member
    rw [EvalKTerminal.coefficientExprs, List.mem_map] at member
    obtain ⟨coordinate, _, rfl⟩ := member
    exact KExpr.varsBelow_mono _
      (below.outputPad (runningSourceIndex coordinate.running) coordinate.coefficient)
      (evalK_offset_le interface offset)

theorem evalA :
    (evalACircuit interface offset).assumptions (evalAOffset interface offset) env := by
  have initialLe : initialOffset interface offset ≤ evalAOffset interface offset :=
    Nat.le_trans (sumcheck_le interface offset)
      (Nat.le_trans (evalK_le interface offset) (evalA_le interface offset))
  have parentLe : offset ≤ evalAOffset interface offset :=
    Nat.le_trans (evalK_offset_le interface offset) (evalA_le interface offset)
  refine ⟨fun coordinate => ⟨?_, ?_⟩, ⟨?_, fun coefficient member => ?_⟩⟩
  · exact KExpr.varsBelow_mono _ (roundPoint_below interface offset below coordinate)
      initialLe
  · exact KExpr.varsBelow_mono _ (below.runningPoint coordinate) parentLe
  · exact KExpr.varsBelow_mono _ (gamma_below interface offset below)
      (Nat.le_trans (initial_le interface offset) initialLe)
  · change coefficient ∈ EvalATerminal.coefficientExprs (evalAInterface interface offset)
      (evalAOffset interface offset) at member
    rw [EvalATerminal.coefficientExprs, List.mem_map] at member
    obtain ⟨coordinate, _, rfl⟩ := member
    exact KExpr.varsBelow_mono _
      (below.outputMatrix (runningSourceIndex coordinate.running) coordinate.matrix
        coordinate.coefficient) parentLe

theorem ccs :
    (ccsCircuit relation interface offset).assumptions (ccsOffset interface offset) env :=
  fun matrix => KExpr.varsBelow_mono _ (below.outputMatrix _ matrix _)
    (ccs_offset_le interface offset)

theorem norm :
    (normCircuit relation interface offset).assumptions (normOffset relation interface offset)
      env := by
  have parentLe : offset ≤ normOffset relation interface offset :=
    Nat.le_trans (ccs_offset_le interface offset) (norm_le relation interface offset)
  change NormTerminal.Assumptions (normInterface relation interface offset)
    (normOffset relation interface offset) env
  refine ⟨KExpr.varsBelow_mono _ (gamma_below interface offset below)
    (rounds_le_norm relation interface offset), fun coefficient member => ?_⟩
  change coefficient ∈ NormTerminal.coefficientExprs (normInterface relation interface offset)
    (normOffset relation interface offset) at member
  rw [NormTerminal.coefficientExprs, List.mem_map] at member
  obtain ⟨source, _, rfl⟩ := member
  apply Layout.PiCCS.v1_2.Leaves.NormTerminal.residualExpr_varsBelow
  by_cases fresh : source.val < LightShape.freshCount
  · simp only [normInterface, fresh, ↓reduceIte]
    exact KExpr.varsBelow_mono _ (below.outputPad source _) parentLe
  · simp [normInterface, fresh, KExpr.zero, KExpr.VarsBelow, Expr.VarsBelow]

theorem finalIdentity :
    (finalIdentityCircuit relation interface offset).assumptions
      (finalIdentityOffset relation interface offset) env := by
  have normLe := finalIdentity_le relation interface offset
  have ccsLe : ccsOffset interface offset ≤ finalIdentityOffset relation interface offset :=
    Nat.le_trans (norm_le relation interface offset) normLe
  have evalALe : evalAOffset interface offset ≤ finalIdentityOffset relation interface offset :=
    Nat.le_trans (ccs_le interface offset) ccsLe
  have evalKLe : evalKOffset interface offset ≤ finalIdentityOffset relation interface offset :=
    Nat.le_trans (evalA_le interface offset) evalALe
  have initialLe : initialOffset interface offset ≤
      finalIdentityOffset relation interface offset :=
    Nat.le_trans (sumcheck_le interface offset) (Nat.le_trans (evalK_le interface offset) evalKLe)
  have roundsLe : roundsOffset interface offset ≤
      finalIdentityOffset relation interface offset :=
    Nat.le_trans (initial_le interface offset) initialLe
  have normEq : normOffset relation interface offset = ccsOffset interface offset + 23 := by
    unfold normOffset Formal.nextOffset Formal.childLength ccsCircuit
    rw [CcsTerminal.localLength_eq]
    rfl
  have evalKEq : evalKOffset interface offset =
      sumcheckOffset interface offset + SumcheckChain.privateCount degreeBound := by
    unfold evalKOffset Formal.nextOffset Formal.childLength sumcheckCircuit
    rw [SumcheckChain.localLength_eq]
  refine ⟨fun coordinate => ⟨?_, ?_⟩, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · exact KExpr.varsBelow_mono _ (roundPoint_below interface offset below coordinate) initialLe
  · have alphaBelow := ChallengeDerivation.alpha_varsBelow (env := fun _ => 0)
      (challengeInterface interface offset) (challengeOffset interface offset)
      (challenge interface offset below (fun _ => 0)) coordinate
    rw [ChallengeDerivation.program_recipes_length, ← roundsOffset_eq] at alphaBelow
    exact KExpr.varsBelow_mono _ alphaBelow roundsLe
  · exact KExpr.varsBelow_mono _ (gamma_below interface offset below) roundsLe
  · exact KExpr.varsBelow_mono _ (PointWeightedHorner.Owned.output_varsBelow
      (EvalKTerminal.coreInterface (evalKInterface interface offset)) (by decide)
      (evalKOffset interface offset) env (evalK interface offset below env)) evalALe
  · exact KExpr.varsBelow_mono _ (PointWeightedHorner.Owned.output_varsBelow
      (EvalATerminal.coreInterface (evalAInterface interface offset)) (by decide)
      (evalAOffset interface offset) env (evalA interface offset below env)) ccsLe
  · change (CcsTerminal.output relation (ccsInterface relation interface offset)
      (ccsOffset interface offset)).VarsBelow (finalIdentityOffset relation interface offset)
    unfold CcsTerminal.output Sparse.Owned.output KExpr.VarsBelow Expr.VarsBelow
    rw [CcsTerminal.productCount_eq]
    rw [normEq] at normLe
    omega
  · exact Horner.Owned.output_varsBelow (NormTerminal.ownedInterface
      (normInterface relation interface offset)) (normOffset relation interface offset) env
      (norm relation interface offset below env)
  · have terminalBelow := SumcheckChain.output_varsBelow (sumcheckInterface interface offset)
      (sumcheckOffset interface offset) (sumcheck interface offset below (fun _ => 0))
    rw [← evalKEq] at terminalBelow
    exact KExpr.varsBelow_mono _ terminalBelow evalKLe

theorem output :
    (outputCircuit interface offset).assumptions (outputOffset relation interface offset) env := by
  have initialLe : initialOffset interface offset ≤ outputOffset relation interface offset :=
    Nat.le_trans (sumcheck_le interface offset) (Nat.le_trans (evalK_le interface offset)
      (Nat.le_trans (evalA_le interface offset) (Nat.le_trans (ccs_le interface offset)
        (Nat.le_trans (norm_le relation interface offset) (Nat.le_trans
          (finalIdentity_le relation interface offset) (output_le relation interface offset))))))
  have parentLe : offset ≤ outputOffset relation interface offset :=
    Nat.le_trans (initial_offset_le interface offset) initialLe
  refine ⟨fun lane => ?_, fun action member => ?_⟩
  · exact Expr.VarsBelow.mono _ (RoundTranscript.finalState_varsBelow
      (roundsInterface interface offset) (roundsOffset interface offset) env
      (rounds interface offset below env) lane) initialLe
  · simp only [OutputBinding.duplexInterface, OutputBinding.actions, List.mem_singleton] at member
    subst member
    intro expression inside
    simp only [OutputBinding.outputWords, OutputBinding.sourceWords, OutputBinding.padWords,
      OutputBinding.matrixWords, StatementAbsorption.serializeKExpr, List.mem_flatMap,
      List.mem_append, List.mem_cons, List.mem_finRange, List.not_mem_nil, or_false,
      true_and] at inside
    obtain ⟨source, inside⟩ := inside
    rcases inside with ⟨coefficient, word | word⟩ | ⟨matrix, coefficient, word | word⟩ <;>
      subst word
    · exact (KExpr.varsBelow_mono _ (below.outputPad source coefficient) parentLe).1
    · exact (KExpr.varsBelow_mono _ (below.outputPad source coefficient) parentLe).2
    · exact (KExpr.varsBelow_mono _ (below.outputMatrix source matrix coefficient) parentLe).1
    · exact (KExpr.varsBelow_mono _ (below.outputMatrix source matrix coefficient) parentLe).2

/-- Input scoping of the light Π_CCS layout: external words below the parent offset give every
leaf's assumptions. -/
theorem assumptions : LightPiCCSLayout.Assumptions relation interface offset env :=
  ⟨statement interface offset below env, challenge interface offset below env,
    rounds interface offset below env, initial interface offset below env,
    sumcheck interface offset below env, evalK interface offset below env,
    evalA interface offset below env, ccs relation interface offset below env,
    norm relation interface offset below env, finalIdentity relation interface offset below env,
    output relation interface offset below env⟩

end NightstreamFPrime.Layout.BoundedDepth.LightPiCCSBounds
