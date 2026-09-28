import NightstreamFPrime.Lifecycle.PiCCS.v1_1.Completeness.Core
import NightstreamFPrime.Lifecycle.PiCCS.v1_1.Support

/-!
Owns semantic transport for one complete PiCCS phase across environments that
agree on the exact caller-selected source support. Phase-local transcript
outputs are supplied as explicit equalities. This module adds no row or
alternate verifier predicate.
-/

namespace NightstreamFPrime.Lifecycle.PiCCS.v1_1.Formal.PhaseTransport

open NightstreamFPrime.Circuit
open NightstreamFPrime.Circuit.Quadratic
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint

private theorem cubePoint_ext
    {Field : Type} {variableCount : Nat}
    (left right : CubePoint Field variableCount)
    (coordinates : left.coordinates = right.coordinates) : left = right := by
  cases left
  cases right
  simp_all

theorem proof_ext
    {degree : Nat}
    (left right : Proof degree)
    (rounds : left.piCcsRounds = right.piCcsRounds)
    (output : left.piCcsOutput = right.piCcsOutput)
    (commitments : left.piDecCommitments = right.piDecCommitments)
    (evaluations : left.piDecEvaluations = right.piDecEvaluations) :
    left = right := by
  cases left
  cases right
  simp_all

theorem fresh_ext
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (left right :
      NightstreamFPrime.Spec.Folding.Nifs.PaperNonInteractive.Fresh
        PaperAlgebra.Commitment
        (PaperAlgebra.PublicInput
          (logicalWidth := logicalWidth) (publicFits := publicFits))
        productionShape)
    (commitments : left.commitments = right.commitments)
    (publicInputs : left.publicInputs = right.publicInputs) : left = right := by
  cases left
  cases right
  simp_all

theorem running_ext
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (left right :
      NightstreamFPrime.Spec.Folding.Nifs.PaperNonInteractive.Running K
        PaperAlgebra.Commitment
        (PaperAlgebra.PublicInput
          (logicalWidth := logicalWidth) (publicFits := publicFits))
        productionShape)
    (point : left.point = right.point)
    (commitments : left.commitments = right.commitments)
    (publicInputs : left.publicInputs = right.publicInputs)
    (evaluations : left.evaluations = right.evaluations) : left = right := by
  cases left
  cases right
  simp_all

private theorem kExpr_eval_eq_of_agree_satisfy
    (value : KExpr) (allowed : Nat → Prop) (left right : Env)
    (support : value.c0.VarsSatisfy allowed ∧
      value.c1.VarsSatisfy allowed)
    (agrees : ∀ index, allowed index → left index = right index) :
    value.eval left = value.eval right := by
  exact congrArg₂ K.mk
    (value.c0.eval_eq_of_agree_satisfy allowed left right support.1 agrees)
    (value.c1.eval_eq_of_agree_satisfy allowed left right support.2 agrees)

theorem messagePolynomial_eq_of_agree_satisfy
    {degreeBound : Nat}
    (message : RoundTranscript.Message degreeBound)
    (allowed : Nat → Prop) (left right : Env)
    (support : ∀ coefficient,
      (message.coefficient coefficient).c0.VarsSatisfy allowed ∧
        (message.coefficient coefficient).c1.VarsSatisfy allowed)
    (agrees : ∀ index, allowed index → left index = right index) :
    message.semanticPolynomial left = message.semanticPolynomial right := by
  unfold RoundTranscript.Message.semanticPolynomial
    NightstreamFPrime.Gadgets.SumCheck.FixedChain.Round.semanticPolynomial
    NightstreamFPrime.Gadgets.SumCheck.FixedChain.Round.coefficients
  congr 1
  apply List.map_congr_left
  intro value member
  rw [List.mem_ofFn'] at member
  rcases member with ⟨coefficient, rfl⟩
  exact kExpr_eval_eq_of_agree_satisfy
    (message.coefficient coefficient) allowed left right
    (support coefficient) agrees

theorem evalRunning_eq_of_agree_satisfy
    {logicalWidth degreeBound : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth degreeBound publicFits)
    (offset : Nat) (allowed : Nat → Prop) (left right : Env)
    (support : ExternalInputsSupported interface offset allowed)
    (agrees : ∀ index, allowed index → left index = right index) :
    evalRunning interface offset left = evalRunning interface offset right := by
  unfold evalRunning StatementAbsorption.evalRunning
  congr 1
  · apply cubePoint_ext
    change List.ofFn (fun coordinate =>
        ((interface.running offset).point coordinate).eval left) =
      List.ofFn (fun coordinate =>
        ((interface.running offset).point coordinate).eval right)
    apply congrArg List.ofFn
    funext coordinate
    exact kExpr_eval_eq_of_agree_satisfy
      ((interface.running offset).point coordinate) allowed left right
      (support.runningPoint coordinate) agrees
  · funext source row coefficient
    exact ((interface.running offset).commitment source row coefficient
      ).eval_eq_of_agree_satisfy allowed left right
        (support.runningCommitment source row coefficient) agrees
  · funext source column
    exact ((interface.running offset).publicInput source column
      ).eval_eq_of_agree_satisfy allowed left right
        (support.runningPublicInput source column) agrees
  · funext source
    unfold StatementAbsorption.evalEvaluation
    congr 1
    · funext coefficient
      exact kExpr_eval_eq_of_agree_satisfy
        (((interface.running offset).evaluation source).eval_K coefficient)
        allowed left right (support.runningEval_K source coefficient) agrees
    · funext matrix coefficient
      exact kExpr_eval_eq_of_agree_satisfy
        (((interface.running offset).evaluation source).eval_A matrix
          coefficient) allowed left right
        (support.runningEval_A source matrix coefficient) agrees

theorem evalFresh_eq_of_agree_satisfy
    {logicalWidth degreeBound : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth degreeBound publicFits)
    (offset : Nat) (allowed : Nat → Prop) (left right : Env)
    (support : ExternalInputsSupported interface offset allowed)
    (agrees : ∀ index, allowed index → left index = right index) :
    evalFresh interface offset left = evalFresh interface offset right := by
  unfold evalFresh StatementAbsorption.evalFresh
  congr 1
  · funext source row coefficient
    exact ((interface.fresh offset).commitment source row coefficient
      ).eval_eq_of_agree_satisfy allowed left right
        (support.freshCommitment source row coefficient) agrees
  · funext source column
    exact ((interface.fresh offset).publicInput source column
      ).eval_eq_of_agree_satisfy allowed left right
        (support.freshPublicInput source column) agrees

theorem evalOutput_eq_of_agree_satisfy
    {logicalWidth degreeBound : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth degreeBound publicFits)
    (offset : Nat) (allowed : Nat → Prop) (left right : Env)
    (support : ExternalInputsSupported interface offset allowed)
    (agrees : ∀ index, allowed index → left index = right index) :
    evalOutput interface offset left = evalOutput interface offset right := by
  unfold evalOutput
  congr 1
  · funext source coefficient
    exact kExpr_eval_eq_of_agree_satisfy
      ((interface.output offset).padCoordinate source coefficient)
      allowed left right (support.outputEval_K source coefficient) agrees
  · funext source matrix coefficient
    exact kExpr_eval_eq_of_agree_satisfy
      ((interface.output offset).matrixCoordinate source matrix coefficient)
      allowed left right (support.outputEval_A source matrix coefficient) agrees

theorem evalProof_eq_of_agree_satisfy
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Interface logicalWidth
      (ProductionKey.degreeBound relation) publicFits)
    (offset : Nat) (allowed : Nat → Prop) (left right : Env)
    (template : Proof (ProductionKey.degreeBound relation))
    (support : ExternalInputsSupported interface offset allowed)
    (agrees : ∀ index, allowed index → left index = right index) :
    evalProof relation interface offset left template =
      evalProof relation interface offset right template := by
  have roundsEq :
      (fun roundIndex =>
        (interface.round offset roundIndex).semanticPolynomial left) =
      fun roundIndex =>
        (interface.round offset roundIndex).semanticPolynomial right := by
    funext roundIndex
    exact messagePolynomial_eq_of_agree_satisfy
      (interface.round offset roundIndex) allowed left right
      (support.roundCoefficient roundIndex) agrees
  have outputEq := evalOutput_eq_of_agree_satisfy interface offset allowed
    left right support agrees
  apply proof_ext
  · exact roundsEq
  · exact outputEq
  · rfl
  · rfl

theorem stateBinding_of_agree_satisfy
    {logicalWidth degreeBound : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth degreeBound publicFits)
    (offset : Nat) (allowed : Nat → Prop) (left right : Env)
    (support : ExternalInputsSupported interface offset allowed)
    (agrees : ∀ index, allowed index → left index = right index)
    (specification : StateBinding.SpecHolds
      (statementBindingInterface (atOffset interface offset)).state
      offset left) :
    StateBinding.SpecHolds
      (statementBindingInterface (atOffset interface offset)).state
      offset right := by
  refine {
    priorCanonical := ?_
    outputCanonical := ?_
    priorContext := ?_
    outputContext := ?_ }
  · intro word member
    calc
      ((statementBindingInterface (atOffset interface offset)).state.priorState
          offset word.index).eval right =
          ((statementBindingInterface (atOffset interface offset)).state.priorState
            offset word.index).eval left := by
        symm
        apply Expr.eval_eq_of_agree_satisfy _ allowed left right
        · simpa [statementBindingInterface, atOffset] using
            support.priorStateFixed word member
        · exact agrees
      _ = word.value := specification.priorCanonical word member
  · intro word member
    calc
      ((statementBindingInterface (atOffset interface offset)).state.outputState
          offset word.index).eval right =
          ((statementBindingInterface (atOffset interface offset)).state.outputState
            offset word.index).eval left := by
        symm
        apply Expr.eval_eq_of_agree_satisfy _ allowed left right
        · simpa [statementBindingInterface, atOffset] using
            support.outputStateFixed word member
        · exact agrees
      _ = word.value := specification.outputCanonical word member
  · intro lane
    calc
      ((statementBindingInterface (atOffset interface offset)).state.priorState
          offset (StateBinding.contextWordStart + lane.val)).eval right =
          ((statementBindingInterface (atOffset interface offset)).state.priorState
            offset (StateBinding.contextWordStart + lane.val)).eval left := by
        symm
        apply Expr.eval_eq_of_agree_satisfy _ allowed left right
        · simpa [statementBindingInterface, atOffset] using
            support.priorStateContext lane
        · exact agrees
      _ = ((statementBindingInterface
          (atOffset interface offset)).state.expectedContext offset lane).eval
            left := specification.priorContext lane
      _ = ((statementBindingInterface
          (atOffset interface offset)).state.expectedContext offset lane).eval
            right := by
        apply Expr.eval_eq_of_agree_satisfy _ allowed left right
        · simpa [statementBindingInterface, atOffset] using
            support.expectedContext lane
        · exact agrees
  · intro lane
    calc
      ((statementBindingInterface (atOffset interface offset)).state.outputState
          offset (StateBinding.contextWordStart + lane.val)).eval right =
          ((statementBindingInterface (atOffset interface offset)).state.outputState
            offset (StateBinding.contextWordStart + lane.val)).eval left := by
        symm
        apply Expr.eval_eq_of_agree_satisfy _ allowed left right
        · simpa [statementBindingInterface, atOffset] using
            support.outputStateContext lane
        · exact agrees
      _ = ((statementBindingInterface
          (atOffset interface offset)).state.expectedContext offset lane).eval
            left := specification.outputContext lane
      _ = ((statementBindingInterface
          (atOffset interface offset)).state.expectedContext offset lane).eval
            right := by
        apply Expr.eval_eq_of_agree_satisfy _ allowed left right
        · simpa [statementBindingInterface, atOffset] using
            support.expectedContext lane
        · exact agrees

/-- Transport one complete PiCCS phase through exact caller-input equality.
The two child-owned transcript projections are explicit premises and cannot be
replaced by prover-selected values. -/
theorem phaseHolds_of_agree_satisfy
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : AjtaiKey
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (interface : Interface logicalWidth
      (ProductionKey.degreeBound relation) publicFits)
    (offset : Nat) (allowed : Nat → Prop) (left right : Env)
    (template : Proof (ProductionKey.degreeBound relation))
    (support : ExternalInputsSupported interface offset allowed)
    (agrees : ∀ index, allowed index → left index = right index)
    (roundPointEq : RoundTranscript.evalRoundPoint
        (roundTranscriptInterface (atOffset interface offset))
        (roundTranscriptOffset interface offset) left =
      RoundTranscript.evalRoundPoint
        (roundTranscriptInterface (atOffset interface offset))
        (roundTranscriptOffset interface offset) right)
    (outgoingStateEq : StatementAbsorption.evalState left
        (outputBindingFinalState relation interface offset) =
      StatementAbsorption.evalState right
        (outputBindingFinalState relation interface offset))
    (phase : PhaseHolds relation ajtai interface offset left template) :
    PhaseHolds relation ajtai interface offset right template := by
  have runningEq := evalRunning_eq_of_agree_satisfy interface offset allowed
    left right support agrees
  have freshEq := evalFresh_eq_of_agree_satisfy interface offset allowed
    left right support agrees
  have proofEq := evalProof_eq_of_agree_satisfy relation interface offset
    allowed left right template support agrees
  refine {
    stateBinding := stateBinding_of_agree_satisfy interface offset allowed
      left right support agrees phase.stateBinding
    accepted := ?_
    roundPoint := ?_
    outgoingState := ?_ }
  · rw [← runningEq, ← freshEq, ← proofEq]
    exact phase.accepted
  · rw [← roundPointEq, ← runningEq, ← freshEq, ← proofEq]
    exact phase.roundPoint
  · rw [← outgoingStateEq, ← runningEq, ← freshEq, ← proofEq]
    exact phase.outgoingState

/-- Transport one complete PiCCS phase between two layouts after proving
equality of every evaluated semantic value. The target state binding remains
explicit because it names layout-owned statement cells rather than a derived
phase value. -/
theorem phaseHolds_of_eval_eq
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : AjtaiKey
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (leftInterface rightInterface : Interface logicalWidth
      (ProductionKey.degreeBound relation) publicFits)
    (leftOffset rightOffset : Nat) (left right : Env)
    (template : Proof (ProductionKey.degreeBound relation))
    (stateBinding : StateBinding.SpecHolds
      (statementBindingInterface
        (atOffset rightInterface rightOffset)).state rightOffset right)
    (runningEq : evalRunning leftInterface leftOffset left =
      evalRunning rightInterface rightOffset right)
    (freshEq : evalFresh leftInterface leftOffset left =
      evalFresh rightInterface rightOffset right)
    (proofEq : evalProof relation leftInterface leftOffset left template =
      evalProof relation rightInterface rightOffset right template)
    (roundPointEq : RoundTranscript.evalRoundPoint
        (roundTranscriptInterface (atOffset leftInterface leftOffset))
        (roundTranscriptOffset leftInterface leftOffset) left =
      RoundTranscript.evalRoundPoint
        (roundTranscriptInterface (atOffset rightInterface rightOffset))
        (roundTranscriptOffset rightInterface rightOffset) right)
    (outgoingStateEq : StatementAbsorption.evalState left
        (outputBindingFinalState relation leftInterface leftOffset) =
      StatementAbsorption.evalState right
        (outputBindingFinalState relation rightInterface rightOffset))
    (phase : PhaseHolds relation ajtai leftInterface leftOffset left template) :
    PhaseHolds relation ajtai rightInterface rightOffset right template := by
  refine {
    stateBinding := stateBinding
    accepted := ?_
    roundPoint := ?_
    outgoingState := ?_ }
  · rw [← runningEq, ← freshEq, ← proofEq]
    exact phase.accepted
  · rw [← roundPointEq, ← runningEq, ← freshEq, ← proofEq]
    exact phase.roundPoint
  · rw [← outgoingStateEq, ← runningEq, ← freshEq, ← proofEq]
    exact phase.outgoingState

/-- Cross-layout PiCCS transport with only named component equalities in the
public theorem type. Full dependent-record equalities are reconstructed inside
the proof, after all layout parameters are fixed. -/
theorem phaseHolds_of_components_eq
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : AjtaiKey
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (leftInterface rightInterface : Interface logicalWidth
      (ProductionKey.degreeBound relation) publicFits)
    (leftOffset rightOffset : Nat) (left right : Env)
    (template : Proof (ProductionKey.degreeBound relation))
    (stateBinding : StateBinding.SpecHolds
      (statementBindingInterface
        (atOffset rightInterface rightOffset)).state rightOffset right)
    (runningPoint : (evalRunning leftInterface leftOffset left).point =
      (evalRunning rightInterface rightOffset right).point)
    (runningCommitments :
      (evalRunning leftInterface leftOffset left).commitments =
        (evalRunning rightInterface rightOffset right).commitments)
    (runningPublicInputs :
      (evalRunning leftInterface leftOffset left).publicInputs =
        (evalRunning rightInterface rightOffset right).publicInputs)
    (runningEvaluations :
      (evalRunning leftInterface leftOffset left).evaluations =
        (evalRunning rightInterface rightOffset right).evaluations)
    (freshCommitments : (evalFresh leftInterface leftOffset left).commitments =
      (evalFresh rightInterface rightOffset right).commitments)
    (freshPublicInputs : (evalFresh leftInterface leftOffset left).publicInputs =
      (evalFresh rightInterface rightOffset right).publicInputs)
    (proofRounds :
      (evalProof relation leftInterface leftOffset left template).piCcsRounds =
        (evalProof relation rightInterface rightOffset right template).piCcsRounds)
    (proofOutput :
      (evalProof relation leftInterface leftOffset left template).piCcsOutput =
        (evalProof relation rightInterface rightOffset right template).piCcsOutput)
    (proofCommitments :
      (evalProof relation leftInterface leftOffset left template
        ).piDecCommitments =
        (evalProof relation rightInterface rightOffset right template
          ).piDecCommitments)
    (proofEvaluations :
      (evalProof relation leftInterface leftOffset left template
        ).piDecEvaluations =
        (evalProof relation rightInterface rightOffset right template
          ).piDecEvaluations)
    (roundPointEq : RoundTranscript.evalRoundPoint
        (roundTranscriptInterface (atOffset leftInterface leftOffset))
        (roundTranscriptOffset leftInterface leftOffset) left =
      RoundTranscript.evalRoundPoint
        (roundTranscriptInterface (atOffset rightInterface rightOffset))
        (roundTranscriptOffset rightInterface rightOffset) right)
    (outgoingStateEq : StatementAbsorption.evalState left
        (outputBindingFinalState relation leftInterface leftOffset) =
      StatementAbsorption.evalState right
        (outputBindingFinalState relation rightInterface rightOffset))
    (phase : PhaseHolds relation ajtai leftInterface leftOffset left template) :
    PhaseHolds relation ajtai rightInterface rightOffset right template := by
  have runningEq : evalRunning leftInterface leftOffset left =
      evalRunning rightInterface rightOffset right :=
    running_ext _ _ runningPoint runningCommitments runningPublicInputs
      runningEvaluations
  have freshEq : evalFresh leftInterface leftOffset left =
      evalFresh rightInterface rightOffset right :=
    fresh_ext _ _ freshCommitments freshPublicInputs
  have proofEq : evalProof relation leftInterface leftOffset left template =
      evalProof relation rightInterface rightOffset right template :=
    proof_ext _ _ proofRounds proofOutput proofCommitments proofEvaluations
  exact phaseHolds_of_eval_eq relation ajtai leftInterface rightInterface
    leftOffset rightOffset left right template stateBinding runningEq freshEq
    proofEq roundPointEq outgoingStateEq phase

end NightstreamFPrime.Lifecycle.PiCCS.v1_1.Formal.PhaseTransport
