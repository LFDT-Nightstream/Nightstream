import NightstreamFPrime.Lifecycle.PiCCS.v1_1.Formal

/-!
Paper authority: SuperNeo v1.1, Section 7.3, complete `Pi_CCS` reduction.
Obligation: Show that the assembled PiCCS specification is the exact
key-facing phase: the production key statement, the transcript coverage, and
the outgoing state, with every shared value taken from the parent carrier.
-/

namespace NightstreamFPrime.Lifecycle.PiCCS.v1_1.Formal

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Circuit.Quadratic
open NightstreamFPrime.Gadgets.Poseidon2
open NightstreamFPrime.Gadgets.SumCheck
open NightstreamFPrime.Gadgets.Multilinear
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.UnifiedSources

private theorem cubePoint_eq_of_coordinates
    {Field : Type} {variableCount : Nat}
    (left right : CubePoint Field variableCount)
    (coordinates : left.coordinates = right.coordinates) : left = right := by
  cases left
  cases right
  simp_all

/-- Mechanical coverage of the exact production PiCCS relation. Every
shared equality is derived from the parent carrier; no transcript state or
challenge is supplied as a premise. -/
theorem spec_implies_phaseHolds
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : AjtaiKey
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (interface : Interface logicalWidth
      (ProductionKey.degreeBound relation) publicFits)
    (offset : Nat) (env : Env)
    (template : Proof (ProductionKey.degreeBound relation))
    (specification : SpecHolds relation interface offset env) :
    PhaseHolds relation ajtai interface offset env template := by
  let shared := atOffset interface offset
  let running := evalRunning interface offset env
  let fresh := evalFresh interface offset env
  let proof := evalProof relation interface offset env template
  let context := ChallengeDerivation.productionContext
    relation ajtai running fresh
  have statementCoverage := StatementBinding.spec_implies_keyStatement
    relation ajtai running fresh (statementBindingInterface shared)
      (statementBindingOffset offset) env specification.statementBinding
  have statementState := StatementAbsorption.spec_implies_keyInitialState
    relation ajtai (statementAbsorptionInterface shared)
      (statementAbsorptionOffset interface offset) env
      specification.statementAbsorption
  dsimp only at statementState
  rw [ProductionKey.key_oracle_eq relation ajtai] at statementState
  have challengeCoverage :=
    ChallengeDerivation.spec_implies_derivePreSumcheck
      (challengeInterface shared offset) (challengeOffset interface offset) env
      context (by
        simpa [shared, running, fresh, context, challengeInterface,
          statementAbsorptionInterface, atOffset, evalRunning, evalFresh]
          using! statementState) specification.challenge
  have keyChallenges :=
    ChallengeDerivation.spec_implies_keyExecution_challenges
      relation ajtai running fresh proof (challengeInterface shared offset)
      (challengeOffset interface offset) env (by
        simpa [shared, running, fresh, context, challengeInterface,
          statementAbsorptionInterface, atOffset, evalRunning, evalFresh]
          using! statementState) specification.challenge
  have roundCoverage := RoundTranscript.spec_implies_keyExecution_rounds
    relation ajtai running fresh proof (roundTranscriptInterface shared)
      (roundTranscriptOffset interface offset) env (by
        simpa [shared, context, challengeInterface,
          roundTranscriptInterface, atOffset] using! challengeCoverage.2.2)
      (by
        intro roundIndex
        rfl)
      specification.roundTranscript
  have initialEq := InitialClaim.spec_implies_keyInitial
    relation ajtai running fresh proof (initialClaimInterface shared)
      (initialClaimOffset interface offset) env (by
        simpa [shared, initialClaimInterface, challengeInterface, atOffset]
          using! keyChallenges.2)
      (by
        intro coordinate
        rfl)
      (by
        intro coordinate
        rfl)
      specification.initialClaim
  have evalKEq := EvalKTerminal.spec_implies_keyPadAtMessage
    relation ajtai running fresh proof (evalKInterface shared)
      (evalKOffset interface offset) env (by
        simpa [shared, evalKInterface, roundTranscriptInterface, roundPoint,
          atOffset] using! roundCoverage.1)
      (by rfl) (by
        simpa [shared, evalKInterface, challengeInterface, atOffset]
          using! keyChallenges.2)
      (by
        intro coordinate
        rfl)
      specification.eval_K
  have evalAEq := EvalATerminal.spec_implies_keyMatrixAtMessage
    relation ajtai running fresh proof (evalAInterface shared)
      (evalAOffset interface offset) env (by
        simpa [shared, evalAInterface, roundTranscriptInterface, roundPoint,
          atOffset] using! roundCoverage.1)
      (by rfl) (by
        simpa [shared, evalAInterface, challengeInterface, atOffset]
          using! keyChallenges.2)
      (by
        intro coordinate
        rfl)
      specification.eval_A
  have ccsEq := CcsTerminal.spec_implies_keyCcsAtMessage
    relation ajtai running fresh proof (ccsInterface relation shared)
      (ccsOffset interface offset) env (by
        intro matrix
        rfl)
      specification.ccs
  have normEq := NormTerminal.spec_implies_keyNormAtMessage
    relation ajtai running fresh proof (normInterface relation shared)
      (normOffset relation interface offset) env (by
        simpa [shared, normInterface, challengeInterface, atOffset]
          using! keyChallenges.2)
      (by
        intro source
        rfl)
      specification.norm
  have alphaInterfaceEq : PointEquality.Owned.evalRightPoint
      (FinalIdentity.pointInterfaceAt (finalIdentityInterface relation shared)
        (finalIdentityOffset relation interface offset))
      (finalIdentityOffset relation interface offset) env =
      ChallengeDerivation.evalAlpha (challengeInterface shared offset)
        (challengeOffset interface offset) env := by
    apply cubePoint_eq_of_coordinates
    simpa [shared, PointEquality.Owned.evalRightPoint,
      FinalIdentity.pointInterfaceAt, finalIdentityInterface,
      challengeAlpha, challengeInterface, atOffset] using!
        (NightstreamFPrime.Lifecycle.PiCCS.v1_1.ChallengeDerivation.evalAlpha_coordinates
          (challengeInterface shared offset)
            (challengeOffset interface offset) env).symm
  have gammaInterfaceEq :
      ((finalIdentityInterface relation shared).gamma
        (finalIdentityOffset relation interface offset)).eval env =
      ChallengeDerivation.evalGamma (challengeInterface shared offset)
        (challengeOffset interface offset) env := by
    have startEq : challengeStart shared =
        challengeOffset interface offset := by
      simpa [shared] using challengeStart_atOffset interface offset
    rw [ChallengeDerivation.evalGamma_eq]
    change (ChallengeDerivation.gamma
      (challengeInterface shared shared.baseOffset)
        (challengeStart shared)).eval env =
      (ChallengeDerivation.gamma
        (challengeInterface shared offset)
          (challengeOffset interface offset)).eval env
    have baseEq : shared.baseOffset = offset := rfl
    rw [baseEq, startEq]
  have terminalEq := FinalIdentity.spec_implies_keyTerminal
    relation ajtai running fresh proof (finalIdentityInterface relation shared)
      (finalIdentityOffset relation interface offset) env (by
        simpa [shared, finalIdentityInterface, roundTranscriptInterface,
          roundPoint, atOffset] using! roundCoverage.1)
      (by
        exact alphaInterfaceEq.trans keyChallenges.1)
      (by
        exact gammaInterfaceEq.trans keyChallenges.2)
      (by
        change (EvalKTerminal.output (evalKInterface shared)
          (evalKStart shared)).eval env = _
        have startEq : evalKStart shared = evalKOffset interface offset := by
          simpa [shared] using evalKStart_atOffset interface offset
        rw [startEq]
        exact evalKEq)
      (by
        change (EvalATerminal.output (evalAInterface shared)
          (evalAStart shared)).eval env = _
        have startEq : evalAStart shared = evalAOffset interface offset := by
          simpa [shared] using evalAStart_atOffset interface offset
        rw [startEq]
        exact evalAEq)
      (by
        change (CcsTerminal.output relation (ccsInterface relation shared)
          (ccsStart shared)).eval env = _
        have startEq : ccsStart shared = ccsOffset interface offset := by
          simpa [shared] using ccsStart_atOffset interface offset
        rw [startEq]
        exact ccsEq)
      (by
        change (NormTerminal.output (normInterface relation shared)
          (normStart shared)).eval env = _
        have startEq : normStart shared =
            normOffset relation interface offset := by
          simpa [shared] using normStart_atOffset relation interface offset
        rw [startEq]
        exact normEq)
      specification.finalIdentity
  have sumcheckRoundPointEq : SumcheckChain.evalRoundPoint
      (sumcheckInterface shared) (sumcheckOffset interface offset) env =
      RoundTranscript.evalRoundPoint (roundTranscriptInterface shared)
        (roundTranscriptOffset interface offset) env := by
    apply cubePoint_eq_of_coordinates
    change (canonicalFinIndices productionShape.cubeVariables).map
        (fun roundIndex =>
          ((roundTranscriptRound shared (sumcheckOffset interface offset)
            roundIndex).challenge).eval env) =
      (canonicalFinIndices productionShape.cubeVariables).map
        (fun roundIndex =>
          (RoundTranscript.challenge (roundTranscriptInterface shared)
            (roundTranscriptOffset interface offset) roundIndex).eval env)
    apply List.map_congr_left
    intro roundIndex _
    have startEq : roundTranscriptStart shared =
        roundTranscriptOffset interface offset := by
      simpa [shared] using roundTranscriptStart_atOffset interface offset
    change (RoundTranscript.challenge (roundTranscriptInterface shared)
      (roundTranscriptStart shared) roundIndex).eval env =
        (RoundTranscript.challenge (roundTranscriptInterface shared)
          (roundTranscriptOffset interface offset) roundIndex).eval env
    rw [startEq]
  have chain := SumcheckChain.spec_implies_keyChain
    relation ajtai running fresh proof (sumcheckInterface shared)
      (sumcheckOffset interface offset) env (by
        change (InitialClaim.output (initialClaimInterface shared)
          (initialClaimStart shared)).eval env = _
        have startEq : initialClaimStart shared =
            initialClaimOffset interface offset := by
          simpa [shared] using initialClaimStart_atOffset interface offset
        rw [startEq]
        exact initialEq)
      (by
        intro roundIndex
        rfl)
      (by
        exact sumcheckRoundPointEq.trans roundCoverage.1)
      (by
        have startEq : sumcheckStart shared =
            sumcheckOffset interface offset := by
          simpa [shared] using sumcheckStart_atOffset interface offset
        have wiring := congrArg (fun expression : KExpr => expression.eval env)
          (finalIdentityTerminal_eq_sumcheckOutput relation shared
            (finalIdentityOffset relation interface offset)
            (sumcheckOffset interface offset) startEq)
        exact wiring.symm.trans terminalEq)
      specification.sumcheck
  have coverage :
      NightstreamFPrime.Spec.Folding.PiCCS.Coverage
        (ProductionKey.key relation ajtai) running fresh proof := {
    transcript := (ProductionKey.key relation ajtai
      ).piCcsExecution_coins_eq_derive running fresh proof
    input_eval_K := by
      intro coordinate
      exact congrFun statementCoverage.eval_K coordinate
    input_eval_A := by
      intro coordinate
      exact congrFun statementCoverage.eval_A coordinate
    output_eval_K := OutputBinding.key_output_eval_K
      relation ajtai running fresh proof
    output_eval_A := OutputBinding.key_output_eval_A
      relation ajtai running fresh proof
    chain := by
      simpa [ChallengeDerivation.productionContext] using! chain
  }
  have outgoing := OutputBinding.spec_implies_keyOutgoingState
    relation ajtai running fresh proof (outputBindingInterface shared)
      (outputBindingOffset relation interface offset) env (by
        simpa [shared, outputBindingInterface, roundTranscriptInterface,
          atOffset] using! roundCoverage.2)
      (by
        intro source coefficient
        rfl)
      (by
        intro source matrix coefficient
        rfl)
      specification.outputBinding
  refine {
    stateBinding := specification.statementBinding.state
    accepted := (NightstreamFPrime.Spec.Folding.PiCCS.accepted_iff_coverage
      (ProductionKey.key relation ajtai) running fresh proof).mpr coverage
    roundPoint := by
      simpa [shared, running, fresh, proof] using roundCoverage.1
    outgoingState := ?_ }
  simpa [shared, running, fresh, proof, evalProof, evalOutput,
    outputBindingFinalState, outputBindingInterface, atOffset] using! outgoing

end NightstreamFPrime.Lifecycle.PiCCS.v1_1.Formal
