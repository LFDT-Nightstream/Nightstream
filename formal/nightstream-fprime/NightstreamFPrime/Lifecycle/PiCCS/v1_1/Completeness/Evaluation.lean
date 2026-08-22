import NightstreamFPrime.Lifecycle.PiCCS.v1_1.Completeness.Core

/-!
Proof-only PiCCS v1_1 assembler group. It composes opaque leaf contracts and
adds no protocol predicate, circuit row, or alternate path.
-/

namespace NightstreamFPrime.Lifecycle.PiCCS.v1_1.Formal.CompletenessSupport

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Circuit.Quadratic
open NightstreamFPrime.Gadgets.Poseidon2
open NightstreamFPrime.Gadgets.SumCheck
open NightstreamFPrime.Gadgets.Multilinear
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ConcreteCarrier

private theorem cubePoint_eq_of_coordinates
    {Field : Type} {variableCount : Nat}
    (left right : CubePoint Field variableCount)
    (coordinates : left.coordinates = right.coordinates) : left = right := by
  cases left
  cases right
  simp_all

def evaluationPrefixOps
    {logicalWidth degreeBound : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (interface : Interface logicalWidth degreeBound publicFits)
    (offset : Nat) : List Op :=
  let shared := atOffset interface offset
  [childOp "piccs.v1_1.initial_claim" (initialClaimCircuit shared)
      (initialClaimOffset interface offset),
    childOp "piccs.v1_1.sumcheck_chain" (sumcheckCircuit shared)
      (sumcheckOffset interface offset),
    childOp "piccs.v1_1.eval_K_terminal" (evalKCircuit shared)
      (evalKOffset interface offset),
    childOp "piccs.v1_1.eval_A_terminal" (evalACircuit shared)
      (evalAOffset interface offset)]

private theorem appendInitialClaim
    {logicalWidth degreeBound : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Interface logicalWidth degreeBound publicFits)
    (env : Env) (offset : Nat)
    (assumptions : Assumptions relation interface offset env)
    (before : Sequence.Prefix env offset)
    (startEq : offset + localLength before.operations =
      initialClaimOffset interface offset) :
    ∃ after : Sequence.Prefix env offset,
      after.operations = before.operations ++
        [childOp "piccs.v1_1.initial_claim"
          (initialClaimCircuit (atOffset interface offset))
            (initialClaimOffset interface offset)] ∧
      offset + localLength after.operations =
        sumcheckOffset interface offset ∧
      Sequence.PreservesPrefix before after ∧
      InitialClaim.SpecHolds
        (initialClaimInterface (atOffset interface offset))
          (initialClaimOffset interface offset) after.current := by
  let shared := atOffset interface offset
  have childAssumptions :=
    (assumptionsAt assumptions before.current).initialClaim
  have childScope := InitialClaim.flatConstraints_varsBelow
    (initialClaimInterface shared) (initialClaimOffset interface offset)
      before.current childAssumptions
  rcases InitialClaim.build (initialClaimInterface shared) before.current
      (initialClaimOffset interface offset) childAssumptions with
    ⟨built, childAgrees, childRows⟩
  rcases appendBuiltAt before "piccs.v1_1.initial_claim"
      (initialClaimCircuit shared) (initialClaimOffset interface offset)
      startEq childScope built childAgrees childRows with
    ⟨after, operationsEq, nextEq, preserves, childHolds⟩
  have childSpec := (initialClaimCircuit shared).soundness after.current
    (initialClaimOffset interface offset)
      (assumptionsAt assumptions after.current).initialClaim
        (holdsFlat_implies_holds after.current _ childHolds)
  refine ⟨after, ?_, ?_, preserves, ?_⟩
  · simpa [shared] using operationsEq
  · simpa [shared, sumcheckOffset] using nextEq
  · simpa [shared] using childSpec

private theorem sumcheckEvidence_of_accepted
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : AjtaiKey
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (interface : Interface logicalWidth
      (ProductionKey.degreeBound relation) publicFits)
    (env : Env) (offset : Nat)
    (template : Proof (ProductionKey.degreeBound relation))
    (assumptions : Assumptions relation interface offset env)
    (accepted : NightstreamFPrime.Spec.Folding.PiCCS.v1_1.Accepted
      (ProductionKey.key relation ajtai)
      (evalRunning interface offset env)
      (evalFresh interface offset env)
      (evalProof relation interface offset env template))
    (before : Sequence.Prefix env offset)
    (statementSpec : StatementAbsorption.SpecHolds
      (statementAbsorptionInterface (atOffset interface offset))
        (statementAbsorptionOffset interface offset) before.current)
    (challengeSpec : ChallengeDerivation.SpecHolds
      (challengeInterface (atOffset interface offset) offset)
        (challengeOffset interface offset) before.current)
    (roundSpec : RoundTranscript.SpecHolds
      (roundTranscriptInterface (atOffset interface offset))
        (roundTranscriptOffset interface offset) before.current)
    (initialSpec : InitialClaim.SpecHolds
      (initialClaimInterface (atOffset interface offset))
        (initialClaimOffset interface offset) before.current) :
    SumcheckChain.SpecHolds
        (sumcheckInterface (atOffset interface offset))
          (sumcheckOffset interface offset) before.current ∧
      (SumcheckChain.output (sumcheckInterface (atOffset interface offset))
        (sumcheckOffset interface offset)).eval before.current =
        ProtocolPolynomial.terminalFromMessage extensionOps
          (ChallengeDerivation.productionContext relation ajtai
            (evalRunning interface offset before.current)
            (evalFresh interface offset before.current)).input
          ((ProductionKey.key relation ajtai).piCcsExecution
            (evalRunning interface offset before.current)
            (evalFresh interface offset before.current)
            (evalProof relation interface offset before.current template)
              ).coins.alpha
          ((ProductionKey.key relation ajtai).piCcsExecution
            (evalRunning interface offset before.current)
            (evalFresh interface offset before.current)
            (evalProof relation interface offset before.current template)
              ).coins.gamma
          ((ProductionKey.key relation ajtai).piCcsExecution
            (evalRunning interface offset before.current)
            (evalFresh interface offset before.current)
            (evalProof relation interface offset before.current template)
              ).coins.roundPoint
          ((ProductionKey.key relation ajtai).piCcsCertificate
            (evalRunning interface offset before.current)
            (evalFresh interface offset before.current)
            (evalProof relation interface offset before.current template)
              ).output := by
  let shared := atOffset interface offset
  let running := evalRunning interface offset before.current
  let fresh := evalFresh interface offset before.current
  let proof := evalProof relation interface offset before.current template
  let context := ChallengeDerivation.productionContext
    relation ajtai running fresh
  have acceptedCurrent := accepted_of_agree_below relation ajtai interface
    offset env before.current template assumptions.external
      (fun index below => (before.agrees index (Or.inl below)).symm) accepted
  have coverage :=
    (NightstreamFPrime.Spec.Folding.PiCCS.v1_1.accepted_iff_coverage
      (ProductionKey.key relation ajtai) running fresh proof).mp (by
        simpa [running, fresh, proof] using acceptedCurrent)
  have statementState := StatementAbsorption.spec_implies_keyInitialState
    relation ajtai (statementAbsorptionInterface shared)
      (statementAbsorptionOffset interface offset) before.current statementSpec
  dsimp only at statementState
  rw [ProductionKey.key_oracle_eq relation ajtai] at statementState
  have challengeCoverage :=
    ChallengeDerivation.spec_implies_derivePreSumcheck
      (challengeInterface shared offset) (challengeOffset interface offset)
      before.current context (by
        simpa [shared, running, fresh, context, challengeInterface,
          statementAbsorptionInterface, atOffset, evalRunning, evalFresh]
          using statementState) challengeSpec
  have keyChallenges :=
    ChallengeDerivation.spec_implies_keyExecution_challenges
      relation ajtai running fresh proof (challengeInterface shared offset)
      (challengeOffset interface offset) before.current (by
        simpa [shared, running, fresh, context, challengeInterface,
          statementAbsorptionInterface, atOffset, evalRunning, evalFresh]
          using statementState) challengeSpec
  have roundCoverage := RoundTranscript.spec_implies_keyExecution_rounds
    relation ajtai running fresh proof (roundTranscriptInterface shared)
      (roundTranscriptOffset interface offset) before.current (by
        simpa [shared, context, challengeInterface,
          roundTranscriptInterface, atOffset] using challengeCoverage.2.2)
      (by
        intro roundIndex
        rfl)
      roundSpec
  have initialEq := InitialClaim.spec_implies_keyInitial
    relation ajtai running fresh proof (initialClaimInterface shared)
      (initialClaimOffset interface offset) before.current (by
        simpa [shared, initialClaimInterface, challengeInterface, atOffset]
          using keyChallenges.2)
      (by
        intro coordinate
        rfl)
      (by
        intro coordinate
        rfl)
      initialSpec
  have sumcheckRoundPointEq : SumcheckChain.evalRoundPoint
      (sumcheckInterface shared) (sumcheckOffset interface offset)
        before.current =
      RoundTranscript.evalRoundPoint (roundTranscriptInterface shared)
        (roundTranscriptOffset interface offset) before.current := by
    apply cubePoint_eq_of_coordinates
    change (canonicalFinIndices productionShape.cubeVariables).map
        (fun roundIndex =>
          ((roundTranscriptRound shared (sumcheckOffset interface offset)
            roundIndex).challenge).eval before.current) =
      (canonicalFinIndices productionShape.cubeVariables).map
        (fun roundIndex =>
          (RoundTranscript.challenge (roundTranscriptInterface shared)
            (roundTranscriptOffset interface offset) roundIndex).eval
              before.current)
    apply List.map_congr_left
    intro roundIndex _
    have startEq : roundTranscriptStart shared =
        roundTranscriptOffset interface offset := by
      simpa [shared] using roundTranscriptStart_atOffset interface offset
    change (RoundTranscript.challenge (roundTranscriptInterface shared)
      (roundTranscriptStart shared) roundIndex).eval before.current =
        (RoundTranscript.challenge (roundTranscriptInterface shared)
          (roundTranscriptOffset interface offset) roundIndex).eval
            before.current
    rw [startEq]
  have evidence := SumcheckChain.keyChain_implies_spec_and_terminal
    relation ajtai running fresh proof (sumcheckInterface shared)
      (sumcheckOffset interface offset) before.current
      (by
        change (InitialClaim.output (initialClaimInterface shared)
          (initialClaimStart shared)).eval before.current = _
        have startEq : initialClaimStart shared =
            initialClaimOffset interface offset := by
          simpa [shared] using initialClaimStart_atOffset interface offset
        rw [startEq]
        exact initialEq)
      (by
        intro roundIndex
        rfl)
      (sumcheckRoundPointEq.trans roundCoverage.1)
      (by
        simpa [ChallengeDerivation.productionContext] using coverage.chain)
  simpa [shared, running, fresh, proof] using evidence

private theorem appendSumcheckChain
    {logicalWidth degreeBound : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Interface logicalWidth degreeBound publicFits)
    (env : Env) (offset : Nat)
    (assumptions : Assumptions relation interface offset env)
    (before : Sequence.Prefix env offset)
    (childSpec : SumcheckChain.SpecHolds
      (sumcheckInterface (atOffset interface offset))
        (sumcheckOffset interface offset) before.current)
    (startEq : offset + localLength before.operations =
      sumcheckOffset interface offset) :
    ∃ after : Sequence.Prefix env offset,
      after.operations = before.operations ++
        [childOp "piccs.v1_1.sumcheck_chain"
          (sumcheckCircuit (atOffset interface offset))
            (sumcheckOffset interface offset)] ∧
      offset + localLength after.operations =
        evalKOffset interface offset ∧
      Sequence.PreservesPrefix before after := by
  let shared := atOffset interface offset
  let childStart := sumcheckOffset interface offset
  have childAssumptions : SumcheckChain.Assumptions
      (sumcheckInterface shared) childStart before.current :=
    (assumptionsAt assumptions before.current).sumcheck
  have zeroAssumptions : SumcheckChain.Assumptions
      (sumcheckInterface shared) childStart (fun _ => 0) := by
        simpa [SumcheckChain.Assumptions, FixedChain.Assumptions] using
          childAssumptions
  have childScope : ∀ expression ∈ flatConstraints
      (Circuit.ops (sumcheckCircuit shared).main childStart),
      expression.VarsBelow (childStart + localLength
        (Circuit.ops (sumcheckCircuit shared).main childStart)) := by
    intro expression member
    have below := SumcheckChain.flatConstraints_varsBelow
      (sumcheckInterface shared) childStart zeroAssumptions expression member
    simpa [sumcheckCircuit, SumcheckChain.localLength_eq] using below
  rcases appendAt before "piccs.v1_1.sumcheck_chain"
      (sumcheckCircuit shared) childStart startEq childScope childAssumptions
      childSpec with
    ⟨after, operationsEq, nextEq, preserves, _childHolds⟩
  refine ⟨after, ?_, ?_, preserves⟩
  · simpa [shared, childStart] using operationsEq
  · simpa [shared, childStart] using nextEq

private theorem appendEvalKTerminal
    {logicalWidth degreeBound : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Interface logicalWidth degreeBound publicFits)
    (env : Env) (offset : Nat)
    (assumptions : Assumptions relation interface offset env)
    (before : Sequence.Prefix env offset)
    (startEq : offset + localLength before.operations =
      evalKOffset interface offset) :
    ∃ after : Sequence.Prefix env offset,
      after.operations = before.operations ++
        [childOp "piccs.v1_1.eval_K_terminal"
          (evalKCircuit (atOffset interface offset))
            (evalKOffset interface offset)] ∧
      offset + localLength after.operations =
        evalAOffset interface offset ∧
      Sequence.PreservesPrefix before after ∧
      EvalKTerminal.SpecHolds
        (evalKInterface (atOffset interface offset))
          (evalKOffset interface offset) after.current := by
  let shared := atOffset interface offset
  have childAssumptions :=
    (assumptionsAt assumptions before.current).eval_K
  have childScope := EvalKTerminal.flatConstraints_varsBelow
    (evalKInterface shared) (evalKOffset interface offset)
      before.current childAssumptions
  rcases EvalKTerminal.build (evalKInterface shared) before.current
      (evalKOffset interface offset) childAssumptions with
    ⟨built, childAgrees, childRows⟩
  rcases appendBuiltAt before "piccs.v1_1.eval_K_terminal"
      (evalKCircuit shared) (evalKOffset interface offset) startEq
      childScope built childAgrees childRows with
    ⟨after, operationsEq, nextEq, preserves, childHolds⟩
  have childSpec := (evalKCircuit shared).soundness after.current
    (evalKOffset interface offset)
      (assumptionsAt assumptions after.current).eval_K
        (holdsFlat_implies_holds after.current _ childHolds)
  refine ⟨after, ?_, ?_, preserves, ?_⟩
  · simpa [shared] using operationsEq
  · simpa [shared, evalAOffset] using nextEq
  · simpa [shared] using childSpec

private theorem appendEvalATerminal
    {logicalWidth degreeBound : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Interface logicalWidth degreeBound publicFits)
    (env : Env) (offset : Nat)
    (assumptions : Assumptions relation interface offset env)
    (before : Sequence.Prefix env offset)
    (startEq : offset + localLength before.operations =
      evalAOffset interface offset) :
    ∃ after : Sequence.Prefix env offset,
      after.operations = before.operations ++
        [childOp "piccs.v1_1.eval_A_terminal"
          (evalACircuit (atOffset interface offset))
            (evalAOffset interface offset)] ∧
      offset + localLength after.operations =
        ccsOffset interface offset ∧
      Sequence.PreservesPrefix before after ∧
      EvalATerminal.SpecHolds
        (evalAInterface (atOffset interface offset))
          (evalAOffset interface offset) after.current := by
  let shared := atOffset interface offset
  have childAssumptions :=
    (assumptionsAt assumptions before.current).eval_A
  have childScope := EvalATerminal.flatConstraints_varsBelow
    (evalAInterface shared) (evalAOffset interface offset)
      before.current childAssumptions
  rcases EvalATerminal.build (evalAInterface shared) before.current
      (evalAOffset interface offset) childAssumptions with
    ⟨built, childAgrees, childRows⟩
  rcases appendBuiltAt before "piccs.v1_1.eval_A_terminal"
      (evalACircuit shared) (evalAOffset interface offset) startEq
      childScope built childAgrees childRows with
    ⟨after, operationsEq, nextEq, preserves, childHolds⟩
  have childSpec := (evalACircuit shared).soundness after.current
    (evalAOffset interface offset)
      (assumptionsAt assumptions after.current).eval_A
        (holdsFlat_implies_holds after.current _ childHolds)
  refine ⟨after, ?_, ?_, preserves, ?_⟩
  · simpa [shared] using operationsEq
  · simpa [shared, ccsOffset] using nextEq
  · simpa [shared] using childSpec

private theorem transcriptSpecs_preserved
    {logicalWidth degreeBound : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Interface logicalWidth degreeBound publicFits)
    (env : Env) (offset : Nat)
    (assumptions : Assumptions relation interface offset env)
    (before after : Sequence.Prefix env offset)
    (endEq : offset + localLength before.operations =
      initialClaimOffset interface offset)
    (preserves : Sequence.PreservesPrefix before after)
    (statementSpec : StatementAbsorption.SpecHolds
      (statementAbsorptionInterface (atOffset interface offset))
        (statementAbsorptionOffset interface offset) before.current)
    (challengeSpec : ChallengeDerivation.SpecHolds
      (challengeInterface (atOffset interface offset) offset)
        (challengeOffset interface offset) before.current)
    (roundSpec : RoundTranscript.SpecHolds
      (roundTranscriptInterface (atOffset interface offset))
        (roundTranscriptOffset interface offset) before.current) :
    StatementAbsorption.SpecHolds
        (statementAbsorptionInterface (atOffset interface offset))
          (statementAbsorptionOffset interface offset) after.current ∧
      ChallengeDerivation.SpecHolds
        (challengeInterface (atOffset interface offset) offset)
          (challengeOffset interface offset) after.current ∧
      RoundTranscript.SpecHolds
        (roundTranscriptInterface (atOffset interface offset))
          (roundTranscriptOffset interface offset) after.current := by
  have agreement : ∀ index, index < initialClaimOffset interface offset →
      after.current index = before.current index := by
    intro index below
    apply preserves.values index
    rw [endEq]
    exact below
  have statementAfter := StatementAbsorption.specHolds_of_agree_below
    (statementAbsorptionInterface (atOffset interface offset))
      (statementAbsorptionOffset interface offset) before.current after.current
      (assumptionsAt assumptions before.current).statementAbsorption
      (fun index below => agreement index (by
        have beforeChallenge : index < challengeOffset interface offset := by
          simpa only [StatementAbsorption.program_recipes_length,
            statementAbsorptionOffset_eq, challengeOffset_eq] using below
        exact lt_of_lt_of_le beforeChallenge
          (challengeOffset_le_initialClaimOffset interface offset)))
      statementSpec
  have challengeAfter := ChallengeDerivation.specHolds_of_agree_below
    (challengeInterface (atOffset interface offset) offset)
      (challengeOffset interface offset) before.current after.current
      (assumptionsAt assumptions before.current).challenge
      (fun index below => agreement index (by
        have beforeRound : index < roundTranscriptOffset interface offset := by
          simpa only [ChallengeDerivation.program_recipes_length,
            roundTranscriptOffset_eq] using below
        exact lt_of_lt_of_le beforeRound
          (roundTranscriptOffset_le_initialClaimOffset interface offset)))
      challengeSpec
  have roundEnd : roundTranscriptOffset interface offset +
      (RoundTranscript.program
        (roundTranscriptInterface (atOffset interface offset))
          (roundTranscriptOffset interface offset)).recipes.length =
      initialClaimOffset interface offset := by
    unfold initialClaimOffset nextOffset childLength roundTranscriptCircuit
    rw [FormalCircuit.withConstantFootprint_main,
      RoundTranscript.localLength_eq,
      RoundTranscript.program_recipes_length]
  have roundAfter := RoundTranscript.specHolds_of_agree_below
    (roundTranscriptInterface (atOffset interface offset))
      (roundTranscriptOffset interface offset) before.current after.current
      (assumptionsAt assumptions before.current).roundTranscript
      (fun index below => agreement index (by simpa only [roundEnd] using below))
      roundSpec
  exact ⟨statementAfter, challengeAfter, roundAfter⟩

structure Evidence
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : AjtaiKey
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (interface : Interface logicalWidth
      (ProductionKey.degreeBound relation) publicFits)
    (offset : Nat) (env : Env)
    (template : Proof (ProductionKey.degreeBound relation)) : Prop where
  alpha : ChallengeDerivation.evalAlpha
    (challengeInterface (atOffset interface offset) offset)
      (challengeOffset interface offset) env =
    ((ProductionKey.key relation ajtai).piCcsExecution
      (evalRunning interface offset env)
      (evalFresh interface offset env)
      (evalProof relation interface offset env template)).coins.alpha
  gamma : ChallengeDerivation.evalGamma
    (challengeInterface (atOffset interface offset) offset)
      (challengeOffset interface offset) env =
    ((ProductionKey.key relation ajtai).piCcsExecution
      (evalRunning interface offset env)
      (evalFresh interface offset env)
      (evalProof relation interface offset env template)).coins.gamma
  roundPoint : RoundTranscript.evalRoundPoint
    (roundTranscriptInterface (atOffset interface offset))
      (roundTranscriptOffset interface offset) env =
    ((ProductionKey.key relation ajtai).piCcsExecution
      (evalRunning interface offset env)
      (evalFresh interface offset env)
      (evalProof relation interface offset env template)).coins.roundPoint
  sumcheckTerminal :
    (SumcheckChain.output (sumcheckInterface (atOffset interface offset))
      (sumcheckOffset interface offset)).eval env =
      ProtocolPolynomial.terminalFromMessage extensionOps
        (ChallengeDerivation.productionContext relation ajtai
          (evalRunning interface offset env)
          (evalFresh interface offset env)).input
        ((ProductionKey.key relation ajtai).piCcsExecution
          (evalRunning interface offset env)
          (evalFresh interface offset env)
          (evalProof relation interface offset env template)).coins.alpha
        ((ProductionKey.key relation ajtai).piCcsExecution
          (evalRunning interface offset env)
          (evalFresh interface offset env)
          (evalProof relation interface offset env template)).coins.gamma
        ((ProductionKey.key relation ajtai).piCcsExecution
          (evalRunning interface offset env)
          (evalFresh interface offset env)
          (evalProof relation interface offset env template)).coins.roundPoint
        ((ProductionKey.key relation ajtai).piCcsCertificate
          (evalRunning interface offset env)
          (evalFresh interface offset env)
          (evalProof relation interface offset env template)).output
  eval_K : (EvalKTerminal.output
    (evalKInterface (atOffset interface offset))
      (evalKOffset interface offset)).eval env =
    ProtocolPolynomial.padAtMessage extensionOps
      (ChallengeDerivation.productionContext relation ajtai
        (evalRunning interface offset env)
        (evalFresh interface offset env)).input
      ((ProductionKey.key relation ajtai).piCcsExecution
        (evalRunning interface offset env)
        (evalFresh interface offset env)
        (evalProof relation interface offset env template)).coins.gamma
      ((ProductionKey.key relation ajtai).piCcsExecution
        (evalRunning interface offset env)
        (evalFresh interface offset env)
        (evalProof relation interface offset env template)).coins.roundPoint
      ((ProductionKey.key relation ajtai).piCcsCertificate
        (evalRunning interface offset env)
        (evalFresh interface offset env)
        (evalProof relation interface offset env template)).output
  eval_A : (EvalATerminal.output
    (evalAInterface (atOffset interface offset))
      (evalAOffset interface offset)).eval env =
    ProtocolPolynomial.matrixAtMessage extensionOps
      (ChallengeDerivation.productionContext relation ajtai
        (evalRunning interface offset env)
        (evalFresh interface offset env)).input
      ((ProductionKey.key relation ajtai).piCcsExecution
        (evalRunning interface offset env)
        (evalFresh interface offset env)
        (evalProof relation interface offset env template)).coins.gamma
      ((ProductionKey.key relation ajtai).piCcsExecution
        (evalRunning interface offset env)
        (evalFresh interface offset env)
        (evalProof relation interface offset env template)).coins.roundPoint
      ((ProductionKey.key relation ajtai).piCcsCertificate
        (evalRunning interface offset env)
        (evalFresh interface offset env)
        (evalProof relation interface offset env template)).output

private theorem evidence_of_specs
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
    (statementSpec : StatementAbsorption.SpecHolds
      (statementAbsorptionInterface (atOffset interface offset))
        (statementAbsorptionOffset interface offset) env)
    (challengeSpec : ChallengeDerivation.SpecHolds
      (challengeInterface (atOffset interface offset) offset)
        (challengeOffset interface offset) env)
    (roundSpec : RoundTranscript.SpecHolds
      (roundTranscriptInterface (atOffset interface offset))
        (roundTranscriptOffset interface offset) env)
    (sumcheckTerminal :
      (SumcheckChain.output (sumcheckInterface (atOffset interface offset))
        (sumcheckOffset interface offset)).eval env =
        ProtocolPolynomial.terminalFromMessage extensionOps
          (ChallengeDerivation.productionContext relation ajtai
            (evalRunning interface offset env)
            (evalFresh interface offset env)).input
          ((ProductionKey.key relation ajtai).piCcsExecution
            (evalRunning interface offset env)
            (evalFresh interface offset env)
            (evalProof relation interface offset env template)).coins.alpha
          ((ProductionKey.key relation ajtai).piCcsExecution
            (evalRunning interface offset env)
            (evalFresh interface offset env)
            (evalProof relation interface offset env template)).coins.gamma
          ((ProductionKey.key relation ajtai).piCcsExecution
            (evalRunning interface offset env)
            (evalFresh interface offset env)
            (evalProof relation interface offset env template)).coins.roundPoint
          ((ProductionKey.key relation ajtai).piCcsCertificate
            (evalRunning interface offset env)
            (evalFresh interface offset env)
            (evalProof relation interface offset env template)).output)
    (evalKSpec : EvalKTerminal.SpecHolds
      (evalKInterface (atOffset interface offset))
        (evalKOffset interface offset) env)
    (evalASpec : EvalATerminal.SpecHolds
      (evalAInterface (atOffset interface offset))
        (evalAOffset interface offset) env) :
    Evidence relation ajtai interface offset env template := by
  let shared := atOffset interface offset
  let running := evalRunning interface offset env
  let fresh := evalFresh interface offset env
  let proof := evalProof relation interface offset env template
  let context := ChallengeDerivation.productionContext
    relation ajtai running fresh
  have statementState := StatementAbsorption.spec_implies_keyInitialState
    relation ajtai (statementAbsorptionInterface shared)
      (statementAbsorptionOffset interface offset) env statementSpec
  dsimp only at statementState
  rw [ProductionKey.key_oracle_eq relation ajtai] at statementState
  have challengeCoverage := ChallengeDerivation.spec_implies_derivePreSumcheck
    (challengeInterface shared offset) (challengeOffset interface offset) env
      context (by
        simpa [shared, running, fresh, context, challengeInterface,
          statementAbsorptionInterface, atOffset, evalRunning, evalFresh]
          using statementState) challengeSpec
  have keyChallenges :=
    ChallengeDerivation.spec_implies_keyExecution_challenges
      relation ajtai running fresh proof (challengeInterface shared offset)
      (challengeOffset interface offset) env (by
        simpa [shared, running, fresh, context, challengeInterface,
          statementAbsorptionInterface, atOffset, evalRunning, evalFresh]
          using statementState) challengeSpec
  have roundCoverage := RoundTranscript.spec_implies_keyExecution_rounds
    relation ajtai running fresh proof (roundTranscriptInterface shared)
      (roundTranscriptOffset interface offset) env (by
        simpa [shared, context, challengeInterface,
          roundTranscriptInterface, atOffset] using challengeCoverage.2.2)
      (by
        intro roundIndex
        rfl)
      roundSpec
  have evalKEq := EvalKTerminal.spec_implies_keyPadAtMessage
    relation ajtai running fresh proof (evalKInterface shared)
      (evalKOffset interface offset) env (by
        simpa [shared, evalKInterface, roundTranscriptInterface, roundPoint,
          atOffset] using roundCoverage.1)
      (by rfl) (by
        simpa [shared, evalKInterface, challengeInterface, atOffset]
          using keyChallenges.2)
      (by
        intro coordinate
        rfl)
      evalKSpec
  have evalAEq := EvalATerminal.spec_implies_keyMatrixAtMessage
    relation ajtai running fresh proof (evalAInterface shared)
      (evalAOffset interface offset) env (by
        simpa [shared, evalAInterface, roundTranscriptInterface, roundPoint,
          atOffset] using roundCoverage.1)
      (by rfl) (by
        simpa [shared, evalAInterface, challengeInterface, atOffset]
          using keyChallenges.2)
      (by
        intro coordinate
        rfl)
      evalASpec
  exact {
    alpha := by simpa [running, fresh, proof] using keyChallenges.1
    gamma := by simpa [running, fresh, proof] using keyChallenges.2
    roundPoint := by simpa [running, fresh, proof] using roundCoverage.1
    sumcheckTerminal := sumcheckTerminal
    eval_K := by simpa [running, fresh, proof] using evalKEq
    eval_A := by simpa [running, fresh, proof] using evalAEq }

/-- Exact semantic evidence remains valid while later children allocate only
at or after the CCS boundary. This theorem uses child footprint contracts and
does not inspect child operations. -/
theorem evidence_preserved
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : AjtaiKey
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (interface : Interface logicalWidth
      (ProductionKey.degreeBound relation) publicFits)
    (initial : Env) (offset : Nat)
    (template : Proof (ProductionKey.degreeBound relation))
    (assumptions : Assumptions relation interface offset initial)
    (before after : Sequence.Prefix initial offset)
    (endEq : offset + localLength before.operations =
      ccsOffset interface offset)
    (preserves : Sequence.PreservesPrefix before after)
    (evidence : Evidence relation ajtai interface offset before.current
      template) :
    Evidence relation ajtai interface offset after.current template := by
  let shared := atOffset interface offset
  have agreesToCcs : ∀ index, index < ccsOffset interface offset →
      after.current index = before.current index := by
    intro index below
    apply preserves.values index
    rw [endEq]
    exact below
  have offsetLeCcs : offset ≤ ccsOffset interface offset := by
    rw [← endEq]
    omega
  have externalAgreement : ∀ index, index < offset →
      before.current index = after.current index := by
    intro index below
    exact (agreesToCcs index (lt_of_lt_of_le below offsetLeCcs)).symm
  have runningEq := evalRunning_eq_of_agree_below interface offset
    before.current after.current assumptions.external externalAgreement
  have freshEq := evalFresh_eq_of_agree_below interface offset
    before.current after.current assumptions.external externalAgreement
  have proofEq := evalProof_eq_of_agree_below relation interface offset
    before.current after.current template assumptions.external
      externalAgreement
  have roundTranscriptLeCcs : roundTranscriptOffset interface offset ≤
      ccsOffset interface offset := by
    unfold ccsOffset evalAOffset evalKOffset sumcheckOffset
      initialClaimOffset nextOffset childLength
    omega
  have initialClaimLeCcs : initialClaimOffset interface offset ≤
      ccsOffset interface offset := by
    unfold ccsOffset evalAOffset evalKOffset sumcheckOffset nextOffset
      childLength
    omega
  have sumcheckLeCcs : sumcheckOffset interface offset ≤
      ccsOffset interface offset := by
    unfold ccsOffset evalAOffset evalKOffset nextOffset childLength
    omega
  have evalALeCcs : evalAOffset interface offset ≤
      ccsOffset interface offset := by
    unfold ccsOffset nextOffset
    omega
  have alphaStable : ChallengeDerivation.evalAlpha
      (challengeInterface shared offset) (challengeOffset interface offset)
        after.current =
      ChallengeDerivation.evalAlpha (challengeInterface shared offset)
        (challengeOffset interface offset) before.current := by
    apply cubePoint_eq_of_coordinates
    rw [ChallengeDerivation.evalAlpha_coordinates,
      ChallengeDerivation.evalAlpha_coordinates]
    apply congrArg List.ofFn
    funext coordinate
    have below := ChallengeDerivation.alpha_varsBelow
      (challengeInterface shared offset) (challengeOffset interface offset)
        (env := before.current)
        (assumptionsAt assumptions before.current).challenge coordinate
    have belowCcs : (ChallengeDerivation.alpha
        (challengeInterface shared offset) (challengeOffset interface offset)
          coordinate).VarsBelow (ccsOffset interface offset) := by
      apply KExpr.varsBelow_mono _ below
      simpa only [ChallengeDerivation.program_recipes_length,
        roundTranscriptOffset_eq] using roundTranscriptLeCcs
    exact (ChallengeDerivation.alpha (challengeInterface shared offset)
      (challengeOffset interface offset) coordinate).eval_eq_of_agree_below
        (ccsOffset interface offset) after.current before.current belowCcs
          agreesToCcs
  have gammaStable : ChallengeDerivation.evalGamma
      (challengeInterface shared offset) (challengeOffset interface offset)
        after.current =
      ChallengeDerivation.evalGamma (challengeInterface shared offset)
        (challengeOffset interface offset) before.current := by
    rw [ChallengeDerivation.evalGamma_eq,
      ChallengeDerivation.evalGamma_eq]
    have below := ChallengeDerivation.gamma_varsBelow
      (challengeInterface shared offset) (challengeOffset interface offset)
        (env := before.current)
        (assumptionsAt assumptions before.current).challenge
    have belowCcs : (ChallengeDerivation.gamma
        (challengeInterface shared offset) (challengeOffset interface offset)
        ).VarsBelow (ccsOffset interface offset) := by
      apply KExpr.varsBelow_mono _ below
      simpa only [ChallengeDerivation.program_recipes_length,
        roundTranscriptOffset_eq] using roundTranscriptLeCcs
    exact (ChallengeDerivation.gamma (challengeInterface shared offset)
      (challengeOffset interface offset)).eval_eq_of_agree_below
        (ccsOffset interface offset) after.current before.current belowCcs
          agreesToCcs
  have roundPointStable : RoundTranscript.evalRoundPoint
      (roundTranscriptInterface shared) (roundTranscriptOffset interface offset)
        after.current =
      RoundTranscript.evalRoundPoint (roundTranscriptInterface shared)
        (roundTranscriptOffset interface offset) before.current := by
    apply cubePoint_eq_of_coordinates
    change (canonicalFinIndices productionShape.cubeVariables).map
        (fun roundIndex => (RoundTranscript.challenge
          (roundTranscriptInterface shared)
            (roundTranscriptOffset interface offset) roundIndex).eval
              after.current) =
      (canonicalFinIndices productionShape.cubeVariables).map
        (fun roundIndex => (RoundTranscript.challenge
          (roundTranscriptInterface shared)
            (roundTranscriptOffset interface offset) roundIndex).eval
              before.current)
    apply List.map_congr_left
    intro roundIndex _
    have below := RoundTranscript.challenge_varsBelow
      (roundTranscriptInterface shared) (roundTranscriptOffset interface offset)
        before.current (assumptionsAt assumptions before.current).roundTranscript
          roundIndex
    have belowCcs : (RoundTranscript.challenge
        (roundTranscriptInterface shared) (roundTranscriptOffset interface offset)
          roundIndex).VarsBelow (ccsOffset interface offset) := by
      apply KExpr.varsBelow_mono _ below
      simpa only [initialClaimOffset, nextOffset, childLength,
        roundTranscriptCircuit] using initialClaimLeCcs
    exact (RoundTranscript.challenge (roundTranscriptInterface shared)
      (roundTranscriptOffset interface offset) roundIndex
      ).eval_eq_of_agree_below (ccsOffset interface offset) after.current
        before.current belowCcs agreesToCcs
  have sumcheckStable :
      (SumcheckChain.output (sumcheckInterface shared)
        (sumcheckOffset interface offset)).eval after.current =
      (SumcheckChain.output (sumcheckInterface shared)
        (sumcheckOffset interface offset)).eval before.current := by
    have below := SumcheckChain.output_varsBelow (sumcheckInterface shared)
      (sumcheckOffset interface offset)
        ((assumptionsAt assumptions (fun _ => 0)).sumcheck)
    have belowCcs : (SumcheckChain.output (sumcheckInterface shared)
        (sumcheckOffset interface offset)).VarsBelow
          (ccsOffset interface offset) :=
      KExpr.varsBelow_mono _ below sumcheckLeCcs
    exact (SumcheckChain.output (sumcheckInterface shared)
      (sumcheckOffset interface offset)).eval_eq_of_agree_below
        (ccsOffset interface offset) after.current before.current belowCcs
          agreesToCcs
  have evalKStable : (EvalKTerminal.output (evalKInterface shared)
      (evalKOffset interface offset)).eval after.current =
    (EvalKTerminal.output (evalKInterface shared)
      (evalKOffset interface offset)).eval before.current := by
    have below := EvalKTerminal.output_varsBelow (evalKInterface shared)
      (evalKOffset interface offset) before.current
        (assumptionsAt assumptions before.current).eval_K
    have belowCcs : (EvalKTerminal.output (evalKInterface shared)
        (evalKOffset interface offset)).VarsBelow
          (ccsOffset interface offset) := by
      apply KExpr.varsBelow_mono _ below
      change evalAOffset interface offset ≤ ccsOffset interface offset
      exact evalALeCcs
    exact (EvalKTerminal.output (evalKInterface shared)
      (evalKOffset interface offset)).eval_eq_of_agree_below
        (ccsOffset interface offset) after.current before.current belowCcs
          agreesToCcs
  have evalAStable : (EvalATerminal.output (evalAInterface shared)
      (evalAOffset interface offset)).eval after.current =
    (EvalATerminal.output (evalAInterface shared)
      (evalAOffset interface offset)).eval before.current := by
    have below := EvalATerminal.output_varsBelow (evalAInterface shared)
      (evalAOffset interface offset) before.current
        (assumptionsAt assumptions before.current).eval_A
    have belowCcs : (EvalATerminal.output (evalAInterface shared)
        (evalAOffset interface offset)).VarsBelow
          (ccsOffset interface offset) := by
      simpa only [ccsOffset, nextOffset, childLength, evalACircuit] using below
    exact (EvalATerminal.output (evalAInterface shared)
      (evalAOffset interface offset)).eval_eq_of_agree_below
        (ccsOffset interface offset) after.current before.current belowCcs
          agreesToCcs
  exact {
    alpha := alphaStable.trans <| evidence.alpha.trans (by
      rw [runningEq, freshEq, proofEq])
    gamma := gammaStable.trans <| evidence.gamma.trans (by
      rw [runningEq, freshEq, proofEq])
    roundPoint := roundPointStable.trans <| evidence.roundPoint.trans (by
      rw [runningEq, freshEq, proofEq])
    sumcheckTerminal := sumcheckStable.trans <|
      evidence.sumcheckTerminal.trans (by
        rw [runningEq, freshEq, proofEq])
    eval_K := evalKStable.trans <| evidence.eval_K.trans (by
      rw [runningEq, freshEq, proofEq])
    eval_A := evalAStable.trans <| evidence.eval_A.trans (by
      rw [runningEq, freshEq, proofEq]) }

theorem completeEvaluationPrefix
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : AjtaiKey
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (interface : Interface logicalWidth
      (ProductionKey.degreeBound relation) publicFits)
    (env : Env) (offset : Nat)
    (template : Proof (ProductionKey.degreeBound relation))
    (assumptions : Assumptions relation interface offset env)
    (accepted : NightstreamFPrime.Spec.Folding.PiCCS.v1_1.Accepted
      (ProductionKey.key relation ajtai)
      (evalRunning interface offset env)
      (evalFresh interface offset env)
      (evalProof relation interface offset env template))
    (before : Sequence.Prefix env offset)
    (statementSpec : StatementAbsorption.SpecHolds
      (statementAbsorptionInterface (atOffset interface offset))
        (statementAbsorptionOffset interface offset) before.current)
    (challengeSpec : ChallengeDerivation.SpecHolds
      (challengeInterface (atOffset interface offset) offset)
        (challengeOffset interface offset) before.current)
    (roundSpec : RoundTranscript.SpecHolds
      (roundTranscriptInterface (atOffset interface offset))
        (roundTranscriptOffset interface offset) before.current)
    (startEq : offset + localLength before.operations =
      initialClaimOffset interface offset) :
    ∃ completed : Sequence.Prefix env offset,
      completed.operations =
        before.operations ++ evaluationPrefixOps interface offset ∧
      offset + localLength completed.operations =
        ccsOffset interface offset ∧
      Sequence.PreservesPrefix before completed ∧
      Evidence relation ajtai interface offset completed.current template := by
  rcases appendInitialClaim relation interface env offset assumptions
      before startEq with
    ⟨p5, o5, n5, p4to5, initialSpecP5⟩
  have transcriptP5 := transcriptSpecs_preserved relation interface env offset
    assumptions before p5 startEq p4to5 statementSpec challengeSpec roundSpec
  have sumcheckEvidenceP5 := sumcheckEvidence_of_accepted relation ajtai
    interface env offset template assumptions accepted p5 transcriptP5.1
      transcriptP5.2.1 transcriptP5.2.2 initialSpecP5
  rcases appendSumcheckChain relation interface env offset assumptions
      p5 sumcheckEvidenceP5.1 n5 with
    ⟨p6, o6, n6, p5to6⟩
  rcases appendEvalKTerminal relation interface env offset assumptions
      p6 n6 with
    ⟨p7, o7, n7, p6to7, _evalKSpecP7⟩
  rcases appendEvalATerminal relation interface env offset assumptions
      p7 n7 with
    ⟨p8, o8, n8, p7to8, evalASpecP8⟩
  have p5to8 := (p5to6.trans p6to7).trans p7to8
  have p4to8 := p4to5.trans p5to8
  have transcriptP8 := transcriptSpecs_preserved relation interface env offset
    assumptions before p8 startEq p4to8 statementSpec challengeSpec roundSpec
  have externalAgreement : ∀ index, index < offset →
      p5.current index = p8.current index := by
    intro index below
    exact (p5to8.values index (by omega)).symm
  have runningEq := evalRunning_eq_of_agree_below interface offset p5.current
    p8.current assumptions.external externalAgreement
  have freshEq := evalFresh_eq_of_agree_below interface offset p5.current
    p8.current assumptions.external externalAgreement
  have proofEq := evalProof_eq_of_agree_below relation interface offset
    p5.current p8.current template assumptions.external externalAgreement
  have sumcheckOutputBelow := SumcheckChain.output_varsBelow
    (sumcheckInterface (atOffset interface offset))
      (sumcheckOffset interface offset)
      ((assumptionsAt assumptions (fun _ => 0)).sumcheck)
  have sumcheckOutputEq :
      (SumcheckChain.output (sumcheckInterface (atOffset interface offset))
        (sumcheckOffset interface offset)).eval p8.current =
      (SumcheckChain.output (sumcheckInterface (atOffset interface offset))
        (sumcheckOffset interface offset)).eval p5.current := by
    exact (SumcheckChain.output (sumcheckInterface
      (atOffset interface offset)) (sumcheckOffset interface offset)
      ).eval_eq_of_agree_below (sumcheckOffset interface offset) p8.current
        p5.current sumcheckOutputBelow (fun index below =>
          p5to8.values index (by rw [n5]; exact below))
  have terminalP8 := sumcheckEvidenceP5.2
  rw [runningEq, freshEq, proofEq] at terminalP8
  have terminalP8' := sumcheckOutputEq.trans terminalP8
  have p8Holds := holdsFlat_implies_holds p8.current p8.operations p8.rows
  have evalKSpecP8 : EvalKTerminal.SpecHolds
      (evalKInterface (atOffset interface offset))
        (evalKOffset interface offset) p8.current := by
    have callHolds := p8Holds
      (childOp "piccs.v1_1.eval_K_terminal"
        (evalKCircuit (atOffset interface offset))
          (evalKOffset interface offset)) (by
            rw [o8, o7, o6, o5]
            simp [evaluationPrefixOps])
    change (evalKCircuit (atOffset interface offset)).assumptions
        (evalKOffset interface offset) p8.current →
      (evalKCircuit (atOffset interface offset)).spec
        (evalKOffset interface offset) p8.current at callHolds
    exact callHolds (assumptionsAt assumptions p8.current).eval_K
  have exactEvidence := evidence_of_specs relation ajtai interface offset
    p8.current template transcriptP8.1 transcriptP8.2.1 transcriptP8.2.2
      terminalP8' evalKSpecP8 evalASpecP8
  refine ⟨p8, ?_, n8, p4to8, exactEvidence⟩
  rw [o8, o7, o6, o5]
  simp [evaluationPrefixOps, List.append_assoc]

end NightstreamFPrime.Lifecycle.PiCCS.v1_1.Formal.CompletenessSupport
