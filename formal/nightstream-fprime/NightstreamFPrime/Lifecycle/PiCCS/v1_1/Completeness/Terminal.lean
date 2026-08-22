import NightstreamFPrime.Lifecycle.PiCCS.v1_1.Completeness.Evaluation

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

def terminalPrefixOps
    {logicalWidth degreeBound : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Interface logicalWidth degreeBound publicFits)
    (offset : Nat) : List Op :=
  let shared := atOffset interface offset
  [childOp "piccs.v1_1.ccs_terminal" (ccsCircuit relation shared)
      (ccsOffset interface offset),
    childOp "piccs.v1_1.norm_terminal" (normCircuit relation shared)
      (normOffset relation interface offset),
    childOp "piccs.v1_1.final_identity" (finalIdentityCircuit relation shared)
      (finalIdentityOffset relation interface offset),
    childOp "piccs.v1_1.output_binding" (outputBindingCircuit shared)
      (outputBindingOffset relation interface offset)]

private theorem appendCcsTerminal
    {logicalWidth degreeBound : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Interface logicalWidth degreeBound publicFits)
    (env : Env) (offset : Nat)
    (assumptions : Assumptions relation interface offset env)
    (before : Sequence.Prefix env offset)
    (startEq : offset + localLength before.operations =
      ccsOffset interface offset) :
    ∃ after : Sequence.Prefix env offset,
      after.operations = before.operations ++
        [childOp "piccs.v1_1.ccs_terminal"
          (ccsCircuit relation (atOffset interface offset))
            (ccsOffset interface offset)] ∧
      offset + localLength after.operations =
        normOffset relation interface offset ∧
      Sequence.PreservesPrefix before after ∧
      CcsTerminal.SpecHolds relation
        (ccsInterface relation (atOffset interface offset))
          (ccsOffset interface offset) after.current := by
  let shared := atOffset interface offset
  let childStart := ccsOffset interface offset
  have childAssumptions : CcsTerminal.Assumptions relation
      (ccsInterface relation shared) childStart before.current :=
    (assumptionsAt assumptions before.current).ccs
  have childScope : ∀ expression ∈ flatConstraints
      (Circuit.ops (ccsCircuit relation shared).main childStart),
      expression.VarsBelow (childStart + localLength
        (Circuit.ops (ccsCircuit relation shared).main childStart)) := by
    intro expression member
    have below := CcsTerminal.flatConstraints_varsBelow relation
      (ccsInterface relation shared) childStart childAssumptions
        expression member
    simpa [ccsCircuit, CcsTerminal.localLength_eq] using below
  rcases CcsTerminal.build relation (ccsInterface relation shared)
      before.current childStart childAssumptions with
    ⟨built, childAgrees, childRows⟩
  rcases appendBuiltAt before "piccs.v1_1.ccs_terminal"
      (ccsCircuit relation shared) childStart startEq childScope
      built childAgrees childRows with
    ⟨after, operationsEq, nextEq, preserves, childHolds⟩
  have childSpec := (ccsCircuit relation shared).soundness after.current
    childStart (assumptionsAt assumptions after.current).ccs
      (holdsFlat_implies_holds after.current _ childHolds)
  refine ⟨after, ?_, ?_, preserves, ?_⟩
  · simpa [shared, childStart] using operationsEq
  · simpa [shared, childStart] using nextEq
  · simpa [shared, childStart] using childSpec

private theorem appendNormTerminal
    {logicalWidth degreeBound : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Interface logicalWidth degreeBound publicFits)
    (env : Env) (offset : Nat)
    (assumptions : Assumptions relation interface offset env)
    (before : Sequence.Prefix env offset)
    (startEq : offset + localLength before.operations =
      normOffset relation interface offset) :
    ∃ after : Sequence.Prefix env offset,
      after.operations = before.operations ++
        [childOp "piccs.v1_1.norm_terminal"
          (normCircuit relation (atOffset interface offset))
            (normOffset relation interface offset)] ∧
      offset + localLength after.operations =
        finalIdentityOffset relation interface offset ∧
      Sequence.PreservesPrefix before after ∧
      NormTerminal.SpecHolds (normInterface relation
        (atOffset interface offset)) (normOffset relation interface offset)
          after.current := by
  let shared := atOffset interface offset
  let childStart := normOffset relation interface offset
  have childAssumptions := (assumptionsAt assumptions before.current).norm
  have childScope := NormTerminal.flatConstraints_varsBelow
    (normInterface relation shared) childStart before.current childAssumptions
  rcases NormTerminal.build (normInterface relation shared) before.current
      childStart childAssumptions with ⟨built, childAgrees, childRows⟩
  rcases appendBuiltAt before "piccs.v1_1.norm_terminal"
      (normCircuit relation shared) childStart startEq childScope built
        childAgrees childRows with
    ⟨after, operationsEq, nextEq, preserves, childHolds⟩
  have childSpec := (normCircuit relation shared).soundness after.current
    childStart (assumptionsAt assumptions after.current).norm
      (holdsFlat_implies_holds after.current _ childHolds)
  refine ⟨after, ?_, ?_, preserves, ?_⟩
  · simpa [shared, childStart] using operationsEq
  · simpa [shared, childStart, finalIdentityOffset] using nextEq
  · simpa [shared, childStart] using childSpec

private theorem appendFinalIdentity
    {logicalWidth degreeBound : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Interface logicalWidth degreeBound publicFits)
    (env : Env) (offset : Nat)
    (assumptions : Assumptions relation interface offset env)
    (before : Sequence.Prefix env offset)
    (childSpec : FinalIdentity.SpecHolds
      (finalIdentityInterface relation (atOffset interface offset))
        (finalIdentityOffset relation interface offset) before.current)
    (startEq : offset + localLength before.operations =
      finalIdentityOffset relation interface offset) :
    ∃ after : Sequence.Prefix env offset,
      after.operations = before.operations ++
        [childOp "piccs.v1_1.final_identity"
          (finalIdentityCircuit relation (atOffset interface offset))
            (finalIdentityOffset relation interface offset)] ∧
      offset + localLength after.operations =
        outputBindingOffset relation interface offset ∧
      Sequence.PreservesPrefix before after := by
  let shared := atOffset interface offset
  have childAssumptions :=
    (assumptionsAt assumptions before.current).finalIdentity
  have childScope := FinalIdentity.flatConstraints_varsBelow
    (finalIdentityInterface relation shared)
      (finalIdentityOffset relation interface offset) before.current
        childAssumptions
  rcases appendAt before "piccs.v1_1.final_identity"
      (finalIdentityCircuit relation shared)
      (finalIdentityOffset relation interface offset) startEq childScope
      childAssumptions childSpec with
    ⟨after, operationsEq, nextEq, preserves, _childHolds⟩
  refine ⟨after, ?_, ?_, preserves⟩
  · simpa [shared] using operationsEq
  · simpa [shared, outputBindingOffset] using nextEq

private theorem appendOutputBinding
    {logicalWidth degreeBound : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Interface logicalWidth degreeBound publicFits)
    (env : Env) (offset : Nat)
    (assumptions : Assumptions relation interface offset env)
    (before : Sequence.Prefix env offset)
    (startEq : offset + localLength before.operations =
      outputBindingOffset relation interface offset) :
    ∃ after : Sequence.Prefix env offset,
      after.operations = before.operations ++
        [childOp "piccs.v1_1.output_binding"
          (outputBindingCircuit (atOffset interface offset))
            (outputBindingOffset relation interface offset)] ∧
      offset + localLength after.operations =
        finalOffset relation interface offset ∧
      Sequence.PreservesPrefix before after ∧
      OutputBinding.SpecHolds
        (outputBindingInterface (atOffset interface offset))
        (outputBindingOffset relation interface offset) after.current := by
  let shared := atOffset interface offset
  have childAssumptions :=
    (assumptionsAt assumptions before.current).outputBinding
  have childScope := OutputBinding.flatConstraints_varsBelow
    (outputBindingInterface shared)
      (outputBindingOffset relation interface offset) before.current
        childAssumptions
  rcases OutputBinding.build (outputBindingInterface shared) before.current
      (outputBindingOffset relation interface offset) childAssumptions with
    ⟨built, childAgrees, childRows⟩
  rcases appendBuiltAt before "piccs.v1_1.output_binding"
      (outputBindingCircuit shared)
      (outputBindingOffset relation interface offset) startEq childScope
      built childAgrees childRows with
    ⟨after, operationsEq, nextEq, preserves, childHolds⟩
  have childSpec := (outputBindingCircuit shared).soundness after.current
    (outputBindingOffset relation interface offset)
      (assumptionsAt assumptions after.current).outputBinding
      (holdsFlat_implies_holds after.current _ childHolds)
  refine ⟨after, ?_, ?_, preserves, ?_⟩
  · simpa [shared] using operationsEq
  · simpa [shared, finalOffset] using nextEq
  · simpa [shared] using childSpec

private theorem roundPointEq_of_evidence
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
    (evidence : Evidence relation ajtai interface offset env template) :
    PointEquality.Owned.evalLeftPoint
      (FinalIdentity.pointInterfaceAt
        (finalIdentityInterface relation (atOffset interface offset))
          (finalIdentityOffset relation interface offset))
      (finalIdentityOffset relation interface offset) env =
    ((ProductionKey.key relation ajtai).piCcsExecution
      (evalRunning interface offset env) (evalFresh interface offset env)
      (evalProof relation interface offset env template)).coins.roundPoint := by
  simpa [finalIdentityInterface, roundTranscriptInterface, roundPoint,
    atOffset] using evidence.roundPoint

private theorem alphaEq_of_evidence
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
    (evidence : Evidence relation ajtai interface offset env template) :
    PointEquality.Owned.evalRightPoint
      (FinalIdentity.pointInterfaceAt
        (finalIdentityInterface relation (atOffset interface offset))
          (finalIdentityOffset relation interface offset))
      (finalIdentityOffset relation interface offset) env =
    ((ProductionKey.key relation ajtai).piCcsExecution
      (evalRunning interface offset env) (evalFresh interface offset env)
      (evalProof relation interface offset env template)).coins.alpha := by
  let shared := atOffset interface offset
  have interfaceEq : PointEquality.Owned.evalRightPoint
      (FinalIdentity.pointInterfaceAt (finalIdentityInterface relation shared)
        (finalIdentityOffset relation interface offset))
      (finalIdentityOffset relation interface offset) env =
      ChallengeDerivation.evalAlpha (challengeInterface shared offset)
        (challengeOffset interface offset) env := by
    apply cubePoint_eq_of_coordinates
    simpa [shared, PointEquality.Owned.evalRightPoint,
      FinalIdentity.pointInterfaceAt, finalIdentityInterface,
      challengeAlpha, challengeInterface, atOffset] using
        (ChallengeDerivation.evalAlpha_coordinates
          (challengeInterface shared offset)
            (challengeOffset interface offset) env).symm
  exact interfaceEq.trans evidence.alpha

private theorem gammaEq_of_evidence
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
    (evidence : Evidence relation ajtai interface offset env template) :
    ((finalIdentityInterface relation (atOffset interface offset)).gamma
      (finalIdentityOffset relation interface offset)).eval env =
    ((ProductionKey.key relation ajtai).piCcsExecution
      (evalRunning interface offset env) (evalFresh interface offset env)
      (evalProof relation interface offset env template)).coins.gamma := by
  let shared := atOffset interface offset
  have interfaceEq :
      ((finalIdentityInterface relation shared).gamma
        (finalIdentityOffset relation interface offset)).eval env =
      ChallengeDerivation.evalGamma (challengeInterface shared offset)
        (challengeOffset interface offset) env := by
    have challengeStartEq : challengeStart shared =
        challengeOffset interface offset := by
      simpa [shared] using challengeStart_atOffset interface offset
    rw [ChallengeDerivation.evalGamma_eq]
    change (ChallengeDerivation.gamma
      (challengeInterface shared shared.baseOffset)
        (challengeStart shared)).eval env =
      (ChallengeDerivation.gamma (challengeInterface shared offset)
        (challengeOffset interface offset)).eval env
    have baseEq : shared.baseOffset = offset := rfl
    rw [baseEq, challengeStartEq]
  exact interfaceEq.trans evidence.gamma

private theorem evalKEq_of_evidence
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
    (evidence : Evidence relation ajtai interface offset env template) :
    ((finalIdentityInterface relation (atOffset interface offset)).eval_K
      (finalIdentityOffset relation interface offset)).eval env =
    ProtocolPolynomial.padAtMessage extensionOps
      (ChallengeDerivation.productionContext relation ajtai
        (evalRunning interface offset env)
        (evalFresh interface offset env)).input
      ((ProductionKey.key relation ajtai).piCcsExecution
        (evalRunning interface offset env) (evalFresh interface offset env)
        (evalProof relation interface offset env template)).coins.gamma
      ((ProductionKey.key relation ajtai).piCcsExecution
        (evalRunning interface offset env) (evalFresh interface offset env)
        (evalProof relation interface offset env template)).coins.roundPoint
      ((ProductionKey.key relation ajtai).piCcsCertificate
        (evalRunning interface offset env) (evalFresh interface offset env)
        (evalProof relation interface offset env template)).output := by
  let shared := atOffset interface offset
  change (EvalKTerminal.output (evalKInterface shared)
    (evalKStart shared)).eval env = _
  have startEq : evalKStart shared = evalKOffset interface offset := by
    simpa [shared] using evalKStart_atOffset interface offset
  rw [startEq]
  exact evidence.eval_K

private theorem evalAEq_of_evidence
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
    (evidence : Evidence relation ajtai interface offset env template) :
    ((finalIdentityInterface relation (atOffset interface offset)).eval_A
      (finalIdentityOffset relation interface offset)).eval env =
    ProtocolPolynomial.matrixAtMessage extensionOps
      (ChallengeDerivation.productionContext relation ajtai
        (evalRunning interface offset env)
        (evalFresh interface offset env)).input
      ((ProductionKey.key relation ajtai).piCcsExecution
        (evalRunning interface offset env) (evalFresh interface offset env)
        (evalProof relation interface offset env template)).coins.gamma
      ((ProductionKey.key relation ajtai).piCcsExecution
        (evalRunning interface offset env) (evalFresh interface offset env)
        (evalProof relation interface offset env template)).coins.roundPoint
      ((ProductionKey.key relation ajtai).piCcsCertificate
        (evalRunning interface offset env) (evalFresh interface offset env)
        (evalProof relation interface offset env template)).output := by
  let shared := atOffset interface offset
  change (EvalATerminal.output (evalAInterface shared)
    (evalAStart shared)).eval env = _
  have startEq : evalAStart shared = evalAOffset interface offset := by
    simpa [shared] using evalAStart_atOffset interface offset
  rw [startEq]
  exact evidence.eval_A

private theorem ccsEq_of_specification
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
    (specification : CcsTerminal.SpecHolds relation
      (ccsInterface relation (atOffset interface offset))
        (ccsOffset interface offset) env) :
    ((finalIdentityInterface relation (atOffset interface offset)).ccs
      (finalIdentityOffset relation interface offset)).eval env =
    ProtocolPolynomial.ccsAtMessage extensionOps
      (ChallengeDerivation.productionContext relation ajtai
        (evalRunning interface offset env)
        (evalFresh interface offset env)).input
      ((ProductionKey.key relation ajtai).piCcsExecution
        (evalRunning interface offset env) (evalFresh interface offset env)
        (evalProof relation interface offset env template)).coins.gamma
      ((ProductionKey.key relation ajtai).piCcsCertificate
        (evalRunning interface offset env) (evalFresh interface offset env)
        (evalProof relation interface offset env template)).output := by
  let shared := atOffset interface offset
  have valueEq := CcsTerminal.spec_implies_keyCcsAtMessage relation ajtai
    (evalRunning interface offset env) (evalFresh interface offset env)
      (evalProof relation interface offset env template)
      (ccsInterface relation shared) (ccsOffset interface offset) env
      (by intro matrix; rfl) specification
  change (CcsTerminal.output relation (ccsInterface relation shared)
    (ccsStart shared)).eval env = _
  have startEq : ccsStart shared = ccsOffset interface offset := by
    simpa [shared] using ccsStart_atOffset interface offset
  rw [startEq]
  exact valueEq

private theorem normEq_of_evidence
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
    (evidence : Evidence relation ajtai interface offset env template)
    (specification : NormTerminal.SpecHolds
      (normInterface relation (atOffset interface offset))
        (normOffset relation interface offset) env) :
    ((finalIdentityInterface relation (atOffset interface offset)).norm
      (finalIdentityOffset relation interface offset)).eval env =
    ProtocolPolynomial.normAtMessage extensionOps
      ((ProductionKey.key relation ajtai).piCcsExecution
        (evalRunning interface offset env) (evalFresh interface offset env)
        (evalProof relation interface offset env template)).coins.gamma
      ((ProductionKey.key relation ajtai).piCcsCertificate
        (evalRunning interface offset env) (evalFresh interface offset env)
        (evalProof relation interface offset env template)).output := by
  let shared := atOffset interface offset
  have gammaEq := gammaEq_of_evidence relation ajtai interface env offset
    template evidence
  have normGammaEq :
      ((normInterface relation shared).gamma
        (normOffset relation interface offset)).eval env =
      ((ProductionKey.key relation ajtai).piCcsExecution
        (evalRunning interface offset env) (evalFresh interface offset env)
        (evalProof relation interface offset env template)).coins.gamma := by
    have wireEq :
        (normInterface relation shared).gamma
          (normOffset relation interface offset) =
        (finalIdentityInterface relation shared).gamma
          (finalIdentityOffset relation interface offset) :=
      normGamma_eq_finalIdentityGamma relation shared _ _
    exact (congrArg (KExpr.eval env) wireEq).trans gammaEq
  have valueEq := NormTerminal.spec_implies_keyNormAtMessage relation ajtai
    (evalRunning interface offset env) (evalFresh interface offset env)
      (evalProof relation interface offset env template)
      (normInterface relation shared) (normOffset relation interface offset)
      env normGammaEq
      (by intro source; rfl) specification
  have startEq : normStart shared = normOffset relation interface offset := by
    simpa [shared] using normStart_atOffset relation interface offset
  have outputWire :
      (finalIdentityInterface relation shared).norm
        (finalIdentityOffset relation interface offset) =
      NormTerminal.output (normInterface relation shared)
        (normOffset relation interface offset) :=
    finalIdentityNorm_eq_normOutput relation shared _ _ startEq
  exact (congrArg (KExpr.eval env) outputWire).trans valueEq

private theorem terminalEq_of_evidence
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
    (evidence : Evidence relation ajtai interface offset env template) :
    ((finalIdentityInterface relation (atOffset interface offset)).terminal
      (finalIdentityOffset relation interface offset)).eval env =
    ProtocolPolynomial.terminalFromMessage extensionOps
      (ChallengeDerivation.productionContext relation ajtai
        (evalRunning interface offset env)
        (evalFresh interface offset env)).input
      ((ProductionKey.key relation ajtai).piCcsExecution
        (evalRunning interface offset env) (evalFresh interface offset env)
        (evalProof relation interface offset env template)).coins.alpha
      ((ProductionKey.key relation ajtai).piCcsExecution
        (evalRunning interface offset env) (evalFresh interface offset env)
        (evalProof relation interface offset env template)).coins.gamma
      ((ProductionKey.key relation ajtai).piCcsExecution
        (evalRunning interface offset env) (evalFresh interface offset env)
        (evalProof relation interface offset env template)).coins.roundPoint
      ((ProductionKey.key relation ajtai).piCcsCertificate
        (evalRunning interface offset env) (evalFresh interface offset env)
        (evalProof relation interface offset env template)).output := by
  let shared := atOffset interface offset
  change (SumcheckChain.output (sumcheckInterface shared)
    (sumcheckStart shared)).eval env = _
  have startEq : sumcheckStart shared = sumcheckOffset interface offset := by
    simpa [shared] using sumcheckStart_atOffset interface offset
  rw [startEq]
  exact evidence.sumcheckTerminal

/-- The exact transcript, SumCheck, and separate evaluation evidence plus
the two terminal child predicates imply the v1_1 final-identity predicate. -/
private theorem finalIdentitySpec_of_evidence
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
    (evidence : Evidence relation ajtai interface offset env template)
    (ccsSpecification : CcsTerminal.SpecHolds relation
      (ccsInterface relation (atOffset interface offset))
        (ccsOffset interface offset) env)
    (normSpecification : NormTerminal.SpecHolds
      (normInterface relation (atOffset interface offset))
        (normOffset relation interface offset) env) :
    FinalIdentity.SpecHolds
      (finalIdentityInterface relation (atOffset interface offset))
        (finalIdentityOffset relation interface offset) env :=
  FinalIdentity.keyTerminal_implies_spec relation ajtai
    (evalRunning interface offset env) (evalFresh interface offset env)
    (evalProof relation interface offset env template)
    (finalIdentityInterface relation (atOffset interface offset))
    (finalIdentityOffset relation interface offset) env
    (roundPointEq_of_evidence relation ajtai interface env offset template
      evidence)
    (alphaEq_of_evidence relation ajtai interface env offset template evidence)
    (gammaEq_of_evidence relation ajtai interface env offset template evidence)
    (evalKEq_of_evidence relation ajtai interface env offset template evidence)
    (evalAEq_of_evidence relation ajtai interface env offset template evidence)
    (ccsEq_of_specification relation ajtai interface env offset template
      ccsSpecification)
    (normEq_of_evidence relation ajtai interface env offset template evidence
      normSpecification)
    (terminalEq_of_evidence relation ajtai interface env offset template
      evidence)

theorem completeTerminalPrefix
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
    (before : Sequence.Prefix env offset)
    (evidence : Evidence relation ajtai interface offset before.current
      template)
    (startEq : offset + localLength before.operations =
      ccsOffset interface offset) :
    ∃ completed : Sequence.Prefix env offset,
      completed.operations =
        before.operations ++ terminalPrefixOps relation interface offset ∧
      offset + localLength completed.operations =
        finalOffset relation interface offset ∧
      Sequence.PreservesPrefix before completed := by
  rcases appendCcsTerminal relation interface env offset assumptions
      before startEq with
    ⟨p9, o9, n9, p8to9, _ccsSpecificationP9⟩
  rcases appendNormTerminal relation interface env offset assumptions
      p9 n9 with
    ⟨p10, o10, n10, p9to10, normSpecificationP10⟩
  have p8to10 := p8to9.trans p9to10
  have exactEvidenceP10 := evidence_preserved relation ajtai interface env
    offset template assumptions before p10 startEq p8to10 evidence
  have p10Holds := holdsFlat_implies_holds p10.current p10.operations p10.rows
  have ccsSpecificationP10 : CcsTerminal.SpecHolds relation
      (ccsInterface relation (atOffset interface offset))
        (ccsOffset interface offset) p10.current := by
    have callHolds := p10Holds
      (childOp "piccs.v1_1.ccs_terminal"
        (ccsCircuit relation (atOffset interface offset))
          (ccsOffset interface offset)) (by
            rw [o10, o9]
            simp)
    change (ccsCircuit relation (atOffset interface offset)).assumptions
        (ccsOffset interface offset) p10.current →
      (ccsCircuit relation (atOffset interface offset)).spec
        (ccsOffset interface offset) p10.current at callHolds
    exact callHolds (assumptionsAt assumptions p10.current).ccs
  have finalSpecificationP10 := finalIdentitySpec_of_evidence relation ajtai
    interface p10.current offset template exactEvidenceP10
      ccsSpecificationP10 normSpecificationP10
  rcases appendFinalIdentity relation interface env offset assumptions
      p10 finalSpecificationP10 n10 with
    ⟨p11, o11, n11, p10to11⟩
  rcases appendOutputBinding relation interface env offset assumptions
      p11 n11 with
    ⟨p12, o12, n12, p11to12, _outputSpecification⟩
  refine ⟨p12, ?_, n12,
    ((p8to9.trans p9to10).trans p10to11).trans p11to12⟩
  rw [o12, o11, o10, o9]
  simp [terminalPrefixOps, List.append_assoc]

end NightstreamFPrime.Lifecycle.PiCCS.v1_1.Formal.CompletenessSupport
