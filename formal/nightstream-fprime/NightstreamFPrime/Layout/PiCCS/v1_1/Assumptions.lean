import NightstreamFPrime.Layout.PiCCS.v1_1.ProductionInputs

/-!
Paper authority: SuperNeo v1_1, section 7.3, PiCCS transcript order.
Obligation: Discharge the syntactic range assumptions of the production
PiCCS parent from its one layout-owned external-input range.

This module owns only causal scope. It supplies no protocol value, challenge,
predicate, row, or alternate circuit.
-/

namespace NightstreamFPrime.Layout.PiCCS.v1_1.Assumptions

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Circuit.Quadratic
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Lifecycle.PiCCS.v1_1
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.UnifiedSources

variable {logicalWidth : Nat}
  {publicFits : ringDegree * publicRingColumns ≤
    Phi81CarrierLayout.carrierWidth logicalWidth}

/-- The parent projection fixes the transcript challenge at the phase base;
child offsets do not select a different challenge. -/
theorem roundPoint_offset_eq
    {degreeBound : Nat}
    (interface : Formal.Interface logicalWidth degreeBound publicFits)
    (leftOffset rightOffset : Nat)
    (coordinate : Fin productionShape.cubeVariables) :
    Formal.roundPoint interface leftOffset coordinate =
      Formal.roundPoint interface rightOffset coordinate := by
  rfl

/-- The parent projection fixes `gamma` at the challenge child; later child
offsets do not select a different value. -/
theorem challengeGamma_offset_eq
    {degreeBound : Nat}
    (interface : Formal.Interface logicalWidth degreeBound publicFits)
    (leftOffset rightOffset : Nat) :
    Formal.challengeGamma interface leftOffset =
      Formal.challengeGamma interface rightOffset := by
  rfl

theorem challengeAlpha_offset_eq
    {degreeBound : Nat}
    (interface : Formal.Interface logicalWidth degreeBound publicFits)
    (leftOffset rightOffset : Nat)
    (coordinate : Fin productionShape.cubeVariables) :
    Formal.challengeAlpha interface leftOffset coordinate =
      Formal.challengeAlpha interface rightOffset coordinate := by
  rfl

theorem evalKOutput_offset_eq
    {degreeBound : Nat}
    (interface : Formal.Interface logicalWidth degreeBound publicFits)
    (leftOffset rightOffset : Nat) :
    Formal.evalKOutput interface leftOffset =
      Formal.evalKOutput interface rightOffset := by
  rfl

theorem evalAOutput_offset_eq
    {degreeBound : Nat}
    (interface : Formal.Interface logicalWidth degreeBound publicFits)
    (leftOffset rightOffset : Nat) :
    Formal.evalAOutput interface leftOffset =
      Formal.evalAOutput interface rightOffset := by
  rfl

theorem ccsOutput_offset_eq
    {degreeBound : Nat}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth degreeBound publicFits)
    (leftOffset rightOffset : Nat) :
    Formal.ccsOutput relation interface leftOffset =
      Formal.ccsOutput relation interface rightOffset := by
  rfl

theorem normOutput_offset_eq
    {degreeBound : Nat}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth degreeBound publicFits)
    (leftOffset rightOffset : Nat) :
    Formal.normOutput relation interface leftOffset =
      Formal.normOutput relation interface rightOffset := by
  rfl

theorem sumcheckOutput_offset_eq
    {degreeBound : Nat}
    (interface : Formal.Interface logicalWidth degreeBound publicFits)
    (leftOffset rightOffset : Nat) :
    Formal.sumcheckOutput interface leftOffset =
      Formal.sumcheckOutput interface rightOffset := by
  rfl

/-- The four transcript-prefix assumption fields of the canonical parent. -/
structure Transcript
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth
      (ProductionKey.degreeBound relation) publicFits)
    (parentOffset : Nat) (env : Env) : Prop where
  statementBinding :
    (Formal.statementBindingCircuit (Formal.atOffset interface parentOffset)
      ).assumptions parentOffset env
  statementAbsorption :
    (Formal.statementAbsorptionCircuit (Formal.atOffset interface parentOffset)
      ).assumptions (Formal.statementAbsorptionOffset interface parentOffset) env
  challenge : (Formal.challengeCircuit interface parentOffset).assumptions
    (Formal.challengeOffset interface parentOffset) env
  roundTranscript :
    (Formal.roundTranscriptCircuit (Formal.atOffset interface parentOffset)
      ).assumptions (Formal.roundTranscriptOffset interface parentOffset) env

/-- The production external-input range and prior child scopes imply every
transcript-prefix assumption. -/
theorem transcript
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth
      (ProductionKey.degreeBound relation) publicFits)
    (parentOffset : Nat)
    (external : ProductionInputs.ExternalInputsLinear interface parentOffset)
    (env : Env) : Transcript relation interface parentOffset env := by
  let frozen := Formal.atOffset interface parentOffset
  have stateAssumption : StateBinding.Assumptions
      (Formal.statementBindingInterface frozen).state parentOffset env := by
    refine ⟨?_, ?_, ?_, ?_, ?_⟩
    · intro word member
      simpa [frozen, Formal.atOffset, Formal.statementBindingInterface] using
        external.below.priorStateFixed word member
    · intro word member
      simpa [frozen, Formal.atOffset, Formal.statementBindingInterface] using
        external.below.outputStateFixed word member
    · intro lane
      simpa [frozen, Formal.atOffset, Formal.statementBindingInterface] using
        external.below.priorStateContext lane
    · intro lane
      simpa [frozen, Formal.atOffset, Formal.statementBindingInterface] using
        external.below.outputStateContext lane
    · intro lane
      simpa [frozen, Formal.atOffset, Formal.statementBindingInterface] using
        external.below.expectedContext lane
  have statementInputs :
      Leaves.StatementAbsorption.InputsBelow
        (Formal.statementAbsorptionInterface frozen) parentOffset := by
    refine ⟨?_, ?_⟩
    · intro source row coefficient
      simpa [frozen, Formal.atOffset, Formal.statementAbsorptionInterface] using
        external.below.freshCommitment source row coefficient
    · intro source column
      simpa [frozen, Formal.atOffset, Formal.statementAbsorptionInterface] using
        external.below.freshPublicInput source column
  have statementAssumption : StatementAbsorption.Assumptions
      (Formal.statementAbsorptionInterface frozen) parentOffset env :=
    Leaves.StatementAbsorption.assumptions_of_inputsBelow
      (Formal.statementAbsorptionInterface frozen) parentOffset statementInputs
      env
  have challengeAssumption : ChallengeDerivation.Assumptions
      (Formal.challengeInterface frozen parentOffset)
      (Formal.challengeOffset interface parentOffset) env := by
    intro lane
    have below := Leaves.StatementAbsorption.finalState_varsBelow
      (Formal.statementAbsorptionInterface frozen) parentOffset statementInputs
      lane
    simpa [Formal.challengeInterface, Formal.statementFinalState, frozen,
      Formal.challengeOffset_eq] using below
  have roundAssumption : RoundTranscript.Assumptions
      (Formal.roundTranscriptInterface frozen)
      (Formal.roundTranscriptOffset interface parentOffset) env := by
    constructor
    · intro lane
      have below := ChallengeDerivation.finalState_varsBelow
        (Formal.challengeInterface frozen parentOffset)
        (Formal.challengeOffset interface parentOffset) challengeAssumption lane
      simpa [Formal.roundTranscriptInterface, Formal.challengeFinalState,
        Formal.challengeStart, frozen, Formal.roundTranscriptOffset_eq] using
        below
    · intro roundIndex coefficient
      have sourceBelow := external.below.roundCoefficient roundIndex coefficient
      have offsetLe : parentOffset ≤
          Formal.roundTranscriptOffset interface parentOffset := by
        rw [Formal.roundTranscriptOffset_eq, Formal.challengeOffset_eq]
        omega
      have widened := KExpr.varsBelow_mono _ sourceBelow offsetLe
      simpa [Formal.roundTranscriptInterface, frozen, Formal.atOffset] using
        widened
  refine {
    statementBinding := stateAssumption
    statementAbsorption := ?_
    challenge := ?_
    roundTranscript := ?_ }
  · rw [Formal.statementAbsorptionOffset_eq]
    exact statementAssumption
  · exact challengeAssumption
  · exact roundAssumption

/-- The transcript-derived `gamma` and the separate external `Eval_K` and
`Eval_A` families discharge the initial-claim child assumption. -/
theorem initialClaim
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth
      (ProductionKey.degreeBound relation) publicFits)
    (parentOffset : Nat)
    (external : ProductionInputs.ExternalInputsLinear interface parentOffset)
    (env : Env) :
    (Formal.initialClaimCircuit (Formal.atOffset interface parentOffset)
      ).assumptions (Formal.initialClaimOffset interface parentOffset) env := by
  let frozen := Formal.atOffset interface parentOffset
  let challengeAt := Formal.challengeOffset interface parentOffset
  let roundAt := Formal.roundTranscriptOffset interface parentOffset
  let initialAt := Formal.initialClaimOffset interface parentOffset
  have transcriptAssumptions :=
    transcript relation interface parentOffset external env
  have challengeAssumption : ChallengeDerivation.Assumptions
      (Formal.challengeInterface frozen parentOffset) challengeAt env := by
    exact transcriptAssumptions.challenge
  have gammaAtRound :
      (Formal.challengeGamma frozen parentOffset).VarsBelow roundAt := by
    have below := ChallengeDerivation.gamma_varsBelow
      (Formal.challengeInterface frozen parentOffset) challengeAt
      challengeAssumption
    simpa [Formal.challengeGamma, Formal.challengeStart, frozen, challengeAt,
      roundAt, Formal.atOffset, Formal.roundTranscriptOffset_eq,
      Formal.challengeOffset_eq] using below
  have roundLeInitial : roundAt ≤ initialAt := by
    exact Formal.roundTranscriptOffset_le_initialClaimOffset interface
      parentOffset
  have parentLeInitial : parentOffset ≤ initialAt := by
    have parentLeRound : parentOffset ≤ roundAt := by
      dsimp [roundAt]
      rw [Formal.roundTranscriptOffset_eq, Formal.challengeOffset_eq]
      omega
    exact Nat.le_trans parentLeRound roundLeInitial
  change InitialClaim.Assumptions (Formal.initialClaimInterface frozen)
    initialAt env
  refine ⟨?_, ?_⟩
  · change (Formal.challengeGamma frozen initialAt).VarsBelow initialAt
    exact KExpr.varsBelow_mono _ gammaAtRound roundLeInitial
  · intro coefficient member
    change coefficient ∈
      InitialClaim.coefficientExprs (Formal.initialClaimInterface frozen)
        initialAt at member
    rw [InitialClaim.coefficientExprs, List.mem_append] at member
    rcases member with member | member
    · rw [List.mem_map] at member
      rcases member with ⟨coordinate, _, rfl⟩
      have widened := KExpr.varsBelow_mono _
        (external.below.runningEval_K coordinate.running
          coordinate.coefficient) parentLeInitial
      simpa [Formal.initialClaimInterface, frozen, Formal.atOffset] using widened
    · rw [List.mem_map] at member
      rcases member with ⟨coordinate, _, rfl⟩
      have widened := KExpr.varsBelow_mono _
        (external.below.runningEval_A coordinate.running coordinate.matrix
          coordinate.coefficient) parentLeInitial
      simpa [Formal.initialClaimInterface, frozen, Formal.atOffset] using widened

/-- The initial-claim output, external round messages, and transcript-derived
round challenges discharge the fixed 25-round SumCheck assumption. -/
theorem sumcheck
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth
      (ProductionKey.degreeBound relation) publicFits)
    (parentOffset : Nat)
    (external : ProductionInputs.ExternalInputsLinear interface parentOffset)
    (env : Env) :
    (Formal.sumcheckCircuit (Formal.atOffset interface parentOffset)
      ).assumptions (Formal.sumcheckOffset interface parentOffset) env := by
  let frozen := Formal.atOffset interface parentOffset
  let roundAt := Formal.roundTranscriptOffset interface parentOffset
  let initialAt := Formal.initialClaimOffset interface parentOffset
  let sumcheckAt := Formal.sumcheckOffset interface parentOffset
  have initialCircuitAssumption :=
    initialClaim relation interface parentOffset external env
  have transcriptAssumptions :=
    transcript relation interface parentOffset external env
  have initialLeSumcheck : initialAt ≤ sumcheckAt := by
    dsimp [sumcheckAt]
    unfold Formal.sumcheckOffset Formal.nextOffset
    omega
  have parentLeSumcheck : parentOffset ≤ sumcheckAt := by
    have parentLeInitial : parentOffset ≤ initialAt := by
      have parentLeRound : parentOffset ≤ roundAt := by
        dsimp [roundAt]
        rw [Formal.roundTranscriptOffset_eq, Formal.challengeOffset_eq]
        omega
      exact Nat.le_trans parentLeRound
        (Formal.roundTranscriptOffset_le_initialClaimOffset interface
          parentOffset)
    exact Nat.le_trans parentLeInitial initialLeSumcheck
  have initialOutputBelow :
      (Formal.initialClaimOutput frozen sumcheckAt).VarsBelow sumcheckAt := by
    exact Leaves.InitialClaim.output_varsBelow_sumcheck interface parentOffset
      env initialCircuitAssumption
  have roundChallengeBelow (roundIndex : Fin productionShape.cubeVariables) :
      (Formal.roundPoint frozen sumcheckAt roundIndex).VarsBelow sumcheckAt := by
    have belowInitial :
        (Formal.roundPoint frozen sumcheckAt roundIndex).VarsBelow initialAt := by
      rw [roundPoint_offset_eq frozen sumcheckAt initialAt roundIndex]
      exact Leaves.RoundTranscript.challenge_varsBelow_initialClaim interface
        parentOffset env transcriptAssumptions.roundTranscript roundIndex
    exact KExpr.varsBelow_mono _ belowInitial initialLeSumcheck
  change SumcheckChain.Assumptions (Formal.sumcheckInterface frozen)
    sumcheckAt env
  refine ⟨?_, ?_⟩
  · exact initialOutputBelow
  · intro roundIndex
    constructor
    · intro coefficient
      have sourceBelow := external.below.roundCoefficient roundIndex coefficient
      have widened := KExpr.varsBelow_mono _ sourceBelow parentLeSumcheck
      simpa [Formal.sumcheckInterface, Formal.roundTranscriptRound, frozen,
        Formal.atOffset] using widened
    · exact roundChallengeBelow roundIndex

/-- The transcript point, prior point, `gamma`, and only the Pad-family output
coefficients discharge the separate `Eval_K` child assumption. -/
theorem evalK
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth
      (ProductionKey.degreeBound relation) publicFits)
    (parentOffset : Nat)
    (external : ProductionInputs.ExternalInputsLinear interface parentOffset)
    (env : Env) :
    (Formal.evalKCircuit (Formal.atOffset interface parentOffset)).assumptions
      (Formal.evalKOffset interface parentOffset) env := by
  let frozen := Formal.atOffset interface parentOffset
  let initialAt := Formal.initialClaimOffset interface parentOffset
  let sumcheckAt := Formal.sumcheckOffset interface parentOffset
  let evalKAt := Formal.evalKOffset interface parentOffset
  have initialCircuitAssumption :=
    initialClaim relation interface parentOffset external env
  have initialAssumption := initialCircuitAssumption
  change InitialClaim.Assumptions (Formal.initialClaimInterface frozen)
    initialAt env at initialAssumption
  have transcriptAssumptions :=
    transcript relation interface parentOffset external env
  have initialLeSumcheck : initialAt ≤ sumcheckAt := by
    dsimp [sumcheckAt]
    unfold Formal.sumcheckOffset Formal.nextOffset
    omega
  have sumcheckLeEvalK : sumcheckAt ≤ evalKAt := by
    dsimp [evalKAt]
    unfold Formal.evalKOffset Formal.nextOffset
    omega
  have initialLeEvalK : initialAt ≤ evalKAt :=
    Nat.le_trans initialLeSumcheck sumcheckLeEvalK
  have parentLeEvalK : parentOffset ≤ evalKAt := by
    have parentLeRound : parentOffset ≤
        Formal.roundTranscriptOffset interface parentOffset := by
      rw [Formal.roundTranscriptOffset_eq, Formal.challengeOffset_eq]
      omega
    have parentLeInitial := Nat.le_trans parentLeRound
      (Formal.roundTranscriptOffset_le_initialClaimOffset interface
        parentOffset)
    exact Nat.le_trans parentLeInitial initialLeEvalK
  have roundPointBelow (coordinate : Fin productionShape.cubeVariables) :
      (Formal.roundPoint frozen evalKAt coordinate).VarsBelow evalKAt := by
    have belowInitial :=
      Leaves.RoundTranscript.challenge_varsBelow_initialClaim interface
        parentOffset env transcriptAssumptions.roundTranscript coordinate
    rw [roundPoint_offset_eq frozen evalKAt initialAt coordinate]
    exact KExpr.varsBelow_mono _ belowInitial initialLeEvalK
  have gammaBelow :
      (Formal.challengeGamma frozen evalKAt).VarsBelow evalKAt := by
    rw [challengeGamma_offset_eq frozen evalKAt initialAt]
    exact KExpr.varsBelow_mono _ initialAssumption.1 initialLeEvalK
  change EvalKTerminal.Assumptions (Formal.evalKInterface frozen) evalKAt env
  refine ⟨?_, ?_⟩
  · intro coordinate
    change (Formal.roundPoint frozen evalKAt coordinate).VarsBelow evalKAt ∧
      ((interface.running parentOffset).point coordinate).VarsBelow evalKAt
    exact ⟨roundPointBelow coordinate,
      KExpr.varsBelow_mono _ (external.below.runningPoint coordinate)
        parentLeEvalK⟩
  · constructor
    · exact gammaBelow
    · intro coefficient member
      change coefficient ∈ EvalKTerminal.coefficientExprs
        (Formal.evalKInterface frozen) evalKAt at member
      rw [EvalKTerminal.coefficientExprs, List.mem_map] at member
      rcases member with ⟨coordinate, _, rfl⟩
      have widened := KExpr.varsBelow_mono _
        (external.below.outputEval_K
          (runningSourceIndex coordinate.running) coordinate.coefficient)
        parentLeEvalK
      simpa [Formal.evalKInterface, frozen, Formal.atOffset] using widened

/-- The transcript point, prior point, `gamma`, and only the genuine
CCS-matrix output coefficients discharge the separate `Eval_A` assumption. -/
theorem evalA
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth
      (ProductionKey.degreeBound relation) publicFits)
    (parentOffset : Nat)
    (external : ProductionInputs.ExternalInputsLinear interface parentOffset)
    (env : Env) :
    (Formal.evalACircuit (Formal.atOffset interface parentOffset)).assumptions
      (Formal.evalAOffset interface parentOffset) env := by
  let frozen := Formal.atOffset interface parentOffset
  let initialAt := Formal.initialClaimOffset interface parentOffset
  let sumcheckAt := Formal.sumcheckOffset interface parentOffset
  let evalKAt := Formal.evalKOffset interface parentOffset
  let evalAAt := Formal.evalAOffset interface parentOffset
  have initialCircuitAssumption :=
    initialClaim relation interface parentOffset external env
  have initialAssumption := initialCircuitAssumption
  change InitialClaim.Assumptions (Formal.initialClaimInterface frozen)
    initialAt env at initialAssumption
  have transcriptAssumptions :=
    transcript relation interface parentOffset external env
  have initialLeSumcheck : initialAt ≤ sumcheckAt := by
    dsimp [sumcheckAt]
    unfold Formal.sumcheckOffset Formal.nextOffset
    omega
  have sumcheckLeEvalK : sumcheckAt ≤ evalKAt := by
    dsimp [evalKAt]
    unfold Formal.evalKOffset Formal.nextOffset
    omega
  have evalKLeEvalA : evalKAt ≤ evalAAt := by
    dsimp [evalAAt]
    unfold Formal.evalAOffset Formal.nextOffset
    omega
  have initialLeEvalA : initialAt ≤ evalAAt :=
    Nat.le_trans (Nat.le_trans initialLeSumcheck sumcheckLeEvalK) evalKLeEvalA
  have parentLeEvalA : parentOffset ≤ evalAAt := by
    have parentLeRound : parentOffset ≤
        Formal.roundTranscriptOffset interface parentOffset := by
      rw [Formal.roundTranscriptOffset_eq, Formal.challengeOffset_eq]
      omega
    have parentLeInitial := Nat.le_trans parentLeRound
      (Formal.roundTranscriptOffset_le_initialClaimOffset interface
        parentOffset)
    exact Nat.le_trans parentLeInitial initialLeEvalA
  have roundPointBelow (coordinate : Fin productionShape.cubeVariables) :
      (Formal.roundPoint frozen evalAAt coordinate).VarsBelow evalAAt := by
    have belowInitial :=
      Leaves.RoundTranscript.challenge_varsBelow_initialClaim interface
        parentOffset env transcriptAssumptions.roundTranscript coordinate
    rw [roundPoint_offset_eq frozen evalAAt initialAt coordinate]
    exact KExpr.varsBelow_mono _ belowInitial initialLeEvalA
  have gammaBelow :
      (Formal.challengeGamma frozen evalAAt).VarsBelow evalAAt := by
    rw [challengeGamma_offset_eq frozen evalAAt initialAt]
    exact KExpr.varsBelow_mono _ initialAssumption.1 initialLeEvalA
  change EvalATerminal.Assumptions (Formal.evalAInterface frozen) evalAAt env
  refine ⟨?_, ?_⟩
  · intro coordinate
    change (Formal.roundPoint frozen evalAAt coordinate).VarsBelow evalAAt ∧
      ((interface.running parentOffset).point coordinate).VarsBelow evalAAt
    exact ⟨roundPointBelow coordinate,
      KExpr.varsBelow_mono _ (external.below.runningPoint coordinate)
        parentLeEvalA⟩
  · constructor
    · exact gammaBelow
    · intro coefficient member
      change coefficient ∈ EvalATerminal.coefficientExprs
        (Formal.evalAInterface frozen) evalAAt at member
      rw [EvalATerminal.coefficientExprs, List.mem_map] at member
      rcases member with ⟨coordinate, _, rfl⟩
      have widened := KExpr.varsBelow_mono _
        (external.below.outputEval_A
          (runningSourceIndex coordinate.running) coordinate.matrix
          coordinate.coefficient) parentLeEvalA
      simpa [Formal.evalAInterface, frozen, Formal.atOffset] using widened

/-- The fresh source's genuine matrix-family values discharge the CCS
terminal assumption at the verifier-owned constant coefficient. -/
theorem ccs
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth
      (ProductionKey.degreeBound relation) publicFits)
    (parentOffset : Nat)
    (external : ProductionInputs.ExternalInputsLinear interface parentOffset)
    (env : Env) :
    (Formal.ccsCircuit relation (Formal.atOffset interface parentOffset)
      ).assumptions (Formal.ccsOffset interface parentOffset) env := by
  let frozen := Formal.atOffset interface parentOffset
  let initialAt := Formal.initialClaimOffset interface parentOffset
  let sumcheckAt := Formal.sumcheckOffset interface parentOffset
  let evalKAt := Formal.evalKOffset interface parentOffset
  let evalAAt := Formal.evalAOffset interface parentOffset
  let ccsAt := Formal.ccsOffset interface parentOffset
  have parentLeRound : parentOffset ≤
      Formal.roundTranscriptOffset interface parentOffset := by
    rw [Formal.roundTranscriptOffset_eq, Formal.challengeOffset_eq]
    omega
  have parentLeInitial : parentOffset ≤ initialAt :=
    Nat.le_trans parentLeRound
      (Formal.roundTranscriptOffset_le_initialClaimOffset interface parentOffset)
  have initialLeSumcheck : initialAt ≤ sumcheckAt := by
    dsimp [sumcheckAt]
    unfold Formal.sumcheckOffset Formal.nextOffset
    omega
  have sumcheckLeEvalK : sumcheckAt ≤ evalKAt := by
    dsimp [evalKAt]
    unfold Formal.evalKOffset Formal.nextOffset
    omega
  have evalKLeEvalA : evalKAt ≤ evalAAt := by
    dsimp [evalAAt]
    unfold Formal.evalAOffset Formal.nextOffset
    omega
  have evalALeCcs : evalAAt ≤ ccsAt := by
    dsimp [ccsAt]
    unfold Formal.ccsOffset Formal.nextOffset
    omega
  have parentLeCcs := Nat.le_trans parentLeInitial
    (Nat.le_trans initialLeSumcheck
      (Nat.le_trans sumcheckLeEvalK
        (Nat.le_trans evalKLeEvalA evalALeCcs)))
  change CcsTerminal.Assumptions relation (Formal.ccsInterface relation frozen)
    ccsAt env
  intro matrix
  have sourceBelow := external.below.outputEval_A
    (freshSourceIndex Formal.freshIndex) matrix
    (Formal.constantCoefficient relation)
  have widened := KExpr.varsBelow_mono _ sourceBelow parentLeCcs
  simpa [Formal.ccsInterface, frozen, Formal.atOffset] using widened

/-- The transcript-derived `gamma` and the 17 strict residuals over the
separate Pad-family outputs discharge the norm child assumption. -/
theorem norm
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth
      (ProductionKey.degreeBound relation) publicFits)
    (parentOffset : Nat)
    (external : ProductionInputs.ExternalInputsLinear interface parentOffset)
    (env : Env) :
    (Formal.normCircuit relation (Formal.atOffset interface parentOffset)
      ).assumptions (Formal.normOffset relation interface parentOffset) env := by
  let frozen := Formal.atOffset interface parentOffset
  let initialAt := Formal.initialClaimOffset interface parentOffset
  let sumcheckAt := Formal.sumcheckOffset interface parentOffset
  let evalKAt := Formal.evalKOffset interface parentOffset
  let evalAAt := Formal.evalAOffset interface parentOffset
  let ccsAt := Formal.ccsOffset interface parentOffset
  let normAt := Formal.normOffset relation interface parentOffset
  have initialCircuitAssumption :=
    initialClaim relation interface parentOffset external env
  have initialAssumption := initialCircuitAssumption
  change InitialClaim.Assumptions (Formal.initialClaimInterface frozen)
    initialAt env at initialAssumption
  have parentLeRound : parentOffset ≤
      Formal.roundTranscriptOffset interface parentOffset := by
    rw [Formal.roundTranscriptOffset_eq, Formal.challengeOffset_eq]
    omega
  have parentLeInitial : parentOffset ≤ initialAt :=
    Nat.le_trans parentLeRound
      (Formal.roundTranscriptOffset_le_initialClaimOffset interface parentOffset)
  have initialLeSumcheck : initialAt ≤ sumcheckAt := by
    dsimp [sumcheckAt]
    unfold Formal.sumcheckOffset Formal.nextOffset
    omega
  have sumcheckLeEvalK : sumcheckAt ≤ evalKAt := by
    dsimp [evalKAt]
    unfold Formal.evalKOffset Formal.nextOffset
    omega
  have evalKLeEvalA : evalKAt ≤ evalAAt := by
    dsimp [evalAAt]
    unfold Formal.evalAOffset Formal.nextOffset
    omega
  have evalALeCcs : evalAAt ≤ ccsAt := by
    dsimp [ccsAt]
    unfold Formal.ccsOffset Formal.nextOffset
    omega
  have ccsLeNorm : ccsAt ≤ normAt := by
    dsimp [normAt]
    unfold Formal.normOffset Formal.nextOffset
    omega
  have initialLeNorm := Nat.le_trans initialLeSumcheck
    (Nat.le_trans sumcheckLeEvalK
      (Nat.le_trans evalKLeEvalA (Nat.le_trans evalALeCcs ccsLeNorm)))
  have parentLeNorm := Nat.le_trans parentLeInitial initialLeNorm
  have gammaBelow :
      (Formal.challengeGamma frozen normAt).VarsBelow normAt := by
    rw [challengeGamma_offset_eq frozen normAt initialAt]
    exact KExpr.varsBelow_mono _ initialAssumption.1 initialLeNorm
  change NormTerminal.Assumptions (Formal.normInterface relation frozen)
    normAt env
  refine ⟨gammaBelow, ?_⟩
  intro coefficient member
  change coefficient ∈ NormTerminal.coefficientExprs
    (Formal.normInterface relation frozen) normAt at member
  rw [NormTerminal.coefficientExprs, List.mem_map] at member
  rcases member with ⟨source, _, rfl⟩
  have sourceBelow := external.below.outputEval_K source
    (Formal.constantCoefficient relation)
  have widened := KExpr.varsBelow_mono _ sourceBelow parentLeNorm
  have residualBelow := Leaves.NormTerminal.residualExpr_varsBelow
    ((Formal.normInterface relation frozen).sourceAssignment normAt source)
    normAt (by
      simpa [Formal.normInterface, frozen, Formal.atOffset] using widened)
  exact residualBelow

/-- All seven separately owned terminal values discharge the exact v1_1 final
joint-identity assumption. -/
theorem finalIdentity
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth
      (ProductionKey.degreeBound relation) publicFits)
    (parentOffset : Nat)
    (external : ProductionInputs.ExternalInputsLinear interface parentOffset)
    (env : Env) :
    (Formal.finalIdentityCircuit relation
      (Formal.atOffset interface parentOffset)).assumptions
      (Formal.finalIdentityOffset relation interface parentOffset) env := by
  let frozen := Formal.atOffset interface parentOffset
  let challengeAt := Formal.challengeOffset interface parentOffset
  let roundAt := Formal.roundTranscriptOffset interface parentOffset
  let initialAt := Formal.initialClaimOffset interface parentOffset
  let sumcheckAt := Formal.sumcheckOffset interface parentOffset
  let evalKAt := Formal.evalKOffset interface parentOffset
  let evalAAt := Formal.evalAOffset interface parentOffset
  let ccsAt := Formal.ccsOffset interface parentOffset
  let normAt := Formal.normOffset relation interface parentOffset
  let finalAt := Formal.finalIdentityOffset relation interface parentOffset
  have transcriptAssumptions :=
    transcript relation interface parentOffset external env
  have initialCircuitAssumption :=
    initialClaim relation interface parentOffset external env
  have initialAssumption := initialCircuitAssumption
  change InitialClaim.Assumptions (Formal.initialClaimInterface frozen)
    initialAt env at initialAssumption
  have sumcheckAssumption :=
    sumcheck relation interface parentOffset external env
  have evalKAssumption := evalK relation interface parentOffset external env
  have evalAAssumption := evalA relation interface parentOffset external env
  have normAssumption := norm relation interface parentOffset external env
  have initialLeSumcheck : initialAt ≤ sumcheckAt := by
    dsimp [sumcheckAt]
    unfold Formal.sumcheckOffset Formal.nextOffset
    omega
  have sumcheckLeEvalK : sumcheckAt ≤ evalKAt := by
    dsimp [evalKAt]
    unfold Formal.evalKOffset Formal.nextOffset
    omega
  have evalKLeEvalA : evalKAt ≤ evalAAt := by
    dsimp [evalAAt]
    unfold Formal.evalAOffset Formal.nextOffset
    omega
  have evalALeCcs : evalAAt ≤ ccsAt := by
    dsimp [ccsAt]
    unfold Formal.ccsOffset Formal.nextOffset
    omega
  have ccsLeNorm : ccsAt ≤ normAt := by
    dsimp [normAt]
    unfold Formal.normOffset Formal.nextOffset
    omega
  have normLeFinal : normAt ≤ finalAt := by
    dsimp [finalAt]
    unfold Formal.finalIdentityOffset Formal.nextOffset
    omega
  have initialLeFinal := Nat.le_trans initialLeSumcheck
    (Nat.le_trans sumcheckLeEvalK
      (Nat.le_trans evalKLeEvalA
        (Nat.le_trans evalALeCcs (Nat.le_trans ccsLeNorm normLeFinal))))
  have sumcheckLeFinal := Nat.le_trans sumcheckLeEvalK
    (Nat.le_trans evalKLeEvalA
      (Nat.le_trans evalALeCcs (Nat.le_trans ccsLeNorm normLeFinal)))
  have evalALeFinal := Nat.le_trans evalALeCcs
    (Nat.le_trans ccsLeNorm normLeFinal)
  have ccsLeFinal := Nat.le_trans ccsLeNorm normLeFinal
  have roundPointBelow (coordinate : Fin productionShape.cubeVariables) :
      (Formal.roundPoint frozen finalAt coordinate).VarsBelow finalAt := by
    have belowInitial :=
      Leaves.RoundTranscript.challenge_varsBelow_initialClaim interface
        parentOffset env transcriptAssumptions.roundTranscript coordinate
    rw [roundPoint_offset_eq frozen finalAt initialAt coordinate]
    exact KExpr.varsBelow_mono _ belowInitial initialLeFinal
  have alphaBelow (coordinate : Fin productionShape.cubeVariables) :
      (Formal.challengeAlpha frozen finalAt coordinate).VarsBelow finalAt := by
    have challengeAssumption : ChallengeDerivation.Assumptions
        (Formal.challengeInterface frozen parentOffset) challengeAt env := by
      exact transcriptAssumptions.challenge
    have below := ChallengeDerivation.alpha_varsBelow
      (Formal.challengeInterface frozen parentOffset) challengeAt
      challengeAssumption coordinate
    have belowRound :
        (Formal.challengeAlpha frozen finalAt coordinate).VarsBelow roundAt := by
      simpa [Formal.challengeAlpha, Formal.challengeStart, frozen, challengeAt,
        roundAt, Formal.atOffset, Formal.roundTranscriptOffset_eq,
        Formal.challengeOffset_eq] using below
    have roundLeInitial :=
      Formal.roundTranscriptOffset_le_initialClaimOffset interface parentOffset
    exact KExpr.varsBelow_mono _ belowRound
      (Nat.le_trans roundLeInitial initialLeFinal)
  have evalKBelow :
      (Formal.evalKOutput frozen finalAt).VarsBelow finalAt := by
    rw [evalKOutput_offset_eq frozen finalAt evalAAt]
    exact KExpr.varsBelow_mono _
      (Leaves.EvalKTerminal.output_varsBelow_evalA interface parentOffset env
        evalKAssumption) evalALeFinal
  have evalABelow :
      (Formal.evalAOutput frozen finalAt).VarsBelow finalAt := by
    rw [evalAOutput_offset_eq frozen finalAt ccsAt]
    exact KExpr.varsBelow_mono _
      (Leaves.EvalATerminal.output_varsBelow_ccs interface parentOffset env
        evalAAssumption) ccsLeFinal
  have ccsBelow :
      (Formal.ccsOutput relation frozen finalAt).VarsBelow finalAt := by
    rw [ccsOutput_offset_eq relation frozen finalAt normAt]
    exact KExpr.varsBelow_mono _
      (Leaves.CcsTerminal.output_varsBelow_norm relation interface parentOffset)
      normLeFinal
  have normBelow :
      (Formal.normOutput relation frozen finalAt).VarsBelow finalAt := by
    exact Leaves.NormTerminal.output_varsBelow_finalIdentity relation interface
      parentOffset env normAssumption
  have terminalBelow :
      (Formal.sumcheckOutput frozen finalAt).VarsBelow finalAt := by
    exact Leaves.SumcheckChain.output_varsBelow_finalIdentity relation interface
      parentOffset env sumcheckAssumption
  change FinalIdentity.Assumptions
    (Formal.finalIdentityInterface relation frozen) finalAt env
  refine {
    point := ?_
    gammaBelow := ?_
    eval_KBelow := evalKBelow
    eval_ABelow := evalABelow
    ccsBelow := ccsBelow
    normBelow := normBelow
    terminalBelow := terminalBelow }
  · intro coordinate
    change (Formal.roundPoint frozen finalAt coordinate).VarsBelow finalAt ∧
      (Formal.challengeAlpha frozen finalAt coordinate).VarsBelow finalAt
    exact ⟨roundPointBelow coordinate, alphaBelow coordinate⟩
  · change (Formal.challengeGamma frozen finalAt).VarsBelow finalAt
    rw [challengeGamma_offset_eq frozen finalAt initialAt]
    exact KExpr.varsBelow_mono _ initialAssumption.1 initialLeFinal

/-- Every child from the initial claim through the final identity allocates a
nonnegative interval before output binding begins. -/
theorem initialClaimOffset_le_outputBindingOffset
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth
      (ProductionKey.degreeBound relation) publicFits)
    (parentOffset : Nat) :
    Formal.initialClaimOffset interface parentOffset ≤
      Formal.outputBindingOffset relation interface parentOffset := by
  let initialAt := Formal.initialClaimOffset interface parentOffset
  let sumcheckAt := Formal.sumcheckOffset interface parentOffset
  let evalKAt := Formal.evalKOffset interface parentOffset
  let evalAAt := Formal.evalAOffset interface parentOffset
  let ccsAt := Formal.ccsOffset interface parentOffset
  let normAt := Formal.normOffset relation interface parentOffset
  let finalAt := Formal.finalIdentityOffset relation interface parentOffset
  let outputAt := Formal.outputBindingOffset relation interface parentOffset
  have initialLeSumcheck : initialAt ≤ sumcheckAt := by
    dsimp [sumcheckAt]
    unfold Formal.sumcheckOffset Formal.nextOffset
    omega
  have sumcheckLeEvalK : sumcheckAt ≤ evalKAt := by
    dsimp [evalKAt]
    unfold Formal.evalKOffset Formal.nextOffset
    omega
  have evalKLeEvalA : evalKAt ≤ evalAAt := by
    dsimp [evalAAt]
    unfold Formal.evalAOffset Formal.nextOffset
    omega
  have evalALeCcs : evalAAt ≤ ccsAt := by
    dsimp [ccsAt]
    unfold Formal.ccsOffset Formal.nextOffset
    omega
  have ccsLeNorm : ccsAt ≤ normAt := by
    dsimp [normAt]
    unfold Formal.normOffset Formal.nextOffset
    omega
  have normLeFinal : normAt ≤ finalAt := by
    dsimp [finalAt]
    unfold Formal.finalIdentityOffset Formal.nextOffset
    omega
  have finalLeOutput : finalAt ≤ outputAt := by
    dsimp [outputAt]
    unfold Formal.outputBindingOffset Formal.nextOffset
    omega
  exact Nat.le_trans initialLeSumcheck
    (Nat.le_trans sumcheckLeEvalK
      (Nat.le_trans evalKLeEvalA
        (Nat.le_trans evalALeCcs
          (Nat.le_trans ccsLeNorm
            (Nat.le_trans normLeFinal finalLeOutput)))))

/-- Caller-owned inputs precede the complete PiCCS child interval. -/
theorem parentOffset_le_outputBindingOffset
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth
      (ProductionKey.degreeBound relation) publicFits)
    (parentOffset : Nat) :
    parentOffset ≤ Formal.outputBindingOffset relation interface parentOffset := by
  have parentLeRound : parentOffset ≤
      Formal.roundTranscriptOffset interface parentOffset := by
    rw [Formal.roundTranscriptOffset_eq, Formal.challengeOffset_eq]
    omega
  have parentLeInitial := Nat.le_trans parentLeRound
    (Formal.roundTranscriptOffset_le_initialClaimOffset interface parentOffset)
  exact Nat.le_trans parentLeInitial
    (initialClaimOffset_le_outputBindingOffset relation interface parentOffset)

/-- Exact parent wiring packet for the output-binding leaf. -/
theorem outputBindingInputsBelow
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth
      (ProductionKey.degreeBound relation) publicFits)
    (parentOffset : Nat)
    (external : ProductionInputs.ExternalInputsLinear interface parentOffset)
    (env : Env) :
    Leaves.OutputBinding.InputsBelow
      (Formal.outputBindingInterface (Formal.atOffset interface parentOffset))
      (Formal.outputBindingOffset relation interface parentOffset) := by
  let frozen := Formal.atOffset interface parentOffset
  let outputAt := Formal.outputBindingOffset relation interface parentOffset
  have transcriptAssumptions :=
    transcript relation interface parentOffset external env
  have stateBelow :=
    Leaves.RoundTranscript.finalState_varsBelow_of_initialClaim_le interface
      parentOffset env transcriptAssumptions.roundTranscript outputAt
      (initialClaimOffset_le_outputBindingOffset relation interface parentOffset)
  have parentLeOutput :=
    parentOffset_le_outputBindingOffset relation interface parentOffset
  refine {
    initialState := stateBelow
    padCoordinate := ?_
    matrixCoordinate := ?_ }
  · intro source coefficient
    have widened := KExpr.varsBelow_mono _
      (external.below.outputEval_K source coefficient) parentLeOutput
    change ((interface.output parentOffset).padCoordinate source coefficient
      ).VarsBelow outputAt
    exact widened
  · intro source matrix coefficient
    have widened := KExpr.varsBelow_mono _
      (external.below.outputEval_A source matrix coefficient) parentLeOutput
    change ((interface.output parentOffset).matrixCoordinate source matrix
      coefficient).VarsBelow outputAt
    exact widened

/-- The round transcript's owned final state and the caller-owned separate
`Eval_K`/`Eval_A` output families discharge the output-binding assumption. -/
theorem outputBinding
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth
      (ProductionKey.degreeBound relation) publicFits)
    (parentOffset : Nat)
    (external : ProductionInputs.ExternalInputsLinear interface parentOffset)
    (env : Env) :
    (Formal.outputBindingCircuit (Formal.atOffset interface parentOffset)
      ).assumptions
      (Formal.outputBindingOffset relation interface parentOffset) env := by
  let frozen := Formal.atOffset interface parentOffset
  let outputAt := Formal.outputBindingOffset relation interface parentOffset
  change OutputBinding.Assumptions (Formal.outputBindingInterface frozen)
    outputAt env
  exact Leaves.OutputBinding.assumptions_of_inputsBelow
    (Formal.outputBindingInterface frozen) outputAt
      (outputBindingInputsBelow relation interface parentOffset external env) env

/-- The one production external-input packet discharges every causal premise
of the complete twelve-child PiCCS `FormalCircuit`. -/
theorem production
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth
      (ProductionKey.degreeBound relation) publicFits)
    (parentOffset : Nat)
    (external : ProductionInputs.ExternalInputsLinear interface parentOffset)
    (env : Env) :
    Formal.Assumptions relation interface parentOffset env := by
  have transcriptAssumptions :=
    transcript relation interface parentOffset external env
  exact {
    external := external.below
    statementBinding := transcriptAssumptions.statementBinding
    statementAbsorption := transcriptAssumptions.statementAbsorption
    challenge := transcriptAssumptions.challenge
    roundTranscript := transcriptAssumptions.roundTranscript
    initialClaim := initialClaim relation interface parentOffset external env
    sumcheck := sumcheck relation interface parentOffset external env
    eval_K := evalK relation interface parentOffset external env
    eval_A := evalA relation interface parentOffset external env
    ccs := ccs relation interface parentOffset external env
    norm := norm relation interface parentOffset external env
    finalIdentity := finalIdentity relation interface parentOffset external env
    outputBinding := outputBinding relation interface parentOffset external env }

end NightstreamFPrime.Layout.PiCCS.v1_1.Assumptions
