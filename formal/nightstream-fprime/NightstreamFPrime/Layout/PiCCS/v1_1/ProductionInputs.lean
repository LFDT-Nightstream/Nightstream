import NightstreamFPrime.Layout.PiCCS.v1_1.Composition

/-!
Paper authority: SuperNeo v1_1, section 7.3, PiCCS Steps 1--5.
Obligation: Derive every physical child-input shape from the caller-owned
production wires and the proved outputs of preceding children.

This file is the production `InputShapes` assembler. It does not supply
protocol values, challenges, or alternate constraints.
-/

namespace NightstreamFPrime.Layout.PiCCS.v1_1.ProductionInputs

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Circuit.Quadratic
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Lifecycle.PiCCS.v1_1
open NightstreamFPrime.Layout.Polynomial.Horner
open NightstreamFPrime.Layout.Poseidon2
open NightstreamFPrime.Layout.Poseidon2.Duplex
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.UnifiedSources

variable {logicalWidth : Nat}
  {publicFits : ringDegree * publicRingColumns ≤
    Phi81CarrierLayout.carrierWidth logicalWidth}

/-- The sole shape contract for caller-owned PiCCS values. Derived transcript
states, challenges, and arithmetic outputs are intentionally absent. -/
structure ExternalInputsLinear
    (interface : Formal.Interface logicalWidth 9 publicFits)
    (parentOffset : Nat) : Prop where
  below : Formal.ExternalInputsBelow interface parentOffset
  priorState : ∀ index,
    R1CS.IsAffine (interface.priorState parentOffset index)
  outputState : ∀ index,
    R1CS.IsAffine (interface.outputState parentOffset index)
  expectedContext : ∀ lane,
    R1CS.IsAffine (interface.expectedContext parentOffset lane)
  runningPoint : ∀ coordinate,
    KExprLinear ((interface.running parentOffset).point coordinate)
  runningCommitment : ∀ source row coefficient,
    R1CS.IsAffine
      ((interface.running parentOffset).commitment source row coefficient)
  runningPublicInput : ∀ source column,
    R1CS.IsAffine
      ((interface.running parentOffset).publicInput source column)
  runningEval_K : ∀ source coefficient,
    KExprLinear
      (((interface.running parentOffset).evaluation source).eval_K coefficient)
  runningEval_A : ∀ source matrix coefficient,
    KExprLinear
      (((interface.running parentOffset).evaluation source).eval_A matrix
        coefficient)
  freshCommitment : ∀ source row coefficient,
    R1CS.IsAffine
      ((interface.fresh parentOffset).commitment source row coefficient)
  freshPublicInput : ∀ source column,
    R1CS.IsAffine
      ((interface.fresh parentOffset).publicInput source column)
  roundCoefficient : ∀ roundIndex coefficient,
    KExprLinear
      ((interface.round parentOffset roundIndex).coefficient coefficient)
  outputEval_K : ∀ source coefficient,
    KExprLinear
      ((interface.output parentOffset).padCoordinate source coefficient)
  outputEval_A : ∀ source matrix coefficient,
    KExprLinear
      ((interface.output parentOffset).matrixCoordinate source matrix
        coefficient)

/-- Every child shape follows from the external production wires and the
opaque output-shape theorem of the child immediately before it. -/
theorem inputShapes
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth 9 publicFits)
    (parentOffset : Nat)
    (external : ExternalInputsLinear interface parentOffset) :
    InputShapes relation interface parentOffset := by
  let frozen := Formal.atOffset interface parentOffset
  let challengeInterface := Formal.challengeInterface frozen parentOffset
  let challengeOffset := Formal.challengeStart frozen
  let roundInterface := Formal.roundTranscriptInterface frozen
  let roundOffset := Formal.roundTranscriptStart frozen

  have statementBindingShape (childOffset : Nat) :
      Leaves.StatementBinding.InputsAffine frozen childOffset := by
    constructor
    · intro index
      simpa [frozen, Formal.atOffset] using external.priorState index
    · intro index
      simpa [frozen, Formal.atOffset] using external.outputState index
    · intro lane
      simpa [frozen, Formal.atOffset] using external.expectedContext lane

  have statementFresh : StateFresh
      (Formal.statementFinalState interface parentOffset) := by
    unfold Formal.statementFinalState
    exact Leaves.StatementAbsorption.finalState_fresh
      (Formal.statementAbsorptionInterface frozen) parentOffset

  have challengeInitialFresh (childOffset : Nat) :
      StateFresh (challengeInterface.initialState childOffset) := by
    simpa [challengeInterface, Formal.challengeInterface, frozen] using
      statementFresh

  have alphaLinear (coordinate : Fin productionShape.cubeVariables) :
      KExprLinear
        (NightstreamFPrime.Lifecycle.PiCCS.v1_1.ChallengeDerivation.alpha
          challengeInterface challengeOffset coordinate) :=
    Leaves.ChallengeDerivation.alpha_linear challengeInterface challengeOffset
      (challengeInitialFresh challengeOffset) coordinate

  have gammaLinear :
      KExprLinear
        (NightstreamFPrime.Lifecycle.PiCCS.v1_1.ChallengeDerivation.gamma
          challengeInterface challengeOffset) :=
    Leaves.ChallengeDerivation.gamma_linear challengeInterface challengeOffset
      (challengeInitialFresh challengeOffset)

  have challengeFinalFresh : StateFresh
      (NightstreamFPrime.Lifecycle.PiCCS.v1_1.ChallengeDerivation.finalState
        challengeInterface challengeOffset) :=
    Leaves.ChallengeDerivation.finalState_fresh challengeInterface
      challengeOffset (challengeInitialFresh challengeOffset)

  have roundInitialFresh (childOffset : Nat) :
      StateFresh (roundInterface.initialState childOffset) := by
    simpa [roundInterface, Formal.roundTranscriptInterface,
      Formal.challengeFinalState, challengeInterface, challengeOffset,
      frozen] using challengeFinalFresh

  have roundChallengeLinear
      (roundIndex : Fin productionShape.cubeVariables) :
      KExprLinear
        (NightstreamFPrime.Lifecycle.PiCCS.v1_1.RoundTranscript.challenge
          roundInterface roundOffset roundIndex) :=
    Leaves.RoundTranscript.challenge_linear roundInterface roundOffset
      (roundInitialFresh roundOffset) roundIndex

  have roundFinalFresh : StateFresh
      (NightstreamFPrime.Lifecycle.PiCCS.v1_1.RoundTranscript.finalState
        roundInterface roundOffset) :=
    Leaves.RoundTranscript.finalState_fresh roundInterface roundOffset
      (roundInitialFresh roundOffset)

  have statementShape (childOffset : Nat) :
      Leaves.StatementAbsorption.InputsAffine
        (Formal.statementAbsorptionInterface frozen) childOffset := by
    refine ⟨?_, ?_⟩
    · intro source row coefficient
      change R1CS.IsAffine
        ((interface.fresh parentOffset).commitment source row coefficient)
      exact external.freshCommitment source row coefficient
    · intro source column
      change R1CS.IsAffine
        ((interface.fresh parentOffset).publicInput source column)
      exact external.freshPublicInput source column

  have challengeShape (childOffset : Nat) :
      Leaves.ChallengeDerivation.InputsAffine challengeInterface childOffset :=
    ⟨(challengeInitialFresh childOffset).affine⟩

  have roundShape (childOffset : Nat) :
      Leaves.RoundTranscript.InputsAffine roundInterface childOffset := by
    refine ⟨(roundInitialFresh childOffset).affine, ?_⟩
    intro roundIndex coefficient
    change NightstreamFPrime.Layout.Poseidon2.Duplex.KExprAffine
      ((interface.round parentOffset roundIndex).coefficient coefficient)
    exact (external.roundCoefficient roundIndex coefficient).isAffine

  have gammaAt (childOffset : Nat) :
      KExprLinear (Formal.challengeGamma frozen childOffset) := by
    simpa [Formal.challengeGamma, challengeInterface, challengeOffset,
      frozen] using gammaLinear

  have alphaAt (childOffset : Nat)
      (coordinate : Fin productionShape.cubeVariables) :
      KExprLinear (Formal.challengeAlpha frozen childOffset coordinate) := by
    simpa [Formal.challengeAlpha, challengeInterface, challengeOffset,
      frozen] using alphaLinear coordinate

  have roundChallengeAt (childOffset : Nat)
      (roundIndex : Fin productionShape.cubeVariables) :
      KExprLinear
        (Formal.roundTranscriptRound frozen childOffset roundIndex).challenge := by
    simpa [Formal.roundTranscriptRound, roundInterface, roundOffset, frozen]
      using roundChallengeLinear roundIndex

  have roundPointAt (childOffset : Nat)
      (coordinate : Fin productionShape.cubeVariables) :
      KExprLinear (Formal.roundPoint frozen childOffset coordinate) := by
    simpa [Formal.roundPoint, roundInterface, roundOffset, frozen] using
      roundChallengeLinear coordinate

  have initialShape (childOffset : Nat) :
      Leaves.InitialClaim.InputsLinear
        (Formal.initialClaimInterface frozen) childOffset := by
    refine ⟨gammaAt childOffset, ?_, ?_⟩
    · intro coordinate
      change KExprLinear
        (((interface.running parentOffset).evaluation coordinate.running).eval_K
          coordinate.coefficient)
      exact external.runningEval_K coordinate.running coordinate.coefficient
    · intro coordinate
      change KExprLinear
        (((interface.running parentOffset).evaluation coordinate.running).eval_A
          coordinate.matrix coordinate.coefficient)
      exact external.runningEval_A coordinate.running coordinate.matrix
        coordinate.coefficient

  have initialOutputLinear :
      KExprLinear (Formal.initialClaimOutput frozen parentOffset) := by
    unfold Formal.initialClaimOutput
    exact Leaves.InitialClaim.output_linear
      (Formal.initialClaimInterface frozen) (Formal.initialClaimStart frozen)
      (initialShape (Formal.initialClaimStart frozen))

  have sumcheckShape (childOffset : Nat) :
      Leaves.SumcheckChain.InputsLinear
        (Formal.sumcheckInterface frozen) childOffset := by
    refine ⟨?_, ?_, ?_⟩
    · simpa [Formal.sumcheckInterface] using initialOutputLinear
    · intro roundIndex coefficient
      change KExprLinear
        ((interface.round parentOffset roundIndex).coefficient coefficient)
      exact external.roundCoefficient roundIndex coefficient
    · intro roundIndex
      exact roundChallengeAt childOffset roundIndex

  have evalKShape (childOffset : Nat) :
      Leaves.EvalKTerminal.InputsLinear
        (Formal.evalKInterface frozen) childOffset := by
    refine ⟨roundPointAt childOffset, ?_, gammaAt childOffset, ?_⟩
    · intro coordinate
      change KExprLinear ((interface.running parentOffset).point coordinate)
      exact external.runningPoint coordinate
    · intro coordinate
      change KExprLinear
        ((interface.output parentOffset).padCoordinate
          (runningSourceIndex coordinate.running) coordinate.coefficient)
      exact external.outputEval_K _ _

  have evalAShape (childOffset : Nat) :
      Leaves.EvalATerminal.InputsLinear
        (Formal.evalAInterface frozen) childOffset := by
    refine ⟨roundPointAt childOffset, ?_, gammaAt childOffset, ?_⟩
    · intro coordinate
      change KExprLinear ((interface.running parentOffset).point coordinate)
      exact external.runningPoint coordinate
    · intro coordinate
      change KExprLinear
        ((interface.output parentOffset).matrixCoordinate
          (runningSourceIndex coordinate.running) coordinate.matrix
            coordinate.coefficient)
      exact external.outputEval_A _ _ _

  have ccsShape (childOffset : Nat) :
      Leaves.CcsTerminal.InputsLinear
        (Formal.ccsInterface relation frozen) childOffset := by
    constructor
    intro matrix
    change KExprLinear
      ((interface.output parentOffset).matrixCoordinate
        (freshSourceIndex Formal.freshIndex) matrix
          (Formal.constantCoefficient relation))
    exact external.outputEval_A _ _ _

  have normShape (childOffset : Nat) :
      Leaves.NormTerminal.InputsLinear
        (Formal.normInterface relation frozen) childOffset := by
    refine ⟨gammaAt childOffset, ?_⟩
    intro source
    change KExprLinear
      ((interface.output parentOffset).padCoordinate source
        (Formal.constantCoefficient relation))
    exact external.outputEval_K _ _

  have finalIdentityShape (childOffset : Nat) :
      Leaves.FinalIdentity.InputsLinear
        (Formal.finalIdentityInterface relation frozen) childOffset := by
    exact ⟨roundPointAt childOffset, alphaAt childOffset,
      gammaAt childOffset⟩

  have outputBindingShape (childOffset : Nat) :
      Leaves.OutputBinding.InputsAffine
        (Formal.outputBindingInterface frozen) childOffset := by
    refine ⟨?_, ?_, ?_⟩
    · simpa only [Formal.outputBindingInterface,
        Formal.roundTranscriptFinalState] using roundFinalFresh.affine
    · intro source coefficient
      change NightstreamFPrime.Layout.Poseidon2.Duplex.KExprAffine
        ((interface.output parentOffset).padCoordinate source coefficient)
      exact (external.outputEval_K source coefficient).isAffine
    · intro source matrix coefficient
      change NightstreamFPrime.Layout.Poseidon2.Duplex.KExprAffine
        ((interface.output parentOffset).matrixCoordinate source matrix
          coefficient)
      exact (external.outputEval_A source matrix coefficient).isAffine

  exact {
    statementBinding := statementBindingShape
    statementAbsorption := statementShape
    challengeDerivation := challengeShape
    roundTranscript := roundShape
    initialClaim := initialShape
    sumcheck := sumcheckShape
    eval_K := evalKShape
    eval_A := evalAShape
    ccs := ccsShape
    norm := normShape
    finalIdentity := finalIdentityShape
    outputBinding := outputBindingShape
  }

theorem physicalFreshColumnCount_eq
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth 9 publicFits)
    (parentOffset : Nat)
    (external : ExternalInputsLinear interface parentOffset) :
    physicalFreshColumnCount relation interface parentOffset = 685348 :=
  physicalFreshColumnCount_eq_production relation interface parentOffset
    (inputShapes relation interface parentOffset external)

theorem physicalRowCount_eq
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth 9 publicFits)
    (parentOffset : Nat)
    (external : ExternalInputsLinear interface parentOffset) :
    physicalRowCount relation interface parentOffset = 5181690 :=
  physicalRowCount_eq_production relation interface parentOffset
    (inputShapes relation interface parentOffset external)

theorem physicalColumnCount_eq
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth 9 publicFits)
    (parentOffset : Nat)
    (external : ExternalInputsLinear interface parentOffset) :
    physicalColumnCount relation interface parentOffset =
      parentOffset + 5181478 :=
  physicalColumnCount_eq_production relation interface parentOffset
    (inputShapes relation interface parentOffset external)

theorem jointDomain_eq
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth 9 publicFits)
    (external : ExternalInputsLinear interface 0) :
    jointDomain relation interface = 5181690 :=
  jointDomain_eq_production relation interface
    (inputShapes relation interface 0 external)

theorem jointDomain_le_twoPow25
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (interface : Formal.Interface logicalWidth 9 publicFits)
    (external : ExternalInputsLinear interface 0) :
    jointDomain relation interface ≤ 2 ^ 25 :=
  NightstreamFPrime.Layout.PiCCS.v1_1.jointDomain_le_twoPow25 relation interface
    (inputShapes relation interface 0 external)

end NightstreamFPrime.Layout.PiCCS.v1_1.ProductionInputs
