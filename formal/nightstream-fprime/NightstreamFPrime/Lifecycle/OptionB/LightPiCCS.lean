import NightstreamFPrime.Lifecycle.OptionB.Keys
import NightstreamFPrime.Lifecycle.PiCCS.v1_2.InitialClaim
import NightstreamFPrime.Lifecycle.PiCCS.v1_2.OutputBinding
import NightstreamFPrime.Lifecycle.PiCCS.v1_2.EvalKTerminal
import NightstreamFPrime.Lifecycle.PiCCS.v1_2.EvalATerminal
import NightstreamFPrime.Lifecycle.PiCCS.v1_2.NormTerminal
import NightstreamFPrime.Lifecycle.PiCCS.v1_2.CcsTerminal
import NightstreamFPrime.Lifecycle.PiCCS.v1_2.FinalIdentity
import NightstreamFPrime.Lifecycle.PiCCS.v1_2.StatementAbsorption
import NightstreamFPrime.Lifecycle.PiCCS.v1_2.ChallengeDerivation
import NightstreamFPrime.Lifecycle.PiCCS.v1_2.RoundTranscript
import NightstreamFPrime.Lifecycle.PiCCS.v1_2.SumcheckChain

/-! The light Π_CCS leaves (CW-2.2), one per production leaf made generic in the shape. Each
bridge gives a value of the production light key's Π_CCS statement from the leaf's rows. -/

namespace NightstreamFPrime.Lifecycle.OptionB.LightPiCCS

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Circuit.Quadratic
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ConcreteCarrier
open NightstreamFPrime.Gadgets.Poseidon2
open NightstreamFPrime.Gadgets.Multilinear
open NightstreamFPrime.Lifecycle.PiCCS.v1_2

variable {logicalWidth : Nat}
  {publicFits : ringDegree * PaperAlgebra.publicRingColumns ≤
    Phi81CarrierLayout.carrierWidth logicalWidth}

/-- The light Π_CCS claimed sum `T` (SuperNeo v1.2 Section 7.3, Step 2) over the single running
claim: 54 + 216 = 270 claimed coefficients, against 4,320 today
(`InitialClaim.coefficientExprs_length`). -/
theorem initial_eq (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : PaperAlgebra.AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
    (running : Spec.Folding.Nifs.PaperNonInteractive.Running K PaperAlgebra.Commitment
      (PaperAlgebra.PublicInput (logicalWidth := logicalWidth) (publicFits := publicFits))
      LightShape)
    (fresh : Spec.Folding.Nifs.PaperNonInteractive.Fresh PaperAlgebra.Commitment
      (PaperAlgebra.PublicInput (logicalWidth := logicalWidth) (publicFits := publicFits))
      LightShape)
    (interface : InitialClaim.ShapeInterface LightShape) (offset : Nat) (env : Env)
    (gamma : K) (gamma_eq : (interface.gamma offset).eval env = gamma)
    (eval_K_eq : ∀ coordinate, (interface.eval_K offset coordinate).eval env =
      (((keys relation ajtai).light.statement running fresh).verifierInput
        (keys relation ajtai).light.lift).claimedPadCoefficient coordinate)
    (eval_A_eq : ∀ coordinate, (interface.eval_A offset coordinate).eval env =
      (((keys relation ajtai).light.statement running fresh).verifierInput
        (keys relation ajtai).light.lift).claimedMatrixCoefficient coordinate)
    (specification : InitialClaim.SpecHolds interface offset env) :
    (InitialClaim.output interface offset).eval env =
      (((keys relation ajtai).light.statement running fresh).verifierInput
        (keys relation ajtai).light.lift).initial extensionOps gamma :=
  InitialClaim.spec_implies_initial interface offset env _ gamma gamma_eq eval_K_eq eval_A_eq
    specification

theorem light_coefficientCount (interface : InitialClaim.ShapeInterface LightShape)
    (offset : Nat) : (InitialClaim.coefficientExprs interface offset).length = 270 := by
  simp only [InitialClaim.coefficientExprs, List.length_append, List.length_map,
    canonicalPadCoordinates_length, canonicalMatrixCoordinates_length]
  rfl

/-- The light Π_CCS output binding (SuperNeo v1.2 Section 7.3, Step 5): the rows absorb the
outputs of the two sources, which is the light key's `absorbPiCcsOutput`: 2 × 540 = 1,080 words,
against 17 × 540 = 9,180 today. -/
theorem outgoing_eq (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : PaperAlgebra.AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
    (interface : OutputBinding.ShapeInterface LightShape) (offset : Nat) (env : Env)
    (state : Transcript.State) (output : FullOutputCoordinates.FullOutput K LightShape)
    (initialEq : List.ofFn (Layer.evalState env (interface.initialState offset)) = state)
    (padEq : ∀ source coefficient,
      ((interface.output offset).padCoordinate source coefficient).eval env =
        output.padCoordinate source coefficient)
    (matrixEq : ∀ source matrix coefficient,
      ((interface.output offset).matrixCoordinate source matrix coefficient).eval env =
        output.matrixCoordinate source matrix coefficient)
    (specification : OutputBinding.SpecHolds interface offset env) :
    List.ofFn (Layer.evalState env (OutputBinding.finalState interface offset)) =
      (keys relation ajtai).light.absorbPiCcsOutput state output :=
  OutputBinding.spec_implies_absorb interface offset env state output initialEq padEq matrixEq
    specification

section Terminals

variable (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
  (ajtai : PaperAlgebra.AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
  (running : Spec.Folding.Nifs.PaperNonInteractive.Running K PaperAlgebra.Commitment
    (PaperAlgebra.PublicInput (logicalWidth := logicalWidth) (publicFits := publicFits))
    LightShape)
  (fresh : Spec.Folding.Nifs.PaperNonInteractive.Fresh PaperAlgebra.Commitment
    (PaperAlgebra.PublicInput (logicalWidth := logicalWidth) (publicFits := publicFits))
    LightShape)

/-- The light statement's verifier input (SuperNeo v1.2 Section 7.3). -/
noncomputable abbrev lightInput :=
  ((keys relation ajtai).light.statement running fresh).verifierInput
    (keys relation ajtai).light.lift

/-- `E_K` of the light Π_CCS from the generic `Eval_K` terminal: 54 output coefficients. -/
theorem padAtMessage_eq (interface : EvalKTerminal.ShapeInterface LightShape) (offset : Nat)
    (env : Env) (gamma : K) (roundPoint : CubePoint K LightShape.cubeVariables)
    (message : ProtocolPolynomial.OutputMessage K LightShape)
    (roundPointEq : PointEquality.Owned.evalLeftPoint
      (EvalKTerminal.pointInterfaceAt interface offset) offset env = roundPoint)
    (priorPointEq : PointEquality.Owned.evalRightPoint
      (EvalKTerminal.pointInterfaceAt interface offset) offset env =
        (lightInput relation ajtai running fresh).priorPoint)
    (gammaEq : (interface.gamma offset).eval env = gamma)
    (outputEvalEq : ∀ coordinate,
      (interface.outputEval_K offset coordinate).eval env = message.padImage coordinate)
    (specification : EvalKTerminal.SpecHolds interface offset env) :
    (EvalKTerminal.output interface offset).eval env =
      ProtocolPolynomial.padAtMessage extensionOps (lightInput relation ajtai running fresh)
        gamma roundPoint message :=
  EvalKTerminal.spec_implies_padAtMessage interface offset env _ gamma roundPoint message
    roundPointEq priorPointEq gammaEq outputEvalEq specification

/-- `E_A` of the light Π_CCS from the generic `Eval_A` terminal: 216 output coefficients. -/
theorem matrixAtMessage_eq (interface : EvalATerminal.ShapeInterface LightShape) (offset : Nat)
    (env : Env) (gamma : K) (roundPoint : CubePoint K LightShape.cubeVariables)
    (message : ProtocolPolynomial.OutputMessage K LightShape)
    (roundPointEq : PointEquality.Owned.evalLeftPoint
      (EvalATerminal.pointInterfaceAt interface offset) offset env = roundPoint)
    (priorPointEq : PointEquality.Owned.evalRightPoint
      (EvalATerminal.pointInterfaceAt interface offset) offset env =
        (lightInput relation ajtai running fresh).priorPoint)
    (gammaEq : (interface.gamma offset).eval env = gamma)
    (outputEvalEq : ∀ coordinate,
      (interface.outputEval_A offset coordinate).eval env = message.matrixImage coordinate)
    (specification : EvalATerminal.SpecHolds interface offset env) :
    (EvalATerminal.output interface offset).eval env =
      ProtocolPolynomial.matrixAtMessage extensionOps (lightInput relation ajtai running fresh)
        gamma roundPoint message :=
  EvalATerminal.spec_implies_matrixAtMessage interface offset env _ gamma roundPoint message
    roundPointEq priorPointEq gammaEq outputEvalEq specification

end Terminals

/-- The norm terms of the light Π_CCS for any output message. The light check reads the message
through `Light.PiCCS.maskMessage`, which sets the running source's norm input to zero, so a light
circuit feeds the constant 0 as the running source's assignment. -/
theorem normAtMessage_eq (interface : NormTerminal.ShapeInterface LightShape) (offset : Nat)
    (env : Env) (gamma : K) (message : ProtocolPolynomial.OutputMessage K LightShape)
    (gammaEq : (interface.gamma offset).eval env = gamma)
    (sourceAssignmentEq : ∀ source,
      (interface.sourceAssignment offset source).eval env = message.sourceAssignment source)
    (specification : NormTerminal.SpecHolds interface offset env) :
    (NormTerminal.output interface offset).eval env =
      ProtocolPolynomial.normAtMessage extensionOps gamma message :=
  NormTerminal.spec_implies_normAtMessage interface offset env gamma message gammaEq
    sourceAssignmentEq specification

/-- The light Π_CCS's one fresh source. -/
def lightFreshIndex : Fin LightShape.freshCount := ⟨0, by decide⟩

/-- The light Π_CCS has one fresh source, so its CCS term is the constraint polynomial at that
source's matrix evaluations. -/
theorem ccsAtMessage_eq_singleFresh (input : ProtocolPolynomial.VerifierInput K LightShape)
    (gamma : K) (message : ProtocolPolynomial.OutputMessage K LightShape) :
    ProtocolPolynomial.ccsAtMessage extensionOps input gamma message =
      CCSResidualTable.evaluatePolynomial extensionOps
        input.constraintPolynomial (message.freshMatrixImage lightFreshIndex) := by
  simp [ProtocolPolynomial.ccsAtMessage, SignedJointIdentity.sumMap,
    SignedJointIdentity.gammaTerm, TargetPolynomial.power,
    BooleanTable.finiteSum, canonicalFinIndices, LightShape,
    HyperNova.Construction2.OptionBFolds.lightShape, productionShape,
    productionProfile, Phi81MatrixSource.phi81Shape, lightFreshIndex,
    extensionLaws.one_mul, extensionLaws.add_zero]

/-- The light Π_CCS CCS term from today's CCS terminal. The light shape keeps the four matrices
and the one fresh source, so the terminal circuit is unchanged: 23 rows. -/
theorem ccsAtMessage_eq (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : PaperAlgebra.AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
    (running : Spec.Folding.Nifs.PaperNonInteractive.Running K PaperAlgebra.Commitment
      (PaperAlgebra.PublicInput (logicalWidth := logicalWidth) (publicFits := publicFits))
      LightShape)
    (fresh : Spec.Folding.Nifs.PaperNonInteractive.Fresh PaperAlgebra.Commitment
      (PaperAlgebra.PublicInput (logicalWidth := logicalWidth) (publicFits := publicFits))
      LightShape)
    (interface : CcsTerminal.Interface) (offset : Nat) (env : Env) (gamma : K)
    (message : ProtocolPolynomial.OutputMessage K LightShape)
    (freshMatrixEq : ∀ matrix,
      (interface.freshMatrix offset matrix).eval env =
        message.freshMatrixImage lightFreshIndex matrix)
    (specification : CcsTerminal.SpecHolds relation interface offset env) :
    (CcsTerminal.output relation interface offset).eval env =
      ProtocolPolynomial.ccsAtMessage extensionOps (lightInput relation ajtai running fresh)
        gamma message := by
  have pointEq : (fun matrix => (interface.freshMatrix offset matrix).eval env) =
      message.freshMatrixImage lightFreshIndex :=
    funext freshMatrixEq
  have value : (CcsTerminal.output relation interface offset).eval env =
      CCSResidualTable.evaluatePolynomial extensionOps
        (lightInput relation ajtai running fresh).constraintPolynomial
        (fun matrix => (interface.freshMatrix offset matrix).eval env) :=
    specification
  exact (value.trans (congrArg _ pointEq)).trans
    (ccsAtMessage_eq_singleFresh _ gamma message).symm

/-- The light final identity's `γ` exponents: 54 for `E_A` and 270 = 54 × 5 for the constraint
terms, against 864 and 4,320 today (`FinalIdentity.matrixExponent_eq`,
`FinalIdentity.constraintExponent_eq`). -/
theorem light_exponents :
    FinalIdentity.matrixExponentOf LightShape = 54 ∧
      FinalIdentity.constraintExponentOf LightShape = 270 :=
  ⟨rfl, rfl⟩

/-- The light Π_CCS terminal `v = Q(r')` (SuperNeo v1.2 Section 7.3, Step 4) from the generic
final identity. -/
theorem terminal_eq (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : PaperAlgebra.AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
    (running : Spec.Folding.Nifs.PaperNonInteractive.Running K PaperAlgebra.Commitment
      (PaperAlgebra.PublicInput (logicalWidth := logicalWidth) (publicFits := publicFits))
      LightShape)
    (fresh : Spec.Folding.Nifs.PaperNonInteractive.Fresh PaperAlgebra.Commitment
      (PaperAlgebra.PublicInput (logicalWidth := logicalWidth) (publicFits := publicFits))
      LightShape)
    (interface : FinalIdentity.ShapeInterface LightShape) (offset : Nat) (env : Env)
    (alpha roundPoint : CubePoint K LightShape.cubeVariables) (gamma : K)
    (message : ProtocolPolynomial.OutputMessage K LightShape)
    (roundPointEq : PointEquality.Owned.evalLeftPoint
      (FinalIdentity.pointInterfaceAt interface offset) offset env = roundPoint)
    (alphaEq : PointEquality.Owned.evalRightPoint
      (FinalIdentity.pointInterfaceAt interface offset) offset env = alpha)
    (gammaEq : (interface.gamma offset).eval env = gamma)
    (evalKEq : (interface.eval_K offset).eval env =
      ProtocolPolynomial.padAtMessage extensionOps (lightInput relation ajtai running fresh)
        gamma roundPoint message)
    (evalAEq : (interface.eval_A offset).eval env =
      ProtocolPolynomial.matrixAtMessage extensionOps (lightInput relation ajtai running fresh)
        gamma roundPoint message)
    (ccsEq : (interface.ccs offset).eval env =
      ProtocolPolynomial.ccsAtMessage extensionOps (lightInput relation ajtai running fresh)
        gamma message)
    (normEq : (interface.norm offset).eval env =
      ProtocolPolynomial.normAtMessage extensionOps gamma message)
    (specification : FinalIdentity.SpecHolds interface offset env) :
    (interface.terminal offset).eval env =
      ProtocolPolynomial.terminalFromMessage extensionOps
        (lightInput relation ajtai running fresh) alpha gamma roundPoint message :=
  FinalIdentity.spec_implies_terminal interface offset env rfl _ alpha roundPoint gamma message
    roundPointEq alphaEq gammaEq evalKEq evalAEq ccsEq normEq specification

/-- The light Π_CCS statement absorption (SuperNeo v1.2 Section 7.3): the generic absorption from
the light fold domain ends in the light key's prior transcript state, the state its Π_CCS
challenges start from. The absorbed statement is today's: the prior digest and the fresh blocks,
1,462 words. -/
theorem statementState_eq (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : PaperAlgebra.AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
    (running : Spec.Folding.Nifs.PaperNonInteractive.Running K PaperAlgebra.Commitment
      (PaperAlgebra.PublicInput (logicalWidth := logicalWidth) (publicFits := publicFits))
      LightShape)
    (fresh : Spec.Folding.Nifs.PaperNonInteractive.Fresh PaperAlgebra.Commitment
      (PaperAlgebra.PublicInput (logicalWidth := logicalWidth) (publicFits := publicFits))
      LightShape)
    (interface : StatementAbsorption.Interface logicalWidth publicFits) (offset : Nat)
    (env : Env) (final : Transcript.State)
    (freshWords : StatementAbsorption.evalFresh (interface.fresh offset) env =
      ⟨fresh.commitments, fresh.publicInputs⟩)
    (trace : Duplex.Formal.TraceHolds (StatementAbsorption.evalState env Hash.zeroE)
      ((StatementAbsorption.domainActions lightFoldDomainChunk interface offset).map
        (Duplex.Formal.Action.eval env)) final) :
    final = (keys relation ajtai).light.absorbPublicInput
      (keys relation ajtai).light.initialTranscriptState running fresh := by
  rw [StatementAbsorption.domainTrace_eq lightFoldDomainChunk interface offset env final trace,
    freshWords]
  rfl

section Transcript

/-- A light Π_CCS challenge label as the production label at the same coin position. -/
def labelToProduction :
    FiatShamir.ChallengeLabel LightShape → FiatShamir.ChallengeLabel productionShape
  | .alpha coordinate => .alpha coordinate
  | .gamma => .gamma
  | .sumcheck round => .sumcheck round

/-- The light oracle squeezes at the production coin positions (`lightCoinPosition`). -/
theorem squeeze_eq (state : Transcript.State) (label : FiatShamir.ChallengeLabel LightShape) :
    lightPiCcsOracle.transcript.squeeze state label =
      Transcript.piCcsOracle.transcript.squeeze state (labelToProduction label) := by
  cases label <;> rfl

theorem squeezeMany_eq : ∀ (state : Transcript.State)
    (labels : List (FiatShamir.ChallengeLabel LightShape)),
    FiatShamir.squeezeMany lightPiCcsOracle.transcript state labels =
      FiatShamir.squeezeMany Transcript.piCcsOracle.transcript state
        (labels.map labelToProduction)
  | _, [] => rfl
  | state, label :: labels => by
      simp only [FiatShamir.squeezeMany, List.map_cons, squeeze_eq]
      rw [squeezeMany_eq _ labels]

theorem alphaLabels_eq :
    (FiatShamir.alphaLabels LightShape).map labelToProduction =
      FiatShamir.alphaLabels productionShape := by
  simp only [FiatShamir.alphaLabels, List.map_map]
  rfl

/-- From one transcript state, the light and the production oracles derive the same `α`, `γ`
and state (SuperNeo v1.2 Section 7.3, Step 1). -/
theorem deriveFromState_eq (state : Transcript.State) :
    (Spec.Folding.PiCCS.Transcript.deriveFromState lightPiCcsOracle.transcript
        state).alpha.coordinates =
      (Spec.Folding.PiCCS.Transcript.deriveFromState Transcript.piCcsOracle.transcript
        state).alpha.coordinates ∧
    (Spec.Folding.PiCCS.Transcript.deriveFromState lightPiCcsOracle.transcript state).gamma =
      (Spec.Folding.PiCCS.Transcript.deriveFromState Transcript.piCcsOracle.transcript
        state).gamma ∧
    (Spec.Folding.PiCCS.Transcript.deriveFromState lightPiCcsOracle.transcript state).state =
      (Spec.Folding.PiCCS.Transcript.deriveFromState Transcript.piCcsOracle.transcript
        state).state := by
  simp only [Spec.Folding.PiCCS.Transcript.deriveFromState, squeezeMany_eq, alphaLabels_eq,
    squeeze_eq, labelToProduction]
  refine ⟨?_, ?_, ?_⟩ <;> trivial

/-- Round by round, the light and the production oracles absorb and squeeze alike. -/
theorem deriveRoundsFrom_eq (rounds : Fin LightShape.cubeVariables → SumCheck.Finite.Message K) :
    ∀ (state : Transcript.State) (indices : List (Fin LightShape.cubeVariables)),
    FiatShamir.deriveRoundsFrom lightPiCcsOracle.transcript rounds state indices =
      FiatShamir.deriveRoundsFrom Transcript.piCcsOracle.transcript rounds state indices
  | _, [] => rfl
  | state, round :: remaining => by
      simp only [FiatShamir.deriveRoundsFrom]
      rw [deriveRoundsFrom_eq rounds _ remaining]
      rfl

variable (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
  (ajtai : PaperAlgebra.AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
  (running : Spec.Folding.Nifs.PaperNonInteractive.Running K PaperAlgebra.Commitment
    (PaperAlgebra.PublicInput (logicalWidth := logicalWidth) (publicFits := publicFits))
    LightShape)
  (fresh : Spec.Folding.Nifs.PaperNonInteractive.Fresh PaperAlgebra.Commitment
    (PaperAlgebra.PublicInput (logicalWidth := logicalWidth) (publicFits := publicFits))
    LightShape)
  (rounds : Fin LightShape.cubeVariables →
    SumCheck.Finite.FixedPolynomial K (ProductionKey.degreeBound relation))

/-- The light key's Fiat–Shamir coins for the given round polynomials. -/
noncomputable abbrev lightCoins := (keys relation ajtai).light.coins running fresh rounds

/-- The light transcript state after `α` and `γ`, where the SumCheck rounds start. -/
noncomputable abbrev lightPreState :=
  (FiatShamir.derivePreSumcheck (keys relation ajtai).light.oracle.transcript
    ((keys relation ajtai).light.transcriptStatement running fresh)).state

/-- The challenge-derivation rows give the light key's `α` and `γ` and the state the SumCheck
rounds start from. -/
theorem challenges_eq (interface : ChallengeDerivation.Interface) (offset : Nat) (env : Env)
    (initialEq : ChallengeDerivation.evalState env (interface.initialState offset) =
      (keys relation ajtai).light.absorbPublicInput
        (keys relation ajtai).light.initialTranscriptState running fresh)
    (specification : ChallengeDerivation.SpecHolds interface offset env) :
    (ChallengeDerivation.evalAlpha interface offset env).coordinates =
        (lightCoins relation ajtai running fresh rounds).alpha.coordinates ∧
      ChallengeDerivation.evalGamma interface offset env =
        (lightCoins relation ajtai running fresh rounds).gamma ∧
      ChallengeDerivation.evalState env (ChallengeDerivation.finalState interface offset) =
        lightPreState relation ajtai running fresh := by
  obtain ⟨alphaEq, gammaEq, finalEq⟩ := specification
  rw [initialEq] at alphaEq gammaEq finalEq
  have same := deriveFromState_eq ((keys relation ajtai).light.absorbPublicInput
    (keys relation ajtai).light.initialTranscriptState running fresh)
  refine ⟨?_, ?_, ?_⟩
  · rw [alphaEq]
    exact same.1.symm
  · rw [gammaEq]
    exact same.2.1.symm
  · rw [finalEq]
    exact same.2.2.symm

/-- The round-transcript rows give the light key's SumCheck point and final state. -/
theorem roundPoint_eq (interface : RoundTranscript.Interface (ProductionKey.degreeBound relation))
    (offset : Nat) (env : Env)
    (initialEq : RoundTranscript.evalState env (interface.initialState offset) =
      lightPreState relation ajtai running fresh)
    (roundsEq : ∀ roundIndex,
      (interface.round offset roundIndex).semanticPolynomial env = rounds roundIndex)
    (specification : RoundTranscript.SpecHolds interface offset env) :
    (RoundTranscript.evalRoundPoint interface offset env).coordinates =
        (lightCoins relation ajtai running fresh rounds).roundPoint.coordinates ∧
      RoundTranscript.evalState env (RoundTranscript.finalState interface offset) =
        (lightCoins relation ajtai running fresh rounds).finalState := by
  obtain ⟨pointEq, finalEq⟩ := specification
  have semantic : RoundTranscript.semanticRounds interface offset env =
      fun roundIndex => (rounds roundIndex).toMessage :=
    funext fun roundIndex =>
      congrArg SumCheck.Finite.FixedPolynomial.toMessage (roundsEq roundIndex)
  rw [initialEq, semantic] at pointEq finalEq
  have holds := Spec.Folding.PiCCS.Transcript.derive_rounds_holds
    (keys relation ajtai).light.oracle.transcript
    ((keys relation ajtai).light.transcriptStatement running fresh)
    { rounds := fun roundIndex => (rounds roundIndex).toMessage }
  have oracleEq : (keys relation ajtai).light.oracle.transcript = lightPiCcsOracle.transcript :=
    rfl
  have pointCoins : (lightCoins relation ajtai running fresh rounds).roundPoint.coordinates =
      (FiatShamir.deriveRoundsFrom Transcript.piCcsOracle.transcript
        (fun roundIndex => (rounds roundIndex).toMessage)
        (lightPreState relation ajtai running fresh)
        (canonicalFinIndices LightShape.cubeVariables)).1 := by
    have point := holds.roundPoint_eq
    rw [oracleEq, deriveRoundsFrom_eq] at point
    exact point
  have finalCoins : (lightCoins relation ajtai running fresh rounds).finalState =
      (FiatShamir.deriveRoundsFrom Transcript.piCcsOracle.transcript
        (fun roundIndex => (rounds roundIndex).toMessage)
        (lightPreState relation ajtai running fresh)
        (canonicalFinIndices LightShape.cubeVariables)).2 := by
    have final := holds.finalState_eq
    rw [oracleEq, deriveRoundsFrom_eq] at final
    exact final
  exact ⟨pointEq.trans pointCoins.symm, finalEq.trans finalCoins.symm⟩

/-- The SumCheck chain rows give the fixed chain of the light Π_CCS from `T` to `Q(r')`. -/
theorem chain_eq (interface : SumcheckChain.Interface (ProductionKey.degreeBound relation))
    (offset : Nat) (env : Env) (message : ProtocolPolynomial.OutputMessage K LightShape)
    (initialEq : (interface.initial offset).eval env =
      (lightInput relation ajtai running fresh).initial extensionOps
        (lightCoins relation ajtai running fresh rounds).gamma)
    (roundsEq : ∀ roundIndex,
      (interface.round offset roundIndex).semanticPolynomial env = rounds roundIndex)
    (pointEq : (SumcheckChain.evalRoundPoint interface offset env).coordinates =
      (lightCoins relation ajtai running fresh rounds).roundPoint.coordinates)
    (terminalEq : (SumcheckChain.output interface offset).eval env =
      ProtocolPolynomial.terminalFromMessage extensionOps (lightInput relation ajtai running fresh)
        (lightCoins relation ajtai running fresh rounds).alpha
        (lightCoins relation ajtai running fresh rounds).gamma
        (lightCoins relation ajtai running fresh rounds).roundPoint message)
    (specification : SumcheckChain.SpecHolds interface offset env) :
    SumCheck.Finite.FixedPhase.Chain extensionOps.toOps
      ((lightInput relation ajtai running fresh).initial extensionOps
        (lightCoins relation ajtai running fresh rounds).gamma)
      (List.ofFn rounds) (lightCoins relation ajtai running fresh rounds).roundPoint.coordinates
      (ProtocolPolynomial.terminalFromMessage extensionOps (lightInput relation ajtai running fresh)
        (lightCoins relation ajtai running fresh rounds).alpha
        (lightCoins relation ajtai running fresh rounds).gamma
        (lightCoins relation ajtai running fresh rounds).roundPoint message) :=
  SumcheckChain.spec_implies_chain interface offset env _ rounds _ _ initialEq roundsEq pointEq
    terminalEq specification

end Transcript

end NightstreamFPrime.Lifecycle.OptionB.LightPiCCS
