import NightstreamFPrime.Lifecycle.OptionB.LightPiCCS

/-! The light Π_CCS phase (CW-2.5a). The leaf gadgets of `LightPiCCS`, with the wires of `Wired`,
make the production light key's Π_CCS check accept: SuperNeo v1.2 Section 7.3, Step 4, on the
output with the running source's norm input set to zero (`Light.PiCCS.maskMessage`), so the norm
terminal reads the constant 0 for the running source. The phase also gives the SumCheck point and
the transcript state after the output binding, which the light Π_RLC reads. A layout gives
`Wired` by its offsets. -/

namespace NightstreamFPrime.Lifecycle.OptionB.LightPiCCSPhase

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Circuit.Quadratic
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ConcreteCarrier
open NightstreamFPrime.Gadgets.Poseidon2
open NightstreamFPrime.Gadgets.Multilinear
open NightstreamFPrime.Lifecycle.PiCCS.v1_2
open NightstreamFPrime.Lifecycle.OptionB.LightPiCCS

/-- The leaf gadgets of the light Π_CCS and their offsets. `statementFinal` is the state after
the statement absorption. -/
structure Leaves (logicalWidth degreeBound : Nat)
    (publicFits : ringDegree * PaperAlgebra.publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) where
  statement : StatementAbsorption.Interface logicalWidth publicFits
  statementOffset : Nat
  statementFinal : Layer.EState
  challenge : ChallengeDerivation.Interface
  challengeOffset : Nat
  rounds : RoundTranscript.Interface degreeBound
  roundsOffset : Nat
  initial : InitialClaim.ShapeInterface LightShape
  initialOffset : Nat
  sumcheck : SumcheckChain.Interface degreeBound
  sumcheckOffset : Nat
  evalK : EvalKTerminal.ShapeInterface LightShape
  evalKOffset : Nat
  evalA : EvalATerminal.ShapeInterface LightShape
  evalAOffset : Nat
  ccs : CcsTerminal.Interface
  ccsOffset : Nat
  norm : NormTerminal.ShapeInterface LightShape
  normOffset : Nat
  finalIdentity : FinalIdentity.ShapeInterface LightShape
  finalIdentityOffset : Nat
  output : OutputBinding.ShapeInterface LightShape
  outputOffset : Nat

variable {logicalWidth : Nat}
  {publicFits : ringDegree * PaperAlgebra.publicRingColumns ≤
    Phi81CarrierLayout.carrierWidth logicalWidth}
  (relation : ProductionKey.LogicalRelation logicalWidth publicFits)

/-- Each leaf's rows hold. The statement absorption is the trace from the light fold domain. -/
structure SpecHolds (leaves : Leaves logicalWidth (ProductionKey.degreeBound relation) publicFits)
    (env : Env) : Prop where
  statement : Duplex.Formal.TraceHolds (StatementAbsorption.evalState env Hash.zeroE)
    ((StatementAbsorption.domainActions lightFoldDomainChunk leaves.statement
      leaves.statementOffset).map (Duplex.Formal.Action.eval env))
    (StatementAbsorption.evalState env leaves.statementFinal)
  challenge : ChallengeDerivation.SpecHolds leaves.challenge leaves.challengeOffset env
  rounds : RoundTranscript.SpecHolds leaves.rounds leaves.roundsOffset env
  initial : InitialClaim.SpecHolds leaves.initial leaves.initialOffset env
  sumcheck : SumcheckChain.SpecHolds leaves.sumcheck leaves.sumcheckOffset env
  evalK : EvalKTerminal.SpecHolds leaves.evalK leaves.evalKOffset env
  evalA : EvalATerminal.SpecHolds leaves.evalA leaves.evalAOffset env
  ccs : CcsTerminal.SpecHolds relation leaves.ccs leaves.ccsOffset env
  norm : NormTerminal.SpecHolds leaves.norm leaves.normOffset env
  finalIdentity : FinalIdentity.SpecHolds leaves.finalIdentity leaves.finalIdentityOffset env
  output : OutputBinding.SpecHolds leaves.output leaves.outputOffset env

variable (ajtai : PaperAlgebra.AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
  (running : Spec.Folding.Nifs.PaperNonInteractive.Running K PaperAlgebra.Commitment
    (PaperAlgebra.PublicInput (logicalWidth := logicalWidth) (publicFits := publicFits))
    LightShape)
  (fresh : Spec.Folding.Nifs.PaperNonInteractive.Fresh PaperAlgebra.Commitment
    (PaperAlgebra.PublicInput (logicalWidth := logicalWidth) (publicFits := publicFits))
    LightShape)
  (proof : Spec.Folding.Light.Proof LightShape (ProductionKey.degreeBound relation))

/-- The output message that the light check reads: the projected output with the running
source's norm input set to zero. -/
noncomputable abbrev lightMessage : ProtocolPolynomial.OutputMessage K LightShape :=
  Spec.Folding.Light.PiCCS.maskMessage extensionOps
    (((keys relation ajtai).light.statement running fresh).projectOutput proof.piCcsOutput)

/-- The wires between the leaves, and from the step's inputs and proof into them. -/
structure Wired (leaves : Leaves logicalWidth (ProductionKey.degreeBound relation) publicFits)
    (env : Env) : Prop where
  statementFresh : StatementAbsorption.evalFresh (leaves.statement.fresh leaves.statementOffset)
    env = ⟨fresh.commitments, fresh.publicInputs⟩
  challengeStart : ChallengeDerivation.evalState env
      (leaves.challenge.initialState leaves.challengeOffset) =
    StatementAbsorption.evalState env leaves.statementFinal
  roundsStart : RoundTranscript.evalState env (leaves.rounds.initialState leaves.roundsOffset) =
    ChallengeDerivation.evalState env
      (ChallengeDerivation.finalState leaves.challenge leaves.challengeOffset)
  roundsMessages : ∀ roundIndex,
    (leaves.rounds.round leaves.roundsOffset roundIndex).semanticPolynomial env =
      proof.piCcsRounds roundIndex
  initialGamma : (leaves.initial.gamma leaves.initialOffset).eval env =
    ChallengeDerivation.evalGamma leaves.challenge leaves.challengeOffset env
  initialPad : ∀ coordinate,
    (leaves.initial.eval_K leaves.initialOffset coordinate).eval env =
      (lightInput relation ajtai running fresh).claimedPadCoefficient coordinate
  initialMatrix : ∀ coordinate,
    (leaves.initial.eval_A leaves.initialOffset coordinate).eval env =
      (lightInput relation ajtai running fresh).claimedMatrixCoefficient coordinate
  evalKRoundPoint : (PointEquality.Owned.evalLeftPoint
      (EvalKTerminal.pointInterfaceAt leaves.evalK leaves.evalKOffset) leaves.evalKOffset
      env).coordinates =
    (RoundTranscript.evalRoundPoint leaves.rounds leaves.roundsOffset env).coordinates
  evalKPriorPoint : PointEquality.Owned.evalRightPoint
      (EvalKTerminal.pointInterfaceAt leaves.evalK leaves.evalKOffset) leaves.evalKOffset env =
    (lightInput relation ajtai running fresh).priorPoint
  evalKGamma : (leaves.evalK.gamma leaves.evalKOffset).eval env =
    ChallengeDerivation.evalGamma leaves.challenge leaves.challengeOffset env
  evalKOutput : ∀ coordinate,
    (leaves.evalK.outputEval_K leaves.evalKOffset coordinate).eval env =
      (lightMessage relation ajtai running fresh proof).padImage coordinate
  evalARoundPoint : (PointEquality.Owned.evalLeftPoint
      (EvalATerminal.pointInterfaceAt leaves.evalA leaves.evalAOffset) leaves.evalAOffset
      env).coordinates =
    (RoundTranscript.evalRoundPoint leaves.rounds leaves.roundsOffset env).coordinates
  evalAPriorPoint : PointEquality.Owned.evalRightPoint
      (EvalATerminal.pointInterfaceAt leaves.evalA leaves.evalAOffset) leaves.evalAOffset env =
    (lightInput relation ajtai running fresh).priorPoint
  evalAGamma : (leaves.evalA.gamma leaves.evalAOffset).eval env =
    ChallengeDerivation.evalGamma leaves.challenge leaves.challengeOffset env
  evalAOutput : ∀ coordinate,
    (leaves.evalA.outputEval_A leaves.evalAOffset coordinate).eval env =
      (lightMessage relation ajtai running fresh proof).matrixImage coordinate
  ccsFresh : ∀ matrix, (leaves.ccs.freshMatrix leaves.ccsOffset matrix).eval env =
    (lightMessage relation ajtai running fresh proof).freshMatrixImage lightFreshIndex matrix
  normGamma : (leaves.norm.gamma leaves.normOffset).eval env =
    ChallengeDerivation.evalGamma leaves.challenge leaves.challengeOffset env
  normSource : ∀ source, (leaves.norm.sourceAssignment leaves.normOffset source).eval env =
    (lightMessage relation ajtai running fresh proof).sourceAssignment source
  finalRoundPoint : (PointEquality.Owned.evalLeftPoint
      (FinalIdentity.pointInterfaceAt leaves.finalIdentity leaves.finalIdentityOffset)
      leaves.finalIdentityOffset env).coordinates =
    (RoundTranscript.evalRoundPoint leaves.rounds leaves.roundsOffset env).coordinates
  finalAlpha : (PointEquality.Owned.evalRightPoint
      (FinalIdentity.pointInterfaceAt leaves.finalIdentity leaves.finalIdentityOffset)
      leaves.finalIdentityOffset env).coordinates =
    (ChallengeDerivation.evalAlpha leaves.challenge leaves.challengeOffset env).coordinates
  finalGamma : (leaves.finalIdentity.gamma leaves.finalIdentityOffset).eval env =
    ChallengeDerivation.evalGamma leaves.challenge leaves.challengeOffset env
  finalPad : (leaves.finalIdentity.eval_K leaves.finalIdentityOffset).eval env =
    (EvalKTerminal.output leaves.evalK leaves.evalKOffset).eval env
  finalMatrix : (leaves.finalIdentity.eval_A leaves.finalIdentityOffset).eval env =
    (EvalATerminal.output leaves.evalA leaves.evalAOffset).eval env
  finalCcs : (leaves.finalIdentity.ccs leaves.finalIdentityOffset).eval env =
    (CcsTerminal.output relation leaves.ccs leaves.ccsOffset).eval env
  finalNorm : (leaves.finalIdentity.norm leaves.finalIdentityOffset).eval env =
    (NormTerminal.output leaves.norm leaves.normOffset).eval env
  sumcheckInitial : (leaves.sumcheck.initial leaves.sumcheckOffset).eval env =
    (InitialClaim.output leaves.initial leaves.initialOffset).eval env
  sumcheckRounds : ∀ roundIndex,
    (leaves.sumcheck.round leaves.sumcheckOffset roundIndex).semanticPolynomial env =
      proof.piCcsRounds roundIndex
  sumcheckPoint : (SumcheckChain.evalRoundPoint leaves.sumcheck leaves.sumcheckOffset
      env).coordinates =
    (RoundTranscript.evalRoundPoint leaves.rounds leaves.roundsOffset env).coordinates
  sumcheckTerminal : (SumcheckChain.output leaves.sumcheck leaves.sumcheckOffset).eval env =
    (leaves.finalIdentity.terminal leaves.finalIdentityOffset).eval env
  outputStart : List.ofFn (Layer.evalState env (leaves.output.initialState leaves.outputOffset)) =
    RoundTranscript.evalState env (RoundTranscript.finalState leaves.rounds leaves.roundsOffset)
  outputPad : ∀ source coefficient,
    ((leaves.output.output leaves.outputOffset).padCoordinate source coefficient).eval env =
      proof.piCcsOutput.padCoordinate source coefficient
  outputMatrix : ∀ source matrix coefficient,
    ((leaves.output.output leaves.outputOffset).matrixCoordinate source matrix coefficient).eval
      env = proof.piCcsOutput.matrixCoordinate source matrix coefficient

/-- What the light Π_CCS phase gives: the light check accepts, the SumCheck point is the light
key's, and the output binding ends in the state the light Π_RLC challenge reads. -/
structure PhaseHolds (leaves : Leaves logicalWidth (ProductionKey.degreeBound relation) publicFits)
    (env : Env) : Prop where
  accepted : Spec.Folding.Light.PiCCS.check (keys relation ajtai).light.extensionOps
    (keys relation ajtai).light.lift ((keys relation ajtai).light.statement running fresh)
    (ProductionKey.degreeBound relation) ((keys relation ajtai).light.probe running fresh proof) =
      true
  roundPoint : (RoundTranscript.evalRoundPoint leaves.rounds leaves.roundsOffset env).coordinates =
    (lightCoins relation ajtai running fresh proof.piCcsRounds).roundPoint.coordinates
  outgoingState :
    List.ofFn (Layer.evalState env (OutputBinding.finalState leaves.output leaves.outputOffset)) =
      (keys relation ajtai).light.absorbPiCcsOutput
        (lightCoins relation ajtai running fresh proof.piCcsRounds).finalState proof.piCcsOutput

/-- The light Π_CCS check accepts when the SumCheck chain runs from `T` to `Q(r')` on the masked
output. -/
theorem check_of_chain
    (chain : SumCheck.Finite.FixedPhase.Chain extensionOps.toOps
      ((lightInput relation ajtai running fresh).initial extensionOps
        (lightCoins relation ajtai running fresh proof.piCcsRounds).gamma)
      (List.ofFn proof.piCcsRounds)
      (lightCoins relation ajtai running fresh proof.piCcsRounds).roundPoint.coordinates
      (ProtocolPolynomial.terminalFromMessage extensionOps (lightInput relation ajtai running fresh)
        (lightCoins relation ajtai running fresh proof.piCcsRounds).alpha
        (lightCoins relation ajtai running fresh proof.piCcsRounds).gamma
        (lightCoins relation ajtai running fresh proof.piCcsRounds).roundPoint
        (lightMessage relation ajtai running fresh proof))) :
    Spec.Folding.Light.PiCCS.check (keys relation ajtai).light.extensionOps
      (keys relation ajtai).light.lift ((keys relation ajtai).light.statement running fresh)
      (ProductionKey.degreeBound relation) ((keys relation ajtai).light.probe running fresh proof) =
        true := by
  simp only [Spec.Folding.Light.PiCCS.check, Spec.Folding.Light.Key.probe,
    ProtocolPolynomial.FixedWidth.check, SumCheck.Finite.FixedPhase.RawCertificate.check_encode]
  exact (SumCheck.Finite.FixedPhase.checkChain_eq_true_iff _ _ _ _ _).mpr chain

/-- The converse of `check_of_chain`: an accepting light Π_CCS check gives the SumCheck chain from
`T` to `Q(r')` on the masked output. -/
theorem chain_of_check
    (accepted : Spec.Folding.Light.PiCCS.check (keys relation ajtai).light.extensionOps
      (keys relation ajtai).light.lift ((keys relation ajtai).light.statement running fresh)
      (ProductionKey.degreeBound relation) ((keys relation ajtai).light.probe running fresh proof) =
        true) :
    SumCheck.Finite.FixedPhase.Chain extensionOps.toOps
      ((lightInput relation ajtai running fresh).initial extensionOps
        (lightCoins relation ajtai running fresh proof.piCcsRounds).gamma)
      (List.ofFn proof.piCcsRounds)
      (lightCoins relation ajtai running fresh proof.piCcsRounds).roundPoint.coordinates
      (ProtocolPolynomial.terminalFromMessage extensionOps (lightInput relation ajtai running fresh)
        (lightCoins relation ajtai running fresh proof.piCcsRounds).alpha
        (lightCoins relation ajtai running fresh proof.piCcsRounds).gamma
        (lightCoins relation ajtai running fresh proof.piCcsRounds).roundPoint
        (lightMessage relation ajtai running fresh proof)) := by
  simp only [Spec.Folding.Light.PiCCS.check, Spec.Folding.Light.Key.probe,
    ProtocolPolynomial.FixedWidth.check, SumCheck.Finite.FixedPhase.RawCertificate.check_encode]
    at accepted
  exact (SumCheck.Finite.FixedPhase.checkChain_eq_true_iff _ _ _ _ _).mp accepted

private theorem point_eq {count : Nat} {left right : CubePoint K count}
    (coordinates : left.coordinates = right.coordinates) : left = right := by
  cases left
  cases right
  simp_all

/-- The leaf rows and the wires give the light Π_CCS phase. -/
theorem spec_implies_phaseHolds
    (leaves : Leaves logicalWidth (ProductionKey.degreeBound relation) publicFits) (env : Env)
    (specification : SpecHolds relation leaves env)
    (wired : Wired relation ajtai running fresh proof leaves env) :
    PhaseHolds relation ajtai running fresh proof leaves env := by
  have statementEq := statementState_eq relation ajtai running fresh leaves.statement
    leaves.statementOffset env _ wired.statementFresh specification.statement
  have challenges := challenges_eq relation ajtai running fresh proof.piCcsRounds
    leaves.challenge leaves.challengeOffset env (wired.challengeStart.trans statementEq)
    specification.challenge
  have rounds := roundPoint_eq relation ajtai running fresh proof.piCcsRounds leaves.rounds
    leaves.roundsOffset env (wired.roundsStart.trans challenges.2.2) wired.roundsMessages
    specification.rounds
  have initialEq := initial_eq relation ajtai running fresh leaves.initial leaves.initialOffset
    env _ (wired.initialGamma.trans challenges.2.1) wired.initialPad wired.initialMatrix
    specification.initial
  have padEq := padAtMessage_eq relation ajtai running fresh leaves.evalK leaves.evalKOffset env
    _ _ (lightMessage relation ajtai running fresh proof)
    (point_eq (wired.evalKRoundPoint.trans rounds.1)) wired.evalKPriorPoint
    (wired.evalKGamma.trans challenges.2.1) wired.evalKOutput specification.evalK
  have matrixEq := matrixAtMessage_eq relation ajtai running fresh leaves.evalA leaves.evalAOffset
    env _ _ (lightMessage relation ajtai running fresh proof)
    (point_eq (wired.evalARoundPoint.trans rounds.1)) wired.evalAPriorPoint
    (wired.evalAGamma.trans challenges.2.1) wired.evalAOutput specification.evalA
  have ccsEq := ccsAtMessage_eq relation ajtai running fresh leaves.ccs leaves.ccsOffset env
    (lightCoins relation ajtai running fresh proof.piCcsRounds).gamma
    (lightMessage relation ajtai running fresh proof) wired.ccsFresh specification.ccs
  have normEq := normAtMessage_eq leaves.norm leaves.normOffset env _
    (lightMessage relation ajtai running fresh proof) (wired.normGamma.trans challenges.2.1)
    wired.normSource specification.norm
  have terminalEq := terminal_eq relation ajtai running fresh leaves.finalIdentity
    leaves.finalIdentityOffset env _ _ _ (lightMessage relation ajtai running fresh proof)
    (point_eq (wired.finalRoundPoint.trans rounds.1))
    (point_eq (wired.finalAlpha.trans challenges.1)) (wired.finalGamma.trans challenges.2.1)
    (wired.finalPad.trans padEq) (wired.finalMatrix.trans matrixEq) (wired.finalCcs.trans ccsEq)
    (wired.finalNorm.trans normEq) specification.finalIdentity
  have chain := chain_eq relation ajtai running fresh proof.piCcsRounds leaves.sumcheck
    leaves.sumcheckOffset env (lightMessage relation ajtai running fresh proof)
    (wired.sumcheckInitial.trans initialEq) wired.sumcheckRounds
    (wired.sumcheckPoint.trans rounds.1) (wired.sumcheckTerminal.trans terminalEq)
    specification.sumcheck
  exact {
    accepted := check_of_chain relation ajtai running fresh proof chain
    roundPoint := rounds.1
    outgoingState := outgoing_eq relation ajtai leaves.output leaves.outputOffset env _ _
      (wired.outputStart.trans rounds.2) wired.outputPad wired.outputMatrix specification.output }

end NightstreamFPrime.Lifecycle.OptionB.LightPiCCSPhase
