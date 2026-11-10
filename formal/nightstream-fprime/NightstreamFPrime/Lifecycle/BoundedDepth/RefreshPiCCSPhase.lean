import NightstreamFPrime.Lifecycle.BoundedDepth.RefreshPiCCS
import NightstreamFPrime.Lifecycle.PiCCS.v1_2.CcsTerminal
import NightstreamFPrime.Lifecycle.PiCCS.v1_2.EvalATerminal
import NightstreamFPrime.Lifecycle.PiCCS.v1_2.EvalKTerminal
import NightstreamFPrime.Lifecycle.PiCCS.v1_2.FinalIdentity
import NightstreamFPrime.Lifecycle.PiCCS.v1_2.InitialClaim
import NightstreamFPrime.Lifecycle.PiCCS.v1_2.NormTerminal
import NightstreamFPrime.Lifecycle.PiCCS.v1_2.OutputBinding
import NightstreamFPrime.Lifecycle.PiCCS.v1_2.SumcheckChain

/-! The refresh Π_CCS phase (CW-3.1). Today's production-shape Π_CCS leaves, with the refresh
statement absorption (`RefreshPiCCS.statementActions`) and the wires of `Wired`, make the refresh
key's Π_CCS check accept on the 16 Π_DEC children and the fresh instance: SuperNeo v1.2
Section 7.3, Step 4, with every source's norm term. The phase also gives the SumCheck point and
the transcript state after the output binding, which the refresh Π_RLC reads. -/

namespace NightstreamFPrime.Lifecycle.BoundedDepth.RefreshPiCCSPhase

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Circuit.Quadratic
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ConcreteCarrier
open NightstreamFPrime.Gadgets.Poseidon2
open NightstreamFPrime.Gadgets.Multilinear
open NightstreamFPrime.Lifecycle.PiCCS.v1_2
open NightstreamFPrime.Lifecycle.BoundedDepth.RefreshPiCCS

/-- The leaf gadgets of the refresh Π_CCS and their offsets. `statementFinal` is the state after
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
  initial : InitialClaim.Interface
  initialOffset : Nat
  sumcheck : SumcheckChain.Interface degreeBound
  sumcheckOffset : Nat
  evalK : EvalKTerminal.Interface
  evalKOffset : Nat
  evalA : EvalATerminal.Interface
  evalAOffset : Nat
  ccs : CcsTerminal.Interface
  ccsOffset : Nat
  norm : NormTerminal.Interface
  normOffset : Nat
  finalIdentity : FinalIdentity.Interface
  finalIdentityOffset : Nat
  output : OutputBinding.Interface
  outputOffset : Nat

variable {logicalWidth : Nat}
  {publicFits : ringDegree * PaperAlgebra.publicRingColumns ≤
    Phi81CarrierLayout.carrierWidth logicalWidth}
  (relation : ProductionKey.LogicalRelation logicalWidth publicFits)

/-- Each leaf's rows hold. The statement absorption is the trace from the refresh fold domain. -/
structure SpecHolds (leaves : Leaves logicalWidth (ProductionKey.degreeBound relation) publicFits)
    (env : Env) : Prop where
  statement : Duplex.Formal.TraceHolds (StatementAbsorption.evalState env Hash.zeroE)
    ((statementActions leaves.statement leaves.statementOffset).map
      (Duplex.Formal.Action.eval env))
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
  (children : Children logicalWidth publicFits)
  (fresh : Spec.Folding.Nifs.PaperNonInteractive.Fresh PaperAlgebra.Commitment
    (PaperAlgebra.PublicInput (logicalWidth := logicalWidth) (publicFits := publicFits))
    productionShape)
  (proof : Spec.Folding.Nifs.PaperNonInteractive.Proof K PaperAlgebra.Commitment productionShape
    (ProductionKey.degreeBound relation))

/-- The refresh statement's verifier input. -/
noncomputable abbrev refreshInput : ProtocolPolynomial.VerifierInput K productionShape :=
  ((refreshKey relation ajtai).statement children fresh).verifierInput
    (refreshKey relation ajtai).lift

/-- The output message that the refresh check reads: the projected output of every source. -/
noncomputable abbrev refreshMessage : ProtocolPolynomial.OutputMessage K productionShape :=
  ((refreshKey relation ajtai).statement children fresh).projectOutput proof.piCcsOutput

/-- The refresh Π_CCS's one fresh source. -/
def freshIndex : Fin productionShape.freshCount := ⟨0, by decide⟩

/-- The wires between the leaves, and from the step's children, fresh instance and proof into
them. -/
structure Wired (leaves : Leaves logicalWidth (ProductionKey.degreeBound relation) publicFits)
    (env : Env) : Prop where
  statementFresh : StatementAbsorption.evalFresh (leaves.statement.fresh leaves.statementOffset)
    env = fresh
  statementCommitments : ∀ child row lane,
    ((leaves.statement.running leaves.statementOffset).commitment child row lane).eval env =
      children.commitments child row lane
  statementEvaluations : ∀ child,
    StatementAbsorption.evalEvaluation
      ((leaves.statement.running leaves.statementOffset).evaluation child) env =
        children.evaluations child
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
      (refreshInput relation ajtai children fresh).claimedPadCoefficient coordinate
  initialMatrix : ∀ coordinate,
    (leaves.initial.eval_A leaves.initialOffset coordinate).eval env =
      (refreshInput relation ajtai children fresh).claimedMatrixCoefficient coordinate
  evalKRoundPoint : (PointEquality.Owned.evalLeftPoint
      (EvalKTerminal.pointInterfaceAt leaves.evalK leaves.evalKOffset) leaves.evalKOffset
      env).coordinates =
    (RoundTranscript.evalRoundPoint leaves.rounds leaves.roundsOffset env).coordinates
  evalKPriorPoint : PointEquality.Owned.evalRightPoint
      (EvalKTerminal.pointInterfaceAt leaves.evalK leaves.evalKOffset) leaves.evalKOffset env =
    (refreshInput relation ajtai children fresh).priorPoint
  evalKGamma : (leaves.evalK.gamma leaves.evalKOffset).eval env =
    ChallengeDerivation.evalGamma leaves.challenge leaves.challengeOffset env
  evalKOutput : ∀ coordinate,
    (leaves.evalK.outputEval_K leaves.evalKOffset coordinate).eval env =
      (refreshMessage relation ajtai children fresh proof).padImage coordinate
  evalARoundPoint : (PointEquality.Owned.evalLeftPoint
      (EvalATerminal.pointInterfaceAt leaves.evalA leaves.evalAOffset) leaves.evalAOffset
      env).coordinates =
    (RoundTranscript.evalRoundPoint leaves.rounds leaves.roundsOffset env).coordinates
  evalAPriorPoint : PointEquality.Owned.evalRightPoint
      (EvalATerminal.pointInterfaceAt leaves.evalA leaves.evalAOffset) leaves.evalAOffset env =
    (refreshInput relation ajtai children fresh).priorPoint
  evalAGamma : (leaves.evalA.gamma leaves.evalAOffset).eval env =
    ChallengeDerivation.evalGamma leaves.challenge leaves.challengeOffset env
  evalAOutput : ∀ coordinate,
    (leaves.evalA.outputEval_A leaves.evalAOffset coordinate).eval env =
      (refreshMessage relation ajtai children fresh proof).matrixImage coordinate
  ccsFresh : ∀ matrix, (leaves.ccs.freshMatrix leaves.ccsOffset matrix).eval env =
    (refreshMessage relation ajtai children fresh proof).freshMatrixImage freshIndex matrix
  normGamma : (leaves.norm.gamma leaves.normOffset).eval env =
    ChallengeDerivation.evalGamma leaves.challenge leaves.challengeOffset env
  normSource : ∀ source, (leaves.norm.sourceAssignment leaves.normOffset source).eval env =
    (refreshMessage relation ajtai children fresh proof).sourceAssignment source
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

/-- What the refresh Π_CCS phase gives: the refresh key's Π_CCS check accepts, and the
SumCheck point and the outgoing transcript state are the refresh key's. -/
structure PhaseHolds (leaves : Leaves logicalWidth (ProductionKey.degreeBound relation) publicFits)
    (env : Env) : Prop where
  accepted : Spec.Folding.Nifs.PaperNonInteractive.piCcsCheck (refreshKey relation ajtai)
    children fresh proof = true
  roundPoint : (RoundTranscript.evalRoundPoint leaves.rounds leaves.roundsOffset env).coordinates =
    (refreshCoins relation ajtai children fresh proof).roundPoint.coordinates
  outgoingState :
    List.ofFn (Layer.evalState env (OutputBinding.finalState leaves.output leaves.outputOffset)) =
      (refreshKey relation ajtai).absorbPiCcsOutput
        (refreshCoins relation ajtai children fresh proof).finalState proof.piCcsOutput

/-- The refresh key's Π_CCS check accepts when the SumCheck chain runs from `T` to `Q(r')`. -/
theorem check_of_chain
    (chain : SumCheck.Finite.FixedPhase.Chain extensionOps.toOps
      ((refreshInput relation ajtai children fresh).initial extensionOps
        (refreshCoins relation ajtai children fresh proof).gamma)
      (List.ofFn proof.piCcsRounds)
      (refreshCoins relation ajtai children fresh proof).roundPoint.coordinates
      (ProtocolPolynomial.terminalFromMessage extensionOps
        (refreshInput relation ajtai children fresh)
        (refreshCoins relation ajtai children fresh proof).alpha
        (refreshCoins relation ajtai children fresh proof).gamma
        (refreshCoins relation ajtai children fresh proof).roundPoint
        (refreshMessage relation ajtai children fresh proof))) :
    Spec.Folding.Nifs.PaperNonInteractive.piCcsCheck (refreshKey relation ajtai) children fresh
      proof = true := by
  unfold Spec.Folding.Nifs.PaperNonInteractive.piCcsCheck
  exact (SumCheck.Finite.FixedPhase.checkChain_eq_true_iff _ _ _ _ _).mpr chain

/-- The converse of `check_of_chain`: an accepting check gives the SumCheck chain. -/
theorem chain_of_check
    (accepted : Spec.Folding.Nifs.PaperNonInteractive.piCcsCheck (refreshKey relation ajtai)
      children fresh proof = true) :
    SumCheck.Finite.FixedPhase.Chain extensionOps.toOps
      ((refreshInput relation ajtai children fresh).initial extensionOps
        (refreshCoins relation ajtai children fresh proof).gamma)
      (List.ofFn proof.piCcsRounds)
      (refreshCoins relation ajtai children fresh proof).roundPoint.coordinates
      (ProtocolPolynomial.terminalFromMessage extensionOps
        (refreshInput relation ajtai children fresh)
        (refreshCoins relation ajtai children fresh proof).alpha
        (refreshCoins relation ajtai children fresh proof).gamma
        (refreshCoins relation ajtai children fresh proof).roundPoint
        (refreshMessage relation ajtai children fresh proof)) := by
  unfold Spec.Folding.Nifs.PaperNonInteractive.piCcsCheck at accepted
  exact (SumCheck.Finite.FixedPhase.checkChain_eq_true_iff _ _ _ _ _).mp accepted

private theorem point_eq {count : Nat} {left right : CubePoint K count}
    (coordinates : left.coordinates = right.coordinates) : left = right := by
  cases left
  cases right
  simp_all

/-- The leaf rows and the wires give the refresh Π_CCS phase. -/
theorem spec_implies_phaseHolds
    (leaves : Leaves logicalWidth (ProductionKey.degreeBound relation) publicFits) (env : Env)
    (specification : SpecHolds relation leaves env)
    (wired : Wired relation ajtai children fresh proof leaves env) :
    PhaseHolds relation ajtai children fresh proof leaves env := by
  have statementEq := statementState_eq relation ajtai children fresh leaves.statement
    leaves.statementOffset env _ wired.statementFresh wired.statementCommitments
    wired.statementEvaluations specification.statement
  have challenges := challenges_eq relation ajtai children fresh proof leaves.challenge
    leaves.challengeOffset env (wired.challengeStart.trans statementEq) specification.challenge
  have rounds := roundPoint_eq relation ajtai children fresh proof leaves.rounds
    leaves.roundsOffset env (wired.roundsStart.trans challenges.2.2) wired.roundsMessages
    specification.rounds
  have initialEq := InitialClaim.spec_implies_initial leaves.initial leaves.initialOffset env
    (refreshInput relation ajtai children fresh) _ (wired.initialGamma.trans challenges.2.1)
    wired.initialPad wired.initialMatrix specification.initial
  have padEq := EvalKTerminal.spec_implies_padAtMessage leaves.evalK leaves.evalKOffset env
    (refreshInput relation ajtai children fresh) _ _
    (refreshMessage relation ajtai children fresh proof)
    (point_eq (wired.evalKRoundPoint.trans rounds.1)) wired.evalKPriorPoint
    (wired.evalKGamma.trans challenges.2.1) wired.evalKOutput specification.evalK
  have matrixEq := EvalATerminal.spec_implies_matrixAtMessage leaves.evalA leaves.evalAOffset
    env (refreshInput relation ajtai children fresh) _ _
    (refreshMessage relation ajtai children fresh proof)
    (point_eq (wired.evalARoundPoint.trans rounds.1)) wired.evalAPriorPoint
    (wired.evalAGamma.trans challenges.2.1) wired.evalAOutput specification.evalA
  have ccsEq : (CcsTerminal.output relation leaves.ccs leaves.ccsOffset).eval env =
      ProtocolPolynomial.ccsAtMessage extensionOps (refreshInput relation ajtai children fresh)
        (refreshCoins relation ajtai children fresh proof).gamma
        (refreshMessage relation ajtai children fresh proof) := by
    have value : (CcsTerminal.output relation leaves.ccs leaves.ccsOffset).eval env =
        CCSResidualTable.evaluatePolynomial extensionOps
          (refreshInput relation ajtai children fresh).constraintPolynomial
          (fun matrix => (leaves.ccs.freshMatrix leaves.ccsOffset matrix).eval env) :=
      specification.ccs
    rw [show (fun matrix => (leaves.ccs.freshMatrix leaves.ccsOffset matrix).eval env) =
        (refreshMessage relation ajtai children fresh proof).freshMatrixImage freshIndex from
      funext wired.ccsFresh] at value
    exact value.trans (CcsTerminal.ccsAtMessage_eq_singleFresh _ _ _).symm
  have normEq := NormTerminal.spec_implies_normAtMessage leaves.norm leaves.normOffset env _
    (refreshMessage relation ajtai children fresh proof)
    (wired.normGamma.trans challenges.2.1) wired.normSource specification.norm
  have terminalEq := FinalIdentity.spec_implies_terminal leaves.finalIdentity
    leaves.finalIdentityOffset env rfl (refreshInput relation ajtai children fresh) _ _ _
    (refreshMessage relation ajtai children fresh proof)
    (point_eq (wired.finalRoundPoint.trans rounds.1))
    (point_eq (wired.finalAlpha.trans (congrArg CubePoint.coordinates challenges.1)))
    (wired.finalGamma.trans challenges.2.1)
    (wired.finalPad.trans padEq) (wired.finalMatrix.trans matrixEq) (wired.finalCcs.trans ccsEq)
    (wired.finalNorm.trans normEq) specification.finalIdentity
  have chain := SumcheckChain.spec_implies_chain leaves.sumcheck leaves.sumcheckOffset env _
    proof.piCcsRounds _ _ (wired.sumcheckInitial.trans initialEq) wired.sumcheckRounds
    (wired.sumcheckPoint.trans rounds.1) (wired.sumcheckTerminal.trans terminalEq)
    specification.sumcheck
  exact {
    accepted := check_of_chain relation ajtai children fresh proof chain
    roundPoint := rounds.1
    outgoingState := OutputBinding.spec_implies_absorb leaves.output leaves.outputOffset env _ _
      (wired.outputStart.trans rounds.2) wired.outputPad wired.outputMatrix specification.output }

end NightstreamFPrime.Lifecycle.BoundedDepth.RefreshPiCCSPhase
