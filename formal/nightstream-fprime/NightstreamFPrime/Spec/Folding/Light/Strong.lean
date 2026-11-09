import NightstreamFPrime.Spec.Folding.Light.PiCCSCompleteness

/-! SuperNeo v1.2 Lemma 7 for the light Π_CCS. When the masked check accepts and witnesses
open the output claims at the ambient bound, either the fresh source satisfies the CCS with
norm below 2 and the running claims hold at the prior point, or a Π_CCS bad event occurred.
The running source gets no norm conclusion. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Folding.Light.PiCCS

open NightstreamFPrime.Spec
open _root_.NightstreamFPrime.Spec.SumCheck.Finite
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.StrongReduction
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.MatrixCoefficientSource

universe uCommitment uPublicInput

variable {Commitment : Type uCommitment} {PublicInput : Type uPublicInput}
  {shape : Shape} {columns blockCount : Nat} {baseOps : InterpolationOps F}

def SumCheckFailure
    (extensionOps : InterpolationOps K) (lift : F -> K)
    (statement : Statement K Commitment PublicInput shape columns blockCount baseOps)
    (width challengeSetSize : Nat) (probe : Probe K shape)
    (witness : OutputWitness shape columns) : Prop :=
  ∃ certificate : SumCheck.Finite.FixedPhase.Certificate K width,
    SumCheck.Finite.FixedPhase.RawCertificate.decode width probe.response.rounds =
        some certificate ∧
      ProtocolPolynomial.FixedWidth.SumCheckCollision extensionOps
        (protocolData baseOps lift
          ((statement.sourceConnectedInputs witness).toUnifiedInputs baseOps))
        probe.coins.alpha probe.coins.gamma width challengeSetSize
        probe.coins.roundPoint certificate

/-- SuperNeo v1.2 Lemma 7, with norm terms on the fresh sources only. -/
theorem accepted_extracts_or_badEvent
    (baseLaws : InterpolationEvaluationLaws baseOps)
    (baseZero : NormResidualTable.BaseZeroAgreement baseOps)
    (noZeroDivisors : NormRange.BaseFieldNoZeroDivisors)
    (extensionOps : InterpolationOps K)
    (extensionLaws : InterpolationEvaluationLaws extensionOps)
    (extensionZeroLaws : InterpolationZeroLaws extensionOps)
    (lift : F -> K)
    (liftLaws : ProtocolDataRefinement.ProtocolLift baseOps extensionOps lift)
    (openingMaps : OpeningMaps Commitment PublicInput columns)
    (params : GlobalParams)
    (statement : Statement K Commitment PublicInput shape columns blockCount baseOps)
    (constantLaw : ConstantTermLaw baseOps statement.matrixSource.kernel)
    (width : Nat)
    (degreeCovers : (statement.verifierInput lift).sumcheckDegreeBound <= width)
    (challengeSetSize : Nat)
    (probe : Probe K shape)
    (witness : OutputWitness shape columns)
    (ambient : AmbientOutputHolds extensionOps lift openingMaps params statement probe witness)
    (accepted : Accepted extensionOps lift statement width probe) :
    SemanticTruth ((statement.sourceConnectedInputs witness).toUnifiedInputs baseOps)
        baseOps extensionOps lift ∨
      SignedCoefficientObject.MixingRoot extensionOps
        ((protocolData baseOps lift
          ((statement.sourceConnectedInputs witness).toUnifiedInputs baseOps)).toJointData
            extensionOps)
        probe.coins.alpha probe.coins.gamma ∨
      SumCheckFailure extensionOps lift statement width challengeSetSize probe witness := by
  let unified := (statement.sourceConnectedInputs witness).toUnifiedInputs baseOps
  let data := protocolData baseOps lift unified
  have inputEqual : data.toVerifierInput = statement.verifierInput lift := rfl
  have checked :
      ProtocolPolynomial.FixedWidth.check extensionOps width data.toVerifierInput
          probe.coins.alpha probe.coins.gamma probe.coins.roundPoint
          (maskMessage extensionOps (statement.projectOutput probe.response.fullOutput))
          probe.response.rounds = true :=
    accepted
  obtain ⟨certificate, decoded, chain⟩ :=
    (ProtocolPolynomial.FixedWidth.check_eq_true_iff extensionOps width
      data.toVerifierInput probe.coins.alpha probe.coins.gamma probe.coins.roundPoint
      (maskMessage extensionOps (statement.projectOutput probe.response.fullOutput))
      probe.response.rounds).1 checked
  have dataDegreeCovers : data.toVerifierInput.sumcheckDegreeBound <= width := by
    rw [inputEqual]
    exact degreeCovers
  rcases
      ProtocolPolynomial.FixedWidth.accepted_implies_tableTruth_or_badEvent
        extensionOps extensionLaws extensionZeroLaws data
        probe.coins.alpha probe.coins.gamma width dataDegreeCovers
        challengeSetSize probe.coins.roundPoint
        (maskMessage extensionOps (statement.projectOutput probe.response.fullOutput))
        certificate chain with
    tableTruth | mixingRoot | badChallenge | outputMismatch
  · left
    have independentTableTruth :
        (TableResidualData.toTableObligations extensionOps
          (SignedCoefficientObject.toTableResidualData extensionOps
            ((independentInputs unified).toJointData baseOps lift))).AllHold := by
      rw [← protocolData_toJointData_eq baseOps extensionOps lift liftLaws unified]
      exact tableTruth
    have independentSemantic :=
      (ConcreteJointData.jointTableTruth_iff_semanticTruth baseOps baseZero noZeroDivisors
        extensionOps extensionLaws lift liftLaws.toZeroReflectingLift
        (independentInputs unified)).mp independentTableTruth
    exact (independentInputs_semanticTruth_iff unified baseOps extensionOps lift).mp
      independentSemantic
  · exact Or.inr (Or.inl mixingRoot)
  · exact Or.inr (Or.inr ⟨certificate, decoded, badChallenge⟩)
  · exfalso
    apply outputMismatch
    unfold ProtocolPolynomial.qAtPoint
    have projected := projectedOutput_eq_messageAt_of_ambientOutputHolds
      baseLaws baseZero extensionOps lift openingMaps params statement constantLaw
      probe witness ambient
    have masked := messageAt_protocolData baseOps baseZero extensionOps extensionLaws
      extensionZeroLaws lift liftLaws unified probe.coins.roundPoint
    rw [projected]
    exact congrArg
      (ProtocolPolynomial.terminalFromMessage extensionOps data.toVerifierInput
        probe.coins.alpha probe.coins.gamma probe.coins.roundPoint) masked.symm

end NightstreamFPrime.Spec.Folding.Light.PiCCS
