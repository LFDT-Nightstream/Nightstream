import NightstreamFPrime.Spec.Folding.Light.PiCCS
import NightstreamFPrime.Spec.Folding.Nifs.PaperStrongCompleteness

/-! SuperNeo v1.2 Lemma 10 for the light Π_CCS. The final check reads the running sources'
norm inputs as zero, and an honest prover is accepted when only the fresh sources are
norm bounded. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Folding.Light.PiCCS

open NightstreamFPrime.Spec
open _root_.NightstreamFPrime.Spec.SumCheck.Finite
open _root_.NightstreamFPrime.Spec.SumCheck.Finite.FieldCausalTrace (issued)
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.StrongReduction
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.MatrixCoefficientSource
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.UnifiedSources

universe uField uCommitment uPublicInput

def maskMessage {Field : Type uField} {shape : Shape} (ops : InterpolationOps Field)
    (message : ProtocolPolynomial.OutputMessage Field shape) :
    ProtocolPolynomial.OutputMessage Field shape :=
  { message with
    sourceAssignment := fun source =>
      if source.val < shape.freshCount then message.sourceAssignment source else ops.zero }

/-- The light Π_CCS check: SuperNeo v1.2 Section 7.3, Step 4, on the masked message. -/
def check {Commitment : Type uCommitment} {PublicInput : Type uPublicInput}
    {shape : Shape} {columns blockCount : Nat} {baseOps : InterpolationOps F}
    (extensionOps : InterpolationOps K) (lift : F -> K)
    (statement : Statement K Commitment PublicInput shape columns blockCount baseOps)
    (width : Nat) (probe : Probe K shape) : Bool :=
  ProtocolPolynomial.FixedWidth.check extensionOps width (statement.verifierInput lift)
    probe.coins.alpha probe.coins.gamma probe.coins.roundPoint
    (maskMessage extensionOps (statement.projectOutput probe.response.fullOutput))
    probe.response.rounds

def Accepted {Commitment : Type uCommitment} {PublicInput : Type uPublicInput}
    {shape : Shape} {columns blockCount : Nat} {baseOps : InterpolationOps F}
    (extensionOps : InterpolationOps K) (lift : F -> K)
    (statement : Statement K Commitment PublicInput shape columns blockCount baseOps)
    (width : Nat) (probe : Probe K shape) : Prop :=
  check extensionOps lift statement width probe = true

theorem evaluateCoordinates_tabulate_zero {Field : Type uField}
    (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops)
    (zeroLaws : InterpolationZeroLaws ops) :
    ∀ {depth : Nat} (coordinates : List Field),
      (BooleanTable.tabulate (fun _ : BooleanVertex depth => ops.zero)).evaluateCoordinates
        ops coordinates = ops.zero
  | 0, [] => by simp [BooleanTable.tabulate, BooleanTable.evaluateCoordinates]
  | 0, _ :: _ => by simp [BooleanTable.tabulate, BooleanTable.evaluateCoordinates]
  | _ + 1, [] => by simp [BooleanTable.tabulate, BooleanTable.evaluateCoordinates]
  | depth + 1, _ :: coordinates => by
      have lower :=
        evaluateCoordinates_tabulate_zero ops laws zeroLaws (depth := depth) coordinates
      simp only [BooleanTable.tabulate, BooleanTable.evaluateCoordinates, lower,
        InterpolationOps.sub, zeroLaws.neg_zero, laws.add_zero, laws.mul_zero]

theorem messageAt_protocolData
    {shape : Shape} {columns : Nat}
    (baseOps : InterpolationOps F)
    (baseZero : NormResidualTable.BaseZeroAgreement baseOps)
    (extensionOps : InterpolationOps K)
    (extensionLaws : InterpolationEvaluationLaws extensionOps)
    (extensionZeroLaws : InterpolationZeroLaws extensionOps)
    (lift : F -> K)
    (liftLaws : ProtocolDataRefinement.ProtocolLift baseOps extensionOps lift)
    (data : UnifiedInputs K shape columns)
    (point : CubePoint K shape.cubeVariables) :
    ProtocolPolynomial.messageAt extensionOps (protocolData baseOps lift data) point =
      maskMessage extensionOps
        (ProtocolPolynomial.messageAt extensionOps
          (ProtocolDataRefinement.toProtocolData baseOps lift data) point) := by
  apply ProtocolPolynomial.OutputMessage.ext
  · rfl
  · funext source
    by_cases isFresh : source.val < shape.freshCount
    · simp [ProtocolPolynomial.messageAt, protocolData, maskMessage,
        ProtocolDataRefinement.toProtocolData, normView, normAssignments, isFresh]
    · have liftZero : lift 0 = extensionOps.zero :=
        (liftLaws.zero_iff 0).mpr baseZero.zero_eq.symm
      have padded : ∀ vertex,
          data.layout.paddedValue 0 (fun _ : Fin columns => (0 : F)) vertex = 0 := by
        intro vertex
        unfold ColumnLayout.paddedValue
        split <;> rfl
      simp only [ProtocolPolynomial.messageAt, protocolData, maskMessage,
        ProtocolDataRefinement.toProtocolData, normView, normAssignments, isFresh, ↓reduceIte,
        padded, liftZero, BooleanTable.evaluate]
      exact evaluateCoordinates_tabulate_zero extensionOps extensionLaws extensionZeroLaws _
  · rfl
  · rfl

variable {Commitment : Type uCommitment} {PublicInput : Type uPublicInput}
  {shape : Shape} {columns blockCount : Nat}

theorem initial_true
    (baseOps : InterpolationOps F)
    (baseZero : NormResidualTable.BaseZeroAgreement baseOps)
    (noZeroDivisors : NormRange.BaseFieldNoZeroDivisors)
    (extensionOps : InterpolationOps K)
    (extensionLaws : InterpolationEvaluationLaws extensionOps)
    (extensionZeroLaws : InterpolationZeroLaws extensionOps)
    (lift : F -> K)
    (liftLaws : ProtocolDataRefinement.ProtocolLift baseOps extensionOps lift)
    (statement : Statement K Commitment PublicInput shape columns blockCount baseOps)
    (witness : OutputWitness shape columns)
    (truth : SemanticTruth ((statement.sourceConnectedInputs witness).toUnifiedInputs baseOps)
      baseOps extensionOps lift)
    (alpha : CubePoint K shape.cubeVariables) (gamma : K) :
    (statement.verifierInput lift).initial extensionOps gamma =
      FixedPhase.semanticInitial extensionOps.toOps
        (ProtocolPolynomial.polynomial extensionOps
          (protocolData baseOps lift
            ((statement.sourceConnectedInputs witness).toUnifiedInputs baseOps))
          alpha gamma)
        shape.cubeVariables := by
  let data := protocolData baseOps lift
    ((statement.sourceConnectedInputs witness).toUnifiedInputs baseOps)
  have coefficients := (SignedCoefficientObject.coefficientTruth_iff_tableObligations
    extensionOps extensionZeroLaws (data.toJointData extensionOps)).mpr
      (jointTableTruth baseOps baseZero noZeroDivisors extensionOps extensionLaws lift liftLaws
        _ truth)
  have zero := (SignedCoefficientObject.evaluate_eq_zero_iff_coefficientTruth_or_mixingRoot
    extensionOps extensionLaws (data.toJointData extensionOps) alpha gamma).mpr
      (Or.inl coefficients)
  rw [← SignedCoefficientPolynomial.paperDifference_eq_evaluate
    extensionOps extensionLaws (data.toJointData extensionOps) alpha gamma] at zero
  have equality := (FiniteSumAlgebra.sub_eq_zero_iff extensionOps extensionLaws _ _).mp zero
  change (statement.verifierInput lift).initial extensionOps gamma =
    HypercubeTruth.sumCompletions extensionOps.toOps
      (ProtocolPolynomial.polynomial extensionOps data alpha gamma) [] shape.cubeVariables
  rw [show statement.verifierInput lift = data.toVerifierInput from rfl,
    ProtocolPolynomial.verifierInput_initial_eq_joint_initial,
    ProtocolPolynomial.sumCompletions_polynomial_eq_summedQ extensionOps extensionLaws]
  exact equality

theorem representable
    (baseOps : InterpolationOps F)
    (extensionOps : InterpolationOps K)
    (extensionLaws : InterpolationEvaluationLaws extensionOps)
    (lift : F -> K)
    (statement : Statement K Commitment PublicInput shape columns blockCount baseOps)
    (witness : OutputWitness shape columns)
    {width : Nat}
    (degree : (statement.verifierInput lift).sumcheckDegreeBound = width)
    (alpha : CubePoint K shape.cubeVariables) (gamma : K) :
    FixedPhase.Sequential.RoundRepresentable extensionOps.toOps
      (ProtocolPolynomial.polynomial extensionOps
        (protocolData baseOps lift
          ((statement.sourceConnectedInputs witness).toUnifiedInputs baseOps))
        alpha gamma)
      width shape.cubeVariables := by
  let data := protocolData baseOps lift
    ((statement.sourceConnectedInputs witness).toUnifiedInputs baseOps)
  have degreeEq : data.toVerifierInput.sumcheckDegreeBound = width := degree
  simpa only [degreeEq] using ProtocolPolynomialDegree.sequentialRoundRepresentable
    extensionOps extensionLaws data alpha gamma

private noncomputable def honestProver
    (baseOps : InterpolationOps F)
    (extensionOps : InterpolationOps K)
    (extensionLaws : InterpolationEvaluationLaws extensionOps)
    (lift : F -> K)
    (statement : Statement K Commitment PublicInput shape columns blockCount baseOps)
    (witness : OutputWitness shape columns)
    (width : Nat)
    (degree : (statement.verifierInput lift).sumcheckDegreeBound = width) :
    CausalExecution.Prover shape columns width where
  rounds := fun alpha gamma => Classical.choose
    (Nifs.PaperStrongCompleteness.exists_causal_honest_strategy extensionOps.toOps
      (ProtocolPolynomial.polynomial extensionOps
        (protocolData baseOps lift
          ((statement.sourceConnectedInputs witness).toUnifiedInputs baseOps))
        alpha gamma)
      width shape.cubeVariables
      (representable baseOps extensionOps extensionLaws lift statement witness degree
        alpha gamma))
  output := fun _ _ coordinates =>
    if dimension : coordinates.length = shape.cubeVariables then
      some (FullOutputCoordinates.FullOutput.honestAt baseOps extensionOps lift
        (statement.sourceConnectedInputs witness) ⟨coordinates, dimension⟩, witness)
    else none

/-- SuperNeo v1.2 Lemma 10, with norm terms on the fresh sources only. -/
theorem exists_honest_prover
    (baseOps : InterpolationOps F)
    (baseLaws : InterpolationEvaluationLaws baseOps)
    (baseZero : NormResidualTable.BaseZeroAgreement baseOps)
    (noZeroDivisors : NormRange.BaseFieldNoZeroDivisors)
    (extensionOps : InterpolationOps K)
    (extensionLaws : InterpolationEvaluationLaws extensionOps)
    (extensionZeroLaws : InterpolationZeroLaws extensionOps)
    (lift : F -> K)
    (liftLaws : ProtocolDataRefinement.ProtocolLift baseOps extensionOps lift)
    (statement : Statement K Commitment PublicInput shape columns blockCount baseOps)
    (constantLaw : ConstantTermLaw baseOps statement.matrixSource.kernel)
    (witness : OutputWitness shape columns)
    (width : Nat)
    (degree : (statement.verifierInput lift).sumcheckDegreeBound = width)
    (truth : SemanticTruth ((statement.sourceConnectedInputs witness).toUnifiedInputs baseOps)
      baseOps extensionOps lift) :
    ∃ prover : CausalExecution.Prover shape columns width,
      ∀ (alpha : CubePoint K shape.cubeVariables) (gamma : K)
        (point : CubePoint K shape.cubeVariables),
        ∃ probe : Probe K shape,
          CausalExecution.run prover alpha gamma point = some (probe, witness) ∧
          Accepted extensionOps lift statement width probe ∧
          probe.coins = ({ alpha, gamma, roundPoint := point } : PublicCoins K shape) ∧
          probe.response.fullOutput =
            FullOutputCoordinates.FullOutput.honestAt baseOps extensionOps lift
              (statement.sourceConnectedInputs witness) point := by
  let prover := honestProver baseOps extensionOps extensionLaws lift statement witness width
    degree
  refine ⟨prover, ?_⟩
  intro alpha gamma point
  let unified := (statement.sourceConnectedInputs witness).toUnifiedInputs baseOps
  let data := protocolData baseOps lift unified
  let q := ProtocolPolynomial.polynomial extensionOps data alpha gamma
  let honestOutput := FullOutputCoordinates.FullOutput.honestAt baseOps extensionOps lift
    (statement.sourceConnectedInputs witness) point
  obtain ⟨certificate, execution, honest⟩ := Classical.choose_spec
    (Nifs.PaperStrongCompleteness.exists_causal_honest_strategy extensionOps.toOps q width
      shape.cubeVariables
      (representable baseOps extensionOps extensionLaws lift statement witness degree
        alpha gamma))
    point.coordinates point.dimension
  let probe : Probe K shape := {
    coins := { alpha, gamma, roundPoint := point }
    response := {
      rounds := FixedPhase.RawCertificate.encode certificate
      fullOutput := honestOutput } }
  refine ⟨probe, ?_, ?_, rfl, rfl⟩
  · have issuedRounds : issued (prover.rounds alpha gamma) [] point.coordinates =
        some certificate.rounds := execution
    unfold CausalExecution.run
    rw [issuedRounds]
    simp only [prover, honestProver, dite_eq_left point.dimension, Option.map_some]
    rfl
  · have masked : maskMessage extensionOps (statement.projectOutput honestOutput) =
        ProtocolPolynomial.messageAt extensionOps data point := by
      rw [statement.projectOutput_eq_toOutputMessage witness,
        FullOutputCoordinates.FullOutput.honestAt_toOutputMessage_eq_messageAt
          baseOps baseLaws baseZero extensionOps lift (statement.sourceConnectedInputs witness)
          constantLaw point,
        messageAt_protocolData baseOps baseZero extensionOps extensionLaws extensionZeroLaws
          lift liftLaws unified point]
    change ProtocolPolynomial.FixedWidth.check extensionOps width
      (statement.verifierInput lift) alpha gamma point
      (maskMessage extensionOps (statement.projectOutput honestOutput))
      (FixedPhase.RawCertificate.encode certificate) = true
    apply (ProtocolPolynomial.FixedWidth.check_eq_true_iff extensionOps width
      (statement.verifierInput lift) alpha gamma point
      (maskMessage extensionOps (statement.projectOutput honestOutput))
      (FixedPhase.RawCertificate.encode certificate)).mpr
    refine ⟨certificate, FixedPhase.RawCertificate.decode_encode certificate, ?_⟩
    have terminalEqual : ProtocolPolynomial.terminalFromMessage extensionOps
        (statement.verifierInput lift) alpha gamma point
        (maskMessage extensionOps (statement.projectOutput honestOutput)) =
        q point.coordinates := by
      rw [masked]
      change ProtocolPolynomial.qAtPoint extensionOps data alpha gamma point =
        ProtocolPolynomial.polynomial extensionOps data alpha gamma point.coordinates
      unfold ProtocolPolynomial.polynomial
      rw [dite_eq_left point.dimension]
    rw [terminalEqual]
    apply FixedPhase.complete extensionOps.toOps q _ point.coordinates certificate _ honest
    simpa only [point.dimension] using initial_true baseOps baseZero noZeroDivisors
      extensionOps extensionLaws extensionZeroLaws lift liftLaws statement witness truth
      alpha gamma

end NightstreamFPrime.Spec.Folding.Light.PiCCS
