import NightstreamFPrime.Spec.Folding.Nifs.PaperStrongInterface
import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.CausalExecution

/-!
Honest causal completeness for the PiCCS prefix of the NIFS. Each selected
round polynomial depends only on the fixed semantic polynomial and prior
challenges. Future challenges are absent from that choice.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Folding.Nifs.PaperStrongCompleteness

open NightstreamFPrime.Spec
open _root_.NightstreamFPrime.Spec.SumCheck.Finite
open GoldilocksCausal (Strategy)
open GoldilocksCausalTrace (issued)

private noncomputable def honestStrategy (ops : Ops K) (q : List K → K)
    (degree roundCount : Nat)
    (representable : FixedPhase.Sequential.RoundRepresentable ops q degree roundCount) :
    Strategy degree := fun fixed =>
  if remaining : fixed.length < roundCount then
    some (Classical.choose (representable fixed (roundCount - fixed.length - 1) (by omega)))
  else none

private theorem honestStrategy_issued (ops : Ops K) (q : List K → K)
    (degree roundCount : Nat)
    (representable : FixedPhase.Sequential.RoundRepresentable ops q degree roundCount)
    (fixed challenges : List K) (length : fixed.length + challenges.length = roundCount) :
    ∃ rounds : List (FixedPolynomial K degree),
      issued (honestStrategy ops q degree roundCount representable) fixed challenges = some rounds ∧
      FixedPhase.Representations ops rounds
        (HypercubeTruth.expectedPolynomialsFrom ops q fixed challenges) := by
  induction challenges generalizing fixed with
  | nil =>
      exact ⟨[], rfl, trivial⟩
  | cons challenge challenges ih =>
      have remaining : fixed.length < roundCount := by
        simp only [List.length_cons] at length
        omega
      have dimension : fixed.length + 1 + (roundCount - fixed.length - 1) = roundCount := by omega
      let message := Classical.choose (representable fixed (roundCount - fixed.length - 1) dimension)
      have represents := Classical.choose_spec (representable fixed (roundCount - fixed.length - 1) dimension)
      have tailLength : (fixed ++ [challenge]).length + challenges.length = roundCount := by
        simp only [List.length_append, List.length_cons, List.length_nil] at length ⊢
        omega
      obtain ⟨tail, later, honestTail⟩ := ih (fixed ++ [challenge]) tailLength
      refine ⟨message :: tail, ?_, ?_⟩
      · simp only [issued, honestStrategy, dif_pos remaining]
        rw [later]
        rfl
      · change FixedPhase.Represents ops message
          (fun point => HypercubeTruth.sumCompletions ops q (fixed ++ [point]) challenges.length) ∧ _
        refine ⟨?_, honestTail⟩
        have remainingEq : roundCount - fixed.length - 1 = challenges.length := by
          simp only [List.length_cons] at length
          omega
        intro point
        exact (represents point).trans
          (congrArg (HypercubeTruth.sumCompletions ops q (fixed ++ [point])) remainingEq)

/-- One fixed prefix-only strategy is honest on every public challenge
stream of the selected length. No messages are chosen from future coins. -/
theorem exists_causal_honest_strategy (ops : Ops K) (q : List K → K)
    (degree roundCount : Nat)
    (representable : FixedPhase.Sequential.RoundRepresentable ops q degree roundCount) :
    ∃ strategy : Strategy degree, ∀ challenges : List K,
      challenges.length = roundCount →
        ∃ certificate : FixedPhase.Certificate K degree,
          issued strategy [] challenges = some certificate.rounds ∧
          FixedPhase.Honest ops q challenges certificate := by
  refine ⟨honestStrategy ops q degree roundCount representable, ?_⟩
  intro challenges length
  obtain ⟨rounds, execution, honest⟩ := honestStrategy_issued ops q degree roundCount
    representable [] challenges (by simpa only [List.length_nil, Nat.zero_add] using length)
  exact ⟨⟨rounds⟩, execution, honest⟩

open NightstreamFPrime.Spec.Folding
open PiCCS.PaperJoint PiCCS.PaperJoint.StrongReduction
open PiCCS.PaperJoint.MatrixCoefficientSource

universe uCommitment uPublicInput uScalar uState

variable {Commitment : Type uCommitment} {PublicInput : Type uPublicInput}
  {Scalar : Type uScalar} {State : Type uState} {shape : Shape}
  {columns blockCount width : Nat}
  (key : PaperNonInteractive.Key K Commitment PublicInput Scalar State shape columns blockCount width)
  (running : PaperNonInteractive.Running K Commitment PublicInput shape)
  (fresh : PaperNonInteractive.Fresh Commitment PublicInput shape)
  (witness : OutputWitness shape columns)

private theorem source_table_truth
    (valid : SourceHolds key.extensionOps key.lift key.openingMaps key.params
      (key.statement running fresh) witness) :
    (TableResidualData.toTableObligations key.extensionOps
      (SignedCoefficientObject.toTableResidualData key.extensionOps
        (((key.statement running fresh).sourceProtocolData key.lift witness).toJointData
          key.extensionOps))).AllHold := by
  let unified := ((key.statement running fresh).sourceConnectedInputs witness).toUnifiedInputs key.baseOps
  have semantic := (unified.toIndependentInputs_semanticTruth_iff
    key.baseOps key.extensionOps key.lift).mpr valid.2
  have tables := (ConcreteJointData.jointTableTruth_iff_semanticTruth key.baseOps key.baseZero
    key.noZeroDivisors key.extensionOps key.extensionLaws key.lift
    key.liftLaws.toZeroReflectingLift unified.toIndependentInputs).mpr semantic
  rw [← ProtocolDataRefinement.toProtocolData_toJointData_eq
    key.baseOps key.extensionOps key.lift key.liftLaws unified] at tables
  simpa only [Statement.sourceProtocolData, unified] using tables

private theorem source_initial_true
    (valid : SourceHolds key.extensionOps key.lift key.openingMaps key.params
      (key.statement running fresh) witness)
    (alpha : CubePoint K shape.cubeVariables) (gamma : K) :
    ((key.statement running fresh).verifierInput key.lift).initial key.extensionOps gamma =
      FixedPhase.semanticInitial key.extensionOps.toOps
        (ProtocolPolynomial.polynomial key.extensionOps
          ((key.statement running fresh).sourceProtocolData key.lift witness) alpha gamma)
        shape.cubeVariables := by
  let data := (key.statement running fresh).sourceProtocolData key.lift witness
  have coefficients := (SignedCoefficientObject.coefficientTruth_iff_tableObligations
    key.extensionOps key.extensionZeroLaws (data.toJointData key.extensionOps)).mpr
      (source_table_truth key running fresh witness valid)
  have zero := (SignedCoefficientObject.evaluate_eq_zero_iff_coefficientTruth_or_mixingRoot
    key.extensionOps key.extensionLaws (data.toJointData key.extensionOps) alpha gamma).mpr
      (Or.inl coefficients)
  rw [← SignedCoefficientPolynomial.paperDifference_eq_evaluate
    key.extensionOps key.extensionLaws (data.toJointData key.extensionOps) alpha gamma] at zero
  have equality := (FiniteSumAlgebra.sub_eq_zero_iff key.extensionOps key.extensionLaws _ _).mp zero
  change ((key.statement running fresh).verifierInput key.lift).initial key.extensionOps gamma =
    HypercubeTruth.sumCompletions key.extensionOps.toOps
      (ProtocolPolynomial.polynomial key.extensionOps data alpha gamma) [] shape.cubeVariables
  rw [← (key.statement running fresh).sourceProtocolData_toVerifierInput key.lift witness,
    ProtocolPolynomial.verifierInput_initial_eq_joint_initial,
    ProtocolPolynomial.sumCompletions_polynomial_eq_summedQ key.extensionOps key.extensionLaws]
  exact equality

private theorem protocolRepresentable
    (alpha : CubePoint K shape.cubeVariables) (gamma : K) :
    FixedPhase.Sequential.RoundRepresentable key.extensionOps.toOps
      (ProtocolPolynomial.polynomial key.extensionOps
        ((key.statement running fresh).sourceProtocolData key.lift witness) alpha gamma)
      width shape.cubeVariables := by
  have degree : ((key.statement running fresh).sourceProtocolData key.lift witness).toVerifierInput.sumcheckDegreeBound =
      width := by
    rw [(key.statement running fresh).sourceProtocolData_toVerifierInput key.lift witness]
    exact key.statement_sumcheckDegreeBound_eq running fresh
  simpa only [degree] using ProtocolPolynomialDegree.sequentialRoundRepresentable
    key.extensionOps key.extensionLaws
    ((key.statement running fresh).sourceProtocolData key.lift witness) alpha gamma

private def honestOutput (point : CubePoint K shape.cubeVariables) :
    FullOutputCoordinates.FullOutput K shape :=
  FullOutputCoordinates.FullOutput.honestAt key.baseOps key.extensionOps key.lift
    ((key.statement running fresh).sourceConnectedInputs witness) point

private theorem honestOutput_projected (point : CubePoint K shape.cubeVariables) :
    (key.statement running fresh).projectOutput (honestOutput key running fresh witness point) =
      ProtocolPolynomial.messageAt key.extensionOps
        ((key.statement running fresh).sourceProtocolData key.lift witness) point := by
  rw [(key.statement running fresh).projectOutput_eq_toOutputMessage witness]
  exact FullOutputCoordinates.FullOutput.honestAt_toOutputMessage_eq_messageAt
    key.baseOps key.baseLaws key.baseZero key.extensionOps key.lift
    ((key.statement running fresh).sourceConnectedInputs witness) key.constantLaw point

private theorem honestOutput_strict
    (valid : SourceHolds key.extensionOps key.lift key.openingMaps key.params
      (key.statement running fresh) witness)
    (probe : Probe K shape)
    (outputEqual : probe.response.fullOutput = honestOutput key running fresh witness probe.coins.roundPoint) :
    ∀ source, CE.Holds key.piRlcSemantics key.params
      ((key.statement running fresh).publicOutput probe source) (witness.assignments source) := by
  intro source
  refine ⟨(key.openingAgreement key.params.b
    ((key.statement running fresh).commitments source)
    ((key.statement running fresh).publicInputs source) (witness.assignments source)).mp (valid.1 source),
    (key.evaluationAgreement (witness.assignments source) probe.coins.roundPoint).1, ?_⟩
  change key.piRlcSemantics.evaluations key.relationSource (witness.assignments source)
    probe.coins.roundPoint = #[{
      pad := probe.response.fullOutput.padCoordinate source
      matrix := probe.response.fullOutput.matrixCoordinate source }]
  unfold PaperNonInteractive.Key.relationSource
  rw [(key.evaluationAgreement (witness.assignments source) probe.coins.roundPoint).2, outputEqual]
  rfl

private noncomputable def honestPiCcsProver : CausalExecution.Prover shape columns width where
  rounds := fun alpha gamma => Classical.choose (exists_causal_honest_strategy
    key.extensionOps.toOps
    (ProtocolPolynomial.polynomial key.extensionOps
      ((key.statement running fresh).sourceProtocolData key.lift witness) alpha gamma)
    width shape.cubeVariables (protocolRepresentable key running fresh witness alpha gamma))
  output := fun _ _ coordinates =>
    if dimension : coordinates.length = shape.cubeVariables then
      some (honestOutput key running fresh witness ⟨coordinates, dimension⟩, witness)
    else none

/-- Lemma 10 at the exact NIFS interface: one causal honest prover succeeds
for every public challenge stream and returns all strict CE(b) output
openings in the existing weak-suffix semantics. The source-validity premise
is the honest-completeness input, not an extraction assumption. -/
theorem exists_honest_piCcs_prover
    (valid : SourceHolds key.extensionOps key.lift key.openingMaps key.params
      (key.statement running fresh) witness) :
    ∃ prover : CausalExecution.Prover shape columns width,
      ∀ (alpha : CubePoint K shape.cubeVariables) (gamma : K)
        (point : CubePoint K shape.cubeVariables),
        ∃ probe : Probe K shape,
          CausalExecution.run prover alpha gamma point = some (probe, witness) ∧
          probe.FixedWidthAccepted key.extensionOps key.lift (key.statement running fresh) width ∧
          (∀ source, CE.Holds key.piRlcSemantics key.params
            ((key.statement running fresh).publicOutput probe source) (witness.assignments source)) := by
  refine ⟨honestPiCcsProver key running fresh witness, ?_⟩
  intro alpha gamma point
  let data := (key.statement running fresh).sourceProtocolData key.lift witness
  let q := ProtocolPolynomial.polynomial key.extensionOps data alpha gamma
  obtain ⟨certificate, execution, honest⟩ := Classical.choose_spec
    (exists_causal_honest_strategy key.extensionOps.toOps q width shape.cubeVariables
      (protocolRepresentable key running fresh witness alpha gamma)) point.coordinates point.dimension
  let probe : Probe K shape := {
    coins := { alpha, gamma, roundPoint := point }
    response := {
      rounds := FixedPhase.RawCertificate.encode certificate
      fullOutput := honestOutput key running fresh witness point } }
  refine ⟨probe, ?_, ?_, honestOutput_strict key running fresh witness valid probe rfl⟩
  · have issuedRounds : issued
        ((honestPiCcsProver key running fresh witness).rounds alpha gamma) [] point.coordinates =
        some certificate.rounds := execution
    unfold CausalExecution.run
    rw [issuedRounds]
    simp only [honestPiCcsProver, dif_pos point.dimension, Option.map_some]
    rfl
  · change ProtocolPolynomial.FixedWidth.check key.extensionOps width
      ((key.statement running fresh).verifierInput key.lift) alpha gamma point
      ((key.statement running fresh).projectOutput (honestOutput key running fresh witness point))
      (FixedPhase.RawCertificate.encode certificate) = true
    apply (ProtocolPolynomial.FixedWidth.check_eq_true_iff key.extensionOps width
      ((key.statement running fresh).verifierInput key.lift) alpha gamma point
      ((key.statement running fresh).projectOutput (honestOutput key running fresh witness point))
      (FixedPhase.RawCertificate.encode certificate)).mpr
    refine ⟨certificate, FixedPhase.RawCertificate.decode_encode certificate, ?_⟩
    have terminalEqual : ProtocolPolynomial.terminalFromMessage key.extensionOps
        ((key.statement running fresh).verifierInput key.lift) alpha gamma point
        ((key.statement running fresh).projectOutput (honestOutput key running fresh witness point)) =
        q point.coordinates := by
      rw [honestOutput_projected key running fresh witness point,
        ← (key.statement running fresh).sourceProtocolData_toVerifierInput key.lift witness]
      change ProtocolPolynomial.qAtPoint key.extensionOps data alpha gamma point =
        ProtocolPolynomial.polynomial key.extensionOps data alpha gamma point.coordinates
      unfold ProtocolPolynomial.polynomial
      rw [dif_pos point.dimension]
    rw [terminalEqual]
    apply FixedPhase.complete key.extensionOps.toOps q _ point.coordinates certificate _ honest
    simpa only [point.dimension] using source_initial_true key running fresh witness valid alpha gamma

end NightstreamFPrime.Spec.Folding.Nifs.PaperStrongCompleteness
