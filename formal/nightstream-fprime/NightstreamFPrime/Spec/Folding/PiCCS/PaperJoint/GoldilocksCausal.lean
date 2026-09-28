import NightstreamFPrime.Spec.SumCheck.GoldilocksCausalTrace
import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.StrongReduction
import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ConcreteCarrier
import NightstreamFPrime.Spec.GoldilocksPrime

/-!
The paper's exact fixed-width PiCCS failure event on a causal message path.
Source assignments are fixed before the independent second execution. After
alpha and gamma are drawn, its semantic q is fixed before all round challenges.
An output witness can depend on the complete second execution; agreement with
the first witness is an event, not a condition on the challenge distribution.
-/

namespace NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.GoldilocksCausal

open NightstreamFPrime.Spec
open SumCheck.Finite
open ConcreteCarrier StrongReduction
open _root_.NightstreamFPrime.Spec.SumCheck.Finite.GoldilocksCausal (Strategy Collision)
open _root_.NightstreamFPrime.Spec.SumCheck.Finite.GoldilocksCausalTrace (issued)

universe uCommitment uPublicInput

/-- The raw certificate is exactly the one issued by the causal strategy.
The decoder receipt prevents a different certificate from replacing it. -/
def IssuedProbe {shape : Shape} {width : Nat} (strategy : Strategy width)
    (probe : Probe K shape) : Prop :=
  ∃ certificate : FixedPhase.Certificate K width,
    FixedPhase.RawCertificate.decode width probe.response.rounds = some certificate ∧
      issued strategy [] probe.coins.roundPoint.coordinates = some certificate.rounds

/-- The paper polynomial has the verifier's selected width at every prefix.
The representation comes from its concrete CCS, norm, and evaluation terms. -/
theorem sequentialRoundRepresentable {shape : Shape}
    (data : ProtocolPolynomial.Data K shape) (alpha : CubePoint K shape.cubeVariables)
    (gamma : K) (width : Nat)
    (degreeCovers : data.toVerifierInput.sumcheckDegreeBound ≤ width) :
    FixedPhase.Sequential.RoundRepresentable GoldilocksRoots.ops
      (ProtocolPolynomial.polynomial extensionOps data alpha gamma)
      width shape.cubeVariables := by
  intro fixed remaining length
  obtain ⟨polynomial, represents⟩ :=
    ProtocolPolynomialDegree.sequentialRoundRepresentable extensionOps extensionLaws
      data alpha gamma fixed remaining length
  refine ⟨FixedPolynomial.widen extensionOps.toOps degreeCovers polynomial, ?_⟩
  intro point
  change (FixedPolynomial.widen extensionOps.toOps degreeCovers polynomial).evaluate
      extensionOps.toOps point = _
  rw [FixedPolynomial.evaluate_widen extensionOps.toOps
    (ProtocolPolynomialDegree.Support.polynomialLaws extensionLaws)]
  exact represents point

/-- The existing strong-reduction SumCheck failure is on the issued path. -/
theorem failure_implies_collision
    {Commitment : Type uCommitment} {PublicInput : Type uPublicInput}
    {shape : Shape} {columns blockCount width : Nat}
    (statement : Statement K Commitment PublicInput shape columns blockCount baseOps)
    (witness : OutputWitness shape columns) (strategy : Strategy width)
    (probe : Probe K shape) (execution : IssuedProbe strategy probe)
    (failure : FixedWidthSumCheckFailure extensionOps K.embed statement width
      (goldilocksModulus ^ 2) probe witness) :
    Collision (ProtocolPolynomial.polynomial extensionOps
      (statement.sourceProtocolData K.embed witness) probe.coins.alpha probe.coins.gamma)
      strategy [] probe.coins.roundPoint.coordinates := by
  obtain ⟨actual, decodedActual, execution⟩ := execution
  obtain ⟨failed, decodedFailed, bad⟩ := failure
  have same : actual = failed := Option.some.inj (decodedActual.symm.trans decodedFailed)
  subst failed
  exact _root_.NightstreamFPrime.Spec.SumCheck.Finite.GoldilocksCausalTrace.badChallenge_implies_collision
    (ProtocolPolynomial.polynomial extensionOps (statement.sourceProtocolData K.embed witness)
      probe.coins.alpha probe.coins.gamma)
    strategy
    ((statement.sourceProtocolData K.embed witness).toVerifierInput.initial
      extensionOps probe.coins.gamma)
    probe.coins.roundPoint.coordinates actual execution bad

/-- Fix the first witness, then inspect an accepted second execution.
On witness agreement it gives the exact mixing or causal SumCheck event. -/
theorem agreed_accepted_implies_mixing_or_collision
    {Commitment : Type uCommitment} {PublicInput : Type uPublicInput}
    {shape : Shape} {columns blockCount width : Nat}
    (openingMaps : OpeningMaps Commitment PublicInput columns) (params : GlobalParams)
    (freshBound : params.b = 2)
    (statement : Statement K Commitment PublicInput shape columns blockCount baseOps)
    (constantLaw : MatrixCoefficientSource.ConstantTermLaw baseOps statement.matrixSource.kernel)
    (degreeCovers : (statement.verifierInput K.embed).sumcheckDegreeBound ≤ width)
    (firstWitness : OutputWitness shape columns)
    (sourceInvalid : ¬ SourceHolds extensionOps K.embed openingMaps params statement firstWitness)
    (strategy : Strategy width) (probe : Probe K shape)
    (secondWitness : OutputWitness shape columns) (agreement : secondWitness = firstWitness)
    (execution : IssuedProbe strategy probe)
    (ambient : AmbientOutputHolds extensionOps K.embed openingMaps params statement probe secondWitness)
    (accepted : probe.FixedWidthAccepted extensionOps K.embed statement width) :
    SignedCoefficientObject.MixingRoot extensionOps
      ((statement.sourceProtocolData K.embed firstWitness).toJointData extensionOps)
      probe.coins.alpha probe.coins.gamma ∨
    Collision (ProtocolPolynomial.polynomial extensionOps
      (statement.sourceProtocolData K.embed firstWitness) probe.coins.alpha probe.coins.gamma)
      strategy [] probe.coins.roundPoint.coordinates := by
  subst secondWitness
  rcases fixedWidthAcceptedProbe_extracts_source_or_badEvent
      baseLaws baseZeroAgreement GoldilocksPrime.baseFieldNoZeroDivisors
      extensionOps extensionLaws extensionZeroLaws K.embed protocolLift openingMaps params
      freshBound statement constantLaw width degreeCovers (goldilocksModulus ^ 2)
      probe firstWitness ambient accepted with source | mixing | failure
  · exact False.elim (sourceInvalid source)
  · exact Or.inl mixing
  · exact Or.inr (failure_implies_collision statement firstWitness strategy probe execution failure)

end NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.GoldilocksCausal
