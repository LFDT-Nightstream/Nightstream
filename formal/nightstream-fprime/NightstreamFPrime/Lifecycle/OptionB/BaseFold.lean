import NightstreamFPrime.Lifecycle.OptionB.RandomOracle.Fidelity
import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ConcreteCarrier

/-! The base fold of option B (CW-10.4), HyperNova's zero proof on the light key. At iteration 0
the light step circuit still verifies a light fold. The prover folds the default fresh instance,
with zero commitment and public input `encHash` of the prior state's hash, into the default slot
with the zero proof: zero sum-check rounds and zero outputs (`zeroProof`, `baseFresh`), as Rust's
`words::base_step` does. The default slot claims zero evaluations, so the light Π_CCS initial
claim is zero (`initial_zero`), the terminal value of zero outputs is zero (`terminal_zero`), and
the zero rounds chain from zero to zero. So the light fold accepts the zero proof
(`zero_folds`). -/

set_option autoImplicit false

namespace NightstreamFPrime.Lifecycle.OptionB.BaseFold

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ConcreteCarrier
open NightstreamFPrime.Spec.HyperNova.Construction2
open NightstreamFPrime.Spec.HyperNova.Construction2.OptionBFolds
  (toLightRunning fromLight freshToLight)
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Lifecycle.OptionB
open NightstreamFPrime.Lifecycle.OptionB.RandomOracle

/-- HyperNova's zero proof on the light key: zero rounds and zero outputs. -/
def zeroProof : Light.Proof LightShape 8 where
  piCcsRounds := fun _ => SumCheck.Finite.FixedPolynomial.zero extensionOps.toOps 8
  piCcsOutput := {
    padCoordinate := fun _ _ => K.zero
    matrixCoordinate := fun _ _ _ => K.zero }

variable {logicalWidth : Nat}
  {publicFits : ringDegree * publicRingColumns ≤ Phi81CarrierLayout.carrierWidth logicalWidth}
  (ajtai : AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))

/-- The default fresh instance of the base step: zero commitment and public input `encHash
digest`. -/
def baseFresh (digest : Digest) : StepFresh logicalWidth publicFits where
  commitments := fun _ _ _ => 0
  publicInputs := fun _ => encHash (publicFits := publicFits) digest

private theorem foldr_map_zero {Index : Type}
    (indices : List Index) (term : Index → K)
    (zero : ∀ index, term index = K.zero) :
    (indices.map term).foldr extensionOps.add K.zero = K.zero := by
  induction indices with
  | nil => rfl
  | cons index indices inductionHypothesis =>
      rw [List.map_cons, List.foldr_cons, zero index, inductionHypothesis]
      exact extensionLaws.zero_add K.zero

/-- `PiCCS.v1_2.ZeroRunningPolynomial.initial_zero` for any shape: zero claimed coefficients give
a zero initial claim. -/
theorem initial_zero {shape : Shape} {columns blockCount : Nat}
    (data : MatrixCoefficientSource.ConnectedInputs K shape columns blockCount)
    (padZero : ∀ coordinate, data.claimedPadCoefficient coordinate = K.zero)
    (matrixZero : ∀ coordinate, data.claimedMatrixCoefficient coordinate = K.zero)
    (gamma : K) :
    (ProtocolDataRefinement.toProtocolData baseOps K.embed
      (data.toUnifiedInputs baseOps)).toVerifierInput.initial extensionOps gamma =
      K.zero := by
  let coefficients : TargetPolynomial.TargetCoefficients K shape :=
    { pad := data.claimedPadCoefficient, matrix := data.claimedMatrixCoefficient }
  have pad : TargetPolynomial.evaluatePad extensionOps.toOps coefficients gamma =
      K.zero := by
    rw [TargetPolynomial.evaluatePad_eq_foldr]
    apply foldr_map_zero
    intro coordinate
    change extensionOps.mul _ (data.claimedPadCoefficient coordinate) = K.zero
    rw [padZero coordinate]
    exact extensionLaws.mul_zero _
  have matrix : TargetPolynomial.evaluateMatrix extensionOps.toOps coefficients gamma =
      K.zero := by
    change ((canonicalMatrixCoordinates shape).map
      (TargetPolynomial.matrixTerm extensionOps.toOps coefficients gamma)).foldr
        extensionOps.add K.zero = K.zero
    apply foldr_map_zero
    intro coordinate
    change extensionOps.mul _ (data.claimedMatrixCoefficient coordinate) = K.zero
    rw [matrixZero coordinate]
    exact extensionLaws.mul_zero _
  change extensionOps.add
    (TargetPolynomial.evaluatePad extensionOps.toOps coefficients gamma)
    (TargetPolynomial.evaluateMatrix extensionOps.toOps coefficients gamma) = K.zero
  rw [pad, matrix]
  exact extensionLaws.zero_add K.zero

private theorem sumMap_zero_of_terms {Index : Type}
    (indices : List Index) (term : Index → K)
    (zero : ∀ index, term index = K.zero) :
    SignedJointIdentity.sumMap extensionOps indices term = K.zero :=
  (FiniteSumAlgebra.sumMap_congr extensionOps indices term
    (fun _ => K.zero) (fun index _ => zero index)).trans
      (FiniteSumAlgebra.sumMap_zero extensionOps extensionLaws indices)

private theorem liftedPolynomial_zero :
    CCSResidualTable.evaluatePolynomial extensionOps
      (ConstraintPolynomialLift.liftConstraintPolynomial K.embed
        ProductionRelation.polynomial) (fun _ => K.zero) = K.zero := by
  have lifted := ProtocolDataRefinement.evaluatePolynomial_lift
    baseOps extensionOps K.embed protocolLift ProductionRelation.polynomial
      (fun _ => (0 : F))
  simpa only [ProductionRelation.polynomial_zeroImages, embed_zero] using! lifted

private theorem extension_mul_zero (value : K) :
    extensionOps.mul value K.zero = K.zero := extensionLaws.mul_zero value

private theorem extension_zero_add (value : K) :
    extensionOps.add K.zero value = value := extensionLaws.zero_add value

private theorem strictNormResidual_zero :
    ProtocolPolynomial.strictNormResidual extensionOps K.zero = K.zero := by
  change extensionOps.mul
    (extensionOps.mul (extensionOps.add extensionOps.zero extensionOps.one)
      extensionOps.zero)
    (extensionOps.sub extensionOps.zero extensionOps.one) = extensionOps.zero
  rw [extensionLaws.mul_zero, extensionLaws.mul_comm, extensionLaws.mul_zero]

private theorem zero_chain (challenges : List K) :
    SumCheck.Finite.FixedPhase.Chain extensionOps.toOps K.zero
      (List.replicate challenges.length
        (SumCheck.Finite.FixedPolynomial.zero extensionOps.toOps 8))
      challenges K.zero := by
  induction challenges with
  | nil => rfl
  | cons challenge challenges previous =>
      simp only [List.length_cons, List.replicate_succ,
        SumCheck.Finite.FixedPhase.Chain,
        SumCheck.Finite.FixedPolynomial.evaluate_zero extensionOps.toOps
          (ProtocolPolynomialDegree.Support.polynomialLaws extensionLaws),
        extensionLaws.zero_add]
      exact ⟨rfl, previous⟩

theorem check_zero (key : CircuitKey logicalWidth publicFits) (digest : Digest) :
    Light.PiCCS.check (keys key.relation ajtai).light.extensionOps
      (keys key.relation ajtai).light.lift
      ((keys key.relation ajtai).light.statement (LightFork.runningOf defaultSlot)
        (freshToLight (baseFresh digest)))
      (ProductionKey.degreeBound key.relation) ((keys key.relation ajtai).light.probe (LightFork.runningOf defaultSlot)
        (freshToLight (baseFresh digest)) zeroProof) = true := by
  apply (ProtocolPolynomial.FixedWidth.check_eq_true_iff _ _ _ _ _ _ _ _).mpr
  refine ⟨{ rounds := List.ofFn zeroProof.piCcsRounds },
    SumCheck.Finite.FixedPhase.RawCertificate.decode_encode _, ?_⟩
  let statement := (keys key.relation ajtai).light.statement (LightFork.runningOf defaultSlot)
    (freshToLight (baseFresh digest))
  have initial : ∀ gamma, (statement.verifierInput K.embed).initial extensionOps gamma = K.zero :=
    fun gamma => initial_zero (statement.sourceConnectedInputs ⟨fun _ _ => 0⟩)
      (fun _ => rfl) (fun _ => rfl) gamma
  let probe := (keys key.relation ajtai).light.probe (LightFork.runningOf defaultSlot)
    (freshToLight (baseFresh digest)) zeroProof
  let input := statement.verifierInput K.embed
  let message := Light.PiCCS.maskMessage extensionOps (statement.projectOutput zeroProof.piCcsOutput)
  have terminal : ProtocolPolynomial.terminalFromMessage extensionOps input probe.coins.alpha
      probe.coins.gamma probe.coins.roundPoint message = K.zero := by
    have ccs : ProtocolPolynomial.ccsAtMessage extensionOps input probe.coins.gamma message =
        K.zero := by
      apply sumMap_zero_of_terms
      intro source
      change extensionOps.mul _
        (CCSResidualTable.evaluatePolynomial extensionOps
          (ConstraintPolynomialLift.liftConstraintPolynomial K.embed
            ProductionRelation.polynomial) (fun _ => K.zero)) = K.zero
      rw [liftedPolynomial_zero]
      exact extensionLaws.mul_zero _
    have norm : ProtocolPolynomial.normAtMessage extensionOps probe.coins.gamma message =
        K.zero := by
      apply sumMap_zero_of_terms
      intro source
      change extensionOps.mul _ (ProtocolPolynomial.strictNormResidual extensionOps
        (if source.val < LightShape.freshCount then K.zero else extensionOps.zero)) = K.zero
      have zero : (if source.val < LightShape.freshCount then K.zero else extensionOps.zero) =
          K.zero := by
        split <;> rfl
      rw [zero, strictNormResidual_zero]
      exact extensionLaws.mul_zero _
    have pad : ProtocolPolynomial.padAtMessage extensionOps input probe.coins.gamma
        probe.coins.roundPoint message = K.zero := by
      unfold ProtocolPolynomial.padAtMessage
      have sum : SignedJointIdentity.sumMap extensionOps (canonicalPadCoordinates LightShape)
          (fun coordinate => SignedJointIdentity.gammaTerm extensionOps probe.coins.gamma
            coordinate.localGammaExponent (message.padImage coordinate)) = K.zero := by
        apply sumMap_zero_of_terms
        intro coordinate
        exact extensionLaws.mul_zero _
      rw [sum]
      exact extensionLaws.mul_zero _
    have matrix : ProtocolPolynomial.matrixAtMessage extensionOps input probe.coins.gamma
        probe.coins.roundPoint message = K.zero := by
      unfold ProtocolPolynomial.matrixAtMessage
      have sum : SignedJointIdentity.sumMap extensionOps (canonicalMatrixCoordinates LightShape)
          (fun coordinate => SignedJointIdentity.gammaTerm extensionOps probe.coins.gamma
            coordinate.localGammaExponent (message.matrixImage coordinate)) = K.zero := by
        apply sumMap_zero_of_terms
        intro coordinate
        exact extensionLaws.mul_zero _
      rw [sum]
      exact extensionLaws.mul_zero _
    simp only [ProtocolPolynomial.terminalFromMessage, pad, matrix, ccs, norm,
      SignedJointIdentity.gammaTerm, extension_mul_zero, extension_zero_add]
  have roundsEqual : List.ofFn zeroProof.piCcsRounds = List.replicate
      probe.coins.roundPoint.coordinates.length
        (SumCheck.Finite.FixedPolynomial.zero extensionOps.toOps 8) := by
    rw [probe.coins.roundPoint.dimension]
    exact List.ofFn_const _ _
  change SumCheck.Finite.FixedPhase.Chain extensionOps.toOps
    (input.initial extensionOps probe.coins.gamma) (List.ofFn zeroProof.piCcsRounds)
    probe.coins.roundPoint.coordinates
    (ProtocolPolynomial.terminalFromMessage extensionOps input probe.coins.alpha
      probe.coins.gamma probe.coins.roundPoint message)
  rw [initial, terminal, roundsEqual]
  exact zero_chain _

/-- The light fold accepts the zero proof of the default fresh instance into the default slot,
and its output is a slot claim. -/
theorem zero_folds (key : CircuitKey logicalWidth publicFits) (digest : Digest) :
    ∃ folded : SlotClaim logicalWidth publicFits,
      OptionBFolds.folds .light (keys key.relation ajtai)
          (toClaim (key.system ajtai) defaultSlot) (baseFresh digest) (.light zeroProof) =
        some (toClaim (key.system ajtai) folded) := by
  let oracle := (Fidelity.deployedOracle (logicalWidth := logicalWidth)
    (publicFits := publicFits)).1
  have deployed : Fidelity.DeployedOn oracle :=
    (Fidelity.deployedOracle_deployed (logicalWidth := logicalWidth) (publicFits := publicFits)).1
  let running := LightFork.runningOf (defaultSlot (logicalWidth := logicalWidth)
    (publicFits := publicFits))
  let fresh := freshToLight (baseFresh (logicalWidth := logicalWidth) (publicFits := publicFits)
    digest)
  have sealed := Fidelity.light_challenge_eq key.relation ajtai running fresh zeroProof deployed
  let result := (keys key.relation ajtai).light.outputFor running fresh zeroProof
    (LightFork.challenge oracle fresh zeroProof)
  have verified : Light.verify (keys key.relation ajtai).light running fresh zeroProof =
      some result := by
    unfold Light.verify
    rw [ite_eq_left (check_zero ajtai key digest), sealed]
    rfl
  have romVerified : LightFork.verify key.relation ajtai oracle running fresh zeroProof =
      some result :=
    (Fidelity.light_verify_eq key.relation ajtai running fresh zeroProof deployed).trans verified
  have resultEq : result = LightFork.outputFor key.relation ajtai oracle running fresh zeroProof
      (LightFork.challenge oracle fresh zeroProof) := by
    unfold LightFork.verify at romVerified
    split at romVerified
    · exact (Option.some.inj romVerified).symm
    · cases romVerified
  have size : (fromLight result).evaluations.size = 1 := by
    show (result.evaluations.map OptionBFolds.evaluationsFromLight).size = 1
    rw [Array.size_map]
    exact (lightExtraction ajtai).combineEvaluations_size _ _ 1 (by decide) fun source => by
      unfold Light.Rlc.pair
      split <;> rfl
  obtain ⟨folded, read⟩ : ∃ folded, fromClaim (fromLight result) = some folded := by
    unfold fromClaim
    rw [dite_eq_left size]
    exact ⟨_, rfl⟩
  have claimEq : fromLight result = toClaim (key.system ajtai) folded := by
    rw [resultEq] at read ⊢
    exact toClaim_of_fromClaim rfl rfl read
  refine ⟨folded, ?_⟩
  show ((toLightRunning (toClaim (key.system ajtai) defaultSlot)).bind fun running =>
      (Light.verify (keys key.relation ajtai).light running fresh zeroProof).map fromLight) = _
  rw [LightFork.toLightRunning_toClaim, Option.bind_some, verified, Option.map_some, claimEq]

end NightstreamFPrime.Lifecycle.OptionB.BaseFold
