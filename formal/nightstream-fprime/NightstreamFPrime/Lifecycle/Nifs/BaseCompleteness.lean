import NightstreamFPrime.Lifecycle.ProductionKey
import NightstreamFPrime.Lifecycle.XOut
import NightstreamFPrime.Lifecycle.PiCCS.v1_1.ZeroRunningPolynomial
import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ProtocolPolynomialDegree.Support

/-!
Owns the canonical base dummy's deterministic PiCCS acceptance. The fresh
commitment is zero and its public input is the encoded prior hash. No zero
opening of that public input is asserted. PiRLC sampling and PiDEC acceptance
remain separate from this first constructor step.
-/

namespace NightstreamFPrime.Lifecycle.Nifs.BaseCompleteness

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Lifecycle.PaperAlgebra
open ConcreteCarrier

/-- Fixed-width zero C messages and zero D commitment/evaluation messages.
The verifier still derives every challenge and child public input. -/
def zeroProof : Proof 9 where
  piCcsRounds := fun _ => SumCheck.Finite.FixedPolynomial.zero extensionOps.toOps 9
  piCcsOutput := {
    padCoordinate := fun _ _ => K.zero
    matrixCoordinate := fun _ _ _ => K.zero }
  piDecCommitments := fun _ _ _ => 0
  piDecEvaluations := fun _ => {
    pad := fun _ => K.zero
    matrix := fun _ _ => K.zero }

/-- The base dummy fresh claim uses the actual prior-state hash as its public
input. This constructor supplies a public claim, not an opening witness. -/
def baseFresh
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (prior : HashPreimage (logicalWidth := logicalWidth) (publicFits := publicFits)) :
    Fresh (logicalWidth := logicalWidth) (publicFits := publicFits) where
  commitments := fun _ _ _ => 0
  publicInputs := fun _ => encHash (stateHash prior)

private theorem sumMap_zero_of_terms {Index : Type}
    (indices : List Index) (term : Index → K)
    (zero : ∀ index, term index = K.zero) :
    SignedJointIdentity.sumMap extensionOps indices term = K.zero := by
  exact (FiniteSumAlgebra.sumMap_congr extensionOps indices term
    (fun _ => K.zero) (fun index _ => zero index)).trans
      (FiniteSumAlgebra.sumMap_zero extensionOps extensionLaws indices)

private theorem extension_mul_zero (value : K) :
    extensionOps.mul value K.zero = K.zero := extensionLaws.mul_zero value

private theorem extension_zero_add (value : K) :
    extensionOps.add K.zero value = value := extensionLaws.zero_add value

private theorem liftedPolynomial_zero :
    CCSResidualTable.evaluatePolynomial extensionOps
      (ConstraintPolynomialLift.liftConstraintPolynomial K.embed
        ProductionRelation.polynomial) (fun _ => K.zero) = K.zero := by
  have lifted := ProtocolDataRefinement.evaluatePolynomial_lift
    baseOps extensionOps K.embed protocolLift ProductionRelation.polynomial
      (fun _ => (0 : F))
  simpa only [ProductionRelation.polynomial_zeroImages, embed_zero] using lifted

private theorem zero_chain (challenges : List K) :
    SumCheck.Finite.FixedPhase.Chain extensionOps.toOps K.zero
      (List.replicate challenges.length
        (SumCheck.Finite.FixedPolynomial.zero extensionOps.toOps 9))
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

variable {logicalWidth : Nat}
  {publicFits : ringDegree * publicRingColumns ≤
    Phi81CarrierLayout.carrierWidth logicalWidth}
  (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
  (ajtai : AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
  (prior : HashPreimage (logicalWidth := logicalWidth) (publicFits := publicFits))

private theorem initial_zero (gamma : K) :
    (((ProductionKey.key relation ajtai).statement defaultRunning (baseFresh prior)).verifierInput
      K.embed).initial extensionOps gamma = K.zero := by
  let statement := (ProductionKey.key relation ajtai).statement defaultRunning (baseFresh prior)
  -- The reused initial-claim lemma ignores these auxiliary assignment values;
  -- it uses only public zero coefficients and assumes no opening validity.
  exact PiCCS.v1_1.ZeroRunningPolynomial.initial_zero
    (statement.sourceConnectedInputs ⟨fun _ _ => 0⟩)
    (fun _ => rfl) (fun _ => rfl) gamma

private theorem terminal_zero
    (alpha : CubePoint K productionShape.cubeVariables) (gamma : K)
    (point : CubePoint K productionShape.cubeVariables) :
    ProtocolPolynomial.terminalFromMessage extensionOps
      (((ProductionKey.key relation ajtai).statement defaultRunning (baseFresh prior)).verifierInput
        K.embed) alpha gamma point
      ((ProductionKey.key relation ajtai).piCcsCertificate defaultRunning
        (baseFresh prior) zeroProof).output = K.zero := by
  let input := ((ProductionKey.key relation ajtai).statement defaultRunning
    (baseFresh prior)).verifierInput K.embed
  let message := ((ProductionKey.key relation ajtai).piCcsCertificate defaultRunning
    (baseFresh prior) zeroProof).output
  have ccs : ProtocolPolynomial.ccsAtMessage extensionOps input gamma message = K.zero := by
    apply sumMap_zero_of_terms
    intro source
    change extensionOps.mul _
      (CCSResidualTable.evaluatePolynomial extensionOps
        (ConstraintPolynomialLift.liftConstraintPolynomial K.embed
          ProductionRelation.polynomial) (fun _ => K.zero)) = K.zero
    rw [liftedPolynomial_zero]
    exact extensionLaws.mul_zero _
  have norm : ProtocolPolynomial.normAtMessage extensionOps gamma message = K.zero := by
    apply sumMap_zero_of_terms
    intro source
    have residual : ProtocolPolynomial.strictNormResidual extensionOps K.zero = K.zero := by
      change extensionOps.mul
        (extensionOps.mul (extensionOps.add K.zero extensionOps.one) K.zero)
        (extensionOps.sub K.zero extensionOps.one) = K.zero
      rw [extension_mul_zero, extensionLaws.mul_comm, extension_mul_zero]
    change extensionOps.mul _
      (ProtocolPolynomial.strictNormResidual extensionOps K.zero) = K.zero
    rw [residual]
    exact extensionLaws.mul_zero _
  have pad : ProtocolPolynomial.padAtMessage extensionOps input gamma point message = K.zero := by
    unfold ProtocolPolynomial.padAtMessage
    have sum : SignedJointIdentity.sumMap extensionOps (canonicalPadCoordinates productionShape)
        (fun coordinate => SignedJointIdentity.gammaTerm extensionOps gamma
          coordinate.localGammaExponent (message.padImage coordinate)) = K.zero := by
      apply sumMap_zero_of_terms
      intro coordinate
      exact extensionLaws.mul_zero _
    rw [sum]
    exact extensionLaws.mul_zero _
  have matrix : ProtocolPolynomial.matrixAtMessage extensionOps input gamma point message = K.zero := by
    unfold ProtocolPolynomial.matrixAtMessage
    have sum : SignedJointIdentity.sumMap extensionOps (canonicalMatrixCoordinates productionShape)
        (fun coordinate => SignedJointIdentity.gammaTerm extensionOps gamma
          coordinate.localGammaExponent (message.matrixImage coordinate)) = K.zero := by
      apply sumMap_zero_of_terms
      intro coordinate
      exact extensionLaws.mul_zero _
    rw [sum]
    exact extensionLaws.mul_zero _
  change ProtocolPolynomial.terminalFromMessage extensionOps input alpha gamma point message = K.zero
  simp only [ProtocolPolynomial.terminalFromMessage, pad, matrix, ccs, norm,
    SignedJointIdentity.gammaTerm, extension_mul_zero, extension_zero_add]

/-- The canonical base dummy passes the actual PiCCS verifier for its derived
transcript coins. The proof is independent of the coin values and supplies no
fresh or child opening, sampler-success, or PiDEC-acceptance premise. -/
theorem zeroProof_piCcsCheck :
    Nifs.PaperNonInteractive.piCcsCheck (ProductionKey.key relation ajtai)
      defaultRunning (baseFresh prior) zeroProof = true := by
  apply (Nifs.PaperNonInteractive.piCcsCheck_eq_true_iff
    (ProductionKey.key relation ajtai) defaultRunning (baseFresh prior) zeroProof).mpr
  let execution := (ProductionKey.key relation ajtai).piCcsExecution
    defaultRunning (baseFresh prior) zeroProof
  have keyOps : (ProductionKey.key relation ajtai).extensionOps = extensionOps := rfl
  have keyLift : (ProductionKey.key relation ajtai).lift = K.embed := rfl
  rw [keyOps, keyLift]
  rw [initial_zero relation ajtai prior,
    terminal_zero relation ajtai prior]
  change SumCheck.Finite.FixedPhase.Chain extensionOps.toOps K.zero
    (List.ofFn (fun _ : Fin productionShape.cubeVariables =>
      SumCheck.Finite.FixedPolynomial.zero extensionOps.toOps 9))
    execution.coins.roundPoint.coordinates K.zero
  have roundsEqual :
      List.ofFn (fun _ : Fin productionShape.cubeVariables =>
        SumCheck.Finite.FixedPolynomial.zero extensionOps.toOps 9) =
      List.replicate execution.coins.roundPoint.coordinates.length
        (SumCheck.Finite.FixedPolynomial.zero extensionOps.toOps 9) := by
    rw [List.ofFn_const, execution.coins.roundPoint.dimension]
  rw [roundsEqual]
  exact zero_chain execution.coins.roundPoint.coordinates

end NightstreamFPrime.Lifecycle.Nifs.BaseCompleteness
