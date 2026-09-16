import NightstreamFPrime.Export.Stage1.PiCCSInputCheck

/-!
Public statement and first-round coins for the selected PiCCS producer.
Old running evaluations are public claims. Prover rounds and final output
claims are absent from Input. Transcript and polynomial semantics stay with
their existing owners; this module supplies no witness or prover message.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.PiCCSPublicReplay

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ConcreteCarrier
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.UnifiedSources
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Layout.Stage1

private abbrev logicalWidth :=
  PerApplicationFixedPoint.logicalWidth Poseidon2HashChainV1Package.application

private abbrev publicFits :=
  PerApplicationFixedPoint.publicFits Poseidon2HashChainV1Package.application

/-- Only the original public statement. The prior running evaluation claims
are inputs; no round polynomial or newly claimed evaluation is carried. -/
structure Input where
  commitment : Vector F 1188
  publicInput : Vector F 270
  running : PiCCSInputCheck.RunningInput

/-- Decode only the three original public fields. Prover rounds and final
output evaluations are not fields of this input file. -/
def decode (value : Lean.Json) : Except String Input := do
  let fields ← value.getArr?
  match fields.toList with
  | [commitment, publicInput, running] =>
      return {
        commitment := ← PiCCSInputCheck.decodeVector 1188 PiCCSInputCheck.decodeField commitment
        publicInput := ← PiCCSInputCheck.decodeVector 270 PiCCSInputCheck.decodeField publicInput
        running := ← PiCCSInputCheck.decodeRunning running }
  | _ => throw "expected three public PiCCS fields"

def parse (text : String) : Except String Input :=
  Lean.Json.parse text >>= decode

/-- Compatibility projection for the existing comparison file. This reads
none of its rounds or its two top-level final evaluation arrays. -/
def fromCheckInput (input : PiCCSInputCheck.Input) : Input :=
  ⟨input.commitment, input.publicInput, input.running⟩

def running (input : Input) :
    Running (logicalWidth := logicalWidth) (publicFits := publicFits) :=
  PiCCSInputCheck.runningFromInput input.running

def fresh (input : Input) :
    Fresh (logicalWidth := logicalWidth) (publicFits := publicFits) where
  commitments := fun _ row coefficient => input.commitment.get
    ⟨row.val * ringDegree + coefficient.val, by
      have rowBound : row.val < 22 := row.isLt
      have coefficientBound : coefficient.val < 54 := coefficient.isLt
      change row.val * 54 + coefficient.val < 1188
      omega⟩
  publicInputs := fun _ => input.publicInput.get

def verifierInput (input : Input) :
    ProtocolPolynomial.VerifierInput K productionShape where
  constraintPolynomial :=
    ConstraintPolynomialLift.liftConstraintPolynomial K.embed
      Spec.ProductionRelation.polynomial
  priorPoint := (running input).point
  claimedPadCoefficient := fun coordinate =>
    ((running input).evaluations coordinate.running).pad coordinate.coefficient
  claimedMatrixCoefficient := fun coordinate =>
    ((running input).evaluations coordinate.running).matrix coordinate.matrix
      coordinate.coefficient

/-- Exact digest-only statement state. Running claims remain bound by the
existing lifecycle digest; this does not add another absorption. -/
def statementState (input : Input) : Transcript.State :=
  ProductionKey.absorbPublicInput
    (Transcript.absorb Transcript.initialState Transcript.piCcsDigestDomainTag)
    (running input) (fresh input)

def context (input : Input) :
    Folding.PiCCS.TranscriptReplay.Statement K Transcript.State productionShape :=
  ⟨statementState input, verifierInput input⟩

/-- Bind once before constructing the first polynomial. -/
def pre (input : Input) : FiatShamir.PreSumcheck K Transcript.State productionShape :=
  Folding.PiCCS.Transcript.deriveFromState Transcript.piCcsOracle.transcript
    (statementState input)

/-- Initial sum-check target from prior public evaluation claims. -/
def initialClaim (input : Input) (gamma : K) : K :=
  SumCheck.Finite.Message.evaluateCoefficients extensionOps.toOps gamma
    (Folding.PiCCS.FinalIdentity.targetCoefficientList (verifierInput input))

/-- The executable constructor retains the ten coefficients required by the
selected sparse CCS syntax. No concrete row plan is evaluated. -/
theorem degree_eq (input : Input) : (verifierInput input).sumcheckDegreeBound = 9 := by
  dsimp only [verifierInput, ProtocolPolynomial.VerifierInput.sumcheckDegreeBound]
  rw [ConstraintPolynomialLift.liftConstraintPolynomial_canonicalEqualityGatedDegreeBound]
  change Nat.max Spec.ProductionRelation.polynomial.canonicalEqualityGatedDegreeBound 4 = 9
  rw [Spec.ProductionRelation.polynomial_canonicalEqualityGatedDegreeBound]
  rfl

private def firstIndex : Fin productionShape.cubeVariables := ⟨0, by decide⟩

/-- Absorb the newly constructed first polynomial, then derive its challenge.
The caller supplies pre.state; no claimed challenge is accepted. -/
def firstRound (state : Transcript.State)
    (polynomial : SumCheck.Finite.FixedPolynomial K 9) : K × Transcript.State :=
  let absorbed := Transcript.piCcsOracle.transcript.absorbRound
    state firstIndex polynomial.toMessage
  Transcript.piCcsOracle.transcript.squeeze absorbed (.sumcheck firstIndex)

theorem running_fromCheckInput (input : PiCCSInputCheck.Input) :
    running (fromCheckInput input) = PiCCSInputCheck.running input := rfl

theorem fresh_fromCheckInput (input : PiCCSInputCheck.Input) :
    fresh (fromCheckInput input) = PiCCSInputCheck.fresh input := rfl

theorem verifierInput_fromCheckInput (input : PiCCSInputCheck.Input) :
    verifierInput (fromCheckInput input) = PiCCSInputCheck.verifierInput input := rfl

theorem statementState_eq_key (input : Input)
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : PaperAlgebra.AjtaiKey
      (logicalWidth := logicalWidth) (publicFits := publicFits)) :
    statementState input =
      (ProductionKey.key relation ajtai).publicInputState (running input) (fresh input) :=
  (ProductionKey.key_publicInputState_eq relation ajtai (running input) (fresh input)).symm

theorem verifierInput_eq_key (input : Input)
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : PaperAlgebra.AjtaiKey
      (logicalWidth := logicalWidth) (publicFits := publicFits)) :
    verifierInput input =
      ((ProductionKey.key relation ajtai).statement (running input) (fresh input)
        ).verifierInput K.embed := rfl

/-- Both the starting state and the oracle are the production key's values.
The entire pre-sumcheck result includes alpha, gamma and the resulting state. -/
theorem pre_eq_oracle (input : Input)
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : PaperAlgebra.AjtaiKey
      (logicalWidth := logicalWidth) (publicFits := publicFits)) :
    pre input = FiatShamir.derivePreSumcheck
      (ProductionKey.key relation ajtai).oracle.transcript (context input) := by
  rw [ProductionKey.key_oracle_eq]
  exact Folding.PiCCS.Transcript.deriveFromState_initialState
    Transcript.piCcsOracle.transcript (context input)

theorem pre_fromCheckInput (input : PiCCSInputCheck.Input) :
    pre (fromCheckInput input) = Folding.PiCCS.Transcript.deriveFromState
      Transcript.piCcsOracle.transcript
      (ProductionKey.absorbPublicInput
        (Transcript.absorb Transcript.initialState Transcript.piCcsDigestDomainTag)
        (PiCCSInputCheck.running input) (PiCCSInputCheck.fresh input)) := rfl

theorem initialClaim_value (input : Input) (gamma : K) :
    initialClaim input gamma = (verifierInput input).initial extensionOps gamma :=
  Folding.PiCCS.FinalIdentity.evaluateTargetCoefficients_eq_initial
    extensionOps extensionLaws (verifierInput input) gamma

theorem initialClaim_fromCheckInput (input : PiCCSInputCheck.Input) (gamma : K) :
    initialClaim (fromCheckInput input) gamma =
      PiCCSInputCheck.initialClaimFast input gamma := rfl

/-- One causal step equals the existing indexed round-replay operation. -/
theorem firstRound_eq_derive (state : Transcript.State)
    (polynomial : SumCheck.Finite.FixedPolynomial K 9) :
    ([(firstRound state polynomial).1], (firstRound state polynomial).2) =
      FiatShamir.deriveRoundsFrom Transcript.piCcsOracle.transcript
        (fun _ => polynomial.toMessage) state [firstIndex] := rfl

/-- Comparison bridge only: the checker's supplied first polynomial uses
this same absorption and challenge. It is not part of the public Input. -/
theorem firstRound_eq_checkTrace (input : PiCCSInputCheck.Input)
    (state : Transcript.State) :
    let polynomial := PiCCSProofInputs.roundPolynomial (PiCCSInputCheck.proofValues input) firstIndex
    ([(firstRound state polynomial).1], (firstRound state polynomial).2) =
      ((PiCCSInputCheck.traceFrom input state [firstIndex]).challenges,
        (PiCCSInputCheck.traceFrom input state [firstIndex]).state) := rfl

end NightstreamFPrime.Export.Stage1.PiCCSPublicReplay
