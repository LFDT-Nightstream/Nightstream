import NightstreamFPrime.Spec.Folding.Nifs.PaperNonInteractive.Types

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Folding.Nifs.PaperNonInteractive

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint

universe uExtension uCommitment uPublicInput uScalar uState

def piCcsCheck
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    [DecidableEq Extension]
    {shape : Shape}
    {columns blockCount degreeBound : Nat}
    (key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound)
    (running : Running Extension Commitment PublicInput shape)
    (fresh : Fresh Commitment PublicInput shape)
    (proof : Proof Extension Commitment shape degreeBound) : Bool :=
  let input := (key.statement running fresh).verifierInput key.lift
  let execution := key.piCcsExecution running fresh proof
  SumCheck.Finite.FixedPhase.checkChain key.extensionOps.toOps
    (input.initial key.extensionOps execution.coins.gamma)
    (key.piCcsFixedCertificate running fresh proof).rounds
    execution.coins.roundPoint.coordinates
    (ProtocolPolynomial.terminalFromMessage key.extensionOps input
      execution.coins.alpha execution.coins.gamma execution.coins.roundPoint
      (key.piCcsCertificate running fresh proof).output)

def piDecCheck
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    {shape : Shape}
    {columns blockCount degreeBound : Nat}
    (key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound)
    (running : Running Extension Commitment PublicInput shape)
    (fresh : Fresh Commitment PublicInput shape)
    (proof : Proof Extension Commitment shape degreeBound) : Bool :=
  match key.piDecAttempt running fresh proof with
  | none => false
  | some attempt =>
      letI := key.piDecDecision attempt
      decide (PiDEC.PaperVerifier.Accepted key.piDecAlgebra
        key.piDecPublicInputSplit key.piDecEvaluationArity attempt)

def verify
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    [DecidableEq Extension]
    {shape : Shape}
    {columns blockCount degreeBound : Nat}
    (key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound)
    (running : Running Extension Commitment PublicInput shape)
    (fresh : Fresh Commitment PublicInput shape)
    (proof : Proof Extension Commitment shape degreeBound) :
    Option (Running Extension Commitment PublicInput shape) :=
  if piCcsCheck key running fresh proof && piDecCheck key running fresh proof then
    key.output running fresh proof
  else
    none

@[simp] theorem piDecCheck_eq_true_iff
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    {shape : Shape}
    {columns blockCount degreeBound : Nat}
    (key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound)
    (running : Running Extension Commitment PublicInput shape)
    (fresh : Fresh Commitment PublicInput shape)
    (proof : Proof Extension Commitment shape degreeBound) :
    piDecCheck key running fresh proof = true <->
      exists attempt,
        key.piDecAttempt running fresh proof = some attempt /\
          PiDEC.PaperVerifier.Accepted key.piDecAlgebra
            key.piDecPublicInputSplit key.piDecEvaluationArity attempt := by
  unfold piDecCheck
  split
  · simp_all
  · rename_i attempt attemptEq
    letI := key.piDecDecision attempt
    simp [attemptEq]

@[simp] theorem piCcsCheck_eq_true_iff
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    [DecidableEq Extension]
    {shape : Shape}
    {columns blockCount degreeBound : Nat}
    (key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound)
    (running : Running Extension Commitment PublicInput shape)
    (fresh : Fresh Commitment PublicInput shape)
    (proof : Proof Extension Commitment shape degreeBound) :
    piCcsCheck key running fresh proof = true <->
      SumCheck.Finite.FixedPhase.Chain key.extensionOps.toOps
        (((key.statement running fresh).verifierInput key.lift).initial
          key.extensionOps
          (key.piCcsExecution running fresh proof).coins.gamma)
        (key.piCcsFixedCertificate running fresh proof).rounds
        (key.piCcsExecution running fresh proof).coins.roundPoint.coordinates
        (ProtocolPolynomial.terminalFromMessage key.extensionOps
          ((key.statement running fresh).verifierInput key.lift)
          (key.piCcsExecution running fresh proof).coins.alpha
          (key.piCcsExecution running fresh proof).coins.gamma
          (key.piCcsExecution running fresh proof).coins.roundPoint
          (key.piCcsCertificate running fresh proof).output) := by
  exact SumCheck.Finite.FixedPhase.checkChain_eq_true_iff
    key.extensionOps.toOps
    (((key.statement running fresh).verifierInput key.lift).initial
      key.extensionOps (key.piCcsExecution running fresh proof).coins.gamma)
    (ProtocolPolynomial.terminalFromMessage key.extensionOps
      ((key.statement running fresh).verifierInput key.lift)
      (key.piCcsExecution running fresh proof).coins.alpha
      (key.piCcsExecution running fresh proof).coins.gamma
      (key.piCcsExecution running fresh proof).coins.roundPoint
      (key.piCcsCertificate running fresh proof).output)
    (key.piCcsFixedCertificate running fresh proof).rounds
    (key.piCcsExecution running fresh proof).coins.roundPoint.coordinates

theorem piCcsProbe_rounds
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    {shape : Shape}
    {columns blockCount degreeBound : Nat}
    (key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound)
    (running : Running Extension Commitment PublicInput shape)
    (fresh : Fresh Commitment PublicInput shape)
    (proof : Proof Extension Commitment shape degreeBound) :
    (key.piCcsProbe running fresh proof).response.rounds =
      SumCheck.Finite.FixedPhase.RawCertificate.encode
        (key.piCcsFixedCertificate running fresh proof) := by
  congr 1
  simp [Key.piCcsProbe, Key.piCcsCertificate,
    NightstreamFPrime.Spec.Folding.PiCCS.TranscriptReplay.Certificate.toFinite,
    NightstreamFPrime.Spec.Folding.PiCCS.TranscriptReplay.Certificate.toTranscript,
    FiatShamir.Certificate.toFinite, Key.piCcsFixedCertificate,
    SumCheck.Finite.FixedPhase.RawCertificate.encode]
  change (List.ofFn fun round => (proof.piCcsRounds round).toMessage) =
    List.ofFn fun round => (proof.piCcsRounds round).toMessage
  rfl

theorem piCcsCheck_eq_true_iff_fixedWidthAccepted
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    [DecidableEq Extension]
    {shape : Shape}
    {columns blockCount degreeBound : Nat}
    (key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound)
    (running : Running Extension Commitment PublicInput shape)
    (fresh : Fresh Commitment PublicInput shape)
    (proof : Proof Extension Commitment shape degreeBound) :
    piCcsCheck key running fresh proof = true <->
      (key.piCcsProbe running fresh proof).FixedWidthAccepted
        key.extensionOps key.lift (key.statement running fresh) degreeBound := by
  unfold StrongReduction.Probe.FixedWidthAccepted
  unfold ProtocolPolynomial.FixedWidth.check
  rw [piCcsProbe_rounds, SumCheck.Finite.FixedPhase.RawCertificate.check_encode]
  rfl

theorem verify_eq_none_of_piRlcFailure
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    [DecidableEq Extension]
    {shape : Shape}
    {columns blockCount degreeBound : Nat}
    (key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound)
    (running : Running Extension Commitment PublicInput shape)
    (fresh : Fresh Commitment PublicInput shape)
    (proof : Proof Extension Commitment shape degreeBound)
    (failure : key.piRlcChallenges running fresh proof = none) :
    verify key running fresh proof = none := by
  have parentFailure : key.parent running fresh proof = none := by
    simp [Key.parent, failure]
  have attemptFailure : key.piDecAttempt running fresh proof = none := by
    simp [Key.piDecAttempt, parentFailure]
  simp [verify, piDecCheck, attemptFailure]

theorem verify_eq_some_iff
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    [DecidableEq Extension]
    {shape : Shape}
    {columns blockCount degreeBound : Nat}
    (key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound)
    (running : Running Extension Commitment PublicInput shape)
    (fresh : Fresh Commitment PublicInput shape)
    (proof : Proof Extension Commitment shape degreeBound)
    (result : Running Extension Commitment PublicInput shape) :
    verify key running fresh proof = some result <->
      piCcsCheck key running fresh proof = true /\
      piDecCheck key running fresh proof = true /\
      key.output running fresh proof = some result := by
  simp only [verify]
  by_cases ccs : piCcsCheck key running fresh proof = true <;>
    by_cases dec : piDecCheck key running fresh proof = true <;>
    simp [ccs, dec]

end NightstreamFPrime.Spec.Folding.Nifs.PaperNonInteractive
