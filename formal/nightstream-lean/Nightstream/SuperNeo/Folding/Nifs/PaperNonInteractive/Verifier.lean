import Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive.Types

/-!
Compact executable verifier for the paper SuperNeo NIFS message.

Owns: the two executable checks (`Pi_CCS` joint polynomial and operational
`Pi_DEC` recomposition), the deterministic `Option` result, and exact Boolean
and graph correspondences.

Does not own: semantic soundness/completeness, extraction, security-event
bounds, concrete hashing/commitments, Rust, R1CS, artifacts, minimality, or
costs.

Emits constraints: no.

`Pi_RLC` has no prover-carried acceptance bit: its challenge vector and parent
are computed by `Types.Key` before `Pi_DEC` is checked.

| Protocol phase | Retained executable obligation | Runtime owner |
|---|---|---|
| `Pi_CCS` | fixed-width round equations and verifier-computed terminal identity | `piCcsCheck` |
| `Pi_RLC` | no independent check; transcript challenges and parent are computed | `Key.piRlcChallenges`, `Key.parent` |
| `Pi_DEC` | parent stage/arity and exact commitment/evaluation recomposition | `piDecCheck` |
| NIFS | accept iff both retained checks pass; return only the computed running output | `verify` |
-/

set_option autoImplicit false

namespace Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Folding
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint

universe uExtension uCommitment uPublicInput uScalar uState

/-- Executable transcript-bound joint-`Pi_CCS` check. -/
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

/-- Executable operational `Pi_DEC` check over the computed parent. -/
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
  let attempt := key.piDecAttempt running fresh proof
  letI := key.piDecDecision attempt
  decide (PiDEC.PaperVerifier.Accepted key.piDecAlgebra
    key.piDecEvaluationArity attempt)

/-- The one-message deterministic NIFS verifier. -/
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
    some (key.output running fresh proof)
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
      PiDEC.PaperVerifier.Accepted key.piDecAlgebra
        key.piDecEvaluationArity (key.piDecAttempt running fresh proof) := by
  letI := key.piDecDecision (key.piDecAttempt running fresh proof)
  simp [piDecCheck]

/-- The typed `Pi_CCS` checker contains only the round recurrence and terminal
equation. Coefficient width is guaranteed by `FixedPolynomial`; no
canonical-list or degree Boolean is present. -/
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

/-- Accepted graph points are exactly the two checks plus the verifier's
computed output. -/
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
      result = key.output running fresh proof := by
  simp only [verify]
  by_cases ccs : piCcsCheck key running fresh proof = true <;>
    by_cases dec : piDecCheck key running fresh proof = true <;>
    simp [ccs, dec, eq_comm]

end Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive
