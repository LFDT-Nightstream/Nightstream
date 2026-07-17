import Nightstream.Protocol.FPrime.Paper.CertificateVerifier

/-!
Protocol-level necessity of the `F'_j` output-hash obligation.

Owns: a mutation that changes only the public digest, transport of the exact
recursive NIFS certificate across that mutation, a verifier predicate with
only the output-hash family omitted, and a forged execution accepted by the
weakened predicate but rejected by the full semantic verifier.

Does not own: digest collision resistance, a concrete digest type, Rust,
R1CS, row deletion, constraint counts, or necessity of any other check family.

Emits constraints: no.

Authority boundary: this is a typed protocol countermodel, not a Boolean
independence model. The mutated execution retains the same input, selected
slot, NIFS edge, control result, application result, and running product. Its
only changed value is `output.x`.

| Protocol | Phase | Constraint family | Mutation / guarantee | Lean owner |
|---|---|---|---|---|
| `F'_j` | output | public digest value | replace only `output.x` | `replaceDigest` |
| `F'_j` | recursive certificate | selected NIFS edge | the same exact edge remains well typed because the selected running output is unchanged | `replaceDigestCertificate` |
| `F'_j` | weakened verifier | all except output hash | every retained outer obligation still holds | `forged_accepts_withoutOutputHash` |
| `F'_j` | full verifier | output hash | a distinct forged digest cannot satisfy the exact next-preimage equation | `forged_rejected_by_fullVerifier` |
| assurance | necessity | inclusion witness | dropping only this family admits an invalid public transition | `outputHash_is_necessary` |
-/

namespace Nightstream.Protocol.FPrime.Paper.Necessity.OutputHash

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Folding

universe uStructure uAssignment uPublicInput uPoint uEvaluation uCommitment
  uScalar uChallenge uValue uState uWitness uVerifierKey uDigest

section

variable {VerifierKey : Type uVerifierKey}
variable {Digest : Type uDigest}
variable {State : Type uState}
variable {Witness : Type uWitness}
variable {Structure : Type uStructure}
variable {Assignment : Type uAssignment}
variable {PublicInput : Type uPublicInput}
variable {Point : Type uPoint}
variable {Evaluation : Type uEvaluation}
variable {Commitment : Type uCommitment}
variable {Scalar : Type uScalar}
variable {Challenge : Type uChallenge}
variable {Value : Type uValue}
variable {relation : RelationSemantics
  Structure Assignment PublicInput Point Evaluation Commitment}
variable {params : GlobalParams}
variable {slotCount : Nat}

/-- Change only the externally visible digest of one output. -/
def replaceDigest
    (output : Output Digest State Structure PublicInput Point Evaluation
      Commitment params slotCount)
    (forged : Digest) :
    Output Digest State Structure PublicInput Point Evaluation Commitment params
      slotCount := {
  output with x := forged
}

@[simp] theorem replaceDigest_x
    (output : Output Digest State Structure PublicInput Point Evaluation
      Commitment params slotCount)
    (forged : Digest) :
    (replaceDigest output forged).x = forged := rfl

@[simp] theorem replaceDigest_runningNext
    (output : Output Digest State Structure PublicInput Point Evaluation
      Commitment params slotCount)
    (forged : Digest) :
    (replaceDigest output forged).runningNext = output.runningNext := rfl

/-- The exact selected NIFS edge is independent of the outer public digest. -/
def replaceDigestCertificate
    {family : NifsFamily VerifierKey Structure Assignment PublicInput Point
      Evaluation Commitment Scalar Challenge Value relation params slotCount}
    {input : Input VerifierKey State Witness Structure PublicInput Point
      Evaluation Commitment params slotCount}
    {output : Output Digest State Structure PublicInput Point Evaluation
      Commitment params slotCount}
    (certificate : RecursiveCertificate family input output)
    (forged : Digest) :
    RecursiveCertificate family input (replaceDigest output forged) := {
  priorPcValid := certificate.priorPcValid
  edge := certificate.edge
}

/-- Recursive acceptance with exactly the output-hash family omitted. -/
structure AcceptsWithoutOutputHash
    (family : NifsFamily VerifierKey Structure Assignment PublicInput Point
      Evaluation Commitment Scalar Challenge Value relation params slotCount)
    (machine : Machine VerifierKey Digest State Witness Structure PublicInput
      Point Evaluation Commitment params slotCount)
    (functionIndex : Fin slotCount)
    (input : Input VerifierKey State Witness Structure PublicInput Point
      Evaluation Commitment params slotCount)
    (output : Output Digest State Structure PublicInput Point Evaluation
      Commitment params slotCount)
    (certificate : RecursiveCertificate family input output) : Prop where
  iterationPositive : 0 < input.iteration
  priorPublicInput : input.fresh.publicInput =
    machine.encodeInstance (machine.hash (priorHashPreimage input))
  application : ApplicationHolds machine functionIndex input output
  unchanged : forall slot, slot ≠ selectedIndex certificate.priorPcValid ->
    output.runningNext slot = input.running slot

/-- Mutating only the digest preserves every other recursive obligation. -/
theorem forged_accepts_withoutOutputHash
    {family : NifsFamily VerifierKey Structure Assignment PublicInput Point
      Evaluation Commitment Scalar Challenge Value relation params slotCount}
    {machine : Machine VerifierKey Digest State Witness Structure PublicInput
      Point Evaluation Commitment params slotCount}
    {functionIndex : Fin slotCount}
    {input : Input VerifierKey State Witness Structure PublicInput Point
      Evaluation Commitment params slotCount}
    {output : Output Digest State Structure PublicInput Point Evaluation
      Commitment params slotCount}
    {certificate : RecursiveCertificate family input output}
    (accepted : RecursiveAccepts
      family machine functionIndex input output certificate)
    (forged : Digest) :
    AcceptsWithoutOutputHash family machine functionIndex input
      (replaceDigest output forged)
      (replaceDigestCertificate certificate forged) := by
  exact {
    iterationPositive := accepted.iterationPositive
    priorPublicInput := accepted.priorPublicInput
    application := {
      control := accepted.application.control
      dispatch := accepted.application.dispatch
      application := accepted.application.application
    }
    unchanged := accepted.unchanged
  }

/-- A distinct replacement digest falsifies the exact next-preimage equation. -/
theorem forged_not_outputHolds
    {machine : Machine VerifierKey Digest State Witness Structure PublicInput
      Point Evaluation Commitment params slotCount}
    {input : Input VerifierKey State Witness Structure PublicInput Point
      Evaluation Commitment params slotCount}
    {output : Output Digest State Structure PublicInput Point Evaluation
      Commitment params slotCount}
    (outputHolds : OutputHolds machine input output)
    {forged : Digest}
    (different : forged ≠ output.x) :
    Not (OutputHolds machine input (replaceDigest output forged)) := by
  intro forgedHolds
  apply different
  calc
    forged = machine.hash
        (nextHashPreimage input (replaceDigest output forged)) := forgedHolds
    _ = machine.hash (nextHashPreimage input output) := by rfl
    _ = output.x := outputHolds.symm

/-- No choice of recursive certificate can rescue the forged output, because
the missing equation is certificate-independent. -/
theorem forged_rejected_by_fullVerifier
    {family : NifsFamily VerifierKey Structure Assignment PublicInput Point
      Evaluation Commitment Scalar Challenge Value relation params slotCount}
    {machine : Machine VerifierKey Digest State Witness Structure PublicInput
      Point Evaluation Commitment params slotCount}
    {functionIndex : Fin slotCount}
    {input : Input VerifierKey State Witness Structure PublicInput Point
      Evaluation Commitment params slotCount}
    {output : Output Digest State Structure PublicInput Point Evaluation
      Commitment params slotCount}
    {certificate : RecursiveCertificate family input output}
    (accepted : RecursiveAccepts
      family machine functionIndex input output certificate)
    {forged : Digest}
    (different : forged ≠ output.x) :
    Not (CertificateRecursiveVerifierAccepts family machine functionIndex input
      (replaceDigest output forged)) := by
  intro forgedAccepted
  rcases forgedAccepted with ⟨_, forgedOuter⟩
  exact forged_not_outputHolds accepted.outputHash different
    forgedOuter.outputHash

/-- Concrete protocol-level inclusion witness for the output-hash family. -/
theorem outputHash_is_necessary
    {family : NifsFamily VerifierKey Structure Assignment PublicInput Point
      Evaluation Commitment Scalar Challenge Value relation params slotCount}
    {machine : Machine VerifierKey Digest State Witness Structure PublicInput
      Point Evaluation Commitment params slotCount}
    {functionIndex : Fin slotCount}
    {input : Input VerifierKey State Witness Structure PublicInput Point
      Evaluation Commitment params slotCount}
    {output : Output Digest State Structure PublicInput Point Evaluation
      Commitment params slotCount}
    {certificate : RecursiveCertificate family input output}
    (accepted : RecursiveAccepts
      family machine functionIndex input output certificate)
    {forged : Digest}
    (different : forged ≠ output.x) :
    AcceptsWithoutOutputHash family machine functionIndex input
        (replaceDigest output forged)
        (replaceDigestCertificate certificate forged) /\
      Not (CertificateRecursiveVerifierAccepts family machine functionIndex input
        (replaceDigest output forged)) := by
  exact ⟨forged_accepts_withoutOutputHash accepted forged,
    forged_rejected_by_fullVerifier accepted different⟩

end

end Nightstream.Protocol.FPrime.Paper.Necessity.OutputHash
