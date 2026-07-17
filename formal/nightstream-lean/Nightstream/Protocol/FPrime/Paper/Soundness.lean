import Nightstream.Protocol.FPrime.Paper.CertificateVerifier

/-!
Knowledge-soundness boundary for the certificate-oriented recursive verifier.

Owns: the exact connection from one retained `F'_j` recursive certificate to
the SuperNeo `Pi_CCS -> Pi_RLC -> Pi_DEC` knowledge theorem.

Does not own: an implementation extractor, probability bounds, random-oracle
security, relaxed-binding security, Fiat--Shamir, Rust, R1CS, constraint
counts, or permission to remove checks.

Emits constraints: no.

Authority boundary: deterministic verifier acceptance is not silently
upgraded to knowledge soundness. `KnowledgeBoundary` names every external
extractor, uniqueness, final-opening, and rewind premise. The conclusion is
therefore exactly “all selected NIFS inputs are valid or a named paper bad
event occurred,” never unconditional validity.

| Protocol | Phase | Proof family | Mathematical obligation | Lean owner |
|---|---|---|---|---|
| `F'_j` | selected output | final CE openings | every one of the `k` selected output children has a valid opening | `KnowledgeBoundary.finalValid` |
| `Pi_RLC` | rewind | weak extractor | two verifier forks produce ambient openings or a named sampling failure | `KnowledgeBoundary.extractor` |
| `Pi_RLC` | binding | uniqueness | unequal ambient openings yield a relaxed-binding collision | `KnowledgeBoundary.uniqueness` |
| `Pi_CCS` | rewind | arithmetization | equal extracted openings instantiate the independent Pi_CCS arithmetization | `KnowledgeBoundary.rewindArithmetization` |
| NIFS | composition | knowledge result | selected source claims are valid or Pi_CCS, sampling, or binding fails | `RecursiveCertificate.inputsValid_or_badEvent` |
-/

namespace Nightstream.Protocol.FPrime.Paper

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

namespace RecursiveCertificate

/-- The verifier selected by the certificate's checked prior counter. -/
def verifier
    {family : NifsFamily VerifierKey Structure Assignment PublicInput Point
      Evaluation Commitment Scalar Challenge Value relation params slotCount}
    {input : Input VerifierKey State Witness Structure PublicInput Point
      Evaluation Commitment params slotCount}
    {output : Output Digest State Structure PublicInput Point Evaluation
      Commitment params slotCount}
    (certificate : RecursiveCertificate family input output) :=
  selectedVerifier family input certificate.priorPcValid

/-- Exact selected output slot whose children are justified by the retained
NIFS edge. -/
def selectedOutput
    {family : NifsFamily VerifierKey Structure Assignment PublicInput Point
      Evaluation Commitment Scalar Challenge Value relation params slotCount}
    {input : Input VerifierKey State Witness Structure PublicInput Point
      Evaluation Commitment params slotCount}
    {output : Output Digest State Structure PublicInput Point Evaluation
      Commitment params slotCount}
    (certificate : RecursiveCertificate family input output) :=
  output.runningNext (selectedIndex certificate.priorPcValid)

end RecursiveCertificate

/--
All non-deterministic or cryptographic premises required by the paper
knowledge theorem for one fixed recursive certificate.

These are theorem premises, not verifier checks and not certificate fields.
-/
structure KnowledgeBoundary
    {family : NifsFamily VerifierKey Structure Assignment PublicInput Point
      Evaluation Commitment Scalar Challenge Value relation params slotCount}
    {input : Input VerifierKey State Witness Structure PublicInput Point
      Evaluation Commitment params slotCount}
    {output : Output Digest State Structure PublicInput Point Evaluation
      Commitment params slotCount}
    (certificate : RecursiveCertificate family input output)
    (bindingOps : PiRLC.RelaxedBindingOps Assignment Commitment Scalar)
    (sampling : PiRLC.SamplingBoundary certificate.verifier.arity.total) where
  finalAssignments : Fin params.k -> Assignment
  finalValid : forall child,
    CE.Holds relation params (certificate.selectedOutput child)
      (finalAssignments child)
  extractor : Composition.WeakExtractor relation params
    certificate.verifier.rlcAlgebra certificate.edge.attempt.piRlc sampling
  uniqueness : PiRLC.UniquenessBridge relation params bindingOps
    (n := certificate.verifier.arity.total)
  rewindArithmetization : forall leftAssignments rightAssignments,
    PiRLC.AmbientOpenings relation params
        certificate.edge.attempt.piRlc.inputs leftAssignments ->
      PiRLC.AmbientOpenings relation params
        certificate.edge.attempt.piRlc.inputs rightAssignments ->
      leftAssignments = rightAssignments ->
      PiCCS.Arithmetization relation params certificate.verifier.sumcheckOps
        certificate.edge.attempt.piCcs leftAssignments

namespace RecursiveCertificate

/--
The recursive verifier's retained NIFS edge satisfies the paper knowledge
direction under exactly the named external boundaries.

This theorem is the semantic security handoff. It does not mention the outer
hash/control checks because those establish the `F'` transition, while the
result here concerns validity of the selected NIFS source batch.
-/
theorem inputsValid_or_badEvent
    {family : NifsFamily VerifierKey Structure Assignment PublicInput Point
      Evaluation Commitment Scalar Challenge Value relation params slotCount}
    {input : Input VerifierKey State Witness Structure PublicInput Point
      Evaluation Commitment params slotCount}
    {output : Output Digest State Structure PublicInput Point Evaluation
      Commitment params slotCount}
    (certificate : RecursiveCertificate family input output)
    (bindingOps : PiRLC.RelaxedBindingOps Assignment Commitment Scalar)
    (sampling : PiRLC.SamplingBoundary certificate.verifier.arity.total)
    (boundary : KnowledgeBoundary certificate bindingOps sampling) :
    Composition.InputsValid relation params certificate.edge.attempt.piCcs \/
      Composition.BadEvent relation params bindingOps sampling
        certificate.edge.attempt.piCcs
        certificate.edge.attempt.piRlc.inputs := by
  have finalValid : forall child,
      CE.Holds relation params
        (certificate.edge.attempt.piDec.children child)
        (boundary.finalAssignments child) := by
    intro child
    rw [certificate.edge.outputExact]
    exact boundary.finalValid child
  exact Nifs.accepted_inputsValid_or_badEvent
    relation params certificate.verifier.sumcheckOps
    certificate.verifier.rlcAlgebra certificate.verifier.decAlgebra
    bindingOps certificate.verifier.arity sampling certificate.edge.attempt
    boundary.finalAssignments certificate.verifier.kPositive
    certificate.edge.accepted finalValid boundary.extractor
    boundary.uniqueness boundary.rewindArithmetization

end RecursiveCertificate

end


end Nightstream.Protocol.FPrime.Paper
