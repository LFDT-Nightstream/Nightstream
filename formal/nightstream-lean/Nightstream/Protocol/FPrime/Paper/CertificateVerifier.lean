import Nightstream.Protocol.FPrime.Paper

/-!
Legacy certificate-oriented packaging for the abstract paper `F'_j`
relation.

Owns: the typed recursive certificate retained by an executable verifier, the
base/recursive semantic acceptance predicates, and exact soundness and
completeness equivalences with the independently stated Construction-2
relation.

Does not own: a canonical active verifier, a minimality proof, concrete Phi81
NIFS semantics, Rust control flow, R1CS rows, Fiat--Shamir, Poseidon2,
concrete encodings, cryptographic assumptions, constraint counts, or
permission to remove a check.

Emits constraints: no.

Authority boundary: `RecursiveCertificate` carries the full accepted
SuperNeo edge. `RecursiveAccepts` checks the outer Construction-2 obligations
against that edge. Neither definition is generated from, indexed by, or
parameterized by a production circuit. The equivalence theorems establish
only abstract paper-level bookkeeping. The independent `RecursiveHolds`
predicate carries the public NIFS transition and selected structure binding,
not this certificate. Even so, these theorems are neither implementation
refinement nor evidence of minimality.

| Protocol | Phase | Constraint family | Mathematical obligation | Lean owner |
|---|---|---|---|---|
| NIFS | retained edge | exact three-phase attempt | preserve the accepted attempt and exact input/output equations for extraction | `NifsVerifier.EdgeWitness` |
| NIFS | retained edge | public transition | project the retained attempt to the independent transition | `NifsVerifier.EdgeWitness.transition` |
| NIFS | retained edge | output structure | derive child structure from the retained edge and explicit source bindings | `NifsVerifier.EdgeWitness.outputStructure` |
| `F'_j` | recursive certificate | selected NIFS edge | retain the exact accepted `Pi_CCS -> Pi_RLC -> Pi_DEC` attempt for the checked prior slot | `RecursiveCertificate.edge` |
| `F'_j` | recursive control | iteration | recursive execution requires `i > 0` | `RecursiveAccepts.iterationPositive` |
| `F'_j` | recursive authority | prior public link | fresh public input equals the encoded hash of the exact prior preimage | `RecursiveAccepts.priorPublicInput` |
| `F'_j` | application | control / dispatch / step | evaluate exactly the fixed `F_j` selected by `phi` | `RecursiveAccepts.application` |
| `F'_j` | running product | inactive slots | every non-selected accumulator is copied unchanged | `RecursiveAccepts.unchanged` |
| `F'_j` | output authority | next public hash | output digest hashes the exact next preimage | `RecursiveAccepts.outputHash` |
| `F'_j` | branch | base / recursive | accept exactly the paper base predicate or one certificate-backed recursive predicate | `CertificateFPrimeVerifierAccepts` |
| assurance | soundness | semantic refinement | certificate-verifier acceptance implies `Paper.Holds` | `certificateFPrimeVerifier_sound` |
| assurance | completeness | semantic refinement | every `Paper.Holds` execution has a retained verifier certificate | `certificateFPrimeVerifier_complete` |
| assurance | public relation | existential projection | certificate verifier and paper `F'_j` expose exactly the same public digests | `certificatePaperFPrimeStep_iff_paperFPrimeStep` |
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

namespace NifsVerifier

/-- Concrete recursive-edge witness retained for later knowledge soundness.
Unlike an opaque verifier callback, this preserves the exact three-phase
attempt on which extraction, uniqueness, and rewind premises must eventually
be instantiated. -/
structure EdgeWitness
    (verifier : NifsVerifier
      Structure Assignment PublicInput Point Evaluation Commitment
        Scalar Challenge Value relation params)
    (input : PiCCS.InputProduct
      Structure PublicInput Point Evaluation Commitment params verifier.arity)
    (output : Fin params.k ->
      CE.Instance Structure PublicInput Point Evaluation Commitment) where
  attempt : Nifs.Attempt Structure PublicInput Point Evaluation Commitment
    Scalar Challenge Value params verifier.arity
  inputExact : attempt.piCcs.inputs = input
  outputExact : attempt.piDec.children = output
  accepted : Nifs.Accepted verifier.sumcheckOps verifier.rlcAlgebra
    verifier.decAlgebra attempt
  freshStructure : forall fresh,
    (input.fresh fresh).constraintSystem = verifier.expectedStructure
  runningStructure : forall child,
    (input.running child).constraintSystem = verifier.expectedStructure

/-- The retained exact edge implies the public paper NIFS transition. -/
theorem EdgeWitness.transition
    {verifier : NifsVerifier
      Structure Assignment PublicInput Point Evaluation Commitment
        Scalar Challenge Value relation params}
    {input : PiCCS.InputProduct
      Structure PublicInput Point Evaluation Commitment params verifier.arity}
    {output : Fin params.k ->
      CE.Instance Structure PublicInput Point Evaluation Commitment}
    (edge : EdgeWitness verifier input output) :
    verifier.Transition input output := by
  exact ⟨edge.attempt, edge.inputExact, edge.outputExact, edge.accepted⟩

/-- Every retained output child has the verifier-owned relation structure. -/
theorem EdgeWitness.outputStructure
    {verifier : NifsVerifier
      Structure Assignment PublicInput Point Evaluation Commitment
        Scalar Challenge Value relation params}
    {input : PiCCS.InputProduct
      Structure PublicInput Point Evaluation Commitment params verifier.arity}
    {output : Fin params.k ->
      CE.Instance Structure PublicInput Point Evaluation Commitment}
    (edge : EdgeWitness verifier input output)
    (child : Fin params.k) :
    (output child).constraintSystem = verifier.expectedStructure :=
  edge.transition.outputStructure {
    fresh := edge.freshStructure
    running := edge.runningStructure
  } child

end NifsVerifier

/--
Verifier-retained recursive proof data.

The checked prior counter is stored with its range proof, so the selected slot
and verifier are total. The edge then fixes the exact NIFS input and selected
output slot; no digest is accepted as a substitute for this data.
-/
structure RecursiveCertificate
    (family : NifsFamily VerifierKey Structure Assignment PublicInput Point
      Evaluation Commitment Scalar Challenge Value relation params slotCount)
    (input : Input VerifierKey State Witness Structure PublicInput Point
      Evaluation Commitment params slotCount)
    (output : Output Digest State Structure PublicInput Point Evaluation
      Commitment params slotCount) where
  priorPcValid : InRange slotCount input.priorPc
  edge : (selectedVerifier family input priorPcValid).EdgeWitness
    (selectedNifsInput family input priorPcValid)
    (output.runningNext (selectedIndex priorPcValid))

/--
Outer recursive obligations checked around one retained SuperNeo edge.

The NIFS obligation is owned by `certificate.edge`; the fields below are the
remaining Construction-2 checks. Keeping these owners separate is what later
allows a necessity proof to remove exactly one family at a time.
-/
structure RecursiveAccepts
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
  outputHash : OutputHolds machine input output

/-- One recursive verifier execution, existentially hiding only its typed
certificate. -/
def CertificateRecursiveVerifierAccepts
    (family : NifsFamily VerifierKey Structure Assignment PublicInput Point
      Evaluation Commitment Scalar Challenge Value relation params slotCount)
    (machine : Machine VerifierKey Digest State Witness Structure PublicInput
      Point Evaluation Commitment params slotCount)
    (functionIndex : Fin slotCount)
    (input : Input VerifierKey State Witness Structure PublicInput Point
      Evaluation Commitment params slotCount)
    (output : Output Digest State Structure PublicInput Point Evaluation
      Commitment params slotCount) : Prop :=
  exists certificate : RecursiveCertificate family input output,
    RecursiveAccepts family machine functionIndex input output certificate

/-- The semantic verifier accepts either the exact paper base branch or one
certificate-backed recursive branch. -/
def CertificateFPrimeVerifierAccepts
    (family : NifsFamily VerifierKey Structure Assignment PublicInput Point
      Evaluation Commitment Scalar Challenge Value relation params slotCount)
    (machine : Machine VerifierKey Digest State Witness Structure PublicInput
      Point Evaluation Commitment params slotCount)
    (functionIndex : Fin slotCount)
    (input : Input VerifierKey State Witness Structure PublicInput Point
      Evaluation Commitment params slotCount)
    (output : Output Digest State Structure PublicInput Point Evaluation
      Commitment params slotCount) : Prop :=
  BaseHolds machine functionIndex input output \/
    CertificateRecursiveVerifierAccepts family machine functionIndex input output

/-- A certificate-backed recursive verifier execution implies the independent
paper recursive predicate. -/
theorem certificateRecursiveVerifier_sound
    {family : NifsFamily VerifierKey Structure Assignment PublicInput Point
      Evaluation Commitment Scalar Challenge Value relation params slotCount}
    {machine : Machine VerifierKey Digest State Witness Structure PublicInput
      Point Evaluation Commitment params slotCount}
    {functionIndex : Fin slotCount}
    {input : Input VerifierKey State Witness Structure PublicInput Point
      Evaluation Commitment params slotCount}
    {output : Output Digest State Structure PublicInput Point Evaluation
      Commitment params slotCount}
    (accepted : CertificateRecursiveVerifierAccepts
      family machine functionIndex input output) :
    RecursiveHolds family machine functionIndex input output := by
  rcases accepted with ⟨certificate, outer⟩
  exact {
    iterationPositive := outer.iterationPositive
    priorPcValid := certificate.priorPcValid
    priorPublicInput := outer.priorPublicInput
    application := outer.application
    selectedStructures := {
      fresh := certificate.edge.freshStructure
      running := certificate.edge.runningStructure
    }
    selectedNifs := certificate.edge.transition
    unchanged := outer.unchanged
    outputHash := outer.outputHash
  }

/-- Every independent paper recursive execution contains enough information
to construct the typed verifier certificate. -/
theorem certificateRecursiveVerifier_complete
    {family : NifsFamily VerifierKey Structure Assignment PublicInput Point
      Evaluation Commitment Scalar Challenge Value relation params slotCount}
    {machine : Machine VerifierKey Digest State Witness Structure PublicInput
      Point Evaluation Commitment params slotCount}
    {functionIndex : Fin slotCount}
    {input : Input VerifierKey State Witness Structure PublicInput Point
      Evaluation Commitment params slotCount}
    {output : Output Digest State Structure PublicInput Point Evaluation
      Commitment params slotCount}
    (accepted : RecursiveHolds family machine functionIndex input output) :
    CertificateRecursiveVerifierAccepts
      family machine functionIndex input output := by
  rcases accepted.selectedNifs with
    ⟨attempt, inputExact, outputExact, phaseAccepted⟩
  let edge :
      (selectedVerifier family input accepted.priorPcValid).EdgeWitness
        (selectedNifsInput family input accepted.priorPcValid)
        (output.runningNext (selectedIndex accepted.priorPcValid)) := {
    attempt := attempt
    inputExact := inputExact
    outputExact := outputExact
    accepted := phaseAccepted
    freshStructure := accepted.selectedStructures.fresh
    runningStructure := accepted.selectedStructures.running
  }
  let certificate : RecursiveCertificate family input output := {
    priorPcValid := accepted.priorPcValid
    edge := edge
  }
  exact ⟨certificate, {
    iterationPositive := accepted.iterationPositive
    priorPublicInput := accepted.priorPublicInput
    application := accepted.application
    unchanged := accepted.unchanged
    outputHash := accepted.outputHash
  }⟩

/-- Exact semantic equivalence for the recursive branch. -/
theorem certificateRecursiveVerifier_iff_recursiveHolds
    {family : NifsFamily VerifierKey Structure Assignment PublicInput Point
      Evaluation Commitment Scalar Challenge Value relation params slotCount}
    {machine : Machine VerifierKey Digest State Witness Structure PublicInput
      Point Evaluation Commitment params slotCount}
    {functionIndex : Fin slotCount}
    {input : Input VerifierKey State Witness Structure PublicInput Point
      Evaluation Commitment params slotCount}
    {output : Output Digest State Structure PublicInput Point Evaluation
      Commitment params slotCount} :
    CertificateRecursiveVerifierAccepts family machine functionIndex input output ↔
      RecursiveHolds family machine functionIndex input output := by
  constructor
  · exact certificateRecursiveVerifier_sound
  · exact certificateRecursiveVerifier_complete

/-- Semantic verifier soundness for the full base/recursive branch split. -/
theorem certificateFPrimeVerifier_sound
    {family : NifsFamily VerifierKey Structure Assignment PublicInput Point
      Evaluation Commitment Scalar Challenge Value relation params slotCount}
    {machine : Machine VerifierKey Digest State Witness Structure PublicInput
      Point Evaluation Commitment params slotCount}
    {functionIndex : Fin slotCount}
    {input : Input VerifierKey State Witness Structure PublicInput Point
      Evaluation Commitment params slotCount}
    {output : Output Digest State Structure PublicInput Point Evaluation
      Commitment params slotCount}
    (accepted : CertificateFPrimeVerifierAccepts
      family machine functionIndex input output) :
    Holds family machine functionIndex input output := by
  rcases accepted with base | recursive
  · exact .base base
  · exact .recursive (certificateRecursiveVerifier_sound recursive)

/-- Semantic verifier completeness for the full paper branch split. -/
theorem certificateFPrimeVerifier_complete
    {family : NifsFamily VerifierKey Structure Assignment PublicInput Point
      Evaluation Commitment Scalar Challenge Value relation params slotCount}
    {machine : Machine VerifierKey Digest State Witness Structure PublicInput
      Point Evaluation Commitment params slotCount}
    {functionIndex : Fin slotCount}
    {input : Input VerifierKey State Witness Structure PublicInput Point
      Evaluation Commitment params slotCount}
    {output : Output Digest State Structure PublicInput Point Evaluation
      Commitment params slotCount}
    (accepted : Holds family machine functionIndex input output) :
    CertificateFPrimeVerifierAccepts
      family machine functionIndex input output := by
  cases accepted with
  | base base => exact Or.inl base
  | recursive recursive =>
      exact Or.inr (certificateRecursiveVerifier_complete recursive)

/-- The certificate-oriented semantic verifier accepts exactly the
independent paper relation on fixed input/output values. -/
theorem certificateFPrimeVerifier_iff_holds
    {family : NifsFamily VerifierKey Structure Assignment PublicInput Point
      Evaluation Commitment Scalar Challenge Value relation params slotCount}
    {machine : Machine VerifierKey Digest State Witness Structure PublicInput
      Point Evaluation Commitment params slotCount}
    {functionIndex : Fin slotCount}
    {input : Input VerifierKey State Witness Structure PublicInput Point
      Evaluation Commitment params slotCount}
    {output : Output Digest State Structure PublicInput Point Evaluation
      Commitment params slotCount} :
    CertificateFPrimeVerifierAccepts family machine functionIndex input output ↔
      Holds family machine functionIndex input output := by
  constructor
  · exact certificateFPrimeVerifier_sound
  · exact certificateFPrimeVerifier_complete

/-- Public-output projection of the semantic verifier. -/
def CertificatePaperFPrimeStep
    (family : NifsFamily VerifierKey Structure Assignment PublicInput Point
      Evaluation Commitment Scalar Challenge Value relation params slotCount)
    (machine : Machine VerifierKey Digest State Witness Structure PublicInput
      Point Evaluation Commitment params slotCount)
    (functionIndex : Fin slotCount)
    (x : Digest) : Prop :=
  exists input : Input VerifierKey State Witness Structure PublicInput Point
      Evaluation Commitment params slotCount,
    exists output : Output Digest State Structure PublicInput Point Evaluation
      Commitment params slotCount,
      output.x = x /\
        CertificateFPrimeVerifierAccepts
          family machine functionIndex input output

/-- The certificate-oriented verifier and independent paper relation expose
exactly the same public digest language. -/
theorem certificatePaperFPrimeStep_iff_paperFPrimeStep
    {family : NifsFamily VerifierKey Structure Assignment PublicInput Point
      Evaluation Commitment Scalar Challenge Value relation params slotCount}
    {machine : Machine VerifierKey Digest State Witness Structure PublicInput
      Point Evaluation Commitment params slotCount}
    {functionIndex : Fin slotCount}
    {x : Digest} :
    CertificatePaperFPrimeStep family machine functionIndex x ↔
      PaperFPrimeStep family machine functionIndex x := by
  constructor
  · rintro ⟨input, output, outputDigest, accepted⟩
    exact ⟨input, output, outputDigest,
      certificateFPrimeVerifier_sound accepted⟩
  · rintro ⟨input, output, outputDigest, accepted⟩
    exact ⟨input, output, outputDigest,
      certificateFPrimeVerifier_complete accepted⟩

end

end Nightstream.Protocol.FPrime.Paper
