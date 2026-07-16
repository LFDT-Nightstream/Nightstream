import Nightstream.Protocol.FPrime.Paper

/-!
Certificate-oriented verifier semantics for the paper `F'_j` relation.

Owns: the typed recursive certificate retained by an executable verifier, the
base/recursive semantic acceptance predicates, and exact soundness and
completeness equivalences with the independently stated Construction-2
relation.

Does not own: Rust control flow, R1CS rows, Fiat--Shamir, Poseidon2, concrete
encodings, cryptographic assumptions, constraint counts, or permission to
remove a check.

Emits constraints: no.

Authority boundary: `RecursiveCertificate` carries the full accepted
SuperNeo edge. `RecursiveAccepts` checks the outer Construction-2 obligations
against that edge. Neither definition is generated from, indexed by, or
parameterized by a production circuit. The equivalence theorems therefore
establish a semantic verifier target; they are not implementation refinement.

| Protocol | Phase | Constraint family | Mathematical obligation | Lean owner |
|---|---|---|---|---|
| `F'_j` | recursive certificate | selected NIFS edge | retain the exact accepted `Pi_CCS -> Pi_RLC -> Pi_DEC` attempt for the checked prior slot | `RecursiveCertificate.edge` |
| `F'_j` | recursive control | iteration | recursive execution requires `i > 0` | `RecursiveAccepts.iterationPositive` |
| `F'_j` | recursive authority | prior public link | fresh public input equals the encoded hash of the exact prior preimage | `RecursiveAccepts.priorPublicInput` |
| `F'_j` | application | control / dispatch / step | evaluate exactly the fixed `F_j` selected by `phi` | `RecursiveAccepts.application` |
| `F'_j` | running product | inactive slots | every non-selected accumulator is copied unchanged | `RecursiveAccepts.unchanged` |
| `F'_j` | output authority | next public hash | output digest hashes the exact next preimage | `RecursiveAccepts.outputHash` |
| `F'_j` | branch | base / recursive | accept exactly the paper base predicate or one certificate-backed recursive predicate | `MinimalFPrimeVerifierAccepts` |
| assurance | soundness | semantic refinement | minimal verifier acceptance implies `Paper.Holds` | `minimalFPrimeVerifier_sound` |
| assurance | completeness | semantic refinement | every `Paper.Holds` execution has a retained verifier certificate | `minimalFPrimeVerifier_complete` |
| assurance | public relation | existential projection | minimal verifier and paper `F'_j` expose exactly the same public digests | `minimalPaperFPrimeStep_iff_paperFPrimeStep` |
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
def MinimalRecursiveVerifierAccepts
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
def MinimalFPrimeVerifierAccepts
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
    MinimalRecursiveVerifierAccepts family machine functionIndex input output

/-- A certificate-backed recursive verifier execution implies the independent
paper recursive predicate. -/
theorem minimalRecursiveVerifier_sound
    {family : NifsFamily VerifierKey Structure Assignment PublicInput Point
      Evaluation Commitment Scalar Challenge Value relation params slotCount}
    {machine : Machine VerifierKey Digest State Witness Structure PublicInput
      Point Evaluation Commitment params slotCount}
    {functionIndex : Fin slotCount}
    {input : Input VerifierKey State Witness Structure PublicInput Point
      Evaluation Commitment params slotCount}
    {output : Output Digest State Structure PublicInput Point Evaluation
      Commitment params slotCount}
    (accepted : MinimalRecursiveVerifierAccepts
      family machine functionIndex input output) :
    RecursiveHolds family machine functionIndex input output := by
  rcases accepted with ⟨certificate, outer⟩
  exact {
    iterationPositive := outer.iterationPositive
    priorPcValid := certificate.priorPcValid
    priorPublicInput := outer.priorPublicInput
    application := outer.application
    selectedNifs := ⟨certificate.edge⟩
    unchanged := outer.unchanged
    outputHash := outer.outputHash
  }

/-- Every independent paper recursive execution contains enough information
to construct the typed verifier certificate. -/
theorem minimalRecursiveVerifier_complete
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
    MinimalRecursiveVerifierAccepts
      family machine functionIndex input output := by
  rcases accepted.selectedNifs with ⟨edge⟩
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
theorem minimalRecursiveVerifier_iff_recursiveHolds
    {family : NifsFamily VerifierKey Structure Assignment PublicInput Point
      Evaluation Commitment Scalar Challenge Value relation params slotCount}
    {machine : Machine VerifierKey Digest State Witness Structure PublicInput
      Point Evaluation Commitment params slotCount}
    {functionIndex : Fin slotCount}
    {input : Input VerifierKey State Witness Structure PublicInput Point
      Evaluation Commitment params slotCount}
    {output : Output Digest State Structure PublicInput Point Evaluation
      Commitment params slotCount} :
    MinimalRecursiveVerifierAccepts family machine functionIndex input output ↔
      RecursiveHolds family machine functionIndex input output := by
  constructor
  · exact minimalRecursiveVerifier_sound
  · exact minimalRecursiveVerifier_complete

/-- Semantic verifier soundness for the full base/recursive branch split. -/
theorem minimalFPrimeVerifier_sound
    {family : NifsFamily VerifierKey Structure Assignment PublicInput Point
      Evaluation Commitment Scalar Challenge Value relation params slotCount}
    {machine : Machine VerifierKey Digest State Witness Structure PublicInput
      Point Evaluation Commitment params slotCount}
    {functionIndex : Fin slotCount}
    {input : Input VerifierKey State Witness Structure PublicInput Point
      Evaluation Commitment params slotCount}
    {output : Output Digest State Structure PublicInput Point Evaluation
      Commitment params slotCount}
    (accepted : MinimalFPrimeVerifierAccepts
      family machine functionIndex input output) :
    Holds family machine functionIndex input output := by
  rcases accepted with base | recursive
  · exact .base base
  · exact .recursive (minimalRecursiveVerifier_sound recursive)

/-- Semantic verifier completeness for the full paper branch split. -/
theorem minimalFPrimeVerifier_complete
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
    MinimalFPrimeVerifierAccepts
      family machine functionIndex input output := by
  cases accepted with
  | base base => exact Or.inl base
  | recursive recursive =>
      exact Or.inr (minimalRecursiveVerifier_complete recursive)

/-- The minimal semantic verifier accepts exactly the independent paper
relation on fixed input/output values. -/
theorem minimalFPrimeVerifier_iff_holds
    {family : NifsFamily VerifierKey Structure Assignment PublicInput Point
      Evaluation Commitment Scalar Challenge Value relation params slotCount}
    {machine : Machine VerifierKey Digest State Witness Structure PublicInput
      Point Evaluation Commitment params slotCount}
    {functionIndex : Fin slotCount}
    {input : Input VerifierKey State Witness Structure PublicInput Point
      Evaluation Commitment params slotCount}
    {output : Output Digest State Structure PublicInput Point Evaluation
      Commitment params slotCount} :
    MinimalFPrimeVerifierAccepts family machine functionIndex input output ↔
      Holds family machine functionIndex input output := by
  constructor
  · exact minimalFPrimeVerifier_sound
  · exact minimalFPrimeVerifier_complete

/-- Public-output projection of the semantic verifier. -/
def MinimalPaperFPrimeStep
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
        MinimalFPrimeVerifierAccepts
          family machine functionIndex input output

/-- The certificate-oriented verifier and independent paper relation expose
exactly the same public digest language. -/
theorem minimalPaperFPrimeStep_iff_paperFPrimeStep
    {family : NifsFamily VerifierKey Structure Assignment PublicInput Point
      Evaluation Commitment Scalar Challenge Value relation params slotCount}
    {machine : Machine VerifierKey Digest State Witness Structure PublicInput
      Point Evaluation Commitment params slotCount}
    {functionIndex : Fin slotCount}
    {x : Digest} :
    MinimalPaperFPrimeStep family machine functionIndex x ↔
      PaperFPrimeStep family machine functionIndex x := by
  constructor
  · rintro ⟨input, output, outputDigest, accepted⟩
    exact ⟨input, output, outputDigest,
      minimalFPrimeVerifier_sound accepted⟩
  · rintro ⟨input, output, outputDigest, accepted⟩
    exact ⟨input, output, outputDigest,
      minimalFPrimeVerifier_complete accepted⟩

end

end Nightstream.Protocol.FPrime.Paper
