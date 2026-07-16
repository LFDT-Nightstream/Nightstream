import Nightstream.Implementation.R1CS.Core.SeededPhi81Sampler

/-!
No-rejection certificates imply bounded seeded-Phi81 sampler success.

Assurance tier: executable-semantics refinement. This file states the
mathematical condition that every initial coefficient word in a finite run is
already canonical, provides a small executable checker for finite ranges, and
proves that such a run cannot exhaust rejection fuel.

Owns: initial-vector acceptance; finite consecutive-run acceptance; exact
range checking; compositional run concatenation; and the generic implication
from no rejection to successful vector traversal.

Does not own: any concrete seed or range certificate; ChaCha8; seed
derivation; Rust conformance; the general rejection path; Phi81 rotation; SIS
security; R1CS rows; Poseidon2; transcript authority; row removal; or costs.

Emits constraints: no.

Authority boundary: this is only a sufficient execution lemma. The general
`FirstAccepted` semantics remains authoritative when rejection occurs. A
production caller must separately prove each concrete finite range accepted;
an unchecked fixture or digest cannot inhabit `InitialRunAccepted`.

| Protocol | Phase | Mathematical branch | Definition/theorem | Exact guarantee |
|---|---|---|---|---|
| seeded SIS | coefficient sampling | one initial vector | `VectorInitiallyAccepted` | all 54 initial words are canonical |
| seeded SIS | coefficient sampling | consecutive vectors | `InitialRunAccepted` | each vector starts at the exact 108-word cursor stride |
| seeded SIS | coefficient sampling | finite certificate | `runInitiallyAcceptedCheck_eq_true_iff` | executable range check is equivalent to the proposition |
| seeded SIS | coefficient sampling | batch composition | `InitialRunAccepted.append` | adjacent certified ranges compose without a cursor gap |
| seeded SIS | coefficient sampling | bounded execution | `sampleVectors_exists_of_initiallyAccepted` | no-rejection run produces some bounded sampler output for any fuel |
-/

namespace Nightstream.Implementation.R1CS.SeededPhi81Sampler

def vectorWordStride : Nat := 2 * dimension

/-- Every initial word of one 54-coefficient vector is already a canonical
Goldilocks representative. -/
def VectorInitiallyAccepted (stream : WordStream) (seed : List Nat)
    (wordPosition : Nat) : Prop :=
  forall value, value ∈ stream seed wordPosition dimension -> value < modulus

/-- Consecutive no-rejection vectors with the exact production cursor rule. -/
def InitialRunAccepted (stream : WordStream) (seed : List Nat)
    (count wordPosition : Nat) : Prop :=
  forall index, index < count ->
    VectorInitiallyAccepted stream seed
      (wordPosition + vectorWordStride * index)

/-- Executable leaf used for small, independently audited concrete ranges. -/
def runInitiallyAcceptedCheck (stream : WordStream) (seed : List Nat)
    (count wordPosition : Nat) : Bool :=
  (List.range count).all fun index =>
    (stream seed (wordPosition + vectorWordStride * index) dimension).all
      fun value => decide (value < modulus)

theorem runInitiallyAcceptedCheck_eq_true_iff
    (stream : WordStream) (seed : List Nat) (count wordPosition : Nat) :
    runInitiallyAcceptedCheck stream seed count wordPosition = true <->
      InitialRunAccepted stream seed count wordPosition := by
  simp [runInitiallyAcceptedCheck, InitialRunAccepted,
    VectorInitiallyAccepted]

theorem InitialRunAccepted.head
    {stream : WordStream} {seed : List Nat} {count wordPosition : Nat}
    (accepted : InitialRunAccepted stream seed (count + 1) wordPosition) :
    VectorInitiallyAccepted stream seed wordPosition := by
  simpa [InitialRunAccepted] using accepted 0 (by omega)

theorem InitialRunAccepted.tail
    {stream : WordStream} {seed : List Nat} {count wordPosition : Nat}
    (accepted : InitialRunAccepted stream seed (count + 1) wordPosition) :
    InitialRunAccepted stream seed count
      (wordPosition + vectorWordStride) := by
  intro index indexLt
  have next := accepted (index + 1) (by omega)
  simpa [InitialRunAccepted, vectorWordStride, Nat.mul_add,
    Nat.add_assoc, Nat.add_left_comm, Nat.add_comm] using next

/-- Adjacent range certificates compose exactly at the next vector cursor. -/
theorem InitialRunAccepted.append
    {stream : WordStream} {seed : List Nat}
    {leftCount rightCount wordPosition : Nat}
    (left : InitialRunAccepted stream seed leftCount wordPosition)
    (right : InitialRunAccepted stream seed rightCount
      (wordPosition + vectorWordStride * leftCount)) :
    InitialRunAccepted stream seed (leftCount + rightCount) wordPosition := by
  intro index indexLt
  by_cases inLeft : index < leftCount
  · exact left index inLeft
  · have rightIndexLt : index - leftCount < rightCount := by omega
    have value := right (index - leftCount) rightIndexLt
    have indexEq : index = leftCount + (index - leftCount) := by
      omega
    have positionEq :
        wordPosition + vectorWordStride * index =
          wordPosition + vectorWordStride * leftCount +
            vectorWordStride * (index - leftCount) := by
      calc
        wordPosition + vectorWordStride * index =
            wordPosition + vectorWordStride *
              (leftCount + (index - leftCount)) :=
          congrArg (fun value =>
            wordPosition + vectorWordStride * value) indexEq
        _ = wordPosition + vectorWordStride * leftCount +
              vectorWordStride * (index - leftCount) := by
          rw [Nat.mul_add, Nat.add_assoc]
    rw [positionEq]
    exact value

private theorem repairRejected_of_allAccepted
    {stream : WordStream} {seed candidates : List Nat}
    {fuel wordPosition : Nat}
    (accepted : forall value, value ∈ candidates -> value < modulus) :
    repairRejected stream seed fuel candidates wordPosition =
      some (candidates, wordPosition) := by
  induction candidates generalizing wordPosition with
  | nil => simp [repairRejected]
  | cons candidate tail ih =>
      have candidateAccepted : candidate < modulus :=
        accepted candidate (by simp)
      have tailAccepted : forall value, value ∈ tail -> value < modulus := by
        intro value member
        exact accepted value (by simp [member])
      simp [repairRejected, candidateAccepted, ih tailAccepted]

theorem sampleVector_of_initiallyAccepted
    {stream : WordStream} {seed : List Nat} {fuel wordPosition : Nat}
    (accepted : VectorInitiallyAccepted stream seed wordPosition) :
    sampleVector stream seed fuel wordPosition =
      some (stream seed wordPosition dimension,
        wordPosition + vectorWordStride) := by
  unfold sampleVector vectorWordStride
  rw [repairRejected_of_allAccepted accepted]

private theorem sampleVectors_go_exists_of_initiallyAccepted
    {stream : WordStream} {seed : List Nat} {fuel count wordPosition : Nat}
    {reversed : List (List Nat)}
    (accepted : InitialRunAccepted stream seed count wordPosition) :
    exists output,
      sampleVectors.go stream seed fuel count wordPosition reversed =
        some output := by
  induction count generalizing wordPosition reversed with
  | zero => exact ⟨reversed.reverse, rfl⟩
  | succ count ih =>
      have headAccepted := accepted.head
      have headSuccess :=
        sampleVector_of_initiallyAccepted (fuel := fuel) headAccepted
      have tailAccepted := accepted.tail
      rcases ih tailAccepted
          (reversed :=
            stream seed wordPosition dimension :: reversed) with
        ⟨output, tailSuccess⟩
      exact ⟨output, by
        simp [sampleVectors.go, headSuccess, tailSuccess]⟩

/-- A no-rejection run succeeds for every fuel value; fuel is never
consulted on this path. -/
theorem sampleVectors_exists_of_initiallyAccepted
    {stream : WordStream} {seed : List Nat} {fuel count wordPosition : Nat}
    (accepted : InitialRunAccepted stream seed count wordPosition) :
    exists output,
      sampleVectors stream seed fuel count wordPosition = some output := by
  exact sampleVectors_go_exists_of_initiallyAccepted accepted

end Nightstream.Implementation.R1CS.SeededPhi81Sampler
