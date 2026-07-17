import Nightstream.Protocol.FPrime.Paper.Output

/-!
Constructive model-level completeness for the Construction-2 `F'_j` relation.

Owns: exact selected-slot update/copy, construction of accepted base and
recursive executions from the canonical output function, and projection of
those executions to the public paper relation.

Does not own: construction of an honest SuperNeo NIFS edge, validity of the
configured default vector, hash injectivity, Fiat-Shamir, Poseidon2, Rust or
R1CS refinement, cryptographic soundness, or any row-removal conclusion.

Emits constraints: no.

Authority boundary: recursive completeness takes the independent public NIFS
transition and the selected-input structure binding as separate premises; it
does not manufacture either from a digest or output self-consistency.
`SetupValid` transfer is only for the machine's fixed per-slot setup and is not
a universal `u_perp` theorem. It also does not establish the still-open
`enc_str(F'_j) = expectedStructure` refinement.

| Protocol | Phase | Obligation | Construction/theorem |
|---|---|---|---|
| `F'_j` | base | retain the configured default vector | `derivedOutput_base_holds`, `base_exists_holds` |
| `F'_j` | recursive | replace only the prior-counter slot | `updatedRunning`, `updatedRunning_selected`, `updatedRunning_other` |
| `F'_j` | recursive | use a public NIFS transition plus explicit selected structure binding for the replacement | `derivedOutput_recursive_holds`, `recursive_exists_holds` |
| `F'_j` | public | expose the constructed digest through the paper relation | `base_paperFPrimeStep`, `recursive_paperFPrimeStep` |
| setup | base validity | transfer fixed-slot default semantic validity to the base output | `defaultValid_transfers_to_base` |
| setup | base validity and structure | transfer the combined setup predicate to the accepted base output | `setupValid_transfers_to_base` |
-/

namespace Nightstream.Protocol.FPrime.Paper

open Nightstream.SuperNeo

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

/-- Replace exactly the running slot selected by the checked prior counter. -/
def updatedRunning
    (input : Input VerifierKey State Witness Structure PublicInput Point
      Evaluation Commitment params slotCount)
    (priorPcValid : InRange slotCount input.priorPc)
    (selectedNext : RunningSlot Structure PublicInput Point Evaluation
      Commitment params) :
    RunningProduct Structure PublicInput Point Evaluation Commitment params
      slotCount :=
  fun slot =>
    if slot = selectedIndex priorPcValid then selectedNext else input.running slot

/-- The selected slot of `updatedRunning` is exactly the supplied NIFS output. -/
@[simp] theorem updatedRunning_selected
    (input : Input VerifierKey State Witness Structure PublicInput Point
      Evaluation Commitment params slotCount)
    (priorPcValid : InRange slotCount input.priorPc)
    (selectedNext : RunningSlot Structure PublicInput Point Evaluation
      Commitment params) :
    updatedRunning input priorPcValid selectedNext
        (selectedIndex priorPcValid) = selectedNext := by
  simp [updatedRunning]

/-- Every non-selected slot of `updatedRunning` is copied exactly. -/
theorem updatedRunning_other
    (input : Input VerifierKey State Witness Structure PublicInput Point
      Evaluation Commitment params slotCount)
    (priorPcValid : InRange slotCount input.priorPc)
    (selectedNext : RunningSlot Structure PublicInput Point Evaluation
      Commitment params)
    (slot : Fin slotCount)
    (notSelected : slot ≠ selectedIndex priorPcValid) :
    updatedRunning input priorPcValid selectedNext slot = input.running slot := by
  simp [updatedRunning, notSelected]

/-- The canonical default-vector output realizes the base branch. -/
theorem derivedOutput_base_holds
    {family : NifsFamily VerifierKey Structure Assignment PublicInput Point
      Evaluation Commitment Scalar Challenge Value relation params slotCount}
    (machine : Machine VerifierKey Digest State Witness Structure PublicInput
      Point Evaluation Commitment params slotCount)
    (input : Input VerifierKey State Witness Structure PublicInput Point
      Evaluation Commitment params slotCount)
    (iterationZero : input.iteration = 0)
    (initialState : input.z0 = input.zi) :
    Holds family machine (machine.control input.zi input.witness).index input
      (derivedOutput machine input machine.defaultRunning) := by
  apply Holds.base
  exact {
    iterationZero := iterationZero
    initialState := initialState
    application := derivedOutput_application machine input machine.defaultRunning
    defaultRunning := rfl
    outputHash := derivedOutput_outputHolds machine input machine.defaultRunning
  }

/-- Base inputs construct a concrete accepted output, not only a branch injection. -/
theorem base_exists_holds
    {family : NifsFamily VerifierKey Structure Assignment PublicInput Point
      Evaluation Commitment Scalar Challenge Value relation params slotCount}
    (machine : Machine VerifierKey Digest State Witness Structure PublicInput
      Point Evaluation Commitment params slotCount)
    (input : Input VerifierKey State Witness Structure PublicInput Point
      Evaluation Commitment params slotCount)
    (iterationZero : input.iteration = 0)
    (initialState : input.z0 = input.zi) :
    exists output,
      Holds family machine (machine.control input.zi input.witness).index input
        output := by
  exact ⟨derivedOutput machine input machine.defaultRunning,
    derivedOutput_base_holds machine input iterationZero initialState⟩

/-- The canonical base digest belongs to the public paper relation. -/
theorem base_paperFPrimeStep
    {family : NifsFamily VerifierKey Structure Assignment PublicInput Point
      Evaluation Commitment Scalar Challenge Value relation params slotCount}
    (machine : Machine VerifierKey Digest State Witness Structure PublicInput
      Point Evaluation Commitment params slotCount)
    (input : Input VerifierKey State Witness Structure PublicInput Point
      Evaluation Commitment params slotCount)
    (iterationZero : input.iteration = 0)
    (initialState : input.z0 = input.zi) :
    PaperFPrimeStep family machine (machine.control input.zi input.witness).index
      (derivedOutput machine input machine.defaultRunning).x := by
  exact paperFPrimeStep_of_holds
    (derivedOutput_base_holds machine input iterationZero initialState)

/-- A concrete selected NIFS edge constructs the canonical recursive branch. -/
theorem derivedOutput_recursive_holds
    {family : NifsFamily VerifierKey Structure Assignment PublicInput Point
      Evaluation Commitment Scalar Challenge Value relation params slotCount}
    (machine : Machine VerifierKey Digest State Witness Structure PublicInput
      Point Evaluation Commitment params slotCount)
    (input : Input VerifierKey State Witness Structure PublicInput Point
      Evaluation Commitment params slotCount)
    (iterationPositive : 0 < input.iteration)
    (priorPcValid : InRange slotCount input.priorPc)
    (priorPublicInput : input.fresh.publicInput =
      machine.encodeInstance (machine.hash (priorHashPreimage input)))
    (selectedNext : RunningSlot Structure PublicInput Point Evaluation
      Commitment params)
    (selectedStructures :
      SelectedInputStructuresBound family input priorPcValid)
    (selectedNifs :
      (selectedVerifier family input priorPcValid).Transition
        (selectedNifsInput family input priorPcValid) selectedNext) :
    Holds family machine (machine.control input.zi input.witness).index input
      (derivedOutput machine input
        (updatedRunning input priorPcValid selectedNext)) := by
  apply Holds.recursive
  refine {
    iterationPositive := iterationPositive
    priorPcValid := priorPcValid
    priorPublicInput := priorPublicInput
    application := derivedOutput_application machine input
      (updatedRunning input priorPcValid selectedNext)
    selectedStructures := selectedStructures
    selectedNifs := ?_
    unchanged := ?_
    outputHash := derivedOutput_outputHolds machine input
      (updatedRunning input priorPcValid selectedNext)
  }
  · simpa [derivedOutput] using selectedNifs
  · intro slot notSelected
    exact updatedRunning_other input priorPcValid selectedNext slot notSelected

/-- Recursive inputs plus an accepted selected edge construct an exact output. -/
theorem recursive_exists_holds
    {family : NifsFamily VerifierKey Structure Assignment PublicInput Point
      Evaluation Commitment Scalar Challenge Value relation params slotCount}
    (machine : Machine VerifierKey Digest State Witness Structure PublicInput
      Point Evaluation Commitment params slotCount)
    (input : Input VerifierKey State Witness Structure PublicInput Point
      Evaluation Commitment params slotCount)
    (iterationPositive : 0 < input.iteration)
    (priorPcValid : InRange slotCount input.priorPc)
    (priorPublicInput : input.fresh.publicInput =
      machine.encodeInstance (machine.hash (priorHashPreimage input)))
    (selectedNext : RunningSlot Structure PublicInput Point Evaluation
      Commitment params)
    (selectedStructures :
      SelectedInputStructuresBound family input priorPcValid)
    (selectedNifs :
      (selectedVerifier family input priorPcValid).Transition
        (selectedNifsInput family input priorPcValid) selectedNext) :
    exists output,
      Holds family machine (machine.control input.zi input.witness).index input
        output := by
  exact ⟨derivedOutput machine input
      (updatedRunning input priorPcValid selectedNext),
    derivedOutput_recursive_holds machine input iterationPositive priorPcValid
      priorPublicInput selectedNext selectedStructures selectedNifs⟩

/-- The canonical recursive digest belongs to the public paper relation. -/
theorem recursive_paperFPrimeStep
    {family : NifsFamily VerifierKey Structure Assignment PublicInput Point
      Evaluation Commitment Scalar Challenge Value relation params slotCount}
    (machine : Machine VerifierKey Digest State Witness Structure PublicInput
      Point Evaluation Commitment params slotCount)
    (input : Input VerifierKey State Witness Structure PublicInput Point
      Evaluation Commitment params slotCount)
    (iterationPositive : 0 < input.iteration)
    (priorPcValid : InRange slotCount input.priorPc)
    (priorPublicInput : input.fresh.publicInput =
      machine.encodeInstance (machine.hash (priorHashPreimage input)))
    (selectedNext : RunningSlot Structure PublicInput Point Evaluation
      Commitment params)
    (selectedStructures :
      SelectedInputStructuresBound family input priorPcValid)
    (selectedNifs :
      (selectedVerifier family input priorPcValid).Transition
        (selectedNifsInput family input priorPcValid) selectedNext) :
    PaperFPrimeStep family machine (machine.control input.zi input.witness).index
      (derivedOutput machine input
        (updatedRunning input priorPcValid selectedNext)).x := by
  exact paperFPrimeStep_of_holds
    (derivedOutput_recursive_holds machine input iterationPositive priorPcValid
      priorPublicInput selectedNext selectedStructures selectedNifs)

/-- Base acceptance transfers default semantic validity to its exact output. -/
theorem defaultValid_transfers_to_base
    (machine : Machine VerifierKey Digest State Witness Structure PublicInput
      Point Evaluation Commitment params slotCount)
    {functionIndex : Fin slotCount}
    {input : Input VerifierKey State Witness Structure PublicInput Point
      Evaluation Commitment params slotCount}
    {output : Output Digest State Structure PublicInput Point Evaluation
      Commitment params slotCount}
    (defaultValid : DefaultValid (relation := relation) machine)
    (accepted : BaseHolds machine functionIndex input output) :
    RunningValid (relation := relation) output.runningNext := by
  rw [accepted.defaultRunning]
  exact defaultValid

/--
Base acceptance transfers both default validity and verifier-owned structures.

The result concerns the exact configured default vector selected by
`accepted.defaultRunning`; it does not identify the selected structure with
`enc_str(F'_j)`.
-/
theorem setupValid_transfers_to_base
    {family : NifsFamily VerifierKey Structure Assignment PublicInput Point
      Evaluation Commitment Scalar Challenge Value relation params slotCount}
    (machine : Machine VerifierKey Digest State Witness Structure PublicInput
      Point Evaluation Commitment params slotCount)
    {functionIndex : Fin slotCount}
    {input : Input VerifierKey State Witness Structure PublicInput Point
      Evaluation Commitment params slotCount}
    {output : Output Digest State Structure PublicInput Point Evaluation
      Commitment params slotCount}
    (setupValid : SetupValid family machine input.verifierKey)
    (accepted : BaseHolds machine functionIndex input output) :
    RunningValid (relation := relation) output.runningNext /\
      RunningStructuresBound family input.verifierKey output.runningNext := by
  constructor
  · exact defaultValid_transfers_to_base machine setupValid.1 accepted
  · rw [accepted.defaultRunning]
    exact setupValid.2

end

end Nightstream.Protocol.FPrime.Paper
