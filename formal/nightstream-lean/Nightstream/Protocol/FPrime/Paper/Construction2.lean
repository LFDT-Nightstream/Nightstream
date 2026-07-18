import Nightstream.Protocol.FPrime.Paper

/-!
HyperNova Construction-2 outer relation over an abstract selected NIFS edge.

Assurance tier: model-level.

Owns: the exact public NIFS input `(U_i[pc_i], u_i)`, verifier-key/slot
selection of one transition predicate and expected relation structure, and the
base/recursive augmented-function split.

Does not own: a particular SuperNeo phase composition, parent caches,
certificates, Fiat--Shamir, Poseidon2, extraction, Rust, R1CS, costs, or row
removal.

Emits constraints: no.

Authority boundary: Construction 2 treats `NIFS.V` as an abstract public
transition. This module does the same. It does not define that transition as
the historical candidate `Nifs.PaperNifsTransition`. Concrete protocols must
refine their independent NIFS semantics into `Family.transition`; they may not
obtain that theorem by aliasing the two relations.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `fprime.paper.nifs.input` | select exactly one fresh claim and the `k` children at `pc_i` | computed | `selectedInput` |
| `fprime.paper.nifs.structure` | selected fresh/running sources use the key/slot-owned structure | checked | `SelectedStructuresBound` |
| `fprime.paper.nifs.transition` | selected public input reaches exactly the installed child vector | abstract protocol boundary | `Family.transition` |
| `fprime.paper.base` | `i = 0`, initial state, default vector, application, and output hash | checked/computed | `Paper.BaseHolds` |
| `fprime.paper.recursive` | prior link, selected fold, inactive copies, application, and output hash | checked/computed | `RecursiveHolds` |
| `fprime.paper.branch` | accept exactly the base or recursive case | relation | `Holds` |
-/

namespace Nightstream.Protocol.FPrime.Paper.Construction2

open Nightstream.SuperNeo

universe uStructure uPublicInput uPoint uEvaluation uCommitment uState
  uWitness uVerifierKey uDigest

/-- Exact public input passed to one selected NIFS verifier. -/
structure SelectedInput
    (Structure : Type uStructure)
    (PublicInput : Type uPublicInput)
    (Point : Type uPoint)
    (Evaluation : Type uEvaluation)
    (Commitment : Type uCommitment)
    (params : GlobalParams) where
  fresh : CCS.Instance Structure PublicInput Commitment
  running : Paper.RunningSlot
    Structure PublicInput Point Evaluation Commitment params

/-- Select only the public NIFS arguments named in Construction 2. -/
def selectedInput
    {VerifierKey : Type uVerifierKey}
    {State : Type uState}
    {Witness : Type uWitness}
    {Structure : Type uStructure}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {params : GlobalParams}
    {slotCount : Nat}
    (input : Paper.Input VerifierKey State Witness Structure PublicInput Point
      Evaluation Commitment params slotCount)
    (slot : Fin slotCount) :
    SelectedInput Structure PublicInput Point Evaluation Commitment params where
  fresh := input.fresh
  running := input.running slot

@[simp] theorem selectedInput_fresh
    {VerifierKey : Type uVerifierKey}
    {State : Type uState}
    {Witness : Type uWitness}
    {Structure : Type uStructure}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {params : GlobalParams}
    {slotCount : Nat}
    (input : Paper.Input VerifierKey State Witness Structure PublicInput Point
      Evaluation Commitment params slotCount)
    (slot : Fin slotCount) :
    (selectedInput input slot).fresh = input.fresh := rfl

@[simp] theorem selectedInput_running
    {VerifierKey : Type uVerifierKey}
    {State : Type uState}
    {Witness : Type uWitness}
    {Structure : Type uStructure}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {params : GlobalParams}
    {slotCount : Nat}
    (input : Paper.Input VerifierKey State Witness Structure PublicInput Point
      Evaluation Commitment params slotCount)
    (slot : Fin slotCount) :
    (selectedInput input slot).running = input.running slot := rfl

/-- The only NIFS interface used by the augmented-function relation.

The verifier key and typed slot determine both the expected relation structure
and the public transition. Internal proof messages and verifier state are
existential inside the chosen transition, not additional outer authority. -/
structure Family
    (VerifierKey : Type uVerifierKey)
    (Structure : Type uStructure)
    (PublicInput : Type uPublicInput)
    (Point : Type uPoint)
    (Evaluation : Type uEvaluation)
    (Commitment : Type uCommitment)
    (params : GlobalParams)
    (slotCount : Nat) where
  expectedStructure : VerifierKey -> Fin slotCount -> Structure
  transition :
    VerifierKey -> Fin slotCount ->
      SelectedInput Structure PublicInput Point Evaluation Commitment params ->
      Paper.RunningSlot Structure PublicInput Point Evaluation Commitment
        params ->
      Prop

/-- Exact source-structure binding for the selected public NIFS input. -/
structure SelectedStructuresBound
    {VerifierKey : Type uVerifierKey}
    {State : Type uState}
    {Witness : Type uWitness}
    {Structure : Type uStructure}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {params : GlobalParams}
    {slotCount : Nat}
    (family : Family VerifierKey Structure PublicInput Point Evaluation
      Commitment params slotCount)
    (input : Paper.Input VerifierKey State Witness Structure PublicInput Point
      Evaluation Commitment params slotCount)
    (slot : Fin slotCount) : Prop where
  fresh : input.fresh.constraintSystem =
    family.expectedStructure input.verifierKey slot
  running : forall child,
    (input.running slot child).constraintSystem =
      family.expectedStructure input.verifierKey slot

/-- Recursive Construction-2 branch over only the abstract selected edge. -/
structure RecursiveHolds
    {VerifierKey : Type uVerifierKey}
    {Digest : Type uDigest}
    {State : Type uState}
    {Witness : Type uWitness}
    {Structure : Type uStructure}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {params : GlobalParams}
    {slotCount : Nat}
    (family : Family VerifierKey Structure PublicInput Point Evaluation
      Commitment params slotCount)
    (machine : Paper.Machine VerifierKey Digest State Witness Structure
      PublicInput Point Evaluation Commitment params slotCount)
    (functionIndex : Fin slotCount)
    (input : Paper.Input VerifierKey State Witness Structure PublicInput Point
      Evaluation Commitment params slotCount)
    (output : Paper.Output Digest State Structure PublicInput Point Evaluation
      Commitment params slotCount) : Prop where
  iterationPositive : 0 < input.iteration
  priorPcValid : Paper.InRange slotCount input.priorPc
  priorPublicInput : input.fresh.publicInput =
    machine.encodeInstance (machine.hash (Paper.priorHashPreimage input))
  application : Paper.ApplicationHolds machine functionIndex input output
  selectedStructures :
    SelectedStructuresBound family input
      (Paper.selectedIndex priorPcValid)
  selectedNifs :
    family.transition input.verifierKey (Paper.selectedIndex priorPcValid)
      (selectedInput input (Paper.selectedIndex priorPcValid))
      (output.runningNext (Paper.selectedIndex priorPcValid))
  unchanged : forall slot, slot ≠ Paper.selectedIndex priorPcValid ->
    output.runningNext slot = input.running slot
  outputHash : Paper.OutputHolds machine input output

/-- Paper-faithful outer relation with no commitment to NIFS internals. -/
inductive Holds
    {VerifierKey : Type uVerifierKey}
    {Digest : Type uDigest}
    {State : Type uState}
    {Witness : Type uWitness}
    {Structure : Type uStructure}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {params : GlobalParams}
    {slotCount : Nat}
    (family : Family VerifierKey Structure PublicInput Point Evaluation
      Commitment params slotCount)
    (machine : Paper.Machine VerifierKey Digest State Witness Structure
      PublicInput Point Evaluation Commitment params slotCount)
    (functionIndex : Fin slotCount)
    (input : Paper.Input VerifierKey State Witness Structure PublicInput Point
      Evaluation Commitment params slotCount)
    (output : Paper.Output Digest State Structure PublicInput Point Evaluation
      Commitment params slotCount) : Prop where
  | base (accepted : Paper.BaseHolds machine functionIndex input output)
  | recursive (accepted :
      RecursiveHolds family machine functionIndex input output)

/-- Project one full accepted execution to its public digest. -/
def PublicStep
    {VerifierKey : Type uVerifierKey}
    {Digest : Type uDigest}
    {State : Type uState}
    {Witness : Type uWitness}
    {Structure : Type uStructure}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {params : GlobalParams}
    {slotCount : Nat}
    (family : Family VerifierKey Structure PublicInput Point Evaluation
      Commitment params slotCount)
    (machine : Paper.Machine VerifierKey Digest State Witness Structure
      PublicInput Point Evaluation Commitment params slotCount)
    (functionIndex : Fin slotCount)
    (x : Digest) : Prop :=
  exists input output,
    Holds family machine functionIndex input output /\ output.x = x

end Nightstream.Protocol.FPrime.Paper.Construction2
