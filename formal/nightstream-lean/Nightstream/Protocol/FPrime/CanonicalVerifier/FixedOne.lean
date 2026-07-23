import Nightstream.Protocol.FPrime.CanonicalVerifier

/-!
Canonical executable verifier specialized to HyperNova Construction 2 with one
augmented function.

Owns:
- the sole typed slot and the one-based prior counter derived from it;
- a smaller paper input with no raw `priorPc`;
- a direct evaluator with no dispatch or counter-range branch;
- equivalence with the generic canonical verifier and frozen paper transition.

Does not own: ConcretePhi81, a concrete NIFS, Fiat--Shamir, hashes,
commitments, Rust, R1CS, lowering, or costs.

The recursive branch still performs exactly one `NIFS.V` call.  This module
removes only obligations made tautological by the `Fin 1` carrier.

Emits constraints: no.
-/

namespace Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne

open Nightstream.HyperNova.NonInteractiveMultiFold
open Nightstream.HyperNova.Construction2.Paper

universe uKey uDigest uState uWitness uRunning uFresh uProof uEncoded

/-- The sole selected augmented-function slot. -/
def selected : Fin 1 :=
  ⟨0, by decide⟩

@[simp] theorem selected_val :
    selected.val = 0 :=
  rfl

@[simp] theorem oneBased_selected :
    oneBased selected = 1 :=
  rfl

/-- Every typed one-slot index is the selected index. -/
@[simp] theorem fin_eq_selected (slot : Fin 1) :
    slot = selected :=
  Subsingleton.elim slot selected

/-- Dispatch is a type-level fact in the one-function profile. -/
@[simp] theorem control_eq_selected
    {Key : Type uKey}
    {Digest : Type uDigest}
    {State : Type uState}
    {Witness : Type uWitness}
    {Running : Type uRunning}
    {Fresh : Type uFresh}
    {Encoded : Type uEncoded}
    (machine : Machine Key Digest State Witness Running Fresh Encoded 1)
    (state : State)
    (witness : Witness) :
    machine.control state witness = selected :=
  fin_eq_selected (machine.control state witness)

/-- The paper range `[1, 1]` contains only the canonical counter. -/
theorem priorPc_eq_one_of_inRange
    {pc : Nat}
    (valid : InRange 1 pc) :
    pc = 1 :=
  Nat.le_antisymm valid.2 valid.1

/-- Selection from any valid one-slot prior counter is the sole slot. -/
@[simp] theorem selectedIndex_eq_selected
    {pc : Nat}
    (valid : InRange 1 pc) :
    selectedIndex valid = selected :=
  fin_eq_selected (selectedIndex valid)

/-- Paper advice after removing the raw prior program counter. -/
structure Input
    (State : Type uState)
    (Witness : Type uWitness)
    (Running : Type uRunning)
    (Fresh : Type uFresh)
    (Proof : Type uProof) where
  iteration : Nat
  z0 : State
  zi : State
  running : Fin 1 -> Running
  fresh : Fresh
  witness : Witness
  nifsProof : Proof

namespace Input

/-- Reinstall the sole verifier-derived one-based prior counter. -/
def toGeneric
    {Key : Type uKey}
    {State : Type uState}
    {Witness : Type uWitness}
    {Running : Type uRunning}
    {Fresh : Type uFresh}
    {Proof : Type uProof}
    (input : Input State Witness Running Fresh Proof) :
    Nightstream.HyperNova.Construction2.Paper.Input
      Key State Witness Running Fresh Proof 1 where
  iteration := input.iteration
  z0 := input.z0
  zi := input.zi
  running := input.running
  fresh := input.fresh
  priorPc := oneBased selected
  witness := input.witness
  nifsProof := input.nifsProof

/-- Erase only the raw prior counter from a generic one-slot input. -/
def erase
    {Key : Type uKey}
    {State : Type uState}
    {Witness : Type uWitness}
    {Running : Type uRunning}
    {Fresh : Type uFresh}
    {Proof : Type uProof}
    (input :
      Nightstream.HyperNova.Construction2.Paper.Input
        Key State Witness Running Fresh Proof 1) :
    Input State Witness Running Fresh Proof where
  iteration := input.iteration
  z0 := input.z0
  zi := input.zi
  running := input.running
  fresh := input.fresh
  witness := input.witness
  nifsProof := input.nifsProof

@[simp] theorem toGeneric_priorPc
    {Key : Type uKey}
    {State : Type uState}
    {Witness : Type uWitness}
    {Running : Type uRunning}
    {Fresh : Type uFresh}
    {Proof : Type uProof}
    (input : Input State Witness Running Fresh Proof) :
    (input.toGeneric (Key := Key)).priorPc = 1 :=
  rfl

/-- The reconstructed prior counter always passes the paper range check. -/
theorem toGeneric_priorPcValid
    {Key : Type uKey}
    {State : Type uState}
    {Witness : Type uWitness}
    {Running : Type uRunning}
    {Fresh : Type uFresh}
    {Proof : Type uProof}
    (input : Input State Witness Running Fresh Proof) :
    InRange 1 (input.toGeneric (Key := Key)).priorPc :=
  by
    change InRange 1 1
    exact ⟨by decide, by decide⟩

/-- Erasing a canonical input recovers the smaller input exactly. -/
@[simp] theorem erase_toGeneric
    {Key : Type uKey}
    {State : Type uState}
    {Witness : Type uWitness}
    {Running : Type uRunning}
    {Fresh : Type uFresh}
    {Proof : Type uProof}
    (input : Input State Witness Running Fresh Proof) :
    erase (input.toGeneric (Key := Key)) = input := by
  cases input
  rfl

/-- A generic input round-trips exactly when its erased counter already is the
canonical verifier-derived value. -/
theorem toGeneric_erase_of_priorPc
    {Key : Type uKey}
    {State : Type uState}
    {Witness : Type uWitness}
    {Running : Type uRunning}
    {Fresh : Type uFresh}
    {Proof : Type uProof}
    (input :
      Nightstream.HyperNova.Construction2.Paper.Input
        Key State Witness Running Fresh Proof 1)
    (canonical : input.priorPc = oneBased selected) :
    (erase input).toGeneric = input := by
  cases input with
  | mk iteration z0 zi running fresh priorPc witness nifsProof =>
      dsimp [erase, toGeneric] at canonical ⊢
      rw [canonical]

/-- In-range generic one-slot inputs satisfy the round-trip premise. -/
theorem toGeneric_erase_of_inRange
    {Key : Type uKey}
    {State : Type uState}
    {Witness : Type uWitness}
    {Running : Type uRunning}
    {Fresh : Type uFresh}
    {Proof : Type uProof}
    (input :
      Nightstream.HyperNova.Construction2.Paper.Input
        Key State Witness Running Fresh Proof 1)
    (valid : InRange 1 input.priorPc) :
    (erase input).toGeneric = input := by
  apply toGeneric_erase_of_priorPc input
  have counter : input.priorPc = 1 :=
    priorPc_eq_one_of_inRange valid
  simpa using counter

end Input

/-- Replacing the selected entry in a one-slot running vector is simply a
constant one-slot vector. -/
@[simp] theorem replaceSelected_eq_const
    {Running : Type uRunning}
    (running : Fin 1 -> Running)
    (value : Running) :
    Nightstream.Protocol.FPrime.CanonicalVerifier.replaceSelected
      running selected value = fun _ => value := by
  funext slot
  have slotEq : slot = selected := fin_eq_selected slot
  subst slot
  simp

/-- Direct fixed-one prior-link preimage. -/
def priorHashPreimage
    {Key : Type uKey}
    {State : Type uState}
    {Witness : Type uWitness}
    {Running : Type uRunning}
    {Fresh : Type uFresh}
    {Proof : Type uProof}
    (setup : Setup Key Running Fresh Proof 1)
    (input : Input State Witness Running Fresh Proof) :
    HashPreimage Key State Running 1 where
  verifierKeys := setup.verifierKeys
  iteration := input.iteration
  z0 := input.z0
  current := input.zi
  running := input.running
  pc := oneBased selected

@[simp] theorem priorHashPreimage_eq_generic
    {Key : Type uKey}
    {State : Type uState}
    {Witness : Type uWitness}
    {Running : Type uRunning}
    {Fresh : Type uFresh}
    {Proof : Type uProof}
    (setup : Setup Key Running Fresh Proof 1)
    (input : Input State Witness Running Fresh Proof) :
    priorHashPreimage setup input =
      Nightstream.HyperNova.Construction2.Paper.priorHashPreimage
        setup (input.toGeneric (Key := Key)) :=
  rfl

/-- Direct fixed-one public output. -/
def outputFor
    {Key : Type uKey}
    {Digest : Type uDigest}
    {State : Type uState}
    {Witness : Type uWitness}
    {Running : Type uRunning}
    {Fresh : Type uFresh}
    {Proof : Type uProof}
    {Encoded : Type uEncoded}
    (setup : Setup Key Running Fresh Proof 1)
    (machine : Machine Key Digest State Witness Running Fresh Encoded 1)
    (input : Input State Witness Running Fresh Proof)
    (runningNext : Fin 1 -> Running) :
    Output Digest State Running 1 :=
  let zNext := machine.step selected input.zi input.witness
  {
    zNext := zNext
    runningNext := runningNext
    pcNext := selected
    x := machine.hash {
      verifierKeys := setup.verifierKeys
      iteration := input.iteration + 1
      z0 := input.z0
      current := zNext
      running := runningNext
      pc := oneBased selected
    }
  }

@[simp] theorem outputFor_eq_generic
    {Key : Type uKey}
    {Digest : Type uDigest}
    {State : Type uState}
    {Witness : Type uWitness}
    {Running : Type uRunning}
    {Fresh : Type uFresh}
    {Proof : Type uProof}
    {Encoded : Type uEncoded}
    (setup : Setup Key Running Fresh Proof 1)
    (machine : Machine Key Digest State Witness Running Fresh Encoded 1)
    (input : Input State Witness Running Fresh Proof)
    (runningNext : Fin 1 -> Running) :
    outputFor setup machine input runningNext =
      Nightstream.Protocol.FPrime.CanonicalVerifier.outputFor
        setup machine selected (input.toGeneric (Key := Key)) runningNext :=
  rfl

/-- Direct one-slot evaluator.  Dispatch and prior-counter range checks are
absent because their propositions follow from the carrier. -/
def eval
    {Key : Type uKey}
    {Digest : Type uDigest}
    {State : Type uState}
    {Witness : Type uWitness}
    {Running : Type uRunning}
    {Fresh : Type uFresh}
    {Proof : Type uProof}
    {Encoded : Type uEncoded}
    [DecidableEq State]
    [DecidableEq Encoded]
    (setup : Setup Key Running Fresh Proof 1)
    (machine : Machine Key Digest State Witness Running Fresh Encoded 1)
    (input : Input State Witness Running Fresh Proof) :
    Option (Output Digest State Running 1) :=
  if input.iteration = 0 then
    if input.z0 = input.zi then
      some (outputFor setup machine input (fun _ => setup.defaultRunning))
    else
      none
  else if machine.freshPublic input.fresh =
      machine.encodeInstance (machine.hash (priorHashPreimage setup input)) then
    match setup.nifs.verify (setup.verifierKeys selected)
        (input.running selected) input.fresh input.nifsProof with
    | none => none
    | some folded =>
        some (outputFor setup machine input (fun _ => folded))
  else
    none

/-- The direct specialization is extensionally equal to the generic
paper-only evaluator after reconstructing the sole prior counter. -/
theorem eval_eq_generic
    {Key : Type uKey}
    {Digest : Type uDigest}
    {State : Type uState}
    {Witness : Type uWitness}
    {Running : Type uRunning}
    {Fresh : Type uFresh}
    {Proof : Type uProof}
    {Encoded : Type uEncoded}
    [DecidableEq State]
    [DecidableEq Encoded]
    (setup : Setup Key Running Fresh Proof 1)
    (machine : Machine Key Digest State Witness Running Fresh Encoded 1)
    (input : Input State Witness Running Fresh Proof) :
    eval setup machine input =
      Nightstream.Protocol.FPrime.CanonicalVerifier.eval
        setup machine selected (input.toGeneric (Key := Key)) := by
  unfold eval Nightstream.Protocol.FPrime.CanonicalVerifier.eval
  simp only [Input.toGeneric]
  rw [if_pos (control_eq_selected machine input.zi input.witness)]
  by_cases iterationZero : input.iteration = 0
  · rw [if_pos iterationZero, if_pos iterationZero]
    by_cases initialState : input.z0 = input.zi
    · rw [if_pos initialState, if_pos initialState]
      rfl
    · rw [if_neg initialState, if_neg initialState]
  · rw [if_neg iterationZero, if_neg iterationZero]
    let priorPcValid : InRange 1 (oneBased selected) := by
      simpa only [oneBased_selected] using
        Input.toGeneric_priorPcValid (Key := Key) input
    rw [dif_pos priorPcValid]
    simp only [selectedIndex_eq_selected]
    by_cases priorPublicInput : machine.freshPublic input.fresh =
        machine.encodeInstance
          (machine.hash (priorHashPreimage setup input))
    · rw [if_pos priorPublicInput]
      have genericPriorPublicInput : machine.freshPublic input.fresh =
          machine.encodeInstance (machine.hash
            (Nightstream.HyperNova.Construction2.Paper.priorHashPreimage setup
              (input.toGeneric (Key := Key)))) := by
        simpa only [priorHashPreimage_eq_generic] using priorPublicInput
      simp only [Input.toGeneric] at genericPriorPublicInput
      rw [if_pos genericPriorPublicInput]
      cases verifierResult : setup.nifs.verify (setup.verifierKeys selected)
          (input.running selected) input.fresh input.nifsProof with
      | none =>
          rfl
      | some folded =>
          simp only [outputFor_eq_generic, replaceSelected_eq_const,
            Input.toGeneric]
    · rw [if_neg priorPublicInput]
      have genericPriorPublicInput : ¬machine.freshPublic input.fresh =
          machine.encodeInstance (machine.hash
            (Nightstream.HyperNova.Construction2.Paper.priorHashPreimage setup
              (input.toGeneric (Key := Key)))) := by
        simpa only [priorHashPreimage_eq_generic] using priorPublicInput
      simp only [Input.toGeneric] at genericPriorPublicInput
      rw [if_neg genericPriorPublicInput]

/-- Computed acceptance by the direct fixed-one evaluator. -/
def Accepts
    {Key : Type uKey}
    {Digest : Type uDigest}
    {State : Type uState}
    {Witness : Type uWitness}
    {Running : Type uRunning}
    {Fresh : Type uFresh}
    {Proof : Type uProof}
    {Encoded : Type uEncoded}
    [DecidableEq State]
    [DecidableEq Encoded]
    (setup : Setup Key Running Fresh Proof 1)
    (machine : Machine Key Digest State Witness Running Fresh Encoded 1)
    (input : Input State Witness Running Fresh Proof)
    (output : Output Digest State Running 1) : Prop :=
  eval setup machine input = some output

/-- Fixed-one acceptance is exactly generic canonical acceptance after the
verifier reconstructs the omitted prior counter. -/
theorem accepts_iff_generic
    {Key : Type uKey}
    {Digest : Type uDigest}
    {State : Type uState}
    {Witness : Type uWitness}
    {Running : Type uRunning}
    {Fresh : Type uFresh}
    {Proof : Type uProof}
    {Encoded : Type uEncoded}
    [DecidableEq State]
    [DecidableEq Encoded]
    (setup : Setup Key Running Fresh Proof 1)
    (machine : Machine Key Digest State Witness Running Fresh Encoded 1)
    (input : Input State Witness Running Fresh Proof)
    (output : Output Digest State Running 1) :
    Accepts setup machine input output <->
      Nightstream.Protocol.FPrime.CanonicalVerifier.Accepts
        setup machine selected (input.toGeneric (Key := Key)) output := by
  simp only [Accepts,
    Nightstream.Protocol.FPrime.CanonicalVerifier.Accepts,
    eval_eq_generic]

/-- The specialized executable verifier is extensionally equal to the frozen
Construction-2 transition at the sole function index. -/
theorem accepts_iff_transition
    {Key : Type uKey}
    {Digest : Type uDigest}
    {State : Type uState}
    {Witness : Type uWitness}
    {Running : Type uRunning}
    {Fresh : Type uFresh}
    {Proof : Type uProof}
    {Encoded : Type uEncoded}
    [DecidableEq State]
    [DecidableEq Encoded]
    (setup : Setup Key Running Fresh Proof 1)
    (machine : Machine Key Digest State Witness Running Fresh Encoded 1)
    (input : Input State Witness Running Fresh Proof)
    (output : Output Digest State Running 1) :
    Accepts setup machine input output <->
      Transition setup machine selected
        (input.toGeneric (Key := Key)) output := by
  rw [accepts_iff_generic]
  exact
    Nightstream.Protocol.FPrime.CanonicalVerifier.accepts_iff_transition
      setup machine selected (input.toGeneric (Key := Key)) output

end Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne
