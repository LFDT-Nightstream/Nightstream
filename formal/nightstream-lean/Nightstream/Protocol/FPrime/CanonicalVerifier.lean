import Nightstream.HyperNova.Construction2.Paper

/-!
Canonical executable verifier for HyperNova Construction 2's augmented
function.

Owns: a compact evaluator over typed paper data; the base branch with no NIFS
call; the recursive branch with exactly one selected call to deterministic
`NIFS.V`; and extensional equality with the independently stated
Construction-2 transition.

Does not own: a concrete SuperNeo verifier, Fiat--Shamir, hash or commitment
internals, Rust, R1CS, lowering, row costs, or terminal relation checks.

Emits constraints: no.

The evaluator computes every public output field.  Its only equality tests are
the paper's dispatch, initial-state, and prior-public-link checks; it receives
no semantic acceptance proposition from its caller.
-/

namespace Nightstream.Protocol.FPrime.CanonicalVerifier

open Nightstream.HyperNova.NonInteractiveMultiFold
open Nightstream.HyperNova.Construction2.Paper

universe uKey uDigest uState uWitness uRunning uFresh uProof uEncoded

/-- Replace exactly the selected running slot. -/
def replaceSelected
    {Running : Type uRunning}
    {slotCount : Nat}
    (running : Fin slotCount -> Running)
    (selected : Fin slotCount)
    (value : Running) : Fin slotCount -> Running :=
  fun slot => if slot = selected then value else running slot

@[simp] theorem replaceSelected_self
    {Running : Type uRunning}
    {slotCount : Nat}
    (running : Fin slotCount -> Running)
    (selected : Fin slotCount)
    (value : Running) :
    replaceSelected running selected value selected = value := by
  simp [replaceSelected]

@[simp] theorem replaceSelected_of_ne
    {Running : Type uRunning}
    {slotCount : Nat}
    (running : Fin slotCount -> Running)
    (selected slot : Fin slotCount)
    (value : Running)
    (different : slot ≠ selected) :
    replaceSelected running selected value slot = running slot := by
  simp [replaceSelected, different]

/-- The unique public output determined by the selected augmented function
and the next running vector. -/
def outputFor
    {Key : Type uKey}
    {Digest : Type uDigest}
    {State : Type uState}
    {Witness : Type uWitness}
    {Running : Type uRunning}
    {Fresh : Type uFresh}
    {Proof : Type uProof}
    {Encoded : Type uEncoded}
    {slotCount : Nat}
    (setup : Setup Key Running Fresh Proof slotCount)
    (machine : Machine Key Digest State Witness Running Fresh Encoded slotCount)
    (functionIndex : Fin slotCount)
    (input : Input Key State Witness Running Fresh Proof slotCount)
    (runningNext : Fin slotCount -> Running) :
    Output Digest State Running slotCount :=
  let zNext := machine.step functionIndex input.zi input.witness
  {
    zNext := zNext
    runningNext := runningNext
    pcNext := functionIndex
    x := machine.hash {
      verifierKeys := setup.verifierKeys
      iteration := input.iteration + 1
      z0 := input.z0
      current := zNext
      running := runningNext
      pc := oneBased functionIndex
    }
  }

theorem outputFor_outputHolds
    {Key : Type uKey}
    {Digest : Type uDigest}
    {State : Type uState}
    {Witness : Type uWitness}
    {Running : Type uRunning}
    {Fresh : Type uFresh}
    {Proof : Type uProof}
    {Encoded : Type uEncoded}
    {slotCount : Nat}
    (setup : Setup Key Running Fresh Proof slotCount)
    (machine : Machine Key Digest State Witness Running Fresh Encoded slotCount)
    (functionIndex : Fin slotCount)
    (input : Input Key State Witness Running Fresh Proof slotCount)
    (runningNext : Fin slotCount -> Running) :
    OutputHolds setup machine input
      (outputFor setup machine functionIndex input runningNext) := by
  rfl

/-- Fieldwise paper equations uniquely determine the computed public output. -/
theorem output_eq_outputFor
    {Key : Type uKey}
    {Digest : Type uDigest}
    {State : Type uState}
    {Witness : Type uWitness}
    {Running : Type uRunning}
    {Fresh : Type uFresh}
    {Proof : Type uProof}
    {Encoded : Type uEncoded}
    {slotCount : Nat}
    (setup : Setup Key Running Fresh Proof slotCount)
    (machine : Machine Key Digest State Witness Running Fresh Encoded slotCount)
    (functionIndex : Fin slotCount)
    (input : Input Key State Witness Running Fresh Proof slotCount)
    (output : Output Digest State Running slotCount)
    (runningNext : Fin slotCount -> Running)
    (stateEquation : output.zNext =
      machine.step functionIndex input.zi input.witness)
    (runningEquation : output.runningNext = runningNext)
    (counterEquation : output.pcNext = functionIndex)
    (hashEquation : OutputHolds setup machine input output) :
    output = outputFor setup machine functionIndex input runningNext := by
  cases output with
  | mk outputState outputRunning outputCounter outputDigest =>
      dsimp at stateEquation runningEquation counterEquation hashEquation |- 
      subst outputState
      subst outputRunning
      subst outputCounter
      simp only [OutputHolds, nextHashPreimage] at hashEquation
      subst outputDigest
      rfl

/-- Compact typed evaluation of one `F'_j` invocation.  The base branch
performs no NIFS call.  The recursive branch invokes `NIFS.V` once at the
checked prior program counter and updates only that slot. -/
def eval
    {Key : Type uKey}
    {Digest : Type uDigest}
    {State : Type uState}
    {Witness : Type uWitness}
    {Running : Type uRunning}
    {Fresh : Type uFresh}
    {Proof : Type uProof}
    {Encoded : Type uEncoded}
    {slotCount : Nat}
    [DecidableEq State]
    [DecidableEq Encoded]
    (setup : Setup Key Running Fresh Proof slotCount)
    (machine : Machine Key Digest State Witness Running Fresh Encoded slotCount)
    (functionIndex : Fin slotCount)
    (input : Input Key State Witness Running Fresh Proof slotCount) :
    Option (Output Digest State Running slotCount) :=
  letI : Decidable (InRange slotCount input.priorPc) := by
    unfold InRange
    infer_instance
  if machine.control input.zi input.witness = functionIndex then
    if input.iteration = 0 then
      if input.z0 = input.zi then
        some (outputFor setup machine functionIndex input
          (fun _ => setup.defaultRunning))
      else
        none
    else if priorPcValid : InRange slotCount input.priorPc then
      let selected := selectedIndex priorPcValid
      if machine.freshPublic input.fresh =
          machine.encodeInstance (machine.hash (priorHashPreimage setup input)) then
        match setup.nifs.verify (setup.verifierKeys selected)
            (input.running selected) input.fresh input.nifsProof with
        | none => none
        | some folded =>
            some (outputFor setup machine functionIndex input
              (replaceSelected input.running selected folded))
      else
        none
    else
      none
  else
    none

/-- Computed acceptance of one canonical `F'` invocation. -/
def Accepts
    {Key : Type uKey}
    {Digest : Type uDigest}
    {State : Type uState}
    {Witness : Type uWitness}
    {Running : Type uRunning}
    {Fresh : Type uFresh}
    {Proof : Type uProof}
    {Encoded : Type uEncoded}
    {slotCount : Nat}
    [DecidableEq State]
    [DecidableEq Encoded]
    (setup : Setup Key Running Fresh Proof slotCount)
    (machine : Machine Key Digest State Witness Running Fresh Encoded slotCount)
    (functionIndex : Fin slotCount)
    (input : Input Key State Witness Running Fresh Proof slotCount)
    (output : Output Digest State Running slotCount) : Prop :=
  eval setup machine functionIndex input = some output

/-- Every computed acceptance realizes the independent Construction-2
transition. -/
theorem accepts_implies_transition
    {Key : Type uKey}
    {Digest : Type uDigest}
    {State : Type uState}
    {Witness : Type uWitness}
    {Running : Type uRunning}
    {Fresh : Type uFresh}
    {Proof : Type uProof}
    {Encoded : Type uEncoded}
    {slotCount : Nat}
    [DecidableEq State]
    [DecidableEq Encoded]
    (setup : Setup Key Running Fresh Proof slotCount)
    (machine : Machine Key Digest State Witness Running Fresh Encoded slotCount)
    (functionIndex : Fin slotCount)
    (input : Input Key State Witness Running Fresh Proof slotCount)
    (output : Output Digest State Running slotCount)
    (accepted : Accepts setup machine functionIndex input output) :
    Transition setup machine functionIndex input output := by
  unfold Accepts eval at accepted
  by_cases dispatch : machine.control input.zi input.witness = functionIndex
  · rw [if_pos dispatch] at accepted
    by_cases iterationZero : input.iteration = 0
    · rw [if_pos iterationZero] at accepted
      by_cases initialState : input.z0 = input.zi
      · rw [if_pos initialState] at accepted
        have outputEquation := (Option.some.inj accepted).symm
        subst output
        exact ⟨dispatch, rfl, rfl,
          outputFor_outputHolds setup machine functionIndex input
            (fun _ => setup.defaultRunning),
          Or.inl ⟨iterationZero, initialState, rfl⟩⟩
      · rw [if_neg initialState] at accepted
        contradiction
    · rw [if_neg iterationZero] at accepted
      by_cases priorPcValid : InRange slotCount input.priorPc
      · rw [dif_pos priorPcValid] at accepted
        let selected := selectedIndex priorPcValid
        by_cases priorPublicInput : machine.freshPublic input.fresh =
            machine.encodeInstance
              (machine.hash (priorHashPreimage setup input))
        · rw [if_pos priorPublicInput] at accepted
          change (match setup.nifs.verify
              (setup.verifierKeys selected) (input.running selected)
              input.fresh input.nifsProof with
            | none => none
            | some folded =>
                some (outputFor setup machine functionIndex input
                  (replaceSelected input.running selected folded))) =
            some output at accepted
          cases verifierResult : setup.nifs.verify
              (setup.verifierKeys selected) (input.running selected)
              input.fresh input.nifsProof with
          | none =>
              rw [verifierResult] at accepted
              cases accepted
          | some folded =>
              rw [verifierResult] at accepted
              have outputEquation := (Option.some.inj accepted).symm
              subst output
              refine ⟨dispatch, rfl, rfl,
                outputFor_outputHolds setup machine functionIndex input
                  (replaceSelected input.running selected folded), ?_⟩
              refine Or.inr ⟨priorPcValid,
                Nat.pos_of_ne_zero iterationZero, priorPublicInput, ?_, ?_⟩
              · change setup.nifs.verify (setup.verifierKeys selected)
                  (input.running selected) input.fresh input.nifsProof =
                    some ((replaceSelected input.running selected folded) selected)
                simpa only [replaceSelected_self] using verifierResult
              · intro slot different
                exact replaceSelected_of_ne input.running selected slot folded
                  different
        · rw [if_neg priorPublicInput] at accepted
          contradiction
      · rw [dif_neg priorPcValid] at accepted
        contradiction
  · rw [if_neg dispatch] at accepted
    contradiction

/-- Every independently valid Construction-2 transition is computed by the
canonical evaluator. -/
theorem transition_implies_accepts
    {Key : Type uKey}
    {Digest : Type uDigest}
    {State : Type uState}
    {Witness : Type uWitness}
    {Running : Type uRunning}
    {Fresh : Type uFresh}
    {Proof : Type uProof}
    {Encoded : Type uEncoded}
    {slotCount : Nat}
    [DecidableEq State]
    [DecidableEq Encoded]
    (setup : Setup Key Running Fresh Proof slotCount)
    (machine : Machine Key Digest State Witness Running Fresh Encoded slotCount)
    (functionIndex : Fin slotCount)
    (input : Input Key State Witness Running Fresh Proof slotCount)
    (output : Output Digest State Running slotCount)
    (transition : Transition setup machine functionIndex input output) :
    Accepts setup machine functionIndex input output := by
  rcases transition with
    ⟨dispatch, counterEquation, stateEquation, hashEquation, branch⟩
  unfold Accepts eval
  rw [if_pos dispatch]
  rcases branch with base | recursive
  · rcases base with ⟨iterationZero, initialState, runningEquation⟩
    rw [if_pos iterationZero, if_pos initialState]
    exact congrArg some (output_eq_outputFor setup machine functionIndex input
      output (fun _ => setup.defaultRunning) stateEquation runningEquation
      counterEquation hashEquation).symm
  · rcases recursive with
      ⟨priorPcValid, iterationPositive, priorPublicInput,
        selectedNifs, unchanged⟩
    have iterationNonzero : input.iteration ≠ 0 :=
      Nat.ne_of_gt iterationPositive
    rw [if_neg iterationNonzero, dif_pos priorPcValid,
      if_pos priorPublicInput]
    let selected := selectedIndex priorPcValid
    let folded := output.runningNext selected
    have runningEquation : output.runningNext =
        replaceSelected input.running selected folded := by
      funext slot
      by_cases same : slot = selected
      · subst slot
        simp [replaceSelected, folded]
      · simpa [replaceSelected, same] using unchanged slot same
    unfold Nightstream.HyperNova.NonInteractiveMultiFold.Accepts at selectedNifs
    rw [show setup.nifs.verify (setup.verifierKeys selected)
        (input.running selected) input.fresh input.nifsProof = some folded by
      simpa [selected, folded] using selectedNifs]
    exact congrArg some (output_eq_outputFor setup machine functionIndex input
      output (replaceSelected input.running selected folded) stateEquation
      runningEquation counterEquation hashEquation).symm

/-- The canonical executable verifier is extensionally equal to the frozen
paper transition. -/
theorem accepts_iff_transition
    {Key : Type uKey}
    {Digest : Type uDigest}
    {State : Type uState}
    {Witness : Type uWitness}
    {Running : Type uRunning}
    {Fresh : Type uFresh}
    {Proof : Type uProof}
    {Encoded : Type uEncoded}
    {slotCount : Nat}
    [DecidableEq State]
    [DecidableEq Encoded]
    (setup : Setup Key Running Fresh Proof slotCount)
    (machine : Machine Key Digest State Witness Running Fresh Encoded slotCount)
    (functionIndex : Fin slotCount)
    (input : Input Key State Witness Running Fresh Proof slotCount)
    (output : Output Digest State Running slotCount) :
    Accepts setup machine functionIndex input output <->
      Transition setup machine functionIndex input output := by
  constructor
  · exact accepts_implies_transition setup machine functionIndex input output
  · exact transition_implies_accepts setup machine functionIndex input output

end Nightstream.Protocol.FPrime.CanonicalVerifier
