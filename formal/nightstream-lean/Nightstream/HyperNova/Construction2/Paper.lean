import Lean.Elab.Tactic.Omega
import Nightstream.HyperNova.NonInteractiveMultiFold

/-!
Paper-owned HyperNova Construction-2 and augmented-function semantics.

Source: HyperNova Section 6.3, Construction 2, and Appendix H.3.

Owns: the exact base/recursive split of `F'_j`, the one-based prior program
counter, deterministic dispatch/application, the prior hash link, one selected
call to the function-valued `NIFS.V`, unchanged inactive slots, the next hash
preimage, and the terminal verifier boundary.

Does not own: SuperNeo, a concrete hash or instance encoder, Fiat--Shamir,
commitment security, memory optimizations, delayed projection state, Rust,
R1CS, or costs.

Emits constraints: no.

The base branch performs no NIFS fold.  The recursive branch consumes exactly
one prover message.  The terminal verifier validates the current running and
fresh relations but performs no additional fold, matching Construction 2.
-/

namespace Nightstream.HyperNova.Construction2.Paper

open Nightstream.HyperNova.NonInteractiveMultiFold

universe uKey uDigest uState uWitness uRunning uFresh uProof uEncoded
  uRunningWitness uFreshWitness

/-- The paper's explicit `1 <= pc <= ell` check. -/
def InRange (slotCount pc : Nat) : Prop :=
  1 <= pc /\ pc <= slotCount

/-- Convert a checked one-based program counter into its selected slot. -/
def selectedIndex
    {slotCount pc : Nat}
    (valid : InRange slotCount pc) : Fin slotCount :=
  ⟨pc - 1, by
    rcases valid with ⟨lower, upper⟩
    omega⟩

/-- Canonical one-based representation of a selected slot. -/
def oneBased {slotCount : Nat} (slot : Fin slotCount) : Nat :=
  slot.val + 1

@[simp] theorem selectedIndex_oneBased
    {slotCount : Nat}
    (slot : Fin slotCount) :
    selectedIndex (show InRange slotCount (oneBased slot) by
      simp only [oneBased]
      constructor <;> omega) = slot := by
  apply Fin.ext
  simp [selectedIndex, oneBased]

/-- The exact typed preimage used by Construction 2 for both the prior link
and the next public digest. -/
structure HashPreimage
    (Key : Type uKey)
    (State : Type uState)
    (Running : Type uRunning)
    (slotCount : Nat) where
  verifierKeys : Fin slotCount -> Key
  iteration : Nat
  z0 : State
  current : State
  running : Fin slotCount -> Running
  pc : Nat

/-- Verifier-owned NIFS keys and the explicit paper default vector. -/
structure Setup
    (Key : Type uKey)
    (Running : Type uRunning)
    (Fresh : Type uFresh)
    (Proof : Type uProof)
    (slotCount : Nat) where
  verifierKeys : Fin slotCount -> Key
  nifs : Verifier Key Running Fresh Proof
  /-- HyperNova Definition 12's single universal default instance `u_perp`.
  The base branch replicates this value into every running slot. -/
  defaultRunning : Running

/-- Deterministic application, public-link encoding, and hash operations used
by the augmented functions. -/
structure Machine
    (Key : Type uKey)
    (Digest : Type uDigest)
    (State : Type uState)
    (Witness : Type uWitness)
    (Running : Type uRunning)
    (Fresh : Type uFresh)
    (Encoded : Type uEncoded)
    (slotCount : Nat) where
  control : State -> Witness -> Fin slotCount
  step : Fin slotCount -> State -> Witness -> State
  freshPublic : Fresh -> Encoded
  encodeInstance : Digest -> Encoded
  hash : HashPreimage Key State Running slotCount -> Digest

/-- All nondeterministic advice to one invocation of `F'_j`. -/
structure Input
    (Key : Type uKey)
    (State : Type uState)
    (Witness : Type uWitness)
    (Running : Type uRunning)
    (Fresh : Type uFresh)
    (Proof : Type uProof)
    (slotCount : Nat) where
  iteration : Nat
  z0 : State
  zi : State
  running : Fin slotCount -> Running
  fresh : Fresh
  priorPc : Nat
  witness : Witness
  nifsProof : Proof

/-- Every public output of one augmented-function invocation is computed. -/
structure Output
    (Digest : Type uDigest)
    (State : Type uState)
    (Running : Type uRunning)
    (slotCount : Nat) where
  zNext : State
  runningNext : Fin slotCount -> Running
  pcNext : Fin slotCount
  x : Digest

/-- Exact prior-link preimage from Construction 2 step 4(b). -/
def priorHashPreimage
    {Key : Type uKey}
    {State : Type uState}
    {Witness : Type uWitness}
    {Running : Type uRunning}
    {Fresh : Type uFresh}
    {Proof : Type uProof}
    {slotCount : Nat}
    (setup : Setup Key Running Fresh Proof slotCount)
    (input : Input Key State Witness Running Fresh Proof slotCount) :
    HashPreimage Key State Running slotCount where
  verifierKeys := setup.verifierKeys
  iteration := input.iteration
  z0 := input.z0
  current := input.zi
  running := input.running
  pc := input.priorPc

/-- Exact next-output preimage from Construction 2 step 5. -/
def nextHashPreimage
    {Key : Type uKey}
    {Digest : Type uDigest}
    {State : Type uState}
    {Witness : Type uWitness}
    {Running : Type uRunning}
    {Fresh : Type uFresh}
    {Proof : Type uProof}
    {slotCount : Nat}
    (setup : Setup Key Running Fresh Proof slotCount)
    (input : Input Key State Witness Running Fresh Proof slotCount)
    (output : Output Digest State Running slotCount) :
    HashPreimage Key State Running slotCount where
  verifierKeys := setup.verifierKeys
  iteration := input.iteration + 1
  z0 := input.z0
  current := output.zNext
  running := output.runningNext
  pc := oneBased output.pcNext

/-- The selected augmented function is exactly the control result and computes
the advertised next application state. -/
structure ApplicationHolds
    {Key : Type uKey}
    {Digest : Type uDigest}
    {State : Type uState}
    {Witness : Type uWitness}
    {Running : Type uRunning}
    {Fresh : Type uFresh}
    {Proof : Type uProof}
    {Encoded : Type uEncoded}
    {slotCount : Nat}
    (machine : Machine Key Digest State Witness Running Fresh Encoded slotCount)
    (functionIndex : Fin slotCount)
    (input : Input Key State Witness Running Fresh Proof slotCount)
    (output : Output Digest State Running slotCount) : Prop where
  dispatch : machine.control input.zi input.witness = functionIndex
  pcNext : output.pcNext = functionIndex
  application : output.zNext =
    machine.step functionIndex input.zi input.witness

/-- The public digest is computed from the complete typed next preimage. -/
def OutputHolds
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
    (input : Input Key State Witness Running Fresh Proof slotCount)
    (output : Output Digest State Running slotCount) : Prop :=
  output.x = machine.hash (nextHashPreimage setup input output)

/-- Construction 2 base branch: no NIFS call, exact initial-state check, and
installation of the explicit default vector. -/
structure BaseHolds
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
    (output : Output Digest State Running slotCount) : Prop where
  iterationZero : input.iteration = 0
  initialState : input.z0 = input.zi
  application : ApplicationHolds machine functionIndex input output
  defaultRunning : output.runningNext = fun _ => setup.defaultRunning
  outputHash : OutputHolds setup machine input output

/-- Construction 2 recursive branch with exactly one computed `NIFS.V` call. -/
structure RecursiveHolds
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
    (output : Output Digest State Running slotCount) : Prop where
  iterationPositive : 0 < input.iteration
  priorPcValid : InRange slotCount input.priorPc
  priorPublicInput : machine.freshPublic input.fresh =
    machine.encodeInstance (machine.hash (priorHashPreimage setup input))
  application : ApplicationHolds machine functionIndex input output
  selectedNifs : Accepts setup.nifs
    (setup.verifierKeys (selectedIndex priorPcValid))
    (input.running (selectedIndex priorPcValid)) input.fresh input.nifsProof
    (output.runningNext (selectedIndex priorPcValid))
  unchanged : forall slot, slot ≠ selectedIndex priorPcValid ->
    output.runningNext slot = input.running slot
  outputHash : OutputHolds setup machine input output

/-- Independently expanded Construction-2 transition equations for one
augmented function.  This is the frozen semantic target; it is not defined in
terms of the helper branch structures below. -/
def Transition
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
    (output : Output Digest State Running slotCount) : Prop :=
  machine.control input.zi input.witness = functionIndex /\
  output.pcNext = functionIndex /\
  output.zNext = machine.step functionIndex input.zi input.witness /\
  output.x = machine.hash (nextHashPreimage setup input output) /\
  ((input.iteration = 0 /\
      input.z0 = input.zi /\
      output.runningNext = fun _ => setup.defaultRunning) \/
    exists priorPcValid : InRange slotCount input.priorPc,
      0 < input.iteration /\
      machine.freshPublic input.fresh =
        machine.encodeInstance (machine.hash (priorHashPreimage setup input)) /\
      Accepts setup.nifs
        (setup.verifierKeys (selectedIndex priorPcValid))
        (input.running (selectedIndex priorPcValid)) input.fresh input.nifsProof
        (output.runningNext (selectedIndex priorPcValid)) /\
      forall slot, slot ≠ selectedIndex priorPcValid ->
        output.runningNext slot = input.running slot)

/-- `F'_j` accepts exactly one paper branch. -/
inductive Holds
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
    (output : Output Digest State Running slotCount) : Prop where
  | base : BaseHolds setup machine functionIndex input output ->
      Holds setup machine functionIndex input output
  | recursive : RecursiveHolds setup machine functionIndex input output ->
      Holds setup machine functionIndex input output

/-- The augmented function accepts exactly the paper's base or recursive
branch.  In particular, there is no third terminal-fold branch. -/
theorem holds_iff_base_or_recursive
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
    (output : Output Digest State Running slotCount) :
    Holds setup machine functionIndex input output <->
      BaseHolds setup machine functionIndex input output \/
      RecursiveHolds setup machine functionIndex input output := by
  constructor
  · intro accepted
    cases accepted with
    | base holds => exact Or.inl holds
    | recursive holds => exact Or.inr holds
  · rintro (holds | holds)
    · exact Holds.base holds
    · exact Holds.recursive holds

/-- The helper inductive accepts exactly the independently expanded paper
transition.  No acceptance proposition or caller-supplied semantic fact occurs
on the right-hand side. -/
theorem holds_iff_transition
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
    (output : Output Digest State Running slotCount) :
    Holds setup machine functionIndex input output <->
      Transition setup machine functionIndex input output := by
  constructor
  · intro accepted
    cases accepted with
    | base holds =>
        exact ⟨holds.application.dispatch, holds.application.pcNext,
          holds.application.application, holds.outputHash,
          Or.inl ⟨holds.iterationZero, holds.initialState,
            holds.defaultRunning⟩⟩
    | recursive holds =>
        exact ⟨holds.application.dispatch, holds.application.pcNext,
          holds.application.application, holds.outputHash,
          Or.inr ⟨holds.priorPcValid, holds.iterationPositive,
            holds.priorPublicInput, holds.selectedNifs, holds.unchanged⟩⟩
  · rintro ⟨dispatch, pcNext, application, outputHash, branch⟩
    have applicationHolds :
        ApplicationHolds machine functionIndex input output :=
      ⟨dispatch, pcNext, application⟩
    rcases branch with base | recursive
    · exact Holds.base {
        iterationZero := base.1
        initialState := base.2.1
        application := applicationHolds
        defaultRunning := base.2.2
        outputHash := outputHash }
    · rcases recursive with
        ⟨priorPcValid, iterationPositive, priorPublicInput,
          selectedNifs, unchanged⟩
      exact Holds.recursive {
        iterationPositive := iterationPositive
        priorPcValid := priorPcValid
        priorPublicInput := priorPublicInput
        application := applicationHolds
        selectedNifs := selectedNifs
        unchanged := unchanged
        outputHash := outputHash }

/-- Recursive terminal proof payload checked by Construction 2's outer NIVC
verifier.  This is not the complete proof syntax: the base proof is the
separate `OuterTerminalProof.bottom` constructor below. -/
structure TerminalProof
    (Running : Type uRunning)
    (RunningWitness : Type uRunningWitness)
    (Fresh : Type uFresh)
    (FreshWitness : Type uFreshWitness)
    (slotCount : Nat) where
  running : Fin slotCount -> Running
  runningWitness : Fin slotCount -> RunningWitness
  fresh : Fresh
  freshWitness : FreshWitness
  pc : Nat

/-- Exact outer terminal-proof syntax from Construction 2.  The base proof is
the unique `bottom` constructor and carries no recursive relation payload. -/
inductive OuterTerminalProof
    (Running : Type uRunning)
    (RunningWitness : Type uRunningWitness)
    (Fresh : Type uFresh)
    (FreshWitness : Type uFreshWitness)
    (slotCount : Nat) where
  | bottom
  | recursive
      (payload : TerminalProof Running RunningWitness Fresh FreshWitness
        slotCount)

/-- The terminal statement contains only the advertised trace endpoint. -/
structure TerminalStatement (State : Type uState) where
  iteration : Nat
  z0 : State
  zi : State

/-- Relation membership checks performed at the terminal boundary. -/
structure TerminalRelations
    (Key : Type uKey)
    (Running : Type uRunning)
    (RunningWitness : Type uRunningWitness)
    (Fresh : Type uFresh)
    (FreshWitness : Type uFreshWitness)
    (slotCount : Nat) where
  runningHolds : (slot : Fin slotCount) ->
    Key -> Running -> RunningWitness -> Prop
  freshHolds : (slot : Fin slotCount) ->
    Key -> Fresh -> FreshWitness -> Prop

/-- The recursive terminal equations.  They check every running
instance/witness pair and the selected fresh instance/witness pair, and never
invoke `NIFS.V`. -/
def RecursiveTerminalTransition
    {Key : Type uKey}
    {Digest : Type uDigest}
    {State : Type uState}
    {Witness : Type uWitness}
    {Running : Type uRunning}
    {RunningWitness : Type uRunningWitness}
    {Fresh : Type uFresh}
    {FreshWitness : Type uFreshWitness}
    {Proof : Type uProof}
    {Encoded : Type uEncoded}
    {slotCount : Nat}
    (setup : Setup Key Running Fresh Proof slotCount)
    (machine : Machine Key Digest State Witness Running Fresh Encoded slotCount)
    (relations : TerminalRelations Key Running RunningWitness Fresh FreshWitness
      slotCount)
    (statement : TerminalStatement State)
    (proof : TerminalProof Running RunningWitness Fresh FreshWitness slotCount) :
    Prop :=
  exists pcValid : InRange slotCount proof.pc,
    0 < statement.iteration /\
    machine.freshPublic proof.fresh =
      machine.encodeInstance (machine.hash {
        verifierKeys := setup.verifierKeys
        iteration := statement.iteration
        z0 := statement.z0
        current := statement.zi
        running := proof.running
        pc := proof.pc
      }) /\
    (forall slot, relations.runningHolds slot (setup.verifierKeys slot)
      (proof.running slot) (proof.runningWitness slot)) /\
    relations.freshHolds (selectedIndex pcValid)
      (setup.verifierKeys (selectedIndex pcValid)) proof.fresh proof.freshWitness

/-- Payload-compatible terminal equations.  This relation is retained for
the recursive payload lowerings.  At iteration zero it treats the payload as
erased.  `OuterTerminalTransition` below owns the exact paper proof syntax. -/
def TerminalTransition
    {Key : Type uKey}
    {Digest : Type uDigest}
    {State : Type uState}
    {Witness : Type uWitness}
    {Running : Type uRunning}
    {RunningWitness : Type uRunningWitness}
    {Fresh : Type uFresh}
    {FreshWitness : Type uFreshWitness}
    {Proof : Type uProof}
    {Encoded : Type uEncoded}
    {slotCount : Nat}
    (setup : Setup Key Running Fresh Proof slotCount)
    (machine : Machine Key Digest State Witness Running Fresh Encoded slotCount)
    (relations : TerminalRelations Key Running RunningWitness Fresh FreshWitness
      slotCount)
    (statement : TerminalStatement State)
    (proof : TerminalProof Running RunningWitness Fresh FreshWitness slotCount) :
    Prop :=
  (statement.iteration = 0 /\ statement.zi = statement.z0) \/
    RecursiveTerminalTransition setup machine relations statement proof

/-- Exact Construction-2 terminal equations over the bottom-or-recursive
proof envelope.  Bottom carries no payload.  Recursive proof data is accepted
only through the recursive constructor. -/
def OuterTerminalTransition
    {Key : Type uKey}
    {Digest : Type uDigest}
    {State : Type uState}
    {Witness : Type uWitness}
    {Running : Type uRunning}
    {RunningWitness : Type uRunningWitness}
    {Fresh : Type uFresh}
    {FreshWitness : Type uFreshWitness}
    {Proof : Type uProof}
    {Encoded : Type uEncoded}
    {slotCount : Nat}
    (setup : Setup Key Running Fresh Proof slotCount)
    (machine : Machine Key Digest State Witness Running Fresh Encoded slotCount)
    (relations : TerminalRelations Key Running RunningWitness Fresh FreshWitness
      slotCount)
    (statement : TerminalStatement State)
    (proof : OuterTerminalProof Running RunningWitness Fresh FreshWitness
      slotCount) : Prop :=
  match proof with
  | .bottom => statement.iteration = 0 /\ statement.zi = statement.z0
  | .recursive payload =>
      RecursiveTerminalTransition setup machine relations statement payload

/-- Payload-compatible terminal helper. The base case treats the recursive
payload as erased and checks only the initial endpoint. A positive iteration
checks the prior public link, counter, all running relations, and the selected
fresh relation; it performs no NIFS fold. -/
inductive TerminalHolds
    {Key : Type uKey}
    {Digest : Type uDigest}
    {State : Type uState}
    {Witness : Type uWitness}
    {Running : Type uRunning}
    {RunningWitness : Type uRunningWitness}
    {Fresh : Type uFresh}
    {FreshWitness : Type uFreshWitness}
    {Proof : Type uProof}
    {Encoded : Type uEncoded}
    {slotCount : Nat}
    (setup : Setup Key Running Fresh Proof slotCount)
    (machine : Machine Key Digest State Witness Running Fresh Encoded slotCount)
    (relations : TerminalRelations Key Running RunningWitness Fresh FreshWitness
      slotCount)
    (statement : TerminalStatement State)
    (proof : TerminalProof Running RunningWitness Fresh FreshWitness slotCount) :
    Prop where
  | base : statement.iteration = 0 -> statement.zi = statement.z0 ->
      TerminalHolds setup machine relations statement proof
  | recursive
      (iterationPositive : 0 < statement.iteration)
      (pcValid : InRange slotCount proof.pc)
      (priorPublicInput : machine.freshPublic proof.fresh =
        machine.encodeInstance (machine.hash {
          verifierKeys := setup.verifierKeys
          iteration := statement.iteration
          z0 := statement.z0
          current := statement.zi
          running := proof.running
          pc := proof.pc
        }))
      (runningValid : forall slot,
        relations.runningHolds slot (setup.verifierKeys slot)
          (proof.running slot) (proof.runningWitness slot))
      (freshValid : relations.freshHolds (selectedIndex pcValid)
        (setup.verifierKeys (selectedIndex pcValid)) proof.fresh
          proof.freshWitness) :
      TerminalHolds setup machine relations statement proof

/-- The payload helper accepts exactly its erased-payload base or recursive
terminal case. The recursive case checks relation membership and performs no
NIFS call. -/
theorem terminalHolds_iff_transition
    {Key : Type uKey}
    {Digest : Type uDigest}
    {State : Type uState}
    {Witness : Type uWitness}
    {Running : Type uRunning}
    {RunningWitness : Type uRunningWitness}
    {Fresh : Type uFresh}
    {FreshWitness : Type uFreshWitness}
    {Proof : Type uProof}
    {Encoded : Type uEncoded}
    {slotCount : Nat}
    (setup : Setup Key Running Fresh Proof slotCount)
    (machine : Machine Key Digest State Witness Running Fresh Encoded slotCount)
    (relations : TerminalRelations Key Running RunningWitness Fresh FreshWitness
      slotCount)
    (statement : TerminalStatement State)
    (proof : TerminalProof Running RunningWitness Fresh FreshWitness slotCount) :
    TerminalHolds setup machine relations statement proof <->
      TerminalTransition setup machine relations statement proof := by
  constructor
  · intro accepted
    cases accepted with
    | base iterationZero initialState =>
        exact Or.inl ⟨iterationZero, initialState⟩
    | recursive iterationPositive pcValid priorPublicInput runningValid freshValid =>
        exact Or.inr ⟨pcValid, iterationPositive, priorPublicInput,
          runningValid, freshValid⟩
  · rintro (⟨iterationZero, initialState⟩ |
      ⟨pcValid, iterationPositive, priorPublicInput, runningValid, freshValid⟩)
    · exact TerminalHolds.base iterationZero initialState
    · exact TerminalHolds.recursive iterationPositive pcValid priorPublicInput
        runningValid freshValid

/-- Exact outer terminal acceptance.  The base constructor contains no
recursive proof payload. -/
inductive OuterTerminalHolds
    {Key : Type uKey}
    {Digest : Type uDigest}
    {State : Type uState}
    {Witness : Type uWitness}
    {Running : Type uRunning}
    {RunningWitness : Type uRunningWitness}
    {Fresh : Type uFresh}
    {FreshWitness : Type uFreshWitness}
    {Proof : Type uProof}
    {Encoded : Type uEncoded}
    {slotCount : Nat}
    (setup : Setup Key Running Fresh Proof slotCount)
    (machine : Machine Key Digest State Witness Running Fresh Encoded slotCount)
    (relations : TerminalRelations Key Running RunningWitness Fresh FreshWitness
      slotCount)
    (statement : TerminalStatement State) :
    OuterTerminalProof Running RunningWitness Fresh FreshWitness slotCount ->
      Prop where
  | bottom
      (iterationZero : statement.iteration = 0)
      (initialState : statement.zi = statement.z0) :
      OuterTerminalHolds setup machine relations statement .bottom
  | recursive
      (payload : TerminalProof Running RunningWitness Fresh FreshWitness
        slotCount)
      (recursiveHolds : RecursiveTerminalTransition setup machine relations
        statement payload) :
      OuterTerminalHolds setup machine relations statement (.recursive payload)

/-- The exact outer helper accepts exactly the independently expanded
bottom-or-recursive terminal transition. -/
theorem outerTerminalHolds_iff_transition
    {Key : Type uKey}
    {Digest : Type uDigest}
    {State : Type uState}
    {Witness : Type uWitness}
    {Running : Type uRunning}
    {RunningWitness : Type uRunningWitness}
    {Fresh : Type uFresh}
    {FreshWitness : Type uFreshWitness}
    {Proof : Type uProof}
    {Encoded : Type uEncoded}
    {slotCount : Nat}
    (setup : Setup Key Running Fresh Proof slotCount)
    (machine : Machine Key Digest State Witness Running Fresh Encoded slotCount)
    (relations : TerminalRelations Key Running RunningWitness Fresh FreshWitness
      slotCount)
    (statement : TerminalStatement State)
    (proof : OuterTerminalProof Running RunningWitness Fresh FreshWitness
      slotCount) :
    OuterTerminalHolds setup machine relations statement proof <->
      OuterTerminalTransition setup machine relations statement proof := by
  constructor
  · intro accepted
    cases accepted with
    | bottom iterationZero initialState =>
        exact ⟨iterationZero, initialState⟩
    | recursive payload recursiveHolds =>
        exact recursiveHolds
  · intro transition
    cases proof with
    | bottom =>
        exact OuterTerminalHolds.bottom transition.1 transition.2
    | recursive payload =>
        exact OuterTerminalHolds.recursive payload transition

end Nightstream.HyperNova.Construction2.Paper
