import Lean.Elab.Tactic.Omega
import Nightstream.HyperNova.Construction2.Paper

/-!
Contract: global trace semantics for HyperNova Construction 2.

Assurance tier: protocol model.

Owns exact adjacency between paper invocations, the unique base invocation,
one recursive branch for every later invocation, equality of adjacent hash
preimages, and the link from the last public digest to the terminal fresh
instance.

Does not own a concrete hash, NIFS knowledge soundness, circuit rows,
recursive-size closure, a deployed verifier, or cryptographic probabilities.

The terminal verifier in Construction 2 checks the trailing fresh relation.
It does not perform another NIFS fold. This file keeps that paper terminal
schedule separate from protocols that add a delayed terminal fold.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.HyperNova.Construction2.PaperTrace

open Nightstream.HyperNova.NonInteractiveMultiFold
open Nightstream.HyperNova.Construction2.Paper

universe uKey uDigest uState uWitness uRunning uFresh uProof uEncoded
  uRunningWitness uFreshWitness

/-- One fixed paper setup and application machine for a complete trace. -/
structure Context
    (Key : Type uKey)
    (Digest : Type uDigest)
    (State : Type uState)
    (Witness : Type uWitness)
    (Running : Type uRunning)
    (Fresh : Type uFresh)
    (Proof : Type uProof)
    (Encoded : Type uEncoded)
    (slotCount : Nat) where
  setup : Setup Key Running Fresh Proof slotCount
  machine : Machine Key Digest State Witness Running Fresh Encoded slotCount

/-- One exact invocation of the paper's fused outer-dispatch and `F'_j`
relation. -/
structure Invocation
    {Key : Type uKey}
    {Digest : Type uDigest}
    {State : Type uState}
    {Witness : Type uWitness}
    {Running : Type uRunning}
    {Fresh : Type uFresh}
    {Proof : Type uProof}
    {Encoded : Type uEncoded}
    {slotCount : Nat}
    (context : Context Key Digest State Witness Running Fresh Proof Encoded
      slotCount) where
  functionIndex : Fin slotCount
  input : Input Key State Witness Running Fresh Proof slotCount
  output : Output Digest State Running slotCount
  holds : Holds context.setup context.machine functionIndex input output

namespace Invocation

/-- This invocation uses the paper base branch. -/
def IsBase
    {Key : Type uKey}
    {Digest : Type uDigest}
    {State : Type uState}
    {Witness : Type uWitness}
    {Running : Type uRunning}
    {Fresh : Type uFresh}
    {Proof : Type uProof}
    {Encoded : Type uEncoded}
    {slotCount : Nat}
    {context : Context Key Digest State Witness Running Fresh Proof Encoded
      slotCount}
    (invocation : Invocation context) : Prop :=
  BaseHolds context.setup context.machine invocation.functionIndex
    invocation.input invocation.output

/-- This invocation uses the paper recursive branch. -/
def IsRecursive
    {Key : Type uKey}
    {Digest : Type uDigest}
    {State : Type uState}
    {Witness : Type uWitness}
    {Running : Type uRunning}
    {Fresh : Type uFresh}
    {Proof : Type uProof}
    {Encoded : Type uEncoded}
    {slotCount : Nat}
    {context : Context Key Digest State Witness Running Fresh Proof Encoded
      slotCount}
    (invocation : Invocation context) : Prop :=
  RecursiveHolds context.setup context.machine invocation.functionIndex
    invocation.input invocation.output

/-- Each satisfying paper invocation is one of the two paper branches. -/
theorem classified
    {Key : Type uKey}
    {Digest : Type uDigest}
    {State : Type uState}
    {Witness : Type uWitness}
    {Running : Type uRunning}
    {Fresh : Type uFresh}
    {Proof : Type uProof}
    {Encoded : Type uEncoded}
    {slotCount : Nat}
    {context : Context Key Digest State Witness Running Fresh Proof Encoded
      slotCount}
    (invocation : Invocation context) :
    invocation.IsBase \/ invocation.IsRecursive := by
  exact (holds_iff_base_or_recursive context.setup context.machine
    invocation.functionIndex invocation.input invocation.output).mp
      invocation.holds

/-- Iteration zero forces the base branch. -/
theorem isBase_of_iteration_zero
    {Key : Type uKey}
    {Digest : Type uDigest}
    {State : Type uState}
    {Witness : Type uWitness}
    {Running : Type uRunning}
    {Fresh : Type uFresh}
    {Proof : Type uProof}
    {Encoded : Type uEncoded}
    {slotCount : Nat}
    {context : Context Key Digest State Witness Running Fresh Proof Encoded
      slotCount}
    (invocation : Invocation context)
    (iterationZero : invocation.input.iteration = 0) :
    invocation.IsBase := by
  rcases invocation.classified with base | recursive
  · exact base
  · have positive := recursive.iterationPositive
    omega

/-- A positive iteration forces the recursive branch and therefore one
selected call to the function-valued paper NIFS verifier. -/
theorem isRecursive_of_iteration_positive
    {Key : Type uKey}
    {Digest : Type uDigest}
    {State : Type uState}
    {Witness : Type uWitness}
    {Running : Type uRunning}
    {Fresh : Type uFresh}
    {Proof : Type uProof}
    {Encoded : Type uEncoded}
    {slotCount : Nat}
    {context : Context Key Digest State Witness Running Fresh Proof Encoded
      slotCount}
    (invocation : Invocation context)
    (iterationPositive : 0 < invocation.input.iteration) :
    invocation.IsRecursive := by
  rcases invocation.classified with base | recursive
  · have zero := base.iterationZero
    omega
  · exact recursive

/-- Every branch recomputes the complete next public hash preimage. -/
theorem outputHolds
    {Key : Type uKey}
    {Digest : Type uDigest}
    {State : Type uState}
    {Witness : Type uWitness}
    {Running : Type uRunning}
    {Fresh : Type uFresh}
    {Proof : Type uProof}
    {Encoded : Type uEncoded}
    {slotCount : Nat}
    {context : Context Key Digest State Witness Running Fresh Proof Encoded
      slotCount}
    (invocation : Invocation context) :
    OutputHolds context.setup context.machine invocation.input
      invocation.output := by
  rcases invocation.classified with base | recursive
  · exact base.outputHash
  · exact recursive.outputHash

end Invocation

/-- Exact public-state identity between two adjacent paper invocations. -/
structure Adjacent
    {Key : Type uKey}
    {Digest : Type uDigest}
    {State : Type uState}
    {Witness : Type uWitness}
    {Running : Type uRunning}
    {Fresh : Type uFresh}
    {Proof : Type uProof}
    {Encoded : Type uEncoded}
    {slotCount : Nat}
    {context : Context Key Digest State Witness Running Fresh Proof Encoded
      slotCount}
    (current next : Invocation context) : Prop where
  iteration : next.input.iteration = current.input.iteration + 1
  initialState : next.input.z0 = current.input.z0
  currentState : next.input.zi = current.output.zNext
  running : next.input.running = current.output.runningNext
  priorPc : next.input.priorPc = oneBased current.output.pcNext

namespace Adjacent

/-- Exact adjacency makes the next prior-hash preimage identical to the
current invocation's next-hash preimage. -/
theorem hashPreimage_eq
    {Key : Type uKey}
    {Digest : Type uDigest}
    {State : Type uState}
    {Witness : Type uWitness}
    {Running : Type uRunning}
    {Fresh : Type uFresh}
    {Proof : Type uProof}
    {Encoded : Type uEncoded}
    {slotCount : Nat}
    {context : Context Key Digest State Witness Running Fresh Proof Encoded
      slotCount}
    {current next : Invocation context}
    (adjacent : Adjacent current next) :
    priorHashPreimage context.setup next.input =
      nextHashPreimage context.setup current.input current.output := by
  simp only [priorHashPreimage, nextHashPreimage]
  rw [adjacent.iteration, adjacent.initialState, adjacent.currentState,
    adjacent.running, adjacent.priorPc]

/-- Equality of the complete typed hash preimages is also sufficient for
exact adjacency. A later concrete hash reduction therefore needs only the
standard split: equal preimages, or a collision in the selected hash. -/
theorem iff_hashPreimage_eq
    {Key : Type uKey}
    {Digest : Type uDigest}
    {State : Type uState}
    {Witness : Type uWitness}
    {Running : Type uRunning}
    {Fresh : Type uFresh}
    {Proof : Type uProof}
    {Encoded : Type uEncoded}
    {slotCount : Nat}
    {context : Context Key Digest State Witness Running Fresh Proof Encoded
      slotCount}
    {current next : Invocation context} :
    Adjacent current next ↔
      priorHashPreimage context.setup next.input =
        nextHashPreimage context.setup current.input current.output := by
  constructor
  · exact hashPreimage_eq
  · intro same
    refine
      { iteration := ?_
        initialState := ?_
        currentState := ?_
        running := ?_
        priorPc := ?_ }
    · have projected := congrArg
        (fun value => value.iteration) same
      simpa [priorHashPreimage, nextHashPreimage] using projected
    · have projected := congrArg
        (fun value => value.z0) same
      simpa [priorHashPreimage, nextHashPreimage] using projected
    · have projected := congrArg
        (fun value => value.current) same
      simpa [priorHashPreimage, nextHashPreimage] using projected
    · have projected := congrArg
        (fun value => value.running) same
      simpa [priorHashPreimage, nextHashPreimage] using projected
    · have projected := congrArg
        (fun value => value.pc) same
      simpa [priorHashPreimage, nextHashPreimage] using projected

/-- The next invocation is recursive because its exact iteration is positive.
-/
theorem next_isRecursive
    {Key : Type uKey}
    {Digest : Type uDigest}
    {State : Type uState}
    {Witness : Type uWitness}
    {Running : Type uRunning}
    {Fresh : Type uFresh}
    {Proof : Type uProof}
    {Encoded : Type uEncoded}
    {slotCount : Nat}
    {context : Context Key Digest State Witness Running Fresh Proof Encoded
      slotCount}
    {current next : Invocation context}
    (adjacent : Adjacent current next) :
    next.IsRecursive := by
  apply next.isRecursive_of_iteration_positive
  rw [adjacent.iteration]
  omega

/-- The next fresh public input is the exact encoded digest emitted by the
current invocation. This conclusion is derived, not carried in `Adjacent`. -/
theorem nextFreshPublic_eq_currentOutput
    {Key : Type uKey}
    {Digest : Type uDigest}
    {State : Type uState}
    {Witness : Type uWitness}
    {Running : Type uRunning}
    {Fresh : Type uFresh}
    {Proof : Type uProof}
    {Encoded : Type uEncoded}
    {slotCount : Nat}
    {context : Context Key Digest State Witness Running Fresh Proof Encoded
      slotCount}
    {current next : Invocation context}
    (adjacent : Adjacent current next) :
    context.machine.freshPublic next.input.fresh =
      context.machine.encodeInstance current.output.x := by
  have recursive := adjacent.next_isRecursive
  calc
    context.machine.freshPublic next.input.fresh =
        context.machine.encodeInstance
          (context.machine.hash
            (priorHashPreimage context.setup next.input)) :=
      recursive.priorPublicInput
    _ = context.machine.encodeInstance
          (context.machine.hash
            (nextHashPreimage context.setup current.input current.output)) := by
      rw [adjacent.hashPreimage_eq]
    _ = context.machine.encodeInstance current.output.x := by
      rw [current.outputHolds]

end Adjacent

/-- A nonempty exact Construction-2 execution. The index is the number of
augmented-function invocations. -/
inductive Run
    {Key : Type uKey}
    {Digest : Type uDigest}
    {State : Type uState}
    {Witness : Type uWitness}
    {Running : Type uRunning}
    {Fresh : Type uFresh}
    {Proof : Type uProof}
    {Encoded : Type uEncoded}
    {slotCount : Nat}
    (context : Context Key Digest State Witness Running Fresh Proof Encoded
      slotCount) : Invocation context -> Invocation context -> Nat -> Prop where
  | start (invocation : Invocation context)
      (iterationZero : invocation.input.iteration = 0) :
      Run context invocation invocation 1
  | step
      {first current : Invocation context}
      {invocationCount : Nat}
      (prior : Run context first current invocationCount)
      (next : Invocation context)
      (adjacent : Adjacent current next) :
      Run context first next (invocationCount + 1)

/-- Structural certificate for the exact base-then-recursive branch schedule.
-/
inductive BranchSchedule
    {Key : Type uKey}
    {Digest : Type uDigest}
    {State : Type uState}
    {Witness : Type uWitness}
    {Running : Type uRunning}
    {Fresh : Type uFresh}
    {Proof : Type uProof}
    {Encoded : Type uEncoded}
    {slotCount : Nat}
    {context : Context Key Digest State Witness Running Fresh Proof Encoded
      slotCount} :
    {first last : Invocation context} -> {invocationCount : Nat} ->
      Run context first last invocationCount -> Prop where
  | start
      (invocation : Invocation context)
      (iterationZero : invocation.input.iteration = 0)
      (base : invocation.IsBase) :
      BranchSchedule (Run.start invocation iterationZero)
  | step
      {first current : Invocation context}
      {invocationCount : Nat}
      {prior : Run context first current invocationCount}
      (schedule : BranchSchedule prior)
      (next : Invocation context)
      (adjacent : Adjacent current next)
      (recursive : next.IsRecursive) :
      BranchSchedule (Run.step prior next adjacent)

namespace Run

/-- Every nonempty run has one base invocation followed only by recursive
invocations. -/
theorem exactBranchSchedule
    {Key : Type uKey}
    {Digest : Type uDigest}
    {State : Type uState}
    {Witness : Type uWitness}
    {Running : Type uRunning}
    {Fresh : Type uFresh}
    {Proof : Type uProof}
    {Encoded : Type uEncoded}
    {slotCount : Nat}
    {context : Context Key Digest State Witness Running Fresh Proof Encoded
      slotCount}
    {first last : Invocation context}
    {invocationCount : Nat}
    (run : Run context first last invocationCount) :
    BranchSchedule run := by
  induction run with
  | start iterationZero =>
      exact BranchSchedule.start _ iterationZero
        (first.isBase_of_iteration_zero iterationZero)
  | step prior next adjacent inductionHypothesis =>
      exact BranchSchedule.step inductionHypothesis next adjacent
        adjacent.next_isRecursive

/-- The final invocation index equals the number of completed application
steps minus one. -/
theorem last_iteration_add_one
    {Key : Type uKey}
    {Digest : Type uDigest}
    {State : Type uState}
    {Witness : Type uWitness}
    {Running : Type uRunning}
    {Fresh : Type uFresh}
    {Proof : Type uProof}
    {Encoded : Type uEncoded}
    {slotCount : Nat}
    {context : Context Key Digest State Witness Running Fresh Proof Encoded
      slotCount}
    {first last : Invocation context}
    {invocationCount : Nat}
    (run : Run context first last invocationCount) :
    last.input.iteration + 1 = invocationCount := by
  induction run with
  | start iterationZero => omega
  | step prior next adjacent inductionHypothesis =>
      rw [adjacent.iteration]
      omega

/-- A paper run is nonempty by construction. -/
theorem invocationCount_positive
    {Key : Type uKey}
    {Digest : Type uDigest}
    {State : Type uState}
    {Witness : Type uWitness}
    {Running : Type uRunning}
    {Fresh : Type uFresh}
    {Proof : Type uProof}
    {Encoded : Type uEncoded}
    {slotCount : Nat}
    {context : Context Key Digest State Witness Running Fresh Proof Encoded
      slotCount}
    {first last : Invocation context}
    {invocationCount : Nat}
    (run : Run context first last invocationCount) :
    0 < invocationCount := by
  cases run <;> omega

end Run

/-- Exact identification of the positive terminal state and payload with the
last invocation's output. -/
structure TerminalEndpoint
    {Key : Type uKey}
    {Digest : Type uDigest}
    {State : Type uState}
    {Witness : Type uWitness}
    {Running : Type uRunning}
    {Fresh : Type uFresh}
    {Proof : Type uProof}
    {Encoded : Type uEncoded}
    {RunningWitness : Type uRunningWitness}
    {FreshWitness : Type uFreshWitness}
    {slotCount : Nat}
    {context : Context Key Digest State Witness Running Fresh Proof Encoded
      slotCount}
    (last : Invocation context)
    (statement : TerminalStatement State)
    (payload : TerminalProof Running RunningWitness Fresh FreshWitness
      slotCount) : Prop where
  iteration : statement.iteration = last.input.iteration + 1
  initialState : statement.z0 = last.input.z0
  currentState : statement.zi = last.output.zNext
  running : payload.running = last.output.runningNext
  pc : payload.pc = oneBased last.output.pcNext

namespace TerminalEndpoint

/-- The terminal hash preimage is exactly the last invocation's next-hash
preimage. -/
theorem hashPreimage_eq
    {Key : Type uKey}
    {Digest : Type uDigest}
    {State : Type uState}
    {Witness : Type uWitness}
    {Running : Type uRunning}
    {Fresh : Type uFresh}
    {Proof : Type uProof}
    {Encoded : Type uEncoded}
    {RunningWitness : Type uRunningWitness}
    {FreshWitness : Type uFreshWitness}
    {slotCount : Nat}
    {context : Context Key Digest State Witness Running Fresh Proof Encoded
      slotCount}
    {last : Invocation context}
    {statement : TerminalStatement State}
    {payload : TerminalProof Running RunningWitness Fresh FreshWitness
      slotCount}
    (endpoint : TerminalEndpoint last statement payload) :
    (show HashPreimage Key State Running slotCount from {
      verifierKeys := context.setup.verifierKeys
      iteration := statement.iteration
      z0 := statement.z0
      current := statement.zi
      running := payload.running
      pc := payload.pc
    }) = nextHashPreimage context.setup last.input last.output := by
  simp only [nextHashPreimage]
  rw [endpoint.iteration, endpoint.initialState, endpoint.currentState,
    endpoint.running, endpoint.pc]

/-- Equality of the complete terminal and final-invocation preimages is
equivalent to the exact terminal endpoint. No endpoint field is an extra
semantic assumption after preimage equality has been established. -/
theorem iff_hashPreimage_eq
    {Key : Type uKey}
    {Digest : Type uDigest}
    {State : Type uState}
    {Witness : Type uWitness}
    {Running : Type uRunning}
    {Fresh : Type uFresh}
    {Proof : Type uProof}
    {Encoded : Type uEncoded}
    {RunningWitness : Type uRunningWitness}
    {FreshWitness : Type uFreshWitness}
    {slotCount : Nat}
    {context : Context Key Digest State Witness Running Fresh Proof Encoded
      slotCount}
    {last : Invocation context}
    {statement : TerminalStatement State}
    {payload : TerminalProof Running RunningWitness Fresh FreshWitness
      slotCount} :
    TerminalEndpoint last statement payload ↔
      (show HashPreimage Key State Running slotCount from {
        verifierKeys := context.setup.verifierKeys
        iteration := statement.iteration
        z0 := statement.z0
        current := statement.zi
        running := payload.running
        pc := payload.pc
      }) = nextHashPreimage context.setup last.input last.output := by
  constructor
  · exact hashPreimage_eq
  · intro same
    refine
      { iteration := ?_
        initialState := ?_
        currentState := ?_
        running := ?_
        pc := ?_ }
    · have projected := congrArg
        (fun value => value.iteration) same
      simpa [nextHashPreimage] using projected
    · have projected := congrArg
        (fun value => value.z0) same
      simpa [nextHashPreimage] using projected
    · have projected := congrArg
        (fun value => value.current) same
      simpa [nextHashPreimage] using projected
    · have projected := congrArg
        (fun value => value.running) same
      simpa [nextHashPreimage] using projected
    · have projected := congrArg
        (fun value => value.pc) same
      simpa [nextHashPreimage] using projected

end TerminalEndpoint

/-- A complete positive Construction-2 run with exact terminal relation
membership. -/
structure ClosedRun
    {Key : Type uKey}
    {Digest : Type uDigest}
    {State : Type uState}
    {Witness : Type uWitness}
    {Running : Type uRunning}
    {Fresh : Type uFresh}
    {Proof : Type uProof}
    {Encoded : Type uEncoded}
    {RunningWitness : Type uRunningWitness}
    {FreshWitness : Type uFreshWitness}
    {slotCount : Nat}
    (context : Context Key Digest State Witness Running Fresh Proof Encoded
      slotCount)
    (relations : TerminalRelations Key Running RunningWitness Fresh
      FreshWitness slotCount)
    (first last : Invocation context)
    (invocationCount : Nat)
    (statement : TerminalStatement State)
    (payload : TerminalProof Running RunningWitness Fresh FreshWitness
      slotCount) : Prop where
  run : Run context first last invocationCount
  endpoint : TerminalEndpoint last statement payload
  terminal : RecursiveTerminalTransition context.setup context.machine
    relations statement payload

namespace ClosedRun

/-- Terminal acceptance derives the exact trailing public link. It is not an
assumption of `ClosedRun`. -/
theorem trailingFreshPublic_eq_lastOutput
    {Key : Type uKey}
    {Digest : Type uDigest}
    {State : Type uState}
    {Witness : Type uWitness}
    {Running : Type uRunning}
    {Fresh : Type uFresh}
    {Proof : Type uProof}
    {Encoded : Type uEncoded}
    {RunningWitness : Type uRunningWitness}
    {FreshWitness : Type uFreshWitness}
    {slotCount : Nat}
    {context : Context Key Digest State Witness Running Fresh Proof Encoded
      slotCount}
    {relations : TerminalRelations Key Running RunningWitness Fresh
      FreshWitness slotCount}
    {first last : Invocation context}
    {invocationCount : Nat}
    {statement : TerminalStatement State}
    {payload : TerminalProof Running RunningWitness Fresh FreshWitness
      slotCount}
    (closed : ClosedRun context relations first last invocationCount statement
      payload) :
    context.machine.freshPublic payload.fresh =
      context.machine.encodeInstance last.output.x := by
  rcases closed.terminal with
    ⟨pcValid, iterationPositive, priorPublicInput, runningValid, freshValid⟩
  calc
    context.machine.freshPublic payload.fresh =
        context.machine.encodeInstance (context.machine.hash {
          verifierKeys := context.setup.verifierKeys
          iteration := statement.iteration
          z0 := statement.z0
          current := statement.zi
          running := payload.running
          pc := payload.pc
        }) := priorPublicInput
    _ = context.machine.encodeInstance
          (context.machine.hash
            (nextHashPreimage context.setup last.input last.output)) := by
      exact congrArg
        (fun preimage => context.machine.encodeInstance
          (context.machine.hash preimage))
        closed.endpoint.hashPreimage_eq
    _ = context.machine.encodeInstance last.output.x := by
      rw [last.outputHolds]

/-- The terminal statement iteration is exactly the number of completed paper
invocations. -/
theorem terminal_iteration_eq_invocationCount
    {Key : Type uKey}
    {Digest : Type uDigest}
    {State : Type uState}
    {Witness : Type uWitness}
    {Running : Type uRunning}
    {Fresh : Type uFresh}
    {Proof : Type uProof}
    {Encoded : Type uEncoded}
    {RunningWitness : Type uRunningWitness}
    {FreshWitness : Type uFreshWitness}
    {slotCount : Nat}
    {context : Context Key Digest State Witness Running Fresh Proof Encoded
      slotCount}
    {relations : TerminalRelations Key Running RunningWitness Fresh
      FreshWitness slotCount}
    {first last : Invocation context}
    {invocationCount : Nat}
    {statement : TerminalStatement State}
    {payload : TerminalProof Running RunningWitness Fresh FreshWitness
      slotCount}
    (closed : ClosedRun context relations first last invocationCount statement
      payload) :
    statement.iteration = invocationCount := by
  rw [closed.endpoint.iteration, closed.run.last_iteration_add_one]

/-- A positive run cannot use the paper's payload-free bottom terminal proof.
-/
theorem bottom_rejected
    {Key : Type uKey}
    {Digest : Type uDigest}
    {State : Type uState}
    {Witness : Type uWitness}
    {Running : Type uRunning}
    {Fresh : Type uFresh}
    {Proof : Type uProof}
    {Encoded : Type uEncoded}
    {RunningWitness : Type uRunningWitness}
    {FreshWitness : Type uFreshWitness}
    {slotCount : Nat}
    {context : Context Key Digest State Witness Running Fresh Proof Encoded
      slotCount}
    {relations : TerminalRelations Key Running RunningWitness Fresh
      FreshWitness slotCount}
    {first last : Invocation context}
    {invocationCount : Nat}
    {statement : TerminalStatement State}
    {payload : TerminalProof Running RunningWitness Fresh FreshWitness
      slotCount}
    (closed : ClosedRun context relations first last invocationCount statement
      payload) :
    ¬ OuterTerminalTransition context.setup context.machine relations
      statement (OuterTerminalProof.bottom :
        OuterTerminalProof Running RunningWitness Fresh FreshWitness
          slotCount) := by
  intro bottom
  have zero : statement.iteration = 0 := by
    have exactBottom :
        statement.iteration = 0 /\ statement.zi = statement.z0 := by
      simpa [OuterTerminalTransition] using bottom
    exact exactBottom.1
  have positive := closed.run.invocationCount_positive
  have iteration := closed.terminal_iteration_eq_invocationCount
  omega

end ClosedRun

/-- Honest positive terminal data. This is the completeness direction for the
paper terminal relation. It lists relation witnesses and the trailing public
link; it does not assume terminal acceptance. -/
structure HonestTerminalData
    {Key : Type uKey}
    {Digest : Type uDigest}
    {State : Type uState}
    {Witness : Type uWitness}
    {Running : Type uRunning}
    {Fresh : Type uFresh}
    {Proof : Type uProof}
    {Encoded : Type uEncoded}
    {RunningWitness : Type uRunningWitness}
    {FreshWitness : Type uFreshWitness}
    {slotCount : Nat}
    {context : Context Key Digest State Witness Running Fresh Proof Encoded
      slotCount}
    (relations : TerminalRelations Key Running RunningWitness Fresh
      FreshWitness slotCount)
    (last : Invocation context)
    (statement : TerminalStatement State)
    (payload : TerminalProof Running RunningWitness Fresh FreshWitness
      slotCount) : Prop where
  endpoint : TerminalEndpoint last statement payload
  trailingPublic : context.machine.freshPublic payload.fresh =
    context.machine.encodeInstance last.output.x
  runningValid : forall slot,
    relations.runningHolds slot (context.setup.verifierKeys slot)
      (payload.running slot) (payload.runningWitness slot)
  freshValid : relations.freshHolds last.output.pcNext
    (context.setup.verifierKeys last.output.pcNext) payload.fresh
      payload.freshWitness

namespace HonestTerminalData

/-- Honest endpoint data constructs the positive paper terminal relation. -/
theorem terminalTransition
    {Key : Type uKey}
    {Digest : Type uDigest}
    {State : Type uState}
    {Witness : Type uWitness}
    {Running : Type uRunning}
    {Fresh : Type uFresh}
    {Proof : Type uProof}
    {Encoded : Type uEncoded}
    {RunningWitness : Type uRunningWitness}
    {FreshWitness : Type uFreshWitness}
    {slotCount : Nat}
    {context : Context Key Digest State Witness Running Fresh Proof Encoded
      slotCount}
    {relations : TerminalRelations Key Running RunningWitness Fresh
      FreshWitness slotCount}
    {last : Invocation context}
    {statement : TerminalStatement State}
    {payload : TerminalProof Running RunningWitness Fresh FreshWitness
      slotCount}
    (data : HonestTerminalData relations last statement payload) :
    RecursiveTerminalTransition context.setup context.machine relations
      statement payload := by
  have selectedValid : InRange slotCount (oneBased last.output.pcNext) := by
    simp only [InRange, oneBased]
    have bound := last.output.pcNext.isLt
    constructor <;> omega
  have pcValid : InRange slotCount payload.pc := by
    rw [data.endpoint.pc]
    exact selectedValid
  refine ⟨pcValid, ?_, ?_, data.runningValid, ?_⟩
  · rw [data.endpoint.iteration]
    omega
  · calc
      context.machine.freshPublic payload.fresh =
          context.machine.encodeInstance last.output.x := data.trailingPublic
      _ = context.machine.encodeInstance
            (context.machine.hash
              (nextHashPreimage context.setup last.input last.output)) := by
        rw [last.outputHolds]
      _ = context.machine.encodeInstance (context.machine.hash {
            verifierKeys := context.setup.verifierKeys
            iteration := statement.iteration
            z0 := statement.z0
            current := statement.zi
            running := payload.running
            pc := payload.pc
          }) := by
        exact (congrArg
          (fun preimage => context.machine.encodeInstance
            (context.machine.hash preimage))
          data.endpoint.hashPreimage_eq).symm
  · have selected : selectedIndex pcValid = last.output.pcNext := by
      apply Fin.ext
      simp [selectedIndex, data.endpoint.pc, oneBased]
    simpa [selected] using data.freshValid

/-- Honest trace and terminal data construct a closed positive paper run. -/
theorem close
    {Key : Type uKey}
    {Digest : Type uDigest}
    {State : Type uState}
    {Witness : Type uWitness}
    {Running : Type uRunning}
    {Fresh : Type uFresh}
    {Proof : Type uProof}
    {Encoded : Type uEncoded}
    {RunningWitness : Type uRunningWitness}
    {FreshWitness : Type uFreshWitness}
    {slotCount : Nat}
    {context : Context Key Digest State Witness Running Fresh Proof Encoded
      slotCount}
    {relations : TerminalRelations Key Running RunningWitness Fresh
      FreshWitness slotCount}
    {first last : Invocation context}
    {invocationCount : Nat}
    {statement : TerminalStatement State}
    {payload : TerminalProof Running RunningWitness Fresh FreshWitness
      slotCount}
    (run : Run context first last invocationCount)
    (data : HonestTerminalData relations last statement payload) :
    ClosedRun context relations first last invocationCount statement payload :=
  ⟨run, data.endpoint, data.terminalTransition⟩

end HonestTerminalData

end Nightstream.HyperNova.Construction2.PaperTrace
