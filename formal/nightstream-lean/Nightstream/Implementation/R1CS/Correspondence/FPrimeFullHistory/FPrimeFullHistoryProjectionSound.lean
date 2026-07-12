import Nightstream.Implementation.R1CS.Ownership.FPrimeFullHistory.FPrimeFullHistoryProjectionArtifact
import Nightstream.Implementation.R1CS.Correspondence.Projection.ProjectionComplete

/-!
Contract: semantic soundness of every PiRLC projection identity emitted by the
two-step full-history circuit profile.

The premise is satisfaction of the exact generated trace rows for all 62
identities. The conclusion is the strongest deterministic statement available
before transcript probability is introduced: all coefficient identities are
exact, or one challenge is a root of a nonzero bounded error polynomial.
-/

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryProjection

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Program
open Nightstream.Implementation.R1CS.ProjectionProgram
open Nightstream.SuperNeo.ProjectionCheck

/-- Exact generated-row satisfaction forces acceptance of every recursive and
terminal projection identity. -/
theorem batchAccepted (assignment : Nat → Nat)
    (assignmentCanonical : ∀ column, assignment column < goldilocksP)
    (constantOne : assignment 0 = 1)
    (holds : Holds assignment) :
    BatchAccepted K.ops (BatchIdentity traces assignment) := by
  apply ProjectionTrace.census_batchAccepted traces assignment constantOne
    trace_layouts trace_pairs_nonempty trace_pair_widths
  · intro definition definitionMember
    rcases List.mem_flatMap.mp definitionMember with
      ⟨trace, traceMember, member⟩
    apply builderDefinitions_sound assignmentCanonical constantOne
      (definitions_canonical trace traceMember)
    intro row rowMember
    apply holds trace traceMember row
    unfold traceRows
    exact List.mem_append_left _ rowMember
    exact member
  · intro row rowMember
    rcases List.mem_flatMap.mp rowMember with
      ⟨trace, traceMember, member⟩
    apply holds trace traceMember row
    unfold traceRows
    exact List.mem_append_right _ member

/-- `CIR-SOUND` for the exact PiRLC projection census. No accepted identity is
silently promoted to coefficient equality: the only alternative is the named
bad-root event consumed by the probabilistic transcript layer. -/
theorem exact_or_badRoot (assignment : Nat → Nat)
    (assignmentCanonical : ∀ column, assignment column < goldilocksP)
    (constantOne : assignment 0 = 1)
    (holds : Holds assignment) :
    BatchExact (BatchIdentity traces assignment) ∨
      BatchBadRoot K.ops (BatchIdentity traces assignment) := by
  apply batchAccepted_implies_exact_or_badRoot
  exact batchAccepted assignment assignmentCanonical constantOne holds

/-- Exact-row premise restricted to the one recursive NIFS projection census. -/
def RecursiveHolds (assignment : Nat → Nat) : Prop :=
  ∀ trace ∈ recursiveTraces, Satisfies (traceRows trace) assignment

theorem recursive_batchAccepted (assignment : Nat → Nat)
    (assignmentCanonical : ∀ column, assignment column < goldilocksP)
    (constantOne : assignment 0 = 1)
    (holds : RecursiveHolds assignment) :
    BatchAccepted K.ops (BatchIdentity recursiveTraces assignment) := by
  apply ProjectionTrace.census_batchAccepted recursiveTraces assignment
    constantOne
  · intro trace member
    exact trace_layouts trace (List.mem_append_left terminalTraces member)
  · intro trace member
    exact trace_pairs_nonempty trace
      (List.mem_append_left terminalTraces member)
  · intro trace member pair pairMember
    exact trace_pair_widths trace (List.mem_append_left terminalTraces member)
      pair pairMember
  · intro definition definitionMember
    rcases List.mem_flatMap.mp definitionMember with
      ⟨trace, traceMember, member⟩
    apply builderDefinitions_sound assignmentCanonical constantOne
      (definitions_canonical trace
        (List.mem_append_left terminalTraces traceMember))
    intro row rowMember
    apply holds trace traceMember row
    unfold traceRows
    exact List.mem_append_left _ rowMember
    exact member
  · intro row rowMember
    rcases List.mem_flatMap.mp rowMember with
      ⟨trace, traceMember, member⟩
    apply holds trace traceMember row
    unfold traceRows
    exact List.mem_append_right _ member

/-- Recursive PiRLC projection acceptance is coefficient-exact or exposes
the named bounded nonzero-root event. -/
theorem recursive_exact_or_badRoot (assignment : Nat → Nat)
    (assignmentCanonical : ∀ column, assignment column < goldilocksP)
    (constantOne : assignment 0 = 1)
    (holds : RecursiveHolds assignment) :
    BatchExact (BatchIdentity recursiveTraces assignment) ∨
      BatchBadRoot K.ops (BatchIdentity recursiveTraces assignment) := by
  apply batchAccepted_implies_exact_or_badRoot
  exact recursive_batchAccepted assignment assignmentCanonical constantOne
    holds

/-- Exact-row premise restricted to the terminal-fold NIFS projection census. -/
def TerminalHolds (assignment : Nat → Nat) : Prop :=
  ∀ trace ∈ terminalTraces, Satisfies (traceRows trace) assignment

theorem terminal_batchAccepted (assignment : Nat → Nat)
    (assignmentCanonical : ∀ column, assignment column < goldilocksP)
    (constantOne : assignment 0 = 1)
    (holds : TerminalHolds assignment) :
    BatchAccepted K.ops (BatchIdentity terminalTraces assignment) := by
  apply ProjectionTrace.census_batchAccepted terminalTraces assignment
    constantOne
  · intro trace member
    exact trace_layouts trace (List.mem_append_right recursiveTraces member)
  · intro trace member
    exact trace_pairs_nonempty trace
      (List.mem_append_right recursiveTraces member)
  · intro trace member pair pairMember
    exact trace_pair_widths trace (List.mem_append_right recursiveTraces member)
      pair pairMember
  · intro definition definitionMember
    rcases List.mem_flatMap.mp definitionMember with
      ⟨trace, traceMember, member⟩
    apply builderDefinitions_sound assignmentCanonical constantOne
      (definitions_canonical trace
        (List.mem_append_right recursiveTraces traceMember))
    intro row rowMember
    apply holds trace traceMember row
    unfold traceRows
    exact List.mem_append_left _ rowMember
    exact member
  · intro row rowMember
    rcases List.mem_flatMap.mp rowMember with
      ⟨trace, traceMember, member⟩
    apply holds trace traceMember row
    unfold traceRows
    exact List.mem_append_right _ member

/-- Terminal PiRLC projection acceptance is coefficient-exact or exposes the
named bounded nonzero-root event. -/
theorem terminal_exact_or_badRoot (assignment : Nat → Nat)
    (assignmentCanonical : ∀ column, assignment column < goldilocksP)
    (constantOne : assignment 0 = 1)
    (holds : TerminalHolds assignment) :
    BatchExact (BatchIdentity terminalTraces assignment) ∨
      BatchBadRoot K.ops (BatchIdentity terminalTraces assignment) := by
  apply batchAccepted_implies_exact_or_badRoot
  exact terminal_batchAccepted assignment assignmentCanonical constantOne holds

/-- Independent native execution of every recursive projection identity
reconstructs the exact generated trace rows. -/
private theorem recursive_definitions_wellFormed :
    ∀ trace ∈ recursiveTraces,
      Program.WellFormed trace.inputColumns trace.definitions := by
  native_decide

private theorem terminal_definitions_wellFormed :
    ∀ trace ∈ terminalTraces,
      Program.WellFormed trace.inputColumns trace.definitions := by
  native_decide

theorem recursive_complete
    {assignment : Nat → Nat}
    (native : ∀ trace ∈ recursiveTraces,
      trace.ExecutionWitness assignment) :
    RecursiveHolds assignment := by
  intro trace member
  apply trace.native_complete assignment
    (recursive_definitions_wellFormed trace member)
    (definitions_canonical trace
      (List.mem_append_left terminalTraces member))
  exact native trace member

/-- Terminal-fold counterpart of `recursive_complete`. -/
theorem terminal_complete
    {assignment : Nat → Nat}
    (native : ∀ trace ∈ terminalTraces,
      trace.ExecutionWitness assignment) :
    TerminalHolds assignment := by
  intro trace member
  apply trace.native_complete assignment
    (terminal_definitions_wellFormed trace member)
    (definitions_canonical trace
      (List.mem_append_right recursiveTraces member))
  exact native trace member

end Nightstream.Implementation.R1CS.FPrimeFullHistoryProjection
