import Nightstream.Implementation.NebulaV2.Commitment.Compact.ChainHashFrameRows
import Nightstream.Implementation.NebulaV2.Memory.Carry.PoseidonRows

/-!
Contract: exact Poseidon2 traces for V2 compact-chain header, leaf, and link
frames.

Assurance tier: implementation model and cryptographic primitive semantics.

Owns the exact absorb schedules for 11-field headers, 64-field leaves, and
11-field links; pure column-independent digest semantics; trace-to-frame
linkage; row soundness; and local honest completeness.

Does not own Poseidon2 collision resistance, token computation, the chain
state update, absolute generated columns, or Rust conformance.

Emits constraints: yes.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.NebulaV2.CompactChainPoseidonRows

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Poseidon2Sponge
open Nightstream.Implementation.NebulaV2.CompactChainHashFrame
open Nightstream.Protocol.NebulaV2

def compactSchedule : Input → List ValueSchedule
  | .header _ _ _ => [.absorb 4, .absorb 4, .absorb 3, .pad]
  | .leaf _ _ _ _ => List.replicate 16 (.absorb 4) ++ [.pad]
  | .link _ _ _ _ => [.absorb 4, .absorb 4, .absorb 3, .pad]

theorem header_schedule_exact (role : CompactCommit.Role)
    (profile : Profile.Identity) (plan : Digest.Value) :
    compactSchedule (.header role profile plan) =
      [.absorb 4, .absorb 4, .absorb 3, .pad] := rfl

theorem leaf_schedule_exact (role : CompactCommit.Role)
    (profile : Profile.Identity) (plan : Digest.Value)
    (token : CompactCommit.Token) :
    (compactSchedule (.leaf role profile plan token)).length = 17 ∧
      ((compactSchedule (.leaf role profile plan token)).filter
        (· = .absorb 4)).length = 16 ∧
      ((compactSchedule (.leaf role profile plan token)).filter
        (· = .pad)).length = 1 := by
  simp [compactSchedule]

theorem link_schedule_exact (role : CompactCommit.Role)
    (index : Fin Lifecycle.claimsPerSegment) (prior leaf : Digest.Value) :
    compactSchedule (.link role index prior leaf) =
      [.absorb 4, .absorb 4, .absorb 3, .pad] := rfl

def representativeRounds (input : Input) : List Round :=
  (compactSchedule input).map MemoryCarryPoseidonRows.representativeRound

theorem representativeRounds_schedule (input : Input) :
    valueSchedules (representativeRounds input) = compactSchedule input := by
  rw [representativeRounds, valueSchedules, List.map_map]
  generalize compactSchedule input = schedules
  induction schedules with
  | nil => rfl
  | cons head tail inductionHypothesis =>
      simp only [List.map_cons, List.cons.injEq]
      exact ⟨MemoryCarryPoseidonRows.representativeRound_schedule head,
        inductionHypothesis⟩

/-- Pure fixed Poseidon2 function selected by the exact typed frame. -/
def pureHash (input : Input) (lane : Nat) : Nat :=
  runValueRounds (representativeRounds input) (encode input) (fun _ => 0) lane

namespace Framed

def rows (frameRows : List Row) (trace : Trace) : List Row :=
  frameRows ++ trace.rows

/-- Structural generated-trace certificate. The source-frame theorem is a
separate row consequence and is not hidden in this certificate. -/
structure Valid
    (input : Input) (frameRows : List Row) (frameColumns : List Nat)
    (trace : Trace) : Prop where
  exactInputColumns : trace.inputColumns = frameColumns
  exactSchedule : valueSchedules trace.rounds = compactSchedule input
  traceValid : trace.Valid (rows frameRows trace)

theorem frame_rows_hold
    {frameRows : List Row} {trace : Trace} {assignment : Nat → Nat}
    (holds : Satisfies (rows frameRows trace) assignment) :
    Satisfies frameRows assignment := by
  intro row member
  exact holds row (List.mem_append_left _ member)

theorem output_exact
    {input : Input} {frameRows : List Row} {frameColumns : List Nat}
    {trace : Trace} (valid : Valid input frameRows frameColumns trace)
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (frameExact : frameColumns.map assignment = encode input)
    (holds : Satisfies (rows frameRows trace) assignment) :
    ∀ lane : Fin 4,
      assignment (trace.outputColumns.getD lane.val 0) =
        pureHash input lane.val := by
  have traceSound := trace_values_sound valid.traceValid canonical one holds
  have inputValues : trace.inputColumns.map assignment = encode input := by
    rw [valid.exactInputColumns]
    exact frameExact
  have schedules :
      valueSchedules trace.rounds =
        valueSchedules (representativeRounds input) :=
    valid.exactSchedule.trans (representativeRounds_schedule input).symm
  have runEqual := runValueRounds_eq_of_schedules schedules
    (encode input) (fun _ => 0)
  intro lane
  calc
    assignment (trace.outputColumns.getD lane.val 0) =
        runValueRounds trace.rounds
          (trace.inputColumns.map assignment) (fun _ => 0) lane.val :=
      traceSound lane.val lane.isLt
    _ = runValueRounds trace.rounds
          (encode input) (fun _ => 0) lane.val := by rw [inputValues]
    _ = runValueRounds (representativeRounds input)
          (encode input) (fun _ => 0) lane.val := congrFun runEqual lane.val
    _ = pureHash input lane.val := rfl

structure Honest (frameRows : List Row) (trace : Trace)
    (assignment : Nat → Nat) : Prop where
  frame : Satisfies frameRows assignment
  trace : trace.ExecutionWitness assignment

theorem rows_complete
    {frameRows : List Row} {trace : Trace} {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (honest : Honest frameRows trace assignment) :
    Satisfies (rows frameRows trace) assignment := by
  intro row member
  rw [rows, List.mem_append] at member
  rcases member with frameMember | traceMember
  · exact honest.frame row frameMember
  · exact Trace.execution_complete canonical one honest.trace row traceMember

end Framed

end Nightstream.Implementation.NebulaV2.CompactChainPoseidonRows
