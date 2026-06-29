import Nightstream.Chip8.Stage2.WitnessMemoryBinding
import Nightstream.Chip8.Stage3.ContinuityBridge
import SuperNeo.Primitives.PolynomialBridge

namespace Nightstream.Chip8.ExecutionSemantics

open Nightstream.Chip8
open Nightstream.Chip8.FetchDecodeBinding
open Nightstream.Chip8.DecodeAddressBinding
open Nightstream.Chip8.WitnessMemoryBinding
open Nightstream.Chip8.ContinuityBridge
open SuperNeo

abbrev F := SuperNeo.Fq
abbrev MachineState := WitnessMemoryBinding.MachineState
abbrev InitialState := WitnessMemoryBinding.InitialState

def RegisterIndexBound (idx : Nat) : Prop :=
  idx < 16

def RamAddressBound (addr : Nat) : Prop :=
  addr < 4096

def StateEq (left right : MachineState) : Prop :=
  left.pc = right.pc ∧
    left.i = right.i ∧
    (∀ idx, RegisterIndexBound idx → left.v idx = right.v idx) ∧
    (∀ addr, RamAddressBound addr → left.ram addr = right.ram addr)

structure ExternalSchedule where
deriving DecidableEq, Repr

def byteAdd (a b : Nat) : Nat :=
  (a + b) % 256

def skipEqBit (a b : Nat) : Nat :=
  if a = b then 1 else 0

def RegistersPreserved (pre post : MachineState) : Prop :=
  ∀ idx, RegisterIndexBound idx → post.v idx = pre.v idx

def RegistersPreservedExcept (pre post : MachineState) (x : Nat) : Prop :=
  ∀ idx, RegisterIndexBound idx → idx ≠ x → post.v idx = pre.v idx

def RegistersPreservedAbove (pre post : MachineState) (x : Nat) : Prop :=
  ∀ idx, RegisterIndexBound idx → x < idx → post.v idx = pre.v idx

def RamPreserved (pre post : MachineState) : Prop :=
  ∀ addr, RamAddressBound addr → post.ram addr = pre.ram addr

def RamPrefixStored
  {Addr : Type*}
  (pre post : MachineState)
  (dec : DecodedStep Addr) : Prop :=
  ∀ idx, idx ≤ dec.x → post.ram (pre.i + idx) = pre.v idx

def RamPreservedOutsidePrefix
  {Addr : Type*}
  (pre post : MachineState)
  (dec : DecodedStep Addr) : Prop :=
  ∀ addr, (∀ idx, idx ≤ dec.x → addr ≠ pre.i + idx) → post.ram addr = pre.ram addr

def RegistersLoadedPrefix
  {Addr : Type*}
  (pre post : MachineState)
  (dec : DecodedStep Addr) : Prop :=
  ∀ idx, idx ≤ dec.x → post.v idx = pre.ram (pre.i + idx)

def ContinuityRowBound
  (stepIdx : Nat)
  (N : Nat)
  (β1 β2 : F)
  (shiftClaim : ContinuityBridge.LaneShiftClaim F)
  (shiftProof : ContinuityBridge.LaneShiftWitness F Unit)
  (currentRow : ContinuityBridge.ContinuityRow F)
  (rowClaim : ContinuityBridge.RowBindingClaim F Unit)
  (z : Nightstream.Chip8.Witness F) : Prop :=
  ContinuityBridge.ContinuityBound N β1 β2 shiftClaim shiftProof currentRow ∧
    currentRow.rowIndex = stepIdx ∧
    currentRow.pcNext = z 2 ∧
    currentRow.xIdx = z 20 ∧
    currentRow.isMemOp = z 19 ∧
    currentRow.burstLast = z 22 ∧
    rowClaim.rowIndex = stepIdx ∧
    ContinuityBridge.RowBound rowClaim z

def MicrostepCorrect
  {Addr : Type*}
  (_rom : Program)
  (_σ : ExternalSchedule)
  (dec : DecodedStep Addr)
  (pre post : MachineState) : Prop :=
  match dec.opcodeId with
  | .ldImm =>
      post.pc = pre.pc + 1 ∧
        post.i = pre.i ∧
        post.v dec.x = dec.kk ∧
        RegistersPreservedExcept pre post dec.x ∧
        RamPreserved pre post
  | .addImm =>
      post.pc = pre.pc + 1 ∧
        post.i = pre.i ∧
        post.v dec.x = byteAdd (pre.v dec.x) dec.kk ∧
        RegistersPreservedExcept pre post dec.x ∧
        RamPreserved pre post
  | .mov =>
      post.pc = pre.pc + 1 ∧
        post.i = pre.i ∧
        post.v dec.x = pre.v dec.y ∧
        RegistersPreservedExcept pre post dec.x ∧
        RamPreserved pre post
  | .addReg =>
      post.pc = pre.pc + 1 ∧
        post.i = pre.i ∧
        post.v dec.x = byteAdd (pre.v dec.x) (pre.v dec.y) ∧
        RegistersPreservedExcept pre post dec.x ∧
        RamPreserved pre post
  | .skipEqImm =>
      post.pc = pre.pc + 1 + skipEqBit (pre.v dec.x) dec.kk ∧
        post.i = pre.i ∧
        RegistersPreserved pre post ∧
        RamPreserved pre post
  | .jump =>
      post.pc = dec.nnnWord ∧
        post.i = pre.i ∧
        RegistersPreserved pre post ∧
        RamPreserved pre post
  | .ldI =>
      post.pc = pre.pc + 1 ∧
        post.i = dec.nnnAddr ∧
        RegistersPreserved pre post ∧
        RamPreserved pre post
  | .storeRegs =>
      post.pc = pre.pc + burstLastValue dec ∧
        post.i = pre.i ∧
        RegistersPreserved pre post ∧
        RamPrefixStored pre post dec ∧
        RamPreservedOutsidePrefix pre post dec
  | .loadRegs =>
      post.pc = pre.pc + burstLastValue dec ∧
        post.i = pre.i ∧
        RegistersLoadedPrefix pre post dec ∧
        RegistersPreservedAbove pre post dec.x ∧
        RamPreserved pre post

def InstructionCorrect
  {Addr : Type*}
  (rom : Program)
  (σ : ExternalSchedule)
  (dec : DecodedStep Addr)
  (pre post : MachineState) : Prop :=
  match dec.opcodeId with
  | .storeRegs =>
      post.pc = pre.pc + 1 ∧
        post.i = pre.i ∧
        RegistersPreserved pre post ∧
        RamPrefixStored pre post dec ∧
        RamPreservedOutsidePrefix pre post dec
  | .loadRegs =>
      post.pc = pre.pc + 1 ∧
        post.i = pre.i ∧
        RegistersLoadedPrefix pre post dec ∧
        RegistersPreservedAbove pre post dec.x ∧
        RamPreserved pre post
  | _ => MicrostepCorrect rom σ dec pre post

structure ExecutionFrame (Addr : Type*) where
  dec : DecodedStep Addr
  pre : MachineState
  post : MachineState
  row : Nightstream.Chip8.Witness F

def ExecutionLinked
  {Addr : Type*}
  : List (ExecutionFrame Addr) → Prop
  | [] => True
  | [_] => True
  | a :: b :: rest => StateEq a.post b.pre ∧ ExecutionLinked (b :: rest)

def InitialStateMatches
  (init : InitialState)
  (st : MachineState) : Prop :=
  st.pc = init.pc ∧
    st.i = init.i ∧
    (∀ idx, RegisterIndexBound idx → st.v idx = init.v idx) ∧
    (∀ addr, RamAddressBound addr → st.ram addr = init.ram addr)

def StartBoundaryFrame
  {Addr : Type*}
  (frame : ExecutionFrame Addr) : Prop :=
  ContinuityBridge.StartBoundaryBound
    { isMemOp := frame.row 19, xIdx := frame.row 20 }

def FirstRowPubliclyBound
  {Addr : Type*}
  (init : InitialState)
  (frame : ExecutionFrame Addr) : Prop :=
  InitialStateMatches init frame.pre ∧
    StartBoundaryFrame frame

def FinalBoundaryFrame
  {Addr : Type*}
  (frame : ExecutionFrame Addr) : Prop :=
  ContinuityBridge.FinalBoundaryBound
    { isMemOp := frame.row 19, burstLast := frame.row 22 }

def BoundaryTraceBound
  {Addr : Type*}
  (init : InitialState)
  (trace : List (ExecutionFrame Addr)) : Prop :=
  match trace with
  | [] => False
  | first :: _ =>
      InitialStateMatches init first.pre ∧
        StartBoundaryFrame first ∧
        match trace.reverse with
        | [] => False
        | last :: _ => FinalBoundaryFrame last

def ExecutionFrameBound
  {Addr : Type*}
  (rom : Program)
  (σ : ExternalSchedule)
  (frame : ExecutionFrame Addr) : Prop :=
  Nightstream.Chip8.wf frame.row ∧
    WitnessMemoryBinding.WitnessBinds (K := F) frame.pre frame.post frame.dec frame.row ∧
    MicrostepCorrect rom σ frame.dec frame.pre frame.post

theorem executionFrameBound_witnessBinds
  {Addr : Type*}
  {rom : Program}
  {σ : ExternalSchedule}
  {frame : ExecutionFrame Addr}
  (h : ExecutionFrameBound rom σ frame) :
  WitnessMemoryBinding.WitnessBinds (K := F) frame.pre frame.post frame.dec frame.row := by
  exact h.2.1

theorem initialStateMatches_of_firstRowPubliclyBound
  {Addr : Type*}
  {init : InitialState}
  {frame : ExecutionFrame Addr}
  (h : FirstRowPubliclyBound init frame) :
  InitialStateMatches init frame.pre :=
  h.1

theorem startBoundaryFrame_of_firstRowPubliclyBound
  {Addr : Type*}
  {init : InitialState}
  {frame : ExecutionFrame Addr}
  (h : FirstRowPubliclyBound init frame) :
  StartBoundaryFrame frame :=
  h.2

theorem firstRowPubliclyBound_of_boundaryTrace
  {Addr : Type*}
  {init : InitialState}
  {first : ExecutionFrame Addr}
  {rest : List (ExecutionFrame Addr)}
  (h : BoundaryTraceBound init (first :: rest)) :
  FirstRowPubliclyBound init first := by
  exact ⟨h.1, h.2.1⟩

theorem executionFrameBound_microstepCorrect
  {Addr : Type*}
  {rom : Program}
  {σ : ExternalSchedule}
  {frame : ExecutionFrame Addr}
  (h : ExecutionFrameBound rom σ frame) :
  MicrostepCorrect rom σ frame.dec frame.pre frame.post := by
  exact h.2.2

def ContinuityTraceBound
  {Addr : Type*}
  : Nat → List (ExecutionFrame Addr) → Prop
  | _, [] => True
  | stepIdx, frame :: rest =>
      (∃ N β1 β2 shiftClaim shiftProof currentRow rowClaim,
        ContinuityRowBound stepIdx N β1 β2 shiftClaim shiftProof currentRow rowClaim frame.row) ∧
        ContinuityTraceBound (stepIdx + 1) rest

def PreparedStepTraceBound
  {Addr W Z Commitment : Type*}
  (rootEncode : ContinuityBridge.RootEncode W Z F)
  (ajtaiCommit : Z → Commitment)
  (trace : List (ExecutionFrame Addr))
  (preparedSteps : List (ContinuityBridge.PreparedStep W Z Commitment F)) : Prop :=
  List.Forall₂
    (fun frame step =>
      ContinuityBridge.PreparedStepBound rootEncode ajtaiCommit frame.row step)
    trace preparedSteps

def ExecutionCorrect
  {Addr : Type*}
  (rom : Program)
  (σ : ExternalSchedule)
  (init : InitialState)
  (trace : List (ExecutionFrame Addr)) : Prop :=
  ExecutionLinked trace ∧
    List.Forall (ExecutionFrameBound rom σ) trace ∧
    ContinuityTraceBound 0 trace ∧
    BoundaryTraceBound init trace

def FinalState
  {Addr : Type*}
  (trace : List (ExecutionFrame Addr)) : Option MachineState :=
  match trace.reverse with
  | [] => none
  | frame :: _ => some frame.post

abbrev GoalPredicate := MachineState → Prop

theorem instructionCorrect_of_nonBurstMicrostep
  {Addr : Type*}
  {rom : Program}
  {σ : ExternalSchedule}
  {dec : DecodedStep Addr}
  {pre post : MachineState}
  (hStore : dec.opcodeId ≠ .storeRegs)
  (hLoad : dec.opcodeId ≠ .loadRegs)
  (hMicro : MicrostepCorrect rom σ dec pre post) :
  InstructionCorrect rom σ dec pre post := by
  cases hOpcode : dec.opcodeId <;>
    simp [InstructionCorrect, MicrostepCorrect, hOpcode] at hStore hLoad ⊢
  · simpa [MicrostepCorrect, hOpcode] using hMicro
  · simpa [MicrostepCorrect, hOpcode] using hMicro
  · simpa [MicrostepCorrect, hOpcode] using hMicro
  · simpa [MicrostepCorrect, hOpcode] using hMicro
  · simpa [MicrostepCorrect, hOpcode] using hMicro
  · simpa [MicrostepCorrect, hOpcode] using hMicro
  · simpa [MicrostepCorrect, hOpcode] using hMicro

theorem executionCorrect_of_trace
  {Addr : Type*}
  {rom : Program}
  {σ : ExternalSchedule}
  {init : InitialState}
  {trace : List (ExecutionFrame Addr)}
  (hLinked : ExecutionLinked trace)
  (hFrames : List.Forall (ExecutionFrameBound rom σ) trace)
  (hContinuity : ContinuityTraceBound 0 trace)
  (hBoundary : BoundaryTraceBound init trace) :
  ExecutionCorrect rom σ init trace := by
  cases trace with
  | nil =>
      exfalso
      exact hBoundary
  | cons first rest =>
      change ExecutionLinked (first :: rest) ∧
          List.Forall (ExecutionFrameBound rom σ) (first :: rest) ∧
          ContinuityTraceBound 0 (first :: rest) ∧
          BoundaryTraceBound init (first :: rest)
      exact ⟨hLinked, hFrames, hContinuity, hBoundary⟩

private theorem rowBound_of_continuityRowBound
  {stepIdx N : Nat}
  {β1 β2 : F}
  {shiftClaim : ContinuityBridge.LaneShiftClaim F}
  {shiftProof : ContinuityBridge.LaneShiftWitness F Unit}
  {currentRow : ContinuityBridge.ContinuityRow F}
  {rowClaim : ContinuityBridge.RowBindingClaim F Unit}
  {z : Nightstream.Chip8.Witness F}
  (h :
    ContinuityRowBound stepIdx N β1 β2 shiftClaim shiftProof currentRow rowClaim z) :
  ContinuityBridge.RowBound rowClaim z := by
  exact h.2.2.2.2.2.2.2

theorem preparedStepTraceBound_of_continuity
  {Addr W Z Commitment : Type*}
  {stepIdx : Nat}
  {rootEncode : ContinuityBridge.RootEncode W Z F}
  {ajtaiCommit : Z → Commitment}
  {trace : List (ExecutionFrame Addr)}
  (hCont : ContinuityTraceBound stepIdx trace) :
  PreparedStepTraceBound rootEncode ajtaiCommit trace
    (trace.map (fun frame =>
      ContinuityBridge.mkPreparedStep rootEncode ajtaiCommit frame.row)) := by
  induction trace generalizing stepIdx with
  | nil =>
      simp [PreparedStepTraceBound]
  | cons frame rest ih =>
      rcases hCont with ⟨hHead, hTail⟩
      rcases hHead with ⟨N, β1, β2, shiftClaim, shiftProof, currentRow, rowClaim, hRow⟩
      refine List.Forall₂.cons ?_ ?_
      · exact ContinuityBridge.preparedStepBound_of_rowBinding
          (claim := rowClaim) (z := frame.row)
          (rootEncode := rootEncode) (ajtaiCommit := ajtaiCommit)
          (rowBound_of_continuityRowBound hRow)
      · simpa [PreparedStepTraceBound] using ih hTail

theorem preparedStepTraceBound_of_execution
  {Addr W Z Commitment : Type*}
  {rom : Program}
  {σ : ExternalSchedule}
  {init : InitialState}
  {trace : List (ExecutionFrame Addr)}
  {rootEncode : ContinuityBridge.RootEncode W Z F}
  {ajtaiCommit : Z → Commitment}
  (hExec : ExecutionCorrect rom σ init trace) :
  PreparedStepTraceBound rootEncode ajtaiCommit trace
    (trace.map (fun frame =>
      ContinuityBridge.mkPreparedStep rootEncode ajtaiCommit frame.row)) := by
  exact preparedStepTraceBound_of_continuity hExec.2.2.1

theorem goalPredicate_of_execution
  {Addr : Type*}
  {rom : Program}
  {σ : ExternalSchedule}
  {init : InitialState}
  {trace : List (ExecutionFrame Addr)}
  {goal : GoalPredicate}
  {_stf : MachineState}
  (_hExec : ExecutionCorrect rom σ init trace)
  (hGoal : match FinalState trace with
    | some st => goal st
    | none => False)
  (hFinal : FinalState trace = some _stf) :
  goal _stf := by
  simpa [hFinal] using hGoal

end Nightstream.Chip8.ExecutionSemantics
