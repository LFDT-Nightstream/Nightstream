import Nightstream.Chip8.Execution.ExecutionSemantics
import Nightstream.Chip8.Execution.BurstSession
import Nightstream.Chip8.Stage2.WitnessMemoryBinding
import Nightstream.Chip8.Stage3.ContinuityBridge
import Nightstream.ClaimedMemorySemantics
import SuperNeo.Primitives.PolynomialBridge

namespace Nightstream.Chip8.StepComposition

open Nightstream.Chip8
open Nightstream.Chip8.ExecutionSemantics
open Nightstream.Chip8.BurstSession
open Nightstream.Chip8.FetchDecodeBinding
open Nightstream.Chip8.DecodeAddressBinding
open Nightstream.Chip8.WitnessMemoryBinding
open Nightstream.Chip8.ContinuityBridge
open SuperNeo

abbrev F := ExecutionSemantics.F
abbrev MachineState := ExecutionSemantics.MachineState
abbrev InitialState := ExecutionSemantics.InitialState
abbrev ExternalSchedule := ExecutionSemantics.ExternalSchedule

def StateWellFormed (st : MachineState) : Prop :=
  st.pc < 4096 ∧
    st.i < 4096 ∧
    (∀ idx, st.v idx < 256) ∧
    (∀ addr, st.ram addr < 256)

def FetchDecodeBound
  {Addr : Type*}
  (rom : Program)
  (pc : Nat)
  (dec : DecodedStep Addr) : Prop :=
  FetchDecodeBinding.FetchDecodeBound rom pc dec.toDecodedCore

abbrev byteAdd := ExecutionSemantics.byteAdd
abbrev skipEqBit := ExecutionSemantics.skipEqBit

def lookupValueOf
  {Addr : Type*}
  (pre : MachineState)
  (dec : DecodedStep Addr) : Nat :=
  match dec.opcodeId with
  | .ldImm => dec.kk
  | .addImm => byteAdd (pre.v dec.x) dec.kk
  | .mov => pre.v dec.y
  | .addReg => byteAdd (pre.v dec.x) (pre.v dec.y)
  | .skipEqImm => skipEqBit (pre.v dec.x) dec.kk
  | .jump => 0
  | .ldI => 0
  | .storeRegs => 0
  | .loadRegs => 0

def LookupBound
  {Addr : Type*}
  (dec : DecodedStep Addr)
  (pre : MachineState)
  (z : Nightstream.Chip8.Witness F) : Prop :=
  z colLookupOutput = (lookupValueOf pre dec : F)

def FramebufferBound
  {Addr : Type*}
  (_pre _post : MachineState)
  (_dec : DecodedStep Addr) : Prop :=
  True

def ScheduleBound
  {Addr : Type*}
  (_σ : ExternalSchedule)
  (_stepIdx : Nat)
  (_pre _post : MachineState)
  (_dec : DecodedStep Addr) : Prop :=
  True

abbrev RegistersPreserved := ExecutionSemantics.RegistersPreserved
abbrev RegistersPreservedExcept := ExecutionSemantics.RegistersPreservedExcept
abbrev RegistersPreservedAbove := ExecutionSemantics.RegistersPreservedAbove
abbrev RamPreserved := ExecutionSemantics.RamPreserved
abbrev RamPrefixStored := @ExecutionSemantics.RamPrefixStored
abbrev RamPreservedOutsidePrefix := @ExecutionSemantics.RamPreservedOutsidePrefix
abbrev RegistersLoadedPrefix := @ExecutionSemantics.RegistersLoadedPrefix

def MemoryBound
  {Addr : Type*}
  (pre post : MachineState)
  (init : InitialState)
  (dec : DecodedStep Addr)
  (z : Nightstream.Chip8.Witness F) : Prop :=
  WitnessMemoryBinding.MemoryBound (K := F) pre post init dec z ∧
    match dec.opcodeId with
    | .ldImm | .addImm | .mov | .addReg =>
        RegistersPreservedExcept pre post dec.x ∧ RamPreserved pre post
    | .skipEqImm | .jump | .ldI =>
        RegistersPreserved pre post ∧ RamPreserved pre post
    | .storeRegs =>
        RegistersPreserved pre post ∧
          RamPrefixStored pre post dec ∧
          RamPreservedOutsidePrefix pre post dec
    | .loadRegs =>
        RegistersLoadedPrefix pre post dec ∧
          RegistersPreservedAbove pre post dec.x ∧
          RamPreserved pre post

def ContinuityRowBound := @ExecutionSemantics.ContinuityRowBound
def MicrostepCorrect := @ExecutionSemantics.MicrostepCorrect
def InstructionCorrect := @ExecutionSemantics.InstructionCorrect
def BurstScheduleCorrect := @BurstSession.BurstScheduleCorrect
abbrev ExecutionFrame := ExecutionSemantics.ExecutionFrame
abbrev ExecutionLinked := @ExecutionSemantics.ExecutionLinked
abbrev InitialStateMatches := @ExecutionSemantics.InitialStateMatches
abbrev StartBoundaryFrame := @ExecutionSemantics.StartBoundaryFrame
abbrev FinalBoundaryFrame := @ExecutionSemantics.FinalBoundaryFrame
abbrev BoundaryTraceBound := @ExecutionSemantics.BoundaryTraceBound
abbrev ExecutionFrameBound := @ExecutionSemantics.ExecutionFrameBound
abbrev executionFrameBound_witnessBinds :=
  @ExecutionSemantics.executionFrameBound_witnessBinds
abbrev executionFrameBound_microstepCorrect :=
  @ExecutionSemantics.executionFrameBound_microstepCorrect
abbrev ContinuityTraceBound := @ExecutionSemantics.ContinuityTraceBound
abbrev PreparedStepTraceBound := @ExecutionSemantics.PreparedStepTraceBound
abbrev ExecutionCorrect := @ExecutionSemantics.ExecutionCorrect
abbrev FinalState := @ExecutionSemantics.FinalState
abbrev GoalPredicate := ExecutionSemantics.GoalPredicate

private theorem primaryValue_eq_vx_of_nonBurst
  {Addr : Type*}
  {st : MachineState}
  {dec : DecodedStep Addr}
  (hStore : dec.opcodeId ≠ .storeRegs)
  (hLoad : dec.opcodeId ≠ .loadRegs) :
  WitnessMemoryBinding.primaryValue st dec = st.v dec.x := by
  simp [WitnessMemoryBinding.primaryValue, WitnessMemoryBinding.primaryIndex,
    DecodeAddressBinding.activeXIndex_of_nonBurst hStore hLoad]

private theorem routing_lookup_to_vx
  {z : Nightstream.Chip8.Witness F}
  (h : Nightstream.Chip8.chip8RoutingSound z) :
  z colWritesLookupToX = 1 → z colRegXNext = z colLookupOutput := h.1

private theorem routing_mem_to_vx
  {z : Nightstream.Chip8.Witness F}
  (h : Nightstream.Chip8.chip8RoutingSound z) :
  z colWritesMemToX = 1 → z colRegXNext = z colMemValue := h.2.1

private theorem routing_preserve_vx
  {z : Nightstream.Chip8.Witness F}
  (h : Nightstream.Chip8.chip8RoutingSound z) :
  z colPreservesX = 1 → z colRegXNext = z colRegX := h.2.2.1

private theorem routing_write_i
  {z : Nightstream.Chip8.Witness F}
  (h : Nightstream.Chip8.chip8RoutingSound z) :
  z colWritesNnnToI = 1 → z colINext = z colNnnAddr := h.2.2.2.1

private theorem routing_preserve_i
  {z : Nightstream.Chip8.Witness F}
  (h : Nightstream.Chip8.chip8RoutingSound z) :
  z colWritesNnnToI = 0 → z colINext = z colIReg := h.2.2.2.2.1

private theorem routing_jump_pc
  {z : Nightstream.Chip8.Witness F}
  (h : Nightstream.Chip8.chip8RoutingSound z) :
  z colIsJump = 1 → z colPcNext = z colNnnWord := h.2.2.2.2.2.1

private theorem routing_branch_pc
  {z : Nightstream.Chip8.Witness F}
  (h : Nightstream.Chip8.chip8RoutingSound z) :
  Nightstream.Chip8.wf z ∧ z colIsBranch = 1 →
    z colPcNext = z colPc + 1 + z colLookupOutput := h.2.2.2.2.2.2.1

private theorem routing_mem_pc
  {z : Nightstream.Chip8.Witness F}
  (h : Nightstream.Chip8.chip8RoutingSound z) :
  z colIsMemOp = 1 → z colPcNext = z colPc + z colBurstLast := h.2.2.2.2.2.2.2.1

private theorem routing_default_pc
  {z : Nightstream.Chip8.Witness F}
  (h : Nightstream.Chip8.chip8RoutingSound z) :
  Nightstream.Chip8.wf z ∧ z colIsJump = 0 ∧ z colIsBranch = 0 ∧
      z colIsMemOp = 0 → z colPcNext = z colPc + 1 :=
  h.2.2.2.2.2.2.2.2.1

private theorem routing_ram_addr
  {z : Nightstream.Chip8.Witness F}
  (h : Nightstream.Chip8.chip8RoutingSound z) :
  z colIsMemOp = 1 → z colRamAddr = z colIReg + z colXIdx := h.2.2.2.2.2.2.2.2.2.1

private theorem witness_lookupFlag
  {Addr : Type*}
  {pre post : MachineState}
  {dec : DecodedStep Addr}
  {z : Nightstream.Chip8.Witness F}
  (hWitness : WitnessMemoryBinding.WitnessBinds (K := F) pre post dec z) :
  z colWritesLookupToX =
      flagWritesLookupToX (Nightstream.Chip8.behaviorFlags (K := F) dec.behavior) := by
  simpa [Nightstream.Chip8.flags] using
    congrArg flagWritesLookupToX (witnessBinds_flags hWitness)

private theorem witness_memFlag
  {Addr : Type*}
  {pre post : MachineState}
  {dec : DecodedStep Addr}
  {z : Nightstream.Chip8.Witness F}
  (hWitness : WitnessMemoryBinding.WitnessBinds (K := F) pre post dec z) :
  z colWritesMemToX =
      flagWritesMemToX (Nightstream.Chip8.behaviorFlags (K := F) dec.behavior) := by
  simpa [Nightstream.Chip8.flags] using
    congrArg flagWritesMemToX (witnessBinds_flags hWitness)

private theorem witness_writeIFlag
  {Addr : Type*}
  {pre post : MachineState}
  {dec : DecodedStep Addr}
  {z : Nightstream.Chip8.Witness F}
  (hWitness : WitnessMemoryBinding.WitnessBinds (K := F) pre post dec z) :
  z colWritesNnnToI =
      flagWritesNnnToI (Nightstream.Chip8.behaviorFlags (K := F) dec.behavior) := by
  simpa [Nightstream.Chip8.flags] using
    congrArg flagWritesNnnToI (witnessBinds_flags hWitness)

private theorem witness_jumpFlag
  {Addr : Type*}
  {pre post : MachineState}
  {dec : DecodedStep Addr}
  {z : Nightstream.Chip8.Witness F}
  (hWitness : WitnessMemoryBinding.WitnessBinds (K := F) pre post dec z) :
  z colIsJump = flagIsJump (Nightstream.Chip8.behaviorFlags (K := F) dec.behavior) := by
  simpa [Nightstream.Chip8.flags] using
    congrArg flagIsJump (witnessBinds_flags hWitness)

private theorem witness_branchFlag
  {Addr : Type*}
  {pre post : MachineState}
  {dec : DecodedStep Addr}
  {z : Nightstream.Chip8.Witness F}
  (hWitness : WitnessMemoryBinding.WitnessBinds (K := F) pre post dec z) :
  z colIsBranch = flagIsBranch (Nightstream.Chip8.behaviorFlags (K := F) dec.behavior) := by
  simpa [Nightstream.Chip8.flags] using
    congrArg flagIsBranch (witnessBinds_flags hWitness)

private theorem witness_memOpFlag
  {Addr : Type*}
  {pre post : MachineState}
  {dec : DecodedStep Addr}
  {z : Nightstream.Chip8.Witness F}
  (hWitness : WitnessMemoryBinding.WitnessBinds (K := F) pre post dec z) :
  z colIsMemOp = flagIsMemOp (Nightstream.Chip8.behaviorFlags (K := F) dec.behavior) := by
  simpa [Nightstream.Chip8.flags] using
    congrArg flagIsMemOp (witnessBinds_flags hWitness)

private theorem witness_lookupFlag_of_opcode
  {Addr : Type*}
  {pre post : MachineState}
  {dec : DecodedStep Addr}
  {z : Nightstream.Chip8.Witness F}
  {opcode : OpcodeId}
  (hWitness : WitnessMemoryBinding.WitnessBinds (K := F) pre post dec z)
  (hOpcode : dec.opcodeId = opcode) :
  z colWritesLookupToX =
      flagWritesLookupToX
        (Nightstream.Chip8.behaviorFlags (K := F) (FetchDecodeBinding.behaviorOfOpcode opcode)) := by
  have h := witness_lookupFlag hWitness
  change z colWritesLookupToX =
      flagWritesLookupToX
        (Nightstream.Chip8.behaviorFlags (K := F) (DecodeAddressBinding.behavior dec)) at h
  simpa [DecodeAddressBinding.behavior, hOpcode] using h

private theorem witness_memFlag_of_opcode
  {Addr : Type*}
  {pre post : MachineState}
  {dec : DecodedStep Addr}
  {z : Nightstream.Chip8.Witness F}
  {opcode : OpcodeId}
  (hWitness : WitnessMemoryBinding.WitnessBinds (K := F) pre post dec z)
  (hOpcode : dec.opcodeId = opcode) :
  z colWritesMemToX =
      flagWritesMemToX
        (Nightstream.Chip8.behaviorFlags (K := F) (FetchDecodeBinding.behaviorOfOpcode opcode)) := by
  have h := witness_memFlag hWitness
  change z colWritesMemToX =
      flagWritesMemToX
        (Nightstream.Chip8.behaviorFlags (K := F) (DecodeAddressBinding.behavior dec)) at h
  simpa [DecodeAddressBinding.behavior, hOpcode] using h

private theorem witness_writeIFlag_of_opcode
  {Addr : Type*}
  {pre post : MachineState}
  {dec : DecodedStep Addr}
  {z : Nightstream.Chip8.Witness F}
  {opcode : OpcodeId}
  (hWitness : WitnessMemoryBinding.WitnessBinds (K := F) pre post dec z)
  (hOpcode : dec.opcodeId = opcode) :
  z colWritesNnnToI =
      flagWritesNnnToI
        (Nightstream.Chip8.behaviorFlags (K := F) (FetchDecodeBinding.behaviorOfOpcode opcode)) := by
  have h := witness_writeIFlag hWitness
  change z colWritesNnnToI =
      flagWritesNnnToI
        (Nightstream.Chip8.behaviorFlags (K := F) (DecodeAddressBinding.behavior dec)) at h
  simpa [DecodeAddressBinding.behavior, hOpcode] using h

private theorem witness_jumpFlag_of_opcode
  {Addr : Type*}
  {pre post : MachineState}
  {dec : DecodedStep Addr}
  {z : Nightstream.Chip8.Witness F}
  {opcode : OpcodeId}
  (hWitness : WitnessMemoryBinding.WitnessBinds (K := F) pre post dec z)
  (hOpcode : dec.opcodeId = opcode) :
  z colIsJump =
      flagIsJump
        (Nightstream.Chip8.behaviorFlags (K := F) (FetchDecodeBinding.behaviorOfOpcode opcode)) := by
  have h := witness_jumpFlag hWitness
  change z colIsJump =
      flagIsJump
        (Nightstream.Chip8.behaviorFlags (K := F) (DecodeAddressBinding.behavior dec)) at h
  simpa [DecodeAddressBinding.behavior, hOpcode] using h

private theorem witness_branchFlag_of_opcode
  {Addr : Type*}
  {pre post : MachineState}
  {dec : DecodedStep Addr}
  {z : Nightstream.Chip8.Witness F}
  {opcode : OpcodeId}
  (hWitness : WitnessMemoryBinding.WitnessBinds (K := F) pre post dec z)
  (hOpcode : dec.opcodeId = opcode) :
  z colIsBranch =
      flagIsBranch
        (Nightstream.Chip8.behaviorFlags (K := F) (FetchDecodeBinding.behaviorOfOpcode opcode)) := by
  have h := witness_branchFlag hWitness
  change z colIsBranch =
      flagIsBranch
        (Nightstream.Chip8.behaviorFlags (K := F) (DecodeAddressBinding.behavior dec)) at h
  simpa [DecodeAddressBinding.behavior, hOpcode] using h

private theorem witness_memOpFlag_of_opcode
  {Addr : Type*}
  {pre post : MachineState}
  {dec : DecodedStep Addr}
  {z : Nightstream.Chip8.Witness F}
  {opcode : OpcodeId}
  (hWitness : WitnessMemoryBinding.WitnessBinds (K := F) pre post dec z)
  (hOpcode : dec.opcodeId = opcode) :
  z colIsMemOp =
      flagIsMemOp
        (Nightstream.Chip8.behaviorFlags (K := F) (FetchDecodeBinding.behaviorOfOpcode opcode)) := by
  have h := witness_memOpFlag hWitness
  change z colIsMemOp =
      flagIsMemOp
        (Nightstream.Chip8.behaviorFlags (K := F) (DecodeAddressBinding.behavior dec)) at h
  simpa [DecodeAddressBinding.behavior, hOpcode] using h

theorem goldilocks_q_gt_256 : 256 < Goldilocks.q := by decide

theorem goldilocks_q_gt_2048 : 2048 < Goldilocks.q := by decide

theorem goldilocks_q_gt_4096 : 4096 < Goldilocks.q := by decide

theorem goldilocks_q_gt_4098 : 4098 < Goldilocks.q := by decide

theorem burstLastValue_le_one
  {Addr : Type*}
  (dec : DecodedStep Addr) :
  burstLastValue dec ≤ 1 := by
  by_cases hMem : dec.isMemOp = 1
  · simp [burstLastValue, hMem, eq4Eval]
    split <;> omega
  · simp [burstLastValue, hMem]

theorem stateWellFormed_pcBurst_lt_q
  {Addr : Type*}
  {pre : MachineState}
  (hPre : StateWellFormed pre)
  (dec : DecodedStep Addr) :
  pre.pc + burstLastValue dec < Goldilocks.q := by
  have hSum : pre.pc + burstLastValue dec < 4097 := by
    have hPc : pre.pc < 4096 := hPre.1
    have hBurst : burstLastValue dec ≤ 1 := burstLastValue_le_one dec
    omega
  have hLift : pre.pc + burstLastValue dec < 4098 := by
    exact lt_trans hSum (by decide)
  exact lt_trans hLift goldilocks_q_gt_4098

theorem natEq_of_fieldEq
  {a b : Nat}
  (ha : a < Goldilocks.q)
  (hb : b < Goldilocks.q)
  (h : (a : F) = (b : F)) :
  a = b := by
  have hMod : a ≡ b [MOD Goldilocks.q] := by
    simpa using (ZMod.natCast_eq_natCast_iff a b Goldilocks.q).mp h
  simpa [Nat.ModEq, Nat.mod_eq_of_lt ha, Nat.mod_eq_of_lt hb] using hMod

theorem stateWellFormed_pc_lt_q
  {st : MachineState}
  (h : StateWellFormed st) :
  st.pc < Goldilocks.q := by
  exact lt_trans h.1 goldilocks_q_gt_4096

theorem stateWellFormed_i_lt_q
  {st : MachineState}
  (h : StateWellFormed st) :
  st.i < Goldilocks.q := by
  exact lt_trans h.2.1 goldilocks_q_gt_4096

theorem stateWellFormed_v_lt_q
  {st : MachineState}
  (h : StateWellFormed st)
  (idx : Nat) :
  st.v idx < Goldilocks.q := by
  exact lt_trans (h.2.2.1 idx) goldilocks_q_gt_256

theorem fetchDecodeBound_wellFormed
  {Addr : Type*}
  {rom : Program}
  {pc : Nat}
  {dec : DecodedStep Addr}
  (h : FetchDecodeBound rom pc dec) :
  dec.x < 16 ∧ dec.y < 16 ∧ dec.kk < 256 ∧ dec.nnn < 4096 := by
  simpa [FetchDecodeBound, DecodedCore.WellFormed]
    using FetchDecodeBinding.fetchDecodeBound_wellFormed (dec := dec.toDecodedCore) h

theorem fetchDecodeBound_nnnWord_lt
  {Addr : Type*}
  {rom : Program}
  {pc : Nat}
  {dec : DecodedStep Addr}
  (h : FetchDecodeBound rom pc dec) :
  dec.nnnWord < 2048 := by
  rcases FetchDecodeBinding.fetchDecodeBound_decodes (dec := dec.toDecodedCore) h with ⟨opcode, hDecode⟩
  simpa using FetchDecodeBinding.decodeOpcodeWord_nnnWord_lt hDecode

theorem byteAdd_lt_256
  (a b : Nat) :
  byteAdd a b < 256 := by
  unfold byteAdd
  exact Nat.mod_lt _ (by decide)

theorem lookupValueOf_lt_256
  {Addr : Type*}
  {rom : Program}
  {pre : MachineState}
  {dec : DecodedStep Addr}
  (hPre : StateWellFormed pre)
  (hFetch : FetchDecodeBound rom pre.pc dec) :
  lookupValueOf pre dec < 256 := by
  rcases fetchDecodeBound_wellFormed hFetch with ⟨_, _, hkk, _⟩
  cases hOpcode : dec.opcodeId <;> simp [lookupValueOf, hOpcode, byteAdd_lt_256, hkk, hPre.2.2.1]
  · by_cases hEq : pre.v dec.x = dec.kk
    · simp [ExecutionSemantics.skipEqBit, hEq]
    · simp [ExecutionSemantics.skipEqBit, hEq]

theorem lookupValueOf_lt_q
  {Addr : Type*}
  {rom : Program}
  {pre : MachineState}
  {dec : DecodedStep Addr}
  (hPre : StateWellFormed pre)
  (hFetch : FetchDecodeBound rom pre.pc dec) :
  lookupValueOf pre dec < Goldilocks.q := by
  exact lt_trans (lookupValueOf_lt_256 hPre hFetch) goldilocks_q_gt_256

private theorem pcSucc_lt_q
  {pre : MachineState}
  (hPre : StateWellFormed pre) :
  pre.pc + 1 < Goldilocks.q := by
  have hPc : pre.pc < 4096 := hPre.1
  have hLt : pre.pc + 1 < 4097 := by
    omega
  have hQ : 4097 < Goldilocks.q := by decide
  exact lt_trans hLt hQ

private theorem pcBranch_lt_q
  {pre : MachineState}
  {n : Nat}
  (hPre : StateWellFormed pre)
  (hn : n ≤ 1) :
  pre.pc + 1 + n < Goldilocks.q := by
  have hPc : pre.pc < 4096 := hPre.1
  have hLt : pre.pc + 1 + n < 4098 := by
    omega
  exact lt_trans hLt goldilocks_q_gt_4098

private theorem pcNat_of_fieldSuccEq
  {pre post : MachineState}
  (hPre : StateWellFormed pre)
  (hPost : StateWellFormed post)
  (h : (post.pc : F) = (pre.pc : F) + 1) :
  post.pc = pre.pc + 1 := by
  have h' : (post.pc : F) = ((pre.pc + 1 : Nat) : F) := by
    simpa [Nat.cast_add] using h
  exact natEq_of_fieldEq (stateWellFormed_pc_lt_q hPost) (pcSucc_lt_q hPre) h'

theorem microstepCorrect_of_bounds
  {Addr : Type*}
  {rom : Program}
  {σ : ExternalSchedule}
  {stepIdx : Nat}
  {pre post : MachineState}
  {init : InitialState}
  {dec : DecodedStep Addr}
  {z : Nightstream.Chip8.Witness F}
  (hPre : StateWellFormed pre)
  (hPost : StateWellFormed post)
  (hWf : Nightstream.Chip8.wf z)
  (hWitness : WitnessMemoryBinding.WitnessBinds (K := F) pre post dec z)
  (hFetch : FetchDecodeBound rom pre.pc dec)
  (hLookup : LookupBound dec pre z)
  (hMem : MemoryBound pre post init dec z)
  (hFramebuffer : FramebufferBound pre post dec)
  (hSchedule : ScheduleBound σ stepIdx pre post dec)
  (hRouting : Nightstream.Chip8.chip8RoutingSound z) :
  MicrostepCorrect rom σ dec pre post := by
  let _ := hFramebuffer
  let _ := hSchedule
  have hLocal : WitnessMemoryBinding.MemoryBound (K := F) pre post init dec z := hMem.1
  rcases fetchDecodeBound_wellFormed hFetch with ⟨_, _, hkk, hnnn⟩
  cases hOpcode : dec.opcodeId
  · have hz13 : z 13 = (1 : F) := by
      simpa [flagWritesLookupToX, FetchDecodeBinding.behaviorOfOpcode,
        Nightstream.Chip8.behaviorFlags] using
        witness_lookupFlag_of_opcode hWitness hOpcode
    have hz16 : z 16 = (0 : F) := by
      simpa [flagWritesNnnToI, FetchDecodeBinding.behaviorOfOpcode,
        Nightstream.Chip8.behaviorFlags] using
        witness_writeIFlag_of_opcode hWitness hOpcode
    have hz17 : z 17 = (0 : F) := by
      simpa [flagIsJump, FetchDecodeBinding.behaviorOfOpcode,
        Nightstream.Chip8.behaviorFlags] using
        witness_jumpFlag_of_opcode hWitness hOpcode
    have hz18 : z 18 = (0 : F) := by
      simpa [flagIsBranch, FetchDecodeBinding.behaviorOfOpcode,
        Nightstream.Chip8.behaviorFlags] using
        witness_branchFlag_of_opcode hWitness hOpcode
    have hz19 : z 19 = (0 : F) := by
      simpa [flagIsMemOp, FetchDecodeBinding.behaviorOfOpcode,
        Nightstream.Chip8.behaviorFlags] using
        witness_memOpFlag_of_opcode hWitness hOpcode
    have hRest : RegistersPreservedExcept pre post dec.x ∧ RamPreserved pre post := by
      simpa [MemoryBound, hOpcode] using hMem.2
    have hz5 : z 5 = z 12 := routing_lookup_to_vx hRouting hz13
    have hIeq : z 7 = z 6 := routing_preserve_i hRouting hz16
    have hPceq : z 2 = z 1 + 1 := routing_default_pc hRouting ⟨hWf, hz17, hz18, hz19⟩
    have hPreVx :
        WitnessMemoryBinding.primaryValue pre dec = pre.v dec.x := by
      exact primaryValue_eq_vx_of_nonBurst (by simp [hOpcode]) (by simp [hOpcode])
    have hPostVx :
        WitnessMemoryBinding.primaryValue post dec = post.v dec.x := by
      exact primaryValue_eq_vx_of_nonBurst (by simp [hOpcode]) (by simp [hOpcode])
    have hVxField : (post.v dec.x : F) = (dec.kk : F) := by
      calc
        (post.v dec.x : F) = (WitnessMemoryBinding.primaryValue post dec : F) := by
          simp [hPostVx]
        _ = z 5 := by symm; exact witnessBinds_vxNext hWitness
        _ = z 12 := hz5
        _ = (lookupValueOf pre dec : F) := hLookup
        _ = (dec.kk : F) := by simp [lookupValueOf, hOpcode]
    have hPcField : (post.pc : F) = (pre.pc : F) + 1 := by
      calc
        (post.pc : F) = z 2 := by symm; exact witnessBinds_pcNext hWitness
        _ = z 1 + 1 := hPceq
        _ = (pre.pc : F) + 1 := by rw [witnessBinds_pc hWitness]
    have hIField : (post.i : F) = (pre.i : F) := by
      calc
        (post.i : F) = z 7 := by symm; exact witnessBinds_iNext hWitness
        _ = z 6 := hIeq
        _ = (pre.i : F) := memoryBound_iReg hLocal
    have hPcNat := pcNat_of_fieldSuccEq hPre hPost hPcField
    have hINat := natEq_of_fieldEq (stateWellFormed_i_lt_q hPost) (stateWellFormed_i_lt_q hPre) hIField
    have hVxNat := natEq_of_fieldEq (stateWellFormed_v_lt_q hPost dec.x)
      (lt_trans hkk goldilocks_q_gt_256) hVxField
    show ExecutionSemantics.MicrostepCorrect rom σ dec pre post
    simpa [ExecutionSemantics.MicrostepCorrect, hOpcode] using
      ⟨hPcNat, hINat, hVxNat, hRest.1, hRest.2⟩
  · have hz13 : z 13 = (1 : F) := by
      simpa [flagWritesLookupToX, FetchDecodeBinding.behaviorOfOpcode,
        Nightstream.Chip8.behaviorFlags] using
        witness_lookupFlag_of_opcode hWitness hOpcode
    have hz16 : z 16 = (0 : F) := by
      simpa [flagWritesNnnToI, FetchDecodeBinding.behaviorOfOpcode,
        Nightstream.Chip8.behaviorFlags] using
        witness_writeIFlag_of_opcode hWitness hOpcode
    have hz17 : z 17 = (0 : F) := by
      simpa [flagIsJump, FetchDecodeBinding.behaviorOfOpcode,
        Nightstream.Chip8.behaviorFlags] using
        witness_jumpFlag_of_opcode hWitness hOpcode
    have hz18 : z 18 = (0 : F) := by
      simpa [flagIsBranch, FetchDecodeBinding.behaviorOfOpcode,
        Nightstream.Chip8.behaviorFlags] using
        witness_branchFlag_of_opcode hWitness hOpcode
    have hz19 : z 19 = (0 : F) := by
      simpa [flagIsMemOp, FetchDecodeBinding.behaviorOfOpcode,
        Nightstream.Chip8.behaviorFlags] using
        witness_memOpFlag_of_opcode hWitness hOpcode
    have hRest : RegistersPreservedExcept pre post dec.x ∧ RamPreserved pre post := by
      simpa [MemoryBound, hOpcode] using hMem.2
    have hz5 : z 5 = z 12 := routing_lookup_to_vx hRouting hz13
    have hIeq : z 7 = z 6 := routing_preserve_i hRouting hz16
    have hPceq : z 2 = z 1 + 1 := routing_default_pc hRouting ⟨hWf, hz17, hz18, hz19⟩
    have hPostVx :
        WitnessMemoryBinding.primaryValue post dec = post.v dec.x := by
      exact primaryValue_eq_vx_of_nonBurst (by simp [hOpcode]) (by simp [hOpcode])
    have hVxField : (post.v dec.x : F) = (byteAdd (pre.v dec.x) dec.kk : F) := by
      calc
        (post.v dec.x : F) = (WitnessMemoryBinding.primaryValue post dec : F) := by
          simp [hPostVx]
        _ = z 5 := by symm; exact witnessBinds_vxNext hWitness
        _ = z 12 := hz5
        _ = (lookupValueOf pre dec : F) := hLookup
        _ = (byteAdd (pre.v dec.x) dec.kk : F) := by simp [lookupValueOf, hOpcode]
    have hPcField : (post.pc : F) = (pre.pc : F) + 1 := by
      calc
        (post.pc : F) = z 2 := by symm; exact witnessBinds_pcNext hWitness
        _ = z 1 + 1 := hPceq
        _ = (pre.pc : F) + 1 := by rw [witnessBinds_pc hWitness]
    have hIField : (post.i : F) = (pre.i : F) := by
      calc
        (post.i : F) = z 7 := by symm; exact witnessBinds_iNext hWitness
        _ = z 6 := hIeq
        _ = (pre.i : F) := memoryBound_iReg hLocal
    have hPcNat := pcNat_of_fieldSuccEq hPre hPost hPcField
    have hINat := natEq_of_fieldEq (stateWellFormed_i_lt_q hPost) (stateWellFormed_i_lt_q hPre) hIField
    have hVxNat := natEq_of_fieldEq (stateWellFormed_v_lt_q hPost dec.x)
      (lt_trans (byteAdd_lt_256 _ _) goldilocks_q_gt_256) hVxField
    show ExecutionSemantics.MicrostepCorrect rom σ dec pre post
    simpa [ExecutionSemantics.MicrostepCorrect, hOpcode] using
      ⟨hPcNat, hINat, hVxNat, hRest.1, hRest.2⟩
  · have hz13 : z 13 = (1 : F) := by
      simpa [flagWritesLookupToX, FetchDecodeBinding.behaviorOfOpcode,
        Nightstream.Chip8.behaviorFlags] using
        witness_lookupFlag_of_opcode hWitness hOpcode
    have hz16 : z 16 = (0 : F) := by
      simpa [flagWritesNnnToI, FetchDecodeBinding.behaviorOfOpcode,
        Nightstream.Chip8.behaviorFlags] using
        witness_writeIFlag_of_opcode hWitness hOpcode
    have hz17 : z 17 = (0 : F) := by
      simpa [flagIsJump, FetchDecodeBinding.behaviorOfOpcode,
        Nightstream.Chip8.behaviorFlags] using
        witness_jumpFlag_of_opcode hWitness hOpcode
    have hz18 : z 18 = (0 : F) := by
      simpa [flagIsBranch, FetchDecodeBinding.behaviorOfOpcode,
        Nightstream.Chip8.behaviorFlags] using
        witness_branchFlag_of_opcode hWitness hOpcode
    have hz19 : z 19 = (0 : F) := by
      simpa [flagIsMemOp, FetchDecodeBinding.behaviorOfOpcode,
        Nightstream.Chip8.behaviorFlags] using
        witness_memOpFlag_of_opcode hWitness hOpcode
    have hRest : RegistersPreservedExcept pre post dec.x ∧ RamPreserved pre post := by
      simpa [MemoryBound, hOpcode] using hMem.2
    have hz5 : z 5 = z 12 := routing_lookup_to_vx hRouting hz13
    have hIeq : z 7 = z 6 := routing_preserve_i hRouting hz16
    have hPceq : z 2 = z 1 + 1 := routing_default_pc hRouting ⟨hWf, hz17, hz18, hz19⟩
    have hPostVx :
        WitnessMemoryBinding.primaryValue post dec = post.v dec.x := by
      exact primaryValue_eq_vx_of_nonBurst (by simp [hOpcode]) (by simp [hOpcode])
    have hVxField : (post.v dec.x : F) = (pre.v dec.y : F) := by
      calc
        (post.v dec.x : F) = (WitnessMemoryBinding.primaryValue post dec : F) := by
          simp [hPostVx]
        _ = z 5 := by symm; exact witnessBinds_vxNext hWitness
        _ = z 12 := hz5
        _ = (lookupValueOf pre dec : F) := hLookup
        _ = (pre.v dec.y : F) := by simp [lookupValueOf, hOpcode]
    have hPcField : (post.pc : F) = (pre.pc : F) + 1 := by
      calc
        (post.pc : F) = z 2 := by symm; exact witnessBinds_pcNext hWitness
        _ = z 1 + 1 := hPceq
        _ = (pre.pc : F) + 1 := by rw [witnessBinds_pc hWitness]
    have hIField : (post.i : F) = (pre.i : F) := by
      calc
        (post.i : F) = z 7 := by symm; exact witnessBinds_iNext hWitness
        _ = z 6 := hIeq
        _ = (pre.i : F) := memoryBound_iReg hLocal
    have hPcNat := pcNat_of_fieldSuccEq hPre hPost hPcField
    have hINat := natEq_of_fieldEq (stateWellFormed_i_lt_q hPost) (stateWellFormed_i_lt_q hPre) hIField
    have hVxNat := natEq_of_fieldEq (stateWellFormed_v_lt_q hPost dec.x)
      (stateWellFormed_v_lt_q hPre dec.y) hVxField
    show ExecutionSemantics.MicrostepCorrect rom σ dec pre post
    simpa [ExecutionSemantics.MicrostepCorrect, hOpcode] using
      ⟨hPcNat, hINat, hVxNat, hRest.1, hRest.2⟩
  · have hz13 : z 13 = (1 : F) := by
      simpa [flagWritesLookupToX, FetchDecodeBinding.behaviorOfOpcode,
        Nightstream.Chip8.behaviorFlags] using
        witness_lookupFlag_of_opcode hWitness hOpcode
    have hz16 : z 16 = (0 : F) := by
      simpa [flagWritesNnnToI, FetchDecodeBinding.behaviorOfOpcode,
        Nightstream.Chip8.behaviorFlags] using
        witness_writeIFlag_of_opcode hWitness hOpcode
    have hz17 : z 17 = (0 : F) := by
      simpa [flagIsJump, FetchDecodeBinding.behaviorOfOpcode,
        Nightstream.Chip8.behaviorFlags] using
        witness_jumpFlag_of_opcode hWitness hOpcode
    have hz18 : z 18 = (0 : F) := by
      simpa [flagIsBranch, FetchDecodeBinding.behaviorOfOpcode,
        Nightstream.Chip8.behaviorFlags] using
        witness_branchFlag_of_opcode hWitness hOpcode
    have hz19 : z 19 = (0 : F) := by
      simpa [flagIsMemOp, FetchDecodeBinding.behaviorOfOpcode,
        Nightstream.Chip8.behaviorFlags] using
        witness_memOpFlag_of_opcode hWitness hOpcode
    have hRest : RegistersPreservedExcept pre post dec.x ∧ RamPreserved pre post := by
      simpa [MemoryBound, hOpcode] using hMem.2
    have hz5 : z 5 = z 12 := routing_lookup_to_vx hRouting hz13
    have hIeq : z 7 = z 6 := routing_preserve_i hRouting hz16
    have hPceq : z 2 = z 1 + 1 := routing_default_pc hRouting ⟨hWf, hz17, hz18, hz19⟩
    have hPostVx :
        WitnessMemoryBinding.primaryValue post dec = post.v dec.x := by
      exact primaryValue_eq_vx_of_nonBurst (by simp [hOpcode]) (by simp [hOpcode])
    have hVxField : (post.v dec.x : F) = (byteAdd (pre.v dec.x) (pre.v dec.y) : F) := by
      calc
        (post.v dec.x : F) = (WitnessMemoryBinding.primaryValue post dec : F) := by
          simp [hPostVx]
        _ = z 5 := by symm; exact witnessBinds_vxNext hWitness
        _ = z 12 := hz5
        _ = (lookupValueOf pre dec : F) := hLookup
        _ = (byteAdd (pre.v dec.x) (pre.v dec.y) : F) := by simp [lookupValueOf, hOpcode]
    have hPcField : (post.pc : F) = (pre.pc : F) + 1 := by
      calc
        (post.pc : F) = z 2 := by symm; exact witnessBinds_pcNext hWitness
        _ = z 1 + 1 := hPceq
        _ = (pre.pc : F) + 1 := by rw [witnessBinds_pc hWitness]
    have hIField : (post.i : F) = (pre.i : F) := by
      calc
        (post.i : F) = z 7 := by symm; exact witnessBinds_iNext hWitness
        _ = z 6 := hIeq
        _ = (pre.i : F) := memoryBound_iReg hLocal
    have hPcNat := pcNat_of_fieldSuccEq hPre hPost hPcField
    have hINat := natEq_of_fieldEq (stateWellFormed_i_lt_q hPost) (stateWellFormed_i_lt_q hPre) hIField
    have hVxNat := natEq_of_fieldEq (stateWellFormed_v_lt_q hPost dec.x)
      (lt_trans (byteAdd_lt_256 _ _) goldilocks_q_gt_256) hVxField
    show ExecutionSemantics.MicrostepCorrect rom σ dec pre post
    simpa [ExecutionSemantics.MicrostepCorrect, hOpcode] using
      ⟨hPcNat, hINat, hVxNat, hRest.1, hRest.2⟩
  · have hz16 : z 16 = (0 : F) := by
      simpa [flagWritesNnnToI, FetchDecodeBinding.behaviorOfOpcode,
        Nightstream.Chip8.behaviorFlags] using
        witness_writeIFlag_of_opcode hWitness hOpcode
    have hz18 : z 18 = (1 : F) := by
      simpa [flagIsBranch, FetchDecodeBinding.behaviorOfOpcode,
        Nightstream.Chip8.behaviorFlags] using
        witness_branchFlag_of_opcode hWitness hOpcode
    have hRest : RegistersPreserved pre post ∧ RamPreserved pre post := by
      simpa [MemoryBound, hOpcode] using hMem.2
    have hIeq : z 7 = z 6 := routing_preserve_i hRouting hz16
    have hPcEq : z colPcNext = z colPc + 1 + z colLookupOutput := routing_branch_pc hRouting ⟨hWf, hz18⟩
    have hPcField : (post.pc : F) = ((pre.pc + 1 + skipEqBit (pre.v dec.x) dec.kk : Nat) : F) := by
      calc
        (post.pc : F) = z 2 := by symm; exact witnessBinds_pcNext hWitness
        _ = z 1 + 1 + z 12 := by simpa using hPcEq
        _ = (pre.pc : F) + 1 + (lookupValueOf pre dec : F) := by
          rw [witnessBinds_pc hWitness]
          simpa [LookupBound] using hLookup
        _ = ((pre.pc + 1 + skipEqBit (pre.v dec.x) dec.kk : Nat) : F) := by
          simp [lookupValueOf, hOpcode, add_assoc]
    have hIField : (post.i : F) = (pre.i : F) := by
      calc
        (post.i : F) = z 7 := by symm; exact witnessBinds_iNext hWitness
        _ = z 6 := hIeq
        _ = (pre.i : F) := memoryBound_iReg hLocal
    have hLookupLtOne : skipEqBit (pre.v dec.x) dec.kk ≤ 1 := by
      by_cases hEq : pre.v dec.x = dec.kk
      · simp [ExecutionSemantics.skipEqBit, hEq]
      · simp [ExecutionSemantics.skipEqBit, hEq]
    have hPcNat := natEq_of_fieldEq (stateWellFormed_pc_lt_q hPost) (pcBranch_lt_q hPre hLookupLtOne) hPcField
    have hINat := natEq_of_fieldEq (stateWellFormed_i_lt_q hPost) (stateWellFormed_i_lt_q hPre) hIField
    show ExecutionSemantics.MicrostepCorrect rom σ dec pre post
    simpa [ExecutionSemantics.MicrostepCorrect, hOpcode] using
      ⟨hPcNat, hINat, hRest.1, hRest.2⟩
  · have hz16 : z 16 = (0 : F) := by
      simpa [flagWritesNnnToI, FetchDecodeBinding.behaviorOfOpcode,
        Nightstream.Chip8.behaviorFlags] using
        witness_writeIFlag_of_opcode hWitness hOpcode
    have hz17 : z 17 = (1 : F) := by
      simpa [flagIsJump, FetchDecodeBinding.behaviorOfOpcode,
        Nightstream.Chip8.behaviorFlags] using
        witness_jumpFlag_of_opcode hWitness hOpcode
    have hRest : RegistersPreserved pre post ∧ RamPreserved pre post := by
      simpa [MemoryBound, hOpcode] using hMem.2
    have hIeq : z 7 = z 6 := routing_preserve_i hRouting hz16
    have hPcEq : z 2 = z 10 := routing_jump_pc hRouting hz17
    have hWord := fetchDecodeBound_nnnWord_lt hFetch
    have hPcField : (post.pc : F) = (dec.nnnWord : F) := by
      calc
        (post.pc : F) = z 2 := by symm; exact witnessBinds_pcNext hWitness
        _ = z 10 := hPcEq
        _ = (dec.nnnWord : F) := witnessBinds_nnnWord hWitness
    have hIField : (post.i : F) = (pre.i : F) := by
      calc
        (post.i : F) = z 7 := by symm; exact witnessBinds_iNext hWitness
        _ = z 6 := hIeq
        _ = (pre.i : F) := memoryBound_iReg hLocal
    have hPcNat := natEq_of_fieldEq (stateWellFormed_pc_lt_q hPost) (lt_trans hWord goldilocks_q_gt_2048) hPcField
    have hINat := natEq_of_fieldEq (stateWellFormed_i_lt_q hPost) (stateWellFormed_i_lt_q hPre) hIField
    show ExecutionSemantics.MicrostepCorrect rom σ dec pre post
    simpa [ExecutionSemantics.MicrostepCorrect, hOpcode] using
      ⟨hPcNat, hINat, hRest.1, hRest.2⟩
  · have hz16 : z 16 = (1 : F) := by
      simpa [flagWritesNnnToI, FetchDecodeBinding.behaviorOfOpcode,
        Nightstream.Chip8.behaviorFlags] using
        witness_writeIFlag_of_opcode hWitness hOpcode
    have hz17 : z 17 = (0 : F) := by
      simpa [flagIsJump, FetchDecodeBinding.behaviorOfOpcode,
        Nightstream.Chip8.behaviorFlags] using
        witness_jumpFlag_of_opcode hWitness hOpcode
    have hz18 : z 18 = (0 : F) := by
      simpa [flagIsBranch, FetchDecodeBinding.behaviorOfOpcode,
        Nightstream.Chip8.behaviorFlags] using
        witness_branchFlag_of_opcode hWitness hOpcode
    have hz19 : z 19 = (0 : F) := by
      simpa [flagIsMemOp, FetchDecodeBinding.behaviorOfOpcode,
        Nightstream.Chip8.behaviorFlags] using
        witness_memOpFlag_of_opcode hWitness hOpcode
    have hRest : RegistersPreserved pre post ∧ RamPreserved pre post := by
      simpa [MemoryBound, hOpcode] using hMem.2
    have hPcEq : z 2 = z 1 + 1 := routing_default_pc hRouting ⟨hWf, hz17, hz18, hz19⟩
    have hIEq : z 7 = z 9 := routing_write_i hRouting hz16
    have hPcField : (post.pc : F) = (pre.pc : F) + 1 := by
      calc
        (post.pc : F) = z 2 := by symm; exact witnessBinds_pcNext hWitness
        _ = z 1 + 1 := hPcEq
        _ = (pre.pc : F) + 1 := by rw [witnessBinds_pc hWitness]
    have hIField : (post.i : F) = (dec.nnnAddr : F) := by
      calc
        (post.i : F) = z 7 := by symm; exact witnessBinds_iNext hWitness
        _ = z 9 := hIEq
        _ = (dec.nnnAddr : F) := witnessBinds_nnnAddr hWitness
    have hPcNat := pcNat_of_fieldSuccEq hPre hPost hPcField
    have hINat := natEq_of_fieldEq (stateWellFormed_i_lt_q hPost) (lt_trans hnnn goldilocks_q_gt_4096) hIField
    show ExecutionSemantics.MicrostepCorrect rom σ dec pre post
    simpa [ExecutionSemantics.MicrostepCorrect, hOpcode] using
      ⟨hPcNat, hINat, hRest.1, hRest.2⟩
  · have hz16 : z 16 = (0 : F) := by
      simpa [flagWritesNnnToI, FetchDecodeBinding.behaviorOfOpcode,
        Nightstream.Chip8.behaviorFlags] using
        witness_writeIFlag_of_opcode hWitness hOpcode
    have hz17 : z 17 = (0 : F) := by
      simpa [flagIsJump, FetchDecodeBinding.behaviorOfOpcode,
        Nightstream.Chip8.behaviorFlags] using
        witness_jumpFlag_of_opcode hWitness hOpcode
    have hz18 : z 18 = (0 : F) := by
      simpa [flagIsBranch, FetchDecodeBinding.behaviorOfOpcode,
        Nightstream.Chip8.behaviorFlags] using
        witness_branchFlag_of_opcode hWitness hOpcode
    have hz19 : z 19 = (1 : F) := by
      simpa [flagIsMemOp, FetchDecodeBinding.behaviorOfOpcode,
        Nightstream.Chip8.behaviorFlags] using
        witness_memOpFlag_of_opcode hWitness hOpcode
    have hRest :
        RegistersPreserved pre post ∧
          RamPrefixStored pre post dec ∧
          RamPreservedOutsidePrefix pre post dec := by
      simpa [MemoryBound, hOpcode] using hMem.2
    have hIeq : z 7 = z 6 := routing_preserve_i hRouting hz16
    have hPcEq : z 2 = z 1 + z 22 := routing_mem_pc hRouting hz19
    have hPcField : (post.pc : F) = ((pre.pc + burstLastValue dec : Nat) : F) := by
      calc
        (post.pc : F) = z 2 := by symm; exact witnessBinds_pcNext hWitness
        _ = z 1 + z 22 := hPcEq
        _ = (pre.pc : F) + (burstLastValue dec : F) := by
          rw [witnessBinds_pc hWitness, witnessBinds_burstLast hWitness]
        _ = ((pre.pc + burstLastValue dec : Nat) : F) := by norm_num
    have hIField : (post.i : F) = (pre.i : F) := by
      calc
        (post.i : F) = z 7 := by symm; exact witnessBinds_iNext hWitness
        _ = z 6 := hIeq
        _ = (pre.i : F) := memoryBound_iReg hLocal
    have hPcNat := natEq_of_fieldEq (stateWellFormed_pc_lt_q hPost)
      (stateWellFormed_pcBurst_lt_q hPre dec) hPcField
    have hINat := natEq_of_fieldEq (stateWellFormed_i_lt_q hPost) (stateWellFormed_i_lt_q hPre) hIField
    show ExecutionSemantics.MicrostepCorrect rom σ dec pre post
    simpa [ExecutionSemantics.MicrostepCorrect, hOpcode] using
      ⟨hPcNat, hINat, hRest.1, hRest.2.1, hRest.2.2⟩
  · have hz14 : z 14 = (1 : F) := by
      simpa [flagWritesMemToX, FetchDecodeBinding.behaviorOfOpcode,
        Nightstream.Chip8.behaviorFlags] using
        witness_memFlag_of_opcode hWitness hOpcode
    have hz16 : z 16 = (0 : F) := by
      simpa [flagWritesNnnToI, FetchDecodeBinding.behaviorOfOpcode,
        Nightstream.Chip8.behaviorFlags] using
        witness_writeIFlag_of_opcode hWitness hOpcode
    have hz17 : z 17 = (0 : F) := by
      simpa [flagIsJump, FetchDecodeBinding.behaviorOfOpcode,
        Nightstream.Chip8.behaviorFlags] using
        witness_jumpFlag_of_opcode hWitness hOpcode
    have hz18 : z 18 = (0 : F) := by
      simpa [flagIsBranch, FetchDecodeBinding.behaviorOfOpcode,
        Nightstream.Chip8.behaviorFlags] using
        witness_branchFlag_of_opcode hWitness hOpcode
    have hz19 : z 19 = (1 : F) := by
      simpa [flagIsMemOp, FetchDecodeBinding.behaviorOfOpcode,
        Nightstream.Chip8.behaviorFlags] using
        witness_memOpFlag_of_opcode hWitness hOpcode
    have hRest :
        RegistersLoadedPrefix pre post dec ∧
          RegistersPreservedAbove pre post dec.x ∧
          RamPreserved pre post := by
      simpa [MemoryBound, hOpcode] using hMem.2
    have hz5 : z 5 = z 11 := routing_mem_to_vx hRouting hz14
    have hIeq : z 7 = z 6 := routing_preserve_i hRouting hz16
    have hPcEq : z 2 = z 1 + z 22 := routing_mem_pc hRouting hz19
    have hPcField : (post.pc : F) = ((pre.pc + burstLastValue dec : Nat) : F) := by
      calc
        (post.pc : F) = z 2 := by symm; exact witnessBinds_pcNext hWitness
        _ = z 1 + z 22 := hPcEq
        _ = (pre.pc : F) + (burstLastValue dec : F) := by
          rw [witnessBinds_pc hWitness, witnessBinds_burstLast hWitness]
        _ = ((pre.pc + burstLastValue dec : Nat) : F) := by norm_num
    have hIField : (post.i : F) = (pre.i : F) := by
      calc
        (post.i : F) = z 7 := by symm; exact witnessBinds_iNext hWitness
        _ = z 6 := hIeq
        _ = (pre.i : F) := memoryBound_iReg hLocal
    have hPcNat := natEq_of_fieldEq (stateWellFormed_pc_lt_q hPost)
      (stateWellFormed_pcBurst_lt_q hPre dec) hPcField
    have hINat := natEq_of_fieldEq (stateWellFormed_i_lt_q hPost) (stateWellFormed_i_lt_q hPre) hIField
    show ExecutionSemantics.MicrostepCorrect rom σ dec pre post
    simpa [ExecutionSemantics.MicrostepCorrect, hOpcode] using
      ⟨hPcNat, hINat, hRest.1, hRest.2.1, hRest.2.2⟩

abbrev instructionCorrect_of_nonBurstMicrostep :=
  @ExecutionSemantics.instructionCorrect_of_nonBurstMicrostep

theorem microstepCorrect_ldImm
  {Addr : Type*}
  {rom : Program}
  {σ : ExternalSchedule}
  {dec : DecodedStep Addr}
  {pre post : MachineState}
  (hOpcode : dec.opcodeId = .ldImm)
  (hMicro : MicrostepCorrect rom σ dec pre post) :
  post.pc = pre.pc + 1 ∧
    post.i = pre.i ∧
    post.v dec.x = dec.kk ∧
    RegistersPreservedExcept pre post dec.x ∧
    RamPreserved pre post := by
  unfold MicrostepCorrect at hMicro
  simpa [ExecutionSemantics.MicrostepCorrect, hOpcode] using hMicro

theorem microstepCorrect_addImm
  {Addr : Type*}
  {rom : Program}
  {σ : ExternalSchedule}
  {dec : DecodedStep Addr}
  {pre post : MachineState}
  (hOpcode : dec.opcodeId = .addImm)
  (hMicro : MicrostepCorrect rom σ dec pre post) :
  post.pc = pre.pc + 1 ∧
    post.i = pre.i ∧
    post.v dec.x = byteAdd (pre.v dec.x) dec.kk ∧
    RegistersPreservedExcept pre post dec.x ∧
    RamPreserved pre post := by
  unfold MicrostepCorrect at hMicro
  simpa [ExecutionSemantics.MicrostepCorrect, hOpcode] using hMicro

theorem microstepCorrect_mov
  {Addr : Type*}
  {rom : Program}
  {σ : ExternalSchedule}
  {dec : DecodedStep Addr}
  {pre post : MachineState}
  (hOpcode : dec.opcodeId = .mov)
  (hMicro : MicrostepCorrect rom σ dec pre post) :
  post.pc = pre.pc + 1 ∧
    post.i = pre.i ∧
    post.v dec.x = pre.v dec.y ∧
    RegistersPreservedExcept pre post dec.x ∧
    RamPreserved pre post := by
  unfold MicrostepCorrect at hMicro
  simpa [ExecutionSemantics.MicrostepCorrect, hOpcode] using hMicro

theorem microstepCorrect_addReg
  {Addr : Type*}
  {rom : Program}
  {σ : ExternalSchedule}
  {dec : DecodedStep Addr}
  {pre post : MachineState}
  (hOpcode : dec.opcodeId = .addReg)
  (hMicro : MicrostepCorrect rom σ dec pre post) :
  post.pc = pre.pc + 1 ∧
    post.i = pre.i ∧
    post.v dec.x = byteAdd (pre.v dec.x) (pre.v dec.y) ∧
    RegistersPreservedExcept pre post dec.x ∧
    RamPreserved pre post := by
  unfold MicrostepCorrect at hMicro
  simpa [ExecutionSemantics.MicrostepCorrect, hOpcode] using hMicro

theorem microstepCorrect_skipEqImm
  {Addr : Type*}
  {rom : Program}
  {σ : ExternalSchedule}
  {dec : DecodedStep Addr}
  {pre post : MachineState}
  (hOpcode : dec.opcodeId = .skipEqImm)
  (hMicro : MicrostepCorrect rom σ dec pre post) :
  post.pc = pre.pc + 1 + skipEqBit (pre.v dec.x) dec.kk ∧
    post.i = pre.i ∧
    RegistersPreserved pre post ∧
    RamPreserved pre post := by
  unfold MicrostepCorrect at hMicro
  simpa [ExecutionSemantics.MicrostepCorrect, hOpcode] using hMicro

theorem microstepCorrect_jump
  {Addr : Type*}
  {rom : Program}
  {σ : ExternalSchedule}
  {dec : DecodedStep Addr}
  {pre post : MachineState}
  (hOpcode : dec.opcodeId = .jump)
  (hMicro : MicrostepCorrect rom σ dec pre post) :
  post.pc = dec.nnnWord ∧
    post.i = pre.i ∧
    RegistersPreserved pre post ∧
    RamPreserved pre post := by
  unfold MicrostepCorrect at hMicro
  simpa [ExecutionSemantics.MicrostepCorrect, hOpcode] using hMicro

theorem microstepCorrect_ldI
  {Addr : Type*}
  {rom : Program}
  {σ : ExternalSchedule}
  {dec : DecodedStep Addr}
  {pre post : MachineState}
  (hOpcode : dec.opcodeId = .ldI)
  (hMicro : MicrostepCorrect rom σ dec pre post) :
  post.pc = pre.pc + 1 ∧
    post.i = dec.nnnAddr ∧
    RegistersPreserved pre post ∧
    RamPreserved pre post := by
  unfold MicrostepCorrect at hMicro
  simpa [ExecutionSemantics.MicrostepCorrect, hOpcode] using hMicro

theorem microstepCorrect_storeRegs
  {Addr : Type*}
  {rom : Program}
  {σ : ExternalSchedule}
  {dec : DecodedStep Addr}
  {pre post : MachineState}
  (hOpcode : dec.opcodeId = .storeRegs)
  (hMicro : MicrostepCorrect rom σ dec pre post) :
  post.pc = pre.pc + burstLastValue dec ∧
    post.i = pre.i ∧
    RegistersPreserved pre post ∧
    RamPrefixStored pre post dec ∧
    RamPreservedOutsidePrefix pre post dec := by
  unfold MicrostepCorrect at hMicro
  simpa [ExecutionSemantics.MicrostepCorrect, hOpcode] using hMicro

theorem microstepCorrect_loadRegs
  {Addr : Type*}
  {rom : Program}
  {σ : ExternalSchedule}
  {dec : DecodedStep Addr}
  {pre post : MachineState}
  (hOpcode : dec.opcodeId = .loadRegs)
  (hMicro : MicrostepCorrect rom σ dec pre post) :
  post.pc = pre.pc + burstLastValue dec ∧
    post.i = pre.i ∧
    RegistersLoadedPrefix pre post dec ∧
    RegistersPreservedAbove pre post dec.x ∧
    RamPreserved pre post := by
  unfold MicrostepCorrect at hMicro
  simpa [ExecutionSemantics.MicrostepCorrect, hOpcode] using hMicro

theorem instructionCorrect_of_burst
  {Addr : Type*}
  {rom : Program}
  {σ : ExternalSchedule}
  {dec : DecodedStep Addr}
  {pre post : MachineState}
  {frames : List (ExecutionFrame Addr)}
  (hBurst : BurstScheduleCorrect rom σ dec pre post frames) :
  InstructionCorrect rom σ dec pre post := by
  exact BurstSession.instructionCorrect_of_burstSession hBurst

abbrev executionCorrect_of_trace :=
  @ExecutionSemantics.executionCorrect_of_trace

abbrev preparedStepTraceBound_of_execution :=
  @ExecutionSemantics.preparedStepTraceBound_of_execution

abbrev goalPredicate_of_execution :=
  @ExecutionSemantics.goalPredicate_of_execution

end Nightstream.Chip8.StepComposition
