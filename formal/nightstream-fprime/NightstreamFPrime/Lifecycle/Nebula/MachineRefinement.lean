import NightstreamFPrime.Lifecycle.Nebula.StepRefinement

/-! Owns Ob7 for the first memory application: when the memory rows and the
machine's port rows hold, the step's decoded ports are one machine step from
the decoded input state to the decoded output state. It does not own the
memory rows, the carry, or the circuit. -/

namespace NightstreamFPrime.Lifecycle.Nebula

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Nebula
open Fin.CommRing

/-- The machine state `(pc, acc)` of two application words. -/
def Machine.State.ofWords (words : Fin 2 → F) : Machine.State :=
  ⟨(words 0).val, (words 1).val⟩

theorem val_add_one {x : F} (small : x.val + 1 < goldilocksModulus) : (x + 1).val = x.val + 1 := by
  rw [← natWord_of_val x, ← natWord_one, ← natWord_add, natWord_val small, natWord_of_val]

theorem toBool_zero : toBool 0 = false := by simp [toBool, StepWitness.modulus_ne_one]

theorem toBool_one : toBool 1 = true := by simp [toBool]

theorem Plan.two_pow_mu_lt {p : Plan} (valid : p.Valid) : 2 ^ p.μ < goldilocksModulus := by
  have below := valid.belowModulus
  have ram : 2 ^ p.μ ≤ p.cells := Nat.le_add_left _ _
  omega

theorem Plan.mu_le {p : Plan} (valid : p.Valid) : p.μ ≤ 63 := by
  have below := Plan.two_pow_mu_lt valid
  by_contra large
  have : 2 ^ 64 ≤ 2 ^ p.μ := Nat.pow_le_pow_right (by norm_num) (by omega)
  have : (2 : ℕ) ^ 64 > goldilocksModulus := by decide
  omega

/-- The low two bits of a 32-bit word and the rest. -/
theorem bitsNat_split (vr : Fin 32 → F) :
    bitsNat vr = (toBool (vr 0)).toNat + 2 * (toBool (vr 1)).toNat +
      4 * chunkValue (((List.ofFn vr).drop 2).map toBool) := by
  rw [bitsNat, List.ofFn_succ, List.ofFn_succ]
  simp only [List.map_cons, List.drop_succ_cons, List.drop_zero, chunkValue, Fin.succ_zero_eq_one]
  ring

namespace OpSlotBits

variable {p : Plan} {slot : OpSlotBits p}

theorem port_pad (pad : slot.pad = 1) : slot.decode.port = none := by
  simp [decode, OpSlot.port, pad, toBool_one]

theorem port_active (active : slot.pad = 0) : slot.decode.port =
    some ⟨toBool slot.isWrite, toBool slot.isRam, bitsNat slot.addr, bitsNat slot.vr,
      bitsNat slot.vw⟩ := by
  simp [decode, OpSlot.port, active, toBool_zero]

end OpSlotBits

namespace StepWitness

variable {p : Plan} {w : StepWitness p} {zIn : List F} (two : p.bOps = 2)

theorem records_ops_two :
    w.records.ops = [(w.ops (fetchSlot two)).decode, (w.ops (dataSlot two)).decode] := by
  apply List.ext_getElem (by simp [records, two])
  intro k h₁ h₂
  simp only [records, List.getElem_ofFn]
  rcases k with _ | _ | k
  · rfl
  · rfl
  · simp at h₂

/-- The instruction argument is the fetched word divided by four. -/
theorem RowsHold.argument_val (rows : w.RowsHold zIn) :
    (w.argument two).val = (w.ops (fetchSlot two)).decode.vr / 4 := by
  have vrBits := (rows.slotBits w (fetchSlot two)).2.2.2.2.1
  have all : ∀ x ∈ (List.ofFn (w.ops (fetchSlot two)).vr).drop 2, IsBit x := fun x member => by
    obtain ⟨k, rfl⟩ := List.mem_ofFn.mp (List.mem_of_mem_drop member)
    exact vrBits k
  have small := chunkValue_lt (((List.ofFn (w.ops (fetchSlot two)).vr).drop 2).map toBool)
  simp only [List.length_map, List.length_drop, List.length_ofFn] at small
  have split := bitsNat_split (w.ops (fetchSlot two)).vr
  have low0 := Bool.toNat_le (toBool ((w.ops (fetchSlot two)).vr 0))
  have low1 := Bool.toNat_le (toBool ((w.ops (fetchSlot two)).vr 1))
  rw [argument, chunkWord_eq all, natWord_val (lt_trans small (by decide))]
  show _ = bitsNat (w.ops (fetchSlot two)).vr / 4
  omega

/-- Ob7: the memory rows and the machine rows give one machine step on the
decoded ports. -/
theorem RowsHold.machineStep (valid : p.Valid) (rows : w.RowsHold zIn)
    (machine : w.MachineRows two) :
    Machine.Step (Machine.State.ofWords w.appIn) (w.records.ops.map OpSlot.port)
      (Machine.State.ofWords w.appOut) := by
  rw [records_ops_two two]
  simp only [List.map_cons, List.map_nil]
  obtain ⟨-, -, -, addrF, vrF, vwF, -⟩ := rows.slotBits w (fetchSlot two)
  obtain ⟨-, -, -, addrD, vrD, vwD, -⟩ := rows.slotBits w (dataSlot two)
  have fetchPad : (w.ops (fetchSlot two)).pad = w.idle := machine.fetchPad
  have pcOut := machine.pcOut
  have accOut := machine.accOut
  have dataPad : (w.ops (dataSlot two)).pad = 1 - (w.isLoad two + w.isStore two) := machine.dataPad
  rcases rows.idleBit with active | idle
  · -- A fetched instruction.
    have fetchActive : (w.ops (fetchSlot two)).pad = 0 := fetchPad.trans active
    have fetchRead : (w.ops (fetchSlot two)).isWrite = 0 := machine.fetchRead
    have fetchRom : (w.ops (fetchSlot two)).isRam = 0 := machine.fetchRom
    have fetchAddr := machine.fetchAddr
    rw [active, sub_zero, one_mul] at fetchAddr
    have keeps := rows.readKeeps (fetchSlot two)
    rw [StepWitness.isWrite, fetchRead, sub_zero, one_mul, sub_eq_zero] at keeps
    have pc : (w.appIn 0).val = bitsNat (w.ops (fetchSlot two)).addr := by
      rw [← fetchAddr]
      exact bitsWord_val addrF (Plan.mu_le valid)
    have pcSmall : (w.appIn 0).val + 1 < goldilocksModulus := by
      have := bitsNat_lt (w.ops (fetchSlot two)).addr
      have := Plan.two_pow_mu_lt valid
      omega
    have fetchPort : (w.ops (fetchSlot two)).decode.port =
        some (Machine.fetch (Machine.State.ofWords w.appIn) (w.ops (fetchSlot two)).decode.vr) := by
      rw [OpSlotBits.port_active fetchActive, fetchRead, fetchRom, toBool_zero,
        bitsWord_inj (by norm_num) vwF vrF keeps]
      simp only [Machine.fetch, Machine.State.ofWords, pc]
      rfl
    have argument : (w.argument two).val = bitsNat (w.ops (fetchSlot two)).vr / 4 :=
      rows.argument_val two
    have split := bitsNat_split (w.ops (fetchSlot two)).vr
    have dataValue := machine.dataValue
    have dataAddr := machine.dataAddr
    have dataWrite : (w.ops (dataSlot two)).isWrite = w.isStore two := machine.dataWrite
    have dataRam : (w.ops (dataSlot two)).isRam = w.isLoad two + w.isStore two := machine.dataRam
    have addrVal : (w.addr (dataSlot two)).val = bitsNat (w.ops (dataSlot two)).addr :=
      bitsWord_val addrD (Plan.mu_le valid)
    have vrVal : (w.vr (dataSlot two)).val = bitsNat (w.ops (dataSlot two)).vr :=
      bitsWord_val vrD (by norm_num)
    have vwVal : (w.vw (dataSlot two)).val = bitsNat (w.ops (dataSlot two)).vw :=
      bitsWord_val vwD (by norm_num)
    refine Or.inr ⟨(w.ops (fetchSlot two)).decode.vr, (w.ops (dataSlot two)).decode.vr, ?_⟩
    rw [fetchPort]
    have bit0 := vrF 0
    have bit1 := vrF 1
    rcases bit0 with b0 | b0 <;> rcases bit1 with b1 | b1
    · -- halt
      have mod : bitsNat (w.ops (fetchSlot two)).vr % 4 = 0 := by
        simp only [split, b0, b1, toBool_zero, Bool.toNat_false]
        omega
      have load : w.isLoad two = 0 := by simp [isLoad, instructionBit, b0]
      have store : w.isStore two = 0 := by simp [isStore, instructionBit, b1]
      have loadi : w.isLoadi two = 0 := by simp [isLoadi, instructionBit, b0]
      simp only [load, store, loadi, add_zero, zero_add, sub_zero, zero_mul,
        one_mul] at dataPad pcOut accOut
      refine ⟨?_, ?_⟩
      · rw [OpSlotBits.port_pad dataPad]
        simp [Machine.dataPort, OpSlotBits.decode, mod]
      · simp [Machine.exec, Machine.State.ofWords, OpSlotBits.decode, mod, pcOut, accOut]
    · -- store
      have mod : bitsNat (w.ops (fetchSlot two)).vr % 4 = 2 := by
        simp only [split, b0, b1, toBool_zero, toBool_one, Bool.toNat_false, Bool.toNat_true]
        omega
      have load : w.isLoad two = 0 := by simp [isLoad, instructionBit, b0]
      have store : w.isStore two = 1 := by simp [isStore, instructionBit, b0, b1]
      have loadi : w.isLoadi two = 0 := by simp [isLoadi, instructionBit, b0]
      simp only [load, store, loadi, add_zero, zero_add, sub_zero, sub_self, zero_mul,
        one_mul] at dataPad dataWrite dataRam dataAddr dataValue pcOut accOut
      refine ⟨?_, ?_⟩
      · rw [OpSlotBits.port_active dataPad, dataWrite, dataRam, toBool_one]
        have addr : bitsNat (w.ops (dataSlot two)).addr =
            bitsNat (w.ops (fetchSlot two)).vr / 4 := by
          rw [← addrVal, dataAddr, argument]
        have acc : bitsNat (w.ops (dataSlot two)).vw = (w.appIn 1).val := by
          rw [← vwVal, dataValue]
        simp only [Machine.dataPort, OpSlotBits.decode, mod, addr, acc, Machine.State.ofWords]
      · simp only [Machine.exec, Machine.State.ofWords, OpSlotBits.decode, mod, pcOut, accOut,
          val_add_one pcSmall]
    · -- load
      have mod : bitsNat (w.ops (fetchSlot two)).vr % 4 = 1 := by
        simp only [split, b0, b1, toBool_zero, toBool_one, Bool.toNat_false, Bool.toNat_true]
        omega
      have load : w.isLoad two = 1 := by simp [isLoad, instructionBit, b0, b1]
      have store : w.isStore two = 0 := by simp [isStore, instructionBit, b0, b1]
      have loadi : w.isLoadi two = 0 := by simp [isLoadi, instructionBit, b1]
      simp only [load, store, loadi, add_zero, zero_add, sub_self, zero_mul,
        one_mul] at dataPad dataWrite dataRam dataAddr dataValue pcOut accOut
      refine ⟨?_, ?_⟩
      · rw [OpSlotBits.port_active dataPad, dataWrite, dataRam, toBool_zero, toBool_one]
        have addr : bitsNat (w.ops (dataSlot two)).addr =
            bitsNat (w.ops (fetchSlot two)).vr / 4 := by
          rw [← addrVal, dataAddr, argument]
        have keeps : bitsNat (w.ops (dataSlot two)).vw = bitsNat (w.ops (dataSlot two)).vr := by
          rw [← vwVal, ← vrVal, dataValue]
        simp only [Machine.dataPort, OpSlotBits.decode, mod, addr, keeps]
      · simp only [Machine.exec, Machine.State.ofWords, OpSlotBits.decode, mod, pcOut, accOut,
          val_add_one pcSmall, vrVal]
    · -- loadi
      have mod : bitsNat (w.ops (fetchSlot two)).vr % 4 = 3 := by
        simp only [split, b0, b1, toBool_one, Bool.toNat_true]
        omega
      have load : w.isLoad two = 0 := by simp [isLoad, instructionBit, b1]
      have store : w.isStore two = 0 := by simp [isStore, instructionBit, b0]
      have loadi : w.isLoadi two = 1 := by simp [isLoadi, instructionBit, b0, b1]
      simp only [load, store, loadi, add_zero, zero_add, sub_zero, sub_self, zero_mul,
        one_mul] at dataPad pcOut accOut
      refine ⟨?_, ?_⟩
      · rw [OpSlotBits.port_pad dataPad]
        simp [Machine.dataPort, OpSlotBits.decode, mod]
      · simp only [Machine.exec, Machine.State.ofWords, OpSlotBits.decode, mod, pcOut, accOut,
          val_add_one pcSmall, argument]
  · -- An idle step.
    have tail := rows.padTail w (fetchSlot two) (fetchPad.trans idle)
    have b0 : w.instructionBit two 0 = 0 := tail.2.2.2.1 0
    have b1 : w.instructionBit two 1 = 0 := tail.2.2.2.1 1
    have load : w.isLoad two = 0 := by rw [isLoad, b0, zero_mul]
    have store : w.isStore two = 0 := by rw [isStore, b1, mul_zero]
    have loadi : w.isLoadi two = 0 := by rw [isLoadi, b0, zero_mul]
    simp only [load, store, loadi, add_zero, zero_add, sub_zero, zero_mul,
      one_mul] at dataPad pcOut accOut
    refine Or.inl ⟨?_, ?_⟩
    · rw [OpSlotBits.port_pad (fetchPad.trans idle), OpSlotBits.port_pad dataPad]
    · simp only [Machine.State.ofWords, pcOut, accOut]

end StepWitness

end NightstreamFPrime.Lifecycle.Nebula
