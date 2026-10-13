import NightstreamFPrime.Lifecycle.Nebula.RowMeaning
import NightstreamFPrime.Lifecycle.Nebula.StepRefinement
import NightstreamFPrime.Lifecycle.Nebula.Machine

/-! Owns the first memory application at the circuit level: the machine of
`Machine` as a `PortApp` of a plan with two ports. Its state words are
`(pc, acc)`, its one private word is the idle flag, and its rows tie the fetch
port and the data port to the instruction (spec §10). It owns the machine's
Ob7 (`Machine.sound`) and its half of Ob5 (`Machine.complete`). It does not
own the memory rows. -/

namespace NightstreamFPrime.Lifecycle.Nebula

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Nebula
open NightstreamFPrime.Circuit
open Fin.CommRing

theorem val_add_one {x : F} (small : x.val + 1 < goldilocksModulus) : (x + 1).val = x.val + 1 := by
  rw [← natWord_of_val x, ← natWord_one, ← natWord_add, natWord_val small, natWord_of_val]

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

/-- A bit vector of zeros has the value zero. -/
theorem bitsWord_zeros {n : ℕ} {bits : Fin n → F} (zero : ∀ k, bits k = 0) : bitsWord bits = 0 := by
  have small := StepWitness.bitsNat_lt_of_high_zero (Nat.zero_le n) (fun k _ => zero k)
  rw [bitsWord_eq fun k => Or.inl (zero k)]
  rw [pow_zero, Nat.lt_one_iff] at small
  rw [small]
  rfl

namespace OpSlotBits

variable {p : Plan} {slot : OpSlotBits p}

theorem port_pad (pad : slot.pad = 1) : slot.decode.port = none := by
  simp [decode, OpSlot.port, pad, StepWitness.toBool_one]

theorem port_active (active : slot.pad = 0) : slot.decode.port =
    some ⟨toBool slot.isWrite, toBool slot.isRam, bitsNat slot.addr, bitsNat slot.vr,
      bitsNat slot.vw⟩ := by
  simp [decode, OpSlot.port, active, StepWitness.toBool_zero]

theorem WellFormed.fieldBits (wf : slot.WellFormed) :
    IsBit slot.pad ∧ IsBit slot.isWrite ∧ IsBit slot.isRam ∧ (∀ k, IsBit (slot.addr k)) ∧
      (∀ k, IsBit (slot.vr k)) ∧ (∀ k, IsBit (slot.vw k)) := by
  have all := wf.bits
  simp only [OpSlotBits.lane, List.mem_append, List.mem_cons, List.mem_ofFn] at all
  refine ⟨all _ (by simp), all _ (by simp), all _ (by simp), fun k => all _ ?_, fun k => all _ ?_,
    fun k => all _ ?_⟩ <;> simp

/-- The other lane words of a padding slot are zero (row O7). -/
theorem WellFormed.zeros (wf : slot.WellFormed) (pad : slot.pad = 1) :
    slot.isWrite = 0 ∧ slot.isRam = 0 ∧ (∀ k, slot.addr k = 0) ∧ (∀ k, slot.vr k = 0) ∧
      (∀ k, slot.vw k = 0) := by
  have zero := wf.padTail pad
  simp only [OpSlotBits.lane, List.cons_append, List.tail_cons, List.mem_append, List.mem_cons,
    List.mem_ofFn] at zero
  exact ⟨zero _ (by simp), zero _ (by simp), fun k => zero _ (by simp), fun k => zero _ (by simp),
    fun k => zero _ (by simp)⟩

theorem WellFormed.pad_one (wf : slot.WellFormed) (none : slot.decode.port = none) : slot.pad = 1 := by
  rcases wf.fieldBits.1 with zero | one
  · rw [port_active zero] at none
    cases none
  · exact one

theorem WellFormed.active (wf : slot.WellFormed) {access : PortAccess}
    (some : slot.decode.port = some access) :
    slot.pad = 0 ∧ toBool slot.isWrite = access.isWrite ∧ toBool slot.isRam = access.isRam ∧
      bitsNat slot.addr = access.addr ∧ bitsNat slot.vr = access.vr ∧
      bitsNat slot.vw = access.vw := by
  rcases wf.fieldBits.1 with zero | one
  · rw [port_active zero, Option.some.injEq] at some
    subst some
    exact ⟨zero, rfl, rfl, rfl, rfl, rfl⟩
  · rw [port_pad one] at some
    cases some

end OpSlotBits

namespace Machine

/-- Two state words `(pc, acc)` and one private word, the idle flag. -/
abbrev shape : AppShape := ⟨2, 1⟩

/-- The machine state of two state words. -/
def State.ofWords (words : Fin 2 → F) : State := ⟨(words 0).val, (words 1).val⟩

/-- The two state words of a machine state. -/
def State.words (s : State) : Fin 2 → F := ![natWord s.pc, natWord s.acc]

/-- The words of a decoded state are the word vector. -/
theorem State.words_ofWords (w : Fin 2 → F) : (State.ofWords w).words = w := by
  funext i
  fin_cases i <;> simp [State.words, State.ofWords, natWord_of_val]

variable {p : Plan} (two : p.bOps = 2)

/-- Port 0, the fetch port. -/
def fetchSlot : Fin p.bOps := ⟨0, by omega⟩

/-- Port 1, the data port. -/
def dataSlot : Fin p.bOps := ⟨1, by omega⟩

/-! ### The port rows on the view -/

variable (v : AppView p shape)

/-- The idle flag. -/
def idle : F := v.aux 0

/-- Bit `k` of the fetched instruction word. -/
def instructionBit (k : Fin 32) : F := (v.ops (fetchSlot two)).vr k

def isLoad : F := instructionBit two v 0 * (1 - instructionBit two v 1)
def isStore : F := (1 - instructionBit two v 0) * instructionBit two v 1
def isLoadi : F := instructionBit two v 0 * instructionBit two v 1

/-- The instruction argument `v / 4`: bits 2–31 of the fetched word. -/
def argument : F := chunkWord ((List.ofFn (v.ops (fetchSlot two)).vr).drop 2)

/-- The machine's port rows (spec §10) and its next state. -/
structure PortRows : Prop where
  idleBit : IsBit (idle v)
  fetchPad : (v.ops (fetchSlot two)).pad = idle v
  fetchRead : (v.ops (fetchSlot two)).isWrite = 0
  fetchRom : (v.ops (fetchSlot two)).isRam = 0
  fetchAddr : bitsWord (v.ops (fetchSlot two)).addr = (1 - idle v) * v.appIn 0
  dataPad : (v.ops (dataSlot two)).pad = 1 - (isLoad two v + isStore two v)
  dataWrite : (v.ops (dataSlot two)).isWrite = isStore two v
  dataRam : (v.ops (dataSlot two)).isRam = isLoad two v + isStore two v
  dataAddr : bitsWord (v.ops (dataSlot two)).addr = (isLoad two v + isStore two v) * argument two v
  dataValue : bitsWord (v.ops (dataSlot two)).vw =
    isStore two v * v.appIn 1 + isLoad two v * bitsWord (v.ops (dataSlot two)).vr
  pcOut : v.appOut 0 = v.appIn 0 + isLoad two v + isStore two v + isLoadi two v
  accOut : v.appOut 1 = isLoad two v * bitsWord (v.ops (dataSlot two)).vr +
    isLoadi two v * argument two v + (1 - isLoad two v - isLoadi two v) * v.appIn 1

/-! ### The rows over local variables -/

/-- Local variable `k` of slot `j`. -/
def slotE (p : Plan) (j k : ℕ) : Circuit.Expr :=
  .var (AppView.slotStart shape + j * Words.opWords p + k)

def inE (i : ℕ) : Circuit.Expr := .var i
def outE (i : ℕ) : Circuit.Expr := .var (2 + i)
def idleE : Circuit.Expr := .var 4
def padE (p : Plan) (j : ℕ) : Circuit.Expr := slotE p j 0
def isWriteE (p : Plan) (j : ℕ) : Circuit.Expr := slotE p j 1
def isRamE (p : Plan) (j : ℕ) : Circuit.Expr := slotE p j 2
def addrE (p : Plan) (j : ℕ) : Circuit.Expr := Expr.bits fun k : Fin p.μ => slotE p j (3 + k)
def vrBitE (p : Plan) (j k : ℕ) : Circuit.Expr := slotE p j (3 + p.μ + k)
def vrE (p : Plan) (j : ℕ) : Circuit.Expr := Expr.bits fun k : Fin 32 => vrBitE p j k
def vwE (p : Plan) (j : ℕ) : Circuit.Expr :=
  Expr.bits fun k : Fin 32 => slotE p j (3 + p.μ + 32 + k)
def isLoadE (p : Plan) : Circuit.Expr := vrBitE p 0 0 * (1 - vrBitE p 0 1)
def isStoreE (p : Plan) : Circuit.Expr := (1 - vrBitE p 0 0) * vrBitE p 0 1
def isLoadiE (p : Plan) : Circuit.Expr := vrBitE p 0 0 * vrBitE p 0 1
def argumentE (p : Plan) : Circuit.Expr :=
  Expr.chunk ((List.ofFn fun k : Fin 32 => vrBitE p 0 k).drop 2)

/-- The machine's rows over its local variables. -/
def rows (p : Plan) : List Circuit.Expr :=
  [bitRow idleE,
    padE p 0 - idleE,
    isWriteE p 0,
    isRamE p 0,
    addrE p 0 - (1 - idleE) * inE 0,
    padE p 1 - (1 - (isLoadE p + isStoreE p)),
    isWriteE p 1 - isStoreE p,
    isRamE p 1 - (isLoadE p + isStoreE p),
    addrE p 1 - (isLoadE p + isStoreE p) * argumentE p,
    vwE p 1 - (isStoreE p * inE 1 + isLoadE p * vrE p 1),
    outE 0 - (inE 0 + isLoadE p + isStoreE p + isLoadiE p),
    outE 1 - (isLoadE p * vrE p 1 + isLoadiE p * argumentE p + (1 - isLoadE p - isLoadiE p) * inE 1)]

theorem opWords_eq : Words.opWords p = 3 + p.μ + 64 + p.wTs + p.wTs := by
  simp [Words.opWords, Plan.opWidth]

theorem eval_slotE (j : Fin p.bOps) {k : ℕ} (hk : k < Words.opWords p) :
    (slotE p j k).eval v.env = (v.ops j).word k :=
  v.env_op j hk

theorem eval_padE (j : Fin p.bOps) : (padE p j).eval v.env = (v.ops j).pad := by
  rw [padE, eval_slotE v j (by rw [opWords_eq]; omega)]
  simp [OpSlotBits.word]

theorem eval_isWriteE (j : Fin p.bOps) : (isWriteE p j).eval v.env = (v.ops j).isWrite := by
  rw [isWriteE, eval_slotE v j (by rw [opWords_eq]; omega)]
  simp [OpSlotBits.word]

theorem eval_isRamE (j : Fin p.bOps) : (isRamE p j).eval v.env = (v.ops j).isRam := by
  rw [isRamE, eval_slotE v j (by rw [opWords_eq]; omega)]
  simp [OpSlotBits.word]

theorem eval_addrE (j : Fin p.bOps) : (addrE p j).eval v.env = bitsWord (v.ops j).addr := by
  rw [addrE, Expr.eval_bits]
  congr 1
  funext k
  have := k.isLt
  rw [eval_slotE v j (by rw [opWords_eq]; omega)]
  simp (disch := omega) only [OpSlotBits.word, if_neg, dif_pos, Nat.add_sub_cancel_left, Fin.eta]

theorem eval_vrBitE (j : Fin p.bOps) (k : Fin 32) :
    (vrBitE p j k).eval v.env = (v.ops j).vr k := by
  have := k.isLt
  rw [vrBitE, eval_slotE v j (by rw [opWords_eq]; omega)]
  simp (disch := omega) only [OpSlotBits.word, if_neg, dif_neg, dif_pos]
  congr 1; ext; simp only; omega

theorem eval_vrE (j : Fin p.bOps) : (vrE p j).eval v.env = bitsWord (v.ops j).vr := by
  rw [vrE, Expr.eval_bits]
  exact congrArg bitsWord (funext fun k => eval_vrBitE v j k)

theorem eval_vwE (j : Fin p.bOps) : (vwE p j).eval v.env = bitsWord (v.ops j).vw := by
  rw [vwE, Expr.eval_bits]
  congr 1
  funext k
  have := k.isLt
  rw [eval_slotE v j (by rw [opWords_eq]; omega)]
  simp (disch := omega) only [OpSlotBits.word, if_neg, dif_neg, dif_pos]
  congr 1; ext; simp only; omega

theorem rows_iff : ConstraintsHold v.env (rows p) ↔ PortRows two v := by
  have pad0 : (padE p 0).eval v.env = (v.ops (fetchSlot two)).pad := eval_padE v (fetchSlot two)
  have pad1 : (padE p 1).eval v.env = (v.ops (dataSlot two)).pad := eval_padE v (dataSlot two)
  have write0 : (isWriteE p 0).eval v.env = (v.ops (fetchSlot two)).isWrite :=
    eval_isWriteE v (fetchSlot two)
  have write1 : (isWriteE p 1).eval v.env = (v.ops (dataSlot two)).isWrite :=
    eval_isWriteE v (dataSlot two)
  have ram0 : (isRamE p 0).eval v.env = (v.ops (fetchSlot two)).isRam :=
    eval_isRamE v (fetchSlot two)
  have ram1 : (isRamE p 1).eval v.env = (v.ops (dataSlot two)).isRam :=
    eval_isRamE v (dataSlot two)
  have addr0 : (addrE p 0).eval v.env = bitsWord (v.ops (fetchSlot two)).addr :=
    eval_addrE v (fetchSlot two)
  have addr1 : (addrE p 1).eval v.env = bitsWord (v.ops (dataSlot two)).addr :=
    eval_addrE v (dataSlot two)
  have vr1 : (vrE p 1).eval v.env = bitsWord (v.ops (dataSlot two)).vr := eval_vrE v (dataSlot two)
  have vw1 : (vwE p 1).eval v.env = bitsWord (v.ops (dataSlot two)).vw := eval_vwE v (dataSlot two)
  have bit (k : Fin 32) : (vrBitE p 0 k).eval v.env = instructionBit two v k :=
    eval_vrBitE v (fetchSlot two) k
  have bit0 : (vrBitE p 0 0).eval v.env = instructionBit two v 0 := bit 0
  have bit1 : (vrBitE p 0 1).eval v.env = instructionBit two v 1 := bit 1
  have load : (isLoadE p).eval v.env = isLoad two v := by
    simp only [isLoadE, Circuit.Expr.eval_hmul, Circuit.Expr.eval_sub, Expr.eval_one, bit0, bit1]
    rfl
  have store : (isStoreE p).eval v.env = isStore two v := by
    simp only [isStoreE, Circuit.Expr.eval_hmul, Circuit.Expr.eval_sub, Expr.eval_one, bit0, bit1]
    rfl
  have loadi : (isLoadiE p).eval v.env = isLoadi two v := by
    simp only [isLoadiE, Circuit.Expr.eval_hmul, bit0, bit1]
    rfl
  have arg : (argumentE p).eval v.env = argument two v := by
    rw [argumentE, Expr.eval_chunk, List.map_drop, List.map_ofFn]
    simp only [argument, Function.comp_def, bit]
    rfl
  have idleEval : idleE.eval v.env = idle v := by
    change v.env 4 = v.aux 0
    exact v.env_aux 0
  have in0 : (inE 0).eval v.env = v.appIn 0 := v.env_appIn 0
  have in1 : (inE 1).eval v.env = v.appIn 1 := v.env_appIn 1
  have out0 : (outE 0).eval v.env = v.appOut 0 := v.env_appOut 0
  have out1 : (outE 1).eval v.env = v.appOut 1 := v.env_appOut 1
  simp only [rows, ConstraintsHold, List.mem_cons, List.not_mem_nil, or_false,
    forall_eq_or_imp, forall_eq, Circuit.Expr.eval_hmul, Circuit.Expr.eval_hadd, sub_eq_zero,
    Circuit.Expr.eval_sub, Expr.eval_one, bitRow_iff, pad0, pad1, write0, write1, ram0, ram1,
    addr0, addr1, vr1, vw1, load, store, loadi, arg, idleEval, in0, in1, out0, out1]
  constructor
  · rintro ⟨r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11⟩
    exact ⟨r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11⟩
  · rintro ⟨r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11⟩
    exact ⟨r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11⟩

/-! ### Ob7 -/

theorem ports_two (ops : Fin p.bOps → OpSlotBits p) :
    (List.ofFn fun j => (ops j).decode.port) =
      [(ops (fetchSlot two)).decode.port, (ops (dataSlot two)).decode.port] := by
  apply List.ext_getElem (by simp [two])
  intro k h₁ h₂
  simp only [List.getElem_ofFn]
  rcases k with _ | _ | k
  · rfl
  · rfl
  · simp at h₂

/-- The instruction argument is the fetched word divided by four. -/
theorem argument_val (vrBits : ∀ k, IsBit ((v.ops (fetchSlot two)).vr k)) :
    (argument two v).val = (v.ops (fetchSlot two)).decode.vr / 4 := by
  have all : ∀ x ∈ (List.ofFn (v.ops (fetchSlot two)).vr).drop 2, IsBit x := fun x member => by
    obtain ⟨k, rfl⟩ := List.mem_ofFn.mp (List.mem_of_mem_drop member)
    exact vrBits k
  have small := chunkValue_lt (((List.ofFn (v.ops (fetchSlot two)).vr).drop 2).map toBool)
  simp only [List.length_map, List.length_drop, List.length_ofFn] at small
  have split := bitsNat_split (v.ops (fetchSlot two)).vr
  have low0 := Bool.toNat_le (toBool ((v.ops (fetchSlot two)).vr 0))
  have low1 := Bool.toNat_le (toBool ((v.ops (fetchSlot two)).vr 1))
  rw [argument, chunkWord_eq all, natWord_val (lt_trans small (by decide))]
  show _ = bitsNat (v.ops (fetchSlot two)).vr / 4
  omega

/-- Ob7: on well-formed slots, the port rows give one machine step on the
decoded ports. -/
theorem sound (valid : p.Valid) (wf : ∀ j, (v.ops j).WellFormed) (rows : PortRows two v) :
    Step (State.ofWords v.appIn) (List.ofFn fun j => (v.ops j).decode.port)
      (State.ofWords v.appOut) := by
  rw [ports_two two]
  obtain ⟨-, -, -, addrF, vrF, vwF⟩ := (wf (fetchSlot two)).fieldBits
  obtain ⟨-, -, -, addrD, vrD, vwD⟩ := (wf (dataSlot two)).fieldBits
  have fetchPad := rows.fetchPad
  have pcOut := rows.pcOut
  have accOut := rows.accOut
  have dataPad := rows.dataPad
  rcases rows.idleBit with active | idle
  · -- A fetched instruction.
    have fetchActive : (v.ops (fetchSlot two)).pad = 0 := fetchPad.trans active
    have fetchRead := rows.fetchRead
    have fetchRom := rows.fetchRom
    have fetchAddr := rows.fetchAddr
    rw [active, sub_zero, one_mul] at fetchAddr
    have keeps := (wf (fetchSlot two)).readKeeps
    rw [fetchRead, sub_zero, one_mul, sub_eq_zero] at keeps
    have pc : (v.appIn 0).val = bitsNat (v.ops (fetchSlot two)).addr := by
      rw [← fetchAddr]
      exact StepWitness.bitsWord_val addrF (Plan.mu_le valid)
    have pcSmall : (v.appIn 0).val + 1 < goldilocksModulus := by
      have := bitsNat_lt (v.ops (fetchSlot two)).addr
      have := Plan.two_pow_mu_lt valid
      omega
    have fetchPort : (v.ops (fetchSlot two)).decode.port =
        some (fetch (State.ofWords v.appIn) (v.ops (fetchSlot two)).decode.vr) := by
      rw [OpSlotBits.port_active fetchActive, fetchRead, fetchRom, StepWitness.toBool_zero,
        StepWitness.bitsWord_inj (by norm_num) vwF vrF keeps]
      simp only [fetch, State.ofWords, pc]
      rfl
    have argument : (argument two v).val = bitsNat (v.ops (fetchSlot two)).vr / 4 :=
      argument_val two v vrF
    have split := bitsNat_split (v.ops (fetchSlot two)).vr
    have dataValue := rows.dataValue
    have dataAddr := rows.dataAddr
    have dataWrite := rows.dataWrite
    have dataRam := rows.dataRam
    have addrVal : (bitsWord (v.ops (dataSlot two)).addr).val = bitsNat (v.ops (dataSlot two)).addr :=
      StepWitness.bitsWord_val addrD (Plan.mu_le valid)
    have vrVal : (bitsWord (v.ops (dataSlot two)).vr).val = bitsNat (v.ops (dataSlot two)).vr :=
      StepWitness.bitsWord_val vrD (by norm_num)
    have vwVal : (bitsWord (v.ops (dataSlot two)).vw).val = bitsNat (v.ops (dataSlot two)).vw :=
      StepWitness.bitsWord_val vwD (by norm_num)
    refine Or.inr ⟨(v.ops (fetchSlot two)).decode.vr, (v.ops (dataSlot two)).decode.vr, ?_⟩
    rw [fetchPort]
    have bit0 := vrF 0
    have bit1 := vrF 1
    rcases bit0 with b0 | b0 <;> rcases bit1 with b1 | b1
    · -- halt
      have mod : bitsNat (v.ops (fetchSlot two)).vr % 4 = 0 := by
        simp only [split, b0, b1, StepWitness.toBool_zero, Bool.toNat_false]
        omega
      have load : isLoad two v = 0 := by simp [isLoad, instructionBit, b0]
      have store : isStore two v = 0 := by simp [isStore, instructionBit, b1]
      have loadi : isLoadi two v = 0 := by simp [isLoadi, instructionBit, b0]
      simp only [load, store, loadi, add_zero, zero_add, sub_zero, zero_mul,
        one_mul] at dataPad pcOut accOut
      refine ⟨?_, ?_⟩
      · rw [OpSlotBits.port_pad dataPad]
        simp [dataPort, OpSlotBits.decode, mod]
      · simp [exec, State.ofWords, OpSlotBits.decode, mod, pcOut, accOut]
    · -- store
      have mod : bitsNat (v.ops (fetchSlot two)).vr % 4 = 2 := by
        simp only [split, b0, b1, StepWitness.toBool_zero, StepWitness.toBool_one,
          Bool.toNat_false, Bool.toNat_true]
        omega
      have load : isLoad two v = 0 := by simp [isLoad, instructionBit, b0]
      have store : isStore two v = 1 := by simp [isStore, instructionBit, b0, b1]
      have loadi : isLoadi two v = 0 := by simp [isLoadi, instructionBit, b0]
      simp only [load, store, loadi, add_zero, zero_add, sub_zero, sub_self, zero_mul,
        one_mul] at dataPad dataWrite dataRam dataAddr dataValue pcOut accOut
      refine ⟨?_, ?_⟩
      · rw [OpSlotBits.port_active dataPad, dataWrite, dataRam, StepWitness.toBool_one]
        have addr : bitsNat (v.ops (dataSlot two)).addr =
            bitsNat (v.ops (fetchSlot two)).vr / 4 := by
          rw [← addrVal, dataAddr, argument]
        have acc : bitsNat (v.ops (dataSlot two)).vw = (v.appIn 1).val := by
          rw [← vwVal, dataValue]
        simp only [dataPort, OpSlotBits.decode, mod, addr, acc, State.ofWords]
      · simp only [exec, State.ofWords, OpSlotBits.decode, mod, pcOut, accOut,
          val_add_one pcSmall]
    · -- load
      have mod : bitsNat (v.ops (fetchSlot two)).vr % 4 = 1 := by
        simp only [split, b0, b1, StepWitness.toBool_zero, StepWitness.toBool_one,
          Bool.toNat_false, Bool.toNat_true]
        omega
      have load : isLoad two v = 1 := by simp [isLoad, instructionBit, b0, b1]
      have store : isStore two v = 0 := by simp [isStore, instructionBit, b0, b1]
      have loadi : isLoadi two v = 0 := by simp [isLoadi, instructionBit, b1]
      simp only [load, store, loadi, add_zero, zero_add, sub_self, zero_mul,
        one_mul] at dataPad dataWrite dataRam dataAddr dataValue pcOut accOut
      refine ⟨?_, ?_⟩
      · rw [OpSlotBits.port_active dataPad, dataWrite, dataRam, StepWitness.toBool_zero,
          StepWitness.toBool_one]
        have addr : bitsNat (v.ops (dataSlot two)).addr =
            bitsNat (v.ops (fetchSlot two)).vr / 4 := by
          rw [← addrVal, dataAddr, argument]
        have keeps : bitsNat (v.ops (dataSlot two)).vw = bitsNat (v.ops (dataSlot two)).vr := by
          rw [← vwVal, ← vrVal, dataValue]
        simp only [dataPort, OpSlotBits.decode, mod, addr, keeps]
      · simp only [exec, State.ofWords, OpSlotBits.decode, mod, pcOut, accOut,
          val_add_one pcSmall, vrVal]
    · -- loadi
      have mod : bitsNat (v.ops (fetchSlot two)).vr % 4 = 3 := by
        simp only [split, b0, b1, StepWitness.toBool_one, Bool.toNat_true]
        omega
      have load : isLoad two v = 0 := by simp [isLoad, instructionBit, b1]
      have store : isStore two v = 0 := by simp [isStore, instructionBit, b0]
      have loadi : isLoadi two v = 1 := by simp [isLoadi, instructionBit, b0, b1]
      simp only [load, store, loadi, add_zero, zero_add, sub_zero, sub_self, zero_mul,
        one_mul] at dataPad pcOut accOut
      refine ⟨?_, ?_⟩
      · rw [OpSlotBits.port_pad dataPad]
        simp [dataPort, OpSlotBits.decode, mod]
      · simp only [exec, State.ofWords, OpSlotBits.decode, mod, pcOut, accOut,
          val_add_one pcSmall, argument]
  · -- An idle step.
    have tail := (wf (fetchSlot two)).zeros (fetchPad.trans idle)
    have b0 : instructionBit two v 0 = 0 := tail.2.2.2.1 0
    have b1 : instructionBit two v 1 = 0 := tail.2.2.2.1 1
    have load : isLoad two v = 0 := by rw [isLoad, b0, zero_mul]
    have store : isStore two v = 0 := by rw [isStore, b1, mul_zero]
    have loadi : isLoadi two v = 0 := by rw [isLoadi, b0, zero_mul]
    simp only [load, store, loadi, add_zero, zero_add, sub_zero, zero_mul,
      one_mul] at dataPad pcOut accOut
    refine Or.inl ⟨?_, ?_⟩
    · rw [OpSlotBits.port_pad (fetchPad.trans idle), OpSlotBits.port_pad dataPad]
    · simp only [State.ofWords, pcOut, accOut]

/-! ### Ob5 for the machine -/

/-- A bit whose `toBool` is known. -/
private theorem bit_of_toBool {x : F} (bit : IsBit x) {b : Bool} (h : toBool x = b) :
    x = if b then 1 else 0 := by
  cases b
  · exact (StepWitness.toBool_false bit).mp h
  · exact StepWitness.toBool_true.mp h

/-- Ob5 for the machine: every machine step has an idle flag that satisfies
the port rows, on well-formed slots whose decoded ports are the step's
ports. -/
theorem complete {s s' : State} {ports : List (Option PortAccess)} (ops : Fin p.bOps → OpSlotBits p)
    (step : Step s ports s') (wf : ∀ j, (ops j).WellFormed)
    (decoded : (List.ofFn fun j => (ops j).decode.port) = ports) :
    ∃ priv : Fin 1 → F, PortRows two (⟨s.words, s'.words, priv, ops⟩ : AppView p shape) := by
  rw [ports_two two] at decoded
  obtain ⟨-, -, -, addrF, vrF, vwF⟩ := (wf (fetchSlot two)).fieldBits
  obtain ⟨-, -, -, addrD, vrD, vwD⟩ := (wf (dataSlot two)).fieldBits
  rcases step with ⟨idlePorts, next⟩ | ⟨word, data, activePorts, next⟩
  · -- An idle step.
    subst next
    rw [idlePorts] at decoded
    simp only [List.cons.injEq, and_true] at decoded
    have pad0 := (wf (fetchSlot two)).pad_one decoded.1
    have pad1 := (wf (dataSlot two)).pad_one decoded.2
    obtain ⟨w0, r0, a0, v0, -⟩ := (wf (fetchSlot two)).zeros pad0
    obtain ⟨w1, r1, a1, v1, u1⟩ := (wf (dataSlot two)).zeros pad1
    refine ⟨![1], ?_⟩
    constructor <;>
      simp [idle, isLoad, isStore, isLoadi, instructionBit, pad0, pad1, w0, r0, w1, r1, v0,
        bitsWord_zeros a0, bitsWord_zeros a1, bitsWord_zeros u1, bitsWord_zeros v1, IsBit]
  · -- A fetched instruction.
    subst next
    rw [activePorts] at decoded
    simp only [List.cons.injEq, and_true] at decoded
    obtain ⟨pad0, write0, ram0, addr0, vr0, -⟩ := (wf (fetchSlot two)).active decoded.1
    simp only [fetch] at write0 ram0 addr0 vr0
    have write0 := bit_of_toBool (wf (fetchSlot two)).fieldBits.2.1 write0
    have ram0 := bit_of_toBool (wf (fetchSlot two)).fieldBits.2.2.1 ram0
    simp only [Bool.false_eq_true, if_false] at write0 ram0
    have addrWord : bitsWord (ops (fetchSlot two)).addr = natWord s.pc := by
      rw [bitsWord_eq addrF, addr0]
    have split := bitsNat_split (ops (fetchSlot two)).vr
    rw [vr0] at split
    have low0 := Bool.toNat_le (toBool ((ops (fetchSlot two)).vr 0))
    have low1 := Bool.toNat_le (toBool ((ops (fetchSlot two)).vr 1))
    have all : ∀ x ∈ (List.ofFn (ops (fetchSlot two)).vr).drop 2, IsBit x := fun x member => by
      obtain ⟨k, rfl⟩ := List.mem_ofFn.mp (List.mem_of_mem_drop member)
      exact vrF k
    have value : chunkValue (((List.ofFn (ops (fetchSlot two)).vr).drop 2).map toBool) = word / 4 := by
      omega
    have argWord : chunkWord ((List.ofFn (ops (fetchSlot two)).vr).drop 2) = natWord (word / 4) := by
      rw [chunkWord_eq all, value]
    have bit0 := bit_of_toBool (vrF 0) rfl
    have bit1 := bit_of_toBool (vrF 1) rfl
    have argEq (x y : Fin 2 → F) (z : Fin 1 → F) :
        argument two (⟨x, y, z, ops⟩ : AppView p shape) = natWord (word / 4) := argWord
    have pcSucc : natWord (s.pc + 1) = natWord s.pc + 1 := by rw [natWord_add, natWord_one]
    rcases (by omega : word % 4 = 0 ∨ word % 4 = 1 ∨ word % 4 = 2 ∨ word % 4 = 3) with r | r | r | r
    · -- halt
      have t0 : toBool ((ops (fetchSlot two)).vr 0) = false := by
        cases h : toBool ((ops (fetchSlot two)).vr 0)
        · rfl
        · rw [h, Bool.toNat_true] at split; omega
      have t1 : toBool ((ops (fetchSlot two)).vr 1) = false := by
        cases h : toBool ((ops (fetchSlot two)).vr 1)
        · rfl
        · rw [h, Bool.toNat_true] at split; omega
      rw [t0] at bit0
      rw [t1] at bit1
      simp only [dataPort, r] at decoded
      have pad1 := (wf (dataSlot two)).pad_one decoded.2
      obtain ⟨w1, r1, a1, v1, u1⟩ := (wf (dataSlot two)).zeros pad1
      refine ⟨![0], ?_⟩
      constructor <;>
        simp [idle, isLoad, isStore, isLoadi, instructionBit, argEq, pad0, write0, ram0,
          addrWord, bit0, bit1, pad1, w1, r1, bitsWord_zeros a1, bitsWord_zeros u1,
          bitsWord_zeros v1, exec, r, State.words, IsBit]
    · -- load
      have t0 : toBool ((ops (fetchSlot two)).vr 0) = true := by
        cases h : toBool ((ops (fetchSlot two)).vr 0)
        · rw [h, Bool.toNat_false] at split; omega
        · rfl
      have t1 : toBool ((ops (fetchSlot two)).vr 1) = false := by
        cases h : toBool ((ops (fetchSlot two)).vr 1)
        · rfl
        · rw [h, Bool.toNat_true] at split; omega
      rw [t0] at bit0
      rw [t1] at bit1
      simp only [dataPort, r] at decoded
      obtain ⟨pad1, write1, ram1, addr1, vr1, vw1⟩ := (wf (dataSlot two)).active decoded.2
      have write1 := bit_of_toBool (wf (dataSlot two)).fieldBits.2.1 write1
      have ram1 := bit_of_toBool (wf (dataSlot two)).fieldBits.2.2.1 ram1
      simp only [Bool.false_eq_true, if_false, if_true] at write1 ram1
      have addrData : bitsWord (ops (dataSlot two)).addr = natWord (word / 4) := by
        rw [bitsWord_eq addrD, addr1]
      have vrData : bitsWord (ops (dataSlot two)).vr = natWord data := by
        rw [bitsWord_eq vrD, vr1]
      have vwData : bitsWord (ops (dataSlot two)).vw = natWord data := by
        rw [bitsWord_eq vwD, vw1]
      refine ⟨![0], ?_⟩
      constructor <;>
        simp [idle, isLoad, isStore, isLoadi, instructionBit, argEq, pad0, write0,
          ram0, addrWord, bit0, bit1, pad1, write1, ram1, addrData, vrData, vwData, exec, r,
          State.words, pcSucc, IsBit]
    · -- store
      have t0 : toBool ((ops (fetchSlot two)).vr 0) = false := by
        cases h : toBool ((ops (fetchSlot two)).vr 0)
        · rfl
        · rw [h, Bool.toNat_true] at split; omega
      have t1 : toBool ((ops (fetchSlot two)).vr 1) = true := by
        cases h : toBool ((ops (fetchSlot two)).vr 1)
        · rw [h, Bool.toNat_false] at split; omega
        · rfl
      rw [t0] at bit0
      rw [t1] at bit1
      simp only [dataPort, r] at decoded
      obtain ⟨pad1, write1, ram1, addr1, -, vw1⟩ := (wf (dataSlot two)).active decoded.2
      have write1 := bit_of_toBool (wf (dataSlot two)).fieldBits.2.1 write1
      have ram1 := bit_of_toBool (wf (dataSlot two)).fieldBits.2.2.1 ram1
      simp only [if_true] at write1 ram1
      have addrData : bitsWord (ops (dataSlot two)).addr = natWord (word / 4) := by
        rw [bitsWord_eq addrD, addr1]
      have vwData : bitsWord (ops (dataSlot two)).vw = natWord s.acc := by
        rw [bitsWord_eq vwD, vw1]
      refine ⟨![0], ?_⟩
      constructor <;>
        simp [idle, isLoad, isStore, isLoadi, instructionBit, argEq, pad0, write0,
          ram0, addrWord, bit0, bit1, pad1, write1, ram1, addrData, vwData, exec, r,
          State.words, pcSucc, IsBit]
    · -- loadi
      have t0 : toBool ((ops (fetchSlot two)).vr 0) = true := by
        cases h : toBool ((ops (fetchSlot two)).vr 0)
        · rw [h, Bool.toNat_false] at split; omega
        · rfl
      have t1 : toBool ((ops (fetchSlot two)).vr 1) = true := by
        cases h : toBool ((ops (fetchSlot two)).vr 1)
        · rw [h, Bool.toNat_false] at split; omega
        · rfl
      rw [t0] at bit0
      rw [t1] at bit1
      simp only [dataPort, r] at decoded
      have pad1 := (wf (dataSlot two)).pad_one decoded.2
      obtain ⟨w1, r1, a1, v1, u1⟩ := (wf (dataSlot two)).zeros pad1
      refine ⟨![0], ?_⟩
      constructor <;>
        simp [idle, isLoad, isStore, isLoadi, instructionBit, argEq, pad0, write0,
          ram0, addrWord, bit0, bit1, pad1, w1, r1, bitsWord_zeros a1, bitsWord_zeros u1,
          bitsWord_zeros v1, exec, r, State.words, pcSucc, IsBit]

/-- The machine over its two state words: one machine step between the
decoded states. -/
def wordApplication (p : Plan) (two : p.bOps = 2) : Application p (Fin 2 → F) where
  Step w ports w' := Step (State.ofWords w) ports (State.ofWords w')
  idle w := (application p two).idle (State.ofWords w)

/-- The machine as a port application of a plan with two ports. -/
def portApp (p : Plan) (two : p.bOps = 2) : PortApp p where
  width := 2
  aux := 1
  model := wordApplication p two
  rows := rows p
  width_lt := by decide
  portsLength := by
    intro s ports s' step
    rcases step with ⟨rfl, -⟩ | ⟨_, _, rfl, -⟩ <;> simp [two]
  sound := fun valid v wf hold => sound two v valid wf ((rows_iff two v).mp hold)
  complete := by
    intro _ _ _ _ ops step wf decoded
    obtain ⟨priv, rows⟩ := complete two ops step wf decoded
    rw [State.words_ofWords, State.words_ofWords] at rows
    exact ⟨priv, (rows_iff two _).mpr rows⟩

end Machine

end NightstreamFPrime.Lifecycle.Nebula
