import NightstreamFPrime.Lifecycle.Nebula.HonestRows
import NightstreamFPrime.Lifecycle.Nebula.WitnessEncoding

/-! Owns Ob5 for the relation of the first memory application: every step that
the model's `invoke` accepts from a reachable carry, with one machine step on
its ports, has witness words that satisfy the memory program's validity
predicate, and the program's step function gives the state digest of the
output state and the output carry. It owns the record chains, the challenge
transcript, the input state, and the machine rows of the honest witness. -/

namespace NightstreamFPrime.Lifecycle.Nebula.Honest

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Nebula
open scoped NightstreamFPrime.Spec.GoldilocksExtensionRing
open Fin.CommRing

variable {p : Plan} {c : MemoryCarry} {inv : Invocation Machine.State Digest} (s : Machine.State)

/-! ### Chains and the challenge transcript -/

theorem ofFn_getD {α : Type} {n : ℕ} (l : List α) (d : α) (length : l.length = n) :
    List.ofFn (fun j : Fin n => l.getD j d) = l := by
  apply List.ext_getElem (by simp [length])
  intro k h₁ h₂
  simp only [List.getElem_ofFn, List.getD_eq_getElem _ _ h₂]

theorem opsLaneBits_eq (accepted : Accepted p c inv) :
    (witness p c inv s).opsLaneBits = (opsLane p inv.records.ops).map bitWord := by
  show (List.ofFn fun j : Fin p.bOps => (encodeOp p (opSlot inv j) (diff c inv j)).lane).flatten = _
  conv_rhs => rw [← ofFn_getD inv.records.ops OpSlot.padSlot accepted.rows.shaped.opsLength]
  simp only [lane_encodeOp, opsLane, List.flatMap, List.map_flatten, List.map_ofFn,
    Function.comp_def, opSlot]

theorem scanLaneBits_eq (l : List ScanSlot) (length : l.length = p.bScan) :
    StepWitness.scanLaneBits (fun j : Fin p.bScan => encodeScan p (l.getD j ⟨0, 0⟩)) =
      (scanLane p l).map bitWord := by
  conv_rhs => rw [← ofFn_getD l ⟨0, 0⟩ length]
  simp only [StepWitness.scanLaneBits, lane_encodeScan, scanLane, List.flatMap, List.map_flatten,
    List.map_ofFn, Function.comp_def]

theorem idxEff_val (valid : p.Valid) (reach : Reach p c) :
    (witness p c inv s).idxEff.val = (start p c inv).idx :=
  natWord_val (lt_trans (start_idx_lt valid reach) (Plan.n_lt valid))

theorem previous_ops : (witness p c inv s).previousDigest 0 = (start p c inv).seen.ops := by
  funext i
  fin_cases i <;> rfl

theorem previous_initial : (witness p c inv s).previousDigest 4 = (start p c inv).seen.initial := by
  funext i
  fin_cases i <;> rfl

theorem previous_final : (witness p c inv s).previousDigest 8 = (start p c inv).seen.final := by
  funext i
  fin_cases i <;> rfl

theorem digest_seenOps :
    StepWitness.carryDigest (witness p c inv s).carryOut 23 = (finished p c inv).seen.ops := by
  funext i
  fin_cases i <;> rfl

theorem digest_seenInitial :
    StepWitness.carryDigest (witness p c inv s).carryOut 27 = (finished p c inv).seen.initial := by
  funext i
  fin_cases i <;> rfl

theorem digest_seenFinal :
    StepWitness.carryDigest (witness p c inv s).carryOut 31 = (finished p c inv).seen.final := by
  funext i
  fin_cases i <;> rfl

theorem chainOps_row (valid : p.Valid) (reach : Reach p c) (accepted : Accepted p c inv) :
    StepWitness.carryDigest (witness p c inv s).carryOut 23 =
      chainLink .ops (witness p c inv s).idxEff ((witness p c inv s).previousDigest 0)
        (packWords (witness p c inv s).opsLaneBits) := by
  rw [chainLink_eq _ _ _ (by rw [opsLaneBits_eq s accepted]; exact bitWord_bits),
    opsLaneBits_eq s accepted, map_toBool_bitWord, idxEff_val s valid reach, previous_ops,
    digest_seenOps, finished_seen]
  rfl

theorem chainInitial_row (valid : p.Valid) (reach : Reach p c) (accepted : Accepted p c inv) :
    StepWitness.carryDigest (witness p c inv s).carryOut 27 =
      chainLink .mem (witness p c inv s).idxEff ((witness p c inv s).previousDigest 4)
        (packWords (StepWitness.scanLaneBits (witness p c inv s).initial)) := by
  have lanes : StepWitness.scanLaneBits (witness p c inv s).initial =
      (scanLane p inv.records.initialScan).map bitWord :=
    scanLaneBits_eq _ accepted.rows.shaped.initialLength
  rw [chainLink_eq _ _ _ (by rw [lanes]; exact bitWord_bits), lanes, map_toBool_bitWord,
    idxEff_val s valid reach, previous_initial, digest_seenInitial, finished_seen]
  rfl

theorem chainFinal_row (valid : p.Valid) (reach : Reach p c) (accepted : Accepted p c inv) :
    StepWitness.carryDigest (witness p c inv s).carryOut 31 =
      chainLink .mem (witness p c inv s).idxEff ((witness p c inv s).previousDigest 8)
        (packWords (StepWitness.scanLaneBits (witness p c inv s).final)) := by
  have lanes : StepWitness.scanLaneBits (witness p c inv s).final =
      (scanLane p inv.records.finalScan).map bitWord :=
    scanLaneBits_eq _ accepted.rows.shaped.finalLength
  rw [chainLink_eq _ _ _ (by rw [lanes]; exact bitWord_bits), lanes, map_toBool_bitWord,
    idxEff_val s valid reach, previous_final, digest_seenFinal, finished_seen]
  rfl

theorem freshEta_row (valid : p.Valid) (reach : Reach p c) :
    etaChallenges ⟨planDigest p, ((witness p c inv s).cIn 2).val,
      (witness p c inv s).proposalDigest 0,
      StepWitness.carryDigest (witness p c inv s).carryIn 35,
      (witness p c inv s).proposalDigest 4⟩ = (witness p c inv s).freshEta := by
  have ts : ((witness p c inv s).cIn 2).val = c.ts := by
    rw [cIn_val s (by decide), carryVector_ts, natWord_val (canonical valid reach).2.2]
  have first : (witness p c inv s).proposalDigest 0 = inv.proposal.1 := by
    funext i
    fin_cases i <;> rfl
  have second : (witness p c inv s).proposalDigest 4 = inv.proposal.2 := by
    funext i
    fin_cases i <;> rfl
  have root : StepWitness.carryDigest (witness p c inv s).carryIn 35 = c.memRoot := by
    funext i
    fin_cases i <;> rfl
  rw [ts, first, second, root]
  rfl

/-- The memory rows of spec §8–§11 and the input state hold on the honest
witness. -/
theorem rowsHold (valid : p.Valid) (reach : Reach p c) (accepted : Accepted p c inv) :
    (witness p c inv s).RowsHold (stateWords (List.ofFn s.words) (carryWords c)) :=
  { polyRows s valid reach accepted with
    chainOps := chainOps_row s valid reach accepted
    chainInitial := chainInitial_row s valid reach accepted
    chainFinal := chainFinal_row s valid reach accepted
    freshEta := freshEta_row s valid reach
    stateIn := rfl }


/-! ### Machine rows -/

theorem natBits_apply (n x : ℕ) (k : Fin n) : natBits n x k = bitWord (x.testBit k) := rfl

theorem bitWord_false : bitWord false = 0 := rfl

theorem bitWord_true : bitWord true = 1 := rfl

theorem testBit_zero_mod (word : ℕ) : word.testBit 0 = decide (word % 4 = 1 ∨ word % 4 = 3) := by
  rw [Nat.testBit_zero]
  exact decide_eq_decide.mpr (by omega)

theorem testBit_one_mod (word : ℕ) : word.testBit 1 = decide (word % 4 = 2 ∨ word % 4 = 3) := by
  rw [Nat.testBit_succ, Nat.testBit_zero]
  exact decide_eq_decide.mpr (by omega)

/-- The instruction argument of a 32-bit word: its value divided by four. -/
theorem argument_natBits {word : ℕ} (small : word < 2 ^ 32) :
    chunkWord ((List.ofFn (natBits 32 word)).drop 2) = natWord (word / 4) := by
  have all : ∀ x ∈ (List.ofFn (natBits 32 word)).drop 2, IsBit x := fun x member => by
    obtain ⟨k, rfl⟩ := List.mem_ofFn.mp (List.mem_of_mem_drop member)
    exact bitWord_isBit _
  have split := bitsNat_split (natBits 32 word)
  rw [bitsNat_natBits small] at split
  have low0 := Bool.toNat_le (toBool (natBits 32 word 0))
  have low1 := Bool.toNat_le (toBool (natBits 32 word 1))
  have value : chunkValue (((List.ofFn (natBits 32 word)).drop 2).map toBool) = word / 4 := by
    omega
  rw [chunkWord_eq all, value]

theorem port_some {t : OpSlot} {a : PortAccess} (h : t.port = some a) :
    t.pad = false ∧ t.isWrite = a.isWrite ∧ t.isRam = a.isRam ∧ t.addr = a.addr ∧ t.vr = a.vr ∧
      t.vw = a.vw := by
  unfold OpSlot.port at h
  split at h
  · cases h
  · next pad =>
    obtain rfl := Option.some.inj h
    exact ⟨by simpa using pad, rfl, rfl, rfl, rfl, rfl⟩

theorem port_none {t : OpSlot} (rows : t.Rows p) (h : t.port = none) : t = OpSlot.padSlot := by
  apply rows.padZero
  unfold OpSlot.port at h
  split at h
  · assumption
  · cases h

theorem ops_two (accepted : Accepted p c inv) (two : p.bOps = 2) :
    inv.records.ops = [opSlot inv 0, opSlot inv 1] := by
  rw [← ofFn_getD inv.records.ops OpSlot.padSlot (accepted.rows.shaped.opsLength.trans two)]
  simp [List.ofFn_succ, opSlot]

theorem bitsWord_natBits_zero (n : ℕ) : bitsWord (natBits n 0) = 0 := by
  rw [bitsWord_natBits (Nat.two_pow_pos n)]
  rfl

/-- The witness words that the machine rows read. -/
macro "machine_view" : tactic =>
  `(tactic| simp only [Honest.witness, StepWitness.pad, StepWitness.isWrite, StepWitness.isRam,
    StepWitness.addr, StepWitness.vr, StepWitness.vw, StepWitness.isLoad, StepWitness.isStore,
    StepWitness.isLoadi, StepWitness.instructionBit, StepWitness.argument, StepWitness.fetchSlot,
    StepWitness.dataSlot, encodeOp, Machine.State.words, Fin.val_zero, Fin.val_one,
    Fin.isValue, Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.head_cons])

theorem machineRows (accepted : Accepted p c inv) (two : p.bOps = 2)
    (machine : Machine.Step s inv.ports inv.next) : (witness p c inv s).MachineRows two := by
  have ops := ops_two accepted two
  have mem0 : opSlot inv 0 ∈ inv.records.ops := by rw [ops]; simp
  have mem1 : opSlot inv 1 ∈ inv.records.ops := by rw [ops]; simp
  have rows0 := accepted.rows.slots _ mem0
  have rows1 := accepted.rows.slots _ mem1
  have fits0 := accepted.rows.shaped.opsFit _ mem0
  have fits1 := accepted.rows.shaped.opsFit _ mem1
  have ports : inv.ports = [(opSlot inv 0).port, (opSlot inv 1).port] := by
    simp only [Invocation.ports, ops, List.map_cons, List.map_nil]
  rw [ports] at machine
  rcases machine with ⟨idlePorts, next⟩ | ⟨word, data, activePorts, next⟩
  · simp only [List.cons.injEq, and_true] at idlePorts
    have slot0 := port_none rows0 idlePorts.1
    have slot1 := port_none rows1 idlePorts.2
    constructor <;> machine_view <;>
      simp only [slot0, slot1, next, OpSlot.padSlot, bitWord_false, bitWord_true, natBits_apply,
        Nat.zero_testBit, bitsWord_natBits_zero, argument_natBits (show 0 < 2 ^ 32 by norm_num),
        Nat.zero_div, natWord_zero] <;> ring
  · simp only [List.cons.injEq, and_true] at activePorts
    obtain ⟨pad0, write0, ram0, addr0, vr0, vw0⟩ := port_some activePorts.1
    simp only [Machine.fetch] at write0 ram0 addr0 vr0 vw0
    have pcSmall : s.pc < 2 ^ p.μ := by rw [← addr0]; exact fits0.1
    have wordSmall : word < 2 ^ 32 := by rw [← vr0]; exact fits0.2.1
    rcases (by omega : word % 4 = 0 ∨ word % 4 = 1 ∨ word % 4 = 2 ∨ word % 4 = 3) with r | r | r | r
    · -- halt
      have b0 : word.testBit ((0 : Fin 32) : ℕ) = false := by
        rw [show ((0 : Fin 32) : ℕ) = 0 from rfl, testBit_zero_mod]; simp [r]
      have b1 : word.testBit ((1 : Fin 32) : ℕ) = false := by
        rw [show ((1 : Fin 32) : ℕ) = 1 from rfl, testBit_one_mod]; simp [r]
      have slot1 := port_none rows1 (by simpa [Machine.dataPort, r] using activePorts.2)
      simp only [Machine.exec, r] at next
      constructor <;> machine_view <;>
        (try simp only [vr0, natBits_apply, b0, b1, bitWord_false,
          argument_natBits wordSmall]) <;>
        simp [pad0, write0, ram0, addr0, slot1, next, OpSlot.padSlot, bitWord_false,
          bitWord_true, bitsWord_natBits_zero,
          bitsWord_natBits pcSmall]
    · -- load
      have b0 : word.testBit ((0 : Fin 32) : ℕ) = true := by
        rw [show ((0 : Fin 32) : ℕ) = 0 from rfl, testBit_zero_mod]; simp [r]
      have b1 : word.testBit ((1 : Fin 32) : ℕ) = false := by
        rw [show ((1 : Fin 32) : ℕ) = 1 from rfl, testBit_one_mod]; simp [r]
      obtain ⟨pad1, write1, ram1, addr1, vr1, vw1⟩ :=
        port_some (by simpa [Machine.dataPort, r] using activePorts.2)
      dsimp only at write1 ram1 addr1 vr1 vw1
      have dataSmall : data < 2 ^ 32 := by rw [← vr1]; exact fits1.2.1
      have argSmall : word / 4 < 2 ^ p.μ := by rw [← addr1]; exact fits1.1
      simp only [Machine.exec, r] at next
      constructor <;> machine_view <;>
        (try simp only [vr0, natBits_apply, b0, b1, bitWord_false, bitWord_true,
          argument_natBits wordSmall]) <;>
        simp [pad0, write0, ram0, addr0, pad1, write1, ram1, addr1, vr1, vw1, next,
          bitWord_false, bitWord_true, 
          bitsWord_natBits pcSmall, bitsWord_natBits dataSmall, bitsWord_natBits argSmall,
          natWord_add, natWord_one]
    · -- store
      have b0 : word.testBit ((0 : Fin 32) : ℕ) = false := by
        rw [show ((0 : Fin 32) : ℕ) = 0 from rfl, testBit_zero_mod]; simp [r]
      have b1 : word.testBit ((1 : Fin 32) : ℕ) = true := by
        rw [show ((1 : Fin 32) : ℕ) = 1 from rfl, testBit_one_mod]; simp [r]
      obtain ⟨pad1, write1, ram1, addr1, vr1, vw1⟩ :=
        port_some (by simpa [Machine.dataPort, r] using activePorts.2)
      dsimp only at write1 ram1 addr1 vr1 vw1
      have accSmall : s.acc < 2 ^ 32 := by rw [← vw1]; exact fits1.2.2.1
      have argSmall : word / 4 < 2 ^ p.μ := by rw [← addr1]; exact fits1.1
      simp only [Machine.exec, r] at next
      constructor <;> machine_view <;>
        (try simp only [vr0, natBits_apply, b0, b1, bitWord_false, bitWord_true,
          argument_natBits wordSmall]) <;>
        simp [pad0, write0, ram0, addr0, pad1, write1, ram1, addr1, vr1, vw1, next,
          bitWord_false, bitWord_true, 
          bitsWord_natBits pcSmall, bitsWord_natBits accSmall, bitsWord_natBits argSmall,
          natWord_add, natWord_one]
    · -- loadi
      have b0 : word.testBit ((0 : Fin 32) : ℕ) = true := by
        rw [show ((0 : Fin 32) : ℕ) = 0 from rfl, testBit_zero_mod]; simp [r]
      have b1 : word.testBit ((1 : Fin 32) : ℕ) = true := by
        rw [show ((1 : Fin 32) : ℕ) = 1 from rfl, testBit_one_mod]; simp [r]
      have slot1 := port_none rows1 (by simpa [Machine.dataPort, r] using activePorts.2)
      simp only [Machine.exec, r] at next
      constructor <;> machine_view <;>
        (try simp only [vr0, natBits_apply, b0, b1, bitWord_true,
          argument_natBits wordSmall]) <;>
        simp [pad0, write0, ram0, addr0, slot1, next, OpSlot.padSlot, bitWord_false,
          bitWord_true, bitsWord_natBits_zero,
          bitsWord_natBits pcSmall, natWord_add, natWord_one]

end NightstreamFPrime.Lifecycle.Nebula.Honest

namespace NightstreamFPrime.Lifecycle.Nebula.MemoryApp

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Nebula
open scoped NightstreamFPrime.Spec.GoldilocksExtensionRing

variable {p : Plan}

/-- Ob5 for the relation: a step that the model accepts from a reachable carry,
with one machine step on its ports, has witness words that satisfy the memory
program's validity predicate, and the program's step function gives the state
digest of the output state and the output carry. -/
theorem complete (valid : p.Valid) (two : p.bOps = 2) {c c' : MemoryCarry} (reach : Reach p c)
    {s : Machine.State} {inv : Invocation Machine.State Digest}
    (accepts : invoke (context p) c inv = some c') (machine : Machine.Step s inv.ports inv.next) :
    ∃ witness : List F, witness.length = Words.count p ∧
      MemoryApp.valid p two (stateWords (List.ofFn s.words) (carryWords c)) witness ∧
      MemoryApp.step (stateWords (List.ofFn s.words) (carryWords c)) witness =
        stateWords (List.ofFn inv.next.words) (carryWords c') := by
  obtain ⟨accepted, rfl⟩ := Honest.accepted p c inv accepts
  have count := Words.count_eq (p := p)
  refine ⟨(Honest.witness p c inv s).words, StepWitness.length_words _, ?_, ?_⟩
  · unfold MemoryApp.valid
    rw [StepWitness.decode_words]
    exact ⟨Honest.rowsHold s valid reach accepted, Honest.machineRows s accepted two machine⟩
  · unfold MemoryApp.step
    congr 1
    · congr 1
      funext k
      have := k.isLt
      rw [StepWitness.wordOf_words _ (by simp only [Words.appOut]; omega),
        StepWitness.wordAt_appOut]
      rfl
    · show _ = List.ofFn (carryVector (Honest.finished p c inv))
      congr 1
      funext k
      have := k.isLt
      rw [StepWitness.wordOf_words _ (by simp only [Words.carryOut]; omega),
        StepWitness.wordAt_carryOut]
      rfl

end NightstreamFPrime.Lifecycle.Nebula.MemoryApp
