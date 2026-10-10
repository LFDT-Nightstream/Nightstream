import NightstreamFPrime.Lifecycle.Nebula.StepInvoke
import NightstreamFPrime.Lifecycle.Nebula.MachineRefinement

/-! Owns the honest witness of one memory-application invocation: the field
words that an honest prover writes for a reachable carry, an invocation that
the model's `invoke` accepts, and the machine state before the step. It also
owns the inversion of `invoke` into the checks it makes. The row proofs
(Ob5 for the relation) are in `HonestRows` and `HonestStep`. -/

namespace NightstreamFPrime.Lifecycle.Nebula

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Nebula
open scoped NightstreamFPrime.Spec.GoldilocksExtensionRing

/-! ### Field encodings -/

/-- A Boolean as a field bit. -/
def bitWord (b : Bool) : F := if b then 1 else 0

theorem bitWord_isBit (b : Bool) : IsBit (bitWord b) := by
  cases b
  · exact Or.inl rfl
  · exact Or.inr rfl

theorem toBool_bitWord (b : Bool) : toBool (bitWord b) = b := by
  cases b
  · exact toBool_zero
  · exact toBool_one

/-- The low `n` bits of `x`, little-endian. -/
def natBits (n x : ℕ) : Fin n → F := fun k => bitWord (x.testBit k)

theorem ofFn_natBits (n x : ℕ) : List.ofFn (natBits n x) = (bitsLE n x).map bitWord := by
  apply List.ext_getElem (by simp [bitsLE])
  intro k _ _
  simp [natBits, bitsLE]

theorem bitWord_bits {l : List Bool} : ∀ x ∈ l.map bitWord, IsBit x := by
  intro x member
  obtain ⟨b, -, rfl⟩ := List.mem_map.mp member
  exact bitWord_isBit b

theorem map_toBool_bitWord (l : List Bool) : (l.map bitWord).map toBool = l := by
  simp [List.map_map, Function.comp_def, toBool_bitWord]

theorem bitsNat_natBits {n x : ℕ} (small : x < 2 ^ n) : bitsNat (natBits n x) = x :=
  bitsLE_injective (bitsNat_lt _) small (by
    rw [bitsLE_bitsNat, ofFn_natBits, map_toBool_bitWord])

theorem bitsWord_natBits {n x : ℕ} (small : x < 2 ^ n) : bitsWord (natBits n x) = natWord x := by
  have bits : ∀ k, IsBit (natBits n x k) := fun _ => bitWord_isBit _
  rw [bitsWord_eq bits, bitsNat_natBits small]

/-- The field inverse, `0` at `0`. -/
def fieldInverse (x : F) : F := @Inv.inv (ZMod goldilocksModulus) _ x

theorem mul_fieldInverse {x : F} (nonzero : x ≠ 0) : x * fieldInverse x = 1 := by
  letI : Fact (Nat.Prime goldilocksModulus) := ⟨GoldilocksPrime.goldilocks_natPrime⟩
  have key : ∀ y : ZMod goldilocksModulus, y ≠ 0 → y * y⁻¹ = 1 := fun _ h => mul_inv_cancel₀ h
  exact key x nonzero

/-- The bits of a typed operation slot, with O4 difference bits `diff`. -/
def encodeOp (p : Plan) (s : OpSlot) (diff : ℕ) : OpSlotBits p :=
  ⟨bitWord s.pad, bitWord s.isWrite, bitWord s.isRam, natBits p.μ s.addr, natBits 32 s.vr,
    natBits 32 s.vw, natBits p.wTs s.rt, natBits p.wTs diff⟩

/-- The bits of a typed scan slot. -/
def encodeScan (p : Plan) (c : ScanSlot) : ScanSlotBits p :=
  ⟨natBits 32 c.value, natBits p.wTs c.stamp⟩

theorem lane_encodeOp (p : Plan) (s : OpSlot) (diff : ℕ) :
    (encodeOp p s diff).lane = (s.bits p).map bitWord := by
  simp only [OpSlotBits.lane, encodeOp, OpSlot.bits, ofFn_natBits, List.map_append,
    List.map_cons, List.cons_append, List.nil_append]

theorem lane_encodeScan (p : Plan) (c : ScanSlot) :
    (encodeScan p c).lane = (c.bits p).map bitWord := by
  simp only [ScanSlotBits.lane, encodeScan, ScanSlot.bits, ofFn_natBits, List.map_append]

theorem decode_encodeOp {p : Plan} {s : OpSlot} (fits : s.Fits p) (diff : ℕ) :
    (encodeOp p s diff).decode = s := by
  obtain ⟨addr, vr, vw, rt⟩ := fits
  simp only [OpSlotBits.decode, encodeOp, toBool_bitWord, bitsNat_natBits addr,
    bitsNat_natBits vr, bitsNat_natBits vw, bitsNat_natBits rt]

theorem decode_encodeScan {p : Plan} {c : ScanSlot} (fits : c.Fits p) :
    (encodeScan p c).decode = c := by
  obtain ⟨value, stamp⟩ := fits
  simp only [ScanSlotBits.decode, encodeScan, bitsNat_natBits value, bitsNat_natBits stamp]

/-- The two application words of a machine state. -/
def Machine.State.words (s : Machine.State) : Fin 2 → F := ![natWord s.pc, natWord s.acc]

/-! ### The invocation -/

namespace Honest

variable (p : Plan) (c : MemoryCarry) (inv : Invocation Machine.State Digest)

/-- The carry that `step` starts from: the arm's open on a closed carry. -/
def start : MemoryCarry := if c.idx = p.n then openedCarry (context p) c inv.proposal else c

/-- The carry after `advance`. -/
def stepped : MemoryCarry := advance (context p) (start p c inv) inv.records

/-- The carry after the step: closed when `advance` sets `idx = N`. -/
def finished : MemoryCarry :=
  let d := stepped p c inv
  if d.idx = p.n then { d with memRoot := d.proposed.2, segIdx := d.segIdx + 1 } else d

/-- The checks that `close` makes. -/
def CloseChecks (d : MemoryCarry) : Prop :=
  d.seen.ops = d.proposed.1 ∧ d.seen.initial = d.memRoot ∧ d.seen.final = d.proposed.2 ∧
    d.products.initial * d.products.write = d.products.read * d.products.final

/-- What `invoke` checks: the segment bound of an open, the step rows, and the
close checks when the step closes. -/
structure Accepted : Prop where
  segBelow : c.idx = p.n → c.segIdx < p.sMax
  rows : StepRows p c.ts inv.records
  closes : (stepped p c inv).idx = p.n → CloseChecks (stepped p c inv)

theorem start_ts : (start p c inv).ts = c.ts := by
  unfold start
  split <;> rfl

/-- `invoke` succeeds exactly through its checks, and returns `finished`. -/
theorem accepted {c' : MemoryCarry} (accepts : invoke (context p) c inv = some c') :
    Accepted p c inv ∧ c' = finished p c inv := by
  simp only [invoke] at accepts
  rw [show (context p).plan.n = p.n from rfl] at accepts
  have segBelow : c.idx = p.n → c.segIdx < p.sMax := by
    intro opens
    by_contra no
    rw [if_pos opens, Spec.Nebula.openSegment,
      if_neg (show ¬ c.segIdx < (context p).plan.sMax from no)] at accepts
    simp at accepts
  have opened : (if c.idx = p.n then Spec.Nebula.openSegment (context p) c inv.proposal
      else some c) =
      some (start p c inv) := by
    unfold start
    by_cases opens : c.idx = p.n
    · rw [if_pos opens, if_pos opens, Spec.Nebula.openSegment,
        if_pos (show c.segIdx < (context p).plan.sMax from segBelow opens)]
    · rw [if_neg opens, if_neg opens]
  rw [opened, Option.bind_some] at accepts
  have rows : StepRows p c.ts inv.records := by
    by_contra no
    rw [stepSegment, if_neg (by rw [start_ts]; exact no)] at accepts
    simp at accepts
  rw [stepSegment, if_pos (by rw [start_ts]; exact rows), Option.bind_some] at accepts
  change finishStep (context p) (stepped p c inv) = some c' at accepts
  unfold finishStep at accepts
  by_cases closes : (stepped p c inv).idx = p.n
  · rw [if_pos (show (stepped p c inv).idx = (context p).plan.n from closes), closeSegment]
      at accepts
    by_cases checks : CloseChecks (stepped p c inv)
    · rw [if_pos (by unfold CloseChecks at checks; exact checks)] at accepts
      refine ⟨⟨segBelow, rows, fun _ => checks⟩, ?_⟩
      unfold finished
      rw [if_pos closes]
      exact (Option.some.inj accepts).symm
    · rw [if_neg (by unfold CloseChecks at checks; exact checks)] at accepts
      simp at accepts
  · rw [if_neg (show ¬ (stepped p c inv).idx = (context p).plan.n from closes)] at accepts
    refine ⟨⟨segBelow, rows, fun h => absurd h closes⟩, ?_⟩
    unfold finished
    rw [if_neg closes]
    exact (Option.some.inj accepts).symm

/-! ### The witness -/

/-- Operation slot `j` of the invocation's records. -/
def opSlot (j : ℕ) : OpSlot := inv.records.ops.getD j OpSlot.padSlot

/-- IS slot `j` of the invocation's records. -/
def initialSlot (j : ℕ) : ScanSlot := inv.records.initialScan.getD j ⟨0, 0⟩

/-- FS slot `j` of the invocation's records. -/
def finalSlot (j : ℕ) : ScanSlot := inv.records.finalScan.getD j ⟨0, 0⟩

/-- The write stamp of slot `j`: the entry timestamp plus the active slots up
to and including `j` (row O2). -/
def writeStamp (j : ℕ) : ℕ := c.ts + activeLength (inv.records.ops.take (j + 1))

/-- The O4 difference word of slot `j`. -/
def diff (j : ℕ) : ℕ := writeStamp c inv j - (opSlot inv j).rt - 1

/-- The challenges that an open draws (spec §9.3). -/
def fresh : K × K :=
  etaChallenges ⟨planDigest p, c.ts, inv.proposal.1, c.memRoot, inv.proposal.2⟩

/-- The read and write products after the first `k` operation slots (rows O8
and O9). -/
def opsAfter (k : ℕ) : K × K :=
  ((start p c inv).products.read *
      (opsFactors p (start p c inv).eta c.ts (inv.records.ops.take k)).1,
    (start p c inv).products.write *
      (opsFactors p (start p c inv).eta c.ts (inv.records.ops.take k)).2)

/-- The IS and FS products after the first `k` scan slots (rows S2 and S3). -/
def scanAfter (k : ℕ) : K × K :=
  ((start p c inv).products.initial * scanFactor (start p c inv).eta
      ((start p c inv).idx * p.bScan) (inv.records.initialScan.take k),
    (start p c inv).products.final * scanFactor (start p c inv).eta
      ((start p c inv).idx * p.bScan) (inv.records.finalScan.take k))

/-- The honest witness of the invocation from machine state `s`. -/
def witness (s : Machine.State) : StepWitness p where
  appIn := s.words
  carryIn := carryVector c
  carryOut := carryVector (finished p c inv)
  appOut := inv.next.words
  proposal i :=
    if h : i.val < 4 then inv.proposal.1 ⟨i.val, h⟩ else inv.proposal.2 ⟨i.val - 4, by omega⟩
  idle := bitWord (opSlot inv 0).pad
  isOpen := bitWord (decide (c.idx = p.n))
  openInverse := fieldInverse (natWord c.idx - natWord p.n)
  isClose := bitWord (decide ((stepped p c inv).idx = p.n))
  closeInverse := fieldInverse (natWord (stepped p c inv).idx - natWord p.n)
  etaFresh := ![(fresh p c inv).1.c0, (fresh p c inv).1.c1, (fresh p c inv).2.c0,
    (fresh p c inv).2.c1]
  eta1Square := ![((finished p c inv).eta.1 * (finished p c inv).eta.1).c0,
    ((finished p c inv).eta.1 * (finished p c inv).eta.1).c1]
  seenPrev i := carryVector (start p c inv) ⟨23 + i.val, by omega⟩
  idxEff := natWord (start p c inv).idx
  ops j := encodeOp p (opSlot inv j) (diff c inv j)
  initial j := encodeScan p (initialSlot inv j)
  final j := encodeScan p (finalSlot inv j)
  tsBits := natBits p.wTs (stepped p c inv).ts
  segBits := natBits p.segWidth (p.sMax - 1 - c.segIdx)
  opsProducts j := ![(opsAfter p c inv (j + 1)).1.c0, (opsAfter p c inv (j + 1)).1.c1,
    (opsAfter p c inv (j + 1)).2.c0, (opsAfter p c inv (j + 1)).2.c1]
  scanProducts j := ![(scanAfter p c inv (j + 1)).1.c0, (scanAfter p c inv (j + 1)).1.c1,
    (scanAfter p c inv (j + 1)).2.c0, (scanAfter p c inv (j + 1)).2.c1]

end Honest

end NightstreamFPrime.Lifecycle.Nebula
