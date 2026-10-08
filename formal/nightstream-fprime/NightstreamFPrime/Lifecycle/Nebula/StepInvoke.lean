import NightstreamFPrime.Lifecycle.Nebula.StepRefinement

/-! Owns the assembly of Ob4 for one invocation: the step rows give the
model's `step` (spec §11.2) on the step-start carry, the close rows give
`close` exactly when the step sets `idx = N`, and so `invoke` maps the decoded
input carry to the decoded output carry, which is reachable again. It does
not own the machine rows or the circuit. -/

namespace NightstreamFPrime.Lifecycle.Nebula

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Nebula
open scoped NightstreamFPrime.Spec.Nebula.GoldilocksFingerprint
open Fin.CommRing

theorem natWord_one : natWord 1 = 1 := rfl

/-- A chain row over field bits is the model's chain hash over their packing. -/
theorem chainLink_eq (lane : Lane) (index : F) (previous : Digest) {bits : List F}
    (allBits : ∀ x ∈ bits, IsBit x) :
    chainLink lane index previous (packWords bits) =
      hash (.chain lane index.val previous (pack (bits.map toBool))) := by
  rw [packWords_eq allBits, chainLink, digestOfBlocks, hash, blocks, natWord_of_val]

namespace StepWitness

variable {p : Plan} (w : StepWitness p)

/-- The carry after the step's `advance`, read from the output words, with
the input carry's segment and memory root. -/
def stepEnd : MemoryCarry where
  segIdx := w.inCarry.segIdx
  idx := (w.cOut 1).val
  ts := (w.cOut 2).val
  eta := (w.eta1, w.eta2)
  products := ⟨kOf (w.cOut 7) (w.cOut 8), kOf (w.cOut 9) (w.cOut 10),
    kOf (w.cOut 11) (w.cOut 12), kOf (w.cOut 13) (w.cOut 14)⟩
  proposed := (carryDigest w.carryOut 15, carryDigest w.carryOut 19)
  seen := ⟨carryDigest w.carryOut 23, carryDigest w.carryOut 27, carryDigest w.carryOut 31⟩
  memRoot := w.inCarry.memRoot

/-- The output carry is the step's end carry with the output segment and root. -/
theorem outCarry_eq : w.outCarry =
    { w.stepEnd with segIdx := (w.cOut 0).val, memRoot := carryDigest w.carryOut 35 } := by
  apply carry_ext
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl
  · exact congrArg₂ Prod.mk (readDigest_eq _ _ _) (readDigest_eq _ _ _)
  · show (⟨readDigest w.carryOut 23 _, readDigest w.carryOut 27 _, readDigest w.carryOut 31 _⟩ :
        Roots Digest) = _
    rw [readDigest_eq, readDigest_eq, readDigest_eq]
    rfl
  · exact readDigest_eq _ _ (by decide)

theorem inTs : w.inCarry.ts = (w.cIn 2).val := by rw [cIn_eq w (by decide)]; rfl

theorem inSeg : w.cIn 0 = natWord w.inCarry.segIdx := by
  rw [cIn_eq w (by decide)]
  exact (natWord_of_val _).symm

theorem inMemRoot : w.inCarry.memRoot = carryDigest w.carryIn 35 := readDigest_eq _ _ (by decide)

theorem opsLaneBits_bits {zIn : List F} (rows : w.RowsHold zIn) :
    ∀ x ∈ w.opsLaneBits, IsBit x := by
  intro x member
  simp only [opsLaneBits, List.mem_flatten, List.mem_ofFn] at member
  obtain ⟨_, ⟨j, rfl⟩, inLane⟩ := member
  exact rows.opsBits j x inLane

theorem scanLaneBits_bits {slots : Fin p.bScan → ScanSlotBits p}
    (bits : ∀ j, ∀ x ∈ (slots j).lane, IsBit x) : ∀ x ∈ scanLaneBits slots, IsBit x := by
  intro x member
  simp only [scanLaneBits, List.mem_flatten, List.mem_ofFn] at member
  obtain ⟨_, ⟨j, rfl⟩, inLane⟩ := member
  exact bits j x inLane

variable {w} {zIn : List F}

theorem inTs_lt (reach : Reach p w.inCarry) : (w.cIn 2).val < 2 ^ p.wTs := by
  rw [← inTs]
  exact reach.ts

/-- Row `idx_out`: the output index is `idx_eff + 1`. -/
theorem RowsHold.idxOut_val (valid : p.Valid) (rows : w.RowsHold zIn)
    (reach : Reach p w.inCarry) : (w.cOut 1).val = w.idxEff.val + 1 := by
  have small : w.idxEff.val + 1 < goldilocksModulus := by
    have := rows.idxEff_lt w valid reach
    have := Plan.n_lt valid
    omega
  rw [rows.idxOut, ← natWord_of_val w.idxEff, ← natWord_one, ← natWord_add, natWord_val small,
    natWord_of_val]

/-- Row §8.4 in values: the output timestamp. -/
theorem RowsHold.tsOut_val (valid : p.Valid) (rows : w.RowsHold zIn)
    (reach : Reach p w.inCarry) :
    (w.cOut 2).val = (w.cIn 2).val + activeCount w.records ∧
      (w.cIn 2).val + activeCount w.records < 2 ^ p.wTs := by
  have boundary := rows.boundary w valid (inTs_lt reach)
  refine ⟨?_, boundary.1⟩
  rw [boundary.2, natWord_val (lt_of_lt_of_le boundary.1
    (le_trans (Plan.wTs_small valid) (two_pow_lt_modulus (by norm_num)).le))]

/-- The rows of spec §8 hold, so `step` runs. -/
theorem RowsHold.stepped (valid : p.Valid) (rows : w.RowsHold zIn) (reach : Reach p w.inCarry) :
    stepSegment (context p) w.stepStart w.records =
      some (advance (context p) w.stepStart w.records) := by
  have holds : StepRows (context p).plan w.stepStart.ts w.records := by
    show StepRows p w.inCarry.ts w.records
    rw [inTs]
    exact rows.stepRows w valid (inTs_lt reach)
  rw [stepSegment, if_pos holds]

/-- Spec §11.2 `step`: the chain rows, the product rows, and the boundary rows
give the model's `advance` on the step-start carry. -/
theorem RowsHold.advanced (valid : p.Valid) (rows : w.RowsHold zIn)
    (reach : Reach p w.inCarry) :
    advance (context p) w.stepStart w.records = w.stepEnd := by
  have tsRange := inTs_lt reach
  have opsLength : w.records.ops.length = p.bOps := by simp [records]
  have initialLength : w.records.initialScan.length = p.bScan := by simp [records]
  have finalLength : w.records.finalScan.length = p.bScan := by simp [records]
  obtain ⟨read, write, initial, final⟩ := rows.productsOut
  apply carry_ext
  · rfl
  · exact (rows.idxOut_val valid reach).symm
  · show w.inCarry.ts + activeCount w.records = (w.cOut 2).val
    rw [inTs, (rows.tsOut_val valid reach).1]
  · rfl
  · show (⟨w.startProduct 7 * (opsFactors p (w.eta1, w.eta2) w.inCarry.ts w.records.ops).1,
        w.startProduct 9 * (opsFactors p (w.eta1, w.eta2) w.inCarry.ts w.records.ops).2,
        w.startProduct 11 * scanFactor (w.eta1, w.eta2) (w.idxEff.val * p.bScan)
          w.records.initialScan,
        w.startProduct 13 * scanFactor (w.eta1, w.eta2) (w.idxEff.val * p.bScan)
          w.records.finalScan⟩ : Products K) =
      ⟨kOf (w.cOut 7) (w.cOut 8), kOf (w.cOut 9) (w.cOut 10), kOf (w.cOut 11) (w.cOut 12),
        kOf (w.cOut 13) (w.cOut 14)⟩
    rw [read, write, initial, final, rows.readProducts w valid p.bOps le_rfl,
      rows.writeProducts w valid tsRange p.bOps le_rfl,
      rows.initialProducts w valid reach p.bScan le_rfl,
      rows.finalProducts w valid reach p.bScan le_rfl, List.take_of_length_le opsLength.le,
      List.take_of_length_le initialLength.le, List.take_of_length_le finalLength.le, inTs]
  · rfl
  · show (⟨hash (.chain .ops w.idxEff.val (w.previousDigest 0) (opsPacked p w.records)),
        hash (.chain .mem w.idxEff.val (w.previousDigest 4) (initialPacked p w.records)),
        hash (.chain .mem w.idxEff.val (w.previousDigest 8) (finalPacked p w.records))⟩ :
          Roots Digest) =
      ⟨carryDigest w.carryOut 23, carryDigest w.carryOut 27, carryDigest w.carryOut 31⟩
    rw [rows.chainOps, rows.chainInitial, rows.chainFinal,
      chainLink_eq _ _ _ (opsLaneBits_bits w rows),
      chainLink_eq _ _ _ (scanLaneBits_bits rows.initialBits),
      chainLink_eq _ _ _ (scanLaneBits_bits rows.finalBits),
      opsPacked, initialPacked, finalPacked, opsLane_records,
      show scanLane p w.records.initialScan = (scanLaneBits w.initial).map toBool from
        scanLane_ofFn w.initial,
      show scanLane p w.records.finalScan = (scanLaneBits w.final).map toBool from
        scanLane_ofFn w.final]
  · rfl

/-- Spec §11.2 `close`, exactly when the step sets `idx = N`: the close rows
give `finishStep`. -/
theorem RowsHold.finished (valid : p.Valid) (rows : w.RowsHold zIn)
    (reach : Reach p w.inCarry) : finishStep (context p) w.stepEnd = some w.outCarry := by
  have segSmall : w.inCarry.segIdx + 1 < goldilocksModulus := by
    have := rows.segBelow w valid reach
    have := Plan.sMax_small valid
    unfold goldilocksModulus
    omega
  rw [finishStep, outCarry_eq]
  rcases rows.closeBit with notClose | close
  · have stays : w.stepEnd.idx ≠ (context p).plan.n := by
      intro closed
      have closed' : (w.cOut 1).val = p.n := closed
      have test := rows.closeTest
      rw [notClose, sub_zero, show w.cOut 1 = natWord p.n by
        rw [← closed']; exact (natWord_of_val _).symm, sub_self, zero_mul] at test
      exact absurd test (by decide)
    rw [if_neg stays]
    congr 1
    apply carry_ext <;> try rfl
    · show w.inCarry.segIdx = (w.cOut 0).val
      rw [rows.segOut, notClose, add_zero, inSeg, natWord_val (by omega)]
    · show w.inCarry.memRoot = carryDigest w.carryOut 35
      rw [inMemRoot]
      funext i
      have row := rows.memOut i
      rw [notClose, zero_mul, zero_add, sub_zero, one_mul] at row
      exact row.symm
  · have closed : w.stepEnd.idx = (context p).plan.n := by
      have zero := rows.closeZero
      rw [close, mul_one, sub_eq_zero] at zero
      show (w.cOut 1).val = p.n
      rw [zero, natWord_val (Plan.n_lt valid)]
    have checks : w.stepEnd.seen.ops = w.stepEnd.proposed.1 ∧
        w.stepEnd.seen.initial = w.stepEnd.memRoot ∧
        w.stepEnd.seen.final = w.stepEnd.proposed.2 ∧
        w.stepEnd.products.initial * w.stepEnd.products.write =
          w.stepEnd.products.read * w.stepEnd.products.final := by
      refine ⟨?_, ?_, ?_, ?_⟩
      · funext i
        have row := rows.closeOps i
        rw [close, one_mul, sub_eq_zero] at row
        exact row
      · show carryDigest w.carryOut 27 = w.inCarry.memRoot
        rw [inMemRoot]
        funext i
        have row := rows.closeInitial i
        rw [close, one_mul, sub_eq_zero] at row
        exact row
      · funext i
        have row := rows.closeFinal i
        rw [close, one_mul, sub_eq_zero] at row
        exact row
      · have row := rows.closeProducts
        have embedOne : embed 1 = (1 : K) := rfl
        rw [close, embedOne] at row
        simp only [← GoldilocksFingerprint.mul_eq, ← GoldilocksFingerprint.sub_eq,
          ← GoldilocksFingerprint.zero_eq, one_mul, sub_eq_zero] at row
        exact row
    rw [if_pos closed, closeSegment, if_pos checks]
    congr 1
    apply carry_ext <;> try rfl
    · show w.inCarry.segIdx + 1 = (w.cOut 0).val
      rw [rows.segOut, close, inSeg, ← natWord_one, ← natWord_add, natWord_val segSmall]
    · show carryDigest w.carryOut 19 = carryDigest w.carryOut 35
      funext i
      have row := rows.memOut i
      rw [close, one_mul, sub_self, zero_mul, add_zero] at row
      exact row.symm

/-- The output carry is reachable again. -/
theorem RowsHold.reachOut (valid : p.Valid) (rows : w.RowsHold zIn)
    (reach : Reach p w.inCarry) : Reach p w.outCarry := by
  have segBelow := rows.segBelow w valid reach
  have segSmall : w.inCarry.segIdx + 1 < goldilocksModulus := by
    have := Plan.sMax_small valid
    unfold goldilocksModulus
    omega
  have idxOut := rows.idxOut_val valid reach
  have idxLt := rows.idxEff_lt w valid reach
  have tsOut := rows.tsOut_val valid reach
  have outSeg : w.outCarry.segIdx = (w.cOut 0).val := by rw [cOut_eq w (by decide)]; rfl
  have outIdx : w.outCarry.idx = (w.cOut 1).val := by rw [cOut_eq w (by decide)]; rfl
  have outTs : w.outCarry.ts = (w.cOut 2).val := by rw [cOut_eq w (by decide)]; rfl
  rcases rows.closeBit with notClose | close
  · have seg : (w.cOut 0).val = w.inCarry.segIdx := by
      rw [rows.segOut, notClose, add_zero, inSeg, natWord_val (by omega)]
    exact ⟨by omega, by omega, fun _ => by omega, by omega⟩
  · have seg : (w.cOut 0).val = w.inCarry.segIdx + 1 := by
      rw [rows.segOut, close, inSeg, ← natWord_one, ← natWord_add, natWord_val segSmall]
    have zero := rows.closeZero
    rw [close, mul_one, sub_eq_zero] at zero
    have closed : (w.cOut 1).val = p.n := by rw [zero, natWord_val (Plan.n_lt valid)]
    exact ⟨by omega, by omega, fun open_ => absurd (outIdx.trans closed) open_, by omega⟩

/-- Ob4 for one invocation: when the rows hold on a reachable input carry, the
model's `invoke` on the decoded records maps the decoded input carry to the
decoded output carry, which is reachable. -/
theorem RowsHold.invoke {σ : Type} (valid : p.Valid) (rows : w.RowsHold zIn)
    (reach : Reach p w.inCarry) (next : σ) :
    Spec.Nebula.invoke (context p) w.inCarry ⟨w.proposals, w.records, next⟩ = some w.outCarry ∧
      Reach p w.outCarry := by
  refine ⟨?_, rows.reachOut valid reach⟩
  show ((if w.inCarry.idx = p.n then Spec.Nebula.openSegment (context p) w.inCarry w.proposals
      else some w.inCarry).bind fun c => stepSegment (context p) c w.records).bind
      (finishStep (context p)) = some w.outCarry
  rw [rows.opened w valid reach, Option.bind_some, rows.stepped valid reach, Option.bind_some,
    rows.advanced valid reach, rows.finished valid reach]

end StepWitness

end NightstreamFPrime.Lifecycle.Nebula
