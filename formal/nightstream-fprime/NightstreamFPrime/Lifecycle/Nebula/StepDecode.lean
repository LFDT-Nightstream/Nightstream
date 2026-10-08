import NightstreamFPrime.Lifecycle.Nebula.StepRows

/-! Owns the decoding of a step witness's field bits into the typed records
of the model: bits to Booleans, bit words to natural values below their
width, slots to `OpSlot` and `ScanSlot`, and field-bit packing to the model's
`pack` (Lemma 1's packing over field words, Ob2). -/

namespace NightstreamFPrime.Lifecycle.Nebula

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Nebula

/-- A field bit as a Boolean. -/
def toBool (x : F) : Bool := decide (x = 1)

theorem natWord_add (a b : ℕ) : natWord (a + b) = natWord a + natWord b := by
  apply Fin.ext
  simp [natWord, Poseidon2.ofNat, Fin.val_add, Nat.add_mod]

theorem natWord_mul (a b : ℕ) : natWord (a * b) = natWord a * natWord b := by
  apply Fin.ext
  simp [natWord, Poseidon2.ofNat, Fin.val_mul, Nat.mul_mod]

theorem natWord_toBool {x : F} (bit : IsBit x) : natWord (toBool x).toNat = x := by
  rcases bit with rfl | rfl <;> rfl

/-- The field value of a list of bits is the natural value of its Booleans. -/
theorem chunkWord_eq : ∀ {bits : List F}, (∀ x ∈ bits, IsBit x) →
    chunkWord bits = natWord (chunkValue (bits.map toBool))
  | [], _ => rfl
  | b :: bs, allBits => by
    have head := natWord_toBool (allBits b (by simp))
    rw [chunkWord, chunkWord_eq (fun x hx => allBits x (by simp [hx])), List.map_cons, chunkValue,
      natWord_add, natWord_mul, head]
    rfl

private theorem chunkValue_testBit :
    ∀ (bs : List Bool) (k : ℕ) (h : k < bs.length), (chunkValue bs).testBit k = bs[k]
  | [], _, h => absurd h (by simp)
  | b :: bs, 0, _ => by cases b <;> simp [chunkValue, Nat.testBit_zero]
  | b :: bs, k + 1, h => by
    have half : (b.toNat + 2 * chunkValue bs) / 2 = chunkValue bs := by
      cases b <;> simp [Nat.add_mul_div_left]
    rw [chunkValue, Nat.testBit_succ, half, chunkValue_testBit bs k (by simpa using h)]
    rfl

/-- The bits of a chunk value are the chunk. -/
theorem bitsLE_chunkValue (bs : List Bool) : bitsLE bs.length (chunkValue bs) = bs := by
  apply List.ext_getElem (by simp [bitsLE])
  intro k h₁ h₂
  simp only [bitsLE, List.getElem_map, List.getElem_range]
  exact chunkValue_testBit bs k h₂

/-- The natural value of a bit vector. -/
def bitsNat {n : ℕ} (bits : Fin n → F) : ℕ := chunkValue ((List.ofFn bits).map toBool)

theorem bitsNat_lt {n : ℕ} (bits : Fin n → F) : bitsNat bits < 2 ^ n := by
  have := chunkValue_lt ((List.ofFn bits).map toBool)
  rwa [List.length_map, List.length_ofFn] at this

theorem bitsWord_eq {n : ℕ} {bits : Fin n → F} (allBits : ∀ k, IsBit (bits k)) :
    bitsWord bits = natWord (bitsNat bits) := by
  apply chunkWord_eq
  intro x member
  obtain ⟨k, rfl⟩ := List.mem_ofFn.mp member
  exact allBits k

theorem bitsLE_bitsNat {n : ℕ} (bits : Fin n → F) :
    bitsLE n (bitsNat bits) = (List.ofFn bits).map toBool := by
  have := bitsLE_chunkValue ((List.ofFn bits).map toBool)
  rwa [List.length_map, List.length_ofFn] at this

/-- A natural value below `q` is its field word's value. -/
theorem natWord_val {n : ℕ} (small : n < goldilocksModulus) : (natWord n).val = n := by
  simp [natWord, Poseidon2.ofNat, Nat.mod_eq_of_lt small]

theorem natWord_injective {a b : ℕ} (ha : a < goldilocksModulus) (hb : b < goldilocksModulus)
    (same : natWord a = natWord b) : a = b := by
  rw [← natWord_val ha, ← natWord_val hb, same]

/-- Spec §9.1 packing of field bits is the model's packing of their Booleans. -/
theorem packWords_eq {bits : List F} (allBits : ∀ x ∈ bits, IsBit x) :
    packWords bits = (pack (bits.map toBool)).map natWord := by
  induction bits using packWords.induct with
  | case1 => simp [packWords, pack_nil]
  | case2 bits nonempty ih =>
    have mapped : bits.map toBool ≠ [] := by simpa using nonempty
    rw [packWords, dif_neg nonempty, pack_of_ne_nil mapped, List.map_cons,
      ih (fun x hx => allBits x (List.mem_of_mem_drop hx)), ← List.map_take, ← List.map_drop]
    congr 1
    exact chunkWord_eq (fun x hx => allBits x (List.mem_of_mem_take hx))

namespace OpSlotBits

variable {p : Plan} (slot : OpSlotBits p)

/-- The typed operation slot of spec §6.1. -/
def decode : OpSlot :=
  ⟨toBool slot.pad, toBool slot.isWrite, toBool slot.isRam, bitsNat slot.addr, bitsNat slot.vr,
    bitsNat slot.vw, bitsNat slot.rt⟩

theorem decode_fits : slot.decode.Fits p :=
  ⟨bitsNat_lt _, bitsNat_lt _, bitsNat_lt _, bitsNat_lt _⟩

/-- The spec §6.3 encoding of the decoded slot is its lane bits. -/
theorem decode_bits : slot.decode.bits p = slot.lane.map toBool := by
  simp only [decode, OpSlot.bits, lane, bitsLE_bitsNat, List.map_append, List.map_cons,
    List.cons_append, List.nil_append]

end OpSlotBits

namespace ScanSlotBits

variable {p : Plan} (slot : ScanSlotBits p)

/-- The typed scan slot of spec §6.2. -/
def decode : ScanSlot := ⟨bitsNat slot.value, bitsNat slot.stamp⟩

theorem decode_fits : slot.decode.Fits p := ⟨bitsNat_lt _, bitsNat_lt _⟩

theorem decode_bits : slot.decode.bits p = slot.lane.map toBool := by
  simp only [decode, ScanSlot.bits, lane, bitsLE_bitsNat, List.map_append]

end ScanSlotBits

namespace StepWitness

variable {p : Plan} (w : StepWitness p)

/-- The typed records of the step. -/
def records : StepRecords :=
  ⟨List.ofFn fun j => (w.ops j).decode, List.ofFn fun j => (w.initial j).decode,
    List.ofFn fun j => (w.final j).decode⟩

theorem records_shaped : w.records.Shaped p where
  opsLength := by simp [records]
  initialLength := by simp [records]
  finalLength := by simp [records]
  opsFit := by
    intro s member
    obtain ⟨j, rfl⟩ := List.mem_ofFn.mp member
    exact (w.ops j).decode_fits
  initialFit := by
    intro c member
    obtain ⟨j, rfl⟩ := List.mem_ofFn.mp member
    exact (w.initial j).decode_fits
  finalFit := by
    intro c member
    obtain ⟨j, rfl⟩ := List.mem_ofFn.mp member
    exact (w.final j).decode_fits

theorem opsLane_records : opsLane p w.records.ops = w.opsLaneBits.map toBool := by
  simp only [opsLane, records, opsLaneBits, List.flatMap, List.map_flatten, List.map_ofFn,
    Function.comp_def, OpSlotBits.decode_bits]

theorem scanLane_ofFn (slots : Fin p.bScan → ScanSlotBits p) :
    scanLane p (List.ofFn fun j => (slots j).decode) = (scanLaneBits slots).map toBool := by
  simp only [scanLane, scanLaneBits, List.flatMap, List.map_flatten, List.map_ofFn,
    Function.comp_def, ScanSlotBits.decode_bits]

end StepWitness

end NightstreamFPrime.Lifecycle.Nebula
