import NightstreamFPrime.Lifecycle.Nebula.StepDecode
import NightstreamFPrime.Lifecycle.Nebula.WitnessWords
import NightstreamFPrime.Circuit.Basic

/-! Owns the interface between the memory rows and an application (spec §10).
The memory rows own the operation slots and the carry. An application owns
its state words, its private words, and its rows. Its rows read those words
and the slots through local variables (`AppView.env`): the input words, the
output words, the private words, then each slot in the `Words.op` layout.
The contract is Ob7 (`PortApp.sound`) and the application half of Ob5
(`PortApp.complete`), both over the decoded ports. It does not own any memory
row or any application. -/

namespace NightstreamFPrime.Lifecycle.Nebula

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Nebula
open NightstreamFPrime.Circuit

variable {p : Plan}

/-! ### Slot words -/

/-- Word `i` of an operation slot: the lane bits of spec §6.3, then the O4
difference bits; `0` past the slot. -/
def OpSlotBits.word (slot : OpSlotBits p) (i : ℕ) : F :=
  if i = 0 then slot.pad
  else if i = 1 then slot.isWrite
  else if i = 2 then slot.isRam
  else if h : i - 3 < p.μ then slot.addr ⟨i - 3, h⟩
  else if h : i - 3 - p.μ < 32 then slot.vr ⟨i - 3 - p.μ, h⟩
  else if h : i - 3 - p.μ - 32 < 32 then slot.vw ⟨i - 3 - p.μ - 32, h⟩
  else if h : i - 3 - p.μ - 64 < p.wTs then slot.rt ⟨i - 3 - p.μ - 64, h⟩
  else if h : i - p.opWidth < p.wTs then slot.diff ⟨i - p.opWidth, h⟩
  else 0

/-- Word `i` of a scan slot: the value bits, then the stamp bits. -/
def ScanSlotBits.word (slot : ScanSlotBits p) (i : ℕ) : F :=
  if h : i < 32 then slot.value ⟨i, h⟩
  else if h : i - 32 < p.wTs then slot.stamp ⟨i - 32, h⟩
  else 0

/-! ### Slots of a word range -/

/-- The slot and the offset of word `k` in `count` slots of `width` words that
start at word `start`, or `none` outside them. -/
def slotAt (start width count k : ℕ) : Option (Fin count × ℕ) :=
  if h : start ≤ k ∧ (k - start) / width < count then
    some (⟨(k - start) / width, h.2⟩, (k - start) % width)
  else none

theorem slotAt_slot {start width count j i : ℕ} (hj : j < count) (hi : i < width) :
    slotAt start width count (start + j * width + i) = some (⟨j, hj⟩, i) := by
  have pos : 0 < width := by omega
  have sub : start + j * width + i - start = i + j * width := by omega
  have div : (i + j * width) / width = j := by
    rw [Nat.add_mul_div_right _ _ pos, Nat.div_eq_of_lt hi, Nat.zero_add]
  have mod : (i + j * width) % width = i := by
    rw [Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt hi]
  unfold slotAt
  rw [dif_pos ⟨by omega, by rw [sub, div]; exact hj⟩]
  simp only [sub, div, mod]

theorem slotAt_above {start width count k : ℕ} (pos : 0 < width)
    (above : start + count * width ≤ k) : slotAt start width count k = none := by
  unfold slotAt
  rw [dif_neg]
  rintro ⟨-, inside⟩
  have := (Nat.div_lt_iff_lt_mul pos).mp inside
  omega

/-- Word `k` of slot `j` lies below the end of `count` slots. -/
theorem slot_lt {j count width k : ℕ} (hj : j < count) (hk : k < width) :
    j * width + k < count * width :=
  calc j * width + k < j * width + width := by omega
    _ = (j + 1) * width := by ring
    _ ≤ count * width := Nat.mul_le_mul_right _ hj

theorem slotAt_offset {start width count k : ℕ} {j : Fin count} {i : ℕ} (pos : 0 < width)
    (h : slotAt start width count k = some (j, i)) :
    i < width ∧ k = start + j.val * width + i := by
  unfold slotAt at h
  split at h
  · next inside =>
    cases h
    refine ⟨Nat.mod_lt _ pos, ?_⟩
    have := Nat.div_add_mod (k - start) width
    simp only
    rw [Nat.mul_comm]
    omega
  · cases h

theorem Words.opWords_pos : 0 < Words.opWords p := by simp [Words.opWords, Plan.opWidth]

/-- Word `k` of slot `j` of a word vector is word `Words.op p j k`. -/
theorem OpSlotBits.word_ofWords (v : ℕ → F) (j : ℕ) {k : ℕ} (hk : k < Words.opWords p) :
    (OpSlotBits.ofWords p v j).word k = v (Words.op p j k) := by
  simp only [Words.opWords, Plan.opWidth] at hk
  unfold OpSlotBits.word OpSlotBits.ofWords
  simp only [Words.opPad, Words.opIsWrite, Words.opIsRam, Words.opAddr, Words.opVr, Words.opVw,
    Words.opRt, Words.opDiff, Plan.opWidth]
  split_ifs <;> first | rfl | omega | (congr 2; omega)

/-! ### Slot facts -/

/-- What the memory rows give an application about one operation slot: every
lane word is a bit (O1), a padding slot has a zero tail (O7), and a read keeps
its value (O3). -/
structure OpSlotBits.WellFormed (slot : OpSlotBits p) : Prop where
  bits : ∀ x ∈ slot.lane, IsBit x
  padTail : slot.pad = 1 → ∀ x ∈ slot.lane.tail, x = 0
  readKeeps : (1 - slot.isWrite) * (bitsWord slot.vw - bitsWord slot.vr) = 0

/-! ### The application view -/

/-- The values that an application's rows read. -/
structure AppView (p : Plan) (a : AppShape) where
  appIn : Fin a.width → F
  appOut : Fin a.width → F
  aux : Fin a.aux → F
  ops : Fin p.bOps → OpSlotBits p

namespace AppView

/-- The first local variable of the slots. -/
def slotStart (a : AppShape) : ℕ := 2 * a.width + a.aux

/-- The number of local variables. -/
def count (p : Plan) (a : AppShape) : ℕ := slotStart a + p.bOps * Words.opWords p

variable {a : AppShape} (v : AppView p a)

/-- Local variable `k`; `0` past `count`. -/
def env (k : ℕ) : F :=
  if h : k < a.width then v.appIn ⟨k, h⟩
  else if h : k < 2 * a.width then v.appOut ⟨k - a.width, by omega⟩
  else if h : k < slotStart a then v.aux ⟨k - 2 * a.width, by simp only [slotStart] at h; omega⟩
  else match slotAt (slotStart a) (Words.opWords p) p.bOps k with
    | some (j, i) => (v.ops j).word i
    | none => 0

theorem env_appIn (i : Fin a.width) : v.env i = v.appIn i := by
  simp [env]

theorem env_appOut (i : Fin a.width) : v.env (a.width + i) = v.appOut i := by
  have := i.isLt
  simp (disch := omega) only [env, dif_neg, dif_pos, Nat.add_sub_cancel_left, Fin.eta]

theorem env_aux (i : Fin a.aux) : v.env (2 * a.width + i) = v.aux i := by
  have := i.isLt
  simp (disch := first | omega | (simp only [slotStart]; omega)) only [env, dif_neg, dif_pos,
    Nat.add_sub_cancel_left, Fin.eta]

theorem env_op (j : Fin p.bOps) {k : ℕ} (hk : k < Words.opWords p) :
    v.env (slotStart a + j * Words.opWords p + k) = (v.ops j).word k := by
  simp (disch := first | omega | (simp only [slotStart]; omega)) only [env, dif_neg,
    slotAt_slot j.isLt hk, Fin.eta]

end AppView

/-- The witness word of local variable `k`: the application words in their
order, then the slot words. Past the local variables it is a word past the
witness. -/
def Words.appVar (p : Plan) (a : AppShape) (k : ℕ) : ℕ :=
  if k < AppView.slotStart a then Words.memoryCount p + k
  else match slotAt (AppView.slotStart a) (Words.opWords p) p.bOps k with
    | some (j, i) => Words.op p j i
    | none => Words.count p a + k

/-- The application view of a step witness. -/
def StepWitness.view {a : AppShape} (w : StepWitness p a) : AppView p a :=
  ⟨w.appIn, w.appOut, w.aux, w.ops⟩

/-- On a word vector that is zero past the witness, local variable `k` of the
decoded view is witness word `Words.appVar p a k`. -/
theorem env_appVar {a : AppShape} (v : ℕ → F) (zero : ∀ k, Words.count p a ≤ k → v k = 0) :
    (StepWitness.ofWords p a v).view.env = v ∘ Words.appVar p a := by
  funext k
  simp only [Function.comp_apply, Words.appVar, AppView.env, StepWitness.view,
    StepWitness.ofWords, AppView.slotStart]
  by_cases hw : k < a.width
  · rw [dif_pos hw, if_pos (by omega)]; rfl
  by_cases h2 : k < 2 * a.width
  · rw [dif_neg hw, dif_pos h2, if_pos (by omega)]
    simp only [Words.appOut]; congr 1; omega
  by_cases hs : k < 2 * a.width + a.aux
  · rw [dif_neg hw, dif_neg h2, dif_pos hs, if_pos hs]
    simp only [Words.aux]; congr 1; omega
  rw [dif_neg hw, dif_neg h2, dif_neg hs, if_neg hs]
  rcases slot : slotAt (2 * a.width + a.aux) (Words.opWords p) p.bOps k with _ | ⟨j, i⟩
  · simp only
    exact (zero _ (by omega)).symm
  · simp only
    exact OpSlotBits.word_ofWords v j (slotAt_offset Words.opWords_pos slot).1

/-! ### The interface -/

/-- An application of the memory phase at the circuit level (spec §10): the
model application, its state words, its private words, and its rows over
`AppView.env`. -/
structure PortApp (p : Plan) extends AppShape where
  State : Type
  model : Application p State
  ofWords : (Fin width → F) → State
  words : State → Fin width → F
  rows : List Circuit.Expr
  /-- The state words fit the length word of the §11.1 state digest. -/
  width_lt : width < goldilocksModulus
  /-- Every step uses one port list of width `B_ops`. -/
  portsLength : ∀ {s ports s'}, model.Step s ports s' → ports.length = p.bOps
  /-- Ob7: rows that hold on well-formed slots are one model step on the
  decoded ports. -/
  sound : p.Valid → ∀ v : AppView p ⟨width, aux⟩, (∀ j, (v.ops j).WellFormed) →
    ConstraintsHold v.env rows →
    model.Step (ofWords v.appIn) (List.ofFn fun j => (v.ops j).decode.port) (ofWords v.appOut)
  /-- Ob5 for the application: every model step has private words that
  satisfy the rows, on well-formed slots with its ports. -/
  complete : p.Valid → ∀ {s ports s'} (ops : Fin p.bOps → OpSlotBits p),
    model.Step s ports s' → (∀ j, (ops j).WellFormed) →
    (List.ofFn fun j => (ops j).decode.port) = ports →
    ∃ priv : Fin aux → F,
      ConstraintsHold (AppView.env (⟨words s, words s', priv, ops⟩ : AppView p ⟨width, aux⟩)) rows

end NightstreamFPrime.Lifecycle.Nebula
