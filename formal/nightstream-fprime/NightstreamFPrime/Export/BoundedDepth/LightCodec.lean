import NightstreamFPrime.Spec.Algebra
import NightstreamFPrime.Spec.AjtaiSetupV1
import Mathlib.Data.Fintype.Pi
import Mathlib.Tactic.Tauto

/-! SPEC.md section 4.5. The light slot and the coefficients of the first sum-check. -/

namespace NightstreamFPrime.Export.BoundedDepth.LightCodec

open Spec

abbrev Byte := Fin 256

def signedByte (x : F) : Option Byte :=
  if h : x.val < 128 then some ⟨x.val, by omega⟩
  else if h : fieldModulus - 128 ≤ x.val then
    some ⟨x.val - (fieldModulus - 256), by have := x.isLt; unfold fieldModulus at *; omega⟩
  else none

def fieldWord (b : Byte) : F :=
  if h : b.val < 128 then ⟨b.val, by unfold fieldModulus; omega⟩
  else ⟨fieldModulus - (256 - b.val), by have := b.isLt; unfold fieldModulus; omega⟩

private theorem signedByte_fieldWord (b : Byte) : signedByte (fieldWord b) = some b := by
  by_cases small : b.val < 128
  · unfold signedByte
    simp only [fieldWord, dite_eq_left small]
  · have hb := b.isLt
    have hq : fieldModulus = 18446744073709551557 := rfl
    have high : fieldModulus - 128 ≤ fieldModulus - (256 - b.val) := by omega
    have large : ¬fieldModulus - (256 - b.val) < 128 := by omega
    unfold signedByte
    simp only [fieldWord, dite_eq_right small, dite_eq_right large, dite_eq_left high, Option.some.injEq]
    apply Fin.ext
    change fieldModulus - (256 - b.val) - (fieldModulus - 256) = b.val
    omega

private theorem fieldWord_signedByte {x : F} {b : Byte} (h : signedByte x = some b) :
    fieldWord b = x := by
  unfold signedByte at h
  split at h
  · have hb := congrArg Fin.val (Option.some.inj h)
    apply Fin.ext
    simp [fieldWord, ← hb, *]
  · split at h
    · have hb : b.val = x.val - (fieldModulus - 256) := (congrArg Fin.val (Option.some.inj h)).symm
      have hq : fieldModulus = 18446744073709551557 := rfl
      have hx := x.isLt
      have large : ¬b.val < 128 := by omega
      apply Fin.ext
      simp only [fieldWord, dite_eq_right large]
      omega
    · contradiction

structure Slot (α : Type) where
  depthCommitment : Vector F (1 + 18 * ringDegree)
  publicWords : Fin (5 * ringDegree) → α
  evaluations : Vector F (2 * 28 + 2 * 5 * ringDegree)
deriving DecidableEq

def zeroSlot : Slot F :=
  ⟨Vector.replicate _ 0, fun _ => 0, Vector.replicate _ 0⟩

inductive SlotWire
  | zero
  | signed (slot : Slot Byte)
  | wide (slot : Slot F)

def SlotWire.tag : SlotWire → String
  | .zero => "NS-ONEINV-FIN--Z"
  | .signed _ => "NS-ONEINV-FIN--S"
  | .wide _ => "NS-ONEINV-FIN--L"

def allSigned (w : Fin (5 * ringDegree) → F) : Prop := ∀ i, (signedByte (w i)).isSome = true

instance (w : Fin (5 * ringDegree) → F) : Decidable (allSigned w) :=
  Fintype.decidableForallFintype

def packSlot (s : Slot F) : SlotWire :=
  if s = zeroSlot then .zero
  else if h : allSigned s.publicWords then
    .signed ⟨s.depthCommitment, fun i => (signedByte (s.publicWords i)).get (h i), s.evaluations⟩
  else .wide s

def unpackSlot : SlotWire → Option (Slot F)
  | .zero => some zeroSlot
  | .signed t =>
    let s : Slot F := ⟨t.depthCommitment, fun i => fieldWord (t.publicWords i), t.evaluations⟩
    if s = zeroSlot then none else some s
  | .wide s =>
    if s = zeroSlot ∨ allSigned s.publicWords then none else some s

theorem unpack_pack_slot (s : Slot F) : unpackSlot (packSlot s) = some s := by
  by_cases hz : s = zeroSlot
  · simp [packSlot, unpackSlot, hz]
  · by_cases h : allSigned s.publicWords
    · have restore : (fun i => fieldWord ((signedByte (s.publicWords i)).get (h i))) = s.publicWords := by
        funext i
        exact fieldWord_signedByte (Option.some_get (h i)).symm
      simp [packSlot, unpackSlot, hz, h, restore]
    · simp [packSlot, unpackSlot, hz, h]

theorem pack_unpacked_slot {w : SlotWire} {s : Slot F} (h : unpackSlot w = some s) :
    packSlot s = w := by
  cases w with
  | zero => cases h; simp [packSlot]
  | signed t =>
    simp only [unpackSlot] at h
    split at h
    · contradiction
    · rename_i hz
      cases h
      have ha : allSigned (fun i => fieldWord (t.publicWords i)) := by
        intro i; simp [signedByte_fieldWord]
      simp [packSlot, hz, ha, signedByte_fieldWord]
  | wide t =>
    simp only [unpackSlot] at h
    split at h
    · contradiction
    · cases h; simp_all [packSlot]

structure Round where
  constant : K
  linear : K
  quadratic : K
  cubic : K
  quartic : K
  quintic : K
  sextic : K
deriving DecidableEq

private def short (d e f g : K) : Bool :=
  decide (d = K.zero ∧ e = K.zero ∧ f = K.zero ∧ g = K.zero)

def roundMask : List Round → Nat
  | [] => 0
  | r :: rs => 2 * roundMask rs + (if short r.cubic r.quartic r.quintic r.sextic then 1 else 0)

def coefficients : List Round → List K
  | [] => []
  | r :: rs => r.constant :: r.linear :: r.quadratic ::
      (if short r.cubic r.quartic r.quintic r.sextic then coefficients rs else r.cubic :: r.quartic :: r.quintic :: r.sextic :: coefficients rs)

structure DefaultRounds where
  count : Nat
  maskBytes : List Nat
  values : List K
deriving DecidableEq

def packRounds (rs : List Round) : DefaultRounds :=
  ⟨rs.length, AjtaiSetupV1.littleEndianBytes 4 (roundMask rs), coefficients rs⟩

private def readLE (bs : List Nat) : Nat := bs.foldr (fun b n => b + 256 * n) 0

private theorem readLE_littleEndian {n x : Nat} (h : x < 256 ^ n) :
    readLE (AjtaiSetupV1.littleEndianBytes n x) = x := by
  induction n generalizing x with
  | zero => simp at h; simp [AjtaiSetupV1.littleEndianBytes, readLE, h]
  | succ n ih =>
    have smaller : x / 256 < 256 ^ n := by
      rw [Nat.div_lt_iff_lt_mul (by decide)]
      simpa only [Nat.pow_succ] using h
    change x % 256 + 256 * readLE (AjtaiSetupV1.littleEndianBytes n (x / 256)) = x
    rw [ih smaller, Nat.mod_add_div]

private theorem roundMask_lt (rs : List Round) : roundMask rs < 2 ^ rs.length := by
  induction rs with
  | nil => decide
  | cons r rs ih => simp only [roundMask, List.length_cons, Nat.pow_succ]; split <;> omega

private def readRounds : Nat → Nat → List K → Option (List Round)
  | 0, mask, words => if mask = 0 ∧ words.isEmpty then some [] else none
  | n + 1, mask, a :: b :: c :: rest =>
    if mask % 2 = 1 then
      (⟨a, b, c, K.zero, K.zero, K.zero, K.zero⟩ :: ·) <$> readRounds n (mask / 2) rest
    else match rest with
      | d :: e :: f :: g :: ds =>
        if short d e f g then none else
        (⟨a, b, c, d, e, f, g⟩ :: ·) <$> readRounds n (mask / 2) ds
      | _ => none
  | _ + 1, _, _ => none

private theorem readRounds_coefficients (rs : List Round) :
    readRounds rs.length (roundMask rs) (coefficients rs) = some rs := by
  induction rs with
  | nil => rfl
  | cons r rs ih =>
    obtain ⟨a, b, c, d, e, f, g⟩ := r
    by_cases h : d = K.zero ∧ e = K.zero ∧ f = K.zero ∧ g = K.zero
    · obtain ⟨rfl, rfl, rfl, rfl⟩ := h
      simp [roundMask, coefficients, readRounds, short, show (2 * roundMask rs + 1) / 2 = roundMask rs by omega, ih]
    · simp [roundMask, coefficients, readRounds, short, h, ih]

def unpackRounds (w : DefaultRounds) : Option (List Round) :=
  if w.count > 30 ∨ w.maskBytes.length ≠ 4 ∨ w.maskBytes.any (· ≥ 256) then none
  else readRounds w.count (readLE w.maskBytes) w.values

private theorem littleEndianBytes_bounded (n x : Nat) :
    (AjtaiSetupV1.littleEndianBytes n x).any (· ≥ 256) = false := by
  induction n generalizing x with
  | zero => rfl
  | succ n ih =>
    simp [AjtaiSetupV1.littleEndianBytes, ih, Nat.mod_lt x (by decide : 0 < 256)]

theorem unpack_pack_rounds (rs : List Round) (bound : rs.length ≤ 30) :
    unpackRounds (packRounds rs) = some rs := by
  have maskBound : roundMask rs < 256 ^ 4 :=
    (roundMask_lt rs).trans_le ((Nat.pow_le_pow_right (by decide : 0 < 2) bound).trans (by decide))
  simp [unpackRounds, packRounds, Nat.not_lt.mpr bound, littleEndianBytes_bounded,
    readLE_littleEndian maskBound, readRounds_coefficients]

private theorem readRounds_image {n m : Nat} {v : List K} {rs : List Round}
    (h : readRounds n m v = some rs) :
    rs.length = n ∧ roundMask rs = m ∧ coefficients rs = v := by
  induction n generalizing m v rs with
  | zero =>
    simp only [readRounds] at h
    split at h
    · rename_i z
      obtain ⟨rfl, hv⟩ := z
      cases h
      simpa [roundMask, coefficients] using (List.isEmpty_iff.mp hv).symm
    · contradiction
  | succ n ih =>
    rcases v with _ | ⟨a, _ | ⟨b, _ | ⟨c, rest⟩⟩⟩ <;> try contradiction
    simp only [readRounds] at h
    split at h
    · rename_i bit
      cases e : readRounds n (m / 2) rest with
      | none => simp [e] at h
      | some tail =>
        simp [e] at h
        subst rs
        obtain ⟨len, mask, words⟩ := ih e
        simp [roundMask, coefficients, short, len, mask, words]
        omega
    · rename_i bit
      rcases rest with _ | ⟨d, _ | ⟨e, _ | ⟨f, _ | ⟨g, ds⟩⟩⟩⟩ <;> try contradiction
      simp only at h
      split at h
      · contradiction
      · rename_i live
        cases decoded : readRounds n (m / 2) ds with
        | none => simp [decoded] at h
        | some tail =>
          simp [decoded] at h
          subst rs
          obtain ⟨len, mask, words⟩ := ih decoded
          simp [roundMask, coefficients, len, mask, words, live]
          omega

private theorem littleEndianBytes_readLE (bs : List Nat) (h : bs.any (· ≥ 256) = false) :
    AjtaiSetupV1.littleEndianBytes bs.length (readLE bs) = bs := by
  induction bs with
  | nil => rfl
  | cons b bs ih =>
    simp only [List.any_cons, Bool.or_eq_false_iff, decide_eq_false_iff_not, Nat.not_le] at h
    change (b + 256 * readLE bs) % 256 ::
      AjtaiSetupV1.littleEndianBytes bs.length ((b + 256 * readLE bs) / 256) = b :: bs
    rw [show (b + 256 * readLE bs) % 256 = b by omega,
      show (b + 256 * readLE bs) / 256 = readLE bs by omega, ih h.2]

theorem pack_unpacked_rounds {w : DefaultRounds} {rs : List Round}
    (h : unpackRounds w = some rs) : packRounds rs = w := by
  obtain ⟨count, bytes, values⟩ := w
  simp only [unpackRounds] at h
  split at h
  · contradiction
  · rename_i bounded
    obtain ⟨len, mask, coeffs⟩ := readRounds_image h
    have four : bytes.length = 4 := by tauto
    have canonical : bytes.any (· ≥ 256) = false := by simpa using (show ¬ bytes.any (· ≥ 256) = true by tauto)
    simp [packRounds, len, mask, coeffs, ← four, littleEndianBytes_readLE bytes canonical]

end NightstreamFPrime.Export.BoundedDepth.LightCodec
