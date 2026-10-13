import Mathlib.Algebra.CharP.Basic
import Mathlib.Data.List.TakeDrop
import NightstreamFPrime.Spec.Nebula.Records

/-! Owns the packing of spec §9.1 and security note Lemma 1: a lane's bits in
chunks of at most 63, each read as a little-endian natural number below
`2 ^ 63`. It also owns the injectivity of the slot encodings of §6.3. It does
not own the hash chains. -/

namespace NightstreamFPrime.Spec.Nebula

/-- Little-endian value of a chunk of bits. -/
def chunkValue : List Bool → ℕ
  | [] => 0
  | b :: bs => b.toNat + 2 * chunkValue bs

/-- Spec §9.1: consecutive chunks of 63 bits; only the last can be shorter. -/
def pack (bits : List Bool) : List ℕ :=
  if _h : bits = [] then [] else chunkValue (bits.take 63) :: pack (bits.drop 63)
termination_by bits.length
decreasing_by
  have : 0 < bits.length := List.length_pos_of_ne_nil _h
  simp only [List.length_drop]
  omega

theorem pack_nil : pack [] = [] := by
  rw [pack.eq_1, dif_pos rfl]

theorem pack_of_ne_nil {bits : List Bool} (nonempty : bits ≠ []) :
    pack bits = chunkValue (bits.take 63) :: pack (bits.drop 63) := by
  rw [pack.eq_1, dif_neg nonempty]

theorem chunkValue_lt (bs : List Bool) : chunkValue bs < 2 ^ bs.length := by
  induction bs with
  | nil => simp [chunkValue]
  | cons b bs ih =>
    have := Bool.toNat_le b
    simp only [chunkValue, List.length_cons, Nat.pow_succ]
    omega

private theorem chunkValue_injective {xs ys : List Bool} (length : xs.length = ys.length)
    (same : chunkValue xs = chunkValue ys) : xs = ys := by
  induction xs generalizing ys with
  | nil => exact (List.eq_nil_of_length_eq_zero length.symm).symm
  | cons x xs ih =>
    cases ys with
    | nil => simp at length
    | cons y ys =>
      simp only [chunkValue] at same
      have heads : x = y ∧ chunkValue xs = chunkValue ys := by
        cases x <;> cases y <;> simp at same ⊢ <;> omega
      rw [heads.1, ih (by simpa using length) heads.2]

/-- Every packed element is below `2 ^ 63`. -/
theorem pack_lt (bits : List Bool) : ∀ x ∈ pack bits, x < 2 ^ 63 := by
  induction bits using pack.induct with
  | case1 => simp [pack_nil]
  | case2 bits nonempty ih =>
    rw [pack_of_ne_nil nonempty]
    intro x mem
    rcases List.mem_cons.1 mem with rfl | mem
    · exact lt_of_lt_of_le (chunkValue_lt _)
        (Nat.pow_le_pow_right Nat.zero_lt_two (List.length_take_le _ _))
    · exact ih x mem

/-- The number of packed elements of a lane of `n` bits. -/
def packedLength (n : ℕ) : ℕ := (n + 62) / 63

/-- The packing of a lane has `packedLength` elements. -/
theorem pack_length (bits : List Bool) : (pack bits).length = packedLength bits.length := by
  induction bits using pack.induct with
  | case1 => simp [pack_nil, packedLength]
  | case2 bits nonempty ih =>
    have pos : 0 < bits.length := List.length_pos_of_ne_nil nonempty
    rw [pack_of_ne_nil nonempty, List.length_cons, ih, List.length_drop]
    unfold packedLength
    omega

/-- Lemma 1, over the natural numbers: equal packings of equal-length bit
strings come from equal bit strings. -/
theorem pack_injective {xs ys : List Bool} (length : xs.length = ys.length)
    (same : pack xs = pack ys) : xs = ys := by
  induction xs using pack.induct generalizing ys with
  | case1 => exact (List.eq_nil_of_length_eq_zero length.symm).symm
  | case2 xs nonempty ih =>
    have ys_nonempty : ys ≠ [] := by
      rintro rfl
      exact nonempty (List.eq_nil_of_length_eq_zero length)
    rw [pack_of_ne_nil nonempty, pack_of_ne_nil ys_nonempty, List.cons.injEq] at same
    have heads := chunkValue_injective (by simp [length]) same.1
    have tails := ih (by simp [length]) same.2
    rw [← List.take_append_drop 63 xs, ← List.take_append_drop 63 ys, heads, tails]

private theorem map_cast_injective {R : Type} [AddGroupWithOne R] [CharP R goldilocksModulus] :
    ∀ {l₁ l₂ : List ℕ}, (∀ x ∈ l₁, x < 2 ^ 63) → (∀ x ∈ l₂, x < 2 ^ 63) →
      l₁.map (Nat.cast : ℕ → R) = l₂.map Nat.cast → l₁ = l₂
  | [], [], _, _, _ => rfl
  | [], _ :: _, _, _, same => by simp at same
  | _ :: _, [], _, _, same => by simp at same
  | a :: l₁, b :: l₂, small₁, small₂, same => by
    have modulus : 2 ^ 63 < goldilocksModulus := by simp [goldilocksModulus]
    simp only [List.map_cons, List.cons.injEq] at same
    have heads : a = b := CharP.natCast_injOn_Iio R goldilocksModulus
      (lt_trans (small₁ a (by simp)) modulus) (lt_trans (small₂ b (by simp)) modulus) same.1
    rw [heads, map_cast_injective (fun x mem => small₁ x (by simp [mem]))
      (fun x mem => small₂ x (by simp [mem])) same.2]

/-- Lemma 1, over a ring of characteristic `q`: the field values of the
packed elements determine the bit string. -/
theorem pack_cast_injective {R : Type} [AddGroupWithOne R] [CharP R goldilocksModulus]
    {xs ys : List Bool} (length : xs.length = ys.length)
    (same : (pack xs).map (Nat.cast : ℕ → R) = (pack ys).map (Nat.cast : ℕ → R)) :
    xs = ys :=
  pack_injective length (map_cast_injective (pack_lt xs) (pack_lt ys) same)

/-- Every operation-slot encoding has the plan's width. -/
theorem OpSlot.bits_length (p : Plan) (s : OpSlot) : (s.bits p).length = p.opWidth := by
  simp only [OpSlot.bits, bitsLE, Plan.opWidth, List.length_append, List.length_map,
    List.length_range, List.length_cons, List.length_nil]

/-- Every scan-slot encoding has the plan's width. -/
theorem ScanSlot.bits_length (p : Plan) (c : ScanSlot) : (c.bits p).length = p.scanWidth := by
  simp only [ScanSlot.bits, bitsLE, Plan.scanWidth, List.length_append, List.length_map,
    List.length_range]

theorem bitsLE_injective {w x y : ℕ} (hx : x < 2 ^ w) (hy : y < 2 ^ w)
    (same : bitsLE w x = bitsLE w y) : x = y := by
  apply Nat.eq_of_testBit_eq
  intro i
  by_cases low : i < w
  · exact List.map_inj_left.1 same i (List.mem_range.2 low)
  · have wide : 2 ^ w ≤ 2 ^ i := Nat.pow_le_pow_right Nat.zero_lt_two (by omega)
    rw [Nat.testBit_lt_two_pow (lt_of_lt_of_le hx wide),
      Nat.testBit_lt_two_pow (lt_of_lt_of_le hy wide)]

private theorem opSlot_bits_injective {p : Plan} {s t : OpSlot} (fitS : s.Fits p)
    (fitT : t.Fits p) (same : s.bits p = t.bits p) : s = t := by
  obtain ⟨sPad, sWrite, sRam, sAddr, sVr, sVw, sRt⟩ := s
  obtain ⟨tPad, tWrite, tRam, tAddr, tVr, tVw, tRt⟩ := t
  obtain ⟨sAddrFits, sVrFits, sVwFits, sRtFits⟩ := fitS
  obtain ⟨tAddrFits, tVrFits, tVwFits, tRtFits⟩ := fitT
  simp only [OpSlot.bits, List.append_assoc, List.cons_append, List.nil_append,
    List.cons.injEq] at same
  obtain ⟨rfl, rfl, rfl, same⟩ := same
  obtain ⟨addr, same⟩ := List.append_inj same (by simp [bitsLE])
  obtain ⟨vr, same⟩ := List.append_inj same (by simp [bitsLE])
  obtain ⟨vw, rt⟩ := List.append_inj same (by simp [bitsLE])
  obtain rfl : sAddr = tAddr := bitsLE_injective sAddrFits tAddrFits addr
  obtain rfl : sVr = tVr := bitsLE_injective sVrFits tVrFits vr
  obtain rfl : sVw = tVw := bitsLE_injective sVwFits tVwFits vw
  obtain rfl : sRt = tRt := bitsLE_injective sRtFits tRtFits rt
  rfl

private theorem scanSlot_bits_injective {p : Plan} {c d : ScanSlot} (fitC : c.Fits p)
    (fitD : d.Fits p) (same : c.bits p = d.bits p) : c = d := by
  obtain ⟨cValue, cStamp⟩ := c
  obtain ⟨dValue, dStamp⟩ := d
  obtain ⟨cValueFits, cStampFits⟩ := fitC
  obtain ⟨dValueFits, dStampFits⟩ := fitD
  simp only [ScanSlot.bits] at same
  obtain ⟨value, stamp⟩ := List.append_inj same (by simp [bitsLE])
  obtain rfl : cValue = dValue := bitsLE_injective cValueFits dValueFits value
  obtain rfl : cStamp = dStamp := bitsLE_injective cStampFits dStampFits stamp
  rfl

/-- A slot-major lane of fixed-width encodings, injective on slots that fit,
is injective on equal-length slot lists that fit. -/
private theorem flatMap_injective {α : Type} {encode : α → List Bool} {width : ℕ}
    {Fits : α → Prop} (fixed : ∀ x, (encode x).length = width)
    (injective : ∀ {x y}, Fits x → Fits y → encode x = encode y → x = y) :
    ∀ {a b : List α}, (∀ x ∈ a, Fits x) → (∀ y ∈ b, Fits y) → a.length = b.length →
      a.flatMap encode = b.flatMap encode → a = b
  | [], [], _, _, _, _ => rfl
  | [], _ :: _, _, _, length, _ => by simp at length
  | _ :: _, [], _, _, length, _ => by simp at length
  | x :: a, y :: b, fitA, fitB, length, same => by
    simp only [List.flatMap_cons] at same
    obtain ⟨heads, tails⟩ := List.append_inj same (by rw [fixed, fixed])
    rw [injective (fitA x (by simp)) (fitB y (by simp)) heads,
      flatMap_injective fixed injective (fun x mem => fitA x (by simp [mem]))
        (fun y mem => fitB y (by simp [mem])) (by simpa using length) tails]

/-- The §6.3 encoding of operation slots is injective on slots that fit. -/
theorem opsLane_injective {p : Plan} {a b : List OpSlot}
    (fitA : ∀ s ∈ a, s.Fits p) (fitB : ∀ s ∈ b, s.Fits p)
    (length : a.length = b.length) (same : opsLane p a = opsLane p b) : a = b :=
  flatMap_injective (OpSlot.bits_length p) opSlot_bits_injective fitA fitB length same

/-- The §6.3 encoding of scan slots is injective on slots that fit. -/
theorem scanLane_injective {p : Plan} {a b : List ScanSlot}
    (fitA : ∀ c ∈ a, c.Fits p) (fitB : ∀ c ∈ b, c.Fits p)
    (length : a.length = b.length) (same : scanLane p a = scanLane p b) : a = b :=
  flatMap_injective (ScanSlot.bits_length p) scanSlot_bits_injective fitA fitB length same

end NightstreamFPrime.Spec.Nebula
