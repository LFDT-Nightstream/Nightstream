import NightstreamFPrime.Spec.Compression.Akita.Replay.Readback

/-! The squeezed strings of the pinned Akita code (revision `d50f700`) as a finite type, the
readers of the code on them, and the count of the work check (Akita, ePrint 2026/1983, revision
of 2026-11-23, Section 8.2 `[KECCAK]` and (146); `jolt_transcript::check_grind`).

- `Bytes m` is the type of the strings of `m` bytes, `Bytes.array` is the `ByteArray` of a
  string, and `Bytes.value` is its number with the weight `256^i` at the byte `i`.
- `word_array`, `readBits_array`: `Wire.word` and `readBits` of a string read a window of bits of
  its number.
- `Bytes.digits c K u` are the first `c` windows of `K` bits. `card_digits`: each value of them
  has `2^(8·m − c·K)` strings.
- `card_worked`: for `bits ≤ 64`, exactly `2^(64 − bits)` strings of 8 bytes pass `worked`. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita.Replay

open NightstreamFPrime.Spec.Compression.Akita

/-- The strings of `m` bytes. -/
abbrev Bytes (m : ℕ) : Type := Fin m → Fin 256

/-- `X % 2^b / 2^a % 2^w = X / 2^a % 2^w` for a window inside the low `b` bits. -/
theorem div_mod_window (X : ℕ) {a b w : ℕ} (fits : a + w ≤ b) :
    X % 2 ^ b / 2 ^ a % 2 ^ w = X / 2 ^ a % 2 ^ w := by
  obtain ⟨rest, rfl⟩ := Nat.exists_eq_add_of_le (Nat.le_of_add_right_le fits)
  rw [Nat.pow_add, Nat.mod_mul_right_div_self,
    Nat.mod_mod_of_dvd _ (Nat.pow_dvd_pow 2 (Nat.le_of_add_le_add_left fits))]

namespace Bytes

variable {m : ℕ}

/-- The byte of the code for a byte value. -/
def byte (b : Fin 256) : UInt8 := UInt8.ofNatLT b b.isLt

/-- The `ByteArray` of a string. -/
def array (u : Bytes m) : ByteArray := ⟨Array.ofFn fun i => byte (u i)⟩

/-- The number of a string: the byte `i` has the weight `256^i`. -/
def value (u : Bytes m) : ℕ := (finFunctionFinEquiv u : Fin (256 ^ m))

/-- The byte `i` of a string as a number, and 0 after the end. -/
def byteAt (u : Bytes m) (i : ℕ) : ℕ := if h : i < m then u ⟨i, h⟩ else 0

/-- The first `c` windows of `K` bits of a string. -/
def digits (c K : ℕ) (u : Bytes m) (j : Fin c) : Fin (2 ^ K) :=
  ⟨value u / 2 ^ ((j : ℕ) * K) % 2 ^ K, Nat.mod_lt _ (Nat.two_pow_pos K)⟩

theorem size_array (u : Bytes m) : (array u).size = m := by
  show (Array.ofFn fun i => byte (u i)).size = m
  simp

theorem get!_array (u : Bytes m) (i : ℕ) : ((array u).get! i).toNat = byteAt u i := by
  show ((Array.ofFn fun i => byte (u i))[i]!).toNat = byteAt u i
  unfold byteAt
  split
  · next inside =>
    rw [getElem!_pos _ _ (by simpa using inside), Array.getElem_ofFn]
    rfl
  · next outside =>
    rw [getElem!_neg _ _ (by simpa using outside)]
    rfl

theorem getD_array (u : Bytes m) (i : ℕ) : ((array u)[i]?.getD 0).toNat = byteAt u i := by
  unfold byteAt
  split
  · next inside =>
    rw [getElem?_pos _ _ ((size_array u).symm ▸ inside), Option.getD_some]
    show ((Array.ofFn fun i => byte (u i))[i]'(by simpa using inside)).toNat = _
    rw [Array.getElem_ofFn]
    rfl
  · next outside =>
    rw [getElem?_neg _ _ ((size_array u).symm ▸ outside)]
    rfl

theorem array_injective : Function.Injective (array (m := m)) := fun u v same =>
  funext fun i => Fin.ext (by
    have entry := (get!_array u i).symm.trans
      ((congrArg (fun b : ByteArray => (b.get! i).toNat) same).trans (get!_array v i))
    simpa [byteAt] using entry)

theorem card_bytes (m : ℕ) : Fintype.card (Bytes m) = 256 ^ m := by
  rw [Fintype.card_fun, Fintype.card_fin, Fintype.card_fin]

theorem value_lt (u : Bytes m) : value u < 256 ^ m := (finFunctionFinEquiv u).isLt

theorem value_digit (u : Bytes m) (i : ℕ) : value u / 256 ^ i % 256 = byteAt u i := by
  unfold byteAt
  split
  · next inside =>
    exact congrArg Fin.val (congrFun (finFunctionFinEquiv.symm_apply_apply u) ⟨i, inside⟩)
  · next outside =>
    rw [Nat.div_eq_of_lt ((value_lt u).trans_le
      (Nat.pow_le_pow_right (by norm_num) (Nat.le_of_not_lt outside))), Nat.zero_mod]

theorem foldr_range' (u : Bytes m) : ∀ n first : ℕ,
    (List.range' first n).foldr (fun i acc => acc * 256 + byteAt u i) 0 =
      value u / 256 ^ first % 256 ^ n
  | 0, first => by simp [Nat.mod_one]
  | n + 1, first => by
      rw [List.range'_succ, List.foldr_cons, foldr_range' u n (first + 1), ← value_digit u first,
        pow_succ' 256 n, Nat.mod_mul, pow_succ 256 first, Nat.div_div_eq_div_mul]
      omega

/-- The bytes `first .. first + n` of a string, the first with the least weight, are a window of
its number. -/
theorem foldr_range (u : Bytes m) (n first : ℕ) :
    (List.range n).foldr (fun i acc => acc * 256 + byteAt u (first + i)) 0 =
      value u / 256 ^ first % 256 ^ n := by
  rw [← foldr_range', List.range'_eq_map_range, List.foldr_map]

/-- `Wire.word` of a string at the byte `first` is the window of 64 bits of its number. -/
theorem word_array (u : Bytes m) (first : ℕ) :
    (Wire.word (array u) first).toNat = value u / 256 ^ first % 256 ^ 8 := by
  unfold Wire.word
  rw [(foldWord _ (List.range 8) (by simp)).1]
  simp only [get!_array]
  exact foldr_range u 8 first

/-- `readBits` of a string is a window of bits of its number. -/
theorem readBits_array (u : Bytes m) (offset width : ℕ) :
    readBits (array u) offset width = value u / 2 ^ offset % 2 ^ width := by
  unfold readBits
  simp only [getD_array]
  rw [foldr_range, show (256 : ℕ) = 2 ^ 8 from rfl, ← pow_mul, ← pow_mul,
    div_mod_window _ (by omega), Nat.div_div_eq_div_mul, ← pow_add]
  congr 3
  omega

/-- Each value of the first `c` windows of `K` bits has `2^(8·m − c·K)` strings. -/
theorem card_digits {c K : ℕ} (fits : c * K ≤ 8 * m) (t : Fin c → Fin (2 ^ K)) :
    Nat.card {u : Bytes m // digits c K u = t} = 2 ^ (8 * m - c * K) := by
  have total : 256 ^ m = 2 ^ (8 * m - c * K) * (2 ^ K) ^ c :=
    calc 256 ^ m = 2 ^ (8 * m) := by rw [show (256 : ℕ) = 2 ^ 8 from rfl, ← pow_mul]
      _ = 2 ^ (8 * m - c * K + c * K) := by rw [Nat.sub_add_cancel fits]
      _ = 2 ^ (8 * m - c * K) * (2 ^ K) ^ c := by rw [pow_add, Nat.mul_comm c K, pow_mul]
  obtain ⟨Θ, second⟩ : ∃ Θ : Bytes m ≃ Fin (2 ^ (8 * m - c * K)) × (Fin c → Fin (2 ^ K)),
      ∀ u, (Θ u).2 = digits c K u :=
    ⟨finFunctionFinEquiv.trans ((finCongr total).trans (finProdFinEquiv.symm.trans
      (Equiv.prodCongr (Equiv.refl _) finFunctionFinEquiv.symm))), fun u => funext fun j =>
        Fin.ext (by
          show value u % (2 ^ K) ^ c / (2 ^ K) ^ (j : ℕ) % 2 ^ K =
            value u / 2 ^ ((j : ℕ) * K) % 2 ^ K
          rw [← pow_mul, ← pow_mul, Nat.mul_comm K c, Nat.mul_comm K j]
          exact div_mod_window _ ((Nat.succ_mul j K).symm.trans_le
            (Nat.mul_le_mul_right K j.isLt)))⟩
  exact (Nat.card_congr
    { toFun := fun u => (Θ u.1).1
      invFun := fun h => ⟨Θ.symm (h, t),
        (second _).symm.trans (congrArg Prod.snd (Θ.apply_symm_apply (h, t)))⟩
      left_inv := fun u => Subtype.ext
        ((congrArg Θ.symm (Prod.ext rfl ((second u.1).trans u.2).symm :
          ((Θ u.1).1, t) = Θ u.1)).trans (Θ.symm_apply_apply u.1))
      right_inv := fun h => congrArg Prod.fst (Θ.apply_symm_apply (h, t)) }).trans
    (Nat.card_eq_fintype_card.trans (Fintype.card_fin _))

end Bytes

open Bytes

/-- A string of 8 bytes passes `worked` exactly when the low `bits` bits of its number are 0. -/
theorem worked_iff (u : Bytes 8) (bits : ℕ) :
    worked (array u) bits = true ↔
      digits 1 bits u = fun _ => ⟨0, Nat.two_pow_pos bits⟩ := by
  have word : (Wire.word (array u) 0).toNat = value u := by
    rw [word_array, pow_zero, Nat.div_one, Nat.mod_eq_of_lt (value_lt u)]
  have low : digits 1 bits u = (fun _ => ⟨0, Nat.two_pow_pos bits⟩) ↔
      value u % 2 ^ bits = 0 := by
    constructor
    · intro same
      have entry := congrArg Fin.val (congrFun same 0)
      simpa [digits] using entry
    · intro zero
      funext j
      refine Fin.ext ?_
      simp [digits, zero]
  rw [low, ← word]
  simp [worked]

/-- For `bits ≤ 64`, exactly `2^(64 − bits)` of the `2^64` strings of 8 bytes pass `worked`. -/
theorem card_worked {bits : ℕ} (le : bits ≤ 64) :
    Nat.card {u : Bytes 8 // worked (array u) bits = true} = 2 ^ (64 - bits) :=
  (Nat.card_congr (Equiv.subtypeEquivRight fun u => worked_iff u bits)).trans
    ((card_digits (m := 8) (c := 1) (K := bits) (by omega) _).trans
      (congrArg (2 ^ ·) (by omega)))

end NightstreamFPrime.Spec.Compression.Akita.Replay
