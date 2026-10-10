import NightstreamFPrime.Export.StreamingIdentity

/-!
Owns canonical Goldilocks arithmetic on machine words: addition,
subtraction, limb splitting, wide reduction, and multiplication. Each
operation keeps canonical inputs canonical and denotes the field operation.
Poseidon2 lanes and rounds belong to `NativePoseidon2RoundCore`.
-/

namespace NightstreamFPrime.Export.NativePoseidon2

open NightstreamFPrime.Spec
open NightstreamFPrime.Export
open Fin.CommRing

abbrev Word := UInt64

private def modulus64 : UInt64 := 0xffffffff00000001
def radix : Nat := 4294967296

/-- Interpret one machine word as a Goldilocks residue. -/
def _root_.UInt64.denote (value : UInt64) : F := Poseidon2.ofNat value.toNat

private theorem modulus64_toNat : modulus64.toNat = goldilocksModulus := by
  decide

theorem uint64_bound (value : UInt64) : value.toNat < UInt64.size :=
  value.toBitVec.isLt

@[inline] private def addRaw (a b : UInt64) : UInt64 :=
  if a < modulus64 - b then a + b else a - (modulus64 - b)

private theorem addRaw_toNat (a b : UInt64)
    (ha : a.toNat < goldilocksModulus)
    (hb : b.toNat < goldilocksModulus) :
    (addRaw a b).toNat =
      (a.toNat + b.toNat) % goldilocksModulus := by
  unfold addRaw
  have modulus_sub_b_toNat : (modulus64 - b).toNat =
      goldilocksModulus - b.toNat := by
    rw [UInt64.toNat_sub_of_le]
    · rw [modulus64_toNat]
    · exact UInt64.le_iff_toNat_le.2 (by
        rw [modulus64_toNat]
        exact Nat.le_of_lt hb)
  split <;> rename_i branch
  · have sum_lt_modulus : a.toNat + b.toNat < goldilocksModulus := by
      rw [UInt64.lt_iff_toNat_lt, modulus_sub_b_toNat] at branch
      omega
    rw [UInt64.toNat_add,
      Nat.mod_eq_of_lt (Nat.lt_trans sum_lt_modulus
        (by decide : goldilocksModulus < 2 ^ 64))]
    exact (Nat.mod_eq_of_lt sum_lt_modulus).symm
  · have modulus_sub_b_le_a : goldilocksModulus - b.toNat ≤ a.toNat := by
      rw [UInt64.lt_iff_toNat_lt, modulus_sub_b_toNat] at branch
      omega
    rw [UInt64.toNat_sub_of_le]
    · rw [modulus_sub_b_toNat]
      have sum_ge_modulus : goldilocksModulus ≤ a.toNat + b.toNat := by omega
      have sum_lt_twice : a.toNat + b.toNat < 2 * goldilocksModulus := by omega
      rw [Nat.mod_eq_sub_mod sum_ge_modulus,
        Nat.mod_eq_of_lt (by omega :
          a.toNat + b.toNat - goldilocksModulus < goldilocksModulus)]
      omega
    · exact UInt64.le_iff_toNat_le.2 (by
        rw [modulus_sub_b_toNat]
        exact modulus_sub_b_le_a)

private theorem addRaw_canonical (a b : UInt64)
    (ha : a.toNat < goldilocksModulus)
    (hb : b.toNat < goldilocksModulus) :
    (addRaw a b).toNat < goldilocksModulus := by
  rw [addRaw_toNat a b ha hb]
  exact Nat.mod_lt _ (by decide)

@[inline] def add64 (a b : Word) : Word := addRaw a b

theorem add64_canonical (a b : Word)
    (ha : a.toNat < goldilocksModulus)
    (hb : b.toNat < goldilocksModulus) :
    (add64 a b).toNat < goldilocksModulus :=
  addRaw_canonical a b ha hb

@[simp] theorem add64_denote (a b : Word)
    (ha : a.toNat < goldilocksModulus)
    (hb : b.toNat < goldilocksModulus) :
    (add64 a b).denote = a.denote + b.denote := by
  apply Fin.ext
  rw [Fin.val_add]
  change (addRaw a b).toNat % goldilocksModulus =
    (a.toNat % goldilocksModulus + b.toNat % goldilocksModulus) %
      goldilocksModulus
  rw [Nat.mod_eq_of_lt (addRaw_canonical a b ha hb),
    Nat.mod_eq_of_lt ha, Nat.mod_eq_of_lt hb, addRaw_toNat a b ha hb]

@[inline] private def subRaw (a b : UInt64) : UInt64 :=
  if b ≤ a then a - b else modulus64 - (b - a)

private theorem subRaw_toNat (a b : UInt64)
    (ha : a.toNat < goldilocksModulus)
    (hb : b.toNat < goldilocksModulus) :
    (subRaw a b).toNat =
      (goldilocksModulus - b.toNat + a.toNat) % goldilocksModulus := by
  unfold subRaw
  split <;> rename_i branch
  · have b_le_a : b.toNat ≤ a.toNat := UInt64.le_iff_toNat_le.1 branch
    rw [UInt64.toNat_sub_of_le _ _ branch]
    have sum_sub : goldilocksModulus - b.toNat + a.toNat =
        goldilocksModulus + (a.toNat - b.toNat) := by omega
    rw [sum_sub, Nat.add_mod_left, Nat.mod_eq_of_lt]
    omega
  · have a_lt_b : a.toNat < b.toNat := by
      rw [UInt64.le_iff_toNat_le] at branch
      omega
    have a_le_b : a ≤ b := UInt64.le_iff_toNat_le.2 (Nat.le_of_lt a_lt_b)
    have difference_lt_modulus : b.toNat - a.toNat < goldilocksModulus := by omega
    have difference_le_modulus : b - a ≤ modulus64 :=
      UInt64.le_iff_toNat_le.2 (by
        rw [UInt64.toNat_sub_of_le _ _ a_le_b, modulus64_toNat]
        exact Nat.le_of_lt difference_lt_modulus)
    rw [UInt64.toNat_sub_of_le _ _ difference_le_modulus]
    rw [modulus64_toNat, UInt64.toNat_sub_of_le _ _ a_le_b]
    have value_eq : goldilocksModulus - b.toNat + a.toNat =
        goldilocksModulus - (b.toNat - a.toNat) := by omega
    rw [value_eq, Nat.mod_eq_of_lt]
    omega

private theorem subRaw_canonical (a b : UInt64)
    (ha : a.toNat < goldilocksModulus)
    (hb : b.toNat < goldilocksModulus) :
    (subRaw a b).toNat < goldilocksModulus := by
  rw [subRaw_toNat a b ha hb]
  exact Nat.mod_lt _ (by decide)

@[inline] def sub64 (a b : Word) : Word := subRaw a b

theorem sub64_canonical (a b : Word)
    (ha : a.toNat < goldilocksModulus)
    (hb : b.toNat < goldilocksModulus) :
    (sub64 a b).toNat < goldilocksModulus :=
  subRaw_canonical a b ha hb

@[simp] theorem sub64_denote (a b : Word)
    (ha : a.toNat < goldilocksModulus)
    (hb : b.toNat < goldilocksModulus) :
    (sub64 a b).denote = a.denote - b.denote := by
  apply Fin.ext
  rw [Fin.val_sub]
  change (subRaw a b).toNat % goldilocksModulus =
    (goldilocksModulus - b.toNat % goldilocksModulus +
      a.toNat % goldilocksModulus) % goldilocksModulus
  rw [Nat.mod_eq_of_lt (subRaw_canonical a b ha hb),
    Nat.mod_eq_of_lt ha, Nat.mod_eq_of_lt hb, subRaw_toNat a b ha hb]

@[inline] def low64 (value : UInt64) : UInt64 := value.toUInt32.toUInt64
@[inline] def high64 (value : UInt64) : UInt64 := value >>> 32

theorem low64_toNat (value : UInt64) :
    (low64 value).toNat = value.toNat % radix := by
  simp [low64, radix]

theorem high64_toNat (value : UInt64) :
    (high64 value).toNat = value.toNat / radix := by
  simp [high64, radix, Nat.shiftRight_eq_div_pow]

theorem low64_bound (value : UInt64) :
    (low64 value).toNat < radix := by
  rw [low64_toNat]
  exact Nat.mod_lt _ (by decide)

theorem high64_bound (value : UInt64) :
    (high64 value).toNat < radix := by
  rw [high64_toNat]
  apply (Nat.div_lt_iff_lt_mul (by decide : 0 < radix)).2
  have bound := uint64_bound value
  norm_num [UInt64.size, radix] at bound ⊢
  exact bound

private theorem decompose64_toNat (value : UInt64) :
    (low64 value).toNat + radix * (high64 value).toNat = value.toNat := by
  rw [low64_toNat, high64_toNat]
  exact Nat.mod_add_div value.toNat radix

@[inline] private def shiftLimb64 (value : UInt64) : UInt64 := value <<< 32

private theorem shiftLimb64_toNat (value : UInt64)
    (bound : value.toNat < radix) :
    (shiftLimb64 value).toNat = radix * value.toNat := by
  simp only [shiftLimb64, UInt64.toNat_shiftLeft, UInt64.reduceToNat,
    Nat.reduceMod, Nat.shiftLeft_eq]
  norm_num [radix]
  rw [Nat.mod_eq_of_lt]
  · omega
  · calc
      value.toNat * 4294967296 ≤
          (4294967296 - 1) * 4294967296 :=
        Nat.mul_le_mul_right 4294967296 (by
          have concreteBound : value.toNat < 4294967296 := by
            simpa [radix] using bound
          omega)
      _ < 2 ^ 64 := by decide

private theorem shiftLimb64_canonical (value : UInt64)
    (bound : value.toNat < radix) :
    (shiftLimb64 value).toNat < goldilocksModulus := by
  rw [shiftLimb64_toNat value bound]
  calc
    radix * value.toNat ≤ radix * (radix - 1) :=
      Nat.mul_le_mul_left radix (by
        have concreteBound := bound
        omega)
    _ < goldilocksModulus := by decide

private theorem shiftLimb64_denote (value : UInt64)
    (bound : value.toNat < radix) :
    (shiftLimb64 value).denote =
      Poseidon2.ofNat radix * value.denote := by
  apply Fin.ext
  rw [Fin.val_mul]
  simp only [UInt64.denote, Poseidon2.ofNat]
  rw [Nat.mod_eq_of_lt (shiftLimb64_canonical value bound),
    Nat.mod_eq_of_lt (by decide : radix < goldilocksModulus),
    Nat.mod_eq_of_lt (Nat.lt_trans bound (by decide)),
    shiftLimb64_toNat value bound, Nat.mod_eq_of_lt]
  calc
    radix * value.toNat ≤ radix * (radix - 1) :=
      Nat.mul_le_mul_left radix (by
        have concreteBound := bound
        omega)
    _ < goldilocksModulus := by decide

private def limbProduct64 (a b : UInt64) : UInt64 := a * b

private theorem limbProduct64_toNat (a b : UInt64)
    (ha : a.toNat < radix) (hb : b.toNat < radix) :
    (limbProduct64 a b).toNat = a.toNat * b.toNat := by
  unfold limbProduct64
  rw [UInt64.toNat_mul, Nat.mod_eq_of_lt]
  have aBound := ha
  have bBound := hb
  calc
    a.toNat * b.toNat ≤ (radix - 1) * (radix - 1) :=
      Nat.mul_le_mul (by omega) (by omega)
    _ < 2 ^ 64 := by decide

private theorem limbProduct64_canonical (a b : UInt64)
    (ha : a.toNat < radix) (hb : b.toNat < radix) :
    (limbProduct64 a b).toNat < goldilocksModulus := by
  rw [limbProduct64_toNat a b ha hb]
  have aBound := ha
  have bBound := hb
  calc
    a.toNat * b.toNat ≤ (radix - 1) * (radix - 1) :=
      Nat.mul_le_mul (by omega) (by omega)
    _ < goldilocksModulus := by decide

private theorem limbProduct64_denote (a b : UInt64)
    (ha : a.toNat < radix) (hb : b.toNat < radix) :
    (limbProduct64 a b).denote = a.denote * b.denote := by
  apply Fin.ext
  rw [Fin.val_mul]
  simp only [UInt64.denote, Poseidon2.ofNat]
  rw [Nat.mod_eq_of_lt (limbProduct64_canonical a b ha hb),
    Nat.mod_eq_of_lt (Nat.lt_trans ha (by decide)),
    Nat.mod_eq_of_lt (Nat.lt_trans hb (by decide)),
    limbProduct64_toNat a b ha hb, Nat.mod_eq_of_lt]
  have aBound := ha
  have bBound := hb
  calc
    a.toNat * b.toNat ≤ (radix - 1) * (radix - 1) :=
      Nat.mul_le_mul (by omega) (by omega)
    _ < goldilocksModulus := by decide

private theorem denote_decompose64 (value : UInt64) :
    value.denote = (low64 value).denote +
      Poseidon2.ofNat radix * (high64 value).denote := by
  apply Fin.ext
  rw [Fin.val_add, Fin.val_mul]
  simp only [UInt64.denote, Poseidon2.ofNat]
  rw [Nat.mod_eq_of_lt (Nat.lt_trans (low64_bound value) (by decide)),
    Nat.mod_eq_of_lt (by decide : radix < goldilocksModulus),
    Nat.mod_eq_of_lt (Nat.lt_trans (high64_bound value) (by decide))]
  have highProduct : radix * (high64 value).toNat < goldilocksModulus := by
    calc
      radix * (high64 value).toNat ≤ radix * (radix - 1) :=
        Nat.mul_le_mul_left radix (by
          have bound := high64_bound value
          omega)
      _ < goldilocksModulus := by decide
  rw [Nat.mod_eq_of_lt highProduct, decompose64_toNat]

private theorem radix_square :
    Poseidon2.ofNat radix * Poseidon2.ofNat radix =
      Poseidon2.ofNat radix - 1 := by
  decide

private theorem radix_cube :
    (Poseidon2.ofNat radix * Poseidon2.ofNat radix) * Poseidon2.ofNat radix =
      Poseidon2.ofNat radix * Poseidon2.ofNat radix - Poseidon2.ofNat radix := by
  calc
    (Poseidon2.ofNat radix * Poseidon2.ofNat radix) * Poseidon2.ofNat radix =
        (Poseidon2.ofNat radix - 1) * Poseidon2.ofNat radix := by rw [radix_square]
    _ = Poseidon2.ofNat radix * Poseidon2.ofNat radix -
        Poseidon2.ofNat radix := by ring

private theorem radix_sub_one :
    Poseidon2.ofNat radix - 1 = Poseidon2.ofNat (radix - 1) := by
  decide

private theorem rawAdd64_toNat (a b : UInt64)
    (bound : a.toNat + b.toNat < 2 ^ 64) :
    (a + b).toNat = a.toNat + b.toNat := by
  rw [UInt64.toNat_add, Nat.mod_eq_of_lt bound]

private theorem rawAdd64_denote (a b : UInt64)
    (bound : a.toNat + b.toNat < 2 ^ 64) :
    (a + b).denote = a.denote + b.denote := by
  apply Fin.ext
  rw [Fin.val_add]
  simp only [UInt64.denote, Poseidon2.ofNat]
  rw [rawAdd64_toNat a b bound, Nat.add_mod]

@[inline] private def canonicalize64 (value : UInt64) : UInt64 :=
  if value < modulus64 then value else value - modulus64

private theorem canonicalize64_toNat (value : UInt64) :
    (canonicalize64 value).toNat = value.toNat % goldilocksModulus := by
  simp only [canonicalize64]
  split
  next isLt =>
    rw [UInt64.lt_iff_toNat_lt, modulus64_toNat] at isLt
    rw [Nat.mod_eq_of_lt isLt]
  next isNotLt =>
    rw [UInt64.lt_iff_toNat_lt, modulus64_toNat] at isNotLt
    have modulusLe : modulus64 ≤ value := UInt64.le_iff_toNat_le.2 (by
      rw [modulus64_toNat]
      omega)
    rw [UInt64.toNat_sub_of_le _ _ modulusLe, modulus64_toNat]
    have valueBound := uint64_bound value
    have differenceBound : value.toNat - goldilocksModulus < goldilocksModulus := by
      norm_num [UInt64.size, goldilocksModulus] at valueBound ⊢
      omega
    rw [Nat.mod_eq_sub_mod (by omega), Nat.mod_eq_of_lt differenceBound]

private theorem canonicalize64_canonical (value : UInt64) :
    (canonicalize64 value).toNat < goldilocksModulus := by
  rw [canonicalize64_toNat]
  exact Nat.mod_lt _ (by decide)

private theorem canonicalize64_denote (value : UInt64) :
    (canonicalize64 value).denote = value.denote := by
  apply Fin.ext
  simp only [UInt64.denote, Poseidon2.ofNat]
  rw [canonicalize64_toNat, Nat.mod_mod]

@[inline] private def mulEpsilonLimb64 (value : UInt64) : UInt64 :=
  shiftLimb64 value - value

private theorem mulEpsilonLimb64_toNat (value : UInt64)
    (bound : value.toNat < radix) :
    (mulEpsilonLimb64 value).toNat = (radix - 1) * value.toNat := by
  have valueLeShift : value ≤ shiftLimb64 value := UInt64.le_iff_toNat_le.2 (by
    rw [shiftLimb64_toNat value bound]
    simp only [radix]
    omega)
  simp only [mulEpsilonLimb64]
  rw [UInt64.toNat_sub_of_le _ _ valueLeShift, shiftLimb64_toNat value bound]
  simp only [radix]
  omega

private theorem mulEpsilonLimb64_canonical (value : UInt64)
    (bound : value.toNat < radix) :
    (mulEpsilonLimb64 value).toNat < goldilocksModulus := by
  rw [mulEpsilonLimb64_toNat value bound]
  calc
    (radix - 1) * value.toNat ≤ (radix - 1) * (radix - 1) :=
      Nat.mul_le_mul_left _ (by omega)
    _ < goldilocksModulus := by decide

private theorem mulEpsilonLimb64_denote (value : UInt64)
    (bound : value.toNat < radix) :
    (mulEpsilonLimb64 value).denote =
      (Poseidon2.ofNat radix - 1) * value.denote := by
  have productBound : (radix - 1) * value.toNat < goldilocksModulus := by
    calc
      (radix - 1) * value.toNat ≤ (radix - 1) * (radix - 1) :=
        Nat.mul_le_mul_left _ (by omega)
      _ < goldilocksModulus := by decide
  rw [radix_sub_one]
  apply Fin.ext
  rw [Fin.val_mul]
  simp only [UInt64.denote, Poseidon2.ofNat]
  rw [Nat.mod_eq_of_lt (mulEpsilonLimb64_canonical value bound),
    mulEpsilonLimb64_toNat value bound,
    Nat.mod_eq_of_lt (by decide : radix - 1 < goldilocksModulus),
    Nat.mod_eq_of_lt (Nat.lt_trans bound (by decide)),
    Nat.mod_eq_of_lt productBound]

@[inline] private def foldHigh64 (value : UInt64) : UInt64 :=
  sub64 (mulEpsilonLimb64 (low64 value)) (high64 value)

private theorem foldHigh64_canonical (value : UInt64) :
    (foldHigh64 value).toNat < goldilocksModulus := by
  apply sub64_canonical
  · exact mulEpsilonLimb64_canonical _ (low64_bound value)
  · exact Nat.lt_trans (high64_bound value) (by decide)

private theorem foldHigh64_denote (value : UInt64) :
    (foldHigh64 value).denote =
      (Poseidon2.ofNat radix * Poseidon2.ofNat radix) * value.denote := by
  rw [foldHigh64, sub64_denote _ _
    (mulEpsilonLimb64_canonical _ (low64_bound value))
    (Nat.lt_trans (high64_bound value) (by decide)),
    mulEpsilonLimb64_denote _ (low64_bound value),
    denote_decompose64 value]
  calc
    (Poseidon2.ofNat radix - 1) * (low64 value).denote -
        (high64 value).denote =
      (Poseidon2.ofNat radix * Poseidon2.ofNat radix) * (low64 value).denote +
        ((Poseidon2.ofNat radix * Poseidon2.ofNat radix) -
          Poseidon2.ofNat radix) * (high64 value).denote := by
      rw [radix_square]
      ring
    _ = (Poseidon2.ofNat radix * Poseidon2.ofNat radix) *
          (low64 value).denote +
        ((Poseidon2.ofNat radix * Poseidon2.ofNat radix) *
          Poseidon2.ofNat radix) * (high64 value).denote := by
      rw [radix_cube]
    _ = (Poseidon2.ofNat radix * Poseidon2.ofNat radix) *
        ((low64 value).denote +
          Poseidon2.ofNat radix * (high64 value).denote) := by ring

@[inline] def reduceWide64 (low high : UInt64) : UInt64 :=
  add64 (canonicalize64 low) (foldHigh64 high)

theorem reduceWide64_canonical (low high : UInt64) :
    (reduceWide64 low high).toNat < goldilocksModulus :=
  add64_canonical _ _ (canonicalize64_canonical low) (foldHigh64_canonical high)

theorem reduceWide64_denote (low high : UInt64) :
    (reduceWide64 low high).denote = low.denote +
      (Poseidon2.ofNat radix * Poseidon2.ofNat radix) * high.denote := by
  rw [reduceWide64, add64_denote _ _
    (canonicalize64_canonical low) (foldHigh64_canonical high),
    canonicalize64_denote, foldHigh64_denote]

@[inline] def mul64 (a b : Word) : Word :=
  let a0 := low64 a
  let a1 := high64 a
  let b0 := low64 b
  let b1 := high64 b
  let p00 := limbProduct64 a0 b0
  let p10 := limbProduct64 a1 b0
  let t0 := p10 + high64 p00
  let p01 := limbProduct64 a0 b1
  let t1 := p01 + low64 t0
  let low := shiftLimb64 (low64 t1) + low64 p00
  let p11 := limbProduct64 a1 b1
  let highBase := p11 + high64 t0
  let high := highBase + high64 t1
  reduceWide64 low high

theorem mul64_canonical (a b : Word) :
    (mul64 a b).toNat < goldilocksModulus := by
  simp only [mul64]
  exact reduceWide64_canonical _ _

@[simp] theorem mul64_denote (a b : Word)
    (_ha : a.toNat < goldilocksModulus)
    (_hb : b.toNat < goldilocksModulus) :
    (mul64 a b).denote = a.denote * b.denote := by
  let a0 := low64 a
  let a1 := high64 a
  let b0 := low64 b
  let b1 := high64 b
  let p00 := limbProduct64 a0 b0
  let p10 := limbProduct64 a1 b0
  let t0 := p10 + high64 p00
  let p01 := limbProduct64 a0 b1
  let t1 := p01 + low64 t0
  let low := shiftLimb64 (low64 t1) + low64 p00
  let p11 := limbProduct64 a1 b1
  let highBase := p11 + high64 t0
  let high := highBase + high64 t1
  have a0Bound : a0.toNat < radix := low64_bound a
  have a1Bound : a1.toNat < radix := high64_bound a
  have b0Bound : b0.toNat < radix := low64_bound b
  have b1Bound : b1.toNat < radix := high64_bound b
  have productMax (x y : UInt64) (hx : x.toNat < radix) (hy : y.toNat < radix) :
      (limbProduct64 x y).toNat ≤ (radix - 1) * (radix - 1) := by
    rw [limbProduct64_toNat x y hx hy]
    exact Nat.mul_le_mul (by omega) (by omega)
  have p00Max := productMax a0 b0 a0Bound b0Bound
  have p10Max := productMax a1 b0 a1Bound b0Bound
  have p01Max := productMax a0 b1 a0Bound b1Bound
  have p11Max := productMax a1 b1 a1Bound b1Bound
  have t0Bound : p10.toNat + (high64 p00).toNat < 2 ^ 64 := by
    calc
      p10.toNat + (high64 p00).toNat ≤
          (radix - 1) * (radix - 1) + (radix - 1) :=
        Nat.add_le_add p10Max (by have bound := high64_bound p00; omega)
      _ < 2 ^ 64 := by decide
  have t1Bound : p01.toNat + (low64 t0).toNat < 2 ^ 64 := by
    calc
      p01.toNat + (low64 t0).toNat ≤
          (radix - 1) * (radix - 1) + (radix - 1) :=
        Nat.add_le_add p01Max (by have bound := low64_bound t0; omega)
      _ < 2 ^ 64 := by decide
  have lowBound : (shiftLimb64 (low64 t1)).toNat + (low64 p00).toNat < 2 ^ 64 := by
    rw [shiftLimb64_toNat _ (low64_bound t1)]
    calc
      radix * (low64 t1).toNat + (low64 p00).toNat ≤
          radix * (radix - 1) + (radix - 1) :=
        Nat.add_le_add (Nat.mul_le_mul_left _ (by
          have bound := low64_bound t1
          omega)) (by have bound := low64_bound p00; omega)
      _ < 2 ^ 64 := by decide
  have highBaseBound : p11.toNat + (high64 t0).toNat < 2 ^ 64 := by
    calc
      p11.toNat + (high64 t0).toNat ≤
          (radix - 1) * (radix - 1) + (radix - 1) :=
        Nat.add_le_add p11Max (by have bound := high64_bound t0; omega)
      _ < 2 ^ 64 := by decide
  have highBound : highBase.toNat + (high64 t1).toNat < 2 ^ 64 := by
    rw [rawAdd64_toNat _ _ highBaseBound]
    calc
      p11.toNat + (high64 t0).toNat + (high64 t1).toNat ≤
          (radix - 1) * (radix - 1) + (radix - 1) + (radix - 1) :=
        Nat.add_le_add
          (Nat.add_le_add p11Max (by have bound := high64_bound t0; omega))
          (by have bound := high64_bound t1; omega)
      _ < 2 ^ 64 := by decide
  have t0Denote : t0.denote = p10.denote + (high64 p00).denote := by
    exact rawAdd64_denote _ _ t0Bound
  have t1Denote : t1.denote = p01.denote + (low64 t0).denote := by
    exact rawAdd64_denote _ _ t1Bound
  have lowDenote : low.denote =
      Poseidon2.ofNat radix * (low64 t1).denote + (low64 p00).denote := by
    rw [rawAdd64_denote _ _ lowBound,
      shiftLimb64_denote _ (low64_bound t1)]
  have highBaseDenote : highBase.denote = p11.denote + (high64 t0).denote := by
    exact rawAdd64_denote _ _ highBaseBound
  have highDenote : high.denote =
      p11.denote + (high64 t0).denote + (high64 t1).denote := by
    rw [rawAdd64_denote _ _ highBound, highBaseDenote]
  change (reduceWide64 low high).denote = _
  rw [reduceWide64_denote, lowDenote, highDenote]
  calc
      Poseidon2.ofNat radix * (low64 t1).denote + (low64 p00).denote +
          (Poseidon2.ofNat radix * Poseidon2.ofNat radix) *
            (p11.denote + (high64 t0).denote + (high64 t1).denote) =
        (low64 p00).denote +
          Poseidon2.ofNat radix *
            ((low64 t1).denote + Poseidon2.ofNat radix * (high64 t1).denote) +
          (Poseidon2.ofNat radix * Poseidon2.ofNat radix) *
            (p11.denote + (high64 t0).denote) := by ring
      _ = (low64 p00).denote + Poseidon2.ofNat radix * t1.denote +
          (Poseidon2.ofNat radix * Poseidon2.ofNat radix) *
            (p11.denote + (high64 t0).denote) := by rw [← denote_decompose64 t1]
      _ = (low64 p00).denote + Poseidon2.ofNat radix *
            (p01.denote + (low64 t0).denote) +
          (Poseidon2.ofNat radix * Poseidon2.ofNat radix) *
            (p11.denote + (high64 t0).denote) := by rw [t1Denote]
      _ = (low64 p00).denote + Poseidon2.ofNat radix * p01.denote +
          Poseidon2.ofNat radix *
            ((low64 t0).denote + Poseidon2.ofNat radix * (high64 t0).denote) +
          (Poseidon2.ofNat radix * Poseidon2.ofNat radix) * p11.denote := by ring
      _ = (low64 p00).denote + Poseidon2.ofNat radix * p01.denote +
          Poseidon2.ofNat radix * t0.denote +
          (Poseidon2.ofNat radix * Poseidon2.ofNat radix) * p11.denote := by
        rw [← denote_decompose64 t0]
      _ = (low64 p00).denote + Poseidon2.ofNat radix * p01.denote +
          Poseidon2.ofNat radix * (p10.denote + (high64 p00).denote) +
          (Poseidon2.ofNat radix * Poseidon2.ofNat radix) * p11.denote := by
        rw [t0Denote]
      _ = ((low64 p00).denote +
            Poseidon2.ofNat radix * (high64 p00).denote) +
          Poseidon2.ofNat radix * p01.denote + Poseidon2.ofNat radix * p10.denote +
          (Poseidon2.ofNat radix * Poseidon2.ofNat radix) * p11.denote := by ring
      _ = p00.denote + Poseidon2.ofNat radix * p01.denote +
          Poseidon2.ofNat radix * p10.denote +
          (Poseidon2.ofNat radix * Poseidon2.ofNat radix) * p11.denote := by
        rw [← denote_decompose64 p00]
      _ = a0.denote * b0.denote + Poseidon2.ofNat radix * (a0.denote * b1.denote) +
          Poseidon2.ofNat radix * (a1.denote * b0.denote) +
          (Poseidon2.ofNat radix * Poseidon2.ofNat radix) *
            (a1.denote * b1.denote) := by
        rw [limbProduct64_denote _ _ a0Bound b0Bound,
          limbProduct64_denote _ _ a0Bound b1Bound,
          limbProduct64_denote _ _ a1Bound b0Bound,
          limbProduct64_denote _ _ a1Bound b1Bound]
      _ = (a0.denote + Poseidon2.ofNat radix * a1.denote) *
          (b0.denote + Poseidon2.ofNat radix * b1.denote) := by ring
      _ = a.denote * b.denote := by rw [← denote_decompose64 a, ← denote_decompose64 b]

@[inline] def square64 (a : Word) : Word :=
  let a0 := low64 a
  let a1 := high64 a
  let p00 := limbProduct64 a0 a0
  let cross := limbProduct64 a1 a0
  let t0 := cross + high64 p00
  let t1 := cross + low64 t0
  let low := shiftLimb64 (low64 t1) + low64 p00
  let p11 := limbProduct64 a1 a1
  let highBase := p11 + high64 t0
  let high := highBase + high64 t1
  reduceWide64 low high

theorem square64_eq_mul64_self (a : Word) : square64 a = mul64 a a := by
  simp only [square64, mul64, limbProduct64]
  rw [UInt64.mul_comm (low64 a) (high64 a)]

theorem square64_canonical (a : Word) :
    (square64 a).toNat < goldilocksModulus := by
  rw [square64_eq_mul64_self]
  exact mul64_canonical _ _

@[simp] theorem square64_denote (a : Word)
    (canonical : a.toNat < goldilocksModulus) :
    (square64 a).denote = a.denote * a.denote := by
  rw [square64_eq_mul64_self, mul64_denote a a canonical canonical]

end NightstreamFPrime.Export.NativePoseidon2
