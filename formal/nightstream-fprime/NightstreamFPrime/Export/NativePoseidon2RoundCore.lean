import NightstreamFPrime.Export.StreamingIdentity

/-!
Owns fixed-width Goldilocks arithmetic and native Poseidon2 round operations.
The exact 4/22/4 schedule, constants, and streaming sponge belong to
`NativePoseidon2`.
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

private theorem uint64_bound (value : UInt64) : value.toNat < UInt64.size :=
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

@[inline] private def reduceWide64 (low high : UInt64) : UInt64 :=
  add64 (canonicalize64 low) (foldHigh64 high)

private theorem reduceWide64_canonical (low high : UInt64) :
    (reduceWide64 low high).toNat < goldilocksModulus :=
  add64_canonical _ _ (canonicalize64_canonical low) (foldHigh64_canonical high)

private theorem reduceWide64_denote (low high : UInt64) :
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
    (ha : a.toNat < goldilocksModulus)
    (hb : b.toNat < goldilocksModulus) :
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

@[inline] private def double64 (value : Word) : Word := add64 value value
@[inline] private def triple64 (value : Word) : Word := add64 (double64 value) value

private theorem double64_canonical (value : Word)
    (canonical : value.toNat < goldilocksModulus) :
    (double64 value).toNat < goldilocksModulus :=
  add64_canonical value value canonical canonical

private theorem triple64_canonical (value : Word)
    (canonical : value.toNat < goldilocksModulus) :
    (triple64 value).toNat < goldilocksModulus :=
  add64_canonical _ value (double64_canonical value canonical) canonical

private theorem double64_denote (value : Word)
    (canonical : value.toNat < goldilocksModulus) :
    (double64 value).denote = 2 * value.denote := by
  rw [double64, add64_denote value value canonical canonical]
  ring

private theorem triple64_denote (value : Word)
    (canonical : value.toNat < goldilocksModulus) :
    (triple64 value).denote = 3 * value.denote := by
  rw [triple64, add64_denote _ value
    (double64_canonical value canonical) canonical,
    double64_denote value canonical]
  ring

@[inline] private def sum4_64 (a b c d : Word) : Word :=
  add64 (add64 (add64 a b) c) d

private theorem sum4_64_canonical (a b c d : Word)
    (ha : a.toNat < goldilocksModulus)
    (hb : b.toNat < goldilocksModulus)
    (hc : c.toNat < goldilocksModulus)
    (hd : d.toNat < goldilocksModulus) :
    (sum4_64 a b c d).toNat < goldilocksModulus :=
  add64_canonical _ d
    (add64_canonical _ c (add64_canonical a b ha hb) hc) hd

private theorem sum4_64_denote (a b c d : Word)
    (ha : a.toNat < goldilocksModulus)
    (hb : b.toNat < goldilocksModulus)
    (hc : c.toNat < goldilocksModulus)
    (hd : d.toNat < goldilocksModulus) :
    (sum4_64 a b c d).denote =
      a.denote + b.denote + c.denote + d.denote := by
  simp only [sum4_64]
  rw [add64_denote _ d
      (add64_canonical _ c (add64_canonical a b ha hb) hc) hd,
    add64_denote _ c (add64_canonical a b ha hb) hc,
    add64_denote a b ha hb]

@[inline] private def combine64 (a b : Word) : Word := add64 (double64 a) b

private theorem combine64_canonical (a b : Word)
    (ha : a.toNat < goldilocksModulus)
    (hb : b.toNat < goldilocksModulus) :
    (combine64 a b).toNat < goldilocksModulus :=
  add64_canonical _ b (double64_canonical a ha) hb

private theorem combine64_denote (a b : Word)
    (ha : a.toNat < goldilocksModulus)
    (hb : b.toNat < goldilocksModulus) :
    (combine64 a b).denote = a.denote + a.denote + b.denote := by
  rw [combine64, add64_denote _ b (double64_canonical a ha) hb,
    double64_denote a ha]
  ring

/-- Eight direct machine-word lanes. `canonical` is erased by compilation. -/
structure State64 where
  x0 : UInt64
  x1 : UInt64
  x2 : UInt64
  x3 : UInt64
  x4 : UInt64
  x5 : UInt64
  x6 : UInt64
  x7 : UInt64
  canonical :
    x0.toNat < goldilocksModulus ∧ x1.toNat < goldilocksModulus ∧
    x2.toNat < goldilocksModulus ∧ x3.toNat < goldilocksModulus ∧
    x4.toNat < goldilocksModulus ∧ x5.toNat < goldilocksModulus ∧
    x6.toNat < goldilocksModulus ∧ x7.toNat < goldilocksModulus

namespace State64

def denote (state : State64) : Poseidon2.State :=
  [state.x0.denote, state.x1.denote, state.x2.denote, state.x3.denote,
   state.x4.denote, state.x5.denote, state.x6.denote, state.x7.denote]

theorem c0 (state : State64) : state.x0.toNat < goldilocksModulus := state.canonical.1
theorem c1 (state : State64) : state.x1.toNat < goldilocksModulus := state.canonical.2.1
theorem c2 (state : State64) : state.x2.toNat < goldilocksModulus := state.canonical.2.2.1
theorem c3 (state : State64) : state.x3.toNat < goldilocksModulus := state.canonical.2.2.2.1
theorem c4 (state : State64) : state.x4.toNat < goldilocksModulus := state.canonical.2.2.2.2.1
theorem c5 (state : State64) : state.x5.toNat < goldilocksModulus := state.canonical.2.2.2.2.2.1
theorem c6 (state : State64) : state.x6.toNat < goldilocksModulus := state.canonical.2.2.2.2.2.2.1
theorem c7 (state : State64) : state.x7.toNat < goldilocksModulus := state.canonical.2.2.2.2.2.2.2

def zero : State64 where
  x0 := 0; x1 := 0; x2 := 0; x3 := 0
  x4 := 0; x5 := 0; x6 := 0; x7 := 0
  canonical := by decide

@[noinline] private def replaceFirstSeven64
    (state : State64)
    (x0 x1 x2 x3 x4 x5 x6 : UInt64)
    (c0 : x0.toNat < goldilocksModulus)
    (c1 : x1.toNat < goldilocksModulus)
    (c2 : x2.toNat < goldilocksModulus)
    (c3 : x3.toNat < goldilocksModulus)
    (c4 : x4.toNat < goldilocksModulus)
    (c5 : x5.toNat < goldilocksModulus)
    (c6 : x6.toNat < goldilocksModulus) : State64 :=
  {
    state with
    x0 := x0
    x1 := x1
    x2 := x2
    x3 := x3
    x4 := x4
    x5 := x5
    x6 := x6
    canonical := ⟨c0, c1, c2, c3, c4, c5, c6, state.c7⟩
  }

@[noinline] private def replaceLast64
    (state : State64) (x7 : UInt64)
    (c7 : x7.toNat < goldilocksModulus) : State64 :=
  {
    state with
    x7 := x7
    canonical := ⟨state.c0, state.c1, state.c2, state.c3,
      state.c4, state.c5, state.c6, c7⟩
  }

@[noinline] private def replaceLanes64
    (state : State64)
    (x0 x1 x2 x3 x4 x5 x6 x7 : UInt64)
    (c0 : x0.toNat < goldilocksModulus)
    (c1 : x1.toNat < goldilocksModulus)
    (c2 : x2.toNat < goldilocksModulus)
    (c3 : x3.toNat < goldilocksModulus)
    (c4 : x4.toNat < goldilocksModulus)
    (c5 : x5.toNat < goldilocksModulus)
    (c6 : x6.toNat < goldilocksModulus)
    (c7 : x7.toNat < goldilocksModulus) : State64 :=
  replaceLast64
    (replaceFirstSeven64 state x0 x1 x2 x3 x4 x5 x6
      c0 c1 c2 c3 c4 c5 c6)
    x7 c7

private theorem replaceLanes64_denote
    (state : State64)
    (x0 x1 x2 x3 x4 x5 x6 x7 : UInt64)
    (c0 : x0.toNat < goldilocksModulus)
    (c1 : x1.toNat < goldilocksModulus)
    (c2 : x2.toNat < goldilocksModulus)
    (c3 : x3.toNat < goldilocksModulus)
    (c4 : x4.toNat < goldilocksModulus)
    (c5 : x5.toNat < goldilocksModulus)
    (c6 : x6.toNat < goldilocksModulus)
    (c7 : x7.toNat < goldilocksModulus) :
    (replaceLanes64 state x0 x1 x2 x3 x4 x5 x6 x7
      c0 c1 c2 c3 c4 c5 c6 c7).denote =
      [x0.denote, x1.denote, x2.denote, x3.denote,
       x4.denote, x5.denote, x6.denote, x7.denote] := by
  simp [replaceLanes64, replaceFirstSeven64, replaceLast64, denote]

@[inline] private def mat0 (a b c d : Word) : Word :=
  sum4_64 (double64 a) (triple64 b) c d
@[inline] private def mat1 (a b c d : Word) : Word :=
  sum4_64 a (double64 b) (triple64 c) d
@[inline] private def mat2 (a b c d : Word) : Word :=
  sum4_64 a b (double64 c) (triple64 d)
@[inline] private def mat3 (a b c d : Word) : Word :=
  sum4_64 (triple64 a) b c (double64 d)

private theorem mat0_canonical (a b c d : Word)
    (ha : a.toNat < goldilocksModulus) (hb : b.toNat < goldilocksModulus)
    (hc : c.toNat < goldilocksModulus) (hd : d.toNat < goldilocksModulus) :
    (mat0 a b c d).toNat < goldilocksModulus :=
  sum4_64_canonical _ _ c d (double64_canonical a ha)
    (triple64_canonical b hb) hc hd

private theorem mat1_canonical (a b c d : Word)
    (ha : a.toNat < goldilocksModulus) (hb : b.toNat < goldilocksModulus)
    (hc : c.toNat < goldilocksModulus) (hd : d.toNat < goldilocksModulus) :
    (mat1 a b c d).toNat < goldilocksModulus :=
  sum4_64_canonical a _ _ d ha (double64_canonical b hb)
    (triple64_canonical c hc) hd

private theorem mat2_canonical (a b c d : Word)
    (ha : a.toNat < goldilocksModulus) (hb : b.toNat < goldilocksModulus)
    (hc : c.toNat < goldilocksModulus) (hd : d.toNat < goldilocksModulus) :
    (mat2 a b c d).toNat < goldilocksModulus :=
  sum4_64_canonical a b _ _ ha hb (double64_canonical c hc)
    (triple64_canonical d hd)

private theorem mat3_canonical (a b c d : Word)
    (ha : a.toNat < goldilocksModulus) (hb : b.toNat < goldilocksModulus)
    (hc : c.toNat < goldilocksModulus) (hd : d.toNat < goldilocksModulus) :
    (mat3 a b c d).toNat < goldilocksModulus :=
  sum4_64_canonical _ b c _ (triple64_canonical a ha) hb hc
    (double64_canonical d hd)

private theorem mat0_denote (a b c d : Word)
    (ha : a.toNat < goldilocksModulus) (hb : b.toNat < goldilocksModulus)
    (hc : c.toNat < goldilocksModulus) (hd : d.toNat < goldilocksModulus) :
    (mat0 a b c d).denote =
      2 * a.denote + 3 * b.denote + c.denote + d.denote := by
  rw [mat0, sum4_64_denote _ _ c d (double64_canonical a ha)
    (triple64_canonical b hb) hc hd, double64_denote a ha,
    triple64_denote b hb]

private theorem mat1_denote (a b c d : Word)
    (ha : a.toNat < goldilocksModulus) (hb : b.toNat < goldilocksModulus)
    (hc : c.toNat < goldilocksModulus) (hd : d.toNat < goldilocksModulus) :
    (mat1 a b c d).denote =
      a.denote + 2 * b.denote + 3 * c.denote + d.denote := by
  rw [mat1, sum4_64_denote a _ _ d ha (double64_canonical b hb)
    (triple64_canonical c hc) hd, double64_denote b hb,
    triple64_denote c hc]

private theorem mat2_denote (a b c d : Word)
    (ha : a.toNat < goldilocksModulus) (hb : b.toNat < goldilocksModulus)
    (hc : c.toNat < goldilocksModulus) (hd : d.toNat < goldilocksModulus) :
    (mat2 a b c d).denote =
      a.denote + b.denote + 2 * c.denote + 3 * d.denote := by
  rw [mat2, sum4_64_denote a b _ _ ha hb (double64_canonical c hc)
    (triple64_canonical d hd), double64_denote c hc,
    triple64_denote d hd]

private theorem mat3_denote (a b c d : Word)
    (ha : a.toNat < goldilocksModulus) (hb : b.toNat < goldilocksModulus)
    (hc : c.toNat < goldilocksModulus) (hd : d.toNat < goldilocksModulus) :
    (mat3 a b c d).denote =
      3 * a.denote + b.denote + c.denote + 2 * d.denote := by
  rw [mat3, sum4_64_denote _ b c _ (triple64_canonical a ha) hb hc
    (double64_canonical d hd), triple64_denote a ha,
    double64_denote d hd]

@[inline] def externalLayer64 (state : State64) : State64 :=
  let m0 := mat0 state.x0 state.x1 state.x2 state.x3
  let m1 := mat1 state.x0 state.x1 state.x2 state.x3
  let m2 := mat2 state.x0 state.x1 state.x2 state.x3
  let m3 := mat3 state.x0 state.x1 state.x2 state.x3
  let m4 := mat0 state.x4 state.x5 state.x6 state.x7
  let m5 := mat1 state.x4 state.x5 state.x6 state.x7
  let m6 := mat2 state.x4 state.x5 state.x6 state.x7
  let m7 := mat3 state.x4 state.x5 state.x6 state.x7
  let c0 := mat0_canonical _ _ _ _ state.c0 state.c1 state.c2 state.c3
  let c1 := mat1_canonical _ _ _ _ state.c0 state.c1 state.c2 state.c3
  let c2 := mat2_canonical _ _ _ _ state.c0 state.c1 state.c2 state.c3
  let c3 := mat3_canonical _ _ _ _ state.c0 state.c1 state.c2 state.c3
  let c4 := mat0_canonical _ _ _ _ state.c4 state.c5 state.c6 state.c7
  let c5 := mat1_canonical _ _ _ _ state.c4 state.c5 state.c6 state.c7
  let c6 := mat2_canonical _ _ _ _ state.c4 state.c5 state.c6 state.c7
  let c7 := mat3_canonical _ _ _ _ state.c4 state.c5 state.c6 state.c7
  replaceLanes64 state
    (combine64 m0 m4) (combine64 m1 m5) (combine64 m2 m6)
    (combine64 m3 m7) (combine64 m4 m0) (combine64 m5 m1)
    (combine64 m6 m2) (combine64 m7 m3)
    (combine64_canonical _ _ c0 c4)
    (combine64_canonical _ _ c1 c5)
    (combine64_canonical _ _ c2 c6)
    (combine64_canonical _ _ c3 c7)
    (combine64_canonical _ _ c4 c0)
    (combine64_canonical _ _ c5 c1)
    (combine64_canonical _ _ c6 c2)
    (combine64_canonical _ _ c7 c3)

theorem externalLayer64_denote (state : State64) :
    (externalLayer64 state).denote = Poseidon2.externalLayer state.denote := by
  have c0 := mat0_canonical _ _ _ _ state.c0 state.c1 state.c2 state.c3
  have c1 := mat1_canonical _ _ _ _ state.c0 state.c1 state.c2 state.c3
  have c2 := mat2_canonical _ _ _ _ state.c0 state.c1 state.c2 state.c3
  have c3 := mat3_canonical _ _ _ _ state.c0 state.c1 state.c2 state.c3
  have c4 := mat0_canonical _ _ _ _ state.c4 state.c5 state.c6 state.c7
  have c5 := mat1_canonical _ _ _ _ state.c4 state.c5 state.c6 state.c7
  have c6 := mat2_canonical _ _ _ _ state.c4 state.c5 state.c6 state.c7
  have c7 := mat3_canonical _ _ _ _ state.c4 state.c5 state.c6 state.c7
  simp only [externalLayer64]
  rw [replaceLanes64_denote]
  simp only [denote]
  rw [combine64_denote _ _ c0 c4, combine64_denote _ _ c1 c5,
    combine64_denote _ _ c2 c6, combine64_denote _ _ c3 c7,
    combine64_denote _ _ c4 c0, combine64_denote _ _ c5 c1,
    combine64_denote _ _ c6 c2, combine64_denote _ _ c7 c3,
    mat0_denote _ _ _ _ state.c0 state.c1 state.c2 state.c3,
    mat1_denote _ _ _ _ state.c0 state.c1 state.c2 state.c3,
    mat2_denote _ _ _ _ state.c0 state.c1 state.c2 state.c3,
    mat3_denote _ _ _ _ state.c0 state.c1 state.c2 state.c3,
    mat0_denote _ _ _ _ state.c4 state.c5 state.c6 state.c7,
    mat1_denote _ _ _ _ state.c4 state.c5 state.c6 state.c7,
    mat2_denote _ _ _ _ state.c4 state.c5 state.c6 state.c7,
    mat3_denote _ _ _ _ state.c4 state.c5 state.c6 state.c7]
  apply List.ext_get
  · simp [Poseidon2.externalLayer, Poseidon2.width, Poseidon2.mat4]
  · intro index leftLt rightLt
    have indexLt : index < 8 := by simpa using leftLt
    interval_cases index <;>
      simp [Poseidon2.externalLayer, Poseidon2.width, Poseidon2.mat4] <;> ring

@[inline] private def sbox64 (value : Word) : Word :=
  let x2 := square64 value
  let x4 := square64 x2
  mul64 (mul64 x4 x2) value

private theorem sbox64_canonical (value : Word) :
    (sbox64 value).toNat < goldilocksModulus := by
  simp only [sbox64]
  exact mul64_canonical _ value

private theorem sbox64_denote (value : Word)
    (canonical : value.toNat < goldilocksModulus) :
    (sbox64 value).denote = Poseidon2.sbox value.denote := by
  let x2 := square64 value
  let x4 := square64 x2
  let x6 := mul64 x4 x2
  have x2Canonical : x2.toNat < goldilocksModulus := square64_canonical _
  have x4Canonical : x4.toNat < goldilocksModulus := square64_canonical _
  have x6Canonical : x6.toNat < goldilocksModulus := mul64_canonical _ _
  change (mul64 x6 value).denote =
    ((value.denote * value.denote) * (value.denote * value.denote)) *
      (value.denote * value.denote) * value.denote
  rw [mul64_denote x6 value x6Canonical canonical,
    mul64_denote x4 x2 x4Canonical x2Canonical,
    square64_denote x2 x2Canonical,
    square64_denote value canonical]

@[noinline] def fullRound64 (state : State64) (constants : @& State64) : State64 :=
  externalLayer64 (replaceLanes64 state
    (sbox64 (add64 state.x0 constants.x0))
    (sbox64 (add64 state.x1 constants.x1))
    (sbox64 (add64 state.x2 constants.x2))
    (sbox64 (add64 state.x3 constants.x3))
    (sbox64 (add64 state.x4 constants.x4))
    (sbox64 (add64 state.x5 constants.x5))
    (sbox64 (add64 state.x6 constants.x6))
    (sbox64 (add64 state.x7 constants.x7))
    (sbox64_canonical _) (sbox64_canonical _) (sbox64_canonical _)
    (sbox64_canonical _) (sbox64_canonical _) (sbox64_canonical _)
    (sbox64_canonical _) (sbox64_canonical _))

theorem fullRound64_denote (constants state : State64) :
    (fullRound64 state constants).denote = Poseidon2.externalLayer [
      Poseidon2.sbox (state.x0.denote + constants.x0.denote),
      Poseidon2.sbox (state.x1.denote + constants.x1.denote),
      Poseidon2.sbox (state.x2.denote + constants.x2.denote),
      Poseidon2.sbox (state.x3.denote + constants.x3.denote),
      Poseidon2.sbox (state.x4.denote + constants.x4.denote),
      Poseidon2.sbox (state.x5.denote + constants.x5.denote),
      Poseidon2.sbox (state.x6.denote + constants.x6.denote),
      Poseidon2.sbox (state.x7.denote + constants.x7.denote)] := by
  rw [fullRound64, externalLayer64_denote]
  rw [replaceLanes64_denote]
  rw [sbox64_denote _ (add64_canonical _ _ state.c0 constants.c0),
    sbox64_denote _ (add64_canonical _ _ state.c1 constants.c1),
    sbox64_denote _ (add64_canonical _ _ state.c2 constants.c2),
    sbox64_denote _ (add64_canonical _ _ state.c3 constants.c3),
    sbox64_denote _ (add64_canonical _ _ state.c4 constants.c4),
    sbox64_denote _ (add64_canonical _ _ state.c5 constants.c5),
    sbox64_denote _ (add64_canonical _ _ state.c6 constants.c6),
    sbox64_denote _ (add64_canonical _ _ state.c7 constants.c7),
    add64_denote _ _ state.c0 constants.c0,
    add64_denote _ _ state.c1 constants.c1,
    add64_denote _ _ state.c2 constants.c2,
    add64_denote _ _ state.c3 constants.c3,
    add64_denote _ _ state.c4 constants.c4,
    add64_denote _ _ state.c5 constants.c5,
    add64_denote _ _ state.c6 constants.c6,
    add64_denote _ _ state.c7 constants.c7]

@[inline] private def sum8_64 (state : State64) : Word :=
  add64 (sum4_64 state.x0 state.x1 state.x2 state.x3)
    (sum4_64 state.x4 state.x5 state.x6 state.x7)

private theorem sum8_64_canonical (state : State64) :
    (sum8_64 state).toNat < goldilocksModulus :=
  add64_canonical _ _
    (sum4_64_canonical _ _ _ _ state.c0 state.c1 state.c2 state.c3)
    (sum4_64_canonical _ _ _ _ state.c4 state.c5 state.c6 state.c7)

private theorem sum8_64_denote (state : State64) :
    (sum8_64 state).denote =
      state.x0.denote + state.x1.denote + state.x2.denote + state.x3.denote +
      state.x4.denote + state.x5.denote + state.x6.denote + state.x7.denote := by
  rw [sum8_64, add64_denote,
    sum4_64_denote _ _ _ _ state.c0 state.c1 state.c2 state.c3,
    sum4_64_denote _ _ _ _ state.c4 state.c5 state.c6 state.c7]
  · ring
  · exact sum4_64_canonical _ _ _ _ state.c0 state.c1 state.c2 state.c3
  · exact sum4_64_canonical _ _ _ _ state.c4 state.c5 state.c6 state.c7

private abbrev half64 : UInt64 := 0x7fffffff80000001
private theorem half64_toNat : half64.toNat = 9223372034707292161 := by decide
private theorem half64_canonical : half64.toNat < goldilocksModulus := by decide

@[inline] private def mulHalf64 (value : Word) : Word :=
  let quotient := value >>> 1
  if value &&& 1 = 0 then quotient else quotient + half64

private theorem shiftRightOne64_toNat (value : UInt64) :
    (value >>> 1).toNat = value.toNat / 2 := by
  simp [Nat.shiftRight_eq_div_pow]

private theorem lowBit64_toNat (value : UInt64) :
    (value &&& 1).toNat = value.toNat % 2 := by
  simp

private theorem lowBit64_eq_zero_iff (value : UInt64) :
    value &&& 1 = 0 ↔ value.toNat % 2 = 0 := by
  constructor
  · intro equality
    have natural := congrArg UInt64.toNat equality
    simpa [lowBit64_toNat] using natural
  · intro equality
    apply UInt64.toNat_inj.1
    simpa [lowBit64_toNat, equality]

private theorem halfAdd64_toNat (value : UInt64) :
    ((value >>> 1) + half64).toNat = value.toNat / 2 + half64.toNat := by
  rw [UInt64.toNat_add, shiftRightOne64_toNat]
  rw [half64_toNat]
  have sumBound : value.toNat / 2 + 9223372034707292161 < 2 ^ 64 := by
    have valueBound := uint64_bound value
    norm_num [UInt64.size] at valueBound ⊢
    omega
  rw [Nat.mod_eq_of_lt sumBound]

private theorem mulHalf64_toNat (value : Word) :
    (mulHalf64 value).toNat =
      if value.toNat % 2 = 0 then value.toNat / 2
      else value.toNat / 2 + half64.toNat := by
  simp only [mulHalf64]
  split <;> rename_i parity
  · rw [shiftRightOne64_toNat]
    simp [(lowBit64_eq_zero_iff value).mp parity]
  · rw [halfAdd64_toNat]
    have nonzero : value.toNat % 2 ≠ 0 := by
      intro zero
      exact parity ((lowBit64_eq_zero_iff value).mpr zero)
    simp [nonzero]

private theorem mulHalf64_canonical (value : Word)
    (canonical : value.toNat < goldilocksModulus) :
    (mulHalf64 value).toNat < goldilocksModulus := by
  rw [mulHalf64_toNat]
  split <;> rename_i parity
  · exact lt_of_le_of_lt (Nat.div_le_self _ _) canonical
  · have parityOne : value.toNat % 2 = 1 := by
      rcases Nat.mod_two_eq_zero_or_one value.toNat with zero | one
      · exact (parity zero).elim
      · exact one
    have decomposition := Nat.mod_add_div value.toNat 2
    rw [half64_toNat]
    norm_num [goldilocksModulus] at canonical ⊢
    omega

private theorem mulHalf64_denote (value : Word)
    (canonical : value.toNat < goldilocksModulus) :
    (mulHalf64 value).denote = half64.denote * value.denote := by
  apply Fin.ext
  rw [Fin.val_mul]
  simp only [UInt64.denote, Poseidon2.ofNat]
  rw [Nat.mod_eq_of_lt (mulHalf64_canonical value canonical),
    Nat.mod_eq_of_lt half64_canonical, Nat.mod_eq_of_lt canonical,
    mulHalf64_toNat]
  rw [half64_toNat]
  split <;> rename_i parity
  all_goals
    have decomposition := Nat.mod_add_div value.toNat 2
    rcases Nat.mod_two_eq_zero_or_one value.toNat with zero | one
    all_goals norm_num [goldilocksModulus] at canonical ⊢
    all_goals omega

@[inline] private def scale0 (value : Word) : Word := sub64 0 (double64 value)
@[inline] private def scale1 (value : Word) : Word := value
@[inline] private def scale2 (value : Word) : Word := double64 value
@[inline] private def scale3 (value : Word) : Word := mulHalf64 value
@[inline] private def scale4 (value : Word) : Word := triple64 value
@[inline] private def scale5 (value : Word) : Word := sub64 0 (mulHalf64 value)
@[inline] private def scale6 (value : Word) : Word := sub64 0 (triple64 value)
@[inline] private def scale7 (value : Word) : Word :=
  sub64 0 (double64 (double64 value))

private theorem scale0_canonical (value : Word)
    (h : value.toNat < goldilocksModulus) :
    (scale0 value).toNat < goldilocksModulus :=
  sub64_canonical _ _ (by decide) (double64_canonical value h)
private theorem scale1_canonical (value : Word)
    (h : value.toNat < goldilocksModulus) :
    (scale1 value).toNat < goldilocksModulus := h
private theorem scale2_canonical (value : Word)
    (h : value.toNat < goldilocksModulus) :
    (scale2 value).toNat < goldilocksModulus := double64_canonical value h
private theorem scale3_canonical (value : Word)
    (h : value.toNat < goldilocksModulus) :
    (scale3 value).toNat < goldilocksModulus := mulHalf64_canonical value h
private theorem scale4_canonical (value : Word)
    (h : value.toNat < goldilocksModulus) :
    (scale4 value).toNat < goldilocksModulus := triple64_canonical value h
private theorem scale5_canonical (value : Word)
    (h : value.toNat < goldilocksModulus) :
    (scale5 value).toNat < goldilocksModulus :=
  sub64_canonical _ _ (by decide) (mulHalf64_canonical value h)
private theorem scale6_canonical (value : Word)
    (h : value.toNat < goldilocksModulus) :
    (scale6 value).toNat < goldilocksModulus :=
  sub64_canonical _ _ (by decide) (triple64_canonical value h)
private theorem scale7_canonical (value : Word)
    (h : value.toNat < goldilocksModulus) :
    (scale7 value).toNat < goldilocksModulus :=
  sub64_canonical _ _ (by decide)
    (double64_canonical _ (double64_canonical value h))

private theorem scale0_denote (value : Word)
    (h : value.toNat < goldilocksModulus) :
    (scale0 value).denote =
      Poseidon2.ofNat (Poseidon2.internalDiagonal.getD 0 0) * value.denote := by
  rw [scale0, sub64_denote _ _ (by decide) (double64_canonical value h),
    double64_denote value h]
  have coefficient : Poseidon2.ofNat (Poseidon2.internalDiagonal.getD 0 0) =
      -2 := by decide
  have zeroDenote : (0 : UInt64).denote = (0 : F) := by decide
  rw [coefficient, zeroDenote]
  ring

private theorem scale1_denote (value : Word) :
    (scale1 value).denote =
      Poseidon2.ofNat (Poseidon2.internalDiagonal.getD 1 0) * value.denote := by
  have coefficient : Poseidon2.ofNat (Poseidon2.internalDiagonal.getD 1 0) = 1 := by
    decide
  rw [coefficient]
  simp [scale1]

private theorem scale2_denote (value : Word)
    (h : value.toNat < goldilocksModulus) :
    (scale2 value).denote =
      Poseidon2.ofNat (Poseidon2.internalDiagonal.getD 2 0) * value.denote := by
  rw [scale2, double64_denote value h]
  congr

private theorem scale3_denote (value : Word)
    (h : value.toNat < goldilocksModulus) :
    (scale3 value).denote =
      Poseidon2.ofNat (Poseidon2.internalDiagonal.getD 3 0) * value.denote := by
  rw [scale3, mulHalf64_denote _ h]
  congr

private theorem scale4_denote (value : Word)
    (h : value.toNat < goldilocksModulus) :
    (scale4 value).denote =
      Poseidon2.ofNat (Poseidon2.internalDiagonal.getD 4 0) * value.denote := by
  rw [scale4, triple64_denote value h]
  congr

private theorem scale5_denote (value : Word)
    (h : value.toNat < goldilocksModulus) :
    (scale5 value).denote =
      Poseidon2.ofNat (Poseidon2.internalDiagonal.getD 5 0) * value.denote := by
  rw [scale5, sub64_denote _ _ (by decide) (mulHalf64_canonical value h),
    mulHalf64_denote _ h]
  have coefficient : Poseidon2.ofNat (Poseidon2.internalDiagonal.getD 5 0) =
      -half64.denote := by decide
  have zeroDenote : (0 : UInt64).denote = (0 : F) := by decide
  rw [coefficient, zeroDenote]
  ring

private theorem scale6_denote (value : Word)
    (h : value.toNat < goldilocksModulus) :
    (scale6 value).denote =
      Poseidon2.ofNat (Poseidon2.internalDiagonal.getD 6 0) * value.denote := by
  rw [scale6, sub64_denote _ _ (by decide) (triple64_canonical value h),
    triple64_denote value h]
  have coefficient : Poseidon2.ofNat (Poseidon2.internalDiagonal.getD 6 0) =
      -3 := by decide
  have zeroDenote : (0 : UInt64).denote = (0 : F) := by decide
  rw [coefficient, zeroDenote]
  ring

private theorem scale7_denote (value : Word)
    (h : value.toNat < goldilocksModulus) :
    (scale7 value).denote =
      Poseidon2.ofNat (Poseidon2.internalDiagonal.getD 7 0) * value.denote := by
  rw [scale7, sub64_denote _ _ (by decide)
      (double64_canonical _ (double64_canonical value h)),
    double64_denote _ (double64_canonical value h), double64_denote value h]
  have coefficient : Poseidon2.ofNat (Poseidon2.internalDiagonal.getD 7 0) =
      -4 := by decide
  have zeroDenote : (0 : UInt64).denote = (0 : F) := by decide
  rw [coefficient, zeroDenote]
  ring

@[inline] def internalLayer64 (state : State64) : State64 :=
  let sum := sum8_64 state
  let sumC := sum8_64_canonical state
  replaceLanes64 state
    (add64 (scale0 state.x0) sum)
    (add64 (scale1 state.x1) sum)
    (add64 (scale2 state.x2) sum)
    (add64 (scale3 state.x3) sum)
    (add64 (scale4 state.x4) sum)
    (add64 (scale5 state.x5) sum)
    (add64 (scale6 state.x6) sum)
    (add64 (scale7 state.x7) sum)
    (add64_canonical _ _ (scale0_canonical _ state.c0) sumC)
    (add64_canonical _ _ (scale1_canonical _ state.c1) sumC)
    (add64_canonical _ _ (scale2_canonical _ state.c2) sumC)
    (add64_canonical _ _ (scale3_canonical _ state.c3) sumC)
    (add64_canonical _ _ (scale4_canonical _ state.c4) sumC)
    (add64_canonical _ _ (scale5_canonical _ state.c5) sumC)
    (add64_canonical _ _ (scale6_canonical _ state.c6) sumC)
    (add64_canonical _ _ (scale7_canonical _ state.c7) sumC)

theorem internalLayer64_denote (state : State64) :
    (internalLayer64 state).denote = Poseidon2.internalLayer state.denote := by
  simp only [internalLayer64]
  rw [replaceLanes64_denote]
  simp only [denote]
  rw [add64_denote _ _ (scale0_canonical _ state.c0) (sum8_64_canonical state),
    add64_denote _ _ (scale1_canonical _ state.c1) (sum8_64_canonical state),
    add64_denote _ _ (scale2_canonical _ state.c2) (sum8_64_canonical state),
    add64_denote _ _ (scale3_canonical _ state.c3) (sum8_64_canonical state),
    add64_denote _ _ (scale4_canonical _ state.c4) (sum8_64_canonical state),
    add64_denote _ _ (scale5_canonical _ state.c5) (sum8_64_canonical state),
    add64_denote _ _ (scale6_canonical _ state.c6) (sum8_64_canonical state),
    add64_denote _ _ (scale7_canonical _ state.c7) (sum8_64_canonical state),
    scale0_denote _ state.c0, scale1_denote, scale2_denote _ state.c2,
    scale3_denote _ state.c3, scale4_denote _ state.c4,
    scale5_denote _ state.c5, scale6_denote _ state.c6,
    scale7_denote _ state.c7, sum8_64_denote]
  apply List.ext_get
  · simp [Poseidon2.internalLayer, Poseidon2.width]
  · intro index leftLt rightLt
    have indexLt : index < 8 := by simpa using leftLt
    interval_cases index <;>
      simp [Poseidon2.internalLayer, Poseidon2.width, Poseidon2.internalDiagonal]

@[noinline] def partialRound64 (state : State64) (constant : UInt64) : State64 :=
  internalLayer64 (replaceLanes64 state
    (sbox64 (add64 state.x0 constant))
    state.x1 state.x2 state.x3 state.x4 state.x5 state.x6 state.x7
    (sbox64_canonical _) state.c1 state.c2 state.c3
    state.c4 state.c5 state.c6 state.c7)

theorem partialRound64_denote (constant : UInt64)
    (constantCanonical : constant.toNat < goldilocksModulus)
    (state : State64) :
    (partialRound64 state constant).denote =
      Poseidon2.internalLayer
        (Poseidon2.sbox (state.x0.denote + constant.denote) ::
          state.denote.drop 1) := by
  rw [partialRound64, internalLayer64_denote]
  rw [replaceLanes64_denote]
  simp only [denote]
  rw [sbox64_denote _ (add64_canonical _ _ state.c0 constantCanonical),
    add64_denote _ _ state.c0 constantCanonical]
  rfl

end State64

end NightstreamFPrime.Export.NativePoseidon2
