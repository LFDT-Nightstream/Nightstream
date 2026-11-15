import NightstreamFPrime.Export.NativeField

namespace NightstreamFPrime.Export.NativePoseidon2

open NightstreamFPrime.Spec
open NightstreamFPrime.Export
open Fin.CommRing

@[inline] private def double64 (value : Word) : Word := add64 value value
@[inline] private def triple64 (value : Word) : Word := add64 (double64 value) value

private theorem double64_canonical (value : Word)
    (canonical : value.toNat < fieldModulus) :
    (double64 value).toNat < fieldModulus :=
  add64_canonical value value canonical canonical

private theorem triple64_canonical (value : Word)
    (canonical : value.toNat < fieldModulus) :
    (triple64 value).toNat < fieldModulus :=
  add64_canonical _ value (double64_canonical value canonical) canonical

private theorem double64_denote (value : Word)
    (canonical : value.toNat < fieldModulus) :
    (double64 value).denote = 2 * value.denote := by
  rw [double64, add64_denote value value canonical canonical]
  ring

private theorem triple64_denote (value : Word)
    (canonical : value.toNat < fieldModulus) :
    (triple64 value).denote = 3 * value.denote := by
  rw [triple64, add64_denote _ value
    (double64_canonical value canonical) canonical,
    double64_denote value canonical]
  ring

@[inline] private def sum4_64 (a b c d : Word) : Word :=
  add64 (add64 (add64 a b) c) d

private theorem sum4_64_canonical (a b c d : Word)
    (ha : a.toNat < fieldModulus)
    (hb : b.toNat < fieldModulus)
    (hc : c.toNat < fieldModulus)
    (hd : d.toNat < fieldModulus) :
    (sum4_64 a b c d).toNat < fieldModulus :=
  add64_canonical _ d
    (add64_canonical _ c (add64_canonical a b ha hb) hc) hd

private theorem sum4_64_denote (a b c d : Word)
    (ha : a.toNat < fieldModulus)
    (hb : b.toNat < fieldModulus)
    (hc : c.toNat < fieldModulus)
    (hd : d.toNat < fieldModulus) :
    (sum4_64 a b c d).denote =
      a.denote + b.denote + c.denote + d.denote := by
  simp only [sum4_64]
  rw [add64_denote _ d
      (add64_canonical _ c (add64_canonical a b ha hb) hc) hd,
    add64_denote _ c (add64_canonical a b ha hb) hc,
    add64_denote a b ha hb]

@[inline] private def combine64 (a b : Word) : Word := add64 (double64 a) b

private theorem combine64_canonical (a b : Word)
    (ha : a.toNat < fieldModulus)
    (hb : b.toNat < fieldModulus) :
    (combine64 a b).toNat < fieldModulus :=
  add64_canonical _ b (double64_canonical a ha) hb

private theorem combine64_denote (a b : Word)
    (ha : a.toNat < fieldModulus)
    (hb : b.toNat < fieldModulus) :
    (combine64 a b).denote = a.denote + a.denote + b.denote := by
  rw [combine64, add64_denote _ b (double64_canonical a ha) hb,
    double64_denote a ha]
  ring

structure State64 where
  x0 : UInt64
  x1 : UInt64
  x2 : UInt64
  x3 : UInt64
  x4 : UInt64
  x5 : UInt64
  x6 : UInt64
  x7 : UInt64
  x8 : UInt64
  x9 : UInt64
  x10 : UInt64
  x11 : UInt64
  x12 : UInt64
  x13 : UInt64
  x14 : UInt64
  x15 : UInt64
  canonical :
    x0.toNat < fieldModulus ∧ x1.toNat < fieldModulus ∧
    x2.toNat < fieldModulus ∧ x3.toNat < fieldModulus ∧
    x4.toNat < fieldModulus ∧ x5.toNat < fieldModulus ∧
    x6.toNat < fieldModulus ∧ x7.toNat < fieldModulus ∧
    x8.toNat < fieldModulus ∧ x9.toNat < fieldModulus ∧
    x10.toNat < fieldModulus ∧ x11.toNat < fieldModulus ∧
    x12.toNat < fieldModulus ∧ x13.toNat < fieldModulus ∧
    x14.toNat < fieldModulus ∧ x15.toNat < fieldModulus

namespace State64

def denote (state : State64) : Poseidon2.State :=
  [state.x0.denote, state.x1.denote, state.x2.denote, state.x3.denote,
   state.x4.denote, state.x5.denote, state.x6.denote, state.x7.denote,
   state.x8.denote, state.x9.denote, state.x10.denote, state.x11.denote,
   state.x12.denote, state.x13.denote, state.x14.denote, state.x15.denote]

theorem c0 (state : State64) : state.x0.toNat < fieldModulus :=
  state.canonical.1
theorem c1 (state : State64) : state.x1.toNat < fieldModulus :=
  state.canonical.2.1
theorem c2 (state : State64) : state.x2.toNat < fieldModulus :=
  state.canonical.2.2.1
theorem c3 (state : State64) : state.x3.toNat < fieldModulus :=
  state.canonical.2.2.2.1
theorem c4 (state : State64) : state.x4.toNat < fieldModulus :=
  state.canonical.2.2.2.2.1
theorem c5 (state : State64) : state.x5.toNat < fieldModulus :=
  state.canonical.2.2.2.2.2.1
theorem c6 (state : State64) : state.x6.toNat < fieldModulus :=
  state.canonical.2.2.2.2.2.2.1
theorem c7 (state : State64) : state.x7.toNat < fieldModulus :=
  state.canonical.2.2.2.2.2.2.2.1
theorem c8 (state : State64) : state.x8.toNat < fieldModulus :=
  state.canonical.2.2.2.2.2.2.2.2.1
theorem c9 (state : State64) : state.x9.toNat < fieldModulus :=
  state.canonical.2.2.2.2.2.2.2.2.2.1
theorem c10 (state : State64) : state.x10.toNat < fieldModulus :=
  state.canonical.2.2.2.2.2.2.2.2.2.2.1
theorem c11 (state : State64) : state.x11.toNat < fieldModulus :=
  state.canonical.2.2.2.2.2.2.2.2.2.2.2.1
theorem c12 (state : State64) : state.x12.toNat < fieldModulus :=
  state.canonical.2.2.2.2.2.2.2.2.2.2.2.2.1
theorem c13 (state : State64) : state.x13.toNat < fieldModulus :=
  state.canonical.2.2.2.2.2.2.2.2.2.2.2.2.2.1
theorem c14 (state : State64) : state.x14.toNat < fieldModulus :=
  state.canonical.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
theorem c15 (state : State64) : state.x15.toNat < fieldModulus :=
  state.canonical.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2

def zero : State64 where
  x0 := 0; x1 := 0; x2 := 0; x3 := 0
  x4 := 0; x5 := 0; x6 := 0; x7 := 0
  x8 := 0; x9 := 0; x10 := 0; x11 := 0
  x12 := 0; x13 := 0; x14 := 0; x15 := 0
  canonical := by decide

@[inline] private def mat0 (a b c d : Word) : Word :=
  sum4_64 (double64 a) (triple64 b) c d
@[inline] private def mat1 (a b c d : Word) : Word :=
  sum4_64 a (double64 b) (triple64 c) d
@[inline] private def mat2 (a b c d : Word) : Word :=
  sum4_64 a b (double64 c) (triple64 d)
@[inline] private def mat3 (a b c d : Word) : Word :=
  sum4_64 (triple64 a) b c (double64 d)

private theorem mat0_canonical (a b c d : Word)
    (ha : a.toNat < fieldModulus) (hb : b.toNat < fieldModulus)
    (hc : c.toNat < fieldModulus) (hd : d.toNat < fieldModulus) :
    (mat0 a b c d).toNat < fieldModulus :=
  sum4_64_canonical _ _ c d (double64_canonical a ha)
    (triple64_canonical b hb) hc hd

private theorem mat1_canonical (a b c d : Word)
    (ha : a.toNat < fieldModulus) (hb : b.toNat < fieldModulus)
    (hc : c.toNat < fieldModulus) (hd : d.toNat < fieldModulus) :
    (mat1 a b c d).toNat < fieldModulus :=
  sum4_64_canonical a _ _ d ha (double64_canonical b hb)
    (triple64_canonical c hc) hd

private theorem mat2_canonical (a b c d : Word)
    (ha : a.toNat < fieldModulus) (hb : b.toNat < fieldModulus)
    (hc : c.toNat < fieldModulus) (hd : d.toNat < fieldModulus) :
    (mat2 a b c d).toNat < fieldModulus :=
  sum4_64_canonical a b _ _ ha hb (double64_canonical c hc)
    (triple64_canonical d hd)

private theorem mat3_canonical (a b c d : Word)
    (ha : a.toNat < fieldModulus) (hb : b.toNat < fieldModulus)
    (hc : c.toNat < fieldModulus) (hd : d.toNat < fieldModulus) :
    (mat3 a b c d).toNat < fieldModulus :=
  sum4_64_canonical _ b c _ (triple64_canonical a ha) hb hc
    (double64_canonical d hd)

private theorem mat0_denote (a b c d : Word)
    (ha : a.toNat < fieldModulus) (hb : b.toNat < fieldModulus)
    (hc : c.toNat < fieldModulus) (hd : d.toNat < fieldModulus) :
    (mat0 a b c d).denote =
      2 * a.denote + 3 * b.denote + c.denote + d.denote := by
  rw [mat0, sum4_64_denote _ _ c d (double64_canonical a ha)
    (triple64_canonical b hb) hc hd, double64_denote a ha,
    triple64_denote b hb]

private theorem mat1_denote (a b c d : Word)
    (ha : a.toNat < fieldModulus) (hb : b.toNat < fieldModulus)
    (hc : c.toNat < fieldModulus) (hd : d.toNat < fieldModulus) :
    (mat1 a b c d).denote =
      a.denote + 2 * b.denote + 3 * c.denote + d.denote := by
  rw [mat1, sum4_64_denote a _ _ d ha (double64_canonical b hb)
    (triple64_canonical c hc) hd, double64_denote b hb,
    triple64_denote c hc]

private theorem mat2_denote (a b c d : Word)
    (ha : a.toNat < fieldModulus) (hb : b.toNat < fieldModulus)
    (hc : c.toNat < fieldModulus) (hd : d.toNat < fieldModulus) :
    (mat2 a b c d).denote =
      a.denote + b.denote + 2 * c.denote + 3 * d.denote := by
  rw [mat2, sum4_64_denote a b _ _ ha hb (double64_canonical c hc)
    (triple64_canonical d hd), double64_denote c hc,
    triple64_denote d hd]

private theorem mat3_denote (a b c d : Word)
    (ha : a.toNat < fieldModulus) (hb : b.toNat < fieldModulus)
    (hc : c.toNat < fieldModulus) (hd : d.toNat < fieldModulus) :
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
  let m8 := mat0 state.x8 state.x9 state.x10 state.x11
  let m9 := mat1 state.x8 state.x9 state.x10 state.x11
  let m10 := mat2 state.x8 state.x9 state.x10 state.x11
  let m11 := mat3 state.x8 state.x9 state.x10 state.x11
  let m12 := mat0 state.x12 state.x13 state.x14 state.x15
  let m13 := mat1 state.x12 state.x13 state.x14 state.x15
  let m14 := mat2 state.x12 state.x13 state.x14 state.x15
  let m15 := mat3 state.x12 state.x13 state.x14 state.x15
  let s0 := sum4_64 m0 m4 m8 m12
  let s1 := sum4_64 m1 m5 m9 m13
  let s2 := sum4_64 m2 m6 m10 m14
  let s3 := sum4_64 m3 m7 m11 m15
  have k0 : m0.toNat < fieldModulus :=
    mat0_canonical _ _ _ _ state.c0 state.c1 state.c2 state.c3
  have k1 : m1.toNat < fieldModulus :=
    mat1_canonical _ _ _ _ state.c0 state.c1 state.c2 state.c3
  have k2 : m2.toNat < fieldModulus :=
    mat2_canonical _ _ _ _ state.c0 state.c1 state.c2 state.c3
  have k3 : m3.toNat < fieldModulus :=
    mat3_canonical _ _ _ _ state.c0 state.c1 state.c2 state.c3
  have k4 : m4.toNat < fieldModulus :=
    mat0_canonical _ _ _ _ state.c4 state.c5 state.c6 state.c7
  have k5 : m5.toNat < fieldModulus :=
    mat1_canonical _ _ _ _ state.c4 state.c5 state.c6 state.c7
  have k6 : m6.toNat < fieldModulus :=
    mat2_canonical _ _ _ _ state.c4 state.c5 state.c6 state.c7
  have k7 : m7.toNat < fieldModulus :=
    mat3_canonical _ _ _ _ state.c4 state.c5 state.c6 state.c7
  have k8 : m8.toNat < fieldModulus :=
    mat0_canonical _ _ _ _ state.c8 state.c9 state.c10 state.c11
  have k9 : m9.toNat < fieldModulus :=
    mat1_canonical _ _ _ _ state.c8 state.c9 state.c10 state.c11
  have k10 : m10.toNat < fieldModulus :=
    mat2_canonical _ _ _ _ state.c8 state.c9 state.c10 state.c11
  have k11 : m11.toNat < fieldModulus :=
    mat3_canonical _ _ _ _ state.c8 state.c9 state.c10 state.c11
  have k12 : m12.toNat < fieldModulus :=
    mat0_canonical _ _ _ _ state.c12 state.c13 state.c14 state.c15
  have k13 : m13.toNat < fieldModulus :=
    mat1_canonical _ _ _ _ state.c12 state.c13 state.c14 state.c15
  have k14 : m14.toNat < fieldModulus :=
    mat2_canonical _ _ _ _ state.c12 state.c13 state.c14 state.c15
  have k15 : m15.toNat < fieldModulus :=
    mat3_canonical _ _ _ _ state.c12 state.c13 state.c14 state.c15
  have t0 : s0.toNat < fieldModulus := sum4_64_canonical _ _ _ _ k0 k4 k8 k12
  have t1 : s1.toNat < fieldModulus := sum4_64_canonical _ _ _ _ k1 k5 k9 k13
  have t2 : s2.toNat < fieldModulus := sum4_64_canonical _ _ _ _ k2 k6 k10 k14
  have t3 : s3.toNat < fieldModulus := sum4_64_canonical _ _ _ _ k3 k7 k11 k15
  { x0 := add64 m0 s0, x1 := add64 m1 s1, x2 := add64 m2 s2, x3 := add64 m3 s3,
    x4 := add64 m4 s0, x5 := add64 m5 s1, x6 := add64 m6 s2, x7 := add64 m7 s3,
    x8 := add64 m8 s0, x9 := add64 m9 s1, x10 := add64 m10 s2, x11 := add64 m11 s3,
    x12 := add64 m12 s0, x13 := add64 m13 s1, x14 := add64 m14 s2, x15 := add64 m15 s3,
    canonical :=
      ⟨add64_canonical _ _ k0 t0, add64_canonical _ _ k1 t1,
       add64_canonical _ _ k2 t2, add64_canonical _ _ k3 t3,
       add64_canonical _ _ k4 t0, add64_canonical _ _ k5 t1,
       add64_canonical _ _ k6 t2, add64_canonical _ _ k7 t3,
       add64_canonical _ _ k8 t0, add64_canonical _ _ k9 t1,
       add64_canonical _ _ k10 t2, add64_canonical _ _ k11 t3,
       add64_canonical _ _ k12 t0, add64_canonical _ _ k13 t1,
       add64_canonical _ _ k14 t2, add64_canonical _ _ k15 t3⟩ }

theorem externalLayer64_denote (state : State64) :
    (externalLayer64 state).denote = Poseidon2.externalLayer state.denote := by
  have k0 := mat0_canonical _ _ _ _ state.c0 state.c1 state.c2 state.c3
  have k1 := mat1_canonical _ _ _ _ state.c0 state.c1 state.c2 state.c3
  have k2 := mat2_canonical _ _ _ _ state.c0 state.c1 state.c2 state.c3
  have k3 := mat3_canonical _ _ _ _ state.c0 state.c1 state.c2 state.c3
  have k4 := mat0_canonical _ _ _ _ state.c4 state.c5 state.c6 state.c7
  have k5 := mat1_canonical _ _ _ _ state.c4 state.c5 state.c6 state.c7
  have k6 := mat2_canonical _ _ _ _ state.c4 state.c5 state.c6 state.c7
  have k7 := mat3_canonical _ _ _ _ state.c4 state.c5 state.c6 state.c7
  have k8 := mat0_canonical _ _ _ _ state.c8 state.c9 state.c10 state.c11
  have k9 := mat1_canonical _ _ _ _ state.c8 state.c9 state.c10 state.c11
  have k10 := mat2_canonical _ _ _ _ state.c8 state.c9 state.c10 state.c11
  have k11 := mat3_canonical _ _ _ _ state.c8 state.c9 state.c10 state.c11
  have k12 := mat0_canonical _ _ _ _ state.c12 state.c13 state.c14 state.c15
  have k13 := mat1_canonical _ _ _ _ state.c12 state.c13 state.c14 state.c15
  have k14 := mat2_canonical _ _ _ _ state.c12 state.c13 state.c14 state.c15
  have k15 := mat3_canonical _ _ _ _ state.c12 state.c13 state.c14 state.c15
  have t0 := sum4_64_canonical _ _ _ _ k0 k4 k8 k12
  have t1 := sum4_64_canonical _ _ _ _ k1 k5 k9 k13
  have t2 := sum4_64_canonical _ _ _ _ k2 k6 k10 k14
  have t3 := sum4_64_canonical _ _ _ _ k3 k7 k11 k15
  simp only [externalLayer64, denote]
  rw [add64_denote _ _ k0 t0,
    add64_denote _ _ k1 t1,
    add64_denote _ _ k2 t2,
    add64_denote _ _ k3 t3,
    add64_denote _ _ k4 t0,
    add64_denote _ _ k5 t1,
    add64_denote _ _ k6 t2,
    add64_denote _ _ k7 t3,
    add64_denote _ _ k8 t0,
    add64_denote _ _ k9 t1,
    add64_denote _ _ k10 t2,
    add64_denote _ _ k11 t3,
    add64_denote _ _ k12 t0,
    add64_denote _ _ k13 t1,
    add64_denote _ _ k14 t2,
    add64_denote _ _ k15 t3,
    sum4_64_denote _ _ _ _ k0 k4 k8 k12,
    sum4_64_denote _ _ _ _ k1 k5 k9 k13,
    sum4_64_denote _ _ _ _ k2 k6 k10 k14,
    sum4_64_denote _ _ _ _ k3 k7 k11 k15,
    mat0_denote _ _ _ _ state.c0 state.c1 state.c2 state.c3,
    mat1_denote _ _ _ _ state.c0 state.c1 state.c2 state.c3,
    mat2_denote _ _ _ _ state.c0 state.c1 state.c2 state.c3,
    mat3_denote _ _ _ _ state.c0 state.c1 state.c2 state.c3,
    mat0_denote _ _ _ _ state.c4 state.c5 state.c6 state.c7,
    mat1_denote _ _ _ _ state.c4 state.c5 state.c6 state.c7,
    mat2_denote _ _ _ _ state.c4 state.c5 state.c6 state.c7,
    mat3_denote _ _ _ _ state.c4 state.c5 state.c6 state.c7,
    mat0_denote _ _ _ _ state.c8 state.c9 state.c10 state.c11,
    mat1_denote _ _ _ _ state.c8 state.c9 state.c10 state.c11,
    mat2_denote _ _ _ _ state.c8 state.c9 state.c10 state.c11,
    mat3_denote _ _ _ _ state.c8 state.c9 state.c10 state.c11,
    mat0_denote _ _ _ _ state.c12 state.c13 state.c14 state.c15,
    mat1_denote _ _ _ _ state.c12 state.c13 state.c14 state.c15,
    mat2_denote _ _ _ _ state.c12 state.c13 state.c14 state.c15,
    mat3_denote _ _ _ _ state.c12 state.c13 state.c14 state.c15]
  simp [Poseidon2.externalLayer, Poseidon2.width, Poseidon2.mat4, List.range_succ]

@[inline] private def sbox64 (value : Word) : Word :=
  let x2 := square64 value
  let x4 := square64 x2
  mul64 (mul64 x4 x2) value

private theorem sbox64_canonical (value : Word) :
    (sbox64 value).toNat < fieldModulus := by
  simp only [sbox64]
  exact mul64_canonical _ value

private theorem sbox64_denote (value : Word)
    (canonical : value.toNat < fieldModulus) :
    (sbox64 value).denote = Poseidon2.sbox value.denote := by
  let x2 := square64 value
  let x4 := square64 x2
  let x6 := mul64 x4 x2
  have x2Canonical : x2.toNat < fieldModulus := square64_canonical _
  have x4Canonical : x4.toNat < fieldModulus := square64_canonical _
  have x6Canonical : x6.toNat < fieldModulus := mul64_canonical _ _
  change (mul64 x6 value).denote =
    ((value.denote * value.denote) * (value.denote * value.denote)) *
      (value.denote * value.denote) * value.denote
  rw [mul64_denote x6 value x6Canonical canonical,
    mul64_denote x4 x2 x4Canonical x2Canonical,
    square64_denote x2 x2Canonical,
    square64_denote value canonical]

@[noinline] def fullRound64 (state : State64) (constants : @& State64) : State64 :=
  externalLayer64 {
    x0 := sbox64 (add64 state.x0 constants.x0)
    x1 := sbox64 (add64 state.x1 constants.x1)
    x2 := sbox64 (add64 state.x2 constants.x2)
    x3 := sbox64 (add64 state.x3 constants.x3)
    x4 := sbox64 (add64 state.x4 constants.x4)
    x5 := sbox64 (add64 state.x5 constants.x5)
    x6 := sbox64 (add64 state.x6 constants.x6)
    x7 := sbox64 (add64 state.x7 constants.x7)
    x8 := sbox64 (add64 state.x8 constants.x8)
    x9 := sbox64 (add64 state.x9 constants.x9)
    x10 := sbox64 (add64 state.x10 constants.x10)
    x11 := sbox64 (add64 state.x11 constants.x11)
    x12 := sbox64 (add64 state.x12 constants.x12)
    x13 := sbox64 (add64 state.x13 constants.x13)
    x14 := sbox64 (add64 state.x14 constants.x14)
    x15 := sbox64 (add64 state.x15 constants.x15)
    canonical := ⟨sbox64_canonical _, sbox64_canonical _, sbox64_canonical _, sbox64_canonical _,
      sbox64_canonical _, sbox64_canonical _, sbox64_canonical _, sbox64_canonical _,
      sbox64_canonical _, sbox64_canonical _, sbox64_canonical _, sbox64_canonical _,
      sbox64_canonical _, sbox64_canonical _, sbox64_canonical _, sbox64_canonical _⟩ }

theorem fullRound64_denote (constants state : State64) :
    (fullRound64 state constants).denote = Poseidon2.externalLayer [
      Poseidon2.sbox (state.x0.denote + constants.x0.denote),
      Poseidon2.sbox (state.x1.denote + constants.x1.denote),
      Poseidon2.sbox (state.x2.denote + constants.x2.denote),
      Poseidon2.sbox (state.x3.denote + constants.x3.denote),
      Poseidon2.sbox (state.x4.denote + constants.x4.denote),
      Poseidon2.sbox (state.x5.denote + constants.x5.denote),
      Poseidon2.sbox (state.x6.denote + constants.x6.denote),
      Poseidon2.sbox (state.x7.denote + constants.x7.denote),
      Poseidon2.sbox (state.x8.denote + constants.x8.denote),
      Poseidon2.sbox (state.x9.denote + constants.x9.denote),
      Poseidon2.sbox (state.x10.denote + constants.x10.denote),
      Poseidon2.sbox (state.x11.denote + constants.x11.denote),
      Poseidon2.sbox (state.x12.denote + constants.x12.denote),
      Poseidon2.sbox (state.x13.denote + constants.x13.denote),
      Poseidon2.sbox (state.x14.denote + constants.x14.denote),
      Poseidon2.sbox (state.x15.denote + constants.x15.denote)] := by
  rw [fullRound64, externalLayer64_denote]
  simp only [denote]
  rw [sbox64_denote _ (add64_canonical _ _ state.c0 constants.c0),
    sbox64_denote _ (add64_canonical _ _ state.c1 constants.c1),
    sbox64_denote _ (add64_canonical _ _ state.c2 constants.c2),
    sbox64_denote _ (add64_canonical _ _ state.c3 constants.c3),
    sbox64_denote _ (add64_canonical _ _ state.c4 constants.c4),
    sbox64_denote _ (add64_canonical _ _ state.c5 constants.c5),
    sbox64_denote _ (add64_canonical _ _ state.c6 constants.c6),
    sbox64_denote _ (add64_canonical _ _ state.c7 constants.c7),
    sbox64_denote _ (add64_canonical _ _ state.c8 constants.c8),
    sbox64_denote _ (add64_canonical _ _ state.c9 constants.c9),
    sbox64_denote _ (add64_canonical _ _ state.c10 constants.c10),
    sbox64_denote _ (add64_canonical _ _ state.c11 constants.c11),
    sbox64_denote _ (add64_canonical _ _ state.c12 constants.c12),
    sbox64_denote _ (add64_canonical _ _ state.c13 constants.c13),
    sbox64_denote _ (add64_canonical _ _ state.c14 constants.c14),
    sbox64_denote _ (add64_canonical _ _ state.c15 constants.c15),
    add64_denote _ _ state.c0 constants.c0,
    add64_denote _ _ state.c1 constants.c1,
    add64_denote _ _ state.c2 constants.c2,
    add64_denote _ _ state.c3 constants.c3,
    add64_denote _ _ state.c4 constants.c4,
    add64_denote _ _ state.c5 constants.c5,
    add64_denote _ _ state.c6 constants.c6,
    add64_denote _ _ state.c7 constants.c7,
    add64_denote _ _ state.c8 constants.c8,
    add64_denote _ _ state.c9 constants.c9,
    add64_denote _ _ state.c10 constants.c10,
    add64_denote _ _ state.c11 constants.c11,
    add64_denote _ _ state.c12 constants.c12,
    add64_denote _ _ state.c13 constants.c13,
    add64_denote _ _ state.c14 constants.c14,
    add64_denote _ _ state.c15 constants.c15]

@[inline] private def sum16_64 (state : State64) : Word :=
  add64
    (add64 (sum4_64 state.x0 state.x1 state.x2 state.x3)
      (sum4_64 state.x4 state.x5 state.x6 state.x7))
    (add64 (sum4_64 state.x8 state.x9 state.x10 state.x11)
      (sum4_64 state.x12 state.x13 state.x14 state.x15))

private theorem sum16_64_canonical (state : State64) :
    (sum16_64 state).toNat < fieldModulus :=
  add64_canonical _ _
    (add64_canonical _ _ (sum4_64_canonical _ _ _ _ state.c0 state.c1 state.c2 state.c3)
      (sum4_64_canonical _ _ _ _ state.c4 state.c5 state.c6 state.c7))
    (add64_canonical _ _ (sum4_64_canonical _ _ _ _ state.c8 state.c9 state.c10 state.c11)
      (sum4_64_canonical _ _ _ _ state.c12 state.c13 state.c14 state.c15))

private theorem sum16_64_denote (state : State64) :
    (sum16_64 state).denote =
      state.x0.denote + state.x1.denote + state.x2.denote + state.x3.denote +
      state.x4.denote + state.x5.denote + state.x6.denote + state.x7.denote +
      state.x8.denote + state.x9.denote + state.x10.denote + state.x11.denote +
      state.x12.denote + state.x13.denote + state.x14.denote + state.x15.denote := by
  have q0 := sum4_64_canonical _ _ _ _ state.c0 state.c1 state.c2 state.c3
  have q1 := sum4_64_canonical _ _ _ _ state.c4 state.c5 state.c6 state.c7
  have q2 := sum4_64_canonical _ _ _ _ state.c8 state.c9 state.c10 state.c11
  have q3 := sum4_64_canonical _ _ _ _ state.c12 state.c13 state.c14 state.c15
  rw [sum16_64, add64_denote _ _ (add64_canonical _ _ q0 q1) (add64_canonical _ _ q2 q3),
    add64_denote _ _ q0 q1, add64_denote _ _ q2 q3,
    sum4_64_denote _ _ _ _ state.c0 state.c1 state.c2 state.c3,
    sum4_64_denote _ _ _ _ state.c4 state.c5 state.c6 state.c7,
    sum4_64_denote _ _ _ _ state.c8 state.c9 state.c10 state.c11,
    sum4_64_denote _ _ _ _ state.c12 state.c13 state.c14 state.c15]
  ring

/-! The internal diagonal is `[1, 2, -2, 3, -3, 4, -4, -5, 6, -6, 7, -7, 8, -8, 9, -9]`
(`Poseidon2.internalDiagonal_signed`), so each scale is a short chain of additions and at most
one negation. -/

@[inline] private def neg64 (value : Word) : Word := sub64 0 value
@[inline] private def times4 (value : Word) : Word := double64 (double64 value)
@[inline] private def times5 (value : Word) : Word := add64 (times4 value) value
@[inline] private def times6 (value : Word) : Word := double64 (triple64 value)
@[inline] private def times7 (value : Word) : Word := add64 (times6 value) value
@[inline] private def times8 (value : Word) : Word := double64 (times4 value)
@[inline] private def times9 (value : Word) : Word := triple64 (triple64 value)

private theorem neg64_canonical (value : Word)
    (h : value.toNat < fieldModulus) :
    (neg64 value).toNat < fieldModulus :=
  sub64_canonical _ _ (by decide) h

private theorem neg64_denote (value : Word)
    (h : value.toNat < fieldModulus) :
    (neg64 value).denote = -value.denote := by
  rw [neg64, sub64_denote _ _ (by decide) h]
  have zeroDenote : (0 : UInt64).denote = (0 : F) := by decide
  rw [zeroDenote]
  ring

private theorem times4_canonical (value : Word)
    (h : value.toNat < fieldModulus) :
    (times4 value).toNat < fieldModulus :=
  double64_canonical _ (double64_canonical value h)

private theorem times4_denote (value : Word)
    (h : value.toNat < fieldModulus) :
    (times4 value).denote = 4 * value.denote := by
  rw [times4, double64_denote _ (double64_canonical value h), double64_denote value h]
  ring

private theorem times5_canonical (value : Word)
    (h : value.toNat < fieldModulus) :
    (times5 value).toNat < fieldModulus :=
  add64_canonical _ _ (times4_canonical value h) h

private theorem times5_denote (value : Word)
    (h : value.toNat < fieldModulus) :
    (times5 value).denote = 5 * value.denote := by
  rw [times5, add64_denote _ _ (times4_canonical value h) h, times4_denote value h]
  ring

private theorem times6_canonical (value : Word)
    (h : value.toNat < fieldModulus) :
    (times6 value).toNat < fieldModulus :=
  double64_canonical _ (triple64_canonical value h)

private theorem times6_denote (value : Word)
    (h : value.toNat < fieldModulus) :
    (times6 value).denote = 6 * value.denote := by
  rw [times6, double64_denote _ (triple64_canonical value h), triple64_denote value h]
  ring

private theorem times7_canonical (value : Word)
    (h : value.toNat < fieldModulus) :
    (times7 value).toNat < fieldModulus :=
  add64_canonical _ _ (times6_canonical value h) h

private theorem times7_denote (value : Word)
    (h : value.toNat < fieldModulus) :
    (times7 value).denote = 7 * value.denote := by
  rw [times7, add64_denote _ _ (times6_canonical value h) h, times6_denote value h]
  ring

private theorem times8_canonical (value : Word)
    (h : value.toNat < fieldModulus) :
    (times8 value).toNat < fieldModulus :=
  double64_canonical _ (times4_canonical value h)

private theorem times8_denote (value : Word)
    (h : value.toNat < fieldModulus) :
    (times8 value).denote = 8 * value.denote := by
  rw [times8, double64_denote _ (times4_canonical value h), times4_denote value h]
  ring

private theorem times9_canonical (value : Word)
    (h : value.toNat < fieldModulus) :
    (times9 value).toNat < fieldModulus :=
  triple64_canonical _ (triple64_canonical value h)

private theorem times9_denote (value : Word)
    (h : value.toNat < fieldModulus) :
    (times9 value).denote = 9 * value.denote := by
  rw [times9, triple64_denote _ (triple64_canonical value h), triple64_denote value h]
  ring

@[inline] private def scale0 (value : Word) : Word := value
@[inline] private def scale1 (value : Word) : Word := double64 value
@[inline] private def scale2 (value : Word) : Word := neg64 (double64 value)
@[inline] private def scale3 (value : Word) : Word := triple64 value
@[inline] private def scale4 (value : Word) : Word := neg64 (triple64 value)
@[inline] private def scale5 (value : Word) : Word := times4 value
@[inline] private def scale6 (value : Word) : Word := neg64 (times4 value)
@[inline] private def scale7 (value : Word) : Word := neg64 (times5 value)
@[inline] private def scale8 (value : Word) : Word := times6 value
@[inline] private def scale9 (value : Word) : Word := neg64 (times6 value)
@[inline] private def scale10 (value : Word) : Word := times7 value
@[inline] private def scale11 (value : Word) : Word := neg64 (times7 value)
@[inline] private def scale12 (value : Word) : Word := times8 value
@[inline] private def scale13 (value : Word) : Word := neg64 (times8 value)
@[inline] private def scale14 (value : Word) : Word := times9 value
@[inline] private def scale15 (value : Word) : Word := neg64 (times9 value)

private theorem scale0_canonical (value : Word)
    (h : value.toNat < fieldModulus) :
    (scale0 value).toNat < fieldModulus :=
  h

private theorem scale1_canonical (value : Word)
    (h : value.toNat < fieldModulus) :
    (scale1 value).toNat < fieldModulus :=
  double64_canonical value h

private theorem scale2_canonical (value : Word)
    (h : value.toNat < fieldModulus) :
    (scale2 value).toNat < fieldModulus :=
  neg64_canonical _ (double64_canonical value h)

private theorem scale3_canonical (value : Word)
    (h : value.toNat < fieldModulus) :
    (scale3 value).toNat < fieldModulus :=
  triple64_canonical value h

private theorem scale4_canonical (value : Word)
    (h : value.toNat < fieldModulus) :
    (scale4 value).toNat < fieldModulus :=
  neg64_canonical _ (triple64_canonical value h)

private theorem scale5_canonical (value : Word)
    (h : value.toNat < fieldModulus) :
    (scale5 value).toNat < fieldModulus :=
  times4_canonical value h

private theorem scale6_canonical (value : Word)
    (h : value.toNat < fieldModulus) :
    (scale6 value).toNat < fieldModulus :=
  neg64_canonical _ (times4_canonical value h)

private theorem scale7_canonical (value : Word)
    (h : value.toNat < fieldModulus) :
    (scale7 value).toNat < fieldModulus :=
  neg64_canonical _ (times5_canonical value h)

private theorem scale8_canonical (value : Word)
    (h : value.toNat < fieldModulus) :
    (scale8 value).toNat < fieldModulus :=
  times6_canonical value h

private theorem scale9_canonical (value : Word)
    (h : value.toNat < fieldModulus) :
    (scale9 value).toNat < fieldModulus :=
  neg64_canonical _ (times6_canonical value h)

private theorem scale10_canonical (value : Word)
    (h : value.toNat < fieldModulus) :
    (scale10 value).toNat < fieldModulus :=
  times7_canonical value h

private theorem scale11_canonical (value : Word)
    (h : value.toNat < fieldModulus) :
    (scale11 value).toNat < fieldModulus :=
  neg64_canonical _ (times7_canonical value h)

private theorem scale12_canonical (value : Word)
    (h : value.toNat < fieldModulus) :
    (scale12 value).toNat < fieldModulus :=
  times8_canonical value h

private theorem scale13_canonical (value : Word)
    (h : value.toNat < fieldModulus) :
    (scale13 value).toNat < fieldModulus :=
  neg64_canonical _ (times8_canonical value h)

private theorem scale14_canonical (value : Word)
    (h : value.toNat < fieldModulus) :
    (scale14 value).toNat < fieldModulus :=
  times9_canonical value h

private theorem scale15_canonical (value : Word)
    (h : value.toNat < fieldModulus) :
    (scale15 value).toNat < fieldModulus :=
  neg64_canonical _ (times9_canonical value h)

private theorem scale0_denote (value : Word)
    (_h : value.toNat < fieldModulus) :
    (scale0 value).denote =
      Poseidon2.ofNat (Poseidon2.internalDiagonal.getD 0 0) * value.denote := by
  have coefficient : Poseidon2.ofNat (Poseidon2.internalDiagonal.getD 0 0) =
      1 := by decide
  rw [coefficient]
  simp [scale0]

private theorem scale1_denote (value : Word)
    (h : value.toNat < fieldModulus) :
    (scale1 value).denote =
      Poseidon2.ofNat (Poseidon2.internalDiagonal.getD 1 0) * value.denote := by
  rw [scale1, double64_denote value h]
  have coefficient : Poseidon2.ofNat (Poseidon2.internalDiagonal.getD 1 0) =
      2 := by decide
  rw [coefficient]

private theorem scale2_denote (value : Word)
    (h : value.toNat < fieldModulus) :
    (scale2 value).denote =
      Poseidon2.ofNat (Poseidon2.internalDiagonal.getD 2 0) * value.denote := by
  rw [scale2, neg64_denote _ (double64_canonical value h), double64_denote value h]
  have coefficient : Poseidon2.ofNat (Poseidon2.internalDiagonal.getD 2 0) =
      -(2) := by decide
  rw [coefficient]
  ring

private theorem scale3_denote (value : Word)
    (h : value.toNat < fieldModulus) :
    (scale3 value).denote =
      Poseidon2.ofNat (Poseidon2.internalDiagonal.getD 3 0) * value.denote := by
  rw [scale3, triple64_denote value h]
  have coefficient : Poseidon2.ofNat (Poseidon2.internalDiagonal.getD 3 0) =
      3 := by decide
  rw [coefficient]

private theorem scale4_denote (value : Word)
    (h : value.toNat < fieldModulus) :
    (scale4 value).denote =
      Poseidon2.ofNat (Poseidon2.internalDiagonal.getD 4 0) * value.denote := by
  rw [scale4, neg64_denote _ (triple64_canonical value h), triple64_denote value h]
  have coefficient : Poseidon2.ofNat (Poseidon2.internalDiagonal.getD 4 0) =
      -(3) := by decide
  rw [coefficient]
  ring

private theorem scale5_denote (value : Word)
    (h : value.toNat < fieldModulus) :
    (scale5 value).denote =
      Poseidon2.ofNat (Poseidon2.internalDiagonal.getD 5 0) * value.denote := by
  rw [scale5, times4_denote value h]
  have coefficient : Poseidon2.ofNat (Poseidon2.internalDiagonal.getD 5 0) =
      4 := by decide
  rw [coefficient]

private theorem scale6_denote (value : Word)
    (h : value.toNat < fieldModulus) :
    (scale6 value).denote =
      Poseidon2.ofNat (Poseidon2.internalDiagonal.getD 6 0) * value.denote := by
  rw [scale6, neg64_denote _ (times4_canonical value h), times4_denote value h]
  have coefficient : Poseidon2.ofNat (Poseidon2.internalDiagonal.getD 6 0) =
      -(4) := by decide
  rw [coefficient]
  ring

private theorem scale7_denote (value : Word)
    (h : value.toNat < fieldModulus) :
    (scale7 value).denote =
      Poseidon2.ofNat (Poseidon2.internalDiagonal.getD 7 0) * value.denote := by
  rw [scale7, neg64_denote _ (times5_canonical value h), times5_denote value h]
  have coefficient : Poseidon2.ofNat (Poseidon2.internalDiagonal.getD 7 0) =
      -(5) := by decide
  rw [coefficient]
  ring

private theorem scale8_denote (value : Word)
    (h : value.toNat < fieldModulus) :
    (scale8 value).denote =
      Poseidon2.ofNat (Poseidon2.internalDiagonal.getD 8 0) * value.denote := by
  rw [scale8, times6_denote value h]
  have coefficient : Poseidon2.ofNat (Poseidon2.internalDiagonal.getD 8 0) =
      6 := by decide
  rw [coefficient]

private theorem scale9_denote (value : Word)
    (h : value.toNat < fieldModulus) :
    (scale9 value).denote =
      Poseidon2.ofNat (Poseidon2.internalDiagonal.getD 9 0) * value.denote := by
  rw [scale9, neg64_denote _ (times6_canonical value h), times6_denote value h]
  have coefficient : Poseidon2.ofNat (Poseidon2.internalDiagonal.getD 9 0) =
      -(6) := by decide
  rw [coefficient]
  ring

private theorem scale10_denote (value : Word)
    (h : value.toNat < fieldModulus) :
    (scale10 value).denote =
      Poseidon2.ofNat (Poseidon2.internalDiagonal.getD 10 0) * value.denote := by
  rw [scale10, times7_denote value h]
  have coefficient : Poseidon2.ofNat (Poseidon2.internalDiagonal.getD 10 0) =
      7 := by decide
  rw [coefficient]

private theorem scale11_denote (value : Word)
    (h : value.toNat < fieldModulus) :
    (scale11 value).denote =
      Poseidon2.ofNat (Poseidon2.internalDiagonal.getD 11 0) * value.denote := by
  rw [scale11, neg64_denote _ (times7_canonical value h), times7_denote value h]
  have coefficient : Poseidon2.ofNat (Poseidon2.internalDiagonal.getD 11 0) =
      -(7) := by decide
  rw [coefficient]
  ring

private theorem scale12_denote (value : Word)
    (h : value.toNat < fieldModulus) :
    (scale12 value).denote =
      Poseidon2.ofNat (Poseidon2.internalDiagonal.getD 12 0) * value.denote := by
  rw [scale12, times8_denote value h]
  have coefficient : Poseidon2.ofNat (Poseidon2.internalDiagonal.getD 12 0) =
      8 := by decide
  rw [coefficient]

private theorem scale13_denote (value : Word)
    (h : value.toNat < fieldModulus) :
    (scale13 value).denote =
      Poseidon2.ofNat (Poseidon2.internalDiagonal.getD 13 0) * value.denote := by
  rw [scale13, neg64_denote _ (times8_canonical value h), times8_denote value h]
  have coefficient : Poseidon2.ofNat (Poseidon2.internalDiagonal.getD 13 0) =
      -(8) := by decide
  rw [coefficient]
  ring

private theorem scale14_denote (value : Word)
    (h : value.toNat < fieldModulus) :
    (scale14 value).denote =
      Poseidon2.ofNat (Poseidon2.internalDiagonal.getD 14 0) * value.denote := by
  rw [scale14, times9_denote value h]
  have coefficient : Poseidon2.ofNat (Poseidon2.internalDiagonal.getD 14 0) =
      9 := by decide
  rw [coefficient]

private theorem scale15_denote (value : Word)
    (h : value.toNat < fieldModulus) :
    (scale15 value).denote =
      Poseidon2.ofNat (Poseidon2.internalDiagonal.getD 15 0) * value.denote := by
  rw [scale15, neg64_denote _ (times9_canonical value h), times9_denote value h]
  have coefficient : Poseidon2.ofNat (Poseidon2.internalDiagonal.getD 15 0) =
      -(9) := by decide
  rw [coefficient]
  ring

@[inline] def internalLayer64 (state : State64) : State64 :=
  let sum := sum16_64 state
  have sumC : sum.toNat < fieldModulus := sum16_64_canonical state
  { x0 := add64 (scale0 state.x0) sum, x1 := add64 (scale1 state.x1) sum, x2 := add64 (scale2 state.x2) sum, x3 := add64 (scale3 state.x3) sum,
    x4 := add64 (scale4 state.x4) sum, x5 := add64 (scale5 state.x5) sum, x6 := add64 (scale6 state.x6) sum, x7 := add64 (scale7 state.x7) sum,
    x8 := add64 (scale8 state.x8) sum, x9 := add64 (scale9 state.x9) sum, x10 := add64 (scale10 state.x10) sum, x11 := add64 (scale11 state.x11) sum,
    x12 := add64 (scale12 state.x12) sum, x13 := add64 (scale13 state.x13) sum, x14 := add64 (scale14 state.x14) sum, x15 := add64 (scale15 state.x15) sum,
    canonical :=
      ⟨add64_canonical _ _ (scale0_canonical _ state.c0) sumC,
       add64_canonical _ _ (scale1_canonical _ state.c1) sumC,
       add64_canonical _ _ (scale2_canonical _ state.c2) sumC,
       add64_canonical _ _ (scale3_canonical _ state.c3) sumC,
       add64_canonical _ _ (scale4_canonical _ state.c4) sumC,
       add64_canonical _ _ (scale5_canonical _ state.c5) sumC,
       add64_canonical _ _ (scale6_canonical _ state.c6) sumC,
       add64_canonical _ _ (scale7_canonical _ state.c7) sumC,
       add64_canonical _ _ (scale8_canonical _ state.c8) sumC,
       add64_canonical _ _ (scale9_canonical _ state.c9) sumC,
       add64_canonical _ _ (scale10_canonical _ state.c10) sumC,
       add64_canonical _ _ (scale11_canonical _ state.c11) sumC,
       add64_canonical _ _ (scale12_canonical _ state.c12) sumC,
       add64_canonical _ _ (scale13_canonical _ state.c13) sumC,
       add64_canonical _ _ (scale14_canonical _ state.c14) sumC,
       add64_canonical _ _ (scale15_canonical _ state.c15) sumC⟩ }

theorem internalLayer64_denote (state : State64) :
    (internalLayer64 state).denote = Poseidon2.internalLayer state.denote := by
  simp only [internalLayer64, denote]
  rw [add64_denote _ _ (scale0_canonical _ state.c0) (sum16_64_canonical state),
    add64_denote _ _ (scale1_canonical _ state.c1) (sum16_64_canonical state),
    add64_denote _ _ (scale2_canonical _ state.c2) (sum16_64_canonical state),
    add64_denote _ _ (scale3_canonical _ state.c3) (sum16_64_canonical state),
    add64_denote _ _ (scale4_canonical _ state.c4) (sum16_64_canonical state),
    add64_denote _ _ (scale5_canonical _ state.c5) (sum16_64_canonical state),
    add64_denote _ _ (scale6_canonical _ state.c6) (sum16_64_canonical state),
    add64_denote _ _ (scale7_canonical _ state.c7) (sum16_64_canonical state),
    add64_denote _ _ (scale8_canonical _ state.c8) (sum16_64_canonical state),
    add64_denote _ _ (scale9_canonical _ state.c9) (sum16_64_canonical state),
    add64_denote _ _ (scale10_canonical _ state.c10) (sum16_64_canonical state),
    add64_denote _ _ (scale11_canonical _ state.c11) (sum16_64_canonical state),
    add64_denote _ _ (scale12_canonical _ state.c12) (sum16_64_canonical state),
    add64_denote _ _ (scale13_canonical _ state.c13) (sum16_64_canonical state),
    add64_denote _ _ (scale14_canonical _ state.c14) (sum16_64_canonical state),
    add64_denote _ _ (scale15_canonical _ state.c15) (sum16_64_canonical state),
    scale0_denote _ state.c0,
    scale1_denote _ state.c1,
    scale2_denote _ state.c2,
    scale3_denote _ state.c3,
    scale4_denote _ state.c4,
    scale5_denote _ state.c5,
    scale6_denote _ state.c6,
    scale7_denote _ state.c7,
    scale8_denote _ state.c8,
    scale9_denote _ state.c9,
    scale10_denote _ state.c10,
    scale11_denote _ state.c11,
    scale12_denote _ state.c12,
    scale13_denote _ state.c13,
    scale14_denote _ state.c14,
    scale15_denote _ state.c15,
    sum16_64_denote]
  simp [Poseidon2.internalLayer, Poseidon2.width, List.range_succ]

@[noinline] def partialRound64 (state : State64) (constant : UInt64) : State64 :=
  internalLayer64 { state with
    x0 := sbox64 (add64 state.x0 constant)
    canonical := ⟨sbox64_canonical _, state.c1, state.c2, state.c3,
      state.c4, state.c5, state.c6, state.c7, state.c8, state.c9,
      state.c10, state.c11, state.c12, state.c13, state.c14, state.c15⟩ }

theorem partialRound64_denote (constant : UInt64)
    (constantCanonical : constant.toNat < fieldModulus)
    (state : State64) :
    (partialRound64 state constant).denote =
      Poseidon2.internalLayer
        (Poseidon2.sbox (state.x0.denote + constant.denote) ::
          state.denote.drop 1) := by
  rw [partialRound64, internalLayer64_denote]
  simp only [denote]
  rw [sbox64_denote _ (add64_canonical _ _ state.c0 constantCanonical),
    add64_denote _ _ state.c0 constantCanonical]
  rfl

end State64

end NightstreamFPrime.Export.NativePoseidon2
