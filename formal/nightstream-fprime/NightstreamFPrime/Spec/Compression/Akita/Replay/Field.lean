import NightstreamFPrime.Spec.Compression.Akita.Trace

/-! Computable `Z_q` and `E = Z_q[u]/(u² − 2)` for `q = 2^64 − 59` on words, for the replay of
the transcript of the pinned Akita verifier (ePrint 2026/1983, revision of 2026-11-23, Section
3.1). An element of `Z_q` is a `UInt64`, and each operation reduces its natural-number result
modulo `q`. An element of `E` is the pair of its coordinates in the basis `(1, u)`, the order of
the wire (`c0` then `c1`).

- `Fq.toZq` and `Ext.toE` are the maps to `Zq` and to `Quadratic`, the extension field of the
  specification (`Akita/Trace.lean`). They map each operation to the operation of the
  specification (`toZq_add`, `toE_mul` and the others), so `Ext.toE` is a ring homomorphism. An
  extension `E` with `Degree2 E` gets the element `Degree2.coords.symm (toE z)`. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita.Replay

/-- An element of `Z_q` as a word. The word of each result of an operation is below `q`. -/
structure Fq where
  word : UInt64
deriving DecidableEq, Inhabited, Repr

namespace Fq

def ofNat (n : ℕ) : Fq := ⟨UInt64.ofNat (n % fieldModulus)⟩

def zero : Fq := ⟨0⟩

def one : Fq := ⟨1⟩

def add (a b : Fq) : Fq := ofNat (a.word.toNat + b.word.toNat)

def neg (a : Fq) : Fq := ofNat (fieldModulus - a.word.toNat % fieldModulus)

def sub (a b : Fq) : Fq := add a (neg b)

def mul (a b : Fq) : Fq := ofNat (a.word.toNat * b.word.toNat)

/-- The decoder of the wire: a word is an element when it is below `q`. -/
def ofWord? (word : UInt64) : Option Fq :=
  if word.toNat < fieldModulus then some ⟨word⟩ else none

def toZq (a : Fq) : Zq := (a.word.toNat : Zq)

theorem toNat_ofNat (n : ℕ) : (ofNat n).word.toNat = n % fieldModulus := by
  have below : n % fieldModulus < 2 ^ 64 :=
    lt_trans (Nat.mod_lt _ (by decide)) (by decide)
  simp only [ofNat, UInt64.toNat_ofNat', Nat.mod_eq_of_lt below]

theorem toZq_ofNat (n : ℕ) : (ofNat n).toZq = (n : Zq) := by
  rw [toZq, toNat_ofNat, ZMod.natCast_mod]

theorem ofNat_canonical (n : ℕ) : (ofNat n).word.toNat < fieldModulus := by
  rw [toNat_ofNat]
  exact Nat.mod_lt _ (by decide)

theorem toZq_zero : zero.toZq = 0 := by
  simp [toZq, zero]

theorem toZq_one : one.toZq = 1 := by
  simp [toZq, one]

theorem toZq_add (a b : Fq) : (add a b).toZq = a.toZq + b.toZq := by
  rw [add, toZq_ofNat, Nat.cast_add]
  rfl

theorem toZq_neg (a : Fq) : (neg a).toZq = -a.toZq := by
  rw [neg, toZq_ofNat, Nat.cast_sub (Nat.mod_lt _ (by decide)).le, ZMod.natCast_self,
    ZMod.natCast_mod, zero_sub]
  rfl

theorem toZq_sub (a b : Fq) : (sub a b).toZq = a.toZq - b.toZq := by
  rw [sub, toZq_add, toZq_neg, sub_eq_add_neg]

theorem toZq_mul (a b : Fq) : (mul a b).toZq = a.toZq * b.toZq := by
  rw [mul, toZq_ofNat, Nat.cast_mul]
  rfl

theorem ofWord?_canonical {word : UInt64} {a : Fq} (decoded : ofWord? word = some a) :
    a.word = word ∧ word.toNat < fieldModulus := by
  unfold ofWord? at decoded
  split at decoded
  · cases decoded
    exact ⟨rfl, by assumption⟩
  · cases decoded

end Fq

/-- An element `c0 + c1·u` of `E`. -/
structure Ext where
  c0 : Fq
  c1 : Fq
deriving DecidableEq, Inhabited, Repr

namespace Ext

def zero : Ext := ⟨Fq.zero, Fq.zero⟩

def one : Ext := ⟨Fq.one, Fq.zero⟩

def ofBase (a : Fq) : Ext := ⟨a, Fq.zero⟩

def add (a b : Ext) : Ext := ⟨a.c0.add b.c0, a.c1.add b.c1⟩

def neg (a : Ext) : Ext := ⟨a.c0.neg, a.c1.neg⟩

def sub (a b : Ext) : Ext := ⟨a.c0.sub b.c0, a.c1.sub b.c1⟩

/-- `(a0 + a1·u)(b0 + b1·u) = (a0·b0 + 2·a1·b1) + (a0·b1 + a1·b0)·u`. -/
def mul (a b : Ext) : Ext :=
  ⟨(a.c0.mul b.c0).add ((Fq.ofNat 2).mul (a.c1.mul b.c1)),
    (a.c0.mul b.c1).add (a.c1.mul b.c0)⟩

instance : Add Ext := ⟨add⟩
instance : Sub Ext := ⟨sub⟩
instance : Mul Ext := ⟨mul⟩
instance : Neg Ext := ⟨neg⟩

def toE (a : Ext) : Quadratic := ⟨a.c0.toZq, a.c1.toZq⟩

theorem toE_zero : zero.toE = 0 := by
  ext <;> simp [toE, zero, Fq.toZq_zero]

theorem toE_one : one.toE = 1 := by
  ext
  · simp [toE, one, Fq.toZq_one, QuadraticAlgebra.re_one]
  · simp [toE, one, Fq.toZq_zero, QuadraticAlgebra.im_one]

theorem toE_ofBase (a : Fq) : (ofBase a).toE = algebraMap Zq Quadratic a.toZq := by
  ext
  · simp [toE, ofBase]
  · simp [toE, ofBase, Fq.toZq_zero]

theorem toE_add (a b : Ext) : (a + b).toE = a.toE + b.toE := by
  show (add a b).toE = _
  ext <;> simp only [toE, add, Fq.toZq_add, QuadraticAlgebra.re_add, QuadraticAlgebra.im_add]

theorem toE_neg (a : Ext) : (-a).toE = -a.toE := by
  show (neg a).toE = _
  ext <;> simp only [toE, neg, Fq.toZq_neg, QuadraticAlgebra.re_neg, QuadraticAlgebra.im_neg]

theorem toE_sub (a b : Ext) : (a - b).toE = a.toE - b.toE := by
  show (sub a b).toE = _
  ext <;> simp only [toE, sub, Fq.toZq_sub, QuadraticAlgebra.re_sub, QuadraticAlgebra.im_sub]

theorem toE_mul (a b : Ext) : (a * b).toE = a.toE * b.toE := by
  have two : (Fq.ofNat 2).toZq = 2 := by rw [Fq.toZq_ofNat]; rfl
  show (mul a b).toE = _
  ext
  · simp only [toE, mul, Fq.toZq_add, Fq.toZq_mul, two, QuadraticAlgebra.re_mul]
    ring
  · simp only [toE, mul, Fq.toZq_add, Fq.toZq_mul, QuadraticAlgebra.im_mul]
    ring

theorem toE_injective {a b : Ext} (ha0 : a.c0.word.toNat < fieldModulus)
    (ha1 : a.c1.word.toNat < fieldModulus) (hb0 : b.c0.word.toNat < fieldModulus)
    (hb1 : b.c1.word.toNat < fieldModulus) (same : a.toE = b.toE) : a = b := by
  have word {x y : Fq} (hx : x.word.toNat < fieldModulus) (hy : y.word.toNat < fieldModulus)
      (equal : x.toZq = y.toZq) : x = y := by
    have := congrArg ZMod.val equal
    simp only [Fq.toZq, ZMod.val_natCast, Nat.mod_eq_of_lt hx, Nat.mod_eq_of_lt hy] at this
    cases x; cases y
    simp only [Fq.mk.injEq]
    exact UInt64.toNat_inj.mp this
  have h0 := congrArg QuadraticAlgebra.re same
  have h1 := congrArg QuadraticAlgebra.im same
  cases a; cases b
  simp only [mk.injEq]
  exact ⟨word ha0 hb0 h0, word ha1 hb1 h1⟩

end Ext

end NightstreamFPrime.Spec.Compression.Akita.Replay
