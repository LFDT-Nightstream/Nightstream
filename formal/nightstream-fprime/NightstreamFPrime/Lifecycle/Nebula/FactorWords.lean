import NightstreamFPrime.Lifecycle.Nebula.WitnessWords

/-! Owns the auxiliary words of a memory step (SPEC.md section 7.3): the
counter `cnt_j` of each operation slot, the four start products, and the
factor words of each fingerprint. With them, each row of a fingerprint step
reads its bit chunks once and has one product, so it lowers to one R1CS row
and no fresh column. It holds the honest values of the words, their decoding
from a word vector, and the field algebra that ties them to the products of
`PolyRows`. It does not own any row. -/

namespace NightstreamFPrime.Lifecycle.Nebula

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Nebula
open Fin.CommRing

/-- The factor words of the fingerprint `f(t, g, v)`: the two coefficients of
`f`. Each has one product with the word `v + 2^32 · t`. -/
def fingerprintWords (η1 η2 : K) (t g v : F) : Fin 2 → F :=
  ![η1.c0 * (v + natWord (2 ^ 32) * t) + (g - η2.c0), η1.c1 * (v + natWord (2 ^ 32) * t) - η2.c1]

/-- The factor words of an ungated fingerprint step over the product `prod`
(rows S2 and S3): the fingerprint words, then the cross products
`prod.c1 · f.c1` and `prod.c1 · f.c0` of `prod · f`. -/
def scanFactorWords (η1 η2 prod : K) (t g v : F) : Fin 4 → F :=
  let f := fingerprintWords η1 η2 t g v
  ![f 0, f 1, prod.c1 * f 1, prod.c1 * f 0]

/-- The factor words of a gated fingerprint step (rows O8 and O9): the
fingerprint words, the gate `pad + (1 − pad) · f`, then the cross products of
`prod` and the gate. -/
def opsFactorWords (η1 η2 prod : K) (pad t g v : F) : Fin 6 → F :=
  let f := fingerprintWords η1 η2 t g v
  let g0 := (1 - pad) * f 0 + pad
  let g1 := (1 - pad) * f 1
  ![f 0, f 1, g0, g1, prod.c1 * g1, prod.c1 * g0]

/-! ### Field algebra -/

theorem fingerprintWords_eq (η1 η2 : K) (t g v : F) :
    kOf (fingerprintWords η1 η2 t g v 0) (fingerprintWords η1 η2 t g v 1) =
      fingerprintK η1 η2 t g v := by
  simp only [fingerprintWords, fingerprintK, kOf, embed, K.sub, K.add, K.mul, K.mk.injEq, Matrix.cons_val]
  constructor <;> ring

theorem gate_eq (pad : F) (f : K) : kOf ((1 - pad) * f.c0 + pad) ((1 - pad) * f.c1) = gatedK pad f := by
  simp only [gatedK, kOf, embed, K.add, K.mul, K.mk.injEq]
  constructor <;> ring

/-- A product row `next = prod · x` in coordinates, with its two cross products. -/
theorem product_iff (prod x next : K) (m n : F) (cross : m = prod.c1 * x.c1 ∧ n = prod.c1 * x.c0) :
    (next.c0 = prod.c0 * x.c0 + 7 * m ∧ next.c1 = prod.c0 * x.c1 + n) ↔ next = K.mul prod x := by
  obtain ⟨rfl, rfl⟩ := cross
  cases next
  simp only [K.mul, K.mk.injEq]
  constructor
  · rintro ⟨h0, h1⟩
    exact ⟨h0.trans (by ring), h1.trans (by ring)⟩
  · rintro ⟨h0, h1⟩
    exact ⟨h0.trans (by ring), h1.trans (by ring)⟩

/-- The equations of an ungated fingerprint step (`Rows.factorRows`) hold
exactly for the factor words and the product `next = prod · f`. -/
theorem scanFactorWords_iff (η1 η2 prod next : K) (t g v : F) (a : Fin 4 → F) :
    (a 0 = η1.c0 * (v + natWord (2 ^ 32) * t) + (g - η2.c0) ∧
      a 1 = η1.c1 * (v + natWord (2 ^ 32) * t) - η2.c1 ∧
      a 2 = prod.c1 * a 1 ∧ a 3 = prod.c1 * a 0 ∧
      next.c0 = prod.c0 * a 0 + 7 * a 2 ∧ next.c1 = prod.c0 * a 1 + a 3) ↔
    (a = scanFactorWords η1 η2 prod t g v ∧ next = K.mul prod (fingerprintK η1 η2 t g v)) := by
  rw [← fingerprintWords_eq]
  constructor
  · rintro ⟨h0, h1, h2, h3, n0, n1⟩
    have words : a = scanFactorWords η1 η2 prod t g v := by
      funext c
      fin_cases c <;> simp [scanFactorWords, fingerprintWords, h0, h1, h2, h3]
    refine ⟨words, ?_⟩
    subst words
    exact (product_iff prod _ next _ _ ⟨rfl, rfl⟩).mp ⟨n0, n1⟩
  · rintro ⟨rfl, rfl⟩
    have := (product_iff prod (kOf (fingerprintWords η1 η2 t g v 0) (fingerprintWords η1 η2 t g v 1))
      _ (prod.c1 * fingerprintWords η1 η2 t g v 1) (prod.c1 * fingerprintWords η1 η2 t g v 0)
      ⟨rfl, rfl⟩).mpr rfl
    exact ⟨rfl, rfl, rfl, rfl, this.1, this.2⟩

/-- The equations of a gated fingerprint step (`Rows.gatedRows`) hold exactly
for the factor words and the product `next = prod · (pad + (1 − pad) · f)`. -/
theorem opsFactorWords_iff (η1 η2 prod next : K) (pad t g v : F) (a : Fin 6 → F) :
    (a 0 = η1.c0 * (v + natWord (2 ^ 32) * t) + (g - η2.c0) ∧
      a 1 = η1.c1 * (v + natWord (2 ^ 32) * t) - η2.c1 ∧
      a 2 = (1 - pad) * a 0 + pad ∧ a 3 = (1 - pad) * a 1 ∧
      a 4 = prod.c1 * a 3 ∧ a 5 = prod.c1 * a 2 ∧
      next.c0 = prod.c0 * a 2 + 7 * a 4 ∧ next.c1 = prod.c0 * a 3 + a 5) ↔
    (a = opsFactorWords η1 η2 prod pad t g v ∧
      next = K.mul prod (gatedK pad (fingerprintK η1 η2 t g v))) := by
  rw [← fingerprintWords_eq, ← gate_eq]
  constructor
  · rintro ⟨h0, h1, h2, h3, h4, h5, n0, n1⟩
    have words : a = opsFactorWords η1 η2 prod pad t g v := by
      funext c
      fin_cases c <;> simp [opsFactorWords, fingerprintWords, h0, h1, h2, h3, h4, h5]
    refine ⟨words, ?_⟩
    subst words
    exact (product_iff prod _ next _ _ ⟨rfl, rfl⟩).mp ⟨n0, n1⟩
  · rintro ⟨rfl, rfl⟩
    have := (product_iff prod
      (kOf ((1 - pad) * (kOf (fingerprintWords η1 η2 t g v 0) (fingerprintWords η1 η2 t g v 1)).c0 + pad)
        ((1 - pad) * (kOf (fingerprintWords η1 η2 t g v 0) (fingerprintWords η1 η2 t g v 1)).c1))
      _ _ _ ⟨rfl, rfl⟩).mpr rfl
    exact ⟨rfl, rfl, rfl, rfl, rfl, rfl, this.1, this.2⟩

/-! ### The auxiliary words of a step -/

/-- The auxiliary words of a step, in the order of `Words.cnt` to
`Words.scanFactor`. -/
structure AuxWords (p : Geometry) where
  cnt : Fin p.bOps → F
  start : Fin 4 → Fin 2 → F
  ops : Fin p.bOps → Fin 2 → Fin 6 → F
  scan : Fin p.bScan → Fin 2 → Fin 4 → F

/-- The auxiliary words of a word vector. -/
def AuxWords.ofWords (p : Geometry) (v : ℕ → F) : AuxWords p where
  cnt j := v (Words.cnt p j)
  start s c := v (Words.start p s c)
  ops j pair c := v (Words.opsFactor p j pair c)
  scan j pair c := v (Words.scanFactor p j pair c)

namespace StepWitness

variable {p : Geometry} {a : AppShape} (w : StepWitness p a)

/-- The timestamp of fingerprint `pair` of slot `j`: `rt_j`, then `wt_j`. -/
def opsStamp (j : Fin p.bOps) (pair : Fin 2) : F := if pair = 0 then w.rt j else w.wt j

/-- The value of fingerprint `pair` of slot `j`: `v_r`, then `v_w`. -/
def opsValue (j : Fin p.bOps) (pair : Fin 2) : F := if pair = 0 then w.vr j else w.vw j

/-- The scan slots of fingerprint `pair`: IS, then FS. -/
def scanSlots (pair : Fin 2) : Fin p.bScan → ScanSlotBits p := if pair = 0 then w.initial else w.final

/-- The honest auxiliary words of a step witness. -/
def auxWords : AuxWords p where
  cnt j := w.cntBefore (j.val + 1)
  start s c := if c = 0 then (w.startProduct (7 + 2 * s.val)).c0 else (w.startProduct (7 + 2 * s.val)).c1
  ops j pair := opsFactorWords w.eta1 w.eta2 (w.opsProductAfter pair j.val) (w.pad j)
    (w.opsStamp j pair) (w.globalIndex j) (w.opsValue j pair)
  scan j pair := scanFactorWords w.eta1 w.eta2 (w.scanProductAfter pair j.val)
    (bitsWord (w.scanSlots pair j).stamp) (w.scanIndex j) (bitsWord (w.scanSlots pair j).value)

theorem cntBefore_succ (j : Fin p.bOps) : w.cntBefore (j.val + 1) = w.cntBefore j.val + (1 - w.pad j) := by
  simp only [cntBefore]
  rw [List.take_add_one, List.sum_append, List.getElem?_ofFn]
  simp [j.isLt]

end StepWitness

/-- The auxiliary words of a word vector are the honest words of its decoded
witness. -/
def AuxHonest (p : Geometry) (a : AppShape) (v : ℕ → F) : Prop :=
  AuxWords.ofWords p v = (StepWitness.ofWords p a v).auxWords

end NightstreamFPrime.Lifecycle.Nebula
