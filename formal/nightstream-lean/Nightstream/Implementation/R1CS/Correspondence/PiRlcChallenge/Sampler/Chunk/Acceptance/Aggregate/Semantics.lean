import Nightstream.Implementation.R1CS.Correspondence.PiRlcChallenge.Sampler.Chunk
import Nightstream.Implementation.R1CS.Correspondence.PiRlcChallenge.Sampler.Chunk.Mod5.PackedRows

/-!
Independent source and product-tree semantics for one sixteen-bit sampler chunk.

Owns: the authoritative source-bit view, Boolean carrier meaning, balanced
fourteen-edge tree, canonical tree materialization, root products, the
all-ones rejection condition, and correspondence with the independent
production-alphabet verifier.

Does not own: paired equations, aggregate exactness, generated artifacts,
Rust emission, production placement, fixed selectors, the 960-chunk image,
cost totals, or row-removal authority.

Emits constraints: no.

| Exact Rust stage path | Zero-cost semantic ownership | Principal result |
|---|---|---|
| `nifs.pi_rlc.challenge.sampler.chunk.accept.packed.tree_bit_pairs` | source/output Boolean meaning | `sourceBits_are_boolean` |
| `nifs.pi_rlc.challenge.sampler.chunk.accept.packed.product_aggregate` | fourteen products form one balanced tree | `productTreeMeaning_iff_equations` |
| `nifs.pi_rlc.challenge.sampler.chunk.accept.packed.root_binding` | roots, rejection bucket, and verifier authority | `sourceAcceptanceMeaning_iff_verifier` |
-/

namespace Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Chunk.Acceptance

open Nightstream.Implementation.R1CS
open Nightstream.SuperNeo.Folding.Nifs.NonInteractive.PiRlcSampler
open Mod5

/-- The sixteen authoritative source coordinates of one active chunk. -/
def sourceBits (assignment : Nat → Nat) (chunk : Nat) : Fin 16 → GateField :=
  fun index =>
    fieldResidue
      (assignment (ChunkRows.sourceBitCol chunk index.val))

/-- Semantic bit membership used after decoding the paired equations. -/
def FieldBit (value : GateField) : Prop :=
  value = 0 ∨ value = 1

theorem sourceBits_are_boolean
    {assignment : Nat → Nat} {chunk : Nat}
    (bits : BitsBoolean assignment chunk) :
    ∀ index, FieldBit (sourceBits assignment chunk index) := by
  intro index
  have bound := bits index.val index.isLt
  have cases :
      assignment (ChunkRows.sourceBitCol chunk index.val) = 0 ∨
        assignment (ChunkRows.sourceBitCol chunk index.val) = 1 := by
    omega
  rcases cases with zero | one
  · left
    unfold sourceBits
    rw [zero]
    apply Fin.ext
    native_decide
  · right
    unfold sourceBits
    rw [one]
    apply Fin.ext
    native_decide

/-! ## Balanced fourteen-edge tree -/

abbrev ProductTreeOutputs := Fin 14 → GateField

def productTreeLeft
    (bits : Fin 16 → GateField) (outputs : ProductTreeOutputs) :
    Fin 14 → GateField
  | ⟨0, _⟩ => bits 0
  | ⟨1, _⟩ => bits 2
  | ⟨2, _⟩ => bits 4
  | ⟨3, _⟩ => bits 6
  | ⟨4, _⟩ => outputs 0
  | ⟨5, _⟩ => outputs 2
  | ⟨6, _⟩ => outputs 4
  | ⟨7, _⟩ => bits 8
  | ⟨8, _⟩ => bits 10
  | ⟨9, _⟩ => bits 12
  | ⟨10, _⟩ => bits 14
  | ⟨11, _⟩ => outputs 7
  | ⟨12, _⟩ => outputs 9
  | _ => outputs 11

def productTreeRight
    (bits : Fin 16 → GateField) (outputs : ProductTreeOutputs) :
    Fin 14 → GateField
  | ⟨0, _⟩ => bits 1
  | ⟨1, _⟩ => bits 3
  | ⟨2, _⟩ => bits 5
  | ⟨3, _⟩ => bits 7
  | ⟨4, _⟩ => outputs 1
  | ⟨5, _⟩ => outputs 3
  | ⟨6, _⟩ => outputs 5
  | ⟨7, _⟩ => bits 9
  | ⟨8, _⟩ => bits 11
  | ⟨9, _⟩ => bits 13
  | ⟨10, _⟩ => bits 15
  | ⟨11, _⟩ => outputs 8
  | ⟨12, _⟩ => outputs 10
  | _ => outputs 12

def ProductTreeMeaning
    (bits : Fin 16 → GateField) (outputs : ProductTreeOutputs) : Prop :=
  ∀ index,
    outputs index =
      productTreeLeft bits outputs index *
        productTreeRight bits outputs index

def ProductTreeEquations
    (bits : Fin 16 → GateField) (outputs : ProductTreeOutputs) : Prop :=
  outputs 0 = bits 0 * bits 1 ∧
    outputs 1 = bits 2 * bits 3 ∧
    outputs 2 = bits 4 * bits 5 ∧
    outputs 3 = bits 6 * bits 7 ∧
    outputs 4 = outputs 0 * outputs 1 ∧
    outputs 5 = outputs 2 * outputs 3 ∧
    outputs 6 = outputs 4 * outputs 5 ∧
    outputs 7 = bits 8 * bits 9 ∧
    outputs 8 = bits 10 * bits 11 ∧
    outputs 9 = bits 12 * bits 13 ∧
    outputs 10 = bits 14 * bits 15 ∧
    outputs 11 = outputs 7 * outputs 8 ∧
    outputs 12 = outputs 9 * outputs 10 ∧
    outputs 13 = outputs 11 * outputs 12

theorem fin14_all
    {predicate : Fin 14 → Prop}
    (h0 : predicate 0) (h1 : predicate 1)
    (h2 : predicate 2) (h3 : predicate 3)
    (h4 : predicate 4) (h5 : predicate 5)
    (h6 : predicate 6) (h7 : predicate 7)
    (h8 : predicate 8) (h9 : predicate 9)
    (h10 : predicate 10) (h11 : predicate 11)
    (h12 : predicate 12) (h13 : predicate 13) :
    ∀ index, predicate index := by
  intro index
  refine Fin.cases h0 ?_ index
  intro index1
  refine Fin.cases h1 ?_ index1
  intro index2
  refine Fin.cases h2 ?_ index2
  intro index3
  refine Fin.cases h3 ?_ index3
  intro index4
  refine Fin.cases h4 ?_ index4
  intro index5
  refine Fin.cases h5 ?_ index5
  intro index6
  refine Fin.cases h6 ?_ index6
  intro index7
  refine Fin.cases h7 ?_ index7
  intro index8
  refine Fin.cases h8 ?_ index8
  intro index9
  refine Fin.cases h9 ?_ index9
  intro index10
  refine Fin.cases h10 ?_ index10
  intro index11
  refine Fin.cases h11 ?_ index11
  intro index12
  refine Fin.cases h12 ?_ index12
  intro index13
  refine Fin.cases h13 ?_ index13
  intro impossible
  exact Fin.elim0 impossible

private theorem fin16_all
    {predicate : Fin 16 → Prop}
    (h0 : predicate 0) (h1 : predicate 1)
    (h2 : predicate 2) (h3 : predicate 3)
    (h4 : predicate 4) (h5 : predicate 5)
    (h6 : predicate 6) (h7 : predicate 7)
    (h8 : predicate 8) (h9 : predicate 9)
    (h10 : predicate 10) (h11 : predicate 11)
    (h12 : predicate 12) (h13 : predicate 13)
    (h14 : predicate 14) (h15 : predicate 15) :
    ∀ index, predicate index := by
  intro index
  refine Fin.cases h0 ?_ index
  intro index1
  refine Fin.cases h1 ?_ index1
  intro index2
  refine Fin.cases h2 ?_ index2
  intro index3
  refine Fin.cases h3 ?_ index3
  intro index4
  refine Fin.cases h4 ?_ index4
  intro index5
  refine Fin.cases h5 ?_ index5
  intro index6
  refine Fin.cases h6 ?_ index6
  intro index7
  refine Fin.cases h7 ?_ index7
  intro index8
  refine Fin.cases h8 ?_ index8
  intro index9
  refine Fin.cases h9 ?_ index9
  intro index10
  refine Fin.cases h10 ?_ index10
  intro index11
  refine Fin.cases h11 ?_ index11
  intro index12
  refine Fin.cases h12 ?_ index12
  intro index13
  refine Fin.cases h13 ?_ index13
  intro index14
  refine Fin.cases h14 ?_ index14
  intro index15
  refine Fin.cases h15 ?_ index15
  intro impossible
  exact Fin.elim0 impossible

theorem productTreeMeaning_iff_equations
    (bits : Fin 16 → GateField) (outputs : ProductTreeOutputs) :
    ProductTreeMeaning bits outputs ↔ ProductTreeEquations bits outputs := by
  constructor
  · intro meaning
    exact ⟨meaning 0, meaning 1, meaning 2, meaning 3, meaning 4,
      meaning 5, meaning 6, meaning 7, meaning 8, meaning 9,
      meaning 10, meaning 11, meaning 12, meaning 13⟩
  · rintro ⟨h0, h1, h2, h3, h4, h5, h6,
      h7, h8, h9, h10, h11, h12, h13⟩ index
    exact fin14_all
      (predicate := fun index =>
        outputs index = productTreeLeft bits outputs index *
          productTreeRight bits outputs index)
      (by simpa [productTreeLeft, productTreeRight] using h0)
      (by simpa [productTreeLeft, productTreeRight] using h1)
      (by simpa [productTreeLeft, productTreeRight] using h2)
      (by simpa [productTreeLeft, productTreeRight] using h3)
      (by simpa [productTreeLeft, productTreeRight] using h4)
      (by simpa [productTreeLeft, productTreeRight] using h5)
      (by simpa [productTreeLeft, productTreeRight] using h6)
      (by simpa [productTreeLeft, productTreeRight] using h7)
      (by simpa [productTreeLeft, productTreeRight] using h8)
      (by simpa [productTreeLeft, productTreeRight] using h9)
      (by simpa [productTreeLeft, productTreeRight] using h10)
      (by simpa [productTreeLeft, productTreeRight] using h11)
      (by simpa [productTreeLeft, productTreeRight] using h12)
      (by simpa [productTreeLeft, productTreeRight] using h13)
      index

theorem fieldBit_mul
    {left right : GateField} (leftBit : FieldBit left)
    (rightBit : FieldBit right) : FieldBit (left * right) := by
  rcases leftBit with leftZero | leftOne
  · left
    rw [leftZero]
    apply Fin.ext
    simp only [Fin.val_mul, Fin.val_zero, Nat.zero_mul, Nat.zero_mod]
  rcases rightBit with rightZero | rightOne
  · left
    rw [rightZero]
    apply Fin.ext
    simp only [Fin.val_mul, Fin.val_zero, Nat.mul_zero, Nat.zero_mod]
  · right
    rw [leftOne, rightOne, gateField_one_mul]

theorem productTreeOperands_boolean
    {bits : Fin 16 → GateField} {outputs : ProductTreeOutputs}
    (bitSource : ∀ index, FieldBit (bits index))
    (bitOutputs : ∀ index, FieldBit (outputs index)) :
    ∀ index,
      FieldBit (productTreeLeft bits outputs index) ∧
        FieldBit (productTreeRight bits outputs index) := by
  exact fin14_all
    ⟨bitSource 0, bitSource 1⟩
    ⟨bitSource 2, bitSource 3⟩
    ⟨bitSource 4, bitSource 5⟩
    ⟨bitSource 6, bitSource 7⟩
    ⟨bitOutputs 0, bitOutputs 1⟩
    ⟨bitOutputs 2, bitOutputs 3⟩
    ⟨bitOutputs 4, bitOutputs 5⟩
    ⟨bitSource 8, bitSource 9⟩
    ⟨bitSource 10, bitSource 11⟩
    ⟨bitSource 12, bitSource 13⟩
    ⟨bitSource 14, bitSource 15⟩
    ⟨bitOutputs 7, bitOutputs 8⟩
    ⟨bitOutputs 9, bitOutputs 10⟩
    ⟨bitOutputs 11, bitOutputs 12⟩

theorem productTreeMeaning_outputs_boolean
    {bits : Fin 16 → GateField} {outputs : ProductTreeOutputs}
    (bitSource : ∀ index, FieldBit (bits index))
    (meaning : ProductTreeMeaning bits outputs) :
    ∀ index, FieldBit (outputs index) := by
  rcases (productTreeMeaning_iff_equations bits outputs).mp meaning with
    ⟨h0, h1, h2, h3, h4, h5, h6,
      h7, h8, h9, h10, h11, h12, h13⟩
  have b0 : FieldBit (outputs 0) := h0 ▸ fieldBit_mul (bitSource 0) (bitSource 1)
  have b1 : FieldBit (outputs 1) := h1 ▸ fieldBit_mul (bitSource 2) (bitSource 3)
  have b2 : FieldBit (outputs 2) := h2 ▸ fieldBit_mul (bitSource 4) (bitSource 5)
  have b3 : FieldBit (outputs 3) := h3 ▸ fieldBit_mul (bitSource 6) (bitSource 7)
  have b4 : FieldBit (outputs 4) := h4 ▸ fieldBit_mul b0 b1
  have b5 : FieldBit (outputs 5) := h5 ▸ fieldBit_mul b2 b3
  have b6 : FieldBit (outputs 6) := h6 ▸ fieldBit_mul b4 b5
  have b7 : FieldBit (outputs 7) := h7 ▸ fieldBit_mul (bitSource 8) (bitSource 9)
  have b8 : FieldBit (outputs 8) := h8 ▸ fieldBit_mul (bitSource 10) (bitSource 11)
  have b9 : FieldBit (outputs 9) := h9 ▸ fieldBit_mul (bitSource 12) (bitSource 13)
  have b10 : FieldBit (outputs 10) := h10 ▸ fieldBit_mul (bitSource 14) (bitSource 15)
  have b11 : FieldBit (outputs 11) := h11 ▸ fieldBit_mul b7 b8
  have b12 : FieldBit (outputs 12) := h12 ▸ fieldBit_mul b9 b10
  have b13 : FieldBit (outputs 13) := h13 ▸ fieldBit_mul b11 b12
  exact fin14_all b0 b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13

/-! ## Tree roots and complete local relation -/

/-- Product of the low eight source bits in the same balanced shape as the
tree. -/
def lowHalfProduct (bits : Fin 16 → GateField) : GateField :=
  ((bits 0 * bits 1) * (bits 2 * bits 3)) *
    ((bits 4 * bits 5) * (bits 6 * bits 7))

/-- Product of the high eight source bits in the same balanced shape as the
tree. -/
def highHalfProduct (bits : Fin 16 → GateField) : GateField :=
  ((bits 8 * bits 9) * (bits 10 * bits 11)) *
    ((bits 12 * bits 13) * (bits 14 * bits 15))

theorem productTreeMeaning_roots
    {bits : Fin 16 → GateField} {outputs : ProductTreeOutputs}
    (meaning : ProductTreeMeaning bits outputs) :
    outputs 6 = lowHalfProduct bits ∧
      outputs 13 = highHalfProduct bits := by
  rcases (productTreeMeaning_iff_equations bits outputs).mp meaning with
    ⟨h0, h1, h2, h3, h4, h5, h6,
      h7, h8, h9, h10, h11, h12, h13⟩
  constructor
  · rw [h6, h4, h5, h0, h1, h2, h3]
    rfl
  · rw [h13, h11, h12, h7, h8, h9, h10]
    rfl

/-- Deterministic topological materialization of all fourteen tree outputs. -/
def canonicalProductTreeOutputs
    (bits : Fin 16 → GateField) : ProductTreeOutputs
  | ⟨0, _⟩ => bits 0 * bits 1
  | ⟨1, _⟩ => bits 2 * bits 3
  | ⟨2, _⟩ => bits 4 * bits 5
  | ⟨3, _⟩ => bits 6 * bits 7
  | ⟨4, _⟩ => (bits 0 * bits 1) * (bits 2 * bits 3)
  | ⟨5, _⟩ => (bits 4 * bits 5) * (bits 6 * bits 7)
  | ⟨6, _⟩ => lowHalfProduct bits
  | ⟨7, _⟩ => bits 8 * bits 9
  | ⟨8, _⟩ => bits 10 * bits 11
  | ⟨9, _⟩ => bits 12 * bits 13
  | ⟨10, _⟩ => bits 14 * bits 15
  | ⟨11, _⟩ => (bits 8 * bits 9) * (bits 10 * bits 11)
  | ⟨12, _⟩ => (bits 12 * bits 13) * (bits 14 * bits 15)
  | _ => highHalfProduct bits

theorem canonicalProductTreeOutputs_meaning (bits : Fin 16 → GateField) :
    ProductTreeMeaning bits (canonicalProductTreeOutputs bits) := by
  apply (productTreeMeaning_iff_equations bits _).mpr
  simp [ProductTreeEquations, canonicalProductTreeOutputs,
    lowHalfProduct, highHalfProduct]

theorem productTreeMeaning_unique
    {bits : Fin 16 → GateField} {left right : ProductTreeOutputs}
    (leftMeaning : ProductTreeMeaning bits left)
    (rightMeaning : ProductTreeMeaning bits right) :
    left = right := by
  rcases (productTreeMeaning_iff_equations bits left).mp leftMeaning with
    ⟨hl0, hl1, hl2, hl3, hl4, hl5, hl6,
      hl7, hl8, hl9, hl10, hl11, hl12, hl13⟩
  rcases (productTreeMeaning_iff_equations bits right).mp rightMeaning with
    ⟨hr0, hr1, hr2, hr3, hr4, hr5, hr6,
      hr7, hr8, hr9, hr10, hr11, hr12, hr13⟩
  have h0 : left 0 = right 0 := hl0.trans hr0.symm
  have h1 : left 1 = right 1 := hl1.trans hr1.symm
  have h2 : left 2 = right 2 := hl2.trans hr2.symm
  have h3 : left 3 = right 3 := hl3.trans hr3.symm
  have h4 : left 4 = right 4 := by rw [hl4, hr4, h0, h1]
  have h5 : left 5 = right 5 := by rw [hl5, hr5, h2, h3]
  have h6 : left 6 = right 6 := by rw [hl6, hr6, h4, h5]
  have h7 : left 7 = right 7 := hl7.trans hr7.symm
  have h8 : left 8 = right 8 := hl8.trans hr8.symm
  have h9 : left 9 = right 9 := hl9.trans hr9.symm
  have h10 : left 10 = right 10 := hl10.trans hr10.symm
  have h11 : left 11 = right 11 := by rw [hl11, hr11, h7, h8]
  have h12 : left 12 = right 12 := by rw [hl12, hr12, h9, h10]
  have h13 : left 13 = right 13 := by rw [hl13, hr13, h11, h12]
  funext index
  exact fin14_all (predicate := fun index => left index = right index)
    h0 h1 h2 h3 h4 h5 h6 h7 h8 h9 h10 h11 h12 h13 index

/-! ## Independent source and production-verifier meaning -/

def AllSourceBitsOne (bits : Fin 16 → GateField) : Prop :=
  ∀ index, bits index = 1

/-- Acceptance meaning stated only in terms of the sixteen source bits. -/
def SourceAcceptanceMeaning
    (bits : Fin 16 → GateField) (accept : GateField) : Prop :=
  accept = fieldSub 1 (lowHalfProduct bits * highHalfProduct bits)

/-- Acceptance meaning stated against the independent production alphabet. -/
def VerifierAcceptanceMeaning
    (assignment : Nat → Nat) (chunk : Nat)
    (bits : BitsBoolean assignment chunk) (accept : GateField) : Prop :=
  accept =
    if ProductionAlphabet.verifier.accepts
        (candidate assignment chunk bits) then 1 else 0

private theorem fieldBit_mul_eq_one_iff
    {left right : GateField} (leftBit : FieldBit left)
    (rightBit : FieldBit right) :
    left * right = 1 ↔ left = 1 ∧ right = 1 := by
  rcases leftBit with rfl | rfl <;>
    rcases rightBit with rfl | rfl <;>
    native_decide

private theorem sourceProduct_boolean
    {bits : Fin 16 → GateField}
    (sourceBoolean : ∀ index, FieldBit (bits index)) :
    FieldBit (lowHalfProduct bits * highHalfProduct bits) := by
  apply fieldBit_mul
  · apply fieldBit_mul
    · exact fieldBit_mul
        (fieldBit_mul (sourceBoolean 0) (sourceBoolean 1))
        (fieldBit_mul (sourceBoolean 2) (sourceBoolean 3))
    · exact fieldBit_mul
        (fieldBit_mul (sourceBoolean 4) (sourceBoolean 5))
        (fieldBit_mul (sourceBoolean 6) (sourceBoolean 7))
  · apply fieldBit_mul
    · exact fieldBit_mul
        (fieldBit_mul (sourceBoolean 8) (sourceBoolean 9))
        (fieldBit_mul (sourceBoolean 10) (sourceBoolean 11))
    · exact fieldBit_mul
        (fieldBit_mul (sourceBoolean 12) (sourceBoolean 13))
        (fieldBit_mul (sourceBoolean 14) (sourceBoolean 15))

theorem sourceProduct_eq_one_iff_allSourceBitsOne
    (bits : Fin 16 → GateField)
    (sourceBoolean : ∀ index, FieldBit (bits index)) :
    lowHalfProduct bits * highHalfProduct bits = 1 ↔
      AllSourceBitsOne bits := by
  let pair01 := bits 0 * bits 1
  let pair23 := bits 2 * bits 3
  let pair45 := bits 4 * bits 5
  let pair67 := bits 6 * bits 7
  let pair89 := bits 8 * bits 9
  let pair1011 := bits 10 * bits 11
  let pair1213 := bits 12 * bits 13
  let pair1415 := bits 14 * bits 15
  have pair01Bit : FieldBit pair01 :=
    fieldBit_mul (sourceBoolean 0) (sourceBoolean 1)
  have pair23Bit : FieldBit pair23 :=
    fieldBit_mul (sourceBoolean 2) (sourceBoolean 3)
  have pair45Bit : FieldBit pair45 :=
    fieldBit_mul (sourceBoolean 4) (sourceBoolean 5)
  have pair67Bit : FieldBit pair67 :=
    fieldBit_mul (sourceBoolean 6) (sourceBoolean 7)
  have pair89Bit : FieldBit pair89 :=
    fieldBit_mul (sourceBoolean 8) (sourceBoolean 9)
  have pair1011Bit : FieldBit pair1011 :=
    fieldBit_mul (sourceBoolean 10) (sourceBoolean 11)
  have pair1213Bit : FieldBit pair1213 :=
    fieldBit_mul (sourceBoolean 12) (sourceBoolean 13)
  have pair1415Bit : FieldBit pair1415 :=
    fieldBit_mul (sourceBoolean 14) (sourceBoolean 15)
  let lowLeft := pair01 * pair23
  let lowRight := pair45 * pair67
  let highLeft := pair89 * pair1011
  let highRight := pair1213 * pair1415
  have lowLeftBit : FieldBit lowLeft := fieldBit_mul pair01Bit pair23Bit
  have lowRightBit : FieldBit lowRight := fieldBit_mul pair45Bit pair67Bit
  have highLeftBit : FieldBit highLeft := fieldBit_mul pair89Bit pair1011Bit
  have highRightBit : FieldBit highRight := fieldBit_mul pair1213Bit pair1415Bit
  have lowBit : FieldBit (lowHalfProduct bits) :=
    fieldBit_mul lowLeftBit lowRightBit
  have highBit : FieldBit (highHalfProduct bits) :=
    fieldBit_mul highLeftBit highRightBit
  constructor
  · intro rootOne
    have roots := (fieldBit_mul_eq_one_iff lowBit highBit).mp rootOne
    have lowParts :=
      (fieldBit_mul_eq_one_iff lowLeftBit lowRightBit).mp roots.1
    have highParts :=
      (fieldBit_mul_eq_one_iff highLeftBit highRightBit).mp roots.2
    have lowLeftPairs :=
      (fieldBit_mul_eq_one_iff pair01Bit pair23Bit).mp lowParts.1
    have lowRightPairs :=
      (fieldBit_mul_eq_one_iff pair45Bit pair67Bit).mp lowParts.2
    have highLeftPairs :=
      (fieldBit_mul_eq_one_iff pair89Bit pair1011Bit).mp highParts.1
    have highRightPairs :=
      (fieldBit_mul_eq_one_iff pair1213Bit pair1415Bit).mp highParts.2
    have h01 := (fieldBit_mul_eq_one_iff
      (sourceBoolean 0) (sourceBoolean 1)).mp lowLeftPairs.1
    have h23 := (fieldBit_mul_eq_one_iff
      (sourceBoolean 2) (sourceBoolean 3)).mp lowLeftPairs.2
    have h45 := (fieldBit_mul_eq_one_iff
      (sourceBoolean 4) (sourceBoolean 5)).mp lowRightPairs.1
    have h67 := (fieldBit_mul_eq_one_iff
      (sourceBoolean 6) (sourceBoolean 7)).mp lowRightPairs.2
    have h89 := (fieldBit_mul_eq_one_iff
      (sourceBoolean 8) (sourceBoolean 9)).mp highLeftPairs.1
    have h1011 := (fieldBit_mul_eq_one_iff
      (sourceBoolean 10) (sourceBoolean 11)).mp highLeftPairs.2
    have h1213 := (fieldBit_mul_eq_one_iff
      (sourceBoolean 12) (sourceBoolean 13)).mp highRightPairs.1
    have h1415 := (fieldBit_mul_eq_one_iff
      (sourceBoolean 14) (sourceBoolean 15)).mp highRightPairs.2
    exact fin16_all (predicate := fun index => bits index = 1)
      h01.1 h01.2 h23.1 h23.2 h45.1 h45.2 h67.1 h67.2
      h89.1 h89.2 h1011.1 h1011.2 h1213.1 h1213.2 h1415.1 h1415.2
  · intro allOne
    simp only [lowHalfProduct, highHalfProduct]
    rw [allOne 0, allOne 1, allOne 2, allOne 3,
      allOne 4, allOne 5, allOne 6, allOne 7,
      allOne 8, allOne 9, allOne 10, allOne 11,
      allOne 12, allOne 13, allOne 14, allOne 15]
    apply Fin.ext
    native_decide

private theorem fieldSub_one_one : fieldSub (1 : GateField) 1 = 0 := by
  apply Fin.ext
  native_decide

private theorem fieldSub_one_zero : fieldSub (1 : GateField) 0 = 1 := by
  apply Fin.ext
  native_decide

theorem sourceAcceptanceMeaning_iff_cases
    (bits : Fin 16 → GateField) (accept : GateField)
    (sourceBoolean : ∀ index, FieldBit (bits index)) :
    SourceAcceptanceMeaning bits accept ↔
      (AllSourceBitsOne bits ∧ accept = 0) ∨
        (¬ AllSourceBitsOne bits ∧ accept = 1) := by
  by_cases allOne : AllSourceBitsOne bits
  · have productOne :=
      (sourceProduct_eq_one_iff_allSourceBitsOne bits sourceBoolean).mpr allOne
    rw [SourceAcceptanceMeaning, productOne, fieldSub_one_one]
    simp [allOne]
  · have productBit := sourceProduct_boolean sourceBoolean
    have productNotOne :
        lowHalfProduct bits * highHalfProduct bits ≠ 1 := by
      intro productOne
      exact allOne
        ((sourceProduct_eq_one_iff_allSourceBitsOne
          bits sourceBoolean).mp productOne)
    have productZero : lowHalfProduct bits * highHalfProduct bits = 0 := by
      rcases productBit with zero | one
      · exact zero
      · exact False.elim (productNotOne one)
    rw [SourceAcceptanceMeaning, productZero, fieldSub_one_zero]
    simp [allOne]

private theorem acceptance_range16 :
    List.range 16 =
      [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] := by
  decide

private theorem sourceBit_eq_one_iff
    {assignment : Nat → Nat} {chunk : Nat}
    (bits : BitsBoolean assignment chunk) (index : Fin 16) :
    sourceBits assignment chunk index = 1 ↔
      assignment (ChunkRows.sourceBitCol chunk index.val) = 1 := by
  constructor
  · intro equal
    have values := congrArg Fin.val equal
    have sourceLtField :
        assignment (ChunkRows.sourceBitCol chunk index.val) < goldilocksP := by
      have sourceLe := bits index.val index.isLt
      have bound : 1 < goldilocksP := by decide
      omega
    simp only [sourceBits, fieldResidue] at values
    rw [Nat.mod_eq_of_lt sourceLtField] at values
    simpa using values
  · intro equal
    unfold sourceBits
    rw [equal]
    apply Fin.ext
    native_decide

theorem allSourceBitsOne_iff_rejectionBucket
    {assignment : Nat → Nat} {chunk : Nat}
    (bits : BitsBoolean assignment chunk) :
    AllSourceBitsOne (sourceBits assignment chunk) ↔
      chunkValue assignment chunk = ProductionAlphabet.rejectionBucket := by
  constructor
  · intro allOne
    have a0 := (sourceBit_eq_one_iff bits 0).mp (allOne 0)
    have a1 := (sourceBit_eq_one_iff bits 1).mp (allOne 1)
    have a2 := (sourceBit_eq_one_iff bits 2).mp (allOne 2)
    have a3 := (sourceBit_eq_one_iff bits 3).mp (allOne 3)
    have a4 := (sourceBit_eq_one_iff bits 4).mp (allOne 4)
    have a5 := (sourceBit_eq_one_iff bits 5).mp (allOne 5)
    have a6 := (sourceBit_eq_one_iff bits 6).mp (allOne 6)
    have a7 := (sourceBit_eq_one_iff bits 7).mp (allOne 7)
    have a8 := (sourceBit_eq_one_iff bits 8).mp (allOne 8)
    have a9 := (sourceBit_eq_one_iff bits 9).mp (allOne 9)
    have a10 := (sourceBit_eq_one_iff bits 10).mp (allOne 10)
    have a11 := (sourceBit_eq_one_iff bits 11).mp (allOne 11)
    have a12 := (sourceBit_eq_one_iff bits 12).mp (allOne 12)
    have a13 := (sourceBit_eq_one_iff bits 13).mp (allOne 13)
    have a14 := (sourceBit_eq_one_iff bits 14).mp (allOne 14)
    have a15 := (sourceBit_eq_one_iff bits 15).mp (allOne 15)
    change assignment (ChunkRows.sourceBitCol chunk 0) = 1 at a0
    change assignment (ChunkRows.sourceBitCol chunk 1) = 1 at a1
    change assignment (ChunkRows.sourceBitCol chunk 2) = 1 at a2
    change assignment (ChunkRows.sourceBitCol chunk 3) = 1 at a3
    change assignment (ChunkRows.sourceBitCol chunk 4) = 1 at a4
    change assignment (ChunkRows.sourceBitCol chunk 5) = 1 at a5
    change assignment (ChunkRows.sourceBitCol chunk 6) = 1 at a6
    change assignment (ChunkRows.sourceBitCol chunk 7) = 1 at a7
    change assignment (ChunkRows.sourceBitCol chunk 8) = 1 at a8
    change assignment (ChunkRows.sourceBitCol chunk 9) = 1 at a9
    change assignment (ChunkRows.sourceBitCol chunk 10) = 1 at a10
    change assignment (ChunkRows.sourceBitCol chunk 11) = 1 at a11
    change assignment (ChunkRows.sourceBitCol chunk 12) = 1 at a12
    change assignment (ChunkRows.sourceBitCol chunk 13) = 1 at a13
    change assignment (ChunkRows.sourceBitCol chunk 14) = 1 at a14
    change assignment (ChunkRows.sourceBitCol chunk 15) = 1 at a15
    simp [chunkValue, acceptance_range16, ProductionAlphabet.rejectionBucket,
      a0, a1, a2, a3, a4, a5, a6, a7,
      a8, a9, a10, a11, a12, a13, a14, a15]
  · intro rejected
    have b0 := bits 0 (by decide)
    have b1 := bits 1 (by decide)
    have b2 := bits 2 (by decide)
    have b3 := bits 3 (by decide)
    have b4 := bits 4 (by decide)
    have b5 := bits 5 (by decide)
    have b6 := bits 6 (by decide)
    have b7 := bits 7 (by decide)
    have b8 := bits 8 (by decide)
    have b9 := bits 9 (by decide)
    have b10 := bits 10 (by decide)
    have b11 := bits 11 (by decide)
    have b12 := bits 12 (by decide)
    have b13 := bits 13 (by decide)
    have b14 := bits 14 (by decide)
    have b15 := bits 15 (by decide)
    simp [chunkValue, acceptance_range16,
      ProductionAlphabet.rejectionBucket] at rejected
    have a0 : assignment (ChunkRows.sourceBitCol chunk 0) = 1 := by omega
    have a1 : assignment (ChunkRows.sourceBitCol chunk 1) = 1 := by omega
    have a2 : assignment (ChunkRows.sourceBitCol chunk 2) = 1 := by omega
    have a3 : assignment (ChunkRows.sourceBitCol chunk 3) = 1 := by omega
    have a4 : assignment (ChunkRows.sourceBitCol chunk 4) = 1 := by omega
    have a5 : assignment (ChunkRows.sourceBitCol chunk 5) = 1 := by omega
    have a6 : assignment (ChunkRows.sourceBitCol chunk 6) = 1 := by omega
    have a7 : assignment (ChunkRows.sourceBitCol chunk 7) = 1 := by omega
    have a8 : assignment (ChunkRows.sourceBitCol chunk 8) = 1 := by omega
    have a9 : assignment (ChunkRows.sourceBitCol chunk 9) = 1 := by omega
    have a10 : assignment (ChunkRows.sourceBitCol chunk 10) = 1 := by omega
    have a11 : assignment (ChunkRows.sourceBitCol chunk 11) = 1 := by omega
    have a12 : assignment (ChunkRows.sourceBitCol chunk 12) = 1 := by omega
    have a13 : assignment (ChunkRows.sourceBitCol chunk 13) = 1 := by omega
    have a14 : assignment (ChunkRows.sourceBitCol chunk 14) = 1 := by omega
    have a15 : assignment (ChunkRows.sourceBitCol chunk 15) = 1 := by omega
    exact fin16_all
      (predicate := fun index => sourceBits assignment chunk index = 1)
      ((sourceBit_eq_one_iff bits 0).mpr a0)
      ((sourceBit_eq_one_iff bits 1).mpr a1)
      ((sourceBit_eq_one_iff bits 2).mpr a2)
      ((sourceBit_eq_one_iff bits 3).mpr a3)
      ((sourceBit_eq_one_iff bits 4).mpr a4)
      ((sourceBit_eq_one_iff bits 5).mpr a5)
      ((sourceBit_eq_one_iff bits 6).mpr a6)
      ((sourceBit_eq_one_iff bits 7).mpr a7)
      ((sourceBit_eq_one_iff bits 8).mpr a8)
      ((sourceBit_eq_one_iff bits 9).mpr a9)
      ((sourceBit_eq_one_iff bits 10).mpr a10)
      ((sourceBit_eq_one_iff bits 11).mpr a11)
      ((sourceBit_eq_one_iff bits 12).mpr a12)
      ((sourceBit_eq_one_iff bits 13).mpr a13)
      ((sourceBit_eq_one_iff bits 14).mpr a14)
      ((sourceBit_eq_one_iff bits 15).mpr a15)

theorem sourceAcceptanceMeaning_iff_verifier
    {assignment : Nat → Nat} {chunk : Nat}
    (bits : BitsBoolean assignment chunk) (accept : GateField) :
    SourceAcceptanceMeaning (sourceBits assignment chunk) accept ↔
      VerifierAcceptanceMeaning assignment chunk bits accept := by
  have sourceBoolean := sourceBits_are_boolean bits
  by_cases allOne : AllSourceBitsOne (sourceBits assignment chunk)
  · have productOne :=
      (sourceProduct_eq_one_iff_allSourceBitsOne
        (sourceBits assignment chunk) sourceBoolean).mpr allOne
    have rejected := (allSourceBitsOne_iff_rejectionBucket bits).mp allOne
    have acceptedIff :=
      ProductionAlphabet.accepts_eq_true_iff_ne_rejectionBucket
        (candidate assignment chunk bits)
    have notAccepted :
        ProductionAlphabet.verifier.accepts
            (candidate assignment chunk bits) = false := by
      apply Bool.eq_false_iff.mpr
      intro accepted
      apply (acceptedIff.mp accepted) rejected
    unfold SourceAcceptanceMeaning VerifierAcceptanceMeaning
    rw [productOne, fieldSub_one_one, notAccepted]
    simp
  · have productBit := sourceProduct_boolean sourceBoolean
    have productNotOne :
        lowHalfProduct (sourceBits assignment chunk) *
            highHalfProduct (sourceBits assignment chunk) ≠ 1 := by
      intro productOne
      exact allOne
        ((sourceProduct_eq_one_iff_allSourceBitsOne
          (sourceBits assignment chunk) sourceBoolean).mp productOne)
    have productZero :
        lowHalfProduct (sourceBits assignment chunk) *
            highHalfProduct (sourceBits assignment chunk) = 0 := by
      rcases productBit with zero | one
      · exact zero
      · exact False.elim (productNotOne one)
    have notRejected :
        chunkValue assignment chunk ≠ ProductionAlphabet.rejectionBucket := by
      intro rejected
      exact allOne ((allSourceBitsOne_iff_rejectionBucket bits).mpr rejected)
    have accepted :
        ProductionAlphabet.verifier.accepts
            (candidate assignment chunk bits) = true :=
      (ProductionAlphabet.accepts_eq_true_iff_ne_rejectionBucket
        (candidate assignment chunk bits)).mpr notRejected
    unfold SourceAcceptanceMeaning VerifierAcceptanceMeaning
    rw [productZero, fieldSub_one_zero, accepted]
    simp

end Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Chunk.Acceptance
