import SuperNeo.FPrimeRecursiveVerifier.PiRlcChallenge.Refinement.PackedChunkRows

/-!
Owns: the two-equation Goldilocks model for one sixteen-bit chunk's product
acceptance check.

Does not own: Rust row emission, transcript sampling, the global enough-accepts
check, selector wiring, or inactive-branch column layout.

Emits constraints: no. This file proves the proposed row relation only.

Authority boundary: all sixteen chunk coordinates remain authoritative inputs
and must already satisfy their ordinary bit-root checks. The auxiliary cells
`u`, `v`, and `a` are derived by these rows; none is accepted as a digest or
prover-supplied authority.

| Row/family | Mathematical obligation | Degree | Guarantee |
|---|---|---:|---|
| `ProductBindingRow` | `(u - P0) + 2 * (v - P1) = 0` | 8 | Binds both half-products once all four values are Boolean |
| inner nonresidue pair | `q0^2 - 7*q1^2` | 4 | Forces the bit residuals of `u` and `v` to zero |
| acceptance residual | `f = (1 - a) - u*v` | 2 | Defines the rejection-style acceptance bit |
| `NestedAcceptanceRow` | `A^2 - 7*f^4 = 0` | 8 | Simultaneously enforces Boolean `u/v` and `a = 1-u*v` |

Here `P0` and `P1` are the products of chunk bits `0..7` and `8..15`,
`q0 = u(u-1)`, `q1 = v(v-1)`, and `A = q0^2 - 7*q1^2`.
The degree-eight rows cannot be multiplied by an additional linear selector
without exceeding the fixed degree-eight CCS ceiling. Instead,
`packedAcceptanceRows_inactive_extension` proves the precise existential fact
needed for an unconditional-row design: every Boolean chunk has a unique
extension to `u`, `v`, and `a`.

| Predicate/theorem | Rust stage | Guarantee | Assumptions | Permits Rust row removal? |
|---|---|---|---|---|
| `productBindingRow_exact` | proposed chunk acceptance lowering | No cancellation between the two product differences | All four compared values are field bits | No - exact Rust trace bridge required |
| `nestedAcceptanceRow_iff` | proposed chunk acceptance lowering | One row is equivalent to Boolean `u/v` plus the exact acceptance value | `7` is a quadratic nonresidue | No - exact Rust trace bridge required |
| `packedAcceptanceRows_iff` | proposed chunk acceptance lowering | The two rows are exactly the readable acceptance obligations | All sixteen source coordinates are field bits | No - exact Rust trace bridge required |
| `packedAcceptanceRows_inactive_extension` | proposed unconditional inactive lowering | Arbitrary Boolean chunk inputs have an extension | Common chunk-bitness authority remains active | No - selector/layout noninterference remains separate |
| `productBindingRow_is_necessary` | obligation minimality | The nested row alone admits unbound products | Concrete Boolean counterexample | No |
| `nestedAcceptanceRow_is_necessary` | obligation minimality | The binding row alone admits an incorrect acceptance value | Concrete Boolean counterexample | No |
-/

namespace SuperNeo.FPrimeRecursiveVerifier.PiRlcChallenge

open scoped BigOperators

/-- Embed a low-half coordinate into the sixteen-coordinate chunk. -/
def lowHalfIndex (index : Fin 8) : Fin 16 :=
  ⟨index.val, by omega⟩

/-- Embed a high-half coordinate into the sixteen-coordinate chunk. -/
def highHalfIndex (index : Fin 8) : Fin 16 :=
  ⟨index.val + 8, by omega⟩

/-- Product of chunk coordinates zero through seven. -/
def lowHalfProduct (bits : Fin 16 → F) : F :=
  ∏ index : Fin 8, bits (lowHalfIndex index)

/-- Product of chunk coordinates eight through fifteen. -/
def highHalfProduct (bits : Fin 16 → F) : F :=
  ∏ index : Fin 8, bits (highHalfIndex index)

/-- The common source obligation: all sixteen chunk coordinates are bits. -/
def ChunkBitsAreBoolean (bits : Fin 16 → F) : Prop :=
  ∀ index, FieldBitRoot (bits index)

/-- Residual of the row that binds both half-products. -/
def productBindingResidual (bits : Fin 16 → F) (u v : F) : F :=
  (u - lowHalfProduct bits) + 2 * (v - highHalfProduct bits)

/-- First proposed acceptance row. -/
def ProductBindingRow (bits : Fin 16 → F) (u v : F) : Prop :=
  productBindingResidual bits u v = 0

/-- Residual defining the rejection-style acceptance bit. -/
def acceptanceValueResidual (u v a : F) : F :=
  (1 - a) - u * v

/-- The degree-four nonresidue pair of the two auxiliary bit residuals. -/
def auxiliaryBitPairResidual (u v : F) : F :=
  fieldBitResidual u * fieldBitResidual u -
    KExt.w * (fieldBitResidual v * fieldBitResidual v)

/--
Second proposed row. Its outer nonresidue pair has left coordinate `A` and
right coordinate `f^2`, hence equation `A^2 - 7*f^4 = 0`.
-/
def NestedAcceptanceRow (u v a : F) : Prop :=
  QuadraticZeroPair (auxiliaryBitPairResidual u v)
    (acceptanceValueResidual u v a * acceptanceValueResidual u v a)

/-- The readable obligations represented by the proposed two rows. -/
def PackedAcceptanceMeaning
    (bits : Fin 16 → F) (u v a : F) : Prop :=
  u = lowHalfProduct bits ∧
    v = highHalfProduct bits ∧
    FieldBitRoot u ∧
    FieldBitRoot v ∧
    a = 1 - u * v

/-- The complete two-row relation. -/
def PackedAcceptanceRows
    (bits : Fin 16 → F) (u v a : F) : Prop :=
  ProductBindingRow bits u v ∧ NestedAcceptanceRow u v a

private theorem fieldBitRoot_cases
    {value : F} (hRoot : FieldBitRoot value) :
    value = 0 ∨ value = 1 := by
  rcases mul_eq_zero.mp hRoot with hZero | hOne
  · exact Or.inl hZero
  · exact Or.inr (sub_eq_zero.mp hOne)

private theorem fieldBitRoot_of_cases
    {value : F} (hCases : value = 0 ∨ value = 1) :
    FieldBitRoot value := by
  rcases hCases with rfl | rfl <;>
    simp [FieldBitRoot, fieldBitResidual]

private theorem field_two_ne_zero : (2 : F) ≠ 0 := by
  intro hZero
  have hVal := congrArg Fin.val hZero
  norm_num [F.ofNat, F.val_zero, Goldilocks.q] at hVal

private theorem field_three_ne_zero : (3 : F) ≠ 0 := by
  intro hZero
  have hVal := congrArg Fin.val hZero
  norm_num [F.ofNat, F.val_zero, Goldilocks.q] at hVal

private theorem halfProduct_cases
    (values : Fin 8 → F)
    (hValues : ∀ index, FieldBitRoot (values index)) :
    (∏ index, values index) = 0 ∨ (∏ index, values index) = 1 := by
  classical
  by_cases hZero : ∃ index, values index = 0
  · rcases hZero with ⟨index, hIndex⟩
    exact Or.inl (Finset.prod_eq_zero (Finset.mem_univ index) hIndex)
  · have hOne : ∀ index, values index = 1 := by
      intro index
      rcases fieldBitRoot_cases (hValues index) with hIndex | hIndex
      · exact False.elim (hZero ⟨index, hIndex⟩)
      · exact hIndex
    exact Or.inr (by simp [hOne])

private theorem lowHalfProduct_cases
    {bits : Fin 16 → F} (hBits : ChunkBitsAreBoolean bits) :
    lowHalfProduct bits = 0 ∨ lowHalfProduct bits = 1 := by
  apply halfProduct_cases
  intro index
  exact hBits (lowHalfIndex index)

private theorem highHalfProduct_cases
    {bits : Fin 16 → F} (hBits : ChunkBitsAreBoolean bits) :
    highHalfProduct bits = 0 ∨ highHalfProduct bits = 1 := by
  apply halfProduct_cases
  intro index
  exact hBits (highHalfIndex index)

private theorem lowHalfProduct_is_bit
    {bits : Fin 16 → F} (hBits : ChunkBitsAreBoolean bits) :
    FieldBitRoot (lowHalfProduct bits) :=
  fieldBitRoot_of_cases (lowHalfProduct_cases hBits)

private theorem highHalfProduct_is_bit
    {bits : Fin 16 → F} (hBits : ChunkBitsAreBoolean bits) :
    FieldBitRoot (highHalfProduct bits) :=
  fieldBitRoot_of_cases (highHalfProduct_cases hBits)

/--
The coefficient two cannot hide a mismatch between Boolean values. The proof
enumerates the exact four bit roots. Every nonzero residual becomes one of
`±1`, `±2`, or `±3`; `field_two_ne_zero` and `field_three_ne_zero`
verify explicitly that Goldilocks characteristic does not collapse the latter
two cases.
-/
theorem productBindingRow_exact
    {u v product0 product1 : F}
    (hU : FieldBitRoot u)
    (hV : FieldBitRoot v)
    (hProduct0 : FieldBitRoot product0)
    (hProduct1 : FieldBitRoot product1)
    (hRow : (u - product0) + 2 * (v - product1) = 0) :
    u = product0 ∧ v = product1 := by
  rcases fieldBitRoot_cases hU with hU | hU <;>
    rcases fieldBitRoot_cases hV with hV | hV <;>
    rcases fieldBitRoot_cases hProduct0 with hProduct0 | hProduct0 <;>
    rcases fieldBitRoot_cases hProduct1 with hProduct1 | hProduct1 <;>
    subst u <;> subst v <;> subst product0 <;> subst product1
  all_goals
    norm_num [field_two_ne_zero, field_three_ne_zero] at hRow
  all_goals exact ⟨rfl, rfl⟩

/-- The nested row is exactly the two auxiliary bit roots and acceptance value. -/
theorem nestedAcceptanceRow_iff (u v a : F) :
    NestedAcceptanceRow u v a ↔
      FieldBitRoot u ∧ FieldBitRoot v ∧ a = 1 - u * v := by
  constructor
  · intro hRow
    have hOuter := quadraticZeroPair_iff.mp hRow
    have hInner : QuadraticZeroPair
        (fieldBitResidual u) (fieldBitResidual v) := by
      exact hOuter.1
    have hBits := quadraticZeroPair_iff.mp hInner
    have hAcceptance : acceptanceValueResidual u v a = 0 :=
      mul_self_eq_zero.mp hOuter.2
    refine ⟨hBits.1, hBits.2, ?_⟩
    dsimp [acceptanceValueResidual] at hAcceptance
    linear_combination -hAcceptance
  · rintro ⟨hU, hV, hAcceptance⟩
    apply quadraticZeroPair_iff.mpr
    constructor
    · change QuadraticZeroPair (fieldBitResidual u) (fieldBitResidual v)
      exact quadraticZeroPair_iff.mpr ⟨hU, hV⟩
    · simp [acceptanceValueResidual, hAcceptance]

/-- The derived acceptance value is itself a field bit. -/
theorem nestedAcceptanceRow_implies_acceptance_bit
    {u v a : F} (hRow : NestedAcceptanceRow u v a) :
    FieldBitRoot a := by
  rcases (nestedAcceptanceRow_iff u v a).mp hRow with
    ⟨hU, hV, hAcceptance⟩
  rcases fieldBitRoot_cases hU with hU | hU <;>
    rcases fieldBitRoot_cases hV with hV | hV <;>
    subst u <;> subst v <;> subst a <;>
    simp [FieldBitRoot, fieldBitResidual]

/-- Under source bitness, the two rows are exactly the readable obligations. -/
theorem packedAcceptanceRows_iff
    (bits : Fin 16 → F) (u v a : F)
    (hBits : ChunkBitsAreBoolean bits) :
    PackedAcceptanceRows bits u v a ↔
      PackedAcceptanceMeaning bits u v a := by
  constructor
  · rintro ⟨hBinding, hNested⟩
    have hNestedMeaning := (nestedAcceptanceRow_iff u v a).mp hNested
    have hProducts := productBindingRow_exact
      hNestedMeaning.1 hNestedMeaning.2.1
      (lowHalfProduct_is_bit hBits) (highHalfProduct_is_bit hBits)
      hBinding
    exact ⟨hProducts.1, hProducts.2, hNestedMeaning.1,
      hNestedMeaning.2.1, hNestedMeaning.2.2⟩
  · rintro ⟨hU, hV, hUBit, hVBit, hAcceptance⟩
    constructor
    · simp [ProductBindingRow, productBindingResidual, hU, hV]
    · exact (nestedAcceptanceRow_iff u v a).mpr
        ⟨hUBit, hVBit, hAcceptance⟩

/--
Existential inactive noninterference for the unconditional two-row design.
No Boolean chunk is rejected: the products and acceptance cell provide a
witness extension for every assignment already admitted by common bitness.
-/
theorem packedAcceptanceRows_inactive_extension
    (bits : Fin 16 → F)
    (hBits : ChunkBitsAreBoolean bits) :
    ∃ u v a, PackedAcceptanceRows bits u v a := by
  refine ⟨lowHalfProduct bits, highHalfProduct bits,
    1 - lowHalfProduct bits * highHalfProduct bits, ?_⟩
  apply (packedAcceptanceRows_iff bits _ _ _ hBits).mpr
  exact ⟨rfl, rfl, lowHalfProduct_is_bit hBits,
    highHalfProduct_is_bit hBits, rfl⟩

/-- The unconditional witness extension is unique on Boolean source chunks. -/
theorem packedAcceptanceRows_extension_unique
    {bits : Fin 16 → F}
    (hBits : ChunkBitsAreBoolean bits)
    {u v a u' v' a' : F}
    (hRows : PackedAcceptanceRows bits u v a)
    (hRows' : PackedAcceptanceRows bits u' v' a') :
    u = u' ∧ v = v' ∧ a = a' := by
  have hMeaning := (packedAcceptanceRows_iff bits u v a hBits).mp hRows
  have hMeaning' := (packedAcceptanceRows_iff bits u' v' a' hBits).mp hRows'
  exact ⟨hMeaning.1.trans hMeaning'.1.symm,
    hMeaning.2.1.trans hMeaning'.2.1.symm, by
      rw [hMeaning.2.2.2.2, hMeaning'.2.2.2.2,
        hMeaning.1, hMeaning.2.1, hMeaning'.1, hMeaning'.2.1]⟩

/-! ## Degree accounting -/

/-- Degree of each eight-coordinate half-product. -/
def halfProductDegree : Nat := 8

/-- Degree of the first product-binding row. -/
def productBindingRowDegree : Nat := max 1 halfProductDegree

/-- Degree of one auxiliary bit residual. -/
def auxiliaryBitResidualDegree : Nat := 2

/-- Degree of the inner nonresidue pair `A`. -/
def auxiliaryBitPairResidualDegree : Nat := 2 * auxiliaryBitResidualDegree

/-- Degree of `f = (1-a)-u*v`. -/
def acceptanceValueResidualDegree : Nat := 2

/-- Degree of the outer nested nonresidue row. -/
def nestedAcceptanceRowDegree : Nat :=
  max (2 * auxiliaryBitPairResidualDegree)
    (4 * acceptanceValueResidualDegree)

/-- Degree that would result from multiplying the nested row by a selector. -/
def selectorGatedNestedAcceptanceRowDegree : Nat :=
  nestedAcceptanceRowDegree + 1

/-- Both proposed rows exactly meet, but do not exceed, the CCS degree ceiling. -/
theorem packedAcceptanceRows_degree_budget :
    halfProductDegree = 8 ∧
      productBindingRowDegree = 8 ∧
      auxiliaryBitResidualDegree = 2 ∧
      auxiliaryBitPairResidualDegree = 4 ∧
      acceptanceValueResidualDegree = 2 ∧
      nestedAcceptanceRowDegree = 8 ∧
      productBindingRowDegree ≤ 8 ∧
      nestedAcceptanceRowDegree ≤ 8 ∧
      selectorGatedNestedAcceptanceRowDegree = 9 ∧
      8 < selectorGatedNestedAcceptanceRowDegree := by
  norm_num [halfProductDegree, productBindingRowDegree,
    auxiliaryBitResidualDegree, auxiliaryBitPairResidualDegree,
    acceptanceValueResidualDegree, nestedAcceptanceRowDegree,
    selectorGatedNestedAcceptanceRowDegree]

/-! ## Per-row necessity witnesses -/

/-- Removing the product-binding row admits unrelated Boolean products. -/
theorem productBindingRow_is_necessary :
    ∃ bits : Fin 16 → F, ∃ u v a,
      ChunkBitsAreBoolean bits ∧
        NestedAcceptanceRow u v a ∧
        ¬ PackedAcceptanceMeaning bits u v a := by
  refine ⟨fun _ => 1, 0, 0, 1, ?_, ?_, ?_⟩
  · intro index
    simp [FieldBitRoot, fieldBitResidual]
  · exact (nestedAcceptanceRow_iff 0 0 1).mpr (by
      simp [FieldBitRoot, fieldBitResidual])
  · simp [PackedAcceptanceMeaning, lowHalfProduct]

/-- Removing the nested row leaves the acceptance value unconstrained. -/
theorem nestedAcceptanceRow_is_necessary :
    ∃ bits : Fin 16 → F, ∃ u v a,
      ChunkBitsAreBoolean bits ∧
        ProductBindingRow bits u v ∧
        ¬ PackedAcceptanceMeaning bits u v a := by
  refine ⟨fun _ => 0, 0, 0, 0, ?_, ?_, ?_⟩
  · intro index
    simp [FieldBitRoot, fieldBitResidual]
  · simp [ProductBindingRow, productBindingResidual,
      lowHalfProduct, highHalfProduct]
  · simp [PackedAcceptanceMeaning, lowHalfProduct, highHalfProduct]

end SuperNeo.FPrimeRecursiveVerifier.PiRlcChallenge
