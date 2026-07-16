import SuperNeo.FPrimeRecursiveVerifier.PiRlcChallenge.Refinement.ChunkRows
import SuperNeo.Primitives.ExtensionField

/-!
Owns: the quadratic-nonresidue packing of the sixteen readable chunk
obligations into eight Goldilocks equations.

Does not own: the underlying chunk semantics, selector implementation, Rust
row emission, or Rust trace conformance.

Emits constraints: no. This file specifies and proves the packed row model.

Authority boundary: packing changes only how zero residuals are enforced.
`KExt.w_not_square` proves that one packed equation cannot cancel two nonzero
residuals. No digest or prover-supplied summary becomes authoritative.

| Packed row | Left zero obligation | Right zero obligation | Degree before selector | Selector-gated degree |
|---:|---|---|---:|---:|
| 0 | low quotient bit 0 | low quotient bit 1 | 4 | 5 |
| 1 | low quotient bit 2 | low quotient bit 3 | 4 | 5 |
| 2 | low quotient bit 4 | low quotient bit 5 | 4 | 5 |
| 3 | low quotient bit 6 | low quotient bit 7 | 4 | 5 |
| 4 | low quotient bit 8 | low quotient bit 9 | 4 | 5 |
| 5 | low quotient bit 10 | low quotient bit 11 | 4 | 5 |
| 6 | low quotient bit 12 | linearly derived high bit | 4 | 5 |
| 7 | left centered cubic | centered residue-pair equation | 6 | 7 |

The fixed CCS maximum degree is eight. A bit residual has degree two, so a
quadratic norm of two bit residuals has degree four. The final row packs a
degree-three cubic residual with a degree-two pair residual, so its maximum
degree is six. Multiplication by the selector raises those degrees to five and
seven respectively. The inactive row packs the shifted linear coordinates
`L+1,R+1`; it has degree two before its inactive selector and degree three
afterward.

| Predicate/theorem | Rust stage | Guarantee | Assumptions | Permits Rust row removal? |
|---|---|---|---|---|
| `quadraticZeroPair_iff` | packed chunk row | `a^2 - 7*b^2 = 0` iff both residuals vanish | `KExt.w_not_square` | No - requires an exact trace bridge |
| `PackedReducedMod5FieldRows` | `challenge.sampler.chunk.mod5` | Eight equations enforce all sixteen direct obligations | Degree-eight CCS | No - requires an exact trace bridge |
| `packedReducedMod5FieldRows_iff_direct` | `challenge.sampler.chunk.mod5` | Packed and direct Goldilocks relations are equivalent | Exact residual pairing | No - requires an exact trace bridge |
| `packedReducedMod5FieldRows_iff_chunkArithmetic` | `challenge.sampler.chunk.mod5` | Packed witness existence equals source arithmetic witness existence | Direct field/source refinement | No - requires an exact trace bridge |
| `packedInactiveResiduePair_iff` | fixed selector inactive branch | One packed equation forces the unique centered encoding of residue index zero | Inactive selector and `KExt.w_not_square` | No - requires the combined selector materializer |
-/

namespace SuperNeo.FPrimeRecursiveVerifier.PiRlcChallenge

/-- One row that soundly packs two scalar zero obligations using `w = 7`. -/
def QuadraticZeroPair (left right : F) : Prop :=
  left * left - KExt.w * (right * right) = 0

/-- Nonresiduosity of seven prevents cancellation between packed residuals. -/
theorem quadraticZeroPair_iff {left right : F} :
    QuadraticZeroPair left right ↔ left = 0 ∧ right = 0 := by
  constructor
  · intro hPacked
    by_cases hRight : right = 0
    · subst right
      simp only [QuadraticZeroPair, mul_zero, sub_zero] at hPacked
      exact ⟨mul_self_eq_zero.mp hPacked, rfl⟩
    · exfalso
      apply KExt.w_not_square (left / right)
      have hEquation : left * left = KExt.w * (right * right) :=
        sub_eq_zero.mp hPacked
      field_simp [hRight]
      simpa [pow_two, mul_comm, mul_left_comm, mul_assoc] using hEquation
  · rintro ⟨rfl, rfl⟩
    simp [QuadraticZeroPair]

/-- Inactive residue coordinates are shifted so their unique accepted pair is
`(-1,-1)`, which decodes to source residue index zero. -/
def PackedInactiveResiduePair (left right : F) : Prop :=
  QuadraticZeroPair (left + 1) (right + 1)

/-- A linear decoded-index equation alone would admit noncanonical pairs. The
nonresidue-packed inactive row instead fixes both coordinates uniquely. -/
theorem packedInactiveResiduePair_iff {left right : F} :
    PackedInactiveResiduePair left right ↔ left = -1 ∧ right = -1 := by
  rw [PackedInactiveResiduePair, quadraticZeroPair_iff]
  constructor
  · rintro ⟨hLeft, hRight⟩
    exact ⟨add_eq_zero_iff_eq_neg.mp hLeft,
      add_eq_zero_iff_eq_neg.mp hRight⟩
  · rintro ⟨rfl, rfl⟩
    simp

/-- Algebraic degree of a row packing two quadratic bit residuals. -/
def packedBitPairDegree : Nat := 4

/-- Algebraic degree of the row packing the cubic and pair residuals. -/
def packedResiduePairDegree : Nat := 6

/-- Algebraic degree of the inactive centered-pair row. -/
def packedInactiveResiduePairDegree : Nat := 2

/-- A CCS selector multiplies the packed residual by one linear factor. -/
def selectorGatedDegree (rowDegree : Nat) : Nat := rowDegree + 1

/-- Both packed row families fit under the fixed degree-eight CCS ceiling. -/
theorem packedChunkRows_degree_budget :
    packedBitPairDegree = 4 ∧
      selectorGatedDegree packedBitPairDegree = 5 ∧
      packedResiduePairDegree = 6 ∧
      selectorGatedDegree packedResiduePairDegree = 7 ∧
      packedInactiveResiduePairDegree = 2 ∧
      selectorGatedDegree packedInactiveResiduePairDegree = 3 ∧
      selectorGatedDegree packedBitPairDegree ≤ 8 ∧
      selectorGatedDegree packedResiduePairDegree ≤ 8 ∧
      selectorGatedDegree packedInactiveResiduePairDegree ≤ 8 := by
  norm_num [packedBitPairDegree, packedResiduePairDegree,
    packedInactiveResiduePairDegree, selectorGatedDegree]

/--
Eight packed equations over the same fifteen cells as the readable direct
relation. Numerals are fixed `Fin 13` indices in little-endian order.
-/
def PackedReducedMod5FieldRows
    (chunk : Chunk) (witness : ReducedMod5FieldWitness) : Prop :=
  QuadraticZeroPair
      (fieldBitResidual (witness.quotientLow (0 : Fin 13)))
      (fieldBitResidual (witness.quotientLow (1 : Fin 13))) ∧
    QuadraticZeroPair
      (fieldBitResidual (witness.quotientLow (2 : Fin 13)))
      (fieldBitResidual (witness.quotientLow (3 : Fin 13))) ∧
    QuadraticZeroPair
      (fieldBitResidual (witness.quotientLow (4 : Fin 13)))
      (fieldBitResidual (witness.quotientLow (5 : Fin 13))) ∧
    QuadraticZeroPair
      (fieldBitResidual (witness.quotientLow (6 : Fin 13)))
      (fieldBitResidual (witness.quotientLow (7 : Fin 13))) ∧
    QuadraticZeroPair
      (fieldBitResidual (witness.quotientLow (8 : Fin 13)))
      (fieldBitResidual (witness.quotientLow (9 : Fin 13))) ∧
    QuadraticZeroPair
      (fieldBitResidual (witness.quotientLow (10 : Fin 13)))
      (fieldBitResidual (witness.quotientLow (11 : Fin 13))) ∧
    QuadraticZeroPair
      (fieldBitResidual (witness.quotientLow (12 : Fin 13)))
      (fieldBitResidual (derivedQuotientHighField chunk witness)) ∧
    QuadraticZeroPair
      (fieldCenteredResidual witness.residueLeft)
      witness.residuePairResidual

/-- The eight packed rows enforce exactly the sixteen direct obligations. -/
theorem packedReducedMod5FieldRows_iff_direct
    (chunk : Chunk) (witness : ReducedMod5FieldWitness) :
    PackedReducedMod5FieldRows chunk witness ↔
      ReducedMod5FieldRows chunk witness := by
  constructor
  · rintro ⟨h01, h23, h45, h67, h89, h1011, h12High, hResidue⟩
    have h01' := quadraticZeroPair_iff.mp h01
    have h23' := quadraticZeroPair_iff.mp h23
    have h45' := quadraticZeroPair_iff.mp h45
    have h67' := quadraticZeroPair_iff.mp h67
    have h89' := quadraticZeroPair_iff.mp h89
    have h1011' := quadraticZeroPair_iff.mp h1011
    have h12High' := quadraticZeroPair_iff.mp h12High
    have hResidue' := quadraticZeroPair_iff.mp hResidue
    refine ⟨?_, ?_, ?_, ?_⟩
    · intro index
      fin_cases index
      · exact h01'.1
      · exact h01'.2
      · exact h23'.1
      · exact h23'.2
      · exact h45'.1
      · exact h45'.2
      · exact h67'.1
      · exact h67'.2
      · exact h89'.1
      · exact h89'.2
      · exact h1011'.1
      · exact h1011'.2
      · exact h12High'.1
    · exact h12High'.2
    · exact hResidue'.1
    · exact hResidue'.2
  · rintro ⟨hBits, hHigh, hLeft, hPair⟩
    exact
      ⟨quadraticZeroPair_iff.mpr
          ⟨hBits (0 : Fin 13), hBits (1 : Fin 13)⟩,
        quadraticZeroPair_iff.mpr
          ⟨hBits (2 : Fin 13), hBits (3 : Fin 13)⟩,
        quadraticZeroPair_iff.mpr
          ⟨hBits (4 : Fin 13), hBits (5 : Fin 13)⟩,
        quadraticZeroPair_iff.mpr
          ⟨hBits (6 : Fin 13), hBits (7 : Fin 13)⟩,
        quadraticZeroPair_iff.mpr
          ⟨hBits (8 : Fin 13), hBits (9 : Fin 13)⟩,
        quadraticZeroPair_iff.mpr
          ⟨hBits (10 : Fin 13), hBits (11 : Fin 13)⟩,
        quadraticZeroPair_iff.mpr ⟨hBits (12 : Fin 13), hHigh⟩,
        quadraticZeroPair_iff.mpr ⟨hLeft, hPair⟩⟩

/-- The packed field relation has the same exact Nat refinement. -/
theorem packedReducedMod5FieldRows_iff_nat
    (chunk : Chunk) (fieldWitness : ReducedMod5FieldWitness) :
    PackedReducedMod5FieldRows chunk fieldWitness ↔
      ∃ natWitness,
        ReducedMod5Holds chunk natWitness ∧
          ReducedMod5FieldRepresents chunk fieldWitness natWitness := by
  exact (packedReducedMod5FieldRows_iff_direct chunk fieldWitness).trans
    (reducedMod5FieldRows_iff_nat chunk fieldWitness)

/-- Packed witness existence is equivalent to source chunk arithmetic. -/
theorem packedReducedMod5FieldRows_iff_chunkArithmetic (chunk : Chunk) :
    (∃ fieldWitness, PackedReducedMod5FieldRows chunk fieldWitness) ↔
      ∃ sourceWitness, ChunkArithmeticHolds chunk sourceWitness := by
  constructor
  · rintro ⟨fieldWitness, hPacked⟩
    exact (reducedMod5FieldRows_iff_chunkArithmetic chunk).mp
      ⟨fieldWitness,
        (packedReducedMod5FieldRows_iff_direct chunk fieldWitness).mp hPacked⟩
  · intro hSource
    rcases (reducedMod5FieldRows_iff_chunkArithmetic chunk).mpr hSource with
      ⟨fieldWitness, hDirect⟩
    exact ⟨fieldWitness,
      (packedReducedMod5FieldRows_iff_direct chunk fieldWitness).mpr hDirect⟩

/-- The packed relation preserves uniqueness of the fifteen-cell witness. -/
theorem packedReducedMod5FieldWitness_unique
    {chunk : Chunk} {left right : ReducedMod5FieldWitness}
    (hLeft : PackedReducedMod5FieldRows chunk left)
    (hRight : PackedReducedMod5FieldRows chunk right) :
    left = right := by
  exact reducedMod5FieldWitness_unique
    ((packedReducedMod5FieldRows_iff_direct chunk left).mp hLeft)
    ((packedReducedMod5FieldRows_iff_direct chunk right).mp hRight)

end SuperNeo.FPrimeRecursiveVerifier.PiRlcChallenge
