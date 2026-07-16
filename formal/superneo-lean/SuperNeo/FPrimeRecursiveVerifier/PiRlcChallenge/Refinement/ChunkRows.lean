import Batteries.Data.BitVec
import Mathlib.Data.BitVec
import Mathlib.Tactic
import SuperNeo.FPrimeRecursiveVerifier.PiRlcChallenge.Sampler.Chunk

/-!
Owns: the exact Goldilocks row model and Nat refinement for one chunk's
mod-5 arithmetic.

Does not own: Rust row emission, Rust trace conformance, chunk acceptance, or
transcript authority.

Emits constraints: no.

Authority boundary: the conservative predicate checks all thirteen committed
quotient bits, the linearly reconstructed high bit, and both centered residue
coordinates. The optional batched-bitness theorem additionally assumes an
outer centered-norm lift; it is not authority for that lift.

| Predicate/theorem | Rust stage | Guarantee | Assumptions | Permits row removal? |
|---|---|---|---|---|
| `ReducedMod5Holds` | `challenge.sampler.chunk.mod5` | Thirteen low bits, one derived high bit, and two centered coordinates reconstruct the canonical quotient/residue | Canonical 16-bit chunk | No - requires an exact Rust trace bridge |
| `reducedMod5_iff_chunkArithmetic` | `challenge.sampler.chunk.mod5` | Reduced and source arithmetic witnesses exist for exactly the same chunks | Exact integer equations | No - requires an exact Rust trace bridge |
| `reducedMod5Witness_unique` | `challenge.sampler.chunk.mod5` | The conservative reduced witness is unique | `ReducedMod5Holds` | No - requires an exact Rust trace bridge |
| `fieldCenteredPair_implies_rightRoot` | `challenge.sampler.chunk.mod5` | The right centered cubic follows from the left cubic and pair equation | Goldilocks has no zero divisors | Field model only - Rust requires a trace bridge |
| `ReducedMod5FieldRows` | `challenge.sampler.chunk.mod5` | Sixteen field equations over fifteen witness cells: thirteen low-bit roots, one derived-high root, one left centered cubic, and one pair equation | Canonical `Fin 65536`, below-modulus reconstruction bounds, and a nonzero high-bit denominator | No - exact Rust trace conformance is separate |
| `reducedMod5FieldRows_iff_nat` | `challenge.sampler.chunk.mod5` | The sixteen-equation field model is equivalent to the unique Nat reduced witness | Exact field encoding and no-wrap bounds | No - exact Rust trace conformance is separate |
| `reducedMod5FieldRows_iff_chunkArithmetic` | `challenge.sampler.chunk.mod5` | The field model exists exactly when the source chunk arithmetic witness exists | Nat/source equivalence plus field/Nat refinement | No - exact Rust trace conformance is separate |
| `normBatch_implies_bits` | proposed norm-backed lowering | One aggregate bitness equation excludes `-1` from thirteen norm-bounded coordinates | Explicit integer lift of the outer norm and batch equation | No - requires a field and Rust lift |
-/

namespace SuperNeo.FPrimeRecursiveVerifier.PiRlcChallenge

private def lowQuotientBits : Nat := quotientBits - 1

/-- A centered coordinate at the `b = 2` committed boundary. -/
def CenteredUnit (value : Int) : Prop :=
  value = -1 ∨ value = 0 ∨ value = 1

/-- The conservative low-width witness: thirteen explicit bits and two trits. -/
structure ReducedMod5Witness where
  quotientLow : BitVec lowQuotientBits
  residueLeft : Int
  residueRight : Int
deriving Repr, DecidableEq

/-- The low thirteen quotient bits, exposed individually in little-endian order. -/
def ReducedMod5Witness.quotientBit
    (witness : ReducedMod5Witness) (index : Fin lowQuotientBits) : Bool :=
  witness.quotientLow.getLsb index

/-- Integer represented by the thirteen committed quotient bits. -/
def ReducedMod5Witness.quotientLowValue (witness : ReducedMod5Witness) : Nat :=
  witness.quotientLow.toNat

/-- Centered residue before the fixed `+2` map into the unsigned index. -/
def ReducedMod5Witness.centeredResidue (witness : ReducedMod5Witness) : Int :=
  witness.residueLeft + witness.residueRight

/-- Unsigned residue reconstructed from the two centered coordinates. -/
def ReducedMod5Witness.residueIndex (witness : ReducedMod5Witness) : Nat :=
  (witness.centeredResidue + 2).toNat

/--
The high quotient bit reconstructed from the chunk equation. The exact
reconstruction conjunct in `ReducedMod5Holds` rules out truncation in both the
subtraction and division.
-/
def derivedQuotientHigh (chunk : Chunk) (witness : ReducedMod5Witness) : Nat :=
  (chunk.val -
      (alphabetSize * witness.quotientLowValue + witness.residueIndex)) /
    (alphabetSize * 2 ^ lowQuotientBits)

/-- Full fourteen-bit quotient decoded from the reduced witness. -/
def ReducedMod5Witness.quotientValue
    (chunk : Chunk) (witness : ReducedMod5Witness) : Nat :=
  witness.quotientLowValue +
    2 ^ lowQuotientBits * derivedQuotientHigh chunk witness

/-- The one-row centered-pair relation selecting exactly five sums. -/
def CenteredResiduePairHolds (witness : ReducedMod5Witness) : Prop :=
  CenteredUnit witness.residueLeft ∧
    CenteredUnit witness.residueRight ∧
    witness.residueRight * (witness.residueLeft - witness.residueRight) = 0

/-- Conservative reduced equations for the current mod-5 source block. -/
def ReducedMod5Holds (chunk : Chunk) (witness : ReducedMod5Witness) : Prop :=
  CenteredResiduePairHolds witness ∧
    derivedQuotientHigh chunk witness < 2 ∧
    chunk.val = alphabetSize * witness.quotientValue chunk + witness.residueIndex

/-- Exact reconstruction keeps the natural-number high-bit subtraction from wrapping. -/
theorem reducedMod5_no_wrap
    {chunk : Chunk} {witness : ReducedMod5Witness}
    (hReduced : ReducedMod5Holds chunk witness) :
    alphabetSize * witness.quotientLowValue + witness.residueIndex ≤ chunk.val := by
  have hChunk := hReduced.2.2
  have hQuotient :
      witness.quotientLowValue ≤ witness.quotientValue chunk := by
    simp [ReducedMod5Witness.quotientValue]
  rw [hChunk]
  exact Nat.add_le_add_right
    (Nat.mul_le_mul_left alphabetSize hQuotient) witness.residueIndex

private theorem centeredResidue_cases
    {left right : Int}
    (hLeft : CenteredUnit left)
    (hRight : CenteredUnit right)
    (hPair : right * (left - right) = 0) :
    (left = -1 ∧ right = -1) ∨
      (left = -1 ∧ right = 0) ∨
      (left = 0 ∧ right = 0) ∨
      (left = 1 ∧ right = 0) ∨
      (left = 1 ∧ right = 1) := by
  rcases hLeft with hLeft | hLeft | hLeft <;>
    rcases hRight with hRight | hRight | hRight <;>
    simp_all

theorem centeredResiduePair_index_lt
    {witness : ReducedMod5Witness}
    (hPair : CenteredResiduePairHolds witness) :
    witness.residueIndex < alphabetSize := by
  rcases centeredResidue_cases hPair.1 hPair.2.1 hPair.2.2 with
    h | h | h | h | h <;>
    rcases h with ⟨hLeft, hRight⟩ <;>
    simp [ReducedMod5Witness.residueIndex,
      ReducedMod5Witness.centeredResidue, hLeft, hRight, alphabetSize]

theorem centeredResiduePair_index_injective
    {left right : ReducedMod5Witness}
    (hLeft : CenteredResiduePairHolds left)
    (hRight : CenteredResiduePairHolds right)
    (hIndex : left.residueIndex = right.residueIndex) :
    left.residueLeft = right.residueLeft ∧
      left.residueRight = right.residueRight := by
  rcases centeredResidue_cases hLeft.1 hLeft.2.1 hLeft.2.2 with
    hl | hl | hl | hl | hl <;>
    rcases hl with ⟨hl0, hl1⟩ <;>
    rcases centeredResidue_cases hRight.1 hRight.2.1 hRight.2.2 with
      hr | hr | hr | hr | hr <;>
    rcases hr with ⟨hr0, hr1⟩ <;>
    simp_all [ReducedMod5Witness.residueIndex,
      ReducedMod5Witness.centeredResidue]

private def canonicalCenteredPair (index : Nat) : Int × Int :=
  match index with
  | 0 => (-1, -1)
  | 1 => (-1, 0)
  | 2 => (0, 0)
  | 3 => (1, 0)
  | 4 => (1, 1)
  | _ => (0, 0)

private theorem canonicalCenteredPair_holds
    {index : Nat} (hIndex : index < alphabetSize) :
    let pair := canonicalCenteredPair index
    CenteredUnit pair.1 ∧ CenteredUnit pair.2 ∧
      pair.2 * (pair.1 - pair.2) = 0 ∧
      (pair.1 + pair.2 + 2).toNat = index := by
  simp only [alphabetSize] at hIndex
  interval_cases index <;>
    norm_num [canonicalCenteredPair, CenteredUnit, Int.toNat]

private def canonicalReducedWitness (chunk : Chunk) : ReducedMod5Witness :=
  let pair := canonicalCenteredPair (residue chunk)
  { quotientLow := BitVec.ofNat lowQuotientBits (quotient chunk)
    residueLeft := pair.1
    residueRight := pair.2 }

private theorem canonicalReduced_lowValue (chunk : Chunk) :
    (canonicalReducedWitness chunk).quotientLowValue =
      quotient chunk % 2 ^ lowQuotientBits := by
  simp [canonicalReducedWitness, ReducedMod5Witness.quotientLowValue,
    BitVec.toNat_ofNat]

private theorem canonicalReduced_index (chunk : Chunk) :
    (canonicalReducedWitness chunk).residueIndex = residue chunk := by
  have hResidue := residue_lt_alphabet chunk
  simpa [canonicalReducedWitness, ReducedMod5Witness.residueIndex,
    ReducedMod5Witness.centeredResidue] using
      (canonicalCenteredPair_holds hResidue).2.2.2

private theorem quotient_split (chunk : Chunk) :
    quotient chunk =
      quotient chunk % 2 ^ lowQuotientBits +
        2 ^ lowQuotientBits * (quotient chunk / 2 ^ lowQuotientBits) := by
  have hPositive : 0 < 2 ^ lowQuotientBits := Nat.two_pow_pos _
  have hDivMod := Nat.mod_add_div (quotient chunk) (2 ^ lowQuotientBits)
  omega

private theorem quotient_high_lt_two (chunk : Chunk) :
    quotient chunk / 2 ^ lowQuotientBits < 2 := by
  have hFits := quotient_fits_bits chunk
  simp only [quotientBits, lowQuotientBits] at hFits ⊢
  omega

private theorem canonicalReduced_high (chunk : Chunk) :
    derivedQuotientHigh chunk (canonicalReducedWitness chunk) =
      quotient chunk / 2 ^ lowQuotientBits := by
  have hChunk := chunk_decomposition chunk
  have hSplit := quotient_split chunk
  have hLow := canonicalReduced_lowValue chunk
  have hIndex := canonicalReduced_index chunk
  simp only [derivedQuotientHigh, alphabetSize] at ⊢
  rw [hLow, hIndex]
  simp only [alphabetSize] at hChunk
  have hChunkExpanded :
      chunk.val =
        5 * (quotient chunk % 2 ^ lowQuotientBits +
          2 ^ lowQuotientBits *
            (quotient chunk / 2 ^ lowQuotientBits)) + residue chunk := by
    calc
      chunk.val = 5 * quotient chunk + residue chunk := hChunk
      _ = 5 * (quotient chunk % 2 ^ lowQuotientBits +
          2 ^ lowQuotientBits *
            (quotient chunk / 2 ^ lowQuotientBits)) + residue chunk := by
              exact congrArg
                (fun value => 5 * value + residue chunk) hSplit
  have hNumerator :
      chunk.val - (5 * (quotient chunk % 2 ^ lowQuotientBits) + residue chunk) =
        5 * 2 ^ lowQuotientBits * (quotient chunk / 2 ^ lowQuotientBits) := by
    calc
      chunk.val -
          (5 * (quotient chunk % 2 ^ lowQuotientBits) + residue chunk) =
          (5 * (quotient chunk % 2 ^ lowQuotientBits +
              2 ^ lowQuotientBits *
                (quotient chunk / 2 ^ lowQuotientBits)) + residue chunk) -
            (5 * (quotient chunk % 2 ^ lowQuotientBits) + residue chunk) := by
              rw [hChunkExpanded]
      _ = 5 *
          (2 ^ lowQuotientBits *
            (quotient chunk / 2 ^ lowQuotientBits)) := by omega
      _ = 5 * 2 ^ lowQuotientBits *
          (quotient chunk / 2 ^ lowQuotientBits) := by
            simp only [Nat.mul_assoc]
  rw [hNumerator]
  have hPositive : 0 < 5 * 2 ^ lowQuotientBits := by positivity
  simpa only [Nat.mul_comm] using
    (Nat.mul_div_left (quotient chunk / 2 ^ lowQuotientBits) hPositive)

private theorem canonicalReduced_holds (chunk : Chunk) :
    ReducedMod5Holds chunk (canonicalReducedWitness chunk) := by
  have hResidue := residue_lt_alphabet chunk
  have hPair := canonicalCenteredPair_holds hResidue
  have hHigh := canonicalReduced_high chunk
  have hHighLt := quotient_high_lt_two chunk
  have hChunk := chunk_decomposition chunk
  have hSplit := quotient_split chunk
  refine ⟨?_, ?_, ?_⟩
  · exact ⟨hPair.1, hPair.2.1, hPair.2.2.1⟩
  · simpa [hHigh] using hHighLt
  · rw [canonicalReduced_index, ReducedMod5Witness.quotientValue,
      canonicalReduced_lowValue, canonicalReduced_high]
    calc
      chunk.val = alphabetSize * quotient chunk + residue chunk := hChunk
      _ = alphabetSize *
          (quotient chunk % 2 ^ lowQuotientBits +
            2 ^ lowQuotientBits *
              (quotient chunk / 2 ^ lowQuotientBits)) +
            residue chunk := by rw [← hSplit]

/-- The reduced witness decodes to the source quotient/residue witness. -/
def reducedArithmeticWitness
    (chunk : Chunk) (witness : ReducedMod5Witness) : ChunkArithmeticWitness :=
  { quotient := witness.quotientValue chunk
    residue := witness.residueIndex }

theorem reducedMod5_sound
    {chunk : Chunk} {witness : ReducedMod5Witness}
    (hReduced : ReducedMod5Holds chunk witness) :
    ChunkArithmeticHolds chunk (reducedArithmeticWitness chunk witness) := by
  rcases hReduced with ⟨hPair, hHigh, hChunk⟩
  refine ⟨?_, centeredResiduePair_index_lt hPair, hChunk⟩
  have hLow : witness.quotientLowValue < 2 ^ lowQuotientBits :=
    witness.quotientLow.isLt
  change witness.quotientValue chunk < 2 ^ quotientBits
  simp only [ReducedMod5Witness.quotientValue]
  norm_num [lowQuotientBits, quotientBits] at hLow ⊢
  omega

theorem reducedMod5_complete
    {chunk : Chunk} {source : ChunkArithmeticWitness}
    (hSource : ChunkArithmeticHolds chunk source) :
    ∃ reduced,
      ReducedMod5Holds chunk reduced ∧
        reducedArithmeticWitness chunk reduced = source := by
  refine ⟨canonicalReducedWitness chunk, canonicalReduced_holds chunk, ?_⟩
  have hUnique := arithmeticWitness_unique chunk source hSource
  cases source with
  | mk sourceQuotient sourceResidue =>
      simp only at hUnique ⊢
      simp only [reducedArithmeticWitness]
      congr
      · rw [ReducedMod5Witness.quotientValue, canonicalReduced_lowValue,
          canonicalReduced_high, ← quotient_split]
        exact hUnique.1.symm
      · rw [canonicalReduced_index]
        exact hUnique.2.symm

/-- Existential source rows and the conservative reduced rows accept identically. -/
theorem reducedMod5_iff_chunkArithmetic (chunk : Chunk) :
    (∃ reduced, ReducedMod5Holds chunk reduced) ↔
      ∃ source, ChunkArithmeticHolds chunk source := by
  constructor
  · rintro ⟨reduced, hReduced⟩
    exact ⟨reducedArithmeticWitness chunk reduced,
      reducedMod5_sound hReduced⟩
  · rintro ⟨source, hSource⟩
    rcases reducedMod5_complete hSource with ⟨reduced, hReduced, _⟩
    exact ⟨reduced, hReduced⟩

private theorem reducedMod5_low_and_index_unique
    {chunk : Chunk} {left right : ReducedMod5Witness}
    (hLeft : ReducedMod5Holds chunk left)
    (hRight : ReducedMod5Holds chunk right) :
    left.quotientLowValue = right.quotientLowValue ∧
      left.residueIndex = right.residueIndex := by
  have hLeftSource := reducedMod5_sound hLeft
  have hRightSource := reducedMod5_sound hRight
  have hLeftUnique := arithmeticWitness_unique chunk _ hLeftSource
  have hRightUnique := arithmeticWitness_unique chunk _ hRightSource
  have hQuotient :
      left.quotientLowValue + 8192 * derivedQuotientHigh chunk left =
        right.quotientLowValue + 8192 * derivedQuotientHigh chunk right := by
    simpa [reducedArithmeticWitness, ReducedMod5Witness.quotientValue,
      lowQuotientBits] using hLeftUnique.1.trans hRightUnique.1.symm
  have hIndex : left.residueIndex = right.residueIndex := by
    exact hLeftUnique.2.trans hRightUnique.2.symm
  have hLeftLow : left.quotientLowValue < 8192 := by
    simpa [ReducedMod5Witness.quotientLowValue, lowQuotientBits] using
      left.quotientLow.isLt
  have hRightLow : right.quotientLowValue < 8192 := by
    simpa [ReducedMod5Witness.quotientLowValue, lowQuotientBits] using
      right.quotientLow.isLt
  have hLeftHigh := hLeft.2.1
  have hRightHigh := hRight.2.1
  constructor
  · omega
  · exact hIndex

theorem reducedMod5Witness_unique
    {chunk : Chunk} {left right : ReducedMod5Witness}
    (hLeft : ReducedMod5Holds chunk left)
    (hRight : ReducedMod5Holds chunk right) :
    left = right := by
  rcases reducedMod5_low_and_index_unique hLeft hRight with
    ⟨hLow, hIndex⟩
  have hBits : left.quotientLow = right.quotientLow :=
    BitVec.eq_of_toNat_eq hLow
  have hPair := centeredResiduePair_index_injective hLeft.1 hRight.1 hIndex
  cases left
  cases right
  simp_all [ReducedMod5Witness.quotientLowValue]

theorem reducedMod5_exact (chunk : Chunk) :
    ∃! witness, ReducedMod5Holds chunk witness := by
  refine ⟨canonicalReducedWitness chunk, canonicalReduced_holds chunk, ?_⟩
  intro witness hWitness
  exact reducedMod5Witness_unique hWitness (canonicalReduced_holds chunk)

/-! ## Conservative Goldilocks rows -/

/-- Raw Goldilocks cells in the conservative reduced mod-5 block. -/
structure ReducedMod5FieldWitness where
  quotientLow : Fin lowQuotientBits → F
  residueLeft : F
  residueRight : F

/-- Polynomial residual of one ordinary field bit-root equation. -/
def fieldBitResidual (value : F) : F :=
  value * (value - 1)

/-- One ordinary field bit-root equation. -/
def FieldBitRoot (value : F) : Prop :=
  fieldBitResidual value = 0

/-- Polynomial residual of one centered cubic root equation. -/
def fieldCenteredResidual (value : F) : F :=
  value * (value - 1) * (value + 1)

/-- One centered cubic root equation, with roots `-1`, `0`, and `1`. -/
def FieldCenteredRoot (value : F) : Prop :=
  fieldCenteredResidual value = 0

/-- Little-endian linear value of a fixed field-bit vector. -/
def fieldBitsValue : (width : Nat) → (Fin width → F) → F
  | 0, _ => 0
  | width + 1, values =>
      2 * fieldBitsValue width (values ∘ Fin.succ) + values 0

/-- Low quotient represented by the thirteen explicit field cells. -/
def ReducedMod5FieldWitness.quotientLowValue
    (witness : ReducedMod5FieldWitness) : F :=
  fieldBitsValue lowQuotientBits witness.quotientLow

/-- Unsigned residue represented by the two centered cells. -/
def ReducedMod5FieldWitness.residueIndex
    (witness : ReducedMod5FieldWitness) : F :=
  witness.residueLeft + witness.residueRight + 2

/-- Polynomial residual selecting the five canonical centered pairs. -/
def ReducedMod5FieldWitness.residuePairResidual
    (witness : ReducedMod5FieldWitness) : F :=
  witness.residueRight * (witness.residueLeft - witness.residueRight)

/-- Fixed nonzero coefficient multiplying the linearly derived high bit. -/
private def mod5HighDenominator : Nat :=
  alphabetSize * 2 ^ lowQuotientBits

private theorem mod5HighDenominator_lt_q :
    mod5HighDenominator < Goldilocks.q := by
  norm_num [mod5HighDenominator, alphabetSize, lowQuotientBits, quotientBits,
    Goldilocks.q]

private theorem mod5HighDenominator_ne_zero :
    F.ofNat mod5HighDenominator ≠ 0 := by
  intro hZero
  have hVal := congrArg Fin.val hZero
  rw [F.ofNat_val_eq_of_canonical mod5HighDenominator_lt_q,
    F.val_zero] at hVal
  norm_num [mod5HighDenominator, alphabetSize, lowQuotientBits, quotientBits]
    at hVal

/--
The high quotient bit as a linear field expression. It is not a witness cell:
the fixed denominator is inverted in the row coefficient.
-/
noncomputable def derivedQuotientHighField
    (chunk : Chunk) (witness : ReducedMod5FieldWitness) : F :=
  (F.ofNat chunk.val -
      (F.ofNat alphabetSize * witness.quotientLowValue + witness.residueIndex)) /
    F.ofNat mod5HighDenominator

/--
Exact 16-row/15-cell Goldilocks relation: thirteen low bit roots, one root
for the linearly derived high bit, one left-centered cubic root, and the pair
row. The pair row forces the right coordinate to be zero or equal to the left,
so its centered cubic is derived rather than emitted. The derived-high
definition is the field reconstruction equation rearranged through a proved
nonzero fixed coefficient; it allocates no cell or row.
-/
def ReducedMod5FieldRows
    (chunk : Chunk) (witness : ReducedMod5FieldWitness) : Prop :=
  (∀ index, FieldBitRoot (witness.quotientLow index)) ∧
    FieldBitRoot (derivedQuotientHighField chunk witness) ∧
    FieldCenteredRoot witness.residueLeft ∧
    witness.residuePairResidual = 0

private theorem fieldBitRoot_cases
    {value : F} (hRoot : FieldBitRoot value) :
    value = 0 ∨ value = 1 := by
  rcases mul_eq_zero.mp hRoot with hZero | hOne
  · exact Or.inl hZero
  · exact Or.inr (sub_eq_zero.mp hOne)

private theorem fieldCenteredRoot_cases
    {value : F} (hRoot : FieldCenteredRoot value) :
    value = -1 ∨ value = 0 ∨ value = 1 := by
  rcases mul_eq_zero.mp hRoot with hZeroOne | hNeg
  · rcases mul_eq_zero.mp hZeroOne with hZero | hOne
    · exact Or.inr (Or.inl hZero)
    · exact Or.inr (Or.inr (sub_eq_zero.mp hOne))
  · exact Or.inl (eq_neg_of_add_eq_zero_left hNeg)

private theorem field_two_ne_zero : (2 : F) ≠ 0 := by
  intro hZero
  have hVal := congrArg Fin.val hZero
  norm_num [F.ofNat, F.val_zero, Goldilocks.q] at hVal

private theorem field_one_ne_neg_one : (1 : F) ≠ -1 := by
  intro hEq
  apply field_two_ne_zero
  calc
    (2 : F) = 1 + 1 := by norm_num
    _ = -1 + 1 := congrArg (fun value : F => value + 1) hEq
    _ = 0 := by ring

/--
The right centered cubic is redundant: the pair equation makes the right
coordinate zero or equal to the already-centered left coordinate.
-/
theorem fieldCenteredPair_implies_rightRoot
    {left right : F}
    (hLeft : FieldCenteredRoot left)
    (hPair : right * (left - right) = 0) :
    FieldCenteredRoot right := by
  rcases mul_eq_zero.mp hPair with hRightZero | hEqual
  · simp [hRightZero, FieldCenteredRoot, fieldCenteredResidual]
  · have hLeftRight : left = right := sub_eq_zero.mp hEqual
    rwa [← hLeftRight]

private theorem fieldCenteredPair_cases
    {left right : F}
    (hLeft : FieldCenteredRoot left)
    (hPair : right * (left - right) = 0) :
    (left = -1 ∧ right = -1) ∨
      (left = -1 ∧ right = 0) ∨
      (left = 0 ∧ right = 0) ∨
      (left = 1 ∧ right = 0) ∨
      (left = 1 ∧ right = 1) := by
  rcases mul_eq_zero.mp hPair with hRightZero | hEqual
  · rcases fieldCenteredRoot_cases hLeft with hLeft | hLeft | hLeft <;>
      simp_all
  · have hLeftRight : left = right := sub_eq_zero.mp hEqual
    rcases fieldCenteredRoot_cases hLeft with hLeft | hLeft | hLeft <;>
      simp_all

private def fieldBitBool (value : F) : Bool :=
  decide (value = 1)

private theorem fieldBit_eq_encoded
    {value : F} (hRoot : FieldBitRoot value) :
    value = if fieldBitBool value then 1 else 0 := by
  rcases fieldBitRoot_cases hRoot with hZero | hOne
  · simp [fieldBitBool, hZero]
  · simp [fieldBitBool, hOne]

private theorem fieldOfNat_mul (left right : Nat) :
    F.ofNat (left * right) = F.ofNat left * F.ofNat right := by
  apply Fin.ext
  simp [F.ofNat, Nat.mul_mod]

private theorem derivedQuotientHighField_reconstruction
    (chunk : Chunk) (witness : ReducedMod5FieldWitness) :
    F.ofNat chunk.val =
      F.ofNat alphabetSize *
          (witness.quotientLowValue +
            F.ofNat (2 ^ lowQuotientBits) *
              derivedQuotientHighField chunk witness) +
        witness.residueIndex := by
  have hCoefficient :
      F.ofNat alphabetSize * F.ofNat (2 ^ lowQuotientBits) =
        F.ofNat mod5HighDenominator := by
    exact (fieldOfNat_mul alphabetSize (2 ^ lowQuotientBits)).symm
  have hCancel :
      F.ofNat mod5HighDenominator *
          derivedQuotientHighField chunk witness =
        F.ofNat chunk.val -
          (F.ofNat alphabetSize * witness.quotientLowValue +
            witness.residueIndex) := by
    unfold derivedQuotientHighField
    exact mul_div_cancel₀ _ mod5HighDenominator_ne_zero
  calc
    F.ofNat chunk.val =
        (F.ofNat alphabetSize * witness.quotientLowValue +
            witness.residueIndex) +
          (F.ofNat chunk.val -
            (F.ofNat alphabetSize * witness.quotientLowValue +
              witness.residueIndex)) := by ring
    _ = (F.ofNat alphabetSize * witness.quotientLowValue +
            witness.residueIndex) +
          F.ofNat mod5HighDenominator *
            derivedQuotientHighField chunk witness := by rw [hCancel]
    _ = F.ofNat alphabetSize *
          (witness.quotientLowValue +
            F.ofNat (2 ^ lowQuotientBits) *
              derivedQuotientHighField chunk witness) +
          witness.residueIndex := by rw [← hCoefficient]; ring

private theorem fieldBitsValue_encoded :
    ∀ {width : Nat} (bits : Fin width → Bool),
      fieldBitsValue width (fun index => if bits index then 1 else 0) =
        F.ofNat (Nat.ofBits bits) := by
  intro width bits
  induction width with
  | zero => simp [fieldBitsValue]
  | succ width ih =>
      rw [fieldBitsValue]
      have hTail :
          fieldBitsValue width
              (fun index => if bits (Fin.succ index) then 1 else 0) =
            F.ofNat (Nat.ofBits (bits ∘ Fin.succ)) := by
        simpa only [Function.comp_apply] using ih (bits ∘ Fin.succ)
      change
        2 * fieldBitsValue width
              (fun index => if bits (Fin.succ index) then 1 else 0) +
            (if bits 0 then 1 else 0) =
          F.ofNat (Nat.ofBits bits)
      rw [hTail]
      cases hBit : bits 0 <;>
        simp [Nat.ofBits_succ, hBit, fieldOfNat_mul]

private theorem fieldOfNat_eq_iff_of_lt
    {left right : Nat}
    (hLeft : left < Goldilocks.q)
    (hRight : right < Goldilocks.q) :
    F.ofNat left = F.ofNat right ↔ left = right := by
  constructor
  · intro hField
    have hVal := congrArg Fin.val hField
    simpa [F.ofNat_val_eq_of_canonical hLeft,
      F.ofNat_val_eq_of_canonical hRight] using hVal
  · exact fun h => congrArg F.ofNat h

private def fieldCenteredInt (value : F) : Int :=
  if value = -1 then -1 else if value = 0 then 0 else 1

private theorem fieldCenteredInt_spec
    {value : F} (hRoot : FieldCenteredRoot value) :
    CenteredUnit (fieldCenteredInt value) ∧
      (fieldCenteredInt value : F) = value := by
  rcases fieldCenteredRoot_cases hRoot with hNeg | hZero | hOne
  · simp [fieldCenteredInt, hNeg, CenteredUnit]
  · simp [fieldCenteredInt, hZero, CenteredUnit]
  · simp [fieldCenteredInt, hOne, CenteredUnit,
      field_one_ne_neg_one]

private def decodeReducedMod5FieldWitness
    (witness : ReducedMod5FieldWitness) : ReducedMod5Witness :=
  { quotientLow := BitVec.ofFnLE fun index =>
      fieldBitBool (witness.quotientLow index)
    residueLeft := fieldCenteredInt witness.residueLeft
    residueRight := fieldCenteredInt witness.residueRight }

private def decodedFieldBit (value : F) : Nat :=
  (fieldBitBool value).toNat

private def encodeReducedMod5FieldWitness
    (witness : ReducedMod5Witness) : ReducedMod5FieldWitness :=
  { quotientLow := fun index => if witness.quotientBit index then 1 else 0
    residueLeft := witness.residueLeft
    residueRight := witness.residueRight }

/-- Exact coordinate correspondence between raw field cells and a Nat witness. -/
def ReducedMod5FieldRepresents
    (chunk : Chunk) (fieldWitness : ReducedMod5FieldWitness)
    (natWitness : ReducedMod5Witness) : Prop :=
  (∀ index,
      fieldWitness.quotientLow index =
        if natWitness.quotientBit index then 1 else 0) ∧
    derivedQuotientHighField chunk fieldWitness =
      F.ofNat (derivedQuotientHigh chunk natWitness) ∧
    fieldWitness.residueLeft = natWitness.residueLeft ∧
    fieldWitness.residueRight = natWitness.residueRight

private theorem decodedLow_fieldValue
    {witness : ReducedMod5FieldWitness}
    (hBits : ∀ index, FieldBitRoot (witness.quotientLow index)) :
    fieldBitsValue lowQuotientBits witness.quotientLow =
      F.ofNat (decodeReducedMod5FieldWitness witness).quotientLowValue := by
  have hCoordinates :
      witness.quotientLow = fun index =>
        if fieldBitBool (witness.quotientLow index) then 1 else 0 := by
    funext index
    exact fieldBit_eq_encoded (hBits index)
  rw [hCoordinates, fieldBitsValue_encoded]
  simp [decodeReducedMod5FieldWitness,
    ReducedMod5Witness.quotientLowValue]

private theorem decodedLow_coordinates
    {witness : ReducedMod5FieldWitness}
    (hBits : ∀ index, FieldBitRoot (witness.quotientLow index)) :
    ∀ index,
      witness.quotientLow index =
        if (decodeReducedMod5FieldWitness witness).quotientBit index then 1 else 0 := by
  intro index
  rw [fieldBit_eq_encoded (hBits index)]
  simp [decodeReducedMod5FieldWitness,
    ReducedMod5Witness.quotientBit]

private theorem decodedHigh_spec
    {value : F}
    (hHigh : FieldBitRoot value) :
    decodedFieldBit value < 2 ∧
      value = F.ofNat (decodedFieldBit value) := by
  rcases fieldBitRoot_cases hHigh with hZero | hOne
  · simp [decodedFieldBit, fieldBitBool, hZero]
  · simp [decodedFieldBit, fieldBitBool, hOne]

private theorem decodedCentered_spec
    {witness : ReducedMod5FieldWitness}
    (hLeft : FieldCenteredRoot witness.residueLeft)
    (hPair :
      witness.residueRight * (witness.residueLeft - witness.residueRight) = 0) :
    let decoded := decodeReducedMod5FieldWitness witness
    CenteredResiduePairHolds decoded ∧
      witness.residueIndex = F.ofNat decoded.residueIndex ∧
      witness.residueLeft = (decoded.residueLeft : F) ∧
      witness.residueRight = (decoded.residueRight : F) := by
  rcases fieldCenteredPair_cases hLeft hPair with
    h | h | h | h | h <;>
    rcases h with ⟨hLeftValue, hRightValue⟩ <;>
    simp [decodeReducedMod5FieldWitness, fieldCenteredInt,
      ReducedMod5FieldWitness.residueIndex,
      ReducedMod5Witness.residueIndex,
      ReducedMod5Witness.centeredResidue,
      CenteredResiduePairHolds, CenteredUnit,
      hLeftValue, hRightValue, field_one_ne_neg_one] <;>
    ring

/--
The conservative field rows decode without modular ambiguity. Both sides of
the reconstruction are proved below the Goldilocks modulus before field
equality is converted back to a natural-number equality.
-/
theorem reducedMod5FieldRows_sound
    {chunk : Chunk} {fieldWitness : ReducedMod5FieldWitness}
    (hRows : ReducedMod5FieldRows chunk fieldWitness) :
    ∃ natWitness,
      ReducedMod5Holds chunk natWitness ∧
        ReducedMod5FieldRepresents chunk fieldWitness natWitness := by
  rcases hRows with
    ⟨hBits, hHighRoot, hLeftRoot, hPair⟩
  let natWitness := decodeReducedMod5FieldWitness fieldWitness
  let high := decodedFieldBit (derivedQuotientHighField chunk fieldWitness)
  have hLowField :
      fieldBitsValue lowQuotientBits fieldWitness.quotientLow =
        F.ofNat natWitness.quotientLowValue := by
    simpa [natWitness] using decodedLow_fieldValue hBits
  have hLowBound : natWitness.quotientLowValue < 2 ^ lowQuotientBits := by
    exact natWitness.quotientLow.isLt
  have hHighSpec := decodedHigh_spec hHighRoot
  have hHighBound : high < 2 := by
    simpa [high] using hHighSpec.1
  have hHighField :
      derivedQuotientHighField chunk fieldWitness = F.ofNat high := by
    simpa [high] using hHighSpec.2
  have hCenteredSpec := decodedCentered_spec hLeftRoot hPair
  have hCentered : CenteredResiduePairHolds natWitness := by
    simpa [natWitness] using hCenteredSpec.1
  have hResidueField :
      fieldWitness.residueIndex = F.ofNat natWitness.residueIndex := by
    simpa [natWitness] using hCenteredSpec.2.1
  let reconstructed :=
    alphabetSize *
        (natWitness.quotientLowValue + 2 ^ lowQuotientBits * high) +
      natWitness.residueIndex
  have hFieldReconstructed :
      F.ofNat chunk.val = F.ofNat reconstructed := by
    have hFieldEquation :=
      derivedQuotientHighField_reconstruction chunk fieldWitness
    calc
      F.ofNat chunk.val =
          F.ofNat alphabetSize *
              (fieldWitness.quotientLowValue +
                F.ofNat (2 ^ lowQuotientBits) *
                  derivedQuotientHighField chunk fieldWitness) +
            fieldWitness.residueIndex := hFieldEquation
      _ = F.ofNat reconstructed := by
        rw [ReducedMod5FieldWitness.quotientLowValue, hLowField,
          hHighField, hResidueField]
        simp [reconstructed, F.ofNat_add, fieldOfNat_mul]
  have hChunkLtQ : chunk.val < Goldilocks.q := by
    exact lt_trans chunk.isLt (by decide)
  have hResidueBound : natWitness.residueIndex < alphabetSize :=
    centeredResiduePair_index_lt hCentered
  have hReconstructedSmall : reconstructed < 81_920 := by
    simp only [reconstructed]
    norm_num [alphabetSize, lowQuotientBits, quotientBits] at hLowBound hResidueBound
    norm_num [alphabetSize, lowQuotientBits, quotientBits]
    omega
  have hReconstructedLtQ : reconstructed < Goldilocks.q :=
    lt_trans hReconstructedSmall (by decide)
  have hNatEquation : chunk.val = reconstructed :=
    (fieldOfNat_eq_iff_of_lt hChunkLtQ hReconstructedLtQ).mp
      hFieldReconstructed
  have hNatExpanded :
      chunk.val =
        alphabetSize *
            (natWitness.quotientLowValue + 2 ^ lowQuotientBits * high) +
          natWitness.residueIndex := by
    simpa only [reconstructed] using hNatEquation
  have hNumerator :
      chunk.val -
          (alphabetSize * natWitness.quotientLowValue +
            natWitness.residueIndex) =
        alphabetSize * 2 ^ lowQuotientBits * high := by
    calc
      chunk.val -
          (alphabetSize * natWitness.quotientLowValue +
            natWitness.residueIndex) =
          (alphabetSize *
              (natWitness.quotientLowValue +
                2 ^ lowQuotientBits * high) +
              natWitness.residueIndex) -
            (alphabetSize * natWitness.quotientLowValue +
              natWitness.residueIndex) := by rw [hNatExpanded]
      _ = alphabetSize * 2 ^ lowQuotientBits * high := by
        simp only [Nat.mul_add, Nat.mul_assoc]
        omega
  have hDerived : derivedQuotientHigh chunk natWitness = high := by
    simp only [derivedQuotientHigh]
    rw [hNumerator]
    have hPositive : 0 < alphabetSize * 2 ^ lowQuotientBits := by
      norm_num [alphabetSize, lowQuotientBits]
    simpa only [Nat.mul_comm] using
      (Nat.mul_div_left high hPositive)
  have hReduced : ReducedMod5Holds chunk natWitness := by
    refine ⟨hCentered, ?_, ?_⟩
    · simpa [hDerived] using hHighBound
    · rw [ReducedMod5Witness.quotientValue, hDerived]
      simpa [reconstructed] using hNatEquation
  refine ⟨natWitness, hReduced, ?_, ?_, ?_, ?_⟩
  · simpa [natWitness] using decodedLow_coordinates hBits
  · calc
      derivedQuotientHighField chunk fieldWitness = F.ofNat high := hHighField
      _ = F.ofNat (derivedQuotientHigh chunk natWitness) := by rw [hDerived]
  · simpa [natWitness] using hCenteredSpec.2.2.1
  · simpa [natWitness] using hCenteredSpec.2.2.2

private theorem encodedLow_fieldValue
    (witness : ReducedMod5Witness) :
    fieldBitsValue lowQuotientBits
        (fun index => if witness.quotientBit index then 1 else 0) =
      F.ofNat witness.quotientLowValue := by
  rw [fieldBitsValue_encoded]
  have hVector :
      BitVec.ofFnLE (fun index => witness.quotientBit index) =
        witness.quotientLow := by
    apply BitVec.eq_of_getElem_eq
    intro index hIndex
    simp [ReducedMod5Witness.quotientBit]
  have hNat :
      Nat.ofBits (fun index => witness.quotientBit index) =
        witness.quotientLow.toNat := by
    calc
      Nat.ofBits (fun index => witness.quotientBit index) =
          (BitVec.ofFnLE fun index => witness.quotientBit index).toNat := by
            symm
            exact BitVec.toNat_ofFnLE _
      _ = witness.quotientLow.toNat := congrArg BitVec.toNat hVector
  exact congrArg F.ofNat hNat

/-- Every valid Nat witness has a coordinate-identical conservative field lift. -/
theorem reducedMod5FieldRows_complete
    {chunk : Chunk} {natWitness : ReducedMod5Witness}
    (hReduced : ReducedMod5Holds chunk natWitness) :
    ReducedMod5FieldRows chunk
        (encodeReducedMod5FieldWitness natWitness) ∧
      ReducedMod5FieldRepresents chunk
        (encodeReducedMod5FieldWitness natWitness) natWitness := by
  rcases hReduced with ⟨hCentered, hHigh, hChunk⟩
  have hLowRoots :
      ∀ index,
        FieldBitRoot
          ((encodeReducedMod5FieldWitness natWitness).quotientLow index) := by
    intro index
    cases hBit : natWitness.quotientBit index <;>
      simp [encodeReducedMod5FieldWitness, FieldBitRoot, fieldBitResidual, hBit]
  have hLeftRoot :
      FieldCenteredRoot
        (encodeReducedMod5FieldWitness natWitness).residueLeft := by
    rcases hCentered.1 with hNeg | hZero | hOne
    · simp [encodeReducedMod5FieldWitness, FieldCenteredRoot,
        fieldCenteredResidual, hNeg]
    · simp [encodeReducedMod5FieldWitness, FieldCenteredRoot,
        fieldCenteredResidual, hZero]
    · simp [encodeReducedMod5FieldWitness, FieldCenteredRoot,
        fieldCenteredResidual, hOne]
  have hCenteredCases := centeredResidue_cases
    hCentered.1 hCentered.2.1 hCentered.2.2
  have hFieldPair :
      (encodeReducedMod5FieldWitness natWitness).residueRight *
          ((encodeReducedMod5FieldWitness natWitness).residueLeft -
            (encodeReducedMod5FieldWitness natWitness).residueRight) = 0 := by
    rcases hCenteredCases with h | h | h | h | h <;>
      rcases h with ⟨hLeft, hRight⟩ <;>
      simp [encodeReducedMod5FieldWitness, hLeft, hRight]
  have hResidueField :
      (encodeReducedMod5FieldWitness natWitness).residueIndex =
        F.ofNat natWitness.residueIndex := by
    rcases hCenteredCases with h | h | h | h | h <;>
      rcases h with ⟨hLeft, hRight⟩ <;>
      simp [encodeReducedMod5FieldWitness,
        ReducedMod5FieldWitness.residueIndex,
        ReducedMod5Witness.residueIndex,
        ReducedMod5Witness.centeredResidue,
        hLeft, hRight] <;>
      ring
  have hLowField :
      fieldBitsValue lowQuotientBits
          (encodeReducedMod5FieldWitness natWitness).quotientLow =
        F.ofNat natWitness.quotientLowValue := by
    simpa [encodeReducedMod5FieldWitness] using
      encodedLow_fieldValue natWitness
  have hExpectedFieldEquation :
      F.ofNat chunk.val =
        F.ofNat alphabetSize *
            (F.ofNat natWitness.quotientLowValue +
              F.ofNat (2 ^ lowQuotientBits) *
                F.ofNat (derivedQuotientHigh chunk natWitness)) +
          F.ofNat natWitness.residueIndex := by
    calc
      F.ofNat chunk.val =
          F.ofNat
            (alphabetSize * natWitness.quotientValue chunk +
              natWitness.residueIndex) := congrArg F.ofNat hChunk
      _ = F.ofNat alphabetSize *
            (F.ofNat natWitness.quotientLowValue +
              F.ofNat (2 ^ lowQuotientBits) *
                F.ofNat (derivedQuotientHigh chunk natWitness)) +
            F.ofNat natWitness.residueIndex := by
              simp [ReducedMod5Witness.quotientValue,
                F.ofNat_add, fieldOfNat_mul]
  have hActualFieldEquation :=
    derivedQuotientHighField_reconstruction chunk
      (encodeReducedMod5FieldWitness natWitness)
  rw [ReducedMod5FieldWitness.quotientLowValue,
    hLowField, hResidueField] at hActualFieldEquation
  have hCoefficient :
      F.ofNat alphabetSize * F.ofNat (2 ^ lowQuotientBits) =
        F.ofNat mod5HighDenominator := by
    exact (fieldOfNat_mul alphabetSize (2 ^ lowQuotientBits)).symm
  have hReconstructionSides :
      F.ofNat alphabetSize *
            (F.ofNat natWitness.quotientLowValue +
              F.ofNat (2 ^ lowQuotientBits) *
                derivedQuotientHighField chunk
                  (encodeReducedMod5FieldWitness natWitness)) +
          F.ofNat natWitness.residueIndex =
        F.ofNat alphabetSize *
            (F.ofNat natWitness.quotientLowValue +
              F.ofNat (2 ^ lowQuotientBits) *
                F.ofNat (derivedQuotientHigh chunk natWitness)) +
          F.ofNat natWitness.residueIndex :=
    hActualFieldEquation.symm.trans hExpectedFieldEquation
  have hCoefficientProduct :
      (F.ofNat alphabetSize * F.ofNat (2 ^ lowQuotientBits)) *
          derivedQuotientHighField chunk
            (encodeReducedMod5FieldWitness natWitness) =
        (F.ofNat alphabetSize * F.ofNat (2 ^ lowQuotientBits)) *
          F.ofNat (derivedQuotientHigh chunk natWitness) := by
    linear_combination hReconstructionSides
  have hDerivedProduct :
      F.ofNat mod5HighDenominator *
          derivedQuotientHighField chunk
            (encodeReducedMod5FieldWitness natWitness) =
        F.ofNat mod5HighDenominator *
          F.ofNat (derivedQuotientHigh chunk natWitness) := by
    simpa only [hCoefficient] using hCoefficientProduct
  have hDerivedField :
      derivedQuotientHighField chunk
          (encodeReducedMod5FieldWitness natWitness) =
        F.ofNat (derivedQuotientHigh chunk natWitness) :=
    mul_left_cancel₀ mod5HighDenominator_ne_zero hDerivedProduct
  have hHighRoot :
      FieldBitRoot
        (derivedQuotientHighField chunk
          (encodeReducedMod5FieldWitness natWitness)) := by
    rw [hDerivedField]
    have hCases :
        derivedQuotientHigh chunk natWitness = 0 ∨
          derivedQuotientHigh chunk natWitness = 1 := by
      omega
    rcases hCases with hZero | hOne
    · simp [FieldBitRoot, fieldBitResidual, hZero]
    · simp [FieldBitRoot, fieldBitResidual, hOne]
  constructor
  · exact ⟨hLowRoots, hHighRoot, hLeftRoot, hFieldPair⟩
  · refine ⟨?_, hDerivedField, rfl, rfl⟩
    intro index
    rfl

private theorem fieldWitness_eq_of_represents
    {chunk : Chunk} {left right : ReducedMod5FieldWitness}
    {natWitness : ReducedMod5Witness}
    (hLeft : ReducedMod5FieldRepresents chunk left natWitness)
    (hRight : ReducedMod5FieldRepresents chunk right natWitness) :
    left = right := by
  have hLow : left.quotientLow = right.quotientLow := by
    funext index
    exact (hLeft.1 index).trans (hRight.1 index).symm
  have hLeftResidue : left.residueLeft = right.residueLeft :=
    hLeft.2.2.1.trans hRight.2.2.1.symm
  have hRightResidue : left.residueRight = right.residueRight :=
    hLeft.2.2.2.trans hRight.2.2.2.symm
  cases left
  cases right
  simp_all

/--
The sixteen-equation Goldilocks relation holds exactly for the coordinate
encoding of a valid Nat reduced witness.
-/
theorem reducedMod5FieldRows_iff_nat
    (chunk : Chunk) (fieldWitness : ReducedMod5FieldWitness) :
    ReducedMod5FieldRows chunk fieldWitness ↔
      ∃ natWitness,
        ReducedMod5Holds chunk natWitness ∧
          ReducedMod5FieldRepresents chunk fieldWitness natWitness := by
  constructor
  · exact reducedMod5FieldRows_sound
  · rintro ⟨natWitness, hReduced, hRepresents⟩
    have hComplete := reducedMod5FieldRows_complete hReduced
    have hFieldWitness :
        fieldWitness = encodeReducedMod5FieldWitness natWitness :=
      fieldWitness_eq_of_represents hRepresents hComplete.2
    rw [hFieldWitness]
    exact hComplete.1

/-- The sixteen-equation field relation admits at most one witness. -/
theorem reducedMod5FieldWitness_unique
    {chunk : Chunk} {left right : ReducedMod5FieldWitness}
    (hLeftRows : ReducedMod5FieldRows chunk left)
    (hRightRows : ReducedMod5FieldRows chunk right) :
    left = right := by
  rcases reducedMod5FieldRows_sound hLeftRows with
    ⟨leftNat, hLeftReduced, hLeftRepresents⟩
  rcases reducedMod5FieldRows_sound hRightRows with
    ⟨rightNat, hRightReduced, hRightRepresents⟩
  have hNat : leftNat = rightNat :=
    reducedMod5Witness_unique hLeftReduced hRightReduced
  subst rightNat
  exact fieldWitness_eq_of_represents hLeftRepresents hRightRepresents

/-- Every canonical chunk has exactly one conservative field witness. -/
theorem reducedMod5Field_exact (chunk : Chunk) :
    ∃! fieldWitness, ReducedMod5FieldRows chunk fieldWitness := by
  let natWitness := canonicalReducedWitness chunk
  let fieldWitness := encodeReducedMod5FieldWitness natWitness
  have hRows : ReducedMod5FieldRows chunk fieldWitness :=
    (reducedMod5FieldRows_complete (canonicalReduced_holds chunk)).1
  refine ⟨fieldWitness, hRows, ?_⟩
  intro other hOther
  exact reducedMod5FieldWitness_unique hOther hRows

/--
Existence of the sixteen-equation field witness is equivalent to existence
of the original source chunk-arithmetic witness.
-/
theorem reducedMod5FieldRows_iff_chunkArithmetic (chunk : Chunk) :
    (∃ fieldWitness, ReducedMod5FieldRows chunk fieldWitness) ↔
      ∃ sourceWitness, ChunkArithmeticHolds chunk sourceWitness := by
  constructor
  · rintro ⟨fieldWitness, hRows⟩
    rcases reducedMod5FieldRows_sound hRows with
      ⟨natWitness, hReduced, _hRepresents⟩
    exact ⟨reducedArithmeticWitness chunk natWitness,
      reducedMod5_sound hReduced⟩
  · rintro ⟨sourceWitness, hSource⟩
    rcases reducedMod5_complete hSource with
      ⟨natWitness, hReduced, _hRepresents⟩
    exact ⟨encodeReducedMod5FieldWitness natWitness,
      (reducedMod5FieldRows_complete hReduced).1⟩

/-- Outer-norm premise for the optional thirteen-coordinate bitness batch. -/
def CenteredNormOne (digits : Fin lowQuotientBits → Int) : Prop :=
  ∀ index, -1 ≤ digits index ∧ digits index ≤ 1

/-- Integer lift of the proposed single ProductSum bitness equation. -/
def LiftedBatchedBitness (digits : Fin lowQuotientBits → Int) : Prop :=
  ∑ index, digits index * (digits index - 1) = 0

private theorem normBounded_bitTerm_nonnegative
    {digits : Fin lowQuotientBits → Int}
    (hNorm : CenteredNormOne digits) (index : Fin lowQuotientBits) :
    0 ≤ digits index * (digits index - 1) := by
  rcases hNorm index with ⟨hLower, hUpper⟩
  interval_cases hDigit : digits index <;> norm_num

/--
Conditional norm-backed optimization: after a concrete outer norm is lifted to
these integer coordinates, one aggregate equation implies all thirteen
individual bitness equations.
-/
theorem normBatch_implies_bits
    {digits : Fin lowQuotientBits → Int}
    (hNorm : CenteredNormOne digits)
    (hBatch : LiftedBatchedBitness digits) :
    ∀ index, digits index = 0 ∨ digits index = 1 := by
  intro index
  have hEach : digits index * (digits index - 1) = 0 :=
    congrFun
      ((Fintype.sum_eq_zero_iff_of_nonneg
        (fun i => normBounded_bitTerm_nonnegative hNorm i)).mp hBatch)
      index
  rcases hNorm index with ⟨hLower, hUpper⟩
  interval_cases hDigit : digits index <;> simp_all

theorem bits_imply_normBatch
    {digits : Fin lowQuotientBits → Int}
    (hBits : ∀ index, digits index = 0 ∨ digits index = 1) :
    CenteredNormOne digits ∧ LiftedBatchedBitness digits := by
  constructor
  · intro index
    rcases hBits index with h | h <;> simp [h]
  · simp only [LiftedBatchedBitness]
    apply Fintype.sum_eq_zero
    intro index
    rcases hBits index with h | h <;> simp [h]

end SuperNeo.FPrimeRecursiveVerifier.PiRlcChallenge
