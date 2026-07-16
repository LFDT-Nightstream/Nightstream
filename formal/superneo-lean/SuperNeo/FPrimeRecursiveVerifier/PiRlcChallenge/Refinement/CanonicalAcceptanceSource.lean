import Batteries.Data.BitVec
import SuperNeo.FPrimeRecursiveVerifier.PiRlcChallenge.Refinement.PackedAcceptanceRows

/-!
Owns: the exact field model of the production inverse-based chunk-acceptance
source rows, including inverse canonicalization for an invertible witness
encoding.

Does not own: the concrete Rust/R1CS trace bridge, the proposed two-row packed lowering,
selector wiring, transcript sampling, or global enough-accepts arithmetic.

Emits constraints: no. Production Rust emits all four modeled rows.

Authority boundary: the sixteen Boolean chunk coordinates define the
little-endian chunk value. The accept and inverse cells are witnesses checked
against that value. Existential acceptance preservation is intentionally kept
separate from pointwise witness preservation for the legacy three-row
relation, which fails on the rejected all-ones chunk.

| Predicate/theorem | Rust stage | Guarantee | Assumptions | Permits Rust row removal? |
|---|---|---|---|---|
| `CurrentAcceptanceSourceRows` | legacy three-row subset of `challenge.sampler.chunk.accept` | Bit, zero-test, and inverse rows without canonicalization | Exact little-endian chunk LC | No |
| `CanonicalAcceptanceSourceRows` | `challenge.sampler.chunk.accept` | Adds the production `(1-a) * inverse = 0` row | Same source rows | No - concrete trace bridge remains open |
| `currentAcceptanceSourceRows_exists_iff` | legacy source semantics | Existential witnesses accept iff `a = 1` exactly off all-ones | Sixteen source bit roots | No |
| `canonicalAcceptanceSourceRows_exists_iff` | production source semantics | The fourth row preserves existential acceptance | Sixteen source bit roots | No |
| `canonicalAcceptanceSourceRows_witness_cases` | witness invertibility | The inverse is zero on rejection and `d⁻¹` on acceptance | Canonical rows | No |
| `canonicalAcceptanceMaterializer_exact` | deterministic materialization | The canonical witness exists uniquely | Exact source difference | No |
| `canonicalAcceptanceSourceRows_iff_packedRows` | packed acceptance bridge | Canonical source and packed rows are equisatisfiable for the same accept bit | Sixteen source bit roots | No - exact Rust packed trace bridge remains separate |
| `currentAcceptanceSourceRows_not_injective` | necessity witness | The legacy all-ones source admits inverse zero and inverse one | Concrete Boolean chunk | No |
| `canonicalInverseRow_is_necessary` | necessity witness | The fourth row removes that second witness | Concrete Boolean chunk | No |
-/

namespace SuperNeo.FPrimeRecursiveVerifier.PiRlcChallenge

/-- The exact little-endian field value used by the Rust chunk LC. -/
def acceptanceChunkValue (bits : Fin 16 → F) : F :=
  fieldBitsValue 16 bits

/-- `d = chunk - 65535`, matching the Rust acceptance source. -/
def acceptanceDifference (bits : Fin 16 → F) : F :=
  acceptanceChunkValue bits - F.ofNat rejectionBucket

/-- The unique semantically rejected sixteen-bit source chunk. -/
def AllChunkBitsOne (bits : Fin 16 → F) : Prop :=
  ∀ index, bits index = 1

/--
The legacy first three source rows: accept bitness, the zero branch, and the
inverse branch. At `d = 0`, the inverse is deliberately unconstrained.
-/
def CurrentAcceptanceSourceRows
    (bits : Fin 16 → F) (accept inverse : F) : Prop :=
  FieldBitRoot accept ∧
    (1 - accept) * acceptanceDifference bits = 0 ∧
    acceptanceDifference bits * inverse = accept

/-- The production fourth source row fixes the rejected inverse to zero. -/
def CanonicalInverseRow (accept inverse : F) : Prop :=
  (1 - accept) * inverse = 0

/-- The four production source rows, including inverse canonicalization. -/
def CanonicalAcceptanceSourceRows
    (bits : Fin 16 → F) (accept inverse : F) : Prop :=
  CurrentAcceptanceSourceRows bits accept inverse ∧
    CanonicalInverseRow accept inverse

/-- Acceptance semantics at the field boundary. -/
def AcceptanceSourceMeaning
    (bits : Fin 16 → F) (accept : F) : Prop :=
  FieldBitRoot accept ∧
    (accept = 1 ↔ ¬ AllChunkBitsOne bits)

private def acceptanceBitBool (value : F) : Bool :=
  decide (value = 1)

private theorem acceptanceFieldBitRoot_cases
    {value : F} (hRoot : FieldBitRoot value) :
    value = 0 ∨ value = 1 := by
  rcases mul_eq_zero.mp hRoot with hZero | hOne
  · exact Or.inl hZero
  · exact Or.inr (sub_eq_zero.mp hOne)

private theorem acceptanceBit_eq_encoded
    {value : F} (hRoot : FieldBitRoot value) :
    value = if acceptanceBitBool value then 1 else 0 := by
  rcases acceptanceFieldBitRoot_cases hRoot with hZero | hOne
  · simp [acceptanceBitBool, hZero]
  · simp [acceptanceBitBool, hOne]

private theorem acceptanceFieldOfNat_mul (left right : Nat) :
    F.ofNat (left * right) = F.ofNat left * F.ofNat right := by
  apply Fin.ext
  simp [F.ofNat, Nat.mul_mod]

private theorem acceptanceFieldBitsValue_encoded :
    ∀ {width : Nat} (bits : Fin width → Bool),
      fieldBitsValue width (fun index ↦ if bits index then 1 else 0) =
        F.ofNat (Nat.ofBits bits) := by
  intro width bits
  induction width with
  | zero => simp [fieldBitsValue]
  | succ width ih =>
      rw [fieldBitsValue]
      have hTail :
          fieldBitsValue width
              (fun index ↦ if bits (Fin.succ index) then 1 else 0) =
            F.ofNat (Nat.ofBits (bits ∘ Fin.succ)) := by
        simpa only [Function.comp_apply] using ih (bits ∘ Fin.succ)
      change
        2 * fieldBitsValue width
              (fun index ↦ if bits (Fin.succ index) then 1 else 0) +
            (if bits 0 then 1 else 0) =
          F.ofNat (Nat.ofBits bits)
      rw [hTail]
      cases hBit : bits 0 <;>
        simp [Nat.ofBits_succ, hBit, acceptanceFieldOfNat_mul]

private theorem acceptanceFieldBitsValue_decoded
    {bits : Fin 16 → F} (hBits : ChunkBitsAreBoolean bits) :
    acceptanceChunkValue bits =
      F.ofNat (Nat.ofBits (fun index ↦ acceptanceBitBool (bits index))) := by
  have hCoordinates :
      bits = fun index ↦ if acceptanceBitBool (bits index) then 1 else 0 := by
    funext index
    exact acceptanceBit_eq_encoded (hBits index)
  calc
    acceptanceChunkValue bits =
        acceptanceChunkValue
          (fun index ↦ if acceptanceBitBool (bits index) then 1 else 0) :=
      congrArg acceptanceChunkValue hCoordinates
    _ = F.ofNat
          (Nat.ofBits (fun index ↦ acceptanceBitBool (bits index))) :=
      acceptanceFieldBitsValue_encoded _

private theorem acceptanceFieldOfNat_eq_iff_of_lt
    {left right : Nat}
    (hLeft : left < Goldilocks.q)
    (hRight : right < Goldilocks.q) :
    F.ofNat left = F.ofNat right ↔ left = right := by
  constructor
  · intro hField
    have hVal := congrArg Fin.val hField
    simpa [F.ofNat_val_eq_of_canonical hLeft,
      F.ofNat_val_eq_of_canonical hRight] using hVal
  · exact fun h ↦ congrArg F.ofNat h

private theorem natBits_eq_rejectionBucket_iff_all_true
    (bits : Fin 16 → Bool) :
    Nat.ofBits bits = rejectionBucket ↔ ∀ index, bits index = true := by
  have hBucket : rejectionBucket = 2 ^ 16 - 1 := by
    norm_num [rejectionBucket]
  constructor
  · intro hValue index
    calc
      bits index = (Nat.ofBits bits).testBit index.val :=
        (Nat.testBit_ofBits_lt bits index.val index.isLt).symm
      _ = rejectionBucket.testBit index.val :=
        congrArg (fun value ↦ value.testBit index.val) hValue
      _ = (2 ^ 16 - 1).testBit index.val := by rw [hBucket]
      _ = true := by fin_cases index <;> decide
  · intro hAll
    have hBits : bits = fun _ ↦ true := by
      funext index
      exact hAll index
    subst bits
    decide

/--
For Boolean source coordinates, the exact Rust difference is zero only on the
all-ones chunk. This is the chunk-LC bridge used below; no product surrogate is
substituted for the source LC.
-/
theorem acceptanceDifference_zero_iff_allBitsOne
    (bits : Fin 16 → F)
    (hBits : ChunkBitsAreBoolean bits) :
    acceptanceDifference bits = 0 ↔ AllChunkBitsOne bits := by
  rw [acceptanceDifference, sub_eq_zero]
  rw [acceptanceFieldBitsValue_decoded hBits]
  have hDecodedLt :
      Nat.ofBits (fun index ↦ acceptanceBitBool (bits index)) <
        Goldilocks.q := by
    exact lt_trans (Nat.ofBits_lt_two_pow _) (by norm_num [Goldilocks.q])
  have hBucketLt : rejectionBucket < Goldilocks.q := by
    norm_num [rejectionBucket, Goldilocks.q]
  rw [acceptanceFieldOfNat_eq_iff_of_lt hDecodedLt hBucketLt]
  rw [natBits_eq_rejectionBucket_iff_all_true]
  constructor
  · intro hAll index
    simpa [acceptanceBitBool] using hAll index
  · intro hAll index
    simp [acceptanceBitBool, hAll index]

private theorem currentAcceptanceSourceRows_accepts_iff_difference_ne_zero
    {bits : Fin 16 → F} {accept inverse : F}
    (hRows : CurrentAcceptanceSourceRows bits accept inverse) :
    accept = 1 ↔ acceptanceDifference bits ≠ 0 := by
  rcases hRows with ⟨hAcceptBit, hZeroBranch, hInverseBranch⟩
  rcases acceptanceFieldBitRoot_cases hAcceptBit with hAccept | hAccept
  · rw [hAccept] at hZeroBranch hInverseBranch ⊢
    have hDifference : acceptanceDifference bits = 0 := by
      simpa using hZeroBranch
    simp [hDifference]
  · rw [hAccept] at hZeroBranch hInverseBranch ⊢
    constructor
    · intro _ hDifference
      simp [hDifference] at hInverseBranch
    · intro _
      rfl

/-- The current three rows imply the exact semantic accept bit. -/
theorem currentAcceptanceSourceRows_sound
    {bits : Fin 16 → F} {accept inverse : F}
    (hBits : ChunkBitsAreBoolean bits)
    (hRows : CurrentAcceptanceSourceRows bits accept inverse) :
    AcceptanceSourceMeaning bits accept := by
  refine ⟨hRows.1, ?_⟩
  calc
    accept = 1 ↔ acceptanceDifference bits ≠ 0 :=
      currentAcceptanceSourceRows_accepts_iff_difference_ne_zero hRows
    _ ↔ ¬ AllChunkBitsOne bits :=
      not_congr (acceptanceDifference_zero_iff_allBitsOne bits hBits)

/-- A deterministic pair of accept and inverse cells. -/
structure CanonicalAcceptanceWitness where
  accept : F
  inverse : F

/-- Canonical materializer: `(0, 0)` on rejection and `(1, d⁻¹)` otherwise. -/
noncomputable def canonicalAcceptanceMaterializer
    (bits : Fin 16 → F) : CanonicalAcceptanceWitness :=
  if acceptanceDifference bits = 0 then
    { accept := 0, inverse := 0 }
  else
    { accept := 1, inverse := (acceptanceDifference bits)⁻¹ }

/-- The deterministic materializer satisfies all four rows for every chunk LC. -/
theorem canonicalAcceptanceMaterializer_holds (bits : Fin 16 → F) :
    CanonicalAcceptanceSourceRows bits
      (canonicalAcceptanceMaterializer bits).accept
      (canonicalAcceptanceMaterializer bits).inverse := by
  by_cases hDifference : acceptanceDifference bits = 0
  · simp [canonicalAcceptanceMaterializer, hDifference,
      CanonicalAcceptanceSourceRows, CurrentAcceptanceSourceRows,
      CanonicalInverseRow, FieldBitRoot, fieldBitResidual]
  · simp [canonicalAcceptanceMaterializer, hDifference,
      CanonicalAcceptanceSourceRows, CurrentAcceptanceSourceRows,
      CanonicalInverseRow, FieldBitRoot, fieldBitResidual,
      mul_inv_cancel₀]

/--
Every canonical witness has the exact rejection or acceptance materialization.
This theorem does not need source bitness: it follows from the four rows alone.
-/
theorem canonicalAcceptanceSourceRows_witness_cases
    {bits : Fin 16 → F} {accept inverse : F}
    (hRows : CanonicalAcceptanceSourceRows bits accept inverse) :
    (acceptanceDifference bits = 0 ∧ accept = 0 ∧ inverse = 0) ∨
      (acceptanceDifference bits ≠ 0 ∧ accept = 1 ∧
        inverse = (acceptanceDifference bits)⁻¹) := by
  rcases hRows with ⟨⟨hAcceptBit, hZeroBranch, hInverseBranch⟩,
    hCanonical⟩
  by_cases hDifference : acceptanceDifference bits = 0
  · left
    have hAccept : accept = 0 := by
      simpa [hDifference] using hInverseBranch.symm
    have hInverse : inverse = 0 := by
      rw [hAccept] at hCanonical
      simpa [CanonicalInverseRow] using hCanonical
    exact ⟨hDifference, hAccept, hInverse⟩
  · right
    have hOneMinus : 1 - accept = 0 := by
      exact (mul_eq_zero.mp hZeroBranch).resolve_right hDifference
    have hAccept : accept = 1 := (sub_eq_zero.mp hOneMinus).symm
    have hInverse : inverse = (acceptanceDifference bits)⁻¹ := by
      rw [hAccept] at hInverseBranch
      calc
        inverse = 1 * inverse := by ring
        _ = ((acceptanceDifference bits)⁻¹ *
              acceptanceDifference bits) * inverse := by
            rw [inv_mul_cancel₀ hDifference]
        _ = (acceptanceDifference bits)⁻¹ *
              (acceptanceDifference bits * inverse) := by ring
        _ = (acceptanceDifference bits)⁻¹ := by
            rw [hInverseBranch]
            ring
    exact ⟨hDifference, hAccept, hInverse⟩

/-- Canonical source witnesses are pointwise unique. -/
theorem canonicalAcceptanceSourceRows_unique
    {bits : Fin 16 → F} {leftAccept leftInverse rightAccept rightInverse : F}
    (hLeft : CanonicalAcceptanceSourceRows bits leftAccept leftInverse)
    (hRight : CanonicalAcceptanceSourceRows bits rightAccept rightInverse) :
    leftAccept = rightAccept ∧ leftInverse = rightInverse := by
  rcases canonicalAcceptanceSourceRows_witness_cases hLeft with hLeft | hLeft <;>
    rcases canonicalAcceptanceSourceRows_witness_cases hRight with hRight | hRight
  · exact ⟨hLeft.2.1.trans hRight.2.1.symm,
      hLeft.2.2.trans hRight.2.2.symm⟩
  · exact False.elim (hRight.1 hLeft.1)
  · exact False.elim (hLeft.1 hRight.1)
  · exact ⟨hLeft.2.1.trans hRight.2.1.symm,
      hLeft.2.2.trans hRight.2.2.symm⟩

/-- The materializer is the unique witness satisfying the canonical rows. -/
theorem canonicalAcceptanceMaterializer_exact (bits : Fin 16 → F) :
    ∃! witness : CanonicalAcceptanceWitness,
      CanonicalAcceptanceSourceRows bits witness.accept witness.inverse := by
  refine ⟨canonicalAcceptanceMaterializer bits,
    canonicalAcceptanceMaterializer_holds bits, ?_⟩
  intro witness hWitness
  rcases canonicalAcceptanceSourceRows_unique hWitness
      (canonicalAcceptanceMaterializer_holds bits) with ⟨hAccept, hInverse⟩
  cases witness
  simp_all

private theorem acceptanceSourceMeaning_accept_unique
    {bits : Fin 16 → F} {left right : F}
    (hLeft : AcceptanceSourceMeaning bits left)
    (hRight : AcceptanceSourceMeaning bits right) :
    left = right := by
  by_cases hAll : AllChunkBitsOne bits
  · have hLeftNe : left ≠ 1 := fun hOne ↦ (hLeft.2.mp hOne) hAll
    have hRightNe : right ≠ 1 := fun hOne ↦ (hRight.2.mp hOne) hAll
    rcases acceptanceFieldBitRoot_cases hLeft.1 with hLeftZero | hLeftOne
    · rcases acceptanceFieldBitRoot_cases hRight.1 with hRightZero | hRightOne
      · exact hLeftZero.trans hRightZero.symm
      · exact False.elim (hRightNe hRightOne)
    · exact False.elim (hLeftNe hLeftOne)
  · exact (hLeft.2.mpr hAll).trans (hRight.2.mpr hAll).symm

/-- The semantic accept bit has a current-source witness. -/
theorem currentAcceptanceSourceRows_complete
    {bits : Fin 16 → F} {accept : F}
    (hBits : ChunkBitsAreBoolean bits)
    (hMeaning : AcceptanceSourceMeaning bits accept) :
    ∃ inverse, CurrentAcceptanceSourceRows bits accept inverse := by
  let witness := canonicalAcceptanceMaterializer bits
  have hCanonical := canonicalAcceptanceMaterializer_holds bits
  have hMaterializedMeaning : AcceptanceSourceMeaning bits witness.accept :=
    currentAcceptanceSourceRows_sound hBits hCanonical.1
  have hAccept : accept = witness.accept :=
    acceptanceSourceMeaning_accept_unique hMeaning hMaterializedMeaning
  refine ⟨witness.inverse, ?_⟩
  rw [hAccept]
  exact hCanonical.1

/-- The current three rows are existentially exact for semantic acceptance. -/
theorem currentAcceptanceSourceRows_exists_iff
    (bits : Fin 16 → F) (accept : F)
    (hBits : ChunkBitsAreBoolean bits) :
    (∃ inverse, CurrentAcceptanceSourceRows bits accept inverse) ↔
      AcceptanceSourceMeaning bits accept := by
  constructor
  · rintro ⟨inverse, hRows⟩
    exact currentAcceptanceSourceRows_sound hBits hRows
  · exact currentAcceptanceSourceRows_complete hBits

/-- The fourth row preserves the exact existential acceptance semantics. -/
theorem canonicalAcceptanceSourceRows_exists_iff
    (bits : Fin 16 → F) (accept : F)
    (hBits : ChunkBitsAreBoolean bits) :
    (∃ inverse, CanonicalAcceptanceSourceRows bits accept inverse) ↔
      AcceptanceSourceMeaning bits accept := by
  constructor
  · rintro ⟨inverse, hRows⟩
    exact currentAcceptanceSourceRows_sound hBits hRows.1
  · intro hMeaning
    let witness := canonicalAcceptanceMaterializer bits
    have hCanonical := canonicalAcceptanceMaterializer_holds bits
    have hMaterializedMeaning : AcceptanceSourceMeaning bits witness.accept :=
      currentAcceptanceSourceRows_sound hBits hCanonical.1
    have hAccept : accept = witness.accept :=
      acceptanceSourceMeaning_accept_unique hMeaning hMaterializedMeaning
    refine ⟨witness.inverse, ?_⟩
    rw [hAccept]
    exact hCanonical

/-- Current and canonical source rows are equisatisfiable, not pointwise equal. -/
theorem current_iff_canonicalAcceptanceSourceRows_exists
    (bits : Fin 16 → F) (accept : F)
    (hBits : ChunkBitsAreBoolean bits) :
    (∃ inverse, CurrentAcceptanceSourceRows bits accept inverse) ↔
      ∃ inverse, CanonicalAcceptanceSourceRows bits accept inverse := by
  rw [currentAcceptanceSourceRows_exists_iff bits accept hBits,
    canonicalAcceptanceSourceRows_exists_iff bits accept hBits]

private theorem acceptanceHalfProduct_cases
    (values : Fin 8 → F)
    (hValues : ∀ index, FieldBitRoot (values index)) :
    (∏ index, values index) = 0 ∨ (∏ index, values index) = 1 := by
  classical
  by_cases hZero : ∃ index, values index = 0
  · rcases hZero with ⟨index, hIndex⟩
    exact Or.inl (Finset.prod_eq_zero (Finset.mem_univ index) hIndex)
  · have hOne : ∀ index, values index = 1 := by
      intro index
      rcases acceptanceFieldBitRoot_cases (hValues index) with hIndex | hIndex
      · exact False.elim (hZero ⟨index, hIndex⟩)
      · exact hIndex
    exact Or.inr (by simp [hOne])

private theorem acceptanceLowHalfProduct_cases
    {bits : Fin 16 → F} (hBits : ChunkBitsAreBoolean bits) :
    lowHalfProduct bits = 0 ∨ lowHalfProduct bits = 1 := by
  apply acceptanceHalfProduct_cases
  intro index
  exact hBits (lowHalfIndex index)

private theorem acceptanceHighHalfProduct_cases
    {bits : Fin 16 → F} (hBits : ChunkBitsAreBoolean bits) :
    highHalfProduct bits = 0 ∨ highHalfProduct bits = 1 := by
  apply acceptanceHalfProduct_cases
  intro index
  exact hBits (highHalfIndex index)

private theorem acceptanceFieldBitRoot_of_cases
    {value : F} (hCases : value = 0 ∨ value = 1) :
    FieldBitRoot value := by
  rcases hCases with rfl | rfl <;>
    simp [FieldBitRoot, fieldBitResidual]

private theorem acceptanceLowHalfProduct_is_bit
    {bits : Fin 16 → F} (hBits : ChunkBitsAreBoolean bits) :
    FieldBitRoot (lowHalfProduct bits) :=
  acceptanceFieldBitRoot_of_cases (acceptanceLowHalfProduct_cases hBits)

private theorem acceptanceHighHalfProduct_is_bit
    {bits : Fin 16 → F} (hBits : ChunkBitsAreBoolean bits) :
    FieldBitRoot (highHalfProduct bits) :=
  acceptanceFieldBitRoot_of_cases (acceptanceHighHalfProduct_cases hBits)

private theorem lowHalfProduct_eq_one_iff
    {bits : Fin 16 → F} (hBits : ChunkBitsAreBoolean bits) :
    lowHalfProduct bits = 1 ↔
      ∀ index : Fin 8, bits (lowHalfIndex index) = 1 := by
  constructor
  · intro hProduct index
    rcases acceptanceFieldBitRoot_cases (hBits (lowHalfIndex index)) with hZero | hOne
    · have hProductZero : lowHalfProduct bits = 0 := by
        exact Finset.prod_eq_zero (Finset.mem_univ index) hZero
      exact False.elim (zero_ne_one (hProductZero.symm.trans hProduct))
    · exact hOne
  · intro hAll
    simp [lowHalfProduct, hAll]

private theorem highHalfProduct_eq_one_iff
    {bits : Fin 16 → F} (hBits : ChunkBitsAreBoolean bits) :
    highHalfProduct bits = 1 ↔
      ∀ index : Fin 8, bits (highHalfIndex index) = 1 := by
  constructor
  · intro hProduct index
    rcases acceptanceFieldBitRoot_cases (hBits (highHalfIndex index)) with hZero | hOne
    · have hProductZero : highHalfProduct bits = 0 := by
        exact Finset.prod_eq_zero (Finset.mem_univ index) hZero
      exact False.elim (zero_ne_one (hProductZero.symm.trans hProduct))
    · exact hOne
  · intro hAll
    simp [highHalfProduct, hAll]

private theorem allChunkBitsOne_iff_halves
    (bits : Fin 16 → F) :
    AllChunkBitsOne bits ↔
      (∀ index : Fin 8, bits (lowHalfIndex index) = 1) ∧
      (∀ index : Fin 8, bits (highHalfIndex index) = 1) := by
  constructor
  · intro hAll
    exact ⟨fun index ↦ hAll (lowHalfIndex index),
      fun index ↦ hAll (highHalfIndex index)⟩
  · rintro ⟨hLow, hHigh⟩ index
    by_cases hIndex : index.val < 8
    · let low : Fin 8 := ⟨index.val, hIndex⟩
      have hEqual : lowHalfIndex low = index := by
        apply Fin.ext
        rfl
      rw [← hEqual]
      exact hLow low
    · let high : Fin 8 := ⟨index.val - 8, by omega⟩
      have hEqual : highHalfIndex high = index := by
        apply Fin.ext
        simp [highHalfIndex, high]
        omega
      rw [← hEqual]
      exact hHigh high

private theorem halfProducts_eq_one_iff_allBitsOne
    {bits : Fin 16 → F} (hBits : ChunkBitsAreBoolean bits) :
    lowHalfProduct bits * highHalfProduct bits = 1 ↔
      AllChunkBitsOne bits := by
  rw [allChunkBitsOne_iff_halves]
  rw [← lowHalfProduct_eq_one_iff hBits,
    ← highHalfProduct_eq_one_iff hBits]
  rcases acceptanceLowHalfProduct_cases hBits with hLow | hLow <;>
    rcases acceptanceHighHalfProduct_cases hBits with hHigh | hHigh <;>
    simp [hLow, hHigh]

/-- The readable packed meaning has exactly the same semantic accept bit. -/
theorem packedAcceptanceMeaning_iff_sourceMeaning
    (bits : Fin 16 → F) (accept : F)
    (hBits : ChunkBitsAreBoolean bits) :
    PackedAcceptanceMeaning bits (lowHalfProduct bits)
        (highHalfProduct bits) accept ↔
      AcceptanceSourceMeaning bits accept := by
  have hProductCases :
      lowHalfProduct bits * highHalfProduct bits = 0 ∨
        lowHalfProduct bits * highHalfProduct bits = 1 := by
    rcases acceptanceLowHalfProduct_cases hBits with hLow | hLow <;>
      rcases acceptanceHighHalfProduct_cases hBits with hHigh | hHigh <;>
      simp [hLow, hHigh]
  constructor
  · rintro ⟨_hLow, _hHigh, hLowBit, hHighBit, hAccept⟩
    refine ⟨?_, ?_⟩
    · rcases hProductCases with hProduct | hProduct <;>
        simp [hAccept, hProduct, FieldBitRoot, fieldBitResidual]
    · constructor
      · intro hAcceptOne hAll
        have hProduct := (halfProducts_eq_one_iff_allBitsOne hBits).mpr hAll
        rw [hAccept, hProduct] at hAcceptOne
        exact zero_ne_one hAcceptOne
      · intro hNotAll
        have hProduct :
            lowHalfProduct bits * highHalfProduct bits = 0 := by
          rcases hProductCases with hZero | hOne
          · exact hZero
          · exact False.elim
              (hNotAll ((halfProducts_eq_one_iff_allBitsOne hBits).mp hOne))
        simp [hAccept, hProduct]
  · rintro ⟨hAcceptBit, hAcceptSemantic⟩
    refine ⟨rfl, rfl, acceptanceLowHalfProduct_is_bit hBits,
      acceptanceHighHalfProduct_is_bit hBits, ?_⟩
    by_cases hAll : AllChunkBitsOne bits
    · have hProduct := (halfProducts_eq_one_iff_allBitsOne hBits).mpr hAll
      have hAcceptNe : accept ≠ 1 := by
        intro hOne
        exact (hAcceptSemantic.mp hOne) hAll
      rcases acceptanceFieldBitRoot_cases hAcceptBit with hZero | hOne
      · simp [hZero, hProduct]
      · exact False.elim (hAcceptNe hOne)
    · have hAcceptOne : accept = 1 := hAcceptSemantic.mpr hAll
      have hProductNe :
          lowHalfProduct bits * highHalfProduct bits ≠ 1 := by
        intro hOne
        exact hAll ((halfProducts_eq_one_iff_allBitsOne hBits).mp hOne)
      rcases hProductCases with hZero | hOne
      · simp [hAcceptOne, hZero]
      · exact False.elim (hProductNe hOne)

/-- Canonical source rows and the readable packed meaning are equisatisfiable. -/
theorem canonicalAcceptanceSourceRows_iff_packedMeaning
    (bits : Fin 16 → F) (accept : F)
    (hBits : ChunkBitsAreBoolean bits) :
    (∃ inverse, CanonicalAcceptanceSourceRows bits accept inverse) ↔
      PackedAcceptanceMeaning bits (lowHalfProduct bits)
        (highHalfProduct bits) accept := by
  rw [canonicalAcceptanceSourceRows_exists_iff bits accept hBits,
    packedAcceptanceMeaning_iff_sourceMeaning bits accept hBits]

/-- Canonical source rows and the proposed two packed rows accept identically. -/
theorem canonicalAcceptanceSourceRows_iff_packedRows
    (bits : Fin 16 → F) (accept : F)
    (hBits : ChunkBitsAreBoolean bits) :
    (∃ inverse, CanonicalAcceptanceSourceRows bits accept inverse) ↔
      PackedAcceptanceRows bits (lowHalfProduct bits)
        (highHalfProduct bits) accept := by
  rw [canonicalAcceptanceSourceRows_iff_packedMeaning bits accept hBits]
  exact (packedAcceptanceRows_iff bits _ _ _ hBits).symm

/-- The packed auxiliary cells may also be existentially hidden. -/
theorem canonicalAcceptanceSourceRows_iff_exists_packedRows
    (bits : Fin 16 → F) (accept : F)
    (hBits : ChunkBitsAreBoolean bits) :
    (∃ inverse, CanonicalAcceptanceSourceRows bits accept inverse) ↔
      ∃ u v, PackedAcceptanceRows bits u v accept := by
  constructor
  · intro hCanonical
    exact ⟨lowHalfProduct bits, highHalfProduct bits,
      (canonicalAcceptanceSourceRows_iff_packedRows bits accept hBits).mp
        hCanonical⟩
  · rintro ⟨u, v, hPacked⟩
    have hMeaning := (packedAcceptanceRows_iff bits u v accept hBits).mp hPacked
    have hCanonicalMeaning :
        PackedAcceptanceMeaning bits (lowHalfProduct bits)
          (highHalfProduct bits) accept := by
      simpa [hMeaning.1, hMeaning.2.1] using hMeaning
    exact (canonicalAcceptanceSourceRows_iff_packedMeaning
      bits accept hBits).mpr hCanonicalMeaning

/--
Pointwise witness preservation is false: the current rejected source admits at
least two different inverse cells.
-/
theorem currentAcceptanceSourceRows_not_injective :
    ∃ bits : Fin 16 → F,
      ChunkBitsAreBoolean bits ∧
        CurrentAcceptanceSourceRows bits 0 0 ∧
        CurrentAcceptanceSourceRows bits 0 1 ∧
        (0 : F) ≠ 1 := by
  let bits : Fin 16 → F := fun _ ↦ 1
  have hBits : ChunkBitsAreBoolean bits := by
    intro index
    simp [bits, FieldBitRoot, fieldBitResidual]
  have hDifference : acceptanceDifference bits = 0 :=
    (acceptanceDifference_zero_iff_allBitsOne bits hBits).mpr (by
      intro index
      simp [bits])
  refine ⟨bits, hBits, ?_, ?_, zero_ne_one⟩ <;>
    simp [CurrentAcceptanceSourceRows, FieldBitRoot, fieldBitResidual,
      hDifference]

/-- The fourth row removes the explicit second rejected witness above. -/
theorem canonicalInverseRow_is_necessary :
    ∃ bits : Fin 16 → F,
      ChunkBitsAreBoolean bits ∧
        CurrentAcceptanceSourceRows bits 0 1 ∧
        ¬ CanonicalAcceptanceSourceRows bits 0 1 ∧
        CanonicalAcceptanceSourceRows bits 0 0 := by
  let bits : Fin 16 → F := fun _ ↦ 1
  have hBits : ChunkBitsAreBoolean bits := by
    intro index
    simp [bits, FieldBitRoot, fieldBitResidual]
  have hDifference : acceptanceDifference bits = 0 :=
    (acceptanceDifference_zero_iff_allBitsOne bits hBits).mpr (by
      intro index
      simp [bits])
  refine ⟨bits, hBits, ?_, ?_, ?_⟩
  · simp [CurrentAcceptanceSourceRows, FieldBitRoot, fieldBitResidual,
      hDifference]
  · simp [CanonicalAcceptanceSourceRows, CanonicalInverseRow,
      CurrentAcceptanceSourceRows, FieldBitRoot, fieldBitResidual,
      hDifference]
  · simp [CanonicalAcceptanceSourceRows, CanonicalInverseRow,
      CurrentAcceptanceSourceRows, FieldBitRoot, fieldBitResidual,
      hDifference]

/-! ## Degree accounting -/

/-- The added row is one product of two linear expressions. -/
def canonicalInverseRowDegree : Nat := 2

/-- The fourth row stays strictly below the fixed degree-eight CCS ceiling. -/
theorem canonicalInverseRow_degree_budget :
    canonicalInverseRowDegree = 2 ∧ canonicalInverseRowDegree ≤ 8 := by
  norm_num [canonicalInverseRowDegree]

end SuperNeo.FPrimeRecursiveVerifier.PiRlcChallenge
