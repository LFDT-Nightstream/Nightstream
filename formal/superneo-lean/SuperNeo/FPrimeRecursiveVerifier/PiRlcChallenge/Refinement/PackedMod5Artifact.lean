import SuperNeo.FPrimeRecursiveVerifier.PiRlcChallenge.Refinement.Generated.PackedMod5ArtifactData

/-!
Owns: direct evaluation of the generated active packed-Mod-5 artifact and its
equivalence to the proved eight-row semantic relation.

Does not own: the fixed-selector materializer, inactive selector rows, or the
outer linear-substitution proof that may replace a checked chunk input before
this leaf is lowered.

Emits constraints: no.

Authority boundary: Lean evaluates the generated row and polynomial lists.
Soundness for arbitrary encoded `L,R` pairs uses the exact generated decoder,
chunk-bit image, and quotient-bit coordinate map. Only the reverse direction
requires the explicit canonical residue image.

| Predicate/theorem | Rust stage | Guarantee | Assumptions | Permits row removal? |
|---|---|---|---|---|
| `generated_shape_exact` | packed Mod-5 leaf | Exact 16/19/20, 15, 6, 8, arity-48, 12-term geometry | Generated production data | Accounting only |
| `generatedEmittedAccepts_iff_packed` | eight active rows | Generated row schedule and polynomial evaluate exactly as `PackedReducedMod5FieldRows` | Active selector is one | Model/artifact equivalence only |
| `generatedEmittedAccepts_implies_source_of_structuralImage` | active-to-source leaf | Active packed rows derive all 20 exact source rows, including the six removed scaffolding rows | Generated decoder, exact chunk bits, quotient-bit coordinate map | Active leaf only |
| `generatedSourceAccepts_implies_emitted_of_canonicalImage` | source-to-active leaf | Exact source rows imply the generated active rows | Structural image plus canonical residue table | Active leaf only |
-/

namespace SuperNeo.FPrimeRecursiveVerifier.PiRlcChallenge.PackedMod5Artifact

open PackedMod5ArtifactData

/-- Direct acceptance of all twenty generated source rows. -/
def GeneratedSourceAccepts (source : SourceAssignment) : Prop :=
  ∀ row ∈ sourceRows, row.Holds source

/-- Direct acceptance of all six generated projected definitions. -/
def GeneratedDecoderAccepts
    (source : SourceAssignment) (coordinates : CoordinateAssignment) : Prop :=
  ∀ definition ∈ decoderDefinitions, definition.Holds source coordinates

/-- Coordinates supplied to the generated decoder by the packed witness. -/
def witnessCoordinates
    (witness : ReducedMod5FieldWitness) : CoordinateAssignment
  | .quotientLow index =>
      if h : index < 13 then witness.quotientLow ⟨index, h⟩ else 0
  | .residueLeft => witness.residueLeft
  | .residueRight => witness.residueRight

/-- Direct acceptance of all eight generated active rows by the generated
12-term sparse-polynomial specialization. -/
noncomputable def GeneratedEmittedAccepts
    (chunk : Chunk) (witness : ReducedMod5FieldWitness) : Prop :=
  ∀ row ∈ activeRows,
    evalPolynomial polynomialTerms (activeRowPoint chunk witness row) = 0

/-- The six non-bitness source rows remain exact generated equations. -/
def GeneratedSourceScaffolding (source : SourceAssignment) : Prop :=
  ∀ row ∈ sourceRows.take 4 ++ sourceRows.drop 18, row.Holds source

/-- The fourteen exact generated source bit rows. -/
def GeneratedSourceBitRows (source : SourceAssignment) : Prop :=
  ∀ row ∈ (sourceRows.drop 4).take 14, row.Holds source

/-- The exact external leaf inputs: constant one and the canonical sixteen
little-endian bits of the semantic chunk. -/
def SourceChunkImage (source : SourceAssignment) (chunk : Chunk) : Prop :=
  source .one = 1 ∧
    ∀ index : Fin 16,
      source (.chunkBit index.val) =
        if chunk.val.testBit index.val then 1 else 0

/-- Exact source/coordinate map at the checked 16-bit leaf boundary. -/
def SourceBitCoordinateImage
    (source : SourceAssignment) (chunk : Chunk)
    (witness : ReducedMod5FieldWitness) : Prop :=
  (∀ index : Fin 13,
      source (.quotientBit index.val) = witness.quotientLow index) ∧
    source (.quotientBit 13) = derivedQuotientHighField chunk witness

/-- Only the five canonical centered images may connect a source index to
the two packed residue cells. -/
def CanonicalResidueImage
    (source : SourceAssignment) (witness : ReducedMod5FieldWitness) : Prop :=
  ∃ entry ∈ canonicalResidues,
    source .index = coefficient entry.index ∧
      witness.residueLeft = coefficient entry.left ∧
      witness.residueRight = coefficient entry.right

/-- Exact structural precondition shared by both leaf-local directions. It
contains no source-row acceptance premise. -/
def StructuralSourceImage
    (source : SourceAssignment) (chunk : Chunk)
    (witness : ReducedMod5FieldWitness) : Prop :=
  GeneratedDecoderAccepts source (witnessCoordinates witness) ∧
    SourceChunkImage source chunk ∧
    SourceBitCoordinateImage source chunk witness

/-- Stronger image required only for the source-to-packed direction. -/
def CanonicalSourceImage
    (source : SourceAssignment) (chunk : Chunk)
    (witness : ReducedMod5FieldWitness) : Prop :=
  StructuralSourceImage source chunk witness ∧
    CanonicalResidueImage source witness

theorem generated_shape_exact :
    schemaVersion = 1 ∧
      sourceInputOrder.length = 16 ∧
      sourceAllocatedOrder.length = 19 ∧
      sourceRows.length = 20 ∧
      coordinateOrder.length = 15 ∧
      decoderDefinitions.length = 6 ∧
      canonicalResidues.length = 5 ∧
      gateArity = 48 ∧
      matrixBindings.map (fun binding => binding.index) = [0, 44, 45, 46, 47] ∧
      activeRows.length = 8 ∧
      polynomialTerms.length = 12 := by
  decide

theorem generated_source_role_order_exact :
    sourceInputOrder = (List.range 16).map SourceRole.chunkBit ∧
      sourceAllocatedOrder =
        [.index, .quotient, .indexProduct 0, .indexProduct 1,
          .indexProduct 2] ++
          (List.range 14).map SourceRole.quotientBit := by
  decide

theorem generated_coordinate_order_exact :
    coordinateOrder =
      (List.range 13).map CoordinateRole.quotientLow ++
        [.residueLeft, .residueRight] := by
  decide

private theorem generated_decoder_index
    {source : SourceAssignment} {coordinates : CoordinateAssignment}
    (hDecoder : GeneratedDecoderAccepts source coordinates) :
    source .index =
      2 * source .one +
        (coordinates .residueLeft + coordinates .residueRight) := by
  have hDefinition := hDecoder (decoderDefinitions[0]) (by native_decide)
  simpa [decoderDefinitions, DecoderDefinition.Holds,
    DecoderAtom.value, evalLinearCombination, coefficient] using hDefinition

private theorem generated_decoder_products
    {source : SourceAssignment} {coordinates : CoordinateAssignment}
    (hDecoder : GeneratedDecoderAccepts source coordinates) :
    source (.indexProduct 0) =
        source .index * (source .index + -source .one) ∧
      source (.indexProduct 1) =
        source (.indexProduct 0) * (source .index + -(2 * source .one)) ∧
      source (.indexProduct 2) =
        source (.indexProduct 1) * (source .index + -(3 * source .one)) := by
  have h0 := hDecoder (decoderDefinitions[3]) (by native_decide)
  have h1 := hDecoder (decoderDefinitions[4]) (by native_decide)
  have h2 := hDecoder (decoderDefinitions[5]) (by native_decide)
  constructor
  · simpa [decoderDefinitions, DecoderDefinition.Holds,
      DecoderAtom.value, evalLinearCombination, coefficient] using h0
  · constructor
    · simpa [decoderDefinitions, DecoderDefinition.Holds,
        DecoderAtom.value, evalLinearCombination, coefficient] using h1
    · simpa [decoderDefinitions, DecoderDefinition.Holds,
        DecoderAtom.value, evalLinearCombination, coefficient] using h2

private def sourceChunkValue (source : SourceAssignment) : F :=
  fieldBitsValue 16 (fun index => source (.chunkBit index.val))

private def sourceQuotientLowValue (source : SourceAssignment) : F :=
  fieldBitsValue 13 (fun index => source (.quotientBit index.val))

private def coordinateQuotientLowValue
    (coordinates : CoordinateAssignment) : F :=
  fieldBitsValue 13 (fun index => coordinates (.quotientLow index.val))

private def SourceScaffoldingEquations (source : SourceAssignment) : Prop :=
  source (.indexProduct 0) = source .index * (source .index - source .one) ∧
    source (.indexProduct 1) =
      source (.indexProduct 0) * (source .index - 2 * source .one) ∧
    source (.indexProduct 2) =
      source (.indexProduct 1) * (source .index - 3 * source .one) ∧
    source (.indexProduct 2) * (source .index - 4 * source .one) = 0 ∧
    source .quotient =
      sourceQuotientLowValue source + 8192 * source (.quotientBit 13) ∧
    sourceChunkValue source = source .index + 5 * source .quotient

set_option maxHeartbeats 500000 in
private theorem generated_scaffolding_iff_equations
    {source : SourceAssignment} (hOne : source .one = 1) :
    GeneratedSourceScaffolding source ↔ SourceScaffoldingEquations source := by
  simp [GeneratedSourceScaffolding, SourceScaffoldingEquations, sourceRows,
    SourceRow.Holds, evalLinearCombination, coefficient,
    sourceQuotientLowValue, sourceChunkValue, fieldBitsValue, hOne]
  ring_nf
  constructor
  · rintro ⟨h0, h1, h2, h3, hQuotient, hChunk⟩
    exact ⟨h0.symm, h1.symm, h2.symm, h3,
      by linear_combination hQuotient,
      by linear_combination hChunk⟩
  · rintro ⟨h0, h1, h2, h3, hQuotient, hChunk⟩
    exact ⟨h0.symm, h1.symm, h2.symm, h3,
      by linear_combination hQuotient,
      by linear_combination hChunk⟩

private theorem fieldOfNat_mul (left right : Nat) :
    F.ofNat (left * right) = F.ofNat left * F.ofNat right := by
  apply Fin.ext
  simp [F.ofNat, Nat.mul_mod]

private theorem fieldBitsValue_ofBits :
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

private theorem sourceChunkValue_eq
    {source : SourceAssignment} {chunk : Chunk}
    (hImage : SourceChunkImage source chunk) :
    sourceChunkValue source = F.ofNat chunk.val := by
  rcases hImage with ⟨_, hBits⟩
  rw [sourceChunkValue]
  have hFunctions :
      (fun index : Fin 16 => source (.chunkBit index.val)) =
        (fun index => if chunk.val.testBit index.val then 1 else 0) := by
    funext index
    exact hBits index
  rw [hFunctions, fieldBitsValue_ofBits, Nat.ofBits_testBit]
  rw [Nat.mod_eq_of_lt]
  simpa only [chunkModulus] using chunk.isLt

private theorem sourceQuotientLowValue_eq
    {source : SourceAssignment} {witness : ReducedMod5FieldWitness}
    (hImage : ∀ index : Fin 13,
      source (.quotientBit index.val) = witness.quotientLow index) :
    sourceQuotientLowValue source = witness.quotientLowValue := by
  rw [sourceQuotientLowValue, ReducedMod5FieldWitness.quotientLowValue]
  congr 1
  funext index
  exact hImage index

private theorem coordinateQuotientLowValue_eq
    (witness : ReducedMod5FieldWitness) :
    coordinateQuotientLowValue (witnessCoordinates witness) =
      witness.quotientLowValue := by
  rw [coordinateQuotientLowValue,
    ReducedMod5FieldWitness.quotientLowValue]
  congr 1
  funext index
  simp [witnessCoordinates, index.isLt]

set_option maxHeartbeats 1000000 in
private theorem generated_decoder_high_reconstruction
    {source : SourceAssignment} {coordinates : CoordinateAssignment}
    (hDecoder : GeneratedDecoderAccepts source coordinates) :
    (40960 : F) * source (.quotientBit 13) =
      sourceChunkValue source -
        (5 * coordinateQuotientLowValue coordinates +
          (2 * source .one +
            coordinates .residueLeft + coordinates .residueRight)) := by
  have hDefinition := hDecoder (decoderDefinitions[1]) (by native_decide)
  simp only [decoderDefinitions, DecoderDefinition.Holds,
    DecoderAtom.value, evalLinearCombination] at hDefinition
  simp only [sourceChunkValue, coordinateQuotientLowValue, fieldBitsValue]
  norm_num [coefficient] at hDefinition ⊢
  linear_combination (norm := ring_nf) (40960 : F) * hDefinition
  apply toZMod_injective
  simp

set_option maxHeartbeats 1000000 in
private theorem generated_decoder_quotient_reconstruction
    {source : SourceAssignment} {coordinates : CoordinateAssignment}
    (hDecoder : GeneratedDecoderAccepts source coordinates) :
    (5 : F) * source .quotient =
      sourceChunkValue source -
        (2 * source .one +
          coordinates .residueLeft + coordinates .residueRight) := by
  have hDefinition := hDecoder (decoderDefinitions[2]) (by native_decide)
  simp only [decoderDefinitions, DecoderDefinition.Holds,
    DecoderAtom.value, evalLinearCombination] at hDefinition
  simp only [sourceChunkValue, fieldBitsValue]
  norm_num [coefficient] at hDefinition ⊢
  linear_combination (norm := ring_nf) (5 : F) * hDefinition
  apply toZMod_injective
  simp

private def bitPoint (left right : F) : MatrixPoint
  | .selector => 1
  | .bitLeft => left
  | .bitRight => right
  | .residueLeft | .residueRight => 0

private def residuePoint (left right : F) : MatrixPoint
  | .selector => 1
  | .bitLeft | .bitRight => 0
  | .residueLeft => left
  | .residueRight => right

private theorem kExt_w_eq_seven : KExt.w = (7 : F) := by
  native_decide

/-- The generated six-term bit specialization is exactly the norm of the two
ordinary bit residuals. -/
theorem generated_bit_polynomial (left right : F) :
    evalPolynomial polynomialTerms (bitPoint left right) =
      fieldBitResidual left * fieldBitResidual left -
        KExt.w * (fieldBitResidual right * fieldBitResidual right) := by
  simp [evalPolynomial, evalPolynomialTerm, polynomialTerms, bitPoint,
    coefficient, fieldBitResidual, kExt_w_eq_seven]
  ring

/-- The generated six-term residue specialization is exactly the norm of the
left cubic and residue-pair residual. -/
theorem generated_residue_polynomial (left right : F) :
    evalPolynomial polynomialTerms (residuePoint left right) =
      fieldCenteredResidual left * fieldCenteredResidual left -
        KExt.w * ((right * (left - right)) * (right * (left - right))) := by
  simp [evalPolynomial, evalPolynomialTerm, polynomialTerms, residuePoint,
    coefficient, fieldCenteredResidual, kExt_w_eq_seven]
  ring

private theorem generated_active_bit
    (chunk : Chunk) (witness : ReducedMod5FieldWitness)
    (left right : BitOperand) :
    evalPolynomial polynomialTerms
        (activeRowPoint chunk witness (.bitPair left right)) = 0 ↔
      QuadraticZeroPair
        (fieldBitResidual (bitOperandValue chunk witness left))
        (fieldBitResidual (bitOperandValue chunk witness right)) := by
  change evalPolynomial polynomialTerms
      (bitPoint (bitOperandValue chunk witness left)
        (bitOperandValue chunk witness right)) = 0 ↔ _
  rw [generated_bit_polynomial]
  rfl

private theorem generated_active_residue
    (chunk : Chunk) (witness : ReducedMod5FieldWitness) :
    evalPolynomial polynomialTerms
        (activeRowPoint chunk witness .residuePair) = 0 ↔
      QuadraticZeroPair
        (fieldCenteredResidual witness.residueLeft)
        witness.residuePairResidual := by
  change evalPolynomial polynomialTerms
      (residuePoint witness.residueLeft witness.residueRight) = 0 ↔ _
  rw [generated_residue_polynomial]
  rfl

/-- The generated row kinds and generated polynomial are directly equivalent
to the proved eight-row semantic predicate. -/
theorem generatedEmittedAccepts_iff_packed
    (chunk : Chunk) (witness : ReducedMod5FieldWitness) :
    GeneratedEmittedAccepts chunk witness ↔
      PackedReducedMod5FieldRows chunk witness := by
  constructor
  · intro hRows
    have h01 := (generated_active_bit chunk witness
      (.quotientLow 0) (.quotientLow 1)).mp (hRows _ (by decide))
    have h23 := (generated_active_bit chunk witness
      (.quotientLow 2) (.quotientLow 3)).mp (hRows _ (by decide))
    have h45 := (generated_active_bit chunk witness
      (.quotientLow 4) (.quotientLow 5)).mp (hRows _ (by decide))
    have h67 := (generated_active_bit chunk witness
      (.quotientLow 6) (.quotientLow 7)).mp (hRows _ (by decide))
    have h89 := (generated_active_bit chunk witness
      (.quotientLow 8) (.quotientLow 9)).mp (hRows _ (by decide))
    have h1011 := (generated_active_bit chunk witness
      (.quotientLow 10) (.quotientLow 11)).mp (hRows _ (by decide))
    have h12High := (generated_active_bit chunk witness
      (.quotientLow 12) .quotientHigh).mp (hRows _ (by decide))
    have hResidue := (generated_active_residue chunk witness).mp
      (hRows _ (by decide))
    simpa [PackedReducedMod5FieldRows, bitOperandValue] using
      And.intro h01 ⟨h23, h45, h67, h89, h1011, h12High, hResidue⟩
  · rintro ⟨h01, h23, h45, h67, h89, h1011, h12High, hResidue⟩ row hRow
    simp only [activeRows, List.mem_cons, List.not_mem_nil, or_false] at hRow
    rcases hRow with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl
    · exact (generated_active_bit chunk witness
        (.quotientLow 0) (.quotientLow 1)).mpr h01
    · exact (generated_active_bit chunk witness
        (.quotientLow 2) (.quotientLow 3)).mpr h23
    · exact (generated_active_bit chunk witness
        (.quotientLow 4) (.quotientLow 5)).mpr h45
    · exact (generated_active_bit chunk witness
        (.quotientLow 6) (.quotientLow 7)).mpr h67
    · exact (generated_active_bit chunk witness
        (.quotientLow 8) (.quotientLow 9)).mpr h89
    · exact (generated_active_bit chunk witness
        (.quotientLow 10) (.quotientLow 11)).mpr h1011
    · exact (generated_active_bit chunk witness
        (.quotientLow 12) .quotientHigh).mpr h12High
    · exact (generated_active_residue chunk witness).mpr hResidue

private theorem generated_source_partition (source : SourceAssignment) :
    GeneratedSourceAccepts source ↔
      GeneratedSourceScaffolding source ∧ GeneratedSourceBitRows source := by
  have hPartition :
      sourceRows = sourceRows.take 4 ++
        (sourceRows.drop 4).take 14 ++ sourceRows.drop 18 := by
    native_decide
  constructor
  · intro hRows
    constructor
    · intro row hRow
      apply hRows row
      rw [hPartition]
      simp only [List.mem_append]
      rcases List.mem_append.mp hRow with hFirst | hLast
      · exact Or.inl (Or.inl hFirst)
      · exact Or.inr hLast
    · intro row hRow
      apply hRows row
      rw [hPartition]
      exact List.mem_append.mpr
        (Or.inl (List.mem_append.mpr (Or.inr hRow)))
  · rintro ⟨hScaffolding, hBits⟩ row hRow
    rw [hPartition] at hRow
    rcases List.mem_append.mp hRow with hRest | hLast
    · rcases List.mem_append.mp hRest with hFirst | hMiddle
      · exact hScaffolding row (List.mem_append.mpr (Or.inl hFirst))
      · exact hBits row hMiddle
    · exact hScaffolding row (List.mem_append.mpr (Or.inr hLast))

private theorem canonicalResidueImage_rows
    {source : SourceAssignment} {witness : ReducedMod5FieldWitness}
    (hImage : CanonicalResidueImage source witness) :
    FieldCenteredRoot witness.residueLeft ∧
      witness.residuePairResidual = 0 := by
  rcases hImage with ⟨entry, hEntry, _hIndex, hLeft, hRight⟩
  simp only [canonicalResidues, List.mem_cons,
    List.not_mem_nil, or_false] at hEntry
  rcases hEntry with rfl | rfl | rfl | rfl | rfl <;>
    simp_all [FieldCenteredRoot, fieldCenteredResidual,
      ReducedMod5FieldWitness.residuePairResidual, coefficient]

private theorem centered_pair_cases
    {left right : F}
    (hLeft : FieldCenteredRoot left)
    (hPair : right * (left - right) = 0) :
    (left = -1 ∧ right = -1) ∨
      (left = -1 ∧ right = 0) ∨
      (left = 0 ∧ right = 0) ∨
      (left = 1 ∧ right = 0) ∨
      (left = 1 ∧ right = 1) := by
  change left * (left - 1) * (left + 1) = 0 at hLeft
  have hLeftCases : left = -1 ∨ left = 0 ∨ left = 1 := by
    rcases mul_eq_zero.mp hLeft with hZeroOne | hNeg
    · rcases mul_eq_zero.mp hZeroOne with hZero | hOne
      · exact Or.inr (Or.inl hZero)
      · exact Or.inr (Or.inr (sub_eq_zero.mp hOne))
    · exact Or.inl (eq_neg_of_add_eq_zero_left hNeg)
  rcases mul_eq_zero.mp hPair with hRightZero | hEqual
  · rcases hLeftCases with hLeft | hLeft | hLeft <;>
      subst left <;> subst right <;> simp
  · have hLeftRight : left = right := sub_eq_zero.mp hEqual
    subst right
    rcases hLeftCases with hLeft | hLeft | hLeft <;>
      subst left <;> simp

private theorem generated_scaffolding_of_structural
    {source : SourceAssignment} {chunk : Chunk}
    {witness : ReducedMod5FieldWitness}
    (hStructural : StructuralSourceImage source chunk witness)
    (hResidue : FieldCenteredRoot witness.residueLeft ∧
      witness.residuePairResidual = 0) :
    GeneratedSourceScaffolding source := by
  rcases hStructural with ⟨hDecoder, hChunkImage, hBitImage⟩
  rcases hChunkImage with ⟨hOne, _hChunkBits⟩
  rcases hBitImage with ⟨hLowImage, _hHighImage⟩
  have hIndex := generated_decoder_index hDecoder
  have hProducts := generated_decoder_products hDecoder
  have hHigh := generated_decoder_high_reconstruction hDecoder
  have hQuotient := generated_decoder_quotient_reconstruction hDecoder
  have hSourceLow := sourceQuotientLowValue_eq hLowImage
  have hCoordinateLow := coordinateQuotientLowValue_eq witness
  rw [hCoordinateLow, ← hSourceLow] at hHigh
  simp only [witnessCoordinates] at hIndex hHigh hQuotient
  have hFive : (5 : F) ≠ 0 := by native_decide
  have hRecompose :
      source .quotient =
        sourceQuotientLowValue source +
          8192 * source (.quotientBit 13) := by
    apply mul_left_cancel₀ hFive
    linear_combination hQuotient - hHigh
  have hChunkRecompose :
      sourceChunkValue source = source .index + 5 * source .quotient := by
    linear_combination -hQuotient - hIndex
  have hIndexCases :
      source .index = 0 ∨ source .index = 1 ∨ source .index = 2 ∨
        source .index = 3 ∨ source .index = 4 := by
    rcases centered_pair_cases hResidue.1 hResidue.2 with
      h | h | h | h | h <;> rcases h with ⟨hLeft, hRight⟩
    · left
      simpa [hOne, hLeft, hRight] using hIndex
    · right; left
      simpa [hOne, hLeft, hRight] using hIndex
    · right; right; left
      simpa [hOne, hLeft, hRight] using hIndex
    · right; right; right; left
      simpa [hOne, hLeft, hRight] using hIndex
    · right; right; right; right
      simpa [hOne, hLeft, hRight] using hIndex
  have hIndexPolynomial :
      source .index * (source .index - source .one) *
          (source .index - 2 * source .one) *
          (source .index - 3 * source .one) *
          (source .index - 4 * source .one) = 0 := by
    rcases hIndexCases with h | h | h | h | h <;>
      rw [h] <;> simp [hOne]
  have hTerminal :
      source (.indexProduct 2) *
          (source .index - 4 * source .one) = 0 := by
    rw [hProducts.2.2, hProducts.2.1, hProducts.1]
    simpa [sub_eq_add_neg, mul_assoc] using hIndexPolynomial
  apply (generated_scaffolding_iff_equations hOne).mpr
  rcases hProducts with ⟨hProduct0, hProduct1, hProduct2⟩
  exact ⟨by simpa [sub_eq_add_neg] using hProduct0,
    by simpa [sub_eq_add_neg] using hProduct1,
    by simpa [sub_eq_add_neg] using hProduct2,
    hTerminal, hRecompose, hChunkRecompose⟩

private def generatedBitSourceRow (index : Nat) : SourceRow :=
  ⟨[⟨.quotientBit index, 1⟩],
    [⟨.one, -1⟩, ⟨.quotientBit index, 1⟩], []⟩

private theorem generated_bit_source_rows_exact :
    (sourceRows.drop 4).take 14 =
      (List.range 14).map generatedBitSourceRow := by
  decide

private theorem generatedBitSourceRow_holds
    {source : SourceAssignment} (hOne : source .one = 1) (index : Nat) :
    (generatedBitSourceRow index).Holds source ↔
      FieldBitRoot (source (.quotientBit index)) := by
  simp only [generatedBitSourceRow, SourceRow.Holds,
    evalLinearCombination, List.map, List.sum_cons, List.sum_nil,
    coefficient, FieldBitRoot, fieldBitResidual]
  rw [hOne]
  ring_nf

private theorem generated_source_bit_rows_iff_roots
    {source : SourceAssignment} (hOne : source .one = 1) :
    GeneratedSourceBitRows source ↔
      ∀ index : Fin 14, FieldBitRoot (source (.quotientBit index.val)) := by
  constructor
  · intro hRows index
    apply (generatedBitSourceRow_holds hOne index.val).mp
    apply hRows (generatedBitSourceRow index.val)
    rw [generated_bit_source_rows_exact]
    exact List.mem_map.mpr
      ⟨index.val, List.mem_range.mpr index.isLt, rfl⟩
  · intro hRoots row hRow
    rw [generated_bit_source_rows_exact] at hRow
    rcases List.mem_map.mp hRow with ⟨index, hIndex, rfl⟩
    apply (generatedBitSourceRow_holds hOne index).mpr
    exact hRoots ⟨index, List.mem_range.mp hIndex⟩

private theorem generated_source_bits_iff_field
    {source : SourceAssignment} {chunk : Chunk}
    {witness : ReducedMod5FieldWitness}
    (hOne : source .one = 1)
    (hImage : SourceBitCoordinateImage source chunk witness) :
    GeneratedSourceBitRows source ↔
      ((∀ index, FieldBitRoot (witness.quotientLow index)) ∧
        FieldBitRoot (derivedQuotientHighField chunk witness)) := by
  rcases hImage with ⟨hLow, hHigh⟩
  rw [generated_source_bit_rows_iff_roots hOne]
  constructor
  · intro hRoots
    constructor
    · intro index
      let index13 : Fin 13 := index
      rw [← hLow index13]
      exact hRoots ⟨index13.val, by omega⟩
    · rw [← hHigh]
      exact hRoots (13 : Fin 14)
  · rintro ⟨hLowRoots, hHighRoot⟩ index
    by_cases hIndex : index.val < 13
    · let lowIndex : Fin 13 := ⟨index.val, hIndex⟩
      rw [hLow lowIndex]
      exact hLowRoots lowIndex
    · have hEq : index = (13 : Fin 14) := Fin.ext (by omega)
      subst index
      have hSourceHigh :
          source (.quotientBit (13 : Fin 14).val) =
            derivedQuotientHighField chunk witness := by
        simpa using hHigh
      rwa [hSourceHigh]

/-- Arbitrary encoded-witness soundness for the active leaf. The generated
decoder and exact input/coordinate images derive every removed source row;
no canonical residue-image premise is assumed. -/
theorem generatedEmittedAccepts_implies_source_of_structuralImage
    {source : SourceAssignment} {chunk : Chunk}
    {witness : ReducedMod5FieldWitness}
    (hStructural : StructuralSourceImage source chunk witness)
    (hEmitted : GeneratedEmittedAccepts chunk witness) :
    GeneratedSourceAccepts source := by
  have hPacked :=
    (generatedEmittedAccepts_iff_packed chunk witness).mp hEmitted
  have hDirect :=
    (packedReducedMod5FieldRows_iff_direct chunk witness).mp hPacked
  rcases hDirect with ⟨hLowRoots, hHighRoot, hLeftRoot, hPair⟩
  have hScaffolding := generated_scaffolding_of_structural hStructural
    ⟨hLeftRoot, hPair⟩
  have hOne := hStructural.2.1.1
  have hBitRows :=
    (generated_source_bits_iff_field hOne hStructural.2.2).mpr
      ⟨hLowRoots, hHighRoot⟩
  exact (generated_source_partition source).mpr ⟨hScaffolding, hBitRows⟩

/-- Canonical-materializer completeness for the active leaf. The canonical
centered-residue image is required in this direction because arbitrary packed
`L,R` cells are not determined by the unsigned source index alone. -/
theorem generatedSourceAccepts_implies_emitted_of_canonicalImage
    {source : SourceAssignment} {chunk : Chunk}
    {witness : ReducedMod5FieldWitness}
    (hCanonical : CanonicalSourceImage source chunk witness)
    (hSource : GeneratedSourceAccepts source) :
    GeneratedEmittedAccepts chunk witness := by
  have hStructural := hCanonical.1
  have hOne := hStructural.2.1.1
  have hBitRows := (generated_source_partition source).mp hSource |>.2
  have hFieldBits :=
    (generated_source_bits_iff_field hOne hStructural.2.2).mp hBitRows
  have hResidue := canonicalResidueImage_rows hCanonical.2
  have hDirect : ReducedMod5FieldRows chunk witness :=
    ⟨hFieldBits.1, hFieldBits.2, hResidue.1, hResidue.2⟩
  apply (generatedEmittedAccepts_iff_packed chunk witness).mpr
  exact (packedReducedMod5FieldRows_iff_direct chunk witness).mpr hDirect

/-- Pointwise equivalence is valid only at the canonical materializer image. -/
theorem generatedSourceAccepts_iff_emitted_of_canonicalImage
    {source : SourceAssignment} {chunk : Chunk}
    {witness : ReducedMod5FieldWitness}
    (hCanonical : CanonicalSourceImage source chunk witness) :
    GeneratedSourceAccepts source ↔ GeneratedEmittedAccepts chunk witness := by
  constructor
  · exact generatedSourceAccepts_implies_emitted_of_canonicalImage hCanonical
  · exact generatedEmittedAccepts_implies_source_of_structuralImage hCanonical.1

end SuperNeo.FPrimeRecursiveVerifier.PiRlcChallenge.PackedMod5Artifact
