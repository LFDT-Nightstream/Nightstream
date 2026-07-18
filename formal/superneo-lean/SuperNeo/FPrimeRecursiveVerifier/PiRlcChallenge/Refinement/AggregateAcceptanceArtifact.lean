import SuperNeo.FPrimeRecursiveVerifier.PiRlcChallenge.Refinement.Generated.AggregateAcceptanceArtifactData

/-!
Owns: direct evaluation of the generated active aggregate-acceptance artifact
and its refinement to the canonical four source rows and proved nine-row
aggregate relation.

Does not own: selector materialization, inactive rows, decoded-LC inputs,
`ChunkBitOuterImage`, or the fixed-F' 960-role bridge.

Emits constraints: no.

Authority boundary: this theorem family applies only to the generated
singleton-input leaf. The sixteen supplied chunk coordinates remain
authoritative and require `ChunkBitsAreBoolean`; generated geometry and rows
cannot authorize an outer decoded linear combination.

| Predicate/theorem | Rust stage | Guarantee | Assumptions | Row-removal scope |
|---|---|---|---|---|
| `generated_shape_exact` | active acceptance artifact | Exact 4/2/16/15/9, 64 chunks, arity 48, 40 bindings, 25 terms | Generated data | Accounting only |
| `generatedSourceAccepts_iff_canonical` | source rows | Generated rows are exactly `CanonicalAcceptanceSourceRows` | Singleton source image | Active leaf |
| `generatedDecoder_exact` | projected inverse | Generated decoder is the canonical inverse materializer | Singleton source image | Active leaf |
| `generatedEmittedAccepts_iff_aggregate` | nine active rows | Generated matrix rows and polynomial equal `AggregateAcceptanceRows` | Singleton coordinate image | Active leaf |
| `generatedLeaf_sound` / `generatedLeaf_complete` | whole active leaf | Source and emitted forms accept identically | Exact generated role image | Active leaf only |
| `generatedLeaf_extension_exact` | whole active leaf | Canonical inverse and tree outputs extend uniquely | Boolean source bits | Active leaf only |
-/

namespace SuperNeo.FPrimeRecursiveVerifier.PiRlcChallenge.AggregateAcceptanceArtifact

open AggregateAcceptanceArtifactData

/-- Exact singleton source image used by the generated Rust fixture. -/
def sourceAssignment
    (bits : Fin 16 → F) (accept inverse : F) : SourceAssignment
  | .one => 1
  | .chunkBit index => if h : index < 16 then bits ⟨index, h⟩ else 0
  | .accept => accept
  | .inverse => inverse

/-- Exact singleton encoded-coordinate image used by the nine active rows. -/
def coordinateAssignment
    (bits : Fin 16 → F) (outputs : ProductTreeOutputs)
    (accept : F) : CoordinateAssignment
  | .one => 1
  | .chunkBit index => if h : index < 16 then bits ⟨index, h⟩ else 0
  | .accept => accept
  | .treeOutput index => if h : index < 14 then outputs ⟨index, h⟩ else 0

/-- Direct acceptance of all four generated source rows. -/
def GeneratedSourceAccepts
    (bits : Fin 16 → F) (accept inverse : F) : Prop :=
  ∀ row ∈ sourceRows, row.Holds (sourceAssignment bits accept inverse)

/-- Direct acceptance of the generated canonical inverse decoder. -/
def GeneratedDecoderAccepts
    (bits : Fin 16 → F) (accept inverse : F) : Prop :=
  inverseDecoder.Holds (sourceAssignment bits accept inverse)

/-- Direct acceptance of all nine generated active rows. -/
def GeneratedEmittedAccepts
    (bits : Fin 16 → F) (outputs : ProductTreeOutputs)
    (accept : F) : Prop :=
  ∀ row ∈ activeRows,
    row.Holds polynomialTerms (coordinateAssignment bits outputs accept)

/-- Exact generated data shape for the active singleton-input leaf. -/
theorem generated_shape_exact :
    schemaVersion = 1 ∧
      sourceInputOrder.length = 16 ∧
      sourceAllocatedOrder.length = 2 ∧
      sourceRows.length = 4 ∧
      inverseDecoder.ownedRowOffsets = [2, 3] ∧
      chunkGeometry.length = 64 ∧
      (∀ geometry ∈ chunkGeometry,
        geometry.sourceRowEnd - geometry.sourceRowStart = 4 ∧
          geometry.sourceColumnEnd - geometry.sourceColumnStart = 2 ∧
          geometry.sourceInputColumns.length = 16 ∧
          geometry.encodedInputColumns.length = 16 ∧
          geometry.encodedAcceptanceColumns.length = 15 ∧
          geometry.activeRowEnd - geometry.activeRowStart = 9) ∧
      gateArity = 48 ∧
      matrixBindings.length = 40 ∧
      activeRows.length = 9 ∧
      polynomialTerms.length = 25 := by
  native_decide

/-- Generated source and coordinate roles have the semantic order claimed by Rust. -/
theorem generated_role_order_exact :
    sourceInputOrder = (List.range 16).map SourceRole.chunkBit ∧
      sourceAllocatedOrder = [.accept, .inverse] ∧
      coordinateOrder = [.accept] ++
        (List.range 14).map CoordinateRole.treeOutput ∧
      inverseDecoder.output = .inverse := by
  native_decide

/-- The generated production matrix indices are exact. -/
theorem generated_matrix_indices_exact :
    matrixBindings.map (fun binding => binding.index) =
      [0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
        17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
        31, 32, 33, 34, 35, 36, 37, 38, 39, 44, 45] := by
  native_decide

/-- Readable form of the exact generated 25-term sparse polynomial. -/
private noncomputable def generatedGatePolynomial (point : MatrixPoint) : F :=
  point .selector *
    (point (.productLeft 0) * point (.productRight 0) +
      point (.productLeft 1) * point (.productRight 1) +
      point (.productLeft 2) * point (.productRight 2) +
      point (.productLeft 3) * point (.productRight 3) +
      point (.productLeft 4) * point (.productRight 4) +
      point (.productLeft 5) * point (.productRight 5) +
      point (.productLeft 6) * point (.productRight 6) +
      point (.productLeft 7) * point (.productRight 7) +
      point (.productLeft 8) * point (.productRight 8) +
      point (.productLeft 9) * point (.productRight 9) +
      point (.productLeft 10) * point (.productRight 10) +
      point (.productLeft 11) * point (.productRight 11) +
      point (.productLeft 12) * point (.productRight 12) +
      point (.productLeft 13) * point (.productRight 13) +
      point (.productLeft 14) * point (.productRight 14) +
      point (.productLeft 15) * point (.productRight 15) +
      point (.productLeft 16) * point (.productRight 16) +
      point (.productLeft 17) * point (.productRight 17) -
      point .productOut +
      point .quadraticBitLeft ^ 4 -
      2 * point .quadraticBitLeft ^ 3 +
      point .quadraticBitLeft ^ 2 -
      7 * point .quadraticBitRight ^ 4 +
      14 * point .quadraticBitRight ^ 3 -
      7 * point .quadraticBitRight ^ 2)

/-- The generated sparse polynomial has exactly the readable gate equation. -/
private theorem generatedPolynomial_exact (point : MatrixPoint) :
    evalPolynomial polynomialTerms point = generatedGatePolynomial point := by
  norm_num [evalPolynomial, evalPolynomialTerm, polynomialTerms,
    generatedGatePolynomial, coefficient]
  ring

/-- Half-open natural-number interval used to audit global row ownership. -/
private def geometryInterval (start stop : Nat) : List Nat :=
  (List.range (stop - start)).map fun offset => start + offset

/-- All source rows owned by the generated chunk blocks, in chunk order. -/
private def generatedSourceRowCells : List Nat :=
  chunkGeometry.flatMap fun geometry =>
    geometryInterval geometry.sourceRowStart geometry.sourceRowEnd

/-- All two-column source allocations, in chunk order. -/
private def generatedSourceAllocatedCells : List Nat :=
  chunkGeometry.flatMap fun geometry =>
    geometryInterval geometry.sourceColumnStart geometry.sourceColumnEnd

/-- All sixteen source input roles, in chunk order. -/
private def generatedSourceInputCells : List Nat :=
  chunkGeometry.flatMap (fun geometry => geometry.sourceInputColumns)

/-- All singleton encoded source input roles, in chunk order. -/
private def generatedEncodedInputCells : List Nat :=
  chunkGeometry.flatMap (fun geometry => geometry.encodedInputColumns)

/-- All fifteen encoded acceptance roles, in chunk order. -/
private def generatedEncodedAcceptanceCells : List Nat :=
  chunkGeometry.flatMap (fun geometry => geometry.encodedAcceptanceColumns)

/-- All nine active rows per chunk, in chunk order. -/
private def generatedActiveRowCells : List Nat :=
  chunkGeometry.flatMap fun geometry =>
    geometryInterval geometry.activeRowStart geometry.activeRowEnd

/-- Start of the active partition, read from generated geometry rather than duplicated. -/
private def generatedActiveRowBase : Nat :=
  (chunkGeometry.head?.map fun geometry => geometry.activeRowStart).getD 0

/-- End of the active partition, read from the final generated chunk. -/
private def generatedActiveRowLimit : Nat :=
  (chunkGeometry.reverse.head?.map fun geometry => geometry.activeRowEnd).getD 0

/-
The generated global geometry has unique roles, disjoint source/encoded
ownership, strict chunk order, and one exact contiguous active-row partition.
-/
set_option maxRecDepth 65536 in
theorem generated_global_geometry_exact :
    (∀ geometry ∈ chunkGeometry,
      geometry.sourceRowEnd - geometry.sourceRowStart = 4 ∧
        geometry.sourceColumnEnd - geometry.sourceColumnStart = 2 ∧
        geometry.sourceInputColumns.length = 16 ∧
        geometry.sourceInputColumns.Nodup ∧
        geometry.sourceAcceptColumn = geometry.sourceColumnStart ∧
        geometry.sourceInverseColumn = geometry.sourceColumnStart + 1 ∧
        geometry.encodedInputColumns.length = 16 ∧
        geometry.encodedInputColumns.Nodup ∧
        geometry.encodedAcceptanceColumns.length = 15 ∧
        geometry.encodedAcceptanceColumns.Nodup ∧
        geometry.activeRowEnd - geometry.activeRowStart = 9) ∧
      generatedSourceRowCells.Nodup ∧
      generatedSourceInputCells.Nodup ∧
      generatedSourceAllocatedCells.Nodup ∧
      (generatedSourceInputCells ++ generatedSourceAllocatedCells).Nodup ∧
      generatedEncodedInputCells.Nodup ∧
      generatedEncodedAcceptanceCells.Nodup ∧
      (generatedEncodedInputCells ++ generatedEncodedAcceptanceCells).Nodup ∧
      generatedActiveRowCells.Nodup ∧
      (chunkGeometry.map (fun geometry => geometry.sourceRowStart)).Pairwise
        (fun left right => left < right) ∧
      (chunkGeometry.map (fun geometry => geometry.sourceColumnStart)).Pairwise
        (fun left right => left < right) ∧
      (chunkGeometry.map (fun geometry => geometry.activeRowStart)).Pairwise
        (fun left right => left < right) ∧
      generatedEncodedInputCells =
        (List.range 1024).map (fun offset => 1 + offset) ∧
      generatedActiveRowCells =
        (List.range (64 * 9)).map
          (fun offset => generatedActiveRowBase + offset) ∧
      generatedActiveRowLimit = generatedActiveRowBase + 64 * 9 := by
  native_decide

/-- Read the sixteen generated source roles through their exact singleton image. -/
private def generatedBits
    (bits : Fin 16 → F) (accept inverse : F) (index : Fin 16) : F :=
  sourceAssignment bits accept inverse (.chunkBit index.val)

private theorem generatedBits_eq
    (bits : Fin 16 → F) (accept inverse : F) :
    generatedBits bits accept inverse = bits := by
  funext index
  simp [generatedBits, sourceAssignment, index.isLt]

private theorem rejectionBucket_field :
    F.ofNat rejectionBucket = (65535 : F) := by
  native_decide

private theorem generatedDifference_eq
    (bits : Fin 16 → F) (accept inverse : F) :
    evalLinearCombination (sourceAssignment bits accept inverse)
        inverseDecoder.difference = acceptanceDifference bits := by
  calc
    evalLinearCombination (sourceAssignment bits accept inverse)
        inverseDecoder.difference =
          fieldBitsValue 16 (generatedBits bits accept inverse) - (65535 : F) := by
      simp only [inverseDecoder, evalLinearCombination, coefficient,
        generatedBits, fieldBitsValue, Function.comp_apply]
      norm_num
      rw [show sourceAssignment bits accept inverse .one = 1 from rfl]
      ring
    _ = acceptanceChunkValue bits - F.ofNat rejectionBucket := by
      rw [generatedBits_eq, rejectionBucket_field]
      rfl
    _ = acceptanceDifference bits := rfl

private theorem generatedSourceRows_explicit
    (bits : Fin 16 → F) (accept inverse : F) :
    GeneratedSourceAccepts bits accept inverse ↔
      (sourceRows[0]).Holds (sourceAssignment bits accept inverse) ∧
      (sourceRows[1]).Holds (sourceAssignment bits accept inverse) ∧
      (sourceRows[2]).Holds (sourceAssignment bits accept inverse) ∧
      (sourceRows[3]).Holds (sourceAssignment bits accept inverse) := by
  simp [GeneratedSourceAccepts, sourceRows]

private theorem generatedSourceRow0_iff
    (bits : Fin 16 → F) (accept inverse : F) :
    (sourceRows[0]).Holds (sourceAssignment bits accept inverse) ↔
      FieldBitRoot accept := by
  unfold SourceRow.Holds
  have hA :
      evalLinearCombination (sourceAssignment bits accept inverse)
          (sourceRows[0]).a = accept := by
    norm_num [sourceRows, evalLinearCombination, coefficient, sourceAssignment]
  have hB :
      evalLinearCombination (sourceAssignment bits accept inverse)
          (sourceRows[0]).b = accept - 1 := by
    norm_num [sourceRows, evalLinearCombination, coefficient, sourceAssignment]
    ring
  have hC :
      evalLinearCombination (sourceAssignment bits accept inverse)
          (sourceRows[0]).c = 0 := by
    norm_num [sourceRows, evalLinearCombination]
  rw [hA, hB, hC]
  rfl

private theorem generatedSourceRow1_iff
    (bits : Fin 16 → F) (accept inverse : F) :
    (sourceRows[1]).Holds (sourceAssignment bits accept inverse) ↔
      (1 - accept) * acceptanceDifference bits = 0 := by
  unfold SourceRow.Holds
  have hA :
      evalLinearCombination (sourceAssignment bits accept inverse)
          (sourceRows[1]).a = 1 - accept := by
    norm_num [sourceRows, evalLinearCombination, coefficient, sourceAssignment]
    ring
  have hB :
      evalLinearCombination (sourceAssignment bits accept inverse)
          (sourceRows[1]).b = acceptanceDifference bits := by
    calc
      _ = evalLinearCombination (sourceAssignment bits accept inverse)
          inverseDecoder.difference := by
        norm_num [sourceRows, inverseDecoder, evalLinearCombination,
          coefficient]
        ring
      _ = acceptanceDifference bits := generatedDifference_eq bits accept inverse
  have hC :
      evalLinearCombination (sourceAssignment bits accept inverse)
          (sourceRows[1]).c = 0 := by
    norm_num [sourceRows, evalLinearCombination]
  rw [hA, hB, hC]

private theorem generatedSourceRow2_iff
    (bits : Fin 16 → F) (accept inverse : F) :
    (sourceRows[2]).Holds (sourceAssignment bits accept inverse) ↔
      acceptanceDifference bits * inverse = accept := by
  unfold SourceRow.Holds
  have hA :
      evalLinearCombination (sourceAssignment bits accept inverse)
          (sourceRows[2]).a = acceptanceDifference bits := by
    calc
      _ = evalLinearCombination (sourceAssignment bits accept inverse)
          inverseDecoder.difference := by
        norm_num [sourceRows, inverseDecoder, evalLinearCombination,
          coefficient]
        ring
      _ = acceptanceDifference bits := generatedDifference_eq bits accept inverse
  have hB :
      evalLinearCombination (sourceAssignment bits accept inverse)
          (sourceRows[2]).b = inverse := by
    norm_num [sourceRows, evalLinearCombination, coefficient, sourceAssignment]
  have hC :
      evalLinearCombination (sourceAssignment bits accept inverse)
          (sourceRows[2]).c = accept := by
    norm_num [sourceRows, evalLinearCombination, coefficient, sourceAssignment]
  rw [hA, hB, hC]

private theorem generatedSourceRow3_iff
    (bits : Fin 16 → F) (accept inverse : F) :
    (sourceRows[3]).Holds (sourceAssignment bits accept inverse) ↔
      (1 - accept) * inverse = 0 := by
  unfold SourceRow.Holds
  have hA :
      evalLinearCombination (sourceAssignment bits accept inverse)
          (sourceRows[3]).a = 1 - accept := by
    norm_num [sourceRows, evalLinearCombination, coefficient, sourceAssignment]
    ring
  have hB :
      evalLinearCombination (sourceAssignment bits accept inverse)
          (sourceRows[3]).b = inverse := by
    norm_num [sourceRows, evalLinearCombination, coefficient, sourceAssignment]
  have hC :
      evalLinearCombination (sourceAssignment bits accept inverse)
          (sourceRows[3]).c = 0 := by
    norm_num [sourceRows, evalLinearCombination]
  rw [hA, hB, hC]

/-- The four generated equations are exactly the canonical source relation. -/
theorem generatedSourceAccepts_iff_canonical
    (bits : Fin 16 → F) (accept inverse : F) :
    GeneratedSourceAccepts bits accept inverse ↔
      CanonicalAcceptanceSourceRows bits accept inverse := by
  rw [generatedSourceRows_explicit,
    generatedSourceRow0_iff, generatedSourceRow1_iff,
    generatedSourceRow2_iff, generatedSourceRow3_iff]
  change (_ ∧ _ ∧ _ ∧ _) ↔ ((_ ∧ _ ∧ _) ∧ _)
  tauto

/-- The generated decoder evaluates to the exact canonical inverse function. -/
theorem generatedDecoder_exact
    (bits : Fin 16 → F) (accept inverse : F) :
    GeneratedDecoderAccepts bits accept inverse ↔
      inverse = if acceptanceDifference bits = 0 then 0
        else (acceptanceDifference bits)⁻¹ := by
  rw [GeneratedDecoderAccepts, CanonicalInverseDecoder.Holds]
  rw [generatedDifference_eq]
  simp [inverseDecoder, sourceAssignment]

/-- Canonical source rows satisfy the generated deterministic decoder. -/
theorem generatedSourceAccepts_implies_decoder
    {bits : Fin 16 → F} {accept inverse : F}
    (hSource : GeneratedSourceAccepts bits accept inverse) :
    GeneratedDecoderAccepts bits accept inverse := by
  have hCanonical :=
    (generatedSourceAccepts_iff_canonical bits accept inverse).mp hSource
  rw [generatedDecoder_exact]
  rcases canonicalAcceptanceSourceRows_witness_cases hCanonical with
    hZero | hNonzero
  · simp [hZero.1, hZero.2.2]
  · simp [hNonzero.1, hNonzero.2.2]

set_option maxHeartbeats 800000 in
set_option maxRecDepth 8192 in
private theorem generatedBitRow0_iff
    (bits : Fin 16 → F) (outputs : ProductTreeOutputs) (accept : F) :
    (activeRows[0]).Holds polynomialTerms
        (coordinateAssignment bits outputs accept) ↔
      QuadraticZeroPair (fieldBitResidual (outputs 0))
        (fieldBitResidual (outputs 1)) := by
  rw [ActiveRow.Holds, generatedPolynomial_exact]
  simp [generatedGatePolynomial, activeRows, ActiveRow.point,
    MatrixLinearCombination.value, evalLinearCombination, coefficient,
    coordinateAssignment]
  unfold QuadraticZeroPair fieldBitResidual
  rw [show KExt.w = (7 : F) from rfl]
  ring_nf

set_option maxHeartbeats 800000 in
set_option maxRecDepth 8192 in
private theorem generatedBitRow1_iff
    (bits : Fin 16 → F) (outputs : ProductTreeOutputs) (accept : F) :
    (activeRows[1]).Holds polynomialTerms
        (coordinateAssignment bits outputs accept) ↔
      QuadraticZeroPair (fieldBitResidual (outputs 2))
        (fieldBitResidual (outputs 3)) := by
  rw [ActiveRow.Holds, generatedPolynomial_exact]
  simp [generatedGatePolynomial, activeRows, ActiveRow.point,
    MatrixLinearCombination.value, evalLinearCombination, coefficient,
    coordinateAssignment]
  unfold QuadraticZeroPair fieldBitResidual
  rw [show KExt.w = (7 : F) from rfl]
  ring_nf

set_option maxHeartbeats 800000 in
set_option maxRecDepth 8192 in
private theorem generatedBitRow2_iff
    (bits : Fin 16 → F) (outputs : ProductTreeOutputs) (accept : F) :
    (activeRows[2]).Holds polynomialTerms
        (coordinateAssignment bits outputs accept) ↔
      QuadraticZeroPair (fieldBitResidual (outputs 4))
        (fieldBitResidual (outputs 5)) := by
  rw [ActiveRow.Holds, generatedPolynomial_exact]
  simp [generatedGatePolynomial, activeRows, ActiveRow.point,
    MatrixLinearCombination.value, evalLinearCombination, coefficient,
    coordinateAssignment]
  unfold QuadraticZeroPair fieldBitResidual
  rw [show KExt.w = (7 : F) from rfl]
  ring_nf

set_option maxHeartbeats 800000 in
set_option maxRecDepth 8192 in
private theorem generatedBitRow3_iff
    (bits : Fin 16 → F) (outputs : ProductTreeOutputs) (accept : F) :
    (activeRows[3]).Holds polynomialTerms
        (coordinateAssignment bits outputs accept) ↔
      QuadraticZeroPair (fieldBitResidual (outputs 6))
        (fieldBitResidual (outputs 7)) := by
  rw [ActiveRow.Holds, generatedPolynomial_exact]
  simp [generatedGatePolynomial, activeRows, ActiveRow.point,
    MatrixLinearCombination.value, evalLinearCombination, coefficient,
    coordinateAssignment]
  unfold QuadraticZeroPair fieldBitResidual
  rw [show KExt.w = (7 : F) from rfl]
  ring_nf

set_option maxHeartbeats 800000 in
set_option maxRecDepth 8192 in
private theorem generatedBitRow4_iff
    (bits : Fin 16 → F) (outputs : ProductTreeOutputs) (accept : F) :
    (activeRows[4]).Holds polynomialTerms
        (coordinateAssignment bits outputs accept) ↔
      QuadraticZeroPair (fieldBitResidual (outputs 8))
        (fieldBitResidual (outputs 9)) := by
  rw [ActiveRow.Holds, generatedPolynomial_exact]
  simp [generatedGatePolynomial, activeRows, ActiveRow.point,
    MatrixLinearCombination.value, evalLinearCombination, coefficient,
    coordinateAssignment]
  unfold QuadraticZeroPair fieldBitResidual
  rw [show KExt.w = (7 : F) from rfl]
  ring_nf

set_option maxHeartbeats 800000 in
set_option maxRecDepth 8192 in
private theorem generatedBitRow5_iff
    (bits : Fin 16 → F) (outputs : ProductTreeOutputs) (accept : F) :
    (activeRows[5]).Holds polynomialTerms
        (coordinateAssignment bits outputs accept) ↔
      QuadraticZeroPair (fieldBitResidual (outputs 10))
        (fieldBitResidual (outputs 11)) := by
  rw [ActiveRow.Holds, generatedPolynomial_exact]
  simp [generatedGatePolynomial, activeRows, ActiveRow.point,
    MatrixLinearCombination.value, evalLinearCombination, coefficient,
    coordinateAssignment]
  unfold QuadraticZeroPair fieldBitResidual
  rw [show KExt.w = (7 : F) from rfl]
  ring_nf

set_option maxHeartbeats 800000 in
set_option maxRecDepth 8192 in
private theorem generatedBitRow6_iff
    (bits : Fin 16 → F) (outputs : ProductTreeOutputs) (accept : F) :
    (activeRows[6]).Holds polynomialTerms
        (coordinateAssignment bits outputs accept) ↔
      QuadraticZeroPair (fieldBitResidual (outputs 12))
        (fieldBitResidual (outputs 13)) := by
  rw [ActiveRow.Holds, generatedPolynomial_exact]
  simp [generatedGatePolynomial, activeRows, ActiveRow.point,
    MatrixLinearCombination.value, evalLinearCombination, coefficient,
    coordinateAssignment]
  unfold QuadraticZeroPair fieldBitResidual
  rw [show KExt.w = (7 : F) from rfl]
  ring_nf

set_option maxHeartbeats 1000000 in
set_option maxRecDepth 8192 in
private theorem generatedAggregateRow_iff
    (bits : Fin 16 → F) (outputs : ProductTreeOutputs) (accept : F) :
    (activeRows[7]).Holds polynomialTerms
        (coordinateAssignment bits outputs accept) ↔
      ProductTreeAggregateRow bits outputs := by
  rw [ActiveRow.Holds, generatedPolynomial_exact]
  simp [generatedGatePolynomial, activeRows, ActiveRow.point,
    MatrixLinearCombination.value, evalLinearCombination, coefficient,
    coordinateAssignment]
  rw [ProductTreeAggregateRow, radix3Field_fourteen_weights]
  simp [productTreeResidual, productTreeLeft, productTreeRight]
  constructor <;> intro h <;> linear_combination -h

set_option maxHeartbeats 800000 in
set_option maxRecDepth 8192 in
private theorem generatedFinalRow_iff
    (bits : Fin 16 → F) (outputs : ProductTreeOutputs) (accept : F) :
    (activeRows[8]).Holds polynomialTerms
        (coordinateAssignment bits outputs accept) ↔
      FinalAcceptanceRow outputs accept := by
  rw [ActiveRow.Holds, generatedPolynomial_exact]
  simp [generatedGatePolynomial, activeRows, ActiveRow.point,
    MatrixLinearCombination.value, evalLinearCombination, coefficient,
    coordinateAssignment, FinalAcceptanceRow]
  constructor <;> intro h <;> linear_combination h

private theorem generatedEmittedRows_explicit
    (bits : Fin 16 → F) (outputs : ProductTreeOutputs) (accept : F) :
    GeneratedEmittedAccepts bits outputs accept ↔
      (activeRows[0]).Holds polynomialTerms
          (coordinateAssignment bits outputs accept) ∧
      (activeRows[1]).Holds polynomialTerms
          (coordinateAssignment bits outputs accept) ∧
      (activeRows[2]).Holds polynomialTerms
          (coordinateAssignment bits outputs accept) ∧
      (activeRows[3]).Holds polynomialTerms
          (coordinateAssignment bits outputs accept) ∧
      (activeRows[4]).Holds polynomialTerms
          (coordinateAssignment bits outputs accept) ∧
      (activeRows[5]).Holds polynomialTerms
          (coordinateAssignment bits outputs accept) ∧
      (activeRows[6]).Holds polynomialTerms
          (coordinateAssignment bits outputs accept) ∧
      (activeRows[7]).Holds polynomialTerms
          (coordinateAssignment bits outputs accept) ∧
      (activeRows[8]).Holds polynomialTerms
          (coordinateAssignment bits outputs accept) := by
  simp [GeneratedEmittedAccepts, activeRows]

set_option maxHeartbeats 2000000 in
set_option maxRecDepth 8192 in
/-- The nine generated rows are exactly the proved aggregate relation. -/
theorem generatedEmittedAccepts_iff_aggregate
    (bits : Fin 16 → F) (outputs : ProductTreeOutputs) (accept : F) :
    GeneratedEmittedAccepts bits outputs accept ↔
      AggregateAcceptanceRows bits outputs accept := by
  constructor
  · intro hGenerated
    rcases (generatedEmittedRows_explicit bits outputs accept).mp hGenerated with
      ⟨h0, h1, h2, h3, h4, h5, h6, h7, h8⟩
    refine ⟨⟨?_, ?_, ?_, ?_, ?_, ?_, ?_⟩, ?_, ?_⟩
    · exact (generatedBitRow0_iff bits outputs accept).mp h0
    · exact (generatedBitRow1_iff bits outputs accept).mp h1
    · exact (generatedBitRow2_iff bits outputs accept).mp h2
    · exact (generatedBitRow3_iff bits outputs accept).mp h3
    · exact (generatedBitRow4_iff bits outputs accept).mp h4
    · exact (generatedBitRow5_iff bits outputs accept).mp h5
    · exact (generatedBitRow6_iff bits outputs accept).mp h6
    · exact (generatedAggregateRow_iff bits outputs accept).mp h7
    · exact (generatedFinalRow_iff bits outputs accept).mp h8
  · rintro ⟨⟨h0, h1, h2, h3, h4, h5, h6⟩, h7, h8⟩
    apply (generatedEmittedRows_explicit bits outputs accept).mpr
    exact ⟨(generatedBitRow0_iff bits outputs accept).mpr h0,
      (generatedBitRow1_iff bits outputs accept).mpr h1,
      (generatedBitRow2_iff bits outputs accept).mpr h2,
      (generatedBitRow3_iff bits outputs accept).mpr h3,
      (generatedBitRow4_iff bits outputs accept).mpr h4,
      (generatedBitRow5_iff bits outputs accept).mpr h5,
      (generatedBitRow6_iff bits outputs accept).mpr h6,
      (generatedAggregateRow_iff bits outputs accept).mpr h7,
      (generatedFinalRow_iff bits outputs accept).mpr h8⟩

/-- Complete derived witness carried by the generated singleton leaf. -/
@[ext] structure GeneratedLeafWitness where
  outputs : ProductTreeOutputs
  accept : F
  inverse : F

/-- Direct acceptance by generated source rows, decoder, and active rows. -/
def GeneratedLeafAccepts
    (bits : Fin 16 → F) (witness : GeneratedLeafWitness) : Prop :=
  GeneratedSourceAccepts bits witness.accept witness.inverse ∧
    GeneratedDecoderAccepts bits witness.accept witness.inverse ∧
    GeneratedEmittedAccepts bits witness.outputs witness.accept

/-- Generated acceptance implies both canonical mathematical relations. -/
theorem generatedLeaf_sound
    {bits : Fin 16 → F} {witness : GeneratedLeafWitness}
    (hGenerated : GeneratedLeafAccepts bits witness) :
    CanonicalAcceptanceSourceRows bits witness.accept witness.inverse ∧
      AggregateAcceptanceRows bits witness.outputs witness.accept := by
  exact ⟨(generatedSourceAccepts_iff_canonical
      bits witness.accept witness.inverse).mp hGenerated.1,
    (generatedEmittedAccepts_iff_aggregate
      bits witness.outputs witness.accept).mp hGenerated.2.2⟩

/-- Both canonical relations materialize every generated acceptance check. -/
theorem generatedLeaf_complete
    {bits : Fin 16 → F} {witness : GeneratedLeafWitness}
    (hCanonical :
      CanonicalAcceptanceSourceRows bits witness.accept witness.inverse ∧
        AggregateAcceptanceRows bits witness.outputs witness.accept) :
    GeneratedLeafAccepts bits witness := by
  have hSource := (generatedSourceAccepts_iff_canonical
    bits witness.accept witness.inverse).mpr hCanonical.1
  exact ⟨hSource, generatedSourceAccepts_implies_decoder hSource,
    (generatedEmittedAccepts_iff_aggregate
      bits witness.outputs witness.accept).mpr hCanonical.2⟩

/-- The handwritten source and aggregate semantics are exact for this artifact. -/
theorem generatedLeaf_exact
    (bits : Fin 16 → F) (witness : GeneratedLeafWitness) :
    GeneratedLeafAccepts bits witness ↔
      CanonicalAcceptanceSourceRows bits witness.accept witness.inverse ∧
        AggregateAcceptanceRows bits witness.outputs witness.accept :=
  ⟨generatedLeaf_sound, generatedLeaf_complete⟩

/-- Boolean source bits have one and only one generated leaf extension. -/
theorem generatedLeaf_extension_exact
    (bits : Fin 16 → F) (hBits : ChunkBitsAreBoolean bits) :
    ∃! witness : GeneratedLeafWitness, GeneratedLeafAccepts bits witness := by
  let sourceWitness := canonicalAcceptanceMaterializer bits
  let witness : GeneratedLeafWitness :=
    { outputs := canonicalProductTreeOutputs bits
      accept := sourceWitness.accept
      inverse := sourceWitness.inverse }
  have hSource : CanonicalAcceptanceSourceRows bits
      sourceWitness.accept sourceWitness.inverse :=
    canonicalAcceptanceMaterializer_holds bits
  have hSourceMeaning : AcceptanceSourceMeaning bits sourceWitness.accept :=
    (canonicalAcceptanceSourceRows_exists_iff
      bits sourceWitness.accept hBits).mp ⟨sourceWitness.inverse, hSource⟩
  have hAggregate : AggregateAcceptanceRows bits
      (canonicalProductTreeOutputs bits) sourceWitness.accept :=
    (aggregateAcceptanceRows_iff_sourceMeaning bits
      (canonicalProductTreeOutputs bits) sourceWitness.accept hBits).mpr
        ⟨canonicalProductTreeOutputs_meaning bits, hSourceMeaning⟩
  have hWitness : GeneratedLeafAccepts bits witness :=
    generatedLeaf_complete ⟨hSource, hAggregate⟩
  refine ⟨witness, hWitness, ?_⟩
  intro other hOther
  have hOtherCanonical := generatedLeaf_sound hOther
  have hSourceEqual := canonicalAcceptanceSourceRows_unique
    hSource hOtherCanonical.1
  have hOtherTree := (aggregateAcceptanceRows_iff_treeAndFinal
    bits other.outputs other.accept hBits).mp hOtherCanonical.2
  have hOutputs : witness.outputs = other.outputs :=
    productTreeMeaning_unique (canonicalProductTreeOutputs_meaning bits)
      hOtherTree.1
  have hAccept : witness.accept = other.accept := hSourceEqual.1
  have hInverse : witness.inverse = other.inverse := hSourceEqual.2
  exact GeneratedLeafWitness.ext hOutputs.symm hAccept.symm hInverse.symm

end SuperNeo.FPrimeRecursiveVerifier.PiRlcChallenge.AggregateAcceptanceArtifact
