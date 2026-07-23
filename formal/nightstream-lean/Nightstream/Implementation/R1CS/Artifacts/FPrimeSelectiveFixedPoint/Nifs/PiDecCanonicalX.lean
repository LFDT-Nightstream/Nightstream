import Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.Nifs.PiDecCanonicalX.Generated.Coordinates
import Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.Nifs.PiDecCanonicalX.Generated.DifferentialCases
import Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.Nifs.PiDecCanonicalX.Generated.Metadata
import Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.Nifs.PiDecCanonicalX.Generated.Rows

/-!
Bounded production-shaped strict-`PiDEC` canonical-X artifact facade.

Assurance tier: proof-free generated data with bounded artifact census facts.

Owns: the exact `54 x 5`, fourteen-child coordinate map returned by the live
strict emitter; 4,590 exact physical sparse rows; row ownership labels; and
kernel composition of shards containing at most 240 proof-free records.

Does not own: equality with the independent Lean compiler, row satisfaction,
whole-`PiDEC` acceptance, fixed-point private columns, or semantic authority.

Emits constraints: no; this module describes rows emitted elsewhere.

| Artifact leaf | Exact payload | Excluded boundary |
|---|---|---|
| coordinates | 270 coordinate records reconstructing 4,591 mapped columns | compiler meaning |
| rows | 270 recomposition plus 4,320 canonicality rows | satisfaction |
| ownership | one indexed label and physical index per row | production composition |
-/

namespace Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.Nifs.PiDecCanonicalX

abbrev coordinates := Generated.coordinates
abbrev rows := Generated.rows
abbrev differentialCases := Generated.DifferentialCases.values

/- Each executable certificate below consumes at most 240 proof-free records.
The dense coordinate shards have cardinalities 100, 100, and 70. The dense
recomposition-row shards have the same cardinalities; the eighteen sparse
canonicality shards contain 240 records each. No proof-bearing structure is
evaluated. -/

private theorem coordinateChunk0_length :
    Generated.Coordinates.Chunk0.values.length = 100 := by native_decide

private theorem coordinateChunk1_length :
    Generated.Coordinates.Chunk1.values.length = 100 := by native_decide

private theorem coordinateChunk2_length :
    Generated.Coordinates.Chunk2.values.length = 70 := by native_decide

private theorem rowChunk0_length :
    Generated.Rows.Chunk0.values.length = 100 := by native_decide

private theorem rowChunk1_length :
    Generated.Rows.Chunk1.values.length = 100 := by native_decide

private theorem rowChunk2_length :
    Generated.Rows.Chunk2.values.length = 70 := by native_decide

private theorem rowChunk3_length :
    Generated.Rows.Chunk3.values.length = 240 := by native_decide

private theorem rowChunk4_length :
    Generated.Rows.Chunk4.values.length = 240 := by native_decide

private theorem rowChunk5_length :
    Generated.Rows.Chunk5.values.length = 240 := by native_decide

private theorem rowChunk6_length :
    Generated.Rows.Chunk6.values.length = 240 := by native_decide

private theorem rowChunk7_length :
    Generated.Rows.Chunk7.values.length = 240 := by native_decide

private theorem rowChunk8_length :
    Generated.Rows.Chunk8.values.length = 240 := by native_decide

private theorem rowChunk9_length :
    Generated.Rows.Chunk9.values.length = 240 := by native_decide

private theorem rowChunk10_length :
    Generated.Rows.Chunk10.values.length = 240 := by native_decide

private theorem rowChunk11_length :
    Generated.Rows.Chunk11.values.length = 240 := by native_decide

private theorem rowChunk12_length :
    Generated.Rows.Chunk12.values.length = 240 := by native_decide

private theorem rowChunk13_length :
    Generated.Rows.Chunk13.values.length = 240 := by native_decide

private theorem rowChunk14_length :
    Generated.Rows.Chunk14.values.length = 240 := by native_decide

private theorem rowChunk15_length :
    Generated.Rows.Chunk15.values.length = 240 := by native_decide

private theorem rowChunk16_length :
    Generated.Rows.Chunk16.values.length = 240 := by native_decide

private theorem rowChunk17_length :
    Generated.Rows.Chunk17.values.length = 240 := by native_decide

private theorem rowChunk18_length :
    Generated.Rows.Chunk18.values.length = 240 := by native_decide

private theorem rowChunk19_length :
    Generated.Rows.Chunk19.values.length = 240 := by native_decide

private theorem rowChunk20_length :
    Generated.Rows.Chunk20.values.length = 240 := by native_decide

theorem coordinates_length : coordinates.length = 270 := by
  change ([Generated.Coordinates.Chunk0.values,
    Generated.Coordinates.Chunk1.values,
    Generated.Coordinates.Chunk2.values].flatten).length = 270
  simp [coordinateChunk0_length, coordinateChunk1_length,
    coordinateChunk2_length]

theorem rows_length : rows.length = 4590 := by
  change ([Generated.Rows.Chunk0.values, Generated.Rows.Chunk1.values,
    Generated.Rows.Chunk2.values, Generated.Rows.Chunk3.values,
    Generated.Rows.Chunk4.values, Generated.Rows.Chunk5.values,
    Generated.Rows.Chunk6.values, Generated.Rows.Chunk7.values,
    Generated.Rows.Chunk8.values, Generated.Rows.Chunk9.values,
    Generated.Rows.Chunk10.values, Generated.Rows.Chunk11.values,
    Generated.Rows.Chunk12.values, Generated.Rows.Chunk13.values,
    Generated.Rows.Chunk14.values, Generated.Rows.Chunk15.values,
    Generated.Rows.Chunk16.values, Generated.Rows.Chunk17.values,
    Generated.Rows.Chunk18.values, Generated.Rows.Chunk19.values,
    Generated.Rows.Chunk20.values].flatten).length = 4590
  simp [rowChunk0_length, rowChunk1_length, rowChunk2_length,
    rowChunk3_length, rowChunk4_length, rowChunk5_length, rowChunk6_length,
    rowChunk7_length, rowChunk8_length, rowChunk9_length, rowChunk10_length,
    rowChunk11_length, rowChunk12_length, rowChunk13_length,
    rowChunk14_length, rowChunk15_length, rowChunk16_length,
    rowChunk17_length, rowChunk18_length, rowChunk19_length,
    rowChunk20_length]

/-- The differential certificate has eleven proof-free records. Each nested
child and evaluation-arity list contains at most fourteen scalars. -/
theorem differentialCases_length : differentialCases.length = 11 := by
  native_decide

theorem profile_exact :
    Generated.Metadata.xRows = 54 ∧
    Generated.Metadata.activeColumns = 5 ∧
    Generated.Metadata.childCount = 14 ∧
    Generated.Metadata.logicalCoordinates = 270 ∧
    Generated.Metadata.canonicalColumnCount = 4591 ∧
    Generated.Metadata.rowCount = 4590 := by
  decide

theorem physical_ranges_exact :
    Generated.Metadata.strictRowStart ≤
        Generated.Metadata.recompositionRowStart ∧
    Generated.Metadata.recompositionRowEnd -
        Generated.Metadata.recompositionRowStart = 270 ∧
    Generated.Metadata.recompositionRowEnd ≤
        Generated.Metadata.canonicalityRowStart ∧
    Generated.Metadata.canonicalityRowEnd -
        Generated.Metadata.canonicalityRowStart = 4320 ∧
    Generated.Metadata.canonicalityRowEnd ≤
        Generated.Metadata.strictRowEnd := by
  decide

theorem shard_bounds :
    (∀ count ∈ Generated.Metadata.coordinateChunkCounts, count ≤ 240) ∧
    (∀ count ∈ Generated.Metadata.rowChunkCounts, count ≤ 240) := by
  decide

end Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.Nifs.PiDecCanonicalX
