import Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.Nifs.PiDec.Generated.Layout
import Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.Nifs.PiDec.Generated.Metadata
import Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.Nifs.PiDec.Generated.Rows

/-!
Bounded active strict-`PiDEC` source artifact.

Assurance tier: proof-free generated data plus artifact-checked census facts
for the tiny fixed-point fixture with `kappa = 4`. The correspondence owner
performs the coefficient comparison with the independent compiler.

Owns: the concrete raw active layout, all 11,845 generated Rust sparse source
rows, and their bounded schema/arity/count checks.

Does not own: final selective-CCS rows, production `kappa = 18`, witness
values, compiler semantics, delayed sidecars, `FixedActive.ResultTransition`,
or row removal.

Emits constraints: no; this module describes existing source rows.

| Artifact leaf | Checked fact | Excluded authority |
|---|---|---|
| metadata | exact bounded profile and source interval | production profile |
| layout | schema, arity, and trace census | semantic equations |
| rows | exact generated row count | satisfaction or acceptance |
-/

namespace Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.Nifs.PiDec

abbrev rawLayout := Generated.Layout.value
abbrev sourceRows := Generated.sourceRows
abbrev commitmentRows := Generated.Metadata.commitmentRows

/- Each executable subject below is one proof-free `List Row`: chunks 0--46
contain exactly 250 rows and chunk 47 contains exactly 95 rows.  The global
count is reconstructed by the kernel; it is never submitted to
`native_decide`. -/
set_option maxRecDepth 100000

private theorem sourceChunk0_length : Generated.Rows.Chunk0.values.length = 250 := by native_decide
private theorem sourceChunk1_length : Generated.Rows.Chunk1.values.length = 250 := by native_decide
private theorem sourceChunk2_length : Generated.Rows.Chunk2.values.length = 250 := by native_decide
private theorem sourceChunk3_length : Generated.Rows.Chunk3.values.length = 250 := by native_decide
private theorem sourceChunk4_length : Generated.Rows.Chunk4.values.length = 250 := by native_decide
private theorem sourceChunk5_length : Generated.Rows.Chunk5.values.length = 250 := by native_decide
private theorem sourceChunk6_length : Generated.Rows.Chunk6.values.length = 250 := by native_decide
private theorem sourceChunk7_length : Generated.Rows.Chunk7.values.length = 250 := by native_decide
private theorem sourceChunk8_length : Generated.Rows.Chunk8.values.length = 250 := by native_decide
private theorem sourceChunk9_length : Generated.Rows.Chunk9.values.length = 250 := by native_decide
private theorem sourceChunk10_length : Generated.Rows.Chunk10.values.length = 250 := by native_decide
private theorem sourceChunk11_length : Generated.Rows.Chunk11.values.length = 250 := by native_decide
private theorem sourceChunk12_length : Generated.Rows.Chunk12.values.length = 250 := by native_decide
private theorem sourceChunk13_length : Generated.Rows.Chunk13.values.length = 250 := by native_decide
private theorem sourceChunk14_length : Generated.Rows.Chunk14.values.length = 250 := by native_decide
private theorem sourceChunk15_length : Generated.Rows.Chunk15.values.length = 250 := by native_decide
private theorem sourceChunk16_length : Generated.Rows.Chunk16.values.length = 250 := by native_decide
private theorem sourceChunk17_length : Generated.Rows.Chunk17.values.length = 250 := by native_decide
private theorem sourceChunk18_length : Generated.Rows.Chunk18.values.length = 250 := by native_decide
private theorem sourceChunk19_length : Generated.Rows.Chunk19.values.length = 250 := by native_decide
private theorem sourceChunk20_length : Generated.Rows.Chunk20.values.length = 250 := by native_decide
private theorem sourceChunk21_length : Generated.Rows.Chunk21.values.length = 250 := by native_decide
private theorem sourceChunk22_length : Generated.Rows.Chunk22.values.length = 250 := by native_decide
private theorem sourceChunk23_length : Generated.Rows.Chunk23.values.length = 250 := by native_decide
private theorem sourceChunk24_length : Generated.Rows.Chunk24.values.length = 250 := by native_decide
private theorem sourceChunk25_length : Generated.Rows.Chunk25.values.length = 250 := by native_decide
private theorem sourceChunk26_length : Generated.Rows.Chunk26.values.length = 250 := by native_decide
private theorem sourceChunk27_length : Generated.Rows.Chunk27.values.length = 250 := by native_decide
private theorem sourceChunk28_length : Generated.Rows.Chunk28.values.length = 250 := by native_decide
private theorem sourceChunk29_length : Generated.Rows.Chunk29.values.length = 250 := by native_decide
private theorem sourceChunk30_length : Generated.Rows.Chunk30.values.length = 250 := by native_decide
private theorem sourceChunk31_length : Generated.Rows.Chunk31.values.length = 250 := by native_decide
private theorem sourceChunk32_length : Generated.Rows.Chunk32.values.length = 250 := by native_decide
private theorem sourceChunk33_length : Generated.Rows.Chunk33.values.length = 250 := by native_decide
private theorem sourceChunk34_length : Generated.Rows.Chunk34.values.length = 250 := by native_decide
private theorem sourceChunk35_length : Generated.Rows.Chunk35.values.length = 250 := by native_decide
private theorem sourceChunk36_length : Generated.Rows.Chunk36.values.length = 250 := by native_decide
private theorem sourceChunk37_length : Generated.Rows.Chunk37.values.length = 250 := by native_decide
private theorem sourceChunk38_length : Generated.Rows.Chunk38.values.length = 250 := by native_decide
private theorem sourceChunk39_length : Generated.Rows.Chunk39.values.length = 250 := by native_decide
private theorem sourceChunk40_length : Generated.Rows.Chunk40.values.length = 250 := by native_decide
private theorem sourceChunk41_length : Generated.Rows.Chunk41.values.length = 250 := by native_decide
private theorem sourceChunk42_length : Generated.Rows.Chunk42.values.length = 250 := by native_decide
private theorem sourceChunk43_length : Generated.Rows.Chunk43.values.length = 250 := by native_decide
private theorem sourceChunk44_length : Generated.Rows.Chunk44.values.length = 250 := by native_decide
private theorem sourceChunk45_length : Generated.Rows.Chunk45.values.length = 250 := by native_decide
private theorem sourceChunk46_length : Generated.Rows.Chunk46.values.length = 250 := by native_decide
private theorem sourceChunk47_length : Generated.Rows.Chunk47.values.length = 1 := by native_decide

theorem sourceRows_length :
    sourceRows.length = Generated.Metadata.sourceRowCount := by
  change Generated.sourceRows.length = 11751
  simp only [Generated.sourceRows, List.length_append,
    sourceChunk0_length, sourceChunk1_length, sourceChunk2_length,
    sourceChunk3_length, sourceChunk4_length, sourceChunk5_length,
    sourceChunk6_length, sourceChunk7_length, sourceChunk8_length,
    sourceChunk9_length, sourceChunk10_length, sourceChunk11_length,
    sourceChunk12_length, sourceChunk13_length, sourceChunk14_length,
    sourceChunk15_length, sourceChunk16_length, sourceChunk17_length,
    sourceChunk18_length, sourceChunk19_length, sourceChunk20_length,
    sourceChunk21_length, sourceChunk22_length, sourceChunk23_length,
    sourceChunk24_length, sourceChunk25_length, sourceChunk26_length,
    sourceChunk27_length, sourceChunk28_length, sourceChunk29_length,
    sourceChunk30_length, sourceChunk31_length, sourceChunk32_length,
    sourceChunk33_length, sourceChunk34_length, sourceChunk35_length,
    sourceChunk36_length, sourceChunk37_length, sourceChunk38_length,
    sourceChunk39_length, sourceChunk40_length, sourceChunk41_length,
    sourceChunk42_length, sourceChunk43_length, sourceChunk44_length,
    sourceChunk45_length, sourceChunk46_length, sourceChunk47_length]

theorem sourceRange_exact :
    Generated.Metadata.sourceRowEnd - Generated.Metadata.sourceRowStart =
      Generated.Metadata.sourceRowCount := by
  decide

theorem schemaVersion_exact : rawLayout.schemaVersion = 1 := by
  decide

theorem childCount_exact :
    rawLayout.children.length = Generated.Metadata.childCount := by
  native_decide

private abbrev tracePrefix := rawLayout.xSignTraces.take 135

private abbrev traceSuffix := rawLayout.xSignTraces.drop 135

/- Both executable subjects are proof-free lists of exactly 135 column pairs.
The kernel-owned `take`/`drop` identity supplies coverage and order. -/
private theorem tracePrefix_length : tracePrefix.length = 135 := by
  native_decide

private theorem traceSuffix_length : traceSuffix.length = 135 := by
  native_decide

private theorem traceChunks_cover_in_order :
    tracePrefix ++ traceSuffix = rawLayout.xSignTraces := by
  exact List.take_append_drop 135 rawLayout.xSignTraces

theorem traceCount_exact :
    rawLayout.xSignTraces.length = Generated.Metadata.logicalPublicWidth := by
  calc
    rawLayout.xSignTraces.length =
        (tracePrefix ++ traceSuffix).length :=
      congrArg List.length traceChunks_cover_in_order.symm
    _ = tracePrefix.length + traceSuffix.length := List.length_append
    _ = 135 + 135 := by rw [tracePrefix_length, traceSuffix_length]
    _ = Generated.Metadata.logicalPublicWidth := by decide

end Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.Nifs.PiDec
