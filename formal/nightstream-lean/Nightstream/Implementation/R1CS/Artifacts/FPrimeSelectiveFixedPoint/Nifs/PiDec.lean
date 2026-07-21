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

set_option maxRecDepth 100000 in
theorem sourceRows_length :
    sourceRows.length = Generated.Metadata.sourceRowCount := by
  native_decide

theorem sourceRange_exact :
    Generated.Metadata.sourceRowEnd - Generated.Metadata.sourceRowStart =
      Generated.Metadata.sourceRowCount := by
  decide

theorem schemaVersion_exact : rawLayout.schemaVersion = 1 := by
  decide

theorem childCount_exact :
    rawLayout.children.length = Generated.Metadata.childCount := by
  native_decide

theorem traceCount_exact :
    rawLayout.xSignTraces.length = Generated.Metadata.logicalPublicWidth := by
  native_decide

end Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.Nifs.PiDec
