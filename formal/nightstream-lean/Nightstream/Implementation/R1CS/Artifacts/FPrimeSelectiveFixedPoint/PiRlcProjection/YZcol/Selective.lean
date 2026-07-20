import Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.Census

/-!
Stable artifact surface for the selected source-to-selective row mapping.

Owns: one checked name for the 14-leaf, 139-fragment, 1,254-row compiler
ownership artifact.

Does not own: rewrite semantics, final matrices, columns, authority, or row
removal.

Emits constraints: no.

| Export | Guarantee |
|---|---|
| `Checked.artifact` | compact generated interval data |
| `Checked.sourceIndexAgreement` | exact join to source-stage owners |
| `Checked.emittedIntervalsDisjoint` | every selected emitted row has one owner |
-/

namespace Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.Checked

abbrev artifact : Selective.Artifact := Generated.SelectiveRows.artifact

abbrev sourceIndexAgreement := Census.sourceIndexAgreement

abbrev fragmentSourceCoverage := Census.fragmentSourceCoverage

abbrev emittedIntervalsDisjoint := Census.emittedIntervalsDisjoint

end Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.Checked
