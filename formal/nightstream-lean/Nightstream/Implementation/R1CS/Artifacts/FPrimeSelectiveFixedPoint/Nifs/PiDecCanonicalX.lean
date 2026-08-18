import Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.Nifs.PiDecCanonicalX.Generated.Coordinates
import Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.Nifs.PiDecCanonicalX.Generated.DifferentialCases
import Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.Nifs.PiDecCanonicalX.Generated.Metadata
import Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.Nifs.PiDecCanonicalX.Generated.Rows

/-!
Bounded Nightstream k_rho=16 strict-`PiDEC` canonical-X artifact facade.

Assurance tier: proof-free generated data with compact profile geometry.

Owns: the exact `54 x 5`, sixteen-child coordinate map and 5,130 physical
sparse rows returned by the live strict emitter, plus their ownership labels.

Does not own: Lean equality with the independent compiler, row satisfaction,
whole-`PiDEC` acceptance, fixed-point private columns, or semantic authority.

Emits constraints: no; this module describes rows emitted elsewhere.

| Artifact leaf | Exact payload | Excluded boundary |
|---|---|---|
| coordinates | 270 coordinate records reconstructing 5,131 mapped columns | compiler meaning |
| rows | 270 recomposition plus 4,860 canonicality rows | satisfaction |
| ownership | one indexed label and physical index per row | production composition |
-/

namespace Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.Nifs.PiDecCanonicalX

abbrev coordinates := Generated.coordinates
abbrev rows := Generated.rows
abbrev differentialCases := Generated.DifferentialCases.values

theorem profile_exact :
    Generated.Metadata.radix = 2 ∧
    Generated.Metadata.xRows = 54 ∧
    Generated.Metadata.activeColumns = 5 ∧
    Generated.Metadata.childCount = 16 ∧
    Generated.Metadata.logicalCoordinates = 270 ∧
    Generated.Metadata.canonicalColumnCount = 5131 ∧
    Generated.Metadata.rowCount = 5130 := by
  decide

theorem physical_ranges_exact :
    Generated.Metadata.strictRowStart ≤
        Generated.Metadata.recompositionRowStart ∧
    Generated.Metadata.recompositionRowEnd -
        Generated.Metadata.recompositionRowStart = 270 ∧
    Generated.Metadata.recompositionRowEnd ≤
        Generated.Metadata.canonicalityRowStart ∧
    Generated.Metadata.canonicalityRowEnd -
        Generated.Metadata.canonicalityRowStart = 4860 ∧
    Generated.Metadata.canonicalityRowEnd ≤
        Generated.Metadata.strictRowEnd := by
  decide

end Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.Nifs.PiDecCanonicalX
