import Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.Nifs.PiDec.Generated.Layout
import Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.Nifs.PiDec.Generated.Metadata
import Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.Nifs.PiDec.Generated.Rows
import Nightstream.SuperNeo.Concrete.Parameters

/-!
Bounded active strict-`PiDEC` source artifact for the Nightstream
`k_rho = 16` profile.

Assurance tier: proof-free generated data plus an artifact-checked compact
profile fact for the bounded fixed-point fixture with `kappa = 4`.

Owns: the concrete raw active layout, all 13,006 generated Rust sparse source
rows, and the compact generated profile.

Does not own: a Lean proof of generated-row identity or count, final
selective-CCS rows, production `kappa`, witness values, compiler semantics,
delayed sidecars, `FixedActive.ResultTransition`, or row removal.

Emits constraints: no; this module describes existing source rows.

| Artifact leaf | Checked fact | Excluded authority |
|---|---|---|
| metadata | selected child and matrix counts | row identity |
| layout | proof-free Rust layout | semantic equations |
| rows | proof-free Rust source rows | satisfaction or acceptance |
-/

namespace Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.Nifs.PiDec

abbrev rawLayout := Generated.Layout.value
abbrev sourceRows := Generated.sourceRows
abbrev commitmentRows := Generated.Metadata.commitmentRows

open Nightstream.SuperNeo.Concrete

/-- The generated bounded profile selects the supported Nightstream child
count. This theorem does not inspect the generated layout or source rows. -/
theorem profile_matches_nightstream :
    Generated.Metadata.childCount = productionGlobalParams.k ∧
    Generated.Metadata.matrixCount = 14 ∧
    Generated.Metadata.commitmentRows = 4 ∧
    Generated.Metadata.logicalPublicWidth = 270 ∧
    Generated.Metadata.sourceRowCount = 13006 ∧
    Generated.Metadata.shardSize = 250 ∧
    Generated.Metadata.shardCount = 53 := by
  decide

end Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.Nifs.PiDec
