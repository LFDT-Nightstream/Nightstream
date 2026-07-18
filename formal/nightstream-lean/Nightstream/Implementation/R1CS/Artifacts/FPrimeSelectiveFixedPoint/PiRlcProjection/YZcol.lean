import Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.ExactRows
import Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Census
import Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective

/-!
Stable handwritten surface for the bounded tiny-lifecycle 15-source/13-matrix
fixed-point PiRLC cross-branch source artifact.

Owns: the stable import boundary for the exact test-fixture artifact and its
kernel-checked physical-row certificate.

Does not own: generated values, selective lowering, semantic correspondence,
final costs, or row removal.

Emits constraints: no.

| Child | Owns | Does not own |
|---|---|---|
| `SourceMap` | producer serializer indices and source columns | consumer equality or source authority |
| `Schema` | trace reconstruction and exact row scheduling | generated payloads or proof conclusions |
| `Checked.artifact` | one stable name for the retained fixture | production-wide conformance |
| `Checked.lowTrace` / `highTrace` | named semantic views of the two limb owners | trace truth under an assignment |
| `Checked.sourceStageLeaves` | 14 exact source-definition/check owners under Rust stage paths | selective lowering or encoded cost |
| `Checked.exactRows` | all 5,724 source rows match the reconstructed equations | selective lowering, satisfaction, or semantics |
| `Checked.structureValid` | exact row/column/owner census | row truth or protocol authority |
| `Selective.Checked.artifact` | exact 139-fragment mapping to 1,254 selective rows | rewrite semantics or columns |

Assurance tier: source-artifact-checked for this bounded tiny fixture only. Generated
modules instantiate the schema behind this facade; handwritten correspondence
must not import an individual generated shard.
-/

namespace Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol

/- Stable checked fixture surface. Downstream import checks enforce this
facade as the convention instead of letting correspondence track shard layout. -/
namespace Checked

abbrev artifact : Artifact := Generated.Metadata.artifact

abbrev lowLimb : LimbOwner := Generated.Metadata.limb0

abbrev highLimb : LimbOwner := Generated.Metadata.limb1

abbrev lowTrace : ProjectionProgram.ProjectionTrace :=
  lowLimb.trace artifact.shared

abbrev highTrace : ProjectionProgram.ProjectionTrace :=
  highLimb.trace artifact.shared

abbrev traces : List ProjectionProgram.ProjectionTrace :=
  [lowTrace, highTrace]

abbrev sourceStageLeaves : List SourceStageLeaf :=
  artifact.sourceStageLeaves Generated.StagePaths.paths

theorem exactRows : artifact.certificate.ExactRows := ExactRows.exact

theorem structureValid : artifact.StructureValid := Census.structureValid

theorem sourceStagePathsUnique :
    (sourceStageLeaves.map SourceStageLeaf.stagePath).Nodup :=
  Census.sourceStagePathsUnique

theorem sourceStageLeafCounts :
    sourceStageLeaves.map (fun leaf =>
      (leaf.definitionCount, leaf.checkCount, leaf.freshColumnCount)) =
      [ (272, 0, 272),
        (1620, 0, 1620),
        (1620, 0, 1620),
        (75, 0, 75),
        (108, 0, 108),
        (106, 0, 106),
        (5, 0, 5),
        (0, 2, 0),
        (1620, 0, 1620),
        (75, 0, 75),
        (108, 0, 108),
        (106, 0, 106),
        (5, 0, 5),
        (0, 2, 0) ] :=
  Census.sourceStageLeafCounts

end Checked


end Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol
