import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol

/-!
Focused compile-time regressions for the fixed-point PiRLC shared + `y_zcol`
source artifact and conditional semantic bridge.

| Tree level | Regression |
|---|---|
| protocol/phase | stable correspondence root exports the source bridge |
| family | exact 14-leaf, 5,724-row, 5,720-fresh-column census remains available |
| leaf | serializer indices and producer/consumer columns remain separate |
| semantics | satisfying selected source rows imply the typed row interface |
-/

namespace Nightstream.Tests.FPrimeSelectiveFixedPointPiRlcProjectionYZcol

open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol

#check Checked.exactRows
#check Checked.structureValid
#check Checked.sourceStagePathsUnique
#check Checked.sourceStageLeafCounts
#check Census.definitionOutputs_eq_allocatedColumns
#check ProducerBinding.serializerIndicesMatch
#check ProducerBinding.producerColumnsMatchTrace
#check ActiveBridge.rowsSatisfied_of_sourceRows
#check ActiveBridge.rows_decodedOutput_eq_messageAggregate_or_badRoot

example : Checked.artifact.sourceRows.length = 5724 :=
  Census.rowCount

example : Checked.artifact.allocatedColumns.length = 5720 :=
  Census.allocatedColumnCount

example : Checked.sourceStageLeaves.length = 14 :=
  Census.sourceStageLeafCount

example :
    (Checked.sourceStageLeaves.map SourceStageLeaf.rowCount).sum = 5724 :=
  Census.sourceStageRowCount

end Nightstream.Tests.FPrimeSelectiveFixedPointPiRlcProjectionYZcol
