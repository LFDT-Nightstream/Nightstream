import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol
import tests.Axioms.Support

/-! Kernel dependency report for the bounded PiRLC shared + `y_zcol` source bridge. -/

open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol

/-- info: 'Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Checked.exactRows' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Checked.exactRows

/-- info: 'Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Checked.structureValid' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Checked.structureValid

/-- info: 'Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Checked.sourceStagePathsUnique' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Checked.sourceStagePathsUnique

/-- info: 'Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Census.definitionOutputs_eq_allocatedColumns' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Census.definitionOutputs_eq_allocatedColumns

/-- info: 'Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.ArtifactRows.certificate_covers' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms ArtifactRows.certificate_covers

/-- info: 'Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.ArtifactRows.rowsSatisfied_of_sourceRows' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms ArtifactRows.rowsSatisfied_of_sourceRows

/-- info: 'Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.ProducerBinding.serializerIndicesMatch' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms ProducerBinding.serializerIndicesMatch

/-- info: 'Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.ActiveBridge.rows_decodedOutput_eq_messageAggregate_or_badRoot' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms ActiveBridge.rows_decodedOutput_eq_messageAggregate_or_badRoot
