import Nightstream.Protocol.NebulaV2.ScanSnapshotCoverage
import tests.Axioms.Support

open Nightstream.Protocol.NebulaV2.ScanSnapshotCoverage

/-- info: 'Nightstream.Protocol.NebulaV2.ScanSnapshotCoverage.nestedTupleList_eq_snapshotTupleList' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms nestedTupleList_eq_snapshotTupleList

/-- info: 'Nightstream.Protocol.NebulaV2.ScanSnapshotCoverage.nestedRecords_eq_snapshotTupleList' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms nestedRecords_eq_snapshotTupleList

/-- info: 'Nightstream.Protocol.NebulaV2.ScanSnapshotCoverage.nestedRecords_eq_snapshotTuples' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms nestedRecords_eq_snapshotTuples
