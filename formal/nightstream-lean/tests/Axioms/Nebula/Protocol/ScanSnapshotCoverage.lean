import Nightstream.Protocol.Nebula.ScanSnapshotCoverage
import tests.Axioms.Support

open Nightstream.Protocol.Nebula.ScanSnapshotCoverage

/-- info: 'Nightstream.Protocol.Nebula.ScanSnapshotCoverage.nestedTupleList_eq_snapshotTupleList' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms nestedTupleList_eq_snapshotTupleList

/-- info: 'Nightstream.Protocol.Nebula.ScanSnapshotCoverage.nestedRecords_eq_snapshotTupleList' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms nestedRecords_eq_snapshotTupleList

/-- info: 'Nightstream.Protocol.Nebula.ScanSnapshotCoverage.nestedRecords_eq_snapshotTuples' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms nestedRecords_eq_snapshotTuples
