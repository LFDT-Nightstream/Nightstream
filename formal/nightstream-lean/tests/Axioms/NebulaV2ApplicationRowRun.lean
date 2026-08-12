import Nightstream.Protocol.NebulaV2.ApplicationRowRun
import tests.Axioms.Support

namespace tests.Axioms.NebulaV2ApplicationRowRun

open Nightstream.Protocol.NebulaV2.ApplicationRowRun
open NightstreamTests.Axioms

/-- info: 'Nightstream.Protocol.NebulaV2.ApplicationRowRun.Runs.complete_inverse' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Runs.complete_inverse

/-- info: 'Nightstream.Protocol.NebulaV2.ApplicationRowRun.CheckedCompletedRows.completedExecution' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms CheckedCompletedRows.completedExecution

/-- info: 'Nightstream.Protocol.NebulaV2.ApplicationRowRun.CheckedCompletedRows.segmentAccessesOfRows_execution' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms CheckedCompletedRows.segmentAccessesOfRows_execution

end tests.Axioms.NebulaV2ApplicationRowRun
