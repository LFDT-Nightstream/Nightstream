import Nightstream.Protocol.Nebula.ApplicationRowRun
import tests.Axioms.Support

namespace tests.Axioms.NebulaApplicationRowRun

open Nightstream.Protocol.Nebula.ApplicationRowRun
open NightstreamTests.Axioms

/-- info: 'Nightstream.Protocol.Nebula.ApplicationRowRun.Runs.complete_inverse' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Runs.complete_inverse

/-- info: 'Nightstream.Protocol.Nebula.ApplicationRowRun.CheckedCompletedRows.completedExecution' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms CheckedCompletedRows.completedExecution

/-- info: 'Nightstream.Protocol.Nebula.ApplicationRowRun.CheckedCompletedRows.segmentAccessesOfRows_execution' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms CheckedCompletedRows.segmentAccessesOfRows_execution

end tests.Axioms.NebulaApplicationRowRun
