import Nightstream.Implementation.NebulaV2.FPrimeIterationInputRows
import tests.Axioms.Support

/-! Fail-closed dependency audit for the base F-prime iteration row. -/

/-- info: 'Nightstream.Implementation.NebulaV2.FPrimeIterationInputRows.sound' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.FPrimeIterationInputRows.sound

/-- info: 'Nightstream.Implementation.NebulaV2.FPrimeIterationInputRows.complete' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.FPrimeIterationInputRows.complete
