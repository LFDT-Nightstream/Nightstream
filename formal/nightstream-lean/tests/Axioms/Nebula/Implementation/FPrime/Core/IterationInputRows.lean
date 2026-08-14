import Nightstream.Implementation.Nebula.FPrime.Core.IterationInputRows
import tests.Axioms.Support

/-! Fail-closed dependency audit for the base F-prime iteration row. -/

/-- info: 'Nightstream.Implementation.Nebula.FPrimeIterationInputRows.sound' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.FPrimeIterationInputRows.sound

/-- info: 'Nightstream.Implementation.Nebula.FPrimeIterationInputRows.complete' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.FPrimeIterationInputRows.complete
