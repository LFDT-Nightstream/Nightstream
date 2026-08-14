import Nightstream.Implementation.Nebula.Production.FPrime.Fresh.FPrimeBranchRows
import tests.Axioms.Support

/-! Fail-closed dependency audit for the fresh F-prime branch relation. -/

/-- info: 'Nightstream.Implementation.Nebula.IterationZeroRows.selector_eq_one_iff_iteration_eq_zero' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.IterationZeroRows.selector_eq_one_iff_iteration_eq_zero

/-- info: 'Nightstream.Implementation.Nebula.IterationZeroRows.selector_eq_zero_iff_iteration_ne_zero' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.IterationZeroRows.selector_eq_zero_iff_iteration_ne_zero

/-- info: 'Nightstream.Implementation.Nebula.ProductionFreshFPrimeBranchRows.sound' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductionFreshFPrimeBranchRows.sound

/-- info: 'Nightstream.Implementation.Nebula.ProductionFreshFPrimeBranchRows.complete_base' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductionFreshFPrimeBranchRows.complete_base

/-- info: 'Nightstream.Implementation.Nebula.ProductionFreshFPrimeBranchRows.complete_recursive' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductionFreshFPrimeBranchRows.complete_recursive
