import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiCcsNc.ProductionDomain
import tests.Axioms.Support

/-! Fail-closed dependencies for the complete fixed-point domain refinement. -/

open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.ProductionDomain

/-- info: 'Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.ProductionDomain.blockLaneDomain_covers' does not depend on any axioms -/
#guard_msgs in
#audit_axioms blockLaneDomain_covers

/-- info: 'Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.ProductionDomain.rowVariables_minimal' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms rowVariables_minimal

/-- info: 'Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.ProductionDomain.flatColumnVariables_minimal' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms flatColumnVariables_minimal

/-- info: 'Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.ProductionDomain.blockVariables_minimal' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms blockVariables_minimal

/-- info: 'Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.ProductionDomain.laneVariables_minimal' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms laneVariables_minimal

/-- info: 'Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.ProductionDomain.artifact_width_accounting' does not depend on any axioms -/
#guard_msgs in
#audit_axioms artifact_width_accounting

/-- info: 'Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.ProductionDomain.artifact_fits_current_constructor_guard' does not depend on any axioms -/
#guard_msgs in
#audit_axioms artifact_fits_current_constructor_guard
