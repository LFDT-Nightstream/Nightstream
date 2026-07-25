import tests.FPrimeProductionFreshPublicSingletonRows
import tests.Axioms.Support

/-!
Fail-closed guards for the singleton compact-adapter/physical-row
composition.
-/

/-- info: 'Nightstream.Implementation.R1CS.FPrimeProductionFreshPublicSingletonRows.rawClaimOfAssignment_length' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeProductionFreshPublicSingletonRows.rawClaimOfAssignment_length

/-- info: 'Nightstream.Implementation.R1CS.FPrimeProductionFreshPublicSingletonRows.sourceProgram_cost_eq_rows_plus_shape' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeProductionFreshPublicSingletonRows.sourceProgram_cost_eq_rows_plus_shape

/-- info: 'Nightstream.Implementation.R1CS.FPrimeProductionFreshPublicSingletonRows.selectedRows_iff_sourceCheck' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeProductionFreshPublicSingletonRows.selectedRows_iff_sourceCheck

/-- info: 'Nightstream.Implementation.R1CS.FPrimeProductionFreshPublicSingletonRows.selectedRows_iff_sourceProgram' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeProductionFreshPublicSingletonRows.selectedRows_iff_sourceProgram

/-- info: 'Nightstream.Implementation.R1CS.FPrimeProductionFreshPublicSingletonRows.selectedRows_iff_freshPublic_eq_encodeInstance' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeProductionFreshPublicSingletonRows.selectedRows_iff_freshPublic_eq_encodeInstance

/-- info: 'Nightstream.Implementation.R1CS.FPrimeProductionFreshPublicSingletonRows.artifactRows_iff_freshPublic_eq_encodeInstance' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeProductionFreshPublicSingletonRows.artifactRows_iff_freshPublic_eq_encodeInstance
