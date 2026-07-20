import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.Carrier270
import tests.Axioms.Support

/-! Kernel dependency report for fixed-point private-alignment padding. -/

open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270

/-- info: 'Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270.PrivatePaddingRefinement.generated_raw_row_refines' depends on axioms: [propext,
 Classical.choice,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms PrivatePaddingRefinement.generated_raw_row_refines

/-- info: 'Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270.PrivatePaddingRefinement.withPrivatePaddingZero_satisfies' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms PrivatePaddingRefinement.withPrivatePaddingZero_satisfies
