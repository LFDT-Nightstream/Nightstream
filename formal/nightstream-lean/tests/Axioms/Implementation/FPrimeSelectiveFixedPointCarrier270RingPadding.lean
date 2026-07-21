import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.Carrier270
import tests.Axioms.Support

/-! Kernel dependency report for fixed-point final ring padding. -/

open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270

/-- info: 'Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270.RingPaddingRefinement.generated_raw_row_refines' depends on axioms: [propext,
 Classical.choice,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms RingPaddingRefinement.generated_raw_row_refines

/-- info: 'Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270.RingPaddingRefinement.withRingPaddingZero_satisfies' depends on axioms: [propext,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms RingPaddingRefinement.withRingPaddingZero_satisfies
