import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.Carrier270
import tests.Axioms.Support

/-! Kernel dependency report for fixed-point selector-row refinement. -/

open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270

/-- info: 'Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270.SelectorRefinement.generated_raw_row_decodes' depends on axioms: [propext,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms SelectorRefinement.generated_raw_row_decodes

/-- info: 'Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270.SelectorRefinement.generatedRowsSatisfied_iff' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms SelectorRefinement.generatedRowsSatisfied_iff

/-- info: 'Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270.SelectorRefinement.withUnitSelectors_satisfies' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms SelectorRefinement.withUnitSelectors_satisfies
