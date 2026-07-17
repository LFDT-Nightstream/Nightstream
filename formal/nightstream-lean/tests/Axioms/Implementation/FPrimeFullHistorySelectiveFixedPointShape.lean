import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.FixedPointShape
import tests.Axioms.Support

/-!
Fail-closed dependency guards for the model-level fixed-point shape contract.

| Guard | Audited boundary |
|---|---|
| Stable header | terminal input, predicted output, and emitted verifier header agree exactly |
| Matrix arity | emitted matrix count and polynomial arity both equal the independent thirteen ports |
| Public carrier | verifier-visible public input is exactly 270 fields |
| Semantic shape | the constructed relation profile retains thirteen matrices |
-/

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.FixedPointShape.Refinement.terminalInput_eq_materialized' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.FixedPointShape.Refinement.terminalInput_eq_materialized

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.FixedPointShape.Refinement.materialized_matrixCount_eq_13' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.FixedPointShape.Refinement.materialized_matrixCount_eq_13

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.FixedPointShape.Refinement.materialized_polynomialArity_eq_13' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.FixedPointShape.Refinement.materialized_polynomialArity_eq_13

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.FixedPointShape.Refinement.materialized_publicInputLength_eq_270' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.FixedPointShape.Refinement.materialized_publicInputLength_eq_270

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.FixedPointShape.Refinement.profile_shape_matrixCount_eq_13' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.FixedPointShape.Refinement.profile_shape_matrixCount_eq_13
