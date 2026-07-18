import Nightstream.Implementation.R1CS.Artifacts.FPrimeRecursive.PiRlcChallenge.SamplerLayout
import tests.Axioms.Support

/-! Fail-closed dependencies for the active PiRLC sampler layout facade. -/

open Nightstream.Implementation.R1CS.FPrimeRecursivePiRlcChallenge.SamplerLayout

/-- info: 'Nightstream.Implementation.R1CS.FPrimeRecursivePiRlcChallenge.SamplerLayout.structure_check' does not depend on any axioms -/
#guard_msgs in
#audit_axioms structure_check

/-- info: 'Nightstream.Implementation.R1CS.FPrimeRecursivePiRlcChallenge.SamplerLayout.selection_zero_column_eq_tail_first_allocated' does not depend on any axioms -/
#guard_msgs in
#audit_axioms selection_zero_column_eq_tail_first_allocated

/-- info: 'Nightstream.Implementation.R1CS.FPrimeRecursivePiRlcChallenge.SamplerLayout.selection_zero_row_eq_tail_row' does not depend on any axioms -/
#guard_msgs in
#audit_axioms selection_zero_row_eq_tail_row

/-- info: 'Nightstream.Implementation.R1CS.FPrimeRecursivePiRlcChallenge.SamplerLayout.output_columns_match_challenge_wiring' does not depend on any axioms -/
#guard_msgs in
#audit_axioms output_columns_match_challenge_wiring
