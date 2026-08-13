import Nightstream.Implementation.R1CS.Artifacts.FPrimeRecursive.PiRlcChallenge.TranscriptLayout
import tests.Axioms.Support

/-! Fail-closed dependencies for the active PiRLC transcript layout facade. -/

open Nightstream.Implementation.R1CS.FPrimeRecursivePiRlcChallenge.TranscriptLayout

/-- info: 'Nightstream.Implementation.R1CS.FPrimeRecursivePiRlcChallenge.TranscriptLayout.structure_check' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms structure_check

/-- info: 'Nightstream.Implementation.R1CS.FPrimeRecursivePiRlcChallenge.TranscriptLayout.ordered_emissions_cover_owned_ranges' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms ordered_emissions_cover_owned_ranges

/-- info: 'Nightstream.Implementation.R1CS.FPrimeRecursivePiRlcChallenge.TranscriptLayout.calls_have_exact_compact_abi' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms calls_have_exact_compact_abi

/-- info: 'Nightstream.Implementation.R1CS.FPrimeRecursivePiRlcChallenge.TranscriptLayout.field_output_aliases_match_calls' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms field_output_aliases_match_calls

/-- info: 'Nightstream.Implementation.R1CS.FPrimeRecursivePiRlcChallenge.TranscriptLayout.pi_ccs_output_digest_input_columns_eq' does not depend on any axioms -/
#guard_msgs in
#audit_axioms pi_ccs_output_digest_input_columns_eq

/-- info: 'Nightstream.Implementation.R1CS.FPrimeRecursivePiRlcChallenge.TranscriptLayout.compact_call_profile' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms compact_call_profile

/-- info: 'Nightstream.Implementation.R1CS.FPrimeRecursivePiRlcChallenge.TranscriptLayout.field_output_alias_at_formula' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms field_output_alias_at_formula
