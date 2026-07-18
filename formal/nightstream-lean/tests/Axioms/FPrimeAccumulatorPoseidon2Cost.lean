import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.Accumulator.Poseidon2Cost
import tests.Axioms.Support

/-! Fail-closed dependency gate for the reduced accumulator Poseidon2 cost leaf. -/

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistory.Accumulator.Poseidon2Cost.permutation_product_rows_eq' depends on axioms: [Lean.trustCompiler] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistory.Accumulator.Poseidon2Cost.permutation_product_rows_eq

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistory.Accumulator.Poseidon2Cost.sponge_rows_formula' depends on axioms: [propext,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistory.Accumulator.Poseidon2Cost.sponge_rows_formula

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistory.Accumulator.Poseidon2Cost.sponge_fresh_columns_eq_rows' depends on axioms: [propext,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistory.Accumulator.Poseidon2Cost.sponge_fresh_columns_eq_rows

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistory.Accumulator.Poseidon2Cost.commitment_family_hash_rows_formula' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistory.Accumulator.Poseidon2Cost.commitment_family_hash_rows_formula

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistory.Accumulator.Poseidon2Cost.canonical_parent_hash_rows_formula' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistory.Accumulator.Poseidon2Cost.canonical_parent_hash_rows_formula
