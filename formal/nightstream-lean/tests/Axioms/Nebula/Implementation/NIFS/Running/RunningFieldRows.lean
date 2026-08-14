import Nightstream.Implementation.Nebula.NIFS.Running.RunningFieldRows
import tests.Axioms.Support

open Nightstream.Implementation.Nebula.ProductNifsRunningFieldRows

/-! Axiom gates for the generated running-claim bit-to-field bridge. -/

/-- info: 'Nightstream.Implementation.Nebula.ProductNifsRunningFieldRows.rows_length_exact' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms rows_length_exact

/-- info: 'Nightstream.Implementation.Nebula.ProductNifsRunningFieldRows.parsed_columns_match' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms parsed_columns_match

/-- info: 'Nightstream.Implementation.Nebula.ProductNifsRunningFieldRows.parse_from_rows' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms parse_from_rows

/-- info: 'Nightstream.Implementation.Nebula.ProductNifsRunningFieldRows.modulus_alias_impossible' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms modulus_alias_impossible

/-- info: 'Nightstream.Implementation.Nebula.ProductNifsRunningParser.parse_success_fields' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductNifsRunningParser.parse_success_fields
