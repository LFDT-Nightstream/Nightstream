import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.CompleteApplicationCertification
import tests.Axioms.Support

/-!
Fail-closed dependency guards for the complete profile-indexed Phase 5
application certificate.
-/

namespace NightstreamTests.Axioms.CompleteApplicationCertification

open NightstreamTests.Axioms
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.CompleteApplicationCertification.allRecipes' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms CompleteApplicationCertification.allRecipes

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.CompleteApplicationCertification.recipe_family_multiplicities' does not depend on any axioms -/
#guard_msgs in
#audit_axioms CompleteApplicationCertification.recipe_family_multiplicities

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.CompleteApplicationCertification.step_call_multiplicities' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms CompleteApplicationCertification.step_call_multiplicities

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.CompleteApplicationCertification.terminal_call_multiplicities' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms CompleteApplicationCertification.terminal_call_multiplicities

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.CompleteApplicationCertification.step_sound' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms CompleteApplicationCertification.step_sound

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.CompleteApplicationCertification.step_complete' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms CompleteApplicationCertification.step_complete

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.CompleteApplicationCertification.stepPhysicalAccepts_iff' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms CompleteApplicationCertification.stepPhysicalAccepts_iff

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.CompleteApplicationCertification.terminal_sound' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms CompleteApplicationCertification.terminal_sound

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.CompleteApplicationCertification.terminal_complete' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms CompleteApplicationCertification.terminal_complete

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.CompleteApplicationCertification.terminalPhysicalAccepts_iff' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms CompleteApplicationCertification.terminalPhysicalAccepts_iff

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.CompleteApplicationCertification.step_obligation10' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms CompleteApplicationCertification.step_obligation10

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.CompleteApplicationCertification.terminal_obligation10' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms CompleteApplicationCertification.terminal_obligation10

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.CompleteApplicationCertification.stepCost_exact' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms CompleteApplicationCertification.stepCost_exact

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.CompleteApplicationCertification.terminalCost_exact' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms CompleteApplicationCertification.terminalCost_exact

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.CompleteApplicationCertification.stepCost_eq_receiptFold' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms CompleteApplicationCertification.stepCost_eq_receiptFold

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.CompleteApplicationCertification.terminalCost_eq_receiptFold' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms CompleteApplicationCertification.terminalCost_eq_receiptFold

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.CompleteApplicationCertification.step_everyRow_has_exact_owner' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms CompleteApplicationCertification.step_everyRow_has_exact_owner

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.CompleteApplicationCertification.step_everyColumn_has_exact_owner' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms CompleteApplicationCertification.step_everyColumn_has_exact_owner

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.CompleteApplicationCertification.terminal_everyRow_has_exact_owner' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms CompleteApplicationCertification.terminal_everyRow_has_exact_owner

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.CompleteApplicationCertification.terminal_everyColumn_has_exact_owner' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms CompleteApplicationCertification.terminal_everyColumn_has_exact_owner

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.CompleteApplicationCertification.step_rows_conserved' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms CompleteApplicationCertification.step_rows_conserved

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.CompleteApplicationCertification.step_columns_conserved' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms CompleteApplicationCertification.step_columns_conserved

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.CompleteApplicationCertification.terminal_rows_conserved' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms CompleteApplicationCertification.terminal_rows_conserved

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.CompleteApplicationCertification.terminal_columns_conserved' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms CompleteApplicationCertification.terminal_columns_conserved

end NightstreamTests.Axioms.CompleteApplicationCertification
