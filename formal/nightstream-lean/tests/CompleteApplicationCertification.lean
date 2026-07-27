import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.CompleteApplicationCertification

/-!
Focused Phase 5 regressions for the complete profile-indexed fixed-one
application certificate.

The tests pin the eleven-call surface, the two application/setup-owned recipe
positions, exact checker correspondence, and the receipt-derived
cost/ownership boundary.  They do not select a deployment application, Rust
program, or generated row artifact.
-/

set_option autoImplicit false

namespace NightstreamTests.CompleteApplicationCertification

open Nightstream.Implementation.Lowering.FPrimeFixedOne.Vocabulary
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.CompleteApplicationCertification

theorem calls_have_exactly_eleven_entries :
    recipeCalls.length = 11 :=
  rfl

theorem calls_have_no_duplicates :
    recipeCalls.Nodup := by
  decide

theorem step_cannot_be_omitted :
    recipeCalls ≠
      [.iterationZero, .stateEqual, .hashPrior, .hashNext, .freshPublic,
        .encodeInstance, .encodedEqual, .nifsVerify, .runningCheck,
        .freshCheck] := by
  decide

theorem nifs_cannot_be_omitted :
    recipeCalls ≠
      [.iterationZero, .stateEqual, .step, .hashPrior, .hashNext,
        .freshPublic, .encodeInstance, .encodedEqual, .runningCheck,
        .freshCheck] := by
  decide

theorem step_and_nifs_cannot_be_substituted :
    recipeCalls ≠
      [.iterationZero, .stateEqual, .nifsVerify, .hashPrior, .hashNext,
        .freshPublic, .encodeInstance, .encodedEqual, .step,
        .runningCheck, .freshCheck] := by
  decide

/-- The complete family preserves the two proof-carrying application/setup
recipes at their exact typed call positions. -/
theorem phase5_recipe_positions
    {parameters : Parameters}
    (certificate : CompleteApplicationCertification parameters) :
    certificate.allRecipes.recipe Call.step = certificate.phase5.step ∧
      certificate.allRecipes.recipe Call.nifsVerify =
        certificate.phase5.nifsVerify :=
  ⟨certificate.allRecipes_step, certificate.allRecipes_nifsVerify⟩

/-- Phase 3/4 recipes remain distinct positions in the same complete family;
assembly does not replace either terminal relation by NIFS verification. -/
theorem phase34_recipe_positions
    {parameters : Parameters}
    (certificate : CompleteApplicationCertification parameters) :
    certificate.allRecipes.recipe Call.hashPrior =
        certificate.phase34.hashPrior ∧
      certificate.allRecipes.recipe Call.hashNext =
        certificate.phase34.hashNext ∧
      certificate.allRecipes.recipe Call.runningCheck =
        certificate.phase34.runningCheck ∧
      certificate.allRecipes.recipe Call.freshCheck =
        certificate.phase34.freshCheck :=
  ⟨certificate.allRecipes_hashPrior, certificate.allRecipes_hashNext,
    certificate.allRecipes_runningCheck, certificate.allRecipes_freshCheck⟩

#check CompleteApplicationCertification.allRecipes
#check CompleteApplicationCertification.stepProgramCalls_exact
#check CompleteApplicationCertification.terminalProgramCalls_exact
#check CompleteApplicationCertification.step_call_multiplicities
#check CompleteApplicationCertification.terminal_call_multiplicities
#check CompleteApplicationCertification.stepPhysicalAccepts_iff
#check CompleteApplicationCertification.terminalPhysicalAccepts_iff
#check CompleteApplicationCertification.step_obligation10
#check CompleteApplicationCertification.terminal_obligation10
#check CompleteApplicationCertification.stepCost_exact
#check CompleteApplicationCertification.terminalCost_exact
#check CompleteApplicationCertification.stepCost_eq_receiptFold
#check CompleteApplicationCertification.terminalCost_eq_receiptFold
#check CompleteApplicationCertification.step_everyRow_has_exact_owner
#check CompleteApplicationCertification.step_everyColumn_has_exact_owner
#check CompleteApplicationCertification.terminal_everyRow_has_exact_owner
#check CompleteApplicationCertification.terminal_everyColumn_has_exact_owner

end NightstreamTests.CompleteApplicationCertification
