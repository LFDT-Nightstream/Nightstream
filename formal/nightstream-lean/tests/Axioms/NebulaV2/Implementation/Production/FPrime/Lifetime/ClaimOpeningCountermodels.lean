import Nightstream.Implementation.NebulaV2.Production.FPrime.Lifetime.ClaimOpeningCountermodels
import tests.Axioms.Support

/-! Fail-closed dependency audit for claim-opening countermodels. -/

/-- info: 'Nightstream.Implementation.NebulaV2.ProductionPaperClaimOpeningCountermodels.detached_authority_exists' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.ProductionPaperClaimOpeningCountermodels.detached_authority_exists

/-- info: 'Nightstream.Implementation.NebulaV2.ProductionPaperClaimOpeningCountermodels.detached_authority_has_no_common_witness' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.ProductionPaperClaimOpeningCountermodels.detached_authority_has_no_common_witness

/-- info: 'Nightstream.Implementation.NebulaV2.ProductionPaperClaimOpeningCountermodels.source_opening_does_not_open_different_receipt' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.ProductionPaperClaimOpeningCountermodels.source_opening_does_not_open_different_receipt

/-- info: 'Nightstream.Implementation.NebulaV2.ProductionPaperClaimOpeningCountermodels.floating_terminal_program_does_not_imply_fixed_program' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.ProductionPaperClaimOpeningCountermodels.floating_terminal_program_does_not_imply_fixed_program
