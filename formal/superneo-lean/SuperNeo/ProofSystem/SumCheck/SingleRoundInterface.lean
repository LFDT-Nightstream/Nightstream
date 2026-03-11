import SuperNeo.ProofSystem.SumCheck.SingleRound

/-!
Interface for `SuperNeo.ProofSystem.SumCheck.SingleRound`.

Spec: `specs/ProofSystem/SumCheck/SingleRound.spec.md`
Paper: `./formal/superneo-lean/SuperNeo.pdf.md`
- Definition 6 (The sum-check protocol), lines 352–355
- Section 7.3 (Π_CCS), lines 481–548

This interface file is the typed boundary companion for the implementation.
-/

namespace SuperNeo

namespace ProofSystem.SumCheck.SingleRoundInterface

/-- Canonical implementation module name for this interface. -/
def implementationModule : String := "SuperNeo.ProofSystem.SumCheck.SingleRound"

/-- Canonical paper source used for this module-level interface/spec pair. -/
def paperSource : String := "./formal/superneo-lean/SuperNeo.pdf.md"

/-- Paper sections used to ground this module boundary. -/
def paperAnchors : List String := ["§2.1 Sum-check reduction role", "§7.3 Interactive reduction for CCS"]

/-- Public symbol inventory extracted from the implementation module. -/
def exportedSymbolNames : List String := ["Instance", "Transcript", "RoundConsistent", "InitialRoundConsistent", "Accepted", "ClaimTrue", "accepted_rounds_eq", "accepted_challenges_eq", "accepted_fold_step", "accepted_initial_round", "accepted_round_sum_step", "not_accepted_of_challenge_size_ne", "not_accepted_of_roundpoly_size_ne", "not_accepted_of_bad_round_shape", "not_accepted_of_no_final_oracle_witness", "not_accepted_of_bad_initial_round"]

/-- Assumption/boundary-oriented symbols extracted by naming convention. -/
def boundarySymbolNames : List String := []

/-- Interface inventory marker for the typed surface exposed here. -/
def interfaceScaffoldReady : Prop := True

theorem interfaceScaffoldReady_true : interfaceScaffoldReady := by
  trivial

end ProofSystem.SumCheck.SingleRoundInterface

end SuperNeo
