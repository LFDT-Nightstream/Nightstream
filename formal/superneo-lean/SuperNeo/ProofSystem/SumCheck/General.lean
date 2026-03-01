import SuperNeo.ProofSystem.SumCheck.SingleRound
import SuperNeo.ProofSystem.Types
import SuperNeo.ProofSystem.Security

namespace SuperNeo.ProofSystem

/- Paper-facing SumCheck namespace wrappers. -/
namespace Sumcheck

abbrev Instance := SuperNeo.SumCheckInstance
abbrev Transcript := SuperNeo.SumCheckTranscript

abbrev RoundConsistent := SuperNeo.SumCheckRoundConsistent
abbrev InitialRoundConsistent := SuperNeo.sumcheckInitialRoundConsistent
abbrev Accepted := SuperNeo.SumCheckAccepted
abbrev ClaimTrue := SuperNeo.SumCheckClaimTrue

abbrev SoundnessAssumption := SuperNeo.SumcheckSoundnessAssumption
abbrev CompletenessAssumption := SuperNeo.SumcheckCompletenessAssumption
abbrev Assumptions := SuperNeo.SumCheckAssumptions

theorem accepted_rounds_eq
  {inst : Instance} {tr : Transcript}
  (hAccepted : Accepted inst tr) :
  tr.roundPolys.size = inst.rounds := by
  exact SingleRound.accepted_rounds_eq hAccepted

theorem accepted_challenges_eq
  {inst : Instance} {tr : Transcript}
  (hAccepted : Accepted inst tr) :
  tr.challenges.size = tr.roundPolys.size := by
  exact SingleRound.accepted_challenges_eq hAccepted

theorem accepted_fold_step
  {inst : Instance} {tr : Transcript}
  (hAccepted : Accepted inst tr)
  {i : Nat}
  (hi : i + 1 < tr.roundPolys.size) :
  SuperNeo.sumcheckEvalPoly (tr.roundPolys[i]!) (tr.challenges[i]!) =
    tr.roundPolys[i + 1]!.getD 0 0 := by
  exact SingleRound.accepted_fold_step hAccepted hi

theorem accepted_initial_round
  {inst : Instance} {tr : Transcript}
  (hAccepted : Accepted inst tr) :
  InitialRoundConsistent inst tr := by
  exact SingleRound.accepted_initial_round hAccepted

theorem accepted_round_sum_step
  {inst : Instance} {tr : Transcript}
  (hAccepted : Accepted inst tr)
  {i : Nat}
  (hi : i + 1 < tr.roundPolys.size) :
  SuperNeo.sumcheckEvalPoly (tr.roundPolys[i + 1]!) 0 +
      SuperNeo.sumcheckEvalPoly (tr.roundPolys[i + 1]!) 1 =
    SuperNeo.sumcheckEvalPoly (tr.roundPolys[i]!) (tr.challenges[i]!) := by
  exact SingleRound.accepted_round_sum_step hAccepted hi

/-- Soundness theorem surface (assumption-instantiated). -/
theorem soundness
  (h : SoundnessAssumption)
  {inst : Instance} {tr : Transcript}
  (hAccepted : Accepted inst tr) :
  ClaimTrue inst := by
  exact h inst tr hAccepted

/-- Completeness theorem surface (assumption-instantiated). -/
theorem completeness
  (h : CompletenessAssumption)
  {inst : Instance}
  (hClaim : ClaimTrue inst) :
  ∃ tr, Accepted inst tr := by
  exact h inst hClaim

end Sumcheck

end SuperNeo.ProofSystem
