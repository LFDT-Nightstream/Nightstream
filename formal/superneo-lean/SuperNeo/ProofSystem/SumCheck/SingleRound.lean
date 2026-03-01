import SuperNeo.SumCheck

namespace SuperNeo.ProofSystem.Sumcheck.SingleRound

abbrev Instance := SuperNeo.SumCheckInstance
abbrev Transcript := SuperNeo.SumCheckTranscript

abbrev RoundConsistent := SuperNeo.SumCheckRoundConsistent
abbrev InitialRoundConsistent := SuperNeo.sumcheckInitialRoundConsistent
abbrev Accepted := SuperNeo.SumCheckAccepted
abbrev ClaimTrue := SuperNeo.SumCheckClaimTrue

theorem accepted_rounds_eq
  {inst : Instance} {tr : Transcript}
  (hAccepted : Accepted inst tr) :
  tr.roundPolys.size = inst.rounds := by
  exact SuperNeo.sumcheckAccepted_rounds_eq hAccepted

theorem accepted_challenges_eq
  {inst : Instance} {tr : Transcript}
  (hAccepted : Accepted inst tr) :
  tr.challenges.size = tr.roundPolys.size := by
  exact SuperNeo.sumcheckAccepted_challenges_eq hAccepted

theorem accepted_fold_step
  {inst : Instance} {tr : Transcript}
  (hAccepted : Accepted inst tr)
  {i : Nat}
  (hi : i + 1 < tr.roundPolys.size) :
  SuperNeo.sumcheckEvalPoly (tr.roundPolys[i]!) (tr.challenges[i]!) =
    tr.roundPolys[i + 1]!.getD 0 0 := by
  exact SuperNeo.sumcheckAccepted_fold_step hAccepted hi

theorem accepted_initial_round
  {inst : Instance} {tr : Transcript}
  (hAccepted : Accepted inst tr) :
  InitialRoundConsistent inst tr := by
  exact SuperNeo.sumcheckAccepted_initial_round hAccepted

theorem accepted_round_sum_step
  {inst : Instance} {tr : Transcript}
  (hAccepted : Accepted inst tr)
  {i : Nat}
  (hi : i + 1 < tr.roundPolys.size) :
  SuperNeo.sumcheckEvalPoly (tr.roundPolys[i + 1]!) 0 +
      SuperNeo.sumcheckEvalPoly (tr.roundPolys[i + 1]!) 1 =
    SuperNeo.sumcheckEvalPoly (tr.roundPolys[i]!) (tr.challenges[i]!) := by
  exact SuperNeo.sumcheckAccepted_round_sum_step hAccepted hi

theorem not_accepted_of_challenge_size_ne
  {inst : Instance} {tr : Transcript}
  (hNe : tr.challenges.size ≠ inst.rounds) :
  ¬ Accepted inst tr := by
  exact SuperNeo.sumcheckAccepted_not_of_challenge_size_ne hNe

theorem not_accepted_of_roundpoly_size_ne
  {inst : Instance} {tr : Transcript}
  (hNe : tr.roundPolys.size ≠ inst.rounds) :
  ¬ Accepted inst tr := by
  exact SuperNeo.sumcheckAccepted_not_of_roundpoly_size_ne hNe

theorem not_accepted_of_bad_round_shape
  {inst : Instance} {tr : Transcript}
  (hBad : ∃ i : Fin tr.roundPolys.size,
    ¬ SuperNeo.sumcheckRoundPolyShape inst tr.roundPolys[i.1]) :
  ¬ Accepted inst tr := by
  exact SuperNeo.sumcheckAccepted_not_of_bad_round_shape hBad

theorem not_accepted_of_bad_final_claim
  {inst : Instance} {tr : Transcript}
  (hBad : ¬ SuperNeo.sumcheckFinalClaimConsistent inst tr) :
  ¬ Accepted inst tr := by
  exact SuperNeo.sumcheckAccepted_not_of_bad_final_claim hBad

theorem not_accepted_of_bad_initial_round
  {inst : Instance} {tr : Transcript}
  (hBad : ¬ InitialRoundConsistent inst tr) :
  ¬ Accepted inst tr := by
  exact SuperNeo.sumcheckAccepted_not_of_bad_initial_round hBad

end SuperNeo.ProofSystem.Sumcheck.SingleRound
