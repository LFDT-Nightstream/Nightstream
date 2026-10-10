import NightstreamFPrime.Export.Stage1.HyperNovaHistoryLaw

/-!
Owns the call count of the reverse history: every generated history calls the
NIFS source extractor at most its advertised iteration count of times,
including aborts and calls after invalid source returns. With a per-call
extractor cost, this bounds the history's extraction work. It is not a
machine-time bound.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.HyperNovaHistoryWork

open scoped BigOperators
open HyperNovaHistory
variable (application : Lifecycle.Stage1.Application.Program)
  (fits : PerApplicationFixedPoint.FitsTwoPow28 application)
  (setup : PerApplicationCanonicalPackage.CommitmentSetup application)

/-- Every generated history makes at most its advertised iteration count of
source calls. No accepted-terminal, valid-source, or no-collision premise is
used. In particular, abort calls and all false-mark paths are included. -/
theorem source_calls_le_iteration
    (source : Statement → Payload application → PMF (SourceResult application))
    (statement : Statement) (proof : Envelope application) (outcomes : List (SourceResult application))
    (supported : outcomes ∈ (HyperNovaHistoryLaw.results application fits source statement proof).support) :
    outcomes.length ≤ statement.iteration := by
  generalize count : statement.iteration = remaining at *
  induction remaining using Nat.strong_induction_on generalizing statement proof outcomes with
  | h remaining induction =>
      rw [HyperNovaHistoryLaw.results_eq] at supported
      cases proof with
      | bottom =>
          have empty := (PMF.mem_support_pure_iff [] outcomes).mp supported
          simp only [empty, List.length_nil, Nat.zero_le]
      | recursive payload =>
          by_cases zero : statement.iteration = 0
          · simp only [if_pos zero] at supported
            have empty := (PMF.mem_support_pure_iff [] outcomes).mp supported
            simp only [empty, List.length_nil, Nat.zero_le]
          · simp only [if_neg zero] at supported
            by_cases counter : (decodedInput application fits payload).iteration + 1 = statement.iteration
            · simp only [if_pos counter] at supported
              by_cases base : (decodedInput application fits payload).iteration = 0
              · simp only [if_pos base] at supported
                have empty := (PMF.mem_support_pure_iff [] outcomes).mp supported
                simp only [empty, List.length_nil, Nat.zero_le]
              · simp only [if_neg base] at supported
                rcases (PMF.mem_support_bind_iff _ _ _).mp supported with
                  ⟨result, _produced, tailSupported⟩
                cases result with
                | none =>
                    have singleton := (PMF.mem_support_pure_iff [none] outcomes).mp tailSupported
                    simp only [singleton, List.length_cons, List.length_nil]
                    omega
                | some values =>
                    rcases (PMF.mem_support_map_iff _ _ _).mp tailSupported with
                      ⟨tail, previousSupported, same⟩
                    have previousCount : (predecessorStatement application fits payload).iteration + 1 = remaining := by
                      simpa only [predecessorStatement] using counter.trans count
                    have previous := induction (predecessorStatement application fits payload).iteration
                      (by omega) (predecessorStatement application fits payload)
                      (.recursive (predecessorPayload application fits payload values)) tail previousSupported rfl
                    rw [← same, List.length_cons]
                    omega
            · simp only [if_neg counter] at supported
              have empty := (PMF.mem_support_pure_iff [] outcomes).mp supported
              simp only [empty, List.length_nil, Nat.zero_le]

end NightstreamFPrime.Export.Stage1.HyperNovaHistoryWork
