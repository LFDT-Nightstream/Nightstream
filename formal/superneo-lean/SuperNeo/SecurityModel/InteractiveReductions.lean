import SuperNeo.FoldingProtocol.PiDEC
import SuperNeo.ProofSystem.SumCheck

/-!
Composition of reduction steps (`Π_CCS`, `Π_RLC`, `Π_DEC`).
-/

namespace SuperNeo

/-- Assumptions for composed interactive reductions. -/
structure InteractiveReductionAssumptions (ctx : ProtocolTargetContext) where
  reduction : ProtocolTargetAssumptions ctx
  sumcheckTransitionWitness : SumCheckTransitionWitness ctx

/-- Strong composition statement (knowledge-style). -/
def strongCompositionStatement (ctx : ProtocolTargetContext) : Prop :=
  piDECKnowledgeStatement ctx

/-- Weak composition statement (relaxed CE + claim truth). -/
def weakCompositionStatement (ctx : ProtocolTargetContext) : Prop :=
  ceRelaxedRelation ctx ∧
  SumCheckClaimTrue (sumcheckInstanceOfContext ctx)

/-- Strong composed reduction theorem. -/
theorem strongComposition_of_assumptions
  {ctx : ProtocolTargetContext}
  (h : InteractiveReductionAssumptions ctx) :
  strongCompositionStatement ctx := by
  exact piDEC_of_assumptions h.reduction h.sumcheckTransitionWitness

/-- Weak composed reduction theorem. -/
theorem weakComposition_of_assumptions
  {ctx : ProtocolTargetContext}
  (h : InteractiveReductionAssumptions ctx) :
  weakCompositionStatement ctx := by
  rcases strongComposition_of_assumptions h with ⟨_deltaInv, _hMul, hWeak, hClaim⟩
  exact ⟨hWeak, hClaim⟩

/--
Witness-level SumCheck soundness-failure advantage bound derived from reduction
assumptions plus a nonnegative target error function.
-/
theorem sumcheckFailureAdvantageBound_of_assumptions
  {ctx : ProtocolTargetContext}
  (h : InteractiveReductionAssumptions ctx)
  (eps : SuperNeo.ProofSystem.ErrorFn)
  (hEpsNonneg : ∀ n : Nat, 0 ≤ eps n) :
  SuperNeo.ProofSystem.Sumcheck.SoundnessFailureAdvantageBound
      (sumcheckInstanceOfContext ctx)
      h.sumcheckTransitionWitness.transcript
      eps := by
  intro prob n
  have hFailFalse :
      SuperNeo.ProofSystem.Sumcheck.SoundnessFailureEvent
          (sumcheckInstanceOfContext ctx)
          h.sumcheckTransitionWitness.transcript → False := by
    intro hFail
    exact hFail.2 (sumcheckSoundness_constructive
      (sumcheckInstanceOfContext ctx)
      h.sumcheckTransitionWitness.transcript
      hFail.1)
  have hLeZero :
      SuperNeo.ProofSystem.Sumcheck.SoundnessFailureAdvantage
          prob
          (sumcheckInstanceOfContext ctx)
          h.sumcheckTransitionWitness.transcript ≤ 0 := by
    unfold SuperNeo.ProofSystem.Sumcheck.SoundnessFailureAdvantage
    calc
      prob.Pr
            (SuperNeo.ProofSystem.Sumcheck.SoundnessFailureEvent
              (sumcheckInstanceOfContext ctx)
              h.sumcheckTransitionWitness.transcript)
          ≤ prob.Pr False := prob.prMonotone hFailFalse
      _ = 0 := prob.prFalse
  exact Rat.le_trans hLeZero (hEpsNonneg n)

end SuperNeo
