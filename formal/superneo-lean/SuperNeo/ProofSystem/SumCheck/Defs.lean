import SuperNeo.ProofSystem.SumCheck.SingleRound
import SuperNeo.ProofSystem.Types
import SuperNeo.ProofSystem.Security
import SuperNeo.Primitives.GoldilocksPrime
import Mathlib
import Init.Data.List.Lemmas
import Init.Data.Rat.Lemmas

/-!
SumCheck instance/acceptance vocabulary and the Lund-style
Schwartz-Zippel soundness-bound statement shapes.
-/

namespace SuperNeo.ProofSystem

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

/-- Soundness-failure event for a fixed instance/transcript. -/
def SoundnessFailureEvent (inst : Instance) (tr : Transcript) : Prop :=
  Accepted inst tr ∧ ¬ ClaimTrue inst

/-- Advantage of soundness failure under the ambient probability model. -/
def SoundnessFailureAdvantage
  (prob : ProbModel)
  (inst : Instance)
  (tr : Transcript) : Rat :=
  prob.Pr (SoundnessFailureEvent inst tr)

/--
Theorem-facing bound shape for soundness-failure advantage against a
security-parameter indexed error function.
-/
def SoundnessFailureAdvantageBound
  (inst : Instance)
  (tr : Transcript)
  (eps : ErrorFn) : Prop :=
  ∀ prob : ProbModel, ∀ n : Nat,
    SoundnessFailureAdvantage prob inst tr ≤ eps n

/-- Paper-facing SumCheck soundness bound `(ℓ·d, |K|)` for an instance. -/
def lundSchwartzZippelSoundnessBound (inst : Instance) : Nat × Nat :=
  SuperNeo.sumcheckLundSoundnessBound inst

/--
Probability model over verifier-coin events.

This is the event space needed for non-scaffold SumCheck soundness games:
events are predicates over sampled verifier challenge arrays.
-/
structure CoinProbModel where
  Pr : (Array F → Prop) → Rat
  prNonneg : ∀ E : Array F → Prop, 0 ≤ Pr E
  prLeOne : ∀ E : Array F → Prop, Pr E ≤ 1
  prFalse : Pr (fun _ => False) = 0
  prMonotone :
    ∀ {E1 E2 : Array F → Prop},
      (∀ coins, E1 coins → E2 coins) → Pr E1 ≤ Pr E2
  prUnionLeAdd :
    ∀ E1 E2 : Array F → Prop,
      Pr (fun coins => E1 coins ∨ E2 coins) ≤ Pr E1 + Pr E2

/--
Canonical full-field challenge domain for concrete SumCheck coin sampling.

This uses all field elements of `F = Fin Goldilocks.q`.
-/
def fullFieldChallengeDomain : List F :=
  List.finRange Goldilocks.q

/-- Full-field product coin space for `m` verifier rounds. -/
def fullFieldCoinSpace : Nat → List (Array F)
  | 0 => [#[]]
  | m + 1 =>
      (fullFieldCoinSpace m).flatMap (fun coins =>
        fullFieldChallengeDomain.map (fun r => coins.push r))

@[simp] theorem fullFieldChallengeDomain_length :
    fullFieldChallengeDomain.length = Goldilocks.q := by
  simp [fullFieldChallengeDomain]

/-- Canonical zero coin-vector used to witness non-emptiness of coin spaces. -/
def zeroCoins : Nat → Array F
  | 0 => #[]
  | m + 1 => (zeroCoins m).push 0

@[simp] theorem zeroCoins_size (m : Nat) : (zeroCoins m).size = m := by
  induction m with
  | zero =>
      simp [zeroCoins]
  | succ m ih =>
      simpa [zeroCoins, ih]

theorem mem_fullFieldCoinSpace_size
  {m : Nat}
  {coins : Array F}
  (hMem : coins ∈ fullFieldCoinSpace m) :
  coins.size = m := by
  induction m generalizing coins with
  | zero =>
      simp [fullFieldCoinSpace] at hMem
      rcases hMem with rfl
      simp
  | succ m ih =>
      simp [fullFieldCoinSpace] at hMem
      rcases hMem with ⟨base, hBaseMem, r, _hRMem, hEq⟩
      rcases hEq with rfl
      simpa [ih hBaseMem]

theorem zeroCoins_mem_fullFieldCoinSpace (m : Nat) :
  zeroCoins m ∈ fullFieldCoinSpace m := by
  induction m with
  | zero =>
      simp [fullFieldCoinSpace, zeroCoins]
  | succ m ih =>
      have hZeroMem : (0 : F) ∈ fullFieldChallengeDomain := by
        simpa [fullFieldChallengeDomain] using (List.mem_finRange (0 : F))
      apply List.mem_flatMap.mpr
      refine ⟨zeroCoins m, ih, ?_⟩
      apply List.mem_map.mpr
      refine ⟨(0 : F), hZeroMem, ?_⟩
      simp [zeroCoins]

theorem fullFieldCoinSpace_length_pos (m : Nat) :
  0 < (fullFieldCoinSpace m).length := by
  have hMem : zeroCoins m ∈ fullFieldCoinSpace m := zeroCoins_mem_fullFieldCoinSpace m
  have hNe : fullFieldCoinSpace m ≠ [] := by
    intro hNil
    simpa [hNil] using hMem
  exact (List.length_pos_iff).2 hNe

end Sumcheck

end SuperNeo.ProofSystem
