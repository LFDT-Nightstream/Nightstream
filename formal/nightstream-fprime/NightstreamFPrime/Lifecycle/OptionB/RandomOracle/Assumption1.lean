import NightstreamFPrime.Lifecycle.OptionB.RandomOracle.RomModel
import NightstreamFPrime.Spec.HyperNova.ReverseStages

/-! HyperNova errata Assumption 1 (`ReverseStages.Assumption1`) for both option-B folds in the
random-oracle model (`romAssumption1`).

A random-oracle adversary (`RomAdversary`) has finite coins with any law and, for each coin, an
oracle computation with at most `Q` queries. Each attempt's circuit key is one of
`circuitKeys`, because the fork games fix the key. Its NIFS adversary (`RomAdversary.nifs`)
draws a coin and a uniform oracle (`uniform`), and outputs the oracle with the computation's
attempt on it. A NIFS adversary is admitted (`RomAdmitted`) when it is the NIFS adversary of
such an adversary.

The extractor (`extractor`) is an algorithm with its own coins: `budget` fresh oracles, drawn
independently of the tape (`RomEfficient`). It reads the attempt's circuit and key. On a light
attempt it runs the budgeted search of `Budgeted` on the light game of that key, with the
adversary restricted to the light part (`RomModel.restrictLight`), and likewise on a refresh
attempt. The search is a `Rewinding` run of `budget` steps with one rerun per step, so it reruns
the adversary at most `budget` times (`Budgeted.runs_le`). `RomEfficient` states only the coin
law; Lean has no running-time model.

Per game, `Budgeted.failure_le` turns the ideal knowledge bounds
(`LightKnowledge.knowledge_error_le`, `RefreshKnowledge.knowledge_error_le`) into bounds on the
joint event "succeeds and the budgeted answer does not extract" (`light_failure_le`,
`refresh_failure_le`). The loss over the ideal extractor is the expected number of reruns over
`budget`: `(Q + 1)/budget` for the light game and `17 · (Q + 17)/budget` for the refresh game.
A failure of the NIFS extractor is a failure in the game of the attempt's circuit and key
(`covered`), so the error (`errorOf`, `romError`) is the sum over the two keys of both games'
knowledge errors, averaged over the coins and the other oracle part, plus
`2 · ((Q + 1) + 17 · (Q + 17)) / budget` (`truncation`).

This theorem gives a value of `error` for which the premise holds in the random-oracle model.
It does not prove the plain-model premise that the step circuits rely on: they recompute the
Poseidon2 challenges in the circuit (`SECURITY_MODEL.md`, item 3). -/

set_option autoImplicit false

namespace NightstreamFPrime.Lifecycle.OptionB.RandomOracle.Assumption1

open scoped BigOperators ENNReal
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.RandomOracle
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.HyperNova.Construction2
open NightstreamFPrime.Spec.HyperNova.ReverseStages
  (NifsAdversary NifsExtractor ExtractionFails event_ne_top)
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Lifecycle.OptionB
open NightstreamFPrime.Lifecycle.RandomOracle
open NightstreamFPrime.Lifecycle.OptionB.RandomOracle.ForkLink (Attempt)
open NightstreamFPrime.Lifecycle.OptionB.RandomOracle.RomModel

attribute [local instance low] Classical.propDecidable

section Laws

/-- The uniform law on a finite type. -/
noncomputable def uniform (α : Type) [Fintype α] [Nonempty α] : PMF α :=
  PMF.ofFintype (fun _ => ((Fintype.card α : ℝ≥0∞))⁻¹) (by
    rw [Finset.sum_const, Finset.card_univ, nsmul_eq_mul]
    exact ENNReal.mul_inv_cancel (by exact_mod_cast Fintype.card_ne_zero)
      (ENNReal.natCast_ne_top _))

theorem toReal_bind {α β : Type} [Fintype α] (law : PMF α) (next : α → PMF β) (event : Set β) :
    ((law.bind next).toOuterMeasure event).toReal =
      ∑ value, (law value).toReal * ((next value).toOuterMeasure event).toReal := by
  rw [PMF.toOuterMeasure_bind_apply, tsum_fintype, ENNReal.toReal_sum
    fun value _ => ENNReal.mul_ne_top (law.apply_ne_top value) (event_ne_top _ _)]
  simp only [ENNReal.toReal_mul]

theorem toReal_uniform_bind {α β : Type} [Fintype α] [Nonempty α] (next : α → PMF β)
    (event : Set β) :
    (((uniform α).bind next).toOuterMeasure event).toReal =
      𝔼 value, ((next value).toOuterMeasure event).toReal := by
  rw [toReal_bind, Finset.expect_eq_sum_div_card, Finset.card_univ, Finset.sum_div]
  refine Finset.sum_congr rfl fun value _ => ?_
  simp only [uniform, PMF.ofFintype_apply, ENNReal.toReal_inv, ENNReal.toReal_natCast]
  ring

theorem toReal_uniform {α : Type} [Fintype α] [Nonempty α] (event : Set α) :
    ((uniform α).toOuterMeasure event).toReal = 𝔼 value, if value ∈ event then 1 else 0 := by
  rw [← PMF.bind_pure (uniform α), toReal_uniform_bind]
  refine Finset.expect_congr rfl fun value _ => ?_
  rw [PMF.toOuterMeasure_pure_apply]
  split <;> simp

theorem toReal_sum_eq_one {α : Type} [Fintype α] (law : PMF α) : ∑ value, (law value).toReal = 1 := by
  have total := law.tsum_coe
  rw [tsum_fintype] at total
  rw [← ENNReal.toReal_sum fun value _ => law.apply_ne_top value, total, ENNReal.toReal_one]

theorem expect_pair {α β : Type} [Fintype α] [Fintype β] (value : α × β → ℝ) :
    𝔼 pair, value pair = 𝔼 second, 𝔼 first, value (first, second) :=
  (Finset.expect_product Finset.univ Finset.univ value).trans (Finset.expect_comm _ _ _)

end Laws

variable {logicalWidth : Nat}
  {publicFits : ringDegree * publicRingColumns ≤ Phi81CarrierLayout.carrierWidth logicalWidth}

noncomputable local instance oracles : Fintype (Oracle logicalWidth publicFits) :=
  instFintypeProd _ _

noncomputable local instance tapes (budget : Nat) :
    Fintype (Fin budget → Oracle logicalWidth publicFits) :=
  Pi.instFintype

/-- A sample's light part answers uniformly at each light point. -/
theorem uniform_light (point : LightPoint logicalWidth publicFits) (value : Test.Answer → ℝ) :
    𝔼 sample : Oracle logicalWidth publicFits, value (sample.1 point) = 𝔼 answer, value answer := by
  rw [expect_pair]
  dsimp only
  rw [Fintype.expect_const]
  exact expect_eval point value

/-- A sample's refresh part answers uniformly at each refresh point. -/
theorem uniform_refresh (point : RefreshPoint logicalWidth publicFits) (value : Test.Answer → ℝ) :
    𝔼 sample : Oracle logicalWidth publicFits, value (sample.2 point) = 𝔼 answer, value answer := by
  rw [expect_pair]
  dsimp only
  simp only [Fintype.expect_const]
  exact expect_eval point value

section Adversary

/-- A random-oracle adversary against the option-B folds: finite coins with any law and, for each
coin, an oracle computation with at most `queries` queries whose attempts have one of the circuit
keys `circuitKeys`. -/
structure RomAdversary (circuitKeys : Fin 2 → CircuitKey logicalWidth publicFits)
    (queries : Nat) where
  Coins : Type
  [coinsFinite : Fintype Coins]
  coins : PMF Coins
  run : Coins → OracleComp (Point logicalWidth publicFits) Test.Answer
    (Attempt logicalWidth publicFits)
  bounded : ∀ coin, (run coin).QueryBound queries
  keyed : ∀ coin oracle, ∃ slot,
    ((run coin).run (answers oracle)).link.fold.key = circuitKeys slot

attribute [instance] RomAdversary.coinsFinite

variable {circuitKeys : Fin 2 → CircuitKey logicalWidth publicFits} {queries : Nat}

/-- The NIFS adversary of a random-oracle adversary: the tape is a coin and a uniform oracle, and
the attempt is that oracle with the computation's output on it. -/
noncomputable def RomAdversary.nifs (adversary : RomAdversary circuitKeys queries) :
    NifsAdversary (RomAttempt logicalWidth publicFits) where
  Tape := adversary.Coins × Oracle logicalWidth publicFits
  tape := adversary.coins.bind fun coin =>
    (uniform (Oracle logicalWidth publicFits)).map (coin, ·)
  run tape := some (tape.2, (adversary.run tape.1).run (answers tape.2))

variable (circuitKeys queries) in
/-- The NIFS adversary is the NIFS adversary of a random-oracle adversary with at most `queries`
queries and circuit keys `circuitKeys`. -/
def RomAdmitted (adversary : NifsAdversary (RomAttempt logicalWidth publicFits)) : Prop :=
  ∃ rom : RomAdversary circuitKeys queries, rom.nifs = adversary

/-- The extractor's coins are `budget` uniform oracles, drawn independently of the tape
(`NifsExtractor.law`). -/
def RomEfficient (budget : Nat) (adversary : NifsAdversary (RomAttempt logicalWidth publicFits))
    (extractor : NifsExtractor (Answer logicalWidth publicFits) adversary) : Prop :=
  ∃ coins : extractor.Coins ≃ (Fin budget → Oracle logicalWidth publicFits),
    extractor.coins.map coins = uniform (Fin budget → Oracle logicalWidth publicFits)

end Adversary

section Extractor

variable (ajtai : AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))

/-- The answer when no retry family extracts: the attempt's own witness twice. -/
def fallback (attempt : Attempt logicalWidth publicFits) : Answer logicalWidth publicFits :=
  (attempt.witness, attempt.witness)

/-- The light answer of a retry family: the extracted witness of a valid pair. -/
noncomputable def lightAnswer (key : CircuitKey logicalWidth publicFits)
    (adversary : OracleComp (LightPoint logicalWidth publicFits) Test.Answer
      (Attempt logicalWidth publicFits))
    (oracle : LightFork.Oracle logicalWidth publicFits)
    (retries : Fin (LightSchedule.schedule logicalWidth publicFits).rhoCount → Test.Answer) :
    Answer logicalWidth publicFits :=
  if valid : Uniqueness.Valid (LightFork.game ajtai key) adversary oracle retries then
    LightUniqueness.answerOf (LightFork.extractedWitness ajtai valid.1 valid.2)
  else fallback (adversary.run oracle)

/-- The refresh answer of a retry family: the extracted witness of a valid pair. -/
noncomputable def refreshAnswer (key : CircuitKey logicalWidth publicFits)
    (adversary : OracleComp (RefreshPoint logicalWidth publicFits) Test.Answer
      (Attempt logicalWidth publicFits))
    (oracle : RefreshFork.Oracle logicalWidth publicFits)
    (retries : Fin (RefreshSchedule.schedule logicalWidth publicFits).rhoCount → Test.Answer) :
    Answer logicalWidth publicFits :=
  if valid : Uniqueness.Valid (RefreshFork.game ajtai key) adversary oracle retries then
    RefreshUniqueness.answerOf ajtai key.relation (RefreshFork.extractedWitness ajtai valid.1 valid.2)
  else fallback (adversary.run oracle)

/-- The budgeted light extractor: samples read at light points. -/
noncomputable def lightExtract (key : CircuitKey logicalWidth publicFits)
    (adversary : OracleComp (LightPoint logicalWidth publicFits) Test.Answer
      (Attempt logicalWidth publicFits))
    (budget : Nat) (oracle : LightFork.Oracle logicalWidth publicFits)
    (tape : Fin budget → Oracle logicalWidth publicFits) : Answer logicalWidth publicFits :=
  Budgeted.extract (LightFork.game ajtai key) adversary
    (fun sample : Oracle logicalWidth publicFits => sample.1) (lightAnswer ajtai key adversary)
    (fun light => fallback (adversary.run light)) budget oracle tape

/-- The budgeted refresh extractor: samples read at refresh points. -/
noncomputable def refreshExtract (key : CircuitKey logicalWidth publicFits)
    (adversary : OracleComp (RefreshPoint logicalWidth publicFits) Test.Answer
      (Attempt logicalWidth publicFits))
    (budget : Nat) (oracle : RefreshFork.Oracle logicalWidth publicFits)
    (tape : Fin budget → Oracle logicalWidth publicFits) : Answer logicalWidth publicFits :=
  Budgeted.extract (RefreshFork.game ajtai key) adversary
    (fun sample : Oracle logicalWidth publicFits => sample.2) (refreshAnswer ajtai key adversary)
    (fun refresh => fallback (adversary.run refresh)) budget oracle tape

/-- The extractor's answer: the budgeted extractor of the game of the attempt's circuit and key,
on the adversary restricted to that game's oracle part. -/
noncomputable def respond (budget : Nat)
    (computation : OracleComp (Point logicalWidth publicFits) Test.Answer
      (Attempt logicalWidth publicFits))
    (oracle : Oracle logicalWidth publicFits) (tape : Fin budget → Oracle logicalWidth publicFits) :
    Answer logicalWidth publicFits :=
  if BoundedDepth.circuitStep (computation.run (answers oracle)).link.fold.circuit = .light then
    lightExtract ajtai (computation.run (answers oracle)).link.fold.key
      (restrictLight oracle.2 computation) budget oracle.1 tape
  else
    refreshExtract ajtai (computation.run (answers oracle)).link.fold.key
      (restrictRefresh oracle.1 computation) budget oracle.2 tape

variable {circuitKeys : Fin 2 → CircuitKey logicalWidth publicFits} {queries : Nat}

/-- The NIFS extractor: coins are `budget` uniform oracles, and the answer is `respond`. -/
noncomputable def extractor (rom : RomAdversary circuitKeys queries) (budget : Nat) :
    NifsExtractor (Answer logicalWidth publicFits) rom.nifs where
  Coins := Fin budget → Oracle logicalWidth publicFits
  coins := uniform (Fin budget → Oracle logicalWidth publicFits)
  run tape coins := respond ajtai budget (rom.run tape.1) tape.2 coins

end Extractor

section Bounds

variable (ajtai : AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))

/-- `LightKnowledge.knowledge_error_le` for the budgeted light extractor: a successful light run
whose budgeted answer does not extract has probability at most the light knowledge error plus
`(Q + 1)/budget`. -/
theorem light_failure_le (key : CircuitKey logicalWidth publicFits)
    (adversary : OracleComp (LightPoint logicalWidth publicFits) Test.Answer
      (Attempt logicalWidth publicFits))
    {queries : Nat} (bounded : adversary.QueryBound queries) (budget : Nat)
    (positive : 0 < budget) :
    𝔼 oracle, 𝔼 tape : Fin budget → Oracle logicalWidth publicFits,
        (if Fork.Succeeds (LightFork.game ajtai key) adversary oracle ∧
            ¬ UniquenessLink.Extracted ajtai (adversary.run oracle)
              (lightExtract ajtai key adversary budget oracle tape) then (1 : ℝ) else 0) ≤
      LightKnowledge.knowledgeError ajtai key adversary queries +
        ((queries + 1 : Nat) : ℝ) / budget := by
  refine (Budgeted.failure_le (LightFork.game ajtai key) adversary
    (fun sample : Oracle logicalWidth publicFits => sample.1)
    (LightKnowledge.Extracts ajtai key adversary) uniform_light budget positive
    (UniquenessLink.Extracted ajtai) (lightAnswer ajtai key adversary)
    (fun light => fallback (adversary.run light)) (fun extracts => ?_)
    (fun extracts => extracts.fst.1) _ (LightKnowledge.knowledge_error_le ajtai key adversary bounded)
    bounded).trans (le_of_eq ?_)
  · obtain ⟨valid, holds⟩ := extracts
    unfold lightAnswer
    rw [dite_eq_left valid]
    exact holds
  · show _ + ((1 : Nat) : ℝ) * ((queries + 1 : Nat) : ℝ) / budget = _
    rw [Nat.cast_one, one_mul]

/-- `RefreshKnowledge.knowledge_error_le` for the budgeted refresh extractor: a successful
refresh run whose budgeted answer does not extract has probability at most the refresh knowledge
error plus `17 · (Q + 17)/budget`. -/
theorem refresh_failure_le (key : CircuitKey logicalWidth publicFits)
    (adversary : OracleComp (RefreshPoint logicalWidth publicFits) Test.Answer
      (Attempt logicalWidth publicFits))
    {queries : Nat} (bounded : adversary.QueryBound queries) (budget : Nat)
    (positive : 0 < budget) :
    𝔼 oracle, 𝔼 tape : Fin budget → Oracle logicalWidth publicFits,
        (if Fork.Succeeds (RefreshFork.game ajtai key) adversary oracle ∧
            ¬ UniquenessLink.Extracted ajtai (adversary.run oracle)
              (refreshExtract ajtai key adversary budget oracle tape) then (1 : ℝ) else 0) ≤
      RefreshKnowledge.knowledgeError ajtai key adversary queries +
        17 * ((queries + 17 : Nat) : ℝ) / budget := by
  refine (Budgeted.failure_le (RefreshFork.game ajtai key) adversary
    (fun sample : Oracle logicalWidth publicFits => sample.2)
    (RefreshKnowledge.Extracts ajtai key adversary) uniform_refresh budget positive
    (UniquenessLink.Extracted ajtai) (refreshAnswer ajtai key adversary)
    (fun refresh => fallback (adversary.run refresh)) (fun extracts => ?_)
    (fun extracts => extracts.fst.1) _
    (RefreshKnowledge.knowledge_error_le ajtai key adversary bounded) bounded).trans (le_of_eq ?_)
  · obtain ⟨valid, holds⟩ := extracts
    unfold refreshAnswer
    rw [dite_eq_left valid]
    exact holds
  · show _ + ((17 : Nat) : ℝ) * ((queries + 17 : Nat) : ℝ) / budget = _
    rw [Nat.cast_ofNat]

variable {circuitKeys : Fin 2 → CircuitKey logicalWidth publicFits} {queries : Nat}

/-- The light knowledge error of the light game with key `circuitKeys slot`, averaged over the
coins and the refresh part. -/
noncomputable def lightError (rom : RomAdversary circuitKeys queries) (slot : Fin 2) : ℝ :=
  ∑ coin, (rom.coins coin).toReal * 𝔼 refresh : RefreshFork.Oracle logicalWidth publicFits,
    LightKnowledge.knowledgeError ajtai (circuitKeys slot) (restrictLight refresh (rom.run coin))
      queries

/-- The refresh knowledge error of the refresh game with key `circuitKeys slot`, averaged over
the coins and the light part. -/
noncomputable def refreshError (rom : RomAdversary circuitKeys queries) (slot : Fin 2) : ℝ :=
  ∑ coin, (rom.coins coin).toReal * 𝔼 light : LightFork.Oracle logicalWidth publicFits,
    RefreshKnowledge.knowledgeError ajtai (circuitKeys slot) (restrictRefresh light (rom.run coin))
      queries

variable (queries) in
/-- The budget loss: the expected reruns of the light and refresh games, `(Q + 1)` and
`17 · (Q + 17)`, for each of the two keys, over `budget`. -/
noncomputable def truncation (budget : Nat) : ℝ :=
  2 * (((queries + 1 : Nat) : ℝ) + 17 * ((queries + 17 : Nat) : ℝ)) / budget

/-- The Assumption 1 error of a random-oracle adversary: both games' knowledge errors for both
keys, and the budget loss. -/
noncomputable def errorOf (rom : RomAdversary circuitKeys queries) (budget : Nat) : ℝ :=
  ∑ slot, (lightError ajtai rom slot + refreshError ajtai rom slot) + truncation queries budget

variable (circuitKeys queries) in
/-- The Assumption 1 error of an admitted NIFS adversary: `errorOf` of the random-oracle adversary
that `Classical.choose` picks for it. -/
noncomputable def romError (budget : Nat)
    (adversary : NifsAdversary (RomAttempt logicalWidth publicFits)) : ℝ :=
  if admitted : RomAdmitted circuitKeys queries adversary then
    errorOf ajtai (Classical.choose admitted) budget
  else 0

theorem law_toReal (rom : RomAdversary circuitKeys queries) (budget : Nat)
    (event : Set ((rom.Coins × Oracle logicalWidth publicFits) ×
      (Fin budget → Oracle logicalWidth publicFits))) :
    ((extractor ajtai rom budget).law.toOuterMeasure event).toReal =
      ∑ coin, (rom.coins coin).toReal * 𝔼 oracle, 𝔼 tape : Fin budget → Oracle logicalWidth publicFits,
        (if ((coin, oracle), tape) ∈ event then 1 else 0) := by
  unfold NifsExtractor.law
  change (((rom.coins.bind fun coin => (uniform (Oracle logicalWidth publicFits)).map (coin, ·)).bind
      fun tape => (uniform (Fin budget → Oracle logicalWidth publicFits)).map (tape, ·)).toOuterMeasure
      event).toReal = _
  rw [PMF.bind_bind, toReal_bind]
  refine Finset.sum_congr rfl fun coin _ => congrArg _ ?_
  rw [PMF.bind_map, toReal_uniform_bind]
  refine Finset.expect_congr rfl fun oracle _ => ?_
  rw [Function.comp_apply, PMF.toOuterMeasure_map_apply, toReal_uniform]
  rfl

/-- A failure of the extractor is a failure in the light or refresh game of the attempt's key. -/
theorem covered (rom : RomAdversary circuitKeys queries) (budget : Nat)
    (coin : rom.Coins) (oracle : Oracle logicalWidth publicFits)
    (tape : Fin budget → Oracle logicalWidth publicFits)
    (fails : ExtractionFails (Succeeds ajtai) (Extracted ajtai) rom.nifs (extractor ajtai rom budget)
      ((coin, oracle), tape)) :
    ∃ slot,
      (Fork.Succeeds (LightFork.game ajtai (circuitKeys slot)) (restrictLight oracle.2 (rom.run coin))
          oracle.1 ∧
        ¬ UniquenessLink.Extracted ajtai ((restrictLight oracle.2 (rom.run coin)).run oracle.1)
          (lightExtract ajtai (circuitKeys slot) (restrictLight oracle.2 (rom.run coin)) budget
            oracle.1 tape)) ∨
      (Fork.Succeeds (RefreshFork.game ajtai (circuitKeys slot))
          (restrictRefresh oracle.1 (rom.run coin)) oracle.2 ∧
        ¬ UniquenessLink.Extracted ajtai ((restrictRefresh oracle.1 (rom.run coin)).run oracle.2)
          (refreshExtract ajtai (circuitKeys slot) (restrictRefresh oracle.1 (rom.run coin)) budget
            oracle.2 tape)) := by
  obtain ⟨attempt, runEq, succeeds, failed⟩ := fails
  have attemptEq : attempt = (oracle, (rom.run coin).run (answers oracle)) :=
    (Option.some.inj runEq).symm
  subst attemptEq
  obtain ⟨slot, keyEq⟩ := rom.keyed coin oracle
  refine ⟨slot, ?_⟩
  by_cases light : BoundedDepth.circuitStep ((rom.run coin).run (answers oracle)).link.fold.circuit =
      .light
  · have answerEq : (extractor ajtai rom budget).run (coin, oracle) tape =
        lightExtract ajtai ((rom.run coin).run (answers oracle)).link.fold.key
          (restrictLight oracle.2 (rom.run coin)) budget oracle.1 tape := by
      show respond ajtai budget (rom.run coin) oracle tape = _
      unfold respond
      rw [ite_eq_left light]
    have game := light_succeeds ajtai succeeds light
    rw [keyEq] at game answerEq
    refine Or.inl ⟨game, ?_⟩
    rw [run_restrictLight, ← answerEq]
    exact failed
  · have refresh : BoundedDepth.circuitStep
        ((rom.run coin).run (answers oracle)).link.fold.circuit = .refresh := by
      cases step : BoundedDepth.circuitStep ((rom.run coin).run (answers oracle)).link.fold.circuit
      · exact absurd step light
      · rfl
    have answerEq : (extractor ajtai rom budget).run (coin, oracle) tape =
        refreshExtract ajtai ((rom.run coin).run (answers oracle)).link.fold.key
          (restrictRefresh oracle.1 (rom.run coin)) budget oracle.2 tape := by
      show respond ajtai budget (rom.run coin) oracle tape = _
      unfold respond
      rw [ite_eq_right light]
    have game := refresh_succeeds ajtai succeeds refresh
    rw [keyEq] at game answerEq
    refine Or.inr ⟨game, ?_⟩
    rw [run_restrictRefresh, ← answerEq]
    exact failed

/-- The indicator of a light-game failure with key `circuitKeys slot`. -/
noncomputable def lightFailure (rom : RomAdversary circuitKeys queries) (budget : Nat)
    (slot : Fin 2) (coin : rom.Coins) (oracle : Oracle logicalWidth publicFits)
    (tape : Fin budget → Oracle logicalWidth publicFits) : ℝ :=
  if Fork.Succeeds (LightFork.game ajtai (circuitKeys slot)) (restrictLight oracle.2 (rom.run coin))
        oracle.1 ∧
      ¬ UniquenessLink.Extracted ajtai ((restrictLight oracle.2 (rom.run coin)).run oracle.1)
        (lightExtract ajtai (circuitKeys slot) (restrictLight oracle.2 (rom.run coin)) budget
          oracle.1 tape) then 1 else 0

/-- The indicator of a refresh-game failure with key `circuitKeys slot`. -/
noncomputable def refreshFailure (rom : RomAdversary circuitKeys queries) (budget : Nat)
    (slot : Fin 2) (coin : rom.Coins) (oracle : Oracle logicalWidth publicFits)
    (tape : Fin budget → Oracle logicalWidth publicFits) : ℝ :=
  if Fork.Succeeds (RefreshFork.game ajtai (circuitKeys slot))
        (restrictRefresh oracle.1 (rom.run coin)) oracle.2 ∧
      ¬ UniquenessLink.Extracted ajtai ((restrictRefresh oracle.1 (rom.run coin)).run oracle.2)
        (refreshExtract ajtai (circuitKeys slot) (restrictRefresh oracle.1 (rom.run coin)) budget
          oracle.2 tape) then 1 else 0

theorem lightFailure_expect_le (rom : RomAdversary circuitKeys queries) (budget : Nat)
    (positive : 0 < budget) (slot : Fin 2) (coin : rom.Coins) :
    𝔼 oracle, 𝔼 tape : Fin budget → Oracle logicalWidth publicFits,
        lightFailure ajtai rom budget slot coin oracle tape ≤
      𝔼 refresh : RefreshFork.Oracle logicalWidth publicFits,
          LightKnowledge.knowledgeError ajtai (circuitKeys slot)
            (restrictLight refresh (rom.run coin)) queries +
        ((queries + 1 : Nat) : ℝ) / budget := by
  calc
    _ = 𝔼 refresh : RefreshFork.Oracle logicalWidth publicFits,
          𝔼 light : LightFork.Oracle logicalWidth publicFits,
            𝔼 tape : Fin budget → Oracle logicalWidth publicFits,
              lightFailure ajtai rom budget slot coin (light, refresh) tape := by
      rw [expect_pair]
    _ ≤ 𝔼 refresh : RefreshFork.Oracle logicalWidth publicFits,
          (LightKnowledge.knowledgeError ajtai (circuitKeys slot)
            (restrictLight refresh (rom.run coin)) queries +
              ((queries + 1 : Nat) : ℝ) / budget) :=
      Finset.expect_le_expect fun refresh _ =>
        light_failure_le ajtai (circuitKeys slot) (restrictLight refresh (rom.run coin))
          (restrictLight_bound (rom.bounded coin) refresh) budget positive
    _ = _ := by rw [Finset.expect_add_distrib, Fintype.expect_const]

theorem refreshFailure_expect_le (rom : RomAdversary circuitKeys queries) (budget : Nat)
    (positive : 0 < budget) (slot : Fin 2) (coin : rom.Coins) :
    𝔼 oracle, 𝔼 tape : Fin budget → Oracle logicalWidth publicFits,
        refreshFailure ajtai rom budget slot coin oracle tape ≤
      𝔼 light : LightFork.Oracle logicalWidth publicFits,
          RefreshKnowledge.knowledgeError ajtai (circuitKeys slot)
            (restrictRefresh light (rom.run coin)) queries +
        17 * ((queries + 17 : Nat) : ℝ) / budget := by
  calc
    _ = 𝔼 light : LightFork.Oracle logicalWidth publicFits,
          𝔼 refresh : RefreshFork.Oracle logicalWidth publicFits,
            𝔼 tape : Fin budget → Oracle logicalWidth publicFits,
              refreshFailure ajtai rom budget slot coin (light, refresh) tape := by
      rw [expect_pair, Finset.expect_comm]
    _ ≤ 𝔼 light : LightFork.Oracle logicalWidth publicFits,
          (RefreshKnowledge.knowledgeError ajtai (circuitKeys slot)
            (restrictRefresh light (rom.run coin)) queries +
              17 * ((queries + 17 : Nat) : ℝ) / budget) :=
      Finset.expect_le_expect fun light _ =>
        refresh_failure_le ajtai (circuitKeys slot) (restrictRefresh light (rom.run coin))
          (restrictRefresh_bound (rom.bounded coin) light) budget positive
    _ = _ := by rw [Finset.expect_add_distrib, Fintype.expect_const]

/-- The extractor of a random-oracle adversary fails with probability at most `errorOf`. -/
theorem extraction_failure_le (rom : RomAdversary circuitKeys queries) (budget : Nat)
    (positive : 0 < budget) :
    ((extractor ajtai rom budget).law.toOuterMeasure
        {draw | ExtractionFails (Succeeds ajtai) (Extracted ajtai) rom.nifs
          (extractor ajtai rom budget) draw}).toReal ≤
      errorOf ajtai rom budget := by
  have indicator (coin : rom.Coins) (oracle : Oracle logicalWidth publicFits)
      (tape : Fin budget → Oracle logicalWidth publicFits) :
      (if ((coin, oracle), tape) ∈ {draw | ExtractionFails (Succeeds ajtai) (Extracted ajtai)
          rom.nifs (extractor ajtai rom budget) draw} then (1 : ℝ) else 0) ≤
        ∑ slot, (lightFailure ajtai rom budget slot coin oracle tape +
          refreshFailure ajtai rom budget slot coin oracle tape) := by
    have nonnegative : ∀ slot ∈ Finset.univ, (0 : ℝ) ≤
        lightFailure ajtai rom budget slot coin oracle tape +
          refreshFailure ajtai rom budget slot coin oracle tape := fun slot _ => by
      unfold lightFailure refreshFailure
      split <;> split <;> norm_num
    by_cases fails : ((coin, oracle), tape) ∈ {draw | ExtractionFails (Succeeds ajtai)
        (Extracted ajtai) rom.nifs (extractor ajtai rom budget) draw}
    · rw [ite_eq_left fails]
      obtain ⟨slot, found⟩ := covered ajtai rom budget coin oracle tape fails
      refine le_trans ?_ (Finset.single_le_sum nonnegative (Finset.mem_univ slot))
      rcases found with lightFails | refreshFails
      · have one : lightFailure ajtai rom budget slot coin oracle tape = 1 := ite_eq_left lightFails
        have other : 0 ≤ refreshFailure ajtai rom budget slot coin oracle tape := by
          unfold refreshFailure
          split <;> norm_num
        rw [one]
        linarith
      · have one : refreshFailure ajtai rom budget slot coin oracle tape = 1 :=
          ite_eq_left refreshFails
        have other : 0 ≤ lightFailure ajtai rom budget slot coin oracle tape := by
          unfold lightFailure
          split <;> norm_num
        rw [one]
        linarith
    · rw [ite_eq_right fails]
      exact Finset.sum_nonneg nonnegative
  have perCoin (coin : rom.Coins) :
      𝔼 oracle, 𝔼 tape : Fin budget → Oracle logicalWidth publicFits,
          (if ((coin, oracle), tape) ∈ {draw | ExtractionFails (Succeeds ajtai) (Extracted ajtai)
            rom.nifs (extractor ajtai rom budget) draw} then (1 : ℝ) else 0) ≤
        ∑ slot : Fin 2,
          ((𝔼 refresh : RefreshFork.Oracle logicalWidth publicFits,
              LightKnowledge.knowledgeError ajtai (circuitKeys slot)
                (restrictLight refresh (rom.run coin)) queries +
                ((queries + 1 : Nat) : ℝ) / budget) +
            (𝔼 light : LightFork.Oracle logicalWidth publicFits,
              RefreshKnowledge.knowledgeError ajtai (circuitKeys slot)
                (restrictRefresh light (rom.run coin)) queries +
                17 * ((queries + 17 : Nat) : ℝ) / budget)) := by
    calc
      _ ≤ 𝔼 oracle, 𝔼 tape : Fin budget → Oracle logicalWidth publicFits,
            ∑ slot, (lightFailure ajtai rom budget slot coin oracle tape +
              refreshFailure ajtai rom budget slot coin oracle tape) :=
        Finset.expect_le_expect fun oracle _ => Finset.expect_le_expect fun tape _ =>
          indicator coin oracle tape
      _ = ∑ slot, (𝔼 oracle, 𝔼 tape : Fin budget → Oracle logicalWidth publicFits,
              lightFailure ajtai rom budget slot coin oracle tape +
            𝔼 oracle, 𝔼 tape : Fin budget → Oracle logicalWidth publicFits,
              refreshFailure ajtai rom budget slot coin oracle tape) := by
        simp only [Finset.expect_sum_comm, Finset.expect_add_distrib]
      _ ≤ _ := Finset.sum_le_sum fun slot _ =>
        add_le_add (lightFailure_expect_le ajtai rom budget positive slot coin)
          (refreshFailure_expect_le ajtai rom budget positive slot coin)
  refine (law_toReal ajtai rom budget _).trans_le ?_
  calc
    _ ≤ ∑ coin, (rom.coins coin).toReal * ∑ slot : Fin 2,
          ((𝔼 refresh : RefreshFork.Oracle logicalWidth publicFits,
              LightKnowledge.knowledgeError ajtai (circuitKeys slot)
                (restrictLight refresh (rom.run coin)) queries +
                ((queries + 1 : Nat) : ℝ) / budget) +
            (𝔼 light : LightFork.Oracle logicalWidth publicFits,
              RefreshKnowledge.knowledgeError ajtai (circuitKeys slot)
                (restrictRefresh light (rom.run coin)) queries +
                17 * ((queries + 17 : Nat) : ℝ) / budget)) :=
      Finset.sum_le_sum fun coin _ =>
        mul_le_mul_of_nonneg_left (perCoin coin) ENNReal.toReal_nonneg
    _ = _ := by
      have total := toReal_sum_eq_one rom.coins
      simp only [Fin.sum_univ_two, mul_add, Finset.sum_add_distrib, ← Finset.sum_mul, total,
        errorOf, lightError, refreshError, truncation]
      ring

/-- `extraction_failure_le` for every NIFS adversary that equals `rom.nifs`. -/
theorem assumption_of_rom (rom : RomAdversary circuitKeys queries) (budget : Nat)
    (positive : 0 < budget) (adversary : NifsAdversary (RomAttempt logicalWidth publicFits))
    (same : rom.nifs = adversary) :
    ∃ extractor : NifsExtractor (Answer logicalWidth publicFits) adversary,
      RomEfficient budget adversary extractor ∧
        (extractor.law.toOuterMeasure
          {draw | ExtractionFails (Succeeds ajtai) (Extracted ajtai) adversary extractor draw}).toReal ≤
          errorOf ajtai rom budget := by
  subst same
  exact ⟨extractor ajtai rom budget, ⟨Equiv.refl _, PMF.map_id _⟩,
    extraction_failure_le ajtai rom budget positive⟩

variable (circuitKeys queries) in
/-- HyperNova errata Assumption 1 for both option-B folds in the random-oracle model, with the
budgeted extractor and the error `romError`. -/
theorem romAssumption1 (budget : Nat) (positive : 0 < budget) :
    Spec.HyperNova.ReverseStages.Assumption1 (Succeeds ajtai) (Extracted ajtai)
      (RomAdmitted circuitKeys queries) (RomEfficient budget)
      (romError ajtai circuitKeys queries budget) := by
  intro adversary admitted
  rw [romError, dite_eq_left admitted]
  exact assumption_of_rom ajtai (Classical.choose admitted) budget positive adversary
    (Classical.choose_spec admitted)

end Bounds

end NightstreamFPrime.Lifecycle.OptionB.RandomOracle.Assumption1
