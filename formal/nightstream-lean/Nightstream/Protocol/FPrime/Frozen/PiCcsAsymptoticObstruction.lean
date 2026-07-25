import Nightstream.Protocol.FPrime.Frozen.Obligations

/-!
Exact frozen-target obstruction for the missing PiCCS unbounded first-success
bridge.

Owns: a minimal `SuperNeoGames` model in which every probability quantity,
composition coupling, and error-budget entry is fixed, while the truth of the
frozen `PiCcsStrong` target is equivalent to an arbitrary runtime proposition.
It also pairs a true frozen target with a concrete retry sequence that never
succeeds.

Does not own: a concrete protocol, a probability law on infinite tapes, a
stopping time, almost-sure termination, expected running time, or an
asymptotic security family.

Emits constraints: no.

| Declaration | Owns | Excluded boundary |
|---|---|---|
| `games` | a complete degenerate frozen game package | no operational sampler or protocol semantics |
| `piCcsStrong_iff_runtime` | exact exposure of the opaque runtime field | no derivation of EPT |
| `frozenTarget_without_samplerLink_countermodel` | a true frozen target paired with a nonterminating retry sequence | no claim about a linked sampler |
| `not_attemptedBridgeWithoutSamplerLink` | falsity of the strongest free-retry bridge expressible without a coupling | no weakening of `PiCcsStrong` |
-/

set_option autoImplicit false

namespace Nightstream.Protocol.FPrime.Frozen.PiCcsAsymptoticObstruction

open Nightstream.Protocol.FPrime.Frozen.Obligations
open Nightstream.SuperNeo.InteractiveReduction.Paper

/-- Degenerate arithmetic isolates the logical runtime field from every
probability coordinate. -/
def unitScale : ProbabilityScale Unit where
  zero := ()
  one := ()
  add := fun _ _ => ()
  subtract := fun weight _ => weight
  le := fun _ _ => True
  le_refl := fun _ => True.intro
  le_trans := fun _ _ => True.intro
  subtract_zero := fun _ => rfl

/-- Strong game with fixed probability data and one caller-selected runtime
proposition. -/
def runtimeOnlyGame (runtime : Prop) : StrongGame Unit Unit Unit where
  perfectComplete := True
  publicCoin := True
  adversaryExpectedPolynomialTime := fun _ => True
  extractorExpectedPolynomialTime := fun _ _ => runtime
  extractionEligible := fun _ => True
  repeatedOutputPhiMismatch := fun _ => ()
  ambientOutputSuccess := fun _ => ()
  repeatedOutputWitnessDisagreement := fun _ => ()
  sourceWitnessExtracted := fun _ _ => ()

/-- The rejection-adjusted target for `runtimeOnlyGame` contains exactly its
opaque runtime proposition. -/
theorem rejectionAdjustedStrong_runtimeOnly_iff (runtime : Prop) :
    RejectionAdjustedStrong unitScale (fun _ _ => ())
      (runtimeOnlyGame runtime) () () () ↔ runtime := by
  constructor
  · rintro ⟨_complete, _publicCoin, _samePhi, extraction⟩
    rcases (extraction () True.intro True.intro).2 True.intro with
      ⟨_extractor, runtimeProof, _bound⟩
    exact runtimeProof
  · intro runtimeProof
    refine ⟨True.intro, True.intro, ?_, ?_⟩
    · intro _adversary _expected
      rfl
    · intro _adversary _expected _eligible
      refine ⟨True.intro, ?_⟩
      intro _uniqueness
      exact ⟨(), runtimeProof, True.intro⟩

/-- Trivial weak game used only to fill fields unrelated to PiCCS. -/
def weakGame : WeakGame Unit Unit Unit Unit where
  perfectComplete := True
  publicCoin := True
  adversaryExpectedPolynomialTime := fun _ => True
  pairedAdversaryExpectedPolynomialTime := fun _ => True
  extractorExpectedPolynomialTime := fun _ _ => True
  extractionEligible := fun _ => True
  adversarySuccess := fun _ => ()
  ambientSourceWitnessExtracted := fun _ _ => ()
  left := fun _ => ()
  right := fun _ => ()
  samePhiInputsAlways := fun _ => True
  pairedWitnessDisagreement := fun _ _ _ => ()

/-- Trivial final-stage knowledge game used only to fill unrelated fields. -/
def piDecGame : KnowledgeGame Unit Unit Unit where
  perfectComplete := True
  publicCoin := True
  adversaryExpectedPolynomialTime := fun _ => True
  extractorExpectedPolynomialTime := fun _ _ => True
  extractionEligible := fun _ => True
  adversarySuccess := fun _ => ()
  sourceWitnessExtracted := fun _ _ => ()

/-- Exact unit-valued strong--weak coupling. -/
def strongWeakCoupling (runtime : Prop) :
    Nightstream.SuperNeo.InteractiveReduction.StrongWeakComposition.Coupling
      unitScale (runtimeOnlyGame runtime) weakGame Unit where
  toWeak := fun _ => ()
  toStrong := fun _ _ => ()
  paired := fun _ _ => ()
  pairedLeft := by
    intro _ _
    rfl
  pairedRight := by
    intro _ _
    rfl
  pairedExpectedPolynomialTime := by
    intro _ _ _ _
    trivial
  pairedSamePhi := by
    intro _ _
    trivial
  intermediateProbability := by
    intro _ _
    rfl
  repeatedWitnessProbability := by
    intro _ _
    rfl

/-- Exact unit-valued coupling to the final knowledge game. -/
def piDecCoupling (runtime : Prop) :
    Nightstream.SuperNeo.InteractiveReduction.KnowledgeComposition.Coupling
      unitScale
      (Nightstream.SuperNeo.InteractiveReduction.StrongWeakComposition.knowledgeGame
        unitScale (runtimeOnlyGame runtime) weakGame
        (strongWeakCoupling runtime))
      piDecGame Unit where
  toSecond := fun _ => ()
  toFirst := fun _ _ => ()
  intermediateProbability := by
    intro _ _
    rfl

/-- All abstract subtraction laws are vacuous over the unit scale. -/
def scaleLaws :
    Nightstream.SuperNeo.InteractiveReduction.StrongWeakComposition.ScaleLaws
      unitScale where
  subtract_mono_left := by
    intro _ _ _ _
    trivial
  subtract_subtract := by
    intro _ _ _
    rfl

/-- Unit error budget; only the PiCCS runtime proposition can vary. -/
def errorBudget : InteractiveErrorBudget Unit where
  piCcsSumCheck := ()
  piCcsSchwartzZippel := ()
  piRlcForkSampling := ()
  piCcsSuccessFloor := ()
  relaxedBindingRaw := ()
  adjustUniqueness := fun _ _ => ()

/-- Complete frozen game package whose sole varying datum is the opaque
PiCCS extractor-runtime proposition. -/
def games (runtime : Prop) : SuperNeoGames where
  Weight := Unit
  scale := unitScale

  PiCcsAdversary := Unit
  PiCcsExtractor := Unit
  piCcs := runtimeOnlyGame runtime

  PiRlcAdversary := Unit
  PiRlcPairedAdversary := Unit
  PiRlcExtractor := Unit
  piRlc := weakGame

  StrongWeakAdversary := Unit
  strongWeakCoupling := strongWeakCoupling runtime

  PiDecAdversary := Unit
  PiDecExtractor := Unit
  piDec := piDecGame

  ComposedAdversary := Unit
  piDecCoupling := piDecCoupling runtime

  IntermediateInstance := Unit
  Projection := Unit
  piCcsProjection := fun _ => ()
  piRlcProjection := fun _ => ()

  errorBudget := errorBudget
  scaleLaws := scaleLaws

/-- The exact frozen target contains no derivation of operational EPT: with
all probability and composition data fixed, it is equivalent to the runtime
proposition inserted by the game owner. -/
theorem piCcsStrong_iff_runtime (runtime : Prop) :
    PiCcsStrong (games runtime) ↔ runtime := by
  change RejectionAdjustedStrong unitScale (fun _ _ => ())
    (runtimeOnlyGame runtime) () () () ↔ runtime
  exact rejectionAdjustedStrong_runtimeOnly_iff runtime

/-- Pointwise eventual success for a concrete retry sequence. Almost-sure
termination would be strictly stronger once a probability law is supplied. -/
def EventuallySucceeds (retry : Nat -> Bool) : Prop :=
  ∃ attempt, retry attempt = true

/-- Every retry reuses one hidden coin; this is deliberately not IID. -/
def sharedCoinRetry (hidden : Bool) (_attempt : Nat) : Bool :=
  hidden

theorem sharedCoinRetry_false_has_no_success :
    ¬ EventuallySucceeds (sharedCoinRetry false) := by
  simp [EventuallySucceeds, sharedCoinRetry]

/-- A true frozen target can coexist with an unrelated retry sequence that
fails forever; the missing ingredient is a sampler link, not another
probability inequality. -/
theorem frozenTarget_without_samplerLink_countermodel :
    PiCcsStrong (games True) /\
      ¬ EventuallySucceeds (sharedCoinRetry false) := by
  exact ⟨(piCcsStrong_iff_runtime True).2 True.intro,
    sharedCoinRetry_false_has_no_success⟩

/-- Strongest direct free-retry bridge expressible without a sampler field. -/
def AttemptedBridgeWithoutSamplerLink : Prop :=
  ∀ retry : Nat -> Bool,
    PiCcsStrong (games True) -> EventuallySucceeds retry

theorem not_attemptedBridgeWithoutSamplerLink :
    ¬ AttemptedBridgeWithoutSamplerLink := by
  intro attempted
  exact sharedCoinRetry_false_has_no_success
    (attempted (sharedCoinRetry false)
      ((piCcsStrong_iff_runtime True).2 True.intro))

end Nightstream.Protocol.FPrime.Frozen.PiCcsAsymptoticObstruction
