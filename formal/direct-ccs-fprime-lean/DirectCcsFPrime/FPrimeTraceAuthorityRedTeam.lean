import DirectCcsFPrime.AggregateChildTableNecessity
import DirectCcsFPrime.FPrimeTraceAuthority
import Mathlib.Tactic

/-!
Red-team checks for trace-carrying F' authority.

These theorems encode adversarial shortcuts as negated claims. They compile by
proving that digest-only authority, aggregate child summaries, and low-norm
DEC checks cannot replace explicit trace or pointwise child evidence.
-/

namespace DirectCcsFPrime

namespace FPrimeTraceAuthorityRedTeam

/-- Tiny public-image universe for concrete negative tests. -/
inductive TwoImage where
  | initial
  | forged
deriving DecidableEq

/-- No transition is valid in this red-team model. -/
def NoTransition : Nat → TwoImage → TwoImage → Prop :=
  fun _ _ _ => False

/-- Digest-only verifier shape: every digest is treated as authority. -/
def AcceptsAnyDigest : Nat → Unit → TwoImage → Prop :=
  fun _ _ _ => True

private theorem forged_unreachable_under_no_transition :
    ¬ FPrimeInduction.Reachable
      NoTransition
      TwoImage.initial
      1
      TwoImage.forged := by
  intro hReach
  cases hReach with
  | step _hPrior hStep =>
      exact hStep

/--
Red-team: accepting a digest alone cannot satisfy trace soundness.

The verifier accepts the forged image, but no valid transition trace reaches it.
-/
theorem digest_only_acceptance_is_not_trace_sound :
    AcceptsAnyDigest 1 () TwoImage.forged ∧
      ¬ FPrimeTraceAuthority.VerifierSound
        NoTransition
        TwoImage.initial
        AcceptsAnyDigest := by
  constructor
  · trivial
  · intro hSound
    exact
      forged_unreachable_under_no_transition
        (FPrimeTraceAuthority.verifierSound_priorAuthoritySound
          hSound
          1
          ()
          TwoImage.forged
          trivial)

/--
Red-team: aggregate digit summaries cannot force pointwise child identity.
-/
theorem aggregate_digit_summary_does_not_authorize_unique_children :
    ¬ (∀ a b : DecDigitUniqueness.ColumnDigits 1,
      DecDigitUniqueness.binaryColumnDigits a →
      DecDigitUniqueness.binaryColumnDigits b →
      BinaryChildTableAuthorization.fixedColumnLength 14 a →
      BinaryChildTableAuthorization.fixedColumnLength 14 b →
      AggregateChildTableNecessity.aggregateDigitSum a =
        AggregateChildTableNecessity.aggregateDigitSum b →
        a = b) :=
  AggregateChildTableNecessity.aggregate_digit_sum_not_functional_for_binary_fixed_length

/--
Red-team: aggregate norm totals cannot force child-position identity.
-/
theorem aggregate_norm_total_does_not_authorize_child_identity :
    ¬ (∀ a b : Fin 14 → Nat,
      AggregateChildTableNecessity.aggregateNormSum a =
        AggregateChildTableNecessity.aggregateNormSum b →
        a = b) :=
  AggregateChildTableNecessity.aggregate_norm_sum_not_functional_for_fixed_child_count

/--
Red-team: signed low-norm base-2 DEC recomposition is not unique.
-/
theorem signed_low_norm_dec_recomposition_does_not_authorize_unique_children :
    DecDigitUniqueness.recompose2 (1, 0) =
        DecDigitUniqueness.recompose2 (-1, 1) ∧
      DecDigitUniqueness.signedLowNorm2 (1, 0) ∧
      DecDigitUniqueness.signedLowNorm2 (-1, 1) ∧
      (1, 0) ≠ (-1, 1) :=
  DecDigitUniqueness.signed_low_norm_base2_not_unique

/--
Red-team: modular recomposition without a range proof is not unique.
-/
theorem modular_binary_recomposition_without_range_does_not_authorize_unique_children :
    ([0, 0] : List Nat).length = ([0, 1] : List Nat).length ∧
      DecDigitUniqueness.binaryNatDigits [0, 0] ∧
      DecDigitUniqueness.binaryNatDigits [0, 1] ∧
      DecDigitUniqueness.recomposeNatDigits [0, 0] % 2 =
        DecDigitUniqueness.recomposeNatDigits [0, 1] % 2 ∧
      ([0, 0] : List Nat) ≠ [0, 1] :=
  DecDigitUniqueness.fixed_length_binary_mod_recomposition_not_unique_without_range

end FPrimeTraceAuthorityRedTeam

end DirectCcsFPrime
