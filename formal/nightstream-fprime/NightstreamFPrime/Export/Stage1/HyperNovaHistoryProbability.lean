import NightstreamFPrime.Export.Stage1.HyperNovaHistory
import Mathlib.Probability.ProbabilityMassFunction.Constructions

/-!
Owns the event bound for the actual selected reverse-history result. The
law retains the terminal statement, its opening, and every supplied source
return, including aborts. An accepted history can fail only at a visited
source-success check or a visited state-hash collision check. Bounds for
those two events must come from the experiment that produces this law.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.HyperNovaHistoryProbability

open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Spec
open HyperNovaHistory

variable (target : Wide.Target)

abbrev Sample := Statement × target.Envelope × List (SourceResult target)

/-- The exact selected terminal membership, before reverse extraction. -/
def Accepted (sample : Sample target) : Prop :=
  target.Holds sample.1 sample.2.1

/-- The actual reverse program returns a complete forward application history. -/
def AdviceReturned (sample : Sample target) : Prop :=
  ∃ advice unused,
    run target sample.1 sample.2.1 sample.2.2 = .ok (advice, unused) ∧
    advice.length = sample.1.iteration ∧
    advice.foldl target.program.step sample.1.z0 = sample.1.zi

/-- An accepted run lacks a required successful source return on its visited path. -/
def SourceFailure (sample : Sample target) : Prop :=
  Accepted target sample ∧ ¬ SuccessfulSources target sample.1 sample.2.1 sample.2.2

/-- An accepted run encounters a state-hash collision on its visited path. -/
def StateHashFailure (sample : Sample target) : Prop :=
  Accepted target sample ∧ ¬ NoStateHashCollisions target sample.1 sample.2.1 sample.2.2

/-- The deterministic reverse program supplies this event inclusion for any
law of terminal openings and source returns. No probability or independence
hypothesis is used. -/
theorem accepted_subset :
    {sample | Accepted target sample} ⊆
      ({sample | AdviceReturned target sample} ∪ {sample | SourceFailure target sample}) ∪
        {sample | StateHashFailure target sample} := by
  classical
  intro sample accepted
  by_cases sources : SuccessfulSources target sample.1 sample.2.1 sample.2.2
  · by_cases safe : NoStateHashCollisions target sample.1 sample.2.1 sample.2.2
    · exact Or.inl (Or.inl (run_correct target sample.1 sample.2.1 sample.2.2 accepted sources safe))
    · exact Or.inr ⟨accepted, safe⟩
  · exact Or.inl (Or.inr ⟨accepted, sources⟩)

/-- Accepted mass is bounded by successful complete-history mass plus the
two actual failure-event masses. The law may be adaptive and may include
aborts; no conditional or independent-call probability is substituted. -/
theorem accepted_probability_le (law : PMF (Sample target)) :
    law.toOuterMeasure {sample | Accepted target sample} ≤
      law.toOuterMeasure {sample | AdviceReturned target sample} +
        law.toOuterMeasure {sample | SourceFailure target sample} +
        law.toOuterMeasure {sample | StateHashFailure target sample} := by
  calc
    _ ≤ law.toOuterMeasure
        (({sample | AdviceReturned target sample} ∪ {sample | SourceFailure target sample}) ∪
          {sample | StateHashFailure target sample}) := MeasureTheory.measure_mono (accepted_subset target)
    _ ≤ law.toOuterMeasure
        ({sample | AdviceReturned target sample} ∪ {sample | SourceFailure target sample}) +
        law.toOuterMeasure {sample | StateHashFailure target sample} :=
      MeasureTheory.measure_union_le _ _
    _ ≤ _ := add_le_add (MeasureTheory.measure_union_le _ _) le_rfl

end NightstreamFPrime.Export.Stage1.HyperNovaHistoryProbability
