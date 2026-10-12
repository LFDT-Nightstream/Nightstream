import NightstreamFPrime.Export.Stage1.HyperNovaHistory
import Mathlib.Probability.ProbabilityMassFunction.Constructions

/-!
Owns the event bound for the actual reverse-history result of one application. The
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
variable (application : Lifecycle.Stage1.Application.Program)
  (fits : PerApplicationFixedPoint.FitsTwoPow28 application)
  (setup : PerApplicationCanonicalPackage.CommitmentSetup application)

abbrev Sample := Statement × Envelope application × List (SourceResult application)

/-- The exact selected terminal membership, before reverse extraction. -/
def Accepted (sample : Sample application) : Prop :=
  PerApplicationTerminal.Holds application fits setup sample.1 sample.2.1

/-- The actual reverse program returns a complete forward application history. -/
def AdviceReturned (sample : Sample application) : Prop :=
  ∃ advice unused,
    run application fits sample.1 sample.2.1 sample.2.2 = .ok (advice, unused) ∧
    advice.length = sample.1.iteration ∧
    advice.foldl application.step sample.1.z0 = sample.1.zi

/-- An accepted run lacks a required successful source return on its visited path. -/
def SourceFailure (sample : Sample application) : Prop :=
  Accepted application fits setup sample ∧ ¬ SuccessfulSources application fits setup sample.1 sample.2.1 sample.2.2

/-- An accepted run encounters a state-hash collision on its visited path. -/
def StateHashFailure (sample : Sample application) : Prop :=
  Accepted application fits setup sample ∧ ¬ NoStateHashCollisions application fits setup sample.1 sample.2.1 sample.2.2

/-- The deterministic reverse program supplies this event inclusion for any
law of terminal openings and source returns. No probability or independence
hypothesis is used. -/
theorem accepted_subset :
    {sample | Accepted application fits setup sample} ⊆
      ({sample | AdviceReturned application fits sample} ∪ {sample | SourceFailure application fits setup sample}) ∪
        {sample | StateHashFailure application fits setup sample} := by
  classical
  intro sample accepted
  by_cases sources : SuccessfulSources application fits setup sample.1 sample.2.1 sample.2.2
  · by_cases safe : NoStateHashCollisions application fits setup sample.1 sample.2.1 sample.2.2
    · exact Or.inl (Or.inl (run_correct application fits setup sample.1 sample.2.1 sample.2.2 accepted sources safe))
    · exact Or.inr ⟨accepted, safe⟩
  · exact Or.inl (Or.inr ⟨accepted, sources⟩)

/-- Accepted mass is bounded by successful complete-history mass plus the
two actual failure-event masses. The law may be adaptive and may include
aborts; no conditional or independent-call probability is substituted. -/
theorem accepted_probability_le (law : PMF (Sample application)) :
    law.toOuterMeasure {sample | Accepted application fits setup sample} ≤
      law.toOuterMeasure {sample | AdviceReturned application fits sample} +
        law.toOuterMeasure {sample | SourceFailure application fits setup sample} +
        law.toOuterMeasure {sample | StateHashFailure application fits setup sample} := by
  calc
    _ ≤ law.toOuterMeasure
        (({sample | AdviceReturned application fits sample} ∪ {sample | SourceFailure application fits setup sample}) ∪
          {sample | StateHashFailure application fits setup sample}) := MeasureTheory.measure_mono (accepted_subset application fits setup)
    _ ≤ law.toOuterMeasure ({sample | AdviceReturned application fits sample} ∪ {sample | SourceFailure application fits setup sample}) +
        law.toOuterMeasure {sample | StateHashFailure application fits setup sample} :=
      MeasureTheory.measure_union_le _ _
    _ ≤ _ := add_le_add (MeasureTheory.measure_union_le _ _) le_rfl

end NightstreamFPrime.Export.Stage1.HyperNovaHistoryProbability
