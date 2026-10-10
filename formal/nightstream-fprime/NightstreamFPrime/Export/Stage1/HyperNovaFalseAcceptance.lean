import NightstreamFPrime.Export.Stage1.HyperNovaVisitedSecurity

/-!
False acceptance at the selected full-opening terminal boundary means that
the verifier accepts but no advice list has the advertised length and forward
application result. This is exactly the history conclusion in AdviceReturned.
It is distinct from bare NIFS Boolean acceptance and from extractor failure.

The bound is on the IVC adversary's original mixed law; no conditioning on
invalid inputs is used. It runs the reverse extractor of HyperNova Lemma 17
(`HyperNovaVisitedSecurity.reverseStages`) under HyperNova errata Assumption 1
(`HyperNovaVisitedSecurity.Assumption1`) and keeps each stage's
hash-collision mass explicit. Honest rejection is separate.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.HyperNovaFalseAcceptance

open scoped BigOperators ENNReal
open NightstreamFPrime.Spec
open NightstreamFPrime.Lifecycle
open HyperNovaHistory (Statement Envelope)
open HyperNovaHistoryProbability (Sample Accepted AdviceReturned)
open HyperNovaFirstFailure (MarkedHashCollision)
open HyperNovaVisitedSecurity (event_ne_top NifsAdversary NifsExtractor Assumption1 Closed IvcAdversary
  Stage reverseStages)
variable (application : Lifecycle.Stage1.Application.Program)
  (fits : PerApplicationFixedPoint.FitsTwoPow28 application)
  (setup : PerApplicationCanonicalPackage.CommitmentSetup application)

/-- Acceptance with no history of circuit witnesses (each of the
application's witness length) that has the advertised length and reaches the
advertised final state. The initial law may mix valid and invalid statements. -/
def FalseAcceptance (input : Statement × Envelope application) : Prop :=
  PerApplicationTerminal.Holds application fits setup input.1 input.2 ∧
    ¬ ∃ advice : List AppWitness,
      (∀ witness ∈ advice, witness.length = application.witnessWordCount) ∧
      advice.length = input.1.iteration ∧
      advice.foldl application.step input.1.z0 = input.1.zi

/-- Invalid accepted statements cannot have a returned valid history. This
uses the event definitions only, with no output-soundness premise. -/
theorem falseAcceptance_not_adviceReturned (sample : Sample application)
    (invalid : FalseAcceptance application fits setup (sample.1, sample.2.1)) :
    ¬ AdviceReturned application fits sample := by
  rintro ⟨advice, _unused, returned, length, evaluated⟩
  exact invalid.2 ⟨advice, HyperNovaHistory.run_witness_length application fits _ _ _ returned,
    length, evaluated⟩

variable {Admitted : NifsAdversary application fits setup → Prop} {StageAdmitted : Stage application fits setup → Prop}
  {Efficient : (adversary : NifsAdversary application fits setup) → NifsExtractor adversary → Prop}
  {error : NifsAdversary application fits setup → ℝ}
  (assumption : Assumption1 Admitted Efficient error) (closed : Closed Admitted StageAdmitted Efficient)
  (adversary : IvcAdversary application fits setup) (admitted : StageAdmitted (Stage.start adversary))

local notation "stages" => reverseStages assumption closed adversary admitted

/-- False-acceptance bound for the selected terminal verifier on the IVC
adversary's original mixed law under Assumption 1 (HyperNova Lemma 17). False
acceptance is an accepted input with no returned history, so the bound is
`HyperNovaVisitedSecurity.history_failure_le`. This is not a bound on bare NIFS
Boolean acceptance. -/
theorem probability_bound (depth : Nat)
    (depthBound : ∀ tape ∈ adversary.tape.support, (adversary.output tape).1.iteration ≤ depth) :
    (adversary.tape.toOuterMeasure {tape | FalseAcceptance application fits setup (adversary.output tape)}).toReal ≤
      ∑ j : Fin depth,
        (((stages j.val).1.tape.toOuterMeasure
            {tape | MarkedHashCollision application fits setup ((stages j.val).1.visit tape)}).toReal +
          error (stages j.val).1.nifs) := by
  refine le_trans (ENNReal.toReal_mono (event_ne_top _ _) ?_)
    (HyperNovaVisitedSecurity.history_failure_le assumption closed adversary admitted depth depthBound)
  calc
    _ = (stages depth).1.reverseLaw.toOuterMeasure
          {sample | FalseAcceptance application fits setup (sample.1, sample.2.1)} :=
        HyperNovaVisitedSecurity.reverse_input_event assumption closed adversary admitted depth
          {input | FalseAcceptance application fits setup input}
    _ ≤ (stages depth).1.reverseLaw.toOuterMeasure
          {sample | Accepted application fits setup sample ∧ ¬ AdviceReturned application fits sample} :=
        PMF.toOuterMeasure_mono _ fun sample member =>
        ⟨member.1.1, falseAcceptance_not_adviceReturned application fits setup sample member.1⟩

end NightstreamFPrime.Export.Stage1.HyperNovaFalseAcceptance
