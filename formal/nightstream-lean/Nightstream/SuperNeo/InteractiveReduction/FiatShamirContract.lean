import Nightstream.SuperNeo.InteractiveReduction.ProbabilityCalculus

/-!
Explicit random-oracle probability contract for the Fiat--Shamir transform.

Owns: six fixed event predicates matching `FiatShamirSecurityEvent`, one
independently stated bound per predicate, their exact nested union, and the
generic theorem that the total failure probability is bounded by
`FiatShamirErrorBudget.total`.

Does not own: an oracle implementation, query encoding, transcript schedule,
adversary, extractor, multi-forking/programming theorem, any event bound,
Poseidon2, Rust, R1CS, artifacts, minimality, or costs.

Emits constraints: no.

A protocol must instantiate these predicates with its actual typed collision,
sampling, and programming events and prove every field of
`ExplicitRandomOracleContract`.  Supplying arbitrary predicates or the desired
headline soundness conclusion is not a protocol proof.
-/

set_option autoImplicit false

namespace Nightstream.SuperNeo.InteractiveReduction.FiatShamirContract

open Nightstream.SuperNeo.InteractiveReduction.Paper
open Nightstream.SuperNeo.InteractiveReduction.ProbabilityCalculus

universe uWeight uOutcome

/-- Exact named predicates in one concrete oracle experiment. -/
structure EventPredicates (Outcome : Type uOutcome) where
  publicInputBindingCollision : Outcome -> Prop
  transcriptReplayCollision : Outcome -> Prop
  transcriptStateCollision : Outcome -> Prop
  outputAbsorptionCollision : Outcome -> Prop
  challengeSamplingFailure : Outcome -> Prop
  multiForkProgrammingFailure : Outcome -> Prop

/-- Select the concrete predicate associated with one closed event tag. -/
def EventPredicates.at
    {Outcome : Type uOutcome}
    (events : EventPredicates Outcome) :
    FiatShamirSecurityEvent -> Outcome -> Prop
  | .publicInputBindingCollision => events.publicInputBindingCollision
  | .transcriptReplayCollision => events.transcriptReplayCollision
  | .transcriptStateCollision => events.transcriptStateCollision
  | .outputAbsorptionCollision => events.outputAbsorptionCollision
  | .challengeSamplingFailure => events.challengeSamplingFailure
  | .multiForkProgrammingFailure => events.multiForkProgrammingFailure

/-- Exact nested union in transcript schedule order. -/
def AnyFailure
    {Outcome : Type uOutcome}
    (events : EventPredicates Outcome)
    (outcome : Outcome) : Prop :=
  events.publicInputBindingCollision outcome \/
    events.transcriptReplayCollision outcome \/
    events.transcriptStateCollision outcome \/
    events.outputAbsorptionCollision outcome \/
    events.challengeSamplingFailure outcome \/
    events.multiForkProgrammingFailure outcome

/-- The nested union is extensionally the existence of one event from the
closed six-constructor family. -/
theorem anyFailure_iff_exists_event
    {Outcome : Type uOutcome}
    (events : EventPredicates Outcome)
    (outcome : Outcome) :
    AnyFailure events outcome <->
      exists event, events.at event outcome := by
  constructor
  · intro failure
    rcases failure with
      publicInput | replay | state | output | sampling | programming
    · exact ⟨.publicInputBindingCollision, publicInput⟩
    · exact ⟨.transcriptReplayCollision, replay⟩
    · exact ⟨.transcriptStateCollision, state⟩
    · exact ⟨.outputAbsorptionCollision, output⟩
    · exact ⟨.challengeSamplingFailure, sampling⟩
    · exact ⟨.multiForkProgrammingFailure, programming⟩
  · rintro ⟨event, failure⟩
    cases event with
    | publicInputBindingCollision => exact Or.inl failure
    | transcriptReplayCollision => exact Or.inr (Or.inl failure)
    | transcriptStateCollision =>
        exact Or.inr (Or.inr (Or.inl failure))
    | outputAbsorptionCollision =>
        exact Or.inr (Or.inr (Or.inr (Or.inl failure)))
    | challengeSamplingFailure =>
        exact Or.inr (Or.inr (Or.inr (Or.inr (Or.inl failure))))
    | multiForkProgrammingFailure =>
        exact Or.inr (Or.inr (Or.inr (Or.inr (Or.inr failure))))

/-- Explicit random-oracle assumption boundary for one actual experiment.
Each field bounds only its corresponding named predicate. -/
structure ExplicitRandomOracleContract
    {Weight : Type uWeight}
    {Outcome : Type uOutcome}
    {scale : ProbabilityScale Weight}
    (experiment : ProbabilityExperiment scale Outcome)
    (events : EventPredicates Outcome)
    (budget : FiatShamirErrorBudget Weight) : Prop where
  publicInputBindingCollision :
    scale.le
      (experiment.probability events.publicInputBindingCollision)
      budget.publicInputBindingCollision
  transcriptReplayCollision :
    scale.le
      (experiment.probability events.transcriptReplayCollision)
      budget.transcriptReplayCollision
  transcriptStateCollision :
    scale.le
      (experiment.probability events.transcriptStateCollision)
      budget.transcriptStateCollision
  outputAbsorptionCollision :
    scale.le
      (experiment.probability events.outputAbsorptionCollision)
      budget.outputAbsorptionCollision
  challengeSamplingFailure :
    scale.le
      (experiment.probability events.challengeSamplingFailure)
      budget.challengeSamplingFailure
  multiForkProgrammingFailure :
    scale.le
      (experiment.probability events.multiForkProgrammingFailure)
      budget.multiForkProgrammingFailure

/-- The six per-event random-oracle contracts imply the exact total failure
bound.  This theorem is probability bookkeeping only; it proves none of the
six contract fields. -/
theorem anyFailure_probability_le_total
    {Weight : Type uWeight}
    {Outcome : Type uOutcome}
    {scale : ProbabilityScale Weight}
    (scaleLaws : ScaleLaws scale)
    (experiment : ProbabilityExperiment scale Outcome)
    (unionLaw : UnionBound experiment)
    (events : EventPredicates Outcome)
    (budget : FiatShamirErrorBudget Weight)
    (contract : ExplicitRandomOracleContract experiment events budget) :
    scale.le
      (experiment.probability (AnyFailure events))
      (budget.total scale) := by
  have samplingProgramming :
      scale.le
        (experiment.probability fun outcome =>
          events.challengeSamplingFailure outcome \/
            events.multiForkProgrammingFailure outcome)
        (scale.add budget.challengeSamplingFailure
          budget.multiForkProgrammingFailure) :=
    scale.le_trans
      (unionLaw.unionBound events.challengeSamplingFailure
        events.multiForkProgrammingFailure)
      (scaleLaws.add_mono contract.challengeSamplingFailure
        contract.multiForkProgrammingFailure)
  have outputTail :
      scale.le
        (experiment.probability fun outcome =>
          events.outputAbsorptionCollision outcome \/
            events.challengeSamplingFailure outcome \/
            events.multiForkProgrammingFailure outcome)
        (scale.add budget.outputAbsorptionCollision
          (scale.add budget.challengeSamplingFailure
            budget.multiForkProgrammingFailure)) :=
    scale.le_trans
      (unionLaw.unionBound events.outputAbsorptionCollision
        (fun outcome =>
          events.challengeSamplingFailure outcome \/
            events.multiForkProgrammingFailure outcome))
      (scaleLaws.add_mono contract.outputAbsorptionCollision
        samplingProgramming)
  have stateTail :
      scale.le
        (experiment.probability fun outcome =>
          events.transcriptStateCollision outcome \/
            events.outputAbsorptionCollision outcome \/
            events.challengeSamplingFailure outcome \/
            events.multiForkProgrammingFailure outcome)
        (scale.add budget.transcriptStateCollision
          (scale.add budget.outputAbsorptionCollision
            (scale.add budget.challengeSamplingFailure
              budget.multiForkProgrammingFailure))) :=
    scale.le_trans
      (unionLaw.unionBound events.transcriptStateCollision
        (fun outcome =>
          events.outputAbsorptionCollision outcome \/
            events.challengeSamplingFailure outcome \/
            events.multiForkProgrammingFailure outcome))
      (scaleLaws.add_mono contract.transcriptStateCollision outputTail)
  have replayTail :
      scale.le
        (experiment.probability fun outcome =>
          events.transcriptReplayCollision outcome \/
            events.transcriptStateCollision outcome \/
            events.outputAbsorptionCollision outcome \/
            events.challengeSamplingFailure outcome \/
            events.multiForkProgrammingFailure outcome)
        (scale.add budget.transcriptReplayCollision
          (scale.add budget.transcriptStateCollision
            (scale.add budget.outputAbsorptionCollision
              (scale.add budget.challengeSamplingFailure
                budget.multiForkProgrammingFailure)))) :=
    scale.le_trans
      (unionLaw.unionBound events.transcriptReplayCollision
        (fun outcome =>
          events.transcriptStateCollision outcome \/
            events.outputAbsorptionCollision outcome \/
            events.challengeSamplingFailure outcome \/
            events.multiForkProgrammingFailure outcome))
      (scaleLaws.add_mono contract.transcriptReplayCollision stateTail)
  exact
    scale.le_trans
      (unionLaw.unionBound events.publicInputBindingCollision
        (fun outcome =>
          events.transcriptReplayCollision outcome \/
            events.transcriptStateCollision outcome \/
            events.outputAbsorptionCollision outcome \/
            events.challengeSamplingFailure outcome \/
            events.multiForkProgrammingFailure outcome))
      (scaleLaws.add_mono contract.publicInputBindingCollision replayTail)

end Nightstream.SuperNeo.InteractiveReduction.FiatShamirContract
