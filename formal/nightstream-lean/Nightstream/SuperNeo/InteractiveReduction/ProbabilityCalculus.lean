import Nightstream.SuperNeo.InteractiveReduction.Paper

/-!
Minimal probability calculus for paper interactive reductions.

Owns: the order laws and event-union law needed to convert a pointwise
`success -> extracted or bad` theorem plus a bound on `bad` into the exact
subtractive extraction inequality used by Definitions 5, 9, and 10.

Does not own: a probability implementation, protocol events, any event bound,
conditioning, rejection sampling, SumCheck, commitments, Fiat--Shamir, Rust,
R1CS, or costs.

Emits constraints: no.

The protocol owner must instantiate these laws for its actual experiment.
In particular, `unionBound` is generic probability arithmetic; it cannot be
replaced by a premise already asserting the protocol's desired extraction
inequality.
-/

namespace Nightstream.SuperNeo.InteractiveReduction.ProbabilityCalculus

open Nightstream.SuperNeo.InteractiveReduction.Paper

universe uWeight uOutcome

/-- Arithmetic/order laws used by one-step bad-event accounting. -/
structure ScaleLaws
    {Weight : Type uWeight}
    (scale : ProbabilityScale Weight) : Prop where
  add_mono : forall {left lower right upper},
    scale.le left lower -> scale.le right upper ->
      scale.le (scale.add left right) (scale.add lower upper)
  subtract_le_of_le_add : forall {probability good error},
    scale.le probability (scale.add good error) ->
      scale.le (scale.subtract probability error) good

/-- The standard union bound for one concrete probability experiment. -/
structure UnionBound
    {Weight : Type uWeight}
    {Outcome : Type uOutcome}
    {scale : ProbabilityScale Weight}
    (experiment : ProbabilityExperiment scale Outcome) : Prop where
  unionBound : forall (left right : Outcome -> Prop),
    scale.le
      (experiment.probability fun outcome => left outcome \/ right outcome)
      (scale.add (experiment.probability left)
        (experiment.probability right))

/-- A pointwise success cover and an independently bounded bad event imply
the exact subtractive extraction inequality. -/
theorem loss_le_of_cover
    {Weight : Type uWeight}
    {Outcome : Type uOutcome}
    {scale : ProbabilityScale Weight}
    (scaleLaws : ScaleLaws scale)
    (experiment : ProbabilityExperiment scale Outcome)
    (unionLaw : UnionBound experiment)
    (success extracted bad : Outcome -> Prop)
    (error : Weight)
    (cover : forall outcome, success outcome ->
      extracted outcome \/ bad outcome)
    (badBound : scale.le (experiment.probability bad) error) :
    scale.le
      (scale.subtract (experiment.probability success) error)
      (experiment.probability extracted) := by
  have successBelowUnion :
      scale.le (experiment.probability success)
        (experiment.probability fun outcome =>
          extracted outcome \/ bad outcome) :=
    experiment.monotone cover
  have unionBelowExact :
      scale.le
        (experiment.probability fun outcome =>
          extracted outcome \/ bad outcome)
        (scale.add (experiment.probability extracted)
          (experiment.probability bad)) :=
    unionLaw.unionBound extracted bad
  have exactBelowBudget :
      scale.le
        (scale.add (experiment.probability extracted)
          (experiment.probability bad))
        (scale.add (experiment.probability extracted) error) :=
    scaleLaws.add_mono (scale.le_refl _) badBound
  apply scaleLaws.subtract_le_of_le_add
  exact scale.le_trans successBelowUnion
    (scale.le_trans unionBelowExact exactBelowBudget)

end Nightstream.SuperNeo.InteractiveReduction.ProbabilityCalculus
