import Nightstream.SuperNeo.InteractiveReduction.CoordinateForking.RenderedBoundObstruction

/-!
Focused interface regression for the finite kernel counterexample to the
denominator rendered in SuperNeo Appendix C, Theorem 10.
-/

namespace tests.CoordinateForkingRenderedBoundObstruction

open Nightstream.SuperNeo.InteractiveReduction.CoordinateForking.RenderedBoundObstruction

#check challengeOfIndex
#check challengeIndex
#check challengeOfIndex_challengeIndex
#check challengeIndex_challengeOfIndex
#check challengeOfIndex_injective
#check challengeOfIndex_surjective
#check acceptedChallengeCount_eq_three
#check noAcceptedCoordinateFork
#check renderedClaimedLowerNumerator_eq_one
#check rendered_denominator_bound_counterexample

example : challengeSpaceCardinality = 9 := by decide

example : acceptedChallengeCount = 3 :=
  acceptedChallengeCount_eq_three

example : renderedClaimedLowerNumerator = 1 :=
  renderedClaimedLowerNumerator_eq_one

end tests.CoordinateForkingRenderedBoundObstruction
