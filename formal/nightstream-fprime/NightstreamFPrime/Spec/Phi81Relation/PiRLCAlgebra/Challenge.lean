import NightstreamFPrime.Spec.Phi81StrongSet

/-! SuperNeo v1.2 Definitions 6 and 22. -/

namespace NightstreamFPrime.Spec.Phi81Relation.PiRLCAlgebra.Challenge

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Phi81StrongSet

def challengeValid (value : RingF) : Prop := ProductionMember value

theorem challengeValid_iff (value : RingF) :
    challengeValid value ↔ ProductionMember value := by
  rfl

theorem embedScalar_valid (scalar : Scalar) :
    challengeValid (embedScalar scalar) :=
  embedScalar_member scalar

theorem pairwiseSecure_of_lowNormInvertibility
    (theorem8 : LowNormInvertibility) :
    forall {left right : RingF},
      challengeValid left -> challengeValid right -> left ≠ right ->
        RingFInvertible (ringFSub left right) := by
  exact productionSet_strong theorem8

end NightstreamFPrime.Spec.Phi81Relation.PiRLCAlgebra.Challenge
