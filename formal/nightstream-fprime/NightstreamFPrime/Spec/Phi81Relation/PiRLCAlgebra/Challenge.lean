import NightstreamFPrime.Spec.Phi81StrongSet

/-!
Challenge-validity predicate for the typed Phi81 `PiRLC.Algebra`.

Protocol: SuperNeo v1.2 Definitions 6 and 22 and `Pi_RLC`.

Owns: the exact unary algebra-field predicate and its connection to the
independently defined 54-coordinate five-symbol production set.

Does not own: Poseidon2 transcript derivation, rejection sampling, decoded
Rust/R1CS challenge membership, the low-norm invertibility proof, norm
growth of folded witnesses, row removal, or constraint counts.
-/

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
