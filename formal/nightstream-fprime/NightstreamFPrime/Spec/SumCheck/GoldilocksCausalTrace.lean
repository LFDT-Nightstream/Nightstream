import NightstreamFPrime.Spec.SumCheck.GoldilocksCausal

/-!
Owns the prefix-only issue order of sum-check prover messages: each message
reads only the challenges before it.
-/

namespace NightstreamFPrime.Spec.SumCheck.Finite.GoldilocksCausalTrace

open GoldilocksCausal

def issued {degree : Nat} (strategy : Strategy degree) (fixed : List K) :
    List K → Option (List (FixedPolynomial K degree))
  | [] => some []
  | challenge :: rest =>
      match strategy fixed with
      | none => none
      | some message => (issued strategy (fixed ++ [challenge]) rest).map (message :: ·)

end NightstreamFPrime.Spec.SumCheck.Finite.GoldilocksCausalTrace
