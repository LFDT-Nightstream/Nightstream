import NightstreamFPrime.Spec.SumCheck.GoldilocksCausal

/-!
Owns the prefix-only issue order of sum-check prover messages: each message
reads only the challenges before it. An abort issues no message list.
-/

namespace NightstreamFPrime.Spec.SumCheck.Finite.GoldilocksCausalTrace

open GoldilocksCausal

/-- Issue each message before using the current challenge to form the next
prefix. A failed message, including a rejected raw width, is represented by none. -/
def issued {degree : Nat} (strategy : Strategy degree) (fixed : List K) :
    List K → Option (List (FixedPolynomial K degree))
  | [] => some []
  | challenge :: rest =>
      match strategy fixed with
      | none => none
      | some message => (issued strategy (fixed ++ [challenge]) rest).map (message :: ·)

end NightstreamFPrime.Spec.SumCheck.Finite.GoldilocksCausalTrace
