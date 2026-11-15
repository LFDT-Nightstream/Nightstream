import NightstreamFPrime.Spec.SumCheck.FieldCausal

namespace NightstreamFPrime.Spec.SumCheck.Finite.FieldCausalTrace

open FieldCausal

def issued {degree : Nat} (strategy : Strategy degree) (fixed : List K) :
    List K → Option (List (FixedPolynomial K degree))
  | [] => some []
  | challenge :: rest =>
      match strategy fixed with
      | none => none
      | some message => (issued strategy (fixed ++ [challenge]) rest).map (message :: ·)

end NightstreamFPrime.Spec.SumCheck.Finite.FieldCausalTrace
