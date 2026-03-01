import SuperNeo.ProofSystem.Negligible

namespace SuperNeo.ProofSystem

/-- Abstract probability model surface used by protocol theorem statements. -/
structure ProbModel where
  Pr : Prop → Rat
  prNonneg : ∀ P : Prop, 0 ≤ Pr P
  prLeOne : ∀ P : Prop, Pr P ≤ 1

private def zeroError : ErrorFn := fun _ => 0

private theorem negligible_zeroError : IsNegligible zeroError := by
  simp [zeroError]

/-- Error accounting model with explicit source terms and total term. -/
structure ErrorModel where
  epsSumcheck : ErrorFn
  epsSchwartzZippel : ErrorFn
  epsBinding : ErrorFn
  epsRelaxedBinding : ErrorFn
  epsTotal : ErrorFn
  epsTotal_decomp :
    ∀ n,
      epsTotal n =
        epsSumcheck n + epsSchwartzZippel n + epsBinding n + epsRelaxedBinding n
  negligibleSumcheck : IsNegligible epsSumcheck
  negligibleSchwartzZippel : IsNegligible epsSchwartzZippel
  negligibleBinding : IsNegligible epsBinding
  negligibleRelaxedBinding : IsNegligible epsRelaxedBinding
  negligibleTotal : IsNegligible epsTotal

/-- A canonical zero-error model used as a default scaffold value. -/
def zeroErrorModel : ErrorModel where
  epsSumcheck := zeroError
  epsSchwartzZippel := zeroError
  epsBinding := zeroError
  epsRelaxedBinding := zeroError
  epsTotal := zeroError
  epsTotal_decomp := by
    intro n
    simp [zeroError]
  negligibleSumcheck := negligible_zeroError
  negligibleSchwartzZippel := negligible_zeroError
  negligibleBinding := negligible_zeroError
  negligibleRelaxedBinding := negligible_zeroError
  negligibleTotal := negligible_zeroError

end SuperNeo.ProofSystem
