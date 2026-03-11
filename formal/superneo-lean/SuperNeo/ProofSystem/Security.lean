import SuperNeo.ProofSystem.Negligible

namespace SuperNeo.ProofSystem

/-- Abstract probability model surface used by protocol theorem statements. -/
structure ProbModel where
  Pr : Prop → Rat
  prNonneg : ∀ P : Prop, 0 ≤ Pr P
  prLeOne : ∀ P : Prop, Pr P ≤ 1
  prFalse : Pr False = 0
  prMonotone : ∀ {P Q : Prop}, (P → Q) → Pr P ≤ Pr Q
  prUnionLeAdd : ∀ P Q : Prop, Pr (P ∨ Q) ≤ Pr P + Pr Q

private def zeroError : ErrorFn := fun _ => 0

private theorem negligible_zeroError : IsNegligible zeroError := by
  simpa [zeroError] using (isNegligible_zero : IsNegligible (fun _ => 0))

/-- Error accounting model with explicit source terms and total term. -/
structure ErrorModel where
  epsSumcheck : ErrorFn
  epsMSIS : ErrorFn
  epsSchwartzZippel : ErrorFn
  epsBinding : ErrorFn
  epsRelaxedBinding : ErrorFn
  epsTotal : ErrorFn
  epsTotal_decomp :
    ∀ n,
      epsTotal n =
        epsSumcheck n + epsMSIS n + epsSchwartzZippel n + epsBinding n + epsRelaxedBinding n
  negligibleSumcheck : IsNegligible epsSumcheck
  negligibleMSIS : IsNegligible epsMSIS
  negligibleSchwartzZippel : IsNegligible epsSchwartzZippel
  negligibleBinding : IsNegligible epsBinding
  negligibleRelaxedBinding : IsNegligible epsRelaxedBinding
  negligibleTotal : IsNegligible epsTotal

namespace ErrorModel

/--
Canonical error-model constructor from the five component error surfaces.
The total error is defined as their pointwise sum and its negligibility is
derived internally.
-/
def ofComponents
  (epsSumcheck epsMSIS epsSchwartzZippel epsBinding epsRelaxedBinding : ErrorFn)
  (negligibleSumcheck : IsNegligible epsSumcheck)
  (negligibleMSIS : IsNegligible epsMSIS)
  (negligibleSchwartzZippel : IsNegligible epsSchwartzZippel)
  (negligibleBinding : IsNegligible epsBinding)
  (negligibleRelaxedBinding : IsNegligible epsRelaxedBinding) :
  ErrorModel where
  epsSumcheck := epsSumcheck
  epsMSIS := epsMSIS
  epsSchwartzZippel := epsSchwartzZippel
  epsBinding := epsBinding
  epsRelaxedBinding := epsRelaxedBinding
  epsTotal := fun n =>
    epsSumcheck n + epsMSIS n + epsSchwartzZippel n + epsBinding n + epsRelaxedBinding n
  epsTotal_decomp := by
    intro n
    rfl
  negligibleSumcheck := negligibleSumcheck
  negligibleMSIS := negligibleMSIS
  negligibleSchwartzZippel := negligibleSchwartzZippel
  negligibleBinding := negligibleBinding
  negligibleRelaxedBinding := negligibleRelaxedBinding
  negligibleTotal := by
    simpa [Function.comp, add_assoc, add_left_comm, add_comm] using
      isNegligible_add
        (isNegligible_add
          (isNegligible_add
            (isNegligible_add negligibleSumcheck negligibleMSIS)
            negligibleSchwartzZippel)
          negligibleBinding)
        negligibleRelaxedBinding

end ErrorModel

/-- A canonical zero-error model used as a default scaffold value. -/
def zeroErrorModel : ErrorModel :=
  ErrorModel.ofComponents
    zeroError
    zeroError
    zeroError
    zeroError
    zeroError
    negligible_zeroError
    negligible_zeroError
    negligible_zeroError
    negligible_zeroError
    negligible_zeroError

end SuperNeo.ProofSystem
