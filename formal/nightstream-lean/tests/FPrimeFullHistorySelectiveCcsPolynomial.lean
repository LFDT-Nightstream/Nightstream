import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs

namespace Tests.FPrimeFullHistorySelectiveCcsPolynomial

open Nightstream.SuperNeo.Concrete
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Polynomial.Ports
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Polynomial.Semantics
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Polynomial.Components
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Polynomial.Rows
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Polynomial.Necessity
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Padding.Refinement

example : terms.length = 27 := term_count_exact

example : polynomial.canonicalEqualityGatedDegreeBound = 9 :=
  canonicalEqualityGatedDegreeBound_exact

example (point : Fin 13 -> F) :
    evaluate point = combinedResidual point :=
  evaluate_eq_combinedResidual point

example (selector bit : F) :
    evaluate (booleanPoint selector bit) =
      booleanResidual (booleanPoint selector bit) :=
  evaluate_booleanPoint selector bit

example (selector input output : F) :
    evaluate (sboxPoint selector input output) =
      productResidual (sboxPoint selector input output) +
        sboxResidual (sboxPoint selector input output) :=
  evaluate_sboxPoint selector input output

example : AcceptsWithout .boolean booleanWitness ∧
    ¬ FullAccepts booleanWitness := boolean_necessary

example : AcceptsWithout .product productWitness ∧
    ¬ FullAccepts productWitness := product_necessary

example : AcceptsWithout .sbox sboxWitness ∧
    ¬ FullAccepts sboxWitness := sbox_necessary

example : AcceptsWithout .centered centeredWitness ∧
    ¬ FullAccepts centeredWitness := centered_necessary

example : AcceptsWithout .evaluation evaluationWitness ∧
    ¬ FullAccepts evaluationWitness := evaluation_necessary

example : AcceptsWithout .canonical canonicalWitness ∧
    ¬ FullAccepts canonicalWitness := canonical_necessary

example (constantValue paddingValue : F) :
    evaluate (paddingPortPoint constantValue paddingValue) =
      -(constantValue * paddingValue) :=
  evaluate_paddingPortPoint constantValue paddingValue

example : Role.generalSelector.index.val = 1 := rfl
example : Role.c.index.val = 4 := rfl
example : Role.evalTailRight.index.val = 12 := rfl
example (role : Role) : Role.ofIndex role.index = role := Role.ofIndex_index role
example (index : Fin 13) : (Role.ofIndex index).index = index :=
  Role.index_ofIndex index

end Tests.FPrimeFullHistorySelectiveCcsPolynomial
