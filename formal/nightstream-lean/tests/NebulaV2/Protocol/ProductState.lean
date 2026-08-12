import Nightstream.Protocol.NebulaV2.ProductState

set_option autoImplicit false

namespace tests.NebulaV2ProductState

open Nightstream.Protocol.NebulaV2
open Nightstream.Protocol.NebulaV2.ProductState

section

variable {ChallengeField : Type} [Field ChallengeField]

theorem opening_products_are_exactly_one
    (repetition : Fin 2) :
    (one : State ChallengeField) repetition =
      ({ initialSnapshot := 1
         writes := 1
         reads := 1
         finalSnapshot := 1 } : Four ChallengeField) :=
  rfl

/-- A state that only has one valid repetition is not a balanced V2 product
state. This excludes challenge-repetition omission. -/
theorem one_repetition_cannot_stand_for_two :
    ¬ Balanced
      (fun repetition : Fin 2 =>
        if repetition = 0 then
          ({ initialSnapshot := (1 : Nat)
             writes := 1
             reads := 1
             finalSnapshot := 1 } : Four Nat)
        else
          ({ initialSnapshot := 0
             writes := 1
             reads := 1
             finalSnapshot := 1 } : Four Nat)) := by
  intro balanced
  have second := balanced (1 : Fin 2)
  simp [Four.Balanced] at second

/-- A balanced closing value alone is not linked to the semantic records. The
`ClaimProductUpdate` and `ProductState.Covers` requirements exclude this
detached accumulator. -/
theorem detached_balanced_product_countermodel :
    ∃ products : State Nat,
      Balanced products ∧ products ≠ one := by
  let products : State Nat := fun _ =>
    { initialSnapshot := 2
      writes := 1
      reads := 2
      finalSnapshot := 1 }
  refine ⟨products, ?_, ?_⟩
  · intro repetition
    norm_num [Four.Balanced, products]
  · intro equal
    have atZero := congrFun equal (0 : Fin 2)
    have initial := congrArg Four.initialSnapshot atZero
    change 2 = 1 at initial
    omega

end

end tests.NebulaV2ProductState
