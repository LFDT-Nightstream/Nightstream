import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingVariableHashRecipeSchema
import Nightstream.Implementation.R1CS.Canonical.GoldilocksField

/-!
Contract: satisfying a variable hash recipe's constant rows fixes every
allocated constant column to its declared Goldilocks value.

Owns only constant-row semantics. It does not own the Poseidon2 trace,
artifact identity, or external-input authority.

Assurance tier: model-level.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingVariableHashRecipeConstantSound

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Program
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingVariableHashRecipe.Artifact

theorem constantRows_values
    (recipe : VariableHashRecipe)
    (assignment : Nat -> Nat)
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (valuesCanonical : ∀ value ∈ recipe.constantValues,
      value < goldilocksP)
    (satisfied : Satisfies (constantRows recipe) assignment) :
    recipe.constantColumns.map assignment = recipe.constantValues := by
  change Satisfies
    ((recipe.constantColumns.zip recipe.constantValues).map fun entry =>
      builderLinearRow entry.1 [(0, entry.2)]) assignment at satisfied
  have lengths : recipe.constantColumns.length =
      recipe.constantValues.length := by
    simp [VariableHashRecipe.constantColumns]
  generalize columnsDefinition : recipe.constantColumns = columns at lengths satisfied ⊢
  generalize valuesDefinition : recipe.constantValues = values at lengths valuesCanonical satisfied ⊢
  clear columnsDefinition valuesDefinition
  induction columns generalizing values with
  | nil =>
      cases values <;> simp_all
  | cons column columns inductionHypothesis =>
      cases values with
      | nil => simp at lengths
      | cons value values =>
          simp only [List.length_cons, Nat.succ.injEq] at lengths
          have rowHolds := satisfied
            (builderLinearRow column [(0, value)]) (by simp)
          have valueCanonical : value < goldilocksP :=
            valuesCanonical value (by simp)
          have head : assignment column = value := by
            by_cases zero : value = 0
            · subst value
              simpa [builderLinearRow, RowHolds, lcEval, negateTerms,
                negCoeff, one, Nat.mod_eq_of_lt (canonical column)] using
                rowHolds
            · have defined := builderLinearRow_sound canonical one column
                [(0, value)]
                (by simp [CanonicalTerms, Nat.pos_of_ne_zero zero,
                  valueCanonical]) rowHolds
              simpa [lcEval, one, Nat.mod_eq_of_lt valueCanonical] using
                defined
          have tailSatisfied : Satisfies
              ((columns.zip values).map fun entry =>
                builderLinearRow entry.1 [(0, entry.2)]) assignment := by
            intro row member
            exact satisfied row (by simp [member])
          simp only [List.map_cons, List.cons.injEq]
          exact ⟨head, inductionHypothesis values lengths
            (fun current member => valuesCanonical current
              (List.mem_cons_of_mem value member))
            tailSatisfied⟩

end Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingVariableHashRecipeConstantSound
