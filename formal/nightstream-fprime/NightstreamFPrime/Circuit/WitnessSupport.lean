import NightstreamFPrime.Circuit.VariableSupport

/-! Variable support of the expressions actually read by a witness batch. -/

namespace NightstreamFPrime.Circuit

def WitnessBatch.ReadsSatisfy (allowed : Nat → Prop) (batch : WitnessBatch) : Prop :=
  (∀ recipe ∈ batch.recipes, recipe.VarsSatisfy allowed) ∧
    ∀ hint ∈ batch.hints, hint.source.VarsSatisfy allowed

@[simp] theorem WitnessBatch.readsSatisfy_arithmetic
    (allowed : Nat → Prop) (start : Nat) (recipes : List Expr) :
    (WitnessBatch.arithmetic start recipes).ReadsSatisfy allowed ↔
      ∀ recipe ∈ recipes, recipe.VarsSatisfy allowed := by
  simp [ReadsSatisfy, arithmetic]

@[simp] theorem WitnessBatch.readsSatisfy_hinted
    (allowed : Nat → Prop) (start : Nat) (hints : List Hint) :
    (WitnessBatch.hinted start hints).ReadsSatisfy allowed ↔
      ∀ hint ∈ hints, hint.source.VarsSatisfy allowed := by
  simp [ReadsSatisfy, hinted]

end NightstreamFPrime.Circuit
