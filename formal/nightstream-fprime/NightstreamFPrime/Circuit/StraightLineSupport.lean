import NightstreamFPrime.Circuit.VariableSupport

namespace NightstreamFPrime.Circuit

theorem recipeConstraints_varsSatisfy
    (start : Nat) (recipes : List Expr) (allowed : Nat → Prop)
    (recipesSupported : ∀ recipe ∈ recipes,
      recipe.VarsSatisfy allowed)
    (targetsSupported : ∀ index, index < recipes.length →
      allowed (start + index)) :
    ∀ expression ∈ recipeConstraints start recipes,
      expression.VarsSatisfy allowed := by
  induction recipes generalizing start with
  | nil =>
      intro expression member
      simp [recipeConstraints] at member
  | cons recipe recipes inductionHypothesis =>
      intro expression member
      simp only [recipeConstraints, List.mem_cons] at member
      rcases member with rfl | member
      · exact ⟨targetsSupported 0 (by simp),
          ⟨trivial, recipesSupported recipe (by simp)⟩⟩
      · apply inductionHypothesis (start := start + 1)
        · intro current currentMember
          exact recipesSupported current (by simp [currentMember])
        · intro index indexBound
          have target := targetsSupported (index + 1) (by simp; omega)
          simpa [Nat.add_assoc, Nat.add_comm, Nat.add_left_comm] using target
        · exact member

end NightstreamFPrime.Circuit
