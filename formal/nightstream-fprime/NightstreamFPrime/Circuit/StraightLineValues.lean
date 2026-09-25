import NightstreamFPrime.Circuit.StraightLine

/-! Causal arithmetic rows determine their local values from the values
before the batch. Hinted cells need their own execution readback. -/

namespace NightstreamFPrime.Circuit

theorem recipeConstraints_values_unique (left right : Env) (start : Nat) (recipes : List Expr)
    (causal : RecipesCausal start recipes)
    (leftRows : ConstraintsHold left (recipeConstraints start recipes))
    (rightRows : ConstraintsHold right (recipeConstraints start recipes))
    (inputs : ∀ index, index < start → left index = right index) :
    ∀ index, index < start + recipes.length → left index = right index := by
  induction recipes generalizing start with
  | nil => simpa only [List.length_nil, Nat.add_zero] using inputs
  | cons recipe rest ih =>
    have headLeft := recipeConstraints_value left start (recipe :: rest) leftRows 0 (by simp)
    have headRight := recipeConstraints_value right start (recipe :: rest) rightRows 0 (by simp)
    have head : left start = right start := by
      simpa only [Nat.add_zero, List.get_cons_zero] using
        headLeft.trans ((Expr.eval_eq_of_agree_below recipe start left right causal.1 inputs).trans headRight.symm)
    have afterInputs : ∀ index, index < start + 1 → left index = right index := by
      intro index below
      by_cases earlier : index < start
      · exact inputs index earlier
      · have equal : index = start := by omega
        subst index
        exact head
    have tail (env : Env) (rows : ConstraintsHold env (recipeConstraints start (recipe :: rest))) :
        ConstraintsHold env (recipeConstraints (start + 1) rest) := by
      intro expression member
      exact rows expression (by simp only [recipeConstraints, List.mem_cons]; exact Or.inr member)
    have all := ih (start + 1) causal.2 (tail left leftRows) (tail right rightRows) afterInputs
    simpa only [List.length_cons, Nat.add_assoc, Nat.add_comm 1] using all

end NightstreamFPrime.Circuit
