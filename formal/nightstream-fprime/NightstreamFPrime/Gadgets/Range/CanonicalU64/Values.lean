import NightstreamFPrime.Gadgets.Range.CanonicalU64

/-! Exact values of the canonical word completion. Equal source words give
equal completed local cells, including the inverse hint and derived flag.
This statement concerns the constructor, not uniqueness of satisfying rows. -/

namespace NightstreamFPrime.Gadgets.Range.CanonicalU64

open NightstreamFPrime.Circuit NightstreamFPrime.Spec

theorem completeEnv_inverse (interface : Interface) (env : Env) (offset : Nat) :
    completeEnv interface env offset (offset + bitCount) =
      Hint.inverse ((highDifferenceExpr offset).eval (completeBits interface env offset)) := by
  simp [completeEnv, completeInverse, executeRecipes, executeHints, Env.set, inverseHint, Hint.eval]

private theorem inverse_highDifference (interface : Interface) (env : Env) (offset : Nat) :
    (highDifferenceExpr offset).eval (completeInverse interface env offset) =
      (highDifferenceExpr offset).eval (completeBits interface env offset) := by
  apply Expr.eval_eq_of_agree_below _ (offset + bitCount) _ _ (highDifference_varsBelow offset)
  intro index below
  exact executeHints_agrees_below _ _ _ index below

theorem completeEnv_flag (interface : Interface) (env : Env) (offset : Nat) :
    completeEnv interface env offset (offset + bitCount + 1) =
      1 - (highDifferenceExpr offset).eval (completeBits interface env offset) *
        Hint.inverse ((highDifferenceExpr offset).eval (completeBits interface env offset)) := by
  simp only [completeEnv, executeRecipes, Env.set_self]
  rw [flagRecipe, Expr.eval_sub]
  change (1 : F) - (highDifferenceExpr offset).eval (completeInverse interface env offset) *
    (inverseExpr offset).eval (completeInverse interface env offset) = _
  rw [inverse_highDifference]
  simp [inverseExpr, completeInverse, executeHints, Env.set, Expr.eval, inverseHint, Hint.eval]

private theorem completeBits_highDifference_congr
    (leftInterface rightInterface : Interface) (left right : Env) (leftOffset rightOffset : Nat)
    (leftInputs : Assumptions leftInterface leftOffset left)
    (rightInputs : Assumptions rightInterface rightOffset right)
    (source : (leftInterface.source leftOffset).eval left = (rightInterface.source rightOffset).eval right) :
    (highDifferenceExpr leftOffset).eval (completeBits leftInterface left leftOffset) =
      (highDifferenceExpr rightOffset).eval (completeBits rightInterface right rightOffset) := by
  simp only [highDifferenceExpr, Expr.eval_sub, Expr.eval_const, highExpr, weightedExpr_eval]
  rw [completeBits_weightedValue leftInterface left leftOffset halfBitCount halfBitCount leftInputs (by decide),
    completeBits_weightedValue rightInterface right rightOffset halfBitCount halfBitCount rightInputs (by decide),
    source]

/-- The complete 66-cell execution depends only on the input word. -/
theorem completeEnv_local_congr
    (leftInterface rightInterface : Interface) (left right : Env) (leftOffset rightOffset : Nat)
    (leftInputs : Assumptions leftInterface leftOffset left)
    (rightInputs : Assumptions rightInterface rightOffset right)
    (source : (leftInterface.source leftOffset).eval left = (rightInterface.source rightOffset).eval right)
    (index : Nat) (bounded : index < auxiliaryCount) :
    completeEnv leftInterface left leftOffset (leftOffset + index) =
      completeEnv rightInterface right rightOffset (rightOffset + index) := by
  by_cases bit : index < bitCount
  · apply Fin.ext
    change bitValue (completeEnv leftInterface left leftOffset) leftOffset index =
      bitValue (completeEnv rightInterface right rightOffset) rightOffset index
    rw [completeEnv_bitValue leftInterface left leftOffset index leftInputs bit,
      completeEnv_bitValue rightInterface right rightOffset index rightInputs bit, source]
  · have high := completeBits_highDifference_congr leftInterface rightInterface left right
      leftOffset rightOffset leftInputs rightInputs source
    have position : index = bitCount ∨ index = bitCount + 1 := by
      change index < 66 at bounded
      change ¬ index < 64 at bit
      change index = 64 ∨ index = 64 + 1
      omega
    rcases position with rfl | rfl
    · rw [completeEnv_inverse, completeEnv_inverse, high]
    · simp only [← Nat.add_assoc, completeEnv_flag, high]

end NightstreamFPrime.Gadgets.Range.CanonicalU64
