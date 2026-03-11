import SuperNeo.MLE

/-!
Contract interface for `SuperNeo.MLE`.

Spec: `specs/MLE.spec.md`

Paper anchors:
- Section 4, line 273: `ṽ(X) = Σ_j eq(X,j) · v_j` — MLE definition.
- Definition 6, Section 4, lines 352-355: sum-check uses MLE claims.
- Section 7.3, lines 440-470: MLE evaluation in folding.
-/

namespace SuperNeo

namespace MLEInterface

/-! ## Core Surfaces -/

/-- [Role: Definitional] Bit predicate over `F`. -/
abbrev IsBit := SuperNeo.IsBit

/-- [Role: Definitional] Bit-vector predicate over arrays. -/
abbrev IsBitVec := SuperNeo.IsBitVec

/-- [Role: Definitional] Index-mask embedding to a field bit vector. -/
abbrev bitsToFieldArray := SuperNeo.bitsToFieldArray

/-- [Role: Definitional] Guarded executable MLE evaluator. -/
abbrev mleEval := SuperNeo.mleEval

/-- [Role: Definitional] Unguarded sum-form MLE expression. -/
abbrev mleInnerProductForm := SuperNeo.mleInnerProductForm

/-- [Role: Definitional] Single basis-weight selector `χ_r(j)`. -/
abbrev chiWeight := SuperNeo.chiWeight

/-- [Role: Definitional] Compatibility `rHat` vector. -/
abbrev rHat := SuperNeo.rHat

/-- [Role: Definitional] Compatibility evaluator (inner-product route). -/
abbrev mleByInnerProduct := SuperNeo.mleByInnerProduct

/-- [Role: Definitional] Executable iterative folding evaluator. -/
abbrev mleByFoldingExec := SuperNeo.mleByFoldingExec

/-- [Role: Definitional] Theorem-facing folding evaluator. -/
abbrev mleByFolding := SuperNeo.mleByFolding

/-- [Role: Definitional] Canonical chi vector indexed by Boolean-cube masks. -/
abbrev chi := SuperNeo.chi

/-- [Role: Definitional] Dot-product surface used by chi-form MLE. -/
abbrev dot := SuperNeo.dot

/-- [Role: Definitional] MLE via dot-product with chi weights. -/
abbrev mleViaChiDot := SuperNeo.mleViaChiDot

/-- [Role: Definitional] Pointwise table linear combination `f + δ*g`. -/
abbrev linComb := SuperNeo.linComb

/-! ## Proved Theorems -/

/-- [Role: Theorem-Target] Size-valid executable evaluator equals sum form. -/
abbrev mleEval_eq_innerProductForm_of_size
  {f r : Array F}
  (hSize : f.size = (2 ^ r.size)) :
  mleEval f r = mleInnerProductForm f r :=
  SuperNeo.mleEval_eq_innerProductForm_of_size hSize

/-- [Role: Theorem-Target] `rHat` has the requested output size. -/
abbrev rHat_size := SuperNeo.rHat_size

/-- [Role: Theorem-Target] `chi` has size `2 ^ r.size`. -/
abbrev chi_size := SuperNeo.chi_size

/-- [Role: Theorem-Target] `linComb` preserves the left-table size. -/
abbrev linComb_size := SuperNeo.linComb_size

/-- [Role: Theorem-Target] Package-level executable-vs-sum identity is closed. -/
abbrev mleIdentityAssumption_holds := SuperNeo.mleIdentityAssumption_holds

/-- [Role: Theorem-Target] Executable identity check is sound. -/
abbrev mleIdentity_sound := SuperNeo.mleIdentity_sound

/-- [Role: Theorem-Target] Executable identity check is complete. -/
abbrev mleIdentity_complete := SuperNeo.mleIdentity_complete

/-- [Role: Theorem-Target] Executable identity check iff theorem. -/
abbrev mleIdentity_eq_true_iff := SuperNeo.mleIdentity_eq_true_iff

/-- [Role: Theorem-Target] Size-guarded identity between theorem-facing inner and folding forms. -/
abbrev mleByInnerProduct_eq_mleByFolding_of_size :=
  SuperNeo.mleByInnerProduct_eq_mleByFolding_of_size

/-- [Role: Theorem-Target] Size-guarded sum-form equals chi/dot form. -/
abbrev mleInnerProductForm_eq_mleViaChiDot_of_size
  {f r : Array F}
  (hSize : f.size = (2 ^ r.size)) :
  mleInnerProductForm f r = mleViaChiDot f r :=
  SuperNeo.mleInnerProductForm_eq_mleViaChiDot_of_size hSize

/-- [Role: Theorem-Target] Package-level chi/dot identity is closed. -/
abbrev mleChiIdentityAssumption_holds := SuperNeo.mleChiIdentityAssumption_holds

/-- [Role: Theorem-Target] Derived guarded linearity under identity + inner linearity packages. -/
abbrev mleEval_linComb_of_assumptions := SuperNeo.mleEval_linComb_of_assumptions

/-! ## Boundary Targets (Definitional Carriers + Bridges) -/

/-- [Role: Definitional] Package target for executable-vs-sum identity. -/
abbrev mleIdentityAssumption := SuperNeo.mleIdentityAssumption

/-- [Role: Definitional] Package target for Boolean-cube delta behavior of `eqPoly`. -/
abbrev eqPolyDeltaOnBitsAssumption := SuperNeo.eqPolyDeltaOnBitsAssumption

/-- [Role: Theorem-Target] Bridge from `EqPoly.eqPolyAssumption` to MLE-local delta package. -/
abbrev eqPolyDeltaOnBitsAssumption_of_eqPolyAssumption :=
  SuperNeo.eqPolyDeltaOnBitsAssumption_of_eqPolyAssumption

/-- [Role: Theorem-Target] Canonical closure of MLE-local delta package from EqPoly selector boundary. -/
abbrev eqPolyDeltaOnBitsAssumption_holds_of_eqPolyAssumption :=
  SuperNeo.eqPolyDeltaOnBitsAssumption_holds_of_eqPolyAssumption

/-- [Role: Theorem-Target] Conditional delta theorem from the package target. -/
abbrev eqPoly_eq_delta_of_isBitVec_of_assumption := SuperNeo.eqPoly_eq_delta_of_isBitVec_of_assumption

/-- [Role: Definitional] Package target for sum-form equals chi/dot form. -/
abbrev mleChiIdentityAssumption := SuperNeo.mleChiIdentityAssumption

/-- [Role: Theorem-Target] Conditional chi/dot identity theorem from the package target. -/
abbrev mleInnerProductForm_eq_mleViaChiDot_of_size_of_assumption := SuperNeo.mleInnerProductForm_eq_mleViaChiDot_of_size_of_assumption

/-- [Role: Definitional] Package target for sum-form linearity in table input. -/
abbrev mleInnerProductLinearityAssumption := SuperNeo.mleInnerProductLinearityAssumption

/-- [Role: Theorem-Target] Canonical closure of inner-product-form linearity. -/
abbrev mleInnerProductLinearityAssumption_holds :=
  SuperNeo.mleInnerProductLinearityAssumption_holds

/-- [Role: Definitional] Package target for guarded evaluator linearity. -/
abbrev mleEvalLinearityAssumption := SuperNeo.mleEvalLinearityAssumption

/-- [Role: Theorem-Target] Conditional guarded linearity theorem from the package target. -/
abbrev mleEval_linComb_of_assumption := SuperNeo.mleEval_linComb_of_assumption

/-- [Role: Theorem-Target] Build guarded-linearity package from identity + inner-linearity packages. -/
abbrev mleEvalLinearityAssumption_of_assumptions :=
  SuperNeo.mleEvalLinearityAssumption_of_assumptions

/-- [Role: Theorem-Target] Canonical closure of guarded evaluator linearity. -/
abbrev mleEvalLinearityAssumption_holds :=
  SuperNeo.mleEvalLinearityAssumption_holds

end MLEInterface

end SuperNeo
