import SuperNeo.MatrixTransform
import SuperNeo.EvalHom
import SuperNeo.ModuleHom
import SuperNeo.InvertibilityAxioms
import SuperNeo.SamplingSet
import SuperNeo.PolyLemmas
import SuperNeo.Decomp
import SuperNeo.Interp

/-! Arithmetic bundle composition for P6/P12/P14/P15/P16/P17/P18/P19. -/


namespace SuperNeo

open F

/-- P14 consequence packaged as a proposition (evaluation homomorphism equality). -/
def p20EvalHomProp
  (bar : Array (Array F))
  (m : Array (Array F))
  (z1 z2 r : Array F)
  (ρ1 ρ2 : F) : Prop :=
  evalHom2Prop bar m z1 z2 r ρ1 ρ2

theorem p20EvalHomProp_of_assumption
  {bar : Array (Array F)}
  {m : Array (Array F)}
  {z1 z2 r : Array F}
  {ρ1 ρ2 : F}
  (hAssm : p14EvalHomAssumption bar m r ρ1 ρ2)
  (hSize : z1.size = z2.size)
  (hRows : MatrixRowsCompatible m z1) :
  p20EvalHomProp bar m z1 z2 r ρ1 ρ2 := by
  exact evalHom2Prop_of_assumption hAssm hSize hRows

theorem p20EvalHomProp_of_checkAssumption
  {bar : Array (Array F)}
  {m : Array (Array F)}
  {z1 z2 r : Array F}
  {ρ1 ρ2 : F}
  (hCheck : p14EvalHomCheckAssumption bar m r ρ1 ρ2)
  (hSize : z1.size = z2.size)
  (hRows : MatrixRowsCompatible m z1) :
  p20EvalHomProp bar m z1 z2 r ρ1 ρ2 := by
  exact p20EvalHomProp_of_assumption
    (p14EvalHomAssumption_of_checkAssumption hCheck) hSize hRows

theorem p20MatrixTransformProp_of_assumption
  {bar : Array (Array F)}
  {m : Array (Array F)}
  {z : Array F}
  (hAssm : p12MatrixTransformAssumption bar m)
  (hRows : MatrixRowsCompatible m z) :
  matrixVecDirect m z = matrixVecCtBar bar m z := by
  exact matrixTransformEq_of_assumption hAssm hRows

theorem p20MatrixTransformProp_of_checkAssumption
  {bar : Array (Array F)}
  {m : Array (Array F)}
  {z : Array F}
  (hCheck : p12MatrixTransformCheckAssumption bar m)
  (hRows : MatrixRowsCompatible m z) :
  matrixVecDirect m z = matrixVecCtBar bar m z := by
  exact p20MatrixTransformProp_of_assumption
    (p12MatrixTransformAssumption_of_checkAssumption hCheck) hRows

/-- P15 consequence packaged as vector-module linearity obligations. -/
def p20VecModuleProp (h : VecModuleHom) (s : F) (x y : Array F) : Prop :=
  h.map (vecAdd x y) = vecAdd (h.map x) (h.map y) ∧
    h.map (vecScale s x) = vecScale s (h.map x)

/-- P15 consequence packaged as scalar-module linearity obligations. -/
def p20ScalarModuleProp (h : ScalarModuleHom) (s : F) (x y : Array F) : Prop :=
  h.map (vecAdd x y) = h.map x + h.map y ∧
    h.map (vecScale s x) = s * h.map x

/-- P17 obligation packaged as a proposition-level expansion bound. -/
def p20SamplingProp (cset samples : Array Coeffs) : Prop :=
  empiricalExpansionFactor cset samples <= theorem9UpperBound (maxRhoNorm cset)

theorem p20SamplingProp_of_empirical_bound
  {cset samples : Array Coeffs}
  (hEmp : empiricalExpansionFactor cset samples <= theorem9UpperBound (maxRhoNorm cset)) :
  p20SamplingProp cset samples := by
  exact hEmp

theorem p20SamplingProp_of_operand_norm_assumptions
  {cset samples : Array Coeffs}
  {BA BB BRaw : Nat}
  (hCset : ∀ i : Fin cset.size, normInfCoeffs cset[i] ≤ BA)
  (hSamples : ∀ j : Fin samples.size, normInfCoeffs samples[j] ≤ BB)
  (hRaw : mulRqRawNormBoundFromOperands BA BB BRaw)
  (hAddSub : rawAddSubCollapseBound BRaw (theorem9UpperBound (maxRhoNorm cset)))
  (hSub : rawSubCollapseBound BRaw (theorem9UpperBound (maxRhoNorm cset))) :
  p20SamplingProp cset samples := by
  have hMax : maxRhoNorm cset ≤ BA := maxRhoNorm_le_of_forall_norm_le hCset
  have hRawAtMax : mulRqRawNormBoundFromOperands (maxRhoNorm cset) BB BRaw := by
    intro a b hA hB
    exact hRaw a b (Nat.le_trans hA hMax) hB
  exact p20SamplingProp_of_empirical_bound
    (empiricalExpansionFactor_le_theorem9UpperBound_of_operand_norm_assumptions
      (cset := cset) (samples := samples) (BB := BB) (BRaw := BRaw)
      hSamples hRawAtMax hAddSub hSub)

theorem p20SamplingProp_of_operand_norm_assumptions_inRange
  {cset samples : Array Coeffs}
  {BA BB BRaw : Nat}
  (hCset : ∀ i : Fin cset.size, normInfCoeffs cset[i] ≤ BA)
  (hSamples : ∀ j : Fin samples.size, normInfCoeffs samples[j] ≤ BB)
  (hRawInRange : mulRqRawInRangeBoundFromOperands BA BB BRaw)
  (hAddSub : rawAddSubCollapseBound BRaw (theorem9UpperBound (maxRhoNorm cset)))
  (hSub : rawSubCollapseBound BRaw (theorem9UpperBound (maxRhoNorm cset))) :
  p20SamplingProp cset samples := by
  have hMax : maxRhoNorm cset ≤ BA := maxRhoNorm_le_of_forall_norm_le hCset
  have hRawInRangeAtMax :
      mulRqRawInRangeBoundFromOperands (maxRhoNorm cset) BB BRaw := by
    intro a b t ht hA hB
    exact hRawInRange a b t ht (Nat.le_trans hA hMax) hB
  exact p20SamplingProp_of_empirical_bound
    (empiricalExpansionFactor_le_theorem9UpperBound_of_operand_norm_assumptions_inRange
      (cset := cset) (samples := samples) (BB := BB) (BRaw := BRaw)
      hSamples hRawInRangeAtMax hAddSub hSub)

theorem p20SamplingProp_of_operand_norm_assumptions_rawCoeff
  {cset samples : Array Coeffs}
  {BA BB BRaw : Nat}
  (hCset : ∀ i : Fin cset.size, normInfCoeffs cset[i] ≤ BA)
  (hSamples : ∀ j : Fin samples.size, normInfCoeffs samples[j] ≤ BB)
  (hRawCoeff : mulRqRawCoeffBoundFromOperands BA BB BRaw)
  (hAddSub : rawAddSubCollapseBound BRaw (theorem9UpperBound (maxRhoNorm cset)))
  (hSub : rawSubCollapseBound BRaw (theorem9UpperBound (maxRhoNorm cset))) :
  p20SamplingProp cset samples := by
  have hMax : maxRhoNorm cset ≤ BA := maxRhoNorm_le_of_forall_norm_le hCset
  have hRawCoeffAtMax :
      mulRqRawCoeffBoundFromOperands (maxRhoNorm cset) BB BRaw := by
    intro a b t hA hB
    exact hRawCoeff a b t (Nat.le_trans hA hMax) hB
  exact p20SamplingProp_of_empirical_bound
    (empiricalExpansionFactor_le_theorem9UpperBound_of_operand_norm_assumptions
      (cset := cset) (samples := samples) (BB := BB) (BRaw := BRaw)
      hSamples
      (mulRqRawNormBoundFromOperands_of_rawCoeff hRawCoeffAtMax)
      hAddSub hSub)

theorem p20SamplingProp_of_operand_norm_assumptions_fieldOp
  {cset samples : Array Coeffs}
  {BA BB : Nat}
  (hCset : ∀ i : Fin cset.size, normInfCoeffs cset[i] ≤ BA)
  (hSamples : ∀ j : Fin samples.size, normInfCoeffs samples[j] ≤ BB)
  (hRaw :
    mulRqRawNormBoundFromOperands BA BB (theorem9UpperBound (maxRhoNorm cset)))
  (hOps :
    rawFieldOpCollapseBound
      (theorem9UpperBound (maxRhoNorm cset))
      (theorem9UpperBound (maxRhoNorm cset))) :
  p20SamplingProp cset samples := by
  have hMax : maxRhoNorm cset ≤ BA := maxRhoNorm_le_of_forall_norm_le hCset
  have hRawAtMax :
      mulRqRawNormBoundFromOperands
        (maxRhoNorm cset) BB (theorem9UpperBound (maxRhoNorm cset)) := by
    intro a b hA hB
    exact hRaw a b (Nat.le_trans hA hMax) hB
  exact p20SamplingProp_of_empirical_bound
    (empiricalExpansionFactor_le_of_operand_norm_assumptions_fieldOp
      (cset := cset) (samples := samples)
      (BA := maxRhoNorm cset) (BB := BB)
      (B := theorem9UpperBound (maxRhoNorm cset))
      (hCset := fun i => normInfCoeffs_le_maxRhoNorm cset i)
      (hSamples := hSamples)
      (hRaw := hRawAtMax)
      (hOps := hOps))

theorem p20SamplingProp_of_operand_norm_assumptions_fieldOp_inRange
  {cset samples : Array Coeffs}
  {BA BB : Nat}
  (hCset : ∀ i : Fin cset.size, normInfCoeffs cset[i] ≤ BA)
  (hSamples : ∀ j : Fin samples.size, normInfCoeffs samples[j] ≤ BB)
  (hRawInRange :
    mulRqRawInRangeBoundFromOperands BA BB (theorem9UpperBound (maxRhoNorm cset)))
  (hOps :
    rawFieldOpCollapseBound
      (theorem9UpperBound (maxRhoNorm cset))
      (theorem9UpperBound (maxRhoNorm cset))) :
  p20SamplingProp cset samples := by
  have hMax : maxRhoNorm cset ≤ BA := maxRhoNorm_le_of_forall_norm_le hCset
  have hRawInRangeAtMax :
      mulRqRawInRangeBoundFromOperands
        (maxRhoNorm cset) BB (theorem9UpperBound (maxRhoNorm cset)) := by
    intro a b t ht hA hB
    exact hRawInRange a b t ht (Nat.le_trans hA hMax) hB
  exact p20SamplingProp_of_empirical_bound
    (empiricalExpansionFactor_le_of_operand_norm_assumptions_fieldOp_inRange
      (cset := cset) (samples := samples)
      (BA := maxRhoNorm cset) (BB := BB)
      (B := theorem9UpperBound (maxRhoNorm cset))
      (hCset := fun i => normInfCoeffs_le_maxRhoNorm cset i)
      (hSamples := hSamples)
      (hRawInRange := hRawInRangeAtMax)
      (hOps := hOps))

/-- Helper: lift a Goldilocks-sized empirical expansion bound into `p20SamplingProp`. -/
theorem p20SamplingProp_of_goldilocks_empirical_le
  {cset samples : Array Coeffs}
  (hEmp : empiricalExpansionFactor cset samples ≤ Parameters.Goldilocks.B)
  (hUpper : Parameters.Goldilocks.B ≤ theorem9UpperBound (maxRhoNorm cset)) :
  p20SamplingProp cset samples := by
  apply p20SamplingProp_of_empirical_bound
  exact Nat.le_trans hEmp hUpper

theorem p20SamplingProp_of_operand_norm_assumptions_fieldOp_rawCoeff
  {cset samples : Array Coeffs}
  {BA BB : Nat}
  (hCset : ∀ i : Fin cset.size, normInfCoeffs cset[i] ≤ BA)
  (hSamples : ∀ j : Fin samples.size, normInfCoeffs samples[j] ≤ BB)
  (hRawCoeff :
    mulRqRawCoeffBoundFromOperands BA BB (theorem9UpperBound (maxRhoNorm cset)))
  (hOps :
    rawFieldOpCollapseBound
      (theorem9UpperBound (maxRhoNorm cset))
      (theorem9UpperBound (maxRhoNorm cset))) :
  p20SamplingProp cset samples := by
  have hMax : maxRhoNorm cset ≤ BA := maxRhoNorm_le_of_forall_norm_le hCset
  have hRawCoeffAtMax :
      mulRqRawCoeffBoundFromOperands
        (maxRhoNorm cset) BB (theorem9UpperBound (maxRhoNorm cset)) := by
    intro a b t hA hB
    exact hRawCoeff a b t (Nat.le_trans hA hMax) hB
  exact p20SamplingProp_of_empirical_bound
    (empiricalExpansionFactor_le_of_operand_norm_assumptions_fieldOp_rawCoeff
      (cset := cset) (samples := samples)
      (BA := maxRhoNorm cset) (BB := BB)
      (B := theorem9UpperBound (maxRhoNorm cset))
      (hCset := fun i => normInfCoeffs_le_maxRhoNorm cset i)
      (hSamples := hSamples)
      (hRawCoeff := hRawCoeffAtMax)
      (hOps := hOps))

theorem p20SamplingProp_of_goldilocks_operand_assumptions
  {cset samples : Array Coeffs}
  (hCset : ∀ i : Fin cset.size, normInfCoeffs cset[i] ≤ Parameters.Goldilocks.B)
  (hSamples : ∀ j : Fin samples.size, normInfCoeffs samples[j] ≤ Parameters.Goldilocks.B)
  (hRaw : GoldilocksRawNormBoundAssumption)
  (hCollapse : GoldilocksRawCollapseAssumption)
  (hUpper : Parameters.Goldilocks.B ≤ theorem9UpperBound (maxRhoNorm cset)) :
  p20SamplingProp cset samples := by
  exact p20SamplingProp_of_goldilocks_empirical_le
    (empiricalExpansionFactor_le_of_goldilocks_operand_assumptions hCset hSamples hRaw hCollapse)
    hUpper

theorem p20SamplingProp_of_goldilocks_operand_assumptions_inRange
  {cset samples : Array Coeffs}
  (hCset : ∀ i : Fin cset.size, normInfCoeffs cset[i] ≤ Parameters.Goldilocks.B)
  (hSamples : ∀ j : Fin samples.size, normInfCoeffs samples[j] ≤ Parameters.Goldilocks.B)
  (hRawInRange : GoldilocksRawInRangeBoundAssumption)
  (hCollapse : GoldilocksRawCollapseAssumption)
  (hUpper : Parameters.Goldilocks.B ≤ theorem9UpperBound (maxRhoNorm cset)) :
  p20SamplingProp cset samples := by
  exact p20SamplingProp_of_goldilocks_empirical_le
    (empiricalExpansionFactor_le_of_goldilocks_operand_assumptions_inRange
      hCset hSamples hRawInRange hCollapse)
    hUpper

theorem p20SamplingProp_of_goldilocks_operand_fieldOp_assumptions
  {cset samples : Array Coeffs}
  (hCset : ∀ i : Fin cset.size, normInfCoeffs cset[i] ≤ Parameters.Goldilocks.B)
  (hSamples : ∀ j : Fin samples.size, normInfCoeffs samples[j] ≤ Parameters.Goldilocks.B)
  (hRaw : GoldilocksRawNormBoundAssumption)
  (hFieldOps : GoldilocksFieldOpCollapseAssumption)
  (hUpper : Parameters.Goldilocks.B ≤ theorem9UpperBound (maxRhoNorm cset)) :
  p20SamplingProp cset samples := by
  exact p20SamplingProp_of_goldilocks_empirical_le
    (empiricalExpansionFactor_le_of_goldilocks_operand_fieldOp_assumptions
      hCset hSamples hRaw hFieldOps)
    hUpper

theorem p20SamplingProp_of_goldilocks_operand_fieldOp_assumptions_inRange
  {cset samples : Array Coeffs}
  (hCset : ∀ i : Fin cset.size, normInfCoeffs cset[i] ≤ Parameters.Goldilocks.B)
  (hSamples : ∀ j : Fin samples.size, normInfCoeffs samples[j] ≤ Parameters.Goldilocks.B)
  (hRawInRange : GoldilocksRawInRangeBoundAssumption)
  (hFieldOps : GoldilocksFieldOpCollapseAssumption)
  (hUpper : Parameters.Goldilocks.B ≤ theorem9UpperBound (maxRhoNorm cset)) :
  p20SamplingProp cset samples := by
  exact p20SamplingProp_of_goldilocks_empirical_le
    (empiricalExpansionFactor_le_of_goldilocks_operand_fieldOp_assumptions_inRange
      hCset hSamples hRawInRange hFieldOps)
    hUpper

theorem p20SamplingProp_of_goldilocks_operand_rawCoeff_assumptions
  {cset samples : Array Coeffs}
  (hCset : ∀ i : Fin cset.size, normInfCoeffs cset[i] ≤ Parameters.Goldilocks.B)
  (hSamples : ∀ j : Fin samples.size, normInfCoeffs samples[j] ≤ Parameters.Goldilocks.B)
  (hRawCoeff : GoldilocksRawCoeffBoundAssumption)
  (hCollapse : GoldilocksRawCollapseAssumption)
  (hUpper : Parameters.Goldilocks.B ≤ theorem9UpperBound (maxRhoNorm cset)) :
  p20SamplingProp cset samples := by
  apply p20SamplingProp_of_empirical_bound
  exact Nat.le_trans
    (empiricalExpansionFactor_le_of_goldilocks_operand_rawCoeff_assumptions
      hCset hSamples hRawCoeff hCollapse)
    hUpper

theorem p20SamplingProp_of_goldilocks_operand_rawCoeff_fieldOp_assumptions
  {cset samples : Array Coeffs}
  (hCset : ∀ i : Fin cset.size, normInfCoeffs cset[i] ≤ Parameters.Goldilocks.B)
  (hSamples : ∀ j : Fin samples.size, normInfCoeffs samples[j] ≤ Parameters.Goldilocks.B)
  (hRawCoeff : GoldilocksRawCoeffBoundAssumption)
  (hFieldOps : GoldilocksFieldOpCollapseAssumption)
  (hUpper : Parameters.Goldilocks.B ≤ theorem9UpperBound (maxRhoNorm cset)) :
  p20SamplingProp cset samples := by
  apply p20SamplingProp_of_empirical_bound
  exact Nat.le_trans
    (empiricalExpansionFactor_le_of_goldilocks_operand_rawCoeff_fieldOp_assumptions
      hCset hSamples hRawCoeff hFieldOps)
    hUpper

theorem p20SamplingProp_of_goldilocks_operand_assumptions_of_maxRhoNorm_ge
  {cset samples : Array Coeffs}
  (hCset : ∀ i : Fin cset.size, normInfCoeffs cset[i] ≤ Parameters.Goldilocks.B)
  (hSamples : ∀ j : Fin samples.size, normInfCoeffs samples[j] ≤ Parameters.Goldilocks.B)
  (hRaw : GoldilocksRawNormBoundAssumption)
  (hCollapse : GoldilocksRawCollapseAssumption)
  (hMax : Parameters.Goldilocks.B ≤ maxRhoNorm cset) :
  p20SamplingProp cset samples := by
  exact p20SamplingProp_of_goldilocks_operand_assumptions
    hCset hSamples hRaw hCollapse
    (goldilocksB_le_theorem9UpperBound_of_maxRhoNorm_ge hMax)

theorem p20SamplingProp_of_goldilocks_operand_assumptions_inRange_of_maxRhoNorm_ge
  {cset samples : Array Coeffs}
  (hCset : ∀ i : Fin cset.size, normInfCoeffs cset[i] ≤ Parameters.Goldilocks.B)
  (hSamples : ∀ j : Fin samples.size, normInfCoeffs samples[j] ≤ Parameters.Goldilocks.B)
  (hRawInRange : GoldilocksRawInRangeBoundAssumption)
  (hCollapse : GoldilocksRawCollapseAssumption)
  (hMax : Parameters.Goldilocks.B ≤ maxRhoNorm cset) :
  p20SamplingProp cset samples := by
  exact p20SamplingProp_of_goldilocks_operand_assumptions_inRange
    hCset hSamples hRawInRange hCollapse
    (goldilocksB_le_theorem9UpperBound_of_maxRhoNorm_ge hMax)

theorem p20SamplingProp_of_goldilocks_operand_fieldOp_assumptions_of_maxRhoNorm_ge
  {cset samples : Array Coeffs}
  (hCset : ∀ i : Fin cset.size, normInfCoeffs cset[i] ≤ Parameters.Goldilocks.B)
  (hSamples : ∀ j : Fin samples.size, normInfCoeffs samples[j] ≤ Parameters.Goldilocks.B)
  (hRaw : GoldilocksRawNormBoundAssumption)
  (hFieldOps : GoldilocksFieldOpCollapseAssumption)
  (hMax : Parameters.Goldilocks.B ≤ maxRhoNorm cset) :
  p20SamplingProp cset samples := by
  exact p20SamplingProp_of_goldilocks_operand_fieldOp_assumptions
    hCset hSamples hRaw hFieldOps
    (goldilocksB_le_theorem9UpperBound_of_maxRhoNorm_ge hMax)

theorem p20SamplingProp_of_goldilocks_operand_fieldOp_assumptions_inRange_of_maxRhoNorm_ge
  {cset samples : Array Coeffs}
  (hCset : ∀ i : Fin cset.size, normInfCoeffs cset[i] ≤ Parameters.Goldilocks.B)
  (hSamples : ∀ j : Fin samples.size, normInfCoeffs samples[j] ≤ Parameters.Goldilocks.B)
  (hRawInRange : GoldilocksRawInRangeBoundAssumption)
  (hFieldOps : GoldilocksFieldOpCollapseAssumption)
  (hMax : Parameters.Goldilocks.B ≤ maxRhoNorm cset) :
  p20SamplingProp cset samples := by
  exact p20SamplingProp_of_goldilocks_operand_fieldOp_assumptions_inRange
    hCset hSamples hRawInRange hFieldOps
    (goldilocksB_le_theorem9UpperBound_of_maxRhoNorm_ge hMax)

theorem p20SamplingProp_of_goldilocks_operand_rawCoeff_assumptions_of_maxRhoNorm_ge
  {cset samples : Array Coeffs}
  (hCset : ∀ i : Fin cset.size, normInfCoeffs cset[i] ≤ Parameters.Goldilocks.B)
  (hSamples : ∀ j : Fin samples.size, normInfCoeffs samples[j] ≤ Parameters.Goldilocks.B)
  (hRawCoeff : GoldilocksRawCoeffBoundAssumption)
  (hCollapse : GoldilocksRawCollapseAssumption)
  (hMax : Parameters.Goldilocks.B ≤ maxRhoNorm cset) :
  p20SamplingProp cset samples := by
  exact p20SamplingProp_of_goldilocks_operand_rawCoeff_assumptions
    hCset hSamples hRawCoeff hCollapse
    (goldilocksB_le_theorem9UpperBound_of_maxRhoNorm_ge hMax)

theorem p20SamplingProp_of_goldilocks_operand_rawCoeff_fieldOp_assumptions_of_maxRhoNorm_ge
  {cset samples : Array Coeffs}
  (hCset : ∀ i : Fin cset.size, normInfCoeffs cset[i] ≤ Parameters.Goldilocks.B)
  (hSamples : ∀ j : Fin samples.size, normInfCoeffs samples[j] ≤ Parameters.Goldilocks.B)
  (hRawCoeff : GoldilocksRawCoeffBoundAssumption)
  (hFieldOps : GoldilocksFieldOpCollapseAssumption)
  (hMax : Parameters.Goldilocks.B ≤ maxRhoNorm cset) :
  p20SamplingProp cset samples := by
  exact p20SamplingProp_of_goldilocks_operand_rawCoeff_fieldOp_assumptions
    hCset hSamples hRawCoeff hFieldOps
    (goldilocksB_le_theorem9UpperBound_of_maxRhoNorm_ge hMax)

/-- P16 window obligation packaged as a proposition. -/
def p20InvertibilityWindowProp (delta : Coeffs) : Prop :=
  withinInvertibilityWindow delta = true

/-- P18 polynomial obligations packaged as proposition-level constraints. -/
def p20PolyProp (qVals : Array F) (ell totalDegree setSize : Nat) : Prop :=
  eqLiftAllBoolean qVals ell = true ∧ setSize ≠ 0 ∧ totalDegree <= setSize

/-- P6 decomposition obligation packaged as a proposition. -/
def p20DecompProp (z : Array F) (b k : Nat) : Prop :=
  b ≥ 2 ∧
    let digits := splitBalancedVec z b k
    recomposeSplitDigits digits b = z ∧ digitsWithinBaseProp digits b

theorem p20DecompProp_base_ge_two
  {z : Array F} {b k : Nat}
  (hProp : p20DecompProp z b k) :
  b ≥ 2 := by
  exact hProp.1

theorem p20DecompProp_recompose_eq
  {z : Array F} {b k : Nat}
  (hProp : p20DecompProp z b k) :
  recomposeSplitDigits (splitBalancedVec z b k) b = z := by
  exact hProp.2.1

theorem p20DecompProp_recompose_size_eq
  {z : Array F} {b k : Nat}
  (hProp : p20DecompProp z b k) :
  (recomposeSplitDigits (splitBalancedVec z b k) b).size = z.size := by
  simpa [p20DecompProp_recompose_eq hProp]

theorem p20DecompProp_digitsWithinBaseProp
  {z : Array F} {b k : Nat}
  (hProp : p20DecompProp z b k) :
  digitsWithinBaseProp (splitBalancedVec z b k) b := by
  exact hProp.2.2

theorem p20DecompProp_digits_size
  {z : Array F} {b k : Nat}
  (_hProp : p20DecompProp z b k) :
  (splitBalancedVec z b k).size = k := by
  exact splitBalancedVec_size z b k

theorem p20DecompProp_digit_bound
  {z : Array F} {b k i j : Nat}
  (hProp : p20DecompProp z b k)
  (hi : i < (splitBalancedVec z b k).size)
  (hj : j < ((splitBalancedVec z b k)[i]'hi).size) :
  normInfF (((splitBalancedVec z b k)[i]'hi)[j]'hj) < b := by
  exact p20DecompProp_digitsWithinBaseProp hProp i hi j hj

theorem p20DecompProp_digit_row_size
  {z : Array F} {b k i : Nat}
  (_hProp : p20DecompProp z b k)
  (hi : i < (splitBalancedVec z b k).size) :
  ((splitBalancedVec z b k)[i]'hi).size = z.size := by
  exact splitBalancedVec_row_size hi

theorem p20DecompProp_of_splitRoundTrip
  {z : Array F} {b k : Nat}
  (hOk : splitRoundTrip z b k = true) :
  p20DecompProp z b k := by
  exact splitRoundTrip_sound_prop hOk

theorem splitRoundTrip_of_p20DecompProp
  {z : Array F} {b k : Nat}
  (hProp : p20DecompProp z b k) :
  splitRoundTrip z b k = true := by
  exact splitRoundTrip_complete_prop hProp

theorem p20DecompProp_iff_splitRoundTrip
  {z : Array F} {b k : Nat} :
  p20DecompProp z b k ↔ splitRoundTrip z b k = true := by
  constructor
  · exact splitRoundTrip_of_p20DecompProp
  · exact p20DecompProp_of_splitRoundTrip

/-- P19 interpolation obligation packaged as a proposition. -/
def p20InterpProp
  (xs ys expectedCoeffs : Array F)
  (evalPoint expectedEval : F) : Prop :=
  let coeffs := interpolateFromEvals xs ys
  coeffs = expectedCoeffs ∧ polyEval coeffs evalPoint = expectedEval

/--
P20 arithmetic bundle: composition obligations for P6/P12/P14/P15 plus
invertibility/sampling/polynomial/interpolation side conditions (P16/P17/P18/P19).
-/
def p20ArithmeticBundle
  (bar : Array (Array F))
  (m : Array (Array F))
  (z z1 z2 zDecomp r : Array F)
  (ρ1 ρ2 : F)
  (b k : Nat)
  (hVec : VecModuleHom)
  (hScal : ScalarModuleHom)
  (cset samples : Array Coeffs)
  (invDelta : Coeffs)
  (qVals : Array F)
  (xs ys expectedCoeffs : Array F)
  (evalPoint expectedEval : F)
  (ell totalDegree setSize : Nat) : Prop :=
  p20DecompProp zDecomp b k ∧
    MatrixRowsCompatible m z ∧
    matrixVecDirect m z = matrixVecCtBar bar m z ∧
    p20EvalHomProp bar m z1 z2 r ρ1 ρ2 ∧
    p20VecModuleProp hVec ρ1 z1 z2 ∧
    p20ScalarModuleProp hScal ρ1 z1 z2 ∧
    invertibilityPreconditionsProp ∧
    p20InvertibilityWindowProp invDelta ∧
    p20SamplingProp cset samples ∧
    p20PolyProp qVals ell totalDegree setSize ∧
    p20InterpProp xs ys expectedCoeffs evalPoint expectedEval

theorem p20ArithmeticBundle_decomp
  {bar : Array (Array F)}
  {m : Array (Array F)}
  {z z1 z2 zDecomp r : Array F}
  {ρ1 ρ2 : F}
  {b k : Nat}
  {hVec : VecModuleHom}
  {hScal : ScalarModuleHom}
  {cset samples : Array Coeffs}
  {invDelta : Coeffs}
  {qVals : Array F}
  {xs ys expectedCoeffs : Array F}
  {evalPoint expectedEval : F}
  {ell totalDegree setSize : Nat}
  (hP20 : p20ArithmeticBundle bar m z z1 z2 zDecomp r ρ1 ρ2 b k hVec hScal cset samples invDelta qVals xs ys expectedCoeffs evalPoint expectedEval ell totalDegree setSize) :
  p20DecompProp zDecomp b k := by
  exact hP20.1

theorem p20ArithmeticBundle_decomp_digit_row_size
  {bar : Array (Array F)}
  {m : Array (Array F)}
  {z z1 z2 zDecomp r : Array F}
  {ρ1 ρ2 : F}
  {b k i : Nat}
  {hVec : VecModuleHom}
  {hScal : ScalarModuleHom}
  {cset samples : Array Coeffs}
  {invDelta : Coeffs}
  {qVals : Array F}
  {xs ys expectedCoeffs : Array F}
  {evalPoint expectedEval : F}
  {ell totalDegree setSize : Nat}
  (hP20 : p20ArithmeticBundle bar m z z1 z2 zDecomp r ρ1 ρ2 b k hVec hScal cset samples invDelta qVals xs ys expectedCoeffs evalPoint expectedEval ell totalDegree setSize)
  (hi : i < (splitBalancedVec zDecomp b k).size) :
  ((splitBalancedVec zDecomp b k)[i]'hi).size = zDecomp.size := by
  exact p20DecompProp_digit_row_size (p20ArithmeticBundle_decomp hP20) hi

theorem p20ArithmeticBundle_matrixRows
  {bar : Array (Array F)}
  {m : Array (Array F)}
  {z z1 z2 zDecomp r : Array F}
  {ρ1 ρ2 : F}
  {b k : Nat}
  {hVec : VecModuleHom}
  {hScal : ScalarModuleHom}
  {cset samples : Array Coeffs}
  {invDelta : Coeffs}
  {qVals : Array F}
  {xs ys expectedCoeffs : Array F}
  {evalPoint expectedEval : F}
  {ell totalDegree setSize : Nat}
  (hP20 : p20ArithmeticBundle bar m z z1 z2 zDecomp r ρ1 ρ2 b k hVec hScal cset samples invDelta qVals xs ys expectedCoeffs evalPoint expectedEval ell totalDegree setSize) :
  MatrixRowsCompatible m z := by
  exact hP20.2.1

theorem p20ArithmeticBundle_matrixEq
  {bar : Array (Array F)}
  {m : Array (Array F)}
  {z z1 z2 zDecomp r : Array F}
  {ρ1 ρ2 : F}
  {b k : Nat}
  {hVec : VecModuleHom}
  {hScal : ScalarModuleHom}
  {cset samples : Array Coeffs}
  {invDelta : Coeffs}
  {qVals : Array F}
  {xs ys expectedCoeffs : Array F}
  {evalPoint expectedEval : F}
  {ell totalDegree setSize : Nat}
  (hP20 : p20ArithmeticBundle bar m z z1 z2 zDecomp r ρ1 ρ2 b k hVec hScal cset samples invDelta qVals xs ys expectedCoeffs evalPoint expectedEval ell totalDegree setSize) :
  matrixVecDirect m z = matrixVecCtBar bar m z := by
  exact hP20.2.2.1

theorem p20ArithmeticBundle_evalHom
  {bar : Array (Array F)}
  {m : Array (Array F)}
  {z z1 z2 zDecomp r : Array F}
  {ρ1 ρ2 : F}
  {b k : Nat}
  {hVec : VecModuleHom}
  {hScal : ScalarModuleHom}
  {cset samples : Array Coeffs}
  {invDelta : Coeffs}
  {qVals : Array F}
  {xs ys expectedCoeffs : Array F}
  {evalPoint expectedEval : F}
  {ell totalDegree setSize : Nat}
  (hP20 : p20ArithmeticBundle bar m z z1 z2 zDecomp r ρ1 ρ2 b k hVec hScal cset samples invDelta qVals xs ys expectedCoeffs evalPoint expectedEval ell totalDegree setSize) :
  p20EvalHomProp bar m z1 z2 r ρ1 ρ2 := by
  exact hP20.2.2.2.1

/-- Extract the invertibility-window obligation from a P20 bundle. -/
theorem p20ArithmeticBundle_invertibilityWindow
  {bar : Array (Array F)}
  {m : Array (Array F)}
  {z z1 z2 zDecomp r : Array F}
  {ρ1 ρ2 : F}
  {b k : Nat}
  {hVec : VecModuleHom}
  {hScal : ScalarModuleHom}
  {cset samples : Array Coeffs}
  {invDelta : Coeffs}
  {qVals : Array F}
  {xs ys expectedCoeffs : Array F}
  {evalPoint expectedEval : F}
  {ell totalDegree setSize : Nat}
  (hP20 :
    p20ArithmeticBundle bar m z z1 z2 zDecomp r ρ1 ρ2 b k hVec hScal
      cset samples invDelta qVals xs ys expectedCoeffs evalPoint expectedEval
      ell totalDegree setSize) :
  withinInvertibilityWindow invDelta = true := by
  rcases hP20 with
    ⟨_hP6, _hP12Rows, _hP12Eq, _hP14, _hP15Vec, _hP15Scal, _hP16, hWin, _hP17, _hP18, _hP19⟩
  exact hWin

theorem p20_pos_normInfCoeffs_of_invertibilityWindow
  {delta : Coeffs}
  (hWin : withinInvertibilityWindow delta = true) :
  0 < normInfCoeffs delta := by
  exact (withinInvertibilityWindow_sound hWin).1

theorem p20_pos_normInfCoeffs_mulRq_of_invertibilityWindow
  {invDelta aDelta bDelta : Coeffs}
  (hWin : withinInvertibilityWindow invDelta = true)
  (hDeltaEq : invDelta = mulRq aDelta bDelta) :
  0 < normInfCoeffs (mulRq aDelta bDelta) := by
  simpa [hDeltaEq] using
    (p20_pos_normInfCoeffs_of_invertibilityWindow (delta := invDelta) hWin)

/--
Proposition-native constructor for the P20 bundle.
-/
theorem p20ArithmeticBundle_of_props
  {bar : Array (Array F)}
  {m : Array (Array F)}
  {z z1 z2 zDecomp r : Array F}
  {ρ1 ρ2 : F}
  {b k : Nat}
  {hVec : VecModuleHom}
  {hScal : ScalarModuleHom}
  {cset samples : Array Coeffs}
  {invDelta : Coeffs}
  {qVals : Array F}
  {xs ys expectedCoeffs : Array F}
  {evalPoint expectedEval : F}
  {ell totalDegree setSize : Nat}
  (hP6 : p20DecompProp zDecomp b k)
  (hP12Rows : MatrixRowsCompatible m z)
  (hP12Eq : matrixVecDirect m z = matrixVecCtBar bar m z)
  (hP14 : p20EvalHomProp bar m z1 z2 r ρ1 ρ2)
  (hP15Vec : p20VecModuleProp hVec ρ1 z1 z2)
  (hP15Scal : p20ScalarModuleProp hScal ρ1 z1 z2)
  (hP16 : invertibilityPreconditionsProp)
  (hP16Win : p20InvertibilityWindowProp invDelta)
  (hP17 : p20SamplingProp cset samples)
  (hP18 : p20PolyProp qVals ell totalDegree setSize)
  (hP19 : p20InterpProp xs ys expectedCoeffs evalPoint expectedEval) :
  p20ArithmeticBundle bar m z z1 z2 zDecomp r ρ1 ρ2 b k hVec hScal cset samples invDelta qVals xs ys expectedCoeffs evalPoint expectedEval ell totalDegree setSize := by
  exact ⟨hP6, hP12Rows, hP12Eq, hP14, hP15Vec, hP15Scal, hP16, hP16Win, hP17, hP18, hP19⟩

/--
Theorem-native core constructor for the P20 bundle.

This is the single intended "theorem-first" entry point:
* P12 is supplied via `p12MatrixTransformAssumption`.
* P14 is supplied via `p14EvalHomAssumption`.

All other P20 constructors in this file are compatibility wrappers that either
package already-proved equalities/props or convert check-style assumptions into
these theorem-style interfaces.
-/
theorem p20ArithmeticBundle_of_assumptions
  {bar : Array (Array F)}
  {m : Array (Array F)}
  {z z1 z2 zDecomp r : Array F}
  {ρ1 ρ2 : F}
  {b k : Nat}
  {hVec : VecModuleHom}
  {hScal : ScalarModuleHom}
  {cset samples : Array Coeffs}
  {invDelta : Coeffs}
  {qVals : Array F}
  {xs ys expectedCoeffs : Array F}
  {evalPoint expectedEval : F}
  {ell totalDegree setSize : Nat}
  (hP6 : p20DecompProp zDecomp b k)
  (hP12Rows : MatrixRowsCompatible m z)
  (hP12Assm : p12MatrixTransformAssumption bar m)
  (hP14Assm : p14EvalHomAssumption bar m r ρ1 ρ2)
  (hP14Size : z1.size = z2.size)
  (hP14Rows : MatrixRowsCompatible m z1)
  (hP15Vec : p20VecModuleProp hVec ρ1 z1 z2)
  (hP15Scal : p20ScalarModuleProp hScal ρ1 z1 z2)
  (hP16 : invertibilityPreconditionsProp)
  (hP16Win : p20InvertibilityWindowProp invDelta)
  (hP17 : p20SamplingProp cset samples)
  (hP18 : p20PolyProp qVals ell totalDegree setSize)
  (hP19 : p20InterpProp xs ys expectedCoeffs evalPoint expectedEval) :
  p20ArithmeticBundle bar m z z1 z2 zDecomp r ρ1 ρ2 b k hVec hScal cset samples invDelta qVals
      xs ys expectedCoeffs evalPoint expectedEval ell totalDegree setSize := by
  exact p20ArithmeticBundle_of_props
    (hP6 := hP6)
    (hP12Rows := hP12Rows)
    (hP12Eq := p20MatrixTransformProp_of_assumption hP12Assm hP12Rows)
    (hP14 := p20EvalHomProp_of_assumption hP14Assm hP14Size hP14Rows)
    (hP15Vec := hP15Vec)
    (hP15Scal := hP15Scal)
    (hP16 := hP16)
    (hP16Win := hP16Win)
    (hP17 := hP17)
    (hP18 := hP18)
    (hP19 := hP19)

theorem p20ArithmeticBundle_of_props_with_sampling_operand_assumptions
  {bar : Array (Array F)}
  {m : Array (Array F)}
  {z z1 z2 zDecomp r : Array F}
  {ρ1 ρ2 : F}
  {b k : Nat}
  {hVec : VecModuleHom}
  {hScal : ScalarModuleHom}
  {cset samples : Array Coeffs}
  {invDelta : Coeffs}
  {qVals : Array F}
  {xs ys expectedCoeffs : Array F}
  {evalPoint expectedEval : F}
  {ell totalDegree setSize : Nat}
  {BA BB BRaw : Nat}
  (hP6 : p20DecompProp zDecomp b k)
  (hP12Rows : MatrixRowsCompatible m z)
  (hP12Eq : matrixVecDirect m z = matrixVecCtBar bar m z)
  (hP14 : p20EvalHomProp bar m z1 z2 r ρ1 ρ2)
  (hP15Vec : p20VecModuleProp hVec ρ1 z1 z2)
  (hP15Scal : p20ScalarModuleProp hScal ρ1 z1 z2)
  (hP16 : invertibilityPreconditionsProp)
  (hP16Win : p20InvertibilityWindowProp invDelta)
  (hCset : ∀ i : Fin cset.size, normInfCoeffs cset[i] ≤ BA)
  (hSamples : ∀ j : Fin samples.size, normInfCoeffs samples[j] ≤ BB)
  (hRaw : mulRqRawNormBoundFromOperands BA BB BRaw)
  (hAddSub : rawAddSubCollapseBound BRaw (theorem9UpperBound (maxRhoNorm cset)))
  (hSub : rawSubCollapseBound BRaw (theorem9UpperBound (maxRhoNorm cset)))
  (hP18 : p20PolyProp qVals ell totalDegree setSize)
  (hP19 : p20InterpProp xs ys expectedCoeffs evalPoint expectedEval) :
  p20ArithmeticBundle bar m z z1 z2 zDecomp r ρ1 ρ2 b k hVec hScal cset samples invDelta qVals xs ys expectedCoeffs evalPoint expectedEval ell totalDegree setSize := by
  exact p20ArithmeticBundle_of_props
    (hP6 := hP6)
    (hP12Rows := hP12Rows)
    (hP12Eq := hP12Eq)
    (hP14 := hP14)
    (hP15Vec := hP15Vec)
    (hP15Scal := hP15Scal)
    (hP16 := hP16)
    (hP16Win := hP16Win)
    (hP17 := p20SamplingProp_of_operand_norm_assumptions hCset hSamples hRaw hAddSub hSub)
    (hP18 := hP18)
    (hP19 := hP19)

theorem p20ArithmeticBundle_of_props_with_sampling_operand_fieldOp_assumptions
  {bar : Array (Array F)}
  {m : Array (Array F)}
  {z z1 z2 zDecomp r : Array F}
  {ρ1 ρ2 : F}
  {b k : Nat}
  {hVec : VecModuleHom}
  {hScal : ScalarModuleHom}
  {cset samples : Array Coeffs}
  {invDelta : Coeffs}
  {qVals : Array F}
  {xs ys expectedCoeffs : Array F}
  {evalPoint expectedEval : F}
  {ell totalDegree setSize : Nat}
  {BA BB : Nat}
  (hP6 : p20DecompProp zDecomp b k)
  (hP12Rows : MatrixRowsCompatible m z)
  (hP12Eq : matrixVecDirect m z = matrixVecCtBar bar m z)
  (hP14 : p20EvalHomProp bar m z1 z2 r ρ1 ρ2)
  (hP15Vec : p20VecModuleProp hVec ρ1 z1 z2)
  (hP15Scal : p20ScalarModuleProp hScal ρ1 z1 z2)
  (hP16 : invertibilityPreconditionsProp)
  (hP16Win : p20InvertibilityWindowProp invDelta)
  (hCset : ∀ i : Fin cset.size, normInfCoeffs cset[i] ≤ BA)
  (hSamples : ∀ j : Fin samples.size, normInfCoeffs samples[j] ≤ BB)
  (hRaw : mulRqRawNormBoundFromOperands BA BB (theorem9UpperBound (maxRhoNorm cset)))
  (hOps :
    rawFieldOpCollapseBound
      (theorem9UpperBound (maxRhoNorm cset))
      (theorem9UpperBound (maxRhoNorm cset)))
  (hP18 : p20PolyProp qVals ell totalDegree setSize)
  (hP19 : p20InterpProp xs ys expectedCoeffs evalPoint expectedEval) :
  p20ArithmeticBundle bar m z z1 z2 zDecomp r ρ1 ρ2 b k hVec hScal cset samples invDelta qVals xs ys expectedCoeffs evalPoint expectedEval ell totalDegree setSize := by
  exact p20ArithmeticBundle_of_props
    (hP6 := hP6)
    (hP12Rows := hP12Rows)
    (hP12Eq := hP12Eq)
    (hP14 := hP14)
    (hP15Vec := hP15Vec)
    (hP15Scal := hP15Scal)
    (hP16 := hP16)
    (hP16Win := hP16Win)
    (hP17 := p20SamplingProp_of_operand_norm_assumptions_fieldOp hCset hSamples hRaw hOps)
    (hP18 := hP18)
    (hP19 := hP19)

theorem p20ArithmeticBundle_of_props_with_sampling_operand_fieldOp_assumptions_inRange
  {bar : Array (Array F)}
  {m : Array (Array F)}
  {z z1 z2 zDecomp r : Array F}
  {ρ1 ρ2 : F}
  {b k : Nat}
  {hVec : VecModuleHom}
  {hScal : ScalarModuleHom}
  {cset samples : Array Coeffs}
  {invDelta : Coeffs}
  {qVals : Array F}
  {xs ys expectedCoeffs : Array F}
  {evalPoint expectedEval : F}
  {ell totalDegree setSize : Nat}
  {BA BB : Nat}
  (hP6 : p20DecompProp zDecomp b k)
  (hP12Rows : MatrixRowsCompatible m z)
  (hP12Eq : matrixVecDirect m z = matrixVecCtBar bar m z)
  (hP14 : p20EvalHomProp bar m z1 z2 r ρ1 ρ2)
  (hP15Vec : p20VecModuleProp hVec ρ1 z1 z2)
  (hP15Scal : p20ScalarModuleProp hScal ρ1 z1 z2)
  (hP16 : invertibilityPreconditionsProp)
  (hP16Win : p20InvertibilityWindowProp invDelta)
  (hCset : ∀ i : Fin cset.size, normInfCoeffs cset[i] ≤ BA)
  (hSamples : ∀ j : Fin samples.size, normInfCoeffs samples[j] ≤ BB)
  (hRawInRange :
    mulRqRawInRangeBoundFromOperands BA BB (theorem9UpperBound (maxRhoNorm cset)))
  (hOps :
    rawFieldOpCollapseBound
      (theorem9UpperBound (maxRhoNorm cset))
      (theorem9UpperBound (maxRhoNorm cset)))
  (hP18 : p20PolyProp qVals ell totalDegree setSize)
  (hP19 : p20InterpProp xs ys expectedCoeffs evalPoint expectedEval) :
  p20ArithmeticBundle bar m z z1 z2 zDecomp r ρ1 ρ2 b k hVec hScal cset samples invDelta qVals xs ys expectedCoeffs evalPoint expectedEval ell totalDegree setSize := by
  exact p20ArithmeticBundle_of_props
    (hP6 := hP6)
    (hP12Rows := hP12Rows)
    (hP12Eq := hP12Eq)
    (hP14 := hP14)
    (hP15Vec := hP15Vec)
    (hP15Scal := hP15Scal)
    (hP16 := hP16)
    (hP16Win := hP16Win)
    (hP17 := p20SamplingProp_of_operand_norm_assumptions_fieldOp_inRange hCset hSamples hRawInRange hOps)
    (hP18 := hP18)
    (hP19 := hP19)

theorem p20ArithmeticBundle_of_props_with_sampling_goldilocks_operand_assumptions
  {bar : Array (Array F)}
  {m : Array (Array F)}
  {z z1 z2 zDecomp r : Array F}
  {ρ1 ρ2 : F}
  {b k : Nat}
  {hVec : VecModuleHom}
  {hScal : ScalarModuleHom}
  {cset samples : Array Coeffs}
  {invDelta : Coeffs}
  {qVals : Array F}
  {xs ys expectedCoeffs : Array F}
  {evalPoint expectedEval : F}
  {ell totalDegree setSize : Nat}
  (hP6 : p20DecompProp zDecomp b k)
  (hP12Rows : MatrixRowsCompatible m z)
  (hP12Eq : matrixVecDirect m z = matrixVecCtBar bar m z)
  (hP14 : p20EvalHomProp bar m z1 z2 r ρ1 ρ2)
  (hP15Vec : p20VecModuleProp hVec ρ1 z1 z2)
  (hP15Scal : p20ScalarModuleProp hScal ρ1 z1 z2)
  (hP16 : invertibilityPreconditionsProp)
  (hP16Win : p20InvertibilityWindowProp invDelta)
  (hCset : ∀ i : Fin cset.size, normInfCoeffs cset[i] ≤ Parameters.Goldilocks.B)
  (hSamples : ∀ j : Fin samples.size, normInfCoeffs samples[j] ≤ Parameters.Goldilocks.B)
  (hRaw : GoldilocksRawNormBoundAssumption)
  (hCollapse : GoldilocksRawCollapseAssumption)
  (hUpper : Parameters.Goldilocks.B ≤ theorem9UpperBound (maxRhoNorm cset))
  (hP18 : p20PolyProp qVals ell totalDegree setSize)
  (hP19 : p20InterpProp xs ys expectedCoeffs evalPoint expectedEval) :
  p20ArithmeticBundle bar m z z1 z2 zDecomp r ρ1 ρ2 b k hVec hScal cset samples invDelta qVals xs ys expectedCoeffs evalPoint expectedEval ell totalDegree setSize := by
  exact p20ArithmeticBundle_of_props
    (hP6 := hP6)
    (hP12Rows := hP12Rows)
    (hP12Eq := hP12Eq)
    (hP14 := hP14)
    (hP15Vec := hP15Vec)
    (hP15Scal := hP15Scal)
    (hP16 := hP16)
    (hP16Win := hP16Win)
    (hP17 := p20SamplingProp_of_goldilocks_operand_assumptions hCset hSamples hRaw hCollapse hUpper)
    (hP18 := hP18)
    (hP19 := hP19)

theorem p20ArithmeticBundle_of_props_with_sampling_goldilocks_operand_fieldOp_assumptions
  {bar : Array (Array F)}
  {m : Array (Array F)}
  {z z1 z2 zDecomp r : Array F}
  {ρ1 ρ2 : F}
  {b k : Nat}
  {hVec : VecModuleHom}
  {hScal : ScalarModuleHom}
  {cset samples : Array Coeffs}
  {invDelta : Coeffs}
  {qVals : Array F}
  {xs ys expectedCoeffs : Array F}
  {evalPoint expectedEval : F}
  {ell totalDegree setSize : Nat}
  (hP6 : p20DecompProp zDecomp b k)
  (hP12Rows : MatrixRowsCompatible m z)
  (hP12Eq : matrixVecDirect m z = matrixVecCtBar bar m z)
  (hP14 : p20EvalHomProp bar m z1 z2 r ρ1 ρ2)
  (hP15Vec : p20VecModuleProp hVec ρ1 z1 z2)
  (hP15Scal : p20ScalarModuleProp hScal ρ1 z1 z2)
  (hP16 : invertibilityPreconditionsProp)
  (hP16Win : p20InvertibilityWindowProp invDelta)
  (hCset : ∀ i : Fin cset.size, normInfCoeffs cset[i] ≤ Parameters.Goldilocks.B)
  (hSamples : ∀ j : Fin samples.size, normInfCoeffs samples[j] ≤ Parameters.Goldilocks.B)
  (hRaw : GoldilocksRawNormBoundAssumption)
  (hFieldOps : GoldilocksFieldOpCollapseAssumption)
  (hUpper : Parameters.Goldilocks.B ≤ theorem9UpperBound (maxRhoNorm cset))
  (hP18 : p20PolyProp qVals ell totalDegree setSize)
  (hP19 : p20InterpProp xs ys expectedCoeffs evalPoint expectedEval) :
  p20ArithmeticBundle bar m z z1 z2 zDecomp r ρ1 ρ2 b k hVec hScal cset samples invDelta qVals xs ys expectedCoeffs evalPoint expectedEval ell totalDegree setSize := by
  exact p20ArithmeticBundle_of_props
    (hP6 := hP6)
    (hP12Rows := hP12Rows)
    (hP12Eq := hP12Eq)
    (hP14 := hP14)
    (hP15Vec := hP15Vec)
    (hP15Scal := hP15Scal)
    (hP16 := hP16)
    (hP16Win := hP16Win)
    (hP17 := p20SamplingProp_of_goldilocks_operand_fieldOp_assumptions hCset hSamples hRaw hFieldOps hUpper)
    (hP18 := hP18)
    (hP19 := hP19)

theorem p20ArithmeticBundle_of_props_with_sampling_goldilocks_operand_fieldOp_assumptions_inRange
  {bar : Array (Array F)}
  {m : Array (Array F)}
  {z z1 z2 zDecomp r : Array F}
  {ρ1 ρ2 : F}
  {b k : Nat}
  {hVec : VecModuleHom}
  {hScal : ScalarModuleHom}
  {cset samples : Array Coeffs}
  {invDelta : Coeffs}
  {qVals : Array F}
  {xs ys expectedCoeffs : Array F}
  {evalPoint expectedEval : F}
  {ell totalDegree setSize : Nat}
  (hP6 : p20DecompProp zDecomp b k)
  (hP12Rows : MatrixRowsCompatible m z)
  (hP12Eq : matrixVecDirect m z = matrixVecCtBar bar m z)
  (hP14 : p20EvalHomProp bar m z1 z2 r ρ1 ρ2)
  (hP15Vec : p20VecModuleProp hVec ρ1 z1 z2)
  (hP15Scal : p20ScalarModuleProp hScal ρ1 z1 z2)
  (hP16 : invertibilityPreconditionsProp)
  (hP16Win : p20InvertibilityWindowProp invDelta)
  (hCset : ∀ i : Fin cset.size, normInfCoeffs cset[i] ≤ Parameters.Goldilocks.B)
  (hSamples : ∀ j : Fin samples.size, normInfCoeffs samples[j] ≤ Parameters.Goldilocks.B)
  (hRawInRange : GoldilocksRawInRangeBoundAssumption)
  (hFieldOps : GoldilocksFieldOpCollapseAssumption)
  (hUpper : Parameters.Goldilocks.B ≤ theorem9UpperBound (maxRhoNorm cset))
  (hP18 : p20PolyProp qVals ell totalDegree setSize)
  (hP19 : p20InterpProp xs ys expectedCoeffs evalPoint expectedEval) :
  p20ArithmeticBundle bar m z z1 z2 zDecomp r ρ1 ρ2 b k hVec hScal cset samples invDelta qVals xs ys expectedCoeffs evalPoint expectedEval ell totalDegree setSize := by
  exact p20ArithmeticBundle_of_props
    (hP6 := hP6)
    (hP12Rows := hP12Rows)
    (hP12Eq := hP12Eq)
    (hP14 := hP14)
    (hP15Vec := hP15Vec)
    (hP15Scal := hP15Scal)
    (hP16 := hP16)
    (hP16Win := hP16Win)
    (hP17 := p20SamplingProp_of_goldilocks_operand_fieldOp_assumptions_inRange hCset hSamples hRawInRange hFieldOps hUpper)
    (hP18 := hP18)
    (hP19 := hP19)

theorem p20ArithmeticBundle_of_props_with_sampling_goldilocks_operand_rawCoeff_assumptions
  {bar : Array (Array F)}
  {m : Array (Array F)}
  {z z1 z2 zDecomp r : Array F}
  {ρ1 ρ2 : F}
  {b k : Nat}
  {hVec : VecModuleHom}
  {hScal : ScalarModuleHom}
  {cset samples : Array Coeffs}
  {invDelta : Coeffs}
  {qVals : Array F}
  {xs ys expectedCoeffs : Array F}
  {evalPoint expectedEval : F}
  {ell totalDegree setSize : Nat}
  (hP6 : p20DecompProp zDecomp b k)
  (hP12Rows : MatrixRowsCompatible m z)
  (hP12Eq : matrixVecDirect m z = matrixVecCtBar bar m z)
  (hP14 : p20EvalHomProp bar m z1 z2 r ρ1 ρ2)
  (hP15Vec : p20VecModuleProp hVec ρ1 z1 z2)
  (hP15Scal : p20ScalarModuleProp hScal ρ1 z1 z2)
  (hP16 : invertibilityPreconditionsProp)
  (hP16Win : p20InvertibilityWindowProp invDelta)
  (hCset : ∀ i : Fin cset.size, normInfCoeffs cset[i] ≤ Parameters.Goldilocks.B)
  (hSamples : ∀ j : Fin samples.size, normInfCoeffs samples[j] ≤ Parameters.Goldilocks.B)
  (hRawCoeff : GoldilocksRawCoeffBoundAssumption)
  (hCollapse : GoldilocksRawCollapseAssumption)
  (hUpper : Parameters.Goldilocks.B ≤ theorem9UpperBound (maxRhoNorm cset))
  (hP18 : p20PolyProp qVals ell totalDegree setSize)
  (hP19 : p20InterpProp xs ys expectedCoeffs evalPoint expectedEval) :
  p20ArithmeticBundle bar m z z1 z2 zDecomp r ρ1 ρ2 b k hVec hScal cset samples invDelta qVals xs ys expectedCoeffs evalPoint expectedEval ell totalDegree setSize := by
  exact p20ArithmeticBundle_of_props
    (hP6 := hP6)
    (hP12Rows := hP12Rows)
    (hP12Eq := hP12Eq)
    (hP14 := hP14)
    (hP15Vec := hP15Vec)
    (hP15Scal := hP15Scal)
    (hP16 := hP16)
    (hP16Win := hP16Win)
    (hP17 := p20SamplingProp_of_goldilocks_operand_rawCoeff_assumptions hCset hSamples hRawCoeff hCollapse hUpper)
    (hP18 := hP18)
    (hP19 := hP19)

theorem p20ArithmeticBundle_of_props_with_sampling_goldilocks_operand_rawCoeff_fieldOp_assumptions
  {bar : Array (Array F)}
  {m : Array (Array F)}
  {z z1 z2 zDecomp r : Array F}
  {ρ1 ρ2 : F}
  {b k : Nat}
  {hVec : VecModuleHom}
  {hScal : ScalarModuleHom}
  {cset samples : Array Coeffs}
  {invDelta : Coeffs}
  {qVals : Array F}
  {xs ys expectedCoeffs : Array F}
  {evalPoint expectedEval : F}
  {ell totalDegree setSize : Nat}
  (hP6 : p20DecompProp zDecomp b k)
  (hP12Rows : MatrixRowsCompatible m z)
  (hP12Eq : matrixVecDirect m z = matrixVecCtBar bar m z)
  (hP14 : p20EvalHomProp bar m z1 z2 r ρ1 ρ2)
  (hP15Vec : p20VecModuleProp hVec ρ1 z1 z2)
  (hP15Scal : p20ScalarModuleProp hScal ρ1 z1 z2)
  (hP16 : invertibilityPreconditionsProp)
  (hP16Win : p20InvertibilityWindowProp invDelta)
  (hCset : ∀ i : Fin cset.size, normInfCoeffs cset[i] ≤ Parameters.Goldilocks.B)
  (hSamples : ∀ j : Fin samples.size, normInfCoeffs samples[j] ≤ Parameters.Goldilocks.B)
  (hRawCoeff : GoldilocksRawCoeffBoundAssumption)
  (hFieldOps : GoldilocksFieldOpCollapseAssumption)
  (hUpper : Parameters.Goldilocks.B ≤ theorem9UpperBound (maxRhoNorm cset))
  (hP18 : p20PolyProp qVals ell totalDegree setSize)
  (hP19 : p20InterpProp xs ys expectedCoeffs evalPoint expectedEval) :
  p20ArithmeticBundle bar m z z1 z2 zDecomp r ρ1 ρ2 b k hVec hScal cset samples invDelta qVals xs ys expectedCoeffs evalPoint expectedEval ell totalDegree setSize := by
  exact p20ArithmeticBundle_of_props
    (hP6 := hP6)
    (hP12Rows := hP12Rows)
    (hP12Eq := hP12Eq)
    (hP14 := hP14)
    (hP15Vec := hP15Vec)
    (hP15Scal := hP15Scal)
    (hP16 := hP16)
    (hP16Win := hP16Win)
    (hP17 := p20SamplingProp_of_goldilocks_operand_rawCoeff_fieldOp_assumptions hCset hSamples hRawCoeff hFieldOps hUpper)
    (hP18 := hP18)
    (hP19 := hP19)

/--
Theorem-native P12 variant: derive the matrix-transform equality directly from
`thm3CoreAssumption` and row compatibility.
-/
theorem p20ArithmeticBundle_of_props_with_thm3CoreAssumption
  {bar : Array (Array F)}
  {m : Array (Array F)}
  {z z1 z2 zDecomp r : Array F}
  {ρ1 ρ2 : F}
  {b k : Nat}
  {hVec : VecModuleHom}
  {hScal : ScalarModuleHom}
  {cset samples : Array Coeffs}
  {invDelta : Coeffs}
  {qVals : Array F}
  {xs ys expectedCoeffs : Array F}
  {evalPoint expectedEval : F}
  {ell totalDegree setSize : Nat}
  (hP6 : p20DecompProp zDecomp b k)
  (hP12Rows : MatrixRowsCompatible m z)
  (hThm3 : thm3CoreAssumption bar)
  (hP14 : p20EvalHomProp bar m z1 z2 r ρ1 ρ2)
  (hP15Vec : p20VecModuleProp hVec ρ1 z1 z2)
  (hP15Scal : p20ScalarModuleProp hScal ρ1 z1 z2)
  (hP16 : invertibilityPreconditionsProp)
  (hP16Win : p20InvertibilityWindowProp invDelta)
  (hP17 : p20SamplingProp cset samples)
  (hP18 : p20PolyProp qVals ell totalDegree setSize)
  (hP19 : p20InterpProp xs ys expectedCoeffs evalPoint expectedEval) :
  p20ArithmeticBundle bar m z z1 z2 zDecomp r ρ1 ρ2 b k hVec hScal cset samples invDelta qVals xs ys expectedCoeffs evalPoint expectedEval ell totalDegree setSize := by
  have hP12Assm : p12MatrixTransformAssumption bar m :=
    p12MatrixTransformAssumption_of_thm3CoreAssumption hThm3
  exact p20ArithmeticBundle_of_props
    (hP6 := hP6)
    (hP12Rows := hP12Rows)
    (hP12Eq := p20MatrixTransformProp_of_assumption hP12Assm hP12Rows)
    (hP14 := hP14)
    (hP15Vec := hP15Vec)
    (hP15Scal := hP15Scal)
    (hP16 := hP16)
    (hP16Win := hP16Win)
    (hP17 := hP17)
    (hP18 := hP18)
    (hP19 := hP19)

/--
Theorem-native constructor path combining Theorem-3-derived P12 with
P14 supplied through the theorem-native assumption interface.
-/
theorem p20ArithmeticBundle_of_props_with_thm3CoreAssumption_with_evalHom_assumption
  {bar : Array (Array F)}
  {m : Array (Array F)}
  {z z1 z2 zDecomp r : Array F}
  {ρ1 ρ2 : F}
  {b k : Nat}
  {hVec : VecModuleHom}
  {hScal : ScalarModuleHom}
  {cset samples : Array Coeffs}
  {invDelta : Coeffs}
  {qVals : Array F}
  {xs ys expectedCoeffs : Array F}
  {evalPoint expectedEval : F}
  {ell totalDegree setSize : Nat}
  (hP6 : p20DecompProp zDecomp b k)
  (hP12Rows : MatrixRowsCompatible m z)
  (hThm3 : thm3CoreAssumption bar)
  (hP14Assm : p14EvalHomAssumption bar m r ρ1 ρ2)
  (hP14Size : z1.size = z2.size)
  (hP14Rows : MatrixRowsCompatible m z1)
  (hP15Vec : p20VecModuleProp hVec ρ1 z1 z2)
  (hP15Scal : p20ScalarModuleProp hScal ρ1 z1 z2)
  (hP16 : invertibilityPreconditionsProp)
  (hP16Win : p20InvertibilityWindowProp invDelta)
  (hP17 : p20SamplingProp cset samples)
  (hP18 : p20PolyProp qVals ell totalDegree setSize)
  (hP19 : p20InterpProp xs ys expectedCoeffs evalPoint expectedEval) :
  p20ArithmeticBundle bar m z z1 z2 zDecomp r ρ1 ρ2 b k hVec hScal cset samples invDelta qVals xs ys expectedCoeffs evalPoint expectedEval ell totalDegree setSize := by
  exact p20ArithmeticBundle_of_assumptions
    (hP6 := hP6)
    (hP12Rows := hP12Rows)
    (hP12Assm := p12MatrixTransformAssumption_of_thm3CoreAssumption hThm3)
    (hP14Assm := hP14Assm)
    (hP14Size := hP14Size)
    (hP14Rows := hP14Rows)
    (hP15Vec := hP15Vec)
    (hP15Scal := hP15Scal)
    (hP16 := hP16)
    (hP16Win := hP16Win)
    (hP17 := hP17)
    (hP18 := hP18)
    (hP19 := hP19)

/--
Theorem-native constructor path combining Theorem-3-derived P12 with
P14 supplied through the check-assumption interface.
-/
theorem p20ArithmeticBundle_of_props_with_thm3CoreAssumption_with_evalHom_checkAssumption
  {bar : Array (Array F)}
  {m : Array (Array F)}
  {z z1 z2 zDecomp r : Array F}
  {ρ1 ρ2 : F}
  {b k : Nat}
  {hVec : VecModuleHom}
  {hScal : ScalarModuleHom}
  {cset samples : Array Coeffs}
  {invDelta : Coeffs}
  {qVals : Array F}
  {xs ys expectedCoeffs : Array F}
  {evalPoint expectedEval : F}
  {ell totalDegree setSize : Nat}
  (hP6 : p20DecompProp zDecomp b k)
  (hP12Rows : MatrixRowsCompatible m z)
  (hThm3 : thm3CoreAssumption bar)
  (hP14Check : p14EvalHomCheckAssumption bar m r ρ1 ρ2)
  (hP14Size : z1.size = z2.size)
  (hP14Rows : MatrixRowsCompatible m z1)
  (hP15Vec : p20VecModuleProp hVec ρ1 z1 z2)
  (hP15Scal : p20ScalarModuleProp hScal ρ1 z1 z2)
  (hP16 : invertibilityPreconditionsProp)
  (hP16Win : p20InvertibilityWindowProp invDelta)
  (hP17 : p20SamplingProp cset samples)
  (hP18 : p20PolyProp qVals ell totalDegree setSize)
  (hP19 : p20InterpProp xs ys expectedCoeffs evalPoint expectedEval) :
  p20ArithmeticBundle bar m z z1 z2 zDecomp r ρ1 ρ2 b k hVec hScal cset samples invDelta qVals xs ys expectedCoeffs evalPoint expectedEval ell totalDegree setSize := by
  exact p20ArithmeticBundle_of_props_with_thm3CoreAssumption_with_evalHom_assumption
    (hP6 := hP6)
    (hP12Rows := hP12Rows)
    (hThm3 := hThm3)
    (hP14Assm := p14EvalHomAssumption_of_checkAssumption hP14Check)
    (hP14Size := hP14Size)
    (hP14Rows := hP14Rows)
    (hP15Vec := hP15Vec)
    (hP15Scal := hP15Scal)
    (hP16 := hP16)
    (hP16Win := hP16Win)
    (hP17 := hP17)
    (hP18 := hP18)
    (hP19 := hP19)

/--
Variant of the proposition-native P20 constructor where P12 is supplied through
the theorem-native assumption interface (rather than directly as matrix equality).
-/
theorem p20ArithmeticBundle_of_props_with_matrixTransform_assumption
  {bar : Array (Array F)}
  {m : Array (Array F)}
  {z z1 z2 zDecomp r : Array F}
  {ρ1 ρ2 : F}
  {b k : Nat}
  {hVec : VecModuleHom}
  {hScal : ScalarModuleHom}
  {cset samples : Array Coeffs}
  {invDelta : Coeffs}
  {qVals : Array F}
  {xs ys expectedCoeffs : Array F}
  {evalPoint expectedEval : F}
  {ell totalDegree setSize : Nat}
  (hP6 : p20DecompProp zDecomp b k)
  (hP12Rows : MatrixRowsCompatible m z)
  (hP12Assm : p12MatrixTransformAssumption bar m)
  (hP14 : p20EvalHomProp bar m z1 z2 r ρ1 ρ2)
  (hP15Vec : p20VecModuleProp hVec ρ1 z1 z2)
  (hP15Scal : p20ScalarModuleProp hScal ρ1 z1 z2)
  (hP16 : invertibilityPreconditionsProp)
  (hP16Win : p20InvertibilityWindowProp invDelta)
  (hP17 : p20SamplingProp cset samples)
  (hP18 : p20PolyProp qVals ell totalDegree setSize)
  (hP19 : p20InterpProp xs ys expectedCoeffs evalPoint expectedEval) :
  p20ArithmeticBundle bar m z z1 z2 zDecomp r ρ1 ρ2 b k hVec hScal cset samples invDelta qVals xs ys expectedCoeffs evalPoint expectedEval ell totalDegree setSize := by
  exact p20ArithmeticBundle_of_props
    (hP6 := hP6)
    (hP12Rows := hP12Rows)
    (hP12Eq := p20MatrixTransformProp_of_assumption hP12Assm hP12Rows)
    (hP14 := hP14)
    (hP15Vec := hP15Vec)
    (hP15Scal := hP15Scal)
    (hP16 := hP16)
    (hP16Win := hP16Win)
    (hP17 := hP17)
    (hP18 := hP18)
    (hP19 := hP19)

/--
Variant of the proposition-native P20 constructor where P12 is supplied through
the check-assumption interface.
-/
theorem p20ArithmeticBundle_of_props_with_matrixTransform_checkAssumption
  {bar : Array (Array F)}
  {m : Array (Array F)}
  {z z1 z2 zDecomp r : Array F}
  {ρ1 ρ2 : F}
  {b k : Nat}
  {hVec : VecModuleHom}
  {hScal : ScalarModuleHom}
  {cset samples : Array Coeffs}
  {invDelta : Coeffs}
  {qVals : Array F}
  {xs ys expectedCoeffs : Array F}
  {evalPoint expectedEval : F}
  {ell totalDegree setSize : Nat}
  (hP6 : p20DecompProp zDecomp b k)
  (hP12Rows : MatrixRowsCompatible m z)
  (hP12Check : p12MatrixTransformCheckAssumption bar m)
  (hP14 : p20EvalHomProp bar m z1 z2 r ρ1 ρ2)
  (hP15Vec : p20VecModuleProp hVec ρ1 z1 z2)
  (hP15Scal : p20ScalarModuleProp hScal ρ1 z1 z2)
  (hP16 : invertibilityPreconditionsProp)
  (hP16Win : p20InvertibilityWindowProp invDelta)
  (hP17 : p20SamplingProp cset samples)
  (hP18 : p20PolyProp qVals ell totalDegree setSize)
  (hP19 : p20InterpProp xs ys expectedCoeffs evalPoint expectedEval) :
  p20ArithmeticBundle bar m z z1 z2 zDecomp r ρ1 ρ2 b k hVec hScal cset samples invDelta qVals xs ys expectedCoeffs evalPoint expectedEval ell totalDegree setSize := by
  exact p20ArithmeticBundle_of_props_with_matrixTransform_assumption
    (hP6 := hP6)
    (hP12Rows := hP12Rows)
    (hP12Assm := p12MatrixTransformAssumption_of_checkAssumption hP12Check)
    (hP14 := hP14)
    (hP15Vec := hP15Vec)
    (hP15Scal := hP15Scal)
    (hP16 := hP16)
    (hP16Win := hP16Win)
    (hP17 := hP17)
    (hP18 := hP18)
    (hP19 := hP19)

/--
Compatibility wrapper: theorem-native constructor path using both P12 and P14
assumption interfaces.

Prefer `p20ArithmeticBundle_of_assumptions`.
-/
theorem p20ArithmeticBundle_of_props_with_matrixTransform_assumption_with_evalHom_assumption
  {bar : Array (Array F)}
  {m : Array (Array F)}
  {z z1 z2 zDecomp r : Array F}
  {ρ1 ρ2 : F}
  {b k : Nat}
  {hVec : VecModuleHom}
  {hScal : ScalarModuleHom}
  {cset samples : Array Coeffs}
  {invDelta : Coeffs}
  {qVals : Array F}
  {xs ys expectedCoeffs : Array F}
  {evalPoint expectedEval : F}
  {ell totalDegree setSize : Nat}
  (hP6 : p20DecompProp zDecomp b k)
  (hP12Rows : MatrixRowsCompatible m z)
  (hP12Assm : p12MatrixTransformAssumption bar m)
  (hP14Assm : p14EvalHomAssumption bar m r ρ1 ρ2)
  (hP14Size : z1.size = z2.size)
  (hP14Rows : MatrixRowsCompatible m z1)
  (hP15Vec : p20VecModuleProp hVec ρ1 z1 z2)
  (hP15Scal : p20ScalarModuleProp hScal ρ1 z1 z2)
  (hP16 : invertibilityPreconditionsProp)
  (hP16Win : p20InvertibilityWindowProp invDelta)
  (hP17 : p20SamplingProp cset samples)
  (hP18 : p20PolyProp qVals ell totalDegree setSize)
  (hP19 : p20InterpProp xs ys expectedCoeffs evalPoint expectedEval) :
  p20ArithmeticBundle bar m z z1 z2 zDecomp r ρ1 ρ2 b k hVec hScal cset samples invDelta qVals xs ys expectedCoeffs evalPoint expectedEval ell totalDegree setSize := by
  exact p20ArithmeticBundle_of_assumptions
    (hP6 := hP6)
    (hP12Rows := hP12Rows)
    (hP12Assm := hP12Assm)
    (hP14Assm := hP14Assm)
    (hP14Size := hP14Size)
    (hP14Rows := hP14Rows)
    (hP15Vec := hP15Vec)
    (hP15Scal := hP15Scal)
    (hP16 := hP16)
    (hP16Win := hP16Win)
    (hP17 := hP17)
    (hP18 := hP18)
    (hP19 := hP19)

/--
Mixed theorem-native constructor path using P12 assumption interface and P14
check-assumption interface.
-/
theorem p20ArithmeticBundle_of_props_with_matrixTransform_assumption_with_evalHom_checkAssumption
  {bar : Array (Array F)}
  {m : Array (Array F)}
  {z z1 z2 zDecomp r : Array F}
  {ρ1 ρ2 : F}
  {b k : Nat}
  {hVec : VecModuleHom}
  {hScal : ScalarModuleHom}
  {cset samples : Array Coeffs}
  {invDelta : Coeffs}
  {qVals : Array F}
  {xs ys expectedCoeffs : Array F}
  {evalPoint expectedEval : F}
  {ell totalDegree setSize : Nat}
  (hP6 : p20DecompProp zDecomp b k)
  (hP12Rows : MatrixRowsCompatible m z)
  (hP12Assm : p12MatrixTransformAssumption bar m)
  (hP14Check : p14EvalHomCheckAssumption bar m r ρ1 ρ2)
  (hP14Size : z1.size = z2.size)
  (hP14Rows : MatrixRowsCompatible m z1)
  (hP15Vec : p20VecModuleProp hVec ρ1 z1 z2)
  (hP15Scal : p20ScalarModuleProp hScal ρ1 z1 z2)
  (hP16 : invertibilityPreconditionsProp)
  (hP16Win : p20InvertibilityWindowProp invDelta)
  (hP17 : p20SamplingProp cset samples)
  (hP18 : p20PolyProp qVals ell totalDegree setSize)
  (hP19 : p20InterpProp xs ys expectedCoeffs evalPoint expectedEval) :
  p20ArithmeticBundle bar m z z1 z2 zDecomp r ρ1 ρ2 b k hVec hScal cset samples invDelta qVals xs ys expectedCoeffs evalPoint expectedEval ell totalDegree setSize := by
  exact p20ArithmeticBundle_of_assumptions
    (hP6 := hP6)
    (hP12Rows := hP12Rows)
    (hP12Assm := hP12Assm)
    (hP14Assm := p14EvalHomAssumption_of_checkAssumption hP14Check)
    (hP14Size := hP14Size)
    (hP14Rows := hP14Rows)
    (hP15Vec := hP15Vec)
    (hP15Scal := hP15Scal)
    (hP16 := hP16)
    (hP16Win := hP16Win)
    (hP17 := hP17)
    (hP18 := hP18)
    (hP19 := hP19)

/--
Mixed theorem-native constructor path using P12 check-assumption interface and
P14 assumption interface.
-/
theorem p20ArithmeticBundle_of_props_with_matrixTransform_checkAssumption_with_evalHom_assumption
  {bar : Array (Array F)}
  {m : Array (Array F)}
  {z z1 z2 zDecomp r : Array F}
  {ρ1 ρ2 : F}
  {b k : Nat}
  {hVec : VecModuleHom}
  {hScal : ScalarModuleHom}
  {cset samples : Array Coeffs}
  {invDelta : Coeffs}
  {qVals : Array F}
  {xs ys expectedCoeffs : Array F}
  {evalPoint expectedEval : F}
  {ell totalDegree setSize : Nat}
  (hP6 : p20DecompProp zDecomp b k)
  (hP12Rows : MatrixRowsCompatible m z)
  (hP12Check : p12MatrixTransformCheckAssumption bar m)
  (hP14Assm : p14EvalHomAssumption bar m r ρ1 ρ2)
  (hP14Size : z1.size = z2.size)
  (hP14Rows : MatrixRowsCompatible m z1)
  (hP15Vec : p20VecModuleProp hVec ρ1 z1 z2)
  (hP15Scal : p20ScalarModuleProp hScal ρ1 z1 z2)
  (hP16 : invertibilityPreconditionsProp)
  (hP16Win : p20InvertibilityWindowProp invDelta)
  (hP17 : p20SamplingProp cset samples)
  (hP18 : p20PolyProp qVals ell totalDegree setSize)
  (hP19 : p20InterpProp xs ys expectedCoeffs evalPoint expectedEval) :
  p20ArithmeticBundle bar m z z1 z2 zDecomp r ρ1 ρ2 b k hVec hScal cset samples invDelta qVals xs ys expectedCoeffs evalPoint expectedEval ell totalDegree setSize := by
  exact p20ArithmeticBundle_of_assumptions
    (hP6 := hP6)
    (hP12Rows := hP12Rows)
    (hP12Assm := p12MatrixTransformAssumption_of_checkAssumption hP12Check)
    (hP14Assm := hP14Assm)
    (hP14Size := hP14Size)
    (hP14Rows := hP14Rows)
    (hP15Vec := hP15Vec)
    (hP15Scal := hP15Scal)
    (hP16 := hP16)
    (hP16Win := hP16Win)
    (hP17 := hP17)
    (hP18 := hP18)
    (hP19 := hP19)

/--
Theorem-native constructor path using both P12 and P14 check-assumption
interfaces.
-/
theorem p20ArithmeticBundle_of_props_with_matrixTransform_checkAssumption_with_evalHom_checkAssumption
  {bar : Array (Array F)}
  {m : Array (Array F)}
  {z z1 z2 zDecomp r : Array F}
  {ρ1 ρ2 : F}
  {b k : Nat}
  {hVec : VecModuleHom}
  {hScal : ScalarModuleHom}
  {cset samples : Array Coeffs}
  {invDelta : Coeffs}
  {qVals : Array F}
  {xs ys expectedCoeffs : Array F}
  {evalPoint expectedEval : F}
  {ell totalDegree setSize : Nat}
  (hP6 : p20DecompProp zDecomp b k)
  (hP12Rows : MatrixRowsCompatible m z)
  (hP12Check : p12MatrixTransformCheckAssumption bar m)
  (hP14Check : p14EvalHomCheckAssumption bar m r ρ1 ρ2)
  (hP14Size : z1.size = z2.size)
  (hP14Rows : MatrixRowsCompatible m z1)
  (hP15Vec : p20VecModuleProp hVec ρ1 z1 z2)
  (hP15Scal : p20ScalarModuleProp hScal ρ1 z1 z2)
  (hP16 : invertibilityPreconditionsProp)
  (hP16Win : p20InvertibilityWindowProp invDelta)
  (hP17 : p20SamplingProp cset samples)
  (hP18 : p20PolyProp qVals ell totalDegree setSize)
  (hP19 : p20InterpProp xs ys expectedCoeffs evalPoint expectedEval) :
  p20ArithmeticBundle bar m z z1 z2 zDecomp r ρ1 ρ2 b k hVec hScal cset samples invDelta qVals xs ys expectedCoeffs evalPoint expectedEval ell totalDegree setSize := by
  exact p20ArithmeticBundle_of_assumptions
    (hP6 := hP6)
    (hP12Rows := hP12Rows)
    (hP12Assm := p12MatrixTransformAssumption_of_checkAssumption hP12Check)
    (hP14Assm := p14EvalHomAssumption_of_checkAssumption hP14Check)
    (hP14Size := hP14Size)
    (hP14Rows := hP14Rows)
    (hP15Vec := hP15Vec)
    (hP15Scal := hP15Scal)
    (hP16 := hP16)
    (hP16Win := hP16Win)
    (hP17 := hP17)
    (hP18 := hP18)
    (hP19 := hP19)

/--
Variant of the proposition-native P20 constructor where P14 is supplied through
the theorem-native assumption interface (rather than directly as `p20EvalHomProp`).
-/
theorem p20ArithmeticBundle_of_props_with_evalHom_assumption
  {bar : Array (Array F)}
  {m : Array (Array F)}
  {z z1 z2 zDecomp r : Array F}
  {ρ1 ρ2 : F}
  {b k : Nat}
  {hVec : VecModuleHom}
  {hScal : ScalarModuleHom}
  {cset samples : Array Coeffs}
  {invDelta : Coeffs}
  {qVals : Array F}
  {xs ys expectedCoeffs : Array F}
  {evalPoint expectedEval : F}
  {ell totalDegree setSize : Nat}
  (hP6 : p20DecompProp zDecomp b k)
  (hP12Rows : MatrixRowsCompatible m z)
  (hP12Eq : matrixVecDirect m z = matrixVecCtBar bar m z)
  (hP14Assm : p14EvalHomAssumption bar m r ρ1 ρ2)
  (hP14Size : z1.size = z2.size)
  (hP14Rows : MatrixRowsCompatible m z1)
  (hP15Vec : p20VecModuleProp hVec ρ1 z1 z2)
  (hP15Scal : p20ScalarModuleProp hScal ρ1 z1 z2)
  (hP16 : invertibilityPreconditionsProp)
  (hP16Win : p20InvertibilityWindowProp invDelta)
  (hP17 : p20SamplingProp cset samples)
  (hP18 : p20PolyProp qVals ell totalDegree setSize)
  (hP19 : p20InterpProp xs ys expectedCoeffs evalPoint expectedEval) :
  p20ArithmeticBundle bar m z z1 z2 zDecomp r ρ1 ρ2 b k hVec hScal cset samples invDelta qVals xs ys expectedCoeffs evalPoint expectedEval ell totalDegree setSize := by
  exact p20ArithmeticBundle_of_props
    (hP6 := hP6)
    (hP12Rows := hP12Rows)
    (hP12Eq := hP12Eq)
    (hP14 := p20EvalHomProp_of_assumption hP14Assm hP14Size hP14Rows)
    (hP15Vec := hP15Vec)
    (hP15Scal := hP15Scal)
    (hP16 := hP16)
    (hP16Win := hP16Win)
    (hP17 := hP17)
    (hP18 := hP18)
    (hP19 := hP19)

/--
Variant of the proposition-native P20 constructor where P14 is supplied through
the check-assumption interface.
-/
theorem p20ArithmeticBundle_of_props_with_evalHom_checkAssumption
  {bar : Array (Array F)}
  {m : Array (Array F)}
  {z z1 z2 zDecomp r : Array F}
  {ρ1 ρ2 : F}
  {b k : Nat}
  {hVec : VecModuleHom}
  {hScal : ScalarModuleHom}
  {cset samples : Array Coeffs}
  {invDelta : Coeffs}
  {qVals : Array F}
  {xs ys expectedCoeffs : Array F}
  {evalPoint expectedEval : F}
  {ell totalDegree setSize : Nat}
  (hP6 : p20DecompProp zDecomp b k)
  (hP12Rows : MatrixRowsCompatible m z)
  (hP12Eq : matrixVecDirect m z = matrixVecCtBar bar m z)
  (hP14Check : p14EvalHomCheckAssumption bar m r ρ1 ρ2)
  (hP14Size : z1.size = z2.size)
  (hP14Rows : MatrixRowsCompatible m z1)
  (hP15Vec : p20VecModuleProp hVec ρ1 z1 z2)
  (hP15Scal : p20ScalarModuleProp hScal ρ1 z1 z2)
  (hP16 : invertibilityPreconditionsProp)
  (hP16Win : p20InvertibilityWindowProp invDelta)
  (hP17 : p20SamplingProp cset samples)
  (hP18 : p20PolyProp qVals ell totalDegree setSize)
  (hP19 : p20InterpProp xs ys expectedCoeffs evalPoint expectedEval) :
  p20ArithmeticBundle bar m z z1 z2 zDecomp r ρ1 ρ2 b k hVec hScal cset samples invDelta qVals xs ys expectedCoeffs evalPoint expectedEval ell totalDegree setSize := by
  exact p20ArithmeticBundle_of_props
    (hP6 := hP6)
    (hP12Rows := hP12Rows)
    (hP12Eq := hP12Eq)
    (hP14 := p20EvalHomProp_of_checkAssumption hP14Check hP14Size hP14Rows)
    (hP15Vec := hP15Vec)
    (hP15Scal := hP15Scal)
    (hP16 := hP16)
    (hP16Win := hP16Win)
    (hP17 := hP17)
    (hP18 := hP18)
    (hP19 := hP19)

/--
Bridge theorem: executable checks imply the proposition-native P20 bundle.
-/
theorem p20ArithmeticBundle_checks_imply_props
  {bar : Array (Array F)}
  {m : Array (Array F)}
  {z z1 z2 zDecomp r : Array F}
  {ρ1 ρ2 : F}
  {b k : Nat}
  {hVec : VecModuleHom}
  {hScal : ScalarModuleHom}
  {cset samples : Array Coeffs}
  {invDelta : Coeffs}
  {qVals : Array F}
  {xs ys expectedCoeffs : Array F}
  {evalPoint expectedEval : F}
  {ell totalDegree setSize : Nat}
  (hP6 : splitRoundTrip zDecomp b k = true)
  (hP12 : matrixTransformIdentity bar m z = true)
  (hP14 : evalHom2 bar m z1 z2 r ρ1 ρ2 = true)
  (hVecAdd : preservesAddVec hVec z1 z2 = true)
  (hVecScale : preservesScaleVec hVec ρ1 z1 = true)
  (hScalAdd : preservesAddScalar hScal z1 z2 = true)
  (hScalScale : preservesScaleScalar hScal ρ1 z1 = true)
  (hP16Win : p20InvertibilityWindowProp invDelta)
  (hP17 : samplingSetBoundCheck cset samples = true)
  (hP18Eq : eqLiftAllBoolean qVals ell = true)
  (hP18SZ : schwartzZippelBoundLeOne totalDegree setSize = true)
  (hP19 : interpolationCase xs ys expectedCoeffs evalPoint expectedEval = true) :
  p20ArithmeticBundle bar m z z1 z2 zDecomp r ρ1 ρ2 b k hVec hScal cset samples invDelta qVals xs ys expectedCoeffs evalPoint expectedEval ell totalDegree setSize := by
  have hP12Full : MatrixRowsCompatible m z ∧ matrixVecDirect m z = matrixVecCtBar bar m z :=
    matrixTransformIdentity_sound_full hP12
  have hP15VecProp : p20VecModuleProp hVec ρ1 z1 z2 := by
    exact vecModulePropPair_of_checkPair (hCheck := ⟨hVecAdd, hVecScale⟩)
  have hP15ScalProp : p20ScalarModuleProp hScal ρ1 z1 z2 := by
    exact scalarModulePropPair_of_checkPair (hCheck := ⟨hScalAdd, hScalScale⟩)
  exact p20ArithmeticBundle_of_props
    (hP6 := p20DecompProp_of_splitRoundTrip hP6)
    (hP12Rows := hP12Full.1)
    (hP12Eq := hP12Full.2)
    (hP14 := evalHom2_sound_full hP14)
    (hP15Vec := hP15VecProp)
    (hP15Scal := hP15ScalProp)
    (hP16 := invertibilityPreconditions_from_constants)
    (hP16Win := hP16Win)
    (hP17 := samplingSetBoundCheck_sound hP17)
    (hP18 := ⟨hP18Eq, (schwartzZippelBoundLeOne_sound hP18SZ).1, (schwartzZippelBoundLeOne_sound hP18SZ).2⟩)
    (hP19 := interpolationCase_sound hP19)

/--
Subset bridge in the opposite direction: proposition-level P20 assumptions imply
check-level obligations for P6/P17/P18/P19.
-/
theorem p20ArithmeticBundle_props_imply_check_subset
  {bar : Array (Array F)}
  {m : Array (Array F)}
  {z z1 z2 zDecomp r : Array F}
  {ρ1 ρ2 : F}
  {b k : Nat}
  {hVec : VecModuleHom}
  {hScal : ScalarModuleHom}
  {cset samples : Array Coeffs}
  {invDelta : Coeffs}
  {qVals : Array F}
  {xs ys expectedCoeffs : Array F}
  {evalPoint expectedEval : F}
  {ell totalDegree setSize : Nat}
  (hP20 : p20ArithmeticBundle bar m z z1 z2 zDecomp r ρ1 ρ2 b k hVec hScal cset samples invDelta qVals xs ys expectedCoeffs evalPoint expectedEval ell totalDegree setSize) :
  splitRoundTrip zDecomp b k = true ∧
  matrixTransformIdentity bar m z = true ∧
  evalHom2 bar m z1 z2 r ρ1 ρ2 = true ∧
    samplingSetBoundCheck cset samples = true ∧
    eqLiftAllBoolean qVals ell = true ∧
    schwartzZippelBoundLeOne totalDegree setSize = true ∧
  interpolationCase xs ys expectedCoeffs evalPoint expectedEval = true := by
  rcases hP20 with ⟨hP6, hP12Rows, hP12Eq, hP14, _hP15Vec, _hP15Scal, _hP16, _hP16Win, hP17, hP18, hP19⟩
  rcases hP18 with ⟨hP18Eq, hSetNonzero, hDegBound⟩
  refine ⟨splitRoundTrip_of_p20DecompProp hP6, ?_, ?_, ?_, hP18Eq, ?_, ?_⟩
  · exact matrixTransformIdentity_complete_of_rowsCompatible hP12Rows hP12Eq
  · exact evalHom2_complete hP14
  · exact samplingSetBoundCheck_complete hP17
  · exact schwartzZippelBoundLeOne_complete hSetNonzero hDegBound
  · exact interpolationCase_complete hP19

/--
Additional proposition -> check bridge for P15 obligations, requiring the
size guard used by additivity checks.
-/
theorem p20ArithmeticBundle_props_imply_module_checks
  {bar : Array (Array F)}
  {m : Array (Array F)}
  {z z1 z2 zDecomp r : Array F}
  {ρ1 ρ2 : F}
  {b k : Nat}
  {hVec : VecModuleHom}
  {hScal : ScalarModuleHom}
  {cset samples : Array Coeffs}
  {invDelta : Coeffs}
  {qVals : Array F}
  {xs ys expectedCoeffs : Array F}
  {evalPoint expectedEval : F}
  {ell totalDegree setSize : Nat}
  (hP20 : p20ArithmeticBundle bar m z z1 z2 zDecomp r ρ1 ρ2 b k hVec hScal cset samples invDelta qVals xs ys expectedCoeffs evalPoint expectedEval ell totalDegree setSize) :
  preservesAddVec hVec z1 z2 = true ∧
    preservesScaleVec hVec ρ1 z1 = true ∧
    preservesAddScalar hScal z1 z2 = true ∧
    preservesScaleScalar hScal ρ1 z1 = true := by
  rcases hP20 with ⟨_hP6, _hP12Rows, _hP12Eq, hP14, hP15Vec, hP15Scal, _hP16, _hP16Win, _hP17, _hP18, _hP19⟩
  have hSize : z1.size = z2.size := evalHom2Prop_size_eq hP14
  have hVecChecks : vecModuleCheckPair hVec ρ1 z1 z2 := by
    exact vecModuleCheckPair_of_propPair hSize hP15Vec
  have hScalChecks : scalarModuleCheckPair hScal ρ1 z1 z2 := by
    exact scalarModuleCheckPair_of_propPair hSize hP15Scal
  exact ⟨hVecChecks.1, hVecChecks.2, hScalChecks.1, hScalChecks.2⟩

/--
Full proposition -> check bridge for the check-side obligations used by the
backward-compatible P20 constructor.
-/
theorem p20ArithmeticBundle_props_imply_checks
  {bar : Array (Array F)}
  {m : Array (Array F)}
  {z z1 z2 zDecomp r : Array F}
  {ρ1 ρ2 : F}
  {b k : Nat}
  {hVec : VecModuleHom}
  {hScal : ScalarModuleHom}
  {cset samples : Array Coeffs}
  {invDelta : Coeffs}
  {qVals : Array F}
  {xs ys expectedCoeffs : Array F}
  {evalPoint expectedEval : F}
  {ell totalDegree setSize : Nat}
  (hP20 : p20ArithmeticBundle bar m z z1 z2 zDecomp r ρ1 ρ2 b k hVec hScal cset samples invDelta qVals xs ys expectedCoeffs evalPoint expectedEval ell totalDegree setSize) :
  splitRoundTrip zDecomp b k = true ∧
    matrixTransformIdentity bar m z = true ∧
    evalHom2 bar m z1 z2 r ρ1 ρ2 = true ∧
    preservesAddVec hVec z1 z2 = true ∧
    preservesScaleVec hVec ρ1 z1 = true ∧
    preservesAddScalar hScal z1 z2 = true ∧
    preservesScaleScalar hScal ρ1 z1 = true ∧
    samplingSetBoundCheck cset samples = true ∧
    eqLiftAllBoolean qVals ell = true ∧
    schwartzZippelBoundLeOne totalDegree setSize = true ∧
    interpolationCase xs ys expectedCoeffs evalPoint expectedEval = true := by
  have hSubset :
      splitRoundTrip zDecomp b k = true ∧
      matrixTransformIdentity bar m z = true ∧
      evalHom2 bar m z1 z2 r ρ1 ρ2 = true ∧
      samplingSetBoundCheck cset samples = true ∧
      eqLiftAllBoolean qVals ell = true ∧
      schwartzZippelBoundLeOne totalDegree setSize = true ∧
      interpolationCase xs ys expectedCoeffs evalPoint expectedEval = true := by
    exact p20ArithmeticBundle_props_imply_check_subset hP20
  have hModule :
      preservesAddVec hVec z1 z2 = true ∧
      preservesScaleVec hVec ρ1 z1 = true ∧
      preservesAddScalar hScal z1 z2 = true ∧
      preservesScaleScalar hScal ρ1 z1 = true := by
    exact p20ArithmeticBundle_props_imply_module_checks hP20
  rcases hSubset with ⟨hSplit, hMat, hEvalHom, hSamp, hEq, hSZ, hInterp⟩
  rcases hModule with ⟨hVecAdd, hVecScale, hScalAdd, hScalScale⟩
  exact ⟨
    hSplit,
    hMat,
    hEvalHom,
    hVecAdd,
    hVecScale,
    hScalAdd,
    hScalScale,
    hSamp,
    hEq,
    hSZ,
    hInterp
  ⟩

/--
P20 check/prop equivalence for the concrete backward-compatible check surface.
-/
theorem p20ArithmeticBundle_iff_checks
  {bar : Array (Array F)}
  {m : Array (Array F)}
  {z z1 z2 zDecomp r : Array F}
  {ρ1 ρ2 : F}
  {b k : Nat}
  {hVec : VecModuleHom}
  {hScal : ScalarModuleHom}
  {cset samples : Array Coeffs}
  {invDelta : Coeffs}
  {qVals : Array F}
  {xs ys expectedCoeffs : Array F}
  {evalPoint expectedEval : F}
  {ell totalDegree setSize : Nat} :
  p20ArithmeticBundle bar m z z1 z2 zDecomp r ρ1 ρ2 b k hVec hScal cset samples invDelta qVals xs ys expectedCoeffs evalPoint expectedEval ell totalDegree setSize ↔
    splitRoundTrip zDecomp b k = true ∧
    matrixTransformIdentity bar m z = true ∧
    evalHom2 bar m z1 z2 r ρ1 ρ2 = true ∧
    preservesAddVec hVec z1 z2 = true ∧
    preservesScaleVec hVec ρ1 z1 = true ∧
    preservesAddScalar hScal z1 z2 = true ∧
    preservesScaleScalar hScal ρ1 z1 = true ∧
    p20InvertibilityWindowProp invDelta ∧
    samplingSetBoundCheck cset samples = true ∧
    eqLiftAllBoolean qVals ell = true ∧
    schwartzZippelBoundLeOne totalDegree setSize = true ∧
    interpolationCase xs ys expectedCoeffs evalPoint expectedEval = true := by
  constructor
  · intro hP20
    have hChecks :
        splitRoundTrip zDecomp b k = true ∧
        matrixTransformIdentity bar m z = true ∧
        evalHom2 bar m z1 z2 r ρ1 ρ2 = true ∧
        preservesAddVec hVec z1 z2 = true ∧
        preservesScaleVec hVec ρ1 z1 = true ∧
        preservesAddScalar hScal z1 z2 = true ∧
        preservesScaleScalar hScal ρ1 z1 = true ∧
        samplingSetBoundCheck cset samples = true ∧
        eqLiftAllBoolean qVals ell = true ∧
        schwartzZippelBoundLeOne totalDegree setSize = true ∧
        interpolationCase xs ys expectedCoeffs evalPoint expectedEval = true := by
      exact p20ArithmeticBundle_props_imply_checks hP20
    rcases hP20 with ⟨_hP6, _hP12Rows, _hP12Eq, _hP14, _hP15Vec, _hP15Scal, _hP16, hInvWin, _hP17, _hP18, _hP19⟩
    rcases hChecks with
      ⟨hSplit, hMat, hEvalHom, hVecAdd, hVecScale, hScalAdd, hScalScale, hSamp, hEq, hSZ, hInterp⟩
    exact ⟨
      hSplit,
      hMat,
      hEvalHom,
      hVecAdd,
      hVecScale,
      hScalAdd,
      hScalScale,
      hInvWin,
      hSamp,
      hEq,
      hSZ,
      hInterp
    ⟩
  · intro hChecks
    rcases hChecks with
      ⟨hSplit, hMat, hEvalHom, hVecAdd, hVecScale, hScalAdd, hScalScale, hInvWin, hSamp, hEq, hSZ, hInterp⟩
    exact p20ArithmeticBundle_checks_imply_props
      hSplit hMat hEvalHom hVecAdd hVecScale hScalAdd hScalScale hInvWin hSamp hEq hSZ hInterp

/--
Backward-compatible check-driven constructor for P20.
-/
theorem p20ArithmeticBundle_of_checks
  {bar : Array (Array F)}
  {m : Array (Array F)}
  {z z1 z2 zDecomp r : Array F}
  {ρ1 ρ2 : F}
  {b k : Nat}
  {hVec : VecModuleHom}
  {hScal : ScalarModuleHom}
  {cset samples : Array Coeffs}
  {invDelta : Coeffs}
  {qVals : Array F}
  {xs ys expectedCoeffs : Array F}
  {evalPoint expectedEval : F}
  {ell totalDegree setSize : Nat}
  (hP6 : splitRoundTrip zDecomp b k = true)
  (hP12 : matrixTransformIdentity bar m z = true)
  (hP14 : evalHom2 bar m z1 z2 r ρ1 ρ2 = true)
  (hVecAdd : preservesAddVec hVec z1 z2 = true)
  (hVecScale : preservesScaleVec hVec ρ1 z1 = true)
  (hScalAdd : preservesAddScalar hScal z1 z2 = true)
  (hScalScale : preservesScaleScalar hScal ρ1 z1 = true)
  (hP16Win : p20InvertibilityWindowProp invDelta)
  (hP17 : samplingSetBoundCheck cset samples = true)
  (hP18Eq : eqLiftAllBoolean qVals ell = true)
  (hP18SZ : schwartzZippelBoundLeOne totalDegree setSize = true)
  (hP19 : interpolationCase xs ys expectedCoeffs evalPoint expectedEval = true) :
  p20ArithmeticBundle bar m z z1 z2 zDecomp r ρ1 ρ2 b k hVec hScal cset samples invDelta qVals xs ys expectedCoeffs evalPoint expectedEval ell totalDegree setSize := by
  exact p20ArithmeticBundle_checks_imply_props
    hP6 hP12 hP14 hVecAdd hVecScale hScalAdd hScalScale hP16Win hP17 hP18Eq hP18SZ hP19

/--
Assumption-driven invertibility witness extraction for the concrete delta tracked in P20.
-/
theorem p20InvertibilityWitness_of_assumption
  {bar : Array (Array F)}
  {m : Array (Array F)}
  {z z1 z2 zDecomp r : Array F}
  {ρ1 ρ2 : F}
  {b k : Nat}
  {hVec : VecModuleHom}
  {hScal : ScalarModuleHom}
  {cset samples : Array Coeffs}
  {invDelta : Coeffs}
  {qVals : Array F}
  {xs ys expectedCoeffs : Array F}
  {evalPoint expectedEval : F}
  {ell totalDegree setSize : Nat}
  (hInv : LowNormInvertibilityAssumption)
  (hP20 : p20ArithmeticBundle bar m z z1 z2 zDecomp r ρ1 ρ2 b k hVec hScal cset samples invDelta qVals xs ys expectedCoeffs evalPoint expectedEval ell totalDegree setSize) :
  ∃ deltaInv : Coeffs, mulRq invDelta deltaInv = oneRq := by
  exact invertible_of_withinInvertibilityWindow_of_assumption
    hInv (p20ArithmeticBundle_invertibilityWindow (hP20 := hP20))

/--
Helper: derive `0 < normInfCoeffs (mulRq aDelta bDelta)` from the P20 bundle's
invertibility-window obligation for `invDelta = mulRq aDelta bDelta`.
-/
theorem p20_pos_normInfCoeffs_mulRq_of_p20ArithmeticBundle
  {bar : Array (Array F)}
  {m : Array (Array F)}
  {z z1 z2 zDecomp r : Array F}
  {ρ1 ρ2 : F}
  {b k : Nat}
  {hVec : VecModuleHom}
  {hScal : ScalarModuleHom}
  {cset samples : Array Coeffs}
  {invDelta aDelta bDelta : Coeffs}
  {qVals : Array F}
  {xs ys expectedCoeffs : Array F}
  {evalPoint expectedEval : F}
  {ell totalDegree setSize : Nat}
  (hP20 :
    p20ArithmeticBundle bar m z z1 z2 zDecomp r ρ1 ρ2 b k hVec hScal cset samples
      invDelta qVals xs ys expectedCoeffs evalPoint expectedEval ell totalDegree setSize)
  (hDeltaEq : invDelta = mulRq aDelta bDelta) :
  0 < normInfCoeffs (mulRq aDelta bDelta) := by
  exact p20_pos_normInfCoeffs_mulRq_of_invertibilityWindow
    (invDelta := invDelta) (aDelta := aDelta) (bDelta := bDelta)
    (p20ArithmeticBundle_invertibilityWindow (hP20 := hP20)) hDeltaEq

/--
Core wrapper for the product-based invertibility-witness specializations below.

Given `invDelta = mulRq aDelta bDelta`, this:
1. extracts the required positivity side-condition from the P20 bundle, and
2. transports any witness for `mulRq aDelta bDelta` back to the tracked `invDelta`.
-/
theorem p20InvertibilityWitness_mulRq_of_assumption_core
  {bar : Array (Array F)}
  {m : Array (Array F)}
  {z z1 z2 zDecomp r : Array F}
  {ρ1 ρ2 : F}
  {b k : Nat}
  {hVec : VecModuleHom}
  {hScal : ScalarModuleHom}
  {cset samples : Array Coeffs}
  {invDelta aDelta bDelta : Coeffs}
  {qVals : Array F}
  {xs ys expectedCoeffs : Array F}
  {evalPoint expectedEval : F}
  {ell totalDegree setSize : Nat}
  (hP20 :
    p20ArithmeticBundle bar m z z1 z2 zDecomp r ρ1 ρ2 b k hVec hScal cset samples
      invDelta qVals xs ys expectedCoeffs evalPoint expectedEval ell totalDegree setSize)
  (hDeltaEq : invDelta = mulRq aDelta bDelta)
  (hInvMul :
    0 < normInfCoeffs (mulRq aDelta bDelta) →
      ∃ deltaInv : Coeffs, mulRq (mulRq aDelta bDelta) deltaInv = oneRq) :
  ∃ deltaInv : Coeffs, mulRq invDelta deltaInv = oneRq := by
  have hPosMul : 0 < normInfCoeffs (mulRq aDelta bDelta) :=
    p20_pos_normInfCoeffs_mulRq_of_p20ArithmeticBundle (hP20 := hP20) (hDeltaEq := hDeltaEq)
  have hInvMul' : ∃ deltaInv : Coeffs, mulRq (mulRq aDelta bDelta) deltaInv = oneRq :=
    hInvMul hPosMul
  simpa [hDeltaEq] using hInvMul'

/--
Generic operand/raw-coeff-bound witness extraction when the tracked `invDelta`
is identified with a product `mulRq aDelta bDelta`.
-/
theorem p20InvertibilityWitness_mulRq_of_operand_norm_assumptions_rawCoeff_of_assumption
  {bar : Array (Array F)}
  {m : Array (Array F)}
  {z z1 z2 zDecomp r : Array F}
  {ρ1 ρ2 : F}
  {b k : Nat}
  {hVec : VecModuleHom}
  {hScal : ScalarModuleHom}
  {cset samples : Array Coeffs}
  {invDelta aDelta bDelta : Coeffs}
  {qVals : Array F}
  {xs ys expectedCoeffs : Array F}
  {evalPoint expectedEval : F}
  {ell totalDegree setSize : Nat}
  {BA BB BRaw B : Nat}
  (hInv : LowNormInvertibilityAssumption)
  (hP20 : p20ArithmeticBundle bar m z z1 z2 zDecomp r ρ1 ρ2 b k hVec hScal cset samples invDelta qVals xs ys expectedCoeffs evalPoint expectedEval ell totalDegree setSize)
  (hDeltaEq : invDelta = mulRq aDelta bDelta)
  (hA : normInfCoeffs aDelta ≤ BA)
  (hB : normInfCoeffs bDelta ≤ BB)
  (hRawCoeff : mulRqRawCoeffBoundFromOperands BA BB BRaw)
  (hAddSub : rawAddSubCollapseBound BRaw B)
  (hSub : rawSubCollapseBound BRaw B)
  (hBLt : B < bInvApprox) :
  ∃ deltaInv : Coeffs, mulRq invDelta deltaInv = oneRq := by
  have hWin : withinInvertibilityWindow invDelta = true :=
    p20ArithmeticBundle_invertibilityWindow (hP20 := hP20)
  have hPosMul : 0 < normInfCoeffs (mulRq aDelta bDelta) :=
    p20_pos_normInfCoeffs_mulRq_of_invertibilityWindow hWin hDeltaEq
  have hInvMul : ∃ deltaInv : Coeffs, mulRq (mulRq aDelta bDelta) deltaInv = oneRq :=
    invertible_mulRq_of_operand_norm_assumptions_rawCoeff_of_assumption
      hInv hA hB hPosMul hRawCoeff hAddSub hSub hBLt
  simpa [hDeltaEq] using hInvMul

/--
Field-op collapse variant of generic operand/raw-coeff witness extraction when
`invDelta = mulRq aDelta bDelta`.
-/
theorem p20InvertibilityWitness_mulRq_of_operand_norm_assumptions_rawCoeff_fieldOp_of_assumption
  {bar : Array (Array F)}
  {m : Array (Array F)}
  {z z1 z2 zDecomp r : Array F}
  {ρ1 ρ2 : F}
  {b k : Nat}
  {hVec : VecModuleHom}
  {hScal : ScalarModuleHom}
  {cset samples : Array Coeffs}
  {invDelta aDelta bDelta : Coeffs}
  {qVals : Array F}
  {xs ys expectedCoeffs : Array F}
  {evalPoint expectedEval : F}
  {ell totalDegree setSize : Nat}
  {BA BB B : Nat}
  (hInv : LowNormInvertibilityAssumption)
  (hP20 : p20ArithmeticBundle bar m z z1 z2 zDecomp r ρ1 ρ2 b k hVec hScal cset samples invDelta qVals xs ys expectedCoeffs evalPoint expectedEval ell totalDegree setSize)
  (hDeltaEq : invDelta = mulRq aDelta bDelta)
  (hA : normInfCoeffs aDelta ≤ BA)
  (hB : normInfCoeffs bDelta ≤ BB)
  (hRawCoeff : mulRqRawCoeffBoundFromOperands BA BB B)
  (hOps : rawFieldOpCollapseBound B B)
  (hBLt : B < bInvApprox) :
  ∃ deltaInv : Coeffs, mulRq invDelta deltaInv = oneRq := by
  have hWin : withinInvertibilityWindow invDelta = true :=
    p20ArithmeticBundle_invertibilityWindow (hP20 := hP20)
  have hPosMul : 0 < normInfCoeffs (mulRq aDelta bDelta) :=
    p20_pos_normInfCoeffs_mulRq_of_invertibilityWindow hWin hDeltaEq
  have hInvMul : ∃ deltaInv : Coeffs, mulRq (mulRq aDelta bDelta) deltaInv = oneRq :=
    invertible_mulRq_of_operand_norm_assumptions_rawCoeff_fieldOp_of_assumption
      hInv hA hB hPosMul hRawCoeff hOps hBLt
  simpa [hDeltaEq] using hInvMul

/--
Concrete Goldilocks-bound witness extraction when the tracked `invDelta`
is identified with a product `mulRq a b` and operand-bound assumptions are provided.
-/
theorem p20InvertibilityWitness_mulRq_of_goldilocks_operand_assumptions_of_assumption
  {bar : Array (Array F)}
  {m : Array (Array F)}
  {z z1 z2 zDecomp r : Array F}
  {ρ1 ρ2 : F}
  {b k : Nat}
  {hVec : VecModuleHom}
  {hScal : ScalarModuleHom}
  {cset samples : Array Coeffs}
  {invDelta aDelta bDelta : Coeffs}
  {qVals : Array F}
  {xs ys expectedCoeffs : Array F}
  {evalPoint expectedEval : F}
  {ell totalDegree setSize : Nat}
  (hInv : LowNormInvertibilityAssumption)
  (hP20 : p20ArithmeticBundle bar m z z1 z2 zDecomp r ρ1 ρ2 b k hVec hScal cset samples invDelta qVals xs ys expectedCoeffs evalPoint expectedEval ell totalDegree setSize)
  (hDeltaEq : invDelta = mulRq aDelta bDelta)
  (hA : normInfCoeffs aDelta ≤ Parameters.Goldilocks.B)
  (hB : normInfCoeffs bDelta ≤ Parameters.Goldilocks.B)
  (hRaw : GoldilocksRawNormBoundAssumption)
  (hCollapse : GoldilocksRawCollapseAssumption) :
  ∃ deltaInv : Coeffs, mulRq invDelta deltaInv = oneRq := by
  refine p20InvertibilityWitness_mulRq_of_assumption_core
    (hP20 := hP20) (hDeltaEq := hDeltaEq) ?_
  intro hPosMul
  exact invertible_mulRq_of_goldilocks_operand_assumptions_of_assumption
    hInv hA hB hPosMul hRaw hCollapse

/--
In-range raw-coefficient variant of the concrete Goldilocks-bound witness extraction.
-/
theorem p20InvertibilityWitness_mulRq_of_goldilocks_operand_assumptions_inRange_of_assumption
  {bar : Array (Array F)}
  {m : Array (Array F)}
  {z z1 z2 zDecomp r : Array F}
  {ρ1 ρ2 : F}
  {b k : Nat}
  {hVec : VecModuleHom}
  {hScal : ScalarModuleHom}
  {cset samples : Array Coeffs}
  {invDelta aDelta bDelta : Coeffs}
  {qVals : Array F}
  {xs ys expectedCoeffs : Array F}
  {evalPoint expectedEval : F}
  {ell totalDegree setSize : Nat}
  (hInv : LowNormInvertibilityAssumption)
  (hP20 : p20ArithmeticBundle bar m z z1 z2 zDecomp r ρ1 ρ2 b k hVec hScal cset samples invDelta qVals xs ys expectedCoeffs evalPoint expectedEval ell totalDegree setSize)
  (hDeltaEq : invDelta = mulRq aDelta bDelta)
  (hA : normInfCoeffs aDelta ≤ Parameters.Goldilocks.B)
  (hB : normInfCoeffs bDelta ≤ Parameters.Goldilocks.B)
  (hRawInRange : GoldilocksRawInRangeBoundAssumption)
  (hCollapse : GoldilocksRawCollapseAssumption) :
  ∃ deltaInv : Coeffs, mulRq invDelta deltaInv = oneRq := by
  refine p20InvertibilityWitness_mulRq_of_assumption_core
    (hP20 := hP20) (hDeltaEq := hDeltaEq) ?_
  intro hPosMul
  exact invertible_mulRq_of_goldilocks_operand_assumptions_inRange_of_assumption
    hInv hA hB hPosMul hRawInRange hCollapse

/--
Field-op-collapse variant of concrete Goldilocks-bound witness extraction when
`invDelta = mulRq aDelta bDelta`.
-/
theorem p20InvertibilityWitness_mulRq_of_goldilocks_operand_fieldOp_assumptions_of_assumption
  {bar : Array (Array F)}
  {m : Array (Array F)}
  {z z1 z2 zDecomp r : Array F}
  {ρ1 ρ2 : F}
  {b k : Nat}
  {hVec : VecModuleHom}
  {hScal : ScalarModuleHom}
  {cset samples : Array Coeffs}
  {invDelta aDelta bDelta : Coeffs}
  {qVals : Array F}
  {xs ys expectedCoeffs : Array F}
  {evalPoint expectedEval : F}
  {ell totalDegree setSize : Nat}
  (hInv : LowNormInvertibilityAssumption)
  (hP20 : p20ArithmeticBundle bar m z z1 z2 zDecomp r ρ1 ρ2 b k hVec hScal cset samples invDelta qVals xs ys expectedCoeffs evalPoint expectedEval ell totalDegree setSize)
  (hDeltaEq : invDelta = mulRq aDelta bDelta)
  (hA : normInfCoeffs aDelta ≤ Parameters.Goldilocks.B)
  (hB : normInfCoeffs bDelta ≤ Parameters.Goldilocks.B)
  (hRaw : GoldilocksRawNormBoundAssumption)
  (hFieldOps : GoldilocksFieldOpCollapseAssumption) :
  ∃ deltaInv : Coeffs, mulRq invDelta deltaInv = oneRq := by
  refine p20InvertibilityWitness_mulRq_of_assumption_core
    (hP20 := hP20) (hDeltaEq := hDeltaEq) ?_
  intro hPosMul
  exact invertible_mulRq_of_goldilocks_operand_fieldOp_assumptions_of_assumption
    hInv hA hB hPosMul hRaw hFieldOps

/--
In-range raw-coefficient + field-op-collapse variant of concrete Goldilocks-bound
witness extraction.
-/
theorem p20InvertibilityWitness_mulRq_of_goldilocks_operand_fieldOp_assumptions_inRange_of_assumption
  {bar : Array (Array F)}
  {m : Array (Array F)}
  {z z1 z2 zDecomp r : Array F}
  {ρ1 ρ2 : F}
  {b k : Nat}
  {hVec : VecModuleHom}
  {hScal : ScalarModuleHom}
  {cset samples : Array Coeffs}
  {invDelta aDelta bDelta : Coeffs}
  {qVals : Array F}
  {xs ys expectedCoeffs : Array F}
  {evalPoint expectedEval : F}
  {ell totalDegree setSize : Nat}
  (hInv : LowNormInvertibilityAssumption)
  (hP20 : p20ArithmeticBundle bar m z z1 z2 zDecomp r ρ1 ρ2 b k hVec hScal cset samples invDelta qVals xs ys expectedCoeffs evalPoint expectedEval ell totalDegree setSize)
  (hDeltaEq : invDelta = mulRq aDelta bDelta)
  (hA : normInfCoeffs aDelta ≤ Parameters.Goldilocks.B)
  (hB : normInfCoeffs bDelta ≤ Parameters.Goldilocks.B)
  (hRawInRange : GoldilocksRawInRangeBoundAssumption)
  (hFieldOps : GoldilocksFieldOpCollapseAssumption) :
  ∃ deltaInv : Coeffs, mulRq invDelta deltaInv = oneRq := by
  refine p20InvertibilityWitness_mulRq_of_assumption_core
    (hP20 := hP20) (hDeltaEq := hDeltaEq) ?_
  intro hPosMul
  exact invertible_mulRq_of_goldilocks_operand_fieldOp_assumptions_inRange_of_assumption
    hInv hA hB hPosMul hRawInRange hFieldOps

theorem p20InvertibilityWitness_mulRq_of_goldilocks_operand_rawCoeff_assumptions_of_assumption
  {bar : Array (Array F)}
  {m : Array (Array F)}
  {z z1 z2 zDecomp r : Array F}
  {ρ1 ρ2 : F}
  {b k : Nat}
  {hVec : VecModuleHom}
  {hScal : ScalarModuleHom}
  {cset samples : Array Coeffs}
  {invDelta aDelta bDelta : Coeffs}
  {qVals : Array F}
  {xs ys expectedCoeffs : Array F}
  {evalPoint expectedEval : F}
  {ell totalDegree setSize : Nat}
  (hInv : LowNormInvertibilityAssumption)
  (hP20 : p20ArithmeticBundle bar m z z1 z2 zDecomp r ρ1 ρ2 b k hVec hScal cset samples invDelta qVals xs ys expectedCoeffs evalPoint expectedEval ell totalDegree setSize)
  (hDeltaEq : invDelta = mulRq aDelta bDelta)
  (hA : normInfCoeffs aDelta ≤ Parameters.Goldilocks.B)
  (hB : normInfCoeffs bDelta ≤ Parameters.Goldilocks.B)
  (hRawCoeff : GoldilocksRawCoeffBoundAssumption)
  (hCollapse : GoldilocksRawCollapseAssumption) :
  ∃ deltaInv : Coeffs, mulRq invDelta deltaInv = oneRq := by
  have hWin : withinInvertibilityWindow invDelta = true :=
    p20ArithmeticBundle_invertibilityWindow (hP20 := hP20)
  have hPosMul : 0 < normInfCoeffs (mulRq aDelta bDelta) :=
    p20_pos_normInfCoeffs_mulRq_of_invertibilityWindow hWin hDeltaEq
  have hInvMul : ∃ deltaInv : Coeffs, mulRq (mulRq aDelta bDelta) deltaInv = oneRq :=
    invertible_mulRq_of_goldilocks_operand_rawCoeff_assumptions_of_assumption
      hInv hA hB hPosMul hRawCoeff hCollapse
  simpa [hDeltaEq] using hInvMul

theorem p20InvertibilityWitness_mulRq_of_goldilocks_operand_rawCoeff_fieldOp_assumptions_of_assumption
  {bar : Array (Array F)}
  {m : Array (Array F)}
  {z z1 z2 zDecomp r : Array F}
  {ρ1 ρ2 : F}
  {b k : Nat}
  {hVec : VecModuleHom}
  {hScal : ScalarModuleHom}
  {cset samples : Array Coeffs}
  {invDelta aDelta bDelta : Coeffs}
  {qVals : Array F}
  {xs ys expectedCoeffs : Array F}
  {evalPoint expectedEval : F}
  {ell totalDegree setSize : Nat}
  (hInv : LowNormInvertibilityAssumption)
  (hP20 : p20ArithmeticBundle bar m z z1 z2 zDecomp r ρ1 ρ2 b k hVec hScal cset samples invDelta qVals xs ys expectedCoeffs evalPoint expectedEval ell totalDegree setSize)
  (hDeltaEq : invDelta = mulRq aDelta bDelta)
  (hA : normInfCoeffs aDelta ≤ Parameters.Goldilocks.B)
  (hB : normInfCoeffs bDelta ≤ Parameters.Goldilocks.B)
  (hRawCoeff : GoldilocksRawCoeffBoundAssumption)
  (hFieldOps : GoldilocksFieldOpCollapseAssumption) :
  ∃ deltaInv : Coeffs, mulRq invDelta deltaInv = oneRq := by
  have hWin : withinInvertibilityWindow invDelta = true :=
    p20ArithmeticBundle_invertibilityWindow (hP20 := hP20)
  have hPosMul : 0 < normInfCoeffs (mulRq aDelta bDelta) :=
    p20_pos_normInfCoeffs_mulRq_of_invertibilityWindow hWin hDeltaEq
  have hInvMul : ∃ deltaInv : Coeffs, mulRq (mulRq aDelta bDelta) deltaInv = oneRq :=
    invertible_mulRq_of_goldilocks_operand_rawCoeff_fieldOp_assumptions_of_assumption
      hInv hA hB hPosMul hRawCoeff hFieldOps
  simpa [hDeltaEq] using hInvMul

end SuperNeo
