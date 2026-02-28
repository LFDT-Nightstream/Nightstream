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

theorem p20EvalHomProp_of_p15EvalBarMzAtAssumption
  {bar : Array (Array F)}
  {m : Array (Array F)}
  {z1 z2 r : Array F}
  {ρ1 ρ2 : F}
  (hLin : p15EvalBarMzAtAssumption bar m r)
  (hSize : z1.size = z2.size)
  (hRows : MatrixRowsCompatible m z1) :
  p20EvalHomProp bar m z1 z2 r ρ1 ρ2 := by
  exact p20EvalHomProp_of_assumption
    (p14EvalHomAssumption_of_p15EvalBarMzAtAssumption hLin)
    hSize hRows

theorem p20EvalHomProp_of_p15EvalBarMzAtCheckAssumption
  {bar : Array (Array F)}
  {m : Array (Array F)}
  {z1 z2 r : Array F}
  {ρ1 ρ2 : F}
  (hCheck : p15EvalBarMzAtCheckAssumption bar m r)
  (hSize : z1.size = z2.size)
  (hRows : MatrixRowsCompatible m z1) :
  p20EvalHomProp bar m z1 z2 r ρ1 ρ2 := by
  exact p20EvalHomProp_of_assumption
    (p14EvalHomAssumption_of_p15EvalBarMzAtCheckAssumption hCheck)
    hSize hRows

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

theorem p20SamplingProp_of_operand_norm_assumptions_via_schoolbook
  {cset samples : Array Coeffs}
  {BA BB BTerm BRaw : Nat}
  (hCset : ∀ i : Fin cset.size, normInfCoeffs cset[i] ≤ BA)
  (hSamples : ∀ j : Fin samples.size, normInfCoeffs samples[j] ≤ BB)
  (hMul : ∀ x y : F, normInfF x ≤ BA → normInfF y ≤ BB → normInfF (x * y) ≤ BTerm)
  (hAdd : ∀ x y : F, normInfF x ≤ BRaw → normInfF y ≤ BTerm → normInfF (x + y) ≤ BRaw)
  (hZero : normInfF (0 : F) ≤ BRaw)
  (hAddSub : rawAddSubCollapseBound BRaw (theorem9UpperBound (maxRhoNorm cset)))
  (hSub : rawSubCollapseBound BRaw (theorem9UpperBound (maxRhoNorm cset))) :
  p20SamplingProp cset samples := by
  have hMax : maxRhoNorm cset ≤ BA := maxRhoNorm_le_of_forall_norm_le hCset
  have hMulAtMax :
      ∀ x y : F,
        normInfF x ≤ maxRhoNorm cset →
        normInfF y ≤ BB →
        normInfF (x * y) ≤ BTerm := by
    intro x y hx hy
    exact hMul x y (Nat.le_trans hx hMax) hy
  exact p20SamplingProp_of_empirical_bound
    (empiricalExpansionFactor_le_theorem9UpperBound_of_operand_norm_assumptions_via_schoolbook
      (cset := cset) (samples := samples) (BB := BB) (BTerm := BTerm) (BRaw := BRaw)
      hSamples hMulAtMax hAdd hZero hAddSub hSub)

theorem p20SamplingProp_of_operand_norm_assumptions_via_schoolbook_sum
  {cset samples : Array Coeffs}
  {BA BB BTerm : Nat}
  (hCset : ∀ i : Fin cset.size, normInfCoeffs cset[i] ≤ BA)
  (hSamples : ∀ j : Fin samples.size, normInfCoeffs samples[j] ≤ BB)
  (hMul : ∀ x y : F, normInfF x ≤ BA → normInfF y ≤ BB → normInfF (x * y) ≤ BTerm)
  (hAddTri : ∀ x y : F, normInfF (x + y) ≤ normInfF x + normInfF y)
  (hAddSub :
    rawAddSubCollapseBound
      ((D * D) * BTerm)
      (theorem9UpperBound (maxRhoNorm cset)))
  (hSub :
    rawSubCollapseBound
      ((D * D) * BTerm)
      (theorem9UpperBound (maxRhoNorm cset))) :
  p20SamplingProp cset samples := by
  have hMax : maxRhoNorm cset ≤ BA := maxRhoNorm_le_of_forall_norm_le hCset
  have hMulAtMax :
      ∀ x y : F,
        normInfF x ≤ maxRhoNorm cset →
        normInfF y ≤ BB →
        normInfF (x * y) ≤ BTerm := by
    intro x y hx hy
    exact hMul x y (Nat.le_trans hx hMax) hy
  exact p20SamplingProp_of_empirical_bound
    (empiricalExpansionFactor_le_theorem9UpperBound_of_operand_norm_assumptions_via_schoolbook_sum
      (cset := cset) (samples := samples) (BB := BB) (BTerm := BTerm)
      hSamples hMulAtMax hAddTri hAddSub hSub)

theorem p20SamplingProp_of_operand_norm_assumptions_via_schoolbook_sum_fieldOp
  {cset samples : Array Coeffs}
  {BA BB BTerm : Nat}
  (hCset : ∀ i : Fin cset.size, normInfCoeffs cset[i] ≤ BA)
  (hSamples : ∀ j : Fin samples.size, normInfCoeffs samples[j] ≤ BB)
  (hMul : ∀ x y : F, normInfF x ≤ BA → normInfF y ≤ BB → normInfF (x * y) ≤ BTerm)
  (hAddTri : ∀ x y : F, normInfF (x + y) ≤ normInfF x + normInfF y)
  (hOps : rawFieldOpCollapseBound ((D * D) * BTerm) ((D * D) * BTerm))
  (hRawLe : ((D * D) * BTerm) ≤ theorem9UpperBound (maxRhoNorm cset)) :
  p20SamplingProp cset samples := by
  have hAddSubRaw : rawAddSubCollapseBound ((D * D) * BTerm) ((D * D) * BTerm) :=
    rawAddSubCollapseBound_of_add_and_sub_same hOps.1 hOps.2
  have hAddSub :
      rawAddSubCollapseBound ((D * D) * BTerm) (theorem9UpperBound (maxRhoNorm cset)) :=
    rawAddSubCollapseBound_mono hAddSubRaw hRawLe
  have hSub :
      rawSubCollapseBound ((D * D) * BTerm) (theorem9UpperBound (maxRhoNorm cset)) :=
    rawSubCollapseBound_mono hOps.2 hRawLe
  exact p20SamplingProp_of_operand_norm_assumptions_via_schoolbook_sum
    (cset := cset) (samples := samples)
    (BA := BA) (BB := BB) (BTerm := BTerm)
    hCset hSamples hMul hAddTri hAddSub hSub

theorem p20SamplingProp_of_operand_norm_assumptions_via_schoolbook_fieldOp
  {cset samples : Array Coeffs}
  {BA BB BTerm : Nat}
  (hCset : ∀ i : Fin cset.size, normInfCoeffs cset[i] ≤ BA)
  (hSamples : ∀ j : Fin samples.size, normInfCoeffs samples[j] ≤ BB)
  (hMul : ∀ x y : F, normInfF x ≤ BA → normInfF y ≤ BB → normInfF (x * y) ≤ BTerm)
  (hAdd :
    ∀ x y : F,
      normInfF x ≤ theorem9UpperBound (maxRhoNorm cset) →
      normInfF y ≤ BTerm →
      normInfF (x + y) ≤ theorem9UpperBound (maxRhoNorm cset))
  (hZero : normInfF (0 : F) ≤ theorem9UpperBound (maxRhoNorm cset))
  (hOps :
    rawFieldOpCollapseBound
      (theorem9UpperBound (maxRhoNorm cset))
      (theorem9UpperBound (maxRhoNorm cset))) :
  p20SamplingProp cset samples := by
  rcases hOps with ⟨hAddCollapse, hSub⟩
  exact p20SamplingProp_of_operand_norm_assumptions_via_schoolbook
    (cset := cset) (samples := samples)
    (BA := BA) (BB := BB) (BTerm := BTerm)
    (BRaw := theorem9UpperBound (maxRhoNorm cset))
    hCset hSamples hMul hAdd hZero
    (rawAddSubCollapseBound_of_add_and_sub_same hAddCollapse hSub)
    hSub

theorem p20SamplingProp_of_operand_norm_assumptions_via_schoolbook_of_term_le
  {cset samples : Array Coeffs}
  {BA BB BTerm BRaw : Nat}
  (hCset : ∀ i : Fin cset.size, normInfCoeffs cset[i] ≤ BA)
  (hSamples : ∀ j : Fin samples.size, normInfCoeffs samples[j] ≤ BB)
  (hMul : ∀ x y : F, normInfF x ≤ BA → normInfF y ≤ BB → normInfF (x * y) ≤ BTerm)
  (hTermLe : BTerm ≤ BRaw)
  (hAddCollapse : rawAddCollapseBound BRaw BRaw)
  (hAddSub : rawAddSubCollapseBound BRaw (theorem9UpperBound (maxRhoNorm cset)))
  (hSub : rawSubCollapseBound BRaw (theorem9UpperBound (maxRhoNorm cset))) :
  p20SamplingProp cset samples := by
  have hMax : maxRhoNorm cset ≤ BA := maxRhoNorm_le_of_forall_norm_le hCset
  have hMulAtMax :
      ∀ x y : F,
        normInfF x ≤ maxRhoNorm cset →
        normInfF y ≤ BB →
        normInfF (x * y) ≤ BTerm := by
    intro x y hx hy
    exact hMul x y (Nat.le_trans hx hMax) hy
  exact p20SamplingProp_of_empirical_bound
    (empiricalExpansionFactor_le_theorem9UpperBound_of_operand_norm_assumptions_via_schoolbook_of_term_le
      (cset := cset) (samples := samples) (BB := BB) (BTerm := BTerm) (BRaw := BRaw)
      hSamples hMulAtMax hTermLe hAddCollapse hAddSub hSub)

theorem p20SamplingProp_of_operand_norm_assumptions_via_schoolbook_sameBound
  {cset samples : Array Coeffs}
  {BA BB BRaw : Nat}
  (hCset : ∀ i : Fin cset.size, normInfCoeffs cset[i] ≤ BA)
  (hSamples : ∀ j : Fin samples.size, normInfCoeffs samples[j] ≤ BB)
  (hMul : ∀ x y : F, normInfF x ≤ BA → normInfF y ≤ BB → normInfF (x * y) ≤ BRaw)
  (hAddCollapse : rawAddCollapseBound BRaw BRaw)
  (hAddSub : rawAddSubCollapseBound BRaw (theorem9UpperBound (maxRhoNorm cset)))
  (hSub : rawSubCollapseBound BRaw (theorem9UpperBound (maxRhoNorm cset))) :
  p20SamplingProp cset samples := by
  exact p20SamplingProp_of_operand_norm_assumptions_via_schoolbook_of_term_le
    (cset := cset) (samples := samples)
    (BA := BA) (BB := BB) (BTerm := BRaw) (BRaw := BRaw)
    hCset hSamples hMul (Nat.le_refl BRaw) hAddCollapse hAddSub hSub

theorem p20SamplingProp_of_operand_norm_assumptions_via_schoolbook_sameBound_fieldOp
  {cset samples : Array Coeffs}
  {BA BB : Nat}
  (hCset : ∀ i : Fin cset.size, normInfCoeffs cset[i] ≤ BA)
  (hSamples : ∀ j : Fin samples.size, normInfCoeffs samples[j] ≤ BB)
  (hMul :
    ∀ x y : F,
      normInfF x ≤ BA →
      normInfF y ≤ BB →
      normInfF (x * y) ≤ theorem9UpperBound (maxRhoNorm cset))
  (hOps :
    rawFieldOpCollapseBound
      (theorem9UpperBound (maxRhoNorm cset))
      (theorem9UpperBound (maxRhoNorm cset))) :
  p20SamplingProp cset samples := by
  rcases hOps with ⟨hAddCollapse, hSub⟩
  exact p20SamplingProp_of_operand_norm_assumptions_via_schoolbook_sameBound
    (cset := cset) (samples := samples)
    (BA := BA) (BB := BB)
    (BRaw := theorem9UpperBound (maxRhoNorm cset))
    hCset hSamples hMul hAddCollapse
    (rawAddSubCollapseBound_of_add_and_sub_same hAddCollapse hSub)
    hSub

theorem p20SamplingProp_of_operand_norm_assumptions_via_universal_blockers
  {cset samples : Array Coeffs}
  {BA BB : Nat}
  (hCset : ∀ i : Fin cset.size, normInfCoeffs cset[i] ≤ BA)
  (hSamples : ∀ j : Fin samples.size, normInfCoeffs samples[j] ≤ BB)
  (hMulUniv : schoolbookMulUniversalBound)
  (hAddTri : schoolbookAddTriangleBound)
  (hSubTri : schoolbookSubTriangleBound)
  (hRawLe :
    ((D * D) * (BA * BB)) + ((D * D) * (BA * BB)) + ((D * D) * (BA * BB))
      ≤ theorem9UpperBound (maxRhoNorm cset)) :
  p20SamplingProp cset samples := by
  exact p20SamplingProp_of_empirical_bound
    (empiricalExpansionFactor_le_theorem9UpperBound_of_operand_norm_assumptions_via_universal_blockers
      (cset := cset) (samples := samples) (BA := BA) (BB := BB)
      hCset hSamples hMulUniv hAddTri hSubTri hRawLe)

theorem p20SamplingProp_of_operand_norm_assumptions_via_universal_blockers_tight
  {cset samples : Array Coeffs}
  {BA BB : Nat}
  (hCset : ∀ i : Fin cset.size, normInfCoeffs cset[i] ≤ BA)
  (hSamples : ∀ j : Fin samples.size, normInfCoeffs samples[j] ≤ BB)
  (hMulUniv : schoolbookMulUniversalBound)
  (hAddTri : schoolbookAddTriangleBound)
  (hSubTri : schoolbookSubTriangleBound)
  (hRawLe :
    (D * (BA * BB)) + (D * (BA * BB)) + (D * (BA * BB))
      ≤ theorem9UpperBound (maxRhoNorm cset)) :
  p20SamplingProp cset samples := by
  exact p20SamplingProp_of_empirical_bound
    (empiricalExpansionFactor_le_theorem9UpperBound_of_operand_norm_assumptions_via_universal_blockers_tight
      (cset := cset) (samples := samples) (BA := BA) (BB := BB)
      hCset hSamples hMulUniv hAddTri hSubTri hRawLe)

/--
Triangle-bundle variant of
`p20SamplingProp_of_operand_norm_assumptions_via_universal_blockers_tight`.
-/
theorem p20SamplingProp_of_operand_norm_assumptions_via_universal_mul_and_triangles_tight
  {cset samples : Array Coeffs}
  {BA BB : Nat}
  (hCset : ∀ i : Fin cset.size, normInfCoeffs cset[i] ≤ BA)
  (hSamples : ∀ j : Fin samples.size, normInfCoeffs samples[j] ≤ BB)
  (hMulUniv : schoolbookMulUniversalBound)
  (hTri : schoolbookTriangleBounds)
  (hRawLe :
    (D * (BA * BB)) + (D * (BA * BB)) + (D * (BA * BB))
      ≤ theorem9UpperBound (maxRhoNorm cset)) :
  p20SamplingProp cset samples := by
  exact p20SamplingProp_of_empirical_bound
    (empiricalExpansionFactor_le_theorem9UpperBound_of_operand_norm_assumptions_via_universal_mul_and_triangles_tight
      (cset := cset) (samples := samples) (BA := BA) (BB := BB)
      hCset hSamples hMulUniv hTri hRawLe)

theorem p20SamplingProp_of_operand_norm_assumptions_via_universal_mul_and_add_tight
  {cset samples : Array Coeffs}
  {BA BB : Nat}
  (hCset : ∀ i : Fin cset.size, normInfCoeffs cset[i] ≤ BA)
  (hSamples : ∀ j : Fin samples.size, normInfCoeffs samples[j] ≤ BB)
  (hMulUniv : schoolbookMulUniversalBound)
  (hAddTri : schoolbookAddTriangleBound)
  (hRawLe :
    (D * (BA * BB)) + (D * (BA * BB)) + (D * (BA * BB))
      ≤ theorem9UpperBound (maxRhoNorm cset)) :
  p20SamplingProp cset samples := by
  exact p20SamplingProp_of_operand_norm_assumptions_via_universal_mul_and_triangles_tight
    (cset := cset) (samples := samples) (BA := BA) (BB := BB)
    hCset hSamples hMulUniv (schoolbookTriangleBounds_of_add hAddTri) hRawLe

theorem p20SamplingProp_of_operand_norm_assumptions_via_centeredRep_mul_and_add_tight
  {cset samples : Array Coeffs}
  {BA BB : Nat}
  (hCset : ∀ i : Fin cset.size, normInfCoeffs cset[i] ≤ BA)
  (hSamples : ∀ j : Fin samples.size, normInfCoeffs samples[j] ≤ BB)
  (hMulRep :
    ∀ x y : F, Int.natAbs (centeredRep (x * y))
      ≤ Int.natAbs (centeredRep x) * Int.natAbs (centeredRep y))
  (hAddRep :
    ∀ x y : F, Int.natAbs (centeredRep (x + y))
      ≤ Int.natAbs (centeredRep x) + Int.natAbs (centeredRep y))
  (hRawLe :
    (D * (BA * BB)) + (D * (BA * BB)) + (D * (BA * BB))
      ≤ theorem9UpperBound (maxRhoNorm cset)) :
  p20SamplingProp cset samples := by
  exact p20SamplingProp_of_operand_norm_assumptions_via_universal_mul_and_add_tight
    (cset := cset) (samples := samples) (BA := BA) (BB := BB)
    hCset hSamples
    (schoolbookMulUniversalBound_of_centeredRep hMulRep)
    (schoolbookAddTriangleBound_of_centeredRep hAddRep)
    hRawLe

theorem p20SamplingProp_of_operand_norm_assumptions_via_centeredRep_mul_tight
  {cset samples : Array Coeffs}
  {BA BB : Nat}
  (hCset : ∀ i : Fin cset.size, normInfCoeffs cset[i] ≤ BA)
  (hSamples : ∀ j : Fin samples.size, normInfCoeffs samples[j] ≤ BB)
  (hMulRep :
    ∀ x y : F, Int.natAbs (centeredRep (x * y))
      ≤ Int.natAbs (centeredRep x) * Int.natAbs (centeredRep y))
  (hRawLe :
    (D * (BA * BB)) + (D * (BA * BB)) + (D * (BA * BB))
      ≤ theorem9UpperBound (maxRhoNorm cset)) :
  p20SamplingProp cset samples := by
  exact p20SamplingProp_of_operand_norm_assumptions_via_centeredRep_mul_and_add_tight
    (cset := cset) (samples := samples) (BA := BA) (BB := BB)
    hCset hSamples hMulRep centeredRepAddTriangleBound_theorem hRawLe

theorem p20SamplingProp_of_operand_norm_assumptions_via_centeredRepMulAddBounds_tight
  {cset samples : Array Coeffs}
  {BA BB : Nat}
  (hCset : ∀ i : Fin cset.size, normInfCoeffs cset[i] ≤ BA)
  (hSamples : ∀ j : Fin samples.size, normInfCoeffs samples[j] ≤ BB)
  (hRep : centeredRepMulAddBounds)
  (hRawLe :
    (D * (BA * BB)) + (D * (BA * BB)) + (D * (BA * BB))
      ≤ theorem9UpperBound (maxRhoNorm cset)) :
  p20SamplingProp cset samples := by
  exact p20SamplingProp_of_operand_norm_assumptions_via_centeredRep_mul_and_add_tight
    (cset := cset) (samples := samples) (BA := BA) (BB := BB)
    hCset hSamples
    (centeredRepMulAddBounds_mul hRep)
    (centeredRepMulAddBounds_add hRep)
    hRawLe

/-- Assumption-free native P20 sampling constructor (`D^2` schoolbook path). -/
theorem p20SamplingProp_of_operand_norm_assumptions_native
  {cset samples : Array Coeffs}
  {BA BB : Nat}
  (hCset : ∀ i : Fin cset.size, normInfCoeffs cset[i] ≤ BA)
  (hSamples : ∀ j : Fin samples.size, normInfCoeffs samples[j] ≤ BB)
  (hRawLe :
    ((D * D) * (BA * BB)) + ((D * D) * (BA * BB)) + ((D * D) * (BA * BB))
      ≤ theorem9UpperBound (maxRhoNorm cset)) :
  p20SamplingProp cset samples := by
  exact p20SamplingProp_of_operand_norm_assumptions_via_universal_blockers
    (cset := cset) (samples := samples) (BA := BA) (BB := BB)
    hCset hSamples
    schoolbookMulUniversalBound_theorem
    (schoolbookTriangleBounds_add schoolbookTriangleBounds_theorem)
    (schoolbookTriangleBounds_sub schoolbookTriangleBounds_theorem)
    hRawLe

/-- Assumption-free native-tight P20 sampling constructor (`3 * D * BA * BB` path). -/
theorem p20SamplingProp_of_operand_norm_assumptions_native_tight
  {cset samples : Array Coeffs}
  {BA BB : Nat}
  (hCset : ∀ i : Fin cset.size, normInfCoeffs cset[i] ≤ BA)
  (hSamples : ∀ j : Fin samples.size, normInfCoeffs samples[j] ≤ BB)
  (hRawLe :
    (D * (BA * BB)) + (D * (BA * BB)) + (D * (BA * BB))
      ≤ theorem9UpperBound (maxRhoNorm cset)) :
  p20SamplingProp cset samples := by
  exact p20SamplingProp_of_operand_norm_assumptions_via_universal_mul_and_triangles_tight
    (cset := cset) (samples := samples) (BA := BA) (BB := BB)
    hCset hSamples
    schoolbookMulUniversalBound_theorem
    schoolbookTriangleBounds_theorem
    hRawLe

theorem p20SamplingProp_of_operand_norm_assumptions_via_universal_blockers_and_raw
  {cset samples : Array Coeffs}
  {BA BB BRaw : Nat}
  (hCset : ∀ i : Fin cset.size, normInfCoeffs cset[i] ≤ BA)
  (hSamples : ∀ j : Fin samples.size, normInfCoeffs samples[j] ≤ BB)
  (hRaw : mulRqRawNormBoundFromOperands BA BB BRaw)
  (hAddTri : schoolbookAddTriangleBound)
  (hSubTri : schoolbookSubTriangleBound)
  (hRawLe : BRaw + BRaw + BRaw ≤ theorem9UpperBound (maxRhoNorm cset)) :
  p20SamplingProp cset samples := by
  exact p20SamplingProp_of_empirical_bound
    (empiricalExpansionFactor_le_theorem9UpperBound_of_operand_norm_assumptions_via_universal_blockers_and_raw
      (cset := cset) (samples := samples) (BA := BA) (BB := BB) (BRaw := BRaw)
      hCset hSamples hRaw hAddTri hSubTri hRawLe)

theorem p20SamplingProp_of_operand_norm_assumptions_via_universal_blockers_and_rawCoeff
  {cset samples : Array Coeffs}
  {BA BB BRaw : Nat}
  (hCset : ∀ i : Fin cset.size, normInfCoeffs cset[i] ≤ BA)
  (hSamples : ∀ j : Fin samples.size, normInfCoeffs samples[j] ≤ BB)
  (hRawCoeff : mulRqRawCoeffBoundFromOperands BA BB BRaw)
  (hAddTri : schoolbookAddTriangleBound)
  (hSubTri : schoolbookSubTriangleBound)
  (hRawLe : BRaw + BRaw + BRaw ≤ theorem9UpperBound (maxRhoNorm cset)) :
  p20SamplingProp cset samples := by
  exact p20SamplingProp_of_operand_norm_assumptions_via_universal_blockers_and_raw
    (cset := cset) (samples := samples) (BA := BA) (BB := BB) (BRaw := BRaw)
    hCset hSamples
    (mulRqRawNormBoundFromOperands_of_rawCoeff hRawCoeff)
    hAddTri hSubTri hRawLe

theorem p20SamplingProp_of_operand_norm_assumptions_via_universal_mul_and_add_and_raw
  {cset samples : Array Coeffs}
  {BA BB BRaw : Nat}
  (hCset : ∀ i : Fin cset.size, normInfCoeffs cset[i] ≤ BA)
  (hSamples : ∀ j : Fin samples.size, normInfCoeffs samples[j] ≤ BB)
  (hRaw : mulRqRawNormBoundFromOperands BA BB BRaw)
  (hAddTri : schoolbookAddTriangleBound)
  (hRawLe : BRaw + BRaw + BRaw ≤ theorem9UpperBound (maxRhoNorm cset)) :
  p20SamplingProp cset samples := by
  exact p20SamplingProp_of_operand_norm_assumptions_via_universal_blockers_and_raw
    (cset := cset) (samples := samples) (BA := BA) (BB := BB) (BRaw := BRaw)
    hCset hSamples hRaw hAddTri (schoolbookSubTriangleBound_of_add hAddTri) hRawLe

theorem p20SamplingProp_of_operand_norm_assumptions_via_universal_mul_and_add_and_rawCoeff
  {cset samples : Array Coeffs}
  {BA BB BRaw : Nat}
  (hCset : ∀ i : Fin cset.size, normInfCoeffs cset[i] ≤ BA)
  (hSamples : ∀ j : Fin samples.size, normInfCoeffs samples[j] ≤ BB)
  (hRawCoeff : mulRqRawCoeffBoundFromOperands BA BB BRaw)
  (hAddTri : schoolbookAddTriangleBound)
  (hRawLe : BRaw + BRaw + BRaw ≤ theorem9UpperBound (maxRhoNorm cset)) :
  p20SamplingProp cset samples := by
  exact p20SamplingProp_of_operand_norm_assumptions_via_universal_mul_and_add_and_raw
    (cset := cset) (samples := samples) (BA := BA) (BB := BB) (BRaw := BRaw)
    hCset hSamples
    (mulRqRawNormBoundFromOperands_of_rawCoeff hRawCoeff)
    hAddTri hRawLe

theorem p20SamplingProp_of_operand_norm_assumptions_via_centeredRep_mul_and_add_and_raw
  {cset samples : Array Coeffs}
  {BA BB BRaw : Nat}
  (hCset : ∀ i : Fin cset.size, normInfCoeffs cset[i] ≤ BA)
  (hSamples : ∀ j : Fin samples.size, normInfCoeffs samples[j] ≤ BB)
  (hRaw : mulRqRawNormBoundFromOperands BA BB BRaw)
  (hAddRep :
    ∀ x y : F, Int.natAbs (centeredRep (x + y))
      ≤ Int.natAbs (centeredRep x) + Int.natAbs (centeredRep y))
  (hRawLe : BRaw + BRaw + BRaw ≤ theorem9UpperBound (maxRhoNorm cset)) :
  p20SamplingProp cset samples := by
  exact p20SamplingProp_of_operand_norm_assumptions_via_universal_mul_and_add_and_raw
    (cset := cset) (samples := samples) (BA := BA) (BB := BB) (BRaw := BRaw)
    hCset hSamples hRaw
    (schoolbookAddTriangleBound_of_centeredRep hAddRep)
    hRawLe

theorem p20SamplingProp_of_operand_norm_assumptions_via_centeredRep_mul_and_add_and_rawCoeff
  {cset samples : Array Coeffs}
  {BA BB BRaw : Nat}
  (hCset : ∀ i : Fin cset.size, normInfCoeffs cset[i] ≤ BA)
  (hSamples : ∀ j : Fin samples.size, normInfCoeffs samples[j] ≤ BB)
  (hRawCoeff : mulRqRawCoeffBoundFromOperands BA BB BRaw)
  (hAddRep :
    ∀ x y : F, Int.natAbs (centeredRep (x + y))
      ≤ Int.natAbs (centeredRep x) + Int.natAbs (centeredRep y))
  (hRawLe : BRaw + BRaw + BRaw ≤ theorem9UpperBound (maxRhoNorm cset)) :
  p20SamplingProp cset samples := by
  exact p20SamplingProp_of_operand_norm_assumptions_via_centeredRep_mul_and_add_and_raw
    (cset := cset) (samples := samples) (BA := BA) (BB := BB) (BRaw := BRaw)
    hCset hSamples
    (mulRqRawNormBoundFromOperands_of_rawCoeff hRawCoeff)
    hAddRep hRawLe

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
  exact p20SamplingProp_of_goldilocks_operand_assumptions
    hCset hSamples
    (goldilocksRawNormBoundAssumption_of_inRange hRawInRange)
    hCollapse hUpper

theorem p20SamplingProp_of_goldilocks_operand_fieldOp_assumptions
  {cset samples : Array Coeffs}
  (hCset : ∀ i : Fin cset.size, normInfCoeffs cset[i] ≤ Parameters.Goldilocks.B)
  (hSamples : ∀ j : Fin samples.size, normInfCoeffs samples[j] ≤ Parameters.Goldilocks.B)
  (hRaw : GoldilocksRawNormBoundAssumption)
  (hFieldOps : GoldilocksFieldOpCollapseAssumption)
  (hUpper : Parameters.Goldilocks.B ≤ theorem9UpperBound (maxRhoNorm cset)) :
  p20SamplingProp cset samples := by
  exact p20SamplingProp_of_goldilocks_operand_assumptions
    hCset hSamples hRaw
    (goldilocksRawCollapseAssumption_of_fieldOp hFieldOps)
    hUpper

theorem p20SamplingProp_of_goldilocks_operand_fieldOp_assumptions_inRange
  {cset samples : Array Coeffs}
  (hCset : ∀ i : Fin cset.size, normInfCoeffs cset[i] ≤ Parameters.Goldilocks.B)
  (hSamples : ∀ j : Fin samples.size, normInfCoeffs samples[j] ≤ Parameters.Goldilocks.B)
  (hRawInRange : GoldilocksRawInRangeBoundAssumption)
  (hFieldOps : GoldilocksFieldOpCollapseAssumption)
  (hUpper : Parameters.Goldilocks.B ≤ theorem9UpperBound (maxRhoNorm cset)) :
  p20SamplingProp cset samples := by
  exact p20SamplingProp_of_goldilocks_operand_fieldOp_assumptions
    hCset hSamples
    (goldilocksRawNormBoundAssumption_of_inRange hRawInRange)
    hFieldOps hUpper

theorem p20SamplingProp_of_goldilocks_operand_rawCoeff_assumptions
  {cset samples : Array Coeffs}
  (hCset : ∀ i : Fin cset.size, normInfCoeffs cset[i] ≤ Parameters.Goldilocks.B)
  (hSamples : ∀ j : Fin samples.size, normInfCoeffs samples[j] ≤ Parameters.Goldilocks.B)
  (hRawCoeff : GoldilocksRawCoeffBoundAssumption)
  (hCollapse : GoldilocksRawCollapseAssumption)
  (hUpper : Parameters.Goldilocks.B ≤ theorem9UpperBound (maxRhoNorm cset)) :
  p20SamplingProp cset samples := by
  exact p20SamplingProp_of_goldilocks_operand_assumptions
    hCset hSamples
    (goldilocksRawNormBoundAssumption_of_rawCoeff hRawCoeff)
    hCollapse hUpper

theorem p20SamplingProp_of_goldilocks_operand_rawCoeff_fieldOp_assumptions
  {cset samples : Array Coeffs}
  (hCset : ∀ i : Fin cset.size, normInfCoeffs cset[i] ≤ Parameters.Goldilocks.B)
  (hSamples : ∀ j : Fin samples.size, normInfCoeffs samples[j] ≤ Parameters.Goldilocks.B)
  (hRawCoeff : GoldilocksRawCoeffBoundAssumption)
  (hFieldOps : GoldilocksFieldOpCollapseAssumption)
  (hUpper : Parameters.Goldilocks.B ≤ theorem9UpperBound (maxRhoNorm cset)) :
  p20SamplingProp cset samples := by
  exact p20SamplingProp_of_goldilocks_operand_fieldOp_assumptions
    hCset hSamples
    (goldilocksRawNormBoundAssumption_of_rawCoeff hRawCoeff)
    hFieldOps hUpper

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
  simp [p20DecompProp_recompose_eq hProp]

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

theorem p20DecompProp_of_p6DecompAssumption
  {z : Array F} {b k : Nat}
  (hP6Assm : p6DecompAssumption b k) :
  p20DecompProp z b k := by
  exact ⟨
    p6DecompAssumption_base_ge_two hP6Assm,
    p6DecompAssumption_recompose_eq (z := z) hP6Assm,
    p6DecompAssumption_digitsWithinBaseProp (z := z) hP6Assm
  ⟩

theorem p20DecompProp_of_p6DecompCheckAssumption
  {z : Array F} {b k : Nat}
  (hP6Check : p6DecompCheckAssumption b k) :
  p20DecompProp z b k := by
  exact p20DecompProp_of_p6DecompAssumption
    (z := z) (p6DecompAssumption_of_checkAssumption hP6Check)

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

end SuperNeo
