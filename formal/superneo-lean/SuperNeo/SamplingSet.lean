import SuperNeo.InvertibilityAxioms

/-! Sampling-set expansion bounds and theorem/check bridges (P17). -/


namespace SuperNeo

open F

def pairwiseWithinBound (cset : Array Coeffs) (bound : Nat) : Bool :=
  Id.run do
    let mut ok := true
    for i in [0:cset.size] do
      for j in [0:cset.size] do
        if i < j then
          let diff := coeffSub cset[i]! cset[j]!
          ok := ok && decide (normInfCoeffs diff < bound)
    return ok

def strongSamplingSet (cset : Array Coeffs) : Bool :=
  pairwiseWithinBound cset bInvApprox

def maxRhoNorm (cset : Array Coeffs) : Nat :=
  cset.foldl (fun m rho => Nat.max m (normInfCoeffs rho)) 0

theorem normInfCoeffs_le_maxRhoNorm
  (cset : Array Coeffs) (i : Fin cset.size) :
  normInfCoeffs cset[i] ≤ maxRhoNorm cset := by
  unfold maxRhoNorm
  have hAll :
      ∀ t (ht : t < cset.size),
        normInfCoeffs (cset[t]!) ≤ cset.foldl (fun m rho => Nat.max m (normInfCoeffs rho)) 0 := by
    exact Array.foldl_induction
      (as := cset)
      (motive := fun j acc => ∀ t, t < j → normInfCoeffs (cset[t]!) ≤ acc)
      (h0 := by
        intro t ht
        exact (Nat.not_lt_zero t ht).elim)
      (hf := by
        intro j acc hAcc t ht
        by_cases htj : t < j.1
        · exact Nat.le_trans (hAcc t htj) (Nat.le_max_left _ _)
        ·
          have hle : t ≤ j.1 := Nat.le_of_lt_succ ht
          have hge : j.1 ≤ t := Nat.le_of_not_gt htj
          have hEq : t = j.1 := Nat.le_antisymm hle hge
          subst hEq
          simpa [j.2] using (Nat.le_max_right acc (normInfCoeffs (cset[j]))))
  have hI :
      normInfCoeffs (cset[i.1]!) ≤ cset.foldl (fun m rho => Nat.max m (normInfCoeffs rho)) 0 :=
    hAll i.1 i.2
  simpa [i.2] using hI

theorem maxRhoNorm_le_of_forall_norm_le
  {cset : Array Coeffs} {B : Nat}
  (h : ∀ i : Fin cset.size, normInfCoeffs cset[i] ≤ B) :
  maxRhoNorm cset ≤ B := by
  unfold maxRhoNorm
  refine Array.foldl_induction
    (as := cset)
    (init := 0)
    (motive := fun _ acc => acc ≤ B) ?h0 ?hStep
  · exact Nat.zero_le B
  · intro i acc hAcc
    have hI : normInfCoeffs cset[i] ≤ B := h i
    exact (Nat.max_le).2 ⟨hAcc, hI⟩

/-- Theorem 9 interface: expansion factor upper bound 2 * phi(eta) * max||rho||∞. -/
def theorem9UpperBound (maxNorm : Nat) : Nat :=
  2 * d * maxNorm

theorem theorem9UpperBound_mono
  {a b : Nat}
  (h : a ≤ b) :
  theorem9UpperBound a ≤ theorem9UpperBound b := by
  unfold theorem9UpperBound
  simpa [Nat.mul_assoc] using (Nat.mul_le_mul_left (2 * d) h)

theorem theorem9UpperBound_ge_self
  (a : Nat) :
  a ≤ theorem9UpperBound a := by
  have hOneLe : 1 ≤ 2 * d := by
    native_decide
  calc
    a = 1 * a := by simp
    _ ≤ (2 * d) * a := Nat.mul_le_mul_right a hOneLe
    _ = theorem9UpperBound a := by simp [theorem9UpperBound]

private def mulRatio (rho v : Coeffs) : Nat :=
  let denom := normInfCoeffs v
  if denom = 0 then
    0
  else
    normInfCoeffs (mulRq rho v) / denom

def empiricalExpansionFactor (cset : Array Coeffs) (samples : Array Coeffs) : Nat :=
  cset.foldl
    (fun outer rho =>
      samples.foldl (fun inner v => Nat.max inner (mulRatio rho v)) outer)
    0

def samplingSetBoundCheck (cset : Array Coeffs) (samples : Array Coeffs) : Bool :=
  let empirical := empiricalExpansionFactor cset samples
  let bound := theorem9UpperBound (maxRhoNorm cset)
  decide (empirical <= bound)

/-- Proposition form of the Theorem 9 sampling expansion bound. -/
def samplingSetBoundProp (cset : Array Coeffs) (samples : Array Coeffs) : Prop :=
  empiricalExpansionFactor cset samples <= theorem9UpperBound (maxRhoNorm cset)

theorem mulRatio_le_of_mulRq_bound
  {rho v : Coeffs} {B : Nat}
  (hMul : normInfCoeffs (mulRq rho v) ≤ B) :
  mulRatio rho v ≤ B := by
  unfold mulRatio
  by_cases hDen : normInfCoeffs v = 0
  · simp [hDen]
  · simp [hDen]
    exact Nat.le_trans (Nat.div_le_self _ _) hMul

theorem empiricalExpansionFactor_le_of_mulRatio_bound
  {cset samples : Array Coeffs} {B : Nat}
  (hBound : ∀ i : Fin cset.size, ∀ j : Fin samples.size, mulRatio cset[i] samples[j] ≤ B) :
  empiricalExpansionFactor cset samples ≤ B := by
  unfold empiricalExpansionFactor
  refine Array.foldl_induction (as := cset) (motive := fun _ outer => outer ≤ B) ?h0 ?hStep
  · exact Nat.zero_le B
  · intro i outer hOuter
    refine Array.foldl_induction
      (as := samples)
      (init := outer)
      (motive := fun _ inner => inner ≤ B) ?hInner0 ?hInnerStep
    · exact hOuter
    · intro j inner hInner
      have hJ : mulRatio cset[i] samples[j] ≤ B := hBound i j
      exact (Nat.max_le).2 ⟨hInner, hJ⟩

theorem empiricalExpansionFactor_le_of_mulRq_bound
  {cset samples : Array Coeffs} {B : Nat}
  (hMul : ∀ i : Fin cset.size, ∀ j : Fin samples.size,
      normInfCoeffs (mulRq cset[i] samples[j]) ≤ B) :
  empiricalExpansionFactor cset samples ≤ B := by
  exact empiricalExpansionFactor_le_of_mulRatio_bound
    (hBound := fun i j => mulRatio_le_of_mulRq_bound (hMul i j))

/-- Theorem-9-shaped specialization: bound is exactly `theorem9UpperBound (maxRhoNorm cset)`. -/
theorem empiricalExpansionFactor_le_theorem9UpperBound_of_mulRq_bound
  {cset samples : Array Coeffs}
  (hMul : ∀ i : Fin cset.size, ∀ j : Fin samples.size,
      normInfCoeffs (mulRq cset[i] samples[j]) ≤ theorem9UpperBound (maxRhoNorm cset)) :
  empiricalExpansionFactor cset samples ≤ theorem9UpperBound (maxRhoNorm cset) := by
  exact empiricalExpansionFactor_le_of_mulRq_bound
    (cset := cset) (samples := samples)
    (B := theorem9UpperBound (maxRhoNorm cset))
    hMul

theorem empiricalExpansionFactor_le_of_operand_norm_assumptions
  {cset samples : Array Coeffs} {BA BB BRaw B : Nat}
  (hCset : ∀ i : Fin cset.size, normInfCoeffs cset[i] ≤ BA)
  (hSamples : ∀ j : Fin samples.size, normInfCoeffs samples[j] ≤ BB)
  (hRaw : mulRqRawNormBoundFromOperands BA BB BRaw)
  (hAddSub : rawAddSubCollapseBound BRaw B)
  (hSub : rawSubCollapseBound BRaw B) :
  empiricalExpansionFactor cset samples ≤ B := by
  exact empiricalExpansionFactor_le_of_mulRq_bound
    (hMul := fun i j =>
      normInfCoeffs_mulRq_le_of_operand_norm_assumptions
        (a := cset[i]) (b := samples[j])
        (BA := BA) (BB := BB) (BRaw := BRaw) (B := B)
        (hA := hCset i)
        (hB := hSamples j)
        (hRawFromOperands := hRaw)
        (hAddSub := hAddSub)
        (hSub := hSub))

theorem empiricalExpansionFactor_le_of_operand_norm_assumptions_inRange
  {cset samples : Array Coeffs} {BA BB BRaw B : Nat}
  (hCset : ∀ i : Fin cset.size, normInfCoeffs cset[i] ≤ BA)
  (hSamples : ∀ j : Fin samples.size, normInfCoeffs samples[j] ≤ BB)
  (hRawInRange : mulRqRawInRangeBoundFromOperands BA BB BRaw)
  (hAddSub : rawAddSubCollapseBound BRaw B)
  (hSub : rawSubCollapseBound BRaw B) :
  empiricalExpansionFactor cset samples ≤ B := by
  exact empiricalExpansionFactor_le_of_operand_norm_assumptions
    (cset := cset) (samples := samples)
    (BA := BA) (BB := BB) (BRaw := BRaw) (B := B)
    hCset hSamples
    (mulRqRawNormBoundFromOperands_of_inRange hRawInRange)
    hAddSub hSub

theorem empiricalExpansionFactor_le_of_operand_norm_assumptions_rawCoeff
  {cset samples : Array Coeffs} {BA BB BRaw B : Nat}
  (hCset : ∀ i : Fin cset.size, normInfCoeffs cset[i] ≤ BA)
  (hSamples : ∀ j : Fin samples.size, normInfCoeffs samples[j] ≤ BB)
  (hRawCoeff : mulRqRawCoeffBoundFromOperands BA BB BRaw)
  (hAddSub : rawAddSubCollapseBound BRaw B)
  (hSub : rawSubCollapseBound BRaw B) :
  empiricalExpansionFactor cset samples ≤ B := by
  exact empiricalExpansionFactor_le_of_operand_norm_assumptions
    (cset := cset) (samples := samples)
    (BA := BA) (BB := BB) (BRaw := BRaw) (B := B)
    hCset hSamples
    (mulRqRawNormBoundFromOperands_of_rawCoeff hRawCoeff)
    hAddSub hSub

/--
Field-op collapse variant of the empirical expansion constructor.
This eliminates repeated proof boilerplate at call sites that have separate add/sub bounds.
-/
theorem empiricalExpansionFactor_le_of_operand_norm_assumptions_fieldOp
  {cset samples : Array Coeffs} {BA BB B : Nat}
  (hCset : ∀ i : Fin cset.size, normInfCoeffs cset[i] ≤ BA)
  (hSamples : ∀ j : Fin samples.size, normInfCoeffs samples[j] ≤ BB)
  (hRaw : mulRqRawNormBoundFromOperands BA BB B)
  (hOps : rawFieldOpCollapseBound B B) :
  empiricalExpansionFactor cset samples ≤ B := by
  rcases hOps with ⟨hAdd, hSub⟩
  exact empiricalExpansionFactor_le_of_operand_norm_assumptions
    (cset := cset) (samples := samples)
    (BA := BA) (BB := BB) (BRaw := B) (B := B)
    hCset hSamples hRaw
    (rawAddSubCollapseBound_of_add_and_sub_same (BRaw := B) hAdd hSub)
    hSub

theorem empiricalExpansionFactor_le_of_operand_norm_assumptions_fieldOp_inRange
  {cset samples : Array Coeffs} {BA BB B : Nat}
  (hCset : ∀ i : Fin cset.size, normInfCoeffs cset[i] ≤ BA)
  (hSamples : ∀ j : Fin samples.size, normInfCoeffs samples[j] ≤ BB)
  (hRawInRange : mulRqRawInRangeBoundFromOperands BA BB B)
  (hOps : rawFieldOpCollapseBound B B) :
  empiricalExpansionFactor cset samples ≤ B := by
  rcases hOps with ⟨hAdd, hSub⟩
  exact empiricalExpansionFactor_le_of_operand_norm_assumptions_inRange
    (cset := cset) (samples := samples)
    (BA := BA) (BB := BB) (BRaw := B) (B := B)
    hCset hSamples hRawInRange
    (rawAddSubCollapseBound_of_add_and_sub_same (BRaw := B) hAdd hSub)
    hSub

theorem empiricalExpansionFactor_le_of_operand_norm_assumptions_fieldOp_rawCoeff
  {cset samples : Array Coeffs} {BA BB B : Nat}
  (hCset : ∀ i : Fin cset.size, normInfCoeffs cset[i] ≤ BA)
  (hSamples : ∀ j : Fin samples.size, normInfCoeffs samples[j] ≤ BB)
  (hRawCoeff : mulRqRawCoeffBoundFromOperands BA BB B)
  (hOps : rawFieldOpCollapseBound B B) :
  empiricalExpansionFactor cset samples ≤ B := by
  exact empiricalExpansionFactor_le_of_operand_norm_assumptions_fieldOp
    (cset := cset) (samples := samples)
    (BA := BA) (BB := BB) (B := B)
    hCset hSamples
    (mulRqRawNormBoundFromOperands_of_rawCoeff hRawCoeff)
    hOps

theorem empiricalExpansionFactor_le_of_goldilocks_operand_assumptions
  {cset samples : Array Coeffs}
  (hCset : ∀ i : Fin cset.size, normInfCoeffs cset[i] ≤ Parameters.Goldilocks.B)
  (hSamples : ∀ j : Fin samples.size, normInfCoeffs samples[j] ≤ Parameters.Goldilocks.B)
  (hRaw : GoldilocksRawNormBoundAssumption)
  (hCollapse : GoldilocksRawCollapseAssumption) :
  empiricalExpansionFactor cset samples ≤ Parameters.Goldilocks.B := by
  rcases hCollapse with ⟨hAddSub, hSub⟩
  exact empiricalExpansionFactor_le_of_operand_norm_assumptions
    (cset := cset) (samples := samples)
    (BA := Parameters.Goldilocks.B)
    (BB := Parameters.Goldilocks.B)
    (BRaw := Parameters.Goldilocks.B)
    (B := Parameters.Goldilocks.B)
    hCset hSamples hRaw hAddSub hSub

theorem empiricalExpansionFactor_le_of_goldilocks_operand_assumptions_inRange
  {cset samples : Array Coeffs}
  (hCset : ∀ i : Fin cset.size, normInfCoeffs cset[i] ≤ Parameters.Goldilocks.B)
  (hSamples : ∀ j : Fin samples.size, normInfCoeffs samples[j] ≤ Parameters.Goldilocks.B)
  (hRawInRange : GoldilocksRawInRangeBoundAssumption)
  (hCollapse : GoldilocksRawCollapseAssumption) :
  empiricalExpansionFactor cset samples ≤ Parameters.Goldilocks.B := by
  exact empiricalExpansionFactor_le_of_goldilocks_operand_assumptions
    hCset hSamples
    (goldilocksRawNormBoundAssumption_of_inRange hRawInRange)
    hCollapse

theorem empiricalExpansionFactor_le_of_goldilocks_operand_fieldOp_assumptions
  {cset samples : Array Coeffs}
  (hCset : ∀ i : Fin cset.size, normInfCoeffs cset[i] ≤ Parameters.Goldilocks.B)
  (hSamples : ∀ j : Fin samples.size, normInfCoeffs samples[j] ≤ Parameters.Goldilocks.B)
  (hRaw : GoldilocksRawNormBoundAssumption)
  (hFieldOps : GoldilocksFieldOpCollapseAssumption) :
  empiricalExpansionFactor cset samples ≤ Parameters.Goldilocks.B := by
  exact empiricalExpansionFactor_le_of_goldilocks_operand_assumptions
    hCset hSamples hRaw
    (goldilocksRawCollapseAssumption_of_fieldOp hFieldOps)

theorem empiricalExpansionFactor_le_of_goldilocks_operand_fieldOp_assumptions_inRange
  {cset samples : Array Coeffs}
  (hCset : ∀ i : Fin cset.size, normInfCoeffs cset[i] ≤ Parameters.Goldilocks.B)
  (hSamples : ∀ j : Fin samples.size, normInfCoeffs samples[j] ≤ Parameters.Goldilocks.B)
  (hRawInRange : GoldilocksRawInRangeBoundAssumption)
  (hFieldOps : GoldilocksFieldOpCollapseAssumption) :
  empiricalExpansionFactor cset samples ≤ Parameters.Goldilocks.B := by
  exact empiricalExpansionFactor_le_of_goldilocks_operand_assumptions_inRange
    hCset hSamples hRawInRange
    (goldilocksRawCollapseAssumption_of_fieldOp hFieldOps)

theorem empiricalExpansionFactor_le_of_goldilocks_operand_rawCoeff_assumptions
  {cset samples : Array Coeffs}
  (hCset : ∀ i : Fin cset.size, normInfCoeffs cset[i] ≤ Parameters.Goldilocks.B)
  (hSamples : ∀ j : Fin samples.size, normInfCoeffs samples[j] ≤ Parameters.Goldilocks.B)
  (hRawCoeff : GoldilocksRawCoeffBoundAssumption)
  (hCollapse : GoldilocksRawCollapseAssumption) :
  empiricalExpansionFactor cset samples ≤ Parameters.Goldilocks.B := by
  exact empiricalExpansionFactor_le_of_goldilocks_operand_assumptions
    hCset hSamples
    (goldilocksRawNormBoundAssumption_of_rawCoeff hRawCoeff)
    hCollapse

theorem empiricalExpansionFactor_le_of_goldilocks_operand_rawCoeff_fieldOp_assumptions
  {cset samples : Array Coeffs}
  (hCset : ∀ i : Fin cset.size, normInfCoeffs cset[i] ≤ Parameters.Goldilocks.B)
  (hSamples : ∀ j : Fin samples.size, normInfCoeffs samples[j] ≤ Parameters.Goldilocks.B)
  (hRawCoeff : GoldilocksRawCoeffBoundAssumption)
  (hFieldOps : GoldilocksFieldOpCollapseAssumption) :
  empiricalExpansionFactor cset samples ≤ Parameters.Goldilocks.B := by
  exact empiricalExpansionFactor_le_of_goldilocks_operand_rawCoeff_assumptions
    hCset hSamples hRawCoeff
    (goldilocksRawCollapseAssumption_of_fieldOp hFieldOps)

/--
Theorem-native Theorem 9 constructor:
derive the full expansion bound directly from operand-norm assumptions (no check-only dependence).
-/
theorem empiricalExpansionFactor_le_theorem9UpperBound_of_operand_norm_assumptions
  {cset samples : Array Coeffs} {BB BRaw : Nat}
  (hSamples : ∀ j : Fin samples.size, normInfCoeffs samples[j] ≤ BB)
  (hRaw : mulRqRawNormBoundFromOperands (maxRhoNorm cset) BB BRaw)
  (hAddSub : rawAddSubCollapseBound BRaw (theorem9UpperBound (maxRhoNorm cset)))
  (hSub : rawSubCollapseBound BRaw (theorem9UpperBound (maxRhoNorm cset))) :
  empiricalExpansionFactor cset samples ≤ theorem9UpperBound (maxRhoNorm cset) := by
  refine empiricalExpansionFactor_le_theorem9UpperBound_of_mulRq_bound (cset := cset) (samples := samples) ?_
  intro i j
  have hA : normInfCoeffs cset[i] ≤ maxRhoNorm cset :=
    normInfCoeffs_le_maxRhoNorm (cset := cset) i
  have hB : normInfCoeffs samples[j] ≤ BB :=
    hSamples j
  exact normInfCoeffs_mulRq_le_of_operand_norm_assumptions
    (a := cset[i]) (b := samples[j])
    (BA := maxRhoNorm cset) (BB := BB) (BRaw := BRaw) (B := theorem9UpperBound (maxRhoNorm cset))
    hA hB hRaw hAddSub hSub

theorem empiricalExpansionFactor_le_theorem9UpperBound_of_operand_norm_assumptions_inRange
  {cset samples : Array Coeffs} {BB BRaw : Nat}
  (hSamples : ∀ j : Fin samples.size, normInfCoeffs samples[j] ≤ BB)
  (hRawInRange : mulRqRawInRangeBoundFromOperands (maxRhoNorm cset) BB BRaw)
  (hAddSub : rawAddSubCollapseBound BRaw (theorem9UpperBound (maxRhoNorm cset)))
  (hSub : rawSubCollapseBound BRaw (theorem9UpperBound (maxRhoNorm cset))) :
  empiricalExpansionFactor cset samples ≤ theorem9UpperBound (maxRhoNorm cset) := by
  exact empiricalExpansionFactor_le_theorem9UpperBound_of_operand_norm_assumptions
    (cset := cset) (samples := samples) (BB := BB) (BRaw := BRaw)
    hSamples
    (mulRqRawNormBoundFromOperands_of_inRange
      (BA := maxRhoNorm cset) (BB := BB) (BRaw := BRaw)
      hRawInRange)
    hAddSub hSub

theorem samplingSetBoundCheck_sound
  {cset : Array Coeffs} {samples : Array Coeffs}
  (hOk : samplingSetBoundCheck cset samples = true) :
  empiricalExpansionFactor cset samples <= theorem9UpperBound (maxRhoNorm cset) := by
  unfold samplingSetBoundCheck at hOk
  exact decide_eq_true_eq.mp hOk

theorem samplingSetBoundCheck_complete
  {cset : Array Coeffs} {samples : Array Coeffs}
  (hBound : empiricalExpansionFactor cset samples <= theorem9UpperBound (maxRhoNorm cset)) :
  samplingSetBoundCheck cset samples = true := by
  unfold samplingSetBoundCheck
  exact decide_eq_true hBound

theorem samplingSetBoundCheck_iff_prop
  {cset : Array Coeffs} {samples : Array Coeffs} :
  samplingSetBoundCheck cset samples = true ↔ samplingSetBoundProp cset samples := by
  constructor
  · intro hOk
    exact samplingSetBoundCheck_sound (cset := cset) (samples := samples) hOk
  · intro h
    exact samplingSetBoundCheck_complete (cset := cset) (samples := samples) h

/-- Convenience: recover the regression check flag from theorem-native hypotheses. -/
theorem samplingSetBoundCheck_true_of_operand_norm_assumptions
  {cset samples : Array Coeffs} {BB BRaw : Nat}
  (hSamples : ∀ j : Fin samples.size, normInfCoeffs samples[j] ≤ BB)
  (hRaw : mulRqRawNormBoundFromOperands (maxRhoNorm cset) BB BRaw)
  (hAddSub : rawAddSubCollapseBound BRaw (theorem9UpperBound (maxRhoNorm cset)))
  (hSub : rawSubCollapseBound BRaw (theorem9UpperBound (maxRhoNorm cset))) :
  samplingSetBoundCheck cset samples = true := by
  apply samplingSetBoundCheck_complete
  exact empiricalExpansionFactor_le_theorem9UpperBound_of_operand_norm_assumptions
    (cset := cset) (samples := samples) (BB := BB) (BRaw := BRaw)
    hSamples hRaw hAddSub hSub

theorem samplingSetBoundCheck_true_of_operand_norm_assumptions_inRange
  {cset samples : Array Coeffs} {BB BRaw : Nat}
  (hSamples : ∀ j : Fin samples.size, normInfCoeffs samples[j] ≤ BB)
  (hRawInRange : mulRqRawInRangeBoundFromOperands (maxRhoNorm cset) BB BRaw)
  (hAddSub : rawAddSubCollapseBound BRaw (theorem9UpperBound (maxRhoNorm cset)))
  (hSub : rawSubCollapseBound BRaw (theorem9UpperBound (maxRhoNorm cset))) :
  samplingSetBoundCheck cset samples = true := by
  apply samplingSetBoundCheck_complete
  exact empiricalExpansionFactor_le_theorem9UpperBound_of_operand_norm_assumptions_inRange
    (cset := cset) (samples := samples) (BB := BB) (BRaw := BRaw)
    hSamples hRawInRange hAddSub hSub

theorem samplingSetBoundCheck_true_of_goldilocks_operand_assumptions
  {cset samples : Array Coeffs}
  (hCset : ∀ i : Fin cset.size, normInfCoeffs cset[i] ≤ Parameters.Goldilocks.B)
  (hSamples : ∀ j : Fin samples.size, normInfCoeffs samples[j] ≤ Parameters.Goldilocks.B)
  (hRaw : GoldilocksRawNormBoundAssumption)
  (hCollapse : GoldilocksRawCollapseAssumption)
  (hBLe : Parameters.Goldilocks.B ≤ theorem9UpperBound (maxRhoNorm cset)) :
  samplingSetBoundCheck cset samples = true := by
  apply samplingSetBoundCheck_complete
  exact Nat.le_trans
    (empiricalExpansionFactor_le_of_goldilocks_operand_assumptions
      (cset := cset) (samples := samples) hCset hSamples hRaw hCollapse)
    hBLe

theorem samplingSetBoundCheck_true_of_goldilocks_operand_assumptions_inRange
  {cset samples : Array Coeffs}
  (hCset : ∀ i : Fin cset.size, normInfCoeffs cset[i] ≤ Parameters.Goldilocks.B)
  (hSamples : ∀ j : Fin samples.size, normInfCoeffs samples[j] ≤ Parameters.Goldilocks.B)
  (hRawInRange : GoldilocksRawInRangeBoundAssumption)
  (hCollapse : GoldilocksRawCollapseAssumption)
  (hBLe : Parameters.Goldilocks.B ≤ theorem9UpperBound (maxRhoNorm cset)) :
  samplingSetBoundCheck cset samples = true := by
  apply samplingSetBoundCheck_complete
  exact Nat.le_trans
    (empiricalExpansionFactor_le_of_goldilocks_operand_assumptions_inRange
      (cset := cset) (samples := samples) hCset hSamples hRawInRange hCollapse)
    hBLe

theorem samplingSetBoundCheck_true_of_goldilocks_operand_fieldOp_assumptions
  {cset samples : Array Coeffs}
  (hCset : ∀ i : Fin cset.size, normInfCoeffs cset[i] ≤ Parameters.Goldilocks.B)
  (hSamples : ∀ j : Fin samples.size, normInfCoeffs samples[j] ≤ Parameters.Goldilocks.B)
  (hRaw : GoldilocksRawNormBoundAssumption)
  (hFieldOps : GoldilocksFieldOpCollapseAssumption)
  (hBLe : Parameters.Goldilocks.B ≤ theorem9UpperBound (maxRhoNorm cset)) :
  samplingSetBoundCheck cset samples = true := by
  apply samplingSetBoundCheck_complete
  exact Nat.le_trans
    (empiricalExpansionFactor_le_of_goldilocks_operand_fieldOp_assumptions
      (cset := cset) (samples := samples) hCset hSamples hRaw hFieldOps)
    hBLe

theorem samplingSetBoundCheck_true_of_goldilocks_operand_fieldOp_assumptions_inRange
  {cset samples : Array Coeffs}
  (hCset : ∀ i : Fin cset.size, normInfCoeffs cset[i] ≤ Parameters.Goldilocks.B)
  (hSamples : ∀ j : Fin samples.size, normInfCoeffs samples[j] ≤ Parameters.Goldilocks.B)
  (hRawInRange : GoldilocksRawInRangeBoundAssumption)
  (hFieldOps : GoldilocksFieldOpCollapseAssumption)
  (hBLe : Parameters.Goldilocks.B ≤ theorem9UpperBound (maxRhoNorm cset)) :
  samplingSetBoundCheck cset samples = true := by
  apply samplingSetBoundCheck_complete
  exact Nat.le_trans
    (empiricalExpansionFactor_le_of_goldilocks_operand_fieldOp_assumptions_inRange
      (cset := cset) (samples := samples) hCset hSamples hRawInRange hFieldOps)
    hBLe

theorem samplingSetBoundCheck_true_of_goldilocks_operand_rawCoeff_assumptions
  {cset samples : Array Coeffs}
  (hCset : ∀ i : Fin cset.size, normInfCoeffs cset[i] ≤ Parameters.Goldilocks.B)
  (hSamples : ∀ j : Fin samples.size, normInfCoeffs samples[j] ≤ Parameters.Goldilocks.B)
  (hRawCoeff : GoldilocksRawCoeffBoundAssumption)
  (hCollapse : GoldilocksRawCollapseAssumption)
  (hBLe : Parameters.Goldilocks.B ≤ theorem9UpperBound (maxRhoNorm cset)) :
  samplingSetBoundCheck cset samples = true := by
  apply samplingSetBoundCheck_complete
  exact Nat.le_trans
    (empiricalExpansionFactor_le_of_goldilocks_operand_rawCoeff_assumptions
      (cset := cset) (samples := samples) hCset hSamples hRawCoeff hCollapse)
    hBLe

theorem samplingSetBoundCheck_true_of_goldilocks_operand_rawCoeff_fieldOp_assumptions
  {cset samples : Array Coeffs}
  (hCset : ∀ i : Fin cset.size, normInfCoeffs cset[i] ≤ Parameters.Goldilocks.B)
  (hSamples : ∀ j : Fin samples.size, normInfCoeffs samples[j] ≤ Parameters.Goldilocks.B)
  (hRawCoeff : GoldilocksRawCoeffBoundAssumption)
  (hFieldOps : GoldilocksFieldOpCollapseAssumption)
  (hBLe : Parameters.Goldilocks.B ≤ theorem9UpperBound (maxRhoNorm cset)) :
  samplingSetBoundCheck cset samples = true := by
  apply samplingSetBoundCheck_complete
  exact Nat.le_trans
    (empiricalExpansionFactor_le_of_goldilocks_operand_rawCoeff_fieldOp_assumptions
      (cset := cset) (samples := samples) hCset hSamples hRawCoeff hFieldOps)
    hBLe

theorem goldilocksB_le_theorem9UpperBound_of_maxRhoNorm_ge
  {cset : Array Coeffs}
  (hMax : Parameters.Goldilocks.B ≤ maxRhoNorm cset) :
  Parameters.Goldilocks.B ≤ theorem9UpperBound (maxRhoNorm cset) := by
  exact Nat.le_trans hMax (theorem9UpperBound_ge_self (maxRhoNorm cset))

theorem samplingSetBoundCheck_true_of_goldilocks_operand_assumptions_of_maxRhoNorm_ge
  {cset samples : Array Coeffs}
  (hCset : ∀ i : Fin cset.size, normInfCoeffs cset[i] ≤ Parameters.Goldilocks.B)
  (hSamples : ∀ j : Fin samples.size, normInfCoeffs samples[j] ≤ Parameters.Goldilocks.B)
  (hRaw : GoldilocksRawNormBoundAssumption)
  (hCollapse : GoldilocksRawCollapseAssumption)
  (hMax : Parameters.Goldilocks.B ≤ maxRhoNorm cset) :
  samplingSetBoundCheck cset samples = true := by
  exact samplingSetBoundCheck_true_of_goldilocks_operand_assumptions
    hCset hSamples hRaw hCollapse
    (goldilocksB_le_theorem9UpperBound_of_maxRhoNorm_ge hMax)

theorem samplingSetBoundCheck_true_of_goldilocks_operand_assumptions_inRange_of_maxRhoNorm_ge
  {cset samples : Array Coeffs}
  (hCset : ∀ i : Fin cset.size, normInfCoeffs cset[i] ≤ Parameters.Goldilocks.B)
  (hSamples : ∀ j : Fin samples.size, normInfCoeffs samples[j] ≤ Parameters.Goldilocks.B)
  (hRawInRange : GoldilocksRawInRangeBoundAssumption)
  (hCollapse : GoldilocksRawCollapseAssumption)
  (hMax : Parameters.Goldilocks.B ≤ maxRhoNorm cset) :
  samplingSetBoundCheck cset samples = true := by
  exact samplingSetBoundCheck_true_of_goldilocks_operand_assumptions_inRange
    hCset hSamples hRawInRange hCollapse
    (goldilocksB_le_theorem9UpperBound_of_maxRhoNorm_ge hMax)

theorem samplingSetBoundCheck_true_of_goldilocks_operand_fieldOp_assumptions_of_maxRhoNorm_ge
  {cset samples : Array Coeffs}
  (hCset : ∀ i : Fin cset.size, normInfCoeffs cset[i] ≤ Parameters.Goldilocks.B)
  (hSamples : ∀ j : Fin samples.size, normInfCoeffs samples[j] ≤ Parameters.Goldilocks.B)
  (hRaw : GoldilocksRawNormBoundAssumption)
  (hFieldOps : GoldilocksFieldOpCollapseAssumption)
  (hMax : Parameters.Goldilocks.B ≤ maxRhoNorm cset) :
  samplingSetBoundCheck cset samples = true := by
  exact samplingSetBoundCheck_true_of_goldilocks_operand_fieldOp_assumptions
    hCset hSamples hRaw hFieldOps
    (goldilocksB_le_theorem9UpperBound_of_maxRhoNorm_ge hMax)

theorem samplingSetBoundCheck_true_of_goldilocks_operand_fieldOp_assumptions_inRange_of_maxRhoNorm_ge
  {cset samples : Array Coeffs}
  (hCset : ∀ i : Fin cset.size, normInfCoeffs cset[i] ≤ Parameters.Goldilocks.B)
  (hSamples : ∀ j : Fin samples.size, normInfCoeffs samples[j] ≤ Parameters.Goldilocks.B)
  (hRawInRange : GoldilocksRawInRangeBoundAssumption)
  (hFieldOps : GoldilocksFieldOpCollapseAssumption)
  (hMax : Parameters.Goldilocks.B ≤ maxRhoNorm cset) :
  samplingSetBoundCheck cset samples = true := by
  exact samplingSetBoundCheck_true_of_goldilocks_operand_fieldOp_assumptions_inRange
    hCset hSamples hRawInRange hFieldOps
    (goldilocksB_le_theorem9UpperBound_of_maxRhoNorm_ge hMax)

theorem samplingSetBoundCheck_true_of_goldilocks_operand_rawCoeff_assumptions_of_maxRhoNorm_ge
  {cset samples : Array Coeffs}
  (hCset : ∀ i : Fin cset.size, normInfCoeffs cset[i] ≤ Parameters.Goldilocks.B)
  (hSamples : ∀ j : Fin samples.size, normInfCoeffs samples[j] ≤ Parameters.Goldilocks.B)
  (hRawCoeff : GoldilocksRawCoeffBoundAssumption)
  (hCollapse : GoldilocksRawCollapseAssumption)
  (hMax : Parameters.Goldilocks.B ≤ maxRhoNorm cset) :
  samplingSetBoundCheck cset samples = true := by
  exact samplingSetBoundCheck_true_of_goldilocks_operand_rawCoeff_assumptions
    hCset hSamples hRawCoeff hCollapse
    (goldilocksB_le_theorem9UpperBound_of_maxRhoNorm_ge hMax)

theorem samplingSetBoundCheck_true_of_goldilocks_operand_rawCoeff_fieldOp_assumptions_of_maxRhoNorm_ge
  {cset samples : Array Coeffs}
  (hCset : ∀ i : Fin cset.size, normInfCoeffs cset[i] ≤ Parameters.Goldilocks.B)
  (hSamples : ∀ j : Fin samples.size, normInfCoeffs samples[j] ≤ Parameters.Goldilocks.B)
  (hRawCoeff : GoldilocksRawCoeffBoundAssumption)
  (hFieldOps : GoldilocksFieldOpCollapseAssumption)
  (hMax : Parameters.Goldilocks.B ≤ maxRhoNorm cset) :
  samplingSetBoundCheck cset samples = true := by
  exact samplingSetBoundCheck_true_of_goldilocks_operand_rawCoeff_fieldOp_assumptions
    hCset hSamples hRawCoeff hFieldOps
    (goldilocksB_le_theorem9UpperBound_of_maxRhoNorm_ge hMax)

def samplingSetSanity : Bool :=
  decide (theorem9UpperBound 2 = 216)

end SuperNeo
