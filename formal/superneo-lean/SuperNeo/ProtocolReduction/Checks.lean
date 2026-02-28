import SuperNeo.ProtocolReduction.Skeleton

/-! Check/prop bridge layer for protocol reduction surfaces. -/

namespace SuperNeo

/-- Compile-only smoke theorem: check-driven assumptions imply proposition-driven assumptions. -/
theorem smoke_checks_imply_props
  {ctx : PSContext} {claim : PSClaim}
  (hP10 : p10CoreCheck ctx.bar claim.a claim.b = true)
  (hP6 : splitRoundTrip claim.zDecomp ctx.bSplit ctx.kSplit = true)
  (hP12 : matrixTransformIdentity ctx.bar claim.m claim.z = true)
  (hP14 : evalHom2 ctx.bar claim.m claim.z1 claim.z2 claim.r claim.rho1 claim.rho2 = true)
  (hVecAdd : preservesAddVec ctx.hVec claim.z1 claim.z2 = true)
  (hVecScale : preservesScaleVec ctx.hVec claim.rho1 claim.z1 = true)
  (hScalAdd : preservesAddScalar ctx.hScal claim.z1 claim.z2 = true)
  (hScalScale : preservesScaleScalar ctx.hScal claim.rho1 claim.z1 = true)
  (hP16Win : p20InvertibilityWindowProp claim.invDelta)
  (hP17 : samplingSetBoundCheck claim.cset claim.samples = true)
  (hP18Eq : eqLiftAllBoolean claim.qVals ctx.ell = true)
  (hP18SZ : schwartzZippelBoundLeOne ctx.totalDegree ctx.setSize = true)
  (hP19 : interpolationCase claim.xs claim.ys claim.expectedCoeffs claim.evalPoint claim.expectedEval = true) :
  p10ForClaim ctx claim ∧ p20ForClaim ctx claim := by
  refine ⟨p10CoreCheck_sound hP10, ?_⟩
  exact p20ArithmeticBundle_of_checks
    (hP6 := hP6) (hP12 := hP12) (hP14 := hP14)
    (hVecAdd := hVecAdd) (hVecScale := hVecScale)
    (hScalAdd := hScalAdd) (hScalScale := hScalScale)
    (hP16Win := hP16Win)
    (hP17 := hP17) (hP18Eq := hP18Eq) (hP18SZ := hP18SZ) (hP19 := hP19)

/--
Compile-only smoke theorem: proposition assumptions recover the complete
check surface (P10/P6/P12/P14/P15/P16-window/P17/P18/P19).
-/
theorem smoke_props_imply_checks
  {ctx : PSContext} {claim : PSClaim}
  (hProps : p10ForClaim ctx claim ∧ p20ForClaim ctx claim) :
  p10CoreCheck ctx.bar claim.a claim.b = true ∧
    splitRoundTrip claim.zDecomp ctx.bSplit ctx.kSplit = true ∧
    matrixTransformIdentity ctx.bar claim.m claim.z = true ∧
    evalHom2 ctx.bar claim.m claim.z1 claim.z2 claim.r claim.rho1 claim.rho2 = true ∧
    preservesAddVec ctx.hVec claim.z1 claim.z2 = true ∧
    preservesScaleVec ctx.hVec claim.rho1 claim.z1 = true ∧
    preservesAddScalar ctx.hScal claim.z1 claim.z2 = true ∧
    preservesScaleScalar ctx.hScal claim.rho1 claim.z1 = true ∧
    p20InvertibilityWindowProp claim.invDelta ∧
    samplingSetBoundCheck claim.cset claim.samples = true ∧
    eqLiftAllBoolean claim.qVals ctx.ell = true ∧
    schwartzZippelBoundLeOne ctx.totalDegree ctx.setSize = true ∧
    interpolationCase claim.xs claim.ys claim.expectedCoeffs claim.evalPoint claim.expectedEval = true := by
  rcases hProps with ⟨hP10, hP20⟩
  have hP10Check : p10CoreCheck ctx.bar claim.a claim.b = true := p10CoreCheck_complete hP10
  have hP20Checks :
      splitRoundTrip claim.zDecomp ctx.bSplit ctx.kSplit = true ∧
      matrixTransformIdentity ctx.bar claim.m claim.z = true ∧
      evalHom2 ctx.bar claim.m claim.z1 claim.z2 claim.r claim.rho1 claim.rho2 = true ∧
      preservesAddVec ctx.hVec claim.z1 claim.z2 = true ∧
      preservesScaleVec ctx.hVec claim.rho1 claim.z1 = true ∧
      preservesAddScalar ctx.hScal claim.z1 claim.z2 = true ∧
      preservesScaleScalar ctx.hScal claim.rho1 claim.z1 = true ∧
      p20InvertibilityWindowProp claim.invDelta ∧
      samplingSetBoundCheck claim.cset claim.samples = true ∧
      eqLiftAllBoolean claim.qVals ctx.ell = true ∧
      schwartzZippelBoundLeOne ctx.totalDegree ctx.setSize = true ∧
      interpolationCase claim.xs claim.ys claim.expectedCoeffs claim.evalPoint claim.expectedEval = true := by
    exact (p20ArithmeticBundle_iff_checks (bar := ctx.bar) (m := claim.m)
      (z := claim.z) (z1 := claim.z1) (z2 := claim.z2)
      (zDecomp := claim.zDecomp) (r := claim.r)
      (ρ1 := claim.rho1) (ρ2 := claim.rho2)
      (b := ctx.bSplit) (k := ctx.kSplit)
      (hVec := ctx.hVec) (hScal := ctx.hScal)
      (cset := claim.cset) (samples := claim.samples)
      (invDelta := claim.invDelta) (qVals := claim.qVals)
      (xs := claim.xs) (ys := claim.ys) (expectedCoeffs := claim.expectedCoeffs)
      (evalPoint := claim.evalPoint) (expectedEval := claim.expectedEval)
      (ell := ctx.ell) (totalDegree := ctx.totalDegree) (setSize := ctx.setSize)).1 hP20
  exact ⟨hP10Check, hP20Checks⟩

/--
Compile-only smoke theorem: proposition assumptions recover a substantial subset
of regression checks (P10/P6/P12/P14/P15/P17/P18/P19).
-/
theorem smoke_props_imply_check_subset
  {ctx : PSContext} {claim : PSClaim}
  (hProps : p10ForClaim ctx claim ∧ p20ForClaim ctx claim) :
  p10CoreCheck ctx.bar claim.a claim.b = true ∧
    splitRoundTrip claim.zDecomp ctx.bSplit ctx.kSplit = true ∧
    matrixTransformIdentity ctx.bar claim.m claim.z = true ∧
    evalHom2 ctx.bar claim.m claim.z1 claim.z2 claim.r claim.rho1 claim.rho2 = true ∧
    preservesAddVec ctx.hVec claim.z1 claim.z2 = true ∧
    preservesScaleVec ctx.hVec claim.rho1 claim.z1 = true ∧
    preservesAddScalar ctx.hScal claim.z1 claim.z2 = true ∧
    preservesScaleScalar ctx.hScal claim.rho1 claim.z1 = true ∧
    samplingSetBoundCheck claim.cset claim.samples = true ∧
    eqLiftAllBoolean claim.qVals ctx.ell = true ∧
    schwartzZippelBoundLeOne ctx.totalDegree ctx.setSize = true ∧
    interpolationCase claim.xs claim.ys claim.expectedCoeffs claim.evalPoint claim.expectedEval = true := by
  rcases smoke_props_imply_checks hProps with
    ⟨hP10Check, hSplit, hMat, hEvalHom, hVecAdd, hVecScale, hScalAdd, hScalScale, _hP16Win, hSamp, hEq, hSZ, hInterp⟩
  exact ⟨
    hP10Check,
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
Compile-only smoke theorem: proposition/check equivalence at protocol reduction
surface, including the invertibility-window predicate.
-/
theorem smoke_props_iff_checks
  {ctx : PSContext} {claim : PSClaim} :
  (p10ForClaim ctx claim ∧ p20ForClaim ctx claim) ↔
    p10CoreCheck ctx.bar claim.a claim.b = true ∧
    splitRoundTrip claim.zDecomp ctx.bSplit ctx.kSplit = true ∧
    matrixTransformIdentity ctx.bar claim.m claim.z = true ∧
    evalHom2 ctx.bar claim.m claim.z1 claim.z2 claim.r claim.rho1 claim.rho2 = true ∧
    preservesAddVec ctx.hVec claim.z1 claim.z2 = true ∧
    preservesScaleVec ctx.hVec claim.rho1 claim.z1 = true ∧
    preservesAddScalar ctx.hScal claim.z1 claim.z2 = true ∧
    preservesScaleScalar ctx.hScal claim.rho1 claim.z1 = true ∧
    p20InvertibilityWindowProp claim.invDelta ∧
    samplingSetBoundCheck claim.cset claim.samples = true ∧
    eqLiftAllBoolean claim.qVals ctx.ell = true ∧
    schwartzZippelBoundLeOne ctx.totalDegree ctx.setSize = true ∧
    interpolationCase claim.xs claim.ys claim.expectedCoeffs claim.evalPoint claim.expectedEval = true := by
  constructor
  · exact smoke_props_imply_checks
  · intro hChecks
    rcases hChecks with
      ⟨hP10, hP6, hP12, hP14, hVecAdd, hVecScale, hScalAdd, hScalScale, hP16Win, hP17, hP18Eq, hP18SZ, hP19⟩
    exact smoke_checks_imply_props
      (hP10 := hP10) (hP6 := hP6) (hP12 := hP12) (hP14 := hP14)
      (hVecAdd := hVecAdd) (hVecScale := hVecScale)
      (hScalAdd := hScalAdd) (hScalScale := hScalScale)
      (hP16Win := hP16Win)
      (hP17 := hP17) (hP18Eq := hP18Eq) (hP18SZ := hP18SZ) (hP19 := hP19)

/--
Canonical protocol reduction bridge: check-driven assumptions imply proposition
assumptions.
-/
theorem protocol_checks_imply_props
  {ctx : PSContext} {claim : PSClaim}
  (hP10 : p10CoreCheck ctx.bar claim.a claim.b = true)
  (hP6 : splitRoundTrip claim.zDecomp ctx.bSplit ctx.kSplit = true)
  (hP12 : matrixTransformIdentity ctx.bar claim.m claim.z = true)
  (hP14 : evalHom2 ctx.bar claim.m claim.z1 claim.z2 claim.r claim.rho1 claim.rho2 = true)
  (hVecAdd : preservesAddVec ctx.hVec claim.z1 claim.z2 = true)
  (hVecScale : preservesScaleVec ctx.hVec claim.rho1 claim.z1 = true)
  (hScalAdd : preservesAddScalar ctx.hScal claim.z1 claim.z2 = true)
  (hScalScale : preservesScaleScalar ctx.hScal claim.rho1 claim.z1 = true)
  (hP16Win : p20InvertibilityWindowProp claim.invDelta)
  (hP17 : samplingSetBoundCheck claim.cset claim.samples = true)
  (hP18Eq : eqLiftAllBoolean claim.qVals ctx.ell = true)
  (hP18SZ : schwartzZippelBoundLeOne ctx.totalDegree ctx.setSize = true)
  (hP19 : interpolationCase claim.xs claim.ys claim.expectedCoeffs claim.evalPoint claim.expectedEval = true) :
  p10ForClaim ctx claim ∧ p20ForClaim ctx claim := by
  exact smoke_checks_imply_props
    (hP10 := hP10) (hP6 := hP6) (hP12 := hP12) (hP14 := hP14)
    (hVecAdd := hVecAdd) (hVecScale := hVecScale)
    (hScalAdd := hScalAdd) (hScalScale := hScalScale)
    (hP16Win := hP16Win)
    (hP17 := hP17) (hP18Eq := hP18Eq) (hP18SZ := hP18SZ) (hP19 := hP19)

/--
Canonical protocol reduction bridge: proposition assumptions imply the complete
check surface, including invertibility-window.
-/
theorem protocol_props_imply_checks
  {ctx : PSContext} {claim : PSClaim}
  (hProps : p10ForClaim ctx claim ∧ p20ForClaim ctx claim) :
  p10CoreCheck ctx.bar claim.a claim.b = true ∧
    splitRoundTrip claim.zDecomp ctx.bSplit ctx.kSplit = true ∧
    matrixTransformIdentity ctx.bar claim.m claim.z = true ∧
    evalHom2 ctx.bar claim.m claim.z1 claim.z2 claim.r claim.rho1 claim.rho2 = true ∧
    preservesAddVec ctx.hVec claim.z1 claim.z2 = true ∧
    preservesScaleVec ctx.hVec claim.rho1 claim.z1 = true ∧
    preservesAddScalar ctx.hScal claim.z1 claim.z2 = true ∧
    preservesScaleScalar ctx.hScal claim.rho1 claim.z1 = true ∧
    p20InvertibilityWindowProp claim.invDelta ∧
    samplingSetBoundCheck claim.cset claim.samples = true ∧
    eqLiftAllBoolean claim.qVals ctx.ell = true ∧
    schwartzZippelBoundLeOne ctx.totalDegree ctx.setSize = true ∧
    interpolationCase claim.xs claim.ys claim.expectedCoeffs claim.evalPoint claim.expectedEval = true := by
  exact smoke_props_imply_checks hProps

/--
Canonical protocol reduction bridge: proposition assumptions imply the regression
check subset used by the backward-compatible check surface.
-/
theorem protocol_props_imply_check_subset
  {ctx : PSContext} {claim : PSClaim}
  (hProps : p10ForClaim ctx claim ∧ p20ForClaim ctx claim) :
  p10CoreCheck ctx.bar claim.a claim.b = true ∧
    splitRoundTrip claim.zDecomp ctx.bSplit ctx.kSplit = true ∧
    matrixTransformIdentity ctx.bar claim.m claim.z = true ∧
    evalHom2 ctx.bar claim.m claim.z1 claim.z2 claim.r claim.rho1 claim.rho2 = true ∧
    preservesAddVec ctx.hVec claim.z1 claim.z2 = true ∧
    preservesScaleVec ctx.hVec claim.rho1 claim.z1 = true ∧
    preservesAddScalar ctx.hScal claim.z1 claim.z2 = true ∧
    preservesScaleScalar ctx.hScal claim.rho1 claim.z1 = true ∧
    samplingSetBoundCheck claim.cset claim.samples = true ∧
    eqLiftAllBoolean claim.qVals ctx.ell = true ∧
    schwartzZippelBoundLeOne ctx.totalDegree ctx.setSize = true ∧
    interpolationCase claim.xs claim.ys claim.expectedCoeffs claim.evalPoint claim.expectedEval = true := by
  exact smoke_props_imply_check_subset hProps

/--
Canonical protocol reduction equivalence between proposition assumptions and the
complete check surface.
-/
theorem protocol_props_iff_checks
  {ctx : PSContext} {claim : PSClaim} :
  (p10ForClaim ctx claim ∧ p20ForClaim ctx claim) ↔
    p10CoreCheck ctx.bar claim.a claim.b = true ∧
    splitRoundTrip claim.zDecomp ctx.bSplit ctx.kSplit = true ∧
    matrixTransformIdentity ctx.bar claim.m claim.z = true ∧
    evalHom2 ctx.bar claim.m claim.z1 claim.z2 claim.r claim.rho1 claim.rho2 = true ∧
    preservesAddVec ctx.hVec claim.z1 claim.z2 = true ∧
    preservesScaleVec ctx.hVec claim.rho1 claim.z1 = true ∧
    preservesAddScalar ctx.hScal claim.z1 claim.z2 = true ∧
    preservesScaleScalar ctx.hScal claim.rho1 claim.z1 = true ∧
    p20InvertibilityWindowProp claim.invDelta ∧
    samplingSetBoundCheck claim.cset claim.samples = true ∧
    eqLiftAllBoolean claim.qVals ctx.ell = true ∧
    schwartzZippelBoundLeOne ctx.totalDegree ctx.setSize = true ∧
    interpolationCase claim.xs claim.ys claim.expectedCoeffs claim.evalPoint claim.expectedEval = true := by
  exact smoke_props_iff_checks

/--
Compile-only smoke theorem: `p21ProtocolTarget_to_CEValid` composes with
`superneoMathProtocolSkeleton_of_props`.
-/
theorem smoke_p21_compose
  {ctx : PSContext} {claim : PSClaim} {witness : PSWitness}
  (hShape : ClaimShapeValid claim)
  (hBar : IsDBarMatrix ctx.bar)
  (hA : IsDVec claim.a)
  (hB : IsDVec claim.b)
  (hProps : p10ForClaim ctx claim ∧ p20ForClaim ctx claim)
  (hWitness : witness.z = claim.z)
  (hNorm : normInfCoeffs witness.z < ctx.ceNormBound) :
  PSCEValid ctx claim witness := by
  rcases hProps with ⟨hP10, hP20⟩
  exact superneoMathProtocolSkeleton_of_props hShape hBar hA hB hP10 hP20 hWitness hNorm

theorem protocol_props_to_CEValid
  {ctx : PSContext} {claim : PSClaim} {witness : PSWitness}
  (hShape : ClaimShapeValid claim)
  (hBar : IsDBarMatrix ctx.bar)
  (hA : IsDVec claim.a)
  (hB : IsDVec claim.b)
  (hProps : p10ForClaim ctx claim ∧ p20ForClaim ctx claim)
  (hWitness : witness.z = claim.z)
  (hNorm : normInfCoeffs witness.z < ctx.ceNormBound) :
  PSCEValid ctx claim witness := by
  exact smoke_p21_compose hShape hBar hA hB hProps hWitness hNorm


end SuperNeo
