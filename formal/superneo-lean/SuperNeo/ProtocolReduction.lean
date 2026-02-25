import SuperNeo.ProtocolRelations

namespace SuperNeo

def p10ForClaim (ctx : ProtocolCtx) (claim : CEClaim) : Prop :=
  p10CoreProp ctx.bar claim.a claim.b

def p20ForClaim (ctx : ProtocolCtx) (claim : CEClaim) : Prop :=
  p20ArithmeticBundle
    ctx.bar
    claim.m
    claim.z claim.z1 claim.z2 claim.zDecomp claim.r
    claim.rho1 claim.rho2
    ctx.bSplit ctx.kSplit
    ctx.hVec ctx.hScal
    claim.cset claim.samples claim.qVals
    claim.xs claim.ys claim.expectedCoeffs
    claim.evalPoint claim.expectedEval
    ctx.ell ctx.totalDegree ctx.setSize

theorem superneoMathProtocolSkeleton_of_props
  {ctx : ProtocolCtx} {claim : CEClaim} {witness : CEWitness}
  (hShape : ClaimShapeValid claim)
  (hBar : IsDBarMatrix ctx.bar)
  (hA : IsDVec claim.a)
  (hB : IsDVec claim.b)
  (hP10 : p10ForClaim ctx claim)
  (hP20 : p20ForClaim ctx claim)
  (hWitness : witness.z = claim.z)
  (hNorm : normInfCoeffs witness.z < ctx.ceNormBound) :
  CEValid ctx claim witness := by
  have hP21 : p21ProtocolTarget
      ctx.bar
      claim.m
      claim.z claim.z1 claim.z2 claim.zDecomp claim.r
      claim.rho1 claim.rho2
      ctx.bSplit ctx.kSplit
      claim.cset claim.samples claim.qVals
      claim.xs claim.ys claim.expectedCoeffs
      claim.evalPoint claim.expectedEval
      ctx.ell ctx.totalDegree ctx.setSize := by
    exact p21ProtocolTarget_of_p20 hP20
  exact p21ProtocolTarget_to_CEValid hShape hBar hA hB hP21 hP10 hWitness hNorm

theorem superneoMathProtocolSkeleton_of_thm3_assumption
  {ctx : ProtocolCtx} {claim : CEClaim} {witness : CEWitness}
  (hShape : ClaimShapeValid claim)
  (hBar : IsDBarMatrix ctx.bar)
  (hA : IsDVec claim.a)
  (hB : IsDVec claim.b)
  (hThm3 : thm3CoreAssumption ctx.bar)
  (hP20 : p20ForClaim ctx claim)
  (hWitness : witness.z = claim.z)
  (hNorm : normInfCoeffs witness.z < ctx.ceNormBound) :
  CEValid ctx claim witness := by
  have hP10 : p10ForClaim ctx claim := p10Core_of_assumption hThm3 hA hB
  exact superneoMathProtocolSkeleton_of_props hShape hBar hA hB hP10 hP20 hWitness hNorm

theorem superneoMathProtocolSkeleton_of_checks
  {ctx : ProtocolCtx} {claim : CEClaim} {witness : CEWitness}
  (hShape : ClaimShapeValid claim)
  (hBar : IsDBarMatrix ctx.bar)
  (hA : IsDVec claim.a)
  (hB : IsDVec claim.b)
  (hP10 : p10CoreCheck ctx.bar claim.a claim.b = true)
  (hP6 : splitRoundTrip claim.zDecomp ctx.bSplit ctx.kSplit = true)
  (hP12 : matrixTransformIdentity ctx.bar claim.m claim.z = true)
  (hP14 : evalHom2 ctx.bar claim.m claim.z1 claim.z2 claim.r claim.rho1 claim.rho2 = true)
  (hVecAdd : preservesAddVec ctx.hVec claim.z1 claim.z2 = true)
  (hVecScale : preservesScaleVec ctx.hVec claim.rho1 claim.z1 = true)
  (hScalAdd : preservesAddScalar ctx.hScal claim.z1 claim.z2 = true)
  (hScalScale : preservesScaleScalar ctx.hScal claim.rho1 claim.z1 = true)
  (hP17 : samplingSetBoundCheck claim.cset claim.samples = true)
  (hP18Eq : eqLiftAllBoolean claim.qVals ctx.ell = true)
  (hP18SZ : schwartzZippelBoundLeOne ctx.totalDegree ctx.setSize = true)
  (hP19 : interpolationCase claim.xs claim.ys claim.expectedCoeffs claim.evalPoint claim.expectedEval = true)
  (hWitness : witness.z = claim.z)
  (hNorm : normInfCoeffs witness.z < ctx.ceNormBound) :
  CEValid ctx claim witness := by
  have hFull : p21FullMathTarget
      ctx.bar
      claim.a claim.b
      claim.m
      claim.z claim.z1 claim.z2 claim.zDecomp claim.r
      claim.rho1 claim.rho2
      ctx.bSplit ctx.kSplit
      claim.cset claim.samples claim.qVals
      claim.xs claim.ys claim.expectedCoeffs
      claim.evalPoint claim.expectedEval
      ctx.ell ctx.totalDegree ctx.setSize := by
    exact p21FullMathTarget_of_checks
      (hP10 := hP10) (hP6 := hP6) (hP12 := hP12) (hP14 := hP14)
      (hVecAdd := hVecAdd) (hVecScale := hVecScale)
      (hScalAdd := hScalAdd) (hScalScale := hScalScale)
      (hP17 := hP17) (hP18Eq := hP18Eq) (hP18SZ := hP18SZ) (hP19 := hP19)
  exact p21FullMathTarget_to_CEValid hShape hBar hA hB hFull hWitness hNorm

/-- Compile-only smoke theorem: check-driven assumptions imply proposition-driven assumptions. -/
theorem smoke_checks_imply_props
  {ctx : ProtocolCtx} {claim : CEClaim}
  (hP10 : p10CoreCheck ctx.bar claim.a claim.b = true)
  (hP6 : splitRoundTrip claim.zDecomp ctx.bSplit ctx.kSplit = true)
  (hP12 : matrixTransformIdentity ctx.bar claim.m claim.z = true)
  (hP14 : evalHom2 ctx.bar claim.m claim.z1 claim.z2 claim.r claim.rho1 claim.rho2 = true)
  (hVecAdd : preservesAddVec ctx.hVec claim.z1 claim.z2 = true)
  (hVecScale : preservesScaleVec ctx.hVec claim.rho1 claim.z1 = true)
  (hScalAdd : preservesAddScalar ctx.hScal claim.z1 claim.z2 = true)
  (hScalScale : preservesScaleScalar ctx.hScal claim.rho1 claim.z1 = true)
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
    (hP17 := hP17) (hP18Eq := hP18Eq) (hP18SZ := hP18SZ) (hP19 := hP19)

/--
Compile-only smoke theorem: proposition assumptions recover a substantial subset
of regression checks (P10/P6/P15/P17/P18/P19).
-/
theorem smoke_props_imply_check_subset
  {ctx : ProtocolCtx} {claim : CEClaim}
  (hShape : ClaimShapeValid claim)
  (hProps : p10ForClaim ctx claim ∧ p20ForClaim ctx claim) :
  p10CoreCheck ctx.bar claim.a claim.b = true ∧
    splitRoundTrip claim.zDecomp ctx.bSplit ctx.kSplit = true ∧
    matrixTransformIdentity ctx.bar claim.m claim.z = true ∧
    preservesAddVec ctx.hVec claim.z1 claim.z2 = true ∧
    preservesScaleVec ctx.hVec claim.rho1 claim.z1 = true ∧
    preservesAddScalar ctx.hScal claim.z1 claim.z2 = true ∧
    preservesScaleScalar ctx.hScal claim.rho1 claim.z1 = true ∧
    samplingSetBoundCheck claim.cset claim.samples = true ∧
    eqLiftAllBoolean claim.qVals ctx.ell = true ∧
    schwartzZippelBoundLeOne ctx.totalDegree ctx.setSize = true ∧
    interpolationCase claim.xs claim.ys claim.expectedCoeffs claim.evalPoint claim.expectedEval = true := by
  rcases hProps with ⟨hP10, hP20⟩
  have hP10Check : p10CoreCheck ctx.bar claim.a claim.b = true := p10CoreCheck_complete hP10
  have hSubset :
      splitRoundTrip claim.zDecomp ctx.bSplit ctx.kSplit = true ∧
      matrixTransformIdentity ctx.bar claim.m claim.z = true ∧
      samplingSetBoundCheck claim.cset claim.samples = true ∧
      eqLiftAllBoolean claim.qVals ctx.ell = true ∧
      schwartzZippelBoundLeOne ctx.totalDegree ctx.setSize = true ∧
      interpolationCase claim.xs claim.ys claim.expectedCoeffs claim.evalPoint claim.expectedEval = true := by
    exact p20ArithmeticBundle_props_imply_check_subset hP20
  have hModule :
      preservesAddVec ctx.hVec claim.z1 claim.z2 = true ∧
      preservesScaleVec ctx.hVec claim.rho1 claim.z1 = true ∧
      preservesAddScalar ctx.hScal claim.z1 claim.z2 = true ∧
      preservesScaleScalar ctx.hScal claim.rho1 claim.z1 = true := by
    exact p20ArithmeticBundle_props_imply_module_checks hShape.1 hP20
  exact ⟨
    hP10Check,
    hSubset.1,
    hSubset.2.1,
    hModule.1,
    hModule.2.1,
    hModule.2.2.1,
    hModule.2.2.2,
    hSubset.2.2.1,
    hSubset.2.2.2.1,
    hSubset.2.2.2.2.1,
    hSubset.2.2.2.2.2
  ⟩

/--
Compile-only smoke theorem: `p21ProtocolTarget_to_CEValid` composes with
`superneoMathProtocolSkeleton_of_props`.
-/
theorem smoke_p21_compose
  {ctx : ProtocolCtx} {claim : CEClaim} {witness : CEWitness}
  (hShape : ClaimShapeValid claim)
  (hBar : IsDBarMatrix ctx.bar)
  (hA : IsDVec claim.a)
  (hB : IsDVec claim.b)
  (hProps : p10ForClaim ctx claim ∧ p20ForClaim ctx claim)
  (hWitness : witness.z = claim.z)
  (hNorm : normInfCoeffs witness.z < ctx.ceNormBound) :
  CEValid ctx claim witness := by
  exact superneoMathProtocolSkeleton_of_props hShape hBar hA hB hProps.1 hProps.2 hWitness hNorm

end SuperNeo
