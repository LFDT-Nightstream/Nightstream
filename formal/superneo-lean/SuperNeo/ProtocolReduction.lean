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
    claim.cset claim.samples claim.invDelta claim.qVals
    claim.xs claim.ys claim.expectedCoeffs
    claim.evalPoint claim.expectedEval
    ctx.ell ctx.totalDegree ctx.setSize

theorem p20ForClaim_of_props_with_thm3CoreAssumption
  {ctx : ProtocolCtx} {claim : CEClaim}
  (hP6 : p20DecompProp claim.zDecomp ctx.bSplit ctx.kSplit)
  (hP12Rows : MatrixRowsCompatible claim.m claim.z)
  (hThm3 : thm3CoreAssumption ctx.bar)
  (hP14 : p20EvalHomProp ctx.bar claim.m claim.z1 claim.z2 claim.r claim.rho1 claim.rho2)
  (hP15Vec : p20VecModuleProp ctx.hVec claim.rho1 claim.z1 claim.z2)
  (hP15Scal : p20ScalarModuleProp ctx.hScal claim.rho1 claim.z1 claim.z2)
  (hP16Win : p20InvertibilityWindowProp claim.invDelta)
  (hP17 : p20SamplingProp claim.cset claim.samples)
  (hP18 : p20PolyProp claim.qVals ctx.ell ctx.totalDegree ctx.setSize)
  (hP19 : p20InterpProp claim.xs claim.ys claim.expectedCoeffs claim.evalPoint claim.expectedEval) :
  p20ForClaim ctx claim := by
  exact p20ArithmeticBundle_of_props_with_thm3CoreAssumption
    (hP6 := hP6)
    (hP12Rows := hP12Rows)
    (hThm3 := hThm3)
    (hP14 := hP14)
    (hP15Vec := hP15Vec)
    (hP15Scal := hP15Scal)
    (hP16 := invertibilityPreconditions_from_constants)
    (hP16Win := hP16Win)
    (hP17 := hP17)
    (hP18 := hP18)
    (hP19 := hP19)

theorem p20ForClaim_of_thm3CoreAssumption_with_evalHom_assumption
  {ctx : ProtocolCtx} {claim : CEClaim}
  (hShape : ClaimShapeValid claim)
  (hP14Rows : MatrixRowsCompatible claim.m claim.z1)
  (hThm3 : thm3CoreAssumption ctx.bar)
  (hP6 : p20DecompProp claim.zDecomp ctx.bSplit ctx.kSplit)
  (hP14Assm : p14EvalHomAssumption ctx.bar claim.m claim.r claim.rho1 claim.rho2)
  (hP15Vec : p20VecModuleProp ctx.hVec claim.rho1 claim.z1 claim.z2)
  (hP15Scal : p20ScalarModuleProp ctx.hScal claim.rho1 claim.z1 claim.z2)
  (hP16Win : p20InvertibilityWindowProp claim.invDelta)
  (hP17 : p20SamplingProp claim.cset claim.samples)
  (hP18 : p20PolyProp claim.qVals ctx.ell ctx.totalDegree ctx.setSize)
  (hP19 : p20InterpProp claim.xs claim.ys claim.expectedCoeffs claim.evalPoint claim.expectedEval) :
  p20ForClaim ctx claim := by
  exact p20ArithmeticBundle_of_props_with_thm3CoreAssumption_with_evalHom_assumption
    (hP6 := hP6)
    (hP12Rows := claimShapeValid_matrixRowsCompatible hShape)
    (hThm3 := hThm3)
    (hP14Assm := hP14Assm)
    (hP14Size := claimShapeValid_z1_size_eq_z2_size hShape)
    (hP14Rows := hP14Rows)
    (hP15Vec := hP15Vec)
    (hP15Scal := hP15Scal)
    (hP16 := invertibilityPreconditions_from_constants)
    (hP16Win := hP16Win)
    (hP17 := hP17)
    (hP18 := hP18)
    (hP19 := hP19)

theorem p20ForClaim_of_thm3CoreAssumption_with_evalHom_checkAssumption
  {ctx : ProtocolCtx} {claim : CEClaim}
  (hShape : ClaimShapeValid claim)
  (hP14Rows : MatrixRowsCompatible claim.m claim.z1)
  (hThm3 : thm3CoreAssumption ctx.bar)
  (hP6 : p20DecompProp claim.zDecomp ctx.bSplit ctx.kSplit)
  (hP14Check : p14EvalHomCheckAssumption ctx.bar claim.m claim.r claim.rho1 claim.rho2)
  (hP15Vec : p20VecModuleProp ctx.hVec claim.rho1 claim.z1 claim.z2)
  (hP15Scal : p20ScalarModuleProp ctx.hScal claim.rho1 claim.z1 claim.z2)
  (hP16Win : p20InvertibilityWindowProp claim.invDelta)
  (hP17 : p20SamplingProp claim.cset claim.samples)
  (hP18 : p20PolyProp claim.qVals ctx.ell ctx.totalDegree ctx.setSize)
  (hP19 : p20InterpProp claim.xs claim.ys claim.expectedCoeffs claim.evalPoint claim.expectedEval) :
  p20ForClaim ctx claim := by
  exact p20ArithmeticBundle_of_props_with_thm3CoreAssumption_with_evalHom_checkAssumption
    (hP6 := hP6)
    (hP12Rows := claimShapeValid_matrixRowsCompatible hShape)
    (hThm3 := hThm3)
    (hP14Check := hP14Check)
    (hP14Size := claimShapeValid_z1_size_eq_z2_size hShape)
    (hP14Rows := hP14Rows)
    (hP15Vec := hP15Vec)
    (hP15Scal := hP15Scal)
    (hP16 := invertibilityPreconditions_from_constants)
    (hP16Win := hP16Win)
    (hP17 := hP17)
    (hP18 := hP18)
    (hP19 := hP19)

theorem p20ForClaim_of_props_with_evalHom_assumption
  {ctx : ProtocolCtx} {claim : CEClaim}
  (hP6 : p20DecompProp claim.zDecomp ctx.bSplit ctx.kSplit)
  (hP12Rows : MatrixRowsCompatible claim.m claim.z)
  (hP12Eq : matrixVecDirect claim.m claim.z = matrixVecCtBar ctx.bar claim.m claim.z)
  (hP14Assm : p14EvalHomAssumption ctx.bar claim.m claim.r claim.rho1 claim.rho2)
  (hP14Size : claim.z1.size = claim.z2.size)
  (hP14Rows : MatrixRowsCompatible claim.m claim.z1)
  (hP15Vec : p20VecModuleProp ctx.hVec claim.rho1 claim.z1 claim.z2)
  (hP15Scal : p20ScalarModuleProp ctx.hScal claim.rho1 claim.z1 claim.z2)
  (hP16Win : p20InvertibilityWindowProp claim.invDelta)
  (hP17 : p20SamplingProp claim.cset claim.samples)
  (hP18 : p20PolyProp claim.qVals ctx.ell ctx.totalDegree ctx.setSize)
  (hP19 : p20InterpProp claim.xs claim.ys claim.expectedCoeffs claim.evalPoint claim.expectedEval) :
  p20ForClaim ctx claim := by
  exact p20ArithmeticBundle_of_props_with_evalHom_assumption
    (hP6 := hP6)
    (hP12Rows := hP12Rows)
    (hP12Eq := hP12Eq)
    (hP14Assm := hP14Assm)
    (hP14Size := hP14Size)
    (hP14Rows := hP14Rows)
    (hP15Vec := hP15Vec)
    (hP15Scal := hP15Scal)
    (hP16 := invertibilityPreconditions_from_constants)
    (hP16Win := hP16Win)
    (hP17 := hP17)
    (hP18 := hP18)
    (hP19 := hP19)

theorem p20ForClaim_of_props_with_evalHom_checkAssumption
  {ctx : ProtocolCtx} {claim : CEClaim}
  (hP6 : p20DecompProp claim.zDecomp ctx.bSplit ctx.kSplit)
  (hP12Rows : MatrixRowsCompatible claim.m claim.z)
  (hP12Eq : matrixVecDirect claim.m claim.z = matrixVecCtBar ctx.bar claim.m claim.z)
  (hP14Check : p14EvalHomCheckAssumption ctx.bar claim.m claim.r claim.rho1 claim.rho2)
  (hP14Size : claim.z1.size = claim.z2.size)
  (hP14Rows : MatrixRowsCompatible claim.m claim.z1)
  (hP15Vec : p20VecModuleProp ctx.hVec claim.rho1 claim.z1 claim.z2)
  (hP15Scal : p20ScalarModuleProp ctx.hScal claim.rho1 claim.z1 claim.z2)
  (hP16Win : p20InvertibilityWindowProp claim.invDelta)
  (hP17 : p20SamplingProp claim.cset claim.samples)
  (hP18 : p20PolyProp claim.qVals ctx.ell ctx.totalDegree ctx.setSize)
  (hP19 : p20InterpProp claim.xs claim.ys claim.expectedCoeffs claim.evalPoint claim.expectedEval) :
  p20ForClaim ctx claim := by
  exact p20ArithmeticBundle_of_props
    (hP6 := hP6)
    (hP12Rows := hP12Rows)
    (hP12Eq := hP12Eq)
    (hP14 := p20EvalHomProp_of_checkAssumption hP14Check hP14Size hP14Rows)
    (hP15Vec := hP15Vec)
    (hP15Scal := hP15Scal)
    (hP16 := invertibilityPreconditions_from_constants)
    (hP16Win := hP16Win)
    (hP17 := hP17)
    (hP18 := hP18)
    (hP19 := hP19)

theorem p20ForClaim_of_props_with_matrixTransform_assumption_with_evalHom_assumption
  {ctx : ProtocolCtx} {claim : CEClaim}
  (hP6 : p20DecompProp claim.zDecomp ctx.bSplit ctx.kSplit)
  (hP12Rows : MatrixRowsCompatible claim.m claim.z)
  (hP12Assm : p12MatrixTransformAssumption ctx.bar claim.m)
  (hP14Assm : p14EvalHomAssumption ctx.bar claim.m claim.r claim.rho1 claim.rho2)
  (hP14Size : claim.z1.size = claim.z2.size)
  (hP14Rows : MatrixRowsCompatible claim.m claim.z1)
  (hP15Vec : p20VecModuleProp ctx.hVec claim.rho1 claim.z1 claim.z2)
  (hP15Scal : p20ScalarModuleProp ctx.hScal claim.rho1 claim.z1 claim.z2)
  (hP16Win : p20InvertibilityWindowProp claim.invDelta)
  (hP17 : p20SamplingProp claim.cset claim.samples)
  (hP18 : p20PolyProp claim.qVals ctx.ell ctx.totalDegree ctx.setSize)
  (hP19 : p20InterpProp claim.xs claim.ys claim.expectedCoeffs claim.evalPoint claim.expectedEval) :
  p20ForClaim ctx claim := by
  exact p20ArithmeticBundle_of_props_with_matrixTransform_assumption_with_evalHom_assumption
    (hP6 := hP6)
    (hP12Rows := hP12Rows)
    (hP12Assm := hP12Assm)
    (hP14Assm := hP14Assm)
    (hP14Size := hP14Size)
    (hP14Rows := hP14Rows)
    (hP15Vec := hP15Vec)
    (hP15Scal := hP15Scal)
    (hP16 := invertibilityPreconditions_from_constants)
    (hP16Win := hP16Win)
    (hP17 := hP17)
    (hP18 := hP18)
    (hP19 := hP19)

theorem p20ForClaim_of_props_with_matrixTransform_checkAssumption_with_evalHom_checkAssumption
  {ctx : ProtocolCtx} {claim : CEClaim}
  (hP6 : p20DecompProp claim.zDecomp ctx.bSplit ctx.kSplit)
  (hP12Rows : MatrixRowsCompatible claim.m claim.z)
  (hP12Check : p12MatrixTransformCheckAssumption ctx.bar claim.m)
  (hP14Check : p14EvalHomCheckAssumption ctx.bar claim.m claim.r claim.rho1 claim.rho2)
  (hP14Size : claim.z1.size = claim.z2.size)
  (hP14Rows : MatrixRowsCompatible claim.m claim.z1)
  (hP15Vec : p20VecModuleProp ctx.hVec claim.rho1 claim.z1 claim.z2)
  (hP15Scal : p20ScalarModuleProp ctx.hScal claim.rho1 claim.z1 claim.z2)
  (hP16Win : p20InvertibilityWindowProp claim.invDelta)
  (hP17 : p20SamplingProp claim.cset claim.samples)
  (hP18 : p20PolyProp claim.qVals ctx.ell ctx.totalDegree ctx.setSize)
  (hP19 : p20InterpProp claim.xs claim.ys claim.expectedCoeffs claim.evalPoint claim.expectedEval) :
  p20ForClaim ctx claim := by
  exact p20ArithmeticBundle_of_props_with_matrixTransform_checkAssumption_with_evalHom_checkAssumption
    (hP6 := hP6)
    (hP12Rows := hP12Rows)
    (hP12Check := hP12Check)
    (hP14Check := hP14Check)
    (hP14Size := hP14Size)
    (hP14Rows := hP14Rows)
    (hP15Vec := hP15Vec)
    (hP15Scal := hP15Scal)
    (hP16 := invertibilityPreconditions_from_constants)
    (hP16Win := hP16Win)
    (hP17 := hP17)
    (hP18 := hP18)
    (hP19 := hP19)

theorem p20ForClaim_of_props_with_matrixTransform_assumption_with_evalHom_checkAssumption
  {ctx : ProtocolCtx} {claim : CEClaim}
  (hP6 : p20DecompProp claim.zDecomp ctx.bSplit ctx.kSplit)
  (hP12Rows : MatrixRowsCompatible claim.m claim.z)
  (hP12Assm : p12MatrixTransformAssumption ctx.bar claim.m)
  (hP14Check : p14EvalHomCheckAssumption ctx.bar claim.m claim.r claim.rho1 claim.rho2)
  (hP14Size : claim.z1.size = claim.z2.size)
  (hP14Rows : MatrixRowsCompatible claim.m claim.z1)
  (hP15Vec : p20VecModuleProp ctx.hVec claim.rho1 claim.z1 claim.z2)
  (hP15Scal : p20ScalarModuleProp ctx.hScal claim.rho1 claim.z1 claim.z2)
  (hP16Win : p20InvertibilityWindowProp claim.invDelta)
  (hP17 : p20SamplingProp claim.cset claim.samples)
  (hP18 : p20PolyProp claim.qVals ctx.ell ctx.totalDegree ctx.setSize)
  (hP19 : p20InterpProp claim.xs claim.ys claim.expectedCoeffs claim.evalPoint claim.expectedEval) :
  p20ForClaim ctx claim := by
  exact p20ForClaim_of_props_with_matrixTransform_assumption_with_evalHom_assumption
    (hP6 := hP6)
    (hP12Rows := hP12Rows)
    (hP12Assm := hP12Assm)
    (hP14Assm := p14EvalHomAssumption_of_checkAssumption hP14Check)
    (hP14Size := hP14Size)
    (hP14Rows := hP14Rows)
    (hP15Vec := hP15Vec)
    (hP15Scal := hP15Scal)
    (hP16Win := hP16Win)
    (hP17 := hP17)
    (hP18 := hP18)
    (hP19 := hP19)

theorem p20ForClaim_of_props_with_matrixTransform_checkAssumption_with_evalHom_assumption
  {ctx : ProtocolCtx} {claim : CEClaim}
  (hP6 : p20DecompProp claim.zDecomp ctx.bSplit ctx.kSplit)
  (hP12Rows : MatrixRowsCompatible claim.m claim.z)
  (hP12Check : p12MatrixTransformCheckAssumption ctx.bar claim.m)
  (hP14Assm : p14EvalHomAssumption ctx.bar claim.m claim.r claim.rho1 claim.rho2)
  (hP14Size : claim.z1.size = claim.z2.size)
  (hP14Rows : MatrixRowsCompatible claim.m claim.z1)
  (hP15Vec : p20VecModuleProp ctx.hVec claim.rho1 claim.z1 claim.z2)
  (hP15Scal : p20ScalarModuleProp ctx.hScal claim.rho1 claim.z1 claim.z2)
  (hP16Win : p20InvertibilityWindowProp claim.invDelta)
  (hP17 : p20SamplingProp claim.cset claim.samples)
  (hP18 : p20PolyProp claim.qVals ctx.ell ctx.totalDegree ctx.setSize)
  (hP19 : p20InterpProp claim.xs claim.ys claim.expectedCoeffs claim.evalPoint claim.expectedEval) :
  p20ForClaim ctx claim := by
  exact p20ForClaim_of_props_with_matrixTransform_assumption_with_evalHom_assumption
    (hP6 := hP6)
    (hP12Rows := hP12Rows)
    (hP12Assm := p12MatrixTransformAssumption_of_checkAssumption hP12Check)
    (hP14Assm := hP14Assm)
    (hP14Size := hP14Size)
    (hP14Rows := hP14Rows)
    (hP15Vec := hP15Vec)
    (hP15Scal := hP15Scal)
    (hP16Win := hP16Win)
    (hP17 := hP17)
    (hP18 := hP18)
    (hP19 := hP19)

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
      claim.cset claim.samples claim.invDelta claim.qVals
      claim.xs claim.ys claim.expectedCoeffs
      claim.evalPoint claim.expectedEval
      ctx.ell ctx.totalDegree ctx.setSize := by
    exact p21ProtocolTarget_of_p20 hP20
  exact p21ProtocolTarget_to_CEValid hShape hBar hA hB hP21 hP10 hWitness hNorm

theorem superneoMathProtocolSkeleton_of_props_with_evalHom_assumption
  {ctx : ProtocolCtx} {claim : CEClaim} {witness : CEWitness}
  (hShape : ClaimShapeValid claim)
  (hP14Rows : MatrixRowsCompatible claim.m claim.z1)
  (hBar : IsDBarMatrix ctx.bar)
  (hA : IsDVec claim.a)
  (hB : IsDVec claim.b)
  (hP10 : p10ForClaim ctx claim)
  (hP6 : p20DecompProp claim.zDecomp ctx.bSplit ctx.kSplit)
  (hP12Eq : matrixVecDirect claim.m claim.z = matrixVecCtBar ctx.bar claim.m claim.z)
  (hP14Assm : p14EvalHomAssumption ctx.bar claim.m claim.r claim.rho1 claim.rho2)
  (hP15Vec : p20VecModuleProp ctx.hVec claim.rho1 claim.z1 claim.z2)
  (hP15Scal : p20ScalarModuleProp ctx.hScal claim.rho1 claim.z1 claim.z2)
  (hP16Win : p20InvertibilityWindowProp claim.invDelta)
  (hP17 : p20SamplingProp claim.cset claim.samples)
  (hP18 : p20PolyProp claim.qVals ctx.ell ctx.totalDegree ctx.setSize)
  (hP19 : p20InterpProp claim.xs claim.ys claim.expectedCoeffs claim.evalPoint claim.expectedEval)
  (hWitness : witness.z = claim.z)
  (hNorm : normInfCoeffs witness.z < ctx.ceNormBound) :
  CEValid ctx claim witness := by
  have hP20 : p20ForClaim ctx claim :=
    p20ForClaim_of_props_with_evalHom_assumption
      (hP6 := hP6)
      (hP12Rows := claimShapeValid_matrixRowsCompatible hShape)
      (hP12Eq := hP12Eq)
      (hP14Assm := hP14Assm)
      (hP14Size := claimShapeValid_z1_size_eq_z2_size hShape)
      (hP14Rows := hP14Rows)
      (hP15Vec := hP15Vec)
      (hP15Scal := hP15Scal)
      (hP16Win := hP16Win)
      (hP17 := hP17)
      (hP18 := hP18)
      (hP19 := hP19)
  exact superneoMathProtocolSkeleton_of_props hShape hBar hA hB hP10 hP20 hWitness hNorm

theorem superneoMathProtocolSkeleton_of_props_with_evalHom_checkAssumption
  {ctx : ProtocolCtx} {claim : CEClaim} {witness : CEWitness}
  (hShape : ClaimShapeValid claim)
  (hP14Rows : MatrixRowsCompatible claim.m claim.z1)
  (hBar : IsDBarMatrix ctx.bar)
  (hA : IsDVec claim.a)
  (hB : IsDVec claim.b)
  (hP10 : p10ForClaim ctx claim)
  (hP6 : p20DecompProp claim.zDecomp ctx.bSplit ctx.kSplit)
  (hP12Eq : matrixVecDirect claim.m claim.z = matrixVecCtBar ctx.bar claim.m claim.z)
  (hP14Check : p14EvalHomCheckAssumption ctx.bar claim.m claim.r claim.rho1 claim.rho2)
  (hP15Vec : p20VecModuleProp ctx.hVec claim.rho1 claim.z1 claim.z2)
  (hP15Scal : p20ScalarModuleProp ctx.hScal claim.rho1 claim.z1 claim.z2)
  (hP16Win : p20InvertibilityWindowProp claim.invDelta)
  (hP17 : p20SamplingProp claim.cset claim.samples)
  (hP18 : p20PolyProp claim.qVals ctx.ell ctx.totalDegree ctx.setSize)
  (hP19 : p20InterpProp claim.xs claim.ys claim.expectedCoeffs claim.evalPoint claim.expectedEval)
  (hWitness : witness.z = claim.z)
  (hNorm : normInfCoeffs witness.z < ctx.ceNormBound) :
  CEValid ctx claim witness := by
  have hP20 : p20ForClaim ctx claim :=
    p20ForClaim_of_props_with_evalHom_checkAssumption
      (hP6 := hP6)
      (hP12Rows := claimShapeValid_matrixRowsCompatible hShape)
      (hP12Eq := hP12Eq)
      (hP14Check := hP14Check)
      (hP14Size := claimShapeValid_z1_size_eq_z2_size hShape)
      (hP14Rows := hP14Rows)
      (hP15Vec := hP15Vec)
      (hP15Scal := hP15Scal)
      (hP16Win := hP16Win)
      (hP17 := hP17)
      (hP18 := hP18)
      (hP19 := hP19)
  exact superneoMathProtocolSkeleton_of_props hShape hBar hA hB hP10 hP20 hWitness hNorm

theorem superneoMathProtocolSkeleton_of_props_with_matrixTransform_assumption_with_evalHom_assumption
  {ctx : ProtocolCtx} {claim : CEClaim} {witness : CEWitness}
  (hShape : ClaimShapeValid claim)
  (hP12Assm : p12MatrixTransformAssumption ctx.bar claim.m)
  (hP14Rows : MatrixRowsCompatible claim.m claim.z1)
  (hBar : IsDBarMatrix ctx.bar)
  (hA : IsDVec claim.a)
  (hB : IsDVec claim.b)
  (hP10 : p10ForClaim ctx claim)
  (hP6 : p20DecompProp claim.zDecomp ctx.bSplit ctx.kSplit)
  (hP14Assm : p14EvalHomAssumption ctx.bar claim.m claim.r claim.rho1 claim.rho2)
  (hP15Vec : p20VecModuleProp ctx.hVec claim.rho1 claim.z1 claim.z2)
  (hP15Scal : p20ScalarModuleProp ctx.hScal claim.rho1 claim.z1 claim.z2)
  (hP16Win : p20InvertibilityWindowProp claim.invDelta)
  (hP17 : p20SamplingProp claim.cset claim.samples)
  (hP18 : p20PolyProp claim.qVals ctx.ell ctx.totalDegree ctx.setSize)
  (hP19 : p20InterpProp claim.xs claim.ys claim.expectedCoeffs claim.evalPoint claim.expectedEval)
  (hWitness : witness.z = claim.z)
  (hNorm : normInfCoeffs witness.z < ctx.ceNormBound) :
  CEValid ctx claim witness := by
  have hP20 : p20ForClaim ctx claim :=
    p20ForClaim_of_props_with_matrixTransform_assumption_with_evalHom_assumption
      (hP6 := hP6)
      (hP12Rows := claimShapeValid_matrixRowsCompatible hShape)
      (hP12Assm := hP12Assm)
      (hP14Assm := hP14Assm)
      (hP14Size := claimShapeValid_z1_size_eq_z2_size hShape)
      (hP14Rows := hP14Rows)
      (hP15Vec := hP15Vec)
      (hP15Scal := hP15Scal)
      (hP16Win := hP16Win)
      (hP17 := hP17)
      (hP18 := hP18)
      (hP19 := hP19)
  exact superneoMathProtocolSkeleton_of_props hShape hBar hA hB hP10 hP20 hWitness hNorm

theorem superneoMathProtocolSkeleton_of_props_with_matrixTransform_checkAssumption_with_evalHom_checkAssumption
  {ctx : ProtocolCtx} {claim : CEClaim} {witness : CEWitness}
  (hShape : ClaimShapeValid claim)
  (hP12Check : p12MatrixTransformCheckAssumption ctx.bar claim.m)
  (hP14Rows : MatrixRowsCompatible claim.m claim.z1)
  (hBar : IsDBarMatrix ctx.bar)
  (hA : IsDVec claim.a)
  (hB : IsDVec claim.b)
  (hP10 : p10ForClaim ctx claim)
  (hP6 : p20DecompProp claim.zDecomp ctx.bSplit ctx.kSplit)
  (hP14Check : p14EvalHomCheckAssumption ctx.bar claim.m claim.r claim.rho1 claim.rho2)
  (hP15Vec : p20VecModuleProp ctx.hVec claim.rho1 claim.z1 claim.z2)
  (hP15Scal : p20ScalarModuleProp ctx.hScal claim.rho1 claim.z1 claim.z2)
  (hP16Win : p20InvertibilityWindowProp claim.invDelta)
  (hP17 : p20SamplingProp claim.cset claim.samples)
  (hP18 : p20PolyProp claim.qVals ctx.ell ctx.totalDegree ctx.setSize)
  (hP19 : p20InterpProp claim.xs claim.ys claim.expectedCoeffs claim.evalPoint claim.expectedEval)
  (hWitness : witness.z = claim.z)
  (hNorm : normInfCoeffs witness.z < ctx.ceNormBound) :
  CEValid ctx claim witness := by
  have hP20 : p20ForClaim ctx claim :=
    p20ForClaim_of_props_with_matrixTransform_checkAssumption_with_evalHom_checkAssumption
      (hP6 := hP6)
      (hP12Rows := claimShapeValid_matrixRowsCompatible hShape)
      (hP12Check := hP12Check)
      (hP14Check := hP14Check)
      (hP14Size := claimShapeValid_z1_size_eq_z2_size hShape)
      (hP14Rows := hP14Rows)
      (hP15Vec := hP15Vec)
      (hP15Scal := hP15Scal)
      (hP16Win := hP16Win)
      (hP17 := hP17)
      (hP18 := hP18)
      (hP19 := hP19)
  exact superneoMathProtocolSkeleton_of_props hShape hBar hA hB hP10 hP20 hWitness hNorm

theorem superneoMathProtocolSkeleton_of_props_with_matrixTransform_assumption_with_evalHom_checkAssumption
  {ctx : ProtocolCtx} {claim : CEClaim} {witness : CEWitness}
  (hShape : ClaimShapeValid claim)
  (hP12Assm : p12MatrixTransformAssumption ctx.bar claim.m)
  (hP14Rows : MatrixRowsCompatible claim.m claim.z1)
  (hBar : IsDBarMatrix ctx.bar)
  (hA : IsDVec claim.a)
  (hB : IsDVec claim.b)
  (hP10 : p10ForClaim ctx claim)
  (hP6 : p20DecompProp claim.zDecomp ctx.bSplit ctx.kSplit)
  (hP14Check : p14EvalHomCheckAssumption ctx.bar claim.m claim.r claim.rho1 claim.rho2)
  (hP15Vec : p20VecModuleProp ctx.hVec claim.rho1 claim.z1 claim.z2)
  (hP15Scal : p20ScalarModuleProp ctx.hScal claim.rho1 claim.z1 claim.z2)
  (hP16Win : p20InvertibilityWindowProp claim.invDelta)
  (hP17 : p20SamplingProp claim.cset claim.samples)
  (hP18 : p20PolyProp claim.qVals ctx.ell ctx.totalDegree ctx.setSize)
  (hP19 : p20InterpProp claim.xs claim.ys claim.expectedCoeffs claim.evalPoint claim.expectedEval)
  (hWitness : witness.z = claim.z)
  (hNorm : normInfCoeffs witness.z < ctx.ceNormBound) :
  CEValid ctx claim witness := by
  exact superneoMathProtocolSkeleton_of_props_with_matrixTransform_assumption_with_evalHom_assumption
    hShape hP12Assm hP14Rows hBar hA hB hP10
    hP6 (p14EvalHomAssumption_of_checkAssumption hP14Check) hP15Vec hP15Scal hP16Win hP17 hP18 hP19
    hWitness hNorm

theorem superneoMathProtocolSkeleton_of_props_with_matrixTransform_checkAssumption_with_evalHom_assumption
  {ctx : ProtocolCtx} {claim : CEClaim} {witness : CEWitness}
  (hShape : ClaimShapeValid claim)
  (hP12Check : p12MatrixTransformCheckAssumption ctx.bar claim.m)
  (hP14Rows : MatrixRowsCompatible claim.m claim.z1)
  (hBar : IsDBarMatrix ctx.bar)
  (hA : IsDVec claim.a)
  (hB : IsDVec claim.b)
  (hP10 : p10ForClaim ctx claim)
  (hP6 : p20DecompProp claim.zDecomp ctx.bSplit ctx.kSplit)
  (hP14Assm : p14EvalHomAssumption ctx.bar claim.m claim.r claim.rho1 claim.rho2)
  (hP15Vec : p20VecModuleProp ctx.hVec claim.rho1 claim.z1 claim.z2)
  (hP15Scal : p20ScalarModuleProp ctx.hScal claim.rho1 claim.z1 claim.z2)
  (hP16Win : p20InvertibilityWindowProp claim.invDelta)
  (hP17 : p20SamplingProp claim.cset claim.samples)
  (hP18 : p20PolyProp claim.qVals ctx.ell ctx.totalDegree ctx.setSize)
  (hP19 : p20InterpProp claim.xs claim.ys claim.expectedCoeffs claim.evalPoint claim.expectedEval)
  (hWitness : witness.z = claim.z)
  (hNorm : normInfCoeffs witness.z < ctx.ceNormBound) :
  CEValid ctx claim witness := by
  exact superneoMathProtocolSkeleton_of_props_with_matrixTransform_assumption_with_evalHom_assumption
    hShape (p12MatrixTransformAssumption_of_checkAssumption hP12Check) hP14Rows hBar hA hB hP10
    hP6 hP14Assm hP15Vec hP15Scal hP16Win hP17 hP18 hP19
    hWitness hNorm

theorem superneoMathProtocolSkeleton_of_props_with_evalHom_assumption_with_invertibility
  {ctx : ProtocolCtx} {claim : CEClaim} {witness : CEWitness}
  (hShape : ClaimShapeValid claim)
  (hP14Rows : MatrixRowsCompatible claim.m claim.z1)
  (hBar : IsDBarMatrix ctx.bar)
  (hA : IsDVec claim.a)
  (hB : IsDVec claim.b)
  (hP10 : p10ForClaim ctx claim)
  (hP6 : p20DecompProp claim.zDecomp ctx.bSplit ctx.kSplit)
  (hP12Eq : matrixVecDirect claim.m claim.z = matrixVecCtBar ctx.bar claim.m claim.z)
  (hP14Assm : p14EvalHomAssumption ctx.bar claim.m claim.r claim.rho1 claim.rho2)
  (hP15Vec : p20VecModuleProp ctx.hVec claim.rho1 claim.z1 claim.z2)
  (hP15Scal : p20ScalarModuleProp ctx.hScal claim.rho1 claim.z1 claim.z2)
  (hP16Win : p20InvertibilityWindowProp claim.invDelta)
  (hP17 : p20SamplingProp claim.cset claim.samples)
  (hP18 : p20PolyProp claim.qVals ctx.ell ctx.totalDegree ctx.setSize)
  (hP19 : p20InterpProp claim.xs claim.ys claim.expectedCoeffs claim.evalPoint claim.expectedEval)
  (hWitness : witness.z = claim.z)
  (hNorm : normInfCoeffs witness.z < ctx.ceNormBound) :
  ∃ deltaInv : Coeffs, mulRq claim.invDelta deltaInv = oneRq ∧ CEValid ctx claim witness := by
  have hP20 : p20ForClaim ctx claim :=
    p20ForClaim_of_props_with_evalHom_assumption
      (hP6 := hP6)
      (hP12Rows := claimShapeValid_matrixRowsCompatible hShape)
      (hP12Eq := hP12Eq)
      (hP14Assm := hP14Assm)
      (hP14Size := claimShapeValid_z1_size_eq_z2_size hShape)
      (hP14Rows := hP14Rows)
      (hP15Vec := hP15Vec)
      (hP15Scal := hP15Scal)
      (hP16Win := hP16Win)
      (hP17 := hP17)
      (hP18 := hP18)
      (hP19 := hP19)
  have hP21 : p21ProtocolTarget
      ctx.bar
      claim.m
      claim.z claim.z1 claim.z2 claim.zDecomp claim.r
      claim.rho1 claim.rho2
      ctx.bSplit ctx.kSplit
      claim.cset claim.samples claim.invDelta claim.qVals
      claim.xs claim.ys claim.expectedCoeffs
      claim.evalPoint claim.expectedEval
      ctx.ell ctx.totalDegree ctx.setSize := by
    exact p21ProtocolTarget_of_p20 hP20
  exact p21ProtocolTarget_to_CEValid_with_invertibility hShape hBar hA hB hP21 hP10 hWitness hNorm

theorem superneoMathProtocolSkeleton_of_props_with_evalHom_checkAssumption_with_invertibility
  {ctx : ProtocolCtx} {claim : CEClaim} {witness : CEWitness}
  (hShape : ClaimShapeValid claim)
  (hP14Rows : MatrixRowsCompatible claim.m claim.z1)
  (hBar : IsDBarMatrix ctx.bar)
  (hA : IsDVec claim.a)
  (hB : IsDVec claim.b)
  (hP10 : p10ForClaim ctx claim)
  (hP6 : p20DecompProp claim.zDecomp ctx.bSplit ctx.kSplit)
  (hP12Eq : matrixVecDirect claim.m claim.z = matrixVecCtBar ctx.bar claim.m claim.z)
  (hP14Check : p14EvalHomCheckAssumption ctx.bar claim.m claim.r claim.rho1 claim.rho2)
  (hP15Vec : p20VecModuleProp ctx.hVec claim.rho1 claim.z1 claim.z2)
  (hP15Scal : p20ScalarModuleProp ctx.hScal claim.rho1 claim.z1 claim.z2)
  (hP16Win : p20InvertibilityWindowProp claim.invDelta)
  (hP17 : p20SamplingProp claim.cset claim.samples)
  (hP18 : p20PolyProp claim.qVals ctx.ell ctx.totalDegree ctx.setSize)
  (hP19 : p20InterpProp claim.xs claim.ys claim.expectedCoeffs claim.evalPoint claim.expectedEval)
  (hWitness : witness.z = claim.z)
  (hNorm : normInfCoeffs witness.z < ctx.ceNormBound) :
  ∃ deltaInv : Coeffs, mulRq claim.invDelta deltaInv = oneRq ∧ CEValid ctx claim witness := by
  have hP20 : p20ForClaim ctx claim :=
    p20ForClaim_of_props_with_evalHom_checkAssumption
      (hP6 := hP6)
      (hP12Rows := claimShapeValid_matrixRowsCompatible hShape)
      (hP12Eq := hP12Eq)
      (hP14Check := hP14Check)
      (hP14Size := claimShapeValid_z1_size_eq_z2_size hShape)
      (hP14Rows := hP14Rows)
      (hP15Vec := hP15Vec)
      (hP15Scal := hP15Scal)
      (hP16Win := hP16Win)
      (hP17 := hP17)
      (hP18 := hP18)
      (hP19 := hP19)
  have hP21 : p21ProtocolTarget
      ctx.bar
      claim.m
      claim.z claim.z1 claim.z2 claim.zDecomp claim.r
      claim.rho1 claim.rho2
      ctx.bSplit ctx.kSplit
      claim.cset claim.samples claim.invDelta claim.qVals
      claim.xs claim.ys claim.expectedCoeffs
      claim.evalPoint claim.expectedEval
      ctx.ell ctx.totalDegree ctx.setSize := by
    exact p21ProtocolTarget_of_p20 hP20
  exact p21ProtocolTarget_to_CEValid_with_invertibility hShape hBar hA hB hP21 hP10 hWitness hNorm

theorem superneoMathProtocolSkeleton_of_props_with_matrixTransform_assumption_with_evalHom_assumption_with_invertibility
  {ctx : ProtocolCtx} {claim : CEClaim} {witness : CEWitness}
  (hShape : ClaimShapeValid claim)
  (hP12Assm : p12MatrixTransformAssumption ctx.bar claim.m)
  (hP14Rows : MatrixRowsCompatible claim.m claim.z1)
  (hBar : IsDBarMatrix ctx.bar)
  (hA : IsDVec claim.a)
  (hB : IsDVec claim.b)
  (hP10 : p10ForClaim ctx claim)
  (hP6 : p20DecompProp claim.zDecomp ctx.bSplit ctx.kSplit)
  (hP14Assm : p14EvalHomAssumption ctx.bar claim.m claim.r claim.rho1 claim.rho2)
  (hP15Vec : p20VecModuleProp ctx.hVec claim.rho1 claim.z1 claim.z2)
  (hP15Scal : p20ScalarModuleProp ctx.hScal claim.rho1 claim.z1 claim.z2)
  (hP16Win : p20InvertibilityWindowProp claim.invDelta)
  (hP17 : p20SamplingProp claim.cset claim.samples)
  (hP18 : p20PolyProp claim.qVals ctx.ell ctx.totalDegree ctx.setSize)
  (hP19 : p20InterpProp claim.xs claim.ys claim.expectedCoeffs claim.evalPoint claim.expectedEval)
  (hWitness : witness.z = claim.z)
  (hNorm : normInfCoeffs witness.z < ctx.ceNormBound) :
  ∃ deltaInv : Coeffs, mulRq claim.invDelta deltaInv = oneRq ∧ CEValid ctx claim witness := by
  have hP20 : p20ForClaim ctx claim :=
    p20ForClaim_of_props_with_matrixTransform_assumption_with_evalHom_assumption
      (hP6 := hP6)
      (hP12Rows := claimShapeValid_matrixRowsCompatible hShape)
      (hP12Assm := hP12Assm)
      (hP14Assm := hP14Assm)
      (hP14Size := claimShapeValid_z1_size_eq_z2_size hShape)
      (hP14Rows := hP14Rows)
      (hP15Vec := hP15Vec)
      (hP15Scal := hP15Scal)
      (hP16Win := hP16Win)
      (hP17 := hP17)
      (hP18 := hP18)
      (hP19 := hP19)
  have hP21 : p21ProtocolTarget
      ctx.bar
      claim.m
      claim.z claim.z1 claim.z2 claim.zDecomp claim.r
      claim.rho1 claim.rho2
      ctx.bSplit ctx.kSplit
      claim.cset claim.samples claim.invDelta claim.qVals
      claim.xs claim.ys claim.expectedCoeffs
      claim.evalPoint claim.expectedEval
      ctx.ell ctx.totalDegree ctx.setSize := by
    exact p21ProtocolTarget_of_p20 hP20
  exact p21ProtocolTarget_to_CEValid_with_invertibility hShape hBar hA hB hP21 hP10 hWitness hNorm

theorem superneoMathProtocolSkeleton_of_props_with_matrixTransform_checkAssumption_with_evalHom_checkAssumption_with_invertibility
  {ctx : ProtocolCtx} {claim : CEClaim} {witness : CEWitness}
  (hShape : ClaimShapeValid claim)
  (hP12Check : p12MatrixTransformCheckAssumption ctx.bar claim.m)
  (hP14Rows : MatrixRowsCompatible claim.m claim.z1)
  (hBar : IsDBarMatrix ctx.bar)
  (hA : IsDVec claim.a)
  (hB : IsDVec claim.b)
  (hP10 : p10ForClaim ctx claim)
  (hP6 : p20DecompProp claim.zDecomp ctx.bSplit ctx.kSplit)
  (hP14Check : p14EvalHomCheckAssumption ctx.bar claim.m claim.r claim.rho1 claim.rho2)
  (hP15Vec : p20VecModuleProp ctx.hVec claim.rho1 claim.z1 claim.z2)
  (hP15Scal : p20ScalarModuleProp ctx.hScal claim.rho1 claim.z1 claim.z2)
  (hP16Win : p20InvertibilityWindowProp claim.invDelta)
  (hP17 : p20SamplingProp claim.cset claim.samples)
  (hP18 : p20PolyProp claim.qVals ctx.ell ctx.totalDegree ctx.setSize)
  (hP19 : p20InterpProp claim.xs claim.ys claim.expectedCoeffs claim.evalPoint claim.expectedEval)
  (hWitness : witness.z = claim.z)
  (hNorm : normInfCoeffs witness.z < ctx.ceNormBound) :
  ∃ deltaInv : Coeffs, mulRq claim.invDelta deltaInv = oneRq ∧ CEValid ctx claim witness := by
  have hP20 : p20ForClaim ctx claim :=
    p20ForClaim_of_props_with_matrixTransform_checkAssumption_with_evalHom_checkAssumption
      (hP6 := hP6)
      (hP12Rows := claimShapeValid_matrixRowsCompatible hShape)
      (hP12Check := hP12Check)
      (hP14Check := hP14Check)
      (hP14Size := claimShapeValid_z1_size_eq_z2_size hShape)
      (hP14Rows := hP14Rows)
      (hP15Vec := hP15Vec)
      (hP15Scal := hP15Scal)
      (hP16Win := hP16Win)
      (hP17 := hP17)
      (hP18 := hP18)
      (hP19 := hP19)
  have hP21 : p21ProtocolTarget
      ctx.bar
      claim.m
      claim.z claim.z1 claim.z2 claim.zDecomp claim.r
      claim.rho1 claim.rho2
      ctx.bSplit ctx.kSplit
      claim.cset claim.samples claim.invDelta claim.qVals
      claim.xs claim.ys claim.expectedCoeffs
      claim.evalPoint claim.expectedEval
      ctx.ell ctx.totalDegree ctx.setSize := by
    exact p21ProtocolTarget_of_p20 hP20
  exact p21ProtocolTarget_to_CEValid_with_invertibility hShape hBar hA hB hP21 hP10 hWitness hNorm

theorem superneoMathProtocolSkeleton_of_props_with_matrixTransform_assumption_with_evalHom_checkAssumption_with_invertibility
  {ctx : ProtocolCtx} {claim : CEClaim} {witness : CEWitness}
  (hShape : ClaimShapeValid claim)
  (hP12Assm : p12MatrixTransformAssumption ctx.bar claim.m)
  (hP14Rows : MatrixRowsCompatible claim.m claim.z1)
  (hBar : IsDBarMatrix ctx.bar)
  (hA : IsDVec claim.a)
  (hB : IsDVec claim.b)
  (hP10 : p10ForClaim ctx claim)
  (hP6 : p20DecompProp claim.zDecomp ctx.bSplit ctx.kSplit)
  (hP14Check : p14EvalHomCheckAssumption ctx.bar claim.m claim.r claim.rho1 claim.rho2)
  (hP15Vec : p20VecModuleProp ctx.hVec claim.rho1 claim.z1 claim.z2)
  (hP15Scal : p20ScalarModuleProp ctx.hScal claim.rho1 claim.z1 claim.z2)
  (hP16Win : p20InvertibilityWindowProp claim.invDelta)
  (hP17 : p20SamplingProp claim.cset claim.samples)
  (hP18 : p20PolyProp claim.qVals ctx.ell ctx.totalDegree ctx.setSize)
  (hP19 : p20InterpProp claim.xs claim.ys claim.expectedCoeffs claim.evalPoint claim.expectedEval)
  (hWitness : witness.z = claim.z)
  (hNorm : normInfCoeffs witness.z < ctx.ceNormBound) :
  ∃ deltaInv : Coeffs, mulRq claim.invDelta deltaInv = oneRq ∧ CEValid ctx claim witness := by
  exact superneoMathProtocolSkeleton_of_props_with_matrixTransform_assumption_with_evalHom_assumption_with_invertibility
    hShape hP12Assm hP14Rows hBar hA hB hP10
    hP6 (p14EvalHomAssumption_of_checkAssumption hP14Check) hP15Vec hP15Scal hP16Win hP17 hP18 hP19
    hWitness hNorm

theorem superneoMathProtocolSkeleton_of_props_with_matrixTransform_checkAssumption_with_evalHom_assumption_with_invertibility
  {ctx : ProtocolCtx} {claim : CEClaim} {witness : CEWitness}
  (hShape : ClaimShapeValid claim)
  (hP12Check : p12MatrixTransformCheckAssumption ctx.bar claim.m)
  (hP14Rows : MatrixRowsCompatible claim.m claim.z1)
  (hBar : IsDBarMatrix ctx.bar)
  (hA : IsDVec claim.a)
  (hB : IsDVec claim.b)
  (hP10 : p10ForClaim ctx claim)
  (hP6 : p20DecompProp claim.zDecomp ctx.bSplit ctx.kSplit)
  (hP14Assm : p14EvalHomAssumption ctx.bar claim.m claim.r claim.rho1 claim.rho2)
  (hP15Vec : p20VecModuleProp ctx.hVec claim.rho1 claim.z1 claim.z2)
  (hP15Scal : p20ScalarModuleProp ctx.hScal claim.rho1 claim.z1 claim.z2)
  (hP16Win : p20InvertibilityWindowProp claim.invDelta)
  (hP17 : p20SamplingProp claim.cset claim.samples)
  (hP18 : p20PolyProp claim.qVals ctx.ell ctx.totalDegree ctx.setSize)
  (hP19 : p20InterpProp claim.xs claim.ys claim.expectedCoeffs claim.evalPoint claim.expectedEval)
  (hWitness : witness.z = claim.z)
  (hNorm : normInfCoeffs witness.z < ctx.ceNormBound) :
  ∃ deltaInv : Coeffs, mulRq claim.invDelta deltaInv = oneRq ∧ CEValid ctx claim witness := by
  exact superneoMathProtocolSkeleton_of_props_with_matrixTransform_assumption_with_evalHom_assumption_with_invertibility
    hShape (p12MatrixTransformAssumption_of_checkAssumption hP12Check) hP14Rows hBar hA hB hP10
    hP6 hP14Assm hP15Vec hP15Scal hP16Win hP17 hP18 hP19
    hWitness hNorm

theorem superneoMathProtocolSkeleton_of_props_with_invertibility
  {ctx : ProtocolCtx} {claim : CEClaim} {witness : CEWitness}
  (hShape : ClaimShapeValid claim)
  (hBar : IsDBarMatrix ctx.bar)
  (hA : IsDVec claim.a)
  (hB : IsDVec claim.b)
  (hP10 : p10ForClaim ctx claim)
  (hP20 : p20ForClaim ctx claim)
  (hWitness : witness.z = claim.z)
  (hNorm : normInfCoeffs witness.z < ctx.ceNormBound) :
  ∃ deltaInv : Coeffs, mulRq claim.invDelta deltaInv = oneRq ∧ CEValid ctx claim witness := by
  have hP21 : p21ProtocolTarget
      ctx.bar
      claim.m
      claim.z claim.z1 claim.z2 claim.zDecomp claim.r
      claim.rho1 claim.rho2
      ctx.bSplit ctx.kSplit
      claim.cset claim.samples claim.invDelta claim.qVals
      claim.xs claim.ys claim.expectedCoeffs
      claim.evalPoint claim.expectedEval
      ctx.ell ctx.totalDegree ctx.setSize := by
    exact p21ProtocolTarget_of_p20 hP20
  exact p21ProtocolTarget_to_CEValid_with_invertibility hShape hBar hA hB hP21 hP10 hWitness hNorm

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

theorem superneoMathProtocolSkeleton_of_thm3_assumption_with_evalHom_assumption
  {ctx : ProtocolCtx} {claim : CEClaim} {witness : CEWitness}
  (hShape : ClaimShapeValid claim)
  (hP14Rows : MatrixRowsCompatible claim.m claim.z1)
  (hBar : IsDBarMatrix ctx.bar)
  (hA : IsDVec claim.a)
  (hB : IsDVec claim.b)
  (hThm3 : thm3CoreAssumption ctx.bar)
  (hP6 : p20DecompProp claim.zDecomp ctx.bSplit ctx.kSplit)
  (hP12Eq : matrixVecDirect claim.m claim.z = matrixVecCtBar ctx.bar claim.m claim.z)
  (hP14Assm : p14EvalHomAssumption ctx.bar claim.m claim.r claim.rho1 claim.rho2)
  (hP15Vec : p20VecModuleProp ctx.hVec claim.rho1 claim.z1 claim.z2)
  (hP15Scal : p20ScalarModuleProp ctx.hScal claim.rho1 claim.z1 claim.z2)
  (hP16Win : p20InvertibilityWindowProp claim.invDelta)
  (hP17 : p20SamplingProp claim.cset claim.samples)
  (hP18 : p20PolyProp claim.qVals ctx.ell ctx.totalDegree ctx.setSize)
  (hP19 : p20InterpProp claim.xs claim.ys claim.expectedCoeffs claim.evalPoint claim.expectedEval)
  (hWitness : witness.z = claim.z)
  (hNorm : normInfCoeffs witness.z < ctx.ceNormBound) :
  CEValid ctx claim witness := by
  have hP10 : p10ForClaim ctx claim := p10Core_of_assumption hThm3 hA hB
  exact superneoMathProtocolSkeleton_of_props_with_evalHom_assumption
    hShape hP14Rows hBar hA hB hP10
    hP6 hP12Eq hP14Assm hP15Vec hP15Scal hP16Win hP17 hP18 hP19
    hWitness hNorm

theorem superneoMathProtocolSkeleton_of_thm3_assumption_with_evalHom_assumption_of_shape
  {ctx : ProtocolCtx} {claim : CEClaim} {witness : CEWitness}
  (hShape : ClaimShapeValid claim)
  (hP14Rows : MatrixRowsCompatible claim.m claim.z1)
  (hBar : IsDBarMatrix ctx.bar)
  (hA : IsDVec claim.a)
  (hB : IsDVec claim.b)
  (hThm3 : thm3CoreAssumption ctx.bar)
  (hP6 : p20DecompProp claim.zDecomp ctx.bSplit ctx.kSplit)
  (hP14Assm : p14EvalHomAssumption ctx.bar claim.m claim.r claim.rho1 claim.rho2)
  (hP15Vec : p20VecModuleProp ctx.hVec claim.rho1 claim.z1 claim.z2)
  (hP15Scal : p20ScalarModuleProp ctx.hScal claim.rho1 claim.z1 claim.z2)
  (hP16Win : p20InvertibilityWindowProp claim.invDelta)
  (hP17 : p20SamplingProp claim.cset claim.samples)
  (hP18 : p20PolyProp claim.qVals ctx.ell ctx.totalDegree ctx.setSize)
  (hP19 : p20InterpProp claim.xs claim.ys claim.expectedCoeffs claim.evalPoint claim.expectedEval)
  (hWitness : witness.z = claim.z)
  (hNorm : normInfCoeffs witness.z < ctx.ceNormBound) :
  CEValid ctx claim witness := by
  exact superneoMathProtocolSkeleton_of_thm3_assumption_with_evalHom_assumption
    (hShape := hShape)
    (hP14Rows := hP14Rows)
    (hBar := hBar)
    (hA := hA)
    (hB := hB)
    (hThm3 := hThm3)
    (hP6 := hP6)
    (hP12Eq := matrixTransformEq_of_thm3CoreAssumption hThm3
      (claimShapeValid_matrixRowsCompatible hShape))
    (hP14Assm := hP14Assm)
    (hP15Vec := hP15Vec)
    (hP15Scal := hP15Scal)
    (hP16Win := hP16Win)
    (hP17 := hP17)
    (hP18 := hP18)
    (hP19 := hP19)
    (hWitness := hWitness)
    (hNorm := hNorm)

theorem superneoMathProtocolSkeleton_of_thm3_assumption_with_evalHom_checkAssumption
  {ctx : ProtocolCtx} {claim : CEClaim} {witness : CEWitness}
  (hShape : ClaimShapeValid claim)
  (hP14Rows : MatrixRowsCompatible claim.m claim.z1)
  (hBar : IsDBarMatrix ctx.bar)
  (hA : IsDVec claim.a)
  (hB : IsDVec claim.b)
  (hThm3 : thm3CoreAssumption ctx.bar)
  (hP6 : p20DecompProp claim.zDecomp ctx.bSplit ctx.kSplit)
  (hP12Eq : matrixVecDirect claim.m claim.z = matrixVecCtBar ctx.bar claim.m claim.z)
  (hP14Check : p14EvalHomCheckAssumption ctx.bar claim.m claim.r claim.rho1 claim.rho2)
  (hP15Vec : p20VecModuleProp ctx.hVec claim.rho1 claim.z1 claim.z2)
  (hP15Scal : p20ScalarModuleProp ctx.hScal claim.rho1 claim.z1 claim.z2)
  (hP16Win : p20InvertibilityWindowProp claim.invDelta)
  (hP17 : p20SamplingProp claim.cset claim.samples)
  (hP18 : p20PolyProp claim.qVals ctx.ell ctx.totalDegree ctx.setSize)
  (hP19 : p20InterpProp claim.xs claim.ys claim.expectedCoeffs claim.evalPoint claim.expectedEval)
  (hWitness : witness.z = claim.z)
  (hNorm : normInfCoeffs witness.z < ctx.ceNormBound) :
  CEValid ctx claim witness := by
  have hP10 : p10ForClaim ctx claim := p10Core_of_assumption hThm3 hA hB
  exact superneoMathProtocolSkeleton_of_props_with_evalHom_checkAssumption
    hShape hP14Rows hBar hA hB hP10
    hP6 hP12Eq hP14Check hP15Vec hP15Scal hP16Win hP17 hP18 hP19
    hWitness hNorm

theorem superneoMathProtocolSkeleton_of_thm3_assumption_with_evalHom_checkAssumption_of_shape
  {ctx : ProtocolCtx} {claim : CEClaim} {witness : CEWitness}
  (hShape : ClaimShapeValid claim)
  (hP14Rows : MatrixRowsCompatible claim.m claim.z1)
  (hBar : IsDBarMatrix ctx.bar)
  (hA : IsDVec claim.a)
  (hB : IsDVec claim.b)
  (hThm3 : thm3CoreAssumption ctx.bar)
  (hP6 : p20DecompProp claim.zDecomp ctx.bSplit ctx.kSplit)
  (hP14Check : p14EvalHomCheckAssumption ctx.bar claim.m claim.r claim.rho1 claim.rho2)
  (hP15Vec : p20VecModuleProp ctx.hVec claim.rho1 claim.z1 claim.z2)
  (hP15Scal : p20ScalarModuleProp ctx.hScal claim.rho1 claim.z1 claim.z2)
  (hP16Win : p20InvertibilityWindowProp claim.invDelta)
  (hP17 : p20SamplingProp claim.cset claim.samples)
  (hP18 : p20PolyProp claim.qVals ctx.ell ctx.totalDegree ctx.setSize)
  (hP19 : p20InterpProp claim.xs claim.ys claim.expectedCoeffs claim.evalPoint claim.expectedEval)
  (hWitness : witness.z = claim.z)
  (hNorm : normInfCoeffs witness.z < ctx.ceNormBound) :
  CEValid ctx claim witness := by
  exact superneoMathProtocolSkeleton_of_thm3_assumption_with_evalHom_checkAssumption
    (hShape := hShape)
    (hP14Rows := hP14Rows)
    (hBar := hBar)
    (hA := hA)
    (hB := hB)
    (hThm3 := hThm3)
    (hP6 := hP6)
    (hP12Eq := matrixTransformEq_of_thm3CoreAssumption hThm3
      (claimShapeValid_matrixRowsCompatible hShape))
    (hP14Check := hP14Check)
    (hP15Vec := hP15Vec)
    (hP15Scal := hP15Scal)
    (hP16Win := hP16Win)
    (hP17 := hP17)
    (hP18 := hP18)
    (hP19 := hP19)
    (hWitness := hWitness)
    (hNorm := hNorm)

theorem superneoMathProtocolSkeleton_of_thm3_assumption_with_matrixTransform_assumption_with_evalHom_assumption
  {ctx : ProtocolCtx} {claim : CEClaim} {witness : CEWitness}
  (hShape : ClaimShapeValid claim)
  (hP12Assm : p12MatrixTransformAssumption ctx.bar claim.m)
  (hP14Rows : MatrixRowsCompatible claim.m claim.z1)
  (hBar : IsDBarMatrix ctx.bar)
  (hA : IsDVec claim.a)
  (hB : IsDVec claim.b)
  (hThm3 : thm3CoreAssumption ctx.bar)
  (hP6 : p20DecompProp claim.zDecomp ctx.bSplit ctx.kSplit)
  (hP14Assm : p14EvalHomAssumption ctx.bar claim.m claim.r claim.rho1 claim.rho2)
  (hP15Vec : p20VecModuleProp ctx.hVec claim.rho1 claim.z1 claim.z2)
  (hP15Scal : p20ScalarModuleProp ctx.hScal claim.rho1 claim.z1 claim.z2)
  (hP16Win : p20InvertibilityWindowProp claim.invDelta)
  (hP17 : p20SamplingProp claim.cset claim.samples)
  (hP18 : p20PolyProp claim.qVals ctx.ell ctx.totalDegree ctx.setSize)
  (hP19 : p20InterpProp claim.xs claim.ys claim.expectedCoeffs claim.evalPoint claim.expectedEval)
  (hWitness : witness.z = claim.z)
  (hNorm : normInfCoeffs witness.z < ctx.ceNormBound) :
  CEValid ctx claim witness := by
  have hP10 : p10ForClaim ctx claim := p10Core_of_assumption hThm3 hA hB
  exact superneoMathProtocolSkeleton_of_props_with_matrixTransform_assumption_with_evalHom_assumption
    hShape hP12Assm hP14Rows hBar hA hB hP10
    hP6 hP14Assm hP15Vec hP15Scal hP16Win hP17 hP18 hP19
    hWitness hNorm

theorem superneoMathProtocolSkeleton_of_thm3_assumption_with_matrixTransform_checkAssumption_with_evalHom_checkAssumption
  {ctx : ProtocolCtx} {claim : CEClaim} {witness : CEWitness}
  (hShape : ClaimShapeValid claim)
  (hP12Check : p12MatrixTransformCheckAssumption ctx.bar claim.m)
  (hP14Rows : MatrixRowsCompatible claim.m claim.z1)
  (hBar : IsDBarMatrix ctx.bar)
  (hA : IsDVec claim.a)
  (hB : IsDVec claim.b)
  (hThm3 : thm3CoreAssumption ctx.bar)
  (hP6 : p20DecompProp claim.zDecomp ctx.bSplit ctx.kSplit)
  (hP14Check : p14EvalHomCheckAssumption ctx.bar claim.m claim.r claim.rho1 claim.rho2)
  (hP15Vec : p20VecModuleProp ctx.hVec claim.rho1 claim.z1 claim.z2)
  (hP15Scal : p20ScalarModuleProp ctx.hScal claim.rho1 claim.z1 claim.z2)
  (hP16Win : p20InvertibilityWindowProp claim.invDelta)
  (hP17 : p20SamplingProp claim.cset claim.samples)
  (hP18 : p20PolyProp claim.qVals ctx.ell ctx.totalDegree ctx.setSize)
  (hP19 : p20InterpProp claim.xs claim.ys claim.expectedCoeffs claim.evalPoint claim.expectedEval)
  (hWitness : witness.z = claim.z)
  (hNorm : normInfCoeffs witness.z < ctx.ceNormBound) :
  CEValid ctx claim witness := by
  have hP10 : p10ForClaim ctx claim := p10Core_of_assumption hThm3 hA hB
  exact superneoMathProtocolSkeleton_of_props_with_matrixTransform_checkAssumption_with_evalHom_checkAssumption
    hShape hP12Check hP14Rows hBar hA hB hP10
    hP6 hP14Check hP15Vec hP15Scal hP16Win hP17 hP18 hP19
    hWitness hNorm

theorem superneoMathProtocolSkeleton_of_thm3_assumption_with_matrixTransform_assumption_with_evalHom_checkAssumption
  {ctx : ProtocolCtx} {claim : CEClaim} {witness : CEWitness}
  (hShape : ClaimShapeValid claim)
  (hP12Assm : p12MatrixTransformAssumption ctx.bar claim.m)
  (hP14Rows : MatrixRowsCompatible claim.m claim.z1)
  (hBar : IsDBarMatrix ctx.bar)
  (hA : IsDVec claim.a)
  (hB : IsDVec claim.b)
  (hThm3 : thm3CoreAssumption ctx.bar)
  (hP6 : p20DecompProp claim.zDecomp ctx.bSplit ctx.kSplit)
  (hP14Check : p14EvalHomCheckAssumption ctx.bar claim.m claim.r claim.rho1 claim.rho2)
  (hP15Vec : p20VecModuleProp ctx.hVec claim.rho1 claim.z1 claim.z2)
  (hP15Scal : p20ScalarModuleProp ctx.hScal claim.rho1 claim.z1 claim.z2)
  (hP16Win : p20InvertibilityWindowProp claim.invDelta)
  (hP17 : p20SamplingProp claim.cset claim.samples)
  (hP18 : p20PolyProp claim.qVals ctx.ell ctx.totalDegree ctx.setSize)
  (hP19 : p20InterpProp claim.xs claim.ys claim.expectedCoeffs claim.evalPoint claim.expectedEval)
  (hWitness : witness.z = claim.z)
  (hNorm : normInfCoeffs witness.z < ctx.ceNormBound) :
  CEValid ctx claim witness := by
  have hP10 : p10ForClaim ctx claim := p10Core_of_assumption hThm3 hA hB
  exact superneoMathProtocolSkeleton_of_props_with_matrixTransform_assumption_with_evalHom_checkAssumption
    hShape hP12Assm hP14Rows hBar hA hB hP10
    hP6 hP14Check hP15Vec hP15Scal hP16Win hP17 hP18 hP19
    hWitness hNorm

theorem superneoMathProtocolSkeleton_of_thm3_assumption_with_matrixTransform_checkAssumption_with_evalHom_assumption
  {ctx : ProtocolCtx} {claim : CEClaim} {witness : CEWitness}
  (hShape : ClaimShapeValid claim)
  (hP12Check : p12MatrixTransformCheckAssumption ctx.bar claim.m)
  (hP14Rows : MatrixRowsCompatible claim.m claim.z1)
  (hBar : IsDBarMatrix ctx.bar)
  (hA : IsDVec claim.a)
  (hB : IsDVec claim.b)
  (hThm3 : thm3CoreAssumption ctx.bar)
  (hP6 : p20DecompProp claim.zDecomp ctx.bSplit ctx.kSplit)
  (hP14Assm : p14EvalHomAssumption ctx.bar claim.m claim.r claim.rho1 claim.rho2)
  (hP15Vec : p20VecModuleProp ctx.hVec claim.rho1 claim.z1 claim.z2)
  (hP15Scal : p20ScalarModuleProp ctx.hScal claim.rho1 claim.z1 claim.z2)
  (hP16Win : p20InvertibilityWindowProp claim.invDelta)
  (hP17 : p20SamplingProp claim.cset claim.samples)
  (hP18 : p20PolyProp claim.qVals ctx.ell ctx.totalDegree ctx.setSize)
  (hP19 : p20InterpProp claim.xs claim.ys claim.expectedCoeffs claim.evalPoint claim.expectedEval)
  (hWitness : witness.z = claim.z)
  (hNorm : normInfCoeffs witness.z < ctx.ceNormBound) :
  CEValid ctx claim witness := by
  have hP10 : p10ForClaim ctx claim := p10Core_of_assumption hThm3 hA hB
  exact superneoMathProtocolSkeleton_of_props_with_matrixTransform_checkAssumption_with_evalHom_assumption
    hShape hP12Check hP14Rows hBar hA hB hP10
    hP6 hP14Assm hP15Vec hP15Scal hP16Win hP17 hP18 hP19
    hWitness hNorm

theorem superneoMathProtocolSkeleton_of_thm3_assumption_with_evalHom_assumption_with_invertibility
  {ctx : ProtocolCtx} {claim : CEClaim} {witness : CEWitness}
  (hShape : ClaimShapeValid claim)
  (hP14Rows : MatrixRowsCompatible claim.m claim.z1)
  (hBar : IsDBarMatrix ctx.bar)
  (hA : IsDVec claim.a)
  (hB : IsDVec claim.b)
  (hThm3 : thm3CoreAssumption ctx.bar)
  (hP6 : p20DecompProp claim.zDecomp ctx.bSplit ctx.kSplit)
  (hP12Eq : matrixVecDirect claim.m claim.z = matrixVecCtBar ctx.bar claim.m claim.z)
  (hP14Assm : p14EvalHomAssumption ctx.bar claim.m claim.r claim.rho1 claim.rho2)
  (hP15Vec : p20VecModuleProp ctx.hVec claim.rho1 claim.z1 claim.z2)
  (hP15Scal : p20ScalarModuleProp ctx.hScal claim.rho1 claim.z1 claim.z2)
  (hP16Win : p20InvertibilityWindowProp claim.invDelta)
  (hP17 : p20SamplingProp claim.cset claim.samples)
  (hP18 : p20PolyProp claim.qVals ctx.ell ctx.totalDegree ctx.setSize)
  (hP19 : p20InterpProp claim.xs claim.ys claim.expectedCoeffs claim.evalPoint claim.expectedEval)
  (hWitness : witness.z = claim.z)
  (hNorm : normInfCoeffs witness.z < ctx.ceNormBound) :
  ∃ deltaInv : Coeffs, mulRq claim.invDelta deltaInv = oneRq ∧ CEValid ctx claim witness := by
  have hP10 : p10ForClaim ctx claim := p10Core_of_assumption hThm3 hA hB
  exact superneoMathProtocolSkeleton_of_props_with_evalHom_assumption_with_invertibility
    hShape hP14Rows hBar hA hB hP10
    hP6 hP12Eq hP14Assm hP15Vec hP15Scal hP16Win hP17 hP18 hP19
    hWitness hNorm

theorem superneoMathProtocolSkeleton_of_thm3_assumption_with_evalHom_assumption_with_invertibility_of_shape
  {ctx : ProtocolCtx} {claim : CEClaim} {witness : CEWitness}
  (hShape : ClaimShapeValid claim)
  (hP14Rows : MatrixRowsCompatible claim.m claim.z1)
  (hBar : IsDBarMatrix ctx.bar)
  (hA : IsDVec claim.a)
  (hB : IsDVec claim.b)
  (hThm3 : thm3CoreAssumption ctx.bar)
  (hP6 : p20DecompProp claim.zDecomp ctx.bSplit ctx.kSplit)
  (hP14Assm : p14EvalHomAssumption ctx.bar claim.m claim.r claim.rho1 claim.rho2)
  (hP15Vec : p20VecModuleProp ctx.hVec claim.rho1 claim.z1 claim.z2)
  (hP15Scal : p20ScalarModuleProp ctx.hScal claim.rho1 claim.z1 claim.z2)
  (hP16Win : p20InvertibilityWindowProp claim.invDelta)
  (hP17 : p20SamplingProp claim.cset claim.samples)
  (hP18 : p20PolyProp claim.qVals ctx.ell ctx.totalDegree ctx.setSize)
  (hP19 : p20InterpProp claim.xs claim.ys claim.expectedCoeffs claim.evalPoint claim.expectedEval)
  (hWitness : witness.z = claim.z)
  (hNorm : normInfCoeffs witness.z < ctx.ceNormBound) :
  ∃ deltaInv : Coeffs, mulRq claim.invDelta deltaInv = oneRq ∧ CEValid ctx claim witness := by
  exact superneoMathProtocolSkeleton_of_thm3_assumption_with_evalHom_assumption_with_invertibility
    (hShape := hShape)
    (hP14Rows := hP14Rows)
    (hBar := hBar)
    (hA := hA)
    (hB := hB)
    (hThm3 := hThm3)
    (hP6 := hP6)
    (hP12Eq := matrixTransformEq_of_thm3CoreAssumption hThm3
      (claimShapeValid_matrixRowsCompatible hShape))
    (hP14Assm := hP14Assm)
    (hP15Vec := hP15Vec)
    (hP15Scal := hP15Scal)
    (hP16Win := hP16Win)
    (hP17 := hP17)
    (hP18 := hP18)
    (hP19 := hP19)
    (hWitness := hWitness)
    (hNorm := hNorm)

theorem superneoMathProtocolSkeleton_of_thm3_assumption_with_evalHom_checkAssumption_with_invertibility
  {ctx : ProtocolCtx} {claim : CEClaim} {witness : CEWitness}
  (hShape : ClaimShapeValid claim)
  (hP14Rows : MatrixRowsCompatible claim.m claim.z1)
  (hBar : IsDBarMatrix ctx.bar)
  (hA : IsDVec claim.a)
  (hB : IsDVec claim.b)
  (hThm3 : thm3CoreAssumption ctx.bar)
  (hP6 : p20DecompProp claim.zDecomp ctx.bSplit ctx.kSplit)
  (hP12Eq : matrixVecDirect claim.m claim.z = matrixVecCtBar ctx.bar claim.m claim.z)
  (hP14Check : p14EvalHomCheckAssumption ctx.bar claim.m claim.r claim.rho1 claim.rho2)
  (hP15Vec : p20VecModuleProp ctx.hVec claim.rho1 claim.z1 claim.z2)
  (hP15Scal : p20ScalarModuleProp ctx.hScal claim.rho1 claim.z1 claim.z2)
  (hP16Win : p20InvertibilityWindowProp claim.invDelta)
  (hP17 : p20SamplingProp claim.cset claim.samples)
  (hP18 : p20PolyProp claim.qVals ctx.ell ctx.totalDegree ctx.setSize)
  (hP19 : p20InterpProp claim.xs claim.ys claim.expectedCoeffs claim.evalPoint claim.expectedEval)
  (hWitness : witness.z = claim.z)
  (hNorm : normInfCoeffs witness.z < ctx.ceNormBound) :
  ∃ deltaInv : Coeffs, mulRq claim.invDelta deltaInv = oneRq ∧ CEValid ctx claim witness := by
  have hP10 : p10ForClaim ctx claim := p10Core_of_assumption hThm3 hA hB
  exact superneoMathProtocolSkeleton_of_props_with_evalHom_checkAssumption_with_invertibility
    hShape hP14Rows hBar hA hB hP10
    hP6 hP12Eq hP14Check hP15Vec hP15Scal hP16Win hP17 hP18 hP19
    hWitness hNorm

theorem superneoMathProtocolSkeleton_of_thm3_assumption_with_evalHom_checkAssumption_with_invertibility_of_shape
  {ctx : ProtocolCtx} {claim : CEClaim} {witness : CEWitness}
  (hShape : ClaimShapeValid claim)
  (hP14Rows : MatrixRowsCompatible claim.m claim.z1)
  (hBar : IsDBarMatrix ctx.bar)
  (hA : IsDVec claim.a)
  (hB : IsDVec claim.b)
  (hThm3 : thm3CoreAssumption ctx.bar)
  (hP6 : p20DecompProp claim.zDecomp ctx.bSplit ctx.kSplit)
  (hP14Check : p14EvalHomCheckAssumption ctx.bar claim.m claim.r claim.rho1 claim.rho2)
  (hP15Vec : p20VecModuleProp ctx.hVec claim.rho1 claim.z1 claim.z2)
  (hP15Scal : p20ScalarModuleProp ctx.hScal claim.rho1 claim.z1 claim.z2)
  (hP16Win : p20InvertibilityWindowProp claim.invDelta)
  (hP17 : p20SamplingProp claim.cset claim.samples)
  (hP18 : p20PolyProp claim.qVals ctx.ell ctx.totalDegree ctx.setSize)
  (hP19 : p20InterpProp claim.xs claim.ys claim.expectedCoeffs claim.evalPoint claim.expectedEval)
  (hWitness : witness.z = claim.z)
  (hNorm : normInfCoeffs witness.z < ctx.ceNormBound) :
  ∃ deltaInv : Coeffs, mulRq claim.invDelta deltaInv = oneRq ∧ CEValid ctx claim witness := by
  exact superneoMathProtocolSkeleton_of_thm3_assumption_with_evalHom_checkAssumption_with_invertibility
    (hShape := hShape)
    (hP14Rows := hP14Rows)
    (hBar := hBar)
    (hA := hA)
    (hB := hB)
    (hThm3 := hThm3)
    (hP6 := hP6)
    (hP12Eq := matrixTransformEq_of_thm3CoreAssumption hThm3
      (claimShapeValid_matrixRowsCompatible hShape))
    (hP14Check := hP14Check)
    (hP15Vec := hP15Vec)
    (hP15Scal := hP15Scal)
    (hP16Win := hP16Win)
    (hP17 := hP17)
    (hP18 := hP18)
    (hP19 := hP19)
    (hWitness := hWitness)
    (hNorm := hNorm)

theorem superneoMathProtocolSkeleton_of_thm3_assumption_with_matrixTransform_assumption_with_evalHom_assumption_with_invertibility
  {ctx : ProtocolCtx} {claim : CEClaim} {witness : CEWitness}
  (hShape : ClaimShapeValid claim)
  (hP12Assm : p12MatrixTransformAssumption ctx.bar claim.m)
  (hP14Rows : MatrixRowsCompatible claim.m claim.z1)
  (hBar : IsDBarMatrix ctx.bar)
  (hA : IsDVec claim.a)
  (hB : IsDVec claim.b)
  (hThm3 : thm3CoreAssumption ctx.bar)
  (hP6 : p20DecompProp claim.zDecomp ctx.bSplit ctx.kSplit)
  (hP14Assm : p14EvalHomAssumption ctx.bar claim.m claim.r claim.rho1 claim.rho2)
  (hP15Vec : p20VecModuleProp ctx.hVec claim.rho1 claim.z1 claim.z2)
  (hP15Scal : p20ScalarModuleProp ctx.hScal claim.rho1 claim.z1 claim.z2)
  (hP16Win : p20InvertibilityWindowProp claim.invDelta)
  (hP17 : p20SamplingProp claim.cset claim.samples)
  (hP18 : p20PolyProp claim.qVals ctx.ell ctx.totalDegree ctx.setSize)
  (hP19 : p20InterpProp claim.xs claim.ys claim.expectedCoeffs claim.evalPoint claim.expectedEval)
  (hWitness : witness.z = claim.z)
  (hNorm : normInfCoeffs witness.z < ctx.ceNormBound) :
  ∃ deltaInv : Coeffs, mulRq claim.invDelta deltaInv = oneRq ∧ CEValid ctx claim witness := by
  have hP10 : p10ForClaim ctx claim := p10Core_of_assumption hThm3 hA hB
  exact superneoMathProtocolSkeleton_of_props_with_matrixTransform_assumption_with_evalHom_assumption_with_invertibility
    hShape hP12Assm hP14Rows hBar hA hB hP10
    hP6 hP14Assm hP15Vec hP15Scal hP16Win hP17 hP18 hP19
    hWitness hNorm

theorem superneoMathProtocolSkeleton_of_thm3_assumption_with_matrixTransform_checkAssumption_with_evalHom_checkAssumption_with_invertibility
  {ctx : ProtocolCtx} {claim : CEClaim} {witness : CEWitness}
  (hShape : ClaimShapeValid claim)
  (hP12Check : p12MatrixTransformCheckAssumption ctx.bar claim.m)
  (hP14Rows : MatrixRowsCompatible claim.m claim.z1)
  (hBar : IsDBarMatrix ctx.bar)
  (hA : IsDVec claim.a)
  (hB : IsDVec claim.b)
  (hThm3 : thm3CoreAssumption ctx.bar)
  (hP6 : p20DecompProp claim.zDecomp ctx.bSplit ctx.kSplit)
  (hP14Check : p14EvalHomCheckAssumption ctx.bar claim.m claim.r claim.rho1 claim.rho2)
  (hP15Vec : p20VecModuleProp ctx.hVec claim.rho1 claim.z1 claim.z2)
  (hP15Scal : p20ScalarModuleProp ctx.hScal claim.rho1 claim.z1 claim.z2)
  (hP16Win : p20InvertibilityWindowProp claim.invDelta)
  (hP17 : p20SamplingProp claim.cset claim.samples)
  (hP18 : p20PolyProp claim.qVals ctx.ell ctx.totalDegree ctx.setSize)
  (hP19 : p20InterpProp claim.xs claim.ys claim.expectedCoeffs claim.evalPoint claim.expectedEval)
  (hWitness : witness.z = claim.z)
  (hNorm : normInfCoeffs witness.z < ctx.ceNormBound) :
  ∃ deltaInv : Coeffs, mulRq claim.invDelta deltaInv = oneRq ∧ CEValid ctx claim witness := by
  have hP10 : p10ForClaim ctx claim := p10Core_of_assumption hThm3 hA hB
  exact superneoMathProtocolSkeleton_of_props_with_matrixTransform_checkAssumption_with_evalHom_checkAssumption_with_invertibility
    hShape hP12Check hP14Rows hBar hA hB hP10
    hP6 hP14Check hP15Vec hP15Scal hP16Win hP17 hP18 hP19
    hWitness hNorm

theorem superneoMathProtocolSkeleton_of_thm3_assumption_with_matrixTransform_assumption_with_evalHom_checkAssumption_with_invertibility
  {ctx : ProtocolCtx} {claim : CEClaim} {witness : CEWitness}
  (hShape : ClaimShapeValid claim)
  (hP12Assm : p12MatrixTransformAssumption ctx.bar claim.m)
  (hP14Rows : MatrixRowsCompatible claim.m claim.z1)
  (hBar : IsDBarMatrix ctx.bar)
  (hA : IsDVec claim.a)
  (hB : IsDVec claim.b)
  (hThm3 : thm3CoreAssumption ctx.bar)
  (hP6 : p20DecompProp claim.zDecomp ctx.bSplit ctx.kSplit)
  (hP14Check : p14EvalHomCheckAssumption ctx.bar claim.m claim.r claim.rho1 claim.rho2)
  (hP15Vec : p20VecModuleProp ctx.hVec claim.rho1 claim.z1 claim.z2)
  (hP15Scal : p20ScalarModuleProp ctx.hScal claim.rho1 claim.z1 claim.z2)
  (hP16Win : p20InvertibilityWindowProp claim.invDelta)
  (hP17 : p20SamplingProp claim.cset claim.samples)
  (hP18 : p20PolyProp claim.qVals ctx.ell ctx.totalDegree ctx.setSize)
  (hP19 : p20InterpProp claim.xs claim.ys claim.expectedCoeffs claim.evalPoint claim.expectedEval)
  (hWitness : witness.z = claim.z)
  (hNorm : normInfCoeffs witness.z < ctx.ceNormBound) :
  ∃ deltaInv : Coeffs, mulRq claim.invDelta deltaInv = oneRq ∧ CEValid ctx claim witness := by
  have hP10 : p10ForClaim ctx claim := p10Core_of_assumption hThm3 hA hB
  exact superneoMathProtocolSkeleton_of_props_with_matrixTransform_assumption_with_evalHom_checkAssumption_with_invertibility
    hShape hP12Assm hP14Rows hBar hA hB hP10
    hP6 hP14Check hP15Vec hP15Scal hP16Win hP17 hP18 hP19
    hWitness hNorm

theorem superneoMathProtocolSkeleton_of_thm3_assumption_with_matrixTransform_checkAssumption_with_evalHom_assumption_with_invertibility
  {ctx : ProtocolCtx} {claim : CEClaim} {witness : CEWitness}
  (hShape : ClaimShapeValid claim)
  (hP12Check : p12MatrixTransformCheckAssumption ctx.bar claim.m)
  (hP14Rows : MatrixRowsCompatible claim.m claim.z1)
  (hBar : IsDBarMatrix ctx.bar)
  (hA : IsDVec claim.a)
  (hB : IsDVec claim.b)
  (hThm3 : thm3CoreAssumption ctx.bar)
  (hP6 : p20DecompProp claim.zDecomp ctx.bSplit ctx.kSplit)
  (hP14Assm : p14EvalHomAssumption ctx.bar claim.m claim.r claim.rho1 claim.rho2)
  (hP15Vec : p20VecModuleProp ctx.hVec claim.rho1 claim.z1 claim.z2)
  (hP15Scal : p20ScalarModuleProp ctx.hScal claim.rho1 claim.z1 claim.z2)
  (hP16Win : p20InvertibilityWindowProp claim.invDelta)
  (hP17 : p20SamplingProp claim.cset claim.samples)
  (hP18 : p20PolyProp claim.qVals ctx.ell ctx.totalDegree ctx.setSize)
  (hP19 : p20InterpProp claim.xs claim.ys claim.expectedCoeffs claim.evalPoint claim.expectedEval)
  (hWitness : witness.z = claim.z)
  (hNorm : normInfCoeffs witness.z < ctx.ceNormBound) :
  ∃ deltaInv : Coeffs, mulRq claim.invDelta deltaInv = oneRq ∧ CEValid ctx claim witness := by
  have hP10 : p10ForClaim ctx claim := p10Core_of_assumption hThm3 hA hB
  exact superneoMathProtocolSkeleton_of_props_with_matrixTransform_checkAssumption_with_evalHom_assumption_with_invertibility
    hShape hP12Check hP14Rows hBar hA hB hP10
    hP6 hP14Assm hP15Vec hP15Scal hP16Win hP17 hP18 hP19
    hWitness hNorm

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
  (hP16Win : p20InvertibilityWindowProp claim.invDelta)
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
      claim.cset claim.samples claim.invDelta claim.qVals
      claim.xs claim.ys claim.expectedCoeffs
      claim.evalPoint claim.expectedEval
      ctx.ell ctx.totalDegree ctx.setSize := by
    exact p21FullMathTarget_of_checks
      (hP10 := hP10) (hP6 := hP6) (hP12 := hP12) (hP14 := hP14)
      (hVecAdd := hVecAdd) (hVecScale := hVecScale)
      (hScalAdd := hScalAdd) (hScalScale := hScalScale)
      (hP16Win := hP16Win)
      (hP17 := hP17) (hP18Eq := hP18Eq) (hP18SZ := hP18SZ) (hP19 := hP19)
  exact p21FullMathTarget_to_CEValid hShape hBar hA hB hFull hWitness hNorm

theorem superneoMathProtocolSkeleton_of_checks_with_invertibility
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
  (hP16Win : p20InvertibilityWindowProp claim.invDelta)
  (hP17 : samplingSetBoundCheck claim.cset claim.samples = true)
  (hP18Eq : eqLiftAllBoolean claim.qVals ctx.ell = true)
  (hP18SZ : schwartzZippelBoundLeOne ctx.totalDegree ctx.setSize = true)
  (hP19 : interpolationCase claim.xs claim.ys claim.expectedCoeffs claim.evalPoint claim.expectedEval = true)
  (hWitness : witness.z = claim.z)
  (hNorm : normInfCoeffs witness.z < ctx.ceNormBound) :
  ∃ deltaInv : Coeffs, mulRq claim.invDelta deltaInv = oneRq ∧ CEValid ctx claim witness := by
  have hFull : p21FullMathTarget
      ctx.bar
      claim.a claim.b
      claim.m
      claim.z claim.z1 claim.z2 claim.zDecomp claim.r
      claim.rho1 claim.rho2
      ctx.bSplit ctx.kSplit
      claim.cset claim.samples claim.invDelta claim.qVals
      claim.xs claim.ys claim.expectedCoeffs
      claim.evalPoint claim.expectedEval
      ctx.ell ctx.totalDegree ctx.setSize := by
    exact p21FullMathTarget_of_checks
      (hP10 := hP10) (hP6 := hP6) (hP12 := hP12) (hP14 := hP14)
      (hVecAdd := hVecAdd) (hVecScale := hVecScale)
      (hScalAdd := hScalAdd) (hScalScale := hScalScale)
      (hP16Win := hP16Win)
      (hP17 := hP17) (hP18Eq := hP18Eq) (hP18SZ := hP18SZ) (hP19 := hP19)
  exact p21FullMathTarget_to_CEValid_with_invertibility hShape hBar hA hB hFull hWitness hNorm

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
  {ctx : ProtocolCtx} {claim : CEClaim}
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
  {ctx : ProtocolCtx} {claim : CEClaim}
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
  {ctx : ProtocolCtx} {claim : CEClaim} :
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
  {ctx : ProtocolCtx} {claim : CEClaim}
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
  {ctx : ProtocolCtx} {claim : CEClaim}
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
  {ctx : ProtocolCtx} {claim : CEClaim}
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
  {ctx : ProtocolCtx} {claim : CEClaim} :
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

theorem protocol_props_to_CEValid
  {ctx : ProtocolCtx} {claim : CEClaim} {witness : CEWitness}
  (hShape : ClaimShapeValid claim)
  (hBar : IsDBarMatrix ctx.bar)
  (hA : IsDVec claim.a)
  (hB : IsDVec claim.b)
  (hProps : p10ForClaim ctx claim ∧ p20ForClaim ctx claim)
  (hWitness : witness.z = claim.z)
  (hNorm : normInfCoeffs witness.z < ctx.ceNormBound) :
  CEValid ctx claim witness := by
  exact smoke_p21_compose hShape hBar hA hB hProps hWitness hNorm

theorem p20ForClaim_invertibilityWitness
  {ctx : ProtocolCtx} {claim : CEClaim}
  (hP20 : p20ForClaim ctx claim) :
  ∃ deltaInv : Coeffs, mulRq claim.invDelta deltaInv = oneRq := by
  exact p20InvertibilityWitness_of_assumption ctx.hLowNormInvertibility hP20

end SuperNeo
