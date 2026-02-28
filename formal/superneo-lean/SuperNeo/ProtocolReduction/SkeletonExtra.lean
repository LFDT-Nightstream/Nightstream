import SuperNeo.ProtocolReduction.SkeletonCore

/-! Protocol-skeleton constructors built from claim-level reductions (extended segment). -/

namespace SuperNeo

theorem superneoMathProtocolSkeleton_of_props_with_evalHom_assumption_with_invertibility
  {ctx : PSContext} {claim : PSClaim} {witness : PSWitness}
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
  ∃ deltaInv : Coeffs, mulRq claim.invDelta deltaInv = oneRq ∧ PSCEValid ctx claim witness := by
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
  exact ceValid_with_invertibility_of_p20
    (hShape := hShape)
    (hBar := hBar)
    (hA := hA)
    (hB := hB)
    (hP10 := hP10)
    (hP20 := hP20)
    (hWitness := hWitness)
    (hNorm := hNorm)

theorem superneoMathProtocolSkeleton_of_props_with_evalHom_checkAssumption_with_invertibility
  {ctx : PSContext} {claim : PSClaim} {witness : PSWitness}
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
  ∃ deltaInv : Coeffs, mulRq claim.invDelta deltaInv = oneRq ∧ PSCEValid ctx claim witness := by
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
  exact ceValid_with_invertibility_of_p20
    (hShape := hShape)
    (hBar := hBar)
    (hA := hA)
    (hB := hB)
    (hP10 := hP10)
    (hP20 := hP20)
    (hWitness := hWitness)
    (hNorm := hNorm)

theorem superneoMathProtocolSkeleton_of_props_with_matrixTransform_assumption_with_evalHom_assumption_with_invertibility
  {ctx : PSContext} {claim : PSClaim} {witness : PSWitness}
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
  ∃ deltaInv : Coeffs, mulRq claim.invDelta deltaInv = oneRq ∧ PSCEValid ctx claim witness := by
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
  exact ceValid_with_invertibility_of_p20
    (hShape := hShape)
    (hBar := hBar)
    (hA := hA)
    (hB := hB)
    (hP10 := hP10)
    (hP20 := hP20)
    (hWitness := hWitness)
    (hNorm := hNorm)

theorem superneoMathProtocolSkeleton_of_props_with_matrixTransform_checkAssumption_with_evalHom_checkAssumption_with_invertibility
  {ctx : PSContext} {claim : PSClaim} {witness : PSWitness}
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
  ∃ deltaInv : Coeffs, mulRq claim.invDelta deltaInv = oneRq ∧ PSCEValid ctx claim witness := by
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
  exact ceValid_with_invertibility_of_p20
    (hShape := hShape)
    (hBar := hBar)
    (hA := hA)
    (hB := hB)
    (hP10 := hP10)
    (hP20 := hP20)
    (hWitness := hWitness)
    (hNorm := hNorm)

theorem superneoMathProtocolSkeleton_of_props_with_matrixTransform_assumption_with_evalHom_checkAssumption_with_invertibility
  {ctx : PSContext} {claim : PSClaim} {witness : PSWitness}
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
  ∃ deltaInv : Coeffs, mulRq claim.invDelta deltaInv = oneRq ∧ PSCEValid ctx claim witness := by
  exact superneoMathProtocolSkeleton_of_props_with_matrixTransform_assumption_with_evalHom_assumption_with_invertibility
    hShape hP12Assm hP14Rows hBar hA hB hP10
    hP6 (p14EvalHomAssumption_of_checkAssumption hP14Check) hP15Vec hP15Scal hP16Win hP17 hP18 hP19
    hWitness hNorm

theorem superneoMathProtocolSkeleton_of_props_with_matrixTransform_checkAssumption_with_evalHom_assumption_with_invertibility
  {ctx : PSContext} {claim : PSClaim} {witness : PSWitness}
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
  ∃ deltaInv : Coeffs, mulRq claim.invDelta deltaInv = oneRq ∧ PSCEValid ctx claim witness := by
  exact superneoMathProtocolSkeleton_of_props_with_matrixTransform_assumption_with_evalHom_assumption_with_invertibility
    hShape (p12MatrixTransformAssumption_of_checkAssumption hP12Check) hP14Rows hBar hA hB hP10
    hP6 hP14Assm hP15Vec hP15Scal hP16Win hP17 hP18 hP19
    hWitness hNorm

theorem superneoMathProtocolSkeleton_of_props_with_invertibility
  {ctx : PSContext} {claim : PSClaim} {witness : PSWitness}
  (hShape : ClaimShapeValid claim)
  (hBar : IsDBarMatrix ctx.bar)
  (hA : IsDVec claim.a)
  (hB : IsDVec claim.b)
  (hP10 : p10ForClaim ctx claim)
  (hP20 : p20ForClaim ctx claim)
  (hWitness : witness.z = claim.z)
  (hNorm : normInfCoeffs witness.z < ctx.ceNormBound) :
  ∃ deltaInv : Coeffs, mulRq claim.invDelta deltaInv = oneRq ∧ PSCEValid ctx claim witness := by
  exact ceValid_with_invertibility_of_p20 hShape hBar hA hB hP10 hP20 hWitness hNorm

theorem superneoMathProtocolSkeleton_of_thm3_assumption
  {ctx : PSContext} {claim : PSClaim} {witness : PSWitness}
  (hShape : ClaimShapeValid claim)
  (hBar : IsDBarMatrix ctx.bar)
  (hA : IsDVec claim.a)
  (hB : IsDVec claim.b)
  (hThm3 : thm3CoreAssumption ctx.bar)
  (hP20 : p20ForClaim ctx claim)
  (hWitness : witness.z = claim.z)
  (hNorm : normInfCoeffs witness.z < ctx.ceNormBound) :
  PSCEValid ctx claim witness := by
  have hP10 : p10ForClaim ctx claim := p10Core_of_assumption hThm3 hA hB
  exact superneoMathProtocolSkeleton_of_props hShape hBar hA hB hP10 hP20 hWitness hNorm

theorem superneoMathProtocolSkeleton_of_thm3_assumption_with_evalHom_assumption
  {ctx : PSContext} {claim : PSClaim} {witness : PSWitness}
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
  PSCEValid ctx claim witness := by
  let _ := hP12Eq
  have hP10 : p10ForClaim ctx claim := p10Core_of_assumption hThm3 hA hB
  have hP20 : p20ForClaim ctx claim :=
    p20ForClaim_of_thm3CoreAssumption_with_evalHom_assumption
      (hShape := hShape)
      (hP14Rows := hP14Rows)
      (hThm3 := hThm3)
      (hP6 := hP6)
      (hP14Assm := hP14Assm)
      (hP15Vec := hP15Vec)
      (hP15Scal := hP15Scal)
      (hP16Win := hP16Win)
      (hP17 := hP17)
      (hP18 := hP18)
      (hP19 := hP19)
  exact superneoMathProtocolSkeleton_of_props hShape hBar hA hB hP10 hP20 hWitness hNorm

theorem superneoMathProtocolSkeleton_of_thm3_assumption_with_evalHom_assumption_of_shape
  {ctx : PSContext} {claim : PSClaim} {witness : PSWitness}
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
  PSCEValid ctx claim witness := by
  exact superneoMathProtocolSkeleton_of_thm3_assumption_with_evalHom_assumption
    (hShape := hShape)
    (hP14Rows := hP14Rows)
    (hBar := hBar)
    (hA := hA)
    (hB := hB)
    (hThm3 := hThm3)
    (hP6 := hP6)
    (hP12Eq := matrixTransformEq_of_thm3CoreAssumption_of_shape hThm3 hShape)
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
  {ctx : PSContext} {claim : PSClaim} {witness : PSWitness}
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
  PSCEValid ctx claim witness := by
  let _ := hP12Eq
  have hP10 : p10ForClaim ctx claim := p10Core_of_assumption hThm3 hA hB
  have hP20 : p20ForClaim ctx claim :=
    p20ForClaim_of_thm3CoreAssumption_with_evalHom_checkAssumption
      (hShape := hShape)
      (hP14Rows := hP14Rows)
      (hThm3 := hThm3)
      (hP6 := hP6)
      (hP14Check := hP14Check)
      (hP15Vec := hP15Vec)
      (hP15Scal := hP15Scal)
      (hP16Win := hP16Win)
      (hP17 := hP17)
      (hP18 := hP18)
      (hP19 := hP19)
  exact superneoMathProtocolSkeleton_of_props hShape hBar hA hB hP10 hP20 hWitness hNorm

theorem superneoMathProtocolSkeleton_of_thm3_assumption_with_evalHom_checkAssumption_of_shape
  {ctx : PSContext} {claim : PSClaim} {witness : PSWitness}
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
  PSCEValid ctx claim witness := by
  exact superneoMathProtocolSkeleton_of_thm3_assumption_with_evalHom_checkAssumption
    (hShape := hShape)
    (hP14Rows := hP14Rows)
    (hBar := hBar)
    (hA := hA)
    (hB := hB)
    (hThm3 := hThm3)
    (hP6 := hP6)
    (hP12Eq := matrixTransformEq_of_thm3CoreAssumption_of_shape hThm3 hShape)
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
  {ctx : PSContext} {claim : PSClaim} {witness : PSWitness}
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
  PSCEValid ctx claim witness := by
  have hP10 : p10ForClaim ctx claim := p10Core_of_assumption hThm3 hA hB
  exact superneoMathProtocolSkeleton_of_props_with_matrixTransform_assumption_with_evalHom_assumption
    hShape hP12Assm hP14Rows hBar hA hB hP10
    hP6 hP14Assm hP15Vec hP15Scal hP16Win hP17 hP18 hP19
    hWitness hNorm

theorem superneoMathProtocolSkeleton_of_thm3_assumption_with_matrixTransform_checkAssumption_with_evalHom_checkAssumption
  {ctx : PSContext} {claim : PSClaim} {witness : PSWitness}
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
  PSCEValid ctx claim witness := by
  have hP10 : p10ForClaim ctx claim := p10Core_of_assumption hThm3 hA hB
  exact superneoMathProtocolSkeleton_of_props_with_matrixTransform_checkAssumption_with_evalHom_checkAssumption
    hShape hP12Check hP14Rows hBar hA hB hP10
    hP6 hP14Check hP15Vec hP15Scal hP16Win hP17 hP18 hP19
    hWitness hNorm

theorem superneoMathProtocolSkeleton_of_thm3_assumption_with_matrixTransform_assumption_with_evalHom_checkAssumption
  {ctx : PSContext} {claim : PSClaim} {witness : PSWitness}
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
  PSCEValid ctx claim witness := by
  have hP10 : p10ForClaim ctx claim := p10Core_of_assumption hThm3 hA hB
  exact superneoMathProtocolSkeleton_of_props_with_matrixTransform_assumption_with_evalHom_checkAssumption
    hShape hP12Assm hP14Rows hBar hA hB hP10
    hP6 hP14Check hP15Vec hP15Scal hP16Win hP17 hP18 hP19
    hWitness hNorm

theorem superneoMathProtocolSkeleton_of_thm3_assumption_with_matrixTransform_checkAssumption_with_evalHom_assumption
  {ctx : PSContext} {claim : PSClaim} {witness : PSWitness}
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
  PSCEValid ctx claim witness := by
  have hP10 : p10ForClaim ctx claim := p10Core_of_assumption hThm3 hA hB
  exact superneoMathProtocolSkeleton_of_props_with_matrixTransform_checkAssumption_with_evalHom_assumption
    hShape hP12Check hP14Rows hBar hA hB hP10
    hP6 hP14Assm hP15Vec hP15Scal hP16Win hP17 hP18 hP19
    hWitness hNorm

theorem superneoMathProtocolSkeleton_of_thm3_assumption_with_evalHom_assumption_with_invertibility
  {ctx : PSContext} {claim : PSClaim} {witness : PSWitness}
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
  ∃ deltaInv : Coeffs, mulRq claim.invDelta deltaInv = oneRq ∧ PSCEValid ctx claim witness := by
  let _ := hP12Eq
  have hP10 : p10ForClaim ctx claim := p10Core_of_assumption hThm3 hA hB
  have hP20 : p20ForClaim ctx claim :=
    p20ForClaim_of_thm3CoreAssumption_with_evalHom_assumption
      (hShape := hShape)
      (hP14Rows := hP14Rows)
      (hThm3 := hThm3)
      (hP6 := hP6)
      (hP14Assm := hP14Assm)
      (hP15Vec := hP15Vec)
      (hP15Scal := hP15Scal)
      (hP16Win := hP16Win)
      (hP17 := hP17)
      (hP18 := hP18)
      (hP19 := hP19)
  exact superneoMathProtocolSkeleton_of_props_with_invertibility
    hShape hBar hA hB hP10 hP20 hWitness hNorm

theorem superneoMathProtocolSkeleton_of_thm3_assumption_with_evalHom_assumption_with_invertibility_of_shape
  {ctx : PSContext} {claim : PSClaim} {witness : PSWitness}
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
  ∃ deltaInv : Coeffs, mulRq claim.invDelta deltaInv = oneRq ∧ PSCEValid ctx claim witness := by
  exact superneoMathProtocolSkeleton_of_thm3_assumption_with_evalHom_assumption_with_invertibility
    (hShape := hShape)
    (hP14Rows := hP14Rows)
    (hBar := hBar)
    (hA := hA)
    (hB := hB)
    (hThm3 := hThm3)
    (hP6 := hP6)
    (hP12Eq := matrixTransformEq_of_thm3CoreAssumption_of_shape hThm3 hShape)
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
  {ctx : PSContext} {claim : PSClaim} {witness : PSWitness}
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
  ∃ deltaInv : Coeffs, mulRq claim.invDelta deltaInv = oneRq ∧ PSCEValid ctx claim witness := by
  let _ := hP12Eq
  have hP10 : p10ForClaim ctx claim := p10Core_of_assumption hThm3 hA hB
  have hP20 : p20ForClaim ctx claim :=
    p20ForClaim_of_thm3CoreAssumption_with_evalHom_checkAssumption
      (hShape := hShape)
      (hP14Rows := hP14Rows)
      (hThm3 := hThm3)
      (hP6 := hP6)
      (hP14Check := hP14Check)
      (hP15Vec := hP15Vec)
      (hP15Scal := hP15Scal)
      (hP16Win := hP16Win)
      (hP17 := hP17)
      (hP18 := hP18)
      (hP19 := hP19)
  exact superneoMathProtocolSkeleton_of_props_with_invertibility
    hShape hBar hA hB hP10 hP20 hWitness hNorm

theorem superneoMathProtocolSkeleton_of_thm3_assumption_with_evalHom_checkAssumption_with_invertibility_of_shape
  {ctx : PSContext} {claim : PSClaim} {witness : PSWitness}
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
  ∃ deltaInv : Coeffs, mulRq claim.invDelta deltaInv = oneRq ∧ PSCEValid ctx claim witness := by
  exact superneoMathProtocolSkeleton_of_thm3_assumption_with_evalHom_checkAssumption_with_invertibility
    (hShape := hShape)
    (hP14Rows := hP14Rows)
    (hBar := hBar)
    (hA := hA)
    (hB := hB)
    (hThm3 := hThm3)
    (hP6 := hP6)
    (hP12Eq := matrixTransformEq_of_thm3CoreAssumption_of_shape hThm3 hShape)
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
  {ctx : PSContext} {claim : PSClaim} {witness : PSWitness}
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
  ∃ deltaInv : Coeffs, mulRq claim.invDelta deltaInv = oneRq ∧ PSCEValid ctx claim witness := by
  have hP10 : p10ForClaim ctx claim := p10Core_of_assumption hThm3 hA hB
  exact superneoMathProtocolSkeleton_of_props_with_matrixTransform_assumption_with_evalHom_assumption_with_invertibility
    hShape hP12Assm hP14Rows hBar hA hB hP10
    hP6 hP14Assm hP15Vec hP15Scal hP16Win hP17 hP18 hP19
    hWitness hNorm

theorem superneoMathProtocolSkeleton_of_thm3_assumption_with_matrixTransform_checkAssumption_with_evalHom_checkAssumption_with_invertibility
  {ctx : PSContext} {claim : PSClaim} {witness : PSWitness}
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
  ∃ deltaInv : Coeffs, mulRq claim.invDelta deltaInv = oneRq ∧ PSCEValid ctx claim witness := by
  have hP10 : p10ForClaim ctx claim := p10Core_of_assumption hThm3 hA hB
  exact superneoMathProtocolSkeleton_of_props_with_matrixTransform_checkAssumption_with_evalHom_checkAssumption_with_invertibility
    hShape hP12Check hP14Rows hBar hA hB hP10
    hP6 hP14Check hP15Vec hP15Scal hP16Win hP17 hP18 hP19
    hWitness hNorm

theorem superneoMathProtocolSkeleton_of_thm3_assumption_with_matrixTransform_assumption_with_evalHom_checkAssumption_with_invertibility
  {ctx : PSContext} {claim : PSClaim} {witness : PSWitness}
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
  ∃ deltaInv : Coeffs, mulRq claim.invDelta deltaInv = oneRq ∧ PSCEValid ctx claim witness := by
  have hP10 : p10ForClaim ctx claim := p10Core_of_assumption hThm3 hA hB
  exact superneoMathProtocolSkeleton_of_props_with_matrixTransform_assumption_with_evalHom_checkAssumption_with_invertibility
    hShape hP12Assm hP14Rows hBar hA hB hP10
    hP6 hP14Check hP15Vec hP15Scal hP16Win hP17 hP18 hP19
    hWitness hNorm

theorem superneoMathProtocolSkeleton_of_thm3_assumption_with_matrixTransform_checkAssumption_with_evalHom_assumption_with_invertibility
  {ctx : PSContext} {claim : PSClaim} {witness : PSWitness}
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
  ∃ deltaInv : Coeffs, mulRq claim.invDelta deltaInv = oneRq ∧ PSCEValid ctx claim witness := by
  have hP10 : p10ForClaim ctx claim := p10Core_of_assumption hThm3 hA hB
  exact superneoMathProtocolSkeleton_of_props_with_matrixTransform_checkAssumption_with_evalHom_assumption_with_invertibility
    hShape hP12Check hP14Rows hBar hA hB hP10
    hP6 hP14Assm hP15Vec hP15Scal hP16Win hP17 hP18 hP19
    hWitness hNorm

theorem superneoMathProtocolSkeleton_of_checks
  {ctx : PSContext} {claim : PSClaim} {witness : PSWitness}
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
  PSCEValid ctx claim witness := by
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
  {ctx : PSContext} {claim : PSClaim} {witness : PSWitness}
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
  ∃ deltaInv : Coeffs, mulRq claim.invDelta deltaInv = oneRq ∧ PSCEValid ctx claim witness := by
  have hValid : PSCEValid ctx claim witness :=
    superneoMathProtocolSkeleton_of_checks
      (hShape := hShape)
      (hBar := hBar)
      (hA := hA)
      (hB := hB)
      (hP10 := hP10)
      (hP6 := hP6)
      (hP12 := hP12)
      (hP14 := hP14)
      (hVecAdd := hVecAdd)
      (hVecScale := hVecScale)
      (hScalAdd := hScalAdd)
      (hScalScale := hScalScale)
      (hP16Win := hP16Win)
      (hP17 := hP17)
      (hP18Eq := hP18Eq)
      (hP18SZ := hP18SZ)
      (hP19 := hP19)
      (hWitness := hWitness)
      (hNorm := hNorm)

  have hP20 : p20ForClaim ctx claim := by
    exact p20ArithmeticBundle_of_checks
      (hP6 := hP6) (hP12 := hP12) (hP14 := hP14)
      (hVecAdd := hVecAdd) (hVecScale := hVecScale)
      (hScalAdd := hScalAdd) (hScalScale := hScalScale)
      (hP16Win := hP16Win)
      (hP17 := hP17) (hP18Eq := hP18Eq) (hP18SZ := hP18SZ) (hP19 := hP19)

  rcases p20InvertibilityWitness_of_assumption ctx.hLowNormInvertibility hP20 with
    ⟨deltaInv, hMul⟩
  exact ⟨deltaInv, hMul, hValid⟩



end SuperNeo
