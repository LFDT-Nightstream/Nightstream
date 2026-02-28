import SuperNeo.ProtocolReduction.Core

/-! Protocol-skeleton constructors built from claim-level reductions (core segment). -/

namespace SuperNeo

theorem superneoMathProtocolSkeleton_of_props
  {ctx : PSContext} {claim : PSClaim} {witness : PSWitness}
  (hShape : ClaimShapeValid claim)
  (hBar : IsDBarMatrix ctx.bar)
  (hA : IsDVec claim.a)
  (hB : IsDVec claim.b)
  (hP10 : p10ForClaim ctx claim)
  (hP20 : p20ForClaim ctx claim)
  (hWitness : witness.z = claim.z)
  (hNorm : normInfCoeffs witness.z < ctx.ceNormBound) :
  PSCEValid ctx claim witness := by
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

theorem superneoMathProtocolSkeleton_of_props_with_sampling_goldilocks_operand_fieldOp_assumptions
  {ctx : PSContext} {claim : PSClaim} {witness : PSWitness}
  (hShape : ClaimShapeValid claim)
  (hBar : IsDBarMatrix ctx.bar)
  (hA : IsDVec claim.a)
  (hB : IsDVec claim.b)
  (hP10 : p10ForClaim ctx claim)
  (hP6 : p20DecompProp claim.zDecomp ctx.bSplit ctx.kSplit)
  (hP12Eq : matrixVecDirect claim.m claim.z = matrixVecCtBar ctx.bar claim.m claim.z)
  (hP14 : p20EvalHomProp ctx.bar claim.m claim.z1 claim.z2 claim.r claim.rho1 claim.rho2)
  (hP15Vec : p20VecModuleProp ctx.hVec claim.rho1 claim.z1 claim.z2)
  (hP15Scal : p20ScalarModuleProp ctx.hScal claim.rho1 claim.z1 claim.z2)
  (hP16Win : p20InvertibilityWindowProp claim.invDelta)
  (hCset : ∀ i : Fin claim.cset.size, normInfCoeffs claim.cset[i] ≤ Parameters.Goldilocks.B)
  (hSamples : ∀ j : Fin claim.samples.size, normInfCoeffs claim.samples[j] ≤ Parameters.Goldilocks.B)
  (hRaw : GoldilocksRawNormBoundAssumption)
  (hFieldOps : GoldilocksFieldOpCollapseAssumption)
  (hUpper : Parameters.Goldilocks.B ≤ theorem9UpperBound (maxRhoNorm claim.cset))
  (hP18 : p20PolyProp claim.qVals ctx.ell ctx.totalDegree ctx.setSize)
  (hP19 : p20InterpProp claim.xs claim.ys claim.expectedCoeffs claim.evalPoint claim.expectedEval)
  (hWitness : witness.z = claim.z)
  (hNorm : normInfCoeffs witness.z < ctx.ceNormBound) :
  PSCEValid ctx claim witness := by
  have hP20 : p20ForClaim ctx claim :=
    p20ForClaim_of_props_with_sampling_goldilocks_operand_fieldOp_assumptions
      (hP6 := hP6)
      (hP12Rows := claimShapeValid_matrixRowsCompatible hShape)
      (hP12Eq := hP12Eq)
      (hP14 := hP14)
      (hP15Vec := hP15Vec)
      (hP15Scal := hP15Scal)
      (hP16Win := hP16Win)
      (hCset := hCset)
      (hSamples := hSamples)
      (hRaw := hRaw)
      (hFieldOps := hFieldOps)
      (hUpper := hUpper)
      (hP18 := hP18)
      (hP19 := hP19)
  exact superneoMathProtocolSkeleton_of_props hShape hBar hA hB hP10 hP20 hWitness hNorm

theorem superneoMathProtocolSkeleton_of_props_with_sampling_goldilocks_operand_fieldOp_assumptions_inRange
  {ctx : PSContext} {claim : PSClaim} {witness : PSWitness}
  (hShape : ClaimShapeValid claim)
  (hBar : IsDBarMatrix ctx.bar)
  (hA : IsDVec claim.a)
  (hB : IsDVec claim.b)
  (hP10 : p10ForClaim ctx claim)
  (hP6 : p20DecompProp claim.zDecomp ctx.bSplit ctx.kSplit)
  (hP12Eq : matrixVecDirect claim.m claim.z = matrixVecCtBar ctx.bar claim.m claim.z)
  (hP14 : p20EvalHomProp ctx.bar claim.m claim.z1 claim.z2 claim.r claim.rho1 claim.rho2)
  (hP15Vec : p20VecModuleProp ctx.hVec claim.rho1 claim.z1 claim.z2)
  (hP15Scal : p20ScalarModuleProp ctx.hScal claim.rho1 claim.z1 claim.z2)
  (hP16Win : p20InvertibilityWindowProp claim.invDelta)
  (hCset : ∀ i : Fin claim.cset.size, normInfCoeffs claim.cset[i] ≤ Parameters.Goldilocks.B)
  (hSamples : ∀ j : Fin claim.samples.size, normInfCoeffs claim.samples[j] ≤ Parameters.Goldilocks.B)
  (hRawInRange : GoldilocksRawInRangeBoundAssumption)
  (hFieldOps : GoldilocksFieldOpCollapseAssumption)
  (hUpper : Parameters.Goldilocks.B ≤ theorem9UpperBound (maxRhoNorm claim.cset))
  (hP18 : p20PolyProp claim.qVals ctx.ell ctx.totalDegree ctx.setSize)
  (hP19 : p20InterpProp claim.xs claim.ys claim.expectedCoeffs claim.evalPoint claim.expectedEval)
  (hWitness : witness.z = claim.z)
  (hNorm : normInfCoeffs witness.z < ctx.ceNormBound) :
  PSCEValid ctx claim witness := by
  have hP20 : p20ForClaim ctx claim :=
    p20ForClaim_of_props_with_sampling_goldilocks_operand_fieldOp_assumptions_inRange
      (hP6 := hP6)
      (hP12Rows := claimShapeValid_matrixRowsCompatible hShape)
      (hP12Eq := hP12Eq)
      (hP14 := hP14)
      (hP15Vec := hP15Vec)
      (hP15Scal := hP15Scal)
      (hP16Win := hP16Win)
      (hCset := hCset)
      (hSamples := hSamples)
      (hRawInRange := hRawInRange)
      (hFieldOps := hFieldOps)
      (hUpper := hUpper)
      (hP18 := hP18)
      (hP19 := hP19)
  exact superneoMathProtocolSkeleton_of_props hShape hBar hA hB hP10 hP20 hWitness hNorm

theorem superneoMathProtocolSkeleton_of_props_with_sampling_goldilocks_operand_assumptions
  {ctx : PSContext} {claim : PSClaim} {witness : PSWitness}
  (hShape : ClaimShapeValid claim)
  (hBar : IsDBarMatrix ctx.bar)
  (hA : IsDVec claim.a)
  (hB : IsDVec claim.b)
  (hP10 : p10ForClaim ctx claim)
  (hP6 : p20DecompProp claim.zDecomp ctx.bSplit ctx.kSplit)
  (hP12Eq : matrixVecDirect claim.m claim.z = matrixVecCtBar ctx.bar claim.m claim.z)
  (hP14 : p20EvalHomProp ctx.bar claim.m claim.z1 claim.z2 claim.r claim.rho1 claim.rho2)
  (hP15Vec : p20VecModuleProp ctx.hVec claim.rho1 claim.z1 claim.z2)
  (hP15Scal : p20ScalarModuleProp ctx.hScal claim.rho1 claim.z1 claim.z2)
  (hP16Win : p20InvertibilityWindowProp claim.invDelta)
  (hCset : ∀ i : Fin claim.cset.size, normInfCoeffs claim.cset[i] ≤ Parameters.Goldilocks.B)
  (hSamples : ∀ j : Fin claim.samples.size, normInfCoeffs claim.samples[j] ≤ Parameters.Goldilocks.B)
  (hRaw : GoldilocksRawNormBoundAssumption)
  (hCollapse : GoldilocksRawCollapseAssumption)
  (hUpper : Parameters.Goldilocks.B ≤ theorem9UpperBound (maxRhoNorm claim.cset))
  (hP18 : p20PolyProp claim.qVals ctx.ell ctx.totalDegree ctx.setSize)
  (hP19 : p20InterpProp claim.xs claim.ys claim.expectedCoeffs claim.evalPoint claim.expectedEval)
  (hWitness : witness.z = claim.z)
  (hNorm : normInfCoeffs witness.z < ctx.ceNormBound) :
  PSCEValid ctx claim witness := by
  have hP20 : p20ForClaim ctx claim :=
    p20ForClaim_of_props_with_sampling_goldilocks_operand_assumptions
      (hP6 := hP6)
      (hP12Rows := claimShapeValid_matrixRowsCompatible hShape)
      (hP12Eq := hP12Eq)
      (hP14 := hP14)
      (hP15Vec := hP15Vec)
      (hP15Scal := hP15Scal)
      (hP16Win := hP16Win)
      (hCset := hCset)
      (hSamples := hSamples)
      (hRaw := hRaw)
      (hCollapse := hCollapse)
      (hUpper := hUpper)
      (hP18 := hP18)
      (hP19 := hP19)
  exact superneoMathProtocolSkeleton_of_props hShape hBar hA hB hP10 hP20 hWitness hNorm

theorem superneoMathProtocolSkeleton_of_props_with_sampling_goldilocks_operand_assumptions_inRange
  {ctx : PSContext} {claim : PSClaim} {witness : PSWitness}
  (hShape : ClaimShapeValid claim)
  (hBar : IsDBarMatrix ctx.bar)
  (hA : IsDVec claim.a)
  (hB : IsDVec claim.b)
  (hP10 : p10ForClaim ctx claim)
  (hP6 : p20DecompProp claim.zDecomp ctx.bSplit ctx.kSplit)
  (hP12Eq : matrixVecDirect claim.m claim.z = matrixVecCtBar ctx.bar claim.m claim.z)
  (hP14 : p20EvalHomProp ctx.bar claim.m claim.z1 claim.z2 claim.r claim.rho1 claim.rho2)
  (hP15Vec : p20VecModuleProp ctx.hVec claim.rho1 claim.z1 claim.z2)
  (hP15Scal : p20ScalarModuleProp ctx.hScal claim.rho1 claim.z1 claim.z2)
  (hP16Win : p20InvertibilityWindowProp claim.invDelta)
  (hCset : ∀ i : Fin claim.cset.size, normInfCoeffs claim.cset[i] ≤ Parameters.Goldilocks.B)
  (hSamples : ∀ j : Fin claim.samples.size, normInfCoeffs claim.samples[j] ≤ Parameters.Goldilocks.B)
  (hRawInRange : GoldilocksRawInRangeBoundAssumption)
  (hCollapse : GoldilocksRawCollapseAssumption)
  (hUpper : Parameters.Goldilocks.B ≤ theorem9UpperBound (maxRhoNorm claim.cset))
  (hP18 : p20PolyProp claim.qVals ctx.ell ctx.totalDegree ctx.setSize)
  (hP19 : p20InterpProp claim.xs claim.ys claim.expectedCoeffs claim.evalPoint claim.expectedEval)
  (hWitness : witness.z = claim.z)
  (hNorm : normInfCoeffs witness.z < ctx.ceNormBound) :
  PSCEValid ctx claim witness := by
  have hP20 : p20ForClaim ctx claim :=
    p20ForClaim_of_props_with_sampling_goldilocks_operand_assumptions_inRange
      (hP6 := hP6)
      (hP12Rows := claimShapeValid_matrixRowsCompatible hShape)
      (hP12Eq := hP12Eq)
      (hP14 := hP14)
      (hP15Vec := hP15Vec)
      (hP15Scal := hP15Scal)
      (hP16Win := hP16Win)
      (hCset := hCset)
      (hSamples := hSamples)
      (hRawInRange := hRawInRange)
      (hCollapse := hCollapse)
      (hUpper := hUpper)
      (hP18 := hP18)
      (hP19 := hP19)
  exact superneoMathProtocolSkeleton_of_props hShape hBar hA hB hP10 hP20 hWitness hNorm

theorem superneoMathProtocolSkeleton_of_props_with_sampling_goldilocks_operand_rawCoeff_assumptions
  {ctx : PSContext} {claim : PSClaim} {witness : PSWitness}
  (hShape : ClaimShapeValid claim)
  (hBar : IsDBarMatrix ctx.bar)
  (hA : IsDVec claim.a)
  (hB : IsDVec claim.b)
  (hP10 : p10ForClaim ctx claim)
  (hP6 : p20DecompProp claim.zDecomp ctx.bSplit ctx.kSplit)
  (hP12Eq : matrixVecDirect claim.m claim.z = matrixVecCtBar ctx.bar claim.m claim.z)
  (hP14 : p20EvalHomProp ctx.bar claim.m claim.z1 claim.z2 claim.r claim.rho1 claim.rho2)
  (hP15Vec : p20VecModuleProp ctx.hVec claim.rho1 claim.z1 claim.z2)
  (hP15Scal : p20ScalarModuleProp ctx.hScal claim.rho1 claim.z1 claim.z2)
  (hP16Win : p20InvertibilityWindowProp claim.invDelta)
  (hCset : ∀ i : Fin claim.cset.size, normInfCoeffs claim.cset[i] ≤ Parameters.Goldilocks.B)
  (hSamples : ∀ j : Fin claim.samples.size, normInfCoeffs claim.samples[j] ≤ Parameters.Goldilocks.B)
  (hRawCoeff : GoldilocksRawCoeffBoundAssumption)
  (hCollapse : GoldilocksRawCollapseAssumption)
  (hUpper : Parameters.Goldilocks.B ≤ theorem9UpperBound (maxRhoNorm claim.cset))
  (hP18 : p20PolyProp claim.qVals ctx.ell ctx.totalDegree ctx.setSize)
  (hP19 : p20InterpProp claim.xs claim.ys claim.expectedCoeffs claim.evalPoint claim.expectedEval)
  (hWitness : witness.z = claim.z)
  (hNorm : normInfCoeffs witness.z < ctx.ceNormBound) :
  PSCEValid ctx claim witness := by
  have hP20 : p20ForClaim ctx claim :=
    p20ForClaim_of_props_with_sampling_goldilocks_operand_rawCoeff_assumptions
      (hP6 := hP6)
      (hP12Rows := claimShapeValid_matrixRowsCompatible hShape)
      (hP12Eq := hP12Eq)
      (hP14 := hP14)
      (hP15Vec := hP15Vec)
      (hP15Scal := hP15Scal)
      (hP16Win := hP16Win)
      (hCset := hCset)
      (hSamples := hSamples)
      (hRawCoeff := hRawCoeff)
      (hCollapse := hCollapse)
      (hUpper := hUpper)
      (hP18 := hP18)
      (hP19 := hP19)
  exact superneoMathProtocolSkeleton_of_props hShape hBar hA hB hP10 hP20 hWitness hNorm

theorem superneoMathProtocolSkeleton_of_props_with_sampling_goldilocks_operand_rawCoeff_fieldOp_assumptions
  {ctx : PSContext} {claim : PSClaim} {witness : PSWitness}
  (hShape : ClaimShapeValid claim)
  (hBar : IsDBarMatrix ctx.bar)
  (hA : IsDVec claim.a)
  (hB : IsDVec claim.b)
  (hP10 : p10ForClaim ctx claim)
  (hP6 : p20DecompProp claim.zDecomp ctx.bSplit ctx.kSplit)
  (hP12Eq : matrixVecDirect claim.m claim.z = matrixVecCtBar ctx.bar claim.m claim.z)
  (hP14 : p20EvalHomProp ctx.bar claim.m claim.z1 claim.z2 claim.r claim.rho1 claim.rho2)
  (hP15Vec : p20VecModuleProp ctx.hVec claim.rho1 claim.z1 claim.z2)
  (hP15Scal : p20ScalarModuleProp ctx.hScal claim.rho1 claim.z1 claim.z2)
  (hP16Win : p20InvertibilityWindowProp claim.invDelta)
  (hCset : ∀ i : Fin claim.cset.size, normInfCoeffs claim.cset[i] ≤ Parameters.Goldilocks.B)
  (hSamples : ∀ j : Fin claim.samples.size, normInfCoeffs claim.samples[j] ≤ Parameters.Goldilocks.B)
  (hRawCoeff : GoldilocksRawCoeffBoundAssumption)
  (hFieldOps : GoldilocksFieldOpCollapseAssumption)
  (hUpper : Parameters.Goldilocks.B ≤ theorem9UpperBound (maxRhoNorm claim.cset))
  (hP18 : p20PolyProp claim.qVals ctx.ell ctx.totalDegree ctx.setSize)
  (hP19 : p20InterpProp claim.xs claim.ys claim.expectedCoeffs claim.evalPoint claim.expectedEval)
  (hWitness : witness.z = claim.z)
  (hNorm : normInfCoeffs witness.z < ctx.ceNormBound) :
  PSCEValid ctx claim witness := by
  have hP20 : p20ForClaim ctx claim :=
    p20ForClaim_of_props_with_sampling_goldilocks_operand_rawCoeff_fieldOp_assumptions
      (hP6 := hP6)
      (hP12Rows := claimShapeValid_matrixRowsCompatible hShape)
      (hP12Eq := hP12Eq)
      (hP14 := hP14)
      (hP15Vec := hP15Vec)
      (hP15Scal := hP15Scal)
      (hP16Win := hP16Win)
      (hCset := hCset)
      (hSamples := hSamples)
      (hRawCoeff := hRawCoeff)
      (hFieldOps := hFieldOps)
      (hUpper := hUpper)
      (hP18 := hP18)
      (hP19 := hP19)
  exact superneoMathProtocolSkeleton_of_props hShape hBar hA hB hP10 hP20 hWitness hNorm

theorem superneoMathProtocolSkeleton_of_props_with_sampling_operand_assumptions
  {ctx : PSContext} {claim : PSClaim} {witness : PSWitness}
  {BA BB BRaw : Nat}
  (hShape : ClaimShapeValid claim)
  (hBar : IsDBarMatrix ctx.bar)
  (hA : IsDVec claim.a)
  (hB : IsDVec claim.b)
  (hP10 : p10ForClaim ctx claim)
  (hP6 : p20DecompProp claim.zDecomp ctx.bSplit ctx.kSplit)
  (hP12Eq : matrixVecDirect claim.m claim.z = matrixVecCtBar ctx.bar claim.m claim.z)
  (hP14 : p20EvalHomProp ctx.bar claim.m claim.z1 claim.z2 claim.r claim.rho1 claim.rho2)
  (hP15Vec : p20VecModuleProp ctx.hVec claim.rho1 claim.z1 claim.z2)
  (hP15Scal : p20ScalarModuleProp ctx.hScal claim.rho1 claim.z1 claim.z2)
  (hP16Win : p20InvertibilityWindowProp claim.invDelta)
  (hCset : ∀ i : Fin claim.cset.size, normInfCoeffs claim.cset[i] ≤ BA)
  (hSamples : ∀ j : Fin claim.samples.size, normInfCoeffs claim.samples[j] ≤ BB)
  (hRaw : mulRqRawNormBoundFromOperands BA BB BRaw)
  (hAddSub : rawAddSubCollapseBound BRaw (theorem9UpperBound (maxRhoNorm claim.cset)))
  (hSub : rawSubCollapseBound BRaw (theorem9UpperBound (maxRhoNorm claim.cset)))
  (hP18 : p20PolyProp claim.qVals ctx.ell ctx.totalDegree ctx.setSize)
  (hP19 : p20InterpProp claim.xs claim.ys claim.expectedCoeffs claim.evalPoint claim.expectedEval)
  (hWitness : witness.z = claim.z)
  (hNorm : normInfCoeffs witness.z < ctx.ceNormBound) :
  PSCEValid ctx claim witness := by
  have hP20 : p20ForClaim ctx claim :=
    p20ForClaim_of_props_with_sampling_operand_assumptions
      (hP6 := hP6)
      (hP12Rows := claimShapeValid_matrixRowsCompatible hShape)
      (hP12Eq := hP12Eq)
      (hP14 := hP14)
      (hP15Vec := hP15Vec)
      (hP15Scal := hP15Scal)
      (hP16Win := hP16Win)
      (hCset := hCset)
      (hSamples := hSamples)
      (hRaw := hRaw)
      (hAddSub := hAddSub)
      (hSub := hSub)
      (hP18 := hP18)
      (hP19 := hP19)
  exact superneoMathProtocolSkeleton_of_props hShape hBar hA hB hP10 hP20 hWitness hNorm

theorem superneoMathProtocolSkeleton_of_props_with_sampling_operand_assumptions_via_universal_blockers
  {ctx : PSContext} {claim : PSClaim} {witness : PSWitness}
  {BA BB : Nat}
  (hShape : ClaimShapeValid claim)
  (hBar : IsDBarMatrix ctx.bar)
  (hA : IsDVec claim.a)
  (hB : IsDVec claim.b)
  (hP10 : p10ForClaim ctx claim)
  (hP6 : p20DecompProp claim.zDecomp ctx.bSplit ctx.kSplit)
  (hP12Eq : matrixVecDirect claim.m claim.z = matrixVecCtBar ctx.bar claim.m claim.z)
  (hP14 : p20EvalHomProp ctx.bar claim.m claim.z1 claim.z2 claim.r claim.rho1 claim.rho2)
  (hP15Vec : p20VecModuleProp ctx.hVec claim.rho1 claim.z1 claim.z2)
  (hP15Scal : p20ScalarModuleProp ctx.hScal claim.rho1 claim.z1 claim.z2)
  (hP16Win : p20InvertibilityWindowProp claim.invDelta)
  (hCset : ∀ i : Fin claim.cset.size, normInfCoeffs claim.cset[i] ≤ BA)
  (hSamples : ∀ j : Fin claim.samples.size, normInfCoeffs claim.samples[j] ≤ BB)
  (hMulUniv : schoolbookMulUniversalBound)
  (hAddTri : schoolbookAddTriangleBound)
  (hSubTri : schoolbookSubTriangleBound)
  (hRawLe :
    ((D * D) * (BA * BB)) + ((D * D) * (BA * BB)) + ((D * D) * (BA * BB))
      ≤ theorem9UpperBound (maxRhoNorm claim.cset))
  (hP18 : p20PolyProp claim.qVals ctx.ell ctx.totalDegree ctx.setSize)
  (hP19 : p20InterpProp claim.xs claim.ys claim.expectedCoeffs claim.evalPoint claim.expectedEval)
  (hWitness : witness.z = claim.z)
  (hNorm : normInfCoeffs witness.z < ctx.ceNormBound) :
  PSCEValid ctx claim witness := by
  have hP20 : p20ForClaim ctx claim :=
    p20ForClaim_of_props_with_sampling_operand_assumptions_via_universal_blockers
      (hP6 := hP6)
      (hP12Rows := claimShapeValid_matrixRowsCompatible hShape)
      (hP12Eq := hP12Eq)
      (hP14 := hP14)
      (hP15Vec := hP15Vec)
      (hP15Scal := hP15Scal)
      (hP16Win := hP16Win)
      (hCset := hCset)
      (hSamples := hSamples)
      (hMulUniv := hMulUniv)
      (hAddTri := hAddTri)
      (hSubTri := hSubTri)
      (hRawLe := hRawLe)
      (hP18 := hP18)
      (hP19 := hP19)
  exact superneoMathProtocolSkeleton_of_props hShape hBar hA hB hP10 hP20 hWitness hNorm

theorem superneoMathProtocolSkeleton_of_props_with_sampling_operand_assumptions_via_universal_blockers_tight
  {ctx : PSContext} {claim : PSClaim} {witness : PSWitness}
  {BA BB : Nat}
  (hShape : ClaimShapeValid claim)
  (hBar : IsDBarMatrix ctx.bar)
  (hA : IsDVec claim.a)
  (hB : IsDVec claim.b)
  (hP10 : p10ForClaim ctx claim)
  (hP6 : p20DecompProp claim.zDecomp ctx.bSplit ctx.kSplit)
  (hP12Eq : matrixVecDirect claim.m claim.z = matrixVecCtBar ctx.bar claim.m claim.z)
  (hP14 : p20EvalHomProp ctx.bar claim.m claim.z1 claim.z2 claim.r claim.rho1 claim.rho2)
  (hP15Vec : p20VecModuleProp ctx.hVec claim.rho1 claim.z1 claim.z2)
  (hP15Scal : p20ScalarModuleProp ctx.hScal claim.rho1 claim.z1 claim.z2)
  (hP16Win : p20InvertibilityWindowProp claim.invDelta)
  (hCset : ∀ i : Fin claim.cset.size, normInfCoeffs claim.cset[i] ≤ BA)
  (hSamples : ∀ j : Fin claim.samples.size, normInfCoeffs claim.samples[j] ≤ BB)
  (hMulUniv : schoolbookMulUniversalBound)
  (hAddTri : schoolbookAddTriangleBound)
  (hSubTri : schoolbookSubTriangleBound)
  (hRawLe :
    (D * (BA * BB)) + (D * (BA * BB)) + (D * (BA * BB))
      ≤ theorem9UpperBound (maxRhoNorm claim.cset))
  (hP18 : p20PolyProp claim.qVals ctx.ell ctx.totalDegree ctx.setSize)
  (hP19 : p20InterpProp claim.xs claim.ys claim.expectedCoeffs claim.evalPoint claim.expectedEval)
  (hWitness : witness.z = claim.z)
  (hNorm : normInfCoeffs witness.z < ctx.ceNormBound) :
  PSCEValid ctx claim witness := by
  have hP20 : p20ForClaim ctx claim :=
    p20ForClaim_of_props_with_sampling_operand_assumptions_via_universal_blockers_tight
      (hP6 := hP6)
      (hP12Rows := claimShapeValid_matrixRowsCompatible hShape)
      (hP12Eq := hP12Eq)
      (hP14 := hP14)
      (hP15Vec := hP15Vec)
      (hP15Scal := hP15Scal)
      (hP16Win := hP16Win)
      (hCset := hCset)
      (hSamples := hSamples)
      (hMulUniv := hMulUniv)
      (hAddTri := hAddTri)
      (hSubTri := hSubTri)
      (hRawLe := hRawLe)
      (hP18 := hP18)
      (hP19 := hP19)
  exact superneoMathProtocolSkeleton_of_props hShape hBar hA hB hP10 hP20 hWitness hNorm

/--
Triangle-bundle variant of
`superneoMathProtocolSkeleton_of_props_with_sampling_operand_assumptions_via_universal_blockers_tight`.
-/
theorem superneoMathProtocolSkeleton_of_props_with_sampling_operand_assumptions_via_universal_mul_and_triangles_tight
  {ctx : PSContext} {claim : PSClaim} {witness : PSWitness}
  {BA BB : Nat}
  (hShape : ClaimShapeValid claim)
  (hBar : IsDBarMatrix ctx.bar)
  (hA : IsDVec claim.a)
  (hB : IsDVec claim.b)
  (hP10 : p10ForClaim ctx claim)
  (hP6 : p20DecompProp claim.zDecomp ctx.bSplit ctx.kSplit)
  (hP12Eq : matrixVecDirect claim.m claim.z = matrixVecCtBar ctx.bar claim.m claim.z)
  (hP14 : p20EvalHomProp ctx.bar claim.m claim.z1 claim.z2 claim.r claim.rho1 claim.rho2)
  (hP15Vec : p20VecModuleProp ctx.hVec claim.rho1 claim.z1 claim.z2)
  (hP15Scal : p20ScalarModuleProp ctx.hScal claim.rho1 claim.z1 claim.z2)
  (hP16Win : p20InvertibilityWindowProp claim.invDelta)
  (hCset : ∀ i : Fin claim.cset.size, normInfCoeffs claim.cset[i] ≤ BA)
  (hSamples : ∀ j : Fin claim.samples.size, normInfCoeffs claim.samples[j] ≤ BB)
  (hMulUniv : schoolbookMulUniversalBound)
  (hTri : schoolbookTriangleBounds)
  (hRawLe :
    (D * (BA * BB)) + (D * (BA * BB)) + (D * (BA * BB))
      ≤ theorem9UpperBound (maxRhoNorm claim.cset))
  (hP18 : p20PolyProp claim.qVals ctx.ell ctx.totalDegree ctx.setSize)
  (hP19 : p20InterpProp claim.xs claim.ys claim.expectedCoeffs claim.evalPoint claim.expectedEval)
  (hWitness : witness.z = claim.z)
  (hNorm : normInfCoeffs witness.z < ctx.ceNormBound) :
  PSCEValid ctx claim witness := by
  have hP20 : p20ForClaim ctx claim :=
    p20ForClaim_of_props_with_sampling_operand_assumptions_via_universal_mul_and_triangles_tight
      (hP6 := hP6)
      (hP12Rows := claimShapeValid_matrixRowsCompatible hShape)
      (hP12Eq := hP12Eq)
      (hP14 := hP14)
      (hP15Vec := hP15Vec)
      (hP15Scal := hP15Scal)
      (hP16Win := hP16Win)
      (hCset := hCset)
      (hSamples := hSamples)
      (hMulUniv := hMulUniv)
      (hTri := hTri)
      (hRawLe := hRawLe)
      (hP18 := hP18)
      (hP19 := hP19)
  exact superneoMathProtocolSkeleton_of_props hShape hBar hA hB hP10 hP20 hWitness hNorm

/--
Add-only variant of
`superneoMathProtocolSkeleton_of_props_with_sampling_operand_assumptions_via_universal_mul_and_triangles_tight`.
-/
theorem superneoMathProtocolSkeleton_of_props_with_sampling_operand_assumptions_via_universal_mul_and_add_tight
  {ctx : PSContext} {claim : PSClaim} {witness : PSWitness}
  {BA BB : Nat}
  (hShape : ClaimShapeValid claim)
  (hBar : IsDBarMatrix ctx.bar)
  (hA : IsDVec claim.a)
  (hB : IsDVec claim.b)
  (hP10 : p10ForClaim ctx claim)
  (hP6 : p20DecompProp claim.zDecomp ctx.bSplit ctx.kSplit)
  (hP12Eq : matrixVecDirect claim.m claim.z = matrixVecCtBar ctx.bar claim.m claim.z)
  (hP14 : p20EvalHomProp ctx.bar claim.m claim.z1 claim.z2 claim.r claim.rho1 claim.rho2)
  (hP15Vec : p20VecModuleProp ctx.hVec claim.rho1 claim.z1 claim.z2)
  (hP15Scal : p20ScalarModuleProp ctx.hScal claim.rho1 claim.z1 claim.z2)
  (hP16Win : p20InvertibilityWindowProp claim.invDelta)
  (hCset : ∀ i : Fin claim.cset.size, normInfCoeffs claim.cset[i] ≤ BA)
  (hSamples : ∀ j : Fin claim.samples.size, normInfCoeffs claim.samples[j] ≤ BB)
  (hMulUniv : schoolbookMulUniversalBound)
  (hAddTri : schoolbookAddTriangleBound)
  (hRawLe :
    (D * (BA * BB)) + (D * (BA * BB)) + (D * (BA * BB))
      ≤ theorem9UpperBound (maxRhoNorm claim.cset))
  (hP18 : p20PolyProp claim.qVals ctx.ell ctx.totalDegree ctx.setSize)
  (hP19 : p20InterpProp claim.xs claim.ys claim.expectedCoeffs claim.evalPoint claim.expectedEval)
  (hWitness : witness.z = claim.z)
  (hNorm : normInfCoeffs witness.z < ctx.ceNormBound) :
  PSCEValid ctx claim witness := by
  have hP20 : p20ForClaim ctx claim :=
    p20ForClaim_of_props_with_sampling_operand_assumptions_via_universal_mul_and_add_tight
      (hP6 := hP6)
      (hP12Rows := claimShapeValid_matrixRowsCompatible hShape)
      (hP12Eq := hP12Eq)
      (hP14 := hP14)
      (hP15Vec := hP15Vec)
      (hP15Scal := hP15Scal)
      (hP16Win := hP16Win)
      (hCset := hCset)
      (hSamples := hSamples)
      (hMulUniv := hMulUniv)
      (hAddTri := hAddTri)
      (hRawLe := hRawLe)
      (hP18 := hP18)
      (hP19 := hP19)
  exact superneoMathProtocolSkeleton_of_props hShape hBar hA hB hP10 hP20 hWitness hNorm

theorem superneoMathProtocolSkeleton_of_props_with_sampling_operand_assumptions_via_centeredRep_mul_and_add_tight
  {ctx : PSContext} {claim : PSClaim} {witness : PSWitness}
  {BA BB : Nat}
  (hShape : ClaimShapeValid claim)
  (hBar : IsDBarMatrix ctx.bar)
  (hA : IsDVec claim.a)
  (hB : IsDVec claim.b)
  (hP10 : p10ForClaim ctx claim)
  (hP6 : p20DecompProp claim.zDecomp ctx.bSplit ctx.kSplit)
  (hP12Eq : matrixVecDirect claim.m claim.z = matrixVecCtBar ctx.bar claim.m claim.z)
  (hP14 : p20EvalHomProp ctx.bar claim.m claim.z1 claim.z2 claim.r claim.rho1 claim.rho2)
  (hP15Vec : p20VecModuleProp ctx.hVec claim.rho1 claim.z1 claim.z2)
  (hP15Scal : p20ScalarModuleProp ctx.hScal claim.rho1 claim.z1 claim.z2)
  (hP16Win : p20InvertibilityWindowProp claim.invDelta)
  (hCset : ∀ i : Fin claim.cset.size, normInfCoeffs claim.cset[i] ≤ BA)
  (hSamples : ∀ j : Fin claim.samples.size, normInfCoeffs claim.samples[j] ≤ BB)
  (hMulRep :
    ∀ x y : F, Int.natAbs (centeredRep (x * y))
      ≤ Int.natAbs (centeredRep x) * Int.natAbs (centeredRep y))
  (hAddRep :
    ∀ x y : F, Int.natAbs (centeredRep (x + y))
      ≤ Int.natAbs (centeredRep x) + Int.natAbs (centeredRep y))
  (hRawLe :
    (D * (BA * BB)) + (D * (BA * BB)) + (D * (BA * BB))
      ≤ theorem9UpperBound (maxRhoNorm claim.cset))
  (hP18 : p20PolyProp claim.qVals ctx.ell ctx.totalDegree ctx.setSize)
  (hP19 : p20InterpProp claim.xs claim.ys claim.expectedCoeffs claim.evalPoint claim.expectedEval)
  (hWitness : witness.z = claim.z)
  (hNorm : normInfCoeffs witness.z < ctx.ceNormBound) :
  PSCEValid ctx claim witness := by
  have hP20 : p20ForClaim ctx claim :=
    p20ForClaim_of_props_with_sampling_operand_assumptions_via_centeredRep_mul_and_add_tight
      (hP6 := hP6)
      (hP12Rows := claimShapeValid_matrixRowsCompatible hShape)
      (hP12Eq := hP12Eq)
      (hP14 := hP14)
      (hP15Vec := hP15Vec)
      (hP15Scal := hP15Scal)
      (hP16Win := hP16Win)
      (hCset := hCset)
      (hSamples := hSamples)
      (hMulRep := hMulRep)
      (hAddRep := hAddRep)
      (hRawLe := hRawLe)
      (hP18 := hP18)
      (hP19 := hP19)
  exact superneoMathProtocolSkeleton_of_props hShape hBar hA hB hP10 hP20 hWitness hNorm

theorem superneoMathProtocolSkeleton_of_props_with_sampling_operand_assumptions_via_centeredRep_mul_tight
  {ctx : PSContext} {claim : PSClaim} {witness : PSWitness}
  {BA BB : Nat}
  (hShape : ClaimShapeValid claim)
  (hBar : IsDBarMatrix ctx.bar)
  (hA : IsDVec claim.a)
  (hB : IsDVec claim.b)
  (hP10 : p10ForClaim ctx claim)
  (hP6 : p20DecompProp claim.zDecomp ctx.bSplit ctx.kSplit)
  (hP12Eq : matrixVecDirect claim.m claim.z = matrixVecCtBar ctx.bar claim.m claim.z)
  (hP14 : p20EvalHomProp ctx.bar claim.m claim.z1 claim.z2 claim.r claim.rho1 claim.rho2)
  (hP15Vec : p20VecModuleProp ctx.hVec claim.rho1 claim.z1 claim.z2)
  (hP15Scal : p20ScalarModuleProp ctx.hScal claim.rho1 claim.z1 claim.z2)
  (hP16Win : p20InvertibilityWindowProp claim.invDelta)
  (hCset : ∀ i : Fin claim.cset.size, normInfCoeffs claim.cset[i] ≤ BA)
  (hSamples : ∀ j : Fin claim.samples.size, normInfCoeffs claim.samples[j] ≤ BB)
  (hMulRep :
    ∀ x y : F, Int.natAbs (centeredRep (x * y))
      ≤ Int.natAbs (centeredRep x) * Int.natAbs (centeredRep y))
  (hRawLe :
    (D * (BA * BB)) + (D * (BA * BB)) + (D * (BA * BB))
      ≤ theorem9UpperBound (maxRhoNorm claim.cset))
  (hP18 : p20PolyProp claim.qVals ctx.ell ctx.totalDegree ctx.setSize)
  (hP19 : p20InterpProp claim.xs claim.ys claim.expectedCoeffs claim.evalPoint claim.expectedEval)
  (hWitness : witness.z = claim.z)
  (hNorm : normInfCoeffs witness.z < ctx.ceNormBound) :
  PSCEValid ctx claim witness := by
  exact superneoMathProtocolSkeleton_of_props_with_sampling_operand_assumptions_via_centeredRep_mul_and_add_tight
    (hShape := hShape)
    (hBar := hBar)
    (hA := hA)
    (hB := hB)
    (hP10 := hP10)
    (hP6 := hP6)
    (hP12Eq := hP12Eq)
    (hP14 := hP14)
    (hP15Vec := hP15Vec)
    (hP15Scal := hP15Scal)
    (hP16Win := hP16Win)
    (hCset := hCset)
    (hSamples := hSamples)
    (hMulRep := hMulRep)
    (hAddRep := centeredRepAddTriangleBound_theorem)
    (hRawLe := hRawLe)
    (hP18 := hP18)
    (hP19 := hP19)
    (hWitness := hWitness)
    (hNorm := hNorm)

/-- Bundle wrapper for centered-representation mul/add blockers (tight path). -/
theorem superneoMathProtocolSkeleton_of_props_with_sampling_operand_assumptions_via_centeredRepMulAddBounds_tight
  {ctx : PSContext} {claim : PSClaim} {witness : PSWitness}
  {BA BB : Nat}
  (hShape : ClaimShapeValid claim)
  (hBar : IsDBarMatrix ctx.bar)
  (hA : IsDVec claim.a)
  (hB : IsDVec claim.b)
  (hP10 : p10ForClaim ctx claim)
  (hP6 : p20DecompProp claim.zDecomp ctx.bSplit ctx.kSplit)
  (hP12Eq : matrixVecDirect claim.m claim.z = matrixVecCtBar ctx.bar claim.m claim.z)
  (hP14 : p20EvalHomProp ctx.bar claim.m claim.z1 claim.z2 claim.r claim.rho1 claim.rho2)
  (hP15Vec : p20VecModuleProp ctx.hVec claim.rho1 claim.z1 claim.z2)
  (hP15Scal : p20ScalarModuleProp ctx.hScal claim.rho1 claim.z1 claim.z2)
  (hP16Win : p20InvertibilityWindowProp claim.invDelta)
  (hCset : ∀ i : Fin claim.cset.size, normInfCoeffs claim.cset[i] ≤ BA)
  (hSamples : ∀ j : Fin claim.samples.size, normInfCoeffs claim.samples[j] ≤ BB)
  (hRep : centeredRepMulAddBounds)
  (hRawLe :
    (D * (BA * BB)) + (D * (BA * BB)) + (D * (BA * BB))
      ≤ theorem9UpperBound (maxRhoNorm claim.cset))
  (hP18 : p20PolyProp claim.qVals ctx.ell ctx.totalDegree ctx.setSize)
  (hP19 : p20InterpProp claim.xs claim.ys claim.expectedCoeffs claim.evalPoint claim.expectedEval)
  (hWitness : witness.z = claim.z)
  (hNorm : normInfCoeffs witness.z < ctx.ceNormBound) :
  PSCEValid ctx claim witness := by
  exact superneoMathProtocolSkeleton_of_props_with_sampling_operand_assumptions_via_centeredRep_mul_and_add_tight
    (hShape := hShape)
    (hBar := hBar)
    (hA := hA)
    (hB := hB)
    (hP10 := hP10)
    (hP6 := hP6)
    (hP12Eq := hP12Eq)
    (hP14 := hP14)
    (hP15Vec := hP15Vec)
    (hP15Scal := hP15Scal)
    (hP16Win := hP16Win)
    (hCset := hCset)
    (hSamples := hSamples)
    (hMulRep := centeredRepMulAddBounds_mul hRep)
    (hAddRep := centeredRepMulAddBounds_add hRep)
    (hRawLe := hRawLe)
    (hP18 := hP18)
    (hP19 := hP19)
    (hWitness := hWitness)
    (hNorm := hNorm)

/--
Assumption-free native skeleton constructor.
Specializes the sampling side to the native P5 path.
-/
theorem superneoMathProtocolSkeleton_of_props_with_sampling_operand_assumptions_native
  {ctx : PSContext} {claim : PSClaim} {witness : PSWitness}
  {BA BB : Nat}
  (hShape : ClaimShapeValid claim)
  (hBar : IsDBarMatrix ctx.bar)
  (hA : IsDVec claim.a)
  (hB : IsDVec claim.b)
  (hP10 : p10ForClaim ctx claim)
  (hP6 : p20DecompProp claim.zDecomp ctx.bSplit ctx.kSplit)
  (hP12Eq : matrixVecDirect claim.m claim.z = matrixVecCtBar ctx.bar claim.m claim.z)
  (hP14 : p20EvalHomProp ctx.bar claim.m claim.z1 claim.z2 claim.r claim.rho1 claim.rho2)
  (hP15Vec : p20VecModuleProp ctx.hVec claim.rho1 claim.z1 claim.z2)
  (hP15Scal : p20ScalarModuleProp ctx.hScal claim.rho1 claim.z1 claim.z2)
  (hP16Win : p20InvertibilityWindowProp claim.invDelta)
  (hCset : ∀ i : Fin claim.cset.size, normInfCoeffs claim.cset[i] ≤ BA)
  (hSamples : ∀ j : Fin claim.samples.size, normInfCoeffs claim.samples[j] ≤ BB)
  (hRawLe :
    ((D * D) * (BA * BB)) + ((D * D) * (BA * BB)) + ((D * D) * (BA * BB))
      ≤ theorem9UpperBound (maxRhoNorm claim.cset))
  (hP18 : p20PolyProp claim.qVals ctx.ell ctx.totalDegree ctx.setSize)
  (hP19 : p20InterpProp claim.xs claim.ys claim.expectedCoeffs claim.evalPoint claim.expectedEval)
  (hWitness : witness.z = claim.z)
  (hNorm : normInfCoeffs witness.z < ctx.ceNormBound) :
  PSCEValid ctx claim witness := by
  have hP20 : p20ForClaim ctx claim :=
    p20ForClaim_of_props_with_sampling_operand_assumptions_native
      (hP6 := hP6)
      (hP12Rows := claimShapeValid_matrixRowsCompatible hShape)
      (hP12Eq := hP12Eq)
      (hP14 := hP14)
      (hP15Vec := hP15Vec)
      (hP15Scal := hP15Scal)
      (hP16Win := hP16Win)
      (hCset := hCset)
      (hSamples := hSamples)
      (hRawLe := hRawLe)
      (hP18 := hP18)
      (hP19 := hP19)
  exact superneoMathProtocolSkeleton_of_props hShape hBar hA hB hP10 hP20 hWitness hNorm

/--
Assumption-free native-tight skeleton constructor.
Specializes the sampling side to the native-tight P5 path.
-/
theorem superneoMathProtocolSkeleton_of_props_with_sampling_operand_assumptions_native_tight
  {ctx : PSContext} {claim : PSClaim} {witness : PSWitness}
  {BA BB : Nat}
  (hShape : ClaimShapeValid claim)
  (hBar : IsDBarMatrix ctx.bar)
  (hA : IsDVec claim.a)
  (hB : IsDVec claim.b)
  (hP10 : p10ForClaim ctx claim)
  (hP6 : p20DecompProp claim.zDecomp ctx.bSplit ctx.kSplit)
  (hP12Eq : matrixVecDirect claim.m claim.z = matrixVecCtBar ctx.bar claim.m claim.z)
  (hP14 : p20EvalHomProp ctx.bar claim.m claim.z1 claim.z2 claim.r claim.rho1 claim.rho2)
  (hP15Vec : p20VecModuleProp ctx.hVec claim.rho1 claim.z1 claim.z2)
  (hP15Scal : p20ScalarModuleProp ctx.hScal claim.rho1 claim.z1 claim.z2)
  (hP16Win : p20InvertibilityWindowProp claim.invDelta)
  (hCset : ∀ i : Fin claim.cset.size, normInfCoeffs claim.cset[i] ≤ BA)
  (hSamples : ∀ j : Fin claim.samples.size, normInfCoeffs claim.samples[j] ≤ BB)
  (hRawLe :
    (D * (BA * BB)) + (D * (BA * BB)) + (D * (BA * BB))
      ≤ theorem9UpperBound (maxRhoNorm claim.cset))
  (hP18 : p20PolyProp claim.qVals ctx.ell ctx.totalDegree ctx.setSize)
  (hP19 : p20InterpProp claim.xs claim.ys claim.expectedCoeffs claim.evalPoint claim.expectedEval)
  (hWitness : witness.z = claim.z)
  (hNorm : normInfCoeffs witness.z < ctx.ceNormBound) :
  PSCEValid ctx claim witness := by
  have hP20 : p20ForClaim ctx claim :=
    p20ForClaim_of_props_with_sampling_operand_assumptions_native_tight
      (hP6 := hP6)
      (hP12Rows := claimShapeValid_matrixRowsCompatible hShape)
      (hP12Eq := hP12Eq)
      (hP14 := hP14)
      (hP15Vec := hP15Vec)
      (hP15Scal := hP15Scal)
      (hP16Win := hP16Win)
      (hCset := hCset)
      (hSamples := hSamples)
      (hRawLe := hRawLe)
      (hP18 := hP18)
      (hP19 := hP19)
  exact superneoMathProtocolSkeleton_of_props hShape hBar hA hB hP10 hP20 hWitness hNorm

theorem superneoMathProtocolSkeleton_of_props_with_sampling_operand_assumptions_via_universal_blockers_and_raw
  {ctx : PSContext} {claim : PSClaim} {witness : PSWitness}
  {BA BB BRaw : Nat}
  (hShape : ClaimShapeValid claim)
  (hBar : IsDBarMatrix ctx.bar)
  (hA : IsDVec claim.a)
  (hB : IsDVec claim.b)
  (hP10 : p10ForClaim ctx claim)
  (hP6 : p20DecompProp claim.zDecomp ctx.bSplit ctx.kSplit)
  (hP12Eq : matrixVecDirect claim.m claim.z = matrixVecCtBar ctx.bar claim.m claim.z)
  (hP14 : p20EvalHomProp ctx.bar claim.m claim.z1 claim.z2 claim.r claim.rho1 claim.rho2)
  (hP15Vec : p20VecModuleProp ctx.hVec claim.rho1 claim.z1 claim.z2)
  (hP15Scal : p20ScalarModuleProp ctx.hScal claim.rho1 claim.z1 claim.z2)
  (hP16Win : p20InvertibilityWindowProp claim.invDelta)
  (hCset : ∀ i : Fin claim.cset.size, normInfCoeffs claim.cset[i] ≤ BA)
  (hSamples : ∀ j : Fin claim.samples.size, normInfCoeffs claim.samples[j] ≤ BB)
  (hRaw : mulRqRawNormBoundFromOperands BA BB BRaw)
  (hAddTri : schoolbookAddTriangleBound)
  (hSubTri : schoolbookSubTriangleBound)
  (hRawLe : BRaw + BRaw + BRaw ≤ theorem9UpperBound (maxRhoNorm claim.cset))
  (hP18 : p20PolyProp claim.qVals ctx.ell ctx.totalDegree ctx.setSize)
  (hP19 : p20InterpProp claim.xs claim.ys claim.expectedCoeffs claim.evalPoint claim.expectedEval)
  (hWitness : witness.z = claim.z)
  (hNorm : normInfCoeffs witness.z < ctx.ceNormBound) :
  PSCEValid ctx claim witness := by
  have hP20 : p20ForClaim ctx claim :=
    p20ForClaim_of_props_with_sampling_operand_assumptions_via_universal_blockers_and_raw
      (hP6 := hP6)
      (hP12Rows := claimShapeValid_matrixRowsCompatible hShape)
      (hP12Eq := hP12Eq)
      (hP14 := hP14)
      (hP15Vec := hP15Vec)
      (hP15Scal := hP15Scal)
      (hP16Win := hP16Win)
      (hCset := hCset)
      (hSamples := hSamples)
      (hRaw := hRaw)
      (hAddTri := hAddTri)
      (hSubTri := hSubTri)
      (hRawLe := hRawLe)
      (hP18 := hP18)
      (hP19 := hP19)
  exact superneoMathProtocolSkeleton_of_props hShape hBar hA hB hP10 hP20 hWitness hNorm

theorem superneoMathProtocolSkeleton_of_props_with_sampling_operand_assumptions_via_universal_blockers_and_rawCoeff
  {ctx : PSContext} {claim : PSClaim} {witness : PSWitness}
  {BA BB BRaw : Nat}
  (hShape : ClaimShapeValid claim)
  (hBar : IsDBarMatrix ctx.bar)
  (hA : IsDVec claim.a)
  (hB : IsDVec claim.b)
  (hP10 : p10ForClaim ctx claim)
  (hP6 : p20DecompProp claim.zDecomp ctx.bSplit ctx.kSplit)
  (hP12Eq : matrixVecDirect claim.m claim.z = matrixVecCtBar ctx.bar claim.m claim.z)
  (hP14 : p20EvalHomProp ctx.bar claim.m claim.z1 claim.z2 claim.r claim.rho1 claim.rho2)
  (hP15Vec : p20VecModuleProp ctx.hVec claim.rho1 claim.z1 claim.z2)
  (hP15Scal : p20ScalarModuleProp ctx.hScal claim.rho1 claim.z1 claim.z2)
  (hP16Win : p20InvertibilityWindowProp claim.invDelta)
  (hCset : ∀ i : Fin claim.cset.size, normInfCoeffs claim.cset[i] ≤ BA)
  (hSamples : ∀ j : Fin claim.samples.size, normInfCoeffs claim.samples[j] ≤ BB)
  (hRawCoeff : mulRqRawCoeffBoundFromOperands BA BB BRaw)
  (hAddTri : schoolbookAddTriangleBound)
  (hSubTri : schoolbookSubTriangleBound)
  (hRawLe : BRaw + BRaw + BRaw ≤ theorem9UpperBound (maxRhoNorm claim.cset))
  (hP18 : p20PolyProp claim.qVals ctx.ell ctx.totalDegree ctx.setSize)
  (hP19 : p20InterpProp claim.xs claim.ys claim.expectedCoeffs claim.evalPoint claim.expectedEval)
  (hWitness : witness.z = claim.z)
  (hNorm : normInfCoeffs witness.z < ctx.ceNormBound) :
  PSCEValid ctx claim witness := by
  exact superneoMathProtocolSkeleton_of_props_with_sampling_operand_assumptions_via_universal_blockers_and_raw
    (hShape := hShape)
    (hBar := hBar)
    (hA := hA)
    (hB := hB)
    (hP10 := hP10)
    (hP6 := hP6)
    (hP12Eq := hP12Eq)
    (hP14 := hP14)
    (hP15Vec := hP15Vec)
    (hP15Scal := hP15Scal)
    (hP16Win := hP16Win)
    (hCset := hCset)
    (hSamples := hSamples)
    (hRaw := mulRqRawNormBoundFromOperands_of_rawCoeff hRawCoeff)
    (hAddTri := hAddTri)
    (hSubTri := hSubTri)
    (hRawLe := hRawLe)
    (hP18 := hP18)
    (hP19 := hP19)
    (hWitness := hWitness)
    (hNorm := hNorm)

theorem superneoMathProtocolSkeleton_of_props_with_sampling_operand_assumptions_via_universal_mul_and_add_and_raw
  {ctx : PSContext} {claim : PSClaim} {witness : PSWitness}
  {BA BB BRaw : Nat}
  (hShape : ClaimShapeValid claim)
  (hBar : IsDBarMatrix ctx.bar)
  (hA : IsDVec claim.a)
  (hB : IsDVec claim.b)
  (hP10 : p10ForClaim ctx claim)
  (hP6 : p20DecompProp claim.zDecomp ctx.bSplit ctx.kSplit)
  (hP12Eq : matrixVecDirect claim.m claim.z = matrixVecCtBar ctx.bar claim.m claim.z)
  (hP14 : p20EvalHomProp ctx.bar claim.m claim.z1 claim.z2 claim.r claim.rho1 claim.rho2)
  (hP15Vec : p20VecModuleProp ctx.hVec claim.rho1 claim.z1 claim.z2)
  (hP15Scal : p20ScalarModuleProp ctx.hScal claim.rho1 claim.z1 claim.z2)
  (hP16Win : p20InvertibilityWindowProp claim.invDelta)
  (hCset : ∀ i : Fin claim.cset.size, normInfCoeffs claim.cset[i] ≤ BA)
  (hSamples : ∀ j : Fin claim.samples.size, normInfCoeffs claim.samples[j] ≤ BB)
  (hRaw : mulRqRawNormBoundFromOperands BA BB BRaw)
  (hAddTri : schoolbookAddTriangleBound)
  (hRawLe : BRaw + BRaw + BRaw ≤ theorem9UpperBound (maxRhoNorm claim.cset))
  (hP18 : p20PolyProp claim.qVals ctx.ell ctx.totalDegree ctx.setSize)
  (hP19 : p20InterpProp claim.xs claim.ys claim.expectedCoeffs claim.evalPoint claim.expectedEval)
  (hWitness : witness.z = claim.z)
  (hNorm : normInfCoeffs witness.z < ctx.ceNormBound) :
  PSCEValid ctx claim witness := by
  have hP20 : p20ForClaim ctx claim :=
    p20ForClaim_of_props_with_sampling_operand_assumptions_via_universal_mul_and_add_and_raw
      (hP6 := hP6)
      (hP12Rows := claimShapeValid_matrixRowsCompatible hShape)
      (hP12Eq := hP12Eq)
      (hP14 := hP14)
      (hP15Vec := hP15Vec)
      (hP15Scal := hP15Scal)
      (hP16Win := hP16Win)
      (hCset := hCset)
      (hSamples := hSamples)
      (hRaw := hRaw)
      (hAddTri := hAddTri)
      (hRawLe := hRawLe)
      (hP18 := hP18)
      (hP19 := hP19)
  exact superneoMathProtocolSkeleton_of_props hShape hBar hA hB hP10 hP20 hWitness hNorm

theorem superneoMathProtocolSkeleton_of_props_with_sampling_operand_assumptions_via_universal_mul_and_add_and_rawCoeff
  {ctx : PSContext} {claim : PSClaim} {witness : PSWitness}
  {BA BB BRaw : Nat}
  (hShape : ClaimShapeValid claim)
  (hBar : IsDBarMatrix ctx.bar)
  (hA : IsDVec claim.a)
  (hB : IsDVec claim.b)
  (hP10 : p10ForClaim ctx claim)
  (hP6 : p20DecompProp claim.zDecomp ctx.bSplit ctx.kSplit)
  (hP12Eq : matrixVecDirect claim.m claim.z = matrixVecCtBar ctx.bar claim.m claim.z)
  (hP14 : p20EvalHomProp ctx.bar claim.m claim.z1 claim.z2 claim.r claim.rho1 claim.rho2)
  (hP15Vec : p20VecModuleProp ctx.hVec claim.rho1 claim.z1 claim.z2)
  (hP15Scal : p20ScalarModuleProp ctx.hScal claim.rho1 claim.z1 claim.z2)
  (hP16Win : p20InvertibilityWindowProp claim.invDelta)
  (hCset : ∀ i : Fin claim.cset.size, normInfCoeffs claim.cset[i] ≤ BA)
  (hSamples : ∀ j : Fin claim.samples.size, normInfCoeffs claim.samples[j] ≤ BB)
  (hRawCoeff : mulRqRawCoeffBoundFromOperands BA BB BRaw)
  (hAddTri : schoolbookAddTriangleBound)
  (hRawLe : BRaw + BRaw + BRaw ≤ theorem9UpperBound (maxRhoNorm claim.cset))
  (hP18 : p20PolyProp claim.qVals ctx.ell ctx.totalDegree ctx.setSize)
  (hP19 : p20InterpProp claim.xs claim.ys claim.expectedCoeffs claim.evalPoint claim.expectedEval)
  (hWitness : witness.z = claim.z)
  (hNorm : normInfCoeffs witness.z < ctx.ceNormBound) :
  PSCEValid ctx claim witness := by
  exact superneoMathProtocolSkeleton_of_props_with_sampling_operand_assumptions_via_universal_mul_and_add_and_raw
    (hShape := hShape)
    (hBar := hBar)
    (hA := hA)
    (hB := hB)
    (hP10 := hP10)
    (hP6 := hP6)
    (hP12Eq := hP12Eq)
    (hP14 := hP14)
    (hP15Vec := hP15Vec)
    (hP15Scal := hP15Scal)
    (hP16Win := hP16Win)
    (hCset := hCset)
    (hSamples := hSamples)
    (hRaw := mulRqRawNormBoundFromOperands_of_rawCoeff hRawCoeff)
    (hAddTri := hAddTri)
    (hRawLe := hRawLe)
    (hP18 := hP18)
    (hP19 := hP19)
    (hWitness := hWitness)
    (hNorm := hNorm)

theorem superneoMathProtocolSkeleton_of_props_with_sampling_operand_assumptions_via_centeredRep_mul_and_add_and_raw
  {ctx : PSContext} {claim : PSClaim} {witness : PSWitness}
  {BA BB BRaw : Nat}
  (hShape : ClaimShapeValid claim)
  (hBar : IsDBarMatrix ctx.bar)
  (hA : IsDVec claim.a)
  (hB : IsDVec claim.b)
  (hP10 : p10ForClaim ctx claim)
  (hP6 : p20DecompProp claim.zDecomp ctx.bSplit ctx.kSplit)
  (hP12Eq : matrixVecDirect claim.m claim.z = matrixVecCtBar ctx.bar claim.m claim.z)
  (hP14 : p20EvalHomProp ctx.bar claim.m claim.z1 claim.z2 claim.r claim.rho1 claim.rho2)
  (hP15Vec : p20VecModuleProp ctx.hVec claim.rho1 claim.z1 claim.z2)
  (hP15Scal : p20ScalarModuleProp ctx.hScal claim.rho1 claim.z1 claim.z2)
  (hP16Win : p20InvertibilityWindowProp claim.invDelta)
  (hCset : ∀ i : Fin claim.cset.size, normInfCoeffs claim.cset[i] ≤ BA)
  (hSamples : ∀ j : Fin claim.samples.size, normInfCoeffs claim.samples[j] ≤ BB)
  (hRaw : mulRqRawNormBoundFromOperands BA BB BRaw)
  (hAddRep :
    ∀ x y : F, Int.natAbs (centeredRep (x + y))
      ≤ Int.natAbs (centeredRep x) + Int.natAbs (centeredRep y))
  (hRawLe : BRaw + BRaw + BRaw ≤ theorem9UpperBound (maxRhoNorm claim.cset))
  (hP18 : p20PolyProp claim.qVals ctx.ell ctx.totalDegree ctx.setSize)
  (hP19 : p20InterpProp claim.xs claim.ys claim.expectedCoeffs claim.evalPoint claim.expectedEval)
  (hWitness : witness.z = claim.z)
  (hNorm : normInfCoeffs witness.z < ctx.ceNormBound) :
  PSCEValid ctx claim witness := by
  have hP20 : p20ForClaim ctx claim :=
    p20ForClaim_of_props_with_sampling_operand_assumptions_via_centeredRep_mul_and_add_and_raw
      (hP6 := hP6)
      (hP12Rows := claimShapeValid_matrixRowsCompatible hShape)
      (hP12Eq := hP12Eq)
      (hP14 := hP14)
      (hP15Vec := hP15Vec)
      (hP15Scal := hP15Scal)
      (hP16Win := hP16Win)
      (hCset := hCset)
      (hSamples := hSamples)
      (hRaw := hRaw)
      (hAddRep := hAddRep)
      (hRawLe := hRawLe)
      (hP18 := hP18)
      (hP19 := hP19)
  exact superneoMathProtocolSkeleton_of_props hShape hBar hA hB hP10 hP20 hWitness hNorm

theorem superneoMathProtocolSkeleton_of_props_with_sampling_operand_assumptions_via_centeredRep_mul_and_add_and_rawCoeff
  {ctx : PSContext} {claim : PSClaim} {witness : PSWitness}
  {BA BB BRaw : Nat}
  (hShape : ClaimShapeValid claim)
  (hBar : IsDBarMatrix ctx.bar)
  (hA : IsDVec claim.a)
  (hB : IsDVec claim.b)
  (hP10 : p10ForClaim ctx claim)
  (hP6 : p20DecompProp claim.zDecomp ctx.bSplit ctx.kSplit)
  (hP12Eq : matrixVecDirect claim.m claim.z = matrixVecCtBar ctx.bar claim.m claim.z)
  (hP14 : p20EvalHomProp ctx.bar claim.m claim.z1 claim.z2 claim.r claim.rho1 claim.rho2)
  (hP15Vec : p20VecModuleProp ctx.hVec claim.rho1 claim.z1 claim.z2)
  (hP15Scal : p20ScalarModuleProp ctx.hScal claim.rho1 claim.z1 claim.z2)
  (hP16Win : p20InvertibilityWindowProp claim.invDelta)
  (hCset : ∀ i : Fin claim.cset.size, normInfCoeffs claim.cset[i] ≤ BA)
  (hSamples : ∀ j : Fin claim.samples.size, normInfCoeffs claim.samples[j] ≤ BB)
  (hRawCoeff : mulRqRawCoeffBoundFromOperands BA BB BRaw)
  (hAddRep :
    ∀ x y : F, Int.natAbs (centeredRep (x + y))
      ≤ Int.natAbs (centeredRep x) + Int.natAbs (centeredRep y))
  (hRawLe : BRaw + BRaw + BRaw ≤ theorem9UpperBound (maxRhoNorm claim.cset))
  (hP18 : p20PolyProp claim.qVals ctx.ell ctx.totalDegree ctx.setSize)
  (hP19 : p20InterpProp claim.xs claim.ys claim.expectedCoeffs claim.evalPoint claim.expectedEval)
  (hWitness : witness.z = claim.z)
  (hNorm : normInfCoeffs witness.z < ctx.ceNormBound) :
  PSCEValid ctx claim witness := by
  exact superneoMathProtocolSkeleton_of_props_with_sampling_operand_assumptions_via_centeredRep_mul_and_add_and_raw
    (hShape := hShape)
    (hBar := hBar)
    (hA := hA)
    (hB := hB)
    (hP10 := hP10)
    (hP6 := hP6)
    (hP12Eq := hP12Eq)
    (hP14 := hP14)
    (hP15Vec := hP15Vec)
    (hP15Scal := hP15Scal)
    (hP16Win := hP16Win)
    (hCset := hCset)
    (hSamples := hSamples)
    (hRaw := mulRqRawNormBoundFromOperands_of_rawCoeff hRawCoeff)
    (hAddRep := hAddRep)
    (hRawLe := hRawLe)
    (hP18 := hP18)
    (hP19 := hP19)
    (hWitness := hWitness)
    (hNorm := hNorm)

/-- Bundle wrapper for centered-representation blockers (raw-bound path). -/
theorem superneoMathProtocolSkeleton_of_props_with_sampling_operand_assumptions_via_centeredRepMulAddBounds_and_raw
  {ctx : PSContext} {claim : PSClaim} {witness : PSWitness}
  {BA BB BRaw : Nat}
  (hShape : ClaimShapeValid claim)
  (hBar : IsDBarMatrix ctx.bar)
  (hA : IsDVec claim.a)
  (hB : IsDVec claim.b)
  (hP10 : p10ForClaim ctx claim)
  (hP6 : p20DecompProp claim.zDecomp ctx.bSplit ctx.kSplit)
  (hP12Eq : matrixVecDirect claim.m claim.z = matrixVecCtBar ctx.bar claim.m claim.z)
  (hP14 : p20EvalHomProp ctx.bar claim.m claim.z1 claim.z2 claim.r claim.rho1 claim.rho2)
  (hP15Vec : p20VecModuleProp ctx.hVec claim.rho1 claim.z1 claim.z2)
  (hP15Scal : p20ScalarModuleProp ctx.hScal claim.rho1 claim.z1 claim.z2)
  (hP16Win : p20InvertibilityWindowProp claim.invDelta)
  (hCset : ∀ i : Fin claim.cset.size, normInfCoeffs claim.cset[i] ≤ BA)
  (hSamples : ∀ j : Fin claim.samples.size, normInfCoeffs claim.samples[j] ≤ BB)
  (hRaw : mulRqRawNormBoundFromOperands BA BB BRaw)
  (hRep : centeredRepMulAddBounds)
  (hRawLe : BRaw + BRaw + BRaw ≤ theorem9UpperBound (maxRhoNorm claim.cset))
  (hP18 : p20PolyProp claim.qVals ctx.ell ctx.totalDegree ctx.setSize)
  (hP19 : p20InterpProp claim.xs claim.ys claim.expectedCoeffs claim.evalPoint claim.expectedEval)
  (hWitness : witness.z = claim.z)
  (hNorm : normInfCoeffs witness.z < ctx.ceNormBound) :
  PSCEValid ctx claim witness := by
  exact superneoMathProtocolSkeleton_of_props_with_sampling_operand_assumptions_via_centeredRep_mul_and_add_and_raw
    (hShape := hShape)
    (hBar := hBar)
    (hA := hA)
    (hB := hB)
    (hP10 := hP10)
    (hP6 := hP6)
    (hP12Eq := hP12Eq)
    (hP14 := hP14)
    (hP15Vec := hP15Vec)
    (hP15Scal := hP15Scal)
    (hP16Win := hP16Win)
    (hCset := hCset)
    (hSamples := hSamples)
    (hRaw := hRaw)
    (hAddRep := centeredRepMulAddBounds_add hRep)
    (hRawLe := hRawLe)
    (hP18 := hP18)
    (hP19 := hP19)
    (hWitness := hWitness)
    (hNorm := hNorm)

/-- Bundle wrapper for centered-representation blockers (raw-coeff path). -/
theorem superneoMathProtocolSkeleton_of_props_with_sampling_operand_assumptions_via_centeredRepMulAddBounds_and_rawCoeff
  {ctx : PSContext} {claim : PSClaim} {witness : PSWitness}
  {BA BB BRaw : Nat}
  (hShape : ClaimShapeValid claim)
  (hBar : IsDBarMatrix ctx.bar)
  (hA : IsDVec claim.a)
  (hB : IsDVec claim.b)
  (hP10 : p10ForClaim ctx claim)
  (hP6 : p20DecompProp claim.zDecomp ctx.bSplit ctx.kSplit)
  (hP12Eq : matrixVecDirect claim.m claim.z = matrixVecCtBar ctx.bar claim.m claim.z)
  (hP14 : p20EvalHomProp ctx.bar claim.m claim.z1 claim.z2 claim.r claim.rho1 claim.rho2)
  (hP15Vec : p20VecModuleProp ctx.hVec claim.rho1 claim.z1 claim.z2)
  (hP15Scal : p20ScalarModuleProp ctx.hScal claim.rho1 claim.z1 claim.z2)
  (hP16Win : p20InvertibilityWindowProp claim.invDelta)
  (hCset : ∀ i : Fin claim.cset.size, normInfCoeffs claim.cset[i] ≤ BA)
  (hSamples : ∀ j : Fin claim.samples.size, normInfCoeffs claim.samples[j] ≤ BB)
  (hRawCoeff : mulRqRawCoeffBoundFromOperands BA BB BRaw)
  (hRep : centeredRepMulAddBounds)
  (hRawLe : BRaw + BRaw + BRaw ≤ theorem9UpperBound (maxRhoNorm claim.cset))
  (hP18 : p20PolyProp claim.qVals ctx.ell ctx.totalDegree ctx.setSize)
  (hP19 : p20InterpProp claim.xs claim.ys claim.expectedCoeffs claim.evalPoint claim.expectedEval)
  (hWitness : witness.z = claim.z)
  (hNorm : normInfCoeffs witness.z < ctx.ceNormBound) :
  PSCEValid ctx claim witness := by
  exact superneoMathProtocolSkeleton_of_props_with_sampling_operand_assumptions_via_centeredRepMulAddBounds_and_raw
    (hShape := hShape)
    (hBar := hBar)
    (hA := hA)
    (hB := hB)
    (hP10 := hP10)
    (hP6 := hP6)
    (hP12Eq := hP12Eq)
    (hP14 := hP14)
    (hP15Vec := hP15Vec)
    (hP15Scal := hP15Scal)
    (hP16Win := hP16Win)
    (hCset := hCset)
    (hSamples := hSamples)
    (hRaw := mulRqRawNormBoundFromOperands_of_rawCoeff hRawCoeff)
    (hRep := hRep)
    (hRawLe := hRawLe)
    (hP18 := hP18)
    (hP19 := hP19)
    (hWitness := hWitness)
    (hNorm := hNorm)

theorem superneoMathProtocolSkeleton_of_props_with_sampling_operand_assumptions_inRange
  {ctx : PSContext} {claim : PSClaim} {witness : PSWitness}
  {BA BB BRaw : Nat}
  (hShape : ClaimShapeValid claim)
  (hBar : IsDBarMatrix ctx.bar)
  (hA : IsDVec claim.a)
  (hB : IsDVec claim.b)
  (hP10 : p10ForClaim ctx claim)
  (hP6 : p20DecompProp claim.zDecomp ctx.bSplit ctx.kSplit)
  (hP12Eq : matrixVecDirect claim.m claim.z = matrixVecCtBar ctx.bar claim.m claim.z)
  (hP14 : p20EvalHomProp ctx.bar claim.m claim.z1 claim.z2 claim.r claim.rho1 claim.rho2)
  (hP15Vec : p20VecModuleProp ctx.hVec claim.rho1 claim.z1 claim.z2)
  (hP15Scal : p20ScalarModuleProp ctx.hScal claim.rho1 claim.z1 claim.z2)
  (hP16Win : p20InvertibilityWindowProp claim.invDelta)
  (hCset : ∀ i : Fin claim.cset.size, normInfCoeffs claim.cset[i] ≤ BA)
  (hSamples : ∀ j : Fin claim.samples.size, normInfCoeffs claim.samples[j] ≤ BB)
  (hRawInRange : mulRqRawInRangeBoundFromOperands BA BB BRaw)
  (hAddSub : rawAddSubCollapseBound BRaw (theorem9UpperBound (maxRhoNorm claim.cset)))
  (hSub : rawSubCollapseBound BRaw (theorem9UpperBound (maxRhoNorm claim.cset)))
  (hP18 : p20PolyProp claim.qVals ctx.ell ctx.totalDegree ctx.setSize)
  (hP19 : p20InterpProp claim.xs claim.ys claim.expectedCoeffs claim.evalPoint claim.expectedEval)
  (hWitness : witness.z = claim.z)
  (hNorm : normInfCoeffs witness.z < ctx.ceNormBound) :
  PSCEValid ctx claim witness := by
  have hP20 : p20ForClaim ctx claim :=
    p20ForClaim_of_props_with_sampling_operand_assumptions_inRange
      (hP6 := hP6)
      (hP12Rows := claimShapeValid_matrixRowsCompatible hShape)
      (hP12Eq := hP12Eq)
      (hP14 := hP14)
      (hP15Vec := hP15Vec)
      (hP15Scal := hP15Scal)
      (hP16Win := hP16Win)
      (hCset := hCset)
      (hSamples := hSamples)
      (hRawInRange := hRawInRange)
      (hAddSub := hAddSub)
      (hSub := hSub)
      (hP18 := hP18)
      (hP19 := hP19)
  exact superneoMathProtocolSkeleton_of_props hShape hBar hA hB hP10 hP20 hWitness hNorm

theorem superneoMathProtocolSkeleton_of_props_with_sampling_operand_rawCoeff_assumptions
  {ctx : PSContext} {claim : PSClaim} {witness : PSWitness}
  {BA BB BRaw : Nat}
  (hShape : ClaimShapeValid claim)
  (hBar : IsDBarMatrix ctx.bar)
  (hA : IsDVec claim.a)
  (hB : IsDVec claim.b)
  (hP10 : p10ForClaim ctx claim)
  (hP6 : p20DecompProp claim.zDecomp ctx.bSplit ctx.kSplit)
  (hP12Eq : matrixVecDirect claim.m claim.z = matrixVecCtBar ctx.bar claim.m claim.z)
  (hP14 : p20EvalHomProp ctx.bar claim.m claim.z1 claim.z2 claim.r claim.rho1 claim.rho2)
  (hP15Vec : p20VecModuleProp ctx.hVec claim.rho1 claim.z1 claim.z2)
  (hP15Scal : p20ScalarModuleProp ctx.hScal claim.rho1 claim.z1 claim.z2)
  (hP16Win : p20InvertibilityWindowProp claim.invDelta)
  (hCset : ∀ i : Fin claim.cset.size, normInfCoeffs claim.cset[i] ≤ BA)
  (hSamples : ∀ j : Fin claim.samples.size, normInfCoeffs claim.samples[j] ≤ BB)
  (hRawCoeff : mulRqRawCoeffBoundFromOperands BA BB BRaw)
  (hAddSub : rawAddSubCollapseBound BRaw (theorem9UpperBound (maxRhoNorm claim.cset)))
  (hSub : rawSubCollapseBound BRaw (theorem9UpperBound (maxRhoNorm claim.cset)))
  (hP18 : p20PolyProp claim.qVals ctx.ell ctx.totalDegree ctx.setSize)
  (hP19 : p20InterpProp claim.xs claim.ys claim.expectedCoeffs claim.evalPoint claim.expectedEval)
  (hWitness : witness.z = claim.z)
  (hNorm : normInfCoeffs witness.z < ctx.ceNormBound) :
  PSCEValid ctx claim witness := by
  have hP20 : p20ForClaim ctx claim :=
    p20ForClaim_of_props_with_sampling_operand_rawCoeff_assumptions
      (hP6 := hP6)
      (hP12Rows := claimShapeValid_matrixRowsCompatible hShape)
      (hP12Eq := hP12Eq)
      (hP14 := hP14)
      (hP15Vec := hP15Vec)
      (hP15Scal := hP15Scal)
      (hP16Win := hP16Win)
      (hCset := hCset)
      (hSamples := hSamples)
      (hRawCoeff := hRawCoeff)
      (hAddSub := hAddSub)
      (hSub := hSub)
      (hP18 := hP18)
      (hP19 := hP19)
  exact superneoMathProtocolSkeleton_of_props hShape hBar hA hB hP10 hP20 hWitness hNorm

theorem superneoMathProtocolSkeleton_of_props_with_evalHom_assumption
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
  PSCEValid ctx claim witness := by
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
  PSCEValid ctx claim witness := by
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

theorem superneoMathProtocolSkeleton_of_props_with_p15EvalBarMzAtAssumption
  {ctx : PSContext} {claim : PSClaim} {witness : PSWitness}
  (hShape : ClaimShapeValid claim)
  (hP14Rows : MatrixRowsCompatible claim.m claim.z1)
  (hBar : IsDBarMatrix ctx.bar)
  (hA : IsDVec claim.a)
  (hB : IsDVec claim.b)
  (hP10 : p10ForClaim ctx claim)
  (hP6 : p20DecompProp claim.zDecomp ctx.bSplit ctx.kSplit)
  (hP12Eq : matrixVecDirect claim.m claim.z = matrixVecCtBar ctx.bar claim.m claim.z)
  (hP14FromP15 : p15EvalBarMzAtAssumption ctx.bar claim.m claim.r)
  (hP15Vec : p20VecModuleProp ctx.hVec claim.rho1 claim.z1 claim.z2)
  (hP15Scal : p20ScalarModuleProp ctx.hScal claim.rho1 claim.z1 claim.z2)
  (hP16Win : p20InvertibilityWindowProp claim.invDelta)
  (hP17 : p20SamplingProp claim.cset claim.samples)
  (hP18 : p20PolyProp claim.qVals ctx.ell ctx.totalDegree ctx.setSize)
  (hP19 : p20InterpProp claim.xs claim.ys claim.expectedCoeffs claim.evalPoint claim.expectedEval)
  (hWitness : witness.z = claim.z)
  (hNorm : normInfCoeffs witness.z < ctx.ceNormBound) :
  PSCEValid ctx claim witness := by
  have hP20 : p20ForClaim ctx claim :=
    p20ForClaim_of_props_with_p15EvalBarMzAtAssumption
      (hP6 := hP6)
      (hP12Rows := claimShapeValid_matrixRowsCompatible hShape)
      (hP12Eq := hP12Eq)
      (hP14FromP15 := hP14FromP15)
      (hP14Size := claimShapeValid_z1_size_eq_z2_size hShape)
      (hP14Rows := hP14Rows)
      (hP15Vec := hP15Vec)
      (hP15Scal := hP15Scal)
      (hP16Win := hP16Win)
      (hP17 := hP17)
      (hP18 := hP18)
      (hP19 := hP19)
  exact superneoMathProtocolSkeleton_of_props hShape hBar hA hB hP10 hP20 hWitness hNorm

theorem superneoMathProtocolSkeleton_of_props_with_p15EvalBarMzAtCheckAssumption
  {ctx : PSContext} {claim : PSClaim} {witness : PSWitness}
  (hShape : ClaimShapeValid claim)
  (hP14Rows : MatrixRowsCompatible claim.m claim.z1)
  (hBar : IsDBarMatrix ctx.bar)
  (hA : IsDVec claim.a)
  (hB : IsDVec claim.b)
  (hP10 : p10ForClaim ctx claim)
  (hP6 : p20DecompProp claim.zDecomp ctx.bSplit ctx.kSplit)
  (hP12Eq : matrixVecDirect claim.m claim.z = matrixVecCtBar ctx.bar claim.m claim.z)
  (hP14CheckFromP15 : p15EvalBarMzAtCheckAssumption ctx.bar claim.m claim.r)
  (hP15Vec : p20VecModuleProp ctx.hVec claim.rho1 claim.z1 claim.z2)
  (hP15Scal : p20ScalarModuleProp ctx.hScal claim.rho1 claim.z1 claim.z2)
  (hP16Win : p20InvertibilityWindowProp claim.invDelta)
  (hP17 : p20SamplingProp claim.cset claim.samples)
  (hP18 : p20PolyProp claim.qVals ctx.ell ctx.totalDegree ctx.setSize)
  (hP19 : p20InterpProp claim.xs claim.ys claim.expectedCoeffs claim.evalPoint claim.expectedEval)
  (hWitness : witness.z = claim.z)
  (hNorm : normInfCoeffs witness.z < ctx.ceNormBound) :
  PSCEValid ctx claim witness := by
  have hP20 : p20ForClaim ctx claim :=
    p20ForClaim_of_props_with_p15EvalBarMzAtCheckAssumption
      (hP6 := hP6)
      (hP12Rows := claimShapeValid_matrixRowsCompatible hShape)
      (hP12Eq := hP12Eq)
      (hP14CheckFromP15 := hP14CheckFromP15)
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
  PSCEValid ctx claim witness := by
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
  PSCEValid ctx claim witness := by
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
  PSCEValid ctx claim witness := by
  exact superneoMathProtocolSkeleton_of_props_with_matrixTransform_assumption_with_evalHom_assumption
    hShape hP12Assm hP14Rows hBar hA hB hP10
    hP6 (p14EvalHomAssumption_of_checkAssumption hP14Check) hP15Vec hP15Scal hP16Win hP17 hP18 hP19
    hWitness hNorm

theorem superneoMathProtocolSkeleton_of_props_with_matrixTransform_checkAssumption_with_evalHom_assumption
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
  PSCEValid ctx claim witness := by
  exact superneoMathProtocolSkeleton_of_props_with_matrixTransform_assumption_with_evalHom_assumption
    hShape (p12MatrixTransformAssumption_of_checkAssumption hP12Check) hP14Rows hBar hA hB hP10
    hP6 hP14Assm hP15Vec hP15Scal hP16Win hP17 hP18 hP19
    hWitness hNorm


end SuperNeo
