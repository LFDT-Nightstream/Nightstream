import SuperNeo.ProtocolRelations

/-! Claim-level protocol reduction constructors and CE-validity bridges. -/


namespace SuperNeo

/-- Claim-level wrapper exposing the P10 core proposition through `ProtocolCtx`/`CEClaim`. -/
def p10ForClaim (ctx : ProtocolCtx) (claim : CEClaim) : Prop :=
  p10CoreProp ctx.bar claim.a claim.b

/-- Claim-level wrapper exposing the full P20 arithmetic bundle obligations. -/
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

/-- Internal helper: derive the P12 matrix-transform equality for this claim from
`thm3CoreAssumption` and `ClaimShapeValid` (no tuple-projection shape extraction). -/
private theorem matrixTransformEq_of_thm3CoreAssumption_of_shape
  {ctx : ProtocolCtx} {claim : CEClaim}
  (hThm3 : thm3CoreAssumption ctx.bar)
  (hShape : ClaimShapeValid claim) :
  matrixVecDirect claim.m claim.z = matrixVecCtBar ctx.bar claim.m claim.z := by
  exact matrixTransformEq_of_thm3CoreAssumption hThm3
    (claimShapeValid_matrixRowsCompatible hShape)

/-- Internal helper: from proposition-level P20 obligations, derive invertibility witness and CE validity. -/
private theorem ceValid_with_invertibility_of_p20
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

theorem p20ForClaim_of_props_with_sampling_goldilocks_operand_assumptions
  {ctx : ProtocolCtx} {claim : CEClaim}
  (hP6 : p20DecompProp claim.zDecomp ctx.bSplit ctx.kSplit)
  (hP12Rows : MatrixRowsCompatible claim.m claim.z)
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
  (hP19 : p20InterpProp claim.xs claim.ys claim.expectedCoeffs claim.evalPoint claim.expectedEval) :
  p20ForClaim ctx claim := by
  exact p20ArithmeticBundle_of_props_with_sampling_operand_assumptions
    (BA := Parameters.Goldilocks.B)
    (BB := Parameters.Goldilocks.B)
    (BRaw := Parameters.Goldilocks.B)
    (hP6 := hP6)
    (hP12Rows := hP12Rows)
    (hP12Eq := hP12Eq)
    (hP14 := hP14)
    (hP15Vec := hP15Vec)
    (hP15Scal := hP15Scal)
    (hP16 := invertibilityPreconditions_from_constants)
    (hP16Win := hP16Win)
    (hCset := hCset)
    (hSamples := hSamples)
    (hRaw := hRaw)
    (hAddSub := rawAddSubCollapseBound_mono hCollapse.1 hUpper)
    (hSub := rawSubCollapseBound_mono hCollapse.2 hUpper)
    (hP18 := hP18)
    (hP19 := hP19)

theorem p20ForClaim_of_props_with_sampling_goldilocks_operand_assumptions_inRange
  {ctx : ProtocolCtx} {claim : CEClaim}
  (hP6 : p20DecompProp claim.zDecomp ctx.bSplit ctx.kSplit)
  (hP12Rows : MatrixRowsCompatible claim.m claim.z)
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
  (hP19 : p20InterpProp claim.xs claim.ys claim.expectedCoeffs claim.evalPoint claim.expectedEval) :
  p20ForClaim ctx claim := by
  exact p20ForClaim_of_props_with_sampling_goldilocks_operand_assumptions
    (hP6 := hP6)
    (hP12Rows := hP12Rows)
    (hP12Eq := hP12Eq)
    (hP14 := hP14)
    (hP15Vec := hP15Vec)
    (hP15Scal := hP15Scal)
    (hP16Win := hP16Win)
    (hCset := hCset)
    (hSamples := hSamples)
    (hRaw := goldilocksRawNormBoundAssumption_of_inRange hRawInRange)
    (hCollapse := hCollapse)
    (hUpper := hUpper)
    (hP18 := hP18)
    (hP19 := hP19)

theorem p20ForClaim_of_props_with_sampling_goldilocks_operand_fieldOp_assumptions
  {ctx : ProtocolCtx} {claim : CEClaim}
  (hP6 : p20DecompProp claim.zDecomp ctx.bSplit ctx.kSplit)
  (hP12Rows : MatrixRowsCompatible claim.m claim.z)
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
  (hP19 : p20InterpProp claim.xs claim.ys claim.expectedCoeffs claim.evalPoint claim.expectedEval) :
  p20ForClaim ctx claim := by
  exact p20ForClaim_of_props_with_sampling_goldilocks_operand_assumptions
    (hP6 := hP6)
    (hP12Rows := hP12Rows)
    (hP12Eq := hP12Eq)
    (hP14 := hP14)
    (hP15Vec := hP15Vec)
    (hP15Scal := hP15Scal)
    (hP16Win := hP16Win)
    (hCset := hCset)
    (hSamples := hSamples)
    (hRaw := hRaw)
    (hCollapse := goldilocksRawCollapseAssumption_of_fieldOp hFieldOps)
    (hUpper := hUpper)
    (hP18 := hP18)
    (hP19 := hP19)

theorem p20ForClaim_of_props_with_sampling_goldilocks_operand_fieldOp_assumptions_inRange
  {ctx : ProtocolCtx} {claim : CEClaim}
  (hP6 : p20DecompProp claim.zDecomp ctx.bSplit ctx.kSplit)
  (hP12Rows : MatrixRowsCompatible claim.m claim.z)
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
  (hP19 : p20InterpProp claim.xs claim.ys claim.expectedCoeffs claim.evalPoint claim.expectedEval) :
  p20ForClaim ctx claim := by
  exact p20ForClaim_of_props_with_sampling_goldilocks_operand_fieldOp_assumptions
    (hP6 := hP6)
    (hP12Rows := hP12Rows)
    (hP12Eq := hP12Eq)
    (hP14 := hP14)
    (hP15Vec := hP15Vec)
    (hP15Scal := hP15Scal)
    (hP16Win := hP16Win)
    (hCset := hCset)
    (hSamples := hSamples)
    (hRaw := goldilocksRawNormBoundAssumption_of_inRange hRawInRange)
    (hFieldOps := hFieldOps)
    (hUpper := hUpper)
    (hP18 := hP18)
    (hP19 := hP19)

theorem p20ForClaim_of_props_with_sampling_goldilocks_operand_rawCoeff_assumptions
  {ctx : ProtocolCtx} {claim : CEClaim}
  (hP6 : p20DecompProp claim.zDecomp ctx.bSplit ctx.kSplit)
  (hP12Rows : MatrixRowsCompatible claim.m claim.z)
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
  (hP19 : p20InterpProp claim.xs claim.ys claim.expectedCoeffs claim.evalPoint claim.expectedEval) :
  p20ForClaim ctx claim := by
  exact p20ForClaim_of_props_with_sampling_goldilocks_operand_assumptions
    (hP6 := hP6)
    (hP12Rows := hP12Rows)
    (hP12Eq := hP12Eq)
    (hP14 := hP14)
    (hP15Vec := hP15Vec)
    (hP15Scal := hP15Scal)
    (hP16Win := hP16Win)
    (hCset := hCset)
    (hSamples := hSamples)
    (hRaw := goldilocksRawNormBoundAssumption_of_rawCoeff hRawCoeff)
    (hCollapse := hCollapse)
    (hUpper := hUpper)
    (hP18 := hP18)
    (hP19 := hP19)

theorem p20ForClaim_of_props_with_sampling_goldilocks_operand_rawCoeff_fieldOp_assumptions
  {ctx : ProtocolCtx} {claim : CEClaim}
  (hP6 : p20DecompProp claim.zDecomp ctx.bSplit ctx.kSplit)
  (hP12Rows : MatrixRowsCompatible claim.m claim.z)
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
  (hP19 : p20InterpProp claim.xs claim.ys claim.expectedCoeffs claim.evalPoint claim.expectedEval) :
  p20ForClaim ctx claim := by
  exact p20ForClaim_of_props_with_sampling_goldilocks_operand_fieldOp_assumptions
    (hP6 := hP6)
    (hP12Rows := hP12Rows)
    (hP12Eq := hP12Eq)
    (hP14 := hP14)
    (hP15Vec := hP15Vec)
    (hP15Scal := hP15Scal)
    (hP16Win := hP16Win)
    (hCset := hCset)
    (hSamples := hSamples)
    (hRaw := goldilocksRawNormBoundAssumption_of_rawCoeff hRawCoeff)
    (hFieldOps := hFieldOps)
    (hUpper := hUpper)
    (hP18 := hP18)
    (hP19 := hP19)

theorem p20ForClaim_of_props_with_sampling_operand_assumptions
  {ctx : ProtocolCtx} {claim : CEClaim}
  {BA BB BRaw : Nat}
  (hP6 : p20DecompProp claim.zDecomp ctx.bSplit ctx.kSplit)
  (hP12Rows : MatrixRowsCompatible claim.m claim.z)
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
  (hP19 : p20InterpProp claim.xs claim.ys claim.expectedCoeffs claim.evalPoint claim.expectedEval) :
  p20ForClaim ctx claim := by
  exact p20ArithmeticBundle_of_props_with_sampling_operand_assumptions
    (hP6 := hP6)
    (hP12Rows := hP12Rows)
    (hP12Eq := hP12Eq)
    (hP14 := hP14)
    (hP15Vec := hP15Vec)
    (hP15Scal := hP15Scal)
    (hP16 := invertibilityPreconditions_from_constants)
    (hP16Win := hP16Win)
    (hCset := hCset)
    (hSamples := hSamples)
    (hRaw := hRaw)
    (hAddSub := hAddSub)
    (hSub := hSub)
    (hP18 := hP18)
    (hP19 := hP19)

/--
Assumption-free native constructor for the sampling-bound leg of `p20ForClaim`.
Specializes to the non-tight native P5 path.
-/
theorem p20ForClaim_of_props_with_sampling_operand_assumptions_native
  {ctx : ProtocolCtx} {claim : CEClaim}
  {BA BB : Nat}
  (hP6 : p20DecompProp claim.zDecomp ctx.bSplit ctx.kSplit)
  (hP12Rows : MatrixRowsCompatible claim.m claim.z)
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
  (hP19 : p20InterpProp claim.xs claim.ys claim.expectedCoeffs claim.evalPoint claim.expectedEval) :
  p20ForClaim ctx claim := by
  exact p20ArithmeticBundle_of_props_with_sampling_operand_assumptions_native
    (hP6 := hP6)
    (hP12Rows := hP12Rows)
    (hP12Eq := hP12Eq)
    (hP14 := hP14)
    (hP15Vec := hP15Vec)
    (hP15Scal := hP15Scal)
    (hP16 := invertibilityPreconditions_from_constants)
    (hP16Win := hP16Win)
    (hCset := hCset)
    (hSamples := hSamples)
    (hRawLe := hRawLe)
    (hP18 := hP18)
    (hP19 := hP19)

/--
Assumption-free native-tight constructor for the sampling-bound leg of `p20ForClaim`.
Specializes to the tight native P5 path.
-/
theorem p20ForClaim_of_props_with_sampling_operand_assumptions_native_tight
  {ctx : ProtocolCtx} {claim : CEClaim}
  {BA BB : Nat}
  (hP6 : p20DecompProp claim.zDecomp ctx.bSplit ctx.kSplit)
  (hP12Rows : MatrixRowsCompatible claim.m claim.z)
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
  (hP19 : p20InterpProp claim.xs claim.ys claim.expectedCoeffs claim.evalPoint claim.expectedEval) :
  p20ForClaim ctx claim := by
  exact p20ArithmeticBundle_of_props_with_sampling_operand_assumptions_native_tight
    (hP6 := hP6)
    (hP12Rows := hP12Rows)
    (hP12Eq := hP12Eq)
    (hP14 := hP14)
    (hP15Vec := hP15Vec)
    (hP15Scal := hP15Scal)
    (hP16 := invertibilityPreconditions_from_constants)
    (hP16Win := hP16Win)
    (hCset := hCset)
    (hSamples := hSamples)
    (hRawLe := hRawLe)
    (hP18 := hP18)
    (hP19 := hP19)

theorem p20ForClaim_of_props_with_sampling_operand_assumptions_via_universal_blockers
  {ctx : ProtocolCtx} {claim : CEClaim}
  {BA BB : Nat}
  (hP6 : p20DecompProp claim.zDecomp ctx.bSplit ctx.kSplit)
  (hP12Rows : MatrixRowsCompatible claim.m claim.z)
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
  (hP19 : p20InterpProp claim.xs claim.ys claim.expectedCoeffs claim.evalPoint claim.expectedEval) :
  p20ForClaim ctx claim := by
  exact p20ArithmeticBundle_of_props_with_sampling_operand_assumptions_via_universal_blockers
    (hP6 := hP6)
    (hP12Rows := hP12Rows)
    (hP12Eq := hP12Eq)
    (hP14 := hP14)
    (hP15Vec := hP15Vec)
    (hP15Scal := hP15Scal)
    (hP16 := invertibilityPreconditions_from_constants)
    (hP16Win := hP16Win)
    (hCset := hCset)
    (hSamples := hSamples)
    (hMulUniv := hMulUniv)
    (hAddTri := hAddTri)
    (hSubTri := hSubTri)
    (hRawLe := hRawLe)
    (hP18 := hP18)
    (hP19 := hP19)

theorem p20ForClaim_of_props_with_sampling_operand_assumptions_via_universal_blockers_tight
  {ctx : ProtocolCtx} {claim : CEClaim}
  {BA BB : Nat}
  (hP6 : p20DecompProp claim.zDecomp ctx.bSplit ctx.kSplit)
  (hP12Rows : MatrixRowsCompatible claim.m claim.z)
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
  (hP19 : p20InterpProp claim.xs claim.ys claim.expectedCoeffs claim.evalPoint claim.expectedEval) :
  p20ForClaim ctx claim := by
  exact p20ArithmeticBundle_of_props_with_sampling_operand_assumptions_via_universal_blockers_tight
    (hP6 := hP6)
    (hP12Rows := hP12Rows)
    (hP12Eq := hP12Eq)
    (hP14 := hP14)
    (hP15Vec := hP15Vec)
    (hP15Scal := hP15Scal)
    (hP16 := invertibilityPreconditions_from_constants)
    (hP16Win := hP16Win)
    (hCset := hCset)
    (hSamples := hSamples)
    (hMulUniv := hMulUniv)
    (hAddTri := hAddTri)
    (hSubTri := hSubTri)
    (hRawLe := hRawLe)
    (hP18 := hP18)
    (hP19 := hP19)

/--
Triangle-bundle variant of
`p20ForClaim_of_props_with_sampling_operand_assumptions_via_universal_blockers_tight`.
-/
theorem p20ForClaim_of_props_with_sampling_operand_assumptions_via_universal_mul_and_triangles_tight
  {ctx : ProtocolCtx} {claim : CEClaim}
  {BA BB : Nat}
  (hP6 : p20DecompProp claim.zDecomp ctx.bSplit ctx.kSplit)
  (hP12Rows : MatrixRowsCompatible claim.m claim.z)
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
  (hP19 : p20InterpProp claim.xs claim.ys claim.expectedCoeffs claim.evalPoint claim.expectedEval) :
  p20ForClaim ctx claim := by
  exact p20ArithmeticBundle_of_props_with_sampling_operand_assumptions_via_universal_mul_and_triangles_tight
    (hP6 := hP6)
    (hP12Rows := hP12Rows)
    (hP12Eq := hP12Eq)
    (hP14 := hP14)
    (hP15Vec := hP15Vec)
    (hP15Scal := hP15Scal)
    (hP16 := invertibilityPreconditions_from_constants)
    (hP16Win := hP16Win)
    (hCset := hCset)
    (hSamples := hSamples)
    (hMulUniv := hMulUniv)
    (hTri := hTri)
    (hRawLe := hRawLe)
    (hP18 := hP18)
    (hP19 := hP19)

/--
Add-only variant of
`p20ForClaim_of_props_with_sampling_operand_assumptions_via_universal_mul_and_triangles_tight`.
-/
theorem p20ForClaim_of_props_with_sampling_operand_assumptions_via_universal_mul_and_add_tight
  {ctx : ProtocolCtx} {claim : CEClaim}
  {BA BB : Nat}
  (hP6 : p20DecompProp claim.zDecomp ctx.bSplit ctx.kSplit)
  (hP12Rows : MatrixRowsCompatible claim.m claim.z)
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
  (hP19 : p20InterpProp claim.xs claim.ys claim.expectedCoeffs claim.evalPoint claim.expectedEval) :
  p20ForClaim ctx claim := by
  exact p20ArithmeticBundle_of_props_with_sampling_operand_assumptions_via_universal_mul_and_add_tight
    (hP6 := hP6)
    (hP12Rows := hP12Rows)
    (hP12Eq := hP12Eq)
    (hP14 := hP14)
    (hP15Vec := hP15Vec)
    (hP15Scal := hP15Scal)
    (hP16 := invertibilityPreconditions_from_constants)
    (hP16Win := hP16Win)
    (hCset := hCset)
    (hSamples := hSamples)
    (hMulUniv := hMulUniv)
    (hAddTri := hAddTri)
    (hRawLe := hRawLe)
    (hP18 := hP18)
    (hP19 := hP19)

theorem p20ForClaim_of_props_with_sampling_operand_assumptions_via_centeredRep_mul_and_add_tight
  {ctx : ProtocolCtx} {claim : CEClaim}
  {BA BB : Nat}
  (hP6 : p20DecompProp claim.zDecomp ctx.bSplit ctx.kSplit)
  (hP12Rows : MatrixRowsCompatible claim.m claim.z)
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
  (hP19 : p20InterpProp claim.xs claim.ys claim.expectedCoeffs claim.evalPoint claim.expectedEval) :
  p20ForClaim ctx claim := by
  exact p20ArithmeticBundle_of_props_with_sampling_operand_assumptions_via_centeredRep_mul_and_add_tight
    (hP6 := hP6)
    (hP12Rows := hP12Rows)
    (hP12Eq := hP12Eq)
    (hP14 := hP14)
    (hP15Vec := hP15Vec)
    (hP15Scal := hP15Scal)
    (hP16 := invertibilityPreconditions_from_constants)
    (hP16Win := hP16Win)
    (hCset := hCset)
    (hSamples := hSamples)
    (hMulRep := hMulRep)
    (hAddRep := hAddRep)
    (hRawLe := hRawLe)
    (hP18 := hP18)
    (hP19 := hP19)

theorem p20ForClaim_of_props_with_sampling_operand_assumptions_via_centeredRep_mul_tight
  {ctx : ProtocolCtx} {claim : CEClaim}
  {BA BB : Nat}
  (hP6 : p20DecompProp claim.zDecomp ctx.bSplit ctx.kSplit)
  (hP12Rows : MatrixRowsCompatible claim.m claim.z)
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
  (hP19 : p20InterpProp claim.xs claim.ys claim.expectedCoeffs claim.evalPoint claim.expectedEval) :
  p20ForClaim ctx claim := by
  exact p20ForClaim_of_props_with_sampling_operand_assumptions_via_centeredRep_mul_and_add_tight
    (hP6 := hP6)
    (hP12Rows := hP12Rows)
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

/-- Bundle wrapper for centered-representation mul/add blockers (tight path). -/
theorem p20ForClaim_of_props_with_sampling_operand_assumptions_via_centeredRepMulAddBounds_tight
  {ctx : ProtocolCtx} {claim : CEClaim}
  {BA BB : Nat}
  (hP6 : p20DecompProp claim.zDecomp ctx.bSplit ctx.kSplit)
  (hP12Rows : MatrixRowsCompatible claim.m claim.z)
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
  (hP19 : p20InterpProp claim.xs claim.ys claim.expectedCoeffs claim.evalPoint claim.expectedEval) :
  p20ForClaim ctx claim := by
  exact p20ForClaim_of_props_with_sampling_operand_assumptions_via_centeredRep_mul_and_add_tight
    (hP6 := hP6)
    (hP12Rows := hP12Rows)
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

theorem p20ForClaim_of_props_with_sampling_operand_assumptions_via_universal_blockers_and_raw
  {ctx : ProtocolCtx} {claim : CEClaim}
  {BA BB BRaw : Nat}
  (hP6 : p20DecompProp claim.zDecomp ctx.bSplit ctx.kSplit)
  (hP12Rows : MatrixRowsCompatible claim.m claim.z)
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
  (hP19 : p20InterpProp claim.xs claim.ys claim.expectedCoeffs claim.evalPoint claim.expectedEval) :
  p20ForClaim ctx claim := by
  exact p20ArithmeticBundle_of_props_with_sampling_operand_assumptions_via_universal_blockers_and_raw
    (hP6 := hP6)
    (hP12Rows := hP12Rows)
    (hP12Eq := hP12Eq)
    (hP14 := hP14)
    (hP15Vec := hP15Vec)
    (hP15Scal := hP15Scal)
    (hP16 := invertibilityPreconditions_from_constants)
    (hP16Win := hP16Win)
    (hCset := hCset)
    (hSamples := hSamples)
    (hRaw := hRaw)
    (hAddTri := hAddTri)
    (hSubTri := hSubTri)
    (hRawLe := hRawLe)
    (hP18 := hP18)
    (hP19 := hP19)

theorem p20ForClaim_of_props_with_sampling_operand_assumptions_via_universal_blockers_and_rawCoeff
  {ctx : ProtocolCtx} {claim : CEClaim}
  {BA BB BRaw : Nat}
  (hP6 : p20DecompProp claim.zDecomp ctx.bSplit ctx.kSplit)
  (hP12Rows : MatrixRowsCompatible claim.m claim.z)
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
  (hP19 : p20InterpProp claim.xs claim.ys claim.expectedCoeffs claim.evalPoint claim.expectedEval) :
  p20ForClaim ctx claim := by
  exact p20ForClaim_of_props_with_sampling_operand_assumptions_via_universal_blockers_and_raw
    (hP6 := hP6)
    (hP12Rows := hP12Rows)
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

theorem p20ForClaim_of_props_with_sampling_operand_assumptions_via_universal_mul_and_add_and_raw
  {ctx : ProtocolCtx} {claim : CEClaim}
  {BA BB BRaw : Nat}
  (hP6 : p20DecompProp claim.zDecomp ctx.bSplit ctx.kSplit)
  (hP12Rows : MatrixRowsCompatible claim.m claim.z)
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
  (hP19 : p20InterpProp claim.xs claim.ys claim.expectedCoeffs claim.evalPoint claim.expectedEval) :
  p20ForClaim ctx claim := by
  exact p20ArithmeticBundle_of_props_with_sampling_operand_assumptions_via_universal_mul_and_add_and_raw
    (hP6 := hP6)
    (hP12Rows := hP12Rows)
    (hP12Eq := hP12Eq)
    (hP14 := hP14)
    (hP15Vec := hP15Vec)
    (hP15Scal := hP15Scal)
    (hP16 := invertibilityPreconditions_from_constants)
    (hP16Win := hP16Win)
    (hCset := hCset)
    (hSamples := hSamples)
    (hRaw := hRaw)
    (hAddTri := hAddTri)
    (hRawLe := hRawLe)
    (hP18 := hP18)
    (hP19 := hP19)

theorem p20ForClaim_of_props_with_sampling_operand_assumptions_via_universal_mul_and_add_and_rawCoeff
  {ctx : ProtocolCtx} {claim : CEClaim}
  {BA BB BRaw : Nat}
  (hP6 : p20DecompProp claim.zDecomp ctx.bSplit ctx.kSplit)
  (hP12Rows : MatrixRowsCompatible claim.m claim.z)
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
  (hP19 : p20InterpProp claim.xs claim.ys claim.expectedCoeffs claim.evalPoint claim.expectedEval) :
  p20ForClaim ctx claim := by
  exact p20ForClaim_of_props_with_sampling_operand_assumptions_via_universal_mul_and_add_and_raw
    (hP6 := hP6)
    (hP12Rows := hP12Rows)
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

theorem p20ForClaim_of_props_with_sampling_operand_assumptions_via_centeredRep_mul_and_add_and_raw
  {ctx : ProtocolCtx} {claim : CEClaim}
  {BA BB BRaw : Nat}
  (hP6 : p20DecompProp claim.zDecomp ctx.bSplit ctx.kSplit)
  (hP12Rows : MatrixRowsCompatible claim.m claim.z)
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
  (hP19 : p20InterpProp claim.xs claim.ys claim.expectedCoeffs claim.evalPoint claim.expectedEval) :
  p20ForClaim ctx claim := by
  exact p20ArithmeticBundle_of_props_with_sampling_operand_assumptions_via_centeredRep_mul_and_add_and_raw
    (hP6 := hP6)
    (hP12Rows := hP12Rows)
    (hP12Eq := hP12Eq)
    (hP14 := hP14)
    (hP15Vec := hP15Vec)
    (hP15Scal := hP15Scal)
    (hP16 := invertibilityPreconditions_from_constants)
    (hP16Win := hP16Win)
    (hCset := hCset)
    (hSamples := hSamples)
    (hRaw := hRaw)
    (hAddRep := hAddRep)
    (hRawLe := hRawLe)
    (hP18 := hP18)
    (hP19 := hP19)

theorem p20ForClaim_of_props_with_sampling_operand_assumptions_via_centeredRep_mul_and_add_and_rawCoeff
  {ctx : ProtocolCtx} {claim : CEClaim}
  {BA BB BRaw : Nat}
  (hP6 : p20DecompProp claim.zDecomp ctx.bSplit ctx.kSplit)
  (hP12Rows : MatrixRowsCompatible claim.m claim.z)
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
  (hP19 : p20InterpProp claim.xs claim.ys claim.expectedCoeffs claim.evalPoint claim.expectedEval) :
  p20ForClaim ctx claim := by
  exact p20ForClaim_of_props_with_sampling_operand_assumptions_via_centeredRep_mul_and_add_and_raw
    (hP6 := hP6)
    (hP12Rows := hP12Rows)
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

/-- Bundle wrapper for centered-representation blockers (raw-bound path). -/
theorem p20ForClaim_of_props_with_sampling_operand_assumptions_via_centeredRepMulAddBounds_and_raw
  {ctx : ProtocolCtx} {claim : CEClaim}
  {BA BB BRaw : Nat}
  (hP6 : p20DecompProp claim.zDecomp ctx.bSplit ctx.kSplit)
  (hP12Rows : MatrixRowsCompatible claim.m claim.z)
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
  (hP19 : p20InterpProp claim.xs claim.ys claim.expectedCoeffs claim.evalPoint claim.expectedEval) :
  p20ForClaim ctx claim := by
  exact p20ForClaim_of_props_with_sampling_operand_assumptions_via_centeredRep_mul_and_add_and_raw
    (hP6 := hP6)
    (hP12Rows := hP12Rows)
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

/-- Bundle wrapper for centered-representation blockers (raw-coeff path). -/
theorem p20ForClaim_of_props_with_sampling_operand_assumptions_via_centeredRepMulAddBounds_and_rawCoeff
  {ctx : ProtocolCtx} {claim : CEClaim}
  {BA BB BRaw : Nat}
  (hP6 : p20DecompProp claim.zDecomp ctx.bSplit ctx.kSplit)
  (hP12Rows : MatrixRowsCompatible claim.m claim.z)
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
  (hP19 : p20InterpProp claim.xs claim.ys claim.expectedCoeffs claim.evalPoint claim.expectedEval) :
  p20ForClaim ctx claim := by
  exact p20ForClaim_of_props_with_sampling_operand_assumptions_via_centeredRepMulAddBounds_and_raw
    (hP6 := hP6)
    (hP12Rows := hP12Rows)
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

theorem p20ForClaim_of_props_with_sampling_operand_assumptions_inRange
  {ctx : ProtocolCtx} {claim : CEClaim}
  {BA BB BRaw : Nat}
  (hP6 : p20DecompProp claim.zDecomp ctx.bSplit ctx.kSplit)
  (hP12Rows : MatrixRowsCompatible claim.m claim.z)
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
  (hP19 : p20InterpProp claim.xs claim.ys claim.expectedCoeffs claim.evalPoint claim.expectedEval) :
  p20ForClaim ctx claim := by
  exact p20ForClaim_of_props_with_sampling_operand_assumptions
    (hP6 := hP6)
    (hP12Rows := hP12Rows)
    (hP12Eq := hP12Eq)
    (hP14 := hP14)
    (hP15Vec := hP15Vec)
    (hP15Scal := hP15Scal)
    (hP16Win := hP16Win)
    (hCset := hCset)
    (hSamples := hSamples)
    (hRaw := mulRqRawNormBoundFromOperands_of_inRange hRawInRange)
    (hAddSub := hAddSub)
    (hSub := hSub)
    (hP18 := hP18)
    (hP19 := hP19)

theorem p20ForClaim_of_props_with_sampling_operand_rawCoeff_assumptions
  {ctx : ProtocolCtx} {claim : CEClaim}
  {BA BB BRaw : Nat}
  (hP6 : p20DecompProp claim.zDecomp ctx.bSplit ctx.kSplit)
  (hP12Rows : MatrixRowsCompatible claim.m claim.z)
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
  (hP19 : p20InterpProp claim.xs claim.ys claim.expectedCoeffs claim.evalPoint claim.expectedEval) :
  p20ForClaim ctx claim := by
  exact p20ForClaim_of_props_with_sampling_operand_assumptions
    (hP6 := hP6)
    (hP12Rows := hP12Rows)
    (hP12Eq := hP12Eq)
    (hP14 := hP14)
    (hP15Vec := hP15Vec)
    (hP15Scal := hP15Scal)
    (hP16Win := hP16Win)
    (hCset := hCset)
    (hSamples := hSamples)
    (hRaw := mulRqRawNormBoundFromOperands_of_rawCoeff hRawCoeff)
    (hAddSub := hAddSub)
    (hSub := hSub)
    (hP18 := hP18)
    (hP19 := hP19)

theorem p20ForClaim_of_props_with_sampling_operand_fieldOp_assumptions
  {ctx : ProtocolCtx} {claim : CEClaim}
  {BA BB : Nat}
  (hP6 : p20DecompProp claim.zDecomp ctx.bSplit ctx.kSplit)
  (hP12Rows : MatrixRowsCompatible claim.m claim.z)
  (hP12Eq : matrixVecDirect claim.m claim.z = matrixVecCtBar ctx.bar claim.m claim.z)
  (hP14 : p20EvalHomProp ctx.bar claim.m claim.z1 claim.z2 claim.r claim.rho1 claim.rho2)
  (hP15Vec : p20VecModuleProp ctx.hVec claim.rho1 claim.z1 claim.z2)
  (hP15Scal : p20ScalarModuleProp ctx.hScal claim.rho1 claim.z1 claim.z2)
  (hP16Win : p20InvertibilityWindowProp claim.invDelta)
  (hCset : ∀ i : Fin claim.cset.size, normInfCoeffs claim.cset[i] ≤ BA)
  (hSamples : ∀ j : Fin claim.samples.size, normInfCoeffs claim.samples[j] ≤ BB)
  (hRaw : mulRqRawNormBoundFromOperands BA BB (theorem9UpperBound (maxRhoNorm claim.cset)))
  (hOps :
    rawFieldOpCollapseBound
      (theorem9UpperBound (maxRhoNorm claim.cset))
      (theorem9UpperBound (maxRhoNorm claim.cset)))
  (hP18 : p20PolyProp claim.qVals ctx.ell ctx.totalDegree ctx.setSize)
  (hP19 : p20InterpProp claim.xs claim.ys claim.expectedCoeffs claim.evalPoint claim.expectedEval) :
  p20ForClaim ctx claim := by
  exact p20ForClaim_of_props_with_sampling_operand_assumptions
    (BRaw := theorem9UpperBound (maxRhoNorm claim.cset))
    (hP6 := hP6)
    (hP12Rows := hP12Rows)
    (hP12Eq := hP12Eq)
    (hP14 := hP14)
    (hP15Vec := hP15Vec)
    (hP15Scal := hP15Scal)
    (hP16Win := hP16Win)
    (hCset := hCset)
    (hSamples := hSamples)
    (hRaw := hRaw)
    (hAddSub := rawAddSubCollapseBound_of_add_and_sub_same hOps.1 hOps.2)
    (hSub := hOps.2)
    (hP18 := hP18)
    (hP19 := hP19)

theorem p20ForClaim_of_props_with_sampling_operand_fieldOp_assumptions_inRange
  {ctx : ProtocolCtx} {claim : CEClaim}
  {BA BB : Nat}
  (hP6 : p20DecompProp claim.zDecomp ctx.bSplit ctx.kSplit)
  (hP12Rows : MatrixRowsCompatible claim.m claim.z)
  (hP12Eq : matrixVecDirect claim.m claim.z = matrixVecCtBar ctx.bar claim.m claim.z)
  (hP14 : p20EvalHomProp ctx.bar claim.m claim.z1 claim.z2 claim.r claim.rho1 claim.rho2)
  (hP15Vec : p20VecModuleProp ctx.hVec claim.rho1 claim.z1 claim.z2)
  (hP15Scal : p20ScalarModuleProp ctx.hScal claim.rho1 claim.z1 claim.z2)
  (hP16Win : p20InvertibilityWindowProp claim.invDelta)
  (hCset : ∀ i : Fin claim.cset.size, normInfCoeffs claim.cset[i] ≤ BA)
  (hSamples : ∀ j : Fin claim.samples.size, normInfCoeffs claim.samples[j] ≤ BB)
  (hRawInRange :
    mulRqRawInRangeBoundFromOperands BA BB (theorem9UpperBound (maxRhoNorm claim.cset)))
  (hOps :
    rawFieldOpCollapseBound
      (theorem9UpperBound (maxRhoNorm claim.cset))
      (theorem9UpperBound (maxRhoNorm claim.cset)))
  (hP18 : p20PolyProp claim.qVals ctx.ell ctx.totalDegree ctx.setSize)
  (hP19 : p20InterpProp claim.xs claim.ys claim.expectedCoeffs claim.evalPoint claim.expectedEval) :
  p20ForClaim ctx claim := by
  exact p20ForClaim_of_props_with_sampling_operand_fieldOp_assumptions
    (hP6 := hP6)
    (hP12Rows := hP12Rows)
    (hP12Eq := hP12Eq)
    (hP14 := hP14)
    (hP15Vec := hP15Vec)
    (hP15Scal := hP15Scal)
    (hP16Win := hP16Win)
    (hCset := hCset)
    (hSamples := hSamples)
    (hRaw := mulRqRawNormBoundFromOperands_of_inRange hRawInRange)
    (hOps := hOps)
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
  exact p20ArithmeticBundle_of_assumptions
    (hP6 := hP6)
    (hP12Rows := claimShapeValid_matrixRowsCompatible hShape)
    (hP12Assm := p12MatrixTransformAssumption_of_thm3CoreAssumption hThm3)
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
  exact p20ArithmeticBundle_of_assumptions
    (hP6 := hP6)
    (hP12Rows := claimShapeValid_matrixRowsCompatible hShape)
    (hP12Assm := p12MatrixTransformAssumption_of_thm3CoreAssumption hThm3)
    (hP14Assm := p14EvalHomAssumption_of_checkAssumption hP14Check)
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
  exact p20ArithmeticBundle_of_props_with_evalHom_checkAssumption
    (hP6 := hP6)
    (hP12Rows := hP12Rows)
    (hP12Eq := hP12Eq)
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

theorem p20ForClaim_of_props_with_p15EvalBarMzAtAssumption
  {ctx : ProtocolCtx} {claim : CEClaim}
  (hP6 : p20DecompProp claim.zDecomp ctx.bSplit ctx.kSplit)
  (hP12Rows : MatrixRowsCompatible claim.m claim.z)
  (hP12Eq : matrixVecDirect claim.m claim.z = matrixVecCtBar ctx.bar claim.m claim.z)
  (hP14FromP15 : p15EvalBarMzAtAssumption ctx.bar claim.m claim.r)
  (hP14Size : claim.z1.size = claim.z2.size)
  (hP14Rows : MatrixRowsCompatible claim.m claim.z1)
  (hP15Vec : p20VecModuleProp ctx.hVec claim.rho1 claim.z1 claim.z2)
  (hP15Scal : p20ScalarModuleProp ctx.hScal claim.rho1 claim.z1 claim.z2)
  (hP16Win : p20InvertibilityWindowProp claim.invDelta)
  (hP17 : p20SamplingProp claim.cset claim.samples)
  (hP18 : p20PolyProp claim.qVals ctx.ell ctx.totalDegree ctx.setSize)
  (hP19 : p20InterpProp claim.xs claim.ys claim.expectedCoeffs claim.evalPoint claim.expectedEval) :
  p20ForClaim ctx claim := by
  exact p20ArithmeticBundle_of_props_with_p15EvalBarMzAtAssumption
    (hP6 := hP6)
    (hP12Rows := hP12Rows)
    (hP12Eq := hP12Eq)
    (hP14FromP15 := hP14FromP15)
    (hP14Size := hP14Size)
    (hP14Rows := hP14Rows)
    (hP15Vec := hP15Vec)
    (hP15Scal := hP15Scal)
    (hP16 := invertibilityPreconditions_from_constants)
    (hP16Win := hP16Win)
    (hP17 := hP17)
    (hP18 := hP18)
    (hP19 := hP19)

theorem p20ForClaim_of_props_with_p15EvalBarMzAtCheckAssumption
  {ctx : ProtocolCtx} {claim : CEClaim}
  (hP6 : p20DecompProp claim.zDecomp ctx.bSplit ctx.kSplit)
  (hP12Rows : MatrixRowsCompatible claim.m claim.z)
  (hP12Eq : matrixVecDirect claim.m claim.z = matrixVecCtBar ctx.bar claim.m claim.z)
  (hP14CheckFromP15 : p15EvalBarMzAtCheckAssumption ctx.bar claim.m claim.r)
  (hP14Size : claim.z1.size = claim.z2.size)
  (hP14Rows : MatrixRowsCompatible claim.m claim.z1)
  (hP15Vec : p20VecModuleProp ctx.hVec claim.rho1 claim.z1 claim.z2)
  (hP15Scal : p20ScalarModuleProp ctx.hScal claim.rho1 claim.z1 claim.z2)
  (hP16Win : p20InvertibilityWindowProp claim.invDelta)
  (hP17 : p20SamplingProp claim.cset claim.samples)
  (hP18 : p20PolyProp claim.qVals ctx.ell ctx.totalDegree ctx.setSize)
  (hP19 : p20InterpProp claim.xs claim.ys claim.expectedCoeffs claim.evalPoint claim.expectedEval) :
  p20ForClaim ctx claim := by
  exact p20ArithmeticBundle_of_props_with_p15EvalBarMzAtCheckAssumption
    (hP6 := hP6)
    (hP12Rows := hP12Rows)
    (hP12Eq := hP12Eq)
    (hP14CheckFromP15 := hP14CheckFromP15)
    (hP14Size := hP14Size)
    (hP14Rows := hP14Rows)
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
  exact p20ArithmeticBundle_of_assumptions
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
  exact p20ArithmeticBundle_of_assumptions
    (hP6 := hP6)
    (hP12Rows := hP12Rows)
    (hP12Assm := p12MatrixTransformAssumption_of_checkAssumption hP12Check)
    (hP14Assm := p14EvalHomAssumption_of_checkAssumption hP14Check)
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
  exact p20ArithmeticBundle_of_assumptions
    (hP6 := hP6)
    (hP12Rows := hP12Rows)
    (hP12Assm := hP12Assm)
    (hP14Assm := p14EvalHomAssumption_of_checkAssumption hP14Check)
    (hP14Size := hP14Size)
    (hP14Rows := hP14Rows)
    (hP15Vec := hP15Vec)
    (hP15Scal := hP15Scal)
    (hP16 := invertibilityPreconditions_from_constants)
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
  exact p20ArithmeticBundle_of_assumptions
    (hP6 := hP6)
    (hP12Rows := hP12Rows)
    (hP12Assm := p12MatrixTransformAssumption_of_checkAssumption hP12Check)
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

theorem superneoMathProtocolSkeleton_of_props_with_sampling_goldilocks_operand_fieldOp_assumptions
  {ctx : ProtocolCtx} {claim : CEClaim} {witness : CEWitness}
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
  CEValid ctx claim witness := by
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
  {ctx : ProtocolCtx} {claim : CEClaim} {witness : CEWitness}
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
  CEValid ctx claim witness := by
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
  {ctx : ProtocolCtx} {claim : CEClaim} {witness : CEWitness}
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
  CEValid ctx claim witness := by
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
  {ctx : ProtocolCtx} {claim : CEClaim} {witness : CEWitness}
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
  CEValid ctx claim witness := by
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
  {ctx : ProtocolCtx} {claim : CEClaim} {witness : CEWitness}
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
  CEValid ctx claim witness := by
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
  {ctx : ProtocolCtx} {claim : CEClaim} {witness : CEWitness}
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
  CEValid ctx claim witness := by
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
  {ctx : ProtocolCtx} {claim : CEClaim} {witness : CEWitness}
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
  CEValid ctx claim witness := by
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
  {ctx : ProtocolCtx} {claim : CEClaim} {witness : CEWitness}
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
  CEValid ctx claim witness := by
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
  {ctx : ProtocolCtx} {claim : CEClaim} {witness : CEWitness}
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
  CEValid ctx claim witness := by
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
  {ctx : ProtocolCtx} {claim : CEClaim} {witness : CEWitness}
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
  CEValid ctx claim witness := by
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
  {ctx : ProtocolCtx} {claim : CEClaim} {witness : CEWitness}
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
  CEValid ctx claim witness := by
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
  {ctx : ProtocolCtx} {claim : CEClaim} {witness : CEWitness}
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
  CEValid ctx claim witness := by
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
  {ctx : ProtocolCtx} {claim : CEClaim} {witness : CEWitness}
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
  CEValid ctx claim witness := by
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
  {ctx : ProtocolCtx} {claim : CEClaim} {witness : CEWitness}
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
  CEValid ctx claim witness := by
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
  {ctx : ProtocolCtx} {claim : CEClaim} {witness : CEWitness}
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
  CEValid ctx claim witness := by
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
  {ctx : ProtocolCtx} {claim : CEClaim} {witness : CEWitness}
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
  CEValid ctx claim witness := by
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
  {ctx : ProtocolCtx} {claim : CEClaim} {witness : CEWitness}
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
  CEValid ctx claim witness := by
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
  {ctx : ProtocolCtx} {claim : CEClaim} {witness : CEWitness}
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
  CEValid ctx claim witness := by
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
  {ctx : ProtocolCtx} {claim : CEClaim} {witness : CEWitness}
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
  CEValid ctx claim witness := by
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
  {ctx : ProtocolCtx} {claim : CEClaim} {witness : CEWitness}
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
  CEValid ctx claim witness := by
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
  {ctx : ProtocolCtx} {claim : CEClaim} {witness : CEWitness}
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
  CEValid ctx claim witness := by
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
  {ctx : ProtocolCtx} {claim : CEClaim} {witness : CEWitness}
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
  CEValid ctx claim witness := by
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
  {ctx : ProtocolCtx} {claim : CEClaim} {witness : CEWitness}
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
  CEValid ctx claim witness := by
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
  {ctx : ProtocolCtx} {claim : CEClaim} {witness : CEWitness}
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
  CEValid ctx claim witness := by
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
  {ctx : ProtocolCtx} {claim : CEClaim} {witness : CEWitness}
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
  CEValid ctx claim witness := by
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
  {ctx : ProtocolCtx} {claim : CEClaim} {witness : CEWitness}
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
  CEValid ctx claim witness := by
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

theorem superneoMathProtocolSkeleton_of_props_with_p15EvalBarMzAtAssumption
  {ctx : ProtocolCtx} {claim : CEClaim} {witness : CEWitness}
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
  CEValid ctx claim witness := by
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
  {ctx : ProtocolCtx} {claim : CEClaim} {witness : CEWitness}
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
  CEValid ctx claim witness := by
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
  exact ceValid_with_invertibility_of_p20 hShape hBar hA hB hP10 hP20 hWitness hNorm

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
  have hValid : CEValid ctx claim witness :=
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
  rcases hProps with ⟨hP10, hP20⟩
  exact superneoMathProtocolSkeleton_of_props hShape hBar hA hB hP10 hP20 hWitness hNorm

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

theorem p20ForClaim_invertibilityWitness_mulRq_of_goldilocks_operand_assumptions
  {ctx : ProtocolCtx} {claim : CEClaim} {aDelta bDelta : Coeffs}
  (hP20 : p20ForClaim ctx claim)
  (hDeltaEq : claim.invDelta = mulRq aDelta bDelta)
  (hA : normInfCoeffs aDelta ≤ Parameters.Goldilocks.B)
  (hB : normInfCoeffs bDelta ≤ Parameters.Goldilocks.B)
  (hRaw : GoldilocksRawNormBoundAssumption)
  (hCollapse : GoldilocksRawCollapseAssumption) :
  ∃ deltaInv : Coeffs, mulRq claim.invDelta deltaInv = oneRq := by
  exact p20InvertibilityWitness_mulRq_of_goldilocks_operand_assumptions_of_assumption
    (hInv := ctx.hLowNormInvertibility)
    (hP20 := hP20)
    (hDeltaEq := hDeltaEq)
    (hA := hA)
    (hB := hB)
    (hRaw := hRaw)
    (hCollapse := hCollapse)

theorem p20ForClaim_invertibilityWitness_mulRq_of_operand_norm_assumptions_via_universal_blockers
  {ctx : ProtocolCtx} {claim : CEClaim} {aDelta bDelta : Coeffs}
  {BA BB : Nat}
  (hP20 : p20ForClaim ctx claim)
  (hDeltaEq : claim.invDelta = mulRq aDelta bDelta)
  (hA : normInfCoeffs aDelta ≤ BA)
  (hB : normInfCoeffs bDelta ≤ BB)
  (hMulUniv : schoolbookMulUniversalBound)
  (hAddTri : schoolbookAddTriangleBound)
  (hSubTri : schoolbookSubTriangleBound)
  (hBLt :
    ((D * D) * (BA * BB)) + ((D * D) * (BA * BB)) + ((D * D) * (BA * BB))
      < bInvApprox) :
  ∃ deltaInv : Coeffs, mulRq claim.invDelta deltaInv = oneRq := by
  exact p20InvertibilityWitness_mulRq_of_operand_norm_assumptions_via_universal_blockers_of_assumption
    (hInv := ctx.hLowNormInvertibility)
    (hP20 := hP20)
    (hDeltaEq := hDeltaEq)
    (hA := hA)
    (hB := hB)
    (hMulUniv := hMulUniv)
    (hAddTri := hAddTri)
    (hSubTri := hSubTri)
    (hBLt := hBLt)

theorem p20ForClaim_invertibilityWitness_mulRq_of_operand_norm_assumptions_via_universal_blockers_tight
  {ctx : ProtocolCtx} {claim : CEClaim} {aDelta bDelta : Coeffs}
  {BA BB : Nat}
  (hP20 : p20ForClaim ctx claim)
  (hDeltaEq : claim.invDelta = mulRq aDelta bDelta)
  (hA : normInfCoeffs aDelta ≤ BA)
  (hB : normInfCoeffs bDelta ≤ BB)
  (hMulUniv : schoolbookMulUniversalBound)
  (hAddTri : schoolbookAddTriangleBound)
  (hSubTri : schoolbookSubTriangleBound)
  (hBLt :
    (D * (BA * BB)) + (D * (BA * BB)) + (D * (BA * BB))
      < bInvApprox) :
  ∃ deltaInv : Coeffs, mulRq claim.invDelta deltaInv = oneRq := by
  exact p20InvertibilityWitness_mulRq_of_operand_norm_assumptions_via_universal_blockers_tight_of_assumption
    (hInv := ctx.hLowNormInvertibility)
    (hP20 := hP20)
    (hDeltaEq := hDeltaEq)
    (hA := hA)
    (hB := hB)
    (hMulUniv := hMulUniv)
    (hAddTri := hAddTri)
    (hSubTri := hSubTri)
    (hBLt := hBLt)

/--
Triangle-bundle variant of
`p20ForClaim_invertibilityWitness_mulRq_of_operand_norm_assumptions_via_universal_blockers_tight`.
-/
theorem p20ForClaim_invertibilityWitness_mulRq_of_operand_norm_assumptions_via_universal_mul_and_triangles_tight
  {ctx : ProtocolCtx} {claim : CEClaim} {aDelta bDelta : Coeffs}
  {BA BB : Nat}
  (hP20 : p20ForClaim ctx claim)
  (hDeltaEq : claim.invDelta = mulRq aDelta bDelta)
  (hA : normInfCoeffs aDelta ≤ BA)
  (hB : normInfCoeffs bDelta ≤ BB)
  (hMulUniv : schoolbookMulUniversalBound)
  (hTri : schoolbookTriangleBounds)
  (hBLt :
    (D * (BA * BB)) + (D * (BA * BB)) + (D * (BA * BB))
      < bInvApprox) :
  ∃ deltaInv : Coeffs, mulRq claim.invDelta deltaInv = oneRq := by
  exact p20InvertibilityWitness_mulRq_of_operand_norm_assumptions_via_universal_mul_and_triangles_tight_of_assumption
    (hInv := ctx.hLowNormInvertibility)
    (hP20 := hP20)
    (hDeltaEq := hDeltaEq)
    (hA := hA)
    (hB := hB)
    (hMulUniv := hMulUniv)
    (hTri := hTri)
    (hBLt := hBLt)

/--
Add-only variant of
`p20ForClaim_invertibilityWitness_mulRq_of_operand_norm_assumptions_via_universal_mul_and_triangles_tight`.
-/
theorem p20ForClaim_invertibilityWitness_mulRq_of_operand_norm_assumptions_via_universal_mul_and_add_tight
  {ctx : ProtocolCtx} {claim : CEClaim} {aDelta bDelta : Coeffs}
  {BA BB : Nat}
  (hP20 : p20ForClaim ctx claim)
  (hDeltaEq : claim.invDelta = mulRq aDelta bDelta)
  (hA : normInfCoeffs aDelta ≤ BA)
  (hB : normInfCoeffs bDelta ≤ BB)
  (hMulUniv : schoolbookMulUniversalBound)
  (hAddTri : schoolbookAddTriangleBound)
  (hBLt :
    (D * (BA * BB)) + (D * (BA * BB)) + (D * (BA * BB))
      < bInvApprox) :
  ∃ deltaInv : Coeffs, mulRq claim.invDelta deltaInv = oneRq := by
  exact p20InvertibilityWitness_mulRq_of_operand_norm_assumptions_via_universal_mul_and_add_tight_of_assumption
    (hInv := ctx.hLowNormInvertibility)
    (hP20 := hP20)
    (hDeltaEq := hDeltaEq)
    (hA := hA)
    (hB := hB)
    (hMulUniv := hMulUniv)
    (hAddTri := hAddTri)
    (hBLt := hBLt)

theorem p20ForClaim_invertibilityWitness_mulRq_of_operand_norm_assumptions_via_centeredRep_mul_and_add_tight
  {ctx : ProtocolCtx} {claim : CEClaim} {aDelta bDelta : Coeffs}
  {BA BB : Nat}
  (hP20 : p20ForClaim ctx claim)
  (hDeltaEq : claim.invDelta = mulRq aDelta bDelta)
  (hA : normInfCoeffs aDelta ≤ BA)
  (hB : normInfCoeffs bDelta ≤ BB)
  (hMulRep :
    ∀ x y : F, Int.natAbs (centeredRep (x * y))
      ≤ Int.natAbs (centeredRep x) * Int.natAbs (centeredRep y))
  (hAddRep :
    ∀ x y : F, Int.natAbs (centeredRep (x + y))
      ≤ Int.natAbs (centeredRep x) + Int.natAbs (centeredRep y))
  (hBLt :
    (D * (BA * BB)) + (D * (BA * BB)) + (D * (BA * BB))
      < bInvApprox) :
  ∃ deltaInv : Coeffs, mulRq claim.invDelta deltaInv = oneRq := by
  exact p20InvertibilityWitness_mulRq_of_operand_norm_assumptions_via_centeredRep_mul_and_add_tight_of_assumption
    (hInv := ctx.hLowNormInvertibility)
    (hP20 := hP20)
    (hDeltaEq := hDeltaEq)
    (hA := hA)
    (hB := hB)
    (hMulRep := hMulRep)
    (hAddRep := hAddRep)
    (hBLt := hBLt)

theorem p20ForClaim_invertibilityWitness_mulRq_of_operand_norm_assumptions_via_centeredRep_mul_tight
  {ctx : ProtocolCtx} {claim : CEClaim} {aDelta bDelta : Coeffs}
  {BA BB : Nat}
  (hP20 : p20ForClaim ctx claim)
  (hDeltaEq : claim.invDelta = mulRq aDelta bDelta)
  (hA : normInfCoeffs aDelta ≤ BA)
  (hB : normInfCoeffs bDelta ≤ BB)
  (hMulRep :
    ∀ x y : F, Int.natAbs (centeredRep (x * y))
      ≤ Int.natAbs (centeredRep x) * Int.natAbs (centeredRep y))
  (hBLt :
    (D * (BA * BB)) + (D * (BA * BB)) + (D * (BA * BB))
      < bInvApprox) :
  ∃ deltaInv : Coeffs, mulRq claim.invDelta deltaInv = oneRq := by
  exact p20ForClaim_invertibilityWitness_mulRq_of_operand_norm_assumptions_via_centeredRep_mul_and_add_tight
    (hP20 := hP20)
    (hDeltaEq := hDeltaEq)
    (hA := hA)
    (hB := hB)
    (hMulRep := hMulRep)
    (hAddRep := centeredRepAddTriangleBound_theorem)
    (hBLt := hBLt)

/-- Bundle wrapper for centered-representation mul/add blockers (tight path). -/
theorem p20ForClaim_invertibilityWitness_mulRq_of_operand_norm_assumptions_via_centeredRepMulAddBounds_tight
  {ctx : ProtocolCtx} {claim : CEClaim} {aDelta bDelta : Coeffs}
  {BA BB : Nat}
  (hP20 : p20ForClaim ctx claim)
  (hDeltaEq : claim.invDelta = mulRq aDelta bDelta)
  (hA : normInfCoeffs aDelta ≤ BA)
  (hB : normInfCoeffs bDelta ≤ BB)
  (hRep : centeredRepMulAddBounds)
  (hBLt :
    (D * (BA * BB)) + (D * (BA * BB)) + (D * (BA * BB))
      < bInvApprox) :
  ∃ deltaInv : Coeffs, mulRq claim.invDelta deltaInv = oneRq := by
  exact p20ForClaim_invertibilityWitness_mulRq_of_operand_norm_assumptions_via_centeredRep_mul_and_add_tight
    (hP20 := hP20)
    (hDeltaEq := hDeltaEq)
    (hA := hA)
    (hB := hB)
    (hMulRep := centeredRepMulAddBounds_mul hRep)
    (hAddRep := centeredRepMulAddBounds_add hRep)
    (hBLt := hBLt)

/-- Assumption-free native invertibility witness extraction for `p20ForClaim`. -/
theorem p20ForClaim_invertibilityWitness_mulRq_of_operand_norm_assumptions_native
  {ctx : ProtocolCtx} {claim : CEClaim} {aDelta bDelta : Coeffs}
  {BA BB : Nat}
  (hP20 : p20ForClaim ctx claim)
  (hDeltaEq : claim.invDelta = mulRq aDelta bDelta)
  (hA : normInfCoeffs aDelta ≤ BA)
  (hB : normInfCoeffs bDelta ≤ BB)
  (hBLt :
    ((D * D) * (BA * BB)) + ((D * D) * (BA * BB)) + ((D * D) * (BA * BB))
      < bInvApprox) :
  ∃ deltaInv : Coeffs, mulRq claim.invDelta deltaInv = oneRq := by
  exact p20InvertibilityWitness_mulRq_of_operand_norm_assumptions_native_of_assumption
    (hInv := ctx.hLowNormInvertibility)
    (hP20 := hP20)
    (hDeltaEq := hDeltaEq)
    (hA := hA)
    (hB := hB)
    (hBLt := hBLt)

/-- Assumption-free native-tight invertibility witness extraction for `p20ForClaim`. -/
theorem p20ForClaim_invertibilityWitness_mulRq_of_operand_norm_assumptions_native_tight
  {ctx : ProtocolCtx} {claim : CEClaim} {aDelta bDelta : Coeffs}
  {BA BB : Nat}
  (hP20 : p20ForClaim ctx claim)
  (hDeltaEq : claim.invDelta = mulRq aDelta bDelta)
  (hA : normInfCoeffs aDelta ≤ BA)
  (hB : normInfCoeffs bDelta ≤ BB)
  (hBLt :
    (D * (BA * BB)) + (D * (BA * BB)) + (D * (BA * BB))
      < bInvApprox) :
  ∃ deltaInv : Coeffs, mulRq claim.invDelta deltaInv = oneRq := by
  exact p20InvertibilityWitness_mulRq_of_operand_norm_assumptions_native_tight_of_assumption
    (hInv := ctx.hLowNormInvertibility)
    (hP20 := hP20)
    (hDeltaEq := hDeltaEq)
    (hA := hA)
    (hB := hB)
    (hBLt := hBLt)

theorem p20ForClaim_invertibilityWitness_mulRq_of_operand_norm_assumptions_via_universal_blockers_and_raw
  {ctx : ProtocolCtx} {claim : CEClaim} {aDelta bDelta : Coeffs}
  {BA BB BRaw : Nat}
  (hP20 : p20ForClaim ctx claim)
  (hDeltaEq : claim.invDelta = mulRq aDelta bDelta)
  (hA : normInfCoeffs aDelta ≤ BA)
  (hB : normInfCoeffs bDelta ≤ BB)
  (hRawFromOperands : mulRqRawNormBoundFromOperands BA BB BRaw)
  (hAddTri : schoolbookAddTriangleBound)
  (hSubTri : schoolbookSubTriangleBound)
  (hBLt : BRaw + BRaw + BRaw < bInvApprox) :
  ∃ deltaInv : Coeffs, mulRq claim.invDelta deltaInv = oneRq := by
  exact p20InvertibilityWitness_mulRq_of_operand_norm_assumptions_via_universal_blockers_and_raw_of_assumption
    (hInv := ctx.hLowNormInvertibility)
    (hP20 := hP20)
    (hDeltaEq := hDeltaEq)
    (hA := hA)
    (hB := hB)
    (hRawFromOperands := hRawFromOperands)
    (hAddTri := hAddTri)
    (hSubTri := hSubTri)
    (hBLt := hBLt)

theorem p20ForClaim_invertibilityWitness_mulRq_of_operand_norm_assumptions_via_universal_blockers_and_rawCoeff
  {ctx : ProtocolCtx} {claim : CEClaim} {aDelta bDelta : Coeffs}
  {BA BB BRaw : Nat}
  (hP20 : p20ForClaim ctx claim)
  (hDeltaEq : claim.invDelta = mulRq aDelta bDelta)
  (hA : normInfCoeffs aDelta ≤ BA)
  (hB : normInfCoeffs bDelta ≤ BB)
  (hRawCoeffFromOperands : mulRqRawCoeffBoundFromOperands BA BB BRaw)
  (hAddTri : schoolbookAddTriangleBound)
  (hSubTri : schoolbookSubTriangleBound)
  (hBLt : BRaw + BRaw + BRaw < bInvApprox) :
  ∃ deltaInv : Coeffs, mulRq claim.invDelta deltaInv = oneRq := by
  exact p20ForClaim_invertibilityWitness_mulRq_of_operand_norm_assumptions_via_universal_blockers_and_raw
    (hP20 := hP20)
    (hDeltaEq := hDeltaEq)
    (hA := hA)
    (hB := hB)
    (hRawFromOperands := mulRqRawNormBoundFromOperands_of_rawCoeff hRawCoeffFromOperands)
    (hAddTri := hAddTri)
    (hSubTri := hSubTri)
    (hBLt := hBLt)

theorem p20ForClaim_invertibilityWitness_mulRq_of_operand_norm_assumptions_via_universal_mul_and_add_and_raw
  {ctx : ProtocolCtx} {claim : CEClaim} {aDelta bDelta : Coeffs}
  {BA BB BRaw : Nat}
  (hP20 : p20ForClaim ctx claim)
  (hDeltaEq : claim.invDelta = mulRq aDelta bDelta)
  (hA : normInfCoeffs aDelta ≤ BA)
  (hB : normInfCoeffs bDelta ≤ BB)
  (hRawFromOperands : mulRqRawNormBoundFromOperands BA BB BRaw)
  (hAddTri : schoolbookAddTriangleBound)
  (hBLt : BRaw + BRaw + BRaw < bInvApprox) :
  ∃ deltaInv : Coeffs, mulRq claim.invDelta deltaInv = oneRq := by
  exact p20InvertibilityWitness_mulRq_of_operand_norm_assumptions_via_universal_mul_and_add_and_raw_of_assumption
    (hInv := ctx.hLowNormInvertibility)
    (hP20 := hP20)
    (hDeltaEq := hDeltaEq)
    (hA := hA)
    (hB := hB)
    (hRawFromOperands := hRawFromOperands)
    (hAddTri := hAddTri)
    (hBLt := hBLt)

theorem p20ForClaim_invertibilityWitness_mulRq_of_operand_norm_assumptions_via_universal_mul_and_add_and_rawCoeff
  {ctx : ProtocolCtx} {claim : CEClaim} {aDelta bDelta : Coeffs}
  {BA BB BRaw : Nat}
  (hP20 : p20ForClaim ctx claim)
  (hDeltaEq : claim.invDelta = mulRq aDelta bDelta)
  (hA : normInfCoeffs aDelta ≤ BA)
  (hB : normInfCoeffs bDelta ≤ BB)
  (hRawCoeffFromOperands : mulRqRawCoeffBoundFromOperands BA BB BRaw)
  (hAddTri : schoolbookAddTriangleBound)
  (hBLt : BRaw + BRaw + BRaw < bInvApprox) :
  ∃ deltaInv : Coeffs, mulRq claim.invDelta deltaInv = oneRq := by
  exact p20ForClaim_invertibilityWitness_mulRq_of_operand_norm_assumptions_via_universal_mul_and_add_and_raw
    (hP20 := hP20)
    (hDeltaEq := hDeltaEq)
    (hA := hA)
    (hB := hB)
    (hRawFromOperands := mulRqRawNormBoundFromOperands_of_rawCoeff hRawCoeffFromOperands)
    (hAddTri := hAddTri)
    (hBLt := hBLt)

theorem p20ForClaim_invertibilityWitness_mulRq_of_operand_norm_assumptions_via_centeredRep_mul_and_add_and_raw
  {ctx : ProtocolCtx} {claim : CEClaim} {aDelta bDelta : Coeffs}
  {BA BB BRaw : Nat}
  (hP20 : p20ForClaim ctx claim)
  (hDeltaEq : claim.invDelta = mulRq aDelta bDelta)
  (hA : normInfCoeffs aDelta ≤ BA)
  (hB : normInfCoeffs bDelta ≤ BB)
  (hRawFromOperands : mulRqRawNormBoundFromOperands BA BB BRaw)
  (hAddRep :
    ∀ x y : F, Int.natAbs (centeredRep (x + y))
      ≤ Int.natAbs (centeredRep x) + Int.natAbs (centeredRep y))
  (hBLt : BRaw + BRaw + BRaw < bInvApprox) :
  ∃ deltaInv : Coeffs, mulRq claim.invDelta deltaInv = oneRq := by
  exact p20InvertibilityWitness_mulRq_of_operand_norm_assumptions_via_centeredRep_mul_and_add_and_raw_of_assumption
    (hInv := ctx.hLowNormInvertibility)
    (hP20 := hP20)
    (hDeltaEq := hDeltaEq)
    (hA := hA)
    (hB := hB)
    (hRawFromOperands := hRawFromOperands)
    (hAddRep := hAddRep)
    (hBLt := hBLt)

theorem p20ForClaim_invertibilityWitness_mulRq_of_operand_norm_assumptions_via_centeredRep_mul_and_add_and_rawCoeff
  {ctx : ProtocolCtx} {claim : CEClaim} {aDelta bDelta : Coeffs}
  {BA BB BRaw : Nat}
  (hP20 : p20ForClaim ctx claim)
  (hDeltaEq : claim.invDelta = mulRq aDelta bDelta)
  (hA : normInfCoeffs aDelta ≤ BA)
  (hB : normInfCoeffs bDelta ≤ BB)
  (hRawCoeffFromOperands : mulRqRawCoeffBoundFromOperands BA BB BRaw)
  (hAddRep :
    ∀ x y : F, Int.natAbs (centeredRep (x + y))
      ≤ Int.natAbs (centeredRep x) + Int.natAbs (centeredRep y))
  (hBLt : BRaw + BRaw + BRaw < bInvApprox) :
  ∃ deltaInv : Coeffs, mulRq claim.invDelta deltaInv = oneRq := by
  exact p20ForClaim_invertibilityWitness_mulRq_of_operand_norm_assumptions_via_centeredRep_mul_and_add_and_raw
    (hP20 := hP20)
    (hDeltaEq := hDeltaEq)
    (hA := hA)
    (hB := hB)
    (hRawFromOperands := mulRqRawNormBoundFromOperands_of_rawCoeff hRawCoeffFromOperands)
    (hAddRep := hAddRep)
    (hBLt := hBLt)

/-- Bundle wrapper for centered-representation blockers (raw-bound path). -/
theorem p20ForClaim_invertibilityWitness_mulRq_of_operand_norm_assumptions_via_centeredRepMulAddBounds_and_raw
  {ctx : ProtocolCtx} {claim : CEClaim} {aDelta bDelta : Coeffs}
  {BA BB BRaw : Nat}
  (hP20 : p20ForClaim ctx claim)
  (hDeltaEq : claim.invDelta = mulRq aDelta bDelta)
  (hA : normInfCoeffs aDelta ≤ BA)
  (hB : normInfCoeffs bDelta ≤ BB)
  (hRawFromOperands : mulRqRawNormBoundFromOperands BA BB BRaw)
  (hRep : centeredRepMulAddBounds)
  (hBLt : BRaw + BRaw + BRaw < bInvApprox) :
  ∃ deltaInv : Coeffs, mulRq claim.invDelta deltaInv = oneRq := by
  exact p20ForClaim_invertibilityWitness_mulRq_of_operand_norm_assumptions_via_centeredRep_mul_and_add_and_raw
    (hP20 := hP20)
    (hDeltaEq := hDeltaEq)
    (hA := hA)
    (hB := hB)
    (hRawFromOperands := hRawFromOperands)
    (hAddRep := centeredRepMulAddBounds_add hRep)
    (hBLt := hBLt)

/-- Bundle wrapper for centered-representation blockers (raw-coeff path). -/
theorem p20ForClaim_invertibilityWitness_mulRq_of_operand_norm_assumptions_via_centeredRepMulAddBounds_and_rawCoeff
  {ctx : ProtocolCtx} {claim : CEClaim} {aDelta bDelta : Coeffs}
  {BA BB BRaw : Nat}
  (hP20 : p20ForClaim ctx claim)
  (hDeltaEq : claim.invDelta = mulRq aDelta bDelta)
  (hA : normInfCoeffs aDelta ≤ BA)
  (hB : normInfCoeffs bDelta ≤ BB)
  (hRawCoeffFromOperands : mulRqRawCoeffBoundFromOperands BA BB BRaw)
  (hRep : centeredRepMulAddBounds)
  (hBLt : BRaw + BRaw + BRaw < bInvApprox) :
  ∃ deltaInv : Coeffs, mulRq claim.invDelta deltaInv = oneRq := by
  exact p20ForClaim_invertibilityWitness_mulRq_of_operand_norm_assumptions_via_centeredRepMulAddBounds_and_raw
    (hP20 := hP20)
    (hDeltaEq := hDeltaEq)
    (hA := hA)
    (hB := hB)
    (hRawFromOperands := mulRqRawNormBoundFromOperands_of_rawCoeff hRawCoeffFromOperands)
    (hRep := hRep)
    (hBLt := hBLt)

theorem p20ForClaim_invertibilityWitness_mulRq_of_goldilocks_operand_assumptions_inRange
  {ctx : ProtocolCtx} {claim : CEClaim} {aDelta bDelta : Coeffs}
  (hP20 : p20ForClaim ctx claim)
  (hDeltaEq : claim.invDelta = mulRq aDelta bDelta)
  (hA : normInfCoeffs aDelta ≤ Parameters.Goldilocks.B)
  (hB : normInfCoeffs bDelta ≤ Parameters.Goldilocks.B)
  (hRawInRange : GoldilocksRawInRangeBoundAssumption)
  (hCollapse : GoldilocksRawCollapseAssumption) :
  ∃ deltaInv : Coeffs, mulRq claim.invDelta deltaInv = oneRq := by
  exact p20ForClaim_invertibilityWitness_mulRq_of_goldilocks_operand_assumptions
    (hP20 := hP20)
    (hDeltaEq := hDeltaEq)
    (hA := hA)
    (hB := hB)
    (hRaw := goldilocksRawNormBoundAssumption_of_inRange hRawInRange)
    (hCollapse := hCollapse)

theorem p20ForClaim_invertibilityWitness_mulRq_of_goldilocks_operand_fieldOp_assumptions
  {ctx : ProtocolCtx} {claim : CEClaim} {aDelta bDelta : Coeffs}
  (hP20 : p20ForClaim ctx claim)
  (hDeltaEq : claim.invDelta = mulRq aDelta bDelta)
  (hA : normInfCoeffs aDelta ≤ Parameters.Goldilocks.B)
  (hB : normInfCoeffs bDelta ≤ Parameters.Goldilocks.B)
  (hRaw : GoldilocksRawNormBoundAssumption)
  (hFieldOps : GoldilocksFieldOpCollapseAssumption) :
  ∃ deltaInv : Coeffs, mulRq claim.invDelta deltaInv = oneRq := by
  exact p20ForClaim_invertibilityWitness_mulRq_of_goldilocks_operand_assumptions
    (hP20 := hP20)
    (hDeltaEq := hDeltaEq)
    (hA := hA)
    (hB := hB)
    (hRaw := hRaw)
    (hCollapse := goldilocksRawCollapseAssumption_of_fieldOp hFieldOps)

theorem p20ForClaim_invertibilityWitness_mulRq_of_goldilocks_operand_fieldOp_assumptions_inRange
  {ctx : ProtocolCtx} {claim : CEClaim} {aDelta bDelta : Coeffs}
  (hP20 : p20ForClaim ctx claim)
  (hDeltaEq : claim.invDelta = mulRq aDelta bDelta)
  (hA : normInfCoeffs aDelta ≤ Parameters.Goldilocks.B)
  (hB : normInfCoeffs bDelta ≤ Parameters.Goldilocks.B)
  (hRawInRange : GoldilocksRawInRangeBoundAssumption)
  (hFieldOps : GoldilocksFieldOpCollapseAssumption) :
  ∃ deltaInv : Coeffs, mulRq claim.invDelta deltaInv = oneRq := by
  exact p20ForClaim_invertibilityWitness_mulRq_of_goldilocks_operand_fieldOp_assumptions
    (hP20 := hP20)
    (hDeltaEq := hDeltaEq)
    (hA := hA)
    (hB := hB)
    (hRaw := goldilocksRawNormBoundAssumption_of_inRange hRawInRange)
    (hFieldOps := hFieldOps)

theorem p20ForClaim_invertibilityWitness_mulRq_of_goldilocks_operand_rawCoeff_assumptions
  {ctx : ProtocolCtx} {claim : CEClaim} {aDelta bDelta : Coeffs}
  (hP20 : p20ForClaim ctx claim)
  (hDeltaEq : claim.invDelta = mulRq aDelta bDelta)
  (hA : normInfCoeffs aDelta ≤ Parameters.Goldilocks.B)
  (hB : normInfCoeffs bDelta ≤ Parameters.Goldilocks.B)
  (hRawCoeff : GoldilocksRawCoeffBoundAssumption)
  (hCollapse : GoldilocksRawCollapseAssumption) :
  ∃ deltaInv : Coeffs, mulRq claim.invDelta deltaInv = oneRq := by
  exact p20ForClaim_invertibilityWitness_mulRq_of_goldilocks_operand_assumptions
    (hP20 := hP20)
    (hDeltaEq := hDeltaEq)
    (hA := hA)
    (hB := hB)
    (hRaw := goldilocksRawNormBoundAssumption_of_rawCoeff hRawCoeff)
    (hCollapse := hCollapse)

theorem p20ForClaim_invertibilityWitness_mulRq_of_goldilocks_operand_rawCoeff_fieldOp_assumptions
  {ctx : ProtocolCtx} {claim : CEClaim} {aDelta bDelta : Coeffs}
  (hP20 : p20ForClaim ctx claim)
  (hDeltaEq : claim.invDelta = mulRq aDelta bDelta)
  (hA : normInfCoeffs aDelta ≤ Parameters.Goldilocks.B)
  (hB : normInfCoeffs bDelta ≤ Parameters.Goldilocks.B)
  (hRawCoeff : GoldilocksRawCoeffBoundAssumption)
  (hFieldOps : GoldilocksFieldOpCollapseAssumption) :
  ∃ deltaInv : Coeffs, mulRq claim.invDelta deltaInv = oneRq := by
  exact p20ForClaim_invertibilityWitness_mulRq_of_goldilocks_operand_fieldOp_assumptions
    (hP20 := hP20)
    (hDeltaEq := hDeltaEq)
    (hA := hA)
    (hB := hB)
    (hRaw := goldilocksRawNormBoundAssumption_of_rawCoeff hRawCoeff)
    (hFieldOps := hFieldOps)

end SuperNeo
