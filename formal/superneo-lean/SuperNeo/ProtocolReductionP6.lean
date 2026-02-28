import SuperNeo.ProtocolReduction
import SuperNeo.DecompNative

/-! Claim-level P6 assumption/check-assumption wrappers for protocol reduction. -/

namespace SuperNeo

theorem splitRoundTrip_forClaim_of_native_base2
  {ctx : PSContext} {claim : PSClaim}
  (hBSplit : ctx.bSplit = 2)
  (hKPos : 0 < ctx.kSplit)
  (hZero : splitScalarTerminalZeroProp claim.zDecomp 2 ctx.kSplit)
  (hCanon : claim.zDecomp.all (fun x => decide (F.Canonical x)) = true) :
  splitRoundTrip claim.zDecomp ctx.bSplit ctx.kSplit = true := by
  simpa [hBSplit] using
    (splitRoundTrip_true_of_base_two_of_state_zero_of_allCanonical
      (z := claim.zDecomp) (k := ctx.kSplit) hKPos hZero hCanon)

theorem splitRoundTrip_forClaim_of_native_base2_of_residue_fold_eq_centeredInt
  {ctx : PSContext} {claim : PSClaim}
  (hBSplit : ctx.bSplit = 2)
  (hKPos : 0 < ctx.kSplit)
  (hEq : ∀ j (_hj : j < claim.zDecomp.size),
    centeredInt claim.zDecomp[j]! = splitScalarResidueFoldInt claim.zDecomp[j]! 2 ctx.kSplit)
  (hCanon : claim.zDecomp.all (fun x => decide (F.Canonical x)) = true) :
  splitRoundTrip claim.zDecomp ctx.bSplit ctx.kSplit = true := by
  simpa [hBSplit] using
    (splitRoundTrip_true_of_base_two_of_residue_fold_eq_centeredInt_of_allCanonical
      (z := claim.zDecomp) (k := ctx.kSplit) hKPos hEq hCanon)

/--
Concrete claim-level P6 closure at `(bSplit,kSplit)=(2,8)` from challenge-coefficient
decomposition rows.
-/
theorem splitRoundTrip_forClaim_of_native_base2_k8_of_allChallenge
  {ctx : PSContext} {claim : PSClaim}
  (hBSplit : ctx.bSplit = 2)
  (hKSplit : ctx.kSplit = 8)
  (hChallenge : ∀ j (hj : j < claim.zDecomp.size), IsChallengeCoeff (claim.zDecomp[j]'hj)) :
  splitRoundTrip claim.zDecomp ctx.bSplit ctx.kSplit = true := by
  simpa [hBSplit, hKSplit] using
    (splitRoundTrip_true_of_base_two_k8_of_allChallenge
      (z := claim.zDecomp) hChallenge)

private theorem splitScalarTerminalZero_forClaim_of_residue_fold_eq_centeredInt
  {ctx : PSContext} {claim : PSClaim}
  (hEq : ∀ j (_hj : j < claim.zDecomp.size),
    centeredInt claim.zDecomp[j]! = splitScalarResidueFoldInt claim.zDecomp[j]! 2 ctx.kSplit) :
  splitScalarTerminalZeroProp claim.zDecomp 2 ctx.kSplit := by
  exact splitScalarTerminalZeroProp_of_residue_fold_eq_centeredInt_base2
    (z := claim.zDecomp) (k := ctx.kSplit) hEq

theorem p20DecompProp_forClaim_of_native_base2
  {ctx : PSContext} {claim : PSClaim}
  (hBSplit : ctx.bSplit = 2)
  (hKPos : 0 < ctx.kSplit)
  (hZero : splitScalarTerminalZeroProp claim.zDecomp 2 ctx.kSplit)
  (hCanon : claim.zDecomp.all (fun x => decide (F.Canonical x)) = true) :
  p20DecompProp claim.zDecomp ctx.bSplit ctx.kSplit := by
  have hRt : splitRoundTrip claim.zDecomp ctx.bSplit ctx.kSplit = true :=
    splitRoundTrip_forClaim_of_native_base2
      (ctx := ctx) (claim := claim) hBSplit hKPos hZero hCanon
  exact p20DecompProp_of_splitRoundTrip
    (z := claim.zDecomp) (b := ctx.bSplit) (k := ctx.kSplit) hRt

theorem p20DecompProp_forClaim_of_native_base2_of_residue_fold_eq_centeredInt
  {ctx : PSContext} {claim : PSClaim}
  (hBSplit : ctx.bSplit = 2)
  (hKPos : 0 < ctx.kSplit)
  (hEq : ∀ j (_hj : j < claim.zDecomp.size),
    centeredInt claim.zDecomp[j]! = splitScalarResidueFoldInt claim.zDecomp[j]! 2 ctx.kSplit)
  (hCanon : claim.zDecomp.all (fun x => decide (F.Canonical x)) = true) :
  p20DecompProp claim.zDecomp ctx.bSplit ctx.kSplit := by
  have hRt : splitRoundTrip claim.zDecomp ctx.bSplit ctx.kSplit = true :=
    splitRoundTrip_forClaim_of_native_base2_of_residue_fold_eq_centeredInt
      (ctx := ctx) (claim := claim) hBSplit hKPos hEq hCanon
  exact p20DecompProp_of_splitRoundTrip
    (z := claim.zDecomp) (b := ctx.bSplit) (k := ctx.kSplit) hRt

theorem p20DecompProp_forClaim_of_native_base2_k8_of_allChallenge
  {ctx : PSContext} {claim : PSClaim}
  (hBSplit : ctx.bSplit = 2)
  (hKSplit : ctx.kSplit = 8)
  (hChallenge : ∀ j (hj : j < claim.zDecomp.size), IsChallengeCoeff (claim.zDecomp[j]'hj)) :
  p20DecompProp claim.zDecomp ctx.bSplit ctx.kSplit := by
  have hRt : splitRoundTrip claim.zDecomp ctx.bSplit ctx.kSplit = true :=
    splitRoundTrip_forClaim_of_native_base2_k8_of_allChallenge
      (ctx := ctx) (claim := claim) hBSplit hKSplit hChallenge
  exact p20DecompProp_of_splitRoundTrip
    (z := claim.zDecomp) (b := ctx.bSplit) (k := ctx.kSplit) hRt

theorem p20ForClaim_of_assumptions_with_native_p6_base2
  {ctx : PSContext} {claim : PSClaim}
  (hBSplit : ctx.bSplit = 2)
  (hKPos : 0 < ctx.kSplit)
  (hZero : splitScalarTerminalZeroProp claim.zDecomp 2 ctx.kSplit)
  (hCanon : claim.zDecomp.all (fun x => decide (F.Canonical x)) = true)
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
  exact p20ForClaim_of_props_with_matrixTransform_assumption_with_evalHom_assumption
    (hP6 := p20DecompProp_forClaim_of_native_base2
      (ctx := ctx) (claim := claim) hBSplit hKPos hZero hCanon)
    (hP12Rows := hP12Rows)
    (hP12Assm := hP12Assm)
    (hP14Assm := hP14Assm)
    (hP14Size := hP14Size)
    (hP14Rows := hP14Rows)
    (hP15Vec := hP15Vec)
    (hP15Scal := hP15Scal)
    (hP16Win := hP16Win)
    (hP17 := hP17)
    (hP18 := hP18)
    (hP19 := hP19)

theorem p20ForClaim_of_assumptions_with_native_p6_base2_k8_of_allChallenge
  {ctx : PSContext} {claim : PSClaim}
  (hBSplit : ctx.bSplit = 2)
  (hKSplit : ctx.kSplit = 8)
  (hChallenge : ∀ j (hj : j < claim.zDecomp.size), IsChallengeCoeff (claim.zDecomp[j]'hj))
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
  have hKPos : 0 < ctx.kSplit := by
    simpa [hKSplit] using (show 0 < (8 : Nat) by decide)
  have hZero : splitScalarTerminalZeroProp claim.zDecomp 2 ctx.kSplit := by
    simpa [hKSplit] using
      (splitScalarTerminalZeroProp_of_allChallenge_base2_k8
        (z := claim.zDecomp) hChallenge)
  have hCanon : claim.zDecomp.all (fun x => decide (F.Canonical x)) = true := by
    exact allCanonical_of_allChallenge (z := claim.zDecomp) hChallenge
  exact p20ForClaim_of_assumptions_with_native_p6_base2
    (ctx := ctx) (claim := claim)
    (hBSplit := hBSplit)
    (hKPos := hKPos)
    (hZero := hZero)
    (hCanon := hCanon)
    (hP12Rows := hP12Rows)
    (hP12Assm := hP12Assm)
    (hP14Assm := hP14Assm)
    (hP14Size := hP14Size)
    (hP14Rows := hP14Rows)
    (hP15Vec := hP15Vec)
    (hP15Scal := hP15Scal)
    (hP16Win := hP16Win)
    (hP17 := hP17)
    (hP18 := hP18)
    (hP19 := hP19)

theorem p20ForClaim_of_checks_with_native_p6_base2
  {ctx : PSContext} {claim : PSClaim}
  (hBSplit : ctx.bSplit = 2)
  (hKPos : 0 < ctx.kSplit)
  (hZero : splitScalarTerminalZeroProp claim.zDecomp 2 ctx.kSplit)
  (hCanon : claim.zDecomp.all (fun x => decide (F.Canonical x)) = true)
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
  p20ForClaim ctx claim := by
  have hP6 : splitRoundTrip claim.zDecomp ctx.bSplit ctx.kSplit = true :=
    splitRoundTrip_forClaim_of_native_base2
      (ctx := ctx) (claim := claim) hBSplit hKPos hZero hCanon
  exact p20ArithmeticBundle_of_checks
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

theorem p20ForClaim_of_checks_with_native_p6_base2_k8_of_allChallenge
  {ctx : PSContext} {claim : PSClaim}
  (hBSplit : ctx.bSplit = 2)
  (hKSplit : ctx.kSplit = 8)
  (hChallenge : ∀ j (hj : j < claim.zDecomp.size), IsChallengeCoeff (claim.zDecomp[j]'hj))
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
  p20ForClaim ctx claim := by
  have hKPos : 0 < ctx.kSplit := by
    simpa [hKSplit] using (show 0 < (8 : Nat) by decide)
  have hZero : splitScalarTerminalZeroProp claim.zDecomp 2 ctx.kSplit := by
    simpa [hKSplit] using
      (splitScalarTerminalZeroProp_of_allChallenge_base2_k8
        (z := claim.zDecomp) hChallenge)
  have hCanon : claim.zDecomp.all (fun x => decide (F.Canonical x)) = true := by
    exact allCanonical_of_allChallenge (z := claim.zDecomp) hChallenge
  exact p20ForClaim_of_checks_with_native_p6_base2
    (ctx := ctx) (claim := claim)
    (hBSplit := hBSplit)
    (hKPos := hKPos)
    (hZero := hZero)
    (hCanon := hCanon)
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

theorem p20ForClaim_of_assumptions_with_native_p6_base2_of_residue_fold_eq_centeredInt
  {ctx : PSContext} {claim : PSClaim}
  (hBSplit : ctx.bSplit = 2)
  (hKPos : 0 < ctx.kSplit)
  (hEq : ∀ j (_hj : j < claim.zDecomp.size),
    centeredInt claim.zDecomp[j]! = splitScalarResidueFoldInt claim.zDecomp[j]! 2 ctx.kSplit)
  (hCanon : claim.zDecomp.all (fun x => decide (F.Canonical x)) = true)
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
  exact p20ForClaim_of_assumptions_with_native_p6_base2
    (hBSplit := hBSplit)
    (hKPos := hKPos)
    (hZero := splitScalarTerminalZero_forClaim_of_residue_fold_eq_centeredInt
      (ctx := ctx) (claim := claim) hEq)
    (hCanon := hCanon)
    (hP12Rows := hP12Rows)
    (hP12Assm := hP12Assm)
    (hP14Assm := hP14Assm)
    (hP14Size := hP14Size)
    (hP14Rows := hP14Rows)
    (hP15Vec := hP15Vec)
    (hP15Scal := hP15Scal)
    (hP16Win := hP16Win)
    (hP17 := hP17)
    (hP18 := hP18)
    (hP19 := hP19)

theorem p20ForClaim_of_checks_with_native_p6_base2_of_residue_fold_eq_centeredInt
  {ctx : PSContext} {claim : PSClaim}
  (hBSplit : ctx.bSplit = 2)
  (hKPos : 0 < ctx.kSplit)
  (hEq : ∀ j (_hj : j < claim.zDecomp.size),
    centeredInt claim.zDecomp[j]! = splitScalarResidueFoldInt claim.zDecomp[j]! 2 ctx.kSplit)
  (hCanon : claim.zDecomp.all (fun x => decide (F.Canonical x)) = true)
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
  p20ForClaim ctx claim := by
  exact p20ForClaim_of_checks_with_native_p6_base2
    (hBSplit := hBSplit)
    (hKPos := hKPos)
    (hZero := splitScalarTerminalZero_forClaim_of_residue_fold_eq_centeredInt
      (ctx := ctx) (claim := claim) hEq)
    (hCanon := hCanon)
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

theorem p20ForClaim_of_assumptions_with_p6DecompAssumption
  {ctx : PSContext} {claim : PSClaim}
  (hP6Assm : p6DecompAssumption ctx.bSplit ctx.kSplit)
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
  exact p20ArithmeticBundle_of_assumptions_with_p6DecompAssumption
    (hP6Assm := hP6Assm)
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

theorem p20ForClaim_of_assumptions_with_p6DecompCheckAssumption
  {ctx : PSContext} {claim : PSClaim}
  (hP6Check : p6DecompCheckAssumption ctx.bSplit ctx.kSplit)
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
  exact p20ArithmeticBundle_of_assumptions_with_p6DecompCheckAssumption
    (hP6Check := hP6Check)
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

theorem p20ForClaim_of_thm3CoreAssumption_with_evalHom_assumption_with_p6DecompAssumption
  {ctx : PSContext} {claim : PSClaim}
  (hShape : ClaimShapeValid claim)
  (hP14Rows : MatrixRowsCompatible claim.m claim.z1)
  (hThm3 : thm3CoreAssumption ctx.bar)
  (hP6Assm : p6DecompAssumption ctx.bSplit ctx.kSplit)
  (hP14Assm : p14EvalHomAssumption ctx.bar claim.m claim.r claim.rho1 claim.rho2)
  (hP15Vec : p20VecModuleProp ctx.hVec claim.rho1 claim.z1 claim.z2)
  (hP15Scal : p20ScalarModuleProp ctx.hScal claim.rho1 claim.z1 claim.z2)
  (hP16Win : p20InvertibilityWindowProp claim.invDelta)
  (hP17 : p20SamplingProp claim.cset claim.samples)
  (hP18 : p20PolyProp claim.qVals ctx.ell ctx.totalDegree ctx.setSize)
  (hP19 : p20InterpProp claim.xs claim.ys claim.expectedCoeffs claim.evalPoint claim.expectedEval) :
  p20ForClaim ctx claim := by
  exact p20ForClaim_of_assumptions_with_p6DecompAssumption
    (hP6Assm := hP6Assm)
    (hP12Rows := claimShapeValid_matrixRowsCompatible hShape)
    (hP12Assm := p12MatrixTransformAssumption_of_thm3CoreAssumption hThm3)
    (hP14Assm := hP14Assm)
    (hP14Size := claimShapeValid_z1_size_eq_z2_size hShape)
    (hP14Rows := hP14Rows)
    (hP15Vec := hP15Vec)
    (hP15Scal := hP15Scal)
    (hP16Win := hP16Win)
    (hP17 := hP17)
    (hP18 := hP18)
    (hP19 := hP19)

theorem p20ForClaim_of_thm3CoreAssumption_with_evalHom_checkAssumption_with_p6DecompCheckAssumption
  {ctx : PSContext} {claim : PSClaim}
  (hShape : ClaimShapeValid claim)
  (hP14Rows : MatrixRowsCompatible claim.m claim.z1)
  (hThm3 : thm3CoreAssumption ctx.bar)
  (hP6Check : p6DecompCheckAssumption ctx.bSplit ctx.kSplit)
  (hP14Check : p14EvalHomCheckAssumption ctx.bar claim.m claim.r claim.rho1 claim.rho2)
  (hP15Vec : p20VecModuleProp ctx.hVec claim.rho1 claim.z1 claim.z2)
  (hP15Scal : p20ScalarModuleProp ctx.hScal claim.rho1 claim.z1 claim.z2)
  (hP16Win : p20InvertibilityWindowProp claim.invDelta)
  (hP17 : p20SamplingProp claim.cset claim.samples)
  (hP18 : p20PolyProp claim.qVals ctx.ell ctx.totalDegree ctx.setSize)
  (hP19 : p20InterpProp claim.xs claim.ys claim.expectedCoeffs claim.evalPoint claim.expectedEval) :
  p20ForClaim ctx claim := by
  exact p20ForClaim_of_assumptions_with_p6DecompCheckAssumption
    (hP6Check := hP6Check)
    (hP12Rows := claimShapeValid_matrixRowsCompatible hShape)
    (hP12Assm := p12MatrixTransformAssumption_of_thm3CoreAssumption hThm3)
    (hP14Assm := p14EvalHomAssumption_of_checkAssumption hP14Check)
    (hP14Size := claimShapeValid_z1_size_eq_z2_size hShape)
    (hP14Rows := hP14Rows)
    (hP15Vec := hP15Vec)
    (hP15Scal := hP15Scal)
    (hP16Win := hP16Win)
    (hP17 := hP17)
    (hP18 := hP18)
    (hP19 := hP19)

theorem p20ForClaim_of_props_with_matrixTransform_assumption_with_evalHom_assumption_with_p6DecompAssumption
  {ctx : PSContext} {claim : PSClaim}
  (hP6Assm : p6DecompAssumption ctx.bSplit ctx.kSplit)
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
  exact p20ForClaim_of_assumptions_with_p6DecompAssumption
    (hP6Assm := hP6Assm)
    (hP12Rows := hP12Rows)
    (hP12Assm := hP12Assm)
    (hP14Assm := hP14Assm)
    (hP14Size := hP14Size)
    (hP14Rows := hP14Rows)
    (hP15Vec := hP15Vec)
    (hP15Scal := hP15Scal)
    (hP16Win := hP16Win)
    (hP17 := hP17)
    (hP18 := hP18)
    (hP19 := hP19)

theorem p20ForClaim_of_props_with_matrixTransform_assumption_with_evalHom_checkAssumption_with_p6DecompAssumption
  {ctx : PSContext} {claim : PSClaim}
  (hP6Assm : p6DecompAssumption ctx.bSplit ctx.kSplit)
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
  exact p20ForClaim_of_assumptions_with_p6DecompAssumption
    (hP6Assm := hP6Assm)
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

theorem p20ForClaim_of_props_with_matrixTransform_checkAssumption_with_evalHom_assumption_with_p6DecompCheckAssumption
  {ctx : PSContext} {claim : PSClaim}
  (hP6Check : p6DecompCheckAssumption ctx.bSplit ctx.kSplit)
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
  exact p20ForClaim_of_assumptions_with_p6DecompCheckAssumption
    (hP6Check := hP6Check)
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

theorem p20ForClaim_of_props_with_matrixTransform_checkAssumption_with_evalHom_checkAssumption_with_p6DecompCheckAssumption
  {ctx : PSContext} {claim : PSClaim}
  (hP6Check : p6DecompCheckAssumption ctx.bSplit ctx.kSplit)
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
  exact p20ForClaim_of_assumptions_with_p6DecompCheckAssumption
    (hP6Check := hP6Check)
    (hP12Rows := hP12Rows)
    (hP12Assm := p12MatrixTransformAssumption_of_checkAssumption hP12Check)
    (hP14Assm := p14EvalHomAssumption_of_checkAssumption hP14Check)
    (hP14Size := hP14Size)
    (hP14Rows := hP14Rows)
    (hP15Vec := hP15Vec)
    (hP15Scal := hP15Scal)
    (hP16Win := hP16Win)
    (hP17 := hP17)
    (hP18 := hP18)
    (hP19 := hP19)

theorem superneoMathProtocolSkeleton_of_props_with_matrixTransform_assumption_with_evalHom_assumption_with_p6DecompAssumption
  {ctx : PSContext} {claim : PSClaim} {witness : PSWitness}
  (hShape : ClaimShapeValid claim)
  (hP12Assm : p12MatrixTransformAssumption ctx.bar claim.m)
  (hP14Rows : MatrixRowsCompatible claim.m claim.z1)
  (hBar : IsDBarMatrix ctx.bar)
  (hA : IsDVec claim.a)
  (hB : IsDVec claim.b)
  (hP10 : p10ForClaim ctx claim)
  (hP6Assm : p6DecompAssumption ctx.bSplit ctx.kSplit)
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
    p20ForClaim_of_assumptions_with_p6DecompAssumption
      (hP6Assm := hP6Assm)
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

theorem superneoMathProtocolSkeleton_of_props_with_matrixTransform_assumption_with_evalHom_assumption_with_native_p6_base2
  {ctx : PSContext} {claim : PSClaim} {witness : PSWitness}
  (hShape : ClaimShapeValid claim)
  (hP12Assm : p12MatrixTransformAssumption ctx.bar claim.m)
  (hP14Rows : MatrixRowsCompatible claim.m claim.z1)
  (hBar : IsDBarMatrix ctx.bar)
  (hA : IsDVec claim.a)
  (hB : IsDVec claim.b)
  (hP10 : p10ForClaim ctx claim)
  (hBSplit : ctx.bSplit = 2)
  (hKPos : 0 < ctx.kSplit)
  (hZero : splitScalarTerminalZeroProp claim.zDecomp 2 ctx.kSplit)
  (hCanon : claim.zDecomp.all (fun x => decide (F.Canonical x)) = true)
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
    p20ForClaim_of_assumptions_with_native_p6_base2
      (hBSplit := hBSplit)
      (hKPos := hKPos)
      (hZero := hZero)
      (hCanon := hCanon)
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

theorem superneoMathProtocolSkeleton_of_props_with_matrixTransform_assumption_with_evalHom_assumption_with_native_p6_base2_of_residue_fold_eq_centeredInt
  {ctx : PSContext} {claim : PSClaim} {witness : PSWitness}
  (hShape : ClaimShapeValid claim)
  (hP12Assm : p12MatrixTransformAssumption ctx.bar claim.m)
  (hP14Rows : MatrixRowsCompatible claim.m claim.z1)
  (hBar : IsDBarMatrix ctx.bar)
  (hA : IsDVec claim.a)
  (hB : IsDVec claim.b)
  (hP10 : p10ForClaim ctx claim)
  (hBSplit : ctx.bSplit = 2)
  (hKPos : 0 < ctx.kSplit)
  (hEq : ∀ j (_hj : j < claim.zDecomp.size),
    centeredInt claim.zDecomp[j]! = splitScalarResidueFoldInt claim.zDecomp[j]! 2 ctx.kSplit)
  (hCanon : claim.zDecomp.all (fun x => decide (F.Canonical x)) = true)
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
  exact superneoMathProtocolSkeleton_of_props_with_matrixTransform_assumption_with_evalHom_assumption_with_native_p6_base2
    (hShape := hShape)
    (hP12Assm := hP12Assm)
    (hP14Rows := hP14Rows)
    (hBar := hBar)
    (hA := hA)
    (hB := hB)
    (hP10 := hP10)
    (hBSplit := hBSplit)
    (hKPos := hKPos)
    (hZero := splitScalarTerminalZero_forClaim_of_residue_fold_eq_centeredInt
      (ctx := ctx) (claim := claim) hEq)
    (hCanon := hCanon)
    (hP14Assm := hP14Assm)
    (hP15Vec := hP15Vec)
    (hP15Scal := hP15Scal)
    (hP16Win := hP16Win)
    (hP17 := hP17)
    (hP18 := hP18)
    (hP19 := hP19)
    (hWitness := hWitness)
    (hNorm := hNorm)

theorem superneoMathProtocolSkeleton_of_props_with_matrixTransform_checkAssumption_with_evalHom_checkAssumption_with_p6DecompCheckAssumption
  {ctx : PSContext} {claim : PSClaim} {witness : PSWitness}
  (hShape : ClaimShapeValid claim)
  (hP12Check : p12MatrixTransformCheckAssumption ctx.bar claim.m)
  (hP14Rows : MatrixRowsCompatible claim.m claim.z1)
  (hBar : IsDBarMatrix ctx.bar)
  (hA : IsDVec claim.a)
  (hB : IsDVec claim.b)
  (hP10 : p10ForClaim ctx claim)
  (hP6Check : p6DecompCheckAssumption ctx.bSplit ctx.kSplit)
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
    p20ForClaim_of_assumptions_with_p6DecompCheckAssumption
      (hP6Check := hP6Check)
      (hP12Rows := claimShapeValid_matrixRowsCompatible hShape)
      (hP12Assm := p12MatrixTransformAssumption_of_checkAssumption hP12Check)
      (hP14Assm := p14EvalHomAssumption_of_checkAssumption hP14Check)
      (hP14Size := claimShapeValid_z1_size_eq_z2_size hShape)
      (hP14Rows := hP14Rows)
      (hP15Vec := hP15Vec)
      (hP15Scal := hP15Scal)
      (hP16Win := hP16Win)
      (hP17 := hP17)
      (hP18 := hP18)
      (hP19 := hP19)
  exact superneoMathProtocolSkeleton_of_props hShape hBar hA hB hP10 hP20 hWitness hNorm

theorem superneoMathProtocolSkeleton_of_props_with_matrixTransform_assumption_with_evalHom_assumption_with_invertibility_with_p6DecompAssumption
  {ctx : PSContext} {claim : PSClaim} {witness : PSWitness}
  (hShape : ClaimShapeValid claim)
  (hP12Assm : p12MatrixTransformAssumption ctx.bar claim.m)
  (hP14Rows : MatrixRowsCompatible claim.m claim.z1)
  (hBar : IsDBarMatrix ctx.bar)
  (hA : IsDVec claim.a)
  (hB : IsDVec claim.b)
  (hP10 : p10ForClaim ctx claim)
  (hP6Assm : p6DecompAssumption ctx.bSplit ctx.kSplit)
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
    p20ForClaim_of_assumptions_with_p6DecompAssumption
      (hP6Assm := hP6Assm)
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
  exact superneoMathProtocolSkeleton_of_props_with_invertibility
    hShape hBar hA hB hP10 hP20 hWitness hNorm

theorem superneoMathProtocolSkeleton_of_props_with_matrixTransform_checkAssumption_with_evalHom_checkAssumption_with_invertibility_with_p6DecompCheckAssumption
  {ctx : PSContext} {claim : PSClaim} {witness : PSWitness}
  (hShape : ClaimShapeValid claim)
  (hP12Check : p12MatrixTransformCheckAssumption ctx.bar claim.m)
  (hP14Rows : MatrixRowsCompatible claim.m claim.z1)
  (hBar : IsDBarMatrix ctx.bar)
  (hA : IsDVec claim.a)
  (hB : IsDVec claim.b)
  (hP10 : p10ForClaim ctx claim)
  (hP6Check : p6DecompCheckAssumption ctx.bSplit ctx.kSplit)
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
    p20ForClaim_of_assumptions_with_p6DecompCheckAssumption
      (hP6Check := hP6Check)
      (hP12Rows := claimShapeValid_matrixRowsCompatible hShape)
      (hP12Assm := p12MatrixTransformAssumption_of_checkAssumption hP12Check)
      (hP14Assm := p14EvalHomAssumption_of_checkAssumption hP14Check)
      (hP14Size := claimShapeValid_z1_size_eq_z2_size hShape)
      (hP14Rows := hP14Rows)
      (hP15Vec := hP15Vec)
      (hP15Scal := hP15Scal)
      (hP16Win := hP16Win)
      (hP17 := hP17)
      (hP18 := hP18)
      (hP19 := hP19)
  exact superneoMathProtocolSkeleton_of_props_with_invertibility
    hShape hBar hA hB hP10 hP20 hWitness hNorm

theorem superneoMathProtocolSkeleton_of_props_with_matrixTransform_assumption_with_evalHom_assumption_with_invertibility_with_native_p6_base2
  {ctx : PSContext} {claim : PSClaim} {witness : PSWitness}
  (hShape : ClaimShapeValid claim)
  (hP12Assm : p12MatrixTransformAssumption ctx.bar claim.m)
  (hP14Rows : MatrixRowsCompatible claim.m claim.z1)
  (hBar : IsDBarMatrix ctx.bar)
  (hA : IsDVec claim.a)
  (hB : IsDVec claim.b)
  (hP10 : p10ForClaim ctx claim)
  (hBSplit : ctx.bSplit = 2)
  (hKPos : 0 < ctx.kSplit)
  (hZero : splitScalarTerminalZeroProp claim.zDecomp 2 ctx.kSplit)
  (hCanon : claim.zDecomp.all (fun x => decide (F.Canonical x)) = true)
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
    p20ForClaim_of_assumptions_with_native_p6_base2
      (hBSplit := hBSplit)
      (hKPos := hKPos)
      (hZero := hZero)
      (hCanon := hCanon)
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
  exact superneoMathProtocolSkeleton_of_props_with_invertibility
    hShape hBar hA hB hP10 hP20 hWitness hNorm

theorem superneoMathProtocolSkeleton_of_props_with_matrixTransform_assumption_with_evalHom_assumption_with_invertibility_with_native_p6_base2_of_residue_fold_eq_centeredInt
  {ctx : PSContext} {claim : PSClaim} {witness : PSWitness}
  (hShape : ClaimShapeValid claim)
  (hP12Assm : p12MatrixTransformAssumption ctx.bar claim.m)
  (hP14Rows : MatrixRowsCompatible claim.m claim.z1)
  (hBar : IsDBarMatrix ctx.bar)
  (hA : IsDVec claim.a)
  (hB : IsDVec claim.b)
  (hP10 : p10ForClaim ctx claim)
  (hBSplit : ctx.bSplit = 2)
  (hKPos : 0 < ctx.kSplit)
  (hEq : ∀ j (_hj : j < claim.zDecomp.size),
    centeredInt claim.zDecomp[j]! = splitScalarResidueFoldInt claim.zDecomp[j]! 2 ctx.kSplit)
  (hCanon : claim.zDecomp.all (fun x => decide (F.Canonical x)) = true)
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
  exact superneoMathProtocolSkeleton_of_props_with_matrixTransform_assumption_with_evalHom_assumption_with_invertibility_with_native_p6_base2
    (hShape := hShape)
    (hP12Assm := hP12Assm)
    (hP14Rows := hP14Rows)
    (hBar := hBar)
    (hA := hA)
    (hB := hB)
    (hP10 := hP10)
    (hBSplit := hBSplit)
    (hKPos := hKPos)
    (hZero := splitScalarTerminalZero_forClaim_of_residue_fold_eq_centeredInt
      (ctx := ctx) (claim := claim) hEq)
    (hCanon := hCanon)
    (hP14Assm := hP14Assm)
    (hP15Vec := hP15Vec)
    (hP15Scal := hP15Scal)
    (hP16Win := hP16Win)
    (hP17 := hP17)
    (hP18 := hP18)
    (hP19 := hP19)
    (hWitness := hWitness)
    (hNorm := hNorm)

/-- Check-surface claim constructor with global P6 check-assumption. -/
theorem p20ForClaim_of_checks_with_p6DecompCheckAssumption
  {ctx : PSContext} {claim : PSClaim}
  (hP6Check : p6DecompCheckAssumption ctx.bSplit ctx.kSplit)
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
  p20ForClaim ctx claim := by
  exact p20ArithmeticBundle_of_checks
    (hP6 := hP6Check.2 claim.zDecomp)
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

/-- Assumption-surface alias of `p20ForClaim_of_checks_with_p6DecompCheckAssumption`. -/
theorem p20ForClaim_of_checks_with_p6DecompAssumption
  {ctx : PSContext} {claim : PSClaim}
  (hP6Assm : p6DecompAssumption ctx.bSplit ctx.kSplit)
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
  p20ForClaim ctx claim := by
  exact p20ForClaim_of_checks_with_p6DecompCheckAssumption
    (hP6Check := p6DecompCheckAssumption_of_assumption hP6Assm)
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

/--
Check-surface protocol constructor with global P6 check-assumption.
This removes per-claim `hP6` plumbing at callsites.
-/
theorem superneoMathProtocolSkeleton_of_checks_with_p6DecompCheckAssumption
  {ctx : PSContext} {claim : PSClaim} {witness : PSWitness}
  (hShape : ClaimShapeValid claim)
  (hBar : IsDBarMatrix ctx.bar)
  (hA : IsDVec claim.a)
  (hB : IsDVec claim.b)
  (hP10 : p10CoreCheck ctx.bar claim.a claim.b = true)
  (hP6Check : p6DecompCheckAssumption ctx.bSplit ctx.kSplit)
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
  exact superneoMathProtocolSkeleton_of_checks
    (hShape := hShape)
    (hBar := hBar)
    (hA := hA)
    (hB := hB)
    (hP10 := hP10)
    (hP6 := hP6Check.2 claim.zDecomp)
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

theorem superneoMathProtocolSkeleton_of_checks_with_native_p6_base2
  {ctx : PSContext} {claim : PSClaim} {witness : PSWitness}
  (hShape : ClaimShapeValid claim)
  (hBar : IsDBarMatrix ctx.bar)
  (hA : IsDVec claim.a)
  (hB : IsDVec claim.b)
  (hP10 : p10CoreCheck ctx.bar claim.a claim.b = true)
  (hBSplit : ctx.bSplit = 2)
  (hKPos : 0 < ctx.kSplit)
  (hZero : splitScalarTerminalZeroProp claim.zDecomp 2 ctx.kSplit)
  (hCanon : claim.zDecomp.all (fun x => decide (F.Canonical x)) = true)
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
  have hP6 : splitRoundTrip claim.zDecomp ctx.bSplit ctx.kSplit = true :=
    splitRoundTrip_forClaim_of_native_base2
      (ctx := ctx) (claim := claim) hBSplit hKPos hZero hCanon
  exact superneoMathProtocolSkeleton_of_checks
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

theorem superneoMathProtocolSkeleton_of_checks_with_native_p6_base2_of_residue_fold_eq_centeredInt
  {ctx : PSContext} {claim : PSClaim} {witness : PSWitness}
  (hShape : ClaimShapeValid claim)
  (hBar : IsDBarMatrix ctx.bar)
  (hA : IsDVec claim.a)
  (hB : IsDVec claim.b)
  (hP10 : p10CoreCheck ctx.bar claim.a claim.b = true)
  (hBSplit : ctx.bSplit = 2)
  (hKPos : 0 < ctx.kSplit)
  (hEq : ∀ j (_hj : j < claim.zDecomp.size),
    centeredInt claim.zDecomp[j]! = splitScalarResidueFoldInt claim.zDecomp[j]! 2 ctx.kSplit)
  (hCanon : claim.zDecomp.all (fun x => decide (F.Canonical x)) = true)
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
  exact superneoMathProtocolSkeleton_of_checks_with_native_p6_base2
    (hShape := hShape)
    (hBar := hBar)
    (hA := hA)
    (hB := hB)
    (hP10 := hP10)
    (hBSplit := hBSplit)
    (hKPos := hKPos)
    (hZero := splitScalarTerminalZero_forClaim_of_residue_fold_eq_centeredInt
      (ctx := ctx) (claim := claim) hEq)
    (hCanon := hCanon)
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

/-- Assumption-surface alias of `superneoMathProtocolSkeleton_of_checks_with_p6DecompCheckAssumption`. -/
theorem superneoMathProtocolSkeleton_of_checks_with_p6DecompAssumption
  {ctx : PSContext} {claim : PSClaim} {witness : PSWitness}
  (hShape : ClaimShapeValid claim)
  (hBar : IsDBarMatrix ctx.bar)
  (hA : IsDVec claim.a)
  (hB : IsDVec claim.b)
  (hP10 : p10CoreCheck ctx.bar claim.a claim.b = true)
  (hP6Assm : p6DecompAssumption ctx.bSplit ctx.kSplit)
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
  exact superneoMathProtocolSkeleton_of_checks_with_p6DecompCheckAssumption
    (hShape := hShape)
    (hBar := hBar)
    (hA := hA)
    (hB := hB)
    (hP10 := hP10)
    (hP6Check := p6DecompCheckAssumption_of_assumption hP6Assm)
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

/-- Invertibility-witness version of `superneoMathProtocolSkeleton_of_checks_with_p6DecompCheckAssumption`. -/
theorem superneoMathProtocolSkeleton_of_checks_with_invertibility_with_p6DecompCheckAssumption
  {ctx : PSContext} {claim : PSClaim} {witness : PSWitness}
  (hShape : ClaimShapeValid claim)
  (hBar : IsDBarMatrix ctx.bar)
  (hA : IsDVec claim.a)
  (hB : IsDVec claim.b)
  (hP10 : p10CoreCheck ctx.bar claim.a claim.b = true)
  (hP6Check : p6DecompCheckAssumption ctx.bSplit ctx.kSplit)
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
  exact superneoMathProtocolSkeleton_of_checks_with_invertibility
    (hShape := hShape)
    (hBar := hBar)
    (hA := hA)
    (hB := hB)
    (hP10 := hP10)
    (hP6 := hP6Check.2 claim.zDecomp)
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

theorem superneoMathProtocolSkeleton_of_checks_with_invertibility_with_native_p6_base2
  {ctx : PSContext} {claim : PSClaim} {witness : PSWitness}
  (hShape : ClaimShapeValid claim)
  (hBar : IsDBarMatrix ctx.bar)
  (hA : IsDVec claim.a)
  (hB : IsDVec claim.b)
  (hP10 : p10CoreCheck ctx.bar claim.a claim.b = true)
  (hBSplit : ctx.bSplit = 2)
  (hKPos : 0 < ctx.kSplit)
  (hZero : splitScalarTerminalZeroProp claim.zDecomp 2 ctx.kSplit)
  (hCanon : claim.zDecomp.all (fun x => decide (F.Canonical x)) = true)
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
  have hP6 : splitRoundTrip claim.zDecomp ctx.bSplit ctx.kSplit = true :=
    splitRoundTrip_forClaim_of_native_base2
      (ctx := ctx) (claim := claim) hBSplit hKPos hZero hCanon
  exact superneoMathProtocolSkeleton_of_checks_with_invertibility
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

theorem superneoMathProtocolSkeleton_of_checks_with_invertibility_with_native_p6_base2_of_residue_fold_eq_centeredInt
  {ctx : PSContext} {claim : PSClaim} {witness : PSWitness}
  (hShape : ClaimShapeValid claim)
  (hBar : IsDBarMatrix ctx.bar)
  (hA : IsDVec claim.a)
  (hB : IsDVec claim.b)
  (hP10 : p10CoreCheck ctx.bar claim.a claim.b = true)
  (hBSplit : ctx.bSplit = 2)
  (hKPos : 0 < ctx.kSplit)
  (hEq : ∀ j (_hj : j < claim.zDecomp.size),
    centeredInt claim.zDecomp[j]! = splitScalarResidueFoldInt claim.zDecomp[j]! 2 ctx.kSplit)
  (hCanon : claim.zDecomp.all (fun x => decide (F.Canonical x)) = true)
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
  exact superneoMathProtocolSkeleton_of_checks_with_invertibility_with_native_p6_base2
    (hShape := hShape)
    (hBar := hBar)
    (hA := hA)
    (hB := hB)
    (hP10 := hP10)
    (hBSplit := hBSplit)
    (hKPos := hKPos)
    (hZero := splitScalarTerminalZero_forClaim_of_residue_fold_eq_centeredInt
      (ctx := ctx) (claim := claim) hEq)
    (hCanon := hCanon)
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

/--
Assumption-surface alias of
`superneoMathProtocolSkeleton_of_checks_with_invertibility_with_p6DecompCheckAssumption`.
-/
theorem superneoMathProtocolSkeleton_of_checks_with_invertibility_with_p6DecompAssumption
  {ctx : PSContext} {claim : PSClaim} {witness : PSWitness}
  (hShape : ClaimShapeValid claim)
  (hBar : IsDBarMatrix ctx.bar)
  (hA : IsDVec claim.a)
  (hB : IsDVec claim.b)
  (hP10 : p10CoreCheck ctx.bar claim.a claim.b = true)
  (hP6Assm : p6DecompAssumption ctx.bSplit ctx.kSplit)
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
  exact superneoMathProtocolSkeleton_of_checks_with_invertibility_with_p6DecompCheckAssumption
    (hShape := hShape)
    (hBar := hBar)
    (hA := hA)
    (hB := hB)
    (hP10 := hP10)
    (hP6Check := p6DecompCheckAssumption_of_assumption hP6Assm)
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

end SuperNeo
