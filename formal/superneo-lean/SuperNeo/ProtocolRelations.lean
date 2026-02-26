import SuperNeo.P21

namespace SuperNeo

open F

/--
Math-level protocol context. This is intentionally lightweight and only carries
the parameters needed to state CE/Eval relations in Lean.
-/
structure ProtocolCtx where
  bar : Array (Array F)
  bSplit : Nat
  kSplit : Nat
  ceNormBound : Nat
  ell : Nat
  totalDegree : Nat
  setSize : Nat
  hVec : VecModuleHom
  hScal : ScalarModuleHom
  hLowNormInvertibility : LowNormInvertibilityAssumption

structure CEClaim where
  a : Array F
  b : Array F
  m : Array (Array F)
  z : Array F
  z1 : Array F
  z2 : Array F
  zDecomp : Array F
  r : Array F
  rho1 : F
  rho2 : F
  cset : Array Coeffs
  samples : Array Coeffs
  invDelta : Coeffs
  qVals : Array F
  xs : Array F
  ys : Array F
  expectedCoeffs : Array F
  evalPoint : F
  expectedEval : F

structure CEWitness where
  z : Array F

def ClaimShapeValid (claim : CEClaim) : Prop :=
  claim.z1.size = claim.z2.size ∧
    MatrixRowsCompatible claim.m claim.z ∧
    claim.xs.size = claim.ys.size

theorem claimShapeValid_z1_size_eq_z2_size
  {claim : CEClaim} (hShape : ClaimShapeValid claim) :
  claim.z1.size = claim.z2.size :=
  hShape.1

theorem claimShapeValid_matrixRowsCompatible
  {claim : CEClaim} (hShape : ClaimShapeValid claim) :
  MatrixRowsCompatible claim.m claim.z :=
  hShape.2.1

theorem claimShapeValid_xs_size_eq_ys_size
  {claim : CEClaim} (hShape : ClaimShapeValid claim) :
  claim.xs.size = claim.ys.size :=
  hShape.2.2

def ClaimArithmeticValid (ctx : ProtocolCtx) (claim : CEClaim) : Prop :=
  p20DecompProp claim.zDecomp ctx.bSplit ctx.kSplit ∧
    MatrixRowsCompatible claim.m claim.z ∧
    matrixVecDirect claim.m claim.z = matrixVecCtBar ctx.bar claim.m claim.z ∧
    p20EvalHomProp ctx.bar claim.m claim.z1 claim.z2 claim.r claim.rho1 claim.rho2 ∧
    invertibilityPreconditionsProp ∧
    p20InvertibilityWindowProp claim.invDelta ∧
    p20SamplingProp claim.cset claim.samples ∧
    p20PolyProp claim.qVals ctx.ell ctx.totalDegree ctx.setSize ∧
    p20InterpProp claim.xs claim.ys claim.expectedCoeffs claim.evalPoint claim.expectedEval

def EvalClaimValid (ctx : ProtocolCtx) (claim : CEClaim) : Prop :=
  ClaimShapeValid claim ∧ ClaimArithmeticValid ctx claim

def CEValid (ctx : ProtocolCtx) (claim : CEClaim) (witness : CEWitness) : Prop :=
  EvalClaimValid ctx claim ∧
    IsDBarMatrix ctx.bar ∧
    IsDVec claim.a ∧
    IsDVec claim.b ∧
    p10CoreProp ctx.bar claim.a claim.b ∧
    witness.z = claim.z ∧
    normInfCoeffs witness.z < ctx.ceNormBound

theorem p21ProtocolTarget_to_ClaimArithmeticValid
  {ctx : ProtocolCtx} {claim : CEClaim}
  (hP21 : p21ProtocolTarget
    ctx.bar
    claim.m
    claim.z claim.z1 claim.z2 claim.zDecomp claim.r
    claim.rho1 claim.rho2
    ctx.bSplit ctx.kSplit
    claim.cset claim.samples claim.invDelta claim.qVals
    claim.xs claim.ys claim.expectedCoeffs
    claim.evalPoint claim.expectedEval
    ctx.ell ctx.totalDegree ctx.setSize) :
  ClaimArithmeticValid ctx claim := by
  rcases hP21 with ⟨hDecomp, hRows, hMat, hEval, hInvPre, hInvWin, hSampling, hPoly, hInterp⟩
  exact ⟨hDecomp, hRows, hMat, hEval, hInvPre, hInvWin, hSampling, hPoly, hInterp⟩

theorem p21ProtocolTarget_to_EvalClaimValid
  {ctx : ProtocolCtx} {claim : CEClaim}
  (hShape : ClaimShapeValid claim)
  (hP21 : p21ProtocolTarget
    ctx.bar
    claim.m
    claim.z claim.z1 claim.z2 claim.zDecomp claim.r
    claim.rho1 claim.rho2
    ctx.bSplit ctx.kSplit
    claim.cset claim.samples claim.invDelta claim.qVals
    claim.xs claim.ys claim.expectedCoeffs
    claim.evalPoint claim.expectedEval
    ctx.ell ctx.totalDegree ctx.setSize) :
  EvalClaimValid ctx claim := by
  exact ⟨hShape, p21ProtocolTarget_to_ClaimArithmeticValid hP21⟩

theorem p21ProtocolTarget_to_CEValid
  {ctx : ProtocolCtx} {claim : CEClaim} {witness : CEWitness}
  (hShape : ClaimShapeValid claim)
  (hBar : IsDBarMatrix ctx.bar)
  (hA : IsDVec claim.a)
  (hB : IsDVec claim.b)
  (hP21 : p21ProtocolTarget
    ctx.bar
    claim.m
    claim.z claim.z1 claim.z2 claim.zDecomp claim.r
    claim.rho1 claim.rho2
    ctx.bSplit ctx.kSplit
    claim.cset claim.samples claim.invDelta claim.qVals
    claim.xs claim.ys claim.expectedCoeffs
    claim.evalPoint claim.expectedEval
    ctx.ell ctx.totalDegree ctx.setSize)
  (hP10 : p10CoreProp ctx.bar claim.a claim.b)
  (hWitness : witness.z = claim.z)
  (hNorm : normInfCoeffs witness.z < ctx.ceNormBound) :
  CEValid ctx claim witness := by
  exact ⟨p21ProtocolTarget_to_EvalClaimValid hShape hP21, hBar, hA, hB, hP10, hWitness, hNorm⟩

theorem p21ProtocolTarget_to_CEValid_with_invertibility
  {ctx : ProtocolCtx} {claim : CEClaim} {witness : CEWitness}
  (hShape : ClaimShapeValid claim)
  (hBar : IsDBarMatrix ctx.bar)
  (hA : IsDVec claim.a)
  (hB : IsDVec claim.b)
  (hP21 : p21ProtocolTarget
    ctx.bar
    claim.m
    claim.z claim.z1 claim.z2 claim.zDecomp claim.r
    claim.rho1 claim.rho2
    ctx.bSplit ctx.kSplit
    claim.cset claim.samples claim.invDelta claim.qVals
    claim.xs claim.ys claim.expectedCoeffs
    claim.evalPoint claim.expectedEval
    ctx.ell ctx.totalDegree ctx.setSize)
  (hP10 : p10CoreProp ctx.bar claim.a claim.b)
  (hWitness : witness.z = claim.z)
  (hNorm : normInfCoeffs witness.z < ctx.ceNormBound) :
  ∃ deltaInv : Coeffs, mulRq claim.invDelta deltaInv = oneRq ∧ CEValid ctx claim witness := by
  have hArith : ClaimArithmeticValid ctx claim := p21ProtocolTarget_to_ClaimArithmeticValid hP21
  rcases hArith with ⟨_hP6, _hRows, _hMat, _hEval, _hInvPre, hInvWin, _hSamp, _hPoly, _hInterp⟩
  rcases invertible_of_withinInvertibilityWindow_of_assumption ctx.hLowNormInvertibility hInvWin with ⟨deltaInv, hMul⟩
  exact ⟨deltaInv, hMul, p21ProtocolTarget_to_CEValid hShape hBar hA hB hP21 hP10 hWitness hNorm⟩

theorem p21FullMathTarget_to_CEValid
  {ctx : ProtocolCtx} {claim : CEClaim} {witness : CEWitness}
  (hShape : ClaimShapeValid claim)
  (hBar : IsDBarMatrix ctx.bar)
  (hA : IsDVec claim.a)
  (hB : IsDVec claim.b)
  (hFull : p21FullMathTarget
    ctx.bar
    claim.a claim.b
    claim.m
    claim.z claim.z1 claim.z2 claim.zDecomp claim.r
    claim.rho1 claim.rho2
    ctx.bSplit ctx.kSplit
    claim.cset claim.samples claim.invDelta claim.qVals
    claim.xs claim.ys claim.expectedCoeffs
    claim.evalPoint claim.expectedEval
    ctx.ell ctx.totalDegree ctx.setSize)
  (hWitness : witness.z = claim.z)
  (hNorm : normInfCoeffs witness.z < ctx.ceNormBound) :
  CEValid ctx claim witness := by
  rcases hFull with ⟨hP10, hP21⟩
  exact p21ProtocolTarget_to_CEValid hShape hBar hA hB hP21 hP10 hWitness hNorm

theorem p21FullMathTarget_to_CEValid_with_invertibility
  {ctx : ProtocolCtx} {claim : CEClaim} {witness : CEWitness}
  (hShape : ClaimShapeValid claim)
  (hBar : IsDBarMatrix ctx.bar)
  (hA : IsDVec claim.a)
  (hB : IsDVec claim.b)
  (hFull : p21FullMathTarget
    ctx.bar
    claim.a claim.b
    claim.m
    claim.z claim.z1 claim.z2 claim.zDecomp claim.r
    claim.rho1 claim.rho2
    ctx.bSplit ctx.kSplit
    claim.cset claim.samples claim.invDelta claim.qVals
    claim.xs claim.ys claim.expectedCoeffs
    claim.evalPoint claim.expectedEval
    ctx.ell ctx.totalDegree ctx.setSize)
  (hWitness : witness.z = claim.z)
  (hNorm : normInfCoeffs witness.z < ctx.ceNormBound) :
  ∃ deltaInv : Coeffs, mulRq claim.invDelta deltaInv = oneRq ∧ CEValid ctx claim witness := by
  rcases hFull with ⟨hP10, hP21⟩
  exact p21ProtocolTarget_to_CEValid_with_invertibility hShape hBar hA hB hP21 hP10 hWitness hNorm

theorem claimArithmetic_invertibilityWitness
  {ctx : ProtocolCtx} {claim : CEClaim}
  (hArith : ClaimArithmeticValid ctx claim) :
  ∃ deltaInv : Coeffs, mulRq claim.invDelta deltaInv = oneRq := by
  rcases hArith with ⟨_hP6, _hRows, _hMat, _hEval, _hInvPre, hInvWin, _hSamp, _hPoly, _hInterp⟩
  exact invertible_of_withinInvertibilityWindow_of_assumption ctx.hLowNormInvertibility hInvWin

theorem claimArithmetic_evalHomCheck
  {ctx : ProtocolCtx} {claim : CEClaim}
  (hArith : ClaimArithmeticValid ctx claim) :
  evalHom2 ctx.bar claim.m claim.z1 claim.z2 claim.r claim.rho1 claim.rho2 = true := by
  rcases hArith with ⟨_hP6, _hRows, _hMat, hP14, _hInvPre, _hInvWin, _hSamp, _hPoly, _hInterp⟩
  exact evalHom2_complete hP14

end SuperNeo
