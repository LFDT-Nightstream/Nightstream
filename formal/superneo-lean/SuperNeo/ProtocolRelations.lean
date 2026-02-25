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

def ClaimArithmeticValid (ctx : ProtocolCtx) (claim : CEClaim) : Prop :=
  p20DecompProp claim.zDecomp ctx.bSplit ctx.kSplit ∧
    MatrixRowsCompatible claim.m claim.z ∧
    matrixVecDirect claim.m claim.z = matrixVecCtBar ctx.bar claim.m claim.z ∧
    p20EvalHomProp ctx.bar claim.m claim.z1 claim.z2 claim.r claim.rho1 claim.rho2 ∧
    invertibilityPreconditionsProp ∧
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
    claim.cset claim.samples claim.qVals
    claim.xs claim.ys claim.expectedCoeffs
    claim.evalPoint claim.expectedEval
    ctx.ell ctx.totalDegree ctx.setSize) :
  ClaimArithmeticValid ctx claim := by
  simpa [ClaimArithmeticValid] using hP21

theorem p21ProtocolTarget_to_EvalClaimValid
  {ctx : ProtocolCtx} {claim : CEClaim}
  (hShape : ClaimShapeValid claim)
  (hP21 : p21ProtocolTarget
    ctx.bar
    claim.m
    claim.z claim.z1 claim.z2 claim.zDecomp claim.r
    claim.rho1 claim.rho2
    ctx.bSplit ctx.kSplit
    claim.cset claim.samples claim.qVals
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
    claim.cset claim.samples claim.qVals
    claim.xs claim.ys claim.expectedCoeffs
    claim.evalPoint claim.expectedEval
    ctx.ell ctx.totalDegree ctx.setSize)
  (hP10 : p10CoreProp ctx.bar claim.a claim.b)
  (hWitness : witness.z = claim.z)
  (hNorm : normInfCoeffs witness.z < ctx.ceNormBound) :
  CEValid ctx claim witness := by
  exact ⟨p21ProtocolTarget_to_EvalClaimValid hShape hP21, hBar, hA, hB, hP10, hWitness, hNorm⟩

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
    claim.cset claim.samples claim.qVals
    claim.xs claim.ys claim.expectedCoeffs
    claim.evalPoint claim.expectedEval
    ctx.ell ctx.totalDegree ctx.setSize)
  (hWitness : witness.z = claim.z)
  (hNorm : normInfCoeffs witness.z < ctx.ceNormBound) :
  CEValid ctx claim witness := by
  rcases hFull with ⟨hP10, hP21⟩
  exact p21ProtocolTarget_to_CEValid hShape hBar hA hB hP21 hP10 hWitness hNorm

end SuperNeo
