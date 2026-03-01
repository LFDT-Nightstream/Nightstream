import SuperNeo.Thm3Core
import SuperNeo.ArithmeticObligations

/-!
Protocol-target layer.

This module binds Theorem-3 and arithmetic obligations into one target context,
then derives the core target proposition used by protocol relations.
-/

namespace SuperNeo

/-- Core protocol target context used by relation/reduction layers. -/
structure ProtocolTargetContext where
  bar : Array (Array F)
  m : Array (Array F)
  r : Array F
  rho1 : F
  rho2 : F
  hVec : VecModuleHom
  hScal : ScalarModuleHom
  splitScalar : F
  kSplit : Nat
  invDelta : Coeffs
  cset : Array Coeffs
  samples : Array Coeffs
  xs : Array F
  ys : Array F
  qVals : Array F
  coeffs : Array F
  xEval : F
  expectedEval : F

/-- Assumption bundle for protocol-target derivation. -/
structure ProtocolTargetAssumptions (ctx : ProtocolTargetContext) where
  thm3 : thm3CoreAssumption ctx.bar
  arithmetic : ArithmeticObligations
    ctx.bar ctx.m ctx.r ctx.rho1 ctx.rho2
    ctx.hVec ctx.hScal
    ctx.splitScalar ctx.kSplit
    ctx.invDelta ctx.cset ctx.samples
    ctx.xs ctx.ys ctx.qVals ctx.coeffs
    ctx.xEval ctx.expectedEval
  lowNormInvertibility : lowNormInvertibilityAssumption Goldilocks.halfQ

/-- Protocol-target proposition (compact P21-style surface). -/
def protocolTargetProp (ctx : ProtocolTargetContext) : Prop :=
  thm3CoreAssumption ctx.bar ∧
  splitBase2TerminalZeroProp ctx.splitScalar ctx.kSplit ∧
  evalHomAssumption ctx.bar ctx.m ctx.r ctx.rho1 ctx.rho2 ∧
  vecModuleAssumption ctx.hVec ∧
  scalarModuleAssumption ctx.hScal ∧
  samplingExpansionProp ctx.cset ctx.samples ∧
  ctx.qVals.size = (2 ^ ctx.r.size) ∧
  mleEval ctx.qVals ctx.r = mleInnerProductForm ctx.qVals ctx.r ∧
  interpolationProp ctx.xs ctx.ys ctx.coeffs ctx.xEval ctx.expectedEval ∧
  invertibleRq ctx.invDelta

/--
Derive Theorem-4 matrix transform assumption from Theorem-3 inner-product
assumption by applying the row statement pointwise.
-/
theorem matrixTransformAssumption_of_thm3CoreAssumption
  {bar m : Array (Array F)}
  (hThm3 : thm3CoreAssumption bar) :
  matrixTransformAssumption bar m := by
  intro z hRows
  apply Array.ext
  · simp [matrixVecDirect, matrixVecCtBar]
  · intro i hiL hiR
    have hi : i < m.size := by
      simpa [matrixVecDirect] using hiL
    have hEq := hThm3 (m[i]'hi) z (hRows i hi)
    simpa [matrixVecDirect, matrixVecCtBar, dotVec] using hEq

/-- Derive the protocol target from explicit theorem/assumption inputs. -/
theorem protocolTargetProp_of_assumptions
  {ctx : ProtocolTargetContext}
  (h : ProtocolTargetAssumptions ctx) :
  protocolTargetProp ctx := by
  refine ⟨h.thm3, h.arithmetic.splitTerminalZero, h.arithmetic.evalHom,
    h.arithmetic.vecModule, h.arithmetic.scalarModule, h.arithmetic.sampling,
    h.arithmetic.mleTableSize, h.arithmetic.mleIdentityAtR, h.arithmetic.interpolation, ?_⟩
  exact invertibleRq_of_lowNormAssumption h.lowNormInvertibility h.arithmetic.invertibilityWindow
end SuperNeo
