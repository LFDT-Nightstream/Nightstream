import SuperNeo.EmbeddingTheory.Thm3Core
import SuperNeo.FoldingProtocol.ArithmeticObligations
import SuperNeo.SecurityModel.InvertibilityAxioms
import SuperNeo.SecurityModel.InvertibilityGoldilocks
import SuperNeo.SecurityModel.SamplingSet

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
    ctx.cset ctx.samples
    ctx.xs ctx.ys ctx.qVals ctx.coeffs
    ctx.xEval ctx.expectedEval
  invDeltaInvertible : invertibleRq ctx.invDelta

/-- Protocol-target proposition (compact protocol-math-target style surface). -/
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

/-- Derive the protocol target from explicit theorem/assumption inputs. -/
theorem protocolTargetProp_of_components
  {ctx : ProtocolTargetContext}
  (hThm3 : thm3CoreAssumption ctx.bar)
  (hArithmetic : ArithmeticObligations
    ctx.bar ctx.m ctx.r ctx.rho1 ctx.rho2
    ctx.hVec ctx.hScal
    ctx.splitScalar ctx.kSplit
    ctx.cset ctx.samples
    ctx.xs ctx.ys ctx.qVals ctx.coeffs
    ctx.xEval ctx.expectedEval)
  (hInvDelta : invertibleRq ctx.invDelta) :
  protocolTargetProp ctx := by
  exact ⟨hThm3, hArithmetic.splitTerminalZero, hArithmetic.evalHom,
    hArithmetic.vecModule, hArithmetic.scalarModule, hArithmetic.sampling,
    hArithmetic.mleTableSize, hArithmetic.mleIdentityAtR,
    hArithmetic.interpolation, hInvDelta⟩

/-- Derive the protocol target from explicit theorem/assumption inputs. -/
theorem protocolTargetProp_of_assumptions
  {ctx : ProtocolTargetContext}
  (h : ProtocolTargetAssumptions ctx) :
  protocolTargetProp ctx := by
  exact protocolTargetProp_of_components h.thm3 h.arithmetic h.invDeltaInvertible

/--
Paper-facing invertibility bridge: if `invDelta` is a nonzero difference of two
elements from the proved `paperCarrier`, then the strict low-norm window `< 5`
holds.
-/
theorem strictInvertibilityWindowProp_five_of_paperCarrierDiff
  {δ : Coeffs}
  (hDiff : samplingDiffSet paperCarrier δ)
  (hNe : δ ≠ zeroRq) :
  strictInvertibilityWindowProp 5 δ := by
  rcases samplingDiffSet_paperCarrier_hasRingDegreeShape_and_norm_le_four hDiff with
    ⟨hShape, hNorm⟩
  exact strictInvertibilityWindowProp_five_of_shape_norm_le_four_of_ne_zeroRq
    hShape hNorm hNe

/-- Derive invertibility on the active paper-carrier-difference route. -/
theorem invertibleRq_of_paperCarrierDiff
  {δ : Coeffs}
  (hDiff : samplingDiffSet paperCarrier δ)
  (hNe : δ ≠ zeroRq) :
  invertibleRq δ := by
  exact paperCarrierDiffInvertibilityAssumption_goldilocks δ hDiff hNe

/--
Canonical protocol-target constructor on the paper-facing challenge-difference
path: `invDelta` is a nonzero difference of two paper-carrier elements, and the
only remaining invertibility boundary is the corresponding paper-carrier
difference predicate.
-/
def ProtocolTargetAssumptions.ofPaperCarrierDiff
  {ctx : ProtocolTargetContext}
  (thm3 : thm3CoreAssumption ctx.bar)
  (arithmetic : ArithmeticObligations
    ctx.bar ctx.m ctx.r ctx.rho1 ctx.rho2
    ctx.hVec ctx.hScal
    ctx.splitScalar ctx.kSplit
    ctx.cset ctx.samples
    ctx.xs ctx.ys ctx.qVals ctx.coeffs
    ctx.xEval ctx.expectedEval)
  (hDiff : samplingDiffSet paperCarrier ctx.invDelta)
  (hNe : ctx.invDelta ≠ zeroRq) :
  ProtocolTargetAssumptions ctx :=
  { thm3 := thm3
    arithmetic := arithmetic
    invDeltaInvertible := invertibleRq_of_paperCarrierDiff hDiff hNe }

end SuperNeo
