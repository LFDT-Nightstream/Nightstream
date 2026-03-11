import SuperNeo.Thm3Core
import SuperNeo.ArithmeticObligations
import SuperNeo.InvertibilityAxioms
import SuperNeo.InvertibilityGoldilocks
import SuperNeo.SamplingSet

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

/--
Native protocol-target assumptions.

This variant closes Theorem-3 through the canonical native bar matrix and does
not require an explicit `thm3CoreAssumption` field.
-/
structure ProtocolTargetNativeAssumptions (ctx : ProtocolTargetContext) where
  barNative : ctx.bar = nativeBarMatrix
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
Derive the protocol target from the finite basis-kernel characterization of
Theorem 3 rather than a raw `thm3CoreAssumption`.
-/
theorem protocolTargetProp_of_basisKernelAssumption
  {ctx : ProtocolTargetContext}
  (hBasis : thm3BasisKernelAssumption ctx.bar)
  (hArithmetic : ArithmeticObligations
    ctx.bar ctx.m ctx.r ctx.rho1 ctx.rho2
    ctx.hVec ctx.hScal
    ctx.splitScalar ctx.kSplit
    ctx.cset ctx.samples
    ctx.xs ctx.ys ctx.qVals ctx.coeffs
    ctx.xEval ctx.expectedEval)
  (hInvDelta : invertibleRq ctx.invDelta) :
  protocolTargetProp ctx := by
  exact protocolTargetProp_of_components
    (thm3CoreAssumption_of_basisKernelAssumption hBasis)
    hArithmetic
    hInvDelta

/--
Derive the protocol target from the executable finite basis-kernel checker for
Theorem 3.
-/
theorem protocolTargetProp_of_basisKernelCheck
  {ctx : ProtocolTargetContext}
  (hCheck : thm3BasisKernelCheck ctx.bar = true)
  (hArithmetic : ArithmeticObligations
    ctx.bar ctx.m ctx.r ctx.rho1 ctx.rho2
    ctx.hVec ctx.hScal
    ctx.splitScalar ctx.kSplit
    ctx.cset ctx.samples
    ctx.xs ctx.ys ctx.qVals ctx.coeffs
    ctx.xEval ctx.expectedEval)
  (hInvDelta : invertibleRq ctx.invDelta) :
  protocolTargetProp ctx := by
  exact protocolTargetProp_of_basisKernelAssumption
    (thm3BasisKernelAssumption_of_check hCheck)
    hArithmetic
    hInvDelta

/--
Derive protocol target from native assumptions, closing Theorem-3 by rewriting
`ctx.bar` to `nativeBarMatrix`.
-/
theorem protocolTargetProp_of_native_assumptions
  {ctx : ProtocolTargetContext}
  (h : ProtocolTargetNativeAssumptions ctx) :
  protocolTargetProp ctx := by
  rcases h with ⟨hBar, hArithmetic, hInvDelta⟩
  have hThm3 : thm3CoreAssumption ctx.bar := by
    simpa [hBar] using thm3CoreAssumption_native
  exact protocolTargetProp_of_components hThm3 hArithmetic hInvDelta

/--
Derive the protocol target directly on the active paper-carrier-difference
route without first packaging the assumptions into a boundary bundle.
-/
theorem protocolTargetProp_of_paperCarrierDiff
  {ctx : ProtocolTargetContext}
  (hThm3 : thm3CoreAssumption ctx.bar)
  (hArithmetic : ArithmeticObligations
    ctx.bar ctx.m ctx.r ctx.rho1 ctx.rho2
    ctx.hVec ctx.hScal
    ctx.splitScalar ctx.kSplit
    ctx.cset ctx.samples
    ctx.xs ctx.ys ctx.qVals ctx.coeffs
    ctx.xEval ctx.expectedEval)
  (hDiff : samplingDiffSet paperCarrier ctx.invDelta)
  (hNe : ctx.invDelta ≠ zeroRq) :
  protocolTargetProp ctx := by
  exact protocolTargetProp_of_components hThm3 hArithmetic
    (paperCarrierDiffInvertibilityAssumption_goldilocks ctx.invDelta hDiff hNe)

/--
Derive the protocol target directly on the active native-bar
paper-carrier-difference route.
-/
theorem protocolTargetProp_of_native_paperCarrierDiff
  {ctx : ProtocolTargetContext}
  (hBarNative : ctx.bar = nativeBarMatrix)
  (hArithmetic : ArithmeticObligations
    ctx.bar ctx.m ctx.r ctx.rho1 ctx.rho2
    ctx.hVec ctx.hScal
    ctx.splitScalar ctx.kSplit
    ctx.cset ctx.samples
    ctx.xs ctx.ys ctx.qVals ctx.coeffs
    ctx.xEval ctx.expectedEval)
  (hDiff : samplingDiffSet paperCarrier ctx.invDelta)
  (hNe : ctx.invDelta ≠ zeroRq) :
  protocolTargetProp ctx := by
  have hThm3 : thm3CoreAssumption ctx.bar := by
    simpa [hBarNative] using thm3CoreAssumption_native
  exact protocolTargetProp_of_paperCarrierDiff hThm3 hArithmetic hDiff hNe

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

/--
Canonical protocol-target constructor on the paper-facing challenge-difference
path, deriving Theorem 3 from its finite basis-kernel characterization.
-/
def ProtocolTargetAssumptions.ofBasisKernelAssumption
  {ctx : ProtocolTargetContext}
  (hBasis : thm3BasisKernelAssumption ctx.bar)
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
  ofPaperCarrierDiff
    (thm3CoreAssumption_of_basisKernelAssumption hBasis)
    arithmetic
    hDiff
    hNe

/--
Canonical protocol-target constructor on the paper-facing challenge-difference
path, deriving Theorem 3 from the executable finite basis-kernel checker.
-/
def ProtocolTargetAssumptions.ofBasisKernelCheck
  {ctx : ProtocolTargetContext}
  (hCheck : thm3BasisKernelCheck ctx.bar = true)
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
  ofBasisKernelAssumption
    (thm3BasisKernelAssumption_of_check hCheck)
    arithmetic
    hDiff
    hNe

/--
Canonical protocol-target constructor from any strict low-norm invertibility
boundary whose threshold is at least `5`, specialized to the active
paper-carrier-difference route.
-/
def ProtocolTargetAssumptions.ofLowNormAtLeastFive
  {ctx : ProtocolTargetContext}
  {B : Nat}
  (hFive : 5 ≤ B)
  (thm3 : thm3CoreAssumption ctx.bar)
  (arithmetic : ArithmeticObligations
    ctx.bar ctx.m ctx.r ctx.rho1 ctx.rho2
    ctx.hVec ctx.hScal
    ctx.splitScalar ctx.kSplit
    ctx.cset ctx.samples
    ctx.xs ctx.ys ctx.qVals ctx.coeffs
    ctx.xEval ctx.expectedEval)
  (hInv : lowNormInvertibilityAssumption B)
  (hDiff : samplingDiffSet paperCarrier ctx.invDelta)
  (hNe : ctx.invDelta ≠ zeroRq) :
  ProtocolTargetAssumptions ctx :=
  let hShape := (samplingDiffSet_paperCarrier_hasRingDegreeShape_and_norm_le_four hDiff).1
  { thm3 := thm3
    arithmetic := arithmetic
    invDeltaInvertible := invertibleRq_of_lowNormAssumption hInv hShape
      (strictInvertibilityWindowProp_mono hFive
        (strictInvertibilityWindowProp_five_of_paperCarrierDiff hDiff hNe)) }

/--
Canonical native protocol-target constructor on the paper-facing
challenge-difference path.
-/
def ProtocolTargetNativeAssumptions.ofPaperCarrierDiff
  {ctx : ProtocolTargetContext}
  (barNative : ctx.bar = nativeBarMatrix)
  (arithmetic : ArithmeticObligations
    ctx.bar ctx.m ctx.r ctx.rho1 ctx.rho2
    ctx.hVec ctx.hScal
    ctx.splitScalar ctx.kSplit
    ctx.cset ctx.samples
    ctx.xs ctx.ys ctx.qVals ctx.coeffs
    ctx.xEval ctx.expectedEval)
  (hDiff : samplingDiffSet paperCarrier ctx.invDelta)
  (hNe : ctx.invDelta ≠ zeroRq) :
  ProtocolTargetNativeAssumptions ctx :=
  { barNative := barNative
    arithmetic := arithmetic
    invDeltaInvertible := invertibleRq_of_paperCarrierDiff hDiff hNe }

/--
Canonical native protocol-target constructor from any strict low-norm
invertibility boundary whose threshold is at least `5`, specialized to the
active paper-carrier-difference route.
-/
def ProtocolTargetNativeAssumptions.ofLowNormAtLeastFive
  {ctx : ProtocolTargetContext}
  {B : Nat}
  (hFive : 5 ≤ B)
  (barNative : ctx.bar = nativeBarMatrix)
  (arithmetic : ArithmeticObligations
    ctx.bar ctx.m ctx.r ctx.rho1 ctx.rho2
    ctx.hVec ctx.hScal
    ctx.splitScalar ctx.kSplit
    ctx.cset ctx.samples
    ctx.xs ctx.ys ctx.qVals ctx.coeffs
    ctx.xEval ctx.expectedEval)
  (hInv : lowNormInvertibilityAssumption B)
  (hDiff : samplingDiffSet paperCarrier ctx.invDelta)
  (hNe : ctx.invDelta ≠ zeroRq) :
  ProtocolTargetNativeAssumptions ctx :=
  let hShape := (samplingDiffSet_paperCarrier_hasRingDegreeShape_and_norm_le_four hDiff).1
  { barNative := barNative
    arithmetic := arithmetic
    invDeltaInvertible := invertibleRq_of_lowNormAssumption hInv hShape
      (strictInvertibilityWindowProp_mono hFive
        (strictInvertibilityWindowProp_five_of_paperCarrierDiff hDiff hNe)) }
end SuperNeo
