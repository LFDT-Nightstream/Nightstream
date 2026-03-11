import SuperNeo.ProtocolTarget

/-!
Explicit protocol-side owner for the Section 7.5 protocol-target inputs on the
paper-facing challenge-difference route.
-/

namespace SuperNeo

/--
Protocol-side data needed to derive the compact protocol target on the active
paper-facing challenge-difference route.

This keeps the actual protocol-facing ingredients visible:
- one Theorem-3 witness for `ctx.bar`,
- one arithmetic-obligation package,
- one proof that `ctx.invDelta` is a nonzero difference of paper-carrier
  elements.
-/
structure ProtocolTargetData (ctx : ProtocolTargetContext) where
  thm3 : thm3CoreAssumption ctx.bar
  arithmetic : ArithmeticObligations
    ctx.bar ctx.m ctx.r ctx.rho1 ctx.rho2
    ctx.hVec ctx.hScal
    ctx.splitScalar ctx.kSplit
    ctx.cset ctx.samples
    ctx.xs ctx.ys ctx.qVals ctx.coeffs
    ctx.xEval ctx.expectedEval
  invDeltaDiff : samplingDiffSet paperCarrier ctx.invDelta
  invDeltaNeZero : ctx.invDelta ≠ zeroRq

namespace ProtocolTargetData

/-- Recover invertibility of `ctx.invDelta` from the carried paper-facing data. -/
theorem invDeltaInvertible
  {ctx : ProtocolTargetContext}
  (h : ProtocolTargetData ctx) :
  invertibleRq ctx.invDelta :=
  invertibleRq_of_paperCarrierDiff h.invDeltaDiff h.invDeltaNeZero

/-- Recover the compact protocol-target proposition from one protocol-side data owner. -/
theorem protocolTargetProp
  {ctx : ProtocolTargetContext}
  (h : ProtocolTargetData ctx) :
  SuperNeo.protocolTargetProp ctx :=
  protocolTargetProp_of_paperCarrierDiff
    h.thm3 h.arithmetic h.invDeltaDiff h.invDeltaNeZero

/-- Recover the compatibility assumption bundle from one protocol-side data owner. -/
def assumptions
  {ctx : ProtocolTargetContext}
  (h : ProtocolTargetData ctx) :
  ProtocolTargetAssumptions ctx :=
  ProtocolTargetAssumptions.ofPaperCarrierDiff
    h.thm3 h.arithmetic h.invDeltaDiff h.invDeltaNeZero

/--
Canonical protocol-side target data from the active paper-carrier-difference
route.
-/
def ofPaperCarrierDiff
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
  ProtocolTargetData ctx :=
  { thm3 := hThm3
    arithmetic := hArithmetic
    invDeltaDiff := hDiff
    invDeltaNeZero := hNe }

/--
Canonical protocol-side target data from the finite basis-kernel
characterization of Theorem 3 on the active paper-carrier-difference route.
-/
def ofBasisKernelAssumption
  {ctx : ProtocolTargetContext}
  (hBasis : thm3BasisKernelAssumption ctx.bar)
  (hArithmetic : ArithmeticObligations
    ctx.bar ctx.m ctx.r ctx.rho1 ctx.rho2
    ctx.hVec ctx.hScal
    ctx.splitScalar ctx.kSplit
    ctx.cset ctx.samples
    ctx.xs ctx.ys ctx.qVals ctx.coeffs
    ctx.xEval ctx.expectedEval)
  (hDiff : samplingDiffSet paperCarrier ctx.invDelta)
  (hNe : ctx.invDelta ≠ zeroRq) :
  ProtocolTargetData ctx :=
  ofPaperCarrierDiff
    (thm3CoreAssumption_of_basisKernelAssumption hBasis)
    hArithmetic
    hDiff
    hNe

/--
Canonical protocol-side target data from the executable finite basis-kernel
checker on the active paper-carrier-difference route.
-/
def ofBasisKernelCheck
  {ctx : ProtocolTargetContext}
  (hCheck : thm3BasisKernelCheck ctx.bar = true)
  (hArithmetic : ArithmeticObligations
    ctx.bar ctx.m ctx.r ctx.rho1 ctx.rho2
    ctx.hVec ctx.hScal
    ctx.splitScalar ctx.kSplit
    ctx.cset ctx.samples
    ctx.xs ctx.ys ctx.qVals ctx.coeffs
    ctx.xEval ctx.expectedEval)
  (hDiff : samplingDiffSet paperCarrier ctx.invDelta)
  (hNe : ctx.invDelta ≠ zeroRq) :
  ProtocolTargetData ctx :=
  ofBasisKernelAssumption
    (thm3BasisKernelAssumption_of_check hCheck)
    hArithmetic
    hDiff
    hNe

/--
Canonical protocol-side target data from the active native-bar
paper-carrier-difference route.
-/
def ofNativePaperCarrierDiff
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
  ProtocolTargetData ctx :=
  ofPaperCarrierDiff
    (by simpa [hBarNative] using thm3CoreAssumption_native)
    hArithmetic
    hDiff
    hNe

end ProtocolTargetData

/-- Derive the compact protocol-target proposition from one protocol-side data owner. -/
theorem protocolTargetProp_of_data
  {ctx : ProtocolTargetContext}
  (h : ProtocolTargetData ctx) :
  protocolTargetProp ctx :=
  h.protocolTargetProp

end SuperNeo
