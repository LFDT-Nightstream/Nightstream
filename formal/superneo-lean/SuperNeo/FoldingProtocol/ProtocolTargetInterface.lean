import SuperNeo.FoldingProtocol.ProtocolTarget

/-!
Contract interface for `SuperNeo.ProtocolTarget`.

Spec: `./formal/superneo-lean/specs/ProtocolTarget.spec.md`

Paper anchors (Source: `./formal/superneo-lean/SuperNeo.pdf.md`):
- Section 7 (Neo's folding scheme for CCS), lines 447–481: Relations (Definitions 11–13), Global Reduction Parameters (Definition 14)
- Section 7.3 (Π_CCS), lines 481–547: Interactive reduction for CCS
-/

namespace SuperNeo

namespace ProtocolTargetInterface

/-! ## Core Surfaces -/

/-- [Role: Theorem-Target] Curated re-export of `ProtocolTargetContext`. -/
abbrev ProtocolTargetContext := SuperNeo.ProtocolTargetContext

/-- [Role: Theorem-Target] Curated re-export of `protocolTargetProp`. -/
abbrev protocolTargetProp := SuperNeo.protocolTargetProp

/-! ## Boundary Surfaces -/

/-- [Role: Boundary] Boundary surface `ProtocolTargetAssumptions` requiring closure. -/
abbrev ProtocolTargetAssumptions := SuperNeo.ProtocolTargetAssumptions

/-- [Role: Theorem-Target] Re-export the theorem-native MatrixTransform constructor from Theorem 3. -/
theorem matrixTransformAssumption_of_thm3CoreAssumption
  {bar m : Array (Array F)}
  (h : thm3CoreAssumption bar) :
  matrixTransformAssumption bar m :=
  SuperNeo.matrixTransformAssumption_of_thm3CoreAssumption h

/-- [Role: Boundary] Boundary surface `protocolTargetProp_of_assumptions` requiring closure. -/
theorem protocolTargetProp_of_assumptions
  {ctx : ProtocolTargetContext}
  (h : ProtocolTargetAssumptions ctx) :
  protocolTargetProp ctx :=
  SuperNeo.protocolTargetProp_of_assumptions h

/-! ## Paper-Facing Invertibility Bridge -/

/-- [Role: Theorem-Target] Strict `< 5` window from a nonzero paper-carrier difference. -/
theorem strictInvertibilityWindowProp_five_of_paperCarrierDiff
  {δ : Coeffs}
  (hDiff : samplingDiffSet paperCarrier δ)
  (hNe : δ ≠ zeroRq) :
  strictInvertibilityWindowProp 5 δ :=
  SuperNeo.strictInvertibilityWindowProp_five_of_paperCarrierDiff hDiff hNe

/-- [Role: Theorem-Target] Derive invertibility on the active paper-carrier-difference path. -/
theorem invertibleRq_of_paperCarrierDiff
  {δ : Coeffs}
  (hDiff : samplingDiffSet paperCarrier δ)
  (hNe : δ ≠ zeroRq) :
  invertibleRq δ :=
  SuperNeo.invertibleRq_of_paperCarrierDiff hDiff hNe

/--
[Role: Theorem-Target] Canonical protocol-target constructor from the
paper-facing challenge-difference route.
-/
def ProtocolTargetAssumptions_ofPaperCarrierDiff
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
  SuperNeo.ProtocolTargetAssumptions ctx :=
  SuperNeo.ProtocolTargetAssumptions.ofPaperCarrierDiff thm3 arithmetic hDiff hNe

end ProtocolTargetInterface

end SuperNeo
