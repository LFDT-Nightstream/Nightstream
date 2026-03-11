import SuperNeo.PiRLC

/-!
Contract interface for `SuperNeo.PiRLC`.

Spec: ./formal/superneo-lean/specs/PiRLC.spec.md

Paper anchors (Source: ./formal/superneo-lean/SuperNeo.pdf.md):
- Section 7.4 (Π_RLC), lines 549-583.
- Lemma 4 (Π_RLC is weak), lines 582-583.
-/

namespace SuperNeo

namespace PiRLCInterface

/-! ## Core Surfaces -/

/-- [Role: Theorem-Target] Curated re-export of `piRLCWeakStatement`. -/
abbrev piRLCWeakStatement := SuperNeo.piRLCWeakStatement

/-- [Role: Theorem-Target] Weak `Π_RLC` follows directly from the CE relation. -/
theorem piRLCWeak_of_ce
  {ctx : ProtocolTargetContext} :
  ceRelation ctx →
  piRLCWeakStatement ctx :=
  SuperNeo.piRLCWeak_of_ce

/--
[Role: Theorem-Target] Weak `Π_RLC` from an explicit Section 7.1 proof-system
CE realization.
-/
theorem piRLCWeak_of_section71_ce
  {ctx : ProtocolTargetContext}
  (hReal : ProtocolSection71Realization ctx) :
  SuperNeo.ProofSystem.ConstraintSystem.CE.Holds
      hReal.ce hReal.ceStatement hReal.ceWitness →
    piRLCWeakStatement ctx :=
  SuperNeo.piRLCWeak_of_section71_ce hReal

/-- [Role: Theorem-Target] Weak `Π_RLC` from one concrete Section 7.1 provider bundle. -/
theorem piRLCWeak_of_section71Provider
  {ctx : ProtocolTargetContext} :
  ProtocolSection71Provider ctx →
  piRLCWeakStatement ctx :=
  SuperNeo.piRLCWeak_of_section71Provider

/-- [Role: Theorem-Target] Weak `Π_RLC` from one generic Section 7.1 theorem instance plus compact specialization. -/
theorem piRLCWeak_of_section71Specialization
  {ctx : ProtocolTargetContext}
  {Commitment : Type}
  (hInst : SuperNeo.ProofSystem.ConstraintSystem.Section71Instance Commitment) :
  ProtocolSection71Specialization ctx hInst →
  piRLCWeakStatement ctx :=
  SuperNeo.piRLCWeak_of_section71Specialization hInst

/-- [Role: Theorem-Target] Weak `Π_RLC` from one theorem-native Section 7.1 setup. -/
theorem piRLCWeak_of_section71Setup
  {ctx : ProtocolTargetContext} :
  ProtocolSection71Setup ctx →
  piRLCWeakStatement ctx :=
  SuperNeo.piRLCWeak_of_section71Setup

/-- [Role: Theorem-Target] Weak `Π_RLC` from one paper-faithful Section 7.1 theorem instance. -/
theorem piRLCWeak_of_section71TheoremInstance
  {ctx : ProtocolTargetContext} :
  ProtocolSection71TheoremInstance ctx →
  piRLCWeakStatement ctx :=
  SuperNeo.piRLCWeak_of_section71TheoremInstance

/-- [Role: Theorem-Target] Weak `Π_RLC` from one theorem-native Section 7.1 context object. -/
theorem piRLCWeak_of_section71Context :
  (hCtx : ProtocolSection71Context) →
  piRLCWeakStatement hCtx.target :=
  SuperNeo.piRLCWeak_of_section71Context

/-- [Role: Theorem-Target] Weak `Π_RLC` from one protocol-side Section 7.1 data package. -/
theorem piRLCWeak_of_section71Data
  {ctx : ProtocolTargetContext} :
  ProtocolSection71Data ctx →
  piRLCWeakStatement ctx :=
  SuperNeo.piRLCWeak_of_section71Data

/-- [Role: Theorem-Target] Weak `Π_RLC` from one protocol-side Section 7.5 target-data owner and a transition witness. -/
theorem piRLCWeak_of_protocolTargetData
  {ctx : ProtocolTargetContext} :
  ProtocolTargetData ctx →
  SumCheckTransitionWitness ctx →
  piRLCWeakStatement ctx :=
  SuperNeo.piRLCWeak_of_protocolTargetData

/-- [Role: Theorem-Target] Weak `Π_RLC` from CCS relation and a transition witness. -/
theorem piRLCWeak_of_ccsRelation
  {ctx : ProtocolTargetContext} :
  ccsRelation ctx →
  SumCheckTransitionWitness ctx →
  piRLCWeakStatement ctx :=
  SuperNeo.piRLCWeak_of_ccsRelation

/-- [Role: Theorem-Target] Weak `Π_RLC` directly on the active paper-carrier-difference route. -/
theorem piRLCWeak_of_paperCarrierDiff
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
  (hNe : ctx.invDelta ≠ zeroRq)
  (hWitness : SumCheckTransitionWitness ctx) :
  piRLCWeakStatement ctx :=
  SuperNeo.piRLCWeak_of_paperCarrierDiff
    hThm3 hArithmetic hDiff hNe hWitness

/--
[Role: Theorem-Target] Weak `Π_RLC` directly on the active
paper-carrier-difference route from the finite basis-kernel characterization
of Theorem 3.
-/
theorem piRLCWeak_of_basisKernelAssumption
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
  (hNe : ctx.invDelta ≠ zeroRq)
  (hWitness : SumCheckTransitionWitness ctx) :
  piRLCWeakStatement ctx :=
  SuperNeo.piRLCWeak_of_basisKernelAssumption
    hBasis hArithmetic hDiff hNe hWitness

/--
[Role: Theorem-Target] Weak `Π_RLC` directly on the active
paper-carrier-difference route from the executable finite basis-kernel
checker for Theorem 3.
-/
theorem piRLCWeak_of_basisKernelCheck
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
  (hNe : ctx.invDelta ≠ zeroRq)
  (hWitness : SumCheckTransitionWitness ctx) :
  piRLCWeakStatement ctx :=
  SuperNeo.piRLCWeak_of_basisKernelCheck
    hCheck hArithmetic hDiff hNe hWitness

/-- [Role: Theorem-Target] Weak `Π_RLC` directly on the active native paper-carrier-difference route. -/
theorem piRLCWeak_of_native_paperCarrierDiff
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
  (hNe : ctx.invDelta ≠ zeroRq)
  (hWitness : SumCheckTransitionWitness ctx) :
  piRLCWeakStatement ctx :=
  SuperNeo.piRLCWeak_of_native_paperCarrierDiff
    hBarNative hArithmetic hDiff hNe hWitness

/-! ## Boundary Surfaces -/

/-- [Role: Boundary] Boundary surface `PiRLCAssumptions` requiring closure. -/
abbrev PiRLCAssumptions := SuperNeo.PiRLCAssumptions

/-- [Role: Theorem-Target] Curated theorem surface `piRLCWeak_of_assumptions`. -/
theorem piRLCWeak_of_assumptions
  {ctx : ProtocolTargetContext} :
  PiRLCAssumptions ctx →
  SumCheckTransitionWitness ctx →
  piRLCWeakStatement ctx :=
  SuperNeo.piRLCWeak_of_assumptions

end PiRLCInterface

end SuperNeo
