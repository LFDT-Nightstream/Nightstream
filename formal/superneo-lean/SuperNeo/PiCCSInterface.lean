import SuperNeo.PiCCS

/-!
Contract interface for `SuperNeo.PiCCS`.

Spec: ./formal/superneo-lean/specs/PiCCS.spec.md

Paper anchors (Source: ./formal/superneo-lean/SuperNeo.pdf.md):
- Section 7.3 (Π_CCS), lines 481-548.
- Lemma 3 (Π_CCS is strong), lines 545-546.
-/

namespace SuperNeo

namespace PiCCSInterface

/-! ## Core Surfaces -/

/-- [Role: Theorem-Target] Curated re-export of `piCCSStrongStatement`. -/
abbrev piCCSStrongStatement := SuperNeo.piCCSStrongStatement

/-- [Role: Theorem-Target] Strong `Π_CCS` follows directly from the CE relation. -/
theorem piCCSStrong_of_ce
  {ctx : ProtocolTargetContext} :
  ceRelation ctx →
  piCCSStrongStatement ctx :=
  SuperNeo.piCCSStrong_of_ce

/--
[Role: Theorem-Target] Strong `Π_CCS` from an explicit Section 7.1 proof-system
CE realization.
-/
theorem piCCSStrong_of_section71_ce
  {ctx : ProtocolTargetContext}
  (hReal : ProtocolSection71Realization ctx) :
  SuperNeo.ProofSystem.ConstraintSystem.CE.Holds
      hReal.ce hReal.ceStatement hReal.ceWitness →
    piCCSStrongStatement ctx :=
  SuperNeo.piCCSStrong_of_section71_ce hReal

/-- [Role: Theorem-Target] Strong `Π_CCS` from one concrete Section 7.1 provider bundle. -/
theorem piCCSStrong_of_section71Provider
  {ctx : ProtocolTargetContext} :
  ProtocolSection71Provider ctx →
  piCCSStrongStatement ctx :=
  SuperNeo.piCCSStrong_of_section71Provider

/-- [Role: Theorem-Target] Strong `Π_CCS` from one generic Section 7.1 theorem instance plus compact specialization. -/
theorem piCCSStrong_of_section71Specialization
  {ctx : ProtocolTargetContext}
  {Commitment : Type}
  (hInst : SuperNeo.ProofSystem.ConstraintSystem.Section71Instance Commitment) :
  ProtocolSection71Specialization ctx hInst →
  piCCSStrongStatement ctx :=
  SuperNeo.piCCSStrong_of_section71Specialization hInst

/-- [Role: Theorem-Target] Strong `Π_CCS` from one theorem-native Section 7.1 setup. -/
theorem piCCSStrong_of_section71Setup
  {ctx : ProtocolTargetContext} :
  ProtocolSection71Setup ctx →
  piCCSStrongStatement ctx :=
  SuperNeo.piCCSStrong_of_section71Setup

/-- [Role: Theorem-Target] Strong `Π_CCS` from one paper-faithful Section 7.1 theorem instance. -/
theorem piCCSStrong_of_section71TheoremInstance
  {ctx : ProtocolTargetContext} :
  ProtocolSection71TheoremInstance ctx →
  piCCSStrongStatement ctx :=
  SuperNeo.piCCSStrong_of_section71TheoremInstance

/-- [Role: Theorem-Target] Strong `Π_CCS` from one theorem-native Section 7.1 context object. -/
theorem piCCSStrong_of_section71Context :
  (hCtx : ProtocolSection71Context) →
  piCCSStrongStatement hCtx.target :=
  SuperNeo.piCCSStrong_of_section71Context

/-- [Role: Theorem-Target] Strong `Π_CCS` from one protocol-side Section 7.1 data package. -/
theorem piCCSStrong_of_section71Data
  {ctx : ProtocolTargetContext} :
  ProtocolSection71Data ctx →
  piCCSStrongStatement ctx :=
  SuperNeo.piCCSStrong_of_section71Data

/-- [Role: Theorem-Target] Strong `Π_CCS` from one protocol-side Section 7.5 target-data owner and a transition witness. -/
theorem piCCSStrong_of_protocolTargetData
  {ctx : ProtocolTargetContext} :
  ProtocolTargetData ctx →
  SumCheckTransitionWitness ctx →
  piCCSStrongStatement ctx :=
  SuperNeo.piCCSStrong_of_protocolTargetData

/-- [Role: Theorem-Target] Strong `Π_CCS` from CCS relation and a transition witness. -/
theorem piCCSStrong_of_ccsRelation
  {ctx : ProtocolTargetContext} :
  ccsRelation ctx →
  SumCheckTransitionWitness ctx →
  piCCSStrongStatement ctx :=
  SuperNeo.piCCSStrong_of_ccsRelation

/--
[Role: Theorem-Target] Strong `Π_CCS` directly on the active
paper-carrier-difference route.
-/
theorem piCCSStrong_of_paperCarrierDiff
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
  piCCSStrongStatement ctx :=
  SuperNeo.piCCSStrong_of_paperCarrierDiff
    hThm3 hArithmetic hDiff hNe hWitness

/--
[Role: Theorem-Target] Strong `Π_CCS` directly on the active
paper-carrier-difference route from the finite basis-kernel characterization
of Theorem 3.
-/
theorem piCCSStrong_of_basisKernelAssumption
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
  piCCSStrongStatement ctx :=
  SuperNeo.piCCSStrong_of_basisKernelAssumption
    hBasis hArithmetic hDiff hNe hWitness

/--
[Role: Theorem-Target] Strong `Π_CCS` directly on the active
paper-carrier-difference route from the executable finite basis-kernel
checker for Theorem 3.
-/
theorem piCCSStrong_of_basisKernelCheck
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
  piCCSStrongStatement ctx :=
  SuperNeo.piCCSStrong_of_basisKernelCheck
    hCheck hArithmetic hDiff hNe hWitness

/--
[Role: Theorem-Target] Strong `Π_CCS` directly on the active native-bar
paper-carrier-difference route.
-/
theorem piCCSStrong_of_native_paperCarrierDiff
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
  piCCSStrongStatement ctx :=
  SuperNeo.piCCSStrong_of_native_paperCarrierDiff
    hBarNative hArithmetic hDiff hNe hWitness

/-! ## Boundary Surfaces -/

/-- [Role: Boundary] Boundary surface `PiCCSAssumptions` requiring closure. -/
abbrev PiCCSAssumptions := SuperNeo.PiCCSAssumptions

/-- [Role: Theorem-Target] Canonical strong `Π_CCS` constructor from assumptions and witness. -/
theorem piCCSStrong_of_assumptions
  {ctx : ProtocolTargetContext} :
  PiCCSAssumptions ctx →
  SumCheckTransitionWitness ctx →
  piCCSStrongStatement ctx :=
  SuperNeo.piCCSStrong_of_assumptions

end PiCCSInterface

end SuperNeo
