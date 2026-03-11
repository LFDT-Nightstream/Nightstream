import SuperNeo.PiDEC

/-!
Contract interface for `SuperNeo.PiDEC`.

Spec: ./formal/superneo-lean/specs/PiDEC.spec.md

Paper anchors (Source: ./formal/superneo-lean/SuperNeo.pdf.md):
- Section 7.5 (Π_DEC), lines 585-593.
- Theorem 7 (Π_DEC is reduction of knowledge), lines 594-596.
-/

namespace SuperNeo

namespace PiDECInterface

/-! ## Core Surfaces -/

/-- [Role: Theorem-Target] Curated re-export of `piDECKnowledgeStatement`. -/
abbrev piDECKnowledgeStatement := SuperNeo.piDECKnowledgeStatement

/-- [Role: Theorem-Target] Derive `Π_DEC` directly from the weak `Π_RLC` statement. -/
theorem piDEC_of_weak
  {ctx : ProtocolTargetContext} :
  piRLCWeakStatement ctx →
  piDECKnowledgeStatement ctx :=
  SuperNeo.piDEC_of_weak

/-- [Role: Theorem-Target] Derive `Π_DEC` directly from the CE relation. -/
theorem piDEC_of_ce
  {ctx : ProtocolTargetContext} :
  ceRelation ctx →
  piDECKnowledgeStatement ctx :=
  SuperNeo.piDEC_of_ce

/--
[Role: Theorem-Target] Derive `Π_DEC` from an explicit Section 7.1 proof-system
CE realization.
-/
theorem piDEC_of_section71_ce
  {ctx : ProtocolTargetContext}
  (hReal : ProtocolSection71Realization ctx) :
  SuperNeo.ProofSystem.ConstraintSystem.CE.Holds
      hReal.ce hReal.ceStatement hReal.ceWitness →
    piDECKnowledgeStatement ctx :=
  SuperNeo.piDEC_of_section71_ce hReal

/-- [Role: Theorem-Target] Derive `Π_DEC` from one concrete Section 7.1 provider bundle. -/
theorem piDEC_of_section71Provider
  {ctx : ProtocolTargetContext} :
  ProtocolSection71Provider ctx →
  piDECKnowledgeStatement ctx :=
  SuperNeo.piDEC_of_section71Provider

/-- [Role: Theorem-Target] Derive `Π_DEC` from one generic Section 7.1 theorem instance plus compact specialization. -/
theorem piDEC_of_section71Specialization
  {ctx : ProtocolTargetContext}
  {Commitment : Type}
  (hInst : SuperNeo.ProofSystem.ConstraintSystem.Section71Instance Commitment) :
  ProtocolSection71Specialization ctx hInst →
  piDECKnowledgeStatement ctx :=
  SuperNeo.piDEC_of_section71Specialization hInst

/-- [Role: Theorem-Target] Derive `Π_DEC` from one theorem-native Section 7.1 setup. -/
theorem piDEC_of_section71Setup
  {ctx : ProtocolTargetContext} :
  ProtocolSection71Setup ctx →
  piDECKnowledgeStatement ctx :=
  SuperNeo.piDEC_of_section71Setup

/-- [Role: Theorem-Target] Derive `Π_DEC` from one paper-faithful Section 7.1 theorem instance. -/
theorem piDEC_of_section71TheoremInstance
  {ctx : ProtocolTargetContext} :
  ProtocolSection71TheoremInstance ctx →
  piDECKnowledgeStatement ctx :=
  SuperNeo.piDEC_of_section71TheoremInstance

/-- [Role: Theorem-Target] Derive `Π_DEC` from one theorem-native Section 7.1 context object. -/
theorem piDEC_of_section71Context :
  (hCtx : ProtocolSection71Context) →
  piDECKnowledgeStatement hCtx.target :=
  SuperNeo.piDEC_of_section71Context

/-- [Role: Theorem-Target] Derive `Π_DEC` from one protocol-side Section 7.1 data package. -/
theorem piDEC_of_section71Data
  {ctx : ProtocolTargetContext} :
  ProtocolSection71Data ctx →
  piDECKnowledgeStatement ctx :=
  SuperNeo.piDEC_of_section71Data

/-- [Role: Theorem-Target] Derive `Π_DEC` from one protocol-side Section 7.5 target-data owner and a transition witness. -/
theorem piDEC_of_protocolTargetData
  {ctx : ProtocolTargetContext} :
  ProtocolTargetData ctx →
  SumCheckTransitionWitness ctx →
  piDECKnowledgeStatement ctx :=
  SuperNeo.piDEC_of_protocolTargetData

/-- [Role: Theorem-Target] Derive `Π_DEC` from CCS relation and a transition witness. -/
theorem piDEC_of_ccsRelation
  {ctx : ProtocolTargetContext} :
  ccsRelation ctx →
  SumCheckTransitionWitness ctx →
  piDECKnowledgeStatement ctx :=
  SuperNeo.piDEC_of_ccsRelation

/-- [Role: Theorem-Target] Derive `Π_DEC` directly on the active paper-carrier-difference route. -/
theorem piDEC_of_paperCarrierDiff
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
  piDECKnowledgeStatement ctx :=
  SuperNeo.piDEC_of_paperCarrierDiff
    hThm3 hArithmetic hDiff hNe hWitness

/--
[Role: Theorem-Target] Derive `Π_DEC` directly on the active
paper-carrier-difference route from the finite basis-kernel characterization
of Theorem 3.
-/
theorem piDEC_of_basisKernelAssumption
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
  piDECKnowledgeStatement ctx :=
  SuperNeo.piDEC_of_basisKernelAssumption
    hBasis hArithmetic hDiff hNe hWitness

/--
[Role: Theorem-Target] Derive `Π_DEC` directly on the active
paper-carrier-difference route from the executable finite basis-kernel
checker for Theorem 3.
-/
theorem piDEC_of_basisKernelCheck
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
  piDECKnowledgeStatement ctx :=
  SuperNeo.piDEC_of_basisKernelCheck
    hCheck hArithmetic hDiff hNe hWitness

/-- [Role: Theorem-Target] Derive `Π_DEC` directly on the active native paper-carrier-difference route. -/
theorem piDEC_of_native_paperCarrierDiff
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
  piDECKnowledgeStatement ctx :=
  SuperNeo.piDEC_of_native_paperCarrierDiff
    hBarNative hArithmetic hDiff hNe hWitness

/-! ## Boundary Surfaces -/

/-- [Role: Boundary] Boundary surface `PiDECAssumptions` requiring closure. -/
abbrev PiDECAssumptions := SuperNeo.PiDECAssumptions

/-- [Role: Theorem-Target] Curated theorem surface `piDEC_of_assumptions`. -/
theorem piDEC_of_assumptions
  {ctx : ProtocolTargetContext} :
  PiDECAssumptions ctx →
  SumCheckTransitionWitness ctx →
  piDECKnowledgeStatement ctx :=
  SuperNeo.piDEC_of_assumptions

end PiDECInterface

end SuperNeo
