import SuperNeo.ProtocolSection71Data

/-!
Strong interactive-reduction step `Π_CCS`.
-/

namespace SuperNeo

/-- Assumptions consumed by the `Π_CCS` reduction step. -/
abbrev PiCCSAssumptions (ctx : ProtocolTargetContext) :=
  ProtocolTargetAssumptions ctx

/-- Native assumptions consumed by the `Π_CCS` reduction step. -/
abbrev PiCCSNativeAssumptions (ctx : ProtocolTargetContext) :=
  ProtocolTargetNativeAssumptions ctx

/-- Strong `Π_CCS` target statement. -/
def piCCSStrongStatement (ctx : ProtocolTargetContext) : Prop :=
  ceRelation ctx ∧
  SumCheckClaimTrue (sumcheckInstanceOfContext ctx)

/-- Derive the strong `Π_CCS` statement directly from the CE relation. -/
theorem piCCSStrong_of_ce
  {ctx : ProtocolTargetContext}
  (hCE : ceRelation ctx) :
  piCCSStrongStatement ctx := by
  exact ⟨hCE, ceClaimTrue_of_ce hCE⟩

/--
Derive the strong `Π_CCS` statement from an explicit Section 7.1 proof-system
CE realization of the compact protocol context.
-/
theorem piCCSStrong_of_section71_ce
  {ctx : ProtocolTargetContext}
  (hReal : ProtocolSection71Realization ctx)
  (hCE :
    SuperNeo.ProofSystem.ConstraintSystem.CE.Holds
      hReal.ce hReal.ceStatement hReal.ceWitness) :
  piCCSStrongStatement ctx := by
  exact piCCSStrong_of_ce (hReal.relation_of_ceHolds hCE)

/--
Derive the strong `Π_CCS` statement directly from one concrete Section 7.1
provider bundle.
-/
theorem piCCSStrong_of_section71Provider
  {ctx : ProtocolTargetContext}
  (hProvider : ProtocolSection71Provider ctx) :
  piCCSStrongStatement ctx := by
  exact piCCSStrong_of_ce (ceRelation_of_section71Provider hProvider)

/--
Derive the strong `Π_CCS` statement from one generic proof-system Section 7.1
theorem instance plus its compact specialization package.
-/
theorem piCCSStrong_of_section71Specialization
  {ctx : ProtocolTargetContext}
  {Commitment : Type}
  (hInst : SuperNeo.ProofSystem.ConstraintSystem.Section71Instance Commitment)
  (hSpec : ProtocolSection71Specialization ctx hInst) :
  piCCSStrongStatement ctx := by
  exact piCCSStrong_of_section71Provider
    (ProtocolSection71Provider.ofSpecialization hInst hSpec)

/--
Derive the strong `Π_CCS` statement directly from one theorem-native Section 7.1
setup bundle.
-/
theorem piCCSStrong_of_section71Setup
  {ctx : ProtocolTargetContext}
  (hSetup : ProtocolSection71Setup ctx) :
  piCCSStrongStatement ctx := by
  exact piCCSStrong_of_section71Provider
    (ProtocolSection71Provider.ofSetup hSetup)

/--
Derive the strong `Π_CCS` statement directly from one paper-faithful Section
7.1 theorem instance.
-/
theorem piCCSStrong_of_section71TheoremInstance
  {ctx : ProtocolTargetContext}
  (hInst : ProtocolSection71TheoremInstance ctx) :
  piCCSStrongStatement ctx := by
  exact piCCSStrong_of_section71Provider hInst.provider

/--
Derive the strong `Π_CCS` statement directly from one theorem-native Section 7.1
context object.
-/
theorem piCCSStrong_of_section71Context
  (hCtx : ProtocolSection71Context) :
  piCCSStrongStatement hCtx.target := by
  exact piCCSStrong_of_section71TheoremInstance hCtx.theoremInstance

/--
Derive the strong `Π_CCS` statement directly from one protocol-side Section 7.1
data package.
-/
theorem piCCSStrong_of_section71Data
  {ctx : ProtocolTargetContext}
  (hData : ProtocolSection71Data ctx) :
  piCCSStrongStatement ctx := by
  exact piCCSStrong_of_section71TheoremInstance hData.theoremInstance

/--
Derive the strong `Π_CCS` statement directly from one protocol-side Section 7.5
target-data owner and one accepted transition witness.
-/
theorem piCCSStrong_of_protocolTargetData
  {ctx : ProtocolTargetContext}
  (hTarget : ProtocolTargetData ctx)
  (hWitness : SumCheckTransitionWitness ctx) :
  piCCSStrongStatement ctx := by
  exact piCCSStrong_of_ce
    (ceRelation_of_protocolTargetData hTarget hWitness)

/-- Derive strong `Π_CCS` directly from CCS relation and a transition witness. -/
theorem piCCSStrong_of_ccsRelation
  {ctx : ProtocolTargetContext}
  (hCCS : ccsRelation ctx)
  (hWitness : SumCheckTransitionWitness ctx) :
  piCCSStrongStatement ctx := by
  exact piCCSStrong_of_ce (ceRelation_of_ccsRelation hCCS hWitness)

/--
Derive strong `Π_CCS` directly on the active paper-carrier-difference route.
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
  piCCSStrongStatement ctx := by
  exact piCCSStrong_of_ccsRelation
    (ccsRelation_of_paperCarrierDiff hThm3 hArithmetic hDiff hNe)
    hWitness

/--
Derive strong `Π_CCS` directly on the active paper-carrier-difference route
from the finite basis-kernel characterization of Theorem 3.
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
  piCCSStrongStatement ctx := by
  exact piCCSStrong_of_ccsRelation
    (ccsRelation_of_basisKernelAssumption hBasis hArithmetic hDiff hNe)
    hWitness

/--
Derive strong `Π_CCS` directly on the active paper-carrier-difference route
from the executable finite basis-kernel checker for Theorem 3.
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
  piCCSStrongStatement ctx := by
  exact piCCSStrong_of_basisKernelAssumption
    (thm3BasisKernelAssumption_of_check hCheck)
    hArithmetic
    hDiff
    hNe
    hWitness

/--
Derive strong `Π_CCS` directly on the active native-bar
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
  piCCSStrongStatement ctx := by
  exact piCCSStrong_of_ccsRelation
    (ccsRelation_of_native_paperCarrierDiff hBarNative hArithmetic hDiff hNe)
    hWitness

/-- Derive strong `Π_CCS` statement from relation assumptions and transcript witness. -/
theorem piCCSStrong_of_assumptions
  {ctx : ProtocolTargetContext}
  (h : PiCCSAssumptions ctx)
  (hWitness : SumCheckTransitionWitness ctx) :
  piCCSStrongStatement ctx := by
  have hCCS : ccsRelation ctx := protocolTargetProp_of_assumptions h
  exact piCCSStrong_of_ce (ceRelation_of_ccsRelation hCCS hWitness)

/-- Derive strong `Π_CCS` statement from native relation assumptions. -/
theorem piCCSStrong_of_native_assumptions
  {ctx : ProtocolTargetContext}
  (h : PiCCSNativeAssumptions ctx)
  (hWitness : SumCheckTransitionWitness ctx) :
  piCCSStrongStatement ctx := by
  have hCCS : ccsRelation ctx := protocolTargetProp_of_native_assumptions h
  exact piCCSStrong_of_ce (ceRelation_of_ccsRelation hCCS hWitness)

end SuperNeo
