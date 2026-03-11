import SuperNeo.PiCCS
import SuperNeo.ProtocolSection71Data

/-!
Weak interactive-reduction step `Π_RLC`.
-/

namespace SuperNeo

/-- Assumptions consumed by the `Π_RLC` reduction step. -/
abbrev PiRLCAssumptions (ctx : ProtocolTargetContext) :=
  ProtocolTargetAssumptions ctx

/-- Native assumptions consumed by the `Π_RLC` reduction step. -/
abbrev PiRLCNativeAssumptions (ctx : ProtocolTargetContext) :=
  ProtocolTargetNativeAssumptions ctx

/-- Weak `Π_RLC` target statement. -/
def piRLCWeakStatement (ctx : ProtocolTargetContext) : Prop :=
  ceRelaxedRelation ctx ∧
  SumCheckClaimTrue (sumcheckInstanceOfContext ctx)

/-- Derive weak `Π_RLC` statement directly from the CE relation. -/
theorem piRLCWeak_of_ce
  {ctx : ProtocolTargetContext}
  (hCE : ceRelation ctx) :
  piRLCWeakStatement ctx := by
  exact ⟨ceRelaxedRelation_of_ce hCE, ceClaimTrue_of_ce hCE⟩

/--
Derive the weak `Π_RLC` statement from an explicit Section 7.1 proof-system
CE realization of the compact protocol context.
-/
theorem piRLCWeak_of_section71_ce
  {ctx : ProtocolTargetContext}
  (hReal : ProtocolSection71Realization ctx)
  (hCE :
    SuperNeo.ProofSystem.ConstraintSystem.CE.Holds
      hReal.ce hReal.ceStatement hReal.ceWitness) :
  piRLCWeakStatement ctx := by
  exact piRLCWeak_of_ce (hReal.relation_of_ceHolds hCE)

/--
Derive the weak `Π_RLC` statement directly from one concrete Section 7.1
provider bundle.
-/
theorem piRLCWeak_of_section71Provider
  {ctx : ProtocolTargetContext}
  (hProvider : ProtocolSection71Provider ctx) :
  piRLCWeakStatement ctx := by
  exact piRLCWeak_of_ce (ceRelation_of_section71Provider hProvider)

/--
Derive the weak `Π_RLC` statement from one generic proof-system Section 7.1
theorem instance plus its compact specialization package.
-/
theorem piRLCWeak_of_section71Specialization
  {ctx : ProtocolTargetContext}
  {Commitment : Type}
  (hInst : SuperNeo.ProofSystem.ConstraintSystem.Section71Instance Commitment)
  (hSpec : ProtocolSection71Specialization ctx hInst) :
  piRLCWeakStatement ctx := by
  exact piRLCWeak_of_section71Provider
    (ProtocolSection71Provider.ofSpecialization hInst hSpec)

/--
Derive the weak `Π_RLC` statement directly from one theorem-native Section 7.1
setup bundle.
-/
theorem piRLCWeak_of_section71Setup
  {ctx : ProtocolTargetContext}
  (hSetup : ProtocolSection71Setup ctx) :
  piRLCWeakStatement ctx := by
  exact piRLCWeak_of_section71Provider
    (ProtocolSection71Provider.ofSetup hSetup)

/--
Derive the weak `Π_RLC` statement directly from one paper-faithful Section 7.1
theorem instance.
-/
theorem piRLCWeak_of_section71TheoremInstance
  {ctx : ProtocolTargetContext}
  (hInst : ProtocolSection71TheoremInstance ctx) :
  piRLCWeakStatement ctx := by
  exact piRLCWeak_of_section71Provider hInst.provider

/--
Derive the weak `Π_RLC` statement directly from one theorem-native Section 7.1
context object.
-/
theorem piRLCWeak_of_section71Context
  (hCtx : ProtocolSection71Context) :
  piRLCWeakStatement hCtx.target := by
  exact piRLCWeak_of_section71TheoremInstance hCtx.theoremInstance

/--
Derive the weak `Π_RLC` statement directly from one protocol-side Section 7.1
data package.
-/
theorem piRLCWeak_of_section71Data
  {ctx : ProtocolTargetContext}
  (hData : ProtocolSection71Data ctx) :
  piRLCWeakStatement ctx := by
  exact piRLCWeak_of_section71TheoremInstance hData.theoremInstance

/--
Derive the weak `Π_RLC` statement directly from one protocol-side Section 7.5
target-data owner and one accepted transition witness.
-/
theorem piRLCWeak_of_protocolTargetData
  {ctx : ProtocolTargetContext}
  (hTarget : ProtocolTargetData ctx)
  (hWitness : SumCheckTransitionWitness ctx) :
  piRLCWeakStatement ctx := by
  exact piRLCWeak_of_ce
    (ceRelation_of_protocolTargetData hTarget hWitness)

/-- Derive weak `Π_RLC` statement from CCS relation and a transition witness. -/
theorem piRLCWeak_of_ccsRelation
  {ctx : ProtocolTargetContext}
  (hCCS : ccsRelation ctx)
  (hWitness : SumCheckTransitionWitness ctx) :
  piRLCWeakStatement ctx := by
  exact piRLCWeak_of_ce (ceRelation_of_ccsRelation hCCS hWitness)

/--
Derive weak `Π_RLC` statement directly on the active paper-carrier-difference
route.
-/
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
  piRLCWeakStatement ctx := by
  exact piRLCWeak_of_ccsRelation
    (ccsRelation_of_paperCarrierDiff hThm3 hArithmetic hDiff hNe)
    hWitness

/--
Derive weak `Π_RLC` directly on the active paper-carrier-difference route from
the finite basis-kernel characterization of Theorem 3.
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
  piRLCWeakStatement ctx := by
  exact piRLCWeak_of_ccsRelation
    (ccsRelation_of_basisKernelAssumption hBasis hArithmetic hDiff hNe)
    hWitness

/--
Derive weak `Π_RLC` directly on the active paper-carrier-difference route from
the executable finite basis-kernel checker for Theorem 3.
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
  piRLCWeakStatement ctx := by
  exact piRLCWeak_of_basisKernelAssumption
    (thm3BasisKernelAssumption_of_check hCheck)
    hArithmetic
    hDiff
    hNe
    hWitness

/--
Derive weak `Π_RLC` statement directly on the active native-bar
paper-carrier-difference route.
-/
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
  piRLCWeakStatement ctx := by
  exact piRLCWeak_of_ccsRelation
    (ccsRelation_of_native_paperCarrierDiff hBarNative hArithmetic hDiff hNe)
    hWitness

/-- Derive weak `Π_RLC` statement from strong `Π_CCS`. -/
theorem piRLCWeak_of_assumptions
  {ctx : ProtocolTargetContext}
  (h : PiRLCAssumptions ctx)
  (hWitness : SumCheckTransitionWitness ctx) :
  piRLCWeakStatement ctx := by
  exact piRLCWeak_of_ccsRelation
    (protocolTargetProp_of_assumptions h)
    hWitness

/-- Derive weak `Π_RLC` statement from native strong `Π_CCS`. -/
theorem piRLCWeak_of_native_assumptions
  {ctx : ProtocolTargetContext}
  (h : PiRLCNativeAssumptions ctx)
  (hWitness : SumCheckTransitionWitness ctx) :
  piRLCWeakStatement ctx := by
  exact piRLCWeak_of_ccsRelation
    (protocolTargetProp_of_native_assumptions h)
    hWitness

end SuperNeo
