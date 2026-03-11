import SuperNeo.PiRLC
import SuperNeo.ProtocolSection71Data

/-!
Reduction-of-knowledge step `Π_DEC`.
-/

namespace SuperNeo

/-- Assumptions consumed by the `Π_DEC` step. -/
abbrev PiDECAssumptions (ctx : ProtocolTargetContext) :=
  ProtocolTargetAssumptions ctx

/-- Native assumptions consumed by the `Π_DEC` step. -/
abbrev PiDECNativeAssumptions (ctx : ProtocolTargetContext) :=
  ProtocolTargetNativeAssumptions ctx

/-- Knowledge-style `Π_DEC` target statement. -/
def piDECKnowledgeStatement (ctx : ProtocolTargetContext) : Prop :=
  ∃ deltaInv : Coeffs,
    mulRq ctx.invDelta deltaInv = oneRq ∧
    ceRelaxedRelation ctx ∧
    SumCheckClaimTrue (sumcheckInstanceOfContext ctx)

/-- Derive `Π_DEC` directly from the weak `Π_RLC` statement. -/
theorem piDEC_of_weak
  {ctx : ProtocolTargetContext}
  (hWeak : piRLCWeakStatement ctx) :
  piDECKnowledgeStatement ctx := by
  have hTarget : protocolTargetProp ctx := hWeak.1
  rcases hTarget with ⟨_hThm3, _hSplit, _hEvalHom, _hVecMod, _hScalMod, _hSampling,
      _hMleSize, _hMleId, _hInterp, hInvDelta⟩
  rcases hInvDelta with ⟨deltaInv, hMul⟩
  exact ⟨deltaInv, hMul, hWeak.1, hWeak.2⟩

/-- Derive `Π_DEC` directly from the CE relation. -/
theorem piDEC_of_ce
  {ctx : ProtocolTargetContext}
  (hCE : ceRelation ctx) :
  piDECKnowledgeStatement ctx := by
  exact piDEC_of_weak (piRLCWeak_of_ce hCE)

/--
Derive the `Π_DEC` statement from an explicit Section 7.1 proof-system CE
realization of the compact protocol context.
-/
theorem piDEC_of_section71_ce
  {ctx : ProtocolTargetContext}
  (hReal : ProtocolSection71Realization ctx)
  (hCE :
    SuperNeo.ProofSystem.ConstraintSystem.CE.Holds
      hReal.ce hReal.ceStatement hReal.ceWitness) :
  piDECKnowledgeStatement ctx := by
  exact piDEC_of_ce (hReal.relation_of_ceHolds hCE)

/--
Derive `Π_DEC` directly from one concrete Section 7.1 provider bundle.
-/
theorem piDEC_of_section71Provider
  {ctx : ProtocolTargetContext}
  (hProvider : ProtocolSection71Provider ctx) :
  piDECKnowledgeStatement ctx := by
  exact piDEC_of_ce (ceRelation_of_section71Provider hProvider)

/--
Derive `Π_DEC` from one generic proof-system Section 7.1 theorem instance plus
its compact specialization package.
-/
theorem piDEC_of_section71Specialization
  {ctx : ProtocolTargetContext}
  {Commitment : Type}
  (hInst : SuperNeo.ProofSystem.ConstraintSystem.Section71Instance Commitment)
  (hSpec : ProtocolSection71Specialization ctx hInst) :
  piDECKnowledgeStatement ctx := by
  exact piDEC_of_section71Provider
    (ProtocolSection71Provider.ofSpecialization hInst hSpec)

/--
Derive `Π_DEC` directly from one theorem-native Section 7.1 setup bundle.
-/
theorem piDEC_of_section71Setup
  {ctx : ProtocolTargetContext}
  (hSetup : ProtocolSection71Setup ctx) :
  piDECKnowledgeStatement ctx := by
  exact piDEC_of_section71Provider
    (ProtocolSection71Provider.ofSetup hSetup)

/--
Derive `Π_DEC` directly from one paper-faithful Section 7.1 theorem instance.
-/
theorem piDEC_of_section71TheoremInstance
  {ctx : ProtocolTargetContext}
  (hInst : ProtocolSection71TheoremInstance ctx) :
  piDECKnowledgeStatement ctx := by
  exact piDEC_of_section71Provider hInst.provider

/--
Derive `Π_DEC` directly from one theorem-native Section 7.1 context object.
-/
theorem piDEC_of_section71Context
  (hCtx : ProtocolSection71Context) :
  piDECKnowledgeStatement hCtx.target := by
  exact piDEC_of_section71TheoremInstance hCtx.theoremInstance

/--
Derive `Π_DEC` directly from one protocol-side Section 7.1 data package.
-/
theorem piDEC_of_section71Data
  {ctx : ProtocolTargetContext}
  (hData : ProtocolSection71Data ctx) :
  piDECKnowledgeStatement ctx := by
  exact piDEC_of_section71TheoremInstance hData.theoremInstance

/--
Derive `Π_DEC` directly from one protocol-side Section 7.5 target-data owner
and one accepted transition witness.
-/
theorem piDEC_of_protocolTargetData
  {ctx : ProtocolTargetContext}
  (hTarget : ProtocolTargetData ctx)
  (hWitness : SumCheckTransitionWitness ctx) :
  piDECKnowledgeStatement ctx := by
  exact piDEC_of_ce
    (ceRelation_of_protocolTargetData hTarget hWitness)

/-- Derive `Π_DEC` directly from CCS relation and a transition witness. -/
theorem piDEC_of_ccsRelation
  {ctx : ProtocolTargetContext}
  (hCCS : ccsRelation ctx)
  (hWitness : SumCheckTransitionWitness ctx) :
  piDECKnowledgeStatement ctx := by
  exact piDEC_of_weak (piRLCWeak_of_ccsRelation hCCS hWitness)

/--
Derive `Π_DEC` directly on the active paper-carrier-difference route.
-/
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
  piDECKnowledgeStatement ctx := by
  exact piDEC_of_ccsRelation
    (ccsRelation_of_paperCarrierDiff hThm3 hArithmetic hDiff hNe)
    hWitness

/--
Derive `Π_DEC` directly on the active paper-carrier-difference route from the
finite basis-kernel characterization of Theorem 3.
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
  piDECKnowledgeStatement ctx := by
  exact piDEC_of_ccsRelation
    (ccsRelation_of_basisKernelAssumption hBasis hArithmetic hDiff hNe)
    hWitness

/--
Derive `Π_DEC` directly on the active paper-carrier-difference route from the
executable finite basis-kernel checker for Theorem 3.
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
  piDECKnowledgeStatement ctx := by
  exact piDEC_of_basisKernelAssumption
    (thm3BasisKernelAssumption_of_check hCheck)
    hArithmetic
    hDiff
    hNe
    hWitness

/--
Derive `Π_DEC` directly on the active native-bar
paper-carrier-difference route.
-/
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
  piDECKnowledgeStatement ctx := by
  exact piDEC_of_ccsRelation
    (ccsRelation_of_native_paperCarrierDiff hBarNative hArithmetic hDiff hNe)
    hWitness

/-- Derive `Π_DEC` statement from weak relation and invertibility boundary. -/
theorem piDEC_of_assumptions
  {ctx : ProtocolTargetContext}
  (h : PiDECAssumptions ctx)
  (hWitness : SumCheckTransitionWitness ctx) :
  piDECKnowledgeStatement ctx := by
  exact piDEC_of_weak (piRLCWeak_of_assumptions h hWitness)

/-- Derive `Π_DEC` statement from native weak relation and invertibility boundary. -/
theorem piDEC_of_native_assumptions
  {ctx : ProtocolTargetContext}
  (h : PiDECNativeAssumptions ctx)
  (hWitness : SumCheckTransitionWitness ctx) :
  piDECKnowledgeStatement ctx := by
  exact piDEC_of_weak (piRLCWeak_of_native_assumptions h hWitness)

end SuperNeo
