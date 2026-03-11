import SuperNeo.InteractiveReductions

/-!
Contract interface for `SuperNeo.InteractiveReductions`.

Spec: `specs/InteractiveReductions.spec.md`

Paper anchors:
- Theorem 6 (Strong-Weak Composition), Section 6, lines 438-447.
- Definition 9 (Weak Interactive Reductions), lines 404-416.
- Definition 10 (Strong Interactive Reductions), lines 418-436.
-/

namespace SuperNeo

namespace InteractiveReductionsInterface

/-! ## Core Surfaces -/

/-- [Role: Theorem-Target] Curated re-export of `strongCompositionStatement`. -/
abbrev strongCompositionStatement := SuperNeo.strongCompositionStatement

/-- [Role: Theorem-Target] Curated re-export of `weakCompositionStatement`. -/
abbrev weakCompositionStatement := SuperNeo.weakCompositionStatement

/-! ## Boundary Surfaces -/

/-- [Role: Boundary] Boundary surface `InteractiveReductionAssumptions` requiring closure. -/
abbrev InteractiveReductionAssumptions := SuperNeo.InteractiveReductionAssumptions

/--
[Role: Theorem-Target] Canonical constructor from protocol-relations assumptions
and a SumCheck transition witness.
-/
abbrev InteractiveReductionAssumptions_ofProtocolRelations
  {ctx : SuperNeo.ProtocolTargetContext} :=
  SuperNeo.InteractiveReductionAssumptions.ofProtocolRelations (ctx := ctx)

/--
[Role: Theorem-Target] Canonical constructor from one explicit protocol-side
Section 7.5 target-data owner and a SumCheck transition witness.
-/
abbrev InteractiveReductionAssumptions_ofProtocolTargetData
  {ctx : SuperNeo.ProtocolTargetContext} :=
  SuperNeo.InteractiveReductionAssumptions.ofProtocolTargetData (ctx := ctx)

/--
[Role: Theorem-Target] Canonical constructor from the paper-facing
challenge-difference route plus a SumCheck transition witness.
-/
abbrev InteractiveReductionAssumptions_ofPaperCarrierDiff
  {ctx : SuperNeo.ProtocolTargetContext} :=
  SuperNeo.InteractiveReductionAssumptions.ofPaperCarrierDiff (ctx := ctx)

/--
[Role: Theorem-Target] Canonical constructor from the finite basis-kernel
characterization of Theorem 3 on the active paper-facing route plus a SumCheck
transition witness.
-/
abbrev InteractiveReductionAssumptions_ofBasisKernelAssumption
  {ctx : SuperNeo.ProtocolTargetContext} :=
  SuperNeo.InteractiveReductionAssumptions.ofBasisKernelAssumption (ctx := ctx)

/--
[Role: Theorem-Target] Canonical constructor from the executable finite
basis-kernel checker on the active paper-facing route plus a SumCheck
transition witness.
-/
abbrev InteractiveReductionAssumptions_ofBasisKernelCheck
  {ctx : SuperNeo.ProtocolTargetContext} :=
  SuperNeo.InteractiveReductionAssumptions.ofBasisKernelCheck (ctx := ctx)

/--
[Role: Theorem-Target] Canonical constructor from the active native-bar
paper-facing challenge-difference route plus a SumCheck transition witness.
-/
abbrev InteractiveReductionAssumptions_ofNativePaperCarrierDiff
  {ctx : SuperNeo.ProtocolTargetContext} :=
  SuperNeo.InteractiveReductionAssumptions.ofNativePaperCarrierDiff (ctx := ctx)

/--
[Role: Theorem-Target] Canonical constructor from a stronger strict low-norm
invertibility theorem with threshold at least `5`, plus a SumCheck transition
witness.
-/
def InteractiveReductionAssumptions_ofLowNormAtLeastFive
  {ctx : SuperNeo.ProtocolTargetContext}
  {B : Nat}
  (hFive : 5 ≤ B)
  (hThm3 : thm3CoreAssumption ctx.bar)
  (hArithmetic : ArithmeticObligations
    ctx.bar ctx.m ctx.r ctx.rho1 ctx.rho2
    ctx.hVec ctx.hScal
    ctx.splitScalar ctx.kSplit
    ctx.cset ctx.samples
    ctx.xs ctx.ys ctx.qVals ctx.coeffs
    ctx.xEval ctx.expectedEval)
  (hInv : lowNormInvertibilityAssumption B)
  (hDiff : samplingDiffSet paperCarrier ctx.invDelta)
  (hNe : ctx.invDelta ≠ zeroRq)
  (hWitness : SumCheckTransitionWitness ctx) :
  InteractiveReductionAssumptions ctx :=
  SuperNeo.InteractiveReductionAssumptions.ofLowNormAtLeastFive
    (ctx := ctx) hFive hThm3 hArithmetic hInv hDiff hNe hWitness

/--
[Role: Theorem-Target] Canonical native constructor from native
protocol-relations assumptions and a SumCheck transition witness.
-/
abbrev InteractiveReductionNativeAssumptions_ofProtocolRelations
  {ctx : SuperNeo.ProtocolTargetContext} :=
  SuperNeo.InteractiveReductionNativeAssumptions.ofProtocolRelations (ctx := ctx)

/--
[Role: Theorem-Target] Canonical native constructor from the paper-facing
challenge-difference route plus a SumCheck transition witness.
-/
abbrev InteractiveReductionNativeAssumptions_ofPaperCarrierDiff
  {ctx : SuperNeo.ProtocolTargetContext} :=
  SuperNeo.InteractiveReductionNativeAssumptions.ofPaperCarrierDiff (ctx := ctx)

/--
[Role: Theorem-Target] Canonical native constructor from a stronger strict
low-norm invertibility theorem with threshold at least `5`, plus a SumCheck
transition witness.
-/
def InteractiveReductionNativeAssumptions_ofLowNormAtLeastFive
  {ctx : SuperNeo.ProtocolTargetContext}
  {B : Nat}
  (hFive : 5 ≤ B)
  (hBarNative : ctx.bar = nativeBarMatrix)
  (hArithmetic : ArithmeticObligations
    ctx.bar ctx.m ctx.r ctx.rho1 ctx.rho2
    ctx.hVec ctx.hScal
    ctx.splitScalar ctx.kSplit
    ctx.cset ctx.samples
    ctx.xs ctx.ys ctx.qVals ctx.coeffs
    ctx.xEval ctx.expectedEval)
  (hInv : lowNormInvertibilityAssumption B)
  (hDiff : samplingDiffSet paperCarrier ctx.invDelta)
  (hNe : ctx.invDelta ≠ zeroRq)
  (hWitness : SumCheckTransitionWitness ctx) :
  InteractiveReductionNativeAssumptions ctx :=
  SuperNeo.InteractiveReductionNativeAssumptions.ofLowNormAtLeastFive
    (ctx := ctx) hFive hBarNative hArithmetic hInv hDiff hNe hWitness

/-- [Role: Theorem-Target] Curated theorem surface `strongComposition_of_assumptions`. -/
abbrev strongComposition_of_assumptions
  {ctx : SuperNeo.ProtocolTargetContext} :=
  SuperNeo.strongComposition_of_assumptions (ctx := ctx)

/--
[Role: Theorem-Target] Strong composition from an explicit realized Section 7.1
CE instance.
-/
theorem strongComposition_of_section71_ce
  {ctx : SuperNeo.ProtocolTargetContext}
  (hReal : ProtocolSection71Realization ctx) :
  SuperNeo.ProofSystem.ConstraintSystem.CE.Holds
      hReal.ce hReal.ceStatement hReal.ceWitness →
    strongCompositionStatement ctx :=
  SuperNeo.strongComposition_of_section71_ce hReal

/-- [Role: Theorem-Target] Strong composition from one concrete Section 7.1 provider bundle. -/
theorem strongComposition_of_section71Provider
  {ctx : SuperNeo.ProtocolTargetContext} :
  ProtocolSection71Provider ctx →
  strongCompositionStatement ctx :=
  SuperNeo.strongComposition_of_section71Provider

/-- [Role: Theorem-Target] Strong composition from one generic Section 7.1 theorem instance plus compact specialization. -/
theorem strongComposition_of_section71Specialization
  {ctx : SuperNeo.ProtocolTargetContext}
  {Commitment : Type}
  (hInst : SuperNeo.ProofSystem.ConstraintSystem.Section71Instance Commitment) :
  ProtocolSection71Specialization ctx hInst →
  strongCompositionStatement ctx :=
  SuperNeo.strongComposition_of_section71Specialization hInst

/-- [Role: Theorem-Target] Strong composition from one theorem-native Section 7.1 setup. -/
theorem strongComposition_of_section71Setup
  {ctx : SuperNeo.ProtocolTargetContext} :
  ProtocolSection71Setup ctx →
  strongCompositionStatement ctx :=
  SuperNeo.strongComposition_of_section71Setup

/-- [Role: Theorem-Target] Strong composition from one paper-faithful Section 7.1 theorem instance. -/
theorem strongComposition_of_section71TheoremInstance
  {ctx : SuperNeo.ProtocolTargetContext} :
  ProtocolSection71TheoremInstance ctx →
  strongCompositionStatement ctx :=
  SuperNeo.strongComposition_of_section71TheoremInstance

/-- [Role: Theorem-Target] Strong composition from one theorem-native Section 7.1 context object. -/
theorem strongComposition_of_section71Context :
  (hCtx : ProtocolSection71Context) →
  strongCompositionStatement hCtx.target :=
  SuperNeo.strongComposition_of_section71Context

/-- [Role: Theorem-Target] Strong composition from one protocol-side Section 7.1 data package. -/
theorem strongComposition_of_section71Data
  {ctx : SuperNeo.ProtocolTargetContext} :
  ProtocolSection71Data ctx →
  strongCompositionStatement ctx :=
  SuperNeo.strongComposition_of_section71Data

/-- [Role: Theorem-Target] Strong composition from one protocol-side Section 7.5 target-data owner and a transition witness. -/
theorem strongComposition_of_protocolTargetData
  {ctx : SuperNeo.ProtocolTargetContext} :
  ProtocolTargetData ctx →
  SumCheckTransitionWitness ctx →
  strongCompositionStatement ctx :=
  SuperNeo.strongComposition_of_protocolTargetData

/-- [Role: Theorem-Target] Curated theorem surface `weakComposition_of_assumptions`. -/
abbrev weakComposition_of_assumptions
  {ctx : SuperNeo.ProtocolTargetContext} :=
  SuperNeo.weakComposition_of_assumptions (ctx := ctx)

/--
[Role: Theorem-Target] Weak composition from an explicit realized Section 7.1
CE instance.
-/
theorem weakComposition_of_section71_ce
  {ctx : SuperNeo.ProtocolTargetContext}
  (hReal : ProtocolSection71Realization ctx) :
  SuperNeo.ProofSystem.ConstraintSystem.CE.Holds
      hReal.ce hReal.ceStatement hReal.ceWitness →
    weakCompositionStatement ctx :=
  SuperNeo.weakComposition_of_section71_ce hReal

/-- [Role: Theorem-Target] Weak composition from one concrete Section 7.1 provider bundle. -/
theorem weakComposition_of_section71Provider
  {ctx : SuperNeo.ProtocolTargetContext} :
  ProtocolSection71Provider ctx →
  weakCompositionStatement ctx :=
  SuperNeo.weakComposition_of_section71Provider

/-- [Role: Theorem-Target] Weak composition from one generic Section 7.1 theorem instance plus compact specialization. -/
theorem weakComposition_of_section71Specialization
  {ctx : SuperNeo.ProtocolTargetContext}
  {Commitment : Type}
  (hInst : SuperNeo.ProofSystem.ConstraintSystem.Section71Instance Commitment) :
  ProtocolSection71Specialization ctx hInst →
  weakCompositionStatement ctx :=
  SuperNeo.weakComposition_of_section71Specialization hInst

/-- [Role: Theorem-Target] Weak composition from one theorem-native Section 7.1 setup. -/
theorem weakComposition_of_section71Setup
  {ctx : SuperNeo.ProtocolTargetContext} :
  ProtocolSection71Setup ctx →
  weakCompositionStatement ctx :=
  SuperNeo.weakComposition_of_section71Setup

/-- [Role: Theorem-Target] Weak composition from one paper-faithful Section 7.1 theorem instance. -/
theorem weakComposition_of_section71TheoremInstance
  {ctx : SuperNeo.ProtocolTargetContext} :
  ProtocolSection71TheoremInstance ctx →
  weakCompositionStatement ctx :=
  SuperNeo.weakComposition_of_section71TheoremInstance

/-- [Role: Theorem-Target] Weak composition from one theorem-native Section 7.1 context object. -/
theorem weakComposition_of_section71Context :
  (hCtx : ProtocolSection71Context) →
  weakCompositionStatement hCtx.target :=
  SuperNeo.weakComposition_of_section71Context

/-- [Role: Theorem-Target] Weak composition from one protocol-side Section 7.1 data package. -/
theorem weakComposition_of_section71Data
  {ctx : SuperNeo.ProtocolTargetContext} :
  ProtocolSection71Data ctx →
  weakCompositionStatement ctx :=
  SuperNeo.weakComposition_of_section71Data

/-- [Role: Theorem-Target] Weak composition from one protocol-side Section 7.5 target-data owner and a transition witness. -/
theorem weakComposition_of_protocolTargetData
  {ctx : SuperNeo.ProtocolTargetContext} :
  ProtocolTargetData ctx →
  SumCheckTransitionWitness ctx →
  weakCompositionStatement ctx :=
  SuperNeo.weakComposition_of_protocolTargetData

/--
[Role: Theorem-Target] Witness-level SumCheck failure-advantage bound from
interactive-reduction assumptions.
-/
abbrev sumcheckFailureAdvantageBound_of_assumptions
  {ctx : SuperNeo.ProtocolTargetContext} :=
  SuperNeo.sumcheckFailureAdvantageBound_of_assumptions (ctx := ctx)

/--
[Role: Theorem-Target] Witness-level SumCheck failure-advantage bound from
native interactive-reduction assumptions.
-/
abbrev sumcheckFailureAdvantageBound_of_native_assumptions
  {ctx : SuperNeo.ProtocolTargetContext} :=
  SuperNeo.sumcheckFailureAdvantageBound_of_native_assumptions (ctx := ctx)

end InteractiveReductionsInterface

end SuperNeo
