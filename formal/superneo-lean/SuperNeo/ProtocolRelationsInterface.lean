import SuperNeo.ProtocolRelations

/-!
Contract interface for `SuperNeo.ProtocolRelations`.

Spec: ./formal/superneo-lean/specs/ProtocolRelations.spec.md

Paper anchors (Source: ./formal/superneo-lean/SuperNeo.pdf.md):
- Definition 12 (Norm-bounded CCS), Section 7.1, lines 457-459.
- Definition 13 (Norm-bounded CCS Evaluation Relation), Section 7.1, lines 461-465.
- Section 7.1 (Relations), lines 449-465.
-/

namespace SuperNeo

namespace ProtocolRelationsInterface

/-! ## Core Surfaces -/

/-- [Role: Theorem-Target] Curated re-export of `sumcheckInstanceOfContext`. -/
abbrev sumcheckInstanceOfContext := SuperNeo.sumcheckInstanceOfContext

/-- [Role: Theorem-Target] Curated re-export of `sumcheckFullFieldDenominatorAlignment`. -/
abbrev sumcheckFullFieldDenominatorAlignment :=
  SuperNeo.sumcheckFullFieldDenominatorAlignment

/-- [Role: Boundary] Named setup-side boundary for the active Goldilocks/full-field Lund route. -/
abbrev GoldilocksFullFieldLundBoundary :=
  SuperNeo.GoldilocksFullFieldLundBoundary

/-- [Role: Theorem-Target] Curated re-export of `ccsRelation`. -/
abbrev ccsRelation := SuperNeo.ccsRelation

/-- [Role: Theorem-Target] Curated re-export of `ceRelation`. -/
abbrev ceRelation := SuperNeo.ceRelation

/-- [Role: Theorem-Target] Curated re-export of `ceRelaxedRelation`. -/
abbrev ceRelaxedRelation := SuperNeo.ceRelaxedRelation

/-! ## Key Theorems -/

/-- [Role: Theorem-Target] Curated theorem surface `ceRelation_of_claimTrue`. -/
abbrev ceRelation_of_claimTrue
  {ctx : ProtocolTargetContext} :
  ProtocolRelationsAssumptions ctx →
  SumCheckClaimTrue (sumcheckInstanceOfContext ctx) →
  ceRelation ctx :=
  SuperNeo.ceRelation_of_claimTrue

/-- [Role: Theorem-Target] Curated theorem surface `ceRelation_of_ccsRelation`. -/
abbrev ceRelation_of_ccsRelation
  {ctx : ProtocolTargetContext} :
  ccsRelation ctx →
  SumCheckTransitionWitness ctx →
  ceRelation ctx :=
  SuperNeo.ceRelation_of_ccsRelation

/-- [Role: Theorem-Target] Curated theorem surface `ceRelation_of_protocolTargetData`. -/
theorem ceRelation_of_protocolTargetData
  {ctx : ProtocolTargetContext} :
  ProtocolTargetData ctx →
  SumCheckTransitionWitness ctx →
  ceRelation ctx :=
  SuperNeo.ceRelation_of_protocolTargetData

/-- [Role: Theorem-Target] Curated theorem surface `ceRelation_of_ccsRelation_claimTrue`. -/
abbrev ceRelation_of_ccsRelation_claimTrue
  {ctx : ProtocolTargetContext} :
  ccsRelation ctx →
  SumCheckClaimTrue (sumcheckInstanceOfContext ctx) →
  ceRelation ctx :=
  SuperNeo.ceRelation_of_ccsRelation_claimTrue

/-- [Role: Theorem-Target] Curated theorem surface `ceClaimTrue_of_ce`. -/
abbrev ceClaimTrue_of_ce
  {ctx : ProtocolTargetContext} :
  ceRelation ctx →
  SumCheckClaimTrue (sumcheckInstanceOfContext ctx) :=
  SuperNeo.ceClaimTrue_of_ce

/-- [Role: Theorem-Target] Curated theorem surface `ceClaimTrue_of_native_ce`. -/
abbrev ceClaimTrue_of_native_ce
  {ctx : ProtocolTargetContext} :
  ceRelation ctx →
  SumCheckClaimTrue (sumcheckInstanceOfContext ctx) :=
  SuperNeo.ceClaimTrue_of_native_ce

/-- [Role: Theorem-Target] Curated theorem surface `ceRelaxedRelation_of_ce`. -/
abbrev ceRelaxedRelation_of_ce
  {ctx : ProtocolTargetContext} :
  ceRelation ctx →
  ceRelaxedRelation ctx :=
  SuperNeo.ceRelaxedRelation_of_ce

/-! ## Boundary Surfaces -/

/-- [Role: Boundary] Boundary surface `SumCheckTransitionWitness` requiring closure. -/
abbrev SumCheckTransitionWitness := SuperNeo.SumCheckTransitionWitness

/--
[Role: Boundary] Specialization package realizing the compact protocol
relations as explicit Section 7.1 proof-system CCS/CE objects built from one
shared Definition-14 global-parameter package.
-/
abbrev ProtocolSection71Objects := SuperNeo.ProtocolSection71Objects

/--
[Role: Boundary] Shared Section 7.1 Definition-14 package plus one coherent
CCS/CE tuple pair for the current theorem instance.
-/
abbrev ProtocolSection71Realization := SuperNeo.ProtocolSection71Realization

/--
[Role: Boundary] Smallest upstream owner of one concrete Section 7.1 theorem
instance: a shared Definition-14 realization plus concrete CCS/CE membership
proofs.
-/
abbrev ProtocolSection71Provider := SuperNeo.ProtocolSection71Provider

/--
[Role: Boundary] Compact-context specialization package for one generic
proof-system Section 7.1 theorem instance.
-/
abbrev ProtocolSection71Specialization :=
  SuperNeo.ProtocolSection71Specialization

/--
[Role: Boundary] Smallest theorem-native owner of one generic Section 7.1
theorem instance together with its compact-context specialization.
-/
abbrev ProtocolSection71Setup := SuperNeo.ProtocolSection71Setup

/--
[Role: Theorem-Target] Canonical paper-faithful Section 7.1 theorem instance
specialized to the compact protocol context.
-/
abbrev ProtocolSection71TheoremInstance :=
  SuperNeo.ProtocolSection71TheoremInstance

/-- [Role: Theorem-Target] CCS relation is exactly `protocolTargetProp`. -/
abbrev ccsRelation_of_protocolTargetProp
  {ctx : ProtocolTargetContext} :
  protocolTargetProp ctx →
  ccsRelation ctx :=
  SuperNeo.ccsRelation_of_protocolTargetProp

/-- [Role: Theorem-Target] One protocol-side Section 7.5 target-data owner implies compact `ccsRelation`. -/
theorem ccsRelation_of_protocolTargetData
  {ctx : ProtocolTargetContext} :
  ProtocolTargetData ctx →
  ccsRelation ctx :=
  SuperNeo.ccsRelation_of_protocolTargetData

/-- [Role: Theorem-Target] Full-field denominator alignment is exactly `ctx.cset.size = Goldilocks.q`. -/
theorem sumcheckFullFieldDenominatorAlignment_iff
  {ctx : ProtocolTargetContext} :
  sumcheckFullFieldDenominatorAlignment ctx ↔
    ctx.cset.size = Goldilocks.q :=
  SuperNeo.sumcheckFullFieldDenominatorAlignment_iff

/-- [Role: Theorem-Target] Build the named Goldilocks/Lund setup boundary from concrete cardinality. -/
abbrev GoldilocksFullFieldLundBoundary_ofCsetCardinality
  {ctx : ProtocolTargetContext} :=
  SuperNeo.GoldilocksFullFieldLundBoundary.ofCsetCardinality (ctx := ctx)

/-- [Role: Theorem-Target] Recover concrete challenge-set cardinality from the named setup boundary. -/
theorem GoldilocksFullFieldLundBoundary_csetCardinality_eq
  {ctx : ProtocolTargetContext}
  (h : GoldilocksFullFieldLundBoundary ctx) :
  ctx.cset.size = Goldilocks.q :=
  SuperNeo.GoldilocksFullFieldLundBoundary.csetCardinality_eq h

/-- [Role: Boundary] Boundary surface `SumCheckTransitionWitness.accepted_exists` requiring closure. -/
abbrev SumCheckTransitionWitness_accepted_exists
  {ctx : ProtocolTargetContext} :
  SumCheckTransitionWitness ctx →
  ∃ tr : SumCheckTranscript, SumCheckAccepted (sumcheckInstanceOfContext ctx) tr :=
  SuperNeo.SumCheckTransitionWitness.accepted_exists

/-- [Role: Boundary] Boundary surface `ProtocolRelationsAssumptions` requiring closure. -/
abbrev ProtocolRelationsAssumptions := SuperNeo.ProtocolRelationsAssumptions

/-- [Role: Boundary] Native boundary bundle for protocol relations assumptions. -/
abbrev ProtocolRelationsNativeAssumptions := SuperNeo.ProtocolRelationsNativeAssumptions

/-- [Role: Theorem-Target] Move from compact `ccsRelation` to realized proof-system CCS membership. -/
theorem ProtocolSection71Realization_ccsHolds_of_relation
  {ctx : ProtocolTargetContext}
  (h : ProtocolSection71Realization ctx)
  (hCCS : ccsRelation ctx) :
  SuperNeo.ProofSystem.ConstraintSystem.CCS.Holds
    h.ccs h.ccsStatement h.ccsWitness :=
  SuperNeo.ProtocolSection71Realization.ccsHolds_of_relation h hCCS

/-- [Role: Theorem-Target] Move from realized proof-system CCS membership back to compact `ccsRelation`. -/
theorem ProtocolSection71Realization_relation_of_ccsHolds
  {ctx : ProtocolTargetContext}
  (h : ProtocolSection71Realization ctx)
  (hCCS :
    SuperNeo.ProofSystem.ConstraintSystem.CCS.Holds
      h.ccs h.ccsStatement h.ccsWitness) :
  ccsRelation ctx :=
  SuperNeo.ProtocolSection71Realization.relation_of_ccsHolds h hCCS

/-- [Role: Theorem-Target] Move from compact `ceRelation` to realized proof-system CE membership. -/
theorem ProtocolSection71Realization_ceHolds_of_relation
  {ctx : ProtocolTargetContext}
  (h : ProtocolSection71Realization ctx)
  (hCE : ceRelation ctx) :
  SuperNeo.ProofSystem.ConstraintSystem.CE.Holds
    h.ce h.ceStatement h.ceWitness :=
  SuperNeo.ProtocolSection71Realization.ceHolds_of_relation h hCE

/-- [Role: Theorem-Target] Move from realized proof-system CE membership back to compact `ceRelation`. -/
theorem ProtocolSection71Realization_relation_of_ceHolds
  {ctx : ProtocolTargetContext}
  (h : ProtocolSection71Realization ctx)
  (hCE :
    SuperNeo.ProofSystem.ConstraintSystem.CE.Holds
      h.ce h.ceStatement h.ceWitness) :
  ceRelation ctx :=
  SuperNeo.ProtocolSection71Realization.relation_of_ceHolds h hCE

/-- [Role: Theorem-Target] Recover the compact challenge-set from the realized Definition-14 package. -/
theorem ProtocolSection71Realization_challengeSet_eq_cset
  {ctx : ProtocolTargetContext}
  (h : ProtocolSection71Realization ctx) :
  h.params.challengeSet = ctx.cset :=
  SuperNeo.ProtocolSection71Realization.challengeSet_eq_cset h

/-- [Role: Theorem-Target] Recover that the realized CCS and CE statements share one commitment. -/
theorem ProtocolSection71Realization_sharedCommitment_eq
  {ctx : ProtocolTargetContext}
  (h : ProtocolSection71Realization ctx) :
  h.ccsStatement.commitment = h.ceStatement.commitment :=
  SuperNeo.ProtocolSection71Realization.sharedCommitment_eq h

/-- [Role: Theorem-Target] Recover that the realized CCS and CE statements share one public input. -/
theorem ProtocolSection71Realization_sharedPublicInput_eq
  {ctx : ProtocolTargetContext}
  (h : ProtocolSection71Realization ctx) :
  h.ccsStatement.publicInput = h.ceStatement.publicInput :=
  SuperNeo.ProtocolSection71Realization.sharedPublicInput_eq h

/-- [Role: Theorem-Target] Recover that the CE witness assignment is the CCS full vector `[x, w]`. -/
theorem ProtocolSection71Realization_ceAssignment_eq_fullVector
  {ctx : ProtocolTargetContext}
  (h : ProtocolSection71Realization ctx) :
  h.ceWitness.assignment =
    SuperNeo.ProofSystem.ConstraintSystem.CCS.fullVector
      h.ccsStatement h.ccsWitness :=
  SuperNeo.ProtocolSection71Realization.ceAssignment_eq_fullVector h

/-- [Role: Theorem-Target] Build protocol-relations assumptions from one protocol-side Section 7.5 target-data owner. -/
abbrev ProtocolRelationsAssumptions_ofProtocolTargetData
  {ctx : ProtocolTargetContext} :=
  SuperNeo.ProtocolRelationsAssumptions.ofProtocolTargetData (ctx := ctx)

/-- [Role: Theorem-Target] Recover the compact challenge-set from one specialized Section 7.1 instance. -/
theorem ProtocolSection71Specialization_challengeSet_eq_cset
  {ctx : ProtocolTargetContext}
  {Commitment : Type}
  {hInst : SuperNeo.ProofSystem.ConstraintSystem.Section71Instance Commitment}
  (h : ProtocolSection71Specialization ctx hInst) :
  hInst.params.challengeSet = ctx.cset :=
  SuperNeo.ProtocolSection71Specialization.challengeSet_eq_cset h

/-- [Role: Theorem-Target] Recover that one specialized Section 7.1 instance shares a commitment across CCS/CE. -/
theorem ProtocolSection71Specialization_sharedCommitment_eq
  {ctx : ProtocolTargetContext}
  {Commitment : Type}
  {hInst : SuperNeo.ProofSystem.ConstraintSystem.Section71Instance Commitment}
  (h : ProtocolSection71Specialization ctx hInst) :
  hInst.ccsStatement.commitment = hInst.ceStatement.commitment :=
  SuperNeo.ProtocolSection71Specialization.sharedCommitment_eq h

/-- [Role: Theorem-Target] Recover that one specialized Section 7.1 instance shares a public input across CCS/CE. -/
theorem ProtocolSection71Specialization_sharedPublicInput_eq
  {ctx : ProtocolTargetContext}
  {Commitment : Type}
  {hInst : SuperNeo.ProofSystem.ConstraintSystem.Section71Instance Commitment}
  (h : ProtocolSection71Specialization ctx hInst) :
  hInst.ccsStatement.publicInput = hInst.ceStatement.publicInput :=
  SuperNeo.ProtocolSection71Specialization.sharedPublicInput_eq h

/-- [Role: Theorem-Target] Recover that the CE witness assignment is the CCS full vector `[x, w]`. -/
theorem ProtocolSection71Specialization_ceAssignment_eq_fullVector
  {ctx : ProtocolTargetContext}
  {Commitment : Type}
  {hInst : SuperNeo.ProofSystem.ConstraintSystem.Section71Instance Commitment}
  (h : ProtocolSection71Specialization ctx hInst) :
  hInst.ceWitness.assignment =
    SuperNeo.ProofSystem.ConstraintSystem.CCS.fullVector
      hInst.ccsStatement hInst.ccsWitness :=
  SuperNeo.ProtocolSection71Specialization.ceAssignment_eq_fullVector h

/-- [Role: Theorem-Target] Specialize one generic proof-system Section 7.1 theorem instance into the compact realization boundary. -/
def ProtocolSection71Specialization_realization
  {ctx : ProtocolTargetContext}
  {Commitment : Type}
  {hInst : SuperNeo.ProofSystem.ConstraintSystem.Section71Instance Commitment}
  (h : ProtocolSection71Specialization ctx hInst) :
  ProtocolSection71Realization ctx :=
  SuperNeo.ProtocolSection71Specialization.realization h

/-- [Role: Theorem-Target] Recover the compact challenge-set from one theorem-native Section 7.1 setup. -/
theorem ProtocolSection71Setup_challengeSet_eq_cset
  {ctx : ProtocolTargetContext}
  (h : ProtocolSection71Setup ctx) :
  h.inst.params.challengeSet = ctx.cset :=
  SuperNeo.ProtocolSection71Setup.challengeSet_eq_cset h

/-- [Role: Theorem-Target] Recover that one theorem-native Section 7.1 setup shares a commitment across CCS/CE. -/
theorem ProtocolSection71Setup_sharedCommitment_eq
  {ctx : ProtocolTargetContext}
  (h : ProtocolSection71Setup ctx) :
  h.inst.ccsStatement.commitment = h.inst.ceStatement.commitment :=
  SuperNeo.ProtocolSection71Setup.sharedCommitment_eq h

/-- [Role: Theorem-Target] Recover that one theorem-native Section 7.1 setup shares a public input across CCS/CE. -/
theorem ProtocolSection71Setup_sharedPublicInput_eq
  {ctx : ProtocolTargetContext}
  (h : ProtocolSection71Setup ctx) :
  h.inst.ccsStatement.publicInput = h.inst.ceStatement.publicInput :=
  SuperNeo.ProtocolSection71Setup.sharedPublicInput_eq h

/-- [Role: Theorem-Target] Recover that the CE witness assignment is the CCS full vector `[x, w]`. -/
theorem ProtocolSection71Setup_ceAssignment_eq_fullVector
  {ctx : ProtocolTargetContext}
  (h : ProtocolSection71Setup ctx) :
  h.inst.ceWitness.assignment =
    SuperNeo.ProofSystem.ConstraintSystem.CCS.fullVector
      h.inst.ccsStatement h.inst.ccsWitness :=
  SuperNeo.ProtocolSection71Setup.ceAssignment_eq_fullVector h

/-- [Role: Theorem-Target] Convert one theorem-native Section 7.1 setup into the compact realization boundary. -/
def ProtocolSection71Setup_realization
  {ctx : ProtocolTargetContext}
  (h : ProtocolSection71Setup ctx) :
  ProtocolSection71Realization ctx :=
  SuperNeo.ProtocolSection71Setup.realization h

/-- [Role: Theorem-Target] Recover the compact challenge-set from one paper-faithful Section 7.1 theorem instance. -/
theorem ProtocolSection71TheoremInstance_challengeSet_eq_cset
  {ctx : ProtocolTargetContext}
  (h : ProtocolSection71TheoremInstance ctx) :
  h.params.challengeSet = ctx.cset :=
  SuperNeo.ProtocolSection71TheoremInstance.challengeSet_eq_cset h

/-- [Role: Theorem-Target] Recover that one paper-faithful Section 7.1 theorem instance shares a commitment across CCS/CE. -/
theorem ProtocolSection71TheoremInstance_sharedCommitment_eq
  {ctx : ProtocolTargetContext}
  (h : ProtocolSection71TheoremInstance ctx) :
  h.ccsStatement.commitment = h.ceStatement.commitment :=
  SuperNeo.ProtocolSection71TheoremInstance.sharedCommitment_eq h

/-- [Role: Theorem-Target] Recover that one paper-faithful Section 7.1 theorem instance shares a public input across CCS/CE. -/
theorem ProtocolSection71TheoremInstance_sharedPublicInput_eq
  {ctx : ProtocolTargetContext}
  (h : ProtocolSection71TheoremInstance ctx) :
  h.ccsStatement.publicInput = h.ceStatement.publicInput :=
  SuperNeo.ProtocolSection71TheoremInstance.sharedPublicInput_eq h

/-- [Role: Theorem-Target] Recover that the CE witness assignment is the CCS full vector `[x, w]`. -/
theorem ProtocolSection71TheoremInstance_ceAssignment_eq_fullVector
  {ctx : ProtocolTargetContext}
  (h : ProtocolSection71TheoremInstance ctx) :
  h.ceWitness.assignment =
    SuperNeo.ProofSystem.ConstraintSystem.CCS.fullVector
      h.ccsStatement h.ccsWitness :=
  SuperNeo.ProtocolSection71TheoremInstance.ceAssignment_eq_fullVector h

/-- [Role: Theorem-Target] Convert one paper-faithful Section 7.1 theorem instance into the compact realization boundary. -/
def ProtocolSection71TheoremInstance_realization
  {ctx : ProtocolTargetContext}
  (h : ProtocolSection71TheoremInstance ctx) :
  ProtocolSection71Realization ctx :=
  SuperNeo.ProtocolSection71TheoremInstance.realization h

/-- [Role: Theorem-Target] Recover the packaged theorem-native Section 7.1 setup from one paper-faithful theorem instance. -/
def ProtocolSection71TheoremInstance_setup
  {ctx : ProtocolTargetContext}
  (h : ProtocolSection71TheoremInstance ctx) :
  ProtocolSection71Setup ctx :=
  SuperNeo.ProtocolSection71TheoremInstance.setup h

/-- [Role: Theorem-Target] Recover the concrete Section 7.1 provider from one paper-faithful theorem instance. -/
def ProtocolSection71TheoremInstance_provider
  {ctx : ProtocolTargetContext}
  (h : ProtocolSection71TheoremInstance ctx) :
  ProtocolSection71Provider ctx :=
  SuperNeo.ProtocolSection71TheoremInstance.provider h

/-- [Role: Theorem-Target] Forget concrete holds proofs and recover the realization boundary. -/
abbrev ProtocolSection71Provider_realization
  {ctx : ProtocolTargetContext} :
  ProtocolSection71Provider ctx →
  ProtocolSection71Realization ctx :=
  SuperNeo.ProtocolSection71Provider.realization

/--
[Role: Theorem-Target] Build the canonical Section 7.1 provider from one
realized CE instance.
-/
def ProtocolSection71Provider_ofSection71CE
  {ctx : ProtocolTargetContext}
  (hReal : ProtocolSection71Realization ctx)
  (hCE :
    SuperNeo.ProofSystem.ConstraintSystem.CE.Holds
      hReal.ce hReal.ceStatement hReal.ceWitness) :
  ProtocolSection71Provider ctx :=
  SuperNeo.ProtocolSection71Provider.ofSection71CE hReal hCE

/--
[Role: Theorem-Target] Build the compact-context Section 7.1 provider from one
generic proof-system theorem instance plus the compact specialization package.
-/
def ProtocolSection71Provider_ofSpecialization
  {ctx : ProtocolTargetContext}
  {Commitment : Type}
  (hInst : SuperNeo.ProofSystem.ConstraintSystem.Section71Instance Commitment)
  (hSpec : ProtocolSection71Specialization ctx hInst) :
  ProtocolSection71Provider ctx :=
  SuperNeo.ProtocolSection71Provider.ofSpecialization hInst hSpec

/-- [Role: Theorem-Target] Build the compact-context Section 7.1 provider from one theorem-native setup. -/
def ProtocolSection71Provider_ofSetup
  {ctx : ProtocolTargetContext}
  (h : ProtocolSection71Setup ctx) :
  ProtocolSection71Provider ctx :=
  SuperNeo.ProtocolSection71Provider.ofSetup h

/-- [Role: Theorem-Target] Recover that the provider CCS and CE statements share one commitment. -/
theorem ProtocolSection71Provider_sharedCommitment_eq
  {ctx : ProtocolTargetContext}
  (h : ProtocolSection71Provider ctx) :
  h.ccsStatement.commitment = h.ceStatement.commitment :=
  SuperNeo.ProtocolSection71Provider.sharedCommitment_eq h

/-- [Role: Theorem-Target] Recover that the provider CCS and CE statements share one public input. -/
theorem ProtocolSection71Provider_sharedPublicInput_eq
  {ctx : ProtocolTargetContext}
  (h : ProtocolSection71Provider ctx) :
  h.ccsStatement.publicInput = h.ceStatement.publicInput :=
  SuperNeo.ProtocolSection71Provider.sharedPublicInput_eq h

/-- [Role: Theorem-Target] Recover that the provider CE witness assignment is the CCS full vector `[x, w]`. -/
theorem ProtocolSection71Provider_ceAssignment_eq_fullVector
  {ctx : ProtocolTargetContext}
  (h : ProtocolSection71Provider ctx) :
  h.ceWitness.assignment =
    SuperNeo.ProofSystem.ConstraintSystem.CCS.fullVector
      h.ccsStatement h.ccsWitness :=
  SuperNeo.ProtocolSection71Provider.ceAssignment_eq_fullVector h

/-- [Role: Theorem-Target] One concrete Section 7.1 provider yields the compact CCS relation. -/
theorem ccsRelation_of_section71Provider
  {ctx : ProtocolTargetContext} :
  ProtocolSection71Provider ctx →
  ccsRelation ctx :=
  SuperNeo.ccsRelation_of_section71Provider

/-- [Role: Theorem-Target] One concrete Section 7.1 provider yields the compact CE relation. -/
theorem ceRelation_of_section71Provider
  {ctx : ProtocolTargetContext} :
  ProtocolSection71Provider ctx →
  ceRelation ctx :=
  SuperNeo.ceRelation_of_section71Provider

/-- [Role: Theorem-Target] One generic Section 7.1 proof-system theorem instance plus compact specialization yields the compact CCS relation. -/
theorem ccsRelation_of_section71Specialization
  {ctx : ProtocolTargetContext}
  {Commitment : Type}
  (hInst : SuperNeo.ProofSystem.ConstraintSystem.Section71Instance Commitment) :
  ProtocolSection71Specialization ctx hInst →
  ccsRelation ctx :=
  SuperNeo.ccsRelation_of_section71Specialization hInst

/-- [Role: Theorem-Target] One generic Section 7.1 proof-system theorem instance plus compact specialization yields the compact CE relation. -/
theorem ceRelation_of_section71Specialization
  {ctx : ProtocolTargetContext}
  {Commitment : Type}
  (hInst : SuperNeo.ProofSystem.ConstraintSystem.Section71Instance Commitment) :
  ProtocolSection71Specialization ctx hInst →
  ceRelation ctx :=
  SuperNeo.ceRelation_of_section71Specialization hInst

/-- [Role: Theorem-Target] One theorem-native Section 7.1 setup yields the compact CCS relation. -/
theorem ccsRelation_of_section71Setup
  {ctx : ProtocolTargetContext} :
  ProtocolSection71Setup ctx →
  ccsRelation ctx :=
  SuperNeo.ProtocolSection71Setup.ccsRelation

/-- [Role: Theorem-Target] One theorem-native Section 7.1 setup yields the compact CE relation. -/
theorem ceRelation_of_section71Setup
  {ctx : ProtocolTargetContext} :
  ProtocolSection71Setup ctx →
  ceRelation ctx :=
  SuperNeo.ProtocolSection71Setup.ceRelation

/-- [Role: Theorem-Target] One paper-faithful Section 7.1 theorem instance yields the compact CCS relation. -/
theorem ccsRelation_of_section71TheoremInstance
  {ctx : ProtocolTargetContext} :
  ProtocolSection71TheoremInstance ctx →
  ccsRelation ctx :=
  SuperNeo.ccsRelation_of_section71TheoremInstance

/-- [Role: Theorem-Target] One paper-faithful Section 7.1 theorem instance yields the compact CE relation. -/
theorem ceRelation_of_section71TheoremInstance
  {ctx : ProtocolTargetContext} :
  ProtocolSection71TheoremInstance ctx →
  ceRelation ctx :=
  SuperNeo.ceRelation_of_section71TheoremInstance

/--
[Role: Theorem-Target] Canonical protocol-relations constructor from the
paper-facing challenge-difference route.
-/
abbrev ProtocolRelationsAssumptions_ofPaperCarrierDiff
  {ctx : ProtocolTargetContext} :=
  SuperNeo.ProtocolRelationsAssumptions.ofPaperCarrierDiff (ctx := ctx)

/--
[Role: Theorem-Target] Canonical protocol-relations constructor from the
finite basis-kernel characterization of Theorem 3 on the active paper-facing
route.
-/
abbrev ProtocolRelationsAssumptions_ofBasisKernelAssumption
  {ctx : ProtocolTargetContext} :=
  SuperNeo.ProtocolRelationsAssumptions.ofBasisKernelAssumption (ctx := ctx)

/--
[Role: Theorem-Target] Canonical protocol-relations constructor from the
executable finite basis-kernel checker on the active paper-facing route.
-/
abbrev ProtocolRelationsAssumptions_ofBasisKernelCheck
  {ctx : ProtocolTargetContext} :=
  SuperNeo.ProtocolRelationsAssumptions.ofBasisKernelCheck (ctx := ctx)

/--
[Role: Theorem-Target] Canonical protocol-relations constructor from a stronger
strict low-norm invertibility theorem with threshold at least `5`.
-/
def ProtocolRelationsAssumptions_ofLowNormAtLeastFive
  {ctx : ProtocolTargetContext}
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
  (hNe : ctx.invDelta ≠ zeroRq) :
  ProtocolRelationsAssumptions ctx :=
  SuperNeo.ProtocolRelationsAssumptions.ofLowNormAtLeastFive
    (ctx := ctx) hFive hThm3 hArithmetic hInv hDiff hNe

/--
[Role: Theorem-Target] Canonical native protocol-relations constructor from the
paper-facing challenge-difference route.
-/
abbrev ProtocolRelationsNativeAssumptions_ofPaperCarrierDiff
  {ctx : ProtocolTargetContext} :=
  SuperNeo.ProtocolRelationsNativeAssumptions.ofPaperCarrierDiff (ctx := ctx)

/--
[Role: Theorem-Target] Canonical native protocol-relations constructor from a
stronger strict low-norm invertibility theorem with threshold at least `5`.
-/
def ProtocolRelationsNativeAssumptions_ofLowNormAtLeastFive
  {ctx : ProtocolTargetContext}
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
  (hNe : ctx.invDelta ≠ zeroRq) :
  ProtocolRelationsNativeAssumptions ctx :=
  SuperNeo.ProtocolRelationsNativeAssumptions.ofLowNormAtLeastFive
    (ctx := ctx) hFive hBarNative hArithmetic hInv hDiff hNe

/-- [Role: Boundary] Boundary surface `ccsRelation_of_assumptions` requiring closure. -/
abbrev ccsRelation_of_assumptions
  {ctx : ProtocolTargetContext} :
  ProtocolRelationsAssumptions ctx →
  ccsRelation ctx :=
  SuperNeo.ccsRelation_of_assumptions

/-- [Role: Boundary] Native constructor surface for `ccsRelation`. -/
abbrev ccsRelation_of_native_assumptions
  {ctx : ProtocolTargetContext} :
  ProtocolRelationsNativeAssumptions ctx →
  ccsRelation ctx :=
  SuperNeo.ccsRelation_of_native_assumptions

/--
[Role: Theorem-Target] Derive `ccsRelation` directly on the active
paper-carrier-difference route.
-/
theorem ccsRelation_of_paperCarrierDiff
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
  ccsRelation ctx :=
  SuperNeo.ccsRelation_of_paperCarrierDiff hThm3 hArithmetic hDiff hNe

/--
[Role: Theorem-Target] Derive `ccsRelation` directly on the active
paper-carrier-difference route from the finite basis-kernel characterization
of Theorem 3.
-/
theorem ccsRelation_of_basisKernelAssumption
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
  ccsRelation ctx :=
  SuperNeo.ccsRelation_of_basisKernelAssumption
    hBasis hArithmetic hDiff hNe

/--
[Role: Theorem-Target] Derive `ccsRelation` directly on the active
paper-carrier-difference route from the executable finite basis-kernel
checker for Theorem 3.
-/
theorem ccsRelation_of_basisKernelCheck
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
  ccsRelation ctx :=
  SuperNeo.ccsRelation_of_basisKernelCheck
    hCheck hArithmetic hDiff hNe

/--
[Role: Theorem-Target] Derive `ccsRelation` directly on the active native-bar
paper-carrier-difference route.
-/
theorem ccsRelation_of_native_paperCarrierDiff
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
  ccsRelation ctx :=
  SuperNeo.ccsRelation_of_native_paperCarrierDiff
    hBarNative hArithmetic hDiff hNe

/-- [Role: Boundary] Boundary surface `ceRelation_of_assumptions` requiring closure. -/
abbrev ceRelation_of_assumptions
  {ctx : ProtocolTargetContext} :
  ProtocolRelationsAssumptions ctx →
  SumCheckTransitionWitness ctx →
  ceRelation ctx :=
  SuperNeo.ceRelation_of_assumptions

/-- [Role: Boundary] Native constructor surface for `ceRelation` from witness. -/
abbrev ceRelation_of_native_assumptions
  {ctx : ProtocolTargetContext} :
  ProtocolRelationsNativeAssumptions ctx →
  SumCheckTransitionWitness ctx →
  ceRelation ctx :=
  SuperNeo.ceRelation_of_native_assumptions

/-- [Role: Boundary] Native constructor surface for `ceRelation` from claim truth. -/
abbrev ceRelation_of_native_claimTrue
  {ctx : ProtocolTargetContext} :
  ProtocolRelationsNativeAssumptions ctx →
  SumCheckClaimTrue (sumcheckInstanceOfContext ctx) →
  ceRelation ctx :=
  SuperNeo.ceRelation_of_native_claimTrue

/--
[Role: Theorem-Target] Derive `ceRelation` directly on the active
paper-carrier-difference route.
-/
theorem ceRelation_of_paperCarrierDiff
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
  ceRelation ctx :=
  SuperNeo.ceRelation_of_paperCarrierDiff
    hThm3 hArithmetic hDiff hNe hWitness

/--
[Role: Theorem-Target] Derive `ceRelation` directly on the active
paper-carrier-difference route from the finite basis-kernel characterization
of Theorem 3.
-/
theorem ceRelation_of_basisKernelAssumption
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
  ceRelation ctx :=
  SuperNeo.ceRelation_of_basisKernelAssumption
    hBasis hArithmetic hDiff hNe hWitness

/--
[Role: Theorem-Target] Derive `ceRelation` directly on the active
paper-carrier-difference route from the executable finite basis-kernel
checker for Theorem 3.
-/
theorem ceRelation_of_basisKernelCheck
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
  ceRelation ctx :=
  SuperNeo.ceRelation_of_basisKernelCheck
    hCheck hArithmetic hDiff hNe hWitness

/--
[Role: Theorem-Target] Derive `ceRelation` directly on the active native-bar
paper-carrier-difference route.
-/
theorem ceRelation_of_native_paperCarrierDiff
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
  ceRelation ctx :=
  SuperNeo.ceRelation_of_native_paperCarrierDiff
    hBarNative hArithmetic hDiff hNe hWitness

end ProtocolRelationsInterface

end SuperNeo
