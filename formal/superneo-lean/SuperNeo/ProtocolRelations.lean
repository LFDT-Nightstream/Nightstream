import SuperNeo.ProtocolTargetData
import SuperNeo.SumCheck
import SuperNeo.ProofSystem.ConstraintSystem

/-!
CCS/CE relation layer.

This module defines paper-facing relation predicates on top of the protocol
context and ties them to the protocol-target and SumCheck boundaries.
-/

namespace SuperNeo

/-- Build a SumCheck instance from protocol-target context fields. -/
def sumcheckInstanceOfContext (ctx : ProtocolTargetContext) : SumCheckInstance :=
  { rounds := ctx.kSplit
    maxDegree := ctx.m.size
    domainSize := ctx.cset.size
    claimedValue := ct ctx.invDelta }

/--
The protocol SumCheck instance is aligned with the full Goldilocks field
denominator required by the full-field Lund endpoint.
-/
def sumcheckFullFieldDenominatorAlignment
  (ctx : ProtocolTargetContext) : Prop :=
  SuperNeo.sumcheckLundSoundnessDenominator (sumcheckInstanceOfContext ctx) =
    Goldilocks.q

theorem sumcheckFullFieldDenominatorAlignment_iff
  {ctx : ProtocolTargetContext} :
  sumcheckFullFieldDenominatorAlignment ctx ↔
    ctx.cset.size = Goldilocks.q := by
  simp [sumcheckFullFieldDenominatorAlignment, sumcheckInstanceOfContext,
    SuperNeo.sumcheckLundSoundnessDenominator]

/--
Minimal setup-side boundary for replaying the active Goldilocks/full-field Lund
endpoint on one protocol context.
-/
structure GoldilocksFullFieldLundBoundary (ctx : ProtocolTargetContext) where
  denominatorAligned : sumcheckFullFieldDenominatorAlignment ctx

namespace GoldilocksFullFieldLundBoundary

/--
Canonical setup boundary from the concrete challenge-set cardinality equality
used by the active Goldilocks route.
-/
def ofCsetCardinality
  {ctx : ProtocolTargetContext}
  (hCard : ctx.cset.size = Goldilocks.q) :
  GoldilocksFullFieldLundBoundary ctx :=
  ⟨(sumcheckFullFieldDenominatorAlignment_iff).2 hCard⟩

/--
Recover the concrete challenge-set cardinality equality from the named setup
boundary.
-/
theorem csetCardinality_eq
  {ctx : ProtocolTargetContext}
  (h : GoldilocksFullFieldLundBoundary ctx) :
  ctx.cset.size = Goldilocks.q :=
  (sumcheckFullFieldDenominatorAlignment_iff).1 h.denominatorAligned

end GoldilocksFullFieldLundBoundary

/-- Explicit SumCheck witness carrying the transition facts used by reductions. -/
structure SumCheckTransitionWitness (ctx : ProtocolTargetContext) where
  transcript : SumCheckTranscript
  accepted : SumCheckAccepted (sumcheckInstanceOfContext ctx) transcript
  initialRound :
    sumcheckInitialRoundConsistent (sumcheckInstanceOfContext ctx) transcript
  roundSumStep :
    ∀ i : Nat,
      i + 1 < transcript.roundPolys.size →
        sumcheckEvalPoly (transcript.roundPolys[i + 1]!) 0 +
            sumcheckEvalPoly (transcript.roundPolys[i + 1]!) 1 =
          sumcheckEvalPoly (transcript.roundPolys[i]!) (transcript.challenges[i]!)

theorem SumCheckTransitionWitness.accepted_exists
  {ctx : ProtocolTargetContext}
  (h : SumCheckTransitionWitness ctx) :
  ∃ tr : SumCheckTranscript,
    SumCheckAccepted (sumcheckInstanceOfContext ctx) tr := by
  exact ⟨h.transcript, h.accepted⟩

/-- CCS relation: protocol target holds. -/
def ccsRelation (ctx : ProtocolTargetContext) : Prop :=
  protocolTargetProp ctx

/-- CE relation: CCS relation plus an accepted SumCheck transcript witness. -/
def ceRelation (ctx : ProtocolTargetContext) : Prop :=
  ccsRelation ctx ∧
  ∃ tr : SumCheckTranscript,
    SumCheckAccepted (sumcheckInstanceOfContext ctx) tr

/-- Relaxed CE relation: keep only CCS relation (claim-truth may be deferred). -/
def ceRelaxedRelation (ctx : ProtocolTargetContext) : Prop :=
  ccsRelation ctx

/--
Actual Section 7.1 proof-system objects specialized to one compact protocol
context.

This packages the Definition-14-style global parameters together with one
concrete CCS statement/witness pair and one concrete CE statement/witness pair.
Compatibility with the compact `ccsRelation` / `ceRelation` surfaces is carried
separately by `ProtocolSection71Realization`.
-/
structure ProtocolSection71Objects (ctx : ProtocolTargetContext) where
  Commitment : Type
  params :
    SuperNeo.ProofSystem.ConstraintSystem.GlobalParams Commitment
  normBound : Nat
  challengeSet_eq : params.challengeSet = ctx.cset
  ccsStatement :
    SuperNeo.ProofSystem.ConstraintSystem.CCS.Statement Commitment
  ccsWitness :
    SuperNeo.ProofSystem.ConstraintSystem.CCS.Witness
  ceStatement :
    SuperNeo.ProofSystem.ConstraintSystem.CE.Statement Commitment
  ceWitness :
    SuperNeo.ProofSystem.ConstraintSystem.CE.Witness
  sharedCommitment :
    ccsStatement.commitment = ceStatement.commitment
  sharedPublicInput :
    ccsStatement.publicInput = ceStatement.publicInput
  sharedAssignment :
    SuperNeo.ProofSystem.ConstraintSystem.CCS.fullVector ccsStatement ccsWitness =
      ceWitness.assignment

namespace ProtocolSection71Objects

/-- Canonical realized CCS carrier from the shared Definition-14 parameters. -/
def ccs
  {ctx : ProtocolTargetContext}
  (h : ProtocolSection71Objects ctx) :
  SuperNeo.ProofSystem.ConstraintSystem.CCS h.Commitment :=
  SuperNeo.ProofSystem.ConstraintSystem.GlobalParams.ccs
    h.params h.normBound

/-- Canonical realized CE carrier from the shared Definition-14 parameters. -/
def ce
  {ctx : ProtocolTargetContext}
  (h : ProtocolSection71Objects ctx) :
  SuperNeo.ProofSystem.ConstraintSystem.CE h.Commitment :=
  SuperNeo.ProofSystem.ConstraintSystem.GlobalParams.ce
    h.params h.normBound

/-- Recover the context challenge-set from the realized Definition-14 package. -/
theorem challengeSet_eq_cset
  {ctx : ProtocolTargetContext}
  (h : ProtocolSection71Objects ctx) :
  h.params.challengeSet = ctx.cset :=
  h.challengeSet_eq

/-- Recover that the realized CCS and CE statements share one commitment. -/
theorem sharedCommitment_eq
  {ctx : ProtocolTargetContext}
  (h : ProtocolSection71Objects ctx) :
  h.ccsStatement.commitment = h.ceStatement.commitment :=
  h.sharedCommitment

/-- Recover that the realized CCS and CE statements share one public input. -/
theorem sharedPublicInput_eq
  {ctx : ProtocolTargetContext}
  (h : ProtocolSection71Objects ctx) :
  h.ccsStatement.publicInput = h.ceStatement.publicInput :=
  h.sharedPublicInput

/-- Recover that the CE witness assignment is the CCS full vector `[x, w]`. -/
theorem ceAssignment_eq_fullVector
  {ctx : ProtocolTargetContext}
  (h : ProtocolSection71Objects ctx) :
  h.ceWitness.assignment =
    SuperNeo.ProofSystem.ConstraintSystem.CCS.fullVector
      h.ccsStatement h.ccsWitness := by
  simpa using h.sharedAssignment.symm

/-- Forget the compact-context specialization and recover the proof-system object package. -/
def toSection71Objects
  {ctx : ProtocolTargetContext}
  (h : ProtocolSection71Objects ctx) :
  SuperNeo.ProofSystem.ConstraintSystem.Section71Objects h.Commitment where
  params := h.params
  normBound := h.normBound
  ccsStatement := h.ccsStatement
  ccsWitness := h.ccsWitness
  ceStatement := h.ceStatement
  ceWitness := h.ceWitness
  sharedCommitment := h.sharedCommitment
  sharedPublicInput := h.sharedPublicInput
  sharedAssignment := h.sharedAssignment

/-- Specialize one proof-system Section 7.1 object package to a compact context. -/
def ofSection71Objects
  {ctx : ProtocolTargetContext}
  {Commitment : Type}
  (hObj : SuperNeo.ProofSystem.ConstraintSystem.Section71Objects Commitment)
  (hChallengeSet : hObj.params.challengeSet = ctx.cset) :
  ProtocolSection71Objects ctx where
  Commitment := Commitment
  params := hObj.params
  normBound := hObj.normBound
  challengeSet_eq := hChallengeSet
  ccsStatement := hObj.ccsStatement
  ccsWitness := hObj.ccsWitness
  ceStatement := hObj.ceStatement
  ceWitness := hObj.ceWitness
  sharedCommitment := hObj.sharedCommitment
  sharedPublicInput := hObj.sharedPublicInput
  sharedAssignment := hObj.sharedAssignment

end ProtocolSection71Objects

/--
Boundary package realizing the compact protocol relations as explicit Section
7.1 proof-system objects.

This is the remaining integration point between the current
`ProtocolTargetContext`-based formulation and the paper-facing relation objects
in `ProofSystem.ConstraintSystem.CCS`. The actual Definition-14 objects live in
`ProtocolSection71Objects`; this structure adds the two-way compatibility
theorems back to the compact relation predicates.
-/
structure ProtocolSection71Realization (ctx : ProtocolTargetContext)
    extends ProtocolSection71Objects ctx where
  ccsHolds_from_relation :
    ccsRelation ctx →
    SuperNeo.ProofSystem.ConstraintSystem.CCS.Holds
      (SuperNeo.ProofSystem.ConstraintSystem.GlobalParams.ccs
        params normBound)
      ccsStatement
      ccsWitness
  ccsRelation_of_holds :
    SuperNeo.ProofSystem.ConstraintSystem.CCS.Holds
      (SuperNeo.ProofSystem.ConstraintSystem.GlobalParams.ccs
        params normBound)
      ccsStatement
      ccsWitness →
    ccsRelation ctx
  ceHolds_from_relation :
    ceRelation ctx →
    SuperNeo.ProofSystem.ConstraintSystem.CE.Holds
      (SuperNeo.ProofSystem.ConstraintSystem.GlobalParams.ce
        params normBound)
      ceStatement
      ceWitness
  ceRelation_of_holds :
    SuperNeo.ProofSystem.ConstraintSystem.CE.Holds
      (SuperNeo.ProofSystem.ConstraintSystem.GlobalParams.ce
        params normBound)
      ceStatement
      ceWitness →
    ceRelation ctx

namespace ProtocolSection71Realization

/-- Canonical realized CCS carrier from the shared Definition-14 parameters. -/
def ccs
  {ctx : ProtocolTargetContext}
  (h : ProtocolSection71Realization ctx) :
  SuperNeo.ProofSystem.ConstraintSystem.CCS h.Commitment :=
  h.toProtocolSection71Objects.ccs

/-- Canonical realized CE carrier from the shared Definition-14 parameters. -/
def ce
  {ctx : ProtocolTargetContext}
  (h : ProtocolSection71Realization ctx) :
  SuperNeo.ProofSystem.ConstraintSystem.CE h.Commitment :=
  h.toProtocolSection71Objects.ce

/-- Recover the context challenge-set from the realized Definition-14 package. -/
theorem challengeSet_eq_cset
  {ctx : ProtocolTargetContext}
  (h : ProtocolSection71Realization ctx) :
  h.params.challengeSet = ctx.cset :=
  h.toProtocolSection71Objects.challengeSet_eq_cset

/-- Recover that the realized CCS and CE statements share one commitment. -/
theorem sharedCommitment_eq
  {ctx : ProtocolTargetContext}
  (h : ProtocolSection71Realization ctx) :
  h.ccsStatement.commitment = h.ceStatement.commitment :=
  h.toProtocolSection71Objects.sharedCommitment_eq

/-- Recover that the realized CCS and CE statements share one public input. -/
theorem sharedPublicInput_eq
  {ctx : ProtocolTargetContext}
  (h : ProtocolSection71Realization ctx) :
  h.ccsStatement.publicInput = h.ceStatement.publicInput :=
  h.toProtocolSection71Objects.sharedPublicInput_eq

/-- Recover that the CE witness assignment is the CCS full vector `[x, w]`. -/
theorem ceAssignment_eq_fullVector
  {ctx : ProtocolTargetContext}
  (h : ProtocolSection71Realization ctx) :
  h.ceWitness.assignment =
    SuperNeo.ProofSystem.ConstraintSystem.CCS.fullVector
      h.ccsStatement h.ccsWitness :=
  h.toProtocolSection71Objects.ceAssignment_eq_fullVector

/--
Build a compact-context Section 7.1 realization from one proof-system
Section-7.1 object package plus the compact relation compatibility theorems.
-/
def ofSection71Objects
  {ctx : ProtocolTargetContext}
  {Commitment : Type}
  (hObj : SuperNeo.ProofSystem.ConstraintSystem.Section71Objects Commitment)
  (hChallengeSet : hObj.params.challengeSet = ctx.cset)
  (hCCSFrom :
    ccsRelation ctx →
      SuperNeo.ProofSystem.ConstraintSystem.CCS.Holds
        (SuperNeo.ProofSystem.ConstraintSystem.GlobalParams.ccs
          hObj.params hObj.normBound)
        hObj.ccsStatement
        hObj.ccsWitness)
  (hCCSTo :
    SuperNeo.ProofSystem.ConstraintSystem.CCS.Holds
        (SuperNeo.ProofSystem.ConstraintSystem.GlobalParams.ccs
          hObj.params hObj.normBound)
        hObj.ccsStatement
        hObj.ccsWitness →
      ccsRelation ctx)
  (hCEFrom :
    ceRelation ctx →
      SuperNeo.ProofSystem.ConstraintSystem.CE.Holds
        (SuperNeo.ProofSystem.ConstraintSystem.GlobalParams.ce
          hObj.params hObj.normBound)
        hObj.ceStatement
        hObj.ceWitness)
  (hCETo :
    SuperNeo.ProofSystem.ConstraintSystem.CE.Holds
        (SuperNeo.ProofSystem.ConstraintSystem.GlobalParams.ce
          hObj.params hObj.normBound)
        hObj.ceStatement
        hObj.ceWitness →
      ceRelation ctx) :
  ProtocolSection71Realization ctx where
  toProtocolSection71Objects :=
    ProtocolSection71Objects.ofSection71Objects hObj hChallengeSet
  ccsHolds_from_relation := hCCSFrom
  ccsRelation_of_holds := hCCSTo
  ceHolds_from_relation := hCEFrom
  ceRelation_of_holds := hCETo

theorem ccsHolds_of_relation
  {ctx : ProtocolTargetContext}
  (h : ProtocolSection71Realization ctx)
  (hCCS : ccsRelation ctx) :
  SuperNeo.ProofSystem.ConstraintSystem.CCS.Holds
    h.ccs h.ccsStatement h.ccsWitness := by
  exact h.ccsHolds_from_relation hCCS

theorem relation_of_ccsHolds
  {ctx : ProtocolTargetContext}
  (h : ProtocolSection71Realization ctx)
  (hCCS :
    SuperNeo.ProofSystem.ConstraintSystem.CCS.Holds
      h.ccs h.ccsStatement h.ccsWitness) :
  ccsRelation ctx := by
  exact h.ccsRelation_of_holds hCCS

theorem ceHolds_of_relation
  {ctx : ProtocolTargetContext}
  (h : ProtocolSection71Realization ctx)
  (hCE : ceRelation ctx) :
  SuperNeo.ProofSystem.ConstraintSystem.CE.Holds
    h.ce h.ceStatement h.ceWitness := by
  exact h.ceHolds_from_relation hCE

theorem relation_of_ceHolds
  {ctx : ProtocolTargetContext}
  (h : ProtocolSection71Realization ctx)
  (hCE :
    SuperNeo.ProofSystem.ConstraintSystem.CE.Holds
      h.ce h.ceStatement h.ceWitness) :
  ceRelation ctx := by
  exact h.ceRelation_of_holds hCE

end ProtocolSection71Realization

/--
Compact-context specialization data for one generic proof-system Section 7.1
theorem instance.

This is the smallest remaining broad Section 7.1 boundary once the paper-facing
Definition-14 objects and `CCS.Holds` / `CE.Holds` proofs are already available
upstream as one `ProofSystem.ConstraintSystem.Section71Instance`.
-/
structure ProtocolSection71Specialization
    (ctx : ProtocolTargetContext)
    {Commitment : Type}
    (hInst : SuperNeo.ProofSystem.ConstraintSystem.Section71Instance Commitment) where
  challengeSet_eq : hInst.params.challengeSet = ctx.cset
  ccsHolds_from_relation :
    ccsRelation ctx →
      SuperNeo.ProofSystem.ConstraintSystem.CCS.Holds
        hInst.ccs
        hInst.ccsStatement
        hInst.ccsWitness
  ccsRelation_of_holds :
    SuperNeo.ProofSystem.ConstraintSystem.CCS.Holds
        hInst.ccs
        hInst.ccsStatement
        hInst.ccsWitness →
      ccsRelation ctx
  ceHolds_from_relation :
    ceRelation ctx →
      SuperNeo.ProofSystem.ConstraintSystem.CE.Holds
        hInst.ce
        hInst.ceStatement
        hInst.ceWitness
  ceRelation_of_holds :
    SuperNeo.ProofSystem.ConstraintSystem.CE.Holds
        hInst.ce
        hInst.ceStatement
        hInst.ceWitness →
      ceRelation ctx

namespace ProtocolSection71Specialization

/-- Recover the compact challenge-set from the specialized Section 7.1 instance. -/
theorem challengeSet_eq_cset
  {ctx : ProtocolTargetContext}
  {Commitment : Type}
  {hInst : SuperNeo.ProofSystem.ConstraintSystem.Section71Instance Commitment}
  (h : ProtocolSection71Specialization ctx hInst) :
  hInst.params.challengeSet = ctx.cset :=
  h.challengeSet_eq

/-- Recover that the specialized CCS and CE statements share one commitment. -/
theorem sharedCommitment_eq
  {ctx : ProtocolTargetContext}
  {Commitment : Type}
  {hInst : SuperNeo.ProofSystem.ConstraintSystem.Section71Instance Commitment}
  (_h : ProtocolSection71Specialization ctx hInst) :
  hInst.ccsStatement.commitment = hInst.ceStatement.commitment :=
  hInst.sharedCommitment_eq

/-- Recover that the specialized CCS and CE statements share one public input. -/
theorem sharedPublicInput_eq
  {ctx : ProtocolTargetContext}
  {Commitment : Type}
  {hInst : SuperNeo.ProofSystem.ConstraintSystem.Section71Instance Commitment}
  (_h : ProtocolSection71Specialization ctx hInst) :
  hInst.ccsStatement.publicInput = hInst.ceStatement.publicInput :=
  hInst.sharedPublicInput_eq

/-- Recover that the CE witness assignment is the CCS full vector `[x, w]`. -/
theorem ceAssignment_eq_fullVector
  {ctx : ProtocolTargetContext}
  {Commitment : Type}
  {hInst : SuperNeo.ProofSystem.ConstraintSystem.Section71Instance Commitment}
  (_h : ProtocolSection71Specialization ctx hInst) :
  hInst.ceWitness.assignment =
    SuperNeo.ProofSystem.ConstraintSystem.CCS.fullVector
      hInst.ccsStatement hInst.ccsWitness :=
  hInst.ceAssignment_eq_fullVector

/--
Specialize one generic proof-system Section 7.1 theorem instance into the
compact realized Section 7.1 boundary.
-/
def realization
  {ctx : ProtocolTargetContext}
  {Commitment : Type}
  {hInst : SuperNeo.ProofSystem.ConstraintSystem.Section71Instance Commitment}
  (h : ProtocolSection71Specialization ctx hInst) :
  ProtocolSection71Realization ctx :=
  ProtocolSection71Realization.ofSection71Objects
    hInst.objects
    h.challengeSet_eq
    h.ccsHolds_from_relation
    h.ccsRelation_of_holds
    h.ceHolds_from_relation
    h.ceRelation_of_holds

end ProtocolSection71Specialization

/--
One theorem-native Section 7.1 setup for a compact protocol context.

This is the smallest explicit owner of the paper-facing Definition-14 data once
one generic proof-system `Section71Instance` and its compact specialization
theorems are already available upstream.
-/
structure ProtocolSection71Setup (ctx : ProtocolTargetContext) where
  Commitment : Type
  inst :
    SuperNeo.ProofSystem.ConstraintSystem.Section71Instance Commitment
  spec :
    ProtocolSection71Specialization ctx inst

namespace ProtocolSection71Setup

/-- Recover the compact challenge-set from the theorem-native Section 7.1 setup. -/
theorem challengeSet_eq_cset
  {ctx : ProtocolTargetContext}
  (h : ProtocolSection71Setup ctx) :
  h.inst.params.challengeSet = ctx.cset :=
  h.spec.challengeSet_eq_cset

/-- Recover that the setup CCS and CE statements share one commitment. -/
theorem sharedCommitment_eq
  {ctx : ProtocolTargetContext}
  (h : ProtocolSection71Setup ctx) :
  h.inst.ccsStatement.commitment = h.inst.ceStatement.commitment :=
  h.spec.sharedCommitment_eq

/-- Recover that the setup CCS and CE statements share one public input. -/
theorem sharedPublicInput_eq
  {ctx : ProtocolTargetContext}
  (h : ProtocolSection71Setup ctx) :
  h.inst.ccsStatement.publicInput = h.inst.ceStatement.publicInput :=
  h.spec.sharedPublicInput_eq

/-- Recover that the CE witness assignment is the CCS full vector `[x, w]`. -/
theorem ceAssignment_eq_fullVector
  {ctx : ProtocolTargetContext}
  (h : ProtocolSection71Setup ctx) :
  h.inst.ceWitness.assignment =
    SuperNeo.ProofSystem.ConstraintSystem.CCS.fullVector
      h.inst.ccsStatement h.inst.ccsWitness :=
  h.spec.ceAssignment_eq_fullVector

/--
Canonical realized Section 7.1 boundary induced by one theorem-native setup.
-/
def realization
  {ctx : ProtocolTargetContext}
  (h : ProtocolSection71Setup ctx) :
  ProtocolSection71Realization ctx :=
  h.spec.realization

/--
One theorem-native Section 7.1 setup yields the compact CCS relation.
-/
theorem ccsRelation
  {ctx : ProtocolTargetContext}
  (h : ProtocolSection71Setup ctx) :
  ccsRelation ctx := by
  exact h.spec.ccsRelation_of_holds h.inst.ccsHolds

/--
One theorem-native Section 7.1 setup yields the compact CE relation.
-/
theorem ceRelation
  {ctx : ProtocolTargetContext}
  (h : ProtocolSection71Setup ctx) :
  ceRelation ctx := by
  exact h.spec.ceRelation_of_holds h.inst.ceHolds

end ProtocolSection71Setup

/--
Upstream provider bundle for a concrete Section 7.1 theorem instance.

This extends the Definition-14 realization boundary with the actual proof-
system `CCS.Holds` / `CE.Holds` witnesses for the current theorem instance.
It is the smallest paper-facing owner for the data needed to feed the broad
Section 7 reductions from explicit Section 7.1 objects.
-/
structure ProtocolSection71Provider (ctx : ProtocolTargetContext)
    extends ProtocolSection71Realization ctx where
  ccsHolds :
    SuperNeo.ProofSystem.ConstraintSystem.CCS.Holds
      (SuperNeo.ProofSystem.ConstraintSystem.GlobalParams.ccs
        params normBound)
      ccsStatement
      ccsWitness
  ceHolds :
    SuperNeo.ProofSystem.ConstraintSystem.CE.Holds
      (SuperNeo.ProofSystem.ConstraintSystem.GlobalParams.ce
        params normBound)
      ceStatement
      ceWitness

namespace ProtocolSection71Provider

/-- Forget the concrete holds proofs and keep only the realization boundary. -/
abbrev realization
  {ctx : ProtocolTargetContext}
  (h : ProtocolSection71Provider ctx) :
  ProtocolSection71Realization ctx :=
  h.toProtocolSection71Realization

/--
Canonical provider bundle from one realized Section 7.1 CE instance.

Once a shared Definition-14 realization and one concrete `CE.Holds` witness are
available, the compact CE relation and the induced CCS membership follow
canonically.
-/
def ofSection71CE
  {ctx : ProtocolTargetContext}
  (hReal : ProtocolSection71Realization ctx)
  (hCE :
    SuperNeo.ProofSystem.ConstraintSystem.CE.Holds
      hReal.ce hReal.ceStatement hReal.ceWitness) :
  ProtocolSection71Provider ctx := by
  have hRel : ceRelation ctx := hReal.relation_of_ceHolds hCE
  refine
    { toProtocolSection71Realization := hReal
      ccsHolds := ?_
      ceHolds := hCE }
  exact hReal.ccsHolds_of_relation hRel.1

/-- Recover that the provider CCS and CE statements share one commitment. -/
theorem sharedCommitment_eq
  {ctx : ProtocolTargetContext}
  (h : ProtocolSection71Provider ctx) :
  h.ccsStatement.commitment = h.ceStatement.commitment :=
  h.realization.sharedCommitment_eq

/-- Recover that the provider CCS and CE statements share one public input. -/
theorem sharedPublicInput_eq
  {ctx : ProtocolTargetContext}
  (h : ProtocolSection71Provider ctx) :
  h.ccsStatement.publicInput = h.ceStatement.publicInput :=
  h.realization.sharedPublicInput_eq

/-- Recover that the provider CE witness assignment is the CCS full vector `[x, w]`. -/
theorem ceAssignment_eq_fullVector
  {ctx : ProtocolTargetContext}
  (h : ProtocolSection71Provider ctx) :
  h.ceWitness.assignment =
    SuperNeo.ProofSystem.ConstraintSystem.CCS.fullVector
      h.ccsStatement h.ccsWitness :=
  h.realization.ceAssignment_eq_fullVector

/--
Build a compact-context Section 7.1 provider from one proof-system Section 7.1
instance plus the compact relation specialization theorems.
-/
def ofSection71Instance
  {ctx : ProtocolTargetContext}
  {Commitment : Type}
  (hInst : SuperNeo.ProofSystem.ConstraintSystem.Section71Instance Commitment)
  (hChallengeSet : hInst.params.challengeSet = ctx.cset)
  (hCCSFrom :
    ccsRelation ctx →
      SuperNeo.ProofSystem.ConstraintSystem.CCS.Holds
        (SuperNeo.ProofSystem.ConstraintSystem.GlobalParams.ccs
          hInst.params hInst.normBound)
        hInst.ccsStatement
        hInst.ccsWitness)
  (hCCSTo :
    SuperNeo.ProofSystem.ConstraintSystem.CCS.Holds
        (SuperNeo.ProofSystem.ConstraintSystem.GlobalParams.ccs
          hInst.params hInst.normBound)
        hInst.ccsStatement
        hInst.ccsWitness →
      ccsRelation ctx)
  (hCEFrom :
    ceRelation ctx →
      SuperNeo.ProofSystem.ConstraintSystem.CE.Holds
        (SuperNeo.ProofSystem.ConstraintSystem.GlobalParams.ce
          hInst.params hInst.normBound)
        hInst.ceStatement
        hInst.ceWitness)
  (hCETo :
    SuperNeo.ProofSystem.ConstraintSystem.CE.Holds
        (SuperNeo.ProofSystem.ConstraintSystem.GlobalParams.ce
          hInst.params hInst.normBound)
        hInst.ceStatement
        hInst.ceWitness →
      ceRelation ctx) :
  ProtocolSection71Provider ctx where
  toProtocolSection71Realization :=
    ProtocolSection71Realization.ofSection71Objects
      hInst.objects
      hChallengeSet
      hCCSFrom
      hCCSTo
      hCEFrom
      hCETo
  ccsHolds := hInst.ccsHolds
  ceHolds := hInst.ceHolds

/--
Build a compact-context Section 7.1 provider directly from one generic proof-
system Section 7.1 theorem instance plus its compact specialization package.
-/
def ofSpecialization
  {ctx : ProtocolTargetContext}
  {Commitment : Type}
  (hInst : SuperNeo.ProofSystem.ConstraintSystem.Section71Instance Commitment)
  (hSpec : ProtocolSection71Specialization ctx hInst) :
  ProtocolSection71Provider ctx :=
  ofSection71Instance
    hInst
    hSpec.challengeSet_eq
    hSpec.ccsHolds_from_relation
    hSpec.ccsRelation_of_holds
    hSpec.ceHolds_from_relation
    hSpec.ceRelation_of_holds

/--
Canonical concrete Section 7.1 provider bundle induced by one theorem-native
setup.
-/
def ofSetup
  {ctx : ProtocolTargetContext}
  (h : ProtocolSection71Setup ctx) :
  ProtocolSection71Provider ctx :=
  ofSpecialization h.inst h.spec

end ProtocolSection71Provider

/--
One paper-faithful Section 7.1 theorem instance specialized to a compact
protocol context.

This is the canonical theorem-native owner for the broad Section 7 stack: one
shared Definition-14 parameter package, one coherent CCS/CE tuple pair, their
concrete proof-system membership proofs, and the specialization theorems back
to the compact `ccsRelation` / `ceRelation` predicates.
-/
structure ProtocolSection71TheoremInstance (ctx : ProtocolTargetContext) where
  Commitment : Type
  params :
    SuperNeo.ProofSystem.ConstraintSystem.GlobalParams Commitment
  normBound : Nat
  ccsStatement :
    SuperNeo.ProofSystem.ConstraintSystem.CCS.Statement Commitment
  ccsWitness :
    SuperNeo.ProofSystem.ConstraintSystem.CCS.Witness
  ceStatement :
    SuperNeo.ProofSystem.ConstraintSystem.CE.Statement Commitment
  ceWitness :
    SuperNeo.ProofSystem.ConstraintSystem.CE.Witness
  challengeSet_eq : params.challengeSet = ctx.cset
  sharedCommitment :
    ccsStatement.commitment = ceStatement.commitment
  sharedPublicInput :
    ccsStatement.publicInput = ceStatement.publicInput
  sharedAssignment :
    SuperNeo.ProofSystem.ConstraintSystem.CCS.fullVector ccsStatement ccsWitness =
      ceWitness.assignment
  ccsHolds :
    SuperNeo.ProofSystem.ConstraintSystem.CCS.Holds
      (SuperNeo.ProofSystem.ConstraintSystem.GlobalParams.ccs
        params normBound)
      ccsStatement
      ccsWitness
  ceHolds :
    SuperNeo.ProofSystem.ConstraintSystem.CE.Holds
      (SuperNeo.ProofSystem.ConstraintSystem.GlobalParams.ce
        params normBound)
      ceStatement
      ceWitness
  ccsHolds_from_relation :
    ccsRelation ctx →
      SuperNeo.ProofSystem.ConstraintSystem.CCS.Holds
        (SuperNeo.ProofSystem.ConstraintSystem.GlobalParams.ccs
          params normBound)
        ccsStatement
        ccsWitness
  ccsRelation_of_holds :
    SuperNeo.ProofSystem.ConstraintSystem.CCS.Holds
        (SuperNeo.ProofSystem.ConstraintSystem.GlobalParams.ccs
          params normBound)
        ccsStatement
        ccsWitness →
      ccsRelation ctx
  ceHolds_from_relation :
    ceRelation ctx →
      SuperNeo.ProofSystem.ConstraintSystem.CE.Holds
        (SuperNeo.ProofSystem.ConstraintSystem.GlobalParams.ce
          params normBound)
        ceStatement
        ceWitness
  ceRelation_of_holds :
    SuperNeo.ProofSystem.ConstraintSystem.CE.Holds
        (SuperNeo.ProofSystem.ConstraintSystem.GlobalParams.ce
          params normBound)
        ceStatement
        ceWitness →
      ceRelation ctx

namespace ProtocolSection71TheoremInstance

/-- Canonical realized CCS carrier from the shared Definition-14 parameters. -/
def ccs
  {ctx : ProtocolTargetContext}
  (h : ProtocolSection71TheoremInstance ctx) :
  SuperNeo.ProofSystem.ConstraintSystem.CCS h.Commitment :=
  SuperNeo.ProofSystem.ConstraintSystem.GlobalParams.ccs
    h.params h.normBound

/-- Canonical realized CE carrier from the shared Definition-14 parameters. -/
def ce
  {ctx : ProtocolTargetContext}
  (h : ProtocolSection71TheoremInstance ctx) :
  SuperNeo.ProofSystem.ConstraintSystem.CE h.Commitment :=
  SuperNeo.ProofSystem.ConstraintSystem.GlobalParams.ce
    h.params h.normBound

/-- Recover the compact challenge-set from the theorem-native Section 7.1 instance. -/
theorem challengeSet_eq_cset
  {ctx : ProtocolTargetContext}
  (h : ProtocolSection71TheoremInstance ctx) :
  h.params.challengeSet = ctx.cset :=
  h.challengeSet_eq

/-- Recover that the theorem-native CCS and CE statements share one commitment. -/
theorem sharedCommitment_eq
  {ctx : ProtocolTargetContext}
  (h : ProtocolSection71TheoremInstance ctx) :
  h.ccsStatement.commitment = h.ceStatement.commitment :=
  h.sharedCommitment

/-- Recover that the theorem-native CCS and CE statements share one public input. -/
theorem sharedPublicInput_eq
  {ctx : ProtocolTargetContext}
  (h : ProtocolSection71TheoremInstance ctx) :
  h.ccsStatement.publicInput = h.ceStatement.publicInput :=
  h.sharedPublicInput

/-- Recover that the CE witness assignment is the CCS full vector `[x, w]`. -/
theorem ceAssignment_eq_fullVector
  {ctx : ProtocolTargetContext}
  (h : ProtocolSection71TheoremInstance ctx) :
  h.ceWitness.assignment =
    SuperNeo.ProofSystem.ConstraintSystem.CCS.fullVector
      h.ccsStatement h.ccsWitness := by
  simpa using h.sharedAssignment.symm

/-- Forget the compact specialization and recover the proof-system object package. -/
def toSection71Objects
  {ctx : ProtocolTargetContext}
  (h : ProtocolSection71TheoremInstance ctx) :
  SuperNeo.ProofSystem.ConstraintSystem.Section71Objects h.Commitment where
  params := h.params
  normBound := h.normBound
  ccsStatement := h.ccsStatement
  ccsWitness := h.ccsWitness
  ceStatement := h.ceStatement
  ceWitness := h.ceWitness
  sharedCommitment := h.sharedCommitment
  sharedPublicInput := h.sharedPublicInput
  sharedAssignment := h.sharedAssignment

/-- Recover the proof-system Section 7.1 theorem instance. -/
def toSection71Instance
  {ctx : ProtocolTargetContext}
  (h : ProtocolSection71TheoremInstance ctx) :
  SuperNeo.ProofSystem.ConstraintSystem.Section71Instance h.Commitment where
  toSection71Objects := h.toSection71Objects
  ccsHolds := h.ccsHolds
  ceHolds := h.ceHolds

/-- Recover the compact realized Section 7.1 boundary. -/
def realization
  {ctx : ProtocolTargetContext}
  (h : ProtocolSection71TheoremInstance ctx) :
  ProtocolSection71Realization ctx :=
  ProtocolSection71Realization.ofSection71Objects
    h.toSection71Objects
    h.challengeSet_eq
    h.ccsHolds_from_relation
    h.ccsRelation_of_holds
    h.ceHolds_from_relation
    h.ceRelation_of_holds

/-- Recover the compact specialization for the induced proof-system theorem instance. -/
def specialization
  {ctx : ProtocolTargetContext}
  (h : ProtocolSection71TheoremInstance ctx) :
  ProtocolSection71Specialization ctx h.toSection71Instance where
  challengeSet_eq := h.challengeSet_eq
  ccsHolds_from_relation := h.ccsHolds_from_relation
  ccsRelation_of_holds := h.ccsRelation_of_holds
  ceHolds_from_relation := h.ceHolds_from_relation
  ceRelation_of_holds := h.ceRelation_of_holds

/-- Recover the packaged theorem-native Section 7.1 setup. -/
def setup
  {ctx : ProtocolTargetContext}
  (h : ProtocolSection71TheoremInstance ctx) :
  ProtocolSection71Setup ctx where
  Commitment := h.Commitment
  inst := h.toSection71Instance
  spec := h.specialization

/-- Recover the smallest concrete Section 7.1 provider from the theorem instance. -/
def provider
  {ctx : ProtocolTargetContext}
  (h : ProtocolSection71TheoremInstance ctx) :
  ProtocolSection71Provider ctx := by
  refine
    { toProtocolSection71Realization := h.realization
      ccsHolds := ?_
      ceHolds := ?_ }
  · simpa [realization, ProtocolSection71Realization.ofSection71Objects,
      toSection71Objects, ccs] using h.ccsHolds
  · simpa [realization, ProtocolSection71Realization.ofSection71Objects,
      toSection71Objects, ce] using h.ceHolds

/-- One theorem-native Section 7.1 instance yields the compact CCS relation. -/
theorem ccsRelation
  {ctx : ProtocolTargetContext}
  (h : ProtocolSection71TheoremInstance ctx) :
  ccsRelation ctx := by
  exact h.ccsRelation_of_holds h.ccsHolds

/-- One theorem-native Section 7.1 instance yields the compact CE relation. -/
theorem ceRelation
  {ctx : ProtocolTargetContext}
  (h : ProtocolSection71TheoremInstance ctx) :
  ceRelation ctx := by
  exact h.ceRelation_of_holds h.ceHolds

end ProtocolSection71TheoremInstance

/--
Derive compact `ccsRelation` directly from one theorem-native Section 7.1
instance.
-/
theorem ccsRelation_of_section71TheoremInstance
  {ctx : ProtocolTargetContext}
  (h : ProtocolSection71TheoremInstance ctx) :
  ccsRelation ctx := by
  exact h.ccsRelation

/--
Derive compact `ccsRelation` directly from one protocol-side Section 7.5
target-data owner.
-/
theorem ccsRelation_of_protocolTargetData
  {ctx : ProtocolTargetContext}
  (h : ProtocolTargetData ctx) :
  ccsRelation ctx := by
  exact h.protocolTargetProp

/--
Derive compact `ceRelation` directly from one theorem-native Section 7.1
instance.
-/
theorem ceRelation_of_section71TheoremInstance
  {ctx : ProtocolTargetContext}
  (h : ProtocolSection71TheoremInstance ctx) :
  ceRelation ctx := by
  exact h.ceRelation

/--
Derive compact `ccsRelation` directly from one concrete Section 7.1 provider
bundle.
-/
theorem ccsRelation_of_section71Provider
  {ctx : ProtocolTargetContext}
  (h : ProtocolSection71Provider ctx) :
  ccsRelation ctx := by
  exact h.realization.relation_of_ccsHolds h.ccsHolds

/--
Derive compact `ceRelation` directly from one concrete Section 7.1 provider
bundle.
-/
theorem ceRelation_of_section71Provider
  {ctx : ProtocolTargetContext}
  (h : ProtocolSection71Provider ctx) :
  ceRelation ctx := by
  exact h.realization.relation_of_ceHolds h.ceHolds

/--
One generic proof-system Section 7.1 theorem instance plus its compact
specialization package yields the compact CCS relation.
-/
theorem ccsRelation_of_section71Specialization
  {ctx : ProtocolTargetContext}
  {Commitment : Type}
  (hInst : SuperNeo.ProofSystem.ConstraintSystem.Section71Instance Commitment)
  (hSpec : ProtocolSection71Specialization ctx hInst) :
  ccsRelation ctx := by
  exact hSpec.ccsRelation_of_holds hInst.ccsHolds

/--
One generic proof-system Section 7.1 theorem instance plus its compact
specialization package yields the compact CE relation.
-/
theorem ceRelation_of_section71Specialization
  {ctx : ProtocolTargetContext}
  {Commitment : Type}
  (hInst : SuperNeo.ProofSystem.ConstraintSystem.Section71Instance Commitment)
  (hSpec : ProtocolSection71Specialization ctx hInst) :
  ceRelation ctx := by
  exact hSpec.ceRelation_of_holds hInst.ceHolds

/-- Assumptions needed to derive relation-level statements. -/
structure ProtocolRelationsAssumptions (ctx : ProtocolTargetContext) where
  target : ProtocolTargetAssumptions ctx

/-- Native assumption bundle: protocol target closes Theorem-3 via native bar. -/
structure ProtocolRelationsNativeAssumptions (ctx : ProtocolTargetContext) where
  target : ProtocolTargetNativeAssumptions ctx

/--
Canonical protocol-relations assumptions using constructive SumCheck closure.
-/
def ProtocolRelationsAssumptions.ofTarget
  {ctx : ProtocolTargetContext}
  (hTarget : ProtocolTargetAssumptions ctx) :
  ProtocolRelationsAssumptions ctx :=
  { target := hTarget }

/--
Canonical protocol-relations assumptions from one protocol-side Section 7.5
target-data owner.
-/
def ProtocolRelationsAssumptions.ofProtocolTargetData
  {ctx : ProtocolTargetContext}
  (hTarget : ProtocolTargetData ctx) :
  ProtocolRelationsAssumptions ctx :=
  { target := hTarget.assumptions }

/--
Canonical protocol-relations assumptions using the paper-facing challenge-
difference route for `invDelta`.
-/
def ProtocolRelationsAssumptions.ofPaperCarrierDiff
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
  ProtocolRelationsAssumptions ctx :=
  { target := ProtocolTargetAssumptions.ofPaperCarrierDiff
      hThm3 hArithmetic hDiff hNe }

/--
Canonical protocol-relations assumptions using the finite basis-kernel
characterization of Theorem 3 on the active paper-carrier-difference route.
-/
def ProtocolRelationsAssumptions.ofBasisKernelAssumption
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
  ProtocolRelationsAssumptions ctx :=
  { target := ProtocolTargetAssumptions.ofBasisKernelAssumption
      hBasis hArithmetic hDiff hNe }

/--
Canonical protocol-relations assumptions using the executable finite
basis-kernel checker for Theorem 3 on the active paper-carrier-difference
route.
-/
def ProtocolRelationsAssumptions.ofBasisKernelCheck
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
  ProtocolRelationsAssumptions ctx :=
  { target := ProtocolTargetAssumptions.ofBasisKernelCheck
      hCheck hArithmetic hDiff hNe }

/--
Canonical protocol-relations assumptions from any strict low-norm
invertibility boundary whose threshold is at least `5`, specialized to the
active paper-carrier-difference route.
-/
def ProtocolRelationsAssumptions.ofLowNormAtLeastFive
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
  { target := ProtocolTargetAssumptions.ofLowNormAtLeastFive
      hFive hThm3 hArithmetic hInv hDiff hNe }

/--
Canonical native protocol-relations assumptions using constructive SumCheck closure.
-/
def ProtocolRelationsNativeAssumptions.ofTarget
  {ctx : ProtocolTargetContext}
  (hTarget : ProtocolTargetNativeAssumptions ctx) :
  ProtocolRelationsNativeAssumptions ctx :=
  { target := hTarget }

/--
Canonical native protocol-relations assumptions using the paper-facing
challenge-difference route for `invDelta`.
-/
def ProtocolRelationsNativeAssumptions.ofPaperCarrierDiff
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
  ProtocolRelationsNativeAssumptions ctx :=
  { target := ProtocolTargetNativeAssumptions.ofPaperCarrierDiff
      hBarNative hArithmetic hDiff hNe }

/--
Canonical native protocol-relations assumptions from any strict low-norm
invertibility boundary whose threshold is at least `5`, specialized to the
active paper-carrier-difference route.
-/
def ProtocolRelationsNativeAssumptions.ofLowNormAtLeastFive
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
  { target := ProtocolTargetNativeAssumptions.ofLowNormAtLeastFive
      hFive hBarNative hArithmetic hInv hDiff hNe }

/-- Derive CCS relation from target assumptions. -/
theorem ccsRelation_of_assumptions
  {ctx : ProtocolTargetContext}
  (h : ProtocolRelationsAssumptions ctx) :
  ccsRelation ctx := by
  exact protocolTargetProp_of_assumptions h.target

/-- Derive CCS relation from native target assumptions. -/
theorem ccsRelation_of_native_assumptions
  {ctx : ProtocolTargetContext}
  (h : ProtocolRelationsNativeAssumptions ctx) :
  ccsRelation ctx := by
  exact protocolTargetProp_of_native_assumptions h.target

/-- CCS relation is just the protocol target proposition. -/
theorem ccsRelation_of_protocolTargetProp
  {ctx : ProtocolTargetContext}
  (hTarget : protocolTargetProp ctx) :
  ccsRelation ctx := by
  exact hTarget

/--
Derive CCS relation directly on the active paper-carrier-difference route.
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
  ccsRelation ctx := by
  exact ccsRelation_of_protocolTargetProp
    (protocolTargetProp_of_paperCarrierDiff hThm3 hArithmetic hDiff hNe)

/--
Derive CCS relation directly on the active paper-carrier-difference route from
the finite basis-kernel characterization of Theorem 3.
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
  ccsRelation ctx := by
  exact ccsRelation_of_protocolTargetProp
    (protocolTargetProp_of_basisKernelAssumption hBasis hArithmetic
      (invertibleRq_of_paperCarrierDiff hDiff hNe))

/--
Derive CCS relation directly on the active paper-carrier-difference route from
the executable finite basis-kernel checker for Theorem 3.
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
  ccsRelation ctx := by
  exact ccsRelation_of_basisKernelAssumption
    (thm3BasisKernelAssumption_of_check hCheck)
    hArithmetic
    hDiff
    hNe

/--
Derive CCS relation directly on the active native-bar
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
  ccsRelation ctx := by
  exact ccsRelation_of_protocolTargetProp
    (protocolTargetProp_of_native_paperCarrierDiff
      hBarNative hArithmetic hDiff hNe)

/-- Derive CE relation from explicit transcript acceptance witness. -/
theorem ccsRelation_iff_protocolTargetProp
  {ctx : ProtocolTargetContext} :
  ccsRelation ctx ↔ protocolTargetProp ctx := by
  rfl

/-- CE is exactly CCS plus an accepted SumCheck transcript witness. -/
theorem ceRelation_iff
  {ctx : ProtocolTargetContext} :
  ceRelation ctx ↔
    ccsRelation ctx ∧
      ∃ tr : SumCheckTranscript,
        SumCheckAccepted (sumcheckInstanceOfContext ctx) tr := by
  rfl

/-- Relaxed CE is definitionally CCS. -/
theorem ceRelaxedRelation_iff
  {ctx : ProtocolTargetContext} :
  ceRelaxedRelation ctx ↔ ccsRelation ctx := by
  rfl

/-- Derive CE relation from CCS relation and an explicit transcript witness. -/
theorem ceRelation_of_ccsRelation
  {ctx : ProtocolTargetContext}
  (hCCS : ccsRelation ctx)
  (hWitness : SumCheckTransitionWitness ctx) :
  ceRelation ctx := by
  exact ⟨hCCS, hWitness.accepted_exists⟩

/--
Derive CE relation directly from one protocol-side Section 7.5 target-data
owner and one accepted transition witness.
-/
theorem ceRelation_of_protocolTargetData
  {ctx : ProtocolTargetContext}
  (hTarget : ProtocolTargetData ctx)
  (hWitness : SumCheckTransitionWitness ctx) :
  ceRelation ctx := by
  exact ceRelation_of_ccsRelation
    (ccsRelation_of_protocolTargetData hTarget)
    hWitness

/-- Derive CE relation from CCS relation and SumCheck claim truth. -/
theorem ceRelation_of_ccsRelation_claimTrue
  {ctx : ProtocolTargetContext}
  (hCCS : ccsRelation ctx)
  (hClaimTrue : SumCheckClaimTrue (sumcheckInstanceOfContext ctx)) :
  ceRelation ctx := by
  rcases sumcheckCompleteness_constructive (sumcheckInstanceOfContext ctx) hClaimTrue with ⟨tr, hAcc⟩
  exact ⟨hCCS, ⟨tr, hAcc⟩⟩

/-- Derive CE relation from explicit transcript acceptance witness. -/
theorem ceRelation_of_assumptions
  {ctx : ProtocolTargetContext}
  (h : ProtocolRelationsAssumptions ctx)
  (hWitness : SumCheckTransitionWitness ctx) :
  ceRelation ctx := by
  exact ceRelation_of_ccsRelation (ccsRelation_of_assumptions h) hWitness

/-- Derive CE relation from claim-truth via SumCheck completeness boundary. -/
theorem ceRelation_of_claimTrue
  {ctx : ProtocolTargetContext}
  (h : ProtocolRelationsAssumptions ctx)
  (hClaimTrue : SumCheckClaimTrue (sumcheckInstanceOfContext ctx)) :
  ceRelation ctx := by
  exact ceRelation_of_ccsRelation_claimTrue (ccsRelation_of_assumptions h) hClaimTrue

/-- Derive CE relation from native assumptions and explicit transcript witness. -/
theorem ceRelation_of_native_assumptions
  {ctx : ProtocolTargetContext}
  (h : ProtocolRelationsNativeAssumptions ctx)
  (hWitness : SumCheckTransitionWitness ctx) :
  ceRelation ctx := by
  exact ceRelation_of_ccsRelation (ccsRelation_of_native_assumptions h) hWitness

/--
Derive CE relation directly on the active paper-carrier-difference route from
the protocol data and a SumCheck transition witness.
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
  ceRelation ctx := by
  exact ceRelation_of_ccsRelation
    (ccsRelation_of_paperCarrierDiff hThm3 hArithmetic hDiff hNe)
    hWitness

/--
Derive CE relation directly on the active paper-carrier-difference route from
the finite basis-kernel characterization of Theorem 3.
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
  ceRelation ctx := by
  exact ceRelation_of_ccsRelation
    (ccsRelation_of_basisKernelAssumption hBasis hArithmetic hDiff hNe)
    hWitness

/--
Derive CE relation directly on the active paper-carrier-difference route from
the executable finite basis-kernel checker for Theorem 3.
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
  ceRelation ctx := by
  exact ceRelation_of_basisKernelAssumption
    (thm3BasisKernelAssumption_of_check hCheck)
    hArithmetic
    hDiff
    hNe
    hWitness

/--
Derive CE relation directly on the active native-bar
paper-carrier-difference route from the protocol data and a SumCheck
transition witness.
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
  ceRelation ctx := by
  exact ceRelation_of_ccsRelation
    (ccsRelation_of_native_paperCarrierDiff hBarNative hArithmetic hDiff hNe)
    hWitness

/-- Derive CE relation from claim-truth via native assumptions. -/
theorem ceRelation_of_native_claimTrue
  {ctx : ProtocolTargetContext}
  (h : ProtocolRelationsNativeAssumptions ctx)
  (hClaimTrue : SumCheckClaimTrue (sumcheckInstanceOfContext ctx)) :
  ceRelation ctx := by
  exact ceRelation_of_ccsRelation_claimTrue
    (ccsRelation_of_native_assumptions h) hClaimTrue

/-- Soundness lift: any CE witness yields SumCheck claim truth. -/
theorem ceClaimTrue_of_ce
  {ctx : ProtocolTargetContext}
  (hCE : ceRelation ctx) :
  SumCheckClaimTrue (sumcheckInstanceOfContext ctx) := by
  rcases hCE.2 with ⟨tr, hAcc⟩
  exact sumcheckSoundness_constructive _ _ hAcc

/-- Soundness lift on the native assumption path. -/
theorem ceClaimTrue_of_native_ce
  {ctx : ProtocolTargetContext}
  (hCE : ceRelation ctx) :
  SumCheckClaimTrue (sumcheckInstanceOfContext ctx) :=
  ceClaimTrue_of_ce hCE

/-- CE implies relaxed CE. -/
theorem ceRelaxedRelation_of_ce
  {ctx : ProtocolTargetContext}
  (hCE : ceRelation ctx) :
  ceRelaxedRelation ctx := by
  exact hCE.1

end SuperNeo
