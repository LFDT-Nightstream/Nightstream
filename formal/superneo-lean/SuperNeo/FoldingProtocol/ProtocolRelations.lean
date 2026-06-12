import SuperNeo.FoldingProtocol.ProtocolTarget
import SuperNeo.Primitives.SumCheck
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

/-- CCS relation is just the protocol target proposition. -/
theorem ccsRelation_of_protocolTargetProp
  {ctx : ProtocolTargetContext}
  (hTarget : protocolTargetProp ctx) :
  ccsRelation ctx := by
  exact hTarget

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

/-- Derive CE relation from CCS relation and SumCheck claim truth. -/
theorem ceRelation_of_ccsRelation_claimTrue
  {ctx : ProtocolTargetContext}
  (hCCS : ccsRelation ctx)
  (hClaimTrue : SumCheckClaimTrue (sumcheckInstanceOfContext ctx)) :
  ceRelation ctx := by
  rcases sumcheckCompleteness_constructive (sumcheckInstanceOfContext ctx) hClaimTrue with ⟨tr, hAcc⟩
  exact ⟨hCCS, ⟨tr, hAcc⟩⟩

/-- Soundness lift: any CE witness yields SumCheck claim truth. -/
theorem ceClaimTrue_of_ce
  {ctx : ProtocolTargetContext}
  (hCE : ceRelation ctx) :
  SumCheckClaimTrue (sumcheckInstanceOfContext ctx) := by
  rcases hCE.2 with ⟨tr, hAcc⟩
  exact sumcheckSoundness_constructive _ _ hAcc

/-- CE implies relaxed CE. -/
theorem ceRelaxedRelation_of_ce
  {ctx : ProtocolTargetContext}
  (hCE : ceRelation ctx) :
  ceRelaxedRelation ctx := by
  exact hCE.1

end SuperNeo
