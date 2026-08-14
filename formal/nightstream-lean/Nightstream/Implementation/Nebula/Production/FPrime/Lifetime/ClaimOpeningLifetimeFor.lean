import Nightstream.Implementation.Nebula.Production.FPrime.Lifetime.CompactChainLifetimeFor

/-!
Contract: same-witness commitment authority for the complete delayed F-prime
lifetime at one generated relation exponent.

Every fresh claim is produced by the selected CCS relation from one bounded
assignment.  The mandatory four-component commitment bundle is therefore the
product commitment of that same assignment.  The delayed schedule then proves
that every verified receipt, including the trailing terminal receipt, is the
exact protocol image of one such produced claim.  Finally, the compact-chain
partition uses exactly those receipts.  Thus every commitment hashed into a
prechallenge `D_pre` chain has a same-witness CCS opening.

No opening, assignment, receipt list, or precommit sequence is an input to the
main lifetime theorem.  They are derived from the fixed base/recursive
producer nodes and the exact delayed schedule.

Does not prove NIFS knowledge extraction from arbitrary deployed proof bytes,
Module-SIS binding, Poseidon2 security, generated-artifact containment, Rust
refinement, or terminal-backend soundness.

Assurance tier: exponent-indexed producer-to-precommit refinement.

Emits constraints: no; it composes row-soundness theorems.
-/

set_option autoImplicit false
set_option maxRecDepth 100000

namespace Nightstream.Implementation.Nebula.ProductionPaperClaimOpeningLifetimeFor

open Nightstream.Implementation.Nebula
open Nightstream.Protocol.Nebula
open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint

namespace ClaimLifetime

open ProductionPaperExactFPrimeLifetimeFor

/-- The CCS statement encoded by one complete typed fresh claim under the one
relation artifact selected by the verifier context. -/
noncomputable def statementOfClaim
    {Program : Type} (context : Context Program) (claim : context.Claim) :
    CCS.Instance
      (ProductPaperAlgebraFor.Structure context.rowVariables
        context.logicalWidth)
      (ProductPaperAlgebraFor.PublicInput context.rowVariables
        context.logicalWidth context.publicFits)
      ProductPaperAlgebraFor.Commitment where
  constraintSystem := ProductPaperAlgebraFor.matrixSource
    context.artifact.system
  commitment := ProductNifsCodec.codecBundle claim.commitmentBundle
  publicInput := ProductNifsCodec.publicInputOfFor
    (ProductPaperAlgebraFor.fullShapeContract context.rowVariables
      context.logicalWidth context.publicFits)
    claim.ccsPublic
  stage := .fresh

/-- The CCS statement encoded by the exact protocol claim stored in one
verified delayed receipt.  This definition lets the receipt-level theorem
state one complete `CCS.Holds` fact, instead of exposing detached existential
witnesses for commitment opening and relation satisfaction. -/
noncomputable def statementOfProtocolClaim
    {Program : Type} (context : Context Program)
    (claim : context.ProtocolClaim) :
    CCS.Instance
      (ProductPaperAlgebraFor.Structure context.rowVariables
        context.logicalWidth)
      (ProductPaperAlgebraFor.PublicInput context.rowVariables
        context.logicalWidth context.publicFits)
      ProductPaperAlgebraFor.Commitment where
  constraintSystem := ProductPaperAlgebraFor.matrixSource
    context.artifact.system
  commitment := ProductNifsCodec.codecBundle claim.commitmentBundle
  publicInput := ProductNifsCodec.publicInputOfFor
    (ProductPaperAlgebraFor.fullShapeContract context.rowVariables
      context.logicalWidth context.publicFits)
    claim.ccsPublic
  stage := .fresh

@[simp] theorem statementOfProtocolClaim_toProtocolClaim
    {Program : Type} (context : Context Program) (claim : context.Claim) :
    statementOfProtocolClaim context claim.toProtocolClaim =
      statementOfClaim context claim := by
  rfl

/-- Data extracted from the producer rows for one fresh claim.  `holds` is
CCS membership for the complete claim statement, so its opening equation,
public projection, norm bound, and relation satisfaction all use
`assignment`. -/
structure ClaimOpeningWitness
    {Program : Type} (context : Context Program) (claim : context.Claim) where
  assignment : context.FreshAssignment
  holds : CCS.Holds (ProductPaperAlgebraFor.semantics context.config)
    productionGlobalParams (statementOfClaim context claim) assignment
  affineOne : claim.ccsPublic.val.getD 0 0 = 1

/-- Propositional existence of one producer-row witness.  Keeping the data
behind `Nonempty` lets schedule induction eliminate list-membership proofs
without using choice. -/
def ClaimOpening
    {Program : Type} (context : Context Program) (claim : context.Claim) : Prop :=
  Nonempty (ClaimOpeningWitness context claim)

namespace ClaimOpening

/-- The mandatory four-component claim bundle opens to the one CCS witness.
This is a projection of CCS membership, not an opening assumption. -/
theorem bundleOpens
    {Program : Type} {context : Context Program} {claim : context.Claim}
    (opening : ClaimOpening context claim) :
    exists assignment : context.FreshAssignment,
      ProductCommitmentAlgebra.commit context.config assignment =
        ProductNifsCodec.codecBundle claim.commitmentBundle := by
  rcases opening with ⟨witness⟩
  exact ⟨witness.assignment, witness.holds.1.1⟩

/-- The public coordinates of the complete claim come from the same witness
that opens its complete bundle. -/
theorem publicInputExact
    {Program : Type} {context : Context Program} {claim : context.Claim}
    (opening : ClaimOpening context claim) :
    exists assignment : context.FreshAssignment,
      Phi81Relation.projectPublicInput assignment =
        ProductNifsCodec.publicInputOfFor
          (ProductPaperAlgebraFor.fullShapeContract context.rowVariables
            context.logicalWidth context.publicFits)
          claim.ccsPublic := by
  rcases opening with ⟨witness⟩
  exact ⟨witness.assignment, witness.holds.1.2.1⟩

/-- The same assignment satisfies the verifier-selected augmented relation. -/
theorem relationSatisfied
    {Program : Type} {context : Context Program} {claim : context.Claim}
    (opening : ClaimOpening context claim) :
    exists assignment : context.FreshAssignment,
      (ProductPaperAlgebraFor.semantics context.config).ccsSatisfied
        (ProductPaperAlgebraFor.matrixSource context.artifact.system)
        assignment := by
  rcases opening with ⟨witness⟩
  exact ⟨witness.assignment, witness.holds.2⟩

/-- The first CCS public coordinate has the affine value required by the
source-R1CS compiler. This fact is retained from claim construction; it is
not inferred from arbitrary CCS membership. -/
theorem affineOne
    {Program : Type} {context : Context Program} {claim : context.Claim}
    (opening : ClaimOpening context claim) :
    claim.ccsPublic.val.getD 0 0 = 1 := by
  rcases opening with ⟨witness⟩
  exact witness.affineOne

end ClaimOpening

/-- Claim zero obtains its same-witness opening from the selected base
producer.  The opening is not a field of `BaseNode`. -/
noncomputable def BaseNode.claimOpening
    {Program : Type} {context : Context Program}
    (node : BaseNode context) : ClaimOpening context node.claim := by
  refine ⟨
    { assignment := node.freshAssignment
      holds := ?_
      affineOne := ?_ }⟩
  simpa [statementOfClaim, BaseNode.claim,
    ProductionPaperBaseInvocationFor.claim,
    ProductionFreshClaimProducerFor.freshStatement] using
    node.exact.freshRelationHolds
  change
    (ProductionMemoryBoundCcsPublic.word _ _).val.getD 0 0 = 1
  exact ProductionMemoryBoundCcsPublic.word_zero _ _

/-- A recursive producer obtains the next claim's same-witness opening from
its exact fresh-relation witness.  The consumed prior claim is not reused as
the next assignment. -/
noncomputable def RecursiveNode.nextClaimOpening
    {Program : Type} {context : Context Program} {previous : context.Claim}
    (node : RecursiveNode context previous) :
    ClaimOpening context node.nextClaim := by
  refine ⟨
    { assignment := node.freshAssignment
      holds := ?_
      affineOne := ?_ }⟩
  simpa [statementOfClaim, RecursiveNode.nextClaim,
    ProductionPaperRecursiveProducerInvocationFor.claim,
    ProductionFreshClaimProducerFor.freshStatement] using
    node.exact.nextFreshRelationHolds
  change
    (ProductionMemoryBoundCcsPublic.word _ _).val.getD 0 0 = 1
  exact ProductionMemoryBoundCcsPublic.word_zero _ _

/-- One accepted delayed receipt is tied to the exact produced typed claim
whose CCS witness opens its commitment bundle. -/
def ReceiptOpening
    {Program : Type} (context : Context Program)
    (receipt : context.Receipt) : Prop :=
  exists sourceClaim : context.Claim,
    receipt.claim = sourceClaim.toProtocolClaim /\
      ClaimOpening context sourceClaim

namespace ReceiptOpening

/-- One assignment simultaneously opens the receipt's complete product
commitment, supplies its public input, meets the fresh norm bound, and
satisfies the verifier-selected CCS relation. -/
theorem claimHolds
    {Program : Type} {context : Context Program} {receipt : context.Receipt}
    (opened : ReceiptOpening context receipt) :
    exists assignment : context.FreshAssignment,
      CCS.Holds (ProductPaperAlgebraFor.semantics context.config)
        productionGlobalParams
        (statementOfProtocolClaim context receipt.claim) assignment := by
  rcases opened with ⟨sourceClaim, claimExact, ⟨witness⟩⟩
  refine ⟨witness.assignment, ?_⟩
  rw [claimExact]
  exact witness.holds

/-- A receipt bundle opens under the same assignment as its exact produced
source claim. -/
theorem bundleOpens
    {Program : Type} {context : Context Program} {receipt : context.Receipt}
    (opened : ReceiptOpening context receipt) :
    exists assignment : context.FreshAssignment,
      ProductCommitmentAlgebra.commit context.config assignment =
        ProductNifsCodec.codecBundle receipt.claim.commitmentBundle := by
  rcases opened.claimHolds with ⟨assignment, holds⟩
  exact ⟨assignment, holds.1.1⟩

/-- The receipt public input is projected from the same assignment that opens
its complete product commitment. -/
theorem publicInputExact
    {Program : Type} {context : Context Program} {receipt : context.Receipt}
    (opened : ReceiptOpening context receipt) :
    exists assignment : context.FreshAssignment,
      Phi81Relation.projectPublicInput assignment =
        ProductNifsCodec.publicInputOfFor
          (ProductPaperAlgebraFor.fullShapeContract context.rowVariables
            context.logicalWidth context.publicFits)
          receipt.claim.ccsPublic := by
  rcases opened.claimHolds with ⟨assignment, holds⟩
  exact ⟨assignment, holds.1.2.1⟩

/-- The same receipt assignment satisfies the verifier-selected augmented
relation. -/
theorem relationSatisfied
    {Program : Type} {context : Context Program} {receipt : context.Receipt}
    (opened : ReceiptOpening context receipt) :
    exists assignment : context.FreshAssignment,
      (ProductPaperAlgebraFor.semantics context.config).ccsSatisfied
        (ProductPaperAlgebraFor.matrixSource context.artifact.system)
        assignment := by
  rcases opened.claimHolds with ⟨assignment, holds⟩
  exact ⟨assignment, holds.2⟩

/-- Every component of the receipt bundle is the corresponding projection of
the same full CCS witness. -/
theorem componentOpens
    {Program : Type} {context : Context Program} {receipt : context.Receipt}
    (opened : ReceiptOpening context receipt)
    (component : CommitmentBundle.Component) :
    exists assignment : context.FreshAssignment,
      ProductCommitmentAlgebra.commit context.config assignment component =
        ProductNifsCodec.codecBundle
          receipt.claim.commitmentBundle component := by
  rcases opened.bundleOpens with ⟨assignment, bundleExact⟩
  exact ⟨assignment, congrFun bundleExact component⟩

/-- The exact producer claim behind the receipt fixes the affine public
coordinate. A detached `CCS.Holds` witness alone would not establish this
fact. -/
theorem affineOne
    {Program : Type} {context : Context Program} {receipt : context.Receipt}
    (opened : ReceiptOpening context receipt) :
    receipt.claim.ccsPublic.val.getD 0 0 = 1 := by
  rcases opened with ⟨sourceClaim, claimExact, sourceOpening⟩
  rw [claimExact]
  exact sourceOpening.affineOne

/-- Reverse relation extraction for the exact same assignment that opens the
receipt commitment. This theorem decodes arbitrary accepted ternary words;
it does not assume the honest deterministic encoder image. -/
theorem exactDecodedBranch
    {Program : Type} {context : Context Program} {receipt : context.Receipt}
    (opened : ReceiptOpening context receipt) :
    exists assignment : context.FreshAssignment,
      CCS.Holds (ProductPaperAlgebraFor.semantics context.config)
          productionGlobalParams
          (statementOfProtocolClaim context receipt.claim) assignment /\
        ProductionFreshClaimProducerFor.RelationAuthority.ExactDecodedBranch
          context.relationAuthority assignment := by
  rcases opened.claimHolds with ⟨assignment, holds⟩
  refine ⟨assignment, holds, ?_⟩
  apply
    context.relationAuthority.selectedBranchOfCcsPublic context.config
      context.artifact assignment
      (ProductNifsCodec.publicInputOfFor
        (ProductPaperAlgebraFor.fullShapeContract context.rowVariables
          context.logicalWidth context.publicFits)
        receipt.claim.ccsPublic)
      holds.1.2.1
  · change ProductNifsCodec.fieldOfBit
      (receipt.claim.ccsPublic.val.getD 0 0) = 1
    rw [opened.affineOne]
    rfl
  · exact holds.2

end ReceiptOpening

namespace Schedule

/-- Every claim in the delayed schedule has the producer opening that existed
before the later recursive or terminal consumer verified it. -/
theorem everyClaimOpened
    {Program : Type} {context : Context Program} {previous : context.Claim}
    (schedule : Schedule context previous)
    (previousOpening : ClaimOpening context previous) :
    forall claim, claim ∈ ExactDelayedSchedule.Schedule.claims schedule ->
      ClaimOpening context claim := by
  induction schedule with
  | terminal event =>
      intro claim member
      simp only [ExactDelayedSchedule.Schedule.claims,
        List.mem_singleton] at member
      subst claim
      exact previousOpening
  | @recursive current next event rest inductionHypothesis =>
      intro claim member
      simp only [ExactDelayedSchedule.Schedule.claims, List.mem_cons] at member
      rcases member with rfl | later
      · exact previousOpening
      · have nextOpening : ClaimOpening context next := by
          rw [event.nextExact]
          exact RecursiveNode.nextClaimOpening event.node
        exact inductionHypothesis nextOpening claim later

/-- Every verified receipt in the delayed schedule is the protocol image of a
claim with a row-derived same-witness opening.  This includes the terminal
receipt for the trailing claim. -/
theorem everyReceiptOpened
    {Program : Type} {context : Context Program} {previous : context.Claim}
    (schedule : Schedule context previous)
    (previousOpening : ClaimOpening context previous) :
    forall receipt, receipt ∈ ExactDelayedSchedule.Schedule.receipts schedule ->
      ReceiptOpening context receipt := by
  induction schedule with
  | terminal event =>
      intro receipt member
      simp only [ExactDelayedSchedule.Schedule.receipts,
        ProductionPaperExactFPrimeLifetimeFor.scheduleInterface,
        List.mem_singleton] at member
      subst receipt
      exact ⟨_, event.node.consumes_trailing, previousOpening⟩
  | @recursive current next event rest inductionHypothesis =>
      intro receipt member
      simp only [ExactDelayedSchedule.Schedule.receipts,
        ProductionPaperExactFPrimeLifetimeFor.scheduleInterface,
        List.mem_cons] at member
      rcases member with rfl | later
      · exact ⟨_, event.node.consumes_previous, previousOpening⟩
      · have nextOpening : ClaimOpening context next := by
          rw [event.nextExact]
          exact RecursiveNode.nextClaimOpening event.node
        exact inductionHypothesis nextOpening receipt later

end Schedule

/-- Complete same-witness authority for all produced claims and all consumed
receipts in one closed F-prime lifetime. -/
structure ExactOpenings
    {Program : Type} {context : Context Program}
    (lifetime : Lifetime context) : Prop where
  produced : forall claim, claim ∈ lifetime.producedClaims ->
    ClaimOpening context claim
  consumed : forall receipt, receipt ∈ lifetime.consumedReceipts ->
    ReceiptOpening context receipt

namespace Lifetime

/-- Main producer-to-consumer same-witness theorem.  Its only input is the
exact base/recursive/terminal lifetime. -/
noncomputable def exactOpenings
    {Program : Type} {context : Context Program}
    (lifetime : Lifetime context) : ExactOpenings lifetime := by
  have firstOpening : ClaimOpening context lifetime.firstClaim := by
    rw [lifetime.firstClaimExact]
    exact BaseNode.claimOpening lifetime.base
  exact
    { produced := by
        intro claim member
        exact Schedule.everyClaimOpened lifetime.schedule firstOpening claim
          member
      consumed := by
        intro receipt member
        exact Schedule.everyReceiptOpened lifetime.schedule firstOpening
          receipt member }

/-- Audit-facing form: each verified receipt bundle in the exact lifetime has
one complete same-witness product opening. -/
theorem everyConsumedBundleOpens
    {Program : Type} {context : Context Program}
    (lifetime : Lifetime context) :
    forall receipt, receipt ∈ lifetime.consumedReceipts ->
      exists assignment : context.FreshAssignment,
        ProductCommitmentAlgebra.commit context.config assignment =
          ProductNifsCodec.codecBundle receipt.claim.commitmentBundle := by
  intro receipt member
  exact ReceiptOpening.bundleOpens
    ((Lifetime.exactOpenings lifetime).consumed receipt member)

/-- Audit-facing strong form: every verified receipt has one complete
`CCS.Holds` witness.  Commitment opening, public projection, fresh norm bound,
and relation satisfaction cannot use different assignments. -/
theorem everyConsumedClaimHolds
    {Program : Type} {context : Context Program}
    (lifetime : Lifetime context) :
    forall receipt, receipt ∈ lifetime.consumedReceipts ->
      exists assignment : context.FreshAssignment,
        CCS.Holds (ProductPaperAlgebraFor.semantics context.config)
          productionGlobalParams
          (statementOfProtocolClaim context receipt.claim) assignment := by
  intro receipt member
  exact ReceiptOpening.claimHolds
    ((Lifetime.exactOpenings lifetime).consumed receipt member)

end Lifetime

end ClaimLifetime

namespace SemanticLifetime

open ProductionPaperExactLifetime

/-- Strong precommit-chain result.  The compact-chain partition and every
same-witness receipt opening use the exact same receipt list from the delayed
F-prime lifetime. -/
theorem LifetimeExtraction.precommitChainWithOpenings
    {Program : Type} {context : Context Program}
    {base : BaseNode context} (lifetime : LifetimeExtraction base) :
    exists allBatches,
      ProductionMemoryRowSegments.receipts allBatches = lifetime.receipts /\
      ProductionMemoryRowSegments.Chain context.candidate context.Schema
        context.Verifier context.headers
        (ProductionPaperBaseInvocationFor.initialClosed
          context.authoritativeInitialMemoryRoot)
        allBatches lifetime.finalMemory context.statement.base.segmentCount /\
      ProductionPaperCompactChainLifetimeFor.ClaimLifetime.PrecommitChainExact
        context.claimLifecycle
        (ProductionPaperBaseInvocationFor.initialClosed
          context.authoritativeInitialMemoryRoot)
        allBatches lifetime.finalMemory context.statement.base.segmentCount /\
      (forall receipt,
        receipt ∈ ProductionMemoryRowSegments.receipts allBatches ->
          exists assignment : context.claimLifecycle.FreshAssignment,
            CCS.Holds (ProductPaperAlgebraFor.semantics context.config)
              productionGlobalParams
              (ClaimLifetime.statementOfProtocolClaim
                context.claimLifecycle receipt.claim) assignment) /\
      allBatches = lifetime.rowDelayed.batches := by
  rcases
      ProductionPaperCompactChainLifetimeFor.SemanticLifetime.LifetimeExtraction.precommitChain
        lifetime with
    ⟨allBatches, receiptsExact, rowChain, precommitChain, batchesExact⟩
  refine ⟨allBatches, receiptsExact, rowChain, precommitChain, ?_,
    batchesExact⟩
  intro receipt member
  apply ClaimLifetime.Lifetime.everyConsumedClaimHolds lifetime.claimLifetime
  rw [← lifetime.receipts_eq_consumedReceipts]
  rw [← receiptsExact]
  exact member

end SemanticLifetime

end Nightstream.Implementation.Nebula.ProductionPaperClaimOpeningLifetimeFor
