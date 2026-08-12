import Nightstream.Implementation.NebulaV2.ProductionPaperExactFPrimeLifetimeFor
import Nightstream.Implementation.NebulaV2.ProductionMemoryRowSegments
import Nightstream.Implementation.NebulaV2.ProductionPaperExactLifetime
import Nightstream.Implementation.NebulaV2.ConcreteCompactChain

/-!
Contract: exact compact commitment-chain extraction for the delayed F-prime
lifetime at one generated relation exponent.

Every consumed full claim is constrained by the recursive rows. Those rows
select the exact mandatory commitment bundle, the one `E = 1` memory suffix,
and the operations, initial-snapshot, and final-snapshot Poseidon2 links. The
base rows select the two chain headers from the same verifier-key seed
manifest. Induction over the exact delayed schedule covers the trailing claim.

This module proves deterministic row semantics only. It does not assume or
prove Poseidon2 collision resistance, compact-token binding, NIFS extraction,
commit-phase sequence-to-`D_pre` compiler refinement, generated-artifact
containment, Rust refinement, or terminal-backend soundness.

Assurance tier: exponent-indexed compact-chain lifetime extraction.

Emits constraints: no; it composes row-soundness theorems.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.NebulaV2.ProductionPaperCompactChainLifetimeFor

open Nightstream.Implementation.NebulaV2
open Nightstream.Implementation.R1CS
open Nightstream.Protocol.NebulaV2
open Nightstream.Protocol.NebulaV2.CommitmentBundle
open Nightstream.Protocol.NebulaV2.ExactDelayedSchedule
open Nightstream.Protocol.NebulaV2.FPrime
open Nightstream.Protocol.NebulaV2.ProductionProfileCandidates
open Nightstream.SuperNeo.Concrete

noncomputable local instance concreteKField : Field K :=
  ConcreteField.superNeoEquiv.field

namespace ClaimLifetime

open ProductionPaperExactFPrimeLifetimeFor

/-- Pure semantics of one compact-chain lane update. The leaf digest is an
existential output of the fixed Poseidon2 function, not an authority input. -/
def LaneStepExact
    (manifest : SeedSchedule.Manifest) (role : CompactCommit.Role)
    (index : Fin Lifecycle.claimsPerSegment)
    (commitment : CompactCommit.CommitmentEncoding)
    (prior after : Digest.Value) : Prop :=
  exists leafDigest : Digest.Value,
    (forall lane : Fin 4,
      (leafDigest.lanes lane).val =
        CompactChainPoseidonRows.pureHash
          (.leaf role manifest.profile manifest.plan
            ((CompactTokenRows.key manifest).token role commitment)) lane.val) /\
    (forall lane : Fin 4,
      (after.lanes lane).val =
        CompactChainPoseidonRows.pureHash
          (.link role index prior leafDigest) lane.val)

/-- Row-independent statement for both verifier-key-selected chain headers. -/
def HeadersExact (manifest : SeedSchedule.Manifest)
    (headers : ChainHeaders Digest.Value) : Prop :=
  (forall lane : Fin 4,
    (headers.operations.lanes lane).val =
      CompactChainPoseidonRows.pureHash
        (.header .operations manifest.profile manifest.plan) lane.val) /\
  (forall lane : Fin 4,
    (headers.memory.lanes lane).val =
      CompactChainPoseidonRows.pureHash
        (.header .memory manifest.profile manifest.plan) lane.val)

/-- Exact compact-chain content of one complete verified protocol claim. -/
def ClaimExact
    {Program : Type} (context : Context Program)
    (claim : context.ProtocolClaim) : Prop :=
  context.candidate = .e1 /\
    exists suffix : MemoryClaimCodec.Claim,
      claim.memory.suffixes = [suffix] /\
      LaneStepExact context.seedManifest .operations suffix.stepIndex
        (claim.commitmentBundle .operations)
        suffix.dSeenBefore.operations suffix.dSeenAfter.operations /\
      LaneStepExact context.seedManifest .memory suffix.stepIndex
        (claim.commitmentBundle .initialSnapshot)
        suffix.dSeenBefore.initialSnapshot suffix.dSeenAfter.initialSnapshot /\
      LaneStepExact context.seedManifest .memory suffix.stepIndex
        (claim.commitmentBundle .finalSnapshot)
        suffix.dSeenBefore.finalSnapshot suffix.dSeenAfter.finalSnapshot

namespace LaneStepExact

/-- Remove layout and assignment data from one row-derived lane theorem. -/
theorem ofRows
    {manifest : SeedSchedule.Manifest} {role : CompactCommit.Role}
    {index : Fin Lifecycle.claimsPerSegment}
    {layout : CompactLaneStepRows.Layout} {assignment : Nat -> Nat}
    {canonical : forall column, assignment column < goldilocksP}
    {commitment : CompactCommit.CommitmentEncoding}
    {prior after : Digest.Value}
    (exact : CompactCheckedStepChainRows.LaneExact manifest role index layout
      assignment canonical commitment prior after) :
    LaneStepExact manifest role index commitment prior after := by
  exact ⟨CompactLaneStepRows.outputDigest layout.leafTrace assignment canonical,
    exact.1, exact.2⟩

/-- One extracted row step is exactly `CompactChain.next` for the canonical
Poseidon2 function and the verifier-key-selected compact-token key. -/
theorem after_eq_next
    {manifest : SeedSchedule.Manifest} {role : CompactCommit.Role}
    {index : Fin Lifecycle.claimsPerSegment}
    {commitment : CompactCommit.CommitmentEncoding}
    {prior after : Digest.Value}
    (exact : LaneStepExact manifest role index commitment prior after) :
    after =
      CompactChain.next ConcreteCompactChain.hash
        (CompactTokenRows.key manifest) role manifest.profile manifest.plan
        index prior commitment := by
  rcases exact with ⟨leafDigest, leafExact, afterExact⟩
  have leafDigestExact :
      leafDigest = ConcreteCompactChain.hash
        (.leaf role manifest.profile manifest.plan
          ((CompactTokenRows.key manifest).token role commitment)) := by
    apply Digest.Value.ext
    funext lane
    apply Subtype.ext
    exact leafExact lane
  rw [leafDigestExact] at afterExact
  rw [CompactChain.next]
  apply Digest.Value.ext
  funext lane
  apply Subtype.ext
  exact afterExact lane

end LaneStepExact

namespace HeadersExact

/-- The two extracted header roots are the exact canonical Poseidon2 header
values used by `CompactChain.chainRoot`. -/
theorem roots_eq_hash_headers
    {manifest : SeedSchedule.Manifest}
    {headers : ChainHeaders Digest.Value}
    (exact : HeadersExact manifest headers) :
    headers.operations = ConcreteCompactChain.hash
        (.header .operations manifest.profile manifest.plan) /\
      headers.memory = ConcreteCompactChain.hash
        (.header .memory manifest.profile manifest.plan) := by
  constructor
  · apply Digest.Value.ext
    funext lane
    apply Subtype.ext
    exact exact.1 lane
  · apply Digest.Value.ext
    funext lane
    apply Subtype.ext
    exact exact.2 lane

end HeadersExact

/-- The three authority-bearing precommit sequences. Initial and final
snapshots share the memory hash role but retain different bundle components
and different carried roots. -/
inductive PrecommitLane where
  | operations
  | initialSnapshot
  | finalSnapshot
deriving DecidableEq, Repr

namespace PrecommitLane

def component : PrecommitLane -> CommitmentBundle.Component
  | .operations => .operations
  | .initialSnapshot => .initialSnapshot
  | .finalSnapshot => .finalSnapshot

def role : PrecommitLane -> CompactCommit.Role
  | .operations => .operations
  | .initialSnapshot | .finalSnapshot => .memory

def domain : PrecommitLane -> SequenceBinding.LaneDomain
  | .operations => .operations
  | .initialSnapshot | .finalSnapshot => .memory

def root {Digest : Type} : PrecommitLane -> Roots Digest -> Digest
  | .operations, roots => roots.operations
  | .initialSnapshot, roots => roots.initialSnapshot
  | .finalSnapshot, roots => roots.finalSnapshot

@[simp] theorem roleOfDomain_domain (lane : PrecommitLane) :
    CompactChain.roleOfDomain lane.domain = lane.role := by
  cases lane <;> rfl

end PrecommitLane

namespace HeadersExact

/-- Header equality projected to any of the three precommit lanes. -/
theorem lane_root_eq_hash_header
    {manifest : SeedSchedule.Manifest}
    {headers : ChainHeaders Digest.Value}
    (exact : HeadersExact manifest headers) (lane : PrecommitLane) :
    lane.root headers.roots = ConcreteCompactChain.hash
      (.header lane.role manifest.profile manifest.plan) := by
  rcases exact.roots_eq_hash_headers with ⟨operations, memory⟩
  cases lane with
  | operations => exact operations
  | initialSnapshot => exact memory
  | finalSnapshot => exact memory

end HeadersExact

namespace ClaimExact

/-- Select one of the three lane equations from one exact complete claim.
The supplied suffix must be the unique suffix carried by that claim. -/
theorem laneStep_of_suffix
    {Program : Type} {context : Context Program}
    {claim : context.ProtocolClaim} {suffix : MemoryClaimCodec.Claim}
    (exact : ClaimExact context claim)
    (suffixExact : claim.memory.suffixes = [suffix])
    (lane : PrecommitLane) :
    LaneStepExact context.seedManifest lane.role suffix.stepIndex
      (claim.commitmentBundle lane.component)
      (lane.root suffix.dSeenBefore) (lane.root suffix.dSeenAfter) := by
  rcases exact with
    ⟨_candidateExact, exactSuffix, exactSuffixList,
      operations, initialSnapshot, finalSnapshot⟩
  have suffixEqual : exactSuffix = suffix := by
    have singletonEqual : [exactSuffix] = [suffix] :=
      exactSuffixList.symm.trans suffixExact
    exact List.singleton_inj.mp singletonEqual
  subst exactSuffix
  cases lane with
  | operations => exact operations
  | initialSnapshot => exact initialSnapshot
  | finalSnapshot => exact finalSnapshot

end ClaimExact

/-- An `e1` checked batch contains exactly the suffix used by the compact
chain rows. No second suffix can remain outside the chain. -/
private theorem suffixBatch_singleton
    {candidate : Id}
    {layout : ProductionMemoryCheckedBatchRows.Layout candidate}
    {assignment : Nat -> Nat} {headers : ChainHeaders Digest.Value}
    (result : ProductionMemoryCheckedBatchRows.Result layout assignment headers)
    (candidateExact : candidate = .e1) :
    result.suffixBatch.suffixes =
      [result.claim
        (ProductionFieldNativeCompactChainRowsFor.firstStep candidate)] := by
  subst candidate
  simp [ProductionMemoryCheckedBatchRows.Result.suffixBatch,
    ProductionMemoryCheckedBatchRows.StepCount,
    ProductionProfileCandidates.checkedStepsPerFreshClaim,
    ProductionFieldNativeCompactChainRowsFor.firstStep]

/-- One accepted recursive assignment fixes the exact compact-chain statement
for the complete claim consumed by that assignment. -/
theorem recursive_claim_exact
    {Program : Type} {context : Context Program}
    {previous : context.Claim} (node : RecursiveNode context previous) :
    ClaimExact context node.recursive.verified.claim := by
  have compactValid := node.recursive.compactValid
  have compactExact := node.recursive.compactExact
  rw [node.seedManifestExact] at compactValid compactExact
  let suffix := node.recursive.memoryResult.claim
    (ProductionFieldNativeCompactChainRowsFor.firstStep context.candidate)
  refine ⟨compactValid.candidateExact, suffix, ?_, ?_, ?_, ?_⟩
  · calc
      node.recursive.verified.claim.memory.suffixes =
          previous.memory.suffixes := by
        rw [node.recursive.claimExact]
        rfl
      _ = node.recursive.memoryResult.suffixBatch.suffixes :=
        congrArg (fun memory => memory.suffixes)
          node.recursive.memoryExact.symm
      _ = [suffix] := suffixBatch_singleton node.recursive.memoryResult
        compactValid.candidateExact
  · rw [node.recursive.claimExact]
    exact LaneStepExact.ofRows compactExact.operations
  · rw [node.recursive.claimExact]
    exact LaneStepExact.ofRows compactExact.initialSnapshot
  · rw [node.recursive.claimExact]
    exact LaneStepExact.ofRows compactExact.finalSnapshot

/-- The trailing terminal assignment has the same exact compact-chain
extraction as a recursive assignment. -/
theorem terminal_claim_exact
    {Program : Type} {context : Context Program}
    {previous : context.Claim} (node : TerminalNode context previous) :
    ClaimExact context node.recursive.verified.claim := by
  have compactValid := node.recursive.compactValid
  have compactExact := node.recursive.compactExact
  have manifestExact :
      node.recursive.compactManifest = context.seedManifest := by
    simpa [TerminalNode.recursive] using node.compactManifestExact
  rw [manifestExact] at compactValid compactExact
  let suffix := node.recursive.memoryResult.claim
    (ProductionFieldNativeCompactChainRowsFor.firstStep context.candidate)
  refine ⟨compactValid.candidateExact, suffix, ?_, ?_, ?_, ?_⟩
  · calc
      node.recursive.verified.claim.memory.suffixes =
          previous.memory.suffixes := by
        rw [node.recursive.claimExact]
        rfl
      _ = node.recursive.memoryResult.suffixBatch.suffixes :=
        congrArg (fun memory => memory.suffixes)
          node.recursive.memoryExact.symm
      _ = [suffix] := suffixBatch_singleton node.recursive.memoryResult
        compactValid.candidateExact
  · rw [node.recursive.claimExact]
    exact LaneStepExact.ofRows compactExact.operations
  · rw [node.recursive.claimExact]
    exact LaneStepExact.ofRows compactExact.initialSnapshot
  · rw [node.recursive.claimExact]
    exact LaneStepExact.ofRows compactExact.finalSnapshot

/-- Base rows derive the initial compact-chain headers from the same seed
manifest used by every recursive and terminal compact-chain block. -/
theorem base_headers_exact
    {Program : Type} {context : Context Program} (node : BaseNode context) :
    HeadersExact context.seedManifest context.headers := by
  have exact := node.baseRows.call.headersExact node.baseRows.satisfies
  rw [node.supplement.headersExact] at exact
  rw [node.seedManifestExact] at exact
  exact exact

namespace Schedule

/-- Every receipt, including the trailing terminal receipt, has exact compact
chain semantics derived from the assignment that verified it. -/
theorem everyReceiptExact
    {Program : Type} {context : Context Program} {previous : context.Claim}
    (schedule : Schedule context previous) :
    forall receipt, receipt ∈ ExactDelayedSchedule.Schedule.receipts schedule ->
      ClaimExact context receipt.claim := by
  induction schedule with
  | terminal event =>
      intro receipt member
      simp only [ExactDelayedSchedule.Schedule.receipts,
        ProductionPaperExactFPrimeLifetimeFor.scheduleInterface,
        List.mem_singleton] at member
      subst receipt
      exact terminal_claim_exact event.node
  | recursive event rest inductionHypothesis =>
      intro receipt member
      simp only [ExactDelayedSchedule.Schedule.receipts,
        ProductionPaperExactFPrimeLifetimeFor.scheduleInterface,
        List.mem_cons] at member
      rcases member with rfl | later
      · exact recursive_claim_exact event.node
      · exact inductionHypothesis receipt later

/-- Every produced complete claim is the exact claim consumed by one later
recursive or terminal assignment, so compact-chain coverage is total. -/
theorem everyClaimExact
    {Program : Type} {context : Context Program} {previous : context.Claim}
    (schedule : Schedule context previous) :
    forall claim, claim ∈ ExactDelayedSchedule.Schedule.claims schedule ->
      ClaimExact context claim.toProtocolClaim := by
  induction schedule with
  | terminal event =>
      intro claim member
      simp only [ExactDelayedSchedule.Schedule.claims,
        List.mem_singleton] at member
      subst claim
      have exact := terminal_claim_exact event.node
      rw [event.node.recursive.claimExact] at exact
      exact exact
  | recursive event rest inductionHypothesis =>
      intro claim member
      simp only [ExactDelayedSchedule.Schedule.claims, List.mem_cons] at member
      rcases member with rfl | later
      · have exact := recursive_claim_exact event.node
        rw [event.node.recursive.claimExact] at exact
        exact exact
      · exact inductionHypothesis claim later

/-- Existence of any closed delayed schedule derives selection of the only
currently compiled compact-chain relation, `e1`. -/
theorem candidateExact
    {Program : Type} {context : Context Program} {previous : context.Claim}
    (schedule : Schedule context previous) : context.candidate = .e1 := by
  cases schedule with
  | terminal event => exact event.node.recursive.compactValid.candidateExact
  | recursive event rest =>
      exact event.node.recursive.compactValid.candidateExact

end Schedule

/-- Combined compact-chain result for the complete delayed F-prime lifetime. -/
structure ExactLifetime
    {Program : Type} {context : Context Program}
    (lifetime : Lifetime context) : Prop where
  headers : HeadersExact context.seedManifest context.headers
  candidate : context.candidate = .e1
  selectedProfile : context.seedManifest.profile = identity .e1
  receipts : forall receipt, receipt ∈ lifetime.consumedReceipts ->
    ClaimExact context receipt.claim
  claims : forall claim, claim ∈ lifetime.producedClaims ->
    ClaimExact context claim.toProtocolClaim

namespace Lifetime

/-- Main compact-chain lifetime theorem. Its inputs are only the exact
row-derived base, recursive, and terminal nodes in `Lifetime`. -/
theorem compactExact
    {Program : Type} {context : Context Program} (lifetime : Lifetime context) :
    ExactLifetime lifetime := by
  have candidate := Schedule.candidateExact lifetime.schedule
  refine
    { headers := base_headers_exact lifetime.base
      candidate := candidate
      selectedProfile := ?_
      receipts := ?_
      claims := ?_ }
  · calc
      context.seedManifest.profile = identity context.candidate :=
        context.seedManifestProfile
      _ = identity .e1 := congrArg identity candidate
  · intro receipt member
    exact Schedule.everyReceiptExact lifetime.schedule receipt member
  · intro claim member
    exact Schedule.everyClaimExact lifetime.schedule claim member

end Lifetime

end ClaimLifetime

/-! ## Exact root continuity -/

/-- One nonempty ordered suffix list whose seen roots form one exact chain. -/
inductive RootRun :
    Roots Digest.Value -> List MemoryClaimCodec.Claim ->
      Roots Digest.Value -> Prop
  | one
      {before after : Roots Digest.Value} {claim : MemoryClaimCodec.Claim}
      (beforeExact : claim.dSeenBefore = before)
      (afterExact : claim.dSeenAfter = after) :
      RootRun before [claim] after
  | cons
      {before after : Roots Digest.Value} {claim : MemoryClaimCodec.Claim}
      {tail : List MemoryClaimCodec.Claim}
      (beforeExact : claim.dSeenBefore = before)
      (rest : RootRun claim.dSeenAfter tail after) :
      RootRun before (claim :: tail) after

/-- General indexed form used to avoid assuming that the input or output
carry has the requested phase. -/
private theorem rootRun_of_consumes
    {balanced : ProductState.State K -> Prop}
    {before after : Carry Digest.Value (ProductState.Challenges K)
      (ProductState.State K)}
    {claims : List MemoryClaimCodec.Claim}
    (consumes : ProductionBatchedFPrime.ConsumesList balanced before claims
      after) :
    forall active closed, before = .active active -> after = .closed closed ->
      RootRun active.dSeen claims active.dPre := by
  induction consumes with
  | nil =>
      intro active closed beforeExact afterExact
      rw [beforeExact] at afterExact
      cases afterExact
  | @cons before middle after claim tail step rest inductionHypothesis =>
      intro active closed beforeExact afterExact
      cases step with
      | interior agreement notLast =>
          cases beforeExact
          have tailRun := inductionHypothesis
            (interiorCarry active claim notLast) closed rfl afterExact
          exact .cons agreement.dSeen (by
            simpa [interiorCarry] using tailRun)
      | close agreement last checks =>
          cases beforeExact
          have empty := rest.from_closed_is_empty
          rw [empty.1]
          exact .one agreement.dSeen checks.seenEqualsPrecommit

/-- F-prime consumption from an active carry through close derives exact
seen-root continuity from the opening header to the precommit root. -/
theorem rootRun_of_consumes_to_closed
    {balanced : ProductState.State K -> Prop}
    {active : ActiveCarry Digest.Value (ProductState.Challenges K)
      (ProductState.State K)}
    {closed : ClosedCarry Digest.Value}
    {claims : List MemoryClaimCodec.Claim}
    (consumes : ProductionBatchedFPrime.ConsumesList balanced
      (.active active) claims (.closed closed)) :
    RootRun active.dSeen claims active.dPre :=
  rootRun_of_consumes consumes active closed rfl rfl

namespace ClaimLifetime

open ProductionPaperExactFPrimeLifetimeFor

/-- One reconstructed segment has exact compact semantics for every complete
claim and one continuous root run from the fixed headers to `D_pre`. -/
structure SegmentExact
    {Program : Type} (context : Context Program)
    {before : ClosedCarry Digest.Value}
    (run : ProductionMemoryRowSegments.SegmentRun context.candidate
      context.Schema context.Verifier context.headers before) : Prop where
  claims : forall receipt,
    receipt ∈ ProductionMemoryRowSegments.receipts run.batches ->
      ClaimExact context receipt.claim
  roots : RootRun context.headers.roots
    ((ProductionMemoryRowSegments.receipts run.batches).flatMap
      fun receipt => receipt.claim.memory.suffixes) run.precommit

/-- Exact order-preserving pairing between verified complete claims and the
single memory suffix carried by each selected `e1` claim. Every constructor
retains the row-derived compact-chain equations for that same claim bundle. -/
inductive VerifiedBundleSequence
    {Program : Type} (context : Context Program) :
    List context.Receipt -> List MemoryClaimCodec.Claim -> Prop
  | nil : VerifiedBundleSequence context [] []
  | cons
      {receipt : context.Receipt} {receipts : List context.Receipt}
      {suffix : MemoryClaimCodec.Claim}
      {suffixes : List MemoryClaimCodec.Claim}
      (claimExact : ClaimExact context receipt.claim)
      (suffixExact : receipt.claim.memory.suffixes = [suffix])
      (rest : VerifiedBundleSequence context receipts suffixes) :
      VerifiedBundleSequence context (receipt :: receipts)
        (suffix :: suffixes)

namespace VerifiedBundleSequence

/-- Exact indexed commitment list retained by one verified bundle sequence.
This definition is intentionally recursive: it cannot compact away a hole or
reorder a claim. -/
def pairedBundles
    {Program : Type} (context : Context Program) (lane : PrecommitLane) :
    List context.Receipt -> List MemoryClaimCodec.Claim ->
      List (Fin Lifecycle.claimsPerSegment ×
        CompactCommit.CommitmentEncoding)
  | receipt :: receipts, suffix :: suffixes =>
      (suffix.stepIndex,
        receipt.claim.commitmentBundle lane.component) ::
          pairedBundles context lane receipts suffixes
  | _, _ => []

/-- The pairing cannot omit or duplicate a complete claim position. -/
theorem length_exact
    {Program : Type} {context : Context Program}
    {receipts : List context.Receipt}
    {suffixes : List MemoryClaimCodec.Claim}
    (sequence : VerifiedBundleSequence context receipts suffixes) :
    suffixes.length = receipts.length := by
  induction sequence with
  | nil => rfl
  | cons _ _ rest inductionHypothesis =>
      simp [inductionHypothesis]

/-- The paired commitment list has one entry for every verified receipt. -/
theorem pairedBundles_length_receipts
    {Program : Type} {context : Context Program}
    {receipts : List context.Receipt}
    {suffixes : List MemoryClaimCodec.Claim}
    (sequence : VerifiedBundleSequence context receipts suffixes)
    (lane : PrecommitLane) :
    (pairedBundles context lane receipts suffixes).length = receipts.length := by
  induction sequence with
  | nil => rfl
  | cons _ _ rest inductionHypothesis =>
      simp [pairedBundles, inductionHypothesis]

/-- The paired commitment list has one entry for every exact suffix. -/
theorem pairedBundles_length_suffixes
    {Program : Type} {context : Context Program}
    {receipts : List context.Receipt}
    {suffixes : List MemoryClaimCodec.Claim}
    (sequence : VerifiedBundleSequence context receipts suffixes)
    (lane : PrecommitLane) :
    (pairedBundles context lane receipts suffixes).length = suffixes.length := by
  rw [pairedBundles_length_receipts sequence lane, sequence.length_exact]

/-- The first coordinate at each paired-list position is the step index of
the suffix at the same position. -/
theorem pairedBundles_get_fst
    {Program : Type} {context : Context Program}
    {receipts : List context.Receipt}
    {suffixes : List MemoryClaimCodec.Claim}
    (sequence : VerifiedBundleSequence context receipts suffixes)
    (lane : PrecommitLane)
    (position : Fin (pairedBundles context lane receipts suffixes).length) :
    ((pairedBundles context lane receipts suffixes).get position).1 =
      (suffixes.get
        (Fin.cast (pairedBundles_length_suffixes sequence lane) position)).stepIndex := by
  induction sequence with
  | nil => exact Fin.elim0 position
  | @cons receipt receipts suffix suffixes claimExact suffixExact rest
      inductionHypothesis =>
      refine Fin.cases ?_ (fun tailPosition => ?_) position
      · rfl
      · simpa [pairedBundles] using inductionHypothesis tailPosition

/-- Every stored step index equals its physical list position. -/
def CanonicallyIndexed
    (bundles : List (Fin Lifecycle.claimsPerSegment ×
      CompactCommit.CommitmentEncoding)) : Prop :=
  forall position : Fin bundles.length,
    (bundles.get position).1.val = position.val

/-- Convert one exact-length list into the total commitment vector required
by `FramedSequence`. -/
def commitmentVector
    (bundles : List (Fin Lifecycle.claimsPerSegment ×
      CompactCommit.CommitmentEncoding))
    (lengthExact : bundles.length = Lifecycle.claimsPerSegment) :
    Fin Lifecycle.claimsPerSegment -> CompactCommit.CommitmentEncoding :=
  fun position =>
    (bundles.get (Fin.cast lengthExact.symm position)).2

/-- Verifier-key-framed sequence for one of the three exact precommit lanes. -/
def framedSequence
    {Program : Type} (context : Context Program) (lane : PrecommitLane)
    (bundles : List (Fin Lifecycle.claimsPerSegment ×
      CompactCommit.CommitmentEncoding))
    (lengthExact : bundles.length = Lifecycle.claimsPerSegment) :
    SequenceBinding.FramedSequence Profile.Identity Digest.Value
      CompactCommit.CommitmentEncoding where
  profile := context.seedManifest.profile
  plan := context.seedManifest.plan
  domain := lane.domain
  commitments := commitmentVector bundles lengthExact

/-- Canonical physical positions reconstruct the exact indexed list used by
`CompactChain.run`; no index is taken from the prover. -/
theorem ofFn_commitmentVector_eq
    (bundles : List (Fin Lifecycle.claimsPerSegment ×
      CompactCommit.CommitmentEncoding))
    (lengthExact : bundles.length = Lifecycle.claimsPerSegment)
    (canonical : CanonicallyIndexed bundles) :
    List.ofFn (fun position : Fin Lifecycle.claimsPerSegment =>
      (position, commitmentVector bundles lengthExact position)) = bundles := by
  have functionsEqual :
      (fun position : Fin Lifecycle.claimsPerSegment =>
        (position, commitmentVector bundles lengthExact position)) =
      (fun position : Fin Lifecycle.claimsPerSegment =>
        bundles.get (Fin.cast lengthExact.symm position)) := by
    funext position
    apply Prod.ext
    · apply Fin.ext
      exact (canonical (Fin.cast lengthExact.symm position)).symm
    · rfl
  rw [functionsEqual]
  have reindexed := List.ofFn_congr lengthExact (List.get bundles)
  rw [List.ofFn_get] at reindexed
  exact reindexed.symm

/-- The framed sequence root is the same concrete run over the complete
canonical indexed list. -/
theorem chainRoot_framedSequence
    {Program : Type} (context : Context Program) (lane : PrecommitLane)
    (bundles : List (Fin Lifecycle.claimsPerSegment ×
      CompactCommit.CommitmentEncoding))
    (lengthExact : bundles.length = Lifecycle.claimsPerSegment)
    (canonical : CanonicallyIndexed bundles) :
    CompactChain.chainRoot ConcreteCompactChain.hash
        (CompactTokenRows.key context.seedManifest)
        (framedSequence context lane bundles lengthExact) =
      CompactChain.run ConcreteCompactChain.hash
        (CompactTokenRows.key context.seedManifest) lane.role
        context.seedManifest.profile context.seedManifest.plan
        (ConcreteCompactChain.hash
          (.header lane.role context.seedManifest.profile
            context.seedManifest.plan)) bundles := by
  rw [CompactChain.chainRoot]
  simp only [framedSequence, PrecommitLane.roleOfDomain_domain]
  rw [ofFn_commitmentVector_eq bundles lengthExact canonical]

/-- A canonical suffix schedule transfers directly to the paired complete
bundle list. -/
theorem pairedBundles_canonicallyIndexed
    {Program : Type} {context : Context Program}
    {receipts : List context.Receipt}
    {suffixes : List MemoryClaimCodec.Claim}
    (sequence : VerifiedBundleSequence context receipts suffixes)
    (lane : PrecommitLane)
    (stepAt : forall position : Fin suffixes.length,
      (suffixes.get position).stepIndex.val = position.val) :
    CanonicallyIndexed (pairedBundles context lane receipts suffixes) := by
  intro position
  rw [pairedBundles_get_fst sequence lane position]
  exact stepAt
    (Fin.cast (pairedBundles_length_suffixes sequence lane) position)

/-- Exact claim equations plus exact seen-root continuity evaluate the whole
paired list under the canonical compact-chain function. -/
theorem run_eq_root
    {Program : Type} {context : Context Program}
    {receipts : List context.Receipt}
    {suffixes : List MemoryClaimCodec.Claim}
    {before after : Roots Digest.Value}
    (sequence : VerifiedBundleSequence context receipts suffixes)
    (roots : RootRun before suffixes after)
    (lane : PrecommitLane) :
    CompactChain.run ConcreteCompactChain.hash
        (CompactTokenRows.key context.seedManifest) lane.role
        context.seedManifest.profile context.seedManifest.plan
        (lane.root before)
        (pairedBundles context lane receipts suffixes) =
      lane.root after := by
  induction sequence generalizing before after with
  | nil => cases roots
  | @cons receipt receipts suffix suffixes claimExact suffixExact rest
      inductionHypothesis =>
      cases roots with
      | one beforeExact afterExact =>
          cases rest
          have step :=
            (claimExact.laneStep_of_suffix suffixExact lane).after_eq_next
          simp only [pairedBundles, CompactChain.run]
          rw [← congrArg lane.root beforeExact]
          rw [← step]
          exact congrArg lane.root afterExact
      | cons beforeExact rootsRest =>
          have step :=
            (claimExact.laneStep_of_suffix suffixExact lane).after_eq_next
          simp only [pairedBundles, CompactChain.run]
          rw [← congrArg lane.root beforeExact]
          rw [← step]
          exact inductionHypothesis rootsRest

/-- Build the order-preserving sequence only from exact claims already
extracted from recursive or terminal rows. -/
theorem of_claims
    {Program : Type} {context : Context Program}
    (receipts : List context.Receipt)
    (claims : forall receipt, receipt ∈ receipts ->
      ClaimExact context receipt.claim) :
    VerifiedBundleSequence context receipts
      (receipts.flatMap fun receipt => receipt.claim.memory.suffixes) := by
  induction receipts with
  | nil => exact .nil
  | cons receipt receipts inductionHypothesis =>
      have headExact := claims receipt (by simp)
      rcases headExact with
        ⟨candidateExact, suffix, suffixExact,
          operations, initialSnapshot, finalSnapshot⟩
      have tailClaims : forall tailReceipt, tailReceipt ∈ receipts ->
          ClaimExact context tailReceipt.claim := by
        intro tailReceipt member
        exact claims tailReceipt (by simp [member])
      have tail := inductionHypothesis tailClaims
      rw [List.flatMap_cons, suffixExact]
      exact .cons
        ⟨candidateExact, suffix, suffixExact,
          operations, initialSnapshot, finalSnapshot⟩
        suffixExact tail

end VerifiedBundleSequence

/-- Row-derived prechallenge authority for one complete segment. `D_pre` is
the endpoint of the ordered compact-chain run over exactly 1,088 verified
atomic claim bundles. No prover-supplied sequence or root-correctness premise
is a field. -/
structure SegmentPrecommitExact
    {Program : Type} (context : Context Program)
    {before : ClosedCarry Digest.Value}
    (run : ProductionMemoryRowSegments.SegmentRun context.candidate
      context.Schema context.Verifier context.headers before) : Prop where
  headers : HeadersExact context.seedManifest context.headers
  candidate : context.candidate = .e1
  bundles : VerifiedBundleSequence context
    (ProductionMemoryRowSegments.receipts run.batches)
    ((ProductionMemoryRowSegments.receipts run.batches).flatMap
      fun receipt => receipt.claim.memory.suffixes)
  roots : RootRun context.headers.roots
    ((ProductionMemoryRowSegments.receipts run.batches).flatMap
      fun receipt => receipt.claim.memory.suffixes) run.precommit
  receiptCount :
    (ProductionMemoryRowSegments.receipts run.batches).length =
      Lifecycle.claimsPerSegment

namespace SegmentPrecommitExact

def laneBundles
    {Program : Type} {context : Context Program}
    {before : ClosedCarry Digest.Value}
    {run : ProductionMemoryRowSegments.SegmentRun context.candidate
      context.Schema context.Verifier context.headers before}
    (_exact : SegmentPrecommitExact context run) (lane : PrecommitLane) :
    List (Fin Lifecycle.claimsPerSegment ×
      CompactCommit.CommitmentEncoding) :=
  VerifiedBundleSequence.pairedBundles context lane
    (ProductionMemoryRowSegments.receipts run.batches)
    ((ProductionMemoryRowSegments.receipts run.batches).flatMap
      fun receipt => receipt.claim.memory.suffixes)

theorem laneBundles_length
    {Program : Type} {context : Context Program}
    {before : ClosedCarry Digest.Value}
    {run : ProductionMemoryRowSegments.SegmentRun context.candidate
      context.Schema context.Verifier context.headers before}
    (exact : SegmentPrecommitExact context run) (lane : PrecommitLane) :
    (exact.laneBundles lane).length = Lifecycle.claimsPerSegment := by
  rw [laneBundles,
    VerifiedBundleSequence.pairedBundles_length_receipts exact.bundles lane,
    exact.receiptCount]

/-- The F-prime state transition, not a prover list, fixes suffix index `i`
to physical position `i`. -/
theorem suffix_step_at
    {Program : Type} {context : Context Program}
    {before : ClosedCarry Digest.Value}
    {run : ProductionMemoryRowSegments.SegmentRun context.candidate
      context.Schema context.Verifier context.headers before}
    (_exact : SegmentPrecommitExact context run)
    (position : Fin
      ((ProductionMemoryRowSegments.receipts run.batches).flatMap
        fun (receipt : context.Receipt) =>
          receipt.claim.memory.suffixes).length) :
    (((ProductionMemoryRowSegments.receipts run.batches).flatMap
      fun (receipt : context.Receipt) =>
        receipt.claim.memory.suffixes).get position).stepIndex.val =
        position.val := by
  simpa [ProductionBatchedScanSchedule.SegmentRun.suffixes,
    ProductionMemoryRowSegments.SegmentRun.toProtocol] using
    ProductionBatchedScanSchedule.SegmentRun.suffix_step_at
      run.toProtocol position

theorem laneBundles_canonicallyIndexed
    {Program : Type} {context : Context Program}
    {before : ClosedCarry Digest.Value}
    {run : ProductionMemoryRowSegments.SegmentRun context.candidate
      context.Schema context.Verifier context.headers before}
    (exact : SegmentPrecommitExact context run) (lane : PrecommitLane) :
    VerifiedBundleSequence.CanonicallyIndexed (exact.laneBundles lane) := by
  exact VerifiedBundleSequence.pairedBundles_canonicallyIndexed
    exact.bundles lane exact.suffix_step_at

/-- Exact framed sequence extracted before the challenge from the verified
complete claim bundles. -/
def laneSequence
    {Program : Type} {context : Context Program}
    {before : ClosedCarry Digest.Value}
    {run : ProductionMemoryRowSegments.SegmentRun context.candidate
      context.Schema context.Verifier context.headers before}
    (exact : SegmentPrecommitExact context run) (lane : PrecommitLane) :
    SequenceBinding.FramedSequence Profile.Identity Digest.Value
      CompactCommit.CommitmentEncoding :=
  VerifiedBundleSequence.framedSequence context lane
    (exact.laneBundles lane) (exact.laneBundles_length lane)

/-- This is the concrete `KnownPrecommit` witness required before challenge
derivation. Its sequence, order, framing, and root are all row consequences. -/
def knownPrecommit
    {Program : Type} {context : Context Program}
    {before : ClosedCarry Digest.Value}
    {run : ProductionMemoryRowSegments.SegmentRun context.candidate
      context.Schema context.Verifier context.headers before}
    (exact : SegmentPrecommitExact context run) (lane : PrecommitLane) :
    SequenceBinding.KnownPrecommit
      (CompactChain.chainRoot ConcreteCompactChain.hash
        (CompactTokenRows.key context.seedManifest)) where
  sequence := exact.laneSequence lane
  committedRoot := lane.root run.precommit
  rootCorrect := by
    rw [laneSequence]
    rw [VerifiedBundleSequence.chainRoot_framedSequence context lane
      (exact.laneBundles lane) (exact.laneBundles_length lane)
      (exact.laneBundles_canonicallyIndexed lane)]
    rw [← exact.headers.lane_root_eq_hash_header lane]
    exact VerifiedBundleSequence.run_eq_root exact.bundles exact.roots lane

@[simp] theorem knownPrecommit_root
    {Program : Type} {context : Context Program}
    {before : ClosedCarry Digest.Value}
    {run : ProductionMemoryRowSegments.SegmentRun context.candidate
      context.Schema context.Verifier context.headers before}
    (exact : SegmentPrecommitExact context run) (lane : PrecommitLane) :
    (exact.knownPrecommit lane).committedRoot = lane.root run.precommit := rfl

/-- The constructed witness proves the concrete chain equation. This theorem
audits the proof field, not only the stored root projection. -/
theorem knownPrecommit_correct
    {Program : Type} {context : Context Program}
    {before : ClosedCarry Digest.Value}
    {run : ProductionMemoryRowSegments.SegmentRun context.candidate
      context.Schema context.Verifier context.headers before}
    (exact : SegmentPrecommitExact context run) (lane : PrecommitLane) :
    CompactChain.chainRoot ConcreteCompactChain.hash
        (CompactTokenRows.key context.seedManifest)
        (exact.laneSequence lane) =
      lane.root run.precommit :=
  (exact.knownPrecommit lane).rootCorrect

end SegmentPrecommitExact

namespace SegmentExact

/-- Exact compact rows, the verifier-key headers, and the selected `e1`
profile derive the complete prechallenge bundle sequence ending at `D_pre`. -/
theorem precommitExact
    {Program : Type} {context : Context Program}
    {before : ClosedCarry Digest.Value}
    {run : ProductionMemoryRowSegments.SegmentRun context.candidate
      context.Schema context.Verifier context.headers before}
    (segment : SegmentExact context run)
    (headers : HeadersExact context.seedManifest context.headers)
    (candidate : context.candidate = .e1) :
    SegmentPrecommitExact context run := by
  refine
    { headers := headers
      candidate := candidate
      bundles := VerifiedBundleSequence.of_claims _ segment.claims
      roots := segment.roots
      receiptCount := ?_ }
  rw [ProductionMemoryRowSegments.receipts]
  simp only [List.length_map]
  rw [run.exactBatchCount, candidate]
  rfl

end SegmentExact

/-- Compose exact claim links with the F-prime carry chain for one complete
segment. -/
theorem segment_exact
    {Program : Type} {context : Context Program}
    {before : ClosedCarry Digest.Value}
    (run : ProductionMemoryRowSegments.SegmentRun context.candidate
      context.Schema context.Verifier context.headers before)
    (claims : forall receipt,
      receipt ∈ ProductionMemoryRowSegments.receipts run.batches ->
        ClaimExact context receipt.claim) :
    SegmentExact context run := by
  have rootRun := rootRun_of_consumes_to_closed
    run.consumed.toVerifiedRun.flattenConsumes
  have seenExact : context.headers.roots = run.active.dSeen := by
    simpa [openSegment] using congrArg
      (fun state => match state with
        | .closed _ => context.headers.roots
        | .active active => active.dSeen)
      run.opened
  have precommitExact : run.precommit = run.active.dPre := by
    simpa [openSegment] using congrArg
      (fun state => match state with
        | .closed _ => run.precommit
        | .active active => active.dPre)
      run.opened
  rw [← seenExact, ← precommitExact] at rootRun
  exact ⟨claims, rootRun⟩

/-- Exact compact-chain evidence follows the same segment partition as the
row-derived delayed F-prime trace. -/
inductive ChainExact
    {Program : Type} (context : Context Program) :
    ClosedCarry Digest.Value ->
      List (ProductionMemoryRowSegments.Evidence context.candidate
        context.Schema context.Verifier context.headers) ->
      ClosedCarry Digest.Value -> Nat -> Prop
  | nil (state : ClosedCarry Digest.Value) :
      ChainExact context state [] state 0
  | cons
      {before final : ClosedCarry Digest.Value}
      {tailBatches : List (ProductionMemoryRowSegments.Evidence
        context.candidate context.Schema context.Verifier context.headers)}
      {tailSegments : Nat}
      (head : ProductionMemoryRowSegments.SegmentRun context.candidate
        context.Schema context.Verifier context.headers before)
      (headExact : SegmentExact context head)
      (tail : ChainExact context head.after tailBatches final tailSegments) :
      ChainExact context before (head.batches ++ tailBatches) final
        (tailSegments + 1)

namespace ChainExact

/-- A row-derived segment chain plus row-derived exactness for each receipt
produces the complete compact-chain partition. -/
theorem ofRows
    {Program : Type} {context : Context Program}
    {initial final : ClosedCarry Digest.Value}
    {batches : List (ProductionMemoryRowSegments.Evidence context.candidate
      context.Schema context.Verifier context.headers)}
    {segmentCount : Nat}
    (chain : ProductionMemoryRowSegments.Chain context.candidate context.Schema
      context.Verifier context.headers initial batches final segmentCount)
    (claims : forall receipt,
      receipt ∈ ProductionMemoryRowSegments.receipts batches ->
        ClaimExact context receipt.claim) :
    ChainExact context initial batches final segmentCount := by
  induction chain with
  | nil => exact .nil _
  | @cons before final tailBatches tailSegments head tail inductionHypothesis =>
      have headClaims : forall receipt,
          receipt ∈ ProductionMemoryRowSegments.receipts head.batches ->
            ClaimExact context receipt.claim := by
        intro receipt member
        apply claims receipt
        rw [ProductionMemoryRowSegments.receipts, List.map_append]
        exact List.mem_append_left _ member
      have tailClaims : forall receipt,
          receipt ∈ ProductionMemoryRowSegments.receipts tailBatches ->
            ClaimExact context receipt.claim := by
        intro receipt member
        apply claims receipt
        rw [ProductionMemoryRowSegments.receipts, List.map_append]
        exact List.mem_append_right _ member
      exact .cons head (segment_exact head headClaims)
        (inductionHypothesis tailClaims)

end ChainExact

/-- Every segment in a cross-segment chain retains its complete row-derived
bundle sequence and the exact endpoint `D_pre`. -/
inductive PrecommitChainExact
    {Program : Type} (context : Context Program) :
    ClosedCarry Digest.Value ->
      List (ProductionMemoryRowSegments.Evidence context.candidate
        context.Schema context.Verifier context.headers) ->
      ClosedCarry Digest.Value -> Nat -> Prop
  | nil (state : ClosedCarry Digest.Value) :
      PrecommitChainExact context state [] state 0
  | cons
      {before final : ClosedCarry Digest.Value}
      {tailBatches : List (ProductionMemoryRowSegments.Evidence
        context.candidate context.Schema context.Verifier context.headers)}
      {tailSegments : Nat}
      (head : ProductionMemoryRowSegments.SegmentRun context.candidate
        context.Schema context.Verifier context.headers before)
      (headExact : SegmentPrecommitExact context head)
      (tail : PrecommitChainExact context head.after tailBatches final
        tailSegments) :
      PrecommitChainExact context before (head.batches ++ tailBatches) final
        (tailSegments + 1)

namespace ChainExact

/-- Lift one exact compact chain to the stronger precommit authority theorem.
The header and candidate facts come from the same complete delayed lifetime. -/
theorem precommitExact
    {Program : Type} {context : Context Program}
    {initial final : ClosedCarry Digest.Value}
    {batches : List (ProductionMemoryRowSegments.Evidence context.candidate
      context.Schema context.Verifier context.headers)}
    {segmentCount : Nat}
    (chain : ChainExact context initial batches final segmentCount)
    (headers : HeadersExact context.seedManifest context.headers)
    (candidate : context.candidate = .e1) :
    PrecommitChainExact context initial batches final segmentCount := by
  induction chain with
  | nil state => exact .nil state
  | cons head headExact tail inductionHypothesis =>
      exact .cons head (headExact.precommitExact headers candidate)
        inductionHypothesis

end ChainExact

end ClaimLifetime

/-! ## Semantic lifetime composition -/

namespace SemanticLifetime

open ProductionPaperExactLifetime

/-- The semantic extraction uses the same receipts as the exact delayed claim
schedule. Therefore its row-derived segment partition also has exact compact
chain semantics for every segment and every trailing-consumed claim. -/
theorem LifetimeExtraction.compactChain
    {Program : Type} {context : Context Program}
    {base : BaseNode context} (lifetime : LifetimeExtraction base) :
    exists allBatches,
      ProductionMemoryRowSegments.receipts allBatches = lifetime.receipts /\
      ProductionMemoryRowSegments.Chain context.candidate context.Schema
        context.Verifier context.headers
        (ProductionPaperBaseInvocationFor.initialClosed
          context.authoritativeInitialMemoryRoot)
        allBatches lifetime.finalMemory context.statement.base.segmentCount /\
      ClaimLifetime.ChainExact context.claimLifecycle
        (ProductionPaperBaseInvocationFor.initialClosed
          context.authoritativeInitialMemoryRoot)
        allBatches lifetime.finalMemory context.statement.base.segmentCount /\
      allBatches = lifetime.rowDelayed.batches := by
  rcases lifetime.rowSegmentChain with
    ⟨allBatches, receiptsExact, chain, batchesExact, accessesExact⟩
  have compact := ClaimLifetime.Lifetime.compactExact lifetime.claimLifetime
  have everyClaim : forall receipt,
      receipt ∈ ProductionMemoryRowSegments.receipts allBatches ->
        ClaimLifetime.ClaimExact context.claimLifecycle receipt.claim := by
    intro receipt member
    apply compact.receipts receipt
    rw [← lifetime.receipts_eq_consumedReceipts]
    rw [← receiptsExact]
    exact member
  exact ⟨allBatches, receiptsExact, chain,
    ClaimLifetime.ChainExact.ofRows chain everyClaim, batchesExact⟩

/-- Strong lifetime form: every reconstructed segment has exactly 1,088
ordered verified claim bundles and its compact-chain endpoint is the carried
`D_pre`. The terminal-consumed trailing claim remains in this sequence. -/
theorem LifetimeExtraction.precommitChain
    {Program : Type} {context : Context Program}
    {base : BaseNode context} (lifetime : LifetimeExtraction base) :
    exists allBatches,
      ProductionMemoryRowSegments.receipts allBatches = lifetime.receipts /\
      ProductionMemoryRowSegments.Chain context.candidate context.Schema
        context.Verifier context.headers
        (ProductionPaperBaseInvocationFor.initialClosed
          context.authoritativeInitialMemoryRoot)
        allBatches lifetime.finalMemory context.statement.base.segmentCount /\
      ClaimLifetime.PrecommitChainExact context.claimLifecycle
        (ProductionPaperBaseInvocationFor.initialClosed
          context.authoritativeInitialMemoryRoot)
        allBatches lifetime.finalMemory context.statement.base.segmentCount /\
      allBatches = lifetime.rowDelayed.batches := by
  rcases
      Nightstream.Implementation.NebulaV2.ProductionPaperCompactChainLifetimeFor.SemanticLifetime.LifetimeExtraction.compactChain
        lifetime with
    ⟨allBatches, receiptsExact, rowChain, compactChain, batchesExact⟩
  have compact := ClaimLifetime.Lifetime.compactExact lifetime.claimLifetime
  exact ⟨allBatches, receiptsExact, rowChain,
    compactChain.precommitExact compact.headers compact.candidate,
    batchesExact⟩

end SemanticLifetime

end Nightstream.Implementation.NebulaV2.ProductionPaperCompactChainLifetimeFor
