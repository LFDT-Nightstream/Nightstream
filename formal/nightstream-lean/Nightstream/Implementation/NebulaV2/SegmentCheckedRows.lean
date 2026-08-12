import Nightstream.Implementation.NebulaV2.ApplicationPortRefinement
import Nightstream.Implementation.NebulaV2.MemoryProductBalanceBridge
import Nightstream.Implementation.NebulaV2.RecursiveManifestNifsCall
import Nightstream.Protocol.NebulaV2.ScanSchedule

/-!
Contract: row-derived product accumulation for one complete V2 segment.

Assurance tier: implementation-to-protocol bridge.

Owns a checked recursive invocation, exact chaining of those invocations,
conversion to the independent delayed full-claim run, and the proof that all
1,088 row-derived record chunks accumulate to a balanced two-repetition
product state.

The invocation premises are the generated-manifest assignment, its complete
row satisfaction, its exact selected-verifier call, and its parsed carry
blocks. No premise states a product update or a final product equation.

Does not prove that opaque NIFS rows imply `verifierAccepted`, construct a
generated artifact, prove record coverage by application semantics, or price
fingerprint failure.

Emits constraints: through `RecursiveManifestSchema.Artifact.programRows`.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.NebulaV2.SegmentCheckedRows

open Nightstream.Implementation.NebulaV2.ConcreteField
open Nightstream.Implementation.NebulaV2.ApplicationPortRefinement
open Nightstream.Implementation.NebulaV2.FullClaimEnvelope
open Nightstream.Implementation.NebulaV2.FullClaimNifsReceipt
open Nightstream.Implementation.NebulaV2.MemoryClaimProductUpdate
open Nightstream.Implementation.NebulaV2.MemoryProductBalanceBridge
open Nightstream.Implementation.NebulaV2.MemoryProductBalanceRows
open Nightstream.Implementation.NebulaV2.RecursiveManifestNifsCall
open Nightstream.Implementation.NebulaV2.RecursiveManifestSchema
open Nightstream.Implementation.R1CS
open Nightstream.Protocol.NebulaV2
open Nightstream.Protocol.NebulaV2.FPrime
open Nightstream.Protocol.NebulaV2.FullClaim
open Nightstream.Protocol.NebulaV2.ProductState
open Nightstream.SuperNeo.Concrete

abbrev ConcreteCarry := Carry Digest.Value (Challenges K) (State K)

/-- One exact invocation of the selected recursive relation. All semantic
facts below are derived from these fields. -/
structure Invocation
    {widths : CompilerWidths}
    (artifact : Artifact widths) (selected : SelectedVerifier widths) where
  assignment : Nat → Nat
  call : RecursiveManifestNifsCall.Call artifact selected assignment
  carry : call.CarryBlocks
  satisfies : Satisfies artifact.programRows assignment

namespace Invocation

variable {widths : CompilerWidths}
variable {artifact : Artifact widths} {selected : SelectedVerifier widths}

def receipt (invocation : Invocation artifact selected) : Receipt selected :=
  { claim := invocation.call.claim
    proof := invocation.call.proof
    output := invocation.call.output
    accepted := ⟨invocation.call.claimCanonical,
      by
        have inputExact :=
          invocation.call.satisfying_manifest_binds_exact_nifs_input
            invocation.satisfies
        rw [← inputExact]
        exact invocation.call.verifierAccepted⟩ }

def verified (invocation : Invocation artifact selected) :
    FullClaim.Verified
      (protocolSchema widths (PackedProof selected)) Digest.Value
      (Challenges K) (State K) (VerifyClaim selected) :=
  invocation.receipt.toVerified

def beforeCarry (invocation : Invocation artifact selected) : ConcreteCarry :=
  MemoryCarryParser.semanticCarry invocation.carry.priorValue
    (MemoryCarryParser.parse_value_canonical
      (invocation.carry.priorAccepted invocation.satisfies)).stepIndex

def afterCarry (invocation : Invocation artifact selected) : ConcreteCarry :=
  MemoryCarryParser.semanticCarry invocation.carry.intermediateValue
    (MemoryCarryParser.parse_value_canonical
      (invocation.carry.intermediateAccepted invocation.satisfies)).stepIndex

/-- The active carry produced after the separate nonterminal continuation.
It is not the result of consuming the checked memory claim. -/
def continuedCarry (invocation : Invocation artifact selected) : ConcreteCarry :=
  MemoryCarryParser.semanticCarry invocation.carry.outgoingValue
    (MemoryCarryParser.parse_value_canonical
      (invocation.carry.outgoingAccepted invocation.satisfies)).stepIndex

theorem transition (invocation : Invocation artifact selected) :
    FullClaim.Transition
      (schema := protocolSchema widths (PackedProof selected))
      (Digest := Digest.Value) (Challenge := Challenges K)
      (Products := State K)
      (VerifyClaim selected) ConcreteBalanced
      invocation.beforeCarry invocation.verified invocation.afterCarry where
  consumes := by
    exact invocation.call.consumesExactAcceptedMemoryClaim
      invocation.carry invocation.satisfies

theorem memoryParsed (invocation : Invocation artifact selected) :
    MemoryClaimRows.ParsedColumnsMatch
      artifact.layouts.memorySource.product.claim invocation.assignment
      invocation.call.claim.memory := by
  rw [artifact.layoutsValid.memorySourceUsesMemoryClaim]
  exact invocation.call.memoryClaimColumnsMatch invocation.satisfies

def source (invocation : Invocation artifact selected) :
    MemorySourceRows.Sound artifact.layouts.memorySource invocation.assignment
      invocation.call.claim.memory :=
  MemorySourceRows.sound invocation.call.canonicalAssignment
    invocation.call.one
    invocation.memoryParsed
    (artifact.memorySource_satisfied invocation.satisfies)

def chunk (invocation : Invocation artifact selected) : ProductState.Chunk :=
  invocation.source.records.chunk

/-- The exact ordered application-access list decoded from the same 63
physical operation slots that feed the product chains. -/
def applicationAccesses (invocation : Invocation artifact selected) :
    List Access :=
  ApplicationPortRefinement.accesses invocation.source.operation

/-- The three normalized application rows decoded from the operation slots.
The application-control relation supplies only their row kinds. -/
def applicationRows
    (invocation : Invocation artifact selected)
    (kinds : Ports.ApplicationRowIndex → Ports.NormalizedRowKind) :
    List Ports.NormalizedRow :=
  ApplicationPortRefinement.rows invocation.source.operation kinds

@[simp]
theorem applicationRows_length
    (invocation : Invocation artifact selected)
    (kinds : Ports.ApplicationRowIndex → Ports.NormalizedRowKind) :
    (invocation.applicationRows kinds).length =
      Ports.applicationRowsPerStep := by
  exact ApplicationPortRefinement.rows_length
    invocation.source.operation kinds

/-- Flattening the three application rows preserves the exact physical-slot
order, including active slots after inactive holes. -/
theorem applicationRows_flatMap_accesses
    (invocation : Invocation artifact selected)
    (kinds : Ports.ApplicationRowIndex → Ports.NormalizedRowKind) :
    (invocation.applicationRows kinds).flatMap Ports.NormalizedRow.accesses =
      invocation.applicationAccesses := by
  exact ApplicationPortRefinement.rows_flatMap_accesses
    invocation.source.operation kinds

/-- The fixed-port rows derive the claim's active-access count. -/
theorem applicationAccesses_length
    (invocation : Invocation artifact selected) :
    invocation.applicationAccesses.length =
      invocation.call.claim.memory.activeAccessCount := by
  exact ApplicationPortRefinement.accesses_length_eq_claimActiveCount
    invocation.source.operation

/-- The memory transition rows derive the timestamp endpoint used by the
application-port ordering theorem. -/
theorem timestampAdvance (invocation : Invocation artifact selected) :
    invocation.call.claim.memory.timestampOut =
      invocation.call.claim.memory.timestampIn +
        invocation.call.claim.memory.activeAccessCount := by
  exact invocation.transition.consumes.timestampAdvance

/-- The exact application accesses obey the strict global integer timestamp
schedule checked by the operation rows. -/
theorem applicationAccessesOrdered
    (invocation : Invocation artifact selected) :
    Ordered invocation.call.claim.memory.timestampIn
      invocation.applicationAccesses
      invocation.call.claim.memory.timestampOut := by
  exact ApplicationPortRefinement.ordered invocation.source.operation
    invocation.timestampAdvance

/-- The read-source multiset is exactly the read tuple of each active
application port, with physical order forgotten only at this boundary. -/
theorem chunk_reads_eq
    (invocation : Invocation artifact selected) :
    invocation.chunk.reads =
      (Memory.readTuples invocation.applicationAccesses :
        Multiset MemTuple) := by
  simpa [chunk, MemorySourceRows.Sound.records,
    MemoryClaimProductUpdate.CheckedStepRecords.chunk,
    applicationAccesses] using
      ApplicationPortRefinement.readRecordMultiset_eq
        invocation.source.operation

/-- The write-source multiset is exactly the write tuple of each active
application port. -/
theorem chunk_writes_eq
    (invocation : Invocation artifact selected) :
    invocation.chunk.writes =
      (Memory.writeTuples invocation.applicationAccesses :
        Multiset MemTuple) := by
  simpa [chunk, MemorySourceRows.Sound.records,
    MemoryClaimProductUpdate.CheckedStepRecords.chunk,
    applicationAccesses] using
      ApplicationPortRefinement.writeRecordMultiset_eq
        invocation.source.operation

/-- The complete source and product rows derive this update. It is not an
invocation field. -/
theorem productUpdate (invocation : Invocation artifact selected) :
    mapState invocation.call.claim.memory.productsAfter =
      ProductState.update encode
        (mapChallenges invocation.call.claim.memory.challenge)
        (mapState invocation.call.claim.memory.productsBefore)
        invocation.chunk := by
  exact MemorySourceRows.product_update
    invocation.call.canonicalAssignment invocation.call.one
    invocation.memoryParsed
    (artifact.memoryCheckedStep_satisfied invocation.satisfies)
    invocation.source

end Invocation

/-- Exact carry chaining for a list of satisfying recursive invocations. -/
inductive Run
    {widths : CompilerWidths}
    (artifact : Artifact widths) (selected : SelectedVerifier widths) :
    ConcreteCarry → List (Invocation artifact selected) →
      ConcreteCarry → Prop
  | nil (state : ConcreteCarry) : Run artifact selected state [] state
  | cons
      {tail : List (Invocation artifact selected)} {final : ConcreteCarry}
      (head : Invocation artifact selected)
      (rest : Run artifact selected head.afterCarry tail final) :
      Run artifact selected head.beforeCarry (head :: tail) final

namespace Run

variable {widths : CompilerWidths}
variable {artifact : Artifact widths} {selected : SelectedVerifier widths}

def verifiedClaims (invocations : List (Invocation artifact selected)) :
    List
      (FullClaim.Verified
        (protocolSchema widths (PackedProof selected)) Digest.Value
        (Challenges K) (State K) (VerifyClaim selected)) :=
  invocations.map Invocation.verified

def chunks (invocations : List (Invocation artifact selected)) :
    List ProductState.Chunk :=
  invocations.map Invocation.chunk

/-- The exact row-checked run is also an independent full-claim run. -/
theorem toVerifiedRun
    {before after : ConcreteCarry}
    {invocations : List (Invocation artifact selected)}
    (run : Run artifact selected before invocations after) :
    FullClaim.VerifiedRun
      (schema := protocolSchema widths (PackedProof selected))
      (Digest := Digest.Value) (Challenge := Challenges K)
      (Products := State K)
      (VerifyClaim selected) ConcreteBalanced before
      (verifiedClaims invocations) after := by
  induction run with
  | nil => exact .nil _
  | @cons tail final head rest inductionHypothesis =>
      exact .cons head.transition inductionHypothesis

theorem fromClosedIsEmpty
    {closed : ClosedCarry Digest.Value} {after : ConcreteCarry}
    {invocations : List (Invocation artifact selected)}
    (run : Run artifact selected (.closed closed) invocations after) :
    invocations = [] ∧ after = .closed closed := by
  have exact := FullClaim.VerifiedRun.from_closed_is_empty run.toVerifiedRun
  exact ⟨by simpa [verifiedClaims] using exact.1, exact.2⟩

private theorem empty_eq
    {before after : ConcreteCarry}
    (run : Run artifact selected before [] after) :
    before = after := by
  cases run
  rfl

private theorem activeWellFormedOfCons
    {before after : ConcreteCarry}
    {active : ActiveCarry Digest.Value (Challenges K) (State K)}
    {head : Invocation artifact selected}
    {tail : List (Invocation artifact selected)}
    (run : Run artifact selected before (head :: tail) after)
    (beforeEq : before = .active active) :
    active.WellFormed := by
  cases run with
  | @cons _ _ _ _ =>
      have consumption := head.transition.consumes
      rw [beforeEq] at consumption
      exact consumption.activeWellFormed

/-- A row-checked run that starts active and reaches close derives the opening
active carry's complete well-formedness predicate from its first consumed
claim. -/
theorem activeWellFormed
    {active : ActiveCarry Digest.Value (Challenges K) (State K)}
    {closed : ClosedCarry Digest.Value}
    {invocations : List (Invocation artifact selected)}
    (run : Run artifact selected (.active active) invocations
      (.closed closed)) :
    active.WellFormed := by
  have nonempty : invocations ≠ [] := by
    intro empty
    subst invocations
    have impossible :
        (Carry.active active : ConcreteCarry) = .closed closed :=
      empty_eq run
    cases impossible
  obtain ⟨head, tail, rfl⟩ := List.exists_cons_of_ne_nil nonempty
  exact activeWellFormedOfCons run rfl

private theorem consumeActiveCases
    {active : ActiveCarry Digest.Value (Challenges K) (State K)}
    {claim : ClaimSuffix Digest.Value (Challenges K) (State K)}
    {after : ConcreteCarry}
    (consumption :
      FPrime.Consumes ConcreteBalanced (.active active) claim after) :
    (∃ (_ : MatchesActive active claim)
        (notLast : active.stepIndex.val + 1 < Lifecycle.claimsPerSegment),
        after = .active (interiorCarry active claim notLast)) ∨
      (∃ (_ : MatchesActive active claim)
        (_ : active.stepIndex.val + 1 = Lifecycle.claimsPerSegment)
        (_ : CloseChecks ConcreteBalanced active claim),
        after = .closed (closedCarryAfter active claim)) := by
  cases consumption with
  | interior agreement notLast =>
      exact Or.inl ⟨agreement, notLast, rfl⟩
  | close agreement last checks =>
      exact Or.inr ⟨agreement, last, checks, rfl⟩

private theorem accumulatedProductsBalancedAux
    {before after : ConcreteCarry}
    {invocations : List (Invocation artifact selected)}
    (run : Run artifact selected before invocations after)
    (beforeActive : ∃ active, before = .active active)
    (afterClosed : ∃ closed, after = .closed closed) :
    ∀ active, before = .active active →
      ProductState.Balanced
        (ProductState.accumulate encode (mapChallenges active.challenge)
          (mapState active.products) (chunks invocations)) := by
  induction run with
  | nil =>
      intro active beforeEqual
      rcases afterClosed with ⟨closed, afterEqual⟩
      rw [beforeEqual] at afterEqual
      cases afterEqual
  | @cons tail final head rest inductionHypothesis =>
      intro requestedActive beforeEqual
      have update := head.productUpdate
      have consumption : FPrime.Consumes ConcreteBalanced
          (.active requestedActive) head.call.claim.memory
          head.afterCarry := by
        rw [← beforeEqual]
        exact head.transition.consumes
      rcases consumeActiveCases consumption with
        ⟨agreement, notLast, afterExact⟩ |
        ⟨agreement, last, checks, afterExact⟩
      · have tailBalanced := inductionHypothesis
          ⟨_, afterExact⟩ afterClosed _ afterExact
        simpa [chunks, ProductState.accumulate, interiorCarry,
          agreement.challenge, agreement.products, update] using
          tailBalanced
      · have closedRest : Run artifact selected
            (.closed (closedCarryAfter requestedActive
              head.call.claim.memory)) tail final := by
          rw [← afterExact]
          exact rest
        have tailEmpty := closedRest.fromClosedIsEmpty
        have mappedBalanced :=
          (concreteBalanced_iff_mapped _).mp checks.productsBalanced
        change ProductState.Balanced
          (ProductState.accumulate encode
            (mapChallenges requestedActive.challenge)
            (mapState requestedActive.products)
            (head.chunk :: chunks tail))
        rw [ProductState.accumulate]
        have headExact :
            ProductState.update encode
                (mapChallenges requestedActive.challenge)
                (mapState requestedActive.products) head.chunk =
              mapState head.call.claim.memory.productsAfter := by
          rw [← agreement.challenge, ← agreement.products]
          exact update.symm
        rw [headExact, tailEmpty.1]
        simpa [chunks, ProductState.accumulate] using mappedBalanced

/-- All row-derived step products accumulate to a balanced state when the
exact invocation chain reaches the canonical close transition. -/
theorem accumulatedProductsBalanced
    {active : ActiveCarry Digest.Value (Challenges K) (State K)}
    {closed : ClosedCarry Digest.Value}
    {invocations : List (Invocation artifact selected)}
    (run : Run artifact selected (.active active) invocations
      (.closed closed)) :
    ProductState.Balanced
      (ProductState.accumulate encode (mapChallenges active.challenge)
        (mapState active.products) (chunks invocations)) :=
  accumulatedProductsBalancedAux run ⟨active, rfl⟩ ⟨closed, rfl⟩
    active rfl

/-- A canonical segment opening starts the row-derived accumulation at one. -/
theorem accumulatedFromOneBalanced
    {active : ActiveCarry Digest.Value (Challenges K) (State K)}
    {closed : ClosedCarry Digest.Value}
    {invocations : List (Invocation artifact selected)}
    (run : Run artifact selected (.active active) invocations
      (.closed closed))
    (openingProducts :
      active.products = MemoryCarryCodec.oneProductsK) :
    ProductState.Balanced
      (ProductState.accumulate encode (mapChallenges active.challenge)
        ProductState.one (chunks invocations)) := by
  have balanced := run.accumulatedProductsBalanced
  rw [openingProducts, mapState_oneProductsK] at balanced
  exact balanced

/-- A full row-checked segment cannot skip or repeat a checked-step claim. -/
theorem exactClaimCount
    {active : ActiveCarry Digest.Value (Challenges K) (State K)}
    {closed : ClosedCarry Digest.Value}
    {invocations : List (Invocation artifact selected)}
    (run : Run artifact selected (.active active) invocations
      (.closed closed))
    (startsAtZero : active.stepIndex.val = 0) :
    invocations.length = Lifecycle.claimsPerSegment := by
  have exact := run.toVerifiedRun.full_segment_has_exact_claim_count
    startsAtZero
  simpa [verifiedClaims] using exact

end Run

end Nightstream.Implementation.NebulaV2.SegmentCheckedRows
