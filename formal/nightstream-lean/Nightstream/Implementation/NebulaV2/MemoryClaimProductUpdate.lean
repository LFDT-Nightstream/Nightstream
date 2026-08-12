import Nightstream.Implementation.NebulaV2.MemoryProductClaimBridge
import Nightstream.Protocol.NebulaV2.IdealAcceptance

/-!
Contract: derive one complete independent fingerprint-state update from the
eight fixed Nebula V2 product chains.

Assurance tier: implementation-to-protocol bridge.

Owns the exact map from concrete SuperNeo extension-field values to the
mathematical challenge field, the typed record chunk for one checked step, and
the proof that all four products in both repetitions implement
`ProductState.update`.

The source premise contains only per-slot record meanings. It contains no
product endpoint equality and no instance of `ClaimProductUpdate`.

Does not own the rows that derive the source premise from operation and
snapshot lane bits, the full-claim NIFS receipt, honest frame allocation, or
the generated V2 artifact.

Emits constraints: no. It gives aggregate meaning to existing constraints.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.NebulaV2.MemoryClaimProductUpdate

open Nightstream.Implementation.NebulaV2.ConcreteField
open Nightstream.Implementation.NebulaV2.MemoryClaimCodec
open Nightstream.Implementation.NebulaV2.MemoryProductClaimBridge
open Nightstream.Implementation.NebulaV2.MemoryProductSemanticBridge
open Nightstream.Implementation.NebulaV2.MemoryProductUpdateRows
open Nightstream.Implementation.R1CS
open Nightstream.Protocol.NebulaV2
open Nightstream.Protocol.NebulaV2.ConcreteLaneGeometry
open Nightstream.Protocol.NebulaV2.Fingerprint
open Nightstream.Protocol.NebulaV2.ProductState
open Nightstream.SuperNeo.Concrete

/-- The exact typed records represented by one checked-step lane slice. The
two operation lists retain physical holes. Snapshot lists are exact-cover and
therefore have no holes. -/
structure CheckedStepRecords where
  reads : Fin operationSlots → Option BoundedTuple
  writes : Fin operationSlots → Option BoundedTuple
  initialSnapshot : Fin scanSlots → BoundedTuple
  finalSnapshot : Fin scanSlots → BoundedTuple

def CheckedStepRecords.operation (records : CheckedStepRecords) :
    OperationRole → Fin operationSlots → Option BoundedTuple
  | .reads => records.reads
  | .writes => records.writes

def CheckedStepRecords.snapshot (records : CheckedStepRecords) :
    SnapshotRole → Fin scanSlots → BoundedTuple
  | .initialSnapshot => records.initialSnapshot
  | .finalSnapshot => records.finalSnapshot

/-- Exact independent protocol chunk obtained after physical-hole removal.
`activeRecordMultiset` preserves multiplicity. -/
def CheckedStepRecords.chunk (records : CheckedStepRecords) :
    ProductState.Chunk :=
  { initialSnapshot := activeRecordMultiset
      (snapshotRecords records.initialSnapshot)
    writes := activeRecordMultiset (operationRecords records.writes)
    reads := activeRecordMultiset (operationRecords records.reads)
    finalSnapshot := activeRecordMultiset
      (snapshotRecords records.finalSnapshot) }

/-- Source-only refinement for all eight product chains. This proposition has
no running-product or endpoint field. -/
structure SourceRefines
    (assignment : Nat → Nat) (layout : Layout)
    (records : CheckedStepRecords) : Prop where
  operation : ∀ repetition role,
    List.Forall₂ (GateRepresents assignment)
      (layout.operationChain repetition role).entries
      (operationRecords (records.operation role))
  snapshot : ∀ repetition role,
    List.Forall₂ (GateRepresents assignment)
      (layout.snapshotChain repetition role).entries
      (snapshotRecords (records.snapshot role))

def mapFour (products : Four K) : Four ChallengeField :=
  { initialSnapshot := superNeoEquiv products.initialSnapshot
    writes := superNeoEquiv products.writes
    reads := superNeoEquiv products.reads
    finalSnapshot := superNeoEquiv products.finalSnapshot }

def mapState (products : State K) : State ChallengeField :=
  fun repetition => mapFour (products repetition)

def mapChallenges (challenges : Challenges K) :
    Challenges ChallengeField :=
  fun repetition => fieldChallenge (challenges repetition)

private theorem operation_component
    {layout : Layout} {assignment : Nat → Nat} {claim : Claim}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (parsed : MemoryClaimRows.ParsedColumnsMatch layout.claim assignment claim)
    (holds : Satisfies (rows layout) assignment)
    (records : CheckedStepRecords)
    (source : SourceRefines assignment layout records)
    (repetition : Fin 2) (role : OperationRole) :
    superNeoEquiv
        (productK claim 1 repetition (productRole role)) =
      superNeoEquiv
          (productK claim 0 repetition (productRole role)) *
        ProductState.recordsProduct
          Nightstream.Implementation.NebulaV2.ConcreteField.encode
          (fieldChallenge (claim.challenge repetition))
          (activeRecordMultiset
            (operationRecords (records.operation role))) := by
  have exactUpdate := operation_claim_product_update canonical one parsed holds
    repetition role (records.operation role) (source.operation repetition role)
  have mapped := congrArg superNeoEquiv exactUpdate
  rw [superNeoEquiv_foldOptionsK] at mapped
  exact mapped

private theorem snapshot_component
    {layout : Layout} {assignment : Nat → Nat} {claim : Claim}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (parsed : MemoryClaimRows.ParsedColumnsMatch layout.claim assignment claim)
    (holds : Satisfies (rows layout) assignment)
    (records : CheckedStepRecords)
    (source : SourceRefines assignment layout records)
    (repetition : Fin 2) (role : SnapshotRole) :
    superNeoEquiv
        (productK claim 1 repetition (snapshotProductRole role)) =
      superNeoEquiv
          (productK claim 0 repetition (snapshotProductRole role)) *
        ProductState.recordsProduct
          Nightstream.Implementation.NebulaV2.ConcreteField.encode
          (fieldChallenge (claim.challenge repetition))
          (activeRecordMultiset
            (snapshotRecords (records.snapshot role))) := by
  have exactUpdate := snapshot_claim_product_update canonical one parsed holds
    repetition role (records.snapshot role) (source.snapshot repetition role)
  have mapped := congrArg superNeoEquiv exactUpdate
  rw [superNeoEquiv_foldOptionsK] at mapped
  exact mapped

/-- The satisfying eight-chain row program derives the complete independent
`ProductState.update`. No premise states any product endpoint equality. -/
theorem claim_product_update
    {layout : Layout} {assignment : Nat → Nat} {claim : Claim}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (parsed : MemoryClaimRows.ParsedColumnsMatch layout.claim assignment claim)
    (holds : Satisfies (rows layout) assignment)
    (records : CheckedStepRecords)
    (source : SourceRefines assignment layout records) :
    mapState claim.productsAfter =
      ProductState.update
        Nightstream.Implementation.NebulaV2.ConcreteField.encode
        (mapChallenges claim.challenge)
        (mapState claim.productsBefore) records.chunk := by
  funext repetition
  apply Four.ext
  · simpa [mapState, mapFour, ProductState.update,
      CheckedStepRecords.chunk, mapChallenges] using
      snapshot_component canonical one parsed holds records source repetition
        SnapshotRole.initialSnapshot
  · simpa [mapState, mapFour, ProductState.update,
      CheckedStepRecords.chunk, mapChallenges] using
      operation_component canonical one parsed holds records source repetition
        OperationRole.writes
  · simpa [mapState, mapFour, ProductState.update,
      CheckedStepRecords.chunk, mapChallenges] using
      operation_component canonical one parsed holds records source repetition
        OperationRole.reads
  · simpa [mapState, mapFour, ProductState.update,
      CheckedStepRecords.chunk, mapChallenges] using
      snapshot_component canonical one parsed holds records source repetition
        SnapshotRole.finalSnapshot

end Nightstream.Implementation.NebulaV2.MemoryClaimProductUpdate
