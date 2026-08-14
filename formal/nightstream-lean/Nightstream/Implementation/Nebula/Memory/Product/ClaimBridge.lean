import Nightstream.Implementation.Nebula.Memory.Product.SemanticBridge
import Nightstream.Implementation.Nebula.Memory.Product.UpdateRows

/-!
Contract: bind the fixed eight product chains to the exact parsed memory claim.

Assurance tier: implementation-to-claim bridge.

Owns coefficient-level placement of both challenge pairs and all products from
the fail-closed claim parser, and derives the exact claim product update for
one operation or snapshot chain from row satisfaction plus source-record
refinement.

Does not own the source-record refinement itself, the aggregate four-product
`ProductState.Chunk` theorem, honest frame allocation, or the generated V2
artifact.

Emits constraints: no. It composes existing emitted constraints.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.Nebula.MemoryProductClaimBridge

open Nightstream.Implementation.Nebula.MemoryClaimCodec
open Nightstream.Implementation.Nebula.MemoryProductSemanticBridge
open Nightstream.Implementation.Nebula.MemoryProductUpdateRows
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical.KConcreteBridge
open Nightstream.Implementation.R1CS.Canonical.KHorner
open Nightstream.Implementation.R1CS.Canonical.KMul
open Nightstream.Protocol.Nebula.ConcreteLaneGeometry
open Nightstream.Protocol.Nebula.Fingerprint
open Nightstream.Protocol.Nebula.IdealFingerprint
open Nightstream.SuperNeo.Concrete

private theorem singleton_eval
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (column : Nat) :
    lcEval assignment [(column, 1)] = assignment column := by
  rw [lcEval_singleton_col, Nat.mod_eq_of_lt (canonical column)]

/-- Parsed challenge coordinates are exactly the carried row values. -/
theorem challenge_carried_eq
    {layout : Layout} {assignment : Nat → Nat} {claim : Claim}
    (canonical : ∀ column, assignment column < goldilocksP)
    (parsed : MemoryClaimRows.ParsedColumnsMatch layout.claim assignment claim)
    (repetition coordinate : Fin 2) :
    carriedValue assignment (layout.challenge repetition coordinate) =
      ofConcrete
        (if coordinate = 0 then (claim.challenge repetition).gamma1
         else (claim.challenge repetition).gamma2) := by
  fin_cases coordinate
  · change
      Pair.mk (lcEval assignment
          [(layout.claimFieldColumn (.challenge repetition 0 0), 1)])
        (lcEval assignment
          [(layout.claimFieldColumn (.challenge repetition 0 1), 1)]) = _
    rw [singleton_eval canonical, singleton_eval canonical]
    have low := parsed.fields (.challenge repetition 0 0)
    have high := parsed.fields (.challenge repetition 0 1)
    change assignment
      (layout.claimFieldColumn (.challenge repetition 0 0)) = _ at low
    change assignment
      (layout.claimFieldColumn (.challenge repetition 0 1)) = _ at high
    rw [low, high]
    simp [MemoryClaimFieldRows.Slot.tag,
      MemoryClaimCodec.Claim.fieldValue, MemoryClaimCodec.challengeValue,
      MemoryClaimCodec.kLimbValue, ofConcrete]
  · change
      Pair.mk (lcEval assignment
          [(layout.claimFieldColumn (.challenge repetition 1 0), 1)])
        (lcEval assignment
          [(layout.claimFieldColumn (.challenge repetition 1 1), 1)]) = _
    rw [singleton_eval canonical, singleton_eval canonical]
    have low := parsed.fields (.challenge repetition 1 0)
    have high := parsed.fields (.challenge repetition 1 1)
    change assignment
      (layout.claimFieldColumn (.challenge repetition 1 0)) = _ at low
    change assignment
      (layout.claimFieldColumn (.challenge repetition 1 1)) = _ at high
    rw [low, high]
    simp [MemoryClaimFieldRows.Slot.tag,
      MemoryClaimCodec.Claim.fieldValue, MemoryClaimCodec.challengeValue,
      MemoryClaimCodec.kLimbValue, ofConcrete]

def productK (claim : Claim) (side repetition : Fin 2)
    (role : MemoryClaimCodec.ProductRole) : K :=
  let products :=
    if side = 0 then claim.productsBefore repetition
    else claim.productsAfter repetition
  match role with
  | .initialSnapshot => products.initialSnapshot
  | .writes => products.writes
  | .reads => products.reads
  | .finalSnapshot => products.finalSnapshot

/-- Parsed product coordinates are exactly the carried row values. -/
theorem product_carried_eq
    {layout : Layout} {assignment : Nat → Nat} {claim : Claim}
    (canonical : ∀ column, assignment column < goldilocksP)
    (parsed : MemoryClaimRows.ParsedColumnsMatch layout.claim assignment claim)
    (side repetition : Fin 2) (role : MemoryClaimCodec.ProductRole) :
    carriedValue assignment (layout.product side repetition role) =
      ofConcrete (productK claim side repetition role) := by
  change
    Pair.mk (lcEval assignment
        [(layout.claimFieldColumn (.product side repetition role 0), 1)])
      (lcEval assignment
        [(layout.claimFieldColumn (.product side repetition role 1), 1)]) = _
  rw [singleton_eval canonical, singleton_eval canonical]
  have low := parsed.fields (.product side repetition role 0)
  have high := parsed.fields (.product side repetition role 1)
  change assignment
    (layout.claimFieldColumn (.product side repetition role 0)) = _ at low
  change assignment
    (layout.claimFieldColumn (.product side repetition role 1)) = _ at high
  rw [low, high]
  fin_cases side <;> cases role <;> rfl

def operationRecords
    (ports : Fin operationSlots → Option BoundedTuple) :
    List (Option BoundedTuple) :=
  List.ofFn fun slot => ports slot

def snapshotRecords
    (records : Fin scanSlots → BoundedTuple) :
    List (Option BoundedTuple) :=
  List.ofFn fun slot => some (records slot)

/-- One operation chain updates the exact matching parsed claim product. The
slot relation contains only source-record facts and cannot manufacture the
endpoint equality. -/
theorem operation_claim_product_update
    {layout : Layout} {assignment : Nat → Nat} {claim : Claim}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (parsed : MemoryClaimRows.ParsedColumnsMatch layout.claim assignment claim)
    (holds : Satisfies (rows layout) assignment)
    (repetition : Fin 2) (role : OperationRole)
    (ports : Fin operationSlots → Option BoundedTuple)
    (represents : List.Forall₂ (GateRepresents assignment)
      (layout.operationChain repetition role).entries
      (operationRecords ports)) :
    productK claim 1 repetition (productRole role) =
      foldOptionsK (claim.challenge repetition)
        (productK claim 0 repetition (productRole role))
        (operationRecords ports) := by
  apply ofConcrete_injective
  have chainHolds : Satisfies
      (MemoryProductChainRows.rows
        (layout.operationChain repetition role)) assignment := by
    intro row member
    exact holds row (by
      apply List.mem_flatMap.mpr
      exact ⟨layout.operationChain repetition role, by
        simp [chains, operationChains]
        fin_cases repetition <;> cases role <;> simp [operationRoles], member⟩)
  have derived := final_eq_foldOptionsK one chainHolds
    (challenge_carried_eq canonical parsed repetition 0)
    (challenge_carried_eq canonical parsed repetition 1)
    (product_carried_eq canonical parsed 0 repetition (productRole role))
    represents
  have derived' :
      carriedValue assignment
          (layout.product 1 repetition (productRole role)) =
        ofConcrete
          (foldOptionsK (claim.challenge repetition)
            (productK claim 0 repetition (productRole role))
            (operationRecords ports)) := by
    simpa [Layout.operationChain] using derived
  rw [product_carried_eq canonical parsed 1 repetition (productRole role)]
    at derived'
  exact derived'

/-- One snapshot chain updates the exact matching parsed claim product. -/
theorem snapshot_claim_product_update
    {layout : Layout} {assignment : Nat → Nat} {claim : Claim}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (parsed : MemoryClaimRows.ParsedColumnsMatch layout.claim assignment claim)
    (holds : Satisfies (rows layout) assignment)
    (repetition : Fin 2) (role : SnapshotRole)
    (records : Fin scanSlots → BoundedTuple)
    (represents : List.Forall₂ (GateRepresents assignment)
      (layout.snapshotChain repetition role).entries
      (snapshotRecords records)) :
    productK claim 1 repetition (snapshotProductRole role) =
      foldOptionsK (claim.challenge repetition)
        (productK claim 0 repetition (snapshotProductRole role))
        (snapshotRecords records) := by
  apply ofConcrete_injective
  have chainHolds : Satisfies
      (MemoryProductChainRows.rows
        (layout.snapshotChain repetition role)) assignment := by
    intro row member
    exact holds row (by
      apply List.mem_flatMap.mpr
      exact ⟨layout.snapshotChain repetition role, by
        simp [chains, snapshotChains]
        fin_cases repetition <;> cases role <;> simp [snapshotRoles], member⟩)
  have derived := final_eq_foldOptionsK one chainHolds
    (challenge_carried_eq canonical parsed repetition 0)
    (challenge_carried_eq canonical parsed repetition 1)
    (product_carried_eq canonical parsed 0 repetition
      (snapshotProductRole role)) represents
  have derived' :
      carriedValue assignment
          (layout.product 1 repetition (snapshotProductRole role)) =
        ofConcrete
          (foldOptionsK (claim.challenge repetition)
            (productK claim 0 repetition (snapshotProductRole role))
            (snapshotRecords records)) := by
    simpa [Layout.snapshotChain] using derived
  rw [product_carried_eq canonical parsed 1 repetition
    (snapshotProductRole role)] at derived'
  exact derived'

end Nightstream.Implementation.Nebula.MemoryProductClaimBridge
