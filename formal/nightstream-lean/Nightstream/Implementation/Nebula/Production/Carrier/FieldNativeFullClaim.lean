import Nightstream.Implementation.Nebula.NIFS.Running.Codec
import Nightstream.Implementation.Nebula.Production.Memory.BoundCcsPublic
import Nightstream.Implementation.Nebula.Production.Memory.SuffixCarrier

/-!
Contract: exact typed full claim for one field-native production candidate.

The claim carries the 540-coordinate CCS public input, one mandatory
four-component commitment bundle, the complete fourteen-running paper-NIFS
state, and one ordered `E`-suffix memory batch. Its mixed authority image has
no bit-serial running, bundle, challenge, product, or root bridge.

The candidate type fixes the profile. The external WASM statement is owned by
the verifier and terminal public-input parser. It is not repeated as a fresh
claim sidecar. This matches the exact paper-NIFS authority boundary: NIFS sees
the running state, bundle, and CCS public input. The CCS input binds the
complete memory batch and prior HyperNova state digest.

Does not own generated columns, state-hash rows, NIFS verifier rows, terminal
verification, external container bytes, cryptographic assumptions, candidate
selection, or a verifier key.

Emits constraints: no.
-/

set_option autoImplicit false
set_option maxRecDepth 10000

namespace Nightstream.Implementation.Nebula.ProductionFieldNativeFullClaim

open Nightstream.Implementation.Nebula
open Nightstream.Implementation.Nebula.ProductionMemorySuffixCarrier
open Nightstream.Protocol.Nebula
open Nightstream.Protocol.Nebula.ProductionBatchedFPrime
open Nightstream.Protocol.Nebula.ProductionProfileCandidates
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation

abbrev CcsPublic := FixedBits.Word MemoryBoundCcsPublic.coordinateCount
abbrev Bundle := CommitmentBundleCodec.Value
abbrev Running (fullShape : Phi81Relation.Shape) :=
  ProductNifsCodec.RunningFor fullShape.rowVariables fullShape
abbrev Fresh (fullShape : Phi81Relation.Shape) :=
  ProductNifsCodec.FreshFor fullShape.rowVariables fullShape
abbrev Batch := ProductionMemoryBatchPoseidonBinding.Batch

@[ext]
structure Value (candidate : Id) (fullShape : Phi81Relation.Shape) where
  ccsPublic : CcsPublic
  commitmentBundle : Bundle
  recursiveState : Running fullShape
  memory : Batch candidate

/-- Canonicality not already carried by field types. Candidate identity is a
type index, and external statement ownership is a separate verifier boundary. -/
structure Value.Canonical
    {candidate : Id} {fullShape : Phi81Relation.Shape}
    (value : Value candidate fullShape) : Prop where
  memoryCanonical : forall claim, claim ∈ value.memory.suffixes ->
    MemoryClaimCodec.Claim.Canonical claim

/-- Separate authority relation between the CCS public carrier and the
complete ordered memory batch. This is not claim canonicality. Production
relation rows must derive it. -/
def Value.MemoryBound
    {candidate : Id} {fullShape : Phi81Relation.Shape}
    (value : Value candidate fullShape) : Prop :=
  ProductionMemoryBoundCcsPublic.MemoryMatches value.ccsPublic value.memory

instance valueMemoryBoundDecidable
    {candidate : Id} {fullShape : Phi81Relation.Shape}
    (value : Value candidate fullShape) : Decidable value.MemoryBound := by
  unfold Value.MemoryBound
  infer_instance

/-! ## Exact field-native authority image -/

noncomputable def bundleFields (bundle : Bundle) : List F :=
  ProductNifsCodec.bundleCodec.encode (ProductNifsCodec.codecBundle bundle)

noncomputable def runningFields
    {fullShape : Phi81Relation.Shape} (running : Running fullShape) : List F :=
  (ProductNifsCodec.runningCodecFor fullShape.rowVariables fullShape).encode
    running

theorem runningFields_lengthFor
    {fullShape : Phi81Relation.Shape}
    (contract : ProductNifsCodec.FullShapeContractFor
      fullShape.rowVariables fullShape)
    (running : Running fullShape) :
    (runningFields running).length =
      ProductNifsCodec.runningFieldCountFor fullShape.rowVariables := by
  rw [runningFields,
    (ProductNifsCodec.runningCodecFor fullShape.rowVariables
      fullShape).encode_length,
    ProductNifsCodec.runningCodecFor_width contract]

/-- Canonical natural-number representatives of the field-native running
state. Physical R1CS assignments use these values. -/
noncomputable def runningNativeValues
    {fullShape : Phi81Relation.Shape} (running : Running fullShape) : List Nat :=
  (runningFields running).map Fin.val

theorem bundleFields_length (bundle : Bundle) :
    (bundleFields bundle).length = 3888 := by
  rw [bundleFields, ProductNifsCodec.bundleCodec.encode_length,
    ProductNifsCodec.bundleCodec_width]

theorem runningFields_length
    {fullShape : Phi81Relation.Shape}
    (contract : ProductNifsCodec.FullShapeContract fullShape)
    (running : Running fullShape) :
    (runningFields running).length = 83210 := by
  rw [runningFields_lengthFor contract.toSelected, contract.rowVariables]
  exact ProductNifsCodec.runningFieldCountFor_25

theorem runningNativeValues_length
    {fullShape : Phi81Relation.Shape}
    (contract : ProductNifsCodec.FullShapeContract fullShape)
    (running : Running fullShape) :
    (runningNativeValues running).length = 83210 := by
  rw [runningNativeValues, List.length_map,
    runningFields_length contract]

theorem runningNativeValues_injective
    {fullShape : Phi81Relation.Shape} :
    Function.Injective (runningNativeValues (fullShape := fullShape)) := by
  intro left right equal
  apply
    (ProductNifsCodec.runningCodecFor fullShape.rowVariables
      fullShape).encode_injective_of_admissible
      (ProductNifsCodec.runningCodecFor_admissible left)
      (ProductNifsCodec.runningCodecFor_admissible right)
  exact (List.map_injective_iff.mpr
    (fun _ _ selected => Fin.ext selected)) equal

structure AuthorityImage (candidate : Id) where
  ccsPublicBits : List Nat
  bundleNativeFields : List F
  runningNativeFields : List F
  memoryCounterBits : List Nat
  memoryNativeFields : List F

noncomputable def authorityImage
    {candidate : Id} {fullShape : Phi81Relation.Shape}
    (value : Value candidate fullShape) : AuthorityImage candidate where
  ccsPublicBits := value.ccsPublic.val
  bundleNativeFields := bundleFields value.commitmentBundle
  runningNativeFields := runningFields value.recursiveState
  memoryCounterBits := batchCounterBits value.memory
  memoryNativeFields := batchNativeFields value.memory

def AuthorityImage.coordinateCount
    {candidate : Id} (image : AuthorityImage candidate) : Nat :=
  image.ccsPublicBits.length + image.bundleNativeFields.length +
    image.runningNativeFields.length + image.memoryCounterBits.length +
    image.memoryNativeFields.length

theorem authorityImage_coordinate_count
    {candidate : Id} {fullShape : Phi81Relation.Shape}
    (contract : ProductNifsCodec.FullShapeContract fullShape)
    (value : Value candidate fullShape) :
    (authorityImage value).coordinateCount =
      fieldNativeEnvelopeCoordinates candidate := by
  rw [AuthorityImage.coordinateCount, authorityImage,
    value.ccsPublic.property.1,
    bundleFields_length, runningFields_length contract,
    batchCounterBits_length, batchNativeFields_length]
  cases candidate <;> decide

/-- Equality of the complete mixed carrier recovers the complete typed claim.
No cryptographic assumption is used because the batch is present, not only
its digest. -/
theorem authorityImage_injective_on_canonical
    {candidate : Id}
    {fullShape : Phi81Relation.Shape}
    {left right : Value candidate fullShape}
    (leftCanonical : left.Canonical)
    (rightCanonical : right.Canonical)
    (equal : authorityImage left = authorityImage right) : left = right := by
  apply Value.ext
  · apply Subtype.ext
    exact congrArg AuthorityImage.ccsPublicBits equal
  · apply ProductNifsCodec.codecBundle_injective
    apply ProductNifsCodec.bundleCodec.encode_injective_of_admissible
      (ProductNifsCodec.bundleCodec_admissible
        (ProductNifsCodec.codecBundle left.commitmentBundle))
      (ProductNifsCodec.bundleCodec_admissible
        (ProductNifsCodec.codecBundle right.commitmentBundle))
    exact congrArg AuthorityImage.bundleNativeFields equal
  · apply
      (ProductNifsCodec.runningCodecFor fullShape.rowVariables
        fullShape).encode_injective_of_admissible
        (ProductNifsCodec.runningCodecFor_admissible left.recursiveState)
        (ProductNifsCodec.runningCodecFor_admissible right.recursiveState)
    exact congrArg AuthorityImage.runningNativeFields equal
  · apply batchImage_injective_on_canonical
      leftCanonical.memoryCanonical rightCanonical.memoryCanonical
    apply Prod.ext
    · exact congrArg AuthorityImage.memoryCounterBits equal
    · exact congrArg AuthorityImage.memoryNativeFields equal

/-! ## Exact paper-NIFS projection -/

def freshOfValue
    {candidate : Id} {fullShape : Phi81Relation.Shape}
    (contract : ProductNifsCodec.FullShapeContractFor
      fullShape.rowVariables fullShape)
    (value : Value candidate fullShape) : Fresh fullShape :=
  ProductNifsCodec.freshOfFor contract value.commitmentBundle
    value.ccsPublic

structure NifsInput (fullShape : Phi81Relation.Shape) where
  running : Running fullShape
  fresh : Fresh fullShape

def nifsInput
    {candidate : Id} {fullShape : Phi81Relation.Shape}
    (contract : ProductNifsCodec.FullShapeContractFor
      fullShape.rowVariables fullShape)
    (value : Value candidate fullShape) : NifsInput fullShape where
  running := value.recursiveState
  fresh := freshOfValue contract value

@[simp] theorem nifsInput_running
    {candidate : Id} {fullShape : Phi81Relation.Shape}
    (contract : ProductNifsCodec.FullShapeContractFor
      fullShape.rowVariables fullShape)
    (value : Value candidate fullShape) :
    (nifsInput contract value).running = value.recursiveState := rfl

@[simp] theorem nifsInput_fresh
    {candidate : Id} {fullShape : Phi81Relation.Shape}
    (contract : ProductNifsCodec.FullShapeContractFor
      fullShape.rowVariables fullShape)
    (value : Value candidate fullShape) :
    (nifsInput contract value).fresh = freshOfValue contract value := rfl

/-- Equal NIFS inputs recover the direct running state, bundle, CCS public
input, and the complete canonical memory batch, except for the one named
batch-digest collision event. There is no unauthenticated sidecar in `Value`. -/
theorem nifsInput_eq_recovers_direct_authority_or_collision
    {candidate : Id}
    {fullShape : Phi81Relation.Shape}
    (contract : ProductNifsCodec.FullShapeContractFor
      fullShape.rowVariables fullShape)
    {left right : Value candidate fullShape}
    (leftCanonical : left.Canonical)
    (rightCanonical : right.Canonical)
    (leftMemoryBound : left.MemoryBound)
    (rightMemoryBound : right.MemoryBound)
    (equal : nifsInput contract left = nifsInput contract right) :
    (left.recursiveState = right.recursiveState /\
      left.commitmentBundle = right.commitmentBundle /\
      left.ccsPublic = right.ccsPublic /\
      left.memory = right.memory) \/
      ProductionMemoryBatchPoseidonBinding.PoseidonCollision candidate := by
  have runningEqual : left.recursiveState = right.recursiveState :=
    congrArg NifsInput.running equal
  have freshEqual : freshOfValue contract left = freshOfValue contract right :=
    congrArg NifsInput.fresh equal
  have pairEqual :
      (left.commitmentBundle, left.ccsPublic) =
        (right.commitmentBundle, right.ccsPublic) := by
    apply ProductNifsCodec.freshOfFor_pair_injective contract
    exact freshEqual
  have bundleEqual : left.commitmentBundle = right.commitmentBundle := by
    simpa using congrArg Prod.fst pairEqual
  have ccsEqual : left.ccsPublic = right.ccsPublic := by
    simpa using congrArg Prod.snd pairEqual
  have rightMatchesLeft :
      ProductionMemoryBoundCcsPublic.MemoryMatches
        left.ccsPublic right.memory := by
    rw [ccsEqual]
    exact rightMemoryBound
  have batchResult :=
    ProductionMemoryBoundCcsPublic.matched_batch_eq_or_collision
      leftCanonical.memoryCanonical rightCanonical.memoryCanonical
      leftMemoryBound rightMatchesLeft
  rcases batchResult with memoryEqual | collision
  · exact Or.inl ⟨runningEqual, bundleEqual, ccsEqual, memoryEqual⟩
  · exact Or.inr collision

/-! ## Candidate-specific protocol claim -/

def protocolSchema
    (fullShape : Phi81Relation.Shape) (NifsProof : Type) :
    ProductionBatchedFPrime.Schema where
  CcsPublic := CcsPublic
  CommitmentBundle := Bundle
  RecursiveState := Running fullShape
  NifsProof := NifsProof

def Value.toProtocolClaim
    {candidate : Id} {fullShape : Phi81Relation.Shape} {NifsProof : Type}
    (value : Value candidate fullShape) :
    ProductionBatchedFPrime.Claim candidate
      (protocolSchema fullShape NifsProof) Digest.Value
      (ProductState.Challenges K) (ProductState.State K) where
  ccsPublic := value.ccsPublic
  commitmentBundle := value.commitmentBundle
  recursiveState := value.recursiveState
  memory := value.memory

theorem Value.toProtocolClaim_injective
    {candidate : Id} {fullShape : Phi81Relation.Shape} {NifsProof : Type} :
    Function.Injective
      (Value.toProtocolClaim
        (candidate := candidate) (fullShape := fullShape)
      (NifsProof := NifsProof)) := by
  intro left right equal
  apply Value.ext
  · exact congrArg ProductionBatchedFPrime.Claim.ccsPublic equal
  · exact congrArg ProductionBatchedFPrime.Claim.commitmentBundle equal
  · exact congrArg ProductionBatchedFPrime.Claim.recursiveState equal
  · exact congrArg ProductionBatchedFPrime.Claim.memory equal

end Nightstream.Implementation.Nebula.ProductionFieldNativeFullClaim
