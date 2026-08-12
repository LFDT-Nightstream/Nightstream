import Nightstream.Implementation.NebulaV2.CommitmentBundleCodec
import Nightstream.Implementation.NebulaV2.FixedBits
import Nightstream.Implementation.NebulaV2.MemoryClaimCodec
import Nightstream.Implementation.NebulaV2.TaggedBitSlices
import Nightstream.Implementation.NebulaV2.WasmPublicStatementCodec
import Nightstream.Protocol.NebulaV2.FullClaim

/-!
Contract: exact authority-bearing bit envelope for one V2 fresh claim.

Assurance tier: implementation model.

Owns the full-claim section order, the fixed V2 profile, memory, and bundle
widths, explicit verifier-key-selected widths for compiler-dependent sections,
exact cover, exact slicing, and injectivity on canonical claims.

Does not own the final compiler-selected widths, absolute generated columns,
the NIFS verifier rows, recursive-size closure, or cryptographic extraction.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.NebulaV2.FullClaimEnvelope

open Nightstream.Implementation.NebulaV2
open Nightstream.Protocol.NebulaV2
open Nightstream.Protocol.NebulaV2.ProductState
open Nightstream.Protocol.NebulaV2.WasmPublicStatementEncoding
open Nightstream.SuperNeo.Concrete

/-- The final compiler and verifier key select these three widths. Keeping them
explicit prevents an old generated relation from being treated as V2. -/
structure CompilerWidths where
  ccsPublicBits : Nat
  applicationPublicBits : Nat
  recursiveStateBits : Nat
  ccsPublicPositive : 0 < ccsPublicBits
  applicationPublicPositive : 0 < applicationPublicBits
  recursiveStatePositive : 0 < recursiveStateBits
deriving Repr

/-- Full-claim sections in their authority-bearing order. -/
inductive Section where
  | profile
  | ccsPublic
  | applicationPublic
  | commitmentBundle
  | recursiveState
  | memory
deriving DecidableEq, Fintype, Repr

def sectionOrder : List Section :=
  [.profile, .ccsPublic, .applicationPublic, .commitmentBundle,
    .recursiveState, .memory]

theorem sectionOrder_nodup : sectionOrder.Nodup := by decide

theorem Section.mem_order (part : Section) : part ∈ sectionOrder := by
  cases part <;> simp [sectionOrder]

def Section.width (widths : CompilerWidths) : Section → Nat
  | .profile => profileSerializedBitCount
  | .ccsPublic => widths.ccsPublicBits
  | .applicationPublic => widths.applicationPublicBits
  | .commitmentBundle => MemoryWireGeometry.mandatoryBundleBits
  | .recursiveState => widths.recursiveStateBits
  | .memory => MemoryWireGeometry.stepPublicBits

def CompilerWidths.totalBits (widths : CompilerWidths) : Nat :=
  profileSerializedBitCount + widths.ccsPublicBits +
    widths.applicationPublicBits + MemoryWireGeometry.mandatoryBundleBits +
    widths.recursiveStateBits + MemoryWireGeometry.stepPublicBits

/-- One complete typed fresh claim. The generic word sections remain explicit
until the V2 compiler fixes their typed codecs and absolute columns. -/
@[ext]
structure Value (widths : CompilerWidths) where
  profile : Profile.Identity
  ccsPublic : FixedBits.Word widths.ccsPublicBits
  applicationPublic : FixedBits.Word widths.applicationPublicBits
  commitmentBundle : CommitmentBundleCodec.Value
  recursiveState : FixedBits.Word widths.recursiveStateBits
  memory : MemoryClaimCodec.Claim

/-- Canonicality not already carried by the field types. -/
structure Value.Canonical {widths : CompilerWidths}
    (value : Value widths) : Prop where
  profileExact : value.profile = Profile.v2
  memoryCanonical : value.memory.Canonical

def Value.sectionBits {widths : CompilerWidths}
    (value : Value widths) : Section → List Nat
  | .profile => WasmPublicStatementCodec.encodeProfile value.profile
  | .ccsPublic => value.ccsPublic.val
  | .applicationPublic => value.applicationPublic.val
  | .commitmentBundle => CommitmentBundleCodec.encode value.commitmentBundle
  | .recursiveState => value.recursiveState.val
  | .memory => MemoryClaimCodec.encode value.memory

theorem Value.sectionBits_length {widths : CompilerWidths}
    (value : Value widths) (part : Section) :
    (value.sectionBits part).length = part.width widths := by
  cases part with
  | profile => exact WasmPublicStatementCodec.encodeProfile_length _
  | ccsPublic => exact value.ccsPublic.property.1
  | applicationPublic => exact value.applicationPublic.property.1
  | commitmentBundle => exact CommitmentBundleCodec.encode_length _
  | recursiveState => exact value.recursiveState.property.1
  | memory => exact MemoryClaimCodec.encode_length _

theorem Value.sectionBits_binary {widths : CompilerWidths}
    (value : Value widths) (part : Section) (digit : Nat)
    (member : digit ∈ value.sectionBits part) : digit < 2 := by
  cases part with
  | profile =>
      exact WasmPublicStatementCodec.encodeProfile_binary _ digit member
  | ccsPublic => exact value.ccsPublic.property.2 digit member
  | applicationPublic => exact value.applicationPublic.property.2 digit member
  | commitmentBundle =>
      exact CommitmentBundleCodec.encode_binary _ digit member
  | recursiveState => exact value.recursiveState.property.2 digit member
  | memory => exact MemoryClaimCodec.encode_binary _ digit member

/-- Exact full-claim bit image. -/
def Value.encode {widths : CompilerWidths} (value : Value widths) : List Nat :=
  TaggedBitSlices.flatten value.sectionBits sectionOrder

theorem section_width_sum (widths : CompilerWidths) :
    (sectionOrder.map (Section.width widths)).sum = widths.totalBits := by
  simp [sectionOrder, Section.width, CompilerWidths.totalBits]
  omega

theorem Value.encode_length {widths : CompilerWidths}
    (value : Value widths) : value.encode.length = widths.totalBits := by
  rw [Value.encode, TaggedBitSlices.flatten, List.length_flatMap]
  simp only [value.sectionBits_length]
  exact section_width_sum widths

theorem Value.encode_binary {widths : CompilerWidths}
    (value : Value widths) (digit : Nat) (member : digit ∈ value.encode) :
    digit < 2 := by
  simp only [Value.encode, TaggedBitSlices.flatten, List.mem_flatMap] at member
  obtain ⟨part, _partMember, digitMember⟩ := member
  exact value.sectionBits_binary part digit digitMember

def Value.block {widths : CompilerWidths}
    (value : Value widths) : FixedBits.Word widths.totalBits :=
  ⟨value.encode, value.encode_length,
    fun digit member => value.encode_binary digit member⟩

def Section.bitOffset (widths : CompilerWidths) (part : Section) : Nat :=
  TaggedBitSlices.offsetAt (Section.width widths) sectionOrder
    (sectionOrder.idxOf part)

theorem Section.slice_fits (widths : CompilerWidths) (part : Section) :
    part.bitOffset widths + part.width widths ≤ widths.totalBits := by
  cases part <;>
    simp [Section.bitOffset, TaggedBitSlices.offsetAt, sectionOrder,
      Section.width, CompilerWidths.totalBits] <;> omega

/-- Every section is recovered from its one exact, non-compacted slice. -/
theorem Value.encode_slice {widths : CompilerWidths}
    (value : Value widths) (part : Section) :
    (value.encode.drop (part.bitOffset widths)).take
        (part.width widths) = value.sectionBits part := by
  have bounded := List.idxOf_lt_length_of_mem part.mem_order
  have sliced := TaggedBitSlices.slice_flatten_at value.sectionBits
    (Section.width widths) value.sectionBits_length sectionOrder
    (sectionOrder.idxOf part) bounded
  simpa [Value.encode, Section.bitOffset] using sliced

/-- Pointwise form of `encode_slice`. It is useful when a generated row
layout links one typed section directly into the complete claim envelope. -/
theorem Value.encode_get_section {widths : CompilerWidths}
    (value : Value widths) (part : Section)
    (index : Fin (part.width widths)) :
    value.encode.get
        ⟨part.bitOffset widths + index.val, by
          have fits := Section.slice_fits widths part
          rw [value.encode_length]
          omega⟩ =
      (value.sectionBits part).get
        ⟨index.val, by simpa [value.sectionBits_length] using index.isLt⟩ := by
  calc
    value.encode.get _ =
        ((value.encode.drop (part.bitOffset widths)).take
          (part.width widths))[index.val]'(by
            have remainingBound :
                index.val < value.encode.length - part.bitOffset widths := by
              rw [value.encode_length]
              have fits := Section.slice_fits widths part
              omega
            simp only [List.length_take, List.length_drop]
            exact lt_min index.isLt remainingBound) := by
      simp
    _ = (value.sectionBits part)[index.val]'_ := by
      simpa only [value.encode_slice part]

theorem Value.sectionBits_equal_of_encode_equal
    {widths : CompilerWidths} {left right : Value widths}
    (equal : left.encode = right.encode) (part : Section) :
    left.sectionBits part = right.sectionBits part := by
  calc
    left.sectionBits part =
        (left.encode.drop (part.bitOffset widths)).take
          (part.width widths) := (left.encode_slice part).symm
    _ = (right.encode.drop (part.bitOffset widths)).take
          (part.width widths) := by rw [equal]
    _ = right.sectionBits part := right.encode_slice part

/-- Equality of the complete canonical envelope recovers all six sections.
In particular, equality of only the memory section is not sufficient. -/
theorem Value.encode_injective_on_canonical
    {widths : CompilerWidths} {left right : Value widths}
    (leftCanonical : left.Canonical)
    (rightCanonical : right.Canonical)
    (equal : left.encode = right.encode) : left = right := by
  apply Value.ext
  · exact leftCanonical.profileExact.trans rightCanonical.profileExact.symm
  · apply Subtype.ext
    exact left.sectionBits_equal_of_encode_equal equal .ccsPublic
  · apply Subtype.ext
    exact left.sectionBits_equal_of_encode_equal equal .applicationPublic
  · apply CommitmentBundleCodec.encode_injective
    exact left.sectionBits_equal_of_encode_equal equal .commitmentBundle
  · apply Subtype.ext
    exact left.sectionBits_equal_of_encode_equal equal .recursiveState
  · apply MemoryClaimCodec.encode_injective_on_canonical
      leftCanonical.memoryCanonical rightCanonical.memoryCanonical
    exact left.sectionBits_equal_of_encode_equal equal .memory

/-- The exact protocol schema corresponding to this envelope. -/
def protocolSchema (widths : CompilerWidths) (NifsProof : Type) :
    FullClaim.Schema where
  CcsPublic := FixedBits.Word widths.ccsPublicBits
  ApplicationPublic := FixedBits.Word widths.applicationPublicBits
  CommitmentBundle := CommitmentBundleCodec.Value
  RecursiveState := FixedBits.Word widths.recursiveStateBits
  NifsProof := NifsProof

def Value.toProtocolClaim {widths : CompilerWidths} {NifsProof : Type}
    (value : Value widths) :
    FullClaim.Claim (protocolSchema widths NifsProof) Digest.Value
      (Challenges K) (State K) where
  profile := value.profile
  ccsPublic := value.ccsPublic
  applicationPublic := value.applicationPublic
  commitmentBundle := value.commitmentBundle
  recursiveState := value.recursiveState
  memory := value.memory

def Value.ofProtocolClaim {widths : CompilerWidths} {NifsProof : Type}
    (claim : FullClaim.Claim (protocolSchema widths NifsProof) Digest.Value
      (Challenges K) (State K)) : Value widths where
  profile := claim.profile
  ccsPublic := claim.ccsPublic
  applicationPublic := claim.applicationPublic
  commitmentBundle := claim.commitmentBundle
  recursiveState := claim.recursiveState
  memory := claim.memory

@[simp]
theorem Value.of_toProtocolClaim {widths : CompilerWidths} {NifsProof : Type}
    (value : Value widths) :
    Value.ofProtocolClaim
      (value.toProtocolClaim (NifsProof := NifsProof)) = value := rfl

@[simp]
theorem Value.to_ofProtocolClaim {widths : CompilerWidths} {NifsProof : Type}
    (claim : FullClaim.Claim (protocolSchema widths NifsProof) Digest.Value
      (Challenges K) (State K)) :
    (Value.ofProtocolClaim claim).toProtocolClaim = claim := by
  cases claim
  rfl

/-- The typed protocol claim retains all envelope sections. -/
theorem Value.toProtocolClaim_injective {widths : CompilerWidths}
    {NifsProof : Type} :
    Function.Injective
      (Value.toProtocolClaim (widths := widths) (NifsProof := NifsProof)) := by
  intro left right equal
  apply Value.ext
  · exact congrArg FullClaim.Claim.profile equal
  · exact congrArg FullClaim.Claim.ccsPublic equal
  · exact congrArg FullClaim.Claim.applicationPublic equal
  · exact congrArg FullClaim.Claim.commitmentBundle equal
  · exact congrArg FullClaim.Claim.recursiveState equal
  · exact congrArg FullClaim.Claim.memory equal

end Nightstream.Implementation.NebulaV2.FullClaimEnvelope
