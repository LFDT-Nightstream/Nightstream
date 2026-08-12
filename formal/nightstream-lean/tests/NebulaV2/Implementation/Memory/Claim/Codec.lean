import Nightstream.Implementation.NebulaV2.Memory.Claim.Codec

set_option autoImplicit false

namespace tests.NebulaV2MemoryClaimCodec

open Nightstream.Implementation.NebulaV2.MemoryClaimCodec
open Nightstream.Protocol.NebulaV2
open Nightstream.Protocol.NebulaV2.FPrime
open Nightstream.Protocol.NebulaV2.ProductState
open Nightstream.SuperNeo.Concrete

def canonicalZero : ShiftedTernary41V1.CanonicalGoldilocks :=
  ⟨0, by norm_num [ShiftedTernary41V1.modulus]⟩

def zeroDigest : Digest.Value :=
  { lanes := fun _ => canonicalZero }

def zeroRoots : Roots Digest.Value :=
  ⟨zeroDigest, zeroDigest, zeroDigest⟩

def zeroChallenges : Challenges K :=
  fun _ => { gamma1 := K.zero, gamma2 := K.zero }

def oneProducts : State K :=
  fun _ =>
    { initialSnapshot := K.one
      writes := K.one
      reads := K.one
      finalSnapshot := K.one }

def claim : Claim :=
  { segmentIndex := 0
    stepIndex := ⟨0, by decide⟩
    timestampIn := 0
    timestampOut := 0
    segmentStartTimestamp := 0
    segmentEndTimestamp := 0
    activeAccessCount := 0
    challenge := zeroChallenges
    dPre := zeroRoots
    dSeenBefore := zeroRoots
    dSeenAfter := zeroRoots
    productsBefore := oneProducts
    productsAfter := oneProducts }

def claimCanonical : claim.Canonical where
  segmentIndex := by decide
  timestampIn := by decide
  timestampOut := by decide
  segmentStartTimestamp := by decide
  segmentEndTimestamp := by decide
  activeAccessCount := by decide

theorem encoded_claim_has_exact_width :
    (encode claim).length = 4980 :=
  encode_exact_length claim

theorem product_block_starts_with_h_rs_repetition_zero_c0 :
    productSchema.head? =
      some (.product 0 0 .reads 0) := by
  rfl

theorem product_role_order_is_normative :
    productRoles = [.reads, .writes, .initialSnapshot, .finalSnapshot] :=
  rfl

theorem encoded_claim_is_binary (digit : Nat) (member : digit ∈ encode claim) :
    digit < 2 :=
  encode_binary claim digit member

def changedChallenges : Challenges K :=
  fun repetition =>
    if repetition = 0 then { gamma1 := K.one, gamma2 := K.zero }
    else { gamma1 := K.zero, gamma2 := K.zero }

def changedClaim : Claim :=
  { claim with challenge := changedChallenges }

def changedClaimCanonical : changedClaim.Canonical where
  segmentIndex := claimCanonical.segmentIndex
  timestampIn := claimCanonical.timestampIn
  timestampOut := claimCanonical.timestampOut
  segmentStartTimestamp := claimCanonical.segmentStartTimestamp
  segmentEndTimestamp := claimCanonical.segmentEndTimestamp
  activeAccessCount := claimCanonical.activeAccessCount

/-- A challenge substitution changes the canonical public block even when
all roots, counters, and products remain unchanged. -/
theorem changed_challenge_changes_encoding :
    encode changedClaim ≠ encode claim := by
  intro equalEncoding
  have equalClaim := encode_injective_on_canonical
    changedClaimCanonical claimCanonical equalEncoding
  have coefficientEqual := congrArg
    (fun suffix : Claim => (suffix.challenge 0).gamma1.c0.val) equalClaim
  change 1 = 0 at coefficientEqual
  omega

end tests.NebulaV2MemoryClaimCodec
