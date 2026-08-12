import Nightstream.Implementation.NebulaV2.MemoryCarryCodec

set_option autoImplicit false

namespace tests.NebulaV2MemoryCarryCodec

open Nightstream.Implementation.NebulaV2.MemoryCarryCodec
open Nightstream.Implementation.NebulaV2.MemoryClaimCodec
open Nightstream.Protocol.NebulaV2
open Nightstream.Protocol.NebulaV2.CarryEncoding
open Nightstream.Protocol.NebulaV2.FPrime
open Nightstream.SuperNeo.Concrete

def canonicalZero : ShiftedTernary41V1.CanonicalGoldilocks :=
  ⟨0, by norm_num [ShiftedTernary41V1.modulus]⟩

def zeroDigest : Digest.Value :=
  { lanes := fun _ => canonicalZero }

def headers : ChainHeaders Digest.Value :=
  ⟨zeroDigest, zeroDigest⟩

def closedValue : Value :=
  { phase := .closed
    segmentIndex := 0
    stepIndex := 0
    globalTimestamp := 0
    segmentStartTimestamp := 0
    segmentActiveAccessCount := 0
    segmentEndTimestamp := 0
    challenges := zeroChallengesK
    products := oneProductsK
    dPre := headers.roots
    dSeen := headers.roots
    memoryRoot := zeroDigest }

def closedCanonical : closedValue.Canonical headers where
  segmentIndex := by decide
  stepIndex := by decide
  globalTimestamp := by decide
  segmentStartTimestamp := by decide
  segmentActiveAccessCount := by decide
  segmentEndTimestamp := by decide
  closedFields := by
    intro _
    exact ⟨rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl⟩

theorem closed_encoding_has_exact_width :
    (encode closedValue).length = 3433 :=
  encode_exact_length closedValue

theorem carry_product_order_is_normative :
    productRoles = [.reads, .writes, .initialSnapshot, .finalSnapshot] :=
  rfl

def noncanonicalClosed : Value :=
  { closedValue with
    challenges := fun _ => { gamma1 := K.one, gamma2 := K.zero } }

/-- A closed wire with a nonzero inactive challenge is rejected even though
its visible closed carry fields are unchanged. -/
theorem nonzero_inactive_challenge_is_not_canonical :
    ¬ noncanonicalClosed.Canonical headers := by
  intro canonical
  have closedFields := canonical.closedFields rfl
  have challengeEqual := congrFun closedFields.2.2.2.2.1 (0 : Fin 2)
  have coefficientEqual := congrArg
    (fun challenge => challenge.gamma1.c0.val) challengeEqual
  change 1 = 0 at coefficientEqual
  omega

end tests.NebulaV2MemoryCarryCodec
