import Nightstream.Implementation.NebulaV2.PriorStateLinkRows
import Nightstream.Implementation.NebulaV2.StateAuthorityBoundaryRows

/-!
Contract: exact full-claim carrier for one normalized Nebula V2 state.

Assurance tier: implementation model and cryptographic boundary.

Owns the canonical four-lane state digest, the complete 540-coordinate CCS
carrier relation, row-derived construction of that relation, and recovery of
one exact typed state and one exact memory suffix, or one named Poseidon2
collision.

Does not own an invocation schedule, selected NIFS soundness, generated
producer rows, Poseidon2 collision resistance, or Rust conformance.

Emits constraints: no new rows. It interprets the mandatory prior-state rows.
-/

set_option autoImplicit false
set_option maxRecDepth 10000

namespace Nightstream.Implementation.NebulaV2.StateAuthorityFullClaim

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.NebulaV2.FullClaimEnvelope
open Nightstream.Protocol.NebulaV2

abbrev Authority := StateAuthorityBoundaryRows.Authority
abbrev Failure := StateAuthorityBoundaryRows.Failure

/-- Every output of the fixed Poseidon2 permutation is a canonical
Goldilocks integer. This is an arithmetic property of the executable
permutation, not a collision-resistance assumption. -/
theorem authorityDigest_canonical (authority : Authority) (lane : Fin 4) :
    authority.digest lane < goldilocksP := by
  change StateOutputPoseidonRows.pureDigest
      (AuthoritativeStateOutputBinding.typedFrame authority.payload
        authority.carryBlock) lane.val < goldilocksP
  unfold StateOutputPoseidonRows.pureDigest
  exact Poseidon2Sponge.runValueRounds_canonical _ _ _
    (fun _ => by norm_num [goldilocksP]) lane.val

/-- Canonical field-lane view used by the exact CCS bit carrier. -/
def canonicalDigest (authority : Authority) :
    Fin 4 → ShiftedTernary41V1.CanonicalGoldilocks :=
  fun lane => ⟨authority.digest lane, authorityDigest_canonical authority lane⟩

/-- Protocol digest wrapper for the exact normalized state authority. -/
def digestValue (authority : Authority) : Digest.Value where
  lanes := canonicalDigest authority

@[simp] theorem canonicalDigest_val (authority : Authority) (lane : Fin 4) :
    (canonicalDigest authority lane).val = authority.digest lane :=
  rfl

@[simp] theorem digestValue_lanes (authority : Authority) :
    (digestValue authority).lanes = canonicalDigest authority :=
  rfl

/-- The authority-bearing relation between one complete full claim, one
normalized prior state, and that claim's complete memory suffix. Equality is
over all 540 coordinates. -/
def Carries {widths : CompilerWidths}
    (authority : Authority) (claim : Value widths) : Prop :=
  claim.ccsPublic.val = PriorStateLinkRows.ccsEncoding
    (canonicalDigest authority) claim.memory

/-- Equality of the four authority digests transports the complete carrier
relation. No typed-state equality or collision-resistance assumption is
needed because the CCS carrier contains only the canonical digest lanes. -/
theorem carries_of_digest_eq
    {widths : CompilerWidths} {claim : Value widths}
    {left right : Authority}
    (digestEqual : left.digest = right.digest)
    (rightCarries : Carries right claim) :
    Carries left claim := by
  have canonicalEqual : canonicalDigest left = canonicalDigest right := by
    funext lane
    apply Subtype.ext
    exact congrFun digestEqual lane
  simpa [Carries, canonicalEqual] using rightCarries

/-- Exact state equality transports the complete carrier relation. This does
not compare digests or assume collision resistance: `Same` already contains
equality of the typed payload and the complete carry block. -/
theorem carries_of_same
    {widths : CompilerWidths} {claim : Value widths}
    {left right : Authority}
    (same : StateAuthorityBoundaryRows.Same left right)
    (rightCarries : Carries right claim) :
    Carries left claim := by
  exact carries_of_digest_eq
    (StateAuthorityBoundaryRows.digest_eq_of_same same) rightCarries

theorem carries_ccs_width
    {widths : CompilerWidths} {authority : Authority} {claim : Value widths}
    (carries : Carries authority claim) :
    widths.ccsPublicBits = PriorStateLinkRows.ccsPublicBitCount := by
  calc
    widths.ccsPublicBits = claim.ccsPublic.val.length :=
      claim.ccsPublic.property.1.symm
    _ = (PriorStateLinkRows.ccsEncoding
          (canonicalDigest authority) claim.memory).length :=
      congrArg List.length carries
    _ = PriorStateLinkRows.ccsPublicBitCount :=
      PriorStateLinkRows.ccsEncoding_length _ _

/-- Mandatory prior-state rows construct the complete carrier relation. The
digest equality only identifies the row-derived output lanes with the
normalized typed authority; it does not assume a carrier equality. -/
theorem carries_of_ccsPublicExact
    {widths : CompilerWidths} {layout : PriorStateLinkRows.Layout widths}
    {assignment : Nat → Nat} {claim : Value widths}
    {canonical : ∀ column, assignment column < goldilocksP}
    {valid : layout.Valid} {authority : Authority}
    (exact : PriorStateLinkRows.CcsPublicExact valid assignment claim canonical)
    (digestEqual :
      PriorStateLinkRows.outputDigest layout assignment canonical =
        canonicalDigest authority) :
    Carries authority claim := by
  have wordEqual := exact.ccsPublic_eq_ccsPublicWord
  have valuesEqual := congrArg Subtype.val wordEqual
  simpa [Carries, PriorStateLinkRows.CcsPublicExact.typedWord,
    digestEqual] using valuesEqual

/-- For one fixed memory suffix, the selected carrier is injective on its
four canonical state-digest lanes. -/
theorem ccsEncoding_injective (memory : MemoryClaimCodec.Claim) :
    Function.Injective (fun digest =>
      PriorStateLinkRows.ccsEncoding digest memory) := by
  intro left right equal
  exact MemoryBoundCcsPublic.stateDigest_eq_of_encode_eq equal

/-- One complete full claim cannot carry two different typed states unless
the fixed inner or outer Poseidon2 state hash has a concrete collision. -/
theorem same_claim_authority_eq_or_failure
    {widths : CompilerWidths} {claim : Value widths}
    {left right : Authority}
    (leftCarries : Carries left claim)
    (rightCarries : Carries right claim) :
    StateAuthorityBoundaryRows.Same left right ∨ Failure := by
  classical
  have encodedEqual :
      PriorStateLinkRows.ccsEncoding (canonicalDigest left) claim.memory =
        PriorStateLinkRows.ccsEncoding (canonicalDigest right) claim.memory :=
    leftCarries.symm.trans rightCarries
  have canonicalEqual := ccsEncoding_injective claim.memory encodedEqual
  have digestEqual : left.digest = right.digest := by
    funext lane
    exact congrArg Subtype.val (congrFun canonicalEqual lane)
  rcases
      AuthoritativeStateOutputBinding.typed_authority_eq_or_two_stage_collision
        (StateOutputAuthorityRows.fullFrame_length _ _)
        (StateOutputAuthorityRows.fullFrame_length _ _)
        left.frameCanonical right.frameCanonical digestEqual with
    same | outerOrInner
  · exact Or.inl same
  · rcases outerOrInner with outer | inner
    · exact Or.inr (.outer outer)
    · exact Or.inr (.inner inner)

inductive CarrierFailure : Prop where
  | state : Failure → CarrierFailure
  | memory : MemoryClaimPoseidonBinding.PoseidonCollision → CarrierFailure

/-- Equal 540-coordinate carriers recover both authority-bearing objects, or
expose one exact named state-hash or memory-suffix-hash collision. -/
theorem equal_carriers_authority_and_memory_or_failure
    {leftWidths rightWidths : CompilerWidths}
    {leftClaim : Value leftWidths} {rightClaim : Value rightWidths}
    {left right : Authority}
    (leftCanonical : leftClaim.memory.Canonical)
    (rightCanonical : rightClaim.memory.Canonical)
    (leftCarries : Carries left leftClaim)
    (rightCarries : Carries right rightClaim)
    (carrierEqual : leftClaim.ccsPublic.val = rightClaim.ccsPublic.val) :
    (StateAuthorityBoundaryRows.Same left right ∧
        leftClaim.memory = rightClaim.memory) ∨ CarrierFailure := by
  classical
  have encodedEqual :
      PriorStateLinkRows.ccsEncoding (canonicalDigest left) leftClaim.memory =
        PriorStateLinkRows.ccsEncoding (canonicalDigest right)
          rightClaim.memory :=
    leftCarries.symm.trans (carrierEqual.trans rightCarries)
  rcases MemoryBoundCcsPublic.authority_eq_or_memory_collision
      leftCanonical rightCanonical encodedEqual with exact | memoryCollision
  · rcases exact with ⟨digestEqual, memoryEqual⟩
    have rawDigestEqual : left.digest = right.digest := by
      funext lane
      exact congrArg Subtype.val (congrFun digestEqual lane)
    rcases
        AuthoritativeStateOutputBinding.typed_authority_eq_or_two_stage_collision
          (StateOutputAuthorityRows.fullFrame_length _ _)
          (StateOutputAuthorityRows.fullFrame_length _ _)
          left.frameCanonical right.frameCanonical rawDigestEqual with
      same | outerOrInner
    · exact Or.inl ⟨same, memoryEqual⟩
    · rcases outerOrInner with outer | inner
      · exact Or.inr (.state (.outer outer))
      · exact Or.inr (.state (.inner inner))
  · exact Or.inr (.memory memoryCollision)

end Nightstream.Implementation.NebulaV2.StateAuthorityFullClaim
