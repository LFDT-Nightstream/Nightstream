import Nightstream.Implementation.NebulaV2.RecursiveManifestStateAuthority
import Nightstream.Implementation.NebulaV2.TerminalManifestStateAuthority

/-! Focused gates for the exact full-claim state carrier. -/

set_option autoImplicit false

namespace tests.NebulaV2StateAuthorityFullClaim

open Nightstream.Implementation.NebulaV2
open Nightstream.Implementation.NebulaV2.FullClaimEnvelope
open Nightstream.Implementation.R1CS

theorem complete_carrier_forces_540_coordinates
    {widths : CompilerWidths}
    {authority : StateAuthorityBoundaryRows.Authority}
    {claim : Value widths}
    (carries : StateAuthorityFullClaim.Carries authority claim) :
    widths.ccsPublicBits = 540 := by
  simpa [PriorStateLinkRows.ccsPublicBitCount] using
    StateAuthorityFullClaim.carries_ccs_width carries

theorem one_claim_has_one_authority_or_named_collision
    {widths : CompilerWidths} {claim : Value widths}
    {left right : StateAuthorityBoundaryRows.Authority}
    (leftCarries : StateAuthorityFullClaim.Carries left claim)
    (rightCarries : StateAuthorityFullClaim.Carries right claim) :
    StateAuthorityBoundaryRows.Same left right ∨
      StateAuthorityBoundaryRows.Failure :=
  StateAuthorityFullClaim.same_claim_authority_eq_or_failure
    leftCarries rightCarries

theorem equal_authority_digests_transport_the_complete_carrier
    {widths : CompilerWidths} {claim : Value widths}
    {left right : StateAuthorityBoundaryRows.Authority}
    (digestEqual : left.digest = right.digest)
    (rightCarries : StateAuthorityFullClaim.Carries right claim) :
    StateAuthorityFullClaim.Carries left claim :=
  StateAuthorityFullClaim.carries_of_digest_eq digestEqual rightCarries

theorem equal_carriers_recover_authority_and_memory_or_named_collision
    {leftWidths rightWidths : CompilerWidths}
    {leftClaim : Value leftWidths} {rightClaim : Value rightWidths}
    {left right : StateAuthorityBoundaryRows.Authority}
    (leftCanonical : leftClaim.memory.Canonical)
    (rightCanonical : rightClaim.memory.Canonical)
    (leftCarries : StateAuthorityFullClaim.Carries left leftClaim)
    (rightCarries : StateAuthorityFullClaim.Carries right rightClaim)
    (carrierEqual : leftClaim.ccsPublic.val = rightClaim.ccsPublic.val) :
    (StateAuthorityBoundaryRows.Same left right ∧
        leftClaim.memory = rightClaim.memory) ∨
      StateAuthorityFullClaim.CarrierFailure :=
  StateAuthorityFullClaim.equal_carriers_authority_and_memory_or_failure
    leftCanonical rightCanonical leftCarries rightCarries carrierEqual

end tests.NebulaV2StateAuthorityFullClaim
