import Nightstream.Implementation.Nebula.FPrime.State.AuthorityFullClaim
import Nightstream.Protocol.FPrime.Step

/-!
Contract: exact delayed fresh-public link for one Nebula V2 full claim.

Assurance tier: implementation model.

Owns the Boolean link from one producer state digest and one complete typed
fresh-claim envelope to the exact 540-coordinate V2 CCS public carrier. It
also proves that the generic F-prime singleton `FreshLinked` predicate is
exactly the authority-bearing `StateAuthorityFullClaim.Carries` relation.

Does not own production of the other full-claim sections, claim canonicality,
NIFS verification, placement of the 540 equality rows, state-hash collision
resistance, recursive-size closure, or Rust conformance.

Emits constraints: no. `PriorStateLinkRows` owns the concrete consumer rows.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.Nebula.FullClaimFreshLink

open Nightstream.Implementation.Nebula.FullClaimEnvelope
open Nightstream.Protocol.FPrime.Step

abbrev Digest := MemoryBoundCcsPublic.CanonicalDigest

/-- The only V2 fresh-public check. It compares all 540 coordinates. The
memory digest is recomputed from the memory suffix of this same typed claim. -/
def check {widths : CompilerWidths}
    (digest : Digest) (claim : Value widths) : Bool :=
  decide (claim.ccsPublic.val =
    PriorStateLinkRows.ccsEncoding digest claim.memory)

theorem check_eq_true_iff
    {widths : CompilerWidths} {digest : Digest} {claim : Value widths} :
    check digest claim = true ↔
      claim.ccsPublic.val =
        PriorStateLinkRows.ccsEncoding digest claim.memory := by
  simp [check]

/-- The V2 check is exactly the full-claim carrier relation for the normalized
producer authority. No digest equality or collision assumption is used. -/
theorem check_authority_eq_true_iff_carries
    {widths : CompilerWidths}
    {authority : StateAuthorityBoundaryRows.Authority}
    {claim : Value widths} :
    check (StateAuthorityFullClaim.canonicalDigest authority) claim = true ↔
      StateAuthorityFullClaim.Carries authority claim := by
  simpa [StateAuthorityFullClaim.Carries] using
    (check_eq_true_iff
      (digest := StateAuthorityFullClaim.canonicalDigest authority)
      (claim := claim))

/-- Factor-one V2 has one fresh claim per augmented invocation. The generic
F-prime list predicate therefore reduces exactly to the 540-coordinate typed
carrier relation. -/
theorem singleton_freshLinked_iff_carries
    {widths : CompilerWidths}
    {authority : StateAuthorityBoundaryRows.Authority}
    {claim : Value widths} :
    FreshLinked check
        (StateAuthorityFullClaim.canonicalDigest authority) [claim] ↔
      StateAuthorityFullClaim.Carries authority claim := by
  simp only [FreshLinked, List.all_cons, List.all_nil, Bool.and_true]
  exact check_authority_eq_true_iff_carries

/-- An exact carrier relation constructs the generic delayed link used by the
F-prime edge. -/
theorem freshLinked_of_carries
    {widths : CompilerWidths}
    {authority : StateAuthorityBoundaryRows.Authority}
    {claim : Value widths}
    (carries : StateAuthorityFullClaim.Carries authority claim) :
    FreshLinked check
      (StateAuthorityFullClaim.canonicalDigest authority) [claim] :=
  singleton_freshLinked_iff_carries.mpr carries

/-- Generic delayed-link acceptance recovers the exact typed V2 carrier. -/
theorem carries_of_freshLinked
    {widths : CompilerWidths}
    {authority : StateAuthorityBoundaryRows.Authority}
    {claim : Value widths}
    (linked : FreshLinked check
      (StateAuthorityFullClaim.canonicalDigest authority) [claim]) :
    StateAuthorityFullClaim.Carries authority claim :=
  singleton_freshLinked_iff_carries.mp linked

end Nightstream.Implementation.Nebula.FullClaimFreshLink
