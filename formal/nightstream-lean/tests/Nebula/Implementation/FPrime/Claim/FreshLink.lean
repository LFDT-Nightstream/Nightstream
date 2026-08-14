import Nightstream.Implementation.Nebula.FPrime.Claim.FreshLink

/-! Compile gate for the exact Nebula V2 delayed fresh-public link. -/

namespace tests.NebulaFullClaimFreshLink

open Nightstream.Implementation.Nebula
open Nightstream.Implementation.Nebula.FullClaimEnvelope
open Nightstream.Implementation.Nebula.FullClaimFreshLink

example {widths : CompilerWidths}
    {authority : StateAuthorityBoundaryRows.Authority}
    {claim : Value widths} :
    Nightstream.Protocol.FPrime.Step.FreshLinked check
        (StateAuthorityFullClaim.canonicalDigest authority) [claim] ↔
      StateAuthorityFullClaim.Carries authority claim :=
  singleton_freshLinked_iff_carries

end tests.NebulaFullClaimFreshLink
