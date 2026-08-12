import Nightstream.Implementation.NebulaV2.FPrime.Claim.FreshLink

/-! Compile gate for the exact Nebula V2 delayed fresh-public link. -/

namespace tests.NebulaV2FullClaimFreshLink

open Nightstream.Implementation.NebulaV2
open Nightstream.Implementation.NebulaV2.FullClaimEnvelope
open Nightstream.Implementation.NebulaV2.FullClaimFreshLink

example {widths : CompilerWidths}
    {authority : StateAuthorityBoundaryRows.Authority}
    {claim : Value widths} :
    Nightstream.Protocol.FPrime.Step.FreshLinked check
        (StateAuthorityFullClaim.canonicalDigest authority) [claim] ↔
      StateAuthorityFullClaim.Carries authority claim :=
  singleton_freshLinked_iff_carries

end tests.NebulaV2FullClaimFreshLink
