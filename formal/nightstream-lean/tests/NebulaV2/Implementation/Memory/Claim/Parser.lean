import Nightstream.Implementation.NebulaV2.Memory.Claim.Parser

/-! Focused soundness and completeness gates for the exact claim parser. -/

set_option autoImplicit false

namespace tests.NebulaV2MemoryClaimParser

open Nightstream.Implementation.NebulaV2.MemoryClaimCodec
open Nightstream.Implementation.NebulaV2.MemoryClaimParser

example (claim : Claim) (canonical : claim.Canonical) :
    parse (blockOfClaim claim canonical) = some claim :=
  parse_blockOfClaim claim canonical

example {block : Block} {claim : Claim} (accepted : parse block = some claim) :
    claim.Canonical :=
  parse_claim_canonical accepted

example {block : Block} {claim : Claim} (accepted : parse block = some claim) :
    Nightstream.Implementation.NebulaV2.MemoryClaimFieldRows.NativeParses
      (rawWords block) claim :=
  parse_native_parses accepted

example {block : Block}
    (aliasEq : fieldWord block (.challenge 0 0 0) =
      Nightstream.Protocol.NebulaV2.CanonicalFieldBits.modulusWord) :
    parse block = none :=
  rejects_modulus_alias _ aliasEq

end tests.NebulaV2MemoryClaimParser
