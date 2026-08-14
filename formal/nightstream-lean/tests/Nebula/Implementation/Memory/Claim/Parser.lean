import Nightstream.Implementation.Nebula.Memory.Claim.Parser

/-! Focused soundness and completeness gates for the exact claim parser. -/

set_option autoImplicit false

namespace tests.NebulaMemoryClaimParser

open Nightstream.Implementation.Nebula.MemoryClaimCodec
open Nightstream.Implementation.Nebula.MemoryClaimParser

example (claim : Claim) (canonical : claim.Canonical) :
    parse (blockOfClaim claim canonical) = some claim :=
  parse_blockOfClaim claim canonical

example {block : Block} {claim : Claim} (accepted : parse block = some claim) :
    claim.Canonical :=
  parse_claim_canonical accepted

example {block : Block} {claim : Claim} (accepted : parse block = some claim) :
    Nightstream.Implementation.Nebula.MemoryClaimFieldRows.NativeParses
      (rawWords block) claim :=
  parse_native_parses accepted

example {block : Block}
    (aliasEq : fieldWord block (.challenge 0 0 0) =
      Nightstream.Protocol.Nebula.CanonicalFieldBits.modulusWord) :
    parse block = none :=
  rejects_modulus_alias _ aliasEq

end tests.NebulaMemoryClaimParser
