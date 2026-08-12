import Nightstream.Implementation.NebulaV2.Memory.Carry.Parser

/-! Focused soundness and completeness gates for the exact carry parser. -/

set_option autoImplicit false

namespace tests.NebulaV2MemoryCarryParser

open Nightstream.Implementation.NebulaV2.MemoryCarryCodec
open Nightstream.Implementation.NebulaV2.MemoryCarryParser
open Nightstream.Protocol.NebulaV2.FPrime

example {headers : ChainHeaders
      Nightstream.Protocol.NebulaV2.Digest.Value}
    (value : Value) (canonical : value.Canonical headers) :
    parse headers (blockOfValue value canonical) = some value :=
  parse_blockOfValue value canonical

example {headers : ChainHeaders
      Nightstream.Protocol.NebulaV2.Digest.Value}
    {block : Block} {value : Value}
    (accepted : parse headers block = some value) :
    value.Canonical headers :=
  parse_value_canonical accepted

example {headers : ChainHeaders
      Nightstream.Protocol.NebulaV2.Digest.Value}
    {block : Block} {value : Value}
    (accepted : parse headers block = some value) :
    Nightstream.Implementation.NebulaV2.MemoryCarryFieldRows.NativeParses
      (rawWords block) value :=
  parse_native_parses accepted

example {headers : ChainHeaders
      Nightstream.Protocol.NebulaV2.Digest.Value}
    {block : Block}
    (aliasEq : fieldWord block (.root .memory 3) =
      Nightstream.Protocol.NebulaV2.CanonicalFieldBits.modulusWord) :
    parse headers block = none :=
  rejects_modulus_alias _ aliasEq

end tests.NebulaV2MemoryCarryParser
