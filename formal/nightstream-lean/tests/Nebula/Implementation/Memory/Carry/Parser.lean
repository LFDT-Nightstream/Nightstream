import Nightstream.Implementation.Nebula.Memory.Carry.Parser

/-! Focused soundness and completeness gates for the exact carry parser. -/

set_option autoImplicit false

namespace tests.NebulaMemoryCarryParser

open Nightstream.Implementation.Nebula.MemoryCarryCodec
open Nightstream.Implementation.Nebula.MemoryCarryParser
open Nightstream.Protocol.Nebula.FPrime

example {headers : ChainHeaders
      Nightstream.Protocol.Nebula.Digest.Value}
    (value : Value) (canonical : value.Canonical headers) :
    parse headers (blockOfValue value canonical) = some value :=
  parse_blockOfValue value canonical

example {headers : ChainHeaders
      Nightstream.Protocol.Nebula.Digest.Value}
    {block : Block} {value : Value}
    (accepted : parse headers block = some value) :
    value.Canonical headers :=
  parse_value_canonical accepted

example {headers : ChainHeaders
      Nightstream.Protocol.Nebula.Digest.Value}
    {block : Block} {value : Value}
    (accepted : parse headers block = some value) :
    Nightstream.Implementation.Nebula.MemoryCarryFieldRows.NativeParses
      (rawWords block) value :=
  parse_native_parses accepted

example {headers : ChainHeaders
      Nightstream.Protocol.Nebula.Digest.Value}
    {block : Block}
    (aliasEq : fieldWord block (.root .memory 3) =
      Nightstream.Protocol.Nebula.CanonicalFieldBits.modulusWord) :
    parse headers block = none :=
  rejects_modulus_alias _ aliasEq

end tests.NebulaMemoryCarryParser
