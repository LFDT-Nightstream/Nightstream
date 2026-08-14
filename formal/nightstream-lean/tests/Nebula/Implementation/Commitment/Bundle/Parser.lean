import Nightstream.Implementation.Nebula.Commitment.Bundle.Parser

set_option autoImplicit false

namespace tests.NebulaCommitmentBundleParser

open Nightstream.Implementation.Nebula.CommitmentBundleCodec
open Nightstream.Implementation.Nebula.CommitmentBundleParser

example (bundle : Value) :
    parse (blockOfBundle bundle) = some bundle :=
  parse_blockOfBundle bundle

example {block : Block} {bundle : Value}
    (accepted : parse block = some bundle) :
    encode bundle = block.val :=
  parse_success_reencodes accepted

example (block : Block) (slot :
    Nightstream.Implementation.Nebula.CommitmentBundleFieldRows.Slot)
    (aliasEq : fieldWord block slot =
      Nightstream.Protocol.Nebula.CanonicalFieldBits.modulusWord) :
    parse block = none :=
  rejects_modulus_alias block slot aliasEq

end tests.NebulaCommitmentBundleParser
