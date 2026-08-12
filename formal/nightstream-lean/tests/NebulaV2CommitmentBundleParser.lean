import Nightstream.Implementation.NebulaV2.CommitmentBundleParser

set_option autoImplicit false

namespace tests.NebulaV2CommitmentBundleParser

open Nightstream.Implementation.NebulaV2.CommitmentBundleCodec
open Nightstream.Implementation.NebulaV2.CommitmentBundleParser

example (bundle : Value) :
    parse (blockOfBundle bundle) = some bundle :=
  parse_blockOfBundle bundle

example {block : Block} {bundle : Value}
    (accepted : parse block = some bundle) :
    encode bundle = block.val :=
  parse_success_reencodes accepted

example (block : Block) (slot :
    Nightstream.Implementation.NebulaV2.CommitmentBundleFieldRows.Slot)
    (aliasEq : fieldWord block slot =
      Nightstream.Protocol.NebulaV2.CanonicalFieldBits.modulusWord) :
    parse block = none :=
  rejects_modulus_alias block slot aliasEq

end tests.NebulaV2CommitmentBundleParser
