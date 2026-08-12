import Nightstream.Implementation.NebulaV2.ProductNifsFieldParser

set_option autoImplicit false

namespace tests.NebulaV2ProductNifsFieldParser

open Nightstream.Implementation.NebulaV2.ProductNifsFieldParser
open Nightstream.Protocol.NebulaV2
open Nightstream.SuperNeo.Concrete

example (values : Fin 2 → F) :
    parse (encode values) = some values :=
  parse_encode values

example (values : Fin 2 → F) :
    AllCanonical (encode values) :=
  encode_allCanonical values

#check parse_success_canonical
#check parse_success_fields
#check parse_rejects_noncanonical
#check parse_rejects_modulus_word
#check fieldWord_encode
#check parse_encode

end tests.NebulaV2ProductNifsFieldParser
