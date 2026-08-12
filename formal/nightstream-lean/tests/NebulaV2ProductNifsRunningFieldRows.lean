import Nightstream.Implementation.NebulaV2.ProductNifsRunningFieldRows

set_option autoImplicit false

namespace tests.NebulaV2ProductNifsRunningFieldRows

open Nightstream.Implementation.NebulaV2.ProductNifsCodec
open Nightstream.Implementation.NebulaV2.ProductNifsRunningFieldRows

example (layout : Layout) : (rows layout).length = 11066930 :=
  rows_length_exact layout

#check parsed_columns_match
#check parse_from_rows
#check modulus_alias_impossible
#check Nightstream.Implementation.NebulaV2.ProductNifsRunningParser.parse_success_fields

end tests.NebulaV2ProductNifsRunningFieldRows
