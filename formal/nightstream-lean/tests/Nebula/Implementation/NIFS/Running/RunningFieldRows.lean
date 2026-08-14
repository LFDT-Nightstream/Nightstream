import Nightstream.Implementation.Nebula.NIFS.Running.RunningFieldRows

set_option autoImplicit false

namespace tests.NebulaProductNifsRunningFieldRows

open Nightstream.Implementation.Nebula.ProductNifsCodec
open Nightstream.Implementation.Nebula.ProductNifsRunningFieldRows

example (layout : Layout) : (rows layout).length = 11066930 :=
  rows_length_exact layout

#check parsed_columns_match
#check parse_from_rows
#check modulus_alias_impossible
#check Nightstream.Implementation.Nebula.ProductNifsRunningParser.parse_success_fields

end tests.NebulaProductNifsRunningFieldRows
