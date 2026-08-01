import Nightstream.Implementation.Lowering.Nebula.SourceSemantics

set_option autoImplicit false

namespace tests.NebulaSourceSemantics

open Nightstream.SuperNeo.Concrete
open Nightstream.Implementation.Lowering.Nebula.Layout
open Nightstream.Implementation.Lowering.Nebula.ProductSemantics
open Nightstream.Implementation.Lowering.Nebula.SourceSemantics
open Nightstream.Protocol.Nebula.Fingerprint

/-- The generic source decoder preserves the exact timestamp/address packing
used by the protocol fingerprint. -/
theorem source_packing_example :
    packed (entryOfFields (5 : F) (7 : F) (11 : F)) =
      5 +
        Nightstream.Implementation.Lowering.Nebula.Rows.LinearCombination.fieldTwoPower
          Nightstream.Implementation.Lowering.Nebula.Layout.timestampBits * 7 :=
  entryOfFields_packed _ _ _

/-- The selected operation factor is not an independent compiler value: it is
the protocol fingerprint of the decoded source tuple. -/
theorem selected_operation_factor_is_source_bound (assignment : Nat -> F) :
    operationFactor assignment wasm42x6 0 false =
      fingerprint (challenges assignment)
        (operationEntry assignment wasm42x6 0 false) :=
  operationFactor_eq_fingerprint assignment wasm42x6 0 false

/-- The same source binding holds for every selected scan factor. -/
theorem selected_scan_factor_is_source_bound (assignment : Nat -> F) :
    scanFactor assignment wasm42x6 true 17 =
      fingerprint (challenges assignment)
        (scanEntry assignment wasm42x6 true 17) :=
  scanFactor_eq_fingerprint assignment wasm42x6 true 17

end tests.NebulaSourceSemantics
