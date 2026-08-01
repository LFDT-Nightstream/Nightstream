import Nightstream.Implementation.Lowering.Nebula.SourceProducts

set_option autoImplicit false

namespace tests.NebulaSourceProducts

open Nightstream.SuperNeo.Concrete
open Nightstream.Implementation.Lowering.Nebula.Layout
open Nightstream.Implementation.Lowering.Nebula.Rows
open Nightstream.Implementation.Lowering.Nebula.Compiler
open Nightstream.Implementation.Lowering.Nebula.ProductSemantics
open Nightstream.Implementation.Lowering.Nebula.SourceProducts

theorem selected_products_are_source_bound
    (assignment : Nat -> F)
    (constantWire : assignment 0 = 1)
    (satisfied : Satisfies (rows wasm42x6) assignment)
    (activeAt : Nat -> Bool)
    (activation : ActivationMatches assignment wasm42x6 activeAt 1) :
    outputProduct assignment 0 =
        K.mul (inputProduct assignment 0)
          (Nightstream.Protocol.Nebula.Fingerprint.product
            (Nightstream.Implementation.Lowering.Nebula.SourceSemantics.challenges
              assignment)
            (operationEntries assignment wasm42x6 false activeAt 1)) :=
  (wasm42x6_public_products_source_bound assignment constantWire satisfied
    activeAt activation).1

end tests.NebulaSourceProducts
