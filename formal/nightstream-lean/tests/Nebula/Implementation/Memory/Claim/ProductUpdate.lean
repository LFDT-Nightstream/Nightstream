import Nightstream.Implementation.Nebula.Memory.Claim.ProductUpdate

/-! Focused type check for the exact aggregate claim-product bridge. -/

set_option autoImplicit false

namespace tests.NebulaMemoryClaimProductUpdate

open Nightstream.Implementation.Nebula.MemoryClaimCodec
open Nightstream.Implementation.Nebula
open Nightstream.Implementation.Nebula.MemoryClaimProductUpdate
open Nightstream.Implementation.Nebula.MemoryProductUpdateRows
open Nightstream.Implementation.R1CS
open Nightstream.Protocol.Nebula.ConcreteLaneGeometry
open Nightstream.SuperNeo.Concrete

theorem all_eight_endpoints_are_derived
    {layout : Layout} {assignment : Nat → Nat} {claim : Claim}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (parsed : MemoryClaimRows.ParsedColumnsMatch layout.claim assignment claim)
    (holds : Satisfies (rows layout) assignment)
    (records : CheckedStepRecords)
    (source : SourceRefines assignment layout records) :
    mapState claim.productsAfter =
      Nightstream.Protocol.Nebula.ProductState.update
        Nightstream.Implementation.Nebula.ConcreteField.encode
        (mapChallenges claim.challenge)
        (mapState claim.productsBefore) records.chunk :=
  claim_product_update canonical one parsed holds records source

end tests.NebulaMemoryClaimProductUpdate
