import Nightstream.Implementation.NebulaV2.MemoryClaimProductUpdate

/-! Focused type check for the exact aggregate claim-product bridge. -/

set_option autoImplicit false

namespace tests.NebulaV2MemoryClaimProductUpdate

open Nightstream.Implementation.NebulaV2.MemoryClaimCodec
open Nightstream.Implementation.NebulaV2
open Nightstream.Implementation.NebulaV2.MemoryClaimProductUpdate
open Nightstream.Implementation.NebulaV2.MemoryProductUpdateRows
open Nightstream.Implementation.R1CS
open Nightstream.Protocol.NebulaV2.ConcreteLaneGeometry
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
      Nightstream.Protocol.NebulaV2.ProductState.update
        Nightstream.Implementation.NebulaV2.ConcreteField.encode
        (mapChallenges claim.challenge)
        (mapState claim.productsBefore) records.chunk :=
  claim_product_update canonical one parsed holds records source

end tests.NebulaV2MemoryClaimProductUpdate
