import Nightstream.Implementation.Nebula.Memory.Segment.CheckedRows

/-! Focused gates for exact row-derived segment product accumulation. -/

set_option autoImplicit false

namespace tests.NebulaSegmentCheckedRows

open Nightstream.Implementation.Nebula.ConcreteField
open Nightstream.Implementation.Nebula
open Nightstream.Implementation.Nebula.FullClaimEnvelope
open Nightstream.Implementation.Nebula.FullClaimNifsReceipt
open Nightstream.Implementation.Nebula.MemoryClaimProductUpdate
open Nightstream.Implementation.Nebula.MemoryProductBalanceBridge
open Nightstream.Implementation.Nebula.MemoryProductBalanceRows
open Nightstream.Implementation.Nebula.RecursiveManifestNifsCall
open Nightstream.Implementation.Nebula.RecursiveManifestSchema
open Nightstream.Implementation.Nebula.SegmentCheckedRows
open Nightstream.Protocol.Nebula
open Nightstream.Protocol.Nebula.FPrime
open Nightstream.Protocol.Nebula.ProductState
open Nightstream.SuperNeo.Concrete

theorem concrete_close_is_exact_math_balance
    (products : State K) :
    ConcreteBalanced products ↔ Balanced (mapState products) :=
  concreteBalanced_iff_mapped products

theorem canonical_open_maps_to_one :
    mapState MemoryCarryCodec.oneProductsK =
      (ProductState.one : State ChallengeField) :=
  mapState_oneProductsK

variable {widths : CompilerWidths}
variable {artifact : Artifact widths} {selected : SelectedVerifier widths}

theorem closed_row_run_has_exact_count
    {active : ActiveCarry Digest.Value (Challenges K) (State K)}
    {closed : ClosedCarry Digest.Value}
    {invocations : List (Invocation artifact selected)}
    (run : Run artifact selected (.active active) invocations (.closed closed))
    (startsAtZero : active.stepIndex.val = 0) :
    invocations.length = Lifecycle.claimsPerSegment :=
  run.exactClaimCount startsAtZero

theorem canonical_closed_row_run_is_balanced
    {active : ActiveCarry Digest.Value (Challenges K) (State K)}
    {closed : ClosedCarry Digest.Value}
    {invocations : List (Invocation artifact selected)}
    (run : Run artifact selected (.active active) invocations (.closed closed))
    (openingProducts : active.products = MemoryCarryCodec.oneProductsK) :
    ProductState.Balanced
      (ProductState.accumulate encode (mapChallenges active.challenge)
        ProductState.one (Run.chunks invocations)) :=
  run.accumulatedFromOneBalanced openingProducts

end tests.NebulaSegmentCheckedRows
