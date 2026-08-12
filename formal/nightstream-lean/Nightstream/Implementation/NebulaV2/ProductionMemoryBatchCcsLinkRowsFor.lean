import Nightstream.Implementation.NebulaV2.ProductionFullClaimCarrierLayoutFor
import Nightstream.Implementation.NebulaV2.ProductionMemoryBatchCcsLinkRows

/-!
Contract: exponent-indexed full-claim carrier adapter for the exact checked
memory-batch CCS public-image relation.

The CCS relation reads only the 540 CCS public coordinates. This adapter proves
that those coordinates are identical in the generated exponent-indexed carrier
and the exponent-independent CCS row program. It does not manufacture or pad a
running state of the historical fixed-25 width.

Assurance tier: exponent-indexed physical authority bridge.

Emits constraints: no new constraints; `rows` is the checked CCS row program.
-/

set_option autoImplicit false
set_option maxRecDepth 100000

namespace Nightstream.Implementation.NebulaV2.ProductionMemoryBatchCcsLinkRowsFor

open Nightstream.Implementation.NebulaV2
open Nightstream.Implementation.R1CS
open Nightstream.Protocol.NebulaV2
open Nightstream.Protocol.NebulaV2.ProductionProfileCandidates
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation

abbrev CanonicalDigest :=
  ProductionMemoryBatchCcsLinkRows.CanonicalDigest

/-- `core` owns the CCS and memory rows. `carrier` owns the complete generated
claim window. Only their starts must agree because the CCS window precedes the
exponent-dependent running window. -/
structure Layout (candidate : Id) (rowVariables : Nat) where
  carrier : ProductionFullClaimCarrierLayoutFor.Layout candidate rowVariables
  core : ProductionMemoryBatchCcsLinkRows.Layout candidate
  carrierStart : core.carrierStart = carrier.start

def rows {candidate : Id} {rowVariables : Nat}
    (layout : Layout candidate rowVariables) : List Row :=
  ProductionMemoryBatchCcsLinkRows.rows layout.core

abbrev Valid {candidate : Id} {rowVariables : Nat}
    (layout : Layout candidate rowVariables) := layout.core.Valid

def StateDigestPlaced
    {candidate : Id} {rowVariables : Nat}
    (layout : Layout candidate rowVariables) (assignment : Nat -> Nat)
    (digest : CanonicalDigest) : Prop :=
  ProductionMemoryBatchCcsLinkRows.StateDigestPlaced
    layout.core assignment digest

theorem ccsPublicPlaced
    {candidate : Id} {rowVariables : Nat}
    {fullShape : Phi81Relation.Shape}
    {contract : ProductNifsCodec.FullShapeContractFor rowVariables fullShape}
    {layout : Layout candidate rowVariables} {assignment : Nat -> Nat}
    {value : ProductionFieldNativeFullClaim.Value candidate fullShape}
    (placed : ProductionFullClaimCarrierLayoutFor.Placed
      contract layout.carrier assignment value) :
    ProductionMemoryBatchCcsLinkRows.CcsPublicPlaced
      layout.core assignment value := by
  intro index
  have source := placed.ccsPublic index
  simpa [ProductionMemoryBatchCcsLinkRows.Layout.ccsBitColumn,
    ProductionFullClaimCarrierLayoutFor.Layout.ccsPublicColumn,
    ProductionMemoryBatchCcsLinkRows.ccsPublicOffset,
    ProductionFullClaimCarrierLayoutFor.ccsPublicOffset,
    layout.carrierStart] using source

/-- Satisfying rows derive every CCS public coordinate at any generated
relation exponent. The conclusion is not a placement assumption. -/
theorem rows_imply_fullMatches
    {candidate : Id} {rowVariables : Nat}
    {fullShape : Phi81Relation.Shape}
    {contract : ProductNifsCodec.FullShapeContractFor rowVariables fullShape}
    {layout : Layout candidate rowVariables} {assignment : Nat -> Nat}
    {headers : FPrime.ChainHeaders Digest.Value}
    {value : ProductionFieldNativeFullClaim.Value candidate fullShape}
    {stateDigest : CanonicalDigest}
    (valid : Valid layout)
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (placed : ProductionFullClaimCarrierLayoutFor.Placed
      contract layout.carrier assignment value)
    (statePlaced : StateDigestPlaced layout assignment stateDigest)
    (memory : ProductionMemoryCheckedBatchRows.Result
      layout.core.batch.frame.memory assignment headers)
    (holds : Satisfies (rows layout) assignment) :
    ProductionMemoryBoundCcsPublic.FullMatches
      value.ccsPublic stateDigest memory.suffixBatch :=
  ProductionMemoryBatchCcsLinkRows.rows_imply_fullMatches_of_ccsPublicPlaced
    valid canonical one (ccsPublicPlaced placed) statePlaced memory holds

theorem rows_length_exact
    {candidate : Id} {rowVariables : Nat}
    {layout : Layout candidate rowVariables} (valid : Valid layout) :
    (rows layout).length = ProductionMemoryBatchCcsLinkRows.rowCount candidate :=
  ProductionMemoryBatchCcsLinkRows.rows_length_exact valid

end Nightstream.Implementation.NebulaV2.ProductionMemoryBatchCcsLinkRowsFor
