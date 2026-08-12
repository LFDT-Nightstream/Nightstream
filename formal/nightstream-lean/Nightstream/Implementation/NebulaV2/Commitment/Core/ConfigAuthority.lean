import Nightstream.Implementation.NebulaV2.Commitment.Terminal.ProductCommitmentBridge

/-!
Contract: verifier-owned authority for the V2 product-commitment map.

The seed manifest fixes the three Ajtai keys. The lane layout fixes the three
whole-ring projections. The semantic commitment configuration is a definition
of those typed objects; it is not an independent prover-supplied value.

Does not own seeded-setup security, Module-SIS binding, terminal witness
placement, generated artifacts, Rust, or a deployed verifier.

Assurance tier: generated-artifact authority.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.NebulaV2.ProductCommitmentConfigAuthority

open Nightstream.Protocol.NebulaV2
open Nightstream.SuperNeo.Concrete.Phi81Relation

/-- The complete non-column authority for one product-commitment map. -/
structure Authority
    (manifest : SeedSchedule.Manifest)
    (fullShape operationsShape snapshotShape : Shape) where
  agreement : TerminalBundleOpeningRows.ShapeAgreement manifest fullShape
    operationsShape snapshotShape
  lanes : LaneLayout.Layout fullShape.carrierWidth
    operationsShape.carrierWidth snapshotShape.carrierWidth

/-- The only semantic product configuration selected by this authority. -/
def Authority.config
    {manifest : SeedSchedule.Manifest}
    {fullShape operationsShape snapshotShape : Shape}
    (authority : Authority manifest fullShape operationsShape snapshotShape) :
    ProductCommitmentAlgebra.Config fullShape operationsShape snapshotShape where
  lanes := authority.lanes
  fullKey := authority.agreement.fullKey
  operationsKey := authority.agreement.operationsKey
  snapshotKey := authority.agreement.snapshotKey

/-- Remove terminal column placement from a typed opening layout. -/
def ofTerminalLayout
    {manifest : SeedSchedule.Manifest}
    {fullShape operationsShape snapshotShape : Shape}
    (layout : TerminalBundleOpeningRows.Layout manifest fullShape
      operationsShape snapshotShape) :
    Authority manifest fullShape operationsShape snapshotShape where
  agreement := layout.agreement
  lanes := layout.lanes

/-- The terminal compiler and the verifier-owned authority construct the same
map when they use the same typed layout. -/
theorem config_ofTerminalLayout
    {manifest : SeedSchedule.Manifest}
    {fullShape operationsShape snapshotShape : Shape}
    (layout : TerminalBundleOpeningRows.Layout manifest fullShape
      operationsShape snapshotShape) :
    (ofTerminalLayout layout).config =
      TerminalProductCommitmentBridge.config layout := rfl

@[simp] theorem config_lanes
    {manifest : SeedSchedule.Manifest}
    {fullShape operationsShape snapshotShape : Shape}
    (authority : Authority manifest fullShape operationsShape snapshotShape) :
    authority.config.lanes = authority.lanes := rfl

@[simp] theorem config_fullKey
    {manifest : SeedSchedule.Manifest}
    {fullShape operationsShape snapshotShape : Shape}
    (authority : Authority manifest fullShape operationsShape snapshotShape) :
    authority.config.fullKey = authority.agreement.fullKey := rfl

@[simp] theorem config_operationsKey
    {manifest : SeedSchedule.Manifest}
    {fullShape operationsShape snapshotShape : Shape}
    (authority : Authority manifest fullShape operationsShape snapshotShape) :
    authority.config.operationsKey = authority.agreement.operationsKey := rfl

@[simp] theorem config_snapshotKey
    {manifest : SeedSchedule.Manifest}
    {fullShape operationsShape snapshotShape : Shape}
    (authority : Authority manifest fullShape operationsShape snapshotShape) :
    authority.config.snapshotKey = authority.agreement.snapshotKey := rfl

end Nightstream.Implementation.NebulaV2.ProductCommitmentConfigAuthority
