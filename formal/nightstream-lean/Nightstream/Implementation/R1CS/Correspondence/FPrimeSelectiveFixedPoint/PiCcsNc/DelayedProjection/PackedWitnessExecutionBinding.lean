import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.PackedWitnessOldBlockProjection
import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.PackedWitnessProduction

/-!
Typed refinement contract for the production raw-old-block execution audit.

Assurance tier: model-level contract. It becomes artifact-checked only when a
generated Rust artifact decodes its ordered child records into `Audit` and
proves `Refines`; it becomes Rust-conformant only when mutation/drift tests
also cover that exporter and the active verifier dataflow.

Owns: the exact data Rust must export while the incoming
`RunningInstance.witnesses` are still available; its refinement to the
literal native projection loop; and its association with that accumulator's
pending one-fold state.

Does not own: generated fixture records, list-to-`Fin` decoding, Ajtai
binding, recursive state digest authority, the distinct outgoing terminal
anchor, `y_ring`, costs, or row-removal permission.

Emits constraints: no.

Authority boundary: every active value is tied to one indexed full
`WitnessMat`. Child `CeClaim.y_zcol` values and digests do not occur. The ten
padding values must be zero. The pending parent and old point are checked
against `context.pending`; the current certificate's newly created outgoing
pending value does not occur in this contract.

| Stable stage path | Obligation | Authority class | Rust owner |
|---|---|---|---|
| `f_prime.pi_ccs_nc.raw_old_block.audit.children` | exactly fourteen ordered raw children | direct dataflow | `RunningInstance.witnesses` capture |
| `f_prime.pi_ccs_nc.raw_old_block.audit.active` | each 54-lane row equals the native full-`Z` projection at `oldBlock` | checked/refinement | execution-audit exporter |
| `f_prime.pi_ccs_nc.raw_old_block.audit.padding` | each of ten virtual lanes is zero | checked/computed | execution-audit exporter |
| `f_prime.pi_ccs_nc.raw_old_block.audit.pending` | audit point and recomposition equal the incoming running pending state | checked | active execution audit |
-/

namespace Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.PackedWitnessExecutionBinding

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Concrete.Phi81Relation.EvaluationHomomorphism
open Nightstream.SuperNeo.Folding
open Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81
open Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.Authority
open Nightstream.SuperNeo.Folding.PiCCS.OutputClaims.EvaluationHomomorphism
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open PackedWitness

namespace Native

export PackedWitnessOldBlockProjection
  (nativeProjectedChild nativeRadixRecomposition
    nativeProjectedChild_eq_packedYZcol
    nativeRadixRecomposition_eq_packedYZcol)

end Native

private abbrev productionShape := ProductionDomain.semanticShape
private abbrev productionDomain := PiCcsDomains.production.nc
private abbrev productionCovers := ProductionDomain.blockLaneDomain_covers

universe uState

/-- Exact Rust authority tag. No sidecar or digest authority variant exists. -/
inductive RawAssignmentAuthority where
  | runningWitnessMat
deriving DecidableEq, Repr

/-- One ordered child's proof-free semantic view after raw old-block
projection. Rust exports the active and padding arrays separately. -/
structure ChildAudit where
  authority : RawAssignmentAuthority
  activeLanes : Fin ringDegree -> K
  zeroPadding : Fin ProductionDomain.virtualLaneCount -> K

/-- Fixed-profile execution-audit view after exact list decoding. The raw
generated schema must separately prove that its list contains exactly fourteen
children in child-major order before constructing this typed value. -/
structure Audit where
  logicalColumns : Nat
  packedRows : Nat
  packedColumns : Nat
  oldBlock : CubePoint K productionDomain.blockVariables
  children : Fin productionGlobalParams.k -> ChildAudit

/-- Canonical semantic execution audit.  Unlike a prover-carried audit, this
value is computed from the authoritative ordered matrices and the verifier's
incoming pending point, so its refinement theorem has no premise.  The Rust
exporter is required to reproduce this function exactly. -/
def capture
    (witnesses : Fin productionGlobalParams.k -> Matrix productionShape)
    (pending : ProductionDelayedBlockLane) : Audit where
  logicalColumns := productionShape.logicalWidth
  packedRows := ringDegree
  packedColumns :=
    Phi81ColumnLayout.blockCount productionShape.logicalWidth
  oldBlock := pending.oldBlock
  children := fun child =>
    { authority := .runningWitnessMat
      activeLanes := Native.nativeProjectedChild
        (witnesses child) pending.oldBlock
      zeroPadding := fun _ => K.zero }

/-- Active child values in the exact form consumed by PiDEC recomposition. -/
def childProjection (audit : Audit)
    (child : Fin productionGlobalParams.k) : RingK :=
  audit.children child |>.activeLanes

/-- Ordered child-major radix recomposition exported by the terminal audit. -/
def radixRecomposition (audit : Audit) : RingK :=
  BaseLinear.combineEvaluations PiDEC.radixWeight (childProjection audit)

/-- Concrete conformance relation between one decoded execution audit and
the full raw matrices captured from `RunningInstance.witnesses`. -/
structure Refines
    (audit : Audit)
    (witnesses : Fin productionGlobalParams.k -> Matrix productionShape) :
    Prop where
  logicalColumns : audit.logicalColumns = productionShape.logicalWidth
  packedRows : audit.packedRows = ringDegree
  packedColumns : audit.packedColumns =
    Phi81ColumnLayout.blockCount productionShape.logicalWidth
  authority : forall child,
    (audit.children child).authority = .runningWitnessMat
  active : forall child lane,
    (audit.children child).activeLanes lane =
      Native.nativeProjectedChild (witnesses child) audit.oldBlock lane
  padding : forall child lane,
    (audit.children child).zeroPadding lane = K.zero

/-- The canonical decoder/capture function refines the literal native loop
definitionally.  In particular there is no `Refines` or raw-authority premise
at a production call site that uses `capture`. -/
theorem capture_refines
    (witnesses : Fin productionGlobalParams.k -> Matrix productionShape)
    (pending : ProductionDelayedBlockLane) :
    Refines (capture witnesses pending) witnesses := by
  exact {
    logicalColumns := rfl
    packedRows := rfl
    packedColumns := rfl
    authority := fun _ => rfl
    active := by
      intro child lane
      simp only [capture]
    padding := fun _ _ => rfl
  }

/-- The audit's ordered recomposition is exactly the native recomposition of
the same fourteen full `WitnessMat` values at the audit's old point. -/
theorem radixRecomposition_eq_native
    (audit : Audit)
    (witnesses : Fin productionGlobalParams.k -> Matrix productionShape)
    (refines : Refines audit witnesses) :
    radixRecomposition audit =
      Native.nativeRadixRecomposition witnesses audit.oldBlock := by
  unfold radixRecomposition Native.nativeRadixRecomposition childProjection
  apply congrArg (BaseLinear.combineEvaluations PiDEC.radixWeight)
  funext child lane
  exact refines.active child lane

/-- The canonical audit's child-major recomposition is the independent
packed projection of the same full matrices. -/
theorem capture_radixRecomposition_eq_packedYZcol
    (witnesses : Fin productionGlobalParams.k -> Matrix productionShape)
    (pending : ProductionDelayedBlockLane) :
    radixRecomposition (capture witnesses pending) =
      PackedBlockAction.packedYZcol productionCovers
        (PiDEC.Raw.recomposeAssignment fun child =>
          unpack (witnesses child)) pending.oldBlock := by
  exact (radixRecomposition_eq_native
      (capture witnesses pending) witnesses
      (capture_refines witnesses pending)).trans
    (Native.nativeRadixRecomposition_eq_packedYZcol
      witnesses pending.oldBlock)

/-- One accepted execution-audit handoff. Both pending fields are compared to
the incoming running pending state consumed by this step. -/
structure Accepted
    {State : Type uState}
    {publicRingColumns verifierRows : Nat}
    {publicFits : ringDegree * publicRingColumns <=
      productionShape.carrierWidth}
    (context : FixedActive.Context productionShape State publicRingColumns
      publicFits verifierRows)
    (certificate : FixedActive.Certificate context)
    (audit : Audit)
    (witnesses : Fin productionGlobalParams.k -> Matrix productionShape)
    (pending : ProductionDelayedBlockLane) :
    Prop where
  refines : Refines audit witnesses
  contextPending : context.pending = some pending
  oldBlock : audit.oldBlock = pending.oldBlock
  parent : pending.parentYZcol = radixRecomposition audit

/-- Accepted raw execution evidence derives the exact incoming pending-parent
projection from the actual full child matrices. -/
theorem accepted_implies_pendingProjection
    {State : Type uState}
    {publicRingColumns verifierRows : Nat}
    {publicFits : ringDegree * publicRingColumns <=
      productionShape.carrierWidth}
    (context : FixedActive.Context productionShape State publicRingColumns
      publicFits verifierRows)
    (certificate : FixedActive.Certificate context)
    (audit : Audit)
    (witnesses : Fin productionGlobalParams.k -> Matrix productionShape)
    (pending : ProductionDelayedBlockLane)
    (accepted : Accepted context certificate audit witnesses pending) :
    pending.parentYZcol =
      PackedBlockAction.packedYZcol productionCovers
        (PiDEC.Raw.recomposeAssignment fun child =>
          unpack (witnesses child))
        pending.oldBlock := by
  calc
    pending.parentYZcol =
        radixRecomposition audit := accepted.parent
    _ = Native.nativeRadixRecomposition witnesses audit.oldBlock :=
      radixRecomposition_eq_native audit witnesses accepted.refines
    _ = PackedBlockAction.packedYZcol productionCovers
        (PiDEC.Raw.recomposeAssignment fun child =>
          unpack (witnesses child)) audit.oldBlock :=
      Native.nativeRadixRecomposition_eq_packedYZcol witnesses audit.oldBlock
    _ = PackedBlockAction.packedYZcol productionCovers
        (PiDEC.Raw.recomposeAssignment fun child =>
          unpack (witnesses child))
        pending.oldBlock := by
      rw [accepted.oldBlock]

/-- Fixed decoded audit cardinalities expected from the Rust exporter. -/
theorem audit_cardinalities :
    productionGlobalParams.k = 14 /\
      ringDegree = 54 /\
      ProductionDomain.virtualLaneCount = 10 /\
      productionDomain.blockVariables = 19 /\
      Phi81ColumnLayout.blockCount productionShape.logicalWidth = 211797 := by
  decide

end Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.PackedWitnessExecutionBinding
