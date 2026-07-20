import Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.Authority.DelayedRawChildren
import Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.Authority.PackedYZcol
import Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.DelayedBlockLane

/-!
Production pending-output computation and packed source-authority composition.

Protocol: active concrete Phi81 NIFS.
Phases: prior `Pi_CCS -> Pi_RLC` output, then successor delayed BlockLane NC.
Constraint family: canonical pending state and sourcewise authority reduction;
this file emits no rows.

Assurance tier: model-level.

Owns: verifier computation of the pending old block point and full packed
`Pi_RLC` source aggregate; conversion of an authoritative raw-child
recomposition projection into the previous sourcewise packed-output bound;
and the exact remaining `Pi_RLC` mixing-collision branch.

Does not own: combined NC acceptance, producer-beta or residual-weight
sampling, cross-step state continuity, raw-child opening extraction, terminal
closure, Poseidon2, Ajtai hardness, Rust/R1CS refinement, costs, or row
removal.

Emits constraints: none.

Authority boundary: `outgoingPending` is computed from the actual certificate
message and verifier-derived challenges. It is not yet true merely because it
is computed. The result theorem additionally requires equality to the packed
projection of the actual raw-child recomposition; false source vectors remain
the named `PiRlcSidecar.MixingCollision` event.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `nifs.pi_rlc.delayed.pending.block` | retain the exact transcript-derived final BlockLane block point | computed | `outgoingPending` |
| `nifs.pi_rlc.delayed.pending.vector` | compute all 54 parent lanes from source claims and exact RingF challenges | computed | `outgoingPending` |
| `nifs.pi_rlc.delayed.parent` | successor raw-child projection equals the computed pending vector | checked upstream premise | `packedBound_or_mixingCollision_of_rawRecomposition` |
| `nifs.pi_rlc.delayed.sources` | recover every previous source claim or name one RingF mixing collision | derived/security boundary | `packedBound_or_mixingCollision_of_parentProjection` |
-/

namespace Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.Authority.DelayedProduction

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Concrete.Phi81Relation.EvaluationHomomorphism
open Nightstream.SuperNeo.Folding
open Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81
open Nightstream.SuperNeo.Folding.PiCCS.OutputClaims.EvaluationHomomorphism
open Nightstream.SuperNeo.Folding.PiCCS.OutputClaims.EvaluationHomomorphism.Authority
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Sources
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier

universe uState

variable
  {shape : SemanticShape}
  {State : Type uState}
  {publicRingColumns verifierRows : Nat}
  {publicFits : ringDegree * publicRingColumns <= shape.carrierWidth}

/-- Canonical pending value emitted by one physical active certificate. The
old block point and source aggregate are both verifier computations. -/
def outgoingPending
    (context : FixedActive.Context shape State publicRingColumns publicFits
      verifierRows)
    (certificate : FixedActive.Certificate context) :
    ProductionDelayedBlockLane where
  oldBlock := (derive context certificate).piCcs.ncPoint.block
  parentYZcol := DelayedPackedProjection.sourceAggregate
    certificate.piRlcChallenges
    (PackedYZcol.sourceClaims context certificate)

@[simp] theorem outgoingPending_oldBlock
    (context : FixedActive.Context shape State publicRingColumns publicFits
      verifierRows)
    (certificate : FixedActive.Certificate context) :
    (outgoingPending context certificate).oldBlock =
      (derive context certificate).piCcs.ncPoint.block := by
  rfl

@[simp] theorem outgoingPending_parentYZcol
    (context : FixedActive.Context shape State publicRingColumns publicFits
      verifierRows)
    (certificate : FixedActive.Certificate context) :
    (outgoingPending context certificate).parentYZcol =
      DelayedPackedProjection.sourceAggregate
        certificate.piRlcChallenges
        (PackedYZcol.sourceClaims context certificate) := by
  rfl

/-- Equality of the computed pending vector with the canonical private parent
projection yields the complete previous `yZcol` output binding, except for
the already named RingF source-mixing collision. -/
theorem packedBound_or_mixingCollision_of_parentProjection
    (context : FixedActive.Context shape State publicRingColumns publicFits
      verifierRows)
    (data : Data shape)
    (certificate : FixedActive.Certificate context)
    (parentProjection :
      (outgoingPending context certificate).parentYZcol =
        PackedBlockAction.packedYZcol context.covers
          (PackedYZcol.canonicalParentAssignment context data certificate)
          (outgoingPending context certificate).oldBlock) :
    Polynomial.Nc.BlockLane.Terminal.PackedYZcolBoundAtBlock
        context.covers data (outgoingPending context certificate).oldBlock
        certificate.piCcs.output ∨
      PiRlcSidecar.MixingCollision context.covers
        certificate.piRlcChallenges
        (InputAuthority.productAssignments data context.alignment)
        (outgoingPending context certificate).oldBlock
        (PackedYZcol.sourceClaims context certificate) := by
  have parentAssignmentMatches :
      PiRlcSidecar.ParentAssignmentMatches
        certificate.piRlcChallenges
        (InputAuthority.productAssignments data context.alignment)
        (PackedYZcol.canonicalParentAssignment context data certificate) := by
    exact (PackedYZcol.rawSourceFold_eq_canonicalParentAssignment
      context data certificate).symm
  rcases PiRlcSidecar.sourceBound_or_mixingCollision context.covers
      certificate.piRlcChallenges
      (InputAuthority.productAssignments data context.alignment)
      (outgoingPending context certificate).oldBlock
      (PackedYZcol.sourceClaims context certificate)
      (PackedYZcol.canonicalParentAssignment context data certificate)
      (outgoingPending context certificate).parentYZcol
      rfl parentProjection parentAssignmentMatches with bound | collision
  · exact Or.inl <|
      (PackedYZcol.sourceBound_iff_packedYZcolBound context.covers data
        context.alignment (outgoingPending context certificate).oldBlock
        certificate.piCcs.output).1 bound
  · exact Or.inr collision

/-- The successor check naturally speaks about a radix recomposition of its
actual raw running assignments. Once cross-step opening authority identifies
that recomposition with the previous canonical private parent, it discharges
the previous packed-output authority directly. -/
theorem packedBound_or_mixingCollision_of_rawRecomposition
    (context : FixedActive.Context shape State publicRingColumns publicFits
      verifierRows)
    (data : Data shape)
    (certificate : FixedActive.Certificate context)
    (rawChildren : Fin productionGlobalParams.k ->
      Phi81Relation.Assignment
        (RelationShape shape publicRingColumns publicFits))
    (recomposes :
      Phi81Relation.EvaluationHomomorphism.PiDEC.Raw.recomposeAssignment
          rawChildren =
        PackedYZcol.canonicalParentAssignment context data certificate)
    (delayedProjection :
      (outgoingPending context certificate).parentYZcol =
        PackedBlockAction.packedYZcol context.covers
          (Phi81Relation.EvaluationHomomorphism.PiDEC.Raw.recomposeAssignment
            rawChildren)
          (outgoingPending context certificate).oldBlock) :
    Polynomial.Nc.BlockLane.Terminal.PackedYZcolBoundAtBlock
        context.covers data (outgoingPending context certificate).oldBlock
        certificate.piCcs.output ∨
      PiRlcSidecar.MixingCollision context.covers
        certificate.piRlcChallenges
        (InputAuthority.productAssignments data context.alignment)
        (outgoingPending context certificate).oldBlock
        (PackedYZcol.sourceClaims context certificate) := by
  apply packedBound_or_mixingCollision_of_parentProjection
    context data certificate
  exact delayedProjection.trans <|
    congrArg
      (fun assignment =>
        PackedBlockAction.packedYZcol context.covers assignment
          (outgoingPending context certificate).oldBlock)
      recomposes

end Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.Authority.DelayedProduction
