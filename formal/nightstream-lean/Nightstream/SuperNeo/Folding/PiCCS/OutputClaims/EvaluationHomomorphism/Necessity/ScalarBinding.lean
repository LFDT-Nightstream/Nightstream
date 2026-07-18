import Nightstream.SuperNeo.Folding.PiCCS.OutputClaims.EvaluationHomomorphism.Authority.DelayedPackedProjection
import Nightstream.SuperNeo.Folding.PiCCS.OutputClaims.EvaluationHomomorphism.Authority.DelayedPackedProjection.PiDecRecomposition

/-!
Necessity of binding the second side of the packed projection scalar.

Protocol: SuperNeo `Pi_CCS -> Pi_RLC` packed-output authority.
Phase: one-point comparison with the canonical combined-parent projection.
Constraint family: parent-side scalar equality; this file emits no rows.

Assurance tier: model-level.

Owns: a closed generic one-source counterexample. A forged packed source claim passes
the raw source-scalar equality, while the authoritative source assignment and
canonical combined parent are zero. The witness also rules out the named
mixing-collision and bad-root alternatives.

Does not own: a production proof backend, commitment opening, transcript
timing, collision probability, Rust/R1CS refinement, costs, or row removal.

Emits constraints: no.

Authority boundary: the claimed scalar is prover-controlled until an
independently justified equality connects it to the derived parent projection.
That equality may come from direct recomputation or a sound opening. Ordinary
kernel reduction checks the two closed scalar evaluations; no external solver
or generated fixture participates. This is generic semantic necessity, not yet
fixed-profile inclusion-minimality: it does not establish the 15-source
transcript or hold every other production obligation fixed.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `nifs.pi_rlc.verify.authority.packed_y_zcol.necessity.fixture` | one zero assignment and one forged unit sidecar | computed | `witnessShape`, `forgedClaim` |
| `nifs.pi_rlc.verify.authority.packed_y_zcol.necessity.source_only` | the forged source aggregate projects to the claimed unit scalar | weakened semantic check | `forgedSourceProjection_matches` |
| `nifs.pi_rlc.verify.authority.packed_y_zcol.necessity.parent` | the canonical combined-parent projection is zero | derived | `canonicalParent_projection_eq_zero` |
| `nifs.pi_rlc.verify.authority.packed_y_zcol.necessity.missing_match` | neither direct-parent nor child-recomposition scalar equality holds | generic semantic countermodel | `sourceOnlyProjectionCheck_admits_forgery` |
| `nifs.pi_rlc.verify.authority.packed_y_zcol.necessity.events` | the witness is neither a mixing collision nor a projection bad root | derived | `mixingCollision_is_false`, `projectionBadRoot_is_false` |
-/

namespace Nightstream.SuperNeo.Folding.PiCCS.OutputClaims.EvaluationHomomorphism.Necessity.ScalarBinding

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation.EvaluationHomomorphism
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.OutputClaims.EvaluationHomomorphism
open Nightstream.SuperNeo.Folding.PiCCS.OutputClaims.EvaluationHomomorphism.Authority
open Authority.DelayedPackedProjection
open Authority.DelayedPackedProjection.PiDecRecomposition

/-- One completed Phi81 carrier block and one semantic source. -/
def witnessShape : SemanticShape where
  rowVariables := 0
  logicalWidth := 1
  freshCount := 0
  runningCount := 1
  matrixCount := 0

/-- The block cube has one live block; six lane bits cover all 54 lanes. -/
def witnessDomain : BlockNcDomain where
  blockVariables := 0
  laneVariables := 6

theorem witnessCovers : witnessDomain.Covers witnessShape := by
  simp [BlockNcDomain.Covers, BlockNcDomain.blockCount,
    BlockNcDomain.laneCount, SemanticShape.carrierWidth, witnessDomain,
    witnessShape, Phi81CarrierLayout.carrierWidth,
    Phi81ColumnLayout.blockCount, ringDegree]

/-- The unique point of the zero-dimensional block cube. -/
def witnessPoint : CubePoint K witnessDomain.blockVariables where
  coordinates := []
  dimension := rfl

/-- The sole authoritative source assignment is identically zero. -/
def zeroAssignment : PackedBlockAction.SemanticAssignment witnessShape :=
  BaseLinear.Raw.assignmentZero

def witnessAssignments : Fin 1 ->
    PackedBlockAction.SemanticAssignment witnessShape :=
  fun _ => zeroAssignment

/-- The sole PiRLC challenge is the multiplicative ring unit. -/
def witnessChallenges : Fin 1 -> RingF := fun _ => ringFOne

/-- A forged packed sidecar with unit constant coefficient. -/
def forgedClaim : RingK :=
  fun lane => if lane.val = 0 then K.one else K.zero

def forgedClaims : Fin 1 -> RingK := fun _ => forgedClaim

/-- The direct canonical-parent value to which a compact proof may bind the
claimed scalar. No PiDEC child is needed to define it. -/
def canonicalParentClaim : RingK :=
  PackedBlockAction.packedYZcol witnessCovers
    (PiRLCFinite.Raw.combineAssignments witnessChallenges witnessAssignments)
    witnessPoint

/-- Fixture specialization of direct canonical-parent scalar binding. -/
def ParentProjectionMatches
    (claimedProjection producerBeta : K) : Prop :=
  PairRightScalarMatches canonicalParentClaim claimedProjection producerBeta

/-- Optional PiDEC route specialized to zero child sidecars. -/
def zeroChildClaims : Fin productionGlobalParams.k -> RingK :=
  fun _ => ringKZero

def sourceZero : Fin 1 := ⟨0, by decide⟩

def laneZero : Fin ringDegree := ⟨0, by decide⟩

set_option maxRecDepth 10000 in
/-- The forged source fold evaluates to one at beta zero. -/
theorem sourceAggregate_projection_eq_one :
    projectedValue
        (sourceAggregate witnessChallenges forgedClaims) K.zero =
      K.one := by
  decide

set_option maxRecDepth 10000 in
/-- The canonical source-derived aggregate evaluates to zero. -/
theorem canonicalSourceAggregate_projection_eq_zero :
    projectedValue
        (PiRLCFinite.combineEvaluation witnessChallenges fun source =>
          PackedBlockAction.packedYZcol witnessCovers
            (witnessAssignments source) witnessPoint)
        K.zero =
      K.zero := by
  have allZero :
      (fun source : Fin 1 =>
          PackedBlockAction.packedYZcol witnessCovers
            (witnessAssignments source) witnessPoint) =
        (fun _ => ringKZero) := by
    funext source
    simpa [witnessAssignments, zeroAssignment] using
      (PackedBlockAction.Linear.packedYZcol_zero witnessCovers witnessPoint)
  rw [allZero]
  decide

/-- The canonical combined-parent projection is the same zero scalar. -/
theorem canonicalParent_projection_eq_zero :
    projectedValue canonicalParentClaim K.zero = K.zero := by
  unfold canonicalParentClaim
  rw [PackedBlockAction.Finite.packedYZcol_combine]
  exact canonicalSourceAggregate_projection_eq_zero

set_option maxRecDepth 10000 in
/-- Recombining fourteen zero child sidecars also projects to zero. -/
theorem zeroChildren_projection_eq_zero :
    projectedValue (recomposeClaims zeroChildClaims) K.zero = K.zero := by
  decide

/-- The raw source-side scalar equality accepts the forged sidecar. -/
theorem forgedSourceProjection_matches :
    SourceProjectionMatches witnessChallenges forgedClaims K.one K.zero := by
  exact sourceAggregate_projection_eq_one

/-- The forged source sidecar is not the packed projection of the zero
authoritative assignment. -/
theorem sourceBinding_is_false :
    ¬ PiRlcSidecar.SourceBound witnessCovers witnessAssignments
      witnessPoint forgedClaims := by
  intro bound
  have zeroProjection :
      PackedBlockAction.packedYZcol witnessCovers
          (witnessAssignments sourceZero) witnessPoint =
        ringKZero := by
    simpa [witnessAssignments, zeroAssignment] using
      (PackedBlockAction.Linear.packedYZcol_zero witnessCovers witnessPoint)
  have impossible : K.one = K.zero := by
    calc
      K.one = forgedClaims sourceZero laneZero := by decide
      _ = PackedBlockAction.packedYZcol witnessCovers
          (witnessAssignments sourceZero) witnessPoint laneZero :=
        congrFun (bound sourceZero) laneZero
      _ = ringKZero laneZero := congrFun zeroProjection laneZero
      _ = K.zero := rfl
  exact (by decide : K.one ≠ K.zero) impossible

/-- The direct canonical-parent scalar binding rejects the claimed unit. -/
theorem parentProjectionMatches_is_false :
    ¬ ParentProjectionMatches K.one K.zero := by
  intro matchProof
  unfold ParentProjectionMatches PairRightScalarMatches at matchProof
  rw [canonicalParent_projection_eq_zero] at matchProof
  exact (by decide : K.one ≠ K.zero) matchProof

/-- The PiDEC child-recomposition specialization rejects the same unit. -/
theorem childRecompositionMatches_is_false :
    ¬ ScalarRecompositionMatches zeroChildClaims K.one K.zero := by
  intro matchProof
  unfold ScalarRecompositionMatches PairRightScalarMatches at matchProof
  rw [zeroChildren_projection_eq_zero] at matchProof
  exact (by decide : K.one ≠ K.zero) matchProof

/-- This is not the named PiRLC mixing event: the forged and canonical
aggregates remain different at beta zero. -/
theorem mixingCollision_is_false :
    ¬ PiRlcSidecar.MixingCollision witnessCovers witnessChallenges
      witnessAssignments witnessPoint forgedClaims := by
  intro collision
  have projected := congrArg
    (fun value : RingK => projectedValue value K.zero) collision.2
  change projectedValue (sourceAggregate witnessChallenges forgedClaims)
      K.zero =
    projectedValue
      (PiRLCFinite.combineEvaluation witnessChallenges fun source =>
        PackedBlockAction.packedYZcol witnessCovers
          (witnessAssignments source) witnessPoint)
      K.zero at projected
  rw [sourceAggregate_projection_eq_one,
    canonicalSourceAggregate_projection_eq_zero] at projected
  exact (by decide : K.one ≠ K.zero) projected

/-- Nor is beta zero a bad root: the two projected scalars are unequal. -/
theorem projectionBadRoot_is_false :
    ¬ ProjectionCheck.BadRoot projectionOps
      (pairIdentity (sourceAggregate witnessChallenges forgedClaims)
        canonicalParentClaim K.zero) := by
  intro badRoot
  have collision := badRoot.collision
  change projectedValue (sourceAggregate witnessChallenges forgedClaims)
      K.zero = projectedValue canonicalParentClaim K.zero at collision
  rw [sourceAggregate_projection_eq_one,
    canonicalParent_projection_eq_zero] at collision
  exact (by decide : K.one ≠ K.zero) collision

/-- Generic semantic countermodel for omitting the right scalar. The
weakened source-only check accepts an invalid packed source vector, and neither
named algebraic failure event explains the acceptance. Both the direct-parent
and optional `PiDEC`-child formulations reject exactly the missing equality.

This theorem does not yet claim fixed-profile inclusion-minimality. -/
theorem sourceOnlyProjectionCheck_admits_forgery :
    SourceProjectionMatches witnessChallenges forgedClaims K.one K.zero ∧
      ¬ PiRlcSidecar.SourceBound witnessCovers witnessAssignments
        witnessPoint forgedClaims ∧
      ¬ ParentProjectionMatches K.one K.zero ∧
      ¬ ScalarRecompositionMatches zeroChildClaims K.one K.zero ∧
      ¬ PiRlcSidecar.MixingCollision witnessCovers witnessChallenges
        witnessAssignments witnessPoint forgedClaims ∧
      ¬ ProjectionCheck.BadRoot projectionOps
        (pairIdentity (sourceAggregate witnessChallenges forgedClaims)
          canonicalParentClaim K.zero) := by
  exact ⟨forgedSourceProjection_matches, sourceBinding_is_false,
    parentProjectionMatches_is_false, childRecompositionMatches_is_false,
    mixingCollision_is_false, projectionBadRoot_is_false⟩

end Nightstream.SuperNeo.Folding.PiCCS.OutputClaims.EvaluationHomomorphism.Necessity.ScalarBinding
