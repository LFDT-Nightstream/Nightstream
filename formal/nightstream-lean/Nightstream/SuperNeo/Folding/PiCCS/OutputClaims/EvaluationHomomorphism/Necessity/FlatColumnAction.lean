import Nightstream.SuperNeo.Concrete.Phi81Relation.EvaluationHomomorphism.CarrierAction
import Nightstream.SuperNeo.Concrete.Phi81Relation.EvaluationHomomorphism.RingKAction
import Nightstream.SuperNeo.Folding.PiCCS.OutputClaims.EvaluationHomomorphism.BaseLinear

/-!
Counterexample to commuting the current flat-column `yZcol` projection with
the typed Phi81 `Pi_RLC` assignment action.

Protocol: SuperNeo `Pi_RLC` handoff of the Split-NC delayed sidecar.
Phase: one complete Phi81 assignment block at a verifier-owned column point.
Constraint family: semantic representation boundary only; this file emits no
rows.

Owns: a one-block, kernel-checked witness in which the blockwise `RingF`
action moves a coefficient from flat carrier column zero to column one. The
Boolean column point still selects column zero, so projecting after the action
is zero in lane one, while acting on the already projected ring value is one.

Does not own: the production witness-matrix definition of `y_zcol`, a claim
that the Rust verifier is unsound, the correct expanded-witness carrier,
transcript derivation, R1CS refinement, row removal, or constraint counts.

Emits constraints: no.

Authority boundary: this theorem refutes one proposed semantic bridge only:
`BaseLinear.yZcolEvaluation` over a flat assignment does not commute with
`CarrierAction.act` at an arbitrary flat column point. A correct `Pi_RLC`
authority theorem must model the production row-by-column witness image or
prove a stronger restriction on the column point; it may not assume this
false homomorphism.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `nifs.pi_rlc.verify.authority.combined_assignment.y_zcol.necessity.fixture` | one 54-coordinate block and a 64-point padded column cube | computed | `witnessShape`, `witnessDomain`, `witnessCovers` |
| `nifs.pi_rlc.verify.authority.combined_assignment.y_zcol.necessity.after_action` | project the acted assignment at Boolean column zero, lane one | derived | `actedProjection_laneOne_eq_zero` |
| `nifs.pi_rlc.verify.authority.combined_assignment.y_zcol.necessity.action_after` | act on the source projection, then read lane one | derived | `projectedAction_laneOne_eq_one` |
| `nifs.pi_rlc.verify.authority.combined_assignment.y_zcol.necessity.separation` | the two candidate handoff orders differ | counterexample | `flatColumnProjection_not_actionHom` |
-/

namespace Nightstream.SuperNeo.Folding.PiCCS.OutputClaims.EvaluationHomomorphism.Necessity.FlatColumnAction

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Concrete.Phi81Relation.EvaluationHomomorphism
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc

/-- One completed Phi81 block. No CCS or source batch is needed for this
representation counterexample. -/
def witnessShape : SemanticShape where
  rowVariables := 0
  logicalWidth := 1
  freshCount := 0
  runningCount := 1
  matrixCount := 0

/-- Six Boolean coordinates cover all 54 flat carrier coordinates. -/
def witnessDomain : FlatNcDomain where
  columnVariables := 6
  laneVariables := 6

/-- The padded column/lane domains contain the complete one-block carrier. -/
theorem witnessCovers : witnessDomain.Covers witnessShape := by
  simp [FlatNcDomain.Covers, FlatNcDomain.columnCount,
    FlatNcDomain.laneCount, SemanticShape.carrierWidth, witnessDomain,
    witnessShape, Phi81CarrierLayout.carrierWidth,
    Phi81ColumnLayout.blockCount, ringDegree]

/-- Boolean column zero, written directly as the all-zero little-endian
coordinate vector. -/
def columnZero : CubePoint K witnessDomain.columnVariables where
  coordinates := List.replicate witnessDomain.columnVariables K.zero
  dimension := by decide

/-- Assignment supported only at flat carrier column zero. -/
def unitAtColumnZero :
    Assignment (BaseLinear.relationShape witnessShape) :=
  fun column => if column.val = 0 then 1 else 0

/-- Multiplication by the coefficient basis `X`. -/
def shiftChallenge : RingF := ringFMonomial 1 1

/-- Output lane one, reached from coefficient zero by `X * 1`. -/
def laneOne : Fin ringDegree := ⟨1, by decide⟩

set_option maxRecDepth 10000 in
/-- Acting first moves the only nonzero coefficient to flat column one, which
the Boolean-zero column point does not select. -/
theorem actedProjection_laneOne_eq_zero :
    BaseLinear.yZcolEvaluation witnessCovers
        (CarrierAction.act shiftChallenge unitAtColumnZero) columnZero laneOne =
      K.zero := by
  decide

set_option maxRecDepth 10000 in
/-- Projecting first selects coefficient zero; the embedded `X` action then
moves that selected unit to lane one. -/
theorem projectedAction_laneOne_eq_one :
    ringKMul (RingKAction.embedChallenge shiftChallenge)
        (BaseLinear.yZcolEvaluation witnessCovers unitAtColumnZero columnZero)
        laneOne = K.one := by
  decide

/-- The current flat-assignment projection is not a `Pi_RLC` action
homomorphism at arbitrary verifier column points. -/
theorem flatColumnProjection_not_actionHom :
    BaseLinear.yZcolEvaluation witnessCovers
        (CarrierAction.act shiftChallenge unitAtColumnZero) columnZero ≠
      ringKMul (RingKAction.embedChallenge shiftChallenge)
        (BaseLinear.yZcolEvaluation witnessCovers unitAtColumnZero columnZero) := by
  intro equal
  have atLaneOne := congrFun equal laneOne
  rw [actedProjection_laneOne_eq_zero,
    projectedAction_laneOne_eq_one] at atLaneOne
  exact (by decide : K.zero ≠ K.one) atLaneOne

end Nightstream.SuperNeo.Folding.PiCCS.OutputClaims.EvaluationHomomorphism.Necessity.FlatColumnAction
