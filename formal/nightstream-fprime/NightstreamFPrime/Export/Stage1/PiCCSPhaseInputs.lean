import NightstreamFPrime.Export.Stage1.PiCCSEndpointCompleteness
import NightstreamFPrime.Export.Stage1.PiCCSTranscriptDirectSemantics

/-!
Owns affine inputs and initial values for the four semantic C traces.
The existing phase shapes supply affine actions. Each noninitial trace reads
its predecessor's actual endpoint in the same completed assignment.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.PiCCSPhaseInputs

open NightstreamFPrime.Circuit
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Lifecycle.PiCCS.v1_1
open NightstreamFPrime.Gadgets.Poseidon2
open NightstreamFPrime.Gadgets.Poseidon2.Duplex
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.Stage1
open NightstreamFPrime.Layout.ProductionRelation
open PiCCSInvocations
open PiCCSTranscriptDirectSemantics
open PiCCSTranscriptEndpointPlan
open PiCCSPoseidonPreservation
open PerApplicationAssignmentTransportExecution

variable {logicalWidth : Nat}
  {publicFits : ringDegree * publicRingColumns ≤ Phi81CarrierLayout.carrierWidth logicalWidth}

private theorem challenge_initial_column (lane : Fin 8) :
    (challengeInterface Data.logicalWidth Data.publicFits).initialState challengeWitnessStart lane =
      Expr.var (endpointColumn statementFamily lane) := by
  exact (congrFun (challengeInitialState_eq_statementFinalState Data.logicalWidth Data.publicFits) lane).trans
    (statementFinalState_endpoint_of_shape Data.logicalWidth Data.publicFits lane)

private theorem round_initial_column (lane : Fin 8) :
    (roundInterface Data.logicalWidth Data.publicFits).initialState roundWitnessStart lane =
      Expr.var (endpointColumn challengeFamily lane) := by
  exact (congrFun (roundInitialState_eq_challengeFinalState_of_shape Data.logicalWidth Data.publicFits) lane).trans
    (challengeFinalState_endpoint_of_shape Data.logicalWidth Data.publicFits lane)

private theorem output_initial_column (lane : Fin 8) :
    (outputInterface Data.logicalWidth Data.publicFits).initialState outputWitnessStart lane =
      Expr.var (endpointColumn roundFamily lane) := by
  exact (congrFun (outputInitialState_eq_roundFinalState_of_shape Data.logicalWidth Data.publicFits) lane).trans
    (roundFinalState_endpoint_of_shape Data.logicalWidth Data.publicFits lane)

/-- The statement initial state and exact owned statement actions are affine. -/
theorem statement_affine (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    Layout.Poseidon2.StateAffine Hash.zeroE ∧
      Invocations.ActionsInvocationInputsAffine PiCCSActionPayloadBlock.statementActions := by
  refine ⟨Layout.Poseidon2.zeroE_affine, ?_⟩
  rw [← statementActions_eq_of_shape logicalWidth publicFits]
  exact Invocations.actionsInvocationInputsAffine_of_actionsAffine _
    (Layout.PiCCS.v1_1.Leaves.StatementAbsorption.actions_affine
      (statementInterface logicalWidth publicFits) statementWitnessStart
      ((inputShapes logicalWidth publicFits relation).statementAbsorption statementWitnessStart))

/-- The challenge initial state is its predecessor endpoint; its action
syntax is the same affine syntax owned by the actual relation's phase. -/
theorem challenge_affine (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    Layout.Poseidon2.StateAffine
      ((challengeInterface Data.logicalWidth Data.publicFits).initialState challengeWitnessStart) ∧
      Invocations.ActionsInvocationInputsAffine PiCCSActionPayloadBlock.challengeActions := by
  constructor
  · intro lane
    rw [challenge_initial_column]
    exact R1CS.isAffine_var _
  · rw [← challengeActions_eq_of_shape logicalWidth publicFits]
    exact Invocations.actionsInvocationInputsAffine_of_actionsAffine _
      (Layout.PiCCS.v1_1.Leaves.ChallengeDerivation.actions_affine
        (challengeInterface logicalWidth publicFits) challengeWitnessStart
        ((inputShapes logicalWidth publicFits relation).challengeDerivation challengeWitnessStart))

/-- The round initial state and exact semantic round actions are affine. -/
theorem round_affine (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    Layout.Poseidon2.StateAffine
      ((roundInterface Data.logicalWidth Data.publicFits).initialState roundWitnessStart) ∧
      Invocations.ActionsInvocationInputsAffine PiCCSActionPayloadBlock.roundActions := by
  constructor
  · intro lane
    rw [round_initial_column]
    exact R1CS.isAffine_var _
  · rw [← roundActions_eq_of_shape logicalWidth publicFits]
    exact Invocations.actionsInvocationInputsAffine_of_actionsAffine _
      (Layout.PiCCS.v1_1.Leaves.RoundTranscript.actions_affine
        (roundInterface logicalWidth publicFits) roundWitnessStart
        ((inputShapes logicalWidth publicFits relation).roundTranscript roundWitnessStart))

/-- The output initial state and exact semantic output actions are affine. -/
theorem output_affine (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    Layout.Poseidon2.StateAffine
      ((outputInterface Data.logicalWidth Data.publicFits).initialState outputWitnessStart) ∧
      Invocations.ActionsInvocationInputsAffine PiCCSActionPayloadBlock.outputActions := by
  constructor
  · intro lane
    rw [output_initial_column]
    exact R1CS.isAffine_var _
  · rw [← outputActions_eq_of_shape logicalWidth publicFits]
    exact Invocations.actionsInvocationInputsAffine_of_actionsAffine _
      (Layout.PiCCS.v1_1.Leaves.OutputBinding.actions_affine
        (outputInterface logicalWidth publicFits) outputWitnessStart
        ((inputShapes logicalWidth publicFits relation).outputBinding outputWitnessStart))

section InitialValues

variable (application : Lifecycle.Stage1.Application.Program)
  (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
  (target : Env)
  (suffix : Fin (PerApplicationPackage.addedPrivateColumnCount application) → F)
  (physical : Layout.PiCCS.v1_1.PhysicalHolds relation
    (PiCCSInputs.interface logicalWidth publicFits) PiCCSInputs.phaseOffset (Spartan.pullback target))

local notation "raw" => canonicalRawValues application
  (PerApplicationSourceAssignment.ofCompleted application target suffix)
local notation "geometry" => PerApplicationCanonicalEncodes.poseidonGeometry application

/-- The first C slice uses the declared zero state. -/
theorem statement_initial :
    List.ofFn (Layer.evalState (Spartan.pullback target) Hash.zeroE) =
      PoseidonActionSemantics.sliceInitial Spec.Poseidon2.zeroState
        (valueState geometry raw.assignment) statementOffset statementOffsetBound := by
  simp only [PoseidonActionSemantics.sliceInitial, statementOffset, dif_pos rfl]
  rfl

include relation physical in
/-- The challenge slice starts at the actual statement endpoint. -/
theorem challenge_initial :
    List.ofFn (Layer.evalState (Spartan.pullback target)
      ((challengeInterface Data.logicalWidth Data.publicFits).initialState challengeWitnessStart)) =
      PoseidonActionSemantics.sliceInitial Spec.Poseidon2.zeroState
        (valueState geometry raw.assignment) challengeOffset challengeOffsetBound := by
  rw [PoseidonActionSemantics.sliceInitial, dif_neg (by decide : challengeOffset ≠ 0)]
  change List.ofFn _ = valueState geometry raw.assignment (endpointInvocation statementFamily)
  apply congrArg List.ofFn
  funext lane
  change ((challengeInterface Data.logicalWidth Data.publicFits).initialState challengeWitnessStart lane).eval
      (Spartan.pullback target) = outputValue geometry raw.assignment (endpointInvocation statementFamily) lane
  rw [challenge_initial_column, Expr.eval_var]
  exact (PiCCSEndpointCompleteness.endpointValue_of_completed application relation target suffix physical
    statementFamily lane).symm

include relation physical in
/-- The round slice starts at the actual challenge endpoint. -/
theorem round_initial :
    List.ofFn (Layer.evalState (Spartan.pullback target)
      ((roundInterface Data.logicalWidth Data.publicFits).initialState roundWitnessStart)) =
      PoseidonActionSemantics.sliceInitial Spec.Poseidon2.zeroState
        (valueState geometry raw.assignment) roundOffset roundOffsetBound := by
  rw [PoseidonActionSemantics.sliceInitial, dif_neg (by decide : roundOffset ≠ 0)]
  change List.ofFn _ = valueState geometry raw.assignment (endpointInvocation challengeFamily)
  apply congrArg List.ofFn
  funext lane
  change ((roundInterface Data.logicalWidth Data.publicFits).initialState roundWitnessStart lane).eval
      (Spartan.pullback target) = outputValue geometry raw.assignment (endpointInvocation challengeFamily) lane
  rw [round_initial_column, Expr.eval_var]
  exact (PiCCSEndpointCompleteness.endpointValue_of_completed application relation target suffix physical
    challengeFamily lane).symm

include relation physical in
/-- The output slice starts at the actual final round endpoint. -/
theorem output_initial :
    List.ofFn (Layer.evalState (Spartan.pullback target)
      ((outputInterface Data.logicalWidth Data.publicFits).initialState outputWitnessStart)) =
      PoseidonActionSemantics.sliceInitial Spec.Poseidon2.zeroState
        (valueState geometry raw.assignment) outputOffset outputOffsetBound := by
  rw [PoseidonActionSemantics.sliceInitial, dif_neg (by decide : outputOffset ≠ 0)]
  change List.ofFn _ = valueState geometry raw.assignment (endpointInvocation roundFamily)
  apply congrArg List.ofFn
  funext lane
  change ((outputInterface Data.logicalWidth Data.publicFits).initialState outputWitnessStart lane).eval
      (Spartan.pullback target) = outputValue geometry raw.assignment (endpointInvocation roundFamily) lane
  rw [output_initial_column, Expr.eval_var]
  exact (PiCCSEndpointCompleteness.endpointValue_of_completed application relation target suffix physical
    roundFamily lane).symm

end InitialValues

end NightstreamFPrime.Export.Stage1.PiCCSPhaseInputs
