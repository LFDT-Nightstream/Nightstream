import NightstreamFPrime.Layout.Stage1.PilotPiCCSPiRLCPiDEC
import NightstreamFPrime.Layout.Stage1.RunningTransitionOwnership

/-!
Owns the cumulative Stage 1 layout through the running-instance transition.
The transition follows PiDEC directly and adds no copy or boundary row.
-/

namespace NightstreamFPrime.Layout.Stage1.PilotPiCCSPiRLCPiDECRunningTransition

open NightstreamFPrime.Circuit
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Layout.Stage1.RunningTransitionLayout
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint

variable {logicalWidth : Nat}
  {publicFits : ringDegree * publicRingColumns ≤
    Phi81CarrierLayout.carrierWidth logicalWidth}

def transitionOffset : Nat :=
  RunningTransitionInputs.phaseOffset

theorem transitionOffset_eq : transitionOffset = 27420586 := by
  rfl

def physicalRows
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    List R1CS.Row :=
  PilotPiCCSPiRLCPiDEC.physicalRows relation ++
    RunningTransitionLayout.physicalRows logicalWidth publicFits

def physicalRowCount
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) : Nat :=
  (physicalRows relation).length

def physicalColumnCount
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) : Nat :=
  max (PilotPiCCSPiRLCPiDEC.physicalColumnCount relation)
    (RunningTransitionLayout.physicalColumnCount logicalWidth publicFits)

def jointDomain
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) : Nat :=
  max (physicalRowCount relation) (physicalColumnCount relation)

def PhysicalHolds
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (env : Env) : Prop :=
  R1CS.RowsHold env (physicalRows relation)

theorem physicalHolds_iff
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (env : Env) :
    PhysicalHolds relation env ↔
      PilotPiCCSPiRLCPiDEC.PhysicalHolds relation env ∧
        RunningTransitionLayout.PhysicalHolds logicalWidth publicFits env := by
  exact R1CS.rowsHold_append env _ _

theorem physicalRowCount_eq
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    physicalRowCount relation = 27584200 := by
  unfold physicalRowCount physicalRows
  rw [List.length_append]
  change PilotPiCCSPiRLCPiDEC.physicalRowCount relation +
    RunningTransitionLayout.physicalRowCount logicalWidth publicFits = 27584200
  rw [PilotPiCCSPiRLCPiDEC.physicalRowCount_eq relation,
    RunningTransitionLayout.physicalRowCount_eq relation]

theorem physicalColumnCount_eq
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    physicalColumnCount relation = 27695988 := by
  unfold physicalColumnCount
  rw [PilotPiCCSPiRLCPiDEC.physicalColumnCount_eq relation,
    RunningTransitionLayout.physicalColumnCount_eq relation]
  norm_num

theorem jointDomain_eq
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    jointDomain relation = 27695988 := by
  unfold jointDomain
  rw [physicalRowCount_eq relation, physicalColumnCount_eq relation]
  norm_num

theorem jointDomain_le_twoPow28
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    jointDomain relation ≤ 2 ^ 28 := by
  rw [jointDomain_eq relation]
  norm_num

def cumulativePhysicalRows
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    List Nat :=
  PilotPiCCSPiRLCPiDEC.cumulativePhysicalRows relation ++
    [physicalRowCount relation]

def cumulativePhysicalColumns
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    List Nat :=
  PilotPiCCSPiRLCPiDEC.cumulativePhysicalColumns relation ++
    [physicalColumnCount relation]

def cumulativeJointDomains
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    List Nat :=
  PilotPiCCSPiRLCPiDEC.cumulativeJointDomains relation ++
    [jointDomain relation]

theorem cumulativeFootprints_eq
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    cumulativePhysicalRows relation =
        [27237625, 27260305, 27261277, 27261385, 27262897, 27262897,
          27584200] ∧
      cumulativePhysicalColumns relation =
        [27402496, 27420586, 27420586, 27420586, 27420586, 27420586,
          27695988] ∧
      cumulativeJointDomains relation =
        [27402496, 27420586, 27420586, 27420586, 27420586, 27420586,
          27695988] := by
  rcases PilotPiCCSPiRLCPiDEC.cumulativeFootprints_eq relation with
    ⟨_rowDeltas, _columnDeltas, rows, columns, joint⟩
  refine ⟨?_, ?_, ?_⟩
  · rw [cumulativePhysicalRows, rows, physicalRowCount_eq relation]
    rfl
  · rw [cumulativePhysicalColumns, columns, physicalColumnCount_eq relation]
    rfl
  · rw [cumulativeJointDomains, joint, jointDomain_eq relation]
    rfl

end NightstreamFPrime.Layout.Stage1.PilotPiCCSPiRLCPiDECRunningTransition
