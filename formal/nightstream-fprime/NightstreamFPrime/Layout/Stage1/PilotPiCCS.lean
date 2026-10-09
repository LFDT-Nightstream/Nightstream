import NightstreamFPrime.Layout.PilotProduction
import NightstreamFPrime.Layout.PiCCS.v1_2.Preservation
import NightstreamFPrime.Layout.Stage1.PiCCSInputs

/-!
Obligation: Assemble the closed pilot, the concrete parent-owned PiCCS proof
inputs, and the PiCCS physical row packet into the current Stage 1 prefix.

The running instance and fresh public input reuse pilot columns. Four public
verifier-context words precede the new 10,872-column proof-input interval,
which owns the fresh commitment, 28 SumCheck messages, and separate output
`Eval_K`/`Eval_A` families. PiCCS local columns start immediately after that
interval. No boundary-copy row is present.
-/

namespace NightstreamFPrime.Layout.Stage1.PilotPiCCS

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Lifecycle.PiCCS.v1_2
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint

variable {logicalWidth : Nat}
  {publicFits : ringDegree * publicRingColumns ≤
    Phi81CarrierLayout.carrierWidth logicalWidth}

/-- The one concrete symbolic PiCCS interface of the current prefix. -/
def interface : Formal.Interface logicalWidth 8 publicFits :=
  PiCCSInputs.interface logicalWidth publicFits

/-- PiCCS starts after the completed pilot and all parent-owned proof inputs. -/
def piCcsOffset : Nat := PiCCSInputs.phaseOffset

theorem piCcsOffset_eq : piCcsOffset = 5156956 := by
  exact PiCCSInputs.phaseOffset_eq

/-- Exact physical row order of the current Stage 1 prefix. -/
def physicalRows
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    List R1CS.Row :=
  Pilot.physicalRows PilotProduction.interface PilotProduction.witnessOffset ++
    NightstreamFPrime.Layout.PiCCS.v1_2.physicalRows relation
      (interface (publicFits := publicFits)) piCcsOffset

def physicalRowCount
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) : Nat :=
  (physicalRows relation).length

def physicalColumnCount
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) : Nat :=
  max
    (Pilot.physicalColumnCount PilotProduction.interface
      PilotProduction.witnessOffset)
    (NightstreamFPrime.Layout.PiCCS.v1_2.physicalColumnCount relation
      (interface (publicFits := publicFits)) piCcsOffset)

def jointDomain
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) : Nat :=
  max (physicalRowCount relation) (physicalColumnCount relation)

def PhysicalHolds
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (env : Env) : Prop :=
  R1CS.RowsHold env (physicalRows relation)

/-- Stage 1 row endpoints after each PiCCS leaf. -/
def cumulativePhysicalRows
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    List Nat :=
  (NightstreamFPrime.Layout.PiCCS.v1_2.cumulativePhysicalRows relation
    (interface (publicFits := publicFits)) piCcsOffset).map
      (PilotProduction.physicalRowCountValue + ·)

/-- Stage 1 private-column endpoints after each PiCCS leaf. -/
def cumulativePhysicalColumns
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    List Nat :=
  (NightstreamFPrime.Layout.PiCCS.v1_2.cumulativePhysicalColumns relation
    (interface (publicFits := publicFits)) piCcsOffset).map
      (piCcsOffset + ·)

/-- Stage 1 joint-domain endpoints after each PiCCS leaf. -/
def cumulativeJointDomains
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    List Nat :=
  List.zipWith max (cumulativePhysicalRows relation)
    (cumulativePhysicalColumns relation)

theorem physicalHolds_iff
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (env : Env) :
    PhysicalHolds relation env ↔
      Pilot.PhysicalHolds PilotProduction.interface
          PilotProduction.witnessOffset env ∧
        NightstreamFPrime.Layout.PiCCS.v1_2.PhysicalHolds relation
          (interface (publicFits := publicFits)) piCcsOffset env := by
  exact R1CS.rowsHold_append env _ _

theorem physicalRowCount_eq
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    physicalRowCount relation = 6352059 := by
  unfold physicalRowCount physicalRows
  rw [List.length_append]
  change Pilot.physicalRowCount PilotProduction.interface
      PilotProduction.witnessOffset +
    NightstreamFPrime.Layout.PiCCS.v1_2.physicalRowCount relation
      (interface (publicFits := publicFits)) piCcsOffset = 6352059
  rw [PilotProduction.physicalRowCount_eq,
    NightstreamFPrime.Layout.PiCCS.v1_2.ProductionInputs.physicalRowCount_eq
      relation (interface (publicFits := publicFits)) piCcsOffset
      (PiCCSInputs.externalInputsLinear logicalWidth publicFits)]

theorem physicalColumnCount_eq
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    physicalColumnCount relation = 6418389 := by
  unfold physicalColumnCount
  rw [PilotProduction.physicalColumnCount_eq,
    NightstreamFPrime.Layout.PiCCS.v1_2.ProductionInputs.physicalColumnCount_eq
      relation (interface (publicFits := publicFits)) piCcsOffset
      (PiCCSInputs.externalInputsLinear logicalWidth publicFits),
    piCcsOffset_eq]
  norm_num

theorem jointDomain_eq
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    jointDomain relation = 6418389 := by
  unfold jointDomain
  rw [physicalRowCount_eq relation, physicalColumnCount_eq relation]
  norm_num

theorem jointDomain_le_twoPow28
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    jointDomain relation ≤ 2 ^ 28 := by
  rw [jointDomain_eq relation]
  norm_num

/-- One transported ledger states every PiCCS delta and every Stage 1
endpoint after the completed pilot and parent-owned proof inputs. -/
theorem cumulativeFootprints_eq
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    NightstreamFPrime.Layout.PiCCS.v1_2.physicalRowDeltas relation
        (interface (publicFits := publicFits)) piCcsOffset =
        [4712, 140288, 95352, 153440, 12957, 728, 3364, 11140,
          23, 800, 3593, 839536] ∧
      NightstreamFPrime.Layout.PiCCS.v1_2.physicalColumnDeltas relation
        (interface (publicFits := publicFits)) piCcsOffset =
        [270, 140288, 95352, 153440, 12957, 672, 3364, 11140,
          23, 800, 3591, 839536] ∧
      cumulativePhysicalRows relation =
        [5090838, 5231126, 5326478, 5479918, 5492875, 5493603,
          5496967, 5508107, 5508130, 5508930, 5512523, 6352059] ∧
      cumulativePhysicalColumns relation =
        [5157226, 5297514, 5392866, 5546306, 5559263, 5559935,
          5563299, 5574439, 5574462, 5575262, 5578853, 6418389] ∧
      cumulativeJointDomains relation =
        [5157226, 5297514, 5392866, 5546306, 5559263, 5559935,
          5563299, 5574439, 5574462, 5575262, 5578853, 6418389] := by
  let inputs :=
    NightstreamFPrime.Layout.PiCCS.v1_2.ProductionInputs.inputShapes relation
      (interface (publicFits := publicFits)) piCcsOffset
      (PiCCSInputs.externalInputsLinear logicalWidth publicFits)
  have rows :=
    NightstreamFPrime.Layout.PiCCS.v1_2.physicalRowDeltas_eq_production
      relation (interface (publicFits := publicFits)) piCcsOffset inputs
  have columns :=
    NightstreamFPrime.Layout.PiCCS.v1_2.physicalColumnDeltas_eq_production
      relation (interface (publicFits := publicFits)) piCcsOffset inputs
  have cumulative :=
    NightstreamFPrime.Layout.PiCCS.v1_2.cumulativeFootprints_eq_production
      relation (interface (publicFits := publicFits)) piCcsOffset inputs
  rcases cumulative with ⟨cumulativeRows, cumulativeColumns, cumulativeJoint⟩
  refine ⟨rows, columns, ?_, ?_, ?_⟩
  · rw [cumulativePhysicalRows, cumulativeRows,
      PilotProduction.physicalRowCountValue_eq]
    norm_num
  · rw [cumulativePhysicalColumns, cumulativeColumns, piCcsOffset_eq]
    norm_num
  · rw [cumulativeJointDomains, cumulativePhysicalRows,
      cumulativePhysicalColumns, cumulativeRows, cumulativeColumns,
      PilotProduction.physicalRowCountValue_eq, piCcsOffset_eq]
    norm_num

end NightstreamFPrime.Layout.Stage1.PilotPiCCS
