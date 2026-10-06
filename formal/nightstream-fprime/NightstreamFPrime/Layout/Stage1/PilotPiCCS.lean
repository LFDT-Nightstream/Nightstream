import NightstreamFPrime.Layout.PilotProduction
import NightstreamFPrime.Layout.PiCCS.v1_2.Preservation
import NightstreamFPrime.Layout.Stage1.PiCCSInputs

/-!
Obligation: Assemble the closed pilot, the concrete parent-owned PiCCS proof
inputs, and the PiCCS physical row packet into the current Stage 1 prefix.

The running instance and fresh public input reuse pilot columns. Four public
verifier-context words precede the new 16,436-column proof-input interval,
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
def interface : Formal.Interface logicalWidth 9 publicFits :=
  PiCCSInputs.interface logicalWidth publicFits

/-- PiCCS starts after the completed pilot and all parent-owned proof inputs. -/
def piCcsOffset : Nat := PiCCSInputs.phaseOffset

theorem piCcsOffset_eq : piCcsOffset = 6908836 := by
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
    physicalRowCount relation = 8597929 := by
  unfold physicalRowCount physicalRows
  rw [List.length_append]
  change Pilot.physicalRowCount PilotProduction.interface
      PilotProduction.witnessOffset +
    NightstreamFPrime.Layout.PiCCS.v1_2.physicalRowCount relation
      (interface (publicFits := publicFits)) piCcsOffset = 8597929
  rw [PilotProduction.physicalRowCount_eq,
    NightstreamFPrime.Layout.PiCCS.v1_2.ProductionInputs.physicalRowCount_eq
      relation (interface (publicFits := publicFits)) piCcsOffset
      (PiCCSInputs.externalInputsLinear logicalWidth publicFits)]

theorem physicalColumnCount_eq
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    physicalColumnCount relation = 8688741 := by
  unfold physicalColumnCount
  rw [PilotProduction.physicalColumnCount_eq,
    NightstreamFPrime.Layout.PiCCS.v1_2.ProductionInputs.physicalColumnCount_eq
      relation (interface (publicFits := publicFits)) piCcsOffset
      (PiCCSInputs.externalInputsLinear logicalWidth publicFits),
    piCcsOffset_eq]
  norm_num

theorem jointDomain_eq
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    jointDomain relation = 8688741 := by
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
        [160, 140288, 95352, 153440, 20733, 812, 3364, 18916,
          56, 800, 3602, 1342600] ∧
      NightstreamFPrime.Layout.PiCCS.v1_2.physicalColumnDeltas relation
        (interface (publicFits := publicFits)) piCcsOffset =
        [0, 140288, 95352, 153440, 20733, 756, 3364, 18916,
          56, 800, 3600, 1342600] ∧
      cumulativePhysicalRows relation =
        [6817966, 6958254, 7053606, 7207046, 7227779, 7228591,
          7231955, 7250871, 7250927, 7251727, 7255329, 8597929] ∧
      cumulativePhysicalColumns relation =
        [6908836, 7049124, 7144476, 7297916, 7318649, 7319405,
          7322769, 7341685, 7341741, 7342541, 7346141, 8688741] ∧
      cumulativeJointDomains relation =
        [6908836, 7049124, 7144476, 7297916, 7318649, 7319405,
          7322769, 7341685, 7341741, 7342541, 7346141, 8688741] := by
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
