import Nightstream.Implementation.NebulaV2.Memory.Product.SemanticBridge
import Nightstream.Implementation.NebulaV2.Memory.Operation.SlotRows

/-!
Contract: prove that one valid operation-slot row block supplies the exact RS
and WS records consumed by every product repetition.

Assurance tier: implementation-to-protocol bridge.

Owns evaluation of the packed timestamp/global-index expressions, value
expressions, and canonical pad selection. The represented record is derived
from the independent `OperationSlot.ValidAt` result.

Does not own prefix-counter rows, aggregate product accumulation,
application-port coverage, or generated column ownership.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.NebulaV2.OperationSlotProductBridge

open Nightstream.Implementation.NebulaV2.MemoryProductSemanticBridge
open Nightstream.Implementation.NebulaV2.MemoryProductUpdateRows
open Nightstream.Implementation.NebulaV2.OperationSlotRows
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical.KHorner
open Nightstream.Protocol.NebulaV2
open Nightstream.Protocol.NebulaV2.ConcreteLaneGeometry
open Nightstream.Protocol.NebulaV2.Fingerprint
open Nightstream.Protocol.NebulaV2.OperationSlot

/-- The product layout reads the same derived write-timestamp column that the
source relation proves. -/
structure Linked (layout : OperationSlotRows.Layout) : Prop where
  writeTimestamp :
    layout.product.writeTimestamp layout.slot =
      [(layout.aux.writeTimestamp, 1)]

private theorem singleton_eval
    {assignment : Nat → Nat}
    (column coefficient : Nat)
    (productBound : coefficient * assignment column < goldilocksP) :
    lcEval assignment [(column, coefficient)] =
      coefficient * assignment column := by
  simp only [lcEval, List.foldl_cons, List.foldl_nil, Nat.zero_add]
  exact Nat.mod_eq_of_lt productBound

private theorem global_index_eval
    {layout : OperationSlotRows.Layout} {assignment : Nat → Nat}
    {countBefore countAfter stepTimestampIn : Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (holds : Satisfies (OperationSlotRows.rows layout) assignment)
    (valid : OperationSlot.ValidAt
      (decoded layout assignment countBefore countAfter) stepTimestampIn) :
    lcEval assignment
        (layout.product.operationGlobalIndex layout.slot) =
      (decoded layout assignment countBefore countAfter).index := by
  have addressBound : assignment layout.aux.address < 2 ^ 16 := by
    simpa [decoded] using valid.addressBound
  have ramBinary : assignment layout.isRamColumn = 0 ∨
      assignment layout.isRamColumn = 1 := by
    simpa [decoded] using valid.isRamBinary
  have addressEval := address_scaled_eval canonical one holds 1 (by
    have addressField : 2 ^ 16 < goldilocksP := by decide
    omega)
  simp only [one_mul] at addressEval
  have ramBound :
      romCells * assignment layout.isRamColumn < goldilocksP := by
    rcases ramBinary with ram | ram <;>
      simp [ram, romCells, goldilocksP]
  have ramEval := singleton_eval layout.isRamColumn romCells ramBound
  have ramEval' :
      lcEval assignment
          [(layout.product.operationIsRamColumn layout.slot, romCells)] =
        romCells * assignment layout.isRamColumn := by
    simpa [Layout.isRamColumn] using ramEval
  rw [Layout.operationGlobalIndex, lcEval_append, addressEval, ramEval']
  have sumBound :
      assignment layout.aux.address +
          romCells * assignment layout.isRamColumn < goldilocksP := by
    rcases ramBinary with ram | ram <;>
      simp [ram, romCells, goldilocksP] <;> omega
  rw [Nat.mod_eq_of_lt sumBound]
  rcases ramBinary with ram | ram
  · simp [decoded, OperationSlot.Value.index, OperationSlot.Value.space,
      ram, globalIndex]
  · simp [decoded, OperationSlot.Value.index, OperationSlot.Value.space,
      ram, globalIndex, Nat.add_comm]

private theorem read_packed_eval
    {layout : OperationSlotRows.Layout} {assignment : Nat → Nat}
    {countBefore countAfter stepTimestampIn : Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (holds : Satisfies (OperationSlotRows.rows layout) assignment)
    (valid : OperationSlot.ValidAt
      (decoded layout assignment countBefore countAfter) stepTimestampIn)
    (active : assignment layout.padColumn = 0) :
    lcEval assignment
        (layout.product.operationPacked .reads layout.slot) =
      packedNat
        (decoded layout assignment countBefore countAfter).readTuple := by
  have activeDecoded :
      (decoded layout assignment countBefore countAfter).pad = 0 := by
    simpa [decoded] using active
  have readTimestampBound :
      assignment layout.aux.readTimestamp < timestampLimit := by
    simpa [decoded] using valid.readTimestampBound
  have addressBound : assignment layout.aux.address < 2 ^ 16 := by
    simpa [decoded] using valid.addressBound
  have ramBinary : assignment layout.isRamColumn = 0 ∨
      assignment layout.isRamColumn = 1 := by
    simpa [decoded] using valid.isRamBinary
  have timestampEval := read_timestamp_scaled_eval canonical one holds 1 (by
    have bound := readTimestampBound
    have field : timestampLimit < goldilocksP := by decide
    omega)
  simp only [one_mul] at timestampEval
  have addressScaled := address_scaled_eval canonical one holds timestampLimit
    (by
      have address := addressBound
      norm_num [timestampLimit,
        Nightstream.Protocol.NebulaV2.timestampBits, goldilocksP] at *
      omega)
  have ramBound : timestampLimit * romCells *
      assignment layout.isRamColumn < goldilocksP := by
    rcases ramBinary with ram | ram <;>
      norm_num [ram, timestampLimit,
        Nightstream.Protocol.NebulaV2.timestampBits, romCells, goldilocksP]
  have ramEval := singleton_eval layout.isRamColumn
    (timestampLimit * romCells) (by
      simpa [Nat.mul_assoc] using ramBound)
  have ramEval' : lcEval assignment
      [(layout.product.operationIsRamColumn layout.slot,
        timestampLimit * romCells)] =
      timestampLimit * romCells * assignment layout.isRamColumn := by
    simpa [Layout.isRamColumn] using ramEval
  simp only [Layout.operationPacked, Layout.operationReadPacked]
  rw [lcEval_append, lcEval_append, timestampEval, addressScaled]
  have packedBound := packedNat_lt_goldilocks
    (valid.read_tuple_in_range activeDecoded)
  have indexFormula :
      (decoded layout assignment countBefore countAfter).index =
        assignment layout.aux.address +
          romCells * assignment layout.isRamColumn := by
    rcases ramBinary with ram | ram <;>
      simp [decoded, OperationSlot.Value.index, OperationSlot.Value.space,
        ram, globalIndex, Nat.add_comm]
  simp only [packedNat, timestampRadix, OperationSlot.Value.readTuple]
    at packedBound ⊢
  rw [indexFormula] at packedBound ⊢
  simp only [decoded] at packedBound ⊢
  change assignment layout.aux.readTimestamp +
      timestampLimit * (assignment layout.aux.address +
        romCells * assignment layout.isRamColumn) < goldilocksP at packedBound
  have firstBound :
      assignment layout.aux.readTimestamp +
          timestampLimit * assignment layout.aux.address < goldilocksP := by
    simp only [Nat.mul_add] at packedBound
    omega
  rw [Nat.mod_eq_of_lt firstBound, ramEval']
  have finalBound :
      assignment layout.aux.readTimestamp +
          timestampLimit * assignment layout.aux.address +
            timestampLimit * romCells * assignment layout.isRamColumn <
        goldilocksP := by
    simpa [Nat.mul_add, Nat.mul_assoc, Nat.add_assoc] using packedBound
  rw [Nat.mod_eq_of_lt finalBound]
  simp only [Nat.mul_add, Nat.mul_assoc, Nat.add_assoc]

private theorem write_packed_eval
    {layout : OperationSlotRows.Layout} {assignment : Nat → Nat}
    {countBefore countAfter stepTimestampIn : Nat}
    (linked : Linked layout)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (holds : Satisfies (OperationSlotRows.rows layout) assignment)
    (valid : OperationSlot.ValidAt
      (decoded layout assignment countBefore countAfter) stepTimestampIn)
    (active : assignment layout.padColumn = 0) :
    lcEval assignment
        (layout.product.operationPacked .writes layout.slot) =
      packedNat
        (decoded layout assignment countBefore countAfter).writeTuple := by
  have activeDecoded :
      (decoded layout assignment countBefore countAfter).pad = 0 := by
    simpa [decoded] using active
  have addressBound : assignment layout.aux.address < 2 ^ 16 := by
    simpa [decoded] using valid.addressBound
  have ramBinary : assignment layout.isRamColumn = 0 ∨
      assignment layout.isRamColumn = 1 := by
    simpa [decoded] using valid.isRamBinary
  have timestampEval : lcEval assignment
      (layout.product.writeTimestamp layout.slot) =
      assignment layout.aux.writeTimestamp := by
    rw [linked.writeTimestamp]
    simpa using singleton_eval _ 1 (by
      simpa using canonical layout.aux.writeTimestamp)
  have addressScaled := address_scaled_eval canonical one holds timestampLimit
    (by
      have address := addressBound
      norm_num [timestampLimit,
        Nightstream.Protocol.NebulaV2.timestampBits, goldilocksP] at *
      omega)
  have ramBound : timestampLimit * romCells *
      assignment layout.isRamColumn < goldilocksP := by
    rcases ramBinary with ram | ram <;>
      norm_num [ram, timestampLimit,
        Nightstream.Protocol.NebulaV2.timestampBits, romCells, goldilocksP]
  have ramEval := singleton_eval layout.isRamColumn
    (timestampLimit * romCells) (by
      simpa [Nat.mul_assoc] using ramBound)
  have ramEval' : lcEval assignment
      [(layout.product.operationIsRamColumn layout.slot,
        timestampLimit * romCells)] =
      timestampLimit * romCells * assignment layout.isRamColumn := by
    simpa [Layout.isRamColumn] using ramEval
  simp only [Layout.operationPacked, Layout.operationWritePacked]
  rw [lcEval_append, lcEval_append, timestampEval, addressScaled]
  have packedBound := packedNat_lt_goldilocks
    (valid.write_tuple_in_range activeDecoded)
  have indexFormula :
      (decoded layout assignment countBefore countAfter).index =
        assignment layout.aux.address +
          romCells * assignment layout.isRamColumn := by
    rcases ramBinary with ram | ram <;>
      simp [decoded, OperationSlot.Value.index, OperationSlot.Value.space,
        ram, globalIndex, Nat.add_comm]
  simp only [packedNat, timestampRadix, OperationSlot.Value.writeTuple]
    at packedBound ⊢
  rw [indexFormula] at packedBound ⊢
  simp only [decoded] at packedBound ⊢
  change assignment layout.aux.writeTimestamp +
      timestampLimit * (assignment layout.aux.address +
        romCells * assignment layout.isRamColumn) < goldilocksP at packedBound
  have firstBound :
      assignment layout.aux.writeTimestamp +
          timestampLimit * assignment layout.aux.address < goldilocksP := by
    simp only [Nat.mul_add] at packedBound
    omega
  rw [Nat.mod_eq_of_lt firstBound, ramEval']
  have finalBound :
      assignment layout.aux.writeTimestamp +
          timestampLimit * assignment layout.aux.address +
            timestampLimit * romCells * assignment layout.isRamColumn <
        goldilocksP := by
    simpa [Nat.mul_add, Nat.mul_assoc, Nat.add_assoc] using packedBound
  rw [Nat.mod_eq_of_lt finalBound]
  simp only [Nat.mul_add, Nat.mul_assoc, Nat.add_assoc]

def representedRecord
    {layout : OperationSlotRows.Layout} {assignment : Nat → Nat}
    {countBefore countAfter stepTimestampIn : Nat}
    (valid : OperationSlot.ValidAt
      (decoded layout assignment countBefore countAfter) stepTimestampIn)
    (role : OperationRole) : Option BoundedTuple :=
  if active :
      (decoded layout assignment countBefore countAfter).pad = 0 then
    some (match role with
      | .reads => valid.readBounded active
      | .writes => valid.writeBounded active)
  else none

/-- Every repetition consumes the same exact typed RS or WS record. -/
theorem gate_represents
    {layout : OperationSlotRows.Layout} {assignment : Nat → Nat}
    {countBefore countAfter stepTimestampIn : Nat}
    (linked : Linked layout)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (holds : Satisfies (OperationSlotRows.rows layout) assignment)
    (valid : OperationSlot.ValidAt
      (decoded layout assignment countBefore countAfter) stepTimestampIn)
    (repetition : Fin 2) (role : OperationRole) :
    GateRepresents assignment
      (operationEntry layout.product repetition role layout.slot)
      (representedRecord valid role) := by
  rcases valid.padBinary with active | padded
  · have activeAssignment : assignment layout.padColumn = 0 := by
      simpa [decoded] using active
    rw [representedRecord, dif_pos active]
    apply GateRepresents.active
      (entry := operationEntry layout.product repetition role layout.slot)
      (record := match role with
        | .reads => valid.readBounded active
        | .writes => valid.writeBounded active)
    · rfl
    · exact activeAssignment
    · cases role with
      | reads =>
          simpa [OperationSlot.ValidAt.readBounded] using
            read_packed_eval canonical one holds valid activeAssignment
      | writes =>
          simpa [OperationSlot.ValidAt.writeBounded] using
            write_packed_eval linked canonical one holds valid activeAssignment
    · cases role with
      | reads =>
          simpa [Layout.operationValue, OperationSlot.ValidAt.readBounded,
            OperationSlot.Value.readTuple] using
            read_value_eval canonical one holds
      | writes =>
          simpa [Layout.operationValue, OperationSlot.ValidAt.writeBounded,
            OperationSlot.Value.writeTuple] using
            write_value_eval canonical one holds
  · have paddedAssignment : assignment layout.padColumn = 1 := by
      simpa [decoded] using padded
    have notActive : ¬ assignment layout.padColumn = 0 := by omega
    rw [representedRecord, dif_neg (by omega)]
    exact GateRepresents.padded rfl paddedAssignment

end Nightstream.Implementation.NebulaV2.OperationSlotProductBridge
