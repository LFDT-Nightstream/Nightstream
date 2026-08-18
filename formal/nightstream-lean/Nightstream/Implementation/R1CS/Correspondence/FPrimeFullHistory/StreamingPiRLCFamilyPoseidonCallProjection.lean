import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingPiRLCFamilyBodyCarryRetained
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingPiRLCFamilyBodyOpeningRows
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingPiRLCFamilyPoseidonLeafModel
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingPiRLCFamilyReplayPlacement

/-!
Contract: same-assignment projection from the absolute production PiRLC body
columns to one relative Poseidon2 replay leaf.

Assurance tier: artifact-checked column projection.

Owns: exact explicit-column and radix-3 slot locations for direct,
partial-start, and chained replay calls. Unsupported class-slot pairs fail
closed to zero. Current-local ownership is rewritten through the exact replay
placement theorem.

Does not own: assignment satisfaction, row semantics, call-class coverage,
family-phase semantics, or lifecycle soundness.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonCallProjection

open Nightstream.SuperNeo.Concrete
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonLeafModel
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonLeafModel.Wire
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyReplayPlacement

private abbrev openingAudit :=
  Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyBodyOpeningRows.audit

private abbrev carryAudit :=
  Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyBodyCarryRetained.audit

def productionFinalColumns : Nat := 8858862
def directSelectorColumn : Nat := 648
def partialSelectorColumn : Nat := 649
def directExternalASlotStart : Nat := 38340
def partialCarriedSlotStart : Nat := 2217769
def externalBSlotStart : Nat := 2217933
def firstLocalSlotStart : Nat := 2218425
def slotWidth : Nat := 41
def localSlotCount : Nat := 86

/-- The fixed roots are exact consequences of the generated opening and carry
receipts. Derived roots use the receipt's consecutive 41-column slot layout. -/
def ArtifactBinding : Prop :=
  openingAudit.selectorColumns = [directSelectorColumn, partialSelectorColumn] ∧
    openingAudit.finalDigitStart = directExternalASlotStart ∧
    openingAudit.finalColumns = productionFinalColumns ∧
    carryAudit.selectorColumns = [directSelectorColumn, partialSelectorColumn] ∧
    carryAudit.finalColumns = productionFinalColumns ∧
    carryAudit.finalStarts = [702, 2142411, 2180049, 2217687, 2217728] ∧
    partialCarriedSlotStart = carryAudit.finalStarts.getD 4 0 + slotWidth ∧
    externalBSlotStart = carryAudit.finalStarts.getD 4 0 + 5 * slotWidth ∧
    firstLocalSlotStart = carryAudit.finalStarts.getD 4 0 + 17 * slotWidth

theorem artifact_binding : ArtifactBinding := by
  have opening :=
    Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyBodyOpeningRows.exact_receipt
  have carry :=
    Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyBodyCarryRetained.audit_valid
  unfold Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyBodyOpeningRows.ExactReceipt at opening
  unfold Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyBodyCarryRetained.AuditValid at carry
  unfold Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyBodyCarryRetained.exactShape at carry
  simp only [ArtifactBinding, productionFinalColumns, directSelectorColumn,
    partialSelectorColumn, directExternalASlotStart, partialCarriedSlotStart,
    externalBSlotStart, firstLocalSlotStart, slotWidth]
  aesop

inductive LeafClass where
  | direct
  | partialStart
  | chained (selector : Nat)
deriving DecidableEq, Repr

def selectorColumn : LeafClass → Nat
  | .direct => directSelectorColumn
  | .partialStart => partialSelectorColumn
  | .chained selector => selector

def currentLocalSlotStart (index : Nat) : Nat :=
  firstLocalSlotStart + index * (localSlotCount * slotWidth)

def previousLocalSlotStart (index : Nat) : Nat :=
  firstLocalSlotStart + (index - 1) * (localSlotCount * slotWidth)

def externalASlotStart : LeafClass → Fin 4 → Nat
  | .direct, lane => directExternalASlotStart + lane.val * slotWidth
  | .chained _, lane =>
      directExternalASlotStart + (4 + lane.val) * slotWidth
  | .partialStart, lane =>
      if lane.val < 2 then
        partialCarriedSlotStart + lane.val * slotWidth
      else
        directExternalASlotStart + (lane.val - 2) * slotWidth

def externalBSlotStartFor (lane : Fin 4) : Nat :=
  externalBSlotStart + lane.val * slotWidth

/-- Absolute column of one relative digit. `none` means that the selected
leaf class does not own that slot class. -/
def digitColumn (kind : LeafClass) (index : Nat) :
    Slot → Fin 41 → Option Nat
  | .externalA lane, digit =>
      some (externalASlotStart kind lane + digit.val)
  | .externalB lane, digit =>
      match kind with
      | .direct | .partialStart =>
          some (externalBSlotStartFor lane + digit.val)
      | .chained _ => none
  | .previousLocal slotIndex, digit =>
      match kind with
      | .chained _ =>
          if 0 < index then
            some
              (previousLocalSlotStart index + slotIndex.val * slotWidth +
                digit.val)
          else
            none
      | .direct | .partialStart => none
  | .local slotIndex, digit =>
      some
        (currentLocalSlotStart index + slotIndex.val * slotWidth + digit.val)

/-- Totalized access to the exact finite CCS assignment. An out-of-range
column fails closed to zero. -/
def absoluteValue
    (assignment : Fin productionFinalColumns → F) (column : Nat) : F :=
  if bounded : column < productionFinalColumns then
    assignment ⟨column, bounded⟩
  else
    0

theorem absoluteValue_of_lt
    (assignment : Fin productionFinalColumns → F) (column : Nat)
    (bounded : column < productionFinalColumns) :
    absoluteValue assignment column = assignment ⟨column, bounded⟩ := by
  simp [absoluteValue, bounded]

def projectFinalAssignment (kind : LeafClass) (index : Nat)
    (assignment : Fin productionFinalColumns → F) : FinalAssignment where
  explicit
    | .one => absoluteValue assignment 0
    | .selector => absoluteValue assignment (selectorColumn kind)
  digit slot digit :=
    match digitColumn kind index slot digit with
    | some column => absoluteValue assignment column
    | none => 0

@[simp] theorem projected_one (kind : LeafClass) (index : Nat)
    (assignment : Fin productionFinalColumns → F) :
    (projectFinalAssignment kind index assignment).explicit .one =
      absoluteValue assignment 0 := by
  rfl

@[simp] theorem projected_selector (kind : LeafClass) (index : Nat)
    (assignment : Fin productionFinalColumns → F) :
    (projectFinalAssignment kind index assignment).explicit .selector =
      absoluteValue assignment (selectorColumn kind) := by
  rfl

theorem projected_digit_of_some (kind : LeafClass) (index : Nat)
    (assignment : Fin productionFinalColumns → F) (slot : Slot)
    (digit : Fin 41) (column : Nat)
    (owned : digitColumn kind index slot digit = some column) :
    (projectFinalAssignment kind index assignment).digit slot digit =
      absoluteValue assignment column := by
  simp [projectFinalAssignment, owned]

theorem projected_digit_of_none (kind : LeafClass) (index : Nat)
    (assignment : Fin productionFinalColumns → F) (slot : Slot)
    (digit : Fin 41)
    (unsupported : digitColumn kind index slot digit = none) :
    (projectFinalAssignment kind index assignment).digit slot digit = 0 := by
  simp [projectFinalAssignment, unsupported]

@[simp] theorem projected_externalA (kind : LeafClass) (index : Nat)
    (assignment : Fin productionFinalColumns → F) (lane : Fin 4)
    (digit : Fin 41) :
    (projectFinalAssignment kind index assignment).digit (.externalA lane) digit =
      absoluteValue assignment (externalASlotStart kind lane + digit.val) := by
  simp [projectFinalAssignment, digitColumn]

@[simp] theorem projected_direct_externalB (index : Nat)
    (assignment : Fin productionFinalColumns → F) (lane : Fin 4)
    (digit : Fin 41) :
    (projectFinalAssignment .direct index assignment).digit (.externalB lane) digit =
      absoluteValue assignment (externalBSlotStartFor lane + digit.val) := by
  simp [projectFinalAssignment, digitColumn]

@[simp] theorem projected_partial_externalB (index : Nat)
    (assignment : Fin productionFinalColumns → F) (lane : Fin 4)
    (digit : Fin 41) :
    (projectFinalAssignment .partialStart index assignment).digit (.externalB lane) digit =
      absoluteValue assignment (externalBSlotStartFor lane + digit.val) := by
  simp [projectFinalAssignment, digitColumn]

@[simp] theorem projected_chained_externalB (selector index : Nat)
    (assignment : Fin productionFinalColumns → F) (lane : Fin 4)
    (digit : Fin 41) :
    (projectFinalAssignment (.chained selector) index assignment).digit
        (.externalB lane) digit = 0 := by
  simp [projectFinalAssignment, digitColumn]

@[simp] theorem projected_direct_previousLocal (index : Nat)
    (assignment : Fin productionFinalColumns → F) (slotIndex : Fin 86)
    (digit : Fin 41) :
    (projectFinalAssignment .direct index assignment).digit
        (.previousLocal slotIndex) digit = 0 := by
  simp [projectFinalAssignment, digitColumn]

@[simp] theorem projected_partial_previousLocal (index : Nat)
    (assignment : Fin productionFinalColumns → F) (slotIndex : Fin 86)
    (digit : Fin 41) :
    (projectFinalAssignment .partialStart index assignment).digit
        (.previousLocal slotIndex) digit = 0 := by
  simp [projectFinalAssignment, digitColumn]

theorem projected_chained_previousLocal (selector index : Nat)
    (positive : 0 < index)
    (assignment : Fin productionFinalColumns → F) (slotIndex : Fin 86)
    (digit : Fin 41) :
    (projectFinalAssignment (.chained selector) index assignment).digit
        (.previousLocal slotIndex) digit =
      absoluteValue assignment
        (previousLocalSlotStart index + slotIndex.val * slotWidth +
          digit.val) := by
  simp [projectFinalAssignment, digitColumn, positive]

@[simp] theorem projected_local (kind : LeafClass) (index : Nat)
    (assignment : Fin productionFinalColumns → F) (slotIndex : Fin 86)
    (digit : Fin 41) :
    (projectFinalAssignment kind index assignment).digit (.local slotIndex) digit =
      absoluteValue assignment
        (currentLocalSlotStart index + slotIndex.val * slotWidth +
          digit.val) := by
  simp [projectFinalAssignment, digitColumn]

/-- The projected current-local digit is the same absolute coordinate owned
by the exact same-index decoder instance. -/
theorem projected_local_from_indexedOwnership
    {placement : Placement} {index : Nat}
    {call : Poseidon2Call.Call}
    (ownership : placement.IndexedOwnership index call)
    (kind : LeafClass) (assignment : Fin productionFinalColumns → F)
    (slotIndex : Fin 86) (digit : Fin 41) :
    (projectFinalAssignment kind index assignment).digit
        (.local slotIndex) digit =
      absoluteValue assignment
        (placement.decoder.finalStart +
          index * placement.decoder.finalStride +
          slotIndex.val * slotWidth + digit.val) := by
  rw [projected_local, ownership.decoderFinalStart]
  rfl

end Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonCallProjection
