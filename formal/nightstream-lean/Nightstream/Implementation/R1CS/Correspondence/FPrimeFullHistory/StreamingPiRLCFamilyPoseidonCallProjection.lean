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

/-! A call site carries every root that changes between input and output runs.
The roots come from one `RawRun`; leaf class alone is not enough to select
them. -/

structure CallSite where
  kind : LeafClass
  localIndex : Nat
  freshFinalStart : Nat
  firstFreshCount : Nat
  initialCarriedFinalStart : Option Nat
  initialCapacityFinalStart : Nat
  localFinalStart : Nat

def CallSite.freshOrdinal (site : CallSite) (lane : Fin 4) : Option Nat :=
  match site.localIndex, site.kind with
  | 0, .direct => some lane.val
  | 0, .partialStart =>
      if lane.val < 2 then none else some (lane.val - 2)
  | index + 1, .chained _ =>
      some (site.firstFreshCount + index * 4 + lane.val)
  | _, _ => none

def CallSite.externalASlotStart
    (site : CallSite) (lane : Fin 4) : Option Nat :=
  match site.freshOrdinal lane with
  | some ordinal => some (site.freshFinalStart + ordinal * slotWidth)
  | none =>
      match site.localIndex, site.kind, site.initialCarriedFinalStart with
      | 0, .partialStart, some start => some (start + lane.val * slotWidth)
      | _, _, _ => none

def CallSite.externalBSlotStart
    (site : CallSite) (lane : Fin 4) : Option Nat :=
  match site.localIndex, site.kind with
  | 0, .direct | 0, .partialStart =>
      some (site.initialCapacityFinalStart + lane.val * slotWidth)
  | _, _ => none

def CallSite.previousLocalSlotStart (site : CallSite) : Option Nat :=
  match site.localIndex, site.kind with
  | _index + 1, .chained _ =>
      some (site.localFinalStart - localSlotCount * slotWidth)
  | _, _ => none

/-- Absolute column of one relative digit. `none` means that the selected
leaf class does not own that slot class. -/
def digitColumn (site : CallSite) :
    Slot → Fin 41 → Option Nat
  | .externalA lane, digit =>
      (site.externalASlotStart lane).map (fun start => start + digit.val)
  | .externalB lane, digit =>
      (site.externalBSlotStart lane).map (fun start => start + digit.val)
  | .previousLocal slotIndex, digit =>
      (site.previousLocalSlotStart).map (fun start =>
        start + slotIndex.val * slotWidth + digit.val)
  | .local slotIndex, digit =>
      some
        (site.localFinalStart + slotIndex.val * slotWidth + digit.val)

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

def projectFinalAssignment (site : CallSite)
    (assignment : Fin productionFinalColumns → F) : FinalAssignment where
  explicit
    | .one => absoluteValue assignment 0
    | .selector => absoluteValue assignment (selectorColumn site.kind)
  digit slot digit :=
    match digitColumn site slot digit with
    | some column => absoluteValue assignment column
    | none => 0

@[simp] theorem projected_one (site : CallSite)
    (assignment : Fin productionFinalColumns → F) :
    (projectFinalAssignment site assignment).explicit .one =
      absoluteValue assignment 0 := by
  rfl

@[simp] theorem projected_selector (site : CallSite)
    (assignment : Fin productionFinalColumns → F) :
    (projectFinalAssignment site assignment).explicit .selector =
      absoluteValue assignment (selectorColumn site.kind) := by
  rfl

theorem projected_digit_of_some (site : CallSite)
    (assignment : Fin productionFinalColumns → F) (slot : Slot)
    (digit : Fin 41) (column : Nat)
    (owned : digitColumn site slot digit = some column) :
    (projectFinalAssignment site assignment).digit slot digit =
      absoluteValue assignment column := by
  simp [projectFinalAssignment, owned]

theorem projected_digit_of_none (site : CallSite)
    (assignment : Fin productionFinalColumns → F) (slot : Slot)
    (digit : Fin 41)
    (unsupported : digitColumn site slot digit = none) :
    (projectFinalAssignment site assignment).digit slot digit = 0 := by
  simp [projectFinalAssignment, unsupported]

@[simp] theorem projected_local (site : CallSite)
    (assignment : Fin productionFinalColumns → F) (slotIndex : Fin 86)
    (digit : Fin 41) :
    (projectFinalAssignment site assignment).digit (.local slotIndex) digit =
      absoluteValue assignment
        (site.localFinalStart + slotIndex.val * slotWidth +
          digit.val) := by
  simp [projectFinalAssignment, digitColumn]

/-- The projected current-local digit is the same absolute coordinate owned
by the exact same-index decoder instance. -/
theorem projected_local_from_indexedOwnership
    {placement : Placement} {index : Nat}
    {call : Poseidon2Call.Call}
    (ownership : placement.IndexedOwnership index call)
    (site : CallSite)
    (localRoot : site.localFinalStart = 2218425 + index * (86 * 41))
    (assignment : Fin productionFinalColumns → F)
    (slotIndex : Fin 86) (digit : Fin 41) :
    (projectFinalAssignment site assignment).digit
        (.local slotIndex) digit =
      absoluteValue assignment
        (placement.decoder.finalStart +
          index * placement.decoder.finalStride +
          slotIndex.val * slotWidth + digit.val) := by
  rw [projected_local, localRoot, ownership.decoderFinalStart]

end Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonCallProjection
