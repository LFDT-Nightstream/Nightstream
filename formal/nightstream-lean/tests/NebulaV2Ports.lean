import Nightstream.Protocol.NebulaV2

set_option autoImplicit false

namespace tests.NebulaV2Ports

open Nightstream.Protocol.NebulaV2
open Nightstream.Protocol.NebulaV2.Ports

def slotZero : Fin slotsPerStep := ⟨0, by decide⟩
def slotOne : Fin slotsPerStep := ⟨1, by decide⟩

/-- Slot zero is a hole and slot one is active. This is valid V2 layout. -/
def holeThenActive : PhysicalPorts Nat := fun position =>
  if position = slotOne then some 7 else none

theorem active_after_hole_is_preserved :
    decodeAt holeThenActive slotOne = some ⟨slotOne, 7⟩ := by
  simp [decodeAt, holeThenActive]

theorem the_hole_does_not_end_decoding :
    decodeAt holeThenActive slotZero = none ∧
      decodeAt holeThenActive slotOne = some ⟨slotOne, 7⟩ := by
  constructor <;> simp [decodeAt, holeThenActive, slotZero, slotOne]

/- A compact payload-only decoder maps two different physical layouts to the
same value list. Such a decoder cannot be an authority-bearing fixed-port
codec. -/
namespace MissingPhysicalPositions

def atZero : PhysicalPorts Nat := fun position =>
  if position = slotZero then some 7 else none

def atOne : PhysicalPorts Nat := holeThenActive

theorem layouts_differ : atZero ≠ atOne := by
  intro equal
  have atSlotZero := congrFun equal slotZero
  simp [atZero, atOne, holeThenActive, slotZero, slotOne] at atSlotZero

theorem compact_decodings_are_equal :
    compactPayloads atZero = compactPayloads atOne := by
  decide

end MissingPhysicalPositions

def rowTwo : ApplicationRowIndex := ⟨2, by decide⟩
def portTwenty : RowPortIndex := ⟨20, by decide⟩

theorem last_row_port_routes_to_last_physical_slot :
    (route rowTwo portTwenty).val = 62 :=
  rfl

theorem every_physical_slot_has_one_exact_row_port
    (position : Fin slotsPerStep) :
    route (unroute position).1 (unroute position).2 = position :=
  route_unroute position

/- A routing table that maps all row ports to one position loses accesses.
Distinct type-level positions are not enough unless the generated table uses
the proved `route` bijection. -/
def collapsedRoute (_row : ApplicationRowIndex)
    (_port : RowPortIndex) : Fin slotsPerStep :=
  ⟨0, by decide⟩

theorem collapsed_route_is_not_injective :
    ¬ Function.Injective
      (fun pair : ApplicationRowIndex × RowPortIndex =>
        collapsedRoute pair.1 pair.2) := by
  intro injective
  let left : ApplicationRowIndex × RowPortIndex :=
    (⟨0, by decide⟩, ⟨0, by decide⟩)
  let right : ApplicationRowIndex × RowPortIndex :=
    (⟨0, by decide⟩, ⟨1, by decide⟩)
  have equal : left = right := injective (by rfl)
  have portsEqual := congrArg (fun pair => pair.2.val) equal
  norm_num [left, right] at portsEqual

def noMemoryRow : NormalizedRow :=
  { kind := .program
    memoryPorts := fun _ => none }

def noMemoryStep : CheckedStep :=
  { rows := fun _ => noMemoryRow }

/-- An application cannot report a semantic access when every fixed port is
inactive. The `Covers` boundary rejects this missing-port countermodel. -/
theorem inactive_ports_do_not_cover_a_semantic_access
    (access : Access) :
    ¬ noMemoryStep.Covers [access] := by
  simp [CheckedStep.Covers, CheckedStep.accesses, compactPayloads,
    CheckedStep.physicalPorts, noMemoryStep, noMemoryRow]

end tests.NebulaV2Ports
