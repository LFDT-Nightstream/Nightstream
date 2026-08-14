import Mathlib.Data.List.FinRange
import Nightstream.Protocol.Nebula.Types

/-!
Contract: fixed physical memory-port positions for one V2 checked step.

Assurance tier: model-level.

Owns the 63-position typed port vector and a decoder that retains each active
port's structural position. `none` represents canonical inactive padding.

Does not own the byte codec, nonzero padding fields, WASM instruction coverage,
three-row routing, or generated circuit columns.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Protocol.Nebula.Ports

def slotsPerStep : Nat := 63
def applicationRowsPerStep : Nat := 3
def slotsPerApplicationRow : Nat := 21

theorem row_slot_capacity :
    applicationRowsPerStep * slotsPerApplicationRow = slotsPerStep := by
  decide

abbrev ApplicationRowIndex := Fin applicationRowsPerStep
abbrev RowPortIndex := Fin slotsPerApplicationRow

abbrev PhysicalPorts (Payload : Type) := Fin slotsPerStep → Option Payload

structure Positioned (Payload : Type) where
  position : Fin slotsPerStep
  payload : Payload
deriving DecidableEq, Repr

def decodeAt
    {Payload : Type}
    (ports : PhysicalPorts Payload)
    (position : Fin slotsPerStep) : Option (Positioned Payload) :=
  (ports position).map fun payload => ⟨position, payload⟩

/-- Active ports remain in increasing physical-position order and retain their
positions. Inactive holes are skipped, not treated as an end marker. -/
def activePorts
    {Payload : Type} (ports : PhysicalPorts Payload) :
    List (Positioned Payload) :=
  (List.finRange slotsPerStep).filterMap (decodeAt ports)

def compactPayloads
    {Payload : Type} (ports : PhysicalPorts Payload) : List Payload :=
  (List.finRange slotsPerStep).filterMap ports

theorem compactPayloads_eq_filterMap_ofFn
    {Payload : Type} (ports : PhysicalPorts Payload) :
    compactPayloads ports = (List.ofFn ports).filterMap id := by
  simp [compactPayloads, List.ofFn_eq_map]

/-- Verifier-key-bound 3-by-21 route. Row order is major and port order is
minor, so the physical order is exactly the normalized application order. -/
def route
    (row : ApplicationRowIndex) (port : RowPortIndex) : Fin slotsPerStep :=
  ⟨row.val * slotsPerApplicationRow + port.val, by
    have rowBound := row.isLt
    have portBound := port.isLt
    simp only [applicationRowsPerStep] at rowBound
    simp only [slotsPerApplicationRow] at portBound ⊢
    simp only [slotsPerStep]
    omega⟩

def unroute (position : Fin slotsPerStep) :
    ApplicationRowIndex × RowPortIndex :=
  (⟨position.val / slotsPerApplicationRow, by
      have positionBound := position.isLt
      simp only [slotsPerStep] at positionBound
      simp only [applicationRowsPerStep, slotsPerApplicationRow]
      omega⟩,
    ⟨position.val % slotsPerApplicationRow, by
      simp only [slotsPerApplicationRow]
      exact Nat.mod_lt _ (by decide)⟩)

theorem route_unroute (position : Fin slotsPerStep) :
    route (unroute position).1 (unroute position).2 = position := by
  apply Fin.ext
  simp only [route, unroute]
  simpa [Nat.mul_comm] using
    Nat.div_add_mod position.val slotsPerApplicationRow

theorem unroute_route
    (row : ApplicationRowIndex) (port : RowPortIndex) :
    unroute (route row port) = (row, port) := by
  apply Prod.ext <;> apply Fin.ext
  · simp only [unroute, route]
    have portBound := port.isLt
    simp only [slotsPerApplicationRow] at portBound ⊢
    omega
  · simp only [unroute, route]
    have portBound := port.isLt
    simp only [slotsPerApplicationRow] at portBound ⊢
    omega

theorem route_injective :
    Function.Injective (fun pair : ApplicationRowIndex × RowPortIndex =>
      route pair.1 pair.2) := by
  intro left right equal
  have decoded := congrArg unroute equal
  simpa only [unroute_route] using decoded

/-- Exact normalized WASM row classes selected by V2. Synthetic padding is
owned by the completed-trace layer and is not an active normalized row. -/
inductive NormalizedRowKind where
  | program
  | callParamInit
  | tailEnter
  | hostCallArg
  | hostCallResult
  | hostEventPerm
  | hostEventGather
  | turnBoundary
deriving DecidableEq, Repr

/-- After the VM has raised `halted`, only the two event-drain row classes can
remain before the one terminal row. A V2 completed execution has no turn
boundary and cannot resume with another program row. -/
def NormalizedRowKind.canDrainAfterHalt : NormalizedRowKind → Prop
  | .hostEventPerm => True
  | .hostEventGather => True
  | _ => False

/-- One normalized application row owns exactly 21 physical memory ports.
There is no separate memory-effect list in this type. -/
structure NormalizedRow where
  kind : NormalizedRowKind
  memoryPorts : RowPortIndex → Option Access

def NormalizedRow.accesses (row : NormalizedRow) : List Access :=
  (List.finRange slotsPerApplicationRow).filterMap row.memoryPorts

theorem NormalizedRow.accessCount_le_capacity (row : NormalizedRow) :
    row.accesses.length ≤ slotsPerApplicationRow := by
  unfold NormalizedRow.accesses
  simpa using
    (List.length_filterMap_le row.memoryPorts
      (List.finRange slotsPerApplicationRow))

def NormalizedRow.Inactive (row : NormalizedRow) : Prop :=
  ∀ port, row.memoryPorts port = none

def NormalizedRow.inactive : NormalizedRow :=
  { kind := .program
    memoryPorts := fun _ => none }

theorem NormalizedRow.inactive_has_no_accesses :
    NormalizedRow.inactive.accesses = [] := by
  simp [NormalizedRow.inactive, NormalizedRow.accesses]

/-- One factor-one V2 checked step owns exactly three normalized rows. -/
structure CheckedStep where
  rows : ApplicationRowIndex → NormalizedRow

def CheckedStep.physicalPorts (step : CheckedStep) : PhysicalPorts Access :=
  fun position =>
    let location := unroute position
    (step.rows location.1).memoryPorts location.2

@[simp]
theorem CheckedStep.physicalPorts_route
    (step : CheckedStep)
    (row : ApplicationRowIndex) (port : RowPortIndex) :
    step.physicalPorts (route row port) =
      (step.rows row).memoryPorts port := by
  simp [CheckedStep.physicalPorts, unroute_route]

/-- This is the only semantic memory-access list exported by a normalized
checked step. It retains physical order and skips holes without compacting the
authority-bearing port vector. -/
def CheckedStep.accesses (step : CheckedStep) : List Access :=
  compactPayloads step.physicalPorts

/-- The three normalized rows in verifier-key order. -/
def CheckedStep.rowList (step : CheckedStep) : List NormalizedRow :=
  List.ofFn step.rows

@[simp]
theorem CheckedStep.rowList_length (step : CheckedStep) :
    step.rowList.length = applicationRowsPerStep := by
  simp [CheckedStep.rowList]

/-- Row-major 3-by-21 decoding is exactly physical-position decoding. This
is an ordered-list equality and therefore fixes every active port around
inactive holes. -/
theorem CheckedStep.rowList_flatMap_accesses (step : CheckedStep) :
    step.rowList.flatMap NormalizedRow.accesses = step.accesses := by
  let flat : Fin (applicationRowsPerStep * slotsPerApplicationRow) →
      Option Access :=
    fun position =>
      (step.rows (unroute (Fin.cast row_slot_capacity position)).1).memoryPorts
        (unroute (Fin.cast row_slot_capacity position)).2
  have split := List.ofFn_mul flat
  have splitRows :
      List.ofFn flat =
        (List.ofFn fun row : ApplicationRowIndex =>
          List.ofFn fun port : RowPortIndex =>
            (step.rows row).memoryPorts port).flatten := by
    rw [split]
    apply congrArg List.flatten
    apply List.ofFn_inj.mpr
    funext row
    apply List.ofFn_inj.mpr
    funext port
    simp only [flat]
    have routed :
        Fin.cast row_slot_capacity
            ⟨row.val * slotsPerApplicationRow + port.val, by
              calc
                row.val * slotsPerApplicationRow + port.val <
                    (row.val + 1) * slotsPerApplicationRow :=
                  (Nat.add_lt_add_left port.isLt _).trans_eq (by
                    rw [Nat.add_mul, Nat.one_mul])
                _ ≤ applicationRowsPerStep * slotsPerApplicationRow :=
                  Nat.mul_le_mul_right slotsPerApplicationRow row.isLt⟩ =
          route row port := by
      apply Fin.ext
      rfl
    rw [routed, unroute_route]
  have eachRow (row : NormalizedRow) :
      (List.finRange slotsPerApplicationRow).filterMap row.memoryPorts =
        (List.ofFn row.memoryPorts).filterMap id := by
    simp [List.ofFn_eq_map]
  have flatPhysical :
      List.ofFn flat = List.ofFn step.physicalPorts := by
    rw [List.ofFn_congr row_slot_capacity flat]
    apply List.ofFn_inj.mpr
    funext position
    simp only [flat, CheckedStep.physicalPorts]
    congr 2
  unfold CheckedStep.rowList NormalizedRow.accesses CheckedStep.accesses
  rw [compactPayloads_eq_filterMap_ofFn]
  rw [List.flatMap_def]
  simp_rw [eachRow]
  rw [show
    List.map
        (fun row : NormalizedRow =>
          List.filterMap id (List.ofFn row.memoryPorts))
        (List.ofFn step.rows) =
      List.map (List.filterMap id)
        (List.map (fun row : NormalizedRow =>
          List.ofFn row.memoryPorts) (List.ofFn step.rows)) by
    rw [List.map_map]
    rfl]
  rw [← List.filterMap_flatten]
  rw [List.map_ofFn]
  change List.filterMap id
      ((List.ofFn fun row : ApplicationRowIndex =>
        List.ofFn fun port : RowPortIndex =>
          (step.rows row).memoryPorts port).flatten) =
    List.filterMap id (List.ofFn step.physicalPorts)
  rw [← splitRows]
  rw [flatPhysical]

theorem CheckedStep.accessCount_le_capacity (step : CheckedStep) :
    step.accesses.length ≤ slotsPerStep := by
  unfold CheckedStep.accesses compactPayloads
  exact List.length_filterMap_le _ _

/-- Refinement target for an application interpreter or generated row table.
It cannot claim an extra memory effect: the complete ordered list must equal
the list extracted from the fixed physical ports. -/
def CheckedStep.Covers
    (step : CheckedStep) (applicationAccesses : List Access) : Prop :=
  applicationAccesses = step.accesses

theorem CheckedStep.covered_access_count_le_capacity
    {step : CheckedStep} {applicationAccesses : List Access}
    (covered : step.Covers applicationAccesses) :
    applicationAccesses.length ≤ slotsPerStep := by
  rw [covered]
  exact step.accessCount_le_capacity

theorem decodeAt_some_preserves_position
    {Payload : Type}
    {ports : PhysicalPorts Payload}
    {position : Fin slotsPerStep}
    {decoded : Positioned Payload}
    (accepted : decodeAt ports position = some decoded) :
    decoded.position = position := by
  unfold decodeAt at accepted
  cases portValue : ports position with
  | none => simp [portValue] at accepted
  | some payload =>
      simp [portValue] at accepted
      subst decoded
      rfl

theorem decodeAt_some_preserves_payload
    {Payload : Type}
    {ports : PhysicalPorts Payload}
    {position : Fin slotsPerStep}
    {decoded : Positioned Payload}
    (accepted : decodeAt ports position = some decoded) :
    ports position = some decoded.payload := by
  unfold decodeAt at accepted
  cases portValue : ports position with
  | none => simp [portValue] at accepted
  | some payload =>
      simp [portValue] at accepted
      subst decoded
      rfl

end Nightstream.Protocol.Nebula.Ports
