import NightstreamFPrime.Spec.Nebula.SlotPlan

/-! Owns batched catalogs: one Nebula step witness that holds `k` trace rows.
The batched catalog has the memories of the catalog in order, and each memory
has the ports of row 0, then of row 1, up to row `k - 1`, with the columns of
row `i` mapped by `shift i`. So a batched step lists its events memory by
memory, and its rows list theirs row by row. Each memory's events come in the
same order in both lists when the memory ids are distinct, so the two lists
are reorders of each other, and batched steps are memory consistent exactly
when their rows are. The batched bridge gives `MemoryConsistent` for the rows
from a Nebula execution of the batched steps. It does not own the choice of
`shift`, the Rust batched catalog, or the Rust rows. -/

namespace NightstreamFPrime.Spec.Nebula.MemoryTrace

open NightstreamFPrime.Spec

/-- The prior-value column of a port kind, if any. -/
def PortKind.columns : PortKind → List Column
  | .read => []
  | .write prior => prior.toList

/-- A port kind with its prior-value column mapped by `f`. -/
def PortKind.shift (f : Column → Column) : PortKind → PortKind
  | .read => .read
  | .write prior => .write (prior.map f)

/-- The activation column, if any. -/
def Activation.columns : Activation → List Column
  | .always => []
  | .when column => [column]
  | .«unless» column => [column]

/-- An activation with its column mapped by `f`. -/
def Activation.shift (f : Column → Column) : Activation → Activation
  | .always => .always
  | .when column => .when (f column)
  | .«unless» column => .«unless» (f column)

/-- The columns that a port reads: address, value, prior value and
activation. -/
def PortSpec.columns (port : PortSpec) : List Column :=
  port.addressColumns ++ port.valueColumn :: port.portKind.columns ++ port.activation.columns

/-- A port with each column mapped by `f`. -/
def PortSpec.shift (f : Column → Column) (port : PortSpec) : PortSpec where
  addressColumns := port.addressColumns.map f
  valueColumn := f port.valueColumn
  portKind := port.portKind.shift f
  activation := port.activation.shift f

/-- The catalog of a step witness that holds `k` rows: the memories of
`catalog` in order, each with the ports of row 0, then of row 1, up to row
`k - 1`, each row's ports in port order with their columns mapped by
`shift i`. -/
def batchCatalog (catalog : List MemorySpec) (k : ℕ) (shift : ℕ → Column → Column) :
    List MemorySpec :=
  catalog.map fun m =>
    { m with ports := (List.range k).flatMap fun i => m.ports.map (PortSpec.shift (shift i)) }

/-- A step witness holds `k` rows under `shift`: for each `i < k`, each column
`c` that a port of `catalog` reads has at `shift i c` the value of `rows[i]`
at `c`. -/
def HoldsRows (catalog : List MemorySpec) (k : ℕ) (shift : ℕ → Column → Column)
    (step : Witness) (rows : List Witness) : Prop :=
  rows.length = k ∧ ∀ i (hi : i < rows.length), ∀ m ∈ catalog, ∀ port ∈ m.ports,
    ∀ c ∈ port.columns, step (shift i c) = rows[i] c

/-- A mapped port in a witness that holds a row's columns at the mapped
columns has the event of the port in the row. -/
theorem portEvent_shift {f : Column → Column} {step w : Witness} {port : PortSpec} (memory : ℕ)
    (agree : ∀ c ∈ port.columns, step (f c) = w c) :
    portEvent memory step (port.shift f) = portEvent memory w port := by
  obtain ⟨address, value, kind, activation⟩ := port
  have active_eq : active step (activation.shift f) = active w activation := by
    cases activation <;>
      simp [active, Activation.shift, agree, PortSpec.columns, Activation.columns]
  have address_eq : (address.map f).map step = address.map w := by
    rw [List.map_map]
    exact List.map_congr_left fun c member => agree c (by simp [PortSpec.columns, member])
  have value_eq : step (f value) = w value := agree value (by simp [PortSpec.columns])
  cases kind with
  | read => simp [portEvent, PortSpec.shift, PortKind.shift, active_eq, address_eq, value_eq]
  | write prior =>
    have prior_eq : (prior.map f).map step = prior.map w := by
      cases prior <;> simp [agree, PortSpec.columns, PortKind.columns]
    simp [portEvent, PortSpec.shift, PortKind.shift, active_eq, address_eq, value_eq, prior_eq]

/-- The event of a port of a memory is an event of that memory. -/
theorem portEvent_memory {memory : ℕ} {w : Witness} {port : PortSpec} {e : Event}
    (shown : portEvent memory w port = some e) : e.memory = memory := by
  obtain ⟨address, value, kind, activation⟩ := port
  unfold portEvent at shown
  split_ifs at shown
  cases kind <;> (cases shown; rfl)

/-- The events of one memory in a row: a filter by a memory keeps all or
none. -/
private theorem filter_portEvents (m : MemorySpec) (w : Witness) (memory : ℕ) :
    (m.ports.filterMap (portEvent m.memoryId w)).filter (·.memory == memory) =
      if m.memoryId = memory then m.ports.filterMap (portEvent m.memoryId w) else [] := by
  have own : ∀ e ∈ m.ports.filterMap (portEvent m.memoryId w), e.memory = m.memoryId :=
    fun e member => by
      obtain ⟨port, -, shown⟩ := List.mem_filterMap.mp member
      exact portEvent_memory shown
  split_ifs with same
  · exact List.filter_eq_self.mpr fun e member => by simp [own e member, same]
  · exact List.filter_eq_nil_iff.mpr fun e member => by simp [own e member, same]

private theorem flatMap_if {α β : Type} (l : List α) (c : Prop) [Decidable c] (g : α → List β) :
    (l.flatMap fun a => if c then g a else []) = if c then l.flatMap g else [] := by
  split_ifs <;> simp

private theorem flatMap_ite {α β : Type} (l : List α) (p : α → Prop) [DecidablePred p]
    (f : α → List β) :
    (l.flatMap fun a => if p a then f a else []) = (l.filter fun a => decide (p a)).flatMap f := by
  induction l with
  | nil => rfl
  | cons a l ih => by_cases h : p a <;> simp [h, ih]

/-- With distinct memory ids, at most one memory has a given id. -/
private theorem length_filter_le {catalog : List MemorySpec}
    (distinct : (catalog.map MemorySpec.memoryId).Nodup) (memory : ℕ) :
    (catalog.filter fun m => decide (m.memoryId = memory)).length ≤ 1 := by
  have nodup := ((List.filter_sublist (p := fun m => decide (m.memoryId = memory))
    (l := catalog)).map MemorySpec.memoryId).nodup distinct
  rw [List.eq_replicate_iff.mpr ⟨List.length_map _, fun id named => by
    obtain ⟨m, member, rfl⟩ := List.mem_map.mp named
    simpa using (List.mem_filter.mp member).2⟩, List.nodup_replicate] at nodup
  exact nodup

variable {catalog : List MemorySpec} {k : ℕ} {shift : ℕ → Column → Column}

/-- A batched step and its rows have each memory's events in the same
order. -/
theorem filter_eventsForRow_batchCatalog (distinct : (catalog.map MemorySpec.memoryId).Nodup)
    {step : Witness} {rows : List Witness} (holds : HoldsRows catalog k shift step rows)
    (memory : ℕ) :
    (eventsForRow (batchCatalog catalog k shift) step).filter (·.memory == memory) =
      (eventsForTrace catalog rows).filter (·.memory == memory) := by
  obtain ⟨rfl, agree⟩ := holds
  have batched : ∀ m ∈ catalog,
      (((List.range rows.length).flatMap fun i => m.ports.map (PortSpec.shift (shift i))).filterMap
        (portEvent m.memoryId step)) =
        rows.flatMap fun row => m.ports.filterMap (portEvent m.memoryId row) := by
    intro m member
    rw [List.filterMap_flatMap, List.flatMap_def, List.flatMap_def]
    congr 1
    refine List.ext_getElem (by simp) fun i small small' => ?_
    simp only [List.getElem_map, List.getElem_range, List.filterMap_map]
    exact List.filterMap_congr fun port inPorts =>
      portEvent_shift _ (agree i (by simpa using small') m member port inPorts)
  unfold eventsForTrace
  simp only [eventsForRow_eq, batchCatalog, List.flatMap_map]
  rw [List.flatMap_congr batched]
  simp only [List.filter_flatMap, filter_portEvents, flatMap_if]
  simp only [flatMap_ite]
  have single := length_filter_le distinct memory
  generalize catalog.filter (fun m => decide (m.memoryId = memory)) = named at single ⊢
  rcases named with _ | ⟨m, _ | ⟨m', rest⟩⟩
  · simp
  · simp
  · simp at single

/-- The events of a batched step and the events of its rows are reorders of
each other. -/
theorem reorders_batchCatalog (distinct : (catalog.map MemorySpec.memoryId).Nodup)
    {step : Witness} {rows : List Witness} (holds : HoldsRows catalog k shift step rows) :
    Reorders (eventsForTrace catalog rows) (eventsForRow (batchCatalog catalog k shift) step) ∧
      Reorders (eventsForRow (batchCatalog catalog k shift) step) (eventsForTrace catalog rows) :=
  have same := filter_eventsForRow_batchCatalog distinct holds
  ⟨reorders_of_filter_eq fun memory => (same memory).symm, reorders_of_filter_eq same⟩

/-- Batched steps and their rows, in order, have each memory's events in the
same order. -/
theorem filter_eventsForTrace_batchCatalog (distinct : (catalog.map MemorySpec.memoryId).Nodup)
    {steps : List Witness} {rows : List (List Witness)}
    (holds : List.Forall₂ (HoldsRows catalog k shift) steps rows) (memory : ℕ) :
    (eventsForTrace (batchCatalog catalog k shift) steps).filter (·.memory == memory) =
      (eventsForTrace catalog rows.flatten).filter (·.memory == memory) := by
  induction holds with
  | nil => rfl
  | cons head _ ih =>
    simp only [eventsForTrace, List.flatMap_cons, List.flatten_cons, List.flatMap_append,
      List.filter_append] at ih ⊢
    rw [ih, filter_eventsForRow_batchCatalog distinct head]
    rfl

/-- Batched steps are memory consistent exactly when their rows, in order,
are. -/
theorem memoryConsistent_batchCatalog_iff (distinct : (catalog.map MemorySpec.memoryId).Nodup)
    {initial : MemoryState} {steps : List Witness} {rows : List (List Witness)}
    (holds : List.Forall₂ (HoldsRows catalog k shift) steps rows) :
    MemoryConsistent (batchCatalog catalog k shift) initial steps ↔
      MemoryConsistent catalog initial rows.flatten := by
  have same := filter_eventsForTrace_batchCatalog distinct holds
  constructor
  · rintro ⟨final, replay⟩
    exact ⟨final, replay_of_reorders (reorders_of_filter_eq fun m => (same m).symm) replay⟩
  · rintro ⟨final, replay⟩
    exact ⟨final, replay_of_reorders (reorders_of_filter_eq same) replay⟩

/-- A batched catalog has the ids and kinds of its catalog, so the same
namespaces. -/
theorem isRam_batchCatalog : isRam (batchCatalog catalog k shift) = isRam catalog := by
  funext memory
  simp [isRam, batchCatalog, List.any_map, Function.comp_def]

variable {b : Binding}

/-- Nebula's memory guarantee for batched steps: an execution from the plan
images whose active ports, step after step, bind a reorder of the events of
each batched step gives `MemoryConsistent` for the rows of the steps, in
order. -/
theorem memoryConsistent_of_batched_executes {p : Plan} {σ : Type}
    {app : Application p.toGeometry σ} {s s' : σ} {final : Machine}
    {steps : List (List (Option PortAccess))} {initial : MemoryState} {witnesses : List Witness}
    {rows : List (List Witness)}
    (exec : Executes p.toGeometry app s (initialMachine p) steps s' final)
    (distinct : (catalog.map MemorySpec.memoryId).Nodup)
    (holds : List.Forall₂ (HoldsRows catalog k shift) witnesses rows)
    (disjoint : b.Disjoint catalog)
    (inRange : ∀ e ∈ eventsForTrace catalog rows.flatten, b.InRange e.memory e.address)
    (start : b.Initial catalog p initial)
    (bound : List.Forall₂ (fun step ports => ∃ order,
      Reorders (eventsForRow (batchCatalog catalog k shift) step) order ∧
        List.Forall₂ (b.Binds catalog) order (ports.filterMap id)) witnesses steps) :
    MemoryConsistent catalog initial rows.flatten := by
  obtain ⟨order, reorders, bound⟩ : ∃ order,
      Reorders (eventsForTrace (batchCatalog catalog k shift) witnesses) order ∧
        List.Forall₂ (b.Binds catalog) order (steps.flatMap fun ports => ports.filterMap id) := by
    clear exec holds
    induction bound with
    | nil => exact ⟨[], Reorders.nil, .nil⟩
    | cons head _ ih =>
      obtain ⟨order, reorders, bound⟩ := head
      obtain ⟨order', reorders', bound'⟩ := ih
      exact ⟨order ++ order', reorders.append reorders', List.rel_append bound bound'⟩
  have same := isRam_batchCatalog (catalog := catalog) (k := k) (shift := shift)
  have binds : b.Binds (batchCatalog catalog k shift) = b.Binds catalog := by
    funext e a
    cases e <;> simp only [Binding.Binds, Binding.At, same]
  refine (memoryConsistent_batchCatalog_iff distinct holds).mp
    (memoryConsistent_of_executes (catalog := batchCatalog catalog k shift) exec ?_ ?_ ?_ reorders
      (binds ▸ bound))
  · unfold Binding.Disjoint
    rw [same]
    exact disjoint
  · intro e member
    refine inRange e (List.mem_of_mem_filter (p := (·.memory == e.memory)) ?_)
    rw [← filter_eventsForTrace_batchCatalog distinct holds]
    exact List.mem_filter.mpr ⟨member, by simp⟩
  · unfold Binding.Initial
    rw [same]
    exact start

/-- Nebula's memory guarantee for batched steps under one slot plan: an
execution from the plan images, where in each batched step the plan obeys
rules 5, 1 and 2 and the active ports bind the slot events, gives
`MemoryConsistent` for the rows of the steps, in order. -/
theorem memoryConsistent_of_slotOrder_executes {p : Plan} {σ : Type}
    {app : Application p.toGeometry σ} {s s' : σ} {final : Machine}
    {steps : List (List (Option PortAccess))} {initial : MemoryState} {witnesses : List Witness}
    {rows : List (List Witness)} {plan : List (List ℕ)}
    (exec : Executes p.toGeometry app s (initialMachine p) steps s' final)
    (distinct : (catalog.map MemorySpec.memoryId).Nodup)
    (holds : List.Forall₂ (HoldsRows catalog k shift) witnesses rows)
    (disjoint : b.Disjoint catalog)
    (inRange : ∀ e ∈ eventsForTrace catalog rows.flatten, b.InRange e.memory e.address)
    (start : b.Initial catalog p initial)
    (bound : List.Forall₂ (fun step ports => SlotOrder (batchCatalog catalog k shift) step plan ∧
      List.Forall₂ (b.Binds catalog) (slotEvents (batchCatalog catalog k shift) step plan)
        (ports.filterMap id)) witnesses steps) :
    MemoryConsistent catalog initial rows.flatten :=
  memoryConsistent_of_batched_executes exec distinct holds disjoint inRange start
    (bound.imp fun _ _ slots => ⟨_, reorders_of_slotOrder slots.1, slots.2⟩)

end NightstreamFPrime.Spec.Nebula.MemoryTrace
