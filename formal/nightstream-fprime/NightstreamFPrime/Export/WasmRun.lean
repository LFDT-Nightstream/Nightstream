import NightstreamFPrime.Export.WasmBinding
import NightstreamFPrime.Spec.Nebula.TableOverlay
import NightstreamFPrime.Lifecycle.Nebula.PortApp
import NightstreamFPrime.Lifecycle.Nebula.Framing
import NightstreamFPrime.Spec.Nebula.Soundness

/-! Owns the memory half of Rocq `validCircuitRun` for a run of the WASM application of an
accepted binding export (`Export.WasmBinding`), over the physical and the logical catalog, and
the proposed WASM premise of the memory claims (`WasmModel`, `decisions/wasm-rocq-premise.md`).
It does not own the Rust rows that give the rules, the rows of the WASM relation, or the program
images. -/

namespace NightstreamFPrime.Export.WasmBinding

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Nebula
open NightstreamFPrime.Spec.Nebula.MemoryTrace

variable (x : WasmBinding)

/-- The active bound ports of a slot in a step. -/
def activeIn (step : Witness) (slot : List BoundPort) : List BoundPort :=
  slot.filter fun q => (rankEvent x.batched step q.rank).isSome

/-- SPEC.md section 7.6 rules 1, 3 and 4 at one slot: no bound port of the slot is active and the
slot is a pad, or one bound port is active and the access of the slot obeys its equations. -/
def SlotBinds (step : Witness) (slot : List BoundPort) : Option PortAccess → Prop
  | none => x.activeIn step slot = []
  | some a => ∃ q, x.activeIn step slot = [q] ∧ x.SlotEquations q step a

/-- SPEC.md section 7.6 rules 1 to 4 in one step: each slot binds its access, and of two
dependent active ports in distinct slots the port of the earlier slot has no higher rank. -/
structure StepBinds (step : Witness) (ports : List (Option PortAccess)) : Prop where
  slots : List.Forall₂ (x.SlotBinds step) x.slots ports
  order : x.plan.Pairwise fun earlier later => ∀ r ∈ earlier, ∀ r' ∈ later,
    ∀ e ∈ rankEvent x.batched step r, ∀ e' ∈ rankEvent x.batched step r', e.Dependent e' → r ≤ r'

/-- The trace rows of step witnesses: the rows of each step, in order. -/
def traceRows (witnesses : List Witness) : List Witness :=
  witnesses.flatMap fun step => (List.range x.rows).map (x.rowOf step)

/-- The table of a logical memory; a memory past the logical catalog gets a table past the
physical catalog. -/
def tableOf (m : ℕ) : ℕ := x.table.getD m x.catalog.length

private theorem filterMap_eq_filter {α β : Type} (f : α → Option β) (l : List α) :
    l.filterMap f = (l.filter fun a => (f a).isSome).filterMap f := by
  induction l with
  | nil => rfl
  | cons a l ih =>
    cases h : f a with
    | none => rw [List.filterMap_cons_none h, List.filter_cons_of_neg (by simp [h]), ih]
    | some b =>
      rw [List.filterMap_cons_some h, List.filter_cons_of_pos (by simp [h]),
        List.filterMap_cons_some h, ih]

private theorem slotEvents_eq (step : Witness) :
    slotEvents x.batched step x.plan =
      x.slots.flatMap fun slot => slot.filterMap fun q => rankEvent x.batched step q.rank := by
  simp [slotEvents, plan, List.flatMap_map, List.filterMap_map]

private theorem bound_of_slots {x : WasmBinding} (accepted : x.Accepted) {step : Witness}
    {slots : List (List BoundPort)} {ports : List (Option PortAccess)}
    (inSlots : ∀ slot ∈ slots, slot ∈ x.slots) (related : List.Forall₂ (x.SlotBinds step) slots ports) :
    List.Forall₂ (fun e a => x.binding.Binds x.catalog e a ∧ x.binding.InRange e.memory e.address)
      (slots.flatMap fun slot => slot.filterMap fun q => rankEvent x.batched step q.rank)
      (ports.filterMap id) := by
  induction related with
  | nil => exact .nil
  | @cons slot port slots ports head _ ih =>
    have rest := ih fun s member => inSlots s (List.mem_cons_of_mem _ member)
    rw [List.flatMap_cons, filterMap_eq_filter]
    change List.Forall₂ _ ((x.activeIn step slot).filterMap _ ++ _) _
    cases port with
    | none =>
      simp only [SlotBinds] at head
      rw [head, List.filterMap_nil, List.nil_append, List.filterMap_cons_none rfl]
      exact rest
    | some a =>
      obtain ⟨q, active, equations⟩ := head
      have member : q ∈ x.activeIn step slot := by rw [active]; exact List.mem_singleton_self q
      obtain ⟨inSlot, shown⟩ := List.mem_filter.mp member
      obtain ⟨e, found⟩ := Option.isSome_iff_exists.mp shown
      rw [active]
      simp only [List.filterMap_cons, List.filterMap_nil, found, id, List.singleton_append]
      exact .cons (binds_of_rankEvent accepted (inSlots slot List.mem_cons_self) inSlot found
        equations) rest

/-- In an accepted export, a step that obeys the rules has a slot plan that obeys rules 5, 1 and
2 (`SlotOrder`), and its active slots bind its slot events in range. -/
theorem slotOrder_of_stepBinds {x : WasmBinding} (accepted : x.Accepted) {step : Witness}
    {ports : List (Option PortAccess)} (binds : x.StepBinds step ports) :
    SlotOrder x.batched step x.plan ∧
      List.Forall₂ (fun e a => x.binding.Binds x.catalog e a ∧ x.binding.InRange e.memory e.address)
        (slotEvents x.batched step x.plan) (ports.filterMap id) := by
  refine ⟨⟨accepted.names, fun ranks inPlan => ?_, binds.order⟩, ?_⟩
  · obtain ⟨slot, inSlots, rfl⟩ := List.mem_map.mp inPlan
    obtain ⟨port, -, related⟩ := exists_of_forall₂_mem binds.slots inSlots
    rw [List.filter_map, List.length_map]
    change (x.activeIn step slot).length ≤ 1
    cases port with
    | none =>
      simp only [SlotBinds] at related
      simp [related]
    | some a =>
      obtain ⟨q, active, -⟩ := related
      simp [active]
  · rw [slotEvents_eq]
    exact bound_of_slots accepted (fun _ member => member) binds.slots

private theorem traceRows_eq (witnesses : List Witness) :
    x.traceRows witnesses = (witnesses.map fun step => (List.range x.rows).map (x.rowOf step)).flatten := by
  simp [traceRows, List.flatMap_def]

/-- The memory half of Rocq `validCircuitRun` over the physical catalog of an accepted export:
an execution from the plan images whose steps obey the rules at the bound ports replays the
events of the trace rows, and each event of each row has 32-bit values. -/
theorem runMemory_of_stepBinds {x : WasmBinding} (accepted : x.Accepted) {p : Plan} {σ : Type}
    {app : Application p.toGeometry σ} {s s' : σ} {final : Machine}
    {steps : List (List (Option PortAccess))} {initial : MemoryState} {witnesses : List Witness}
    (exec : Executes p.toGeometry app s (initialMachine p) steps s' final)
    (binds : List.Forall₂ x.StepBinds witnesses steps) (start : x.binding.Initial x.catalog p initial) :
    ∃ last, RunMemory x.catalog initial (x.traceRows witnesses) last := by
  have holds : List.Forall₂ (HoldsRows x.catalog x.rows x.shiftFn) witnesses
      (witnesses.map fun step => (List.range x.rows).map (x.rowOf step)) :=
    List.forall₂_map_right_iff.mpr (List.forall₂_same.mpr fun step _ => holdsRows_rowOf x step)
  have inRange : ∀ e ∈ eventsForTrace x.catalog
      (witnesses.map fun step => (List.range x.rows).map (x.rowOf step)).flatten,
      x.binding.InRange e.memory e.address := by
    intro e member
    have same := filter_eventsForTrace_batchCatalog accepted.distinct holds e.memory
    have inBatched : e ∈ (eventsForTrace x.batched witnesses).filter (·.memory == e.memory) := by
      rw [batched, same]
      exact List.mem_filter.mpr ⟨member, by simp⟩
    obtain ⟨step, inWitnesses, inStep⟩ := List.mem_flatMap.mp (List.mem_of_mem_filter inBatched)
    obtain ⟨ports, -, stepBinds⟩ := exists_of_forall₂_mem binds inWitnesses
    obtain ⟨slots, bound⟩ := slotOrder_of_stepBinds accepted stepBinds
    obtain ⟨a, -, -, found⟩ :=
      exists_of_forall₂_mem bound ((reorders_of_slotOrder slots).mem_order inStep)
    exact found
  rw [traceRows_eq]
  exact runMemory_of_slotOrder_executes (k := x.rows) (shift := x.shiftFn) exec accepted.distinct
    holds accepted.disjoint inRange start
    (binds.imp fun _ _ stepBinds => have both := slotOrder_of_stepBinds accepted stepBinds
      ⟨both.1, both.2.imp fun _ _ bound => bound.1⟩)

private theorem portEvent_rename (table : ℕ → ℕ) (m : ℕ) (w : Witness) (port : PortSpec) :
    (portEvent m w port).map (Event.rename table) = portEvent (table m) w port := by
  obtain ⟨address, value, kind, activation⟩ := port
  unfold portEvent
  split_ifs <;> cases kind <;> rfl

private theorem filter_portEvents (t P : ℕ) (w : Witness) (ports : List PortSpec) :
    (ports.filterMap (portEvent t w)).filter (·.memory == P) =
      if t = P then ports.filterMap (portEvent t w) else [] := by
  split_ifs with same
  · refine List.filter_eq_self.mpr fun e member => ?_
    obtain ⟨port, -, shown⟩ := List.mem_filterMap.mp member
    simp [portEvent_memory shown, same]
  · refine List.filter_eq_nil_iff.mpr fun e member => ?_
    obtain ⟨port, -, shown⟩ := List.mem_filterMap.mp member
    simp [portEvent_memory shown, same]

private theorem members_events (pairs : List (MemorySpec × ℕ)) (P : ℕ) (w : Witness) :
    (pairs.flatMap fun e => e.1.ports.filterMap (portEvent e.2 w)).filter (·.memory == P) =
      ((pairs.filterMap fun e => if e.2 = P then some e.1 else none).flatMap
        MemorySpec.ports).filterMap (portEvent P w) := by
  induction pairs with
  | nil => rfl
  | cons e pairs ih =>
    rw [List.flatMap_cons, List.filter_append, ih, filter_portEvents]
    by_cases same : e.2 = P
    · simp [same, List.filterMap_append]
    · simp [same]

private theorem lt_count {table : List ℕ} {t : ℕ} (member : t ∈ table) :
    t < table.foldr (fun t count => max (t + 1) count) 0 := by
  induction table with
  | nil => cases member
  | cons a table ih =>
    rcases List.mem_cons.mp member with rfl | later
    · simp
    · simp only [List.foldr_cons]
      exact lt_max_of_lt_right (ih later)

private theorem range_flatMap_ite (n P : ℕ) (f : ℕ → List Event) :
    (List.range n).flatMap (fun t => if t = P then f t else []) = if P < n then f P else [] := by
  induction n with
  | zero => simp
  | succ n ih =>
    rw [List.range_succ, List.flatMap_append, ih]
    by_cases below : P < n
    · simp [below, Nat.lt_succ_of_lt below, Nat.ne_of_gt below]
    · by_cases top : P = n
      · subst top
        simp
      · simp [below, top, Nat.le_iff_lt_or_eq, Ne.symm top]

private theorem renamed_events {logical : List MemorySpec} {table : List ℕ} {tbl : ℕ → ℕ}
    (paired : List.Forall₂ (fun L t => tbl L.memoryId = t) logical table) (w : Witness) :
    (eventsForRow logical w).map (Event.rename tbl) =
      (logical.zip table).flatMap fun e => e.1.ports.filterMap (portEvent e.2 w) := by
  rw [eventsForRow_eq]
  induction paired with
  | nil => rfl
  | @cons L t logical table head _ ih =>
    simp only [List.flatMap_cons, List.map_append, List.zip_cons_cons] at ih ⊢
    rw [ih, List.map_filterMap]
    congr 1
    refine List.filterMap_congr fun port _ => ?_
    rw [portEvent_rename, head]

/-- In an accepted export, each row lists the events of each table of the physical catalog as
the logical events renamed by `tableOf`, in order. -/
theorem filter_eventsForRow_logical {x : WasmBinding} (accepted : x.Accepted) (w : Witness)
    (P : ℕ) : (eventsForRow x.catalog w).filter (·.memory == P) =
      ((eventsForRow x.logical w).map (Event.rename x.tableOf)).filter (·.memory == P) := by
  have lengths := accepted.tableKinds.length_eq
  have ids : ∀ i (h : i < x.logical.length), x.logical[i].memoryId = i := fun i h => by
    have := congrArg (·[i]?) accepted.logicalIds
    simpa [h] using this
  have paired : List.Forall₂ (fun L t => x.tableOf L.memoryId = t) x.logical x.table := by
    refine List.forall₂_iff_get.mpr ⟨lengths, fun i h₁ h₂ => ?_⟩
    simp [ids i h₁, tableOf, List.getD_eq_getElem?_getD, h₂]
  rw [renamed_events paired, members_events, accepted.isOverlay, eventsForRow_eq]
  simp only [overlay, List.flatMap_map, List.filter_flatMap, filter_portEvents]
  rw [range_flatMap_ite]
  split_ifs with below
  · rfl
  · symm
    convert List.filterMap_nil (f := portEvent P w)
    rw [List.flatMap_eq_nil_iff]
    intro L member
    obtain ⟨e, inZip, found⟩ := List.mem_filterMap.mp member
    split_ifs at found with same
    exact absurd (same ▸ lt_count (List.of_mem_zip inZip).2) below

private theorem write_kind {m : ℕ} {w : Witness} {port : PortSpec} {e : Event}
    (shown : portEvent m w port = some e) (write : e.isWrite = true) : port.portKind ≠ .read := by
  obtain ⟨address, value, kind, activation⟩ := port
  unfold portEvent at shown
  split_ifs at shown
  cases kind with
  | read => cases shown; cases write
  | write => exact fun h => by cases h

/-- In an accepted export, no other logical memory has the table of a memory that a row
writes. -/
theorem written_alone {x : WasmBinding} (accepted : x.Accepted) (w : Witness) :
    ∀ e ∈ eventsForRow x.logical w, e.isWrite →
      ∀ m, x.tableOf m = x.tableOf e.memory → m = e.memory := by
  intro e member write m same
  rw [eventsForRow_eq] at member
  obtain ⟨L, inLogical, inL⟩ := List.mem_flatMap.mp member
  obtain ⟨port, inPorts, shown⟩ := List.mem_filterMap.mp inL
  rw [portEvent_memory shown] at same ⊢
  have lengths := accepted.tableKinds.length_eq
  obtain ⟨i, hi, rfl⟩ := List.getElem_of_mem inLogical
  have ids : ∀ j (h : j < x.logical.length), x.logical[j].memoryId = j := fun j h => by
    have := congrArg (·[j]?) accepted.logicalIds
    simpa [h] using this
  have pairI : (x.logical[i], x.table[i]'(lengths ▸ hi)) ∈ x.logical.zip x.table :=
    List.mem_iff_getElem.mpr ⟨i, by simp [hi, ← lengths], by simp⟩
  have alone := accepted.writesAlone _ pairI ⟨port, inPorts, write_kind shown write⟩
  rw [ids i hi] at same ⊢
  have tableI : x.tableOf i = x.table[i]'(lengths ▸ hi) := by
    simp [tableOf, List.getD_eq_getElem?_getD, lengths ▸ hi]
  by_cases small : m < x.table.length
  · have pairM : (x.logical[m]'(lengths ▸ small), x.table[m]) ∈ x.logical.zip x.table :=
      List.mem_iff_getElem.mpr ⟨m, by simp [small, lengths], by simp⟩
    have tableM : x.tableOf m = x.table[m] := by simp [tableOf, List.getD_eq_getElem?_getD, small]
    have := alone _ pairM (by rw [← tableM, ← tableI, same])
    simpa [ids m (lengths ▸ small), ids i hi] using this
  · have past : x.tableOf m = x.catalog.length := by
      simp [tableOf, List.getD_eq_getElem?_getD, Nat.le_of_not_lt small]
    have inside : x.table[i]'(lengths ▸ hi) < x.catalog.length := by
      rw [accepted.isOverlay, overlay, List.length_map, List.length_range]
      exact lt_count (List.getElem_mem _)
    omega

/-- The memory half of Rocq `validCircuitRun` over the logical catalog of an accepted export
(`x.logical`, the catalog of Rocq `MemoryCatalog.memories`): an execution from the plan images
whose steps obey the rules at the bound ports replays the logical events of the trace rows from
the state that reads each logical memory from its table, and each event of each row has 32-bit
values. -/
theorem runMemory_logical_of_stepBinds {x : WasmBinding} (accepted : x.Accepted) {p : Plan}
    {σ : Type} {app : Application p.toGeometry σ} {s s' : σ} {final : Machine}
    {steps : List (List (Option PortAccess))} {initial : MemoryState} {witnesses : List Witness}
    (exec : Executes p.toGeometry app s (initialMachine p) steps s' final)
    (binds : List.Forall₂ x.StepBinds witnesses steps) (start : x.binding.Initial x.catalog p initial) :
    ∃ last, RunMemory x.logical (untable x.tableOf initial) (x.traceRows witnesses) last := by
  obtain ⟨last, run⟩ := runMemory_of_stepBinds accepted exec binds start
  exact ⟨_, runMemory_untable (filter_eventsForRow_logical accepted) (written_alone accepted) run⟩

/-- The proposed premise of the WASM case of `app.*` (`decisions/wasm-rocq-premise.md`): each
execution of the model of the WASM application `app` from the plan images has one step witness
for each step, whose trace rows satisfy `circuit` (the first two parts of Rocq `validCircuitRun`:
each row satisfies the WASM relation, and the rows chain) and whose slots obey SPEC.md section
7.6 rules 1 to 4 at the bound ports of the export `x`. -/
def WasmModel (p : Plan) (circuit : List Witness → Prop)
    (app : Lifecycle.Nebula.PortApp p.toGeometry) : Prop :=
  ∀ {s₀ s : app.State} {steps : List (List (Option PortAccess))} {final : Machine},
    Executes p.toGeometry app.model s₀ (initialMachine p) steps s final →
      ∃ witnesses, circuit (x.traceRows witnesses) ∧ List.Forall₂ x.StepBinds witnesses steps

/-- Under the proposed premise, an execution of the WASM application from the plan images gives
the premise `validCircuitRun` of Rocq `run_refinement` with `circuit` for its first two parts:
trace rows that satisfy `circuit`, whose logical events replay from the state that reads each
logical memory from its table, with 32-bit values. -/
theorem circuitRun_of_wasmModel {x : WasmBinding} (accepted : x.Accepted) {p : Plan}
    {circuit : List Witness → Prop} {app : Lifecycle.Nebula.PortApp p.toGeometry}
    (model : x.WasmModel p circuit app) {initial : MemoryState}
    (start : x.binding.Initial x.catalog p initial) {s₀ s : app.State}
    {steps : List (List (Option PortAccess))} {final : Machine}
    (exec : Executes p.toGeometry app.model s₀ (initialMachine p) steps s final) :
    ∃ witnesses last, circuit (x.traceRows witnesses) ∧
      RunMemory x.logical (untable x.tableOf initial) (x.traceRows witnesses) last := by
  obtain ⟨witnesses, holds, binds⟩ := model exec
  obtain ⟨last, run⟩ := runMemory_logical_of_stepBinds accepted exec binds start
  exact ⟨witnesses, last, holds, run⟩

/-- The join with the memory claims: outside the event that `Claims.Nebula.memory_sound` bounds,
the extracted run attests the claim (`Lifecycle.Nebula.EndToEnd.Experiment.Attested`). Under the
proposed premise, a run that attests a statement gives the premise `validCircuitRun` of Rocq
`run_refinement` with `circuit` for its first two parts. -/
theorem circuitRun_of_attests {x : WasmBinding} (accepted : x.Accepted) {p : Plan}
    {circuit : List Witness → Prop} {app : Lifecycle.Nebula.PortApp p.toGeometry}
    (model : x.WasmModel p circuit app) {initial : MemoryState}
    (start : x.binding.Initial x.catalog p initial) {stmt run}
    (attests : Attests (Lifecycle.Nebula.context p) app.model stmt run) :
    ∃ witnesses last, circuit (x.traceRows witnesses) ∧
      RunMemory x.logical (untable x.tableOf initial) (x.traceRows witnesses) last := by
  obtain ⟨_, exec, -, -⟩ := attests.executes
  exact circuitRun_of_wasmModel accepted model start exec

end NightstreamFPrime.Export.WasmBinding
