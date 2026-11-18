import NightstreamFPrime.Spec.Nebula.RunMemory

/-! Owns the passage from a table catalog to its logical catalog for the memory half of Rocq
`validCircuitRun`. A table catalog puts the ports of several logical memories into one table
(`table m` is the table of logical memory `m`), as the WASM catalog does for its ROMs. When each
row lists the events of each table in the order of the logical events, renamed by `table`, and
no other logical memory shares the table of a memory that the row writes, the table run gives the
logical run: the logical state reads each memory from its table. It does not own the table map or
the program images. -/

namespace NightstreamFPrime.Spec.Nebula.MemoryTrace

/-- An event with its memory renamed by `table`. -/
def Event.rename (table : ℕ → ℕ) : Event → Event
  | .read memory address value => .read (table memory) address value
  | .write memory address value prior => .write (table memory) address value prior

/-- The logical state of a table state: memory `m` holds the cells of its table. -/
def untable (table : ℕ → ℕ) (state : MemoryState) : MemoryState :=
  fun memory address => state (table memory) address

theorem Event.rename_memory (table : ℕ → ℕ) (e : Event) :
    (e.rename table).memory = table e.memory := by
  cases e <;> rfl

theorem eventValuesBounded_rename {table : ℕ → ℕ} {e : Event} :
    eventValuesBounded (e.rename table) ↔ eventValuesBounded e := by
  cases e <;> rfl

private theorem untable_update {table : ℕ → ℕ} {state : MemoryState} {memory : ℕ}
    {address : Address} {value : ℤ} (alone : ∀ m, table m = table memory → m = memory) :
    untable table (update state (table memory) address value) =
      update (untable table state) memory address value := by
  funext m a
  unfold untable update
  by_cases same : m = memory
  · subst same
    simp
  · have other : table m ≠ table memory := fun shared => same (alone m shared)
    simp [same, other]

/-- A replay of renamed events is a replay of the events from the logical states, when no other
memory shares the table of a written memory. -/
theorem replay_untable {table : ℕ → ℕ} {events : List Event}
    (alone : ∀ e ∈ events, e.isWrite → ∀ m, table m = table e.memory → m = e.memory)
    {state final : MemoryState} (replay : Replay state (events.map (Event.rename table)) final) :
    Replay (untable table state) events (untable table final) := by
  induction events generalizing state with
  | nil =>
    cases replay
    exact .nil _
  | cons e rest ih =>
    cases replay with
    | cons step tail =>
      have restAlone := fun x member => alone x (List.mem_cons_of_mem e member)
      cases e with
      | read memory address value =>
        cases step with
        | read held => exact .cons (.read held) (ih restAlone tail)
      | write memory address value prior =>
        cases step with
        | write held =>
          have own : ∀ m, table m = table memory → m = memory := alone _ List.mem_cons_self rfl
          refine .cons (.write ?_) ?_
          · cases prior <;> exact held
          · rw [← untable_update own]
            exact ih restAlone tail

/-- The memory half of Rocq `validCircuitRun` for a logical catalog from the one of its table
catalog: each row lists the events of each table as the renamed logical events, in order, and
no other logical memory shares the table of a memory that a row writes. -/
theorem runMemory_untable {tables logical : List MemorySpec} {table : ℕ → ℕ}
    {initial final : MemoryState} {rows : List Witness}
    (same : ∀ w memory, (eventsForRow tables w).filter (·.memory == memory) =
      ((eventsForRow logical w).map (Event.rename table)).filter (·.memory == memory))
    (alone : ∀ w, ∀ e ∈ eventsForRow logical w, e.isWrite →
      ∀ m, table m = table e.memory → m = e.memory)
    (run : RunMemory tables initial rows final) :
    RunMemory logical (untable table initial) rows (untable table final) := by
  obtain ⟨replay, bounded⟩ := run
  have trace : ∀ memory, ((eventsForTrace logical rows).map (Event.rename table)).filter
      (·.memory == memory) = (eventsForTrace tables rows).filter (·.memory == memory) := by
    intro memory
    simp only [eventsForTrace, List.map_flatMap, List.filter_flatMap]
    exact List.flatMap_congr fun w _ => (same w memory).symm
  refine ⟨replay_untable ?_ (replay_of_reorders (reorders_of_filter_eq trace) replay), ?_⟩
  · intro e member
    obtain ⟨w, _, inRow⟩ := List.mem_flatMap.mp member
    exact alone w e inRow
  · intro w row e member
    have renamed : e.rename table ∈ (eventsForRow tables w).filter (·.memory == table e.memory) := by
      rw [same]
      exact List.mem_filter.mpr ⟨List.mem_map_of_mem member, by simp [Event.rename_memory]⟩
    exact eventValuesBounded_rename.mp (bounded w row _ (List.mem_of_mem_filter renamed))

end NightstreamFPrime.Spec.Nebula.MemoryTrace
