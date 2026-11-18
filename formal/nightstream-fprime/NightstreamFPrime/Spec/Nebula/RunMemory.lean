import NightstreamFPrime.Spec.Nebula.Batch

/-! Owns the memory half of the run premise `validCircuitRun` of the Rocq proof of the WASM
relation (`formal/wasm-rocq/TraceRefinement.v`): the events of the trace replay
(`Replay`), and the events of each row have 32-bit values (`boundedMemoryEvents`). A Nebula
execution whose active ports bind the events gives both, because each access that applies is
valid (`PortAccess.Valid`), so its values are below `2^32`. It does not own the other half of
`validCircuitRun`: the rows of the relation and their continuity. -/

namespace NightstreamFPrime.Spec.Nebula.MemoryTrace

open NightstreamFPrime.Spec

/-- The memory half of Rocq `validCircuitRun`: its third and fourth parts. -/
def RunMemory (catalog : List MemorySpec) (initial : MemoryState) (rows : List Witness)
    (final : MemoryState) : Prop :=
  Replay initial (eventsForTrace catalog rows) final ∧
    ∀ w ∈ rows, boundedMemoryEvents (eventsForRow catalog w)

variable {catalog : List MemorySpec} {b : Binding}

/-- An event that a valid access binds has 32-bit values. -/
theorem eventValuesBounded_of_binds {p : Geometry} {e : Event} {a : PortAccess}
    (valid : a.Valid p) (binds : b.Binds catalog e a) : eventValuesBounded e := by
  obtain ⟨-, vr, vw, -, -⟩ := valid
  have word : ∀ n : ℕ, n < 2 ^ 32 → u32Value (n : ℤ) := fun n small =>
    ⟨Int.natCast_nonneg n, by exact_mod_cast small⟩
  cases e with
  | read memory address value =>
    obtain ⟨-, -, rfl⟩ := binds
    exact word _ vr
  | write memory address value prior =>
    obtain ⟨-, -, rfl, before⟩ := binds
    refine ⟨word _ vw, ?_⟩
    cases prior with
    | none => trivial
    | some v =>
      rw [← before v rfl]
      exact word _ vr

/-- Each access that applies is valid. -/
theorem accessAll_valid {p : Geometry} {m m' : Machine} {as : List PortAccess}
    (applies : accessAll p m as = some m') : ∀ a ∈ as, a.Valid p := by
  induction as generalizing m with
  | nil => simp
  | cons a rest ih =>
    have unfold : accessAll p m (a :: rest) = (access p m a).bind fun m'' =>
        accessAll p m'' rest := rfl
    rw [unfold] at applies
    unfold access at applies
    split_ifs at applies with checked
    · intro x member
      rcases List.mem_cons.mp member with rfl | later
      · exact checked.1
      · exact ih applies x later
    · cases applies

/-- Each event of `events` is an event of a reorder of `events`. -/
theorem Reorders.mem_order {events order : List Event} (reorders : Reorders events order)
    {e : Event} (member : e ∈ events) : e ∈ order := by
  obtain ⟨tagged, perm, -, rfl⟩ := reorders
  obtain ⟨i, tagged_member⟩ : ∃ i, (e, i) ∈ events.zipIdx := by
    obtain ⟨i, small, rfl⟩ := List.getElem_of_mem member
    exact ⟨i, List.mem_zipIdx_iff_getElem?.mpr (by simp [small])⟩
  exact List.mem_map.mpr ⟨_, perm.symm.subset tagged_member, rfl⟩

/-- A member of a list related to another list is related to a member of it. -/
theorem exists_of_forall₂_mem {α β : Type} {R : α → β → Prop} {xs : List α} {ys : List β}
    (related : List.Forall₂ R xs ys) {x : α} (member : x ∈ xs) : ∃ y ∈ ys, R x y := by
  induction related with
  | nil => cases member
  | cons head _ ih =>
    rcases List.mem_cons.mp member with rfl | later
    · exact ⟨_, List.mem_cons_self, head⟩
    · obtain ⟨y, inTail, rel⟩ := ih later
      exact ⟨y, List.mem_cons_of_mem _ inTail, rel⟩

/-- The memory half of Rocq `validCircuitRun` from Nebula: an execution from the plan images
whose active ports, step after step, bind a reorder of the events of the trace replays the
events, and each event of each row has 32-bit values. -/
theorem runMemory_of_executes {p : Plan} {σ : Type} {app : Application p.toGeometry σ}
    {s s' : σ} {final : Machine} {steps : List (List (Option PortAccess))} {initial : MemoryState}
    {rows : List Witness} {order : List Event}
    (exec : Executes p.toGeometry app s (initialMachine p) steps s' final)
    (disjoint : b.Disjoint catalog)
    (inRange : ∀ e ∈ eventsForTrace catalog rows, b.InRange e.memory e.address)
    (start : b.Initial catalog p initial) (reorders : Reorders (eventsForTrace catalog rows) order)
    (bound : List.Forall₂ (b.Binds catalog) order
      (steps.flatMap fun ports => ports.filterMap id)) :
    ∃ last, RunMemory catalog initial rows last := by
  obtain ⟨last, replay⟩ :=
    memoryConsistent_of_executes exec disjoint inRange start reorders bound
  refine ⟨last, replay, fun w row e member => ?_⟩
  have inTrace : e ∈ eventsForTrace catalog rows := List.mem_flatMap.mpr ⟨w, row, member⟩
  obtain ⟨a, applied, binds⟩ := exists_of_forall₂_mem bound (reorders.mem_order inTrace)
  exact eventValuesBounded_of_binds (accessAll_valid (executes_accessAll exec) a applied) binds

variable {k : ℕ} {shift : ℕ → Column → Column}

/-- The memory half of Rocq `validCircuitRun` from Nebula for batched steps under one slot plan
(`memoryConsistent_of_slotOrder_executes`): the rows of the steps, in order, replay, and each
event of each row has 32-bit values. -/
theorem runMemory_of_slotOrder_executes {p : Plan} {σ : Type}
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
    ∃ last, RunMemory catalog initial rows.flatten last := by
  obtain ⟨last, replay⟩ :=
    memoryConsistent_of_slotOrder_executes exec distinct holds disjoint inRange start bound
  refine ⟨last, replay, fun w row e member => ?_⟩
  have valid := accessAll_valid (executes_accessAll exec)
  obtain ⟨group, inRows, inGroup⟩ := List.mem_flatten.mp row
  clear exec inRange start replay row
  induction holds generalizing steps with
  | nil => cases inRows
  | @cons step group' witnesses' rows' head _ ih =>
    cases bound with
    | cons slots rest =>
      rcases List.mem_cons.mp inRows with rfl | later
      · have inStep : e ∈ eventsForRow (batchCatalog catalog k shift) step := by
          have same := filter_eventsForRow_batchCatalog distinct head e.memory
          have : e ∈ (eventsForTrace catalog group).filter (·.memory == e.memory) :=
            List.mem_filter.mpr ⟨List.mem_flatMap.mpr ⟨w, inGroup, member⟩, by simp⟩
          rw [← same] at this
          exact List.mem_of_mem_filter this
        obtain ⟨a, applied, binds⟩ :=
          exists_of_forall₂_mem slots.2 ((reorders_of_slotOrder slots.1).mem_order inStep)
        exact eventValuesBounded_of_binds
          (valid a (List.mem_flatMap.mpr ⟨_, List.mem_cons_self, applied⟩)) binds
      · refine ih (bound := rest) (inRows := later) fun a member => valid a ?_
        obtain ⟨ports, inSteps, applied⟩ := List.mem_flatMap.mp member
        exact List.mem_flatMap.mpr ⟨ports, List.mem_cons_of_mem _ inSteps, applied⟩

end NightstreamFPrime.Spec.Nebula.MemoryTrace
