import NightstreamFPrime.Spec.Nebula.MemoryTrace
import NightstreamFPrime.Spec.Nebula.Reference

/-! Owns the binding of catalog memory events to Nebula port accesses (SPEC.md
section 7.6) and the bridge from a Nebula execution to `MemoryConsistent`. A binding
gives each bound memory a region in its namespace: RAM for a memory that the
catalog declares RAM, ROM otherwise. An access binds an event when it names
the event's packed address, a read reads the event value, and a write writes
the event value and reads the prior value when the event has one. An inactive
port has no event, so it has no access: a pad slot. The bound accesses list the
events in an order that keeps each pair of one memory with a write
(`Reorders`). It does not own the slot layout of a step or the Rust
rows. -/

namespace NightstreamFPrime.Spec.Nebula.MemoryTrace

open NightstreamFPrime.Spec

/-- Little-endian packing of address parts with bit widths `ws`:
`x₀ + 2^w₀ · (x₁ + 2^w₁ · …)`. -/
def packParts : List ℕ → Address → ℕ
  | w :: ws, x :: xs => x.toNat + 2 ^ w * packParts ws xs
  | _, _ => 0

/-- A memory's region in its namespace: the first address and the bit width of
each address part. -/
structure Region where
  base : ℕ
  widths : List ℕ

namespace Region

/-- The addresses of a region: one part per width, each in `[0, 2^w)`. -/
def Fits (r : Region) (address : Address) : Prop :=
  List.Forall₂ (fun x w => 0 ≤ x ∧ x < 2 ^ w) address r.widths

/-- The number of addresses of a region. -/
def size (r : Region) : ℕ := 2 ^ r.widths.sum

/-- The namespace address of an address of the region. -/
def addr (r : Region) (address : Address) : ℕ := r.base + packParts r.widths address

end Region

/-- The namespace of a memory: RAM when the catalog declares it RAM. -/
def isRam (catalog : List MemorySpec) (memory : ℕ) : Bool :=
  catalog.any fun m => m.memoryId == memory && m.memoryKind == .ram

/-- An address packing config: the region of each bound memory. -/
structure Binding where
  region : ℕ → Option Region

/-- The memory of an event. -/
def Event.memory : Event → ℕ
  | .read memory _ _ | .write memory _ _ _ => memory

/-- The address of an event. -/
def Event.address : Event → Address
  | .read _ address _ | .write _ address _ _ => address

namespace Binding

variable (catalog : List MemorySpec) (b : Binding)

/-- Distinct bound memories of one namespace have disjoint regions. -/
def Disjoint : Prop :=
  ∀ i j ri rj, b.region i = some ri → b.region j = some rj → i ≠ j →
    isRam catalog i = isRam catalog j →
      ri.base + ri.size ≤ rj.base ∨ rj.base + rj.size ≤ ri.base

/-- An address in range: its memory is bound and its parts fit the region. -/
def InRange (memory : ℕ) (address : Address) : Prop :=
  ∃ r, b.region memory = some r ∧ r.Fits address

/-- The initial state holds the plan images on the in-range set: an address of
a RAM memory starts at the RAM image word of its region address, and an
address of a ROM memory at the ROM image word. -/
def Initial (p : Plan) (initial : MemoryState) : Prop :=
  ∀ memory r address, b.region memory = some r → r.Fits address →
    initial memory address = some
      ((if isRam catalog memory then p.ram (r.addr address) else p.rom (r.addr address) : ℕ) : ℤ)

/-- An access at an address of a memory: the memory's namespace and the region
address. -/
def At (memory : ℕ) (address : Address) (a : PortAccess) : Prop :=
  ∃ r, b.region memory = some r ∧ a.isRam = isRam catalog memory ∧ a.addr = r.addr address

/-- An access binds an event: it is at the event's address, a read reads the
event value, and a write writes the event value and reads the prior value when
the event has one. -/
def Binds : Event → PortAccess → Prop
  | .read memory address value, a =>
    b.At catalog memory address a ∧ a.isWrite = false ∧ (a.vr : ℤ) = value
  | .write memory address value prior, a =>
    b.At catalog memory address a ∧ a.isWrite = true ∧ (a.vw : ℤ) = value ∧
      ∀ v ∈ prior, (a.vr : ℤ) = v

end Binding

/-! ### Packing -/

/-- Parts that fit pack below the region size. -/
theorem packParts_lt {ws : List ℕ} {xs : Address}
    (fits : List.Forall₂ (fun x w => 0 ≤ x ∧ x < 2 ^ w) xs ws) :
    packParts ws xs < 2 ^ ws.sum := by
  induction fits with
  | nil => simp [packParts]
  | @cons x w xs ws head _ ih =>
    obtain ⟨n, rfl⟩ := Int.eq_ofNat_of_zero_le head.1
    have small : n < 2 ^ w := by exact_mod_cast head.2
    rw [packParts, Int.toNat_natCast, List.sum_cons, Nat.pow_add]
    calc n + 2 ^ w * packParts ws xs < 2 ^ w * (packParts ws xs + 1) := by
          rw [Nat.mul_add, Nat.mul_one]; omega
      _ ≤ 2 ^ w * 2 ^ ws.sum := Nat.mul_le_mul_left _ ih

/-- Packing is injective on parts that fit. -/
theorem packParts_injective {ws : List ℕ} {xs ys : Address}
    (fitsX : List.Forall₂ (fun x w => 0 ≤ x ∧ x < 2 ^ w) xs ws)
    (fitsY : List.Forall₂ (fun x w => 0 ≤ x ∧ x < 2 ^ w) ys ws)
    (same : packParts ws xs = packParts ws ys) : xs = ys := by
  induction fitsX generalizing ys with
  | nil => cases fitsY; rfl
  | @cons x w xs ws head _ ih =>
    cases fitsY with
    | @cons y _ ys _ headY tailY =>
      obtain ⟨n, rfl⟩ := Int.eq_ofNat_of_zero_le head.1
      obtain ⟨k, rfl⟩ := Int.eq_ofNat_of_zero_le headY.1
      have hn : n < 2 ^ w := by exact_mod_cast head.2
      have hk : k < 2 ^ w := by exact_mod_cast headY.2
      simp only [packParts, Int.toNat_natCast] at same
      have low : n = k := by
        have := congrArg (· % 2 ^ w) same
        simpa [Nat.add_mul_mod_self_left, Nat.mod_eq_of_lt hn, Nat.mod_eq_of_lt hk] using this
      have high : packParts ws xs = packParts ws ys := by
        have := congrArg (· / 2 ^ w) same
        simpa [Nat.add_mul_div_left _ _ (Nat.two_pow_pos w), Nat.div_eq_of_lt hn,
          Nat.div_eq_of_lt hk] using this
      rw [low, ih tailY high]

/-! ### Simulation -/

/-- The global index (SPEC.md section 7.2 `g`) of an address in region `r` of `memory`. -/
private def cellIndex (p : Geometry) (catalog : List MemorySpec) (memory : ℕ) (r : Region)
    (address : Address) : ℕ :=
  if isRam catalog memory then p.romSize + r.addr address else r.addr address

/-- The region address lies in the memory's namespace, as a valid access needs. -/
private def Live (p : Geometry) (catalog : List MemorySpec) (memory : ℕ) (r : Region)
    (address : Address) : Prop :=
  if isRam catalog memory then r.addr address < p.ramSize else r.addr address < p.romSize

/-- Every in-range address in its namespace holds the value of its Nebula
cell. -/
private def Agrees (p : Geometry) (catalog : List MemorySpec) (b : Binding) (state : MemoryState)
    (mem : Memory) : Prop :=
  ∀ memory r address, b.region memory = some r → r.Fits address →
    Live p catalog memory r address →
      state memory address = some ((mem (cellIndex p catalog memory r address)).value : ℤ)

variable {p : Geometry} {catalog : List MemorySpec} {b : Binding}

/-- Distinct in-range addresses in their namespaces have distinct cells. -/
private theorem cell_injective (disjoint : b.Disjoint catalog) {i j : ℕ} {ri rj : Region}
    {x y : Address} (hi : b.region i = some ri) (hj : b.region j = some rj) (fi : ri.Fits x)
    (fj : rj.Fits y) (li : Live p catalog i ri x) (lj : Live p catalog j rj y)
    (same : cellIndex p catalog i ri x = cellIndex p catalog j rj y) : i = j ∧ x = y := by
  have lti := packParts_lt fi
  have ltj := packParts_lt fj
  unfold cellIndex at same
  unfold Live at li lj
  have flag : isRam catalog i = isRam catalog j := by
    cases ri' : isRam catalog i <;> cases rj' : isRam catalog j <;>
      simp only [ri', rj', Bool.false_eq_true, ite_true, ite_false] at same li lj ⊢ <;> omega
  have addr : ri.addr x = rj.addr y := by
    rw [flag] at same
    split at same <;> omega
  by_cases ij : i = j
  · subst ij
    rw [hi] at hj
    cases hj
    refine ⟨rfl, packParts_injective fi fj ?_⟩
    unfold Region.addr at addr
    omega
  · have apart := disjoint i j ri rj hi hj ij flag
    unfold Region.addr at addr
    unfold Region.size at apart
    omega

private theorem agrees_initial {p : Plan} {initial : MemoryState}
    (start : b.Initial catalog p initial) :
    Agrees p.toGeometry catalog b initial (initialMachine p).memory := by
  intro memory r address hr fits live
  rw [start memory r address hr fits]
  unfold Live at live
  simp only [initialMachine, initialMemory, cellIndex]
  cases ram : isRam catalog memory <;>
    simp only [ram, Bool.false_eq_true, ite_true, ite_false] at live ⊢
  · rw [ite_eq_left live]
  · rw [ite_eq_right (by omega), Nat.add_sub_cancel_left]

/-- The cell that a valid access at an in-range address names. -/
private theorem at_cell {memory : ℕ} {address : Address} {a : PortAccess}
    (at_ : b.At catalog memory address a) (range : b.InRange memory address) (valid : a.Valid p) :
    ∃ r, b.region memory = some r ∧ r.Fits address ∧ Live p catalog memory r address ∧
      a.globalIndex p = cellIndex p catalog memory r address := by
  obtain ⟨r, hr, ram, addr⟩ := at_
  obtain ⟨r', hr', fits⟩ := range
  rw [hr] at hr'
  cases hr'
  obtain ⟨namespaceRange, -⟩ := valid
  refine ⟨r, hr, fits, ?_, ?_⟩
  · rw [ram, addr] at namespaceRange
    exact namespaceRange
  · unfold PortAccess.globalIndex cellIndex
    rw [ram, addr]

/-- One bound access that applies is one event step, and the invariant holds
after both. -/
private theorem step_agrees (disjoint : b.Disjoint catalog) {state : MemoryState}
    {m m' : Machine} {e : Event} {a : PortAccess} (agree : Agrees p catalog b state m.memory)
    (range : b.InRange e.memory e.address) (binds : b.Binds catalog e a)
    (applies : access p m a = some m') :
    ∃ state', EventStep state e state' ∧ Agrees p catalog b state' m'.memory := by
  unfold access at applies
  split_ifs at applies with checks
  cases applies
  obtain ⟨valid, current⟩ := checks
  cases e with
  | read memory address value =>
    obtain ⟨at_, notWrite, readValue⟩ := binds
    obtain ⟨r, hr, fits, live, index⟩ := at_cell at_ range valid
    obtain ⟨-, -, -, keeps, -⟩ := valid
    refine ⟨state, EventStep.read ?_, ?_⟩
    · rw [agree memory r address hr fits live, ← index, current, readValue]
    · intro memory' r' address' hr' fits' live'
      simp only [Machine.apply, Function.update_apply]
      split_ifs with same
      · rw [agree memory' r' address' hr' fits' live', same, current, keeps notWrite]
      · exact agree memory' r' address' hr' fits' live'
  | write memory address value prior =>
    obtain ⟨at_, -, written, priorRead⟩ := binds
    obtain ⟨r, hr, fits, live, index⟩ := at_cell at_ range valid
    refine ⟨update state memory address value, EventStep.write ?_, ?_⟩
    · cases prior with
      | none => trivial
      | some v =>
        show state memory address = some v
        rw [agree memory r address hr fits live, ← index, current, priorRead v rfl]
    · intro memory' r' address' hr' fits' live'
      simp only [Machine.apply, Function.update_apply]
      by_cases cell : memory' = memory ∧ address' = address
      · obtain ⟨rfl, rfl⟩ := cell
        rw [hr] at hr'
        cases hr'
        simp [update, ← index, written]
      · have differ : cellIndex p catalog memory' r' address' ≠ a.globalIndex p := fun same =>
          cell (cell_injective disjoint hr' hr fits' fits live' live (same.trans index))
        rw [ite_eq_right differ, ← agree memory' r' address' hr' fits' live']
        unfold update
        split_ifs with h₁ h₂
        · exact absurd ⟨h₁, h₂⟩ cell
        · rfl
        · rfl

private theorem replay_of_agrees (disjoint : b.Disjoint catalog) {events : List Event}
    {accesses : List PortAccess} (bound : List.Forall₂ (b.Binds catalog) events accesses) :
    ∀ {state : MemoryState} {m final : Machine}, Agrees p catalog b state m.memory →
      (∀ e ∈ events, b.InRange e.memory e.address) → accessAll p m accesses = some final →
      ∃ final', Replay state events final' := by
  induction bound with
  | nil =>
    intro state _ _ _ _ _
    exact ⟨state, .nil state⟩
  | @cons e a events accesses binds _ ih =>
    intro state m final agree range replays
    rw [show accessAll p m (a :: accesses) = (access p m a).bind fun m' =>
      accessAll p m' accesses from rfl, Option.bind_eq_some_iff] at replays
    obtain ⟨m', first, replays⟩ := replays
    obtain ⟨state', step, agree'⟩ :=
      step_agrees disjoint agree (range e List.mem_cons_self) binds first
    obtain ⟨final', replay⟩ :=
      ih agree' (fun e' member => range e' (List.mem_cons_of_mem e member)) replays
    exact ⟨final', .cons step replay⟩

/-! ### Reorder

A write changes one cell of one memory, and a read changes nothing. So two
events of distinct memories swap, and two reads swap, with the same replay. -/

/-- A write event. -/
def Event.isWrite : Event → Bool
  | .read .. => false
  | .write .. => true

/-- Two events that may not swap: one memory, and one of them writes. -/
def Event.Dependent (e f : Event) : Prop :=
  e.memory = f.memory ∧ (e.isWrite ∨ f.isWrite)

/-- `order` lists the events of `events` once each, and each dependent pair in
the order of `events`: each event carries its index in `events`. -/
def Reorders (events order : List Event) : Prop :=
  ∃ tagged : List (Event × ℕ), tagged.Perm events.zipIdx ∧
    tagged.Pairwise (fun a c => a.1.Dependent c.1 → a.2 < c.2) ∧ tagged.map Prod.fst = order

private theorem priorMatches_update {state : MemoryState} {memory memory' : ℕ}
    {address address' : Address} {value : ℤ} {prior : Option ℤ} (differ : memory' ≠ memory) :
    priorMatches (update state memory address value) memory' address' prior ↔
      priorMatches state memory' address' prior := by
  cases prior <;> simp [priorMatches, update, differ]

private theorem eventStep_swap {e f : Event} (independent : ¬ e.Dependent f)
    {s m t : MemoryState} (first : EventStep s e m) (second : EventStep m f t) :
    ∃ m', EventStep s f m' ∧ EventStep m' e t := by
  cases first with
  | read hold =>
    cases second with
    | read hold' => exact ⟨_, .read hold', .read hold⟩
    | write held' =>
      simp [Event.Dependent, Event.memory, Event.isWrite] at independent
      exact ⟨_, .write held', .read (by simp [update, independent, hold])⟩
  | write held =>
    cases second with
    | read hold' =>
      simp [Event.Dependent, Event.memory, Event.isWrite] at independent
      exact ⟨_, .read (by simpa [update, Ne.symm independent] using hold'), .write held⟩
    | write held' =>
      rename_i m₁ a₁ v₁ p₁ m₂ a₂ v₂ p₂
      simp [Event.Dependent, Event.memory, Event.isWrite] at independent
      refine ⟨_, .write ((priorMatches_update (Ne.symm independent)).mp held'), ?_⟩
      have step := EventStep.write (value := v₁)
        ((priorMatches_update (state := s) (address := a₂) (value := v₂) independent).mpr held)
      convert step using 1
      funext n x
      simp only [update]
      split_ifs <;> simp_all

/-- An event that may swap with each event of `pre` replays after them alike. -/
private theorem replay_move {x : Event} {pre post : List Event}
    (independent : ∀ y ∈ pre, ¬ x.Dependent y) :
    ∀ {s t : MemoryState}, Replay s (x :: (pre ++ post)) t → Replay s (pre ++ x :: post) t := by
  induction pre with
  | nil => exact id
  | cons y pre ih =>
    intro s t replay
    obtain _ | ⟨first, rest⟩ := replay
    obtain _ | ⟨second, rest⟩ := rest
    obtain ⟨_, stepY, stepX⟩ := eventStep_swap (independent y List.mem_cons_self) first second
    exact .cons stepY (ih (fun z member => independent z (List.mem_cons_of_mem y member))
      (.cons stepX rest))

private theorem replay_of_tagged {tagged : List (Event × ℕ)} :
    ∀ {es : List (Event × ℕ)}, tagged.Perm es → es.Pairwise (fun a c => a.2 < c.2) →
      tagged.Pairwise (fun a c => a.1.Dependent c.1 → a.2 < c.2) →
      ∀ {s t : MemoryState}, Replay s (tagged.map Prod.fst) t → Replay s (es.map Prod.fst) t := by
  induction tagged with
  | nil =>
    intro es perm _ _ s t replay
    rw [List.nil_perm.mp perm]
    exact replay
  | cons x rest ih =>
    intro es perm sorted keeps s t replay
    obtain ⟨pre, post, rfl⟩ := List.append_of_mem (perm.subset List.mem_cons_self)
    have restPerm : rest.Perm (pre ++ post) := (perm.trans List.perm_middle).cons_inv
    rw [List.pairwise_cons] at keeps
    obtain _ | ⟨first, replay⟩ := replay
    have moved := replay_move (x := x.1) (pre := pre.map Prod.fst) (post := post.map Prod.fst)
      ?_ (.cons first (by
        simpa using ih restPerm (sorted.sublist ((List.sublist_cons_self x post).append_left pre))
          keeps.2 replay))
    · simpa using moved
    · intro y member dependent
      obtain ⟨a, inPre, rfl⟩ := List.mem_map.mp member
      have before : a.2 < x.2 :=
        (List.pairwise_append.mp sorted).2.2 a inPre x List.mem_cons_self
      have after : x.2 < a.2 :=
        keeps.1 a (restPerm.symm.subset (List.mem_append_left post inPre)) dependent
      omega

private theorem zipIdx_sorted (events : List Event) (n : ℕ) :
    (events.zipIdx n).Pairwise (fun a c => a.2 < c.2) := by
  induction events generalizing n with
  | nil => simp
  | cons e events ih =>
    rw [List.zipIdx_cons, List.pairwise_cons]
    refine ⟨fun ⟨_, i⟩ member => ?_, ih (n + 1)⟩
    have := (List.mem_zipIdx member).1
    simp only
    omega

/-- A reorder that keeps each dependent pair replays as the events do. -/
theorem replay_of_reorders {events order : List Event} (reorders : Reorders events order)
    {s t : MemoryState} (replays : Replay s order t) : Replay s events t := by
  obtain ⟨tagged, perm, keeps, rfl⟩ := reorders
  simpa [List.zipIdx_map_fst] using replay_of_tagged perm (zipIdx_sorted events 0) keeps replays

private theorem Reorders.mem {events order : List Event} (reorders : Reorders events order) {e : Event}
    (member : e ∈ order) : e ∈ events := by
  obtain ⟨tagged, perm, -, rfl⟩ := reorders
  obtain ⟨a, inTagged, rfl⟩ := List.mem_map.mp member
  exact List.fst_mem_of_mem_zipIdx (perm.subset inTagged)

/-! ### The bridge -/

/-- If every event of the trace is in range, the initial state holds the plan
images on the in-range set, and accesses that bind a reorder of the events
replay from the plan images, then the trace is `MemoryConsistent`. -/
theorem memoryConsistent_of_accessAll {p : Plan} {initial : MemoryState} {rows : List Witness}
    {order : List Event} {accesses : List PortAccess} {final : Machine}
    (disjoint : b.Disjoint catalog)
    (inRange : ∀ e ∈ eventsForTrace catalog rows, b.InRange e.memory e.address)
    (start : b.Initial catalog p initial) (reorders : Reorders (eventsForTrace catalog rows) order)
    (bound : List.Forall₂ (b.Binds catalog) order accesses)
    (replays : accessAll p.toGeometry (initialMachine p) accesses = some final) :
    MemoryConsistent catalog initial rows := by
  obtain ⟨final', replay⟩ := replay_of_agrees disjoint bound (agrees_initial start)
    (fun e member => inRange e (reorders.mem member)) replays
  exact ⟨final', replay_of_reorders reorders replay⟩

/-- An execution replays its active ports, step after step. -/
theorem executes_accessAll {σ : Type} {app : Application p σ} {s s' : σ} {m m' : Machine}
    {steps : List (List (Option PortAccess))} (exec : Executes p app s m steps s' m') :
    accessAll p m (steps.flatMap fun ports => ports.filterMap id) = some m' := by
  induction exec with
  | nil => rfl
  | step _ first _ ih =>
    rw [List.flatMap_cons, accessAll_append, first, Option.bind_some]
    exact ih

/-- Nebula's memory guarantee for a bound catalog: an execution from the plan
images whose active ports, step after step, bind a reorder of the events of
the trace gives `MemoryConsistent`. -/
theorem memoryConsistent_of_executes {p : Plan} {σ : Type} {app : Application p.toGeometry σ}
    {s s' : σ} {final : Machine} {steps : List (List (Option PortAccess))} {initial : MemoryState}
    {rows : List Witness} {order : List Event}
    (exec : Executes p.toGeometry app s (initialMachine p) steps s' final)
    (disjoint : b.Disjoint catalog)
    (inRange : ∀ e ∈ eventsForTrace catalog rows, b.InRange e.memory e.address)
    (start : b.Initial catalog p initial) (reorders : Reorders (eventsForTrace catalog rows) order)
    (bound : List.Forall₂ (b.Binds catalog) order
      (steps.flatMap fun ports => ports.filterMap id)) :
    MemoryConsistent catalog initial rows :=
  memoryConsistent_of_accessAll disjoint inRange start reorders bound (executes_accessAll exec)

end NightstreamFPrime.Spec.Nebula.MemoryTrace
