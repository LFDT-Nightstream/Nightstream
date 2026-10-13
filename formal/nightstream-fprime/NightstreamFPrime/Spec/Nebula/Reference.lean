import Mathlib.Logic.Function.Basic
import NightstreamFPrime.Spec.Nebula.Packing
import NightstreamFPrime.Spec.Nebula.Hash

/-! Owns the reference memory semantics of spec §5 and §10: cells with a value
and the stamp of their last access, one global timestamp, the access rule, the
application step relation with idle steps, and executions from the plan
images. It also owns snapshots and the memory root of spec §9.2. `access` is
the only memory update rule; nothing else defines a correct execution. -/

namespace NightstreamFPrime.Spec.Nebula

/-- A cell state `(v, t)` (spec §5). -/
structure Cell where
  value : ℕ
  stamp : ℕ

/-- Memory, indexed by global index. Only indices below `R + M` are meaningful. -/
abbrev Memory := ℕ → Cell

/-- Memory and the global timestamp. -/
structure Machine where
  memory : Memory
  ts : ℕ

/-- The plan images with every stamp zero (spec §5, chain start). -/
def initialMemory (p : Plan) : Memory := fun g =>
  if g < p.romSize then ⟨p.rom g, 0⟩ else ⟨p.ram (g - p.romSize), 0⟩

/-- The machine at chain start. -/
def initialMachine (p : Plan) : Machine := ⟨initialMemory p, 0⟩

/-- The machine after an access: the cell becomes `(v_w, ts + 1)` and `ts`
advances. -/
def Machine.apply (p : Plan) (m : Machine) (a : PortAccess) : Machine :=
  ⟨Function.update m.memory (a.globalIndex p) ⟨a.vw, m.ts + 1⟩, m.ts + 1⟩

/-- One active access (spec §5): a valid access whose value matches the cell
applies. -/
def access (p : Plan) (m : Machine) (a : PortAccess) : Option Machine :=
  if a.Valid p ∧ (m.memory (a.globalIndex p)).value = a.vr then some (m.apply p a) else none

/-- The accesses of one or more steps, in order. -/
def accessAll (p : Plan) (m : Machine) (as : List PortAccess) : Option Machine :=
  as.foldlM (access p) m

/-- The ports of an idle step: all inactive (spec §10). -/
def idlePorts (p : Plan) : List (Option PortAccess) := List.replicate p.bOps none

/-- An application relation over states `σ`. Memory is reached only through
ports (Ob7), and an idle step keeps the state (spec §10 MUST). -/
structure Application (p : Plan) (σ : Type) where
  Step : σ → List (Option PortAccess) → σ → Prop
  idle : ∀ s, Step s (idlePorts p) s

/-- An execution: each step is an application step whose active ports are
valid accesses against the current memory. -/
inductive Executes {σ : Type} (p : Plan) (app : Application p σ) :
    σ → Machine → List (List (Option PortAccess)) → σ → Machine → Prop
  | nil (s : σ) (m : Machine) : Executes p app s m [] s m
  | step {s s' s'' : σ} {m m' m'' : Machine} {ports : List (Option PortAccess)}
      {rest : List (List (Option PortAccess))} :
      app.Step s ports s' → accessAll p m (ports.filterMap id) = some m' →
      Executes p app s' m' rest s'' m'' → Executes p app s m (ports :: rest) s'' m''

/-- The snapshot of a memory: one tuple per cell (spec §5, IS and FS). -/
def snapshot (p : Plan) (mem : Memory) : Multiset Tuple :=
  ((List.range p.cells).map fun g => ((mem g).stamp, g, (mem g).value) : List Tuple)

/-- The scan slots of step `j` of a segment: cells `j · B_scan + i`. -/
def scanOf (p : Plan) (mem : Memory) (j : ℕ) : List ScanSlot :=
  (List.range p.bScan).map fun i => ⟨(mem (j * p.bScan + i)).value, (mem (j * p.bScan + i)).stamp⟩

/-- The packed IS or FS lanes of a whole snapshot, one per step. -/
def memoryLanes (p : Plan) (mem : Memory) : List (List ℕ) :=
  (List.range p.n).map fun j => pack (scanLane p (scanOf p mem j))

/-- The FS root of a memory (spec §9.2). `D_init` is the root of the initial
memory. -/
def memoryRoot {Digest : Type} (H : HashInput Digest → Digest) (p : Plan) (pd : Digest)
    (mem : Memory) : Digest :=
  chainRoot H .mem pd (memoryLanes p mem)

theorem accessAll_nil (p : Plan) (m : Machine) : accessAll p m [] = some m := rfl

theorem accessAll_append (p : Plan) (m : Machine) (as bs : List PortAccess) :
    accessAll p m (as ++ bs) = (accessAll p m as).bind fun m' => accessAll p m' bs := by
  unfold accessAll
  rw [List.foldlM_append]
  rfl

/-- A valid access names a cell below `R + M`. -/
theorem PortAccess.Valid.globalIndex_lt {p : Plan} {a : PortAccess} (valid : a.Valid p) :
    a.globalIndex p < p.cells := by
  obtain ⟨range, -⟩ := valid
  unfold PortAccess.globalIndex Plan.cells
  cases h : a.isRam
  · simp only [h, Bool.false_eq_true, ite_false] at range ⊢
    exact Nat.lt_add_right _ range
  · simp only [h, ite_true] at range ⊢
    exact Nat.add_lt_add_left range _

private theorem apply_congr {p : Plan} {m₁ m₂ : Machine} (ts : m₁.ts = m₂.ts)
    (agree : Set.EqOn m₁.memory m₂.memory (Set.Iio p.cells)) (a : PortAccess) :
    (m₁.apply p a).ts = (m₂.apply p a).ts ∧
      Set.EqOn (m₁.apply p a).memory (m₂.apply p a).memory (Set.Iio p.cells) := by
  refine ⟨by simp [Machine.apply, ts], fun g hg => ?_⟩
  simp only [Machine.apply, ts]
  by_cases h : g = a.globalIndex p
  · subst h
    simp
  · simp [Function.update_of_ne h, agree hg]

/-- Accesses read and write only valid global indices, so two machines that
agree below `R + M` stay in agreement. -/
theorem accessAll_congr {p : Plan} {m₁ m₂ : Machine} (ts : m₁.ts = m₂.ts)
    (agree : Set.EqOn m₁.memory m₂.memory (Set.Iio p.cells)) (as : List PortAccess) :
    ((accessAll p m₁ as).map Machine.ts = (accessAll p m₂ as).map Machine.ts) ∧
      ∀ m₁' m₂', accessAll p m₁ as = some m₁' → accessAll p m₂ as = some m₂' →
        Set.EqOn m₁'.memory m₂'.memory (Set.Iio p.cells) := by
  induction as generalizing m₁ m₂ with
  | nil =>
    refine ⟨by simp [accessAll_nil, ts], ?_⟩
    simp only [accessAll_nil, Option.some.injEq]
    rintro _ _ rfl rfl
    exact agree
  | cons a rest ih =>
    have unfold : ∀ m, accessAll p m (a :: rest) = (access p m a).bind fun m' =>
        accessAll p m' rest := fun _ => rfl
    have step : (access p m₁ a = none ∧ access p m₂ a = none) ∨
        (access p m₁ a = some (m₁.apply p a) ∧ access p m₂ a = some (m₂.apply p a)) := by
      by_cases hv : a.Valid p
      · have same := agree (Set.mem_Iio.mpr hv.globalIndex_lt)
        by_cases hr : (m₁.memory (a.globalIndex p)).value = a.vr
        · right
          simp [access, hv, hr, ← same]
        · left
          simp [access, hv, hr, ← same]
      · left
        simp [access, hv]
    rcases step with ⟨h₁, h₂⟩ | ⟨h₁, h₂⟩
    · simp [unfold, h₁, h₂]
    · obtain ⟨ts', agree'⟩ := apply_congr ts agree a
      simpa [unfold, h₁, h₂] using ih (m₁ := m₁.apply p a) (m₂ := m₂.apply p a) ts' agree'

private theorem scanOf_congr {p : Plan} (valid : p.Valid) {mem₁ mem₂ : Memory}
    (agree : Set.EqOn mem₁ mem₂ (Set.Iio p.cells)) {j : ℕ} (hj : j < p.n) :
    scanOf p mem₁ j = scanOf p mem₂ j := by
  unfold scanOf
  refine List.map_congr_left fun i hi => ?_
  have hi : i < p.bScan := List.mem_range.mp hi
  have lt : j * p.bScan + i < p.cells := by
    rw [← valid.exactCover]
    calc j * p.bScan + i < (j + 1) * p.bScan := by rw [Nat.add_mul, Nat.one_mul]; omega
      _ ≤ p.n * p.bScan := Nat.mul_le_mul_right _ hj
  rw [agree (Set.mem_Iio.mpr lt)]

/-- The memory root reads only cells below `R + M`. -/
theorem memoryRoot_congr {Digest : Type} (H : HashInput Digest → Digest) {p : Plan}
    (valid : p.Valid) (pd : Digest) {mem₁ mem₂ : Memory}
    (agree : Set.EqOn mem₁ mem₂ (Set.Iio p.cells)) :
    memoryRoot H p pd mem₁ = memoryRoot H p pd mem₂ := by
  unfold memoryRoot memoryLanes
  congr 1
  refine List.map_congr_left fun j hj => ?_
  rw [scanOf_congr valid agree (List.mem_range.mp hj)]

/-- Spec §10: an execution extends by idle steps. -/
theorem executes_padIdle {σ : Type} {p : Plan} {app : Application p σ} {s₀ s : σ}
    {m₀ m : Machine} {steps : List (List (Option PortAccess))}
    (exec : Executes p app s₀ m₀ steps s m) (k : ℕ) :
    Executes p app s₀ m₀ (steps ++ List.replicate k (idlePorts p)) s m := by
  induction exec with
  | nil s m =>
    rw [List.nil_append]
    induction k with
    | zero => exact Executes.nil s m
    | succ k ih =>
      rw [List.replicate_succ]
      exact Executes.step (app.idle s) (by simp [idlePorts, accessAll_nil]) ih
  | step hstep hacc _ ih =>
    rw [List.cons_append]
    exact Executes.step hstep hacc ih

end NightstreamFPrime.Spec.Nebula
