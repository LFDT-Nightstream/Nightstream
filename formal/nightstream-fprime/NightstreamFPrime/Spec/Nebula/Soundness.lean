import NightstreamFPrime.Spec.Nebula.Binding
import NightstreamFPrime.Spec.Nebula.Consistency

/-! Owns security note Lemma 6, deterministic form: an accepted extracted run
refines the reference machine, or the hash has a collision among the run's own
chain inputs and the `D_init` chain, or one segment's challenges passed the
product test although its multisets are not balanced. The last event is the
one that Lemma 3 bounds (`RetryAveraging`). A1–A4 and A6, which turn verifier
acceptance into `Accepts`, are not owned here. -/

namespace NightstreamFPrime.Spec.Nebula

variable {E Digest σ : Type} [CommRing E]

/-- What an accepted run attests (security note Lemma 6): one execution from
the initial state and the plan images, over the run's ports, that ends in the
statement's final state, timestamp, and memory root. -/
structure Attests (ctx : Context E Digest) (app : Application ctx.plan σ)
    (stmt : Statement σ Digest) (run : List (Invocation σ Digest)) : Prop where
  executes : ∃ final : Machine,
    Executes ctx.plan app stmt.initial (initialMachine ctx.plan) (run.map Invocation.ports)
      stmt.final final ∧
    final.ts = stmt.finalTs ∧
    memoryRoot ctx.hash ctx.plan ctx.planDigest final.memory = stmt.finalRoot

/-- The segment's challenges pass the product test, but its multisets are not
balanced. -/
def BadChallenge (ctx : Context E Digest) (v : SegmentView Digest) : Prop :=
  ¬ (v.multisets ctx.plan).Balanced ∧ (v.multisets ctx.plan).ProductEq (v.eta ctx)

/-- Every hash input that the verifier and the first `S` segments of the run
compute: the `D_init` chain and the three chains of each segment. -/
def runInputs (ctx : Context E Digest) (run : List (Invocation σ Digest)) (S : ℕ) :
    List (HashInput Digest) :=
  chainInputs ctx.hash .mem ctx.planDigest (memoryLanes ctx.plan (initialMemory ctx.plan)) ++
    (List.range S).flatMap fun k => (segmentView ctx run k).chainInputs ctx

/-- A collision among the run's own hash inputs. -/
def RunCollision (ctx : Context E Digest) (run : List (Invocation σ Digest)) (S : ℕ) : Prop :=
  CollisionIn ctx.hash (runInputs ctx run S) (runInputs ctx run S)

/-- The port accesses of a segment, in order. -/
def SegmentView.accesses (v : SegmentView Digest) : List PortAccess :=
  v.records.flatMap fun z => z.ops.filterMap OpSlot.port

/-! ### One segment -/

private theorem port_valid {p : Plan} {s : OpSlot} {a : PortAccess} (rows : s.Rows p)
    (fits : s.Fits p) (port : s.port = some a) : a.Valid p := by
  unfold OpSlot.port at port
  split_ifs at port
  cases port
  obtain ⟨addr, vr, vw, -⟩ := fits
  unfold PortAccess.Valid
  refine ⟨?_, vr, vw, rows.readKeeps, rows.noRomWrite⟩
  cases ram : s.isRam
  · simpa [ram] using rows.romRange ram
  · simpa [ram, Plan.ramSize] using addr

/-- The active operations of rows that hold are fresh and valid. -/
private theorem segmentOps_rows {p : Plan} {ts : ℕ} {rs : List StepRecords}
    (rows : SegmentRows p ts rs) : ∀ o ∈ segmentOps ts rs, o.rt < o.wt ∧ o.access.Valid p := by
  induction rs generalizing ts with
  | nil => simp [segmentOps]
  | cons z rest ih =>
    obtain ⟨step, rows⟩ := rows
    intro o member
    simp only [segmentOps, List.mem_append] at member
    rcases member with member | member
    · obtain ⟨s, slot, port, -⟩ := mem_activeOps member
      exact ⟨step.fresh o member,
        port_valid (step.slots s slot) (step.shaped.opsFit s slot) port⟩
    · exact ih rows o member

private theorem snapshot_scanMemory {p : Plan} (valid : p.Valid) {scans : List (List ScanSlot)}
    (length : scans.length = p.n) (shaped : ScansShaped p scans) :
    snapshot p (scanMemory p scans) = chunkTuples p 0 scans := by
  rw [← chunkTuples_scansOf valid, scansOf_scanMemory valid length shaped fun _ _ => rfl]

/-- Lemma 5 on one closing, balanced segment: from a machine that holds the IS
snapshot, the segment's accesses replay and end at the FS snapshot. -/
theorem segment_replays {ctx : Context E Digest} {η : E × E} {v : SegmentView Digest}
    (valid : ctx.plan.Valid) (closes : v.ClosesAt ctx η)
    (balanced : (v.multisets ctx.plan).Balanced) {m : Machine}
    (start : Set.EqOn m.memory (scanMemory ctx.plan v.initialScans) (Set.Iio ctx.plan.cells))
    (ts : m.ts = v.openTs) :
    ∃ m', accessAll ctx.plan m v.accesses = some m' ∧
      Set.EqOn m'.memory (scanMemory ctx.plan v.finalScans) (Set.Iio ctx.plan.cells) ∧
      m'.ts = v.openTs + (v.records.map activeCount).sum := by
  have lengthIS : v.initialScans.length = ctx.plan.n := by
    simp [SegmentView.initialScans, closes.length]
  have lengthFS : v.finalScans.length = ctx.plan.n := by
    simp [SegmentView.finalScans, closes.length]
  have tuples := balanced
  unfold Multisets.Balanced SegmentView.multisets at tuples
  rw [segmentMultisets_initial, segmentMultisets_write, segmentMultisets_read,
    segmentMultisets_final] at tuples
  have snapshots : snapshot ctx.plan (scanMemory ctx.plan v.initialScans) +
      (((segmentOps v.openTs v.records).map (MemOp.write ctx.plan) : List Tuple) :
        Multiset Tuple) =
      (((segmentOps v.openTs v.records).map (MemOp.read ctx.plan) : List Tuple) :
        Multiset Tuple) + snapshot ctx.plan (scanMemory ctx.plan v.finalScans) := by
    rw [snapshot_scanMemory valid lengthIS closes.initialShaped,
      snapshot_scanMemory valid lengthFS closes.finalShaped]
    exact tuples
  obtain ⟨mem, replay, agree⟩ := segment_sequential ctx.plan
    (start := ⟨scanMemory ctx.plan v.initialScans, v.openTs⟩)
    (final := scanMemory ctx.plan v.finalScans) (ops := segmentOps v.openTs v.records)
    (segmentOps_wt v.openTs v.records) (fun o member => (segmentOps_rows closes.rows o member).1)
    (fun o member => (segmentOps_rows closes.rows o member).2) snapshots
  have accesses : (segmentOps v.openTs v.records).map MemOp.access = v.accesses :=
    segmentOps_access _ _
  rw [accesses] at replay
  obtain ⟨sameTs, sameMem⟩ := accessAll_congr (p := ctx.plan) (m₁ := m)
    (m₂ := ⟨scanMemory ctx.plan v.initialScans, v.openTs⟩) ts start v.accesses
  have tsEq : (accessAll ctx.plan m v.accesses).map Machine.ts =
      some (v.openTs + (segmentOps v.openTs v.records).length) := by
    rw [sameTs, replay]
    rfl
  obtain ⟨m', replay', ts'⟩ := Option.map_eq_some_iff.1 tsEq
  refine ⟨m', replay', (sameMem m' _ replay' replay).trans agree, ?_⟩
  rw [ts', segmentOps_length]

/-- A replay of a run's accesses along its application thread is an
execution. -/
theorem executes_of_replay {p : Plan} {app : Application p σ} {s s' : σ} {m m' : Machine}
    {run : List (Invocation σ Digest)} (thread : AppThread app s run s')
    (replay : accessAll p m (run.flatMap fun inv => inv.ports.filterMap id) = some m') :
    Executes p app s m (run.map Invocation.ports) s' m' := by
  induction run generalizing s m with
  | nil =>
    have same : s = s' := thread
    simp only [List.flatMap_nil, accessAll_nil, Option.some.injEq] at replay
    subst same replay
    exact Executes.nil s m
  | cons inv rest ih =>
    obtain ⟨step, thread⟩ := thread
    rw [List.flatMap_cons, accessAll_append, Option.bind_eq_some_iff] at replay
    obtain ⟨mid, first, replay⟩ := replay
    rw [List.map_cons]
    exact Executes.step step first (ih thread replay)

/-! ### The run -/

private theorem take_succ_mul {α : Type} (run : List α) (n k : ℕ) :
    run.take ((k + 1) * n) = run.take (k * n) ++ (run.drop (k * n)).take n := by
  rw [Nat.succ_mul, List.take_add]

omit [CommRing E] in
private theorem segmentView_accesses (ctx : Context E Digest) (run : List (Invocation σ Digest))
    (k : ℕ) : (segmentView ctx run k).accesses =
      ((run.drop (k * ctx.plan.n)).take ctx.plan.n).flatMap fun inv => inv.ports.filterMap id := by
  simp only [SegmentView.accesses, segmentView, segmentRun, List.flatMap_map, Invocation.ports,
    List.filterMap_map]
  rfl

omit [CommRing E] in
private theorem tsBefore_succ (ctx : Context E Digest) (run : List (Invocation σ Digest))
    (k : ℕ) : tsBefore run ((k + 1) * ctx.plan.n) =
      (segmentView ctx run k).openTs + ((segmentView ctx run k).records.map activeCount).sum := by
  show tsBefore run ((k + 1) * ctx.plan.n) = tsBefore run (k * ctx.plan.n) +
    (((segmentRun ctx.plan run k).map Invocation.records).map activeCount).sum
  unfold tsBefore segmentRun
  rw [take_succ_mul, List.map_append, List.sum_append, List.map_map]
  rfl

omit [CommRing E] in
private theorem segment_inputs_subset {ctx : Context E Digest} {run : List (Invocation σ Digest)}
    {S k : ℕ} (hk : k < S) : (segmentView ctx run k).chainInputs ctx ⊆ runInputs ctx run S := by
  intro x member
  exact List.mem_append_right _ (List.mem_flatMap.2 ⟨k, List.mem_range.2 hk, member⟩)

/-- Every hash input of an accepted run is canonical (Ob3 input range). -/
theorem runInputs_canonical {ctx : Context E Digest} {app : Application ctx.plan σ}
    {stmt : Statement σ Digest} {run : List (Invocation σ Digest)}
    (valid : ctx.plan.Valid) (accepted : Accepts ctx app stmt run) :
    ∀ x ∈ runInputs ctx run stmt.segments, x.Canonical ctx.plan.laneLength ctx.plan.n := by
  have count : run.length = stmt.segments * ctx.plan.n := accepted.steps.trans accepted.stepCount
  obtain ⟨c, final, -⟩ := accepted.terminal
  have closes := ((finalCarry_isSome_iff valid count).1 ⟨c, final⟩).2
  intro x member
  rcases List.mem_append.1 member with initial | segments
  · exact initialInputs_canonical valid _ _ x initial
  · obtain ⟨k, hk, member⟩ := List.mem_flatMap.1 segments
    exact (closes k (List.mem_range.1 hk)).chainInputs_canonical x member

/-- The final root of a machine that holds the FS snapshot of a closing
segment is the segment's FS proposal. -/
private theorem memoryRoot_finalScans {ctx : Context E Digest} {η : E × E}
    {v : SegmentView Digest} (valid : ctx.plan.Valid) (closes : v.ClosesAt ctx η) {mem : Memory}
    (agree : Set.EqOn mem (scanMemory ctx.plan v.finalScans) (Set.Iio ctx.plan.cells)) :
    memoryRoot ctx.hash ctx.plan ctx.planDigest mem = v.proposal.2 := by
  have length : v.finalScans.length = ctx.plan.n := by
    simp [SegmentView.finalScans, closes.length]
  rw [memoryRoot_congr ctx.hash valid ctx.planDigest agree, memoryRoot, memoryLanes_eq,
    scansOf_scanMemory valid length closes.finalShaped fun _ _ => rfl, ← closes.finalRoot,
    SegmentView.finalScans, List.map_map]
  rfl

/-- Segment `k` replays after the first `k` segments. -/
private theorem replay_segment {ctx : Context E Digest} {run : List (Invocation σ Digest)}
    {k : ℕ} {η : E × E} (valid : ctx.plan.Valid)
    (closes : (segmentView ctx run k).ClosesAt ctx η)
    (balanced : ((segmentView ctx run k).multisets ctx.plan).Balanced) {m : Machine}
    (replay : accessAll ctx.plan (initialMachine ctx.plan)
      ((run.take (k * ctx.plan.n)).flatMap fun inv => inv.ports.filterMap id) = some m)
    (ts : m.ts = tsBefore run (k * ctx.plan.n))
    (start : Set.EqOn m.memory (scanMemory ctx.plan (segmentView ctx run k).initialScans)
      (Set.Iio ctx.plan.cells)) :
    ∃ m', accessAll ctx.plan (initialMachine ctx.plan)
        ((run.take ((k + 1) * ctx.plan.n)).flatMap fun inv => inv.ports.filterMap id) =
          some m' ∧
      m'.ts = tsBefore run ((k + 1) * ctx.plan.n) ∧
      Set.EqOn m'.memory (scanMemory ctx.plan (segmentView ctx run k).finalScans)
        (Set.Iio ctx.plan.cells) := by
  obtain ⟨m', replay', agree, ts'⟩ := segment_replays valid closes balanced start ts
  refine ⟨m', ?_, ?_, agree⟩
  · rw [take_succ_mul, List.flatMap_append, accessAll_append, replay, Option.bind_some,
      ← segmentView_accesses ctx]
    exact replay'
  · rw [ts', tsBefore_succ]

/-- The first `k` segments replay from the plan images and end at the IS
snapshot of segment `k`. -/
private theorem replay_prefix {ctx : Context E Digest} {run : List (Invocation σ Digest)}
    {S : ℕ} (valid : ctx.plan.Valid)
    (closes : ∀ k < S, (segmentView ctx run k).ClosesAt ctx ((segmentView ctx run k).eta ctx))
    (balanced : ∀ k < S, ((segmentView ctx run k).multisets ctx.plan).Balanced)
    (init : (segmentView ctx run 0).initialScans = scansOf ctx.plan (initialMemory ctx.plan))
    (join : ∀ k, k + 1 < S →
      (segmentView ctx run (k + 1)).initialScans = (segmentView ctx run k).finalScans) :
    ∀ k < S, ∃ m, accessAll ctx.plan (initialMachine ctx.plan)
        ((run.take (k * ctx.plan.n)).flatMap fun inv => inv.ports.filterMap id) = some m ∧
      m.ts = tsBefore run (k * ctx.plan.n) ∧
      Set.EqOn m.memory (scanMemory ctx.plan (segmentView ctx run k).initialScans)
        (Set.Iio ctx.plan.cells)
  | 0, _ => ⟨initialMachine ctx.plan, by simp [accessAll_nil], by simp [tsBefore, initialMachine],
      by rw [init]; exact (scanMemory_scansOf valid _).symm⟩
  | k + 1, hk => by
    obtain ⟨m, replay, ts, start⟩ := replay_prefix valid closes balanced init join k (by omega)
    obtain ⟨m', replay', ts', agree⟩ :=
      replay_segment valid (closes k (by omega)) (balanced k (by omega)) replay ts start
    exact ⟨m', replay', ts', by rw [join k hk]; exact agree⟩

/-- Security note Lemma 6, deterministic form. -/
theorem soundness {ctx : Context E Digest} {app : Application ctx.plan σ}
    {stmt : Statement σ Digest} {run : List (Invocation σ Digest)}
    (valid : ctx.plan.Valid) (accepted : Accepts ctx app stmt run) :
    Attests ctx app stmt run ∨ RunCollision ctx run stmt.segments ∨
      ∃ k < stmt.segments, BadChallenge ctx (segmentView ctx run k) := by
  classical
  have count : run.length = stmt.segments * ctx.plan.n := accepted.steps.trans accepted.stepCount
  have pos : 1 ≤ stmt.segments := accepted.segmentsRange.1
  obtain ⟨c, final, -, -, finalTs, finalRoot⟩ := accepted.terminal
  have closes := ((finalCarry_isSome_iff valid count).1 ⟨c, final⟩).2
  obtain ⟨-, -, carryTs, carryRoot⟩ := finalCarry_fields valid count pos final
  by_cases bad : ∃ k < stmt.segments, ¬ ((segmentView ctx run k).multisets ctx.plan).Balanced
  · obtain ⟨k, hk, unbalanced⟩ := bad
    exact Or.inr (Or.inr ⟨k, hk, unbalanced, (closes k hk).products⟩)
  push Not at bad
  by_cases collision : RunCollision ctx run stmt.segments
  · exact Or.inr (Or.inl collision)
  have init : (segmentView ctx run 0).initialScans = scansOf ctx.plan (initialMemory ctx.plan) := by
    refine (initial_scans_or_collision valid (closes 0 (by omega))).resolve_right
      fun hit => collision ?_
    refine CollisionIn.mono (segment_inputs_subset (by omega)) ?_ hit
    intro x member
    exact List.mem_append_left _ member
  have join : ∀ k, k + 1 < stmt.segments →
      (segmentView ctx run (k + 1)).initialScans = (segmentView ctx run k).finalScans := by
    intro k hk
    refine (join_scans_or_collision (closes k (by omega)) (closes (k + 1) hk)).resolve_right
      fun hit => collision ?_
    exact CollisionIn.mono (segment_inputs_subset hk) (segment_inputs_subset (by omega)) hit
  obtain ⟨m, replay, ts, start⟩ :=
    replay_prefix valid closes bad init join (stmt.segments - 1) (by omega)
  obtain ⟨final', replay', ts', agree⟩ :=
    replay_segment valid (closes _ (by omega)) (bad _ (by omega)) replay ts start
  rw [Nat.sub_add_cancel pos, ← count, List.take_length] at replay'
  rw [Nat.sub_add_cancel pos, ← count] at ts'
  refine Or.inl ⟨final', executes_of_replay accepted.application replay', ?_, ?_⟩
  · rw [ts', ← carryTs, finalTs]
  · rw [memoryRoot_finalScans valid (closes _ (by omega)) agree, ← finalRoot, carryRoot]
    rfl

end NightstreamFPrime.Spec.Nebula
