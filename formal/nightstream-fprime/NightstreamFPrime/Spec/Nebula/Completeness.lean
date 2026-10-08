import NightstreamFPrime.Spec.Nebula.Binding
import NightstreamFPrime.Spec.Nebula.Consistency

/-! Owns completeness (Ob5): every reference execution of `S · N` steps, with
`1 ≤ S ≤ S_max` and one port list of width `B_ops` per step, has a run that
passes every check, for every hash function and every challenge function. The
honest run puts each access's current cell stamp in `rt`, the open and close
snapshots of each segment in the IS and FS chunks, and the honest roots in the
proposals. Padding a shorter execution with idle steps is
`executes_padIdle`. -/

namespace NightstreamFPrime.Spec.Nebula

variable {E Digest σ : Type} [CommRing E]

/-- The statement that an honest execution proves. -/
def honestStatement (ctx : Context E Digest) (initial final : σ) (machine : Machine)
    (segments : ℕ) : Statement σ Digest :=
  ⟨segments * ctx.plan.n, initial, final, segments, machine.ts,
    memoryRoot ctx.hash ctx.plan ctx.planDigest machine.memory⟩

/-! ### Machines of an execution -/

/-- Accesses applied without checks. -/
private def replay (p : Plan) (m : Machine) (as : List PortAccess) : Machine :=
  as.foldl (Machine.apply p) m

private theorem replay_ts (p : Plan) (m : Machine) (as : List PortAccess) :
    (replay p m as).ts = m.ts + as.length := by
  induction as generalizing m with
  | nil => rfl
  | cons a as ih =>
    show (replay p (m.apply p a) as).ts = _
    rw [ih]
    simp only [Machine.apply, List.length_cons]
    omega

private theorem apply_below {p : Plan} {m : Machine} (a : PortAccess)
    (below : ∀ g, (m.memory g).stamp ≤ m.ts) :
    ∀ g, ((m.apply p a).memory g).stamp ≤ (m.apply p a).ts := by
  intro g
  simp only [Machine.apply, Function.update_apply]
  split
  · exact le_rfl
  · exact Nat.le_succ_of_le (below g)

private theorem replay_below {p : Plan} {m : Machine} {as : List PortAccess}
    (below : ∀ g, (m.memory g).stamp ≤ m.ts) :
    ∀ g, ((replay p m as).memory g).stamp ≤ (replay p m as).ts := by
  induction as generalizing m with
  | nil => exact below
  | cons a as ih => exact ih (apply_below a below)

private theorem replay_values {p : Plan} {m : Machine} {as : List PortAccess}
    (written : ∀ a ∈ as, a.vw < 2 ^ 32) (values : ∀ g < p.cells, (m.memory g).value < 2 ^ 32) :
    ∀ g < p.cells, ((replay p m as).memory g).value < 2 ^ 32 := by
  induction as generalizing m with
  | nil => exact values
  | cons a as ih =>
    show ∀ g < p.cells, ((replay p (m.apply p a) as).memory g).value < 2 ^ 32
    refine ih (fun b hb => written b (List.mem_cons.2 (Or.inr hb))) fun g hg => ?_
    simp only [Machine.apply, Function.update_apply]
    split
    · exact written a (List.mem_cons.2 (Or.inl rfl))
    · exact values g hg

private theorem accessAll_eq_some {p : Plan} {m m' : Machine} {as : List PortAccess}
    (run : accessAll p m as = some m') : m' = replay p m as ∧ ∀ a ∈ as, a.Valid p := by
  induction as generalizing m with
  | nil =>
    have same : some m = some m' := run
    exact ⟨(Option.some.inj same).symm, by simp⟩
  | cons a as ih =>
    unfold accessAll at run
    rw [List.foldlM_cons] at run
    unfold access at run
    split_ifs at run with h
    · have rest : accessAll p (m.apply p a) as = some m' := run
      obtain ⟨eq, valid⟩ := ih rest
      refine ⟨eq, fun b hb => ?_⟩
      rcases List.mem_cons.1 hb with rfl | hb
      · exact h.1
      · exact valid b hb
    · cases run

private theorem honestOps_append (p : Plan) (m : Machine) (as bs : List PortAccess) :
    honestOps p m (as ++ bs) = honestOps p m as ++ honestOps p (replay p m as) bs := by
  induction as generalizing m with
  | nil => rfl
  | cons a as ih =>
    show _ :: honestOps p (m.apply p a) (as ++ bs) = _ :: (honestOps p (m.apply p a) as ++ _)
    rw [ih]
    rfl

/-- The accesses of step `i`. -/
private def stepAccesses (steps : List (List (Option PortAccess))) (i : ℕ) : List PortAccess :=
  (steps.getD i []).filterMap id

private theorem stepAccesses_cons_succ (ports : List (Option PortAccess))
    (rest : List (List (Option PortAccess))) (i : ℕ) :
    stepAccesses (ports :: rest) (i + 1) = stepAccesses rest i := by
  simp [stepAccesses]

private theorem getD_mem {steps : List (List (Option PortAccess))} {i : ℕ}
    (hi : i < steps.length) : steps.getD i [] ∈ steps := by
  rw [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem hi, Option.getD_some]
  exact List.getElem_mem hi

/-- The machine before step `i`, from `m`. -/
private def machineAt (p : Plan) (m : Machine) (steps : List (List (Option PortAccess))) :
    ℕ → Machine
  | 0 => m
  | i + 1 => replay p (machineAt p m steps i) (stepAccesses steps i)

private theorem machineAt_cons (p : Plan) (m : Machine) (ports : List (Option PortAccess))
    (rest : List (List (Option PortAccess))) :
    ∀ i, machineAt p m (ports :: rest) (i + 1) =
      machineAt p (replay p m (ports.filterMap id)) rest i
  | 0 => rfl
  | i + 1 => by
    show replay p (machineAt p m (ports :: rest) (i + 1)) (stepAccesses (ports :: rest) (i + 1)) =
      replay p (machineAt p (replay p m (ports.filterMap id)) rest i) (stepAccesses rest i)
    rw [machineAt_cons p m ports rest i, stepAccesses_cons_succ]

private theorem executes_trace {p : Plan} {app : Application p σ} {s s' : σ} {m m' : Machine}
    {steps : List (List (Option PortAccess))} (exec : Executes p app s m steps s' m') :
    (∀ i < steps.length,
        accessAll p (machineAt p m steps i) (stepAccesses steps i) =
          some (machineAt p m steps (i + 1))) ∧
      machineAt p m steps steps.length = m' ∧
      ∃ st : ℕ → σ, st 0 = s ∧
        (∀ i < steps.length, app.Step (st i) (steps.getD i []) (st (i + 1))) ∧
        st steps.length = s' := by
  induction exec with
  | nil s m =>
    exact ⟨fun i hi => absurd hi (Nat.not_lt_zero i), rfl, fun _ => s, rfl,
      fun i hi => absurd hi (Nat.not_lt_zero i), rfl⟩
  | @step s₁ s₂ s₃ m₁ m₂ m₃ ports rest hstep hacc _ ih =>
    obtain ⟨next, last, st, st0, stStep, stLast⟩ := ih
    have hm : replay p m₁ (ports.filterMap id) = m₂ := (accessAll_eq_some hacc).1.symm
    refine ⟨fun i hi => ?_, ?_, fun i => match i with | 0 => s₁ | i + 1 => st i, rfl,
      fun i hi => ?_, stLast⟩
    · cases i with
      | zero =>
        show accessAll p m₁ (ports.filterMap id) = some (replay p m₁ (ports.filterMap id))
        rw [hm]
        exact hacc
      | succ i =>
        rw [machineAt_cons, machineAt_cons, hm, stepAccesses_cons_succ]
        exact next i (Nat.lt_of_succ_lt_succ hi)
    · show machineAt p m₁ (ports :: rest) (rest.length + 1) = m₃
      rw [machineAt_cons, hm]
      exact last
    · cases i with
      | zero =>
        show app.Step s₁ ports (st 0)
        rw [st0]
        exact hstep
      | succ i =>
        rw [List.getD_cons_succ]
        exact stStep i (Nat.lt_of_succ_lt_succ hi)

/-- Machines `ms i` before each step `i` of an execution: each step's accesses
run on the machine before it. -/
private structure Trace (p : Plan) (steps : List (List (Option PortAccess)))
    (ms : ℕ → Machine) : Prop where
  start : ms 0 = initialMachine p
  next : ∀ i < steps.length, accessAll p (ms i) (stepAccesses steps i) = some (ms (i + 1))

section Trace

variable {p : Plan} {steps : List (List (Option PortAccess))} {ms : ℕ → Machine}

private theorem trace_replay (tr : Trace p steps ms) {i : ℕ} (hi : i < steps.length) :
    ms (i + 1) = replay p (ms i) (stepAccesses steps i) :=
  (accessAll_eq_some (tr.next i hi)).1

private theorem trace_valid (tr : Trace p steps ms) {i : ℕ} (hi : i < steps.length) :
    ∀ a ∈ stepAccesses steps i, a.Valid p :=
  (accessAll_eq_some (tr.next i hi)).2

private theorem trace_ts_succ (tr : Trace p steps ms) {i : ℕ} (hi : i < steps.length) :
    (ms (i + 1)).ts = (ms i).ts + (stepAccesses steps i).length := by
  rw [trace_replay tr hi, replay_ts]

private theorem trace_below (tr : Trace p steps ms) :
    ∀ i ≤ steps.length, ∀ g, ((ms i).memory g).stamp ≤ (ms i).ts
  | 0, _ => by
    intro g
    rw [tr.start]
    show (initialMemory p g).stamp ≤ 0
    unfold initialMemory
    split <;> exact le_rfl
  | i + 1, hi => by
    rw [trace_replay tr hi]
    exact replay_below (trace_below tr i (Nat.le_of_succ_le hi))

private theorem trace_ts_le (width : ∀ ports ∈ steps, ports.length = p.bOps)
    (tr : Trace p steps ms) : ∀ i ≤ steps.length, (ms i).ts ≤ i * p.bOps
  | 0, _ => by
    rw [tr.start]
    exact Nat.zero_le _
  | i + 1, hi => by
    have step : (stepAccesses steps i).length ≤ p.bOps := by
      rw [← width _ (getD_mem hi)]
      exact List.length_filterMap_le _ _
    have before := trace_ts_le width tr i (Nat.le_of_succ_le hi)
    rw [trace_ts_succ tr hi, Nat.add_one_mul]
    omega

private theorem trace_ts_lt (width : ∀ ports ∈ steps, ports.length = p.bOps)
    (bound : steps.length * p.bOps < 2 ^ p.wTs) (tr : Trace p steps ms) {i : ℕ}
    (hi : i ≤ steps.length) : (ms i).ts < 2 ^ p.wTs :=
  lt_of_le_of_lt ((trace_ts_le width tr i hi).trans (Nat.mul_le_mul_right _ hi)) bound

private theorem trace_values (valid : p.Valid) (tr : Trace p steps ms) :
    ∀ i ≤ steps.length, ∀ g < p.cells, ((ms i).memory g).value < 2 ^ 32
  | 0, _ => fun g hg => by
    rw [tr.start]
    exact (initialMemory_fits valid g hg).1
  | i + 1, hi => by
    rw [trace_replay tr hi]
    exact replay_values (fun a ha => (trace_valid tr hi a ha).2.2.1)
      (trace_values valid tr i (Nat.le_of_succ_le hi))

private theorem trace_fits (valid : p.Valid) (width : ∀ ports ∈ steps, ports.length = p.bOps)
    (bound : steps.length * p.bOps < 2 ^ p.wTs) (tr : Trace p steps ms) {i : ℕ}
    (hi : i ≤ steps.length) : (ms i).memory.Fits p := fun g hg =>
  ⟨trace_values valid tr i hi g hg,
    lt_of_le_of_lt (trace_below tr i hi g) (trace_ts_lt width bound tr hi)⟩

end Trace

/-! ### Honest records -/

/-- The operation slots of one step: the pad slot for an inactive port, and the
access with the current stamp of its cell for an active port. -/
private def honestSlots (p : Plan) : Machine → List (Option PortAccess) → List OpSlot
  | _, [] => []
  | m, none :: rest => OpSlot.padSlot :: honestSlots p m rest
  | m, some a :: rest =>
    ⟨false, a.isWrite, a.isRam, a.addr, a.vr, a.vw, (m.memory (a.globalIndex p)).stamp⟩ ::
      honestSlots p (m.apply p a) rest

private theorem honestSlots_ports (p : Plan) (m : Machine) (ports : List (Option PortAccess)) :
    (honestSlots p m ports).map OpSlot.port = ports := by
  induction ports generalizing m with
  | nil => rfl
  | cons port rest ih =>
    cases port with
    | none => simp [honestSlots, OpSlot.port, OpSlot.padSlot, ih]
    | some a =>
      simp only [honestSlots, List.map_cons, ih]
      rfl

private theorem honestSlots_length (p : Plan) (m : Machine) (ports : List (Option PortAccess)) :
    (honestSlots p m ports).length = ports.length := by
  simpa using congrArg List.length (honestSlots_ports p m ports)

private theorem honestSlots_filterMap (p : Plan) (m : Machine)
    (ports : List (Option PortAccess)) :
    (honestSlots p m ports).filterMap OpSlot.port = ports.filterMap id := by
  conv_rhs => rw [← honestSlots_ports p m ports]
  rw [List.filterMap_map]
  rfl

private theorem activeOps_honestSlots (p : Plan) (m : Machine)
    (ports : List (Option PortAccess)) :
    activeOps m.ts (honestSlots p m ports) = honestOps p m (ports.filterMap id) := by
  induction ports generalizing m with
  | nil => rfl
  | cons port rest ih =>
    cases port with
    | none => exact ih m
    | some a =>
      show (⟨⟨a.isWrite, a.isRam, a.addr, a.vr, a.vw⟩, (m.memory (a.globalIndex p)).stamp,
          m.ts + 1⟩ : MemOp) :: activeOps (m.apply p a).ts (honestSlots p (m.apply p a) rest) =
        ⟨a, (m.memory (a.globalIndex p)).stamp, m.ts + 1⟩ ::
          honestOps p (m.apply p a) (rest.filterMap id)
      rw [ih]

private theorem padSlot_fits_rows (p : Plan) : OpSlot.padSlot.Fits p ∧ OpSlot.padSlot.Rows p :=
  ⟨⟨Nat.two_pow_pos _, Nat.two_pow_pos _, Nat.two_pow_pos _, Nat.two_pow_pos _⟩,
    ⟨fun _ => rfl, fun h => absurd h (by decide), fun _ => Nat.two_pow_pos _, fun _ => rfl⟩⟩

private theorem activeSlot_fits_rows {p : Plan} (valid : p.Valid) {a : PortAccess}
    (ok : a.Valid p) {t : ℕ} (ht : t < 2 ^ p.wTs) :
    (OpSlot.mk false a.isWrite a.isRam a.addr a.vr a.vw t).Fits p ∧
      (OpSlot.mk false a.isWrite a.isRam a.addr a.vr a.vw t).Rows p := by
  obtain ⟨range, vr, vw, keeps, noRom⟩ := ok
  have rom : a.isRam = false → a.addr < p.romSize := fun h => by simpa [h] using range
  have addr : a.addr < 2 ^ p.μ := by
    cases h : a.isRam
    · exact lt_of_lt_of_le (rom h) (Plan.romSize_le_ramSize valid)
    · have ram : a.addr < p.ramSize := by simpa [h] using range
      exact ram
  exact ⟨⟨addr, vr, vw, ht⟩, ⟨keeps, noRom, rom, fun h => by simp at h⟩⟩

private theorem honestSlots_fit {p : Plan} (valid : p.Valid) {m : Machine}
    {ports : List (Option PortAccess)} (below : ∀ g, (m.memory g).stamp ≤ m.ts)
    (bound : m.ts + (ports.filterMap id).length < 2 ^ p.wTs)
    (accessValid : ∀ a ∈ ports.filterMap id, a.Valid p) :
    ∀ s ∈ honestSlots p m ports, s.Fits p ∧ s.Rows p := by
  induction ports generalizing m with
  | nil => intro s hs; cases hs
  | cons port rest ih =>
    intro s hs
    cases port with
    | none =>
      rcases List.mem_cons.1 hs with rfl | hs
      · exact padSlot_fits_rows p
      · exact ih below bound accessValid s hs
    | some a =>
      have bound' : m.ts + ((rest.filterMap id).length + 1) < 2 ^ p.wTs := bound
      have accessValid' : ∀ b ∈ a :: rest.filterMap id, b.Valid p := accessValid
      rcases List.mem_cons.1 hs with rfl | hs
      · have stamp := below (a.globalIndex p)
        exact activeSlot_fits_rows valid (accessValid' a (List.mem_cons.2 (Or.inl rfl)))
          (by omega)
      · refine ih (apply_below a below) ?_ (fun b hb => accessValid' b (List.mem_cons.2 (Or.inr hb)))
          s hs
        show m.ts + 1 + (rest.filterMap id).length < 2 ^ p.wTs
        omega

/-- The records of step `i`: its slots, and chunk `i mod N` of the snapshots at
the open and the close of its segment. -/
private def honestRecords (p : Plan) (ms : ℕ → Machine) (steps : List (List (Option PortAccess)))
    (i : ℕ) : StepRecords :=
  ⟨honestSlots p (ms i) (steps.getD i []), scanOf p (ms (i / p.n * p.n)).memory (i % p.n),
    scanOf p (ms ((i / p.n + 1) * p.n)).memory (i % p.n)⟩

private theorem honestRecords_segment {p : Plan} {ms : ℕ → Machine}
    {steps : List (List (Option PortAccess))} (npos : 0 < p.n) {k j : ℕ} (hj : j < p.n) :
    honestRecords p ms steps (k * p.n + j) =
      ⟨honestSlots p (ms (k * p.n + j)) (steps.getD (k * p.n + j) []),
        scanOf p (ms (k * p.n)).memory j, scanOf p (ms ((k + 1) * p.n)).memory j⟩ := by
  have div : (k * p.n + j) / p.n = k := by
    rw [Nat.add_comm, Nat.add_mul_div_right _ _ npos, Nat.div_eq_of_lt hj, Nat.zero_add]
  have mod : (k * p.n + j) % p.n = j := by
    rw [Nat.add_comm, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt hj]
  simp only [honestRecords, div, mod]

private theorem activeCount_honestRecords (p : Plan) (ms : ℕ → Machine)
    (steps : List (List (Option PortAccess))) (i : ℕ) :
    activeCount (honestRecords p ms steps i) = (stepAccesses steps i).length := by
  rw [activeCount, show (honestRecords p ms steps i).ops = honestSlots p (ms i) (steps.getD i [])
    from rfl, honestSlots_filterMap]
  rfl

section Segment

variable {p : Plan} {steps : List (List (Option PortAccess))} {ms : ℕ → Machine} {S : ℕ}

private theorem honest_stepRows (valid : p.Valid) (width : ∀ ports ∈ steps, ports.length = p.bOps)
    (count : steps.length = S * p.n) (bound : steps.length * p.bOps < 2 ^ p.wTs)
    (tr : Trace p steps ms) {i : ℕ} (hi : i < steps.length) :
    StepRows p (ms i).ts (honestRecords p ms steps i) := by
  have npos : 0 < p.n := valid.positive.2.2.1
  have hk : i / p.n + 1 ≤ S := (Nat.div_lt_iff_lt_mul npos).2 (count ▸ hi)
  have hstart : i / p.n * p.n ≤ steps.length := le_trans (Nat.div_mul_le_self i p.n) hi.le
  have hend : (i / p.n + 1) * p.n ≤ steps.length := by
    rw [count]
    exact Nat.mul_le_mul_right _ hk
  have scans : ∀ j ≤ steps.length, ∀ c ∈ scanOf p (ms j).memory (i % p.n), c.Fits p :=
    fun j hj => (scansOf_shaped valid (trace_fits valid width bound tr hj) _
      (List.mem_map.2 ⟨i % p.n, List.mem_range.2 (Nat.mod_lt i npos), rfl⟩)).2
  have next : (ms i).ts + (stepAccesses steps i).length < 2 ^ p.wTs := by
    rw [← trace_ts_succ tr hi]
    exact trace_ts_lt width bound tr hi
  have slots := honestSlots_fit valid (trace_below tr i hi.le) next (trace_valid tr hi)
  refine ⟨⟨?_, ?_, ?_, fun s hs => (slots s hs).1, scans _ hstart, scans _ hend⟩,
    fun s hs => (slots s hs).2, ?_, ?_⟩
  · rw [show (honestRecords p ms steps i).ops = honestSlots p (ms i) (steps.getD i []) from rfl,
      honestSlots_length]
    exact width _ (getD_mem hi)
  · simp [honestRecords, scanOf]
  · simp [honestRecords, scanOf]
  · show ∀ o ∈ activeOps (ms i).ts (honestSlots p (ms i) (steps.getD i [])), o.rt < o.wt
    rw [activeOps_honestSlots]
    exact (honest_balanced p (trace_below tr i hi.le) (tr.next i hi)).2.1
  · rw [activeCount_honestRecords]
    exact next

private theorem segment_rows (valid : p.Valid) (width : ∀ ports ∈ steps, ports.length = p.bOps)
    (count : steps.length = S * p.n) (bound : steps.length * p.bOps < 2 ^ p.wTs)
    (tr : Trace p steps ms) :
    ∀ len i₀, i₀ + len ≤ steps.length →
      SegmentRows p (ms i₀).ts ((List.range' i₀ len).map (honestRecords p ms steps))
  | 0, _, _ => trivial
  | len + 1, i₀, h => by
    have hi : i₀ < steps.length := by omega
    rw [List.range'_succ, List.map_cons, SegmentRows, activeCount_honestRecords,
      ← trace_ts_succ tr hi]
    exact ⟨honest_stepRows valid width count bound tr hi,
      segment_rows valid width count bound tr len (i₀ + 1) (by omega)⟩

private theorem segment_accessAll (tr : Trace p steps ms) :
    ∀ len i₀, i₀ + len ≤ steps.length →
      accessAll p (ms i₀) ((List.range' i₀ len).flatMap (stepAccesses steps)) =
        some (ms (i₀ + len))
  | 0, _, _ => rfl
  | len + 1, i₀, h => by
    have hi : i₀ < steps.length := by omega
    rw [List.range'_succ, List.flatMap_cons, accessAll_append, tr.next i₀ hi,
      show i₀ + (len + 1) = i₀ + 1 + len by omega]
    exact segment_accessAll tr len (i₀ + 1) (by omega)

private theorem segment_ops (tr : Trace p steps ms) :
    ∀ len i₀, i₀ + len ≤ steps.length →
      segmentOps (ms i₀).ts ((List.range' i₀ len).map (honestRecords p ms steps)) =
        honestOps p (ms i₀) ((List.range' i₀ len).flatMap (stepAccesses steps))
  | 0, _, _ => rfl
  | len + 1, i₀, h => by
    have hi : i₀ < steps.length := by omega
    rw [List.range'_succ, List.map_cons, List.flatMap_cons, segmentOps, honestOps_append,
      activeCount_honestRecords, ← trace_ts_succ tr hi, ← trace_replay tr hi,
      segment_ops tr len (i₀ + 1) (by omega),
      show (honestRecords p ms steps i₀).ops = honestSlots p (ms i₀) (steps.getD i₀ []) from rfl,
      activeOps_honestSlots]
    rfl

private theorem segment_scans (npos : 0 < p.n) (k : ℕ) :
    ((List.range' (k * p.n) p.n).map (honestRecords p ms steps)).map StepRecords.initialScan =
        scansOf p (ms (k * p.n)).memory ∧
      ((List.range' (k * p.n) p.n).map (honestRecords p ms steps)).map StepRecords.finalScan =
        scansOf p (ms ((k + 1) * p.n)).memory := by
  simp only [List.range'_eq_map_range, List.map_map, scansOf]
  constructor <;> refine List.map_congr_left fun j hj => ?_ <;>
    simp only [Function.comp_apply, honestRecords_segment npos (List.mem_range.1 hj)]

private theorem segment_lanes (npos : 0 < p.n) (k : ℕ) :
    ((List.range' (k * p.n) p.n).map (honestRecords p ms steps)).map (initialPacked p) =
        memoryLanes p (ms (k * p.n)).memory ∧
      ((List.range' (k * p.n) p.n).map (honestRecords p ms steps)).map (finalPacked p) =
        memoryLanes p (ms ((k + 1) * p.n)).memory := by
  obtain ⟨initial, final⟩ := segment_scans (ms := ms) (steps := steps) npos k
  rw [memoryLanes_eq, memoryLanes_eq, ← initial, ← final]
  simp only [List.map_map]
  exact ⟨rfl, rfl⟩

end Segment

/-! ### The honest run -/

/-- The proposals of segment `k`: its ops root and the FS root of the memory at
its close. -/
private def honestProposal (ctx : Context E Digest) (ms : ℕ → Machine)
    (steps : List (List (Option PortAccess))) (k : ℕ) : Digest × Digest :=
  (chainRoot ctx.hash .ops ctx.planDigest
      ((List.range' (k * ctx.plan.n) ctx.plan.n).map fun i =>
        opsPacked ctx.plan (honestRecords ctx.plan ms steps i)),
    memoryRoot ctx.hash ctx.plan ctx.planDigest (ms ((k + 1) * ctx.plan.n)).memory)

private def honestInvocation (ctx : Context E Digest) (ms : ℕ → Machine) (st : ℕ → σ)
    (steps : List (List (Option PortAccess))) (i : ℕ) : Invocation σ Digest :=
  ⟨honestProposal ctx ms steps (i / ctx.plan.n), honestRecords ctx.plan ms steps i, st (i + 1)⟩

private def honestRun (ctx : Context E Digest) (ms : ℕ → Machine) (st : ℕ → σ)
    (steps : List (List (Option PortAccess))) : List (Invocation σ Digest) :=
  (List.range steps.length).map (honestInvocation ctx ms st steps)

section Run

omit [CommRing E]

variable {ctx : Context E Digest} {steps : List (List (Option PortAccess))} {ms : ℕ → Machine}
  {st : ℕ → σ}

private theorem honestRun_length : (honestRun ctx ms st steps).length = steps.length := by
  simp [honestRun]

private theorem honestRun_ports : (honestRun ctx ms st steps).map Invocation.ports = steps := by
  refine List.ext_getElem (by simp [honestRun]) fun i _ hi => ?_
  simp [honestRun, honestInvocation, Invocation.ports, honestRecords, honestSlots_ports, hi]

private theorem sum_activeCount (tr : Trace ctx.plan steps ms) :
    ∀ i ≤ steps.length,
      ((List.range i).map fun j => activeCount (honestRecords ctx.plan ms steps j)).sum = (ms i).ts
  | 0, _ => by
    rw [tr.start]
    rfl
  | i + 1, hi => by
    rw [List.range_succ, List.map_append, List.sum_append,
      sum_activeCount tr i (Nat.le_of_succ_le hi), trace_ts_succ tr hi,
      ← activeCount_honestRecords ctx.plan ms steps i]
    simp

private theorem honestRun_tsBefore (tr : Trace ctx.plan steps ms) {i : ℕ}
    (hi : i ≤ steps.length) : tsBefore (honestRun ctx ms st steps) i = (ms i).ts := by
  rw [tsBefore, honestRun, ← List.map_take, List.take_range, Nat.min_eq_left hi, List.map_map]
  exact sum_activeCount tr i hi

private theorem honestRun_segment {k : ℕ} (hk : (k + 1) * ctx.plan.n ≤ steps.length) :
    segmentRun ctx.plan (honestRun ctx ms st steps) k =
      (List.range' (k * ctx.plan.n) ctx.plan.n).map (honestInvocation ctx ms st steps) := by
  have e : (k + 1) * ctx.plan.n = k * ctx.plan.n + ctx.plan.n := Nat.add_one_mul _ _
  rw [segmentRun, honestRun, ← List.map_drop, ← List.map_take, List.range_eq_range',
    List.drop_range', List.take_range'_of_length_ge (by omega)]
  simp

private theorem honestRun_proposal (npos : 0 < ctx.plan.n) {k : ℕ}
    (hk : k * ctx.plan.n < steps.length) :
    proposalAt ctx (honestRun ctx ms st steps) k = honestProposal ctx ms steps k := by
  rw [proposalAt, honestRun, ← List.map_drop, List.range_eq_range', List.drop_range',
    List.head?_map, List.head?_range', if_neg (by omega)]
  simp [honestInvocation, Nat.mul_div_cancel _ npos]

private theorem honestRun_app {app : Application ctx.plan σ}
    (stStep : ∀ i < steps.length, app.Step (st i) (steps.getD i []) (st (i + 1))) :
    ∀ len i₀, i₀ + len ≤ steps.length →
      AppThread app (st i₀) ((List.range' i₀ len).map (honestInvocation ctx ms st steps))
        (st (i₀ + len))
  | 0, _, _ => rfl
  | len + 1, i₀, h => by
    rw [List.range'_succ, List.map_cons]
    refine ⟨?_, ?_⟩
    · have step := stStep i₀ (by omega)
      rwa [← honestSlots_ports ctx.plan (ms i₀) (steps.getD i₀ [])] at step
    · have rest := honestRun_app stStep len (i₀ + 1) (by omega)
      rwa [show i₀ + 1 + len = i₀ + (len + 1) by omega] at rest

end Run

section Closes

variable {ctx : Context E Digest} {steps : List (List (Option PortAccess))} {ms : ℕ → Machine}
  {st : ℕ → σ}

omit [CommRing E] in
private theorem honest_segmentView (valid : ctx.plan.Valid) {S : ℕ}
    (count : steps.length = S * ctx.plan.n) (tr : Trace ctx.plan steps ms) {k : ℕ}
    (hk : k < S) :
    segmentView ctx (honestRun ctx ms st steps) k =
      ⟨k, (ms (k * ctx.plan.n)).ts,
        memoryRoot ctx.hash ctx.plan ctx.planDigest (ms (k * ctx.plan.n)).memory,
        honestProposal ctx ms steps k,
        (List.range' (k * ctx.plan.n) ctx.plan.n).map (honestRecords ctx.plan ms steps)⟩ := by
  have npos : 0 < ctx.plan.n := valid.positive.2.2.1
  have e : (k + 1) * ctx.plan.n = k * ctx.plan.n + ctx.plan.n := Nat.add_one_mul _ _
  have hend : (k + 1) * ctx.plan.n ≤ steps.length := by
    rw [count]
    exact Nat.mul_le_mul_right _ hk
  have hstart : k * ctx.plan.n < steps.length := by omega
  rw [segmentView.eq_def, honestRun_tsBefore tr hstart.le, honestRun_proposal npos hstart,
    honestRun_segment hend, List.map_map, SegmentView.mk.injEq]
  refine ⟨rfl, rfl, ?_, rfl, rfl⟩
  cases k with
  | zero =>
    rw [Nat.zero_mul, tr.start]
    rfl
  | succ k =>
    have hprev : k * ctx.plan.n < steps.length := by
      have := Nat.add_one_mul k ctx.plan.n
      omega
    show (proposalAt ctx (honestRun ctx ms st steps) k).2 = _
    rw [honestRun_proposal npos hprev]
    rfl

private theorem honest_closes (valid : ctx.plan.Valid)
    (width : ∀ ports ∈ steps, ports.length = ctx.plan.bOps) {S : ℕ}
    (count : steps.length = S * ctx.plan.n)
    (bound : steps.length * ctx.plan.bOps < 2 ^ ctx.plan.wTs) (tr : Trace ctx.plan steps ms)
    {k : ℕ} (hk : k < S) (η : E × E) :
    (segmentView ctx (honestRun ctx ms st steps) k).ClosesAt ctx η := by
  have npos : 0 < ctx.plan.n := valid.positive.2.2.1
  have e : (k + 1) * ctx.plan.n = k * ctx.plan.n + ctx.plan.n := Nat.add_one_mul _ _
  have hend : k * ctx.plan.n + ctx.plan.n ≤ steps.length := by
    rw [← e, count]
    exact Nat.mul_le_mul_right _ hk
  obtain ⟨initialScans, finalScans⟩ := segment_scans (ms := ms) (steps := steps) npos k
  obtain ⟨initialLanes, finalLanes⟩ := segment_lanes (ms := ms) (steps := steps) npos k
  rw [honest_segmentView valid count tr hk]
  refine ⟨by simp, segment_rows valid width count bound tr ctx.plan.n (k * ctx.plan.n) hend,
    ?_, congrArg (chainRoot ctx.hash .mem ctx.planDigest) initialLanes,
    congrArg (chainRoot ctx.hash .mem ctx.planDigest) finalLanes, productEq_of_balanced η ?_⟩
  · simp only [SegmentView.opsLanes, List.map_map]
    rfl
  · obtain ⟨-, -, -, balanced, -, -⟩ := honest_balanced ctx.plan
      (trace_below tr (k * ctx.plan.n) (by omega))
      (segment_accessAll tr ctx.plan.n (k * ctx.plan.n) hend)
    simp only [Multisets.Balanced, SegmentView.multisets]
    rw [segmentMultisets_initial, segmentMultisets_write, segmentMultisets_read,
      segmentMultisets_final, initialScans, finalScans, chunkTuples_scansOf valid,
      chunkTuples_scansOf valid, segment_ops tr ctx.plan.n (k * ctx.plan.n) hend, e]
    exact balanced

end Closes

/-- Ob5. -/
theorem completeness {ctx : Context E Digest} {app : Application ctx.plan σ}
    {s₀ s : σ} {final : Machine} {steps : List (List (Option PortAccess))} {segments : ℕ}
    (valid : ctx.plan.Valid) (width : ∀ ports ∈ steps, ports.length = ctx.plan.bOps)
    (count : steps.length = segments * ctx.plan.n)
    (range : 1 ≤ segments ∧ segments ≤ ctx.plan.sMax)
    (exec : Executes ctx.plan app s₀ (initialMachine ctx.plan) steps s final) :
    ∃ run : List (Invocation σ Digest),
      Accepts ctx app (honestStatement ctx s₀ s final segments) run ∧
        run.map Invocation.ports = steps := by
  have npos : 0 < ctx.plan.n := valid.positive.2.2.1
  obtain ⟨next, last, st, st0, stStep, stLast⟩ := executes_trace exec
  have tr : Trace ctx.plan steps (machineAt ctx.plan (initialMachine ctx.plan) steps) :=
    ⟨rfl, next⟩
  have bound : steps.length * ctx.plan.bOps < 2 ^ ctx.plan.wTs := by
    rw [count]
    exact lt_of_le_of_lt (Nat.mul_le_mul_right _ (Nat.mul_le_mul_right _ range.2))
      valid.timestampRange
  have hlen : (honestRun ctx (machineAt ctx.plan (initialMachine ctx.plan) steps) st steps).length =
      segments * ctx.plan.n := by
    rw [honestRun_length, count]
  obtain ⟨c, hc⟩ := (finalCarry_isSome_iff valid hlen).2
    ⟨range.2, fun k hk => honest_closes valid width count bound tr hk _⟩
  obtain ⟨hidx, hseg, hts, hroot⟩ := finalCarry_fields valid hlen range.1 hc
  have hlast : (segments - 1) * ctx.plan.n < steps.length := by
    rw [count]
    exact Nat.mul_lt_mul_of_pos_right (by omega) npos
  refine ⟨_, ⟨hlen, range, rfl, ⟨c, hc, hidx, hseg, ?_, ?_⟩, ?_⟩, honestRun_ports⟩
  · rw [hts, honestRun_length, honestRun_tsBefore tr le_rfl, last]
    rfl
  · rw [hroot, honestRun_proposal npos hlast]
    show memoryRoot ctx.hash ctx.plan ctx.planDigest
        (machineAt ctx.plan (initialMachine ctx.plan) steps
          ((segments - 1 + 1) * ctx.plan.n)).memory = _
    rw [Nat.sub_add_cancel range.1, ← count, last]
    rfl
  · have thread := honestRun_app (ctx := ctx)
      (ms := machineAt ctx.plan (initialMachine ctx.plan) steps) stStep steps.length 0 (by simp)
    rw [Nat.zero_add, ← List.range_eq_range', st0, stLast] at thread
    exact thread

end NightstreamFPrime.Spec.Nebula
