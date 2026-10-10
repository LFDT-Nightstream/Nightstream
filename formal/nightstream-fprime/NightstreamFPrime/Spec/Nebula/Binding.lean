import NightstreamFPrime.Spec.Nebula.Lifecycle

/-! Owns security note Lemmas 1 and 2 at the segment level: equal chain roots
over shaped records give equal records, or a collision between the two chain
computations. It also owns the scan memory, the memory that the IS or FS
chunks of a segment describe. -/

namespace NightstreamFPrime.Spec.Nebula

variable {E Digest σ : Type} [CommRing E]

/-- The scan chunks of a memory: chunk `j` holds the cells `j · B_scan + i`. -/
def scansOf (p : Plan) (mem : Memory) : List (List ScanSlot) := (List.range p.n).map (scanOf p mem)

/-- The memory that scan chunks describe: cell `g` is slot `g mod B_scan` of
chunk `g / B_scan`. -/
def scanMemory (p : Plan) (scans : List (List ScanSlot)) : Memory := fun g =>
  let c := (scans.getD (g / p.bScan) []).getD (g % p.bScan) ⟨0, 0⟩
  ⟨c.value, c.stamp⟩

/-- Cells below `R + M` hold 32-bit values and stamps below `2 ^ W_ts`. -/
def Memory.Fits (p : Plan) (mem : Memory) : Prop :=
  ∀ g < p.cells, (mem g).value < 2 ^ 32 ∧ (mem g).stamp < 2 ^ p.wTs

/-- Scan chunks of the plan's shape. -/
def ScansShaped (p : Plan) (scans : List (List ScanSlot)) : Prop :=
  ∀ c ∈ scans, c.length = p.bScan ∧ ∀ s ∈ c, s.Fits p

/-- Operation lanes of the plan's shape. -/
def OpsShaped (p : Plan) (ops : List (List OpSlot)) : Prop :=
  ∀ c ∈ ops, c.length = p.bOps ∧ ∀ s ∈ c, s.Fits p

/-- The number of packed elements of each lane of a step (spec §6.3, §9.1). -/
def Plan.laneLength (p : Plan) : Lane → ℕ
  | .ops => packedLength (p.bOps * p.opWidth)
  | .mem => packedLength (p.bScan * p.scanWidth)

namespace SegmentView

/-- The ops lanes of a segment, one per step. -/
def opsSlots (v : SegmentView Digest) : List (List OpSlot) := v.records.map StepRecords.ops

/-- The IS chunks of a segment, one per step. -/
def initialScans (v : SegmentView Digest) : List (List ScanSlot) :=
  v.records.map StepRecords.initialScan

/-- The FS chunks of a segment, one per step. -/
def finalScans (v : SegmentView Digest) : List (List ScanSlot) :=
  v.records.map StepRecords.finalScan

end SegmentView

/-- The active operations of consecutive steps from timestamp `ts`. -/
def segmentOps (ts : ℕ) : List StepRecords → List MemOp
  | [] => []
  | z :: rest => activeOps ts z.ops ++ segmentOps (ts + activeCount z) rest

/-- The scan tuples of consecutive chunks from step index `idx`. -/
def chunkTuples (p : Plan) : ℕ → List (List ScanSlot) → Multiset Tuple
  | _, [] => 0
  | idx, c :: rest => (scanTuples p idx c : Multiset Tuple) + chunkTuples p (idx + 1) rest

theorem segmentMultisets_read (p : Plan) (ts idx : ℕ) (rs : List StepRecords) :
    (segmentMultisets p ts idx rs).read =
      (((segmentOps ts rs).map (MemOp.read p) : List Tuple) : Multiset Tuple) := by
  induction rs generalizing ts idx with
  | nil => rfl
  | cons z rest ih =>
    change (stepMultisets p ts idx z).read +
      (segmentMultisets p (ts + activeCount z) (idx + 1) rest).read = _
    rw [ih, segmentOps, List.map_append, ← Multiset.coe_add]
    rfl

theorem segmentMultisets_write (p : Plan) (ts idx : ℕ) (rs : List StepRecords) :
    (segmentMultisets p ts idx rs).write =
      (((segmentOps ts rs).map (MemOp.write p) : List Tuple) : Multiset Tuple) := by
  induction rs generalizing ts idx with
  | nil => rfl
  | cons z rest ih =>
    change (stepMultisets p ts idx z).write +
      (segmentMultisets p (ts + activeCount z) (idx + 1) rest).write = _
    rw [ih, segmentOps, List.map_append, ← Multiset.coe_add]
    rfl

theorem segmentMultisets_initial (p : Plan) (ts idx : ℕ) (rs : List StepRecords) :
    (segmentMultisets p ts idx rs).initial =
      chunkTuples p idx (rs.map StepRecords.initialScan) := by
  induction rs generalizing ts idx with
  | nil => rfl
  | cons z rest ih =>
    change (stepMultisets p ts idx z).initial +
      (segmentMultisets p (ts + activeCount z) (idx + 1) rest).initial = _
    rw [ih]
    rfl

theorem segmentMultisets_final (p : Plan) (ts idx : ℕ) (rs : List StepRecords) :
    (segmentMultisets p ts idx rs).final = chunkTuples p idx (rs.map StepRecords.finalScan) := by
  induction rs generalizing ts idx with
  | nil => rfl
  | cons z rest ih =>
    change (stepMultisets p ts idx z).final +
      (segmentMultisets p (ts + activeCount z) (idx + 1) rest).final = _
    rw [ih]
    rfl

private theorem activeOps_access (ts : ℕ) (ops : List OpSlot) :
    (activeOps ts ops).map MemOp.access = ops.filterMap OpSlot.port := by
  induction ops generalizing ts with
  | nil => rfl
  | cons s rest ih =>
    cases h : s.port <;> simp [activeOps, h, ih]

private theorem activeOps_wt (ts : ℕ) (ops : List OpSlot) (k : ℕ)
    (hk : k < (activeOps ts ops).length) : ((activeOps ts ops)[k]).wt = ts + k + 1 := by
  induction ops generalizing ts k with
  | nil => simp [activeOps] at hk
  | cons s rest ih =>
    cases h : s.port with
    | none =>
      simp only [activeOps, h] at hk ⊢
      exact ih ts k hk
    | some a =>
      simp only [activeOps, h] at hk ⊢
      cases k with
      | zero => rfl
      | succ k =>
        simp only [List.getElem_cons_succ]
        rw [ih (ts + 1) k (by simpa using hk)]
        omega

/-- Write stamps are consecutive from `ts + 1` (row O2, carried `ts`). -/
theorem segmentOps_wt (ts : ℕ) (rs : List StepRecords) (k : ℕ)
    (hk : k < (segmentOps ts rs).length) :
    ((segmentOps ts rs).get ⟨k, hk⟩).wt = ts + k + 1 := by
  induction rs generalizing ts k with
  | nil => simp [segmentOps] at hk
  | cons z rest ih =>
    have hlen : (activeOps ts z.ops).length = activeCount z := activeOps_length ts z
    simp only [List.get_eq_getElem, segmentOps] at hk ⊢
    rw [List.getElem_append]
    split_ifs with h
    · exact activeOps_wt ts z.ops k h
    · have hk' : k - (activeOps ts z.ops).length <
          (segmentOps (ts + activeCount z) rest).length := by
        rw [List.length_append] at hk
        omega
      have := ih (ts + activeCount z) _ hk'
      simp only [List.get_eq_getElem] at this
      rw [this]
      omega

theorem segmentOps_length (ts : ℕ) (rs : List StepRecords) :
    (segmentOps ts rs).length = (rs.map activeCount).sum := by
  induction rs generalizing ts with
  | nil => rfl
  | cons z rest ih =>
    rw [segmentOps, List.length_append, ih, activeOps_length, List.map_cons, List.sum_cons]

theorem segmentOps_access (ts : ℕ) (rs : List StepRecords) :
    (segmentOps ts rs).map MemOp.access = rs.flatMap fun z => z.ops.filterMap OpSlot.port := by
  induction rs generalizing ts with
  | nil => rfl
  | cons z rest ih =>
    rw [segmentOps, List.map_append, ih, activeOps_access, List.flatMap_cons]

private theorem scanTuplesFrom_range' (mem : Memory) (base n : ℕ) :
    scanTuplesFrom base ((List.range' base n).map fun g => ⟨(mem g).value, (mem g).stamp⟩) =
      (List.range' base n).map fun g => ((mem g).stamp, g, (mem g).value) := by
  induction n generalizing base with
  | zero => rfl
  | succ n ih => simp only [List.range'_succ, List.map_cons, scanTuplesFrom, ih]

private theorem scanOf_eq (p : Plan) (mem : Memory) (j : ℕ) :
    scanOf p mem j = (List.range' (j * p.bScan) p.bScan).map fun g =>
      (⟨(mem g).value, (mem g).stamp⟩ : ScanSlot) := by
  apply List.ext_getElem (by simp [scanOf])
  intro i h₁ h₂
  simp [scanOf, List.getElem_range']

private theorem chunkTuples_range' (p : Plan) (mem : Memory) (idx k : ℕ) :
    chunkTuples p idx ((List.range' idx k).map (scanOf p mem)) =
      (((List.range' (idx * p.bScan) (k * p.bScan)).map fun g =>
        ((mem g).stamp, g, (mem g).value) : List Tuple) : Multiset Tuple) := by
  induction k generalizing idx with
  | zero => simp [chunkTuples]
  | succ k ih =>
    have e₁ : (idx + 1) * p.bScan = idx * p.bScan + 1 * p.bScan := by ring
    have e₂ : (k + 1) * p.bScan = p.bScan + k * p.bScan := by ring
    rw [List.range'_succ, List.map_cons, chunkTuples, ih, scanTuples, scanOf_eq,
      scanTuplesFrom_range', Multiset.coe_add, ← List.map_append, e₁, e₂, List.range'_append]

/-- The scan tuples of a memory's own chunks are its snapshot (exact cover). -/
theorem chunkTuples_scansOf {p : Plan} (valid : p.Valid) (mem : Memory) :
    chunkTuples p 0 (scansOf p mem) = snapshot p mem := by
  rw [scansOf, List.range_eq_range', chunkTuples_range', Nat.zero_mul, valid.exactCover, snapshot,
    List.range_eq_range']

theorem initialMemory_fits {p : Plan} (valid : p.Valid) : (initialMemory p).Fits p := by
  intro g hg
  unfold initialMemory
  split_ifs with h
  · exact ⟨valid.romWords g h, Nat.two_pow_pos _⟩
  · refine ⟨valid.ramWords _ ?_, Nat.two_pow_pos _⟩
    have : g < p.romSize + p.ramSize := hg
    omega

private theorem cell_lt {p : Plan} (valid : p.Valid) {j i : ℕ} (hj : j < p.n) (hi : i < p.bScan) :
    j * p.bScan + i < p.cells := by
  rw [← valid.exactCover]
  calc j * p.bScan + i < j * p.bScan + p.bScan := by omega
    _ = (j + 1) * p.bScan := by ring
    _ ≤ p.n * p.bScan := Nat.mul_le_mul_right _ hj

theorem scansOf_shaped {p : Plan} (valid : p.Valid) {mem : Memory} (fits : mem.Fits p) :
    ScansShaped p (scansOf p mem) := by
  intro c hc
  simp only [scansOf, List.mem_map, List.mem_range] at hc
  obtain ⟨j, hj, rfl⟩ := hc
  refine ⟨by simp [scanOf], fun s hs => ?_⟩
  simp only [scanOf, List.mem_map, List.mem_range] at hs
  obtain ⟨i, hi, rfl⟩ := hs
  exact fits _ (cell_lt valid hj hi)

theorem memoryLanes_eq (p : Plan) (mem : Memory) :
    memoryLanes p mem = (scansOf p mem).map fun c => pack (scanLane p c) := by
  rw [memoryLanes, scansOf, List.map_map]
  rfl

/-- The scan memory of a memory's own chunks agrees with it below `R + M`. -/
theorem scanMemory_scansOf {p : Plan} (valid : p.Valid) (mem : Memory) :
    Set.EqOn (scanMemory p (scansOf p mem)) mem (Set.Iio p.cells) := by
  intro g hg
  have hb := valid.positive.2.1
  have hj : g / p.bScan < p.n := by
    rw [Nat.div_lt_iff_lt_mul hb, valid.exactCover]
    exact hg
  have hi : g % p.bScan < p.bScan := Nat.mod_lt _ hb
  simp [scanMemory, scansOf, scanOf, List.getD_eq_getElem?_getD, hj, hi, Nat.div_add_mod']

private theorem div_mod_cell {b j i : ℕ} (hi : i < b) :
    (j * b + i) / b = j ∧ (j * b + i) % b = i := by
  have hb : 0 < b := Nat.lt_of_le_of_lt (Nat.zero_le _) hi
  rw [Nat.add_comm, Nat.add_mul_div_right _ _ hb, Nat.div_eq_of_lt hi, Nat.zero_add,
    Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt hi]
  exact ⟨rfl, rfl⟩

/-- A memory that agrees with the scan memory of shaped chunks has those
chunks as its own. -/
theorem scansOf_scanMemory {p : Plan} (valid : p.Valid) {scans : List (List ScanSlot)}
    (length : scans.length = p.n) (shaped : ScansShaped p scans) {mem : Memory}
    (agree : Set.EqOn mem (scanMemory p scans) (Set.Iio p.cells)) :
    scansOf p mem = scans := by
  apply List.ext_getElem (by simp [scansOf, length])
  intro j h₁ h₂
  have hj : j < p.n := length ▸ h₂
  have hlen := (shaped _ (List.getElem_mem h₂)).1
  apply List.ext_getElem (by simp [scansOf, scanOf, hlen])
  intro i h₃ h₄
  have hi : i < p.bScan := hlen ▸ h₄
  obtain ⟨hdiv, hmod⟩ := div_mod_cell (j := j) hi
  simp only [scansOf, scanOf, List.getElem_map, List.getElem_range]
  rw [agree (cell_lt valid hj hi)]
  simp [scanMemory, hdiv, hmod, List.getD_eq_getElem?_getD, h₂, h₄]

private theorem eq_of_map_eq {α β : Type} {f : α → β} {xs ys : List α}
    (inj : ∀ a ∈ xs, ∀ b ∈ ys, f a = f b → a = b) (same : xs.map f = ys.map f) : xs = ys := by
  induction xs generalizing ys with
  | nil => cases ys with
    | nil => rfl
    | cons => simp at same
  | cons x xs ih => cases ys with
    | nil => simp at same
    | cons y ys =>
      simp only [List.map_cons, List.cons.injEq] at same
      rw [inj x (by simp) y (by simp) same.1,
        ih (fun a ha b hb => inj a (by simp [ha]) b (by simp [hb])) same.2]

private theorem scanLane_length (p : Plan) (c : List ScanSlot) :
    (scanLane p c).length = c.length * p.scanWidth := by
  simp [scanLane, List.length_flatMap, ScanSlot.bits_length]

private theorem opsLane_length (p : Plan) (c : List OpSlot) :
    (opsLane p c).length = c.length * p.opWidth := by
  simp [opsLane, List.length_flatMap, OpSlot.bits_length]

/-- Every packed lane of a valid plan has fewer than `q` elements. -/
theorem Plan.laneLength_lt {p : Plan} (valid : p.Valid) (lane : Lane) :
    p.laneLength lane < goldilocksModulus := by
  have shrink : ∀ n, packedLength n ≤ n := fun n => by unfold packedLength; omega
  cases lane
  · exact (shrink _).trans_lt valid.laneBits.1
  · exact (shrink _).trans_lt valid.laneBits.2

/-- Packed scan lanes of shaped chunks have the plan's length and elements
below `2 ^ 63`. -/
theorem scanLanes_shape {p : Plan} {scans : List (List ScanSlot)} (shaped : ScansShaped p scans) :
    ∀ P ∈ scans.map (fun c => pack (scanLane p c)),
      P.length = p.laneLength .mem ∧ ∀ x ∈ P, x < 2 ^ 63 := by
  intro P member
  obtain ⟨c, hc, rfl⟩ := List.mem_map.1 member
  refine ⟨?_, pack_lt _⟩
  rw [pack_length, scanLane_length, (shaped c hc).1]
  rfl

/-- Packed operation lanes of shaped slots have the plan's length and elements
below `2 ^ 63`. -/
theorem opsLanes_shape {p : Plan} {ops : List (List OpSlot)} (shaped : OpsShaped p ops) :
    ∀ P ∈ ops.map (fun c => pack (opsLane p c)),
      P.length = p.laneLength .ops ∧ ∀ x ∈ P, x < 2 ^ 63 := by
  intro P member
  obtain ⟨c, hc, rfl⟩ := List.mem_map.1 member
  refine ⟨?_, pack_lt _⟩
  rw [pack_length, opsLane_length, (shaped c hc).1]
  rfl

/-- Lemmas 1 and 2 for scan chunks. -/
theorem scans_eq_or_collision (H : HashInput Digest → Digest) (p : Plan) (pd : Digest)
    {xs ys : List (List ScanSlot)} (shapedX : ScansShaped p xs) (shapedY : ScansShaped p ys)
    (same : chainRoot H .mem pd (xs.map fun c => pack (scanLane p c)) =
      chainRoot H .mem pd (ys.map fun c => pack (scanLane p c))) :
    xs = ys ∨ CollisionIn H (chainInputs H .mem pd (xs.map fun c => pack (scanLane p c)))
      (chainInputs H .mem pd (ys.map fun c => pack (scanLane p c))) := by
  refine (chainRoot_eq_or_collision H .mem pd same).imp_left fun lanes => ?_
  refine eq_of_map_eq (fun a ha b hb packed => ?_) lanes
  obtain ⟨lengthA, fitA⟩ := shapedX a ha
  obtain ⟨lengthB, fitB⟩ := shapedY b hb
  exact scanLane_injective fitA fitB (lengthA.trans lengthB.symm)
    (pack_injective (by rw [scanLane_length, scanLane_length, lengthA, lengthB]) packed)

/-- Lemmas 1 and 2 for operation lanes. -/
theorem ops_eq_or_collision (H : HashInput Digest → Digest) (p : Plan) (pd : Digest)
    {xs ys : List (List OpSlot)} (shapedX : OpsShaped p xs) (shapedY : OpsShaped p ys)
    (same : chainRoot H .ops pd (xs.map fun c => pack (opsLane p c)) =
      chainRoot H .ops pd (ys.map fun c => pack (opsLane p c))) :
    xs = ys ∨ CollisionIn H (chainInputs H .ops pd (xs.map fun c => pack (opsLane p c)))
      (chainInputs H .ops pd (ys.map fun c => pack (opsLane p c))) := by
  refine (chainRoot_eq_or_collision H .ops pd same).imp_left fun lanes => ?_
  refine eq_of_map_eq (fun a ha b hb packed => ?_) lanes
  obtain ⟨lengthA, fitA⟩ := shapedX a ha
  obtain ⟨lengthB, fitB⟩ := shapedY b hb
  exact opsLane_injective fitA fitB (lengthA.trans lengthB.symm)
    (pack_injective (by rw [opsLane_length, opsLane_length, lengthA, lengthB]) packed)

private theorem segmentRows_shaped {p : Plan} {ts : ℕ} {rs : List StepRecords}
    (rows : SegmentRows p ts rs) : ∀ z ∈ rs, z.Shaped p := by
  induction rs generalizing ts with
  | nil => simp
  | cons z rest ih =>
    obtain ⟨step, rows⟩ := rows
    intro w hw
    rcases List.mem_cons.1 hw with rfl | hw
    · exact step.shaped
    · exact ih rows w hw

namespace SegmentView.ClosesAt

variable {ctx : Context E Digest} {η : E × E} {v : SegmentView Digest}

theorem opsShaped (closes : v.ClosesAt ctx η) : OpsShaped ctx.plan v.opsSlots := by
  intro c hc
  obtain ⟨z, hz, rfl⟩ := List.mem_map.1 hc
  have shaped := segmentRows_shaped closes.rows z hz
  exact ⟨shaped.opsLength, shaped.opsFit⟩

theorem initialShaped (closes : v.ClosesAt ctx η) : ScansShaped ctx.plan v.initialScans := by
  intro c hc
  obtain ⟨z, hz, rfl⟩ := List.mem_map.1 hc
  have shaped := segmentRows_shaped closes.rows z hz
  exact ⟨shaped.initialLength, shaped.initialFit⟩

theorem finalShaped (closes : v.ClosesAt ctx η) : ScansShaped ctx.plan v.finalScans := by
  intro c hc
  obtain ⟨z, hz, rfl⟩ := List.mem_map.1 hc
  have shaped := segmentRows_shaped closes.rows z hz
  exact ⟨shaped.finalLength, shaped.finalFit⟩

end SegmentView.ClosesAt

namespace SegmentView

private theorem opsLanes_eq (p : Plan) (v : SegmentView Digest) :
    v.opsLanes p = v.opsSlots.map fun c => pack (opsLane p c) := by
  rw [opsLanes, opsSlots, List.map_map]
  rfl

private theorem initialLanes_eq (p : Plan) (v : SegmentView Digest) :
    v.initialLanes p = v.initialScans.map fun c => pack (scanLane p c) := by
  rw [initialLanes, initialScans, List.map_map]
  rfl

private theorem finalLanes_eq (p : Plan) (v : SegmentView Digest) :
    v.finalLanes p = v.finalScans.map fun c => pack (scanLane p c) := by
  rw [finalLanes, finalScans, List.map_map]
  rfl

omit [CommRing E] in
private theorem opsInputs_subset {ctx : Context E Digest} (v : SegmentView Digest) :
    Nebula.chainInputs ctx.hash .ops ctx.planDigest (v.opsLanes ctx.plan) ⊆ v.chainInputs ctx :=
  fun _ h => by simp [chainInputs, h]

omit [CommRing E] in
private theorem initialInputs_subset {ctx : Context E Digest} (v : SegmentView Digest) :
    Nebula.chainInputs ctx.hash .mem ctx.planDigest (v.initialLanes ctx.plan) ⊆
      v.chainInputs ctx :=
  fun _ h => by simp [chainInputs, h]

omit [CommRing E] in
private theorem finalInputs_subset {ctx : Context E Digest} (v : SegmentView Digest) :
    Nebula.chainInputs ctx.hash .mem ctx.planDigest (v.finalLanes ctx.plan) ⊆
      v.chainInputs ctx :=
  fun _ h => by simp [chainInputs, h]

end SegmentView

/-- Every chain input of a closing segment is canonical (Ob3 input range). -/
theorem SegmentView.ClosesAt.chainInputs_canonical {ctx : Context E Digest} {η : E × E}
    {v : SegmentView Digest} (closes : v.ClosesAt ctx η) :
    ∀ x ∈ v.chainInputs ctx, x.Canonical ctx.plan.laneLength ctx.plan.n := by
  have count : v.records.length ≤ ctx.plan.n := closes.length.le
  intro x member
  simp only [SegmentView.chainInputs, List.mem_append] at member
  rcases member with (ops | initial) | final
  · rw [SegmentView.opsLanes_eq] at ops
    exact Nebula.chainInputs_canonical _ _ _ (by simpa [SegmentView.opsSlots] using count)
      (opsLanes_shape closes.opsShaped) x ops
  · rw [SegmentView.initialLanes_eq] at initial
    exact Nebula.chainInputs_canonical _ _ _ (by simpa [SegmentView.initialScans] using count)
      (scanLanes_shape closes.initialShaped) x initial
  · rw [SegmentView.finalLanes_eq] at final
    exact Nebula.chainInputs_canonical _ _ _ (by simpa [SegmentView.finalScans] using count)
      (scanLanes_shape closes.finalShaped) x final

/-- Every input of the `D_init` chain is canonical. -/
theorem initialInputs_canonical {p : Plan} (valid : p.Valid) (H : HashInput Digest → Digest)
    (pd : Digest) :
    ∀ x ∈ chainInputs H .mem pd (memoryLanes p (initialMemory p)),
      x.Canonical p.laneLength p.n := by
  rw [memoryLanes_eq]
  exact chainInputs_canonical _ _ _ (by simp [scansOf])
    (scanLanes_shape (scansOf_shaped valid (initialMemory_fits valid)))

/-- Segment 0 reads the plan images, or the IS chain and the `D_init` chain
collide. -/
theorem initial_scans_or_collision {ctx : Context E Digest} {run : List (Invocation σ Digest)}
    {η : E × E} (valid : ctx.plan.Valid) (closes : (segmentView ctx run 0).ClosesAt ctx η) :
    (segmentView ctx run 0).initialScans = scansOf ctx.plan (initialMemory ctx.plan) ∨
      CollisionIn ctx.hash ((segmentView ctx run 0).chainInputs ctx)
        (chainInputs ctx.hash .mem ctx.planDigest
          (memoryLanes ctx.plan (initialMemory ctx.plan))) := by
  have root := closes.initialRoot
  rw [SegmentView.initialLanes_eq] at root
  rw [memoryLanes_eq]
  refine (scans_eq_or_collision ctx.hash ctx.plan ctx.planDigest closes.initialShaped
    (scansOf_shaped valid (initialMemory_fits valid)) ?_).imp_right fun collision => ?_
  · rw [root, ← memoryLanes_eq]
    rfl
  · rw [← SegmentView.initialLanes_eq] at collision
    exact collision.mono (SegmentView.initialInputs_subset _) (List.Subset.refl _)

/-- Segment `k + 1` starts from the final snapshot of segment `k`, or their
chains collide. -/
theorem join_scans_or_collision {ctx : Context E Digest} {run : List (Invocation σ Digest)}
    {η η' : E × E} {k : ℕ} (closes : (segmentView ctx run k).ClosesAt ctx η)
    (closesNext : (segmentView ctx run (k + 1)).ClosesAt ctx η') :
    (segmentView ctx run (k + 1)).initialScans = (segmentView ctx run k).finalScans ∨
      CollisionIn ctx.hash ((segmentView ctx run (k + 1)).chainInputs ctx)
        ((segmentView ctx run k).chainInputs ctx) := by
  refine (scans_eq_or_collision ctx.hash ctx.plan ctx.planDigest closesNext.initialShaped
    closes.finalShaped ?_).imp_right fun collision => ?_
  · rw [← SegmentView.initialLanes_eq, ← SegmentView.finalLanes_eq, closesNext.initialRoot,
      closes.finalRoot]
    rfl
  · rw [← SegmentView.initialLanes_eq, ← SegmentView.finalLanes_eq] at collision
    exact collision.mono (SegmentView.initialInputs_subset _) (SegmentView.finalInputs_subset _)

private theorem records_ext : ∀ {xs ys : List StepRecords},
    xs.map StepRecords.ops = ys.map StepRecords.ops →
      xs.map StepRecords.initialScan = ys.map StepRecords.initialScan →
        xs.map StepRecords.finalScan = ys.map StepRecords.finalScan → xs = ys
  | [], [], _, _, _ => rfl
  | [], _ :: _, h, _, _ => by simp at h
  | _ :: _, [], h, _, _ => by simp at h
  | ⟨_, _, _⟩ :: _, ⟨_, _, _⟩ :: _, h₁, h₂, h₃ => by
    simp only [List.map_cons, List.cons.injEq] at h₁ h₂ h₃
    rw [h₁.1, h₂.1, h₃.1, records_ext h₁.2 h₂.2 h₃.2]

/-- Two closing record sets for the same transcript input are equal, or their
chains collide (Lemma 3 part 2, disagreeing branch). -/
theorem records_eq_or_collision {ctx : Context E Digest} {v w : SegmentView Digest}
    {η η' : E × E} (closesV : v.ClosesAt ctx η) (closesW : w.ClosesAt ctx η')
    (same : v.etaInput ctx = w.etaInput ctx) :
    v.records = w.records ∨ CollisionIn ctx.hash (v.chainInputs ctx) (w.chainInputs ctx) := by
  have sameOps : v.proposal.1 = w.proposal.1 := congrArg EtaInput.opsRoot same
  have sameMem : v.memIn = w.memIn := congrArg EtaInput.memRoot same
  have sameFinal : v.proposal.2 = w.proposal.2 := congrArg EtaInput.finalRoot same
  have ops := ops_eq_or_collision ctx.hash ctx.plan ctx.planDigest closesV.opsShaped
    closesW.opsShaped (by
      rw [← SegmentView.opsLanes_eq, ← SegmentView.opsLanes_eq, closesV.opsRoot, closesW.opsRoot,
        sameOps])
  have initial := scans_eq_or_collision ctx.hash ctx.plan ctx.planDigest closesV.initialShaped
    closesW.initialShaped (by
      rw [← SegmentView.initialLanes_eq, ← SegmentView.initialLanes_eq, closesV.initialRoot,
        closesW.initialRoot, sameMem])
  have final := scans_eq_or_collision ctx.hash ctx.plan ctx.planDigest closesV.finalShaped
    closesW.finalShaped (by
      rw [← SegmentView.finalLanes_eq, ← SegmentView.finalLanes_eq, closesV.finalRoot,
        closesW.finalRoot, sameFinal])
  rw [← SegmentView.opsLanes_eq, ← SegmentView.opsLanes_eq] at ops
  rw [← SegmentView.initialLanes_eq, ← SegmentView.initialLanes_eq] at initial
  rw [← SegmentView.finalLanes_eq, ← SegmentView.finalLanes_eq] at final
  rcases ops with ops | ops
  · rcases initial with initial | initial
    · rcases final with final | final
      · exact Or.inl (records_ext ops initial final)
      · exact Or.inr (final.mono (SegmentView.finalInputs_subset _)
          (SegmentView.finalInputs_subset _))
    · exact Or.inr (initial.mono (SegmentView.initialInputs_subset _)
        (SegmentView.initialInputs_subset _))
  · exact Or.inr (ops.mono (SegmentView.opsInputs_subset _) (SegmentView.opsInputs_subset _))

end NightstreamFPrime.Spec.Nebula
