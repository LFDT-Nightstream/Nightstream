import NightstreamFPrime.Spec.Nebula.Fingerprint

/-! Owns the per-step memory rows of spec §8 at the typed level and security
note Lemma 4: the running products of rows O8, O9, S2, and S3 equal the
fingerprint products of the step's multisets, and every tuple is small. The
field-level meaning of rows O1, O4, and the O8 gate is in `FieldRows`. The
tuple functions take no challenge, which is hash residency (spec §6.4, Ob6). -/

namespace NightstreamFPrime.Spec.Nebula

/-- Rows O3, O5, O6, and O7 of one operation slot. -/
structure OpSlot.Rows (p : Plan) (s : OpSlot) : Prop where
  readKeeps : s.isWrite = false → s.vw = s.vr
  noRomWrite : s.isWrite = true → s.isRam = true
  romRange : s.isRam = false → s.addr < p.romSize
  padZero : s.pad = true → s = OpSlot.padSlot

/-- All memory rows of one step entered with timestamp `ts` (spec §8). -/
structure StepRows (p : Plan) (ts : ℕ) (z : StepRecords) : Prop where
  shaped : z.Shaped p
  slots : ∀ s ∈ z.ops, s.Rows p
  fresh : ∀ o ∈ activeOps ts z.ops, o.rt < o.wt
  boundary : ts + activeCount z < 2 ^ p.wTs

/-- The four running products of the carry (spec §11.1). -/
structure Products (E : Type) where
  read : E
  write : E
  initial : E
  final : E

variable {E : Type} [CommRing E]

/-- Rows O8 and O9 over the slots in order: a pad slot multiplies by `1`, an
active slot by the fingerprint of its read and write tuples. -/
def opsFactors (p : Plan) (η : E × E) : ℕ → List OpSlot → E × E
  | _, [] => (1, 1)
  | ts, s :: rest =>
    if s.pad then opsFactors p η ts rest
    else
      let g := (PortAccess.mk s.isWrite s.isRam s.addr s.vr s.vw).globalIndex p
      let tail := opsFactors p η (ts + 1) rest
      (fingerprint η (s.rt, g, s.vr) * tail.1, fingerprint η (ts + 1, g, s.vw) * tail.2)

/-- Rows S2 and S3 over the scan slots, with structural indices from `base`. -/
def scanFactor (η : E × E) : ℕ → List ScanSlot → E
  | _, [] => 1
  | base, c :: rest => fingerprint η (c.stamp, base, c.value) * scanFactor η (base + 1) rest

/-- The carry products after one step. -/
def stepProducts (p : Plan) (η : E × E) (ts idx : ℕ) (z : StepRecords) (h : Products E) :
    Products E :=
  ⟨h.read * (opsFactors p η ts z.ops).1, h.write * (opsFactors p η ts z.ops).2,
   h.initial * scanFactor η (idx * p.bScan) z.initialScan,
   h.final * scanFactor η (idx * p.bScan) z.finalScan⟩

private theorem product_coe (η : E × E) (l : List Tuple) :
    product η (l : Multiset Tuple) = (l.map (fingerprint η)).prod := by
  simp [product]

private theorem opsFactors_eq (p : Plan) (η : E × E) (ts : ℕ) (ops : List OpSlot) :
    opsFactors p η ts ops =
      (product η ((activeOps ts ops).map (MemOp.read p) : List Tuple),
       product η ((activeOps ts ops).map (MemOp.write p) : List Tuple)) := by
  induction ops generalizing ts with
  | nil => simp [opsFactors, activeOps, product]
  | cons s rest ih =>
    cases hpad : s.pad
    · simp [opsFactors, activeOps, OpSlot.port, hpad, ih, product_coe, MemOp.read, MemOp.write,
        PortAccess.globalIndex]
    · simp [opsFactors, activeOps, OpSlot.port, hpad, ih]

private theorem scanFactor_eq (η : E × E) (base : ℕ) (scan : List ScanSlot) :
    scanFactor η base scan = product η (scanTuplesFrom base scan : List Tuple) := by
  induction scan generalizing base with
  | nil => simp [scanFactor, scanTuplesFrom, product]
  | cons c rest ih => simp [scanFactor, scanTuplesFrom, ih, product_coe]

/-- Security note Lemma 4: the rows compute the fingerprint products of the
step's multisets. -/
theorem stepProducts_eq (p : Plan) (η : E × E) (ts idx : ℕ) (z : StepRecords)
    (h : Products E) :
    stepProducts p η ts idx z h =
      ⟨h.read * product η (stepMultisets p ts idx z).read,
       h.write * product η (stepMultisets p ts idx z).write,
       h.initial * product η (stepMultisets p ts idx z).initial,
       h.final * product η (stepMultisets p ts idx z).final⟩ := by
  simp only [stepProducts, stepMultisets, opsFactors_eq, scanFactor_eq, scanTuples]

private theorem activeOps_length_eq (ts : ℕ) (ops : List OpSlot) :
    (activeOps ts ops).length = (ops.filterMap OpSlot.port).length := by
  induction ops generalizing ts with
  | nil => rfl
  | cons s rest ih => cases hp : s.port <;> simp [activeOps, hp, ih]

/-- The number of active operations does not depend on the entry timestamp. -/
theorem activeOps_length (ts : ℕ) (z : StepRecords) :
    (activeOps ts z.ops).length = activeCount z :=
  activeOps_length_eq ts z.ops

/-- The number of active slots of a list. -/
def activeLength (ops : List OpSlot) : ℕ := (ops.filterMap OpSlot.port).length

theorem activeLength_append (l : List OpSlot) (s : OpSlot) :
    activeLength (l ++ [s]) = activeLength l + if s.pad then 0 else 1 := by
  unfold activeLength
  rw [List.filterMap_append, List.length_append]
  cases h : s.pad <;> simp [OpSlot.port, h]

/-- One more slot at the end of a step's slot list. -/
theorem activeOps_append (ts : ℕ) (l : List OpSlot) (s : OpSlot) :
    activeOps ts (l ++ [s]) = activeOps ts l ++
      match s.port with
      | none => []
      | some a => [⟨a, s.rt, ts + activeLength l + 1⟩] := by
  induction l generalizing ts with
  | nil => cases h : s.port <;> simp [activeOps, activeLength, h]
  | cons t l ih =>
    cases ht : t.port with
    | none =>
      simp only [List.cons_append, activeOps, ht, ih, activeLength, List.filterMap_cons]
    | some a =>
      simp only [List.cons_append, activeOps, ht, ih, List.cons_append, List.cons.injEq, true_and]
      cases hs : s.port <;> simp [activeLength, ht, Nat.add_assoc, Nat.add_comm 1]

variable {E : Type} [CommRing E] in
/-- One more slot at the end: rows O8 and O9 multiply by its gated factors. -/
theorem opsFactors_append (p : Plan) (η : E × E) (ts : ℕ) (l : List OpSlot) (s : OpSlot) :
    opsFactors p η ts (l ++ [s]) =
      ((opsFactors p η ts l).1 *
          (if s.pad then 1 else fingerprint η (s.rt,
            (PortAccess.mk s.isWrite s.isRam s.addr s.vr s.vw).globalIndex p, s.vr)),
        (opsFactors p η ts l).2 *
          (if s.pad then 1 else fingerprint η (ts + activeLength l + 1,
            (PortAccess.mk s.isWrite s.isRam s.addr s.vr s.vw).globalIndex p, s.vw))) := by
  induction l generalizing ts with
  | nil => cases h : s.pad <;> simp [opsFactors, activeLength, OpSlot.port, h]
  | cons t l ih =>
    cases ht : t.pad
    · have length : activeLength (t :: l) = activeLength l + 1 := by
        simp [activeLength, OpSlot.port, ht]
      rw [length, show ts + (activeLength l + 1) + 1 = ts + 1 + activeLength l + 1 by omega]
      simp only [List.cons_append, opsFactors, ht, ih, Bool.false_eq_true, ite_false]
      simp only [mul_assoc]
    · have length : activeLength (t :: l) = activeLength l := by
        simp [activeLength, OpSlot.port, ht]
      rw [length]
      simp only [List.cons_append, opsFactors, ht, ih, ite_true]

variable {E : Type} [CommRing E] in
/-- One more scan slot at the end: rows S2 and S3 multiply by its factor. -/
theorem scanFactor_append (η : E × E) (base : ℕ) (l : List ScanSlot) (c : ScanSlot) :
    scanFactor η base (l ++ [c]) =
      scanFactor η base l * fingerprint η (c.stamp, base + l.length, c.value) := by
  induction l generalizing base with
  | nil => simp [scanFactor]
  | cons d l ih =>
    rw [List.length_cons, show base + (l.length + 1) = base + 1 + l.length by omega]
    simp only [List.cons_append, scanFactor, ih, mul_assoc]

/-- All tuples of the four multisets of a step. -/
def Multisets.all (m : Multisets) : Multiset Tuple := m.read + m.write + m.initial + m.final

/-- An active operation comes from an active slot of the step: same port and
`rt`, with a write stamp after `ts` and at most `ts` plus the active count. -/
theorem mem_activeOps {o : MemOp} {ts : ℕ} {ops : List OpSlot}
    (mem : o ∈ activeOps ts ops) :
    ∃ s ∈ ops, s.port = some o.access ∧ o.rt = s.rt ∧ ts < o.wt ∧
      o.wt ≤ ts + (ops.filterMap OpSlot.port).length := by
  induction ops generalizing ts with
  | nil => simp [activeOps] at mem
  | cons s rest ih =>
    cases hp : s.port with
    | none =>
      simp only [activeOps, hp] at mem
      obtain ⟨s', hs', port, rt, lo, hi⟩ := ih mem
      have len : ((s :: rest).filterMap OpSlot.port).length =
          (rest.filterMap OpSlot.port).length := by
        simp [hp]
      exact ⟨s', List.mem_cons_of_mem _ hs', port, rt, lo, by omega⟩
    | some a =>
      simp only [activeOps, hp, List.mem_cons] at mem
      have len : ((s :: rest).filterMap OpSlot.port).length =
          (rest.filterMap OpSlot.port).length + 1 := by
        simp [hp]
      rcases mem with rfl | mem
      · exact ⟨s, by simp, hp, rfl, Nat.lt_succ_self ts, by show ts + 1 ≤ _; omega⟩
      · obtain ⟨s', hs', port, rt, lo, hi⟩ := ih mem
        exact ⟨s', List.mem_cons_of_mem _ hs', port, rt, by omega, by omega⟩

private theorem mem_scanTuplesFrom {τ : Tuple} {base : ℕ} {scan : List ScanSlot}
    (mem : τ ∈ scanTuplesFrom base scan) :
    ∃ c ∈ scan, ∃ j < scan.length, τ = (c.stamp, base + j, c.value) := by
  induction scan generalizing base with
  | nil => simp [scanTuplesFrom] at mem
  | cons c rest ih =>
    simp only [scanTuplesFrom, List.mem_cons] at mem
    rcases mem with rfl | mem
    · exact ⟨c, by simp, 0, by simp, by simp⟩
    · obtain ⟨c', hc', j, hj, rfl⟩ := ih mem
      refine ⟨c', List.mem_cons_of_mem _ hc', j + 1, by simp; omega, ?_⟩
      rw [Nat.add_assoc, Nat.add_comm 1 j]

private theorem tuple_small {p : Plan} (valid : p.Valid) {τ : Tuple} (ht : τ.1 < 2 ^ p.wTs)
    (hg : τ.2.1 < p.cells) (hv : τ.2.2 < 2 ^ 32) :
    τ.Small ∧ τ.1 < 2 ^ p.wTs ∧ τ.2.1 < p.cells ∧ τ.2.2 < 2 ^ 32 := by
  have hts : 2 ^ p.wTs < goldilocksModulus :=
    lt_of_le_of_lt (Nat.pow_le_pow_right (by norm_num) valid.fieldEncoding)
      (by norm_num [goldilocksModulus])
  have h32 : 2 ^ 32 < goldilocksModulus := by norm_num [goldilocksModulus]
  exact ⟨⟨ht.trans hts, hg.trans valid.belowModulus, hv.trans h32⟩, ht, hg, hv⟩

/-- Security note Lemma 4, ranges: every tuple of a step whose rows hold is
small, with `t < 2 ^ W_ts`, `g < R + M`, and `v < 2 ^ 32`. -/
theorem stepMultisets_small {p : Plan} (valid : p.Valid) {ts idx : ℕ} {z : StepRecords}
    (rows : StepRows p ts z) (hidx : idx < p.n) :
    ∀ τ ∈ (stepMultisets p ts idx z).all, τ.Small ∧ τ.1 < 2 ^ p.wTs ∧ τ.2.1 < p.cells ∧
      τ.2.2 < 2 ^ 32 := by
  have opBound : ∀ o ∈ activeOps ts z.ops, o.rt < 2 ^ p.wTs ∧ o.wt < 2 ^ p.wTs ∧
      o.access.globalIndex p < p.cells ∧ o.access.vr < 2 ^ 32 ∧ o.access.vw < 2 ^ 32 := by
    intro o ho
    obtain ⟨s, hs, port, rt, -, hi⟩ := mem_activeOps ho
    have fit := rows.shaped.opsFit s hs
    have romRange := (rows.slots s hs).romRange
    have hpad : s.pad = false := by
      cases h : s.pad
      · rfl
      · simp [OpSlot.port, h] at port
    have access : o.access = ⟨s.isWrite, s.isRam, s.addr, s.vr, s.vw⟩ := by
      simp only [OpSlot.port, hpad] at port
      exact (Option.some.inj port).symm
    refine ⟨rt ▸ fit.2.2.2, hi.trans_lt rows.boundary, ?_, access ▸ fit.2.1, access ▸ fit.2.2.1⟩
    rw [access]
    unfold PortAccess.globalIndex Plan.cells Plan.ramSize
    cases hr : s.isRam
    · have := romRange hr
      simp only [Bool.false_eq_true, ite_false]
      exact Nat.lt_add_right _ this
    · have := fit.1
      simp only [ite_true]
      omega
  have scanBound : ∀ scan : List ScanSlot, scan.length = p.bScan → (∀ c ∈ scan, c.Fits p) →
      ∀ τ ∈ scanTuplesFrom (idx * p.bScan) scan,
        τ.Small ∧ τ.1 < 2 ^ p.wTs ∧ τ.2.1 < p.cells ∧ τ.2.2 < 2 ^ 32 := by
    intro scan len fit τ hτ
    obtain ⟨c, hc, j, hj, rfl⟩ := mem_scanTuplesFrom hτ
    refine tuple_small valid (fit c hc).2 ?_ (fit c hc).1
    show idx * p.bScan + j < p.cells
    rw [← valid.exactCover]
    calc idx * p.bScan + j < (idx + 1) * p.bScan := by rw [Nat.add_mul, Nat.one_mul]; omega
      _ ≤ p.n * p.bScan := Nat.mul_le_mul_right _ hidx
  intro τ mem
  simp only [Multisets.all, stepMultisets, scanTuples, Multiset.mem_add, Multiset.mem_coe,
    List.mem_map] at mem
  rcases mem with ((⟨o, ho, rfl⟩ | ⟨o, ho, rfl⟩) | hτ) | hτ
  · obtain ⟨rt, -, g, vr, -⟩ := opBound o ho
    exact tuple_small valid rt g vr
  · obtain ⟨-, wt, g, -, vw⟩ := opBound o ho
    exact tuple_small valid wt g vw
  · exact scanBound _ rows.shaped.initialLength rows.shaped.initialFit τ hτ
  · exact scanBound _ rows.shaped.finalLength rows.shaped.finalFit τ hτ

end NightstreamFPrime.Spec.Nebula
