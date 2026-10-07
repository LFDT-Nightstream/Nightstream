import NightstreamFPrime.Spec.Nebula.Rows
import NightstreamFPrime.Spec.Nebula.Reference

/-! Owns the memory carry of spec §11, the three lifecycle arms of §12, and the
public statement and terminal checks of §13, over an extracted run: one
invocation per step, each with its proposal, its records, and its next
application state. The input carry selects the arm; no invocation field does.
It also owns the segment views that the security proofs read, and the
lifecycle theorem (Ob8): the trace closes every segment exactly when each
segment view passes the close checks with its own challenges.

The Stage 1 fold, the HyperNova state link, and the carry encoding are not
owned here (A1–A4, Ob1). Guards use classical decisions, so the actions are
total and deterministic but not executable. -/

namespace NightstreamFPrime.Spec.Nebula

/-- The three chain digests of a segment: ops, IS, FS. -/
structure Roots (Digest : Type) where
  ops : Digest
  initial : Digest
  final : Digest

/-- Spec §11.1 `MemoryCarry`. `proposed` is `D_pre = (ops, fs)`, `seen` is
`D_seen`, and `memRoot` is `D_mem`. -/
structure Carry (E Digest : Type) where
  segIdx : ℕ
  idx : ℕ
  ts : ℕ
  eta : E × E
  products : Products E
  proposed : Digest × Digest
  seen : Roots Digest
  memRoot : Digest

/-- The verifier context: the plan, the abstract hash, the abstract §9.3
challenge function, and the package constant `plan_digest`. `D_init` is
derived from the plan (`Context.initRoot`), never supplied. -/
structure Context (E Digest : Type) where
  plan : Plan
  hash : HashInput Digest → Digest
  eta : EtaInput Digest → E × E
  planDigest : Digest

/-- One invocation of an extracted run: the proposals that an opening arm
reads, the records of the step, and the application state after the step. -/
structure Invocation (σ Digest : Type) where
  proposal : Digest × Digest
  records : StepRecords
  next : σ

/-- The public statement of spec §13. -/
structure Statement (σ Digest : Type) where
  steps : ℕ
  initial : σ
  final : σ
  segments : ℕ
  finalTs : ℕ
  finalRoot : Digest

/-- The ports of an invocation: one per operation slot (spec §10, Ob7). -/
def Invocation.ports {σ Digest : Type} (inv : Invocation σ Digest) : List (Option PortAccess) :=
  inv.records.ops.map OpSlot.port

/-- The packed ops lane of a step. -/
def opsPacked (p : Plan) (z : StepRecords) : List ℕ := pack (opsLane p z.ops)

/-- The packed IS lane of a step. -/
def initialPacked (p : Plan) (z : StepRecords) : List ℕ := pack (scanLane p z.initialScan)

/-- The packed FS lane of a step. -/
def finalPacked (p : Plan) (z : StepRecords) : List ℕ := pack (scanLane p z.finalScan)

variable {E Digest σ : Type} [CommRing E]

namespace Context

/-- `D_init`: the IS root over the initial snapshot (spec §9.2). -/
def initRoot (ctx : Context E Digest) : Digest :=
  memoryRoot ctx.hash ctx.plan ctx.planDigest (initialMemory ctx.plan)

/-- `header_l` of spec §9.2. -/
def header (ctx : Context E Digest) (lane : Lane) : Digest :=
  ctx.hash (.header lane ctx.planDigest)

end Context

/-- Chain start (spec §11.1): `seg_idx = 0`, `ts = 0`, `D_mem = D_init`.
`open` sets every other field. -/
def startCarry (ctx : Context E Digest) : Carry E Digest :=
  ⟨0, 0, 0, (0, 0), ⟨1, 1, 1, 1⟩, (ctx.initRoot, ctx.initRoot),
    ⟨ctx.initRoot, ctx.initRoot, ctx.initRoot⟩, ctx.initRoot⟩

/-- Spec §11.2 `open`. -/
def openSegment (ctx : Context E Digest) (c : Carry E Digest) (proposal : Digest × Digest) :
    Option (Carry E Digest) :=
  if c.segIdx < ctx.plan.sMax then
    some { c with
      proposed := proposal
      eta := ctx.eta ⟨ctx.planDigest, c.ts, proposal.1, c.memRoot, proposal.2⟩
      products := ⟨1, 1, 1, 1⟩
      seen := ⟨ctx.header .ops, ctx.header .mem, ctx.header .mem⟩
      idx := 0 }
  else none

open Classical in
/-- Spec §11.2 `step`: the §8 rows on the invocation's own records, then the
three chain links, the products, the timestamp, and the index. -/
noncomputable def stepSegment (ctx : Context E Digest) (c : Carry E Digest) (z : StepRecords) :
    Option (Carry E Digest) :=
  if StepRows ctx.plan c.ts z then
    some { c with
      seen := ⟨ctx.hash (.chain .ops c.idx c.seen.ops (opsPacked ctx.plan z)),
        ctx.hash (.chain .mem c.idx c.seen.initial (initialPacked ctx.plan z)),
        ctx.hash (.chain .mem c.idx c.seen.final (finalPacked ctx.plan z))⟩
      products := stepProducts ctx.plan c.eta c.ts c.idx z c.products
      ts := c.ts + activeCount z
      idx := c.idx + 1 }
  else none

open Classical in
/-- Spec §11.2 `close`. -/
noncomputable def closeSegment (_ctx : Context E Digest) (c : Carry E Digest) :
    Option (Carry E Digest) :=
  if c.seen.ops = c.proposed.1 ∧ c.seen.initial = c.memRoot ∧ c.seen.final = c.proposed.2 ∧
      c.products.initial * c.products.write = c.products.read * c.products.final then
    some { c with memRoot := c.proposed.2, segIdx := c.segIdx + 1 }
  else none

/-- `close` exactly when `step` has set `idx = N`. -/
noncomputable def finishStep (ctx : Context E Digest) (c : Carry E Digest) :
    Option (Carry E Digest) :=
  if c.idx = ctx.plan.n then closeSegment ctx c else some c

/-- Spec §12: the input carry selects the arm. `none` is the base arm; an
open segment continues; a closed one reopens. -/
noncomputable def invoke (ctx : Context E Digest) (input : Option (Carry E Digest))
    (inv : Invocation σ Digest) : Option (Carry E Digest) :=
  let opened : Option (Carry E Digest) :=
    match input with
    | none => openSegment ctx (startCarry ctx) inv.proposal
    | some c => if c.idx < ctx.plan.n then some c else openSegment ctx c inv.proposal
  (opened.bind fun c => stepSegment ctx c inv.records).bind (finishStep ctx)

/-- The invocations after the base invocation. -/
noncomputable def continueRun (ctx : Context E Digest) (c : Carry E Digest)
    (run : List (Invocation σ Digest)) : Option (Carry E Digest) :=
  run.foldlM (fun c inv => invoke ctx (some c) inv) c

/-- The final carry of a run. An empty run has none (spec §13 rejects
`T = 0`). -/
noncomputable def finalCarry (ctx : Context E Digest) :
    List (Invocation σ Digest) → Option (Carry E Digest)
  | [] => none
  | inv :: rest => (invoke ctx none inv).bind fun c => continueRun ctx c rest

/-- Each invocation is one application step on its own ports. -/
def AppThread {p : Plan} (app : Application p σ) : σ → List (Invocation σ Digest) → σ → Prop
  | s, [], s' => s = s'
  | s, inv :: rest, s' => app.Step s inv.ports inv.next ∧ AppThread app inv.next rest s'

/-- Spec §12 and §13 on an extracted run: the terminal checks, the final
carry, and the application thread. -/
structure Accepts (ctx : Context E Digest) (app : Application ctx.plan σ)
    (stmt : Statement σ Digest) (run : List (Invocation σ Digest)) : Prop where
  steps : run.length = stmt.steps
  segmentsRange : 1 ≤ stmt.segments ∧ stmt.segments ≤ ctx.plan.sMax
  stepCount : stmt.steps = stmt.segments * ctx.plan.n
  terminal : ∃ c, finalCarry ctx run = some c ∧ c.idx = ctx.plan.n ∧
    c.segIdx = stmt.segments ∧ c.ts = stmt.finalTs ∧ c.memRoot = stmt.finalRoot
  application : AppThread app stmt.initial run stmt.final

/-! ### Segment views

A segment view is read off the run, not off the carries, so that the events
of the security proofs name deterministic objects. -/

/-- The global timestamp at the entry of invocation `i`. -/
def tsBefore (run : List (Invocation σ Digest)) (i : ℕ) : ℕ :=
  ((run.take i).map fun inv => activeCount inv.records).sum

/-- Invocations `k·N … k·N + N − 1`. -/
def segmentRun (p : Plan) (run : List (Invocation σ Digest)) (k : ℕ) :
    List (Invocation σ Digest) :=
  (run.drop (k * p.n)).take p.n

/-- The proposals that the opening invocation of segment `k` carries. -/
def proposalAt (ctx : Context E Digest) (run : List (Invocation σ Digest)) (k : ℕ) :
    Digest × Digest :=
  (((run.drop (k * ctx.plan.n)).head?).map Invocation.proposal).getD (ctx.initRoot, ctx.initRoot)

/-- What the security proofs read about one segment. -/
structure SegmentView (Digest : Type) where
  openTs : ℕ
  memIn : Digest
  proposal : Digest × Digest
  records : List StepRecords

/-- Segment `k` of a run. `memIn` is `D_init` for segment 0 and the FS
proposal of segment `k − 1` otherwise. -/
def segmentView (ctx : Context E Digest) (run : List (Invocation σ Digest)) (k : ℕ) :
    SegmentView Digest where
  openTs := tsBefore run (k * ctx.plan.n)
  memIn := match k with
    | 0 => ctx.initRoot
    | k + 1 => (proposalAt ctx run k).2
  proposal := proposalAt ctx run k
  records := (segmentRun ctx.plan run k).map Invocation.records

/-- The multisets of consecutive steps from timestamp `ts` and index `idx`. -/
def segmentMultisets (p : Plan) : ℕ → ℕ → List StepRecords → Multisets
  | _, _, [] => 0
  | ts, idx, z :: rest =>
    stepMultisets p ts idx z + segmentMultisets p (ts + activeCount z) (idx + 1) rest

/-- The rows of consecutive steps from timestamp `ts`. -/
def SegmentRows (p : Plan) : ℕ → List StepRecords → Prop
  | _, [] => True
  | ts, z :: rest => StepRows p ts z ∧ SegmentRows p (ts + activeCount z) rest

namespace SegmentView

/-- The §9.3 transcript input of the segment. -/
def etaInput (ctx : Context E Digest) (v : SegmentView Digest) : EtaInput Digest :=
  ⟨ctx.planDigest, v.openTs, v.proposal.1, v.memIn, v.proposal.2⟩

/-- The challenges of the segment. -/
def eta (ctx : Context E Digest) (v : SegmentView Digest) : E × E := ctx.eta (v.etaInput ctx)

/-- The four multisets of the segment. -/
def multisets (p : Plan) (v : SegmentView Digest) : Multisets :=
  segmentMultisets p v.openTs 0 v.records

/-- The packed ops lanes, one per step. -/
def opsLanes (p : Plan) (v : SegmentView Digest) : List (List ℕ) := v.records.map (opsPacked p)

/-- The packed IS lanes, one per step. -/
def initialLanes (p : Plan) (v : SegmentView Digest) : List (List ℕ) :=
  v.records.map (initialPacked p)

/-- The packed FS lanes, one per step. -/
def finalLanes (p : Plan) (v : SegmentView Digest) : List (List ℕ) :=
  v.records.map (finalPacked p)

/-- The close checks of spec §11.2 on a whole segment, with challenges `η`. -/
structure ClosesAt (ctx : Context E Digest) (η : E × E) (v : SegmentView Digest) : Prop where
  length : v.records.length = ctx.plan.n
  rows : SegmentRows ctx.plan v.openTs v.records
  opsRoot : chainRoot ctx.hash .ops ctx.planDigest (v.opsLanes ctx.plan) = v.proposal.1
  initialRoot : chainRoot ctx.hash .mem ctx.planDigest (v.initialLanes ctx.plan) = v.memIn
  finalRoot : chainRoot ctx.hash .mem ctx.planDigest (v.finalLanes ctx.plan) = v.proposal.2
  products : (v.multisets ctx.plan).ProductEq η

/-- The hash inputs of the segment's three chains. -/
def chainInputs (ctx : Context E Digest) (v : SegmentView Digest) : List (HashInput Digest) :=
  Nebula.chainInputs ctx.hash .ops ctx.planDigest (v.opsLanes ctx.plan) ++
    Nebula.chainInputs ctx.hash .mem ctx.planDigest (v.initialLanes ctx.plan) ++
    Nebula.chainInputs ctx.hash .mem ctx.planDigest (v.finalLanes ctx.plan)

end SegmentView

/-! ### Lifecycle theorem (Ob8) -/

/-- The start carry marked closed. The base arm is the reopen arm on it. -/
private def closedStart (ctx : Context E Digest) : Carry E Digest :=
  { startCarry ctx with idx := ctx.plan.n }

/-- The carry that `open` writes, without the `S_max` check. -/
private def openedCarry (ctx : Context E Digest) (c : Carry E Digest)
    (proposal : Digest × Digest) : Carry E Digest :=
  { c with
    proposed := proposal
    eta := ctx.eta ⟨ctx.planDigest, c.ts, proposal.1, c.memRoot, proposal.2⟩
    products := ⟨1, 1, 1, 1⟩
    seen := ⟨ctx.header .ops, ctx.header .mem, ctx.header .mem⟩
    idx := 0 }

/-- The carry that `step` writes, without the row check. -/
private def advance (ctx : Context E Digest) (c : Carry E Digest) (z : StepRecords) :
    Carry E Digest :=
  { c with
    seen := ⟨ctx.hash (.chain .ops c.idx c.seen.ops (opsPacked ctx.plan z)),
      ctx.hash (.chain .mem c.idx c.seen.initial (initialPacked ctx.plan z)),
      ctx.hash (.chain .mem c.idx c.seen.final (finalPacked ctx.plan z))⟩
    products := stepProducts ctx.plan c.eta c.ts c.idx z c.products
    ts := c.ts + activeCount z
    idx := c.idx + 1 }

/-- `advance` over consecutive steps. -/
private def runSteps (ctx : Context E Digest) (c : Carry E Digest) (zs : List StepRecords) :
    Carry E Digest :=
  zs.foldl (advance ctx) c

private theorem runSteps_cons (ctx : Context E Digest) (c : Carry E Digest) (z : StepRecords)
    (zs : List StepRecords) : runSteps ctx c (z :: zs) = runSteps ctx (advance ctx c z) zs := rfl

private theorem read_add (a b : Multisets) : (a + b).read = a.read + b.read := rfl
private theorem write_add (a b : Multisets) : (a + b).write = a.write + b.write := rfl
private theorem initial_add (a b : Multisets) : (a + b).initial = a.initial + b.initial := rfl
private theorem final_add (a b : Multisets) : (a + b).final = a.final + b.final := rfl

private theorem segmentRows_cons (p : Plan) (ts : ℕ) (z : StepRecords) (zs : List StepRecords) :
    SegmentRows p ts (z :: zs) ↔ StepRows p ts z ∧ SegmentRows p (ts + activeCount z) zs :=
  Iff.rfl

private theorem runSteps_fixed (ctx : Context E Digest) :
    ∀ (zs : List StepRecords) (c : Carry E Digest),
      (runSteps ctx c zs).segIdx = c.segIdx ∧ (runSteps ctx c zs).proposed = c.proposed ∧
        (runSteps ctx c zs).memRoot = c.memRoot ∧
        (runSteps ctx c zs).idx = c.idx + zs.length ∧
        (runSteps ctx c zs).ts = c.ts + (zs.map activeCount).sum
  | [], c => ⟨rfl, rfl, rfl, rfl, rfl⟩
  | z :: zs, c => by
    obtain ⟨hSeg, hProp, hMem, hIdx, hTs⟩ := runSteps_fixed ctx zs (advance ctx c z)
    rw [runSteps_cons]
    refine ⟨hSeg, hProp, hMem, ?_, ?_⟩
    · rw [hIdx, List.length_cons]
      show c.idx + 1 + zs.length = _
      omega
    · rw [hTs, List.map_cons, List.sum_cons]
      show c.ts + activeCount z + _ = _
      omega

private theorem runSteps_seen (ctx : Context E Digest) :
    ∀ (zs : List StepRecords) (c : Carry E Digest),
      ((runSteps ctx c zs).seen.ops, (runSteps ctx c zs).idx) =
          (zs.map (opsPacked ctx.plan)).foldl
            (fun acc P => (ctx.hash (.chain .ops acc.2 acc.1 P), acc.2 + 1)) (c.seen.ops, c.idx) ∧
        ((runSteps ctx c zs).seen.initial, (runSteps ctx c zs).idx) =
          (zs.map (initialPacked ctx.plan)).foldl
            (fun acc P => (ctx.hash (.chain .mem acc.2 acc.1 P), acc.2 + 1))
            (c.seen.initial, c.idx) ∧
        ((runSteps ctx c zs).seen.final, (runSteps ctx c zs).idx) =
          (zs.map (finalPacked ctx.plan)).foldl
            (fun acc P => (ctx.hash (.chain .mem acc.2 acc.1 P), acc.2 + 1)) (c.seen.final, c.idx)
  | [], _ => ⟨rfl, rfl, rfl⟩
  | z :: zs, c => by
    rw [runSteps_cons]
    simp only [List.map_cons, List.foldl_cons]
    exact runSteps_seen ctx zs (advance ctx c z)

private theorem runSteps_products (ctx : Context E Digest) :
    ∀ (zs : List StepRecords) (c : Carry E Digest),
      (runSteps ctx c zs).products =
        ⟨c.products.read * product c.eta (segmentMultisets ctx.plan c.ts c.idx zs).read,
         c.products.write * product c.eta (segmentMultisets ctx.plan c.ts c.idx zs).write,
         c.products.initial * product c.eta (segmentMultisets ctx.plan c.ts c.idx zs).initial,
         c.products.final * product c.eta (segmentMultisets ctx.plan c.ts c.idx zs).final⟩
  | [], c => by
    change c.products = ⟨c.products.read * product c.eta 0, c.products.write * product c.eta 0,
      c.products.initial * product c.eta 0, c.products.final * product c.eta 0⟩
    simp only [product, Multiset.map_zero, Multiset.prod_zero, mul_one]
  | z :: zs, c => by
    rw [runSteps_cons, runSteps_products ctx zs (advance ctx c z)]
    simp only [advance, stepProducts_eq, segmentMultisets, read_add, write_add, initial_add,
      final_add, product_add, mul_assoc]

/-- The fields of a segment run from `open` on `c`, before `close`. -/
private theorem runSteps_opened {ctx : Context E Digest} {c r : Carry E Digest}
    {proposal : Digest × Digest} {zs : List StepRecords}
    (run : r = runSteps ctx (openedCarry ctx c proposal) zs) :
    r.segIdx = c.segIdx ∧ r.idx = zs.length ∧ r.ts = c.ts + (zs.map activeCount).sum ∧
      r.proposed = proposal ∧ r.memRoot = c.memRoot ∧
      r.seen.ops = chainRoot ctx.hash .ops ctx.planDigest (zs.map (opsPacked ctx.plan)) ∧
      r.seen.initial = chainRoot ctx.hash .mem ctx.planDigest (zs.map (initialPacked ctx.plan)) ∧
      r.seen.final = chainRoot ctx.hash .mem ctx.planDigest (zs.map (finalPacked ctx.plan)) ∧
      ((r.products.initial * r.products.write = r.products.read * r.products.final) ↔
        (segmentMultisets ctx.plan c.ts 0 zs).ProductEq
          (ctx.eta ⟨ctx.planDigest, c.ts, proposal.1, c.memRoot, proposal.2⟩)) := by
  subst run
  obtain ⟨hSeg, hProp, hMem, hIdx, hTs⟩ := runSteps_fixed ctx zs (openedCarry ctx c proposal)
  obtain ⟨hOps, hInit, hFin⟩ := runSteps_seen ctx zs (openedCarry ctx c proposal)
  refine ⟨hSeg, hIdx.trans (Nat.zero_add _), hTs, hProp, hMem, congrArg Prod.fst hOps,
    congrArg Prod.fst hInit, congrArg Prod.fst hFin, ?_⟩
  rw [runSteps_products]
  simp only [openedCarry, one_mul]
  rfl

private theorem stepSegment_of_rows {ctx : Context E Digest} {c : Carry E Digest}
    {z : StepRecords} (rows : StepRows ctx.plan c.ts z) :
    stepSegment ctx c z = some (advance ctx c z) := by
  unfold stepSegment
  rw [if_pos rows]
  rfl

private theorem stepSegment_of_not_rows {ctx : Context E Digest} {c : Carry E Digest}
    {z : StepRecords} (rows : ¬ StepRows ctx.plan c.ts z) : stepSegment ctx c z = none := by
  unfold stepSegment
  rw [if_neg rows]

private theorem closeSegment_eq_some {ctx : Context E Digest} {r c' : Carry E Digest} :
    closeSegment ctx r = some c' ↔
      (r.seen.ops = r.proposed.1 ∧ r.seen.initial = r.memRoot ∧ r.seen.final = r.proposed.2 ∧
        r.products.initial * r.products.write = r.products.read * r.products.final) ∧
      { r with memRoot := r.proposed.2, segIdx := r.segIdx + 1 } = c' := by
  unfold closeSegment
  split_ifs with checks
  · exact ⟨fun h => ⟨checks, Option.some.inj h⟩, fun h => congrArg some h.2⟩
  · exact ⟨fun h => (by cases h), fun h => absurd h.1 checks⟩

private theorem finalCarry_cons (ctx : Context E Digest) (inv : Invocation σ Digest)
    (rest : List (Invocation σ Digest)) :
    finalCarry ctx (inv :: rest) = continueRun ctx (closedStart ctx) (inv :: rest) := by
  have reopen : ¬ (closedStart ctx).idx < ctx.plan.n := lt_irrefl ctx.plan.n
  simp only [finalCarry, continueRun, List.foldlM_cons, Option.bind_eq_bind, invoke,
    if_neg reopen]
  rfl

private theorem continueRun_cons_open {ctx : Context E Digest} {d : Carry E Digest}
    (isOpen : d.idx < ctx.plan.n) (inv : Invocation σ Digest)
    (rest : List (Invocation σ Digest)) :
    continueRun ctx d (inv :: rest) =
      ((stepSegment ctx d inv.records).bind (finishStep ctx)).bind
        fun d' => continueRun ctx d' rest := by
  simp only [continueRun, List.foldlM_cons, Option.bind_eq_bind, invoke, if_pos isOpen,
    Option.bind_some]

private theorem continueRun_reopen {ctx : Context E Digest} {c c' : Carry E Digest}
    {inv : Invocation σ Digest} {rest : List (Invocation σ Digest)}
    (closed : c.idx = ctx.plan.n) :
    continueRun ctx c (inv :: rest) = some c' ↔ c.segIdx < ctx.plan.sMax ∧
      (((stepSegment ctx (openedCarry ctx c inv.proposal) inv.records).bind
        (finishStep ctx)).bind fun d => continueRun ctx d rest) = some c' := by
  have reopen : ¬ c.idx < ctx.plan.n := by omega
  have opens : openSegment ctx c inv.proposal =
      if c.segIdx < ctx.plan.sMax then some (openedCarry ctx c inv.proposal) else none := rfl
  by_cases fits : c.segIdx < ctx.plan.sMax
  · simp only [continueRun, List.foldlM_cons, Option.bind_eq_bind, invoke, if_neg reopen, opens,
      fits, if_true, Option.bind_some, true_and]
  · simp only [continueRun, List.foldlM_cons, Option.bind_eq_bind, invoke, if_neg reopen, opens,
      fits, if_false, Option.bind_none, false_and, reduceCtorEq]

/-- An opened segment with `rest.length + 1` steps left runs its steps and
closes at the last one. -/
private theorem steps_eq_some (ctx : Context E Digest) (c' : Carry E Digest)
    (rest : List (Invocation σ Digest)) :
    ∀ (c : Carry E Digest) (inv : Invocation σ Digest), c.idx + rest.length + 1 = ctx.plan.n →
      ((((stepSegment ctx c inv.records).bind (finishStep ctx)).bind
          fun d => continueRun ctx d rest) = some c' ↔
        SegmentRows ctx.plan c.ts ((inv :: rest).map Invocation.records) ∧
          closeSegment ctx (runSteps ctx c ((inv :: rest).map Invocation.records)) = some c') := by
  induction rest with
  | nil =>
    intro c inv len
    by_cases rows : StepRows ctx.plan c.ts inv.records
    · have closes : (advance ctx c inv.records).idx = ctx.plan.n := by
        simp only [List.length_nil] at len
        exact len
      have fin : finishStep ctx (advance ctx c inv.records) =
          closeSegment ctx (advance ctx c inv.records) := by
        unfold finishStep
        rw [if_pos closes]
      have done : ∀ o : Option (Carry E Digest),
          (o.bind fun d => continueRun ctx d ([] : List (Invocation σ Digest))) = o := by
        intro o
        cases o <;> rfl
      rw [stepSegment_of_rows rows, Option.bind_some, fin, done]
      exact ⟨fun h => ⟨(segmentRows_cons _ _ _ _).mpr ⟨rows, trivial⟩, h⟩, fun h => h.2⟩
    · rw [stepSegment_of_not_rows rows]
      exact ⟨fun h => (by cases h), fun h => absurd ((segmentRows_cons _ _ _ _).mp h.1).1 rows⟩
  | cons inv' rest ih =>
    intro c inv len
    simp only [List.length_cons] at len
    by_cases rows : StepRows ctx.plan c.ts inv.records
    · have isOpen : (advance ctx c inv.records).idx < ctx.plan.n := by
        show c.idx + 1 < _
        omega
      have fin : finishStep ctx (advance ctx c inv.records) = some (advance ctx c inv.records) := by
        unfold finishStep
        rw [if_neg isOpen.ne]
      have next := ih (advance ctx c inv.records) inv' (by show c.idx + 1 + _ + 1 = _; omega)
      rw [← continueRun_cons_open isOpen] at next
      rw [stepSegment_of_rows rows, Option.bind_some, fin]
      simp only [Option.bind_some]
      rw [next]
      exact ⟨fun h => ⟨(segmentRows_cons _ _ _ _).mpr ⟨rows, h.1⟩, h.2⟩,
        fun h => ⟨((segmentRows_cons _ _ _ _).mp h.1).2, h.2⟩⟩
    · rw [stepSegment_of_not_rows rows]
      exact ⟨fun h => (by cases h), fun h => absurd ((segmentRows_cons _ _ _ _).mp h.1).1 rows⟩

/-- One segment from a closed carry: it closes exactly when `open` fits under
`S_max` and its view passes the close checks with its own challenges. -/
private theorem continueRun_segment {ctx : Context E Digest} {c : Carry E Digest}
    {inv : Invocation σ Digest} {rest : List (Invocation σ Digest)} {v : SegmentView Digest}
    (closed : c.idx = ctx.plan.n) (len : rest.length + 1 = ctx.plan.n)
    (openTs : v.openTs = c.ts) (memIn : v.memIn = c.memRoot)
    (proposal : v.proposal = inv.proposal)
    (records : v.records = (inv :: rest).map Invocation.records) :
    ((∃ c', continueRun ctx c (inv :: rest) = some c') ↔
        c.segIdx < ctx.plan.sMax ∧ v.ClosesAt ctx (v.eta ctx)) ∧
      ∀ c', continueRun ctx c (inv :: rest) = some c' →
        c'.idx = ctx.plan.n ∧ c'.segIdx = c.segIdx + 1 ∧
          c'.ts = c.ts + (((inv :: rest).map Invocation.records).map activeCount).sum ∧
          c'.memRoot = inv.proposal.2 := by
  obtain ⟨vTs, vMem, vProp, vRec⟩ := v
  dsimp only at openTs memIn proposal records
  subst openTs memIn proposal records
  have len' : ((inv :: rest).map Invocation.records).length = ctx.plan.n := by
    simpa using len
  obtain ⟨rSeg, rIdx, rTs, rProp, rMem, rOps, rInit, rFin, rProd⟩ :=
    runSteps_opened (ctx := ctx) (c := c) (proposal := inv.proposal)
      (zs := (inv :: rest).map Invocation.records) rfl
  have key : ∀ c', continueRun ctx c (inv :: rest) = some c' ↔ c.segIdx < ctx.plan.sMax ∧
      SegmentRows ctx.plan c.ts ((inv :: rest).map Invocation.records) ∧
        closeSegment ctx (runSteps ctx (openedCarry ctx c inv.proposal)
          ((inv :: rest).map Invocation.records)) = some c' := by
    intro c'
    rw [continueRun_reopen closed,
      steps_eq_some ctx c' rest _ inv (by show 0 + rest.length + 1 = _; omega)]
    rfl
  refine ⟨⟨fun ⟨c', h⟩ => ?_, fun ⟨fits, closes⟩ => ?_⟩, fun c' h => ?_⟩
  · obtain ⟨fits, rows, close⟩ := (key c').mp h
    obtain ⟨⟨ops, init, fin, prods⟩, -⟩ := closeSegment_eq_some.mp close
    rw [rOps, rProp] at ops
    rw [rInit, rMem] at init
    rw [rFin, rProp] at fin
    exact ⟨fits, len', rows, ops, init, fin, rProd.mp prods⟩
  · refine ⟨_, (key _).mpr
      ⟨fits, closes.rows, closeSegment_eq_some.mpr ⟨⟨?_, ?_, ?_, ?_⟩, rfl⟩⟩⟩
    · rw [rOps, rProp]
      exact closes.opsRoot
    · rw [rInit, rMem]
      exact closes.initialRoot
    · rw [rFin, rProp]
      exact closes.finalRoot
    · exact rProd.mpr closes.products
  · obtain ⟨-, -, close⟩ := (key c').mp h
    obtain ⟨-, rfl⟩ := closeSegment_eq_some.mp close
    exact ⟨rIdx.trans len', congrArg (· + 1) rSeg, rTs, congrArg Prod.snd rProp⟩

/-- The first `k` segments of a run of `S · N` invocations. -/
private theorem prefix_run {ctx : Context E Digest} {run : List (Invocation σ Digest)} {S : ℕ}
    (valid : ctx.plan.Valid) (count : run.length = S * ctx.plan.n) (k : ℕ) (hk : k ≤ S) :
    ((∃ c, continueRun ctx (closedStart ctx) (run.take (k * ctx.plan.n)) = some c) ↔
        k ≤ ctx.plan.sMax ∧
          ∀ j < k, (segmentView ctx run j).ClosesAt ctx ((segmentView ctx run j).eta ctx)) ∧
      ∀ c, continueRun ctx (closedStart ctx) (run.take (k * ctx.plan.n)) = some c →
        c.idx = ctx.plan.n ∧ c.segIdx = k ∧ c.ts = tsBefore run (k * ctx.plan.n) ∧
          c.memRoot = (segmentView ctx run k).memIn := by
  induction k with
  | zero =>
    have start : continueRun ctx (closedStart ctx) (run.take (0 * ctx.plan.n)) =
        some (closedStart ctx) := by
      rw [Nat.zero_mul, List.take_zero]
      rfl
    rw [start]
    refine ⟨⟨fun _ => ⟨Nat.zero_le _, fun j hj => absurd hj (Nat.not_lt_zero j)⟩,
      fun _ => ⟨_, rfl⟩⟩, fun c hc => ?_⟩
    obtain rfl := Option.some.inj hc
    refine ⟨rfl, rfl, ?_, rfl⟩
    simp [tsBefore, closedStart, startCarry]
  | succ k ih =>
    obtain ⟨ihIff, ihFields⟩ := ih (by omega)
    have segLen : (segmentRun ctx.plan run k).length = ctx.plan.n := by
      have within : k * ctx.plan.n + ctx.plan.n ≤ S * ctx.plan.n := by
        rw [← Nat.add_one_mul]
        exact Nat.mul_le_mul_right _ hk
      simp only [segmentRun, List.length_take, List.length_drop, count]
      omega
    obtain ⟨inv, rest, hseg⟩ : ∃ inv rest, segmentRun ctx.plan run k = inv :: rest := by
      cases h : segmentRun ctx.plan run k with
      | nil =>
        rw [h, List.length_nil] at segLen
        exact absurd segLen.symm (Nat.ne_of_gt valid.positive.2.2.1)
      | cons inv rest => exact ⟨inv, rest, rfl⟩
    have split : run.take ((k + 1) * ctx.plan.n) = run.take (k * ctx.plan.n) ++ inv :: rest := by
      rw [← hseg, Nat.add_one_mul, List.take_add]
      rfl
    have head : proposalAt ctx run k = inv.proposal := by
      have whole := List.take_append_drop ctx.plan.n (run.drop (k * ctx.plan.n))
      unfold segmentRun at hseg
      rw [hseg] at whole
      unfold proposalAt
      rw [← whole]
      rfl
    have tsEq : tsBefore run ((k + 1) * ctx.plan.n) = tsBefore run (k * ctx.plan.n) +
        (((inv :: rest).map Invocation.records).map activeCount).sum := by
      rw [tsBefore, split, List.map_append, List.sum_append, List.map_map]
      rfl
    have runEq : continueRun ctx (closedStart ctx) (run.take ((k + 1) * ctx.plan.n)) =
        (continueRun ctx (closedStart ctx) (run.take (k * ctx.plan.n))).bind
          fun c => continueRun ctx c (inv :: rest) := by
      rw [split]
      simp only [continueRun, List.foldlM_append, Option.bind_eq_bind]
    rw [runEq]
    cases hprev : continueRun ctx (closedStart ctx) (run.take (k * ctx.plan.n)) with
    | none =>
      refine ⟨⟨fun ⟨c, hc⟩ => (by cases hc), fun ⟨fits, closes⟩ => ?_⟩,
        fun c hc => by cases hc⟩
      obtain ⟨c, hc⟩ := ihIff.mpr ⟨by omega, fun j hj => closes j (by omega)⟩
      rw [hprev] at hc
      cases hc
    | some c₀ =>
      obtain ⟨cIdx, cSeg, cTs, cMem⟩ := ihFields c₀ hprev
      have good := ihIff.mp ⟨c₀, hprev⟩
      have len : rest.length + 1 = ctx.plan.n := by
        rw [hseg, List.length_cons] at segLen
        exact segLen
      obtain ⟨segIff, segFields⟩ := continueRun_segment (v := segmentView ctx run k) cIdx len
        cTs.symm cMem.symm head (by show (segmentRun ctx.plan run k).map _ = _; rw [hseg])
      simp only [Option.bind_some]
      refine ⟨?_, fun c' hc' => ?_⟩
      · rw [segIff, cSeg]
        constructor
        · rintro ⟨fits, closes⟩
          refine ⟨fits, fun j hj => ?_⟩
          rcases Nat.lt_succ_iff_lt_or_eq.mp hj with hj | rfl
          · exact good.2 j hj
          · exact closes
        · rintro ⟨fits, closes⟩
          exact ⟨fits, closes k (Nat.lt_succ_self k)⟩
      · obtain ⟨i1, i2, i3, i4⟩ := segFields c' hc'
        refine ⟨i1, by rw [i2, cSeg], by rw [i3, tsEq, cTs], ?_⟩
        rw [i4, ← head]
        rfl

/-- A run of `S · N ≥ 1` invocations is its `S`-segment prefix, run from the
closed start carry. -/
private theorem finalCarry_eq_prefix {ctx : Context E Digest} {run : List (Invocation σ Digest)}
    {S : ℕ} (valid : ctx.plan.Valid) (count : run.length = S * ctx.plan.n) (pos : 1 ≤ S) :
    finalCarry ctx run = continueRun ctx (closedStart ctx) (run.take (S * ctx.plan.n)) := by
  obtain ⟨inv, rest, rfl⟩ := List.exists_cons_of_length_pos
    (count ▸ Nat.mul_pos pos valid.positive.2.2.1 : 0 < run.length)
  rw [← count, List.take_length, finalCarry_cons]

/-- The trace of a run of `S · N` invocations succeeds exactly when there are
at most `S_max` segments and every segment view passes the close checks with
its own challenges. -/
theorem finalCarry_isSome_iff {ctx : Context E Digest} {run : List (Invocation σ Digest)} {S : ℕ}
    (valid : ctx.plan.Valid) (count : run.length = S * ctx.plan.n) (pos : 1 ≤ S) :
    (∃ c, finalCarry ctx run = some c) ↔
      S ≤ ctx.plan.sMax ∧
        ∀ k < S, (segmentView ctx run k).ClosesAt ctx ((segmentView ctx run k).eta ctx) := by
  rw [finalCarry_eq_prefix valid count pos]
  exact (prefix_run valid count S le_rfl).1

/-- The fields of the final carry of a successful trace. -/
theorem finalCarry_fields {ctx : Context E Digest} {run : List (Invocation σ Digest)} {S : ℕ}
    {c : Carry E Digest} (valid : ctx.plan.Valid) (count : run.length = S * ctx.plan.n)
    (pos : 1 ≤ S) (final : finalCarry ctx run = some c) :
    c.idx = ctx.plan.n ∧ c.segIdx = S ∧ c.ts = tsBefore run run.length ∧
      c.memRoot = (proposalAt ctx run (S - 1)).2 := by
  rw [finalCarry_eq_prefix valid count pos] at final
  obtain ⟨hIdx, hSeg, hTs, hMem⟩ := (prefix_run valid count S le_rfl).2 c final
  refine ⟨hIdx, hSeg, by rw [hTs, count], ?_⟩
  obtain ⟨S', rfl⟩ : ∃ S', S = S' + 1 := ⟨S - 1, by omega⟩
  exact hMem

end NightstreamFPrime.Spec.Nebula
