import CompressionLean.Circuit.Appends
import CompressionLean.Circuit.Sim

/-!
Owns `terminal_exec`: what an accepting executor run of the final program
holds in its terminal wrapper (`finalProgram`, `apply`).

- The statement words are the first nine values. The iteration has an
  inverse, so it is not zero.
- One handle per system column (`columns`). A one column holds one, a public
  context column holds its context word, and a statement column of the prior
  state preimage is the statement handle (`ColumnValue`).
- Every row of the package system holds on the column values.
- The claim that layer 1 reads is the claim of the output columns
  (`claimVars`).

The checks come from the run itself: every instruction appends, so the run
of the final program starts with the wrapper's run (`verifyWith_appends`),
and its last check holds only if each earlier one does (`holds_of_steps`).
-/

namespace CompressionLean.Circuit

open CompressionLean.Circuit.Package CompressionLean.Circuit.Terminal
open CompressionLean.Replay (Gl)
open NightstreamFPrime.Spec (goldilocksModulus)
open NightstreamFPrime.Layout

/-! ## Runs -/

section Runs

variable {F : Type} [CommRing F] (c : Ctx F)

/-- A step keeps the checks so far. -/
theorem Ctx.holds_of_step {E : Exec.State F} {i : Instr} (h : (c.step E i).holds) : E.holds := by
  cases i <;> first | exact h.1 | exact h

/-- Steps keep the checks so far. -/
theorem Ctx.holds_of_steps {L : List Instr} {E : Exec.State F} (h : (c.steps L E).holds) : E.holds := by
  induction L generalizing E with
  | nil => exact h
  | cons i L ih => exact c.holds_of_step (ih h)

/-- The run of `A ++ B` runs `B` after `A`. -/
theorem Exec.run_append (A B : Array Instr) :
    Exec.run c.perm c.stmt c.tape (A ++ B) = c.steps B.toList (Exec.run c.perm c.stmt c.tape A) := by
  rw [Exec.run, Array.toList_append, List.foldl_append]
  rfl

end Runs

/-! ## The wrapper's values -/

section Values

variable {F : Type} [CommRing F] [CharP F goldilocksModulus]

/-- The value of the handle of a system column, by its source: one, a public
context word, or a statement word of the prior state preimage. -/
def ColumnValue (val : Var → F) (stmt : List F) (context : Array Gl) : Source → Var → Prop
  | .one, h => val h = 1
  | .priv i, h => contextStart + 4 ≤ i → i < PilotProduction.stateHashWords → val h = stmt.getD (i - contextStart - 4) 0
  | .pub i, h => publicContextStart ≤ i → val h = word (context.getD (i - publicContextStart) 0)
  | .computed _, _ => True

/-- What `column` writes for a source: its handle is a value of the run,
and a statement column is the statement handle. -/
def ColumnHandle (c : Ctx F) (statement : Array Var) (context : Array Gl) (s : Source) (h : Var) (m : ℕ) :
    Prop :=
  h < m ∧ match s with
    | .one => c.val h = 1
    | .priv i => contextStart + 4 ≤ i → i < PilotProduction.stateHashWords →
        h = statement.getD (i - contextStart - 4) 0
    | .pub i => publicContextStart ≤ i → c.val h = word (context.getD (i - publicContextStart) 0)
    | .computed _ => True

theorem stateDomainChunk_length : NightstreamFPrime.Lifecycle.stateDomainChunk.length = 12 := by
  simp [NightstreamFPrime.Lifecycle.stateDomainChunk, NightstreamFPrime.Spec.Poseidon2.rate]

theorem contextStart_eq : contextStart = 27806 := by
  simp [contextStart, statementWords, PilotProduction.stateHashWords_eq]

theorem Sim.column {c : Ctx F} {n : ℕ} {P : Prop} (statement : Array Var) (context : Array Gl) (computed : Var)
    (s : Source) (hs : P → ∀ j, statement.getD j 0 < n) (hc : P → ∀ k, s = .computed k → computed + k < n) :
    Sim c n P (Terminal.column statement context computed s) (ColumnHandle c statement context s) True := by
  have chunk := stateDomainChunk_length
  have start := contextStart_eq
  have words := PilotProduction.stateHashWords_eq
  cases s with
  | one => exact (Sim.const 1).post (fun _ _ _ h => ⟨h.1, h.2.trans word_one⟩) fun _ => Iff.rfl
  | priv i =>
      dsimp only [Terminal.column]
      split_ifs with below context statementRange
      · exact (Sim.const _).post (fun _ _ _ h => ⟨h.1, fun lo _ => by omega⟩) fun _ => Iff.rfl
      · exact (Sim.const _).post (fun _ _ _ h => ⟨h.1, fun lo _ => by omega⟩) fun _ => Iff.rfl
      · exact Sim.pure fun hP => ⟨hs hP _, fun _ _ => rfl⟩
      · exact Sim.priv.post (fun _ _ _ h => ⟨h.1, fun lo hi => absurd ⟨lo, hi⟩ statementRange⟩) fun _ => Iff.rfl
  | pub i =>
      dsimp only [Terminal.column]
      split_ifs with high
      · exact (Sim.const _).post (fun _ _ _ h => ⟨h.1, fun _ => h.2⟩) fun _ => Iff.rfl
      · exact Sim.priv.post (fun _ _ _ h => ⟨h.1, fun lo => absurd lo high⟩) fun _ => Iff.rfl
  | computed k => exact Sim.pure fun hP => ⟨hc hP k rfl, trivial⟩

omit [CharP F goldilocksModulus] in
/-- A loop of `mapM` over handles: each element's handle meets `R`, which
later values keep. -/
theorem Sim.mapM_list {c : Ctx F} {P : Prop} {β : Type} {f : β → Build Var} (R : β → Var → ℕ → Prop)
    (mono : ∀ x h m m', m ≤ m' → R x h m → R x h m') :
    ∀ (xs : List β) (n : ℕ), (∀ x ∈ xs, ∀ m, n ≤ m → Sim c m P (f x) (R x) True) →
      Sim c n P (xs.mapM f) (fun hs m => List.Forall₂ (fun x h => R x h m) xs hs) True
  | [], _, _ => Sim.pure fun _ => .nil
  | x :: xs, n, hf => by
      rw [List.mapM_cons]
      refine Sim.bind₀ (hf x List.mem_cons_self n le_rfl) fun a m le => ?_
      refine Sim.bind₀ ((Sim.mapM_list R mono xs m fun y hy m' le' =>
        hf y (List.mem_cons_of_mem x hy) m' (le.trans le')).pre fun h => h.1) fun hs m' le' => ?_
      exact Sim.pure fun h => .cons (mono _ _ _ _ le' h.1.2) h.2

/-- The columns of `apply`: the hint of the computed values, then one
handle per source. -/
theorem Sim.columns {c : Ctx F} {n : ℕ} {P : Prop} (package : Loaded) (statement : Array Var)
    (context : Array Gl) (hs : ∀ j, statement.getD j 0 < n)
    (hc : ∀ s ∈ package.sources, ∀ k, s = .computed k → k < package.computed.size) :
    Sim c n P (systemColumns package statement context)
      (fun columns m => List.Forall₂ (fun s h => ColumnHandle c statement context s h m)
        package.sources.toList columns.toList) True := by
  unfold systemColumns
  refine Sim.bind₀ (Sim.emit (Q := fun h m => h = n ∧ m = n + package.computed.size) _
    fun _ E _ _ => ⟨⟨rfl, rfl⟩, Iff.rfl⟩) fun computed m le => ?_
  rw [Array.mapM_eq_mapM_toList]
  refine Sim.map (Sim.mapM_list (ColumnHandle c statement context)
    (fun _ _ _ _ le h => ⟨lt_of_lt_of_le h.1 le, h.2⟩) package.sources.toList m fun s member m' le' =>
      Sim.column statement context computed s (fun _ j => lt_of_lt_of_le (hs j) (le.trans le'))
        fun hP k e => ?_) fun _ _ h => by simpa using h
  obtain ⟨rfl, rfl⟩ := hP.2
  exact Nat.lt_of_lt_of_le (Nat.add_lt_add_left (hc s (Array.mem_toList_iff.mp member) k e) _) le'

end Values

/-! ## The wrapper in Goldilocks -/

section Wrapper

open CompressionLean.Bridge (Fp)

/-- A field value as a canonical word. -/
def ofF (y : Fp) : Gl := Gl.ofNat y.val

theorem word_ofF (y : Fp) : (word (ofF y) : Fp) = y := by
  rw [ofF, word_ofNat]
  exact ZMod.natCast_zmod_val y

/-- The iteration check (`terminal/mod.rs:60-68`) after the statement words
`statement`: the iteration times a hint is one. -/
theorem Sim.iteration {c : Ctx Fp} {n : ℕ} {P : Prop} (statement : Array Var) {x : Fp}
    (ha : P → statement.getD 0 0 < n ∧ c.val (statement.getD 0 0) = x) :
    Sim c n P (do
        let inverse ← Build.emit (.hint 1)
        let product ← Build.mul (statement.getD 0 0) inverse
        let one ← Build.const 1
        Build.assertEqual product one
        Pure.pure statement) (fun r _ => r = statement) (x * c.tape n = 1) := by
  refine Sim.bind₀ (Sim.emit (Q := fun h m => h = n ∧ n < m ∧ c.val n = c.tape n) _ fun _ E hl hF =>
    ⟨⟨rfl, by simp [Instr.outputs], ?_⟩, Iff.rfl⟩) fun inverse m le => ?_
  · rw [← hl, ← Nat.add_zero E.values.length, Ctx.val_next hF (by simp)]
    simp
  refine Sim.bind₀ (Sim.mul (ga := ofF x) (gb := ofF (c.tape n)) (fun h => ⟨lt_of_lt_of_le (ha h.1).1 le, by
      rw [(ha h.1).2, word_ofF]⟩) fun h => by
      obtain ⟨rfl, lt, value⟩ := h.2
      exact ⟨lt, by rw [value, word_ofF]⟩) fun product m' le' => ?_
  refine Sim.bind₀ (Sim.const 1) fun one m'' le'' => ?_
  refine ((Sim.assertEqual (fun h => h.1.2.mono le'') fun h => h.2).bind fun _ _ _ =>
    Sim.pure (a := statement) fun _ => rfl).post (fun _ _ _ h => h) fun _ => ?_
  rw [and_true]
  constructor
  · intro h
    have := congrArg (word (F := Fp)) h
    rwa [word_mul, word_ofF, word_ofF, word_one] at this
  · intro h
    exact word_injective (F := Fp) (by rw [word_mul, word_ofF, word_ofF, word_one, h])

/-- A right element of `Forall₂` has a left partner. -/
theorem forall₂_exists_left {α β : Type} {R : α → β → Prop} :
    ∀ {xs : List α} {ys : List β}, List.Forall₂ R xs ys → ∀ {y}, y ∈ ys → ∃ x ∈ xs, R x y
  | _, _, .nil, _, member => absurd member List.not_mem_nil
  | x :: _, _ :: _, .cons hxy rest, y, member => by
      rcases List.mem_cons.mp member with rfl | member
      · exact ⟨x, List.mem_cons_self, hxy⟩
      · obtain ⟨x', mem, h⟩ := forall₂_exists_left rest member
        exact ⟨x', List.mem_cons_of_mem _ mem, h⟩

/-- One bind of a builder run. -/
theorem run_bind {α β : Type} (b : Build α) (f : α → Build β) (p : Builder) :
    (b >>= f).run p = (f (b.run p).1).run (b.run p).2 := rfl

/-- The nine statement words. -/
theorem statementWords_run :
    ((Array.range statementWords).mapM fun _ => Build.pub).run {} =
      (Array.range 9, { instrs := Array.replicate 9 .pub, values := 9 }) := by
  rw [Array.mapM_eq_mapM_toList]
  rfl

/-- The run of the nine statement words. -/
theorem statementWords_exec (perm : List Fp → List Fp) (stmt : List Fp) (tape : ℕ → Fp) :
    Exec.run perm stmt tape (Array.replicate 9 .pub) = ⟨(List.range 9).map (stmt.getD · 0), 9, True⟩ := rfl

/-- The layer-1 part appends. -/
theorem layer1Views.appends (key : Layer1Input) (claim : ClaimVars) (state : Fin 16 → Var) :
    Appends (layer1Views key claim state) := by
  appends

/-- The layer-1 part returns the claim that it reads. -/
theorem layer1Views_claim (key : Layer1Input) (claim : ClaimVars) (state : Fin 16 → Var) (p : Builder) :
    ((layer1Views key claim state).run p).1.claim = claimView claim := by
  unfold layer1Views
  rw [run_bind]
  generalize (verifyWith key state (claimView claim)).run p = q
  obtain ⟨⟨_, _⟩, _⟩ := q
  rfl

/-- **The terminal wrapper of an accepting run.** If the executor accepts the
final program `L` on the statement words `stmt`, then the iteration is not
zero, and some handles `columns` of the system columns hold their sources'
values and every row of the package system, and give the claim that layer 1
reads. -/
theorem terminal_exec (package : Loaded) (context : Array Gl) (key : Layer1Input) (perm : List Fp → List Fp)
    (stmt : List Fp) (tape : ℕ → Fp)
    (computed : ∀ s ∈ package.sources, ∀ k, s = .computed k → k < package.computed.size)
    (L : Array Instr) (program : (finalViews package context key).program = L)
    (accepts : Exec perm L stmt tape) :
    ∃ columns : Array Var,
      ((finalViews package context key).run {}).1.claim = claimView (claimVars package.outputs columns) ∧
      List.Forall₂ (ColumnValue (Exec.run perm stmt tape L).value stmt context) package.sources.toList
        columns.toList ∧
      (∀ r ∈ package.system.toList,
        R1csRow.Holds (fun j => (columns.toList.map (Exec.run perm stmt tape L).value).getD j 0) r) ∧
      stmt.getD 0 0 ≠ 0 := by
  let c : Ctx Fp := ⟨perm, stmt, tape, (Exec.run perm stmt tape L).value⟩
  let p9 : Builder := { instrs := Array.replicate 9 .pub, values := 9 }
  let E9 : Exec.State Fp := ⟨(List.range 9).map (stmt.getD · 0), 9, True⟩
  -- The builder: the statement words, the iteration, the columns, the rows, then layer 1.
  let iteration : Build (Array Var) := do
    let inverse ← Build.emit (.hint 1)
    let product ← Build.mul ((Array.range 9).getD 0 0) inverse
    let one ← Build.const 1
    Build.assertEqual product one
    Pure.pure (Array.range 9)
  have readRun : readStatement.run {} = iteration.run p9 := by
    unfold readStatement
    rw [run_bind, statementWords_run]
  let p13 := (iteration.run p9).2
  have statementRun : (iteration.run p9).1 = Array.range 9 := rfl
  have values13 : p13.values = 13 := rfl
  let columns := ((systemColumns package (Array.range 9) context).run p13).1
  let pC := ((systemColumns package (Array.range 9) context).run p13).2
  let pR : Builder := { instrs := pC.instrs.push (.r1cs package.system.toList columns.toList), values := pC.values }
  let state : Fin 16 → Var := fun lane => columns.getD (package.outputs.state.getD lane 0) 0
  have applyRun : (apply package (Array.range 9) context).run p13 = ((claimVars package.outputs columns, state), pR) := by
    unfold apply
    rw [run_bind, run_bind]
    rfl
  have finalRun : (finalViews package context key).run {} = (layer1Views key (claimVars package.outputs columns) state).run pR := by
    unfold finalViews finalProgram
    rw [run_bind, readRun, statementRun, run_bind, applyRun]
  obtain ⟨more, instrsMore⟩ := (layer1Views.appends key (claimVars package.outputs columns) state).keeps pR
  -- The executor: the same parts, in order.
  obtain ⟨newB, instrsB, lengthB, simB⟩ := Sim.iteration (c := c) (n := 9) (P := ∀ j < 9, c.val j = stmt.getD j 0)
    (Array.range 9) (x := stmt.getD 0 0) (fun h => ⟨by simp, by simpa using h 0 (Nat.zero_lt_succ 8)⟩) p9 E9 rfl rfl
  let EB := c.steps newB E9
  have handles9 : ∀ j, (Array.range 9).getD j 0 < 13 := fun j => by
    rw [Array.getD_eq_getD_getElem?]
    by_cases small : j < 9
    · simp [small]
      omega
    · simp [small]
  obtain ⟨newC, instrsC, lengthC, simC⟩ := Sim.columns (c := c) (n := 13) (P := True) package (Array.range 9) context
    handles9 computed p13 EB values13 (lengthB.trans values13)
  let EC := c.steps newC EB
  let ER := c.step EC (.r1cs package.system.toList columns.toList)
  have run : Exec.run perm stmt tape L = c.steps more.toList ER := by
    rw [← program, Build.program, finalRun, instrsMore]
    show Exec.run c.perm c.stmt c.tape (pC.instrs.push _ ++ more) = _
    rw [Array.push_eq_append, instrsC, instrsB, Exec.run_append, Exec.run_append, Exec.run_append, Exec.run_append]
    rfl
  have followsR : c.Follows ER := Ctx.follows_of_steps (L := more.toList) fun h _ => by rw [← run]
  have followsC : c.Follows EC := Ctx.follows_of_step followsR
  have followsB : c.Follows EB := Ctx.follows_of_steps followsC
  have statement : ∀ j < 9, c.val j = stmt.getD j 0 := fun j lt => by
    rw [← Ctx.follows_of_steps followsB j (by simpa [E9] using lt)]
    simp only [E9, Exec.State.value]
    rw [List.getD_eq_getElem?_getD, List.getElem?_map, List.getElem?_range lt]
    rfl
  have holdsR : ER.holds := Ctx.holds_of_steps c (L := more.toList) (by rw [← run]; exact accepts)
  obtain ⟨-, iterationHolds⟩ := simB statement followsB
  obtain ⟨handles, -⟩ := simC trivial followsC
  have below : ∀ h ∈ columns.toList, h < EC.values.length := fun h member => by
    obtain ⟨s, -, handle⟩ := forall₂_exists_left handles member
    exact lengthC ▸ handle.1
  refine ⟨columns, ?_, ?_, ?_, ?_⟩
  · rw [finalRun]
    exact layer1Views_claim key _ state pR
  · refine handles.imp fun {s h} handle => ?_
    cases s with
    | one => exact handle.2
    | priv i =>
        intro lo hi
        have j := handle.2 lo hi
        have small : i - contextStart - 4 < 9 := by have := contextStart_eq; simp [PilotProduction.stateHashWords_eq] at hi; omega
        rw [j, Array.getD_eq_getD_getElem?, Array.getElem?_range, if_pos small]
        exact statement _ small
    | pub i => exact handle.2
    | computed _ => trivial
  · intro r member
    have check := holdsR.2 r member
    rwa [List.map_congr_left fun h member => followsC h (below h member)] at check
  · have holdsB : EB.holds := Ctx.holds_of_steps c (L := newC) holdsR.1
    exact left_ne_zero_of_mul_eq_one (iterationHolds.mp holdsB).2

end Wrapper

end CompressionLean.Circuit
