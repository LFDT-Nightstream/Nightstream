import CompressionLean.Circuit.Terminal
import CompressionLean.Circuit.PackageRows
import CompressionLean.Circuit.Sim
import CompressionLean.Final.Claim

/-!
Owns the terminal part of `ProgramEncodes`: when the executor accepts the
final program (`FinalProgram::run`, `lifecycle/terminal/mod.rs:54-80`), its
run holds a terminal assignment for its statement, and the claim and state
handles that `Package::apply` (`terminal/package.rs:185-243`) hands to layer 1
hold the output values of that assignment.

- `terminalPrefix`: `finalProgram` up to layer 1 (`finalProgram_eq`): the
  nine statement words, the iteration check, `apply`.
- `terminalEnv`: the package assignment of a run. A column that a row names
  holds the value of its system column, and every other column `unnamed`.
- `Handoff`: what layer 1 receives. The assignment is a terminal assignment
  for the run's statement (`TerminalFor`, `statementOf`), and the claim and
  state handles hold its output values (`ClaimHolds`, `StateHolds`).
- `terminalPrefix_ensures`: when the first nine values are the statement
  words (`ReadsStatement`), an accepting run of `terminalPrefix` meets
  `Handoff`. `finalProgram_exec` proves those values from the run of the
  final program, for a layer 1 that has its own `Ensures` lemma.

Does not own layer 1 or the meaning of the rows (`Terminal/Soundness.lean`).
-/

namespace CompressionLean.Circuit.Terminal

open CompressionLean.Circuit.Package
open CompressionLean.Replay (Gl)
open NightstreamFPrime.Circuit (Env Expr)
open NightstreamFPrime.Layout
open Fin.CommRing

/-! ## The program -/

/-- The statement of a run: its first nine statement words
(`FinalProgram::statement`, `terminal/mod.rs:46-52`). -/
def statementOf (c : Ctx NightstreamFPrime.Spec.F) : Final.StateWords := fun k => c.stmt.getD k 0

/-- The first nine values of the run are the statement words
(`terminal/mod.rs:55-59`). -/
def ReadsStatement (c : Ctx NightstreamFPrime.Spec.F) : Prop := ∀ k : Fin 9, c.val k = statementOf c k

/-- The nine `public` words of the statement (`terminal/mod.rs:55-59`). -/
def statementHandles : Build (Array Var) := (Array.range statementWords).mapM fun _ => Build.pub

/-- The iteration check, then `Package::apply` (`terminal/mod.rs:60-72`). -/
def checkedApply (package : Loaded) (context : Array Gl) (statement : Array Var) :
    Build (ClaimVars × (Fin 16 → Var)) := do
  let inverse ← Build.emit (.hint 1)
  let product ← Build.mul (statement.getD 0 0) inverse
  let one ← Build.const 1
  Build.assertEqual product one
  apply package statement context

/-- `FinalProgram::run` up to `verify_with` (`terminal/mod.rs:54-72`). -/
def terminalPrefix (package : Loaded) (context : Array Gl) : Build (ClaimVars × (Fin 16 → Var)) :=
  statementHandles >>= checkedApply package context

/-- `FinalProgram::run` is `terminalPrefix`, then `verify_with`
(`terminal/mod.rs:73-79`). -/
theorem finalProgram_eq {α : Type} (package : Loaded) (context : Array Gl)
    (layer1 : ClaimVars → (Fin 16 → Var) → Build α) :
    finalProgram package context layer1 = terminalPrefix package context >>= fun r => layer1 r.1 r.2 :=
  rfl

/-- Each `public` call appends one `pub` instruction (`terminal/mod.rs:58`). -/
private theorem pubs_run {δ : Type} : ∀ (xs : List δ) (p : Builder), (xs.mapM fun _ => Build.pub).run p =
    (List.range' p.values xs.length,
      { instrs := p.instrs ++ Array.replicate xs.length Instr.pub, values := p.values + xs.length })
  | [], p => by simp; rfl
  | x :: xs, p => by
      rw [List.mapM_cons]
      show ((xs.mapM fun _ => Build.pub).run _ >>= fun r => pure (p.values :: r.1, r.2)) = _
      rw [pubs_run xs]
      show (p.values :: List.range' (p.values + 1) xs.length, _) = _
      simp only [List.length_cons, List.range'_succ, Prod.mk.injEq, true_and, Builder.mk.injEq, Instr.outputs]
      refine ⟨Array.toList_inj.mp ?_, by omega⟩
      simp [List.replicate_succ]

/-- The statement handles are the next nine values, after nine `pub`
instructions (`terminal/mod.rs:55-59`). -/
theorem statementHandles_run (p : Builder) : statementHandles.run p =
    ((List.range' p.values 9).toArray,
      { instrs := p.instrs ++ Array.replicate 9 Instr.pub, values := p.values + 9 }) := by
  rw [statementHandles, Array.mapM_eq_mapM_toList]
  show (fun r => (r.1.toArray, r.2)) ((List.mapM (fun _ => Build.pub) (Array.range statementWords).toList).run p) = _
  rw [pubs_run]
  simp [statementWords]

/-- The statement handles, read as `statement[index]`
(`terminal/mod.rs:61-66`, `terminal/package.rs:207`). -/
private theorem range_getD (j : ℕ) : (List.range' 0 9).toArray.getD j 0 = if j < 9 then j else 0 := by
  rw [Array.getD_eq_getD_getElem?, List.getElem?_toArray]
  by_cases h : j < 9
  · rw [if_pos h, List.getElem?_range' h]
    simp
  · rw [if_neg h, List.getElem?_eq_none (by simpa using h)]
    rfl

section

variable {c : Ctx NightstreamFPrime.Spec.F} {n : ℕ} {P : Prop}

/-- The nine `public` words, simulated (`terminal/mod.rs:55-59`). -/
theorem statementHandles_ensures :
    Ensures c n P statementHandles (fun s m => s = (List.range' n 9).toArray ∧ m = n + 9) := by
  intro p E hp hE
  refine ⟨List.replicate 9 .pub, by rw [statementHandles_run]; rfl, ?_, fun _ _ _ => ?_⟩
  · obtain ⟨more, hm, hl⟩ := c.steps_values (List.replicate 9 .pub) E
    rw [hm, statementHandles_run, List.length_append, hl, hE, hp]
    simp [Instr.outputs]
  · rw [statementHandles_run, hp]
    exact ⟨rfl, rfl⟩

end

/-! ## The columns -/

/-- What `column` gives for one source (`terminal/package.rs:197-218`): a
handle below `m`. The one column holds one, a public context column its
context word, and a statement column of the prior preimage its statement
handle. -/
private def ColumnHolds (c : Ctx NightstreamFPrime.Spec.F) (statement : Array Var) (context : Array Gl) (src : Source)
    (h : Var) (m : ℕ) : Prop :=
  h < m ∧ match src with
    | .one => c.val h = 1
    | .priv i => contextStart + 4 ≤ i → i < PilotProduction.stateHashWords →
        h = statement.getD (i - contextStart - 4) 0
    | .pub i => publicContextStart ≤ i → c.val h = word (context.getD (i - publicContextStart) 0)
    | .computed _ => True

/-- A column handle stays below a larger count (`terminal/package.rs:197-218`). -/
private theorem ColumnHolds.mono {c : Ctx NightstreamFPrime.Spec.F} {statement : Array Var} {context : Array Gl}
    {src : Source} {h : Var} {m m' : ℕ} (hold : ColumnHolds c statement context src h m) (le : m ≤ m') :
    ColumnHolds c statement context src h m' :=
  ⟨lt_of_lt_of_le hold.1 le, hold.2⟩

section

variable {c : Ctx NightstreamFPrime.Spec.F} {n : ℕ} {P : Prop}

/-- `column`, simulated (`terminal/package.rs:200-217`). -/
private theorem column_sim (statement : Array Var) (context : Array Gl) (computed : Var) (src : Source)
    (hs : P → ∀ j, statement.getD j 0 < n) (hc : P → ∀ i, src = .computed i → computed + i < n) :
    Sim c n P (column statement context computed src) (ColumnHolds c statement context src) True := by
  have chunk : NightstreamFPrime.Lifecycle.stateDomainChunk.length ≤ contextStart := by
    rw [NightstreamFPrime.Lifecycle.stateDomainChunk_length]
    decide
  cases src with
  | one => exact (Sim.const 1).post (fun _ h m w => ⟨w.1, by rw [w.2, word_one]⟩) fun _ => Iff.rfl
  | priv i =>
      dsimp only [column]
      split_ifs with h1 h2 h3
      · exact (Sim.const _).post (fun _ h m w => ⟨w.1, fun lo => absurd h1 (by omega)⟩) fun _ => Iff.rfl
      · exact (Sim.const _).post (fun _ h m w => ⟨w.1, fun lo => absurd h2.2 (by omega)⟩) fun _ => Iff.rfl
      · exact Sim.pure fun hP => ⟨hs hP _, fun _ _ => rfl⟩
      · exact Sim.priv.post (fun _ h m w => ⟨w.1, fun lo hi => absurd ⟨lo, hi⟩ h3⟩) fun _ => Iff.rfl
  | pub i =>
      dsimp only [column]
      split_ifs with h1
      · exact (Sim.const _).post (fun _ h m w => ⟨w.1, fun _ => w.2⟩) fun _ => Iff.rfl
      · exact Sim.priv.post (fun _ h m w => ⟨w.1, fun lo => absurd lo h1⟩) fun _ => Iff.rfl
  | computed k => exact Sim.pure fun hP => ⟨hc hP k rfl, trivial⟩

end

/-! ## The terminal layout -/

/-- A fold that pushes one entry per step keeps its prefix
(`private_input_index`, `nightstream-fprime/src/package.rs:365-377`). -/
private theorem foldl_prefix {α β γ : Type} {f : Array α × β → γ → Array α × β}
    (hf : ∀ acc x, ∃ y, (f acc x).1 = acc.1.push y) :
    ∀ (l : List γ) acc, ∃ more, (l.foldl f acc).1.toList = acc.1.toList ++ more
  | [], acc => ⟨[], by simp⟩
  | x :: l, acc => by
      obtain ⟨y, hy⟩ := hf acc x
      obtain ⟨more, e⟩ := foldl_prefix hf l (f acc x)
      exact ⟨y :: more, by rw [List.foldl_cons, e, hy]; simp⟩

/-- The prior state preimage is the first caller input, from package column
0, as `load` requires (`terminal/package.rs:95`). -/
private theorem callerSegments_head : ∃ more, callerSegments (CompressionLean.Terminal.circuitPackage ()).layout =
    (⟨NightstreamFPrime.Export.Stage1.Data.Role.priorPreimage, 0, PilotProduction.stateHashWords⟩, 0) :: more := by
  obtain ⟨rest, hrest⟩ : ∃ rest, (CompressionLean.Terminal.circuitPackage ()).layout.privateSegments =
      ⟨NightstreamFPrime.Export.Stage1.Data.Role.priorPreimage, 0, PilotProduction.stateHashWords⟩ :: rest :=
    ⟨_, rfl⟩
  unfold callerSegments
  rw [hrest, List.filter_cons_of_pos (by decide), List.foldl_cons]
  exact foldl_prefix (fun acc x => ⟨_, rfl⟩) _ _

/-- A column of the prior preimage is the private input of its own index
(`private_input_index`, `nightstream-fprime/src/package.rs:365-377`). -/
private theorem privateInput_prior {pc : ℕ} (hpc : pc < PilotProduction.stateHashWords) :
    privateInput (callerSegments (CompressionLean.Terminal.circuitPackage ()).layout) pc = some pc := by
  obtain ⟨more, h⟩ := callerSegments_head
  rw [h, privateInput]
  simp [hpc]

/-- The statement words of the prior preimage are the package columns from
`contextStart + 4` on (`STATE_CONTEXT_START`, `terminal/package.rs:205-209`). -/
private theorem priorColumn (k : Fin 9) :
    Stage1.Spartan.sourceToSpartan (PilotProduction.priorPreimageStart +
      (NightstreamFPrime.Lifecycle.PiCCS.v1_2.StateBinding.contextWordStart + 4 + k.val)) = contextStart + 4 + k := by
  revert k
  decide

/-- The column counts of the terminal layout (`PUBLIC_CONTEXT_START`,
`terminal/package.rs:46`; `STATE_CONTEXT_START`, `lifecycle/inputs.rs:43`). -/
private theorem layout_facts :
    (CompressionLean.Terminal.circuitPackage ()).layout.privateColumnCount = Stage1.Spartan.privateColumnCount ∧
    (CompressionLean.Terminal.circuitPackage ()).layout.constantColumn = Stage1.Spartan.privateColumnCount ∧
    (CompressionLean.Terminal.circuitPackage ()).layout.totalColumnCount = Stage1.Spartan.privateColumnCount + 279 ∧
    Stage1.Spartan.expectedContextPublicStart = Stage1.Spartan.privateColumnCount + 275 ∧
    PilotProduction.stateHashWords < Stage1.Spartan.privateColumnCount ∧
    contextStart + 13 = PilotProduction.stateHashWords := by
  refine ⟨rfl, rfl, ?_, rfl, ?_, by decide⟩
  · show Stage1.Spartan.spartanColumnCount = _
    rw [Stage1.Spartan.spartanColumnCount_decomposition, Stage1.Spartan.publicColumnCount_eq]
  · rw [Stage1.Spartan.privateColumnCount_decomposition, PilotProduction.stateHashWords_eq]
    show 27819 < 5141486 + _
    omega

/-! ## What layer 1 receives -/

/-- Handles `hs` lie below `m` and hold the values of the expressions `list`
on `env`, in order (`terminal/package.rs:221-241`). -/
def HoldsValues (c : Ctx NightstreamFPrime.Spec.F) (m : ℕ) (env : Env) (hs : List Var) (list : List Expr) : Prop :=
  List.Forall₂ (fun h e => h < m ∧ c.val h = e.eval env) hs list

/-- The handles of word pairs, in order (`terminal/package.rs:227-237`). -/
def pairHandles (pairs : Array (Var × Var)) : List Var := pairs.toList.flatMap fun p => [p.1, p.2]

/-- The claim handles hold the output values of the PiRLC parent on `env`, in
`claimOf`'s order (`terminal/package.rs:230-240`). -/
def ClaimHolds (c : Ctx NightstreamFPrime.Spec.F) (m : ℕ) (claim : ClaimVars) (env : Env) : Prop :=
  HoldsValues c m env (claim.commitment.toList.flatMap Array.toList) CompressionLean.Terminal.outputs.commitment ∧
  HoldsValues c m env (claim.publicInput.toList.flatMap Array.toList) CompressionLean.Terminal.outputs.publicInput ∧
  HoldsValues c m env (pairHandles claim.point) CompressionLean.Terminal.outputs.point ∧
  HoldsValues c m env (pairHandles claim.evalK) CompressionLean.Terminal.outputs.eval_K ∧
  HoldsValues c m env (claim.evalA.toList.flatMap pairHandles) CompressionLean.Terminal.outputs.eval_A

/-- The state handles hold the sampler's output state on `env`
(`terminal/package.rs:241`). -/
def StateHolds (c : Ctx NightstreamFPrime.Spec.F) (m : ℕ) (state : Fin 16 → Var) (env : Env) : Prop :=
  HoldsValues c m env (List.ofFn state) CompressionLean.Terminal.outputs.state

/-- What `apply` hands to `verify_with` (`terminal/mod.rs:70-79`): a terminal
assignment for the run's statement, whose output values the claim and state
handles hold. -/
def Handoff (c : Ctx NightstreamFPrime.Spec.F) (authority : NightstreamFPrime.Lifecycle.VerifierContext.Authority)
    (r : ClaimVars × (Fin 16 → Var)) (m : ℕ) : Prop :=
  ∃ env, Final.TerminalFor authority (statementOf c) env ∧ ClaimHolds c m r.1 env ∧ StateHolds c m r.2 env

/-! ## Chunks -/

/-- The first `q` chunks of `xs`, joined (`chunks_exact`,
`terminal/package.rs:222-239`). -/
private theorem chunks_take {α : Type} (n : ℕ) (xs : Array α) :
    ∀ q, n * q ≤ xs.size →
      ((List.range q).flatMap fun k => (xs.extract (n * k) (n * k + n)).toList) = xs.toList.take (n * q)
  | 0, _ => by simp
  | q + 1, le => by
      rw [Nat.mul_succ] at le
      rw [List.range_succ, List.flatMap_append, chunks_take n xs q (by omega), List.flatMap_singleton,
        Array.toList_extract, List.extract_eq_take_drop, Nat.mul_succ, List.take_add]
      simp

/-- Chunks of a size that divides the length join to the list (`chunks_exact`,
`terminal/package.rs:222-239`). -/
private theorem chunks_flatten {α : Type} {n : ℕ} {xs : Array α} (dvd : n ∣ xs.size) :
    (chunks n xs).toList.flatMap Array.toList = xs.toList := by
  rw [chunks, Array.toList_map, Array.toList_range, List.flatMap_map,
    chunks_take n xs _ (Nat.mul_div_le xs.size n), Nat.mul_div_cancel' dvd]
  simp

/-- Every chunk has `n` entries (`chunks_exact`, `terminal/package.rs:222-239`). -/
private theorem chunks_size {α : Type} {n : ℕ} {xs : Array α} {chunk : Array α} (mem : chunk ∈ chunks n xs) :
    chunk.size = n := by
  rw [chunks, Array.mem_map] at mem
  obtain ⟨k, hk, rfl⟩ := mem
  rw [Array.mem_range] at hk
  have le : n * (k + 1) ≤ xs.size := le_trans (Nat.mul_le_mul_left n hk) (Nat.mul_div_le xs.size n)
  rw [Nat.mul_succ] at le
  rw [Array.size_extract]
  omega

/-- `flatMap` over equal images (`terminal/package.rs:230-240`). -/
private theorem flatMap_congr {α β : Type} {l : List α} {f g : α → List β} (h : ∀ x ∈ l, f x = g x) :
    l.flatMap f = l.flatMap g := by
  rw [List.flatMap_def, List.flatMap_def, List.map_congr_left h]

/-- Pairs of an even list join to the list (`chunks_exact(2)`,
`terminal/package.rs:233-237`). -/
private theorem pairs_flatten {xs : Array Var} (dvd : 2 ∣ xs.size) : pairHandles (pairs xs) = xs.toList := by
  rw [pairHandles, pairs, Array.toList_map, List.flatMap_map, ← chunks_flatten dvd]
  refine flatMap_congr fun pair mem => ?_
  have size := chunks_size (Array.mem_toList_iff.mp mem)
  obtain ⟨l⟩ := pair
  match l, size with
  | [a, b], _ => rfl

/-! ## The assignment -/

/-- The values of the package columns that no row names: one on the one
column, the context words on the public context columns, the statement words
on the statement columns of the prior preimage, and zero elsewhere
(`terminal/package.rs:197-218`). -/
private def unnamed (c : Ctx NightstreamFPrime.Spec.F) (context : Array Gl) (pc : ℕ) : NightstreamFPrime.Spec.F :=
  if pc = Stage1.Spartan.privateColumnCount then 1
  else if Stage1.Spartan.expectedContextPublicStart ≤ pc then
    word (context.getD (pc - Stage1.Spartan.expectedContextPublicStart) 0)
  else if contextStart + 4 ≤ pc ∧ pc < PilotProduction.stateHashWords then c.stmt.getD (pc - (contextStart + 4)) 0
  else 0

/-- The package assignment of a run: a named column holds the value of its
system column (`assignment`, `terminal/package.rs:197-219`). -/
private def terminalEnv (c : Ctx NightstreamFPrime.Spec.F) (context : Array Gl) (columns : Array Var) : Env :=
  assignment (numbering (CompressionLean.Terminal.circuitPackage ())).1 (fun j => (columns.toList.map c.val).getD j 0)
    (unnamed c context)

section Assignment

variable {c : Ctx NightstreamFPrime.Spec.F} {statement : Array Var} {context : Array Gl} {columns : Array Var}
  {m : ℕ} {loaded : Loaded}

/-- A computed source names a hint output (`Source::Computed`,
`terminal/package.rs:216`). -/
private theorem computed_lt
    (loads : load (CompressionLean.Terminal.circuitPackage ()) CompressionLean.Terminal.outputs = .ok loaded) :
    ∀ x ∈ loaded.sources, ∀ i, x = .computed i → i < loaded.computed.size := by
  obtain ⟨-, sources, computed, -⟩ := load_ok loads
  intro x mem i e
  rw [sources] at mem
  rw [computed]
  obtain ⟨k, hk, rfl⟩ := Array.mem_iff_getElem.mp mem
  exact (numbering_numbered _).computed k hk i e

/-- Every column handle lies below the count (`terminal/package.rs:197-218`). -/
private theorem columns_lt : ∀ {xs : List Source} {hs : List Var},
    List.Forall₂ (fun x a => ColumnHolds c statement context x a m) xs hs → ∀ h ∈ hs, h < m
  | _, _, .nil => fun _ mem => absurd mem List.not_mem_nil
  | _, _, .cons hold rest => fun h mem => (List.mem_cons.mp mem).elim (fun e => e ▸ hold.1) (columns_lt rest h)

/-- The numbering has an entry for every package column (`load`,
`terminal/package.rs:92-160`). -/
private theorem index_some (pc : ℕ) (lt : pc < (CompressionLean.Terminal.circuitPackage ()).layout.totalColumnCount) :
    ∃ x, (numbering (CompressionLean.Terminal.circuitPackage ())).1[pc]? = some x :=
  ⟨_, Array.getElem?_eq_getElem (by rw [(numbering_numbered _).size]; exact lt)⟩

/-- A named column: its source, what `column` gives for it, and its value in
`terminalEnv` (`terminal/package.rs:197-219`). -/
private theorem env_named
    (loads : load (CompressionLean.Terminal.circuitPackage ()) CompressionLean.Terminal.outputs = .ok loaded)
    (cols : List.Forall₂ (fun x a => ColumnHolds c statement context x a m) loaded.sources.toList columns.toList)
    {pc k : ℕ} (hk : (numbering (CompressionLean.Terminal.circuitPackage ())).1[pc]? = some (some k)) :
    ∃ lt : k < (numbering (CompressionLean.Terminal.circuitPackage ())).2.1.size,
      SourceOf (CompressionLean.Terminal.circuitPackage ()).layout
        (callerSegments (CompressionLean.Terminal.circuitPackage ()).layout) pc
        (numbering (CompressionLean.Terminal.circuitPackage ())).2.1[k] ∧
      ColumnHolds c statement context (numbering (CompressionLean.Terminal.circuitPackage ())).2.1[k]
        (columns.getD k 0) m ∧
      terminalEnv c context columns pc = c.val (columns.getD k 0) := by
  rw [(load_ok loads).2.1] at cols
  obtain ⟨lt, src⟩ := (numbering_numbered _).source pc k hk
  obtain ⟨len, each⟩ := List.forall₂_iff_get.mp cols
  rw [Array.length_toList, Array.length_toList] at len
  have klt : k < columns.size := by
    rw [← len]
    exact lt
  have hold := each k (by rw [Array.length_toList]; exact lt) (by rw [Array.length_toList]; exact klt)
  rw [List.get_eq_getElem, List.get_eq_getElem, Array.getElem_toList, Array.getElem_toList] at hold
  have getD : columns.getD k 0 = columns[k] := by simp [Array.getD, klt]
  have bang : (numbering (CompressionLean.Terminal.circuitPackage ())).1[pc]! = some k := by
    rw [Array.getElem!_eq_getD, Array.getD_eq_getD_getElem?, hk]
    rfl
  rw [getD]
  refine ⟨lt, src, hold, ?_⟩
  rw [terminalEnv, assignment, if_neg (by rw [hk]; simp), bang, Option.getD_some]
  simp [List.getD_eq_getElem?_getD, klt]

/-- The one column holds one (`Source::One`, `terminal/package.rs:201`). -/
private theorem env_one
    (loads : load (CompressionLean.Terminal.circuitPackage ()) CompressionLean.Terminal.outputs = .ok loaded)
    (cols : List.Forall₂ (fun x a => ColumnHolds c statement context x a m) loaded.sources.toList columns.toList) :
    terminalEnv c context columns (CompressionLean.Terminal.circuitPackage ()).layout.constantColumn = 1 := by
  obtain ⟨hpc, hcc, htotal, -, -, -⟩ := layout_facts
  rw [hcc]
  obtain ⟨x, hx⟩ := index_some Stage1.Spartan.privateColumnCount (by rw [htotal]; omega)
  cases x with
  | none => rw [terminalEnv, assignment, if_pos hx, unnamed, if_pos rfl]
  | some k =>
      obtain ⟨lt, src, hold, env⟩ := env_named loads cols hx
      rw [env]
      generalize (numbering (CompressionLean.Terminal.circuitPackage ())).2.1[k] = s at src hold
      cases s with
      | one => exact hold.2
      | priv i => exact absurd src.1 (by rw [hpc]; omega)
      | pub i => exact absurd src.1 (by rw [hpc]; omega)
      | computed i => exact absurd src.1 (by rw [hpc]; omega)

/-- The public context columns hold the context digest
(`terminal/package.rs:212-214`). -/
private theorem env_context
    (loads : load (CompressionLean.Terminal.circuitPackage ()) CompressionLean.Terminal.outputs = .ok loaded)
    (cols : List.Forall₂ (fun x a => ColumnHolds c statement context x a m) loaded.sources.toList columns.toList)
    {authority : NightstreamFPrime.Lifecycle.VerifierContext.Authority}
    (hctx : ∀ lane : Fin 4,
      word (context.getD lane 0) = (NightstreamFPrime.Lifecycle.VerifierContext.digest authority).getD lane 0) :
    NightstreamFPrime.Export.Stage1.Package.SelectedVerifierContext authority (terminalEnv c context columns) := by
  obtain ⟨hpc, -, htotal, hstart, -, -⟩ := layout_facts
  intro lane
  have lane4 := lane.isLt
  obtain ⟨x, hx⟩ := index_some (Stage1.Spartan.expectedContextPublicStart + lane) (by rw [htotal, hstart]; omega)
  cases x with
  | none =>
      rw [terminalEnv, assignment, if_pos hx, unnamed, if_neg (by rw [hstart]; omega), if_pos (Nat.le_add_right _ _),
        Nat.add_sub_cancel_left]
      exact hctx lane
  | some k =>
      obtain ⟨lt, src, hold, env⟩ := env_named loads cols hx
      rw [env]
      generalize (numbering (CompressionLean.Terminal.circuitPackage ())).2.1[k] = s at src hold
      cases s with
      | pub i =>
          obtain ⟨-, hi⟩ := src
          have e : i = publicContextStart + lane := by
            rw [hi, hpc, hstart, publicContextStart]
            omega
          rw [hold.2 (by rw [e]; omega), e, Nat.add_sub_cancel_left]
          exact hctx lane
      | one =>
          simp only [SourceOf] at src
          rw [hpc, hstart] at src
          omega
      | priv i => exact absurd src.1 (by rw [hpc, hstart]; omega)
      | computed i => exact absurd src.1 (by rw [hpc, hstart]; omega)

/-- The statement words of the prior preimage are the run's statement
(`terminal/package.rs:205-209`). -/
private theorem env_prior
    (loads : load (CompressionLean.Terminal.circuitPackage ()) CompressionLean.Terminal.outputs = .ok loaded)
    (cols : List.Forall₂ (fun x a => ColumnHolds c statement context x a m) loaded.sources.toList columns.toList)
    (hstmt : statement = (List.range' 0 9).toArray) (reads : ReadsStatement c) :
    Final.priorStatement (terminalEnv c context columns) = statementOf c := by
  obtain ⟨hpc, -, htotal, hstart, hshw, hcs⟩ := layout_facts
  funext k
  have k9 := k.isLt
  rw [Final.priorStatement, Stage1.PiCCSInputs.priorStateWord, NightstreamFPrime.Circuit.Expr.eval_var,
    Stage1.Spartan.pullback, priorColumn k]
  obtain ⟨x, hx⟩ := index_some (contextStart + 4 + k) (by rw [htotal]; omega)
  cases x with
  | none =>
      rw [terminalEnv, assignment, if_pos hx, unnamed, if_neg (by omega), if_neg (by rw [hstart]; omega),
        if_pos ⟨by omega, by omega⟩, Nat.add_sub_cancel_left]
      rfl
  | some j =>
      obtain ⟨lt, src, hold, env⟩ := env_named loads cols hx
      rw [env]
      generalize (numbering (CompressionLean.Terminal.circuitPackage ())).2.1[j] = s at src hold
      cases s with
      | priv i =>
          obtain ⟨-, e⟩ := src
          rw [privateInput_prior (by omega)] at e
          cases e
          rw [hold.2 (by omega) (by omega), hstmt, show contextStart + 4 + k - contextStart - 4 = k by omega,
            range_getD, if_pos k9]
          exact reads k
      | one =>
          simp only [SourceOf] at src
          rw [hpc] at src
          omega
      | pub i => exact absurd src.1 (by rw [hpc]; omega)
      | computed i =>
          obtain ⟨-, e⟩ := src
          rw [privateInput_prior (by omega)] at e
          cases e

/-- The rows hold on `terminalEnv` (`b.r1cs`, `terminal/package.rs:219`). -/
private theorem env_rows
    (loads : load (CompressionLean.Terminal.circuitPackage ()) CompressionLean.Terminal.outputs = .ok loaded)
    (valid : (CompressionLean.Terminal.circuitPackage ()).CompactTemplatesValid)
    (cols : List.Forall₂ (fun x a => ColumnHolds c statement context x a m) loaded.sources.toList columns.toList)
    (rows : ∀ r ∈ loaded.system.toList, R1csRow.Holds (fun j => (columns.toList.map c.val).getD j 0) r) :
    (CompressionLean.Terminal.circuitPackage ()).RowsHold (terminalEnv c context columns) :=
  (system_holds_iff loads valid (fun j => (columns.toList.map c.val).getD j 0) (unnamed c context)
    (env_one loads cols)).mp fun r mem => rows r (Array.mem_toList_iff.mpr mem)

/-- The handles of an output list hold its values (`columns[list[i]]`,
`terminal/package.rs:221-241`). -/
private theorem read_holds
    (loads : load (CompressionLean.Terminal.circuitPackage ()) CompressionLean.Terminal.outputs = .ok loaded)
    (cols : List.Forall₂ (fun x a => ColumnHolds c statement context x a m) loaded.sources.toList columns.toList)
    {list : List Expr} {ks : Array ℕ}
    (h : outputColumns (CompressionLean.Terminal.circuitPackage ()).layout
      (numbering (CompressionLean.Terminal.circuitPackage ())).1 list = .ok ks) :
    HoldsValues c m (terminalEnv c context columns) (ks.toList.map fun k => columns.getD k 0) list := by
  obtain ⟨size, each⟩ := outputColumns_ok h
  refine List.forall₂_iff_get.mpr ⟨by simp [size], fun i h₁ h₂ => ?_⟩
  obtain ⟨pc, k, e, lt, hk, hks⟩ := each i h₂
  have hk' : (numbering (CompressionLean.Terminal.circuitPackage ())).1[pc]? = some (some k) := by
    obtain ⟨x, hx⟩ := index_some pc lt
    rw [Array.getElem!_eq_getD, Array.getD_eq_getD_getElem?, hx] at hk
    rw [hx, ← hk]
    rfl
  obtain ⟨_, _, hold, env⟩ := env_named loads cols hk'
  have iks : i < ks.size := by rw [size]; exact h₂
  have ki : ks[i] = k := by
    rw [Array.getElem?_eq_getElem iks] at hks
    exact Option.some.inj hks
  simp only [List.get_eq_getElem, List.getElem_map, Array.getElem_toList, ki, e,
    NightstreamFPrime.Circuit.Expr.eval]
  exact ⟨hold.1, env.symm⟩

end Assignment

/-! ## The program, simulated -/

/-- The lengths of the output lists that `ring` and `extension` read
(`terminal/package.rs:222-241`). -/
private theorem outputs_length :
    CompressionLean.Terminal.outputs.commitment.length = 22 * Replay.D ∧
    CompressionLean.Terminal.outputs.publicInput.length = 5 * Replay.D ∧
    CompressionLean.Terminal.outputs.point.length = 2 * 28 ∧
    CompressionLean.Terminal.outputs.eval_K.length = 2 * Replay.D ∧
    CompressionLean.Terminal.outputs.eval_A.length = 4 * (2 * Replay.D) ∧
    CompressionLean.Terminal.outputs.state.length = 16 := by
  refine ⟨?_, ?_, ?_, ?_, ?_, by simp [CompressionLean.Terminal.outputs]⟩ <;>
    simp [CompressionLean.Terminal.outputs, CompressionLean.Terminal.extension, List.length_flatten, List.map_ofFn,
      Function.comp_def] <;> decide

/-- The state handles, as a list (`terminal/package.rs:241`). -/
private theorem state_list {columns : Array Var} {ks : Array ℕ} (size : ks.size = 16) :
    List.ofFn (fun lane : Fin 16 => columns.getD (ks.getD lane 0) 0) = ks.toList.map fun k => columns.getD k 0 := by
  apply List.ext_getElem (by simp [size])
  intro i h₁ h₂
  have hi : i < ks.size := by simpa using h₂
  have getD : ks.getD i 0 = ks[i] := by
    rw [Array.getD_eq_getD_getElem?, Array.getElem?_eq_getElem hi]
    rfl
  rw [List.getElem_ofFn, List.getElem_map, Array.getElem_toList, ← getD]

section Program

variable {c : Ctx NightstreamFPrime.Spec.F} {P : Prop} {loaded : Loaded} {context : Array Gl}
  {authority : NightstreamFPrime.Lifecycle.VerifierContext.Authority}

/-- `Package::apply`, simulated (`terminal/package.rs:185-243`). -/
theorem apply_ensures
    (loads : load (CompressionLean.Terminal.circuitPackage ()) CompressionLean.Terminal.outputs = .ok loaded)
    (valid : (CompressionLean.Terminal.circuitPackage ()).CompactTemplatesValid)
    (hctx : ∀ lane : Fin 4,
      word (context.getD lane 0) = (NightstreamFPrime.Lifecycle.VerifierContext.digest authority).getD lane 0)
    {statement : Array Var} {n : ℕ}
    (hP : P → statement = (List.range' 0 9).toArray ∧ 9 ≤ n ∧ ReadsStatement c ∧ statementOf c 0 ≠ 0) :
    Ensures c n P (apply loaded statement context) (Handoff c authority) := by
  have hs : P → ∀ j, statement.getD j 0 < n := fun h j => by
    obtain ⟨e, le, -⟩ := hP h
    rw [e, range_getD]
    split
    · rename_i hj
      exact lt_of_lt_of_le hj le
    · exact lt_of_lt_of_le (by decide) le
  unfold apply
  refine (Sim.emit (Q := fun h m => h + loaded.computed.size ≤ m) (C := True) (.hint loaded.computed.size)
    fun _ E hl hF => ⟨by simp [Instr.outputs], Iff.rfl⟩).ensures.bind fun computed m₁ le₁ => ?_
  refine (Sim.mapM (ColumnHolds c statement context) (fun _ _ _ _ h le => h.mono le) fun x mem m' le' =>
    column_sim statement context computed x (fun h j => lt_of_lt_of_le (hs h.1 j) (le₁.trans le'))
      fun h i e => lt_of_lt_of_le (Nat.add_lt_add_left (computed_lt loads x mem i e) computed)
        (h.2.1.trans le')).ensures.bind fun columns m₂ le₂ => ?_
  refine (Sim.r1cs (rows := loaded.system.toList) fun h => columns_lt h.2.1).ensures.bind fun _ m₃ le₃ => ?_
  refine Ensures.pure fun h => ?_
  obtain ⟨⟨⟨hP', -, -⟩, cols, -⟩, -, rows⟩ := h
  obtain ⟨hstmt, -, reads, nz⟩ := hP hP'
  replace cols := cols.imp fun _ _ hold => hold.mono le₃
  obtain ⟨-, -, -, hcom, hpub, hpt, hk, ha, hst⟩ := load_ok loads
  obtain ⟨lcom, lpub, lpt, lk, la, lst⟩ := outputs_length
  have read : ∀ {list : List Expr} {ks : Array ℕ}, outputColumns (CompressionLean.Terminal.circuitPackage ()).layout
      (numbering (CompressionLean.Terminal.circuitPackage ())).1 list = .ok ks →
      (ks.map fun k => columns.getD k 0).toList = (ks.toList.map fun k => columns.getD k 0) ∧
        (ks.map fun k => columns.getD k 0).size = list.length := fun h =>
    ⟨Array.toList_map .., by rw [Array.size_map, (outputColumns_ok h).1]⟩
  refine ⟨terminalEnv c context columns, ⟨env_rows loads valid cols rows, env_context loads cols hctx,
    env_prior loads cols hstmt reads, nz⟩, ⟨?_, ?_, ?_, ?_, ?_⟩, ?_⟩
  · show HoldsValues c m₃ _ ((chunks Replay.D (loaded.outputs.commitment.map _)).toList.flatMap Array.toList) _
    rw [chunks_flatten (by rw [(read hcom).2, lcom]; exact Nat.dvd_mul_left _ _), (read hcom).1]
    exact read_holds loads cols hcom
  · show HoldsValues c m₃ _ ((chunks Replay.D (loaded.outputs.publicInput.map _)).toList.flatMap Array.toList) _
    rw [chunks_flatten (by rw [(read hpub).2, lpub]; exact Nat.dvd_mul_left _ _), (read hpub).1]
    exact read_holds loads cols hpub
  · show HoldsValues c m₃ _ (pairHandles (pairs (loaded.outputs.point.map _))) _
    rw [pairs_flatten (by rw [(read hpt).2, lpt]; exact Nat.dvd_mul_right _ _), (read hpt).1]
    exact read_holds loads cols hpt
  · show HoldsValues c m₃ _ (pairHandles (pairs (loaded.outputs.evalK.map _))) _
    rw [pairs_flatten (by rw [(read hk).2, lk]; exact Nat.dvd_mul_right _ _), (read hk).1]
    exact read_holds loads cols hk
  · show HoldsValues c m₃ _ (((chunks (2 * Replay.D) (loaded.outputs.evalA.map _)).map pairs).toList.flatMap
      pairHandles) _
    rw [Array.toList_map, List.flatMap_map]
    rw [flatMap_congr fun chunk mem => pairs_flatten (by
      rw [chunks_size (Array.mem_toList_iff.mp mem)]
      exact Nat.dvd_mul_right _ _)]
    rw [chunks_flatten (by rw [(read ha).2, la]; exact Nat.dvd_mul_left _ _), (read ha).1]
    exact read_holds loads cols ha
  · show HoldsValues c m₃ _ (List.ofFn fun lane : Fin 16 => columns.getD (loaded.outputs.state.getD lane 0) 0) _
    rw [state_list (by rw [(outputColumns_ok hst).1, lst])]
    exact read_holds loads cols hst

/-- The iteration check, then `Package::apply`, simulated: the check gives a
nonzero iteration (`terminal/mod.rs:60-72`). -/
theorem checkedApply_ensures
    (loads : load (CompressionLean.Terminal.circuitPackage ()) CompressionLean.Terminal.outputs = .ok loaded)
    (valid : (CompressionLean.Terminal.circuitPackage ()).CompactTemplatesValid)
    (hctx : ∀ lane : Fin 4,
      word (context.getD lane 0) = (NightstreamFPrime.Lifecycle.VerifierContext.digest authority).getD lane 0)
    {statement : Array Var} {n : ℕ}
    (hP : P → statement = (List.range' 0 9).toArray ∧ 9 ≤ n ∧ ReadsStatement c) :
    Ensures c n P (checkedApply loaded context statement) (Handoff c authority) := by
  have zero : P → statement.getD 0 0 = 0 := fun h => by rw [(hP h).1, range_getD]; rfl
  unfold checkedApply
  refine (Sim.emit (Q := fun h m => h < m) (C := True) (.hint 1) fun _ E hl hF =>
    ⟨by simp [Instr.outputs], Iff.rfl⟩).ensures.bind fun inverse m₁ le₁ => ?_
  refine (Sim.mul (ga := Package.word (c.val (statement.getD 0 0))) (gb := Package.word (c.val inverse))
    (fun h => ⟨by rw [zero h.1]; exact lt_of_lt_of_le (lt_of_lt_of_le (by decide) (hP h.1).2.1) le₁,
      (word_word _).symm⟩)
    fun h => ⟨h.2.1, (word_word _).symm⟩).ensures.bind fun product m₂ le₂ => ?_
  refine (Sim.const 1).ensures.bind fun one m₃ le₃ => ?_
  refine (Sim.assertEqual (fun h => h.1.2.1.mono le₃) fun h => h.2.1).ensures.bind fun _ m₄ le₄ => ?_
  refine apply_ensures loads valid hctx fun h => ?_
  obtain ⟨⟨⟨⟨hP', -⟩, -⟩, -⟩, -, one⟩ := h
  obtain ⟨hstmt, le, reads⟩ := hP hP'
  refine ⟨hstmt, by omega, reads, fun zero0 => ?_⟩
  have := congrArg (word : Gl → NightstreamFPrime.Spec.F) one
  rw [word_mul, word_word, word_word, word_one, zero hP', show c.val 0 = statementOf c 0 from reads 0, zero0,
    zero_mul] at this
  exact absurd this (by decide)

/-- **The terminal part of `ProgramEncodes`.** When the run's checks hold and
its first nine values are the statement words, `terminalPrefix`
(`FinalProgram::run` up to `verify_with`, `terminal/mod.rs:54-72`) hands to
layer 1 claim and state handles that hold the output values of a terminal
assignment for the run's statement (`terminal/package.rs:185-243`). -/
theorem terminalPrefix_ensures
    (loads : load (CompressionLean.Terminal.circuitPackage ()) CompressionLean.Terminal.outputs = .ok loaded)
    (valid : (CompressionLean.Terminal.circuitPackage ()).CompactTemplatesValid)
    (hctx : ∀ lane : Fin 4,
      word (context.getD lane 0) = (NightstreamFPrime.Lifecycle.VerifierContext.digest authority).getD lane 0)
    (reads : P → ReadsStatement c) :
    Ensures c 0 P (terminalPrefix loaded context) (Handoff c authority) :=
  statementHandles_ensures.bind fun _ _ _ =>
    checkedApply_ensures loads valid hctx fun h => ⟨h.2.1, by rw [h.2.2], reads h.1⟩

/-- `FinalProgram::run` with a layer 1 that has its own `Ensures` lemma
from `Handoff` (`terminal/mod.rs:54-80`). -/
theorem finalProgram_ensures {α : Type} {layer1 : ClaimVars → (Fin 16 → Var) → Build α}
    {R : α → ℕ → Prop}
    (loads : load (CompressionLean.Terminal.circuitPackage ()) CompressionLean.Terminal.outputs = .ok loaded)
    (valid : (CompressionLean.Terminal.circuitPackage ()).CompactTemplatesValid)
    (hctx : ∀ lane : Fin 4,
      word (context.getD lane 0) = (NightstreamFPrime.Lifecycle.VerifierContext.digest authority).getD lane 0)
    (reads : P → ReadsStatement c)
    (hlayer1 : ∀ r m, Ensures c m (P ∧ Handoff c authority r m) (layer1 r.1 r.2) R) :
    Ensures c 0 P (finalProgram loaded context layer1) R := by
  rw [finalProgram_eq]
  exact (terminalPrefix_ensures loads valid hctx reads).bind fun r m _ => hlayer1 r m

/-- **An accepting run of the final program.** Its first nine values are the
statement words, so layer 1 receives `Handoff` and its result meets `R`
(`terminal/mod.rs:54-80`). -/
theorem finalProgram_exec {α : Type} {layer1 : ClaimVars → (Fin 16 → Var) → Build α}
    {R : α → ℕ → Prop}
    (loads : load (CompressionLean.Terminal.circuitPackage ()) CompressionLean.Terminal.outputs = .ok loaded)
    (valid : (CompressionLean.Terminal.circuitPackage ()).CompactTemplatesValid)
    (hctx : ∀ lane : Fin 4,
      word (context.getD lane 0) = (NightstreamFPrime.Lifecycle.VerifierContext.digest authority).getD lane 0)
    (hlayer1 : ∀ r m, Ensures c m (P ∧ Handoff c authority r m) (layer1 r.1 r.2) R) (hP : P)
    (hval : c.val = (Exec.run c.perm c.stmt c.tape (finalProgram loaded context layer1).program).value)
    (accepts : Exec c.perm (finalProgram loaded context layer1).program c.stmt c.tape) :
    R ((finalProgram loaded context layer1).run {}).1 ((finalProgram loaded context layer1).run {}).2.values := by
  have tail : Ensures c 9 (P ∧ ReadsStatement c)
      (checkedApply loaded context (List.range' 0 9).toArray >>= fun r => layer1 r.1 r.2) R :=
    (checkedApply_ensures loads valid hctx fun h => ⟨rfl, le_rfl, h.2⟩).bind fun r m _ =>
      (hlayer1 r m).pre fun h => ⟨h.1.1, h.2⟩
  obtain ⟨new, hi, -⟩ := tail { instrs := #[] ++ Array.replicate 9 .pub, values := 0 + 9 }
    ⟨List.replicate 9 0, 0, True⟩ rfl (by simp)
  have pre : (finalProgram loaded context layer1).program.toList = List.replicate 9 .pub ++ new := by
    rw [Build.program, finalProgram_eq, terminalPrefix, bind_assoc, StateT.run_bind, statementHandles_run]
    show ((checkedApply loaded context (List.range' 0 9).toArray >>= fun r => layer1 r.1 r.2).run
      { instrs := #[] ++ Array.replicate 9 .pub, values := 0 + 9 }).2.instrs.toList = _
    rw [hi]
    simp
  have reads : ReadsStatement c := fun k => by
    rw [hval]
    exact c.run_pubs pre k.isLt
  exact (finalProgram_ensures loads valid hctx (fun _ => reads) hlayer1).exec hP hval accepts

end Program

end CompressionLean.Circuit.Terminal
