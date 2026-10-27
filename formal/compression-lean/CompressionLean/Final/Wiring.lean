import CompressionLean.Final.Claim
import CompressionLean.Bridge.Relation
import CompressionLean.Circuit.TerminalExec
import CompressionLean.Circuit.PackageRows

/-!
Owns `wiring_of_exec`: an accepting executor run of the final program holds
a terminal assignment whose claim is the claim that layer 1 reads.

- The assignment (`assignment`, `PackageRows.lean`): a package column with a
  system column takes that column's value in the run. A column that no row
  names takes `freeValue`: one, the context digest or the statement word
  that `TerminalFor` asks of it, else zero.
- Rows: the run checks the loaded system (`terminal_exec`), which holds
  exactly when F′'s rows hold (`system_holds_iff`).
- Context and statement: a named public context column is a context
  constant, and a named statement column of the prior state preimage is a
  statement handle (`ColumnValue`). The layout facts place them
  (`sourceToSpartan_prior`, `privateInput_prior`).
- Claim: layer 1 reads the output columns through `claimVars`; `claimOf`
  reads the same columns of the assignment (`chunk_entry`, `pair_entry`).
-/

namespace CompressionLean.Final

open NightstreamFPrime NightstreamFPrime.Export.Package NightstreamFPrime.Export.Stage1 NightstreamFPrime.Layout
open CompressionLean.Circuit CompressionLean.Circuit.Package CompressionLean.Circuit.Terminal
open CompressionLean.Bridge (Fp)

/-! ## The terminal package's layout -/

/-- Every compact invocation of the terminal package names a template. -/
theorem compactTemplatesValid : (Terminal.circuitPackage ()).CompactTemplatesValid := by
  intro invocation member
  simp only [Terminal.circuitPackage, Data.compactRowInvocations, Data.compactRowTemplates,
    PiRLCCombinationInvocations.invocations, List.mem_append, PiRLCCombinationInvocations.commitmentInvocations,
    PiRLCCombinationInvocations.publicInputInvocations, PiRLCCombinationInvocations.evalKInvocations,
    PiRLCCombinationInvocations.evalAInvocations, PiRLCCombinationInvocations.familyInvocations,
    List.mem_flatMap, List.mem_ofFn] at member ⊢
  rcases member with ((⟨s, -, i, rfl⟩ | ⟨s, -, i, rfl⟩) | ⟨s, -, i, rfl⟩) | ⟨s, -, i, rfl⟩
  all_goals exact ⟨_, List.getElem?_eq_getElem
    (PiRLCCombinationTemplates.templateIndex_lt s (Lifecycle.PiRLC.v1_2.CombinationStep.coordinates i).2.1)⟩

/-- A word of the prior state preimage keeps its column. -/
theorem sourceToSpartan_prior {j : ℕ} (small : j < PilotProduction.stateHashWords) :
    Stage1.Spartan.sourceToSpartan j = j := by
  rw [PilotProduction.stateHashWords_eq] at small
  have pilot : PilotSpartan.sourceToSpartan j = j := by
    unfold PilotSpartan.sourceToSpartan
    rw [if_pos (by rw [PilotSpartan.priorPublicStart_value]; omega)]
  unfold Stage1.Spartan.sourceToSpartan Stage1.Spartan.liftPilotColumn
  rw [pilot, if_pos (by norm_num [Stage1.Spartan.pilotSourceColumnCount]; omega),
    if_pos (by norm_num [Stage1.Spartan.pilotInputPrivateColumnCount]; omega)]

theorem foldl_segments_prefix (rest : List Segment) (out : Array (Segment × ℕ)) (offset : ℕ) :
    ∃ more : List (Segment × ℕ), (rest.foldl (fun (out, offset) s => (out.push (s, offset), offset + s.length))
      (out, offset)).1.toList = out.toList ++ more := by
  induction rest generalizing out offset with
  | nil => exact ⟨[], by simp⟩
  | cons s rest ih =>
      obtain ⟨more, h⟩ := ih (out.push (s, offset)) (offset + s.length)
      exact ⟨(s, offset) :: more, by rw [List.foldl_cons, h]; simp⟩

/-- The prior state preimage is the first caller input, from column zero. -/
theorem privateInput_prior {c : ℕ} (small : c < PilotProduction.stateHashWords) :
    privateInput (callerSegments (Terminal.circuitPackage ()).layout) c = some c := by
  have first : (Terminal.circuitPackage ()).layout.privateSegments =
      ⟨Data.Role.priorPreimage, 0, PilotValues.stateHashWords⟩ ::
        ((Data.privateSegments).drop 1 ++ [⟨PerApplicationPackage.Role.applicationWitness,
          Data.physicalLayout.constantColumn, 0⟩, ⟨PerApplicationPackage.Role.applicationLocal,
          Data.physicalLayout.constantColumn, 0⟩]) := rfl
  unfold callerSegments
  rw [first, List.filter_cons_of_pos (by decide), List.foldl_cons]
  obtain ⟨more, h⟩ := foldl_segments_prefix _ (#[].push (⟨Data.Role.priorPreimage, 0, PilotValues.stateHashWords⟩, 0)) _
  rw [h]
  unfold privateInput
  rw [PilotProduction.stateHashWords_eq] at small
  have length : PilotValues.stateHashWords = 27819 := rfl
  simp [length, small]

/-! ## Reading the output columns -/

theorem toF_getElem! (b : Array Replay.Gl) (i : ℕ) : b[i]!.toF = ((b[i]?).map Replay.Gl.toF).getD 0 := by
  rw [getElem!_def]
  cases b[i]? with
  | none => exact Replay.Gl.toF_zero
  | some _ => rfl

theorem chunks_getElem? (n : ℕ) (a : Array Var) (ρ : ℕ) :
    (Terminal.chunks n a)[ρ]? = if ρ < a.size / n then some (a.extract (n * ρ) (n * ρ + n)) else none := by
  simp only [Terminal.chunks, Array.getElem?_map, Array.getElem?_range]
  split <;> rfl

/-- Entry `l` of row `ρ` of `a` in rows of `n`: entry `n ρ + l` of `a`. -/
theorem chunk_entry (a : Array Var) (f : Var → Replay.Gl) (n ρ l : ℕ) (dvd : n ∣ a.size) :
    (((Terminal.chunks n a).map (·.map f))[ρ]!)[l]!.toF =
      if l < n then ((a[n * ρ + l]?).map fun h => (f h).toF).getD 0 else 0 := by
  obtain ⟨rows, size⟩ := dvd
  rw [toF_getElem!, getElem!_def, Array.getElem?_map, chunks_getElem?]
  by_cases hρ : ρ < a.size / n
  · rw [if_pos hρ]
    have hn : 0 < n := Nat.pos_of_ne_zero fun zero => by simp [zero] at hρ
    have fits : n * ρ + n ≤ a.size := by
      rw [size, Nat.mul_div_cancel_left _ hn] at hρ
      have := Nat.mul_le_mul_left n (show ρ + 1 ≤ rows by omega)
      rw [size]
      linarith [Nat.mul_add n ρ 1]
    simp only [Option.map_some, Array.getElem?_map, Array.getElem?_extract]
    split_ifs with low high <;> simp_all
    rfl
  · rw [if_neg hρ]
    show (Option.map Replay.Gl.toF (#[] : Array Replay.Gl)[l]?).getD 0 = _
    split_ifs with low
    · have out : a[n * ρ + l]? = none := Array.getElem?_eq_none (by
        have hn : 0 < n := by omega
        rw [size, Nat.mul_div_cancel_left _ hn] at hρ
        have := Nat.mul_le_mul_left n (show rows ≤ ρ by omega)
        rw [size]
        omega)
      simp [out]
    · simp

/-- Pair `k` of `a`, mapped by `f`: entries `2k` and `2k + 1` of `a`. -/
theorem pair_entry (a : Array Var) (f : Var → Replay.Gl) (k : ℕ) (even : 2 ∣ a.size) :
    (((pairs a).map fun p => (f p.1, f p.2))[k]!).1.toF = ((a[2 * k]?).map fun h => (f h).toF).getD 0 ∧
      (((pairs a).map fun p => (f p.1, f p.2))[k]!).2.toF = ((a[2 * k + 1]?).map fun h => (f h).toF).getD 0 := by
  obtain ⟨half, size⟩ := even
  rw [getElem!_def, Array.getElem?_map, pairs, Array.getElem?_map, chunks_getElem?]
  by_cases hk : k < a.size / 2
  · have below : 2 * k + 1 < a.size := by omega
    simp only [if_pos hk, Option.map_some, Array.getD_eq_getD_getElem?, Array.getElem?_extract,
      Array.getElem?_eq_getElem below, Array.getElem?_eq_getElem (show 2 * k < a.size by omega)]
    simp [show 0 < min (2 * k + 2) a.size - 2 * k by omega, show 1 < min (2 * k + 2) a.size - 2 * k by omega,
      Array.getElem?_eq_getElem (show 2 * k < a.size by omega)]
  · rw [if_neg hk, Option.map_none, Option.map_none,
      Array.getElem?_eq_none (show a.size ≤ 2 * k by omega),
      Array.getElem?_eq_none (show a.size ≤ 2 * k + 1 by omega)]
    exact ⟨Replay.Gl.toF_zero, Replay.Gl.toF_zero⟩

/-- An output list that `load` looks up reads, on an assignment, the values
of its system columns. -/
theorem word_of_columns {layout : PhysicalLayout} {index : Array (Option ℕ)} {list : List Circuit.Expr}
    {ks : Array ℕ} (h : outputColumns layout index list = .ok ks) (env : Circuit.Env) (value : ℕ → Fp)
    (named : ∀ c k, c < layout.totalColumnCount → index[c]! = some k → (env c : Fp) = value k) (i : ℕ) :
    (word list env i : Fp) = ((ks[i]?).map value).getD 0 := by
  obtain ⟨size, each⟩ := outputColumns_ok h
  by_cases hi : i < list.length
  · obtain ⟨c, k, var, below, found, at_i⟩ := each i hi
    rw [word, List.getD_eq_getElem?_getD, List.getElem?_map, List.getElem?_eq_getElem hi, var, at_i]
    simp only [Option.map_some, Option.getD_some, Circuit.Expr.eval]
    exact (ZMod.natCast_zmod_val (n := Spec.goldilocksModulus) (env c)).trans (named c k below found)
  · rw [word, List.getD_eq_getElem?_getD, List.getElem?_map, List.getElem?_eq_none (by omega),
      Array.getElem?_eq_none (by omega)]
    rfl

/-- Pair `l` of row `j` of `a` in rows of `2 m`: entries `2 m j + 2 l` and
`2 m j + 2 l + 1` of `a`. -/
theorem chunkPair_entry (a : Array Var) (f : Var → Replay.Gl) (m j l : ℕ) (dvd : 2 * m ∣ a.size) :
    ((((Terminal.chunks (2 * m) a).map pairs).map (·.map fun p => (f p.1, f p.2)))[j]!)[l]!.1.toF =
        (if l < m then ((a[2 * m * j + 2 * l]?).map fun h => (f h).toF).getD 0 else 0) ∧
      ((((Terminal.chunks (2 * m) a).map pairs).map (·.map fun p => (f p.1, f p.2)))[j]!)[l]!.2.toF =
        (if l < m then ((a[2 * m * j + 2 * l + 1]?).map fun h => (f h).toF).getD 0 else 0) := by
  obtain ⟨rows, size⟩ := dvd
  rw [getElem!_def (c := ((Terminal.chunks (2 * m) a).map pairs).map _), Array.getElem?_map, Array.getElem?_map,
    chunks_getElem?]
  by_cases hj : j < a.size / (2 * m)
  · have hm : 0 < m := Nat.pos_of_ne_zero fun zero => by simp [zero] at hj
    have fits : 2 * m * j + 2 * m ≤ a.size := by
      rw [size, Nat.mul_div_cancel_left _ (by omega)] at hj
      have := Nat.mul_le_mul_left (2 * m) (show j + 1 ≤ rows by omega)
      rw [size]
      linarith [Nat.mul_add (2 * m) j 1]
    simp only [if_pos hj, Option.map_some]
    have row : (a.extract (2 * m * j) (2 * m * j + 2 * m)).size = 2 * m := by
      rw [Array.size_extract]
      omega
    obtain ⟨first, second⟩ := pair_entry (a.extract (2 * m * j) (2 * m * j + 2 * m)) f l ⟨m, by rw [row]⟩
    refine ⟨first.trans ?_, second.trans ?_⟩ <;> rw [Array.getElem?_extract] <;> split_ifs <;>
      first | rfl | omega
  · simp only [if_neg hj, Option.map_none]
    have empty : ∀ i, (#[] : Array (Replay.Gl × Replay.Gl))[i]! = default := fun i =>
      getElem!_neg _ _ (Nat.not_lt_zero i)
    show ((#[] : Array (Replay.Gl × Replay.Gl))[l]!).1.toF = _ ∧ ((#[] : Array (Replay.Gl × Replay.Gl))[l]!).2.toF = _
    rw [empty]
    constructor <;> split_ifs <;> first | exact Replay.Gl.toF_zero | skip
    all_goals
      rw [Array.getElem?_eq_none]
      · exact Replay.Gl.toF_zero
      · rcases Nat.eq_zero_or_pos m with zero | pos
        · omega
        · rw [size, Nat.mul_div_cancel_left _ (by omega)] at hj
          have := Nat.mul_le_mul_left (2 * m) (show rows ≤ j by omega)
          rw [size]
          omega

/-! ## The columns that `TerminalFor` reads -/

theorem constantColumn_lt : Stage1.Spartan.constantColumn < Stage1.Spartan.spartanColumnCount := by
  rw [Stage1.Spartan.spartanColumnCount_decomposition]
  unfold Stage1.Spartan.constantColumn
  omega

theorem privateColumn_lt {c : ℕ} (private_ : c < Stage1.Spartan.privateColumnCount) :
    c < Stage1.Spartan.spartanColumnCount := by
  rw [Stage1.Spartan.spartanColumnCount_decomposition]
  omega

/-- Public context word `lane`: after the constant column, public input
`publicContextStart + lane`. -/
theorem contextColumn (lane : Fin 4) :
    Stage1.Spartan.privateColumnCount < Stage1.Spartan.expectedContextPublicStart + lane ∧
      Stage1.Spartan.expectedContextPublicStart + lane < Stage1.Spartan.spartanColumnCount ∧
      Stage1.Spartan.expectedContextPublicStart + lane - Stage1.Spartan.privateColumnCount - 1 =
        publicContextStart + lane := by
  have lt := lane.isLt
  rw [Stage1.Spartan.spartanColumnCount_decomposition, Stage1.Spartan.publicColumnCount_eq]
  unfold Stage1.Spartan.expectedContextPublicStart Stage1.Spartan.pilotPublicColumnCount publicContextStart
  omega

/-- Statement word `k` of the prior state preimage: a private column of the
preimage, before the public columns. -/
theorem statementColumn (k : Fin 9) :
    contextStart + 4 + k < Stage1.Spartan.privateColumnCount ∧
      contextStart + 4 + k < PilotProduction.stateHashWords ∧
      ¬ Stage1.Spartan.expectedContextPublicStart ≤ contextStart + 4 + k := by
  have lt := k.isLt
  have pilot := Stage1.Spartan.pilotPrivateColumnCount_le_constantColumn
  have pilotValue : Stage1.Spartan.pilotPrivateColumnCount = 5141486 := rfl
  unfold Stage1.Spartan.constantColumn at pilot
  rw [contextStart_eq, PilotProduction.stateHashWords_eq]
  unfold Stage1.Spartan.expectedContextPublicStart
  omega

theorem ofFn_getD {n : ℕ} (x : Fin n → Spec.F) (k : Fin n) : (List.ofFn x).getD k 0 = x k := by
  rw [List.getD_eq_getElem?_getD, List.getElem?_ofFn]
  simp

/-! ## The terminal assignment -/

/-- The value of a package column that no row names: one at the constant
column, the context digest at the public context columns, the statement at
the end of the prior state preimage, else zero. -/
def freeValue (authority : Lifecycle.VerifierContext.Authority) (x : StateWords) (c : ℕ) : Spec.F :=
  if c = Stage1.Spartan.constantColumn then 1
  else if Stage1.Spartan.expectedContextPublicStart ≤ c then
    (Lifecycle.VerifierContext.digest authority).getD (c - Stage1.Spartan.expectedContextPublicStart) 0
  else if contextStart + 4 ≤ c then (List.ofFn x).getD (c - contextStart - 4) 0
  else 0

/-- The context words of the final program: the authority's digest. -/
def contextWords (authority : Lifecycle.VerifierContext.Authority) : Array Replay.Gl :=
  ((Lifecycle.VerifierContext.digest authority).map Package.word).toArray

/-- **The terminal wiring of an accepting run.** If the executor accepts the
final program on the statement `x`, then the package assignment of the
run's system values is a terminal assignment for `x`, and the statement that
layer 1 reads from the run is its claim. -/
theorem wiring_of_exec (package : Loaded)
    (loads : Package.load (Terminal.circuitPackage ()) Terminal.outputs = .ok package)
    (authority : Lifecycle.VerifierContext.Authority) (key : Layer1Input) (x : StateWords)
    (perm : List Fp → List Fp) (tape : ℕ → Fp) (L : Array Instr)
    (program : (finalViews package (contextWords authority) key).program = L)
    (accepts : Exec perm L (List.ofFn x) tape) :
    let v := fun h => Package.word ((Exec.run perm (List.ofFn x) tape L).value h)
    let claim := ((finalViews package (contextWords authority) key).run {}).1.claim
    ∃ env, TerminalFor authority x env ∧
      Bridge.statementOf ⟨claim.commitment.map (·.map v), claim.publicInput.map (·.map v),
        claim.point.map (fun p => (v p.1, v p.2)), claim.evalK.map (fun p => (v p.1, v p.2)),
        claim.evalA.map (·.map fun p => (v p.1, v p.2))⟩ = claimOf env := by
  intro v claim
  obtain ⟨system, sources, computedColumns, commitment, publicInput, point, evalK, evalA, -⟩ := load_ok loads
  have numbered := numbering_numbered (Terminal.circuitPackage ())
  have computedOk : ∀ s ∈ package.sources, ∀ k, s = .computed k → k < package.computed.size := by
    intro s member k e
    obtain ⟨j, hj, rfl⟩ := Array.mem_iff_getElem.mp member
    rw [computedColumns]
    simp only [sources] at hj e
    exact numbered.computed j hj k e
  obtain ⟨columns, claimEq, values, rows, nonzero⟩ :=
    terminal_exec package (contextWords authority) key perm (List.ofFn x) tape computedOk L program accepts
  let val := (Exec.run perm (List.ofFn x) tape L).value
  let index := (numbering (Terminal.circuitPackage ())).1
  let env := assignment index (fun j => (columns.toList.map val).getD j 0) (freeValue authority x)
  have length : columns.size = package.sources.size := by simpa using values.length_eq.symm
  have indexSize : index.size = Stage1.Spartan.spartanColumnCount := numbered.size
  -- A named column: the value of its system column, and what its source gives.
  have named : ∀ c k, index[c]? = some (some k) → ∃ hk : k < package.sources.size,
      env c = val (columns[k]'(length ▸ hk)) ∧
      SourceOf (Terminal.circuitPackage ()).layout (callerSegments (Terminal.circuitPackage ()).layout) c
        package.sources[k] ∧
      ColumnValue val (List.ofFn x) (contextWords authority) package.sources[k] (columns[k]'(length ▸ hk)) := by
    intro c k hk
    obtain ⟨klt, source⟩ := numbered.source c k hk
    have klt' : k < package.sources.size := by rw [sources]; exact klt
    have same : package.sources[k] = (numbering (Terminal.circuitPackage ())).2.1[k] := by simp [sources]
    refine ⟨klt', ?_, same ▸ source, ?_⟩
    · have bang : index[c]! = some k := by rw [getElem!_def, hk]
      show (if index[c]? = some none then _ else (List.map val columns.toList).getD ((index[c]!).getD 0) 0) = _
      rw [if_neg (by rw [hk]; simp), bang, Option.getD_some, List.getD_eq_getElem?_getD, List.getElem?_map,
        Array.getElem?_toList, Array.getElem?_eq_getElem (length ▸ klt')]
      rfl
    · have each := (List.forall₂_iff_get.mp values).2 k (by simpa using klt') (by simpa [length] using klt')
      simpa [val] using each
  have unnamed : ∀ c, index[c]? = some none → env c = freeValue authority x c := fun c h => by
    show (if index[c]? = some none then _ else _) = _
    rw [if_pos h]
  have numberedColumn : ∀ c, c < Stage1.Spartan.spartanColumnCount →
      index[c]? = some none ∨ ∃ k, index[c]? = some (some k) := fun c below => by
    obtain ⟨o, ho⟩ : ∃ o, index[c]? = some o := ⟨_, Array.getElem?_eq_getElem (indexSize ▸ below)⟩
    rw [ho]
    rcases o with _ | k
    · exact .inl rfl
    · exact .inr ⟨k, rfl⟩
  have constant : Stage1.Spartan.constantColumn = (Terminal.circuitPackage ()).layout.privateColumnCount := rfl
  -- The constant column holds one.
  have one : env Stage1.Spartan.constantColumn = 1 := by
    rcases numberedColumn _ constantColumn_lt with h | ⟨k, h⟩
    · rw [unnamed _ h]
      simp [freeValue]
    · obtain ⟨hk, value, source, column⟩ := named _ k h
      rw [value]
      generalize package.sources[k] = s at source column
      cases s with
      | one => exact column
      | pub _ => exact absurd source.1 (by rw [← constant]; exact Nat.lt_irrefl _)
      | priv _ => exact absurd source.1 (by rw [← constant]; exact Nat.lt_irrefl _)
      | computed _ => exact absurd source.1 (by rw [← constant]; exact Nat.lt_irrefl _)
  have rowsHold : (Terminal.circuitPackage ()).RowsHold env :=
    (system_holds_iff loads compactTemplatesValid _ (freeValue authority x) one).mp fun r member =>
      rows r (Array.mem_toList_iff.mpr member)
  -- The public context columns hold the digest.
  have digestWord : ∀ lane : Fin 4, (contextWords authority).getD lane 0 =
      Package.word ((Lifecycle.VerifierContext.digest authority).getD lane 0) := fun lane => by
    simp only [contextWords, Array.getD_eq_getD_getElem?, List.getElem?_toArray, List.getElem?_map,
      List.getD_eq_getElem?_getD]
    cases (Lifecycle.VerifierContext.digest authority)[lane.val]? <;> rfl
  have context : Export.Stage1.Package.SelectedVerifierContext authority env := by
    intro lane
    obtain ⟨after, below, index'⟩ := contextColumn lane
    rcases numberedColumn _ below with h | ⟨k, h⟩
    · rw [unnamed _ h]
      simp only [freeValue]
      rw [if_neg (show _ ≠ Stage1.Spartan.constantColumn from Nat.ne_of_gt after), if_pos (Nat.le_add_right _ _),
        Nat.add_sub_cancel_left]
    · obtain ⟨hk, value, source, column⟩ := named _ k h
      rw [value]
      generalize package.sources[k] = s at source column
      cases s with
      | pub i =>
          have hi : i = publicContextStart + lane.val := source.2.trans index'
          subst hi
          rw [column (Nat.le_add_right _ _), Nat.add_sub_cancel_left, digestWord lane]
          exact word_ofF _
      | one => exact absurd source (Nat.ne_of_gt after)
      | priv _ => exact absurd source.1 (Nat.lt_asymm after)
      | computed _ => exact absurd source.1 (Nat.lt_asymm after)
  -- The statement words of the prior state preimage are `x`.
  have statement : priorStatement env = x := by
    funext k
    obtain ⟨private_, small, before⟩ := statementColumn k
    have words : contextStart = Lifecycle.PiCCS.v1_2.StateBinding.contextWordStart := contextStart_eq
    show env (Stage1.Spartan.sourceToSpartan (PilotProduction.priorPreimageStart +
      (Lifecycle.PiCCS.v1_2.StateBinding.contextWordStart + 4 + k.val))) = x k
    rw [show PilotProduction.priorPreimageStart = 0 from rfl, Nat.zero_add, ← words, sourceToSpartan_prior small]
    have offset : contextStart + 4 + k.val - contextStart - 4 = k.val := by
      rw [Nat.add_assoc, Nat.add_sub_cancel_left, Nat.add_sub_cancel_left]
    rcases numberedColumn _ (privateColumn_lt private_) with h | ⟨kk, h⟩
    · rw [unnamed _ h]
      simp only [freeValue]
      rw [if_neg (show _ ≠ Stage1.Spartan.constantColumn from Nat.ne_of_lt private_), if_neg before,
        if_pos (Nat.le_add_right _ _), offset, ofFn_getD]
    · obtain ⟨hk, value, source, column⟩ := named _ kk h
      rw [value]
      generalize package.sources[kk] = s at source column
      cases s with
      | priv i =>
          obtain ⟨-, input⟩ := source
          rw [privateInput_prior small] at input
          cases input
          rw [column (Nat.le_add_right _ _) small, offset]
          exact ofFn_getD x k
      | computed _ =>
          have none := source.2
          rw [privateInput_prior small] at none
          cases none
      | one => exact absurd source (Nat.ne_of_lt private_)
      | pub _ => exact absurd source.1 (Nat.lt_asymm private_)
  refine ⟨env, ⟨rowsHold, context, statement, fun zero => nonzero ((ofFn_getD x 0).trans zero)⟩, ?_⟩
  -- The claim: each output list reads its system columns.
  have readNamed : ∀ c k, c < (Terminal.circuitPackage ()).layout.totalColumnCount → index[c]! = some k →
      (env c : Fp) = val (columns.getD k 0) := by
    intro c k below found
    have inside : c < index.size := by rw [indexSize]; exact below
    have h : index[c]? = some (some k) := by
      rw [Array.getElem?_eq_getElem inside, ← found, getElem!_pos index c inside]
    obtain ⟨hk, value, -, -⟩ := named c k h
    rw [value, Array.getD_eq_getD_getElem?, Array.getElem?_eq_getElem (length ▸ hk)]
    rfl
  have toF_v : ∀ h, (v h).toF = val h := fun h => by
    show (Replay.Gl.ofNat (val h).val).toF = _
    rw [Replay.Gl.toF_ofNat]
    exact ZMod.natCast_zmod_val _
  have flat : ∀ {list : List Circuit.Expr} {ks : Array ℕ},
      outputColumns (Terminal.circuitPackage ()).layout index list = .ok ks → ∀ i,
        (((ks.map fun k => columns.getD k 0)[i]?).map fun h => (v h).toF).getD 0 = word list env i := by
    intro list ks h i
    rw [word_of_columns h env _ readNamed i, Array.getElem?_map, Option.map_map]
    cases ks[i]? with
    | none => rfl
    | some k => exact toF_v _
  have claimIs : claim = claimView (claimVars package.outputs columns) := claimEq
  rw [claimIs]
  sorry

end CompressionLean.Final
