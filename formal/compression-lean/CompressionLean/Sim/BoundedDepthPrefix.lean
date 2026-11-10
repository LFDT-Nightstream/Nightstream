import CompressionLean.Sim.Statement
import CompressionLean.Sim.Merkle
import CompressionLean.Lowering.PackageRows
import CompressionLean.Sim.BoundedDepthColumns

/-! The bounded-depth scheme's final program before layer 1 (CW-6.5): an accepted run gives the finish columns an
environment in which every check of the program holds (`Spec.BoundedDepth.FinishTerminal`). The
environment is the value of its cell on a live column and 0 on every other column (`finishEnv`). A lane's
linear form evaluates to the lane (`linearForm_eval`), so each block's checks give its outputs as
the permutation of its lanes (`checkBlock_ensures`); the R1CS block gives every row of the
package's pieces (`Lowering.Package.row_holds`). The package must fit its columns (`Fits`). -/

namespace CompressionLean.Sim.BoundedDepth

open CompressionLean.Circuit CompressionLean.BoundedDepth CompressionLean.Lowering
open CompressionLean.Replay (Word)
open CompressionLean.Spec.BoundedDepth
open NightstreamFPrime.Circuit (Expr Env)
open NightstreamFPrime.Export.BoundedDepth.StepPackage (Package PackagePiece)
open NightstreamFPrime.Gadgets.Poseidon2

abbrev F := NightstreamFPrime.Spec.F

/-! ### Lanes -/

theorem scaleForm_value (env : Env) (factor : F) (form : F × List (ℕ × F)) :
    (scaleForm factor form).1 + ((scaleForm factor form).2.map fun t => t.2 * env t.1).sum =
      factor * (form.1 + (form.2.map fun t => t.2 * env t.1).sum) := by
  obtain ⟨constant, terms⟩ := form
  simp only [scaleForm, List.map_map, Function.comp_def]
  induction terms with
  | nil => simp; ring
  | cons t rest ih =>
      simp only [List.map_cons, List.sum_cons] at ih ⊢
      linear_combination ih

theorem linearForm_eval (env : Env) :
    ∀ (e : Expr) (form : F × List (ℕ × F)), linearForm e = some form →
      e.eval env = form.1 + (form.2.map fun t => t.2 * env t.1).sum
  | .var column, form, h => by
      simp only [linearForm, Option.some.injEq] at h
      subst h
      simp [Expr.eval]
  | .const value, form, h => by
      simp only [linearForm, Option.some.injEq] at h
      subst h
      simp [Expr.eval]
  | .add left right, form, h => by
      simp only [linearForm, Option.bind_eq_bind, Option.bind_eq_some_iff, Option.pure_def,
        Option.some.injEq] at h
      obtain ⟨⟨a, s⟩, ha, ⟨b, t⟩, hb, rfl⟩ := h
      rw [Expr.eval, linearForm_eval env left _ ha, linearForm_eval env right _ hb]
      simp only [List.map_append, List.sum_append]
      ring
  | .mul left right, form, h => by
      match left, right, h with
      | .const factor, right, h =>
          rw [show linearForm (.mul (.const factor) right) = scaleForm factor <$> linearForm right
            from rfl, Option.map_eq_map, Option.map_eq_some_iff] at h
          obtain ⟨inner, hinner, rfl⟩ := h
          rw [Expr.eval, linearForm_eval env right _ hinner, scaleForm_value]
          rfl
      | .var a, .const factor, h =>
          rw [show linearForm (.mul (.var a) (.const factor)) = scaleForm factor <$> linearForm (.var a)
            from rfl, Option.map_eq_map, Option.map_eq_some_iff] at h
          obtain ⟨inner, hinner, rfl⟩ := h
          rw [Expr.eval, linearForm_eval env _ _ hinner, scaleForm_value, Expr.eval]
          ring
      | .add a b, .const factor, h =>
          rw [show linearForm (.mul (.add a b) (.const factor)) = scaleForm factor <$> linearForm (.add a b)
            from rfl, Option.map_eq_map, Option.map_eq_some_iff] at h
          obtain ⟨inner, hinner, rfl⟩ := h
          rw [Expr.eval, linearForm_eval env _ _ hinner, scaleForm_value, Expr.eval]
          ring
      | .mul a b, .const factor, h =>
          rw [show linearForm (.mul (.mul a b) (.const factor)) = scaleForm factor <$> linearForm (.mul a b)
            from rfl, Option.map_eq_map, Option.map_eq_some_iff] at h
          obtain ⟨inner, hinner, rfl⟩ := h
          rw [Expr.eval, linearForm_eval env _ _ hinner, scaleForm_value, Expr.eval]
          ring
      | .var _, .var _, h | .var _, .add _ _, h | .var _, .mul _ _, h
      | .add _ _, .var _, h | .add _ _, .add _ _, h | .add _ _, .mul _ _, h
      | .mul _ _, .var _, h | .mul _ _, .add _ _, h | .mul _ _, .mul _ _, h =>
          simp [linearForm] at h

/-! ### Loops -/

variable {c : Ctx F} {δ : Type}

/-- Each item's facts, from an item that ensures them at any later count. -/
theorem forM_ensures {f : δ → Build PUnit} (P : δ → Prop) (h' : ∀ x, Appends (f x)) :
    ∀ (xs : List δ) (n : ℕ), (∀ x ∈ xs, ∀ m, n ≤ m → Ensures c m (f x) fun _ _ => P x) →
      Ensures c n (xs.forM f) fun _ _ => ∀ x ∈ xs, P x
  | [], _, _ => Ensures.pure fun _ member => absurd member (List.not_mem_nil)
  | x :: xs, n, h => by
      show Ensures c n (f x >>= fun _ => xs.forM f) _
      refine (h x List.mem_cons_self n le_rfl).bind (fun _ m le hx => ?_) (h' x)
        fun _ => Appends.forM_list h' xs
      refine (forM_ensures P h' xs m fun y member k le' =>
        h y (List.mem_cons_of_mem _ member) k (le.trans le')).post fun _ _ rest y member => ?_
      rcases List.mem_cons.mp member with rfl | inside
      · exact hx
      · exact rest y inside

/-- Each item's result, from an item that ensures its facts at any later count; the facts grow
with the count. -/
theorem mapM_ensures {f : δ → Build Var} (P : δ → Var → ℕ → Prop) (h' : ∀ x, Appends (f x))
    (mono : ∀ x r m m', m ≤ m' → P x r m → P x r m') :
    ∀ (xs : List δ) (n : ℕ), (∀ x ∈ xs, ∀ m, n ≤ m → Ensures c m (f x) (P x)) →
      Ensures c n (xs.mapM f) fun rs m => rs.length = xs.length ∧
        ∀ i (inside : i < xs.length), P xs[i] (rs.getD i 0) m
  | [], _, _ => Ensures.pure ⟨rfl, fun _ inside => absurd inside (Nat.not_lt_zero _)⟩
  | x :: xs, n, h => by
      rw [List.mapM_cons]
      refine (h x List.mem_cons_self n le_rfl).bind (fun r m le hr => ?_) (h' x) fun _ =>
        Appends.bind (Appends.mapM_list h' xs) fun _ => Appends.pure _
      refine (mapM_ensures P h' mono xs m fun y member k le' =>
        h y (List.mem_cons_of_mem _ member) k (le.trans le')).bind
          (fun rs m' le' hrs => Ensures.pure ⟨by simp [hrs.1], fun i inside => ?_⟩)
          (Appends.mapM_list h' xs) fun _ => Appends.pure _
      cases i with
      | zero => exact mono x r m m' le' hr
      | succ i => exact hrs.2 i (by simpa using inside)

/-! ### Columns and lanes -/

theorem cast_ofF (x : F) : (Word.cast (Word.ofF x) : F) = x := Word.cast_ofF x

/-- The handle of column `k` holds `env k` before `n`. -/
def ColumnIs (c : Ctx F) (column : ℕ → Var) (env : Env) (n k : ℕ) : Prop :=
  column k < n ∧ c.val (column k) = env k

theorem ColumnIs.mono {column : ℕ → Var} {env : Env} {n m k : ℕ} (h : ColumnIs c column env n k)
    (le : n ≤ m) : ColumnIs c column env m k :=
  ⟨lt_of_lt_of_le h.1 le, h.2⟩

theorem ColumnIs.word {column : ℕ → Var} {env : Env} {n k : ℕ} (h : ColumnIs c column env n k) :
    IsWord c n (column k) (Word.ofF (env k)) :=
  ⟨h.1, h.2.trans (cast_ofF _).symm⟩

theorem ofF_cast {n : ℕ} {h : Var} {x : F} {g : Word} (word : IsWord c n h g) (value : Word.cast g = x) :
    IsWord c n h (Word.ofF x) :=
  word.congr (value.trans (cast_ofF x).symm)

private theorem laneFold_ensures (column : ℕ → Var) (env : Env) :
    ∀ (terms : List (ℕ × F)) (start : Var) (value : F) (n : ℕ),
      (∀ t ∈ terms, ColumnIs c column env n t.1) → IsWord c n start (Word.ofF value) →
      Ensures c n (terms.foldlM (fun sum term => do
          let scaled ← Build.scale (column term.1) (Word.ofF term.2)
          Build.add sum scaled) start)
        fun r m => IsWord c m r (Word.ofF (value + (terms.map fun t => t.2 * env t.1).sum))
  | [], start, value, _, _, hstart => Ensures.pure (by simpa using hstart)
  | t :: rest, start, value, n, columns, hstart => by
      rw [List.foldlM_cons]
      refine ((scale_ensures (Word.ofF t.2) (columns t List.mem_cons_self).word).bind
        (fun scaled m le hscaled => add_ensures (hstart.mono le) hscaled)).bind
          (fun r m le hr => ?_) (hf' := fun _ => Appends.foldlM_list (fun _ _ => by appends) _ _)
      refine (laneFold_ensures column env rest r (value + t.2 * env t.1) m
        (fun u member => (columns u (List.mem_cons_of_mem _ member)).mono le) (ofF_cast hr ?_)).post
          fun _ _ h => by simpa [add_assoc] using h
      rw [word_add, word_mul, cast_ofF, cast_ofF, cast_ofF]
      ring

instance laneValue_appends (column : ℕ → Var) (lane : Expr) : Appends (BoundedDepth.laneValue column lane) := by
  unfold BoundedDepth.laneValue
  exact Appends.bind (by appends) fun _ => Appends.foldlM_list (fun _ _ => by appends) _ _

/-- Handle `r` holds the value of `lane` before `m`. -/
def LaneIs (c : Ctx F) (env : Env) (lane : Expr) (r : Var) (m : ℕ) : Prop :=
  IsWord c m r (Word.ofF (lane.eval env))

theorem LaneIs.mono {env : Env} {lane : Expr} {r : Var} {m m' : ℕ} (h : LaneIs c env lane r m) (le : m ≤ m') :
    LaneIs c env lane r m' :=
  IsWord.mono h le

theorem laneValue_ensures {column : ℕ → Var} {env : Env} {n : ℕ} {lane : Expr} {form : F × List (ℕ × F)}
    (h : linearForm lane = some form) (columns : ∀ t ∈ form.2, ColumnIs c column env n t.1) :
    Ensures c n (BoundedDepth.laneValue column lane) (LaneIs c env lane) := by
  unfold BoundedDepth.laneValue
  rw [h, Option.getD_some]
  refine (const_ensures _).bind (fun start m le hstart => ?_)
    (hf' := fun _ => Appends.foldlM_list (fun _ _ => by appends) _ _)
  refine (laneFold_ensures column env form.2 start form.1 m (fun t member => (columns t member).mono le)
    hstart).post fun _ _ r => ?_
  unfold LaneIs
  rwa [linearForm_eval env lane form h]

/-! ### Blocks -/

theorem permute_runF (state : Fin 16 → F) :
    NightstreamFPrime.Spec.Poseidon2.permute (List.ofFn state) = List.ofFn (Permutation.runF Permutation.schedule state) :=
  (NightstreamFPrime.Gadgets.Poseidon2.Permutation.runReference_schedule _).symm.trans
    (NightstreamFPrime.Gadgets.Poseidon2.Permutation.runF_eq_reference _ _).symm

/-- A block's outputs are the permutation of its lanes, when its lanes' terms and its outputs are
finish columns. -/
theorem checkBlock_ensures (hperm : c.perm = NightstreamFPrime.Spec.Poseidon2.permute) {column : ℕ → Var}
    {env : Env} {n : ℕ} {block : NightstreamFPrime.Export.Generic.Frame.Block} (lanes : block.lanes.length = 16)
    (forms : ∀ lane ∈ block.lanes, ∃ form, linearForm lane = some form ∧ ∀ t ∈ form.2, ColumnIs c column env n t.1)
    (outputs : ∀ lane : Fin 16, ColumnIs c column env n (block.start + 1080 + lane)) :
    Ensures c n (checkBlock column block) fun _ _ =>
      (fun lane : Fin 16 => env (block.start + 1080 + lane)) =
        Permutation.runF Permutation.schedule fun lane => (block.lanes.getD lane 0).eval env := by
  let words : Array Word := Array.ofFn fun lane : Fin 16 => Word.ofF ((block.lanes.getD lane 0).eval env)
  have wordAt : ∀ k : Fin 16, words[k]! = Word.ofF ((block.lanes.getD k 0).eval env) := fun k => by
    simp [words]
  have permutes : Permutes c.perm := by rw [hperm]; exact permutes_poseidon2
  unfold checkBlock
  refine (mapM_ensures (LaneIs c env) (fun _ => by appends)
    (fun _ _ _ _ le h => h.mono le) block.lanes n fun lane member m le => ?_).bind
      (fun rs m le hrs => ?_) (hf' := fun _ => Appends.bind (by appends) fun _ =>
        Appends.forM_list (fun _ => by appends) _)
  · obtain ⟨form, hform, columns⟩ := forms lane member
    exact laneValue_ensures hform fun t inside => (columns t inside).mono le
  · have state : ∀ k : Fin 16, IsWord c m (rs.getD k 0) words[k]! := fun k => by
      have inside : k.val < block.lanes.length := by rw [lanes]; exact k.isLt
      rw [wordAt k, List.getD_eq_getElem _ _ inside]
      exact hrs.2 k inside
    refine (permute_ensures permutes state).bind (fun out m' le' hout => ?_)
      (hf' := fun _ => Appends.forM_list (fun _ => by appends) _)
    refine (forM_ensures (fun lane : Fin 16 => env (block.start + 1080 + lane) =
        Word.cast (Replay.permute words)[lane]!) (fun _ => by appends) (List.finRange 16) m'
      fun lane _ k le'' => (assertEqual_ensures ((outputs lane).mono (by omega)).word
        ((hout lane).mono le'')).post fun _ _ h => ?_).post fun _ _ each => ?_
    · rw [← cast_ofF (env _), h]
    · funext lane
      have cast : (fun i : Fin 16 => (Word.cast words[i]! : F)) =
          fun i : Fin 16 => (block.lanes.getD i 0).eval env := by
        funext i
        rw [wordAt i]
        exact cast_ofF _
      rw [each lane (List.mem_finRange lane)]
      calc (Word.cast (Replay.permute words)[lane]! : F)
          = (NightstreamFPrime.Spec.Poseidon2.permute
              (List.ofFn fun i : Fin 16 => (Word.cast words[i]! : F))).getD lane 0 :=
            (permutes_poseidon2 words lane).symm
        _ = _ := by
            rw [cast, permute_runF, List.getD_eq_getElem _ _ (by simp), List.getElem_ofFn]

theorem checkBlock_appends (column : ℕ → Var) (block : NightstreamFPrime.Export.Generic.Frame.Block) :
    Appends (checkBlock column block) := by
  unfold checkBlock
  exact Appends.bind (by appends) fun _ => Appends.bind (by appends) fun _ =>
    Appends.forM_list (fun _ => by appends) _

instance checkBlocks_appends (column : ℕ → Var) (blocks : List NightstreamFPrime.Export.Generic.Frame.Block) :
    Appends (checkBlocks column blocks) :=
  Appends.forM_list (checkBlock_appends column) blocks

instance bindWords_appends (column : ℕ → Var) (statement : Array Var) (one : Var) (stateKeys : List Word) :
    Appends (bindWords column statement one stateKeys) := by
  unfold bindWords
  exact Appends.bind (by appends) fun _ => Appends.bind (by appends) fun _ =>
    Appends.bind (Appends.forM_list (fun _ => by appends) _) fun _ => Appends.forM_list (fun _ => by appends) _

/-! ### The prefix -/

theorem laneFits_spec {bound : ℕ} {lane : Expr} (h : laneFits bound lane = true) :
    ∃ form, linearForm lane = some form ∧ ∀ t ∈ form.2, t.1 < bound := by
  unfold laneFits at h
  split at h
  · rename_i form hform
    exact ⟨form, hform, fun t member => by simpa using List.all_eq_true.mp h t member⟩
  · cases h

theorem fits_spec {finish : Package} (h : fits finish = true) :
    bitsStart finish.application.1 + 256 ≤ finish.sboxStart ∧
      ∀ block ∈ finish.blocks, block.lanes.length = 16 ∧ block.start + 1096 ≤ finish.sboxStart ∧
        ∀ lane ∈ block.lanes, ∃ form, linearForm lane = some form ∧ ∀ t ∈ form.2, t.1 < finish.sboxStart := by
  simp only [fits, Bool.and_eq_true, decide_eq_true_eq, List.all_eq_true] at h
  exact ⟨h.1, fun block member => ⟨(h.2 block member).1.1, (h.2 block member).1.2,
    fun lane inside => laneFits_spec ((h.2 block member).2 lane inside)⟩⟩

/-- The finish columns that a run gives: the value of its cell on a live column, and 0 on every
other column. -/
def finishEnv (c : Ctx F) (live : Array Bool) (column : ℕ → Var) : Env :=
  fun k => if live.getD k false then c.val (column k) else 0

theorem checkBlocks_ensures (hperm : c.perm = NightstreamFPrime.Spec.Poseidon2.permute) {column : ℕ → Var}
    {env : Env} {n bound : ℕ} {blocks : List NightstreamFPrime.Export.Generic.Frame.Block}
    (columns : ∀ k < bound, ColumnIs c column env n k)
    (fit : ∀ block ∈ blocks, block.lanes.length = 16 ∧ block.start + 1096 ≤ bound ∧
      ∀ lane ∈ block.lanes, ∃ form, linearForm lane = some form ∧ ∀ t ∈ form.2, t.1 < bound) :
    Ensures c n (checkBlocks column blocks) fun _ _ => ∀ block ∈ blocks,
      (fun lane : Fin 16 => env (block.start + 1080 + lane)) =
        Permutation.runF Permutation.schedule fun lane => (block.lanes.getD lane 0).eval env := by
  unfold checkBlocks
  refine forM_ensures _ (checkBlock_appends column) blocks n fun block member m le => ?_
  obtain ⟨lanes, outputs, forms⟩ := fit block member
  exact checkBlock_ensures hperm lanes
    (fun lane inside => by
      obtain ⟨form, hform, below⟩ := forms lane inside
      exact ⟨form, hform, fun t term => (columns t.1 (below t term)).mono le⟩)
    (fun lane => (columns _ (by omega)).mono le)

private theorem statement_getD {j : ℕ} (inside : j < 9) : (List.range' 0 9).toArray.getD j 0 = j := by
  rw [Array.getD_eq_getD_getElem?, List.getElem?_toArray, List.getElem?_range' inside]
  simp

private theorem value_of {k : ℕ} {env : Env} {x : F}
    (word : Word.ofF (env k) = Word.ofF x) : env k = x := by
  rw [← cast_ofF (env k), word, cast_ofF]

theorem bindWords_ensures {column : ℕ → Var} {env : Env} {n bound : ℕ} {statement : Array Var} {one : Var}
    {stateKeys : List Word} (reads : ReadsStatement c) (hstatement : statement = (List.range' 0 9).toArray)
    (nine : 9 ≤ n) (hone : IsWord c n one 1) (columns : ∀ k < bound, ColumnIs c column env n k)
    (room : keysWord + stateKeys.length ≤ bound) (words : currentWord + 4 ≤ bound) :
    Ensures c n (bindWords column statement one stateKeys) fun _ _ =>
      env 0 = 1 ∧ env iterationWord = statementOf c 0 ∧
        (∀ lane (inside : lane < 4), env (z0Word + lane) = statementOf c ⟨1 + lane, by omega⟩ ∧
          env (currentWord + lane) = statementOf c ⟨5 + lane, by omega⟩) ∧
        ∀ k (inside : k < stateKeys.length), env (keysWord + k) = Word.cast stateKeys[k] := by
  have stmt : ∀ j (inside : j < 9), IsWord c n (statement.getD j 0) (Word.ofF (statementOf c ⟨j, inside⟩)) :=
    fun j inside => by
      rw [hstatement, statement_getD inside]
      exact ⟨Nat.lt_of_lt_of_le inside nine, (reads ⟨j, inside⟩).trans (cast_ofF _).symm⟩
  have column : ∀ k, k < bound → ∀ m, n ≤ m → IsWord c m (column k) (Word.ofF (env k)) :=
    fun k inside m le => ((columns k inside).mono le).word
  unfold bindWords
  refine (assertEqual_ensures (column 0 (by unfold currentWord z0Word keysWord publicWords at words; omega) n le_rfl)
    hone).bind (fun _ m₁ le₁ h₁ => ?_)
  refine (assertEqual_ensures (column iterationWord (by unfold currentWord z0Word keysWord iterationWord at *; omega)
    m₁ le₁) ((stmt 0 (by omega)).mono le₁)).bind (fun _ m₂ le₂ h₂ => ?_)
  refine (forM_ensures (fun lane : ℕ => ∀ inside : lane < 4,
      env (z0Word + lane) = statementOf c ⟨1 + lane, by omega⟩ ∧
        env (currentWord + lane) = statementOf c ⟨5 + lane, by omega⟩) (fun _ => by appends) (List.range 4) m₂
      fun lane member m le => ?_).bind (fun _ m₃ le₃ h₃ => ?_)
  · have small : lane < 4 := List.mem_range.mp member
    refine (assertEqual_ensures (column _ (by unfold currentWord at words; omega) m (by omega))
      ((stmt (1 + lane) (by omega)).mono (by omega))).bind (fun _ m' le' hz => ?_)
    refine (assertEqual_ensures (column _ (by omega) m' (by omega))
      ((stmt (5 + lane) (by omega)).mono (by omega))).post fun _ _ hc _ => ⟨value_of hz, value_of hc⟩
  · refine (forM_ensures (fun entry : Word × ℕ => env (keysWord + entry.2) = Word.cast entry.1)
      (fun _ => by appends) stateKeys.zipIdx m₃ fun entry member m le => ?_).post fun _ _ h₄ => ?_
    · obtain ⟨word, index⟩ := entry
      have inside : index < stateKeys.length := by
        have := List.mem_zipIdx_iff_getElem?.mp member
        exact (List.getElem?_eq_some_iff.mp this).1
      refine (const_ensures word).bind fun key m' le' hkey => ?_
      refine (assertEqual_ensures (column _ (by omega) m' (by omega)) hkey).post fun _ _ h => ?_
      show env (keysWord + index) = Word.cast word
      rw [← cast_ofF (env _), h]
    · refine ⟨by rw [← cast_ofF (env 0), h₁, word_one], value_of h₂,
        fun lane inside => h₃ lane (List.mem_range.mpr inside) inside, fun k inside => ?_⟩
      exact h₄ (stateKeys[k], k) (List.mem_zipIdx_iff_getElem?.mpr (List.getElem?_eq_getElem inside))

instance finishColumns_appends (rows : List (Shrink.R1csRow Word))
    (blocks : List NightstreamFPrime.Export.Generic.Frame.Block) (live : Array Bool) (statement : Array Var)
    (one : Var) (stateKeys : List Word) : Appends (finishColumns rows blocks live statement one stateKeys) := by
  unfold finishColumns; appends

instance finishPrefix_appends (finish : Package) (stateKeys : List Word) :
    Appends (finishPrefix finish stateKeys) := by
  unfold finishPrefix; appends

/-- The checks of `finishColumns` hold on the finish columns `env`, and each column's handle holds
it. -/
def ColumnsHold (c : Ctx F) (rows : List (Shrink.R1csRow Word))
    (blocks : List NightstreamFPrime.Export.Generic.Frame.Block) (stateKeys : List Word) (bound : ℕ) (env : Env)
    (column : ℕ → Var) (m : ℕ) : Prop :=
  (env 0 = 1 ∧ env iterationWord = statementOf c 0 ∧
    (∀ lane (inside : lane < 4), env (z0Word + lane) = statementOf c ⟨1 + lane, by omega⟩ ∧
      env (currentWord + lane) = statementOf c ⟨5 + lane, by omega⟩) ∧
    ∀ k (inside : k < stateKeys.length), env (keysWord + k) = Word.cast stateKeys[k]) ∧
  (∀ r ∈ rows, R1csRow.Holds env r) ∧
  (∀ block ∈ blocks, (fun lane : Fin 16 => env (block.start + 1080 + lane)) =
    Permutation.runF Permutation.schedule fun lane => (block.lanes.getD lane 0).eval env) ∧
  ∀ k < bound, ColumnIs c column env m k

/-- The hint of the finish columns to the blocks, for any rows, blocks and live columns below
`bound`: the checks hold on the cells of the live columns and 0 on every other column. -/
theorem finishColumns_ensures (hperm : c.perm = NightstreamFPrime.Spec.Poseidon2.permute)
    {rows : List (Shrink.R1csRow Word)} {blocks : List NightstreamFPrime.Export.Generic.Frame.Block}
    {live : Array Bool} {statement : Array Var} {one : Var} {stateKeys : List Word} {bound n : ℕ}
    (size : live.size = bound) (reads : ReadsStatement c) (hstatement : statement = (List.range' 0 9).toArray)
    (nine : 9 ≤ n) (hone : IsWord c n one 1) (room : keysWord + stateKeys.length ≤ bound)
    (words : currentWord + 4 ≤ bound)
    (fit : ∀ block ∈ blocks, block.lanes.length = 16 ∧ block.start + 1096 ≤ bound ∧
      ∀ lane ∈ block.lanes, ∃ form, linearForm lane = some form ∧ ∀ t ∈ form.2, t.1 < bound) :
    Ensures c n (finishColumns rows blocks live statement one stateKeys) fun column m =>
      ColumnsHold c rows blocks stateKeys bound (finishEnv c live column) column m := by
  unfold finishColumns
  refine (Ensures.emit (Q := fun h m => h = n ∧ m = n + live.count true) (.hint (live.count true))
    fun E hE _ _ => by simp [Instr.outputs]).bind fun start m₁ le₁ hstart => ?_
  obtain ⟨rfl, rfl⟩ := hstart
  refine (const_ensures 0).bind fun zeroForm m₂ le₂ hzero => ?_
  obtain ⟨formsSize, forms⟩ := columnForms_spec live start zeroForm
  rw [size] at formsSize forms
  let column : ℕ → Var := fun k => (columnForms live start zeroForm).getD k zeroForm
  let env := finishEnv c live column
  have columns : ∀ k < bound, ColumnIs c column env m₂ k := fun k inside => by
    cases flag : live.getD k false with
    | true => exact ⟨lt_of_lt_of_le ((forms k inside).1 flag).2 le₂, by simp [env, finishEnv, flag]⟩
    | false =>
        have dead : column k = zeroForm := (forms k inside).2 flag
        exact ⟨dead ▸ hzero.1, by simp [env, finishEnv, flag, dead, hzero.2]⟩
  refine (bindWords_ensures (env := env) (bound := bound) reads hstatement (by omega) (hone.mono (by omega))
    columns room words).bind fun _ m₃ le₃ hwords => ?_
  refine (r1cs_ensures (rows := rows) (columns := (columnForms live start zeroForm).toList)
    fun h member => ?_).bind fun _ m₄ le₄ hrows => ?_
  · obtain ⟨k, inside, rfl⟩ := List.getElem_of_mem member
    have below : k < bound := by rw [← formsSize]; simpa using inside
    have form : column k = (columnForms live start zeroForm).toList[k] := by
      simp [column, Array.getD_eq_getD_getElem?, formsSize, below]
    exact form ▸ lt_of_lt_of_le (columns k below).1 le₃
  refine (checkBlocks_ensures hperm (env := env) (fun k inside => (columns k inside).mono (le₃.trans le₄))
    fit).bind fun _ m₅ le₅ hblocks => Ensures.pure ⟨hwords, ?_, hblocks,
      fun k inside => (columns k inside).mono ((le₃.trans le₄).trans le₅)⟩
  have envRows : (fun j => (((columnForms live start zeroForm).toList).map c.val).getD j 0) = env := by
    funext j
    by_cases inside : j < bound
    · rw [← (columns j inside).2]
      simp [column, List.getD_eq_getElem?_getD, Array.getD_eq_getD_getElem?, formsSize, inside]
    · have dead : live.getD j false = false := by
        rw [Array.getD_eq_getD_getElem?, Array.getElem?_eq_none (by rw [size]; exact Nat.le_of_not_lt inside)]
        rfl
      simp [env, finishEnv, dead, List.getD_eq_getElem?_getD, formsSize, inside]
  rw [envRows] at hrows
  exact hrows

/-- What `finishPrefix` hands layer 1: the program's checks hold on the finish columns, and each
column's handle holds it. -/
def Handoff (c : Ctx F) (finish : Package) (stateKeys : List Word) (r : Array Var × (ℕ → Var)) (m : ℕ) : Prop :=
  FinishTerminal finish stateKeys (statementOf c) (finishEnv c (live finish) r.2) ∧
    ∀ k < finish.sboxStart, ColumnIs c r.2 (finishEnv c (live finish) r.2) m k

theorem finishPrefix_ensures (hperm : c.perm = NightstreamFPrime.Spec.Poseidon2.permute) {finish : Package}
    {stateKeys : List Word} (fit : fits finish = true) (keys : stateKeys.length ≤ 8) (reads : ReadsStatement c) :
    Ensures c 0 (finishPrefix finish stateKeys) (Handoff c finish stateKeys) := by
  obtain ⟨words, blocks⟩ := fits_spec fit
  have below : ∀ k, k ≤ keysWord + 8 ∨ k < currentWord + 4 → k < finish.sboxStart := fun k inside => by
    unfold bitsStart slotStart witnessWord currentWord z0Word keysWord publicWords at *
    omega
  have room : keysWord + stateKeys.length ≤ finish.sboxStart := by
    have := below (keysWord + 8) (Or.inl le_rfl)
    omega
  have current : currentWord + 4 ≤ finish.sboxStart := by
    have := below (currentWord + 3) (Or.inr (by omega))
    omega
  unfold finishPrefix
  refine (statementHandles_ensures (c := c) (n := 0)).bind (fun statement m₀ _ hstatement => ?_)
  refine (Ensures.emit (Q := fun h m => h < m) (.hint 1) fun _ _ _ _ => by simp [Instr.outputs]).bind
    fun inverse m₁ le₁ hinverse => ?_
  have first : IsWord c m₁ (statement.getD 0 0) (Word.ofF (statementOf c 0)) := by
    rw [hstatement.1, statement_getD (by decide : 0 < 9)]
    exact ⟨show 0 < m₁ by omega, (reads 0).trans (cast_ofF _).symm⟩
  refine (mul_ensures first ⟨hinverse, (cast_ofF _).symm⟩).bind fun product m₂ le₂ hproduct => ?_
  refine (const_ensures 1).bind fun one m₃ le₃ hone => ?_
  refine (assertEqual_ensures hproduct.mono hone).bind fun _ m₄ le₄ hpositive => ?_
  have nine : 9 ≤ m₄ := by omega
  have positive : statementOf c 0 ≠ 0 := by
    intro zero
    have := congrArg (Word.cast : Word → F) hpositive
    rw [word_mul, cast_ofF, cast_ofF, word_one] at this
    change statementOf c 0 * _ = 1 at this
    rw [zero, zero_mul] at this
    exact absurd this (by decide)
  refine (finishColumns_ensures hperm (live := live finish) (live_size finish) reads hstatement.1 nine
    (hone.mono le₄) room current blocks).bind fun column m₅ le₅ holds => Ensures.pure ?_
  obtain ⟨hwords, hrows, hblocks, columns⟩ := holds
  refine ⟨⟨positive, hwords.1, hwords.2.1, fun lane => hwords.2.2.1 lane lane.isLt, hwords.2.2.2,
    fun r member => ?_, hblocks⟩, columns⟩
  exact (Lowering.Package.row_holds _ 0 hwords.1 r).mp
    (hrows (Terminal.Package.row 0 r) (List.mem_map_of_mem member))

end CompressionLean.Sim.BoundedDepth
