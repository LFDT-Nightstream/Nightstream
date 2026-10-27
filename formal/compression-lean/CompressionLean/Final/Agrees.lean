import CompressionLean.Final.Shipped
import CompressionLean.Circuit.Counts
import CompressionLean.Circuit.SimRun

/-!
Owns `circuitAgrees_of_verify`: the circuit-replay agreement of the shipped
program (`Shipped.CircuitAgrees`) from the agreement of the layer-1
verifier alone (`VerifyAgrees`, open) and the claim sizes
(`OutputsFit`).

- `lower_counts` moves the witness to the lowering on its own statement,
  and `lower_sound` gives an accepting executor run.
- The terminal part of the final program writes the statement words, the
  iteration check and the package rows (`finalProgram_accepts`). The claim
  and the transcript state that it hands layer 1 are among its values
  (`Within`), because every computed source names a hint output
  (`load_computed`), and the claim is read from the package columns
  (`claimOf`).
- `VerifyAgrees key`: on a run of `verifyWith` whose checks hold, the
  replay accepts, from `start`, the seed, the claim and the proof that the
  run's handles hold. It is a statement about the verifier's builder
  program and the replay only, for one key. The static facts of the key
  that the replay checks and the circuit does not (tree shapes, WHIR batch
  counts and sizes) are inside it.
-/

namespace CompressionLean.Circuit.Package

open CompressionLean.Replay (Gl)

/-- Every computed source names a hint output. -/
def ComputedBelow (sources : Array Source) (computed : Array ℕ) : Prop :=
  ∀ k, Source.computed k ∈ sources → k < computed.size

/-- A pure loop that always yields keeps an invariant of its steps. -/
theorem forIn_invariant {α β : Type} (Inv : β → Prop) (f : α → β → Id (ForInStep β))
    (step : ∀ x s, Inv s → ∃ s', f x s = pure (.yield s') ∧ Inv s') :
    ∀ (l : List α) (init : β), Inv init → Inv (forIn l init f)
  | [], _, h => h
  | x :: l, init, h => by
      obtain ⟨s', e, hs'⟩ := step x init h
      rw [List.forIn_cons, e]
      exact forIn_invariant Inv f step l s' hs'

theorem load_computed {pkg : NightstreamFPrime.Export.Package.CircuitPackage} {map : Terminal.Outputs}
    {loaded : Loaded} (h : load pkg map = .ok loaded) : ComputedBelow loaded.sources loaded.computed := by
  unfold load at h
  simp only [bind, Except.bind] at h
  split at h
  · repeat' (split at h; · cases h)
    cases h
    dsimp only [pure]
    rw [Std.Legacy.Range.forIn_eq_forIn_range']
    refine forIn_invariant (fun s : Array (Option ℕ) × Array Source × Array ℕ => ComputedBelow s.2.1 s.2.2) _
      (fun column s hs => ?_) _ _
      fun k hk => absurd hk (by simp)
    have keep : ∀ {x : Source} {extra : Array ℕ}, (∀ k, x = .computed k → k < (s.2.2 ++ extra).size) →
        ComputedBelow (s.2.1.push x) (s.2.2 ++ extra) := fun hx k hk => by
      rcases Array.mem_push.mp hk with hk | hk
      · exact (hs k hk).trans_le (by simp)
      · exact hx k hk.symm
    split
    · split
      · exact ⟨_, rfl, by simpa using keep (extra := #[]) fun k e => by cases e⟩
      split
      · exact ⟨_, rfl, by simpa using keep (extra := #[]) fun k e => by cases e⟩
      split
      · exact ⟨_, rfl, by simpa using keep (extra := #[]) fun k e => by cases e⟩
      · refine ⟨_, rfl, ?_⟩
        have := keep (x := .computed s.2.2.size) (extra := #[column]) fun k e => by
          injection e with e
          simp [← e]
        simpa using this
    · exact ⟨_, rfl, hs⟩
  · cases h

end CompressionLean.Circuit.Package

namespace CompressionLean.Circuit

open CompressionLean.Circuit.Package CompressionLean.Circuit.Terminal
open CompressionLean.Replay (Gl)

variable {F : Type} [CommRing F] {c : Ctx F}

/-- The handles of a terminal claim and transcript state, below `m`. -/
def Within (m : ℕ) (claim : ClaimVars) (state : Fin 16 → Var) : Prop :=
  (∀ h ∈ (claimView claim).words, h < m) ∧ ∀ lane, state lane < m

theorem mem_of_mem_extract {α : Type} {xs : Array α} {a b : ℕ} {y : α} (h : y ∈ xs.extract a b) : y ∈ xs := by
  obtain ⟨i, hi, rfl⟩ := Array.mem_iff_getElem.mp h
  rw [Array.getElem_extract]
  exact Array.getElem_mem _

theorem getD_lt {xs : Array ℕ} {m : ℕ} (h : ∀ y ∈ xs, y < m) (zero : 0 < m) (k : ℕ) : xs.getD k 0 < m := by
  rw [Array.getD_eq_getD_getElem?]
  cases e : xs[k]? with
  | none => exact zero
  | some y => exact h y (Array.mem_of_getElem? e)

theorem chunks_lt {xs : Array ℕ} {m n : ℕ} (h : ∀ y ∈ xs, y < m) :
    ∀ chunk ∈ Terminal.chunks n xs, ∀ y ∈ chunk, y < m := by
  intro chunk hc y hy
  obtain ⟨k, -, rfl⟩ := Array.mem_map.mp hc
  exact h y (mem_of_mem_extract hy)

theorem pairs_lt {xs : Array ℕ} {m : ℕ} (h : ∀ y ∈ xs, y < m) (zero : 0 < m) :
    ∀ p ∈ Terminal.pairs xs, p.1 < m ∧ p.2 < m := by
  intro p hp
  obtain ⟨pair, hpair, rfl⟩ := Array.mem_map.mp hp
  exact ⟨getD_lt (chunks_lt h pair hpair) zero 0, getD_lt (chunks_lt h pair hpair) zero 1⟩

/-- `column` writes at most one handle, or names a statement word or a hint
output. -/
theorem column_accepts {statement : Array Var} {context : Array Gl} {computed : Var} (s : Source) {m : ℕ} {P : Prop}
    (words : P → ∀ y ∈ statement, y < m) (zero : P → 0 < m)
    (hint : P → ∀ k, s = .computed k → computed + k < m) :
    Accepts c m P (column statement context computed s) fun y m' => y < m' := by
  have new (i : Instr) (one : i.outputs = 1) : Accepts c m P (Build.emit i) fun y m' => y < m' :=
    (Accepts.emit i).post fun _ y m' e => by
      rw [e.1, e.2, one]
      exact Nat.lt_succ_self m
  cases s with
  | one => exact new _ rfl
  | priv i =>
      dsimp only [column]
      split
      · exact new _ rfl
      split
      · exact new _ rfl
      split
      · exact Accepts.pure fun hP => getD_lt (words hP) (zero hP) _
      · exact new _ rfl
  | pub i =>
      dsimp only [column]
      split
      · exact new _ rfl
      · exact new _ rfl
  | computed k => exact Accepts.pure fun hP => hint hP k rfl

theorem r1cs_accepts (rows : List (Shrink.R1csRow Gl)) (columns : List Var) {m : ℕ} {P : Prop} :
    Accepts c m P (Build.r1cs rows columns) fun _ _ => True :=
  (Accepts.emit _).bind fun _ _ _ => Accepts.pure fun _ => trivial

theorem assertEqual_accepts (a b : Var) {m : ℕ} {P : Prop} : Accepts c m P (Build.assertEqual a b) fun _ _ => True :=
  (Accepts.emit _).bind fun _ _ _ => (Accepts.emit _).bind fun _ _ _ => Accepts.pure fun _ => trivial

/-- The claim that `apply` reads from the package columns. -/
def claimOf (outputs : OutputColumns) (columns : Array Var) : ClaimVars :=
  let read := fun (list : Array ℕ) => list.map fun k => columns.getD k 0
  { commitment := Terminal.chunks Replay.D (read outputs.commitment),
    publicInput := Terminal.chunks Replay.D (read outputs.publicInput), point := pairs (read outputs.point),
    evalK := pairs (read outputs.evalK), evalA := (Terminal.chunks (2 * Replay.D) (read outputs.evalA)).map pairs }

/-- `Package::apply`: the claim and the state are among the columns, which
are below the count after the rows. -/
theorem apply_accepts {package : Loaded} {statement : Array Var} {context : Array Gl} {m : ℕ} {P : Prop}
    (computed : ComputedBelow package.sources package.computed) (words : P → ∀ y ∈ statement, y < m)
    (zero : P → 0 < m) :
    Accepts c m P (apply package statement context) fun r m' =>
      Within m' r.1 r.2 ∧ ∃ columns, r.1 = claimOf package.outputs columns := by
  unfold apply
  refine (Accepts.emit (c := c) _).bind fun hint m₁ l₁ => ?_
  refine (Accepts.arrayMapM (R := fun _ y m => y < m) (fun _ _ _ _ h l => lt_of_lt_of_le h l) _
    fun s hs m' le => column_accepts s (fun h y hy => lt_of_lt_of_le (words h.1 y hy) (l₁.trans le))
      (fun h => lt_of_lt_of_le (zero h.1) (l₁.trans le)) fun h k e => ?_).bind fun columns m₂ l₂ => ?_
  · subst e
    obtain ⟨-, rfl, rfl⟩ := h
    simp only [Instr.outputs] at le
    exact Nat.lt_of_lt_of_le (Nat.add_lt_add_left (computed k hs) _) le
  refine (r1cs_accepts (c := c) _ _).bind fun _ m₃ l₃ => Accepts.pure fun h => ?_
  have cols : ∀ y ∈ columns, y < m₃ := fun y hy => by
    obtain ⟨i, hi, rfl⟩ := Array.mem_iff_getElem.mp hy
    exact lt_of_lt_of_le (h.1.2.2 i (h.1.2.1 ▸ hi) hi) l₃
  have pos : 0 < m₃ := lt_of_lt_of_le (zero h.1.1.1) (l₁.trans (l₂.trans l₃))
  have read : ∀ list : Array ℕ, ∀ y ∈ list.map (fun k => columns.getD k 0), y < m₃ := fun list y hy => by
    obtain ⟨k, -, rfl⟩ := Array.mem_map.mp hy
    exact getD_lt cols pos k
  refine ⟨⟨fun y hy => ?_, fun lane => getD_lt cols pos _⟩, columns, rfl⟩
  simp only [claimView, ClaimView.words, Array.mem_append, Array.mem_flatten, Array.mem_flatMap] at hy
  have pair : ∀ a, (a ∈ pairs (Array.map (fun k => columns.getD k 0) package.outputs.point) ∨
      a ∈ pairs (Array.map (fun k => columns.getD k 0) package.outputs.evalK)) ∨
      (∃ xs ∈ Array.map pairs (Terminal.chunks (2 * Replay.D) (Array.map (fun k => columns.getD k 0)
        package.outputs.evalA)), a ∈ xs) → a.1 < m₃ ∧ a.2 < m₃ := by
    rintro a ((ha | ha) | ⟨xs, hxs, ha⟩)
    · exact pairs_lt (read _) pos a ha
    · exact pairs_lt (read _) pos a ha
    · obtain ⟨chunk, hchunk, rfl⟩ := Array.mem_map.mp hxs
      exact pairs_lt (chunks_lt (read _) chunk hchunk) pos a ha
  rcases hy with ((⟨xs, hxs, hy⟩ | ⟨xs, hxs, hy⟩) | ⟨a, ha, hy⟩)
  · exact chunks_lt (read _) xs hxs y hy
  · exact chunks_lt (read _) xs hxs y hy
  · have := pair a ha
    simp only [List.mem_toArray, List.mem_cons, List.not_mem_nil, or_false] at hy
    rcases hy with rfl | rfl
    · exact this.1
    · exact this.2

/-- **The terminal part of the final program.** It writes the statement
words, the iteration check and the package rows; the claim and the
transcript state that it hands layer 1 are among its values. -/
theorem finalProgram_accepts {α : Type} {package : Loaded} {context : Array Gl}
    {layer1 : ClaimVars → (Fin 16 → Var) → Build α} {P : Prop} {Q : α → ℕ → Prop}
    (computed : ComputedBelow package.sources package.computed)
    (h : ∀ claim state m,
      Accepts c m (P ∧ Within m claim state ∧ ∃ columns, claim = claimOf package.outputs columns)
        (layer1 claim state) Q) :
    Accepts c 0 P (finalProgram package context layer1) Q := by
  unfold finalProgram
  refine (Accepts.arrayMapM (R := fun _ y m => y < m) (fun _ _ _ _ h l => lt_of_lt_of_le h l) _
    fun _ _ m _ => (Accepts.emit .pub).post fun _ y m' e => by rw [e.1, e.2]; simp [Instr.outputs]).bind
    fun statement m₁ _ => ?_
  refine (Accepts.emit (c := c) _).bind fun inverse m₂ l₂ => ?_
  refine (Accepts.emit (c := c) _).bind fun product m₃ l₃ => ?_
  refine (Accepts.emit (c := c) _).bind fun one m₄ l₄ => ?_
  refine (assertEqual_accepts (c := c) _ _).bind fun _ m₅ l₅ => ?_
  have le : m₁ ≤ m₅ := l₂.trans (l₃.trans (l₄.trans l₅))
  refine (apply_accepts (c := c) computed (fun h y hy => ?_) (fun h => ?_)).bind
    fun r m₆ l₆ => (h r.1 r.2 m₆).pre fun h => ⟨h.1.1.1.1.1.1, h.2⟩
  · obtain ⟨i, hi, rfl⟩ := Array.mem_iff_getElem.mp hy
    exact lt_of_lt_of_le (h.1.1.1.1.2.2 i (by rw [← h.1.1.1.1.2.1]; exact hi) hi) le
  · have size := h.1.1.1.1.2.1
    simp only [Array.size_range, statementWords] at size
    exact lt_of_le_of_lt (Nat.zero_le _) (lt_of_lt_of_le (h.1.1.1.1.2.2 0 (by simp [statementWords]) (by omega)) le)

end CompressionLean.Circuit

namespace CompressionLean.Final

open NightstreamFPrime.Spec
open CompressionLean.Replay (D)

/-! ## The claim sizes -/

/-- The claim view has the relation's sizes, as `Statement.fits` checks. -/
def ClaimFits (claim : Circuit.ClaimView) (rel : Replay.Relation) : Prop :=
  claim.commitment.size = rel.kappa ∧ (∀ row ∈ claim.commitment, row.size = D) ∧
    claim.publicInput.size = rel.publicBlocks ∧ (∀ block ∈ claim.publicInput, block.size = D) ∧
    claim.point.size = rel.pointVariables ∧ claim.evalK.size = D ∧ claim.evalA.size = rel.matrices ∧
    ∀ values ∈ claim.evalA, values.size = D

/-- A fitting claim view decodes to a statement that fits. -/
theorem fits_of_claimFits {claim : Circuit.ClaimView} {rel : Replay.Relation} (h : ClaimFits claim rel)
    (v : Circuit.Var → Replay.Gl) : (Decode.statement v claim).fits rel = true := by
  obtain ⟨c1, c2, p1, p2, pt, k, a1, a2⟩ := h
  have all {α β : Type} (xs : Array (Array α)) (f : α → β) (h : ∀ x ∈ xs, x.size = D) :
      (xs.map (·.map f)).all (fun a => a.size == D) = true :=
    Array.all_eq_true'.mpr fun y hy => by
      obtain ⟨x, hx, rfl⟩ := Array.mem_map.mp hy
      simp [h x hx]
  unfold Replay.Statement.fits Decode.statement Decode.kValues
  rw [all _ _ c2, all _ _ p2, all _ _ a2]
  simp [c1, p1, pt, k, a1]

/-- **The terminal package's outputs fit the key.** Each claim part that the
package outputs has the relation's size, counted as `apply` reads it: rows
of `D` words, and `K` values as pairs of words. Static: it reads only sizes. -/
def OutputsFit (package : Circuit.Package.Loaded) (rel : Replay.Relation) : Prop :=
  package.outputs.commitment.size / D = rel.kappa ∧ package.outputs.publicInput.size / D = rel.publicBlocks ∧
    package.outputs.point.size / 2 = rel.pointVariables ∧ package.outputs.evalK.size / 2 = D ∧
    package.outputs.evalA.size / (2 * D) = rel.matrices

theorem size_chunks {α : Type} (n : ℕ) (xs : Array α) : (Circuit.Terminal.chunks n xs).size = xs.size / n := by
  simp [Circuit.Terminal.chunks]

theorem size_of_mem_chunks {α : Type} {n : ℕ} {xs chunk : Array α} (h : chunk ∈ Circuit.Terminal.chunks n xs) :
    chunk.size = n := by
  obtain ⟨k, hk, rfl⟩ := Array.mem_map.mp h
  have hk : k < xs.size / n := by simpa using hk
  have : n * k + n ≤ xs.size := by
    have := Nat.mul_le_of_le_div n (k + 1) xs.size (by omega)
    rw [Nat.succ_mul, Nat.mul_comm] at this
    exact this
  simp only [Array.size_extract]
  omega

theorem size_pairs (xs : Array Circuit.Var) : (Circuit.Terminal.pairs xs).size = xs.size / 2 := by
  simp [Circuit.Terminal.pairs, size_chunks]

/-- The outputs fit, so the claim that `apply` reads fits. -/
theorem claimFits_of_outputsFit {package : Circuit.Package.Loaded} {rel : Replay.Relation} (h : OutputsFit package rel)
    (columns : Array Circuit.Var) : ClaimFits (Circuit.claimView (Circuit.claimOf package.outputs columns)) rel := by
  obtain ⟨c, p, pt, k, a⟩ := h
  refine ⟨?_, fun _ hr => size_of_mem_chunks hr, ?_, fun _ hb => size_of_mem_chunks hb, ?_, ?_, ?_,
    fun values hv => ?_⟩
  · simp [Circuit.claimView, Circuit.claimOf, size_chunks, c]
  · simp [Circuit.claimView, Circuit.claimOf, size_chunks, p]
  · simp [Circuit.claimView, Circuit.claimOf, size_pairs, pt]
  · simp [Circuit.claimView, Circuit.claimOf, size_pairs, k]
  · simp [Circuit.claimView, Circuit.claimOf, size_chunks, a]
  · simp only [Circuit.claimView, Circuit.claimOf] at hv
    obtain ⟨chunk, hchunk, rfl⟩ := Array.mem_map.mp hv
    rw [size_pairs, size_of_mem_chunks hchunk]
    simp

/-! ## The agreement -/

/-- **The verifier agrees with the replay.** Open: the executor-simulation
units will prove it, stage by stage. On a run of `verifyWith` for `key`
whose checks hold, from a builder state with `n` values that holds the
transcript state and a fitting claim, the replay accepts from `start` the
seed, the claim and the proof that the run's handles hold, read by `v`.

It reads only the verifier's builder program, the replay and the key. The
static facts of the key that the replay checks and the circuit does not
(the tree shapes of `readGkr`, the WHIR batch counts and sizes of
`whirTranscript` and `openingsAgree`) hold for a consistent key; they are
part of what it states for `key`. -/
def VerifyAgrees (key : Circuit.Layer1Input) : Prop :=
  ∀ (c : Circuit.Ctx Gl) (v : Circuit.Var → Replay.Gl) (n : ℕ) (state : Fin 16 → Circuit.Var)
    (claim : Circuit.ClaimView),
    Circuit.Accepts c n
      (Circuit.Permutes c.perm ∧ (∀ h, c.val h = Circuit.word (v h)) ∧ (∀ lane, state lane < n) ∧
        (∀ h ∈ claim.words, h < n) ∧ ClaimFits claim key.relation)
      (Circuit.verifyWith key state claim) fun r _ =>
        ∃ points t', Replay.Runs (Replay.verify key.relation (Decode.digest v r.1) (Decode.statement v claim)
          (Decode.proof v r.2)) Shipped.start points t'

/-- `lower_sound` over `Gl`, for any program. -/
theorem lower_sound_gl (L : Array Circuit.Instr) (stmt : List Gl) (w : ℕ → Gl)
    (holds : ((Circuit.lower L stmt).ccs Shipped.template (Circuit.lower L stmt).counts).Relation w) :
    Circuit.Exec (F := Gl) NightstreamFPrime.Spec.Poseidon2.permute L stmt (Circuit.tapeOf Shipped.template L stmt w) :=
  Circuit.lower_sound L stmt w holds

/-- A field value is the word of its canonical word. -/
theorem word_packageWord (x : Gl) : Circuit.word (Circuit.Package.word x) = x := by
  rw [Circuit.word_eq_cast, ZMod.castHom_self, RingHom.id_apply, Circuit.Package.word, Replay.Gl.toF_ofNat]
  exact ZMod.natCast_zmod_val x

namespace Shipped

variable (sp : Shipped)

/-- An accepting executor run of a program `L` that is the final program
gives the replay's run on what its values decode to. -/
theorem replay_of_exec (verify : VerifyAgrees sp.key) (fit : OutputsFit sp.package sp.key.relation)
    (stmt : List Gl) (tape : ℕ → Gl) {L : Array Circuit.Instr} (program : sp.program = L)
    (exec : Circuit.Exec (F := Gl) NightstreamFPrime.Spec.Poseidon2.permute L stmt tape) (v : Circuit.Var → Replay.Gl)
    (hv : ∀ h, v h = Circuit.Package.word
      ((Circuit.Exec.run (F := Gl) NightstreamFPrime.Spec.Poseidon2.permute stmt tape L).value h)) :
    ∃ points t', Replay.Runs (Replay.verify sp.key.relation (Decode.digest v sp.views.seed)
      (Decode.statement v sp.views.claim) (Decode.proof v sp.views.proof)) start points t' := by
  have accepts : Circuit.Accepts (⟨NightstreamFPrime.Spec.Poseidon2.permute, stmt, tape,
      (Circuit.Exec.run (F := Gl) NightstreamFPrime.Spec.Poseidon2.permute stmt tape L).value⟩ : Circuit.Ctx Gl) 0
      (OutputsFit sp.package sp.key.relation)
      (Circuit.finalViews sp.package sp.context sp.key) fun views _ =>
        ∃ points t', Replay.Runs (Replay.verify sp.key.relation (Decode.digest v views.seed)
          (Decode.statement v views.claim) (Decode.proof v views.proof)) start points t' := by
    unfold Circuit.finalViews
    refine Circuit.finalProgram_accepts (Circuit.Package.load_computed sp.load) fun claim state m => ?_
    refine ((verify _ v m state (Circuit.claimView claim)).pre fun h =>
      ⟨Circuit.permutes_poseidon2, fun h => by rw [hv]; exact (word_packageWord _).symm, h.2.1.2, h.2.1.1, ?_⟩).bind
        fun r m' _ => Circuit.Accepts.pure fun h => h.2
    obtain ⟨columns, rfl⟩ := h.2.2
    exact claimFits_of_outputsFit h.1 columns
  rw [show sp.views = ((Circuit.finalViews sp.package sp.context sp.key).run {}).1 from rfl]
  exact accepts.exec_of_eq program fit rfl exec

/-- **The circuit-replay agreement, from the verifier's.** A witness of the
final program's CCS gives an accepting executor run (`lower_sound`, on the
counts of the statement's own lowering). The terminal part hands layer 1 a
claim and a state among its values, and the claim fits the key by
`OutputsFit`; `VerifyAgrees` then gives the replay's run on what the
witness decodes to. -/
theorem circuitAgrees_of_verify (verify : VerifyAgrees sp.key) (fit : OutputsFit sp.package sp.key.relation) :
    sp.CircuitAgrees := fun x w holds => by
  rw [show sp.counts = (sp.prog x).counts from (Circuit.lower_counts (F := Gl) sp.program (List.ofFn x)).symm] at holds
  show ∃ points t', Replay.Runs (Replay.verify sp.key.relation (sp.decode x w).seed (sp.decode x w).statement
    (sp.decode x w).proof) start points t'
  unfold decode exec
  unfold prog at holds
  dsimp only
  generalize hL : sp.program = L at holds ⊢
  exact sp.replay_of_exec verify fit (List.ofFn x) (Circuit.tapeOf template L (List.ofFn x) w) hL
    (lower_sound_gl L (List.ofFn x) w holds) _ fun _ => rfl

end Shipped

end CompressionLean.Final
