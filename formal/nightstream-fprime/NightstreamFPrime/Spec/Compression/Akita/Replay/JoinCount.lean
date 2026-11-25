import NightstreamFPrime.Spec.Compression.Akita.Replay.JoinTerminal

/-! The draws of the checks of a level and the messages of the paths of the code, for theorem A of
the join of `key20.aks` (Akita revision `d50f700`; ePrint 2026/1983, revision of 2026-11-23,
Figure 6, Sections 5.2, 6.1 and 6.3; `docs/akita-work/w16/design.md` section 2).

- `DrawsLe game n`: each run of `game` has at most `n` draws (`DrawsLe.runs`). The checks of each
  kind of level draw at most once for each bound of `Tier.costs` (`foldCheck_draws`,
  `foldCheckSq_draws`, `tensorCheck_draws`, `Tier.tensored_draws`).
- `Fits E width cap slots path index next`: the path packs from the message `index`
  (`Extract.Packs`), and its next message index is at most `next`. `narrow width arities nodes`:
  each degree of the range tree and each width of the claims of the children is at most `width`.
  The paths of the code fit from the message 0 (`fits_foldPathIn`): the round messages have at
  most `width` elements, at most two messages follow one draw, and the energy is at most
  `cap`. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita.Replay

open NightstreamFPrime.Spec.Compression.Lib NightstreamFPrime.Spec.Compression.Game
open NightstreamFPrime.Spec.Compression (Game)
open NightstreamFPrime.Spec.Compression.Akita
open NightstreamFPrime.Spec.Compression.Akita.Extract (Tier countDraws nextIndex Packs)

open Replayer

section Draws

variable {α β : Type}

/-- Each run of the game has at most `n` draws. -/
def DrawsLe : Game α → ℕ → Prop
  | .pure _, _ => True
  | @Game.draw _ _ _ _ _ next, n => 0 < n ∧ ∀ c, DrawsLe (next c) (n - 1)
  | @Game.recv _ _ _ next, n => ∀ m, DrawsLe (next m) n

theorem DrawsLe.mono {game : Game α} {n m : ℕ} (drawn : DrawsLe game n) (le : n ≤ m) :
    DrawsLe game m := by
  induction game generalizing n m with
  | pure a => trivial
  | draw C bad next induction =>
      exact ⟨Nat.lt_of_lt_of_le drawn.1 le, fun c =>
        induction c (drawn.2 c) (Nat.sub_le_sub_right le 1)⟩
  | recv M next induction => exact fun x => induction x (drawn x) le

theorem DrawsLe.bind {game : Game α} {after : α → Game β} {n m : ℕ} (first : DrawsLe game n)
    (second : ∀ a, DrawsLe (after a) m) : DrawsLe (game.bind after) (n + m) := by
  induction game generalizing n with
  | pure a => exact (second a).mono (Nat.le_add_left m n)
  | draw C bad next induction =>
      refine ⟨Nat.lt_of_lt_of_le first.1 (Nat.le_add_right n m), fun c => ?_⟩
      have later := induction c (first.2 c)
      rwa [show n - 1 + m = n + m - 1 by have := first.1; omega] at later
  | recv M next induction => exact fun x => induction x (first x)

theorem DrawsLe.ite {p : Prop} [Decidable p] {yes no : Game α} {n : ℕ} (first : DrawsLe yes n)
    (second : DrawsLe no n) : DrawsLe (if p then yes else no) n := by
  split
  · exact first
  · exact second

/-- A run of a game with at most `n` draws has at most `n` draws on its path. -/
theorem DrawsLe.runs {game : Game α} {n : ℕ} (drawn : DrawsLe game n) {path : List Move}
    {result : α} (runs : game.Runs path result) : countDraws path ≤ n := by
  induction game generalizing path n with
  | pure a =>
      cases path with
      | nil => exact Nat.zero_le _
      | cons move path => cases move <;> exact runs.elim
  | draw C bad next induction =>
      cases path with
      | nil => exact runs.elim
      | cons move path =>
          cases move with
          | recv m => exact runs.elim
          | draw c =>
              obtain ⟨same, runs⟩ := runs
              subst same
              have later := induction _ (drawn.2 _) runs
              have positive := drawn.1
              show countDraws path + 1 ≤ n
              omega
  | recv M next induction =>
      cases path with
      | nil => exact runs.elim
      | cons move path =>
          cases move with
          | draw c => exact runs.elim
          | recv m =>
              obtain ⟨same, runs⟩ := runs
              subst same
              show countDraws path ≤ n
              exact induction _ (drawn _) runs

variable {K : Type} [Field K] [Fintype K] [DecidableEq K]

theorem sumcheck_draws (d : ℕ) (round : List K → Polynomial K) :
    ∀ (rounds : ℕ) (point : List K) (claim : K),
      DrawsLe (Game.sumcheck d round rounds point claim) rounds
  | 0, _, _ => trivial
  | rounds + 1, _, _ => fun _ =>
      DrawsLe.ite ⟨Nat.succ_pos _, fun _ => sumcheck_draws d round rounds _ _⟩ trivial

theorem sumcheckWith_draws (d : ℕ) (round : List K → Polynomial K) (extra : List K → Finset K) :
    ∀ (rounds : ℕ) (point : List K) (claim : K),
      DrawsLe (sumcheckWith d round extra rounds point claim) rounds
  | 0, _, _ => trivial
  | rounds + 1, _, _ => fun _ =>
      DrawsLe.ite ⟨Nat.succ_pos _, fun _ => sumcheckWith_draws d round extra rounds _ _⟩ trivial

theorem eqSumcheck_draws (D : ℕ) (round : List K → Polynomial K) (extra : List K → Finset K)
    (τ : List K) :
    ∀ (rounds : ℕ) (point : List K) (claim : K),
      DrawsLe (eqSumcheck D round extra τ rounds point claim) rounds
  | 0, _, _ => trivial
  | rounds + 1, _, _ => fun _ => ⟨Nat.succ_pos _, fun _ => eqSumcheck_draws D round extra τ rounds _ _⟩

variable {E : Type} [Field E] [Fintype E] [DecidableEq E] [Algebra Zq E]

theorem rangeTree_draws {γ : Type} (leaf μ : ℕ) (w : Table)
    (last : ℕ → (ℕ → E) → List E → E → Game (Option γ)) {count : ℕ}
    (lasts : ∀ nodes weights point claim, DrawsLe (last nodes weights point claim) count) :
    ∀ (arities : List ℕ) (nodes : ℕ) (weights : ℕ → E) (point : List E) (claim : E),
      DrawsLe (rangeTree leaf μ w last arities nodes weights point claim)
        (treeDraws μ count arities)
  | [], _, _, _, _ => lasts _ _ _ _
  | a :: rest, nodes, weights, point, claim => by
      simp only [rangeTree]
      refine DrawsLe.mono (n := μ + (treeDraws μ count rest + 1)) ?_ (by simp only [treeDraws]; omega)
      refine DrawsLe.bind (eqSumcheck_draws _ _ _ point μ [] claim) fun ended child => ?_
      exact DrawsLe.ite ⟨Nat.succ_pos _, fun γ => rangeTree_draws leaf μ w last lasts rest _ _ _ _⟩
        trivial

theorem leafStage_draws (leaf μ : ℕ) (support : ℕ → Prop) [DecidablePred support] (w : Table)
    (nodes : ℕ) (weights : ℕ → E) (point : List E) (claim : E) :
    DrawsLe (leafStage leaf μ support w nodes weights point claim) μ := by
  unfold leafStage
  exact (eqSumcheck_draws _ _ _ point μ [] claim).bind (m := 0) fun _ _ =>
    DrawsLe.ite trivial trivial

theorem rangeCheck_draws (b μ : ℕ) (support : ℕ → Prop) [DecidablePred support] (w : Table)
    (τ0 : List E) : DrawsLe (rangeCheck b μ support w τ0) (rangeDraws μ b) :=
  rangeTree_draws _ μ w _ (leafStage_draws _ μ support w) _ _ _ _ _

theorem normLeafStage_draws (leaf μ limit : ℕ) (kernel : ℕ → ℕ → Zq) (support : ℕ → Prop)
    [DecidablePred support] (w : Table) (nodes : ℕ) (weights : ℕ → E) (point : List E)
    (claim : E) :
    DrawsLe (normLeafStage leaf μ limit kernel support w nodes weights point claim) (μ + 1) := by
  unfold normLeafStage
  refine fun claimed => DrawsLe.ite ⟨Nat.succ_pos _, fun ζ => ?_⟩ trivial
  refine (sumcheckWith_draws _ _ _ μ [] _).bind (m := 0) fun result => ?_
  rcases result with _ | ⟨r, value⟩
  · exact trivial
  · exact fun _ => DrawsLe.ite trivial trivial

theorem fusedCheck_draws (μ : ℕ) (support : ℕ → Prop) [DecidablePred support] (w : Table)
    (weight : ℕ → E) (target : E) (r : List E) (sClaim ζ γ : E) :
    DrawsLe (fusedCheck μ support w weight target r sClaim ζ γ) μ := by
  unfold fusedCheck
  refine (sumcheck_draws _ _ μ [] _).bind (m := 0) fun result => ?_
  rcases result with _ | ⟨r2, value⟩
  · exact trivial
  · exact fun _ => DrawsLe.ite trivial trivial

omit [DecidableEq E] [Algebra Zq E] in
theorem drawBin_draws {γ : Type} (bin : Bool) (bad : Finset E) {next : E → Game γ} {n : ℕ}
    (nexts : ∀ ζ, DrawsLe (next ζ) n) : DrawsLe (drawBin bin bad next) (binDraws bin + n) := by
  cases bin
  · exact (nexts 0).mono (by simp [binDraws])
  · exact ⟨by simp [binDraws], fun ζ => (nexts ζ).mono (by simp [binDraws])⟩

/-- `foldCheck` draws `foldDraws` times at most: `α`, `τ_0`, `τ_1`, the range tree, `ζ_bin`, `γ`
and the rounds of the fused sum-check. -/
theorem foldCheck_draws (R : Reduction E) (b : ℕ) (support : ℕ → Prop) [DecidablePred support]
    (bin : Bool) (w : Table) : DrawsLe (foldCheck R b support bin w) (foldDraws R.vars b bin) := by
  unfold foldCheck
  refine ⟨by unfold foldDraws; omega, fun α => ⟨by unfold foldDraws; omega, fun τ0 =>
    ⟨by unfold foldDraws; omega, fun τ1 => ?_⟩⟩⟩
  refine ((rangeCheck_draws b R.vars support w _).bind (m := binDraws bin + (R.vars + 1))
    fun result => ?_).mono ?_
  · rcases result with _ | ⟨r, sClaim⟩
    · exact trivial
    · exact drawBin_draws bin _ fun ζ => ⟨Nat.succ_pos _, fun γ => fusedCheck_draws _ _ _ _ _ _ _ _ _⟩
  · unfold foldDraws
    omega

/-- `foldCheckSq` draws `foldDrawsSq` times at most: the draws of `foldCheck` with `ζ_norm` and
`η_norm`. -/
theorem foldCheckSq_draws (R : Reduction E) (b : ℕ) (support : ℕ → Prop) [DecidablePred support]
    (bin : Bool) (limit : ℕ) (kernel : ℕ → ℕ → Zq) (w : Table) :
    DrawsLe (foldCheckSq R b support bin limit kernel w) (Extract.foldDrawsSq R.vars b bin) := by
  unfold foldCheckSq
  refine ⟨by unfold Extract.foldDrawsSq; omega, fun α => ⟨by unfold Extract.foldDrawsSq; omega,
    fun τ0 => ⟨by unfold Extract.foldDrawsSq; omega, fun τ1 => ?_⟩⟩⟩
  refine ((rangeTree_draws _ R.vars w _ (normLeafStage_draws _ R.vars limit kernel support w) _ _ _
    _ _).bind (m := 1 + (binDraws bin + (R.vars + 1))) fun result => ?_).mono ?_
  · rcases result with _ | ⟨r, sClaim, y⟩
    · exact trivial
    · refine ⟨by omega, fun η => DrawsLe.mono (n := binDraws bin + (R.vars + 1)) ?_ (by omega)⟩
      exact drawBin_draws bin _ fun ζ => ⟨Nat.succ_pos _, fun γ => fusedCheck_draws _ _ _ _ _ _ _ _ _⟩
  · unfold Extract.foldDrawsSq
    omega

theorem tensorCheck_draws [Degree2 E] (ν : ℕ) (w : Table) :
    ∀ (point : List E) (v : E), DrawsLe (tensorCheck ν w point v) (ν + 1)
  | [], _ => trivial
  | _ :: _, _ => DrawsLe.ite (fun _ => DrawsLe.ite ⟨Nat.succ_pos _, fun _ =>
      (sumcheck_draws _ _ ν [] _).bind (m := 0) fun result => by
        rcases result with _ | ⟨ρ, final⟩
        · exact trivial
        · exact fun _ => DrawsLe.ite trivial trivial⟩ trivial) trivial

/-- A level with the tensor reduction of its final claim draws at most once for each of its
bounds when the level does. -/
theorem Tier.tensored_draws [Degree2 E] {S : Shape} {O : OpeningShape S} {fold : Descriptor}
    {Ring Stmt Brk Wire Coord Slot : Type}
    (T : Tier (Fin S.blocks) (fun _ => Ring) Stmt Brk (Witness O fold) (Weak O) Wire Coord Slot
      (Option (List E × E)) Table) (ν : ℕ) {wire : Wire} {c : Fin S.blocks → Ring}
    {source : Table} (drawn : DrawsLe (T.check wire c source) T.costs.length) :
    DrawsLe ((T.tensored ν).check wire c source) (T.tensored ν).costs.length := by
  show DrawsLe ((T.check wire c source).bind _) (T.costs ++ Extract.tensorCosts E ν).length
  rw [List.length_append, show (Extract.tensorCosts E ν).length = ν + 1 by
    simp [Extract.tensorCosts]]
  refine drawn.bind fun result => ?_
  rcases result with _ | ⟨r2, v⟩
  · exact trivial
  · exact tensorCheck_draws ν source r2 v

end Draws

section Fits

variable (E : Type) [Zero E] (width cap slots : ℕ)

/-- The path packs from the message `index` (`Extract.Packs`), and its next message index is at
most `next`. -/
def Fits (path : List Move) (index next : ℕ) : Prop :=
  Packs E width cap slots path index ∧ nextIndex path index ≤ next

/-- Each degree of the range tree and each width `nodes·a` of the claims of the children of a
product level is at most `width`. -/
def narrow : List ℕ → ℕ → Bool
  | [], _ => true
  | a :: rest, nodes => decide (a ≤ width) && decide (nodes * a ≤ width) && narrow rest (nodes * a)

variable {E width cap slots}

theorem packs_mono : ∀ {path : List Move} {index other : ℕ},
    Packs E width cap slots path index → other ≤ index → Packs E width cap slots path other
  | [], _, _, _, _ => trivial
  | Move.draw _ :: _, _, _, packs, _ => packs
  | Move.recv _ :: _, _, _, packs, le =>
      ⟨Nat.lt_of_le_of_lt le packs.1, packs.2.1, packs_mono packs.2.2 (Nat.succ_le_succ le)⟩

theorem nextIndex_mono : ∀ {path : List Move} {index other : ℕ}, other ≤ index →
    nextIndex path other ≤ nextIndex path index
  | [], _, _, le => le
  | Move.draw _ :: _, _, _, _ => le_rfl
  | Move.recv _ :: path, index, other, le => by
      show nextIndex path (other + 1) ≤ nextIndex path (index + 1)
      exact nextIndex_mono (Nat.succ_le_succ le)

theorem nextIndex_append : ∀ (first second : List Move) (index : ℕ),
    nextIndex (first ++ second) index = nextIndex second (nextIndex first index)
  | [], _, _ => rfl
  | Move.draw _ :: first, second, _ => nextIndex_append first second 0
  | Move.recv _ :: first, second, index => nextIndex_append first second (index + 1)

theorem Fits.mono {path : List Move} {index next index' next' : ℕ}
    (fits : Fits E width cap slots path index next) (le : index' ≤ index) (le' : next ≤ next') :
    Fits E width cap slots path index' next' :=
  ⟨packs_mono fits.1 le, (nextIndex_mono le).trans (fits.2.trans le')⟩

theorem fits_nil {index next : ℕ} (le : index ≤ next) : Fits E width cap slots [] index next :=
  ⟨trivial, le⟩

theorem fits_draw {C : Type} (c : C) {path : List Move} {index next : ℕ}
    (fits : Fits E width cap slots path 0 next) :
    Fits E width cap slots (Move.draw c :: path) index next :=
  fits

theorem fits_recv {M : Type} {m : M} {path : List Move} {index next : ℕ} (inside : index < slots)
    (sendable : Extract.Cell.Sendable E width cap M m)
    (fits : Fits E width cap slots path (index + 1) next) :
    Fits E width cap slots (Move.recv m :: path) index next :=
  ⟨⟨inside, sendable, fits.1⟩, fits.2⟩

theorem Fits.append {first second : List Move} {index middle next : ℕ}
    (left : Fits E width cap slots first index middle)
    (right : Fits E width cap slots second middle next) :
    Fits E width cap slots (first ++ second) index next := by
  refine ⟨(Extract.Packs.append first second index).mpr ⟨left.1, packs_mono right.1 left.2⟩, ?_⟩
  rw [nextIndex_append]
  exact (nextIndex_mono left.2).trans right.2

theorem fits_ite {p : Prop} [Decidable p] {yes no : List Move} {index next : ℕ}
    (first : Fits E width cap slots yes index next) (second : Fits E width cap slots no index next) :
    Fits E width cap slots (if p then yes else no) index next := by
  split
  · exact first
  · exact second

end Fits

section Paths

theorem fits_binPath {E : Type} [Field E] [Algebra Zq E] [Degree2 E] {width cap slots : ℕ}
    (bin : Bool) (ζ : E) {index : ℕ} : Fits E width cap slots (binPath bin ζ) index index := by
  unfold binPath
  exact fits_ite (fits_draw _ (fits_nil (Nat.zero_le _))) (fits_nil le_rfl)

variable {E : Type} [Field E] [Fintype E] {width cap slots : ℕ} (two : 2 ≤ width)
include two

theorem fits_plainPath (φ : Ext → E) (d : ℕ) (wide : d + 1 + 1 ≤ width) :
    ∀ (claim : Ext) (rounds : List (List Ext × Ext)) {index : ℕ}, index < slots →
      Fits E width cap slots (plainPath φ d claim rounds) index index
  | _, [], _, _ => fits_nil le_rfl
  | claim, (message, r) :: rounds, _, inside =>
      fits_recv inside ((Extract.Page.sends two).vector _ _ wide) (fits_draw _
        ((fits_plainPath φ d wide (plainStep claim message r) rounds
          (Nat.lt_of_le_of_lt (Nat.zero_le _) inside)).mono le_rfl (Nat.zero_le _)))

theorem fits_eqCodePath (φ : Ext → E) (D : ℕ) (wide : D ≤ width) :
    ∀ (rounds : List (List Ext × Ext)) {index : ℕ}, index < slots →
      Fits E width cap slots (eqCodePath φ D rounds) index index
  | [], _, _ => fits_nil le_rfl
  | _ :: rounds, _, inside =>
      fits_recv inside ((Extract.Page.sends two).vector _ _ wide) (fits_draw _
        ((fits_eqCodePath φ D wide rounds (Nat.lt_of_le_of_lt (Nat.zero_le _) inside)).mono le_rfl
          (Nat.zero_le _)))

/-- The path of the range tree fits from the message 0 when its last stage does and its degrees
and widths are at most `width`. -/
theorem fits_treePath (φ : Ext → E) (inside : 0 < slots)
    (last : ℕ → (ℕ → Ext) → List Ext → Ext → List Move)
    (lasts : ∀ nodes weights point claim, Fits E width cap slots (last nodes weights point claim) 0 1) :
    ∀ (stages : List Product) (nodes : ℕ) (weights : ℕ → Ext) (point : List Ext) (claim : Ext),
      narrow width (stages.map Product.arity) nodes = true →
      Fits E width cap slots (treePath φ last stages nodes weights point claim) 0 1
  | [], _, _, _, _, _ => lasts _ _ _ _
  | S :: rest, nodes, weights, point, claim, narrowed => by
      simp only [List.map_cons, narrow, Bool.and_eq_true, decide_eq_true_eq] at narrowed
      obtain ⟨⟨wideA, wideC⟩, narrowed⟩ := narrowed
      simp only [treePath]
      refine (fits_eqCodePath two φ S.arity wideA S.rounds inside).append
        (fits_recv inside ((Extract.Page.sends two).vector _ _ wideC) ?_)
      split
      · exact fits_nil le_rfl
      · split
        · exact fits_draw _ (fits_treePath φ inside last lasts rest _ _ _ _ narrowed)
        · exact fits_nil le_rfl

theorem fits_leafPath (φ : Ext → E) (leaf : ℕ) (wide : leaf ≤ width) (inside : 0 < slots)
    (rounds : List (List Ext × Ext)) (image : Ext) :
    Fits E width cap slots (leafPath φ leaf rounds image) 0 1 :=
  (fits_eqCodePath two φ leaf wide rounds inside).append
    (fits_recv inside ((Extract.Page.sends two).scalar _) (fits_nil le_rfl))

theorem fits_normPath (φ : Ext → E) (leaf limit energy : ℕ) (wide : leaf + 1 + 1 ≤ width)
    (inside : 0 < slots) (small : energy ≤ cap) (merge : Ext) (rounds : List (List Ext × Ext))
    (virtual image claim : Ext) :
    Fits E width cap slots (normPath φ leaf limit energy merge rounds virtual image claim) 0 1 :=
  fits_recv inside ((Extract.Page.sends two).count energy small) (fits_ite
    (fits_draw _ ((fits_plainPath two φ leaf wide _ rounds inside).append
      (fits_recv inside ((Extract.Page.sends two).pair _) (fits_nil le_rfl))))
    (fits_nil le_rfl))

variable [Algebra Zq E] [Degree2 E]

theorem fits_fusedPath (wide : 4 ≤ width) (claim : Ext) (rounds : List (List Ext × Ext))
    (finish : List Ext → FusedEnd) {index : ℕ} (inside : index < slots) :
    Fits E width cap slots (fusedPath E claim rounds finish) index (index + 1) :=
  (fits_plainPath two Ext.into 2 wide claim rounds inside).append
    (fits_recv inside ((Extract.Page.sends two).scalar _) (fits_nil le_rfl))

theorem fits_tensorPath (wide : 3 ≤ width) (ν : ℕ) (point : List Ext) (v p0 p1 η : Ext)
    (rounds : List (List Ext × Ext)) (final : Ext) {index : ℕ} (inside : index < slots) :
    Fits E width cap slots (tensorPath E ν point v p0 p1 η rounds final) index (index + 1) := by
  have first : 0 < slots := Nat.lt_of_le_of_lt (Nat.zero_le _) inside
  unfold tensorPath
  split
  · exact fits_nil (Nat.le_succ _)
  · exact fits_ite (fits_recv inside ((Extract.Page.sends two).pair _) (fits_ite (fits_draw _
      ((fits_plainPath two Ext.into 1 wide _ rounds first).append (fits_recv first
        ((Extract.Page.sends two).scalar _) (fits_nil (Nat.succ_le_succ (Nat.zero_le _))))))
      (fits_nil le_rfl))) (fits_nil (Nat.le_succ _))

theorem fits_tensoredPath (wide : 3 ≤ width) (ν : ℕ) (run : Option (List Ext × Ext))
    (p0 p1 η : Ext) (rounds : List (List Ext × Ext)) (final : Ext) {index : ℕ}
    (inside : index < slots) :
    Fits E width cap slots (tensoredPath E ν run p0 p1 η rounds final) index (index + 1) := by
  cases run with
  | none => exact fits_nil (Nat.le_succ _)
  | some claim => exact fits_tensorPath two wide ν claim.1 claim.2 p0 p1 η rounds final inside

/-- The path of `foldCheck` of the code fits from each message index. -/
theorem fits_foldPath (wide : 4 ≤ width) (inside : 1 < slots) (vars batch low b : ℕ) (bin : Bool)
    (W : FoldWords) (wideLeaf : rangeLeaf b ≤ width)
    (narrowed : narrow width (W.stages.map Product.arity) 1 = true)
    (leaf : List (List Ext × Ext)) (claim : Ext) (finish : List Ext → FusedEnd) {index : ℕ} :
    Fits E width cap slots (foldPath E vars batch low b bin W leaf claim finish) index 1 := by
  unfold foldPath
  refine fits_draw _ (fits_draw _ (fits_draw _ ((fits_treePath two Ext.into (by omega) _
    (fun _ _ _ _ => fits_leafPath two _ _ wideLeaf (by omega) _ _) _ _ _ _ _ narrowed).append ?_)))
  split
  · exact fits_nil le_rfl
  · exact (fits_binPath bin _).append (fits_draw _ (fits_fusedPath two wide claim W.rounds
      finish (by omega)))

/-- The path of `foldCheckSq` of the code fits from each message index when the energy is at
most `cap`. -/
theorem fits_foldPathSq (wide : 4 ≤ width) (inside : 1 < slots) (vars batch low leaf limit : ℕ)
    (bin : Bool) (W : FoldWords) (N : NormWords) (wideLeaf : leaf + 1 + 1 ≤ width)
    (small : N.energy ≤ cap) (narrowed : narrow width (W.stages.map Product.arity) 1 = true)
    (claim : Ext) (finish : List Ext → FusedEnd) {index : ℕ} :
    Fits E width cap slots (foldPathSq E vars batch low leaf limit bin W N claim finish) index 1 := by
  unfold foldPathSq
  refine fits_draw _ (fits_draw _ (fits_draw _ ((fits_treePath two Ext.into (by omega) _
    (fun _ _ _ _ => fits_normPath two _ _ _ _ wideLeaf (by omega) small _ _ _ _ _) _ _ _ _ _
      narrowed).append ?_)))
  split
  · exact fits_nil le_rfl
  · exact (fits_draw _ (fits_binPath bin _)).append (fits_draw _ (fits_fusedPath two wide claim
      W.rounds finish (by omega)))

end Paths

/-- The degrees of the products of the code are the degrees of the schedule. -/
theorem productsIn_arity (s : Replayer) (level : ℕ) (arities : List ℕ) :
    (productsIn s level arities).map Product.arity = arities := by
  simp only [productsIn, List.map_map]
  exact (List.map_congr_left fun _ _ => rfl).trans (List.zipIdx_map_fst 0 arities)

section Fold

variable {E : Type} [Field E] [Fintype E] [Algebra Zq E] [Degree2 E] {width cap slots : ℕ}

/-- The path of the fold of a level of the code fits from the message 0: the degrees of the range
tree and the leaf degrees fit `width`, and on the Euclidean route the energy is at most `cap`. -/
theorem fits_foldPathIn (two : 2 ≤ width) (wide : 4 ≤ width) (inside : 1 < slots) (pub : Public)
    (s : Replayer) (level : ℕ) (F : Fold) (setup : Array Fq) (vars batch : ℕ)
    (narrowed : narrow width ((productStages F.logBasis).map (·.1)) 1 = true)
    (wideLeaf : rangeLeaf (2 ^ F.logBasis) ≤ width)
    (wideNorm : leafDegree F.logBasis + 1 + 1 ≤ width)
    (small : F.euclidean = true → (normsIn s level).energy ≤ cap) :
    Fits E width cap slots (foldPathIn E pub s level F setup vars batch) 0 1 := by
  have arities := productsIn_arity s level ((productStages F.logBasis).map (·.1))
  unfold foldPathIn
  split
  · rename_i euclidean
    exact fits_foldPathSq two wide inside _ _ _ _ _ _ _ _ wideNorm (small euclidean)
      (by rw [show (wordsIn s level F).stages = productsIn s level
        ((productStages F.logBasis).map (·.1)) from rfl, arities]; exact narrowed) _ _
  · exact fits_foldPath two wide inside _ _ _ _ _ _ wideLeaf
      (by rw [show (wordsIn s level F).stages = productsIn s level
        ((productStages F.logBasis).map (·.1)) from rfl, arities]; exact narrowed) _ _ _

end Fold

end NightstreamFPrime.Spec.Compression.Akita.Replay
