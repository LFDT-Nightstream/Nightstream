import CompressionLean.Layer1.Linear
import CompressionLean.Layer1.Queries

namespace CompressionLean.Layer1

open Polynomial

variable {F K : Type} [Field F] [Field K] [Algebra F K]

local notation "ι" => algebraMap F K

noncomputable def runGroup (S : Setup F) (mu : K) (x : ℕ) : K :=
  rowColumn S x + mu * slotColumn S x + mu ^ 2 * coefColumn S x + mu ^ 3 * blockColumn S x

noncomputable def runValue (s : Shape) (S : Setup F) (o : EarlyOut F K) (mu : K) : K :=
  o.rowClaim + mu * (o.slotClaim s S + mu * (o.coefClaim + mu * o.embeddedBlock s))

noncomputable def runGroupPoly (s : Shape) (S : Setup F) (o : EarlyOut F K) : K[X] :=
  C (o.rowClaim - mle o.runPoint (rowColumn S)) +
    C (o.slotClaim s S - mle o.runPoint (slotColumn S)) * X +
    C (o.coefClaim - mle o.runPoint (coefColumn S)) * X ^ 2 +
    C (o.embeddedBlock s - mle o.runPoint (blockColumn S)) * X ^ 3

def padded (n : ℕ) (point : List K) : List K := point ++ List.replicate (n - point.length) 0

noncomputable def truePartials (G : ℕ → K) (point : List K) (u : ℕ) : K :=
  mle (point.drop 3) fun y => G (u + 2 ^ 3 * y)

structure HonestOut (K : Type) (t L : ℕ) where
  mu : K
  partials : Fin 2 → ℕ → K
  alpha : List K
  p1 : List (P1Tables K)
  queries : Fin t → Fin L

theorem runGroupPoly_eval (s : Shape) (S : Setup F) (o : EarlyOut F K) (mu : K) :
    (runGroupPoly s S o).eval mu = runValue s S o mu - mle o.runPoint (runGroup S mu) := by
  simp only [runGroupPoly, runValue, eval_add, eval_mul, eval_X, eval_C, eval_pow]
  unfold runGroup
  rw [mle_add, mle_add, mle_add, mle_mul_left, mle_mul_left, mle_mul_left]
  ring

theorem runGroupPoly_natDegree (s : Shape) (S : Setup F) (o : EarlyOut F K) :
    (runGroupPoly s S o).natDegree ≤ 3 := by
  unfold runGroupPoly
  refine (natDegree_add_le _ _).trans (max_le ?_ ?_)
  · refine (natDegree_add_le _ _).trans (max_le ?_ ?_)
    · refine (natDegree_add_le _ _).trans (max_le (by simp) ?_)
      exact (natDegree_C_mul_le _ _).trans (by simp)
    · exact (natDegree_C_mul_le _ _).trans (by simp)
  · exact (natDegree_C_mul_le _ _).trans (by simp)

theorem setupTrue_of_poly (s : Shape) (S : Setup F) (o : EarlyOut F K) (zero : runGroupPoly s S o = 0) :
    SetupTrue s S o := by
  have c0 := congrArg (fun p => p.coeff 0) zero
  have c1 := congrArg (fun p => p.coeff 1) zero
  have c2 := congrArg (fun p => p.coeff 2) zero
  have c3 := congrArg (fun p => p.coeff 3) zero
  simp only [runGroupPoly, coeff_add, coeff_C_mul_X_pow, coeff_C_mul_X, coeff_C, coeff_zero] at c0 c1 c2 c3
  norm_num at c0 c1 c2 c3
  exact ⟨sub_eq_zero.mp c0, sub_eq_zero.mp c1, sub_eq_zero.mp c2, sub_eq_zero.mp c3⟩

theorem mle_partials (p q : List K) (G : ℕ → K) :
    mle (p ++ q) G = mle p fun u => mle q fun y => G (u + 2 ^ p.length * y) := by
  rw [mle_append_split, mle_swap]

theorem mle_fold_partials (α q : List K) (G : ℕ → K) :
    mle q (fold α G) = mle α fun u => mle q fun y => G (u + 2 ^ α.length * y) := by
  have : fold α G = fun y => mle α fun u => G (u + 2 ^ α.length * y) := by
    funext y
    exact fold_shift α G y
  rw [this, mle_swap]

section Game

variable [Fintype K] [DecidableEq K]

noncomputable def drawPoint (bad : List K → Finset K) : ℕ → List K → Game (List K)
  | 0, point => .pure point
  | m + 1, point => .draw K (bad point) fun c => drawPoint bad m (point ++ [c])

omit [DecidableEq K] in
theorem drawPoint_avoids (bad : List K → Finset K) :
    ∀ (m : ℕ) (point : List K), (drawPoint bad m point).Avoids fun r =>
      ∃ cs, r = point ++ cs ∧ cs.length = m ∧ Path bad point cs := by
  intro m
  induction m with
  | zero => intro point; exact ⟨[], by simp, rfl, trivial⟩
  | succ m ih =>
      intro point c outside
      refine (ih (point ++ [c])).mono ?_
      rintro r ⟨cs, rfl, length, path⟩
      exact ⟨c :: cs, by simp, by simp [length], outside, path⟩

omit [DecidableEq K] in
theorem drawPoint_cost (bad : List K → Finset K) (E : ℕ) (card : ∀ point, (bad point).card ≤ E) :
    ∀ (m : ℕ) (point : List K), (drawPoint bad m point).CostLe (m * ((E : ℝ) / Fintype.card K)) := by
  intro m
  induction m with
  | zero => intro point; simp [drawPoint, Game.CostLe]
  | succ m ih =>
      intro point
      refine ⟨m * ((E : ℝ) / Fintype.card K), ?_, fun c => ih _⟩
      have positive : (0 : ℝ) < Fintype.card K := by exact_mod_cast Fintype.card_pos
      have : ((bad point).card : ℝ) / Fintype.card K ≤ (E : ℝ) / Fintype.card K :=
        div_le_div_of_nonneg_right (by exact_mod_cast card point) positive.le
      push_cast
      linarith

open Classical in
noncomputable def foldQueryBad (m t L : ℕ) (q : ℕ → K) (honestKey honestRun : ℕ → K) (v : P1Tables K) :
    Finset (Fin t → Fin L) :=
  if ∀ i, i < 2 ^ m → v.foldKey i = honestKey i ∧ v.foldRun i = honestRun i then ∅
  else queryBad t fun j => (univ m v.foldKey).eval (q j) = (univ m honestKey).eval (q j) ∧
      (univ m v.foldRun).eval (q j) = (univ m honestRun).eval (q j)

omit [Fintype K] in
theorem foldQueryBad_card (m t L P : ℕ) [NeZero L] (q : ℕ → K) (honestKey honestRun : ℕ → K)
    (v : P1Tables K) (distinct : ∀ i j, i < L → j < L → q i = q j → i = j) :
    ((Finset.univ.filter fun x : Fin t → Fin P => lowBits L x ∈ foldQueryBad m t L q honestKey honestRun v).card :
        ℝ) / Fintype.card (Fin t → Fin P) ≤
      ((((2 ^ m - 1) * (P / L + 1) : ℕ) : ℝ) / P) ^ t := by
  classical
  unfold foldQueryBad
  split
  · simp only [Finset.notMem_empty, Finset.filter_false, Finset.card_empty, Nat.cast_zero, zero_div]
    positivity
  · rename_i wrong
    apply lowBitsBad_density
    push Not at wrong
    obtain ⟨i, lt, differs⟩ := wrong
    have bound : ∀ (g h : ℕ → K), g i ≠ h i →
        ((Finset.range L).filter fun j => (univ m g).eval (q j) = (univ m h).eval (q j)).card ≤ 2 ^ m - 1 :=
      fun g h ne => agreement_card m g h ⟨i, lt, ne⟩ q L distinct
    have image : (Finset.univ.filter fun j : Fin L =>
        (univ m v.foldKey).eval (q j) = (univ m honestKey).eval (q j) ∧
          (univ m v.foldRun).eval (q j) = (univ m honestRun).eval (q j)).card ≤
        ((Finset.range L).filter fun j => (univ m v.foldKey).eval (q j) = (univ m honestKey).eval (q j)).card ∧
      (Finset.univ.filter fun j : Fin L =>
        (univ m v.foldKey).eval (q j) = (univ m honestKey).eval (q j) ∧
          (univ m v.foldRun).eval (q j) = (univ m honestRun).eval (q j)).card ≤
        ((Finset.range L).filter fun j => (univ m v.foldRun).eval (q j) = (univ m honestRun).eval (q j)).card := by
      constructor <;>
      · rw [← Finset.card_map Fin.valEmbedding]
        apply Finset.card_le_card
        intro j member
        simp only [Finset.mem_map, Finset.mem_filter, Finset.mem_univ, true_and, Fin.valEmbedding_apply] at member
        obtain ⟨k, agree, rfl⟩ := member
        simp [agree]
    by_cases key : v.foldKey i = honestKey i
    · convert image.2.trans (bound _ _ (differs key))
    · convert image.1.trans (bound _ _ key)

end Game

section Stage

variable [Fintype K] [DecidableEq K]

noncomputable def keyPoint (s : Shape) (lin : LinearOut K) : List K :=
  padded s.setupVars (lanePoint lin.zeta s.laneVars ++ lin.blockPoint)

noncomputable def alphaBad (s : Shape) (S : Setup F) (o : EarlyOut F K) (lin : LinearOut K) (mu : K)
    (partials : Fin 2 → ℕ → K) (point : List K) : Finset K :=
  foldBad (fun u => partials 0 u - truePartials (keyGroup s S lin.lam) (keyPoint s lin) u) 3 point ∪
    foldBad (fun u => partials 1 u - truePartials (runGroup S mu) (padded s.setupVars o.runPoint) u) 3 point

open Classical in
noncomputable def honestGame (s : Shape) (S : Setup F) (L1 L P : ℕ) [NeZero L] [NeZero P] (q : ℕ → K)
    (o : EarlyOut F K) (lin : LinearOut K) : Game (Option (HonestOut K s.queries L)) :=
  .draw K (zeros (runGroupPoly s S o)) fun mu =>
  .recv (Fin 2 → ℕ → K) fun partials =>
  Game.bind (drawPoint (alphaBad s S o lin mu partials) 3 []) fun alpha =>
    if mle ((keyPoint s lin).take 3) (partials 0) = lin.keyValue ∧
        mle ((padded s.setupVars o.runPoint).take 3) (partials 1) = runValue s S o mu then
      .recv {l : List (P1Tables K) // l.length ≤ L1} fun p1 =>
      .draw (Fin s.queries → Fin P) (Finset.univ.filter fun words => lowBits L words ∈
          Finset.univ.biUnion fun v : Fin p1.1.length =>
            foldQueryBad (s.setupVars - 3) s.queries L q (fold alpha (keyGroup s S lin.lam))
              (fold alpha (runGroup S mu)) (p1.1.get v))
        fun words => .pure (some ⟨mu, partials, alpha, p1.1, lowBits L words⟩)
    else .pure none

def HonestConsistent (s : Shape) (S : Setup F) (q : ℕ → K) (o : EarlyOut F K) (lin : LinearOut K)
    {L : ℕ} (h : HonestOut K s.queries L) (v : P1Tables K) : Prop :=
  mle ((keyPoint s lin).drop 3) v.foldKey = mle h.alpha (h.partials 0) ∧
    mle ((padded s.setupVars o.runPoint).drop 3) v.foldRun = mle h.alpha (h.partials 1) ∧
    ∀ j, mle (powerPoint (q (h.queries j)) (s.setupVars - 3)) v.foldKey =
        (univ (s.setupVars - 3) (fold h.alpha (keyGroup s S lin.lam))).eval (q (h.queries j)) ∧
      mle (powerPoint (q (h.queries j)) (s.setupVars - 3)) v.foldRun =
        (univ (s.setupVars - 3) (fold h.alpha (runGroup S h.mu))).eval (q (h.queries j))

theorem partials_true (G : ℕ → K) (point : List K) (alpha : List K) (w : ℕ → K)
    (alphaLength : alpha.length = 3)
    (path : Path (foldBad (fun u => w u - truePartials G point u) 3) [] alpha)
    (combined : mle alpha w = mle (point.drop 3) (fold alpha G)) :
    ∀ u, u < 2 ^ 3 → w u = truePartials G point u := by
  by_contra wrong
  push Not at wrong
  obtain ⟨u, lt, ne⟩ := wrong
  have nonzero := mle_ne_zero_of_path' (fun u => w u - truePartials G point u) alpha
    ⟨u, by rw [alphaLength]; exact lt, sub_ne_zero.mpr ne⟩ (by rw [alphaLength]; exact path)
  apply nonzero
  rw [mle_sub, combined, mle_fold_partials, alphaLength]
  unfold truePartials
  exact sub_self _

omit [Fintype K] [DecidableEq K] in
theorem value_of_partials (G : ℕ → K) (point : List K) (long : 3 ≤ point.length) :
    mle (point.take 3) (truePartials G point) = mle point G := by
  conv_rhs => rw [← List.take_append_drop 3 point]
  rw [mle_partials, List.length_take, min_eq_left long]
  rfl

theorem honest_world (s : Shape) (S : Setup F) (q : ℕ → K)
    (o : EarlyOut F K) (lin : LinearOut K) {L : ℕ} (h : HonestOut K s.queries L) (v : P1Tables K)
    (keyLong : 3 ≤ (keyPoint s lin).length) (runLong : 3 ≤ (padded s.setupVars o.runPoint).length)
    (keyDrop : ((keyPoint s lin).drop 3).length = s.setupVars - 3)
    (runDrop : ((padded s.setupVars o.runPoint).drop 3).length = s.setupVars - 3)
    (alphaLength : h.alpha.length = 3)
    (hμ : h.mu ∉ zeros (runGroupPoly s S o))
    (path : Path (alphaBad s S o lin h.mu h.partials) [] h.alpha)
    (hq : h.queries ∉ foldQueryBad (s.setupVars - 3) s.queries L q (fold h.alpha (keyGroup s S lin.lam))
      (fold h.alpha (runGroup S h.mu)) v)
    (check : mle ((keyPoint s lin).take 3) (h.partials 0) = lin.keyValue ∧
      mle ((padded s.setupVars o.runPoint).take 3) (h.partials 1) = runValue s S o h.mu)
    (consistent : HonestConsistent s S q o lin h v) :
    lin.keyValue = mle (lanePoint lin.zeta s.laneVars ++ lin.blockPoint) (keyGroup s S lin.lam) ∧
      SetupTrue s S o := by
  classical
  obtain ⟨keyFold, runFold, queried⟩ := consistent
  have honest : ∀ i, i < 2 ^ (s.setupVars - 3) →
      v.foldKey i = fold h.alpha (keyGroup s S lin.lam) i ∧ v.foldRun i = fold h.alpha (runGroup S h.mu) i := by
    by_contra wrong
    apply hq
    unfold foldQueryBad
    rw [ite_eq_right wrong]
    unfold queryBad
    rw [Fintype.mem_piFinset]
    intro j
    simp only [Finset.mem_filter, Finset.mem_univ, true_and]
    rw [univ_eval (s.setupVars - 3) v.foldKey, univ_eval (s.setupVars - 3) v.foldRun]
    exact queried j
  have keyPartials := partials_true (keyGroup s S lin.lam) (keyPoint s lin) h.alpha (h.partials 0)
    alphaLength (Path.mono_short 3 (fun q _ => Finset.subset_union_left) _ _ (by simp [alphaLength]) path)
    (by rw [← keyFold]; exact mle_congr _ fun i lt => (honest i (by rw [← keyDrop]; exact lt)).1)
  have runPartials := partials_true (runGroup S h.mu) (padded s.setupVars o.runPoint) h.alpha (h.partials 1)
    alphaLength (Path.mono_short 3 (fun q _ => Finset.subset_union_right) _ _ (by simp [alphaLength]) path)
    (by rw [← runFold]; exact mle_congr _ fun i lt => (honest i (by rw [← runDrop]; exact lt)).2)
  have keyValue : lin.keyValue = mle (keyPoint s lin) (keyGroup s S lin.lam) := by
    rw [← check.1, ← value_of_partials _ _ keyLong]
    exact mle_congr _ fun u lt => keyPartials u (by simpa [List.length_take, min_eq_left keyLong] using lt)
  have runTrue : runValue s S o h.mu = mle (padded s.setupVars o.runPoint) (runGroup S h.mu) := by
    rw [← check.2, ← value_of_partials _ _ runLong]
    exact mle_congr _ fun u lt => runPartials u (by simpa [List.length_take, min_eq_left runLong] using lt)
  refine ⟨by rw [keyValue, keyPoint, padded, mle_append_zeros], ?_⟩
  apply setupTrue_of_poly
  by_contra nonzero
  apply hμ
  unfold zeros
  rw [ite_eq_right nonzero]
  simp only [Finset.mem_filter, Finset.mem_univ, true_and]
  rw [runGroupPoly_eval, runTrue, padded, mle_append_zeros, sub_self]

end Stage

end CompressionLean.Layer1
