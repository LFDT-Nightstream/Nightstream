import NightstreamFPrime.Spec.Compression.Akita.Replay.LevelFused

/-! One fold of the replay after the commitment of the next witness as one function, and its
relation to the checks of the specification (Akita revision `d50f700`: `verify_fold` of
`fold/mod.rs` from the draw of `α` to `fused.end`; `foldCheck` of `Akita/FoldCheck.lean`,
`foldCheckSq` of `Akita/NormCheck.lean` and `Tier.tensored` of `Extract/TraceLink.lean`).

- `FoldWords` are the values that the checks of a fold read in the order of the code: `α`, `τ_0`,
  `τ_1`, the product stages of the range proof and `s_claim`, `ζ_bin`, `γ` and the rounds of the
  fused sum-check. `NormWords` are the values of the norm stage of the Euclidean route: `E_resp`,
  `ζ_norm`, the rounds, `z̃(r_virt)` and `η_norm`.
- `foldRun` is `rangeRun` and then `fusedRun`; `foldRunSq` is `treeRun` with `normRun` and then
  `fusedRun`. `foldPath` and `foldPathSq` are the paths of the games for the values of the code:
  the draws `α`, `τ_0`, `τ_1`, the path of the range proof and, when it accepts, `η_norm`, `ζ_bin`
  at a level with compression digits, `γ` and the path of the fused sum-check.
- `foldRun_runs` and `foldRunSq_runs`: for each reduction `R`, `foldCheck` and `foldCheckSq` on
  these paths end in the result of the code, when the claim of the fused sum-check is
  `V_{α,τ_1} + γ·s_claim` (with `η_norm·z̃(r_virt)` on the Euclidean route) and the values of the
  end check have the weight and the virtual weight of the game at each point (`weighs`,
  `supports`). A result `none` is a rejection of the game on the same path.
- `traceFoldRun_runs` and `traceFoldRunSq_runs` are these statements at a trace level for
  `traceReduction` of `traceStatementFor`, with the claim of `stage2` and the end check
  `traceFused` of the values of the replay: no hypothesis on the weights is left.
- `tensored_runs` joins a fold with the tensor reduction of the next level (`Tier.tensored`), and
  `traceTensored_runs` and `traceTensoredSq_runs` are it at a trace level. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita.Replay

open NightstreamFPrime.Spec.Compression.Lib NightstreamFPrime.Spec.Compression.Game
open NightstreamFPrime.Spec.Compression (Game)
open NightstreamFPrime.Spec.Compression.Akita

section Facts

theorem ofFn_listFn {K : Type} [Field K] {n : ℕ} {l : List K} (length : l.length = n) :
    List.ofFn (listFn n l) = l := by
  subst length
  refine List.ext_getElem (by simp) fun i _ inside => ?_
  simp [listFn, List.getD_eq_getElem?_getD, inside]

/-- The range point of the drawn `τ_0` is the range point of the code (`Before.turned`). -/
theorem rangePoint_listFn {K : Type} [Field K] (low : ℕ) {μ : ℕ} {l : List Ext} (φ : Ext → K)
    (length : l.length = μ) :
    rangePoint low (listFn μ (l.map φ)) = (l.rotate (l.length - low)).map φ := by
  rw [rangePoint, ofFn_listFn (by rw [List.length_map, length]), List.map_rotate, length]

theorem treeRun_some {β : Type} {last : ℕ → (ℕ → Ext) → List Ext → Ext → Option β} {out : β} :
    ∀ (stages : List Product) (nodes : ℕ) (weights : ℕ → Ext) (point : List Ext) (claim : Ext),
      treeRun last stages nodes weights point claim = some out →
        ∃ nodes weights point claim, last nodes weights point claim = some out
  | [], nodes, weights, point, claim, ran => ⟨nodes, weights, point, claim, ran⟩
  | S :: rest, nodes, weights, point, claim, ran => by
      cases ended : eqRun claim Ext.one point S.rounds with
      | none => simp [treeRun, ended] at ran
      | some e =>
          simp only [treeRun, ended] at ran
          split at ran
          · exact treeRun_some rest _ _ _ _ ran
          · simp at ran

/-- An accepted range proof ends in the challenges of the leaf rounds and `s_claim`. -/
theorem rangeRun_some {b : ℕ} {stages : List Product} {rounds : List (List Ext × Ext)}
    {image : Ext} {point : List Ext} {out : List Ext × Ext}
    (ran : rangeRun b stages rounds image point = some out) :
    out = (rounds.map (·.2), image) := by
  obtain ⟨nodes, weights, point, claim, last⟩ := treeRun_some _ _ _ _ _ ran
  unfold leafRun at last
  cases ended : eqRun claim Ext.one point rounds with
  | none => simp [ended] at last
  | some e =>
      simp only [ended] at last
      split at last
      · simp only [Option.some.injEq] at last
        rw [← last, eqRun_point _ _ _ _ ended]
      · simp at last

/-- An accepted norm stage ends in the challenges of its rounds, `s_claim` and `z̃(r_virt)`. -/
theorem normRun_some {leaf limit energy : ℕ} {merge : Ext} {rounds : List (List Ext × Ext)}
    {virtual image : Ext} {nodes : ℕ} {weights : ℕ → Ext} {point : List Ext} {claim : Ext}
    {out : List Ext × Ext × Ext}
    (ran : normRun leaf limit energy merge rounds virtual image nodes weights point claim =
      some out) :
    out = (rounds.map (·.2), image, virtual) := by
  unfold normRun at ran
  split at ran
  · dsimp only at ran
    split at ran
    · simp only [Option.some.injEq] at ran
      rw [← ran, plainRun_point]
    · simp at ran
  · simp at ran

variable {E : Type} [Field E] [Algebra Zq E] [Degree2 E]

/-- The draw of `ζ_bin` on the path of the code: the challenge at a level with compression
digits, and no move without them. -/
def binPath (bin : Bool) (ζ : E) : List Move := if bin then [Move.draw ζ] else []

omit [Algebra Zq E] [Degree2 E] in
theorem runs_drawBin [Fintype E] {β : Type} {bin : Bool} {bad : Finset E} {next : E → Game β}
    {ζ : E} {path : List Move} {result : β}
    (runs : (next (if bin then ζ else 0)).Runs path result) :
    (drawBin bin bad next).Runs (binPath bin ζ ++ path) result := by
  cases bin
  · exact runs
  · exact runs_draw runs

theorem batched_one (y η : Ext) : ((batched 1 [y] η).into : E) = y.into := by
  simp [batched, sumTo, powExt, Ext.into_add, Ext.into_mul, Ext.into_zero, Ext.into_one]

end Facts

/-- The values that the checks of one fold read, in the order of the code: `α`, `τ_0`, `τ_1`,
the product stages of the range proof, `s_claim` (`range.image`), `ζ_bin`, `γ`
(`stage2_batch`) and the rounds of the fused sum-check. -/
structure FoldWords where
  alpha : Ext
  tau0 : List Ext
  tau1 : List Ext
  stages : List Product
  image : Ext
  binary : Ext
  batch : Ext
  rounds : List (List Ext × Ext)

/-- The values of the norm stage of the Euclidean route: `E_resp`, `ζ_norm`, the rounds,
`z̃(r_virt)` and `η_norm` (`norm.virtual_batch`). -/
structure NormWords where
  energy : ℕ
  merge : Ext
  rounds : List (List Ext × Ext)
  virtual : Ext
  eta : Ext

namespace FoldWords

variable (W : FoldWords)

/-- The range point of the code: `τ_0` with its last `low` coordinates first. -/
def point (low : ℕ) : List Ext := W.tau0.rotate (W.tau0.length - low)

end FoldWords

/-- One fold on the coefficient route: the range proof with the leaf rounds `leaf`, and then the
fused sum-check from `claim` with the end check of `finish`. -/
def foldRun (b low : ℕ) (W : FoldWords) (leaf : List (List Ext × Ext)) (claim : Ext)
    (finish : List Ext → FusedEnd) : Option (List Ext × Ext) :=
  match rangeRun b W.stages leaf W.image (W.point low) with
  | none => none
  | some _ => fusedRun claim W.rounds finish

/-- One fold on the Euclidean route: the product stages with the norm stage `N` of the leaf
degree `leaf` and the cap `limit`, and then the fused sum-check. -/
def foldRunSq (leaf limit low : ℕ) (W : FoldWords) (N : NormWords) (claim : Ext)
    (finish : List Ext → FusedEnd) : Option (List Ext × Ext) :=
  match treeRun (normRun leaf limit N.energy N.merge N.rounds N.virtual W.image) W.stages 1
      (fun _ => Ext.one) (W.point low) Ext.zero with
  | none => none
  | some _ => fusedRun claim W.rounds finish

section Paths

variable (E : Type) [Field E] [Algebra Zq E] [Degree2 E]

/-- The path of `foldCheck` for the values of the code. -/
noncomputable def foldPath (vars batch low b : ℕ) (bin : Bool) (W : FoldWords)
    (leaf : List (List Ext × Ext)) (claim : Ext) (finish : List Ext → FusedEnd) : List Move :=
  Move.draw (W.alpha.into : E) :: Move.draw (listFn vars (W.tau0.map Ext.into) : Fin vars → E) ::
    Move.draw (listFn batch (W.tau1.map Ext.into) : Fin batch → E) ::
      (treePath (Ext.into (E := E))
          (fun _ _ _ _ => leafPath (Ext.into (E := E)) (rangeLeaf b) leaf W.image)
          W.stages 1 (fun _ => Ext.one) (W.point low) Ext.zero ++
        match rangeRun b W.stages leaf W.image (W.point low) with
        | none => []
        | some _ => binPath bin (W.binary.into : E) ++
            Move.draw (W.batch.into : E) :: fusedPath E claim W.rounds finish)

/-- The path of `foldCheckSq` for the values of the code. -/
noncomputable def foldPathSq (vars batch low leaf limit : ℕ) (bin : Bool) (W : FoldWords)
    (N : NormWords) (claim : Ext) (finish : List Ext → FusedEnd) : List Move :=
  Move.draw (W.alpha.into : E) :: Move.draw (listFn vars (W.tau0.map Ext.into) : Fin vars → E) ::
    Move.draw (listFn batch (W.tau1.map Ext.into) : Fin batch → E) ::
      (treePath (Ext.into (E := E)) (fun _ _ _ claim => normPath (Ext.into (E := E)) leaf limit
          N.energy N.merge N.rounds N.virtual W.image claim) W.stages 1 (fun _ => Ext.one)
          (W.point low)
          Ext.zero ++
        match treeRun (normRun leaf limit N.energy N.merge N.rounds N.virtual W.image) W.stages 1
            (fun _ => Ext.one) (W.point low) Ext.zero with
        | none => []
        | some _ => Move.draw (N.eta.into : E) :: binPath bin (W.binary.into : E) ++
            Move.draw (W.batch.into : E) :: fusedPath E claim W.rounds finish)

end Paths

variable {E : Type} [Field E] [Fintype E] [DecidableEq E] [Algebra Zq E] [Degree2 E]

/-- `foldCheck` of the specification, for each reduction `R`, on the path of the values of the
code ends in the result of `foldRun`. -/
theorem foldRun_runs (R : Reduction E) (b : ℕ) (support : ℕ → Prop) [DecidablePred support]
    (bin : Bool) (w : Table) (W : FoldWords) (leaf : List (List Ext × Ext)) (claim : Ext)
    (finish : List Ext → FusedEnd) (vars : W.tau0.length = R.vars)
    (batch : W.tau1.length = R.batch) (arities : W.stages.map (·.arity) = rangeArities b)
    (shaped : ∀ S ∈ W.stages, S.rounds.length = R.vars ∧ ∀ x ∈ S.rounds, x.1.length = S.arity)
    (count : leaf.length = R.vars) (lengths : ∀ x ∈ leaf, x.1.length = rangeLeaf b)
    (fusedCount : W.rounds.length = R.vars) (fusedLengths : ∀ x ∈ W.rounds, x.1.length = 3)
    (claimed : (claim.into : E) =
      (R.rows W.alpha.into).batchTarget (W.tau1.map Ext.into) + W.batch.into * W.image.into)
    (weighs : ∀ p : List Ext, p.length = R.vars →
      ((finish p).weight.into + (finish p).norm.into * (finish p).normWeight.into : E) =
        mle (p.map Ext.into) ((R.rows W.alpha.into).batchWeight (W.tau1.map Ext.into)))
    (supports : ∀ p : List Ext, p.length = R.vars →
      ((finish p).batch.into * (finish p).virt.into + (finish p).binary.into *
        (finish p).support.into : E) = mle (p.map Ext.into) (virtWeight support
          ((leaf.map (·.2)).map Ext.into) W.batch.into (if bin then W.binary.into else 0))) :
    (foldCheck R b support bin w).Runs (foldPath E R.vars R.batch R.low b bin W leaf claim finish)
      ((foldRun b R.low W leaf claim finish).map fun o => (o.1.map Ext.into, o.2.into)) := by
  have long : (W.point R.low).length = R.vars := by
    rw [FoldWords.point, List.length_rotate, vars]
  have range := rangeRun_runs (degree2_extHom (E := E)) (fun a b => same_degree2 (E := E) a b) b
    R.vars support w W.stages leaf W.image (W.point R.low) arities long shaped count lengths
  unfold foldCheck foldPath
  refine runs_draw (runs_draw (runs_draw ?_))
  rw [rangePoint_listFn R.low Ext.into vars]
  refine runs_bind range ?_
  unfold foldRun
  cases ran : rangeRun b W.stages leaf W.image (W.point R.low) with
  | none => rfl
  | some out =>
      obtain rfl := rangeRun_some ran
      refine runs_drawBin (runs_draw ?_)
      rw [ofFn_listFn (by rw [List.length_map, batch])]
      exact fusedRun_runs R.vars support w _ _ _ _ _ _ claim W.rounds finish fusedCount
        fusedLengths claimed weighs supports

/-- `foldCheckSq` of the specification, for each reduction `R`, on the path of the values of the
code ends in the result of `foldRunSq`. -/
theorem foldRunSq_runs (R : Reduction E) (b : ℕ) (support : ℕ → Prop) [DecidablePred support]
    (bin : Bool) (limit : ℕ) (kernel : ℕ → ℕ → Zq) (w : Table) (W : FoldWords) (N : NormWords)
    (claim : Ext) (finish : List Ext → FusedEnd) (positive : 1 ≤ rangeLeaf b)
    (small : limit < fieldModulus) (vars : W.tau0.length = R.vars)
    (batch : W.tau1.length = R.batch) (arities : W.stages.map (·.arity) = rangeArities b)
    (shaped : ∀ S ∈ W.stages, S.rounds.length = R.vars ∧ ∀ x ∈ S.rounds, x.1.length = S.arity)
    (count : N.rounds.length = R.vars)
    (lengths : ∀ x ∈ N.rounds, x.1.length = rangeLeaf b + 1)
    (fusedCount : W.rounds.length = R.vars) (fusedLengths : ∀ x ∈ W.rounds, x.1.length = 3)
    (claimed : (claim.into : E) =
      (R.rows W.alpha.into).batchTarget (W.tau1.map Ext.into) + N.eta.into * N.virtual.into +
        W.batch.into * W.image.into)
    (weighs : ∀ p : List Ext, p.length = R.vars →
      ((finish p).weight.into + (finish p).norm.into * (finish p).normWeight.into : E) =
        mle (p.map Ext.into) fun x =>
          (R.rows W.alpha.into).batchWeight (W.tau1.map Ext.into) x +
            N.eta.into * normWeight R.vars kernel ((N.rounds.map (·.2)).map Ext.into) x)
    (supports : ∀ p : List Ext, p.length = R.vars →
      ((finish p).batch.into * (finish p).virt.into + (finish p).binary.into *
        (finish p).support.into : E) = mle (p.map Ext.into) (virtWeight support
          ((N.rounds.map (·.2)).map Ext.into) W.batch.into (if bin then W.binary.into else 0))) :
    (foldCheckSq R b support bin limit kernel w).Runs
      (foldPathSq E R.vars R.batch R.low (rangeLeaf b) limit bin W N claim finish)
      ((foldRunSq (rangeLeaf b) limit R.low W N claim finish).map fun o =>
        (o.1.map Ext.into, o.2.into)) := by
  have long : (W.point R.low).length = R.vars := by
    rw [FoldWords.point, List.length_rotate, vars]
  have range := treeRun_runs (degree2_extHom (E := E)) (fun a b => same_degree2 (E := E) a b)
    (rangeLeaf b) R.vars w (fun o : List Ext × Ext × Ext => (o.1.map Ext.into, (o.2.1.into : E),
      (o.2.2.into : E)))
    (fun nodes weights point claim =>
      normLeafStage (rangeLeaf b) R.vars limit kernel support w nodes weights point claim)
    (normRun (rangeLeaf b) limit N.energy N.merge N.rounds N.virtual W.image)
    (fun _ _ _ claim => normPath Ext.into (rangeLeaf b) limit N.energy N.merge N.rounds N.virtual
      W.image claim)
    (fun nodes weights point claim long' => normRun_runs (degree2_extHom (E := E))
      (fun a b => same_degree2 (E := E) a b) (nat_degree2 (E := E)) (rangeLeaf b) R.vars limit
      kernel support w _ _ _ _ _ nodes weights point claim positive count long' lengths small)
    W.stages 1 (fun _ => Ext.one) (W.point R.low) Ext.zero long shaped
  rw [arities, Ext.into_one, Ext.into_zero] at range
  unfold foldCheckSq foldPathSq
  refine runs_draw (runs_draw (runs_draw ?_))
  rw [rangePoint_listFn R.low Ext.into vars]
  refine runs_bind range ?_
  unfold foldRunSq
  cases ran : treeRun (normRun (rangeLeaf b) limit N.energy N.merge N.rounds N.virtual W.image)
      W.stages 1 (fun _ => Ext.one) (W.point R.low) Ext.zero with
  | none => rfl
  | some out =>
      obtain ⟨nodes, weights, point, claim', last⟩ := treeRun_some _ _ _ _ _ ran
      obtain rfl := normRun_some last
      refine runs_draw (runs_drawBin (runs_draw ?_))
      rw [ofFn_listFn (by rw [List.length_map, batch])]
      exact fusedRun_runs R.vars support w _ _ _ _ _ _ claim W.rounds finish fusedCount
        fusedLengths claimed weighs supports

section Trace

variable {S : Shape} {O : OpeningShape S} {fold : Descriptor}

/-- The values of the checks of a fold for the values `V` of the end of the fused sum-check of the
replay at a trace level and the other values of the fold. -/
def TraceValues.words (V : TraceValues) (tau0 : List Ext) (stages : List Product) (image : Ext)
    (rounds : List (List Ext × Ext)) : FoldWords :=
  ⟨V.alpha, tau0, V.tau1, stages, image, V.binary, V.batch, rounds⟩

/-- The input claim of the fused sum-check of a level on the coefficient route
(`Replayer.stage2Of`): `stage2` with `rest = γ·s_claim`. -/
def fusedClaim (V : TraceValues) (data : RhsRun → Array Fq) (runs : List RhsRun) (opening : ℕ)
    (coefficients claims : Array Ext) (image : Ext) : Ext :=
  (stage2 V.tau1 V.alpha data runs opening coefficients claims (V.batch * image + Ext.zero)).fused

/-- The input claim of the fused sum-check of a level on the Euclidean route
(`Replayer.stage2Of`): `stage2` with `rest = γ·s_claim + η_norm·Σ_i η_norm^i·y_i` for the one
virtual evaluation `y`. -/
def fusedClaimSq (V : TraceValues) (data : RhsRun → Array Fq) (runs : List RhsRun)
    (opening : ℕ) (coefficients claims : Array Ext) (image virtual : Ext) : Ext :=
  (stage2 V.tau1 V.alpha data runs opening coefficients claims
    (V.batch * image + V.norm * batched 1 [virtual] V.norm)).fused

/-- The norm stage of the values of the replay: `η_norm` is the coefficient of `V`. -/
def TraceValues.norms (V : TraceValues) (energy : ℕ) (merge : Ext)
    (rounds : List (List Ext × Ext)) (virtual : Ext) : NormWords :=
  ⟨energy, merge, rounds, virtual, V.norm⟩

variable (O fold) in
/-- The fold of a trace level on the coefficient route: `foldRun` with the claim of `stage2` and
the end check `traceFused` of the values `V` of the replay. -/
def traceFoldRun (setup : Array Fq) (challenges : List (Array Fq)) (V : TraceValues) (b : ℕ)
    (tau0 : List Ext) (stages : List Product) (leaf : List (List Ext × Ext)) (image : Ext)
    (rounds : List (List Ext × Ext)) (data : RhsRun → Array Fq) (runs : List RhsRun)
    (opening : ℕ) (coefficients claims : Array Ext) : Option (List Ext × Ext) :=
  foldRun b O.low (V.words tau0 stages image rounds) leaf
    (fusedClaim V data runs opening coefficients claims image)
    (traceFinish O fold setup challenges V)

variable (O fold) in
/-- The fold of a trace level on the Euclidean route: `foldRunSq` with the claim of `stage2` and
the end check `traceFused` of the values `V` of the replay. -/
def traceFoldRunSq (setup : Array Fq) (challenges : List (Array Fq)) (V : TraceValues)
    (leaf limit : ℕ) (tau0 : List Ext) (stages : List Product) (image : Ext) (energy : ℕ)
    (merge : Ext) (normRounds : List (List Ext × Ext)) (virtual : Ext)
    (rounds : List (List Ext × Ext)) (data : RhsRun → Array Fq) (runs : List RhsRun)
    (opening : ℕ) (coefficients claims : Array Ext) : Option (List Ext × Ext) :=
  foldRunSq leaf limit O.low (V.words tau0 stages image rounds)
    (V.norms energy merge normRounds virtual)
    (fusedClaimSq V data runs opening coefficients claims image virtual)
    (traceFinish O fold setup challenges V)

omit [DecidableEq E] in
theorem fusedClaim_into (quarter : 4 * (S.innerDegree / 4) = S.innerDegree) (setup : Array Fq)
    (V : TraceValues) (data : RhsRun → Array Fq) (source : RhsSource)
    (coefficients claims : Array Ext) (image : Ext) (challenge : Fin S.blocks → Rq S.innerDegree)
    {μ : ℕ} (fits : O.cells fold ≤ 2 ^ μ) {batch : ℕ} (rowsFit : O.traceRowCount + 1 ≤ 2 ^ batch)
    (tau : V.tau1.length = batch) :
    ((fusedClaim V data
        (relation (fun _ : Fin 1 => S) O.openRank O.openDegree O.chain fun _ => source).1
        (relation (fun _ : Fin 1 => S) O.openRank O.openDegree O.chain fun _ => source).2
        coefficients claims image).into : E) =
      ((traceReduction (traceStatementFor O quarter setup V data source coefficients claims)
        challenge μ fits batch rowsFit).rows V.alpha.into).batchTarget (V.tau1.map Ext.into) +
        V.batch.into * image.into := by
  rw [fusedClaim, trace_fused_for quarter setup V data source coefficients claims _ challenge μ fits
    batch rowsFit V.tau1 V.alpha tau, Ext.into_add, Ext.into_mul, Ext.into_zero, add_zero]

omit [DecidableEq E] in
theorem fusedClaimSq_into (quarter : 4 * (S.innerDegree / 4) = S.innerDegree)
    (setup : Array Fq) (V : TraceValues) (data : RhsRun → Array Fq) (source : RhsSource)
    (coefficients claims : Array Ext) (image virtual : Ext)
    (challenge : Fin S.blocks → Rq S.innerDegree) {μ : ℕ} (fits : O.cells fold ≤ 2 ^ μ)
    {batch : ℕ} (rowsFit : O.traceRowCount + 1 ≤ 2 ^ batch) (tau : V.tau1.length = batch) :
    ((fusedClaimSq V data
        (relation (fun _ : Fin 1 => S) O.openRank O.openDegree O.chain fun _ => source).1
        (relation (fun _ : Fin 1 => S) O.openRank O.openDegree O.chain fun _ => source).2
        coefficients claims image virtual).into : E) =
      ((traceReduction (traceStatementFor O quarter setup V data source coefficients claims)
        challenge μ fits batch rowsFit).rows V.alpha.into).batchTarget (V.tau1.map Ext.into) +
        V.norm.into * virtual.into + V.batch.into * image.into := by
  rw [fusedClaimSq, trace_fused_for quarter setup V data source coefficients claims _ challenge μ
    fits batch rowsFit V.tau1 V.alpha tau, Ext.into_add, Ext.into_mul, Ext.into_mul, batched_one]
  ring

/-- A trace level on the coefficient route: `foldCheck` for `traceReduction` of the statement of
the words of the replay, on the path of the values of the code, ends in the result of
`traceFoldRun`. The values `V` are those of `fused.end` (`traceValuesIn`), `leaf` the rounds of
`range.leaf`, whose challenges are the point `r_virt` of `V`. -/
theorem traceFoldRun_runs (quarter : 4 * (S.innerDegree / 4) = S.innerDegree)
    (two : ((2 * (S.innerDegree / 4) : ℕ) : Zq) ≠ 0) (setup : Array Fq)
    (challenges : List (Array Fq)) (V : TraceValues) (tau0 : List Ext) (stages : List Product)
    (leaf : List (List Ext × Ext)) (image : Ext) (rounds : List (List Ext × Ext))
    (data : RhsRun → Array Fq) (source : RhsSource) (coefficients claims : Array Ext)
    (challenge : Fin S.blocks → Rq S.innerDegree)
    (challenged : ∀ i, challenge i =
      Poly.toRq S.innerDegree (challenges.getD i (Poly.zero S.innerDegree))) {μ : ℕ}
    (fits : O.cells fold ≤ 2 ^ μ) {batch : ℕ} (rowsFit : O.traceRowCount + 1 ≤ 2 ^ batch)
    (w : Table) (plain : V.euclidean = false) (compressed : V.compressed = O.compressed fold)
    (virtual : V.virtual = leaf.map (·.2)) (vars : tau0.length = μ) (tau : V.tau1.length = batch)
    (arities : stages.map (·.arity) = rangeArities S.certifiedBase)
    (shaped : ∀ P ∈ stages, P.rounds.length = μ ∧ ∀ x ∈ P.rounds, x.1.length = P.arity)
    (count : leaf.length = μ) (lengths : ∀ x ∈ leaf, x.1.length = rangeLeaf S.certifiedBase)
    (fusedCount : rounds.length = μ) (fusedLengths : ∀ x ∈ rounds, x.1.length = 3) :
    (foldCheck (traceReduction (traceStatementFor (E := E) O quarter setup V data source
        coefficients claims) challenge μ fits batch rowsFit) S.certifiedBase (O.binarySupport fold)
        (O.compressed fold) w).Runs
      (foldPath E μ batch O.low S.certifiedBase (O.compressed fold)
        (V.words tau0 stages image rounds) leaf (fusedClaim V data
          (relation (fun _ : Fin 1 => S) O.openRank O.openDegree O.chain fun _ => source).1
          (relation (fun _ : Fin 1 => S) O.openRank O.openDegree O.chain fun _ => source).2
          coefficients claims image) (traceFinish O fold setup challenges V))
      ((traceFoldRun O fold setup challenges V S.certifiedBase tau0 stages leaf image rounds data
        (relation (fun _ : Fin 1 => S) O.openRank O.openDegree O.chain fun _ => source).1
        (relation (fun _ : Fin 1 => S) O.openRank O.openDegree O.chain fun _ => source).2
        coefficients claims).map fun o => (o.1.map Ext.into, o.2.into)) := by
  have virtualLength : V.virtual.length = μ := by rw [virtual, List.length_map, count]
  refine foldRun_runs (traceReduction (traceStatementFor O quarter setup V data source coefficients
    claims) challenge μ fits batch rowsFit) S.certifiedBase (O.binarySupport fold)
    (O.compressed fold) w (V.words tau0 stages image rounds) leaf _
    (traceFinish O fold setup challenges V) vars tau arities shaped count lengths fusedCount
    fusedLengths (fusedClaim_into quarter setup V data source coefficients claims image challenge
      fits rowsFit tau) (fun p length => traceFinish_weighs quarter two setup challenges V _ _ _
      challenge challenged fits rowsFit tau plain p length) (fun p length => ?_)
  rw [traceFinish_supports setup challenges V fits virtualLength compressed p length, virtual]
  rfl

/-- A trace level on the Euclidean route: `foldCheckSq` for `traceReduction` of the statement of
the words of the replay, with the cap `limit` and the kernel `Shape.normKernel`, on the path of
the values of the code, ends in the result of `traceFoldRunSq`. `normRounds` are the rounds of
`norm.sumcheck`, whose challenges are the point `r_virt` of `V`. -/
theorem traceFoldRunSq_runs (quarter : 4 * (S.innerDegree / 4) = S.innerDegree)
    (two : ((2 * (S.innerDegree / 4) : ℕ) : Zq) ≠ 0) (setup : Array Fq)
    (challenges : List (Array Fq)) (V : TraceValues) (limit : ℕ) (tau0 : List Ext)
    (stages : List Product) (image : Ext) (energy : ℕ) (merge : Ext)
    (normRounds : List (List Ext × Ext)) (y : Ext) (rounds : List (List Ext × Ext))
    (data : RhsRun → Array Fq) (source : RhsSource) (coefficients claims : Array Ext)
    (challenge : Fin S.blocks → Rq S.innerDegree)
    (challenged : ∀ i, challenge i =
      Poly.toRq S.innerDegree (challenges.getD i (Poly.zero S.innerDegree))) {μ : ℕ}
    (fits : O.cells fold ≤ 2 ^ μ) {batch : ℕ} (rowsFit : O.traceRowCount + 1 ≤ 2 ^ batch)
    (w : Table) (positive : 1 ≤ rangeLeaf S.certifiedBase) (small : limit < fieldModulus)
    (digits : 0 < fold.responseDigits) (euclidean : V.euclidean = true)
    (compressed : V.compressed = O.compressed fold) (virtual : V.virtual = normRounds.map (·.2))
    (vars : tau0.length = μ) (tau : V.tau1.length = batch)
    (arities : stages.map (·.arity) = rangeArities S.certifiedBase)
    (shaped : ∀ P ∈ stages, P.rounds.length = μ ∧ ∀ x ∈ P.rounds, x.1.length = P.arity)
    (count : normRounds.length = μ)
    (lengths : ∀ x ∈ normRounds, x.1.length = rangeLeaf S.certifiedBase + 1)
    (fusedCount : rounds.length = μ) (fusedLengths : ∀ x ∈ rounds, x.1.length = 3) :
    (foldCheckSq (traceReduction (traceStatementFor (E := E) O quarter setup V data source
        coefficients claims) challenge μ fits batch rowsFit) S.certifiedBase (O.binarySupport fold)
        (O.compressed fold) limit (S.normKernel fold) w).Runs
      (foldPathSq E μ batch O.low (rangeLeaf S.certifiedBase) limit (O.compressed fold)
        (V.words tau0 stages image rounds) (V.norms energy merge normRounds y)
        (fusedClaimSq V data
          (relation (fun _ : Fin 1 => S) O.openRank O.openDegree O.chain fun _ => source).1
          (relation (fun _ : Fin 1 => S) O.openRank O.openDegree O.chain fun _ => source).2
          coefficients claims image y) (traceFinish O fold setup challenges V))
      ((traceFoldRunSq O fold setup challenges V (rangeLeaf S.certifiedBase) limit tau0 stages
        image energy merge normRounds y rounds data
        (relation (fun _ : Fin 1 => S) O.openRank O.openDegree O.chain fun _ => source).1
        (relation (fun _ : Fin 1 => S) O.openRank O.openDegree O.chain fun _ => source).2
        coefficients claims).map fun o => (o.1.map Ext.into, o.2.into)) := by
  have virtualLength : V.virtual.length = μ := by rw [virtual, List.length_map, count]
  refine foldRunSq_runs (traceReduction (traceStatementFor O quarter setup V data source
    coefficients claims) challenge μ fits batch rowsFit) S.certifiedBase (O.binarySupport fold)
    (O.compressed fold) limit (S.normKernel fold) w (V.words tau0 stages image rounds)
    (V.norms energy merge normRounds y) _ (traceFinish O fold setup challenges V) positive small
    vars tau arities shaped count lengths fusedCount fusedLengths
    (fusedClaimSq_into quarter setup V data source coefficients claims image y challenge fits
      rowsFit tau) (fun p length => ?_) (fun p length => ?_)
  · rw [traceFinish_weighsSq quarter two setup challenges V _ _ _ challenge challenged fits rowsFit
      digits virtualLength tau euclidean p length, virtual]
    rfl
  · rw [traceFinish_supports setup challenges V fits virtualLength compressed p length, virtual]
    rfl

end Trace

section Tensored

/-- The tensor reduction of the next level on the claim `(r_2, v_W)` of a fold with the result
`run`: the point `r_2` and the end of `tensorRun`. -/
def tensoredRun (ν : ℕ) (run : Option (List Ext × Ext)) (p0 p1 η : Ext)
    (rounds : List (List Ext × Ext)) (final : Ext) : Option (List Ext × Ended) :=
  run.bind fun claim => (tensorRun ν claim.1 claim.2 p0 p1 η rounds final).map fun out =>
    (claim.1, out)

variable (E) in
/-- The path of the tensor reduction of the next level after a fold with the result `run`. -/
noncomputable def tensoredPath (ν : ℕ) (run : Option (List Ext × Ext)) (p0 p1 η : Ext)
    (rounds : List (List Ext × Ext)) (final : Ext) : List Move :=
  run.elim [] fun claim => tensorPath E ν claim.1 claim.2 p0 p1 η rounds final

/-- A fold that ends in the result `run` of the code, then the tensor reduction of the next level
(`Tier.tensored`): the game ends in the result of `tensoredRun`, on the path of the fold and of
the tensor reduction of the code. -/
theorem tensored_runs {g : Game (Option (List E × E))} {path : List Move}
    {run : Option (List Ext × Ext)} (ν : ℕ) (w : Table) (p0 p1 η : Ext)
    (rounds : List (List Ext × Ext)) (final : Ext) (count : rounds.length = ν)
    (lengths : ∀ x ∈ rounds, x.1.length = 2)
    (runs : g.Runs path (run.map fun o => (o.1.map Ext.into, o.2.into))) :
    (g.bind fun
      | none => .pure none
      | some (r2, v) => tensorCheck ν w r2 v).Runs
      (path ++ tensoredPath E ν run p0 p1 η rounds final)
      ((tensoredRun ν run p0 p1 η rounds final).map fun x =>
        (x.2.point.map Ext.into,
          mle (x.2.point.map Ext.into) (tensorWeight (x.1.tail.map Ext.into) η.into),
          x.2.claim.into)) := by
  refine runs_bind runs ?_
  cases run with
  | none => rfl
  | some claim =>
      simp only [tensoredRun, Option.bind_some, Option.map_map]
      exact tensorRun_runs ν w claim.1 claim.2 p0 p1 η rounds final count lengths

variable {S : Shape} {O : OpeningShape S} {fold : Descriptor}

/-- `Tier.tensored` of a trace level on the coefficient route (`committedTrace`): the fold of
`traceFoldRun_runs`, then the tensor reduction of the next level on its claim. -/
theorem traceTensored_runs (quarter : 4 * (S.innerDegree / 4) = S.innerDegree)
    (two : ((2 * (S.innerDegree / 4) : ℕ) : Zq) ≠ 0) (setup : Array Fq)
    (challenges : List (Array Fq)) (V : TraceValues) (tau0 : List Ext) (stages : List Product)
    (leaf : List (List Ext × Ext)) (image : Ext) (rounds : List (List Ext × Ext))
    (data : RhsRun → Array Fq) (source : RhsSource) (coefficients claims : Array Ext)
    (challenge : Fin S.blocks → Rq S.innerDegree)
    (challenged : ∀ i, challenge i =
      Poly.toRq S.innerDegree (challenges.getD i (Poly.zero S.innerDegree))) {μ : ℕ}
    (fits : O.cells fold ≤ 2 ^ μ) {batch : ℕ} (rowsFit : O.traceRowCount + 1 ≤ 2 ^ batch)
    (w : Table) (plain : V.euclidean = false) (compressed : V.compressed = O.compressed fold)
    (virtual : V.virtual = leaf.map (·.2)) (vars : tau0.length = μ) (tau : V.tau1.length = batch)
    (arities : stages.map (·.arity) = rangeArities S.certifiedBase)
    (shaped : ∀ P ∈ stages, P.rounds.length = μ ∧ ∀ x ∈ P.rounds, x.1.length = P.arity)
    (count : leaf.length = μ) (lengths : ∀ x ∈ leaf, x.1.length = rangeLeaf S.certifiedBase)
    (fusedCount : rounds.length = μ) (fusedLengths : ∀ x ∈ rounds, x.1.length = 3) (ν : ℕ)
    (p0 p1 η : Ext) (tensorRounds : List (List Ext × Ext)) (final : Ext)
    (tensorCount : tensorRounds.length = ν) (tensorLengths : ∀ x ∈ tensorRounds, x.1.length = 2) :
    ((foldCheck (traceReduction (traceStatementFor (E := E) O quarter setup V data source
        coefficients claims) challenge μ fits batch rowsFit) S.certifiedBase (O.binarySupport fold)
        (O.compressed fold) w).bind fun
      | none => .pure none
      | some (r2, v) => tensorCheck ν w r2 v).Runs
      (foldPath E μ batch O.low S.certifiedBase (O.compressed fold)
        (V.words tau0 stages image rounds) leaf (fusedClaim V data
          (relation (fun _ : Fin 1 => S) O.openRank O.openDegree O.chain fun _ => source).1
          (relation (fun _ : Fin 1 => S) O.openRank O.openDegree O.chain fun _ => source).2
          coefficients claims image) (traceFinish O fold setup challenges V) ++
        tensoredPath E ν (traceFoldRun O fold setup challenges V S.certifiedBase tau0 stages leaf
          image rounds data
          (relation (fun _ : Fin 1 => S) O.openRank O.openDegree O.chain fun _ => source).1
          (relation (fun _ : Fin 1 => S) O.openRank O.openDegree O.chain fun _ => source).2
          coefficients claims) p0 p1 η tensorRounds final)
      ((tensoredRun ν (traceFoldRun O fold setup challenges V S.certifiedBase tau0 stages leaf
          image rounds data
          (relation (fun _ : Fin 1 => S) O.openRank O.openDegree O.chain fun _ => source).1
          (relation (fun _ : Fin 1 => S) O.openRank O.openDegree O.chain fun _ => source).2
          coefficients claims) p0 p1 η tensorRounds final).map fun x =>
        (x.2.point.map Ext.into,
          mle (x.2.point.map Ext.into) (tensorWeight (x.1.tail.map Ext.into) η.into),
          x.2.claim.into)) :=
  tensored_runs ν w p0 p1 η tensorRounds final tensorCount tensorLengths
    (traceFoldRun_runs quarter two setup challenges V tau0 stages leaf image rounds data source
      coefficients claims challenge challenged fits rowsFit w plain compressed virtual vars tau
      arities shaped count lengths fusedCount fusedLengths)

/-- `Tier.tensored` of a trace level on the Euclidean route (`committedTraceSq`): the fold of
`traceFoldRunSq_runs`, then the tensor reduction of the next level on its claim. -/
theorem traceTensoredSq_runs (quarter : 4 * (S.innerDegree / 4) = S.innerDegree)
    (two : ((2 * (S.innerDegree / 4) : ℕ) : Zq) ≠ 0) (setup : Array Fq)
    (challenges : List (Array Fq)) (V : TraceValues) (limit : ℕ) (tau0 : List Ext)
    (stages : List Product) (image : Ext) (energy : ℕ) (merge : Ext)
    (normRounds : List (List Ext × Ext)) (y : Ext) (rounds : List (List Ext × Ext))
    (data : RhsRun → Array Fq) (source : RhsSource) (coefficients claims : Array Ext)
    (challenge : Fin S.blocks → Rq S.innerDegree)
    (challenged : ∀ i, challenge i =
      Poly.toRq S.innerDegree (challenges.getD i (Poly.zero S.innerDegree))) {μ : ℕ}
    (fits : O.cells fold ≤ 2 ^ μ) {batch : ℕ} (rowsFit : O.traceRowCount + 1 ≤ 2 ^ batch)
    (w : Table) (positive : 1 ≤ rangeLeaf S.certifiedBase) (small : limit < fieldModulus)
    (digits : 0 < fold.responseDigits) (euclidean : V.euclidean = true)
    (compressed : V.compressed = O.compressed fold) (virtual : V.virtual = normRounds.map (·.2))
    (vars : tau0.length = μ) (tau : V.tau1.length = batch)
    (arities : stages.map (·.arity) = rangeArities S.certifiedBase)
    (shaped : ∀ P ∈ stages, P.rounds.length = μ ∧ ∀ x ∈ P.rounds, x.1.length = P.arity)
    (count : normRounds.length = μ)
    (lengths : ∀ x ∈ normRounds, x.1.length = rangeLeaf S.certifiedBase + 1)
    (fusedCount : rounds.length = μ) (fusedLengths : ∀ x ∈ rounds, x.1.length = 3) (ν : ℕ)
    (p0 p1 η : Ext) (tensorRounds : List (List Ext × Ext)) (final : Ext)
    (tensorCount : tensorRounds.length = ν) (tensorLengths : ∀ x ∈ tensorRounds, x.1.length = 2) :
    ((foldCheckSq (traceReduction (traceStatementFor (E := E) O quarter setup V data source
        coefficients claims) challenge μ fits batch rowsFit) S.certifiedBase (O.binarySupport fold)
        (O.compressed fold) limit (S.normKernel fold) w).bind fun
      | none => .pure none
      | some (r2, v) => tensorCheck ν w r2 v).Runs
      (foldPathSq E μ batch O.low (rangeLeaf S.certifiedBase) limit (O.compressed fold)
        (V.words tau0 stages image rounds) (V.norms energy merge normRounds y)
        (fusedClaimSq V data
          (relation (fun _ : Fin 1 => S) O.openRank O.openDegree O.chain fun _ => source).1
          (relation (fun _ : Fin 1 => S) O.openRank O.openDegree O.chain fun _ => source).2
          coefficients claims image y) (traceFinish O fold setup challenges V) ++
        tensoredPath E ν (traceFoldRunSq O fold setup challenges V (rangeLeaf S.certifiedBase) limit
          tau0 stages image energy merge normRounds y rounds data
          (relation (fun _ : Fin 1 => S) O.openRank O.openDegree O.chain fun _ => source).1
          (relation (fun _ : Fin 1 => S) O.openRank O.openDegree O.chain fun _ => source).2
          coefficients claims) p0 p1 η tensorRounds final)
      ((tensoredRun ν (traceFoldRunSq O fold setup challenges V (rangeLeaf S.certifiedBase) limit
          tau0 stages image energy merge normRounds y rounds data
          (relation (fun _ : Fin 1 => S) O.openRank O.openDegree O.chain fun _ => source).1
          (relation (fun _ : Fin 1 => S) O.openRank O.openDegree O.chain fun _ => source).2
          coefficients claims) p0 p1 η tensorRounds final).map fun x =>
        (x.2.point.map Ext.into,
          mle (x.2.point.map Ext.into) (tensorWeight (x.1.tail.map Ext.into) η.into),
          x.2.claim.into)) :=
  tensored_runs ν w p0 p1 η tensorRounds final tensorCount tensorLengths
    (traceFoldRunSq_runs quarter two setup challenges V limit tau0 stages image energy merge
      normRounds y rounds data source coefficients claims challenge challenged fits rowsFit w
      positive small digits euclidean compressed virtual vars tau arities shaped count lengths
      fusedCount fusedLengths)

end Tensored

end NightstreamFPrime.Spec.Compression.Akita.Replay
