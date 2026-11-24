import NightstreamFPrime.Spec.Compression.Akita.Replay.RunStep

/-! The level functions of `Replay/LevelIn.lean` on the state of a run of the operations of a fold
(Akita revision `d50f700`, `verify_fold` of `fold/mod.rs`, `Build.fold` of `Replay/Script.lean`).

- `isRangeCheck`, `isFusedClaim`, `isPlainEnd`, `isStage3End` and `isSetupClaim` recognize the
  operations of a fold; `Fold.sameClaim` compares the fields of a fold that `stage2Of` reads.
- `frees l K R`: no operation of `l` writes a key of `K` or ends a sum-check of a key of `R`.
- `rangeKeys` and `rangeRounds`, `stage2Keys`, `stage3Keys` are the keys that the stage function
  of the last range check, `stage2In` and `stage3In` read (`stageValue_range_frame`,
  `stage2In_frame`, `stage3In_frame`).
- `foldShape` is the shape of the operations of a fold before its `fused.end` check: the last
  check of the range proof, the absorb of `fused.claim`, the end of `fused`, and the frames after
  them. `foldRunIn_of_run`: when the run of these operations and of `fused.end` has no stop,
  `foldRunIn` on the state at `fused.end` is the recorded `(fused.point, witness_evaluation)`.
  `offloadRunIn_of_run` is the same at an offloaded level with `stage3.end`.
- `foldRunIn_of_reject` and `foldRunIn_of_range_reject`: a reject at `fused.end`, or at the last
  check of the range proof, gives `foldRunIn = none` on the state of the stop. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita.Replay

/-- The fields of a fold that `stage2Of` reads. -/
def Fold.sameClaim (G F : Fold) : Bool :=
  G.rhs == F.rhs && G.openingRow == F.openingRow &&
    decide (1 < G.groups.length) == decide (1 < F.groups.length) &&
    G.trace.isSome == F.trace.isSome && G.euclidean == F.euclidean

/-- The last check of the range proof of `F` at `level` (`rangeCheckOf`). -/
def isRangeCheck (level : ℕ) (F : Fold) : Op → Bool
  | .check _ (.range lv b ar (.turned k low)) =>
      !F.euclidean && lv == level && b == 2 ^ F.logBasis &&
        ar == (productStages F.logBasis).map (·.1) && k == ⟨level, "tau0", none⟩ && low == F.low
  | .check _ (.rangeNorm lv leaf limit ar (.turned k low)) =>
      F.euclidean && lv == level && leaf == leafDegree F.logBasis && limit == F.limit &&
        ar == (productStages F.logBasis).map (·.1) && k == ⟨level, "tau0", none⟩ && low == F.low
  | _ => false

/-- The absorb of the claim of the fused sum-check of a fold with the claim of `F`. -/
def isFusedClaim (level : ℕ) (F : Fold) : Op → Bool
  | .absorb k (.fused G) => k == ⟨level, "fused.claim", none⟩ && G.sameClaim F
  | _ => false

/-- The end of the plain sum-check `key` with `n` rounds. -/
def isPlainEnd (key : Key) (n : ℕ) : Op → Bool
  | .ended k m .plain => k == key && m == n
  | _ => false

/-- The end of Stage 3 of the level `level` for the plan `plan`. -/
def isStage3End (level : ℕ) (plan : SetupPlan) : Op → Bool
  | .check _ (.stage3 lv p) => lv == level && p == plan
  | _ => false

/-- The absorb of the claim of Stage 3: the received `stage3.setup_claim`. -/
def isSetupClaim (level : ℕ) : Op → Bool
  | .absorb k (.recorded keys) =>
      k == ⟨level, "stage3.claim", none⟩ && keys == [⟨level, "stage3.setup_claim", none⟩]
  | _ => false

/-- No operation of `l` writes a key of `K` or ends a sum-check of a key of `R`. -/
def frees (l : List (Op × ℕ)) (K R : List Key) : Bool :=
  decide (∀ x ∈ l, (∀ k ∈ x.1.writes, k ∉ K) ∧ ∀ k ∈ x.1.ends, k ∉ R)

/-- The keys of the product stages of the range proof. -/
def productKeys (level count : ℕ) : List Key :=
  (List.range count).flatMap fun k =>
    [⟨level, s!"range.{k}.children", none⟩, ⟨level, s!"range.{k}.batch", none⟩]

/-- The keys that the stage function of the last check of the range proof reads. -/
def rangeKeys (level : ℕ) (F : Fold) : List Key :=
  ⟨level, "tau0", none⟩ :: ⟨level, "range.image", none⟩ :: ⟨level, "norm.response_l2_sq", none⟩ ::
    ⟨level, "norm.merge", none⟩ :: ⟨level, "norm.virtual_evaluations", none⟩ ::
      productKeys level (productStages F.logBasis).length

/-- The sum-checks whose rounds the stage function of the last check of the range proof reads. -/
def rangeRounds (level : ℕ) (F : Fold) : List Key :=
  ⟨level, "range.leaf", none⟩ :: ⟨level, "norm.sumcheck", none⟩ ::
    (List.range (productStages F.logBasis).length).map fun k => ⟨level, s!"range.{k}", none⟩

/-- The keys that `stage2In` reads. -/
def stage2Keys (level : ℕ) : List Key :=
  [⟨level, "tau1", none⟩, ⟨level, "alpha", none⟩, ⟨level, "row_coefficients", none⟩,
    ⟨level, "tensor.final_claims", none⟩, ⟨level, "openings", none⟩,
    ⟨level, "stage2_batch", none⟩, ⟨level, "range.image", none⟩,
    ⟨level, "norm.virtual_batch", none⟩, ⟨level, "norm.virtual_evaluations", none⟩,
    ⟨level, "opening_payload", none⟩, ⟨level - 1, "next_witness", none⟩]

/-- The keys that `stage3In` reads. -/
def stage3Keys (level : ℕ) : List Key :=
  [⟨level, "stage3.claim", none⟩, ⟨level, "stage3.prefix_evaluation", none⟩,
    ⟨level, "alpha", none⟩, ⟨level, "tau1", none⟩, ⟨level, "fused.point", none⟩]

/-- The shape of the operations `pre` of a fold before its `fused.end` check: the last check of
the range proof at `r`, the absorb of `fused.claim` at `a` and the end of `fused` at `e`, in this
order; no later operation writes what they read, `fused.claim` and `fused.output` have one
writer, no operation before `e` ends `fused`, and the operations before `e` append `F.vars`
challenges to `fused.point`. -/
def foldShape (pre : List (Op × ℕ)) (level : ℕ) (F : Fold) (r a e : ℕ) : Bool :=
  decide (r < a) && decide (a < e) &&
    (pre[r]?.any fun x => isRangeCheck level F x.1) &&
    (pre[a]?.any fun x => isFusedClaim level F x.1) &&
    (pre[e]?.any fun x => isPlainEnd ⟨level, "fused", none⟩ F.vars x.1) &&
    frees (pre.drop r) (rangeKeys level F) (rangeRounds level F) &&
    frees (pre.drop a) (stage2Keys level) [] &&
    frees (pre.take a) [⟨level, "fused.claim", none⟩] [] &&
    frees (pre.drop (a + 1)) [⟨level, "fused.claim", none⟩] [] &&
    frees (pre.take e) [⟨level, "fused.output", none⟩] [⟨level, "fused", none⟩] &&
    frees (pre.drop (e + 1)) [⟨level, "fused.point", none⟩, ⟨level, "fused.output", none⟩] [] &&
    drawnCount ⟨level, "fused.point", none⟩ (pre.take e) == some F.vars

/-- The shape of the operations `pre` of an offloaded fold before its `fused.end` check:
`foldShape`, the coefficient route, the absorb of `stage3.claim` at `b` from the one writer of
`stage3.setup_claim`, and the check `stage3.end` at `t`, with no later write of what `stage3In`
reads. -/
def offloadShape (pre : List (Op × ℕ)) (level : ℕ) (F : Fold) (plan : SetupPlan)
    (r a e b t : ℕ) : Bool :=
  foldShape pre level F r a e && !F.euclidean &&
    (pre[b]?.any fun x => isSetupClaim level x.1) &&
    (pre[t]?.any fun x => isStage3End level plan x.1) &&
    frees (pre.take b) [⟨level, "stage3.claim", none⟩] [] &&
    frees (pre.drop (b + 1)) [⟨level, "stage3.claim", none⟩] [] &&
    frees (pre.drop b) [⟨level, "stage3.setup_claim", none⟩] [] &&
    frees (pre.drop t) (stage3Keys level) [⟨level, "stage3", none⟩]

theorem any_at {l : List (Op × ℕ)} {j : ℕ} {p : Op × ℕ → Bool} (h : l[j]?.any p = true) :
    ∃ inside : j < l.length, p l[j] = true := by
  obtain ⟨x, found, holds⟩ := (Option.any_eq_true p _).mp h
  obtain ⟨inside, same⟩ := List.getElem?_eq_some_iff.mp found
  exact ⟨inside, same ▸ holds⟩

namespace Replayer

theorem isRangeCheck_sound {level : ℕ} {F : Fold} {op : Op} (h : isRangeCheck level F op = true) :
    ∃ key, op = .check key (rangeCheckOf level F) := by
  unfold isRangeCheck at h
  split at h
  next key lv b ar k low =>
    simp only [Bool.and_eq_true, beq_iff_eq, Bool.not_eq_true'] at h
    obtain ⟨⟨⟨⟨⟨plain, rfl⟩, rfl⟩, rfl⟩, rfl⟩, rfl⟩ := h
    exact ⟨key, by rw [rangeCheckOf_plain _ F plain]⟩
  next key lv leaf limit ar k low =>
    simp only [Bool.and_eq_true, beq_iff_eq] at h
    obtain ⟨⟨⟨⟨⟨⟨euclidean, rfl⟩, rfl⟩, rfl⟩, rfl⟩, rfl⟩, rfl⟩ := h
    exact ⟨key, by rw [rangeCheckOf_euclidean _ F euclidean]⟩
  next => cases h

theorem isFusedClaim_sound {level : ℕ} {F : Fold} {op : Op} (h : isFusedClaim level F op = true) :
    ∃ G : Fold, op = .absorb ⟨level, "fused.claim", none⟩ (.fused G) ∧ G.sameClaim F = true := by
  unfold isFusedClaim at h
  split at h
  next k G =>
    simp only [Bool.and_eq_true, beq_iff_eq] at h
    obtain ⟨rfl, same⟩ := h
    exact ⟨G, rfl, same⟩
  next => cases h

theorem isPlainEnd_sound {key : Key} {n : ℕ} {op : Op} (h : isPlainEnd key n op = true) :
    op = .ended key n .plain := by
  unfold isPlainEnd at h
  split at h
  next k m =>
    simp only [Bool.and_eq_true, beq_iff_eq] at h
    obtain ⟨rfl, rfl⟩ := h
    rfl
  next => cases h

theorem isStage3End_sound {level : ℕ} {plan : SetupPlan} {op : Op}
    (h : isStage3End level plan op = true) : ∃ key, op = .check key (.stage3 level plan) := by
  unfold isStage3End at h
  split at h
  next key lv p =>
    simp only [Bool.and_eq_true, beq_iff_eq] at h
    obtain ⟨rfl, rfl⟩ := h
    exact ⟨key, rfl⟩
  next => cases h

theorem isSetupClaim_sound {level : ℕ} {op : Op} (h : isSetupClaim level op = true) :
    op = .absorb ⟨level, "stage3.claim", none⟩ (.recorded [⟨level, "stage3.setup_claim", none⟩]) := by
  unfold isSetupClaim at h
  split at h
  next k keys =>
    simp only [Bool.and_eq_true, beq_iff_eq] at h
    obtain ⟨rfl, rfl⟩ := h
    rfl
  next => cases h

theorem frees_writes {l : List (Op × ℕ)} {K R : List Key} (h : frees l K R = true) {k : Key}
    (inside : k ∈ K) : ∀ y ∈ l, k ∉ y.1.writes := fun y member written =>
  ((of_decide_eq_true h) y member).1 k written inside

theorem frees_ends {l : List (Op × ℕ)} {K R : List Key} (h : frees l K R = true) {k : Key}
    (inside : k ∈ R) : ∀ y ∈ l, k ∉ y.1.ends := fun y member ended =>
  ((of_decide_eq_true h) y member).2 k ended inside

theorem frees_drop {l : List (Op × ℕ)} {K R : List Key} (h : frees l K R = true) (n : ℕ) :
    frees (l.drop n) K R = true :=
  decide_eq_true fun y member => (of_decide_eq_true h) y (List.mem_of_mem_drop member)

theorem stage2In_sameClaim (pub : Public) (s : Replayer) (level : ℕ) {G F : Fold}
    (same : G.sameClaim F = true) : stage2In pub s level G = stage2In pub s level F := by
  simp only [Fold.sameClaim, Bool.and_eq_true, beq_iff_eq, decide_eq_decide] at same
  obtain ⟨⟨⟨⟨rhs, row⟩, many⟩, trace⟩, euclidean⟩ := same
  unfold stage2In rowsIn claimsIn
  rw [rhs, row, trace, euclidean]
  simp only [many]

theorem productsIn_frame {s s' : Replayer} (level : ℕ) (arities : List ℕ)
    (recorded : ∀ k ∈ productKeys level arities.length, recordedIn s' k = recordedIn s k)
    (rounds : ∀ k < arities.length,
      roundsIn s' ⟨level, s!"range.{k}", none⟩ = roundsIn s ⟨level, s!"range.{k}", none⟩) :
    productsIn s' level arities = productsIn s level arities := by
  unfold productsIn
  refine List.map_congr_left fun ⟨arity, k⟩ member => ?_
  have below := (List.mem_zipIdx' member).1
  have children := recorded ⟨level, s!"range.{k}.children", none⟩
    (List.mem_flatMap.mpr ⟨k, List.mem_range.mpr below, List.mem_cons_self⟩)
  have batch := recorded ⟨level, s!"range.{k}.batch", none⟩
    (List.mem_flatMap.mpr ⟨k, List.mem_range.mpr below,
      List.mem_cons_of_mem _ List.mem_cons_self⟩)
  simp only [firstIn, children, batch, rounds k below]

/-- The stage function of the last check of the range proof reads only `rangeKeys` and the
rounds of `rangeRounds`. -/
theorem stageValue_range_frame (pub : Public) (level : ℕ) (F : Fold) {s s' : Replayer}
    (recorded : ∀ k ∈ rangeKeys level F, recordedIn s' k = recordedIn s k)
    (integers : s'.integers.get? ⟨level, "norm.response_l2_sq", none⟩ =
      s.integers.get? ⟨level, "norm.response_l2_sq", none⟩)
    (rounds : ∀ k ∈ rangeRounds level F, roundsIn s' k = roundsIn s k) :
    stageValue pub s' (rangeCheckOf level F) = stageValue pub s (rangeCheckOf level F) := by
  have products := productsIn_frame (s := s) (s' := s') level
    ((productStages F.logBasis).map (·.1))
    (fun k inside => recorded k (by
      rw [List.length_map] at inside
      exact List.mem_cons_of_mem _ (List.mem_cons_of_mem _ (List.mem_cons_of_mem _
        (List.mem_cons_of_mem _ (List.mem_cons_of_mem _ inside))))))
    (fun k below => rounds _ (by
      rw [List.length_map] at below
      exact List.mem_cons_of_mem _ (List.mem_cons_of_mem _
        (List.mem_map.mpr ⟨k, List.mem_range.mpr below, rfl⟩))))
  rw [rangeKeys] at recorded
  rw [rangeRounds] at rounds
  cases euclidean : F.euclidean
  · rw [rangeCheckOf_plain level F euclidean]
    simp only [stageValue, products, pointIn, firstIn]
    simp (disch := simp) only [recorded, rounds]
  · rw [rangeCheckOf_euclidean level F euclidean]
    simp only [stageValue, products, pointIn, firstIn, integers]
    simp (disch := simp) only [recorded, rounds]

/-- `stage2In` reads only `stage2Keys`. -/
theorem stage2In_frame (pub : Public) (level : ℕ) (F : Fold) {s s' : Replayer}
    (recorded : ∀ k ∈ stage2Keys level, recordedIn s' k = recordedIn s k)
    (fields : ∀ k ∈ stage2Keys level, s'.fields.get? k = s.fields.get? k) :
    stage2In pub s' level F = stage2In pub s level F := by
  have data : dataIn pub s' level = dataIn pub s level := by
    funext run
    unfold dataIn
    rw [fields ⟨level - 1, "next_witness", none⟩ (by simp [stage2Keys]),
      fields ⟨level, "opening_payload", none⟩ (by simp [stage2Keys])]
  have claims : claimsIn s' level F = claimsIn s level F := by
    unfold claimsIn
    split <;> exact recorded _ (by simp [stage2Keys])
  have rows : rowsIn s' level F = rowsIn s level F := by
    unfold rowsIn
    rw [recorded ⟨level, "row_coefficients", none⟩ (by simp [stage2Keys])]
  unfold stage2In
  rw [data, claims, rows]
  rw [stage2Keys] at recorded
  simp only [firstIn]
  simp (disch := simp) only [recorded]

/-- `stage3In` reads only `stage3Keys` and the rounds of `stage3`. -/
theorem stage3In_frame (level : ℕ) (plan : SetupPlan) {s s' : Replayer}
    (recorded : ∀ k ∈ stage3Keys level, recordedIn s' k = recordedIn s k)
    (rounds : roundsIn s' ⟨level, "stage3", none⟩ = roundsIn s ⟨level, "stage3", none⟩) :
    stage3In s' level plan = stage3In s level plan := by
  unfold stage3In
  rw [stage3Keys] at recorded
  simp only [firstIn, rounds]
  simp (disch := simp) only [recorded]

theorem fusedEndIn_value (setup : Array Fq) (s : Replayer) (level : ℕ) (F : Fold) :
    (fusedEndIn setup s level F).value = firstIn s ⟨level, "witness_evaluation", none⟩ := by
  unfold fusedEndIn
  cases F.fused <;> rfl

theorem passing_some {f g : FusedEnd} (passes : f.passing = some g) : g = f := by
  unfold FusedEnd.passing at passes
  split at passes
  · cases passes; rfl
  · cases passes

section Run

variable (pub : Public) (proof : ByteArray) (messages : Array Message)

/-- What a run of the operations `pre` of the shape `foldShape` without a stop gives, from a
state with no recorded value and no rounds: on its last state the stage function of the last
range check accepts, and the hypotheses `point` and `ended` of `foldRunIn_eq` hold. -/
theorem fold_facts (s₀ : Replayer) (fresh : ∀ k, recordedIn s₀ k = #[])
    (empty : s₀.derived.rounds = #[]) (pre : List (Op × ℕ)) (level : ℕ) (F : Fold)
    {r a e : ℕ} (shape : foldShape pre level F r a e = true)
    (okPre : (runFrom pub proof messages pre s₀).1 = .ok ()) :
    ((stageValue pub (runFrom pub proof messages pre s₀).2 (rangeCheckOf level F)).map
        StageOut.accepts).getD false = true ∧
      (roundsIn (runFrom pub proof messages pre s₀).2 ⟨level, "fused", none⟩).map (·.2) =
        (recordedIn (runFrom pub proof messages pre s₀).2 ⟨level, "fused.point", none⟩).toList ∧
      (plainRun (stage2In pub (runFrom pub proof messages pre s₀).2 level F).fused
        (roundsIn (runFrom pub proof messages pre s₀).2 ⟨level, "fused", none⟩)).claim =
          firstIn (runFrom pub proof messages pre s₀).2 ⟨level, "fused.output", none⟩ := by
  simp only [foldShape, Bool.and_eq_true, decide_eq_true_eq, beq_iff_eq] at shape
  obtain ⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨ra, ae⟩, atR⟩, atA⟩, atE⟩, freeR⟩, freeA⟩, beforeA⟩, afterA⟩, beforeE⟩,
    afterE⟩, count⟩ := shape
  obtain ⟨rIn, rRec⟩ := any_at atR
  obtain ⟨kr, opR⟩ := isRangeCheck_sound rRec
  obtain ⟨_, stepR, lastR, _⟩ := runFrom_at pub proof messages pre r rIn s₀ okPre
  rw [opR] at stepR
  have rangeOk := step_check_ok pub proof messages _ kr (rangeCheckOf level F) _
    (holds_rangeCheckOf pub level F _) stepR
  have rangeAt : ((stageValue pub (runFrom pub proof messages pre s₀).2
      (rangeCheckOf level F)).map StageOut.accepts).getD false = true := by
    rw [lastR, stageValue_range_frame pub level F
      (fun k inside => recorded_free pub proof messages _ _ k (frees_writes freeR inside))
      (integers_free pub proof messages _ _ _ (frees_writes freeR (by simp [rangeKeys])))
      (fun k inside => roundsIn_free pub proof messages _ _ k (frees_ends freeR inside))]
    exact rangeOk
  obtain ⟨aIn, aRec⟩ := any_at atA
  obtain ⟨G, opA, same⟩ := isFusedClaim_sound aRec
  obtain ⟨_, _, lastA, nextA⟩ := runFrom_at pub proof messages pre a aIn s₀ okPre
  rw [opA] at nextA
  have absorbed := step_absorb_fused pub proof messages pre[a].2 ⟨level, "fused.claim", none⟩ G
    (runFrom pub proof messages (pre.take a) s₀).2
  have claimBefore : recordedIn (runFrom pub proof messages (pre.take a) s₀).2
      ⟨level, "fused.claim", none⟩ = #[] :=
    (recorded_free pub proof messages _ _ _ (frees_writes beforeA List.mem_cons_self)).trans
      (fresh _)
  have claimAfter : recordedIn (runFrom pub proof messages pre s₀).2
      ⟨level, "fused.claim", none⟩ = #[(stage2In pub (runFrom pub proof messages pre s₀).2
        level F).fused] := by
    have frame : stage2In pub (runFrom pub proof messages pre s₀).2 level F =
        stage2In pub (runFrom pub proof messages (pre.take a) s₀).2 level F := by
      rw [lastA]
      exact stage2In_frame pub level F
        (fun k inside => recorded_free pub proof messages _ _ k (frees_writes freeA inside))
        (fun k inside => fields_free pub proof messages _ _ k (frees_writes freeA inside))
    rw [frame, ← stage2In_sameClaim pub _ level same]
    rw [nextA, recorded_free pub proof messages _ _ _ (frees_writes afterA List.mem_cons_self),
      absorbed.2, claimBefore, Array.empty_append]
  obtain ⟨eIn, eRec⟩ := any_at atE
  have opE := isPlainEnd_sound eRec
  obtain ⟨okE, _, lastE, nextE⟩ := runFrom_at pub proof messages pre e eIn s₀ okPre
  rw [opE] at nextE
  have ended := step_ended_plain pub proof messages pre[e].2 ⟨level, "fused", none⟩ F.vars
    (runFrom pub proof messages (pre.take e) s₀).2
  have output : recordedIn (runFrom pub proof messages pre s₀).2
      ⟨level, "fused.output", none⟩ = #[(plainRun (firstIn (runFrom pub proof messages
        (pre.take e) s₀).2 ⟨level, "fused.claim", none⟩) (endRoundsIn (runFrom pub proof messages
          (pre.take e) s₀).2 ⟨level, "fused", none⟩ F.vars)).claim] := by
    have before : recordedIn (runFrom pub proof messages (pre.take e) s₀).2
        ⟨level, "fused.output", none⟩ = #[] :=
      (recorded_free pub proof messages _ _ _ (frees_writes beforeE List.mem_cons_self)).trans
        (fresh _)
    have recordedEnd : recordedIn (((step pub proof messages pre[e].2
        (.ended ⟨level, "fused", none⟩ F.vars .plain)).run).run
          (runFrom pub proof messages (pre.take e) s₀).2).2 ⟨level, "fused.output", none⟩ =
        recordedIn (runFrom pub proof messages (pre.take e) s₀).2 ⟨level, "fused.output", none⟩ ++
          #[(plainRun (firstIn (runFrom pub proof messages (pre.take e) s₀).2
            ⟨level, "fused.claim", none⟩) (endRoundsIn (runFrom pub proof messages (pre.take e)
              s₀).2 ⟨level, "fused", none⟩ F.vars)).claim] := ended.2.1
    rw [nextE, recorded_free pub proof messages _ _ _
      (frees_writes afterE (List.mem_cons_of_mem _ List.mem_cons_self)), recordedEnd, before,
      Array.empty_append]
  have claimEnd : recordedIn (runFrom pub proof messages (pre.take e) s₀).2
      ⟨level, "fused.claim", none⟩ = recordedIn (runFrom pub proof messages pre s₀).2
        ⟨level, "fused.claim", none⟩ := by
    have later := frees_drop afterA (e - (a + 1))
    rw [List.drop_drop, show a + 1 + (e - (a + 1)) = e by omega] at later
    rw [lastE]
    exact (recorded_free pub proof messages _ _ _ (frees_writes later List.mem_cons_self)).symm
  have rounds : roundsIn (runFrom pub proof messages pre s₀).2 ⟨level, "fused", none⟩ =
      endRoundsIn (runFrom pub proof messages (pre.take e) s₀).2 ⟨level, "fused", none⟩ F.vars := by
    have missing : (runFrom pub proof messages (pre.take e) s₀).2.derived.rounds.find?
        (·.1 == ⟨level, "fused", none⟩) = none := by
      rw [find_tail (runFrom_tail pub proof messages _ s₀) fun inside => by
        obtain ⟨y, member, found⟩ := List.mem_flatMap.mp inside
        exact frees_ends beforeE List.mem_cons_self y member found, empty]
      rfl
    have pushed := ended.2.2
    have found : (((step pub proof messages pre[e].2
        (.ended ⟨level, "fused", none⟩ F.vars .plain)).run).run
          (runFrom pub proof messages (pre.take e) s₀).2).2.derived.rounds.find?
            (·.1 == ⟨level, "fused", none⟩) = some (⟨level, "fused", none⟩,
              endRoundsIn (runFrom pub proof messages (pre.take e) s₀).2
                ⟨level, "fused", none⟩ F.vars) := by
      rw [pushed, Array.find?_push, missing]
      simp
    unfold roundsIn
    rw [nextE, find_grows (runFrom_grows pub proof messages _ _).rounds found]
    rfl
  have point : (recordedIn (runFrom pub proof messages pre s₀).2
      ⟨level, "fused.point", none⟩) = recordedIn (runFrom pub proof messages (pre.take e) s₀).2
        ⟨level, "fused.point", none⟩ := by
    rw [nextE, recorded_free pub proof messages _ _ _ (frees_writes afterE List.mem_cons_self)]
    exact ((step_keeps pub proof messages _ _ _).same _ (by simp [Op.writes])).1
  have size : (recordedIn (runFrom pub proof messages (pre.take e) s₀).2
      ⟨level, "fused.point", none⟩).size = F.vars := by
    rw [size_drawnCount pub proof messages _ _ _ count okE, fresh]
    simp
  have pointHyp : (roundsIn (runFrom pub proof messages pre s₀).2
      ⟨level, "fused", none⟩).map (·.2) = (recordedIn (runFrom pub proof messages pre s₀).2
        ⟨level, "fused.point", none⟩).toList := by
    rw [rounds, point]
    exact List.map_snd_zip (by simp [size])
  have endedHyp : (plainRun (stage2In pub (runFrom pub proof messages pre s₀).2 level F).fused
      (roundsIn (runFrom pub proof messages pre s₀).2 ⟨level, "fused", none⟩)).claim =
        firstIn (runFrom pub proof messages pre s₀).2 ⟨level, "fused.output", none⟩ := by
    rw [rounds]
    simp only [firstIn, output, claimEnd, claimAfter]
    rfl
  exact ⟨rangeAt, pointHyp, endedHyp⟩

/-- The check `fused.end` on a state `s` has no stop exactly when the setup prefix exists and the
end check holds there. -/
theorem fused_accepts (key : Key) (c level : ℕ) (F : Fold) (s : Replayer) :
    (((step pub proof messages c (.check key (.fused level F))).run).run s).1 = .ok () ↔
      ∃ setup, pub.setupFor F.setupCount = some setup ∧
        ((fusedEndIn setup s level F).passing).isSome = true := by
  constructor
  · intro ran
    have endOk := step_check_ok pub proof messages c key (.fused level F) s
      (holds_fused pub level F s) ran
    simp only [stageValue] at endOk
    cases found : pub.setupFor F.setupCount with
    | none => rw [found] at endOk; cases endOk
    | some setup =>
        rw [found] at endOk
        exact ⟨setup, rfl, endOk⟩
  · rintro ⟨setup, expanded, passes⟩
    rw [step_check, checkStep_run]
    have value : stageValue pub s (.fused level F) =
        some (.fused (fusedEndIn setup s level F).passing) := by
      simp only [stageValue, expanded, Option.map_some]
    rw [value, checkWith_run]
    simp [StageOut.accepts, passes]

/-- The fold of the level `level` on the state at its `fused.end` check: when the run of the
operations `pre` of the shape `foldShape` and of `fused.end` has no stop, from a state with no
recorded value and no rounds, `foldRunIn` is the recorded `(fused.point, witness_evaluation)`. -/
theorem foldRunIn_of_run (s₀ : Replayer) (fresh : ∀ k, recordedIn s₀ k = #[])
    (empty : s₀.derived.rounds = #[]) (pre : List (Op × ℕ)) (key : Key) (c level : ℕ) (F : Fold)
    {r a e : ℕ} (shape : foldShape pre level F r a e = true)
    (ran : (runFrom pub proof messages (pre ++ [(.check key (.fused level F), c)]) s₀).1 =
      .ok ()) :
    foldRunIn pub (runFrom pub proof messages pre s₀).2 level F =
      some ((recordedIn (runFrom pub proof messages pre s₀).2 ⟨level, "fused.point", none⟩).toList,
        firstIn (runFrom pub proof messages pre s₀).2 ⟨level, "witness_evaluation", none⟩) := by
  obtain ⟨okPre, okEnd⟩ := (runFrom_append_fst pub proof messages _ _ s₀).mp ran
  obtain ⟨rangeAt, pointHyp, endedHyp⟩ :=
    fold_facts pub proof messages s₀ fresh empty pre level F shape okPre
  obtain ⟨setup, expanded, passes⟩ := (fused_accepts pub proof messages key c level F _).mp
    ((runFrom_cons_fst pub proof messages _ _ _).mp okEnd).1
  rw [foldRunIn_eq pub _ level F expanded pointHyp endedHyp, ite_eq_left rangeAt]
  obtain ⟨f, hf⟩ := Option.isSome_iff_exists.mp passes
  rw [hf, Option.map_some, passing_some hf, fusedEndIn_value]

/-- A reject at `fused.end`: when the run of the operations `pre` of the shape `foldShape` has no
stop and the check `fused.end` stops, `foldRunIn` is `none` on the state at the check. -/
theorem foldRunIn_of_reject (s₀ : Replayer) (fresh : ∀ k, recordedIn s₀ k = #[])
    (empty : s₀.derived.rounds = #[]) (pre : List (Op × ℕ)) (key : Key) (c level : ℕ) (F : Fold)
    {r a e : ℕ} (shape : foldShape pre level F r a e = true)
    (okPre : (runFrom pub proof messages pre s₀).1 = .ok ())
    (stopped : (((step pub proof messages c (.check key (.fused level F))).run).run
      (runFrom pub proof messages pre s₀).2).1 ≠ .ok ()) :
    foldRunIn pub (runFrom pub proof messages pre s₀).2 level F = none := by
  obtain ⟨_, pointHyp, endedHyp⟩ :=
    fold_facts pub proof messages s₀ fresh empty pre level F shape okPre
  cases expanded : pub.setupFor F.setupCount with
  | none => simp only [foldRunIn, expanded, Option.bind_none]
  | some setup =>
      rw [foldRunIn_eq pub _ level F expanded pointHyp endedHyp]
      cases passes : (fusedEndIn setup (runFrom pub proof messages pre s₀).2 level F).passing with
      | none => simp
      | some f =>
          exact absurd ((fused_accepts pub proof messages key c level F _).mpr
            ⟨setup, expanded, by rw [passes]; rfl⟩) stopped

/-- The offloaded fold of the level `level` on the state at its `fused.end` check: when the run of
the operations `pre` of the shape `offloadShape` and of `fused.end` has no stop, `stage3In` is
`some` and `offloadRunIn` is the recorded `(fused.point, witness_evaluation)` with it. -/
theorem offloadRunIn_of_run (s₀ : Replayer) (fresh : ∀ k, recordedIn s₀ k = #[])
    (empty : s₀.derived.rounds = #[]) (pre : List (Op × ℕ)) (key : Key) (c level : ℕ) (F : Fold)
    (plan : SetupPlan) {r a e b t : ℕ} (shape : offloadShape pre level F plan r a e b t = true)
    (ran : (runFrom pub proof messages (pre ++ [(.check key (.fused level F), c)]) s₀).1 =
      .ok ()) :
    offloadRunIn pub (runFrom pub proof messages pre s₀).2 level F plan =
        (stage3In (runFrom pub proof messages pre s₀).2 level plan).map (fun o =>
          (((recordedIn (runFrom pub proof messages pre s₀).2 ⟨level, "fused.point", none⟩).toList,
            firstIn (runFrom pub proof messages pre s₀).2 ⟨level, "witness_evaluation", none⟩), o)) ∧
      (stage3In (runFrom pub proof messages pre s₀).2 level plan).isSome = true := by
  simp only [offloadShape, Bool.and_eq_true, Bool.not_eq_true'] at shape
  obtain ⟨⟨⟨⟨⟨⟨⟨fold, plain⟩, atB⟩, atT⟩, beforeB⟩, afterB⟩, setupAfter⟩, freeT⟩ := shape
  obtain ⟨okPre, okEnd⟩ := (runFrom_append_fst pub proof messages _ _ s₀).mp ran
  obtain ⟨rangeAt, pointHyp, endedHyp⟩ :=
    fold_facts pub proof messages s₀ fresh empty pre level F fold okPre
  obtain ⟨setup, expanded, passes⟩ := (fused_accepts pub proof messages key c level F _).mp
    ((runFrom_cons_fst pub proof messages _ _ _).mp okEnd).1
  obtain ⟨tIn, tRec⟩ := any_at atT
  obtain ⟨kt, opT⟩ := isStage3End_sound tRec
  obtain ⟨_, stepT, lastT, _⟩ := runFrom_at pub proof messages pre t tIn s₀ okPre
  rw [opT] at stepT
  have stage3Ok := step_check_ok pub proof messages _ kt (.stage3 level plan) _
    (holds_stage3 pub level plan _) stepT
  have stage3At : (stage3In (runFrom pub proof messages pre s₀).2 level plan).isSome = true := by
    rw [lastT, stage3In_frame level plan
      (fun k inside => recorded_free pub proof messages _ _ k (frees_writes freeT inside))
      (roundsIn_free pub proof messages _ _ _ (frees_ends freeT List.mem_cons_self))]
    exact stage3Ok
  obtain ⟨bIn, bRec⟩ := any_at atB
  have opB := isSetupClaim_sound bRec
  obtain ⟨_, _, lastB, nextB⟩ := runFrom_at pub proof messages pre b bIn s₀ okPre
  rw [opB] at nextB
  have absorbed := step_absorb_recorded pub proof messages pre[b].2 ⟨level, "stage3.claim", none⟩
    [⟨level, "stage3.setup_claim", none⟩] (runFrom pub proof messages (pre.take b) s₀).2
  have claimHyp : firstIn (runFrom pub proof messages pre s₀).2 ⟨level, "stage3.claim", none⟩ =
      firstIn (runFrom pub proof messages pre s₀).2 ⟨level, "stage3.setup_claim", none⟩ := by
    have before : recordedIn (runFrom pub proof messages (pre.take b) s₀).2
        ⟨level, "stage3.claim", none⟩ = #[] :=
      (recorded_free pub proof messages _ _ _ (frees_writes beforeB List.mem_cons_self)).trans
        (fresh _)
    have kept : recordedIn (runFrom pub proof messages pre s₀).2
        ⟨level, "stage3.setup_claim", none⟩ = recordedIn (runFrom pub proof messages (pre.take b)
          s₀).2 ⟨level, "stage3.setup_claim", none⟩ := by
      rw [lastB]
      exact recorded_free pub proof messages _ _ _ (frees_writes setupAfter List.mem_cons_self)
    unfold firstIn
    rw [nextB, recorded_free pub proof messages _ _ _ (frees_writes afterB List.mem_cons_self),
      absorbed.2, before, ← nextB, kept]
    simp
  rw [offloadRunIn_eq pub _ level F plan expanded plain pointHyp endedHyp claimHyp,
    ite_eq_left rangeAt]
  obtain ⟨f, hf⟩ := Option.isSome_iff_exists.mp passes
  rw [hf, Option.map_some, Option.bind_some, passing_some hf, fusedEndIn_value]
  exact ⟨rfl, stage3At⟩

theorem stageValue_rangeCheckOf (pub : Public) (s : Replayer) (level : ℕ) (F : Fold) :
    stageValue pub s (rangeCheckOf level F) = some (if F.euclidean then .norm (treeRun
      (normRun (leafDegree F.logBasis) F.limit (normsIn s level).energy (normsIn s level).merge
        (normsIn s level).rounds (normsIn s level).virtual (wordsIn s level F).image)
      (wordsIn s level F).stages 1 (fun _ => Ext.one) ((wordsIn s level F).point F.low) Ext.zero)
    else .range (rangeRun (2 ^ F.logBasis) (wordsIn s level F).stages
      (roundsIn s ⟨level, "range.leaf", none⟩) (wordsIn s level F).image
      ((wordsIn s level F).point F.low))) := by
  unfold rangeCheckOf
  cases F.euclidean <;> rfl

/-- When the stage function of the last range check does not accept, `foldRunIn` is `none`. -/
theorem foldRunIn_range_none (s : Replayer) (level : ℕ) (F : Fold)
    (rejects : ((stageValue pub s (rangeCheckOf level F)).map StageOut.accepts).getD false =
      false) : foldRunIn pub s level F = none := by
  rw [stageValue_rangeCheckOf] at rejects
  unfold foldRunIn
  cases pub.setupFor F.setupCount with
  | none => rfl
  | some setup =>
      simp only [Option.bind_some]
      cases euclidean : F.euclidean
      · simp only [euclidean, Bool.false_eq_true, ↓reduceIte, Option.map_some, Option.getD_some,
          StageOut.accepts] at rejects ⊢
        rw [foldRun_if, rejects]
        rfl
      · simp only [euclidean, ↓reduceIte, Option.map_some, Option.getD_some,
          StageOut.accepts] at rejects ⊢
        rw [foldRunSq_if, rejects]
        rfl

theorem step_check_accepts (i : ℕ) (key : Key) (c : Check) (s : Replayer)
    (accepts : ((stageValue pub s c).map StageOut.accepts).getD false = true) :
    (((step pub proof messages i (.check key c)).run).run s).1 = .ok () := by
  rw [step_check, checkStep_run]
  cases value : stageValue pub s c with
  | none => rw [value] at accepts; cases accepts
  | some out =>
      rw [value] at accepts
      rw [checkWith_run]
      simp only [Option.map_some, Option.getD_some] at accepts
      simp [accepts]

/-- A reject at the last check of the range proof (`range.leaf.end` or `norm.end`): `foldRunIn`
is `none` on the state at the check. -/
theorem foldRunIn_of_range_reject (s : Replayer) (key : Key) (i level : ℕ) (F : Fold)
    (stopped : (((step pub proof messages i (.check key (rangeCheckOf level F))).run).run s).1 ≠
      .ok ()) : foldRunIn pub s level F = none := by
  apply foldRunIn_range_none
  cases accepts : ((stageValue pub s (rangeCheckOf level F)).map StageOut.accepts).getD false
  · rfl
  · exact absurd (step_check_accepts pub proof messages i key _ s accepts) stopped

end Run

end Replayer

end NightstreamFPrime.Spec.Compression.Akita.Replay
