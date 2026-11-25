import NightstreamFPrime.Spec.Compression.Akita.Replay.ShapeStages

/-! The converse of `verifyProof_accept` from the level functions (Akita revision `d50f700`;
`verifyProof_accept_of` of `Replay/RunProof.lean`, finding 16 of `Replay/ShapeRange.lean`,
`Replay/RunProof.lean` and `Replay/RunWhole.lean`).

- `holds_run`, `check_keeps`: the step of a check changes only the recorded stage results and
  checks. `stateBefore_stopped`: after a first stop at the operation `c`, the state before each
  later operation is the state of the stop. `holds_of_later`: a check at `c` holds when a function
  of a later state is not `none` and a stop at `c` gives `none` for it.
- `tensorRunIn` is `tensorRun` of the stage function of `tensor.end` on the recorded values.
- `levelStart L ℓ` is the index after `fused.end` of the level before `ℓ`. `stageIndices` are the
  indices of the checks `range.k.end` of a level, and `erased_stage_reject` is the part of
  `erased_stages` for a stop at one of them.
- `coveredAt` is a Bool on the skeleton: each check is `fused.end` of a level at `fusedIndex`, the
  last check of the range proof, a check `range.k.end` or `norm.integer` of a level before its
  `fused.end`, `tensor.opening` before the `tensor.end` of the same claim, or `tensor.end`.
- `erased_checks`: on operations whose erased forms are a covered skeleton, each check holds on
  the state before it when the level function of each level is not `none` on the state before its
  `fused.end` and `tensorRunIn` accepts on the state before each `tensor.end`. Each of the two
  hypotheses is needed only when the run before `levelStart` of the level has no stop.
- `skeleton20_covered`, `verifyProof_accept_of_levels`, `verifyProof_levels_of_accept` and
  `verifyProof_accept_iff_levels`: `runProof` accepts a proof of `layout20` exactly when the
  layout takes the plan, the proof decodes with no byte left, the works hold, the level functions
  and the tensor reductions accept and `terminalRun` accepts. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita.Replay

open NightstreamFPrime.Spec.Compression.Akita

namespace Replayer

/-- `tensorRun` of the tensor reduction of `level` for the claim `(point, opening)` on the recorded
values of `s`: the stage function of `tensor.end`. -/
abbrev tensorRunIn (s : Replayer) (level count : ℕ) (point opening : Key) : Option Ended :=
  tensorRun count (recordedIn s point).toList (firstIn s opening)
    ((recordedIn s ⟨level, "tensor.partials", none⟩).getD 0 Ext.zero)
    ((recordedIn s ⟨level, "tensor.partials", none⟩).getD 1 Ext.zero)
    (firstIn s ⟨level, "tensor.eta", none⟩) (roundsIn s ⟨level, "tensor.sumcheck", none⟩)
    (firstIn s ⟨level, "tensor.final_claims", none⟩)

section Stuck

variable (pub : Public) (proof : ByteArray) (messages : Array Message)

/-- `holds` gives a Bool and keeps the state. -/
theorem holds_run (check : Check) (s : Replayer) :
    ∃ b, ((holds pub check).run).run s = (.ok b, s) := by
  cases check with
  | stage nodes arity weights output children =>
      exact ⟨_, run_holds_stage pub nodes arity weights output children s⟩
  | _ => exact ⟨_, rfl⟩

/-- A function of the state that does not read the recorded stage results and checks has the same
value after the step of a check, with a stop or not. -/
theorem check_keeps {α : Type} (g : Replayer → α)
    (staged_ : ∀ s key out, g (staged s key out) = g s)
    (checked_ : ∀ s key, g (checked s key) = g s) (i : ℕ) (key : Key) (check : Check)
    (s : Replayer) :
    g (((step pub proof messages i (.check key check)).run).run s).2 = g s := by
  cases value : stageValue pub s check with
  | some out =>
      rw [step_stage pub proof messages i key check s value]
      split
      · rw [checked_, staged_]
      · rw [staged_]
  | none =>
      obtain ⟨b, held⟩ := holds_run pub check s
      rw [step_check, checkStep_run, value]
      simp only [checkWith]
      rw [run_bind, held]
      cases b
      · rfl
      · exact checked_ s key

/-- After a first stop at the operation `c`, the state before each later operation is the state of
the stop. -/
theorem stateBefore_stopped {ops : Array Op} {c n : ℕ} {op : Op} (at_ : ops[c]? = some op)
    (lt : c < n)
    (before : (runFrom pub proof messages (ops.zipIdx.toList.take c) (channelStart pub)).1 =
      .ok ())
    (stopped : (((step pub proof messages c op).run).run
      (stateBefore pub proof messages ops c)).1 ≠ .ok ()) :
    stateBefore pub proof messages ops n =
      (((step pub proof messages c op).run).run (stateBefore pub proof messages ops c)).2 := by
  have cut : ops.zipIdx.toList.take n =
      ops.zipIdx.toList.take c ++ ((op, c) :: (ops.zipIdx.toList.take n).drop (c + 1)) := by
    conv_lhs => rw [← List.take_append_drop (c + 1) (ops.zipIdx.toList.take n)]
    rw [List.take_take, Nat.min_eq_left (by omega), take_succ_zipIdx ops at_, List.append_assoc]
    rfl
  unfold stateBefore at stopped ⊢
  rw [cut, runFrom_append]
  rcases found : runFrom pub proof messages (ops.zipIdx.toList.take c) (channelStart pub) with
    ⟨_ | ⟨⟩, s₁⟩
  · rw [found] at before
    cases before
  · rw [found] at stopped
    simp only
    rw [runFrom_cons]
    rcases stepped : ((step pub proof messages c op).run).run s₁ with ⟨_ | ⟨⟩, s₂⟩
    · rfl
    · rw [stepped] at stopped
      exact absurd rfl stopped

/-- A step of a check that stops has `holds` false on its state. -/
theorem holds_of_stopped (i : ℕ) (key : Key) (check : Check) (s : Replayer)
    (stopped : (((step pub proof messages i (.check key check)).run).run s).1 ≠ .ok ()) :
    (((holds pub check).run).run s).1 = .ok false := by
  obtain ⟨b, held⟩ := holds_run pub check s
  cases b
  · rw [held]
  · exact absurd ((step_check_iff pub proof messages i key check s).mpr (by rw [held])) stopped

/-- A check at `c` holds on the state before it when a function `g` of the state before a later
operation `f` is not `none`, `g` does not read the recorded stage results and checks, and a stop
at `c` gives `g = none` on the state before `c`. -/
theorem holds_of_later {ops : Array Op} {c f : ℕ} {key : Key} {check : Check}
    (at_ : ops[c]? = some (.check key check)) (lt : c < f)
    (before : (runFrom pub proof messages (ops.zipIdx.toList.take c) (channelStart pub)).1 =
      .ok ())
    {α : Type} (g : Replayer → Option α) (staged_ : ∀ s key out, g (staged s key out) = g s)
    (checked_ : ∀ s key, g (checked s key) = g s)
    (some_ : g (stateBefore pub proof messages ops f) ≠ none)
    (reject : (((step pub proof messages c (.check key check)).run).run
      (stateBefore pub proof messages ops c)).1 ≠ .ok () →
        g (stateBefore pub proof messages ops c) = none) :
    (((holds pub check).run).run (stateBefore pub proof messages ops c)).1 = .ok true := by
  by_contra bad
  have stopped : (((step pub proof messages c (.check key check)).run).run
      (stateBefore pub proof messages ops c)).1 ≠ .ok () :=
    fun ok => bad ((step_check_iff pub proof messages c key check _).mp ok)
  apply some_
  rw [stateBefore_stopped pub proof messages at_ lt before stopped,
    check_keeps pub proof messages g staged_ checked_]
  exact reject stopped

end Stuck

end Replayer

open Replayer

section Cover

/-- The index after `fused.end` of the level before `ℓ`, and 0 for the root. -/
def levelStart (L : Layout) (ℓ : ℕ) : ℕ := if ℓ = 0 then 0 else fusedIndex L (ℓ - 1) + 1

/-- The index `c` is after the level before `ℓ` and before `fused.end` of the level `ℓ`. -/
def inLevel (L : Layout) (ℓ c : ℕ) : Bool :=
  decide (levelStart L ℓ ≤ c) && decide (c < fusedIndex L ℓ)

/-- The indices of the checks `range.k.end` of the fold `F` of `level` in `l`. -/
def stageIndices (l : List (Op × ℕ)) (level : ℕ) (F : Fold) : List ℕ :=
  match productStages F.logBasis with
  | [(arity, _)] => [indexOf l (isCheck0 level arity)]
  | [(arity, nodes), (arity1, _)] =>
      [indexOf l (isCheck0 level arity), indexOf l (isCheck1 level nodes arity1)]
  | _ => []

/-- The check `tensor.end` of the claim `(point, value)` with the partials `partials`. -/
def isTensorOf (point value partials : Key) : Op → Bool
  | .check _ (.tensor level _ p o) =>
      p == point && o == value && partials == ⟨level, "tensor.partials", none⟩
  | _ => false

theorem isTensorOf_sound {point value partials : Key} {op : Op}
    (h : isTensorOf point value partials op = true) :
    ∃ key level count, op = .check key (.tensor level count point value) ∧
      partials = ⟨level, "tensor.partials", none⟩ := by
  unfold isTensorOf at h
  split at h
  · rename_i key level count p o
    simp only [Bool.and_eq_true, beq_iff_eq] at h
    obtain ⟨⟨rfl, rfl⟩, rfl⟩ := h
    exact ⟨key, level, count, rfl, rfl⟩
  · cases h

/-- The check of the entry `x` of the skeleton `l` of `L` has a place that `erased_checks` reads:
`fused.end` of a level at its `fusedIndex`; the last check of the range proof, a check
`range.k.end` or `norm.integer` of a level, in the operations of the level before its `fused.end`;
`tensor.opening` before the `tensor.end` of its claim; `tensor.end`. There is no `stage3.end`. -/
def coveredAt (L : Layout) (l : List (Op × ℕ)) (x : Op × ℕ) : Bool :=
  match x.1 with
  | .check _ (.fused level _) =>
      decide (level < L.folds.length) && x.2 == fusedIndex L level &&
        decide (levelStart L level ≤ x.2)
  | .check _ (.range level _ _ _) | .check _ (.rangeNorm level _ _ _ _) =>
      (L.folds[level]?.any fun F => isRangeCheck level F x.1) && inLevel L level x.2
  | .check _ (.stage _ _ _ output _) =>
      (L.folds[output.level]?.any fun F => decide (x.2 ∈ stageIndices l output.level F)) &&
        inLevel L output.level x.2
  | .check _ (.energy key limit) =>
      (L.folds[key.level]?.any fun F => F.euclidean &&
        key == ⟨key.level, "norm.response_l2_sq", none⟩ && limit == F.limit) &&
        inLevel L key.level x.2
  | .check _ (.tensorOpening point value partials) =>
      match l.findIdx? (fun y => isTensorOf point value partials y.1) with
      | some t => decide (x.2 < t) && decide (levelStart L partials.level ≤ x.2)
      | none => false
  | .check _ (.tensor level _ _ _) => decide (levelStart L level ≤ x.2)
  | .check _ (.stage3 _ _) => false
  | _ => true

variable (pub : Public) (proof : ByteArray) (messages : Array Message) {L : Layout}
  {ops : Array Op}

/-- The part of `erased_stages` for a stop: at an index of `stageIndices`, the operation is a check
`range.k.end`, and when it does not hold on the state before it, the level function is `none`
there. -/
theorem erased_stage_reject (erased : ops.toList.map Op.erase = skeleton L)
    (aligned : ReadsAligned messages ops.zipIdx.toList 0) {ℓ : ℕ} {F : Fold}
    (shaped : stagesShape (skeleton L).zipIdx ℓ F = true) {c : ℕ}
    (member : c ∈ stageIndices (skeleton L).zipIdx ℓ F) :
    ∃ key nodes arity weights output children,
      ops[c]? = some (.check key (.stage nodes arity weights output children)) ∧
      ((runFrom pub proof messages (ops.zipIdx.toList.take c) (channelStart pub)).1 = .ok () →
        (((holds pub (.stage nodes arity weights output children)).run).run
          (stateBefore pub proof messages ops c)).1 = .ok false →
        foldRunIn pub (stateBefore pub proof messages ops c) ℓ F = none) := by
  unfold stagesShape at shaped
  simp only [Bool.and_eq_true] at shaped
  obtain ⟨-, stagesAt⟩ := shaped
  unfold stageIndices at member
  revert stagesAt member
  rcases hp : productStages F.logBasis with
    _ | ⟨⟨arity, nodes⟩, _ | ⟨⟨arity1, nodes1⟩, _ | ⟨x, rest⟩⟩⟩ <;> intro member stagesAt
  · exact (List.not_mem_nil member).elim
  · have here : c ∈ [indexOf (skeleton L).zipIdx (isCheck0 ℓ arity)] := member
    obtain rfl := List.mem_singleton.mp here
    obtain ⟨-, ⟨key, check⟩, -, bwd⟩ := stage0_of_shape pub proof messages erased
      (by rw [hp]; rfl) stagesAt
    exact ⟨key, _, _, _, _, _, check, bwd⟩
  · simp only [Bool.and_eq_true] at stagesAt
    obtain ⟨first, second⟩ := stagesAt
    have here : c ∈ [indexOf (skeleton L).zipIdx (isCheck0 ℓ arity),
        indexOf (skeleton L).zipIdx (isCheck1 ℓ nodes arity1)] := member
    simp only [List.mem_cons, List.not_mem_nil, or_false] at here
    rcases here with rfl | rfl
    · obtain ⟨-, ⟨key, check⟩, -, bwd⟩ := stage0_of_shape pub proof messages erased
        (by rw [hp]; rfl) first
      exact ⟨key, _, _, _, _, _, check, bwd⟩
    · obtain ⟨-, ⟨key, check⟩, -, bwd⟩ := stage1_of_shape pub proof messages erased aligned
        (by rw [hp]; rfl) first second
      exact ⟨key, _, _, _, _, _, check, bwd⟩
  · exact (List.not_mem_nil member).elim

/-- On operations whose erased forms are a covered skeleton with the shapes of each level, each
check holds on the state before it, when the run before it has no stop, from two facts. `levels`:
the level function of each level is not `none` on the state before its `fused.end`. `tensors`:
`tensorRunIn` accepts on the state before each `tensor.end`. Each fact is needed only when the run
before `levelStart` of its level has no stop. -/
theorem erased_checks (erased : ops.toList.map Op.erase = skeleton L)
    (aligned : ReadsAligned messages ops.zipIdx.toList 0)
    (covered : (skeleton L).zipIdx.all (coveredAt L (skeleton L).zipIdx) = true)
    (shaped : ∀ ℓ F, L.folds[ℓ]? = some F →
      levelShaped L ℓ = true ∧ stagesShape (skeleton L).zipIdx ℓ F = true)
    (levels : ∀ ℓ F, L.folds[ℓ]? = some F →
      (runFrom pub proof messages (ops.zipIdx.toList.take (levelStart L ℓ))
        (channelStart pub)).1 = .ok () →
      foldRunIn pub (stateBefore pub proof messages ops (fusedIndex L ℓ)) ℓ F ≠ none)
    (tensors : ∀ c key level count point opening,
      ops[c]? = some (.check key (.tensor level count point opening)) →
      (runFrom pub proof messages (ops.zipIdx.toList.take (levelStart L level))
        (channelStart pub)).1 = .ok () →
      (tensorRunIn (stateBefore pub proof messages ops c) level count point opening).isSome =
        true)
    (c : ℕ) (key : Key) (check : Check) (at_ : ops[c]? = some (.check key check))
    (before : (runFrom pub proof messages (ops.zipIdx.toList.take c) (channelStart pub)).1 =
      .ok ()) :
    (((holds pub check).run).run (stateBefore pub proof messages ops c)).1 = .ok true := by
  have entry : (skeleton L)[c]? = some (.check key check) := by
    rw [← erased, List.getElem?_map, Array.getElem?_toList, at_]
    rfl
  have here : coveredAt L (skeleton L).zipIdx (.check key check, c) = true :=
    List.all_eq_true.mp covered _
      (List.mem_of_getElem? (i := c) (by rw [List.getElem?_zipIdx, entry]; simp))
  have started : ∀ {j : ℕ}, j ≤ c →
      (runFrom pub proof messages (ops.zipIdx.toList.take j) (channelStart pub)).1 = .ok () :=
    fun le => (stateBefore_le pub proof messages ops le before).1
  have within : ∀ {ℓ : ℕ}, inLevel L ℓ c = true → levelStart L ℓ ≤ c ∧ c < fusedIndex L ℓ :=
    fun h => by simpa [inLevel] using h
  have byLevel : ∀ {ℓ : ℕ} {F : Fold}, L.folds[ℓ]? = some F → inLevel L ℓ c = true →
      ((((step pub proof messages c (.check key check)).run).run
        (stateBefore pub proof messages ops c)).1 ≠ .ok () →
          foldRunIn pub (stateBefore pub proof messages ops c) ℓ F = none) →
      (((holds pub check).run).run (stateBefore pub proof messages ops c)).1 = .ok true :=
    fun {ℓ F} fold inside reject =>
      holds_of_later pub proof messages at_ (within inside).2 before
        (fun s => foldRunIn pub s ℓ F) (fun _ _ _ => rfl) (fun _ _ => rfl)
        (levels ℓ F fold (started (within inside).1)) reject
  cases check with
  | fused level F =>
      have here' : (decide (level < L.folds.length) && c == fusedIndex L level &&
          decide (levelStart L level ≤ c)) = true := here
      simp only [Bool.and_eq_true, decide_eq_true_eq, beq_iff_eq] at here'
      obtain ⟨⟨below, rfl⟩, start⟩ := here'
      have fold : L.folds[level]? = some L.folds[level] := List.getElem?_eq_getElem below
      have fusedAt := (erased_level erased fold (shaped level _ fold).1).1
      have same := fusedAt.at_
      rw [at_] at same
      cases same
      obtain ⟨r, a, e, shape⟩ := fusedAt.shape
      by_contra bad
      exact levels level _ fold (started start)
        (foldRunIn_of_reject pub proof messages _ (channelStart_fresh pub)
          (channelStart_empty pub) _ ⟨level, "fused.end", none⟩ (fusedIndex L level) level _ shape
          before fun ok => bad ((step_check_iff pub proof messages _ _ _ _).mp ok))
  | range lv base arities point =>
      have here' : ((L.folds[lv]?.any fun F =>
          isRangeCheck lv F (.check key (.range lv base arities point))) &&
          inLevel L lv c) = true := here
      simp only [Bool.and_eq_true] at here'
      obtain ⟨F, fold, isR⟩ := (Option.any_eq_true _ _).mp here'.1
      obtain ⟨key', sameOp⟩ := isRangeCheck_sound isR
      obtain ⟨-, sameCheck⟩ := Op.check.inj sameOp
      exact byLevel fold here'.2 fun stopped =>
        foldRunIn_of_range_reject pub proof messages _ key c lv F (sameCheck ▸ stopped)
  | rangeNorm lv leaf limit arities point =>
      have here' : ((L.folds[lv]?.any fun F =>
          isRangeCheck lv F (.check key (.rangeNorm lv leaf limit arities point))) &&
          inLevel L lv c) = true := here
      simp only [Bool.and_eq_true] at here'
      obtain ⟨F, fold, isR⟩ := (Option.any_eq_true _ _).mp here'.1
      obtain ⟨key', sameOp⟩ := isRangeCheck_sound isR
      obtain ⟨-, sameCheck⟩ := Op.check.inj sameOp
      exact byLevel fold here'.2 fun stopped =>
        foldRunIn_of_range_reject pub proof messages _ key c lv F (sameCheck ▸ stopped)
  | stage nodes arity weights output children =>
      have here' : ((L.folds[output.level]?.any fun F =>
          decide (c ∈ stageIndices (skeleton L).zipIdx output.level F)) &&
          inLevel L output.level c) = true := here
      simp only [Bool.and_eq_true] at here'
      obtain ⟨F, fold, member⟩ := (Option.any_eq_true _ _).mp here'.1
      obtain ⟨key', n', a', w', o', ch', atC, bwd⟩ := erased_stage_reject pub proof messages
        erased aligned (shaped _ F fold).2 (of_decide_eq_true member)
      rw [at_] at atC
      cases atC
      exact byLevel fold here'.2 fun stopped =>
        bwd before (holds_of_stopped pub proof messages c _ _ _ stopped)
  | energy energy limit =>
      obtain ⟨lv, site, index⟩ := energy
      have here' : ((L.folds[lv]?.any fun F => F.euclidean &&
          (⟨lv, site, index⟩ : Key) == ⟨lv, "norm.response_l2_sq", none⟩ && limit == F.limit) &&
          inLevel L lv c) = true := here
      simp only [Bool.and_eq_true] at here'
      obtain ⟨F, fold, facts⟩ := (Option.any_eq_true _ _).mp here'.1
      simp only [Bool.and_eq_true, beq_iff_eq, Key.mk.injEq, true_and] at facts
      obtain ⟨⟨euclidean, rfl, rfl⟩, rfl⟩ := facts
      exact byLevel fold here'.2 fun stopped =>
        foldRunIn_of_energy_reject pub proof messages _ key c lv F euclidean stopped
  | tensorOpening point value partials =>
      have here' : (match (skeleton L).zipIdx.findIdx?
          (fun y => isTensorOf point value partials y.1) with
        | some t => decide (c < t) && decide (levelStart L partials.level ≤ c)
        | none => false) = true := here
      split at here'
      · rename_i t found
        simp only [Bool.and_eq_true, decide_eq_true_eq] at here'
        obtain ⟨inside, hit, -⟩ := List.findIdx?_eq_some_iff_getElem.mp found
        have below : t < (skeleton L).length := by simpa using inside
        obtain ⟨key', lv, count, sameOp, rfl⟩ :=
          isTensorOf_sound (op := (skeleton L)[t]) (by simpa using hit)
        have atT : ops[t]? = some (.check key' (.tensor lv count point value)) :=
          erased_check erased (by rw [List.getElem?_eq_getElem below, sameOp])
        exact holds_of_later pub proof messages at_ here'.1 before
          (fun s => tensorRunIn s lv count point value) (fun _ _ _ => rfl) (fun _ _ => rfl)
          (Option.isSome_iff_ne_none.mp
            (tensors t key' lv count point value atT (started here'.2)))
          fun stopped => tensorRun_of_opening_reject pub proof messages c key _ count point value
            ⟨lv, "tensor.partials", none⟩ _ _ _ stopped
      · cases here'
  | tensor lv count point opening =>
      have here' : decide (levelStart L lv ≤ c) = true := here
      have value : stageValue pub (stateBefore pub proof messages ops c)
          (.tensor lv count point opening) = some (.tensor
            (tensorRunIn (stateBefore pub proof messages ops c) lv count point opening)) := rfl
      rw [holds_stage pub _ _ value]
      exact congrArg Except.ok
        (tensors c key lv count point opening at_ (started (of_decide_eq_true here')))
  | stage3 lv plan => exact Bool.noConfusion (here : false = true)

/-- The converse of `verifyProof_accept` from the level functions, for operations whose erased
forms are a covered skeleton with the shapes of each level: a proof that decodes with no byte left, whose works
hold, whose level function is not `none` on the state before each `fused.end`, whose tensor
reductions accept and whose terminal accepts, is accepted. -/
theorem runProof_accept_of_levels (scripted : script pub.layout pub.plan = .ok ops)
    (erased : ops.toList.map Op.erase = skeleton L)
    (covered : (skeleton L).zipIdx.all (coveredAt L (skeleton L).zipIdx) = true)
    (ends : endsFit (skeleton L).zipIdx = true)
    (shaped : ∀ ℓ F, L.folds[ℓ]? = some F →
      levelShaped L ℓ = true ∧ stagesShape (skeleton L).zipIdx ℓ F = true)
    {full : Array Fq} (terminalSetup : pub.setupFor pub.layout.terminal.setupCount = some full)
    (decoded : decode ops proof = .ok (messages, 0))
    (worked : works pub ops proof messages = true)
    (levels : ∀ ℓ F, L.folds[ℓ]? = some F →
      (runFrom pub proof messages (ops.zipIdx.toList.take (levelStart L ℓ))
        (channelStart pub)).1 = .ok () →
      foldRunIn pub (stateBefore pub proof messages ops (fusedIndex L ℓ)) ℓ F ≠ none)
    (tensors : ∀ c key level count point opening,
      ops[c]? = some (.check key (.tensor level count point opening)) →
      (runFrom pub proof messages (ops.zipIdx.toList.take (levelStart L level))
        (channelStart pub)).1 = .ok () →
      (tensorRunIn (stateBefore pub proof messages ops c) level count point opening).isSome =
        true)
    (terminal : ∀ c level, ops[c]? = some (.terminal level) →
      (runFrom pub proof messages (ops.zipIdx.toList.take c) (channelStart pub)).1 = .ok () →
      ∃ out, terminalRun pub.layout.terminal.shape pub.layout.terminal.limit
        pub.layout.terminal.params full
        (terminalMessagesIn (stateBefore pub proof messages ops c) level)
        (terminalChallengesIn (stateBefore pub proof messages ops c) level) = .ok out) :
    runProof pub proof = .verdict .accept :=
  verifyProof_accept_of pub proof scripted (erased_ends erased ends) terminalSetup decoded worked
    (erased_checks pub proof messages erased (decode_aligned decoded) covered shaped levels
      tensors)
    terminal

end Cover

section Key20

theorem skeleton20_covered :
    (skeleton (layout20 0 0)).zipIdx.all
      (coveredAt (layout20 0 0) (skeleton (layout20 0 0)).zipIdx) = true := by
  decide +kernel

theorem levelStart20 (zMax rice ℓ : ℕ) :
    levelStart (layout20 zMax rice) ℓ = levelStart (layout20 0 0) ℓ := rfl

/-- The fold of an index of the folds of `layout20` is `fold20` of the index. -/
theorem fold20_of {ℓ : ℕ} {F : Fold} (fold : (layout20 0 0).folds[ℓ]? = some F) :
    ∃ i : Fin 6, i.val = ℓ ∧ F = fold20 i := by
  have below : ℓ < 6 := (List.getElem?_eq_some_iff.mp fold).1
  refine ⟨⟨ℓ, below⟩, rfl, ?_⟩
  have known := layout20_fold 0 0 ⟨ℓ, below⟩
  rw [fold] at known
  exact Option.some.inj known

variable (pub : Public) (proof : ByteArray)

/-- The converse of `verifyProof_accept` for `layout20` from the level functions: a proof that
decodes with no byte left, whose works hold, whose level function is not `none` on the state
before the `fused.end` of each level, whose tensor reductions accept on the state before each
`tensor.end` and whose terminal accepts, is accepted. `levels` and `tensors` are needed only when
the run before `levelStart` of the level has no stop. -/
theorem verifyProof_accept_of_levels {zMax rice : ℕ} (layout : pub.layout = layout20 zMax rice)
    (scripted : script pub.layout pub.plan = .ok pub.ops) {full : Array Fq}
    (terminalSetup : pub.setupFor pub.layout.terminal.setupCount = some full)
    {messages : Array Message} (decoded : decode pub.ops proof = .ok (messages, 0))
    (worked : works pub pub.ops proof messages = true)
    (levels : ∀ ℓ : Fin 6,
      (runFrom pub proof messages (pub.ops.zipIdx.toList.take (levelStart pub.layout ℓ))
        (channelStart pub)).1 = .ok () →
      foldRunIn pub (stateBefore pub proof messages pub.ops (fusedIndex pub.layout ℓ)) ℓ
        (fold20 ℓ) ≠ none)
    (tensors : ∀ c key level count point opening,
      pub.ops[c]? = some (.check key (.tensor level count point opening)) →
      (runFrom pub proof messages (pub.ops.zipIdx.toList.take (levelStart pub.layout level))
        (channelStart pub)).1 = .ok () →
      (tensorRunIn (stateBefore pub proof messages pub.ops c) level count point opening).isSome =
        true)
    (terminal : ∀ c level, pub.ops[c]? = some (.terminal level) →
      (runFrom pub proof messages (pub.ops.zipIdx.toList.take c) (channelStart pub)).1 = .ok () →
      ∃ out, terminalRun pub.layout.terminal.shape pub.layout.terminal.limit
        pub.layout.terminal.params full
        (terminalMessagesIn (stateBefore pub proof messages pub.ops c) level)
        (terminalChallengesIn (stateBefore pub proof messages pub.ops c) level) = .ok out) :
    runProof pub proof = .verdict .accept := by
  have erased : pub.ops.toList.map Op.erase = skeleton (layout20 0 0) := by
    rw [script_erase scripted, layout, skeleton20]
  refine runProof_accept_of_levels pub proof messages scripted erased skeleton20_covered
    skeleton20_ends ?_ terminalSetup decoded worked ?_ ?_ terminal
  · intro ℓ F fold
    obtain ⟨i, rfl, rfl⟩ := fold20_of fold
    exact ⟨skeleton20_shaped i, skeleton20_stages i⟩
  · intro ℓ F fold started
    obtain ⟨i, rfl, rfl⟩ := fold20_of fold
    have known := levels i (by rw [layout, levelStart20]; exact started)
    rw [layout, fusedIndex20] at known
    exact known
  · intro c key level count point opening at_ started
    exact tensors c key level count point opening at_ (by rw [layout, levelStart20]; exact started)

/-- The guard of `verifyProof_accept_of_levels`: an accepted proof of `layout20` has each of its
hypotheses. -/
theorem verifyProof_levels_of_accept {zMax rice : ℕ} (layout : pub.layout = layout20 zMax rice)
    {full : Array Fq} (terminalSetup : pub.setupFor pub.layout.terminal.setupCount = some full)
    (accepted : runProof pub proof = .verdict .accept) :
    script pub.layout pub.plan = .ok pub.ops ∧
      ∃ messages, decode pub.ops proof = .ok (messages, 0) ∧
        works pub pub.ops proof messages = true ∧
        (∀ ℓ : Fin 6, foldRunIn pub
          (stateBefore pub proof messages pub.ops (fusedIndex pub.layout ℓ)) ℓ (fold20 ℓ) ≠
            none) ∧
        (∀ c key level count point opening,
          pub.ops[c]? = some (.check key (.tensor level count point opening)) →
          (tensorRunIn (stateBefore pub proof messages pub.ops c) level count point
            opening).isSome = true) ∧
        ∀ c level, pub.ops[c]? = some (.terminal level) →
          ∃ out, terminalRun pub.layout.terminal.shape pub.layout.terminal.limit
            pub.layout.terminal.params full
            (terminalMessagesIn (stateBefore pub proof messages pub.ops c) level)
            (terminalChallengesIn (stateBefore pub proof messages pub.ops c) level) = .ok out := by
  obtain ⟨scripted, ends, fused⟩ := verifyProof_script20 layout accepted
  obtain ⟨messages, decoded, worked, -, -, folds, -, tensors, terminal⟩ :=
    verifyProof_accept pub proof scripted ends terminalSetup accepted
  refine ⟨scripted, messages, decoded, worked, fun ℓ => ?_, tensors, terminal⟩
  obtain ⟨r, a, e, shape⟩ := (fused ℓ).shape
  rw [folds _ _ _ _ r a e (fused ℓ).at_ shape]
  exact Option.some_ne_none _

/-- `runProof` accepts a proof of `layout20` exactly when the layout takes the plan, the proof
decodes with no byte left, the works hold, the level function of each level is not `none` on the
state before its `fused.end`, the tensor reduction accepts on the state before each `tensor.end`,
and `terminalRun` accepts on the state before the terminal. -/
theorem verifyProof_accept_iff_levels {zMax rice : ℕ} (layout : pub.layout = layout20 zMax rice)
    {full : Array Fq} (terminalSetup : pub.setupFor pub.layout.terminal.setupCount = some full) :
    runProof pub proof = .verdict .accept ↔
      script pub.layout pub.plan = .ok pub.ops ∧
        ∃ messages, decode pub.ops proof = .ok (messages, 0) ∧
          works pub pub.ops proof messages = true ∧
          (∀ ℓ : Fin 6, foldRunIn pub
            (stateBefore pub proof messages pub.ops (fusedIndex pub.layout ℓ)) ℓ (fold20 ℓ) ≠
              none) ∧
          (∀ c key level count point opening,
            pub.ops[c]? = some (.check key (.tensor level count point opening)) →
            (tensorRunIn (stateBefore pub proof messages pub.ops c) level count point
              opening).isSome = true) ∧
          ∀ c level, pub.ops[c]? = some (.terminal level) →
            ∃ out, terminalRun pub.layout.terminal.shape pub.layout.terminal.limit
              pub.layout.terminal.params full
              (terminalMessagesIn (stateBefore pub proof messages pub.ops c) level)
              (terminalChallengesIn (stateBefore pub proof messages pub.ops c) level) =
                .ok out := by
  constructor
  · exact verifyProof_levels_of_accept pub proof layout terminalSetup
  · rintro ⟨scripted, messages, decoded, worked, levels, tensors, terminal⟩
    exact verifyProof_accept_of_levels pub proof layout scripted terminalSetup decoded worked
      (fun ℓ _ => levels ℓ) (fun c key level count point opening at_ _ =>
        tensors c key level count point opening at_)
      (fun c level at_ _ => terminal c level at_)

end Key20

end NightstreamFPrime.Spec.Compression.Akita.Replay
