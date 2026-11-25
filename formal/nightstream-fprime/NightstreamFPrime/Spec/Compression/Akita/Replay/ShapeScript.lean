import NightstreamFPrime.Spec.Compression.Akita.Replay.Script

/-! The operations of `script` up to the bits of the plan are a function of the layout (Akita
revision `d50f700`; `Replay/Script.lean`).

- `Op.erase` sets the bits of a work, the width of a response nonce and the most bytes of the
  payload of `z` to zero: the plan gives the first two, and no shape of the replay reads them.
- `Skeleton` follows each function of `Build` and gives its operations with `Op.erase` as a list;
  `skeleton L` is the list of the whole script of the layout `L`.
- `Build.Emits m out`: from each state where `m` succeeds, it appends operations whose erased
  forms are `out`. `script_erase`: each script that the layout takes from a plan has the erased
  operations `skeleton L`. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita.Replay

/-- An operation without the numbers that the plan gives (the bits of a work, the width of a
response nonce) and without the most bytes of the payload of `z`. -/
def Op.erase : Op → Op
  | .absorb key value => .absorb key value
  | .fields key count => .fields key count
  | .exts key count => .exts key count
  | .integer key => .integer key
  | .digest key => .digest key
  | .bounded key _ => .bounded key 0
  | .work site level _ => .work site level 0
  | .challenge key count => .challenge key count
  | .foldNonce level _ => .foldNonce level 0
  | .squeeze level group G => .squeeze level group G
  | .ended key rounds before => .ended key rounds before
  | .check key c => .check key c
  | .terminal level => .terminal level

namespace Skeleton

def challenges (level : ℕ) (site : Site) (site' : String) (count : ℕ) : List Op :=
  [.work site level 0, .challenge ⟨level, site', none⟩ count]

def round (level : ℕ) (name : String) (protocol : Protocol) (stage degree r : ℕ) : List Op :=
  .exts ⟨level, name ++ ".round", some r⟩ degree ::
    challenges level (.round protocol level stage r) (name ++ ".point") 1

def sumcheck (level : ℕ) (name : String) (protocol : Protocol) (stage : ℕ) (claim : Value)
    (rounds degree : ℕ) (before : Before := .plain) : List Op :=
  .absorb ⟨level, name ++ ".claim", none⟩ claim ::
    ((List.range rounds).flatMap (round level name protocol stage degree) ++
      [.ended ⟨level, name, none⟩ rounds before])

def foldChallenges (level : ℕ) (groups : List FoldGroup) : List Op :=
  .foldNonce level 0 :: .work (.foldChallengeWork level) level 0 ::
    groups.zipIdx.flatMap fun x => [.squeeze level x.2 x.1]

def tensor (level rounds : ℕ) (point opening : Key) : List Op :=
  .absorb ⟨level, "tensor.openings", none⟩ (.recorded [opening]) ::
    .exts ⟨level, "tensor.partials", none⟩ 2 ::
    (challenges level (.extensionPoint level) "tensor.eta" 1 ++
      .check ⟨level, "tensor.opening", none⟩
        (.tensorOpening point opening ⟨level, "tensor.partials", none⟩) ::
      (sumcheck level "tensor.sumcheck" .extensionOpeningReduction 0
          (.tensor ⟨level, "tensor.partials", none⟩ ⟨level, "tensor.eta", none⟩) rounds 2 ++
        [.exts ⟨level, "tensor.final_claims", none⟩ 1,
          .check ⟨level, "tensor.end", none⟩ (.tensor level rounds point opening)]))

/-- The product stages of the range proof, then the norm check on the Euclidean route or the leaf
stage. -/
def range (level : ℕ) (F : Fold) : List Op :=
  let key (site : String) : Key := ⟨level, site, none⟩
  let stages := productStages F.logBasis
  let batched (k : ℕ) : Option (Key × Key) :=
    if k = 0 then none else some (key s!"range.{k - 1}.children", key s!"range.{k - 1}.batch")
  let claim (k : ℕ) : Value := match batched k with
    | some (children, gamma) => .batched children gamma
    | none => .zero
  let before (k : ℕ) : Before :=
    if k = 0 then .turned (key "tau0") F.low else .point (key s!"range.{k - 1}.point")
  let weights (k : ℕ) : Weights :=
    if k = 0 then .ones else .powers (key s!"range.{k - 1}.batch")
  let nodes (k : ℕ) : ℕ := if k = 0 then 1 else ((stages[k - 1]?).map (·.2)).getD 1
  stages.zipIdx.flatMap (fun x =>
      sumcheck level s!"range.{x.2}" .stage1 x.2 (claim x.2) F.vars x.1.1 (before x.2) ++
        (.exts (key s!"range.{x.2}.children") x.1.2 ::
          .check (key s!"range.{x.2}.end") (.stage (nodes x.2) x.1.1 (weights x.2)
            (key s!"range.{x.2}.output") (key s!"range.{x.2}.children")) ::
          challenges level (.interstage level x.2) s!"range.{x.2}.batch" 1)) ++
    if F.euclidean then
      .integer (key "norm.response_l2_sq") ::
        (challenges level (.merge level) "norm.merge" 1 ++
          (sumcheck level "norm.sumcheck" .physicalL2 0
              (.norm (batched stages.length) (key "norm.merge") (key "norm.response_l2_sq")) F.vars
              (leafDegree F.logBasis + 1) ++
            (.exts (key "norm.virtual_evaluations") 1 :: .exts (key "range.image") 1 ::
              .check (key "norm.integer") (.energy (key "norm.response_l2_sq") F.limit) ::
              .check (key "norm.end") (.rangeNorm level (leafDegree F.logBasis) F.limit
                (stages.map (·.1)) (before 0)) ::
              challenges level (.virtualBatch level) "norm.virtual_batch" 1)))
    else
      sumcheck level "range.leaf" .stage1 stages.length (claim stages.length) F.vars
          (leafDegree F.logBasis) (before stages.length) ++
        [.exts (key "range.image") 1, .check (key "range.leaf.end")
          (.range level (2 ^ F.logBasis) (stages.map (·.1)) (before 0))]

/-- The operations of Stage 3 and of `fused.end` after `witness_evaluation`. -/
def close (level : ℕ) (F : Fold) : List Op :=
  let key (site : String) : Key := ⟨level, site, none⟩
  (match F.stage3 with
    | some rounds =>
        .absorb (key "stage3.slot") .slot :: .exts (key "stage3.setup_claim") 1 ::
          (sumcheck level "stage3" .stage3 0 (.recorded [key "stage3.setup_claim"]) rounds 2 ++
            .exts (key "stage3.prefix_evaluation") 1 ::
            match F.setup with
            | some setup => [.check (key "stage3.end") (.stage3 level setup)]
            | none => [])
    | none => []) ++
    [.check (key "fused.end") (.fused level F)]

def fold (level : ℕ) (F : Fold) : List Op :=
  let key (site : String) : Key := ⟨level, site, none⟩
  [.fields (key "opening_payload") F.payload] ++
    (if 1 < F.groups.length then
      challenges level (.evaluationBatch level) "row_coefficients" F.groups.length else []) ++
    foldChallenges level F.groups ++
    (if F.last then
      [.fields (key "terminal_t") F.witness, .digest (key "terminal_t_last_block_digest")]
    else [.fields (key "next_witness") F.witness]) ++
    challenges level (.ringSwitchAlpha level) "alpha" 1 ++
    challenges level (.tau0 level) "tau0" F.vars ++
    challenges level (.tau1 level) "tau1" F.batch ++
    range level F ++
    (if F.compressed then challenges level (.binary level) "binary_batch" 1 else []) ++
    challenges level (.stage2Batch level) "stage2_batch" 1 ++
    sumcheck level "fused" .stage2 0 (.fused F) F.vars 3 ++
    [.exts (key "witness_evaluation") 1] ++ close level F

def prefix_ (level : ℕ) (F : Fold) (before : Option Fold) : List Op :=
  let key (site : String) : Key := ⟨level, site, none⟩
  let back (site : String) : Key := ⟨level - 1, site, none⟩
  let setup := (before.bind (·.stage3)).isSome
  match before, F.trace with
  | none, _ =>
      (List.range F.groups.length).flatMap (fun g =>
          [.absorb ⟨0, "root.commitment_digest", some g⟩ (.digest g)]) ++
        ((List.range F.groups.length).flatMap (fun g =>
            [.absorb (key s!"root.point.{g}") (.point g)]) ++
          [.absorb (key "openings") .values])
  | some _, none =>
      (((if setup then [back "stage3.point"] else []) ++ [back "fused.point"]).zipIdx.flatMap
          fun x => [.absorb (key s!"point.{x.2}") (.recorded [x.1])]) ++
        [.absorb (key "openings") (.recorded
          ((if setup then [back "stage3.prefix_evaluation"] else []) ++
            [back "witness_evaluation"]))]
  | some _, some (rounds, degree) =>
      tensor level rounds (back "fused.point") (back "witness_evaluation") ++
        [.absorb (key "point.0") (.packed degree (key "tensor.sumcheck.point"))]

def terminal (level : ℕ) (T : Terminal) : List Op :=
  let key (site : String) : Key := ⟨level, site, none⟩
  tensor level T.rounds ⟨level - 1, "fused.point", none⟩ ⟨level - 1, "witness_evaluation", none⟩ ++
    .absorb (key "terminal.point") (.packed T.degree (key "tensor.sumcheck.point")) ::
    .fields (key "terminal.e") T.e :: .digest (key "terminal.e_last_block_digest") ::
    (foldChallenges level [T.group] ++ [.bounded (key "terminal.z") 0, .terminal level])

/-- The operations of the level `x.2` with its fold `x.1.1` and the fold before it `x.1.2`. -/
def level (x : (Fold × Option Fold) × ℕ) : List Op := prefix_ x.2 x.1.1 x.1.2 ++ fold x.2 x.1.1

end Skeleton

/-- The erased operations of the script of the layout `L`. -/
def skeleton (L : Layout) : List Op :=
  (L.folds.zip (none :: L.folds.map some)).zipIdx.flatMap Skeleton.level ++
    Skeleton.terminal L.folds.length L.terminal

namespace Build

/-- From each state where `m` succeeds, it appends operations whose erased forms are `out`. -/
def Emits {α : Type} (m : Build α) (out : List Op) : Prop :=
  ∀ s a s', m s = .ok (a, s') → s'.2.toList.map Op.erase = s.2.toList.map Op.erase ++ out

/-- The body of a loop: `Emits`, and each result is `yield`. -/
def Steps (m : Build (ForInStep Unit)) (out : List Op) : Prop :=
  ∀ s r s', m s = .ok (r, s') → r = .yield () ∧
    s'.2.toList.map Op.erase = s.2.toList.map Op.erase ++ out

section Emits

theorem Emits.of_eq {α : Type} {m : Build α} {o o' : List Op} (h : Emits m o) (same : o = o') :
    Emits m o' := same ▸ h

theorem Emits.bind {α β : Type} {m : Build α} {f : α → Build β} {o₁ o₂ : List Op}
    (hm : Emits m o₁) (hf : ∀ a, Emits (f a) o₂) : Emits (m >>= f) (o₁ ++ o₂) := by
  intro s b s'' h
  have h' : StateT.bind m f s = .ok (b, s'') := h
  unfold StateT.bind at h'
  rcases hs : m s with e | ⟨a, s'⟩
  · rw [hs] at h'
    cases h'
  · rw [hs] at h'
    rw [hf a s' b s'' h', hm s a s' hs, List.append_assoc]

theorem Steps.of_eq {m : Build (ForInStep Unit)} {o o' : List Op} (h : Steps m o)
    (same : o = o') : Steps m o' := same ▸ h

theorem Steps.bind {α : Type} {m : Build α} {f : α → Build (ForInStep Unit)} {o₁ o₂ : List Op}
    (hm : Emits m o₁) (hf : ∀ a, Steps (f a) o₂) : Steps (m >>= f) (o₁ ++ o₂) := by
  intro s b s'' h
  have h' : StateT.bind m f s = .ok (b, s'') := h
  unfold StateT.bind at h'
  rcases hs : m s with e | ⟨a, s'⟩
  · rw [hs] at h'
    cases h'
  · rw [hs] at h'
    obtain ⟨yielded, later⟩ := hf a s' b s'' h'
    rw [later, hm s a s' hs, List.append_assoc]
    exact ⟨yielded, rfl⟩

theorem emits_pure {α : Type} (a : α) : Emits (pure a : Build α) [] := by
  intro s b s' h
  cases h
  simp

theorem steps_yield : Steps (pure (.yield ()) : Build (ForInStep Unit)) [] := by
  intro s b s' h
  cases h
  simp

theorem emits_throw {α : Type} (e : String) (o : List Op) : Emits (throw e : Build α) o := by
  intro s b s' h
  cases h

theorem emits_throw_bind {α β : Type} (e : String) (f : α → Build β) (o : List Op) :
    Emits (throw e >>= f) o := by
  intro s b s' h
  cases h

theorem steps_throw_bind {α : Type} (e : String) (f : α → Build (ForInStep Unit)) (o : List Op) :
    Steps (throw e >>= f) o := by
  intro s b s' h
  cases h

theorem emits_ite {α : Type} (c : Prop) [Decidable c] {A B : Build α} {o : List Op}
    (hA : Emits A o) (hB : Emits B o) : Emits (if c then A else B) o := by
  split
  · exact hA
  · exact hB

theorem steps_ite (c : Prop) [Decidable c] {A B : Build (ForInStep Unit)} {o : List Op}
    (hA : Steps A o) (hB : Steps B o) : Steps (if c then A else B) o := by
  split
  · exact hA
  · exact hB

theorem emits_emit (op : Op) : Emits (emit op) [op.erase] := by
  intro s a s' h
  obtain ⟨n, ops⟩ := s
  cases h
  simp

theorem emits_emit_work (site : Site) (level bits : ℕ) :
    Emits (emit (.work site level bits)) [.work site level 0] := emits_emit _

theorem emits_emit_nonce (level width : ℕ) :
    Emits (emit (.foldNonce level width)) [.foldNonce level 0] := emits_emit _

theorem emits_emit_bounded (key : Key) (max : ℕ) :
    Emits (emit (.bounded key max)) [.bounded key 0] := emits_emit _

theorem emits_get : Emits (get : Build (ℕ × Array Op)) [] := by
  intro s a s' h
  cases h
  simp

theorem emits_take (plan : Array Run) (site : Site) (loss : ℕ) : Emits (take plan site loss) [] := by
  intro s a s' h
  obtain ⟨n, ops⟩ := s
  unfold take at h
  simp only [bind, StateT.bind, get, getThe, MonadStateOf.get, StateT.get, pure, Except.pure,
    Except.bind] at h
  rcases hp : plan[n]? with _ | run <;> simp only [hp] at h
  · cases h
  · by_cases hs : (run.site == site.text) = true
    · by_cases hl : run.loss = loss
      · simp only [hs, hl, ↓reduceIte] at h
        cases h
        simp
      · simp only [hs, hl, ↓reduceIte] at h
        cases h
    · simp only [hs, Bool.false_eq_true, ↓reduceIte] at h
      cases h

theorem emits_forIn {β : Type} (l : List β) (f : β → Unit → Build (ForInStep Unit))
    (g : β → List Op) (each : ∀ x, Steps (f x ()) (g x)) :
    Emits (forIn l () f) (l.flatMap g) := by
  induction l with
  | nil =>
      rw [List.forIn_nil]
      exact emits_pure ()
  | cons x l ih =>
      rw [List.forIn_cons, List.flatMap_cons]
      intro s a s'' h
      have h' : StateT.bind (f x ()) _ s = .ok (a, s'') := h
      unfold StateT.bind at h'
      rcases hs : f x () s with e | ⟨r, s'⟩
      · rw [hs] at h'
        cases h'
      · rw [hs] at h'
        obtain ⟨rfl, first⟩ := each x s r s' hs
        rw [ih s' a s'' h', first, List.append_assoc]

theorem emits_forIn_range (n : ℕ) (f : ℕ → Unit → Build (ForInStep Unit)) (g : ℕ → List Op)
    (each : ∀ r, Steps (f r ()) (g r)) :
    Emits (forIn [:n] () f) ((List.range n).flatMap g) := by
  rw [Std.Legacy.Range.forIn_eq_forIn_range']
  have size : ([:n] : Std.Legacy.Range).size = n := by simp [Std.Legacy.Range.size]
  rw [size, List.range_eq_range']
  exact emits_forIn _ f g each

theorem emits_work (plan : Array Run) (level : ℕ) (site : Site) (loss : ℕ) :
    Emits (work plan level site loss) [.work site level 0] := by
  unfold work
  dsimp only
  exact (Emits.bind (emits_take plan site loss) fun run =>
    emits_ite _ (emits_emit_work _ _ _) (emits_throw_bind _ _ _)).of_eq rfl

theorem emits_challenges (plan : Array Run) (level : ℕ) (site : Site) (site' : String)
    (count loss : ℕ) :
    Emits (challenges plan level site site' count loss)
      (Skeleton.challenges level site site' count) := by
  unfold challenges
  exact (Emits.bind (emits_work plan level site loss) fun _ => emits_emit _).of_eq rfl

theorem emits_sumcheck (plan : Array Run) (level : ℕ) (name : String) (protocol : Protocol)
    (stage : ℕ) (claim : Value) (rounds degree : ℕ) (before : Before) :
    Emits (sumcheck plan level name protocol stage claim rounds degree before)
      (Skeleton.sumcheck level name protocol stage claim rounds degree before) := by
  unfold sumcheck
  refine (Emits.bind (emits_emit _) fun _ => Emits.bind (emits_forIn_range rounds _
    (Skeleton.round level name protocol stage degree) fun r => ?_) fun _ => emits_emit _).of_eq rfl
  exact (Steps.bind (emits_emit _) fun _ => Steps.bind (emits_challenges _ _ _ _ _ _) fun _ =>
    steps_yield).of_eq rfl

theorem emits_foldChallenges (plan : Array Run) (level : ℕ) (groups : List FoldGroup) :
    Emits (foldChallenges plan level groups) (Skeleton.foldChallenges level groups) := by
  unfold foldChallenges
  refine (Emits.bind (emits_take plan _ 0) fun run => Emits.bind (emits_emit_nonce _ _) fun _ =>
    Emits.bind (emits_work plan _ _ _) fun _ => Emits.bind (emits_forIn groups.zipIdx _
      (fun x => [.squeeze level x.2 x.1]) fun x => ?_) fun _ => emits_pure ()).of_eq
        (by simp [Skeleton.foldChallenges])
  obtain ⟨G, g⟩ := x
  dsimp only
  exact (Steps.bind (emits_emit _) fun _ => Steps.bind (emits_take plan _ 0) fun run =>
    steps_ite _ steps_yield (steps_throw_bind _ _ _)).of_eq rfl

theorem emits_tensor (plan : Array Run) (level rounds : ℕ) (point opening : Key) :
    Emits (tensor plan level rounds point opening) (Skeleton.tensor level rounds point opening) := by
  unfold tensor
  dsimp only
  exact (Emits.bind (emits_emit _) fun _ => Emits.bind (emits_emit _) fun _ =>
    Emits.bind (emits_challenges _ _ _ _ _ _) fun _ => Emits.bind (emits_emit _) fun _ =>
    Emits.bind (emits_sumcheck _ _ _ _ _ _ _ _ _) fun _ => Emits.bind (emits_emit _) fun _ =>
    emits_emit _).of_eq rfl

theorem emits_range (plan : Array Run) (level : ℕ) (F : Fold) :
    Emits (range plan level F) (Skeleton.range level F) := by
  unfold range Skeleton.range
  dsimp only
  refine (Emits.bind (emits_forIn _ _ _ fun x => ?_) fun _ => ?_).of_eq rfl
  · obtain ⟨⟨arity, children⟩, k⟩ := x
    dsimp only
    exact (Steps.bind (emits_sumcheck _ _ _ _ _ _ _ _ _) fun _ => Steps.bind (emits_emit _) fun _ =>
      Steps.bind (emits_emit _) fun _ => Steps.bind (emits_challenges _ _ _ _ _ _) fun _ =>
      steps_yield).of_eq rfl
  · split
    · exact (Emits.bind (emits_emit _) fun _ => Emits.bind (emits_challenges _ _ _ _ _ _) fun _ =>
        Emits.bind (emits_sumcheck _ _ _ _ _ _ _ _ _) fun _ => Emits.bind (emits_emit _) fun _ =>
        Emits.bind (emits_emit _) fun _ => Emits.bind (emits_emit _) fun _ =>
        Emits.bind (emits_emit _) fun _ => emits_challenges _ _ _ _ _ _).of_eq rfl
    · exact (Emits.bind (emits_sumcheck _ _ _ _ _ _ _ _ _) fun _ => Emits.bind (emits_emit _) fun _ =>
        emits_emit _).of_eq rfl

/-- The steps of a chain of `Emits` and `bind`. -/
macro "emits_chain" : tactic => `(tactic| repeat' (first
  | with_reducible exact emits_emit_work _ _ _
  | with_reducible exact emits_emit_nonce _ _
  | with_reducible exact emits_emit_bounded _ _
  | with_reducible exact emits_emit _
  | with_reducible exact emits_take _ _ _
  | with_reducible exact emits_work _ _ _ _
  | with_reducible exact emits_challenges _ _ _ _ _ _
  | with_reducible exact emits_sumcheck _ _ _ _ _ _ _ _ _
  | with_reducible exact emits_foldChallenges _ _ _
  | with_reducible exact emits_tensor _ _ _ _ _
  | with_reducible exact emits_range _ _ _
  | with_reducible exact emits_pure _
  | intro (_ : Unit)
  | intro (_ : Run)
  | with_reducible apply Emits.bind))

theorem emits_fold (plan : Array Run) (level : ℕ) (F : Fold) :
    Emits (fold plan level F) (Skeleton.fold level F) := by
  unfold fold Skeleton.fold Skeleton.close
  by_cases h1 : 1 < F.groups.length <;> cases F.last <;> cases F.compressed <;>
    rcases F.stage3 with _ | rounds <;> rcases F.setup with _ | setup <;>
    simp only [h1, ↓reduceIte, Bool.false_eq_true] <;>
    exact Emits.of_eq (by emits_chain) (by simp [Op.erase])

theorem emits_prefix (plan : Array Run) (level : ℕ) (F : Fold) (before : Option Fold) :
    Emits (prefix_ plan level F before) (Skeleton.prefix_ level F before) := by
  unfold prefix_ Skeleton.prefix_
  rcases before with _ | B
  · dsimp only
    exact (Emits.bind (emits_forIn_range _ _
      (fun g => [.absorb ⟨0, "root.commitment_digest", some g⟩ (.digest g)])
        fun g => (Steps.bind (emits_emit _) fun _ => steps_yield).of_eq rfl) fun _ =>
      Emits.bind (emits_forIn_range _ _
        (fun g => [.absorb ⟨level, s!"root.point.{g}", none⟩ (.point g)])
          fun g => (Steps.bind (emits_emit _) fun _ => steps_yield).of_eq rfl) fun _ =>
      emits_emit _).of_eq rfl
  · rcases F.trace with _ | ⟨rounds, degree⟩
    · dsimp only
      refine (Emits.bind (emits_forIn _ _ (fun x => [.absorb ⟨level, s!"point.{x.2}", none⟩
        (.recorded [x.1])]) fun x => ?_) fun _ => emits_emit _).of_eq ?_
      · obtain ⟨point, g⟩ := x
        exact (Steps.bind (emits_emit _) fun _ => steps_yield).of_eq (by simp [Op.erase])
      · simp [Op.erase]
    · dsimp only
      exact emits_ite _ ((Emits.bind (emits_tensor _ _ _ _ _) fun _ => emits_emit _).of_eq rfl)
        (emits_throw_bind _ _ _)

theorem emits_terminal (plan : Array Run) (level : ℕ) (T : Terminal) :
    Emits (terminal plan level T) (Skeleton.terminal level T) := by
  unfold terminal
  dsimp only
  exact (Emits.bind (emits_tensor _ _ _ _ _) fun _ => Emits.bind (emits_emit _) fun _ =>
    Emits.bind (emits_emit _) fun _ => Emits.bind (emits_emit _) fun _ =>
    Emits.bind (emits_foldChallenges _ _ _) fun _ => Emits.bind (emits_emit_bounded _ _) fun _ =>
    emits_emit _).of_eq rfl

end Emits

end Build

theorem except_bind_ok {ε α β : Type} {m : Except ε α} {f : α → Except ε β} {b : β}
    (h : (m >>= f) = .ok b) : ∃ a, m = .ok a ∧ f a = .ok b := by
  rcases hm : m with e | a
  · rw [hm] at h
    cases h
  · rw [hm] at h
    exact ⟨a, rfl, h⟩

open Build in
/-- Each script that the layout `L` takes from a plan has the erased operations `skeleton L`. -/
theorem script_erase {L : Layout} {plan : Array Run} {ops : Array Op}
    (scripted : script L plan = .ok ops) : ops.toList.map Op.erase = skeleton L := by
  unfold script at scripted
  dsimp only at scripted
  obtain ⟨⟨⟨⟩, n, ops'⟩, ran, found⟩ := except_bind_ok scripted
  cases found
  have emitted := (?_ : Emits _ (skeleton L)) (0, #[]) () (n, ops') ran
  · simpa using emitted
  refine (Emits.bind (emits_forIn _ _ Skeleton.level fun x => ?_) fun _ =>
    Emits.bind (emits_terminal _ _ _) fun _ => Emits.bind (o₂ := []) emits_get fun x => ?_).of_eq ?_
  · obtain ⟨⟨F, before⟩, level⟩ := x
    exact (Steps.bind (emits_prefix _ _ _ _) fun _ => Steps.bind (emits_fold _ _ _) fun _ =>
      steps_yield).of_eq (by simp [Skeleton.level])
  · exact emits_ite _ (emits_pure ()) (emits_throw _ _)
  · simp [skeleton]

end NightstreamFPrime.Spec.Compression.Akita.Replay
