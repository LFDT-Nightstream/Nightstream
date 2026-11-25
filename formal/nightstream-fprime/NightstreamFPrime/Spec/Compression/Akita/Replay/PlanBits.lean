import NightstreamFPrime.Spec.Compression.Akita.Replay.ShapeScript

/-! The work bits of each script of a layout are a function of the layout (Akita revision
`d50f700`: `akita-config`, `derive_transcript_grinding_plan`; `akita-params`,
`derive_transcript_grinding_plan_from_public_shape`, `GrindingRun::proof_of_work`,
`GrindingRun::fold_challenge_work`; `Replay/Script.lean`).

- `Op.bits` is the work bits of a work. A `Rule` is the input of the rule of the planner for one
  work: the loss of a field site, or the fold coordinates and the set bits of a fold. `PlanBits`
  follows each function of `Build` and gives the rules of its works in order; `planBits L` is the
  list of the work bits of the whole script of `L`.
- `Build.Works m out`: from each state where `m` succeeds, it appends operations whose work bits
  are the bits of the rules `out`. `script_bits`: each script that the layout takes from a plan has the work bits
  `planBits L`, for each plan.
- `work_bits_at`: the work number `k` of a list of operations has the entry `k` of its work
  bits. `isWorkOp_erase` and `workIndices_erase`: the indices of the works are those of the
  erased operations. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita.Replay

open NightstreamFPrime.Spec.Compression.Akita

/-- The work bits of a work, and no entry for each other operation. -/
def Op.bits : Op → List ℕ
  | .work _ _ bits => [bits]
  | _ => []

/-- The operation is a work. -/
def isWorkOp : Op → Bool
  | .work .. => true
  | _ => false

/-- The input of the rule of the planner for one work. -/
inductive Rule
  | field (loss : ℕ)
  | fold (coordinates setBits : ℕ)

/-- The work bits of the rule of the planner. -/
def Rule.bits : Rule → ℕ
  | .field loss => Extract.workBits loss
  | .fold coordinates setBits => Extract.foldWorkBits coordinates setBits

namespace PlanBits

def challenges (loss : ℕ) : List Rule := [.field loss]

def sumcheck (protocol : Protocol) (rounds degree : ℕ) : List Rule :=
  (List.range rounds).flatMap fun _ =>
    challenges (if protocol = .stage1 then degree + 1 else degree)

def foldChallenges (groups : List FoldGroup) : List Rule :=
  [.fold (groups.map (·.coordinates)).sum (leastSetBits groups)]

def tensor (rounds : ℕ) : List Rule :=
  challenges 1 ++ sumcheck .extensionOpeningReduction rounds 2

def range (F : Fold) : List Rule :=
  (productStages F.logBasis).zipIdx.flatMap (fun x =>
      sumcheck .stage1 F.vars x.1.1 ++ challenges (x.1.2 - 1)) ++
    if F.euclidean then
      challenges 1 ++ (sumcheck .physicalL2 F.vars (leafDegree F.logBasis + 1) ++ challenges 1)
    else sumcheck .stage1 F.vars (leafDegree F.logBasis)

def fold (F : Fold) : List Rule :=
  (if 1 < F.groups.length then challenges F.groups.length else []) ++
    foldChallenges F.groups ++
    challenges (2 * (F.groups.map (·.ring)).foldl max 0 - 1) ++
    challenges F.vars ++ challenges F.batch ++ range F ++
    (if F.compressed then challenges 1 else []) ++ challenges 1 ++
    sumcheck .stage2 F.vars 3 ++
    (match F.stage3 with
      | some rounds => sumcheck .stage3 rounds 2
      | none => [])

def prefix_ (F : Fold) (before : Option Fold) : List Rule :=
  match before, F.trace with
  | some _, some (rounds, _) => tensor rounds
  | _, _ => []

def terminal (T : Terminal) : List Rule := tensor T.rounds ++ foldChallenges [T.group]

def level (x : (Fold × Option Fold) × ℕ) : List Rule := prefix_ x.1.1 x.1.2 ++ fold x.1.1

end PlanBits

/-- The rules of the works of the script of the layout `L`, in their order. -/
def planRules (L : Layout) : List Rule :=
  (L.folds.zip (none :: L.folds.map some)).zipIdx.flatMap PlanBits.level ++
    PlanBits.terminal L.terminal

/-- The work bits of the script of the layout `L`, in the order of its works. -/
def planBits (L : Layout) : List ℕ := (planRules L).map Rule.bits

namespace Build

/-- From each state where `m` succeeds, it appends operations whose work bits are the bits of
the rules `out`. -/
def Works {α : Type} (m : Build α) (out : List Rule) : Prop :=
  ∀ s a s', m s = .ok (a, s') →
    s'.2.toList.flatMap Op.bits = s.2.toList.flatMap Op.bits ++ out.map Rule.bits

/-- The body of a loop: `Works`, and each result is `yield`. -/
def Turns (m : Build (ForInStep Unit)) (out : List Rule) : Prop :=
  ∀ s r s', m s = .ok (r, s') → r = .yield () ∧
    s'.2.toList.flatMap Op.bits = s.2.toList.flatMap Op.bits ++ out.map Rule.bits

section Works

theorem Works.of_eq {α : Type} {m : Build α} {o o' : List Rule} (h : Works m o) (same : o = o') :
    Works m o' := same ▸ h

theorem Works.bind {α β : Type} {m : Build α} {f : α → Build β} {o₁ o₂ : List Rule}
    (hm : Works m o₁) (hf : ∀ a, Works (f a) o₂) : Works (m >>= f) (o₁ ++ o₂) := by
  intro s b s'' h
  have h' : StateT.bind m f s = .ok (b, s'') := h
  unfold StateT.bind at h'
  rcases hs : m s with e | ⟨a, s'⟩
  · rw [hs] at h'
    cases h'
  · rw [hs] at h'
    rw [hf a s' b s'' h', hm s a s' hs, List.append_assoc, List.map_append]

theorem Turns.of_eq {m : Build (ForInStep Unit)} {o o' : List Rule} (h : Turns m o)
    (same : o = o') : Turns m o' := same ▸ h

theorem Turns.bind {α : Type} {m : Build α} {f : α → Build (ForInStep Unit)} {o₁ o₂ : List Rule}
    (hm : Works m o₁) (hf : ∀ a, Turns (f a) o₂) : Turns (m >>= f) (o₁ ++ o₂) := by
  intro s b s'' h
  have h' : StateT.bind m f s = .ok (b, s'') := h
  unfold StateT.bind at h'
  rcases hs : m s with e | ⟨a, s'⟩
  · rw [hs] at h'
    cases h'
  · rw [hs] at h'
    obtain ⟨yielded, later⟩ := hf a s' b s'' h'
    rw [later, hm s a s' hs, List.append_assoc, List.map_append]
    exact ⟨yielded, rfl⟩

theorem works_pure {α : Type} (a : α) : Works (pure a : Build α) [] := by
  intro s b s' h
  cases h
  simp

theorem turns_yield : Turns (pure (.yield ()) : Build (ForInStep Unit)) [] := by
  intro s b s' h
  cases h
  simp

theorem works_throw {α : Type} (e : String) (o : List Rule) : Works (throw e : Build α) o := by
  intro s b s' h
  cases h

theorem works_throw_bind {α β : Type} (e : String) (f : α → Build β) (o : List Rule) :
    Works (throw e >>= f) o := by
  intro s b s' h
  cases h

theorem turns_throw_bind {α : Type} (e : String) (f : α → Build (ForInStep Unit)) (o : List Rule) :
    Turns (throw e >>= f) o := by
  intro s b s' h
  cases h

theorem works_ite {α : Type} (c : Prop) [Decidable c] {A B : Build α} {o : List Rule}
    (hA : c → Works A o) (hB : ¬c → Works B o) : Works (if c then A else B) o := by
  split
  · exact hA ‹_›
  · exact hB ‹_›

theorem works_ite₀ {α : Type} (c : Prop) [Decidable c] {A B : Build α} {o : List Rule}
    (hA : Works A o) (hB : Works B o) : Works (if c then A else B) o :=
  works_ite c (fun _ => hA) fun _ => hB

theorem turns_ite (c : Prop) [Decidable c] {A B : Build (ForInStep Unit)} {o : List Rule}
    (hA : Turns A o) (hB : Turns B o) : Turns (if c then A else B) o := by
  split
  · exact hA
  · exact hB

/-- An operation that is not a work appends no work bits. -/
theorem works_emit (op : Op) (quiet : op.bits = [] := by rfl) : Works (emit op) [] := by
  intro s a s' h
  obtain ⟨n, ops⟩ := s
  cases h
  simp [quiet]

theorem works_emit_work (site : Site) (level bits : ℕ) (rule : Rule) (same : bits = rule.bits) :
    Works (emit (.work site level bits)) [rule] := by
  intro s a s' h
  obtain ⟨n, ops⟩ := s
  cases h
  simp [Op.bits, same]

theorem works_get : Works (get : Build (ℕ × Array Op)) [] := by
  intro s a s' h
  cases h
  simp

theorem works_take (plan : Array Run) (site : Site) (loss : ℕ) : Works (take plan site loss) [] := by
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

theorem works_forIn {β : Type} (l : List β) (f : β → Unit → Build (ForInStep Unit))
    (g : β → List Rule) (each : ∀ x, Turns (f x ()) (g x)) :
    Works (forIn l () f) (l.flatMap g) := by
  induction l with
  | nil =>
      rw [List.forIn_nil]
      exact works_pure ()
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
        rw [ih s' a s'' h', first, List.append_assoc, List.map_append]

theorem works_forIn_range (n : ℕ) (f : ℕ → Unit → Build (ForInStep Unit)) (g : ℕ → List Rule)
    (each : ∀ r, Turns (f r ()) (g r)) :
    Works (forIn [:n] () f) ((List.range n).flatMap g) := by
  rw [Std.Legacy.Range.forIn_eq_forIn_range']
  have size : ([:n] : Std.Legacy.Range).size = n := by simp [Std.Legacy.Range.size]
  rw [size, List.range_eq_range']
  exact works_forIn _ f g each

/-- A work that the plan passes has the bits of the rule. -/
theorem works_work (plan : Array Run) (level : ℕ) (site : Site) (loss bits : ℕ) (rule : Rule)
    (ruled : bits = rule.bits) : Works (work plan level site loss bits) [rule] := by
  unfold work
  dsimp only
  exact (Works.bind (works_take plan site loss) fun run =>
    works_ite _ (fun same => works_ite₀ _ (works_emit_work _ _ _ rule (same.trans ruled))
      (works_throw_bind _ _ _)) fun _ => works_throw_bind _ _ _).of_eq rfl

theorem works_challenges (plan : Array Run) (level : ℕ) (site : Site) (site' : String)
    (count loss : ℕ) :
    Works (challenges plan level site site' count loss) (PlanBits.challenges loss) := by
  unfold challenges
  exact (Works.bind (works_work plan level site loss _ (.field loss) rfl) fun _ =>
    works_emit _).of_eq rfl

theorem works_sumcheck (plan : Array Run) (level : ℕ) (name : String) (protocol : Protocol)
    (stage : ℕ) (claim : Value) (rounds degree : ℕ) (before : Before) :
    Works (sumcheck plan level name protocol stage claim rounds degree before)
      (PlanBits.sumcheck protocol rounds degree) := by
  unfold sumcheck
  refine (Works.bind (works_emit _) fun _ => Works.bind (works_forIn_range rounds _
    (fun _ => PlanBits.challenges (if protocol = .stage1 then degree + 1 else degree))
      fun r => ?_) fun _ => works_emit _).of_eq (by simp [PlanBits.sumcheck])
  exact (Turns.bind (works_emit _) fun _ => Turns.bind (works_challenges _ _ _ _ _ _) fun _ =>
    turns_yield).of_eq (by simp)

theorem works_foldChallenges (plan : Array Run) (level : ℕ) (groups : List FoldGroup) :
    Works (foldChallenges plan level groups) (PlanBits.foldChallenges groups) := by
  unfold foldChallenges
  refine Works.of_eq (Works.bind (works_take plan _ 0) fun run => works_ite₀ _
    (Works.bind (works_emit _) fun _ => Works.bind (works_work plan _ _ _ _ (.fold _ _) rfl) fun _ =>
      Works.bind (works_forIn groups.zipIdx _ (fun _ => []) fun x => ?_) fun _ => works_pure ())
        (works_throw_bind _ _ _)) ?_
  swap
  · simp [PlanBits.foldChallenges]
  obtain ⟨G, g⟩ := x
  dsimp only
  exact (Turns.bind (works_emit _) fun _ => Turns.bind (works_take plan _ 0) fun run =>
    turns_ite _ turns_yield (turns_throw_bind _ _ _)).of_eq rfl

theorem works_tensor (plan : Array Run) (level rounds : ℕ) (point opening : Key) :
    Works (tensor plan level rounds point opening) (PlanBits.tensor rounds) := by
  unfold tensor
  dsimp only
  exact (Works.bind (works_emit _) fun _ => Works.bind (works_emit _) fun _ =>
    Works.bind (works_challenges _ _ _ _ _ _) fun _ => Works.bind (works_emit _) fun _ =>
    Works.bind (works_sumcheck _ _ _ _ _ _ _ _ _) fun _ => Works.bind (works_emit _) fun _ =>
    works_emit _).of_eq (by simp [PlanBits.tensor])

theorem works_range (plan : Array Run) (level : ℕ) (F : Fold) :
    Works (range plan level F) (PlanBits.range F) := by
  unfold range PlanBits.range
  dsimp only
  refine Works.bind (works_forIn _ _ (fun x : (ℕ × ℕ) × ℕ =>
    PlanBits.sumcheck .stage1 F.vars x.1.1 ++ PlanBits.challenges (x.1.2 - 1)) fun x => ?_)
      fun _ => ?_
  · obtain ⟨⟨arity, children⟩, k⟩ := x
    dsimp only
    exact (Turns.bind (works_sumcheck _ _ _ _ _ _ _ _ _) fun _ => Turns.bind (works_emit _) fun _ =>
      Turns.bind (works_emit _) fun _ => Turns.bind (works_challenges _ _ _ _ _ _) fun _ =>
      turns_yield).of_eq (by simp)
  · split
    · exact (Works.bind (works_emit _) fun _ => Works.bind (works_challenges _ _ _ _ _ _) fun _ =>
        Works.bind (works_sumcheck _ _ _ _ _ _ _ _ _) fun _ => Works.bind (works_emit _) fun _ =>
        Works.bind (works_emit _) fun _ => Works.bind (works_emit _) fun _ =>
        Works.bind (works_emit _) fun _ => works_challenges _ _ _ _ _ _).of_eq
          (by simp)
    · exact (Works.bind (works_sumcheck _ _ _ _ _ _ _ _ _) fun _ => Works.bind (works_emit _) fun _ =>
        works_emit _).of_eq (by simp)

/-- The steps of a chain of `Works` and `bind`. -/
macro "works_chain" : tactic => `(tactic| repeat' (first
  | ((with_reducible refine works_emit _ ?_) <;> exact rfl)
  | with_reducible exact works_challenges _ _ _ _ _ _
  | with_reducible exact works_sumcheck _ _ _ _ _ _ _ _ _
  | with_reducible exact works_foldChallenges _ _ _
  | with_reducible exact works_range _ _ _
  | with_reducible exact works_pure _
  | intro (_ : Unit)
  | with_reducible apply Works.bind))

theorem works_fold (plan : Array Run) (level : ℕ) (F : Fold) :
    Works (fold plan level F) (PlanBits.fold F) := by
  unfold fold PlanBits.fold
  by_cases h1 : 1 < F.groups.length <;> cases F.last <;> cases F.compressed <;>
    rcases F.stage3 with _ | rounds <;> rcases F.setup with _ | setup <;>
    simp only [h1, ↓reduceIte, Bool.false_eq_true] <;>
    exact Works.of_eq (by works_chain) (by simp)

theorem works_prefix (plan : Array Run) (level : ℕ) (F : Fold) (before : Option Fold) :
    Works (prefix_ plan level F before) (PlanBits.prefix_ F before) := by
  unfold prefix_ PlanBits.prefix_
  rcases before with _ | B
  · dsimp only
    exact (Works.bind (works_forIn_range _ _ (fun _ => [])
        fun g => (Turns.bind (works_emit _) fun _ => turns_yield).of_eq rfl) fun _ =>
      Works.bind (works_forIn_range _ _ (fun _ => [])
          fun g => (Turns.bind (works_emit _) fun _ => turns_yield).of_eq rfl) fun _ =>
      works_emit _).of_eq (by simp)
  · rcases F.trace with _ | ⟨rounds, degree⟩
    · dsimp only
      refine (Works.bind (works_forIn _ _ (fun _ => []) fun x => ?_) fun _ =>
        works_emit _).of_eq (by simp)
      obtain ⟨point, g⟩ := x
      exact (Turns.bind (works_emit _) fun _ => turns_yield).of_eq rfl
    · dsimp only
      exact works_ite₀ _ ((Works.bind (works_tensor _ _ _ _ _) fun _ =>
        works_emit _).of_eq (by simp)) (works_throw_bind _ _ _)

theorem works_terminal (plan : Array Run) (level : ℕ) (T : Terminal) :
    Works (terminal plan level T) (PlanBits.terminal T) := by
  unfold terminal
  dsimp only
  exact (Works.bind (works_tensor _ _ _ _ _) fun _ => Works.bind (works_emit _) fun _ =>
    Works.bind (works_emit _) fun _ => Works.bind (works_emit _) fun _ =>
    Works.bind (works_foldChallenges _ _ _) fun _ => Works.bind (works_emit _) fun _ =>
    works_emit _).of_eq (by simp [PlanBits.terminal])

end Works

end Build

open Build in
/-- Each script that the layout `L` takes from a plan has the work bits `planBits L`: the bits of
the rule of the planner of the code. -/
theorem script_bits {L : Layout} {plan : Array Run} {ops : Array Op}
    (scripted : script L plan = .ok ops) : ops.toList.flatMap Op.bits = planBits L := by
  unfold script at scripted
  dsimp only at scripted
  obtain ⟨⟨⟨⟩, n, ops'⟩, ran, found⟩ := except_bind_ok scripted
  cases found
  have emitted := (?_ : Works _ (planRules L)) (0, #[]) () (n, ops') ran
  · simpa [planBits] using emitted
  refine (Works.bind (works_forIn _ _ PlanBits.level fun x => ?_) fun _ =>
    Works.bind (works_terminal _ _ _) fun _ => Works.bind (o₂ := []) works_get fun x => ?_).of_eq ?_
  · obtain ⟨⟨F, before⟩, level⟩ := x
    exact (Turns.bind (works_prefix _ _ _ _) fun _ => Turns.bind (works_fold _ _ _) fun _ =>
      turns_yield).of_eq (by simp [PlanBits.level])
  · exact works_ite₀ _ (works_pure ()) (works_throw _ _)
  · simp [planRules]

theorem isWorkOp_erase (op : Op) : isWorkOp op.erase = isWorkOp op := by cases op <;> rfl

/-- The indices of the works of a list of operations are those of the erased operations. -/
theorem workIndices_erase : ∀ (l : List Op) (n : ℕ),
    (((l.map Op.erase).zipIdx n).filter fun x => isWorkOp x.1).map (·.2) =
      ((l.zipIdx n).filter fun x => isWorkOp x.1).map (·.2)
  | [], _ => rfl
  | op :: l, n => by
      simp only [List.map_cons, List.zipIdx_cons, List.filter_cons, isWorkOp_erase]
      cases isWorkOp op
      · simpa using workIndices_erase l (n + 1)
      · simpa using workIndices_erase l (n + 1)

/-- The work number `k` of a list of operations, at the index `i`, has the entry `k` of the work
bits of the list. -/
theorem work_bits_at : ∀ (l : List Op) (n k i : ℕ),
    (((l.zipIdx n).filter fun x => isWorkOp x.1).map (·.2))[k]? = some i →
      n ≤ i ∧ ∃ site level bits, l[i - n]? = some (.work site level bits) ∧
        (l.flatMap Op.bits)[k]? = some bits
  | [], _, _, _, found => by cases found
  | op :: l, n, k, i, found => by
      rw [List.zipIdx_cons, List.filter_cons] at found
      by_cases worked : isWorkOp op = true
      · rw [ite_eq_left worked, List.map_cons] at found
        obtain ⟨site, level, bits, rfl⟩ : ∃ site level bits, op = .work site level bits := by
          cases op <;> first | exact ⟨_, _, _, rfl⟩ | cases worked
        cases k with
        | zero =>
            rw [List.getElem?_cons_zero] at found
            cases found
            exact ⟨le_rfl, site, level, bits, by simp, by simp [Op.bits]⟩
        | succ k =>
            rw [List.getElem?_cons_succ] at found
            obtain ⟨above, site', level', bits', at_, entry⟩ := work_bits_at l (n + 1) k i found
            refine ⟨by omega, site', level', bits', ?_, by simpa [Op.bits] using entry⟩
            have shift : i - n = (i - (n + 1)) + 1 := by omega
            rw [shift, List.getElem?_cons_succ]
            exact at_
      · rw [ite_eq_right worked] at found
        obtain ⟨above, site', level', bits', at_, entry⟩ := work_bits_at l (n + 1) k i found
        have none_ : op.bits = [] := by
          cases op <;> first | rfl | exact absurd rfl worked
        refine ⟨by omega, site', level', bits', ?_, by simpa [none_] using entry⟩
        have shift : i - n = (i - (n + 1)) + 1 := by omega
        rw [shift, List.getElem?_cons_succ]
        exact at_

end NightstreamFPrime.Spec.Compression.Akita.Replay
