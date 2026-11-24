import NightstreamFPrime.Spec.Compression.Akita.Replay.RunProof

/-! The range proof of a fold against its game on the path of the code (Akita revision `d50f700`;
`Build.range` of `Replay/Script.lean`; `step_range_runs` and `step_norm_runs` of
`Replay/Transcript.lean`).

- `rangeShape` is the shape of the operations before the last check of the range proof of a fold
  for the state hypotheses of its game: the challenges of `τ_0` and the sum-checks of the product
  stages and of the leaf (or norm) stage. `RangeState` is these hypotheses on a state;
  `range_state`: a run of aligned operations of this shape without a stop gives them.
- `verifyProof_range_game` and `verifyProof_norm_game`: when `verifyProof` accepts, `rangeCheck`
  (coefficient route) and `rangeTree` with `normLeafStage` (Euclidean route) run on the path of
  the recorded values of the state before the last check of the range proof to acceptance.
  `verifyProof_reject_range_game` and `verifyProof_reject_norm_game`: at a reject of that check
  they run to `none`. `normRunIn`, `normPathIn` and `normGameIn` are the norm stage, its path and
  its game on the recorded values of a state. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita.Replay

open NightstreamFPrime.Spec.Compression.Lib NightstreamFPrime.Spec.Compression.Game
open NightstreamFPrime.Spec.Compression (Game)
open NightstreamFPrime.Spec.Compression.Akita

/-- The shape of the operations `pre` before the last check of the range proof of the fold `F` at
`level` for the state hypotheses of its game. -/
def rangeShape (pre : List (Op × ℕ)) (level : ℕ) (F : Fold) : Bool :=
  drawnCount ⟨level, "tau0", none⟩ pre == some F.vars &&
    ((productStages F.logBasis).map (·.1)).zipIdx.all (fun x =>
      roundsAt pre ⟨level, s!"range.{x.2}", none⟩ F.vars x.1) &&
    (if F.euclidean then
      roundsAt pre ⟨level, "norm.sumcheck", none⟩ F.vars (leafDegree F.logBasis + 1)
    else roundsAt pre ⟨level, "range.leaf", none⟩ F.vars (leafDegree F.logBasis))

namespace Replayer

/-- The state hypotheses of the game of the range proof of the fold `F` at `level` on `s`. -/
structure RangeState (s : Replayer) (level : ℕ) (F : Fold) : Prop where
  vars : (recordedIn s ⟨level, "tau0", none⟩).size = F.vars
  stages : ∀ Q ∈ productsIn s level ((productStages F.logBasis).map (·.1)),
    Q.rounds.length = F.vars ∧ ∀ x ∈ Q.rounds, x.1.length = Q.arity
  leaf : F.euclidean = false →
    (roundsIn s ⟨level, "range.leaf", none⟩).length = F.vars ∧
      ∀ x ∈ roundsIn s ⟨level, "range.leaf", none⟩, x.1.length = leafDegree F.logBasis
  norm : F.euclidean = true →
    (roundsIn s ⟨level, "norm.sumcheck", none⟩).length = F.vars ∧
      ∀ x ∈ roundsIn s ⟨level, "norm.sumcheck", none⟩, x.1.length = leafDegree F.logBasis + 1

section State

variable (pub : Public) (proof : ByteArray) (messages : Array Message)

/-- A run of aligned operations of the shape `rangeShape` without a stop, from a state with no
recorded value and no rounds, gives the state hypotheses of the game of the range proof. -/
theorem range_state (s₀ : Replayer) (fresh : ∀ k, recordedIn s₀ k = #[])
    (empty : s₀.derived.rounds = #[]) (pre : List (Op × ℕ)) (level : ℕ) (F : Fold)
    (shape : rangeShape pre level F = true) (aligned : ReadsAligned messages pre s₀.next)
    (ok : (runFrom pub proof messages pre s₀).1 = .ok ()) :
    RangeState (runFrom pub proof messages pre s₀).2 level F := by
  simp only [rangeShape, Bool.and_eq_true, beq_iff_eq, List.all_eq_true] at shape
  obtain ⟨⟨tau0, stages⟩, route⟩ := shape
  have facts := fun (key : Key) (n d : ℕ) (at_ : roundsAt pre key n d = true) =>
    let ⟨e, shaped⟩ := roundsAt_sound at_
    rounds_facts pub proof messages s₀ fresh empty pre key n d e shaped aligned ok
  refine ⟨?_, ?_, ?_, ?_⟩
  · rw [size_drawnCount pub proof messages _ _ _ tau0 ok, fresh]
    simp
  · intro Q member
    unfold productsIn at member
    obtain ⟨⟨arity, k⟩, inside, rfl⟩ := List.mem_map.mp member
    obtain ⟨count, lengths, _⟩ := facts _ _ _ (stages (arity, k) inside)
    exact ⟨count, lengths⟩
  · intro plain
    rw [plain] at route
    simp only [Bool.false_eq_true, ↓reduceIte] at route
    obtain ⟨count, lengths, _⟩ := facts _ _ _ route
    exact ⟨count, lengths⟩
  · intro euclidean
    rw [euclidean] at route
    simp only [↓reduceIte] at route
    obtain ⟨count, lengths, _⟩ := facts _ _ _ route
    exact ⟨count, lengths⟩

theorem pointIn_turned_length (s : Replayer) (k : Key) (low : ℕ) :
    (pointIn s (.turned k low)).length = (recordedIn s k).size := by
  simp [pointIn]

end State

end Replayer

open Replayer

section Games

variable (pub : Public) (proof : ByteArray) {ops : Array Op}
variable {K : Type} [Field K] [Fintype K] [DecidableEq K] [Algebra Zq K] {φ : Ext → K}
  (hom : ExtHom φ) (same : ∀ a b : Ext, a.same b = true ↔ φ a = φ b)
include hom same

/-- The step of the last check of the range proof on the coefficient route and its game, on the
state before it, after the operations before it without a stop. -/
theorem range_game_at {messages : Array Message} {rest : ℕ}
    (decoded : decode ops proof = .ok (messages, rest)) {c : ℕ}
    (before : (runFrom pub proof messages (ops.zipIdx.toList.take c) (channelStart pub)).1 = .ok ())
    {level : ℕ} {F : Fold} (shape : rangeShape (ops.zipIdx.toList.take c) level F = true)
    {s : Replayer} (state : stateBefore pub proof messages ops c = s)
    (plain : F.euclidean = false)
    (arities : (productStages F.logBasis).map (·.1) = rangeArities (2 ^ F.logBasis))
    (leaf : leafDegree F.logBasis = rangeLeaf (2 ^ F.logBasis)) (key : Key)
    (support : ℕ → Prop) [DecidablePred support] (w : Table) :
    ((step pub proof messages c (.check key (rangeCheckOf level F))).run).run s =
        (if (rangeRun (2 ^ F.logBasis) (productsIn s level ((productStages F.logBasis).map (·.1)))
            (roundsIn s ⟨level, "range.leaf", none⟩) (firstIn s ⟨level, "range.image", none⟩)
            (pointIn s (.turned ⟨level, "tau0", none⟩ F.low))).isSome then
          (.ok (), checked (staged s key (.range (rangeRun (2 ^ F.logBasis)
            (productsIn s level ((productStages F.logBasis).map (·.1)))
            (roundsIn s ⟨level, "range.leaf", none⟩) (firstIn s ⟨level, "range.image", none⟩)
            (pointIn s (.turned ⟨level, "tau0", none⟩ F.low))))) key)
        else (.error (.reject c key.site), staged s key (.range (rangeRun (2 ^ F.logBasis)
            (productsIn s level ((productStages F.logBasis).map (·.1)))
            (roundsIn s ⟨level, "range.leaf", none⟩) (firstIn s ⟨level, "range.image", none⟩)
            (pointIn s (.turned ⟨level, "tau0", none⟩ F.low)))))) ∧
      (rangeCheck (2 ^ F.logBasis) F.vars support w
          ((pointIn s (.turned ⟨level, "tau0", none⟩ F.low)).map φ)).Runs
        (treePath φ (fun _ _ _ _ => leafPath φ (rangeLeaf (2 ^ F.logBasis))
            (roundsIn s ⟨level, "range.leaf", none⟩) (firstIn s ⟨level, "range.image", none⟩))
          (productsIn s level ((productStages F.logBasis).map (·.1))) 1 (fun _ => Ext.one)
          (pointIn s (.turned ⟨level, "tau0", none⟩ F.low)) Ext.zero)
        ((rangeRun (2 ^ F.logBasis) (productsIn s level ((productStages F.logBasis).map (·.1)))
            (roundsIn s ⟨level, "range.leaf", none⟩) (firstIn s ⟨level, "range.image", none⟩)
            (pointIn s (.turned ⟨level, "tau0", none⟩ F.low))).map fun o => (o.1.map φ, φ o.2)) := by
  have st := range_state pub proof messages _ (channelStart_fresh pub) (channelStart_empty pub) _
    level F shape (aligned_take messages _ c _ (decode_aligned decoded)) before
  unfold stateBefore at state
  rw [state] at st
  obtain ⟨count, lengths⟩ := st.leaf plain
  rw [rangeCheckOf_plain level F plain]
  have runs := step_range_runs hom same pub proof messages c key level (2 ^ F.logBasis)
    ((productStages F.logBasis).map (·.1)) (.turned ⟨level, "tau0", none⟩ F.low) s F.vars support w
    ((productsIn_arities s level _).trans arities) ((pointIn_turned_length s _ _).trans st.vars)
    st.stages count (fun x member => (lengths x member).trans leaf)
  dsimp only at runs
  exact runs

/-- The range proof of a fold on the coefficient route: when `verifyProof` accepts, `rangeCheck`
runs on the path of the recorded values of the state before `range.leaf.end` to acceptance. -/
theorem verifyProof_range_game (scripted : script pub.layout pub.plan = .ok ops)
    (ends : endsFit ops.zipIdx.toList = true) {full : Array Fq}
    (terminalSetup : pub.setupFor pub.layout.terminal.setupCount = some full)
    (accepted : verifyProof pub proof = .verdict .accept) {messages : Array Message} {rest : ℕ}
    (decoded : decode ops proof = .ok (messages, rest)) {c : ℕ} {key : Key} {level : ℕ}
    {F : Fold} (at_ : ops[c]? = some (.check key (rangeCheckOf level F)))
    (shape : rangeShape (ops.zipIdx.toList.take c) level F = true) {s : Replayer}
    (state : stateBefore pub proof messages ops c = s) (plain : F.euclidean = false)
    (arities : (productStages F.logBasis).map (·.1) = rangeArities (2 ^ F.logBasis))
    (leaf : leafDegree F.logBasis = rangeLeaf (2 ^ F.logBasis))
    (support : ℕ → Prop) [DecidablePred support] (w : Table) :
    ∃ o, rangeRun (2 ^ F.logBasis) (productsIn s level ((productStages F.logBasis).map (·.1)))
        (roundsIn s ⟨level, "range.leaf", none⟩) (firstIn s ⟨level, "range.image", none⟩)
        (pointIn s (.turned ⟨level, "tau0", none⟩ F.low)) = some o ∧
      (rangeCheck (2 ^ F.logBasis) F.vars support w
          ((pointIn s (.turned ⟨level, "tau0", none⟩ F.low)).map φ)).Runs
        (treePath φ (fun _ _ _ _ => leafPath φ (rangeLeaf (2 ^ F.logBasis))
            (roundsIn s ⟨level, "range.leaf", none⟩) (firstIn s ⟨level, "range.image", none⟩))
          (productsIn s level ((productStages F.logBasis).map (·.1))) 1 (fun _ => Ext.one)
          (pointIn s (.turned ⟨level, "tau0", none⟩ F.low)) Ext.zero)
        (some (o.1.map φ, φ o.2)) := by
  have none_ := (replay_of_decoded pub proof scripted ends terminalSetup accepted decoded).2
  obtain ⟨before, stepped⟩ := replay_step_ok pub proof messages none_ at_
  obtain ⟨run, game⟩ := range_game_at pub proof hom same decoded before shape state plain arities
    leaf key support w
  rw [state, run] at stepped
  cases found : rangeRun (2 ^ F.logBasis) (productsIn s level ((productStages F.logBasis).map (·.1)))
      (roundsIn s ⟨level, "range.leaf", none⟩) (firstIn s ⟨level, "range.image", none⟩)
      (pointIn s (.turned ⟨level, "tau0", none⟩ F.low)) with
  | none =>
      rw [found] at stepped
      cases stepped
  | some o =>
      rw [found] at game
      exact ⟨o, rfl, game⟩

/-- A reject at `range.leaf.end` on the coefficient route: `rangeRun` is `none` on the state before
it and `rangeCheck` runs on the path of the recorded values to `none`. -/
theorem verifyProof_reject_range_game (scripted : script pub.layout pub.plan = .ok ops) {i : ℕ}
    {site : String} (rejected : verifyProof pub proof = .verdict (.reject i site))
    {messages : Array Message} {rest : ℕ} (decoded : decode ops proof = .ok (messages, rest))
    {key : Key} {level : ℕ} {F : Fold} (at_ : ops[i]? = some (.check key (rangeCheckOf level F)))
    (shape : rangeShape (ops.zipIdx.toList.take i) level F = true) {s : Replayer}
    (state : stateBefore pub proof messages ops i = s) (plain : F.euclidean = false)
    (arities : (productStages F.logBasis).map (·.1) = rangeArities (2 ^ F.logBasis))
    (leaf : leafDegree F.logBasis = rangeLeaf (2 ^ F.logBasis))
    (support : ℕ → Prop) [DecidablePred support] (w : Table) :
    rangeRun (2 ^ F.logBasis) (productsIn s level ((productStages F.logBasis).map (·.1)))
        (roundsIn s ⟨level, "range.leaf", none⟩) (firstIn s ⟨level, "range.image", none⟩)
        (pointIn s (.turned ⟨level, "tau0", none⟩ F.low)) = none ∧
      (rangeCheck (2 ^ F.logBasis) F.vars support w
          ((pointIn s (.turned ⟨level, "tau0", none⟩ F.low)).map φ)).Runs
        (treePath φ (fun _ _ _ _ => leafPath φ (rangeLeaf (2 ^ F.logBasis))
            (roundsIn s ⟨level, "range.leaf", none⟩) (firstIn s ⟨level, "range.image", none⟩))
          (productsIn s level ((productStages F.logBasis).map (·.1))) 1 (fun _ => Ext.one)
          (pointIn s (.turned ⟨level, "tau0", none⟩ F.low)) Ext.zero) none := by
  obtain ⟨m, rest', decoded', stop, before, _⟩ := verifyProof_reject pub proof scripted rejected
  rw [decoded] at decoded'
  cases decoded'
  obtain ⟨op, at', -, stepped⟩ := replay_stop_at pub proof messages ops stop
  simp only [Stop.op] at at' stepped
  rw [at_] at at'
  cases at'
  obtain ⟨run, game⟩ := range_game_at pub proof hom same decoded before shape state plain arities
    leaf key support w
  rw [state, run] at stepped
  cases found : rangeRun (2 ^ F.logBasis) (productsIn s level ((productStages F.logBasis).map (·.1)))
      (roundsIn s ⟨level, "range.leaf", none⟩) (firstIn s ⟨level, "range.image", none⟩)
      (pointIn s (.turned ⟨level, "tau0", none⟩ F.low)) with
  | some o =>
      rw [found] at stepped
      cases stepped
  | none =>
      rw [found] at game
      exact ⟨rfl, game⟩

/-- The norm stage of the fold `F` at `level` on the recorded values of `s`: `treeRun` with
`normRun`, the stage function of `norm.end`. -/
def normRunIn (s : Replayer) (level : ℕ) (F : Fold) : Option (List Ext × Ext × Ext) :=
  treeRun (normRun (leafDegree F.logBasis) F.limit
      ((s.integers.get? ⟨level, "norm.response_l2_sq", none⟩).getD 0)
      (firstIn s ⟨level, "norm.merge", none⟩) (roundsIn s ⟨level, "norm.sumcheck", none⟩)
      (firstIn s ⟨level, "norm.virtual_evaluations", none⟩) (firstIn s ⟨level, "range.image", none⟩))
    (productsIn s level ((productStages F.logBasis).map (·.1))) 1 (fun _ => Ext.one)
    (pointIn s (.turned ⟨level, "tau0", none⟩ F.low)) Ext.zero

/-- The path of the norm stage of the fold `F` at `level` for the recorded values of `s`. -/
noncomputable def normPathIn (s : Replayer) (level : ℕ) (F : Fold) : List Move :=
  treePath φ (fun _ _ _ claim => normPath φ (leafDegree F.logBasis) F.limit
      ((s.integers.get? ⟨level, "norm.response_l2_sq", none⟩).getD 0)
      (firstIn s ⟨level, "norm.merge", none⟩) (roundsIn s ⟨level, "norm.sumcheck", none⟩)
      (firstIn s ⟨level, "norm.virtual_evaluations", none⟩) (firstIn s ⟨level, "range.image", none⟩)
      claim)
    (productsIn s level ((productStages F.logBasis).map (·.1))) 1 (fun _ => Ext.one)
    (pointIn s (.turned ⟨level, "tau0", none⟩ F.low)) Ext.zero

/-- The game of the norm stage of the fold `F` at `level` from the point of `s`: `rangeTree` with
`normLeafStage`. -/
noncomputable def normGameIn (s : Replayer) (level : ℕ) (F : Fold) (kernel : ℕ → ℕ → Zq)
    (support : ℕ → Prop) [DecidablePred support] (w : Table) :=
  rangeTree (leafDegree F.logBasis) F.vars w (fun nodes weights point claim =>
      normLeafStage (leafDegree F.logBasis) F.vars F.limit kernel support w nodes weights point
        claim)
    ((productsIn s level ((productStages F.logBasis).map (·.1))).map (·.arity)) 1
    (fun _ => φ Ext.one) ((pointIn s (.turned ⟨level, "tau0", none⟩ F.low)).map φ) (φ Ext.zero)

/-- The step of `norm.end` on the Euclidean route and its game, on the state before it, after the
operations before it without a stop. -/
theorem norm_game_at (nat : ∀ n : ℕ, φ (Ext.ofBase (Fq.ofNat n)) = (n : K))
    {messages : Array Message} {rest : ℕ}
    (decoded : decode ops proof = .ok (messages, rest)) {c : ℕ}
    (before : (runFrom pub proof messages (ops.zipIdx.toList.take c) (channelStart pub)).1 = .ok ())
    {level : ℕ} {F : Fold} (shape : rangeShape (ops.zipIdx.toList.take c) level F = true)
    {s : Replayer} (state : stateBefore pub proof messages ops c = s)
    (euclidean : F.euclidean = true) (positive : 1 ≤ leafDegree F.logBasis)
    (small : F.limit < fieldModulus) (key : Key) (kernel : ℕ → ℕ → Zq)
    (support : ℕ → Prop) [DecidablePred support] (w : Table) :
    ((step pub proof messages c (.check key (rangeCheckOf level F))).run).run s =
        (if (normRunIn s level F).isSome then
          (.ok (), checked (staged s key (.norm (normRunIn s level F))) key)
        else (.error (.reject c key.site), staged s key (.norm (normRunIn s level F)))) ∧
      (normGameIn (φ := φ) s level F kernel support w).Runs (normPathIn (φ := φ) s level F)
        ((normRunIn s level F).map fun o => (o.1.map φ, φ o.2.1, φ o.2.2)) := by
  have st := range_state pub proof messages _ (channelStart_fresh pub) (channelStart_empty pub) _
    level F shape (aligned_take messages _ c _ (decode_aligned decoded)) before
  unfold stateBefore at state
  rw [state] at st
  obtain ⟨count, lengths⟩ := st.norm euclidean
  rw [rangeCheckOf_euclidean level F euclidean]
  have runs := step_norm_runs hom same nat pub proof messages c key level (leafDegree F.logBasis)
    F.limit ((productStages F.logBasis).map (·.1)) (.turned ⟨level, "tau0", none⟩ F.low) s F.vars
    kernel support w positive small ((pointIn_turned_length s _ _).trans st.vars) st.stages count
    lengths
  dsimp only at runs
  exact runs

/-- The norm stage of a fold on the Euclidean route: when `verifyProof` accepts, `rangeTree` with
`normLeafStage` runs on the path of the recorded values of the state before `norm.end` to
acceptance. -/
theorem verifyProof_norm_game (nat : ∀ n : ℕ, φ (Ext.ofBase (Fq.ofNat n)) = (n : K))
    (scripted : script pub.layout pub.plan = .ok ops)
    (ends : endsFit ops.zipIdx.toList = true) {full : Array Fq}
    (terminalSetup : pub.setupFor pub.layout.terminal.setupCount = some full)
    (accepted : verifyProof pub proof = .verdict .accept) {messages : Array Message} {rest : ℕ}
    (decoded : decode ops proof = .ok (messages, rest)) {c : ℕ} {key : Key} {level : ℕ}
    {F : Fold} (at_ : ops[c]? = some (.check key (rangeCheckOf level F)))
    (shape : rangeShape (ops.zipIdx.toList.take c) level F = true) {s : Replayer}
    (state : stateBefore pub proof messages ops c = s) (euclidean : F.euclidean = true)
    (positive : 1 ≤ leafDegree F.logBasis) (small : F.limit < fieldModulus)
    (kernel : ℕ → ℕ → Zq) (support : ℕ → Prop) [DecidablePred support] (w : Table) :
    ∃ o, normRunIn s level F = some o ∧
      (normGameIn (φ := φ) s level F kernel support w).Runs (normPathIn (φ := φ) s level F)
        (some (o.1.map φ, φ o.2.1, φ o.2.2)) := by
  have none_ := (replay_of_decoded pub proof scripted ends terminalSetup accepted decoded).2
  obtain ⟨before, stepped⟩ := replay_step_ok pub proof messages none_ at_
  obtain ⟨run, game⟩ := norm_game_at pub proof hom same nat decoded before shape state euclidean
    positive small key kernel support w
  rw [state, run] at stepped
  cases found : normRunIn s level F with
  | none =>
      rw [found] at stepped
      cases stepped
  | some o =>
      rw [found] at game
      exact ⟨o, rfl, game⟩

/-- A reject at `norm.end` on the Euclidean route: the norm stage is `none` on the state before it
and `rangeTree` with `normLeafStage` runs on the path of the recorded values to `none`. -/
theorem verifyProof_reject_norm_game (nat : ∀ n : ℕ, φ (Ext.ofBase (Fq.ofNat n)) = (n : K))
    (scripted : script pub.layout pub.plan = .ok ops) {i : ℕ}
    {site : String} (rejected : verifyProof pub proof = .verdict (.reject i site))
    {messages : Array Message} {rest : ℕ} (decoded : decode ops proof = .ok (messages, rest))
    {key : Key} {level : ℕ} {F : Fold} (at_ : ops[i]? = some (.check key (rangeCheckOf level F)))
    (shape : rangeShape (ops.zipIdx.toList.take i) level F = true) {s : Replayer}
    (state : stateBefore pub proof messages ops i = s) (euclidean : F.euclidean = true)
    (positive : 1 ≤ leafDegree F.logBasis) (small : F.limit < fieldModulus)
    (kernel : ℕ → ℕ → Zq) (support : ℕ → Prop) [DecidablePred support] (w : Table) :
    normRunIn s level F = none ∧
      (normGameIn (φ := φ) s level F kernel support w).Runs (normPathIn (φ := φ) s level F)
        none := by
  obtain ⟨m, rest', decoded', stop, before, _⟩ := verifyProof_reject pub proof scripted rejected
  rw [decoded] at decoded'
  cases decoded'
  obtain ⟨op, at', -, stepped⟩ := replay_stop_at pub proof messages ops stop
  simp only [Stop.op] at at' stepped
  rw [at_] at at'
  cases at'
  obtain ⟨run, game⟩ := norm_game_at pub proof hom same nat decoded before shape state euclidean
    positive small key kernel support w
  rw [state, run] at stepped
  cases found : normRunIn s level F with
  | some o =>
      rw [found] at stepped
      cases stepped
  | none =>
      rw [found] at game
      exact ⟨rfl, game⟩

end Games

end NightstreamFPrime.Spec.Compression.Akita.Replay
