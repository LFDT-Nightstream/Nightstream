import CompressionLean.Circuit.Final

/-!
Owns `Appends`: a builder program keeps the instructions that come before
it and writes its own after them. Every program of the layer-1 verifier
appends (`verifyWith_appends`), so the executor's run of the final program
starts with the run of the terminal wrapper.

- `Appends` for `Build` programs and `TAppends` for `Transcript` programs,
  with one lemma per way to combine programs: binds, maps, loops over
  lists, arrays and ranges, `mapM`, `forM`, branches, and the duplex's
  `get`, `set` and `modify`, which leave the builder alone.
- `appends`: the tactic that applies those lemmas, case-splits branches and
  matches, and unfolds every other definition at the head of a program.
-/

namespace CompressionLean.Circuit

/-- `b` keeps the instructions of any builder and writes after them. -/
structure Appends {α : Type} (b : Build α) : Prop where
  intro ::
  keeps : ∀ p : Builder, ∃ new : Array Instr, (b.run p).2.instrs = p.instrs ++ new

/-- `t` appends from every duplex. -/
structure TAppends {α : Type} (t : Transcript α) : Prop where
  intro ::
  run : ∀ d, Appends (t.run d)

namespace Appends

variable {α β γ : Type}

theorem of_state {b : Build α} (h : ∀ p, (b.run p).2 = p) : Appends b :=
  ⟨fun p => ⟨#[], by rw [h, Array.append_empty]⟩⟩

theorem pure (a : α) : Appends (Pure.pure a : Build α) := of_state fun _ => rfl

theorem bind {b : Build α} {f : α → Build β} (hb : Appends b) (hf : ∀ a, Appends (f a)) :
    Appends (b >>= f) := by
  refine ⟨fun p => ?_⟩
  obtain ⟨new, hnew⟩ := hb.keeps p
  obtain ⟨more, hmore⟩ := (hf (b.run p).1).keeps (b.run p).2
  exact ⟨new ++ more, by
    show ((f (b.run p).1).run (b.run p).2).2.instrs = _
    rw [hmore, hnew, Array.append_assoc]⟩

theorem map {b : Build α} (g : α → β) (hb : Appends b) : Appends (g <$> b) := ⟨fun p => hb.keeps p⟩

theorem emit (i : Instr) : Appends (Build.emit i) :=
  ⟨fun p => ⟨#[i], by show p.instrs.push i = _; simp⟩⟩

theorem forIn_list {xs : List β} {init : γ} {body : β → γ → Build (ForInStep γ)}
    (h : ∀ x acc, Appends (body x acc)) : Appends (forIn xs init body) := by
  induction xs generalizing init with
  | nil => exact pure _
  | cons x rest ih =>
      rw [List.forIn_cons]
      refine bind (h x init) fun r => ?_
      cases r with
      | done b => exact pure b
      | yield b => exact ih

theorem forIn_array {xs : Array β} {init : γ} {body : β → γ → Build (ForInStep γ)}
    (h : ∀ x acc, Appends (body x acc)) : Appends (forIn xs init body) := by
  rw [← Array.forIn_toList]
  exact forIn_list h

theorem forIn_range {r : Std.Legacy.Range} {init : γ} {body : ℕ → γ → Build (ForInStep γ)}
    (h : ∀ x acc, Appends (body x acc)) : Appends (forIn r init body) := by
  rw [Std.Legacy.Range.forIn_eq_forIn_range']
  exact forIn_list h

theorem mapM_list {xs : List β} {f : β → Build α} (h : ∀ x, Appends (f x)) : Appends (xs.mapM f) := by
  induction xs with
  | nil => exact pure _
  | cons x rest ih =>
      rw [List.mapM_cons]
      exact bind (h x) fun _ => bind ih fun _ => pure _

theorem mapM_array {xs : Array β} {f : β → Build α} (h : ∀ x, Appends (f x)) : Appends (xs.mapM f) := by
  rw [Array.mapM_eq_mapM_toList]
  exact map _ (mapM_list h)

theorem foldlM_list {xs : List β} {init : γ} {f : γ → β → Build γ} (h : ∀ acc x, Appends (f acc x)) :
    Appends (xs.foldlM f init) := by
  induction xs generalizing init with
  | nil => exact pure _
  | cons x rest ih =>
      rw [List.foldlM_cons]
      exact bind (h init x) fun _ => ih

theorem forM_array {xs : Array β} {f : β → Build PUnit} (h : ∀ x, Appends (f x)) : Appends (xs.forM f) := by
  show Appends (xs.foldlM (fun _ => f) ⟨⟩)
  rw [← Array.foldlM_toList]
  exact foldlM_list fun _ => h

theorem run {t : Transcript α} (h : TAppends t) (d : Duplex) : Appends (t.run d) := h.run d

end Appends

namespace TAppends

variable {α β γ : Type}

theorem pure (a : α) : TAppends (Pure.pure a : Transcript α) := ⟨fun _ => Appends.pure _⟩

theorem bind {t : Transcript α} {g : α → Transcript β} (ht : TAppends t) (hg : ∀ a, TAppends (g a)) :
    TAppends (t >>= g) := ⟨fun d => by
  rw [StateT.run_bind]
  exact Appends.bind (ht.run d) fun r => (hg r.1).run r.2⟩

theorem map {t : Transcript α} (g : α → β) (ht : TAppends t) : TAppends (g <$> t) := ⟨fun d => by
  rw [StateT.run_map]
  exact Appends.map _ (ht.run d)⟩

theorem lift {b : Build α} (h : Appends b) : TAppends (monadLift b : Transcript α) := ⟨fun d => by
  rw [StateT.run_monadLift]
  exact Appends.bind h fun _ => Appends.pure _⟩

theorem get : TAppends (MonadState.get : Transcript Duplex) := ⟨fun _ => Appends.pure _⟩

theorem set (d : Duplex) : TAppends (MonadStateOf.set d : Transcript PUnit) := ⟨fun _ => Appends.pure _⟩

theorem modifyGet (f : Duplex → α × Duplex) : TAppends (MonadState.modifyGet f : Transcript α) :=
  ⟨fun _ => Appends.pure _⟩

theorem forIn_list {xs : List β} {init : γ} {body : β → γ → Transcript (ForInStep γ)}
    (h : ∀ x acc, TAppends (body x acc)) : TAppends (forIn xs init body) := by
  induction xs generalizing init with
  | nil => exact pure _
  | cons x rest ih =>
      rw [List.forIn_cons]
      refine bind (h x init) fun r => ?_
      cases r with
      | done b => exact pure b
      | yield b => exact ih

theorem forIn_array {xs : Array β} {init : γ} {body : β → γ → Transcript (ForInStep γ)}
    (h : ∀ x acc, TAppends (body x acc)) : TAppends (forIn xs init body) := by
  rw [← Array.forIn_toList]
  exact forIn_list h

theorem forIn_range {r : Std.Legacy.Range} {init : γ} {body : ℕ → γ → Transcript (ForInStep γ)}
    (h : ∀ x acc, TAppends (body x acc)) : TAppends (forIn r init body) := by
  rw [Std.Legacy.Range.forIn_eq_forIn_range']
  exact forIn_list h

theorem mapM_list {xs : List β} {f : β → Transcript α} (h : ∀ x, TAppends (f x)) : TAppends (xs.mapM f) := by
  induction xs with
  | nil => exact pure _
  | cons x rest ih =>
      rw [List.mapM_cons]
      exact bind (h x) fun _ => bind ih fun _ => pure _

theorem mapM_array {xs : Array β} {f : β → Transcript α} (h : ∀ x, TAppends (f x)) :
    TAppends (xs.mapM f) := by
  rw [Array.mapM_eq_mapM_toList]
  exact map _ (mapM_list h)

theorem foldlM_list {xs : List β} {init : γ} {f : γ → β → Transcript γ} (h : ∀ acc x, TAppends (f acc x)) :
    TAppends (xs.foldlM f init) := by
  induction xs generalizing init with
  | nil => exact pure _
  | cons x rest ih =>
      rw [List.foldlM_cons]
      exact bind (h init x) fun _ => ih

theorem forM_array {xs : Array β} {f : β → Transcript PUnit} (h : ∀ x, TAppends (f x)) :
    TAppends (xs.forM f) := by
  show TAppends (xs.foldlM (fun _ => f) ⟨⟩)
  rw [← Array.foldlM_toList]
  exact foldlM_list fun _ => h

end TAppends

open Lean Meta Elab Tactic in
/-- Apply the lemma `c.appends` when the program is an application of the
definition `c` and that lemma exists. -/
elab "appends_known" : tactic => do
  let target ← instantiateMVars (← getMainTarget)
  let .app _ prog := target | throwError "appends_known: no program"
  let .const c _ := prog.getAppFn | throwError "appends_known: no definition"
  let known := c ++ `appends
  unless (← getEnv).contains known do throwError "appends_known: no lemma {known}"
  evalTactic (← `(tactic| with_reducible apply $(mkIdent known)))

open Lean Meta Elab Tactic in
/-- Move a `let` at the head of the program into the context. -/
elab "appends_let" : tactic => liftMetaTactic fun g => do
  let target ← instantiateMVars (← g.getType)
  let .app f (.letE name type value body nondep) := target | throwError "appends_let: no let"
  let g ← g.replaceTargetDefEq (.letE name type value (.app f body) nondep)
  return [(← g.intro1P).2]

open Lean Meta Elab Tactic in
/-- Split a `match` at the head of the program. A discriminant that is a
local definition is first replaced by its value, which `split` needs. -/
elab "appends_match" : tactic => withMainContext do
  let g ← getMainGoal
  let target ← instantiateMVars (← g.getType)
  let .app f prog := target | throwError "appends_match: no program"
  let some app ← matchMatcherApp? prog | throwError "appends_match: no match"
  let discrs ← app.discrs.mapM fun d => do
    let .fvar id := d | return d
    return (← id.getValue?).getD d
  replaceMainGoal [← g.replaceTargetDefEq (.app f { app with discrs }.toExpr)]
  evalTactic (← `(tactic| split))

open Lean Meta Elab Tactic in
/-- Unfold the definition, or the local definition, at the head of the
program in an `Appends` or `TAppends` goal. -/
elab "appends_unfold" : tactic => liftMetaTactic fun g => do
  let target ← instantiateMVars (← g.getType)
  let .app f prog := target | throwError "appends_unfold: no program"
  let prog' ← match prog.getAppFn with
    | .fvar id => do
        let some value ← id.getValue? | throwError "appends_unfold: no local definition"
        pure (value.beta prog.getAppArgs)
    | _ => do
        let some prog' ← unfoldDefinition? prog | throwError "appends_unfold: no definition to unfold"
        pure prog'
  return [← g.replaceTargetDefEq (.app f prog')]

/-- Prove `Appends` or `TAppends` of a builder program. -/
macro "appends" : tactic => `(tactic| repeat' first
  | intro _
  | with_reducible exact Appends.pure _ | with_reducible exact TAppends.pure _
  | with_reducible apply Appends.bind | with_reducible apply TAppends.bind
  | with_reducible apply Appends.map | with_reducible apply TAppends.map
  | with_reducible exact Appends.emit _
  | with_reducible apply Appends.run | with_reducible apply TAppends.lift
  | with_reducible exact TAppends.get | with_reducible exact TAppends.set _
  | with_reducible exact TAppends.modifyGet _
  | with_reducible apply Appends.forIn_list | with_reducible apply Appends.forIn_array
  | with_reducible apply Appends.forIn_range | with_reducible apply TAppends.forIn_list
  | with_reducible apply TAppends.forIn_array | with_reducible apply TAppends.forIn_range
  | with_reducible apply Appends.mapM_list | with_reducible apply Appends.mapM_array
  | with_reducible apply TAppends.mapM_list | with_reducible apply TAppends.mapM_array
  | with_reducible apply Appends.forM_array | with_reducible apply TAppends.forM_array
  | appends_let
  | appends_known
  | appends_match
  | split
  | appends_unfold)

/-! ## The verifier

The lemmas of the duplex steps and the stages keep each proof to one level
of the call tree; `appends_known` applies them. -/

open CompressionLean.Replay (Relation)

theorem Duplex.duplexing.appends : TAppends Duplex.duplexing := by appends

theorem Duplex.observe.appends (value : Var) : TAppends (Duplex.observe value) := by appends

theorem Duplex.sample.appends : TAppends Duplex.sample := by appends

theorem whirVerify.appends (shape : Replay.WhirShape) (key : WhirKey) (root : DigestVar) (view : OpeningView)
    (points : Array (Array ExtVar)) : TAppends (whirVerify shape key root view points) := by appends

theorem earlyStage.appends (rel : Relation) (statement : ClaimView) (proof : ProofView) :
    TAppends (earlyStage rel statement proof) := by appends

theorem linearStage.appends (rel : Relation) (statement : ClaimView) (quotient : Array ExtVar)
    (linear : Array (Array ExtVar)) (finals : Array ExtVar) :
    TAppends (linearStage rel statement quotient linear finals) := by appends

theorem honestStage.appends (rel : Relation) (input : HonestInput) : TAppends (honestStage rel input) := by
  appends

theorem lateStage.appends (rel : Relation) (tau : Array ExtVar) (evalA : Array (ExtVar × ExtVar))
    (late : GkrView) (lateValues : Array (ExtVar × ExtVar)) :
    TAppends (lateStage rel tau evalA late lateValues) := by appends

theorem openingsStage.appends (rel : Relation) (p1Key p0Key : WhirKey) (o : OpeningsInput) :
    TAppends (openingsStage rel p1Key p0Key o) := by appends

theorem verify.appends (rel : Relation) (p1Key p0Key : WhirKey) (seed : DigestVar) (statement : ClaimView)
    (proof : ProofView) : Appends (verify rel p1Key p0Key seed statement proof) := by appends

/-- **The layer-1 verifier appends.** -/
theorem verifyWith.appends (key : Layer1Input) (state : Fin 16 → Var) (claim : ClaimView) :
    Appends (verifyWith key state claim) := by appends

end CompressionLean.Circuit
