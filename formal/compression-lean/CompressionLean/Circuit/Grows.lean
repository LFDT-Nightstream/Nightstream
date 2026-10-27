import CompressionLean.Circuit.SimRun
import CompressionLean.Circuit.Verifier

/-!
Owns `Grows`: a builder program keeps the instructions before it, writes
its own after them, and counts one value per output. It is the shape part
of `Accepts` (`Accepts.of_grows`): a program that grows accepts with no
claim about its result. The verifier's programs whose values the replay
does not read (the WHIR checks, the Merkle paths of the opening queries)
enter the stage proofs this way.

`Grows` for `Build` programs and `TGrows` for `Transcript` programs, with
one lemma per way to combine programs, and the tactic `grows` that applies
them, as `Appends` of the terminal-wiring branch does.
-/

namespace CompressionLean.Circuit

/-- `b` keeps the instructions of any builder, writes `new` after them, and
counts their outputs. -/
structure Grows {α : Type} (b : Build α) : Prop where
  intro ::
  keeps : ∀ p : Builder, ∃ new : List Instr, (b.run p).2.instrs = p.instrs ++ new.toArray ∧
    (b.run p).2.values = p.values + (new.map Instr.outputs).sum

/-- `t` grows from every duplex. -/
structure TGrows {α : Type} (t : Transcript α) : Prop where
  intro ::
  run : ∀ d, Grows (t.run d)

namespace Grows

variable {α β γ : Type}

theorem of_state {b : Build α} (h : ∀ p, (b.run p).2 = p) : Grows b :=
  ⟨fun p => ⟨[], by rw [h]; simp, by rw [h]; simp⟩⟩

theorem pure (a : α) : Grows (Pure.pure a : Build α) := of_state fun _ => rfl

theorem bind {b : Build α} {f : α → Build β} (hb : Grows b) (hf : ∀ a, Grows (f a)) :
    Grows (b >>= f) := by
  refine ⟨fun p => ?_⟩
  obtain ⟨new, hnew, cnew⟩ := hb.keeps p
  obtain ⟨more, hmore, cmore⟩ := (hf (b.run p).1).keeps (b.run p).2
  refine ⟨new ++ more, ?_, ?_⟩
  · show ((f (b.run p).1).run (b.run p).2).2.instrs = _
    rw [hmore, hnew]
    simp
  · show ((f (b.run p).1).run (b.run p).2).2.values = _
    rw [cmore, cnew]
    simp [Nat.add_assoc]

theorem map {b : Build α} (g : α → β) (hb : Grows b) : Grows (g <$> b) := ⟨fun p => hb.keeps p⟩

theorem emit (i : Instr) : Grows (Build.emit i) :=
  ⟨fun p => ⟨[i], by show p.instrs.push i = _; simp, by show p.values + i.outputs = _; simp⟩⟩

theorem forIn_list {xs : List β} {init : γ} {body : β → γ → Build (ForInStep γ)}
    (h : ∀ x acc, Grows (body x acc)) : Grows (forIn xs init body) := by
  induction xs generalizing init with
  | nil => exact pure _
  | cons x rest ih =>
      rw [List.forIn_cons]
      refine bind (h x init) fun r => ?_
      cases r with
      | done b => exact pure b
      | yield b => exact ih

theorem forIn_array {xs : Array β} {init : γ} {body : β → γ → Build (ForInStep γ)}
    (h : ∀ x acc, Grows (body x acc)) : Grows (forIn xs init body) := by
  rw [← Array.forIn_toList]
  exact forIn_list h

theorem forIn_range {r : Std.Legacy.Range} {init : γ} {body : ℕ → γ → Build (ForInStep γ)}
    (h : ∀ x acc, Grows (body x acc)) : Grows (forIn r init body) := by
  rw [Std.Legacy.Range.forIn_eq_forIn_range']
  exact forIn_list h

theorem mapM_list {xs : List β} {f : β → Build α} (h : ∀ x, Grows (f x)) : Grows (xs.mapM f) := by
  induction xs with
  | nil => exact pure _
  | cons x rest ih =>
      rw [List.mapM_cons]
      exact bind (h x) fun _ => bind ih fun _ => pure _

theorem mapM_array {xs : Array β} {f : β → Build α} (h : ∀ x, Grows (f x)) : Grows (xs.mapM f) := by
  rw [Array.mapM_eq_mapM_toList]
  exact map _ (mapM_list h)

theorem foldlM_list {xs : List β} {init : γ} {f : γ → β → Build γ} (h : ∀ acc x, Grows (f acc x)) :
    Grows (xs.foldlM f init) := by
  induction xs generalizing init with
  | nil => exact pure _
  | cons x rest ih =>
      rw [List.foldlM_cons]
      exact bind (h init x) fun _ => ih

theorem forM_array {xs : Array β} {f : β → Build PUnit} (h : ∀ x, Grows (f x)) : Grows (xs.forM f) := by
  show Grows (xs.foldlM (fun _ => f) ⟨⟩)
  rw [← Array.foldlM_toList]
  exact foldlM_list fun _ => h

theorem run {t : Transcript α} (h : TGrows t) (d : Duplex) : Grows (t.run d) := h.run d

end Grows

namespace TGrows

variable {α β γ : Type}

theorem pure (a : α) : TGrows (Pure.pure a : Transcript α) := ⟨fun _ => Grows.pure _⟩

theorem bind {t : Transcript α} {g : α → Transcript β} (ht : TGrows t) (hg : ∀ a, TGrows (g a)) :
    TGrows (t >>= g) := ⟨fun d => by
  rw [StateT.run_bind]
  exact Grows.bind (ht.run d) fun r => (hg r.1).run r.2⟩

theorem map {t : Transcript α} (g : α → β) (ht : TGrows t) : TGrows (g <$> t) := ⟨fun d => by
  rw [StateT.run_map]
  exact Grows.map _ (ht.run d)⟩

theorem lift {b : Build α} (h : Grows b) : TGrows (monadLift b : Transcript α) := ⟨fun d => by
  rw [StateT.run_monadLift]
  exact Grows.bind h fun _ => Grows.pure _⟩

theorem get : TGrows (MonadState.get : Transcript Duplex) := ⟨fun _ => Grows.pure _⟩

theorem set (d : Duplex) : TGrows (MonadStateOf.set d : Transcript PUnit) := ⟨fun _ => Grows.pure _⟩

theorem modifyGet (f : Duplex → α × Duplex) : TGrows (MonadState.modifyGet f : Transcript α) :=
  ⟨fun _ => Grows.pure _⟩

theorem forIn_list {xs : List β} {init : γ} {body : β → γ → Transcript (ForInStep γ)}
    (h : ∀ x acc, TGrows (body x acc)) : TGrows (forIn xs init body) := by
  induction xs generalizing init with
  | nil => exact pure _
  | cons x rest ih =>
      rw [List.forIn_cons]
      refine bind (h x init) fun r => ?_
      cases r with
      | done b => exact pure b
      | yield b => exact ih

theorem forIn_array {xs : Array β} {init : γ} {body : β → γ → Transcript (ForInStep γ)}
    (h : ∀ x acc, TGrows (body x acc)) : TGrows (forIn xs init body) := by
  rw [← Array.forIn_toList]
  exact forIn_list h

theorem forIn_range {r : Std.Legacy.Range} {init : γ} {body : ℕ → γ → Transcript (ForInStep γ)}
    (h : ∀ x acc, TGrows (body x acc)) : TGrows (forIn r init body) := by
  rw [Std.Legacy.Range.forIn_eq_forIn_range']
  exact forIn_list h

theorem mapM_list {xs : List β} {f : β → Transcript α} (h : ∀ x, TGrows (f x)) : TGrows (xs.mapM f) := by
  induction xs with
  | nil => exact pure _
  | cons x rest ih =>
      rw [List.mapM_cons]
      exact bind (h x) fun _ => bind ih fun _ => pure _

theorem mapM_array {xs : Array β} {f : β → Transcript α} (h : ∀ x, TGrows (f x)) :
    TGrows (xs.mapM f) := by
  rw [Array.mapM_eq_mapM_toList]
  exact map _ (mapM_list h)

theorem foldlM_list {xs : List β} {init : γ} {f : γ → β → Transcript γ} (h : ∀ acc x, TGrows (f acc x)) :
    TGrows (xs.foldlM f init) := by
  induction xs generalizing init with
  | nil => exact pure _
  | cons x rest ih =>
      rw [List.foldlM_cons]
      exact bind (h init x) fun _ => ih

theorem forM_array {xs : Array β} {f : β → Transcript PUnit} (h : ∀ x, TGrows (f x)) :
    TGrows (xs.forM f) := by
  show TGrows (xs.foldlM (fun _ => f) ⟨⟩)
  rw [← Array.foldlM_toList]
  exact foldlM_list fun _ => h

end TGrows

/-- A program that grows accepts, with no claim about its result. -/
theorem Accepts.of_grows {F : Type} [CommRing F] {c : Ctx F} {α : Type} {n : ℕ} {P : Prop} {b : Build α}
    (h : Grows b) : Accepts c n P b fun _ _ => True := by
  intro p E hp hE
  obtain ⟨new, hi, hv⟩ := h.keeps p
  refine ⟨new, hi, ?_, fun _ _ _ => trivial⟩
  obtain ⟨more, hm, hl⟩ := c.steps_values new E
  rw [hm, List.length_append, hE, hl, hv, hp]

open Lean Meta Elab Tactic in
/-- Apply the lemma `c.grows` when the program is an application of the
definition `c` and that lemma exists. -/
elab "grows_known" : tactic => do
  let target ← instantiateMVars (← getMainTarget)
  let .app _ prog := target | throwError "grows_known: no program"
  let .const c _ := prog.getAppFn | throwError "grows_known: no definition"
  let known := c ++ `grows
  unless (← getEnv).contains known do throwError "grows_known: no lemma {known}"
  evalTactic (← `(tactic| with_reducible apply $(mkIdent known)))

open Lean Meta Elab Tactic in
/-- Move a `let` at the head of the program into the context. -/
elab "grows_let" : tactic => liftMetaTactic fun g => do
  let target ← instantiateMVars (← g.getType)
  let .app f (.letE name type value body nondep) := target | throwError "grows_let: no let"
  let g ← g.replaceTargetDefEq (.letE name type value (.app f body) nondep)
  return [(← g.intro1P).2]

open Lean Meta Elab Tactic in
/-- Split a `match` at the head of the program. A discriminant that is a
local definition is first replaced by its value, which `split` needs. -/
elab "grows_match" : tactic => withMainContext do
  let g ← getMainGoal
  let target ← instantiateMVars (← g.getType)
  let .app f prog := target | throwError "grows_match: no program"
  let some app ← matchMatcherApp? prog | throwError "grows_match: no match"
  let discrs ← app.discrs.mapM fun d => do
    let .fvar id := d | return d
    return (← id.getValue?).getD d
  replaceMainGoal [← g.replaceTargetDefEq (.app f { app with discrs }.toExpr)]
  evalTactic (← `(tactic| split))

open Lean Meta Elab Tactic in
/-- Unfold the definition, or the local definition, at the head of the
program in a `Grows` or `TGrows` goal. -/
elab "grows_unfold" : tactic => liftMetaTactic fun g => do
  let target ← instantiateMVars (← g.getType)
  let .app f prog := target | throwError "grows_unfold: no program"
  let prog' ← match prog.getAppFn with
    | .fvar id => do
        let some value ← id.getValue? | throwError "grows_unfold: no local definition"
        pure (value.beta prog.getAppArgs)
    | _ => do
        let some prog' ← unfoldDefinition? prog | throwError "grows_unfold: no definition to unfold"
        pure prog'
  return [← g.replaceTargetDefEq (.app f prog')]

/-- Prove `Grows` or `TGrows` of a builder program. -/
macro "grows" : tactic => `(tactic| repeat' first
  | intro _
  | with_reducible exact Grows.pure _ | with_reducible exact TGrows.pure _
  | with_reducible apply Grows.bind | with_reducible apply TGrows.bind
  | with_reducible apply Grows.map | with_reducible apply TGrows.map
  | with_reducible exact Grows.emit _
  | with_reducible apply Grows.run | with_reducible apply TGrows.lift
  | with_reducible exact TGrows.get | with_reducible exact TGrows.set _
  | with_reducible exact TGrows.modifyGet _
  | with_reducible apply Grows.forIn_list | with_reducible apply Grows.forIn_array
  | with_reducible apply Grows.forIn_range | with_reducible apply TGrows.forIn_list
  | with_reducible apply TGrows.forIn_array | with_reducible apply TGrows.forIn_range
  | with_reducible apply Grows.mapM_list | with_reducible apply Grows.mapM_array
  | with_reducible apply TGrows.mapM_list | with_reducible apply TGrows.mapM_array
  | with_reducible apply Grows.forM_array | with_reducible apply TGrows.forM_array
  | grows_let
  | grows_known
  | grows_match
  | split
  | grows_unfold)

/-! ## The verifier

The lemmas of the duplex steps and the stages keep each proof to one level
of the call tree; `grows_known` applies them. -/

open CompressionLean.Replay (Relation)

theorem Duplex.duplexing.grows : TGrows Duplex.duplexing := by grows

theorem Duplex.observe.grows (value : Var) : TGrows (Duplex.observe value) := by grows

theorem Duplex.sample.grows : TGrows Duplex.sample := by grows

theorem whirVerify.grows (shape : Replay.WhirShape) (key : WhirKey) (root : DigestVar) (view : OpeningView)
    (points : Array (Array ExtVar)) : TGrows (whirVerify shape key root view points) := by grows

theorem earlyStage.grows (rel : Relation) (statement : ClaimView) (proof : ProofView) :
    TGrows (earlyStage rel statement proof) := by grows

theorem linearStage.grows (rel : Relation) (statement : ClaimView) (quotient : Array ExtVar)
    (linear : Array (Array ExtVar)) (finals : Array ExtVar) :
    TGrows (linearStage rel statement quotient linear finals) := by grows

theorem honestStage.grows (rel : Relation) (input : HonestInput) : TGrows (honestStage rel input) := by
  grows

theorem lateStage.grows (rel : Relation) (tau : Array ExtVar) (evalA : Array (ExtVar × ExtVar))
    (late : GkrView) (lateValues : Array (ExtVar × ExtVar)) :
    TGrows (lateStage rel tau evalA late lateValues) := by grows

theorem openingsStage.grows (rel : Relation) (p1Key p0Key : WhirKey) (o : OpeningsInput) :
    TGrows (openingsStage rel p1Key p0Key o) := by grows

theorem verify.grows (rel : Relation) (p1Key p0Key : WhirKey) (seed : DigestVar) (statement : ClaimView)
    (proof : ProofView) : Grows (verify rel p1Key p0Key seed statement proof) := by grows

/-- **The layer-1 verifier grows.** -/
theorem verifyWith.grows (key : Layer1Input) (state : Fin 16 → Var) (claim : ClaimView) :
    Grows (verifyWith key state claim) := by grows

end CompressionLean.Circuit
