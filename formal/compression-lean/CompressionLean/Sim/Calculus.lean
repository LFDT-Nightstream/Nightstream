import CompressionLean.Sim.Run

namespace CompressionLean.Sim

open CompressionLean.Circuit
open CompressionLean.Lowering
open CompressionLean.Replay (Word)

class Appends {α : Type} (b : Build α) : Prop where
  out : ∀ p : Builder, ∃ new : List Instr, (b.run p).2.instrs = p.instrs ++ new.toArray ∧
    (b.run p).2.values = p.values + (new.map Instr.outputs).sum

namespace Appends

variable {α β γ δ : Type}

theorem le {b : Build α} (h : Appends b) (p : Builder) : p.values ≤ (b.run p).2.values := by
  obtain ⟨new, -, hv⟩ := h.out p
  omega

instance pure (a : α) : Appends (Pure.pure a : Build α) :=
  ⟨fun p => ⟨[], by show p.instrs = p.instrs ++ #[]; simp, by show p.values = p.values + 0; rfl⟩⟩

instance emit (i : Instr) : Appends (Build.emit i) :=
  ⟨fun p => ⟨[i], by show p.instrs.push i = _; simp, by show p.values + i.outputs = _; simp⟩⟩

theorem bind {b : Build α} {f : α → Build β} (hb : Appends b) (hf : ∀ a, Appends (f a)) : Appends (b >>= f) :=
  ⟨fun p => by
    obtain ⟨new₁, hi₁, hv₁⟩ := hb.out p
    obtain ⟨new₂, hi₂, hv₂⟩ := (hf (b.run p).1).out (b.run p).2
    refine ⟨new₁ ++ new₂, ?_, ?_⟩
    · show ((f (b.run p).1).run (b.run p).2).2.instrs = _
      rw [hi₂, hi₁]
      simp
    · show ((f (b.run p).1).run (b.run p).2).2.values = _
      rw [hv₂, hv₁]
      simp only [List.map_append, List.sum_append]
      omega⟩

theorem map {b : Build α} (f : α → β) (hb : Appends b) : Appends (f <$> b) := ⟨fun p => hb.out p⟩

theorem ite {c : Prop} [Decidable c] {x y : Build α} (hx : Appends x) (hy : Appends y) :
    Appends (if c then x else y) := by
  split <;> assumption

theorem dite {c : Prop} [Decidable c] {x : c → Build α} {y : ¬c → Build α} (hx : ∀ h, Appends (x h))
    (hy : ∀ h, Appends (y h)) : Appends (if h : c then x h else y h) := by
  split
  · exact hx _
  · exact hy _

theorem forIn_list (l : List δ) (init : γ) {body : δ → γ → Build (ForInStep γ)} (h : ∀ x acc, Appends (body x acc)) :
    Appends (forIn l init body) := by
  induction l generalizing init with
  | nil => exact pure _
  | cons x l ih =>
      rw [List.forIn_cons]
      refine bind (h x init) fun r => ?_
      cases r with
      | done b => exact pure _
      | yield b => exact ih b

theorem forIn_array (xs : Array δ) (init : γ) {body : δ → γ → Build (ForInStep γ)}
    (h : ∀ x acc, Appends (body x acc)) : Appends (forIn xs init body) := by
  rw [← Array.forIn_toList]
  exact forIn_list _ _ h

theorem forIn_range (r : Std.Legacy.Range) (init : γ) {body : ℕ → γ → Build (ForInStep γ)}
    (h : ∀ x acc, Appends (body x acc)) : Appends (forIn r init body) := by
  rw [Std.Legacy.Range.forIn_eq_forIn_range']
  exact forIn_list _ _ h

theorem mapM_list {f : δ → Build α} (h : ∀ x, Appends (f x)) : ∀ l : List δ, Appends (l.mapM f)
  | [] => pure _
  | x :: l => by
      rw [List.mapM_cons]
      exact bind (h x) fun _ => bind (mapM_list h l) fun _ => pure _

theorem mapM_array {f : δ → Build α} (h : ∀ x, Appends (f x)) (xs : Array δ) : Appends (xs.mapM f) := by
  rw [Array.mapM_eq_mapM_toList]
  exact map _ (mapM_list h _)

instance tpure (a : α) (d : Duplex) : Appends ((Pure.pure a : Transcript α).run d) := pure _

instance tget (d : Duplex) : Appends ((MonadState.get : Transcript Duplex).run d) := pure _

instance tset (s d : Duplex) : Appends ((MonadStateOf.set s : Transcript PUnit).run d) := pure _

instance tmodify (f : Duplex → Duplex) (d : Duplex) : Appends ((modify f : Transcript PUnit).run d) := pure _

instance tmodifyGet (f : Duplex → α × Duplex) (d : Duplex) : Appends ((MonadState.modifyGet f : Transcript α).run d) :=
  pure _

theorem tbind {t : Transcript α} {g : α → Transcript β} {d : Duplex} (ht : ∀ d, Appends (t.run d))
    (hg : ∀ a d, Appends ((g a).run d)) : Appends ((t >>= g).run d) := by
  rw [StateT.run_bind]
  exact bind (ht d) fun r => hg r.1 r.2

theorem tite {c : Prop} [Decidable c] {x y : Transcript α} {d : Duplex} (hx : ∀ d, Appends (x.run d))
    (hy : ∀ d, Appends (y.run d)) : Appends ((if c then x else y).run d) := by
  split
  · exact hx d
  · exact hy d

theorem tlift {b : Build α} {d : Duplex} (hb : Appends b) : Appends ((liftM b : Transcript α).run d) := by
  show Appends ((monadLift b : Transcript α).run d)
  rw [StateT.run_monadLift]
  exact bind hb fun _ => pure _

theorem tforIn_list (l : List δ) (init : γ) {body : δ → γ → Transcript (ForInStep γ)} {d : Duplex}
    (h : ∀ x acc d, Appends ((body x acc).run d)) : Appends ((forIn l init body).run d) := by
  induction l generalizing init d with
  | nil => exact pure _
  | cons x l ih =>
      rw [List.forIn_cons]
      refine tbind (h x init) fun r d => ?_
      cases r with
      | done b => exact pure _
      | yield b => exact ih b

theorem tforIn_array (xs : Array δ) (init : γ) {body : δ → γ → Transcript (ForInStep γ)} {d : Duplex}
    (h : ∀ x acc d, Appends ((body x acc).run d)) : Appends ((forIn xs init body).run d) := by
  rw [← Array.forIn_toList]
  exact tforIn_list _ _ h

theorem tforIn_range (r : Std.Legacy.Range) (init : γ) {body : ℕ → γ → Transcript (ForInStep γ)} {d : Duplex}
    (h : ∀ x acc d, Appends ((body x acc).run d)) : Appends ((forIn r init body).run d) := by
  rw [Std.Legacy.Range.forIn_eq_forIn_range']
  exact tforIn_list _ _ h

theorem tfoldlM_list {f : γ → δ → Transcript γ} (h : ∀ s x d, Appends ((f s x).run d)) :
    ∀ (l : List δ) (init : γ) (d : Duplex), Appends ((l.foldlM f init).run d)
  | [], _, _ => pure _
  | x :: l, init, _ => by
      rw [List.foldlM_cons]
      exact tbind (h init x) fun s d => tfoldlM_list h l s d

theorem tforM_array {f : δ → Transcript PUnit} (h : ∀ x d, Appends ((f x).run d)) (xs : Array δ) (d : Duplex) :
    Appends ((xs.forM f).run d) := by
  unfold Array.forM
  rw [← Array.foldlM_toList]
  exact tfoldlM_list (fun _ x d => h x d) _ _ _

theorem tmapM_list {f : δ → Transcript α} (h : ∀ x d, Appends ((f x).run d)) :
    ∀ (l : List δ) (d : Duplex), Appends ((l.mapM f).run d)
  | [], _ => pure _
  | x :: l, _ => by
      rw [List.mapM_cons]
      exact tbind (h x) fun _ _ => tbind (fun d => tmapM_list h l d) fun _ _ => tpure _ _

theorem tmapM_array {f : δ → Transcript α} (h : ∀ x d, Appends ((f x).run d)) (xs : Array δ) (d : Duplex) :
    Appends ((xs.mapM f).run d) := by
  rw [Array.mapM_eq_mapM_toList, StateT.run_map]
  exact map _ (tmapM_list h _ d)

end Appends

class Keeps {α : Type} (t : Transcript α) : Prop where
  keeps : ∀ d p, ((t.run d).run p).1.2 = d

namespace Keeps

variable {α β γ δ : Type}

instance pure (a : α) : Keeps (Pure.pure a : Transcript α) := ⟨fun _ _ => rfl⟩

instance lift (b : Build α) : Keeps (liftM b : Transcript α) := ⟨fun _ _ => rfl⟩

theorem bind {t : Transcript α} {f : α → Transcript β} (ht : Keeps t) (hf : ∀ a, Keeps (f a)) : Keeps (t >>= f) :=
  ⟨fun d p => by
    show (((f ((t.run d).run p).1.1).run ((t.run d).run p).1.2).run ((t.run d).run p).2).1.2 = d
    rw [(hf _).keeps, ht.keeps]⟩

theorem map {t : Transcript α} (f : α → β) (ht : Keeps t) : Keeps (f <$> t) := ⟨fun d p => ht.keeps d p⟩

theorem ite {c : Prop} [Decidable c] {x y : Transcript α} (hx : Keeps x) (hy : Keeps y) :
    Keeps (if c then x else y) := by
  split <;> assumption

theorem dite {c : Prop} [Decidable c] {x : c → Transcript α} {y : ¬c → Transcript α} (hx : ∀ h, Keeps (x h))
    (hy : ∀ h, Keeps (y h)) : Keeps (if h : c then x h else y h) := by
  split
  · exact hx _
  · exact hy _

theorem forIn_list (l : List δ) (init : γ) {body : δ → γ → Transcript (ForInStep γ)}
    (h : ∀ x acc, Keeps (body x acc)) : Keeps (forIn l init body) := by
  induction l generalizing init with
  | nil => exact pure _
  | cons x l ih =>
      rw [List.forIn_cons]
      refine bind (h x init) fun r => ?_
      cases r with
      | done b => exact pure _
      | yield b => exact ih b

theorem forIn_array (xs : Array δ) (init : γ) {body : δ → γ → Transcript (ForInStep γ)}
    (h : ∀ x acc, Keeps (body x acc)) : Keeps (forIn xs init body) := by
  rw [← Array.forIn_toList]
  exact forIn_list _ _ h

theorem forIn_range (r : Std.Legacy.Range) (init : γ) {body : ℕ → γ → Transcript (ForInStep γ)}
    (h : ∀ x acc, Keeps (body x acc)) : Keeps (forIn r init body) := by
  rw [Std.Legacy.Range.forIn_eq_forIn_range']
  exact forIn_list _ _ h

theorem mapM_list {f : δ → Transcript α} (h : ∀ x, Keeps (f x)) : ∀ l : List δ, Keeps (l.mapM f)
  | [] => pure _
  | x :: l => by
      rw [List.mapM_cons]
      exact bind (h x) fun _ => bind (mapM_list h l) fun _ => pure _

theorem mapM_array {f : δ → Transcript α} (h : ∀ x, Keeps (f x)) (xs : Array δ) : Keeps (xs.mapM f) := by
  rw [Array.mapM_eq_mapM_toList]
  exact map _ (mapM_list h _)

end Keeps

section Tactic

open Lean Meta Elab Tactic

private def forInRule (ρ : Expr) (list array range : Name) : MetaM (Option Name) := do
  return match (← whnfR ρ).getAppFn with
    | .const ``List _ => some list
    | .const ``Array _ => some array
    | .const ``Std.Legacy.Range _ => some range
    | _ => none

private def appendsRule (b : Expr) : MetaM (Option Name) := do
  let forIn := forInRule
  match b.getAppFn with
  | .const ``Bind.bind _ => return some ``Appends.bind
  | .const ``Functor.map _ => return some ``Appends.map
  | .const ``ite _ => return some ``Appends.ite
  | .const ``dite _ => return some ``Appends.dite
  | .const ``ForIn.forIn _ => forIn b.getAppArgs[1]! ``Appends.forIn_list ``Appends.forIn_array ``Appends.forIn_range
  | .const ``Array.mapM _ => return some ``Appends.mapM_array
  | .const ``List.mapM _ => return some ``Appends.mapM_list
  | .const ``StateT.run _ =>
    let t := b.getAppArgs[3]!.headBeta
    match t.getAppFn with
    | .const ``Bind.bind _ => return some ``Appends.tbind
    | .const ``ite _ => return some ``Appends.tite
    | .const ``liftM _ | .const ``MonadLiftT.monadLift _ => return some ``Appends.tlift
    | .const ``ForIn.forIn _ =>
      forIn t.getAppArgs[1]! ``Appends.tforIn_list ``Appends.tforIn_array ``Appends.tforIn_range
    | .const ``List.foldlM _ => return some ``Appends.tfoldlM_list
    | .const ``Array.forM _ => return some ``Appends.tforM_array
    | .const ``Array.mapM _ => return some ``Appends.tmapM_array
    | .const ``List.mapM _ => return some ``Appends.tmapM_list
    | _ => return none
  | _ => return none

private def unfoldAbbrev? (b : Expr) : MetaM (Option Expr) := do
  let unfold (e : Expr) : MetaM (Option Expr) := do
    let .const name _ := e.getAppFn | return none
    unless ← isReducible name do return none
    unfoldDefinition? e
  match b.getAppFn with
  | .const ``StateT.run _ =>
    let args := b.getAppArgs
    return (← unfold args[3]!).map fun t => mkAppN b.getAppFn (args.set! 3 t)
  | _ => unfold b

private def keepsRule (t : Expr) : MetaM (Option Name) := do
  match t.getAppFn with
  | .const ``Bind.bind _ => return some ``Keeps.bind
  | .const ``Functor.map _ => return some ``Keeps.map
  | .const ``ite _ => return some ``Keeps.ite
  | .const ``dite _ => return some ``Keeps.dite
  | .const ``ForIn.forIn _ => forInRule t.getAppArgs[1]! ``Keeps.forIn_list ``Keeps.forIn_array ``Keeps.forIn_range
  | .const ``Array.mapM _ => return some ``Keeps.mapM_array
  | .const ``List.mapM _ => return some ``Keeps.mapM_list
  | _ => return none

private def shapeStep (cls : Name) (rule : Expr → MetaM (Option Name)) (g : MVarId) : TacticM (List MVarId) :=
    g.withContext do
  let ty ← instantiateMVars (← g.getType)
  if ty.isForall then return [(← g.intro1).2]
  let some (α, b) := ty.app2? cls | throwError "{cls}: the goal is not `{cls}`"
  if let some inst ← synthInstance? ty then
    g.assign inst
    return []
  let b := b.headBeta
  let letOf? := match b with
    | .letE n t v body _ => some (n, t, v, body, id)
    | _ => match b.getAppFn, b.getAppArgs[3]? with
      | Expr.const ``StateT.run _, some (Expr.letE n t v body _) =>
        some (n, t, v, body, fun e => mkAppN b.getAppFn (b.getAppArgs.set! 3 e))
      | _, _ => none
  if let some (n, t, v, body, wrap) := letOf? then
    if ← forallTelescope t fun _ r => return (← whnfR r).isAppOf ``StateT then
      return [← g.replaceTargetDefEq (mkApp2 (.const cls []) α (wrap (body.instantiate1 v)))]
    let g' ← mkFreshExprSyntheticOpaqueMVar (.forallE n t (mkApp2 (.const cls []) α (wrap body)) .default)
    g.assign (mkApp g' v)
    return [g'.mvarId!]
  if let some name ← rule b then
    return ← withReducible (g.apply (← mkConstWithFreshMVarLevels name))
  if let some b' ← unfoldAbbrev? b then
    return [← g.replaceTargetDefEq (mkApp2 (.const cls []) α b')]
  setGoals [g]
  try evalTactic (← `(tactic| split))
  catch _ => throwError "{cls}: no rule for the program{indentExpr b}"
  getGoals

private def shape (cls : Name) (rule : Expr → MetaM (Option Name)) : TacticM Unit := do
  let open_ ← Meta.repeat' (shapeStep cls rule) [← getMainGoal]
  if let g :: _ := open_ then
    discard <| shapeStep cls rule g
    throwError "{cls}: open goals remain"
  replaceMainGoal []

elab "appends" : tactic => shape ``Appends appendsRule

elab "keeps" : tactic => shape ``Keeps keepsRule

end Tactic

instance const_appends (g : Word) : Appends (Build.const g) := Appends.emit _
instance priv_appends : Appends Build.priv := Appends.emit _
instance pub_appends : Appends Build.pub := Appends.emit _
instance linear_appends (parts : List (Var × Word)) : Appends (Build.linear parts) := Appends.emit _
instance add_appends (a b : Var) : Appends (Build.add a b) := Appends.emit _
instance sub_appends (a b : Var) : Appends (Build.sub a b) := Appends.emit _
instance scale_appends (a : Var) (g : Word) : Appends (Build.scale a g) := Appends.emit _
instance mul_appends (a b : Var) : Appends (Build.mul a b) := Appends.emit _
instance extMul_appends (a b : ExtVar) : Appends (Build.extMul a b) := Appends.map _ (Appends.emit _)
instance extInv_appends (a : ExtVar) : Appends (Build.extInv a) := Appends.map _ (Appends.emit _)
instance bits_appends (a : Var) (count : ℕ) : Appends (Build.bits a count) := Appends.map _ (Appends.emit _)
instance hint_appends (count : ℕ) : Appends (Build.hint count) := Appends.map _ (Appends.emit _)
instance permute_appends (s : Fin 16 → Var) : Appends (Build.permute s) := Appends.map _ (Appends.emit _)
instance assertZero_appends (a : Var) : Appends (Build.assertZero a) := by unfold Build.assertZero; appends
instance r1cs_appends (rows : List (Shrink.R1csRow Word)) (columns : List Var) :
    Appends (Build.r1cs rows columns) := by
  unfold Build.r1cs; appends
instance extAdd_appends (a b : ExtVar) : Appends (Build.extAdd a b) := by
  unfold Build.extAdd Build.coordinatewise; appends
instance extSub_appends (a b : ExtVar) : Appends (Build.extSub a b) := by
  unfold Build.extSub Build.coordinatewise; appends
instance extScale_appends (a : ExtVar) (s : Var) : Appends (Build.extScale a s) := by
  unfold Build.extScale Build.coordinatewise; appends
instance select_appends (b w₀ w₁ : Var) : Appends (Build.select b w₀ w₁) := by unfold Build.select; appends
instance assertEqual_appends (a b : Var) : Appends (Build.assertEqual a b) := by unfold Build.assertEqual; appends
instance assertExtZero_appends (a : ExtVar) : Appends (Build.assertExtZero a) := by
  unfold Build.assertExtZero; appends
instance assertExtEqual_appends (a b : ExtVar) : Appends (Build.assertExtEqual a b) := by
  unfold Build.assertExtEqual; appends

variable {F : Type} [CommRing F]

def EnsuresAt {α : Type} (c : Ctx F) (p : Builder) (b : Build α) (Q : α → ℕ → Prop) : Prop :=
  ∀ (E : Exec.State F) (new : List Instr), E.values.length = p.values →
    (b.run p).2.instrs = p.instrs ++ new.toArray → c.Follows (c.steps new E) → (c.steps new E).holds →
    Q (b.run p).1 (b.run p).2.values

def Ensures {α : Type} (c : Ctx F) (n : ℕ) (b : Build α) (Q : α → ℕ → Prop) : Prop :=
  ∀ p : Builder, p.values = n → EnsuresAt c p b Q

variable {c : Ctx F} {α β γ δ : Type} {n : ℕ} {Q : α → ℕ → Prop}

namespace EnsuresAt

theorem bind {p : Builder} {b : Build α} {f : α → Build β} {R : β → ℕ → Prop} (hb : EnsuresAt c p b Q)
    (hf : p.values ≤ (b.run p).2.values → Q (b.run p).1 (b.run p).2.values →
      EnsuresAt c (b.run p).2 (f (b.run p).1) R)
    (hb' : Appends b := by appends) (hf' : ∀ a, Appends (f a) := by appends) : EnsuresAt c p (b >>= f) R := by
  intro E new hE hi hF hh
  obtain ⟨new₁, hi₁, hv₁⟩ := hb'.out p
  obtain ⟨new₂, hi₂, -⟩ := (hf' (b.run p).1).out (b.run p).2
  obtain rfl : new = new₁ ++ new₂ := by
    have : p.instrs ++ new.toArray = p.instrs ++ (new₁ ++ new₂).toArray := by
      rw [← hi]
      show ((f (b.run p).1).run (b.run p).2).2.instrs = _
      rw [hi₂, hi₁]
      simp
    exact List.toArray_inj ((Array.append_right_inj _).mp this)
  rw [Ctx.steps_append] at hF hh
  have hl : (c.steps new₁ E).values.length = (b.run p).2.values := by
    obtain ⟨more, hm, hlen⟩ := c.steps_values new₁ E
    rw [hm, List.length_append, hE, hlen, hv₁]
  exact hf (hb'.le p) (hb E new₁ hE hi₁ (Ctx.follows_of_steps hF) (Ctx.holds_of_steps hh)) _ new₂ hl hi₂ hF hh

theorem post {p : Builder} {b : Build α} {Q' : α → ℕ → Prop} (h : EnsuresAt c p b Q)
    (hq : Q (b.run p).1 (b.run p).2.values → Q' (b.run p).1 (b.run p).2.values) : EnsuresAt c p b Q' :=
  fun E new hE hi hF hh => hq (h E new hE hi hF hh)

theorem exec {b : Build α} (h : EnsuresAt c {} b Q) (hval : c.val = (Exec.run c.perm c.stmt c.tape b.program).value)
    (accepts : Exec c.perm b.program c.stmt c.tape) : Q (b.run {}).1 (b.run {}).2.values := by
  have run : Exec.run c.perm c.stmt c.tape b.program = c.steps b.program.toList Exec.start := rfl
  refine h Exec.start b.program.toList rfl (by simp [Build.program]) (fun v _ => by rw [hval, run]) ?_
  rw [Exec, run] at accepts
  exact accepts

end EnsuresAt

namespace Ensures

theorem pure {a : α} (h : Q a n) : Ensures c n (Pure.pure a) Q := fun _ hp _ _ _ _ _ _ => hp ▸ h

theorem bind {b : Build α} {f : α → Build β} {R : β → ℕ → Prop} (hb : Ensures c n b Q)
    (hf : ∀ a m, n ≤ m → Q a m → Ensures c m (f a) R)
    (hb' : Appends b := by appends) (hf' : ∀ a, Appends (f a) := by appends) : Ensures c n (b >>= f) R :=
  fun p hp => (hb p hp).bind (fun le hq => hf _ _ (hp ▸ le) hq _ rfl) hb' hf'

theorem keeps {t : Transcript α} {d : Duplex} (hk : Keeps t := by keeps) (ht : Appends (t.run d) := by appends) :
    Ensures c n (t.run d) fun r m => n ≤ m ∧ r.2 = d :=
  fun p hp _ _ _ _ _ _ => ⟨hp ▸ ht.le p, hk.keeps d p⟩

theorem post {b : Build α} {Q' : α → ℕ → Prop} (h : Ensures c n b Q) (hq : ∀ a m, Q a m → Q' a m) :
    Ensures c n b Q' :=
  fun p hp E new hE hi hF hh => hq _ _ (h p hp E new hE hi hF hh)

theorem map {b : Build α} {f : α → β} {Q' : β → ℕ → Prop} (h : Ensures c n b Q) (hq : ∀ a m, Q a m → Q' (f a) m) :
    Ensures c n (f <$> b) Q' :=
  fun p hp E new hE hi hF hh => hq _ _ (h p hp E new hE hi hF hh)

theorem emit (i : Instr) {Q : Var → ℕ → Prop}
    (h : ∀ E : Exec.State F, E.values.length = n → c.Follows (c.step E i) → (c.step E i).holds →
      Q n (n + i.outputs)) :
    Ensures c n (Build.emit i) Q := by
  intro p hp E new hE hi hF hh
  obtain rfl : new = [i] := by
    have : p.instrs ++ new.toArray = p.instrs ++ [i].toArray := by
      rw [← hi]
      show p.instrs.push i = _
      simp
    exact List.toArray_inj ((Array.append_right_inj _).mp this)
  subst hp
  exact h E hE hF hh

private theorem loop_from {xs : List δ} {body : δ → γ → Build (ForInStep γ)} (I : ℕ → γ → ℕ → Prop)
    (step : ∀ k (hk : k < xs.length) acc m, n ≤ m → I k acc m →
      Ensures c m (body xs[k] acc) fun r m' => r = .yield r.value ∧ I (k + 1) r.value m')
    (hb : ∀ x acc, Appends (body x acc)) :
    ∀ t j, xs.length - j = t → j ≤ xs.length → ∀ acc m, n ≤ m → I j acc m →
      Ensures c m (forIn (xs.drop j) acc body) fun r m' => I xs.length r m'
  | 0, j, ht, hj, acc, m, _, hI => by
      obtain rfl : j = xs.length := by omega
      rw [List.drop_length, List.forIn_nil]
      exact Ensures.pure hI
  | t + 1, j, ht, hj, acc, m, hm, hI => by
      have hj' : j < xs.length := by omega
      rw [List.drop_eq_getElem_cons hj', List.forIn_cons]
      refine (step j hj' acc m hm hI).bind (fun r m' le hr => ?_) (hb _ _) fun r => ?_
      · obtain ⟨hy, hI'⟩ := hr
        rw [hy]
        exact loop_from I step hb t (j + 1) (by omega) hj' r.value m' (hm.trans le) hI'
      · cases r with
        | done b => exact Appends.pure _
        | yield b => exact Appends.forIn_list _ _ hb

theorem loop {xs : List δ} {init : γ} {body : δ → γ → Build (ForInStep γ)} (I : ℕ → γ → ℕ → Prop)
    (h0 : I 0 init n)
    (step : ∀ k (hk : k < xs.length) acc m, n ≤ m → I k acc m →
      Ensures c m (body xs[k] acc) fun r m' => r = .yield r.value ∧ I (k + 1) r.value m')
    (hb : ∀ x acc, Appends (body x acc) := by appends) :
    Ensures c n (forIn xs init body) fun r m => I xs.length r m :=
  loop_from I step hb _ 0 rfl (Nat.zero_le _) init n le_rfl h0

theorem loopArray {xs : Array δ} {init : γ} {body : δ → γ → Build (ForInStep γ)} (I : ℕ → γ → ℕ → Prop)
    (h0 : I 0 init n)
    (step : ∀ k (hk : k < xs.size) acc m, n ≤ m → I k acc m →
      Ensures c m (body xs[k] acc) fun r m' => r = .yield r.value ∧ I (k + 1) r.value m')
    (hb : ∀ x acc, Appends (body x acc) := by appends) :
    Ensures c n (forIn xs init body) fun r m => I xs.size r m := by
  rw [← Array.forIn_toList]
  have h := loop (xs := xs.toList) I h0 (fun k hk acc m le hI => by
    simpa only [Array.getElem_toList] using step k (by simpa using hk) acc m le hI) hb
  simpa only [Array.length_toList] using h

theorem loopRange {a b : ℕ} {init : γ} {body : ℕ → γ → Build (ForInStep γ)} (I : ℕ → γ → ℕ → Prop)
    (h0 : I 0 init n)
    (step : ∀ k < b - a, ∀ acc m, n ≤ m → I k acc m →
      Ensures c m (body (a + k) acc) fun r m' => r = .yield r.value ∧ I (k + 1) r.value m')
    (hb : ∀ x acc, Appends (body x acc) := by appends) :
    Ensures c n (forIn [a:b] init body) fun r m => I (b - a) r m := by
  rw [Std.Legacy.Range.forIn_eq_forIn_range']
  have size : [a:b].size = b - a := by simp [Std.Legacy.Range.size]
  refine (loop I h0 (fun k hk acc m le hI => ?_) hb).post fun _ _ h => by simpa [size] using h
  rw [List.getElem_range', Nat.one_mul]
  exact step k (by simpa [size] using hk) acc m le hI

theorem tbind {t : Transcript α} {g : α → Transcript β} {d : Duplex} {Q : α × Duplex → ℕ → Prop}
    {R : β × Duplex → ℕ → Prop} (ht : Ensures c n (t.run d) Q)
    (hg : ∀ a e m, n ≤ m → Q (a, e) m → Ensures c m ((g a).run e) R)
    (ht' : Appends (t.run d) := by appends) (hg' : ∀ a e, Appends ((g a).run e) := by appends) :
    Ensures c n ((t >>= g).run d) R := by
  rw [StateT.run_bind]
  exact ht.bind (fun r m le hq => hg r.1 r.2 m le hq) ht' fun r => hg' r.1 r.2

theorem lift {b : Build α} {d : Duplex} (h : Ensures c n b Q) (hb : Appends b := by appends) :
    Ensures c n ((liftM b : Transcript α).run d) fun r m => Q r.1 m ∧ r.2 = d := by
  show Ensures c n ((monadLift b : Transcript α).run d) _
  rw [StateT.run_monadLift]
  exact h.bind (fun a m _ hq => Ensures.pure ⟨hq, rfl⟩) hb

theorem liftBind {b : Build α} {g : α → Transcript β} {d : Duplex} {R : β × Duplex → ℕ → Prop}
    (hb : Ensures c n b Q) (hg : ∀ a m, n ≤ m → Q a m → Ensures c m ((g a).run d) R)
    (hb' : Appends b := by appends) (hg' : ∀ a e, Appends ((g a).run e) := by appends) :
    Ensures c n (((liftM b : Transcript α) >>= g).run d) R :=
  (lift hb hb').tbind (fun a e m le hq => by
    obtain ⟨hq, he⟩ := hq
    dsimp only at he
    subst he
    exact hg a m le hq) (Appends.tlift hb') hg'

private theorem loopT_from {xs : List δ} {body : δ → γ → Transcript (ForInStep γ)} (I : ℕ → γ → Duplex → ℕ → Prop)
    (step : ∀ k (hk : k < xs.length) acc e m, n ≤ m → I k acc e m →
      Ensures c m ((body xs[k] acc).run e) fun r m' => r.1 = .yield r.1.value ∧ I (k + 1) r.1.value r.2 m')
    (hb : ∀ x acc e, Appends ((body x acc).run e)) :
    ∀ t j, xs.length - j = t → j ≤ xs.length → ∀ acc e m, n ≤ m → I j acc e m →
      Ensures c m ((forIn (xs.drop j) acc body).run e) fun r m' => I xs.length r.1 r.2 m'
  | 0, j, ht, hj, acc, e, m, _, hI => by
      obtain rfl : j = xs.length := by omega
      rw [List.drop_length, List.forIn_nil, StateT.run_pure]
      exact Ensures.pure hI
  | t + 1, j, ht, hj, acc, e, m, hm, hI => by
      have hj' : j < xs.length := by omega
      rw [List.drop_eq_getElem_cons hj', List.forIn_cons, StateT.run_bind]
      refine (step j hj' acc e m hm hI).bind (fun r m' le hr => ?_) (hb _ _ _) fun r => ?_
      · obtain ⟨r, e'⟩ := r
        obtain ⟨hy, hI'⟩ := hr
        dsimp only at hy hI' ⊢
        rw [hy]
        exact loopT_from I step hb t (j + 1) (by omega) hj' r.value e' m' (hm.trans le) hI'
      · obtain ⟨r, e'⟩ := r
        cases r with
        | done b => exact Appends.pure _
        | yield b => exact Appends.tforIn_list _ _ hb

theorem loopT {xs : List δ} {init : γ} {body : δ → γ → Transcript (ForInStep γ)} {d : Duplex}
    (I : ℕ → γ → Duplex → ℕ → Prop) (h0 : I 0 init d n)
    (step : ∀ k (hk : k < xs.length) acc e m, n ≤ m → I k acc e m →
      Ensures c m ((body xs[k] acc).run e) fun r m' => r.1 = .yield r.1.value ∧ I (k + 1) r.1.value r.2 m')
    (hb : ∀ x acc e, Appends ((body x acc).run e) := by appends) :
    Ensures c n ((forIn xs init body).run d) fun r m => I xs.length r.1 r.2 m :=
  loopT_from I step hb _ 0 rfl (Nat.zero_le _) init d n le_rfl h0

theorem loopTArray {xs : Array δ} {init : γ} {body : δ → γ → Transcript (ForInStep γ)} {d : Duplex}
    (I : ℕ → γ → Duplex → ℕ → Prop) (h0 : I 0 init d n)
    (step : ∀ k (hk : k < xs.size) acc e m, n ≤ m → I k acc e m →
      Ensures c m ((body xs[k] acc).run e) fun r m' => r.1 = .yield r.1.value ∧ I (k + 1) r.1.value r.2 m')
    (hb : ∀ x acc e, Appends ((body x acc).run e) := by appends) :
    Ensures c n ((forIn xs init body).run d) fun r m => I xs.size r.1 r.2 m := by
  rw [← Array.forIn_toList]
  have h := loopT (xs := xs.toList) I h0 (fun k hk acc e m le hI => by
    simpa only [Array.getElem_toList] using step k (by simpa using hk) acc e m le hI) hb
  simpa only [Array.length_toList] using h

theorem listMapM {f : δ → Build α} (R : δ → α → ℕ → Prop) (mono : ∀ x a m m', R x a m → m ≤ m' → R x a m')
    (hf : ∀ x, Appends (f x)) :
    ∀ (xs : List δ) {n : ℕ}, (∀ x ∈ xs, ∀ m, n ≤ m → Ensures c m (f x) (R x)) →
      Ensures c n (xs.mapM f) fun as m => List.Forall₂ (fun x a => R x a m) xs as
  | [], n, _ => by
      rw [List.mapM_nil]
      exact Ensures.pure List.Forall₂.nil
  | x :: xs, n, h => by
      rw [List.mapM_cons]
      refine (h x List.mem_cons_self n le_rfl).bind (fun a m le ha => ?_) (hf x)
        fun _ => Appends.bind (Appends.mapM_list hf xs) fun _ => Appends.pure _
      refine (listMapM R mono hf xs fun y hy m' le' => h y (List.mem_cons_of_mem _ hy) m' (le.trans le')).bind
        (fun as m' le' has => Ensures.pure (List.Forall₂.cons (mono _ _ _ _ ha le') has))
        (Appends.mapM_list hf xs) fun _ => Appends.pure _

theorem mapM {xs : Array δ} {f : δ → Build α} (R : δ → α → ℕ → Prop)
    (mono : ∀ x a m m', R x a m → m ≤ m' → R x a m') (h : ∀ x ∈ xs, ∀ m, n ≤ m → Ensures c m (f x) (R x))
    (hf : ∀ x, Appends (f x) := by appends) :
    Ensures c n (xs.mapM f) fun as m => List.Forall₂ (fun x a => R x a m) xs.toList as.toList := by
  rw [Array.mapM_eq_mapM_toList]
  exact (listMapM R mono hf xs.toList fun x hx => h x (Array.mem_toList_iff.mp hx)).map fun as m h => by simpa using h

end Ensures

end CompressionLean.Sim
