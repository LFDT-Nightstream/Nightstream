import CompressionLean.Circuit.Record

namespace CompressionLean.Circuit

open CompressionLean.Shrink

section Names

variable {F : Type}

def go {α : Type} (m : Lower F α) (s : State F) : α × State F := (m.run s).run

theorem go_bind {α β : Type} (m : Lower F α) (k : α → Lower F β) (s : State F) :
    go (m >>= k) s = go (k (go m s).1) (go m s).2 := rfl

namespace State

def WireIn (s : State F) : Wire → Prop
  | .glue c => c < s.cells
  | .block b k => b < s.blocks ∧ k < blockCells
  | .pub i => i ≤ s.publics.size

def TermsIn (s : State F) (terms : List (Wire × F)) : Prop := ∀ t ∈ terms, s.WireIn t.1

structure Le (s t : State F) : Prop where
  cells : s.cells ≤ t.cells
  blocks : s.blocks ≤ t.blocks
  publics : s.publics.size ≤ t.publics.size

theorem Le.refl (s : State F) : s.Le s := ⟨le_rfl, le_rfl, le_rfl⟩

theorem Le.trans {s t u : State F} (h : s.Le t) (h' : t.Le u) : s.Le u :=
  ⟨h.cells.trans h'.cells, h.blocks.trans h'.blocks, h.publics.trans h'.publics⟩

theorem WireIn.mono {s t : State F} (h : s.Le t) : ∀ {w : Wire}, s.WireIn w → t.WireIn w
  | .glue _, hw => lt_of_lt_of_le hw h.cells
  | .block _ _, hw => ⟨lt_of_lt_of_le hw.1 h.blocks, hw.2⟩
  | .pub _, hw => le_trans hw h.publics

theorem TermsIn.mono {s t : State F} (h : s.Le t) {terms : List (Wire × F)} (hs : s.TermsIn terms) :
    t.TermsIn terms :=
  fun x hx => WireIn.mono h (hs x hx)

theorem TermsIn.append {s : State F} {a b : List (Wire × F)} (ha : s.TermsIn a) (hb : s.TermsIn b) :
    s.TermsIn (a ++ b) := by
  intro x hx
  rcases List.mem_append.mp hx with h | h
  · exact ha x h
  · exact hb x h

theorem TermsIn.nil (s : State F) : s.TermsIn ([] : List (Wire × F)) := by
  intro x hx
  simp at hx

theorem TermsIn.one (s : State F) (c : F) : s.TermsIn [(Wire.pub 0, c)] := by
  intro x hx
  simp only [List.mem_singleton] at hx
  subst hx
  exact Nat.zero_le _

theorem TermsIn.glue {s : State F} {c : ℕ} (h : c < s.cells) (coef : F) : s.TermsIn [(Wire.glue c, coef)] := by
  intro x hx
  simp only [List.mem_singleton] at hx
  subst hx
  exact h

end State

def Stable (P : State F → Prop) : Prop := ∀ s t, s.Le t → P s → P t

namespace Stable

theorem and {P Q : State F → Prop} (hp : Stable P) (hq : Stable Q) : Stable fun s => P s ∧ Q s :=
  fun s t le h => ⟨hp s t le h.1, hq s t le h.2⟩

theorem terms (ts : List (Wire × F)) : Stable fun s : State F => s.TermsIn ts :=
  fun _ _ le h => h.mono le

theorem all {n : ℕ} (a : Fin n → Form F) : Stable fun s : State F => ∀ i, s.TermsIn (a i).terms :=
  fun _ _ le h i => (h i).mono le

theorem cells (n : ℕ) : Stable fun s : State F => n ≤ s.cells :=
  fun _ _ le h => h.trans le.cells

theorem blocks (b : ℕ) : Stable fun s : State F => b < s.blocks :=
  fun _ _ le h => lt_of_lt_of_le h le.blocks

theorem below (cs : List ℕ) : Stable fun s : State F => ∀ c ∈ cs, c < s.cells :=
  fun _ _ le h c hc => lt_of_lt_of_le (h c hc) le.cells

end Stable

def ExtIn (a b : Fin 3 → Form F) (s : State F) : Prop :=
  (∀ i, s.TermsIn (a i).terms) ∧ ∀ i, s.TermsIn (b i).terms

theorem stable_extIn (a b : Fin 3 → Form F) : Stable (ExtIn a b) :=
  (Stable.all a).and (Stable.all b)

theorem vector_terms {s : State F} {x y z : Form F} (hx : s.TermsIn x.terms) (hy : s.TermsIn y.terms)
    (hz : s.TermsIn z.terms) : ∀ i, s.TermsIn ((![x, y, z] : Fin 3 → Form F) i).terms := by
  intro i
  fin_cases i
  exacts [hx, hy, hz]

theorem ofCall_terms {s : State F} {products : List (List (Wire × F) × List (Wire × F))}
    {linear : List (Wire × F)} (hp : ∀ p ∈ products, s.TermsIn p.1 ∧ s.TermsIn p.2) (hl : s.TermsIn linear)
    (side : Fin 7) : s.TermsIn ((GlueRow.ofCall products linear).side side) := by
  unfold GlueRow.ofCall
  dsimp only
  split_ifs
  · exact hl
  all_goals
    split
    · rename_i pair h
      first
        | exact (hp pair (List.mem_of_getElem? h)).1
        | exact (hp pair (List.mem_of_getElem? h)).2
    · exact State.TermsIn.nil s

end Names

section Lowering

variable {F : Type} [Field F]

namespace State

def CallIn (s : State F) (c : Call F) : Prop := ∀ row ∈ c.rows, ∀ side, s.TermsIn (row.side side)

def Ok (s : State F) : Prop := (∀ c ∈ s.calls, s.CallIn c) ∧ ∀ f ∈ s.values, s.TermsIn f.terms

theorem CallIn.mono {s t : State F} (h : s.Le t) {c : Call F} (hc : s.CallIn c) : t.CallIn c :=
  fun row hrow side => (hc row hrow side).mono h

theorem Ok.grow {s t : State F} (ok : s.Ok) (h : s.Le t) (calls : t.calls = s.calls) (values : t.values = s.values) :
    t.Ok := by
  refine ⟨fun c hc => (ok.1 c (calls ▸ hc)).mono h, fun f hf => (ok.2 f (values ▸ hf)).mono h⟩

end State

def Spec {α : Type} (P : State F → Prop) (m : Lower F α) (Q : α → State F → Prop) : Prop :=
  ∀ s, s.Ok → P s → (go m s).2.Ok ∧ s.Le (go m s).2 ∧ Q (go m s).1 (go m s).2

namespace Spec

theorem pure {α : Type} {a : α} {P : State F → Prop} {Q : α → State F → Prop} (h : ∀ s, P s → Q a s) :
    Spec P (Pure.pure a : Lower F α) Q :=
  fun s ok p => ⟨ok, State.Le.refl s, h s p⟩

theorem bind {α β : Type} {m : Lower F α} {k : α → Lower F β} {P : State F → Prop} {Q : α → State F → Prop}
    {R : β → State F → Prop} (hm : Spec P m Q) (hk : ∀ a, Spec (Q a) (k a) R) : Spec P (m >>= k) R := by
  intro s ok p
  obtain ⟨ok₁, le₁, q⟩ := hm s ok p
  obtain ⟨ok₂, le₂, r⟩ := hk _ _ ok₁ q
  rw [go_bind]
  exact ⟨ok₂, le₁.trans le₂, r⟩

theorem weaken {α : Type} {m : Lower F α} {P P' : State F → Prop} {Q Q' : α → State F → Prop} (h : Spec P m Q)
    (hp : ∀ s, P' s → P s) (hq : ∀ a s, Q a s → Q' a s) : Spec P' m Q' := by
  intro s ok p
  obtain ⟨ok', le, q⟩ := h s ok (hp s p)
  exact ⟨ok', le, hq _ _ q⟩

theorem keep {α β : Type} {m : Lower F α} {k : α → Lower F β} {P : State F → Prop} {Q : α → State F → Prop}
    {R : β → State F → Prop} (stable : Stable P) (hm : Spec P m Q) (hk : ∀ a, Spec (fun t => P t ∧ Q a t) (k a) R) :
    Spec P (m >>= k) R := by
  intro s ok p
  obtain ⟨ok₁, le₁, q⟩ := hm s ok p
  obtain ⟨ok₂, le₂, r⟩ := hk _ _ ok₁ ⟨stable _ _ le₁ p, q⟩
  rw [go_bind]
  exact ⟨ok₂, le₁.trans le₂, r⟩

theorem seq {β : Type} {m : Lower F Unit} {k : Unit → Lower F β} {P : State F → Prop}
    {R : β → State F → Prop} (stable : Stable P) (hm : Spec P m fun _ _ => True) (hk : ∀ u, Spec P (k u) R) :
    Spec P (m >>= k) R :=
  keep stable hm fun u => (hk u).weaken (fun _ h => h.1) fun _ _ h => h

end Spec

theorem getD_terms {s : State F} {columns : List (Form F)} (hc : ∀ f ∈ columns, s.TermsIn f.terms) (i : ℕ) :
    s.TermsIn (columns.getD i ⟨[], 0⟩).terms := by
  rw [List.getD_eq_getElem?_getD]
  cases h : columns[i]? with
  | none => exact State.TermsIn.nil s
  | some f => exact hc f (List.mem_of_getElem? h)

theorem expand_terms {s : State F} {columns : List (Form F)} (hc : ∀ f ∈ columns, s.TermsIn f.terms)
    (side : List (ℕ × F)) : s.TermsIn (expand side columns) := by
  intro x hx
  unfold expand at hx
  simp only [List.mem_flatMap, List.mem_append, List.mem_map] at hx
  obtain ⟨t, -, hx⟩ := hx
  rcases hx with ⟨u, hu, rfl⟩ | hx
  · exact getD_terms hc t.1 u hu
  · split_ifs at hx
    · exact State.TermsIn.one s _ x hx
    · simp at hx

theorem State.Ok.pushCall {s : State F} {c : Call F} (ok : s.Ok) (hc : s.CallIn c) :
    ({ s with calls := s.calls.push c } : State F).Ok := by
  refine ⟨fun c' hc' => ?_, ok.2⟩
  rcases Array.mem_push.mp hc' with h | h
  · exact ok.1 c' h
  · exact h ▸ hc

theorem row_spec (products : List (List (Wire × F) × List (Wire × F))) (linear : List (Wire × F))
    (three : products.length ≤ 3) :
    Spec (fun s => (∀ p ∈ products, s.TermsIn p.1 ∧ s.TermsIn p.2) ∧ s.TermsIn linear)
      (row products linear three) (fun _ _ => True) := by
  intro s ok p
  refine ⟨ok.pushCall ?_, ⟨le_rfl, le_rfl, le_rfl⟩, trivial⟩
  intro r hr side
  simp only [Call.rows, List.mem_singleton] at hr
  subst hr
  exact ofCall_terms p.1 p.2 side

theorem r1cs_spec (rows : List (R1csRow F)) (columns : List (Form F)) :
    Spec (fun s => ∀ f ∈ columns, s.TermsIn f.terms) (r1cs rows columns) (fun _ _ => True) := by
  intro s ok hc
  refine ⟨ok.pushCall ?_, ⟨le_rfl, le_rfl, le_rfl⟩, trivial⟩
  intro r hr side
  simp only [Call.rows, List.mem_map] at hr
  obtain ⟨_, -, rfl⟩ := hr
  refine ofCall_terms ?_ (expand_terms hc _) side
  intro p hp
  simp only [List.mem_singleton] at hp
  subst hp
  exact ⟨expand_terms hc _, expand_terms hc _⟩

theorem cell_spec : Spec (fun _ => True) (cell : Lower F (Form F)) (fun f t => t.TermsIn f.terms) := by
  intro s ok _
  have le : s.Le { s with cells := s.cells + 1 } := ⟨Nat.le_succ _, le_rfl, le_rfl⟩
  exact ⟨ok.grow le rfl rfl, le, State.TermsIn.glue (Nat.lt_succ_self _) 1⟩

theorem block_spec : Spec (fun _ => True) (block : Lower F ℕ) (fun b t => b < t.blocks) := by
  intro s ok _
  have le : s.Le { s with blocks := s.blocks + 1 } := ⟨le_rfl, Nat.le_succ _, le_rfl⟩
  exact ⟨ok.grow le rfl rfl, le, Nat.lt_succ_self _⟩

theorem cells_spec (count : ℕ) :
    Spec (fun _ => True) (cells count : Lower F ℕ) (fun first t => first + count ≤ t.cells) := by
  intro s ok _
  have le : s.Le { s with cells := s.cells + count } := ⟨Nat.le_add_right _ _, le_rfl, le_rfl⟩
  exact ⟨ok.grow le rfl rfl, le, le_rfl⟩

theorem statementWord_spec (stmt : Array F) :
    Spec (fun _ => True) (statementWord stmt) (fun f t => t.TermsIn f.terms) := by
  intro s ok _
  have le : s.Le { s with publics := s.publics.push (stmt.getD s.publics.size 0) } :=
    ⟨le_rfl, le_rfl, by show s.publics.size ≤ (s.publics.push _).size; simp⟩
  refine ⟨ok.grow le rfl rfl, le, ?_⟩
  intro x hx
  change x ∈ [(Wire.pub (s.publics.size + 1), (1 : F))] at hx
  simp only [List.mem_singleton] at hx
  subst hx
  show s.publics.size + 1 ≤ (s.publics.push _).size
  simp

theorem push_spec (forms : List (Form F)) :
    Spec (fun s => ∀ f ∈ forms, s.TermsIn f.terms) (push forms) (fun _ _ => True) := by
  intro s ok hf
  refine ⟨⟨ok.1, fun f h => ?_⟩, ⟨le_rfl, le_rfl, le_rfl⟩, trivial⟩
  rcases Array.mem_append.mp h with h | h
  · exact ok.2 f h
  · exact hf f (List.mem_toArray.mp h)

theorem values_terms {s : State F} (ok : s.Ok) (v : ℕ) : s.TermsIn (s.values.getD v (.const 0)).terms := by
  rw [Array.getD_eq_getD_getElem?]
  cases h : s.values[v]? with
  | none => exact State.TermsIn.nil s
  | some f => exact ok.2 f (Array.mem_of_getElem? h)

theorem readValue_spec (v : Var) :
    Spec (fun _ => True) (readValue v : Lower F (Form F)) (fun f t => t.TermsIn f.terms) :=
  fun s ok _ => ⟨ok, State.Le.refl s, values_terms ok v⟩

theorem readValues_spec (vs : List Var) :
    Spec (fun _ => True) (readValues vs : Lower F (List (Form F))) (fun fs t => ∀ f ∈ fs, t.TermsIn f.terms) := by
  intro s ok _
  refine ⟨ok, State.Le.refl s, fun f hf => ?_⟩
  obtain ⟨v, -, rfl⟩ := List.mem_map.mp hf
  exact values_terms ok v

theorem readCoordinates_spec {n : ℕ} (a : Fin n → Var) :
    Spec (fun _ => True) (readCoordinates a : Lower F (Fin n → Form F)) (fun fs t => ∀ k, t.TermsIn (fs k).terms) := by
  intro s ok _
  refine ⟨ok, State.Le.refl s, fun k => ?_⟩
  show s.TermsIn ((Array.ofFn fun k => s.values.getD (a k) (.const 0)).getD k (.const 0)).terms
  rw [Array.getD_eq_getD_getElem?]
  cases h : (Array.ofFn fun k => s.values.getD (a k) (.const 0))[(k : ℕ)]? with
  | none => exact State.TermsIn.nil s
  | some f =>
      obtain ⟨j, rfl⟩ := Array.mem_ofFn.mp (Array.mem_of_getElem? h)
      exact values_terms ok (a j)


theorem addTerm_terms {s : State F} {t : Wire × F} (ht : s.WireIn t.1) :
    ∀ {acc : List (Wire × F)}, s.TermsIn acc → s.TermsIn (addTerm t acc)
  | [], _ => by
      intro x hx
      simp only [addTerm, List.mem_singleton] at hx
      subst hx
      exact ht
  | u :: rest, h => by
      unfold addTerm
      split_ifs
      · intro x hx
        rcases List.mem_cons.mp hx with rfl | hx
        · exact h u List.mem_cons_self
        · exact h x (List.mem_cons_of_mem _ hx)
      · intro x hx
        rcases List.mem_cons.mp hx with rfl | hx
        · exact h x List.mem_cons_self
        · exact addTerm_terms ht (fun y hy => h y (List.mem_cons_of_mem _ hy)) x hx

theorem foldl_addTerm_terms {s : State F} (w : F) :
    ∀ (terms acc : List (Wire × F)), s.TermsIn terms → s.TermsIn acc →
      s.TermsIn (terms.foldl (fun acc t => addTerm (t.1, t.2 * w) acc) acc)
  | [], _, _, h => h
  | t :: rest, _, ht, h =>
      foldl_addTerm_terms w rest _ (fun x hx => ht x (List.mem_cons_of_mem _ hx))
        (addTerm_terms (t := (t.1, t.2 * w)) (ht t List.mem_cons_self) h)

theorem mergeTerms_terms {s : State F} :
    ∀ (parts : List (Form F × F)) (acc : List (Wire × F)), (∀ p ∈ parts, s.TermsIn p.1.terms) → s.TermsIn acc →
      s.TermsIn (parts.foldl (fun terms p => p.1.terms.foldl (fun terms t => addTerm (t.1, t.2 * p.2) terms) terms) acc)
  | [], _, _, h => h
  | p :: rest, _, hp, h =>
      mergeTerms_terms rest _ (fun q hq => hp q (List.mem_cons_of_mem _ hq))
        (foldl_addTerm_terms p.2 _ _ (hp p List.mem_cons_self) h)

theorem negate_terms {s : State F} {terms : List (Wire × F)} (h : s.TermsIn terms) : s.TermsIn (negate terms) := by
  intro x hx
  obtain ⟨t, ht, rfl⟩ := List.mem_map.mp hx
  exact h t ht

section Folding

variable [DecidableEq F]

theorem combine_terms {s : State F} {parts : List (Form F × F)} (hp : ∀ p ∈ parts, s.TermsIn p.1.terms) :
    s.TermsIn (combine parts).1 :=
  fun x hx => mergeTerms_terms parts [] hp (State.TermsIn.nil s) x (List.mem_filter.mp hx).1

theorem rowTerms_terms {s : State F} {f : Form F} (h : s.TermsIn f.terms) : s.TermsIn f.rowTerms := by
  unfold Form.rowTerms
  refine h.append ?_
  split_ifs
  · exact State.TermsIn.nil s
  · exact State.TermsIn.one s _

theorem sumTerms_terms {s : State F} {forms : List (Form F)} (h : ∀ f ∈ forms, s.TermsIn f.terms) :
    s.TermsIn (sumTerms forms) := by
  unfold sumTerms
  refine State.TermsIn.append (combine_terms ?_) ?_
  · intro p hp
    obtain ⟨f, hf, rfl⟩ := List.mem_map.mp hp
    exact h f hf
  · split_ifs
    · exact State.TermsIn.nil s
    · exact State.TermsIn.one s _

theorem extRow_terms {s : State F} {a b : Fin 3 → Form F} (ha : ∀ i, s.TermsIn (a i).terms)
    (hb : ∀ i, s.TermsIn (b i).terms) (k : Fin 3) : ∀ p ∈ extRow a b k, s.TermsIn p.1 ∧ s.TermsIn p.2 := by
  have pair : ∀ i forms, (∀ f ∈ forms, s.TermsIn (Form.terms f)) →
      s.TermsIn (a i).rowTerms ∧ s.TermsIn (sumTerms forms) :=
    fun i forms h => ⟨rowTerms_terms (ha i), sumTerms_terms h⟩
  intro p hp
  fin_cases k <;> simp [extRow] at hp <;> rcases hp with rfl | rfl | rfl <;> exact pair _ _ (by simp [hb])

theorem form_spec (terms : List (Wire × F)) (constant : F) :
    Spec (fun s => s.TermsIn terms) (form terms constant) (fun f t => t.TermsIn f.terms) := by
  unfold form
  by_cases short : terms.length ≤ 4
  · rw [ite_eq_left short]
    exact Spec.pure fun _ h => h
  · rw [ite_eq_right short]
    refine Spec.keep (Stable.terms _) (cell_spec.weaken (fun _ _ => trivial) fun _ _ h => h) fun c => ?_
    refine Spec.keep ((Stable.terms _).and (Stable.terms _)) ((row_spec _ _ _).weaken ?_ fun _ _ h => h)
      fun _ => Spec.pure fun _ h => h.1.2
    intro s h
    refine ⟨fun p hp => by simp at hp, (h.2.append (negate_terms h.1)).append ?_⟩
    split_ifs
    · exact State.TermsIn.nil s
    · exact State.TermsIn.one s _

theorem linear_spec (parts : List (Form F × F)) :
    Spec (fun s => ∀ p ∈ parts, s.TermsIn p.1.terms) (linear parts) (fun f t => t.TermsIn f.terms) :=
  (form_spec (combine parts).1 (combine parts).2).weaken (fun _ h => combine_terms h) fun _ _ h => h

theorem linear3_spec (o : Fin 3 → Form F) (w0 w1 w2 : F) :
    Spec (fun s => ∀ i, s.TermsIn (o i).terms) (linear [(o 0, w0), (o 1, w1), (o 2, w2)])
      (fun f t => t.TermsIn f.terms) :=
  (linear_spec _).weaken (fun _ h p hp => by
    simp only [List.mem_cons, List.not_mem_nil, or_false] at hp
    rcases hp with rfl | rfl | rfl <;> exact h _) fun _ _ h => h

theorem mul_spec (a b : Form F) :
    Spec (fun s => s.TermsIn a.terms ∧ s.TermsIn b.terms) (mul a b) (fun f t => t.TermsIn f.terms) := by
  unfold mul
  by_cases ca : a.isConstant = true
  · rw [ite_eq_left ca]
    refine (linear_spec _).weaken (fun _ h p hp => ?_) fun _ _ h => h
    simp only [List.mem_singleton] at hp
    subst hp
    exact h.2
  rw [ite_eq_right ca]
  by_cases cb : b.isConstant = true
  · rw [ite_eq_left cb]
    refine (linear_spec _).weaken (fun _ h p hp => ?_) fun _ _ h => h
    simp only [List.mem_singleton] at hp
    subst hp
    exact h.1
  rw [ite_eq_right cb]
  refine Spec.keep ((Stable.terms _).and (Stable.terms _)) (cell_spec.weaken (fun _ _ => trivial) fun _ _ h => h)
    fun p => ?_
  refine Spec.keep (((Stable.terms _).and (Stable.terms _)).and (Stable.terms _))
    ((row_spec _ _ _).weaken ?_ fun _ _ h => h) fun _ => Spec.pure fun _ h => h.1.2
  intro s h
  refine ⟨fun q hq => ?_, h.2⟩
  simp only [List.mem_singleton] at hq
  subst hq
  exact ⟨rowTerms_terms h.1.1, rowTerms_terms h.1.2⟩

theorem extCell_spec (a b : Fin 3 → Form F) (k : Fin 3) :
    Spec (ExtIn a b) (extCell a b k) (fun f t => t.TermsIn f.terms) := by
  unfold extCell
  refine Spec.keep (stable_extIn a b) (cell_spec.weaken (fun _ _ => trivial) fun _ _ h => h) fun c => ?_
  refine Spec.keep ((stable_extIn a b).and (Stable.terms _)) ((row_spec _ _ _).weaken ?_ fun _ _ h => h)
    fun _ => Spec.pure fun _ h => h.1.2
  intro s h
  exact ⟨extRow_terms h.1.1 h.1.2 k, h.2⟩

omit [DecidableEq F] in
theorem fromFn_spec {P : State F → Prop} (stable : Stable P) {m : Fin 3 → Lower F (Form F)}
    (hm : ∀ k, Spec P (m k) (fun f t => t.TermsIn f.terms)) :
    Spec P (fromFn m) (fun f t => ∀ i, t.TermsIn (f i).terms) := by
  unfold fromFn
  refine Spec.keep stable (hm 0) fun x => ?_
  refine Spec.keep (stable.and (Stable.terms _)) ((hm 1).weaken (fun _ h => h.1) fun _ _ h => h) fun y => ?_
  refine Spec.keep ((stable.and (Stable.terms _)).and (Stable.terms _))
    ((hm 2).weaken (fun _ h => h.1.1) fun _ _ h => h) fun z => ?_
  exact Spec.pure fun _ h => vector_terms h.1.1.2 h.1.2 h.2

theorem extMul_spec (a b : Fin 3 → Form F) :
    Spec (ExtIn a b) (extMul a b) (fun f t => ∀ i, t.TermsIn (f i).terms) := by
  unfold extMul
  by_cases hc : (constants a || constants b) = true
  · rw [ite_eq_left hc]
    have other : ∀ s, ExtIn a b s → ∀ i, s.TermsIn ((if constants a = true then b else a) i).terms := by
      intro s h i
      by_cases ca : constants a = true
      · rw [ite_eq_left ca]
        exact h.2 i
      · rw [ite_eq_right ca]
        exact h.1 i
    dsimp only
    refine fromFn_spec (stable_extIn a b) fun k => ?_
    exact (linear3_spec _ _ _ _).weaken (fun s h => other s h) fun _ _ h => h
  · rw [ite_eq_right hc]
    exact fromFn_spec (stable_extIn a b) (extCell_spec a b)

theorem extInv_spec (a : Fin 3 → Form F) :
    Spec (fun s => ∀ i, s.TermsIn (a i).terms) (extInv a) (fun f t => ∀ i, t.TermsIn (f i).terms) := by
  unfold extInv
  split
  · exact Spec.pure fun _ _ _ => State.TermsIn.nil _
  · refine Spec.keep (Stable.all a)
      (fromFn_spec (Stable.all a) fun _ => cell_spec.weaken (fun _ _ => trivial) fun _ _ h => h) fun inverse => ?_
    have rows : ∀ (k : Fin 3) (linear : List (Wire × F)), (∀ s : State F, s.TermsIn linear) →
        Spec (ExtIn a inverse) (row (extRow a inverse k) linear (extRow_length _ _ _)) fun _ _ => True :=
      fun k linear hl => (row_spec _ _ _).weaken (fun s h => ⟨extRow_terms h.1 h.2 k, hl s⟩) fun _ _ h => h
    refine Spec.seq (stable_extIn a inverse) (rows 0 _ fun s => State.TermsIn.one s 1) fun _ => ?_
    refine Spec.seq (stable_extIn a inverse) (rows 1 _ State.TermsIn.nil) fun _ => ?_
    refine Spec.seq (stable_extIn a inverse) (rows 2 _ State.TermsIn.nil) fun _ => ?_
    exact Spec.pure fun _ h => h.2

theorem assertZero_spec (a : Form F) : Spec (fun s => s.TermsIn a.terms) (assertZero a) fun _ _ => True := by
  unfold assertZero
  split_ifs
  · exact Spec.pure fun _ _ => trivial
  · exact (row_spec _ _ _).weaken (fun _ h => ⟨fun p hp => by simp at hp, rowTerms_terms h⟩) fun _ _ h => h

omit [DecidableEq F] in
theorem booleanRows_spec :
    ∀ cs : List ℕ, Spec (fun s : State F => ∀ c ∈ cs, c < s.cells) (booleanRows cs) fun _ _ => True
  | [] => Spec.pure fun _ _ => trivial
  | c :: rest => by
      unfold booleanRows
      refine Spec.seq (Stable.below _) ((row_spec _ _ _).weaken (fun s h => ?_) fun _ _ h => h) fun _ =>
        (booleanRows_spec rest).weaken (fun _ h c' hc' => h c' (List.mem_cons_of_mem _ hc')) fun _ _ h => h
      have hc := State.TermsIn.glue (h c List.mem_cons_self) (1 : F)
      refine ⟨fun p hp => ?_, hc⟩
      simp only [List.mem_singleton] at hp
      subst hp
      exact ⟨hc, hc⟩

theorem chain_spec : ∀ (cs : List ℕ) (high : Form F),
    Spec (fun s => s.TermsIn high.terms ∧ ∀ c ∈ cs, c < s.cells) (chain high cs) fun f t => t.TermsIn f.terms
  | [], _ => Spec.pure fun _ h => h.1
  | c :: rest, high => by
      unfold chain
      refine Spec.keep ((Stable.terms _).and (Stable.below _))
        ((mul_spec _ _).weaken (fun _ h => ⟨h.1, State.TermsIn.glue (h.2 c List.mem_cons_self) 1⟩) fun _ _ h => h)
        fun next => ?_
      exact (chain_spec rest next).weaken (fun _ h => ⟨h.2, fun c' hc' => h.1.2 c' (List.mem_cons_of_mem _ hc')⟩)
        fun _ _ h => h

theorem bits_spec (a : Form F) (count : ℕ) :
    Spec (fun s => s.TermsIn a.terms) (bits a count) fun fs t => ∀ f ∈ fs, t.TermsIn f.terms := by
  unfold bits
  refine Spec.keep (Stable.terms _) ((cells_spec count).weaken (fun _ _ => trivial) fun _ _ h => h)
    fun first => ?_
  have stable : Stable fun t : State F => t.TermsIn a.terms ∧ first + count ≤ t.cells :=
    (Stable.terms _).and (Stable.cells _)
  have cell : ∀ (t : State F) (c : ℕ), t.TermsIn a.terms ∧ first + count ≤ t.cells → first ≤ c → c < first + count →
      t.WireIn (Wire.glue c) :=
    fun t c h low high => show c < t.cells by omega
  refine Spec.seq stable ((booleanRows_spec _).weaken (fun t h c hc => ?_) fun _ _ h => h) fun _ => ?_
  · rw [List.mem_range'_1] at hc
    omega
  refine Spec.seq stable ((row_spec _ _ _).weaken (fun t h => ⟨fun p hp => by simp at hp, ?_⟩) fun _ _ h => h)
    fun _ => ?_
  · refine (rowTerms_terms h.1).append fun x hx => ?_
    obtain ⟨i, hi, rfl⟩ := List.mem_map.mp hx
    rw [List.mem_range] at hi
    exact cell t (first + i) h (by omega) (by omega)
  have outputs : ∀ t : State F, t.TermsIn a.terms ∧ first + count ≤ t.cells →
      ∀ f ∈ (List.range' first count).map (fun c => Form.wire (F := F) (Wire.glue c)), t.TermsIn f.terms := by
    intro t h f hf
    obtain ⟨c, hc, rfl⟩ := List.mem_map.mp hf
    rw [List.mem_range'_1] at hc
    exact State.TermsIn.glue (cell t c h hc.1 hc.2) 1
  dsimp only
  by_cases h64 : count = 64
  · rw [ite_eq_left h64]
    refine Spec.keep stable ((chain_spec _ _).weaken
      (fun t h => ⟨State.TermsIn.glue (cell t _ h (by omega) (by omega)) 1, fun c hc => ?_⟩) fun _ _ h => h)
      fun high => ?_
    · rw [List.mem_range'_1] at hc
      exact cell t c h (by omega) (by omega)
    refine Spec.keep (stable.and (Stable.terms _))
      ((row_spec _ _ _).weaken (fun t h => ⟨fun p hp => ?_, State.TermsIn.nil t⟩) fun _ _ h => h) fun _ => ?_
    · simp only [List.mem_singleton] at hp
      subst hp
      refine ⟨rowTerms_terms h.2, fun x hx => ?_⟩
      obtain ⟨c, hc, rfl⟩ := List.mem_map.mp hx
      rw [List.mem_range'_1] at hc
      exact cell t c h.1 hc.1 (by omega)
    exact Spec.pure fun t h => outputs t h.1.1
  · rw [ite_eq_right h64]
    exact Spec.pure fun t h => outputs t h

theorem tieRows_spec (b : ℕ) (inputs : Fin 16 → Form F) : ∀ ks : List (Fin 16),
    Spec (fun s => b < s.blocks ∧ ∀ k, s.TermsIn (inputs k).terms) (tieRows b inputs ks) fun _ _ => True
  | [] => Spec.pure fun _ _ => trivial
  | k :: rest => by
      unfold tieRows
      refine Spec.seq ((Stable.blocks b).and (Stable.all inputs))
        ((row_spec _ _ _).weaken (fun s h => ⟨fun p hp => by simp at hp, ?_⟩) fun _ _ h => h)
        fun _ => tieRows_spec b inputs rest
      intro x hx
      rcases List.mem_cons.mp hx with rfl | hx
      · exact ⟨h.1, by have := k.isLt; unfold blockCells; omega⟩
      · exact negate_terms (rowTerms_terms (h.2 k)) x hx

theorem permute_spec (inputs : Fin 16 → Form F) :
    Spec (fun s => ∀ k, s.TermsIn (inputs k).terms) (permute inputs) fun f t => ∀ k, t.TermsIn (f k).terms := by
  unfold permute
  refine Spec.keep (Stable.all inputs) (block_spec.weaken (fun _ _ => trivial) fun _ _ h => h) fun b => ?_
  refine Spec.seq ((Stable.all inputs).and (Stable.blocks b))
    ((tieRows_spec b inputs _).weaken (fun _ h => ⟨h.2, h.1⟩) fun _ _ h => h) fun _ => ?_
  refine Spec.pure fun s h k x hx => ?_
  simp only [Form.wire, List.mem_singleton] at hx
  subst hx
  exact ⟨h.2, by have := k.isLt; unfold outputCell blockCells; omega⟩

omit [DecidableEq F] in
theorem push_one {P : State F → Prop} {f : Form F} :
    Spec (fun s => P s ∧ s.TermsIn f.terms) (push [f]) fun _ _ => True :=
  (push_spec _).weaken (fun _ h g hg => by
    simp only [List.mem_singleton] at hg
    subst hg
    exact h.2) fun _ _ h => h

theorem step_spec (stmt : Array F) (i : Instr) : Spec (fun _ => True) (step stmt i) fun _ _ => True := by
  cases i with
  | const value =>
      exact (push_spec _).weaken (fun s _ f hf => by
        simp only [List.mem_singleton] at hf
        subst hf
        exact State.TermsIn.nil s) fun _ _ h => h
  | priv => exact Spec.bind cell_spec fun _ => push_one.weaken (fun _ h => ⟨trivial, h⟩) fun _ _ h => h
  | pub => exact Spec.bind (statementWord_spec stmt) fun _ => push_one.weaken (fun _ h => ⟨trivial, h⟩) fun _ _ h => h
  | linear parts =>
      refine Spec.bind (readValues_spec _) fun forms => Spec.bind ((linear_spec _).weaken ?_ fun _ _ h => h)
        fun _ => push_one.weaken (fun _ h => ⟨trivial, h⟩) fun _ _ h => h
      intro s h p hp
      exact h p.1 (List.of_mem_zip hp).1
  | mul a b =>
      refine Spec.keep (fun _ _ _ _ => trivial) (readValue_spec a) fun x => ?_
      refine Spec.keep (fun _ _ le h => ⟨trivial, h.2.mono le⟩) ((readValue_spec b).weaken (fun _ _ => trivial)
        fun _ _ h => h) fun y => ?_
      exact Spec.bind ((mul_spec x y).weaken (fun _ h => ⟨h.1.2, h.2⟩) fun _ _ h => h) fun _ =>
        push_one.weaken (fun _ h => ⟨trivial, h⟩) fun _ _ h => h
  | extMul a b =>
      refine Spec.keep (fun _ _ _ _ => trivial) (readCoordinates_spec a) fun x => ?_
      refine Spec.keep (fun _ _ le h => ⟨trivial, fun i => (h.2 i).mono le⟩)
        ((readCoordinates_spec b).weaken (fun _ _ => trivial) fun _ _ h => h) fun y => ?_
      refine Spec.bind ((extMul_spec x y).weaken (fun _ h => ⟨h.1.2, h.2⟩) fun _ _ h => h) fun p => ?_
      refine (push_spec _).weaken (fun _ h f hf => ?_) fun _ _ h => h
      simp only [List.mem_cons, List.not_mem_nil, or_false] at hf
      rcases hf with rfl | rfl | rfl <;> exact h _
  | extInv a =>
      refine Spec.bind (readCoordinates_spec a) fun x => ?_
      refine Spec.bind (extInv_spec x) fun p => ?_
      refine (push_spec _).weaken (fun _ h f hf => ?_) fun _ _ h => h
      simp only [List.mem_cons, List.not_mem_nil, or_false] at hf
      rcases hf with rfl | rfl | rfl <;> exact h _
  | assertZero a => exact Spec.bind (readValue_spec a) fun x => assertZero_spec x
  | bits a count =>
      exact Spec.bind (readValue_spec a) fun x => Spec.bind (bits_spec x count) fun fs => push_spec fs
  | permute state =>
      refine Spec.bind (readCoordinates_spec state) fun x => Spec.bind (permute_spec x) fun out => ?_
      refine (push_spec _).weaken (fun _ h f hf => ?_) fun _ _ h => h
      obtain ⟨k, rfl⟩ := List.mem_ofFn.mp hf
      exact h k
  | hint count =>
      refine Spec.bind (cells_spec count) fun first => (push_spec _).weaken (fun s h f hf => ?_) fun _ _ h => h
      obtain ⟨c, hc, rfl⟩ := List.mem_map.mp hf
      rw [List.mem_range'_1] at hc
      exact State.TermsIn.glue (show c < s.cells by omega) 1
  | r1cs rows columns => exact Spec.bind (readValues_spec columns) fun cs => r1cs_spec _ cs

theorem steps_ok (stmt : Array F) : ∀ (L : List Instr) (s : State F), s.Ok → (steps stmt L s).Ok
  | [], _, ok => ok
  | i :: rest, s, ok => steps_ok stmt rest _ (step_spec stmt i s ok trivial).1

end Folding

theorem empty_ok : (({} : State F)).Ok :=
  ⟨fun c hc => by simp at hc, fun f hf => by simp at hf⟩

theorem lower_valid [DecidableEq F] (T : Template F) (cells : blockCells ≤ T.cells) (L : Array Instr)
    (stmt : List F) : (lower L stmt).Valid T := by
  have ok := steps_ok stmt.toArray L.toList {} empty_ok
  unfold lower
  dsimp only
  generalize steps stmt.toArray L.toList {} = s at ok
  intro r hr side t ht
  simp only [Run.rows, List.mem_flatMap] at hr
  obtain ⟨c, hc, hr⟩ := hr
  have wire := ok.1 c (Array.mem_toList_iff.mp hc) r hr side t ht
  revert wire
  cases t.1 with
  | glue c => exact id
  | block b k => exact fun w => ⟨w.1, lt_of_lt_of_le w.2 cells⟩
  | pub i => exact fun w => by simpa [Wire.Valid, State.WireIn] using w

end Lowering

end CompressionLean.Circuit
