import CompressionLean.Circuit.Exec
import CompressionLean.Circuit.Valid
import CompressionLean.Circuit.Rows
import Mathlib.Tactic.LinearCombination

/-!
Owns what the rows of each lowering step say about its forms: the value of a
form at the wire values `v` (`Form.evaluate`), and for each part of the
lowering, the fact that its rows give about its result (`Sound`).

- `Form.evaluate` is `Form::evaluate` (`record.rs:78-88`): the terms, and the
  constant on the one column. Merging terms, dropping zero coefficients and
  moving the constant onto the one column keep the value, so a form's row
  terms have its value (`termsValue_rowTerms`, `combine_value`).
- `State.Holds`: every row of every call holds at `v`, and every block's
  output cells are `perm` of its input cells.
- `Sound v perm m Q`: `m` only adds calls, glue cells and blocks
  (`State.Adds`), and its result has `Q` whenever its end state holds. The
  rows of a part are in its end state, and the end state of a run holds, so
  each part may use its own rows.
-/

namespace CompressionLean.Shrink

variable {F : Type} [CommRing F]

/-- `Form::evaluate` (`record.rs:78-88`): the terms, and the constant on the
one column. -/
def Form.evaluate (v : Wire → F) (f : Form F) : F := termsValue v f.terms + f.constant * v (.pub 0)

@[simp] theorem termsValue_nil (v : Wire → F) : termsValue v ([] : List (Wire × F)) = 0 := rfl

@[simp] theorem termsValue_cons (v : Wire → F) (t : Wire × F) (ts : List (Wire × F)) :
    termsValue v (t :: ts) = t.2 * v t.1 + termsValue v ts := by
  simp [termsValue]

@[simp] theorem termsValue_append (v : Wire → F) (a b : List (Wire × F)) :
    termsValue v (a ++ b) = termsValue v a + termsValue v b := by
  simp [termsValue]

end CompressionLean.Shrink

namespace CompressionLean.Circuit

open CompressionLean.Shrink

section Values

variable {F : Type} [Field F] (v : Wire → F)

@[simp] theorem evaluate_const (c : F) : (Form.const c).evaluate v = c * v (.pub 0) := by
  simp [Form.evaluate, Form.const]

@[simp] theorem evaluate_wire (x : Wire) : (Form.wire x : Form F).evaluate v = v x := by
  simp [Form.evaluate, Form.wire]

@[simp] theorem termsValue_wire (x : Wire) : termsValue v (Form.wire x : Form F).terms = v x := by
  simp [Form.wire]

theorem termsValue_negate (terms : List (Wire × F)) : termsValue v (negate terms) = -termsValue v terms := by
  induction terms with
  | nil => simp [negate]
  | cons t rest ih =>
      simp only [negate, List.map_cons] at ih ⊢
      rw [termsValue_cons, termsValue_cons, ih]
      ring

theorem termsValue_addTerm (t : Wire × F) :
    ∀ acc : List (Wire × F), termsValue v (addTerm t acc) = termsValue v acc + t.2 * v t.1
  | [] => by simp [addTerm]
  | u :: rest => by
      unfold addTerm
      split_ifs with same
      · rw [termsValue_cons, termsValue_cons, ← same]
        ring
      · rw [termsValue_cons, termsValue_cons, termsValue_addTerm t rest]
        ring

theorem termsValue_foldl_addTerm (w : F) :
    ∀ terms acc : List (Wire × F),
      termsValue v (terms.foldl (fun acc t => addTerm (t.1, t.2 * w) acc) acc) =
        termsValue v acc + termsValue v terms * w
  | [], acc => by simp
  | t :: rest, acc => by
      rw [List.foldl_cons, termsValue_foldl_addTerm w rest, termsValue_addTerm, termsValue_cons]
      ring

theorem termsValue_mergeFold :
    ∀ (parts : List (Form F × F)) (acc : List (Wire × F)),
      termsValue v (parts.foldl
          (fun terms p => p.1.terms.foldl (fun terms t => addTerm (t.1, t.2 * p.2) terms) terms) acc) =
        termsValue v acc + (parts.map fun p => termsValue v p.1.terms * p.2).sum
  | [], acc => by simp
  | p :: rest, acc => by
      rw [List.foldl_cons, termsValue_mergeFold rest, termsValue_foldl_addTerm, List.map_cons, List.sum_cons]
      ring

theorem termsValue_expand (side : List (ℕ × F)) (columns : List (Form F)) :
    termsValue v (expand side columns) = (side.map fun t => (columns.getD t.1 ⟨[], 0⟩).evaluate v * t.2).sum := by
  induction side with
  | nil => simp [expand]
  | cons t rest ih =>
      unfold expand at ih ⊢
      rw [List.flatMap_cons, termsValue_append, ih, List.map_cons, List.sum_cons]
      congr 1
      rw [termsValue_append]
      have scaled : termsValue v ((columns.getD t.1 ⟨[], 0⟩).terms.map fun u => (u.1, u.2 * t.2)) =
          termsValue v (columns.getD t.1 ⟨[], 0⟩).terms * t.2 := by
        unfold termsValue
        rw [List.map_map, ← List.sum_map_mul_right]
        congr 1
        apply List.map_congr_left
        intro u _
        simp only [Function.comp_apply]
        ring
      rw [scaled, Form.evaluate]
      split_ifs with zero
      · simp only [termsValue_cons, termsValue_nil]
        ring
      · simp only [ne_eq, not_not] at zero
        rw [termsValue_nil, zero]
        ring

/-- A row call of at most three products holds when its products sum to its
linear side. -/
theorem ofCall_holds {products : List (List (Wire × F) × List (Wire × F))} {linear : List (Wire × F)}
    (three : products.length ≤ 3) (holds : (GlueRow.ofCall products linear).Holds v) :
    (products.map fun p => termsValue v p.1 * termsValue v p.2).sum = termsValue v linear := by
  unfold GlueRow.Holds GlueRow.ofCall at holds
  match products, three with
  | [], _ => simpa using holds
  | [p], _ => simpa using holds
  | [p, q], _ => simpa [add_assoc] using holds
  | [p, q, r], _ => simpa [add_assoc] using holds

variable [DecidableEq F]

theorem termsValue_rowTerms (f : Form F) : termsValue v f.rowTerms = f.evaluate v := by
  unfold Form.rowTerms Form.evaluate
  split_ifs with zero
  · simp [zero]
  · simp

theorem termsValue_filter (terms : List (Wire × F)) :
    termsValue v (terms.filter (·.2 ≠ 0)) = termsValue v terms := by
  induction terms with
  | nil => rfl
  | cons t rest ih =>
      by_cases zero : t.2 = 0
      · rw [List.filter_cons_of_neg (by simp [zero]), ih, termsValue_cons, zero, zero_mul, zero_add]
      · rw [List.filter_cons_of_pos (by simp [zero]), termsValue_cons, termsValue_cons, ih]

/-- `combine` keeps the value: its terms and its constant on the one column are
`Σ weight·value`. -/
theorem combine_value (parts : List (Form F × F)) :
    termsValue v (combine parts).1 + (combine parts).2 * v (.pub 0) =
      (parts.map fun p => p.1.evaluate v * p.2).sum := by
  unfold combine mergeTerms
  dsimp only
  rw [termsValue_filter, termsValue_mergeFold, termsValue_nil, zero_add, ← List.sum_map_mul_right,
    ← List.sum_map_add]
  congr 1
  apply List.map_congr_left
  intro p _
  simp only [Form.evaluate]
  ring

theorem termsValue_sumTerms (forms : List (Form F)) :
    termsValue v (sumTerms forms) = (forms.map fun f => f.evaluate v).sum := by
  have total := combine_value v (forms.map fun f => (f, 1))
  simp only [List.map_map, Function.comp_def, mul_one] at total
  unfold sumTerms
  dsimp only
  rw [termsValue_append, ← total]
  split_ifs with zero
  · simp [zero]
  · simp

end Values

section Logic

variable {F : Type} [Field F] (v : Wire → F) (perm : List F → List F)

/-- Block `b`'s output cells are `perm` of its input cells. -/
def BlockPermutes (b : ℕ) : Prop :=
  List.ofFn (fun k : Fin 16 => v (.block b (outputCell + k))) = perm (List.ofFn fun k : Fin 16 => v (.block b k))

/-- Every row of every call holds at `v`, and every block permutes. -/
def State.Holds (s : State F) : Prop :=
  (∀ c ∈ s.calls, ∀ row ∈ c.rows, row.Holds v) ∧ ∀ b < s.blocks, BlockPermutes v perm b

/-- `t` is `s` with more calls, glue cells and blocks; its values and statement
words are those of `s`. -/
structure State.Adds (s t : State F) : Prop where
  calls : ∀ c ∈ s.calls, c ∈ t.calls
  blocks : s.blocks ≤ t.blocks
  values : t.values = s.values
  publics : t.publics = s.publics

omit [Field F] in
theorem State.Adds.refl (s : State F) : s.Adds s := ⟨fun _ h => h, le_rfl, rfl, rfl⟩

omit [Field F] in
theorem State.Adds.trans {s t u : State F} (h : s.Adds t) (h' : t.Adds u) : s.Adds u :=
  ⟨fun c hc => h'.calls c (h.calls c hc), h.blocks.trans h'.blocks, h'.values.trans h.values,
    h'.publics.trans h.publics⟩

variable {v perm} in
theorem State.Holds.mono {s t : State F} (calls : ∀ c ∈ s.calls, c ∈ t.calls) (blocks : s.blocks ≤ t.blocks)
    (h : t.Holds v perm) : s.Holds v perm :=
  ⟨fun c hc => h.1 c (calls c hc), fun b hb => h.2 b (lt_of_lt_of_le hb blocks)⟩

/-- `m` only adds calls, glue cells and blocks, and its result has `Q`
whenever the rows and blocks of its end state hold. -/
def Sound {α : Type} (m : Lower F α) (Q : α → Prop) : Prop :=
  ∀ s, s.Adds (go m s).2 ∧ ((go m s).2.Holds v perm → Q (go m s).1)

variable {v perm}

namespace Sound

theorem pure {α : Type} {a : α} {Q : α → Prop} (h : Q a) : Sound v perm (Pure.pure a : Lower F α) Q :=
  fun s => ⟨State.Adds.refl s, fun _ => h⟩

theorem bind {α β : Type} {m : Lower F α} {k : α → Lower F β} {Q : α → Prop} {R : β → Prop}
    (hm : Sound v perm m Q) (hk : ∀ a, Sound v perm (k a) fun b => Q a → R b) : Sound v perm (m >>= k) R := by
  intro s
  obtain ⟨adds, q⟩ := hm s
  obtain ⟨adds', r⟩ := hk (go m s).1 (go m s).2
  rw [go_bind]
  exact ⟨adds.trans adds', fun holds => r holds (q (holds.mono adds'.calls adds'.blocks))⟩

theorem weaken {α : Type} {m : Lower F α} {Q Q' : α → Prop} (h : Sound v perm m Q) (hq : ∀ a, Q a → Q' a) :
    Sound v perm m Q' :=
  fun s => ⟨(h s).1, fun holds => hq _ ((h s).2 holds)⟩

end Sound

theorem cell_sound : Sound v perm (cell : Lower F (Form F)) fun f => ∃ c, f = .wire (.glue c) :=
  fun s => ⟨⟨fun _ h => h, le_rfl, rfl, rfl⟩, fun _ => ⟨s.cells, rfl⟩⟩

theorem cells_sound (count : ℕ) : Sound v perm (cells count : Lower F ℕ) fun _ => True :=
  fun _ => ⟨⟨fun _ h => h, le_rfl, rfl, rfl⟩, fun _ => trivial⟩

theorem row_sound (products : List (List (Wire × F) × List (Wire × F))) (linear : List (Wire × F))
    (three : products.length ≤ 3) :
    Sound v perm (row products linear three) fun _ => (GlueRow.ofCall products linear).Holds v := by
  intro s
  refine ⟨⟨fun c hc => Array.mem_push.mpr (Or.inl hc), le_rfl, rfl, rfl⟩, fun holds => ?_⟩
  exact holds.1 (.row products linear three) (Array.mem_push.mpr (Or.inr rfl)) _ (by simp [Call.rows])

theorem r1cs_sound (rows : List (R1csRow F)) (columns : List (Form F)) :
    Sound v perm (r1cs rows columns) fun _ => ∀ r ∈ rows,
      (GlueRow.ofCall [(expand r.a columns, expand r.b columns)] (expand r.c columns)).Holds v := by
  intro s
  refine ⟨⟨fun c hc => Array.mem_push.mpr (Or.inl hc), le_rfl, rfl, rfl⟩, fun holds r hr => ?_⟩
  exact holds.1 (.r1cs rows columns) (Array.mem_push.mpr (Or.inr rfl)) _ (List.mem_map.mpr ⟨r, hr, rfl⟩)

private theorem block_sound : Sound v perm (block : Lower F ℕ) fun b => BlockPermutes v perm b :=
  fun s => ⟨⟨fun _ h => h, Nat.le_succ _, rfl, rfl⟩, fun holds => holds.2 s.blocks (Nat.lt_succ_self _)⟩

end Logic


section Parts

variable {F : Type} [Field F] [DecidableEq F] {v : Wire → F} {perm : List F → List F}

omit [Field F] [DecidableEq F] in
private theorem terms_of_isConstant {f : Form F} (h : f.isConstant = true) : f.terms = [] :=
  List.isEmpty_iff.mp h

omit [Field F] [DecidableEq F] in
private theorem terms_of_constants {a : Fin 3 → Form F} (h : constants a = true) : ∀ i, (a i).terms = [] := by
  simp only [constants, Bool.and_eq_true] at h
  intro i
  fin_cases i
  exacts [terms_of_isConstant h.1.1, terms_of_isConstant h.1.2, terms_of_isConstant h.2]

theorem form_sound (terms : List (Wire × F)) (constant : F) :
    Sound v perm (form terms constant) fun f => f.evaluate v = termsValue v terms + constant * v (.pub 0) := by
  unfold form
  by_cases short : terms.length ≤ 4
  · rw [if_pos short]
    exact Sound.pure rfl
  rw [if_neg short]
  refine cell_sound.bind fun c => (row_sound _ _ _).bind fun _ => Sound.pure fun holds isCell => ?_
  obtain ⟨c, rfl⟩ := isCell
  have row := ofCall_holds v (by simp) holds
  simp only [List.map_nil, List.sum_nil, termsValue_append, termsValue_wire, termsValue_negate] at row
  rw [evaluate_wire]
  split_ifs at row with zero
  · simp only [termsValue_nil] at row
    rw [zero]
    linear_combination -row
  · simp only [termsValue_cons, termsValue_nil] at row
    linear_combination -row

theorem linear_sound (parts : List (Form F × F)) :
    Sound v perm (linear parts) fun f => f.evaluate v = (parts.map fun p => p.1.evaluate v * p.2).sum :=
  (form_sound _ _).weaken fun _ h => h.trans (combine_value v parts)

theorem mul_sound (one : v (.pub 0) = 1) (a b : Form F) :
    Sound v perm (mul a b) fun f => f.evaluate v = a.evaluate v * b.evaluate v := by
  unfold mul
  by_cases ca : a.isConstant = true
  · rw [if_pos ca]
    refine (linear_sound _).weaken fun f h => ?_
    rw [h]
    simp [Form.evaluate, terms_of_isConstant ca, one]
    ring
  rw [if_neg ca]
  by_cases cb : b.isConstant = true
  · rw [if_pos cb]
    refine (linear_sound _).weaken fun f h => ?_
    rw [h]
    simp [Form.evaluate, terms_of_isConstant cb, one]
  rw [if_neg cb]
  refine cell_sound.bind fun p => (row_sound _ _ _).bind fun _ => Sound.pure fun holds isCell => ?_
  obtain ⟨c, rfl⟩ := isCell
  have row := ofCall_holds v (by simp) holds
  simp only [List.map_cons, List.map_nil, List.sum_cons, List.sum_nil, add_zero, termsValue_rowTerms,
    termsValue_wire] at row
  rw [evaluate_wire, ← row]

/-- The products of row `k` of `a·b` sum to coordinate `k` of the product. -/
theorem extRow_value (a b : Fin 3 → Form F) (k : Fin 3) :
    ((extRow a b k).map fun p => termsValue v p.1 * termsValue v p.2).sum =
      extProduct (fun i => (a i).evaluate v) (fun i => (b i).evaluate v) k := by
  fin_cases k <;> simp [extRow, extProduct, termsValue_rowTerms, termsValue_sumTerms] <;> ring

theorem extCell_sound (a b : Fin 3 → Form F) (k : Fin 3) :
    Sound v perm (extCell a b k) fun f =>
      f.evaluate v = extProduct (fun i => (a i).evaluate v) (fun i => (b i).evaluate v) k := by
  unfold extCell
  refine cell_sound.bind fun c => (row_sound _ _ _).bind fun _ => Sound.pure fun holds isCell => ?_
  obtain ⟨c, rfl⟩ := isCell
  have row := ofCall_holds v (extRow_length a b k) holds
  rw [extRow_value, termsValue_wire] at row
  rw [evaluate_wire, row]

omit [DecidableEq F] in
theorem fromFn_sound {m : Fin 3 → Lower F (Form F)} {P : Fin 3 → Form F → Prop}
    (hm : ∀ k, Sound v perm (m k) (P k)) : Sound v perm (fromFn m) fun fs => ∀ k, P k (fs k) := by
  unfold fromFn
  refine (hm 0).bind fun x => (hm 1).bind fun y => (hm 2).bind fun z => Sound.pure fun pz py px k => ?_
  fin_cases k
  exacts [px, py, pz]

theorem extMul_sound (one : v (.pub 0) = 1) (a b : Fin 3 → Form F) :
    Sound v perm (extMul a b) fun fs =>
      ∀ k, (fs k).evaluate v = extProduct (fun i => (a i).evaluate v) (fun i => (b i).evaluate v) k := by
  unfold extMul
  by_cases hc : (constants a || constants b) = true
  · rw [if_pos hc]
    dsimp only
    refine fromFn_sound
      (P := fun k f => f.evaluate v = extProduct (fun i => (a i).evaluate v) (fun i => (b i).evaluate v) k)
      fun k => (linear_sound _).weaken fun f h => ?_
    rw [h]
    by_cases ca : constants a = true
    · have consts := terms_of_constants ca
      have value : ∀ i, (a i).evaluate v = (a i).constant := fun i => by simp [Form.evaluate, consts i, one]
      simp only [ca, if_true, value]
      fin_cases k <;> simp [extProduct] <;> ring
    · have cb : constants b = true := by simpa [ca] using hc
      have consts := terms_of_constants cb
      have value : ∀ i, (b i).evaluate v = (b i).constant := fun i => by simp [Form.evaluate, consts i, one]
      simp only [ca, Bool.false_eq_true, if_false, value]
      fin_cases k <;> simp [extProduct] <;> ring
  · rw [if_neg hc]
    exact fromFn_sound (extCell_sound a b)

/-- Cramer's rule: the inverse that `extInverse` finds is an inverse. -/
theorem extInverse_mul {a u : Fin 3 → F} (h : extInverse a = some u) : extProduct a u = ![1, 0, 0] := by
  unfold extInverse at h
  dsimp only at h
  split_ifs at h with det
  simp only [Option.some.injEq] at h
  subst h
  generalize hd : a 0 * ((a 0 + a 2) * (a 0 + a 2) - a 1 * (a 1 + a 2)) +
      a 2 * (a 2 * (a 1 + a 2) - a 1 * (a 0 + a 2)) + a 1 * (a 1 * a 1 - a 2 * (a 0 + a 2)) = d at det ⊢
  funext k
  fin_cases k <;> simp [extProduct] <;> field_simp
  · linear_combination hd
  · ring
  · ring

theorem extInv_sound (one : v (.pub 0) = 1) (a : Fin 3 → Form F) :
    Sound v perm (extInv a) fun fs =>
      extProduct (fun i => (a i).evaluate v) (fun k => (fs k).evaluate v) = ![1, 0, 0] := by
  unfold extInv
  split
  · rename_i u hu
    refine Sound.pure ?_
    split_ifs at hu with ca
    have consts := terms_of_constants ca
    have inverse := extInverse_mul hu
    have av : (fun i => (a i).evaluate v) = fun i => (a i).constant := by
      funext i
      simp [Form.evaluate, consts i, one]
    have uv : (fun k => (Form.const (u k)).evaluate v) = u := by
      funext k
      simp [one]
    rw [av, uv, inverse]
  · refine (fromFn_sound fun _ => cell_sound).bind fun inverse => ?_
    refine (row_sound _ _ _).bind fun _ => (row_sound _ _ _).bind fun _ => (row_sound _ _ _).bind fun _ =>
      Sound.pure fun h2 h1 h0 _ => ?_
    have e0 := ofCall_holds v (extRow_length _ _ _) h0
    have e1 := ofCall_holds v (extRow_length _ _ _) h1
    have e2 := ofCall_holds v (extRow_length _ _ _) h2
    rw [extRow_value] at e0 e1 e2
    funext k
    fin_cases k
    · simpa [one] using e0
    · simpa using e1
    · simpa using e2

theorem assertZero_sound (a : Form F) : Sound v perm (assertZero a) fun _ => a.evaluate v = 0 := by
  unfold assertZero
  split_ifs with zero
  · exact Sound.pure (by simp [Form.evaluate, zero.2, terms_of_isConstant zero.1])
  · exact (row_sound _ _ _).weaken fun _ h => by simpa [termsValue_rowTerms] using (ofCall_holds v (by simp) h).symm

omit [DecidableEq F] in
/-- A field element equal to its square is 0 or 1. -/
private theorem boolean_of_sq {x : F} (h : x * x = x) : x = 0 ∨ x = 1 := by
  have : x * (x - 1) = 0 := by rw [mul_sub, h, mul_one, sub_self]
  rcases mul_eq_zero.mp this with h0 | h1
  · exact Or.inl h0
  · exact Or.inr (sub_eq_zero.mp h1)

omit [DecidableEq F] in
theorem booleanRows_sound :
    ∀ cs : List ℕ,
      Sound v perm (booleanRows cs : Lower F Unit) fun _ => ∀ c ∈ cs, v (.glue c) = 0 ∨ v (.glue c) = 1
  | [] => Sound.pure (by simp)
  | c :: rest => by
      unfold booleanRows
      refine (row_sound _ _ _).bind fun _ => (booleanRows_sound rest).weaken fun _ later here => ?_
      have row := ofCall_holds v (by simp) here
      simp only [List.map_cons, List.map_nil, List.sum_cons, List.sum_nil, add_zero, termsValue_cons,
        termsValue_nil, one_mul] at row
      intro c' member
      rcases List.mem_cons.mp member with rfl | member
      · exact boolean_of_sq row
      · exact later c' member

theorem chain_sound (one : v (.pub 0) = 1) :
    ∀ (cs : List ℕ) (high : Form F),
      Sound v perm (chain high cs) fun f => f.evaluate v = high.evaluate v * (cs.map fun c => v (.glue c)).prod
  | [], high => Sound.pure (by simp)
  | c :: rest, high => by
      unfold chain
      refine (mul_sound one _ _).bind fun next => (chain_sound one rest next).weaken fun f hf hnext => ?_
      rw [hf, hnext, evaluate_wire, List.map_cons, List.prod_cons]
      ring

/-- Booleans that sum to zero are zero, when no count up to `n` is zero in `F`. -/
private theorem booleans_zero {n : ℕ} (small : ∀ k : ℕ, k ≤ n → (k : F) = 0 → k = 0) :
    ∀ l : List F, l.length ≤ n → (∀ x ∈ l, x = 0 ∨ x = 1) → l.sum = 0 → ∀ x ∈ l, x = 0 := by
  have count : ∀ l : List F, (∀ x ∈ l, x = 0 ∨ x = 1) → l.sum = ((l.filter (· = 1)).length : F) := by
    intro l
    induction l with
    | nil => simp
    | cons x rest ih =>
        intro h
        have rest_sum := ih fun y hy => h y (List.mem_cons_of_mem _ hy)
        rcases h x List.mem_cons_self with rfl | rfl
        · simp [rest_sum]
        · rw [List.sum_cons, rest_sum, List.filter_cons_of_pos (by simp), List.length_cons, Nat.cast_succ]
          ring
  intro l len boolean sum x member
  have none : (l.filter (· = 1)).length = 0 := by
    apply small _ ((List.length_filter_le _ _).trans len)
    rw [← count l boolean, sum]
  rcases boolean x member with zero | one
  · exact zero
  · have : x ∈ l.filter (· = 1) := List.mem_filter.mpr ⟨member, by simp [one]⟩
    rw [List.length_eq_zero_iff.mp none] at this
    simp at this

theorem bits_sound (one : v (.pub 0) = 1) (small : ∀ k : ℕ, k ≤ 32 → (k : F) = 0 → k = 0) (a : Form F)
    (count : ℕ) :
    Sound v perm (bits a count) fun fs =>
      ∃ first, fs = (List.range' first count).map (fun c => .wire (.glue c)) ∧
        Bits (a.evaluate v) count fun i => v (.glue (first + i)) := by
  unfold bits
  refine (cells_sound count).bind fun first => (booleanRows_sound _).bind fun _ => (row_sound _ _ _).bind fun _ => ?_
  have finish : ∀ (canonical : count = 64 → (∀ i, 32 ≤ i → i < 64 → v (.glue (first + i)) = 1) →
        ∀ i < 32, v (.glue (first + i)) = 0),
      (GlueRow.ofCall [] (a.rowTerms ++
        (List.range count).map fun i => (.glue (first + i), -((2 : F) ^ i)))).Holds v →
      (∀ c ∈ List.range' first count, v (.glue c) = 0 ∨ v (.glue c) = 1) → True →
      ∃ first', (List.range' first count).map (fun c => (Form.wire (.glue c) : Form F)) =
          (List.range' first' count).map (fun c => (Form.wire (.glue c) : Form F)) ∧
        Bits (a.evaluate v) count fun i => v (.glue (first' + i)) := by
    intro canonical sumRow boolean _
    refine ⟨first, rfl, ⟨fun i hi => boolean _ (List.mem_range'_1.mpr ⟨by omega, by omega⟩), ?_, canonical⟩⟩
    have row := ofCall_holds v (by simp) sumRow
    simp only [List.map_nil, List.sum_nil, termsValue_append, termsValue_rowTerms] at row
    unfold termsValue at row
    rw [List.map_map] at row
    have neg : (List.map ((fun t : Wire × F => t.2 * v t.1) ∘ fun i => (Wire.glue (first + i), -((2 : F) ^ i)))
        (List.range count)).sum = ((List.range count).map fun i => 2 ^ i * v (.glue (first + i))).sum * (-1) := by
      rw [← List.sum_map_mul_right]
      congr 1
      apply List.map_congr_left
      intro i _
      simp only [Function.comp_apply]
      ring
    rw [neg] at row
    linear_combination -row
  split_ifs with h64
  · refine (chain_sound one _ _).bind fun high => (row_sound _ _ _).bind fun _ => Sound.pure ?_
    intro canonicalRow highValue sumRow boolean cellsFact
    refine finish (fun _ allOne i hi => ?_) sumRow boolean cellsFact
    have row := ofCall_holds v (by simp) canonicalRow
    simp only [List.map_cons, List.map_nil, List.sum_cons, List.sum_nil, add_zero, termsValue_rowTerms,
      termsValue_nil] at row
    have highOne : high.evaluate v = 1 := by
      rw [highValue, evaluate_wire, allOne 32 le_rfl (by norm_num), one_mul]
      apply List.prod_eq_one
      intro x hx
      obtain ⟨c, hc, rfl⟩ := List.mem_map.mp hx
      rw [List.mem_range'_1] at hc
      have := allOne (c - first) (by omega) (by omega)
      rwa [show first + (c - first) = c by omega] at this
    rw [highOne, one_mul] at row
    have low := booleans_zero small ((List.range' first 32).map fun c => v (.glue c)) (by simp)
      (fun x hx => by
        obtain ⟨c, hc, rfl⟩ := List.mem_map.mp hx
        rw [List.mem_range'_1] at hc
        exact boolean c (List.mem_range'_1.mpr ⟨hc.1, by omega⟩))
      (by
        rw [← row]
        unfold termsValue
        rw [List.map_map]
        congr 1
        apply List.map_congr_left
        intro c _
        simp)
    exact low _ (List.mem_map.mpr ⟨first + i, List.mem_range'_1.mpr ⟨by omega, by omega⟩, rfl⟩)
  · exact Sound.pure fun sumRow boolean cellsFact =>
      finish (fun h => absurd h h64) sumRow boolean cellsFact

theorem tieRows_sound (b : ℕ) (inputs : Fin 16 → Form F) :
    ∀ ks : List (Fin 16),
      Sound v perm (tieRows b inputs ks) fun _ => ∀ k ∈ ks, v (.block b k) = (inputs k).evaluate v
  | [] => Sound.pure (by simp)
  | k :: rest => by
      unfold tieRows
      refine (row_sound _ _ _).bind fun _ => (tieRows_sound b inputs rest).weaken fun _ later here => ?_
      have row := ofCall_holds v (by simp) here
      simp only [List.map_nil, List.sum_nil, termsValue_cons, termsValue_negate, termsValue_rowTerms,
        one_mul] at row
      intro k' member
      rcases List.mem_cons.mp member with rfl | member
      · linear_combination -row
      · exact later k' member

theorem permute_sound (inputs : Fin 16 → Form F) :
    Sound v perm (permute inputs) fun fs =>
      List.ofFn (fun k => (fs k).evaluate v) = perm (List.ofFn fun k => (inputs k).evaluate v) := by
  unfold permute
  refine block_sound.bind fun b => (tieRows_sound b inputs _).bind fun _ => Sound.pure fun ties permutes => ?_
  have outputs : (fun k : Fin 16 => (Form.wire (.block b (outputCell + k)) : Form F).evaluate v) =
      fun k : Fin 16 => v (.block b (outputCell + k)) := by
    funext k
    simp
  have tied : (fun k : Fin 16 => (inputs k).evaluate v) = fun k : Fin 16 => v (.block b k) := by
    funext k
    exact (ties k (List.mem_finRange k)).symm
  rw [outputs, tied]
  exact permutes

end Parts

end CompressionLean.Circuit
