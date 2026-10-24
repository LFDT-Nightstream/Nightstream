import CompressionLean.Final.Bridge
import CompressionLean.Spec.PoseidonRows

/-!
Owns the proof that the fast evaluator of the rows of a Poseidon2 block
(`Spec/PoseidonRows.lean`) gives F′'s coefficients.

- `invocationRows?_spec`: the rows of `invocationRows?` have the coefficients
  of F′'s `Poseidon.Block.row?`.
- `matrixSource_entry`: for a Poseidon2 block of an application's matrix
  program, they are the entries of F′'s completed CCS matrices.

Does not own a check of Rust's runs against these rows (`RunsGive`).
-/

namespace CompressionLean.Checks

open CompressionLean.Spec CompressionLean.Spec.PoseidonRows
open CompressionLean.Final
open NightstreamFPrime NightstreamFPrime.Spec NightstreamFPrime.Export.Stage1
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Layout NightstreamFPrime.Layout.ProductionRelation NightstreamFPrime.Layout.MatrixProgram
open NightstreamFPrime.Gadgets.Poseidon2
open Fin.CommRing

/-! ## Coefficients of entry lists -/

theorem coefficient_cons {n : Nat} (e : SparseEntry n) (es : Entries n) (c : Fin n) :
    (⟨e :: es⟩ : SparseForm n).coefficient c =
      (if e.column = c then e.coefficient else 0) + (⟨es⟩ : SparseForm n).coefficient c := by
  have split : (⟨e :: es⟩ : SparseForm n) =
      SparseForm.add (SparseForm.singleton e.column e.coefficient) ⟨es⟩ := rfl
  rw [split, SparseForm.add_coefficient, SparseForm.singleton_coefficient]

theorem coefficient_append {n : Nat} (xs ys : Entries n) (c : Fin n) :
    (⟨xs ++ ys⟩ : SparseForm n).coefficient c =
      (⟨xs⟩ : SparseForm n).coefficient c + (⟨ys⟩ : SparseForm n).coefficient c :=
  SparseForm.add_coefficient ⟨xs⟩ ⟨ys⟩ c

theorem coefficient_eq_sum {n : Nat} (xs : Entries n) (c : Fin n) :
    (⟨xs⟩ : SparseForm n).coefficient c = (xs.map fun e => if e.column = c then e.coefficient else 0).sum := by
  induction xs with
  | nil => rfl
  | cons e es ih => rw [coefficient_cons, ih, List.map_cons, List.sum_cons]

theorem merge_coefficient {n : Nat} (xs ys : Entries n) (c : Fin n) :
    (⟨merge xs ys⟩ : SparseForm n).coefficient c =
      (⟨xs⟩ : SparseForm n).coefficient c + (⟨ys⟩ : SparseForm n).coefficient c := by
  induction xs, ys using merge.induct with
  | case1 ys => rw [merge]; simp [SparseForm.coefficient]
  | case2 x xs => rw [merge]; simp [SparseForm.coefficient]
  | case3 x xs y ys lt ih =>
      rw [merge, if_pos lt, coefficient_cons, ih, coefficient_cons x]
      ring
  | case4 x xs y ys nlt gt ih =>
      rw [merge, if_neg nlt, if_pos gt, coefficient_cons, ih, coefficient_cons y]
      ring
  | case5 x xs y ys nlt ngt zero ih =>
      have same : x.column = y.column := Fin.ext (by omega)
      rw [merge, if_neg nlt, if_neg ngt, if_pos zero, ih, coefficient_cons, coefficient_cons y, ← same]
      by_cases hc : x.column = c
      · rw [if_pos hc, if_pos hc]
        linear_combination -zero
      · rw [if_neg hc, if_neg hc]
        ring
  | case6 x xs y ys nlt ngt nonzero ih =>
      have same : x.column = y.column := Fin.ext (by omega)
      rw [merge, if_neg nlt, if_neg ngt, if_neg nonzero, coefficient_cons, ih, coefficient_cons,
        coefficient_cons y, ← same]
      by_cases hc : x.column = c
      · rw [if_pos hc, if_pos hc, if_pos hc]
        ring
      · rw [if_neg hc, if_neg hc, if_neg hc]
        ring

theorem combine_coefficient {n : Nat} (xs : Entries n) (c : Fin n) :
    (⟨combine xs⟩ : SparseForm n).coefficient c = (⟨xs⟩ : SparseForm n).coefficient c := by
  induction xs with
  | nil => rfl
  | cons e rest ih =>
      rw [coefficient_cons, ← ih, combine.eq_2]
      generalize combine rest = r
      cases r with
      | nil =>
          by_cases zero : e.coefficient = 0
          · simp [zero, SparseForm.coefficient]
          · simp only [if_neg zero, coefficient_cons]
      | cons h t =>
          simp only
          by_cases same : h.column = e.column
          · rw [if_pos same, coefficient_cons h, same]
            by_cases zero : e.coefficient + h.coefficient = 0
            · rw [if_pos zero]
              by_cases hc : e.column = c
              · rw [if_pos hc, if_pos hc, ← add_assoc, zero, zero_add]
              · rw [if_neg hc, if_neg hc, zero_add, zero_add]
            · rw [if_neg zero, coefficient_cons]
              by_cases hc : e.column = c
              · rw [if_pos hc, if_pos hc, if_pos hc, add_assoc]
              · rw [if_neg hc, if_neg hc, if_neg hc, zero_add, zero_add]
          · rw [if_neg same]
            by_cases zero : e.coefficient = 0
            · rw [if_pos zero, zero, ite_self, zero_add]
            · rw [if_neg zero, coefficient_cons]

theorem normalize_coefficient {n : Nat} (form : SparseForm n) (c : Fin n) :
    (⟨normalize form⟩ : SparseForm n).coefficient c = form.coefficient c := by
  rw [PoseidonRows.normalize, combine_coefficient, coefficient_eq_sum, ((List.mergeSort_perm _ _).map _).sum_eq,
    ← coefficient_eq_sum]

theorem scaled_coefficient {n : Nat} (k : F) (xs : Entries n) (c : Fin n) :
    (⟨scaled k xs⟩ : SparseForm n).coefficient c = k * (⟨xs⟩ : SparseForm n).coefficient c :=
  SparseForm.scale_coefficient k ⟨xs⟩ c

/-! ## The value of a combination -/

theorem Sym.coef_add (s t : Sym) (a : Nat) : Sym.coef (Sym.add s t) a = Sym.coef s a + Sym.coef t a := by
  unfold Sym.coef Sym.add
  split <;> simp_all

theorem Sym.coef_scale (k : F) (s : Sym) (a : Nat) : Sym.coef (Sym.scale k s) a = k * Sym.coef s a := by
  unfold Sym.coef Sym.scale
  split <;> simp_all

theorem Sym.coef_zero (a : Nat) : Sym.coef Sym.zero a = 0 := by
  unfold Sym.coef Sym.zero
  split <;> simp_all

theorem Sym.coef_atom (a b : Nat) : Sym.coef (Sym.atom a) b = if b < 167 ∧ b = a then 1 else 0 := by
  unfold Sym.coef Sym.atom
  split <;> simp_all

/-- Coefficient `c` of the combination `v` of the forms `atom`. -/
def eval {n : Nat} (atom : Nat → SparseForm n) (v : Sym) (c : Fin n) : F :=
  ∑ a ∈ Finset.range 167, Sym.coef v a * (atom a).coefficient c

section eval

variable {n : Nat} (atom : Nat → SparseForm n) (c : Fin n)

theorem eval_add (s t : Sym) : eval atom (Sym.add s t) c = eval atom s c + eval atom t c := by
  simp only [eval, Sym.coef_add, add_mul, Finset.sum_add_distrib]

theorem eval_scale (k : F) (s : Sym) : eval atom (Sym.scale k s) c = k * eval atom s c := by
  simp only [eval, Sym.coef_scale, Finset.mul_sum, mul_assoc]

theorem eval_zero : eval atom Sym.zero c = 0 := by
  simp only [eval, Sym.coef_zero, zero_mul, Finset.sum_const_zero]

theorem eval_atom {a : Nat} (inside : a < 167) : eval atom (Sym.atom a) c = (atom a).coefficient c := by
  unfold eval
  rw [Finset.sum_eq_single a]
  · rw [Sym.coef_atom, if_pos ⟨inside, rfl⟩, one_mul]
  · intro b _ other
    rw [Sym.coef_atom, if_neg fun h => other h.2, zero_mul]
  · intro outside
    exact absurd (Finset.mem_range.mpr inside) outside

theorem eval_congr {other : Nat → SparseForm n}
    (same : ∀ a, a < 167 → (atom a).coefficient c = (other a).coefficient c) (v : Sym) :
    eval atom v c = eval other v c :=
  Finset.sum_congr rfl fun a member => by rw [same a (Finset.mem_range.mp member)]

end eval

theorem sum_range_list {M : Type} [AddCommMonoid M] (f : Nat → M) :
    ∀ m, ((List.range m).map f).sum = ∑ a ∈ Finset.range m, f a
  | 0 => rfl
  | m + 1 => by
      rw [List.range_succ, List.map_append, List.sum_append, sum_range_list f m, Finset.sum_range_succ]
      simp

theorem foldl_merge_coefficient {n : Nat} (g : Nat → F) (forms : Nat → Entries n) (c : Fin n) :
    ∀ (l : List Nat) (acc : Entries n),
      (⟨l.foldl (fun acc a => if g a = 0 then acc else merge acc (scaled (g a) (forms a))) acc⟩ :
        SparseForm n).coefficient c =
        (⟨acc⟩ : SparseForm n).coefficient c + (l.map fun a => g a * (⟨forms a⟩ : SparseForm n).coefficient c).sum
  | [], acc => by simp
  | a :: l, acc => by
      rw [List.foldl_cons, foldl_merge_coefficient g forms c l, List.map_cons, List.sum_cons]
      by_cases zero : g a = 0
      · rw [if_pos zero, zero, zero_mul, zero_add]
      · rw [if_neg zero, merge_coefficient, scaled_coefficient, add_assoc]

theorem flatMap_coefficient {n : Nat} (f : Nat → Entries n) (c : Fin n) :
    ∀ l : List Nat, (⟨l.flatMap f⟩ : SparseForm n).coefficient c =
      (l.map fun a => (⟨f a⟩ : SparseForm n).coefficient c).sum
  | [] => rfl
  | a :: l => by rw [List.flatMap_cons, coefficient_append, flatMap_coefficient f c l, List.map_cons, List.sum_cons]

theorem sum_range_atoms (f : Nat → F) :
    ∑ a ∈ Finset.range 167, f a = ∑ a ∈ Finset.range 16, f a + (∑ k ∈ Finset.range 150, f (16 + k) + f 166) := by
  rw [show (167 : Nat) = 16 + (150 + 1) from rfl, Finset.sum_range_add, Finset.sum_range_succ _ 150]

/-- **The expansion.** The form of `atoms.expand v` is the combination `v` of
the atom forms. -/
theorem expand_coefficient {n : Nat} (atoms : Atoms n) (v : Sym) (c : Fin n) :
    (⟨atoms.expand v⟩ : SparseForm n).coefficient c = eval atoms.form v c := by
  have inputs := foldl_merge_coefficient (Sym.coef v) (fun l => atoms.inputs[l]?.getD []) c (List.range 16) []
  have sboxes := flatMap_coefficient (fun k => if Sym.coef v (16 + k) = 0 then []
    else scaled (Sym.coef v (16 + k)) (atoms.sbox[k]?.getD [])) c (List.range 150)
  have one : (⟨if Sym.coef v 166 = 0 then [] else [⟨atoms.one, Sym.coef v 166⟩]⟩ : SparseForm n).coefficient c =
      Sym.coef v 166 * (atoms.form 166).coefficient c := by
    rw [Atoms.form, if_neg (show ¬166 < 16 by omega), if_neg (show ¬166 < 166 by omega),
      SparseForm.singleton_coefficient]
    by_cases zero : Sym.coef v 166 = 0
    · rw [if_pos zero, zero, zero_mul]
      rfl
    · rw [if_neg zero, coefficient_cons]
      by_cases same : atoms.one = c <;> simp [same, SparseForm.coefficient]
  have parts : ∀ k ∈ Finset.range 150, (⟨if Sym.coef v (16 + k) = 0 then []
      else scaled (Sym.coef v (16 + k)) (atoms.sbox[k]?.getD [])⟩ : SparseForm n).coefficient c =
      Sym.coef v (16 + k) * (atoms.form (16 + k)).coefficient c := by
    intro k member
    have small := Finset.mem_range.mp member
    rw [Atoms.form, if_neg (show ¬16 + k < 16 by omega), if_pos (show 16 + k < 166 by omega),
      show 16 + k - 16 = k by omega]
    by_cases zero : Sym.coef v (16 + k) = 0
    · rw [if_pos zero, zero, zero_mul]
      rfl
    · rw [if_neg zero, scaled_coefficient]
  have lanes : ∀ l ∈ Finset.range 16, Sym.coef v l * (⟨atoms.inputs[l]?.getD []⟩ : SparseForm n).coefficient c =
      Sym.coef v l * (atoms.form l).coefficient c := by
    intro l member
    rw [Atoms.form, if_pos (Finset.mem_range.mp member)]
  simp only [Atoms.expand, merge_coefficient, inputs, sboxes, one, sum_range_list]
  rw [eval, sum_range_atoms, Finset.sum_congr rfl parts, Finset.sum_congr rfl lanes]
  simp [SparseForm.coefficient]

/-! ## The trace over the atoms gives F′'s trace -/

section trace

variable {n : Nat} (I : PoseidonSboxPlan.Interface n)

/-- The atom forms of an interface: its input lanes, its S-box outputs and its
one column. -/
def atomOf (a : Nat) : SparseForm n :=
  if h : a < 16 then I.input ⟨a, h⟩
  else if a < 166 then PoseidonSboxPlan.sboxOutputAt I (a - 16)
  else SparseForm.singleton I.oneColumn 1

/-- `f` has the coefficients of the combination `s` of the atoms of `I`. -/
def Rep (f : SparseForm n) (s : Sym) : Prop := ∀ c, f.coefficient c = eval (atomOf I) s c

/-- Each lane of `S` has the coefficients of that lane of `S'`. -/
def StateRep (S : PoseidonSboxPlan.State n) (S' : State) : Prop := ∀ lane : Fin 16, Rep I (S lane) S'[lane]

/-- An S-box row of F′ and a row over the atoms that gives it. -/
def RowRep (row : SboxRow.Forms n) (r : Row) : Prop :=
  Rep I row.input r.input ∧ row.output = PoseidonSboxPlan.sboxOutputAt I r.output

/-- A step of F′'s trace and the same step over the atoms. -/
structure StepRep (ref : PoseidonSboxPlan.StepResult n) (sym : StepResult) : Prop where
  nextSbox : ref.nextSbox = sym.nextSbox
  state : StateRep I ref.state sym.state
  rows : List.Forall₂ (RowRep I) ref.rows sym.rows

variable {I}

theorem rep_add {f g : SparseForm n} {s t : Sym} (hf : Rep I f s) (hg : Rep I g t) :
    Rep I (SparseForm.add f g) (Sym.add s t) := fun c => by
  rw [SparseForm.add_coefficient, hf, hg, eval_add]

theorem rep_scale {f : SparseForm n} {s : Sym} (k : F) (hf : Rep I f s) :
    Rep I (SparseForm.scale k f) (Sym.scale k s) := fun c => by
  rw [SparseForm.scale_coefficient, hf, eval_scale]

theorem rep_empty : Rep I SparseForm.empty Sym.zero := fun c => by
  rw [SparseForm.empty_coefficient, eval_zero]

theorem rep_get {S : PoseidonSboxPlan.State n} {S' : State} (h : StateRep I S S') (index : Nat) :
    Rep I (SparseLayer.get S index) (get S' index) := by
  unfold SparseLayer.get PoseidonRows.get
  split
  · exact h _
  · exact rep_empty

theorem rep_addConstant {f : SparseForm n} {s : Sym} (hf : Rep I f s) (value : F) :
    Rep I (SparseLayer.addConstant I.oneColumn f value) (addConstant s value) := by
  refine rep_add hf fun c => ?_
  have one : atomOf I 166 = SparseForm.singleton I.oneColumn 1 := by
    rw [atomOf, dif_neg (show ¬166 < 16 by decide), if_neg (show ¬166 < 166 by decide)]
  rw [SparseLayer.constant, eval_scale, eval_atom _ _ (by decide : 166 < 167), one,
    SparseForm.singleton_coefficient, SparseForm.singleton_coefficient]
  split_ifs <;> simp

theorem rep_mat4 {S : PoseidonSboxPlan.State n} {S' : State} (h : StateRep I S S') (base lane : Nat) :
    Rep I (SparseLayer.mat4 S base lane) (mat4 S' base lane) := by
  unfold SparseLayer.mat4 mat4 SparseLayer.add SparseLayer.scale scaleNat
  rcases lane with _ | _ | _ | lane <;>
    repeat' first | exact rep_get h _ | apply rep_add | apply rep_scale

theorem rep_foldl {fs : List (SparseForm n)} {ss : List Sym} (h : List.Forall₂ (Rep I) fs ss) :
    ∀ {acc : SparseForm n} {acc' : Sym}, Rep I acc acc' → Rep I (fs.foldl SparseLayer.add acc) (ss.foldl Sym.add acc') := by
  induction h with
  | nil => exact fun hacc => hacc
  | cons hd _ ih => exact fun hacc => ih (rep_add hacc hd)

theorem forall₂_map {α : Type} (l : List α) {f : α → SparseForm n} {g : α → Sym} (h : ∀ a, Rep I (f a) (g a)) :
    List.Forall₂ (Rep I) (l.map f) (l.map g) := by
  induction l with
  | nil => exact .nil
  | cons a l ih => exact .cons (h a) ih

theorem rep_column {S : PoseidonSboxPlan.State n} {S' : State} (h : StateRep I S S') (index : Nat) :
    Rep I (SparseLayer.column S index) (column S' index) := by
  unfold SparseLayer.column column SparseLayer.sumForms sumForms
  exact rep_foldl (forall₂_map _ fun _ => rep_mat4 h _ _) rep_empty

theorem rep_sum {S : PoseidonSboxPlan.State n} {S' : State} (h : StateRep I S S') :
    Rep I (SparseLayer.sum S) (sum S') := by
  unfold SparseLayer.sum sum SparseLayer.sumForms sumForms
  exact rep_foldl (forall₂_map _ fun index => rep_get h index) rep_empty

theorem rep_external {S : PoseidonSboxPlan.State n} {S' : State} (h : StateRep I S S') :
    StateRep I (SparseLayer.external S) (external S') := by
  intro lane
  rw [external, Fin.getElem_fin, Vector.getElem_ofFn]
  exact rep_add (rep_mat4 h _ _) (rep_column h _)

theorem rep_internal {S : PoseidonSboxPlan.State n} {S' : State} (h : StateRep I S S') :
    StateRep I (SparseLayer.internal S) (internal S') := by
  intro lane
  rw [internal, Fin.getElem_fin, Vector.getElem_ofFn]
  exact rep_add (rep_scale _ (h lane)) (rep_sum h)

theorem rep_sboxAt (index : Nat) : Rep I (PoseidonSboxPlan.sboxOutputAt I index) (sboxAt index) := by
  unfold sboxAt
  split_ifs with small
  · intro c
    rw [eval_atom _ _ (show 16 + index < 167 by omega), atomOf, dif_neg (show ¬16 + index < 16 by omega),
      if_pos (show 16 + index < 166 by omega), show 16 + index - 16 = index by omega]
  · have empty : PoseidonSboxPlan.sboxOutputAt I index = .empty := by
      unfold PoseidonSboxPlan.sboxOutputAt
      rw [dif_neg (by rw [PoseidonRetainedSlots.rows_length]; exact small)]
    rw [empty]
    exact rep_empty

theorem rep_fullOutput (nextSbox : Nat) :
    StateRep I (PoseidonSboxPlan.fullOutput I nextSbox) (fullOutput nextSbox) := by
  intro lane
  rw [fullOutput, Fin.getElem_fin, Vector.getElem_ofFn]
  exact rep_sboxAt _

theorem rep_partialState {S : PoseidonSboxPlan.State n} {S' : State} (h : StateRep I S S') (nextSbox : Nat) :
    StateRep I (PoseidonSboxPlan.partialState I nextSbox S) (partialState nextSbox S') := by
  intro lane
  rw [partialState, Fin.getElem_fin, Vector.getElem_ofFn]
  unfold PoseidonSboxPlan.partialState PoseidonSboxPlan.partialOutput
  split_ifs
  · exact rep_sboxAt _
  · exact h lane

theorem forall₂_ofFn {m : Nat} {f : Fin m → SboxRow.Forms n} {g : Fin m → Row} (h : ∀ i, RowRep I (f i) (g i)) :
    List.Forall₂ (RowRep I) (List.ofFn f) (List.ofFn g) := by
  induction m with
  | zero => simp
  | succ m ih =>
      rw [List.ofFn_succ, List.ofFn_succ]
      exact .cons (h 0) (ih fun i => h i.succ)

theorem rep_fullRows {S : PoseidonSboxPlan.State n} {S' : State} (h : StateRep I S S')
    (constants : List (List Nat)) (round nextSbox : Nat) :
    List.Forall₂ (RowRep I) (PoseidonSboxPlan.fullRows I constants round nextSbox S)
      (fullRows constants round nextSbox S') := by
  unfold PoseidonSboxPlan.fullRows fullRows
  refine forall₂_ofFn fun lane => ⟨rep_addConstant (h lane) _, ?_⟩
  dsimp only
  unfold PoseidonSboxPlan.fullOutput
  rfl

theorem rep_partialRows {S : PoseidonSboxPlan.State n} {S' : State} (h : StateRep I S S') (round nextSbox : Nat) :
    List.Forall₂ (RowRep I) (PoseidonSboxPlan.partialRows I round nextSbox S) (partialRows round nextSbox S') := by
  unfold PoseidonSboxPlan.partialRows partialRows
  refine .cons ⟨rep_addConstant (h 0) _, ?_⟩ .nil
  dsimp only
  unfold PoseidonSboxPlan.partialOutput
  rfl

theorem rep_compileStep {S : PoseidonSboxPlan.State n} {S' : State} (h : StateRep I S S') (nextSbox : Nat)
    (step : Permutation.Step) :
    StepRep I (PoseidonSboxPlan.compileStep I nextSbox S step) (compileStep nextSbox S' step) := by
  cases step with
  | initialLayer => exact ⟨rfl, rep_external h, .nil⟩
  | initialFullRound round => exact ⟨rfl, rep_external (rep_fullOutput nextSbox), rep_fullRows h _ _ _⟩
  | partialRound round => exact ⟨rfl, rep_internal (rep_partialState h nextSbox), rep_partialRows h _ _⟩
  | terminalFullRound round => exact ⟨rfl, rep_external (rep_fullOutput nextSbox), rep_fullRows h _ _ _⟩

theorem rep_compile : ∀ (steps : List Permutation.Step) (nextSbox : Nat) {S : PoseidonSboxPlan.State n}
    {S' : State}, StateRep I S S' → StepRep I (PoseidonSboxPlan.compile I nextSbox S steps) (compile nextSbox S' steps)
  | [], _, _, _, h => ⟨rfl, h, .nil⟩
  | step :: rest, nextSbox, S, S', h => by
      have head := rep_compileStep h nextSbox step
      have tail := rep_compile rest (compileStep nextSbox S' step).nextSbox head.state
      rw [PoseidonSboxPlan.compile, compile, head.nextSbox]
      exact ⟨tail.nextSbox, tail.state, List.rel_append head.rows tail.rows⟩

theorem rep_input : StateRep I I.input (Vector.ofFn fun lane => Sym.atom lane.val) := by
  intro lane c
  rw [Fin.getElem_fin, Vector.getElem_ofFn, eval_atom _ _ (by omega), atomOf, dif_pos lane.isLt]

/-- **The trace over the atoms.** Row `r` of F′'s trace has the input of row
`r` of `atomRows` over the atoms of the interface, and the S-box output it names. -/
theorem rep_trace : List.Forall₂ (RowRep I) (PoseidonSboxPlan.trace I).rows atomRows.toList :=
  (rep_compile Permutation.schedule 0 rep_input).rows

end trace

/-! ## The atoms of an invocation of a block -/

theorem recompose_coefficient {n : Nat} (c : Fin n) : ∀ {m : Nat} (col : Fin m → Fin n),
    (RetainedSlot.recomposeForms (List.ofFn fun t => SparseForm.singleton (col t) 1)).coefficient c =
      ∑ t : Fin m, if col t = c then Phi81Relation.PiDECAlgebra.Radix.fieldOfNat 3 ^ t.val else 0
  | 0, _ => by simp [RetainedSlot.recomposeForms]
  | m + 1, col => by
      rw [List.ofFn_succ, RetainedSlot.recomposeForms, SparseForm.add_coefficient, SparseForm.scale_coefficient,
        recompose_coefficient c (fun t => col t.succ), SparseForm.singleton_coefficient, Fin.sum_univ_succ,
        Finset.mul_sum]
      refine congrArg₂ (· + ·) (by simp) (Finset.sum_congr rfl fun t _ => ?_)
      split_ifs <;> simp [pow_succ, mul_comm]

theorem power_eq (t : Nat) : power t = Phi81Relation.PiDECAlgebra.Radix.fieldOfNat 3 ^ t := by
  unfold power powers
  split <;> simp

theorem slotEntries_coefficient {sourceWidth n : Nat} (block : LowNormBlock.Block sourceWidth) (start : Nat)
    (fits : start + block.coordinateCount ≤ n) (slot : Fin block.slotCount) (c : Fin n) :
    (⟨slotEntries block start fits slot⟩ : SparseForm n).coefficient c = (block.form start fits slot).coefficient c := by
  rw [slotEntries, LowNormBlock.Block.form, recompose_coefficient, coefficient_eq_sum, List.map_ofFn, Fin.sum_ofFn]
  refine Finset.sum_congr rfl fun t _ => ?_
  simp only [Function.comp, power_eq]

section invocation

variable (block : Poseidon.Block) (n : Nat) (oneBound : block.oneColumn < n)
  (slotCountEq : block.retained.slotCount = block.invocationCount * 150)
  (fits : block.retained.start + block.retained.coordinateCount ≤ n)
  (input : PoseidonSboxPlan.State n) (inv : Fin block.invocationCount)

theorem sboxOutput_eq (k : Nat) (small : k < PoseidonRetainedSlots.rows.length) :
    (block.invocationInterface n oneBound slotCountEq fits input inv).sboxOutput ⟨k, small⟩ =
      block.retained.semantic.form block.retained.start fits ⟨150 * inv.val + k, by
        rw [PoseidonRetainedSlots.rows_length] at small
        show _ < block.retained.slotCount
        rw [slotCountEq]
        omega⟩ := by
  show block.retained.semantic.form _ _ _ = _
  congr 1

/-- The atom forms of `atomsOf` have the coefficients of the atom forms of the
interface that `Poseidon.Block.row?` builds. -/
theorem atomsOf_form (a : Nat) (inside : a < 167) (c : Fin n) :
    ((atomsOf block n oneBound slotCountEq fits input inv).form a).coefficient c =
      (atomOf (block.invocationInterface n oneBound slotCountEq fits input inv) a).coefficient c := by
  unfold Atoms.form atomOf
  by_cases lane : a < 16
  · rw [if_pos lane, dif_pos lane, atomsOf, Vector.getElem?_ofFn, dif_pos lane, Option.getD_some,
      normalize_coefficient]
    rfl
  · rw [if_neg lane, dif_neg lane]
    by_cases sbox : a < 166
    · have small : a - 16 < PoseidonRetainedSlots.rows.length := by
        rw [PoseidonRetainedSlots.rows_length]
        omega
      rw [if_pos sbox, if_pos sbox, atomsOf, Vector.getElem?_ofFn, dif_pos (by omega), Option.getD_some,
        slotEntries_coefficient, PoseidonSboxPlan.sboxOutputAt, dif_pos small, sboxOutput_eq]
    · rw [if_neg sbox, if_neg sbox]
      rfl

end invocation

/-! ## The rows of an invocation -/

theorem meaningfulForm_coefficient {n : Nat} {f g : SboxRow.Forms n}
    (input : ∀ c, f.input.coefficient c = g.input.coefficient c)
    (output : ∀ c, f.output.coefficient c = g.output.coefficient c)
    (port : Fin NightstreamFPrime.Spec.ProductionRelation.meaningfulPortCount) (c : Fin n) :
    (f.meaningfulForm port).coefficient c = (g.meaningfulForm port).coefficient c := by
  unfold SboxRow.Forms.meaningfulForm
  split
  · exact output c
  · exact input c
  · rfl

/-- **The rows of an invocation.** When `invocationRows?` returns the rows of
invocation `inv` of `block`, there are 150, and each port of row `r` has the
coefficients of that port of row `150 * inv + r` of F′'s `block.row?`. -/
theorem invocationRows?_spec {block : Poseidon.Block} {n inv : Nat} {rows : Array (SboxRow.Forms n)}
    (computed : invocationRows? block n inv = some rows) :
    rows.size = 150 ∧ ∀ (r : Nat) (inside : r < rows.size), ∃ forms, block.row? n (150 * inv + r) = some forms ∧
      ∀ port c, ((rows[r]'inside).meaningfulForm port).coefficient c = (forms port).coefficient c := by
  unfold invocationRows? at computed
  split_ifs at computed with invBound oneBound kind slotCountEq fits
  obtain ⟨input, loaded, rfl⟩ := Option.map_eq_some_iff.mp computed
  have rep := rep_trace (I := block.invocationInterface n oneBound slotCountEq fits input ⟨inv, invBound⟩)
  have length : atomRows.size = 150 := by
    have equal := rep.length_eq
    rw [PoseidonSboxPlan.trace_rows_length] at equal
    simpa using equal.symm
  refine ⟨by rw [Array.size_map, length], fun r inside => ?_⟩
  have small : r < 150 := by rwa [Array.size_map, length] at inside
  have rs : r < atomRows.size := by rw [length]; exact small
  have rowBound : 150 * inv + r < block.invocationCount * 150 := by
    have := Nat.mul_le_mul_right 150 (show inv + 1 ≤ block.invocationCount by omega)
    omega
  have decoded : Fin.decodeProd (⟨150 * inv + r, rowBound⟩ : Fin (block.invocationCount * 150)) =
      (⟨inv, invBound⟩, ⟨r, small⟩) := by
    ext
    · show (150 * inv + r) / 150 = inv
      omega
    · show (150 * inv + r) % 150 = r
      omega
  have row := Poseidon.Block.rowWithInput?_of_valid block n (block.input.state? n block.oneColumn) (150 * inv + r)
    rowBound oneBound kind slotCountEq fits input (by rw [decoded]; exact loaded)
  generalize Fin.decodeProd (⟨150 * inv + r, rowBound⟩ : Fin (block.invocationCount * 150)) = d at row decoded
  subst decoded
  refine ⟨_, row, fun port c => ?_⟩
  have pair := rep.get (by rw [PoseidonSboxPlan.trace_rows_length]; exact small) (by simpa [length] using small)
  simp only [List.get_eq_getElem, Array.getElem_toList] at pair
  obtain ⟨repInput, repOutput⟩ := pair
  simp only [Array.getElem_map, PoseidonRetainedRows.rows, List.get_eq_getElem, List.getElem_map]
  refine meaningfulForm_coefficient (fun c => ?_) (fun c => ?_) port c
  · rw [Atoms.row, expand_coefficient, repInput c]
    exact eval_congr _ c (atomsOf_form block n oneBound slotCountEq fits input ⟨inv, invBound⟩ · · c) _
  · rw [Atoms.row, repOutput]
    by_cases sbox : atomRows[r].output < 150
    · have form := atomsOf_form block n oneBound slotCountEq fits input ⟨inv, invBound⟩ (16 + atomRows[r].output)
        (by omega) c
      rw [Atoms.form, if_neg (show ¬16 + atomRows[r].output < 16 by omega),
        if_pos (show 16 + atomRows[r].output < 166 by omega), show 16 + atomRows[r].output - 16 = atomRows[r].output by omega,
        atomOf, dif_neg (show ¬16 + atomRows[r].output < 16 by omega), if_pos (show 16 + atomRows[r].output < 166 by omega),
        show 16 + atomRows[r].output - 16 = atomRows[r].output by omega] at form
      exact form
    · have none : (atomsOf block n oneBound slotCountEq fits input ⟨inv, invBound⟩).sbox[atomRows[r].output]? = none :=
        by simp [sbox]
      rw [none, PoseidonSboxPlan.sboxOutputAt, dif_neg (by rw [PoseidonRetainedSlots.rows_length]; exact sbox)]
      rfl

/-! ## The entries of F′'s matrices -/

/-- The first row of block `k` of a matrix program. -/
def blockStart (blocks : List MatrixProgram.Block) (k : Nat) : Nat :=
  ((blocks.take k).map MatrixProgram.Block.rowCount).sum

theorem select_blockStart (lw : Nat) (sourceRow : Nat → Option NightstreamFPrime.Layout.R1CS.Row) :
    ∀ (blocks : List MatrixProgram.Block) (k : Nat) (block : MatrixProgram.Block) (o : Nat),
      blocks[k]? = some block → o < block.rowCount →
        (MatrixProgram.Program.mk blocks).row? lw sourceRow (blockStart blocks k + o) = block.row? lw sourceRow o
  | [], _, _, _, found, _ => by simp at found
  | b :: rest, 0, block, o, found, small => by
      simp only [List.getElem?_cons_zero, Option.some.injEq] at found
      subst found
      rw [blockStart, List.take_zero, List.map_nil, List.sum_nil, Nat.zero_add]
      exact MatrixProgram.Program.cons_first_row? _ rest lw sourceRow o small
  | b :: rest, k + 1, block, o, found, small => by
      rw [List.getElem?_cons_succ] at found
      rw [blockStart, List.take_succ_cons, List.map_cons, List.sum_cons,
        MatrixProgram.Program.cons_rest_row? b rest lw sourceRow _ (by omega),
        show b.rowCount + ((rest.take k).map MatrixProgram.Block.rowCount).sum + o - b.rowCount =
          blockStart rest k + o by rw [blockStart]; omega]
      exact select_blockStart lw sourceRow rest k block o found small

/-- Row `blockStart k + o` of a matrix program is row `o` of its block `k`
(`Program.row?`). -/
theorem row?_blockStart (lw : Nat) (sourceRow : Nat → Option NightstreamFPrime.Layout.R1CS.Row)
    (program : MatrixProgram.Program) (k : Nat) (block : MatrixProgram.Block) (o : Nat)
    (found : program.blocks[k]? = some block) (small : o < block.rowCount) :
    program.row? lw sourceRow (blockStart program.blocks k + o) = block.row? lw sourceRow o := by
  cases program with
  | mk blocks => exact select_blockStart lw sourceRow blocks k block o found small

theorem blockStart_add_le : ∀ (blocks : List MatrixProgram.Block) (k : Nat) (block : MatrixProgram.Block),
    blocks[k]? = some block → blockStart blocks k + block.rowCount ≤ (blocks.map MatrixProgram.Block.rowCount).sum
  | [], _, _, found => by simp at found
  | b :: rest, 0, block, found => by
      simp only [List.getElem?_cons_zero, Option.some.injEq] at found
      subst found
      simp [blockStart]
  | b :: rest, k + 1, block, found => by
      rw [List.getElem?_cons_succ] at found
      have := blockStart_add_le rest k block found
      simp only [blockStart, List.take_succ_cons, List.map_cons, List.sum_cons] at this ⊢
      omega

theorem row?_lt {block : Poseidon.Block} {n ordinal : Nat} {forms : RowForms n}
    (found : block.row? n ordinal = some forms) : ordinal < block.rowCount := by
  by_contra outside
  unfold Poseidon.Block.row? Poseidon.Block.rowWithInput? at found
  rw [dif_neg outside] at found
  simp at found

theorem source_entry {cv fresh running count lw : Nat}
    (matrices : Fin count → PaperLinearAlgebra.BooleanMatrix F cv lw)
    (polynomial : CCSResidualTable.ConstraintPolynomial F count) (j : Fin count)
    (vertex : BooleanVertex cv) (c : Fin (Phi81CarrierLayout.carrierWidth lw)) (below : c.val < lw) :
    (Phi81MatrixSource.source cv fresh running count lw matrices polynomial).matrices j vertex c =
      matrices j vertex ⟨c.val, below⟩ := by
  show Phi81CarrierLayout.extendMatrix 0 (matrices j) vertex c = _
  unfold Phi81CarrierLayout.extendMatrix Phi81CarrierLayout.logicalColumn?
  rw [dif_pos below]

theorem plan_matrix {lw : Nat} (plan : ProductionRelation.Plan lw) (row : Fin plan.rowCount)
    (port : Fin Spec.ProductionRelation.matrixCount) (below : port.val < Spec.ProductionRelation.meaningfulPortCount)
    (column : Fin lw) :
    plan.matrix port (plan.rowLayout.toVertex row) column = (plan.forms row ⟨port.val, below⟩).coefficient column := by
  unfold ProductionRelation.Plan.matrix ProductionRelation.Plan.portForm ProductionRelation.meaningfulPort?
  rw [plan.rowLayout.toColumn_toVertex, dif_pos below]

/-- **The Poseidon2 rows of F′'s matrices.** Let block `k` of the matrix program
of `application` be a Poseidon2 block. When `invocationRows?` returns the rows
of its invocation `inv`, row `r` gives row `blockStart k + 150 * inv + r` of
every completed CCS matrix of F′'s relation (`PaperAlgebra.matrixSource`):
matrix `j` has at column `c` the coefficient of port `j` of row `r` below the
logical width, and zero past it. -/
theorem matrixSource_entry (application : Lifecycle.Stage1.Application.Program)
    (fits : PerApplicationFixedPoint.FitsTwoPow28 application) {k : Nat} {block : Poseidon.Block}
    (found : (PerApplicationMatrixProgram.matrixProgram application).blocks[k]? =
      some (MatrixProgram.Block.poseidon block))
    {inv : Nat} {rows : Array (SboxRow.Forms (PerApplicationFixedPoint.logicalWidth application))}
    (computed : invocationRows? block (PerApplicationFixedPoint.logicalWidth application) inv = some rows)
    (r : Nat) (inside : r < rows.size) (j : Fin productionProfile.ccsMatrices)
    (c : Fin (Phi81CarrierLayout.carrierWidth (PerApplicationFixedPoint.logicalWidth application))) :
    (Lifecycle.PaperAlgebra.matrixSource (PerApplicationFixedPoint.relation application fits).system).matrices j
        (vertexAt Lifecycle.cubeVariables
          (blockStart (PerApplicationMatrixProgram.matrixProgram application).blocks k + 150 * inv + r)) c =
      if h : c.val < PerApplicationFixedPoint.logicalWidth application then
        ((rows[r]'inside).meaningfulForm ⟨j.val, j.isLt⟩).coefficient ⟨c.val, h⟩
      else 0 := by
  obtain ⟨-, spec⟩ := invocationRows?_spec computed
  obtain ⟨forms, row, same⟩ := spec r inside
  have selected := row?_blockStart (PerApplicationFixedPoint.logicalWidth application)
    (PerApplicationCanonicalPackage.sourceRow application fits)
    (PerApplicationMatrixProgram.matrixProgram application) k (MatrixProgram.Block.poseidon block)
    (150 * inv + r) found (row?_lt row)
  have total : blockStart (PerApplicationMatrixProgram.matrixProgram application).blocks k + 150 * inv + r <
      (PerApplicationFixedPoint.structuralPlan application fits).rowCount := by
    rw [← PerApplicationMatrixProgram.matrixProgram_rowCount_eq_structuralPlan application fits]
    have before := blockStart_add_le _ k _ found
    have small := row?_lt row
    simp only [MatrixProgram.Block.rowCount] at before
    unfold MatrixProgram.Program.rowCount
    omega
  have planRow := PerApplicationCanonicalPackage.matrixProgram_row? application fits ⟨_, total⟩
  rw [← Nat.add_assoc] at selected
  have formsEq : (PerApplicationFixedPoint.structuralPlan application fits).forms ⟨_, total⟩ = forms :=
    Option.some.inj (planRow.symm.trans (selected.trans row))
  have vertex : vertexAt Lifecycle.cubeVariables
      (blockStart (PerApplicationMatrixProgram.matrixProgram application).blocks k + 150 * inv + r) =
      (PerApplicationFixedPoint.structuralPlan application fits).rowLayout.toVertex ⟨_, total⟩ := by
    unfold vertexAt
    congr 1
    exact Fin.ext (Nat.mod_eq_of_lt (Nat.lt_of_lt_of_le total
      (PerApplicationFixedPoint.structuralPlan application fits).rowCount_le))
  rw [Lifecycle.PaperAlgebra.matrixSource, vertex]
  split_ifs with below
  · have system : (PerApplicationFixedPoint.relation application fits).system.matrices =
        (PerApplicationFixedPoint.structuralPlan application fits).matrix := by
      rw [← PerApplicationFixedPoint.relation_matrices]
      rfl
    rw [source_entry _ _ _ _ _ below, system,
      plan_matrix _ _ _ (show j.val < NightstreamFPrime.Spec.ProductionRelation.meaningfulPortCount from j.isLt),
      formsEq]
    exact (same _ _).symm
  · exact Phi81MatrixSource.source_matrix_tail_zero _ _ _ _ _ _ _ _ _ _ (by omega)

end CompressionLean.Checks
