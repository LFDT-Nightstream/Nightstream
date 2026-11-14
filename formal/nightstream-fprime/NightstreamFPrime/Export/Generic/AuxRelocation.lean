import NightstreamFPrime.Export.Generic.NativeBlock

namespace NightstreamFPrime.Export.Generic.AuxRelocation

open NightstreamFPrime.Spec NightstreamFPrime.Circuit NightstreamFPrime.Layout
open NightstreamFPrime.Export NightstreamFPrime.Export.Generic
open NightstreamFPrime.Export.Generic.Frame NightstreamFPrime.Export.Generic.Pieces

abbrev rename (map : Nat → Nat) : Expr → Expr :=
  Lifecycle.Nebula.Expr.subst (fun column => Expr.var (map column))

def combinationExpr (form : R1CS.LinearCombination) : Expr :=
  form.terms.foldr (fun term rest => Expr.const term.2 * Expr.var term.1 + rest)
    (Expr.const form.constant)

def rowExpr (row : R1CS.Row) : Expr :=
  combinationExpr row.c - combinationExpr row.a * combinationExpr row.b

theorem combinationExpr_eval (form : R1CS.LinearCombination) (env : Env) :
    (combinationExpr form).eval env = form.eval env := by
  rcases form with ⟨constant, terms⟩
  induction terms with
  | nil => simp [combinationExpr, R1CS.LinearCombination.eval]
  | cons term rest ih =>
      simp only [combinationExpr, List.foldr_cons, Expr.eval_hadd, Expr.eval_hmul,
        Expr.eval_const, Expr.eval_var, R1CS.LinearCombination.eval, List.map_cons,
        List.sum_cons] at ih ⊢
      rw [ih]
      exact add_left_comm _ _ _

theorem rowExpr_zero (row : R1CS.Row) (env : Env) :
    (rowExpr row).eval env = 0 ↔ row.Holds env := by
  simp only [rowExpr, Expr.eval_sub, Expr.eval_hmul, combinationExpr_eval,
    sub_eq_zero, R1CS.Row.Holds]
  exact eq_comm

def templates (blocks : List Block) : List Expr :=
  blocks.flatMap fun block =>
    (Piece.permutation block.start block.lanes).constraints (PilotData.canonicalConstraints ())

def Matched (map : Nat → Nat) (source : List Expr) (kept : List R1CS.Row)
    (blocks : List Block) : Prop :=
  ∃ ordered, (source.map (rename map)).Perm ordered ∧
    List.Forall₂ (fun left right => ∀ env, left.eval env = 0 ↔ right.eval env = 0)
      ordered (kept.map rowExpr ++ templates blocks)

theorem holds_permuted {left right : List Expr} (order : left.Perm right) (env : Env) :
    ConstraintsHold env left ↔ ConstraintsHold env right := by
  exact ⟨fun holds row member => holds row (order.mem_iff.mpr member),
    fun holds row member => holds row (order.mem_iff.mp member)⟩

theorem holds_equivalent {left right : List Expr}
    (related : List.Forall₂ (fun left right => ∀ env, left.eval env = 0 ↔ right.eval env = 0) left right)
    (env : Env) : ConstraintsHold env left ↔ ConstraintsHold env right := by
  induction related with
  | nil => rfl
  | @cons a b as bs equal _ ih =>
      have split (head : Expr) (tail : List Expr) :
          ConstraintsHold env (head :: tail) ↔ head.eval env = 0 ∧ ConstraintsHold env tail := by
        simp [ConstraintsHold]
      rw [split, split, equal env, ih]

theorem matched_holds {map : Nat → Nat} {source : List Expr} {kept : List R1CS.Row}
    {blocks : List Block} (matched : Matched map source kept blocks) (env : Env) :
    ConstraintsHold (env ∘ map) source ↔
      R1CS.RowsHold env kept ∧
      ∀ block ∈ blocks, ConstraintsHold env
        ((Piece.permutation block.start block.lanes).constraints
          (PilotData.canonicalConstraints ())) := by
  have renamed : ConstraintsHold (env ∘ map) source ↔
      ConstraintsHold env (source.map (rename map)) := by
    constructor
    · intro holds row member
      obtain ⟨original, originalMember, rfl⟩ := List.mem_map.mp member
      simpa only [rename, Lifecycle.Nebula.Expr.eval_subst, Expr.eval_var,
        Function.comp_def] using holds original originalMember
    · intro holds row member
      have := holds _ (List.mem_map_of_mem member)
      simpa only [rename, Lifecycle.Nebula.Expr.eval_subst, Expr.eval_var,
        Function.comp_def] using this
  obtain ⟨ordered, order, related⟩ := matched
  rw [renamed, holds_permuted order env, holds_equivalent related env]
  constructor
  · intro holds
    refine ⟨?_, ?_⟩
    · intro row member
      apply (rowExpr_zero row env).mp
      exact holds _ (List.mem_append_left _ (List.mem_map_of_mem member))
    · intro block member row rowMember
      apply holds
      exact List.mem_append_right _ (List.mem_flatMap.mpr ⟨block, member, rowMember⟩)
  · rintro ⟨keptHold, templateHold⟩ row member
    rcases List.mem_append.mp member with member | member
    · obtain ⟨original, originalMember, rfl⟩ := List.mem_map.mp member
      exact (rowExpr_zero original env).mpr (keptHold original originalMember)
    · obtain ⟨block, member, rowMember⟩ := List.mem_flatMap.mp member
      exact templateHold block member row rowMember

theorem reconstruct {map : Nat → Nat} {source : List Expr} {kept : List R1CS.Row}
    {blocks : List Block} (matched : Matched map source kept blocks)
    (frame : Framed blocks) (avoid : kept.all (NativeBlock.rowAvoids (deadIn blocks)) = true)
    {env : Env} (keptHold : R1CS.RowsHold env kept) (outputs : OutputsHold blocks env) :
    ∃ original : Env, ConstraintsHold original source ∧
      ∀ column, map column = column → deadIn blocks column = false →
        original column = env column := by
  obtain ⟨completed, agree, keptHold, templatesHold⟩ :=
    NativeBlock.completed_rows frame avoid keptHold outputs
  refine ⟨completed ∘ map, (matched_holds matched completed).mpr
    ⟨keptHold, fun block member => (templatesHold block member).2⟩, ?_⟩
  intro column fixed live
  simpa only [Function.comp_apply, fixed] using agree column live

noncomputable def extend (allowed : Nat → Prop) (map : Nat → Nat) (source : Env) : Env := by
  classical
  exact fun column => if h : ∃ original : { column // allowed column }, map original = column then
    source (Classical.choose h).val else source column

theorem extend_at {allowed : Nat → Prop} {map : Nat → Nat}
    (injective : Function.Injective (fun column : { column // allowed column } => map column))
    (source : Env) (column : { column // allowed column }) :
    extend allowed map source (map column) = source column := by
  unfold extend
  rw [dite_eq_left ⟨column, rfl⟩]
  have found := Classical.choose_spec (show ∃ original : { column // allowed column }, map original = map column from
    ⟨column, rfl⟩)
  rw [injective found]

theorem honest_complete {allowed : Nat → Prop} {map : Nat → Nat} {source : List Expr}
    {kept : List R1CS.Row} {blocks : List Block}
    (matched : Matched map source kept blocks)
    (injective : Function.Injective (fun column : { column // allowed column } => map column))
    (scope : ∀ row ∈ source, row.VarsSatisfy allowed)
    {honest : Env} (holds : ConstraintsHold honest source) :
    ∃ extended : Env,
      (∀ column : { column // allowed column }, extended (map column) = honest column) ∧
      R1CS.RowsHold extended kept ∧
      ∀ block ∈ blocks, ConstraintsHold extended
        ((Piece.permutation block.start block.lanes).constraints
          (PilotData.canonicalConstraints ())) := by
  let extended := extend allowed map honest
  have agree : ∀ column : { column // allowed column }, extended (map column) = honest column :=
    extend_at injective honest
  refine ⟨extended, agree, ?_⟩
  apply (matched_holds matched extended).mp
  intro row member
  have eval_equal : row.eval (extended ∘ map) = row.eval honest := by
    apply Expr.eval_eq_of_agree_satisfy row allowed _ _ (scope row member)
    intro column member
    exact agree ⟨column, member⟩
  rw [eval_equal]
  exact holds row member

end NightstreamFPrime.Export.Generic.AuxRelocation
