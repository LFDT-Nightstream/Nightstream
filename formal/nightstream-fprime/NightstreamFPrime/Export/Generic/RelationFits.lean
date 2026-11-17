import NightstreamFPrime.Export.Generic.RelationCompleteness
import NightstreamFPrime.Export.Generic.PermutationWindow
import NightstreamFPrime.Gadgets.Poseidon2.LiveRows

/-! The capacity of a step package (SPEC.md section 4.4). A packed package fits its source width
and the cube (`CircuitRelation.Fits`) when its pieces read only columns before its first fresh
column and two numbers fit: the first S-box column plus 150 S-box columns for each block is at
most the width, and the relation rows plus 150 S-box rows for each block are at most `2^28`
(`fits_of_capacity`). Each lowered row reads a column of its piece or a fresh column before the
first S-box column (`lowerList_varsSatisfy`), and each output form reads S-box columns of the
width (`outputForms_varsSatisfy`).

Rows that are free rows reading only columns below a bound, and the rows of marks that lie below
it, read only columns below it (`segmented_varsBelow`). So the constraints of operations whose
marks are placed and whose rows are live rows read only columns before the end of their locals,
for every assignment (`ops_varsBelow`). -/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Generic.RelationFits

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Export
open NightstreamFPrime.Export.Package
open NightstreamFPrime.Export.Generic.Pieces
open NightstreamFPrime.Export.Generic.PieceRows
open NightstreamFPrime.Export.Generic.Frame
open NightstreamFPrime.Export.Generic.CircuitRelation
open NightstreamFPrime.Export.Generic.PieceRowsCompleteness
open NightstreamFPrime.Export.Generic.RowReads
open NightstreamFPrime.Export.Generic.RelationCompleteness
open NightstreamFPrime.Gadgets.Poseidon2

/-! ### Lowered rows -/

/-- The rows of one piece that reads only columns below `bound`, lowered from `next ≥ bound`,
read only columns before the piece's last fresh column. -/
theorem lowerPiece_varsSatisfy {bound next : Nat} (piece : Piece)
    (below : pieceBelow bound piece = true) (after : bound ≤ next) :
    ∀ row ∈ (lowerPiece (PilotData.canonicalConstraints ()) piece next).1,
      row.VarsSatisfy (· < (lowerPiece (PilotData.canonicalConstraints ()) piece next).2) := by
  have generic : ∀ constraints : List Expr, (∀ expression ∈ constraints, expression.VarsBelow bound) →
      ∀ row ∈ (R1CS.lowerConstraints constraints next).rows,
        row.VarsSatisfy (· < (R1CS.lowerConstraints constraints next).next) := by
    intro constraints scope row member
    refine (R1CS.lowerConstraints_rows_varsSatisfy constraints next (· < bound) (fun expression inside =>
      (Expr.varsSatisfy_lt_iff_varsBelow expression bound).mpr (scope expression inside))
      row member).mono row fun column fresh => ?_
    rw [R1CS.lowerConstraints_next]
    rcases fresh with low | ⟨_, high⟩
    · exact Nat.lt_of_lt_of_le low (Nat.le_trans after (Nat.le_add_right _ _))
    · exact high
  cases piece with
  | constraint expression =>
      simp only [lowerPiece]
      exact generic _ (piece_varsBelow (.constraint expression) below)
  | permutation start lanes =>
      simp only [lowerPiece]
      exact generic _ (piece_varsBelow (.permutation start lanes) below)
  | ringProduct cells start lanes =>
      simp only [lowerPiece]
      simp only [pieceBelow, Bool.and_eq_true, decide_eq_true_eq] at below
      have lane : ∀ index, (lanes.getD index 0).VarsSatisfy (· < next + ringDegree * cells) :=
        fun index => ((Expr.varsSatisfy_lt_iff_varsBelow _ bound).mpr
          (lanes_below below.1 index)).mono _ fun _ inside => by omega
      refine ringRows_varsSatisfy cells start next lanes (fun _ => lane _) (fun _ _ _ => lane _)
        (fun _ _ _ => lane _) (fun cell inside lane => ?_) (fun cell inside lane => ?_)
      · have := slot_lt lane inside
        simp only [outputColumn]
        omega
      · exact quotientColumn_lt inside lane

/-- The rows of pieces that read only columns below `bound`, lowered from `next ≥ bound`, read
only columns before the last fresh column. -/
theorem lowerList_varsSatisfy {bound : Nat} :
    ∀ (pieces : List Piece) (next : Nat), (∀ piece ∈ pieces, pieceBelow bound piece = true) →
      bound ≤ next →
      ∀ row ∈ (lowerList (PilotData.canonicalConstraints ()) pieces next).1,
        row.VarsSatisfy (· < (lowerList (PilotData.canonicalConstraints ()) pieces next).2)
  | [], _, _, _, row, member => by simp [lowerList] at member
  | piece :: rest, next, below, after, row, member => by
      have pieceNext := lowerPiece_next_le (PilotData.canonicalConstraints ()) piece next
      simp only [lowerList, List.mem_append] at member ⊢
      rcases member with inPiece | inRest
      · exact (lowerPiece_varsSatisfy piece (below piece List.mem_cons_self) after row
          inPiece).mono row fun _ inside => Nat.lt_of_lt_of_le inside
            (lowerList_next_le _ rest _)
      · exact lowerList_varsSatisfy rest _ (fun other member => below other
          (List.mem_cons_of_mem _ member)) (Nat.le_trans after pieceNext) row inRest

/-! ### The row count -/

/-- The rows of a piece's lowering: 128 for each cell of a ring product, and the lowered rows of
its constraints otherwise. -/
def pieceRowCount : Piece → Nat
  | .ringProduct cells _ _ => cells * 128
  | piece => R1CS.totalRowCount (piece.constraints (PilotData.canonicalConstraints ()))

/-- The number of rows of a packed relation, without building them (`relationRowCount_eq`). -/
def relationRowCount (pieces : List Piece) : Nat :=
  ((otherPieces pieces).map pieceRowCount).sum

theorem lowerPiece_length (piece : Piece) (next : Nat) :
    (lowerPiece (PilotData.canonicalConstraints ()) piece next).1.length = pieceRowCount piece := by
  cases piece with
  | ringProduct cells start lanes =>
      simp only [lowerPiece, ringRows, pieceRowCount, List.length_flatMap, List.length_ofFn,
        List.map_const', List.sum_replicate, List.length_range, smul_eq_mul]
  | constraint expression => simp [lowerPiece, pieceRowCount, R1CS.lowerConstraints_rows_length]
  | permutation start lanes => simp [lowerPiece, pieceRowCount, R1CS.lowerConstraints_rows_length]

theorem lowerList_length :
    ∀ (pieces : List Piece) (next : Nat),
      (lowerList (PilotData.canonicalConstraints ()) pieces next).1.length =
        (pieces.map pieceRowCount).sum
  | [], _ => rfl
  | piece :: rest, next => by
      simp only [lowerList, List.length_append, lowerPiece_length, lowerList_length rest,
        List.map_cons, List.sum_cons]

theorem relationRowCount_eq {publicColumns : Nat} {pieces : List Piece} {next width : Nat}
    (packed : packable publicColumns pieces = true) :
    relationRowCount pieces = (relationRows publicColumns pieces next width).length := by
  simp only [relationRows, packed, ite_true, List.length_map]
  rw [lowerPieces_eq, lowerList_length]
  rfl

/-! ### Output forms -/

theorem fill_varsSatisfy {allowed : Nat → Prop} :
    ∀ (writes : List (Nat × R1CS.LinearCombination)) (start : Array (Option R1CS.LinearCombination)),
      (∀ column form, Substitute.lookup start column = some form → form.VarsSatisfy allowed) →
      (∀ write ∈ writes, write.2.VarsSatisfy allowed) →
      ∀ column form, Substitute.lookup (Substitute.fill start writes) column = some form →
        form.VarsSatisfy allowed
  | [], start, initial, _, column, form, found => initial column form found
  | write :: rest, start, initial, written, column, form, found => by
      refine fill_varsSatisfy rest _ (fun other otherForm otherFound => ?_)
        (fun later member => written later (List.mem_cons_of_mem _ member)) column form found
      simp only [Substitute.lookup, Array.getD_eq_getD_getElem?,
        Array.getElem?_setIfInBounds] at otherFound
      split at otherFound
      · split at otherFound
        · simp only [Option.getD_some, Option.some.injEq] at otherFound
          subst otherFound
          exact written write List.mem_cons_self
        · simp at otherFound
      · exact initial other otherForm (by
          simpa only [Substitute.lookup, Array.getD_eq_getD_getElem?] using otherFound)

theorem toCombination_varsSatisfy {width : Nat} (form : SparseForm width) :
    (toCombination form).VarsSatisfy (· < width) := by
  intro term member
  simp only [toCombination, List.mem_map] at member
  obtain ⟨entry, _, rfl⟩ := member
  exact entry.column.isLt

/-- Each output form reads only columns of the width. -/
theorem outputForms_varsSatisfy (width sboxStart : Nat) (blocks : List Block) :
    ∀ column form, outputForms width blocks sboxStart column = some form →
      form.VarsSatisfy (· < width) := by
  refine fill_varsSatisfy _ _ (fun column form found => ?_) (fun write member => ?_)
  · simp only [Substitute.lookup, Array.getD_eq_getD_getElem?, Array.getElem?_replicate] at found
    split at found <;> simp at found
  · simp only [outputWrites, List.mem_flatMap, List.mem_ofFn] at member
    obtain ⟨_, _, lane, rfl⟩ := member
    exact toCombination_varsSatisfy _

theorem substitute_varsSatisfy {forms : Nat → Option R1CS.LinearCombination}
    {allowed : Nat → Prop} {combination : R1CS.LinearCombination}
    (source : combination.VarsSatisfy allowed)
    (formed : ∀ column form, forms column = some form → form.VarsSatisfy allowed) :
    (Substitute.substitute forms combination).VarsSatisfy allowed := by
  intro entry member
  rcases Substitute.substitute_terms forms combination member with
    ⟨_, inside, _, rfl⟩ | ⟨_, _, form, found, column⟩
  · exact source _ inside
  · obtain ⟨term, termMember, equal⟩ := List.mem_map.mp column
    rw [← equal]
    exact formed _ form found term termMember

/-! ### The capacity -/

theorem foldl_max_lt {α : Type} (measure : α → Nat) {width : Nat} :
    ∀ (items : List α) (initial : Nat), initial < width → (∀ item ∈ items, measure item < width) →
      items.foldl (fun bound item => max bound (measure item)) initial < width
  | [], _, low, _ => low
  | item :: rest, initial, low, bounded => by
      simp only [List.foldl_cons]
      exact foldl_max_lt measure rest _ (max_lt low (bounded item List.mem_cons_self))
        fun other member => bounded other (List.mem_cons_of_mem _ member)

theorem rowsMax_lt {rows : List R1CS.Row} {width : Nat} (positive : 0 < width)
    (scope : ∀ row ∈ rows, row.VarsSatisfy (· < width)) : LogicalPlan.rowsMax rows < width := by
  have combination : ∀ combination : R1CS.LinearCombination, combination.VarsSatisfy (· < width) →
      LogicalPlan.combinationMax combination < width := fun combination bounded =>
    foldl_max_lt Prod.fst _ 0 positive bounded
  refine foldl_max_lt LogicalPlan.rowMax rows 0 positive fun row member => ?_
  obtain ⟨a, b, c⟩ := scope row member
  exact max_lt (combination _ a) (max_lt (combination _ b) (combination _ c))

/-- The rows of a packed relation read only columns of the width when its pieces read only
columns before its first fresh column and its first S-box column is inside the width. -/
theorem relationRows_varsSatisfy {publicColumns : Nat} {pieces : List Piece} {next width : Nat}
    (packed : packable publicColumns pieces = true)
    (below : ∀ piece ∈ pieces, pieceBelow next piece = true)
    (columns : sboxStart pieces next ≤ width) :
    ∀ row ∈ relationRows publicColumns pieces next width, row.VarsSatisfy (· < width) := by
  intro row member
  simp only [relationRows, packed, ite_true, List.mem_map] at member
  obtain ⟨source, sourceMember, rfl⟩ := member
  rw [lowerPieces_eq] at sourceMember
  have lowered := lowerList_varsSatisfy (otherPieces pieces) next
    (fun piece member => below piece (List.mem_filter.mp member).1) (Nat.le_refl _) source
    sourceMember
  have last : (lowerList (PilotData.canonicalConstraints ()) (otherPieces pieces) next).2 =
      sboxStart pieces next := by
    rw [sboxStart, lowerPieces_eq]
  rw [last] at lowered
  have sourceScope := lowered.mono source fun _ inside => Nat.lt_of_lt_of_le inside columns
  have formed := outputForms_varsSatisfy width (sboxStart pieces next) (blocksOf pieces)
  exact ⟨substitute_varsSatisfy sourceScope.1 formed, substitute_varsSatisfy sourceScope.2.1 formed,
    substitute_varsSatisfy sourceScope.2.2 formed⟩

theorem input_varsSatisfy {allowed : Nat → Prop} (block : Block)
    (lanes : ∀ lane ∈ block.lanes, lane.VarsSatisfy allowed) (lane : Nat) :
    (invocationInputCombination block.invocation lane).toR1CS.VarsSatisfy allowed := by
  simp only [invocationInputCombination, Block.invocation, Pieces.invocation]
  by_cases inside : lane < block.lanes.length
  · rw [List.getD_eq_getElem _ _ (by simpa using inside), List.getElem_map]
    exact laneCombination_varsSatisfy _ (lanes _ (List.getElem_mem inside))
  · rw [List.getD_eq_default _ _ (by simpa using inside)]
    intro term member
    simp [zeroSparseCombination, SparseCombination.toR1CS] at member

/-- A packed relation fits its width and the cube when its pieces read only columns before its
first fresh column, its S-box columns end inside the width, and its rows with the S-box rows are
at most `2^28`. -/
theorem fits_of_capacity {publicColumns : Nat} {pieces : List Piece} {next width : Nat}
    (packed : packable publicColumns pieces = true)
    (below : pieces.all (pieceBelow next) = true)
    (positive : 0 < width)
    (columns : sboxStart pieces next + (blocksOf pieces).length * 150 ≤ width)
    (rows : (relationRows publicColumns pieces next width).length +
      (blocksOf pieces).length * 150 ≤ 2 ^ cubeVariables) :
    Fits publicColumns pieces next width := by
  have pieceBelowAll := List.all_eq_true.mp below
  have sboxInside : sboxStart pieces next ≤ width := Nat.le_trans (Nat.le_add_right _ _) columns
  refine ⟨⟨Nat.le_trans (Nat.le_add_right _ _) rows,
    rowsMax_lt positive (relationRows_varsSatisfy packed pieceBelowAll sboxInside)⟩,
    fun _ => ⟨fun block member lane term termMember => ?_, columns, rows⟩⟩
  have permutation := pieceBelowAll _ (permutation_mem_of_block member)
  simp only [pieceBelow, Bool.and_eq_true, decide_eq_true_eq] at permutation
  have laneScope : ∀ lane ∈ block.lanes, lane.VarsSatisfy (· < width) := fun lane inside =>
    ((Expr.varsSatisfy_lt_iff_varsBelow lane next).mpr
      ((exprBelow_iff next lane).mp (List.all_eq_true.mp permutation.1 lane inside))).mono lane
      fun _ low => Nat.lt_of_lt_of_le low (Nat.le_trans (next_le_sboxStart pieces next) sboxInside)
  exact substitute_varsSatisfy (input_varsSatisfy block laneScope lane.val)
    (outputForms_varsSatisfy width (sboxStart pieces next) (blocksOf pieces)) term termMember

/-! ### The scope of live rows -/

/-- Free rows that read only columns below `bound`, and the rows of marks whose lanes read only
columns below it and whose locals end at or before it, read only columns below it. -/
theorem segmented_varsBelow {bound : Nat} {marks : List PermutationMark} {rows : List Expr}
    (segmented : Segmented (· < bound) Permutation.markWindow marks rows)
    (lanes : ∀ mark ∈ marks, mark.Lanes (· < bound))
    (inside : ∀ mark ∈ marks, mark.start + 1096 ≤ bound) :
    ∀ row ∈ rows, row.VarsBelow bound := by
  induction segmented with
  | nil => intro row member; simp at member
  | free row reads _ restScope =>
      intro other member
      rcases List.mem_cons.mp member with rfl | later
      · exact (Expr.varsSatisfy_lt_iff_varsBelow _ bound).mp reads
      · exact restScope lanes inside other later
  | block mark _ restScope =>
      intro row member
      rcases List.mem_append.mp member with inWindow | later
      · have markLanes := lanes mark List.mem_cons_self
        have below : pieceBelow bound (.permutation mark.start mark.lanes) = true := by
          simp only [pieceBelow, Bool.and_eq_true, decide_eq_true_eq, List.all_eq_true]
          exact ⟨fun lane member => (exprBelow_iff bound lane).mpr
            ((Expr.varsSatisfy_lt_iff_varsBelow lane bound).mp
              ((markLanes.reads lane member).mono lane fun _ low => low.2)),
            inside mark List.mem_cons_self⟩
        rw [← PermutationWindow.window_eq_markWindow, ← permutation_constraints] at inWindow
        exact piece_varsBelow _ below row inWindow
      · exact restScope (fun other member => lanes other (List.mem_cons_of_mem _ member))
          (fun other member => inside other (List.mem_cons_of_mem _ member)) row later

/-- The constraints of operations whose marks lie in their locals from `shared`, and whose rows
are live rows for every live set that holds below `shared` and on their own locals, read only
columns before any `bound` at or after the end of their locals. -/
theorem ops_varsBelow {ops : List Op} {shared bound : Nat}
    (placed : Placed shared (localLength ops) ops)
    (rows : ∀ live : Nat → Prop, (∀ column, column < shared → live column) →
      OwnLive live shared (localLength ops) ops → LiveRows live ops)
    (after : shared + localLength ops ≤ bound) :
    ∀ constraint ∈ flatConstraints ops, constraint.VarsBelow bound := by
  have live := rows (· < bound) (fun _ low => by omega) (fun _ _ high _ => by omega)
  exact segmented_varsBelow live.segmented live.lanes fun mark member =>
    Nat.le_trans (placed.inside mark member).2 after

end NightstreamFPrime.Export.Generic.RelationFits
