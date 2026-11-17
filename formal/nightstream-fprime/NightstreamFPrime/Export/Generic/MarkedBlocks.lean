import NightstreamFPrime.Export.Generic.CircuitRelation
import NightstreamFPrime.Export.Generic.PieceRowsCompleteness
import NightstreamFPrime.Export.Generic.PermutationWindow

/-! The blocks of segmented rows pack (SPEC.md section 4.4). Let a circuit's rows be free rows that
read no inner local of a mark, and the rows of each mark in order (`Circuit.Segmented`). At the
first row of a mark, the next mark is that mark and its rows follow, so `split` makes its block;
at a free row no mark starts, because the row reads its first column and a mark's first local is
an inner local. A ring-product block ends before the next mark's first row, so it holds free rows
only (`splitAux_segmented`). So the blocks are the marks, and every other piece reads free rows
only, which read no dead local. With ascending marks after the public columns and lanes that are
affine and read only live columns before their block, the blocks pack (`packable_of_segmented`). -/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Generic.MarkedBlocks

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Export.Generic.Pieces
open NightstreamFPrime.Export.Generic.Frame
open NightstreamFPrime.Export.Generic.CircuitRelation
open NightstreamFPrime.Export.Generic.PieceRowsCompleteness

/-- A mark's block. -/
def blockOf (mark : PermutationMark) : Block := ⟨mark.start, mark.lanes⟩

/-- The dead locals of the marks' blocks. -/
def deadOf (marks : List PermutationMark) : Nat → Bool := deadIn (marks.map blockOf)

theorem deadOf_eq_true_iff (marks : List PermutationMark) (column : Nat) :
    deadOf marks column = true ↔ InnerOf marks column := by
  simp only [deadOf, deadIn, List.any_map, List.any_eq_true, Function.comp_apply, Block.dead,
    blockOf, Bool.and_eq_true, InnerOf, PermutationMark.Inner]
  constructor
  · rintro ⟨mark, member, lower, upper⟩
    exact ⟨mark, member, of_decide_eq_true lower, of_decide_eq_true upper⟩
  · rintro ⟨mark, member, lower, upper⟩
    exact ⟨mark, member, decide_eq_true lower, decide_eq_true upper⟩

theorem deadOf_eq_false_iff (marks : List PermutationMark) (column : Nat) :
    deadOf marks column = false ↔ ¬ InnerOf marks column := by
  rw [← deadOf_eq_true_iff, Bool.not_eq_true]

theorem deadOf_start {marks : List PermutationMark} {mark : PermutationMark}
    (member : mark ∈ marks) : deadOf marks mark.start = true := by
  simp only [deadOf, deadIn, List.any_eq_true, List.mem_map]
  exact ⟨blockOf mark, ⟨mark, member, rfl⟩, by simp [Block.dead, blockOf]⟩

theorem avoids_iff (dead : Nat → Bool) (expression : Expr) :
    avoids dead expression = true ↔ expression.VarsSatisfy fun column => dead column = false := by
  induction expression with
  | var column => simp [avoids, Expr.VarsSatisfy]
  | const value => simp [avoids, Expr.VarsSatisfy]
  | add left right leftIH rightIH =>
      simp only [avoids, Bool.and_eq_true, leftIH, rightIH, Expr.VarsSatisfy]
  | mul left right leftIH rightIH =>
      simp only [avoids, Bool.and_eq_true, leftIH, rightIH, Expr.VarsSatisfy]

/-- A row reads the column that it starts with. -/
theorem blockStart_reads {allowed : Nat → Prop} :
    ∀ {row : Expr} {column : Nat}, blockStart row = some column → row.VarsSatisfy allowed →
      allowed column
  | .add (.var other) _, column, found, reads => by
      simp only [blockStart, Option.some.injEq] at found
      exact found ▸ reads.1
  | .var _, _, found, _ => by simp [blockStart] at found
  | .const _, _, found, _ => by simp [blockStart] at found
  | .add (.const _) _, _, found, _ => by simp [blockStart] at found
  | .add (.add _ _) _, _, found, _ => by simp [blockStart] at found
  | .add (.mul _ _) _, _, found, _ => by simp [blockStart] at found
  | .mul _ _, _, found, _ => by simp [blockStart] at found

/-- An affine expression is linear. -/
theorem linear_of_affine : ∀ {expression : Expr}, expression.Affine →
    (linear? expression).isSome = true
  | _, .var _ => rfl
  | _, .const _ => rfl
  | _, .add left right => by
      simp only [linear?, Option.isSome_bind, Option.isSome_map]
      rw [Option.any_eq_true_iff_get]
      refine ⟨linear_of_affine left, ?_⟩
      exact linear_of_affine right
  | _, .scaleLeft factor (expression := expression) inner => by
      show ((linear? expression).map (scale factor)).isSome = true
      rw [Option.isSome_map]
      exact linear_of_affine inner
  | _, .scaleRight factor (expression := expression) inner => by
      cases expression with
      | var column => rfl
      | const value => rfl
      | add left right =>
          show ((linear? (.add left right)).map (scale factor)).isSome = true
          rw [Option.isSome_map]
          exact linear_of_affine inner
      | mul left right =>
          show ((linear? (.mul left right)).map (scale factor)).isSome = true
          rw [Option.isSome_map]
          exact linear_of_affine inner

/-! ### The window's first row -/

/-- The first row of a mark's window starts with the mark's first local. -/
theorem window_head {canonical : List Expr} (first : Expr) (tail : List Expr)
    (shape : canonical = (Expr.var 16 - first) :: tail) (mark : PermutationMark) :
    ∃ row rest, window canonical mark = row :: rest ∧ blockStart row = some mark.start := by
  subst shape
  refine ⟨_, _, rfl, ?_⟩
  show some (mark.start + (16 - 16)) = some mark.start
  rfl

theorem window_length (canonical : List Expr) (mark : PermutationMark) :
    (window canonical mark).length = canonical.length := by
  simp [window]

/-! ### Ring products of free rows -/

/-- A ring-product block of `cells` cells over free rows reads no dead column: its lanes are
subterms of the rows, and each output starts one of them. -/
theorem ringProduct_avoids {dead : Nat → Bool} {cells start : Nat} {lanes rows : List Expr}
    (found : matchTemplate (ringInputs cells) (ringTemplate cells) rows = some (start, lanes))
    (free : ∀ row ∈ rows.take (ringDegree * cells),
      row.VarsSatisfy fun column => dead column = false) :
    avoidsDead dead (.ringProduct cells start lanes) = true := by
  have prefixRows := (matchTemplate_spec found).1
  have taken : (ringTemplate cells).map (subst (laneSubst (ringInputs cells) start lanes)) =
      rows.take (ringDegree * cells) := by
    rw [List.prefix_iff_eq_take.mp prefixRows]
    simp
  simp only [avoidsDead, Bool.and_eq_true, List.all_eq_true, List.mem_range]
  constructor
  · intro lane member
    exact (avoids_iff dead lane).mpr
      (matchTemplate_lanes_satisfy found (by simpa using free) lane member)
  · intro index below
    have member : subst (laneSubst (ringInputs cells) start lanes)
        (ringConstraint cells ⟨index, below⟩) ∈ rows.take (ringDegree * cells) := by
      rw [← taken]
      exact List.mem_map_of_mem (List.mem_ofFn.mpr ⟨⟨index, below⟩, rfl⟩)
    have reads := free _ member
    have output : (laneSubst (ringInputs cells) start lanes (ringInputs cells + index)).VarsSatisfy
        fun column => dead column = false := reads.1
    rw [laneSubst_local] at output
    have dead : dead (start + index) = false := output
    simp [dead]

theorem recognizeRing_avoids {dead : Nat → Bool} {rows : List Expr} {piece : Piece}
    (found : recognizeRing rows = some piece)
    (free : ∀ row ∈ rows.take (piece.length []),
      row.VarsSatisfy fun column => dead column = false) :
    avoidsDead dead piece = true := by
  unfold recognizeRing at found
  split at found
  · rename_i start lanes matched
    cases found
    exact ringProduct_avoids matched free
  · split at found
    · rename_i start lanes matched
      cases found
      exact ringProduct_avoids matched free
    · cases found

theorem recognizeRing_isBlock {rows : List Expr} {piece : Piece}
    (found : recognizeRing rows = some piece) : isBlock piece = false := by
  unfold recognizeRing at found
  split at found
  · cases found
    rfl
  · split at found
    · cases found
      rfl
    · cases found

theorem recognizeRing_length {rows : List Expr} {piece : Piece}
    (found : recognizeRing rows = some piece) (canonical : List Expr) :
    piece.length canonical = piece.length [] ∧ 0 < piece.length canonical := by
  unfold recognizeRing at found
  split at found
  · cases found
    exact ⟨rfl, show 0 < ringDegree * 1 by decide⟩
  · split at found
    · cases found
      exact ⟨rfl, show 0 < ringDegree * 2 by decide⟩
    · cases found

/-- The piece that is no marked block, at a free row: its rows are free when it ends before the
next mark's first row, and the rows after it keep their shape. -/
theorem otherPiece_segmented {canonical : List Expr} {dead : Nat → Bool}
    {marks : List PermutationMark} {row : Expr} {frees rest : List Expr}
    (free : ∀ other ∈ row :: frees, other.VarsSatisfy fun column => dead column = false)
    (beforeRest : ∀ length, beforeMark marks ((row :: (frees ++ rest)).take length) = true →
      length ≤ (row :: frees).length)
    (segmented : Segmented (fun column => dead column = false) (window canonical) marks rest) :
    avoidsDead dead (otherPiece canonical marks row (frees ++ rest)) = true ∧
      isBlock (otherPiece canonical marks row (frees ++ rest)) = false ∧
      0 < (otherPiece canonical marks row (frees ++ rest)).length canonical ∧
      Segmented (fun column => dead column = false) (window canonical) marks
        ((row :: (frees ++ rest)).drop
          ((otherPiece canonical marks row (frees ++ rest)).length canonical)) := by
  have single : avoidsDead dead (.constraint row) = true ∧ isBlock (.constraint row) = false ∧
      0 < (Piece.constraint row).length canonical ∧
      Segmented (fun column => dead column = false) (window canonical) marks
        ((row :: (frees ++ rest)).drop ((Piece.constraint row).length canonical)) :=
    ⟨(avoids_iff dead row).mpr (free row List.mem_cons_self), rfl, Nat.one_pos,
      Segmented.freeAppend (fun other member => free other (List.mem_cons_of_mem _ member))
        segmented⟩
  unfold otherPiece
  split
  · rename_i piece found
    split
    · rename_i guard
      obtain ⟨sameLength, positive⟩ := recognizeRing_length found canonical
      have fits := beforeRest _ guard
      have taken : (row :: (frees ++ rest)).take (piece.length []) =
          (row :: frees).take (piece.length []) := by
        rw [← List.cons_append, List.take_append_of_le_length (sameLength ▸ fits)]
      refine ⟨recognizeRing_avoids found ?_, recognizeRing_isBlock found, positive, ?_⟩
      · rw [taken]
        exact fun other member => free other (List.mem_of_mem_take member)
      · rw [← List.cons_append, List.drop_append_of_le_length fits]
        exact Segmented.freeAppend (fun other member =>
          free other (List.mem_of_mem_drop member)) segmented
    · exact single
  · exact single

/-- The piece that is no marked block, when every row is free. -/
theorem otherPiece_free {canonical : List Expr} {dead : Nat → Bool}
    {marks : List PermutationMark} {row : Expr} {rest : List Expr}
    (free : ∀ other ∈ row :: rest, other.VarsSatisfy fun column => dead column = false) :
    avoidsDead dead (otherPiece canonical marks row rest) = true ∧
      isBlock (otherPiece canonical marks row rest) = false ∧
      0 < (otherPiece canonical marks row rest).length canonical := by
  unfold otherPiece
  split
  · rename_i piece found
    split
    · obtain ⟨_, positive⟩ := recognizeRing_length found canonical
      exact ⟨recognizeRing_avoids found fun other member => free other (List.mem_of_mem_take member),
        recognizeRing_isBlock found, positive⟩
    · exact ⟨(avoids_iff dead row).mpr (free row List.mem_cons_self), rfl, Nat.one_pos⟩
  · exact ⟨(avoids_iff dead row).mpr (free row List.mem_cons_self), rfl, Nat.one_pos⟩

theorem blocksOf_single {piece : Piece} (notBlock : isBlock piece = false) :
    blocksOf [piece] = [] := by
  cases piece <;> simp_all [isBlock, blocksOf]

/-- `split` of segmented rows: the blocks are the marks, and every other piece reads no dead
column. -/
theorem splitAux_segmented (canonical : List Expr) (dead : Nat → Bool) (first : Expr)
    (tail : List Expr) (shape : canonical = (Expr.var 16 - first) :: tail) :
    ∀ (fuel : Nat) (marks : List PermutationMark) (rows : List Expr) (pieces : List Piece),
      rows.length ≤ fuel →
      Segmented (fun column => dead column = false) (window canonical) marks rows →
      (∀ mark ∈ marks, mark.lanes.length = 16 ∧ LinearLanes mark.lanes ∧
        dead mark.start = true) →
      ∃ later, splitAux canonical fuel marks rows pieces = pieces.reverse ++ later ∧
        blocksOf later = marks.map blockOf ∧
        ∀ piece ∈ later, isBlock piece = false → avoidsDead dead piece = true := by
  have noMarks : ∀ {marks : List PermutationMark},
      Segmented (fun column => dead column = false) (window canonical) marks [] → marks = [] := by
    intro marks segmented
    cases marks with
    | nil => rfl
    | cons mark marks =>
        obtain ⟨frees, rest, whole, _, _⟩ := segmented.split_head
        obtain ⟨row, others, windowShape, _⟩ := window_head first tail shape mark
        rw [windowShape] at whole
        simp at whole
  intro fuel
  induction fuel with
  | zero =>
      intro marks rows pieces length segmented _
      have empty : rows = [] := List.eq_nil_of_length_eq_zero (Nat.le_zero.mp length)
      subst empty
      rw [noMarks segmented]
      exact ⟨[], by simp [splitAux], rfl, by simp⟩
  | succ fuel ih =>
      intro marks rows pieces length segmented facts
      cases rows with
      | nil =>
          rw [noMarks segmented]
          exact ⟨[], by simp [splitAux], rfl, by simp⟩
      | cons row rest =>
          have step : ∀ (piece : Piece) (later : List PermutationMark) (remaining : List Expr),
              (nextPiece canonical marks row rest) = (piece, later) →
              remaining = (row :: rest).drop (piece.length canonical) →
              0 < piece.length canonical →
              Segmented (fun column => dead column = false) (window canonical) later remaining →
              (∀ mark ∈ later, mark.lanes.length = 16 ∧ LinearLanes mark.lanes ∧
                dead mark.start = true) →
              blocksOf [piece] ++ later.map blockOf = marks.map blockOf →
              (isBlock piece = false → avoidsDead dead piece = true) →
              ∃ found, splitAux canonical (fuel + 1) marks (row :: rest) pieces =
                  pieces.reverse ++ found ∧ blocksOf found = marks.map blockOf ∧
                  ∀ other ∈ found, isBlock other = false → avoidsDead dead other = true := by
            intro piece later remaining next dropped positive tailSegmented tailFacts blocks avoid
            obtain ⟨found, run, foundBlocks, foundAvoid⟩ := ih later remaining (piece :: pieces)
              (by
                rw [dropped, List.length_drop]
                simp only [List.length_cons] at length ⊢
                omega) tailSegmented tailFacts
            refine ⟨piece :: found, ?_, ?_, ?_⟩
            · simp only [splitAux, next]
              rw [← dropped, run]
              simp
            · rw [← blocks, ← foundBlocks]
              cases piece <;> simp [blocksOf]
            · intro other member notBlock
              rcases List.mem_cons.mp member with rfl | member
              · exact avoid notBlock
              · exact foundAvoid other member notBlock
          cases marks with
          | nil =>
              have free := segmented.free_of_nil
              obtain ⟨avoid, notBlock, positive⟩ :=
                otherPiece_free (canonical := canonical) (marks := []) free
              refine step _ [] _ (by simp [nextPiece]) rfl positive
                (Segmented.ofFree fun other member => free other (List.mem_of_mem_drop member))
                (by simp) (by simp [blocksOf_single notBlock]) (fun _ => avoid)
          | cons mark marks =>
              obtain ⟨frees, rest', whole, freeRows, tailSegmented⟩ := segmented.split_head
              obtain ⟨markRow, markOthers, windowShape, markStart⟩ :=
                window_head first tail shape mark
              have markFacts := facts mark List.mem_cons_self
              cases frees with
              | nil =>
                  simp only [List.nil_append] at whole
                  have rowEq : row = markRow := by
                    rw [windowShape] at whole
                    exact (List.cons.inj whole).1
                  have isPrefix : (window canonical mark).isPrefixOf (row :: rest) = true := by
                    rw [whole]
                    exact List.isPrefixOf_iff_prefix.mpr (List.prefix_append _ _)
                  refine step (.permutation mark.start mark.lanes) marks rest' ?_ ?_ ?_
                    tailSegmented (fun other member => facts other (List.mem_cons_of_mem _ member))
                    (by simp [blocksOf, blockOf]) (by simp [isBlock])
                  · simp only [nextPiece, rowEq ▸ markStart, isPrefix, markFacts.1, markFacts.2.1,
                      and_self, ↓reduceIte]
                  · rw [whole, Piece.length, ← window_length canonical mark, List.drop_left]
                  · rw [Piece.length, shape]
                    exact Nat.succ_pos _
              | cons free frees =>
                  simp only [List.cons_append, List.append_assoc] at whole
                  obtain ⟨rfl, rfl⟩ := List.cons.inj whole
                  have notStart : blockStart row ≠ some mark.start := by
                    intro starts
                    have := blockStart_reads starts (freeRows row List.mem_cons_self)
                    simp [markFacts.2.2] at this
                  obtain ⟨avoid, notBlock, positive, restSegmented⟩ :=
                    otherPiece_segmented (canonical := canonical) (frees := frees)
                      (rest := window canonical mark ++ rest') freeRows
                      (by
                        intro length guard
                        by_contra long
                        rw [windowShape] at guard
                        simp only [beforeMark, List.all_eq_true, bne_iff_ne, ne_eq] at guard
                        refine guard markRow ?_ markStart
                        rw [List.mem_take_iff_getElem]
                        refine ⟨(row :: frees).length, ?_, ?_⟩
                        · simp only [List.length_cons, List.length_append] at long ⊢
                          omega
                        · simp only [List.length_cons, List.getElem_cons_succ]
                          rw [List.getElem_append_right (by simp)]
                          simp)
                      (Segmented.block mark tailSegmented)
                  refine step _ (mark :: marks) _ ?_ rfl positive restSegmented facts ?_
                    (fun _ => avoid)
                  · simp only [nextPiece, notStart, ↓reduceIte]
                  · simp [blocksOf_single notBlock]

theorem ordered_of_pairwise :
    ∀ blocks : List Block, blocks.Pairwise (fun first second => first.start + 1096 ≤ second.start) →
      ordered blocks = true
  | [], _ => rfl
  | [_], _ => rfl
  | first :: second :: rest, pairwise => by
      simp only [ordered, Bool.and_eq_true, decide_eq_true_eq]
      exact ⟨(List.pairwise_cons.mp pairwise).1 second List.mem_cons_self,
        ordered_of_pairwise (second :: rest) (List.pairwise_cons.mp pairwise).2⟩

/-- The canonical template's first row starts with its first local. -/
theorem canonical_shape :
    ∃ first tail, PilotData.canonicalConstraints () = (Expr.var 16 - first) :: tail := by
  obtain ⟨tail, schedule⟩ : ∃ tail, Gadgets.Poseidon2.Permutation.schedule =
      Gadgets.Poseidon2.Permutation.Step.initialLayer :: tail := ⟨_, rfl⟩
  unfold PilotData.canonicalConstraints PilotData.canonicalRecipes
  rw [schedule]
  simp only [Gadgets.Poseidon2.Permutation.compile, Gadgets.Poseidon2.Permutation.stepRecipes,
    List.ofFn_succ, List.cons_append, recipeConstraints]
  exact ⟨_, _, rfl⟩

/-- Rows that are free rows reading no inner local of a mark and the marks' rows, in order, cut
into packed blocks: one block per mark. -/
theorem packable_of_segmented {publicColumns : Nat} {marks : List PermutationMark}
    {rows : List Expr}
    (segmented : Segmented (fun column => ¬ InnerOf marks column)
      Gadgets.Poseidon2.Permutation.markWindow marks rows)
    (lanes : ∀ mark ∈ marks, mark.Lanes fun column => ¬ InnerOf marks column)
    (ascend : marks.Pairwise fun first second => first.start + 1096 ≤ second.start)
    (starts : ∀ mark ∈ marks, publicColumns ≤ mark.start) :
    packable publicColumns (split (PilotData.canonicalConstraints ()) marks rows) = true := by
  let dead := deadOf marks
  have live : ∀ column, ¬ InnerOf marks column → dead column = false :=
    fun column outside => (deadOf_eq_false_iff marks column).mpr outside
  have windowed : Segmented (fun column => dead column = false)
      (window (PilotData.canonicalConstraints ())) marks rows :=
    (segmented.window_congr fun mark _ =>
      (PermutationWindow.window_eq_markWindow mark).symm).mono live
  obtain ⟨first, tail, shape⟩ := canonical_shape
  obtain ⟨later, run, blocks, avoid⟩ := splitAux_segmented _ dead first tail shape rows.length
    marks rows [] le_rfl windowed fun mark member =>
      ⟨(lanes mark member).length,
        List.all_eq_true.mpr fun lane laneMember =>
          linear_of_affine ((lanes mark member).affine lane laneMember),
        deadOf_start member⟩
  have pieces : split (PilotData.canonicalConstraints ()) marks rows = later := by
    simpa [split] using run
  rw [pieces]
  simp only [packable, Bool.and_eq_true]
  refine ⟨⟨?_, ?_⟩, ?_⟩
  · simp only [checkFrame, Bool.and_eq_true, List.all_eq_true]
    rw [blocks]
    refine ⟨ordered_of_pairwise _ (List.pairwise_map.mpr (ascend.imp fun order => order)), ?_⟩
    intro block member lane laneMember
    obtain ⟨mark, markMember, rfl⟩ := List.mem_map.mp member
    have facts := lanes mark markMember
    refine ⟨linear_of_affine (facts.affine lane laneMember), ?_⟩
    rw [avoids_iff]
    refine (facts.reads lane laneMember).mono lane fun column inside => ?_
    simp only [Bool.or_eq_false_iff, Block.covers, blockOf, Bool.and_eq_false_iff]
    exact ⟨Or.inl (decide_eq_false (by have := inside.1; omega)), live column inside.2⟩
  · simp only [startsFrom, List.all_eq_true, decide_eq_true_eq]
    rw [blocks]
    intro block member
    obtain ⟨mark, markMember, rfl⟩ := List.mem_map.mp member
    exact starts mark markMember
  · rw [List.all_eq_true]
    intro piece member
    cases piece with
    | permutation => rfl
    | constraint expression =>
        rw [blocks]
        exact avoid _ member rfl
    | ringProduct cells start lanes =>
        rw [blocks]
        exact avoid _ member rfl

end NightstreamFPrime.Export.Generic.MarkedBlocks
