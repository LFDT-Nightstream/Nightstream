import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.Phi81RingActionHonest
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.DirectCalls.Common

/-!
Contract: positional ownership and whole-program column conservation for the
Lean-owned Phi81 ring-action rows.

Owns:
- one semantic receipt for every schoolbook-product or output row position;
- duplicate-free receipt order and exact reconstruction of the raw program;
- unique physical row identities after ownership is attached; and
- an exact support list containing authoritative reads and allocated products.

Does not own: placement into a selected NIFS call frame, activation, codecs,
the surrounding verifier, Rust, or generated artifacts.
-/

set_option autoImplicit false
set_option maxRecDepth 100000
set_option maxHeartbeats 2000000

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.Phi81RingAction

open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.DirectCalls
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation.PiRLCAlgebra.Norm

/-! ## Positional receipts -/

/-- A receipt names the semantic row position, not merely its numeric offset. -/
inductive RowOwner where
  | product (source left right : Nat)
  | output (lane : Nat)
deriving DecidableEq, Repr

/-- The raw equation denoted by one receipt. -/
def ownedRawRow {count : Nat} (frame : Frame count) : RowOwner → Row
  | .product source left right => productRow frame source left right
  | .output lane => outputRow frame lane

/-- Every semantic receipt in exact emitter order. -/
def allOwners (count : Nat) : List RowOwner :=
  ((List.range count).flatMap fun source =>
    (List.range ringDegree).flatMap fun left =>
      (List.range ringDegree).map fun right =>
        RowOwner.product source left right)
  ++ (List.range ringDegree).map RowOwner.output

theorem rawRows_eq_map_owners
    {count : Nat} (frame : Frame count) :
    rawRows frame = (allOwners count).map (ownedRawRow frame) := by
  unfold rawRows productRows outputRows allOwners
  simp only [List.map_append, List.map_flatMap, List.map_map,
    Function.comp_def, ownedRawRow]

private def rightOwners (source left : Nat) : List RowOwner :=
  (List.range ringDegree).map (RowOwner.product source left)

private def sourceOwners (source : Nat) : List RowOwner :=
  (List.range ringDegree).flatMap (rightOwners source)

private theorem rightOwnersUpTo_nodup (source left : Nat) :
    ∀ limit : Nat,
      ((List.range limit).map (RowOwner.product source left)).Nodup
  | 0 => by simp
  | limit + 1 => by
      rw [List.range_succ, List.map_append]
      refine List.nodup_append.2
        ⟨rightOwnersUpTo_nodup source left limit, by simp, ?_⟩
      intro existing existingMember added addedMember equal
      rcases List.mem_map.1 existingMember with
        ⟨right, rightMember, rfl⟩
      have addedExact :
          added = RowOwner.product source left limit := by
        simpa using addedMember
      subst added
      cases addedExact
      exact (Nat.ne_of_lt (List.mem_range.1 rightMember)) rfl

private theorem rightOwners_nodup (source left : Nat) :
    (rightOwners source left).Nodup :=
  rightOwnersUpTo_nodup source left ringDegree

private theorem outputOwners_nodup :
    ((List.range ringDegree).map RowOwner.output).Nodup := by
  let outputOwnersUpTo := fun limit =>
    (List.range limit).map RowOwner.output
  have complete :
      ∀ limit : Nat, (outputOwnersUpTo limit).Nodup := by
    intro limit
    induction limit with
    | zero =>
        simp [outputOwnersUpTo]
    | succ limit inductionHypothesis =>
        simp only [outputOwnersUpTo, List.range_succ, List.map_append]
        refine List.nodup_append.2
          ⟨inductionHypothesis, by simp, ?_⟩
        intro existing existingMember added addedMember equal
        rcases List.mem_map.1 existingMember with
          ⟨lane, laneMember, rfl⟩
        have addedExact : added = RowOwner.output limit := by
          simpa using addedMember
        subst added
        cases addedExact
        exact (Nat.ne_of_lt (List.mem_range.1 laneMember)) rfl
  exact complete ringDegree

private theorem leftBlocks_nodup (source : Nat) :
    ∀ limit : Nat,
      ((List.range limit).flatMap (rightOwners source)).Nodup
  | 0 => by simp
  | limit + 1 => by
      rw [List.range_succ, List.flatMap_append]
      refine List.nodup_append.2
        ⟨leftBlocks_nodup source limit, ?_, ?_⟩
      · simp only [List.flatMap_cons, List.flatMap_nil, List.append_nil]
        exact rightOwners_nodup source limit
      · intro leftOwner leftMember rightOwner rightMember equal
        rcases List.mem_flatMap.1 leftMember with
          ⟨left, leftInRange, leftOwnerMember⟩
        rcases List.mem_map.1 leftOwnerMember with
          ⟨leftRight, _, leftExact⟩
        simp only [List.flatMap_cons, List.flatMap_nil,
          List.append_nil] at rightMember
        rcases List.mem_map.1 rightMember with
          ⟨rightRight, _, rightExact⟩
        have leftEqLimit : left = limit := by
          rw [← leftExact, ← rightExact] at equal
          cases equal
          rfl
        have leftLt : left < limit :=
          List.mem_range.1 leftInRange
        omega

private theorem sourceOwners_nodup (source : Nat) :
    (sourceOwners source).Nodup := by
  unfold sourceOwners
  exact leftBlocks_nodup source ringDegree

private theorem productOwners_nodup (count : Nat) :
    ((List.range count).flatMap sourceOwners).Nodup := by
  induction count with
  | zero =>
      simp
  | succ count inductionHypothesis =>
      rw [List.range_succ, List.flatMap_append]
      refine List.nodup_append.2
        ⟨inductionHypothesis, ?_, ?_⟩
      · simp only [List.flatMap_cons, List.flatMap_nil, List.append_nil]
        exact sourceOwners_nodup count
      · intro leftOwner leftMember rightOwner rightMember equal
        rcases List.mem_flatMap.1 leftMember with
          ⟨source, sourceInRange, sourceOwnerMember⟩
        unfold sourceOwners at sourceOwnerMember
        rcases List.mem_flatMap.1 sourceOwnerMember with
          ⟨left, _, leftOwnerMember⟩
        rcases List.mem_map.1 leftOwnerMember with
          ⟨right, _, leftExact⟩
        simp only [List.flatMap_cons, List.flatMap_nil,
          List.append_nil] at rightMember
        unfold sourceOwners at rightMember
        rcases List.mem_flatMap.1 rightMember with
          ⟨otherLeft, _, otherLeftOwnerMember⟩
        rcases List.mem_map.1 otherLeftOwnerMember with
          ⟨otherRight, _, rightExact⟩
        have sourceEqCount : source = count := by
          rw [← leftExact, ← rightExact] at equal
          cases equal
          rfl
        have sourceLt : source < count :=
          List.mem_range.1 sourceInRange
        omega

theorem allOwners_nodup (count : Nat) :
    (allOwners count).Nodup := by
  unfold allOwners
  change
    ((List.range count).flatMap sourceOwners ++
      (List.range ringDegree).map RowOwner.output).Nodup
  rw [List.nodup_append]
  refine ⟨productOwners_nodup count,
    outputOwners_nodup,
    ?_⟩
  intro productOwner productMember outputOwner outputMember equal
  rcases List.mem_map.1 outputMember with ⟨lane, _, rfl⟩
  rcases List.mem_flatMap.1 productMember with
    ⟨source, _, sourceMember⟩
  unfold sourceOwners at sourceMember
  rcases List.mem_flatMap.1 sourceMember with
    ⟨left, _, leftMember⟩
  unfold rightOwners at leftMember
  rcases List.mem_map.1 leftMember with ⟨right, _, rfl⟩
  cases equal

theorem allOwners_length (count : Nat) :
    (allOwners count).length = rowCount count := by
  rw [← rawRows_length (frame := {
    owner := .prelude
    firstOrdinal := 0
    one := { owner := .prelude, bundleIndex := 0, coordinateIndex := 0 }
    challenges := fun _ _ => []
    values := fun _ _ => []
    output := fun _ => []
    productColumn := fun _ _ _ =>
      { owner := .prelude, bundleIndex := 0, coordinateIndex := 0 }
  })]
  rw [rawRows_eq_map_owners, List.length_map]

/-- Exactly one semantic receipt occupies every raw program position. -/
theorem ownership_is_positional
    {count : Nat} (frame : Frame count) :
    (rawRows frame).length = (allOwners count).length
      ∧ (allOwners count).Nodup
      ∧ rawRows frame = (allOwners count).map (ownedRawRow frame) := by
  refine ⟨?_, allOwners_nodup count, rawRows_eq_map_owners frame⟩
  rw [rawRows_length, allOwners_length]

private theorem ownRows_eq_direct
    (owner : PhysicalOwner)
    (ordinal : Nat)
    (source : List Row) :
    ownRows owner ordinal source =
      DirectCalls.ownRowsFrom owner ordinal source := by
  induction source generalizing ordinal with
  | nil =>
      rfl
  | cons head tail inductionHypothesis =>
      simp only [ownRows, DirectCalls.ownRowsFrom]
      rw [inductionHypothesis]

theorem rows_owned
    {count : Nat}
    (frame : Frame count)
    (row : OwnedRow)
    (member : row ∈ rows frame) :
    row.id.owner = frame.owner := by
  rw [rows, ownRows_eq_direct] at member
  exact DirectCalls.ownRowsFrom_owner frame.owner frame.firstOrdinal
    (rawRows frame) row member

theorem row_ids_nodup
    {count : Nat} (frame : Frame count) :
    ((rows frame).map fun row => row.id).Nodup := by
  rw [rows, ownRows_eq_direct]
  exact DirectCalls.ownRowsFrom_ids_nodup frame.owner frame.firstOrdinal
    (rawRows frame)

theorem owned_row_mem_rawRows
    {count : Nat}
    (frame : Frame count)
    (row : OwnedRow)
    (member : row ∈ rows frame) :
    row.row ∈ rawRows frame := by
  rw [rows, ownRows_eq_direct] at member
  exact DirectCalls.ownRowsFrom_row_mem frame.owner frame.firstOrdinal
    (rawRows frame) row member

/-! ## Exact support -/

/-- Ordered columns read by one sparse combination. -/
def combinationIds (combination : LinearCombination) : List ColumnId :=
  combination.map Term.column

/-- Ordered columns read by one carried ring. -/
def carriedIds (value : CarriedRing) : List ColumnId :=
  (List.finRange ringDegree).flatMap fun lane =>
    combinationIds (value lane)

/-- Ordered columns read by a finite carried-ring family. -/
def familyIds {count : Nat}
    (values : Fin count → CarriedRing) : List ColumnId :=
  (List.finRange count).flatMap fun source =>
    carriedIds (values source)

/-- Authoritative reads visible before this occurrence is completed. -/
def visibleIds {count : Nat} (frame : Frame count) : List ColumnId :=
  frame.one ::
    (familyIds frame.challenges ++
      familyIds frame.values ++ carriedIds frame.output)

/-- Every permitted dependency: authoritative reads, then allocated products. -/
def allowedIds {count : Nat} (frame : Frame count) : List ColumnId :=
  visibleIds frame ++ productIds frame

private theorem combination_column_mem
    (combination : LinearCombination)
    (term : Term)
    (member : term ∈ combination) :
    term.column ∈ combinationIds combination :=
  List.mem_map.2 ⟨term, member, rfl⟩

private theorem carried_column_mem
    (value : CarriedRing)
    (lane : Fin ringDegree)
    (term : Term)
    (member : term ∈ value lane) :
    term.column ∈ carriedIds value := by
  apply List.mem_flatMap.2
  exact ⟨lane, List.mem_finRange lane,
    combination_column_mem (value lane) term member⟩

private theorem family_column_mem
    {count : Nat}
    (values : Fin count → CarriedRing)
    (source : Fin count)
    (lane : Fin ringDegree)
    (term : Term)
    (member : term ∈ values source lane) :
    term.column ∈ familyIds values := by
  apply List.mem_flatMap.2
  exact ⟨source, List.mem_finRange source,
    carried_column_mem (values source) lane term member⟩

private theorem product_column_mem
    {count : Nat}
    (frame : Frame count)
    (source : Fin count)
    (left right : Fin ringDegree) :
    frame.productColumn source.val left.val right.val ∈ productIds frame := by
  unfold productIds
  apply List.mem_flatMap.2
  refine ⟨source.val, List.mem_range.2 source.isLt, ?_⟩
  apply List.mem_flatMap.2
  refine ⟨left.val, List.mem_range.2 left.isLt, ?_⟩
  apply List.mem_map.2
  exact ⟨right.val, List.mem_range.2 right.isLt, rfl⟩

private theorem tail_product_mem
    {count : Nat}
    (frame : Frame (count + 1))
    (column : ColumnId)
    (member : column ∈ productIds (tailFrame frame)) :
    column ∈ productIds frame := by
  simp only [productIds, tailFrame] at member ⊢
  rcases List.mem_flatMap.1 member with
    ⟨source, sourceMember, sourceRows⟩
  have sourceLt : source < count :=
    List.mem_range.1 sourceMember
  rcases List.mem_flatMap.1 sourceRows with
    ⟨left, leftMember, leftRows⟩
  rcases List.mem_map.1 leftRows with
    ⟨right, rightMember, rfl⟩
  apply List.mem_flatMap.2
  refine ⟨source + 1, List.mem_range.2 ?_, ?_⟩
  · exact Nat.succ_lt_succ sourceLt
  apply List.mem_flatMap.2
  refine ⟨left, leftMember, ?_⟩
  exact List.mem_map.2 ⟨right, rightMember, rfl⟩

private theorem sourceTerms_supported
    {count : Nat}
    (frame : Frame count)
    (source : Fin count)
    (degree : Nat)
    (indices : List Nat)
    (indicesBound : ∀ index ∈ indices, index < ringDegree)
    (term : Term)
    (member : term ∈ sourceTerms frame source.val degree indices) :
    term.column = frame.one ∨ term.column ∈ productIds frame := by
  unfold sourceTerms at member
  rcases List.mem_map.1 member with ⟨left, leftMember, rfl⟩
  by_cases active : Product.supportActive degree left
  · rw [if_pos active]
    right
    exact product_column_mem frame source
      ⟨left, indicesBound left leftMember⟩
      ⟨degree - left, active.2⟩
  · rw [if_neg active]
    exact Or.inl rfl

private theorem sourceRawCombination_supported
    {count : Nat}
    (frame : Frame count)
    (source : Fin count)
    (degree : Nat)
    (term : Term)
    (member : term ∈ sourceRawCombination frame source.val degree) :
    term.column = frame.one ∨ term.column ∈ productIds frame := by
  exact sourceTerms_supported frame source degree (List.range ringDegree)
    (fun index indexMember => List.mem_range.1 indexMember) term member

private theorem rawCombination_supported
    {count : Nat}
    (frame : Frame count)
    (degree : Nat)
    (term : Term)
    (member : term ∈ rawCombination frame degree) :
    term.column = frame.one ∨ term.column ∈ productIds frame := by
  induction count with
  | zero =>
      simp [rawCombination] at member
  | succ count inductionHypothesis =>
      rw [rawCombination, List.mem_append] at member
      rcases member with inHead | inTail
      · exact sourceRawCombination_supported frame 0 degree term inHead
      · rcases inductionHypothesis (tailFrame frame) inTail with
          wire | product
        · exact Or.inl wire
        · exact Or.inr (tail_product_mem frame term.column product)

private theorem negate_supported
    (combination : LinearCombination)
    (term : Term)
    (member : term ∈ negate combination) :
    ∃ original ∈ combination, term.column = original.column := by
  unfold negate at member
  rcases List.mem_map.1 member with ⟨original, originalMember, rfl⟩
  exact ⟨original, originalMember, rfl⟩

private theorem reducedCombination_supported
    {count : Nat}
    (frame : Frame count)
    (output : Nat)
    (term : Term)
    (member : term ∈ reducedCombination frame output) :
    term.column = frame.one ∨ term.column ∈ productIds frame := by
  unfold reducedCombination at member
  by_cases twiceActive : output + 81 ≤ 106
  · simp only [if_pos twiceActive, List.mem_append] at member
    rcases member with (inDirect | inFolded) | inTwice
    · exact rawCombination_supported frame output term inDirect
    · rcases negate_supported _ term inFolded with
        ⟨original, originalMember, sameColumn⟩
      rcases rawCombination_supported frame
          (if output < ringMiddleDegree then output + ringDegree
            else output + ringMiddleDegree)
          original originalMember with wire | product
      · exact Or.inl (sameColumn.trans wire)
      · exact Or.inr (sameColumn ▸ product)
    · exact rawCombination_supported frame (output + 81) term inTwice
  · simp only [if_neg twiceActive, List.mem_append, List.not_mem_nil,
      or_false] at member
    rcases member with inDirect | inFolded
    · exact rawCombination_supported frame output term inDirect
    · rcases negate_supported _ term inFolded with
        ⟨original, originalMember, sameColumn⟩
      rcases rawCombination_supported frame
          (if output < ringMiddleDegree then output + ringDegree
            else output + ringMiddleDegree)
          original originalMember with wire | product
      · exact Or.inl (sameColumn.trans wire)
      · exact Or.inr (sameColumn ▸ product)

private theorem one_visible
    {count : Nat} (frame : Frame count) :
    frame.one ∈ visibleIds frame :=
  List.mem_cons_self

private theorem challenge_column_visible
    {count : Nat}
    (frame : Frame count)
    (source : Fin count)
    (lane : Fin ringDegree)
    (term : Term)
    (member : term ∈ frame.challenges source lane) :
    term.column ∈ visibleIds frame := by
  apply List.mem_cons.2
  right
  apply List.mem_append.2
  left
  apply List.mem_append.2
  left
  exact family_column_mem frame.challenges source lane term member

private theorem value_column_visible
    {count : Nat}
    (frame : Frame count)
    (source : Fin count)
    (lane : Fin ringDegree)
    (term : Term)
    (member : term ∈ frame.values source lane) :
    term.column ∈ visibleIds frame := by
  apply List.mem_cons.2
  right
  apply List.mem_append.2
  left
  apply List.mem_append.2
  right
  exact family_column_mem frame.values source lane term member

private theorem output_column_visible
    {count : Nat}
    (frame : Frame count)
    (lane : Fin ringDegree)
    (term : Term)
    (member : term ∈ frame.output lane) :
    term.column ∈ visibleIds frame := by
  apply List.mem_cons.2
  right
  apply List.mem_append.2
  right
  exact carried_column_mem frame.output lane term member

private theorem productRow_supported
    {count : Nat}
    (frame : Frame count)
    (source : Fin count)
    (left right : Fin ringDegree)
    (column : ColumnId)
    (member :
      column ∈ (productRow frame source.val left.val right.val).columnIds) :
    column ∈ allowedIds frame := by
  unfold Row.columnIds at member
  rcases List.mem_map.1 member with
    ⟨term, termMember, sameColumn⟩
  subst column
  simp only [productRow, dif_pos source.isLt, dif_pos left.isLt,
    dif_pos right.isLt] at termMember
  change
    term ∈
      frame.challenges source left ++ frame.values source right ++
        Goldilocks.singleton
          (frame.productColumn source.val left.val right.val) 1
    at termMember
  rw [List.mem_append] at termMember
  rcases termMember with inInputs | inProduct
  · rw [List.mem_append] at inInputs
    rcases inInputs with inChallenge | inValue
    · exact List.mem_append.2
        (Or.inl (challenge_column_visible frame source left term inChallenge))
    · exact List.mem_append.2
        (Or.inl (value_column_visible frame source right term inValue))
  · simp only [Goldilocks.singleton, List.mem_singleton] at inProduct
    subst term
    exact List.mem_append.2
      (Or.inr (product_column_mem frame source left right))

private theorem outputRow_supported
    {count : Nat}
    (frame : Frame count)
    (output : Fin ringDegree)
    (column : ColumnId)
    (member : column ∈ (outputRow frame output.val).columnIds) :
    column ∈ allowedIds frame := by
  unfold Row.columnIds at member
  rcases List.mem_map.1 member with
    ⟨term, termMember, sameColumn⟩
  subst column
  simp only [outputRow, dif_pos output.isLt] at termMember
  change
    term ∈ reducedCombination frame output.val ++
      Goldilocks.singleton frame.one 1 ++ frame.output output
    at termMember
  rw [List.mem_append] at termMember
  rcases termMember with inLeft | inOutput
  · rw [List.mem_append] at inLeft
    rcases inLeft with inReduced | inOne
    · rcases reducedCombination_supported frame output.val term inReduced with
        wire | product
      · exact List.mem_append.2 (Or.inl (wire ▸ one_visible frame))
      · exact List.mem_append.2 (Or.inr product)
    · simp only [Goldilocks.singleton, List.mem_singleton] at inOne
      subst term
      exact List.mem_append.2 (Or.inl (one_visible frame))
  · exact List.mem_append.2
      (Or.inl (output_column_visible frame output term inOutput))

/-- Every raw row dependency is an authoritative visible read or an exact
product-cell allocation. -/
theorem rawRows_supported
    {count : Nat}
    (frame : Frame count)
    (row : Row)
    (rowMember : row ∈ rawRows frame)
    (column : ColumnId)
    (columnMember : column ∈ row.columnIds) :
    column ∈ allowedIds frame := by
  rcases List.mem_append.1 rowMember with inProduct | inOutput
  · unfold productRows at inProduct
    rcases List.mem_flatMap.1 inProduct with
      ⟨source, sourceMember, sourceRows⟩
    rcases List.mem_flatMap.1 sourceRows with
      ⟨left, leftMember, leftRows⟩
    rcases List.mem_map.1 leftRows with
      ⟨right, rightMember, rowExact⟩
    subst row
    exact productRow_supported frame
      ⟨source, List.mem_range.1 sourceMember⟩
      ⟨left, List.mem_range.1 leftMember⟩
      ⟨right, List.mem_range.1 rightMember⟩
      column columnMember
  · unfold outputRows at inOutput
    rcases List.mem_map.1 inOutput with
      ⟨output, outputMember, rowExact⟩
    subst row
    exact outputRow_supported frame
      ⟨output, List.mem_range.1 outputMember⟩
      column columnMember

/-- Every declared product cell is constrained by its own schoolbook product
row.  This is the converse missing from support-only conservation: appending an
unused product column to `productIds` would make this theorem false. -/
theorem productIds_written
    {count : Nat}
    (frame : Frame count)
    (column : ColumnId)
    (member : column ∈ productIds frame) :
    ∃ row ∈ rawRows frame, column ∈ row.columnIds := by
  unfold productIds at member
  rcases List.mem_flatMap.1 member with
    ⟨source, sourceMember, sourceColumns⟩
  rcases List.mem_flatMap.1 sourceColumns with
    ⟨left, leftMember, leftColumns⟩
  rcases List.mem_map.1 leftColumns with
    ⟨right, rightMember, rfl⟩
  refine ⟨productRow frame source left right, ?_, ?_⟩
  · unfold rawRows productRows
    apply List.mem_append_left
    apply List.mem_flatMap.2
    refine ⟨source, sourceMember, ?_⟩
    apply List.mem_flatMap.2
    exact ⟨left, leftMember,
      List.mem_map.2 ⟨right, rightMember, rfl⟩⟩
  · simp [Row.columnIds, productRow, Goldilocks.singleton]

/-- **Whole-program conservation.** No owned row touches anything outside the
declared authoritative reads and exact product allocation. -/
theorem rows_supported
    {count : Nat}
    (frame : Frame count)
    (row : OwnedRow)
    (rowMember : row ∈ rows frame)
    (column : ColumnId)
    (columnMember : column ∈ row.columnIds) :
    column ∈ allowedIds frame :=
  rawRows_supported frame row.row
    (owned_row_mem_rawRows frame row rowMember)
    column columnMember

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.Phi81RingAction
