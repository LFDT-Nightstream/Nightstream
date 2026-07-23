import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.DirectCalls.Common

/-!
Contract: activation-aware unary affine coordinate maps for direct fixed-one
calls.

Owns:
- an executable affine map between two canonical codec strings;
- one gated residual row per output coordinate;
- artifact-independent active soundness, honest completeness, inactive
  satisfiability, and exact row count.

Does not own: a claim that a protocol function is affine.  A caller must prove
the exact encoder equation for the selected semantic function.

Emits constraints: exactly the target codec width, with no temporary columns.
-/

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.DirectCalls

open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.Implementation.Lowering.Typed

universe u v

/-- Head-first dot product.  Unequal tails contribute nothing; every selected
map separately proves exact source width. -/
def fieldDot : List Field -> List Field -> Field
  | coefficient :: coefficients, coordinate :: coordinates =>
      coefficient * coordinate + fieldDot coefficients coordinates
  | _, _ => 0

/-- One affine output coordinate. -/
structure AffineCoordinate where
  constant : Field
  coefficients : List Field
deriving DecidableEq, Repr

namespace AffineCoordinate

def eval (coordinate : AffineCoordinate) (source : List Field) : Field :=
  coordinate.constant + fieldDot coordinate.coefficients source

end AffineCoordinate

/-- Exact executable coordinate graph of one unary semantic function. -/
structure AffineEncodingMap
    {α : Type u}
    {β : Type v}
    (source : Codec α)
    (target : Codec β)
    (function : α -> β) where
  coordinates : List AffineCoordinate
  coordinateCount : coordinates.length = target.width
  coefficientCounts :
    ∀ coordinate ∈ coordinates,
      coordinate.coefficients.length = source.width
  outputAdmissible :
    ∀ value, source.Admissible value ->
      target.Admissible (function value)
  encode_eq :
    ∀ value, source.Admissible value ->
      target.encode (function value) =
        coordinates.map (fun coordinate =>
          coordinate.eval (source.encode value))

private def affineTerms :
    List OwnedColumn -> List Field -> LinearCombination
  | column :: columns, coefficient :: coefficients =>
      { column := column.id, coefficient := coefficient } ::
        affineTerms columns coefficients
  | _, _ => []

private theorem affineTerms_eval
    (columns : List OwnedColumn)
    (coefficients : List Field)
    (assignment : ColumnId -> Field)
    (lengthEqual : columns.length = coefficients.length) :
    (affineTerms columns coefficients).eval assignment =
      fieldDot coefficients
        (columns.map fun column => assignment column.id) := by
  induction columns generalizing coefficients with
  | nil =>
      cases coefficients <;> simp [affineTerms, fieldDot]
  | cons column columns inductionHypothesis =>
      cases coefficients with
      | nil =>
          simp at lengthEqual
      | cons coefficient coefficients =>
          simp only [List.length_cons, Nat.succ.injEq] at lengthEqual
          simp only [affineTerms, LinearCombination.eval, fieldDot,
            List.map_cons]
          rw [inductionHypothesis coefficients lengthEqual]

private theorem affineTerms_column_mem
    (columns : List OwnedColumn)
    (coefficients : List Field)
    (column : ColumnId)
    (member :
      column ∈
        (affineTerms columns coefficients).map fun term => term.column) :
    column ∈ columns.map fun item => item.id := by
  induction columns generalizing coefficients with
  | nil =>
      cases coefficients <;> simp [affineTerms] at member
  | cons head tail inductionHypothesis =>
      cases coefficients with
      | nil =>
          simp [affineTerms] at member
      | cons coefficient coefficients =>
          simp only [affineTerms, List.map_cons, List.mem_cons] at member ⊢
          rcases member with equal | tailMember
          · exact Or.inl equal
          · exact Or.inr
              (inductionHypothesis coefficients tailMember)

private theorem linearCombination_eval_append
    (left right : LinearCombination)
    (assignment : ColumnId -> Field) :
    (left ++ right).eval assignment =
      left.eval assignment + right.eval assignment := by
  induction left with
  | nil =>
      simp
  | cons term terms inductionHypothesis =>
      simp only [List.cons_append, LinearCombination.eval,
        inductionHypothesis, Lean.Grind.Fin.add_assoc]

private def affineResidual
    (one output : ColumnId)
    (source : List OwnedColumn)
    (coordinate : AffineCoordinate) : LinearCombination :=
  { column := one, coefficient := coordinate.constant } ::
    affineTerms source coordinate.coefficients ++
      [{ column := output, coefficient := -1 }]

private theorem affineResidual_eval
    (one output : ColumnId)
    (source : List OwnedColumn)
    (coordinate : AffineCoordinate)
    (assignment : ColumnId -> Field)
    (constantOne : assignment one = 1)
    (lengthEqual :
      source.length = coordinate.coefficients.length) :
    (affineResidual one output source coordinate).eval assignment =
      coordinate.eval
          (source.map fun column => assignment column.id) -
        assignment output := by
  unfold affineResidual AffineCoordinate.eval
  rw [linearCombination_eval_append, LinearCombination.eval_cons,
    affineTerms_eval source coordinate.coefficients assignment lengthEqual]
  simp only [LinearCombination.eval, constantOne, Fin.mul_one, Fin.one_mul,
    Fin.add_zero,
    Lean.Grind.Fin.neg_mul, Fin.sub_eq_add_neg,
    Lean.Grind.Fin.add_assoc]

private def affineRowsFrom
    (owner : PhysicalOwner)
    (one active : ColumnId)
    (source : List OwnedColumn) :
    Nat -> List OwnedColumn -> List AffineCoordinate -> List OwnedRow
  | _, [], _ => []
  | _, _ :: _, [] => []
  | ordinal, output :: outputs, coordinate :: coordinates =>
      { id := { owner := owner, ordinal := ordinal }
        row :=
          { a := singleton active 1
            b := affineResidual one output.id source coordinate
            c := [] } } ::
        affineRowsFrom owner one active source (ordinal + 1)
          outputs coordinates

private theorem affineRowsFrom_length
    (owner : PhysicalOwner)
    (one active : ColumnId)
    (source : List OwnedColumn)
    (ordinal : Nat)
    (outputs : List OwnedColumn)
    (coordinates : List AffineCoordinate)
    (lengthEqual : outputs.length = coordinates.length) :
    (affineRowsFrom owner one active source ordinal outputs coordinates).length =
      outputs.length := by
  induction outputs generalizing ordinal coordinates with
  | nil =>
      cases coordinates with
      | nil =>
          simp [affineRowsFrom]
      | cons coordinate coordinates =>
          simp at lengthEqual
  | cons output outputs inductionHypothesis =>
      cases coordinates with
      | nil =>
          simp at lengthEqual
      | cons coordinate coordinates =>
          simp only [List.length_cons, Nat.succ.injEq] at lengthEqual
          simp only [affineRowsFrom, List.length_cons]
          rw [inductionHypothesis (ordinal := ordinal + 1)
            (coordinates := coordinates) lengthEqual]

private theorem affineRowsFrom_owner
    (owner : PhysicalOwner)
    (one active : ColumnId)
    (source : List OwnedColumn)
    (ordinal : Nat)
    (outputs : List OwnedColumn)
    (coordinates : List AffineCoordinate)
    (row : OwnedRow)
    (member :
      row ∈
        affineRowsFrom owner one active source ordinal outputs coordinates) :
    row.id.owner = owner := by
  induction outputs generalizing ordinal coordinates with
  | nil =>
      cases coordinates <;> simp [affineRowsFrom] at member
  | cons output outputs inductionHypothesis =>
      cases coordinates with
      | nil =>
          simp [affineRowsFrom] at member
      | cons coordinate coordinates =>
          simp only [affineRowsFrom, List.mem_cons] at member
          rcases member with equal | tail
          · subst row
            rfl
          · exact inductionHypothesis (ordinal + 1) coordinates tail

private theorem affineRowsFrom_ordinal_lower_bound
    (owner : PhysicalOwner)
    (one active : ColumnId)
    (source : List OwnedColumn)
    (ordinal : Nat)
    (outputs : List OwnedColumn)
    (coordinates : List AffineCoordinate)
    (row : OwnedRow)
    (member :
      row ∈
        affineRowsFrom owner one active source ordinal outputs coordinates) :
    ordinal ≤ row.id.ordinal := by
  induction outputs generalizing ordinal coordinates with
  | nil =>
      cases coordinates <;> simp [affineRowsFrom] at member
  | cons output outputs inductionHypothesis =>
      cases coordinates with
      | nil =>
          simp [affineRowsFrom] at member
      | cons coordinate coordinates =>
          simp only [affineRowsFrom, List.mem_cons] at member
          rcases member with equal | tail
          · subst row
            exact Nat.le_refl ordinal
          · have lower :=
              inductionHypothesis (ordinal + 1) coordinates tail
            omega

private theorem affineRowsFrom_ids_nodup
    (owner : PhysicalOwner)
    (one active : ColumnId)
    (source : List OwnedColumn)
    (ordinal : Nat)
    (outputs : List OwnedColumn)
    (coordinates : List AffineCoordinate) :
    ((affineRowsFrom owner one active source ordinal outputs coordinates).map
      fun row => row.id).Nodup := by
  induction outputs generalizing ordinal coordinates with
  | nil =>
      cases coordinates <;> simp [affineRowsFrom]
  | cons output outputs inductionHypothesis =>
      cases coordinates with
      | nil =>
          simp [affineRowsFrom]
      | cons coordinate coordinates =>
          simp only [affineRowsFrom, List.map_cons, List.nodup_cons]
          constructor
          · intro member
            have mapped :
                ∃ row ∈ affineRowsFrom owner one active source
                    (ordinal + 1) outputs coordinates,
                  row.id = { owner := owner, ordinal := ordinal } := by
              simpa only [List.mem_map] using member
            rcases mapped with ⟨row, rowMember, equal⟩
            have lower :=
              affineRowsFrom_ordinal_lower_bound owner one active source
                (ordinal + 1) outputs coordinates row rowMember
            have ordinalEqual : row.id.ordinal = ordinal :=
              congrArg RowId.ordinal equal
            omega
          · exact inductionHypothesis (ordinal + 1) coordinates

private theorem affineRowsFrom_active_iff
    (owner : PhysicalOwner)
    (one active : ColumnId)
    (source : List OwnedColumn)
    (ordinal : Nat)
    (outputs : List OwnedColumn)
    (coordinates : List AffineCoordinate)
    (assignment : ColumnId -> Field)
    (constantOne : assignment one = 1)
    (activeOne : assignment active = 1)
    (lengthEqual : outputs.length = coordinates.length)
    (coefficientCounts :
      ∀ coordinate ∈ coordinates,
        coordinate.coefficients.length = source.length) :
    Satisfies
        (affineRowsFrom owner one active source ordinal outputs coordinates)
        assignment ↔
      outputs.map (fun output => assignment output.id) =
        coordinates.map (fun coordinate =>
          coordinate.eval
            (source.map fun column => assignment column.id)) := by
  induction outputs generalizing ordinal coordinates with
  | nil =>
      cases coordinates with
      | nil =>
          simp [affineRowsFrom]
      | cons coordinate coordinates =>
          simp at lengthEqual
  | cons output outputs inductionHypothesis =>
      cases coordinates with
      | nil =>
          simp at lengthEqual
      | cons coordinate coordinates =>
          simp only [List.length_cons, Nat.succ.injEq] at lengthEqual
          have headCount :
              coordinate.coefficients.length = source.length :=
            coefficientCounts coordinate (List.mem_cons_self)
          have tailCounts :
              ∀ item ∈ coordinates,
                item.coefficients.length = source.length := by
            intro item member
            exact coefficientCounts item
              (List.mem_cons_of_mem coordinate member)
          have residual :
              (affineResidual one output.id source coordinate).eval
                  assignment =
                coordinate.eval
                    (source.map fun column => assignment column.id) -
                  assignment output.id :=
            affineResidual_eval one output.id source coordinate assignment
              constantOne headCount.symm
          have activeEval :
              (Goldilocks.singleton active 1).eval assignment = 1 := by
            simp only [Goldilocks.singleton, LinearCombination.eval,
              activeOne, Fin.one_mul, Fin.add_zero]
          rw [affineRowsFrom]
          simp only [satisfies_cons, Row.Holds, activeEval,
            LinearCombination.eval_nil, Fin.one_mul, residual,
            List.map_cons, List.cons.injEq]
          rw [Lean.Grind.AddCommGroup.sub_eq_zero_iff]
          rw [inductionHypothesis
            (ordinal := ordinal + 1)
            (coordinates := coordinates)
            lengthEqual tailCounts]
          constructor
          · rintro ⟨head, tail⟩
            exact ⟨head.symm, tail⟩
          · rintro ⟨head, tail⟩
            exact ⟨head.symm, tail⟩

private theorem affineRowsFrom_inactive
    (owner : PhysicalOwner)
    (one active : ColumnId)
    (source : List OwnedColumn)
    (ordinal : Nat)
    (outputs : List OwnedColumn)
    (coordinates : List AffineCoordinate)
    (assignment : ColumnId -> Field)
    (activeZero : assignment active = 0) :
    Satisfies
      (affineRowsFrom owner one active source ordinal outputs coordinates)
      assignment := by
  induction outputs generalizing ordinal coordinates with
  | nil =>
      cases coordinates <;> simp [affineRowsFrom]
  | cons output outputs inductionHypothesis =>
      cases coordinates with
      | nil =>
          simp [affineRowsFrom]
      | cons coordinate coordinates =>
          have activeEval :
              (Goldilocks.singleton active 1).eval assignment = 0 := by
            simp only [Goldilocks.singleton, LinearCombination.eval,
              activeZero, Fin.one_mul, Fin.add_zero]
          rw [affineRowsFrom]
          exact ⟨by
              simp only [Row.Holds, activeEval,
                LinearCombination.eval_nil, Fin.zero_mul],
            inductionHypothesis (ordinal := ordinal + 1)
              (coordinates := coordinates)⟩

private theorem affineRowsFrom_supported
    (owner : PhysicalOwner)
    (one active : ColumnId)
    (source : List OwnedColumn)
    (ordinal : Nat)
    (outputs : List OwnedColumn)
    (coordinates : List AffineCoordinate)
    (row : OwnedRow)
    (member :
      row ∈
        affineRowsFrom owner one active source ordinal outputs coordinates)
    (column : ColumnId)
    (columnMember : column ∈ row.columnIds) :
    column = one ∨ column = active ∨
      column ∈ source.map (fun item => item.id) ∨
      column ∈ outputs.map (fun item => item.id) := by
  induction outputs generalizing ordinal coordinates with
  | nil =>
      cases coordinates <;> simp [affineRowsFrom] at member
  | cons output outputs inductionHypothesis =>
      cases coordinates with
      | nil =>
          simp [affineRowsFrom] at member
      | cons coordinate coordinates =>
          simp only [affineRowsFrom, List.mem_cons] at member
          rcases member with equal | tailMember
          · subst row
            simp [OwnedRow.columnIds, Row.columnIds,
              Goldilocks.singleton, affineResidual] at columnMember
            rcases columnMember with
              activeEqual | oneEqual | sourceOrOutput
            · exact Or.inr (Or.inl activeEqual)
            · exact Or.inl oneEqual
            · rcases sourceOrOutput with sourceMember | outputEqual
              · exact Or.inr (Or.inr (Or.inl
                  (affineTerms_column_mem source
                    coordinate.coefficients column sourceMember)))
              · subst column
                exact Or.inr (Or.inr (Or.inr List.mem_cons_self))
          · rcases inductionHypothesis
                (ordinal := ordinal + 1) (coordinates := coordinates)
                tailMember columnMember with
              oneEqual | activeEqual | sourceMember | outputMember
            · exact Or.inl oneEqual
            · exact Or.inr (Or.inl activeEqual)
            · exact Or.inr (Or.inr (Or.inl sourceMember))
            · exact Or.inr (Or.inr (Or.inr (by
                simpa only [List.map_cons] using
                  List.mem_cons_of_mem output.id outputMember)))

/-- Physical occurrence of one exact unary affine coordinate map. -/
structure AffineMapRecipe
    {α : Type u}
    {β : Type v}
    {sourceCodec : Codec α}
    {targetCodec : Codec β}
    {function : α -> β}
    (map : AffineEncodingMap sourceCodec targetCodec function)
    (sourceLayout targetLayout : Layout) where
  owner : PhysicalOwner
  firstOrdinal : Nat
  one : ColumnId
  active : ColumnId
  source : ColumnBundle sourceLayout
  output : ColumnBundle targetLayout
  sourceWidth : sourceCodec.width = sourceLayout.owners.length
  targetWidth : targetCodec.width = targetLayout.owners.length

namespace AffineMapRecipe

def rows
    {α : Type u}
    {β : Type v}
    {sourceCodec : Codec α}
    {targetCodec : Codec β}
    {function : α -> β}
    {map : AffineEncodingMap sourceCodec targetCodec function}
    {sourceLayout targetLayout : Layout}
    (recipe : AffineMapRecipe map sourceLayout targetLayout) :
    List OwnedRow :=
  affineRowsFrom recipe.owner recipe.one recipe.active
    recipe.source.columns recipe.firstOrdinal recipe.output.columns
    map.coordinates

theorem row_count
    {α : Type u}
    {β : Type v}
    {sourceCodec : Codec α}
    {targetCodec : Codec β}
    {function : α -> β}
    {map : AffineEncodingMap sourceCodec targetCodec function}
    {sourceLayout targetLayout : Layout}
    (recipe : AffineMapRecipe map sourceLayout targetLayout) :
    recipe.rows.length = targetCodec.width := by
  calc
    recipe.rows.length = recipe.output.columns.length :=
      affineRowsFrom_length recipe.owner recipe.one recipe.active
        recipe.source.columns recipe.firstOrdinal recipe.output.columns
        map.coordinates (by
          rw [recipe.output.length_eq, ← recipe.targetWidth,
            map.coordinateCount])
    _ = targetLayout.owners.length := recipe.output.length_eq
    _ = targetCodec.width := recipe.targetWidth.symm

theorem rows_owned
    {α : Type u}
    {β : Type v}
    {sourceCodec : Codec α}
    {targetCodec : Codec β}
    {function : α -> β}
    {map : AffineEncodingMap sourceCodec targetCodec function}
    {sourceLayout targetLayout : Layout}
    (recipe : AffineMapRecipe map sourceLayout targetLayout)
    (row : OwnedRow)
    (member : row ∈ recipe.rows) :
    row.id.owner = recipe.owner :=
  affineRowsFrom_owner recipe.owner recipe.one recipe.active
    recipe.source.columns recipe.firstOrdinal recipe.output.columns
    map.coordinates row member

theorem row_ids_nodup
    {α : Type u}
    {β : Type v}
    {sourceCodec : Codec α}
    {targetCodec : Codec β}
    {function : α -> β}
    {map : AffineEncodingMap sourceCodec targetCodec function}
    {sourceLayout targetLayout : Layout}
    (recipe : AffineMapRecipe map sourceLayout targetLayout) :
    (recipe.rows.map fun row => row.id).Nodup :=
  affineRowsFrom_ids_nodup recipe.owner recipe.one recipe.active
    recipe.source.columns recipe.firstOrdinal recipe.output.columns
    map.coordinates

theorem active_sound
    {α : Type u}
    {β : Type v}
    {sourceCodec : Codec α}
    {targetCodec : Codec β}
    {function : α -> β}
    {map : AffineEncodingMap sourceCodec targetCodec function}
    {sourceLayout targetLayout : Layout}
    (recipe : AffineMapRecipe map sourceLayout targetLayout)
    (assignment : ColumnId -> Field)
    (value : α)
    (constantOne : assignment recipe.one = 1)
    (activeOne : assignment recipe.active = 1)
    (sourceDecoded :
      sourceCodec.decode (recipe.source.values assignment) = some value)
    (holds : Satisfies recipe.rows assignment) :
    targetCodec.decode (recipe.output.values assignment) =
      some (function value) := by
  have sourceExact :=
    (sourceCodec.encode_decode
      (recipe.source.values assignment) value sourceDecoded).2
  have sourceAdmissible :=
    sourceCodec.admissible_of_decode sourceDecoded
  have coefficientCounts :
      ∀ coordinate ∈ map.coordinates,
        coordinate.coefficients.length = recipe.source.columns.length := by
    intro coordinate member
    rw [map.coefficientCounts coordinate member,
      recipe.source.length_eq, ← recipe.sourceWidth]
  have outputCoordinates :
      recipe.output.values assignment =
        map.coordinates.map (fun coordinate =>
          coordinate.eval
            (recipe.source.values assignment)) := by
    apply (affineRowsFrom_active_iff
      recipe.owner recipe.one recipe.active recipe.source.columns
      recipe.firstOrdinal recipe.output.columns map.coordinates
      assignment constantOne activeOne
      (by rw [recipe.output.length_eq, ← recipe.targetWidth,
        map.coordinateCount])
      coefficientCounts).mp
    exact holds
  have encoded :
      recipe.output.values assignment =
        targetCodec.encode (function value) := by
    rw [outputCoordinates, ← sourceExact]
    exact (map.encode_eq value sourceAdmissible).symm
  rw [encoded]
  exact targetCodec.decode_encode (function value)
    (map.outputAdmissible value sourceAdmissible)

theorem active_complete
    {α : Type u}
    {β : Type v}
    {sourceCodec : Codec α}
    {targetCodec : Codec β}
    {function : α -> β}
    {map : AffineEncodingMap sourceCodec targetCodec function}
    {sourceLayout targetLayout : Layout}
    (recipe : AffineMapRecipe map sourceLayout targetLayout)
    (assignment : ColumnId -> Field)
    (value : α)
    (constantOne : assignment recipe.one = 1)
    (activeOne : assignment recipe.active = 1)
    (sourceCoordinates :
      recipe.source.values assignment = sourceCodec.encode value)
    (outputCoordinates :
      recipe.output.values assignment =
        targetCodec.encode (function value))
    (sourceAdmissible : sourceCodec.Admissible value) :
    Satisfies recipe.rows assignment := by
  apply (affineRowsFrom_active_iff
    recipe.owner recipe.one recipe.active recipe.source.columns
    recipe.firstOrdinal recipe.output.columns map.coordinates
    assignment constantOne activeOne
    (by rw [recipe.output.length_eq, ← recipe.targetWidth,
      map.coordinateCount])
    (by
      intro coordinate member
      rw [map.coefficientCounts coordinate member,
        recipe.source.length_eq, ← recipe.sourceWidth])).mpr
  rw [show recipe.output.columns.map
      (fun output => assignment output.id) =
        recipe.output.values assignment by rfl,
    outputCoordinates, map.encode_eq value sourceAdmissible,
    ← sourceCoordinates]
  rfl

theorem inactive_complete
    {α : Type u}
    {β : Type v}
    {sourceCodec : Codec α}
    {targetCodec : Codec β}
    {function : α -> β}
    {map : AffineEncodingMap sourceCodec targetCodec function}
    {sourceLayout targetLayout : Layout}
    (recipe : AffineMapRecipe map sourceLayout targetLayout)
    (assignment : ColumnId -> Field)
    (activeZero : assignment recipe.active = 0) :
    Satisfies recipe.rows assignment :=
  affineRowsFrom_inactive recipe.owner recipe.one recipe.active
    recipe.source.columns recipe.firstOrdinal recipe.output.columns
    map.coordinates assignment activeZero

theorem rows_supported
    {α : Type u}
    {β : Type v}
    {sourceCodec : Codec α}
    {targetCodec : Codec β}
    {function : α -> β}
    {map : AffineEncodingMap sourceCodec targetCodec function}
    {sourceLayout targetLayout : Layout}
    (recipe : AffineMapRecipe map sourceLayout targetLayout)
    (row : OwnedRow)
    (member : row ∈ recipe.rows)
    (column : ColumnId)
    (columnMember : column ∈ row.columnIds) :
    column = recipe.one ∨ column = recipe.active ∨
      column ∈ recipe.source.ids ∨
      column ∈ recipe.output.ids := by
  exact affineRowsFrom_supported recipe.owner recipe.one recipe.active
    recipe.source.columns recipe.firstOrdinal recipe.output.columns
    map.coordinates row member column columnMember

end AffineMapRecipe

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.DirectCalls
