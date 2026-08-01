import Nightstream.Implementation.Lowering.Goldilocks.Optimization.Manifest
import Nightstream.Implementation.Lowering.Goldilocks.Optimization.Pass

/-!
Contract: remove auxiliary allocations that no emitted row or protected
external boundary uses.

Assurance tier: model-level.

Owns: exact row preservation, exact decoded acceptance, committed and public
allocation preservation, and protected auxiliary retention.

Does not own: row removal, liveness across a protocol boundary not supplied
in `protected`, column renumbering, or Rust.

Emits constraints: the same normalized rows with a reduced allocation list.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.Lowering.Goldilocks.Optimization.Passes.PruneAllocations

open Nightstream.Implementation.Lowering.Typed
open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.Implementation.Lowering.Goldilocks.CanonicalManifest

universe u

private abbrev Assignment := R1CS.Assignment

def combinationColumns
    (combination : ManifestCombination) : List ColumnId :=
  combination.map ManifestTerm.column

def rowColumns (row : ManifestRow) : List ColumnId :=
  combinationColumns row.a ++
    combinationColumns row.b ++
      combinationColumns row.c

def mentionedColumns (program : CanonicalManifest.Program) : List ColumnId :=
  program.rows.flatMap rowColumns

def Keep
    (protectedIds mentioned : List ColumnId)
    (column : OwnedColumn) : Prop :=
  column.ownership != .auxiliaryColumn \/
    column.id ∈ protectedIds \/
      column.id ∈ mentioned

instance keepDecidable
    (protectedIds mentioned : List ColumnId)
    (column : OwnedColumn) :
    Decidable (Keep protectedIds mentioned column) := by
  unfold Keep
  infer_instance

def pruneReceipt
    (protectedIds mentioned : List ColumnId)
    (receipt : ManifestReceipt) : ManifestReceipt where
  owner := receipt.owner
  kind := receipt.kind
  allocations :=
    receipt.allocations.filter fun column =>
      decide (Keep protectedIds mentioned column)
  rows := receipt.rows

def program
    (protectedIds : List ColumnId)
    (source : CanonicalManifest.Program) : CanonicalManifest.Program where
  one := source.one
  receipts :=
    source.receipts.map
      (pruneReceipt protectedIds (mentionedColumns source))

@[simp] theorem rows_exact
    (protectedIds : List ColumnId)
    (source : CanonicalManifest.Program) :
    (program protectedIds source).rows = source.rows := by
  simp [program, CanonicalManifest.Program.rows, pruneReceipt,
    List.flatMap_map]

@[simp] theorem decoded_rows_exact
    (protectedIds : List ColumnId)
    (source : CanonicalManifest.Program) :
    (program protectedIds source).decode.rows =
      source.decode.rows := by
  simp [program, CanonicalManifest.Program.decode,
    CanonicalManifest.ProgramImage.rows, pruneReceipt,
    ManifestReceipt.decode, List.flatMap_map]

theorem decoded_satisfies_iff
    (protectedIds : List ColumnId)
    (source : CanonicalManifest.Program)
    (assignment : Assignment) :
    Goldilocks.Satisfies
        (program protectedIds source).decode.rows assignment <->
      Goldilocks.Satisfies source.decode.rows assignment := by
  rw [decoded_rows_exact]

def target
    {Observable : Type u}
    (protectedIds : List ColumnId)
    (source : CanonicalManifest.Program)
    (observe : Assignment -> Observable) :=
  Manifest.decodedSystem (program protectedIds source) observe

def replacement
    {Observable : Type u}
    (protectedIds : List ColumnId)
    (source : CanonicalManifest.Program)
    (observe : Assignment -> Observable)
    (degreeLimit : Nat)
    (withinLimit : R1CS.degree <= degreeLimit) :
    Optimization.Replacement
      (Manifest.decodedSystem source observe)
      (target protectedIds source observe)
      degreeLimit where
  recover := fun assignment => assignment
  derive := fun assignment => assignment
  sound := by
    intro assignment accepted
    exact ⟨accepted.1,
      (decoded_satisfies_iff protectedIds source assignment).mp accepted.2⟩
  complete := by
    intro assignment accepted
    exact ⟨accepted.1,
      (decoded_satisfies_iff protectedIds source assignment).mpr accepted.2⟩
  recover_observes := fun _ _ => rfl
  derive_observes := fun _ _ => rfl
  source_degree := withinLimit
  target_degree := withinLimit

theorem retained_from_source
    (protectedIds mentioned : List ColumnId)
    (receipt : ManifestReceipt)
    (column : OwnedColumn)
    (member :
      column ∈ (pruneReceipt protectedIds mentioned receipt).allocations) :
    column ∈ receipt.allocations := by
  exact (List.mem_filter.1 member).1

theorem committed_kept
    (protectedIds mentioned : List ColumnId)
    (receipt : ManifestReceipt)
    (column : OwnedColumn)
    (member : column ∈ receipt.allocations)
    (committed : column.ownership = .committedColumn) :
    column ∈ (pruneReceipt protectedIds mentioned receipt).allocations := by
  apply List.mem_filter.2
  exact ⟨member, by simp [Keep, committed]⟩

theorem public_kept
    (protectedIds mentioned : List ColumnId)
    (receipt : ManifestReceipt)
    (column : OwnedColumn)
    (member : column ∈ receipt.allocations)
    (publicColumn : column.ownership = .publicColumn) :
    column ∈ (pruneReceipt protectedIds mentioned receipt).allocations := by
  apply List.mem_filter.2
  exact ⟨member, by simp [Keep, publicColumn]⟩

theorem protected_kept
    (protectedIds mentioned : List ColumnId)
    (receipt : ManifestReceipt)
    (column : OwnedColumn)
    (member : column ∈ receipt.allocations)
    (protectedMember : column.id ∈ protectedIds) :
    column ∈ (pruneReceipt protectedIds mentioned receipt).allocations := by
  apply List.mem_filter.2
  exact ⟨member, by simp [Keep, protectedMember]⟩

theorem allocations_length_le
    (protectedIds mentioned : List ColumnId)
    (receipt : ManifestReceipt) :
    (pruneReceipt protectedIds mentioned receipt).allocations.length <=
      receipt.allocations.length := by
  exact List.length_filter_le _ _

end Nightstream.Implementation.Lowering.Goldilocks.Optimization.Passes.PruneAllocations
