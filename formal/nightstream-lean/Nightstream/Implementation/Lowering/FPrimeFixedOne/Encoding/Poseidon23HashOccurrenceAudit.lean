import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.Poseidon23HashOccurrence
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.CanonicalPoseidon2Sponge23RecipeAudit

/-!
Contract: exact sparse-column support for one total fixed-23 binding-hash
occurrence.

Owns: the proof that every wrapper and core row mentions only the adapter
visible columns or the nonoptional temporary receipt.

Does not own: semantic soundness, honest completion, typed call decoding,
Rust, or generated rows.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding

open Nightstream.Implementation.Lowering.Typed
open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Vocabulary
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.DirectCalls

namespace Poseidon23HashOccurrence

private theorem owned_id_mem_ids
    {layout : Layout}
    (bundle : ColumnBundle layout)
    (owned : OwnedColumn)
    (member : owned ∈ bundle.columns) :
    owned.id ∈ bundle.ids := by
  unfold ColumnBundle.ids
  exact List.mem_map.mpr ⟨owned, member, rfl⟩

private theorem sourceAt_supported
    {sourceWidth alignmentWidth : Nat}
    (frame : Frame sourceWidth alignmentWidth)
    (index : Fin sourceWidth) :
    (frame.sourceAt index).id ∈
      frame.visibleIds ++ frame.prefixTemporaryIds := by
  have member : frame.sourceAt index ∈ frame.source :=
    List.get_mem frame.source
      ⟨index.val, by
        rw [frame.source_length]
        exact index.isLt⟩
  rw [Frame.source] at member
  rcases List.mem_cons.mp member with normalized | tail
  · rw [normalized]
    apply List.mem_append_right
    have normalizedMember :
        frame.normalizedColumn.id ∈ frame.normalized.ids :=
      owned_id_mem_ids frame.normalized frame.normalizedColumn
        (List.get_mem _ _)
    simpa [Frame.prefixTemporaryIds, normalizedMember]
  · apply List.mem_append_left
    unfold Frame.visibleIds
    apply List.mem_append_left
    apply List.mem_append_right
    exact List.mem_map.mpr ⟨frame.sourceAt index, tail, rfl⟩

private theorem projected_supported
    {sourceWidth alignmentWidth targetWidth : Nat}
    (frame : Frame sourceWidth alignmentWidth)
    (projection : Fin targetWidth -> Fin sourceWidth)
    (column : ColumnId)
    (member :
      column ∈ (frame.projected projection).map
        (fun item => item.id)) :
    column ∈ frame.visibleIds ++ frame.prefixTemporaryIds := by
  rcases List.mem_map.mp member with ⟨owned, ownedMember, rfl⟩
  rcases List.mem_ofFn.mp (by
    simpa [Frame.projected] using ownedMember) with ⟨index, rfl⟩
  exact sourceAt_supported frame (projection index)

private theorem prefix_bundle_supported
    {sourceWidth alignmentWidth : Nat}
    (frame : Frame sourceWidth alignmentWidth)
    {bundleIds : List ColumnId}
    (member :
      bundleIds ∈
        [frame.normalized.ids, frame.preimage.ids,
          frame.inverses.ids, frame.equals.ids, frame.products.ids,
          frame.equalityOutput.ids, frame.selected.ids,
          frame.coreOutput.ids])
    (column : ColumnId)
    (columnMember : column ∈ bundleIds) :
    column ∈ frame.visibleIds ++ frame.prefixTemporaryIds := by
  apply List.mem_append_right
  unfold Frame.prefixTemporaryIds
  simp only [List.mem_cons, List.not_mem_nil, or_false] at member
  rcases member with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl
  · simp [columnMember]
  · simp [columnMember]
  · simp [columnMember]
  · simp [columnMember]
  · simp [columnMember]
  · simp [columnMember]
  · simp [columnMember]
  · simp [columnMember]

theorem wrapperRawRows_supported
    {sourceWidth alignmentWidth : Nat}
    (frame : Frame sourceWidth alignmentWidth) :
    RawRowsSupportedBy
      (frame.visibleIds ++ frame.prefixTemporaryIds)
      (wrapperRawRows frame) := by
  intro row rowMember column columnMember
  unfold wrapperRawRows at rowMember
  rcases List.mem_append.mp rowMember with normalization | rest
  · simp only [List.mem_singleton] at normalization
    subst row
    cases nextExact : frame.next
    · simp [normalizationRow, nextExact, Row.columnIds,
        Goldilocks.singleton, Goldilocks.difference] at columnMember
      rcases columnMember with one | iteration | normalized
      · subst column
        simp [Frame.visibleIds]
      · subst column
        simp [Frame.visibleIds]
      · exact prefix_bundle_supported frame
          (bundleIds := frame.normalized.ids) (by simp) column (by
            rw [normalized]
            exact owned_id_mem_ids frame.normalized
              frame.normalizedColumn (List.get_mem _ _))
    · simp [normalizationRow, nextExact, Row.columnIds,
        Goldilocks.singleton, Goldilocks.difference] at columnMember
      rcases columnMember with one | iteration | one | normalized
      · subst column
        simp [Frame.visibleIds]
      · subst column
        simp [Frame.visibleIds]
      · subst column
        simp [Frame.visibleIds]
      · exact prefix_bundle_supported frame
          (bundleIds := frame.normalized.ids) (by simp) column (by
            rw [normalized]
            exact owned_id_mem_ids frame.normalized
              frame.normalizedColumn (List.get_mem _ _))
  rcases List.mem_append.mp rest with preimage | rest
  · rcases List.mem_ofFn.mp preimage with ⟨index, rfl⟩
    simp [preimageRow, Row.columnIds, Goldilocks.singleton,
      Goldilocks.difference] at columnMember
    rcases columnMember with one | source | destination
    · subst column
      simp [Frame.visibleIds]
    · subst column
      exact sourceAt_supported frame (frame.plan.preimage index)
    · exact prefix_bundle_supported frame
        (bundleIds := frame.preimage.ids) (by simp) column (by
          rw [destination]
          exact owned_id_mem_ids frame.preimage
            (frame.preimage.columns.get
              ⟨index.val, by
                rw [frame.preimage.length_eq]
                simpa [auxiliary, auxiliaryLayout, ownedLayout]
                  using index.isLt⟩)
            (List.get_mem _ _))
  rcases List.mem_append.mp rest with equality | rest
  · rcases frame.equality.rawRows_supported row equality column
        columnMember with
      one | active | left | right | output | inverse | equal | product
    · subst column
      simp [Frame.visibleIds, Frame.equality]
    · subst column
      simp [Frame.visibleIds, Frame.equality]
    · exact projected_supported frame frame.plan.alignmentLeft column
        (by simpa [Frame.equality] using left)
    · exact projected_supported frame frame.plan.alignmentRight column
        (by simpa [Frame.equality] using right)
    · exact prefix_bundle_supported frame
        (bundleIds := frame.equalityOutput.ids) (by simp) column (by
        rw [output]
        exact owned_id_mem_ids frame.equalityOutput
          frame.equalityOutputColumn (List.get_mem _ _))
    · exact prefix_bundle_supported frame
        (bundleIds := frame.inverses.ids) (by simp) column (by
        unfold ColumnBundle.ids
        simpa [Frame.equality] using inverse)
    · exact prefix_bundle_supported frame
        (bundleIds := frame.equals.ids) (by simp) column (by
        unfold ColumnBundle.ids
        simpa [Frame.equality] using equal)
    · exact prefix_bundle_supported frame
        (bundleIds := frame.products.ids) (by simp) column (by
        unfold ColumnBundle.ids
        simpa [Frame.equality] using product)
  rcases List.mem_append.mp rest with selectedOrTag | payload
  rcases List.mem_cons.mp selectedOrTag with selected | tag
  · subst row
    simp [selectedRow, Row.columnIds, Goldilocks.singleton] at columnMember
    rcases columnMember with active | equality | selected
    · subst column
      simp [Frame.visibleIds]
    · exact prefix_bundle_supported frame
        (bundleIds := frame.equalityOutput.ids) (by simp) column (by
        rw [equality]
        exact owned_id_mem_ids frame.equalityOutput
          frame.equalityOutputColumn (List.get_mem _ _))
    · exact prefix_bundle_supported frame
        (bundleIds := frame.selected.ids) (by simp) column (by
        rw [selected]
        exact owned_id_mem_ids frame.selected frame.selectedColumn
          (List.get_mem _ _))
  rcases List.mem_cons.mp tag with tag | impossible
  · subst row
    simp [tagRow, Row.columnIds, Goldilocks.singleton,
      Goldilocks.difference] at columnMember
    rcases columnMember with active | selected | output
    · subst column
      simp [Frame.visibleIds]
    · exact prefix_bundle_supported frame
        (bundleIds := frame.selected.ids) (by simp) column (by
        rw [selected]
        exact owned_id_mem_ids frame.selected frame.selectedColumn
          (List.get_mem _ _))
    · apply List.mem_append_left
      unfold Frame.visibleIds
      apply List.mem_append_right
      exact List.mem_map.mpr
        ⟨frame.outputAt 0, List.get_mem _ _, output.symm⟩
  · exact False.elim (by simpa using impossible)
  · rcases List.mem_append.mp payload with success | absent
    · rcases List.mem_ofFn.mp success with ⟨lane, rfl⟩
      simp [payloadSuccessRow, Row.columnIds, Goldilocks.singleton,
        Goldilocks.difference] at columnMember
      rcases columnMember with selected | core | output
      · exact prefix_bundle_supported frame
          (bundleIds := frame.selected.ids) (by simp) column (by
          rw [selected]
          exact owned_id_mem_ids frame.selected frame.selectedColumn
            (List.get_mem _ _))
      · exact prefix_bundle_supported frame
          (bundleIds := frame.coreOutput.ids) (by simp) column (by
          rw [core]
          exact owned_id_mem_ids frame.coreOutput
            (frame.coreOutputAt lane) (List.get_mem _ _))
      · apply List.mem_append_left
        unfold Frame.visibleIds
        apply List.mem_append_right
        exact List.mem_map.mpr
          ⟨frame.outputAt ⟨lane.val + 1, by omega⟩,
            List.get_mem _ _, output.symm⟩
    · rcases List.mem_ofFn.mp absent with ⟨lane, rfl⟩
      simp [payloadAbsentRow, Row.columnIds, Goldilocks.singleton,
        Goldilocks.difference] at columnMember
      rcases columnMember with active | selected | output
      · subst column
        simp [Frame.visibleIds]
      · exact prefix_bundle_supported frame
          (bundleIds := frame.selected.ids) (by simp) column (by
          rw [selected]
          exact owned_id_mem_ids frame.selected frame.selectedColumn
            (List.get_mem _ _))
      · apply List.mem_append_left
        unfold Frame.visibleIds
        apply List.mem_append_right
        exact List.mem_map.mpr
          ⟨frame.outputAt ⟨lane.val + 1, by omega⟩,
            List.get_mem _ _, output.symm⟩

/-- Every emitted dependency is either adapter-visible or belongs to the
complete nonoptional temporary receipt. -/
theorem rows_supported
    {sourceWidth alignmentWidth : Nat}
    (frame : Frame sourceWidth alignmentWidth)
    (facts : CoreAllocationFacts frame)
    (owned : OwnedRow)
    (member : owned ∈ rows frame facts)
    (column : ColumnId)
    (columnMember : column ∈ owned.columnIds) :
    column ∈ frame.visibleIds ++ frame.temporaryIds := by
  have rawMember :=
    ownRows_row_mem frame.owner (rawRows frame facts) owned member
  rw [rawRows_eq_wrapper_append_core] at rawMember
  rcases List.mem_append.mp rawMember with wrapper | coreMember
  · have supported :=
      wrapperRawRows_supported frame owned.row wrapper column columnMember
    rcases List.mem_append.mp supported with visible | prefixMember
    · exact List.mem_append_left _ visible
    · exact List.mem_append_right _
        (List.mem_append_left _ prefixMember)
  · rcases List.mem_map.mp coreMember with ⟨coreRow, coreOwned, rowExact⟩
    have support :=
      CanonicalPoseidon2Sponge23Recipe.rows_supported
        (core frame facts) coreRow coreOwned column
        (by
          simpa [OwnedRow.columnIds, rowExact] using columnMember)
    rcases List.mem_append.mp support with coreVisible | coreTemporary
    · have coreCases :
        column = frame.one ∨
          column = frame.selectedColumn.id ∨
          column ∈ frame.preimage.ids ∨
          column ∈ frame.coreOutput.ids := by
        simpa [CanonicalPoseidon2Sponge23Recipe.Frame.visibleIds,
          core] using coreVisible
      rcases coreCases with one | selected | preimage | output
      · subst column
        exact List.mem_append_left _ (by simp [Frame.visibleIds])
      · exact List.mem_append_right _ (List.mem_append_left _ (by
          subst column
          unfold Frame.prefixTemporaryIds
          have selectedMember :
              frame.selectedColumn.id ∈ frame.selected.ids := by
            unfold ColumnBundle.ids
            exact List.mem_map.mpr
              ⟨frame.selectedColumn, List.get_mem _ _, rfl⟩
          simp [selectedMember]))
      · exact List.mem_append_right _ (List.mem_append_left _ (by
          unfold Frame.prefixTemporaryIds
          simp [preimage]))
      · exact List.mem_append_right _ (List.mem_append_left _ (by
          unfold Frame.prefixTemporaryIds
          simp [output]))
    · exact List.mem_append_right _ (List.mem_append_right _ (by
        simpa [core] using coreTemporary))

end Poseidon23HashOccurrence

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding
