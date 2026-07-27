import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.Poseidon23HashOccurrenceSemantics
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.Poseidon23HashOccurrenceAudit
import Nightstream.Implementation.Lowering.Goldilocks.SelectedBranchSupport

/-!
Contract: temporary-only honest completion of one total fixed-23 binding-hash
occurrence.

Owns: exact values for every non-core temporary, composition with the
canonical sponge witness, active and inactive satisfaction, and preservation
of the adapter-owned visible coordinates.

Does not own: typed call decoding, application serialization, Rust, generated
rows, collision resistance, or deployment selection.
-/

set_option autoImplicit false
set_option maxRecDepth 32768

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding

open Nightstream.Implementation.Lowering.Typed
open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Vocabulary
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.DirectCalls

namespace Poseidon23HashOccurrence

namespace Honest

private theorem rawSatisfies_of_forall
    {rows : List Row}
    {assignment : ColumnId -> Field}
    (holds : ∀ row, row ∈ rows -> row.Holds assignment) :
    RawSatisfies rows assignment := by
  induction rows with
  | nil =>
      trivial
  | cons row tail inductionHypothesis =>
      exact
        ⟨holds row (by simp),
          inductionHypothesis (fun item member =>
            holds item (by simp [member]))⟩

private theorem copyRow_holds
    (assignment : ColumnId -> Field)
    (one source destination : ColumnId)
    (constantOne : assignment one = 1)
    (destinationExact :
      assignment destination = assignment source) :
    (Row.mk (Goldilocks.singleton one 1)
      (Goldilocks.difference source destination) []).Holds assignment := by
  simp [Row.Holds, Goldilocks.singleton, Goldilocks.difference,
    Goldilocks.LinearCombination.eval, constantOne, destinationExact,
    Fin.one_mul, Fin.add_zero, Lean.Grind.Fin.neg_mul]
  exact Lean.Grind.AddCommGroup.add_neg_cancel _

def normalizedValue
    {sourceWidth alignmentWidth : Nat}
    (frame : Frame sourceWidth alignmentWidth)
    (assignment : ColumnId -> Field) : Field :=
  if frame.next then assignment frame.iteration.id + 1
  else assignment frame.iteration.id

def sourceValues
    {sourceWidth alignmentWidth : Nat}
    (frame : Frame sourceWidth alignmentWidth)
    (assignment : ColumnId -> Field) : List Field :=
  normalizedValue frame assignment ::
    frame.sourceTail.map (fun column => assignment column.id)

@[simp] theorem sourceValues_length
    {sourceWidth alignmentWidth : Nat}
    (frame : Frame sourceWidth alignmentWidth)
    (assignment : ColumnId -> Field) :
    (sourceValues frame assignment).length = sourceWidth := by
  simp [sourceValues, frame.sourceTailLength]

def preimageValues
    {sourceWidth alignmentWidth : Nat}
    (frame : Frame sourceWidth alignmentWidth)
    (assignment : ColumnId -> Field) : List Field :=
  Poseidon23Hash.select (sourceValues frame assignment) frame.plan.preimage

def alignmentLeftValues
    {sourceWidth alignmentWidth : Nat}
    (frame : Frame sourceWidth alignmentWidth)
    (assignment : ColumnId -> Field) : List Field :=
  Poseidon23Hash.select
    (sourceValues frame assignment) frame.plan.alignmentLeft

def alignmentRightValues
    {sourceWidth alignmentWidth : Nat}
    (frame : Frame sourceWidth alignmentWidth)
    (assignment : ColumnId -> Field) : List Field :=
  Poseidon23Hash.select
    (sourceValues frame assignment) frame.plan.alignmentRight

def inverseValues
    (inverseLaw : InverseLaw)
    {sourceWidth alignmentWidth : Nat}
    (frame : Frame sourceWidth alignmentWidth)
    (assignment : ColumnId -> Field) : List Field :=
  coordinateInverseValues inverseLaw
    (alignmentLeftValues frame assignment)
    (alignmentRightValues frame assignment)

def equalValues
    {sourceWidth alignmentWidth : Nat}
    (frame : Frame sourceWidth alignmentWidth)
    (assignment : ColumnId -> Field) : List Field :=
  coordinateEqualValues
    (alignmentLeftValues frame assignment)
    (alignmentRightValues frame assignment)

def productCoordinates
    {sourceWidth alignmentWidth : Nat}
    (frame : Frame sourceWidth alignmentWidth)
    (assignment : ColumnId -> Field) : List Field :=
  productValues (equalValues frame assignment)

def equalityValue
    {sourceWidth alignmentWidth : Nat}
    (frame : Frame sourceWidth alignmentWidth)
    (assignment : ColumnId -> Field) : Field :=
  if alignmentLeftValues frame assignment =
      alignmentRightValues frame assignment
  then 1 else 0

def selectedValue
    {sourceWidth alignmentWidth : Nat}
    (frame : Frame sourceWidth alignmentWidth)
    (assignment : ColumnId -> Field) : Field :=
  assignment frame.active * equalityValue frame assignment

def digestValues
    {sourceWidth alignmentWidth : Nat}
    (frame : Frame sourceWidth alignmentWidth)
    (assignment : ColumnId -> Field) : List Field :=
  Poseidon23Hash.digestCoordinates (preimageValues frame assignment)

def prefixValues
    (inverseLaw : InverseLaw)
    {sourceWidth alignmentWidth : Nat}
    (frame : Frame sourceWidth alignmentWidth)
    (assignment : ColumnId -> Field) : List Field :=
  [normalizedValue frame assignment] ++
    preimageValues frame assignment ++
    inverseValues inverseLaw frame assignment ++
    equalValues frame assignment ++
    productCoordinates frame assignment ++
    [equalityValue frame assignment] ++
    [selectedValue frame assignment] ++
    digestValues frame assignment

@[simp] theorem prefixValues_length
    (inverseLaw : InverseLaw)
    {sourceWidth alignmentWidth : Nat}
    (frame : Frame sourceWidth alignmentWidth)
    (assignment : ColumnId -> Field) :
    (prefixValues inverseLaw frame assignment).length =
      30 + 2 * alignmentWidth + alignmentWidth.pred := by
  simp [prefixValues, preimageValues, inverseValues, equalValues,
    productCoordinates, alignmentLeftValues, alignmentRightValues,
    digestValues, Poseidon23Hash.digestCoordinates_length,
    coordinateInverseValues_length, coordinateEqualValues_length,
    productValues_length]
  omega

def preCore
    (inverseLaw : InverseLaw)
    {sourceWidth alignmentWidth : Nat}
    (frame : Frame sourceWidth alignmentWidth)
    (assignment : ColumnId -> Field) : ColumnId -> Field :=
  writeColumns assignment frame.prefixTemporaryIds
    (prefixValues inverseLaw frame assignment)

def complete
    (inverseLaw : InverseLaw)
    {sourceWidth alignmentWidth : Nat}
    (frame : Frame sourceWidth alignmentWidth)
    (facts : CoreAllocationFacts frame)
    (assignment : ColumnId -> Field) : ColumnId -> Field :=
  CanonicalPoseidon2Sponge23Recipe.Honest.complete
    (core frame facts) (preCore inverseLaw frame assignment)

private theorem prefixTemporaryIds_length
    {sourceWidth alignmentWidth : Nat}
    (frame : Frame sourceWidth alignmentWidth) :
    frame.prefixTemporaryIds.length =
      30 + 2 * alignmentWidth + alignmentWidth.pred := by
  simp [Frame.prefixTemporaryIds, ColumnBundle.ids,
    frame.normalized.length_eq, frame.preimage.length_eq,
    frame.inverses.length_eq, frame.equals.length_eq,
    frame.products.length_eq, frame.equalityOutput.length_eq,
    frame.selected.length_eq, frame.coreOutput.length_eq,
    auxiliary, auxiliaryLayout, ownedLayout]
  omega

theorem preCore_changesOnly
    (inverseLaw : InverseLaw)
    {sourceWidth alignmentWidth : Nat}
    (frame : Frame sourceWidth alignmentWidth)
    (assignment : ColumnId -> Field) :
    ChangesOnly frame.prefixTemporaryIds assignment
      (preCore inverseLaw frame assignment) :=
  writeColumns_changesOnly assignment frame.prefixTemporaryIds
    (prefixValues inverseLaw frame assignment)

theorem preCore_recovery
    (inverseLaw : InverseLaw)
    {sourceWidth alignmentWidth : Nat}
    (frame : Frame sourceWidth alignmentWidth)
    (assignment : ColumnId -> Field)
    (prefixNodup : frame.prefixTemporaryIds.Nodup) :
    frame.prefixTemporaryIds.map
        (preCore inverseLaw frame assignment) =
      prefixValues inverseLaw frame assignment := by
  apply writeColumns_map_eq
  · rw [prefixTemporaryIds_length, prefixValues_length]
  · exact prefixNodup

private theorem writeColumns_segment
    (assignment : ColumnId -> Field)
    (prior columns suffix : List ColumnId)
    (priorValues values suffixValues : List Field)
    (priorLength : prior.length = priorValues.length)
    (columnsLength : columns.length = values.length)
    (suffixLength : suffix.length = suffixValues.length)
    (nodup : (prior ++ (columns ++ suffix)).Nodup) :
    columns.map
        (writeColumns assignment
          (prior ++ (columns ++ suffix))
          (priorValues ++ (values ++ suffixValues))) =
      values := by
  have recovered :=
    writeColumns_map_eq assignment
      (prior ++ (columns ++ suffix))
      (priorValues ++ (values ++ suffixValues))
      (by simp [priorLength, columnsLength, suffixLength])
      nodup
  simp only [List.map_append] at recovered
  have tails :=
    (List.append_inj recovered (by simpa using priorLength)).2
  exact
    (List.append_inj tails (by simpa using columnsLength)).1

private theorem preCore_segment
    (inverseLaw : InverseLaw)
    {sourceWidth alignmentWidth : Nat}
    (frame : Frame sourceWidth alignmentWidth)
    (assignment : ColumnId -> Field)
    (prior columns suffix : List ColumnId)
    (priorValues values suffixValues : List Field)
    (idsExact :
      frame.prefixTemporaryIds = prior ++ (columns ++ suffix))
    (valuesExact :
      prefixValues inverseLaw frame assignment =
        priorValues ++ (values ++ suffixValues))
    (priorLength : prior.length = priorValues.length)
    (columnsLength : columns.length = values.length)
    (suffixLength : suffix.length = suffixValues.length)
    (prefixNodup : frame.prefixTemporaryIds.Nodup) :
    columns.map (preCore inverseLaw frame assignment) = values := by
  unfold preCore
  rw [idsExact, valuesExact]
  exact writeColumns_segment assignment prior columns suffix
    priorValues values suffixValues priorLength columnsLength
    suffixLength (idsExact ▸ prefixNodup)

theorem preCore_normalized
    (inverseLaw : InverseLaw)
    {sourceWidth alignmentWidth : Nat}
    (frame : Frame sourceWidth alignmentWidth)
    (assignment : ColumnId -> Field)
    (prefixNodup : frame.prefixTemporaryIds.Nodup) :
    frame.normalized.values (preCore inverseLaw frame assignment) =
      [normalizedValue frame assignment] := by
  simpa [ColumnBundle.values, ColumnBundle.ids] using
    preCore_segment inverseLaw frame assignment
      []
      frame.normalized.ids
      (frame.preimage.ids ++ frame.inverses.ids ++ frame.equals.ids ++
        frame.products.ids ++ frame.equalityOutput.ids ++
        frame.selected.ids ++ frame.coreOutput.ids)
      []
      [normalizedValue frame assignment]
      (preimageValues frame assignment ++
        inverseValues inverseLaw frame assignment ++
        equalValues frame assignment ++
        productCoordinates frame assignment ++
        [equalityValue frame assignment] ++
        [selectedValue frame assignment] ++
        digestValues frame assignment)
      (by simp [Frame.prefixTemporaryIds, List.append_assoc])
      (by simp [prefixValues, List.append_assoc])
      (by simp)
      (by
        simp [ColumnBundle.ids, frame.normalized.length_eq,
          auxiliary, auxiliaryLayout, ownedLayout])
      (by
        simp [ColumnBundle.ids, frame.preimage.length_eq,
          frame.inverses.length_eq, frame.equals.length_eq,
          frame.products.length_eq, frame.equalityOutput.length_eq,
          frame.selected.length_eq, frame.coreOutput.length_eq,
          preimageValues, inverseValues, equalValues, productCoordinates,
          alignmentLeftValues, alignmentRightValues, digestValues,
          coordinateInverseValues_length, coordinateEqualValues_length,
          productValues_length, auxiliary, auxiliaryLayout, ownedLayout])
      prefixNodup

theorem preCore_preimage
    (inverseLaw : InverseLaw)
    {sourceWidth alignmentWidth : Nat}
    (frame : Frame sourceWidth alignmentWidth)
    (assignment : ColumnId -> Field)
    (prefixNodup : frame.prefixTemporaryIds.Nodup) :
    frame.preimage.values (preCore inverseLaw frame assignment) =
      preimageValues frame assignment := by
  simpa [ColumnBundle.values, ColumnBundle.ids] using
    preCore_segment inverseLaw frame assignment
      frame.normalized.ids
      frame.preimage.ids
      (frame.inverses.ids ++ frame.equals.ids ++ frame.products.ids ++
        frame.equalityOutput.ids ++ frame.selected.ids ++
        frame.coreOutput.ids)
      [normalizedValue frame assignment]
      (preimageValues frame assignment)
      (inverseValues inverseLaw frame assignment ++
        equalValues frame assignment ++
        productCoordinates frame assignment ++
        [equalityValue frame assignment] ++
        [selectedValue frame assignment] ++
        digestValues frame assignment)
      (by simp [Frame.prefixTemporaryIds, List.append_assoc])
      (by simp [prefixValues, List.append_assoc])
      (by
        simp [ColumnBundle.ids, frame.normalized.length_eq,
          auxiliary, auxiliaryLayout, ownedLayout])
      (by
        simp [ColumnBundle.ids, frame.preimage.length_eq, preimageValues,
          auxiliary, auxiliaryLayout, ownedLayout])
      (by
        simp [ColumnBundle.ids, frame.inverses.length_eq,
          frame.equals.length_eq, frame.products.length_eq,
          frame.equalityOutput.length_eq, frame.selected.length_eq,
          frame.coreOutput.length_eq, inverseValues, equalValues,
          productCoordinates, alignmentLeftValues, alignmentRightValues,
          digestValues, coordinateInverseValues_length,
          coordinateEqualValues_length, productValues_length,
          auxiliary, auxiliaryLayout, ownedLayout])
      prefixNodup

theorem preCore_equality_values
    (inverseLaw : InverseLaw)
    {sourceWidth alignmentWidth : Nat}
    (frame : Frame sourceWidth alignmentWidth)
    (assignment : ColumnId -> Field)
    (prefixNodup : frame.prefixTemporaryIds.Nodup) :
    frame.inverses.values (preCore inverseLaw frame assignment) =
        inverseValues inverseLaw frame assignment ∧
      frame.equals.values (preCore inverseLaw frame assignment) =
        equalValues frame assignment ∧
      frame.products.values (preCore inverseLaw frame assignment) =
        productCoordinates frame assignment := by
  have recovered :=
    preCore_recovery inverseLaw frame assignment prefixNodup
  simp only [Frame.prefixTemporaryIds, prefixValues, List.map_append,
    ColumnBundle.ids, ColumnBundle.values, List.append_assoc] at recovered
  have normalizedSplit :=
    List.append_inj recovered (by
      simp [frame.normalized.length_eq, auxiliary, auxiliaryLayout,
        ownedLayout])
  have preimageSplit :=
    List.append_inj normalizedSplit.2 (by
      simp [frame.preimage.length_eq, preimageValues, auxiliary,
        auxiliaryLayout, ownedLayout])
  have inverseSplit :=
    List.append_inj preimageSplit.2 (by
      simp [frame.inverses.length_eq, inverseValues,
        alignmentLeftValues, alignmentRightValues,
        coordinateInverseValues_length, auxiliary, auxiliaryLayout,
        ownedLayout])
  have equalSplit :=
    List.append_inj inverseSplit.2 (by
      simp [frame.equals.length_eq, equalValues,
        alignmentLeftValues, alignmentRightValues,
        coordinateEqualValues_length, auxiliary, auxiliaryLayout,
        ownedLayout])
  have productSplit :=
    List.append_inj equalSplit.2 (by
      simp [frame.products.length_eq, productCoordinates, equalValues,
        alignmentLeftValues, alignmentRightValues,
        coordinateEqualValues_length, productValues_length,
        auxiliary, auxiliaryLayout, ownedLayout])
  exact ⟨
    by simpa [ColumnBundle.values, List.map_map,
        Function.comp_apply] using inverseSplit.1,
    by simpa [ColumnBundle.values, List.map_map,
        Function.comp_apply] using equalSplit.1,
    by simpa [ColumnBundle.values, List.map_map,
        Function.comp_apply] using productSplit.1⟩

theorem preCore_scalar_values
    (inverseLaw : InverseLaw)
    {sourceWidth alignmentWidth : Nat}
    (frame : Frame sourceWidth alignmentWidth)
    (assignment : ColumnId -> Field)
    (prefixNodup : frame.prefixTemporaryIds.Nodup) :
    preCore inverseLaw frame assignment frame.equalityOutputColumn.id =
        equalityValue frame assignment ∧
      preCore inverseLaw frame assignment frame.selectedColumn.id =
        selectedValue frame assignment := by
  have recovered :=
    preCore_recovery inverseLaw frame assignment prefixNodup
  simp only [Frame.prefixTemporaryIds, prefixValues, List.map_append,
    ColumnBundle.ids, ColumnBundle.values, List.append_assoc] at recovered
  have normalizedSplit :=
    List.append_inj recovered (by
      simp [frame.normalized.length_eq, auxiliary, auxiliaryLayout,
        ownedLayout])
  have preimageSplit :=
    List.append_inj normalizedSplit.2 (by
      simp [frame.preimage.length_eq, preimageValues, auxiliary,
        auxiliaryLayout, ownedLayout])
  have inverseSplit :=
    List.append_inj preimageSplit.2 (by
      simp [frame.inverses.length_eq, inverseValues,
        alignmentLeftValues, alignmentRightValues,
        coordinateInverseValues_length, auxiliary, auxiliaryLayout,
        ownedLayout])
  have equalSplit :=
    List.append_inj inverseSplit.2 (by
      simp [frame.equals.length_eq, equalValues,
        alignmentLeftValues, alignmentRightValues,
        coordinateEqualValues_length, auxiliary, auxiliaryLayout,
        ownedLayout])
  have productSplit :=
    List.append_inj equalSplit.2 (by
      simp [frame.products.length_eq, productCoordinates, equalValues,
        alignmentLeftValues, alignmentRightValues,
        coordinateEqualValues_length, productValues_length,
        auxiliary, auxiliaryLayout, ownedLayout])
  have equalitySplit :=
    List.append_inj productSplit.2 (by
      simp [frame.equalityOutput.length_eq, auxiliary, auxiliaryLayout,
        ownedLayout])
  have selectedSplit :=
    List.append_inj equalitySplit.2 (by
      simp [frame.selected.length_eq, auxiliary, auxiliaryLayout,
        ownedLayout])
  have equalityAt :
      frame.equalityOutput.values
          (preCore inverseLaw frame assignment) =
        [equalityValue frame assignment] :=
    by simpa [ColumnBundle.values, List.map_map,
        Function.comp_apply] using equalitySplit.1
  have selectedAt :
      frame.selected.values (preCore inverseLaw frame assignment) =
        [selectedValue frame assignment] :=
    by simpa [ColumnBundle.values, List.map_map,
        Function.comp_apply] using selectedSplit.1
  constructor
  · have singleton :=
      bundle_values_eq_singleton frame.equalityOutput
        (preCore inverseLaw frame assignment)
        (by simp [auxiliary, auxiliaryLayout, ownedLayout])
    rw [singleton] at equalityAt
    simpa [Frame.equalityOutputColumn] using
      (List.cons.inj equalityAt).1
  · have singleton :=
      bundle_values_eq_singleton frame.selected
        (preCore inverseLaw frame assignment)
        (by simp [auxiliary, auxiliaryLayout, ownedLayout])
    rw [singleton] at selectedAt
    simpa [Frame.selectedColumn] using
      (List.cons.inj selectedAt).1

theorem preCore_digest
    (inverseLaw : InverseLaw)
    {sourceWidth alignmentWidth : Nat}
    (frame : Frame sourceWidth alignmentWidth)
    (assignment : ColumnId -> Field)
    (prefixNodup : frame.prefixTemporaryIds.Nodup) :
    frame.coreOutput.values (preCore inverseLaw frame assignment) =
      digestValues frame assignment := by
  have recovered :=
    preCore_recovery inverseLaw frame assignment prefixNodup
  simp only [Frame.prefixTemporaryIds, prefixValues, List.map_append,
    ColumnBundle.ids, ColumnBundle.values, List.append_assoc] at recovered
  have normalizedSplit :=
    List.append_inj recovered (by
      simp [frame.normalized.length_eq, auxiliary, auxiliaryLayout,
        ownedLayout])
  have preimageSplit :=
    List.append_inj normalizedSplit.2 (by
      simp [frame.preimage.length_eq, preimageValues, auxiliary,
        auxiliaryLayout, ownedLayout])
  have inverseSplit :=
    List.append_inj preimageSplit.2 (by
      simp [frame.inverses.length_eq, inverseValues,
        alignmentLeftValues, alignmentRightValues,
        coordinateInverseValues_length, auxiliary, auxiliaryLayout,
        ownedLayout])
  have equalSplit :=
    List.append_inj inverseSplit.2 (by
      simp [frame.equals.length_eq, equalValues,
        alignmentLeftValues, alignmentRightValues,
        coordinateEqualValues_length, auxiliary, auxiliaryLayout,
        ownedLayout])
  have productSplit :=
    List.append_inj equalSplit.2 (by
      simp [frame.products.length_eq, productCoordinates, equalValues,
        alignmentLeftValues, alignmentRightValues,
        coordinateEqualValues_length, productValues_length,
        auxiliary, auxiliaryLayout, ownedLayout])
  have equalitySplit :=
    List.append_inj productSplit.2 (by
      simp [frame.equalityOutput.length_eq, auxiliary, auxiliaryLayout,
        ownedLayout])
  have selectedSplit :=
    List.append_inj equalitySplit.2 (by
      simp [frame.selected.length_eq, auxiliary, auxiliaryLayout,
        ownedLayout])
  simpa [ColumnBundle.values, List.map_map,
    Function.comp_apply] using selectedSplit.2

theorem preCore_agrees_visible
    (inverseLaw : InverseLaw)
    {sourceWidth alignmentWidth : Nat}
    (frame : Frame sourceWidth alignmentWidth)
    (assignment : ColumnId -> Field)
    (temporariesDisjointVisible :
      IdsDisjoint frame.temporaryIds frame.visibleIds) :
    AgreesOn frame.visibleIds assignment
      (preCore inverseLaw frame assignment) := by
  apply writeColumns_agreesOn
  intro id prefixMember visibleMember
  exact temporariesDisjointVisible id
    (List.mem_append_left frame.coreTemporaries.ids prefixMember)
    visibleMember

theorem preCore_normalized_coordinate
    (inverseLaw : InverseLaw)
    {sourceWidth alignmentWidth : Nat}
    (frame : Frame sourceWidth alignmentWidth)
    (assignment : ColumnId -> Field)
    (prefixNodup : frame.prefixTemporaryIds.Nodup) :
    preCore inverseLaw frame assignment frame.normalizedColumn.id =
      normalizedValue frame assignment := by
  have values :=
    preCore_normalized inverseLaw frame assignment prefixNodup
  have singleton :=
    bundle_values_eq_singleton frame.normalized
      (preCore inverseLaw frame assignment)
      (by simp [auxiliary, auxiliaryLayout, ownedLayout])
  rw [singleton] at values
  simpa [Frame.normalizedColumn] using (List.cons.inj values).1

theorem preCore_source_values
    (inverseLaw : InverseLaw)
    {sourceWidth alignmentWidth : Nat}
    (frame : Frame sourceWidth alignmentWidth)
    (assignment : ColumnId -> Field)
    (prefixNodup : frame.prefixTemporaryIds.Nodup)
    (temporariesDisjointVisible :
      IdsDisjoint frame.temporaryIds frame.visibleIds) :
    Poseidon23HashOccurrence.sourceValues frame
        (preCore inverseLaw frame assignment) =
      Honest.sourceValues frame assignment := by
  unfold Poseidon23HashOccurrence.sourceValues Frame.source
    Honest.sourceValues
  simp only [List.map_cons]
  congr 1
  · exact preCore_normalized_coordinate inverseLaw frame assignment
      prefixNodup
  · apply List.map_congr_left
    intro column member
    apply preCore_agrees_visible inverseLaw frame assignment
      temporariesDisjointVisible
    unfold Frame.visibleIds
    exact List.mem_append_left _
      (List.mem_append_right _
        (List.mem_map.mpr ⟨column, member, rfl⟩))

theorem preCore_output_values
    (inverseLaw : InverseLaw)
    {sourceWidth alignmentWidth : Nat}
    (frame : Frame sourceWidth alignmentWidth)
    (assignment : ColumnId -> Field)
    (temporariesDisjointVisible :
      IdsDisjoint frame.temporaryIds frame.visibleIds) :
    Poseidon23HashOccurrence.outputValues frame
        (preCore inverseLaw frame assignment) =
      Poseidon23HashOccurrence.outputValues frame assignment := by
  unfold Poseidon23HashOccurrence.outputValues
  apply List.map_congr_left
  intro column member
  apply preCore_agrees_visible inverseLaw frame assignment
    temporariesDisjointVisible
  unfold Frame.visibleIds
  exact List.mem_append_right _
    (List.mem_map.mpr ⟨column, member, rfl⟩)

theorem preCore_projected_values
    (inverseLaw : InverseLaw)
    {sourceWidth alignmentWidth targetWidth : Nat}
    (frame : Frame sourceWidth alignmentWidth)
    (assignment : ColumnId -> Field)
    (projection : Fin targetWidth -> Fin sourceWidth)
    (prefixNodup : frame.prefixTemporaryIds.Nodup)
    (temporariesDisjointVisible :
      IdsDisjoint frame.temporaryIds frame.visibleIds) :
    (frame.projected projection).map
        (fun column =>
          preCore inverseLaw frame assignment column.id) =
      Poseidon23Hash.select
        (Honest.sourceValues frame assignment) projection := by
  rw [Poseidon23HashOccurrence.projected_values,
    preCore_source_values inverseLaw frame assignment prefixNodup
      temporariesDisjointVisible]

theorem complete_changesOnly
    (inverseLaw : InverseLaw)
    {sourceWidth alignmentWidth : Nat}
    (frame : Frame sourceWidth alignmentWidth)
    (facts : CoreAllocationFacts frame)
    (assignment : ColumnId -> Field) :
    ChangesOnly frame.temporaryIds assignment
      (complete inverseLaw frame facts assignment) := by
  intro id notMember
  have prefixNotMember : id ∉ frame.prefixTemporaryIds := by
    intro member
    exact notMember (List.mem_append_left _ member)
  have coreNotMember : id ∉ frame.coreTemporaries.ids := by
    intro member
    exact notMember (List.mem_append_right _ member)
  unfold complete
  rw [
    CanonicalPoseidon2Sponge23Recipe.Honest.complete_changesOnly
      (core frame facts) (preCore inverseLaw frame assignment)
      id coreNotMember,
    preCore_changesOnly inverseLaw frame assignment id prefixNotMember]

theorem complete_agrees_visible
    (inverseLaw : InverseLaw)
    {sourceWidth alignmentWidth : Nat}
    (frame : Frame sourceWidth alignmentWidth)
    (facts : CoreAllocationFacts frame)
    (assignment : ColumnId -> Field)
    (temporariesDisjointVisible :
      IdsDisjoint frame.temporaryIds frame.visibleIds) :
    AgreesOn frame.visibleIds assignment
      (complete inverseLaw frame facts assignment) := by
  have prefixAgrees :
      AgreesOn frame.visibleIds assignment
        (preCore inverseLaw frame assignment) := by
    apply writeColumns_agreesOn
    intro id prefixMember visibleMember
    exact temporariesDisjointVisible id
      (List.mem_append_left frame.coreTemporaries.ids prefixMember)
      visibleMember
  have coreAgrees :
      AgreesOn frame.visibleIds
        (preCore inverseLaw frame assignment)
        (complete inverseLaw frame facts assignment) := by
    unfold complete
    apply writeColumns_agreesOn
    intro id coreMember visibleMember
    exact temporariesDisjointVisible id
      (List.mem_append_right frame.prefixTemporaryIds coreMember)
      visibleMember
  exact agreesOn_trans prefixAgrees coreAgrees

theorem preCore_normalization_holds
    (inverseLaw : InverseLaw)
    {sourceWidth alignmentWidth : Nat}
    (frame : Frame sourceWidth alignmentWidth)
    (assignment : ColumnId -> Field)
    (constantOne : assignment frame.one = 1)
    (prefixNodup : frame.prefixTemporaryIds.Nodup)
    (temporariesDisjointVisible :
      IdsDisjoint frame.temporaryIds frame.visibleIds) :
    (normalizationRow frame).Holds
      (preCore inverseLaw frame assignment) := by
  have onePreserved :
      preCore inverseLaw frame assignment frame.one = 1 := by
    rw [preCore_agrees_visible inverseLaw frame assignment
      temporariesDisjointVisible frame.one]
    · exact constantOne
    · simp [Frame.visibleIds]
  have iterationPreserved :
      preCore inverseLaw frame assignment frame.iteration.id =
        assignment frame.iteration.id := by
    apply preCore_agrees_visible inverseLaw frame assignment
      temporariesDisjointVisible
    simp [Frame.visibleIds]
  have normalized :=
    preCore_normalized_coordinate inverseLaw frame assignment prefixNodup
  cases nextExact : frame.next
  · simp [normalizationRow, nextExact, Row.Holds,
      Goldilocks.singleton, Goldilocks.difference,
      Goldilocks.LinearCombination.eval, onePreserved,
      iterationPreserved, normalized, normalizedValue, nextExact,
      Fin.one_mul, Fin.add_zero, Lean.Grind.Fin.neg_mul,
      ← Lean.Grind.Fin.add_assoc]
    exact Lean.Grind.AddCommGroup.add_neg_cancel _
  · simp [normalizationRow, nextExact, Row.Holds,
      Goldilocks.singleton, Goldilocks.difference,
      Goldilocks.LinearCombination.eval, onePreserved,
      iterationPreserved, normalized, normalizedValue, nextExact,
      Fin.one_mul, Fin.add_zero, Lean.Grind.Fin.neg_mul,
      ← Lean.Grind.Fin.add_assoc]
    exact Lean.Grind.AddCommGroup.add_neg_cancel _

theorem preCore_preimage_holds
    (inverseLaw : InverseLaw)
    {sourceWidth alignmentWidth : Nat}
    (frame : Frame sourceWidth alignmentWidth)
    (assignment : ColumnId -> Field)
    (constantOne : assignment frame.one = 1)
    (prefixNodup : frame.prefixTemporaryIds.Nodup)
    (temporariesDisjointVisible :
      IdsDisjoint frame.temporaryIds frame.visibleIds) :
    RawSatisfies (preimageRows frame)
      (preCore inverseLaw frame assignment) := by
  have onePreserved :
      preCore inverseLaw frame assignment frame.one = 1 := by
    rw [preCore_agrees_visible inverseLaw frame assignment
      temporariesDisjointVisible frame.one]
    · exact constantOne
    · simp [Frame.visibleIds]
  have preimageExact :=
    preCore_preimage inverseLaw frame assignment prefixNodup
  have sourceExact :=
    preCore_source_values inverseLaw frame assignment prefixNodup
      temporariesDisjointVisible
  have preimageProjected :
      frame.preimage.values
          (preCore inverseLaw frame assignment) =
        Poseidon23Hash.select
          (Poseidon23HashOccurrence.sourceValues frame
            (preCore inverseLaw frame assignment))
          frame.plan.preimage := by
    rw [preimageExact, sourceExact]
    rfl
  rw [← projected_values frame
    (preCore inverseLaw frame assignment) frame.plan.preimage]
    at preimageProjected
  apply rawSatisfies_of_forall
  intro row member
  rcases List.mem_ofFn.mp member with ⟨index, rfl⟩
  have coordinateExact :
      preCore inverseLaw frame assignment
          (frame.preimage.columns.get
            ⟨index.val, by
              rw [frame.preimage.length_eq]
              simpa [auxiliary, auxiliaryLayout, ownedLayout]
                using index.isLt⟩).id =
        preCore inverseLaw frame assignment
          (frame.sourceAt (frame.plan.preimage index)).id := by
    have leftLt :
        index.val <
          (frame.preimage.values
            (preCore inverseLaw frame assignment)).length := by
      rw [ColumnBundle.values_length]
      simpa [auxiliary, auxiliaryLayout, ownedLayout] using index.isLt
    have rightLt :
        index.val <
          ((frame.projected frame.plan.preimage).map
            (fun column =>
              preCore inverseLaw frame assignment column.id)).length := by
      simp
    have entryExact := congrArg
      (fun values => values.getD index.val 0) preimageProjected
    change
      (frame.preimage.values
          (preCore inverseLaw frame assignment)).getD index.val 0 =
        ((frame.projected frame.plan.preimage).map
          (fun column =>
            preCore inverseLaw frame assignment column.id)).getD
              index.val 0 at entryExact
    rw [← List.getElem_eq_getD
        (l := frame.preimage.values
          (preCore inverseLaw frame assignment))
        (i := index.val) (h := leftLt) 0,
      ← List.getElem_eq_getD
        (l := (frame.projected frame.plan.preimage).map
          (fun column =>
            preCore inverseLaw frame assignment column.id))
        (i := index.val) (h := rightLt) 0] at entryExact
    have projectedAt :
        (frame.projected frame.plan.preimage)[index.val] =
          frame.sourceAt (frame.plan.preimage index) := by
      unfold Frame.projected
      rw [List.getElem_ofFn]
    rw [List.getElem_map] at entryExact
    have mappedProjectedAt :
        preCore inverseLaw frame assignment
            ((frame.projected frame.plan.preimage)[index.val]).id =
          preCore inverseLaw frame assignment
            (frame.sourceAt (frame.plan.preimage index)).id :=
      congrArg
        (fun column =>
          preCore inverseLaw frame assignment column.id)
        projectedAt
    simpa only [ColumnBundle.values, List.get_eq_getElem,
      List.getElem_map] using
      entryExact.trans mappedProjectedAt
  simpa [preimageRow] using
    copyRow_holds
      (preCore inverseLaw frame assignment)
      frame.one
      (frame.sourceAt (frame.plan.preimage index)).id
      (frame.preimage.columns.get
        ⟨index.val, by
          rw [frame.preimage.length_eq]
          simpa [auxiliary, auxiliaryLayout, ownedLayout]
            using index.isLt⟩).id
      onePreserved coordinateExact

end Honest

end Poseidon23HashOccurrence

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding
