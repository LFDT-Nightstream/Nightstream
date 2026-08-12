import Nightstream.Implementation.NebulaV2.ProductNifsRunningParser

/-!
Contract: semantic correctness of the executable V2 running-claim parser.

Assurance tier: implementation-model refinement.

Owns reconstruction of every typed running-claim field from the exact
canonical field-coordinate image.

Does not own the outer full-claim envelope, generated parser rows, NIFS
verification, Rust conformance, or cryptographic soundness.

Emits constraints: no.
-/

set_option autoImplicit false
set_option maxRecDepth 100000
set_option maxHeartbeats 1000000

namespace Nightstream.Implementation.NebulaV2.ProductNifsRunningParser

open Nightstream.Implementation.NebulaV2.ProductNifsCodec
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint

private theorem getD_eq_getElem_of_lt
    {Alpha : Type} (values : List Alpha) (index : Nat)
    (default : Alpha) (bounded : index < values.length) :
    values.getD index default = values[index] := by
  rw [List.getD_eq_getElem?_getD,
    List.getElem?_eq_getElem bounded]
  rfl

private theorem k_eq_of_components
    {left right : K}
    (c0Equal : left.c0 = right.c0)
    (c1Equal : left.c1 = right.c1) : left = right := by
  cases left
  cases right
  simp_all

private theorem cubePoint_eq_of_coordinates
    {variableCount : Nat}
    {left right : CubePoint K variableCount}
    (coordinatesEqual : left.coordinates = right.coordinates) :
    left = right := by
  cases left
  cases right
  simp_all

private theorem running_eq_of_fields
    {fullShape : Phi81Relation.Shape}
    {left right : Running fullShape}
    (pointEqual : left.point = right.point)
    (commitmentsEqual : left.commitments = right.commitments)
    (publicInputsEqual : left.publicInputs = right.publicInputs)
    (evaluationsEqual : left.evaluations = right.evaluations) :
    left = right := by
  cases left
  cases right
  simp_all

noncomputable def fieldsOfRunning
    {fullShape : Phi81Relation.Shape}
    (value : Running fullShape) : Fields :=
  fun index => ((runningCodec fullShape).encode value).getD index.val 0

theorem coordinate_fieldsOfRunning
    {fullShape : Phi81Relation.Shape}
    (value : Running fullShape)
    (index : Nat) (bounded : index < runningFieldCount) :
    coordinate (fieldsOfRunning value) index =
      ((runningCodec fullShape).encode value).getD index 0 := by
  rw [coordinate_of_lt _ index bounded]
  rfl

theorem extensionAt_fieldsOfRunning
    {fullShape : Phi81Relation.Shape}
    (value : Running fullShape)
    (index : Fin shape.cubeVariables) :
    extensionAt (fieldsOfRunning value)
        (pointOffset + index.val * 2) =
      value.point.coordinates.getD index.val K.zero := by
  have c0Equal :
      (extensionAt (fieldsOfRunning value)
          (pointOffset + index.val * 2)).c0 =
        (value.point.coordinates.getD index.val K.zero).c0 := by
    rw [extensionAt_c0]
    have bounded : pointOffset + index.val * 2 < runningFieldCount := by
      simpa using point_coordinate_bound index ⟨0, by decide⟩
    rw [coordinate_fieldsOfRunning _ _ bounded]
    have selected :=
      runningCodec_point_getD value index ⟨0, by decide⟩
    simpa only [Fin.val_zero, Nat.add_zero, if_pos] using selected
  have c1Equal :
      (extensionAt (fieldsOfRunning value)
          (pointOffset + index.val * 2)).c1 =
        (value.point.coordinates.getD index.val K.zero).c1 := by
    rw [extensionAt_c1]
    rw [coordinate_fieldsOfRunning _ _
      (point_coordinate_bound index ⟨1, by decide⟩)]
    have selected :=
      runningCodec_point_getD value index ⟨1, by decide⟩
    simpa only [Fin.val_one, if_false, Nat.reduceEqDiff] using selected
  exact k_eq_of_components c0Equal c1Equal

theorem pointOf_fieldsOfRunning
    {fullShape : Phi81Relation.Shape}
    (value : Running fullShape) :
    pointOf (fieldsOfRunning value) = value.point := by
  apply cubePoint_eq_of_coordinates
  apply List.ext_get
  · exact (pointOf (fieldsOfRunning value)).dimension.trans
      value.point.dimension.symm
  · intro index leftBound rightBound
    let typed : Fin shape.cubeVariables :=
      ⟨index, by
        rw [← value.point.dimension]
        exact rightBound⟩
    have reconstructed := extensionAt_fieldsOfRunning value typed
    change
      (List.ofFn (fun position : Fin shape.cubeVariables =>
        extensionAt (fieldsOfRunning value)
          (pointOffset + position.val * 2)))[index]'leftBound =
        value.point.coordinates[index]'rightBound
    rw [List.getElem_ofFn]
    change extensionAt (fieldsOfRunning value)
        (pointOffset + typed.val * 2) =
      value.point.coordinates[index]'rightBound
    rw [reconstructed]
    simpa only [typed] using
      getD_eq_getElem_of_lt value.point.coordinates index K.zero rightBound

theorem bundleOf_fieldsOfRunning
    {fullShape : Phi81Relation.Shape}
    (value : Running fullShape)
    (claim : Fin shape.runningCount) :
    bundleOf (fieldsOfRunning value) claim = value.commitments claim := by
  funext component row coefficient
  change coordinate (fieldsOfRunning value)
      (bundleCoordinateIndex claim component row coefficient) = _
  rw [coordinate_fieldsOfRunning _ _
    (bundle_coordinate_bound claim component row coefficient)]
  exact runningCodec_bundle_getD value claim component row coefficient

theorem publicInputOfFields_fieldsOfRunning
    {fullShape : Phi81Relation.Shape}
    (contract : FullShapeContract fullShape)
    (value : Running fullShape)
    (claim : Fin shape.runningCount) :
    publicInputOfFields (fullShape := fullShape)
        (fieldsOfRunning value) claim = value.publicInputs claim := by
  funext column
  change coordinate (fieldsOfRunning value)
      (publicInputCoordinateIndex claim column) = _
  rw [coordinate_fieldsOfRunning _ _
    (public_input_coordinate_bound contract claim column)]
  exact runningCodec_publicInput_getD contract value claim column

theorem evaluationOf_fieldsOfRunning
    {fullShape : Phi81Relation.Shape}
    (contract : FullShapeContract fullShape)
    (value : Running fullShape)
    (claim : Fin shape.runningCount) :
    evaluationOf (fullShape := fullShape) (fieldsOfRunning value) claim =
      value.evaluations claim := by
  funext matrix coefficient
  have c0Equal :
      (evaluationOf (fullShape := fullShape) (fieldsOfRunning value)
          claim matrix coefficient).c0 =
        (value.evaluations claim matrix coefficient).c0 := by
    rw [evaluationOf_c0]
    rw [coordinate_fieldsOfRunning _ _
      (evaluation_coordinate_bound contract claim matrix coefficient
        ⟨0, by decide⟩)]
    have selected := runningCodec_evaluation_getD contract value claim matrix
      coefficient ⟨0, by decide⟩
    simpa only [Fin.val_zero, if_pos] using selected
  have c1Equal :
      (evaluationOf (fullShape := fullShape) (fieldsOfRunning value)
          claim matrix coefficient).c1 =
        (value.evaluations claim matrix coefficient).c1 := by
    rw [evaluationOf_c1]
    rw [coordinate_fieldsOfRunning _ _
      (evaluation_coordinate_bound contract claim matrix coefficient
        ⟨1, by decide⟩)]
    have selected := runningCodec_evaluation_getD contract value claim matrix
      coefficient ⟨1, by decide⟩
    simpa only [Fin.val_one, if_false, Nat.reduceEqDiff] using selected
  exact k_eq_of_components c0Equal c1Equal

theorem runningOfFields_fieldsOfRunning
    {fullShape : Phi81Relation.Shape}
    (contract : FullShapeContract fullShape)
    (value : Running fullShape) :
    runningOfFields contract (fieldsOfRunning value) = value := by
  apply running_eq_of_fields
  · exact pointOf_fieldsOfRunning value
  · funext claim
    exact bundleOf_fieldsOfRunning value claim
  · funext claim
    exact publicInputOfFields_fieldsOfRunning contract value claim
  · funext claim
    exact evaluationOf_fieldsOfRunning contract value claim

theorem listOfFn_fieldsOfRunning
    {fullShape : Phi81Relation.Shape}
    (contract : FullShapeContract fullShape)
    (value : Running fullShape) :
    List.ofFn (fieldsOfRunning value) =
      (runningCodec fullShape).encode value := by
  apply List.ext_get
  · rw [List.length_ofFn, (runningCodec fullShape).encode_length,
      runningCodec_width contract]
    rfl
  · intro index leftBound rightBound
    rw [List.get_ofFn]
    change ((runningCodec fullShape).encode value).getD index 0 =
      ((runningCodec fullShape).encode value)[index]'rightBound
    exact getD_eq_getElem_of_lt
      ((runningCodec fullShape).encode value) index 0 rightBound

theorem fieldBlockOfRunning_eq
    {fullShape : Phi81Relation.Shape}
    (contract : FullShapeContract fullShape)
    (value : Running fullShape) :
    ProductNifsFieldParser.encode (fieldsOfRunning value) =
      blockOfRunning contract value := by
  apply Subtype.ext
  rw [ProductNifsFieldParser.encode_value,
    blockOfRunning_value]
  rw [ProductNifsFieldParser.valuesList_eq]
  rw [listOfFn_fieldsOfRunning contract value]

/-- Honest parsing reconstructs the complete typed running claim. -/
theorem parse_blockOfRunning
    {fullShape : Phi81Relation.Shape}
    (contract : FullShapeContract fullShape)
    (value : Running fullShape) :
    parse contract (blockOfRunning contract value) = some value := by
  rw [← fieldBlockOfRunning_eq contract value]
  rw [parse, ProductNifsFieldParser.parse_encode]
  change some (runningOfFields contract (fieldsOfRunning value)) = some value
  rw [runningOfFields_fieldsOfRunning contract value]

end Nightstream.Implementation.NebulaV2.ProductNifsRunningParser
