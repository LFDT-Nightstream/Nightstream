import Nightstream.Implementation.Nebula.NIFS.Running.FieldParser

/-!
Contract: executable structured parser for the V2 paper NIFS running claim.

Assurance tier: implementation model.

Owns the exact field-vector sections and coordinate formulas for the shared
point, fourteen mandatory commitment bundles, fourteen public inputs, and
fourteen complete evaluation families.

Does not own the outer full-claim bit envelope, generated parser rows, the
paper NIFS verifier, Rust conformance, or cryptographic soundness.

Emits constraints: no.
-/

set_option autoImplicit false
set_option maxRecDepth 100000
set_option maxHeartbeats 200000

namespace Nightstream.Implementation.Nebula.ProductNifsRunningParser

open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.Implementation.Encoding.NifsCanonicalCodec
open Nightstream.Implementation.Nebula
open Nightstream.Implementation.Nebula.ProductNifsCodec
open Nightstream.Protocol.Nebula
open Nightstream.Protocol.Nebula.CommitmentBundle
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint

def pointFieldCount : Nat := 50
def commitmentsFieldCount : Nat := 54432
def publicInputsFieldCount (fullShape : Phi81Relation.Shape) : Nat :=
  shape.runningCount * fullShape.publicWidth
def evaluationsFieldCount : Nat := 21168

def pointOffset : Nat := 0
def commitmentsOffset : Nat := pointOffset + pointFieldCount
def publicInputsOffset : Nat := commitmentsOffset + commitmentsFieldCount
def evaluationsOffset (fullShape : Phi81Relation.Shape) : Nat :=
  publicInputsOffset + publicInputsFieldCount fullShape

theorem exact_section_counts
    {fullShape : Phi81Relation.Shape}
    (contract : FullShapeContract fullShape) :
    pointFieldCount = 50 /\
      commitmentsFieldCount = 54432 /\
      publicInputsFieldCount fullShape = 7560 /\
      evaluationsFieldCount = 21168 /\
      evaluationsOffset fullShape + evaluationsFieldCount =
        runningFieldCount := by
  simp [pointFieldCount, commitmentsFieldCount, publicInputsFieldCount,
    evaluationsFieldCount, pointOffset, commitmentsOffset,
    publicInputsOffset, evaluationsOffset, shape, contract.publicWidth,
    runningFieldCount, MemoryBoundCcsPublic.coordinateCount]

abbrev Fields := Fin runningFieldCount → F

/-! Total, constant-time selection for the executable parser. Every actual
layout index is later proved to take the bounded branch. -/
def coordinate (fields : Fields) (index : Nat) : F :=
  if bounded : index < runningFieldCount then fields ⟨index, bounded⟩ else 0

theorem coordinate_of_lt
    (fields : Fields) (index : Nat) (bounded : index < runningFieldCount) :
    coordinate fields index = fields ⟨index, bounded⟩ := by
  simp [coordinate, bounded]

private theorem getD_append_left
    {Alpha : Type} (left right : List Alpha) (index : Nat)
    (default : Alpha) (bounded : index < left.length) :
    (left ++ right).getD index default = left.getD index default := by
  rw [List.getD_eq_getElem?_getD, List.getD_eq_getElem?_getD,
    List.getElem?_eq_getElem (by simp; omega),
    List.getElem?_eq_getElem bounded, List.getElem_append_left]

private theorem getD_append_right
    {Alpha : Type} (left right : List Alpha) (index : Nat)
    (default : Alpha) :
    (left ++ right).getD (left.length + index) default =
      right.getD index default := by
  rw [List.getD_eq_getElem?_getD,
    List.getElem?_append_right (by omega)]
  simp only [Nat.add_sub_cancel_left]
  rfl

def extensionAt (fields : Fields) (offset : Nat) : K :=
  ⟨coordinate fields offset, coordinate fields (offset + 1)⟩

theorem extensionAt_c0 (fields : Fields) (offset : Nat) :
    (extensionAt fields offset).c0 = coordinate fields offset :=
  rfl

theorem extensionAt_c1 (fields : Fields) (offset : Nat) :
    (extensionAt fields offset).c1 = coordinate fields (offset + 1) :=
  rfl

def componentIndex : Component → Nat
  | .full => 0
  | .operations => 1
  | .initialSnapshot => 2
  | .finalSnapshot => 3

def pointOf (fields : Fields) : CubePoint K shape.cubeVariables where
  coordinates := List.ofFn fun index : Fin shape.cubeVariables =>
    extensionAt fields (pointOffset + index.val * 2)
  dimension := by simp

theorem point_coordinate_bound
    (index : Fin shape.cubeVariables) (limb : Fin 2) :
    pointOffset + index.val * 2 + limb.val < runningFieldCount := by
  have indexLt := index.isLt
  have limbLt := limb.isLt
  change index.val < 25 at indexLt
  change limb.val < 2 at limbLt
  change 0 + index.val * 2 + limb.val < 83210
  omega

def bundleCoordinateIndex
    (claim : Fin shape.runningCount)
    (component : Component)
    (row : Fin ProductCommitmentAlgebra.Rank)
    (coefficient : Fin ringDegree) : Nat :=
  commitmentsOffset + claim.val * 3888 +
    componentIndex component *
      (ProductCommitmentAlgebra.Rank * ringDegree) +
    row.val * ringDegree + coefficient.val

def bundleOf (fields : Fields) (claim : Fin shape.runningCount) :
    ProductCommitmentAlgebra.BundleValue :=
  fun component row coefficient =>
    coordinate fields
      (bundleCoordinateIndex claim component row coefficient)

theorem bundle_coordinate_bound
    (claim : Fin shape.runningCount)
    (component : Component)
    (row : Fin ProductCommitmentAlgebra.Rank)
    (coefficient : Fin ringDegree) :
    bundleCoordinateIndex claim component row coefficient <
      runningFieldCount := by
  have claimLt := claim.isLt
  have rowLt := row.isLt
  have coefficientLt := coefficient.isLt
  change claim.val < 14 at claimLt
  change row.val < 18 at rowLt
  change coefficient.val < 54 at coefficientLt
  cases component <;>
    simp [bundleCoordinateIndex, commitmentsOffset, pointOffset,
      pointFieldCount, componentIndex, ProductCommitmentAlgebra.Rank,
      MemoryWireGeometry.commitmentRank, ringDegree, runningFieldCount] <;>
    omega

def publicInputCoordinateIndex
    {fullShape : Phi81Relation.Shape}
    (claim : Fin shape.runningCount)
    (column : Fin fullShape.publicWidth) : Nat :=
  publicInputsOffset + claim.val * fullShape.publicWidth + column.val

def publicInputOfFields
    {fullShape : Phi81Relation.Shape}
    (fields : Fields)
    (claim : Fin shape.runningCount) : PublicInput fullShape :=
  fun column => coordinate fields (publicInputCoordinateIndex claim column)

theorem public_input_coordinate_bound
    {fullShape : Phi81Relation.Shape}
    (contract : FullShapeContract fullShape)
    (claim : Fin shape.runningCount)
    (column : Fin fullShape.publicWidth) :
    publicInputCoordinateIndex claim column < runningFieldCount := by
  have claimLt := claim.isLt
  have columnLt := column.isLt
  change claim.val < 14 at claimLt
  have columnLt540 : column.val < 540 := by
    simpa only [contract.publicWidth,
      MemoryBoundCcsPublic.coordinateCount] using columnLt
  have claimWidth : claim.val * fullShape.publicWidth = claim.val * 540 :=
    congrArg (fun width => claim.val * width) contract.publicWidth
  change 54482 + claim.val * fullShape.publicWidth + column.val < 83210
  rw [claimWidth]
  omega

def evaluationCoordinateIndex
    {fullShape : Phi81Relation.Shape}
    (claim : Fin shape.runningCount)
    (matrix : Fin shape.matrixCount)
    (coefficient : Fin shape.coefficientCount)
    (limb : Fin 2) : Nat :=
  evaluationsOffset fullShape + claim.val * 1512 +
    matrix.val * (shape.coefficientCount * 2) +
    coefficient.val * 2 + limb.val

def evaluationOf
    {fullShape : Phi81Relation.Shape}
    (fields : Fields)
    (claim : Fin shape.runningCount) : ProductNifsCodec.Evaluation :=
  fun matrix coefficient =>
    ⟨coordinate fields
        (evaluationCoordinateIndex (fullShape := fullShape)
          claim matrix coefficient ⟨0, by decide⟩),
      coordinate fields
        (evaluationCoordinateIndex (fullShape := fullShape)
          claim matrix coefficient ⟨1, by decide⟩)⟩

theorem evaluationOf_c0
    {fullShape : Phi81Relation.Shape}
    (fields : Fields) (claim : Fin shape.runningCount)
    (matrix : Fin shape.matrixCount)
    (coefficient : Fin shape.coefficientCount) :
    (evaluationOf (fullShape := fullShape) fields
      claim matrix coefficient).c0 =
      coordinate fields
        (evaluationCoordinateIndex (fullShape := fullShape)
          claim matrix coefficient ⟨0, by decide⟩) :=
  rfl

theorem evaluationOf_c1
    {fullShape : Phi81Relation.Shape}
    (fields : Fields) (claim : Fin shape.runningCount)
    (matrix : Fin shape.matrixCount)
    (coefficient : Fin shape.coefficientCount) :
    (evaluationOf (fullShape := fullShape) fields
      claim matrix coefficient).c1 =
      coordinate fields
        (evaluationCoordinateIndex (fullShape := fullShape)
          claim matrix coefficient ⟨1, by decide⟩) :=
  rfl

theorem evaluation_coordinate_bound
    {fullShape : Phi81Relation.Shape}
    (contract : FullShapeContract fullShape)
    (claim : Fin shape.runningCount)
    (matrix : Fin shape.matrixCount)
    (coefficient : Fin shape.coefficientCount)
    (limb : Fin 2) :
    evaluationCoordinateIndex (fullShape := fullShape)
        claim matrix coefficient limb < runningFieldCount := by
  have claimLt := claim.isLt
  have matrixLt := matrix.isLt
  have coefficientLt := coefficient.isLt
  have limbLt := limb.isLt
  change claim.val < 14 at claimLt
  change matrix.val < 14 at matrixLt
  change coefficient.val < 54 at coefficientLt
  change limb.val < 2 at limbLt
  simp [evaluationCoordinateIndex, evaluationsOffset,
    publicInputsOffset, publicInputsFieldCount, commitmentsOffset,
    pointOffset, pointFieldCount, commitmentsFieldCount,
    contract.publicWidth, shape, ringDegree, runningFieldCount,
    MemoryBoundCcsPublic.coordinateCount]
  omega

/-! ## Exact agreement with the canonical field codec -/

theorem commitmentCodec_getD
    (value : ComponentCommitment)
    (row : Fin ProductCommitmentAlgebra.Rank)
    (coefficient : Fin ringDegree) :
    ((commitmentCodec ProductCommitmentAlgebra.Rank).encode value).getD
        (row.val * ringDegree + coefficient.val) 0 =
      value row coefficient := by
  let ringCoordinate : Fin ringFCodec.width :=
    ⟨coefficient.val, by simpa using coefficient.isLt⟩
  have outer := Codec.encodeFin_getD
    ringFCodec ProductCommitmentAlgebra.Rank value row ringCoordinate 0
  have inner := Codec.encodeFin_getD
    fieldCodec ringDegree (value row) coefficient ⟨0, by decide⟩ 0
  calc
    ((commitmentCodec ProductCommitmentAlgebra.Rank).encode value).getD
          (row.val * ringDegree + coefficient.val) 0 =
        (ringFCodec.encode (value row)).getD coefficient.val 0 := by
      simpa [commitmentCodec, ringFCodec_width, ringCoordinate] using outer
    _ = value row coefficient := by
      simpa [ringFCodec, Codec.finFunction, Codec.ofInjectiveEncoding,
        fieldCodec] using inner

theorem commitment_local_bound
    (row : Fin ProductCommitmentAlgebra.Rank)
    (coefficient : Fin ringDegree) :
    row.val * ringDegree + coefficient.val < 972 := by
  have rowLt := row.isLt
  have coefficientLt := coefficient.isLt
  change row.val < 18 at rowLt
  change coefficient.val < 54 at coefficientLt
  change row.val * 54 + coefficient.val < 972
  omega

theorem bundleCodec_sections
    (value : ProductCommitmentAlgebra.BundleValue) :
    bundleCodec.encode value =
      (commitmentCodec ProductCommitmentAlgebra.Rank).encode (value .full) ++
      ((commitmentCodec ProductCommitmentAlgebra.Rank).encode
          (value .operations) ++
        ((commitmentCodec ProductCommitmentAlgebra.Rank).encode
            (value .initialSnapshot) ++
          (commitmentCodec ProductCommitmentAlgebra.Rank).encode
            (value .finalSnapshot))) := by
  rfl

theorem commitment_encoded_length
    (value : ComponentCommitment) :
    ((commitmentCodec ProductCommitmentAlgebra.Rank).encode value).length =
      972 := by
  rw [(commitmentCodec ProductCommitmentAlgebra.Rank).encode_length]
  rfl

theorem bundleCodec_getD
    (value : ProductCommitmentAlgebra.BundleValue)
    (component : Component)
    (row : Fin ProductCommitmentAlgebra.Rank)
    (coefficient : Fin ringDegree) :
    (bundleCodec.encode value).getD
        (componentIndex component * 972 +
          row.val * ringDegree + coefficient.val) 0 =
      value component row coefficient := by
  let localIndex := row.val * ringDegree + coefficient.val
  have localBound : localIndex < 972 := by
    exact commitment_local_bound row coefficient
  rw [bundleCodec_sections]
  cases component with
  | full =>
      have componentShape :
          componentIndex .full * 972 +
              row.val * ringDegree + coefficient.val = localIndex := by
        simp [componentIndex, localIndex]
      rw [componentShape]
      change
        (((commitmentCodec ProductCommitmentAlgebra.Rank).encode
              (value .full)) ++ _).getD localIndex 0 =
          value .full row coefficient
      rw [getD_append_left]
      · exact commitmentCodec_getD (value .full) row coefficient
      · rw [commitment_encoded_length]
        exact localBound
  | operations =>
      have componentShape :
          componentIndex .operations * 972 +
              row.val * ringDegree + coefficient.val =
            972 + localIndex := by
        simp [componentIndex, localIndex]
        omega
      rw [componentShape]
      change
        (((commitmentCodec ProductCommitmentAlgebra.Rank).encode
              (value .full)) ++ _).getD (972 + localIndex) 0 =
          value .operations row coefficient
      rw [← commitment_encoded_length (value .full), getD_append_right]
      rw [getD_append_left]
      · exact commitmentCodec_getD (value .operations) row coefficient
      · rw [commitment_encoded_length]
        exact localBound
  | initialSnapshot =>
      have componentShape :
          componentIndex .initialSnapshot * 972 +
              row.val * ringDegree + coefficient.val =
            1944 + localIndex := by
        simp [componentIndex, localIndex]
        omega
      rw [componentShape]
      change
        (((commitmentCodec ProductCommitmentAlgebra.Rank).encode
              (value .full)) ++ _).getD (1944 + localIndex) 0 =
          value .initialSnapshot row coefficient
      have indexShape : 1944 + localIndex =
          ((commitmentCodec ProductCommitmentAlgebra.Rank).encode
              (value .full)).length + (972 + localIndex) := by
        rw [commitment_encoded_length]
        omega
      rw [indexShape, getD_append_right]
      rw [← commitment_encoded_length (value .operations),
        getD_append_right]
      rw [getD_append_left]
      · exact commitmentCodec_getD
          (value .initialSnapshot) row coefficient
      · rw [commitment_encoded_length]
        exact localBound
  | finalSnapshot =>
      have componentShape :
          componentIndex .finalSnapshot * 972 +
              row.val * ringDegree + coefficient.val =
            2916 + localIndex := by
        simp [componentIndex, localIndex]
        omega
      rw [componentShape]
      change
        (((commitmentCodec ProductCommitmentAlgebra.Rank).encode
              (value .full)) ++ _).getD (2916 + localIndex) 0 =
          value .finalSnapshot row coefficient
      have firstShape : 2916 + localIndex =
          ((commitmentCodec ProductCommitmentAlgebra.Rank).encode
              (value .full)).length + (1944 + localIndex) := by
        rw [commitment_encoded_length]
        omega
      rw [firstShape, getD_append_right]
      have secondShape : 1944 + localIndex =
          ((commitmentCodec ProductCommitmentAlgebra.Rank).encode
              (value .operations)).length + (972 + localIndex) := by
        rw [commitment_encoded_length]
        omega
      rw [secondShape, getD_append_right]
      rw [← commitment_encoded_length (value .initialSnapshot),
        getD_append_right]
      exact commitmentCodec_getD (value .finalSnapshot) row coefficient

theorem pointCodec_getD
    (value : CubePoint K shape.cubeVariables)
    (index : Fin shape.cubeVariables)
    (limb : Fin 2) :
    ((pointCodec shape.cubeVariables).encode value).getD
        (index.val * 2 + limb.val) 0 =
      if limb.val = 0 then
        (value.coordinates.getD index.val K.zero).c0
      else
        (value.coordinates.getD index.val K.zero).c1 := by
  let codecLimb : Fin kCodec.width :=
    ⟨limb.val, by simpa using limb.isLt⟩
  have selected := Codec.encodeFin_getD
    kCodec shape.cubeVariables
      (fun position => value.coordinates.getD position.val K.zero)
      index codecLimb 0
  fin_cases limb <;>
    simpa [pointCodec, Codec.pullback, Codec.fixedList,
      Codec.ofInjectiveEncoding,
      pointData, kCodec_width, codecLimb, kCodec_encode] using selected

theorem publicInputCodec_getD
    {fullShape : Phi81Relation.Shape}
    (value : PublicInput fullShape)
    (column : Fin fullShape.publicWidth) :
    ((publicInputCodec fullShape.publicWidth).encode value).getD
        column.val 0 = value column := by
  have selected := Codec.encodeFin_getD
    fieldCodec fullShape.publicWidth value column ⟨0, by decide⟩ 0
  simpa [publicInputCodec, Codec.finFunction, Codec.ofInjectiveEncoding,
    fieldCodec] using selected

theorem evaluationCodec_getD
    (value : ProductNifsCodec.Evaluation)
    (matrix : Fin shape.matrixCount)
    (coefficient : Fin shape.coefficientCount)
    (limb : Fin 2) :
    (evaluationCodec.encode value).getD
        (matrix.val * (shape.coefficientCount * 2) +
          coefficient.val * 2 + limb.val) 0 =
      if limb.val = 0 then (value matrix coefficient).c0
      else (value matrix coefficient).c1 := by
  let coefficientFamilyCodec :=
    Codec.finFunction shape.coefficientCount kCodec
  let matrixCoordinate : Fin coefficientFamilyCodec.width :=
    ⟨coefficient.val * 2 + limb.val, by
      have coefficientLt := coefficient.isLt
      have limbLt := limb.isLt
      change coefficient.val < 54 at coefficientLt
      change limb.val < 2 at limbLt
      change coefficient.val * 2 + limb.val < 108
      omega⟩
  have outer := Codec.encodeFin_getD
    coefficientFamilyCodec shape.matrixCount value matrix matrixCoordinate 0
  let coefficientCoordinate : Fin kCodec.width :=
    ⟨limb.val, by simpa using limb.isLt⟩
  have middle := Codec.encodeFin_getD
    kCodec shape.coefficientCount (value matrix) coefficient
      coefficientCoordinate 0
  have limbCases : limb.val = 0 ∨ limb.val = 1 := by omega
  calc
    (evaluationCodec.encode value).getD
          (matrix.val * (shape.coefficientCount * 2) +
            coefficient.val * 2 + limb.val) 0 =
        (coefficientFamilyCodec.encode (value matrix)).getD
          (coefficient.val * 2 + limb.val) 0 := by
      simpa [evaluationCodec, coefficientFamilyCodec, Codec.finFunction,
        Codec.ofInjectiveEncoding, kCodec_width, matrixCoordinate,
        Nat.add_assoc] using outer
    _ = (kCodec.encode (value matrix coefficient)).getD limb.val 0 := by
      simpa [kCodec_width, coefficientCoordinate] using middle
    _ = if limb.val = 0 then (value matrix coefficient).c0
        else (value matrix coefficient).c1 := by
      rcases limbCases with zero | one
      · simp [zero, kCodec_encode]
      · simp [one, kCodec_encode]

theorem bundle_local_bound
    (component : Component)
    (row : Fin ProductCommitmentAlgebra.Rank)
    (coefficient : Fin ringDegree) :
    componentIndex component * 972 +
        row.val * ringDegree + coefficient.val < 3888 := by
  have rowLt := row.isLt
  have coefficientLt := coefficient.isLt
  change row.val < 18 at rowLt
  change coefficient.val < 54 at coefficientLt
  cases component <;>
    simp [componentIndex, ringDegree] <;>
    omega

theorem commitmentsSection_getD
    (values : Fin shape.runningCount →
      ProductCommitmentAlgebra.BundleValue)
    (claim : Fin shape.runningCount)
    (component : Component)
    (row : Fin ProductCommitmentAlgebra.Rank)
    (coefficient : Fin ringDegree) :
    (Codec.encodeFin bundleCodec shape.runningCount values).getD
        (claim.val * 3888 + componentIndex component * 972 +
          row.val * ringDegree + coefficient.val) 0 =
      values claim component row coefficient := by
  let bundleCoordinate : Fin bundleCodec.width :=
    ⟨componentIndex component * 972 +
        row.val * ringDegree + coefficient.val, by
      change componentIndex component * 972 +
          row.val * ringDegree + coefficient.val < 3888
      exact bundle_local_bound component row coefficient⟩
  have selected := Codec.encodeFin_getD
    bundleCodec shape.runningCount values claim bundleCoordinate 0
  calc
    (Codec.encodeFin bundleCodec shape.runningCount values).getD
          (claim.val * 3888 + componentIndex component * 972 +
            row.val * ringDegree + coefficient.val) 0 =
        (bundleCodec.encode (values claim)).getD
          (componentIndex component * 972 +
            row.val * ringDegree + coefficient.val) 0 := by
      simpa [bundleCodec_width, bundleCoordinate, Nat.add_assoc] using selected
    _ = values claim component row coefficient :=
      bundleCodec_getD (values claim) component row coefficient

theorem publicInputsSection_getD
    {fullShape : Phi81Relation.Shape}
    (values : Fin shape.runningCount → PublicInput fullShape)
    (claim : Fin shape.runningCount)
    (column : Fin fullShape.publicWidth) :
    (Codec.encodeFin (publicInputCodec fullShape.publicWidth)
        shape.runningCount values).getD
        (claim.val * fullShape.publicWidth + column.val) 0 =
      values claim column := by
  let publicCoordinate :
      Fin (publicInputCodec fullShape.publicWidth).width :=
    ⟨column.val, by simpa using column.isLt⟩
  have selected := Codec.encodeFin_getD
    (publicInputCodec fullShape.publicWidth) shape.runningCount
      values claim publicCoordinate 0
  calc
    (Codec.encodeFin (publicInputCodec fullShape.publicWidth)
          shape.runningCount values).getD
          (claim.val * fullShape.publicWidth + column.val) 0 =
        ((publicInputCodec fullShape.publicWidth).encode
          (values claim)).getD column.val 0 := by
      simpa [publicInputCodec_width, publicCoordinate] using selected
    _ = values claim column := publicInputCodec_getD (values claim) column

theorem evaluationsSection_getD
    (values : Fin shape.runningCount → ProductNifsCodec.Evaluation)
    (claim : Fin shape.runningCount)
    (matrix : Fin shape.matrixCount)
    (coefficient : Fin shape.coefficientCount)
    (limb : Fin 2) :
    (Codec.encodeFin evaluationCodec shape.runningCount values).getD
        (claim.val * 1512 +
          matrix.val * (shape.coefficientCount * 2) +
          coefficient.val * 2 + limb.val) 0 =
      if limb.val = 0 then (values claim matrix coefficient).c0
      else (values claim matrix coefficient).c1 := by
  let evaluationCoordinate : Fin evaluationCodec.width :=
    ⟨matrix.val * (shape.coefficientCount * 2) +
        coefficient.val * 2 + limb.val, by
      have matrixLt := matrix.isLt
      have coefficientLt := coefficient.isLt
      have limbLt := limb.isLt
      change matrix.val < 14 at matrixLt
      change coefficient.val < 54 at coefficientLt
      change limb.val < 2 at limbLt
      change matrix.val * (54 * 2) +
          coefficient.val * 2 + limb.val < 1512
      omega⟩
  have selected := Codec.encodeFin_getD
    evaluationCodec shape.runningCount values claim evaluationCoordinate 0
  calc
    (Codec.encodeFin evaluationCodec shape.runningCount values).getD
          (claim.val * 1512 +
            matrix.val * (shape.coefficientCount * 2) +
            coefficient.val * 2 + limb.val) 0 =
        (evaluationCodec.encode (values claim)).getD
          (matrix.val * (shape.coefficientCount * 2) +
            coefficient.val * 2 + limb.val) 0 := by
      simpa [evaluationCodec_width, evaluationCoordinate,
        Nat.add_assoc] using selected
    _ = if limb.val = 0 then (values claim matrix coefficient).c0
        else (values claim matrix coefficient).c1 :=
      evaluationCodec_getD (values claim) matrix coefficient limb

theorem runningCodec_sections
    {fullShape : Phi81Relation.Shape}
    (value : Running fullShape) :
    (runningCodec fullShape).encode value =
      (pointCodec shape.cubeVariables).encode value.point ++
      (Codec.encodeFin bundleCodec shape.runningCount value.commitments ++
        (Codec.encodeFin (publicInputCodec fullShape.publicWidth)
            shape.runningCount value.publicInputs ++
          Codec.encodeFin evaluationCodec shape.runningCount
            value.evaluations)) := by
  rfl

theorem point_encoded_length
    (value : CubePoint K shape.cubeVariables) :
    ((pointCodec shape.cubeVariables).encode value).length = 50 := by
  rw [(pointCodec shape.cubeVariables).encode_length]
  rfl

theorem commitments_section_length
    (values : Fin shape.runningCount →
      ProductCommitmentAlgebra.BundleValue) :
    (Codec.encodeFin bundleCodec shape.runningCount values).length =
      54432 := by
  rw [Codec.encodeFin_length, bundleCodec_width]
  rfl

theorem public_inputs_section_length
    {fullShape : Phi81Relation.Shape}
    (values : Fin shape.runningCount → PublicInput fullShape) :
    (Codec.encodeFin (publicInputCodec fullShape.publicWidth)
        shape.runningCount values).length =
      shape.runningCount * fullShape.publicWidth := by
  rw [Codec.encodeFin_length, publicInputCodec_width]

theorem evaluations_section_length
    (values : Fin shape.runningCount → ProductNifsCodec.Evaluation) :
    (Codec.encodeFin evaluationCodec shape.runningCount values).length =
      21168 := by
  rw [Codec.encodeFin_length, evaluationCodec_width]
  rfl

theorem runningCodec_point_getD
    {fullShape : Phi81Relation.Shape}
    (value : Running fullShape)
    (index : Fin shape.cubeVariables)
    (limb : Fin 2) :
    ((runningCodec fullShape).encode value).getD
        (pointOffset + index.val * 2 + limb.val) 0 =
      if limb.val = 0 then
        (value.point.coordinates.getD index.val K.zero).c0
      else
        (value.point.coordinates.getD index.val K.zero).c1 := by
  have localBound : index.val * 2 + limb.val < 50 := by
    have indexLt := index.isLt
    have limbLt := limb.isLt
    change index.val < 25 at indexLt
    change limb.val < 2 at limbLt
    omega
  rw [runningCodec_sections]
  simp only [pointOffset, Nat.zero_add]
  change
    (((pointCodec shape.cubeVariables).encode value.point) ++ _).getD
        (index.val * 2 + limb.val) 0 = _
  rw [getD_append_left]
  · exact pointCodec_getD value.point index limb
  · rw [point_encoded_length]
    exact localBound

theorem runningCodec_bundle_getD
    {fullShape : Phi81Relation.Shape}
    (value : Running fullShape)
    (claim : Fin shape.runningCount)
    (component : Component)
    (row : Fin ProductCommitmentAlgebra.Rank)
    (coefficient : Fin ringDegree) :
    ((runningCodec fullShape).encode value).getD
        (bundleCoordinateIndex claim component row coefficient) 0 =
      value.commitments claim component row coefficient := by
  let localIndex := claim.val * 3888 + componentIndex component * 972 +
    row.val * ringDegree + coefficient.val
  have localBound : localIndex < 54432 := by
    have claimLt := claim.isLt
    have bundleBound := bundle_local_bound component row coefficient
    change claim.val < 14 at claimLt
    dsimp only [localIndex]
    omega
  rw [runningCodec_sections]
  have indexShape :
      bundleCoordinateIndex claim component row coefficient =
        ((pointCodec shape.cubeVariables).encode value.point).length +
          localIndex := by
    rw [point_encoded_length]
    simp [bundleCoordinateIndex, commitmentsOffset, pointOffset,
      pointFieldCount, localIndex, ProductCommitmentAlgebra.Rank,
      MemoryWireGeometry.commitmentRank, ringDegree]
    omega
  rw [indexShape, getD_append_right]
  rw [getD_append_left]
  · exact commitmentsSection_getD value.commitments claim component
      row coefficient
  · rw [commitments_section_length]
    exact localBound

theorem runningCodec_publicInput_getD
    {fullShape : Phi81Relation.Shape}
    (contract : FullShapeContract fullShape)
    (value : Running fullShape)
    (claim : Fin shape.runningCount)
    (column : Fin fullShape.publicWidth) :
    ((runningCodec fullShape).encode value).getD
        (publicInputCoordinateIndex claim column) 0 =
      value.publicInputs claim column := by
  let localIndex := claim.val * fullShape.publicWidth + column.val
  have claimLt := claim.isLt
  have columnLt := column.isLt
  change claim.val < 14 at claimLt
  have columnLt540 : column.val < 540 := by
    simpa only [contract.publicWidth,
      MemoryBoundCcsPublic.coordinateCount] using columnLt
  have claimWidth : claim.val * fullShape.publicWidth = claim.val * 540 :=
    congrArg (fun width => claim.val * width) contract.publicWidth
  have localBound : localIndex < 7560 := by
    dsimp only [localIndex]
    rw [claimWidth]
    omega
  rw [runningCodec_sections]
  have pointLength := point_encoded_length value.point
  have commitmentsLength := commitments_section_length value.commitments
  have publicLength :
      (Codec.encodeFin (publicInputCodec fullShape.publicWidth)
          shape.runningCount value.publicInputs).length = 7560 := by
    rw [public_inputs_section_length, contract.publicWidth]
    rfl
  have indexShape : publicInputCoordinateIndex claim column =
      ((pointCodec shape.cubeVariables).encode value.point).length +
        ((Codec.encodeFin bundleCodec shape.runningCount
            value.commitments).length + localIndex) := by
    rw [pointLength, commitmentsLength]
    simp [publicInputCoordinateIndex, publicInputsOffset,
      commitmentsOffset, pointOffset, pointFieldCount,
      commitmentsFieldCount, localIndex]
    omega
  rw [indexShape, getD_append_right, getD_append_right]
  rw [getD_append_left]
  · exact publicInputsSection_getD value.publicInputs claim column
  · rw [publicLength]
    exact localBound

theorem runningCodec_evaluation_getD
    {fullShape : Phi81Relation.Shape}
    (contract : FullShapeContract fullShape)
    (value : Running fullShape)
    (claim : Fin shape.runningCount)
    (matrix : Fin shape.matrixCount)
    (coefficient : Fin shape.coefficientCount)
    (limb : Fin 2) :
    ((runningCodec fullShape).encode value).getD
        (evaluationCoordinateIndex (fullShape := fullShape)
          claim matrix coefficient limb) 0 =
      if limb.val = 0 then
        (value.evaluations claim matrix coefficient).c0
      else
        (value.evaluations claim matrix coefficient).c1 := by
  let localIndex := claim.val * 1512 +
    matrix.val * (shape.coefficientCount * 2) +
    coefficient.val * 2 + limb.val
  rw [runningCodec_sections]
  have pointLength := point_encoded_length value.point
  have commitmentsLength := commitments_section_length value.commitments
  have publicLength :
      (Codec.encodeFin (publicInputCodec fullShape.publicWidth)
          shape.runningCount value.publicInputs).length = 7560 := by
    rw [public_inputs_section_length, contract.publicWidth]
    rfl
  have indexShape :
      evaluationCoordinateIndex (fullShape := fullShape)
          claim matrix coefficient limb =
        ((pointCodec shape.cubeVariables).encode value.point).length +
          ((Codec.encodeFin bundleCodec shape.runningCount
              value.commitments).length +
            ((Codec.encodeFin (publicInputCodec fullShape.publicWidth)
                shape.runningCount value.publicInputs).length +
              localIndex)) := by
    rw [pointLength, commitmentsLength, publicLength]
    simp [evaluationCoordinateIndex, evaluationsOffset,
      publicInputsOffset, publicInputsFieldCount, commitmentsOffset,
      pointOffset, pointFieldCount, commitmentsFieldCount,
      contract.publicWidth, shape, localIndex,
      MemoryBoundCcsPublic.coordinateCount]
    omega
  rw [indexShape, getD_append_right, getD_append_right,
    getD_append_right]
  exact evaluationsSection_getD value.evaluations claim matrix
    coefficient limb

/-- The structured inverse of the exact field-vector layout. -/
def runningOfFields
    {fullShape : Phi81Relation.Shape}
    (_contract : FullShapeContract fullShape)
    (fields : Fields) : Running fullShape where
  point := pointOf fields
  commitments := bundleOf fields
  publicInputs := publicInputOfFields fields
  evaluations := evaluationOf (fullShape := fullShape) fields

/-- Executable parser. Canonical 64-bit field parsing happens before any
semantic object is constructed. -/
def parse
    {fullShape : Phi81Relation.Shape}
    (contract : FullShapeContract fullShape)
    (block : ProductNifsFieldParser.Block runningFieldCount) :
    Option (Running fullShape) :=
  Option.map (runningOfFields contract) (ProductNifsFieldParser.parse block)

/-- Successful structured parsing exposes the exact canonical field vector
from which the paper running claim was constructed. -/
theorem parse_success_fields
    {fullShape : Phi81Relation.Shape}
    (contract : FullShapeContract fullShape)
    {block : ProductNifsFieldParser.Block runningFieldCount}
    {running : Running fullShape}
    (accepted : parse contract block = some running) :
    ∃ fields : Fields,
      ProductNifsFieldParser.parse block = some fields ∧
        running = runningOfFields contract fields := by
  rcases Option.map_eq_some_iff.mp accepted with
    ⟨fields, parsed, constructed⟩
  exact ⟨fields, parsed, constructed.symm⟩

theorem parse_rejects_noncanonical
    {fullShape : Phi81Relation.Shape}
    (contract : FullShapeContract fullShape)
    (block : ProductNifsFieldParser.Block runningFieldCount)
    (notCanonical : ¬ ProductNifsFieldParser.AllCanonical block) :
    parse contract block = none := by
  rw [parse, ProductNifsFieldParser.parse_rejects_noncanonical
    block notCanonical]
  rfl

theorem parse_rejects_modulus_word
    {fullShape : Phi81Relation.Shape}
    (contract : FullShapeContract fullShape)
    (block : ProductNifsFieldParser.Block runningFieldCount)
    (index : Fin runningFieldCount)
    (modulusAt : ProductNifsFieldParser.fieldWord block index =
      CanonicalFieldBits.modulusWord) :
    parse contract block = none := by
  rw [parse, ProductNifsFieldParser.parse_rejects_modulus_word
    block index modulusAt]
  rfl

end Nightstream.Implementation.Nebula.ProductNifsRunningParser
