import Nightstream.Implementation.NebulaV2.NIFS.Running.RunningParser

/-!
Contract: exponent-indexed coordinates of the canonical paper-NIFS running
codec.

The point, fourteen commitment bundles, fourteen public inputs, and fourteen
complete evaluation families have one exact order. All offsets depend on the
selected row exponent where required. The main selectors prove that each
coordinate of `runningCodecFor` is the corresponding typed running value.

This module does not parse bytes, emit rows, select physical columns, verify
NIFS, or prove cryptographic soundness.

Assurance tier: model-level canonical-codec coordinate refinement.
-/

set_option autoImplicit false
set_option maxRecDepth 100000

namespace Nightstream.Implementation.NebulaV2.ProductNifsRunningCoordinatesFor

open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsCanonicalCodecCore
open Nightstream.Implementation.NebulaV2
open Nightstream.Implementation.NebulaV2.ProductNifsCodec
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.Protocol.NebulaV2.CommitmentBundle

def pointFieldCount (rowVariables : Nat) : Nat := rowVariables * 2
def commitmentsFieldCount : Nat := 14 * 3888
def publicInputsFieldCount (fullShape : Phi81Relation.Shape) : Nat :=
  14 * fullShape.publicWidth
def evaluationsFieldCount : Nat := 14 * 1512

def pointOffset : Nat := 0
def commitmentsOffset (rowVariables : Nat) : Nat :=
  pointOffset + pointFieldCount rowVariables
def publicInputsOffset (rowVariables : Nat) : Nat :=
  commitmentsOffset rowVariables + commitmentsFieldCount
def evaluationsOffset (rowVariables : Nat)
    (fullShape : Phi81Relation.Shape) : Nat :=
  publicInputsOffset rowVariables + publicInputsFieldCount fullShape

theorem sections_exact
    {rowVariables : Nat} {fullShape : Phi81Relation.Shape}
    (contract : FullShapeContractFor rowVariables fullShape) :
    evaluationsOffset rowVariables fullShape + evaluationsFieldCount =
      runningFieldCountFor rowVariables := by
  simp [evaluationsOffset, publicInputsOffset, commitmentsOffset,
    pointOffset, pointFieldCount, commitmentsFieldCount,
    publicInputsFieldCount, evaluationsFieldCount, runningFieldCountFor,
    MemoryBoundCcsPublic.coordinateCount, contract.publicWidth]
  omega

def pointCoordinateIndex {rowVariables : Nat}
    (coordinate : Fin rowVariables) (limb : Fin 2) : Nat :=
  pointOffset + coordinate.val * 2 + limb.val

def commitmentCoordinateIndex
    {rowVariables : Nat} (child : Fin 14) (component : Component)
    (row : Fin ProductCommitmentAlgebra.Rank) (lane : Fin ringDegree) : Nat :=
  commitmentsOffset rowVariables + child.val * 3888 +
    ProductNifsRunningParser.componentIndex component * 972 +
      row.val * ringDegree + lane.val

def publicInputCoordinateIndex
    {rowVariables : Nat} {fullShape : Phi81Relation.Shape}
    (child : Fin 14) (column : Fin fullShape.publicWidth) : Nat :=
  publicInputsOffset rowVariables +
    child.val * fullShape.publicWidth + column.val

def evaluationCoordinateIndex
    {rowVariables : Nat} {fullShape : Phi81Relation.Shape}
    (child : Fin 14) (matrix : Fin 14)
    (lane : Fin ringDegree) (limb : Fin 2) : Nat :=
  evaluationsOffset rowVariables fullShape + child.val * 1512 +
    matrix.val * (ringDegree * 2) + lane.val * 2 + limb.val

theorem point_coordinate_bound
    {rowVariables : Nat} (coordinate : Fin rowVariables) (limb : Fin 2) :
    pointCoordinateIndex coordinate limb < runningFieldCountFor rowVariables := by
  have coordinateLt := coordinate.isLt
  have limbLt := limb.isLt
  simp [pointCoordinateIndex, pointOffset, runningFieldCountFor]
  omega

theorem commitment_coordinate_bound
    {rowVariables : Nat} (child : Fin 14) (component : Component)
    (row : Fin ProductCommitmentAlgebra.Rank) (lane : Fin ringDegree) :
    commitmentCoordinateIndex (rowVariables := rowVariables) child component
        row lane < runningFieldCountFor rowVariables := by
  have childLt := child.isLt
  have rowLt := row.isLt
  have laneLt := lane.isLt
  change row.val < 18 at rowLt
  change lane.val < 54 at laneLt
  cases component <;>
    simp [commitmentCoordinateIndex, commitmentsOffset, pointOffset,
      pointFieldCount, ProductNifsRunningParser.componentIndex,
      runningFieldCountFor, ringDegree] <;>
    omega

theorem public_input_coordinate_bound
    {rowVariables : Nat} {fullShape : Phi81Relation.Shape}
    (contract : FullShapeContractFor rowVariables fullShape)
    (child : Fin 14) (column : Fin fullShape.publicWidth) :
    publicInputCoordinateIndex (rowVariables := rowVariables) child column <
      runningFieldCountFor rowVariables := by
  have childLt := child.isLt
  have columnLt := column.isLt
  have columnLt540 : column.val < 540 := by
    simpa only [contract.publicWidth,
      MemoryBoundCcsPublic.coordinateCount] using columnLt
  simp [publicInputCoordinateIndex, publicInputsOffset, commitmentsOffset,
    pointOffset, pointFieldCount, commitmentsFieldCount,
    runningFieldCountFor, MemoryBoundCcsPublic.coordinateCount,
    contract.publicWidth]
  omega

theorem evaluation_coordinate_bound
    {rowVariables : Nat} {fullShape : Phi81Relation.Shape}
    (contract : FullShapeContractFor rowVariables fullShape)
    (child matrix : Fin 14) (lane : Fin ringDegree) (limb : Fin 2) :
    evaluationCoordinateIndex (rowVariables := rowVariables)
        (fullShape := fullShape) child matrix lane limb <
      runningFieldCountFor rowVariables := by
  have childLt := child.isLt
  have matrixLt := matrix.isLt
  have laneLt := lane.isLt
  have limbLt := limb.isLt
  change lane.val < 54 at laneLt
  simp [evaluationCoordinateIndex, evaluationsOffset, publicInputsOffset,
    commitmentsOffset, pointOffset, pointFieldCount,
    commitmentsFieldCount, publicInputsFieldCount, runningFieldCountFor,
    MemoryBoundCcsPublic.coordinateCount, ringDegree, contract.publicWidth]
  omega

/-! ## Exhaustive typed coordinates -/

/-- Verifier-key order of the four bundle components. -/
def componentAt : Fin 4 -> Component :=
  ![.full, .operations, .initialSnapshot, .finalSnapshot]

@[simp] theorem componentIndex_componentAt (component : Fin 4) :
    ProductNifsRunningParser.componentIndex (componentAt component) =
      component.val := by
  fin_cases component <;> rfl

/-- Every field of the running carrier has one typed coordinate. -/
inductive RunningCoordinate (rowVariables : Nat) where
  | point (coordinate : Fin rowVariables) (limb : Fin 2)
  | commitment (child : Fin 14) (component : Fin 4)
      (row : Fin ProductCommitmentAlgebra.Rank) (lane : Fin ringDegree)
  | publicInput (child : Fin 14) (column : Fin 540)
  | evaluation (child : Fin 14) (matrix : Fin 14)
      (lane : Fin ringDegree) (limb : Fin 2)

/-- Canonical flat index of one typed running coordinate. -/
def RunningCoordinate.indexNat {rowVariables : Nat} :
    RunningCoordinate rowVariables -> Nat
  | .point coordinate limb => coordinate.val * 2 + limb.val
  | .commitment child component row lane =>
      commitmentsOffset rowVariables + child.val * 3888 +
        component.val * 972 + row.val * ringDegree + lane.val
  | .publicInput child column =>
      publicInputsOffset rowVariables + child.val * 540 + column.val
  | .evaluation child matrix lane limb =>
      publicInputsOffset rowVariables + 14 * 540 + child.val * 1512 +
        matrix.val * (ringDegree * 2) + lane.val * 2 + limb.val

theorem RunningCoordinate.indexNat_lt
    {rowVariables : Nat} (coordinate : RunningCoordinate rowVariables) :
    coordinate.indexNat < runningFieldCountFor rowVariables := by
  cases coordinate with
  | point coordinate limb =>
      have coordinateLt := coordinate.isLt
      have limbLt := limb.isLt
      simp [RunningCoordinate.indexNat, runningFieldCountFor]
      omega
  | commitment child component row lane =>
      have childLt := child.isLt
      have componentLt := component.isLt
      have rowLt := row.isLt
      have laneLt := lane.isLt
      change row.val < 18 at rowLt
      change lane.val < 54 at laneLt
      simp [RunningCoordinate.indexNat, commitmentsOffset, pointOffset,
        pointFieldCount, publicInputsOffset, runningFieldCountFor, ringDegree]
      omega
  | publicInput child column =>
      have childLt := child.isLt
      have columnLt := column.isLt
      simp [RunningCoordinate.indexNat, publicInputsOffset,
        commitmentsOffset, pointOffset, pointFieldCount,
        commitmentsFieldCount, evaluationsFieldCount, runningFieldCountFor]
      omega
  | evaluation child matrix lane limb =>
      have childLt := child.isLt
      have matrixLt := matrix.isLt
      have laneLt := lane.isLt
      have limbLt := limb.isLt
      change lane.val < 54 at laneLt
      simp [RunningCoordinate.indexNat, publicInputsOffset,
        commitmentsOffset, pointOffset, pointFieldCount,
        commitmentsFieldCount, evaluationsFieldCount, runningFieldCountFor,
        ringDegree]
      omega

def RunningCoordinate.index
    {rowVariables : Nat} (coordinate : RunningCoordinate rowVariables) :
    Fin (runningFieldCountFor rowVariables) :=
  ⟨coordinate.indexNat, coordinate.indexNat_lt⟩

/-- No flat coordinate lies outside the four typed sections. -/
theorem runningCoordinate_surjective
    {rowVariables : Nat}
    (index : Fin (runningFieldCountFor rowVariables)) :
    exists coordinate : RunningCoordinate rowVariables,
      coordinate.index = index := by
  by_cases inPoint : index.val < rowVariables * 2
  · let coordinate : Fin rowVariables :=
      ⟨index.val / 2, by omega⟩
    let limb : Fin 2 := ⟨index.val % 2, Nat.mod_lt _ (by decide)⟩
    refine ⟨.point coordinate limb, ?_⟩
    apply Fin.ext
    have quotient := Nat.mod_add_div index.val 2
    simp [RunningCoordinate.index, RunningCoordinate.indexNat,
      coordinate, limb]
    omega
  by_cases inCommitment :
      index.val < rowVariables * 2 + 14 * 3888
  · have afterPoint : rowVariables * 2 <= index.val := by omega
    let localIndex := index.val - rowVariables * 2
    have localOrigin : rowVariables * 2 + localIndex = index.val := by
      exact Nat.add_sub_of_le afterPoint
    have localLt : localIndex < 14 * 3888 := by omega
    let child : Fin 14 := ⟨localIndex / 3888, by omega⟩
    let childLocal := localIndex % 3888
    let component : Fin 4 := ⟨childLocal / 972, by
      have childLocalLt := Nat.mod_lt localIndex (by decide : 0 < 3888)
      omega⟩
    let componentLocal := childLocal % 972
    let row : Fin ProductCommitmentAlgebra.Rank :=
      ⟨componentLocal / ringDegree, by
        have componentLocalLt := Nat.mod_lt childLocal (by decide : 0 < 972)
        change componentLocal / 54 < 18
        omega⟩
    let lane : Fin ringDegree :=
      ⟨componentLocal % ringDegree, Nat.mod_lt _ (by decide)⟩
    refine ⟨.commitment child component row lane, ?_⟩
    apply Fin.ext
    have childDivision := Nat.mod_add_div localIndex 3888
    have componentDivision := Nat.mod_add_div childLocal 972
    have rowDivision := Nat.mod_add_div componentLocal ringDegree
    norm_num [ringDegree] at rowDivision
    have localDecomposition :
        child.val * 3888 + component.val * 972 +
          row.val * ringDegree + lane.val = localIndex := by
      simp [child, childLocal, component, componentLocal, row, lane]
      norm_num [ringDegree]
      omega
    simp [RunningCoordinate.index, RunningCoordinate.indexNat,
      commitmentsOffset, pointOffset, pointFieldCount]
    omega
  by_cases inPublic :
      index.val < rowVariables * 2 + 14 * 3888 + 14 * 540
  · have afterCommitment :
        rowVariables * 2 + 14 * 3888 <= index.val := by omega
    let localIndex := index.val - (rowVariables * 2 + 14 * 3888)
    have localOrigin :
        rowVariables * 2 + 14 * 3888 + localIndex = index.val := by
      omega
    have localLt : localIndex < 14 * 540 := by omega
    let child : Fin 14 := ⟨localIndex / 540, by omega⟩
    let column : Fin 540 :=
      ⟨localIndex % 540, Nat.mod_lt _ (by decide)⟩
    refine ⟨.publicInput child column, ?_⟩
    apply Fin.ext
    have division := Nat.mod_add_div localIndex 540
    have localDecomposition :
        child.val * 540 + column.val = localIndex := by
      simp [child, column]
      omega
    simp [RunningCoordinate.index, RunningCoordinate.indexNat,
      publicInputsOffset, commitmentsOffset, pointOffset, pointFieldCount]
    norm_num [commitmentsFieldCount, runningFieldCountFor] at *
    omega
  · have afterPublic :
        rowVariables * 2 + 14 * 3888 + 14 * 540 <= index.val := by omega
    let localIndex := index.val -
      (rowVariables * 2 + 14 * 3888 + 14 * 540)
    have localOrigin :
        rowVariables * 2 + 14 * 3888 + 14 * 540 + localIndex = index.val := by
      omega
    have localLt : localIndex < 14 * 1512 := by
      have indexLt : index.val < 83160 + 2 * rowVariables := by
        simpa [runningFieldCountFor] using index.isLt
      omega
    let child : Fin 14 := ⟨localIndex / 1512, by omega⟩
    let childLocal := localIndex % 1512
    let matrix : Fin 14 := ⟨childLocal / 108, by
      have childLocalLt := Nat.mod_lt localIndex (by decide : 0 < 1512)
      omega⟩
    let matrixLocal := childLocal % 108
    let lane : Fin ringDegree := ⟨matrixLocal / 2, by
      have matrixLocalLt := Nat.mod_lt childLocal (by decide : 0 < 108)
      norm_num [ringDegree]
      omega⟩
    let limb : Fin 2 := ⟨matrixLocal % 2, Nat.mod_lt _ (by decide)⟩
    refine ⟨.evaluation child matrix lane limb, ?_⟩
    apply Fin.ext
    have childDivision := Nat.mod_add_div localIndex 1512
    have matrixDivision := Nat.mod_add_div childLocal 108
    have laneDivision := Nat.mod_add_div matrixLocal 2
    have localDecomposition :
        child.val * 1512 + matrix.val * 108 +
          lane.val * 2 + limb.val = localIndex := by
      simp [child, childLocal, matrix, matrixLocal, lane, limb]
      omega
    simp [RunningCoordinate.index, RunningCoordinate.indexNat,
      publicInputsOffset, commitmentsOffset, pointOffset, pointFieldCount,
      ringDegree]
    norm_num [commitmentsFieldCount, runningFieldCountFor, ringDegree] at *
    omega

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

theorem pointCodec_getD
    {rowVariables : Nat} (value : CubePoint K rowVariables)
    (coordinate : Fin rowVariables) (limb : Fin 2) :
    ((pointCodec rowVariables).encode value).getD
        (coordinate.val * 2 + limb.val) 0 =
      if limb.val = 0 then
        (value.coordinates.getD coordinate.val K.zero).c0
      else (value.coordinates.getD coordinate.val K.zero).c1 := by
  let codecLimb : Fin kCodec.width :=
    ⟨limb.val, by simpa using limb.isLt⟩
  have selected := Codec.encodeFin_getD kCodec rowVariables
    (fun position => value.coordinates.getD position.val K.zero)
    coordinate codecLimb 0
  fin_cases limb <;>
    simpa [pointCodec, Codec.pullback, Codec.fixedList,
      Codec.ofInjectiveEncoding, pointData, kCodec_width, codecLimb,
      kCodec_encode] using selected

theorem publicInputCodec_getD
    {fullShape : Phi81Relation.Shape}
    (value : PublicInput fullShape) (column : Fin fullShape.publicWidth) :
    ((publicInputCodec fullShape.publicWidth).encode value).getD
        column.val 0 = value column := by
  have selected := Codec.encodeFin_getD fieldCodec fullShape.publicWidth
    value column ⟨0, by decide⟩ 0
  simpa [publicInputCodec, Codec.finFunction, Codec.ofInjectiveEncoding,
    fieldCodec] using selected

theorem evaluationCodecFor_getD
    {rowVariables : Nat} (value : EvaluationFor rowVariables)
    (matrix : Fin 14) (lane : Fin ringDegree) (limb : Fin 2) :
    ((evaluationCodecFor rowVariables).encode value).getD
        (matrix.val * (ringDegree * 2) + lane.val * 2 + limb.val) 0 =
      if limb.val = 0 then (value matrix lane).c0
      else (value matrix lane).c1 := by
  let laneFamilyCodec := Codec.finFunction ringDegree kCodec
  let matrixCoordinate : Fin laneFamilyCodec.width :=
    ⟨lane.val * 2 + limb.val, by
      have laneLt := lane.isLt
      have limbLt := limb.isLt
      change lane.val < 54 at laneLt
      change limb.val < 2 at limbLt
      change lane.val * 2 + limb.val < 108
      omega⟩
  have outer := Codec.encodeFin_getD laneFamilyCodec 14 value matrix
    matrixCoordinate 0
  let laneCoordinate : Fin kCodec.width :=
    ⟨limb.val, by simpa using limb.isLt⟩
  have inner := Codec.encodeFin_getD kCodec ringDegree (value matrix) lane
    laneCoordinate 0
  have limbCases : limb.val = 0 ∨ limb.val = 1 := by omega
  calc
    ((evaluationCodecFor rowVariables).encode value).getD
          (matrix.val * (ringDegree * 2) +
            lane.val * 2 + limb.val) 0 =
        (laneFamilyCodec.encode (value matrix)).getD
          (lane.val * 2 + limb.val) 0 := by
      simpa [evaluationCodecFor, shapeFor, laneFamilyCodec,
        Codec.finFunction, Codec.ofInjectiveEncoding, kCodec_width,
        matrixCoordinate, Nat.add_assoc] using outer
    _ = (kCodec.encode (value matrix lane)).getD limb.val 0 := by
      simpa [kCodec_width, laneCoordinate] using inner
    _ = if limb.val = 0 then (value matrix lane).c0
        else (value matrix lane).c1 := by
      rcases limbCases with zero | one
      · simp [zero, kCodec_encode]
      · simp [one, kCodec_encode]

theorem publicInputsSection_getD
    {fullShape : Phi81Relation.Shape}
    (values : Fin 14 -> PublicInput fullShape)
    (child : Fin 14) (column : Fin fullShape.publicWidth) :
    (Codec.encodeFin (publicInputCodec fullShape.publicWidth) 14 values).getD
        (child.val * fullShape.publicWidth + column.val) 0 =
      values child column := by
  let localCoordinate : Fin (publicInputCodec fullShape.publicWidth).width :=
    ⟨column.val, by simp⟩
  have selected := Codec.encodeFin_getD
    (publicInputCodec fullShape.publicWidth) 14 values child
      localCoordinate 0
  calc
    (Codec.encodeFin (publicInputCodec fullShape.publicWidth) 14 values).getD
          (child.val * fullShape.publicWidth + column.val) 0 =
        ((publicInputCodec fullShape.publicWidth).encode
          (values child)).getD column.val 0 := by
      simp [publicInputCodec_width, localCoordinate] at selected
      exact selected
    _ = values child column := publicInputCodec_getD (values child) column

theorem evaluationsSection_getD
    {rowVariables : Nat}
    (values : Fin 14 -> EvaluationFor rowVariables)
    (child matrix : Fin 14) (lane : Fin ringDegree) (limb : Fin 2) :
    (Codec.encodeFin (evaluationCodecFor rowVariables) 14 values).getD
        (child.val * 1512 + matrix.val * (ringDegree * 2) +
          lane.val * 2 + limb.val) 0 =
      if limb.val = 0 then (values child matrix lane).c0
      else (values child matrix lane).c1 := by
  let localCoordinate : Fin (evaluationCodecFor rowVariables).width :=
    ⟨matrix.val * (ringDegree * 2) + lane.val * 2 + limb.val, by
      have matrixLt := matrix.isLt
      have laneLt := lane.isLt
      have limbLt := limb.isLt
      change matrix.val < 14 at matrixLt
      change lane.val < 54 at laneLt
      change limb.val < 2 at limbLt
      change matrix.val * (54 * 2) + lane.val * 2 + limb.val < 1512
      omega⟩
  have selected := Codec.encodeFin_getD (evaluationCodecFor rowVariables)
    14 values child localCoordinate 0
  calc
    (Codec.encodeFin (evaluationCodecFor rowVariables) 14 values).getD
          (child.val * 1512 + matrix.val * (ringDegree * 2) +
            lane.val * 2 + limb.val) 0 =
        ((evaluationCodecFor rowVariables).encode (values child)).getD
          (matrix.val * (ringDegree * 2) +
            lane.val * 2 + limb.val) 0 := by
      simpa [evaluationCodecFor_width, localCoordinate, Nat.add_assoc]
        using selected
    _ = if limb.val = 0 then (values child matrix lane).c0
        else (values child matrix lane).c1 :=
      evaluationCodecFor_getD (values child) matrix lane limb

theorem runningCodecFor_sections
    {rowVariables : Nat} {fullShape : Phi81Relation.Shape}
    (value : RunningFor rowVariables fullShape) :
    (runningCodecFor rowVariables fullShape).encode value =
      (pointCodec rowVariables).encode value.point ++
      (Codec.encodeFin bundleCodec 14 value.commitments ++
        (Codec.encodeFin (publicInputCodec fullShape.publicWidth) 14
            value.publicInputs ++
          Codec.encodeFin (evaluationCodecFor rowVariables) 14
            value.evaluations)) := by
  rfl

theorem point_encoded_length
    {rowVariables : Nat} (value : CubePoint K rowVariables) :
    ((pointCodec rowVariables).encode value).length = rowVariables * 2 := by
  rw [(pointCodec rowVariables).encode_length, pointCodec_width]

theorem commitments_section_length
    (values : Fin 14 -> ProductCommitmentAlgebra.BundleValue) :
    (Codec.encodeFin bundleCodec 14 values).length = commitmentsFieldCount := by
  rw [Codec.encodeFin_length, bundleCodec_width]
  rfl

theorem commitmentsSectionFor_getD
    (values : Fin 14 -> ProductCommitmentAlgebra.BundleValue)
    (child : Fin 14) (component : Component)
    (row : Fin ProductCommitmentAlgebra.Rank) (lane : Fin ringDegree) :
    (Codec.encodeFin bundleCodec 14 values).getD
        (child.val * 3888 +
          ProductNifsRunningParser.componentIndex component * 972 +
            row.val * ringDegree + lane.val) 0 =
      values child component row lane := by
  let localCoordinate : Fin bundleCodec.width :=
    ⟨ProductNifsRunningParser.componentIndex component * 972 +
        row.val * ringDegree + lane.val, by
      simpa only [bundleCodec_width] using
        ProductNifsRunningParser.bundle_local_bound component row lane⟩
  have selected := Codec.encodeFin_getD bundleCodec 14 values child
    localCoordinate 0
  calc
    (Codec.encodeFin bundleCodec 14 values).getD
          (child.val * 3888 +
            ProductNifsRunningParser.componentIndex component * 972 +
              row.val * ringDegree + lane.val) 0 =
        (bundleCodec.encode (values child)).getD
          (ProductNifsRunningParser.componentIndex component * 972 +
            row.val * ringDegree + lane.val) 0 := by
      simpa [bundleCodec_width, localCoordinate, Nat.add_assoc] using selected
    _ = values child component row lane :=
      ProductNifsRunningParser.bundleCodec_getD
        (values child) component row lane

theorem public_inputs_section_length
    {fullShape : Phi81Relation.Shape}
    (values : Fin 14 -> PublicInput fullShape) :
    (Codec.encodeFin (publicInputCodec fullShape.publicWidth) 14 values).length =
      publicInputsFieldCount fullShape := by
  rw [Codec.encodeFin_length, publicInputCodec_width]
  rfl

theorem runningCodecFor_point_getD
    {rowVariables : Nat} {fullShape : Phi81Relation.Shape}
    (value : RunningFor rowVariables fullShape)
    (coordinate : Fin rowVariables) (limb : Fin 2) :
    ((runningCodecFor rowVariables fullShape).encode value).getD
        (pointCoordinateIndex coordinate limb) 0 =
      if limb.val = 0 then
        (value.point.coordinates.getD coordinate.val K.zero).c0
      else (value.point.coordinates.getD coordinate.val K.zero).c1 := by
  rw [runningCodecFor_sections]
  simp only [pointCoordinateIndex, pointOffset, Nat.zero_add]
  rw [getD_append_left]
  · exact pointCodec_getD value.point coordinate limb
  · rw [point_encoded_length]
    have coordinateLt := coordinate.isLt
    have limbLt := limb.isLt
    omega

theorem runningCodecFor_commitment_getD
    {rowVariables : Nat} {fullShape : Phi81Relation.Shape}
    (value : RunningFor rowVariables fullShape)
    (child : Fin 14) (component : Component)
    (row : Fin ProductCommitmentAlgebra.Rank) (lane : Fin ringDegree) :
    ((runningCodecFor rowVariables fullShape).encode value).getD
        (commitmentCoordinateIndex (rowVariables := rowVariables) child
          component row lane) 0 =
      value.commitments child component row lane := by
  let localIndex := child.val * 3888 +
    ProductNifsRunningParser.componentIndex component * 972 +
      row.val * ringDegree + lane.val
  rw [runningCodecFor_sections]
  have pointLength :
      ((pointCodec rowVariables).encode value.point).length =
        rowVariables * 2 := by
    rw [(pointCodec rowVariables).encode_length, pointCodec_width]
  have indexShape :
      commitmentCoordinateIndex (rowVariables := rowVariables) child component
          row lane =
        ((pointCodec rowVariables).encode value.point).length + localIndex := by
    rw [pointLength]
    simp [commitmentCoordinateIndex, commitmentsOffset, pointOffset,
      pointFieldCount, localIndex]
    omega
  rw [indexShape, getD_append_right]
  have localBound : localIndex <
      (Codec.encodeFin bundleCodec 14 value.commitments).length := by
    rw [commitments_section_length]
    have childLt := child.isLt
    have bundleLt :=
      ProductNifsRunningParser.bundle_local_bound component row lane
    simp only [commitmentsFieldCount]
    omega
  rw [getD_append_left]
  · exact commitmentsSectionFor_getD value.commitments child component row lane
  · exact localBound

theorem runningCodecFor_publicInput_getD
    {rowVariables : Nat} {fullShape : Phi81Relation.Shape}
    (value : RunningFor rowVariables fullShape)
    (child : Fin 14) (column : Fin fullShape.publicWidth) :
    ((runningCodecFor rowVariables fullShape).encode value).getD
        (publicInputCoordinateIndex (rowVariables := rowVariables)
          child column) 0 = value.publicInputs child column := by
  let localIndex := child.val * fullShape.publicWidth + column.val
  rw [runningCodecFor_sections]
  have pointLength :
      ((pointCodec rowVariables).encode value.point).length =
        rowVariables * 2 := by
    rw [(pointCodec rowVariables).encode_length, pointCodec_width]
  have commitmentsLength := commitments_section_length value.commitments
  have indexShape :
      publicInputCoordinateIndex (rowVariables := rowVariables) child column =
        ((pointCodec rowVariables).encode value.point).length +
          ((Codec.encodeFin bundleCodec 14 value.commitments).length +
            localIndex) := by
    rw [pointLength, commitmentsLength]
    simp [publicInputCoordinateIndex, publicInputsOffset,
      commitmentsOffset, pointOffset, pointFieldCount, localIndex]
    omega
  rw [indexShape, getD_append_right, getD_append_right]
  have localBound : localIndex <
      (Codec.encodeFin (publicInputCodec fullShape.publicWidth) 14
        value.publicInputs).length := by
    rw [public_inputs_section_length]
    change child.val * fullShape.publicWidth + column.val <
      14 * fullShape.publicWidth
    calc
      child.val * fullShape.publicWidth + column.val <
          child.val * fullShape.publicWidth + fullShape.publicWidth :=
        Nat.add_lt_add_left column.isLt _
      _ = (child.val + 1) * fullShape.publicWidth := by
        simp [Nat.add_mul]
      _ ≤ 14 * fullShape.publicWidth :=
        Nat.mul_le_mul_right fullShape.publicWidth
          (Nat.succ_le_iff.mpr child.isLt)
  rw [getD_append_left]
  · exact publicInputsSection_getD value.publicInputs child column
  · exact localBound

theorem runningCodecFor_evaluation_getD
    {rowVariables : Nat} {fullShape : Phi81Relation.Shape}
    (value : RunningFor rowVariables fullShape)
    (child matrix : Fin 14) (lane : Fin ringDegree) (limb : Fin 2) :
    ((runningCodecFor rowVariables fullShape).encode value).getD
        (evaluationCoordinateIndex (rowVariables := rowVariables)
          (fullShape := fullShape) child matrix lane limb) 0 =
      if limb.val = 0 then (value.evaluations child matrix lane).c0
      else (value.evaluations child matrix lane).c1 := by
  let localIndex := child.val * 1512 + matrix.val * (ringDegree * 2) +
    lane.val * 2 + limb.val
  rw [runningCodecFor_sections]
  have pointLength :
      ((pointCodec rowVariables).encode value.point).length =
        rowVariables * 2 := by
    rw [(pointCodec rowVariables).encode_length, pointCodec_width]
  have commitmentsLength := commitments_section_length value.commitments
  have publicLength := public_inputs_section_length value.publicInputs
  have indexShape :
      evaluationCoordinateIndex (rowVariables := rowVariables)
          (fullShape := fullShape) child matrix lane limb =
        ((pointCodec rowVariables).encode value.point).length +
          ((Codec.encodeFin bundleCodec 14 value.commitments).length +
            ((Codec.encodeFin (publicInputCodec fullShape.publicWidth) 14
              value.publicInputs).length + localIndex)) := by
    rw [pointLength, commitmentsLength, publicLength]
    simp [evaluationCoordinateIndex, evaluationsOffset, publicInputsOffset,
      commitmentsOffset, pointOffset, pointFieldCount,
      publicInputsFieldCount, localIndex]
    omega
  rw [indexShape, getD_append_right, getD_append_right,
    getD_append_right]
  exact evaluationsSection_getD value.evaluations child matrix lane limb

end Nightstream.Implementation.NebulaV2.ProductNifsRunningCoordinatesFor
