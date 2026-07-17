import Nightstream.SuperNeo.Concrete.Phi81Relation.EvaluationHomomorphism.PiRLC
import Nightstream.SuperNeo.Concrete.Phi81Relation.EvaluationHomomorphism.RingFLaws
import Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270.PiCcsDomain
import Nightstream.SuperNeo.Folding.PiCCS.OutputClaims.EvaluationHomomorphism.PackedBlockAction
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.BooleanProduct

/-!
Padded identity evaluation for the five-ring F-prime carrier.

Assurance tier: model-level.

Owns: one separate 512-row by 270-column verifier-owned matrix; its exact
`lane + 64 * block` row codec; zero padding outside five blocks and 54 lanes;
and the semantic evaluation theorem relating this matrix at `[0^6] ++ s` to
the canonical packed block projection.

Its nonzero entries are exactly:

```text
M[lane + 64 * block, carrierLane + 54 * carrierBlock] = 1
iff block = carrierBlock and lane = carrierLane.
```

Does not own: the paper's existing `M_1`, the active thirteen-matrix
production relation, a shared PiRLC point/structure redesign, transcript
authority, Rust decoding, R1CS rows, row removal, or constraint counts.

Emits constraints: no.

Authority boundary: this is an independently defined auxiliary CE matrix.
Its entries and padding are definitions. Until a later protocol composition
binds this separate opening to the recursive verifier, the equality below is
not permission to replace or remove a production check.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `nifs.pi_ccs.aux_packed_identity.row.codec` | row `a + 64b` owns live block `b` and lane `a` | computed | `decodeRow?` |
| `nifs.pi_ccs.aux_packed_identity.matrix.live` | a live row selects carrier column `a + 54b` with coefficient one | computed | `paddedIdentityMatrix` |
| `nifs.pi_ccs.aux_packed_identity.matrix.padding` | blocks 5--7 and lanes 54--63 are zero | computed | `paddedIdentityMatrix` |
| `nifs.pi_ccs.aux_packed_identity.row.semantic` | a live row evaluates to the selected Phi81 kernel image; padding evaluates to zero | derived | `rowRing_eq_expectedRowRing` |
| `nifs.pi_ccs.aux_packed_identity.point` | six zero lane coordinates followed by three block coordinates form the nine-variable point | computed | `packedPoint` |
| `nifs.pi_ccs.aux_packed_identity.evaluation` | the auxiliary CE evaluation equals canonical packed `y_zcol` | derived | `matrixEvaluation_packedPoint_eq_packedYZcol` |
| `nifs.pi_ccs.aux_packed_identity.ce_witness_claim` | all-lane CE evaluation equality, or semantic CE witness membership, binds the sole claimed evaluation to packed `y_zcol` relative to that witness | checked / derived | `claimedEvaluation_eq_packedYZcol_of_evaluationsBound`, `claimedEvaluation_eq_packedYZcol_of_ceHolds` |
-/

namespace Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270.PaddedIdentityEvaluation

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation.EvaluationHomomorphism
open Nightstream.SuperNeo.Folding.PiCCS.OutputClaims.EvaluationHomomorphism
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open PaperLinearAlgebra

/-! ## Fixed profile and live-row codec -/

/-- Fixed relation shape of the separate padded-identity opening. -/
def shape : Phi81Relation.Shape where
  rowVariables :=
    PiCcsDomain.blockDomain.blockVariables +
      PiCcsDomain.blockDomain.laneVariables
  logicalWidth := alignedPublicWidth
  matrixCount := 1
  publicRingColumns := FPrimeCarrier270.publicRingColumns
  publicFits := by decide

@[simp] theorem shape_rowVariables : shape.rowVariables = 9 := by
  rfl

@[simp] theorem shape_logicalWidth : shape.logicalWidth = 270 := by
  rfl

@[simp] theorem shape_carrierWidth : shape.carrierWidth = 270 := by
  decide

@[simp] theorem shape_blockCount :
    Phi81ColumnLayout.blockCount shape.carrierWidth = 5 := by
  decide

/-- The matching Split-NC carrier shape. Source arities are irrelevant to the
packed assignment projection and are fixed to zero here. -/
def semanticShape : SemanticShape :=
  PiCcsDomain.plainShape shape.rowVariables 0 0 1

@[simp] theorem semanticShape_carrierWidth :
    semanticShape.carrierWidth = shape.carrierWidth := by
  decide

/-- The empty constraint polynomial is inert: this component proves only the
CE evaluation of its sole matrix. -/
def emptyConstraintPolynomial :
    CCSResidualTable.ConstraintPolynomial F shape.matrixCount where
  degreeBound := 1
  terms := []
  termsBelowDegree := by simp

/-- Padded Boolean-row number in canonical little-endian order. -/
def rowNumber (row : BooleanVertex shape.rowVariables) : Nat :=
  NumericBooleanDomain.index row

/-- Block coordinate of one padded row, using the 64-lane Boolean stride. -/
def rowBlockNumber (row : BooleanVertex shape.rowVariables) : Nat :=
  rowNumber row / PiCcsDomain.blockDomain.laneCount

/-- Lane coordinate of one padded row, using the 64-lane Boolean stride. -/
def rowLaneNumber (row : BooleanVertex shape.rowVariables) : Nat :=
  rowNumber row % PiCcsDomain.blockDomain.laneCount

/-- Numeric padded-row owner of one live carrier block/lane pair. -/
def paddedRowNumber
    (block : Fin (Phi81ColumnLayout.blockCount shape.carrierWidth))
    (lane : Fin ringDegree) : Nat :=
  lane.val + PiCcsDomain.blockDomain.laneCount * block.val

/-- Stride 64 is wider than every live lane, so a padded row has one unique
live block/lane owner. -/
theorem paddedRowNumber_eq_iff
    (leftBlock rightBlock :
      Fin (Phi81ColumnLayout.blockCount shape.carrierWidth))
    (leftLane rightLane : Fin ringDegree) :
    paddedRowNumber leftBlock leftLane =
        paddedRowNumber rightBlock rightLane ↔
      leftBlock = rightBlock ∧ leftLane = rightLane := by
  constructor
  · intro equal
    have leftLaneBound : leftLane.val < 64 := by
      have := leftLane.isLt
      simp only [ringDegree] at this
      omega
    have rightLaneBound : rightLane.val < 64 := by
      have := rightLane.isLt
      simp only [ringDegree] at this
      omega
    simp only [paddedRowNumber, PiCcsDomain.blockDomain_laneCount] at equal
    have blockEqual : leftBlock.val = rightBlock.val := by omega
    have laneEqual : leftLane.val = rightLane.val := by omega
    exact ⟨Fin.ext blockEqual, Fin.ext laneEqual⟩
  · rintro ⟨rfl, rfl⟩
    rfl

/-- Decode a row exactly when both its block and lane are live. -/
def decodeRow? (row : BooleanVertex shape.rowVariables) :
    Option
      (Fin (Phi81ColumnLayout.blockCount shape.carrierWidth) ×
        Fin ringDegree) :=
  if blockLive : rowBlockNumber row <
      Phi81ColumnLayout.blockCount shape.carrierWidth then
    if laneLive : rowLaneNumber row < ringDegree then
      some (⟨rowBlockNumber row, blockLive⟩,
        ⟨rowLaneNumber row, laneLive⟩)
    else
      none
  else
    none

/-- The decoder names a live owner exactly when the row number is that
owner's stride-64 encoding. -/
theorem decodeRow?_eq_some_iff
    (row : BooleanVertex shape.rowVariables)
    (block : Fin (Phi81ColumnLayout.blockCount shape.carrierWidth))
    (lane : Fin ringDegree) :
    decodeRow? row = some (block, lane) ↔
      rowNumber row = paddedRowNumber block lane := by
  constructor
  · intro decoded
    unfold decodeRow? at decoded
    split at decoded
    next blockLive =>
      split at decoded
      next laneLive =>
        have pairEqual := Option.some.inj decoded
        have blockEqual : rowBlockNumber row = block.val :=
          congrArg (fun owner => owner.1.val) pairEqual
        have laneEqual : rowLaneNumber row = lane.val :=
          congrArg (fun owner => owner.2.val) pairEqual
        unfold paddedRowNumber
        rw [← blockEqual, ← laneEqual]
        unfold rowBlockNumber rowLaneNumber
        exact (Nat.mod_add_div (rowNumber row)
          PiCcsDomain.blockDomain.laneCount).symm
      next lanePadding => contradiction
    next blockPadding => contradiction
  · intro encoded
    have laneLt64 : lane.val < PiCcsDomain.blockDomain.laneCount := by
      have laneBound := lane.isLt
      simp only [ringDegree, PiCcsDomain.blockDomain_laneCount] at laneBound ⊢
      omega
    have laneLt64Numeral : lane.val < 64 := by
      simpa only [PiCcsDomain.blockDomain_laneCount] using laneLt64
    have blockNumberEqual : rowBlockNumber row = block.val := by
      unfold rowBlockNumber
      rw [encoded]
      simp only [paddedRowNumber, PiCcsDomain.blockDomain_laneCount]
      rw [Nat.add_mul_div_left lane.val block.val (by decide)]
      rw [Nat.div_eq_of_lt laneLt64Numeral, Nat.zero_add]
    have laneNumberEqual : rowLaneNumber row = lane.val := by
      unfold rowLaneNumber
      rw [encoded]
      simp only [paddedRowNumber, PiCcsDomain.blockDomain_laneCount]
      rw [Nat.add_mul_mod_self_left]
      exact Nat.mod_eq_of_lt laneLt64Numeral
    unfold decodeRow?
    rw [dif_pos (blockNumberEqual ▸ block.isLt)]
    rw [dif_pos (laneNumberEqual ▸ lane.isLt)]
    congr 2 <;> apply Fin.ext
    · exact blockNumberEqual
    · exact laneNumberEqual

/-- The sole original matrix is a stride-64 padded injection into the
block-major stride-54 carrier. -/
def paddedIdentityMatrix :
    BooleanMatrix F shape.rowVariables shape.logicalWidth :=
  fun row column =>
    let owner := Phi81ColumnLayout.decode column
    if rowNumber row = paddedRowNumber owner.1 owner.2 then 1 else 0

/-- Sole-matrix family for the auxiliary relation. -/
def matrices : Fin shape.matrixCount ->
    BooleanMatrix F shape.rowVariables shape.logicalWidth :=
  fun _ => paddedIdentityMatrix

/-- Verifier-owned auxiliary relation structure. -/
def system : Phi81Relation.Structure shape where
  matrices := matrices
  constraintPolynomial := emptyConstraintPolynomial

/-- Sole matrix index. -/
def matrixIndex : Fin shape.matrixCount := ⟨0, by decide⟩

/-- Reading the completed matrix source at a live block/lane position exposes
exactly the stride-64 padded identity selector. -/
theorem paddedMatrixEntry_eq
    (row : BooleanVertex shape.rowVariables)
    (block : Fin (Phi81ColumnLayout.blockCount shape.carrierWidth))
    (lane : Fin ringDegree) :
    system.matrixSource.paddedMatrixEntry ConcreteCarrier.baseOps
        matrixIndex row block lane =
      if rowNumber row = paddedRowNumber block lane then 1 else 0 := by
  unfold MatrixCoefficientSource.MatrixSource.paddedMatrixEntry
  have columnLayoutEq :
      system.matrixSource.columnLayout =
        Phi81CarrierLayout.layout shape.logicalWidth := by
    rfl
  rw [columnLayoutEq]
  rw [Phi81CarrierLayout.layout_encode?_isSome block lane]
  change
    system.matrixSource.matrices matrixIndex row
        (⟨Phi81ColumnLayout.flatIndex block lane,
          Phi81CarrierLayout.flatIndex_lt_carrierWidth block lane⟩ :
          Fin shape.carrierWidth) = _
  have flatLtLogical :
      Phi81ColumnLayout.flatIndex block lane < shape.logicalWidth := by
    have flatLt := Phi81CarrierLayout.flatIndex_lt_carrierWidth block lane
    simpa only [shape_carrierWidth, shape_logicalWidth] using flatLt
  let logicalColumn : Fin shape.logicalWidth :=
    ⟨Phi81ColumnLayout.flatIndex block lane, flatLtLogical⟩
  have carrierColumnEq :
      (⟨Phi81ColumnLayout.flatIndex block lane,
          Phi81CarrierLayout.flatIndex_lt_carrierWidth block lane⟩ :
        Fin shape.carrierWidth) =
        Phi81CarrierLayout.embedLogical logicalColumn := by
    apply Fin.ext
    rfl
  rw [carrierColumnEq]
  have sourceAtLogical :
      system.matrixSource.matrices matrixIndex row
          (Phi81CarrierLayout.embedLogical logicalColumn) =
        system.matrices matrixIndex row logicalColumn := by
    exact Phi81MatrixSource.source_matrix_embedLogical
      shape.rowVariables 0 0 shape.matrixCount shape.logicalWidth
      system.matrices system.constraintPolynomial matrixIndex row logicalColumn
  rw [sourceAtLogical]
  change paddedIdentityMatrix row logicalColumn = _
  unfold paddedIdentityMatrix
  have encodedLogical :
      Phi81ColumnLayout.encode? block lane = some logicalColumn := by
    have flatLtCarrier :=
      Phi81CarrierLayout.flatIndex_lt_carrierWidth block lane
    have flatLtShapeCarrier :
        Phi81ColumnLayout.flatIndex block lane < shape.carrierWidth := by
      exact flatLtCarrier
    unfold Phi81ColumnLayout.encode?
    rw [dif_pos flatLtShapeCarrier]
    congr 1
  have decodedLogical :
      Phi81ColumnLayout.decode logicalColumn = (block, lane) :=
    Phi81ColumnLayout.decode_encode block lane logicalColumn encodedLogical
  rw [decodedLogical]

/-! ## Matrix-row semantics -/

/-- Independent semantic row value of the auxiliary matrix. -/
def expectedRowRing
    (assignment : Phi81Relation.Assignment shape)
    (row : BooleanVertex shape.rowVariables) : RingF :=
  match decodeRow? row with
  | none => ringFZero
  | some (block, lane) =>
      CarrierAction.kernelImage lane
        (CarrierAction.assignmentBlock assignment block)

/-- Every padded-identity row has exactly its independently specified live or
zero semantic value. -/
theorem rowRing_eq_expectedRowRing
    (assignment : Phi81Relation.Assignment shape)
    (row : BooleanVertex shape.rowVariables) :
    PiRLC.rowRing system assignment matrixIndex row =
      expectedRowRing assignment row := by
  unfold expectedRowRing
  generalize decodedEq : decodeRow? row = decoded
  cases decoded with
  | none =>
      apply PiRLC.rowRing_eq_zero_of_padded_row_zero
      intro block lane
      rw [paddedMatrixEntry_eq]
      rw [if_neg]
      intro encoded
      have live := (decodeRow?_eq_some_iff row block lane).2 encoded
      rw [decodedEq] at live
      contradiction
  | some owner =>
      rcases owner with ⟨selectedBlock, selectedLane⟩
      apply PiRLC.rowRing_eq_kernelImage_of_unit_padded_row
      intro block lane
      rw [paddedMatrixEntry_eq]
      have selectedEncoded :
          rowNumber row = paddedRowNumber selectedBlock selectedLane :=
        (decodeRow?_eq_some_iff row selectedBlock selectedLane).1 decodedEq
      by_cases blockEqual : block = selectedBlock
      · subst block
        rw [if_pos rfl]
        by_cases laneEqual : lane = selectedLane
        · subst lane
          rw [if_pos rfl, if_pos selectedEncoded]
        · rw [if_neg laneEqual, if_neg]
          intro encoded
          have ownersEqual :=
            (paddedRowNumber_eq_iff selectedBlock selectedBlock
              lane selectedLane).1 (encoded.symm.trans selectedEncoded)
          exact laneEqual ownersEqual.2
      · rw [if_neg blockEqual, if_neg]
        intro encoded
        have ownersEqual :=
          (paddedRowNumber_eq_iff block selectedBlock lane selectedLane).1
            (encoded.symm.trans selectedEncoded)
        exact blockEqual ownersEqual.1

/-- At an arbitrary nine-coordinate point, the sole auxiliary matrix is the
Boolean MLE of the independently specified padded row semantics. -/
theorem matrixEvaluation_eq_expectedRows
    (assignment : Phi81Relation.Assignment shape)
    (point : Phi81Relation.Point shape) :
    Phi81Relation.matrixEvaluation system assignment point matrixIndex =
      RingKAction.evaluateRows
        (fun row => RingKAction.embedChallenge
          (expectedRowRing assignment row)) point := by
  rw [PiRLC.matrixEvaluation_eq_evaluateRows]
  apply congrArg (fun rows => RingKAction.evaluateRows rows point)
  funext row
  rw [rowRing_eq_expectedRowRing]

/-! ## Packed projection -/

/-- The relation assignment viewed through the matching Split-NC shape. -/
def asPackedAssignment
    (assignment : Phi81Relation.Assignment shape) :
    PackedBlockAction.SemanticAssignment semanticShape :=
  fun column => assignment ⟨column.val, by
    simpa only [semanticShape_carrierWidth] using column.isLt⟩

/-- Canonical row with lane bits first and block bits second. -/
def zeroLaneRow
    (block : BooleanVertex PiCcsDomain.blockDomain.blockVariables) :
    BooleanVertex shape.rowVariables :=
  (BooleanVertex.zeros PiCcsDomain.blockDomain.laneVariables).withLowPrefix
    block

/-- The zero-lane row has the exact stride-64 numeric encoding. -/
theorem rowNumber_zeroLaneRow
    (block : BooleanVertex PiCcsDomain.blockDomain.blockVariables) :
    rowNumber (zeroLaneRow block) =
      PiCcsDomain.blockDomain.laneCount *
        (BlockNcDomain.blockIndex block).val := by
  unfold rowNumber zeroLaneRow BlockNcDomain.blockIndex
  calc
    NumericBooleanDomain.index
        ((BooleanVertex.zeros PiCcsDomain.blockDomain.laneVariables)
          |>.withLowPrefix block) =
      NumericBooleanDomain.index
          (BooleanVertex.zeros PiCcsDomain.blockDomain.laneVariables) +
        2 ^ PiCcsDomain.blockDomain.laneVariables *
          NumericBooleanDomain.index block :=
      NumericBooleanDomain.index_withLowPrefix _ _
    _ = PiCcsDomain.blockDomain.laneCount *
          NumericBooleanDomain.index block := by
      rw [NumericBooleanDomain.index_zeros]
      simp only [Nat.zero_add, PiCcsDomain.blockDomain_laneVariables,
        PiCcsDomain.blockDomain_laneCount]

/-- A live block row decodes to that block at Phi81's constant lane. -/
theorem decodeRow?_zeroLaneRow_of_live
    (block : BooleanVertex PiCcsDomain.blockDomain.blockVariables)
    (live : (BlockNcDomain.blockIndex block).val <
      Phi81ColumnLayout.blockCount shape.carrierWidth) :
    decodeRow? (zeroLaneRow block) =
      some
        (⟨(BlockNcDomain.blockIndex block).val, live⟩,
          Phi81CoefficientKernel.constant) := by
  apply (decodeRow?_eq_some_iff _ _ _).2
  rw [rowNumber_zeroLaneRow]
  simp only [paddedRowNumber, Phi81CoefficientKernel.constant, Nat.zero_add]

/-- A padded block row has no live carrier owner. -/
theorem decodeRow?_zeroLaneRow_of_padding
    (block : BooleanVertex PiCcsDomain.blockDomain.blockVariables)
    (padding : ¬(BlockNcDomain.blockIndex block).val <
      Phi81ColumnLayout.blockCount shape.carrierWidth) :
    decodeRow? (zeroLaneRow block) = none := by
  unfold decodeRow?
  rw [dif_neg]
  intro live
  apply padding
  have blockNumber :
      rowBlockNumber (zeroLaneRow block) =
        (BlockNcDomain.blockIndex block).val := by
    unfold rowBlockNumber
    rw [rowNumber_zeroLaneRow]
    have laneCountPositive : 0 < PiCcsDomain.blockDomain.laneCount := by
      rw [PiCcsDomain.blockDomain_laneCount]
      decide
    calc
      PiCcsDomain.blockDomain.laneCount *
            (BlockNcDomain.blockIndex block).val /
          PiCcsDomain.blockDomain.laneCount =
        (BlockNcDomain.blockIndex block).val *
            PiCcsDomain.blockDomain.laneCount /
          PiCcsDomain.blockDomain.laneCount := by rw [Nat.mul_comm]
      _ = (BlockNcDomain.blockIndex block).val :=
        Nat.mul_div_left _ laneCountPositive
  rw [← blockNumber]
  exact live

/-- On a live zero-lane row, the constant Phi81 kernel image is exactly the
authoritative assignment block. -/
theorem expectedRowRing_zeroLane_of_live
    (assignment : Phi81Relation.Assignment shape)
    (block : BooleanVertex PiCcsDomain.blockDomain.blockVariables)
    (live : (BlockNcDomain.blockIndex block).val <
      Phi81ColumnLayout.blockCount shape.carrierWidth) :
    expectedRowRing assignment (zeroLaneRow block) =
      CarrierAction.assignmentBlock assignment
        ⟨(BlockNcDomain.blockIndex block).val, live⟩ := by
  unfold expectedRowRing
  rw [decodeRow?_zeroLaneRow_of_live block live]
  exact RingFLaws.kernelImage_constant _

/-- On a padded zero-lane row, the auxiliary semantic value is zero. -/
theorem expectedRowRing_zeroLane_of_padding
    (assignment : Phi81Relation.Assignment shape)
    (block : BooleanVertex PiCcsDomain.blockDomain.blockVariables)
    (padding : ¬(BlockNcDomain.blockIndex block).val <
      Phi81ColumnLayout.blockCount shape.carrierWidth) :
    expectedRowRing assignment (zeroLaneRow block) = ringFZero := by
  unfold expectedRowRing
  rw [decodeRow?_zeroLaneRow_of_padding block padding]

private theorem embedChallenge_zero :
    RingKAction.embedChallenge ringFZero = ringKZero := by
  funext lane
  rfl

/-- Restricting the nine-row auxiliary semantics to six zero lane bits gives
exactly the canonical three-variable packed block table. -/
theorem embeddedExpected_zeroLane_eq_blockRows
    (assignment : Phi81Relation.Assignment shape) :
    (fun block : BooleanVertex PiCcsDomain.blockDomain.blockVariables =>
      RingKAction.embedChallenge
        (expectedRowRing assignment (zeroLaneRow block))) =
      PackedBlockAction.blockRows
        (domain := PiCcsDomain.blockDomain)
        (asPackedAssignment assignment) := by
  funext block
  by_cases live :
      (BlockNcDomain.blockIndex block).val <
        Phi81ColumnLayout.blockCount shape.carrierWidth
  · rw [expectedRowRing_zeroLane_of_live assignment block live]
    unfold PackedBlockAction.blockRows
    rw [dif_pos (by simpa only [semanticShape_carrierWidth] using live)]
    apply congrArg RingKAction.embedChallenge
    funext lane
    unfold CarrierAction.assignmentBlock asPackedAssignment
    apply congrArg assignment
    apply Fin.ext
    rfl
  · rw [expectedRowRing_zeroLane_of_padding assignment block live,
      embedChallenge_zero]
    unfold PackedBlockAction.blockRows
    rw [dif_neg (by simpa only [semanticShape_carrierWidth] using live)]

/-- Six verifier-fixed zero lane coordinates followed by the three block
coordinates form the auxiliary relation's nine-coordinate point. -/
def packedPoint
    (blockPoint : CubePoint K PiCcsDomain.blockDomain.blockVariables) :
    Phi81Relation.Point shape :=
  (BooleanVertex.zeros PiCcsDomain.blockDomain.laneVariables
      |>.toCubePoint ConcreteCarrier.extensionOps)
    |>.withLowPrefix blockPoint

/-- The separate padded-identity CE evaluation is exactly the canonical
packed block projection of the same complete authoritative assignment. -/
theorem matrixEvaluation_packedPoint_eq_packedYZcol
    (assignment : Phi81Relation.Assignment shape)
    (blockPoint : CubePoint K PiCcsDomain.blockDomain.blockVariables) :
    Phi81Relation.matrixEvaluation system assignment
        (packedPoint blockPoint) matrixIndex =
      PackedBlockAction.packedYZcol
        (PiCcsDomain.blockDomain_covers shape.rowVariables 0 0 1)
        (asPackedAssignment assignment) blockPoint := by
  rw [matrixEvaluation_eq_expectedRows]
  unfold PackedBlockAction.packedYZcol
  funext lane
  unfold RingKAction.evaluateRows
  calc
    (BooleanTable.tabulate (fun row =>
        RingKAction.embedChallenge (expectedRowRing assignment row) lane)
      |>.evaluate ConcreteCarrier.extensionOps
        ((BooleanVertex.zeros PiCcsDomain.blockDomain.laneVariables
            |>.toCubePoint ConcreteCarrier.extensionOps)
          |>.withLowPrefix blockPoint)) =
      (BooleanTable.tabulate (fun block =>
          RingKAction.embedChallenge
            (expectedRowRing assignment (zeroLaneRow block)) lane)
        |>.evaluate ConcreteCarrier.extensionOps blockPoint) :=
      BooleanTable.evaluate_tabulate_booleanPrefix
        ConcreteCarrier.extensionOps ConcreteCarrier.extensionLaws
        (BooleanVertex.zeros PiCcsDomain.blockDomain.laneVariables)
        (fun row =>
          RingKAction.embedChallenge (expectedRowRing assignment row) lane)
        blockPoint
    _ = _ := by
      apply congrArg (fun table =>
        BooleanTable.evaluate ConcreteCarrier.extensionOps table blockPoint)
      apply congrArg BooleanTable.tabulate
      funext block
      exact congrFun
        (congrFun (embeddedExpected_zeroLane_eq_blockRows assignment) block)
        lane

/-! ## Semantic CE witness handoff -/

universe uCommitment

/-- The all-matrix, all-lane CE evaluation predicate binds the sole claimed
evaluation to the packed projection relative to the supplied witness
assignment. This is evaluation equality, not commitment-binding authority. -/
theorem claimedEvaluation_eq_packedYZcol_of_evaluationsBound
    {Commitment : Type uCommitment}
    (statement : Phi81Relation.CEStatement shape Commitment)
    (assignment : Phi81Relation.Assignment shape)
    (blockPoint : CubePoint K PiCcsDomain.blockDomain.blockVariables)
    (systemEq : statement.constraintSystem = system)
    (pointEq : statement.point = packedPoint blockPoint)
    (bound : Phi81Relation.EvaluationsBound statement.constraintSystem
      assignment statement.point statement.evaluations) :
    statement.evaluations[matrixIndex.val]'(by
      rw [bound.size_eq]
      exact matrixIndex.isLt) =
      PackedBlockAction.packedYZcol
        (PiCcsDomain.blockDomain_covers shape.rowVariables 0 0 1)
        (asPackedAssignment assignment) blockPoint := by
  funext lane
  have laneEq := bound.lane_eq matrixIndex lane
  calc
    (statement.evaluations[matrixIndex.val]'(by
        rw [bound.size_eq]
        exact matrixIndex.isLt)) lane =
        Phi81Relation.matrixEvaluation statement.constraintSystem assignment
          statement.point matrixIndex lane := laneEq
    _ =
        Phi81Relation.matrixEvaluation system assignment
          (packedPoint blockPoint) matrixIndex lane := by
      rw [systemEq, pointEq]
    _ = PackedBlockAction.packedYZcol
          (PiCcsDomain.blockDomain_covers shape.rowVariables 0 0 1)
          (asPackedAssignment assignment) blockPoint lane :=
      congrFun
        (matrixEvaluation_packedPoint_eq_packedYZcol assignment blockPoint)
        lane

/-- Semantic auxiliary CE witness membership supplies the all-lane evaluation
predicate above. Because `commit` remains arbitrary, this corollary proves no
unique opening, commitment binding, verifier-acceptance refinement, or active
recursive-relation integration. -/
theorem claimedEvaluation_eq_packedYZcol_of_ceHolds
    {Commitment : Type uCommitment}
    (commit : Phi81Relation.Assignment shape -> Commitment)
    (params : GlobalParams)
    (statement : Phi81Relation.CEStatement shape Commitment)
    (assignment : Phi81Relation.Assignment shape)
    (blockPoint : CubePoint K PiCcsDomain.blockDomain.blockVariables)
    (systemEq : statement.constraintSystem = system)
    (pointEq : statement.point = packedPoint blockPoint)
    (holds : CE.Holds (Phi81Relation.relationSemantics commit) params
      statement assignment) :
    statement.evaluations[matrixIndex.val]'(by
      rw [Phi81Relation.ce_evaluations_size_of_holds
        commit params statement assignment holds]
      exact matrixIndex.isLt) =
      PackedBlockAction.packedYZcol
        (PiCcsDomain.blockDomain_covers shape.rowVariables 0 0 1)
        (asPackedAssignment assignment) blockPoint := by
  have bound :=
    ((Phi81Relation.ceMembership_iff_evaluationsBound
      commit params statement assignment).1 holds).2.2.2
  exact claimedEvaluation_eq_packedYZcol_of_evaluationsBound
    statement assignment blockPoint systemEq pointEq bound

end Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270.PaddedIdentityEvaluation
