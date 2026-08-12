import Nightstream.Implementation.NebulaV2.NIFS.PiDEC.LinearCombination
import Nightstream.Implementation.R1CS.Canonical.KBooleanMleSemantics
import Nightstream.Implementation.R1CS.Canonical.KConcreteBridge
import Nightstream.SuperNeo.Concrete.Phi81Relation.Evaluation

/-!
Contract: uniform dynamic-point rows for one complete Phi81 CE evaluation.

The evaluation point and complete assignment are read from circuit columns.
The relation structure is verifier-owned. For every matrix and every one of
the 54 Phi81 lanes, the emitted Boolean-MLE rows compute the exact value of
`Phi81Relation.matrixEvaluation` and bind it to two claimed field columns.

The compiler never accepts an evaluation equation or a validity bit. Its
semantic bridge is proved against the independent SuperNeo definition in
`Phi81Relation.Evaluation`.

This module does not own physical placement in a generated terminal artifact,
the public-input projection, commitment opening, norm rows, Rust refinement,
or a compact proof backend.

This is a reference compiler, not the production terminal compiler. It builds
one full Boolean-MLE tree for each matrix lane. Its exact row count is
exponential in `shape.rowVariables` and repeats that cost for every matrix
lane. A production profile must use a shared tensor and sparse row support,
and must prove that implementation against the same Phi81 definition.

Assurance tier: model-level dynamic terminal-evaluation soundness.
-/

set_option autoImplicit false
set_option maxRecDepth 100000

namespace Nightstream.Implementation.R1CS.Phi81DynamicEvaluationRows

open Nightstream.Implementation.NebulaV2.ProductPiDecLinearCombination
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical
open Nightstream.Implementation.R1CS.Canonical.KBooleanMle
open Nightstream.Implementation.R1CS.Canonical.KBooleanMleSemantics
open Nightstream.Implementation.R1CS.Canonical.KConcreteBridge
open Nightstream.Implementation.R1CS.Canonical.KHorner
open Nightstream.Implementation.R1CS.Canonical.KMul
open Nightstream.Implementation.R1CS.Canonical.LinCombNormal
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.PaperLinearAlgebra

/-- Exact circuit columns for one uniform Phi81 terminal evaluator. -/
structure Frame (shape : Phi81Relation.Shape) where
  witness : Fin shape.carrierWidth -> Nat
  pointLow : Fin shape.rowVariables -> Nat
  pointHigh : Fin shape.rowVariables -> Nat
  claimLow : Fin shape.matrixCount -> Fin ringDegree -> Nat
  claimHigh : Fin shape.matrixCount -> Fin ringDegree -> Nat
  firstAuxiliary : Nat

/-- One circuit-carried extension value read directly from two columns. -/
def inputCarried (low high : Nat) : Carried :=
  ⟨[(low, 1)], [(high, 1)]⟩

/-- Dynamic point coordinates in the same low/high order as the typed point. -/
def pointCoordinates {shape : Phi81Relation.Shape}
    (frame : Frame shape) : List Carried :=
  List.ofFn fun coordinate =>
    inputCarried (frame.pointLow coordinate) (frame.pointHigh coordinate)

/-- The symbolic matrix-vector value at one Boolean row. Coefficients come
only from the verifier-owned Phi81 matrix source. -/
def rowCarried {shape : Phi81Relation.Shape}
    (frame : Frame shape) (system : Phi81Relation.Structure shape)
    (matrix : Fin shape.matrixCount) (lane : Fin ringDegree)
    (vertex : BooleanVertex shape.rowVariables) : Carried where
  low :=
    (List.ofFn frame.witness).zip
      (List.ofFn fun column =>
        (system.matrixSource.coefficientMatrix
          ConcreteCarrier.baseOps matrix lane vertex column).val)
  high := []

/-- Exact symbolic Boolean table for one matrix and one Phi81 lane. -/
def table {shape : Phi81Relation.Shape}
    (frame : Frame shape) (system : Phi81Relation.Structure shape)
    (matrix : Fin shape.matrixCount) (lane : Fin ringDegree) :
    BooleanTable Carried shape.rowVariables :=
  BooleanTable.tabulate (rowCarried frame system matrix lane)

/-- Canonical flattened matrix/lane position. -/
def evaluationIndex {shape : Phi81Relation.Shape}
    (matrix : Fin shape.matrixCount) (lane : Fin ringDegree) : Nat :=
  matrix.val * ringDegree + lane.val

/-- Each matrix/lane MLE owns a disjoint fixed-size auxiliary block. -/
def auxiliaryBase {shape : Phi81Relation.Shape}
    (frame : Frame shape) (matrix : Fin shape.matrixCount)
    (lane : Fin ringDegree) : Nat :=
  frame.firstAuxiliary +
    3 * KBooleanMle.frameCount shape.rowVariables *
      evaluationIndex matrix lane

/-- Symbolic output of one complete dynamic-point MLE. -/
def computed {shape : Phi81Relation.Shape}
    (frame : Frame shape) (system : Phi81Relation.Structure shape)
    (matrix : Fin shape.matrixCount) (lane : Fin ringDegree) : Carried :=
  KBooleanMle.carried
    (KFrames.frameAt (auxiliaryBase frame matrix lane))
    (table frame system matrix lane) (pointCoordinates frame) 0

/-- Bind a symbolic base-field value to one claimed column. -/
def outputRow (value : LinComb) (claim : Nat) : Row where
  a := value
  b := [(0, 1)]
  c := [(claim, 1)]

/-- Rows for one matrix/lane evaluation and both claimed coordinates. -/
def rowsFor {shape : Phi81Relation.Shape}
    (frame : Frame shape) (system : Phi81Relation.Structure shape)
    (matrix : Fin shape.matrixCount) (lane : Fin ringDegree) : List Row :=
  KBooleanMle.rows
      (KFrames.frameAt (auxiliaryBase frame matrix lane))
      (table frame system matrix lane) (pointCoordinates frame) 0 ++
    [outputRow (computed frame system matrix lane).low
        (frame.claimLow matrix lane),
     outputRow (computed frame system matrix lane).high
        (frame.claimHigh matrix lane)]

/-- Complete dynamic terminal-evaluation row family in matrix-major order. -/
def rows {shape : Phi81Relation.Shape}
    (frame : Frame shape) (system : Phi81Relation.Structure shape) : List Row :=
  (canonicalFinIndices shape.matrixCount).flatMap fun matrix =>
    (canonicalFinIndices ringDegree).flatMap fun lane =>
      rowsFor frame system matrix lane

@[simp] theorem rowsFor_length {shape : Phi81Relation.Shape}
    (frame : Frame shape) (system : Phi81Relation.Structure shape)
    (matrix : Fin shape.matrixCount) (lane : Fin ringDegree) :
    (rowsFor frame system matrix lane).length =
      3 * KBooleanMle.frameCount shape.rowVariables + 2 := by
  simp [rowsFor, KBooleanMle.rows_length]

/-- Exact cost of the reference compiler. This theorem prevents a generated
profile from treating this row family as a compact production evaluator. -/
@[simp] theorem rows_length {shape : Phi81Relation.Shape}
    (frame : Frame shape) (system : Phi81Relation.Structure shape) :
    (rows frame system).length =
      shape.matrixCount * ringDegree *
        (3 * KBooleanMle.frameCount shape.rowVariables + 2) := by
  simp [rows, canonicalFinIndices_length, rowsFor_length, Nat.mul_assoc]

/-- Decode the complete assignment directly from its authoritative columns. -/
def decodedAssignment {shape : Phi81Relation.Shape}
    (frame : Frame shape) (assignment : Nat -> Nat)
    (canonical : forall column, assignment column < goldilocksP) :
    Phi81Relation.Assignment shape :=
  fun column => fieldAt assignment canonical (frame.witness column)

/-- Decode the dynamic evaluation point directly from its columns. -/
def decodedPoint {shape : Phi81Relation.Shape}
    (frame : Frame shape) (assignment : Nat -> Nat)
    (canonical : forall column, assignment column < goldilocksP) :
    Phi81Relation.Point shape where
  coordinates :=
    (canonicalFinIndices shape.rowVariables).map fun coordinate =>
      ⟨fieldAt assignment canonical (frame.pointLow coordinate),
       fieldAt assignment canonical (frame.pointHigh coordinate)⟩
  dimension := by simp [canonicalFinIndices_length]

/-- Decode all claimed evaluations in exact matrix and lane order. -/
def decodedEvaluations {shape : Phi81Relation.Shape}
    (frame : Frame shape) (assignment : Nat -> Nat)
    (canonical : forall column, assignment column < goldilocksP) :
    Array Phi81Relation.Evaluation :=
  Array.ofFn fun matrix lane =>
    ⟨fieldAt assignment canonical (frame.claimLow matrix lane),
     fieldAt assignment canonical (frame.claimHigh matrix lane)⟩

/-! ## Independent finite-sum bridge -/

private def listSum {Index : Type}
    (indices : List Index) (term : Index -> F) : F :=
  match indices with
  | [] => 0
  | index :: rest => term index + listSum rest term

private theorem foldl_eq_add_listSum
    {Index : Type} (indices : List Index) (term : Index -> F)
    (initial : F) :
    indices.foldl (fun accumulated index => accumulated + term index) initial =
      initial + listSum indices term := by
  induction indices generalizing initial with
  | nil => exact (ConcreteCarrier.baseLaws.add_zero initial).symm
  | cons index indices inductionHypothesis =>
      rw [List.foldl_cons, inductionHypothesis]
      exact ConcreteCarrier.baseLaws.add_assoc _ _ _

private theorem listSum_map
    {Left Right : Type} (indices : List Left) (map : Left -> Right)
    (term : Right -> F) :
    listSum (indices.map map) term =
      listSum indices (fun index => term (map index)) := by
  induction indices with
  | nil => rfl
  | cons index indices inductionHypothesis =>
      simp only [List.map_cons, listSum, inductionHypothesis]

private theorem listSum_canonical_eq_combineFields :
    forall {count : Nat} (weights values : Fin count -> F),
      listSum (canonicalFinIndices count)
          (fun index => weights index * values index) =
        combineFields weights values
  | 0, _, _ => rfl
  | count + 1, weights, values => by
      rw [canonicalFinIndices, List.ofFn_succ]
      simp only [listSum, combineFields]
      congr 1
      rw [show (List.ofFn fun index : Fin count => id index.succ) =
          (canonicalFinIndices count).map Fin.succ by
        simp [canonicalFinIndices, List.map_ofFn]]
      rw [listSum_map]
      exact listSum_canonical_eq_combineFields
        (fun index => weights index.succ)
        (fun index => values index.succ)

/-- The numeric compiler fold is exactly the independent typed matrix-vector
definition. -/
theorem combineFields_eq_matrixVectorAt
    {variableCount columns : Nat}
    (matrix : BooleanMatrix F variableCount columns)
    (assignment : Assignment F columns)
    (vertex : BooleanVertex variableCount) :
    combineFields (fun column => matrix vertex column) assignment =
      matrixVectorAt ConcreteCarrier.baseOps matrix assignment vertex := by
  unfold matrixVectorAt
  change combineFields (fun column => matrix vertex column) assignment =
    (canonicalFinIndices columns).foldl
      (fun accumulated column =>
        accumulated + matrix vertex column * assignment column) 0
  rw [foldl_eq_add_listSum]
  rw [Fin.zero_add]
  exact (listSum_canonical_eq_combineFields
    (fun column => matrix vertex column) assignment).symm

/-! ## Decoding bridge to the independent Phi81 evaluator -/

/-- One symbolic matrix row decodes to the exact typed matrix-vector value. -/
theorem decodeCarried_rowCarried
    {shape : Phi81Relation.Shape}
    (frame : Frame shape) (system : Phi81Relation.Structure shape)
    (matrix : Fin shape.matrixCount) (lane : Fin ringDegree)
    (vertex : BooleanVertex shape.rowVariables)
    (assignment : Nat -> Nat)
    (canonical : forall column, assignment column < goldilocksP) :
    decodeCarried assignment (rowCarried frame system matrix lane vertex) =
      K.embed
        (matrixVectorAt ConcreteCarrier.baseOps
          (system.matrixSource.coefficientMatrix
            ConcreteCarrier.baseOps matrix lane)
          (decodedAssignment frame assignment canonical) vertex) := by
  change K.mk _ _ = K.mk _ _
  congr 1
  · apply Fin.ext
    change lcEval assignment _ = _
    have linear := lcEval_ofFn_zip assignment canonical frame.witness
      (fun column =>
        system.matrixSource.coefficientMatrix
          ConcreteCarrier.baseOps matrix lane vertex column)
    rw [combineFields_eq_matrixVectorAt] at linear
    exact congrArg Fin.val linear

private theorem decodeTable_tabulate
    (assignment : Nat -> Nat) :
    forall {variableCount : Nat}
      (values : BooleanVertex variableCount -> Carried),
      decodeTable assignment (BooleanTable.tabulate values) =
        BooleanTable.tabulate (fun vertex => decodeCarried assignment (values vertex))
  | 0, _ => rfl
  | variableCount + 1, values => by
      simp only [BooleanTable.tabulate, decodeTable]
      rw [decodeTable_tabulate assignment
        (fun tail => values (.cons false tail))]
      rw [decodeTable_tabulate assignment
        (fun tail => values (.cons true tail))]

/-- The compiler table is definitionally sourced from, and extensionally
equal to, SuperNeo's authoritative Phi81 evaluation table. -/
theorem decodeTable_eq_phi81Table
    {shape : Phi81Relation.Shape}
    (frame : Frame shape) (system : Phi81Relation.Structure shape)
    (matrix : Fin shape.matrixCount) (lane : Fin ringDegree)
    (assignment : Nat -> Nat)
    (canonical : forall column, assignment column < goldilocksP) :
    decodeTable assignment (table frame system matrix lane) =
      Phi81Evaluation.table system.matrixSource
        (decodedAssignment frame assignment canonical) matrix lane := by
  rw [table, decodeTable_tabulate]
  unfold Phi81Evaluation.table
  apply congrArg BooleanTable.tabulate
  funext vertex
  exact decodeCarried_rowCarried frame system matrix lane vertex assignment
    canonical

/-- The compiler point columns decode in exact typed coordinate order. -/
@[simp] theorem pointCoordinates_length
    {shape : Phi81Relation.Shape} (frame : Frame shape) :
    (pointCoordinates frame).length = shape.rowVariables := by
  simp [pointCoordinates]

/-- Semantic point decoded by the generic Boolean-MLE bridge. The result type
fixes the dimension before the proof argument is elaborated. -/
def mleDecodedPoint {shape : Phi81Relation.Shape}
    (frame : Frame shape) (assignment : Nat -> Nat) :
    CubePoint K shape.rowVariables :=
  KBooleanMleSemantics.decodePoint assignment (pointCoordinates frame)
    (pointCoordinates_length frame)

private theorem cubePoint_eq_of_coordinates
    {Field : Type} {variableCount : Nat}
    (left right : CubePoint Field variableCount)
    (equal : left.coordinates = right.coordinates) :
    left = right := by
  cases left with
  | mk leftCoordinates leftDimension =>
      cases right with
      | mk rightCoordinates rightDimension =>
          simp only at equal
          subst rightCoordinates
          rfl

theorem decodePoint_eq_decodedPoint
    {shape : Phi81Relation.Shape}
    (frame : Frame shape) (assignment : Nat -> Nat)
    (canonical : forall column, assignment column < goldilocksP) :
    mleDecodedPoint frame assignment =
      decodedPoint frame assignment canonical := by
  have coordinatesEqual :
      (mleDecodedPoint frame assignment).coordinates =
      (decodedPoint frame assignment canonical).coordinates := by
    simp only [mleDecodedPoint, KBooleanMleSemantics.decodePoint, decodedPoint,
      pointCoordinates, List.map_ofFn, canonicalFinIndices]
    apply List.ext_get
    · simp
    · intro index leftBound rightBound
      simp [inputCarried, decodeCarried, fieldAt, lcEval,
        Nat.mod_eq_of_lt (canonical _)]
  exact cubePoint_eq_of_coordinates _ _ coordinatesEqual

/-! ## Row soundness -/

private theorem satisfies_rowsFor_of_rows
    {shape : Phi81Relation.Shape}
    (frame : Frame shape) (system : Phi81Relation.Structure shape)
    (assignment : Nat -> Nat)
    (satisfied : Satisfies (rows frame system) assignment)
    (matrix : Fin shape.matrixCount) (lane : Fin ringDegree) :
    Satisfies (rowsFor frame system matrix lane) assignment := by
  intro row member
  apply satisfied row
  unfold rows
  rw [List.mem_flatMap]
  refine ⟨matrix, ?_, ?_⟩
  · simp [canonicalFinIndices]
  · rw [List.mem_flatMap]
    exact ⟨lane, by simp [canonicalFinIndices], member⟩

private theorem outputRow_sound
    (assignment : Nat -> Nat) (constantOne : assignment 0 = 1)
    (value : LinComb) (claim : Nat)
    (holds : RowHolds assignment (outputRow value claim)) :
    lcEval assignment value = assignment claim % goldilocksP := by
  simpa [outputRow, RowHolds, lcEval, constantOne] using holds

/-- Satisfying dynamic-point rows bind every claimed output to the exact
independent Phi81 matrix evaluation. -/
theorem rows_sound
    {shape : Phi81Relation.Shape}
    (frame : Frame shape) (system : Phi81Relation.Structure shape)
    (assignment : Nat -> Nat)
    (canonical : forall column, assignment column < goldilocksP)
    (constantOne : assignment 0 = 1)
    (satisfied : Satisfies (rows frame system) assignment) :
    decodedEvaluations frame assignment canonical =
      Phi81Relation.evaluations system
        (decodedAssignment frame assignment canonical)
        (decodedPoint frame assignment canonical) := by
  apply Array.ext
  · simp [decodedEvaluations, Phi81Relation.evaluations]
  · intro matrix matrixBound _
    have matrixLt : matrix < shape.matrixCount := by
      simpa [decodedEvaluations] using matrixBound
    let matrixIndex : Fin shape.matrixCount := ⟨matrix, matrixLt⟩
    apply funext
    intro lane
    simp only [decodedEvaluations, Phi81Relation.evaluations,
      Array.getElem_ofFn]
    have localSatisfied :=
      satisfies_rowsFor_of_rows frame system assignment satisfied
        matrixIndex lane
    have mleSatisfied : Satisfies
        (KBooleanMle.rows
          (KFrames.frameAt (auxiliaryBase frame matrixIndex lane))
          (table frame system matrixIndex lane)
          (pointCoordinates frame) 0) assignment := by
      intro row member
      exact localSatisfied row (List.mem_append_left _ member)
    have computedExact := KBooleanMleSemantics.rows_compute_evaluate assignment
      (auxiliaryBase frame matrixIndex lane)
      (table frame system matrixIndex lane)
      (pointCoordinates frame)
      (pointCoordinates_length frame) mleSatisfied
    have lowRow : RowHolds assignment
        (outputRow (computed frame system matrixIndex lane).low
          (frame.claimLow matrixIndex lane)) := by
      exact localSatisfied _ (by simp [rowsFor])
    have highRow : RowHolds assignment
        (outputRow (computed frame system matrixIndex lane).high
          (frame.claimHigh matrixIndex lane)) := by
      exact localSatisfied _ (by simp [rowsFor])
    have lowExact := outputRow_sound assignment constantOne _ _ lowRow
    have highExact := outputRow_sound assignment constantOne _ _ highRow
    have tableExact := decodeTable_eq_phi81Table frame system
      matrixIndex lane assignment canonical
    have pointExact := decodePoint_eq_decodedPoint frame assignment canonical
    rw [tableExact] at computedExact
    have semanticExact :
        carriedValue assignment
            (computed frame system matrixIndex lane) =
          KConcreteBridge.ofConcrete
            (Phi81Relation.matrixEvaluation system
              (decodedAssignment frame assignment canonical)
              (decodedPoint frame assignment canonical)
              matrixIndex lane) := by
      have dynamicExact :
          carriedValue assignment (computed frame system matrixIndex lane) =
            KConcreteBridge.ofConcrete
              (Phi81Evaluation.evaluate system.matrixSource
                (decodedAssignment frame assignment canonical)
                (mleDecodedPoint frame assignment) matrixIndex lane) := by
        simpa [computed, mleDecodedPoint] using computedExact
      rw [pointExact] at dynamicExact
      exact dynamicExact
    apply KConcreteBridge.ofConcrete_injective
    change Pair.mk _ _ = Pair.mk _ _
    simp only [Pair.mk.injEq]
    constructor
    · change (fieldAt assignment canonical
          (frame.claimLow matrixIndex lane)).val = _
      rw [fieldAt]
      change assignment _ = _
      rw [← Nat.mod_eq_of_lt (canonical _)]
      rw [← lowExact]
      simpa [matrixIndex] using congrArg Pair.low semanticExact
    · change (fieldAt assignment canonical
          (frame.claimHigh matrixIndex lane)).val = _
      rw [fieldAt]
      change assignment _ = _
      rw [← Nat.mod_eq_of_lt (canonical _)]
      rw [← highExact]
      simpa [matrixIndex] using congrArg Pair.high semanticExact

end Nightstream.Implementation.R1CS.Phi81DynamicEvaluationRows
