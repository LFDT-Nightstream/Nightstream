import Nightstream.Implementation.NebulaV2.ProductPiDecLinearCombination
import Nightstream.Implementation.R1CS.Correspondence.Gadgets.Phi81SharedTensorRows
import Nightstream.SuperNeo.Concrete.Phi81Relation.Evaluation

/-!
Contract: sparse, shared-tensor rows for complete dynamic Phi81 terminal
evaluation.

One tensor is shared by every matrix and every Phi81 lane.  For each
verifier-derived nonzero matrix row, two base-field product rows multiply the
row image by the low and high tensor limbs.  Two final linear rows bind each
claimed ring lane.  Omitted rows are proved zero from the verifier-owned
coefficient matrix; a caller cannot supply a support list.

The main theorem derives the complete claimed evaluation array from the same
full assignment and dynamic point used by the independent SuperNeo Phi81
definition.  It does not accept an evaluation equation, a support-completeness
claim, or a validity bit.

This module does not own public-input projection, commitment opening, strict
norm rows, physical artifact placement, frame disjointness, Rust refinement,
or a compact proof backend.

Assurance tier: model-level sparse terminal-evaluation soundness.
-/

set_option autoImplicit false
set_option maxRecDepth 100000

namespace Nightstream.Implementation.R1CS.Phi81SharedEvaluationRows

open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.TerminalR1cs
open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.Implementation.Lowering.Typed
open Nightstream.Implementation.NebulaV2.ProductPiDecLinearCombination
open Nightstream.Implementation.R1CS.Phi81SharedTensorRows
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.PaperLinearAlgebra

/-- Two output columns for multiplying one base-field row image by one
quadratic-extension equality weight. -/
structure ProductFrame where
  low : ColumnId
  high : ColumnId

/-- Exact physical inputs and outputs for one complete Phi81 evaluator. -/
structure Frame (shape : Phi81Relation.Shape) where
  tensor : Phi81SharedTensorRows.Frame shape.rowVariables
  witness : Fin shape.carrierWidth -> ColumnId
  claimLow : Fin shape.matrixCount -> Fin ringDegree -> ColumnId
  claimHigh : Fin shape.matrixCount -> Fin ringDegree -> ColumnId
  productOwner : PhysicalOwner
  productFirstOrdinal : Fin shape.matrixCount -> Fin ringDegree ->
    BooleanVertex shape.rowVariables -> Nat
  productFrame : Fin shape.matrixCount -> Fin ringDegree ->
    BooleanVertex shape.rowVariables -> ProductFrame
  outputOwner : PhysicalOwner
  outputFirstOrdinal : Fin shape.matrixCount -> Fin ringDegree -> Nat

/-- Complete assignment decoded from the one authoritative witness column
family. -/
def decodedAssignment {shape : Phi81Relation.Shape}
    (frame : Frame shape) (assignment : ColumnId -> F) :
    Phi81Relation.Assignment shape :=
  fun column => assignment (frame.witness column)

/-- Claimed evaluation array decoded from its exact matrix/lane columns. -/
def decodedEvaluations {shape : Phi81Relation.Shape}
    (frame : Frame shape) (assignment : ColumnId -> F) :
    Array Phi81Relation.Evaluation :=
  Array.ofFn fun matrix lane =>
    ⟨assignment (frame.claimLow matrix lane),
     assignment (frame.claimHigh matrix lane)⟩

/-- Sparse linear combination for one exact coefficient-matrix row. -/
def rowCombination {shape : Phi81Relation.Shape}
    (frame : Frame shape) (system : Phi81Relation.Structure shape)
    (matrix : Fin shape.matrixCount) (lane : Fin ringDegree)
    (vertex : BooleanVertex shape.rowVariables) : LinearCombination :=
  List.ofFn fun column : Fin shape.carrierWidth =>
    ⟨frame.witness column,
     system.matrixSource.coefficientMatrix ConcreteCarrier.baseOps
       matrix lane vertex column⟩

/-! ## Exact row-image bridge -/

private theorem eval_ofFn
    {count : Nat} (assignment : ColumnId -> F)
    (columns : Fin count -> ColumnId) (weights : Fin count -> F) :
    LinearCombination.eval assignment (List.ofFn fun index : Fin count =>
      (⟨columns index, weights index⟩ : Term)) =
      combineFields weights (fun index => assignment (columns index)) := by
  induction count with
  | zero => rfl
  | succ count inductionHypothesis =>
      rw [List.ofFn_succ]
      simp only [LinearCombination.eval_cons, combineFields]
      rw [inductionHypothesis
        (fun index => columns index.succ)
        (fun index => weights index.succ)]

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

private theorem combineFields_eq_matrixVectorAt
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

/-- The sparse circuit row image is exactly the independent typed
matrix-vector value. -/
theorem rowCombination_eval
    {shape : Phi81Relation.Shape}
    (frame : Frame shape) (system : Phi81Relation.Structure shape)
    (assignment : ColumnId -> F)
    (matrix : Fin shape.matrixCount) (lane : Fin ringDegree)
    (vertex : BooleanVertex shape.rowVariables) :
    (rowCombination frame system matrix lane vertex).eval assignment =
      matrixVectorAt ConcreteCarrier.baseOps
        (system.matrixSource.coefficientMatrix ConcreteCarrier.baseOps
          matrix lane)
        (decodedAssignment frame assignment) vertex := by
  rw [rowCombination, eval_ofFn]
  exact combineFields_eq_matrixVectorAt _ _ _

/-! ## Verifier-derived sparse support -/

/-- A row is live exactly when its verifier-owned coefficient matrix has a
nonzero coefficient. -/
noncomputable def rowActive {shape : Phi81Relation.Shape}
    (system : Phi81Relation.Structure shape)
    (matrix : Fin shape.matrixCount) (lane : Fin ringDegree)
    (vertex : BooleanVertex shape.rowVariables) : Bool := by
  classical
  exact decide (exists column : Fin shape.carrierWidth,
    system.matrixSource.coefficientMatrix ConcreteCarrier.baseOps
      matrix lane vertex column ≠ 0)

/-- Canonical live-row order.  No prover-supplied support list exists. -/
noncomputable def activeVertices {shape : Phi81Relation.Shape}
    (system : Phi81Relation.Structure shape)
    (matrix : Fin shape.matrixCount) (lane : Fin ringDegree) :
    List (BooleanVertex shape.rowVariables) :=
  (BooleanVertex.all shape.rowVariables).filter
    (rowActive system matrix lane)

theorem rowActive_false_coefficients_zero
    {shape : Phi81Relation.Shape}
    (system : Phi81Relation.Structure shape)
    (matrix : Fin shape.matrixCount) (lane : Fin ringDegree)
    (vertex : BooleanVertex shape.rowVariables)
    (inactive : rowActive system matrix lane vertex = false) :
    forall column,
      system.matrixSource.coefficientMatrix ConcreteCarrier.baseOps
        matrix lane vertex column = 0 := by
  classical
  intro column
  by_contra nonzero
  have existsNonzero : exists selected : Fin shape.carrierWidth,
      system.matrixSource.coefficientMatrix ConcreteCarrier.baseOps
        matrix lane vertex selected ≠ 0 :=
    ⟨column, nonzero⟩
  have active : rowActive system matrix lane vertex = true := by
    simp [rowActive, existsNonzero]
  rw [active] at inactive
  cases inactive

private theorem combineFields_zero_weights :
    forall {count : Nat} (weights values : Fin count -> F),
      (forall index, weights index = 0) ->
      combineFields weights values = 0
  | 0, _, _, _ => rfl
  | count + 1, weights, values, zero => by
      have tailZero := combineFields_zero_weights
        (fun index => weights index.succ)
        (fun index => values index.succ)
        (fun index => zero index.succ)
      simp only [combineFields]
      rw [zero, tailZero]
      simp only [Fin.zero_mul, Fin.zero_add]

theorem inactive_matrixVector_zero
    {shape : Phi81Relation.Shape}
    (frame : Frame shape) (system : Phi81Relation.Structure shape)
    (assignment : ColumnId -> F)
    (matrix : Fin shape.matrixCount) (lane : Fin ringDegree)
    (vertex : BooleanVertex shape.rowVariables)
    (inactive : rowActive system matrix lane vertex = false) :
    matrixVectorAt ConcreteCarrier.baseOps
        (system.matrixSource.coefficientMatrix ConcreteCarrier.baseOps
          matrix lane)
        (decodedAssignment frame assignment) vertex = 0 := by
  rw [← combineFields_eq_matrixVectorAt]
  apply combineFields_zero_weights
  exact rowActive_false_coefficients_zero system matrix lane vertex inactive

/-! ## Two-row base-by-extension products -/

def productValue (product : ProductFrame) : Extension.Value where
  low := [⟨product.low, 1⟩]
  high := [⟨product.high, 1⟩]

def productRows
    (owner : PhysicalOwner) (firstOrdinal : Nat)
    (base : LinearCombination) (weight : Extension.Value)
    (product : ProductFrame) : List OwnedRow :=
  [ ⟨⟨owner, firstOrdinal⟩,
      ⟨base, weight.low, [⟨product.low, 1⟩]⟩⟩,
    ⟨⟨owner, firstOrdinal + 1⟩,
      ⟨base, weight.high, [⟨product.high, 1⟩]⟩⟩ ]

@[simp] theorem productRows_length
    (owner : PhysicalOwner) (firstOrdinal : Nat)
    (base : LinearCombination) (weight : Extension.Value)
    (product : ProductFrame) :
    (productRows owner firstOrdinal base weight product).length = 2 :=
  rfl

theorem productRows_sound
    (owner : PhysicalOwner) (firstOrdinal : Nat)
    (base : LinearCombination) (weight : Extension.Value)
    (product : ProductFrame) (assignment : ColumnId -> F)
    (satisfied :
      Nightstream.Implementation.Lowering.Goldilocks.Satisfies
        (productRows owner firstOrdinal base weight product) assignment) :
    Phi81SharedTensorRows.value assignment (productValue product) =
      K.mul (K.embed (base.eval assignment))
        (Phi81SharedTensorRows.value assignment weight) := by
  change
    (base.eval assignment * weight.low.eval assignment =
        LinearCombination.eval assignment
          ([(⟨product.low, 1⟩ : Term)] : LinearCombination)) /\
      (base.eval assignment * weight.high.eval assignment =
        LinearCombination.eval assignment
          ([(⟨product.high, 1⟩ : Term)] : LinearCombination)) /\
      True at satisfied
  rcases satisfied with ⟨low, high, _⟩
  simp only [LinearCombination.eval_cons, LinearCombination.eval_nil,
    Fin.one_mul, Fin.add_zero] at low high
  simp only [Phi81SharedTensorRows.value, productValue, K.embed, K.mul,
    K.mk.injEq, LinearCombination.eval_cons, LinearCombination.eval_nil,
    Fin.one_mul, Fin.add_zero, Fin.mul_zero, Fin.zero_mul]
  exact ⟨low.symm, high.symm⟩

def localProductRows {shape : Phi81Relation.Shape}
    (frame : Frame shape) (system : Phi81Relation.Structure shape)
    (matrix : Fin shape.matrixCount) (lane : Fin ringDegree)
    (vertex : BooleanVertex shape.rowVariables) : List OwnedRow :=
  productRows frame.productOwner
    (frame.productFirstOrdinal matrix lane vertex)
    (rowCombination frame system matrix lane vertex)
    (Phi81SharedTensorRows.chi frame.tensor vertex)
    (frame.productFrame matrix lane vertex)

def ProductsSatisfied {shape : Phi81Relation.Shape}
    (frame : Frame shape) (system : Phi81Relation.Structure shape)
    (assignment : ColumnId -> F) : Prop :=
  forall matrix lane vertex,
    rowActive system matrix lane vertex = true ->
    Nightstream.Implementation.Lowering.Goldilocks.Satisfies
      (localProductRows frame system matrix lane vertex) assignment

theorem localProduct_sound
    {shape : Phi81Relation.Shape}
    (frame : Frame shape) (system : Phi81Relation.Structure shape)
    (assignment : ColumnId -> F)
    (tensorOne : assignment frame.tensor.one = 1)
    (tensorRows : Phi81SharedTensorRows.RowsSatisfied frame.tensor assignment)
    (products : ProductsSatisfied frame system assignment)
    (matrix : Fin shape.matrixCount) (lane : Fin ringDegree)
    (vertex : BooleanVertex shape.rowVariables)
    (active : rowActive system matrix lane vertex = true) :
    Phi81SharedTensorRows.value assignment
        (productValue (frame.productFrame matrix lane vertex)) =
      K.mul
        (vertex.equalityWeight ConcreteCarrier.extensionOps
          (Phi81SharedTensorRows.decodedPoint frame.tensor assignment))
        (K.embed
          (matrixVectorAt ConcreteCarrier.baseOps
            (system.matrixSource.coefficientMatrix ConcreteCarrier.baseOps
              matrix lane)
            (decodedAssignment frame assignment) vertex)) := by
  have product := productRows_sound frame.productOwner
    (frame.productFirstOrdinal matrix lane vertex)
    (rowCombination frame system matrix lane vertex)
    (Phi81SharedTensorRows.chi frame.tensor vertex)
    (frame.productFrame matrix lane vertex) assignment
    (products matrix lane vertex active)
  rw [rowCombination_eval] at product
  rw [Phi81SharedTensorRows.rows_sound frame.tensor assignment tensorOne
    tensorRows vertex] at product
  rw [product]
  exact ConcreteCarrier.extensionLaws.mul_comm _ _

/-! ## Sparse sums and claimed outputs -/

def zeroValue : Extension.Value := ⟨[], []⟩

def addValue (left right : Extension.Value) : Extension.Value :=
  ⟨left.low ++ right.low, left.high ++ right.high⟩

def sumValues : List Extension.Value -> Extension.Value
  | [] => zeroValue
  | head :: tail => addValue head (sumValues tail)

private theorem value_zeroValue (assignment : ColumnId -> F) :
    Phi81SharedTensorRows.value assignment zeroValue = K.zero := by
  rfl

private theorem value_addValue
    (assignment : ColumnId -> F) (left right : Extension.Value) :
    Phi81SharedTensorRows.value assignment (addValue left right) =
      K.add (Phi81SharedTensorRows.value assignment left)
        (Phi81SharedTensorRows.value assignment right) := by
  simp only [Phi81SharedTensorRows.value, addValue, K.add, K.mk.injEq]
  rw [Phi81SharedTensorRows.eval_append,
    Phi81SharedTensorRows.eval_append]
  exact ⟨rfl, rfl⟩

private theorem value_sumValues
    (assignment : ColumnId -> F) : forall values : List Extension.Value,
    Phi81SharedTensorRows.value assignment (sumValues values) =
      FiniteSumAlgebra.sumMap ConcreteCarrier.extensionOps values
        (Phi81SharedTensorRows.value assignment)
  | [] => rfl
  | head :: tail => by
      rw [sumValues, value_addValue, value_sumValues assignment tail]
      rfl

noncomputable def productSum {shape : Phi81Relation.Shape}
    (frame : Frame shape) (system : Phi81Relation.Structure shape)
    (matrix : Fin shape.matrixCount) (lane : Fin ringDegree) :
    Extension.Value :=
  sumValues <| (activeVertices system matrix lane).map fun vertex =>
    productValue (frame.productFrame matrix lane vertex)

def claimValue {shape : Phi81Relation.Shape}
    (frame : Frame shape) (matrix : Fin shape.matrixCount)
    (lane : Fin ringDegree) : Extension.Value where
  low := [⟨frame.claimLow matrix lane, 1⟩]
  high := [⟨frame.claimHigh matrix lane, 1⟩]

noncomputable def outputRows {shape : Phi81Relation.Shape}
    (frame : Frame shape) (system : Phi81Relation.Structure shape)
    (matrix : Fin shape.matrixCount) (lane : Fin ringDegree) :
    List OwnedRow :=
  let sum := productSum frame system matrix lane
  [ Atoms.linearCheckOwnedRow frame.outputOwner
      (frame.outputFirstOrdinal matrix lane) frame.tensor.one
      (claimValue frame matrix lane).low sum.low,
    Atoms.linearCheckOwnedRow frame.outputOwner
      (frame.outputFirstOrdinal matrix lane + 1) frame.tensor.one
      (claimValue frame matrix lane).high sum.high ]

def OutputsSatisfied {shape : Phi81Relation.Shape}
    (frame : Frame shape) (system : Phi81Relation.Structure shape)
    (assignment : ColumnId -> F) : Prop :=
  forall matrix lane,
    Nightstream.Implementation.Lowering.Goldilocks.Satisfies
      (outputRows frame system matrix lane) assignment

/-! ## One exact finite evaluator program -/

/-- All verifier-derived live product rows in canonical
matrix/lane/Boolean-vertex order. -/
noncomputable def allProductRows {shape : Phi81Relation.Shape}
    (frame : Frame shape) (system : Phi81Relation.Structure shape) :
    List OwnedRow :=
  (canonicalFinIndices shape.matrixCount).flatMap fun matrix =>
    (canonicalFinIndices ringDegree).flatMap fun lane =>
      (activeVertices system matrix lane).flatMap fun vertex =>
        localProductRows frame system matrix lane vertex

/-- Both claimed-output rows for every matrix and lane in canonical order. -/
noncomputable def allOutputRows {shape : Phi81Relation.Shape}
    (frame : Frame shape) (system : Phi81Relation.Structure shape) :
    List OwnedRow :=
  (canonicalFinIndices shape.matrixCount).flatMap fun matrix =>
    (canonicalFinIndices ringDegree).flatMap fun lane =>
      outputRows frame system matrix lane

/-- Exact finite row list for the complete sparse evaluator. -/
noncomputable def rows {shape : Phi81Relation.Shape}
    (frame : Frame shape) (system : Phi81Relation.Structure shape) :
    List OwnedRow :=
  Phi81SharedTensorRows.rows frame.tensor ++
    allProductRows frame system ++ allOutputRows frame system

private theorem satisfies_flatMap_member
    {Index : Type} {parts : List Index}
    {rowsOf : Index -> List OwnedRow}
    {assignment : ColumnId -> F}
    (satisfied : Nightstream.Implementation.Lowering.Goldilocks.Satisfies
      (parts.flatMap rowsOf) assignment)
    {part : Index} (member : part ∈ parts) :
    Nightstream.Implementation.Lowering.Goldilocks.Satisfies
      (rowsOf part) assignment := by
  induction parts with
  | nil => simp at member
  | cons head tail inductionHypothesis =>
      rw [List.flatMap_cons, satisfies_append_iff] at satisfied
      rcases List.mem_cons.mp member with rfl | tailMember
      · exact satisfied.1
      · exact inductionHypothesis satisfied.2 tailMember

private theorem satisfies_flatMap_of_forall
    {Index : Type} (parts : List Index)
    (rowsOf : Index -> List OwnedRow)
    (assignment : ColumnId -> F)
    (satisfied : forall part, part ∈ parts ->
      Nightstream.Implementation.Lowering.Goldilocks.Satisfies
        (rowsOf part) assignment) :
    Nightstream.Implementation.Lowering.Goldilocks.Satisfies
      (parts.flatMap rowsOf) assignment := by
  induction parts with
  | nil => simp
  | cons head tail inductionHypothesis =>
      rw [List.flatMap_cons, satisfies_append_iff]
      exact ⟨satisfied head (by simp),
        inductionHypothesis (fun part member =>
          satisfied part (by simp [member]))⟩

private theorem allProductRows_satisfied_iff
    {shape : Phi81Relation.Shape}
    (frame : Frame shape) (system : Phi81Relation.Structure shape)
    (assignment : ColumnId -> F) :
    Nightstream.Implementation.Lowering.Goldilocks.Satisfies
        (allProductRows frame system) assignment ↔
      ProductsSatisfied frame system assignment := by
  constructor
  · intro satisfied matrix lane vertex active
    have matrixRows := satisfies_flatMap_member satisfied
      (part := matrix) (by simp [canonicalFinIndices])
    have laneRows := satisfies_flatMap_member matrixRows
      (part := lane) (by simp [canonicalFinIndices])
    exact satisfies_flatMap_member laneRows
      (part := vertex) (by
        simp [activeVertices, BooleanVertex.mem_all vertex, active])
  · intro satisfied
    apply satisfies_flatMap_of_forall
    intro matrix _
    apply satisfies_flatMap_of_forall
    intro lane _
    apply satisfies_flatMap_of_forall
    intro vertex member
    exact satisfied matrix lane vertex (List.mem_filter.mp member).2

private theorem allOutputRows_satisfied_iff
    {shape : Phi81Relation.Shape}
    (frame : Frame shape) (system : Phi81Relation.Structure shape)
    (assignment : ColumnId -> F) :
    Nightstream.Implementation.Lowering.Goldilocks.Satisfies
        (allOutputRows frame system) assignment ↔
      OutputsSatisfied frame system assignment := by
  constructor
  · intro satisfied matrix lane
    have matrixRows := satisfies_flatMap_member satisfied
      (part := matrix) (by simp [canonicalFinIndices])
    exact satisfies_flatMap_member matrixRows
      (part := lane) (by simp [canonicalFinIndices])
  · intro satisfied
    apply satisfies_flatMap_of_forall
    intro matrix _
    apply satisfies_flatMap_of_forall
    intro lane _
    exact satisfied matrix lane

theorem outputRows_sound
    {shape : Phi81Relation.Shape}
    (frame : Frame shape) (system : Phi81Relation.Structure shape)
    (assignment : ColumnId -> F)
    (constantOne : assignment frame.tensor.one = 1)
    (outputs : OutputsSatisfied frame system assignment)
    (matrix : Fin shape.matrixCount) (lane : Fin ringDegree) :
    Phi81SharedTensorRows.value assignment (claimValue frame matrix lane) =
      Phi81SharedTensorRows.value assignment
        (productSum frame system matrix lane) := by
  have satisfied := outputs matrix lane
  change
    (Atoms.linearCheckRow frame.tensor.one
      (claimValue frame matrix lane).low
      (productSum frame system matrix lane).low).Holds assignment /\
    (Atoms.linearCheckRow frame.tensor.one
      (claimValue frame matrix lane).high
      (productSum frame system matrix lane).high).Holds assignment /\
    True at satisfied
  have low :=
    (Atoms.linearCheckRow_iff assignment frame.tensor.one _ _ constantOne).1
      satisfied.1
  have high :=
    (Atoms.linearCheckRow_iff assignment frame.tensor.one _ _ constantOne).1
      satisfied.2.1
  exact congrArg₂ K.mk low high

private theorem sumMap_filter_of_false_zero
    {Index : Type} (indices : List Index) (keep : Index -> Bool)
    (term : Index -> K)
    (zero : forall index, keep index = false -> term index = K.zero) :
    FiniteSumAlgebra.sumMap ConcreteCarrier.extensionOps
        (indices.filter keep) term =
      FiniteSumAlgebra.sumMap ConcreteCarrier.extensionOps indices term := by
  induction indices with
  | nil => rfl
  | cons index indices inductionHypothesis =>
      have tail := inductionHypothesis
      have tailExact :
          BooleanTable.finiteSum ConcreteCarrier.extensionOps
              (List.map term (List.filter keep indices)) =
            BooleanTable.finiteSum ConcreteCarrier.extensionOps
              (List.map term indices) := by
        simpa only [FiniteSumAlgebra.sumMap] using tail
      cases kept : keep index with
      | false =>
          simp only [List.filter_cons, kept, Bool.false_eq_true,
            ↓reduceIte, FiniteSumAlgebra.sumMap, List.map_cons,
            BooleanTable.finiteSum]
          rw [tailExact, zero index kept]
          exact (ConcreteCarrier.extensionLaws.zero_add _).symm
      | true =>
          simp only [List.filter_cons, kept, ↓reduceIte,
            FiniteSumAlgebra.sumMap, List.map_cons,
            BooleanTable.finiteSum]
          rw [tailExact]

theorem lane_sound
    {shape : Phi81Relation.Shape}
    (frame : Frame shape) (system : Phi81Relation.Structure shape)
    (assignment : ColumnId -> F)
    (constantOne : assignment frame.tensor.one = 1)
    (tensorRows : Phi81SharedTensorRows.RowsSatisfied frame.tensor assignment)
    (products : ProductsSatisfied frame system assignment)
    (outputs : OutputsSatisfied frame system assignment)
    (matrix : Fin shape.matrixCount) (lane : Fin ringDegree) :
    Phi81SharedTensorRows.value assignment (claimValue frame matrix lane) =
      Phi81Relation.matrixEvaluation system
        (decodedAssignment frame assignment)
        (Phi81SharedTensorRows.decodedPoint frame.tensor assignment)
        matrix lane := by
  let point := Phi81SharedTensorRows.decodedPoint frame.tensor assignment
  let matrixValues := fun vertex : BooleanVertex shape.rowVariables =>
    K.embed
      (matrixVectorAt ConcreteCarrier.baseOps
        (system.matrixSource.coefficientMatrix ConcreteCarrier.baseOps
          matrix lane)
        (decodedAssignment frame assignment) vertex)
  have productsExact :
      forall vertex, vertex ∈ activeVertices system matrix lane ->
        Phi81SharedTensorRows.value assignment
            (productValue (frame.productFrame matrix lane vertex)) =
          K.mul
            (vertex.equalityWeight ConcreteCarrier.extensionOps point)
            (matrixValues vertex) := by
    intro vertex member
    have active : rowActive system matrix lane vertex = true :=
      (List.mem_filter.mp member).2
    simpa [point, matrixValues] using
      localProduct_sound frame system assignment constantOne tensorRows
        products matrix lane vertex active
  calc
    Phi81SharedTensorRows.value assignment (claimValue frame matrix lane) =
        Phi81SharedTensorRows.value assignment
          (productSum frame system matrix lane) :=
      outputRows_sound frame system assignment constantOne outputs matrix lane
    _ = FiniteSumAlgebra.sumMap ConcreteCarrier.extensionOps
          (activeVertices system matrix lane)
          (fun vertex => Phi81SharedTensorRows.value assignment
            (productValue (frame.productFrame matrix lane vertex))) := by
      rw [productSum, value_sumValues]
      unfold FiniteSumAlgebra.sumMap
      rw [List.map_map]
      rfl
    _ = FiniteSumAlgebra.sumMap ConcreteCarrier.extensionOps
          (activeVertices system matrix lane)
          (fun vertex =>
            K.mul
              (vertex.equalityWeight ConcreteCarrier.extensionOps point)
              (matrixValues vertex)) := by
      apply FiniteSumAlgebra.sumMap_congr
      exact productsExact
    _ = FiniteSumAlgebra.sumMap ConcreteCarrier.extensionOps
          (BooleanVertex.all shape.rowVariables)
          (fun vertex =>
            K.mul
              (vertex.equalityWeight ConcreteCarrier.extensionOps point)
              (matrixValues vertex)) := by
      apply sumMap_filter_of_false_zero
      intro vertex inactive
      have matrixZero := inactive_matrixVector_zero frame system assignment
        matrix lane vertex inactive
      simp [matrixValues, matrixZero, ConcreteCarrier.extensionOps, K.embed,
        K.mul, K.zero]
    _ = BooleanReproduction.equalityWeighted ConcreteCarrier.extensionOps
          point matrixValues := rfl
    _ = (BooleanTable.tabulate matrixValues).evaluate
          ConcreteCarrier.extensionOps point :=
      BooleanReproduction.equalityWeighted_tabulate_eq_evaluate
        ConcreteCarrier.extensionOps ConcreteCarrier.extensionLaws point
          matrixValues
    _ = Phi81Relation.matrixEvaluation system
          (decodedAssignment frame assignment) point matrix lane := by
      rfl

/-- All row families needed by the dynamic sparse evaluator. -/
structure RowsSatisfied {shape : Phi81Relation.Shape}
    (frame : Frame shape) (system : Phi81Relation.Structure shape)
    (assignment : ColumnId -> F) : Prop where
  tensor : Phi81SharedTensorRows.RowsSatisfied frame.tensor assignment
  products : ProductsSatisfied frame system assignment
  outputs : OutputsSatisfied frame system assignment

/-- Satisfaction of the one finite evaluator row list is exactly the
previous semantic row-family record. No quantified row family can be omitted
from a generated manifest. -/
theorem rows_satisfied_iff
    {shape : Phi81Relation.Shape}
    (frame : Frame shape) (system : Phi81Relation.Structure shape)
    (assignment : ColumnId -> F) :
    Nightstream.Implementation.Lowering.Goldilocks.Satisfies
        (rows frame system) assignment ↔
      RowsSatisfied frame system assignment := by
  rw [rows, satisfies_append_iff, satisfies_append_iff,
    Phi81SharedTensorRows.rows_satisfied_iff,
    allProductRows_satisfied_iff, allOutputRows_satisfied_iff]
  constructor
  · rintro ⟨⟨tensor, products⟩, outputs⟩
    exact ⟨tensor, products, outputs⟩
  · intro satisfied
    exact ⟨⟨satisfied.tensor, satisfied.products⟩, satisfied.outputs⟩

/-- Headline theorem: actual tensor, sparse product, and output rows bind the
entire claimed array to the independent Phi81 relation. -/
theorem rows_sound
    {shape : Phi81Relation.Shape}
    (frame : Frame shape) (system : Phi81Relation.Structure shape)
    (assignment : ColumnId -> F)
    (constantOne : assignment frame.tensor.one = 1)
    (satisfied : RowsSatisfied frame system assignment) :
    decodedEvaluations frame assignment =
      Phi81Relation.evaluations system
        (decodedAssignment frame assignment)
        (Phi81SharedTensorRows.decodedPoint frame.tensor assignment) := by
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
    simpa [claimValue, Phi81SharedTensorRows.value,
      LinearCombination.eval] using
      lane_sound frame system assignment constantOne satisfied.tensor
        satisfied.products satisfied.outputs matrixIndex lane

/-- Exact sparse row cost for one concrete verifier-owned structure. -/
noncomputable def productRowCount {shape : Phi81Relation.Shape}
    (system : Phi81Relation.Structure shape) : Nat :=
  (canonicalFinIndices shape.matrixCount).foldl
    (fun total matrix =>
      total + (canonicalFinIndices ringDegree).foldl
        (fun laneTotal lane =>
          laneTotal + 2 * (activeVertices system matrix lane).length) 0) 0

noncomputable def rowCount {shape : Phi81Relation.Shape}
    (system : Phi81Relation.Structure shape) : Nat :=
  Phi81SharedTensorRows.rowCount shape.rowVariables +
    productRowCount system + 2 * shape.matrixCount * ringDegree

end Nightstream.Implementation.R1CS.Phi81SharedEvaluationRows
