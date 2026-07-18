import Nightstream.SuperNeo.Concrete.Phi81Relation.Evaluation
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ConcreteCarrier

/-!
Base-field linearity of the typed Phi81 carried-evaluation map.

Protocol: SuperNeo Theorem 5 restricted to constant-ring scalars, and the
base-`b` recomposition used by `Pi_DEC`.
Phase: complete-carrier assignment combination followed by one matrix/Phi81
evaluation.
Constraint family: semantic evaluation only; this file emits no rows.

Owns: raw width-only and typed relation assignment operations; their exact
refinement equality; the corresponding operations on all 54 `RingK` lanes;
canonical finite base-field combinations; generic zero/add/scale laws for
`K`-valued Boolean table evaluation; and proofs that `matrixEvaluation`
preserves them.

Does not own: arbitrary `RingF` challenge action, complete-carrier block
packing, Phi81 multiplication associativity, commitments, public-input
projection, norm bounds, transcripts, Rust, R1CS, row removal, or counts.

Emits constraints: no.

Authority boundary: the theorem starts from the sole derived Phi81 matrix
source and the typed complete assignment. No evaluation array, matrix image,
or linearity oracle is supplied by a caller. This is enough for the field
weights `b^(i-1)` in `Pi_DEC`; it is deliberately not advertised as the
`RingF`-module theorem required by `Pi_RLC`.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `nifs.pi_dec.verify.recomposition.assignment` | add, scale, and finite-combine every coordinate of the complete assignment | computed | `assignmentAdd`, `assignmentScale`, `combineAssignments` |
| `nifs.shared.assignment.raw_refinement` | width-only and relation-typed finite folds are exactly equal | derived | `raw_combineAssignments_eq` |
| `nifs.pi_dec.verify.recomposition.matrix_row` | each finite matrix image is base-`F` linear in that assignment | derived | `matrixEvaluation_zero`, `matrixEvaluation_add`, `matrixEvaluation_scale` |
| `nifs.pi_dec.verify.recomposition.mle` | the independently defined Boolean MLE preserves the same embedded-`F` operations in all 54 lanes | derived | `matrixEvaluation_combine` |
| `nifs.pi_dec.verify.recomposition.evaluations` | every matrix in canonical order preserves the same finite combination | derived | `evaluations_combine` |
-/

namespace Nightstream.SuperNeo.Concrete.Phi81Relation.EvaluationHomomorphism.BaseLinear

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.PaperLinearAlgebra

/-! ## Exact base-linear carriers -/

namespace Raw

/-- Add two raw finite assignments coordinatewise. -/
def assignmentAdd {columns : Nat}
    (left right : PaperLinearAlgebra.Assignment F columns) :
    PaperLinearAlgebra.Assignment F columns :=
  fun column => left column + right column

/-- Scale one raw finite assignment by a base-field element. -/
def assignmentScale {columns : Nat}
    (scalar : F) (assignment : PaperLinearAlgebra.Assignment F columns) :
    PaperLinearAlgebra.Assignment F columns :=
  fun column => scalar * assignment column

/-- The canonical zero raw finite assignment. -/
def assignmentZero {columns : Nat} :
    PaperLinearAlgebra.Assignment F columns :=
  fun _ => 0

/-- Canonical head-first finite base-field combination at an arbitrary width. -/
def combineAssignments {columns : Nat} :
    {count : Nat} ->
      (Fin count -> F) ->
      (Fin count -> PaperLinearAlgebra.Assignment F columns) ->
      PaperLinearAlgebra.Assignment F columns
  | 0, _, _ => assignmentZero
  | _ + 1, weights, assignments =>
      assignmentAdd
        (assignmentScale (weights 0) (assignments 0))
        (combineAssignments
          (fun index => weights index.succ)
          (fun index => assignments index.succ))

end Raw

/-- Add two typed complete-carrier relation assignments coordinatewise. -/
def assignmentAdd {shape : Shape}
    (left right : Assignment shape) : Assignment shape :=
  Raw.assignmentAdd left right

/-- Scale one typed complete-carrier relation assignment. -/
def assignmentScale {shape : Shape}
    (scalar : F) (assignment : Assignment shape) : Assignment shape :=
  Raw.assignmentScale scalar assignment

/-- The canonical zero typed complete-carrier relation assignment. -/
def assignmentZero {shape : Shape} : Assignment shape :=
  Raw.assignmentZero

/-- Add two Phi81 evaluation rings coefficientwise. -/
def evaluationAdd (left right : Evaluation) : Evaluation :=
  fun lane => K.add (left lane) (right lane)

/-- Scale every Phi81 evaluation coefficient by an embedded base scalar. -/
def evaluationScale (scalar : F) (evaluation : Evaluation) : Evaluation :=
  fun lane => K.mul (K.embed scalar) (evaluation lane)

/-- The canonical zero Phi81 evaluation. -/
def evaluationZero : Evaluation := ringKZero

/-- Canonical head-first finite base-field combination of typed assignments. -/
def combineAssignments {shape : Shape} :
    {count : Nat} ->
      (Fin count -> F) -> (Fin count -> Assignment shape) -> Assignment shape
  | 0, _, _ => assignmentZero
  | _ + 1, weights, assignments =>
      assignmentAdd
        (assignmentScale (weights 0) (assignments 0))
        (combineAssignments
          (fun index => weights index.succ)
          (fun index => assignments index.succ))

/-- The raw width-only fold and the typed relation fold are extensionally
identical. This is the refinement seam used by independent packed semantics;
it prevents the two recursive definitions from drifting silently. -/
theorem raw_combineAssignments_eq
    {shape : Shape} {count : Nat}
    (weights : Fin count -> F)
    (assignments : Fin count -> Assignment shape) :
    Raw.combineAssignments weights assignments =
      combineAssignments weights assignments := by
  induction count with
  | zero => rfl
  | succ count inductionHypothesis =>
      simp only [Raw.combineAssignments, combineAssignments]
      rw [inductionHypothesis
        (fun index => weights index.succ)
        (fun index => assignments index.succ)]
      rfl

/-- The identical finite combination on one `RingK` evaluation. -/
def combineEvaluations :
    {count : Nat} ->
      (Fin count -> F) -> (Fin count -> Evaluation) -> Evaluation
  | 0, _, _ => evaluationZero
  | _ + 1, weights, evaluations =>
      evaluationAdd
        (evaluationScale (weights 0) (evaluations 0))
        (combineEvaluations
          (fun index => weights index.succ)
          (fun index => evaluations index.succ))

/-! ## Finite matrix-image linearity -/

private def sumTerms
    {Value Index : Type}
    (zero : Value) (add : Value -> Value -> Value) :
    List Index -> (Index -> Value) -> Value
  | [], _ => zero
  | index :: indices, term =>
      add (term index) (sumTerms zero add indices term)

private theorem foldl_eq_add_sumTerms
    {Value Index : Type}
    (zero : Value) (add : Value -> Value -> Value)
    (addAssoc : forall left middle right,
      add (add left middle) right = add left (add middle right))
    (addZero : forall value, add value zero = value)
    (indices : List Index) (term : Index -> Value) (initial : Value) :
    indices.foldl (fun accumulated index => add accumulated (term index)) initial =
      add initial (sumTerms zero add indices term) := by
  induction indices generalizing initial with
  | nil => exact (addZero initial).symm
  | cons index indices inductionHypothesis =>
      rw [List.foldl_cons, inductionHypothesis]
      exact addAssoc initial (term index) (sumTerms zero add indices term)

private theorem sumTerms_zero
    {Value Index : Type}
    (zero : Value) (add : Value -> Value -> Value)
    (zeroAdd : forall value, add zero value = value)
    (indices : List Index) :
    sumTerms zero add indices (fun _ => zero) = zero := by
  induction indices with
  | nil => rfl
  | cons index indices inductionHypothesis =>
      simp only [sumTerms, inductionHypothesis]
      exact zeroAdd zero

private theorem sumTerms_add
    {Value Index : Type}
    (zero : Value) (add : Value -> Value -> Value)
    (addAssoc : forall left middle right,
      add (add left middle) right = add left (add middle right))
    (addComm : forall left right, add left right = add right left)
    (zeroAdd : forall value, add zero value = value)
    (indices : List Index) (left right : Index -> Value) :
    sumTerms zero add indices (fun index => add (left index) (right index)) =
      add (sumTerms zero add indices left) (sumTerms zero add indices right) := by
  induction indices with
  | nil => exact (zeroAdd zero).symm
  | cons index indices inductionHypothesis =>
      simp only [sumTerms, inductionHypothesis]
      calc
        add (add (left index) (right index))
            (add (sumTerms zero add indices left)
              (sumTerms zero add indices right)) =
            add (left index)
              (add (right index)
                (add (sumTerms zero add indices left)
                  (sumTerms zero add indices right))) :=
          addAssoc _ _ _
        _ = add (left index)
              (add (sumTerms zero add indices left)
                (add (right index)
                  (sumTerms zero add indices right))) := by
          congr 1
          calc
            add (right index)
                (add (sumTerms zero add indices left)
                  (sumTerms zero add indices right)) =
                add (add (right index) (sumTerms zero add indices left))
                  (sumTerms zero add indices right) :=
              (addAssoc _ _ _).symm
            _ = add (add (sumTerms zero add indices left) (right index))
                  (sumTerms zero add indices right) := by
              rw [addComm (right index) (sumTerms zero add indices left)]
            _ = add (sumTerms zero add indices left)
                  (add (right index) (sumTerms zero add indices right)) :=
              addAssoc _ _ _
        _ = add
              (add (left index) (sumTerms zero add indices left))
              (add (right index) (sumTerms zero add indices right)) :=
          (addAssoc _ _ _).symm

private theorem sumTerms_scale
    {Value Scalar Index : Type}
    (zero : Value) (add : Value -> Value -> Value)
    (scale : Scalar -> Value -> Value)
    (zeroScale : forall scalar, scale scalar zero = zero)
    (scaleAdd : forall scalar left right,
      scale scalar (add left right) =
        add (scale scalar left) (scale scalar right))
    (indices : List Index) (scalar : Scalar) (term : Index -> Value) :
    sumTerms zero add indices (fun index => scale scalar (term index)) =
      scale scalar (sumTerms zero add indices term) := by
  induction indices with
  | nil => exact (zeroScale scalar).symm
  | cons index indices inductionHypothesis =>
      simp only [sumTerms, inductionHypothesis, scaleAdd]

private theorem matrixVectorAt_zero
    {variables columns : Nat}
    (matrix : BooleanMatrix F variables columns)
    (vertex : BooleanVertex variables) :
    matrixVectorAt ConcreteCarrier.baseOps matrix (fun _ => 0) vertex = 0 := by
  unfold matrixVectorAt
  change (canonicalFinIndices columns).foldl
      (fun accumulated column =>
        accumulated + matrix vertex column * 0) 0 = 0
  let indices := canonicalFinIndices columns
  let add : F -> F -> F := fun left right => left + right
  have termsZero :
      (fun column : Fin columns => matrix vertex column * 0) =
        (fun _ => (0 : F)) := by
    funext column
    exact Fin.mul_zero _
  calc
    indices.foldl
        (fun accumulated column =>
          accumulated + matrix vertex column * 0) 0 =
        0 + sumTerms 0 add indices
          (fun column => matrix vertex column * 0) :=
      foldl_eq_add_sumTerms 0 add ConcreteCarrier.baseLaws.add_assoc
        ConcreteCarrier.baseLaws.add_zero indices _ 0
    _ = sumTerms 0 add indices
          (fun column => matrix vertex column * 0) :=
      ConcreteCarrier.baseLaws.zero_add _
    _ = sumTerms 0 add indices (fun _ => 0) := by rw [termsZero]
    _ = 0 := sumTerms_zero 0 add ConcreteCarrier.baseLaws.zero_add indices

private theorem matrixVectorAt_add
    {variables columns : Nat}
    (matrix : BooleanMatrix F variables columns)
    (left right : PaperLinearAlgebra.Assignment F columns)
    (vertex : BooleanVertex variables) :
    matrixVectorAt ConcreteCarrier.baseOps matrix
        (fun column => left column + right column) vertex =
      matrixVectorAt ConcreteCarrier.baseOps matrix left vertex +
        matrixVectorAt ConcreteCarrier.baseOps matrix right vertex := by
  unfold matrixVectorAt
  change (canonicalFinIndices columns).foldl
      (fun accumulated column =>
        accumulated + matrix vertex column * (left column + right column)) 0 =
    (canonicalFinIndices columns).foldl
        (fun accumulated column =>
          accumulated + matrix vertex column * left column) 0 +
      (canonicalFinIndices columns).foldl
        (fun accumulated column =>
          accumulated + matrix vertex column * right column) 0
  let indices := canonicalFinIndices columns
  let add : F -> F -> F := fun left right => left + right
  let leftTerm : Fin columns -> F :=
    fun column => matrix vertex column * left column
  let rightTerm : Fin columns -> F :=
    fun column => matrix vertex column * right column
  have distribute :
      (fun column : Fin columns =>
        matrix vertex column * (left column + right column)) =
        (fun column => add (leftTerm column) (rightTerm column)) := by
    funext column
    exact ConcreteCarrier.baseLaws.left_distrib _ _ _
  have leftFold :
      indices.foldl
          (fun accumulated column => accumulated + leftTerm column) 0 =
        sumTerms 0 add indices leftTerm := by
    calc
      _ = 0 + sumTerms 0 add indices leftTerm :=
        foldl_eq_add_sumTerms 0 add ConcreteCarrier.baseLaws.add_assoc
          ConcreteCarrier.baseLaws.add_zero indices leftTerm 0
      _ = _ := ConcreteCarrier.baseLaws.zero_add _
  have rightFold :
      indices.foldl
          (fun accumulated column => accumulated + rightTerm column) 0 =
        sumTerms 0 add indices rightTerm := by
    calc
      _ = 0 + sumTerms 0 add indices rightTerm :=
        foldl_eq_add_sumTerms 0 add ConcreteCarrier.baseLaws.add_assoc
          ConcreteCarrier.baseLaws.add_zero indices rightTerm 0
      _ = _ := ConcreteCarrier.baseLaws.zero_add _
  calc
    indices.foldl
        (fun accumulated column =>
          accumulated + matrix vertex column * (left column + right column)) 0 =
        0 + sumTerms 0 add indices
          (fun column => matrix vertex column * (left column + right column)) :=
      foldl_eq_add_sumTerms 0 add ConcreteCarrier.baseLaws.add_assoc
        ConcreteCarrier.baseLaws.add_zero indices _ 0
    _ = sumTerms 0 add indices
          (fun column => matrix vertex column * (left column + right column)) :=
      ConcreteCarrier.baseLaws.zero_add _
    _ = sumTerms 0 add indices
          (fun column => add (leftTerm column) (rightTerm column)) := by
      rw [distribute]
    _ = add (sumTerms 0 add indices leftTerm)
          (sumTerms 0 add indices rightTerm) :=
      sumTerms_add 0 add ConcreteCarrier.baseLaws.add_assoc
        ConcreteCarrier.baseLaws.add_comm ConcreteCarrier.baseLaws.zero_add
        indices leftTerm rightTerm
    _ = indices.foldl
          (fun accumulated column => accumulated + leftTerm column) 0 +
        indices.foldl
          (fun accumulated column => accumulated + rightTerm column) 0 := by
      rw [leftFold, rightFold]

private theorem matrixVectorAt_scale
    {variables columns : Nat}
    (matrix : BooleanMatrix F variables columns)
    (scalar : F) (assignment : PaperLinearAlgebra.Assignment F columns)
    (vertex : BooleanVertex variables) :
    matrixVectorAt ConcreteCarrier.baseOps matrix
        (fun column => scalar * assignment column) vertex =
      scalar * matrixVectorAt ConcreteCarrier.baseOps matrix assignment vertex := by
  unfold matrixVectorAt
  change (canonicalFinIndices columns).foldl
      (fun accumulated column =>
        accumulated + matrix vertex column * (scalar * assignment column)) 0 =
    scalar * (canonicalFinIndices columns).foldl
      (fun accumulated column =>
        accumulated + matrix vertex column * assignment column) 0
  let indices := canonicalFinIndices columns
  let add : F -> F -> F := fun left right => left + right
  let scale : F -> F -> F := fun s value => s * value
  let term : Fin columns -> F :=
    fun column => matrix vertex column * assignment column
  have commuteScale :
      (fun column : Fin columns =>
        matrix vertex column * (scalar * assignment column)) =
        (fun column => scale scalar (term column)) := by
    funext column
    calc
      matrix vertex column * (scalar * assignment column) =
          (matrix vertex column * scalar) * assignment column :=
        (ConcreteCarrier.baseLaws.mul_assoc _ _ _).symm
      _ = (scalar * matrix vertex column) * assignment column := by
        rw [Fin.mul_comm (matrix vertex column) scalar]
      _ = scalar * (matrix vertex column * assignment column) :=
        ConcreteCarrier.baseLaws.mul_assoc _ _ _
  have termFold :
      indices.foldl
          (fun accumulated column => accumulated + term column) 0 =
        sumTerms 0 add indices term := by
    calc
      _ = 0 + sumTerms 0 add indices term :=
        foldl_eq_add_sumTerms 0 add ConcreteCarrier.baseLaws.add_assoc
          ConcreteCarrier.baseLaws.add_zero indices term 0
      _ = _ := ConcreteCarrier.baseLaws.zero_add _
  calc
    indices.foldl
        (fun accumulated column =>
          accumulated + matrix vertex column * (scalar * assignment column)) 0 =
        0 + sumTerms 0 add indices
          (fun column => matrix vertex column * (scalar * assignment column)) :=
      foldl_eq_add_sumTerms 0 add ConcreteCarrier.baseLaws.add_assoc
        ConcreteCarrier.baseLaws.add_zero indices _ 0
    _ = sumTerms 0 add indices
          (fun column => matrix vertex column * (scalar * assignment column)) :=
      ConcreteCarrier.baseLaws.zero_add _
    _ = sumTerms 0 add indices (fun column => scale scalar (term column)) := by
      rw [commuteScale]
    _ = scale scalar (sumTerms 0 add indices term) :=
      sumTerms_scale 0 add scale ConcreteCarrier.baseLaws.mul_zero
        ConcreteCarrier.baseLaws.left_distrib indices scalar term
    _ = scalar * indices.foldl
          (fun accumulated column => accumulated + term column) 0 := by
      rw [termFold]

/-! ## Independently recursive MLE linearity -/

private theorem mul_neg
    {Value : Type}
    (ops : InterpolationOps Value)
    (laws : InterpolationEvaluationLaws ops)
    (left right : Value) :
    ops.mul left (ops.neg right) = ops.neg (ops.mul left right) := by
  calc
    ops.mul left (ops.neg right) = ops.mul (ops.neg right) left :=
      laws.mul_comm _ _
    _ = ops.neg (ops.mul right left) := laws.neg_mul _ _
    _ = ops.neg (ops.mul left right) := by rw [laws.mul_comm right left]

private theorem scale_sub
    {Value : Type}
    (ops : InterpolationOps Value)
    (laws : InterpolationEvaluationLaws ops)
    (scalar left right : Value) :
    ops.mul scalar (ops.sub left right) =
      ops.sub (ops.mul scalar left) (ops.mul scalar right) := by
  unfold InterpolationOps.sub
  rw [laws.left_distrib, mul_neg ops laws]

private theorem interpolate_add_identity
    {Value : Type}
    (ops : InterpolationOps Value)
    (laws : InterpolationEvaluationLaws ops)
    (coordinate leftLow leftHigh rightLow rightHigh : Value) :
    ops.add
        (ops.add leftLow rightLow)
        (ops.mul coordinate
          (ops.sub (ops.add leftHigh rightHigh)
            (ops.add leftLow rightLow))) =
      ops.add
        (ops.add leftLow
          (ops.mul coordinate (ops.sub leftHigh leftLow)))
        (ops.add rightLow
          (ops.mul coordinate (ops.sub rightHigh rightLow))) := by
  unfold InterpolationOps.sub
  simp only [laws.neg_add, laws.left_distrib]
  letI : Std.Associative ops.add := ⟨laws.add_assoc⟩
  letI : Std.Commutative ops.add := ⟨laws.add_comm⟩
  ac_rfl

private theorem interpolate_scale_identity
    {Value : Type}
    (ops : InterpolationOps Value)
    (laws : InterpolationEvaluationLaws ops)
    (scalar coordinate low high : Value) :
    ops.add
        (ops.mul scalar low)
        (ops.mul coordinate
          (ops.sub (ops.mul scalar high) (ops.mul scalar low))) =
      ops.mul scalar
        (ops.add low (ops.mul coordinate (ops.sub high low))) := by
  rw [← scale_sub ops laws]
  rw [laws.left_distrib]
  congr 1
  calc
    ops.mul coordinate (ops.mul scalar (ops.sub high low)) =
        ops.mul (ops.mul coordinate scalar) (ops.sub high low) :=
      (laws.mul_assoc _ _ _).symm
    _ = ops.mul (ops.mul scalar coordinate) (ops.sub high low) := by
      rw [laws.mul_comm coordinate scalar]
    _ = ops.mul scalar (ops.mul coordinate (ops.sub high low)) :=
      laws.mul_assoc _ _ _

private theorem evaluateCoordinates_tabulate_zero
    {variables : Nat}
    (coordinates : List K) :
    (BooleanTable.tabulate (fun _ : BooleanVertex variables => K.zero)).evaluateCoordinates
        ConcreteCarrier.extensionOps coordinates = K.zero := by
  induction variables generalizing coordinates with
  | zero =>
      cases coordinates with
      | nil => rfl
      | cons coordinate coordinates => rfl
  | succ variables inductionHypothesis =>
      cases coordinates with
      | nil => rfl
      | cons coordinate coordinates =>
          simp only [BooleanTable.tabulate, BooleanTable.evaluateCoordinates,
            inductionHypothesis]
          rw [show ConcreteCarrier.extensionOps.sub K.zero K.zero = K.zero by
            exact ConcreteCarrier.extensionLaws.add_neg K.zero]
          calc
            ConcreteCarrier.extensionOps.add K.zero
                (ConcreteCarrier.extensionOps.mul coordinate K.zero) =
              ConcreteCarrier.extensionOps.mul coordinate K.zero :=
                ConcreteCarrier.extensionLaws.zero_add _
            _ = K.zero := ConcreteCarrier.extensionLaws.mul_zero coordinate

private theorem evaluateCoordinates_tabulate_add
    {variables : Nat}
    (left right : BooleanVertex variables -> K)
    (coordinates : List K) :
    (BooleanTable.tabulate (fun vertex => K.add (left vertex) (right vertex))).evaluateCoordinates
        ConcreteCarrier.extensionOps coordinates =
      K.add
        ((BooleanTable.tabulate left).evaluateCoordinates
          ConcreteCarrier.extensionOps coordinates)
        ((BooleanTable.tabulate right).evaluateCoordinates
          ConcreteCarrier.extensionOps coordinates) := by
  induction variables generalizing coordinates with
  | zero =>
      cases coordinates with
      | nil => rfl
      | cons coordinate coordinates =>
          exact (ConcreteCarrier.extensionLaws.zero_add K.zero).symm
  | succ variables inductionHypothesis =>
      cases coordinates with
      | nil => exact (ConcreteCarrier.extensionLaws.zero_add K.zero).symm
      | cons coordinate coordinates =>
          simp only [BooleanTable.tabulate, BooleanTable.evaluateCoordinates,
            inductionHypothesis]
          exact interpolate_add_identity ConcreteCarrier.extensionOps
            ConcreteCarrier.extensionLaws _ _ _ _ _

private theorem evaluateCoordinates_tabulate_scale
    {variables : Nat}
    (scalar : K) (values : BooleanVertex variables -> K)
    (coordinates : List K) :
    (BooleanTable.tabulate (fun vertex => K.mul scalar (values vertex))).evaluateCoordinates
        ConcreteCarrier.extensionOps coordinates =
      K.mul scalar
        ((BooleanTable.tabulate values).evaluateCoordinates
          ConcreteCarrier.extensionOps coordinates) := by
  induction variables generalizing coordinates with
  | zero =>
      cases coordinates with
      | nil => rfl
      | cons coordinate coordinates =>
          exact (ConcreteCarrier.extensionLaws.mul_zero scalar).symm
  | succ variables inductionHypothesis =>
      cases coordinates with
      | nil => exact (ConcreteCarrier.extensionLaws.mul_zero scalar).symm
      | cons coordinate coordinates =>
          simp only [BooleanTable.tabulate, BooleanTable.evaluateCoordinates,
            inductionHypothesis]
          exact interpolate_scale_identity ConcreteCarrier.extensionOps
            ConcreteCarrier.extensionLaws _ _ _ _

/-- A canonically tabulated zero `K` table evaluates to zero at every typed
point. -/
theorem evaluateTabulated_zero
    {variables : Nat}
    (point : CubePoint K variables) :
    (BooleanTable.tabulate
      (fun _ : BooleanVertex variables => K.zero)).evaluate
        ConcreteCarrier.extensionOps point = K.zero := by
  unfold BooleanTable.evaluate
  exact evaluateCoordinates_tabulate_zero point.coordinates

/-- Canonical `K`-valued Boolean-table evaluation is additive. -/
theorem evaluateTabulated_add
    {variables : Nat}
    (left right : BooleanVertex variables -> K)
    (point : CubePoint K variables) :
    (BooleanTable.tabulate
      (fun vertex => K.add (left vertex) (right vertex))).evaluate
        ConcreteCarrier.extensionOps point =
      K.add
        ((BooleanTable.tabulate left).evaluate
          ConcreteCarrier.extensionOps point)
        ((BooleanTable.tabulate right).evaluate
          ConcreteCarrier.extensionOps point) := by
  unfold BooleanTable.evaluate
  exact evaluateCoordinates_tabulate_add left right point.coordinates

/-- Canonical `K`-valued Boolean-table evaluation commutes with a fixed
extension-field scalar. -/
theorem evaluateTabulated_scale
    {variables : Nat}
    (scalar : K)
    (values : BooleanVertex variables -> K)
    (point : CubePoint K variables) :
    (BooleanTable.tabulate
      (fun vertex => K.mul scalar (values vertex))).evaluate
        ConcreteCarrier.extensionOps point =
      K.mul scalar
        ((BooleanTable.tabulate values).evaluate
          ConcreteCarrier.extensionOps point) := by
  unfold BooleanTable.evaluate
  exact evaluateCoordinates_tabulate_scale scalar values point.coordinates

/-! ## Typed Phi81 evaluator theorems -/

/-- The typed Phi81 evaluation of the zero complete assignment is zero in all
54 lanes. -/
theorem matrixEvaluation_zero
    {shape : Shape}
    (system : Structure shape) (point : Point shape)
    (matrix : Fin shape.matrixCount) :
    matrixEvaluation system assignmentZero point matrix = evaluationZero := by
  funext lane
  unfold matrixEvaluation Phi81Evaluation.evaluate Phi81Evaluation.table
    assignmentZero Raw.assignmentZero evaluationZero ringKZero
  unfold BooleanTable.evaluate
  simpa only [matrixVectorAt_zero] using
    (evaluateCoordinates_tabulate_zero
      (variables := shape.rowVariables) point.coordinates)

/-- One matrix's complete 54-lane Phi81 evaluation is additive in the exact
complete-carrier assignment. -/
theorem matrixEvaluation_add
    {shape : Shape}
    (system : Structure shape)
    (left right : Assignment shape) (point : Point shape)
    (matrix : Fin shape.matrixCount) :
    matrixEvaluation system (assignmentAdd left right) point matrix =
      evaluationAdd
        (matrixEvaluation system left point matrix)
        (matrixEvaluation system right point matrix) := by
  funext lane
  unfold matrixEvaluation Phi81Evaluation.evaluate Phi81Evaluation.table
    assignmentAdd Raw.assignmentAdd evaluationAdd
  unfold BooleanTable.evaluate
  simpa only [matrixVectorAt_add, ConcreteCarrier.embed_add] using
    (evaluateCoordinates_tabulate_add
      (fun vertex => K.embed (matrixVectorAt ConcreteCarrier.baseOps
        (system.matrixSource.coefficientMatrix ConcreteCarrier.baseOps matrix lane)
        left vertex))
      (fun vertex => K.embed (matrixVectorAt ConcreteCarrier.baseOps
        (system.matrixSource.coefficientMatrix ConcreteCarrier.baseOps matrix lane)
        right vertex))
      point.coordinates)

/-- One matrix's complete 54-lane Phi81 evaluation commutes with an embedded
base-field scalar. -/
theorem matrixEvaluation_scale
    {shape : Shape}
    (system : Structure shape)
    (scalar : F) (assignment : Assignment shape) (point : Point shape)
    (matrix : Fin shape.matrixCount) :
    matrixEvaluation system (assignmentScale scalar assignment) point matrix =
      evaluationScale scalar (matrixEvaluation system assignment point matrix) := by
  funext lane
  unfold matrixEvaluation Phi81Evaluation.evaluate Phi81Evaluation.table
    assignmentScale Raw.assignmentScale evaluationScale
  unfold BooleanTable.evaluate
  have embedScale (value : F) :
      K.embed (scalar * value) = K.mul (K.embed scalar) (K.embed value) := by
    simpa only [ConcreteCarrier.baseOps, ConcreteCarrier.extensionOps] using
      (ConcreteCarrier.embed_mul scalar value)
  simpa only [matrixVectorAt_scale, embedScale] using
    (evaluateCoordinates_tabulate_scale (K.embed scalar)
      (fun vertex => K.embed (matrixVectorAt ConcreteCarrier.baseOps
        (system.matrixSource.coefficientMatrix ConcreteCarrier.baseOps matrix lane)
        assignment vertex))
      point.coordinates)

/-- `Pi_DEC`'s exact finite base-field recomposition commutes with every
matrix and every one of the 54 Phi81 evaluation lanes. -/
theorem matrixEvaluation_combine
    {shape : Shape} {count : Nat}
    (system : Structure shape)
    (weights : Fin count -> F)
    (assignments : Fin count -> Assignment shape)
    (point : Point shape) (matrix : Fin shape.matrixCount) :
    matrixEvaluation system (combineAssignments weights assignments) point matrix =
      combineEvaluations weights
        (fun index => matrixEvaluation system (assignments index) point matrix) := by
  induction count with
  | zero => exact matrixEvaluation_zero system point matrix
  | succ count inductionHypothesis =>
      rw [combineAssignments, combineEvaluations, matrixEvaluation_add,
        matrixEvaluation_scale, inductionHypothesis]

/-- Array-level form: canonical matrix ordering and all 54 lanes preserve the
same finite base-field combination. -/
theorem evaluations_combine
    {shape : Shape} {count : Nat}
    (system : Structure shape)
    (weights : Fin count -> F)
    (assignments : Fin count -> Assignment shape)
    (point : Point shape) :
    evaluations system (combineAssignments weights assignments) point =
      Array.ofFn fun matrix =>
        combineEvaluations weights
          (fun index => matrixEvaluation system (assignments index) point matrix) := by
  apply Array.ext
  · simp [evaluations]
  · intro index leftLt rightLt
    let matrix : Fin shape.matrixCount :=
      ⟨index, by simpa [evaluations] using leftLt⟩
    simpa [matrix, evaluations] using
      matrixEvaluation_combine system weights assignments point matrix

end Nightstream.SuperNeo.Concrete.Phi81Relation.EvaluationHomomorphism.BaseLinear
