import Batteries.Data.Fin.Lemmas
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsCanonicalViews

/-!
Contract: prove that the Lean-owned running codec names every serialized
base-field coordinate exactly once through `RunningCoordinate`.

Owns: the inverse map from each physical codec index to its semantic parent
or child coordinate, and the exact index law for that map.

Does not own: verifier acceptance, output equations, application data,
physical R1CS columns, Rust, or artifacts.

Emits constraints: none.
-/

set_option autoImplicit false
set_option maxRecDepth 100000

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsCanonicalRunningCoverage

open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsParameters
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsCanonicalCodecCore
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsCanonicalRunningCodec
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsCanonicalViews
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsCarrierViews
open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier

private def componentAt : Fin 2 → KComponent :=
  Fin.cases .c0 (fun _ => .c1)

@[simp] private theorem componentAt_zero :
    componentAt 0 = .c0 := by
  rfl

@[simp] private theorem componentAt_one :
    componentAt 1 = .c1 := by
  rfl

@[simp] private theorem componentAt_succ (tail : Fin 1) :
    componentAt tail.succ = .c1 := by
  rfl

private theorem componentAt_view_index
    (coordinate : Fin 2) :
    ((componentAt coordinate).view kView).index.val = coordinate.val := by
  refine Fin.cases ?_ (fun tail => ?_) coordinate
  · rfl
  · have tailZero : tail = 0 := Subsingleton.elim _ _
    subst tail
    rfl

private theorem component_view_index
    {α : Type}
    {codec : Codec α}
    {source : α → K}
    (component : KComponent)
    (view : PaperNifsCodecProjection.KView codec source) :
    (component.view view).index.val =
      match component with
      | .c0 => view.c0Index.val
      | .c1 => view.c1Index.val := by
  cases component <;> rfl

private theorem runningCoordinate_view_index_congr
    {shape : SemanticShape}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    {codec :
      Codec
        (SelectedRunning
          shape publicRingColumns publicFits verifierRows)}
    (views : RunningViews codec)
    {left right :
      RunningCoordinate shape publicRingColumns verifierRows}
    (equal : left = right) :
    (left.view views).index.val = (right.view views).index.val :=
  congrArg (fun semantic => (semantic.view views).index.val) equal

private noncomputable def parentCoordinateAt
    (shape : SemanticShape)
    (publicRingColumns verifierRows : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth)
    (index :
      Fin
        (parentPayloadCodec
          shape publicRingColumns verifierRows publicFits).width) :
    RunningCoordinate shape publicRingColumns verifierRows :=
  Fin.addCases
    (fun coordinate =>
      let physical :=
        Fin.cast (commitmentCodec_width verifierRows) coordinate
      .parentCommitment physical.divNat physical.modNat)
    (Fin.addCases
      (fun coordinate =>
        .parentPublic
        (Fin.cast
          (publicInputCodec_width
            (ringDegree * publicRingColumns)) coordinate))
      (Fin.addCases
        (fun coordinate =>
          let physical :=
            Fin.cast (pointCodec_width shape.rowVariables) coordinate
          .parentPoint physical.divNat
            (componentAt physical.modNat))
        (fun coordinate =>
          let physical :=
            Fin.cast (evaluationsCodec_width shape.matrixCount) coordinate
          .parentEvaluation physical.divNat
            physical.modNat.divNat
            (componentAt physical.modNat.modNat))))
    index

@[simp] private theorem parentCoordinateAt_commitment
    (shape : SemanticShape)
    (publicRingColumns verifierRows : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth)
    (coordinate : Fin (commitmentCodec verifierRows).width) :
    parentCoordinateAt shape publicRingColumns verifierRows publicFits
        (Fin.castAdd
          (Codec.product
            (publicInputCodec (ringDegree * publicRingColumns))
            (Codec.product
              (pointCodec shape.rowVariables)
              (evaluationsCodec shape.matrixCount))).width
          coordinate) =
      let physical :=
        Fin.cast (commitmentCodec_width verifierRows) coordinate
      .parentCommitment physical.divNat physical.modNat := by
  unfold parentCoordinateAt
  exact Fin.addCases_left coordinate

@[simp] private theorem parentCoordinateAt_public
    (shape : SemanticShape)
    (publicRingColumns verifierRows : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth)
    (coordinate :
      Fin (publicInputCodec
        (ringDegree * publicRingColumns)).width) :
    parentCoordinateAt shape publicRingColumns verifierRows publicFits
        (Fin.natAdd (commitmentCodec verifierRows).width
          (Fin.castAdd
            (Codec.product
              (pointCodec shape.rowVariables)
              (evaluationsCodec shape.matrixCount)).width
            coordinate)) =
      .parentPublic
        (Fin.cast
          (publicInputCodec_width
            (ringDegree * publicRingColumns)) coordinate) := by
  unfold parentCoordinateAt
  rw [Fin.addCases_right]
  exact Fin.addCases_left coordinate

@[simp] private theorem parentCoordinateAt_point
    (shape : SemanticShape)
    (publicRingColumns verifierRows : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth)
    (coordinate : Fin (pointCodec shape.rowVariables).width) :
    parentCoordinateAt shape publicRingColumns verifierRows publicFits
        (Fin.natAdd (commitmentCodec verifierRows).width
          (Fin.natAdd
            (publicInputCodec
              (ringDegree * publicRingColumns)).width
            (Fin.castAdd
              (evaluationsCodec shape.matrixCount).width coordinate))) =
      let physical :=
        Fin.cast (pointCodec_width shape.rowVariables) coordinate
      .parentPoint physical.divNat (componentAt physical.modNat) := by
  unfold parentCoordinateAt
  rw [Fin.addCases_right, Fin.addCases_right, Fin.addCases_left]

@[simp] private theorem parentCoordinateAt_evaluation
    (shape : SemanticShape)
    (publicRingColumns verifierRows : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth)
    (coordinate : Fin (evaluationsCodec shape.matrixCount).width) :
    parentCoordinateAt shape publicRingColumns verifierRows publicFits
        (Fin.natAdd (commitmentCodec verifierRows).width
          (Fin.natAdd
            (publicInputCodec
              (ringDegree * publicRingColumns)).width
            (Fin.natAdd
              (pointCodec shape.rowVariables).width coordinate))) =
      let physical :=
        Fin.cast (evaluationsCodec_width shape.matrixCount) coordinate
      .parentEvaluation physical.divNat physical.modNat.divNat
        (componentAt physical.modNat.modNat) := by
  unfold parentCoordinateAt
  rw [Fin.addCases_right, Fin.addCases_right, Fin.addCases_right]

private noncomputable def childCoordinateAt
    (shape : SemanticShape)
    (publicRingColumns verifierRows : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth)
    (child : Fin productionGlobalParams.k)
    (index :
      Fin
        (runningPayloadCodec
          shape publicRingColumns verifierRows publicFits).width) :
    RunningCoordinate shape publicRingColumns verifierRows :=
  Fin.addCases
    (fun coordinate =>
      let physical :=
        Fin.cast (commitmentCodec_width verifierRows) coordinate
      .childCommitment child physical.divNat physical.modNat)
    (Fin.addCases
      (fun coordinate =>
        .childPublic child
        (Fin.cast
          (publicInputCodec_width
            (ringDegree * publicRingColumns)) coordinate))
      (Fin.addCases
        (fun coordinate =>
          let physical :=
            Fin.cast (pointCodec_width shape.rowVariables) coordinate
          .childPoint child physical.divNat
            (componentAt physical.modNat))
        (fun coordinate =>
          let physical :=
            Fin.cast (evaluationsCodec_width shape.matrixCount) coordinate
          .childEvaluation child physical.divNat
            physical.modNat.divNat
            (componentAt physical.modNat.modNat))))
    index

@[simp] private theorem childCoordinateAt_commitment
    (shape : SemanticShape)
    (publicRingColumns verifierRows : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth)
    (child : Fin productionGlobalParams.k)
    (coordinate : Fin (commitmentCodec verifierRows).width) :
    childCoordinateAt
        shape publicRingColumns verifierRows publicFits child
        (Fin.castAdd
          (Codec.product
            (publicInputCodec (ringDegree * publicRingColumns))
            (Codec.product
              (pointCodec shape.rowVariables)
              (evaluationsCodec shape.matrixCount))).width
          coordinate) =
      let physical :=
        Fin.cast (commitmentCodec_width verifierRows) coordinate
      .childCommitment child physical.divNat physical.modNat := by
  unfold childCoordinateAt
  exact Fin.addCases_left coordinate

@[simp] private theorem childCoordinateAt_public
    (shape : SemanticShape)
    (publicRingColumns verifierRows : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth)
    (child : Fin productionGlobalParams.k)
    (coordinate :
      Fin (publicInputCodec
        (ringDegree * publicRingColumns)).width) :
    childCoordinateAt
        shape publicRingColumns verifierRows publicFits child
        (Fin.natAdd (commitmentCodec verifierRows).width
          (Fin.castAdd
            (Codec.product
              (pointCodec shape.rowVariables)
              (evaluationsCodec shape.matrixCount)).width
            coordinate)) =
      .childPublic child
        (Fin.cast
          (publicInputCodec_width
            (ringDegree * publicRingColumns)) coordinate) := by
  unfold childCoordinateAt
  rw [Fin.addCases_right]
  exact Fin.addCases_left coordinate

@[simp] private theorem childCoordinateAt_point
    (shape : SemanticShape)
    (publicRingColumns verifierRows : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth)
    (child : Fin productionGlobalParams.k)
    (coordinate : Fin (pointCodec shape.rowVariables).width) :
    childCoordinateAt
        shape publicRingColumns verifierRows publicFits child
        (Fin.natAdd (commitmentCodec verifierRows).width
          (Fin.natAdd
            (publicInputCodec
              (ringDegree * publicRingColumns)).width
            (Fin.castAdd
              (evaluationsCodec shape.matrixCount).width coordinate))) =
      let physical :=
        Fin.cast (pointCodec_width shape.rowVariables) coordinate
      .childPoint child physical.divNat
        (componentAt physical.modNat) := by
  unfold childCoordinateAt
  rw [Fin.addCases_right, Fin.addCases_right, Fin.addCases_left]

@[simp] private theorem childCoordinateAt_evaluation
    (shape : SemanticShape)
    (publicRingColumns verifierRows : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth)
    (child : Fin productionGlobalParams.k)
    (coordinate : Fin (evaluationsCodec shape.matrixCount).width) :
    childCoordinateAt
        shape publicRingColumns verifierRows publicFits child
        (Fin.natAdd (commitmentCodec verifierRows).width
          (Fin.natAdd
            (publicInputCodec
              (ringDegree * publicRingColumns)).width
            (Fin.natAdd
              (pointCodec shape.rowVariables).width coordinate))) =
      let physical :=
        Fin.cast (evaluationsCodec_width shape.matrixCount) coordinate
      .childEvaluation child physical.divNat physical.modNat.divNat
        (componentAt physical.modNat.modNat) := by
  unfold childCoordinateAt
  rw [Fin.addCases_right, Fin.addCases_right, Fin.addCases_right]

private theorem parentPoint_component_index
    (shape : SemanticShape)
    (publicRingColumns verifierRows : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth)
    (coordinate : Fin shape.rowVariables)
    (component : Fin 2) :
    ((componentAt component).view
        ((runningViews
          shape publicRingColumns verifierRows publicFits).parentPoint
          coordinate)).index.val =
      (commitmentCodec verifierRows).width +
        (publicInputCodec
          (ringDegree * publicRingColumns)).width +
        coordinate.val * 2 + component.val := by
  refine Fin.cases ?_ (fun tail => ?_) component
  · rw [component_view_index]
    simp [componentAt_zero, runningViews,
      parentPointView, completePointView, pointView,
      PaperNifsCodecProjection.KView.throughPullback,
      PaperNifsCodecProjection.KView.productLeft,
      PaperNifsCodecProjection.KView.productRight,
      PaperNifsCodecProjection.KView.fixedListElement,
      PaperNifsCodecProjection.KView.congrValue,
      kView, PaperNifsCodecProjection.FView.throughPullback,
      PaperNifsCodecProjection.FView.productLeft,
      fieldView, fieldCodec, Nat.add_assoc]
  · have tailZero : tail = 0 := Subsingleton.elim _ _
    subst tail
    rw [component_view_index]
    simp [componentAt_succ, runningViews,
      parentPointView, completePointView, pointView,
      PaperNifsCodecProjection.KView.throughPullback,
      PaperNifsCodecProjection.KView.productLeft,
      PaperNifsCodecProjection.KView.productRight,
      PaperNifsCodecProjection.KView.fixedListElement,
      PaperNifsCodecProjection.KView.congrValue,
      kView, PaperNifsCodecProjection.FView.throughPullback,
      PaperNifsCodecProjection.FView.productRight,
      fieldView, fieldCodec, Nat.add_assoc]

private theorem parentEvaluation_component_index
    (shape : SemanticShape)
    (publicRingColumns verifierRows : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth)
    (matrix : Fin shape.matrixCount)
    (lane : Fin ringDegree)
    (component : Fin 2) :
    ((componentAt component).view
        ((runningViews
          shape publicRingColumns verifierRows publicFits).parentEvaluation
          matrix lane)).index.val =
      (commitmentCodec verifierRows).width +
        (publicInputCodec
          (ringDegree * publicRingColumns)).width +
        (pointCodec shape.rowVariables).width +
        matrix.val * (ringDegree * 2) +
        lane.val * 2 + component.val := by
  refine Fin.cases ?_ (fun tail => ?_) component
  · rw [component_view_index]
    simp [componentAt_zero, runningViews,
      parentEvaluationView, completeEvaluationView, evaluationView,
      ringKView,
      PaperNifsCodecProjection.KView.throughPullback,
      PaperNifsCodecProjection.KView.productLeft,
      PaperNifsCodecProjection.KView.productRight,
      PaperNifsCodecProjection.KView.fixedArrayElement,
      PaperNifsCodecProjection.KView.finElement,
      kView, PaperNifsCodecProjection.FView.throughPullback,
      PaperNifsCodecProjection.FView.productLeft,
      fieldView, fieldCodec, Nat.add_assoc]
  · have tailZero : tail = 0 := Subsingleton.elim _ _
    subst tail
    rw [component_view_index]
    simp [componentAt_succ, runningViews,
      parentEvaluationView, completeEvaluationView, evaluationView,
      ringKView,
      PaperNifsCodecProjection.KView.throughPullback,
      PaperNifsCodecProjection.KView.productLeft,
      PaperNifsCodecProjection.KView.productRight,
      PaperNifsCodecProjection.KView.fixedArrayElement,
      PaperNifsCodecProjection.KView.finElement,
      kView, PaperNifsCodecProjection.FView.throughPullback,
      PaperNifsCodecProjection.FView.productRight,
      fieldView, fieldCodec, Nat.add_assoc]

private theorem childPoint_component_index
    (shape : SemanticShape)
    (publicRingColumns verifierRows : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth)
    (child : Fin productionGlobalParams.k)
    (coordinate : Fin shape.rowVariables)
    (component : Fin 2) :
    ((componentAt component).view
        ((runningViews
          shape publicRingColumns verifierRows publicFits).childPoint
          child coordinate)).index.val =
      (parentPayloadCodec
          shape publicRingColumns verifierRows publicFits).width +
        child.val *
          (runningPayloadCodec
            shape publicRingColumns verifierRows publicFits).width +
        (commitmentCodec verifierRows).width +
        (publicInputCodec
          (ringDegree * publicRingColumns)).width +
        coordinate.val * 2 + component.val := by
  refine Fin.cases ?_ (fun tail => ?_) component
  · rw [component_view_index]
    simp [componentAt_zero, runningViews,
      childPointPayloadView, completePointView, pointView,
      PaperNifsCodecProjection.KView.throughPullback,
      PaperNifsCodecProjection.KView.productLeft,
      PaperNifsCodecProjection.KView.productRight,
      PaperNifsCodecProjection.KView.fixedListElement,
      PaperNifsCodecProjection.KView.finElement,
      PaperNifsCodecProjection.KView.congrValue,
      kView, PaperNifsCodecProjection.FView.throughPullback,
      PaperNifsCodecProjection.FView.productLeft,
      fieldView, fieldCodec, Nat.add_assoc]
  · have tailZero : tail = 0 := Subsingleton.elim _ _
    subst tail
    rw [component_view_index]
    simp [componentAt_succ, runningViews,
      childPointPayloadView, completePointView, pointView,
      PaperNifsCodecProjection.KView.throughPullback,
      PaperNifsCodecProjection.KView.productLeft,
      PaperNifsCodecProjection.KView.productRight,
      PaperNifsCodecProjection.KView.fixedListElement,
      PaperNifsCodecProjection.KView.finElement,
      PaperNifsCodecProjection.KView.congrValue,
      kView, PaperNifsCodecProjection.FView.throughPullback,
      PaperNifsCodecProjection.FView.productRight,
      fieldView, fieldCodec, Nat.add_assoc]

private theorem childEvaluation_component_index
    (shape : SemanticShape)
    (publicRingColumns verifierRows : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth)
    (child : Fin productionGlobalParams.k)
    (matrix : Fin shape.matrixCount)
    (lane : Fin ringDegree)
    (component : Fin 2) :
    ((componentAt component).view
        ((runningViews
          shape publicRingColumns verifierRows publicFits).childEvaluation
          child matrix lane)).index.val =
      (parentPayloadCodec
          shape publicRingColumns verifierRows publicFits).width +
        child.val *
          (runningPayloadCodec
            shape publicRingColumns verifierRows publicFits).width +
        (commitmentCodec verifierRows).width +
        (publicInputCodec
          (ringDegree * publicRingColumns)).width +
        (pointCodec shape.rowVariables).width +
        matrix.val * (ringDegree * 2) +
        lane.val * 2 + component.val := by
  refine Fin.cases ?_ (fun tail => ?_) component
  · rw [component_view_index]
    simp [componentAt_zero, runningViews,
      childEvaluationPayloadView, completeEvaluationView,
      evaluationView, ringKView,
      PaperNifsCodecProjection.KView.throughPullback,
      PaperNifsCodecProjection.KView.productLeft,
      PaperNifsCodecProjection.KView.productRight,
      PaperNifsCodecProjection.KView.fixedArrayElement,
      PaperNifsCodecProjection.KView.finElement,
      kView, PaperNifsCodecProjection.FView.throughPullback,
      PaperNifsCodecProjection.FView.productLeft,
      fieldView, fieldCodec, Nat.add_assoc]
  · have tailZero : tail = 0 := Subsingleton.elim _ _
    subst tail
    rw [component_view_index]
    simp [componentAt_succ, runningViews,
      childEvaluationPayloadView, completeEvaluationView,
      evaluationView, ringKView,
      PaperNifsCodecProjection.KView.throughPullback,
      PaperNifsCodecProjection.KView.productLeft,
      PaperNifsCodecProjection.KView.productRight,
      PaperNifsCodecProjection.KView.fixedArrayElement,
      PaperNifsCodecProjection.KView.finElement,
      kView, PaperNifsCodecProjection.FView.throughPullback,
      PaperNifsCodecProjection.FView.productRight,
      fieldView, fieldCodec, Nat.add_assoc]

private theorem parentCoordinateAt_index
    (shape : SemanticShape)
    (publicRingColumns verifierRows : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth)
    (index :
      Fin
        (parentPayloadCodec
          shape publicRingColumns verifierRows publicFits).width) :
    ((parentCoordinateAt
        shape publicRingColumns verifierRows publicFits index).view
      (runningViews
        shape publicRingColumns verifierRows publicFits)).index.val =
      index.val := by
  change
    Fin
      ((commitmentCodec verifierRows).width +
        ((publicInputCodec (ringDegree * publicRingColumns)).width +
          ((pointCodec shape.rowVariables).width +
            (evaluationsCodec shape.matrixCount).width)))
    at index
  refine Fin.addCases (motive := fun coordinate =>
    ((parentCoordinateAt
        shape publicRingColumns verifierRows publicFits coordinate).view
      (runningViews
        shape publicRingColumns verifierRows publicFits)).index.val =
      coordinate.val) ?_ ?_ index
  · intro coordinate
    let physical :=
      Fin.cast (commitmentCodec_width verifierRows) coordinate
    calc
      _ =
          ((RunningCoordinate.parentCommitment
              physical.divNat physical.modNat).view
            (runningViews
              shape publicRingColumns verifierRows publicFits)).index.val := by
            exact congrArg
              (fun semantic :
                  RunningCoordinate
                    shape publicRingColumns verifierRows =>
                (semantic.view
                  (runningViews
                    shape publicRingColumns verifierRows publicFits)).index.val)
              (parentCoordinateAt_commitment
                shape publicRingColumns verifierRows publicFits coordinate)
      _ = coordinate.val := by
        simp only [RunningCoordinate.view, runningViews,
          parentCommitmentView, completeCommitmentView, commitmentView,
          PaperNifsCodecProjection.FView.throughPullback,
          PaperNifsCodecProjection.FView.productLeft,
          PaperNifsCodecProjection.FView.productRight,
          PaperNifsCodecProjection.FView.finElement, ringFView, fieldView]
        simpa [physical, fieldCodec, Nat.mul_comm] using
          (Nat.div_add_mod physical.val ringDegree)
  · intro afterCommitment
    refine Fin.addCases (motive := fun coordinate =>
      ((parentCoordinateAt
          shape publicRingColumns verifierRows publicFits
          (Fin.natAdd (commitmentCodec verifierRows).width coordinate)).view
        (runningViews
          shape publicRingColumns verifierRows publicFits)).index.val =
        (Fin.natAdd
          (commitmentCodec verifierRows).width coordinate).val) ?_ ?_
      afterCommitment
    · intro coordinate
      let physical :=
        Fin.cast
          (publicInputCodec_width
            (ringDegree * publicRingColumns)) coordinate
      calc
        _ =
            ((RunningCoordinate.parentPublic physical).view
              (runningViews
                shape publicRingColumns verifierRows publicFits)).index.val := by
              exact runningCoordinate_view_index_congr
                (runningViews
                  shape publicRingColumns verifierRows publicFits)
                (parentCoordinateAt_public
                  shape publicRingColumns verifierRows publicFits coordinate)
        _ =
            (Fin.natAdd (commitmentCodec verifierRows).width
              (Fin.castAdd
                (Codec.product
                  (pointCodec shape.rowVariables)
                  (evaluationsCodec shape.matrixCount)).width
                coordinate)).val := by
              simp [RunningCoordinate.view, runningViews,
                parentPublicView, completePublicView, publicInputView,
                PaperNifsCodecProjection.FView.throughPullback,
                PaperNifsCodecProjection.FView.productLeft,
                PaperNifsCodecProjection.FView.productRight,
                PaperNifsCodecProjection.FView.finElement, fieldView,
                fieldCodec, physical]
    · intro afterPublic
      refine Fin.addCases (motive := fun coordinate =>
        ((parentCoordinateAt
            shape publicRingColumns verifierRows publicFits
            (Fin.natAdd (commitmentCodec verifierRows).width
              (Fin.natAdd
                (publicInputCodec
                  (ringDegree * publicRingColumns)).width coordinate))).view
          (runningViews
            shape publicRingColumns verifierRows publicFits)).index.val =
          (Fin.natAdd (commitmentCodec verifierRows).width
            (Fin.natAdd
              (publicInputCodec
                (ringDegree * publicRingColumns)).width coordinate)).val)
        ?_ ?_ afterPublic
      · intro coordinate
        let physical :=
          Fin.cast (pointCodec_width shape.rowVariables) coordinate
        calc
          _ =
              ((RunningCoordinate.parentPoint
                  physical.divNat (componentAt physical.modNat)).view
                (runningViews
                  shape publicRingColumns verifierRows publicFits)).index.val := by
                exact runningCoordinate_view_index_congr
                  (runningViews
                    shape publicRingColumns verifierRows publicFits)
                  (parentCoordinateAt_point
                    shape publicRingColumns verifierRows publicFits coordinate)
          _ =
              (Fin.natAdd (commitmentCodec verifierRows).width
                (Fin.natAdd
                  (publicInputCodec
                    (ringDegree * publicRingColumns)).width
                  (Fin.castAdd
                    (evaluationsCodec shape.matrixCount).width
                    coordinate))).val := by
                calc
                  _ =
                      (commitmentCodec verifierRows).width +
                        (publicInputCodec
                          (ringDegree * publicRingColumns)).width +
                        physical.divNat.val * 2 +
                        physical.modNat.val :=
                    parentPoint_component_index
                      shape publicRingColumns verifierRows publicFits
                      physical.divNat physical.modNat
                  _ = _ := by
                    simp only [Fin.val_natAdd]
                    simpa [physical, Nat.mul_comm, Nat.add_assoc] using
                      (Nat.div_add_mod physical.val 2)
      · intro coordinate
        let physical :=
          Fin.cast (evaluationsCodec_width shape.matrixCount) coordinate
        calc
          _ =
              ((RunningCoordinate.parentEvaluation
                  physical.divNat physical.modNat.divNat
                  (componentAt physical.modNat.modNat)).view
                (runningViews
                  shape publicRingColumns verifierRows publicFits)).index.val := by
                exact runningCoordinate_view_index_congr
                  (runningViews
                    shape publicRingColumns verifierRows publicFits)
                  (parentCoordinateAt_evaluation
                    shape publicRingColumns verifierRows publicFits coordinate)
          _ =
              (Fin.natAdd (commitmentCodec verifierRows).width
                (Fin.natAdd
                  (publicInputCodec
                    (ringDegree * publicRingColumns)).width
                  (Fin.natAdd
                    (pointCodec shape.rowVariables).width
                    coordinate))).val := by
                calc
                  _ =
                      (commitmentCodec verifierRows).width +
                        (publicInputCodec
                          (ringDegree * publicRingColumns)).width +
                        (pointCodec shape.rowVariables).width +
                        physical.divNat.val * (ringDegree * 2) +
                        physical.modNat.divNat.val * 2 +
                        physical.modNat.modNat.val :=
                    parentEvaluation_component_index
                      shape publicRingColumns verifierRows publicFits
                      physical.divNat physical.modNat.divNat
                      physical.modNat.modNat
                  _ = _ := by
                    simp only [Fin.val_natAdd]
                    have outer :
                        physical.divNat.val * (ringDegree * 2) +
                            physical.modNat.val =
                          physical.val := by
                      simpa [Nat.mul_comm] using
                        (Nat.div_add_mod
                          physical.val (ringDegree * 2))
                    have inner :
                        physical.modNat.divNat.val * 2 +
                            physical.modNat.modNat.val =
                          physical.modNat.val := by
                      simpa [Nat.mul_comm] using
                        (Nat.div_add_mod physical.modNat.val 2)
                    simp [physical] at outer inner ⊢
                    omega

private theorem childCoordinateAt_index
    (shape : SemanticShape)
    (publicRingColumns verifierRows : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth)
    (child : Fin productionGlobalParams.k)
    (index :
      Fin
        (runningPayloadCodec
          shape publicRingColumns verifierRows publicFits).width) :
    ((childCoordinateAt
        shape publicRingColumns verifierRows publicFits child index).view
      (runningViews
        shape publicRingColumns verifierRows publicFits)).index.val =
      (parentPayloadCodec
          shape publicRingColumns verifierRows publicFits).width +
        child.val *
          (runningPayloadCodec
            shape publicRingColumns verifierRows publicFits).width +
        index.val := by
  change
    Fin
      ((commitmentCodec verifierRows).width +
        ((publicInputCodec (ringDegree * publicRingColumns)).width +
          ((pointCodec shape.rowVariables).width +
            (evaluationsCodec shape.matrixCount).width)))
    at index
  refine Fin.addCases (motive := fun coordinate =>
    ((childCoordinateAt
        shape publicRingColumns verifierRows publicFits child coordinate).view
      (runningViews
        shape publicRingColumns verifierRows publicFits)).index.val =
      (parentPayloadCodec
          shape publicRingColumns verifierRows publicFits).width +
        child.val *
          (runningPayloadCodec
            shape publicRingColumns verifierRows publicFits).width +
        coordinate.val) ?_ ?_ index
  · intro coordinate
    let physical :=
      Fin.cast (commitmentCodec_width verifierRows) coordinate
    calc
      _ =
          ((RunningCoordinate.childCommitment child
              physical.divNat physical.modNat).view
            (runningViews
              shape publicRingColumns verifierRows publicFits)).index.val := by
            exact runningCoordinate_view_index_congr
              (runningViews
                shape publicRingColumns verifierRows publicFits)
              (childCoordinateAt_commitment
                shape publicRingColumns verifierRows publicFits
                child coordinate)
      _ =
          (parentPayloadCodec
              shape publicRingColumns verifierRows publicFits).width +
            child.val *
              (runningPayloadCodec
                shape publicRingColumns verifierRows publicFits).width +
            coordinate.val := by
        simp only [RunningCoordinate.view, runningViews,
          childCommitmentPayloadView, completeCommitmentView,
          commitmentView,
          PaperNifsCodecProjection.FView.throughPullback,
          PaperNifsCodecProjection.FView.productLeft,
          PaperNifsCodecProjection.FView.productRight,
          PaperNifsCodecProjection.FView.finElement,
          ringFView, fieldView]
        have decomposition :
            physical.divNat.val * ringDegree +
                physical.modNat.val =
              physical.val := by
          simpa [Nat.mul_comm] using
            (Nat.div_add_mod physical.val ringDegree)
        simp [physical, fieldCodec] at decomposition ⊢
        omega
  · intro afterCommitment
    refine Fin.addCases (motive := fun coordinate =>
      ((childCoordinateAt
          shape publicRingColumns verifierRows publicFits child
          (Fin.natAdd (commitmentCodec verifierRows).width coordinate)).view
        (runningViews
          shape publicRingColumns verifierRows publicFits)).index.val =
        (parentPayloadCodec
            shape publicRingColumns verifierRows publicFits).width +
          child.val *
            (runningPayloadCodec
              shape publicRingColumns verifierRows publicFits).width +
          (Fin.natAdd
            (commitmentCodec verifierRows).width coordinate).val) ?_ ?_
      afterCommitment
    · intro coordinate
      let physical :=
        Fin.cast
          (publicInputCodec_width
            (ringDegree * publicRingColumns)) coordinate
      calc
        _ =
            ((RunningCoordinate.childPublic child physical).view
              (runningViews
                shape publicRingColumns verifierRows publicFits)).index.val := by
              exact runningCoordinate_view_index_congr
                (runningViews
                  shape publicRingColumns verifierRows publicFits)
                (childCoordinateAt_public
                  shape publicRingColumns verifierRows publicFits
                  child coordinate)
        _ =
            (parentPayloadCodec
                shape publicRingColumns verifierRows publicFits).width +
              child.val *
                (runningPayloadCodec
                  shape publicRingColumns verifierRows publicFits).width +
              (Fin.natAdd (commitmentCodec verifierRows).width
                (Fin.castAdd
                  (Codec.product
                    (pointCodec shape.rowVariables)
                    (evaluationsCodec shape.matrixCount)).width
                  coordinate)).val := by
            simp [RunningCoordinate.view, runningViews,
              childPublicPayloadView, completePublicView, publicInputView,
              PaperNifsCodecProjection.FView.throughPullback,
              PaperNifsCodecProjection.FView.productLeft,
              PaperNifsCodecProjection.FView.productRight,
              PaperNifsCodecProjection.FView.finElement,
              fieldView, fieldCodec, physical, Nat.add_assoc]
    · intro afterPublic
      refine Fin.addCases (motive := fun coordinate =>
        ((childCoordinateAt
            shape publicRingColumns verifierRows publicFits child
            (Fin.natAdd (commitmentCodec verifierRows).width
              (Fin.natAdd
                (publicInputCodec
                  (ringDegree * publicRingColumns)).width coordinate))).view
          (runningViews
            shape publicRingColumns verifierRows publicFits)).index.val =
          (parentPayloadCodec
              shape publicRingColumns verifierRows publicFits).width +
            child.val *
              (runningPayloadCodec
                shape publicRingColumns verifierRows publicFits).width +
            (Fin.natAdd (commitmentCodec verifierRows).width
              (Fin.natAdd
                (publicInputCodec
                  (ringDegree * publicRingColumns)).width coordinate)).val)
        ?_ ?_ afterPublic
      · intro coordinate
        let physical :=
          Fin.cast (pointCodec_width shape.rowVariables) coordinate
        calc
          _ =
              ((RunningCoordinate.childPoint child
                  physical.divNat (componentAt physical.modNat)).view
                (runningViews
                  shape publicRingColumns verifierRows publicFits)).index.val := by
                exact runningCoordinate_view_index_congr
                  (runningViews
                    shape publicRingColumns verifierRows publicFits)
                  (childCoordinateAt_point
                    shape publicRingColumns verifierRows publicFits
                    child coordinate)
          _ =
              (parentPayloadCodec
                  shape publicRingColumns verifierRows publicFits).width +
                child.val *
                  (runningPayloadCodec
                    shape publicRingColumns verifierRows publicFits).width +
                (Fin.natAdd (commitmentCodec verifierRows).width
                  (Fin.natAdd
                    (publicInputCodec
                      (ringDegree * publicRingColumns)).width
                    (Fin.castAdd
                      (evaluationsCodec shape.matrixCount).width
                      coordinate))).val := by
                calc
                  _ =
                      (parentPayloadCodec
                          shape publicRingColumns verifierRows publicFits).width +
                        child.val *
                          (runningPayloadCodec
                            shape publicRingColumns verifierRows
                              publicFits).width +
                        (commitmentCodec verifierRows).width +
                        (publicInputCodec
                          (ringDegree * publicRingColumns)).width +
                        physical.divNat.val * 2 +
                        physical.modNat.val :=
                    childPoint_component_index
                      shape publicRingColumns verifierRows publicFits
                      child physical.divNat physical.modNat
                  _ = _ := by
                    simp only [Fin.val_natAdd]
                    simpa [physical, Nat.mul_comm, Nat.add_assoc] using
                      (Nat.div_add_mod physical.val 2)
      · intro coordinate
        let physical :=
          Fin.cast (evaluationsCodec_width shape.matrixCount) coordinate
        calc
          _ =
              ((RunningCoordinate.childEvaluation child
                  physical.divNat physical.modNat.divNat
                  (componentAt physical.modNat.modNat)).view
                (runningViews
                  shape publicRingColumns verifierRows publicFits)).index.val := by
                exact runningCoordinate_view_index_congr
                  (runningViews
                    shape publicRingColumns verifierRows publicFits)
                  (childCoordinateAt_evaluation
                    shape publicRingColumns verifierRows publicFits
                    child coordinate)
          _ =
              (parentPayloadCodec
                  shape publicRingColumns verifierRows publicFits).width +
                child.val *
                  (runningPayloadCodec
                    shape publicRingColumns verifierRows publicFits).width +
                (Fin.natAdd (commitmentCodec verifierRows).width
                  (Fin.natAdd
                    (publicInputCodec
                      (ringDegree * publicRingColumns)).width
                    (Fin.natAdd
                      (pointCodec shape.rowVariables).width
                      coordinate))).val := by
                calc
                  _ =
                      (parentPayloadCodec
                          shape publicRingColumns verifierRows publicFits).width +
                        child.val *
                          (runningPayloadCodec
                            shape publicRingColumns verifierRows
                              publicFits).width +
                        (commitmentCodec verifierRows).width +
                        (publicInputCodec
                          (ringDegree * publicRingColumns)).width +
                        (pointCodec shape.rowVariables).width +
                        physical.divNat.val * (ringDegree * 2) +
                        physical.modNat.divNat.val * 2 +
                        physical.modNat.modNat.val :=
                    childEvaluation_component_index
                      shape publicRingColumns verifierRows publicFits
                      child physical.divNat physical.modNat.divNat
                      physical.modNat.modNat
                  _ = _ := by
                    simp only [Fin.val_natAdd]
                    have outer :
                        physical.divNat.val * (ringDegree * 2) +
                            physical.modNat.val =
                          physical.val := by
                      simpa [Nat.mul_comm] using
                        (Nat.div_add_mod
                          physical.val (ringDegree * 2))
                    have inner :
                        physical.modNat.divNat.val * 2 +
                            physical.modNat.modNat.val =
                          physical.modNat.val := by
                      simpa [Nat.mul_comm] using
                        (Nat.div_add_mod physical.modNat.val 2)
                    simp [physical] at outer inner ⊢
                    omega

private noncomputable def coordinateAt
    (shape : SemanticShape)
    (publicRingColumns verifierRows : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth)
    (index :
      Fin
        (runningCodec
          shape publicRingColumns verifierRows publicFits).width) :
    RunningCoordinate shape publicRingColumns verifierRows :=
  Fin.addCases
    (parentCoordinateAt
      shape publicRingColumns verifierRows publicFits)
    (fun coordinate =>
      childCoordinateAt
        shape publicRingColumns verifierRows publicFits
        coordinate.divNat coordinate.modNat)
    index

@[simp] private theorem coordinateAt_parent
    (shape : SemanticShape)
    (publicRingColumns verifierRows : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth)
    (parent :
      Fin
        (parentPayloadCodec
          shape publicRingColumns verifierRows publicFits).width) :
    coordinateAt shape publicRingColumns verifierRows publicFits
        (Fin.castAdd
          (Codec.finFunction productionGlobalParams.k
            (runningPayloadCodec
              shape publicRingColumns verifierRows publicFits)).width
          parent) =
      parentCoordinateAt
        shape publicRingColumns verifierRows publicFits parent := by
  unfold coordinateAt
  exact Fin.addCases_left parent

@[simp] private theorem coordinateAt_child
    (shape : SemanticShape)
    (publicRingColumns verifierRows : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth)
    (children :
      Fin
        (Codec.finFunction productionGlobalParams.k
          (runningPayloadCodec
            shape publicRingColumns verifierRows publicFits)).width) :
    coordinateAt shape publicRingColumns verifierRows publicFits
        (Fin.natAdd
          (parentPayloadCodec
            shape publicRingColumns verifierRows publicFits).width
          children) =
      childCoordinateAt
        shape publicRingColumns verifierRows publicFits
        children.divNat children.modNat := by
  unfold coordinateAt
  exact Fin.addCases_right children

/-- Every physical running-codec index is one explicit semantic coordinate,
and verifier-owned size facts are exactly the remaining codec-domain
obligations. -/
noncomputable def coverage
    (shape : SemanticShape)
    (publicRingColumns verifierRows : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth) :
    RunningCodecCoverage
      (runningCodec shape publicRingColumns verifierRows publicFits)
      (runningViews shape publicRingColumns verifierRows publicFits) where
  coordinateAt :=
    coordinateAt shape publicRingColumns verifierRows publicFits
  indexExact := by
    intro index
    apply Fin.ext
    change
      Fin
        ((parentPayloadCodec
            shape publicRingColumns verifierRows publicFits).width +
          productionGlobalParams.k *
            (runningPayloadCodec
              shape publicRingColumns verifierRows publicFits).width)
      at index
    refine Fin.addCases (motive := fun physical =>
      ((coordinateAt
          shape publicRingColumns verifierRows publicFits physical).view
        (runningViews
          shape publicRingColumns verifierRows publicFits)).index.val =
        physical.val) ?_ ?_ index
    · intro parent
      calc
        _ =
            ((parentCoordinateAt
                shape publicRingColumns verifierRows publicFits parent).view
              (runningViews
                shape publicRingColumns verifierRows publicFits)).index.val := by
              exact runningCoordinate_view_index_congr
                (runningViews
                  shape publicRingColumns verifierRows publicFits)
                (coordinateAt_parent
                  shape publicRingColumns verifierRows publicFits parent)
        _ = parent.val :=
          parentCoordinateAt_index
            shape publicRingColumns verifierRows publicFits parent
        _ = _ := rfl
    · intro children
      calc
        _ =
            ((childCoordinateAt
                shape publicRingColumns verifierRows publicFits
                children.divNat children.modNat).view
              (runningViews
                shape publicRingColumns verifierRows publicFits)).index.val := by
              exact runningCoordinate_view_index_congr
                (runningViews
                  shape publicRingColumns verifierRows publicFits)
                (coordinateAt_child
                  shape publicRingColumns verifierRows publicFits children)
        _ =
            (parentPayloadCodec
                shape publicRingColumns verifierRows publicFits).width +
              children.divNat.val *
                (runningPayloadCodec
                  shape publicRingColumns verifierRows publicFits).width +
              children.modNat.val :=
          childCoordinateAt_index
            shape publicRingColumns verifierRows publicFits
            children.divNat children.modNat
        _ =
            (parentPayloadCodec
                shape publicRingColumns verifierRows publicFits).width +
              children.val := by
          have decomposition :
              children.divNat.val *
                    (runningPayloadCodec
                      shape publicRingColumns verifierRows publicFits).width +
                  children.modNat.val =
                children.val := by
            simpa [Nat.mul_comm] using
              (Nat.div_add_mod children.val
                (runningPayloadCodec
                  shape publicRingColumns verifierRows publicFits).width)
          omega
        _ = _ := rfl
  resultAdmissible := by
    intro result parentSize childSize
    exact
      (runningCodec_admissible_iff
        (SelectedRunning.ofResult result)).2
        ⟨parentSize, childSize⟩

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsCanonicalRunningCoverage
