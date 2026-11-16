import NightstreamFPrime.Gadgets.Multilinear.PointWeightedHornerSupport
import NightstreamFPrime.Lifecycle.PiCCS.v1_2.EvalKTerminal

namespace NightstreamFPrime.Lifecycle.PiCCS.v1_2.EvalKTerminal

open NightstreamFPrime.Circuit
open NightstreamFPrime.Circuit.Quadratic
open NightstreamFPrime.Gadgets.Multilinear
open NightstreamFPrime.Gadgets.Polynomial
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint

private theorem productionCubeVariables_positive :
    0 < productionShape.cubeVariables := by
  norm_num [productionShape, RingMatrixSource.ringShape, cubeVariables]

private theorem coefficientExprs_supported (interface : Interface)
    (offset : Nat) (allowed : Nat → Prop)
    (outputEvalSupport : ∀ coordinate,
      Horner.KSupported (interface.outputEval_K offset coordinate) allowed) :
    ∀ coefficient ∈ coefficientExprs interface offset,
      Horner.KSupported coefficient allowed := by
  intro coefficient member
  rw [coefficientExprs, List.mem_map] at member
  rcases member with ⟨coordinate, _coordinateMember, rfl⟩
  exact outputEvalSupport coordinate

theorem flatConstraints_varsSatisfy (interface : Interface) (offset : Nat)
    (allowed : Nat → Prop)
    (roundPointSupport : ∀ coordinate,
      Horner.KSupported (interface.roundPoint offset coordinate) allowed)
    (priorPointSupport : ∀ coordinate,
      Horner.KSupported (interface.priorPoint offset coordinate) allowed)
    (gammaSupport : Horner.KSupported (interface.gamma offset) allowed)
    (outputEvalSupport : ∀ coordinate,
      Horner.KSupported (interface.outputEval_K offset coordinate) allowed)
    (localSupport : ∀ index,
      offset ≤ index →
      index < offset + localLength
        (Circuit.ops (circuit interface).main offset) →
      allowed index) :
    ∀ expression ∈ flatConstraints
        (Circuit.ops (circuit interface).main offset),
      expression.VarsSatisfy allowed := by
  have supported := PointWeightedHorner.Owned.flatConstraints_varsSatisfy
    (coreInterface interface) productionCubeVariables_positive offset allowed
    (by intro coordinate; simpa [coreInterface] using roundPointSupport coordinate)
    (by intro coordinate; simpa [coreInterface] using priorPointSupport coordinate)
    (by simpa [coreInterface] using gammaSupport)
    (coefficientExprs_supported interface offset allowed outputEvalSupport)
    (by
      intro index lower upper
      apply localSupport index lower
      simpa [circuit] using upper)
  simpa [circuit] using supported

theorem output_varsSatisfy (interface : Interface) (offset : Nat)
    (allowed : Nat → Prop)
    (roundPointSupport : ∀ coordinate,
      Horner.KSupported (interface.roundPoint offset coordinate) allowed)
    (priorPointSupport : ∀ coordinate,
      Horner.KSupported (interface.priorPoint offset coordinate) allowed)
    (gammaSupport : Horner.KSupported (interface.gamma offset) allowed)
    (outputEvalSupport : ∀ coordinate,
      Horner.KSupported (interface.outputEval_K offset coordinate) allowed)
    (localSupport : ∀ index,
      offset ≤ index →
      index < offset + localLength
        (Circuit.ops (circuit interface).main offset) →
      allowed index) :
    Horner.KSupported (output interface offset) allowed := by
  have supported := PointWeightedHorner.Owned.output_varsSatisfy
    (coreInterface interface) productionCubeVariables_positive offset allowed
    (by intro coordinate; simpa [coreInterface] using roundPointSupport coordinate)
    (by intro coordinate; simpa [coreInterface] using priorPointSupport coordinate)
    (by simpa [coreInterface] using gammaSupport)
    (coefficientExprs_supported interface offset allowed outputEvalSupport)
    (by
      intro index lower upper
      apply localSupport index lower
      simpa [circuit] using upper)
  simpa [output, circuit] using supported

/-! ### The exporter's blocks (SPEC.md section 4.4) -/

theorem marks_nil (interface : Interface) (offset : Nat) :
    marks (Circuit.ops (circuit interface).main offset) = [] :=
  PointWeightedHorner.Owned.marks_nil (coreInterface interface)
    productionCubeVariables_positive offset

/-- The rows read both points, γ, the output evaluations and the gadget's own locals. -/
theorem free (interface : Interface) (offset : Nat) {live : Nat → Prop}
    (roundPoint : ∀ coordinate, Horner.KSupported (interface.roundPoint offset coordinate) live)
    (priorPoint : ∀ coordinate, Horner.KSupported (interface.priorPoint offset coordinate) live)
    (gamma : Horner.KSupported (interface.gamma offset) live)
    (outputEval : ∀ coordinate, Horner.KSupported (interface.outputEval_K offset coordinate) live)
    (locals : ∀ column, offset ≤ column →
      column < offset + localLength (Circuit.ops (circuit interface).main offset) → live column) :
    ∀ row ∈ flatConstraints (Circuit.ops (circuit interface).main offset), row.VarsSatisfy live :=
  flatConstraints_varsSatisfy interface offset live roundPoint priorPoint gamma outputEval locals

theorem output_live (interface : Interface) (offset : Nat) {live : Nat → Prop}
    (roundPoint : ∀ coordinate, Horner.KSupported (interface.roundPoint offset coordinate) live)
    (priorPoint : ∀ coordinate, Horner.KSupported (interface.priorPoint offset coordinate) live)
    (gamma : Horner.KSupported (interface.gamma offset) live)
    (outputEval : ∀ coordinate, Horner.KSupported (interface.outputEval_K offset coordinate) live)
    (locals : ∀ column, offset ≤ column →
      column < offset + localLength (Circuit.ops (circuit interface).main offset) → live column) :
    Horner.KSupported (output interface offset) live :=
  output_varsSatisfy interface offset live roundPoint priorPoint gamma outputEval locals

/-! The light step places the core gadget at its shape (`LightPiCCSLayout`). -/

section Shape

variable {shape : Shape}

theorem coefficientExprs_live (interface : ShapeInterface shape) (offset : Nat)
    {live : Nat → Prop}
    (outputEval : ∀ coordinate, Horner.KSupported (interface.outputEval_K offset coordinate) live) :
    ∀ coefficient ∈ coefficientExprs interface offset, Horner.KSupported coefficient live := by
  intro coefficient member
  obtain ⟨coordinate, _, rfl⟩ := List.mem_map.mp member
  exact outputEval coordinate

theorem shapeMarks_nil (interface : ShapeInterface shape) (positive : 0 < shape.cubeVariables)
    (offset : Nat) :
    marks (Circuit.ops
      (PointWeightedHorner.Owned.circuit (coreInterface interface) positive).main offset) = [] :=
  PointWeightedHorner.Owned.marks_nil (coreInterface interface) positive offset

/-- The rows read both points, γ, the output evaluations and the gadget's own locals. -/
theorem shapeFree (interface : ShapeInterface shape) (positive : 0 < shape.cubeVariables)
    (offset : Nat) {live : Nat → Prop}
    (roundPoint : ∀ coordinate, Horner.KSupported (interface.roundPoint offset coordinate) live)
    (priorPoint : ∀ coordinate, Horner.KSupported (interface.priorPoint offset coordinate) live)
    (gamma : Horner.KSupported (interface.gamma offset) live)
    (outputEval : ∀ coordinate, Horner.KSupported (interface.outputEval_K offset coordinate) live)
    (locals : ∀ column, offset ≤ column →
      column < offset + localLength (Circuit.ops
        (PointWeightedHorner.Owned.circuit (coreInterface interface) positive).main offset) →
        live column) :
    ∀ row ∈ flatConstraints (Circuit.ops
        (PointWeightedHorner.Owned.circuit (coreInterface interface) positive).main offset),
      row.VarsSatisfy live :=
  PointWeightedHorner.Owned.free (coreInterface interface) positive offset roundPoint priorPoint
    gamma (coefficientExprs_live interface offset outputEval) locals

theorem shapeOutput_live (interface : ShapeInterface shape) (positive : 0 < shape.cubeVariables)
    (offset : Nat) {live : Nat → Prop}
    (roundPoint : ∀ coordinate, Horner.KSupported (interface.roundPoint offset coordinate) live)
    (priorPoint : ∀ coordinate, Horner.KSupported (interface.priorPoint offset coordinate) live)
    (gamma : Horner.KSupported (interface.gamma offset) live)
    (outputEval : ∀ coordinate, Horner.KSupported (interface.outputEval_K offset coordinate) live)
    (locals : ∀ column, offset ≤ column →
      column < offset + localLength (Circuit.ops
        (PointWeightedHorner.Owned.circuit (coreInterface interface) positive).main offset) →
        live column) :
    Horner.KSupported (output interface offset) live :=
  PointWeightedHorner.Owned.output_live (coreInterface interface) positive offset roundPoint
    priorPoint gamma (coefficientExprs_live interface offset outputEval) locals

end Shape

end NightstreamFPrime.Lifecycle.PiCCS.v1_2.EvalKTerminal
