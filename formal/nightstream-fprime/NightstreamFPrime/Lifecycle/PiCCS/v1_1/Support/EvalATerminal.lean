import NightstreamFPrime.Gadgets.Multilinear.PointWeightedHornerSupport
import NightstreamFPrime.Lifecycle.PiCCS.v1_1.EvalATerminal

/-!
Owns variable-support propagation for the production PiCCS Eval-A terminal.
It changes no coordinate order, circuit, or row.
-/

namespace NightstreamFPrime.Lifecycle.PiCCS.v1_1.EvalATerminal

open NightstreamFPrime.Circuit
open NightstreamFPrime.Circuit.Quadratic
open NightstreamFPrime.Gadgets.Multilinear
open NightstreamFPrime.Gadgets.Polynomial
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint

private theorem productionCubeVariables_positive :
    0 < productionShape.cubeVariables := by
  norm_num [productionShape, Phi81MatrixSource.phi81Shape, cubeVariables]

private theorem coefficientExprs_supported (interface : Interface)
    (offset : Nat) (allowed : Nat → Prop)
    (outputEvalSupport : ∀ coordinate,
      Horner.KSupported (interface.outputEval_A offset coordinate) allowed) :
    ∀ coefficient ∈ coefficientExprs interface offset,
      Horner.KSupported coefficient allowed := by
  intro coefficient member
  rw [coefficientExprs, List.mem_map] at member
  rcases member with ⟨coordinate, _coordinateMember, rfl⟩
  exact outputEvalSupport coordinate

/-- Exact support propagation through the production Eval-A terminal. -/
theorem flatConstraints_varsSatisfy (interface : Interface) (offset : Nat)
    (allowed : Nat → Prop)
    (roundPointSupport : ∀ coordinate,
      Horner.KSupported (interface.roundPoint offset coordinate) allowed)
    (priorPointSupport : ∀ coordinate,
      Horner.KSupported (interface.priorPoint offset coordinate) allowed)
    (gammaSupport : Horner.KSupported (interface.gamma offset) allowed)
    (outputEvalSupport : ∀ coordinate,
      Horner.KSupported (interface.outputEval_A offset coordinate) allowed)
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

/-- The 14-matrix Eval-A terminal output preserves the exact row support. -/
theorem output_varsSatisfy (interface : Interface) (offset : Nat)
    (allowed : Nat → Prop)
    (roundPointSupport : ∀ coordinate,
      Horner.KSupported (interface.roundPoint offset coordinate) allowed)
    (priorPointSupport : ∀ coordinate,
      Horner.KSupported (interface.priorPoint offset coordinate) allowed)
    (gammaSupport : Horner.KSupported (interface.gamma offset) allowed)
    (outputEvalSupport : ∀ coordinate,
      Horner.KSupported (interface.outputEval_A offset coordinate) allowed)
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

end NightstreamFPrime.Lifecycle.PiCCS.v1_1.EvalATerminal
