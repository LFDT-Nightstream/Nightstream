import NightstreamFPrime.Layout.Stage1.RunningTransitionRecursiveBounds
import NightstreamFPrime.Layout.Stage1.RunningTransitionSourceSupportData
import NightstreamFPrime.Lifecycle.Stage1.RunningTransitionSupport

/-! Owns compact source support for the recursive running vector. -/

namespace NightstreamFPrime.Layout.Stage1.RunningTransitionSourceSupport

open NightstreamFPrime.Circuit
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Lifecycle.Stage1
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open RunningTransitionInputs

theorem recursiveSupported
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) :
    RunningTransition.RunningSupported
      (recursiveRunningExpr logicalWidth publicFits) Logical := by
  refine {
    point := ?_
    commitment := ?_
    publicInput := ?_
    eval_K := ?_
    eval_A := ?_ }
  · intro coordinate
    rw [recursivePoint_eq_direct coordinate]
    exact ⟨logical_roundPointC0 coordinate,
      logical_roundPointC1 coordinate⟩
  · intro source row coefficient
    simp only [recursiveRunningExpr, piDecInterface, PiDECInputs.interface,
      PiDECInputs.message, PiDECInputs.childCommitment, Expr.VarsSatisfy]
    apply logical_piDec
    exact Or.inl ⟨childOfRunning source, row, coefficient, rfl⟩
  · intro source column
    simp only [recursiveRunningExpr, piDecInterface, PiDECInputs.interface,
      PiDECInputs.childPublicInput, Expr.VarsSatisfy]
    apply logical_piDec
    let coordinate : Fin 270 :=
      ⟨column.val, by
        have bound := column.isLt
        norm_num [FullShape, fullShape, Phi81Relation.Shape.publicWidth,
          publicRingColumns, ringDegree] at bound ⊢
        exact bound⟩
    exact Or.inr (Or.inl ⟨childOfRunning source, coordinate, rfl⟩)
  · intro source coefficient
    simp only [recursiveRunningExpr, piDecInterface, PiDECInputs.interface,
      PiDECInputs.message, PiDECInputs.childEvalK, Expr.VarsSatisfy]
    constructor
    · apply logical_piDec
      exact Or.inr (Or.inr (Or.inl
        ⟨childOfRunning source, coefficient, Or.inl rfl⟩))
    · apply logical_piDec
      exact Or.inr (Or.inr (Or.inl
        ⟨childOfRunning source, coefficient, Or.inr rfl⟩))
  · intro source matrix coefficient
    simp only [recursiveRunningExpr, piDecInterface, PiDECInputs.interface,
      PiDECInputs.message, PiDECInputs.childEvalA, Expr.VarsSatisfy]
    constructor
    · apply logical_piDec
      exact Or.inr (Or.inr (Or.inr
        ⟨childOfRunning source, matrix, coefficient, Or.inl rfl⟩))
    · apply logical_piDec
      exact Or.inr (Or.inr (Or.inr
        ⟨childOfRunning source, matrix, coefficient, Or.inr rfl⟩))

end NightstreamFPrime.Layout.Stage1.RunningTransitionSourceSupport
