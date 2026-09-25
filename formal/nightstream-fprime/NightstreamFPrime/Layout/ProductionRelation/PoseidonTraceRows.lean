import NightstreamFPrime.Layout.ProductionRelation.PinRow
import NightstreamFPrime.Layout.ProductionRelation.SboxRow
import NightstreamFPrime.Layout.ProductionRelation.SourceCompiler

/-!
Owns the two direct row constructors used by the production Poseidon2 trace
rewrite: one seventh-power S-box equation and one retained linear-output
equation. Both consume the same proved source substitution.

This module does not select the fixed Poseidon2 schedule or invocation order.
-/

namespace NightstreamFPrime.Layout.ProductionRelation.PoseidonTraceRows

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra

/-- One collapsed Poseidon2 S-box source equation. -/
structure SboxStep (sourceWidth : Nat) where
  input : R1CS.LinearCombination
  inputBounded : SourceCompiler.CombinationBounded sourceWidth input
  output : Fin sourceWidth

namespace SboxStep

def compile {sourceWidth logicalWidth : Nat} (step : SboxStep sourceWidth)
    (sourceMap : SourceCompiler.SourceMap sourceWidth logicalWidth)
    (oneColumn : Fin logicalWidth) : SboxRow.Forms logicalWidth where
  selector := SparseForm.singleton oneColumn 1
  input := SourceCompiler.compileCombination sourceMap oneColumn
    step.input step.inputBounded
  output := sourceMap.form step.output

theorem compile_preserves {sourceWidth logicalWidth : Nat}
    (step : SboxStep sourceWidth)
    (sourceMap : SourceCompiler.SourceMap sourceWidth logicalWidth)
    (oneColumn : Fin logicalWidth) (assignment : Assignment F logicalWidth)
    (source : Circuit.Env) (one : assignment oneColumn = 1)
    (preserves : sourceMap.Preserves assignment source) :
    (step.compile sourceMap oneColumn).Preserves assignment
      (step.input.eval source) (source step.output.val) := by
  refine ⟨?_, ?_, ?_⟩
  · simp [compile, one]
  · exact SourceCompiler.compileCombination_eval sourceMap oneColumn
      step.input step.inputBounded assignment source one preserves
  · exact preserves step.output

/-- The compiled S-box row is exact under a preserving source map. -/
theorem residual_zero_iff {sourceWidth logicalWidth : Nat}
    (step : SboxStep sourceWidth)
    (sourceMap : SourceCompiler.SourceMap sourceWidth logicalWidth)
    (oneColumn : Fin logicalWidth) (assignment : Assignment F logicalWidth)
    (source : Circuit.Env) (one : assignment oneColumn = 1)
    (preserves : sourceMap.Preserves assignment source) :
    (step.compile sourceMap oneColumn).residual assignment = 0 ↔
      Spec.ProductionRelation.RowSemantics.seventhPower
          (step.input.eval source) = source step.output.val :=
  SboxRow.Forms.residual_zero_iff _ _ _ _
    (step.compile_preserves sourceMap oneColumn assignment source one preserves)

end SboxStep

/-- One retained Poseidon2 output bound to its source affine linear form. -/
structure OutputStep (sourceWidth : Nat) where
  output : Fin sourceWidth
  linear : R1CS.LinearCombination
  linearBounded : SourceCompiler.CombinationBounded sourceWidth linear

namespace OutputStep

def difference {sourceWidth logicalWidth : Nat}
    (step : OutputStep sourceWidth)
    (sourceMap : SourceCompiler.SourceMap sourceWidth logicalWidth)
    (oneColumn : Fin logicalWidth) : SparseForm logicalWidth :=
  SparseForm.add (sourceMap.form step.output)
    (SparseForm.scale (-1)
      (SourceCompiler.compileCombination sourceMap oneColumn
        step.linear step.linearBounded))

def compile {sourceWidth logicalWidth : Nat} (step : OutputStep sourceWidth)
    (sourceMap : SourceCompiler.SourceMap sourceWidth logicalWidth)
    (oneColumn : Fin logicalWidth) : PinRow.Forms logicalWidth where
  selector := SparseForm.singleton oneColumn 1
  value := step.difference sourceMap oneColumn

theorem difference_eval {sourceWidth logicalWidth : Nat}
    (step : OutputStep sourceWidth)
    (sourceMap : SourceCompiler.SourceMap sourceWidth logicalWidth)
    (oneColumn : Fin logicalWidth) (assignment : Assignment F logicalWidth)
    (source : Circuit.Env) (one : assignment oneColumn = 1)
    (preserves : sourceMap.Preserves assignment source) :
    (step.difference sourceMap oneColumn).eval assignment =
      source step.output.val - step.linear.eval source := by
  rw [difference, SparseForm.add_eval, preserves step.output,
    SparseForm.scale_eval,
    SourceCompiler.compileCombination_eval sourceMap oneColumn step.linear
      step.linearBounded assignment source one preserves]
  simp [sub_eq_add_neg]

theorem compile_preserves {sourceWidth logicalWidth : Nat}
    (step : OutputStep sourceWidth)
    (sourceMap : SourceCompiler.SourceMap sourceWidth logicalWidth)
    (oneColumn : Fin logicalWidth) (assignment : Assignment F logicalWidth)
    (source : Circuit.Env) (one : assignment oneColumn = 1)
    (preserves : sourceMap.Preserves assignment source) :
    (step.compile sourceMap oneColumn).Preserves assignment
      (source step.output.val - step.linear.eval source) := by
  constructor
  · simp [compile, one]
  · exact step.difference_eval sourceMap oneColumn assignment source one preserves

/-- The compiled output pin is exact under a preserving source map. -/
theorem residual_zero_iff {sourceWidth logicalWidth : Nat}
    (step : OutputStep sourceWidth)
    (sourceMap : SourceCompiler.SourceMap sourceWidth logicalWidth)
    (oneColumn : Fin logicalWidth) (assignment : Assignment F logicalWidth)
    (source : Circuit.Env) (one : assignment oneColumn = 1)
    (preserves : sourceMap.Preserves assignment source) :
    (step.compile sourceMap oneColumn).residual assignment = 0 ↔
      source step.output.val = step.linear.eval source := by
  rw [PinRow.Forms.residual_zero_iff _ _ _
    (step.compile_preserves sourceMap oneColumn assignment source one preserves)]
  exact Lean.Grind.AddCommGroup.sub_eq_zero_iff

end OutputStep

end NightstreamFPrime.Layout.ProductionRelation.PoseidonTraceRows
