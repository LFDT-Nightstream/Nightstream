import NightstreamFPrime.Layout.ProductionRelation.PoseidonTraceRows
import NightstreamFPrime.Layout.ProductionRelation.SourceCompiler

/-!
Owns fail-closed compilation from one pair of Poseidon trace expressions to
one direct selective source step. Inputs and linear recipes must be bounded
affine expressions. Retained outputs must be exact bounded source variables.
-/

namespace NightstreamFPrime.Layout.ProductionRelation.PoseidonSourceRows

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra

/-- A trace expression recognized as one exact source column. -/
structure ColumnSource (sourceWidth : Nat) (expression : Circuit.Expr) where
  column : Fin sourceWidth
  sound : ∀ env, env column.val = expression.eval env

/-- Fail-closed exact-variable recognition. -/
def sourceColumn? (sourceWidth : Nat) (expression : Circuit.Expr) :
    Option (ColumnSource sourceWidth expression) :=
  match expression with
  | .var column =>
      if bounded : column < sourceWidth then
        some ⟨⟨column, bounded⟩, by intro env; rfl⟩
      else
        none
  | _ => none

/-- One source expression pair compiled as a direct S-box step. -/
structure SboxSource (sourceWidth : Nat) where
  inputExpression : Circuit.Expr
  outputExpression : Circuit.Expr
  step : PoseidonTraceRows.SboxStep sourceWidth
  inputSound : ∀ env, step.input.eval env = inputExpression.eval env
  outputSound : ∀ env, env step.output.val = outputExpression.eval env

def compileSbox? (sourceWidth : Nat) (input output : Circuit.Expr) :
    Option (SboxSource sourceWidth) :=
  match affineFound : SourceCompiler.lowerAffine? sourceWidth input with
  | none => none
  | some affine =>
      match sourceColumn? sourceWidth output with
      | none => none
      | some column =>
          some
            { inputExpression := input
              outputExpression := output
              step :=
                { input := affine.combination
                  inputBounded := affine.bounded
                  output := column.column }
              inputSound := SourceCompiler.lowerAffine?_sound
                input affine affineFound
              outputSound := column.sound }

/-- A successfully compiled S-box expression pair has exactly the source
expression semantics. -/
theorem SboxSource.residual_zero_iff {sourceWidth logicalWidth : Nat}
    (source : SboxSource sourceWidth)
    (sourceMap : SourceCompiler.SourceMap sourceWidth logicalWidth)
    (oneColumn : Fin logicalWidth) (assignment : Assignment F logicalWidth)
    (env : Circuit.Env) (one : assignment oneColumn = 1)
    (preserves : sourceMap.Preserves assignment env) :
    (source.step.compile sourceMap oneColumn).residual assignment = 0 ↔
      Spec.ProductionRelation.RowSemantics.seventhPower
          (source.inputExpression.eval env) =
        source.outputExpression.eval env := by
  rw [PoseidonTraceRows.SboxStep.residual_zero_iff source.step sourceMap
    oneColumn assignment env one preserves]
  rw [source.inputSound env, source.outputSound env]

/-- One source expression pair compiled as a direct linear-output step. -/
structure OutputSource (sourceWidth : Nat) where
  outputExpression : Circuit.Expr
  linearExpression : Circuit.Expr
  step : PoseidonTraceRows.OutputStep sourceWidth
  outputSound : ∀ env, env step.output.val = outputExpression.eval env
  linearSound : ∀ env, step.linear.eval env = linearExpression.eval env

def compileOutput? (sourceWidth : Nat) (output linear : Circuit.Expr) :
    Option (OutputSource sourceWidth) :=
  match sourceColumn? sourceWidth output with
  | none => none
  | some column =>
      match affineFound : SourceCompiler.lowerAffine? sourceWidth linear with
      | none => none
      | some affine =>
          some
            { outputExpression := output
              linearExpression := linear
              step :=
                { output := column.column
                  linear := affine.combination
                  linearBounded := affine.bounded }
              outputSound := column.sound
              linearSound := SourceCompiler.lowerAffine?_sound
                linear affine affineFound }

/-- A successfully compiled linear-output expression pair has exactly the
source expression semantics. -/
theorem OutputSource.residual_zero_iff {sourceWidth logicalWidth : Nat}
    (source : OutputSource sourceWidth)
    (sourceMap : SourceCompiler.SourceMap sourceWidth logicalWidth)
    (oneColumn : Fin logicalWidth) (assignment : Assignment F logicalWidth)
    (env : Circuit.Env) (one : assignment oneColumn = 1)
    (preserves : sourceMap.Preserves assignment env) :
    (source.step.compile sourceMap oneColumn).residual assignment = 0 ↔
      source.outputExpression.eval env = source.linearExpression.eval env := by
  rw [PoseidonTraceRows.OutputStep.residual_zero_iff source.step sourceMap
    oneColumn assignment env one preserves]
  rw [source.outputSound env, source.linearSound env]

end NightstreamFPrime.Layout.ProductionRelation.PoseidonSourceRows
