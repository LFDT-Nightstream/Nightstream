import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.QuadraticRefinement.UnitOutput

/-!
Scalar terminal-value transport for the focused compact `y_zcol` quadratic
refinement.

Owns: conversion of a unit decoded output and an equivalent symbolic form to
the corresponding expected terminal equality.

Does not own: certificate aggregation, listwise terminal matching,
selected-row materialization, protocol authority, or security events.

Emits constraints: no.

| Value leaf | Mathematical obligation | Authority class |
|---|---|---|
| output column | a checked unit output denotes the expected source-field coordinate | direct dataflow |
| symbolic form | equivalent normalized quadratic forms have equal evaluation | derived |
| terminal equality | chaining output and form equality yields the expected terminal equation | derived |
-/

namespace Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.QuadraticRefinement

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.ProjectionProgram
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.SourceDecode
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.RewriteBridge

theorem terminalMatches_value {assignment : Nat → Nat}
    {actual : TerminalExpression} {expected : ExpectedTerminal}
    (valid : sourceValue assignment actual.output =
      Materialized.QuadraticForm.eval (sourceFieldAssignment assignment) actual.expression)
    (matching : TerminalMatches actual expected) :
    sourceFieldAssignment assignment expected.outputColumn =
      Materialized.QuadraticForm.eval (sourceFieldAssignment assignment) expected.expression := by
  rcases matching with ⟨unit, equivalent⟩
  calc
    sourceFieldAssignment assignment expected.outputColumn =
        sourceValue assignment actual.output :=
      (sourceValue_of_unitOutput assignment actual.output
        expected.outputColumn unit).symm
    _ = Materialized.QuadraticForm.eval (sourceFieldAssignment assignment) actual.expression := valid
    _ = Materialized.QuadraticForm.eval (sourceFieldAssignment assignment) expected.expression :=
      Materialized.QuadraticForm.eval_eq_of_equivalent equivalent _

end Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.QuadraticRefinement
