import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.QuadraticRefinement.TerminalValue

/-!
Terminal transport for the focused compact `y_zcol` quadratic refinement.

Owns: conversion from valid symbolic terminals and exact terminal matching to
the independent expected quadratic obligations.

Does not own: certificate aggregation, symbolic execution internals,
selected-row materialization, protocol authority, or security events.

Emits constraints: no.

| Transport leaf | Mathematical obligation | Authority class |
|---|---|---|
| terminal validity | every executed terminal evaluates to its decoded source output | direct dataflow |
| terminal matching | pairwise exact matches transport validity to the independent expected forms | checked + derived |
| group conclusion | satisfied rewrite steps imply every expected terminal equation | derived |
-/

namespace Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.QuadraticRefinement

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.ProjectionProgram
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.SourceDecode
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.RewriteBridge

def ExpectedHolds (assignment : Nat → Nat)
    (expected : List ExpectedTerminal) : Prop :=
  ∀ terminal ∈ expected,
    sourceFieldAssignment assignment terminal.outputColumn =
      Materialized.QuadraticForm.eval (sourceFieldAssignment assignment) terminal.expression

private theorem expectedHolds_of_terminalsMatch {assignment : Nat → Nat}
    {actual : List TerminalExpression} {expected : List ExpectedTerminal}
    (valid : ∀ terminal ∈ actual,
      sourceValue assignment terminal.output =
        Materialized.QuadraticForm.eval (sourceFieldAssignment assignment) terminal.expression)
    (matching : TerminalsMatch actual expected) :
    ExpectedHolds assignment expected := by
  induction matching with
  | nil =>
      intro terminal member
      simp at member
  | @cons actualHead expectedHead actualTail expectedTail headMatch tailMatch
      inductionHypothesis =>
      intro terminal member
      simp only [List.mem_cons] at member
      rcases member with rfl | tailMember
      · apply terminalMatches_value
        · exact valid actualHead (by simp)
        · exact headMatch
      · apply inductionHypothesis
        · intro candidate candidateMember
          exact valid candidate (by simp [candidateMember])
        · exact tailMember

theorem groupExpectedHolds {assignment : Nat → Nat}
    {steps : List DecodedRewriteStep} {expected : List ExpectedTerminal}
    (stepsHold : StepsHold assignment steps)
    (matching : GroupMatches steps expected) :
    ExpectedHolds assignment expected := by
  rcases matching with ⟨actual, executed, pairwise⟩
  exact expectedHolds_of_terminalsMatch
    (executeGroup_terminalsValid stepsHold executed) pairwise

end Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.QuadraticRefinement
