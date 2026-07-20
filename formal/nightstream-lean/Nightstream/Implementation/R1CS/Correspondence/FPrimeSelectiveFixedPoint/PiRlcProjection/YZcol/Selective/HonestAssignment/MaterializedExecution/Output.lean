import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.HonestAssignment.MaterializedExecution.Core

/-!
Output-arm transport for one decoded `y_zcol` rewrite step.

Owns: pointwise equality between a materialized rewrite output and its
abstract source or derived value.

Does not own: source/factor transport, full-program recurrence composition,
retained checks, selected-row completeness, or producer authority.

Emits constraints: no.

| Leaf | Mathematical obligation | Authority class |
|---|---|---|
| `materialized.output` | source and derived output arms preserve the abstract value | derived |
-/

namespace Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.HonestAssignment

open Nightstream.Implementation.R1CS.ProjectionProgram
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.Materialized
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.RewriteBridge

structure OutputValueEvidence (source : Nat → Nat) (derived : Nat → F)
    (step : DecodedRewriteStep) : Prop where
  valueEq :
    outputValue (selectedAssignment source derived) step.output =
      match step.output with
      | .source linear => abstractSourceValue source linear
      | .derivedProductSum slot =>
          derived slot.compilerIndex

theorem outputValue_eq_abstract
    {source : Nat → Nat} {derived : Nat → F}
    (honest : HonestSourceBoundary source)
    (step : DecodedRewriteStep) (member : step ∈ decodedRewriteSteps) :
    OutputValueEvidence source derived step := by
  constructor
  have sources := rewriteSourcesKnown step member
  have slots := (rewriteDerivedSlotsCovered step member).1
  cases outputEq : step.output with
  | source linear =>
      have linearKnown : LinearKnown linear := by
        simpa only [outputEq] using sources.2.2
      unfold outputValue selectedAssignment
      exact sourceValue_eq_abstract
        (derived := derivedNat derived) honest linear linearKnown
  | derivedProductSum slot =>
      have slotKnown : slot ∈ decodedDerivedSlots := by
        simpa only [outputEq] using slots
      exact (constructedValuesEvidence source derived).derivedEq slot slotKnown

end Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.HonestAssignment
