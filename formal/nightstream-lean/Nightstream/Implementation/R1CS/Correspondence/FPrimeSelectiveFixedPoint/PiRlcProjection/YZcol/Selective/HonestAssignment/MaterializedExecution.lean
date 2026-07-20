import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.HonestAssignment.MaterializedExecution.Output

/-!
Full-program materialized execution for the bounded selective fixed-point
`y_zcol` projection slice.

Owns: composition of pointwise field/value transport with the exact abstract
program and checked derived-recurrence registry.

Does not own: retained final checks, selected-row completeness, producer
authority, projection bad-root soundness, transcript security, or permission
to remove rows.

Emits constraints: no.

| Leaf | Mathematical obligation | Authority class |
|---|---|---|
| `materialized.step_transport` | abstract recurrence values equal centered-word assignment values | derived |
| `materialized.registry` | every derived output obeys its exported predecessor/factor recurrence | artifact-checked + derived |
-/

namespace Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.HonestAssignment

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.ProjectionProgram
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.Materialized
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.RewriteBridge

/-- Every abstract recurrence of the deterministic intermediate program is
implemented by the concrete centered-word assignment. -/
theorem materializedStepsHold
    {source : Nat → Nat} {derived : Nat → F}
    (honest : HonestSourceBoundary source)
    (abstractSteps : AbstractStepsHold source derived decodedRewriteSteps) :
    MaterializedStepsEvidence (selectedAssignment source derived) := by
  constructor
  intro step member
  have abstract := abstractSteps.holds step member
  have sources := rewriteSourcesKnown step member
  have slots := rewriteDerivedSlotsCovered step member
  have previousEq :=
    (constructedValuesEvidence source derived).previousEq
      step.previous slots.2
  unfold selectedAssignment at previousEq
  unfold StepHolds
  rw [(outputValue_eq_abstract (derived := derived) honest step member).valueEq]
  unfold selectedAssignment
  rw [sourceValue_eq_abstract
      (derived := derivedNat derived) honest step.base sources.1,
    previousEq,
    factorSum_eq_abstract
      (derived := derivedNat derived) honest step.factors sources.2.1]
  cases outputEq : step.output <;>
    simpa only [AbstractStepHolds, derivedRhs, outputEq] using abstract

/-- The constructed assignment implements the exact derived-column witness
program exported by Rust, including its zero-base predecessor/factor
recurrences, and satisfies every decoded rewrite equation. -/
theorem materializedWitnessRewriteProgramHolds
    {source : Nat → Nat} {derived : Nat → F}
    (honest : HonestSourceBoundary source)
    (abstractSteps : AbstractStepsHold source derived decodedRewriteSteps) :
    WitnessRewriteProgramHolds (selectedAssignment source derived) := by
  apply witnessRewriteProgramHolds
    (assignment := selectedAssignment source derived)
  exact (materializedStepsHold honest abstractSteps).holds

/-- Every constructed derived field is tied to its exact registry payload and
obeys the corresponding zero-base Rust witness recurrence. -/
theorem materializedDerivedWitnessRecurrencesHold
    {source : Nat → Nat} {derived : Nat → F}
    (honest : HonestSourceBoundary source)
    (abstractSteps : AbstractStepsHold source derived decodedRewriteSteps) :
    ∀ step ∈ decodedRewriteSteps,
      match step.output with
      | .source _ => True
      | .derivedProductSum slot =>
          decodedDerivedRecurrencePayload step slot ∈
              Materialized.Checked.derivedProductSums.map
                rawDerivedRecurrence ∧
            derivedValue (selectedAssignment source derived) slot =
              previousValue (selectedAssignment source derived) step.previous +
                factorSum (selectedAssignment source derived) step.factors := by
  intro step member
  have holds := (materializedStepsHold honest abstractSteps).holds step member
  cases outputEq : step.output with
  | source linear => trivial
  | derivedProductSum slot =>
      exact derivedStepHolds_witnessRecurrence member outputEq holds

end Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.HonestAssignment
