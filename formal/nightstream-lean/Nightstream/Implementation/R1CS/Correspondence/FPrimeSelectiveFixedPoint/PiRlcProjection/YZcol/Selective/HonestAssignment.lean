import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.HonestAssignment.MaterializedExecution
import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.HonestAssignment.RetainedFacts

/-!
Honest retained checks and selected-row completeness for the bounded
selective fixed-point `y_zcol` projection slice.

Owns: the four retained final checks and construction of a satisfying
selected-row assignment from independently stated source equations.

Does not own: abstract or materialized rewrite execution, producer authority,
projection bad-root soundness, transcript security, production-wide
assignment conformance, or permission to remove rows.

Emits constraints: no.

| Leaf | Mathematical obligation | Authority class |
|---|---|---|
| `honest.retained_checks` | the four retained source rows hold under the materialized assignment | derived |
| `honest.selected_rows` | honest abstract execution constructs a satisfying compact assignment | computed + derived |
-/

namespace Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.HonestAssignment

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.ProjectionProgram
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.Materialized
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.Materialized.Semantics
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.RewriteBridge
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.Completeness

/-! ## Four honest final checks -/

private theorem retainedAbstractHolds
    {source : Nat → Nat} (honest : HonestSourceBoundary source)
    (step : DecodedRetainedStep) (member : step ∈ decodedRetainedSteps) :
    abstractSourceValue source step.a * abstractSourceValue source step.b =
      abstractSourceValue source step.c := by
  have rowHolds := retainedSourceRowHolds_of_checks
    honest.finalChecks step member
  unfold retainedSourceRow RowHolds at rowHolds
  unfold abstractSourceValue
  have reduced := congrArg
    (fun value : Nat =>
      value % Nightstream.SuperNeo.Concrete.goldilocksModulus) rowHolds
  apply Fin.ext
  simpa [Materialized.Semantics.fieldResidue,
    Materialized.Semantics.modulus_eq, Fin.val_mul, Nat.mod_mod] using reduced

theorem materializedRetainedHold
    {source : Nat → Nat} {derived : Nat → F}
    (honest : HonestSourceBoundary source) :
    ∀ step ∈ decodedRetainedSteps,
      RetainedHolds (selectedAssignment source derived) step := by
  intro step member
  have known := retainedSourcesKnown step member
  unfold RetainedHolds
  unfold selectedAssignment
  rw [sourceValue_eq_abstract
      (derived := derivedNat derived) honest step.a known.1,
    sourceValue_eq_abstract
      (derived := derivedNat derived) honest step.b known.2.1,
    sourceValue_eq_abstract
      (derived := derivedNat derived) honest step.c known.2.2]
  exact retainedAbstractHolds honest step member

/-- Honest completeness once the independently stated source/direct equations
have discharged `RewriteTerminalsHold`. No selected-row satisfaction or
decoded equality is assumed. -/
theorem selectedRowsSatisfied_of_honestSource
    {source : Nat → Nat} {derived : Nat → F}
    (honest : HonestSourceBoundary source)
    (abstractSteps : AbstractStepsHold source derived decodedRewriteSteps) :
    RowsSatisfied Materialized.Artifact.decodedRows
      (selectedAssignment source derived) := by
  have witnessProgram :=
    materializedWitnessRewriteProgramHolds honest abstractSteps
  apply Completeness.selectedRowsSatisfied_of_stepsHold
    (materializedAssignment_selectorOne source (derivedNat derived))
  · exact witnessProgram.2
  · exact materializedRetainedHold honest

theorem exists_selectedRows_of_honestSource
    {source : Nat → Nat} (honest : HonestSourceBoundary source)
    (terminals : RewriteTerminalsHold source) :
    ∃ assignment,
      assignment Materialized.Checked.constantOneColumn = 1 ∧
      assignment Materialized.Checked.steadySelectorColumn = 1 ∧
      AssignmentCanonical assignment ∧
      RowsSatisfied Materialized.Artifact.decodedRows assignment := by
  let derived : Nat → F := derivedAssignment source
  have abstractSteps :
      AbstractStepsHold source derived decodedRewriteSteps := by
    change AbstractStepsHold source (derivedAssignment source)
      decodedRewriteSteps
    exact constructedAbstractStepsHold terminals
  refine ⟨selectedAssignment source derived, ?_, ?_, ?_,
    selectedRowsSatisfied_of_honestSource honest abstractSteps⟩
  · exact materializedAssignment_constantOne source
      (derivedNat derived)
  · exact materializedAssignment_selectorOne source
      (derivedNat derived)
  · exact materializedAssignment_canonical source
      (derivedNat derived)

end Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.HonestAssignment
