import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.HonestAssignment.RewriteProvenance
import Nightstream.Implementation.R1CS.Correspondence.Projection.IndexedRows

/-!
Checked provenance for the four retained final rows in the bounded selective
`y_zcol` artifact.

Owns: source-closure coverage and exact source-row membership for retained
checks.

Does not own: honest semantic truth, selected-row satisfaction, projection
authority, or security reduction.

Emits constraints: no.

| Retained leaf | Mathematical obligation | Authority class |
|---|---|---|
| source closure | every retained linear form reads checked compiler columns | checked |
| source membership | each retained row names its exact source obligation | checked |
-/

namespace Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.HonestAssignment

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.ProjectionIndexedRows
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.RewriteBridge

private abbrev certificate := Materialized.Checked.sourceArtifact.certificate

def RetainedSourcesKnown (step : DecodedRetainedStep) : Prop :=
  LinearKnown step.a ∧ LinearKnown step.b ∧ LinearKnown step.c

def retainedSourceRow (step : DecodedRetainedStep) : Row :=
  { a := step.a.programTerms
    b := step.b.programTerms
    c := step.c.programTerms }

def RetainedSourceRowCovered (step : DecodedRetainedStep) : Prop :=
  ∃ indexedCheck ∈ certificate.indexedChecks,
    indexedCheck.1 = step.sourceRow ∧
      RowsPermutationEquivalent (retainedSourceRow step) indexedCheck.2

structure RetainedFactShape where
  aColumns : List Nat
  bColumns : List Nat
  cColumns : List Nat
  sourceIndex : Nat
  sourceRow : Row
deriving DecidableEq, Repr

/-- Proof-free projection used by the four-record retained certificate. -/
def retainedFactShape (step : DecodedRetainedStep) : RetainedFactShape :=
  { aColumns := linearColumns step.a
    bColumns := linearColumns step.b
    cColumns := linearColumns step.c
    sourceIndex := step.sourceRow
    sourceRow := retainedSourceRow step }

private def retainedSourcesShapeCheck
    (shape : RetainedFactShape) : Bool :=
  columnsKnownCheck shape.aColumns &&
    (columnsKnownCheck shape.bColumns &&
      columnsKnownCheck shape.cColumns)

private theorem retainedSourcesKnown_of_shape_check_true
    {step : DecodedRetainedStep}
    (checked : retainedSourcesShapeCheck (retainedFactShape step) = true) :
    RetainedSourcesKnown step := by
  have parts :
      columnsKnownCheck (linearColumns step.a) = true ∧
        columnsKnownCheck (linearColumns step.b) = true ∧
          columnsKnownCheck (linearColumns step.c) = true := by
    simpa only [retainedSourcesShapeCheck, retainedFactShape,
      Bool.and_eq_true] using checked
  exact ⟨linearKnown_of_shape_check_true parts.1,
    linearKnown_of_shape_check_true parts.2.1,
    linearKnown_of_shape_check_true parts.2.2⟩

private def retainedSourceShapeCheck (shape : RetainedFactShape) : Bool :=
  decide (∃ indexedCheck ∈ certificate.indexedChecks,
    indexedCheck.1 = shape.sourceIndex ∧
      RowsPermutationEquivalent shape.sourceRow indexedCheck.2)

private theorem retainedSourceRowCovered_of_shape_check_true
    {step : DecodedRetainedStep}
    (checked : retainedSourceShapeCheck (retainedFactShape step) = true) :
    RetainedSourceRowCovered step := by
  unfold retainedSourceShapeCheck at checked
  simpa only [RetainedSourceRowCovered, retainedFactShape] using
    (of_decide_eq_true checked)

private def retainedFactShapeCheck (shape : RetainedFactShape) : Bool :=
  retainedSourcesShapeCheck shape && retainedSourceShapeCheck shape

private def retainedFactsShapeCheck
    (shapes : List RetainedFactShape) : Bool :=
  shapes.all retainedFactShapeCheck

private theorem retainedFacts_of_shape_check_true
    {steps : List DecodedRetainedStep}
    (checked :
      retainedFactsShapeCheck (steps.map retainedFactShape) = true) :
    ∀ step ∈ steps,
      RetainedSourcesKnown step ∧
        RetainedSourceRowCovered step := by
  intro step member
  have shapeMember :
      retainedFactShape step ∈ steps.map retainedFactShape :=
    List.mem_map.mpr ⟨step, member, rfl⟩
  have allChecked :
      (steps.map retainedFactShape).all retainedFactShapeCheck = true := by
    simpa only [retainedFactsShapeCheck] using checked
  have stepChecked :=
    (List.all_eq_true.mp allChecked) (retainedFactShape step) shapeMember
  have parts :
      retainedSourcesShapeCheck (retainedFactShape step) = true ∧
        retainedSourceShapeCheck (retainedFactShape step) = true := by
    simpa only [retainedFactShapeCheck, Bool.and_eq_true] using stepChecked
  exact ⟨retainedSourcesKnown_of_shape_check_true parts.1,
    retainedSourceRowCovered_of_shape_check_true parts.2⟩

def retainedFactsData : List RetainedFactShape :=
  decodedRetainedSteps.map retainedFactShape

theorem decodedRetainedStepsLengthExact :
    decodedRetainedSteps.length = 4 := by
  calc
    decodedRetainedSteps.length = retainedPairs.length := by
      simpa only [List.length_map] using
        (congrArg List.length retainedPairStepsExact).symm
    _ = Materialized.Artifact.retainedRows.length := by
      simpa only [List.length_map] using
        congrArg List.length retainedPairRowsExact
    _ = 4 := Materialized.Artifact.retainedRowCount

theorem retainedFactsDataLengthExact : retainedFactsData.length = 4 := by
  simpa only [retainedFactsData, List.length_map] using
    decodedRetainedStepsLengthExact

theorem retainedFactsDataWithinCertificateLimit :
    retainedFactsData.length ≤ 256 := by
  rw [retainedFactsDataLengthExact]
  decide

set_option maxRecDepth 100000 in
private theorem retainedFactsShapeCheck_true :
    retainedFactsShapeCheck retainedFactsData = true := by
  native_decide

theorem retainedSourcesKnown :
    ∀ step ∈ decodedRetainedSteps, RetainedSourcesKnown step := by
  intro step member
  exact (retainedFacts_of_shape_check_true retainedFactsShapeCheck_true
    step member).1

theorem retainedSourceRowsCovered :
    ∀ step ∈ decodedRetainedSteps,
      RetainedSourceRowCovered step := by
  intro step member
  exact (retainedFacts_of_shape_check_true retainedFactsShapeCheck_true
    step member).2

theorem retainedSourceRowHolds_of_checks
    {assignment : Nat → Nat}
    (checksHold : Satisfies certificate.checks assignment)
    (step : DecodedRetainedStep) (member : step ∈ decodedRetainedSteps) :
    RowHolds assignment (retainedSourceRow step) := by
  rcases retainedSourceRowsCovered step member with
    ⟨indexedCheck, indexedMember, _, equivalent⟩
  have checkMember : indexedCheck.2 ∈ certificate.checks := by
    apply List.mem_map.mpr
    exact ⟨indexedCheck, indexedMember, rfl⟩
  have reverseEquivalent :
      RowsPermutationEquivalent indexedCheck.2 (retainedSourceRow step) :=
    ⟨equivalent.1.symm, equivalent.2.1.symm, equivalent.2.2.symm⟩
  exact rowHolds_of_permutationEquivalent reverseEquivalent
    (checksHold indexedCheck.2 checkMember)

end Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.HonestAssignment
