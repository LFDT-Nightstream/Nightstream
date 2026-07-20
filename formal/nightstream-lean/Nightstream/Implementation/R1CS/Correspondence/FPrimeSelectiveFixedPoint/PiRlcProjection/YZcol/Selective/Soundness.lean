import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.ActiveBridge
import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.Agreement

/-!
Selected-row soundness for the bounded fixed-point `y_zcol` projection slice.

Owns: transport from every decoded selected row, an active selector, and
constant one to every source definition and retained final trace check under
the independently reconstructed source assignment. These consequences yield
the generic low/high projection-row interface for `ActiveBridge.tracePair`.

Does not own: selector enforcement, honest selected-row construction, producer
authority, the projection bad-root reduction, production conformance, global
row-removal authority, or a monolithic SSA claim. The reconstruction is the
composition of the checked compiler and source programs. Their only shared
output targets are the ladder bases; the other source outputs are owned by the
direct-operation and trace-elimination paths.

Emits constraints: no.

Assurance tier: artifact-checked for this bounded fixture, followed by the
existing model-level projection-row interface.

| Child path | Mathematical obligation | Authority class | Theorem owner |
|---|---|---|---|
| `selected.rewrite_rows` | active compact rows imply every rewrite recurrence | checked coefficients | `RewriteBridge.allRewriteStepsHold` |
| `selected.final_checks` | retained compact rows imply their source A/B/C checks | checked coefficients + direct dataflow | `retainedSourceRowHolds` |
| `source.definitions` | the deterministic source assignment satisfies every source equation | computed + checked | `SourceProgram.sourceAssignmentDefinitionsHold` |
| `source.trace_handoff` | exact certificate coverage transports definitions and checks to both traces | artifact-checked | `selectedRows_imply_rowsSatisfied` |
| `source.message_aggregate` | selected rows and bound producer inputs imply the typed aggregate or named bad-root event | artifact-checked + security boundary | `selectedRows_decodedOutput_eq_messageAggregate_or_badRoot` |
-/

namespace Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.Soundness

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Program
open Nightstream.Implementation.R1CS.ProjectionArtifactProgram
open Nightstream.Implementation.R1CS.ProjectionIndexedRows
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.Materialized.Semantics
open Nightstream.Implementation.R1CS.PiCcsOutputDigest.ActiveSourceLayout
open Nightstream.Implementation.R1CS.ProjectionPhi81
open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.OutputClaims.EvaluationHomomorphism.Authority.DelayedPackedProjection
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier
open Nightstream.SuperNeo.ProjectionCheck

private abbrev certificate :=
  Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Checked.artifact.certificate

private theorem rewriteRowsSatisfied_of_selected
    {assignment : Nat → Nat}
    (selected : RowsSatisfied Materialized.Artifact.decodedRows assignment) :
    RowsSatisfied Materialized.Artifact.rewriteRows assignment := by
  intro row member
  apply selected row
  exact (List.mem_filter.mp member).1

private theorem retainedRowsSatisfied_of_selected
    {assignment : Nat → Nat}
    (selected : RowsSatisfied Materialized.Artifact.decodedRows assignment) :
    RowsSatisfied Materialized.Artifact.retainedRows assignment := by
  intro row member
  apply selected row
  exact (List.mem_filter.mp member).1

/-! ## Retained final checks -/

private def retainedSourceRow
    (step : RewriteBridge.DecodedRetainedStep) : Row :=
  { a := step.a.programTerms
    b := step.b.programTerms
    c := step.c.programTerms }

private def retainedSourceRows : List Row :=
  RewriteBridge.decodedRetainedSteps.map retainedSourceRow

private def TermsKnown
    (linear : SourceDecode.DecodedSourceLinearCombination) : Prop :=
  ∀ term ∈ linear.programTerms, term.1 ∈ Agreement.finalKnown

private def termsKnownCheck
    (linear : SourceDecode.DecodedSourceLinearCombination) : Bool :=
  linear.programTerms.all fun term =>
    Agreement.finalKnown.contains term.1

set_option maxRecDepth 100000 in
private theorem retainedTermsKnown :
    ∀ step ∈ RewriteBridge.decodedRetainedSteps,
      TermsKnown step.a ∧ TermsKnown step.b ∧ TermsKnown step.c := by
  have checked : RewriteBridge.decodedRetainedSteps.all (fun step =>
      termsKnownCheck step.a &&
        (termsKnownCheck step.b && termsKnownCheck step.c)) = true := by
    native_decide
  simpa only [List.all_eq_true, Bool.and_eq_true, termsKnownCheck,
    TermsKnown, List.contains_iff_mem] using checked

private theorem lcEval_eq_of_agreeOn
    {left right : Nat → Nat} {known : List Nat}
    (agreement : AgreeOn left right known)
    (terms : List (Nat × Nat))
    (references : ∀ term ∈ terms, term.1 ∈ known) :
    lcEval left terms = lcEval right terms := by
  unfold lcEval
  have foldAgree : ∀ initial,
      terms.foldl (fun total term => total + term.2 * left term.1) initial =
        terms.foldl (fun total term => total + term.2 * right term.1)
          initial := by
    intro initial
    induction terms generalizing initial with
    | nil => rfl
    | cons head tail inductionHypothesis =>
        simp only [List.foldl]
        rw [agreement head.1 (references head (by simp))]
        apply inductionHypothesis
        intro term member
        exact references term (by simp [member])
  rw [foldAgree 0]

private theorem retainedSourceRowHolds
    {assignment : Nat → Nat}
    (agreement : AgreeOn
      (SourceProgram.sourceAssignment assignment)
      (SourceDecode.compilerAssignment assignment) Agreement.finalKnown)
    (step : RewriteBridge.DecodedRetainedStep)
    (member : step ∈ RewriteBridge.decodedRetainedSteps)
    (holds : RewriteBridge.RetainedHolds assignment step) :
    RowHolds (SourceProgram.sourceAssignment assignment)
      (retainedSourceRow step) := by
  have known := retainedTermsKnown step member
  have aAgreement := lcEval_eq_of_agreeOn agreement
    step.a.programTerms known.1
  have bAgreement := lcEval_eq_of_agreeOn agreement
    step.b.programTerms known.2.1
  have cAgreement := lcEval_eq_of_agreeOn agreement
    step.c.programTerms known.2.2
  unfold retainedSourceRow RowHolds
  rw [aAgreement, bAgreement, cAgreement]
  unfold RewriteBridge.RetainedHolds RewriteBridge.sourceValue at holds
  have values := congrArg Fin.val holds
  simpa [Materialized.Semantics.fieldResidue,
    Materialized.Semantics.modulus_eq, Fin.val_mul, lcEval,
    Nat.mod_mod] using values

private theorem retainedSourceRowsSatisfied
    {assignment : Nat → Nat}
    (agreement : AgreeOn
      (SourceProgram.sourceAssignment assignment)
      (SourceDecode.compilerAssignment assignment) Agreement.finalKnown)
    (retained : ∀ step ∈ RewriteBridge.decodedRetainedSteps,
      RewriteBridge.RetainedHolds assignment step) :
    Satisfies retainedSourceRows
      (SourceProgram.sourceAssignment assignment) := by
  intro row rowMember
  rcases List.mem_map.mp rowMember with ⟨step, stepMember, rowEq⟩
  subst row
  exact retainedSourceRowHolds agreement step stepMember
    (retained step stepMember)

set_option maxRecDepth 100000 in
private theorem certificateChecksCovered :
    ∀ row ∈ certificate.checks,
      ∃ reconstructed,
        reconstructed ∈ retainedSourceRows ∧
          RowsPermutationEquivalent reconstructed row := by
  native_decide

private theorem certificateChecksSatisfied
    {assignment : Nat → Nat}
    (agreement : AgreeOn
      (SourceProgram.sourceAssignment assignment)
      (SourceDecode.compilerAssignment assignment) Agreement.finalKnown)
    (retained : ∀ step ∈ RewriteBridge.decodedRetainedSteps,
      RewriteBridge.RetainedHolds assignment step) :
    Satisfies certificate.checks
      (SourceProgram.sourceAssignment assignment) := by
  intro row member
  rcases certificateChecksCovered row member with
    ⟨reconstructed, reconstructedMember, equivalent⟩
  exact rowHolds_of_permutationEquivalent equivalent
    (retainedSourceRowsSatisfied agreement retained reconstructed
      reconstructedMember)

/-! ## Selected rows to the typed projection-row interface -/

/-- Satisfaction of the exact decoded selected rows forces every source
projection definition and final check under the independently reconstructed
source assignment. No source-row satisfaction, decoded equality, or semantic
acceptance appears among the premises. -/
theorem selectedRows_imply_rowsSatisfied
    {shape : Nightstream.SuperNeo.Folding.PiCCS.SplitNc.SemanticShape}
    (sourceCount : shape.sourceCount = 15)
    {assignment : Nat → Nat}
    (selected : RowsSatisfied Materialized.Artifact.decodedRows assignment)
    (selectorOne :
      assignment Materialized.Checked.steadySelectorColumn = 1)
    (constantOne : assignment 0 = 1) :
    PiCcsOutputDigest.ActiveSourceLayout.ProjectionIdentity.RowsSatisfied
      (ActiveBridge.tracePair shape sourceCount)
      (SourceProgram.sourceAssignment assignment) := by
  have rewriteRows := rewriteRowsSatisfied_of_selected selected
  have retainedRows := retainedRowsSatisfied_of_selected selected
  have rewrites := RewriteBridge.allRewriteStepsHold
    rewriteRows selectorOne
  have retained := RewriteBridge.allRetainedStepsHold
    retainedRows selectorOne
  have agreement := Agreement.sourceCompilerAgreeOnFinalKnown
    constantOne rewrites
  have coverage := ArtifactRows.certificate_covers
    Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Checked.structureValid
  have certificateChecks := certificateChecksSatisfied agreement retained
  constructor
  · rw [ActiveBridge.tracePair_traces shape sourceCount]
    intro definition member
    apply SourceProgram.sourceAssignmentDefinitionsHold assignment definition
    exact (coverage.definitionsIff definition).mp member
  · rw [ActiveBridge.tracePair_traces shape sourceCount]
    intro row member
    exact certificateChecks row ((coverage.checksIff row).mp member)

/-- End-to-end selected-row consequence for this bounded slice. Producer
authority remains an explicit upstream column binding, and failure remains the
exact sampled `BatchBadRoot`; this theorem assigns no probability to it. -/
theorem selectedRows_decodedOutput_eq_messageAggregate_or_badRoot
    {shape : SemanticShape}
    (sourceCount : shape.sourceCount = 15)
    {assignment : Nat → Nat}
    (selected : RowsSatisfied Materialized.Artifact.decodedRows assignment)
    (selectorOne :
      assignment Materialized.Checked.steadySelectorColumn = 1)
    (constantOne : assignment 0 = 1)
    {producer : SourceRole shape → Nat}
    (upstream : ProducerBinding.UpstreamProducerColumnsBound producer)
    {message : OutputMessage shape}
    (yZcolBound : BindingsHoldFor .yZcolOutput
      (semanticAssignment (SourceProgram.sourceAssignment assignment))
      producer message) :
    ProjectionIdentity.decodedOutput
          (ActiveBridge.tracePair shape sourceCount)
          (SourceProgram.sourceAssignment assignment) =
        sourceAggregate
          (ProjectionIdentity.decodedChallenges
            (ActiveBridge.tracePair shape sourceCount)
            (SourceProgram.sourceAssignment assignment))
          message.yZcol \/
      BatchBadRoot ProjectionProgram.K.ops
        (ProjectionProgram.BatchIdentity
          (ActiveBridge.tracePair shape sourceCount).traces
          (SourceProgram.sourceAssignment assignment)) := by
  have rows := selectedRows_imply_rowsSatisfied sourceCount selected
    selectorOne constantOne
  exact ProjectionIdentity.rows_decodedOutput_eq_messageAggregate_or_badRoot
    (ActiveBridge.tracePairShapeValid shape sourceCount)
    (SourceProgram.sourceAssignmentConstantOne constantOne)
    rows
    (ActiveBridge.consumerMatches sourceCount upstream)
    yZcolBound

end Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.Soundness
