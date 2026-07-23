import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.SourceRowsSoundness
import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.SourceSatisfaction
import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.VisibleAgreement

/-!
Selected combined-NC rows imply the exact typed source-stage consequences.

Owns: composition of the ordered source/compiler agreement with reconstruction
and the independent source-row soundness theorem.

Does not own: selector or constant-one enforcement, transcript/state binding,
raw-child authority, SumCheck soundness, commitment binding, or `y_ring`.

Emits constraints: none.
-/

/-!
| Stable stage path | Obligation | Authority class |
|---|---|---|
| `f_prime.pi_ccs_nc.delayed.combined.selected_rows_soundness` | Derive all selected source obligations from satisfaction of exact emitted rows. | derived |

-/

namespace Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.SelectedRowsSoundness

open Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized

set_option maxRecDepth 100000 in
/-- Literal satisfaction of the exact selected production rows establishes
the padding, claimed-initial, complete claimed-chain, and terminal semantics
on the independently reconstructed source assignment. No source-row truth or
visible-column equality is accepted from the caller. -/
theorem generatedEmittedRowsSatisfy_implies_consequences
    {assignment : Nat → Nat}
    (selectedRows :
      SelectiveArtifactPairs.Artifact.GeneratedEmittedRowsSatisfy assignment)
    (selectorOne : assignment Metadata.steadySelectorColumn = 1)
    (constantOne : assignment 0 = 1) :
    SourceRowsSoundness.Consequences
      (PhysicalAgreement.reconstructedAssignment assignment) := by
  have visibleAgreement :=
    VisibleAgreement.selectedRows_imply_visibleOutputs_agree selectedRows
      selectorOne constantOne
  have allVisibleAgreement :
      Program.AgreeOn (PhysicalAgreement.reconstructedAssignment assignment)
        (SourceAssignment.compilerAssignment assignment)
        SourceDisposition.visibleDefinitionColumns := by
    intro column member
    rw [SourceDisposition.visibleDefinitionColumns] at member
    rcases List.mem_append.mp member with input | visible
    · exact PhysicalAgreement.inputAgreement assignment column input
    · exact visibleAgreement column visible
  have sourceRowsSatisfy :=
    SourceSatisfaction.generatedEmittedRowsSatisfy_implies_generatedSourceRowsSatisfy
      selectedRows selectorOne constantOne allVisibleAgreement
  exact SourceRowsSoundness.sourceRowsSatisfy_implies_consequences
    (PhysicalAgreement.reconstructed_canonical assignment)
    (PhysicalAgreement.reconstructed_constantOne constantOne)
    sourceRowsSatisfy

end Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.SelectedRowsSoundness
