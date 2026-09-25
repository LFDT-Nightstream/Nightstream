import NightstreamFPrime.Export.Stage1.CheckedReplayComposition
import NightstreamFPrime.Export.Stage1.CheckedReplayHandoff
import NightstreamFPrime.Export.Stage1.FreshRowsCheck
import NightstreamFPrime.Export.Stage1.CheckedReplaySuccessor
import NightstreamFPrime.Export.Stage1.PiDECComputedChildren
import NightstreamFPrime.Export.Stage1.FreshCommitmentFold
import NightstreamFPrime.Export.Stage1.CheckedReplayStep
import tests.AxiomAudit

set_option autoImplicit false

namespace LeanGraph.Targets

open NightstreamFPrime.Export.Stage1
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Layout.Stage1
open NightstreamFPrime.Spec.Folding
open Poseidon2HashChainV1Package (application fits)
open Poseidon2HashChainV1Setup (productionAjtaiKey)

/-- The supplied C proof, actual post-output R sample, computed public parent,
and accepted D fields determine the exact selected NIFS return. This target
does not assert source opening validity or an accepted successor envelope. -/
def CheckedReplayNifsResult : Prop :=
  ∀ (input : PiCCSInputCheck.Input) (batch : PiRLCParent.Batch)
    (parent : PiRLCParent.Values) (messages : PiDECInputCheck.Messages),
    PiRLCInputCheck.sampled input = some batch →
    PiRLCParent.computedParent input batch = some parent →
    PiDECInputCheck.accepted parent messages = true →
    Nifs.PaperNonInteractive.verify
      (ProductionKey.key (PerApplicationFixedPoint.relation application fits) productionAjtaiKey)
      (PiCCSInputCheck.running input) (PiCCSInputCheck.fresh input)
      (CheckedReplayNifs.proof input messages) =
      some (PiCCSInputCheck.runningFromInput messages)

theorem checkedReplayNifsResult : CheckedReplayNifsResult := by
  intro input batch parent messages sampled returned checked
  exact CheckedReplayNifs.checked_verifies_selected input batch parent messages
    sampled returned checked

#audit_axioms NightstreamFPrime.Export.Stage1.CheckedReplayNifs.checked_verifies
#audit_axioms NightstreamFPrime.Export.Stage1.CheckedReplayNifs.checked_verifies_selected
#audit_axioms NightstreamFPrime.Export.Stage1.CheckedReplayStep.checked_step
#audit_axioms checkedReplayNifsResult

end LeanGraph.Targets

#audit_axioms NightstreamFPrime.Export.Stage1.CheckedReplaySuccessor.output_unique
#audit_axioms NightstreamFPrime.Export.Stage1.CheckedReplaySuccessor.accepted_of_checked_rows
#audit_axioms NightstreamFPrime.Export.Stage1.HyperNovaAcceptedNext.freshHolds_of_rows
#audit_axioms NightstreamFPrime.Export.Stage1.HyperNovaAcceptedNext.terminal_of_memberships
#audit_axioms NightstreamFPrime.Export.Stage1.PiDECComputedChildren.child_openings
#audit_axioms NightstreamFPrime.Export.Stage1.FreshCommitmentFold.fold_value
#audit_axioms NightstreamFPrime.Export.Stage1.FreshCommitmentFold.completeRow_value

#audit_axioms NightstreamFPrime.Export.Stage1.CheckedReplayHandoff.prior_eq_payload
#audit_axioms NightstreamFPrime.Export.Stage1.FreshRowsCheck.checkBlock_sound
#audit_axioms NightstreamFPrime.Export.Stage1.FreshRowsCheck.rowsZero_of_blockChecks
#audit_axioms NightstreamFPrime.Export.Stage1.FreshRowsCheck.canonical_blocks_rowsZero
#audit_axioms NightstreamFPrime.Export.Stage1.FreshRowsCheck.checkProgram_rowsZero
#audit_axioms NightstreamFPrime.Export.Stage1.FreshRowsCheck.rowsZero_allVertices
#audit_axioms NightstreamFPrime.Export.Stage1.FreshRowsCheck.field_bounded
#audit_axioms NightstreamFPrime.Export.Stage1.FreshRowsCheck.checked_raw

#audit_axioms NightstreamFPrime.Export.Stage1.CheckedReplayParent.evaluations_eq_family

#audit_axioms NightstreamFPrime.Export.Stage1.CheckedReplayParent.returned_witness

#audit_axioms NightstreamFPrime.Export.Stage1.CheckedReplayParent.parent_opening

#audit_axioms NightstreamFPrime.Export.Stage1.CheckedReplayComposition.accepted_and_handoff

#audit_axioms NightstreamFPrime.Export.Stage1.FreshRowsCheck.checkBlockRange_unit

#audit_axioms NightstreamFPrime.Export.Stage1.FreshRowsCheck.checkBlock_of_units

#audit_axioms NightstreamFPrime.Export.Stage1.FreshRowsCheck.checkBlock_of_ranges

#audit_axioms NightstreamFPrime.Export.Stage1.FreshRowsCheck.ceiling_ranges_cover

#audit_axioms NightstreamFPrime.Export.Stage1.FreshRowsCheck.checkBlock_of_workerRanges
