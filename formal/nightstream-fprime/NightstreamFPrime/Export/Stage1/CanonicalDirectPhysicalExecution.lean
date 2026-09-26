import NightstreamFPrime.Export.Stage1.StoredDirectPhysicalExecution
import NightstreamFPrime.Export.Stage1.PiRLCCombinationEventReadSupport
import NightstreamFPrime.Export.Stage1.PiRLCCombinationPermutationReadSupport
import NightstreamFPrime.Export.Stage1.PiRLCFirst54ReadSupport
import NightstreamFPrime.Export.Stage1.PiRLCCombinationScratchCustody

/-!
Construct the selected stored witness without PiRLC product scratch execution.
Canonical event provenance supplies every replacement and read-support fact.
The actual array loop preserves rejection, the complete CCS carrier, and its
public digest. No completed physical witness is an input to the constructor.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.CanonicalDirectPhysicalExecution

open NightstreamFPrime.Spec
open NightstreamFPrime.Export.Package
open StoredDirectPhysicalExecution (EventSafe Outside)
open StoredPhysicalExecution (ResultAgree)

local notation "application" => Poseidon2HashChainV1Package.application
local notation "context" => PerApplicationCachedShift.Context.ofProgram application

private theorem combination_template_lt (invocation : CompactRowInvocation)
    (member : invocation ∈ PackagePlan.canonicalCombinationBlock.expand) :
    invocation.templateIndex < 2 * ringDegree := by
  rw [PackagePlan.canonicalCombinationBlock_expand,
    ← PiRLCProductSchedule.compactInvocations_eq] at member
  rcases List.mem_ofFn.mp member with ⟨index, rfl⟩
  rw [congrFun PiRLCProductSchedule.compactInvocation_eq_descriptor index]
  generalize PiRLCProductSchedule.descriptor index = descriptor
  rcases descriptor with ⟨family, source, block, lane, cell⟩
  cases family <;>
    change PiRLCCombinationTemplates.templateIndex source.val lane.val < 2 * ringDegree
  all_goals
    simpa only [PiRLCCombinationTemplates.templates_length, ringDegree] using
      PiRLCCombinationTemplates.templateIndex_lt source.val lane

private theorem compact_safe (templates : Array CompactRowTemplate)
    (canonical : templates.toList = Data.compactRowTemplates ())
    (block : PackagePlan.CompactInvocationBlock)
    (blockMember : block ∈ PackagePlan.canonicalCompactBlocks)
    (invocation : { value : CompactRowInvocation // value ∈ block.expand }) :
    EventSafe context (PilotData.circuitPackage ()) templates
      (StoredPhysicalPlan.compactEvent context templates canonical block invocation) := by
  dsimp only [StoredPhysicalPlan.compactEvent, EventSafe]
  split_ifs with product
  · exact PiRLCCombinationInvocationOrigin.shifted_origin context block blockMember
      invocation.val invocation.property product
  · simp only [PackagePlan.canonicalCompactBlocks, List.mem_cons,
      List.not_mem_nil, or_false] at blockMember
    rcases blockMember with rfl | rfl
    · exact PiRLCFirst54ReadSupport.canonical_supported context templates canonical
        invocation.val invocation.property _
    · exact False.elim (product (combination_template_lt invocation.val invocation.property))

private theorem canonical_sources_safe (templates : Array CompactRowTemplate)
    (canonical : templates.toList = Data.compactRowTemplates ()) :
    StoredPhysicalPlan.EventSources templates canonical
      (EventSafe context (PilotData.circuitPackage ()) templates) := by
  refine {
    hashes := ?_
    permutations := ?_
    applicationPermutations := ?_
    compact := compact_safe templates canonical
    pilotBatches := PiRLCCombinationEventReadSupport.pilotBatches_supported templates
    piCcsBatches := PiRLCCombinationEventReadSupport.piCcsBatches_supported templates
    witnessBatches := PiRLCCombinationEventReadSupport.witnessBatches_supported templates
    applicationBatches := PiRLCCombinationEventReadSupport.applicationBatches_supported templates
    pilotInstructions := PiRLCCombinationEventReadSupport.pilotInstructions_supported templates
    ordinaryInstructions := PiRLCCombinationEventReadSupport.ordinaryInstructions_supported templates
    applicationInstructions := PiRLCCombinationEventReadSupport.applicationInstructions_supported templates }
  · exact PiRLCCombinationPermutationReadSupport.hashes_supported
  · exact PiRLCCombinationPermutationReadSupport.permutations_supported
  · exact PiRLCCombinationPermutationReadSupport.applicationPermutations_supported

/-- Every event of the actual sorted plan has its canonical replacement contract. -/
theorem events_safe (plan : StoredPhysicalPlan.Plan) :
    ∀ event ∈ plan.events, EventSafe context plan.pilot.val plan.templates event := by
  rw [plan.pilot.property]
  exact plan.events_induction _ (canonical_sources_safe plan.templates plan.templates_eq)

/-- Execute only the direct selected producer, starting from the caller-seeded array. -/
def execute (plan : StoredPhysicalPlan.Plan) (initial : Array F) : Except String (Array F) :=
  StoredPhysicalExecution.runWith
    (StoredDirectPhysicalExecution.executeEvent plan.pilot.val plan.templates) plan.events initial

/-- Exact rejection or retained-array agreement, with no caller-supplied support premise. -/
theorem execute_agree (plan : StoredPhysicalPlan.Plan) (initial : Array F) :
    ResultAgree Outside
      (StoredPhysicalExecution.runWith
        (StoredPhysicalExecution.executeEvent plan.pilot.val plan.templates) plan.events initial)
      (execute plan initial) := by
  apply StoredPhysicalExecution.runWith_agree
  · intro event member left right agree
    exact StoredDirectPhysicalExecution.executeEvent_agree context plan.pilot.val plan.templates
      plan.templates_eq event left right (events_safe plan event member) agree
  · exact ⟨rfl, fun _ _ => rfl⟩

/-- Successful production gives exactly the same complete CCS witness and public digest. -/
theorem successful_assignment_eq (plan : StoredPhysicalPlan.Plan) (initial full direct : Array F)
    (fullResult : StoredPhysicalExecution.runWith
      (StoredPhysicalExecution.executeEvent plan.pilot.val plan.templates) plan.events initial =
        .ok full)
    (directResult : execute plan initial = .ok direct) :
    (PerApplicationAssignmentTransportExecution.canonicalRawValues application
      (fun column => StoredWitnessExecution.asEnv full column.val)).completeAssignment =
      (PerApplicationAssignmentTransportExecution.canonicalRawValues application
        (fun column => StoredWitnessExecution.asEnv direct column.val)).completeAssignment ∧
    (PerApplicationAssignmentTransportExecution.canonicalRawValues application
      (fun column => StoredWitnessExecution.asEnv full column.val)).outputDigest =
      (PerApplicationAssignmentTransportExecution.canonicalRawValues application
        (fun column => StoredWitnessExecution.asEnv direct column.val)).outputDigest := by
  have agrees := execute_agree plan initial
  rw [fullResult, directResult] at agrees
  change StoredExecutionSupport.Agree Outside full direct at agrees
  have baseAgree : ∀ column : Fin (PiRLCProductPlan.baseSourceWidth application),
      PiRLCCombinationScratchCustody.Outside column.val →
        StoredWitnessExecution.asEnv full column.val =
          StoredWitnessExecution.asEnv direct column.val := by
    intro column outside
    exact agrees.2 column.val outside
  exact ⟨PiRLCCombinationScratchCustody.completeAssignment_congr application _ _ baseAgree,
    PiRLCCombinationScratchCustody.canonicalOutputDigest_congr application _ _ baseAgree⟩

end NightstreamFPrime.Export.Stage1.CanonicalDirectPhysicalExecution
