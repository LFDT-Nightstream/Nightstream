import NightstreamFPrime.Export.Stage1.PiRLCCombinationPiCCSReadSupport
import NightstreamFPrime.Export.Stage1.PiDECOrdinaryDirectSource
import NightstreamFPrime.Export.Stage1.ApplicationDirectSource
import NightstreamFPrime.Export.Stage1.OrdinaryRowPlan
import NightstreamFPrime.Layout.Stage1.RunningTransitionLoweringSupport

/-! Read support for all canonical ordinary-row witness instructions. -/

namespace NightstreamFPrime.Export.Stage1.PiRLCCombinationOrdinaryReadSupport

open NightstreamFPrime.Circuit
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.Stage1
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Export.Package
open PiRLCCombinationWitnessReadSupport (Outside)
open PiRLCCombinationReadSupport

private theorem samplerSource_before (column : Nat)
    (source : PiRLCSamplerOrdinaryDirectSource.Source column) :
    column < PiRLCStarts.commitmentFreshStart := by
  change column < 21124348
  cases source with
  | poseidon source round lane sourceLt roundLt =>
      have laneLt := lane.isLt
      change source < 17 at sourceLt
      change round < 8 at roundLt
      unfold PiRLCSamplerOrdinaryDirectSource.poseidonSource
      cases round with
      | zero =>
          norm_num [PiRLCStarts.samplerSourceLogicalStart, PiRLCStarts.samplerLogicalStart,
            PiRLCStarts.phaseLogicalStart, PiRLCInputs.phaseOffset, PiRLC.v1_1.Formal.samplerOffset]
          omega
      | succ previous =>
          norm_num [PiRLC.v1_1.DigestWindow.permutationOffset,
            PiRLC.v1_1.Sampler.windowOffset, PiRLC.v1_1.Sampler.windowBase,
            PiRLC.v1_1.SamplerChain.sourceOffset, PiRLCStarts.samplerLogicalStart,
            PiRLCStarts.phaseLogicalStart, PiRLCInputs.phaseOffset,
            PiRLC.v1_1.Formal.samplerOffset, PiRLC.v1_1.Sampler.logicalPrivateCount,
            PiRLC.v1_1.Sampler.entryPrivateCount, PiRLC.v1_1.DigestWindow.logicalPrivateCount,
            PiRLC.v1_1.DigestLane.logicalPrivateCount]
          omega
  | logical source round lane position sourceLt roundLt laneLt positionLt =>
      change source < 17 at sourceLt
      change round < 8 at roundLt
      norm_num [PiRLCStarts.digestLaneLogicalStart, PiRLCStarts.windowLogicalStart,
        PiRLCStarts.samplerSourceLogicalStart, PiRLCStarts.samplerLogicalStart,
        PiRLCStarts.phaseLogicalStart, PiRLCInputs.phaseOffset, PiRLC.v1_1.Formal.samplerOffset]
      omega
  | fresh source round lane position sourceLt roundLt laneLt positionLt =>
      change source < 17 at sourceLt
      change round < 8 at roundLt
      norm_num [PiRLCStarts.digestLaneFreshStart, PiRLCStarts.windowFreshStart,
        PiRLCStarts.samplerSourceFreshStart, PiRLCStarts.samplerFreshStart,
        PiRLCStarts.phaseFreshStart_eq]
      omega
  | selector source sourceLt =>
      change source < 17 at sourceLt
      norm_num [PiRLCSamplerOrdinaryDirectSource.selectorSource,
        Gadgets.Sampling.First54.positionOffset, Gadgets.Sampling.First54.candidateCount,
        Gadgets.Sampling.First54.roundPrivateCount, Gadgets.Sampling.First54.fullSlot,
        Gadgets.Sampling.First54Step.slotCount, Gadgets.Sampling.First54Step.fullSlot,
        Gadgets.Sampling.First54ValueStep.outputCount,
        PiRLCStarts.selectorLogicalStart, PiRLCStarts.samplerSourceLogicalStart,
        PiRLCStarts.samplerLogicalStart, PiRLCStarts.phaseLogicalStart,
        PiRLCInputs.phaseOffset, PiRLC.v1_1.Formal.samplerOffset]
      omega

private theorem runningSource_outside (column : Nat)
    (source : RunningTransitionSourceSupport.Source column) :
    Outside (Spartan.sourceToSpartan column) := by
  apply mapped_outside column (RunningTransitionSourceSupport.source_lt_sourceColumnCount source)
  rcases source with (state | output | roundPoint | piDec) | localRange
  · left
    have upper := state.2
    change column < 28 + 11 at upper
    change column < 21124348
    omega
  · left
    have upper := output.2
    change column < 49663 + 49393 at upper
    change column < 21124348
    omega
  · left
    rcases roundPoint with ⟨coordinate, c0 | c1⟩
    all_goals subst column
    all_goals
      have bound := coordinate.isLt
      change coordinate.val < 28 at bound
      rw [PiCCSStarts.roundTranscriptWitnessStart_eq]
      norm_num [RunningTransitionInputs.roundStride, RunningTransitionInputs.roundSampleC0Offset,
        RunningTransitionInputs.roundSampleC1Offset, PiRLCStarts.commitmentFreshStart_eq]
      omega
  · right
    have lower := (RunningTransitionSourceSupport.piDecField_inRange piDec).1
    exact lower
  · right
    exact Nat.le_trans (by decide) localRange.1

private theorem applicationSource_outside (application : Stage1.Application.Program)
    (column : Nat) (source : ApplicationDirectSource.SourceAllowed application column) :
    Outside column := by
  rcases source with input | witness | output | localRange
  · rcases input with ⟨index, rfl⟩
    rw [ApplicationInputs.inputColumn_value]
    have bound := index.isLt
    change index.val < 4 at bound
    left
    change 35 + index.val < 21124070
    omega
  · rcases witness with ⟨index, rfl⟩
    right
    unfold ApplicationInputs.witnessColumn ApplicationInputs.witnessStart
    rw [Spartan.privateColumnCount_eq]
    change 28972970 ≤ 29336446 + index.val
    omega
  · rcases output with ⟨index, rfl⟩
    rw [ApplicationInputs.outputColumn_value]
    have bound := index.isLt
    change index.val < 4 at bound
    left
    change 49428 + index.val < 21124070
    omega
  · right
    have lower := localRange.1
    unfold ApplicationInputs.localStart ApplicationInputs.witnessStart at lower
    rw [Spartan.privateColumnCount_eq] at lower
    change 28972970 ≤ column
    omega

theorem arithmeticRows_supported
    (relation : ProductionKey.LogicalRelation Data.logicalWidth Data.publicFits) :
    ∀ row ∈ (Data.arithmeticRows ()).map Rows.CompiledRow.toR1CS,
      row.VarsSatisfy Outside := by
  intro row member
  simp only [Data.arithmeticRows, List.map_append, List.mem_append] at member
  rcases member with ((piCcs | sampler) | piDec) | running
  · have support := PiCCSOrdinaryDirectSupport.sourceRows_varsSatisfy relation row piCcs
    apply support.mono row
    intro column source
    rcases source with ⟨original, support, rfl⟩
    exact PiRLCCombinationPiCCSReadSupport.source_outside original support
  · have support := PiRLCSamplerOrdinaryDirectSource.sourceRows_varsSatisfy row sampler
    apply support.mono row
    intro column source
    rcases source with ⟨original, support, rfl⟩
    exact mapped_before original (samplerSource_before original support)
  · rw [← PiDECOrdinaryDirectSource.sourceRows_eq_canonical] at piDec
    simp only [PiDECOrdinaryDirectSource.sourceRows, List.mem_append] at piDec
    have support : row.VarsSatisfy PiDECSourceSupport.Target := by
      rcases piDec with ((publicInput | commitment) | evalK) | evalA
      · exact PiDECOrdinaryDirectSource.publicRows_varsSatisfy relation row publicInput
      · exact PiDECOrdinaryDirectSource.commitmentRows_varsSatisfy relation row commitment
      · exact PiDECOrdinaryDirectSource.evalKRows_varsSatisfy relation row evalK
      · exact PiDECOrdinaryDirectSource.evalARows_varsSatisfy relation row evalA
    apply support.mono row
    intro column source
    rcases source with ⟨original, support, rfl⟩
    exact PiRLCCombinationWitnessReadSupport.piDecSource_outside original support
  · rw [RunningTransitionArithmetic.Plan.rows_to_layout _ _
      (RunningTransitionArithmetic.canonicalPlan_matches Data.logicalWidth Data.publicFits)] at running
    have support := RunningTransitionSourceSupport.remappedRows_varsSatisfy relation row running
    apply support.mono row
    intro column source
    rcases source with ⟨original, support, rfl⟩
    exact runningSource_outside original support

theorem canonicalInstructions_supported
    (relation : ProductionKey.LogicalRelation Data.logicalWidth Data.publicFits)
    (context : PerApplicationCachedShift.Context)
    (block : OrdinaryRowPlan.Block) (blockMember : block ∈ OrdinaryRowPlan.canonicalBlocks ())
    (instruction : WitnessInstruction)
    (member : instruction ∈ Rows.witnessInstructionsTR (block.rows Data.logicalWidth Data.publicFits)) :
    (PerApplicationCachedShift.shiftWitnessInstruction context instruction).a.toR1CS.VarsSatisfy
        Outside ∧
      (PerApplicationCachedShift.shiftWitnessInstruction context instruction).b.toR1CS.VarsSatisfy
        Outside := by
  have selected : instruction ∈ (OrdinaryRowPlan.canonicalBlocks ()).flatMap
      (fun block => Rows.witnessInstructionsTR (block.rows Data.logicalWidth Data.publicFits)) :=
    List.mem_flatMap.mpr ⟨block, blockMember, member⟩
  rw [OrdinaryRowPlan.canonicalWitnessInstructions_expand] at selected
  exact shiftInstruction_supported context instruction
    (instruction_supported _ Outside (arithmeticRows_supported relation) instruction selected)

theorem applicationInstructions_supported (application : Stage1.Application.Program)
    (instruction : WitnessInstruction)
    (member : instruction ∈ (PerApplicationPackage.directApplicationPlan application).witnessInstructions) :
    instruction.a.toR1CS.VarsSatisfy Outside ∧ instruction.b.toR1CS.VarsSatisfy Outside := by
  rw [PerApplicationPackage.directApplicationPlan_eq_applicationPlan] at member
  apply instruction_supported _ Outside _ instruction member
  intro row member
  have support := ApplicationDirectSource.sourceRows_varsSatisfy application row member
  exact support.mono row (applicationSource_outside application)

end NightstreamFPrime.Export.Stage1.PiRLCCombinationOrdinaryReadSupport
