import NightstreamFPrime.Export.Stage1.PiRLCSamplerCompleteness
import NightstreamFPrime.Export.Stage1.PoseidonRetainedBlock

/-!
All retained Poseidon source columns precede the PiRLC combination scratch.
The bounds use the fixed phase schedules, not an expanded invocation list.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.PiRLCCombinationScratchPoseidon

open NightstreamFPrime.Export.Package
open NightstreamFPrime.Gadgets.Poseidon2
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Layout.Stage1
open NightstreamFPrime.Lifecycle

theorem priorWitnessEnd_le
    (invocation : Fin PoseidonRetainedBlock.priorInvocationCount) :
    PoseidonRetainedBlock.priorWitnessStart invocation +
      PoseidonScheduleTrace.localColumnCount ≤ 20198590 := by
  have bound : invocation.val < 12350 := invocation.isLt
  change 128074 + invocation.val * 592 + 592 ≤ 20198590
  omega

theorem outputWitnessEnd_le
    (invocation : Fin PoseidonRetainedBlock.outputInvocationCount) :
    PoseidonRetainedBlock.outputWitnessStart invocation +
      PoseidonScheduleTrace.localColumnCount ≤ 20198590 := by
  have bound : invocation.val < 12350 := invocation.isLt
  change 7439538 + invocation.val * 592 + 592 ≤ 20198590
  omega

private theorem piCcs_witnessEnd_le (invocation : PermutationInvocation)
    (member : invocation ∈
      PiCCSInvocations.invocations Data.logicalWidth Data.publicFits) :
    invocation.witnessStart + 592 ≤ 20198590 := by
  let relation : ProductionKey.LogicalRelation
      Data.logicalWidth Data.publicFits :=
    { matrices := fun _ _ _ => 0
      cubeFits := by
        norm_num [Data.logicalWidth,
          NightstreamFPrime.Export.Stage1.PhaseReference.logicalWidth,
          Spec.Folding.PiCCS.PaperJoint.Phi81CarrierLayout.carrierWidth,
          Spec.Folding.PiCCS.PaperJoint.Phi81ColumnLayout.blockCount,
          Spec.ringDegree, cubeVariables] }
  have bound := (PiCCSInvocations.invocations_scheduleWithin
    Data.logicalWidth Data.publicFits relation).2 invocation member
  rw [PiCCSInvocations.invocationCeiling_eq] at bound
  omega

private theorem sampler_witnessEnd_le (invocation : PermutationInvocation)
    (member : invocation ∈ PiRLCSamplerInvocations.invocations
      (logicalWidth := Data.logicalWidth) (publicFits := Data.publicFits)) :
    invocation.witnessStart + 592 ≤ 20198590 := by
  unfold PiRLCSamplerInvocations.invocations at member
  rcases List.mem_flatMap.mp member with ⟨source, sourceMember, sourceMember'⟩
  have sourceBound : source < 17 := List.mem_range.mp sourceMember
  unfold PiRLCSamplerInvocations.sourceInvocations at sourceMember'
  rcases List.mem_append.mp sourceMember' with entry | advance
  · rw [PiRLCSamplerCompleteness.entryInvocations_eq_singleton] at entry
    rcases List.mem_singleton.mp entry with rfl
    change Spartan.sourceToSpartan (20064823 + source * 3259) + 592 ≤ 20198590
    rw [Spartan.sourceToSpartan_add_of_piCcsLocal _ _ (by decide)]
    change 20064545 + source * 3259 + 592 ≤ 20198590
    omega
  · rw [List.mem_singleton] at advance
    subst invocation
    simp only [PiRLCSamplerInvocations.advanceInvocation, Invocations.invocation]
    change Spartan.sourceToSpartan (20064823 + source * 3259 + 2613) + 592 ≤ 20198590
    rw [show 20064823 + source * 3259 + 2613 = 20064823 + (source * 3259 + 2613) by omega,
      Spartan.sourceToSpartan_add_of_piCcsLocal _ _ (by decide)]
    change 20064545 + (source * 3259 + 2613) + 592 ≤ 20198590
    omega

theorem laterWitnessEnd_le
    (invocation : Fin PoseidonRetainedBlock.laterInvocationCount) :
    PoseidonRetainedBlock.laterWitnessStart invocation +
      PoseidonScheduleTrace.localColumnCount ≤ 20198590 := by
  let index : Fin PoseidonRetainedBlock.basePackage.permutationInvocations.length :=
    ⟨invocation.val, by
      simpa only [PoseidonRetainedBlock.basePackage_permutationInvocations_length]
        using invocation.isLt⟩
  let selected := PoseidonRetainedBlock.basePackage.permutationInvocations.get index
  change selected.witnessStart + 592 ≤ 20198590
  have member := List.get_mem PoseidonRetainedBlock.basePackage.permutationInvocations index
  change selected ∈ PoseidonRetainedBlock.basePackage.permutationInvocations at member
  rw [PoseidonRetainedBlock.basePackage_permutationInvocations_eq,
    Data.permutationInvocations_eq, List.mem_append] at member
  rcases member with member | member
  · exact piCcs_witnessEnd_le selected member
  · exact sampler_witnessEnd_le selected member

end NightstreamFPrime.Export.Stage1.PiRLCCombinationScratchPoseidon
