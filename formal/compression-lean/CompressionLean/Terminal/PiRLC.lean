import NightstreamFPrime.Layout.Stage1.PiRLCInputBounds
import CompressionLean.Terminal.Rows

/-!
Owns the PiRLC edge of the terminal package: its rows imply the exact
seven-child PiRLC specification and phase, with the sampler challenges and
the combined parent.

The proofs are F′'s `Export/Stage1/Package` proofs with the terminal package
in place of the Stage 1 package. Only the membership steps differ.
-/

namespace CompressionLean.Terminal

open NightstreamFPrime.Export
open NightstreamFPrime.Export.Package
open NightstreamFPrime.Export.Stage1
open NightstreamFPrime.Circuit
open NightstreamFPrime.Gadgets.Poseidon2
open NightstreamFPrime.Layout
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Lifecycle.PiRLC.v1_2
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint

variable {lw : Nat} {pf : ringDegree * publicRingColumns ≤ Phi81CarrierLayout.carrierWidth lw}

theorem piRlcSamplerInvocations_hold (env : Env)
    (holds : (circuitPackage ()).RowsHold env) :
    ∀ invocation ∈
      PiRLCSamplerInvocations.invocations
        (logicalWidth := lw) (publicFits := pf),
      PermutationInvocationHolds (PilotData.circuitPackage ()) invocation env := by
  intro invocation member
  apply permutationInvocation_holds env holds invocation
  rw [permutationInvocations_at lw pf]
  exact List.mem_append_right _ member

private theorem piRlcEntry (env : Env)
    (holds : (circuitPackage ()).RowsHold env)
    (source : Nat) (sourceLt : source < PiRLCSamplerInvocations.sourceCount) :
    TranscriptAbsorption.SpecHolds
      (PiRLCSamplerInvocations.sourceInterface
        (logicalWidth := lw) (publicFits := pf) source)
      source (PiRLCSamplerInvocations.sourceLogicalStart source)
      (NightstreamFPrime.Layout.Stage1.Spartan.pullback env) := by
  apply PiRLCSamplerInvocations.entryTrace_implies_spec source env
  intro invocation member
  apply piRlcSamplerInvocations_hold env holds invocation
  unfold PiRLCSamplerInvocations.invocations
  apply List.mem_flatMap.mpr
  refine ⟨source, List.mem_range.mpr sourceLt, ?_⟩
  unfold PiRLCSamplerInvocations.sourceInvocations
    PiRLCSamplerInvocations.entryInvocations
  exact List.mem_append_left _ member

private theorem piRlcAdvance (env : Env)
    (holds : (circuitPackage ()).RowsHold env)
    (source : Nat) (sourceLt : source < PiRLCSamplerInvocations.sourceCount) :
    Permutation.Owned.SpecHolds
      (Sampler.advanceInterface
        (PiRLCSamplerInvocations.sourceInterface (logicalWidth := lw)
          (publicFits := pf) source) source
        (PiRLCSamplerInvocations.sourceLogicalStart source))
      (NightstreamFPrime.Layout.Stage1.PiRLCStarts.advanceLogicalStart source)
      (NightstreamFPrime.Layout.Stage1.Spartan.pullback env) := by
  apply PiRLCSamplerInvocations.advanceInvocation_implies_spec source env
  apply piRlcSamplerInvocations_hold (lw := lw) (pf := pf) env holds
  apply List.mem_flatMap.mpr
  refine ⟨source, List.mem_range.mpr sourceLt, ?_⟩
  simp [PiRLCSamplerInvocations.sourceInvocations]

private theorem samplerSourceRows (env : Env)
    (holds : (circuitPackage ()).RowsHold env)
    (source : Nat) (sourceLt : source < PiRLCSamplerInvocations.sourceCount) :
    R1CS.RowsHold env ((PiRLCSamplerOrdinaryRows.rangeRows
      (logicalWidth := lw) (publicFits := pf)
      source).map Rows.CompiledRow.toR1CS) ∧
    R1CS.RowsHold env ((PiRLCSamplerOrdinaryRows.wordRows source).map
      Rows.CompiledRow.toR1CS) := by
  have all := piRlcSamplerOrdinaryRows_hold (lw := lw) (pf := pf) env holds
  have selected : R1CS.RowsHold env
      ((PiRLCSamplerOrdinaryRows.sourceRows (logicalWidth := lw)
        (publicFits := pf) source).map Rows.CompiledRow.toR1CS) := by
    intro row member
    obtain ⟨compiled, inside, rfl⟩ := List.mem_map.mp member
    apply all
    apply List.mem_map.mpr
    refine ⟨compiled, ?_, rfl⟩
    exact List.mem_flatMap.mpr ⟨source, List.mem_range.mpr sourceLt, inside⟩
  rw [PiRLCSamplerOrdinaryRows.sourceRows, List.map_append] at selected
  exact (R1CS.rowsHold_append env _ _).mp selected

private theorem piRlcSamplerSpec (env : Env)
    (holds : (circuitPackage ()).RowsHold env)
    (source : Nat) (sourceLt : source < PiRLCSamplerInvocations.sourceCount)
    (assumptions : Sampler.Assumptions
      (PiRLCSamplerInvocations.sourceInterface (logicalWidth := lw)
        (publicFits := pf) source)
      (PiRLCSamplerInvocations.sourceLogicalStart source)) :
    Sampler.SpecHolds
      (PiRLCSamplerInvocations.sourceInterface (logicalWidth := lw)
        (publicFits := pf) source)
      source (PiRLCSamplerInvocations.sourceLogicalStart source)
      (NightstreamFPrime.Layout.Stage1.Spartan.pullback env) := by
  have ordinary := samplerSourceRows (lw := lw) (pf := pf) env holds source sourceLt
  have range := PiRLCSamplerOrdinaryRows.rangeRows_imply_spec source env
    (by
      rw [PiRLCSamplerOrdinaryRows.rangeInterface_eq]
      exact Sampler.range_inputs _ _ _ assumptions)
    ordinary.1
  rw [PiRLCSamplerOrdinaryRows.rangeInterface_eq] at range
  apply Sampler.soundness _ _ _ _ assumptions
  intro operation member
  simp only [Sampler.opsAt, List.mem_cons, List.not_mem_nil, or_false] at member
  rcases member with rfl | rfl | rfl | rfl
  · intro _
    exact piRlcEntry env holds source sourceLt
  · intro _
    exact range
  · intro _
    exact piRlcAdvance env holds source sourceLt
  · intro _
    exact PiRLCSamplerOrdinaryRows.wordRows_imply_spec source env ordinary.2

theorem piRlcSamplerChain (env : Env)
    (holds : (circuitPackage ()).RowsHold env)
    (assumptions : SamplerChain.Assumptions
      (PiRLCSamplerRows.samplerInterface (logicalWidth := lw)
        (publicFits := pf))
      NightstreamFPrime.Layout.Stage1.PiRLCStarts.samplerLogicalStart) :
    SamplerChain.SpecHolds
      (PiRLCSamplerRows.samplerInterface (logicalWidth := lw)
        (publicFits := pf))
      NightstreamFPrime.Layout.Stage1.PiRLCStarts.samplerLogicalStart
      (NightstreamFPrime.Layout.Stage1.Spartan.pullback env) := by
  apply SamplerChain.soundness
    (PiRLCSamplerRows.samplerInterface (logicalWidth := lw)
      (publicFits := pf))
    (NightstreamFPrime.Layout.Stage1.Spartan.pullback env)
    NightstreamFPrime.Layout.Stage1.PiRLCStarts.samplerLogicalStart
  · exact @assumptions
  intro operation member
  change operation ∈ (List.range SamplerChain.sourceCount).map
    (SamplerChain.childOp _ _) at member
  obtain ⟨source, inside, rfl⟩ := List.mem_map.mp member
  change Sampler.Assumptions _ _ → Sampler.SpecHolds _ _ _ _
  intro childInputs
  have sourceLt : source < PiRLCSamplerInvocations.sourceCount := by
    simpa [PiRLCSamplerInvocations.sourceCount, SamplerChain.sourceCount_eq]
      using List.mem_range.mp inside
  exact piRlcSamplerSpec env holds source sourceLt childInputs

theorem piRlcCombinationTemplateSelection :
    Package.PiRLCCombinationTemplateSelection (circuitPackage ()) := by
  intro source lane
  change PiRLCCombinationTemplates.templates[
      PiRLCCombinationTemplates.templateIndex source.val lane.val]? = _
  simpa [PiRLCCombinationInvocations.firstSource] using
    PiRLCCombinationTemplates.template_getElem? source.val lane

private theorem familyInvocationRows (env : Env)
    (holds : (circuitPackage ()).RowsHold env)
    (logicalStart rowStart freshStart blockCount cellCount valueStride : Nat)
    [NeZero cellCount] (valueSourceStart : Nat → Nat → Nat → Nat)
    (member : ∀ source : Fin PiRLCCombinationInvocations.sourceCount,
      ∀ index : Fin (CombinationStep.privateCount blockCount cellCount),
        let coordinates := CombinationStep.coordinates index
        PiRLCCombinationInvocations.invocation logicalStart rowStart freshStart
            blockCount cellCount valueStride source.val coordinates.1.val
              coordinates.2.1.val coordinates.2.2.val valueSourceStart ∈
          Data.compactRowInvocations ()) :
    PiRLCCombinationConformance.FamilyInvocationRowsHold logicalStart rowStart
      freshStart blockCount cellCount valueStride valueSourceStart env := by
  intro source index
  let coordinates := CombinationStep.coordinates index
  let selected := PiRLCCombinationInvocations.invocation logicalStart rowStart
    freshStart blockCount cellCount valueStride source.val coordinates.1.val
      coordinates.2.1.val coordinates.2.2.val valueSourceStart
  have packageRows := compactInvocation_holds env holds selected (member source index)
  unfold CompactRowInvocationHolds at packageRows
  have selectedIndex : selected.templateIndex =
      PiRLCCombinationTemplates.templateIndex source.val
        coordinates.2.1.val := by
    rfl
  rw [selectedIndex, piRlcCombinationTemplateSelection source coordinates.2.1]
    at packageRows
  dsimp only
  rw [CompactRows.instantiateRows_eq_package]
  exact packageRows

/-- The four PiRLC combination families hold with their Lean templates. -/
theorem piRlcCombinationRows (env : Env)
    (holds : (circuitPackage ()).RowsHold env) :
    Package.PiRLCCombinationRowsHold env := by
  constructor
  · apply familyInvocationRows env holds
    intro source index
    rw [Data.compactRowInvocations_eq, PiRLCCombinationInvocations.invocations]
    apply List.mem_append_left
    apply List.mem_append_left
    apply List.mem_append_left
    exact PiRLCCombinationInvocations.familyInvocation_mem _ _ _ _ _ _ _ source index
  · apply familyInvocationRows env holds
    intro source index
    rw [Data.compactRowInvocations_eq, PiRLCCombinationInvocations.invocations]
    apply List.mem_append_left
    apply List.mem_append_left
    apply List.mem_append_right
    exact PiRLCCombinationInvocations.familyInvocation_mem _ _ _ _ _ _ _ source index
  · apply familyInvocationRows env holds
    intro source index
    rw [Data.compactRowInvocations_eq, PiRLCCombinationInvocations.invocations]
    apply List.mem_append_left
    apply List.mem_append_right
    exact PiRLCCombinationInvocations.familyInvocation_mem _ _ _ _ _ _ _ source index
  · apply familyInvocationRows env holds
    intro source index
    rw [Data.compactRowInvocations_eq, PiRLCCombinationInvocations.invocations]
    apply List.mem_append_right
    exact PiRLCCombinationInvocations.familyInvocation_mem _ _ _ _ _ _ _ source index

/-- The terminal rows imply the exact seven-child PiRLC parent specification. -/
theorem piRlcSpecHolds
    (relation : ProductionKey.LogicalRelation lw pf)
    (env : Env) (holds : (circuitPackage ()).RowsHold env) :
    Formal.SpecHolds relation
      (NightstreamFPrime.Layout.Stage1.PiRLCInputs.interface
        (logicalWidth := lw) (publicFits := pf))
      NightstreamFPrime.Layout.Stage1.PiRLCInputs.phaseOffset
      (NightstreamFPrime.Layout.Stage1.Spartan.pullback env) := by
  have assumptions := NightstreamFPrime.Layout.Stage1.PiRLCInputBounds.assumptions
    relation (NightstreamFPrime.Layout.Stage1.Spartan.pullback env)
  have combinationRows := piRlcCombinationRows env holds
  refine {
    inputBinding := ?_
    sampler := ?_
    commitment := ?_
    publicInput := ?_
    eval_K := ?_
    eval_A := ?_
    outputBinding := ?_
  }
  · apply InputBinding.soundness
    intro operation member
    cases member
  · apply piRlcSamplerChain env holds
    exact @assumptions.sampler
  · simpa [PiRLCCombinationInvocations.productionCommitmentFamilyInterface,
      PiRLCCombinationInvocations.productionSharedInterface,
      NightstreamFPrime.Layout.Stage1.PiRLCStarts.commitmentLogicalStart,
      NightstreamFPrime.Layout.Stage1.PiRLCStarts.phaseLogicalStart] using
      PiRLCCombinationConformance.commitmentFamilyRows_imply_canonical
        (logicalWidth := lw) (publicFits := pf)
        env combinationRows.commitment
  · simpa [PiRLCCombinationInvocations.productionPublicInputFamilyInterface,
      PiRLCCombinationInvocations.productionSharedInterface,
      NightstreamFPrime.Layout.Stage1.PiRLCStarts.publicInputLogicalStart,
      NightstreamFPrime.Layout.Stage1.PiRLCStarts.phaseLogicalStart] using
      PiRLCCombinationConformance.publicInputFamilyRows_imply_canonical
        (logicalWidth := lw) (publicFits := pf)
        env combinationRows.publicInput
  · simpa [PiRLCCombinationInvocations.productionEvalKFamilyInterface,
      PiRLCCombinationInvocations.productionSharedInterface,
      NightstreamFPrime.Layout.Stage1.PiRLCStarts.evalKLogicalStart,
      NightstreamFPrime.Layout.Stage1.PiRLCStarts.phaseLogicalStart] using
      PiRLCCombinationConformance.evalKFamilyRows_imply_canonical
        (logicalWidth := lw) (publicFits := pf)
        env combinationRows.eval_K
  · simpa [PiRLCCombinationInvocations.productionEvalAFamilyInterface,
      PiRLCCombinationInvocations.productionSharedInterface,
      NightstreamFPrime.Layout.Stage1.PiRLCStarts.evalALogicalStart,
      NightstreamFPrime.Layout.Stage1.PiRLCStarts.phaseLogicalStart] using
      PiRLCCombinationConformance.evalAFamilyRows_imply_canonical
        (logicalWidth := lw) (publicFits := pf)
        env combinationRows.eval_A
  · apply OutputBinding.soundness
    intro operation member
    cases member

/-- The terminal rows imply the exact deterministic PiRLC phase. -/
theorem piRlcPhaseHolds
    (relation : ProductionKey.LogicalRelation lw pf)
    (ajtai : AjtaiKey
      (logicalWidth := lw) (publicFits := pf))
    (env : Env) (holds : (circuitPackage ()).RowsHold env) :
    Semantics.PhaseHolds relation ajtai
      (NightstreamFPrime.Layout.Stage1.PiRLCInputs.interface
        (logicalWidth := lw) (publicFits := pf))
      NightstreamFPrime.Layout.Stage1.PiRLCInputs.phaseOffset
      (NightstreamFPrime.Layout.Stage1.Spartan.pullback env) := by
  apply Semantics.spec_implies_phaseHolds
  exact piRlcSpecHolds relation env holds

end CompressionLean.Terminal
