import NightstreamFPrime.Export.Stage1.PiRLCProductSemanticCustody
import NightstreamFPrime.Export.Stage1.PiDECOrdinaryDirectSource

/-!
Owns exact environment custody for the PiDEC direct-row source support. PiDEC
reads either final PiRLC product outputs or columns after every sampler-owned
source. This module adds no row and does not close PiDEC status.
-/

namespace NightstreamFPrime.Export.Stage1.PiDECEnvironmentCustody

open NightstreamFPrime.Circuit
open NightstreamFPrime.Gadgets.Sampling
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Layout.Stage1
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Lifecycle.PiRLC.v1_1
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra

private theorem ordinaryLocation_beforePiDECProof
    (location : PiRLCSamplerOrdinaryDirectPlan.Location) :
    location.sourceColumn < PiDECInputs.proofInputStart := by
  rw [PiDECInputs.proofInputStart, PiRLCStarts.finalBoundaries_eq.2]
  cases location with
  | poseidon source position =>
      have sourceLt : source.val < 17 := source.isLt
      have positionLt : position.val < 4 := position.isLt
      rw [PiRLCSamplerOrdinaryDirectPlan.poseidonColumn]
      simp only [PiRLCStarts.samplerLogicalStart, Formal.samplerOffset, PiRLCStarts.phaseLogicalStart_eq]
      omega
  | logical source position =>
      have sourceLt : source.val < 17 := source.isLt
      have positionLt : position.val < 617 := position.isLt
      rw [PiRLCSamplerOrdinaryDirectPlan.logicalColumn]
      simp only [PiRLCStarts.samplerLogicalStart, Formal.samplerOffset, PiRLCStarts.phaseLogicalStart_eq]
      omega
  | word source position =>
      have sourceLt : source.val < 17 := source.isLt
      have positionLt : position.val < 54 := position.isLt
      rw [PiRLCSamplerOrdinaryDirectPlan.wordColumn]
      simp only [PiRLCStarts.samplerLogicalStart, Formal.samplerOffset, PiRLCStarts.phaseLogicalStart_eq]
      omega
  | fresh source position =>
      have sourceLt : source.val < 17 := source.isLt
      have positionLt : position.val < 1548 := position.isLt
      change PiRLCStarts.samplerFreshStart + source.val * 1548 + position.val < 27360382
      rw [PiRLCStarts.samplerFreshStart, PiRLCStarts.phaseFreshStart_eq]
      omega

private theorem stateLocation_beforePiDECProof
    (location : PiRLCSamplerRetainedCustody.StateLocation) :
    location.sourceColumn < PiDECInputs.proofInputStart := by
  rw [PiDECInputs.proofInputStart, PiRLCStarts.finalBoundaries_eq.2]
  have sourceBound := location.source.isLt
  have stepBound := location.step.isLt
  have laneBound := location.lane.isLt
  norm_num [PiRLCSamplerRetainedCustody.StateLocation.sourceColumn,
    PiRLCSamplerRetainedCustody.stateOutputOffset,
    PiRLCSamplerPoseidonPlan.sourceCount,
    PiRLCSamplerPoseidonPlan.invocationsPerSource, Spec.Poseidon2.width,
    Sampler.counts.1, PiRLCSamplerRetainedCustody.stateStepStride,
    PiRLCStarts.samplerLogicalStart, PiRLCStarts.phaseLogicalStart,
    PiRLCInputs.phaseOffset, Formal.samplerOffset_eq,
    PiDECInputs.proofInputStart] at sourceBound stepBound laneBound ⊢
  omega

private theorem ordinaryTarget_none_of_afterPiDECProof {source : Nat}
    (after : PiDECInputs.proofInputStart ≤ source)
    (bound : source < Spartan.SourceColumnCount) :
    PiRLCSamplerOrdinaryDirectPlan.classifyTarget
        (Spartan.sourceToSpartan source) = none := by
  unfold PiRLCSamplerOrdinaryDirectPlan.classifyTarget
  rw [Spartan.spartanToSource_sourceToSpartan source bound]
  cases found : PiRLCSamplerOrdinaryDirectPlan.classifySource source with
  | none => exact found
  | some location =>
      have owns := PiRLCSamplerOrdinaryDirectPlan.classifySource_sound found
      have before := ordinaryLocation_beforePiDECProof location
      exfalso
      omega

private theorem stateTarget_none_of_afterPiDECProof {source : Nat}
    (after : PiDECInputs.proofInputStart ≤ source)
    (bound : source < Spartan.SourceColumnCount) :
    PiRLCSamplerRetainedCustody.classifyStateTarget
        (Spartan.sourceToSpartan source) = none := by
  unfold PiRLCSamplerRetainedCustody.classifyStateTarget
  rw [Spartan.spartanToSource_sourceToSpartan source bound]
  cases found : PiRLCSamplerRetainedCustody.classifyStateSource source with
  | none => exact found
  | some location =>
      have owns := PiRLCSamplerRetainedCustody.classifyStateSource_sound found
      have before := stateLocation_beforePiDECProof location
      exfalso
      omega

theorem semanticEnv_source_eq_transitionEnv_of_afterPiDECProof
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : PiRLCSamplerOrdinaryRetainedGeometry.Geometry program
      logicalWidth) (assignment : Assignment F logicalWidth)
    (base : Fin (PiRLCProductPlan.baseSourceWidth program) → F)
    {source : Nat} (after : PiDECInputs.proofInputStart ≤ source)
    (bound : source < Spartan.SourceColumnCount) :
    PiRLCSamplerRetainedCustody.semanticEnv geometry assignment base
        (Spartan.sourceToSpartan source) =
      RunningTransitionDirectPlan.transitionEnv program base
        (Spartan.sourceToSpartan source) := by
  unfold PiRLCSamplerRetainedCustody.semanticEnv
  rw [ordinaryTarget_none_of_afterPiDECProof after bound,
    stateTarget_none_of_afterPiDECProof after bound]

private theorem parent_interval {source : Nat}
    (support : PiDECSourceSupport.Parent source) :
    PiRLCStarts.commitmentLogicalStart ≤ source ∧
      source < PiRLCStarts.phaseFreshStart := by
  exact PiDECSourceSupport.parent_within_piRlc support

/-- Every source used by a nonempty PiDEC row has the same value in the
complete sampler environment and canonical transition environment. -/
theorem semanticEnv_source_eq_transitionEnv
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : PiRLCSamplerOrdinaryRetainedGeometry.Geometry program
      logicalWidth) (assignment : Assignment F logicalWidth)
    (base : Fin (PiRLCProductPlan.baseSourceWidth program) → F)
    {source : Nat} (support : PiDECSourceSupport.Source source) :
    PiRLCSamplerRetainedCustody.semanticEnv geometry assignment base
        (Spartan.sourceToSpartan source) =
      RunningTransitionDirectPlan.transitionEnv program base
        (Spartan.sourceToSpartan source) := by
  have sourceBound := PiDECSourceSupport.source_lt_sourceColumnCount support
  rcases support with ((parent | proof) | logical) | fresh
  · have interval := parent_interval parent
    exact PiRLCProductSemanticCustody.semanticEnv_source_eq_transitionEnv_of_productInterval
      geometry assignment base interval.1 interval.2
  · exact semanticEnv_source_eq_transitionEnv_of_afterPiDECProof geometry
      assignment base proof.1 sourceBound
  · exact semanticEnv_source_eq_transitionEnv_of_afterPiDECProof geometry
      assignment base (by
        have lower := logical.1
        norm_num [PiDECStarts.phaseLogicalStart, PiDECInputs.phaseOffset,
          PiDECInputs.proofInputStart, PiDECInputs.proofInputColumnCount,
          PiDECInputs.childCount, PiDECInputs.commitmentWordsPerChild,
          PiDECInputs.evalKWordsPerChild, PiDECInputs.evalAWordsPerChild,
          PiDECInputs.publicInputWordsPerChild] at lower ⊢
        omega) sourceBound
  · exact semanticEnv_source_eq_transitionEnv_of_afterPiDECProof geometry
      assignment base (by
        have lower := fresh.1
        norm_num [PiDECStarts.phaseFreshStart, PiDECStarts.phaseLogicalStart,
          PiDECInputs.phaseOffset, PiDECInputs.proofInputStart,
          PiDECInputs.proofInputColumnCount, PiDECInputs.childCount,
          PiDECInputs.commitmentWordsPerChild, PiDECInputs.evalKWordsPerChild,
          PiDECInputs.evalAWordsPerChild, PiDECInputs.publicInputWordsPerChild,
          PiDEC.v1_1.Formal.logicalPrivateCount] at lower ⊢
        omega) sourceBound

theorem semanticEnv_eq_transitionEnv_of_target
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : PiRLCSamplerOrdinaryRetainedGeometry.Geometry program
      logicalWidth) (assignment : Assignment F logicalWidth)
    (base : Fin (PiRLCProductPlan.baseSourceWidth program) → F)
    {column : Nat} (support : PiDECSourceSupport.Target column) :
    PiRLCSamplerRetainedCustody.semanticEnv geometry assignment base column =
      RunningTransitionDirectPlan.transitionEnv program base column := by
  rcases support with ⟨source, sourceSupport, mapped⟩
  rw [← mapped]
  exact semanticEnv_source_eq_transitionEnv geometry assignment base
    sourceSupport

end NightstreamFPrime.Export.Stage1.PiDECEnvironmentCustody
