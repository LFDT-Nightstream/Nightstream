import NightstreamFPrime.Circuit.WitnessSupport
import NightstreamFPrime.Gadgets.Poseidon2.Formal.Witness
import NightstreamFPrime.Lifecycle.PiDEC.v1_1.PublicInputSplit.Witness
import NightstreamFPrime.Export.Stage1.PiDECDirectSupport
import NightstreamFPrime.Export.Stage1.PiRLCCombinationScratchGeometry
import NightstreamFPrime.Export.Stage1.PiRLCSamplerOrdinaryDirectSource
import NightstreamFPrime.Export.Stage1.Poseidon2HashChainV1Package
import NightstreamFPrime.Export.Stage1.WitnessPlan

/-!
Owns actual recipe and hint read support for the canonical sampler, PiDEC,
running-transition and selected application witness batches. The proofs use
child witness contracts and the fixed physical scratch interval.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.PiRLCCombinationWitnessReadSupport

open NightstreamFPrime.Circuit
open NightstreamFPrime.Spec
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.Stage1
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint

abbrev Outside (column : Nat) : Prop :=
  column < PiRLCCombinationScratchGeometry.scratchStart ∨
    PiRLCCombinationScratchGeometry.scratchEnd ≤ column

private abbrev SourceAllowed (column : Nat) : Prop :=
  Outside (Spartan.sourceToSpartan column)

private theorem mapped_outside (column : Nat)
    (lower : Spartan.piCcsPhaseOffset ≤ column)
    (outside : column < PiRLCStarts.commitmentFreshStart ∨
      PiRLCStarts.outputFreshStart ≤ column) : SourceAllowed column := by
  change 14751804 ≤ column at lower
  unfold SourceAllowed Spartan.sourceToSpartan
  rw [if_neg (by change ¬ column < 14722512; omega),
    if_neg (by change ¬ column < 14722516; omega),
    if_neg (by change ¬ column < 14751804; omega)]
  change column < 21124348 ∨ 28973248 ≤ column at outside
  change 14751526 + (column - 14751804) < 21124070 ∨
    28972970 ≤ 14751526 + (column - 14751804)
  omega

private theorem remapExpr_supported (expression : Expr)
    (supported : expression.VarsSatisfy SourceAllowed) :
    (WitnessProgram.remapExpr expression).VarsSatisfy Outside := by
  induction expression with
  | const => trivial
  | var => exact supported
  | add left right leftIH rightIH => exact ⟨leftIH supported.1, rightIH supported.2⟩
  | mul left right leftIH rightIH => exact ⟨leftIH supported.1, rightIH supported.2⟩

private theorem remapBatch_supported (batch : WitnessBatch)
    (supported : batch.ReadsSatisfy SourceAllowed) :
    (WitnessProgram.remapBatch batch).ReadsSatisfy Outside := by
  constructor
  · intro expression member
    rcases List.mem_map.mp member with ⟨source, sourceMember, rfl⟩
    exact remapExpr_supported source (supported.1 source sourceMember)
  · intro hint member
    rcases List.mem_map.mp member with ⟨source, sourceMember, rfl⟩
    have support := remapExpr_supported source.source (supported.2 source sourceMember)
    cases source <;> exact support

private theorem shifted_outside (application : Stage1.Application.Program) (column : Nat)
    (outside : Outside column) :
    Outside (PerApplicationPackage.shiftColumn application column) := by
  unfold PerApplicationPackage.shiftColumn
  split_ifs with before
  · exact outside
  · right
    have constant : PerApplicationPackage.basePackage.layout.constantColumn = 29336446 :=
      Package.circuitPackage_layout_values.2.2.1
    rw [constant] at before
    change 28972970 ≤ _
    omega

private theorem shiftExpr_supported (application : Stage1.Application.Program)
    (expression : Expr) (supported : expression.VarsSatisfy Outside) :
    (PerApplicationPackage.shiftExpr application expression).VarsSatisfy Outside := by
  induction expression with
  | const => trivial
  | var column => exact shifted_outside application column supported
  | add left right leftIH rightIH => exact ⟨leftIH supported.1, rightIH supported.2⟩
  | mul left right leftIH rightIH => exact ⟨leftIH supported.1, rightIH supported.2⟩

/-- The application insertion moves only the constant/public suffix. -/
theorem shiftBatch_readsSatisfy (application : Stage1.Application.Program)
    (batch : WitnessBatch) (supported : batch.ReadsSatisfy Outside) :
    (PerApplicationPackage.shiftBatch application batch).ReadsSatisfy Outside := by
  constructor
  · intro expression member
    rcases List.mem_map.mp member with ⟨source, sourceMember, rfl⟩
    exact shiftExpr_supported application source (supported.1 source sourceMember)
  · intro hint member
    rcases List.mem_map.mp member with ⟨source, sourceMember, rfl⟩
    have support := shiftExpr_supported application source.source
      (supported.2 source sourceMember)
    cases source <;> exact support

private theorem piDecSource_allowed (column : Nat)
    (supported : PiDECSourceSupport.Source column) : SourceAllowed column := by
  apply mapped_outside
  · have lower := PiDECSourceSupport.parentStart_le_source supported
    exact Nat.le_trans (by decide) lower
  · rcases supported with ((parent | proof) | logical) | fresh
    · left
      rcases parent with commitment | publicInput | evalK | evalA
      all_goals
        first
        | have bounded := commitment.2
        | have bounded := publicInput.2
        | have bounded := evalK.2
        | have bounded := evalA.2
        change column < 21124348
        norm_num [PiDECSourceSupport.parentCommitmentStart_eq,
          PiDECSourceSupport.parentPublicInputStart_eq,
          PiDECSourceSupport.parentEvalKStart_eq,
          PiDECSourceSupport.parentEvalAStart_eq,
          PiDECInputs.commitmentWordsPerChild, PiDECInputs.publicInputWordsPerChild,
          PiDECInputs.evalKWordsPerChild, PiDECInputs.evalAWordsPerChild,
          productionProfile, PiDEC.v1_1.PublicInputSplit.exactCoordinateCount,
          PiDEC.v1_1.CommitmentRecomposition.coordinateCount,
          PiDEC.v1_1.CommitmentRecomposition.rowCount,
          PiDEC.v1_1.RingKRecomposition.coordinateCount,
          PiDEC.v1_1.RingKRecomposition.cellCount,
          PiDEC.v1_1.EvalKRecomposition.blockCount,
          PiDEC.v1_1.EvalARecomposition.blockCount, productionShape,
          Phi81MatrixSource.phi81Shape,
          ringDegree, publicRingColumns] at bounded
        omega
    · right
      have lower := proof.1
      change 28973248 ≤ column at lower
      exact lower
    · right
      have before : PiRLCStarts.outputFreshStart ≤ PiDECStarts.phaseLogicalStart := by decide
      exact Nat.le_trans before logical.1
    · right
      have before : PiRLCStarts.outputFreshStart ≤ PiDECStarts.phaseFreshStart := by decide
      exact Nat.le_trans before fresh.1

theorem piDecSource_outside (column : Nat)
    (supported : PiDECSourceSupport.Source column) :
    Outside (Spartan.sourceToSpartan column) := piDecSource_allowed column supported

/-- Every actual PiDEC witness batch reads only a parent output. -/
theorem piDecBatches_readsSatisfy (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤ Phi81CarrierLayout.carrierWidth logicalWidth) :
    ∀ batch ∈ WitnessProgram.piDecBatches logicalWidth publicFits,
      batch.ReadsSatisfy Outside := by
  intro batch member
  unfold WitnessProgram.piDecBatches WitnessProgram.childBatches at member
  rcases List.mem_map.mp member with ⟨source, sourceMember, rfl⟩
  apply remapBatch_supported
  apply PiDEC.v1_1.PublicInputSplit.witnesses_main_readsSatisfy
    _ _ SourceAllowed _ source sourceMember
  intro coordinate
  have support := PiDECDirectSupport.parentPublicInput_supported
    (logicalWidth := logicalWidth) (publicFits := publicFits) coordinate
  exact support.mono _ piDecSource_allowed

private theorem samplerLocal_allowed (source round : Nat) (lane : Fin 4)
    (sourceLt : source < 17) (roundLt : round < 8)
    (index : Nat) (indexLt : index < PiRLC.v1_1.DigestLane.logicalPrivateCount) :
    SourceAllowed (PiRLCStarts.digestLaneLogicalStart source round lane.val + index) := by
  apply mapped_outside
  · change 14751804 ≤ _
    norm_num [PiRLCStarts.digestLaneLogicalStart, PiRLCStarts.windowLogicalStart,
      PiRLCStarts.samplerSourceLogicalStart, PiRLCStarts.samplerLogicalStart,
      PiRLCStarts.phaseLogicalStart, PiRLCInputs.phaseOffset,
      PiRLC.v1_1.Formal.samplerOffset]
    omega
  · left
    have laneLt := lane.isLt
    change index < 100 at indexLt
    change _ < 21124348
    norm_num [PiRLCStarts.digestLaneLogicalStart, PiRLCStarts.windowLogicalStart,
      PiRLCStarts.samplerSourceLogicalStart, PiRLCStarts.samplerLogicalStart,
      PiRLCStarts.phaseLogicalStart, PiRLCInputs.phaseOffset,
      PiRLC.v1_1.Formal.samplerOffset]
    omega

private theorem samplerInput_allowed (source round : Nat) (lane : Fin 4)
    (sourceLt : source < 17) (roundLt : round < 8) :
    SourceAllowed (PiRLCSamplerOrdinaryDirectSource.poseidonSource source round lane) := by
  have laneLt := lane.isLt
  apply mapped_outside
  all_goals
    unfold PiRLCSamplerOrdinaryDirectSource.poseidonSource
    cases round with
    | zero =>
      first | left | skip
      norm_num [PiRLCStarts.samplerSourceLogicalStart, PiRLCStarts.samplerLogicalStart,
        PiRLCStarts.phaseLogicalStart, PiRLCInputs.phaseOffset,
        PiRLC.v1_1.Formal.samplerOffset, Spartan.piCcsPhaseOffset,
        PiRLCStarts.commitmentFreshStart_eq]
      omega
    | succ previous =>
      first | left | skip
      norm_num [PiRLC.v1_1.DigestWindow.permutationOffset,
        PiRLC.v1_1.Sampler.windowOffset, PiRLC.v1_1.Sampler.windowBase,
        PiRLC.v1_1.SamplerChain.sourceOffset,
        PiRLCStarts.samplerLogicalStart, PiRLCStarts.phaseLogicalStart,
        PiRLCInputs.phaseOffset, PiRLC.v1_1.Formal.samplerOffset,
        PiRLC.v1_1.Sampler.logicalPrivateCount,
        PiRLC.v1_1.Sampler.entryPrivateCount,
        PiRLC.v1_1.DigestWindow.logicalPrivateCount,
        PiRLC.v1_1.DigestLane.logicalPrivateCount,
        Spartan.piCcsPhaseOffset, PiRLCStarts.commitmentFreshStart_eq]
      omega

theorem piRlcDigestLaneBatches_readsSatisfy (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤ Phi81CarrierLayout.carrierWidth logicalWidth)
    (source round : Nat) (lane : Fin 4) (sourceLt : source < 17) (roundLt : round < 8) :
    ∀ batch ∈ WitnessProgram.piRlcDigestLaneBatches logicalWidth publicFits source round lane,
      batch.ReadsSatisfy Outside := by
  intro batch member
  unfold WitnessProgram.piRlcDigestLaneBatches WitnessProgram.digestLaneBatches at member
  rcases List.mem_map.mp member with ⟨sourceBatch, sourceMember, rfl⟩
  apply remapBatch_supported
  let interface : PiRLC.v1_1.DigestLane.Interface :=
    { source := fun _ => PiRLCSamplerOrdinaryRows.fastLaneSource
        (logicalWidth := logicalWidth) (publicFits := publicFits) source round lane }
  apply PiRLC.v1_1.DigestLane.witnessBatches_readsSatisfy
    interface _ SourceAllowed _ (samplerLocal_allowed source round lane sourceLt roundLt)
      sourceBatch sourceMember
  change (PiRLCSamplerOrdinaryRows.fastLaneSource
    (logicalWidth := logicalWidth) (publicFits := publicFits) source round lane).VarsSatisfy
      SourceAllowed
  rw [PiRLCSamplerOrdinaryDirectSource.fastLaneSource_eq_var]
  exact samplerInput_allowed source round lane sourceLt roundLt

theorem piRlcSamplerBatches_readsSatisfy (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤ Phi81CarrierLayout.carrierWidth logicalWidth) :
    ∀ batch ∈ WitnessProgram.piRlcSamplerBatches logicalWidth publicFits,
      batch.ReadsSatisfy Outside := by
  intro batch member
  rcases List.mem_flatMap.mp member with ⟨source, sourceMember, member⟩
  rcases List.mem_flatMap.mp member with ⟨round, roundMember, member⟩
  rcases List.mem_flatMap.mp member with ⟨lane, _, member⟩
  exact piRlcDigestLaneBatches_readsSatisfy logicalWidth publicFits source round lane
    (List.mem_range.mp sourceMember) (List.mem_range.mp roundMember) batch member

theorem runningTransitionBatches_readsSatisfy (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤ Phi81CarrierLayout.carrierWidth logicalWidth) :
    ∀ batch ∈ WitnessProgram.runningTransitionBatches logicalWidth publicFits,
      batch.ReadsSatisfy Outside := by
  rw [← WitnessProgram.directRunningTransitionBatches_eq]
  intro batch member
  simp only [WitnessProgram.directRunningTransitionBatches, List.mem_singleton] at member
  subst batch
  apply remapBatch_supported
  rw [WitnessBatch.readsSatisfy_hinted]
  intro hint member
  rw [List.mem_singleton] at member
  subst hint
  change SourceAllowed 28
  left
  decide

/-- The complete compact witness plan is supported by its canonical owners. -/
theorem canonicalBlock_readsSatisfy (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤ Phi81CarrierLayout.carrierWidth logicalWidth)
    (block : WitnessPlan.Block)
    (member : block ∈ WitnessPlan.canonicalBlocks logicalWidth publicFits) :
    ∀ batch ∈ block.expand, batch.ReadsSatisfy Outside := by
  intro batch batchMember
  have selected : batch ∈ (WitnessPlan.canonicalBlocks logicalWidth publicFits).flatMap
      WitnessPlan.Block.expand := List.mem_flatMap.mpr ⟨block, member, batchMember⟩
  rw [WitnessPlan.canonicalBlocks_expand] at selected
  rcases List.mem_append.mp selected with sampler | remaining
  · exact piRlcSamplerBatches_readsSatisfy logicalWidth publicFits batch sampler
  · rcases List.mem_append.mp remaining with piDec | running
    · exact piDecBatches_readsSatisfy logicalWidth publicFits batch piDec
    · exact runningTransitionBatches_readsSatisfy logicalWidth publicFits batch running

/-- The selected application's actual hash recipes read only its declared
state, message, and application-local cells. The generic application record
is not given a stronger witness-support assumption. -/
theorem selectedApplicationBatches_readsSatisfy :
    ∀ batch ∈ (PerApplicationPackage.directApplicationPlan
      Poseidon2HashChainV1Package.application).witnessBatches,
      batch.ReadsSatisfy Outside := by
  let application := Poseidon2HashChainV1Package.application
  let interface := ApplicationInputs.interface application
  let offset := ApplicationInputs.localStart application
  have inputSupport : ∀ expression ∈
      Stage1.Poseidon2HashChainV1.inputExpressions interface offset,
      expression.VarsSatisfy Outside := by
    intro expression member
    unfold Stage1.Poseidon2HashChainV1.inputExpressions at member
    rcases List.mem_append.mp member with before | message
    · rcases List.mem_append.mp before with domain | state
      · rcases List.mem_map.mp domain with ⟨word, _, rfl⟩
        trivial
      · rcases List.mem_ofFn.mp state with ⟨index, rfl⟩
        change Outside (ApplicationInputs.inputColumn index)
        rw [ApplicationInputs.inputColumn_value]
        left
        have bound := index.isLt
        change index.val < 4 at bound
        change 35 + index.val < 21124070
        omega
    · rcases List.mem_ofFn.mp message with ⟨index, rfl⟩
      change Outside (ApplicationInputs.witnessColumn index)
      right
      change 28972970 ≤ 29336446 + index.val
      omega
  have compiled := Gadgets.Poseidon2.Formal.witnesses_main_readsSatisfy
    (Stage1.Poseidon2HashChainV1.hashInterface interface) offset Outside
    inputSupport (by
      intro index _
      right
      change 28972970 ≤ 29336446 + 4 + index
      omega)
  intro batch member
  rw [PerApplicationPackage.directApplicationPlan_eq_applicationPlan] at member
  change batch ∈ witnesses (Circuit.ops
    (application.circuit (ApplicationPackage.productionColumns application).interface).main
    offset) at member
  rw [ApplicationPackage.productionColumns_interface] at member
  exact compiled batch member

end NightstreamFPrime.Export.Stage1.PiRLCCombinationWitnessReadSupport
