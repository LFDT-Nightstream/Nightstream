import NightstreamFPrime.Export.Stage1.PiRLCCombinationReadSupport
import NightstreamFPrime.Export.Stage1.StoredPhysicalExecution
import NightstreamFPrime.Export.Stage1.PackageCompleteness
import NightstreamFPrime.Export.Stage1.PiRLCSamplerCompleteness
import NightstreamFPrime.Layout.Stage1.PiRLCInputBounds

/-!
Canonical hash and permutation inputs do not read PiRLC product scratch.
The proofs use the selected schedules and their existing source bounds.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.PiRLCCombinationPermutationReadSupport

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.Poseidon2
open NightstreamFPrime.Layout.Stage1
open NightstreamFPrime.Gadgets.Poseidon2
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Export.Package
open PiRLCCombinationWitnessReadSupport (Outside)
open StoredPhysicalExecution (PermutationSupported)

private abbrev application := Poseidon2HashChainV1Package.application
private abbrev shift := PerApplicationCachedShift.Context.ofProgram application

/-- Used only for the shape-dependent schedule and source-bound theorems. -/
private def shapeRelation : ProductionKey.LogicalRelation Data.logicalWidth Data.publicFits where
  matrices := fun _ _ _ => 0
  cubeFits := by
    norm_num [Data.logicalWidth, VerifierContext.candidateLogicalWidth,
      Phi81CarrierLayout.carrierWidth, Phi81ColumnLayout.blockCount,
      ringDegree, cubeVariables]

private theorem shifted_permutation (invocation : PermutationInvocation)
    (supported : PermutationSupported Outside invocation) :
    PermutationSupported Outside
      (PerApplicationCachedShift.shiftPermutationInvocation shift invocation) := by
  intro lane laneBound
  rw [PerApplicationCachedShift.shiftPermutationInvocation_eq,
    PerApplicationPreservation.shiftedInvocationInputCombination]
  exact PiRLCCombinationReadSupport.shiftSparseCombination_supported application _
    (supported lane laneBound)

theorem hashes_supported (chain : HashChain)
    (member : chain ∈ [Data.priorChain, Data.outputChain]) (ordinal : Nat)
    (bounded : ordinal <
      (PerApplicationCachedShift.shiftHashChain shift chain).absorbCount + 1) :
    PermutationSupported Outside (StoredPhysicalExecution.hashInvocation
      (PilotData.circuitPackage ())
      (PerApplicationCachedShift.shiftHashChain shift chain) ordinal) := by
  have inputBound : (PerApplicationCachedShift.shiftHashChain shift chain).inputStart +
      (PerApplicationCachedShift.shiftHashChain shift chain).inputLength ≤ 20572364 := by
    rcases List.mem_cons.mp member with rfl | member
    · change 0 + 49393 ≤ 20572364
      decide
    · rcases List.mem_cons.mp member with rfl | member
      · change 49393 + 49393 ≤ 20572364
        decide
      · simp at member
  have witnessBound : (PerApplicationCachedShift.shiftHashChain shift chain).witnessStart +
      ((PerApplicationCachedShift.shiftHashChain shift chain).absorbCount + 1) * 592 ≤
        20572364 := by
    rcases List.mem_cons.mp member with rfl | member
    · change 128074 + 12350 * 592 ≤ 20572364
      decide
    · rcases List.mem_cons.mp member with rfl | member
      · change 7439538 + 12350 * 592 ≤ 20572364
        decide
      · simp at member
  intro lane laneBound
  have selected : invocationInputCombination
      (StoredPhysicalExecution.hashInvocation (PilotData.circuitPackage ())
        (PerApplicationCachedShift.shiftHashChain shift chain) ordinal) lane =
      Rows.sparseCombination (invocationInput (PilotData.circuitPackage ())
        (PerApplicationCachedShift.shiftHashChain shift chain) ordinal lane) := by
    change (List.ofFn (fun current : Fin 8 => Rows.sparseCombination
      (invocationInput (PilotData.circuitPackage ())
        (PerApplicationCachedShift.shiftHashChain shift chain) ordinal current.val))).getD
          lane zeroSparseCombination = _
    exact PriorStateHash.ofFn_getD _ ⟨lane, laneBound⟩ zeroSparseCombination
  rw [selected, Rows.sparseCombination_toR1CS]
  intro term termMember
  left
  exact PackageCompleteness.pilotHashInvocationInput_varsBelow
    (PerApplicationCachedShift.shiftHashChain shift chain) ordinal ⟨lane, laneBound⟩
    20572364 (by omega) inputBound witnessBound term termMember

private theorem piCcs_supported (invocation : PermutationInvocation)
    (member : invocation ∈ PiCCSInvocations.invocations Data.logicalWidth Data.publicFits) :
    PermutationSupported Outside invocation := by
  have schedule := PiCCSInvocations.invocations_scheduleWithin
    Data.logicalWidth Data.publicFits shapeRelation
  have inputs := PiCCSCompleteness.schedule_stableInputs schedule.1 invocation member
  have before := schedule.2 invocation member
  rw [PiCCSInvocations.invocationCeiling_eq] at before
  intro lane laneBound term termMember
  rcases inputs ⟨lane, laneBound⟩ term termMember with earlier | suffix
  · left
    change term.1 < 20572364
    omega
  · right
    change 28421264 ≤ term.1
    rw [Spartan.privateColumnCount_eq] at suffix
    omega

private theorem source_invocation_supported (phase rowStart start : Nat)
    (state : Layer.EState) (affine : StateAffine state)
    (bounded : ∀ lane, (state lane).VarsBelow PiRLCStarts.commitmentFreshStart) :
    PermutationSupported Outside (Invocations.invocation phase rowStart start state) := by
  intro lane laneBound
  have selected : invocationInputCombination
      (Invocations.invocation phase rowStart start state) lane =
      Invocations.inputCombination (state ⟨lane, laneBound⟩) := by
    change (List.ofFn (fun current : Fin 8 => Invocations.inputCombination
      (state current))).getD lane zeroSparseCombination = _
    exact PriorStateHash.ofFn_getD _ ⟨lane, laneBound⟩ zeroSparseCombination
  rw [selected]
  exact Invocations.inputCombination_termsOutside (state ⟨lane, laneBound⟩)
    PiRLCStarts.commitmentFreshStart PiRLCCombinationScratchGeometry.scratchEnd
    PiRLCCombinationInvocations.commitmentFreshStart_local
    (by rw [Spartan.privateColumnCount_eq]; decide)
    (affine ⟨lane, laneBound⟩) (bounded ⟨lane, laneBound⟩)

private theorem entryState_below (source : Nat) (sourceBound : source < 17)
    (lane : Fin 8) :
    (PiRLCSamplerInvocations.entryState
      (logicalWidth := Data.logicalWidth) (publicFits := Data.publicFits) source lane).VarsBelow
        PiRLCStarts.commitmentFreshStart := by
  have assumptions := (PiRLCInputBounds.assumptions shapeRelation (fun _ => 0)).sampler
  have bounded := PiRLC.v1_1.SamplerChain.stateAtExpr_varsBelow
    (PiRLCSamplerRows.samplerInterface
      (logicalWidth := Data.logicalWidth) (publicFits := Data.publicFits))
    PiRLCStarts.samplerLogicalStart (fun _ => 0) assumptions source (by
      rw [PiRLC.v1_1.SamplerChain.sourceCount_eq]
      omega) lane
  have equal := congrFun
    (PiRLCSamplerProjection.fastProductionEntryState_eq
      (logicalWidth := Data.logicalWidth) (publicFits := Data.publicFits) source) lane
  rw [← equal] at bounded
  rw [← PiRLCSamplerInvocations.fastEntryState_eq_entryState]
  apply Expr.VarsBelow.mono _ bounded
  change 19513117 + source * 15504 ≤ 20572642
  omega

private theorem windowState_below (source round : Nat)
    (sourceBound : source < 17) (roundBound : round < 8) (lane : Fin 8) :
    (PiRLCSamplerInvocations.fastWindowState
      (logicalWidth := Data.logicalWidth) (publicFits := Data.publicFits)
      source round lane).VarsBelow PiRLCStarts.commitmentFreshStart := by
  have laneBound := lane.isLt
  unfold PiRLCSamplerInvocations.fastWindowState
    PiRLCSamplerProjection.fastProductionWindowInitialState
  cases round with
  | zero =>
      rw [PiRLCSamplerProjection.fastProductionEntryOutput_eq_scheduleOutput]
      change 19513117 + source * 15504 + 584 + lane.val < 20572642
      omega
  | succ previous =>
      change 19513117 + source * 15504 + 592 + previous * 992 + 400 + 584 + lane.val <
        20572642
      omega

private theorem sampler_supported (invocation : PermutationInvocation)
    (member : invocation ∈ PiRLCSamplerInvocations.invocations
      (logicalWidth := Data.logicalWidth) (publicFits := Data.publicFits)) :
    PermutationSupported Outside invocation := by
  rcases List.mem_flatMap.mp member with ⟨source, sourceMember, selected⟩
  have sourceBound : source < 17 := List.mem_range.mp sourceMember
  rcases List.mem_append.mp selected with entry | window
  · rw [PiRLCSamplerCompleteness.entryInvocations_eq_singleton] at entry
    rcases List.mem_singleton.mp entry with rfl
    apply source_invocation_supported
    · apply NightstreamFPrime.Layout.Poseidon2.absorbE_affine
      · exact PiRLCSamplerInvocations.entryState_affine source
      · intro expression member
        rcases List.mem_map.mp member with ⟨word, _, rfl⟩
        exact R1CS.isAffine_const word
    · intro lane
      apply Hash.absorbE_varsBelow
      · exact entryState_below source sourceBound
      · intro expression member
        rcases List.mem_map.mp member with ⟨word, _, rfl⟩
        trivial
  · rcases List.mem_map.mp window with ⟨round, roundMember, rfl⟩
    have roundBound : round < 8 := List.mem_range.mp roundMember
    apply source_invocation_supported
    · rw [PiRLCSamplerInvocations.fastWindowState_eq_windowState]
      exact PiRLCSamplerInvocations.windowState_affine source round
    · exact windowState_below source round sourceBound roundBound

theorem permutations_supported (block : PermutationPlan.Block)
    (blockMember : block ∈ PermutationPlan.canonicalBlocks ())
    (invocation : PermutationInvocation) (member : invocation ∈ block.expand) :
    PermutationSupported Outside
      (PerApplicationCachedShift.shiftPermutationInvocation shift invocation) := by
  apply shifted_permutation
  have canonical : invocation ∈ (PermutationPlan.canonicalBlocks ()).flatMap
      PermutationPlan.Block.expand := List.mem_flatMap.mpr ⟨block, blockMember, member⟩
  rw [PermutationPlan.canonicalBlocks_expand, Data.permutationInvocations_eq] at canonical
  rcases List.mem_append.mp canonical with piCcs | sampler
  · exact piCcs_supported invocation piCcs
  · exact sampler_supported invocation sampler

theorem applicationPermutations_supported (invocation : PermutationInvocation)
    (member : invocation ∈
      (PerApplicationPackage.directApplicationPlan application).permutationInvocations) :
    PermutationSupported Outside invocation := by
  change invocation ∈ [] at member
  simp at member

end NightstreamFPrime.Export.Stage1.PiRLCCombinationPermutationReadSupport
