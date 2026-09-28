import NightstreamFPrime.Layout.Stage1.StateDecoder

/-!
Owns the fixed-word checks for an honestly serialized Stage 1 state. The
existing serializer supplies the exact tag, lengths and program counter used
by PiCCS StateBinding. This adds no rows or alternate state representation.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Layout.Stage1.StateEncodingCanonical

open NightstreamFPrime.Spec
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Lifecycle.PiCCS.v1_1
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint

variable {logicalWidth : Nat}
  {publicFits : ringDegree * publicRingColumns ≤
    Phi81CarrierLayout.carrierWidth logicalWidth}

private theorem middle_word (before middle after : List F) (index : Nat)
    (bound : index < middle.length) :
    (before ++ middle ++ after).getD (before.length + index) 0 = middle.getD index 0 := by
  rw [List.append_assoc, List.getD_append_right]
  · rw [Nat.add_sub_cancel_left, List.getD_append _ _ _ _ bound]
  · omega

private theorem framed_header (before payload after : List F) :
    (before ++ block payload ++ after).getD before.length 0 = natWord payload.length := by
  rw [List.append_assoc, List.getD_append_right _ _ _ _ le_rfl, Nat.sub_self]
  rfl

private theorem running_group_length
    (running : Running (logicalWidth := logicalWidth) (publicFits := publicFits))
    (source : Fin productionShape.runningCount) :
    (block (serializeCommitment (running.commitments source)) ++
      block (serializePublicInput (publicFits := publicFits) (running.publicInputs source)) ++
      block (serializeEvaluations (running.evaluations source))).length = 3081 := by
  simp [productionProfile, productionShape, FullShape, fullShape,
    Phi81Relation.Shape.publicWidth, publicRingColumns, ringDegree, Phi81MatrixSource.phi81Shape]

private theorem running_group_word
    (running : Running (logicalWidth := logicalWidth) (publicFits := publicFits))
    (source : Fin productionShape.runningCount) (index : Nat) (bound : index < 3081) :
    (serializeRunning (publicFits := publicFits) running).getD
      (57 + source.val * 3081 + index) 0 =
    (block (serializeCommitment (running.commitments source)) ++
      block (serializePublicInput (publicFits := publicFits) (running.publicInputs source)) ++
      block (serializeEvaluations (running.evaluations source))).getD index 0 := by
  rw [serializeRunning, List.getD_append_right]
  · have pointLength : (block (serializePoint running.point)).length = 57 := by
      simp [cubeVariables]
    rw [pointLength]
    have shift : 57 + source.val * 3081 + index - 57 = source.val * 3081 + index := by omega
    rw [shift]
    exact PiCCSRepresentation.finRange_flatMap_getD _ (running_group_length running) source index bound
  · simp only [block_length, serializePoint_length, cubeVariables]
    omega

private theorem running_headers
    (running : Running (logicalWidth := logicalWidth) (publicFits := publicFits))
    (source : Fin productionShape.runningCount) (word : StateBinding.FixedWord)
    (member : word ∈
      [⟨StateBinding.runningGroupStart source.val, Poseidon2.ofNat 1188⟩,
       ⟨StateBinding.runningGroupStart source.val + 1189, Poseidon2.ofNat 270⟩,
       ⟨StateBinding.runningGroupStart source.val + 1460, Poseidon2.ofNat 1620⟩]) :
    (serializeRunning (publicFits := publicFits) running).getD (word.index - 39) 0 = word.value := by
  have commitmentLength : (serializeCommitment (running.commitments source)).length = 1188 := by
    simp [productionProfile, ringDegree]
  have publicLength : (serializePublicInput (publicFits := publicFits)
      (running.publicInputs source)).length = 270 := by
    simp [FullShape, fullShape, Phi81Relation.Shape.publicWidth, publicRingColumns, ringDegree]
  have evaluationLength : (serializeEvaluations (running.evaluations source)).length = 1620 := by
    simp [productionShape, Phi81MatrixSource.phi81Shape, productionProfile, ringDegree]
  simp only [List.mem_cons, List.not_mem_nil, or_false] at member
  rcases member with rfl | rfl | rfl
  · have shift : StateBinding.runningGroupStart source.val - 39 = 57 + source.val * 3081 + 0 := by
      simp only [StateBinding.runningGroupStart, cubeVariables]
      omega
    rw [shift, running_group_word running source 0 (by decide)]
    simp only [block, List.cons_append, List.getD_cons_zero, commitmentLength, natWord]
  · have shift : StateBinding.runningGroupStart source.val + 1189 - 39 =
        57 + source.val * 3081 + 1189 := by
      simp only [StateBinding.runningGroupStart, cubeVariables]
      omega
    rw [shift, running_group_word running source 1189 (by decide)]
    have header := framed_header (block (serializeCommitment (running.commitments source)))
      (serializePublicInput (publicFits := publicFits) (running.publicInputs source))
      (block (serializeEvaluations (running.evaluations source)))
    simpa only [block_length, commitmentLength, publicLength, natWord] using header
  · have shift : StateBinding.runningGroupStart source.val + 1460 - 39 =
        57 + source.val * 3081 + 1460 := by
      simp only [StateBinding.runningGroupStart, cubeVariables]
      omega
    rw [shift, running_group_word running source 1460 (by decide)]
    have header := framed_header
      (block (serializeCommitment (running.commitments source)) ++
        block (serializePublicInput (publicFits := publicFits) (running.publicInputs source)))
      (serializeEvaluations (running.evaluations source)) []
    simpa only [List.append_nil, List.length_append, block_length,
      commitmentLength, publicLength, evaluationLength, natWord] using header

/-- The canonical serializer satisfies every fixed-word check used by the
PiCCS state boundary. Only fixed context/state widths and the selected program
counter are required; iteration injectivity is a separate range condition. -/
theorem serializePreimage_canonical
    (preimage : HashPreimage (logicalWidth := logicalWidth) (publicFits := publicFits))
    (fixed : PilotProduction.FixedPreimage preimage) (pc : preimage.pc = 1) :
    StateDecoder.Canonical (fun index => (serializePreimage (publicFits := publicFits) preimage).getD index 0) := by
  rcases fixed with ⟨keyLength, initialLength, currentLength⟩
  change (preimage.verifierKeys functionIndex).length = 4 at keyLength
  change preimage.z0.length = 4 at initialLength
  change preimage.current.length = 4 at currentLength
  let beforeInitial := stateDomainTag ++ block (preimage.verifierKeys functionIndex) ++ [natWord preimage.iteration]
  let beforeCurrent := beforeInitial ++ block preimage.z0
  let beforeRunning := beforeCurrent ++ block preimage.current
  have initialStart : beforeInitial.length = 29 := by
    simp [beforeInitial, stateDomainTag_length, keyLength]
  have currentStart : beforeCurrent.length = 34 := by
    simp [beforeCurrent, initialStart, initialLength]
  have runningStart : beforeRunning.length = 39 := by
    simp [beforeRunning, currentStart, currentLength]
  have keyHeader : (serializePreimage (publicFits := publicFits) preimage).getD 23 0 = Poseidon2.ofNat 4 := by
    have header := framed_header stateDomainTag (preimage.verifierKeys functionIndex)
      ([natWord preimage.iteration] ++ block preimage.z0 ++ block preimage.current ++
        serializeRunning (publicFits := publicFits) (preimage.running functionIndex) ++ [natWord preimage.pc])
    simpa only [serializePreimage, List.append_assoc, stateDomainTag_length, keyLength, natWord] using header
  have initialHeader : (serializePreimage (publicFits := publicFits) preimage).getD 29 0 = Poseidon2.ofNat 4 := by
    have header := framed_header beforeInitial preimage.z0
      (block preimage.current ++ serializeRunning (publicFits := publicFits)
        (preimage.running functionIndex) ++ [natWord preimage.pc])
    rw [initialStart, initialLength] at header
    simpa only [beforeInitial, serializePreimage, List.append_assoc, natWord] using header
  have currentHeader : (serializePreimage (publicFits := publicFits) preimage).getD 34 0 = Poseidon2.ofNat 4 := by
    have header := framed_header beforeCurrent preimage.current
      (serializeRunning (publicFits := publicFits) (preimage.running functionIndex) ++ [natWord preimage.pc])
    rw [currentStart, currentLength] at header
    simpa only [beforeCurrent, beforeInitial, serializePreimage, List.append_assoc, natWord] using header
  have runningWord (index : Nat) (bound : index < 49353) :
      (serializePreimage (publicFits := publicFits) preimage).getD (39 + index) 0 =
        (serializeRunning (publicFits := publicFits) (preimage.running functionIndex)).getD index 0 := by
    have selected := middle_word beforeRunning
      (serializeRunning (publicFits := publicFits) (preimage.running functionIndex))
      [natWord preimage.pc] index (by rwa [serializeRunning_length])
    rw [runningStart] at selected
    exact selected
  intro word member
  simp only [StateBinding.fixedWords, List.mem_append] at member
  rcases member with ((tagMember | fixedMember) | runningMember) | pcMember
  · rw [StateBinding.tagWords, List.mem_map] at tagMember
    rcases tagMember with ⟨index, _member, rfl⟩
    simp only [serializePreimage, List.append_assoc]
    exact List.getD_append _ _ _ _ index.isLt
  · simp only [List.mem_cons, List.not_mem_nil, or_false] at fixedMember
    rcases fixedMember with rfl | rfl | rfl | rfl
    · exact keyHeader
    · exact initialHeader
    · exact currentHeader
    · have point := runningWord 0 (by decide)
      simpa only [Nat.add_zero, serializeRunning, block, List.cons_append,
        List.getD_cons_zero, serializePoint_length, natWord] using point
  · rw [StateBinding.runningPrefixWords, List.mem_flatMap] at runningMember
    rcases runningMember with ⟨source, _sourceMember, wordMember⟩
    have indexBounds : 39 ≤ word.index ∧ word.index < 49392 := by
      have sourceBound := source.isLt
      change source.val < 16 at sourceBound
      simp only [List.mem_cons, List.not_mem_nil, or_false] at wordMember
      rcases wordMember with rfl | rfl | rfl <;>
        simp only [StateBinding.runningGroupStart, cubeVariables] <;> omega
    have selected := runningWord (word.index - 39) (by omega)
    rw [Nat.add_sub_of_le indexBounds.1] at selected
    exact selected.trans (running_headers (preimage.running functionIndex) source word wordMember)
  · simp only [List.mem_singleton] at pcMember
    subst word
    have selected := middle_word
      (beforeRunning ++ serializeRunning (publicFits := publicFits) (preimage.running functionIndex))
      [natWord preimage.pc] [] 0 (by simp)
    rw [List.length_append, runningStart, serializeRunning_length, Nat.add_zero] at selected
    simpa only [List.append_nil, beforeRunning, beforeCurrent, beforeInitial, serializePreimage,
      List.getD_cons_zero, pc, natWord] using selected

/-- Context payload words occur at the existing fixed state positions. -/
theorem serializePreimage_context_word
    (preimage : HashPreimage (logicalWidth := logicalWidth) (publicFits := publicFits))
    (fixed : PilotProduction.FixedPreimage preimage) (lane : Fin 4) :
    (serializePreimage (publicFits := publicFits) preimage).getD (24 + lane.val) 0 =
      (preimage.verifierKeys functionIndex).getD lane.val 0 := by
  have keyLength := fixed.1
  change (preimage.verifierKeys functionIndex).length = 4 at keyLength
  have selected := middle_word stateDomainTag (block (preimage.verifierKeys functionIndex))
    ([natWord preimage.iteration] ++ block preimage.z0 ++ block preimage.current ++
      serializeRunning (publicFits := publicFits) (preimage.running functionIndex) ++ [natWord preimage.pc])
    (1 + lane.val) (by simp only [block_length, keyLength]; omega)
  have index : stateDomainTag.length + (1 + lane.val) = 24 + lane.val := by
    rw [stateDomainTag_length]
    omega
  rw [index] at selected
  simpa only [serializePreimage, List.append_assoc, block, Nat.add_comm 1 lane.val,
    List.getD_cons_succ] using selected

end NightstreamFPrime.Layout.Stage1.StateEncodingCanonical
