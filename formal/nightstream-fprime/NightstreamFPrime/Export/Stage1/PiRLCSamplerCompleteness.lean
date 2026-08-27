import NightstreamFPrime.Export.Stage1.PiRLCPackageCompleteness
import NightstreamFPrime.Export.Stage1.PermutationCompilerTransport
import NightstreamFPrime.Export.Stage1.PiRLCSamplerInvocations
import NightstreamFPrime.Export.Stage1.PiRLCSamplerOrdinaryRows
import NightstreamFPrime.Layout.PiRLC.v1_1.Leaves.DigestWindowSegments
import NightstreamFPrime.Layout.PiRLC.v1_1.SamplerSegments

/-!
Owns the constructive bridge from the exact PiRLC sampler physical packet to
the compact sampler package. Child selection stays structural and does not
unfold the heavy sampler or selector owners.
-/

namespace NightstreamFPrime.Export.Stage1.PiRLCSamplerCompleteness

open NightstreamFPrime.Circuit
open NightstreamFPrime.Export.Package
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.PiRLC.v1_1
open NightstreamFPrime.Layout.PiRLC.v1_1.Leaves
open NightstreamFPrime.Layout.Stage1
open NightstreamFPrime.Lifecycle

def chainInterface : SamplerChain.Logical.Interface :=
  NightstreamFPrime.Lifecycle.PiRLC.v1_1.Formal.samplerInterface
    (NightstreamFPrime.Lifecycle.PiRLC.v1_1.Formal.atOffset
      PiRLCPackageCompleteness.phaseInterface PiRLCInputs.phaseOffset)

private theorem chainConstraints_eq_sourceConstraintLists :
    SamplerChain.logicalConstraints chainInterface
        PiRLCStarts.samplerLogicalStart =
      (List.ofFn fun source : Fin SamplerChain.Logical.sourceCount =>
        SamplerChain.childConstraints chainInterface
          PiRLCStarts.samplerLogicalStart
          source.val).flatten := by
  rw [SamplerChain.logicalConstraints_eq_ordered]
  unfold SamplerChain.orderedConstraints SamplerChain.childConstraintLists
  apply congrArg List.flatten
  rw [List.ofFn_eq_map, ← List.map_coe_finRange_eq_range, List.map_map]
  simp [Function.comp_def]

private theorem sourceFreshCount
    (source : Fin SamplerChain.Logical.sourceCount) :
    R1CS.totalFreshCount
        (SamplerChain.childConstraints chainInterface
          PiRLCStarts.samplerLogicalStart source.val) =
      43743 := by
  exact Sampler.totalFreshCount_eq
    (SamplerChain.Logical.childInterface chainInterface
      PiRLCStarts.samplerLogicalStart source.val)
    source.val
    (SamplerChain.Logical.sourceOffset PiRLCStarts.samplerLogicalStart
      source.val)
    (SamplerChain.childInputs chainInterface PiRLCStarts.samplerLogicalStart
      (PiRLCInputs.samplerInputs (logicalWidth := Data.logicalWidth)
        (publicFits := Data.publicFits)) source.val)

private theorem sum_take_ofFn_const {count : Nat} (value : Nat)
    (index : Fin count) :
    ((List.ofFn fun _ : Fin count => value).take index.val).sum =
      index.val * value := by
  simp [index.isLt.le]

private theorem sourceFreshPrefix
    (source : Fin SamplerChain.Logical.sourceCount) :
    ((List.ofFn fun current : Fin SamplerChain.Logical.sourceCount =>
      R1CS.totalFreshCount
        (SamplerChain.childConstraints chainInterface
          PiRLCStarts.samplerLogicalStart current.val)).take source.val).sum =
      source.val * 43743 := by
  have countsEq :
      (List.ofFn fun current : Fin SamplerChain.Logical.sourceCount =>
        R1CS.totalFreshCount
          (SamplerChain.childConstraints chainInterface
            PiRLCStarts.samplerLogicalStart current.val)) =
        List.ofFn
          (fun _ : Fin SamplerChain.Logical.sourceCount => 43743) := by
    apply congrArg List.ofFn
    funext current
    exact sourceFreshCount current
  rw [countsEq]
  exact sum_take_ofFn_const 43743 source

/-- The remapped sampler packet projects to one exact source sampler lowering
under the final-column pullback. -/
theorem remappedPacket_implies_sourceRows (env : Env)
    (packets : PiRLCPackageCompleteness.RemappedPacketRowsHold env)
    (source : Fin SamplerChain.Logical.sourceCount) :
    R1CS.RowsHold (Spartan.pullback env)
      (R1CS.lowerConstraints
        (SamplerChain.childConstraints chainInterface
          PiRLCStarts.samplerLogicalStart source.val)
        (PiRLCStarts.samplerFreshStart + source.val * 43743)).rows := by
  have samplerRows := (Spartan.remapRows_hold env _).mp packets.sampler
  change R1CS.RowsHold (Spartan.pullback env)
    (R1CS.lowerConstraints
      (SamplerChain.logicalConstraints chainInterface
        PiRLCStarts.samplerLogicalStart)
      PiRLCStarts.samplerFreshStart).rows at samplerRows
  rw [chainConstraints_eq_sourceConstraintLists] at samplerRows
  have segments := (R1CS.rowsHold_flatten_iff _ _ _).mp samplerRows
  have sourceRows := R1CS.segmentsHold_ofFn_get (Spartan.pullback env)
    (fun current : Fin SamplerChain.Logical.sourceCount =>
      SamplerChain.childConstraints chainInterface
        PiRLCStarts.samplerLogicalStart current.val)
    PiRLCStarts.samplerFreshStart segments source
  rw [sourceFreshPrefix source] at sourceRows
  exact sourceRows

def sourceInterface (source : Nat) : Sampler.Logical.Interface :=
  SamplerChain.Logical.childInterface chainInterface
    PiRLCStarts.samplerLogicalStart source

def sourceOffset (source : Nat) : Nat :=
  SamplerChain.Logical.sourceOffset PiRLCStarts.samplerLogicalStart source

private def sourceInputs (source : Nat) :
    ∀ current, Sampler.InputsAffine (sourceInterface source) current :=
  SamplerChain.childInputs chainInterface PiRLCStarts.samplerLogicalStart
    (PiRLCInputs.samplerInputs (logicalWidth := Data.logicalWidth)
      (publicFits := Data.publicFits)) source

private theorem entryFreshCount (source : Nat) :
    R1CS.totalFreshCount
        (Sampler.childConstraints
          (Sampler.Logical.entryCircuit (sourceInterface source) source)
          (Sampler.Logical.entryOffset (sourceOffset source))) =
      0 := by
  change R1CS.totalFreshCount (flatConstraints (Circuit.ops
    (NightstreamFPrime.Lifecycle.PiRLC.v1_1.TranscriptAbsorption.circuit
      (Sampler.Logical.entryInterface (sourceInterface source)) source).main
        (Sampler.Logical.entryOffset (sourceOffset source)))) = 0
  exact TranscriptAbsorption.freshColumnCount_eq
    (Sampler.Logical.entryInterface (sourceInterface source)) source
    (fun current =>
      { initialState := by
          simpa [Sampler.Logical.entryInterface,
            Sampler.Logical.entryOffset] using
            (sourceInputs source current).initialState })
    (Sampler.Logical.entryOffset (sourceOffset source))

private theorem windowFreshCount (source round : Nat) :
    R1CS.totalFreshCount
        (Sampler.childConstraints
          (Sampler.Logical.windowCircuit (sourceInterface source) source
            (sourceOffset source) round)
          (Sampler.Logical.windowOffset (sourceOffset source) round)) =
      1212 := by
  change R1CS.totalFreshCount
    (DigestWindow.logicalConstraints
      (Sampler.Logical.windowInterface (sourceInterface source) source
        (sourceOffset source) round)
      (Sampler.Logical.windowOffset (sourceOffset source) round)) = 1212
  exact DigestWindow.totalFreshCount_eq _ _
    (Sampler.windowInputs (sourceInterface source) source
      (sourceOffset source) round)

private theorem selectorFreshCount (source : Nat) :
    R1CS.totalFreshCount
        (Sampler.childConstraints
          (Sampler.Logical.selectorCircuit (sourceInterface source) source
            (sourceOffset source))
          (Sampler.Logical.selectorOffset (sourceOffset source))) =
      34047 := by
  change R1CS.totalFreshCount
    (First54.logicalConstraints (sourceInterface source) source
      (sourceOffset source)
      (Sampler.Logical.selectorOffset (sourceOffset source))) = 34047
  exact First54.totalFreshCount_eq (sourceInterface source) source
    (sourceOffset source)
    (Sampler.Logical.selectorOffset (sourceOffset source))

def windowInterface (source round : Nat) : DigestWindow.Logical.Interface :=
  Sampler.Logical.windowInterface (sourceInterface source) source
    (sourceOffset source) round

def windowOffset (source round : Nat) : Nat :=
  Sampler.Logical.windowOffset (sourceOffset source) round

private theorem laneFreshCount (source round : Nat) (lane : Fin 4) :
    R1CS.totalFreshCount
        (flatConstraints (Circuit.ops
          (NightstreamFPrime.Lifecycle.PiRLC.v1_1.DigestWindow.laneCircuit
            (windowInterface source round) (windowOffset source round) lane).main
          (DigestWindow.Logical.laneOffset (windowOffset source round) lane))) =
      303 := by
  change R1CS.totalFreshCount
    (DigestLane.logicalConstraints
      (PiRLCSamplerOrdinaryRows.laneInterface
        (logicalWidth := Data.logicalWidth) (publicFits := Data.publicFits)
        source round lane)
      (PiRLCStarts.digestLaneLogicalStart source round lane.val)) = 303
  exact DigestLane.totalFreshCount_eq _ _
    (PiRLCSamplerOrdinaryRows.laneInputs
      (logicalWidth := Data.logicalWidth) (publicFits := Data.publicFits)
      source round lane)

private theorem sub_affine {left right : Expr}
    (leftAffine : R1CS.IsAffine left) (rightAffine : R1CS.IsAffine right) :
    R1CS.IsAffine (left - right) :=
  R1CS.IsAffine.add leftAffine
    (R1CS.IsAffine.const_mul (-1) rightAffine)

private theorem selectorFinalFreshCount (source : Nat) :
    R1CS.constraintFreshCount
        (PiRLCSamplerOrdinaryRows.selectorFinalConstraint source) = 0 := by
  apply R1CS.constraintFreshCount_eq_zero_of_affine
  unfold PiRLCSamplerOrdinaryRows.selectorFinalConstraint
    NightstreamFPrime.Gadgets.Sampling.First54.finalFull
    NightstreamFPrime.Gadgets.Sampling.First54Step.output
  exact sub_affine (R1CS.isAffine_var _) (R1CS.isAffine_const _)

/-- One selected source sampler projects further to its entry, eight digest
windows, and selector child segments without unfolding any child. -/
theorem remappedPacket_implies_sourceSegments (env : Env)
    (packets : PiRLCPackageCompleteness.RemappedPacketRowsHold env)
    (source : Fin SamplerChain.Logical.sourceCount) :
    R1CS.SegmentsHold (Spartan.pullback env)
      (Sampler.childConstraintLists
        (SamplerChain.Logical.childInterface chainInterface
          PiRLCStarts.samplerLogicalStart source.val)
        source.val
        (SamplerChain.Logical.sourceOffset PiRLCStarts.samplerLogicalStart
          source.val))
      (PiRLCStarts.samplerFreshStart + source.val * 43743) := by
  apply Sampler.rowsHold_implies_childSegments
  exact remappedPacket_implies_sourceRows env packets source

/-- One selected source projects to each exact digest-window lowering. -/
theorem remappedPacket_implies_windowRows (env : Env)
    (packets : PiRLCPackageCompleteness.RemappedPacketRowsHold env)
    (source : Fin SamplerChain.Logical.sourceCount)
    (round : Fin 8) :
    R1CS.RowsHold (Spartan.pullback env)
      (R1CS.lowerConstraints
        (DigestWindow.logicalConstraints
          (Sampler.Logical.windowInterface (sourceInterface source.val)
            source.val (sourceOffset source.val) round.val)
          (Sampler.Logical.windowOffset (sourceOffset source.val) round.val))
        (PiRLCStarts.samplerFreshStart + source.val * 43743 +
          round.val * 1212)).rows := by
  have segments := remappedPacket_implies_sourceSegments env packets source
  have selected := Sampler.childSegments_imply_window
    (sourceInterface source.val) source.val (sourceOffset source.val)
    (Spartan.pullback env)
    (PiRLCStarts.samplerFreshStart + source.val * 43743)
    (entryFreshCount source.val) (windowFreshCount source.val)
    segments round
  exact selected

private theorem windowWitnessLocal (source round : Nat) :
    Spartan.piCcsPhaseOffset ≤
      PiRLCStarts.digestPermutationLogicalStart source round := by
  unfold PiRLCStarts.digestPermutationLogicalStart
    PiRLCStarts.windowLogicalStart PiRLCStarts.samplerSourceLogicalStart
    PiRLCStarts.samplerLogicalStart
    NightstreamFPrime.Lifecycle.PiRLC.v1_1.Formal.samplerOffset
    PiRLCStarts.phaseLogicalStart PiRLCInputs.phaseOffset
  norm_num [Spartan.piCcsPhaseOffset]
  omega

def entryWords (source : Nat) : List Expr :=
  NightstreamFPrime.Lifecycle.PiRLC.v1_1.TranscriptAbsorption.constantWords
    (NightstreamFPrime.Lifecycle.PiRLC.v1_1.TranscriptAbsorption.frameWords
      source)

def entryPermutationState (source : Nat) :
    NightstreamFPrime.Gadgets.Poseidon2.Layer.EState :=
  NightstreamFPrime.Gadgets.Poseidon2.Hash.absorbE
    (PiRLCSamplerInvocations.entryState
      (logicalWidth := Data.logicalWidth) (publicFits := Data.publicFits)
      source)
    (entryWords source)

private theorem entryInputChunks_eq (source : Nat) :
    NightstreamFPrime.Gadgets.Poseidon2.Hash.inputChunks
        (entryWords source) =
      [entryWords source] := by
  unfold entryWords
    NightstreamFPrime.Lifecycle.PiRLC.v1_1.TranscriptAbsorption.constantWords
    NightstreamFPrime.Lifecycle.PiRLC.v1_1.TranscriptAbsorption.frameWords
    NightstreamFPrime.Gadgets.Poseidon2.Hash.inputChunks
  norm_num [NightstreamFPrime.Spec.Poseidon2.rate]

private theorem entryInvocations_eq_singleton (source : Nat) :
    PiRLCSamplerInvocations.entryInvocations
        (logicalWidth := Data.logicalWidth) (publicFits := Data.publicFits)
        source =
      [NightstreamFPrime.Export.Stage1.Invocations.invocation
        PiRLCSamplerInvocations.phase
        (PiRLCStarts.entryRowStart source)
        (PiRLCSamplerInvocations.sourceLogicalStart source)
        (entryPermutationState source)] := by
  unfold PiRLCSamplerInvocations.entryInvocations
    PiRLCSamplerInvocations.entryTrace
    NightstreamFPrime.Lifecycle.PiRLC.v1_1.TranscriptAbsorption.actions
  change
    (NightstreamFPrime.Export.Stage1.Invocations.compileActions
      PiRLCSamplerInvocations.phase (PiRLCStarts.entryRowStart source)
      (PiRLCSamplerInvocations.sourceLogicalStart source)
      (PiRLCSamplerInvocations.entryState
        (logicalWidth := Data.logicalWidth) (publicFits := Data.publicFits)
        source)
      [.absorb (entryWords source)]).invocations =
    [NightstreamFPrime.Export.Stage1.Invocations.invocation
      PiRLCSamplerInvocations.phase (PiRLCStarts.entryRowStart source)
      (PiRLCSamplerInvocations.sourceLogicalStart source)
      (NightstreamFPrime.Gadgets.Poseidon2.Hash.absorbE
        (PiRLCSamplerInvocations.entryState
          (logicalWidth := Data.logicalWidth) (publicFits := Data.publicFits)
          source) (entryWords source))]
  simp only [NightstreamFPrime.Export.Stage1.Invocations.compileActions]
  rw [entryInputChunks_eq]
  simp only [NightstreamFPrime.Export.Stage1.Invocations.compileBlocks,
    List.append_nil]

private theorem entryConstraints_eq_recipeConstraints (source : Nat) :
    Sampler.childConstraints
        (Sampler.Logical.entryCircuit (sourceInterface source) source)
        (Sampler.Logical.entryOffset (sourceOffset source)) =
      recipeConstraints (PiRLCSamplerInvocations.sourceLogicalStart source)
        (NightstreamFPrime.Gadgets.Poseidon2.Duplex.Formal.Owned.program
          (NightstreamFPrime.Lifecycle.PiRLC.v1_1.TranscriptAbsorption.ownedInterface
            (Sampler.Logical.entryInterface (sourceInterface source)) source)
          (PiRLCSamplerInvocations.sourceLogicalStart source)).recipes := by
  change flatConstraints
      (NightstreamFPrime.Gadgets.Poseidon2.Duplex.Formal.Owned.opsAt
        (NightstreamFPrime.Lifecycle.PiRLC.v1_1.TranscriptAbsorption.ownedInterface
          (Sampler.Logical.entryInterface (sourceInterface source)) source)
        (PiRLCSamplerInvocations.sourceLogicalStart source)) = _
  rw [NightstreamFPrime.Gadgets.Poseidon2.Duplex.Formal.Owned.flatConstraints_opsAt]
  have noAssertions :
      NightstreamFPrime.Gadgets.Poseidon2.Duplex.Formal.Owned.allAssertions
        (NightstreamFPrime.Lifecycle.PiRLC.v1_1.TranscriptAbsorption.ownedInterface
          (Sampler.Logical.entryInterface (sourceInterface source)) source)
        (PiRLCSamplerInvocations.sourceLogicalStart source) = [] := by
    rfl
  rw [noAssertions, List.append_nil]

private theorem entryProgramRecipes_eq (source : Nat) :
    (NightstreamFPrime.Gadgets.Poseidon2.Duplex.Formal.Owned.program
      (NightstreamFPrime.Lifecycle.PiRLC.v1_1.TranscriptAbsorption.ownedInterface
        (Sampler.Logical.entryInterface (sourceInterface source)) source)
      (PiRLCSamplerInvocations.sourceLogicalStart source)).recipes =
    (NightstreamFPrime.Gadgets.Poseidon2.Permutation.compile
      (PiRLCSamplerInvocations.sourceLogicalStart source)
      (entryPermutationState source)
      NightstreamFPrime.Gadgets.Poseidon2.Permutation.schedule).recipes := by
  unfold NightstreamFPrime.Gadgets.Poseidon2.Duplex.Formal.Owned.program
    NightstreamFPrime.Lifecycle.PiRLC.v1_1.TranscriptAbsorption.ownedInterface
    NightstreamFPrime.Lifecycle.PiRLC.v1_1.TranscriptAbsorption.actions
  change
    (NightstreamFPrime.Gadgets.Poseidon2.Duplex.Formal.compile
      (PiRLCSamplerInvocations.sourceLogicalStart source)
      (PiRLCSamplerInvocations.entryState
        (logicalWidth := Data.logicalWidth) (publicFits := Data.publicFits)
        source)
      [.absorb (entryWords source)]).recipes =
    (NightstreamFPrime.Gadgets.Poseidon2.Permutation.compile
      (PiRLCSamplerInvocations.sourceLogicalStart source)
      (NightstreamFPrime.Gadgets.Poseidon2.Hash.absorbE
        (PiRLCSamplerInvocations.entryState
          (logicalWidth := Data.logicalWidth) (publicFits := Data.publicFits)
          source) (entryWords source))
      NightstreamFPrime.Gadgets.Poseidon2.Permutation.schedule).recipes
  simp only [NightstreamFPrime.Gadgets.Poseidon2.Duplex.Formal.compile]
  rw [entryInputChunks_eq]
  simp only [NightstreamFPrime.Gadgets.Poseidon2.Hash.compileAbsorptions,
    List.append_nil]

private theorem entryWitnessLocal (source : Nat) :
    Spartan.piCcsPhaseOffset ≤
      PiRLCSamplerInvocations.sourceLogicalStart source := by
  unfold PiRLCSamplerInvocations.sourceLogicalStart
    PiRLCStarts.samplerSourceLogicalStart PiRLCStarts.samplerLogicalStart
    NightstreamFPrime.Lifecycle.PiRLC.v1_1.Formal.samplerOffset
    PiRLCStarts.phaseLogicalStart PiRLCInputs.phaseOffset
  norm_num [Spartan.piCcsPhaseOffset]
  omega

private theorem entryWords_affine (source : Nat) :
    NightstreamFPrime.Layout.Poseidon2.ListAffine (entryWords source) := by
  intro expression member
  unfold entryWords
    NightstreamFPrime.Lifecycle.PiRLC.v1_1.TranscriptAbsorption.constantWords
      at member
  rcases List.mem_map.mp member with ⟨word, _, rfl⟩
  exact R1CS.isAffine_const word

private theorem entryPermutationState_affine (source : Nat) :
    NightstreamFPrime.Layout.Poseidon2.StateAffine
      (entryPermutationState source) := by
  apply NightstreamFPrime.Layout.Poseidon2.absorbE_affine
  · exact PiRLCSamplerInvocations.entryState_affine source
  · exact entryWords_affine source

/-- One selected scalar-entry child constructs its sole compact Poseidon2
invocation, including the exact 592 internal witness rows. -/
theorem remappedPacket_implies_entryPermutations (env : Env)
    (packets : PiRLCPackageCompleteness.RemappedPacketRowsHold env)
    (source : Fin SamplerChain.Logical.sourceCount) :
    ∀ current ∈ PiRLCSamplerInvocations.entryInvocations
      (logicalWidth := Data.logicalWidth) (publicFits := Data.publicFits)
      source.val,
      PermutationInvocationHolds (PilotData.circuitPackage ()) current env := by
  intro current member
  rw [entryInvocations_eq_singleton] at member
  simp only [List.mem_singleton] at member
  subst current
  have segments := remappedPacket_implies_sourceSegments env packets source
  have expanded := segments
  simp only [Sampler.childConstraintLists, R1CS.SegmentsHold] at expanded
  have entryRows := expanded.1
  have sourceHolds := R1CS.lowerConstraints_sound (Spartan.pullback env)
    _ _ entryRows
  change ConstraintsHold (Spartan.pullback env)
    (Sampler.childConstraints
      (Sampler.Logical.entryCircuit (sourceInterface source.val) source.val)
      (Sampler.Logical.entryOffset (sourceOffset source.val))) at sourceHolds
  rw [entryConstraints_eq_recipeConstraints,
    entryProgramRecipes_eq] at sourceHolds
  apply PermutationCompilerTransport.invocation_complete_of_sourceConstraints
  · exact entryWitnessLocal source.val
  · exact entryPermutationState_affine source.val
  · exact sourceHolds

/-- One selected digest-window permutation child constructs its exact compact
canonical Poseidon2 invocation, including all 592 internal witness rows. -/
theorem remappedPacket_implies_windowPermutation (env : Env)
    (packets : PiRLCPackageCompleteness.RemappedPacketRowsHold env)
    (source : Fin SamplerChain.Logical.sourceCount) (round : Fin 8) :
    PermutationInvocationHolds (PilotData.circuitPackage ())
      (PiRLCSamplerInvocations.windowInvocation
        (logicalWidth := Data.logicalWidth) (publicFits := Data.publicFits)
        source.val round.val) env := by
  have windowRows := remappedPacket_implies_windowRows env packets source round
  have childSegments := DigestWindow.rowsHold_implies_childSegments
    (windowInterface source.val round.val)
    (windowOffset source.val round.val) (Spartan.pullback env)
    (PiRLCStarts.samplerFreshStart + source.val * 43743 +
      round.val * 1212) windowRows
  have permutationRows := DigestWindow.childSegments_imply_permutationLogical
    (windowInterface source.val round.val)
    (windowOffset source.val round.val) (Spartan.pullback env)
    (PiRLCStarts.samplerFreshStart + source.val * 43743 +
      round.val * 1212)
    (laneFreshCount source.val round.val) childSegments
  have sourceHolds := R1CS.lowerConstraints_sound (Spartan.pullback env)
    _ _ permutationRows
  unfold NightstreamFPrime.Layout.Poseidon2.PermutationOwned.logicalConstraints
    at sourceHolds
  rw [NightstreamFPrime.Layout.Poseidon2.PermutationOwned.Owned.flatConstraints_operations]
    at sourceHolds
  unfold NightstreamFPrime.Gadgets.Poseidon2.Permutation.Owned.program
    at sourceHolds
  change ConstraintsHold (Spartan.pullback env)
    (recipeConstraints
      (PiRLCStarts.digestPermutationLogicalStart source.val round.val)
      (NightstreamFPrime.Gadgets.Poseidon2.Permutation.compile
        (PiRLCStarts.digestPermutationLogicalStart source.val round.val)
        (PiRLCSamplerInvocations.windowState
          (logicalWidth := Data.logicalWidth) (publicFits := Data.publicFits)
          source.val round.val)
        NightstreamFPrime.Gadgets.Poseidon2.Permutation.schedule).recipes)
      at sourceHolds
  unfold PiRLCSamplerInvocations.windowInvocation
  apply PermutationCompilerTransport.invocation_complete_of_sourceConstraints
  · exact windowWitnessLocal source.val round.val
  · exact PiRLCSamplerInvocations.windowState_affine source.val round.val
  · exact sourceHolds

/-- The exact remapped sampler packet constructs all 153 canonical compact
Poseidon2 invocations in source order. -/
theorem remappedPacket_implies_permutationInvocations (env : Env)
    (packets : PiRLCPackageCompleteness.RemappedPacketRowsHold env) :
    ∀ current ∈ PiRLCSamplerInvocations.invocations
      (logicalWidth := Data.logicalWidth) (publicFits := Data.publicFits),
      PermutationInvocationHolds (PilotData.circuitPackage ()) current env := by
  intro current member
  unfold PiRLCSamplerInvocations.invocations at member
  rcases List.mem_flatMap.mp member with
    ⟨source, sourceMember, sourceInvocationMember⟩
  have sourceLt := List.mem_range.mp sourceMember
  let sourceFin : Fin SamplerChain.Logical.sourceCount :=
    ⟨source, by
      simpa [PiRLCSamplerInvocations.sourceCount,
        NightstreamFPrime.Lifecycle.PiRLC.v1_1.SamplerChain.sourceCount_eq]
        using sourceLt⟩
  unfold PiRLCSamplerInvocations.sourceInvocations at sourceInvocationMember
  rcases List.mem_append.mp sourceInvocationMember with
      entryMember | windowMember
  · exact remappedPacket_implies_entryPermutations env packets sourceFin
      current entryMember
  · unfold PiRLCSamplerInvocations.windowInvocations at windowMember
    rcases List.mem_map.mp windowMember with
      ⟨round, roundMember, rfl⟩
    have roundLt := List.mem_range.mp roundMember
    let roundFin : Fin 8 :=
      ⟨round, by
        simpa [PiRLCSamplerInvocations.digestRoundCount] using roundLt⟩
    exact remappedPacket_implies_windowPermutation env packets sourceFin
      roundFin

/-- One selected digest window projects to one exact source-column digest
lane lowering. -/
theorem remappedPacket_implies_laneSourceRows (env : Env)
    (packets : PiRLCPackageCompleteness.RemappedPacketRowsHold env)
    (source : Fin SamplerChain.Logical.sourceCount) (round : Fin 8)
    (lane : Fin 4) :
    R1CS.RowsHold (Spartan.pullback env)
      (R1CS.lowerConstraints
        (PiRLCSamplerOrdinaryRows.laneConstraints
          (logicalWidth := Data.logicalWidth) (publicFits := Data.publicFits)
          source.val round.val lane)
        (PiRLCStarts.digestLaneFreshStart source.val round.val lane.val)).rows := by
  have windowRows := remappedPacket_implies_windowRows env packets source round
  have childSegments := DigestWindow.rowsHold_implies_childSegments
    (windowInterface source.val round.val)
    (windowOffset source.val round.val) (Spartan.pullback env)
    (PiRLCStarts.samplerFreshStart + source.val * 43743 +
      round.val * 1212) windowRows
  have laneRows := DigestWindow.childSegments_imply_lane
    (windowInterface source.val round.val)
    (windowOffset source.val round.val) (Spartan.pullback env)
    (PiRLCStarts.samplerFreshStart + source.val * 43743 +
      round.val * 1212)
    (laneFreshCount source.val round.val) childSegments lane
  rw [PiRLCSamplerOrdinaryRows.laneConstraints_eq_fromCircuit]
  simpa [PiRLCStarts.digestLaneFreshStart,
    PiRLCStarts.windowFreshStart, PiRLCStarts.samplerSourceFreshStart]
    using laneRows

/-- Exact source-column digest-lane rows construct the canonical compiled
ordinary-row packet after the Stage 1 column remap. -/
theorem remappedPacket_implies_laneRows (env : Env)
    (packets : PiRLCPackageCompleteness.RemappedPacketRowsHold env)
    (source : Fin SamplerChain.Logical.sourceCount) (round : Fin 8)
    (lane : Fin 4) :
    R1CS.RowsHold env
      ((PiRLCSamplerOrdinaryRows.laneRows
        (logicalWidth := Data.logicalWidth) (publicFits := Data.publicFits)
        source.val round.val lane).map Rows.CompiledRow.toR1CS) := by
  rw [PiRLCSamplerOrdinaryRows.laneRows_toR1CS]
  apply (Spartan.remapRows_hold env _).mpr
  exact remappedPacket_implies_laneSourceRows env packets source round lane

/-- One exact sampler source projects to its complete First54 selector
lowering at the canonical source and fresh starts. -/
theorem remappedPacket_implies_selectorSourceRows (env : Env)
    (packets : PiRLCPackageCompleteness.RemappedPacketRowsHold env)
    (source : Fin SamplerChain.Logical.sourceCount) :
    R1CS.RowsHold (Spartan.pullback env)
      (R1CS.lowerConstraints
        (First54.logicalConstraints (sourceInterface source.val) source.val
          (PiRLCStarts.samplerSourceLogicalStart source.val)
          (PiRLCStarts.selectorLogicalStart source.val))
        (PiRLCStarts.selectorFreshStart source.val)).rows := by
  have segments := remappedPacket_implies_sourceSegments env packets source
  have selectorRows := Sampler.childSegments_imply_selector
    (sourceInterface source.val) source.val (sourceOffset source.val)
    (Spartan.pullback env)
    (PiRLCStarts.samplerFreshStart + source.val * 43743)
    (entryFreshCount source.val) (windowFreshCount source.val) segments
  simpa [sourceOffset, PiRLCStarts.samplerSourceLogicalStart,
    PiRLCStarts.selectorLogicalStart, PiRLCStarts.selectorFreshStart,
    PiRLCStarts.samplerSourceFreshStart] using selectorRows

/-- The exact selector child lowering projects to its final fail-closed
assertion after the complete 64-round fresh prefix. -/
theorem remappedPacket_implies_selectorFinalSourceRows (env : Env)
    (packets : PiRLCPackageCompleteness.RemappedPacketRowsHold env)
    (source : Fin SamplerChain.Logical.sourceCount) :
    R1CS.RowsHold (Spartan.pullback env)
      (R1CS.lowerConstraints
        [PiRLCSamplerOrdinaryRows.selectorFinalConstraint source.val]
        (PiRLCStarts.selectorFreshStart source.val + 34047)).rows := by
  have segments := remappedPacket_implies_sourceSegments env packets source
  have selectorRows := Sampler.childSegments_imply_selector
    (sourceInterface source.val) source.val (sourceOffset source.val)
    (Spartan.pullback env)
    (PiRLCStarts.samplerFreshStart + source.val * 43743)
    (entryFreshCount source.val) (windowFreshCount source.val) segments
  change R1CS.RowsHold (Spartan.pullback env)
    (R1CS.lowerConstraints
      (First54.logicalConstraints (sourceInterface source.val) source.val
        (sourceOffset source.val)
        (Sampler.Logical.selectorOffset (sourceOffset source.val)))
      (PiRLCStarts.samplerFreshStart + source.val * 43743 + 9696)).rows
    at selectorRows
  have prefixFresh :
      R1CS.totalFreshCount
          (flatConstraints
            (NightstreamFPrime.Gadgets.Sampling.First54.roundOpsPrefix
              (First54.selectorInterface (sourceInterface source.val)
                source.val (sourceOffset source.val))
              (Sampler.Logical.selectorOffset (sourceOffset source.val))
              NightstreamFPrime.Gadgets.Sampling.First54.candidateCount)) =
        34047 := by
    have total := First54.totalFreshCount_eq
      (sourceInterface source.val) source.val (sourceOffset source.val)
      (Sampler.Logical.selectorOffset (sourceOffset source.val))
    rw [First54.logicalConstraints_eq_rounds_append_final,
      R1CS.totalFreshCount_append] at total
    have finalFresh : R1CS.totalFreshCount
        [NightstreamFPrime.Gadgets.Sampling.First54.finalFull
          (Sampler.Logical.selectorOffset (sourceOffset source.val)) - 1] = 0 := by
      change R1CS.constraintFreshCount
        (PiRLCSamplerOrdinaryRows.selectorFinalConstraint source.val) = 0
      exact selectorFinalFreshCount source.val
    rw [finalFresh, Nat.add_zero] at total
    exact total
  rw [First54.logicalConstraints_eq_rounds_append_final,
    R1CS.lowerConstraints_append_rows] at selectorRows
  have finalRows := (R1CS.rowsHold_append (Spartan.pullback env) _ _).mp
    selectorRows |>.2
  rw [prefixFresh] at finalRows
  simpa [PiRLCStarts.selectorFreshStart,
    PiRLCStarts.samplerSourceFreshStart]
    using finalRows

/-- The source selector assertion constructs its exact canonical compiled
ordinary row after the Stage 1 column remap. -/
theorem remappedPacket_implies_selectorFinalRows (env : Env)
    (packets : PiRLCPackageCompleteness.RemappedPacketRowsHold env)
    (source : Fin SamplerChain.Logical.sourceCount) :
    R1CS.RowsHold env
      ((PiRLCSamplerOrdinaryRows.selectorFinalRows source.val).map
        Rows.CompiledRow.toR1CS) := by
  rw [PiRLCSamplerOrdinaryRows.selectorFinalRows_toR1CS]
  apply (Spartan.remapRows_hold env _).mpr
  exact remappedPacket_implies_selectorFinalSourceRows env packets source

/-- The exact remapped sampler packet constructs every canonical ordinary
sampler row: all digest lanes and every final fail-closed selector assertion. -/
theorem remappedPacket_implies_ordinaryRows (env : Env)
    (packets : PiRLCPackageCompleteness.RemappedPacketRowsHold env) :
    R1CS.RowsHold env
      ((PiRLCSamplerOrdinaryRows.rows
        (logicalWidth := Data.logicalWidth)
        (publicFits := Data.publicFits)).map Rows.CompiledRow.toR1CS) := by
  intro row member
  rcases List.mem_map.mp member with ⟨compiled, compiledMember, rfl⟩
  unfold PiRLCSamplerOrdinaryRows.rows at compiledMember
  rcases List.mem_flatMap.mp compiledMember with
    ⟨source, sourceMember, sourceRowMember⟩
  have sourceLt := List.mem_range.mp sourceMember
  let sourceFin : Fin SamplerChain.Logical.sourceCount :=
    ⟨source, by
      simpa [PiRLCSamplerOrdinaryRows.sourceCount,
        NightstreamFPrime.Lifecycle.PiRLC.v1_1.SamplerChain.sourceCount_eq]
        using sourceLt⟩
  unfold PiRLCSamplerOrdinaryRows.sourceRows at sourceRowMember
  rcases List.mem_append.mp sourceRowMember with
      digestMember | selectorMember
  · rcases List.mem_flatMap.mp digestMember with
      ⟨round, roundMember, windowMember⟩
    have roundLt := List.mem_range.mp roundMember
    let roundFin : Fin 8 :=
      ⟨round, by
        simpa [PiRLCSamplerOrdinaryRows.digestRoundCount] using roundLt⟩
    unfold PiRLCSamplerOrdinaryRows.windowRows at windowMember
    rcases List.mem_flatMap.mp windowMember with
      ⟨lane, _, laneMember⟩
    have laneRows := remappedPacket_implies_laneRows env packets sourceFin
      roundFin lane
    apply laneRows
    exact List.mem_map.mpr ⟨compiled, laneMember, rfl⟩
  · have selectorRows := remappedPacket_implies_selectorFinalRows env
      packets sourceFin
    apply selectorRows
    exact List.mem_map.mpr ⟨compiled, selectorMember, rfl⟩

end NightstreamFPrime.Export.Stage1.PiRLCSamplerCompleteness
