import NightstreamFPrime.Export.Stage1.PiRLCPackageCompleteness
import NightstreamFPrime.Export.Stage1.PermutationCompilerTransport
import NightstreamFPrime.Export.Stage1.PiRLCSamplerInvocations
import NightstreamFPrime.Export.Stage1.PiRLCSamplerOrdinaryRows
import NightstreamFPrime.Layout.PiRLC.v1_2.SamplerSegments

/-!
Owns the constructive bridge from the exact PiRLC sampler physical packet to
the compact sampler package. Child selection stays structural and does not
unfold the opaque scalar children.
-/

namespace NightstreamFPrime.Export.Stage1.PiRLCSamplerCompleteness

open NightstreamFPrime.Circuit
open NightstreamFPrime.Export.Package
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.PiRLC.v1_2
open NightstreamFPrime.Layout.PiRLC.v1_2.Leaves
open NightstreamFPrime.Layout.Stage1
open NightstreamFPrime.Lifecycle

def chainInterface : SamplerChain.Logical.Interface :=
  NightstreamFPrime.Lifecycle.PiRLC.v1_2.Formal.samplerInterface
    (NightstreamFPrime.Lifecycle.PiRLC.v1_2.Formal.atOffset
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
      144 := by
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
      source.val * 144 := by
  have countsEq :
      (List.ofFn fun current : Fin SamplerChain.Logical.sourceCount =>
        R1CS.totalFreshCount
          (SamplerChain.childConstraints chainInterface
            PiRLCStarts.samplerLogicalStart current.val)) =
        List.ofFn
          (fun _ : Fin SamplerChain.Logical.sourceCount => 144) := by
    apply congrArg List.ofFn
    funext current
    exact sourceFreshCount current
  rw [countsEq]
  exact sum_take_ofFn_const 144 source

/-- The remapped sampler packet projects to one exact source sampler lowering
under the final-column pullback. -/
theorem remappedPacket_implies_sourceRows (env : Env)
    (packets : PiRLCPackageCompleteness.RemappedPacketRowsHold env)
    (source : Fin SamplerChain.Logical.sourceCount) :
    R1CS.RowsHold (Spartan.pullback env)
      (R1CS.lowerConstraints
        (SamplerChain.childConstraints chainInterface
          PiRLCStarts.samplerLogicalStart source.val)
        (PiRLCStarts.samplerFreshStart + source.val * 144)).rows := by
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

theorem remappedPacket_implies_sourceChildren (env : Env)
    (packets : PiRLCPackageCompleteness.RemappedPacketRowsHold env)
    (source : Fin SamplerChain.Logical.sourceCount) :
    Sampler.ChildRows (sourceInterface source.val) source.val (sourceOffset source.val)
      (Spartan.pullback env) (PiRLCStarts.samplerFreshStart + source.val * 144) :=
  Sampler.rowsHold_implies_childRows _ _ _ _ _ (sourceInputs source.val)
    (remappedPacket_implies_sourceRows env packets source)

def entryWords (source : Nat) : List Expr :=
  NightstreamFPrime.Lifecycle.PiRLC.v1_2.TranscriptAbsorption.constantWords
    (NightstreamFPrime.Lifecycle.PiRLC.v1_2.TranscriptAbsorption.frameWords
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
    NightstreamFPrime.Lifecycle.PiRLC.v1_2.TranscriptAbsorption.constantWords
    NightstreamFPrime.Lifecycle.PiRLC.v1_2.TranscriptAbsorption.frameWords
    NightstreamFPrime.Gadgets.Poseidon2.Hash.inputChunks
  norm_num [NightstreamFPrime.Spec.Poseidon2.rate]

/-- The scalar-domain entry list contains exactly its one additive
Poseidon2 invocation, with the existing state and frame words. -/
theorem entryInvocations_eq_singleton (source : Nat) :
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
    NightstreamFPrime.Lifecycle.PiRLC.v1_2.TranscriptAbsorption.actions
  rw [PiRLCSamplerInvocations.fastEntryState_eq_entryState]
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
    (NightstreamFPrime.Lifecycle.PiRLC.v1_2.Sampler.entryOp
      (sourceInterface source) source (sourceOffset source)).flatConstraints =
      recipeConstraints (PiRLCSamplerInvocations.sourceLogicalStart source)
        (NightstreamFPrime.Gadgets.Poseidon2.Duplex.Formal.Owned.program
          (NightstreamFPrime.Lifecycle.PiRLC.v1_2.TranscriptAbsorption.ownedInterface
            (sourceInterface source) source)
          (PiRLCSamplerInvocations.sourceLogicalStart source)).recipes := by
  rw [NightstreamFPrime.Lifecycle.PiRLC.v1_2.Sampler.entryOp, Sampler.child_constraints,
    NightstreamFPrime.Lifecycle.PiRLC.v1_2.Sampler.entry, FormalCircuit.withConstantFootprint_main]
  change flatConstraints (NightstreamFPrime.Gadgets.Poseidon2.Duplex.Formal.Owned.opsAt
    (NightstreamFPrime.Lifecycle.PiRLC.v1_2.TranscriptAbsorption.ownedInterface
      (sourceInterface source) source) (PiRLCSamplerInvocations.sourceLogicalStart source)) = _
  rw [NightstreamFPrime.Gadgets.Poseidon2.Duplex.Formal.Owned.flatConstraints_opsAt]
  have noAssertions : NightstreamFPrime.Gadgets.Poseidon2.Duplex.Formal.Owned.allAssertions
      (NightstreamFPrime.Lifecycle.PiRLC.v1_2.TranscriptAbsorption.ownedInterface
        (sourceInterface source) source) (PiRLCSamplerInvocations.sourceLogicalStart source) = [] := rfl
  rw [noAssertions, List.append_nil]

private theorem entryProgramRecipes_eq (source : Nat) :
    (NightstreamFPrime.Gadgets.Poseidon2.Duplex.Formal.Owned.program
      (NightstreamFPrime.Lifecycle.PiRLC.v1_2.TranscriptAbsorption.ownedInterface
        (sourceInterface source) source)
      (PiRLCSamplerInvocations.sourceLogicalStart source)).recipes =
    (NightstreamFPrime.Gadgets.Poseidon2.Permutation.compile
      (PiRLCSamplerInvocations.sourceLogicalStart source)
      (entryPermutationState source)
      NightstreamFPrime.Gadgets.Poseidon2.Permutation.schedule).recipes := by
  unfold NightstreamFPrime.Gadgets.Poseidon2.Duplex.Formal.Owned.program
    NightstreamFPrime.Lifecycle.PiRLC.v1_2.TranscriptAbsorption.ownedInterface
    NightstreamFPrime.Lifecycle.PiRLC.v1_2.TranscriptAbsorption.actions
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
    NightstreamFPrime.Lifecycle.PiRLC.v1_2.Formal.samplerOffset
    PiRLCStarts.phaseLogicalStart PiRLCInputs.phaseOffset
    NightstreamFPrime.Lifecycle.PiRLC.v1_2.SamplerChain.sourceOffset
  norm_num [Spartan.piCcsPhaseOffset]
  omega

private theorem entryWords_affine (source : Nat) :
    NightstreamFPrime.Layout.Poseidon2.ListAffine (entryWords source) := by
  intro expression member
  unfold entryWords
    NightstreamFPrime.Lifecycle.PiRLC.v1_2.TranscriptAbsorption.constantWords
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
invocation, including the exact 1096 internal witness rows. -/
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
  have rows := (remappedPacket_implies_sourceChildren env packets source).entry
  have sourceHolds := R1CS.lowerConstraints_sound (Spartan.pullback env) _ _ rows
  rw [entryConstraints_eq_recipeConstraints,
    entryProgramRecipes_eq] at sourceHolds
  apply PermutationCompilerTransport.invocation_complete_of_sourceConstraints
  · exact entryWitnessLocal source.val
  · exact entryPermutationState_affine source.val
  · exact sourceHolds

/-- The scalar's sole advance permutation is the emitted permutation. -/
theorem remappedPacket_implies_advancePermutation (env : Env)
    (packets : PiRLCPackageCompleteness.RemappedPacketRowsHold env)
    (source : Fin SamplerChain.Logical.sourceCount) :
    PermutationInvocationHolds (PilotData.circuitPackage ())
      (PiRLCSamplerInvocations.advanceInvocation
        (logicalWidth := Data.logicalWidth) (publicFits := Data.publicFits) source.val) env := by
  have rows := (remappedPacket_implies_sourceChildren env packets source).advance
  have sourceHolds := R1CS.lowerConstraints_sound (Spartan.pullback env) _ _ rows
  rw [NightstreamFPrime.Lifecycle.PiRLC.v1_2.Sampler.advanceOp, Sampler.child_constraints] at sourceHolds
  change ConstraintsHold (Spartan.pullback env)
    (flatConstraints (NightstreamFPrime.Gadgets.Poseidon2.Permutation.Owned.operations
      (NightstreamFPrime.Lifecycle.PiRLC.v1_2.Sampler.advanceInterface
        (sourceInterface source.val) source.val (sourceOffset source.val))
      (PiRLCStarts.advanceLogicalStart source.val))) at sourceHolds
  rw [NightstreamFPrime.Gadgets.Poseidon2.Permutation.Owned.flatConstraints_operations] at sourceHolds
  unfold PiRLCSamplerInvocations.advanceInvocation
  rw [PiRLCSamplerInvocations.fastAdvanceState_eq]
  apply PermutationCompilerTransport.invocation_complete_of_sourceConstraints
  · have earlier := entryWitnessLocal source.val
    apply Nat.le_trans earlier
    change sourceOffset source.val ≤
      NightstreamFPrime.Lifecycle.PiRLC.v1_2.Sampler.advanceOffset (sourceOffset source.val)
    unfold NightstreamFPrime.Lifecycle.PiRLC.v1_2.Sampler.advanceOffset
      NightstreamFPrime.Lifecycle.PiRLC.v1_2.Sampler.rangeOffset
    omega
  · exact PiRLCSamplerInvocations.advanceState_affine source.val
  · exact sourceHolds

/-- The physical packet supplies both permutations for every scalar. -/
theorem remappedPacket_implies_permutationInvocations (env : Env)
    (packets : PiRLCPackageCompleteness.RemappedPacketRowsHold env) :
    ∀ current ∈ PiRLCSamplerInvocations.invocations
      (logicalWidth := Data.logicalWidth) (publicFits := Data.publicFits),
      PermutationInvocationHolds (PilotData.circuitPackage ()) current env := by
  intro current member
  obtain ⟨source, sourceMember, sourceInvocationMember⟩ := List.mem_flatMap.mp member
  have sourceLt := List.mem_range.mp sourceMember
  let sourceFin : Fin SamplerChain.Logical.sourceCount := ⟨source, by
    simpa only [PiRLCSamplerInvocations.sourceCount,
      NightstreamFPrime.Lifecycle.PiRLC.v1_2.SamplerChain.sourceCount_eq] using sourceLt⟩
  rcases List.mem_append.mp sourceInvocationMember with entryMember | advanceMember
  · exact remappedPacket_implies_entryPermutations env packets sourceFin current entryMember
  · have same := List.mem_singleton.mp advanceMember
    subst current
    exact remappedPacket_implies_advancePermutation env packets sourceFin

theorem remappedPacket_implies_rangeRows (env : Env)
    (packets : PiRLCPackageCompleteness.RemappedPacketRowsHold env)
    (source : Fin SamplerChain.Logical.sourceCount) :
    R1CS.RowsHold env ((PiRLCSamplerOrdinaryRows.rangeRows
      (logicalWidth := Data.logicalWidth) (publicFits := Data.publicFits) source.val).map
      Rows.CompiledRow.toR1CS) := by
  have held := (remappedPacket_implies_sourceChildren env packets source).range
  rw [PiRLCSamplerOrdinaryRows.rangeRows_toR1CS, Spartan.remapRows_hold]
  rw [PiRLCSamplerOrdinaryRows.rangeConstraints_eq]
  exact held

theorem remappedPacket_implies_wordRows (env : Env)
    (packets : PiRLCPackageCompleteness.RemappedPacketRowsHold env)
    (source : Fin SamplerChain.Logical.sourceCount) :
    R1CS.RowsHold env ((PiRLCSamplerOrdinaryRows.wordRows source.val).map Rows.CompiledRow.toR1CS) := by
  rw [PiRLCSamplerOrdinaryRows.wordRows_toR1CS, Spartan.remapRows_hold]
  exact (remappedPacket_implies_sourceChildren env packets source).words

/-- The physical packet supplies all checked range and coefficient-word rows. -/
theorem remappedPacket_implies_ordinaryRows (env : Env)
    (packets : PiRLCPackageCompleteness.RemappedPacketRowsHold env) :
    R1CS.RowsHold env ((PiRLCSamplerOrdinaryRows.rows
      (logicalWidth := Data.logicalWidth) (publicFits := Data.publicFits)).map Rows.CompiledRow.toR1CS) := by
  intro row member
  obtain ⟨compiled, compiledMember, rfl⟩ := List.mem_map.mp member
  obtain ⟨source, sourceMember, sourceRowMember⟩ := List.mem_flatMap.mp compiledMember
  have sourceLt := List.mem_range.mp sourceMember
  let sourceFin : Fin SamplerChain.Logical.sourceCount := ⟨source, by
    simpa only [PiRLCSamplerInvocations.sourceCount,
      NightstreamFPrime.Lifecycle.PiRLC.v1_2.SamplerChain.sourceCount_eq] using sourceLt⟩
  rcases List.mem_append.mp sourceRowMember with rangeMember | wordMember
  · exact remappedPacket_implies_rangeRows env packets sourceFin _
      (List.mem_map.mpr ⟨compiled, rangeMember, rfl⟩)
  · exact remappedPacket_implies_wordRows env packets sourceFin _
      (List.mem_map.mpr ⟨compiled, wordMember, rfl⟩)

end NightstreamFPrime.Export.Stage1.PiRLCSamplerCompleteness
