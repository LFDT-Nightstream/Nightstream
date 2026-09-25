import NightstreamFPrime.Export.Stage1.InvocationLastOutput
import NightstreamFPrime.Export.Stage1.PiCCSProjection
import NightstreamFPrime.Export.Stage1.PiRLCSamplerRows

/-!
Owns the recipe-free symbolic projection of PiRLC sampler transcript states.

The full Duplex compiler remains the semantic authority. The exported
equalities prove that this bounded projection returns the same expressions
without constructing unused Poseidon2 recipes.
-/

namespace NightstreamFPrime.Export.Stage1.PiRLCSamplerProjection

open NightstreamFPrime.Gadgets.Poseidon2
open NightstreamFPrime.Gadgets.Poseidon2.Duplex
open NightstreamFPrime.Lifecycle.PiRLC.v1_1

def fastOwnedOutput (interface : Formal.Owned.Interface) (offset : Nat) :
    Layer.EState :=
  (Formal.compileWiringLazy offset (fun _ => interface.initial offset)
    (interface.actions offset)).output

theorem fastOwnedOutput_eq (interface : Formal.Owned.Interface)
    (offset : Nat) :
    fastOwnedOutput interface offset = Formal.Owned.output interface offset := by
  unfold fastOwnedOutput Formal.Owned.output Formal.Owned.program
  rw [Formal.compileWiringLazy_eq offset
    (fun _ => interface.initial offset) (interface.initial offset)
    (interface.actions offset) rfl]
  exact (Formal.compileWiring_matches offset (interface.initial offset)
    (interface.actions offset)).2

def fastEntryOutput (interface : Sampler.Interface) (coordinate offset : Nat) :
    Layer.EState :=
  fastOwnedOutput
    (TranscriptAbsorption.ownedInterface
      (Sampler.entryInterface interface) coordinate)
    (Sampler.entryOffset offset)

theorem fastEntryOutput_eq (interface : Sampler.Interface)
    (coordinate offset : Nat) :
    fastEntryOutput interface coordinate offset =
      TranscriptAbsorption.output (Sampler.entryInterface interface)
        coordinate (Sampler.entryOffset offset) := by
  unfold fastEntryOutput TranscriptAbsorption.output
  exact fastOwnedOutput_eq _ _

/-- Closed-form initial state of one chained scalar sampler. A successor
starts from the fixed output columns of the previous source's eighth digest
permutation; no prior sampler contents need to be reconstructed. -/
def fastChainedEntryStateFrom (initialState : Layer.EState)
    (offset source : Nat) : Layer.EState :=
  match source with
  | 0 => initialState
  | previous + 1 =>
      Permutation.scheduleOutput
        (DigestWindow.permutationOffset
          (Sampler.windowOffset (SamplerChain.sourceOffset offset previous)
            (Sampler.digestRoundCount - 1)))

theorem fastChainedEntryStateFrom_eq (interface : SamplerChain.Interface)
    (initialState : Layer.EState) (offset source : Nat)
    (initialEq : initialState = interface.initialState offset) :
    fastChainedEntryStateFrom initialState offset source =
      SamplerChain.stateAtExpr interface offset source := by
  cases source with
  | zero => exact initialEq
  | succ previous =>
      rw [fastChainedEntryStateFrom, SamplerChain.stateAtExpr_succ]
      rfl

def fastChainedEntryState (interface : SamplerChain.Interface)
    (offset source : Nat) : Layer.EState :=
  fastChainedEntryStateFrom (interface.initialState offset) offset source

theorem fastChainedEntryState_eq (interface : SamplerChain.Interface)
    (offset source : Nat) :
    fastChainedEntryState interface offset source =
      SamplerChain.stateAtExpr interface offset source := by
  exact fastChainedEntryStateFrom_eq interface
    (interface.initialState offset) offset source rfl

/-- Recipe-free entry absorption from a caller-supplied direct state. -/
def fastEntryOutputFromState (state : Layer.EState)
    (coordinate offset : Nat) : Layer.EState :=
  (Formal.compileWiringLazy offset (fun _ => state)
    (TranscriptAbsorption.actions coordinate)).output

/-- One scalar-entry absorption owns exactly one permutation, so its output
variables depend only on the compiler start. -/
theorem fastEntryOutputFromState_eq_scheduleOutput (state : Layer.EState)
    (coordinate offset : Nat) :
    fastEntryOutputFromState state coordinate offset =
      Permutation.scheduleOutput offset := by
  have chunkCount :
      (Hash.inputChunks (TranscriptAbsorption.constantWords
        (TranscriptAbsorption.frameWords coordinate))).length = 1 := by
    simp [Hash.inputChunks, TranscriptAbsorption.constantWords,
      TranscriptAbsorption.frameWords, Spec.Poseidon2.rate]
  calc
    fastEntryOutputFromState state coordinate offset =
        (Formal.compile offset state
          (TranscriptAbsorption.actions coordinate)).output := by
      unfold fastEntryOutputFromState
      rw [Formal.compileWiringLazy_eq offset (fun _ => state) state
        (TranscriptAbsorption.actions coordinate) rfl]
      exact (Formal.compileWiring_matches offset state
        (TranscriptAbsorption.actions coordinate)).2
    _ = Permutation.scheduleOutput offset := by
      have positive : InvocationLastOutput.ActionsPositive
          (TranscriptAbsorption.actions coordinate) := by
        intro action member
        simp only [TranscriptAbsorption.actions, List.mem_singleton] at member
        subst action
        simp [Invocations.Action.invocationCount, chunkCount]
      have endpoint :=
        InvocationLastOutput.compileActions_state_scheduleOutput
          0 0 offset state (TranscriptAbsorption.actions coordinate)
          (by simp [TranscriptAbsorption.actions]) positive
      rw [Invocations.compileActions_state_eq] at endpoint
      simpa [TranscriptAbsorption.actions, Invocations.invocationCount,
        Invocations.Action.invocationCount, chunkCount] using endpoint

theorem fastEntryOutputFromState_eq (interface : Sampler.Interface)
    (state : Layer.EState) (coordinate offset : Nat)
    (stateEq : state = interface.initialState offset) :
    fastEntryOutputFromState state coordinate offset =
      TranscriptAbsorption.output (Sampler.entryInterface interface)
        coordinate offset := by
  unfold fastEntryOutputFromState TranscriptAbsorption.output
    TranscriptAbsorption.ownedInterface Sampler.entryInterface
    Formal.Owned.output Formal.Owned.program
  rw [Formal.compileWiringLazy_eq offset (fun _ => state)
    (interface.initialState offset) (TranscriptAbsorption.actions coordinate)
    stateEq]
  exact (Formal.compileWiring_matches offset (interface.initialState offset)
    (TranscriptAbsorption.actions coordinate)).2

def fastChainedEntryOutput (interface : SamplerChain.Interface)
    (offset coordinate source : Nat) : Layer.EState :=
  fastEntryOutputFromState (fastChainedEntryState interface offset source)
    coordinate (SamplerChain.sourceOffset offset source)

theorem fastChainedEntryOutput_eq (interface : SamplerChain.Interface)
    (offset coordinate source : Nat) :
    fastChainedEntryOutput interface offset coordinate source =
      TranscriptAbsorption.output
        (Sampler.entryInterface
          (SamplerChain.childInterface interface offset source))
        coordinate (SamplerChain.sourceOffset offset source) := by
  apply fastEntryOutputFromState_eq
  simpa [SamplerChain.childInterface] using
    fastChainedEntryState_eq interface offset source

/-- Closed-form initial state for every digest window in one chained source. -/
def fastChainedWindowInitialState (interface : SamplerChain.Interface)
    (offset coordinate source round : Nat) : Layer.EState :=
  match round with
  | 0 => fastChainedEntryOutput interface offset coordinate source
  | previous + 1 =>
      Permutation.scheduleOutput
        (DigestWindow.permutationOffset
          (Sampler.windowOffset (SamplerChain.sourceOffset offset source)
            previous))

theorem fastChainedWindowInitialState_eq (interface : SamplerChain.Interface)
    (offset coordinate source round : Nat) :
    fastChainedWindowInitialState interface offset coordinate source round =
      Sampler.windowInitialState
        (SamplerChain.childInterface interface offset source)
        coordinate (SamplerChain.sourceOffset offset source) round := by
  cases round with
  | zero => exact fastChainedEntryOutput_eq interface offset coordinate source
  | succ previous => rfl

/-! ## Fixed production projection -/

open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint

variable {logicalWidth : Nat}
  {publicFits : ringDegree * publicRingColumns ≤
    Phi81CarrierLayout.carrierWidth logicalWidth}

def productionInitialState : Layer.EState :=
  PiCCSProjection.fastOutputState logicalWidth publicFits

theorem productionInitialState_eq :
    productionInitialState (logicalWidth := logicalWidth)
        (publicFits := publicFits) =
      (PiRLCSamplerRows.samplerInterface
        (logicalWidth := logicalWidth) (publicFits := publicFits)).initialState
        NightstreamFPrime.Layout.Stage1.PiRLCStarts.samplerLogicalStart := by
  calc
    productionInitialState (logicalWidth := logicalWidth)
        (publicFits := publicFits) =
        (PiCCSInvocations.outputTrace logicalWidth publicFits).state :=
      PiCCSProjection.fastOutputState_eq logicalWidth publicFits
    _ = (PiCCSInvocations.outputSemanticTrace logicalWidth publicFits).state :=
      congrArg Invocations.Trace.state
        (PiCCSInvocations.outputTrace_eq_semantic logicalWidth publicFits)
    _ = NightstreamFPrime.Lifecycle.PiCCS.v1_1.OutputBinding.finalState
          (PiCCSInvocations.outputInterface logicalWidth publicFits)
          PiCCSInvocations.outputWitnessStart :=
      PiCCSInvocations.outputSemanticTrace_state_matches logicalWidth publicFits
    _ = _ := by rfl

def fastProductionEntryState (source : Nat) : Layer.EState :=
  fastChainedEntryStateFrom
    (productionInitialState (logicalWidth := logicalWidth)
      (publicFits := publicFits))
    NightstreamFPrime.Layout.Stage1.PiRLCStarts.samplerLogicalStart source

theorem fastProductionEntryState_eq (source : Nat) :
    fastProductionEntryState (logicalWidth := logicalWidth)
        (publicFits := publicFits) source =
      SamplerChain.stateAtExpr
        (PiRLCSamplerRows.samplerInterface
          (logicalWidth := logicalWidth) (publicFits := publicFits))
        NightstreamFPrime.Layout.Stage1.PiRLCStarts.samplerLogicalStart
        source := by
  apply fastChainedEntryStateFrom_eq
  exact productionInitialState_eq

def fastProductionEntryOutput (source : Nat) : Layer.EState :=
  fastEntryOutputFromState
    (fastProductionEntryState (logicalWidth := logicalWidth)
      (publicFits := publicFits) source)
    source
    (SamplerChain.sourceOffset
      NightstreamFPrime.Layout.Stage1.PiRLCStarts.samplerLogicalStart source)

theorem fastProductionEntryOutput_eq_scheduleOutput (source : Nat) :
    fastProductionEntryOutput (logicalWidth := logicalWidth)
        (publicFits := publicFits) source =
      Permutation.scheduleOutput
        (NightstreamFPrime.Layout.Stage1.PiRLCStarts.samplerSourceLogicalStart
          source) := by
  unfold fastProductionEntryOutput
  rw [fastEntryOutputFromState_eq_scheduleOutput]
  rfl

theorem fastProductionEntryOutput_eq (source : Nat) :
    fastProductionEntryOutput (logicalWidth := logicalWidth)
        (publicFits := publicFits) source =
      TranscriptAbsorption.output
        (Sampler.entryInterface
          (SamplerChain.childInterface
            (PiRLCSamplerRows.samplerInterface
              (logicalWidth := logicalWidth) (publicFits := publicFits))
            NightstreamFPrime.Layout.Stage1.PiRLCStarts.samplerLogicalStart
            source))
        source
        (SamplerChain.sourceOffset
          NightstreamFPrime.Layout.Stage1.PiRLCStarts.samplerLogicalStart
          source) := by
  apply fastEntryOutputFromState_eq
  simpa [SamplerChain.childInterface] using
    fastProductionEntryState_eq (logicalWidth := logicalWidth)
      (publicFits := publicFits) source

def fastProductionWindowInitialState (source round : Nat) : Layer.EState :=
  match round with
  | 0 => fastProductionEntryOutput (logicalWidth := logicalWidth)
      (publicFits := publicFits) source
  | previous + 1 =>
      Permutation.scheduleOutput
        (DigestWindow.permutationOffset
          (Sampler.windowOffset
            (SamplerChain.sourceOffset
              NightstreamFPrime.Layout.Stage1.PiRLCStarts.samplerLogicalStart
              source)
            previous))

theorem fastProductionWindowInitialState_eq (source round : Nat) :
    fastProductionWindowInitialState (logicalWidth := logicalWidth)
        (publicFits := publicFits) source round =
      Sampler.windowInitialState
        (SamplerChain.childInterface
          (PiRLCSamplerRows.samplerInterface
            (logicalWidth := logicalWidth) (publicFits := publicFits))
          NightstreamFPrime.Layout.Stage1.PiRLCStarts.samplerLogicalStart source)
        source
        (SamplerChain.sourceOffset
          NightstreamFPrime.Layout.Stage1.PiRLCStarts.samplerLogicalStart source)
        round := by
  cases round with
  | zero => exact fastProductionEntryOutput_eq source
  | succ previous => rfl

def fastWindowInitialState (interface : Sampler.Interface)
    (coordinate offset round : Nat) : Layer.EState :=
  match round with
  | 0 => fastEntryOutput interface coordinate offset
  | previous + 1 =>
      Permutation.scheduleOutput
        (DigestWindow.permutationOffset (Sampler.windowOffset offset previous))

theorem fastWindowInitialState_eq (interface : Sampler.Interface)
    (coordinate offset round : Nat) :
    fastWindowInitialState interface coordinate offset round =
      Sampler.windowInitialState interface coordinate offset round := by
  cases round with
  | zero => exact fastEntryOutput_eq interface coordinate offset
  | succ previous => rfl

end NightstreamFPrime.Export.Stage1.PiRLCSamplerProjection
