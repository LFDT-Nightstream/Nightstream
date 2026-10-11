import NightstreamFPrime.Export.Stage1.Invocations
import NightstreamFPrime.Export.Stage1.PiRLCSamplerProjection
import NightstreamFPrime.Export.Stage1.PiRLCSamplerRows

/-!
Owns the Poseidon2 invocation schedule for the production PiRLC sampler
chain.

Each of the 17 scalar samplers contains one domain-entry absorption and one
advance permutation. The schedule uses the same canonical 1096-row
template as the pilot and PiCCS transcript paths.
-/

namespace NightstreamFPrime.Export.Stage1.PiRLCSamplerInvocations

open NightstreamFPrime.Circuit
open NightstreamFPrime.Export.Package
open NightstreamFPrime.Export.Stage1.Invocations
open NightstreamFPrime.Gadgets.Poseidon2
open NightstreamFPrime.Gadgets.Poseidon2.Duplex
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.Poseidon2
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Lifecycle.PiRLC.v1_2
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint

variable {logicalWidth : Nat}
  {publicFits : ringDegree * publicRingColumns ≤
    Phi81CarrierLayout.carrierWidth logicalWidth}

def phase : Nat := 7
def sourceCount : Nat := 17

def chainInterface : SamplerChain.Interface :=
  PiRLCSamplerRows.samplerInterface
    (logicalWidth := logicalWidth) (publicFits := publicFits)

def sourceInterface (source : Nat) : Sampler.Interface :=
  SamplerChain.childInterface
    (chainInterface (logicalWidth := logicalWidth) (publicFits := publicFits))
    NightstreamFPrime.Layout.Stage1.PiRLCStarts.samplerLogicalStart source

def sourceLogicalStart (source : Nat) : Nat :=
  NightstreamFPrime.Layout.Stage1.PiRLCStarts.samplerSourceLogicalStart source

def entryState (source : Nat) : Invocations.EState :=
  ((sourceInterface (logicalWidth := logicalWidth)
      (publicFits := publicFits) source)).initialState
    (sourceLogicalStart source)

def fastEntryState (source : Nat) : Invocations.EState :=
  PiRLCSamplerProjection.fastProductionEntryState
    (logicalWidth := logicalWidth) (publicFits := publicFits) source

theorem fastEntryState_eq_entryState (source : Nat) :
    fastEntryState (logicalWidth := logicalWidth) (publicFits := publicFits)
        source =
      entryState (logicalWidth := logicalWidth) (publicFits := publicFits)
        source := by
  unfold fastEntryState entryState sourceInterface sourceLogicalStart
  exact PiRLCSamplerProjection.fastProductionEntryState_eq source

def entryTrace (source : Nat) : Trace :=
  compileActions phase
    (NightstreamFPrime.Layout.Stage1.PiRLCStarts.entryRowStart source)
    (sourceLogicalStart source)
    (fastEntryState (logicalWidth := logicalWidth) (publicFits := publicFits)
      source)
    (TranscriptAbsorption.actions source)

def entryInvocations (source : Nat) : List PermutationInvocation :=
  (entryTrace (logicalWidth := logicalWidth) (publicFits := publicFits)
    source).invocations

def advanceState (source : Nat) : Invocations.EState :=
  Sampler.enteredState (sourceInterface (logicalWidth := logicalWidth)
    (publicFits := publicFits) source) source (sourceLogicalStart source)

def fastAdvanceState (source : Nat) : Invocations.EState :=
  PiRLCSamplerProjection.fastProductionEntryOutput
    (logicalWidth := logicalWidth) (publicFits := publicFits) source

theorem fastAdvanceState_eq (source : Nat) :
    fastAdvanceState (logicalWidth := logicalWidth) (publicFits := publicFits) source =
      advanceState (logicalWidth := logicalWidth) (publicFits := publicFits) source := by
  unfold fastAdvanceState advanceState Sampler.enteredState sourceInterface sourceLogicalStart
  exact PiRLCSamplerProjection.fastProductionEntryOutput_eq source

def advanceInvocation (source : Nat) : PermutationInvocation :=
  invocation phase
    (NightstreamFPrime.Layout.Stage1.PiRLCStarts.advanceRowStart source)
    (NightstreamFPrime.Layout.Stage1.PiRLCStarts.advanceLogicalStart source)
    (fastAdvanceState (logicalWidth := logicalWidth) (publicFits := publicFits) source)

def sourceInvocations (source : Nat) : List PermutationInvocation :=
  entryInvocations (logicalWidth := logicalWidth) (publicFits := publicFits)
      source ++
    [advanceInvocation (logicalWidth := logicalWidth) (publicFits := publicFits) source]

def invocations : List PermutationInvocation :=
  (List.range sourceCount).flatMap
    (sourceInvocations (logicalWidth := logicalWidth) (publicFits := publicFits))

@[simp] theorem entryInvocations_length (source : Nat) :
    (entryInvocations (logicalWidth := logicalWidth)
      (publicFits := publicFits) source).length = 1 := by
  rw [entryInvocations, entryTrace, compileActions_invocations_length]
  norm_num [invocationCount, Action.invocationCount,
    TranscriptAbsorption.actions, TranscriptAbsorption.constantWords,
    TranscriptAbsorption.frameWords, Hash.inputChunks, Spec.Poseidon2.rate]

@[simp] theorem sourceInvocations_length (source : Nat) :
    (sourceInvocations (logicalWidth := logicalWidth)
      (publicFits := publicFits) source).length = 2 := by
  simp [sourceInvocations]

@[simp] theorem invocations_length :
    (invocations (logicalWidth := logicalWidth)
      (publicFits := publicFits)).length = 34 := by
  simp [invocations, sourceCount]

theorem entryState_affine (source : Nat) :
    StateAffine
      (entryState (logicalWidth := logicalWidth) (publicFits := publicFits)
        source) := by
  have child :=
    NightstreamFPrime.Layout.PiRLC.v1_2.SamplerChain.childInputs
      (chainInterface (logicalWidth := logicalWidth) (publicFits := publicFits))
      NightstreamFPrime.Layout.Stage1.PiRLCStarts.samplerLogicalStart
      (NightstreamFPrime.Layout.Stage1.PiRLCInputs.samplerInputs
        (logicalWidth := logicalWidth) (publicFits := publicFits))
      source (sourceLogicalStart source)
  simpa [entryState, sourceInterface, sourceLogicalStart] using!
    child.initialState

theorem entryTrace_state_matches (source : Nat) :
    (entryTrace (logicalWidth := logicalWidth) (publicFits := publicFits)
      source).state =
      TranscriptAbsorption.output
        ((sourceInterface (logicalWidth := logicalWidth)
            (publicFits := publicFits) source))
        source (sourceLogicalStart source) := by
  unfold entryTrace TranscriptAbsorption.output
    TranscriptAbsorption.ownedInterface Formal.Owned.output
    Formal.Owned.program
  rw [fastEntryState_eq_entryState]
  exact compileActions_state_eq phase
    (NightstreamFPrime.Layout.Stage1.PiRLCStarts.entryRowStart source)
    (sourceLogicalStart source)
    (entryState (logicalWidth := logicalWidth) (publicFits := publicFits)
      source)
    (TranscriptAbsorption.actions source)

/-- Held entry invocations imply the exact verifier-owned scalar-domain
entry relation for one production source. -/
theorem entryTrace_implies_spec (source : Nat) (env : Env)
    (holds : ∀ current ∈
      (entryTrace (logicalWidth := logicalWidth) (publicFits := publicFits)
        source).invocations,
      PermutationInvocationHolds (PilotData.circuitPackage ()) current env) :
    TranscriptAbsorption.SpecHolds
      ((sourceInterface (logicalWidth := logicalWidth)
          (publicFits := publicFits) source))
      source (sourceLogicalStart source)
      (NightstreamFPrime.Layout.Stage1.Spartan.pullback env) := by
  simp only [entryTrace, fastEntryState_eq_entryState] at holds
  have witnessLocal :
      NightstreamFPrime.Layout.Stage1.Spartan.piCcsPhaseOffset ≤
        sourceLogicalStart source := by
    unfold sourceLogicalStart
      NightstreamFPrime.Layout.Stage1.PiRLCStarts.samplerSourceLogicalStart
      NightstreamFPrime.Layout.Stage1.PiRLCStarts.samplerLogicalStart
      SamplerChain.sourceOffset
      NightstreamFPrime.Lifecycle.PiRLC.v1_2.Formal.samplerOffset
      NightstreamFPrime.Layout.Stage1.PiRLCStarts.phaseLogicalStart
      NightstreamFPrime.Layout.Stage1.PiRLCInputs.phaseOffset
    norm_num [NightstreamFPrime.Layout.Stage1.Spartan.piCcsPhaseOffset]
    omega
  have trace := compileActions_traceHolds phase
    (NightstreamFPrime.Layout.Stage1.PiRLCStarts.entryRowStart source)
    (sourceLogicalStart source)
    (entryState (logicalWidth := logicalWidth) (publicFits := publicFits)
      source)
    (TranscriptAbsorption.actions source) env witnessLocal
    (entryState_affine (logicalWidth := logicalWidth)
      (publicFits := publicFits) source)
    (NightstreamFPrime.Layout.PiRLC.v1_2.Leaves.TranscriptAbsorption.actions_affine
      source)
    (expectedSamples_eq_samples_of_assertionCount_zero
      (sourceLogicalStart source)
      (entryState (logicalWidth := logicalWidth) (publicFits := publicFits)
        source)
      (TranscriptAbsorption.actions source) rfl)
    holds
  have stateMatches := entryTrace_state_matches (logicalWidth := logicalWidth)
    (publicFits := publicFits) source
  unfold entryTrace at stateMatches
  rw [fastEntryState_eq_entryState] at stateMatches
  rw [stateMatches] at trace
  apply (TranscriptAbsorption.ownedSpec_iff_specHolds
    ((sourceInterface (logicalWidth := logicalWidth)
        (publicFits := publicFits) source))
    source (sourceLogicalStart source)
    (NightstreamFPrime.Layout.Stage1.Spartan.pullback env)).mp
  exact trace

theorem advanceState_affine (source : Nat) :
    StateAffine (advanceState (logicalWidth := logicalWidth) (publicFits := publicFits) source) :=
  NightstreamFPrime.Layout.PiRLC.v1_2.Sampler.entered_affine _ _ _

/-- The held advance invocation is the permutation selected by the scalar circuit. -/
theorem advanceInvocation_implies_spec (source : Nat) (env : Env)
    (holds : PermutationInvocationHolds (PilotData.circuitPackage ())
      (advanceInvocation (logicalWidth := logicalWidth) (publicFits := publicFits) source) env) :
    Permutation.Owned.SpecHolds
      (Sampler.advanceInterface (sourceInterface (logicalWidth := logicalWidth)
        (publicFits := publicFits) source) source (sourceLogicalStart source))
      (NightstreamFPrime.Layout.Stage1.PiRLCStarts.advanceLogicalStart source)
      (NightstreamFPrime.Layout.Stage1.Spartan.pullback env) := by
  simp only [advanceInvocation, fastAdvanceState_eq] at holds
  have witnessLocal : NightstreamFPrime.Layout.Stage1.Spartan.piCcsPhaseOffset ≤
      NightstreamFPrime.Layout.Stage1.PiRLCStarts.advanceLogicalStart source := by
    unfold NightstreamFPrime.Layout.Stage1.PiRLCStarts.advanceLogicalStart
      NightstreamFPrime.Layout.Stage1.PiRLCStarts.samplerSourceLogicalStart
      NightstreamFPrime.Layout.Stage1.PiRLCStarts.samplerLogicalStart
      NightstreamFPrime.Lifecycle.PiRLC.v1_2.Formal.samplerOffset
      NightstreamFPrime.Layout.Stage1.PiRLCStarts.phaseLogicalStart
      NightstreamFPrime.Layout.Stage1.PiRLCInputs.phaseOffset
      Sampler.advanceOffset Sampler.rangeOffset SamplerChain.sourceOffset
    norm_num [NightstreamFPrime.Layout.Stage1.Spartan.piCcsPhaseOffset]
    omega
  have transition := invocation_sound phase
    (NightstreamFPrime.Layout.Stage1.PiRLCStarts.advanceRowStart source)
    (NightstreamFPrime.Layout.Stage1.PiRLCStarts.advanceLogicalStart source)
    (advanceState (logicalWidth := logicalWidth) (publicFits := publicFits) source)
    env witnessLocal (advanceState_affine (logicalWidth := logicalWidth) (publicFits := publicFits) source) holds
  unfold Permutation.Owned.SpecHolds
  calc
    _ = List.ofFn (Layer.evalState (NightstreamFPrime.Layout.Stage1.Spartan.pullback env)
        (permutationOutput (NightstreamFPrime.Layout.Stage1.PiRLCStarts.advanceLogicalStart source))) := rfl
    _ = List.ofFn (Permutation.runF Permutation.schedule
        (Layer.evalState (NightstreamFPrime.Layout.Stage1.Spartan.pullback env)
          (advanceState (logicalWidth := logicalWidth) (publicFits := publicFits) source))) :=
      congrArg List.ofFn transition
    _ = Permutation.runReference Permutation.schedule
        (List.ofFn (Layer.evalState (NightstreamFPrime.Layout.Stage1.Spartan.pullback env)
          (advanceState (logicalWidth := logicalWidth) (publicFits := publicFits) source))) :=
      Permutation.runF_eq_reference _ _
    _ = Spec.Poseidon2.permute
        (List.ofFn (Layer.evalState (NightstreamFPrime.Layout.Stage1.Spartan.pullback env)
          (advanceState (logicalWidth := logicalWidth) (publicFits := publicFits) source))) :=
      Permutation.runReference_schedule _
    _ = _ := rfl

end NightstreamFPrime.Export.Stage1.PiRLCSamplerInvocations
