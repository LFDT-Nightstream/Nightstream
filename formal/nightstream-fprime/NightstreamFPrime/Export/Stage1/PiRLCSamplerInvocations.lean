import NightstreamFPrime.Export.Stage1.Invocations
import NightstreamFPrime.Export.Stage1.PiRLCSamplerRows

/-!
Owns the Poseidon2 invocation schedule for the production PiRLC sampler
chain.

Each of the 17 scalar samplers contains one domain-entry absorption and eight
raw digest-window permutations. The schedule uses the same canonical 592-row
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
open NightstreamFPrime.Lifecycle.PiRLC.v1_1
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint

variable {logicalWidth : Nat}
  {publicFits : ringDegree * publicRingColumns ≤
    Phi81CarrierLayout.carrierWidth logicalWidth}

def phase : Nat := 7
def sourceCount : Nat := 17
def digestRoundCount : Nat := 8

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
  (Sampler.entryInterface
    (sourceInterface (logicalWidth := logicalWidth)
      (publicFits := publicFits) source)).initialState
    (sourceLogicalStart source)

def entryTrace (source : Nat) : Trace :=
  compileActions phase
    (NightstreamFPrime.Layout.Stage1.PiRLCStarts.entryRowStart source)
    (sourceLogicalStart source)
    (entryState (logicalWidth := logicalWidth) (publicFits := publicFits)
      source)
    (TranscriptAbsorption.actions source)

def entryInvocations (source : Nat) : List PermutationInvocation :=
  (entryTrace (logicalWidth := logicalWidth) (publicFits := publicFits)
    source).invocations

def windowState (source round : Nat) : Invocations.EState :=
  (Sampler.windowInterface
    (sourceInterface (logicalWidth := logicalWidth)
      (publicFits := publicFits) source)
    source (sourceLogicalStart source) round).initialState
      (NightstreamFPrime.Layout.Stage1.PiRLCStarts.windowLogicalStart
        source round)

def windowInvocation (source round : Nat) : PermutationInvocation :=
  invocation phase
    (NightstreamFPrime.Layout.Stage1.PiRLCStarts.digestPermutationRowStart
      source round)
    (NightstreamFPrime.Layout.Stage1.PiRLCStarts.digestPermutationLogicalStart
      source round)
    (windowState (logicalWidth := logicalWidth) (publicFits := publicFits)
      source round)

def windowInvocations (source : Nat) : List PermutationInvocation :=
  (List.range digestRoundCount).map
    (windowInvocation (logicalWidth := logicalWidth) (publicFits := publicFits)
      source)

def sourceInvocations (source : Nat) : List PermutationInvocation :=
  entryInvocations (logicalWidth := logicalWidth) (publicFits := publicFits)
      source ++
    windowInvocations (logicalWidth := logicalWidth) (publicFits := publicFits)
      source

def invocations : List PermutationInvocation :=
  (List.range sourceCount).flatMap
    (sourceInvocations (logicalWidth := logicalWidth) (publicFits := publicFits))

theorem entryState_affine (source : Nat) :
    StateAffine
      (entryState (logicalWidth := logicalWidth) (publicFits := publicFits)
        source) := by
  have child :=
    NightstreamFPrime.Layout.PiRLC.v1_1.SamplerChain.childInputs
      (chainInterface (logicalWidth := logicalWidth) (publicFits := publicFits))
      NightstreamFPrime.Layout.Stage1.PiRLCStarts.samplerLogicalStart
      (NightstreamFPrime.Layout.Stage1.PiRLCInputs.samplerInputs
        (logicalWidth := logicalWidth) (publicFits := publicFits))
      source (sourceLogicalStart source)
  simpa [entryState, sourceInterface, sourceLogicalStart] using
    child.initialState

theorem entryTrace_state_matches (source : Nat) :
    (entryTrace (logicalWidth := logicalWidth) (publicFits := publicFits)
      source).state =
      TranscriptAbsorption.output
        (Sampler.entryInterface
          (sourceInterface (logicalWidth := logicalWidth)
            (publicFits := publicFits) source))
        source (sourceLogicalStart source) := by
  unfold entryTrace TranscriptAbsorption.output
    TranscriptAbsorption.ownedInterface Formal.Owned.output
    Formal.Owned.program
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
      (Sampler.entryInterface
        (sourceInterface (logicalWidth := logicalWidth)
          (publicFits := publicFits) source))
      source (sourceLogicalStart source)
      (NightstreamFPrime.Layout.Stage1.Spartan.pullback env) := by
  have witnessLocal :
      NightstreamFPrime.Layout.Stage1.Spartan.piCcsPhaseOffset ≤
        sourceLogicalStart source := by
    unfold sourceLogicalStart
      NightstreamFPrime.Layout.Stage1.PiRLCStarts.samplerSourceLogicalStart
      NightstreamFPrime.Layout.Stage1.PiRLCStarts.samplerLogicalStart
      NightstreamFPrime.Lifecycle.PiRLC.v1_1.Formal.samplerOffset
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
    (NightstreamFPrime.Layout.PiRLC.v1_1.Leaves.TranscriptAbsorption.actions_affine
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
  rw [stateMatches] at trace
  apply (TranscriptAbsorption.ownedSpec_iff_specHolds
    (Sampler.entryInterface
      (sourceInterface (logicalWidth := logicalWidth)
        (publicFits := publicFits) source))
    source (sourceLogicalStart source)
    (NightstreamFPrime.Layout.Stage1.Spartan.pullback env)).mp
  exact trace

theorem windowState_affine (source round : Nat) :
    StateAffine
      (windowState (logicalWidth := logicalWidth) (publicFits := publicFits)
        source round) := by
  have inputs := NightstreamFPrime.Layout.PiRLC.v1_1.Sampler.windowInputs
    (sourceInterface (logicalWidth := logicalWidth)
      (publicFits := publicFits) source)
    source (sourceLogicalStart source) round
  simpa [windowState, sourceLogicalStart,
    NightstreamFPrime.Layout.Stage1.PiRLCStarts.windowLogicalStart,
    NightstreamFPrime.Layout.Stage1.PiRLCStarts.samplerSourceLogicalStart,
    Sampler.windowOffset, Sampler.windowBase, Sampler.entryPrivateCount] using
      inputs.initialState

/-- One held raw window invocation is exactly the permutation child selected
by the production digest-window interface. -/
theorem windowInvocation_implies_spec (source round : Nat) (env : Env)
    (holds : PermutationInvocationHolds (PilotData.circuitPackage ())
      (windowInvocation (logicalWidth := logicalWidth)
        (publicFits := publicFits) source round) env) :
    Permutation.Owned.SpecHolds
      (DigestWindow.permutationInterface
        (Sampler.windowInterface
          (sourceInterface (logicalWidth := logicalWidth)
            (publicFits := publicFits) source)
          source (sourceLogicalStart source) round)
        (NightstreamFPrime.Layout.Stage1.PiRLCStarts.windowLogicalStart
          source round))
      (NightstreamFPrime.Layout.Stage1.PiRLCStarts.digestPermutationLogicalStart
        source round)
      (NightstreamFPrime.Layout.Stage1.Spartan.pullback env) := by
  have witnessLocal :
      NightstreamFPrime.Layout.Stage1.Spartan.piCcsPhaseOffset ≤
        NightstreamFPrime.Layout.Stage1.PiRLCStarts.digestPermutationLogicalStart
          source round := by
    unfold NightstreamFPrime.Layout.Stage1.PiRLCStarts.digestPermutationLogicalStart
      NightstreamFPrime.Layout.Stage1.PiRLCStarts.windowLogicalStart
      NightstreamFPrime.Layout.Stage1.PiRLCStarts.samplerSourceLogicalStart
      NightstreamFPrime.Layout.Stage1.PiRLCStarts.samplerLogicalStart
      NightstreamFPrime.Lifecycle.PiRLC.v1_1.Formal.samplerOffset
      NightstreamFPrime.Layout.Stage1.PiRLCStarts.phaseLogicalStart
      NightstreamFPrime.Layout.Stage1.PiRLCInputs.phaseOffset
    norm_num [NightstreamFPrime.Layout.Stage1.Spartan.piCcsPhaseOffset]
    omega
  have transition := invocation_sound phase
    (NightstreamFPrime.Layout.Stage1.PiRLCStarts.digestPermutationRowStart
      source round)
    (NightstreamFPrime.Layout.Stage1.PiRLCStarts.digestPermutationLogicalStart
      source round)
    (windowState (logicalWidth := logicalWidth) (publicFits := publicFits)
      source round)
    env witnessLocal
    (windowState_affine (logicalWidth := logicalWidth)
      (publicFits := publicFits) source round)
    holds
  unfold Permutation.Owned.SpecHolds
  calc
    List.ofFn (Layer.evalState
        (NightstreamFPrime.Layout.Stage1.Spartan.pullback env)
        (Permutation.Owned.output
          (DigestWindow.permutationInterface
            (Sampler.windowInterface
              (sourceInterface (logicalWidth := logicalWidth)
                (publicFits := publicFits) source)
              source (sourceLogicalStart source) round)
            (NightstreamFPrime.Layout.Stage1.PiRLCStarts.windowLogicalStart
              source round))
          (NightstreamFPrime.Layout.Stage1.PiRLCStarts.digestPermutationLogicalStart
            source round))) =
        List.ofFn (Layer.evalState
          (NightstreamFPrime.Layout.Stage1.Spartan.pullback env)
          (permutationOutput
            (NightstreamFPrime.Layout.Stage1.PiRLCStarts.digestPermutationLogicalStart
              source round))) := by rfl
    _ = List.ofFn (Permutation.runF Permutation.schedule
          (Layer.evalState
            (NightstreamFPrime.Layout.Stage1.Spartan.pullback env)
            (windowState (logicalWidth := logicalWidth)
              (publicFits := publicFits) source round))) :=
      congrArg List.ofFn transition
    _ = Permutation.runReference Permutation.schedule
          (List.ofFn (Layer.evalState
            (NightstreamFPrime.Layout.Stage1.Spartan.pullback env)
            (windowState (logicalWidth := logicalWidth)
              (publicFits := publicFits) source round))) :=
      Permutation.runF_eq_reference _ _
    _ = Spec.Poseidon2.permute
          (List.ofFn (Layer.evalState
            (NightstreamFPrime.Layout.Stage1.Spartan.pullback env)
            (windowState (logicalWidth := logicalWidth)
              (publicFits := publicFits) source round))) :=
      Permutation.runReference_schedule _
    _ = Spec.Poseidon2.permute
          (List.ofFn (Layer.evalState
            (NightstreamFPrime.Layout.Stage1.Spartan.pullback env)
            ((DigestWindow.permutationInterface
              (Sampler.windowInterface
                (sourceInterface (logicalWidth := logicalWidth)
                  (publicFits := publicFits) source)
                source (sourceLogicalStart source) round)
              (NightstreamFPrime.Layout.Stage1.PiRLCStarts.windowLogicalStart
                source round)).initialState
              (NightstreamFPrime.Layout.Stage1.PiRLCStarts.digestPermutationLogicalStart
                source round)))) := by rfl

end NightstreamFPrime.Export.Stage1.PiRLCSamplerInvocations
