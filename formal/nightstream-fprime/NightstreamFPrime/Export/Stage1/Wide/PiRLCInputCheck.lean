import NightstreamFPrime.Export.Stage1.PiRLCInputCheck
import NightstreamFPrime.Export.Stage1.Wide.BaseStepFixture
import NightstreamFPrime.Lifecycle.PiRLC.Wide.Key

/-! Selected C/R execution uses the exact total wide transcript batch.
The indexed commitment and evaluation traces reuse the materialized algebra. -/

namespace NightstreamFPrime.Export.Stage1.Wide.PiRLCInputCheck

open NightstreamFPrime.Spec NightstreamFPrime.Lifecycle
open PiRLCNonzero (SourceCount)

def sampler (state : Transcript.State) : Option (Transcript.PiRlcSampler.Batch SourceCount) :=
  some (BaseStepFixture.batch state)

theorem sampler_response (state : Transcript.State) :
    (sampler state).map Transcript.PiRlcSampler.Batch.challenges = PiRLC.Wide.Key.piRlcResponse state := by
  apply congrArg some
  funext source lane
  exact BaseStepFixture.batch_challenges state source lane

def sampled (input : Stage1.PiRLCInputCheck.Input) : Option (Transcript.PiRlcSampler.Batch SourceCount) :=
  let result := PiCCSInputCheck.execute input
  if result.accepted then sampler result.outgoing else none

theorem sampled_on_rejection (input : Stage1.PiRLCInputCheck.Input)
    (rejected : (PiCCSInputCheck.execute input).accepted = false) : sampled input = none := by
  simp only [sampled, rejected, Bool.false_eq_true, ↓reduceIte]

theorem sampled_response (input : Stage1.PiRLCInputCheck.Input)
    (batch : Transcript.PiRlcSampler.Batch SourceCount) (returned : sampled input = some batch) :
    (PiCCSInputCheck.execute input).accepted = true ∧
      PiRLC.Wide.Key.piRlcResponse (PiCCSInputCheck.execute input).outgoing = some batch.challenges := by
  dsimp only [sampled] at returned
  split at returned
  · rename_i accepted
    refine ⟨accepted, ?_⟩
    rw [← sampler_response]
    simp only [returned, Option.map_some]
  · cases returned

def checkIO (input : Stage1.PiRLCInputCheck.Input) (packageIdentity : VerifierContext.Digest4) :
    IO Stage1.PiRLCInputCheck.Execution :=
  Stage1.PiRLCInputCheck.checkIOWith sampler input packageIdentity

def checkValueIO (input : Stage1.PiRLCInputCheck.Input) (packageIdentity : VerifierContext.Digest4) : IO Codec.Value := do
  let result ← checkIO input packageIdentity
  return .array result.fields

end NightstreamFPrime.Export.Stage1.Wide.PiRLCInputCheck
