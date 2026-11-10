import NightstreamFPrime.Lifecycle.BoundedDepth.Keys
import NightstreamFPrime.Lifecycle.PiRLC.v1_2.SamplerChain

/-! The light Π_RLC challenge: one production sampler at coordinate 0 (`lightChallenge`). The
`Sampler` gadget has 2,927 rows (`Sampler.counts`); today's step samples 17 challenges. -/

namespace NightstreamFPrime.Lifecycle.BoundedDepth.LightSampler

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Lifecycle.PiRLC.v1_2

/-- The sampler rows give the light key's challenge for the transcript state they start from. -/
theorem challenge_eq (interface : Sampler.Interface) (env : Env) (offset : Nat)
    (spec : Sampler.SpecHolds interface 0 offset env) :
    (fun position => (Sampler.outputChallenge offset position).eval env) =
      lightChallenge (Sampler.evalState env (interface.initialState offset)) := by
  funext position
  change (Sampler.outputWord offset position - 2).eval env = _
  rw [Expr.eval_sub, spec.digits position]
  simp only [lightChallenge, Transcript.PiRlcSampler.piRlcChallenges,
    Transcript.PiRlcSampler.piRlcChallengesWithState_challenges,
    Transcript.PiRlcSampler.sampleRingChallenge]
  exact SamplerChain.centered_digit _

theorem challenge_valid (interface : Sampler.Interface) (env : Env) (offset : Nat)
    (spec : Sampler.SpecHolds interface 0 offset env) :
    Phi81Relation.PiRLCAlgebra.Challenge.challengeValid
      (fun position => (Sampler.outputChallenge offset position).eval env) := by
  rw [challenge_eq interface env offset spec]
  exact lightChallenge_valid _

end NightstreamFPrime.Lifecycle.BoundedDepth.LightSampler
