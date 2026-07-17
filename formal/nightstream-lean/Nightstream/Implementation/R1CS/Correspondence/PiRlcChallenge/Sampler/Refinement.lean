import Nightstream.Implementation.R1CS.Correspondence.PiRlcChallenge.Sampler.Refinement.LaneRows
import Nightstream.Implementation.R1CS.Correspondence.PiRlcChallenge.Sampler.Refinement.ScalarLanes
import Nightstream.Implementation.R1CS.Correspondence.PiRlcChallenge.Sampler.Refinement.TailCandidateSemantics
import Nightstream.Implementation.R1CS.Correspondence.PiRlcChallenge.Sampler.Refinement.TailPrefixCounts
import Nightstream.Implementation.R1CS.Correspondence.PiRlcChallenge.Sampler.Refinement.TailFirstAccepted
import Nightstream.Implementation.R1CS.Correspondence.PiRlcChallenge.Sampler.Refinement.OneScalarRows
import Nightstream.Implementation.R1CS.Correspondence.PiRlcChallenge.Sampler.Refinement.OneScalar
import Nightstream.Implementation.R1CS.Correspondence.PiRlcChallenge.Sampler.Refinement.TailRows
import Nightstream.Implementation.R1CS.Correspondence.PiRlcChallenge.Sampler.Refinement.TailInputs
import Nightstream.Implementation.R1CS.Correspondence.PiRlcChallenge.Sampler.Refinement.PrefixCounts
import Nightstream.Implementation.R1CS.Correspondence.PiRlcChallenge.Sampler.Refinement.RingEncoding
import Nightstream.Implementation.R1CS.Correspondence.PiRlcChallenge.Sampler.Refinement.Recursive.RingAssembly
import Nightstream.Implementation.R1CS.Correspondence.PiRlcChallenge.Sampler.Refinement.Terminal

/-! Parent for sampler row-to-semantics refinement.

Owns: lane, scalar, tail, prefix, selection-output, and ring-encoding
refinement for recursive and terminal profiles.

Does not own: the base chunk equations, candidate-selection rows, transcript
authority, or complete NIFS soundness.

Emits constraints: no.

| Child family | Mathematical obligation | Excluded boundary |
|---|---|---|
| lane/scalar | decoded row lanes equal scalar candidates | transcript source |
| tail/prefix/first accepted | bounded rejection-control semantics | security probability |
| ring encoding/assembly | selected scalars assemble into the ring carrier | Pi_RLC algebra verification |
| terminal | fixed terminal-profile refinement and handoff | recursive profile |
-/
