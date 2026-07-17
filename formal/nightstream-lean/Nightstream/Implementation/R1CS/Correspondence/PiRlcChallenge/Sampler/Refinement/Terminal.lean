import Nightstream.Implementation.R1CS.Correspondence.PiRlcChallenge.Sampler.Refinement.Terminal.ScalarRows
import Nightstream.Implementation.R1CS.Correspondence.PiRlcChallenge.Sampler.Refinement.Terminal.ScalarSemantics
import Nightstream.Implementation.R1CS.Correspondence.PiRlcChallenge.Sampler.Refinement.Terminal.TailRows
import Nightstream.Implementation.R1CS.Correspondence.PiRlcChallenge.Sampler.Refinement.Terminal.TailSources
import Nightstream.Implementation.R1CS.Correspondence.PiRlcChallenge.Sampler.Refinement.Terminal.FirstAccepted
import Nightstream.Implementation.R1CS.Correspondence.PiRlcChallenge.Sampler.Refinement.Terminal.MachineOutput
import Nightstream.Implementation.R1CS.Correspondence.PiRlcChallenge.Sampler.Refinement.Terminal.RingAssembly
import Nightstream.Implementation.R1CS.Correspondence.PiRlcChallenge.Sampler.Refinement.Terminal.Batch
import Nightstream.Implementation.R1CS.Correspondence.PiRlcChallenge.Sampler.Refinement.Terminal.Certificate
import Nightstream.Implementation.R1CS.Correspondence.PiRlcChallenge.Sampler.Refinement.Terminal.SemanticHandoff

/-! Parent for terminal sampler refinement.

Owns: the terminal scalar/tail row image, first-accepted output, ring assembly,
batch certificate, and semantic handoff.

Does not own: recursive sampler placement, transcript generation, or the
independent paper-level sampling theorem.

Emits constraints: no.

| Child family | Mathematical obligation | Excluded boundary |
|---|---|---|
| scalar and tail | exact terminal row/source interpretation | recursive profile |
| first accepted and machine output | terminal control-flow result | Fiat-Shamir authority |
| ring, batch, certificate, handoff | output assembly and typed semantic boundary | complete NIFS composition |
-/
