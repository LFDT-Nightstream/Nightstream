import Nightstream.Implementation.R1CS.Correspondence.PiCcsTranscript.Refinement.RawAbsorption
import Nightstream.Implementation.R1CS.Correspondence.PiCcsTranscript.Refinement.Terminal

/-!
Artifact-refinement tree for the production-shaped `Pi_CCS` transcript.

Owns: only the parent boundary between reusable raw sponge correspondence and
the fixed terminal F-prime artifact.

Does not own: transcript semantics, paper soundness, native or Rust
conformance, cost accounting, necessity, or row removal.

Emits constraints: no.

Authority boundary: accepted rows are evidence only after a child theorem
connects them to independently computed transcript state.

| Child path | Mathematical obligation | Artifact scope |
|---|---|---|
| `RawAbsorption` | lazy constant absorption and eager semantic absorption are observationally equal | reusable transcript gadgets |
| `Terminal` | the fixed terminal artifact refines named binding, challenge, FE/NC, and catch-up phases | terminal F-prime owner |
-/
