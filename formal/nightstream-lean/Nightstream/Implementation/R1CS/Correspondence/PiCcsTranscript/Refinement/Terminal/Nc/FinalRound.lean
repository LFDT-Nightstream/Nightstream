import Nightstream.Implementation.R1CS.Correspondence.PiCcsTranscript.Refinement.Terminal.Nc.FinalRound.Artifact

/-!
Selected physical leaf for the final terminal-NC round.

Owns: only the explicit round-fourteen artifact view used by the post-NC
boundary.

Does not own: typed message selection, transcript execution, complete replay,
final SumCheck algebra, Rust conformance, costs, necessity, or row removal.

Emits constraints: no.

Authority boundary: this module exposes accepted physical rows only.
`LaterRound` owns the indexed semantic execution, while `Replay` proves that
the exact typed schedule reaches this selected artifact.

| Family path | Mathematical obligation | Child owner |
|---|---|---|
| `nifs.pi_ccs.nc_sumcheck.round.14.artifact` | exact length, marker, and three Poseidon2 call owners | `Artifact` |
-/
