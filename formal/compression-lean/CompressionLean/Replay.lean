import CompressionLean.Replay.Data
import CompressionLean.Replay.Field
import CompressionLean.Replay.Formulas
import CompressionLean.Replay.Gkr
import CompressionLean.Replay.Stages
import CompressionLean.Replay.Transcript
import CompressionLean.Replay.Whir

/-!
`Replay/` owns the replay of the layer-1 verifier
(`crates/neo-spartan/src/verifier.rs`) in Lean, on Goldilocks words: the
relation data, the words and their extension, the transcript, and the stages.
It reads only F′. `replayLayer1` runs it on Rust's proofs, the simulation
(`Sim/`) proves that the circuit computes it, and the bridge (`Bridge/`)
turns an accepted replay into a layer-1 path.

It owns no soundness proof.
-/
