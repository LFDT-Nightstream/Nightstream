import CompressionLean.Ports.Layer1.Budget
import CompressionLean.Ports.Replay.Data
import CompressionLean.Ports.Replay.Field
import CompressionLean.Ports.Replay.Formulas
import CompressionLean.Ports.Replay.Gkr
import CompressionLean.Ports.Replay.Stages
import CompressionLean.Ports.Replay.Transcript
import CompressionLean.Ports.Replay.Whir
import CompressionLean.Ports.Shrink.BlockPart
import CompressionLean.Ports.Shrink.Budget
import CompressionLean.Ports.Shrink.Ccs
import CompressionLean.Ports.Shrink.Diagonal
import CompressionLean.Ports.Shrink.Evaluate
import CompressionLean.Ports.Shrink.Game
import CompressionLean.Ports.Shrink.Layout
import CompressionLean.Ports.Shrink.Rounds

/-!
`Ports/` owns the hand transcriptions of Rust, and nothing else: definitions,
each with a `ports.json` entry or a reason in `CompressionLeanTests/Ports.lean`.
A port's namespace is its path without `Ports/`. The proofs about the ports
are in the folders above.
-/
