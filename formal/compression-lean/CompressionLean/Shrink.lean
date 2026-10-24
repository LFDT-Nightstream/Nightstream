import CompressionLean.Shrink.BlockPart
import CompressionLean.Shrink.Budget
import CompressionLean.Shrink.Ccs
import CompressionLean.Shrink.Cost
import CompressionLean.Shrink.Diagonal
import CompressionLean.Shrink.Evaluate
import CompressionLean.Shrink.Game
import CompressionLean.Shrink.Layout
import CompressionLean.Shrink.Rounds
import CompressionLean.Shrink.Soundness

/-!
`Shrink/` owns the soundness of the shrink verifier
(`crates/neo-spartan/src/shrink/mod.rs`) as a game: the CCS of a run and its
layout (`Ccs`, `Layout`), the verifier's matrix value (`Evaluate`,
`BlockPart`, `Diagonal`), the sum-check tables and the batching (`Rounds`),
the game and its accepting paths (`Game`, `Soundness`), and its cost and
budget (`Cost`, `Budget`).

It imports no layer-1 module. It does not own the lowering of a program to
a run (`Circuit/Record.lean`, `Lowering/`).
-/
