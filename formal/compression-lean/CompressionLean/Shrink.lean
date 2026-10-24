import CompressionLean.Shrink.BlockPart
import CompressionLean.Shrink.Budget
import CompressionLean.Shrink.Ccs
import CompressionLean.Shrink.Cost
import CompressionLean.Shrink.Diagonal
import CompressionLean.Shrink.Evaluate
import CompressionLean.Shrink.Layout
import CompressionLean.Shrink.Rounds
import CompressionLean.Shrink.Soundness

/-!
`Shrink/` owns the soundness of the shrink verifier
(`crates/neo-spartan/src/shrink/mod.rs`) as a game, over its port in
`Ports/Shrink/`, file by file: the relation and the fit of a run's CCS
(`Ccs`, `Layout`), the value of the verifier's matrix value (`Evaluate`,
`BlockPart`, `Diagonal`), the true sums of the sum-checks (`Rounds`), the
game's accepting paths (`Soundness`), and its cost and budget (`Cost`,
`Budget`).

It imports no layer-1 module. It does not own the shrink verifier itself
(`Ports/Shrink/`) or the lowering of a program to a run
(`Ports/Circuit/Record.lean`, `Lowering/`).
-/
