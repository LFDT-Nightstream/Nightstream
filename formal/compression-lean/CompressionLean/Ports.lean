import CompressionLean.Ports.Circuit.Algebra
import CompressionLean.Ports.Circuit.Block
import CompressionLean.Ports.Circuit.Early
import CompressionLean.Ports.Circuit.Fold
import CompressionLean.Ports.Circuit.Gkr
import CompressionLean.Ports.Circuit.Hash
import CompressionLean.Ports.Circuit.Honest
import CompressionLean.Ports.Circuit.Ir
import CompressionLean.Ports.Circuit.Late
import CompressionLean.Ports.Circuit.Linear
import CompressionLean.Ports.Circuit.Mle
import CompressionLean.Ports.Circuit.Openings
import CompressionLean.Ports.Circuit.Record
import CompressionLean.Ports.Circuit.Ring
import CompressionLean.Ports.Circuit.Sumcheck
import CompressionLean.Ports.Circuit.Verifier
import CompressionLean.Ports.Circuit.Views
import CompressionLean.Ports.Circuit.Whir
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
import CompressionLean.Ports.Setup
import CompressionLean.Ports.Shrink.Rounds
import CompressionLean.Ports.Terminal.Package
import CompressionLean.Ports.Terminal.Program

/-!
`Ports/` owns the hand transcriptions of Rust, and nothing else: definitions,
each with a `ports.json` entry or a reason in `CompressionLeanTests/Ports.lean`.
A port's namespace is its path without `Ports/`. The proofs about the ports
are in the folders above it. The folders follow the Rust modules
(`crates/neo-spartan/src/` unless named):

- `Circuit/`: the builder and its recorder (`Ir`, `Record`:
  `circuit/record.rs`), the Poseidon2 block (`Block`: `circuit/poseidon2.rs`),
  the gadgets (`Algebra`, `Hash`, `Fold`: `circuit/algebra.rs`, `hash.rs`,
  `fold.rs`), and the layer-1 verifier as builder programs (`Sumcheck`, `Gkr`,
  `Mle`, `Ring`, `Whir`: `sumcheck.rs`, `gkr.rs`, `mle.rs`, `ring.rs`,
  `whir/mod.rs`; `Views`, `Early`, `Linear`, `Honest`, `Late`, `Openings`,
  `Verifier`: the stages of `verifier.rs`, with `matrix.rs` and `lib.rs`).
- `Shrink/`: the shrink verifier (`shrink/mod.rs`, `layout.rs`, `evaluate.rs`).
- `Replay/`: the replay of the layer-1 verifier on words (`verifier.rs` and
  the modules that it calls).
- `Terminal/`: the terminal package as `Package::apply` reads it (`Package`:
  `crates/nightstream/src/lifecycle/terminal/package.rs`) and the final
  program (`Program`: `terminal/mod.rs`).
- `Setup.lean`: Rust's layer-1 setup build (`lib.rs`, `matrix.rs`).
- `Layer1/Budget.lean`: the degrees of `outer_terms` (`lib.rs`).
-/
