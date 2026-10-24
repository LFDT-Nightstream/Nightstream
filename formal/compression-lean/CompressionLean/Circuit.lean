import CompressionLean.Circuit.Algebra
import CompressionLean.Circuit.Block
import CompressionLean.Circuit.Early
import CompressionLean.Circuit.Fold
import CompressionLean.Circuit.Gkr
import CompressionLean.Circuit.Hash
import CompressionLean.Circuit.Honest
import CompressionLean.Circuit.Ir
import CompressionLean.Circuit.Late
import CompressionLean.Circuit.Linear
import CompressionLean.Circuit.Mle
import CompressionLean.Circuit.Openings
import CompressionLean.Circuit.Package
import CompressionLean.Circuit.Record
import CompressionLean.Circuit.Ring
import CompressionLean.Circuit.Sumcheck
import CompressionLean.Circuit.Terminal
import CompressionLean.Circuit.Verifier
import CompressionLean.Circuit.Views
import CompressionLean.Circuit.Whir

/-!
`Circuit/` owns the programs that transcribe Rust: the instruction set and
the builder (`Ir`), the recorder that lowers a program to a shrink run
(`Record`), the Poseidon2 block template (`Block`), the layer-1 verifier
written with the builder (`Algebra` to `Verifier`), the terminal program
(`Terminal`) and the terminal package as `Package::apply` reads it
(`Package`).

It owns no proof about these programs: the lowering proofs are `Lowering/`
and the simulation proofs are `Sim/`.
-/
