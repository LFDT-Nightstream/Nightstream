import CompressionLean.Sim.Algebra
import CompressionLean.Sim.Calculus
import CompressionLean.Sim.Claim
import CompressionLean.Sim.Early
import CompressionLean.Sim.Encodes
import CompressionLean.Sim.Fold
import CompressionLean.Sim.Gkr
import CompressionLean.Sim.Hash
import CompressionLean.Sim.Honest
import CompressionLean.Sim.Late
import CompressionLean.Sim.Linear
import CompressionLean.Sim.Merkle
import CompressionLean.Sim.Ring
import CompressionLean.Sim.Sumcheck
import CompressionLean.Sim.Terminal
import CompressionLean.Sim.Verify
import CompressionLean.Sim.Views
import CompressionLean.Sim.Whir

/-!
`Sim/` owns the simulation of the circuit's programs: the calculus that
reads a builder program as replay words (`Calculus`), one file per part of
the verifier and of the terminal program, and the encoding premise for the
production final program (`Encodes`, `programEncodes`).

It does not own the programs (`Ports/Circuit/`) or the lowering proofs
(`Lowering/`).
-/
