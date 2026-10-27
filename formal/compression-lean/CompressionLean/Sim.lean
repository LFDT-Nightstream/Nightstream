import CompressionLean.Sim.Algebra
import CompressionLean.Sim.Calculus
import CompressionLean.Sim.Claim
import CompressionLean.Sim.Early
import CompressionLean.Sim.Encodes
import CompressionLean.Sim.EvalK
import CompressionLean.Sim.Fold
import CompressionLean.Sim.Gkr
import CompressionLean.Sim.Hash
import CompressionLean.Sim.Honest
import CompressionLean.Sim.Instr
import CompressionLean.Sim.Late
import CompressionLean.Sim.Linear
import CompressionLean.Sim.Merkle
import CompressionLean.Sim.Ring
import CompressionLean.Sim.Run
import CompressionLean.Sim.SetupLeaf
import CompressionLean.Sim.Sumcheck
import CompressionLean.Sim.Terminal
import CompressionLean.Sim.Tracks
import CompressionLean.Sim.Verify
import CompressionLean.Sim.Views
import CompressionLean.Sim.Whir

/-!
`Sim/` owns the simulation of the circuit's programs: the calculus
(`Calculus`: the instruction shape `Appends` and what a program `Ensures`;
`Instr`: what each instruction ensures; `Tracks`: a transcript program beside
the replay), one file per part of the verifier and of the terminal program,
and the encoding premise for the production final program (`Encodes`,
`programEncodes`).

It does not own the programs (`Ports/Circuit/`) or the lowering proofs
(`Lowering/`).
-/
