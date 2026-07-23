/-
Generated file: production combined-NC artifact; do not hand-edit.

Owns: the ordered facade over every bounded terminal raw-old-block row and assignment chunk.

Does not own: decoding, row satisfaction, transcript authority, commitment
binding, semantic acceptance, costs, or permission to remove rows.

Emits constraints: no.

| Stable stage path | Obligation | Authority class |
|---|---|---|
| `f_prime.pi_ccs_nc.delayed.combined.generated` | The generated payload named by `Owns` above | computed artifact |
-/

import Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated.Execution.TerminalProjectionFixture.Metadata
import Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated.Execution.TerminalProjectionFixture.Rows.Chunk0
import Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated.Execution.TerminalProjectionFixture.Rows.Chunk1
import Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated.Execution.TerminalProjectionFixture.Rows.Chunk2
import Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated.Execution.TerminalProjectionFixture.Assignment.Chunk0
import Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated.Execution.TerminalProjectionFixture.Assignment.Chunk1
import Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated.Execution.TerminalProjectionFixture.Assignment.Chunk2
import Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated.Execution.TerminalProjectionFixture.Assignment.Chunk3
import Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated.Execution.TerminalProjectionFixture.Assignment.Chunk4
import Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated.Execution.TerminalProjectionFixture.Assignment.Chunk5
import Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated.Execution.TerminalProjectionFixture.Assignment.Chunk6
import Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated.Execution.TerminalProjectionFixture.Assignment.Chunk7
import Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated.Execution.TerminalProjectionFixture.Assignment.Chunk8
import Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated.Execution.TerminalProjectionFixture.Assignment.Chunk9

namespace Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated.Execution.TerminalProjectionFixture

def rows : List Nightstream.Implementation.R1CS.Row := Rows.Chunk0.values ++ Rows.Chunk1.values ++ Rows.Chunk2.values
def assignmentValues : List Nat := Assignment.Chunk0.values ++ Assignment.Chunk1.values ++ Assignment.Chunk2.values ++ Assignment.Chunk3.values ++ Assignment.Chunk4.values ++ Assignment.Chunk5.values ++ Assignment.Chunk6.values ++ Assignment.Chunk7.values ++ Assignment.Chunk8.values ++ Assignment.Chunk9.values
def artifactRow (index : Fin rows.length) : Nightstream.Implementation.R1CS.Row :=
rows.get index
def assignment (column : Nat) : Nat := assignmentValues.getD column 0

end Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated.Execution.TerminalProjectionFixture
