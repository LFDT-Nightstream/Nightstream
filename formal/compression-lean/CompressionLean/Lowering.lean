import CompressionLean.Lowering.Exec
import CompressionLean.Lowering.Forms
import CompressionLean.Lowering.PackageRows
import CompressionLean.Lowering.RecordCounts
import CompressionLean.Lowering.Rows
import CompressionLean.Lowering.Sound
import CompressionLean.Lowering.Valid

/-!
`Lowering/` owns the proofs about the lowering of a program to a shrink run:
the executor reading of a program (`Exec`), what the rows of each step say
(`Forms`, `Rows`), that a run names only its own cells (`Valid`), that a CCS
witness gives an accepting run of the executor (`Sound`), that the counts do
not depend on the statement (`RecordCounts`), and that the rows of the
terminal package system hold exactly when F′'s rows hold (`PackageRows`).

It does not own the recorder (`Circuit/Record.lean`) or the meaning of a
program in replay words (`Sim/`).
-/
