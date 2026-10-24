import CompressionLean.Terminal.Data
import CompressionLean.Terminal.Outputs
import CompressionLean.Terminal.PiCCS
import CompressionLean.Terminal.PiRLC
import CompressionLean.Terminal.Rows
import CompressionLean.Terminal.Soundness

/-!
`Terminal/` owns the terminal circuit package of F′ and its soundness edge:
the package rows (`Data`, `Rows`), the PiCCS and PiRLC checks that they make
(`PiCCS`, `PiRLC`), the output columns (`Outputs`), and the parent claim that
holds when every row holds (`Soundness`).

It does not own the program that reads the package (`Circuit/Terminal.lean`).
-/
