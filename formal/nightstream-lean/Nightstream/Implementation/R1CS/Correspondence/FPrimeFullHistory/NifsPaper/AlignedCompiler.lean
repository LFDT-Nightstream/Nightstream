import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.NifsPaper.AlignedCompiler.ColumnMap
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.NifsPaper.AlignedCompiler.AssignmentMap
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.NifsPaper.AlignedCompiler.MatrixMap
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.NifsPaper.AlignedCompiler.CommitmentShape

/-!
Stable public surface for the formally specified aligned F' compiler.

| Child | Mathematical ownership | Emits constraints? | Rust owner |
|---|---|---|---|
| `ColumnMap` | old/public/padding/private column placement and 54-lane coordinates | no | not connected yet |
| `AssignmentMap` | old scalar preservation and fixed-zero padding at every mapped coordinate | no | not connected yet |
| `MatrixMap` | exhaustive old-coefficient-or-zero row lowering at every aligned coordinate | no | not connected yet |
| `CommitmentShape` | exact packing/setup width, key-row shape, and key-reuse necessity | no | not connected yet |

This parent intentionally exports only proved compiler contracts. Sparse
matrix storage, assignment serialization, Ajtai setup artifacts, and emitted R1CS rows
remain fail-closed until their own children exist and are proved.
-/
