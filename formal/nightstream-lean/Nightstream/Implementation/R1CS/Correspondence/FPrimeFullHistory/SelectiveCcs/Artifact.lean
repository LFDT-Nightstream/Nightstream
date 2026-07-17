import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.Artifact.Schema
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.Artifact.Interpreter
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.Artifact.Decoder
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.Artifact.RowAction
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.Artifact.Row
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.Artifact.SelectorCoverage
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.Artifact.PayloadRefinement

/-!
Stable handwritten surface for compact selective-CCS artifact data.

| Child | Ownership | Emits constraints? | Assurance tier |
|---|---|---|---|
| `Schema` | raw CSC arrays, dimensions, thirteen compact matrices, and fail-closed validity | no | model-level artifact schema |
| `Interpreter` | direct CSC/seeded/geometric expansion to a typed relation with fixed polynomial | no | model-level artifact schema |
| `Decoder` | canonical wire decoding, exact variant handling, and production-validity gate | no | model-level implementation correspondence |
| `RowAction` | exact finite row action, paper-row bridge, and polynomial row-shape reductions | no | model-level implementation correspondence |
| `SelectorCoverage` | run-compressed selector support, ledger/gate-class reconciliation, and exact polynomial syntax | no | model-level implementation correspondence |
| `PayloadRefinement` | fixed-point dimensions, all named decoded matrices, and independent polynomial attachment | no | model-level implementation correspondence |

There is intentionally no generated full production value yet. Importing this
facade does not establish that production Rust emitted any value or that a
decoded matrix implements the independent F' semantics.
-/
