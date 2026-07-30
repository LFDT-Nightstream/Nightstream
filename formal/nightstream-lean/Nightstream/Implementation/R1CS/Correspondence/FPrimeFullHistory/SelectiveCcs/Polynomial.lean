import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.Polynomial.Ports
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.Polynomial.Semantics
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.Polynomial.Components
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.Polynomial.Rows
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.Polynomial.Necessity

/-!
Stable model-level surface for the complete selective CCS gate polynomial.

| Child | Mathematical ownership | Emits constraints? | Assurance tier |
|---|---|---|---|
| `Ports` | exact thirteen matrix-image roles and indices | no | model-level |
| `Semantics` | exact 27 sparse terms and degree bound | no | model-level |
| `Components` | exact six-family decomposition of all 66 terms | no | model-level |
| `Rows` | exact active components for six emitted row shapes | no | model-level |
| `Necessity` | one omission counterexample per chosen term family | no | model-level |

No production row is Rust-conformant through this facade yet. That requires a
raw matrix/polynomial artifact and a separate refinement module.
-/
