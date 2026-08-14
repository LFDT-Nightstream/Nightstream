import Nightstream.Implementation.R1CS.Correspondence.SelectiveCcs.Polynomial.Ports
import Nightstream.Implementation.R1CS.Correspondence.SelectiveCcs.Polynomial.Semantics
import Nightstream.Implementation.R1CS.Correspondence.SelectiveCcs.Polynomial.Components
import Nightstream.Implementation.R1CS.Correspondence.SelectiveCcs.Polynomial.Rows
import Nightstream.Implementation.R1CS.Correspondence.SelectiveCcs.Polynomial.PackedRows
import Nightstream.Implementation.R1CS.Correspondence.SelectiveCcs.Polynomial.Necessity

/-!
Stable model-level surface for the complete selective CCS gate polynomial.

| Child | Mathematical ownership | Emits constraints? | Assurance tier |
|---|---|---|---|
| `Ports` | exact thirteen matrix-image roles and indices | no | model-level |
| `Semantics` | exact 74 sparse terms and degree bound | no | model-level |
| `Components` | exact six-family decomposition of all 74 terms | no | model-level |
| `Rows` | exact active components for six emitted row shapes | no | model-level |
| `PackedRows` | pair and odd-tail centered-domain equivalence under the named nonresidue premise | no | model-level |
| `Necessity` | one omission counterexample per chosen term family | no | model-level |

No production row is Rust-conformant through this facade yet. That requires a
raw matrix/polynomial artifact and a separate refinement module.
-/
